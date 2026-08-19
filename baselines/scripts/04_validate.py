"""Validate experiment completeness, model identity, and result consistency."""

from __future__ import annotations

from typing import Any

import modal

from ehrforge_repro.config import (
    APP_PREFIX,
    CONTEXT_TYPES,
    CORPUS_PATH,
    DATA_VOLUME_NAME,
    DATASET_PATH,
    MANIFEST_DIR,
    MODEL_IDS,
    QA_METRICS_PATH,
    READER_TYPES,
    RESULTS_VOLUME_NAME,
    RETRIEVAL_METRICS_PATH,
    RETRIEVAL_RESULTS_PATH,
    VALIDATION_DIR,
    combination_key,
    prediction_metrics_path,
    prediction_path,
)
from ehrforge_repro.data import dataset_key_set, load_corpus, load_dataset, load_retrieval_map
from ehrforge_repro.io_utils import read_json, read_jsonl_deduplicated, write_json_atomic


app = modal.App(f"{APP_PREFIX}-validate")
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy<2", "pandas>=2.1,<3")
    .add_local_python_source("ehrforge_repro")
)


def _check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    detail: str,
    severity: str = "error",
) -> None:
    checks.append(
        {
            "name": name,
            "passed": bool(passed),
            "severity": severity,
            "detail": detail,
        }
    )


@app.function(
    image=image,
    volumes={"/data": data_volume, "/results": results_volume},
    cpu=4,
    memory=32768,
    timeout=2 * 60 * 60,
)
def validate(strict: bool = True) -> dict[str, Any]:
    """Validate all configured retrieval and reader experiments."""
    results_volume.reload()
    frame = load_dataset(DATASET_PATH)
    corpus = load_corpus(CORPUS_PATH)
    expected_keys = dataset_key_set(frame)
    expected_count = len(frame)
    checks: list[dict[str, Any]] = []

    _check(checks, "dataset_nonempty", expected_count > 0, f"n_samples={expected_count}")
    _check(
        checks,
        "retrieval_results_present",
        RETRIEVAL_RESULTS_PATH.exists(),
        str(RETRIEVAL_RESULTS_PATH),
    )

    if RETRIEVAL_RESULTS_PATH.exists():
        retrieval_map = load_retrieval_map(RETRIEVAL_RESULTS_PATH)
        retrieval_keys = set(retrieval_map)
        _check(
            checks,
            "retrieval_key_coverage",
            retrieval_keys == expected_keys,
            f"expected={len(expected_keys)}, actual={len(retrieval_keys)}, "
            f"missing={len(expected_keys - retrieval_keys)}, "
            f"extra={len(retrieval_keys - expected_keys)}",
        )

        invalid_rankings = 0
        for key, record in retrieval_map.items():
            visit_count = len(corpus.get(key[0], []))
            for field in (
                "bm25_top20",
                "medcpt_top20",
                "nvembed_top20",
                "hybrid_top20",
                "oracle_indices",
            ):
                values = record.get(field)
                if not isinstance(values, list):
                    invalid_rankings += 1
                    continue
                if any(
                    not isinstance(value, int) or value < 0 or value >= visit_count
                    for value in values
                ):
                    invalid_rankings += 1
        _check(
            checks,
            "retrieval_indices_valid",
            invalid_rankings == 0,
            f"invalid_rankings={invalid_rankings}",
        )

    nv_manifest_path = MANIFEST_DIR / "nvembed.json"
    if nv_manifest_path.exists():
        nv_manifest = read_json(nv_manifest_path)
        _check(
            checks,
            "nvembed_model_identity",
            nv_manifest.get("model") == MODEL_IDS["nvembed"],
            f"expected={MODEL_IDS['nvembed']}, actual={nv_manifest.get('model')}",
        )
        _check(
            checks,
            "nvembed_no_fallback",
            nv_manifest.get("fallback_used") is False,
            f"fallback_used={nv_manifest.get('fallback_used')}",
        )
    else:
        _check(checks, "nvembed_manifest_present", False, str(nv_manifest_path))

    expected_combinations = [
        (context_type, reader)
        for context_type in CONTEXT_TYPES
        for reader in READER_TYPES
    ]
    prediction_summary: dict[str, Any] = {}
    for context_type, reader in expected_combinations:
        key = combination_key(context_type, reader)
        path = prediction_path(context_type, reader)
        records, invalid, duplicates = read_jsonl_deduplicated(path)
        keys = {
            (int(record["person_id"]), int(record["qa_index"])) for record in records
        }
        model_ids = {str(record.get("model_id", "")) for record in records}
        expected_model = MODEL_IDS[reader]
        complete = keys == expected_keys and len(records) == expected_count
        correct_model = model_ids == {expected_model}
        _check(
            checks,
            f"prediction_coverage:{key}",
            complete,
            f"expected={expected_count}, actual={len(records)}, "
            f"missing={len(expected_keys - keys)}, extra={len(keys - expected_keys)}, "
            f"invalid_lines={invalid}, duplicate_lines={duplicates}",
        )
        _check(
            checks,
            f"prediction_model_identity:{key}",
            correct_model,
            f"expected={expected_model}, actual={sorted(model_ids)}",
        )
        prediction_summary[key] = {
            "path": str(path),
            "n_records": len(records),
            "complete": complete,
            "model_ids": sorted(model_ids),
        }

    for reader in READER_TYPES:
        manifest_path = MANIFEST_DIR / f"reader_{reader}.json"
        if not manifest_path.exists():
            _check(checks, f"reader_manifest_present:{reader}", False, str(manifest_path))
            continue
        manifest = read_json(manifest_path)
        _check(
            checks,
            f"reader_no_fallback:{reader}",
            manifest.get("fallback_used") is False,
            f"fallback_used={manifest.get('fallback_used')}",
        )
        _check(
            checks,
            f"reader_model_identity:{reader}",
            manifest.get("model") == MODEL_IDS[reader],
            f"expected={MODEL_IDS[reader]}, actual={manifest.get('model')}",
        )
        if reader == "clinicalbert" and manifest.get("missing_keys"):
            _check(
                checks,
                "clinicalbert_qa_head_initialization",
                True,
                "The selected ClinicalBERT checkpoint reports missing QA-head weights. "
                "Confirm the exact extractive-QA checkpoint required by the experiment.",
                severity="warning",
            )

    _check(
        checks,
        "retrieval_metrics_present",
        RETRIEVAL_METRICS_PATH.exists(),
        str(RETRIEVAL_METRICS_PATH),
    )
    _check(
        checks,
        "qa_metrics_present",
        QA_METRICS_PATH.exists(),
        str(QA_METRICS_PATH),
    )

    if QA_METRICS_PATH.exists():
        qa_metrics = read_json(QA_METRICS_PATH)
        available = set(qa_metrics.get("combinations", {}))
        expected_metric_keys = {
            combination_key(context_type, reader)
            for context_type, reader in expected_combinations
        }
        _check(
            checks,
            "qa_metrics_combination_coverage",
            available == expected_metric_keys,
            f"expected={len(expected_metric_keys)}, actual={len(available)}, "
            f"missing={sorted(expected_metric_keys - available)}",
        )

        for context_type, reader in expected_combinations:
            key = combination_key(context_type, reader)
            instance_path = prediction_metrics_path(context_type, reader)
            records, invalid, duplicates = read_jsonl_deduplicated(instance_path)
            valid_metric_fields = all(
                all(
                    field in record and record[field] is not None
                    for field in (
                        "exact_match",
                        "token_f1",
                        "bertscore_precision",
                        "bertscore_recall",
                        "bertscore_f1",
                    )
                )
                for record in records
            )
            _check(
                checks,
                f"instance_metrics:{key}",
                len(records) == expected_count and valid_metric_fields,
                f"expected={expected_count}, actual={len(records)}, "
                f"invalid_lines={invalid}, duplicate_lines={duplicates}, "
                f"fields_complete={valid_metric_fields}",
            )

    errors = [
        item for item in checks if not item["passed"] and item["severity"] == "error"
    ]
    warnings = [item for item in checks if item["severity"] == "warning"]
    report = {
        "passed": len(errors) == 0,
        "n_samples": expected_count,
        "expected_combinations": len(expected_combinations),
        "n_checks": len(checks),
        "n_errors": len(errors),
        "n_warnings": len(warnings),
        "checks": checks,
        "predictions": prediction_summary,
    }
    report_path = VALIDATION_DIR / "validation_report.json"
    write_json_atomic(report_path, report)
    results_volume.commit()

    if strict and errors:
        failed_names = [item["name"] for item in errors[:20]]
        raise RuntimeError(
            f"Validation failed with {len(errors)} errors. "
            f"First failures: {failed_names}. Report: {report_path}"
        )
    return report


@app.local_entrypoint()
def main(strict: bool = True) -> None:
    """Run the completeness and identity audit."""
    report = validate.remote(strict=strict)
    print(
        f"passed={report['passed']} checks={report['n_checks']} "
        f"errors={report['n_errors']} warnings={report['n_warnings']}"
    )
    print(
        "Validation report is stored in Modal volume "
        f"{RESULTS_VOLUME_NAME} under {VALIDATION_DIR}."
    )
