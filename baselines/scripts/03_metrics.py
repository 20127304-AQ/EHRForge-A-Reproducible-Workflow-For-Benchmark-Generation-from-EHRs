"""Compute retrieval and QA metrics for every configured experiment."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import modal

from ehrforge_repro.config import (
    APP_PREFIX,
    BERTSCORE_BATCH_SIZE,
    BERTSCORE_CHUNK_SIZE,
    BERTSCORE_LAYER,
    CONTEXT_TYPES,
    DATA_VOLUME_NAME,
    DATASET_PATH,
    HF_SECRET_NAME,
    METRICS_DIR,
    MODEL_IDS,
    QA_METRICS_PATH,
    READER_TYPES,
    RESULTS_VOLUME_NAME,
    RETRIEVAL_K_VALUES,
    RETRIEVAL_METRICS_PATH,
    RETRIEVAL_RESULTS_PATH,
    RETRIEVER_TYPES,
    STRATIFICATION_AXES,
    combination_key,
    prediction_metrics_path,
    prediction_path,
)
from ehrforge_repro.data import dataset_key_set, load_dataset, load_retrieval_map
from ehrforge_repro.io_utils import (
    read_json,
    read_jsonl_deduplicated,
    write_json_atomic,
    write_jsonl_atomic,
)
from ehrforge_repro.metrics import (
    aggregate_qa_records,
    aggregate_qa_stratified,
    evidence_recall_at_k,
    exact_coverage_at_k,
    exact_match,
    safe_mean,
    token_f1,
)


app = modal.App(f"{APP_PREFIX}-metrics")
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name("huggingface-cache-qa", create_if_missing=True)

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy<2", "pandas>=2.1,<3")
    .add_local_python_source("ehrforge_repro")
)

metrics_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11"
    )
    .entrypoint([])
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.2",
        "bert-score==0.3.13",
        "numpy<2",
        "pandas>=2.1,<3",
        "tqdm>=4.66,<5",
    )
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("ehrforge_repro")
)

volume_mounts = {
    "/data": data_volume,
    "/results": results_volume,
    "/root/.cache/huggingface": hf_cache,
}

RETRIEVAL_FIELDS = {
    "bm25": "bm25_top20",
    "medcpt": "medcpt_top20",
    "nvembed": "nvembed_top20",
    "hybrid": "hybrid_top20",
}


def _parse_combinations(value: str) -> list[tuple[str, str]]:
    all_combinations = [
        (context_type, reader)
        for context_type in CONTEXT_TYPES
        for reader in READER_TYPES
    ]
    if value.strip().lower() == "all":
        return all_combinations

    selected: list[tuple[str, str]] = []
    for raw_item in value.split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        if "+" not in item:
            raise ValueError(
                "Each combination must use the form context+reader, for example "
                "hybrid+qwen2.5-7b"
            )
        context_type, reader = item.split("+", 1)
        pair = (context_type, reader)
        if pair not in all_combinations:
            raise ValueError(f"Unsupported experiment combination: {item}")
        if pair not in selected:
            selected.append(pair)
    if not selected:
        raise ValueError("At least one experiment combination must be selected")
    return selected


def _retrieval_group_metrics(
    rows: list[dict[str, Any]], retriever: str
) -> dict[str, Any]:
    field = RETRIEVAL_FIELDS[retriever]
    metrics: dict[str, Any] = {"n_samples": len(rows)}
    for k in RETRIEVAL_K_VALUES:
        recall_values = [
            evidence_recall_at_k(row[field], row["oracle_indices"], k) for row in rows
        ]
        coverage_values = [
            exact_coverage_at_k(row[field], row["oracle_indices"], k)
            for row in rows
        ]
        metrics[f"recall_at_{k}"] = safe_mean(recall_values)
        metrics[f"exact_coverage_at_{k}"] = safe_mean(coverage_values)
        metrics[f"n_evaluable_at_{k}"] = sum(
            value is not None for value in recall_values
        )
    return metrics


@app.function(
    image=cpu_image,
    volumes=volume_mounts,
    cpu=4,
    memory=32768,
    timeout=4 * 60 * 60,
)
def compute_retrieval_metrics(overwrite: bool = False) -> dict[str, Any]:
    """Compute evidence Recall@K and Exact Coverage@K."""
    results_volume.reload()
    if RETRIEVAL_METRICS_PATH.exists() and not overwrite:
        return read_json(RETRIEVAL_METRICS_PATH)

    frame = load_dataset(DATASET_PATH)
    retrieval_map = load_retrieval_map(RETRIEVAL_RESULTS_PATH)
    dataset_records = {
        (int(row["person_id"]), int(row["qa_index"])): row
        for row in frame.to_dict("records")
    }

    joined: list[dict[str, Any]] = []
    for key, dataset_record in dataset_records.items():
        retrieval_record = retrieval_map.get(key)
        if retrieval_record is None:
            raise KeyError(f"Missing retrieval record for sample {key}")
        joined.append(
            {
                **retrieval_record,
                "difficulty": str(dataset_record.get("difficulty", "")),
                "visit_group": str(dataset_record.get("visit_group", "")),
                "reasoning_type": str(dataset_record.get("reasoning_type", "")),
            }
        )

    output: dict[str, Any] = {
        "metadata": {
            "definition": (
                "Recall@K is the macro-averaged proportion of gold evidence visits "
                "recovered in the top-K ranking. Exact Coverage@K is the fraction "
                "of samples for which every gold evidence visit is in the top-K."
            ),
            "k_values": list(RETRIEVAL_K_VALUES),
            "n_samples": len(joined),
        },
        "retrievers": {},
    }

    for retriever in RETRIEVER_TYPES:
        retriever_output: dict[str, Any] = {
            "overall": _retrieval_group_metrics(joined, retriever)
        }
        for axis in STRATIFICATION_AXES:
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in joined:
                grouped[str(row.get(axis) or "unknown")].append(row)
            retriever_output[f"by_{axis}"] = {
                group: _retrieval_group_metrics(group_rows, retriever)
                for group, group_rows in sorted(grouped.items())
            }
        output["retrievers"][retriever] = retriever_output

    write_json_atomic(RETRIEVAL_METRICS_PATH, output)
    results_volume.commit()
    return output


@app.function(
    image=metrics_image,
    volumes=volume_mounts,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    gpu="A100-80GB",
    memory=131072,
    timeout=24 * 60 * 60,
)
def compute_qa_metrics(
    combinations: str = "all",
    overwrite: bool = False,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Compute EM, Token F1, and Bio_ClinicalBERT BERTScore."""
    import torch
    from bert_score import BERTScorer

    results_volume.reload()
    frame = load_dataset(DATASET_PATH)
    expected_keys = dataset_key_set(frame)
    selected = _parse_combinations(combinations)

    if QA_METRICS_PATH.exists():
        output = read_json(QA_METRICS_PATH)
    else:
        output = {
            "metadata": {
                "bertscore_model": MODEL_IDS["bertscore"],
                "bertscore_layer": BERTSCORE_LAYER,
                "bertscore_rescale_with_baseline": False,
                "normalization": "SQuAD-style lowercase/articles/punctuation/whitespace",
                "n_expected_samples": len(frame),
            },
            "combinations": {},
        }

    scorer: BERTScorer | None = None
    started = time.time()

    for context_type, reader in selected:
        key = combination_key(context_type, reader)
        source_path = prediction_path(context_type, reader)
        target_path = prediction_metrics_path(context_type, reader)

        if (
            not overwrite
            and key in output.get("combinations", {})
            and target_path.exists()
        ):
            existing_metrics, _, _ = read_jsonl_deduplicated(target_path)
            if len(existing_metrics) == len(frame):
                print(f"Skipping completed metrics: {key}")
                continue

        predictions, invalid_lines, duplicate_lines = read_jsonl_deduplicated(
            source_path
        )
        prediction_keys = {
            (int(record["person_id"]), int(record["qa_index"]))
            for record in predictions
        }
        missing = expected_keys - prediction_keys
        extra = prediction_keys - expected_keys
        if (missing or extra) and not allow_incomplete:
            raise RuntimeError(
                f"Prediction coverage mismatch for {key}: "
                f"missing={len(missing)}, extra={len(extra)}"
            )
        if not predictions:
            raise RuntimeError(f"No predictions found for {key}: {source_path}")

        predictions.sort(key=lambda item: int(item.get("row_index", 0)))
        candidates = [str(record.get("prediction", "")) for record in predictions]
        references = [str(record.get("gold_answer", "")) for record in predictions]

        if scorer is None:
            scorer = BERTScorer(
                model_type=MODEL_IDS["bertscore"],
                num_layers=BERTSCORE_LAYER,
                batch_size=BERTSCORE_BATCH_SIZE,
                nthreads=4,
                all_layers=False,
                idf=False,
                device="cuda",
                rescale_with_baseline=False,
            )

        precision_values: list[float] = []
        recall_values: list[float] = []
        f1_values: list[float] = []
        for start in range(0, len(predictions), BERTSCORE_CHUNK_SIZE):
            stop = min(start + BERTSCORE_CHUNK_SIZE, len(predictions))
            precision, recall, f1 = scorer.score(
                candidates[start:stop], references[start:stop]
            )
            precision_values.extend(float(value) for value in precision.tolist())
            recall_values.extend(float(value) for value in recall.tolist())
            f1_values.extend(float(value) for value in f1.tolist())
            print(f"{key}: BERTScore {stop}/{len(predictions)}")

        enriched: list[dict[str, Any]] = []
        for record, bert_precision, bert_recall, bert_f1 in zip(
            predictions,
            precision_values,
            recall_values,
            f1_values,
        ):
            prediction = str(record.get("prediction", ""))
            reference = str(record.get("gold_answer", ""))
            enriched.append(
                {
                    **record,
                    "exact_match": exact_match(prediction, reference),
                    "token_f1": token_f1(prediction, reference),
                    "bertscore_precision": bert_precision,
                    "bertscore_recall": bert_recall,
                    "bertscore_f1": bert_f1,
                }
            )

        write_jsonl_atomic(target_path, enriched)
        combination_output = {
            "overall": aggregate_qa_records(enriched),
            **aggregate_qa_stratified(enriched, STRATIFICATION_AXES),
            "metadata": {
                "context_type": context_type,
                "reader": reader,
                "source_path": str(source_path),
                "instance_metrics_path": str(target_path),
                "invalid_prediction_lines_removed": invalid_lines,
                "duplicate_prediction_lines_removed": duplicate_lines,
                "complete": prediction_keys == expected_keys,
            },
        }
        output.setdefault("combinations", {})[key] = combination_output
        output["metadata"]["bertscore_hash"] = getattr(scorer, "hash", None)
        output["metadata"]["elapsed_seconds"] = time.time() - started
        write_json_atomic(QA_METRICS_PATH, output)
        results_volume.commit()
        torch.cuda.empty_cache()

    return output


@app.local_entrypoint()
def main(
    mode: str = "all",
    combinations: str = "all",
    overwrite: bool = False,
    allow_incomplete: bool = False,
) -> None:
    """Run retrieval metrics, QA metrics, or both."""
    mode = mode.strip().lower()
    if mode not in {"all", "retrieval", "qa"}:
        raise ValueError("mode must be all, retrieval, or qa")

    if mode in {"all", "retrieval"}:
        retrieval_output = compute_retrieval_metrics.remote(overwrite=overwrite)
        print(
            "Retrieval metrics completed for "
            f"{len(retrieval_output['retrievers'])} retrievers."
        )
    if mode in {"all", "qa"}:
        qa_output = compute_qa_metrics.remote(
            combinations=combinations,
            overwrite=overwrite,
            allow_incomplete=allow_incomplete,
        )
        print(
            "QA metrics available for "
            f"{len(qa_output.get('combinations', {}))} combinations."
        )

    print(
        "Metrics are stored in Modal volume "
        f"{RESULTS_VOLUME_NAME} under {METRICS_DIR}."
    )
