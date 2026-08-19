#!/usr/bin/env python3
"""Verify that every generated evidence snippet occurs in its source visit."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ehrforge.io_utils import load_jsonl_dataframe, load_jsonl_records


def normalize_text(text: object) -> str:
    """Normalize quotes and whitespace for exact evidence comparison."""
    normalized = str(text)
    normalized = normalized.replace("“", '"').replace("”", '"')
    normalized = normalized.replace("‘", "'").replace("’", "'")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip().lower()


def split_snippet_on_ellipsis(snippet: str, min_chars: int = 6) -> list[str]:
    """Split a shortened evidence snippet into exact-matchable segments."""
    parts = re.split(r"\s*(?:\.{3,}|…)\s*", str(snippet))
    normalized_parts = [normalize_text(part).strip('"') for part in parts]
    return [part for part in normalized_parts if len(part) >= min_chars]


def snippet_matches_original(
    snippet: str,
    original_text: str,
    min_part_match_ratio: float = 1.0,
) -> tuple[bool, str]:
    """Check direct containment or complete ellipsis-segment containment."""
    snippet_normalized = normalize_text(snippet).strip('"')
    original_normalized = normalize_text(original_text)

    if not snippet_normalized:
        return False, "empty_snippet"
    if snippet_normalized in original_normalized:
        return True, "exact_match"

    parts = split_snippet_on_ellipsis(snippet_normalized)
    if not parts:
        return False, "empty_or_too_short_after_split"

    matched_parts = [part for part in parts if part in original_normalized]
    ratio = len(matched_parts) / len(parts)
    if ratio >= min_part_match_ratio:
        return True, f"ellipsis_segment_match_{len(matched_parts)}_of_{len(parts)}"

    return False, f"partial_segment_match_{len(matched_parts)}_of_{len(parts)}"


def build_original_visit_lookup(
    original_dataframe: pd.DataFrame,
) -> dict[tuple[int, str], list[dict[str, Any]]]:
    """Index source visits by patient identifier and visit timestamp."""
    lookup: dict[tuple[int, str], list[dict[str, Any]]] = {}

    for row in original_dataframe.itertuples(index=False):
        person_id = int(row.person_id)
        visits = row.visits
        if not isinstance(visits, list):
            continue

        for original_visit_index, visit in enumerate(visits):
            if not isinstance(visit, dict):
                continue
            visit_datetime = str(visit.get("visit_datetime", "")).strip()
            text = str(visit.get("text", ""))
            lookup.setdefault((person_id, visit_datetime), []).append(
                {
                    "original_visit_index": original_visit_index,
                    "visit_datetime": visit_datetime,
                    "text": text,
                    "normalized_text": normalize_text(text),
                }
            )

    return lookup


def check_evidence_against_original(
    output_records: list[dict[str, Any]],
    original_dataframe: pd.DataFrame,
    min_part_match_ratio: float = 1.0,
) -> pd.DataFrame:
    """Produce one validation row for every generated evidence reference."""
    original_lookup = build_original_visit_lookup(original_dataframe)
    rows: list[dict[str, Any]] = []

    for record in output_records:
        person_id = int(record["person_id"])

        for qa_index, qa in enumerate(record.get("qas", [])):
            for evidence_index, evidence in enumerate(qa.get("evidence", [])):
                visit_datetime = str(evidence.get("visit_datetime", "")).strip()
                evidence_snippet = str(
                    evidence.get("evidence_snippet", "")
                ).strip()
                original_visit_index = evidence.get("original_visit_index")
                candidates = original_lookup.get((person_id, visit_datetime), [])

                matched = False
                match_type: str | None = None
                matched_original_visit_index: int | None = None

                # First, prefer the exact original visit index recorded at generation time.
                if original_visit_index is not None:
                    for candidate in candidates:
                        if (
                            candidate["original_visit_index"]
                            != original_visit_index
                        ):
                            continue

                        matched_original_visit_index = candidate[
                            "original_visit_index"
                        ]
                        matched, detail = snippet_matches_original(
                            snippet=evidence_snippet,
                            original_text=candidate["text"],
                            min_part_match_ratio=min_part_match_ratio,
                        )
                        if matched:
                            match_type = (
                                "matched_by_datetime_and_original_visit_index"
                                f"__{detail}"
                            )
                        else:
                            match_type = (
                                "datetime_and_index_found_but_snippet_not_found"
                                f"__{detail}"
                            )
                        break

                # If needed, try every source visit that shares the same timestamp.
                if not matched and candidates:
                    for candidate in candidates:
                        candidate_match, detail = snippet_matches_original(
                            snippet=evidence_snippet,
                            original_text=candidate["text"],
                            min_part_match_ratio=min_part_match_ratio,
                        )
                        if candidate_match:
                            matched = True
                            matched_original_visit_index = candidate[
                                "original_visit_index"
                            ]
                            match_type = f"matched_by_datetime_only__{detail}"
                            break

                    if not matched and match_type is None:
                        match_type = "datetime_found_but_snippet_not_found"

                if not candidates:
                    match_type = "no_original_visit_with_same_datetime"

                rows.append(
                    {
                        "person_id": person_id,
                        "qa_index": qa.get("qa_index", qa_index),
                        "chunk_id": qa.get("chunk_id"),
                        "timeline_sampling_strategy": qa.get(
                            "timeline_sampling_strategy"
                        ),
                        "question": qa.get("question", ""),
                        "evidence_index": evidence_index,
                        "visit_datetime": visit_datetime,
                        "output_visit_index": evidence.get("visit_index"),
                        "output_original_visit_index": original_visit_index,
                        "matched_original_visit_index": matched_original_visit_index,
                        "evidence_snippet": evidence_snippet,
                        "matched": matched,
                        "match_type": match_type,
                        "num_original_candidates_same_datetime": len(candidates),
                    }
                )

    return pd.DataFrame(rows)


def attach_evidence_match_to_excel(
    qa_excel_path: Path,
    check_dataframe: pd.DataFrame,
    output_excel_path: Path,
) -> pd.DataFrame:
    """Attach one all-evidence-matched flag to each flattened QA row."""
    qa_dataframe = pd.read_excel(qa_excel_path)

    if check_dataframe.empty:
        qa_dataframe["evidence_match"] = False
    else:
        match_summary = (
            check_dataframe.groupby(["person_id", "qa_index"])["matched"]
            .all()
            .reset_index()
            .rename(columns={"matched": "evidence_match"})
        )
        qa_dataframe = qa_dataframe.merge(
            match_summary,
            on=["person_id", "qa_index"],
            how="left",
        )
        qa_dataframe["evidence_match"] = (
            qa_dataframe["evidence_match"].fillna(False).astype(bool)
        )

    output_excel_path.parent.mkdir(parents=True, exist_ok=True)
    qa_dataframe.to_excel(output_excel_path, index=False)
    return qa_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check generated evidence snippets against the original patient timelines."
        )
    )
    parser.add_argument(
        "--original-jsonl",
        type=Path,
        default=Path("patient_sequences.jsonl"),
        help="Source patient timeline JSONL.",
    )
    parser.add_argument(
        "--generated-jsonl",
        type=Path,
        required=True,
        help="Generated temporal QA JSONL.",
    )
    parser.add_argument(
        "--check-output-excel",
        type=Path,
        default=Path("evidence_context_check_from_jsonl.xlsx"),
        help="Detailed evidence-level validation report.",
    )
    parser.add_argument(
        "--qa-flat-excel",
        type=Path,
        default=None,
        help="Optional flattened QA Excel file to receive an evidence_match column.",
    )
    parser.add_argument(
        "--qa-with-match-excel",
        type=Path,
        default=Path("qa_with_match_flag.xlsx"),
        help="Output path for the QA-level evidence-match file.",
    )
    parser.add_argument(
        "--min-part-match-ratio",
        type=float,
        default=1.0,
        help=(
            "Required proportion of ellipsis-separated snippet segments found in "
            "the source visit. The benchmark's strict setting is 1.0."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.min_part_match_ratio <= 1.0:
        raise ValueError("--min-part-match-ratio must be between 0 and 1.")

    original_dataframe = load_jsonl_dataframe(args.original_jsonl)
    required_columns = {"person_id", "visits"}
    missing_columns = required_columns - set(original_dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing source columns: {sorted(missing_columns)}")

    output_records = load_jsonl_records(args.generated_jsonl)
    check_dataframe = check_evidence_against_original(
        output_records=output_records,
        original_dataframe=original_dataframe,
        min_part_match_ratio=args.min_part_match_ratio,
    )

    args.check_output_excel.parent.mkdir(parents=True, exist_ok=True)
    check_dataframe.to_excel(args.check_output_excel, index=False)

    print("\n=== MATCHED COUNTS ===")
    if check_dataframe.empty:
        print("No evidence references were found.")
    else:
        print(check_dataframe["matched"].value_counts().to_string())
        print("\n=== MATCH TYPE COUNTS ===")
        print(check_dataframe["match_type"].value_counts().to_string())
    print(f"\nSaved evidence report to: {args.check_output_excel}")

    if args.qa_flat_excel is not None:
        merged = attach_evidence_match_to_excel(
            qa_excel_path=args.qa_flat_excel,
            check_dataframe=check_dataframe,
            output_excel_path=args.qa_with_match_excel,
        )
        print("\n=== QA MATCH SUMMARY ===")
        print(merged["evidence_match"].value_counts().to_string())
        print(f"\nSaved QA match file to: {args.qa_with_match_excel}")


if __name__ == "__main__":
    main()
