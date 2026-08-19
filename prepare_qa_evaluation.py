#!/usr/bin/env python3
"""Convert evidence-aligned QA rows from Excel into evaluation JSONL."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {
    "person_id",
    "question",
    "answer",
    "reasoning_type",
    "difficulty",
    "is_temporal",
    "temporal_signal",
    "evidence",
    "evidence_match",
}


def parse_evidence(value: Any) -> list[dict[str, Any]]:
    """Parse an evidence cell serialized as JSON or a Python literal."""
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    if not isinstance(value, str):
        raise ValueError(f"Unsupported evidence value: {type(value).__name__}")

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = ast.literal_eval(value)

    if not isinstance(parsed, list):
        raise ValueError("The evidence field must decode to a list.")
    return parsed


def build_evaluation_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """Keep fully aligned QA rows and map them to the evaluation schema."""
    filtered = dataframe[dataframe["evidence_match"].fillna(False).astype(bool)].copy()
    records: list[dict[str, Any]] = []

    for row in filtered.itertuples(index=False):
        records.append(
            {
                "person_id": int(row.person_id),
                "question": str(row.question),
                "gold_answer": str(row.answer),
                "reasoning_type": str(row.reasoning_type),
                "difficulty": str(row.difficulty),
                "is_temporal": bool(row.is_temporal),
                "temporal_signal": str(row.temporal_signal),
                "gold_evidence": parse_evidence(row.evidence),
            }
        )

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create evaluation JSONL from fully evidence-aligned QA rows."
    )
    parser.add_argument(
        "--input-excel",
        type=Path,
        default=Path("qa_eval.xlsx"),
        help="Excel file containing an evidence_match column.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("qa_eval.jsonl"),
        help="Destination evaluation JSONL file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataframe = pd.read_excel(args.input_excel)
    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    records = build_evaluation_records(dataframe)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved {len(records)} QA pairs to {args.output_jsonl}")


if __name__ == "__main__":
    main()
