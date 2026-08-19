#!/usr/bin/env python3
"""Build chronological patient timelines from row-level clinical notes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ehrforge.text_processing import clean_clinical_text


REQUIRED_COLUMNS = {"person_id", "Visit_DateTime", "doc_text"}


def classify_visit_count(number_of_visits: int) -> str:
    """Assign a descriptive visit-count group for corpus reporting."""
    if number_of_visits <= 10:
        return "1-10 visits"
    if number_of_visits <= 100:
        return "11-100 visits"
    if number_of_visits <= 1000:
        return "101-1000 visits"
    return ">1000 visits"


def load_and_validate_csv(csv_path: Path) -> pd.DataFrame:
    """Read the source CSV and validate the fields required by the pipeline."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV was not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    dataframe = dataframe.copy()
    dataframe["Visit_DateTime"] = pd.to_datetime(
        dataframe["Visit_DateTime"], unit="ms", errors="coerce"
    )
    invalid_timestamps = int(dataframe["Visit_DateTime"].isna().sum())
    if invalid_timestamps:
        raise ValueError(
            f"Found {invalid_timestamps} invalid Visit_DateTime values after "
            "millisecond Unix conversion."
        )

    dataframe["clean_text"] = dataframe["doc_text"].apply(clean_clinical_text)
    dataframe = dataframe[dataframe["clean_text"] != ""].copy()
    dataframe = dataframe.sort_values(
        ["person_id", "Visit_DateTime"], kind="stable"
    ).reset_index(drop=True)
    return dataframe


def aggregate_visits(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Combine note fragments that share a patient and timestamp."""
    aggregations: dict[str, Any] = {"clean_text": " ".join}
    if "doc_id" in dataframe.columns:
        aggregations["doc_id"] = lambda values: [
            int(value) if float(value).is_integer() else float(value)
            for value in values.dropna()
        ]

    visit_dataframe = (
        dataframe.groupby(["person_id", "Visit_DateTime"], sort=True)
        .agg(aggregations)
        .reset_index()
    )
    return visit_dataframe.sort_values(
        ["person_id", "Visit_DateTime"], kind="stable"
    ).reset_index(drop=True)


def build_patient_records(visit_dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert visit-level rows into serializable patient timeline records."""
    records: list[dict[str, Any]] = []

    for person_id, patient_visits in visit_dataframe.groupby("person_id", sort=True):
        visits: list[dict[str, Any]] = []
        for row in patient_visits.itertuples(index=False):
            visit: dict[str, Any] = {
                "visit_datetime": row.Visit_DateTime.strftime("%Y-%m-%d %H:%M:%S"),
                "text": row.clean_text,
            }
            if hasattr(row, "doc_id"):
                visit["doc_ids"] = row.doc_id
            visits.append(visit)

        records.append({"person_id": int(person_id), "visits": visits})

    return records


def write_patient_sequences(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write one patient timeline per JSONL line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_summary(
    source_dataframe: pd.DataFrame,
    visit_dataframe: pd.DataFrame,
    records: list[dict[str, Any]],
) -> None:
    """Print row, visit, patient, and visit-group counts."""
    visit_counts = pd.Series(
        [len(record["visits"]) for record in records], dtype="int64"
    )
    group_counts = visit_counts.apply(classify_visit_count).value_counts()

    print("\n=== TIMELINE CONSTRUCTION SUMMARY ===")
    print(f"Non-empty source rows: {len(source_dataframe):,}")
    print(f"Aggregated visits: {len(visit_dataframe):,}")
    print(f"Patient timelines: {len(records):,}")
    print("\nVisit groups:")
    print(group_counts.to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean row-level clinical notes, aggregate same-time fragments, and "
            "write chronological patient timelines as JSONL."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("inter_df_personid.csv"),
        help="Path to the source clinical-note CSV file.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("patient_sequences.jsonl"),
        help="Destination for chronological patient timelines.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dataframe = load_and_validate_csv(args.input_csv)
    visit_dataframe = aggregate_visits(source_dataframe)
    records = build_patient_records(visit_dataframe)
    write_patient_sequences(records, args.output_jsonl)
    print_summary(source_dataframe, visit_dataframe, records)
    print(f"\nSaved patient timelines to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
