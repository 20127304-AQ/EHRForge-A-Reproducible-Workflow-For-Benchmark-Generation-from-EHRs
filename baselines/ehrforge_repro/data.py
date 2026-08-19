"""Dataset, corpus, and retrieval loading utilities."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from .config import REQUIRED_DATASET_COLUMNS
from .io_utils import iter_jsonl


SampleKey = tuple[int, int]


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate the QA dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path)
    missing = REQUIRED_DATASET_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame["person_id"] = frame["person_id"].astype(int)
    frame["qa_index"] = frame["qa_index"].astype(int)
    frame["_row_index"] = range(len(frame))

    duplicated = frame.duplicated(subset=["person_id", "qa_index"], keep=False)
    if duplicated.any():
        duplicate_rows = frame.loc[duplicated, ["person_id", "qa_index"]].head(10)
        raise ValueError(
            "Dataset contains duplicate (person_id, qa_index) keys. "
            f"Examples: {duplicate_rows.to_dict('records')}"
        )
    return frame


def load_corpus(path: Path) -> dict[int, list[dict[str, Any]]]:
    """Load the patient timeline corpus from JSON Lines."""
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")

    corpus: dict[int, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                person_id = int(record["person_id"])
                visits = record["visits"]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid corpus record at {path}:{line_number}") from exc
            if not isinstance(visits, list):
                raise ValueError(f"Corpus visits must be a list at {path}:{line_number}")
            corpus[person_id] = visits
    return corpus


def parse_evidence_indices(value: Any) -> list[int]:
    """Parse gold evidence indices in complete patient-timeline coordinates."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []

    if isinstance(value, list):
        evidence = value
    else:
        try:
            evidence = json.loads(str(value))
        except json.JSONDecodeError:
            return []

    if not isinstance(evidence, list):
        return []

    indices: list[int] = []

    for item in evidence:
        if not isinstance(item, dict):
            continue

        raw_index = item.get("original_visit_index")

        if raw_index is None:
            raw_index = item.get("matched_original_visit_index")

        # Backward compatibility with older datasets.
        if raw_index is None:
            raw_index = item.get("visit_index")

        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue

        if index < 0:
            continue

        if index not in indices:
            indices.append(index)

    return indices


def group_dataset_by_patient(frame: pd.DataFrame) -> dict[int, list[dict[str, Any]]]:
    """Group dataset records by patient while preserving dataset order."""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in frame.to_dict("records"):
        grouped[int(record["person_id"])].append(record)
    return dict(grouped)


def load_retrieval_map(path: Path) -> dict[SampleKey, dict[str, Any]]:
    """Load combined retrieval results indexed by sample key."""
    if not path.exists():
        raise FileNotFoundError(f"Retrieval results not found: {path}")

    result: dict[SampleKey, dict[str, Any]] = {}
    for record in iter_jsonl(path):
        key = (int(record["person_id"]), int(record["qa_index"]))
        if key in result:
            raise ValueError(f"Duplicate retrieval result for sample {key}")
        result[key] = record
    return result


def dataset_key_set(frame: pd.DataFrame) -> set[SampleKey]:
    """Return all canonical sample keys in a dataset."""
    return {
        (int(person_id), int(qa_index))
        for person_id, qa_index in zip(frame["person_id"], frame["qa_index"])
    }