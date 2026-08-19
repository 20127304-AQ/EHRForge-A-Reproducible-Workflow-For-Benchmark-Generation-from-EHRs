"""Shared file I/O helpers for JSONL records and batch checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def load_jsonl_dataframe(path: str | Path) -> pd.DataFrame:
    """Load a JSONL file into a pandas DataFrame."""
    return pd.DataFrame(load_jsonl_records(path))


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Load every non-empty JSONL line as a dictionary."""
    file_path = Path(path)
    if not file_path.exists():
        return []

    records: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {file_path} at line {line_number}: {exc}"
                ) from exc
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: str | Path) -> None:
    """Write records to a JSONL file, replacing any existing file."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(record: dict[str, Any], path: str | Path) -> None:
    """Append one record to a JSONL file and flush it to disk."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_processed_ids(path: str | Path) -> set[int]:
    """Load processed patient identifiers from a checkpoint file."""
    file_path = Path(path)
    if not file_path.exists():
        return set()

    processed: set[int] = set()
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = line.strip()
            if not value:
                continue
            try:
                processed.add(int(value))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid patient identifier in {file_path} at line {line_number}: "
                    f"{value!r}"
                ) from exc
    return processed


def mark_processed(path: str | Path, person_id: int) -> None:
    """Append a processed patient identifier to the checkpoint file."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{int(person_id)}\n")
        handle.flush()
        os.fsync(handle.fileno())
