"""File and serialization helpers used by all experiment stages."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


Record = dict[str, Any]


def ensure_parent(path: Path) -> None:
    """Create the parent directory for a path if it does not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute a SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    """Read a JSON document."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON atomically to avoid partially written result files."""
    ensure_parent(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def iter_jsonl(path: Path, ignore_invalid: bool = False) -> Iterator[Record]:
    """Yield JSON objects from a JSON Lines file."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                if ignore_invalid:
                    continue
                raise ValueError(f"Invalid JSON at {path}:{line_number}")
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            yield value


def append_jsonl(path: Path, records: Iterable[Record]) -> int:
    """Append records to a JSON Lines file and return the number written."""
    ensure_parent(path)
    written = 0
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
        handle.flush()
        os.fsync(handle.fileno())
    return written


def write_jsonl_atomic(path: Path, records: Iterable[Record]) -> int:
    """Write records to a JSON Lines file atomically."""
    ensure_parent(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    written = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return written


def sample_key(record: Record) -> tuple[int, int]:
    """Return the canonical sample identifier."""
    return int(record["person_id"]), int(record["qa_index"])


def read_jsonl_deduplicated(
    path: Path,
    key_fields: tuple[str, ...] = ("person_id", "qa_index"),
    ignore_invalid: bool = True,
) -> tuple[list[Record], int, int]:
    """Read a JSONL file and keep the last valid record for each key.

    Returns the deduplicated records, the number of invalid lines, and the
    number of duplicate records removed.
    """
    if not path.exists():
        return [], 0, 0

    by_key: dict[tuple[Any, ...], Record] = {}
    invalid = 0
    valid = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                key = tuple(record[field] for field in key_fields)
            except (json.JSONDecodeError, KeyError, TypeError):
                if ignore_invalid:
                    invalid += 1
                    continue
                raise
            valid += 1
            by_key[key] = record

    duplicates = valid - len(by_key)
    records = list(by_key.values())
    records.sort(key=lambda item: tuple(item[field] for field in key_fields))
    return records, invalid, duplicates


def merge_jsonl_records(
    existing: Iterable[Record],
    new_records: Iterable[Record],
    key_fields: tuple[str, ...] = ("person_id", "qa_index"),
) -> list[Record]:
    """Merge record collections with last-write-wins semantics."""
    merged: dict[tuple[Any, ...], Record] = {}
    for record in existing:
        merged[tuple(record[field] for field in key_fields)] = record
    for record in new_records:
        merged[tuple(record[field] for field in key_fields)] = record
    output = list(merged.values())
    output.sort(key=lambda item: tuple(item[field] for field in key_fields))
    return output
