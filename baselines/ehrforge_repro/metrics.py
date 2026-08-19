"""Metric implementations used by the retrieval and QA evaluations."""

from __future__ import annotations

import re
import string
from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np


ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize_answer(value: Any) -> str:
    """Apply SQuAD-style answer normalization."""
    text = str(value).lower()
    text = ARTICLES.sub(" ", text)
    text = "".join(character for character in text if character not in string.punctuation)
    return " ".join(text.split())


def exact_match(prediction: Any, reference: Any) -> float:
    """Return normalized exact match."""
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: Any, reference: Any) -> float:
    """Return token-level F1 after normalized bag-of-words matching."""
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()

    if not reference_tokens:
        return 1.0 if not prediction_tokens else 0.0
    if not prediction_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def evidence_recall_at_k(
    retrieved: Iterable[int], gold: Iterable[int], k: int
) -> float | None:
    """Return the proportion of gold evidence visits recovered in the top K.

    Samples without gold evidence return None and should be excluded from the
    macro average.
    """
    gold_set = {int(value) for value in gold}
    if not gold_set:
        return None
    retrieved_set = {int(value) for value in list(retrieved)[:k]}
    return len(retrieved_set & gold_set) / len(gold_set)


def exact_coverage_at_k(
    retrieved: Iterable[int], gold: Iterable[int], k: int
) -> float | None:
    """Return one when all gold evidence visits occur in the top K."""
    gold_set = {int(value) for value in gold}
    if not gold_set:
        return None
    retrieved_set = {int(value) for value in list(retrieved)[:k]}
    return float(gold_set.issubset(retrieved_set))


def safe_mean(values: Iterable[float | None]) -> float | None:
    """Return a mean after excluding None values."""
    filtered = [float(value) for value in values if value is not None]
    return float(np.mean(filtered)) if filtered else None


def aggregate_qa_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate QA metrics from instance-level metric records."""
    if not records:
        return {
            "exact_match": None,
            "token_f1": None,
            "bertscore_precision": None,
            "bertscore_recall": None,
            "bertscore_f1": None,
            "n_samples": 0,
        }

    return {
        "exact_match": float(np.mean([float(item["exact_match"]) for item in records])),
        "token_f1": float(np.mean([float(item["token_f1"]) for item in records])),
        "bertscore_precision": float(
            np.mean([float(item["bertscore_precision"]) for item in records])
        ),
        "bertscore_recall": float(
            np.mean([float(item["bertscore_recall"]) for item in records])
        ),
        "bertscore_f1": float(
            np.mean([float(item["bertscore_f1"]) for item in records])
        ),
        "n_samples": len(records),
    }


def aggregate_qa_stratified(
    records: list[dict[str, Any]], axes: Iterable[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    """Aggregate QA metrics over each requested stratification axis."""
    output: dict[str, dict[str, dict[str, Any]]] = {}
    for axis in axes:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[str(record.get(axis) or "unknown")].append(record)
        output[f"by_{axis}"] = {
            group: aggregate_qa_records(group_records)
            for group, group_records in sorted(grouped.items())
        }
    return output
