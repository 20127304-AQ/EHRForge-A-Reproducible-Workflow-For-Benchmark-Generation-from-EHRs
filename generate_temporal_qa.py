#!/usr/bin/env python3
"""Generate schema-constrained temporal QA candidates from patient timelines."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from openai import OpenAI

from ehrforge.config import GenerationConfig
from ehrforge.io_utils import (
    append_jsonl,
    load_jsonl_dataframe,
    load_jsonl_records,
    load_processed_ids,
    mark_processed,
)


SYSTEM_PROMPT = """
You are creating a high-quality temporal QA dataset from longitudinal clinical notes.

Your task:
- Read one patient's visit timeline in chronological order.
- Generate as many high-quality QA pairs as the timeline strongly supports, up to the maximum number requested by the user.
- Do not force quantity. Stop when additional QA pairs would become repetitive, weak, trivial, or unsupported.
- Prefer questions that require genuine temporal reasoning across visits, timestamps, or temporally distinct events.
- If the timeline only supports one strong question, generate only one.
- Use only the provided timeline.
- Do not invent facts.
- Do not ask demographic-only, trivial extraction, or single-span lookup questions unless no stronger temporal question is possible.
- Return valid JSON that exactly follows the schema.

A strong QA should involve one or more of:
- before_after
- progression
- first_last_occurrence
- recurrence_count
- trend
- comparison_across_visits
- treatment_response

Definitions:
- easy: answer is directly supported and time-aware, usually requiring simple temporal linkage
- medium: requires linking at least two temporally distinct pieces of evidence
- hard: requires integrating multiple visits, multiple timestamps, or subtle longitudinal reasoning

Core requirements:
- At least one QA must be temporal.
- Prefer all QA pairs to be temporal if supported by the timeline.
- Each QA must be evidence-grounded and answerable from the cited evidence alone.
- Cite the minimal sufficient evidence needed to support the answer.
- evidence_snippets must be short, tightly grounded excerpts from the cited visits.
- Keep answers concise, specific, and clinically faithful to the notes.
- Use explicit visit timestamps when they help establish sequence, interval, recurrence, first occurrence, last occurrence, or change over time.
- When possible, make the temporal anchor explicit in the question, such as before discharge, after medication, during the admission, between two dates, or across two named visits.
- Do not include any patient personal identifiers in QA pairs.
- Use neutral wording such as "the patient" instead of names or personal identifiers.

Temporal quality rules:
- Prefer questions that require at least two temporally distinct evidence items.
- Do not label a question as temporal if it can be answered from a single isolated statement without any time comparison or sequencing.
- Do not ask compound questions with multiple main targets unless both parts are necessary to answer a single temporal question.
- Prefer one clear question with one main answer target.

Privacy and de-identification rules:
- Do not ask questions about patient names, addresses, phone numbers, emails, IDs, medical record numbers, account numbers, insurance numbers, dates of birth, exact ages, occupations, relatives' names, or other personally identifying details.
- Do not include patient identifiers in answers or evidence snippets.
- If personal identifiers appear in the notes, ignore them unless they are clinically necessary; even then, paraphrase without exposing the identifier.
- Visit timestamps may be used only as temporal anchors when needed for reasoning.

Faithfulness rules:
- Answers must be directly supported by the cited evidence.
- Do not include clinical interpretation beyond explicitly stated evidence unless the interpretation is directly quoted or clearly stated in the note.
- Do not infer causality from sequence alone.
- Do not claim change, progression, improvement, worsening, comparison, trend, recurrence, or treatment response unless the cited evidence directly supports that claim.
- Do not describe a finding as new, worse, improved, resolved, recurrent, or stable unless that wording is explicitly supported by the notes or by direct comparison of the cited evidence.
- If the evidence is ambiguous, choose safer and narrower wording.

Reasoning type guidance:
- before_after: answer depends on what happened before versus after an event or timepoint
- progression: answer depends on documented evolution within or across visits
- first_last_occurrence: answer depends on identifying the first or last documented instance
- recurrence_count: answer depends on repeated events over time
- trend: answer depends on directional change across measurements or repeated assessments
- comparison_across_visits: answer depends on comparing findings or status across visits
- treatment_response: answer depends on documented response after an intervention or treatment

Output quality rules:
- Questions should sound natural and clinically meaningful.
- Answers should be short and precise rather than narrative when possible.
- Avoid copying long phrases from the notes into the question.
- Avoid overly broad questions like "what happened over time" if a more precise temporal question is possible.

Output only JSON.
""".strip()


BASE_QA_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "qas": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "answer": {"type": "string"},
                    "reasoning_type": {
                        "type": "string",
                        "enum": [
                            "before_after",
                            "progression",
                            "first_last_occurrence",
                            "recurrence_count",
                            "trend",
                            "comparison_across_visits",
                            "treatment_response",
                            "other",
                        ],
                    },
                    "difficulty": {
                        "type": "string",
                        "enum": ["easy", "medium", "hard"],
                    },
                    "evidence": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "visit_index": {"type": "integer"},
                                "evidence_snippet": {"type": "string"},
                            },
                            "required": ["visit_index", "evidence_snippet"],
                            "additionalProperties": False,
                        },
                    },
                    "is_temporal": {"type": "boolean"},
                    "temporal_signal": {"type": "string"},
                },
                "required": [
                    "question",
                    "answer",
                    "reasoning_type",
                    "difficulty",
                    "evidence",
                    "is_temporal",
                    "temporal_signal",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["qas"],
    "additionalProperties": False,
}

TEMPORAL_REASONING_TYPES = {
    "before_after",
    "progression",
    "first_last_occurrence",
    "recurrence_count",
    "trend",
    "comparison_across_visits",
    "treatment_response",
}

MULTI_VISIT_REASONING_TYPES = {
    "before_after",
    "progression",
    "recurrence_count",
    "trend",
    "comparison_across_visits",
    "treatment_response",
}

CHANGE_WORDS = {
    "change",
    "changed",
    "improve",
    "improved",
    "improvement",
    "worsen",
    "worsened",
    "progress",
    "progressed",
    "trend",
    "over time",
    "later",
    "earlier",
    "first",
    "last",
    "subsequent",
    "after",
    "before",
    "compare",
    "comparison",
    "evolve",
    "evolution",
    "ultimately",
    "response",
    "recurrence",
}

OVERINFERENCE_WORDS = {
    "suggests",
    "implies",
    "likely",
    "overall",
    "probably",
    "supports",
    "does not support",
    "consistent with",
    "indicates",
}


def make_qa_schema(max_qas: int) -> dict[str, Any]:
    """Create a strict JSON schema with a call-specific QA limit."""
    schema = deepcopy(BASE_QA_SCHEMA)
    schema["properties"]["qas"]["maxItems"] = max_qas
    return schema


def trim_text(text: object, max_chars: int) -> str:
    """Collapse whitespace and cap the number of prompt characters per visit."""
    normalized = " ".join(str(text).split())
    if len(normalized) > max_chars:
        return normalized[:max_chars] + " ..."
    return normalized


def parse_datetime_safe(value: str) -> pd.Timestamp:
    """Parse a timestamp without interrupting the full generation run."""
    try:
        return pd.to_datetime(value)
    except Exception:
        return pd.NaT


def assign_visit_group(number_of_visits: int) -> str | None:
    """Apply the benchmark's patient-eligibility and visit-group rules."""
    if 2 <= number_of_visits <= 10:
        return "2-10 visits"
    if 11 <= number_of_visits <= 100:
        return "11-100 visits"
    if 101 <= number_of_visits <= 1000:
        return "101-1000 visits"
    return None


def build_timeline_from_selected(
    selected_visits: list[dict[str, Any]],
    original_indices: list[int],
    sampling_strategy: str,
    visit_group: str,
    max_qas_for_call: int,
    config: GenerationConfig,
) -> list[dict[str, Any]]:
    """Build one prompt-ready timeline and retain original visit provenance."""
    timeline: list[dict[str, Any]] = []

    for original_index, visit in zip(original_indices, selected_visits):
        if not isinstance(visit, dict):
            continue

        visit_datetime = str(visit.get("visit_datetime", "")).strip()
        text = trim_text(visit.get("text", ""), config.max_chars_per_visit)
        if not text:
            continue

        timeline.append(
            {
                "visit_index": len(timeline),
                "original_visit_index": original_index,
                "visit_datetime": visit_datetime,
                "parsed_datetime": parse_datetime_safe(visit_datetime),
                "text": text,
                "sampling_strategy": sampling_strategy,
                "visit_group": visit_group,
                "max_qas_for_call": max_qas_for_call,
            }
        )

    return timeline


def build_timelines_for_patient(
    visits: list[dict[str, Any]], config: GenerationConfig
) -> list[list[dict[str, Any]]]:
    """Apply full, U-shaped, global, and overlapping local sampling rules."""
    number_of_visits = len(visits)

    if 2 <= number_of_visits <= 10:
        return [
            build_timeline_from_selected(
                selected_visits=visits,
                original_indices=list(range(number_of_visits)),
                sampling_strategy="full_timeline_2_10",
                visit_group="2-10 visits",
                max_qas_for_call=config.max_qas_small_group,
                config=config,
            )
        ]

    if 11 <= number_of_visits <= 100:
        if number_of_visits <= config.max_visits_medium_group:
            selected = visits
            original_indices = list(range(number_of_visits))
            strategy = "full_timeline_11_100"
        else:
            selected = (
                visits[: config.early_visits_medium_group]
                + visits[-config.late_visits_medium_group :]
            )
            original_indices = list(range(config.early_visits_medium_group)) + list(
                range(
                    number_of_visits - config.late_visits_medium_group,
                    number_of_visits,
                )
            )
            strategy = (
                f"u_shape_first_{config.early_visits_medium_group}"
                f"_last_{config.late_visits_medium_group}"
            )

        return [
            build_timeline_from_selected(
                selected_visits=selected,
                original_indices=original_indices,
                sampling_strategy=strategy,
                visit_group="11-100 visits",
                max_qas_for_call=config.max_qas_medium_group,
                config=config,
            )
        ]

    if 101 <= number_of_visits <= 1000:
        timelines: list[list[dict[str, Any]]] = []

        global_selected = (
            visits[: config.global_early_visits_long_group]
            + visits[-config.global_late_visits_long_group :]
        )
        global_original_indices = list(
            range(config.global_early_visits_long_group)
        ) + list(
            range(
                number_of_visits - config.global_late_visits_long_group,
                number_of_visits,
            )
        )
        global_timeline = build_timeline_from_selected(
            selected_visits=global_selected,
            original_indices=global_original_indices,
            sampling_strategy=(
                f"global_u_shape_first_{config.global_early_visits_long_group}"
                f"_last_{config.global_late_visits_long_group}"
            ),
            visit_group="101-1000 visits",
            max_qas_for_call=config.max_qas_global_chunk,
            config=config,
        )
        if global_timeline:
            timelines.append(global_timeline)

        step = config.chunk_size_long_group - config.chunk_overlap_long_group
        for start in range(0, number_of_visits, step):
            end = min(start + config.chunk_size_long_group, number_of_visits)
            timeline = build_timeline_from_selected(
                selected_visits=visits[start:end],
                original_indices=list(range(start, end)),
                sampling_strategy=(
                    f"local_chunk_{start}_{end}"
                    f"_size_{config.chunk_size_long_group}"
                    f"_overlap_{config.chunk_overlap_long_group}"
                ),
                visit_group="101-1000 visits",
                max_qas_for_call=config.max_qas_per_chunk,
                config=config,
            )
            if timeline:
                timelines.append(timeline)
            if end == number_of_visits:
                break

        return timelines

    return []


def format_timeline_for_prompt(timeline: list[dict[str, Any]]) -> str:
    """Format selected visits for the model prompt."""
    return "\n".join(
        f"[Visit {visit['visit_index']}] [Time: {visit['visit_datetime']}]\n"
        f"{visit['text']}\n"
        for visit in timeline
    )


def build_user_prompt(
    person_id: int, timeline: list[dict[str, Any]], max_qas: int
) -> str:
    """Create the per-call prompt while keeping patient identifiers out of QA text."""
    visit_group = timeline[0].get("visit_group", "") if timeline else ""
    return f"""
Patient ID: {person_id}
Visit group: {visit_group}

{format_timeline_for_prompt(timeline)}

Generate as many high-quality, non-redundant QA pairs as this timeline strongly supports, up to a maximum of {max_qas} QA pairs.
Prioritize quality over quantity.
Return only JSON.
""".strip()


def normalize_result(
    person_id: int,
    timeline: list[dict[str, Any]],
    model_output: dict[str, Any],
) -> dict[str, Any]:
    """Normalize model output and attach source-visit metadata."""
    maximum_index = len(timeline) - 1
    normalized_qas: list[dict[str, Any]] = []

    for qa in model_output.get("qas", []):
        evidence_items: list[dict[str, Any]] = []
        seen_evidence: set[tuple[int, str]] = set()

        for evidence in qa.get("evidence", []):
            visit_index = evidence.get("visit_index")
            snippet = str(evidence.get("evidence_snippet", "")).strip()
            if (
                not isinstance(visit_index, int)
                or visit_index < 0
                or visit_index > maximum_index
                or not snippet
            ):
                continue

            key = (visit_index, snippet)
            if key in seen_evidence:
                continue
            seen_evidence.add(key)

            evidence_items.append(
                {
                    "visit_index": visit_index,
                    "original_visit_index": timeline[visit_index].get(
                        "original_visit_index", visit_index
                    ),
                    "visit_datetime": timeline[visit_index]["visit_datetime"],
                    "evidence_snippet": snippet,
                }
            )

        question = str(qa.get("question", "")).strip()
        answer = str(qa.get("answer", "")).strip()
        if not question or not answer or not evidence_items:
            continue

        normalized_qas.append(
            {
                "question": question,
                "answer": answer,
                "reasoning_type": qa.get("reasoning_type", "other"),
                "difficulty": qa.get("difficulty", "medium"),
                "evidence": evidence_items,
                "is_temporal": bool(qa.get("is_temporal", False)),
                "temporal_signal": str(qa.get("temporal_signal", "")).strip(),
            }
        )

    return {
        "person_id": int(person_id),
        "num_visits_in_prompt": len(timeline),
        "timeline_sampling_strategy": (
            timeline[0].get("sampling_strategy") if timeline else None
        ),
        "visit_group": timeline[0].get("visit_group") if timeline else None,
        "max_qas_for_call": (
            timeline[0].get("max_qas_for_call") if timeline else None
        ),
        "qas": normalized_qas,
    }


def contains_any(text: str, keywords: set[str]) -> bool:
    """Return whether lowercase text contains any configured keyword."""
    lowercase_text = str(text).lower()
    return any(keyword in lowercase_text for keyword in keywords)


def get_evidence_indices(qa: dict[str, Any]) -> list[int]:
    """Extract valid visit indices from one QA record."""
    return [
        evidence["visit_index"]
        for evidence in qa.get("evidence", [])
        if "visit_index" in evidence
    ]


def get_evidence_text(qa: dict[str, Any]) -> str:
    """Concatenate evidence snippets for heuristic validation."""
    return " ".join(
        evidence.get("evidence_snippet", "")
        for evidence in qa.get("evidence", [])
    )


def evidence_datetimes(
    qa: dict[str, Any], timeline: list[dict[str, Any]]
) -> list[pd.Timestamp]:
    """Map evidence indices to parsed visit timestamps."""
    datetimes: list[pd.Timestamp] = []
    for visit_index in get_evidence_indices(qa):
        if 0 <= visit_index < len(timeline):
            datetimes.append(timeline[visit_index]["parsed_datetime"])
    return datetimes


def has_monotonic_time_order(
    qa: dict[str, Any], timeline: list[dict[str, Any]]
) -> bool:
    """Check that cited evidence is listed in chronological order."""
    datetimes = [
        timestamp
        for timestamp in evidence_datetimes(qa, timeline)
        if pd.notna(timestamp)
    ]
    if len(datetimes) <= 1:
        return True
    return datetimes == sorted(datetimes)


def answer_looks_over_inferred(answer: str, evidence_text: str) -> bool:
    """Flag inference language that is absent from the cited evidence."""
    if contains_any(answer, OVERINFERENCE_WORDS) and not contains_any(
        evidence_text, OVERINFERENCE_WORDS
    ):
        return True
    return False


def requires_multi_evidence(
    question: str,
    answer: str,
    reasoning_type: str,
    is_temporal: bool,
    config: GenerationConfig,
) -> bool:
    """Determine whether a QA item requires at least two evidence references."""
    if (
        config.require_multi_visit_for_temporal
        and reasoning_type in MULTI_VISIT_REASONING_TYPES
    ):
        return True
    if is_temporal and contains_any(f"{question} {answer}", CHANGE_WORDS):
        return True
    return False


def validate_single_qa(
    qa: dict[str, Any],
    timeline: list[dict[str, Any]],
    config: GenerationConfig,
) -> tuple[bool, list[str]]:
    """Apply deterministic structural and temporal checks to one QA item."""
    reasons: list[str] = []
    question = str(qa.get("question", "")).strip()
    answer = str(qa.get("answer", "")).strip()
    reasoning_type = qa.get("reasoning_type", "other")
    is_temporal = bool(qa.get("is_temporal", False))
    evidence = qa.get("evidence", [])
    evidence_text = get_evidence_text(qa)

    if len(question.split()) < 4:
        reasons.append("question_too_short")
    if not answer:
        reasons.append("answer_empty")
    if not evidence:
        reasons.append("no_evidence")
        return False, reasons
    if config.strict_temporal_filter and not is_temporal:
        reasons.append("non_temporal_removed")
    if is_temporal and reasoning_type not in TEMPORAL_REASONING_TYPES | {"other"}:
        reasons.append("temporal_reasoning_type_unexpected")
    if requires_multi_evidence(
        question, answer, reasoning_type, is_temporal, config
    ) and len(evidence) < 2:
        reasons.append("insufficient_evidence_for_temporal_reasoning")
    if not has_monotonic_time_order(qa, timeline):
        reasons.append("evidence_not_in_time_order")
    if answer_looks_over_inferred(answer, evidence_text):
        reasons.append("possible_over_inference")

    return len(reasons) == 0, reasons


def basic_quality_filter(
    example: dict[str, Any],
    timeline: list[dict[str, Any]],
    config: GenerationConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Keep valid QA items and record deterministic rejection reasons."""
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for qa in example["qas"]:
        is_valid, reasons = validate_single_qa(qa, timeline, config)
        if is_valid:
            kept.append(qa)
        else:
            rejected.append(
                {"question": qa.get("question", ""), "reasons": reasons}
            )

    example["qas"] = kept
    return example, rejected


def normalize_for_dedupe(text: str) -> str:
    """Normalize free text for question-answer duplicate detection."""
    normalized = str(text).lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^\w\s]", "", normalized)
    return normalized.strip()


def dedupe_qas(qas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate QA pairs while preserving the first occurrence."""
    seen: set[tuple[str, str]] = set()
    deduplicated: list[dict[str, Any]] = []

    for qa in qas:
        key = (
            normalize_for_dedupe(qa.get("question", "")),
            normalize_for_dedupe(qa.get("answer", ""))[:120],
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(qa)

    return deduplicated


def call_openai_generate(
    client: Any,
    person_id: int,
    timeline: list[dict[str, Any]],
    max_qas: int,
    config: GenerationConfig,
) -> dict[str, Any]:
    """Call the Responses API with strict JSON-schema output constraints."""
    response = client.responses.create(
        model=config.model_name,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(person_id, timeline, max_qas),
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "temporal_qa_dataset",
                "schema": make_qa_schema(max_qas),
                "strict": True,
            }
        },
    )
    return json.loads(response.output_text)


def generate_for_patient(
    client: Any,
    person_id: int,
    visits: list[dict[str, Any]],
    config: GenerationConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Generate, normalize, filter, and deduplicate QA candidates for one patient."""
    timelines = build_timelines_for_patient(visits, config)
    if not timelines:
        raise ValueError("No valid timeline or chunks remained after preprocessing.")

    all_qas: list[dict[str, Any]] = []
    all_rejections: list[dict[str, Any]] = []
    sampling_strategies: list[str] = []

    for chunk_id, timeline in enumerate(timelines):
        maximum_qas = int(
            timeline[0].get("max_qas_for_call", config.max_qas_per_chunk)
        )
        strategy = timeline[0].get("sampling_strategy") if timeline else None
        visit_group = timeline[0].get("visit_group") if timeline else None
        last_error: Exception | None = None

        for attempt in range(config.max_retries):
            try:
                raw_output = call_openai_generate(
                    client=client,
                    person_id=person_id,
                    timeline=timeline,
                    max_qas=maximum_qas,
                    config=config,
                )
                result = normalize_result(person_id, timeline, raw_output)
                result, rejected = basic_quality_filter(result, timeline, config)

                if strategy:
                    sampling_strategies.append(strategy)

                for item in rejected:
                    item.update(
                        {
                            "chunk_id": chunk_id,
                            "visit_group": visit_group,
                            "timeline_sampling_strategy": strategy,
                        }
                    )
                    all_rejections.append(item)

                for qa in result["qas"]:
                    qa.update(
                        {
                            "chunk_id": chunk_id,
                            "visit_group": visit_group,
                            "timeline_sampling_strategy": strategy,
                            "max_qas_for_call": maximum_qas,
                        }
                    )
                    all_qas.append(qa)
                break
            except Exception as exc:
                last_error = exc
                if attempt < config.max_retries - 1:
                    print(
                        f"Retry patient={person_id}, chunk={chunk_id}, "
                        f"attempt={attempt + 1}: {exc}"
                    )
                    time.sleep(config.retry_sleep)
                else:
                    all_rejections.append(
                        {
                            "question": "",
                            "reasons": [f"chunk_generation_failed: {last_error}"],
                            "chunk_id": chunk_id,
                            "visit_group": visit_group,
                            "timeline_sampling_strategy": strategy,
                        }
                    )

        time.sleep(config.sleep_between_calls)

    all_qas = dedupe_qas(all_qas)
    if not all_qas:
        raise ValueError(
            "No valid QA remained after generation, filtering, and deduplication."
        )

    return (
        {
            "person_id": int(person_id),
            "num_total_visits": len(visits),
            "num_chunks": len(timelines),
            "timeline_sampling_strategy": sorted(set(sampling_strategies)),
            "qas": all_qas,
        },
        all_rejections,
    )


def flatten_examples_for_excel(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten nested patient QA records into one row per QA pair."""
    rows: list[dict[str, Any]] = []

    for record in records:
        for qa_index, qa in enumerate(record.get("qas", [])):
            rows.append(
                {
                    "person_id": record["person_id"],
                    "num_total_visits": record.get("num_total_visits"),
                    "num_chunks": record.get("num_chunks"),
                    "qa_index": qa_index,
                    "chunk_id": qa.get("chunk_id"),
                    "visit_group": qa.get("visit_group"),
                    "timeline_sampling_strategy": qa.get(
                        "timeline_sampling_strategy"
                    ),
                    "max_qas_for_call": qa.get("max_qas_for_call"),
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "reasoning_type": qa.get("reasoning_type"),
                    "difficulty": qa.get("difficulty"),
                    "is_temporal": qa.get("is_temporal"),
                    "temporal_signal": qa.get("temporal_signal"),
                    "evidence": json.dumps(
                        qa.get("evidence", []), ensure_ascii=False
                    ),
                }
            )

    return pd.DataFrame(rows)


def prepare_patient_dataframe(
    dataframe: pd.DataFrame, config: GenerationConfig
) -> pd.DataFrame:
    """Apply eligibility rules and an optional visit-group restriction."""
    prepared = dataframe.copy()
    prepared["num_visits"] = prepared["visits"].apply(
        lambda visits: len(visits) if isinstance(visits, list) else 0
    )
    prepared["visit_group"] = prepared["num_visits"].apply(assign_visit_group)
    prepared = prepared[prepared["visit_group"].notna()].reset_index(drop=True)

    if config.only_visit_group is not None:
        prepared = prepared[
            prepared["visit_group"] == config.only_visit_group
        ].reset_index(drop=True)

    return prepared


def select_next_batch(
    dataframe: pd.DataFrame, config: GenerationConfig
) -> pd.DataFrame:
    """Select the next unprocessed patient batch from the checkpoint state."""
    processed_ids = load_processed_ids(config.checkpoint_path)
    remaining = dataframe[
        ~dataframe["person_id"].astype(int).isin(processed_ids)
    ].copy()
    return remaining.head(config.batch_size).reset_index(drop=True)


def save_batch_excel(config: GenerationConfig) -> pd.DataFrame:
    """Rebuild the flattened Excel file from the current run's JSONL output."""
    records = load_jsonl_records(config.output_jsonl)
    flat_dataframe = flatten_examples_for_excel(records)
    flat_dataframe.to_excel(config.output_flat_excel, index=False)
    return flat_dataframe


def prepare_output_files(config: GenerationConfig, overwrite: bool) -> None:
    """Create the output directory and optionally clear current-run artifacts."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    output_files = [
        config.output_jsonl,
        config.output_flat_excel,
        config.output_run_errors_jsonl,
        config.output_qa_rejections_jsonl,
        config.output_stats_json,
    ]
    if overwrite:
        for path in output_files:
            if path.exists():
                path.unlink()
    elif config.output_jsonl.exists():
        print(
            f"Appending to existing run output: {config.output_jsonl}. "
            "Use --overwrite-output to start this run name again."
        )

    if config.reset_checkpoint and config.checkpoint_path.exists():
        config.checkpoint_path.unlink()
        print(f"Deleted checkpoint: {config.checkpoint_path}")


def run_generation(
    client: Any, config: GenerationConfig, overwrite_output: bool
) -> None:
    """Execute one resumable patient batch and save all stage-level artifacts."""
    config.validate()
    prepare_output_files(config, overwrite_output)

    dataframe = load_jsonl_dataframe(config.input_jsonl)
    required_columns = {"person_id", "visits"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    dataframe = prepare_patient_dataframe(dataframe, config)
    batch_dataframe = select_next_batch(dataframe, config)

    print("\n=== ALL VALID PATIENTS BY VISIT GROUP ===")
    print(dataframe["visit_group"].value_counts().to_string())
    print("\n=== THIS BATCH ===")
    print(f"Run name: {config.run_name}")
    print(f"Batch size: {config.batch_size}")
    print(f"Patients selected: {len(batch_dataframe)}")

    if batch_dataframe.empty:
        print("No remaining patients to process.")
        return

    print(batch_dataframe["visit_group"].value_counts().to_string())
    print(
        batch_dataframe[["person_id", "num_visits", "visit_group"]]
        .head(20)
        .to_string(index=False)
    )

    stats: dict[str, Any] = {
        "run_name": config.run_name,
        "batch_size": config.batch_size,
        "patients_selected": len(batch_dataframe),
        "patients_success": 0,
        "patients_failed": 0,
        "total_qas": 0,
        "total_rejected_qas": 0,
        "model_name": config.model_name,
        "random_seed": config.random_seed,
        "output_jsonl": str(config.output_jsonl),
        "output_flat_excel": str(config.output_flat_excel),
        "output_run_errors_jsonl": str(config.output_run_errors_jsonl),
        "output_qa_rejections_jsonl": str(config.output_qa_rejections_jsonl),
        "checkpoint_path": str(config.checkpoint_path),
    }

    for batch_index, (_, row) in enumerate(
        tqdm(
            batch_dataframe.iterrows(),
            total=len(batch_dataframe),
            desc="Patients",
        ),
        start=1,
    ):
        person_id = int(row["person_id"])
        print(
            f"\n[{batch_index}/{len(batch_dataframe)}] Processing "
            f"person_id={person_id}, visits={row['num_visits']}, "
            f"group={row['visit_group']}"
        )

        try:
            result, rejected = generate_for_patient(
                client=client,
                person_id=person_id,
                visits=row["visits"],
                config=config,
            )
            append_jsonl(result, config.output_jsonl)

            for item in rejected:
                append_jsonl(
                    {
                        "person_id": person_id,
                        "question": item.get("question", ""),
                        "reasons": item.get("reasons", []),
                        "chunk_id": item.get("chunk_id"),
                        "visit_group": item.get("visit_group"),
                        "timeline_sampling_strategy": item.get(
                            "timeline_sampling_strategy"
                        ),
                    },
                    config.output_qa_rejections_jsonl,
                )

            mark_processed(config.checkpoint_path, person_id)
            stats["patients_success"] += 1
            stats["total_qas"] += len(result["qas"])
            stats["total_rejected_qas"] += len(rejected)
            tqdm.write(
                f"[{person_id}] QA={len(result['qas'])}, rejected={len(rejected)}"
            )
        except Exception as exc:
            append_jsonl(
                {
                    "person_id": person_id,
                    "num_visits": int(row["num_visits"]),
                    "visit_group": row["visit_group"],
                    "error": str(exc),
                },
                config.output_run_errors_jsonl,
            )
            if config.mark_failed_as_processed:
                mark_processed(config.checkpoint_path, person_id)
            stats["patients_failed"] += 1
            print(f"Failed: {exc}")

        config.output_stats_json.write_text(
            json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        time.sleep(config.sleep_between_calls)

    flat_dataframe = save_batch_excel(config)
    config.output_stats_json.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== FINAL STATS ===")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nSaved JSONL: {config.output_jsonl}")
    print(f"Saved Excel: {config.output_flat_excel} ({len(flat_dataframe)} QA rows)")
    print(f"Saved checkpoint: {config.checkpoint_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate evidence-grounded temporal QA candidates in resumable batches."
        )
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("patient_sequences.jsonl"),
        help="Chronological patient timelines created by build_patient_sequences.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for run outputs.",
    )
    parser.add_argument("--run-name", default="batch_001")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path("processed_patient_ids.txt"),
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument(
        "--only-visit-group",
        choices=["2-10 visits", "11-100 visits", "101-1000 visits"],
        default=None,
    )
    parser.add_argument("--strict-temporal-filter", action="store_true")
    parser.add_argument("--reset-checkpoint", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Do not checkpoint failed patients, so a later batch can retry them.",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--sleep-between-calls", type=float, default=0.5)
    parser.add_argument("--max-chars-per-visit", type=int, default=2500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Export it in the environment before running."
        )

    config = GenerationConfig(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        run_name=args.run_name,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        model_name=args.model,
        only_visit_group=args.only_visit_group,
        strict_temporal_filter=args.strict_temporal_filter,
        reset_checkpoint=args.reset_checkpoint,
        mark_failed_as_processed=not args.retry_failed,
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        sleep_between_calls=args.sleep_between_calls,
        max_chars_per_visit=args.max_chars_per_visit,
    )
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The openai package is required for generation. Install dependencies "
            "with: pip install -r requirements.txt"
        ) from exc

    run_generation(OpenAI(), config, args.overwrite_output)


if __name__ == "__main__":
    main()
