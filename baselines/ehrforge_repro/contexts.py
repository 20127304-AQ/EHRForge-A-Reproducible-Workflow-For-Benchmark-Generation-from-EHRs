"""Context selection and prompt construction for reader experiments."""

from __future__ import annotations

from typing import Any, Sequence

from .config import READER_TOP_K, SYSTEM_PROMPT
from .data import parse_evidence_indices


RETRIEVAL_FIELDS = {
    "bm25": "bm25_top20",
    "medcpt": "medcpt_top20",
    "nvembed": "nvembed_top20",
    "hybrid": "hybrid_top20",
    "oracle": "oracle_indices",
}


def select_visit_indices(
    dataset_record: dict[str, Any],
    retrieval_record: dict[str, Any],
    context_type: str,
    reader_top_k: int = READER_TOP_K,
) -> list[int]:
    """Return the visit indices used by a reader for one sample."""
    if context_type == "oracle":
        indices = retrieval_record.get("oracle_indices")
        if indices is None:
            indices = parse_evidence_indices(dataset_record.get("evidence"))
        limit = None
    else:
        field = RETRIEVAL_FIELDS.get(context_type)
        if field is None:
            raise ValueError(f"Unsupported context type: {context_type}")
        indices = retrieval_record.get(field, [])
        limit = reader_top_k

    selected: list[int] = []
    for value in indices or []:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if index not in selected:
            selected.append(index)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def select_visits(
    visits: Sequence[dict[str, Any]], indices: Sequence[int]
) -> list[tuple[int, dict[str, Any]]]:
    """Select valid visits and sort them chronologically."""
    selected: list[tuple[int, dict[str, Any]]] = []
    for index in indices:
        if 0 <= index < len(visits):
            selected.append((index, visits[index]))
    selected.sort(
        key=lambda item: (
            str(item[1].get("visit_datetime", "")),
            int(item[0]),
        )
    )
    return selected


def format_visit(rank: int, visit: dict[str, Any]) -> str:
    """Format one visit in the representation passed to readers."""
    raw_datetime = str(visit.get("visit_datetime", ""))
    display_date = raw_datetime[:10] if raw_datetime else "unknown"
    text = str(visit.get("text", ""))
    return f"[Visit {rank} | {display_date}]\n{text}"


def build_context(
    visits: Sequence[dict[str, Any]], indices: Sequence[int]
) -> str:
    """Build a chronological context string from selected visits."""
    selected = select_visits(visits, indices)
    return "\n\n".join(
        format_visit(rank, visit)
        for rank, (_, visit) in enumerate(selected, start=1)
    )


def build_context_with_token_budget(
    visits: Sequence[dict[str, Any]],
    indices: Sequence[int],
    tokenizer: Any,
    token_budget: int,
) -> str:
    """Build a chronological context and truncate only when it exceeds a budget.

    The available text-token budget is distributed across selected visits. This
    keeps every selected visit represented while preserving chronological order.
    """
    selected = select_visits(visits, indices)
    if not selected:
        return ""

    full_context = "\n\n".join(
        format_visit(rank, visit)
        for rank, (_, visit) in enumerate(selected, start=1)
    )
    full_ids = tokenizer.encode(full_context, add_special_tokens=False)
    if len(full_ids) <= token_budget:
        return full_context

    headings: list[str] = []
    text_token_ids: list[list[int]] = []
    for rank, (_, visit) in enumerate(selected, start=1):
        raw_datetime = str(visit.get("visit_datetime", ""))
        display_date = raw_datetime[:10] if raw_datetime else "unknown"
        headings.append(f"[Visit {rank} | {display_date}]\n")
        text_token_ids.append(
            tokenizer.encode(str(visit.get("text", "")), add_special_tokens=False)
        )

    heading_cost = sum(
        len(tokenizer.encode(heading, add_special_tokens=False))
        for heading in headings
    )
    separator_cost = max(0, len(selected) - 1) * len(
        tokenizer.encode("\n\n", add_special_tokens=False)
    )
    available = max(token_budget - heading_cost - separator_cost, len(selected))

    allocations = [0] * len(selected)
    remaining = available
    active = set(range(len(selected)))
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        progressed = False
        for position in list(active):
            capacity = len(text_token_ids[position]) - allocations[position]
            if capacity <= 0:
                active.remove(position)
                continue
            granted = min(share, capacity, remaining)
            allocations[position] += granted
            remaining -= granted
            progressed = progressed or granted > 0
            if allocations[position] >= len(text_token_ids[position]):
                active.remove(position)
            if remaining == 0:
                break
        if not progressed:
            break

    parts: list[str] = []
    for heading, token_ids, allocation in zip(
        headings, text_token_ids, allocations
    ):
        text = tokenizer.decode(
            token_ids[:allocation],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        parts.append(heading + text)
    return "\n\n".join(parts)


def build_qwen_messages(question: str, context: str) -> list[dict[str, str]]:
    """Build the chat messages used by both Qwen readers."""
    user_message = f"Patient Visit Notes:\n{context}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
