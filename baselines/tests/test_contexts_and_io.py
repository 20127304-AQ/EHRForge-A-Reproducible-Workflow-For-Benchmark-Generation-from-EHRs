import json
from pathlib import Path

from ehrforge_repro.contexts import build_context, select_visits
from ehrforge_repro.data import parse_evidence_indices
from ehrforge_repro.io_utils import read_jsonl_deduplicated


def test_parse_evidence_indices_deduplicates() -> None:
    value = json.dumps(
        [
            {"visit_index": 4},
            {"visit_index": 2},
            {"visit_index": 4},
        ]
    )
    assert parse_evidence_indices(value) == [4, 2]


def test_context_is_sorted_chronologically() -> None:
    visits = [
        {"visit_datetime": "2021-03-01", "text": "third"},
        {"visit_datetime": "2019-01-01", "text": "first"},
        {"visit_datetime": "2020-02-01", "text": "second"},
    ]
    selected = select_visits(visits, [0, 1, 2])
    assert [index for index, _ in selected] == [1, 2, 0]
    context = build_context(visits, [0, 1, 2])
    assert context.index("first") < context.index("second") < context.index("third")


def test_jsonl_deduplication_last_record_wins(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"person_id": 1, "qa_index": 2, "prediction": "old"}),
                "not-json",
                json.dumps({"person_id": 1, "qa_index": 2, "prediction": "new"}),
            ]
        ),
        encoding="utf-8",
    )
    records, invalid, duplicates = read_jsonl_deduplicated(path)
    assert invalid == 1
    assert duplicates == 1
    assert records[0]["prediction"] == "new"
