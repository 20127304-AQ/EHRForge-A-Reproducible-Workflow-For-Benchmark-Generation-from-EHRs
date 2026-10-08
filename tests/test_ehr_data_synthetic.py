"""test_ehr_data.py-style tests of evaluation/common/ehr_data.py on the SYNTHETIC fixture (no restricted data).
The original 44-test suite (shared/canonical/test_ehr_data.py) runs on the real data and is distributed only to
credentialed users (it contains real keys and visit datetimes). Run: python -m pytest tests -q
"""
import copy
import hashlib
import json
import os
import sys
import importlib

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import synthetic_fixture as SF  # noqa: E402


@pytest.fixture(scope="module")
def E(tmp_path_factory):
    root = SF.build(str(tmp_path_factory.mktemp("ws")))
    os.environ["EHRFORGE_ROOT"] = root
    sys.path.insert(0, os.path.join(root, "shared", "canonical"))
    sys.modules.pop("ehr_data", None)
    E = importlib.import_module("ehr_data")
    assert E.ROOT == root
    # point the frozen-input constants at the synthetic files (the real values are asserted on real data only)
    E.DATASET_SHA256 = hashlib.sha256(open(E.DATASET_PATH, "rb").read()).hexdigest()
    E.CORPUS_SHA256 = hashlib.sha256(open(E.CORPUS_PATH, "rb").read()).hexdigest()
    E.CORPUS_SIZE = os.path.getsize(E.CORPUS_PATH)
    E.N_ROWS, E.N_DATASET_PATIENTS, E.N_EVIDENCE = len(SF.ROWS), 3, sum(len(r[3]) for r in SF.ROWS)
    E.N_CORPUS_PATIENTS, E.N_CORPUS_VISITS = len(SF.CORPUS), sum(len(v) for v in SF.CORPUS.values())
    for f in (E.load_dataset, E.load_corpus, E._dataset_by_key):
        f.cache_clear()
    yield E
    sys.path.remove(os.path.join(root, "shared", "canonical"))
    os.environ.pop("EHRFORGE_ROOT", None)


@pytest.fixture(scope="module")
def df(E):
    return E.load_dataset()


@pytest.fixture(scope="module")
def corpus(E):
    return E.load_corpus()


def test_shipped_ehr_data_is_byte_identical_to_canonical():
    rel = os.path.dirname(HERE)
    rows = {r.split(",")[0]: r.strip().split(",")[2] for r in open(os.path.join(rel, "evaluation/protocol/CANONICAL_SHA256.csv"))}
    for fn in ("evaluation/common/ehr_data.py", "evaluation/common/sampled_window.py"):
        assert hashlib.sha256(open(os.path.join(rel, fn), "rb").read()).hexdigest() == rows[fn]


def test_protocol_sidecars_match():
    rel = os.path.dirname(HERE)
    for v in ("", "_v1.1", "_v1.2", "_v1.3", "_v1.4"):
        p = os.path.join(rel, "evaluation/protocol", f"protocol{v}.yaml")
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == open(p + ".sha256").read().split()[0]


def test_load_protocol_v11_sha_checked(E):
    p = E.load_protocol("1.1")
    assert str(p["protocol_version"]) == "1.1"
    assert p["evidence_mapping"]["field"] == "original_visit_index" and p["evidence_mapping"]["never_use"] == "visit_index"


def test_dataset_sha_mismatch_raises(E, monkeypatch):
    monkeypatch.setattr(E, "DATASET_SHA256", "0" * 64)
    E.load_dataset.cache_clear()
    with pytest.raises(E.ShaMismatchError):
        E.load_dataset()
    E.load_dataset.cache_clear()


def test_counts(df, corpus):
    assert len(df) == 5 and sorted(corpus) == [101, 102, 103]


def test_gold_uses_original_visit_index_not_visit_index(E, df, corpus):
    row = E.get_row(102, 0)
    assert E.gold_visits(row, corpus) == [27, 30]                       # full-timeline indices
    chunk = sorted({e["visit_index"] for e in json.loads(row["evidence"])})
    assert chunk == [2, 5] and chunk != E.gold_visits(row, corpus)       # the pre-fix lookup would point elsewhere


def test_visit_index_lookup_would_hit_wrong_datetime(E, df, corpus):
    """Regression guard for the pre-fix bug: corpus[visit_index] has a different datetime than the evidence."""
    row = E.get_row(102, 0)
    for e in json.loads(row["evidence"]):
        assert corpus[102][e["original_visit_index"]]["visit_datetime"] == e["visit_datetime"]
        assert corpus[102][e["visit_index"]]["visit_datetime"] != e["visit_datetime"]


def test_missing_original_visit_index_raises_no_fallback(E, corpus):
    """The guard: evidence without original_visit_index must raise (never fall back to visit_index)."""
    row = dict(copy.deepcopy(E.get_row(101, 0)))
    ev = json.loads(row["evidence"])
    for e in ev:
        e.pop("original_visit_index")
    row["evidence_items"] = ev
    with pytest.raises(KeyError):
        E.gold_visits(row, corpus)


def test_dedup_and_sort(E, corpus):
    assert E.gold_visits(E.get_row(101, 1), corpus) == [2]
    row = dict(E.get_row(102, 1)); ev = list(reversed(json.loads(row["evidence"]))) * 2
    row["evidence_items"] = ev
    assert E.gold_visits(row, corpus) == [3, 38]


@pytest.mark.parametrize("bad", [-1, 6, 99])
def test_out_of_range_raises(E, corpus, bad):
    row = dict(E.get_row(101, 0)); ev = json.loads(row["evidence"]); ev[0]["original_visit_index"] = bad
    row["evidence_items"] = ev
    with pytest.raises(E.VisitIndexOutOfRange):
        E.gold_visits(row, corpus)


def test_datetime_mismatch_raises(E, corpus):
    row = dict(E.get_row(101, 0)); ev = json.loads(row["evidence"]); ev[0]["visit_datetime"] = "2099-12-31 00:00:00"
    row["evidence_items"] = ev
    with pytest.raises(E.EvidenceDatetimeMismatch):
        E.gold_visits(row, corpus)


def test_sampled_window_contains_gold_at_chunk_position(E, df, corpus):
    for r in df.to_dict("records"):
        win = E.window_visits(r, corpus)
        for e in r["evidence_items"]:
            assert win.index(e["original_visit_index"]) == e["visit_index"]


def test_bm25_tokenizer_protocol_v11(E):
    assert E.bm25_tokenize("BP 120/80, HR-72; Pt OK.") == ["bp", "120", "80", "hr", "72", "pt", "ok"]


def test_join_on_key_row_order_independent_and_strict(E, df):
    import pandas as pd
    a = df[["person_id", "qa_index"]].assign(x=range(len(df)))
    b = a.sample(frac=1, random_state=0).rename(columns={"x": "y"})
    j = E.join_on_key(a, b)
    assert (j.x == j.y).all()
    with pytest.raises(E.KeyIntegrityError):
        E.join_on_key(a, b.iloc[1:])
    with pytest.raises(E.KeyIntegrityError):
        E.assert_unique_keys(pd.concat([a, a.iloc[:1]]))


def test_build_context_chronological_headers_and_budget(E, corpus):
    tok = SF.CharTokenizer()
    ctx, rec = E.build_context(102, [30, 3, 27], budget=10_000, qa_index=0, corpus=corpus, tokenizer=tok)
    assert rec["selected_visit_ids"] == [3, 27, 30] and not rec["truncated"]
    assert ctx.index("=== VISIT 3 |") < ctx.index("=== VISIT 27 |") < ctx.index("=== VISIT 30 |")
    assert rec["gold_visit_ids"] == [27, 30] and rec["gold_fully_survived"] is True
    small = rec["header_separator_tokens"] + 60
    ctx2, rec2 = E.build_context(102, [30, 3, 27], budget=small, qa_index=0, corpus=corpus, tokenizer=tok)
    assert rec2["truncated"] and rec2["total_tokens_after"] <= small
    assert all(h in ctx2 for h in ("=== VISIT 3 |", "=== VISIT 27 |", "=== VISIT 30 |"))   # every header kept
    avail = small - rec2["header_separator_tokens"]
    exp = [b * avail // sum(rec2["per_visit_tokens_before"]) for b in rec2["per_visit_tokens_before"]]
    assert rec2["per_visit_tokens_after"] == exp                                       # proportional, from the start
    with pytest.raises(E.VisitIndexOutOfRange):
        E.build_context(103, [5], corpus=corpus, tokenizer=tok)
    with pytest.raises(E.EHRDataError):
        E.build_context(103, [0, 0], corpus=corpus, tokenizer=tok)
