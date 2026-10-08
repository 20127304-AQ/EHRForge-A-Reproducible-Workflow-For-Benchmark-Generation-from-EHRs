"""EHRForge shared data layer (protocol v1.1). Canonical copy: shared/canonical/ehr_data.py
Author: agent_step1_protocol_v1_1_full_depth_retriever_72af8248 (Step 1).

Rules enforced here
- every frozen input is sha256-checked on load (dataset, corpus, protocol);
- gold visits are ALWAYS original_visit_index (never visit_index); each evidence datetime must equal the corpus
  datetime at that index; out-of-range indices raise;
- all joins are on (person_id, qa_index) with missing/duplicate-key assertions; never by row position;
- BM25 tokenizer = protocol v1.1 change (1);
- build_context implements protocol v1.0 `context` + `context_budget.over_budget_policy` and returns the v1.1
  `context_metadata` record (reader-specific fields null at this stage).

Paths: the repository root is found as the parent of the directory containing `shared/` (this file lives in
shared/canonical/). Override with env EHRFORGE_ROOT (e.g. inside Modal containers).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("EHRFORGE_ROOT") or os.path.abspath(os.path.join(_HERE, "..", ".."))
CANON = os.path.join(ROOT, "shared", "canonical")

DATASET_PATH = os.path.join(ROOT, "data", "dataset.csv")
CORPUS_PATH = os.path.join(ROOT, "data", "corpus.json")
DATASET_SHA256 = "66a968330bea85bef75f847a6a0c4e8bb8352e0ac363c85c75436b0ee034847f"
CORPUS_SHA256 = "b77d4b75ff4a8bd6ce8c9138a718ae04c0d76bf3066218b762a2c887d1a5d0a2"
CORPUS_SIZE = 162124534
N_ROWS, N_DATASET_PATIENTS, N_EVIDENCE = 10742, 829, 24141
N_CORPUS_PATIENTS, N_CORPUS_VISITS = 961, 170827

PROTOCOLS = {
    "1.0-step0": ("protocol.yaml", "3762cf63e8ebe43fa7c3729f6847673f15efa971455498af987791227c2b0ad8"),
    "1.1": ("protocol_v1.1.yaml", "6a4d381ba818925707cdf65a42cc111af87de78cc976df1b55efd6e6eb466be1"),
}
KEY = ("person_id", "qa_index")

# context (protocol v1.0 `context` / `context_budget`, unchanged in v1.1)
BUDGET_TOKENIZER_REPO = "Qwen/Qwen2.5-7B-Instruct"
BUDGET_TOKENIZER_SHA = "a09a35458c702b33eeacc393d103063234e8bc28"
HEADER_FMT = "=== VISIT {idx} | {datetime} ==="
BLOCK_FMT = "{header}\n{text}"
BLOCK_SEP = "\n\n"
DEFAULT_BUDGET = 3500


class EHRDataError(ValueError):
    """Base error for data-layer integrity violations."""


class ShaMismatchError(EHRDataError):
    pass


class EvidenceDatetimeMismatch(EHRDataError):
    pass


class VisitIndexOutOfRange(EHRDataError, IndexError):
    pass


class KeyIntegrityError(EHRDataError):
    pass


# --------------------------------------------------------------------------------------------- hashing
def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _assert_sha(path: str, expected: str) -> None:
    got = sha256_file(path)
    if got != expected:
        raise ShaMismatchError(f"{path}: sha256 {got} != expected {expected}")


# --------------------------------------------------------------------------------------------- loaders
@lru_cache(maxsize=None)
def load_protocol(version: str = "1.1") -> dict:
    """Load shared/canonical protocol `version` ('1.1' or '1.0-step0'); asserts its sha256."""
    import yaml
    if version not in PROTOCOLS:
        raise KeyError(f"unknown protocol version {version!r}; known {sorted(PROTOCOLS)}")
    fn, sha = PROTOCOLS[version]
    path = os.path.join(CANON, fn)
    _assert_sha(path, sha)
    with open(path, encoding="utf-8") as f:
        p = yaml.safe_load(f)
    if str(p["protocol_version"]) != version:
        raise EHRDataError(f"{fn} declares protocol_version {p['protocol_version']!r}, expected {version!r}")
    return p


@lru_cache(maxsize=None)
def load_dataset():
    """pandas DataFrame of data/dataset.csv (sha-checked), with key uniqueness asserted and an extra column
    `evidence_items` (parsed JSON list). Row order is the file order but must never be used for joins."""
    import pandas as pd
    _assert_sha(DATASET_PATH, DATASET_SHA256)
    df = pd.read_csv(DATASET_PATH)
    if len(df) != N_ROWS:
        raise EHRDataError(f"dataset rows {len(df)} != {N_ROWS}")
    assert_unique_keys(df, "dataset")
    if df["person_id"].nunique() != N_DATASET_PATIENTS:
        raise EHRDataError("dataset patient count mismatch")
    df["evidence_items"] = [json.loads(s) for s in df["evidence"]]
    n_ev = int(sum(len(x) for x in df["evidence_items"]))
    if n_ev != N_EVIDENCE:
        raise EHRDataError(f"evidence items {n_ev} != {N_EVIDENCE}")
    return df


@lru_cache(maxsize=None)
def load_corpus() -> dict:
    """dict person_id(int) -> list of visits [{visit_datetime, text}] (list position = original_visit_index)."""
    if os.path.getsize(CORPUS_PATH) != CORPUS_SIZE:
        raise ShaMismatchError(f"{CORPUS_PATH}: size {os.path.getsize(CORPUS_PATH)} != {CORPUS_SIZE}")
    _assert_sha(CORPUS_PATH, CORPUS_SHA256)
    corpus: dict[int, list] = {}
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            pid = int(d["person_id"])
            if pid in corpus:
                raise KeyIntegrityError(f"duplicate person_id {pid} in corpus")
            corpus[pid] = d["visits"]
    nv = sum(len(v) for v in corpus.values())
    if len(corpus) != N_CORPUS_PATIENTS or nv != N_CORPUS_VISITS:
        raise EHRDataError(f"corpus counts {len(corpus)} patients / {nv} visits != {N_CORPUS_PATIENTS}/{N_CORPUS_VISITS}")
    return corpus


# --------------------------------------------------------------------------------------------- keys / joins
def assert_unique_keys(df, name: str = "frame") -> None:
    missing = [k for k in KEY if k not in df.columns]
    if missing:
        raise KeyIntegrityError(f"{name}: missing key columns {missing}")
    if df[list(KEY)].isna().any().any():
        raise KeyIntegrityError(f"{name}: null key values")
    dup = df.duplicated(list(KEY), keep=False)
    if dup.any():
        raise KeyIntegrityError(f"{name}: {int(dup.sum())} rows with duplicate (person_id, qa_index)")


def join_on_key(left, right, *, how: str = "inner", require_complete: bool = True, suffixes=("", "_r")):
    """Join two frames on (person_id, qa_index). Asserts unique keys on both sides and (by default) that the
    key sets are identical (no missing keys on either side). Never joins by row position."""
    import pandas as pd
    assert_unique_keys(left, "left")
    assert_unique_keys(right, "right")
    lk = set(map(tuple, left[list(KEY)].astype("int64").itertuples(index=False, name=None)))
    rk = set(map(tuple, right[list(KEY)].astype("int64").itertuples(index=False, name=None)))
    if require_complete and lk != rk:
        raise KeyIntegrityError(f"key sets differ: {len(lk - rk)} only-left, {len(rk - lk)} only-right")
    out = pd.merge(left, right, on=list(KEY), how=how, validate="one_to_one", suffixes=suffixes)
    return out


def get_row(person_id: int, qa_index: int) -> Mapping[str, Any]:
    """Dataset row by key (exactly one row, else KeyIntegrityError)."""
    df = _dataset_by_key()
    try:
        return df.loc[(int(person_id), int(qa_index))]
    except KeyError:
        raise KeyIntegrityError(f"no dataset row for key ({person_id}, {qa_index})") from None


@lru_cache(maxsize=None)
def _dataset_by_key():
    d = load_dataset().set_index(list(KEY), drop=False).sort_index()
    if not d.index.is_unique:
        raise KeyIntegrityError("duplicate (person_id, qa_index) in dataset index")
    return d


# --------------------------------------------------------------------------------------------- gold / window
def _evidence(row) -> list:
    ev = row["evidence_items"] if "evidence_items" in row else row["evidence"]
    return json.loads(ev) if isinstance(ev, str) else list(ev)


def gold_visits(row, corpus: Mapping[int, list] | None = None) -> list[int]:
    """Sorted, deduplicated original_visit_index of the row's evidence (NEVER visit_index).
    Raises VisitIndexOutOfRange if an index is outside the patient's corpus timeline and
    EvidenceDatetimeMismatch if evidence.visit_datetime != corpus visit_datetime at that index."""
    corpus = load_corpus() if corpus is None else corpus
    pid = int(row["person_id"])
    if pid not in corpus:
        raise KeyIntegrityError(f"person_id {pid} not in corpus")
    visits = corpus[pid]
    ids = set()
    for e in _evidence(row):
        ovi = e["original_visit_index"]
        if isinstance(ovi, bool) or int(ovi) != ovi:
            raise EHRDataError(f"non-integer original_visit_index {ovi!r} for person {pid}")
        ovi = int(ovi)
        if not 0 <= ovi < len(visits):
            raise VisitIndexOutOfRange(f"person {pid}: original_visit_index {ovi} outside [0, {len(visits)})")
        if visits[ovi]["visit_datetime"] != e["visit_datetime"]:
            raise EvidenceDatetimeMismatch(
                f"person {pid} idx {ovi}: evidence {e['visit_datetime']!r} != corpus {visits[ovi]['visit_datetime']!r}")
        ids.add(ovi)
    if not ids:
        raise EHRDataError(f"person {pid}: no evidence items")
    return sorted(ids)


def _sampled_window_module():
    # compile from source (never writes __pycache__ into shared/canonical/)
    import types
    path = os.path.join(CANON, "sampled_window.py")
    mod = types.ModuleType("ehr_sampled_window")
    mod.__file__ = path
    with open(path, encoding="utf-8") as f:
        exec(compile(f.read(), path, "exec"), mod.__dict__)
    return mod


_SW = None


def window_visits(row, corpus: Mapping[int, list] | None = None) -> list[int]:
    """Candidate original_visit_index list of the row's sampled_window (delegates to shared/canonical/sampled_window.py)."""
    global _SW
    if _SW is None:
        _SW = _sampled_window_module()
    corpus = load_corpus() if corpus is None else corpus
    V = len(corpus[int(row["person_id"])])
    return list(_SW.window(str(row["timeline_sampling_strategy"]), V))


# --------------------------------------------------------------------------------------------- BM25
_PUNCT = re.compile(r"[^\w\s]")


def bm25_tokenize(s: str) -> list[str]:
    """Protocol v1.1 change (1): re.sub(r'[^\\w\\s]', ' ', s.lower()).split(). Same for queries and documents."""
    return _PUNCT.sub(" ", s.lower()).split()


# --------------------------------------------------------------------------------------------- context
@lru_cache(maxsize=None)
def budget_tokenizer():
    """Qwen/Qwen2.5-7B-Instruct tokenizer.json at the pinned sha (tokenizer files only, no weights), loaded with
    the `tokenizers` library (same Rust tokenizer that transformers' fast tokenizer wraps)."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    path = hf_hub_download(BUDGET_TOKENIZER_REPO, "tokenizer.json", revision=BUDGET_TOKENIZER_SHA)
    return Tokenizer.from_file(path)


def _enc(tok, s: str) -> list[int]:
    return tok.encode(s, add_special_tokens=False).ids


def _dec(tok, ids: Sequence[int]) -> str:
    # transformers Qwen2TokenizerFast.decode defaults: skip_special_tokens=False, clean_up_tokenization_spaces=False
    return tok.decode(list(ids), skip_special_tokens=False)


def _assemble(headers: Sequence[str], texts: Sequence[str]) -> str:
    return BLOCK_SEP.join(BLOCK_FMT.format(header=h, text=t) for h, t in zip(headers, texts))


def build_context(person_id: int, visit_ids: Iterable[int], budget: int = DEFAULT_BUDGET, *,
                  qa_index: int | None = None, condition: str | None = None, reader: str | None = None,
                  gold_visit_ids: Sequence[int] | None = None, corpus: Mapping[int, list] | None = None,
                  tokenizer=None) -> tuple[str, dict]:
    """Assemble the reader context for `visit_ids` (full-timeline indices) per protocol v1.0 `context` and
    `context_budget.over_budget_policy`, returning (context_string, context_metadata_record).

    Policy (v1.0, unchanged): blocks in chronological order (ascending index); block = header + '\\n' + text;
    blocks joined by '\\n\\n'; tokens measured with the Qwen2.5-7B tokenizer, add_special_tokens=False, on the full
    assembled string. If over budget: keep every header; available = budget - tokens(headers+separators);
    visit text i keeps floor(len_i * available / sum_j len_j) tokens from its START; decode kept tokens.
    tokens(headers+separators) := sum_i tokens(header_i + '\\n') + (n-1) * tokens('\\n\\n') (piecewise).

    If gold_visit_ids is None and qa_index is given, gold is taken from the dataset row (person_id, qa_index).
    Reader-specific fields (reader_tokenizer_*, reader_input_tokens, n_windows, window_stride, max_seq_len) are None.
    """
    corpus = load_corpus() if corpus is None else corpus
    tok = budget_tokenizer() if tokenizer is None else tokenizer
    pid = int(person_id)
    if pid not in corpus:
        raise KeyIntegrityError(f"person_id {pid} not in corpus")
    visits = corpus[pid]
    ids_in = [int(v) for v in visit_ids]
    if len(set(ids_in)) != len(ids_in):
        raise EHRDataError(f"duplicate visit ids in selection: {ids_in}")
    if not ids_in:
        raise EHRDataError("empty visit selection")
    for v in ids_in:
        if not 0 <= v < len(visits):
            raise VisitIndexOutOfRange(f"person {pid}: visit {v} outside [0, {len(visits)})")
    ids = sorted(ids_in)  # chronological (ascending full-timeline index)
    if gold_visit_ids is None and qa_index is not None:
        gold_visit_ids = gold_visits(get_row(pid, qa_index), corpus)
    gold = sorted({int(g) for g in gold_visit_ids}) if gold_visit_ids is not None else None

    headers = [HEADER_FMT.format(idx=v, datetime=visits[v]["visit_datetime"]) for v in ids]
    texts = [visits[v]["text"] for v in ids]
    text_ids = [_enc(tok, t) for t in texts]
    before = [len(x) for x in text_ids]
    ctx_before = _assemble(headers, texts)
    total_before = len(_enc(tok, ctx_before))
    hs_tokens = sum(len(_enc(tok, h + "\n")) for h in headers) + (len(ids) - 1) * len(_enc(tok, BLOCK_SEP))
    truncated = total_before > budget

    if truncated:
        available = budget - hs_tokens
        if available < 0:
            raise EHRDataError(f"headers+separators alone ({hs_tokens}) exceed budget {budget}")
        denom = sum(before)
        keep = [math.floor(b * available / denom) if denom else 0 for b in before]
        texts_after = [_dec(tok, ti[:k]) if k < len(ti) else t for ti, k, t in zip(text_ids, keep, texts)]
        context = _assemble(headers, texts_after)
        after = [len(_enc(tok, t)) for t in texts_after]
    else:
        context, texts_after, after = ctx_before, texts, list(before)
    total_after = len(_enc(tok, context))

    reaching = [v for v, a in zip(ids, after) if a >= 1]
    rec: dict[str, Any] = {
        "person_id": pid,
        "qa_index": None if qa_index is None else int(qa_index),
        "reader": reader,
        "condition": condition,
        "n_gold_visits": None if gold is None else len(gold),
        "gold_visit_ids": gold,
        "n_selected_visits": len(ids),
        "selected_visit_ids": ids,
        "n_visits_reaching_model": len(reaching),
        "visit_ids_reaching_model": reaching,
        "total_tokens_before": total_before,
        "total_tokens_after": total_after,
        "per_visit_tokens_before": before,
        "per_visit_tokens_after": after,
        "header_separator_tokens": hs_tokens,
        "truncated": truncated,
        "n_gold_visits_surviving": None,
        "gold_fully_survived": None,
        "gold_tokens_before": None,
        "gold_tokens_after": None,
        "budget_tokens": int(budget),
        "tokenizer_name": BUDGET_TOKENIZER_REPO,
        "tokenizer_revision": BUDGET_TOKENIZER_SHA,
        "reader_tokenizer_name": None,
        "reader_tokenizer_revision": None,
        "reader_input_tokens": None,
        "n_windows": None,
        "window_stride": None,
        "max_seq_len": None,
        "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
    }
    if gold is not None:
        pos = {v: i for i, v in enumerate(ids)}
        reach = set(reaching)
        rec["n_gold_visits_surviving"] = sum(1 for g in gold if g in reach)
        rec["gold_fully_survived"] = all(g in pos and after[pos[g]] == before[pos[g]] for g in gold)
        rec["gold_tokens_before"] = sum(before[pos[g]] for g in gold if g in pos)
        rec["gold_tokens_after"] = sum(after[pos[g]] for g in gold if g in pos)
    return context, rec


CONTEXT_METADATA_FIELDS = (
    "person_id", "qa_index", "reader", "condition", "n_gold_visits", "gold_visit_ids", "n_selected_visits",
    "selected_visit_ids", "n_visits_reaching_model", "visit_ids_reaching_model", "total_tokens_before",
    "total_tokens_after", "per_visit_tokens_before", "per_visit_tokens_after", "header_separator_tokens",
    "truncated", "n_gold_visits_surviving", "gold_fully_survived", "gold_tokens_before", "gold_tokens_after",
    "budget_tokens", "tokenizer_name", "tokenizer_revision", "reader_tokenizer_name", "reader_tokenizer_revision",
    "reader_input_tokens", "n_windows", "window_stride", "max_seq_len", "context_sha256",
)
