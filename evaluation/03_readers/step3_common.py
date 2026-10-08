"""Step-3 shared constants / helpers (pure python; safe to import locally and in containers)."""
import hashlib
import json
import re
import string

import os  # release adaptation: Modal names from EHRFORGE_* env vars (evaluation/config.yaml)
VOLUME_NAME = os.environ.get("EHRFORGE_STEP3_VOLUME", "ehrforge-step3-v13")
HF_CACHE_VOLUME = os.environ.get("EHRFORGE_STEP3_HF_VOLUME", "ehrforge-step3-hfcache")
VOL = "/vol"
OUT = "/vol/step3"

RETRIEVERS = ["bm25", "medcpt", "nvembed_v2", "hybrid_rrf60"]
CONDITIONS = [f"{r}@full_timeline" for r in RETRIEVERS] + ["oracle"]
READER_TOP_K = 5
BUDGET = 3500

MANIFEST_SHA = {  # from shared/retrieval_v1.1/MANIFEST_rankings.csv (verified again at runtime against the CSV)
    "bm25@full_timeline": "d70b726567e9f7b9191c9cf99fac054ee5eea63bca44f73825d399bd8b5524cf",
    "medcpt@full_timeline": "d2ee74487e8fb37262dac69cf526c70194da177911392f3520fd0d6bb0855f82",
    "nvembed_v2@full_timeline": "98764f6cbd2ccfab466fabd7f67c0cf19cb1a6b1dfd5909bc4f442a1449729ae",
    "hybrid_rrf60@full_timeline": "7df417f5466b60c147c97a3a027b60397e072c5ad5ce44e9c5f1e75e445a142f",
}

PROTOCOL_V12_SHA = "625dc5b7655cec682f44f43dc2872502448a18575d432175c7d9cdefa7b24cbd"

READERS = {
    "roberta_base_squad2": {"repo": "deepset/roberta-base-squad2", "kind": "extractive", "max_seq_len": 512},
    "biobert_v1_1_pubmed_squad_v2": {"repo": "ktrapeznikov/biobert_v1.1_pubmed_squad_v2", "kind": "extractive",
                                     "max_seq_len": 512},
    "longformer_squadv2": {"repo": "mrm8488/longformer-base-4096-finetuned-squadv2", "kind": "extractive",
                           "max_seq_len": 4096},
    "qwen2_5_7b": {"repo": "Qwen/Qwen2.5-7B-Instruct", "kind": "generative"},
    "qwen2_5_32b": {"repo": "Qwen/Qwen2.5-32B-Instruct", "kind": "generative"},
}
EXTRACTIVE = ["roberta_base_squad2", "biobert_v1_1_pubmed_squad_v2", "longformer_squadv2"]
REMOVED_READER = "bio_clinicalbert_qa"  # failed pre-check (b); replaced (user-approved) by biobert_v1_1_pubmed_squad_v2
GENERATIVE = ["qwen2_5_7b", "qwen2_5_32b"]

# extractive decoding (protocol v1.2 readers.extractive + Step-3 task)
DOC_STRIDE = 128
MAX_ANSWER_LEN = 64
EXTRACTIVE_BATCH_SIZE = {"roberta_base_squad2": 64, "biobert_v1_1_pubmed_squad_v2": 64, "longformer_squadv2": 8}

SUBMISSION_CHUNK = 1024


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canon_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


# SQuAD v1.1 normalize_answer (protocol scoring.token_f1_em)
def normalize_answer(s: str) -> str:
    def remove_articles(t):
        return re.sub(r"\b(a|an|the)\b", " ", t)

    def white_space_fix(t):
        return " ".join(t.split())

    def remove_punc(t):
        exclude = set(string.punctuation)
        return "".join(ch for ch in t if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def f1_em(pred: str, gold: str):
    from collections import Counter
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    em = float(normalize_answer(pred) == normalize_answer(gold))
    if not p or not g:
        return float(p == g), em
    common = Counter(p) & Counter(g)
    ns = sum(common.values())
    if ns == 0:
        return 0.0, em
    pr, rc = ns / len(p), ns / len(g)
    return 2 * pr * rc / (pr + rc), em


def extractive_spec(P, reader):
    """Reader spec from protocol v1.3: readers.extractive (v1.2 text) or reader_conditions.reader_substitution.added."""
    sub = P["reader_conditions"]["reader_substitution"]
    if reader == sub["removed"]["reader"]:
        raise ValueError(f"{reader} was removed from the v1.3 reader set")
    if reader in sub["added"]:
        d = sub["added"][reader]
    else:
        d = P["readers"]["extractive"][reader]
    assert reader in P["reader_conditions"]["readers"], reader
    return {"repo": d["repo"], "sha": d["sha"], "max_seq_len": d["max_seq_len"], "doc_stride": d["doc_stride"]}
