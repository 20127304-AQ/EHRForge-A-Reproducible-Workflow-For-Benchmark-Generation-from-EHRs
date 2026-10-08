"""PART 1 assembly: EM/F1 (SQuAD v1.1, single gold) for all 268,550 rows + BERTScore from the volume
(download first: modal volume get <STEP4_VOLUME> /bertscore W/results/).
Writes W/results/per_row_scores.parquet, bertscore_truncation_log.csv, empty_predictions_log.csv. Run from project root."""
import hashlib, json, os
import numpy as np
import pandas as pd
import sys

import os as _os, sys as _sys  # release adaptation (paths/config only)
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))  # evaluation/ (release_config.py)
import release_config as _rc  # noqa: E402  reads evaluation/config.yaml
_rc.enter_workspace()  # inputs resolve relative to WORKSPACE_DIR (original project layout)
W = _rc.step_out("04_scoring")  # release: was the agent work directory
sys.path.insert(0, _HERE)  # release: was W/code
import judge_common as J  # noqa

K = ["person_id", "qa_index", "reader", "condition"]
READERS = ["roberta_base_squad2", "biobert_v1_1_pubmed_squad_v2", "longformer_squadv2", "qwen2_5_7b", "qwen2_5_32b"]
CONDS = ["bm25@full_timeline", "medcpt@full_timeline", "nvembed_v2@full_timeline", "hybrid_rrf60@full_timeline", "oracle"]
pairs_p = f"{W}/results/inputs/pairs_v14.parquet"
assert hashlib.sha256(open(pairs_p, "rb").read()).hexdigest() == json.load(open(pairs_p.replace(".parquet", ".meta.json")))["sha256"]
P = pd.read_parquet(pairs_p)
emp = P.empty_prediction.values
P["em"] = [0.0 if e else J.exact_match_score(p, a) for p, a, e in zip(P.prediction, P.answer, emp)]
P["f1"] = [0.0 if e else J.f1_score(p, a) for p, a, e in zip(P.prediction, P.answer, emp)]
P["n_words_pred"] = [0 if e else len(p.strip().split()) for p, e in zip(P.prediction, emp)]

trunc_rows = []
for mk in ["roberta_large", "bio_clinicalbert"]:
    fs = [f"{W}/results/bertscore/{mk}/{r}__{c}.parquet" for r in READERS for c in CONDS]
    missing = [f for f in fs if not os.path.exists(f)]
    assert not missing, missing
    B = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    assert len(B) == 268550 and not B.duplicated(K).any()
    B = B.drop(columns=["empty_prediction"])
    n0 = len(P)
    P = P.merge(B, on=K, how="left", validate="one_to_one")
    assert len(P) == n0 and P[f"bs_{mk}_F"].notna().all()
    assert (P.loc[P.empty_prediction, [f"bs_{mk}_P", f"bs_{mk}_R", f"bs_{mk}_F"]] == 0).all().all()
    g = pd.read_parquet(f"{W}/results/bertscore/{mk}/gold_tokens.parquet")
    assert len(g) == 10742
    trunc_rows.append({"model": mk, "text": "gold_answer", "reader": "", "condition": "", "n_texts": len(g),
                       "n_over_512": int((g.n_tokens_gold > 512).sum()), "max_tokens": int(g.n_tokens_gold.max())})
    for (r, c), s in P.groupby(["reader", "condition"], sort=False):
        nt = s[f"n_tokens_cand_{mk}"]
        trunc_rows.append({"model": mk, "text": "prediction", "reader": r, "condition": c, "n_texts": int((~s.empty_prediction).sum()),
                           "n_over_512": int((nt > 512).sum()), "max_tokens": int(nt.max())})
    for r, s in P.groupby("reader", sort=False):
        nt = s[f"n_tokens_cand_{mk}"]
        trunc_rows.append({"model": mk, "text": "prediction", "reader": r, "condition": "ALL", "n_texts": int((~s.empty_prediction).sum()),
                           "n_over_512": int((nt > 512).sum()), "max_tokens": int(nt.max())})

cols = K + ["reasoning_type", "difficulty", "in_judge_keys", "gold_fully_survived", "empty_prediction", "n_words_pred", "em", "f1"] + \
       [f"bs_{m}_{x}" for m in ["roberta_large", "bio_clinicalbert"] for x in "PRF"] + [f"n_tokens_cand_{m}" for m in ["roberta_large", "bio_clinicalbert"]]
out = P[cols].copy()
assert len(out) == 268550 and not out.duplicated(K).any()
assert (out.groupby(["reader", "condition"]).size() == 10742).all() and out.groupby(["reader", "condition"]).ngroups == 25
out.to_parquet(f"{W}/results/per_row_scores.parquet", index=False)
pd.DataFrame(trunc_rows).to_csv(f"{W}/results/bertscore_truncation_log.csv", index=False)
el = P.groupby(["reader", "condition"], sort=False).agg(n_rows=("prediction", "size"), n_empty_after_strip=("empty_prediction", "sum"),
                                                         n_exact_empty=("prediction", lambda s: int((s == "").sum()))).reset_index()
tot = P.groupby("reader", sort=False).agg(n_rows=("prediction", "size"), n_empty_after_strip=("empty_prediction", "sum"),
                                          n_exact_empty=("prediction", lambda s: int((s == "").sum()))).reset_index()
tot["condition"] = "ALL"
el = pd.concat([el, tot], ignore_index=True)
el["rate_empty_after_strip"] = el.n_empty_after_strip / el.n_rows
jk = P[P.in_judge_keys].groupby(["reader", "condition"], sort=False).empty_prediction.sum().rename("n_empty_in_judge_keys").reset_index()
el = el.merge(jk, on=["reader", "condition"], how="left")
el.to_csv(f"{W}/results/empty_predictions_log.csv", index=False)
print(out.groupby(["reader", "condition"], sort=False)[["em", "f1", "bs_roberta_large_F", "bs_bio_clinicalbert_F"]].mean().round(4).to_string())
print(pd.DataFrame(trunc_rows).query("n_over_512>0"))
