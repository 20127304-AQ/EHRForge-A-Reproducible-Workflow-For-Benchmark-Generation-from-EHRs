"""PART 4 aggregation (+ judge_scores assembly). Run from project root:
  W/.venv/bin/python W/code/aggregate.py
Inputs : W/results/per_row_scores.parquet, W/results/judge_full/out/chunk_*.parquet (downloaded from the volume)
Outputs: W/results/tables/*.csv, W/results/judge_scores.parquet
Bootstrap (protocol v1.4 scoring.bootstrap, unchanged): for n in {10742 (full), 1000 (judge)} a FRESH
numpy.random.default_rng(42); idx = rng.integers(0, n, size=(1000, n)) over keys sorted by (person_id, qa_index);
converted to per-resample count vectors C (1000 x n), so mean_b = C[b] @ x / n is identical to x[idx[b]].mean().
The SAME C is used for every reader/condition/stratum (paired). Stratified / subset means are ratio estimators
recomputed inside each resample: sum_k C[b,k] m_k x_k / sum_k C[b,k] m_k. CI = 95% percentile (np.percentile, linear).
Scales: EM, token F1, BERTScore F1 and all rates are x100; judge mean on 1..5."""
import glob, json, os, sys
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import os as _os, sys as _sys  # release adaptation (paths/config only)
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))  # evaluation/ (release_config.py)
import release_config as _rc  # noqa: E402  reads evaluation/config.yaml
_rc.enter_workspace()  # inputs resolve relative to WORKSPACE_DIR (original project layout)
W = _rc.step_out("04_scoring")  # release: was the agent work directory
T = f"{W}/results/tables"; os.makedirs(T, exist_ok=True)
READERS = ["roberta_base_squad2", "biobert_v1_1_pubmed_squad_v2", "longformer_squadv2", "qwen2_5_7b", "qwen2_5_32b"]
CONDS = ["bm25@full_timeline", "medcpt@full_timeline", "nvembed_v2@full_timeline", "hybrid_rrf60@full_timeline", "oracle"]
RETR = CONDS[:4]
RUNS = [(r, c) for r in READERS for c in CONDS]
RI = {rc: i for i, rc in enumerate(RUNS)}
FAMILY = {r: ("qwen" if r.startswith("qwen") else "extractive") for r in READERS}
K = ["person_id", "qa_index", "reader", "condition"]
B = 1000
SURV_NOTE = ("gold_fully_survived depends on the CONDITION (retrieval + budget), so survived/not-survived strata contain different keys "
             "in different conditions: between-condition comparisons inside a stratum are NOT paired. Only the common subset "
             "(survived in all 5 conditions) gives paired between-condition comparisons. Within-condition reader comparisons are paired.")

# ------------------------------------------------------------------ load
pr = pd.read_parquet(f"{W}/results/per_row_scores.parquet")
assert len(pr) == 268550 and not pr.duplicated(K).any()
ds = pd.read_csv("data/dataset.csv")[["person_id", "qa_index", "reasoning_type"]]
keys_full = ds.sort_values(["person_id", "qa_index"])[["person_id", "qa_index"]].reset_index(drop=True)
jk = pd.read_csv("shared/canonical/judge_keys_1000.csv").sort_values(["person_id", "qa_index"]).reset_index(drop=True)
keys_j = jk[["person_id", "qa_index"]]
n_full, n_j = len(keys_full), len(keys_j)
assert n_full == 10742 and n_j == 1000

# ---- judge_scores assembly: called rows (volume) + empty predictions (score 1, no call)
items = pd.read_parquet(f"{W}/results/judge_full/items.parquet")
outs = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{W}/results/judge_full/out/chunk_*.parquet"))], ignore_index=True)
assert not outs.item_id.duplicated().any(), "duplicate judge item"
assert set(outs.item_id) == set(items.item_id), f"judge incomplete: {len(set(items.item_id) - set(outs.item_id))} missing"
emp = pr[pr.in_judge_keys & pr.empty_prediction][K].copy()
emp = emp.assign(item_id=emp.reader + "|" + emp.condition + "|" + emp.person_id.astype(str) + "|" + emp.qa_index.astype(str),
                 score=1, score_first=np.nan, invalid_first=False, invalid_final=False, retried=False, judge_called=False,
                 empty_prediction=True, raw_feedback=None, raw_feedback_retry=None, finish_reason=None, finish_reason_retry=None,
                 n_output_tokens=0, n_prompt_tokens=0, prompt_sha256=None)
jcols = ["item_id"] + K + ["score", "score_first", "invalid_first", "invalid_final", "retried", "judge_called", "empty_prediction",
                           "raw_feedback", "raw_feedback_retry", "finish_reason", "finish_reason_retry", "n_prompt_tokens", "n_output_tokens", "prompt_sha256"]
js = pd.concat([outs[jcols], emp[jcols]], ignore_index=True)
js["score"] = js.score.astype(int)
assert len(js) == 25000 and not js.duplicated(K).any()
jset = set(zip(keys_j.person_id, keys_j.qa_index))
for (r, c), g in js.groupby(["reader", "condition"]):
    assert set(zip(g.person_id, g.qa_index)) == jset, (r, c)
assert js.score.between(1, 5).all()
js.to_parquet(f"{W}/results/judge_scores.parquet", index=False)


# ------------------------------------------------------------------ matrices
def matrix(df, col, keys):
    """R x n matrix aligned to RUNS and keys (sorted)."""
    piv = df.pivot_table(index=["person_id", "qa_index"], columns=["reader", "condition"], values=col, aggfunc="first", observed=True)
    piv = piv.reindex(pd.MultiIndex.from_frame(keys))
    M = np.vstack([piv[rc].to_numpy(dtype=np.float64) for rc in RUNS])
    assert not np.isnan(M).any(), col
    return M


def counts(n):
    rng = np.random.default_rng(42)
    idx = rng.integers(0, n, size=(B, n))
    C = np.zeros((B, n), dtype=np.float64)
    for b in range(B):
        C[b] = np.bincount(idx[b], minlength=n)
    return C


C_full, C_j = counts(n_full), counts(n_j)
prj = pr[pr.in_judge_keys]
FULL = {"em": 100 * matrix(pr, "em", keys_full), "f1": 100 * matrix(pr, "f1", keys_full),
        "bertscore_roberta_large_F1": 100 * matrix(pr, "bs_roberta_large_F", keys_full),
        "bertscore_bio_clinicalbert_F1": 100 * matrix(pr, "bs_bio_clinicalbert_F", keys_full)}
sc = matrix(js, "score", keys_j)
inv_f = matrix(js.assign(v=js.invalid_final.astype(float)), "v", keys_j)
inv_1 = matrix(js.assign(v=js.invalid_first.astype(float)), "v", keys_j)
JUDGE = {"judge_mean": sc, "judge_correct_ge4": 100 * (sc >= 4), "judge_correct_ge3": 100 * (sc >= 3), "judge_correct_eq5": 100 * (sc == 5),
         "judge_invalid_rate_final": 100 * inv_f, "judge_invalid_rate_first": 100 * inv_1}
JSUB = {f"{k}": 100 * matrix(prj, col, keys_j) for k, col in
        [("em", "em"), ("f1", "f1"), ("bertscore_roberta_large_F1", "bs_roberta_large_F"), ("bertscore_bio_clinicalbert_F1", "bs_bio_clinicalbert_F")]}
VALID = 1.0 - inv_f  # sensitivity: exclude final-INVALID rows (per-run mask)


def boot(M, mask, C):
    """mask: (n,) or (R,n) 0/1. returns point (R,), boot (B,R), n (R,)"""
    mask = np.broadcast_to(mask, M.shape).astype(np.float64)
    num = C @ (M * mask).T
    den = C @ mask.T
    with np.errstate(invalid="ignore", divide="ignore"):
        bt = num / den
    pt = (M * mask).sum(1) / mask.sum(1)
    return pt, bt, mask.sum(1).astype(int)


def ci(x):
    x = x[~np.isnan(x)]
    return (np.percentile(x, 2.5), np.percentile(x, 97.5)) if len(x) else (np.nan, np.nan)


def pval(d):
    d = d[~np.isnan(d)]
    return min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1)))


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for rank, i in enumerate(o):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


METRIC_SETS = [("full_10742", FULL, C_full, np.ones(n_full)), ("judge_1000", {**JUDGE, **JSUB}, C_j, np.ones(n_j))]

# ------------------------------------------------------------------ (a) per run
rows = []
for subset, mets, C, m1 in METRIC_SETS:
    for met, M in mets.items():
        pt, bt, n = boot(M, m1, C)
        for (r, c), i in RI.items():
            lo, hi = ci(bt[:, i])
            rows.append(dict(reader=r, condition=c, family=FAMILY[r], subset=subset, metric=met, n=int(n[i]), estimate=pt[i], ci_low=lo, ci_high=hi))
# sensitivity excluding invalids
for met in ["judge_mean", "judge_correct_ge4", "judge_correct_ge3", "judge_correct_eq5"]:
    pt, bt, n = boot(JUDGE[met], VALID, C_j)
    for (r, c), i in RI.items():
        lo, hi = ci(bt[:, i])
        rows.append(dict(reader=r, condition=c, family=FAMILY[r], subset="judge_1000_excl_invalid", metric=met, n=int(n[i]), estimate=pt[i], ci_low=lo, ci_high=hi))
main = pd.DataFrame(rows)
main.to_csv(f"{T}/table_qa_main.csv", index=False)
wide = main.assign(v=main.apply(lambda x: f"{x.estimate:.2f} [{x.ci_low:.2f}, {x.ci_high:.2f}]", axis=1))
wide = wide.pivot_table(index=["reader", "condition"], columns=["subset", "metric"], values="v", aggfunc="first")
wide = wide.reindex(pd.MultiIndex.from_tuples(RUNS)); wide.columns = [f"{a}:{b}" for a, b in wide.columns]
wide.to_csv(f"{T}/table_qa_main_wide.csv")

# ------------------------------------------------------------------ (b) families
DIFF_METRICS = [("full_10742", k, FULL[k], C_full) for k in FULL] + [("judge_1000", k, JUDGE[k], C_j) for k in ["judge_mean", "judge_correct_ge4"]]


def diff_rows(pairs_, family, mask=None, extra=None):
    out = []
    for subset, met, M, C in DIFF_METRICS:
        mk = np.ones(M.shape[1]) if mask is None else mask[subset]
        if mk.sum() == 0:
            continue
        pt, bt, n = boot(M, mk, C)
        rr = []
        for a, b in pairs_:
            d = bt[:, RI[a]] - bt[:, RI[b]]; lo, hi = ci(d)
            rr.append(dict(family=family, subset=subset, metric=met, minuend_reader=a[0], minuend_condition=a[1], subtrahend_reader=b[0],
                           subtrahend_condition=b[1], n=int(n[RI[a]]), estimate=pt[RI[a]] - pt[RI[b]], ci_low=lo, ci_high=hi,
                           ci_excludes_0=bool(lo > 0 or hi < 0), boot_p_two_sided=pval(d), **(extra or {})))
        if family.startswith("F1"):
            adj = holm([x["boot_p_two_sided"] for x in rr])
            for x, p in zip(rr, adj):
                x["holm_p_optional"] = p
        out += rr
    return out


F1p = [((r, "oracle"), (r, c)) for r in READERS for c in RETR]
F2p = [((b, c), (a, c)) for c in CONDS for a, b in combinations(READERS, 2)]
F3p = [((r, "nvembed_v2@full_timeline"), (r, "hybrid_rrf60@full_timeline")) for r in READERS]
assert len(F1p) == 20 and len(F2p) == 50 and len(F3p) == 5
f1t = pd.DataFrame(diff_rows(F1p, "F1_oracle_minus_retrieval"))
f1t.to_csv(f"{T}/table_diff_F1_oracle_vs_retrieval.csv", index=False)
pd.DataFrame(diff_rows(F2p, "F2_reader_pairs_within_condition")).drop(columns=["boot_p_two_sided"]).to_csv(f"{T}/table_diff_F2_reader_pairs.csv", index=False)
pd.DataFrame(diff_rows(F3p, "F3_nvembed_minus_hybrid")).drop(columns=["boot_p_two_sided"]).to_csv(f"{T}/table_diff_F3_nvembed_vs_hybrid.csv", index=False)
f1t = f1t.rename(columns={"boot_p_two_sided": "boot_p_two_sided_optional"}); f1t.to_csv(f"{T}/table_diff_F1_oracle_vs_retrieval.csv", index=False)

# ------------------------------------------------------------------ (c) reasoning type
rt_full = keys_full.merge(ds, on=["person_id", "qa_index"], how="left").reasoning_type.to_numpy()
rt_j = keys_j.merge(ds, on=["person_id", "qa_index"], how="left").reasoning_type.to_numpy()
rows = []
for subset, mets, C, rt in [("full_10742", FULL, C_full, rt_full), ("judge_1000", {k: JUDGE[k] for k in ["judge_mean", "judge_correct_ge4", "judge_correct_ge3", "judge_correct_eq5"]}, C_j, rt_j)]:
    for t in sorted(set(rt)):
        mk = (rt == t).astype(float)
        for met, M in mets.items():
            pt, bt, n = boot(M, mk, C)
            for (r, c), i in RI.items():
                lo, hi = ci(bt[:, i])
                rows.append(dict(reasoning_type=t, reader=r, condition=c, subset=subset, metric=met, n=int(n[i]), estimate=pt[i], ci_low=lo, ci_high=hi))
pd.DataFrame(rows).to_csv(f"{T}/table_by_reasoning_type.csv", index=False)

# ------------------------------------------------------------------ (d) gold_fully_survived
gfs = matrix(pr.assign(g=pr.gold_fully_survived.astype(float)), "g", keys_full)  # R x n ; identical across readers
for c in CONDS:
    rows_c = [RI[(r, c)] for r in READERS]
    assert (gfs[rows_c] == gfs[rows_c[0]]).all(), "survival must be identical across readers within a condition"
G = {c: gfs[RI[(READERS[0], c)]] for c in CONDS}
pos_j = pd.MultiIndex.from_frame(keys_full).get_indexer(pd.MultiIndex.from_frame(keys_j)); assert (pos_j >= 0).all()
Gj = {c: G[c][pos_j] for c in CONDS}
rows, prow = [], []
VIEW_I = [("full_10742", k, FULL[k], C_full, G) for k in FULL] + [("judge_1000", k, JUDGE[k], C_j, Gj) for k in ["judge_mean", "judge_correct_ge4"]]
for subset, met, M, C, GG in VIEW_I:
    for c in CONDS:
        for sv in (True, False):
            mk = GG[c] if sv else 1 - GG[c]
            if mk.sum() == 0:
                continue
            pt, bt, n = boot(M, mk, C)
            for r in READERS:
                i = RI[(r, c)]; lo, hi = ci(bt[:, i])
                rows.append(dict(view="i_within_condition", condition=c, gold_fully_survived=sv, reader=r, subset=subset, metric=met,
                                 n=int(n[i]), estimate=pt[i], ci_low=lo, ci_high=hi, n_nan_resamples=int(np.isnan(bt[:, i]).sum())))
            for a, b in combinations(READERS, 2):
                d = bt[:, RI[(b, c)]] - bt[:, RI[(a, c)]]; lo, hi = ci(d)
                prow.append(dict(view="i_within_condition_reader_pairs", condition=c, gold_fully_survived=sv, minuend_reader=b, subtrahend_reader=a,
                                 subset=subset, metric=met, n=int(n[RI[(a, c)]]), estimate=pt[RI[(b, c)]] - pt[RI[(a, c)]], ci_low=lo, ci_high=hi))
surv = pd.DataFrame(rows); surv["note"] = SURV_NOTE
surv.to_csv(f"{T}/table_by_gold_survival.csv", index=False)
spairs = pd.DataFrame(prow); spairs["note"] = SURV_NOTE
spairs.to_csv(f"{T}/table_by_gold_survival_reader_pairs.csv", index=False)
# (ii) common subset
common = np.prod([G[c] for c in CONDS], axis=0); common_j = common[pos_j]
rows = []
for subset, mets, C, mk in [("full_10742", FULL, C_full, common), ("judge_1000", {k: JUDGE[k] for k in ["judge_mean", "judge_correct_ge4", "judge_correct_ge3", "judge_correct_eq5"]}, C_j, common_j)]:
    for met, M in mets.items():
        pt, bt, n = boot(M, mk, C)
        for (r, c), i in RI.items():
            lo, hi = ci(bt[:, i])
            rows.append(dict(view="ii_common_subset_run", reader=r, condition=c, subtrahend_condition="", subset=subset, metric=met, n=int(n[i]),
                             estimate=pt[i], ci_low=lo, ci_high=hi))
        for r in READERS:
            for c in RETR:
                d = bt[:, RI[(r, "oracle")]] - bt[:, RI[(r, c)]]; lo, hi = ci(d)
                rows.append(dict(view="ii_common_subset_oracle_minus_retrieval", reader=r, condition="oracle", subtrahend_condition=c, subset=subset,
                                 metric=met, n=int(n[RI[(r, c)]]), estimate=pt[RI[(r, "oracle")]] - pt[RI[(r, c)]], ci_low=lo, ci_high=hi))
cs = pd.DataFrame(rows); cs["n_common_full"] = int(common.sum()); cs["n_common_judge"] = int(common_j.sum())
cs["note"] = "common subset = keys with gold_fully_survived True in ALL 5 conditions (paired between conditions)"
cs.to_csv(f"{T}/table_common_subset.csv", index=False)
# (iii) survival rate per condition
sr = []
for c in CONDS:
    for subset, g, C in [("full_10742", G[c], C_full), ("judge_1000", Gj[c], C_j)]:
        bt = C @ g / C.sum(1); lo, hi = ci(bt)
        sr.append(dict(condition=c, subset=subset, n_keys=len(g), n_survived=int(g.sum()), survival_rate=100 * g.mean(), ci_low=100 * lo, ci_high=100 * hi))
for subset, g, C in [("full_10742", common, C_full), ("judge_1000", common_j, C_j)]:
    bt = C @ g / C.sum(1); lo, hi = ci(bt)
    sr.append(dict(condition="COMMON_ALL_5", subset=subset, n_keys=len(g), n_survived=int(g.sum()), survival_rate=100 * g.mean(), ci_low=100 * lo, ci_high=100 * hi))
srt = pd.DataFrame(sr); srt["note"] = SURV_NOTE
srt.to_csv(f"{T}/table_gold_survival_rate.csv", index=False)

# ------------------------------------------------------------------ judge distribution + (e) length bias
jd = []
jj = js.merge(pr[K + ["n_words_pred"]], on=K, how="left")
for (r, c) in RUNS:
    g = jj[(jj.reader == r) & (jj.condition == c)]
    h = g.score.value_counts().reindex(range(1, 6), fill_value=0)
    v = g[~g.invalid_final]
    jd.append(dict(reader=r, condition=c, family=FAMILY[r], n=len(g), n_judge_called=int(g.judge_called.sum()), n_empty_scored_1=int(g.empty_prediction.sum()),
                   **{f"n_score_{s}": int(h[s]) for s in range(1, 6)}, **{f"pct_score_{s}": 100 * h[s] / len(g) for s in range(1, 6)},
                   n_invalid_first=int(g.invalid_first.sum()), n_retried=int(g.retried.sum()), n_invalid_final=int(g.invalid_final.sum()),
                   invalid_rate_first=100 * g.invalid_first.mean(), invalid_rate_final=100 * g.invalid_final.mean(),
                   mean_score=g.score.mean(), mean_score_excl_invalid=v.score.mean(), n_excl_invalid=len(v),
                   finish_length=int((g.finish_reason == "length").sum()), mean_output_tokens=g[g.judge_called].n_output_tokens.mean()))
pd.DataFrame(jd).to_csv(f"{T}/table_judge_distribution.csv", index=False)
lb = []
for (r, c) in RUNS:
    g = jj[(jj.reader == r) & (jj.condition == c)]
    gn = g[~g.empty_prediction]
    rho, p = spearmanr(g.score, g.n_words_pred)
    rho2, p2 = spearmanr(gn.score, gn.n_words_pred) if gn.n_words_pred.nunique() > 1 else (np.nan, np.nan)
    full_len = pr[(pr.reader == r) & (pr.condition == c)].n_words_pred
    lb.append(dict(reader=r, condition=c, family=FAMILY[r], n=len(g), spearman_score_vs_words=rho, p=p, spearman_excl_empty=rho2, p_excl_empty=p2,
                   mean_words_judge_1000=g.n_words_pred.mean(), median_words_judge_1000=g.n_words_pred.median(), mean_words_full_10742=full_len.mean()))
lbt = pd.DataFrame(lb)
per_reader = pr.groupby("reader").n_words_pred.agg(["mean", "median"]).reindex(READERS).reset_index()
for _, x in per_reader.iterrows():
    lb.append(dict(reader=x.reader, condition="ALL", family=FAMILY[x.reader], n=int(5 * n_full), mean_words_full_10742=x["mean"], median_words_judge_1000=np.nan))
pd.DataFrame(lb).to_csv(f"{T}/judge_length_bias.csv", index=False)
summ = {"n_judge_rows": len(js), "n_called": int(js.judge_called.sum()), "n_empty": int(js.empty_prediction.sum()),
        "invalid_first": int(js.invalid_first.sum()), "invalid_final": int(js.invalid_final.sum()), "n_common_full": int(common.sum()),
        "n_common_judge": int(common_j.sum()), "tables": sorted(os.listdir(T))}
json.dump(summ, open(f"{W}/results/aggregate_summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
