"""PART 1 + PART 2: per-question retrieval metrics (protocol v1.2 retrieval_metrics) for 8 conditions, exact random
baselines, paired bootstrap CIs, lift, and stratified tables. CPU only; reads shared/retrieval_v1.1 rankings (read-only).
Run from project root:  PYTHONDONTWRITEBYTECODE=1 <venv>/bin/python <workdir>/code/part12_metrics.py
"""
import json, os, sys, time
from fractions import Fraction
from functools import lru_cache
from math import comb

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, "shared/canonical")
import ehr_data as E  # noqa: E402

import os as _os, sys as _sys  # release adaptation (paths/config only)
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))  # evaluation/ (release_config.py)
import release_config as _rc  # noqa: E402  reads evaluation/config.yaml
_rc.enter_workspace()  # inputs resolve relative to WORKSPACE_DIR (original project layout)
WD = _rc.step_out("02_retrieval_metrics")  # release: was the agent work directory
OUT = os.path.join(WD, "results", "tables")
os.makedirs(OUT, exist_ok=True)
RK = "shared/retrieval_v1.1/rankings"
RETRIEVERS = ["bm25", "medcpt", "nvembed_v2", "hybrid_rrf60"]
SPACES = ["full_timeline", "sampled_window"]
KS = [5, 10, 20]
KEY = ["person_id", "qa_index"]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


# protocol v1.2 sanity: the definitions implemented here are the v1.2 ones
P = yaml.safe_load(open("shared/canonical/protocol_v1.2.yaml", encoding="utf-8"))
assert P["protocol_version"] == 1.2 and P["retrieval_metrics"]["k"] == KS
B = P["scoring"]["bootstrap"]
assert B["n_resamples"] == 1000 and B["seed"] == 42

# ---------------------------------------------------------------------------------------------- gold from ehr_data
df = E.load_dataset()
corpus = E.load_corpus()
df = df.sort_values(KEY).reset_index(drop=True)          # protocol bootstrap order: keys sorted by (person_id, qa_index)
gold = [E.gold_visits(r, corpus) for r in df.to_dict("records")]
win = [set(E.window_visits(r, corpus)) for r in df.to_dict("records")]
base = df[KEY + ["visit_group", "reasoning_type", "difficulty", "timeline_sampling_strategy"]].copy()
base["gold"] = gold
base["G"] = [len(g) for g in gold]
base["n_gold_outside_window"] = [sum(1 for v in g if v not in w) for g, w in zip(gold, win)]
base["V"] = [len(corpus[int(p)]) for p in df.person_id]
E.assert_unique_keys(base, "dataset")
NQ = len(base)
assert NQ == 10742
log("gold built", NQ, "questions; gold items", int(base.G.sum()),
    "; questions with gold outside window:", int((base.n_gold_outside_window > 0).sum()))
gold_rows = base[KEY + ["gold"]].explode("gold").rename(columns={"gold": "original_visit_index"})
gold_rows["original_visit_index"] = gold_rows["original_visit_index"].astype("int64")


# ---------------------------------------------------------------------------------------------- exact random baselines
@lru_cache(maxsize=None)
def rand_baseline(N, Gin, G):
    """Exact expectations under a uniformly random ranking of N candidates, Gin of which are gold (G total gold;
    G - Gin gold are outside the candidate set and can never be retrieved -> misses)."""
    out = {}
    for k in KS:
        ke = min(k, N)
        if Gin == 0:
            out[f"hit@{k}"] = 0.0; out[f"recall@{k}"] = 0.0; out[f"coverage@{k}"] = 0.0
            continue
        out[f"hit@{k}"] = float(1 - Fraction(comb(N - Gin, ke), comb(N, ke)))
        out[f"recall@{k}"] = float(Fraction(ke, N) * Fraction(Gin, G))
        out[f"coverage@{k}"] = float(Fraction(comb(N - Gin, ke - Gin), comb(N, ke))) if (ke >= Gin and Gin == G) else 0.0
    if Gin == 0:
        out["mrr"] = 0.0; out["last_gold_rank"] = np.nan; out["first_gold_pct_rank"] = np.nan
    else:
        cg = comb(N, Gin)
        out["mrr"] = float(sum(Fraction(comb(N - r, Gin - 1), r * cg) for r in range(1, N - Gin + 2)))
        out["last_gold_rank"] = float(Fraction(Gin * (N + 1), Gin + 1)) if Gin == G else np.nan
        out["first_gold_pct_rank"] = float(Fraction(N + 1, (Gin + 1) * N))
    return out


# self-check of closed forms against brute-force enumeration on small cases (exact, not simulation)
from itertools import combinations  # noqa: E402
for N_, G_ in [(1, 1), (3, 1), (5, 2), (7, 3), (8, 8), (12, 2), (25, 4)]:
    pos = list(combinations(range(1, N_ + 1), G_)); n = len(pos)
    ref = {}
    for k in KS:
        ref[f"hit@{k}"] = sum(any(p <= k for p in c) for c in pos) / n
        ref[f"recall@{k}"] = sum(sum(p <= k for p in c) / G_ for c in pos) / n
        ref[f"coverage@{k}"] = sum(all(p <= k for p in c) for c in pos) / n
    ref["mrr"] = sum(1 / min(c) for c in pos) / n
    ref["last_gold_rank"] = sum(max(c) for c in pos) / n
    ref["first_gold_pct_rank"] = sum(min(c) / N_ for c in pos) / n
    got = rand_baseline(N_, G_, G_)
    for m in ref:
        assert abs(ref[m] - got[m]) < 1e-12, (N_, G_, m, ref[m], got[m])
log("closed-form random baselines verified against exact enumeration")

METRICS = [f"{m}@{k}" for m in ["hit", "recall", "coverage"] for k in KS] + ["mrr", "last_gold_rank", "first_gold_pct_rank"]

# ---------------------------------------------------------------------------------------------- per-question metrics
frames, audit = [], {}
for sp in SPACES:
    for r in RETRIEVERS:
        cond = f"{r}@{sp}"
        h = pd.read_parquet(f"{RK}/{cond}.header.parquet",
                            columns=KEY + ["n_candidates", "ranking_depth", "gold_visit_ids", "candidate_visit_ids", "n_visits_patient"])
        hb = E.join_on_key(base[KEY + ["gold", "G", "V", "n_gold_outside_window", "timeline_sampling_strategy"]], h)  # strict
        assert len(hb) == NQ
        assert (hb.ranking_depth == hb.n_candidates).all() and (hb.n_visits_patient == hb.V).all()
        hdr_gold_mismatch = int(sum(list(map(int, a)) != b for a, b in zip(hb.gold_visit_ids, hb.gold)))
        if sp == "full_timeline":
            assert (hb.n_candidates == hb.V).all()
        else:
            exp_n = [len(w) for w in win]  # base order == sorted keys; hb is merged in same key order? re-derive safely
            wN = pd.DataFrame({"person_id": base.person_id, "qa_index": base.qa_index, "wN": exp_n})
            chk = hb.merge(wN, on=KEY, validate="one_to_one")
            assert (chk.n_candidates == chk.wN).all()
        rk = pd.read_parquet(f"{RK}/{cond}.ranked.parquet", columns=KEY + ["original_visit_index", "rank"])
        cnt = rk.groupby(KEY).size().rename("nrows").reset_index()
        cc = E.join_on_key(hb[KEY + ["n_candidates"]], cnt)
        assert (cc.nrows == cc.n_candidates).all(), f"{cond}: ranked rows != N"
        assert not rk.duplicated(KEY + ["original_visit_index"]).any() and not rk.duplicated(KEY + ["rank"]).any()
        g = gold_rows.merge(rk, on=KEY + ["original_visit_index"], how="left", validate="one_to_one")
        del rk
        g["in_rank"] = g["rank"].notna()
        agg = {"G_in": ("in_rank", "sum"), "first_rank": ("rank", "min"), "max_rank": ("rank", "max")}
        for k in KS:
            g[f"c{k}"] = (g["rank"] <= k)
            agg[f"cnt@{k}"] = (f"c{k}", "sum")
        q = g.groupby(KEY).agg(**agg).reset_index()
        q = E.join_on_key(hb[KEY + ["G", "n_candidates", "n_gold_outside_window"]], q)
        q = q.rename(columns={"n_candidates": "N"})
        assert ((q.G - q.G_in) == (q.n_gold_outside_window if sp == "sampled_window" else 0)).all()
        for k in KS:
            q[f"hit@{k}"] = (q[f"cnt@{k}"] > 0).astype(float)
            q[f"recall@{k}"] = q[f"cnt@{k}"] / q.G
            q[f"coverage@{k}"] = (q[f"cnt@{k}"] == q.G).astype(float)
        q["mrr"] = np.where(q.G_in > 0, 1.0 / q.first_rank, 0.0)
        q["last_gold_rank"] = np.where(q.G_in == q.G, q.max_rank, np.nan)    # undefined if some gold not retrievable
        q["first_gold_rank"] = q.first_rank
        q["first_gold_pct_rank"] = q.first_rank / q.N
        rb = pd.DataFrame([rand_baseline(int(n), int(gi), int(gg)) for n, gi, gg in zip(q.N, q.G_in, q.G)])
        for m in METRICS:
            q[f"rand_{m}"] = rb[m].values
        for k in KS:
            q[f"N_le_{k}"] = q.N <= k
        q.insert(0, "condition", cond); q.insert(1, "retriever", r); q.insert(2, "search_space", sp)
        q["gold_outside_window"] = q.n_gold_outside_window > 0 if sp == "sampled_window" else False
        q = q.drop(columns=["first_rank", "max_rank"] + [f"cnt@{k}" for k in KS])
        frames.append(q)
        audit[cond] = {"n_questions": int(len(q)), "header_gold_mismatch_vs_ehr_data": hdr_gold_mismatch,
                       "n_questions_gold_outside_window": int(q.gold_outside_window.sum()),
                       "pct_questions_gold_outside_window": float(100 * q.gold_outside_window.mean()),
                       "n_gold_items_outside_window": int((q.G - q.G_in).sum()),
                       "n_questions_N_le_5": int(q.N_le_5.sum()), "n_questions_N_le_10": int(q.N_le_10.sum()),
                       "n_questions_N_le_20": int(q.N_le_20.sum())}
        log(cond, audit[cond], "hit@5=%.4f recall@5=%.4f mrr=%.4f" % (q["hit@5"].mean(), q["recall@5"].mean(), q.mrr.mean()))

pq = pd.concat(frames, ignore_index=True)
strata = base[KEY + ["visit_group", "reasoning_type", "difficulty", "timeline_sampling_strategy"]]
pq = pq.merge(strata, on=KEY, how="left", validate="many_to_one")
assert not pq.duplicated(["condition"] + KEY).any() and len(pq) == 8 * NQ


def n_bin(n):
    return "<=10" if n <= 10 else "11-30" if n <= 30 else "31-100" if n <= 100 else "101-300" if n <= 300 else ">300"


pq["N_stratum"] = pq.N.map(n_bin)
pq["G_stratum"] = np.where(pq.G == 1, "1", np.where(pq.G == 2, "2", ">=3"))
pq = pq.sort_values(["search_space", "retriever"] + KEY, key=lambda s: s.map({x: i for i, x in enumerate(RETRIEVERS + SPACES)}) if s.name in ("retriever", "search_space") else s).reset_index(drop=True)
pq.to_parquet(os.path.join(OUT, "per_question_metrics.parquet"), index=False)
json.dump(audit, open(os.path.join(OUT, "audit_part1.json"), "w"), indent=1)
log("per_question_metrics.parquet", pq.shape)

# ---------------------------------------------------------------------------------------------- bootstrap (paired, shared idx)
rng = np.random.default_rng(42)
IDX = rng.integers(0, NQ, size=(1000, NQ))
np.save(os.path.join(OUT, "bootstrap_idx_sha_check.npy"), IDX[:2, :5])  # tiny fingerprint only
ci = lambda a: (float(np.nanpercentile(a, 2.5)), float(np.nanpercentile(a, 97.5)))
rows = []
for cond, q in pq.groupby("condition", sort=False):
    q = q.sort_values(KEY).reset_index(drop=True)
    assert (q[KEY].values == base[KEY].values).all()          # same key order as IDX
    for m in METRICS:
        x = q[m].to_numpy(float); y = q[f"rand_{m}"].to_numpy(float)
        valid = ~np.isnan(x)
        xb = np.where(valid, x, 0.0)[IDX]; yb = np.where(valid, y, 0.0)[IDX]; vb = valid[IDX]
        nb = vb.sum(1)
        mx_b = xb.sum(1) / nb; my_b = yb.sum(1) / nb
        mx, my = float(np.nanmean(x)), float(np.nanmean(np.where(valid, y, np.nan)))
        rows.append({"condition": cond, "retriever": q.retriever[0], "search_space": q.search_space[0], "metric": m,
                     "n": int(valid.sum()), "mean": mx, "ci95_lo": ci(mx_b)[0], "ci95_hi": ci(mx_b)[1],
                     "median": float(np.nanmedian(x)),
                     "random_mean": my, "random_ci95_lo": ci(my_b)[0], "random_ci95_hi": ci(my_b)[1],
                     "lift_diff": mx - my, "lift_diff_ci95_lo": ci(mx_b - my_b)[0], "lift_diff_ci95_hi": ci(mx_b - my_b)[1],
                     "lift_ratio": mx / my if my else np.nan,
                     "lift_ratio_ci95_lo": ci(mx_b / my_b)[0], "lift_ratio_ci95_hi": ci(mx_b / my_b)[1],
                     "mean_N": float(q.N.mean()), "mean_G": float(q.G.mean()),
                     "n_questions_gold_outside_window": int(q.gold_outside_window.sum()),
                     "pct_questions_gold_outside_window": float(100 * q.gold_outside_window.mean()),
                     "n_questions_N_le_k": int((q.N <= int(m.split("@")[1])).sum()) if "@" in m else np.nan,
                     "bootstrap": "1000 resamples, numpy default_rng(42).integers(0,10742,(1000,10742)) over keys sorted by (person_id,qa_index); same idx for all conditions; 95% percentile"})
main = pd.DataFrame(rows)
main.to_csv(os.path.join(OUT, "table_retrieval_main.csv"), index=False)
log("table_retrieval_main.csv", main.shape)

# ---------------------------------------------------------------------------------------------- stratified tables
ORDERS = {"N_stratum": ["<=10", "11-30", "31-100", "101-300", ">300"], "G_stratum": ["1", "2", ">=3"]}
for var, fname in [("N_stratum", "by_N"), ("G_stratum", "by_G"), ("visit_group", "by_visit_group"),
                   ("reasoning_type", "by_reasoning_type"), ("difficulty", "by_difficulty")]:
    srows = []
    for cond, q in pq.groupby("condition", sort=False):
        levels = ORDERS.get(var, sorted(q[var].unique()))
        for lv in levels:
            s = q[q[var] == lv]
            if len(s) == 0:
                srows.append({"condition": cond, "retriever": q.retriever.iloc[0], "search_space": q.search_space.iloc[0],
                              "stratum_var": var, "stratum": lv, "n": 0}); continue
            for m in METRICS:
                x = s[m].to_numpy(float); y = s[f"rand_{m}"].to_numpy(float); v = ~np.isnan(x)
                mx = float(x[v].mean()) if v.any() else np.nan; my = float(y[v].mean()) if v.any() else np.nan
                k = int(m.split("@")[1]) if "@" in m else None
                nle = int((s.N <= k).sum()) if k else np.nan
                srows.append({"condition": cond, "retriever": s.retriever.iloc[0], "search_space": s.search_space.iloc[0],
                              "stratum_var": var, "stratum": lv, "n": int(len(s)), "metric": m, "n_valid": int(v.sum()),
                              "mean": mx, "median": float(np.median(x[v])) if v.any() else np.nan,
                              "random_mean": my, "lift_diff": mx - my, "lift_ratio": mx / my if my else np.nan,
                              "mean_N": float(s.N.mean()), "min_N": int(s.N.min()), "max_N": int(s.N.max()),
                              "mean_G": float(s.G.mean()),
                              "n_questions_N_le_k": nle, "pct_questions_N_le_k": (100 * nle / len(s)) if k else np.nan,
                              "flag_N_le_k": (bool(nle > 0) if k else np.nan),
                              "flag_all_N_le_k_trivial": (bool(nle == len(s)) if k else np.nan),
                              "n_questions_gold_outside_window": int(s.gold_outside_window.sum())})
    st = pd.DataFrame(srows)
    st.to_csv(os.path.join(OUT, f"table_retrieval_{fname}.csv"), index=False)
    log(f"table_retrieval_{fname}.csv", st.shape)
log("done")
