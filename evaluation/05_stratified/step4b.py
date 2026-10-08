"""Step 4b: stratified QA tables + sensitivity analyses from EXISTING scored outputs (CPU only, no inference).
Run (release):  python evaluation/05_stratified/step4b.py   (inputs under WORKSPACE_DIR, see evaluation/config.yaml)
Bootstrap = protocol v1.4 scheme, copied verbatim from shared/scoring_v1.4/code/aggregate.py:
  for n in {10742, 1000}: FRESH np.random.default_rng(42); idx = rng.integers(0, n, size=(1000, n)) over keys sorted by
  (person_id, qa_index); count matrix C; stratum/subset means = ratio estimators recomputed inside each resample
  sum_k C[b,k] m_k x_k / sum_k C[b,k] m_k ; 95% percentile CI (np.percentile, linear). Same C for every run -> paired.
Scales: F1, EM, BERTScore and rates x100; judge mean 1..5."""
import hashlib, json, os, sys, time
import numpy as np
import pandas as pd

T0 = time.time()
import os as _os, sys as _sys  # release adaptation (paths/config only)
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))  # evaluation/ (release_config.py)
import release_config as _rc  # noqa: E402  reads evaluation/config.yaml
_rc.enter_workspace()  # inputs resolve relative to WORKSPACE_DIR (original project layout)
W = _rc.step_out("05_stratified")  # release: was the agent work directory
OUT = f"{W}/results"; os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, "shared/canonical")
import ehr_data as ED  # noqa: E402  (only _evidence used; corpus.json not present locally)

READERS = ["roberta_base_squad2", "biobert_v1_1_pubmed_squad_v2", "longformer_squadv2", "qwen2_5_7b", "qwen2_5_32b"]
CONDS = ["bm25@full_timeline", "medcpt@full_timeline", "nvembed_v2@full_timeline", "hybrid_rrf60@full_timeline", "oracle"]
RETR = CONDS[:4]; NV, OR = "nvembed_v2@full_timeline", "oracle"
RUNS = [(r, c) for r in READERS for c in CONDS]
RI = {rc: i for i, rc in enumerate(RUNS)}
K = ["person_id", "qa_index", "reader", "condition"]
B = 1000
MIN_J = 30           # judge cells reported only if n_judge >= 30
MIN_RT = 100         # Part 3(e) reasoning-type strata need n >= 100
FULL_METS = ["f1", "bertscore_roberta_large_F1", "bertscore_bio_clinicalbert_F1", "em"]
J_METS = ["judge_mean", "judge_correct_ge4"]
ALL_METS = FULL_METS + J_METS
log = lambda *a: print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ============================================================ 0. input sha256
def expected_shas():
    exp = {"data/dataset.csv": "66a968330bea85bef75f847a6a0c4e8bb8352e0ac363c85c75436b0ee034847f"}
    m = pd.read_csv("shared/scoring_v1.4/MANIFEST.csv")
    for p in ["per_row_scores.parquet", "judge_scores.parquet", "table_qa_main.csv", "table_by_gold_survival.csv",
              "table_common_subset.csv", "table_by_reasoning_type.csv", "code/aggregate.py"]:
        exp[f"shared/scoring_v1.4/{p}"] = m.set_index("path").loc[f"shared/scoring_v1.4/{p}", "sha256"]
    m = pd.read_csv("shared/readers_v1.3/MANIFEST.csv").set_index("path")
    for r in READERS:
        for c in CONDS:
            p = f"shared/readers_v1.3/predictions/{r}/{c}.parquet"; exp[p] = m.loc[p, "sha256"]
    for c in CONDS:
        p = f"shared/readers_v1.3/contexts/{c}.parquet"; exp[p] = m.loc[p, "sha256"]
    m = pd.read_csv("shared/retrieval_tables_v1.2/MANIFEST.csv").set_index("path")
    p = "shared/retrieval_tables_v1.2/per_question_metrics.parquet"; exp[p] = m.loc[p, "sha256"]
    # no manifest: record (canonical files) -- checked unchanged start vs end
    for p in ["shared/canonical/ehr_data.py", "shared/canonical/judge_keys_1000.csv"]:
        exp[p] = None
    return exp


EXP = expected_shas()
SHA_START = {p: sha(p) for p in EXP}
sha_start_ok = {p: (EXP[p] is None or SHA_START[p] == EXP[p]) for p in EXP}
assert all(sha_start_ok.values()), [p for p, v in sha_start_ok.items() if not v]
log("input sha256 OK:", len(EXP), "files")

# ============================================================ 1. load + key integrity
ds = pd.read_csv("data/dataset.csv")
assert len(ds) == 10742 and not ds.duplicated(["person_id", "qa_index"]).any()
keys_full = ds.sort_values(["person_id", "qa_index"])[["person_id", "qa_index"]].reset_index(drop=True)
jk = pd.read_csv("shared/canonical/judge_keys_1000.csv").sort_values(["person_id", "qa_index"]).reset_index(drop=True)
keys_j = jk[["person_id", "qa_index"]]
n_full, n_j = len(keys_full), len(keys_j)
assert n_full == 10742 and n_j == 1000 and not keys_j.duplicated().any()
KSET = set(zip(keys_full.person_id, keys_full.qa_index)); JSET = set(zip(keys_j.person_id, keys_j.qa_index))
assert JSET <= KSET

pr = pd.read_parquet("shared/scoring_v1.4/per_row_scores.parquet")
js = pd.read_parquet("shared/scoring_v1.4/judge_scores.parquet")
CHECKS = {}
CHECKS["per_row_rows_268550"] = len(pr) == 268550
CHECKS["per_row_no_duplicate_key_reader_condition"] = not pr.duplicated(K).any()
CHECKS["per_row_each_run_keys_equal_dataset"] = all(
    set(zip(g.person_id, g.qa_index)) == KSET and len(g) == n_full for _, g in pr.groupby(["reader", "condition"])) and \
    set(map(tuple, pr[["reader", "condition"]].drop_duplicates().to_numpy())) == set(RUNS)
CHECKS["judge_rows_25000"] = len(js) == 25000
CHECKS["judge_no_duplicate_key_reader_condition"] = not js.duplicated(K).any()
CHECKS["judge_each_run_keys_equal_judge_keys"] = all(
    set(zip(g.person_id, g.qa_index)) == JSET and len(g) == n_j for _, g in js.groupby(["reader", "condition"])) and \
    set(map(tuple, js[["reader", "condition"]].drop_duplicates().to_numpy())) == set(RUNS)
CHECKS["per_row_in_judge_keys_flag_matches_judge_keys"] = set(zip(*pr[pr.in_judge_keys][["person_id", "qa_index"]].drop_duplicates().to_numpy().T)) == JSET
assert all(CHECKS.values()), CHECKS


def matrix(df, col, keys):
    """R x n matrix aligned to RUNS and keys (sorted). (verbatim Step 4)"""
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
assert (C_full.sum(1) == n_full).all() and (C_j.sum(1) == n_j).all()
M = {"em": 100 * matrix(pr, "em", keys_full), "f1": 100 * matrix(pr, "f1", keys_full),
     "bertscore_roberta_large_F1": 100 * matrix(pr, "bs_roberta_large_F", keys_full),
     "bertscore_bio_clinicalbert_F1": 100 * matrix(pr, "bs_bio_clinicalbert_F", keys_full)}
sc = matrix(js, "score", keys_j)
M["judge_mean"] = sc; M["judge_correct_ge4"] = 100 * (sc >= 4)
prj = pr[pr.in_judge_keys]
MJSUB = {"f1": 100 * matrix(prj, "f1", keys_j)}  # judge-subset F1 (for table_qa_main reproduction)
IS_J = {m: m in J_METS for m in ALL_METS}
CMAT = {m: (C_j if IS_J[m] else C_full) for m in ALL_METS}
pos_j = pd.MultiIndex.from_frame(keys_full).get_indexer(pd.MultiIndex.from_frame(keys_j)); assert (pos_j >= 0).all()
CHECKS["each_key_reader_condition_used_exactly_once_in_matrices"] = all(
    M[m].shape == (25, (n_j if IS_J[m] else n_full)) and not np.isnan(M[m]).any() for m in ALL_METS)
log("matrices built")

# ============================================================ 2. key-level covariates (all joins on keys, asserted)
def join1(left, right, name):
    assert not right.duplicated(["person_id", "qa_index"]).any(), f"{name}: duplicate keys"
    out = left.merge(right, on=["person_id", "qa_index"], how="left", validate="one_to_one", indicator=True)
    assert (out._merge == "both").all(), f"{name}: missing keys"
    assert len(out) == len(left)
    return out.drop(columns="_merge")


cov = join1(keys_full, ds[["person_id", "qa_index", "difficulty", "visit_group", "reasoning_type", "evidence"]], "dataset")
cov["G"] = [len(sorted({int(e["original_visit_index"]) for e in ED._evidence({"evidence": ev})})) for ev in cov.evidence]
pq = pd.read_parquet("shared/retrieval_tables_v1.2/per_question_metrics.parquet", columns=["condition", "person_id", "qa_index", "G", "N"])
pqG = pq.groupby(["person_id", "qa_index"]).G.agg(["min", "max"]).reset_index()
assert (pqG["min"] == pqG["max"]).all()
cov = join1(cov, pqG.rename(columns={"min": "G_step2"}).drop(columns="max"), "per_question_metrics")
pqN = pq[pq.condition == "bm25@full_timeline"][["person_id", "qa_index", "N"]]
cov = join1(cov, pqN, "per_question_metrics N")

META_COLS = ["n_gold_visits", "n_selected_visits", "n_visits_reaching_model", "gold_fully_survived", "truncated"]
meta = {}
meta_identical = True; meta_vs_ctx = True; meta_gfs_vs_scores = True
for c in CONDS:
    ctx = join1(keys_full, pd.read_parquet(f"shared/readers_v1.3/contexts/{c}.parquet", columns=["person_id", "qa_index"] + META_COLS), f"ctx {c}")
    for r in READERS:
        p = join1(keys_full, pd.read_parquet(f"shared/readers_v1.3/predictions/{r}/{c}.parquet", columns=["person_id", "qa_index"] + META_COLS), f"pred {r} {c}")
        meta_vs_ctx &= all((p[k].to_numpy() == ctx[k].to_numpy()).all() for k in META_COLS)
        gs = M  # noqa
        s = join1(keys_full, pr[(pr.reader == r) & (pr.condition == c)][["person_id", "qa_index", "gold_fully_survived"]], f"scores {r} {c}")
        meta_gfs_vs_scores &= bool((s.gold_fully_survived.to_numpy() == p.gold_fully_survived.to_numpy()).all())
        if r == READERS[0]:
            meta[c] = p
        else:
            meta_identical &= all((p[k].to_numpy() == meta[c][k].to_numpy()).all() for k in META_COLS)
CHECKS["context_meta_identical_across_readers_within_condition"] = bool(meta_identical)
CHECKS["prediction_meta_equals_contexts_parquet"] = bool(meta_vs_ctx)
CHECKS["gold_fully_survived_predictions_equals_per_row_scores"] = bool(meta_gfs_vs_scores)
CHECKS["G_evidence_dedup_equals_step2_per_question_G"] = bool((cov.G == cov.G_step2).all())
CHECKS["G_equals_contexts_n_gold_visits_all_conditions"] = all((meta[c].n_gold_visits.to_numpy() == cov.G.to_numpy()).all() for c in CONDS)
assert all(CHECKS.values()), {k: v for k, v in CHECKS.items() if not v}
GS = {c: meta[c].gold_fully_survived.to_numpy().astype(float) for c in CONDS}
G = cov.G.to_numpy(); ORREACH = meta[OR].n_visits_reaching_model.to_numpy()
log("covariates joined; G distribution", dict(pd.Series(G).value_counts().sort_index()))


# ============================================================ 3. bootstrap core
def boot(Mx, mask, C):
    mask = np.broadcast_to(mask, Mx.shape).astype(np.float64)
    num = C @ (Mx * mask).T
    den = C @ mask.T
    with np.errstate(invalid="ignore", divide="ignore"):
        bt = num / den
        pt = (Mx * mask).sum(1) / mask.sum(1)
    return pt, bt, mask.sum(1).astype(int)


def ci(x):
    x = x[~np.isnan(x)]
    return (np.percentile(x, 2.5), np.percentile(x, 97.5)) if len(x) else (np.nan, np.nan)


def evaluate(mask_full, mets=ALL_METS):
    """mask over the 10,742 sorted keys -> {metric: (pt, bt, n)}; judge metrics use mask restricted to judge keys."""
    out = {}
    for m in mets:
        mk = mask_full[pos_j] if IS_J[m] else mask_full
        out[m] = boot(M[m], mk, CMAT[m])
    return out


def flag_for(m, n):
    if IS_J[m] and n < MIN_J:
        return "n<30"
    if n < MIN_J:
        return "small_n_descriptive"
    return ""


def run_rows(res, runs, **lab):
    rows = []
    for m, (pt, bt, n) in res.items():
        for rc in runs:
            i = RI[rc]; f = flag_for(m, n[i]); lo, hi = ci(bt[:, i]) if n[i] > 0 else (np.nan, np.nan)
            est = pt[i] if n[i] > 0 else np.nan
            if f == "n<30":
                est, lo, hi = np.nan, np.nan, np.nan
            rows.append(dict(**lab, row_type="run", reader=rc[0], condition=rc[1], metric=m, n=int(n[i]), estimate=est, ci_low=lo, ci_high=hi,
                             flag=f, n_nan_resamples=int(np.isnan(bt[:, i]).sum())))
    return rows


def diff_rows(res, pairs, **lab):
    """pairs: list of ((reader, condA), (reader, condB)) -> A minus B (paired: same mask, same resample)."""
    rows = []
    for m, (pt, bt, n) in res.items():
        for a, b in pairs:
            ia, ib = RI[a], RI[b]; assert n[ia] == n[ib]
            f = flag_for(m, n[ia])
            if n[ia] == 0:
                continue
            d = bt[:, ia] - bt[:, ib]; lo, hi = ci(d); est = pt[ia] - pt[ib]
            if f == "n<30":
                est, lo, hi = np.nan, np.nan, np.nan
            rows.append(dict(**lab, row_type="diff", reader=a[0], condition=f"{a[1]}_minus_{b[1]}", minuend_condition=a[1], subtrahend_condition=b[1],
                             metric=m, n=int(n[ia]), estimate=est, ci_low=lo, ci_high=hi, ci_excludes_0=bool(lo > 0 or hi < 0) if f != "n<30" else np.nan,
                             flag=f, n_nan_resamples=int(np.isnan(d).sum())))
    return rows


ONE = np.ones(n_full)
OR_NV = [((r, OR), (r, NV)) for r in READERS]
OR_R = [((r, OR), (r, c)) for r in READERS for c in RETR]

# ============================================================ 4. reproduce Step 4 tables (Part 4 checks)
REPRO = {}
ref = pd.read_csv("shared/scoring_v1.4/table_qa_main.csv")
res_all = evaluate(ONE)
mine = []
for m, (pt, bt, n) in res_all.items():
    sub = "judge_1000" if IS_J[m] else "full_10742"
    for rc, i in RI.items():
        lo, hi = ci(bt[:, i]); mine.append(dict(reader=rc[0], condition=rc[1], subset=sub, metric=m, est=pt[i], lo=lo, hi=hi, nn=n[i]))
pt, bt, n = boot(MJSUB["f1"], np.ones(n_j), C_j)
for rc, i in RI.items():
    lo, hi = ci(bt[:, i]); mine.append(dict(reader=rc[0], condition=rc[1], subset="judge_1000", metric="f1", est=pt[i], lo=lo, hi=hi, nn=n[i]))
cmp_ = ref.merge(pd.DataFrame(mine), on=["reader", "condition", "subset", "metric"], how="inner", validate="one_to_one")


def maxdiff(df, a, b):
    return float(np.nanmax(np.abs(df[a].to_numpy() - df[b].to_numpy())))


REPRO["table_qa_main"] = dict(rows_compared=len(cmp_), metrics=sorted(cmp_.metric.unique()), max_abs_diff_estimate=maxdiff(cmp_, "estimate", "est"),
                              max_abs_diff_ci_low=maxdiff(cmp_, "ci_low", "lo"), max_abs_diff_ci_high=maxdiff(cmp_, "ci_high", "hi"),
                              n_equal=bool((cmp_.n == cmp_.nn).all()))
hl = [("qwen2_5_32b", "oracle", "full_10742", "f1"), ("qwen2_5_32b", NV, "full_10742", "f1"), ("qwen2_5_32b", "oracle", "judge_1000", "judge_mean")]
REPRO["headline_CIs"] = []
for r, c, s, m in hl:
    x = cmp_[(cmp_.reader == r) & (cmp_.condition == c) & (cmp_.subset == s) & (cmp_.metric == m)].iloc[0]
    REPRO["headline_CIs"].append(dict(reader=r, condition=c, subset=s, metric=m, step4=[x.estimate, x.ci_low, x.ci_high], step4b=[x.est, x.lo, x.hi],
                                      within_0_1=bool(max(abs(x.ci_low - x.lo), abs(x.ci_high - x.hi)) < 0.1)))
f1_judge_exact = cmp_[(cmp_.metric.isin(["f1", "judge_mean"]))]
CHECKS["reproduce_step4_overall_F1_and_judge_mean_exact"] = bool(maxdiff(f1_judge_exact, "estimate", "est") < 1e-9)
CHECKS["reproduce_step4_table_qa_main_all_compared_rows_estimates_and_CIs_exact_1e-9"] = bool(
    max(REPRO["table_qa_main"]["max_abs_diff_estimate"], REPRO["table_qa_main"]["max_abs_diff_ci_low"], REPRO["table_qa_main"]["max_abs_diff_ci_high"]) < 1e-9)
CHECKS["headline_3_CIs_within_0.1"] = all(x["within_0_1"] for x in REPRO["headline_CIs"])

# reasoning-type table
ref = pd.read_csv("shared/scoring_v1.4/table_by_reasoning_type.csv")
rt_arr = cov.reasoning_type.to_numpy(); mine = []
for t in sorted(set(rt_arr)):
    res = evaluate((rt_arr == t).astype(float))
    for m, (pt, bt, n) in res.items():
        for rc, i in RI.items():
            lo, hi = ci(bt[:, i]); mine.append(dict(reasoning_type=t, reader=rc[0], condition=rc[1], subset="judge_1000" if IS_J[m] else "full_10742",
                                                   metric=m, est=pt[i], lo=lo, hi=hi, nn=n[i]))
c2 = ref.merge(pd.DataFrame(mine), on=["reasoning_type", "reader", "condition", "subset", "metric"], how="inner", validate="one_to_one")
REPRO["table_by_reasoning_type"] = dict(rows_compared=len(c2), max_abs_diff_estimate=maxdiff(c2, "estimate", "est"),
                                        max_abs_diff_ci=max(maxdiff(c2, "ci_low", "lo"), maxdiff(c2, "ci_high", "hi")), n_equal=bool((c2.n == c2.nn).all()))
CHECKS["reproduce_step4_table_by_reasoning_type_exact"] = REPRO["table_by_reasoning_type"]["max_abs_diff_estimate"] < 1e-9 and \
    REPRO["table_by_reasoning_type"]["max_abs_diff_ci"] < 1e-9 and REPRO["table_by_reasoning_type"]["n_equal"]
log("Step 4 reproduction:", json.dumps(REPRO, default=float)[:600])


# ============================================================ PART 1: QA stratification
VG_ORDER = ["2-10 visits", "11-100 visits", "101-1000 visits"]
DIFF_ORDER = ["easy", "medium", "hard"]
RT_ORDER = sorted(set(rt_arr))
strat_specs = [("difficulty", cov.difficulty.to_numpy(), DIFF_ORDER, RUNS),
               ("visit_group", cov.visit_group.to_numpy(), VG_ORDER, RUNS),
               ("reasoning_type", rt_arr, RT_ORDER, RUNS)]
cross_specs = [("difficulty_x_visit_group", cov.difficulty.to_numpy(), DIFF_ORDER, cov.visit_group.to_numpy(), VG_ORDER),
               ("reasoning_type_x_visit_group", rt_arr, RT_ORDER, cov.visit_group.to_numpy(), VG_ORDER)]
NVOR_RUNS = [(r, c) for r in READERS for c in (NV, OR)]
P1 = []
P1 += run_rows(res_all, RUNS, stratum_type="overall", stratum="all")
P1 += diff_rows(res_all, OR_NV, stratum_type="overall", stratum="all")
for st, arr, order, runs in strat_specs:
    assert set(arr) == set(order), (st, set(arr))
    for s in order:
        mk = (arr == s).astype(float); res = evaluate(mk)
        P1 += run_rows(res, runs, stratum_type=st, stratum=s, n_keys=int(mk.sum()), n_judge_keys=int(mk[pos_j].sum()))
        P1 += diff_rows(res, OR_NV, stratum_type=st, stratum=s, n_keys=int(mk.sum()), n_judge_keys=int(mk[pos_j].sum()))
for st, a1, o1, a2, o2 in cross_specs:
    for s1 in o1:
        for s2 in o2:
            mk = ((a1 == s1) & (a2 == s2)).astype(float)
            if mk.sum() == 0:
                P1.append(dict(stratum_type=st, stratum=f"{s1} | {s2}", n_keys=0, n_judge_keys=0, row_type="empty_cell", flag="n=0"))
                continue
            res = evaluate(mk)
            P1 += run_rows(res, NVOR_RUNS, stratum_type=st, stratum=f"{s1} | {s2}", n_keys=int(mk.sum()), n_judge_keys=int(mk[pos_j].sum()))
            P1 += diff_rows(res, OR_NV, stratum_type=st, stratum=f"{s1} | {s2}", n_keys=int(mk.sum()), n_judge_keys=int(mk[pos_j].sum()))
P1 = pd.DataFrame(P1)
P1.loc[P1.stratum_type == "overall", ["n_keys", "n_judge_keys"]] = [n_full, n_j]
log("Part 1 rows:", len(P1))


def fmt(e, lo, hi, d=2):
    if pd.isna(e):
        return ""
    return f"{e:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def wide(df):
    """one row per (stratum_type, stratum, reader, condition); metric cells 'est [lo, hi]' or 'n<30'."""
    d = df[df.row_type.isin(["run", "diff"])].copy()
    d["cell"] = [("n<30" if f == "n<30" else fmt(e, lo, hi, 3 if m == "judge_mean" else 2))
                 for f, e, lo, hi, m in zip(d.flag, d.estimate, d.ci_low, d.ci_high, d.metric)]
    idx = ["stratum_type", "stratum", "n_keys", "n_judge_keys", "reader", "condition"]
    w = d.pivot_table(index=idx, columns="metric", values="cell", aggfunc="first").reset_index()
    cols = idx + [m for m in ["f1", "bertscore_roberta_large_F1", "bertscore_bio_clinicalbert_F1", "judge_mean", "judge_correct_ge4", "em"] if m in w.columns]
    w = w[cols].rename(columns={"em": "em_appendix"})
    # stable order
    so = {s: i for i, s in enumerate(list(dict.fromkeys(df.stratum)))}
    co = {c: i for i, c in enumerate(CONDS + [f"{OR}_minus_{c}" for c in CONDS])}
    w["_s"] = w.stratum.map(so); w["_r"] = w.reader.map({r: i for i, r in enumerate(READERS)}); w["_c"] = w.condition.map(co)
    return w.sort_values(["_s", "_r", "_c"]).drop(columns=["_s", "_r", "_c"]).reset_index(drop=True)


P1W = {}
for st, fn in [("difficulty", "qa_by_difficulty.csv"), ("visit_group", "qa_by_visit_group.csv"), ("reasoning_type", "qa_by_reasoning_type.csv"),
               ("difficulty_x_visit_group", "qa_cross_difficulty_x_visitgroup.csv"), ("reasoning_type_x_visit_group", "qa_cross_reasoning_x_visitgroup.csv")]:
    sub = P1[P1.stratum_type.isin([st, "overall"])]
    w = wide(sub); P1W[st] = w; w.to_csv(f"{OUT}/{fn}", index=False)
    empties = P1[(P1.stratum_type == st) & (P1.row_type == "empty_cell")]
    if len(empties):
        empties[["stratum_type", "stratum", "n_keys", "n_judge_keys", "flag"]].to_csv(f"{OUT}/{fn}", mode="a", header=False, index=False)
LONG_COLS = ["stratum_type", "stratum", "reader", "condition", "metric", "n", "estimate", "ci_low", "ci_high", "flag", "row_type", "n_keys",
             "n_judge_keys", "ci_excludes_0", "n_nan_resamples"]
P1[P1.row_type != "empty_cell"][LONG_COLS].to_csv(f"{OUT}/qa_strata_long.csv", index=False)
P1[(P1.row_type != "empty_cell") & (P1.metric == "em")][LONG_COLS].to_csv(f"{OUT}/qa_strata_em_appendix.csv", index=False)
log("Part 1 written")

# ============================================================ PART 2: Oracle fairness
def subset_block(mask, label, extra_flag=""):
    res = evaluate(mask)
    lab = dict(subset=label, n_keys=int(mask.sum()), n_judge_keys=int(mask[pos_j].sum()))
    rows = run_rows(res, RUNS, **lab) + diff_rows(res, OR_R, **lab)
    for x in rows:
        if extra_flag:
            x["flag"] = (x["flag"] + ";" if x["flag"] else "") + extra_flag
    return rows, res


def gap_shift(res_a, res_b, mets, label):
    """(Oracle-R gap on ALL keys) minus (gap on subset): same resample, overlapping keys (descriptive)."""
    rows = []
    for m in mets:
        pa, ba, _ = res_a[m]; pb, bb, nb = res_b[m]
        if IS_J[m] and nb[0] < MIN_J:
            continue
        for (a, b) in OR_R:
            ia, ib = RI[a], RI[b]
            d = (ba[:, ia] - ba[:, ib]) - (bb[:, ia] - bb[:, ib]); lo, hi = ci(d)
            rows.append(dict(subset=label, row_type="gap_all_minus_gap_subset", reader=a[0], condition=f"{OR}_minus_{b[1]}", minuend_condition=OR,
                             subtrahend_condition=b[1], metric=m, n=int(nb[ia]), estimate=(pa[ia] - pa[ib]) - (pb[ia] - pb[ib]), ci_low=lo, ci_high=hi,
                             ci_excludes_0=bool(lo > 0 or hi < 0), flag=""))
    return rows


P2_METS = ["f1", "bertscore_roberta_large_F1", "bertscore_bio_clinicalbert_F1", "judge_mean", "judge_correct_ge4"]
le5 = (G <= 5).astype(float); gt5 = (G > 5).astype(float)
r_all, _ = subset_block(ONE, "all_keys")
r_le5, res_le5 = subset_block(le5, "G_le5")
r_gt5, _ = subset_block(gt5, "G_gt5", "DESCRIPTIVE_tiny_n")
P2 = pd.DataFrame(r_all + r_le5 + r_gt5 + gap_shift(res_all, res_le5, P2_METS, "G_le5"))
P2 = P2[P2.metric != "em"]
P2.to_csv(f"{OUT}/oracle_fairness_G_le5.csv", index=False)
rle5 = (ORREACH <= 5).astype(float); rgt5 = (ORREACH > 5).astype(float)
q_all, _ = subset_block(ONE, "all_keys")
q_le5, res_rle5 = subset_block(rle5, "oracle_reaching_le5")
q_gt5, _ = subset_block(rgt5, "oracle_reaching_gt5", "DESCRIPTIVE_tiny_n")
P2b = pd.DataFrame(q_all + q_le5 + q_gt5 + gap_shift(res_all, res_rle5, P2_METS, "oracle_reaching_le5"))
P2b = P2b[P2b.metric != "em"]
P2b.to_csv(f"{OUT}/oracle_fairness_reaching_le5.csv", index=False)
P2INFO = dict(n_keys=n_full, n_G_le5=int(le5.sum()), n_G_gt5=int(gt5.sum()), n_judge_G_le5=int(le5[pos_j].sum()), n_judge_G_gt5=int(gt5[pos_j].sum()),
              G_gt5_distribution={int(k): int(v) for k, v in pd.Series(G[G > 5]).value_counts().sort_index().items()},
              n_oracle_reaching_le5=int(rle5.sum()), n_oracle_reaching_gt5=int(rgt5.sum()), n_judge_oracle_reaching_gt5=int(rgt5[pos_j].sum()),
              oracle_reaching_equals_G_all_keys=bool((ORREACH == G).all()),
              reaching_le5_mask_identical_to_G_le5=bool((rle5 == le5).all()),
              retrieval_n_visits_reaching_model_max={c: int(meta[c].n_visits_reaching_model.max()) for c in RETR},
              retrieval_n_visits_reaching_model_mean={c: float(meta[c].n_visits_reaching_model.mean()) for c in RETR})
log("Part 2:", P2INFO)

# ============================================================ PART 3: gold survival
SURV_METS = ["f1", "bertscore_roberta_large_F1", "bertscore_bio_clinicalbert_F1", "em", "judge_mean", "judge_correct_ge4"]
# (a) within condition
P3a = []
for c in CONDS:
    rs = {}
    for sv in (True, False):
        mk = GS[c] if sv else 1 - GS[c]
        if mk.sum() == 0:
            continue
        rs[sv] = evaluate(mk)
        P3a += run_rows(rs[sv], [(r, c) for r in READERS], view="a_within_condition", gold_fully_survived=sv,
                        n_keys=int(mk.sum()), n_judge_keys=int(mk[pos_j].sum()))
    if len(rs) == 2:
        for m in SURV_METS:
            (p1, b1, n1), (p0, b0, n0) = rs[True][m], rs[False][m]
            for r in READERS:
                i = RI[(r, c)]
                if IS_J[m] and min(n1[i], n0[i]) < MIN_J:
                    P3a.append(dict(view="a_survived_minus_not", condition=c, reader=r, metric=m, row_type="diff_unpaired_groups", n=int(n1[i]),
                                    n_not=int(n0[i]), flag="n<30"))
                    continue
                d = b1[:, i] - b0[:, i]; lo, hi = ci(d)
                P3a.append(dict(view="a_survived_minus_not", condition=c, reader=r, metric=m, row_type="diff_unpaired_groups", n=int(n1[i]), n_not=int(n0[i]),
                                estimate=p1[i] - p0[i], ci_low=lo, ci_high=hi, ci_excludes_0=bool(lo > 0 or hi < 0), flag="",
                                n_nan_resamples=int(np.isnan(d).sum())))
P3a = pd.DataFrame(P3a)
P3a["note"] = ("survived/not-survived strata contain different keys in different conditions (not paired across conditions); "
               "survived-minus-not is a between-group difference inside each resample (same keys, different groups)")
P3a.to_csv(f"{OUT}/survival_within_condition.csv", index=False)
# reproduce Step 4 table_by_gold_survival (raw, unsuppressed values)
ref = pd.read_csv("shared/scoring_v1.4/table_by_gold_survival.csv")
mine = []
for c in CONDS:
    for sv in (True, False):
        mk = GS[c] if sv else 1 - GS[c]
        if mk.sum() == 0:
            continue
        res = evaluate(mk)
        for m, (pt, bt, n) in res.items():
            for r in READERS:
                i = RI[(r, c)]; lo, hi = ci(bt[:, i])
                mine.append(dict(condition=c, gold_fully_survived=sv, reader=r, subset="judge_1000" if IS_J[m] else "full_10742", metric=m,
                                 est=pt[i], lo=lo, hi=hi, nn=n[i]))
c3 = ref.merge(pd.DataFrame(mine), on=["condition", "gold_fully_survived", "reader", "subset", "metric"], how="inner", validate="one_to_one")
REPRO["table_by_gold_survival"] = dict(rows_ref=len(ref), rows_compared=len(c3), max_abs_diff_estimate=maxdiff(c3, "estimate", "est"),
                                       max_abs_diff_ci=max(maxdiff(c3, "ci_low", "lo"), maxdiff(c3, "ci_high", "hi")), n_equal=bool((c3.n == c3.nn).all()))
CHECKS["reproduce_step4_table_by_gold_survival_exact"] = REPRO["table_by_gold_survival"]["rows_compared"] == len(ref) and \
    REPRO["table_by_gold_survival"]["max_abs_diff_estimate"] < 1e-9 and REPRO["table_by_gold_survival"]["max_abs_diff_ci"] < 1e-9 and \
    REPRO["table_by_gold_survival"]["n_equal"]

# (b) common subset (+ (e) by reasoning type)
common = np.prod([GS[c] for c in CONDS], axis=0)
res_common = evaluate(common)
P3b = run_rows(res_common, RUNS, view="b_common_all5", stratum="all", n_keys=int(common.sum()), n_judge_keys=int(common[pos_j].sum())) + \
      diff_rows(res_common, OR_R, view="b_common_all5", stratum="all", n_keys=int(common.sum()), n_judge_keys=int(common[pos_j].sum()))
for t in RT_ORDER:
    mk = common * (rt_arr == t)
    if mk.sum() < MIN_RT:
        P3b.append(dict(view="e_common_all5_by_reasoning_type", stratum=t, n_keys=int(mk.sum()), n_judge_keys=int(mk[pos_j].sum()),
                        row_type="skipped", flag=f"n<{MIN_RT}"))
        continue
    res = evaluate(mk)
    lab = dict(view="e_common_all5_by_reasoning_type", stratum=t, n_keys=int(mk.sum()), n_judge_keys=int(mk[pos_j].sum()))
    P3b += run_rows(res, RUNS, **lab) + diff_rows(res, OR_R, **lab)
P3b = pd.DataFrame(P3b)
P3b.to_csv(f"{OUT}/survival_common_all5.csv", index=False)
ref = pd.read_csv("shared/scoring_v1.4/table_common_subset.csv")
mine = []
for m, (pt, bt, n) in res_common.items():
    sub = "judge_1000" if IS_J[m] else "full_10742"
    for rc, i in RI.items():
        lo, hi = ci(bt[:, i]); mine.append(dict(view="ii_common_subset_run", reader=rc[0], condition=rc[1], subtrahend_condition="", subset=sub, metric=m, est=pt[i], lo=lo, hi=hi, nn=n[i]))
    for r in READERS:
        for c in RETR:
            d = bt[:, RI[(r, OR)]] - bt[:, RI[(r, c)]]; lo, hi = ci(d)
            mine.append(dict(view="ii_common_subset_oracle_minus_retrieval", reader=r, condition=OR, subtrahend_condition=c, subset=sub, metric=m,
                             est=pt[RI[(r, OR)]] - pt[RI[(r, c)]], lo=lo, hi=hi, nn=n[RI[(r, c)]]))
ref["subtrahend_condition"] = ref.subtrahend_condition.fillna("")
c4 = ref.merge(pd.DataFrame(mine), on=["view", "reader", "condition", "subtrahend_condition", "subset", "metric"], how="inner", validate="one_to_one")
REPRO["table_common_subset"] = dict(rows_ref=len(ref), rows_compared=len(c4), metrics_compared=sorted(c4.metric.unique()),
                                    max_abs_diff_estimate=maxdiff(c4, "estimate", "est"),
                                    max_abs_diff_ci=max(maxdiff(c4, "ci_low", "lo"), maxdiff(c4, "ci_high", "hi")), n_equal=bool((c4.n == c4.nn).all()),
                                    n_common_full=int(common.sum()), n_common_judge=int(common[pos_j].sum()))
CHECKS["reproduce_step4_table_common_subset_exact"] = REPRO["table_common_subset"]["max_abs_diff_estimate"] < 1e-9 and \
    REPRO["table_common_subset"]["max_abs_diff_ci"] < 1e-9 and REPRO["table_common_subset"]["n_equal"]
CHECKS["common_subset_n_1277_judge_123"] = int(common.sum()) == 1277 and int(common[pos_j].sum()) == 123

# (c) pairwise common subsets (+ (e))
P3c = []
PW_INFO = {}
for c in RETR:
    mk0 = GS[c] * GS[OR]
    PW_INFO[c] = dict(n_keys=int(mk0.sum()), n_judge_keys=int(mk0[pos_j].sum()), n_survived_in_R=int(GS[c].sum()),
                      n_survived_in_oracle=int(GS[OR].sum()), R_survival_subset_of_oracle_survival=bool((GS[c] <= GS[OR]).all()))
    pairs = [((r, OR), (r, c)) for r in READERS]
    runs = [(r, cc) for r in READERS for cc in (c, OR)]
    for t in ["all"] + RT_ORDER:
        mk = mk0 if t == "all" else mk0 * (rt_arr == t)
        lab = dict(view="c_pairwise_common" if t == "all" else "e_pairwise_common_by_reasoning_type", retrieval_condition=c, stratum=t,
                   n_keys=int(mk.sum()), n_judge_keys=int(mk[pos_j].sum()))
        if t != "all" and mk.sum() < MIN_RT:
            P3c.append(dict(**lab, row_type="skipped", flag=f"n<{MIN_RT}")); continue
        res = evaluate(mk)
        P3c += run_rows(res, runs, **lab) + diff_rows(res, pairs, **lab)
P3c = pd.DataFrame(P3c)
P3c.to_csv(f"{OUT}/survival_pairwise_common.csv", index=False)
log("Part 3 a-c,e done", PW_INFO)

# (d) attribution of Oracle-minus-R gap to survived / not-survived keys in R
P3d = []
for m in ["f1", "bertscore_roberta_large_F1", "judge_mean"]:
    C = CMAT[m]
    for c in RETR:
        s = GS[c][pos_j] if IS_J[m] else GS[c]; ns = 1 - s
        for r in READERS:
            d = M[m][RI[(r, OR)]] - M[m][RI[(r, c)]]
            tot = C.sum(1)
            Q = {"share_survived": C @ s / tot, "share_not_survived": C @ ns / tot, "gap_total": C @ d / tot}
            with np.errstate(invalid="ignore", divide="ignore"):
                Q["gap_survived"] = C @ (d * s) / (C @ s)
                Q["gap_not_survived"] = C @ (d * ns) / (C @ ns)
                Q["contrib_survived"] = Q["share_survived"] * Q["gap_survived"]
                Q["contrib_not_survived"] = Q["share_not_survived"] * Q["gap_not_survived"]
                Q["frac_gap_from_not_survived"] = Q["contrib_not_survived"] / Q["gap_total"]
            P = {"share_survived": s.mean(), "share_not_survived": ns.mean(), "gap_total": d.mean(),
                 "gap_survived": d[s == 1].mean() if s.sum() else np.nan, "gap_not_survived": d[ns == 1].mean() if ns.sum() else np.nan}
            P["contrib_survived"] = P["share_survived"] * P["gap_survived"]; P["contrib_not_survived"] = P["share_not_survived"] * P["gap_not_survived"]
            P["frac_gap_from_not_survived"] = P["contrib_not_survived"] / P["gap_total"]
            ident = abs(P["contrib_survived"] + P["contrib_not_survived"] - P["gap_total"])
            assert ident < 1e-9, (m, c, r, ident)
            lo_t, hi_t = ci(Q["gap_total"])
            for q in Q:
                lo, hi = ci(Q[q])
                P3d.append(dict(metric=m, subset="judge_1000" if IS_J[m] else "full_10742", reader=r, retrieval_condition=c, quantity=q,
                                estimate=P[q] * (100 if q.startswith(("share", "frac")) else 1), ci_low=lo * (100 if q.startswith(("share", "frac")) else 1),
                                ci_high=hi * (100 if q.startswith(("share", "frac")) else 1), n_keys=len(s), n_survived=int(s.sum()), n_not_survived=int(ns.sum()),
                                identity_abs_err=ident, gap_total_ci_excludes_0=bool(lo_t > 0 or hi_t < 0),
                                flag=("judge_not_survived_n<30" if IS_J[m] and ns.sum() < MIN_J else "") +
                                     ("" if (lo_t > 0 or hi_t < 0) else ";gap_total_CI_includes_0_fraction_unstable")))
P3d = pd.DataFrame(P3d)
P3d["note"] = "gap_total = share_survived*gap_survived + share_not_survived*gap_not_survived (survival defined in retrieval condition R); share_* and frac_* in %"
P3d.to_csv(f"{OUT}/survival_gap_attribution.csv", index=False)
CHECKS["attribution_identity_holds_all_cells"] = bool(P3d.identity_abs_err.max() < 1e-9)
log("Part 3d done")


# ============================================================ PART 4: final checks
# every output row uses each (key, reader, condition) at most once per estimate by construction (matrix cells); verify
# that every run row in the main tables has n consistent with its mask.
CHECKS["part1_overall_rows_equal_step4_all_runs"] = bool(
    (P1[(P1.stratum_type == "overall") & (P1.row_type == "run")].groupby("metric").size() == 25).all())
CHECKS["part1_strata_partition_keys"] = all(
    P1[(P1.stratum_type == st) & (P1.row_type == "run") & (P1.metric == "f1") & (P1.reader == READERS[0]) & (P1.condition == OR)].n.sum() == n_full
    for st in ["difficulty", "visit_group", "reasoning_type", "difficulty_x_visit_group", "reasoning_type_x_visit_group"])
CHECKS["part1_judge_strata_partition_judge_keys"] = all(
    P1[(P1.stratum_type == st) & (P1.row_type == "run") & (P1.metric == "judge_mean") & (P1.reader == READERS[0]) & (P1.condition == OR)].n.sum() == n_j
    for st in ["difficulty", "visit_group", "reasoning_type"])
CHECKS["part1_no_judge_estimate_printed_where_n_judge_lt_30"] = bool(
    P1[(P1.metric.isin(J_METS)) & (P1.n < MIN_J) & (P1.row_type != "empty_cell")].estimate.isna().all())
CHECKS["part2_G_partition"] = P2INFO["n_G_le5"] + P2INFO["n_G_gt5"] == n_full
CHECKS["part3_survival_partition"] = all(
    P3a[(P3a.view == "a_within_condition") & (P3a.condition == c) & (P3a.reader == READERS[0]) & (P3a.metric == "f1")].n.sum() == n_full for c in CONDS)

SHA_END = {p: sha(p) for p in EXP}
CHECKS["input_sha256_unchanged_start_vs_end"] = SHA_START == SHA_END
CHECKS["input_sha256_match_manifests_at_end"] = all(EXP[p] is None or SHA_END[p] == EXP[p] for p in EXP)
prot = {}
for mf, col in [("shared/scoring_v1.4/MANIFEST.csv", "sha256"), ("shared/readers_v1.3/MANIFEST.csv", "sha256"),
                ("shared/retrieval_tables_v1.2/MANIFEST.csv", "sha256")]:
    m = pd.read_csv(mf)
    ok = [(sha(p) == s) for p, s in zip(m.path, m[col]) if os.path.exists(p)]
    prot[mf] = dict(files_checked=len(ok), all_match=bool(all(ok)), missing=int(sum(not os.path.exists(p) for p in m.path)))
CHECKS["protected_dirs_all_manifest_files_unchanged"] = all(v["all_match"] for v in prot.values())
CHECKS["dataset_sha256_expected"] = SHA_END["data/dataset.csv"] == "66a968330bea85bef75f847a6a0c4e8bb8352e0ac363c85c75436b0ee034847f"

out = dict(checks={k: bool(v) for k, v in CHECKS.items()}, all_pass=bool(all(CHECKS.values())), reproduction=REPRO,
           part2_info=P2INFO, pairwise_common_info=PW_INFO, protected_manifest_recheck=prot,
           input_sha256={p: dict(expected=EXP[p], start=SHA_START[p], end=SHA_END[p]) for p in EXP},
           G_note=("data/corpus.json is not present locally, so ehr_data.gold_visits(row) (which validates evidence datetimes against the corpus) "
                   "could not be called; G was computed with the identical dedup rule (ehr_data._evidence -> set(original_visit_index)) and "
                   "asserted equal to Step 2 per_question_metrics.G (computed by gold_visits with corpus validation) and to Step 3 "
                   "contexts n_gold_visits for every key and every condition."),
           bootstrap=dict(recipe="verbatim shared/scoring_v1.4/code/aggregate.py counts(): fresh np.random.default_rng(42) per n; "
                          "rng.integers(0,n,size=(1000,n)); keys sorted by (person_id, qa_index); ratio estimators; 95% percentile CI",
                          n_full=n_full, n_judge=n_j, B=B, stored_index_matrices_found=False),
           thresholds=dict(judge_min_n=MIN_J, reasoning_type_min_n_part3e=MIN_RT), runtime_s=time.time() - T0)
json.dump(out, open(f"{OUT}/checks.json", "w"), indent=1, default=float)
log("ALL PASS" if out["all_pass"] else "SOME CHECKS FAILED", {k: v for k, v in CHECKS.items() if not v})
