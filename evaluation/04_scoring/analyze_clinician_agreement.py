"""Judge-vs-clinician agreement for the clinician validation sheet (protocol v1.4 scoring.judge_validation).
Usage (after clinicians fill clinician1_score / clinician2_score in the xlsx):
  python analyze_clinician_agreement.py --sheet shared/scoring_v1.4/clinician_validation_100.xlsx \
         --key shared/scoring_v1.4/clinician_validation_100_key.csv --out clinician_agreement.json
Requires: pandas, numpy, scipy, openpyxl. No sklearn.
Computes, for judge vs each filled clinician column (and vs the clinician mean, rounded half-up, if both are filled):
  quadratic-weighted kappa and Spearman rho on the 1-5 scale; Cohen's kappa and accuracy on binary score>=4;
  inter-clinician quadratic-weighted kappa, binary kappa and Spearman if both columns are filled;
  bootstrap 95% percentile CIs (1000 resamples over items, numpy.random.default_rng(42)).
PRE-REGISTERED DECISION RULE: if the binary (>=4) Cohen's kappa between the judge and clinician labels is < 0.6, the judge is
reported only as a SECONDARY signal. Operationalisation: the rule is evaluated for every available clinician column; it triggers
if ANY judge-vs-clinician binary kappa point estimate is < 0.6 (conservative)."""
import argparse, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def cohen_kappa(a, b, labels, weights=None):
    a = np.asarray(a); b = np.asarray(b); L = len(labels); ix = {l: i for i, l in enumerate(labels)}
    O = np.zeros((L, L))
    for x, y in zip(a, b):
        O[ix[x], ix[y]] += 1
    if O.sum() == 0:
        return np.nan
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    i, j = np.indices((L, L))
    Wt = ((i - j) ** 2) / (L - 1) ** 2 if weights == "quadratic" else (i != j).astype(float)
    den = (Wt * E).sum()
    return np.nan if den == 0 else 1 - (Wt * O).sum() / den


def stats(x, y):
    x = np.asarray(x, int); y = np.asarray(y, int)
    bx, by = (x >= 4).astype(int), (y >= 4).astype(int)
    rho = spearmanr(x, y).statistic if len(set(x)) > 1 and len(set(y)) > 1 else np.nan
    return {"qwk_1to5": cohen_kappa(x, y, [1, 2, 3, 4, 5], "quadratic"), "spearman_1to5": rho,
            "cohen_kappa_binary_ge4": cohen_kappa(bx, by, [0, 1]), "accuracy_binary_ge4": float((bx == by).mean())}


def with_ci(x, y, B=1000):
    pt = stats(x, y); rng = np.random.default_rng(42); n = len(x); bs = {k: [] for k in pt}
    x = np.asarray(x); y = np.asarray(y)
    for _ in range(B):
        idx = rng.integers(0, n, n); s = stats(x[idx], y[idx])
        for k, v in s.items():
            bs[k].append(v)
    out = {}
    for k, v in pt.items():
        arr = np.array(bs[k], float); arr = arr[~np.isnan(arr)]
        out[k] = {"estimate": None if v is None or np.isnan(v) else float(v),
                  "ci_low": float(np.percentile(arr, 2.5)) if len(arr) else None, "ci_high": float(np.percentile(arr, 97.5)) if len(arr) else None,
                  "n_valid_resamples": int(len(arr))}
    out["n_items"] = int(n)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="shared/scoring_v1.4/clinician_validation_100.xlsx")
    ap.add_argument("--key", default="shared/scoring_v1.4/clinician_validation_100_key.csv")
    ap.add_argument("--out", default="clinician_agreement.json")
    a = ap.parse_args()
    sh = pd.read_excel(a.sheet, sheet_name="items")
    key = pd.read_csv(a.key)
    df = sh.merge(key, on="item_id", how="inner", validate="one_to_one")
    assert len(df) == len(sh) == len(key), "sheet/key mismatch"
    res = {"n_items": len(df), "comparisons": {}}
    cols = []
    for c in ["clinician1_score", "clinician2_score"]:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().any():
            bad = v.notna() & ~v.isin([1, 2, 3, 4, 5])
            assert not bad.any(), f"{c}: values must be integers 1-5"
            cols.append(c)
    if not cols:
        print("No clinician labels filled yet; nothing to compute."); return
    for c in cols:
        m = pd.to_numeric(df[c], errors="coerce").notna()
        res["comparisons"][f"judge_vs_{c}"] = with_ci(df.judge_score[m].to_numpy(), pd.to_numeric(df[c])[m].astype(int).to_numpy())
    if len(cols) == 2:
        m = pd.to_numeric(df.clinician1_score, errors="coerce").notna() & pd.to_numeric(df.clinician2_score, errors="coerce").notna()
        c1 = pd.to_numeric(df.clinician1_score)[m].astype(int).to_numpy(); c2 = pd.to_numeric(df.clinician2_score)[m].astype(int).to_numpy()
        res["comparisons"]["inter_clinician"] = with_ci(c1, c2)
        mean_lab = np.floor((c1 + c2) / 2 + 0.5).astype(int)
        res["comparisons"]["judge_vs_clinician_mean"] = with_ci(df.judge_score[m].to_numpy(), mean_lab)
    kap = {k: v["cohen_kappa_binary_ge4"]["estimate"] for k, v in res["comparisons"].items() if k.startswith("judge_vs_clinician")}
    trig = any(v is None or v < 0.6 for v in kap.values())
    res["decision_rule"] = {"rule": "binary (>=4) Cohen's kappa judge vs clinician < 0.6 -> judge reported only as a SECONDARY signal",
                            "binary_kappas": kap, "triggered": bool(trig),
                            "judge_role": "SECONDARY signal only" if trig else "may be reported as a headline metric"}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
