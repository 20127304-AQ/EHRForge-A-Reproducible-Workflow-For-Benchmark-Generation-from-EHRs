"""PART 2 pilot analysis. Run from project root after downloading /judge/<tag>/out:
  modal volume get <STEP4_VOLUME> /judge/<tag>/out W/results/pilot/<tag>/
  W/.venv/bin/python W/code/analyze_pilot.py --tag <tag>
Writes W/results/pilot/<tag>/pilot_summary.json and pilot_report.md (all numbers generated here)."""
import argparse, glob, json
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

import os as _os, sys as _sys  # release adaptation (paths/config only)
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))  # evaluation/ (release_config.py)
import release_config as _rc  # noqa: E402  reads evaluation/config.yaml
_rc.enter_workspace()  # inputs resolve relative to WORKSPACE_DIR (original project layout)
W = _rc.step_out("04_scoring")  # release: was the agent work directory
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); a = ap.parse_args()
base = f"{W}/results/pilot/{a.tag}"
items = pd.read_parquet(f"{base}/items.parquet")
out = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{base}/out/chunk_*.parquet"))], ignore_index=True)
assert not out.item_id.duplicated().any() and set(out.item_id) == set(items.item_id), "pilot incomplete / duplicates"
crit = yaml.safe_load(open("shared/canonical/protocol_v1.4.yaml"))["changelog_v1_4"]["pilot_design"]["pass_criteria"]

S = {"tag": a.tag, "n_calls": len(out)}
S["invalid_rate_first"] = float(out.invalid_first.mean()); S["invalid_rate_final"] = float(out.invalid_final.mean())
S["n_invalid_first"] = int(out.invalid_first.sum()); S["n_invalid_final"] = int(out.invalid_final.sum())
S["finish_reasons"] = out.finish_reason.value_counts().to_dict()
S["mean_output_tokens"] = float(out.n_output_tokens.mean())
smp = out[out.kind == "sample"]
S["hist_by_family"] = {f: {int(s): int(n) for s, n in g.score.value_counts().sort_index().items()} for f, g in smp.groupby("family")}
S["mean_by_family"] = smp.groupby("family").score.mean().round(4).to_dict()
S["mean_by_family_condition"] = {f"{f}|{c}": round(float(v), 4) for (f, c), v in smp.groupby(["family", "condition"]).score.mean().items()}
sp = {}
for r, g in smp.groupby("reader"):
    rho, p = spearmanr(g.score, g.n_words) if g.score.nunique() > 1 and g.n_words.nunique() > 1 else (float("nan"), float("nan"))
    sp[r] = {"n": len(g), "spearman_rho": None if np.isnan(rho) else round(float(rho), 4), "p": None if np.isnan(p) else round(float(p), 4),
             "mean_score": round(float(g.score.mean()), 4), "median_words": float(g.n_words.median())}
S["spearman_score_vs_words_by_reader"] = sp
pm = {k: float(out[out.kind == k].score.mean()) for k in ["probe_a", "probe_b", "probe_c", "probe_d"]}
S["probe_means"] = {k: round(v, 4) for k, v in pm.items()}
S["probe_hist"] = {k: {int(s): int(n) for s, n in out[out.kind == k].score.value_counts().sort_index().items()} for k in pm}
pa = out[out.kind == "probe_a"].set_index(["person_id", "qa_index"]).score
pd_ = out[out.kind == "probe_d"].set_index(["person_id", "qa_index"]).score
pc = out[out.kind == "probe_c"].set_index(["person_id", "qa_index"]).score
assert set(pa.index) == set(pd_.index) == set(pc.index)
S["probe_d_minus_a_paired_mean"] = round(float((pd_ - pa.loc[pd_.index]).mean()), 4)
S["probe_d_vs_a_counts"] = {"d>a": int((pd_ > pa.loc[pd_.index]).sum()), "d=a": int((pd_ == pa.loc[pd_.index]).sum()), "d<a": int((pd_ < pa.loc[pd_.index]).sum())}
S["probe_c_minus_a_paired_mean"] = round(float((pc - pa.loc[pc.index]).mean()), 4)
checks = {
    "invalid_rate_first<=0.02": S["invalid_rate_first"] <= crit["invalid_rate_max"],
    "probe_a_mean>=4.5": pm["probe_a"] >= crit["probe_a_mean_min"],
    "probe_b_mean<=2.0": pm["probe_b"] <= crit["probe_b_mean_max"],
    "probe_d_minus_a<0.3": S["probe_d_minus_a_paired_mean"] < crit["probe_d_minus_a_max_exclusive"],
    "probe_c_mean<probe_a_mean": pm["probe_c"] < pm["probe_a"],
}
S["criteria"] = checks
S["PASS"] = all(checks.values())
S["note"] = "invalid criterion applied to the FIRST-attempt invalid rate (stricter than the post-retry rate; both reported)"
json.dump(S, open(f"{base}/pilot_summary.json", "w"), indent=1)

L = [f"# Judge pilot `{a.tag}` (Prometheus 8x7B v2.0 @2db013b, greedy)", "", f"**Result: {'PASS' if S['PASS'] else 'FAIL'}**", "",
     "| criterion | value | pass |", "|---|---|---|",
     f"| invalid rate (first attempt) <= 2% | {S['invalid_rate_first']:.2%} ({S['n_invalid_first']}/{len(out)}); final {S['invalid_rate_final']:.2%} | {checks['invalid_rate_first<=0.02']} |",
     f"| (a) gold mean >= 4.5 | {pm['probe_a']:.3f} | {checks['probe_a_mean>=4.5']} |",
     f"| (b) other-question gold mean <= 2.0 | {pm['probe_b']:.3f} | {checks['probe_b_mean<=2.0']} |",
     f"| (d) - (a) paired mean < 0.3 | {S['probe_d_minus_a_paired_mean']:+.3f} (d>a {S['probe_d_vs_a_counts']['d>a']}, d=a {S['probe_d_vs_a_counts']['d=a']}, d<a {S['probe_d_vs_a_counts']['d<a']}) | {checks['probe_d_minus_a<0.3']} |",
     f"| (c) truncated mean < (a) mean | {pm['probe_c']:.3f} vs {pm['probe_a']:.3f} | {checks['probe_c_mean<probe_a_mean']} |", "",
     "## Probe score histograms", "", "| probe | 1 | 2 | 3 | 4 | 5 | mean |", "|---|---|---|---|---|---|---|"]
for k in pm:
    h = S["probe_hist"][k]; L.append(f"| {k} | " + " | ".join(str(h.get(s, 0)) for s in range(1, 6)) + f" | {pm[k]:.3f} |")
L += ["", "## Pilot sample: score histogram per reader family", "", "| family | n | 1 | 2 | 3 | 4 | 5 | mean |", "|---|---|---|---|---|---|---|---|"]
for f, h in S["hist_by_family"].items():
    L.append(f"| {f} | {sum(h.values())} | " + " | ".join(str(h.get(s, 0)) for s in range(1, 6)) + f" | {S['mean_by_family'][f]:.3f} |")
L += ["", "## Spearman(score, candidate length in words) within reader (pilot sample)", "", "| reader | n | rho | p | mean score | median words |", "|---|---|---|---|---|---|"]
for r, v in sp.items():
    L.append(f"| {r} | {v['n']} | {v['spearman_rho']} | {v['p']} | {v['mean_score']:.3f} | {v['median_words']:.0f} |")
L += ["", f"finish reasons: {S['finish_reasons']}; mean output tokens {S['mean_output_tokens']:.1f}"]
open(f"{base}/pilot_report.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
