"""PART 2 pilot items (design fixed in protocol_v1.4 changelog_v1_4.pilot_design). Run from project root:
  W/.venv/bin/python W/code/build_pilot.py --tag pilot1 [--exclude-tag pilot1]   (exclude = previous pilot keys, for a fresh rerun)
Writes W/results/pilot/<tag>/items.parquet (+ .meta.json)."""
import argparse, hashlib, json, math, os
import numpy as np
import pandas as pd
import yaml

import os as _os, sys as _sys  # release adaptation (paths/config only)
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))  # evaluation/ (release_config.py)
import release_config as _rc  # noqa: E402  reads evaluation/config.yaml
_rc.enter_workspace()  # inputs resolve relative to WORKSPACE_DIR (original project layout)
W = _rc.step_out("04_scoring")  # release: was the agent work directory
FAM = {"extractive": ["roberta_base_squad2", "biobert_v1_1_pubmed_squad_v2", "longformer_squadv2"],
       "qwen": ["qwen2_5_7b", "qwen2_5_32b"]}
CONDS = ["bm25@full_timeline", "medcpt@full_timeline", "nvembed_v2@full_timeline", "hybrid_rrf60@full_timeline", "oracle"]

ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--exclude-tag", default=None)
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--probe-seed", type=int, default=4242)
a = ap.parse_args()
proto = yaml.safe_load(open("shared/canonical/protocol_v1.4.yaml"))
FILLER = proto["changelog_v1_4"]["pilot_design"]["probes"]["filler"]
pairs = pd.read_parquet(f"{W}/results/inputs/pairs_v14.parquet")
nonjudge = pairs[~pairs.in_judge_keys]
excl = set()
if a.exclude_tag:
    prev = pd.read_parquet(f"{W}/results/pilot/{a.exclude_tag}/items.parquet")
    excl = set(zip(prev.person_id, prev.qa_index))
rng = np.random.default_rng(a.seed)
rows = []
order = 0
used_keys = set()
for fam, readers in FAM.items():
    for c in CONDS:
        cand = nonjudge[(nonjudge.condition == c) & nonjudge.reader.isin(readers)]
        keys = sorted(set(zip(cand.person_id, cand.qa_index)) - excl - used_keys)
        pick = rng.choice(len(keys), size=len(keys), replace=False)  # permutation; take first 20 that have a non-empty prediction for the drawn reader
        got = 0
        for j in pick:
            k = keys[j]
            r = readers[int(rng.integers(0, len(readers)))]
            row = cand[(cand.person_id == k[0]) & (cand.qa_index == k[1]) & (cand.reader == r)].iloc[0]
            if row.empty_prediction:
                continue
            rows.append(dict(item_id=f"sample|{fam}|{r}|{c}|{k[0]}|{k[1]}", kind="sample", family=fam, reader=r, condition=c,
                             person_id=int(k[0]), qa_index=int(k[1]), question=row.question, answer=row.answer,
                             prediction=row.prediction, order=order))
            order += 1; got += 1; used_keys.add(k)
            if got == 20:
                break
        assert got == 20
# ---- probes: 50 non-judge keys disjoint from the sample (and from excluded keys)
ds = pairs[(pairs.reader == "qwen2_5_7b") & (pairs.condition == "oracle") & (~pairs.in_judge_keys)][["person_id", "qa_index", "question", "answer"]]
ds = ds.sort_values(["person_id", "qa_index"]).reset_index(drop=True)
ok = [i for i, k in enumerate(zip(ds.person_id, ds.qa_index)) if k not in used_keys and k not in excl]
prng = np.random.default_rng(a.probe_seed)
sel = sorted(prng.choice(ok, size=50, replace=False).tolist())
P = ds.iloc[sel].reset_index(drop=True)
perm = prng.permutation(50)
donor = []
for i in range(50):  # donor = next item in the permutation cycle with a different person_id
    pos = int(np.where(perm == i)[0][0])
    for step in range(1, 50):
        j = int(perm[(pos + step) % 50])
        if P.person_id[j] != P.person_id[i]:
            donor.append(j); break
for i, row in P.iterrows():
    words = row.answer.split()
    trunc = " ".join(words[:max(1, len(words) // 3)])
    cands = {"probe_a": row.answer, "probe_b": P.answer[donor[i]], "probe_c": trunc, "probe_d": row.answer + " " + FILLER}
    for kind, cand in cands.items():
        assert cand.strip()
        rows.append(dict(item_id=f"{kind}|{row.person_id}|{row.qa_index}", kind=kind, family="probe", reader=kind, condition="probe",
                         person_id=int(row.person_id), qa_index=int(row.qa_index), question=row.question, answer=row.answer,
                         prediction=cand, order=order, donor_person_id=int(P.person_id[donor[i]]) if kind == "probe_b" else None,
                         donor_qa_index=int(P.qa_index[donor[i]]) if kind == "probe_b" else None))
        order += 1
items = pd.DataFrame(rows)
items["n_words"] = items.prediction.str.strip().str.split().str.len()
assert not items.item_id.duplicated().any()
assert not set(zip(items.person_id, items.qa_index)) & set(zip(pairs[pairs.in_judge_keys].person_id, pairs[pairs.in_judge_keys].qa_index))
pb = items[items.kind == "probe_b"]; assert (pb.donor_person_id != pb.person_id).all()
os.makedirs(f"{W}/results/pilot/{a.tag}", exist_ok=True)
out = f"{W}/results/pilot/{a.tag}/items.parquet"
items.to_parquet(out, index=False)
meta = {"tag": a.tag, "n_items": len(items), "by_kind": items.kind.value_counts().to_dict(),
        "sample_by_reader": items[items.kind == "sample"].reader.value_counts().to_dict(),
        "exclude_tag": a.exclude_tag, "n_excluded_keys": len(excl), "seed": a.seed, "probe_seed": a.probe_seed,
        "sha256": hashlib.sha256(open(out, "rb").read()).hexdigest()}
json.dump(meta, open(out.replace(".parquet", ".meta.json"), "w"), indent=1)
print(json.dumps(meta, indent=1))
print(items[items.kind == "probe_c"].prediction.head(3).tolist())
