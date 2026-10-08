"""Build W/results/inputs/pairs_v14.parquet: one row per (person_id, qa_index, reader, condition), 268,550 rows.
Columns: keys, prediction (raw), empty_prediction, question, answer (gold), reasoning_type, difficulty, in_judge_keys,
gold_fully_survived (from shared/readers_v1.3/contexts/{condition}.parquet; asserted equal to the predictions column).
Run from project root with W/.venv/bin/python."""
import hashlib, json, os
import pandas as pd

import os as _os, sys as _sys  # release adaptation (paths/config only)
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))  # evaluation/ (release_config.py)
import release_config as _rc  # noqa: E402  reads evaluation/config.yaml
_rc.enter_workspace()  # inputs resolve relative to WORKSPACE_DIR (original project layout)
W = _rc.step_out("04_scoring")  # release: was the agent work directory
READERS = ["roberta_base_squad2", "biobert_v1_1_pubmed_squad_v2", "longformer_squadv2", "qwen2_5_7b", "qwen2_5_32b"]
CONDS = ["bm25@full_timeline", "medcpt@full_timeline", "nvembed_v2@full_timeline", "hybrid_rrf60@full_timeline", "oracle"]
gate = json.load(open(f"{W}/results/gate.json"))["checks"]
psha = {(p["reader"], p["condition"]): p["sha256"] for p in gate["predictions"]}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


ds = pd.read_csv("data/dataset.csv")
assert sha("data/dataset.csv") == gate["dataset_sha"]
jk = pd.read_csv("shared/canonical/judge_keys_1000.csv")
jset = set(zip(jk.person_id, jk.qa_index)); assert len(jset) == 1000
ds["in_judge_keys"] = [k in jset for k in zip(ds.person_id, ds.qa_index)]
assert ds.in_judge_keys.sum() == 1000
assert ds.answer.notna().all() and (ds.answer.str.strip() != "").all()
base = ds[["person_id", "qa_index", "question", "answer", "reasoning_type", "difficulty", "in_judge_keys"]]

frames = []
for c in CONDS:
    cp = f"shared/readers_v1.3/contexts/{c}.parquet"
    assert sha(cp) == gate["contexts"][c]["sha256"]
    cx = pd.read_parquet(cp, columns=["person_id", "qa_index", "gold_fully_survived"]).rename(columns={"gold_fully_survived": "gfs_ctx"})
    for r in READERS:
        p = f"shared/readers_v1.3/predictions/{r}/{c}.parquet"
        assert sha(p) == psha[(r, c)], p
        df = pd.read_parquet(p, columns=["person_id", "qa_index", "reader", "condition", "prediction", "gold_fully_survived"])
        df = df.merge(cx, on=["person_id", "qa_index"], how="left", validate="one_to_one")
        assert df.gfs_ctx.notna().all() and (df.gfs_ctx.astype(bool) == df.gold_fully_survived.astype(bool)).all(), (r, c)
        df = df.drop(columns=["gold_fully_survived"]).rename(columns={"gfs_ctx": "gold_fully_survived"})
        df = df.merge(base, on=["person_id", "qa_index"], how="left", validate="one_to_one")
        assert len(df) == 10742 and df.answer.notna().all()
        frames.append(df)
allp = pd.concat(frames, ignore_index=True)
allp["empty_prediction"] = allp.prediction.str.strip() == ""
allp["reader"] = pd.Categorical(allp.reader, READERS, ordered=True)
allp["condition"] = pd.Categorical(allp.condition, CONDS, ordered=True)
allp = allp.sort_values(["reader", "condition", "person_id", "qa_index"]).reset_index(drop=True)
allp["reader"] = allp.reader.astype(str); allp["condition"] = allp.condition.astype(str)
assert len(allp) == 268550 and not allp.duplicated(["person_id", "qa_index", "reader", "condition"]).any()
os.makedirs(f"{W}/results/inputs", exist_ok=True)
out = f"{W}/results/inputs/pairs_v14.parquet"
allp.to_parquet(out, index=False)
s = sha(out)
json.dump({"path": out, "sha256": s, "rows": len(allp), "empty_prediction_total": int(allp.empty_prediction.sum()),
           "empty_by_reader": allp.groupby("reader").empty_prediction.sum().astype(int).to_dict()},
          open(f"{W}/results/inputs/pairs_v14.meta.json", "w"), indent=1)
print(s, len(allp), allp.empty_prediction.sum(), os.path.getsize(out))
