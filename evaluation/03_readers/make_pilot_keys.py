"""Pilot keys: 50 questions, seed 42, stratified by reasoning_type (proportional, largest-remainder allocation;
within stratum random.Random(42).sample over keys sorted by (person_id, qa_index))."""
import json, math, os, random
import pandas as pd
import os as _os, sys as _sys  # release adaptation (paths/config only)
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))  # evaluation/ (release_config.py)
import release_config as _rc  # noqa: E402  reads evaluation/config.yaml
_rc.enter_workspace()  # inputs resolve relative to WORKSPACE_DIR (original project layout)
W = _rc.step_out("03_readers")  # release: was the agent work directory
df = pd.read_csv(os.path.join(_rc.DATA_DIR, "dataset.csv"))  # release: was W/../data
N = 50
cnt = df.reasoning_type.value_counts().sort_index()
quota = {k: N * v / len(df) for k, v in cnt.items()}
alloc = {k: math.floor(q) for k, q in quota.items()}
for k in sorted(quota, key=lambda k: (-(quota[k] - alloc[k]), k))[:N - sum(alloc.values())]:
    alloc[k] += 1
alloc = {k: max(v, 1) for k, v in alloc.items()}
while sum(alloc.values()) > N:  # keep every stratum >=1
    k = max(alloc, key=lambda k: (alloc[k], k)); alloc[k] -= 1
keys = []
for k in sorted(alloc):
    ks = sorted(map(tuple, df[df.reasoning_type == k][["person_id", "qa_index"]].astype(int).values.tolist()))
    keys += random.Random(42).sample(ks, alloc[k])
keys = sorted(keys)
assert len(keys) == N == len(set(keys))
json.dump([list(map(int, k)) for k in keys], open(os.path.join(W, "results", "pilot_keys.json"), "w"))
json.dump(alloc, open(os.path.join(W, "results", "pilot_alloc.json"), "w"), indent=1)
print(alloc, keys[:3])
