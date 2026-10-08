"""Regenerate the fixed 1,000-key judge subset (shared/canonical/judge_keys_1000.csv) from data/dataset.csv.

The key file itself is NOT in the public release (it contains person_id). This script is the sampling block of
agent_step0_protocol_freeze_e3f56aca_workdir/code/part_d_judge_keys.py copied VERBATIM (seed 42, largest-remainder
proportional allocation over reasoning_type x difficulty with a floor of 1, numpy.random.default_rng(42) per cell
over key-sorted rows). Only the input/output paths are parameters, and the reporting part of the original script
(which compared against pre-fix judge outputs) is omitted. The result is asserted to be byte-identical to the
frozen file via its sha256 (protocol.yaml judge_keys sha256).

usage: python regenerate_judge_keys.py <path/to/dataset.csv> <out.csv>
"""
import hashlib
import sys

import numpy as np
import pandas as pd

EXPECTED_DATASET_SHA256 = "66a968330bea85bef75f847a6a0c4e8bb8352e0ac363c85c75436b0ee034847f"
EXPECTED_KEYS_SHA256 = "0aab500b691b2102fc80f96ea937a3e8e1f2abd3772d1a4c4d149d729c3d8fd0"

DATASET, OUT = sys.argv[1], sys.argv[2]
assert hashlib.sha256(open(DATASET, "rb").read()).hexdigest() == EXPECTED_DATASET_SHA256, "dataset.csv sha256 mismatch"

# ---- verbatim from part_d_judge_keys.py (paths parametrised) ----
N, SEED = 1000, 42
df = pd.read_csv(DATASET).sort_values(["person_id", "qa_index"]).reset_index(drop=True)
cells = df.groupby(["reasoning_type", "difficulty"]).size().sort_index()
cells = cells[cells > 0]
# Largest-remainder proportional allocation with floor of 1 per non-empty cell
quota = cells / cells.sum() * N
alloc = np.maximum(np.floor(quota).astype(int), 1)
rem = (quota - alloc)  # cells raised to the floor of 1 get a negative remainder (no double counting)
rem = rem.sort_values(ascending=False, kind="mergesort")
i = 0
while alloc.sum() < N:
    alloc[rem.index[i % len(rem)]] += 1; i += 1
while alloc.sum() > N:  # only if floors of 1 pushed us over: take from largest cells with smallest remainder
    for k in rem.index[::-1]:
        if alloc[k] > 1 and alloc.sum() > N:
            alloc[k] -= 1
alloc = np.minimum(alloc, cells)
assert alloc.sum() == N

rng = np.random.default_rng(SEED)
parts = []
for (rt, dif), k in alloc.items():
    sub = df[(df.reasoning_type == rt) & (df.difficulty == dif)]
    idx = rng.choice(len(sub), size=int(k), replace=False)
    parts.append(sub.iloc[np.sort(idx)])
keys = pd.concat(parts)[["person_id", "qa_index", "reasoning_type", "difficulty", "visit_group"]]
keys = keys.sort_values(["person_id", "qa_index"]).reset_index(drop=True)
assert len(keys) == N and not keys.duplicated(["person_id", "qa_index"]).any()
keys.to_csv(OUT, index=False, lineterminator="\n")
sha = hashlib.sha256(open(OUT, "rb").read()).hexdigest()
# ---- end verbatim ----

print("sha256", sha)
assert sha == EXPECTED_KEYS_SHA256, f"regenerated key file sha256 {sha} != frozen {EXPECTED_KEYS_SHA256}"
print("OK: byte-identical to the frozen judge_keys_1000.csv")
