"""Step 4 GATE (read-only). Run from project root:
   W/.venv/bin/python W/code/gate.py
Writes W/results/gate.json; raises AssertionError on any failure."""
import hashlib, json, os, re, subprocess, sys, datetime
import pandas as pd

import os as _os, sys as _sys  # release adaptation (paths/config only)
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))  # evaluation/ (release_config.py)
import release_config as _rc  # noqa: E402  reads evaluation/config.yaml
_rc.enter_workspace()  # inputs resolve relative to WORKSPACE_DIR (original project layout)
ROOT = os.getcwd()
W = _rc.step_out("04_scoring")  # release: was the agent work directory
READERS = ["roberta_base_squad2", "biobert_v1_1_pubmed_squad_v2", "longformer_squadv2", "qwen2_5_7b", "qwen2_5_32b"]
CONDS = ["bm25@full_timeline", "medcpt@full_timeline", "nvembed_v2@full_timeline", "hybrid_rrf60@full_timeline", "oracle"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


out = {"created_utc": datetime.datetime.utcnow().isoformat() + "Z", "checks": {}}
C = out["checks"]

assert os.path.exists("data/corpus.json"), "data/corpus.json MISSING -> STOP (do not re-download)"
C["dataset_sha"] = sha("data/dataset.csv")
assert C["dataset_sha"] == "66a968330bea85bef75f847a6a0c4e8bb8352e0ac363c85c75436b0ee034847f"
C["corpus_sha"] = sha("data/corpus.json"); C["corpus_bytes"] = os.path.getsize("data/corpus.json")
assert C["corpus_sha"] == "b77d4b75ff4a8bd6ce8c9138a718ae04c0d76bf3066218b762a2c887d1a5d0a2" and C["corpus_bytes"] == 162124534
C["judge_keys_sha"] = sha("shared/canonical/judge_keys_1000.csv")
assert C["judge_keys_sha"] == "0aab500b691b2102fc80f96ea937a3e8e1f2abd3772d1a4c4d149d729c3d8fd0"
# protocol v1.3: full sha read from step3_report.md
rep = open("shared/readers_v1.3/step3_report.md").read()
m = re.search(r"sha256 (8db0ab6c[0-9a-f]{56})", rep)
assert m, "full v1.3 sha not found in step3_report.md"
C["protocol_v13_sha_expected_from_step3_report"] = m.group(1)
C["protocol_v13_sha"] = sha("shared/canonical/protocol_v1.3.yaml")
assert C["protocol_v13_sha"] == m.group(1) and C["protocol_v13_sha"].startswith("8db0ab6c")
assert open("shared/canonical/protocol_v1.3.yaml.sha256").read().split()[0] == m.group(1)
# other protocol files recorded (must stay untouched)
C["protocol_shas_untouched"] = {f: sha(f"shared/canonical/{f}") for f in
                                ["protocol.yaml", "protocol_v1.1.yaml", "protocol_v1.2.yaml", "protocol_v1.3.yaml"]}

# predictions vs MANIFEST
man = pd.read_csv("shared/readers_v1.3/MANIFEST.csv")
ds = pd.read_csv("data/dataset.csv")
dkeys = set(zip(ds.person_id.astype(int), ds.qa_index.astype(int)))
assert len(dkeys) == len(ds) == 10742
C["predictions"] = []
n_files = 0
for r in READERS:
    for c in CONDS:
        p = f"shared/readers_v1.3/predictions/{r}/{c}.parquet"
        row = man[man.path == p]
        assert len(row) == 1, p
        s = sha(p)
        assert s == row.sha256.iloc[0], (p, s)
        assert os.path.getsize(p) == int(row.bytes.iloc[0])
        df = pd.read_parquet(p, columns=["person_id", "qa_index", "reader", "condition", "prediction", "gold_fully_survived"])
        keys = list(zip(df.person_id.astype(int), df.qa_index.astype(int)))
        assert len(df) == 10742 and len(set(keys)) == 10742 and set(keys) == dkeys, p
        assert (df.reader == r).all() and (df.condition == c).all()
        assert df.prediction.notna().all()
        st = df.prediction.str.strip()
        C["predictions"].append({"reader": r, "condition": c, "sha256": s, "rows": len(df),
                                 "exact_empty": int((df.prediction == "").sum()),
                                 "empty_after_strip": int((st == "").sum())})
        n_files += 1
assert n_files == 25
C["n_prediction_files_ok"] = n_files

# contexts: gold_fully_survived present, join check vs predictions (equality)
C["contexts"] = {}
for c in CONDS:
    p = f"shared/readers_v1.3/contexts/{c}.parquet"
    cols = pd.read_parquet(p).columns.tolist() if False else None
    import pyarrow.parquet as pq
    cols = pq.read_schema(p).names
    assert "gold_fully_survived" in cols, (p, cols)
    cx = pd.read_parquet(p, columns=["person_id", "qa_index", "gold_fully_survived"])
    assert len(cx) == 10742 and not cx.duplicated(["person_id", "qa_index"]).any()
    C["contexts"][c] = {"sha256": sha(p), "columns": cols, "survived_rate": float(cx.gold_fully_survived.mean())}

# ehr_data import + tests
sys.path.insert(0, "shared/canonical")
import ehr_data  # noqa
C["ehr_data_sha"] = sha("shared/canonical/ehr_data.py")
res = subprocess.run([sys.executable, "-m", "pytest", "shared/canonical/test_ehr_data.py", "-q"], capture_output=True, text=True)
open(f"{W}/logs/pytest_canonical.log", "w").write(res.stdout + res.stderr)
mm = re.search(r"(\d+) passed", res.stdout)
C["pytest_passed"] = int(mm.group(1)) if mm else 0
C["pytest_tail"] = res.stdout.strip().splitlines()[-1]
assert res.returncode == 0 and C["pytest_passed"] == 44, C["pytest_tail"]

out["status"] = "PASS"
json.dump(out, open(f"{W}/results/gate.json", "w"), indent=1)
print(json.dumps({k: v for k, v in C.items() if k not in ("predictions", "contexts")}, indent=1))
print(pd.DataFrame(C["predictions"])[["reader", "condition", "exact_empty", "empty_after_strip"]].to_string())
print({c: (v["survived_rate"]) for c, v in C["contexts"].items()})
print("GATE PASS")
