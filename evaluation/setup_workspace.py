"""Populate <WORKSPACE_DIR>/shared/canonical/ from this release so that the evaluation scripts (which resolve
inputs relative to the original project root) run unchanged. Added for the public release; no scientific logic.

Copies (and sha256-verifies against evaluation/protocol/CANONICAL_SHA256.csv):
  evaluation/protocol/protocol*.yaml (+ .sha256 sidecars)  -> shared/canonical/
  evaluation/common/ehr_data.py, sampled_window.py          -> shared/canonical/
If data/dataset.csv is present and shared/canonical/judge_keys_1000.csv is not, the key file is regenerated with
evaluation/protocol/regenerate_judge_keys.py (byte-identical, sha-asserted).

usage: python evaluation/setup_workspace.py
"""
import csv
import hashlib
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import release_config as rc  # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    canon = os.path.join(rc.WORKSPACE, "shared", "canonical")
    os.makedirs(canon, exist_ok=True)
    os.makedirs(rc.DATA_DIR, exist_ok=True)
    n = 0
    with open(os.path.join(HERE, "protocol", "CANONICAL_SHA256.csv"), newline="") as f:
        for r in csv.DictReader(f):
            src = os.path.join(REL, r["release_path"])
            dst = os.path.join(rc.WORKSPACE, r["workspace_path"])
            assert sha(src) == r["sha256"], f"{src}: sha256 mismatch with CANONICAL_SHA256.csv"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(src, dst)
            assert sha(dst) == r["sha256"]
            n += 1
    print(f"copied {n} canonical files into {canon}")

    ds = os.path.join(rc.DATA_DIR, "dataset.csv")
    jk = os.path.join(canon, "judge_keys_1000.csv")
    if os.path.exists(ds) and not os.path.exists(jk):
        subprocess.run([sys.executable, os.path.join(HERE, "protocol", "regenerate_judge_keys.py"), ds, jk], check=True)


if __name__ == "__main__":
    main()
