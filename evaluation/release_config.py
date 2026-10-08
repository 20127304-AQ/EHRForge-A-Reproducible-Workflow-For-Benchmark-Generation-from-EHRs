"""Release path/config layer for the EHRForge v1.4 evaluation code (added for the public release; no scientific logic).

Reads evaluation/config.yaml (flat KEY: value lines), lets EHRFORGE_<KEY> environment variables override it,
and exports every key as EHRFORGE_<KEY> into os.environ (setdefault) so that Modal scripts can read their
object names with os.environ.get(...) both locally and (with defaults) inside containers.

Public API
  WORKSPACE      absolute path of the restricted-input workspace (original project layout: data/, shared/)
  DATA_DIR       absolute path, must equal WORKSPACE/data
  OUT_DIR        absolute path of the output root
  CFG            dict of all (resolved) config values
  step_out(s)    -> OUT_DIR/s (creates s/results and s/logs); replaces the original agent work directories
  enter_workspace()  sets EHRFORGE_ROOT (used by ehr_data.py) and chdir()s into WORKSPACE, because every
                     original script resolves its inputs relative to the project root (data/..., shared/...)
"""
from __future__ import annotations

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.environ.get("EHRFORGE_CONFIG", os.path.join(HERE, "config.yaml"))
_PATH_KEYS = ("WORKSPACE_DIR", "DATA_DIR", "OUT_DIR")


def _parse(path: str) -> dict:
    cfg = {}
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.split("#", 1)[0].strip()
            if not s:
                continue
            m = re.fullmatch(r"([A-Z0-9_]+)\s*:\s*(.*)", s)
            if not m:
                raise ValueError(f"{path}:{ln}: expected 'KEY: value', got {line!r}")
            cfg[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return cfg


CFG = _parse(CONFIG_PATH)
for _k in list(CFG):
    CFG[_k] = os.environ.get("EHRFORGE_" + _k, CFG[_k])
_base = os.path.dirname(os.path.abspath(CONFIG_PATH))
for _k in _PATH_KEYS:
    CFG[_k] = os.path.abspath(os.path.join(_base, CFG[_k]))

WORKSPACE = CFG["WORKSPACE_DIR"]
DATA_DIR = CFG["DATA_DIR"]
OUT_DIR = CFG["OUT_DIR"]
if os.path.realpath(DATA_DIR) != os.path.realpath(os.path.join(WORKSPACE, "data")):
    raise ValueError(f"DATA_DIR ({DATA_DIR}) must be WORKSPACE_DIR/data ({os.path.join(WORKSPACE, 'data')}); "
                     "ehr_data.py and all step scripts read <workspace>/data/")

for _k, _v in CFG.items():
    os.environ.setdefault("EHRFORGE_" + _k, _v)


def step_out(step: str) -> str:
    p = os.path.join(OUT_DIR, step)
    for sub in ("results", "logs"):
        os.makedirs(os.path.join(p, sub), exist_ok=True)
    return p


def enter_workspace() -> str:
    if not os.path.isdir(WORKSPACE):
        raise FileNotFoundError(f"WORKSPACE_DIR {WORKSPACE} does not exist (see README: Data access)")
    os.environ.setdefault("EHRFORGE_ROOT", WORKSPACE)
    os.chdir(WORKSPACE)
    return WORKSPACE
