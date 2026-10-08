"""Tests of the release-only layer (config, workspace setup) + compile check of every shipped Python file."""
import glob
import hashlib
import os
import py_compile
import subprocess
import sys

REL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_every_shipped_python_file_compiles(tmp_path):
    files = sorted(glob.glob(os.path.join(REL, "**", "*.py"), recursive=True))
    assert len(files) > 30
    for f in files:
        py_compile.compile(f, cfile=str(tmp_path / (hashlib.md5(f.encode()).hexdigest() + ".pyc")), doraise=True)


def test_config_and_setup_workspace(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(open(os.path.join(REL, "evaluation", "config.yaml")).read()
                   .replace("WORKSPACE_DIR: ../workspace", f"WORKSPACE_DIR: {tmp_path / 'ws'}")
                   .replace("DATA_DIR: ../workspace/data", f"DATA_DIR: {tmp_path / 'ws' / 'data'}")
                   .replace("OUT_DIR: ../out", f"OUT_DIR: {tmp_path / 'out'}"))
    env = dict(os.environ, EHRFORGE_CONFIG=str(cfg))
    r = subprocess.run([sys.executable, os.path.join(REL, "evaluation", "setup_workspace.py")], env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    for fn in ("ehr_data.py", "sampled_window.py", "protocol.yaml", "protocol_v1.4.yaml", "protocol_v1.4.yaml.sha256"):
        assert (tmp_path / "ws" / "shared" / "canonical" / fn).exists()
    code = ("import sys; sys.path.insert(0, %r); import release_config as rc, os; "
            "p = rc.step_out('04_scoring'); print(p); print(rc.CFG['STEP4_VOLUME']); print(os.environ['EHRFORGE_STEP4_VOLUME'])"
            % os.path.join(REL, "evaluation"))
    out = subprocess.run([sys.executable, "-c", code], env=dict(env, EHRFORGE_STEP4_VOLUME="my-vol"),
                         capture_output=True, text=True, check=True).stdout.split()
    assert out[0] == str(tmp_path / "out" / "04_scoring") and os.path.isdir(os.path.join(out[0], "results"))
    assert out[1] == out[2] == "my-vol"                              # env override wins and is exported
