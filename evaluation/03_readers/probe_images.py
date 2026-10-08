"""Probe both Step-3 images: record runtime versions (python, CUDA, torch, vllm, transformers, driver)."""
import json
import os
import sys

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/step3code")
from images import vllm_image, extractive_image, VLLM_IMAGE_REF, WORKDIR  # noqa: E402

app = modal.App("ehrforge-step3-probe")


@app.function(image=vllm_image, gpu="A100-80GB", timeout=900)
def probe_vllm():
    import sys
    sys.path.insert(0, "/step3code")
    from images import runtime_versions
    v = runtime_versions()
    import subprocess
    v["which_python"] = sys.executable
    for f in ["/opt/vllm_image_pip_freeze_before.txt", "/opt/vllm_image_pip_freeze_after.txt"]:
        v[os.path.basename(f)] = open(f).read() if os.path.exists(f) else "missing"
    v["os_release"] = open("/etc/os-release").read()[:200]
    return v


@app.function(image=extractive_image, gpu="A100-80GB", timeout=900)
def probe_extractive():
    import sys
    sys.path.insert(0, "/step3code")
    from images import runtime_versions
    v = runtime_versions()
    import subprocess
    v["which_python"] = sys.executable
    v["pip_freeze"] = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    v["os_release"] = open("/etc/os-release").read()[:200]
    return v


@app.local_entrypoint()
def main():
    a = probe_vllm.spawn()
    b = probe_extractive.spawn()
    out = {"vllm_image_ref": VLLM_IMAGE_REF, "vllm": a.get(), "extractive": b.get()}
    p = os.path.join(WORKDIR, "results", "image_probe.json")
    json.dump(out, open(p, "w"), indent=1)
    for k in ("vllm", "extractive"):
        print(k, {x: y for x, y in out[k].items() if "freeze" not in x})
