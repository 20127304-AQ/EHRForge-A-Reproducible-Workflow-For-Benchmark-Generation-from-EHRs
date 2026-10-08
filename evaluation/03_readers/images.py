"""Modal image definitions for Step 3 (protocol v1.3). Imported by every Step-3 Modal script.

VLLM image: official vllm/vllm-openai image pinned by tag AND amd64 manifest digest (Docker Hub), plus a
few pure-python helpers (pandas/pyarrow/pyyaml are already present or added). Nothing in torch/vllm/transformers
is re-installed, so the versions are exactly the ones shipped in the pinned image.

EXTRACTIVE image: the project-mandated ml_general_image (debian_slim py3.11 + uv torch/transformers/...),
plus pyarrow/pyyaml/tokenizers helpers. Versions recorded at runtime.
"""
import os
import modal

HERE = os.path.dirname(os.path.abspath(__file__))
try:  # release adaptation: paths from evaluation/config.yaml (module absent inside Modal containers)
    import sys as _sys
    _sys.path.insert(0, os.path.join(HERE, ".."))
    import release_config as _rc
    WORKDIR = _rc.step_out("03_readers")  # outputs (was the agent work directory)
    ROOT = _rc.WORKSPACE  # inputs in the original project layout
except ImportError:
    WORKDIR = os.path.abspath(os.path.join(HERE, ".."))
    ROOT = os.path.abspath(os.path.join(WORKDIR, ".."))
SDK_PATH = os.environ.get("ORCHESTRA_SDK_PATH", "")  # optional platform tracking SDK (not shipped)

VLLM_TAG = "v0.28.0"
VLLM_INDEX_DIGEST = "sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14"
VLLM_AMD64_DIGEST = "sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635"
VLLM_IMAGE_REF = f"vllm/vllm-openai:{VLLM_TAG}@{VLLM_AMD64_DIGEST}"
VLLM_PIP_VERSION = "0.28.0"

_ENV = {
    "EHRFORGE_ROOT": "/vol",
    "PYTHONDONTWRITEBYTECODE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_ENABLE_HF_TRANSFER": "0",
    "DISABLE_SAFETENSORS_CONVERSION": "1",
}


def _add_code(img):
    img = img.add_local_file(os.path.join(ROOT, "shared/canonical/ehr_data.py"), "/ehr/shared/canonical/ehr_data.py", copy=False)
    img = img.add_local_file(os.path.join(ROOT, "shared/canonical/sampled_window.py"), "/ehr/shared/canonical/sampled_window.py", copy=False)
    img = img.add_local_dir(HERE, "/step3code", copy=False,  # release: code dir is this directory
                            ignore=["__pycache__", "*.pyc"])
    if SDK_PATH and os.path.isdir(SDK_PATH):
        img = img.add_local_dir(SDK_PATH, remote_path="/root/src", copy=False)
    return img


vllm_base = (
    modal.Image.from_registry(VLLM_IMAGE_REF, setup_dockerfile_commands=[
        "RUN which python3 && python3 --version && (test -e /usr/bin/python || ln -s $(which python3) /usr/bin/python)"])
    .entrypoint([])
    .run_commands("python3 -m pip freeze > /opt/vllm_image_pip_freeze_before.txt",
                  "python3 -m pip install --quiet pyarrow==25.0.1 pandas==3.0.6 pyyaml==6.0.3 requests",
                  "python3 -m pip freeze > /opt/vllm_image_pip_freeze_after.txt")
    .env(_ENV)
)
vllm_image = _add_code(vllm_base)

ml_general_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system torch torchvision numpy transformers datasets tiktoken tqdm matplotlib pandas"
    )
)
extractive_base = (
    ml_general_image
    .run_commands("uv pip install --system pyarrow pyyaml tokenizers huggingface_hub requests")
    .env(_ENV)
)
extractive_image = _add_code(extractive_base)

VERSION_PKGS = ["torch", "transformers", "tokenizers", "vllm", "huggingface_hub", "safetensors", "numpy",
                "pandas", "pyarrow", "pyyaml", "datasets", "flashinfer-python", "xformers", "triton", "accelerate"]


def runtime_versions():
    """Call INSIDE a container."""
    import importlib.metadata as md
    import platform
    import subprocess
    v = {"python": platform.python_version()}
    for p in VERSION_PKGS:
        try:
            v[p] = md.version(p)
        except Exception:
            v[p] = "NOT_INSTALLED"
    try:
        import torch
        v["torch_cuda"] = torch.version.cuda
        v["cudnn"] = str(torch.backends.cudnn.version())
        if torch.cuda.is_available():
            v["gpu_name"] = torch.cuda.get_device_name(0)
            v["gpu_count_visible"] = torch.cuda.device_count()
    except Exception as e:  # noqa
        v["torch_err"] = repr(e)
    try:
        v["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except Exception:
        v["nvidia_smi"] = "unavailable"
    for f in ["/usr/local/cuda/version.json", "/usr/local/cuda/version.txt"]:
        if os.path.exists(f):
            v["cuda_toolkit_file"] = open(f).read()[:300]
            break
    v["env_CUDA_VERSION"] = os.environ.get("CUDA_VERSION", "")
    v["env_NV_CUDA_LIB_VERSION"] = os.environ.get("NV_CUDA_LIB_VERSION", "")
    return v
