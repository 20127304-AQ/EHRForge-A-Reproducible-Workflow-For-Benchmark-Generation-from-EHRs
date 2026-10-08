"""Record library versions of BOTH images and smoke-test NV-Embed-v2 model.encode() on real data.
Run: modal run code/ehr_modal.py   (from the workdir)"""
import json, os, sys
import modal

try:  # release adaptation: evaluation/config.yaml -> EHRFORGE_* env vars (absent inside Modal containers)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import release_config as _rc
except ImportError:
    _rc = None
SDK_PATH = os.environ.get("ORCHESTRA_SDK_PATH", "")  # optional platform tracking SDK (not shipped)
_ENV = {"HF_HUB_ENABLE_HF_TRANSFER": "0"}  # release: platform tracking ids removed

main_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system torch torchvision numpy transformers datasets tiktoken tqdm matplotlib pandas"
    )
    .run_commands("uv pip install --system sentence-transformers rank_bm25 pyyaml pyarrow requests")
    .env(_ENV)
)

# NV-Embed-v2 model card requirements: torch==2.2.0 transformers==4.42.4 flash-attn==2.2.0 sentence-transformers==2.7.0
nvembed_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system torch==2.2.0 'numpy<2' transformers==4.42.4 sentence-transformers==2.7.0 "
        "datasets==2.20.0 'pyarrow<17' einops accelerate pandas pyyaml tqdm requests"
    )
    .env(_ENV)
)

VOLUME_NAME = os.environ.get("EHRFORGE_STEP1_VOLUME", "ehrforge-step1-v11")

app = modal.App(os.environ.get("EHRFORGE_STEP1_APP_SMOKE", "ehrforge-v11-smoke"))
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name(os.environ.get("EHRFORGE_HF_CACHE_VOLUME", "huggingface-cache"), create_if_missing=True)

NV_REPO, NV_SHA = "nvidia/NV-Embed-v2", "3fa59658547db50a1e8e3346cf057fd0c77ed6ef"
QUERY_INSTR = "Instruct: Given a clinical question, retrieve patient visit notes that answer the question\nQuery: "


def _versions():
    import importlib, platform
    out = {"python": platform.python_version()}
    from importlib.metadata import version as _v
    for m in ["torch", "transformers", "sentence-transformers", "numpy", "tokenizers", "rank-bm25", "datasets", "pyarrow", "accelerate", "einops"]:
        try:
            out[m] = _v(m)
        except Exception as e:
            out[m] = f"NOT_INSTALLED ({type(e).__name__})"
    return out


@app.function(image=main_image, cpu=2, timeout=600, volumes={"/vol": vol})
def main_versions():
    v = _versions(); os.makedirs("/vol/meta", exist_ok=True)
    json.dump(v, open("/vol/meta/main_image_versions.json", "w"), indent=2); vol.commit(); return v


@app.function(image=nvembed_image, gpu="A100-80GB", timeout=1800,
              volumes={"/vol": vol, "/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_name(os.environ.get("EHRFORGE_HF_SECRET", "huggingface-secret"))])
def nvembed_smoke(queries, passages):
    import torch, time
    import torch.nn.functional as F
    from transformers import AutoModel
    sys.path.insert(0, "/root")
    try:
        from src.orchestra_sdk.experiment import Experiment
    except ImportError:  # release adaptation: platform experiment tracking is optional -> no-op stand-in
        class Experiment:
            @staticmethod
            def init(*a, **k):
                class _Noop:
                    def __getattr__(self, _n):
                        return lambda *a, **k: None
                return _Noop()
    v = _versions()
    exp = Experiment.init(name="EHRForge v1.1 NV-Embed-v2 image smoke test",
                          description="transformers 4.42.4 own image; official model.encode on 1 real QA + its 2 gold visits",
                          config={"model": NV_REPO, "sha": NV_SHA, "max_length": 4096, "gpu_type": "a100", "gpu_count": 1, **{f"v_{k}": str(x) for k, x in v.items()}},
                          x_axis_label="Step")
    exp.set_metadata({"platform": "Modal", "gpu_spec": "1x A100-80GB"})
    try:
        assert v["transformers"].startswith("4.42."), v
        t0 = time.time()
        model = AutoModel.from_pretrained(NV_REPO, revision=NV_SHA, trust_remote_code=True).cuda().eval()
        dtype = str(next(model.parameters()).dtype)
        with torch.no_grad():
            q = model.encode(queries, instruction=QUERY_INSTR, max_length=4096)
            p = model.encode(passages, instruction="", max_length=4096)
            q = F.normalize(q, p=2, dim=1); p = F.normalize(p, p=2, dim=1)
            scores = (q @ p.T).float().cpu().tolist()
        res = {"versions": v, "model": NV_REPO, "revision": NV_SHA, "param_dtype_default_load": dtype,
               "emb_dim": int(q.shape[1]), "q_norms": q.norm(dim=1).tolist(), "cosine_q_x_p": scores,
               "pooling": type(getattr(model, "latent_attention_model", None)).__name__,
               "seconds": time.time() - t0, "gpu": torch.cuda.get_device_name(0)}
        os.makedirs("/vol/meta", exist_ok=True)
        json.dump(res, open("/vol/meta/nvembed_smoke.json", "w"), indent=2); vol.commit(); hf_cache.commit()
        exp.log({"emb_dim": res["emb_dim"], "cos_q0_p0": scores[0][0], "cos_q0_p1": scores[0][1]}, step=0)
        exp.finish("completed")
        return res
    except Exception as e:
        import traceback; tb = traceback.format_exc(); print(tb)
        exp.log_text(f"FAILED: {e!r}", level="error"); exp.finish("failed")
        raise RuntimeError(f"NV-Embed-v2 FAILED (no fallback): {e!r}\n{tb}") from None


@app.local_entrypoint()
def main(smoke_input: str = "local_results/smoke_input.json"):
    inp = json.load(open(smoke_input))
    mv = main_versions.remote(); print("MAIN", json.dumps(mv))
    os.makedirs("local_results", exist_ok=True)
    json.dump(mv, open("local_results/main_image_versions.json", "w"), indent=2)
    r = nvembed_smoke.remote(inp["queries"], inp["passages"]); print("NVEMBED", json.dumps(r))
    json.dump(r, open("local_results/nvembed_smoke.json", "w"), indent=2)


@app.function(image=main_image, cpu=4, timeout=1800, volumes={"/vol": vol, "/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_name(os.environ.get("EHRFORGE_HF_SECRET", "huggingface-secret"))])
def tok_xcheck(items):
    """transformers AutoTokenizer (pinned sha) vs the `tokenizers`-based counts/decodes computed by ehr_data."""
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", revision="a09a35458c702b33eeacc393d103063234e8bc28")
    res = {"n_items": len(items), "ctx_count_mismatch": 0, "text_count_mismatch": 0, "n_trims": 0,
           "trim_decode_mismatch": 0, "examples": [], "versions": _versions(), "tokenizer_class": type(t).__name__}
    for it in items:
        if len(t(it["ctx"], add_special_tokens=False)["input_ids"]) != it["n_ctx"]:
            res["ctx_count_mismatch"] += 1; res["examples"].append(("ctx", it["n_ctx"]))
        for s, n in zip(it["texts"], it["n_texts"]):
            res["text_count_mismatch"] += len(t(s, add_special_tokens=False)["input_ids"]) != n
        for tr in it["trims"]:
            res["n_trims"] += 1
            res["trim_decode_mismatch"] += t.decode(tr["ids"]) != tr["decoded"]
    os.makedirs("/vol/meta", exist_ok=True)
    json.dump(res, open("/vol/meta/tok_xcheck.json", "w"), indent=2); vol.commit()
    return res


@app.local_entrypoint()
def xcheck(path: str = "local_results/tok_xcheck_input.json"):
    r = tok_xcheck.remote(json.load(open(path))); print("XCHECK", json.dumps(r))
    json.dump(r, open("local_results/tok_xcheck.json", "w"), indent=2)


# ============================================================ benchmark (timing on a real sample) ==============
MEDCPT_Q = ("ncbi/MedCPT-Query-Encoder", "d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc")
MEDCPT_A = ("ncbi/MedCPT-Article-Encoder", "d05a736da4bb84ee4057b7f7999485be6ed85465")


def _repo_env():
    os.environ["EHRFORGE_ROOT"] = "/vol"
    sys.path.insert(0, "/vol/shared/canonical")
    import ehr_data as E
    return E


def _sample_visits(E, n=512, seed=42):
    import random
    df = E.load_dataset(); C = E.load_corpus()
    pids = sorted(set(int(p) for p in df.person_id))
    allv = [(p, i) for p in pids for i in range(len(C[p]))]
    pick = random.Random(seed).sample(allv, n)
    return allv, [C[p][i]["text"] for p, i in pick], C, pids


def _batches_by_tokens(lengths, max_tokens, max_bs):
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    out, cur = [], []
    for i in order:
        if cur and (len(cur) + 1) * lengths[cur[0]] > max_tokens or len(cur) >= max_bs:
            out.append(cur); cur = []
        cur.append(i)
    if cur: out.append(cur)
    return out


@app.function(image=nvembed_image, gpu="A100-80GB", timeout=3600,
              volumes={"/vol": vol, "/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_name(os.environ.get("EHRFORGE_HF_SECRET", "huggingface-secret"))])
def bench_nvembed(n=512):
    import time, torch, numpy as np
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    E = _repo_env()
    allv, texts, C, pids = _sample_visits(E, n)
    tok = AutoTokenizer.from_pretrained(NV_REPO, revision=NV_SHA)
    t0 = time.time()
    all_len = [min(4096, len(x)) for x in tok([C[p][i]["text"] for p, i in allv], add_special_tokens=True)["input_ids"]]
    res = {"n_all_visits": len(allv), "tok_seconds": time.time() - t0,
           "all_tokens_capped_sum": int(sum(all_len)), "all_tokens_mean": float(np.mean(all_len)),
           "all_pct_truncated_4096": float(np.mean([l >= 4096 for l in all_len])),
           "all_len_pcts": {str(q): float(np.percentile(all_len, q)) for q in (50, 90, 95, 99, 100)}}
    samp_len = [min(4096, len(x)) for x in tok(texts, add_special_tokens=True)["input_ids"]]
    res["sample_tokens_sum"] = int(sum(samp_len)); res["sample_tokens_mean"] = float(np.mean(samp_len))
    for dtype_name in ["float32", "bfloat16"]:
        dt = getattr(torch, dtype_name)
        model = AutoModel.from_pretrained(NV_REPO, revision=NV_SHA, trust_remote_code=True, torch_dtype=dt).cuda().eval()
        for mt, mbs in [(16384, 32), (32768, 64)]:
            try:
                bs = _batches_by_tokens(samp_len, mt, mbs)
                torch.cuda.synchronize(); t = time.time()
                with torch.no_grad():
                    for b in bs:
                        e = model.encode([texts[i] for i in b], instruction="", max_length=4096)
                        F.normalize(e, p=2, dim=1)
                torch.cuda.synchronize(); s = time.time() - t
                res[f"{dtype_name}_tok{mt}"] = {"seconds": s, "visits_per_s": n / s, "tokens_per_s": sum(samp_len) / s,
                                                "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
            except Exception as ex:  # OOM etc.: record, don't fall back to any other model
                res[f"{dtype_name}_tok{mt}"] = {"error": repr(ex)[:300]}
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            print(dtype_name, mt, res[f"{dtype_name}_tok{mt}"], flush=True)
        del model; torch.cuda.empty_cache()
    res["versions"] = _versions(); res["gpu"] = torch.cuda.get_device_name(0)
    os.makedirs("/vol/meta", exist_ok=True); json.dump(res, open("/vol/meta/bench_nvembed.json", "w"), indent=2); vol.commit()
    return res


@app.function(image=main_image, gpu="A100-80GB", timeout=3600,
              volumes={"/vol": vol, "/root/.cache/huggingface": hf_cache},
              secrets=[modal.Secret.from_name(os.environ.get("EHRFORGE_HF_SECRET", "huggingface-secret"))])
def bench_medcpt(n=512):
    import time, torch
    from transformers import AutoModel, AutoTokenizer
    E = _repo_env()
    allv, texts, C, pids = _sample_visits(E, n)
    tok = AutoTokenizer.from_pretrained(MEDCPT_A[0], revision=MEDCPT_A[1])
    model = AutoModel.from_pretrained(MEDCPT_A[0], revision=MEDCPT_A[1]).cuda().eval()
    res = {}
    for bsz in (64, 128):
        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad():
            for i in range(0, n, bsz):
                enc = tok([["", x] for x in texts[i:i + bsz]], truncation=True, padding=True, return_tensors="pt", max_length=512).to("cuda")
                model(**enc).last_hidden_state[:, 0, :]
        torch.cuda.synchronize(); s = time.time() - t
        res[f"bs{bsz}"] = {"seconds": s, "visits_per_s": n / s}
    full = tok([["", C[p][i]["text"]] for p, i in allv], truncation=False)["input_ids"]
    res["pct_truncated_512"] = sum(len(x) > 512 for x in full) / len(full)
    res["versions"] = _versions(); res["gpu"] = torch.cuda.get_device_name(0)
    json.dump(res, open("/vol/meta/bench_medcpt.json", "w"), indent=2); vol.commit()
    return res


@app.local_entrypoint()
def bench():
    m = bench_medcpt.spawn()
    r = bench_nvembed.remote(); print("BENCH_NV", json.dumps(r))
    json.dump(r, open("local_results/bench_nvembed.json", "w"), indent=2)
    rm = m.get(); print("BENCH_MEDCPT", json.dumps(rm))
    json.dump(rm, open("local_results/bench_medcpt.json", "w"), indent=2)
