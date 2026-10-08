"""Step 4 (protocol v1.4) Modal jobs. Run from project root, e.g.
  modal run W/code/s4_modal.py --mode download_judge
  modal run --detach W/code/s4_modal.py --mode bertscore
  modal run --detach W/code/s4_modal.py --mode judge --tag pilot1 --items W/results/pilot/pilot1/items.parquet
Results persist on Volume <STEP4_VOLUME> after EVERY batch; reruns resume (dedup by item key)."""
import hashlib
import json
import os
import sys
import time

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
try:  # release adaptation: evaluation/config.yaml -> EHRFORGE_* env vars (absent inside Modal containers)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import release_config as _rc
except ImportError:
    _rc = None
W = _rc.step_out("04_scoring") if _rc else os.path.abspath(os.path.join(HERE, ".."))  # release: outputs
SDK_PATH = os.environ.get("ORCHESTRA_SDK_PATH", "")  # optional platform tracking SDK (not shipped)

VOL_NAME = os.environ.get("EHRFORGE_STEP4_VOLUME", "ehrforge-step4-v14")
HF_VOL_NAME = os.environ.get("EHRFORGE_STEP4_HF_VOLUME", "ehrforge-step4-hf")
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)
hfvol = modal.Volume.from_name(HF_VOL_NAME, create_if_missing=True)
app = modal.App(os.environ.get("EHRFORGE_STEP4_APP", "ehrforge-step4-v14"))

VLLM_IMAGE_REF = "vllm/vllm-openai:v0.28.0@sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635"
PROMETHEUS_REPO = "prometheus-eval/prometheus-8x7b-v2.0"
PROMETHEUS_SHA = "2db013b60e3e91f7a06113e436410899768e8228"
BS_MODELS = {
    "roberta_large": {"repo": "FacebookAI/roberta-large", "sha": "722cf37b1afa9454edce342e7895e588b6ff1d59", "num_layers": 17},
    "bio_clinicalbert": {"repo": "emilyalsentzer/Bio_ClinicalBERT", "sha": "d5892b39a4adaed74b92212a44081509db72f87b", "num_layers": 9},
}
BS_BATCH = 64
PAIRS_SHA = "4b62bf4ec92c53bbca71405275810a2c06d7539580c5ce166856a66a1dd57490"
READERS = ["roberta_base_squad2", "biobert_v1_1_pubmed_squad_v2", "longformer_squadv2", "qwen2_5_7b", "qwen2_5_32b"]
CONDS = ["bm25@full_timeline", "medcpt@full_timeline", "nvembed_v2@full_timeline", "hybrid_rrf60@full_timeline", "oracle"]

_ENV = {  # release: platform tracking ids removed
        "HF_HOME": "/hf", "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false",
        "DISABLE_SAFETENSORS_CONVERSION": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}


def _code(img):
    img = img.add_local_file(os.path.join(HERE, "judge_common.py"), "/s4code/judge_common.py", copy=False)
    if SDK_PATH and os.path.isdir(SDK_PATH):
        img = img.add_local_dir(SDK_PATH, remote_path="/root/src", copy=False)
    return img


judge_image = _code(
    modal.Image.from_registry(VLLM_IMAGE_REF, setup_dockerfile_commands=[
        "RUN which python3 && (test -e /usr/bin/python || ln -s $(which python3) /usr/bin/python)"])
    .entrypoint([])
    .run_commands("python3 -m pip freeze > /opt/pip_freeze_before.txt",
                  "python3 -m pip install --quiet pyarrow==25.0.1 pandas==3.0.6 pyyaml==6.0.3 requests",
                  "python3 -m pip install --quiet --no-deps prometheus-eval==0.1.20 fschat==0.2.36",
                  "python3 -m pip freeze > /opt/pip_freeze_after.txt")
    .env(_ENV))

bs_image = _code(
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .run_commands("uv pip install --system torch==2.8.0 transformers==4.56.2 bert-score==0.3.13 numpy pandas==3.0.6 "
                  "pyarrow==25.0.1 huggingface_hub tqdm matplotlib requests")
    .env(_ENV))


def _sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _versions(pkgs):
    import importlib.metadata as md, platform
    v = {"python": platform.python_version()}
    for p in pkgs:
        try:
            v[p] = md.version(p)
        except Exception:
            v[p] = "NOT_INSTALLED"
    try:
        import torch
        v["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            v["gpu_name"] = torch.cuda.get_device_name(0); v["gpu_count"] = torch.cuda.device_count()
    except Exception as e:
        v["torch_err"] = repr(e)
    return v


def _exp(name, config, xlabel):
    try:
        sys.path.insert(0, "/root")
        from src.orchestra_sdk.experiment import Experiment
        e = Experiment.init(name=name, config=config, x_axis_label=xlabel)
        return e
    except Exception as ex:  # SDK must never break a scoring run
        print("experiment SDK unavailable:", repr(ex))
        return None


# --------------------------------------------------------------------------------------------- weights
@app.function(image=judge_image, volumes={"/hf": hfvol, "/vol": vol}, cpu=8, memory=32768, timeout=6 * 3600)
def download_judge(expected: dict):
    from huggingface_hub import snapshot_download
    from concurrent.futures import ThreadPoolExecutor
    t0 = time.time()
    p = snapshot_download(PROMETHEUS_REPO, revision=PROMETHEUS_SHA, ignore_patterns=["*.JPG", "*.jpg"], max_workers=8)
    hfvol.commit()
    assert f"/snapshots/{PROMETHEUS_SHA}" in p, p
    print("downloaded", p, round(time.time() - t0), "s", flush=True)
    files = sorted(expected)
    with ThreadPoolExecutor(8) as ex:
        got = dict(zip(files, ex.map(lambda f: _sha_file(os.path.join(p, f)), files)))
    rec = {"snapshot_path": p, "commit": PROMETHEUS_SHA, "files": {f: {"expected": expected[f], "got": got[f],
           "bytes": os.path.getsize(os.path.join(p, f)), "ok": expected[f] == got[f]} for f in files},
           "seconds": round(time.time() - t0)}
    rec["all_ok"] = all(v["ok"] for v in rec["files"].values())
    os.makedirs("/vol/meta", exist_ok=True)
    json.dump(rec, open("/vol/meta/prometheus_weights_verify.json", "w"), indent=1)
    vol.commit()
    assert rec["all_ok"], "WEIGHT SHA MISMATCH"
    return rec


# --------------------------------------------------------------------------------------------- BERTScore
@app.function(image=bs_image, volumes={"/hf": hfvol, "/vol": vol}, gpu="A100-80GB", memory=65536, timeout=8 * 3600)
def bertscore(model_key: str):
    import numpy as np, pandas as pd, torch
    from huggingface_hub import snapshot_download
    import bert_score
    from bert_score.utils import sent_encode
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = BS_MODELS[model_key]
    path = snapshot_download(cfg["repo"], revision=cfg["sha"],
                             allow_patterns=["*.json", "*.txt", "pytorch_model.bin", "model.safetensors"])
    hfvol.commit()
    assert f"/snapshots/{cfg['sha']}" in path and "t5" not in path, path
    pairs_p = "/vol/inputs/pairs_v14.parquet"
    assert _sha_file(pairs_p) == PAIRS_SHA
    df = pd.read_parquet(pairs_p, columns=["person_id", "qa_index", "reader", "condition", "prediction", "answer", "empty_prediction"])
    scorer = bert_score.BERTScorer(model_type=path, num_layers=cfg["num_layers"], idf=False, rescale_with_baseline=False,
                                   batch_size=BS_BATCH, device="cuda")
    tok = scorer._tokenizer
    orig_mml = tok.model_max_length
    tok.model_max_length = 512
    meta = {"model_key": model_key, **cfg, "snapshot": path, "tokenizer_class": type(tok).__name__,
            "tokenizer_model_max_length_orig": int(orig_mml) if orig_mml < 1e12 else str(orig_mml),
            "tokenizer_model_max_length_used": 512, "batch_size": BS_BATCH, "dtype": str(next(scorer._model.parameters()).dtype),
            "tf32": [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32],
            "bert_score_hash": scorer.hash, "versions": _versions(["torch", "transformers", "bert-score", "tokenizers", "numpy", "pandas"])}
    print(json.dumps(meta, indent=1, default=str), flush=True)

    def ntok(texts):  # untruncated length as sent_encode would produce
        mml = tok.model_max_length
        tok.model_max_length = int(1e9)
        try:
            return np.array([len(sent_encode(tok, t)) for t in texts], dtype=np.int32)
        finally:
            tok.model_max_length = mml

    outdir = f"/vol/bertscore/{model_key}"
    os.makedirs(outdir, exist_ok=True)
    json.dump(meta, open(f"{outdir}/meta.json", "w"), indent=1, default=str)
    # gold token lengths (once)
    gold = df[(df.reader == READERS[0]) & (df.condition == CONDS[0])][["person_id", "qa_index", "answer"]].sort_values(["person_id", "qa_index"]).reset_index(drop=True)
    if not os.path.exists(f"{outdir}/gold_tokens.parquet"):
        g = gold[["person_id", "qa_index"]].copy(); g["n_tokens_gold"] = ntok(gold.answer.tolist())
        g.to_parquet(f"{outdir}/gold_tokens.parquet", index=False); vol.commit()
    exp = _exp(f"Step4 BERTScore {model_key}", {"model": cfg["repo"], "sha": cfg["sha"], "layer": cfg["num_layers"],
                                                 "gpu_type": "a100", "gpu_count": 1, "batch_size": BS_BATCH}, "Run index")
    for i, (r, c) in enumerate([(r, c) for r in READERS for c in CONDS]):
        out = f"{outdir}/{r}__{c}.parquet"
        if os.path.exists(out):
            print("skip (done)", out, flush=True); continue
        t0 = time.time()
        sub = df[(df.reader == r) & (df.condition == c)].sort_values(["person_id", "qa_index"]).reset_index(drop=True)
        assert len(sub) == 10742
        # alignment: candidates and references come from the SAME row; also check against gold frame order
        assert (sub.person_id.values == gold.person_id.values).all() and (sub.qa_index.values == gold.qa_index.values).all()
        assert (sub.answer.values == gold.answer.values).all()
        m = ~sub.empty_prediction.values
        cands = sub.prediction[m].tolist(); refs = sub.answer[m].tolist()
        P, R, F = scorer.score(cands, refs, batch_size=BS_BATCH)
        res = sub[["person_id", "qa_index", "reader", "condition", "empty_prediction"]].copy()
        for name, arr in (("P", P), ("R", R), ("F", F)):
            col = np.zeros(len(sub), dtype=np.float64); col[m] = arr.numpy().astype(np.float64)
            res[f"bs_{model_key}_{name}"] = col
        nt = np.zeros(len(sub), dtype=np.int32); nt[m] = ntok(cands)
        res[f"n_tokens_cand_{model_key}"] = nt
        res.to_parquet(out + ".tmp", index=False); os.replace(out + ".tmp", out)
        vol.commit()
        msg = f"{model_key} {r} {c}: F mean {res[f'bs_{model_key}_F'].mean():.4f} >512 cand {(nt > 512).sum()} ({time.time() - t0:.0f}s)"
        print(msg, flush=True)
        if exp:
            try:
                exp.log({"bertscore_F_mean": float(res[f"bs_{model_key}_F"].mean()), "n_cand_over_512": int((nt > 512).sum())}, step=i)
                exp.set_progress(int(100 * (i + 1) / 25))
            except Exception:
                pass
    if exp:
        try:
            exp.finish("completed")
        except Exception:
            pass
    return model_key


# --------------------------------------------------------------------------------------------- Judge
@app.function(image=judge_image, volumes={"/hf": hfvol, "/vol": vol}, gpu="H100!:2", memory=131072, cpu=8, timeout=10 * 3600)  # release: platform tracking secret removed
def judge(tag: str, items_sha: str, chunk_size: int = 1000):
    import glob
    import pandas as pd
    sys.path.insert(0, "/s4code")
    import judge_common as J
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import SamplingType
    assert J.PROMETHEUS_SHA == PROMETHEUS_SHA
    vol.reload()
    base = f"/vol/judge/{tag}"
    assert _sha_file(f"{base}/items.parquet") == items_sha, "items file sha mismatch"
    items = pd.read_parquet(f"{base}/items.parquet")
    assert not items.item_id.duplicated().any()
    assert (items.prediction.str.strip() != "").all(), "empty predictions must not be sent to the judge"
    ver = json.load(open("/vol/meta/prometheus_weights_verify.json"))
    assert ver["all_ok"] and ver["commit"] == PROMETHEUS_SHA
    items["prompt"] = [J.build_prompt(q, p, a) for q, p, a in zip(items.question, items.prediction, items.answer)]
    os.makedirs(f"{base}/out", exist_ok=True)
    done = set()
    for f in sorted(glob.glob(f"{base}/out/chunk_*.parquet")):
        done |= set(pd.read_parquet(f, columns=["item_id"]).item_id)
    pending = items[~items.item_id.isin(done)].sort_values("order").reset_index(drop=True)
    print(f"[{tag}] items {len(items)}, done {len(done)}, pending {len(pending)}", flush=True)
    if len(pending) == 0:
        return {"tag": tag, "pending": 0}

    t0 = time.time()
    llm = LLM(model=PROMETHEUS_REPO, revision=PROMETHEUS_SHA, tokenizer=PROMETHEUS_REPO, tokenizer_revision=PROMETHEUS_SHA,
              dtype="bfloat16", tensor_parallel_size=2, max_model_len=4096, enable_prefix_caching=False,
              gpu_memory_utilization=0.90, max_num_seqs=256, seed=42, trust_remote_code=False)
    load_s = time.time() - t0
    # ---- revision check
    from huggingface_hub import scan_cache_dir
    from transformers.utils import cached_file
    vc = llm.llm_engine.vllm_config
    cfgp = cached_file(PROMETHEUS_REPO, "config.json", revision=PROMETHEUS_SHA)
    snaps = sorted(rv.commit_hash for r in scan_cache_dir("/hf/hub").repos if r.repo_id == PROMETHEUS_REPO for rv in r.revisions)
    rev = {"engine_model_revision": vc.model_config.revision, "engine_tokenizer_revision": vc.model_config.tokenizer_revision,
           "config_snapshot_path": cfgp, "hf_cache_snapshots": snaps, "dtype": str(vc.model_config.dtype),
           "max_model_len": vc.model_config.max_model_len, "tp": vc.parallel_config.tensor_parallel_size,
           "prefix_caching": vc.cache_config.enable_prefix_caching, "max_num_seqs": vc.scheduler_config.max_num_seqs,
           "seed": vc.model_config.seed, "load_seconds": round(load_s)}
    rev["PASS"] = (rev["engine_model_revision"] == PROMETHEUS_SHA and rev["engine_tokenizer_revision"] == PROMETHEUS_SHA
                   and f"/snapshots/{PROMETHEUS_SHA}/" in cfgp and snaps == [PROMETHEUS_SHA])
    print(json.dumps(rev, indent=1), flush=True)
    if not rev["PASS"]:
        raise RuntimeError(f"PROMETHEUS REVISION MISMATCH {rev}")
    sp = SamplingParams(**J.SAMPLING)
    assert sp.sampling_type == SamplingType.GREEDY
    tok = llm.get_tokenizer()
    lens = [len(tok(p).input_ids) for p in pending.prompt]
    assert max(lens) + J.SAMPLING["max_tokens"] <= 4096, max(lens)
    meta = {"tag": tag, "revision": rev, "sampling_params": repr(sp), "versions": _versions(
        ["vllm", "torch", "transformers", "tokenizers", "prometheus-eval", "fschat", "huggingface_hub", "pandas", "pyarrow"]),
        "weights_verify": "/vol/meta/prometheus_weights_verify.json", "max_prompt_tokens": max(lens), "chunk_size": chunk_size,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(meta, open(f"{base}/run_meta_{int(time.time())}.json", "w"), indent=1, default=str)
    vol.commit()
    exp = _exp(f"Step4 Prometheus judge [{tag}]", {"judge": PROMETHEUS_REPO, "sha": PROMETHEUS_SHA, "n_pending": len(pending),
                                                     "gpu_type": "h100", "gpu_count": 2, **J.SAMPLING}, "Chunk")

    def run(prompts):
        outs = llm.generate(prompts, sp, use_tqdm=False)
        return [(o.outputs[0].text, o.outputs[0].finish_reason, len(o.outputs[0].token_ids), len(o.prompt_token_ids),
                 o.prompt_token_ids[:2]) for o in outs]

    nchunks = (len(pending) + chunk_size - 1) // chunk_size
    for ci in range(nchunks):
        ch = pending.iloc[ci * chunk_size:(ci + 1) * chunk_size].copy()
        t1 = time.time()
        res = run(ch.prompt.tolist())
        ch["raw_feedback"] = [x[0] for x in res]; ch["finish_reason"] = [x[1] for x in res]
        ch["n_output_tokens"] = [x[2] for x in res]; ch["n_prompt_tokens"] = [x[3] for x in res]
        assert all(x[4][0] == 1 and x[4][1] != 1 for x in res), "BOS check failed"
        parsed = [J.parse_score(t) for t in ch.raw_feedback]
        ch["score_first"] = [p[0] for p in parsed]; ch["invalid_first"] = [not p[1] for p in parsed]
        ch["retried"] = ch.invalid_first.values
        ch["raw_feedback_retry"] = None; ch["finish_reason_retry"] = None
        ri = ch.index[ch.invalid_first]
        if len(ri):
            rr = run(ch.loc[ri, "prompt"].tolist())
            ch.loc[ri, "raw_feedback_retry"] = [x[0] for x in rr]; ch.loc[ri, "finish_reason_retry"] = [x[1] for x in rr]
        final = []
        for i, row in ch.iterrows():
            if not row.invalid_first:
                final.append((row.score_first, False))
            else:
                s, ok = J.parse_score(row.raw_feedback_retry)
                final.append((s, False) if ok else (1, True))
        ch["score"] = [f[0] for f in final]; ch["invalid_final"] = [f[1] for f in final]
        ch["judge_called"] = True; ch["empty_prediction"] = False
        ch["prompt_sha256"] = [hashlib.sha256(p.encode()).hexdigest() for p in ch.prompt]
        ch = ch.drop(columns=["prompt"])
        out = f"{base}/out/chunk_{int(ch.order.min()):07d}.parquet"
        ch.to_parquet(out + ".tmp", index=False); os.replace(out + ".tmp", out)
        vol.commit()
        msg = (f"[{tag}] chunk {ci + 1}/{nchunks}: n={len(ch)} mean={ch.score.mean():.3f} inv_first={ch.invalid_first.mean():.3%} "
               f"inv_final={ch.invalid_final.mean():.3%} out_tok={ch.n_output_tokens.mean():.0f} {time.time() - t1:.0f}s")
        print(msg, flush=True)
        if exp:
            try:
                exp.log({"mean_score": float(ch.score.mean()), "invalid_first_rate": float(ch.invalid_first.mean()),
                         "mean_output_tokens": float(ch.n_output_tokens.mean())}, step=ci)
                exp.set_progress(int(100 * (ci + 1) / nchunks))
            except Exception:
                pass
    if exp:
        try:
            exp.finish("completed")
        except Exception:
            pass
    return {"tag": tag, "done": len(pending)}


@app.local_entrypoint()
def main(mode: str = "", tag: str = "", items: str = "", chunk_size: int = 1000):
    if mode == "download_judge":
        meta = json.load(open(os.path.join(W, "results/prometheus_hub_metadata.json")))
        exp = {f["path"]: f["lfs_sha256"] for f in meta["files"] if f["lfs_sha256"]}
        rec = download_judge.remote(exp)
        json.dump(rec, open(os.path.join(W, "results/prometheus_weights_verify.json"), "w"), indent=1)
        print("all_ok", rec["all_ok"], rec["seconds"])
    elif mode == "upload_inputs":
        with vol.batch_upload(force=True) as b:
            b.put_file(os.path.join(W, "results/inputs/pairs_v14.parquet"), "/inputs/pairs_v14.parquet")
        print("uploaded")
    elif mode == "bertscore":
        for k in bertscore.map(list(BS_MODELS)):
            print("done", k)
    elif mode == "judge":
        assert tag and items
        with vol.batch_upload(force=True) as b:
            b.put_file(items, f"/judge/{tag}/items.parquet")
        print(judge.remote(tag, _sha_file(items), chunk_size))
    else:
        raise SystemExit("unknown mode")
