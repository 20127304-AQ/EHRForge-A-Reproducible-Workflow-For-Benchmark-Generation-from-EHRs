"""Generative reader on vLLM (protocol v1.3 generative_engine). Written from scratch for Step 3.
ENGINE[model] = FIXED per-model engine args; the LLM is built ONCE per model; engine_record() reads the live
configuration back from the engine objects for the per-condition config hash.
"""
import hashlib
import json
import re

ENGINE_COMMON = {
    "dtype": "bfloat16",
    "max_model_len": 8192,
    "max_num_batched_tokens": 8192,
    "gpu_memory_utilization": 0.90,
    "enable_prefix_caching": False,
    "enable_chunked_prefill": True,
    "enforce_eager": False,
    "seed": 42,
    "kv_cache_dtype": "auto",
    "trust_remote_code": False,
}
ENGINE = {
    "qwen2_5_7b": dict(ENGINE_COMMON, tensor_parallel_size=1, max_num_seqs=128),
    "qwen2_5_32b": dict(ENGINE_COMMON, tensor_parallel_size=2, max_num_seqs=64),
}
GPU = {"qwen2_5_7b": ("H100!", 1), "qwen2_5_32b": ("H100!", 2)}
SAMPLING = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 256, "seed": 42, "n": 1}


def _snap(p):
    m = re.search(r"/snapshots/([0-9a-f]{40})(/|$)", p or "")
    return m.group(1) if m else None


def predownload(repo, sha):
    from huggingface_hub import snapshot_download
    p = snapshot_download(repo, revision=sha,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "merges.txt", "vocab.json", "tokenizer*"])
    assert _snap(p + "/") == sha, (p, sha)
    return p


def build_engine(model_key, repo, sha):
    from vllm import LLM
    kw = dict(ENGINE[model_key])
    llm = LLM(model=repo, revision=sha, tokenizer=repo, tokenizer_revision=sha, **kw)
    return llm, kw


def hf_tokenizer(repo, sha):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(repo, revision=sha)


def _g(o, *names):
    for n in names:
        if o is not None and hasattr(o, n):
            return getattr(o, n)
    return "MISSING"


def get_vllm_config(llm):
    eng = llm.llm_engine
    if hasattr(eng, "vllm_config"):
        return eng.vllm_config
    if hasattr(eng, "get_vllm_config"):
        return eng.get_vllm_config()
    if hasattr(llm, "vllm_config"):
        return llm.vllm_config
    raise RuntimeError("cannot locate vllm_config")


def engine_record(llm, gpu_type, gpu_count):
    """Live engine configuration read back from the engine objects (NOT from ENGINE)."""
    import torch
    vc = get_vllm_config(llm)
    mc, sc, cc, pc = vc.model_config, vc.scheduler_config, vc.cache_config, vc.parallel_config
    rec = {
        "model": _g(mc, "model"), "revision": _g(mc, "revision"), "tokenizer": _g(mc, "tokenizer"),
        "tokenizer_revision": _g(mc, "tokenizer_revision"),
        "dtype": str(_g(mc, "dtype")), "max_model_len": _g(mc, "max_model_len"), "seed": _g(mc, "seed"),
        "quantization": _g(mc, "quantization"), "enforce_eager": _g(mc, "enforce_eager"),
        "max_num_seqs": _g(sc, "max_num_seqs"), "max_num_batched_tokens": _g(sc, "max_num_batched_tokens"),
        "enable_chunked_prefill": _g(sc, "enable_chunked_prefill", "chunked_prefill_enabled"),
        "async_scheduling": _g(sc, "async_scheduling"),
        "gpu_memory_utilization": _g(cc, "gpu_memory_utilization"),
        "enable_prefix_caching": _g(cc, "enable_prefix_caching"), "kv_cache_dtype": _g(cc, "cache_dtype"),
        "block_size": _g(cc, "block_size"), "num_gpu_blocks": _g(cc, "num_gpu_blocks"),
        "tensor_parallel_size": _g(pc, "tensor_parallel_size"),
        "pipeline_parallel_size": _g(pc, "pipeline_parallel_size"),
        "data_parallel_size": _g(pc, "data_parallel_size"),
        "distributed_executor_backend": str(_g(pc, "distributed_executor_backend")),
        "speculative_config": str(getattr(vc, "speculative_config", None)),
        "gpu_type_requested": gpu_type, "gpu_count_requested": gpu_count,
        "gpu_name": torch.cuda.get_device_name(0), "gpu_count_visible": torch.cuda.device_count(),
    }
    rec = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in rec.items()}
    rec["engine_config_hash"] = hashlib.sha256(json.dumps(rec, sort_keys=True, default=str).encode()).hexdigest()
    try:
        full = str(vc)
    except Exception as e:  # noqa
        full = f"repr-failed {e!r}"
    rec["vllm_config_repr_sha256"] = hashlib.sha256(full.encode()).hexdigest()
    rec["vllm_config_repr"] = full[:20000]
    return rec


def sampling_params():
    from vllm import SamplingParams
    return SamplingParams(**SAMPLING)


def sampling_record(sp):
    keys = ["temperature", "top_p", "top_k", "max_tokens", "seed", "n", "min_p", "repetition_penalty",
            "presence_penalty", "frequency_penalty", "stop", "stop_token_ids", "skip_special_tokens"]
    r = {k: getattr(sp, k, "MISSING") for k in keys}
    r = {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v)) for k, v in r.items()}
    st = getattr(sp, "sampling_type", None)
    r["sampling_type"] = getattr(st, "name", str(st))
    try:
        from vllm.sampling_params import SamplingType
        r["sampling_type_is_GREEDY_enum"] = bool(st == SamplingType.GREEDY)
    except Exception as e:  # noqa
        r["sampling_type_is_GREEDY_enum"] = f"check-failed {e!r}"
    r["repr"] = repr(sp)
    return r


def resolve_revisions(repo, sha, llm, tok):
    from huggingface_hub import scan_cache_dir
    from transformers.utils import cached_file
    vc = get_vllm_config(llm)
    hf_conf = getattr(vc.model_config, "hf_config", None)
    snaps = []
    for r in scan_cache_dir().repos:
        if r.repo_id == repo:
            snaps = sorted(rv.commit_hash for rv in r.revisions)
    tok_files = {}
    for fn in ["tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt"]:
        p = cached_file(repo, fn, revision=sha, _raise_exceptions_for_missing_entries=False)
        tok_files[fn] = _snap(p)
    wf = cached_file(repo, "model.safetensors.index.json", revision=sha, _raise_exceptions_for_missing_entries=False)
    rec = {
        "repo": repo, "pinned": sha,
        "engine_model_revision": vc.model_config.revision,
        "engine_tokenizer_revision": vc.model_config.tokenizer_revision,
        "engine_model_path": str(vc.model_config.model),
        "hf_config_commit_hash": getattr(hf_conf, "_commit_hash", None),
        "weights_index_snapshot": _snap(wf), "hf_tokenizer_files_snapshot": tok_files,
        "hf_tokenizer_name_or_path": getattr(tok, "name_or_path", None),
        "hf_cache_snapshots_for_repo": snaps,
    }
    model_c = {c for c in [rec["weights_index_snapshot"], rec["hf_config_commit_hash"],
                           _snap(rec["engine_model_path"] + "/")] if c}
    tok_c = {c for c in tok_files.values() if c}
    rec["model_resolved"] = sorted(model_c)[0] if len(model_c) == 1 else sorted(model_c)
    rec["tokenizer_resolved"] = sorted(tok_c)[0] if len(tok_c) == 1 else sorted(tok_c)
    rec["model_PASS"] = (rec["model_resolved"] == sha and snaps == [sha] and rec["engine_model_revision"] == sha)
    rec["tokenizer_PASS"] = (rec["tokenizer_resolved"] == sha and rec["engine_tokenizer_revision"] == sha)
    return rec


def build_prompt_ids(tok, system_prompt, user_template, context, question):
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_template.format(context=context, question=question)}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(text, add_special_tokens=False)
    return text, ids
