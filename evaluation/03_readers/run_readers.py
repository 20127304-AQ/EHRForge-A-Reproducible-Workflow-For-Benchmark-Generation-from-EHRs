"""Step 3 reader runs (protocol v1.3). Modes:
  prechecks   : (a) extractive load-warning checks + revision resolution, (b) SQuAD2 dev 500 answerable EM/F1
  extractive  : --reader <roberta_base_squad2|biobert_v1_1_pubmed_squad_v2|longformer_squadv2> over 5 conditions
  generative  : --reader <qwen2_5_7b|qwen2_5_32b> over 5 conditions (LLM built ONCE)
--tag pilot|full ; --keys-file JSON list of [person_id, qa_index] (pilot) or empty (= all 10,742)
Shards: /vol/step3/runs/{tag}/{reader}/{condition}/part_{i:03d}.parquet (i = chunk of 1,024 sorted keys), streamed
back and written to <workdir>/results/runs/... after every shard. Resume: an existing valid shard is not recomputed.
"""
import json
import os
import sys
import time

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/step3code")
from images import extractive_image, vllm_image, WORKDIR, ROOT  # noqa: E402  (release: ROOT added)
from step3_common import VOLUME_NAME, HF_CACHE_VOLUME, CONDITIONS, READERS  # noqa: E402

app = modal.App("ehrforge-step3-readers")
vol = modal.Volume.from_name(VOLUME_NAME)
hf = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)
SECRETS = [modal.Secret.from_name(os.environ.get("EHRFORGE_HF_SECRET", "huggingface-secret"))]  # release: platform tracking secret removed
VOLS = {"/vol": vol, "/root/.cache/huggingface": hf}
PROTO13 = "/vol/shared/canonical/protocol_v1.3.yaml"

META_LIST_FIELDS = ["gold_visit_ids", "selected_visit_ids", "visit_ids_reaching_model", "per_visit_tokens_before",
                    "per_visit_tokens_after"]


# ----------------------------------------------------------------------------------------------- helpers (container)
def _setup():
    sys.path.insert(0, "/step3code")
    sys.path.insert(0, "/ehr/shared/canonical")
    sys.path.insert(0, "/root")


def _protocol(expected_sha):
    import hashlib
    import yaml
    b = open(PROTO13, "rb").read()
    got = hashlib.sha256(b).hexdigest()
    assert got == expected_sha, f"protocol_v1.3 sha {got} != expected {expected_sha}"
    p = yaml.safe_load(b)
    assert str(p["protocol_version"]) == "1.3"
    return p


def _exp(name, config, tags):
    try:
        from src.orchestra_sdk.experiment import Experiment
        e = Experiment.init(name=name, description="EHRForge Step 3 canonical readers (protocol v1.3)",
                            config=config, x_axis_label="Shard (1,024 questions)",
                            # user explicitly approved the 2xH100 Qwen-32B pilot + full run in chat (Step 3 checkpoint)
                            **({"acknowledge_high_gpu_cost": True} if config.get("gpu_count", 1) >= 2 else {}))
        e.add_tags(tags)
        e.set_metadata({"platform": "Modal", "gpu_spec": f"{config.get('gpu_count')}x {config.get('gpu_type')}"})
        return e
    except Exception as ex:  # noqa
        print("experiment SDK unavailable:", repr(ex), flush=True)
        return None


def _elog(e, fn, *a, **k):
    if e is None:
        return
    try:
        getattr(e, fn)(*a, **k)
    except Exception as ex:  # noqa
        print("SDK log failed", repr(ex), flush=True)


def _load_contexts(condition, keys):
    import pandas as pd
    import ehr_data as E
    p = f"/vol/step3/contexts/{condition}.parquet"
    meta = json.load(open(f"/vol/step3/contexts/{condition}.meta.json"))
    assert E.sha256_file(p) == meta["file_sha256"], f"context file sha mismatch {condition}"
    df = pd.read_parquet(p)
    assert len(df) == 10742 and not df.duplicated(["person_id", "qa_index"]).any()
    df = df.set_index(["person_id", "qa_index"], drop=False).sort_index()
    if keys is not None:
        df = df.loc[[tuple(k) for k in keys]]
    df = df.sort_index()
    from step3_common import sha256_text
    for c, s in zip(df.context, df.context_sha256):
        assert sha256_text(c) == s
    return df.reset_index(drop=True), meta


def _keys(keys_json):
    import ehr_data as E
    df = E.load_dataset()
    allk = sorted(map(tuple, df[["person_id", "qa_index"]].astype("int64").itertuples(index=False, name=None)))
    if not keys_json:
        return allk, None
    ks = sorted(tuple(int(x) for x in k) for k in json.loads(keys_json))
    assert set(ks) <= set(allk) and len(set(ks)) == len(ks)
    return ks, ks


def _out_row(crow, reader, condition, reader_repo, reader_sha, reaching, pred_fields, reader_input_tokens,
             n_windows, window_stride, max_seq_len):
    from ehr_data import CONTEXT_METADATA_FIELDS
    rec = {f: crow[f] for f in CONTEXT_METADATA_FIELDS if f in crow}
    rec["reader"], rec["condition"] = reader, condition
    gold = json.loads(crow["gold_visit_ids"])
    sel = json.loads(crow["selected_visit_ids"])
    before = json.loads(crow["per_visit_tokens_before"])
    after = json.loads(crow["per_visit_tokens_after"])
    reach = [v for v in sel if v in set(reaching)]
    rec["n_visits_reaching_model"] = len(reach)
    rec["visit_ids_reaching_model"] = json.dumps(reach)
    rec["n_gold_visits_surviving"] = sum(1 for g in gold if g in set(reach))
    pos = {v: i for i, v in enumerate(sel)}
    rec["gold_fully_survived"] = bool(all(g in pos and after[pos[g]] == before[pos[g]] and g in set(reach) for g in gold))
    rec["reader_tokenizer_name"], rec["reader_tokenizer_revision"] = reader_repo, reader_sha
    rec["reader_input_tokens"] = reader_input_tokens
    rec["n_windows"], rec["window_stride"], rec["max_seq_len"] = n_windows, window_stride, max_seq_len
    rec.update(pred_fields)
    # invariants
    assert rec["n_visits_reaching_model"] <= rec["n_selected_visits"]
    assert rec["n_gold_visits_surviving"] <= min(rec["n_gold_visits"], rec["n_visits_reaching_model"])
    if rec["gold_fully_survived"]:
        assert rec["n_gold_visits_surviving"] == rec["n_gold_visits"]
    return rec


def _shard_path(tag, reader, condition, i):
    return f"/vol/step3/runs/{tag}/{reader}/{condition}/part_{i:03d}.parquet"


def _existing_shard(path, chunk_keys, reader, condition):
    import pandas as pd
    if not os.path.exists(path):
        return None
    try:
        d = pd.read_parquet(path)
        ks = list(zip(d.person_id.astype(int), d.qa_index.astype(int)))
        if ks == list(chunk_keys) and set(d.reader) == {reader} and set(d.condition) == {condition} \
                and d.prediction.notna().all():
            return open(path, "rb").read()
    except Exception as e:  # noqa
        print("bad shard, recomputing", path, repr(e), flush=True)
    return None


def _write_shard(rows, path):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False), tmp, compression="zstd")
    os.replace(tmp, path)
    vol.commit()
    return open(path, "rb").read()


# ----------------------------------------------------------------------------------------------- pre-checks (a), (b)
@app.function(image=extractive_image, gpu="A100-80GB", timeout=7200, volumes=VOLS, secrets=SECRETS)
def prechecks(protocol_sha: str):
    _setup()
    import random
    from datasets import load_dataset
    from images import runtime_versions
    from extractive import load_reader, predict
    from step3_common import EXTRACTIVE, DOC_STRIDE, MAX_ANSWER_LEN, EXTRACTIVE_BATCH_SIZE, f1_em
    P = _protocol(protocol_sha)
    from step3_common import extractive_spec
    t0 = time.time()
    ds = load_dataset("rajpurkar/squad_v2", split="validation")
    ans_idx = [i for i, a in enumerate(ds["answers"]) if len(a["text"]) > 0]
    sample = sorted(random.Random(42).sample(ans_idx, 500))
    sub = ds.select(sample)
    out = {"squad": {"dataset": "rajpurkar/squad_v2 validation", "n_answerable_total": len(ans_idx),
                     "n_sample": 500, "sample_rule": "sorted(random.Random(42).sample(answerable_indices, 500))",
                     "sample_ids_first5": list(sub["id"])[:5]}, "readers": {}, "versions": runtime_versions()}
    for r in EXTRACTIVE:
        spec = extractive_spec(P, r)
        repo, sha, msl = spec["repo"], spec["sha"], spec["max_seq_len"]
        assert spec["doc_stride"] == DOC_STRIDE and P["readers"]["extractive"]["decoding"]["max_answer_len_tokens"] == MAX_ANSWER_LEN
        tok, model, resolved = load_reader(repo, sha)
        res = predict(tok, model, list(sub["question"]), list(sub["context"]), msl, DOC_STRIDE, MAX_ANSWER_LEN,
                      EXTRACTIVE_BATCH_SIZE[r])
        f1s, ems = [], []
        for x, a in zip(res, list(sub["answers"])):
            sc = [f1_em(x["prediction"], g) for g in a["text"]]
            f1s.append(max(s[0] for s in sc))
            ems.append(max(s[1] for s in sc))
        F1, EM = 100 * sum(f1s) / len(f1s), 100 * sum(ems) / len(ems)
        resolved.update({"squad_F1": F1, "squad_EM": EM, "squad_PASS": F1 > 40, "max_seq_len": msl,
                         "doc_stride": DOC_STRIDE, "max_answer_len": MAX_ANSWER_LEN,
                         "batch_size": EXTRACTIVE_BATCH_SIZE[r],
                         "examples": [{"q": q, "pred": x["prediction"], "gold": a["text"][:2]}
                                      for q, x, a in list(zip(list(sub["question"]), res, list(sub["answers"])))[:5]]})
        out["readers"][r] = resolved
        print(r, "load_check", resolved["load_check_PASS"], "model", resolved["model_resolved"],
              "tok", resolved["tokenizer_resolved"], f"SQuAD EM {EM:.2f} F1 {F1:.2f}", flush=True)
        del model
    hf.commit()
    out["seconds"] = time.time() - t0
    return out


# ----------------------------------------------------------------------------------------------- extractive runs
@app.function(image=extractive_image, gpu="A100-80GB", timeout=86400, volumes=VOLS, secrets=SECRETS)
def run_extractive(reader: str, tag: str, keys_json: str, protocol_sha: str):
    _setup()
    from images import runtime_versions
    from extractive import load_reader, predict
    from step3_common import DOC_STRIDE, MAX_ANSWER_LEN, EXTRACTIVE_BATCH_SIZE, SUBMISSION_CHUNK, f1_em
    import ehr_data as E
    P = _protocol(protocol_sha)
    from step3_common import extractive_spec
    pr = extractive_spec(P, reader)
    repo, sha, msl = pr["repo"], pr["sha"], pr["max_seq_len"]
    assert pr["doc_stride"] == DOC_STRIDE and P["readers"]["extractive"]["decoding"]["max_answer_len_tokens"] == MAX_ANSWER_LEN
    bs = EXTRACTIVE_BATCH_SIZE[reader]
    keys, sel = _keys(keys_json)
    df = E._dataset_by_key()
    t_start = time.time()
    tok, model, resolved = load_reader(repo, sha)
    run_meta = {"reader": reader, "tag": tag, "protocol_v1_3_sha256": protocol_sha, "resolved": resolved,
                "max_seq_len": msl, "doc_stride": DOC_STRIDE, "max_answer_len": MAX_ANSWER_LEN, "batch_size": bs,
                "dtype": resolved["model_dtype"], "tf32": False, "versions": runtime_versions(),
                "gpu": "A100-80GB x1", "conditions": {}}
    exp = _exp(f"Step3 {reader} [{tag}]", {"reader": reader, "repo": repo, "sha": sha, "tag": tag, "max_seq_len": msl,
                                            "doc_stride": DOC_STRIDE, "batch_size": bs, "gpu_type": "a100",
                                            "gpu_count": 1, "n_questions": len(keys)}, ["step3", "extractive", tag])
    yield ("meta", reader, "_start", -1, json.dumps(run_meta, default=str).encode())
    step = 0
    for condition in CONDITIONS:
        t_c = time.time()
        ctx, cmeta = _load_contexts(condition, sel)
        assert [tuple(x) for x in zip(ctx.person_id, ctx.qa_index)] == keys
        f1_acc = []
        for i in range(0, len(keys), SUBMISSION_CHUNK):
            ck = keys[i:i + SUBMISSION_CHUNK]
            path = _shard_path(tag, reader, condition, i // SUBMISSION_CHUNK)
            b = _existing_shard(path, ck, reader, condition)
            if b is None:
                sub = ctx.iloc[i:i + SUBMISSION_CHUNK]
                res = predict(tok, model, list(sub.question), list(sub.context), msl, DOC_STRIDE, MAX_ANSWER_LEN, bs)
                rows = []
                for (_, crow), x in zip(sub.iterrows(), res):
                    spans = json.loads(crow["visit_text_spans"])
                    sel_ids = json.loads(crow["selected_visit_ids"])
                    after = json.loads(crow["per_visit_tokens_after"])
                    reach = [v for v, (s, e), a in zip(sel_ids, spans, after) if a >= 1 and any(
                        ws < e and s < we for ws, we in x["windows_cover"])]
                    n_ctx_tok = len(tok(crow["context"], add_special_tokens=False)["input_ids"])
                    assert x["prediction"] is not None and len(x["prediction"]) > 0 or x["start_char"] == x["end_char"]
                    pf = {"prediction": x["prediction"], "answer_start_char": x["start_char"],
                          "answer_end_char": x["end_char"], "span_score": x["score"], "best_window": x["window"],
                          "reader_repo": repo, "reader_revision": sha, "protocol_version": "1.3", "run_tag": tag,
                          "context_file_sha256": cmeta["file_sha256"]}
                    rows.append(_out_row(crow, reader, condition, repo, sha, reach, pf, n_ctx_tok, x["n_windows"],
                                         DOC_STRIDE, msl))
                b = _write_shard(rows, path)
                for r_ in rows:
                    f1_acc.append(f1_em(r_["prediction"], str(df.loc[(r_["person_id"], r_["qa_index"])]["answer"]))[0])
            yield ("shard", reader, condition, i // SUBMISSION_CHUNK, b)
            step += 1
            _elog(exp, "log", {"sanity_token_f1_running": (sum(f1_acc) / len(f1_acc)) if f1_acc else 0.0,
                               "elapsed_min": (time.time() - t_start) / 60}, step=step)
            _elog(exp, "set_progress", int(100 * step / (len(CONDITIONS) * ((len(keys) + SUBMISSION_CHUNK - 1) // SUBMISSION_CHUNK))))
            print(f"[{reader}|{condition}] shard {i // SUBMISSION_CHUNK} done {time.time() - t_start:.0f}s", flush=True)
        run_meta["conditions"][condition] = {"seconds": time.time() - t_c, "context_file_sha256": cmeta["file_sha256"],
                                             "n": len(keys)}
    run_meta["seconds_total"] = time.time() - t_start
    hf.commit()
    _elog(exp, "finish", "completed")
    yield ("meta", reader, "_end", -1, json.dumps(run_meta, default=str).encode())


# ----------------------------------------------------------------------------------------------- generative runs
def _run_generative(reader, tag, keys_json, protocol_sha, gpu_type, gpu_count):
    _setup()
    import string
    from images import runtime_versions
    import generative as G
    from step3_common import SUBMISSION_CHUNK, f1_em
    import ehr_data as E
    P = _protocol(protocol_sha)
    pg = P["readers"]["generative"]
    ge = P["generative_engine"]
    repo, sha = pg[reader]["repo"], pg[reader]["sha"]
    # protocol <-> code consistency (engine values come from the protocol file)
    bp = ge["batching_policy"][reader]
    eng = G.ENGINE[reader]
    assert (bp["tensor_parallel_size"], bp["max_num_seqs"], bp["max_num_batched_tokens"], bp["gpu_memory_utilization"],
            bp["gpu_count"]) == (eng["tensor_parallel_size"], eng["max_num_seqs"], eng["max_num_batched_tokens"],
                                 eng["gpu_memory_utilization"], gpu_count), (bp, eng)
    assert ge["max_model_len"] == eng["max_model_len"] and ge["dtype"] == eng["dtype"]
    assert ge["enable_prefix_caching"] is False and eng["enable_prefix_caching"] is False
    assert ge["sampling_params"] == G.SAMPLING and ge["submission"]["chunk_size"] == SUBMISSION_CHUNK
    sys_p, user_t = pg["system_prompt"], pg["user_prompt_template"]
    fields = {f for _, f, _, _ in string.Formatter().parse(user_t) if f}
    assert fields == {"context", "question"}, fields  # (c) template never inserts the gold answer
    keys, sel = _keys(keys_json)
    df = E._dataset_by_key()
    t_start = time.time()
    G.predownload(repo, sha)
    hf.commit()
    t_load = time.time()
    llm, kw = G.build_engine(reader, repo, sha)
    tok = G.hf_tokenizer(repo, sha)
    sp = G.sampling_params()
    resolved = G.resolve_revisions(repo, sha, llm, tok)
    if not (resolved["model_PASS"] and resolved["tokenizer_PASS"]):
        raise RuntimeError(f"revision mismatch: {resolved}")
    rec0 = G.engine_record(llm, gpu_type, gpu_count)
    spr = G.sampling_record(sp)
    # runtime confirmations
    assert rec0["max_model_len"] == 8192, rec0["max_model_len"]
    assert "bfloat16" in rec0["dtype"], rec0["dtype"]
    assert rec0["enable_prefix_caching"] in (False, "False"), rec0["enable_prefix_caching"]
    assert rec0["tensor_parallel_size"] == eng["tensor_parallel_size"] and rec0["max_num_seqs"] == eng["max_num_seqs"]
    assert rec0["max_num_batched_tokens"] == eng["max_num_batched_tokens"]
    assert abs(float(rec0["gpu_memory_utilization"]) - 0.90) < 1e-9
    assert rec0["speculative_config"] in ("None", None), rec0["speculative_config"]
    assert spr["temperature"] == 0.0 and spr["top_p"] == 1.0 and spr["seed"] == 42 and spr["n"] == 1 \
        and spr["max_tokens"] == 256 and spr["top_k"] in (-1, 0), spr
    assert spr["sampling_type_is_GREEDY_enum"] is True and spr["sampling_type"] == "GREEDY", spr
    assert rec0["gpu_count_visible"] == gpu_count and "H100" in rec0["gpu_name"], rec0
    run_meta = {"reader": reader, "tag": tag, "protocol_v1_3_sha256": protocol_sha, "resolved": resolved,
                "engine_kwargs": kw, "engine_record_at_load": rec0, "sampling_record": spr,
                "versions": runtime_versions(), "load_seconds": time.time() - t_load, "conditions": {}}
    exp = _exp(f"Step3 {reader} vLLM [{tag}]", {"reader": reader, "repo": repo, "sha": sha, "tag": tag, **kw,
                                                  "gpu_type": "h100", "gpu_count": gpu_count, "n_questions": len(keys)},
               ["step3", "vllm", tag])
    yield ("meta", reader, "_start", -1, json.dumps(run_meta, default=str).encode())
    step = 0
    for condition in CONDITIONS:
        t_c = time.time()
        rec = G.engine_record(llm, gpu_type, gpu_count)
        assert rec["engine_config_hash"] == rec0["engine_config_hash"], "engine config changed between conditions"
        assert rec["vllm_config_repr_sha256"] == rec0["vllm_config_repr_sha256"]
        ctx, cmeta = _load_contexts(condition, sel)
        assert [tuple(x) for x in zip(ctx.person_id, ctx.qa_index)] == keys
        f1_acc, n_out_tok, n_in_tok = [], 0, 0
        for i in range(0, len(keys), SUBMISSION_CHUNK):
            ck = keys[i:i + SUBMISSION_CHUNK]
            path = _shard_path(tag, reader, condition, i // SUBMISSION_CHUNK)
            b = _existing_shard(path, ck, reader, condition)
            if b is None:
                sub = ctx.iloc[i:i + SUBMISSION_CHUNK]
                prompts, plen = [], []
                for q, c in zip(sub.question, sub.context):
                    _, ids = G.build_prompt_ids(tok, sys_p, user_t, c, q)
                    assert len(ids) + 256 <= 8192, len(ids)
                    prompts.append({"prompt_token_ids": ids})
                    plen.append(len(ids))
                t_g = time.time()
                outs = llm.generate(prompts, sp, use_tqdm=False)
                dt = time.time() - t_g
                assert len(outs) == len(prompts)
                rows = []
                for (_, crow), o, pl, pdict in zip(sub.iterrows(), outs, plen, prompts):
                    assert list(o.prompt_token_ids) == pdict["prompt_token_ids"]
                    sel_ids = json.loads(crow["selected_visit_ids"])
                    after = json.loads(crow["per_visit_tokens_after"])
                    reach = [v for v, a in zip(sel_ids, after) if a >= 1]
                    co = o.outputs[0]
                    pf = {"prediction": co.text.strip(), "raw_output": co.text, "finish_reason": str(co.finish_reason),
                          "n_output_tokens": len(co.token_ids), "reader_repo": repo, "reader_revision": sha,
                          "protocol_version": "1.3", "run_tag": tag, "context_file_sha256": cmeta["file_sha256"],
                          "engine_config_hash": rec["engine_config_hash"], "submission_chunk": i // SUBMISSION_CHUNK,
                          "chunk_generate_seconds": dt}
                    rows.append(_out_row(crow, reader, condition, repo, sha, reach, pf, pl, None, None, None))
                    n_out_tok += len(co.token_ids)
                    n_in_tok += pl
                b = _write_shard(rows, path)
                for r_ in rows:
                    f1_acc.append(f1_em(r_["prediction"], str(df.loc[(r_["person_id"], r_["qa_index"])]["answer"]))[0])
                print(f"[{reader}|{condition}] chunk {i // SUBMISSION_CHUNK} {len(prompts)} prompts {dt:.0f}s "
                      f"in_tok {sum(plen)} -> {(sum(plen)) / max(dt, 1e-9):.0f} tok/s", flush=True)
            yield ("shard", reader, condition, i // SUBMISSION_CHUNK, b)
            step += 1
            _elog(exp, "log", {"sanity_token_f1_running": (sum(f1_acc) / len(f1_acc)) if f1_acc else 0.0,
                               "elapsed_min": (time.time() - t_start) / 60}, step=step)
            _elog(exp, "set_progress", int(100 * step / (len(CONDITIONS) * ((len(keys) + SUBMISSION_CHUNK - 1) // SUBMISSION_CHUNK))))
        rec_end = G.engine_record(llm, gpu_type, gpu_count)
        assert rec_end["engine_config_hash"] == rec0["engine_config_hash"]
        run_meta["conditions"][condition] = {"engine_config_hash_start": rec["engine_config_hash"],
                                             "engine_config_hash_end": rec_end["engine_config_hash"],
                                             "vllm_config_repr_sha256": rec["vllm_config_repr_sha256"],
                                             "engine_record": {k: v for k, v in rec.items() if k != "vllm_config_repr"},
                                             "sampling_record": G.sampling_record(sp),
                                             "seconds": time.time() - t_c, "n": len(keys), "input_tokens": n_in_tok,
                                             "output_tokens": n_out_tok, "context_file_sha256": cmeta["file_sha256"]}
    run_meta["seconds_total"] = time.time() - t_start
    _elog(exp, "finish", "completed")
    yield ("meta", reader, "_end", -1, json.dumps(run_meta, default=str).encode())


@app.function(image=vllm_image, gpu="H100!", timeout=86400, volumes=VOLS, secrets=SECRETS, memory=65536)
def run_qwen7b(tag: str, keys_json: str, protocol_sha: str):
    yield from _run_generative("qwen2_5_7b", tag, keys_json, protocol_sha, "H100!", 1)


@app.function(image=vllm_image, gpu="H100!:2", timeout=86400, volumes=VOLS, secrets=SECRETS, memory=131072)
def run_qwen32b(tag: str, keys_json: str, protocol_sha: str):
    yield from _run_generative("qwen2_5_32b", tag, keys_json, protocol_sha, "H100!", 2)


# ----------------------------------------------------------------------------------------------- local entrypoint
@app.local_entrypoint()
def main(mode: str, reader: str = "", tag: str = "pilot", keys_file: str = "", protocol_sha: str = ""):
    import hashlib
    proto_local = os.path.join(ROOT, "shared", "canonical", "protocol_v1.3.yaml")  # release: WORKDIR/.. -> ROOT
    psha = hashlib.sha256(open(proto_local, "rb").read()).hexdigest()
    assert not protocol_sha or protocol_sha == psha
    keys_json = open(keys_file).read() if keys_file else ""
    res_dir = os.path.join(WORKDIR, "results")
    if mode == "prechecks":
        out = prechecks.remote(psha)
        json.dump(out, open(os.path.join(res_dir, "prechecks_ab.json"), "w"), indent=1, default=str)
        for r, v in out["readers"].items():
            print(r, v["load_check_PASS"], v["model_PASS"], v["tokenizer_PASS"], round(v["squad_EM"], 2), round(v["squad_F1"], 2))
        return
    if mode == "extractive":
        gen = run_extractive.remote_gen(reader, tag, keys_json, psha)
    elif mode == "generative":
        fn = {"qwen2_5_7b": run_qwen7b, "qwen2_5_32b": run_qwen32b}[reader]
        gen = fn.remote_gen(tag, keys_json, psha)
    else:
        raise ValueError(mode)
    for kind, rd, cond, part, payload in gen:
        d = os.path.join(res_dir, "runs", tag, rd)
        os.makedirs(os.path.join(d, cond) if kind == "shard" else d, exist_ok=True)
        if kind == "shard":
            p = os.path.join(d, cond, f"part_{part:03d}.parquet")
        else:
            p = os.path.join(d, f"run_meta{cond}.json")
        with open(p + ".tmp", "wb") as f:
            f.write(payload)
        os.replace(p + ".tmp", p)
        print(time.strftime("%H:%M:%S"), "synced", p, len(payload), flush=True)
