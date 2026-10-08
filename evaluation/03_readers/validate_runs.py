"""Validate synced reader runs (pilot or full). Local, light (pandas/pyarrow only).
usage: python validate_runs.py <tag> [n_expected_per_run]
Checks: every (reader, condition) present; n rows == expected; resume key unique, no missing (vs expected keys);
context_metadata schema (protocol v1.1 fields) + prediction; context_sha256 identical across the 5 readers per
(key, condition) and equal to the context build; non-empty predictions; extractive window fields; generative
reader_input_tokens + 256 <= 8192; revision PASS; engine config hash equal across the 5 conditions per Qwen model;
runtime dtype/max_model_len/sampling values. Writes results/validation_<tag>.json.
"""
import glob
import json
import os
import sys

import pandas as pd

import os as _os, sys as _sys  # release adaptation (paths/config only)
_HERE = _os.path.dirname(_os.path.abspath(__file__))
_sys.path.insert(0, _os.path.dirname(_HERE))  # evaluation/ (release_config.py)
import release_config as _rc  # noqa: E402  reads evaluation/config.yaml
_rc.enter_workspace()  # inputs resolve relative to WORKSPACE_DIR (original project layout)
W = _rc.step_out("03_readers")  # release: was the agent work directory
sys.path.insert(0, _HERE)
from step3_common import CONDITIONS, EXTRACTIVE, GENERATIVE  # noqa: E402

FIELDS = ["person_id", "qa_index", "reader", "condition", "n_gold_visits", "gold_visit_ids", "n_selected_visits",
          "selected_visit_ids", "n_visits_reaching_model", "visit_ids_reaching_model", "total_tokens_before",
          "total_tokens_after", "per_visit_tokens_before", "per_visit_tokens_after", "header_separator_tokens",
          "truncated", "n_gold_visits_surviving", "gold_fully_survived", "gold_tokens_before", "gold_tokens_after",
          "budget_tokens", "tokenizer_name", "tokenizer_revision", "reader_tokenizer_name", "reader_tokenizer_revision",
          "reader_input_tokens", "n_windows", "window_stride", "max_seq_len", "context_sha256", "prediction"]


def main(tag, n_exp=None):
    readers = EXTRACTIVE + GENERATIVE
    if tag == "pilot":
        keys = sorted(tuple(k) for k in json.load(open(os.path.join(W, "results", "pilot_keys.json"))))
    else:
        keys = None
    n_exp = n_exp or (len(keys) if keys else 10742)
    rep = {"tag": tag, "n_expected_per_run": n_exp, "runs": {}, "errors": [], "engine_hash_tables": {},
           "resolved": {}, "runtime": {}}
    frames = []
    for r in readers:
        for c in CONDITIONS:
            files = sorted(glob.glob(os.path.join(W, "results", "runs", tag, r, c, "part_*.parquet")))
            if not files:
                rep["errors"].append(f"missing run {r}/{c}")
                continue
            d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            frames.append(d)
            e = []
            miss = [f for f in FIELDS if f not in d.columns]
            if miss:
                e.append(f"missing fields {miss}")
            if len(d) != n_exp:
                e.append(f"rows {len(d)} != {n_exp}")
            if d.duplicated(["person_id", "qa_index"]).any():
                e.append("duplicate keys")
            ks = sorted(zip(d.person_id.astype(int), d.qa_index.astype(int)))
            if keys is not None and ks != keys:
                e.append("key set != pilot keys")
            if set(d.reader) != {r} or set(d.condition) != {c}:
                e.append("reader/condition label mismatch")
            if d.prediction.isna().any():
                e.append("null predictions")
            emp = d[d.prediction.fillna("").str.len() == 0]
            if len(emp):  # reported separately (zero-width whitespace-token span), not a structural error
                rep.setdefault("empty_string_predictions", {})[f"{r}|{c}"] = [
                    [int(a), int(b)] for a, b in zip(emp.person_id, emp.qa_index)]
            if (d.budget_tokens != 3500).any() or (d.total_tokens_after > 3500).any():
                e.append("budget violation")
            if not (d.n_visits_reaching_model <= d.n_selected_visits).all():
                e.append("reaching > selected")
            if r in EXTRACTIVE:
                if not ((d.n_windows >= 1) & (d.window_stride == 128)).all():
                    e.append("window fields")
            else:
                if d.n_windows.notna().any() or not (d.reader_input_tokens + 256 <= 8192).all():
                    e.append("generative fields")
                if "finish_reason" in d:
                    rep.setdefault("finish_reasons", {})[f"{r}|{c}"] = d.finish_reason.value_counts().to_dict()
            rep["runs"][f"{r}|{c}"] = {"n": len(d), "errors": e, "n_shards": len(files),
                                       "mean_pred_chars": float(d.prediction.str.len().mean()),
                                       "gold_fully_survived_rate": float(d.gold_fully_survived.mean()),
                                       "n_gold_surviving_eq_n_gold_rate": float((d.n_gold_visits_surviving == d.n_gold_visits).mean())}
            rep["errors"] += [f"{r}|{c}: {x}" for x in e]
    allp = pd.concat(frames, ignore_index=True)
    dup = allp.duplicated(["person_id", "qa_index", "reader", "condition"]).sum()
    rep["total_rows"] = len(allp)
    rep["total_expected"] = n_exp * len(readers) * len(CONDITIONS)
    rep["duplicate_resume_keys"] = int(dup)
    if rep["total_rows"] != rep["total_expected"] or dup:
        rep["errors"].append("total/duplicate check failed")
    # same context across readers
    g = allp.groupby(["person_id", "qa_index", "condition"]).context_sha256.nunique()
    rep["context_sha_distinct_max"] = int(g.max())
    if g.max() != 1:
        rep["errors"].append("context sha differs across readers")
    # context sha equals context build
    # run meta: revisions, engine hashes
    for r in readers:
        ms = os.path.join(W, "results", "runs", tag, r, "run_meta_start.json")
        me = os.path.join(W, "results", "runs", tag, r, "run_meta_end.json")
        if not os.path.exists(me):
            rep["errors"].append(f"{r}: run_meta_end missing")
            continue
        m = json.load(open(me))
        res = m["resolved"]
        rep["resolved"][r] = {k: res.get(k) for k in ["repo", "pinned", "model_resolved", "tokenizer_resolved",
                                                     "model_PASS", "tokenizer_PASS", "hf_cache_snapshots_for_repo",
                                                     "engine_model_revision", "engine_tokenizer_revision",
                                                     "hf_config_commit_hash", "load_check_PASS"] if k in res}
        if not (res["model_PASS"] and res["tokenizer_PASS"]):
            rep["errors"].append(f"{r}: revision FAIL")
        if r in GENERATIVE:
            hashes = {c: (v["engine_config_hash_start"], v["engine_config_hash_end"], v["vllm_config_repr_sha256"])
                      for c, v in m["conditions"].items()}
            rep["engine_hash_tables"][r] = hashes
            if len({h for t in hashes.values() for h in t[:2]}) != 1 or len({t[2] for t in hashes.values()}) != 1 \
                    or set(hashes) != set(CONDITIONS):
                rep["errors"].append(f"{r}: engine config hash differs between conditions")
            er = m["engine_record_at_load"]
            sr = m["sampling_record"]
            rep["runtime"][r] = {k: er[k] for k in ["dtype", "max_model_len", "seed", "max_num_seqs",
                                                    "max_num_batched_tokens", "gpu_memory_utilization",
                                                    "enable_prefix_caching", "enable_chunked_prefill",
                                                    "tensor_parallel_size", "speculative_config", "gpu_name",
                                                    "gpu_count_visible", "num_gpu_blocks", "async_scheduling",
                                                    "engine_config_hash"]}
            rep["runtime"][r]["sampling"] = {k: sr[k] for k in ["temperature", "top_p", "top_k", "max_tokens", "seed",
                                                                "n", "sampling_type", "sampling_type_is_GREEDY_enum"]}
            rep["runtime"][r]["seconds_total"] = m.get("seconds_total")
            rep["runtime"][r]["per_condition_seconds"] = {c: v["seconds"] for c, v in m["conditions"].items()}
            rep["runtime"][r]["versions"] = {k: m["versions"].get(k) for k in ["vllm", "torch", "transformers", "torch_cuda", "nvidia_smi"]}
        else:
            rep["runtime"][r] = {"batch_size": m["batch_size"], "max_seq_len": m["max_seq_len"], "doc_stride": m["doc_stride"],
                                 "max_answer_len": m["max_answer_len"], "dtype": m["dtype"], "seconds_total": m.get("seconds_total"),
                                 "gpu": m["gpu"]}
    rep["PASS"] = not rep["errors"]
    json.dump(rep, open(os.path.join(W, "results", f"validation_{tag}.json"), "w"), indent=1, default=str)
    print(json.dumps({k: rep[k] for k in ["PASS", "errors", "total_rows", "total_expected", "duplicate_resume_keys",
                                          "context_sha_distinct_max"]}, indent=1))
    return rep


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
