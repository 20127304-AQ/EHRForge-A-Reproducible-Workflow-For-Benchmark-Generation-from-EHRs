"""Extractive QA reader (ONE code path for roberta_base_squad2, biobert_v1_1_pubmed_squad_v2, longformer_squadv2).

Written from scratch for Step 3 (no logic from older reader scripts).
* load at the pinned revision (model AND tokenizer: revision=<sha>), capture all load-time log/warning output,
  FAIL on 'newly initialized' / qa_outputs messages, resolve the actually-loaded commit (config._commit_hash and HF
  cache snapshot path of every tokenizer/model file) and FAIL if != pinned.
* sliding windows over the FULL context: tokenizer(question, context, truncation='only_second',
  max_length=max_seq_len, stride=128, return_overflowing_tokens=True, return_offsets_mapping=True).
  doc_stride=128 uses HF `stride` semantics (= number of context tokens shared by consecutive windows), as in the
  transformers question-answering pipeline.
* span selection: exact argmax over (i, j) of start_logit[i] + end_logit[j], i <= j, j - i + 1 <= 64 tokens,
  i and j restricted to CONTEXT tokens of the window (CLS / question / special / padding excluded => no null answer),
  best over ALL windows of the example (ties: earliest window, then smallest i, then smallest length).
* fixed batch size (windows per forward pass) per reader; float32 weights (from_pretrained default), TF32 disabled.
"""
import io
import logging
import os
import re
import sys
import warnings

BAD_LOAD_PATTERNS = [r"newly initialized", r"qa_outputs"]


class LoadCheckError(RuntimeError):
    pass


def _snapshot_commit(path):
    m = re.search(r"/snapshots/([0-9a-f]{40})/", path)
    return m.group(1) if m else None


def load_reader(repo, sha, device="cuda"):
    import torch
    import transformers
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    from transformers.utils import cached_file
    from huggingface_hub import scan_cache_dir

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # capture everything emitted while loading
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    tlog = transformers.utils.logging.get_logger("transformers")
    transformers.utils.logging.set_verbosity_info()
    tlog.addHandler(handler)
    root_handler = logging.StreamHandler(buf)
    logging.getLogger().addHandler(root_handler)
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        tok = AutoTokenizer.from_pretrained(repo, revision=sha, use_fast=True)
        model = AutoModelForQuestionAnswering.from_pretrained(repo, revision=sha)
    tlog.removeHandler(handler)
    logging.getLogger().removeHandler(root_handler)
    transformers.utils.logging.set_verbosity_warning()
    log_text = buf.getvalue() + "\n".join(f"WARNING {w.category.__name__}: {w.message}" for w in wrec)
    bad = [ln for ln in log_text.splitlines() if any(re.search(p, ln, re.I) for p in BAD_LOAD_PATTERNS)]
    if not tok.is_fast:
        raise LoadCheckError(f"{repo}: fast tokenizer required for offsets")
    # resolved revisions
    model_commit = getattr(model.config, "_commit_hash", None)
    tok_files = {}
    for fn in ["tokenizer_config.json", "tokenizer.json", "vocab.txt", "vocab.json", "merges.txt",
               "special_tokens_map.json", "config.json"]:
        try:
            p = cached_file(repo, fn, revision=sha, _raise_exceptions_for_missing_entries=False)
        except Exception:
            p = None
        if p:
            tok_files[fn] = _snapshot_commit(p)
    tok_commits = sorted({c for fn, c in tok_files.items() if fn != "config.json"})
    snaps = []
    for r in scan_cache_dir().repos:
        if r.repo_id == repo:
            snaps = sorted(rv.commit_hash for rv in r.revisions)
    weight_paths = {}
    for fn in ["model.safetensors", "pytorch_model.bin"]:
        p = cached_file(repo, fn, revision=sha, _raise_exceptions_for_missing_entries=False)
        if p:
            weight_paths[fn] = _snapshot_commit(p)
    model_commits = sorted({model_commit} | set(weight_paths.values()) - {None})
    resolved = {
        "repo": repo, "pinned": sha,
        "model_commit_config_hash": model_commit, "model_weight_files_commit": weight_paths,
        "tokenizer_files_commit": tok_files,
        "model_resolved": model_commits[0] if len(model_commits) == 1 else model_commits,
        "tokenizer_resolved": tok_commits[0] if len(tok_commits) == 1 else tok_commits,
        "hf_cache_snapshots_for_repo": snaps,
        "tokenizer_class": type(tok).__name__, "model_class": type(model).__name__,
        "model_dtype": str(next(model.parameters()).dtype),
    }
    resolved["model_PASS"] = resolved["model_resolved"] == sha
    resolved["tokenizer_PASS"] = resolved["tokenizer_resolved"] == sha
    resolved["load_warnings_bad_lines"] = bad
    resolved["load_check_PASS"] = not bad
    resolved["load_log"] = log_text[-20000:]
    if not (resolved["model_PASS"] and resolved["tokenizer_PASS"]):
        raise LoadCheckError(f"{repo}: resolved revision mismatch {resolved}")
    if bad:
        raise LoadCheckError(f"{repo}: bad load warnings: {bad}")
    model.to(device).eval()
    return tok, model, resolved


def featurize(tok, questions, contexts, max_seq_len, doc_stride):
    enc = tok(questions, contexts, truncation="only_second", max_length=max_seq_len, stride=doc_stride,
              return_overflowing_tokens=True, return_offsets_mapping=True, padding=False)
    feats = []
    for w in range(len(enc["input_ids"])):
        seq_ids = enc.sequence_ids(w)
        ctx_mask = [1 if s == 1 else 0 for s in seq_ids]
        feats.append({"ex": enc["overflow_to_sample_mapping"][w], "input_ids": enc["input_ids"][w],
                      "attention_mask": enc["attention_mask"][w],
                      "token_type_ids": enc["token_type_ids"][w] if "token_type_ids" in enc else None,
                      "offsets": enc["offset_mapping"][w], "ctx_mask": ctx_mask})
    return feats


def best_spans_batch(start, end, ctx_mask, max_answer_len):
    """start/end: [B, L] float; ctx_mask: [B, L] bool. Exact best (i, j) per row with i<=j<i+max_answer_len,
    both context tokens. Returns (score[B], i[B], j[B]) tensors."""
    import torch
    NEG = torch.finfo(start.dtype).min / 4
    s = start.masked_fill(~ctx_mask, NEG)
    e = end.masked_fill(~ctx_mask, NEG)
    B, L = s.shape
    best = torch.full((B,), NEG * 4, device=s.device, dtype=s.dtype)
    bi = torch.zeros(B, dtype=torch.long, device=s.device)
    bj = torch.zeros(B, dtype=torch.long, device=s.device)
    for d in range(max_answer_len):  # length d+1; ascending d keeps the shortest span on ties
        if d >= L:
            break
        sc = s[:, :L - d] + e[:, d:]
        v, i = sc.max(dim=1)  # first max index on ties -> smallest i
        upd = v > best
        best = torch.where(upd, v, best)
        bi = torch.where(upd, i, bi)
        bj = torch.where(upd, i + d, bj)
    return best, bi, bj


def predict(tok, model, questions, contexts, max_seq_len, doc_stride, max_answer_len, batch_size, device="cuda"):
    """Returns list of dicts per example: prediction, char span, score, window index, n_windows, window char cover."""
    import torch
    feats = featurize(tok, questions, contexts, max_seq_len, doc_stride)
    n = len(questions)
    res = [{"score": None, "prediction": None, "start_char": None, "end_char": None, "window": None,
            "n_windows": 0, "windows_cover": []} for _ in range(n)]
    pad_id = tok.pad_token_id
    win_count = [0] * n
    for b0 in range(0, len(feats), batch_size):
        fb = feats[b0:b0 + batch_size]
        L = max(len(f["input_ids"]) for f in fb)
        ids = torch.full((len(fb), L), pad_id, dtype=torch.long)
        am = torch.zeros((len(fb), L), dtype=torch.long)
        cm = torch.zeros((len(fb), L), dtype=torch.bool)
        tt = torch.zeros((len(fb), L), dtype=torch.long) if fb[0]["token_type_ids"] is not None else None
        for r, f in enumerate(fb):
            k = len(f["input_ids"])
            ids[r, :k] = torch.tensor(f["input_ids"])
            am[r, :k] = torch.tensor(f["attention_mask"])
            cm[r, :k] = torch.tensor(f["ctx_mask"], dtype=torch.bool)
            if tt is not None:
                tt[r, :k] = torch.tensor(f["token_type_ids"])
        kw = {"input_ids": ids.to(device), "attention_mask": am.to(device)}
        if tt is not None:
            kw["token_type_ids"] = tt.to(device)
        with torch.inference_mode():
            out = model(**kw)
        sc, bi, bj = best_spans_batch(out.start_logits.float(), out.end_logits.float(), cm.to(device), max_answer_len)
        sc, bi, bj = sc.cpu().tolist(), bi.cpu().tolist(), bj.cpu().tolist()
        for r, f in enumerate(fb):
            ex = f["ex"]
            w = win_count[ex]
            win_count[ex] += 1
            ctx_offs = [o for o, m in zip(f["offsets"], f["ctx_mask"]) if m]
            if ctx_offs:
                res[ex]["windows_cover"].append([ctx_offs[0][0], ctx_offs[-1][1]])
            if not ctx_offs:
                continue
            if res[ex]["score"] is None or sc[r] > res[ex]["score"]:
                s_char = f["offsets"][bi[r]][0]
                e_char = f["offsets"][bj[r]][1]
                res[ex].update({"score": sc[r], "start_char": s_char, "end_char": e_char, "window": w,
                                "start_tok": bi[r], "end_tok": bj[r]})
    for ex in range(n):
        res[ex]["n_windows"] = win_count[ex]
        if res[ex]["score"] is None:
            raise RuntimeError(f"example {ex}: no context tokens in any window")
        res[ex]["prediction"] = contexts[ex][res[ex]["start_char"]:res[ex]["end_char"]]
    return res
