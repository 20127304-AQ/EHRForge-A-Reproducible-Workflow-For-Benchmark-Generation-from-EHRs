"""EHRForge protocol v1.1 full-depth retrieval (agent 72af8248).

Stages (each resumable; every output is committed to the Modal Volume and copied back to the workdir):
  embed   : MedCPT / NV-Embed-v2 visit + query embeddings, sharded by patient (GPU, 1x A100-80GB per container)
  rank    : full-depth rankings for bm25 / medcpt / nvembed_v2 x {full_timeline, sampled_window} (CPU)
  hybrid  : hybrid_rrf60 from the NEW v1.1 bm25 + medcpt ranked files of the same search space (CPU)
  validate: every retrieval_output invariant of protocol v1.1 (CPU)

Run from the workdir, e.g.
  modal run code/retrieval_v11.py::embed --model medcpt [--limit-shards 1]
  modal run code/retrieval_v11.py::rank --retriever bm25 [--smoke]
"""
import hashlib
import json
import os
import subprocess
import sys
import time

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
app = modal.App(os.environ.get("EHRFORGE_STEP1_APP_RETRIEVAL", "ehrforge-v11-retrieval"))
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name(os.environ.get("EHRFORGE_HF_CACHE_VOLUME", "huggingface-cache"), create_if_missing=True)
HF = modal.Secret.from_name(os.environ.get("EHRFORGE_HF_SECRET", "huggingface-secret"))

RUN_REL = "runs/v1.1"
RUN = "/vol/" + RUN_REL
NV_REPO, NV_SHA = "nvidia/NV-Embed-v2", "3fa59658547db50a1e8e3346cf057fd0c77ed6ef"
MEDCPT_Q = ("ncbi/MedCPT-Query-Encoder", "d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc")
MEDCPT_A = ("ncbi/MedCPT-Article-Encoder", "d05a736da4bb84ee4057b7f7999485be6ed85465")
QUERY_INSTR = "Instruct: Given a clinical question, retrieve patient visit notes that answer the question\nQuery: "
N_SHARDS = {"nvembed_v2": 32, "medcpt": 8}
SPACES = ("full_timeline", "sampled_window")
EHR_DATA_SHA = "ab3f1387100a8d512ab5154ec402ea5becb009d5ab7406d3723f8f2ec39172de"
PROTOCOL_SHA = "6a4d381ba818925707cdf65a42cc111af87de78cc976df1b55efd6e6eb466be1"
WORKDIR = _rc.step_out("01_retrieval") if _rc else os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # release: outputs under OUT_DIR


# ============================================================================ helpers (run remotely)
def _versions():
    import platform
    from importlib.metadata import version as _v
    out = {"python": platform.python_version()}
    for m in ["torch", "transformers", "sentence-transformers", "numpy", "tokenizers", "rank-bm25", "datasets",
              "pyarrow", "pandas", "accelerate", "einops"]:
        try:
            out[m] = _v(m)
        except Exception as e:
            out[m] = f"NOT_INSTALLED ({type(e).__name__})"
    return out


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _E():
    """ehr_data from the volume mirror of shared/canonical (sha-asserted)."""
    os.environ["EHRFORGE_ROOT"] = "/vol"
    p = "/vol/shared/canonical/ehr_data.py"
    assert _sha(p) == EHR_DATA_SHA, "ehr_data.py on volume != canonical sha"
    if "/vol/shared/canonical" not in sys.path:
        sys.path.insert(0, "/vol/shared/canonical")
    import ehr_data as E
    assert E.sha256_file("/vol/shared/canonical/protocol_v1.1.yaml") == PROTOCOL_SHA
    return E


def _keys(E):
    df = E.load_dataset()
    return sorted(zip(df["person_id"].astype(int), df["qa_index"].astype(int)))


def shard_plan(E, n):
    """Deterministic patient sharding balanced by total visit characters."""
    C = E.load_corpus()
    pids = sorted({p for p, _ in _keys(E)})
    w = {p: sum(len(v["text"]) for v in C[p]) + 1 for p in pids}
    loads, shards = [0] * n, [[] for _ in range(n)]
    for p in sorted(pids, key=lambda p: (-w[p], p)):
        k = min(range(n), key=lambda i: (loads[i], i))
        shards[k].append(p)
        loads[k] += w[p]
    return [sorted(s) for s in shards]


def _shard_inputs(E, k, n):
    C = E.load_corpus()
    pids = shard_plan(E, n)[k]
    pset = set(pids)
    vkeys = [(p, i) for p in pids for i in range(len(C[p]))]
    vtexts = [C[p][i]["text"] for p, i in vkeys]
    qkeys = [key for key in _keys(E) if key[0] in pset]
    qtexts = [str(E.get_row(*key)["question"]) for key in qkeys]
    return pids, vkeys, vtexts, qkeys, qtexts


def _batches_by_tokens(lengths, max_tokens, max_bs):
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    out, cur = [], []
    for i in order:
        if cur and ((len(cur) + 1) * lengths[cur[0]] > max_tokens or len(cur) >= max_bs):
            out.append(cur)
            cur = []
        cur.append(i)
    if cur:
        out.append(cur)
    return out


def _save_npz(path, **arrays):
    import numpy as np
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


# ============================================================================ embedding (GPU)
@app.cls(image=nvembed_image, gpu="A100-80GB", timeout=4 * 3600, max_containers=4,
         volumes={"/vol": vol, "/root/.cache/huggingface": hf_cache}, secrets=[HF])
class NVEmbed:
    @modal.enter()
    def load(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.E = _E()
        self.model = AutoModel.from_pretrained(NV_REPO, revision=NV_SHA, trust_remote_code=True).cuda().eval()
        self.dtype = str(next(self.model.parameters()).dtype)
        assert self.dtype == "torch.float32", self.dtype  # approved: fp32 (from_pretrained default)
        self.tok = AutoTokenizer.from_pretrained(NV_REPO, revision=NV_SHA)

    def _enc(self, texts, instr):
        import numpy as np
        import torch
        import torch.nn.functional as F
        full = [len(x) for x in self.tok(texts, add_special_tokens=True)["input_ids"]]
        lens = [min(4096, x) for x in full]
        out = np.zeros((len(texts), 4096), np.float32)
        with torch.no_grad():
            for b in _batches_by_tokens(lens, 16384, 32):
                e = self.model.encode([texts[i] for i in b], instruction=instr, max_length=4096)
                out[b] = F.normalize(e.float(), p=2, dim=1).cpu().numpy()
        return out, full

    @modal.method()
    def shard(self, k, n):
        import numpy as np
        path = f"{RUN}/emb/nvembed_v2/shard_{k:03d}.npz"
        vol.reload()
        if os.path.exists(path):
            return {"k": k, "skipped": True, "path": path}
        t0 = time.time()
        pids, vkeys, vtexts, qkeys, qtexts = _shard_inputs(self.E, k, n)
        ve, vfull = self._enc(vtexts, "")
        qe, qfull = self._enc(qtexts, QUERY_INSTR)
        st = {"k": k, "n_shards": n, "n_patients": len(pids), "n_visits": len(vkeys), "n_queries": len(qkeys),
              "visits_over_4096_tokens": int(sum(x > 4096 for x in vfull)), "visit_tokens_sum": int(sum(vfull)),
              "queries_over_4096_tokens": int(sum(x > 4096 for x in qfull)), "dtype": self.dtype,
              "seconds": time.time() - t0, "versions": _versions(), "model": NV_REPO, "revision": NV_SHA,
              "batching": "length-sorted desc, <=16384 padded tokens and <=32 texts per batch",
              "encode": "model.encode(texts, instruction=<query_instruction|''>, max_length=4096); F.normalize(p=2)"}
        _save_npz(path, visit_keys=np.array(vkeys, np.int64), visit_emb=ve, query_keys=np.array(qkeys, np.int64),
                  query_emb=qe, stats=np.array(json.dumps(st)))
        vol.commit()
        return {**{x: st[x] for x in ("k", "n_visits", "n_queries", "seconds", "visits_over_4096_tokens")}, "path": path}


@app.cls(image=main_image, gpu="A100-80GB", timeout=4 * 3600, max_containers=1,
         volumes={"/vol": vol, "/root/.cache/huggingface": hf_cache}, secrets=[HF])
class MedCPT:
    @modal.enter()
    def load(self):
        from transformers import AutoModel, AutoTokenizer
        self.E = _E()
        self.qtok = AutoTokenizer.from_pretrained(MEDCPT_Q[0], revision=MEDCPT_Q[1])
        self.qm = AutoModel.from_pretrained(MEDCPT_Q[0], revision=MEDCPT_Q[1]).cuda().eval()
        self.atok = AutoTokenizer.from_pretrained(MEDCPT_A[0], revision=MEDCPT_A[1])
        self.am = AutoModel.from_pretrained(MEDCPT_A[0], revision=MEDCPT_A[1]).cuda().eval()
        self.dtype = str(next(self.am.parameters()).dtype)

    @modal.method()
    def shard(self, k, n):
        import numpy as np
        import torch
        path = f"{RUN}/emb/medcpt/shard_{k:03d}.npz"
        vol.reload()
        if os.path.exists(path):
            return {"k": k, "skipped": True, "path": path}
        t0 = time.time()
        pids, vkeys, vtexts, qkeys, qtexts = _shard_inputs(self.E, k, n)
        vfull = [len(x) for x in self.atok([["", t] for t in vtexts], truncation=False)["input_ids"]]
        qfull = [len(x) for x in self.qtok(qtexts, truncation=False)["input_ids"]]
        ve = np.zeros((len(vtexts), 768), np.float32)
        qe = np.zeros((len(qtexts), 768), np.float32)
        with torch.no_grad():
            for b in _batches_by_tokens([min(512, x) for x in vfull], 65536, 128):
                enc = self.atok([["", vtexts[i]] for i in b], truncation=True, padding=True, return_tensors="pt",
                                max_length=512).to("cuda")
                ve[b] = self.am(**enc).last_hidden_state[:, 0, :].float().cpu().numpy()
            for i in range(0, len(qtexts), 256):
                enc = self.qtok(qtexts[i:i + 256], truncation=True, padding=True, return_tensors="pt",
                                max_length=64).to("cuda")
                qe[i:i + 256] = self.qm(**enc).last_hidden_state[:, 0, :].float().cpu().numpy()
        st = {"k": k, "n_shards": n, "n_patients": len(pids), "n_visits": len(vkeys), "n_queries": len(qkeys),
              "visits_truncated_512": int(sum(x > 512 for x in vfull)),
              "queries_truncated_64": int(sum(x > 64 for x in qfull)), "dtype": self.dtype,
              "seconds": time.time() - t0, "versions": _versions(),
              "query_encoder": list(MEDCPT_Q), "article_encoder": list(MEDCPT_A),
              "batching": "visits length-sorted desc, <=65536 padded tokens and <=128 per batch; queries 256 in key order",
              "encode": "CLS last_hidden_state[:,0]; article input [['', text]] max_length 512; query max_length 64"}
        _save_npz(path, visit_keys=np.array(vkeys, np.int64), visit_emb=ve, query_keys=np.array(qkeys, np.int64),
                  query_emb=qe, stats=np.array(json.dumps(st)))
        vol.commit()
        return {**{x: st[x] for x in ("k", "n_visits", "n_queries", "seconds", "visits_truncated_512")}, "path": path}


def _copy_back(rel):
    dst = os.path.join(WORKDIR, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    r = subprocess.run(["modal", "volume", "get", "--force", VOLUME_NAME, "/" + rel, dst], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"copy-back failed for {rel}: {r.stderr[-500:]}")
    return dst


def _experiment(name, desc, config, x_label):
    try:
        sys.path.insert(0, "/app")
        from src.orchestra_sdk.experiment import Experiment
        return Experiment.init(name=name, description=desc, config=config, x_axis_label=x_label)
    except Exception as e:  # tracking must never block the run
        print("experiment tracking unavailable:", repr(e))
        return None


@app.local_entrypoint()
def embed(model: str = "medcpt", limit_shards: int = 0):
    n = N_SHARDS[model]
    ks = list(range(n))[: limit_shards or n]
    exp = _experiment(f"EHRForge v1.1 {model} embeddings", f"{model} visit+query embeddings, {len(ks)}/{n} shards",
                      {"model": model, "n_shards": n, "run_shards": len(ks), "gpu_type": "a100", "gpu_count": 1,
                       "dtype": "float32", "protocol_sha256": PROTOCOL_SHA}, "Shards done")
    if exp:
        exp.set_metadata({"platform": "Modal", "gpu_spec": "1x A100-80GB per container"})
    obj = NVEmbed() if model == "nvembed_v2" else MedCPT()
    done, failed = 0, []
    for r in obj.shard.map(ks, kwargs={"n": n}, order_outputs=False, return_exceptions=True):
        if isinstance(r, Exception):
            failed.append(repr(r)[:500]); print("SHARD FAILED", repr(r)[:500], flush=True)
            continue
        rel = r["path"][len("/vol/"):]
        _copy_back(rel)
        done += 1
        print("SHARD", json.dumps(r), f"copied -> {rel}", flush=True)
        if exp:
            exp.log({"visits_per_s": r["n_visits"] / r["seconds"] if not r.get("skipped") else 0.0,
                     "shard_seconds": r.get("seconds", 0.0)}, step=done)
            exp.set_progress(int(100 * done / len(ks)))
    print(f"EMBED DONE {model}: {done}/{len(ks)} ok, {len(failed)} failed")
    if exp:
        exp.finish("completed" if not failed else "failed")
    if failed:
        raise SystemExit(f"{len(failed)} shards failed: {failed[:2]}")


# ============================================================================ ranking (CPU)
def _prefix(smoke):
    return f"{RUN}/smoke" if smoke else RUN


def _paths(retriever, space, smoke):
    base = f"{_prefix(smoke)}/rankings/{retriever}@{space}"
    return base + ".header.parquet", base + ".ranked.parquet", base + ".meta.json"


def _image_versions_record():
    rec = {"main_image": _versions()}
    for p in [f"{RUN}/emb/nvembed_v2/shard_000.npz"]:
        if os.path.exists(p):
            import numpy as np
            rec["nvembed_image"] = json.loads(str(np.load(p)["stats"]))["versions"]
    if "nvembed_image" not in rec and os.path.exists("/vol/meta/nvembed_smoke.json"):
        rec["nvembed_image"] = json.load(open("/vol/meta/nvembed_smoke.json"))["versions"]
    return rec


def _write(retriever, space, smoke, header_cols, ranked_cols, meta):
    import pyarrow as pa
    import pyarrow.parquet as pq
    hp, rp, mp = _paths(retriever, space, smoke)
    os.makedirs(os.path.dirname(hp), exist_ok=True)
    n = len(header_cols["person_id"])
    hschema = pa.schema([
        ("person_id", pa.int64()), ("qa_index", pa.int64()), ("retriever", pa.string()), ("search_space", pa.string()),
        ("timeline_sampling_strategy", pa.string()), ("n_visits_patient", pa.int32()), ("n_candidates", pa.int32()),
        ("candidate_visit_ids", pa.list_(pa.int32())), ("gold_visit_ids", pa.list_(pa.int32())),
        ("n_gold_visits", pa.int16()), ("ranking_depth", pa.int32()), ("score_native", pa.bool_()),
        ("inputs_source", pa.string())])
    header_cols = {**header_cols, "retriever": [retriever] * n, "search_space": [space] * n}
    pq.write_table(pa.Table.from_pydict(header_cols, schema=hschema), hp + ".tmp", compression="zstd")
    m = len(ranked_cols["person_id"])
    rschema = pa.schema([
        ("person_id", pa.int64()), ("qa_index", pa.int64()), ("retriever", pa.string()), ("search_space", pa.string()),
        ("original_visit_index", pa.int32()), ("visit_datetime", pa.string()), ("rank", pa.int32()),
        ("score", pa.float64()), ("is_gold", pa.bool_())])
    ranked_cols = {**ranked_cols, "retriever": [retriever] * m, "search_space": [space] * m}
    pq.write_table(pa.Table.from_pydict(ranked_cols, schema=rschema), rp + ".tmp", compression="zstd")
    os.replace(hp + ".tmp", hp)
    os.replace(rp + ".tmp", rp)
    meta = {**meta, "outputs": {"header": {"path": hp[len("/vol/"):], "sha256": _sha(hp), "rows": n},
                                "ranked": {"path": rp[len("/vol/"):], "sha256": _sha(rp), "rows": m}}}
    json.dump(meta, open(mp, "w"), indent=2)
    return meta


def _base_meta(E, retriever, space, code_sha, smoke, extra):
    import datetime
    return {"protocol_version": "1.1", "protocol_sha256": PROTOCOL_SHA, "dataset_sha256": E.DATASET_SHA256,
            "corpus_sha256": E.CORPUS_SHA256, "code_sha256": {"retrieval_v11.py": code_sha, "ehr_data.py": EHR_DATA_SHA},
            "library_versions": _image_versions_record(), "retriever": retriever, "search_space": space,
            "condition_id": f"{retriever}@{space}", "smoke": smoke,
            "created_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"), **extra}


def _load_dense(model):
    import glob
    import numpy as np
    files = sorted(glob.glob(f"{RUN}/emb/{model}/shard_*.npz"))
    assert len(files) == N_SHARDS[model], f"{model}: {len(files)} shard files, expected {N_SHARDS[model]}"
    V, Q, src, stats = {}, {}, [], []
    for f in files:
        z = np.load(f)
        st = json.loads(str(z["stats"]))
        stats.append(st)
        src.append({"path": f[len("/vol/"):], "sha256": _sha(f)})
        vk, ve = z["visit_keys"], z["visit_emb"]
        for pid in np.unique(vk[:, 0]):
            sel = vk[:, 0] == pid
            idx = vk[sel, 1]
            assert pid not in V and (idx == np.arange(len(idx))).all()
            V[int(pid)] = ve[sel]
        for (p, q), e in zip(z["query_keys"], z["query_emb"]):
            assert (int(p), int(q)) not in Q
            Q[(int(p), int(q))] = e
    return V, Q, src, stats


@app.function(image=main_image, cpu=8, memory=65536, timeout=4 * 3600, volumes={"/vol": vol})
def rank_retriever(retriever, code_sha, smoke=False):
    import numpy as np
    from rank_bm25 import BM25Okapi
    vol.reload()
    E = _E()
    C = E.load_corpus()
    P = E.load_protocol("1.1")
    keys = _keys(E)[:200] if smoke else _keys(E)
    t0 = time.time()
    extra = {}
    if retriever == "bm25":
        bp = P["retrievers"]["bm25"]["params"]
        assert (bp["k1"], bp["b"], bp["epsilon"]) == (1.5, 0.75, 0.25)
        doc_tok, full_idx, win_idx = {}, {}, {}
        extra = {"bm25": {"impl": "rank_bm25.BM25Okapi", "params": bp, "tokenizer": P["retrievers"]["bm25"]["tokenizer"],
                          "full_timeline_index": "one per patient over all visits",
                          "sampled_window_index": "one per (person_id, timeline_sampling_strategy) over ONLY window visits (own IDF)"}}
    else:
        V, Q, src, stats = _load_dense(retriever)
        extra = {"embedding_inputs": src, "embedding_stats_summary": {
            "n_visits": sum(s["n_visits"] for s in stats), "n_queries": sum(s["n_queries"] for s in stats),
            "dtype": sorted({s["dtype"] for s in stats}),
            **({"visits_over_4096_tokens": sum(s["visits_over_4096_tokens"] for s in stats)} if retriever == "nvembed_v2" else
               {"visits_truncated_512": sum(s["visits_truncated_512"] for s in stats),
                "queries_truncated_64": sum(s["queries_truncated_64"] for s in stats)})},
            "model": ({"repo": NV_REPO, "sha": NV_SHA} if retriever == "nvembed_v2" else
                      {"query_encoder": list(MEDCPT_Q), "article_encoder": list(MEDCPT_A)}),
            "similarity": "cosine (dot of L2-normalised fp32 vectors, computed in float64)" if retriever == "nvembed_v2"
            else "dot product of CLS vectors (fp32 vectors, computed in float64)",
            "embedding_images": sorted({json.dumps(s["versions"], sort_keys=True) for s in stats})}
    H = {s: {c: [] for c in ("person_id", "qa_index", "timeline_sampling_strategy", "n_visits_patient", "n_candidates",
                             "candidate_visit_ids", "gold_visit_ids", "n_gold_visits", "ranking_depth", "score_native",
                             "inputs_source")} for s in SPACES}
    R = {s: {c: [] for c in ("person_id", "qa_index", "original_visit_index", "visit_datetime", "rank", "score", "is_gold")}
         for s in SPACES}
    for n_done, (pid, qi) in enumerate(keys):
        row = E.get_row(pid, qi)
        strat = str(row["timeline_sampling_strategy"])
        gold = E.gold_visits(row, C)
        win = np.array(E.window_visits(row, C), np.int64)
        Vn = len(C[pid])
        if retriever == "bm25":
            if pid not in doc_tok:
                doc_tok[pid] = [E.bm25_tokenize(v["text"]) for v in C[pid]]
                full_idx[pid] = BM25Okapi(doc_tok[pid], k1=1.5, b=0.75, epsilon=0.25)
            q = E.bm25_tokenize(str(row["question"]))
            s_full = np.asarray(full_idx[pid].get_scores(q), np.float64)
            if (pid, strat) not in win_idx:
                win_idx[(pid, strat)] = BM25Okapi([doc_tok[pid][i] for i in win], k1=1.5, b=0.75, epsilon=0.25)
            s_win = np.asarray(win_idx[(pid, strat)].get_scores(q), np.float64)
        else:
            s_full = V[pid].astype(np.float64) @ Q[(pid, qi)].astype(np.float64)
            s_win = s_full[win]  # exact restriction (dense scores independent of other documents)
        assert len(s_full) == Vn and np.isfinite(s_full).all() and np.isfinite(s_win).all()
        gset = set(gold)
        for space, ids, sc in (("full_timeline", np.arange(Vn), s_full), ("sampled_window", win, s_win)):
            order = np.lexsort((ids, -sc))  # score desc, then visit index asc
            oid = ids[order]
            h, r = H[space], R[space]
            for c, v in (("person_id", pid), ("qa_index", qi), ("timeline_sampling_strategy", strat),
                         ("n_visits_patient", Vn), ("n_candidates", len(ids)), ("candidate_visit_ids", ids.tolist()),
                         ("gold_visit_ids", gold), ("n_gold_visits", len(gold)), ("ranking_depth", len(ids)),
                         ("score_native", True), ("inputs_source", None)):
                h[c].append(v)
            r["person_id"].append(np.full(len(ids), pid, np.int64))
            r["qa_index"].append(np.full(len(ids), qi, np.int64))
            r["original_visit_index"].append(oid.astype(np.int32))
            r["visit_datetime"].extend(C[pid][int(i)]["visit_datetime"] for i in oid)
            r["rank"].append(np.arange(1, len(ids) + 1, dtype=np.int32))
            r["score"].append(sc[order])
            r["is_gold"].append(np.array([int(i) in gset for i in oid], bool))
        if n_done % 1000 == 0:
            print(f"[{retriever}] {n_done}/{len(keys)} {time.time() - t0:.0f}s", flush=True)
    metas = []
    for space in SPACES:
        r = {c: (np.concatenate(v) if c != "visit_datetime" else v) for c, v in R[space].items()}
        meta = _base_meta(E, retriever, space, code_sha, smoke, {**extra, "seconds": time.time() - t0, "n_questions": len(keys)})
        metas.append(_write(retriever, space, smoke, H[space], r, meta))
    vol.commit()
    return metas


@app.function(image=main_image, cpu=8, memory=65536, timeout=4 * 3600, volumes={"/vol": vol})
def build_hybrid(code_sha, smoke=False):
    import pandas as pd
    vol.reload()
    E = _E()
    P = E.load_protocol("1.1")
    hp_ = P["retrievers"]["hybrid_rrf60"]
    assert hp_["k"] == 60 and hp_["inputs"] == ["bm25", "medcpt"]
    metas = []
    for space in SPACES:
        src, frames = {}, {}
        for r in ("bm25", "medcpt"):
            hp, rp, mp = _paths(r, space, smoke)
            m = json.load(open(mp))
            assert m["protocol_version"] == "1.1" and m["outputs"]["ranked"]["sha256"] == _sha(rp)
            assert "archive/pre_fix" not in rp and "retrieval_results_p2_top20" not in rp
            h = pd.read_parquet(hp)
            assert (h["ranking_depth"] == h["n_candidates"]).all(), "input not full depth"
            src[r] = {"path": rp[len("/vol/"):], "sha256": m["outputs"]["ranked"]["sha256"], "rows": m["outputs"]["ranked"]["rows"]}
            frames[r] = pd.read_parquet(rp, columns=["person_id", "qa_index", "original_visit_index", "visit_datetime", "rank", "is_gold"])
            if r == "bm25":
                header = h
        b, m_ = frames["bm25"], frames["medcpt"].drop(columns=["visit_datetime", "is_gold"])
        j = b.merge(m_, on=["person_id", "qa_index", "original_visit_index"], suffixes=("_bm25", "_medcpt"), validate="one_to_one")
        assert len(j) == len(b) == len(m_), "bm25/medcpt candidate sets differ"
        rb, rm = j["rank_bm25"].astype("float64"), j["rank_medcpt"].astype("float64")
        j["score"] = 1.0 / (60.0 + rb) + 1.0 / (60.0 + rm)
        j["min_rank"] = j[["rank_bm25", "rank_medcpt"]].min(axis=1)
        j = j.sort_values(["person_id", "qa_index", "score", "min_rank", "original_visit_index"],
                          ascending=[True, True, False, True, True], kind="mergesort")
        j["rank"] = j.groupby(["person_id", "qa_index"]).cumcount().astype("int32") + 1
        inputs_source = json.dumps({"protocol_version": "1.1", "bm25": src["bm25"], "medcpt": src["medcpt"]}, sort_keys=True)
        hc = {c: header[c].tolist() for c in ("person_id", "qa_index", "timeline_sampling_strategy", "n_visits_patient",
                                              "n_candidates", "gold_visit_ids", "n_gold_visits", "ranking_depth", "score_native")}
        hc["candidate_visit_ids"] = [list(x) for x in header["candidate_visit_ids"]]
        hc["gold_visit_ids"] = [list(x) for x in hc["gold_visit_ids"]]
        hc["inputs_source"] = [inputs_source] * len(header)
        rc = {"person_id": j["person_id"].to_numpy(), "qa_index": j["qa_index"].to_numpy(),
              "original_visit_index": j["original_visit_index"].to_numpy(), "visit_datetime": j["visit_datetime"].tolist(),
              "rank": j["rank"].to_numpy(), "score": j["score"].to_numpy(), "is_gold": j["is_gold"].to_numpy()}
        meta = _base_meta(E, "hybrid_rrf60", space, code_sha, smoke, {
            "inputs_source": json.loads(inputs_source), "k": 60,
            "score": "1/(60+rank_bm25) + 1/(60+rank_medcpt), float64, ranks 1-based within this search_space",
            "tie_break": "min(rank_bm25, rank_medcpt) asc, then original_visit_index asc",
            "forbidden_inputs_checked": ["retrieval_results_p2_top20.json", "archive/pre_fix/*"]})
        metas.append(_write("hybrid_rrf60", space, smoke, hc, rc, meta))
    vol.commit()
    return metas


# ============================================================================ validation (CPU)
@app.function(image=main_image, cpu=8, memory=65536, timeout=4 * 3600, volumes={"/vol": vol})
def validate(retriever, space, smoke=False):
    import numpy as np
    import pandas as pd
    vol.reload()
    E = _E()
    C = E.load_corpus()
    keys = _keys(E)[:200] if smoke else _keys(E)
    hp, rp, mp = _paths(retriever, space, smoke)
    meta = json.load(open(mp))
    h = pd.read_parquet(hp)
    r = pd.read_parquet(rp)
    res, fails = {"retriever": retriever, "space": space, "smoke": smoke, "n_header": len(h), "n_ranked": len(r)}, []

    def chk(name, ok, info=None):
        res[name] = bool(ok) if info is None else {"pass": bool(ok), "info": info}
        if not ok:
            fails.append(name)

    chk("file_sha_matches_meta", _sha(rp) == meta["outputs"]["ranked"]["sha256"] and _sha(hp) == meta["outputs"]["header"]["sha256"])
    chk("meta_required_fields", all(k in meta for k in ("protocol_version", "protocol_sha256", "dataset_sha256", "corpus_sha256",
                                                         "code_sha256", "library_versions", "created_utc"))
        and meta["protocol_version"] == "1.1" and "main_image" in meta["library_versions"] and "nvembed_image" in meta["library_versions"])
    hk = list(zip(h.person_id, h.qa_index))
    chk("I1_header_pk_unique_one_per_dataset_row", len(set(hk)) == len(hk) == len(keys) and set(hk) == set(keys))
    chk("header_enums", (h.retriever == retriever).all() and (h.search_space == space).all()
        and (r.retriever == retriever).all() and (r.search_space == space).all())
    chk("I2_ranked_pk_unique", not r.duplicated(["person_id", "qa_index", "rank"]).any())
    chk("I2_ranked_alt_key_unique", not r.duplicated(["person_id", "qa_index", "original_visit_index"]).any())
    g = r.groupby(["person_id", "qa_index"]).agg(cnt=("rank", "size"), rmin=("rank", "min"), rmax=("rank", "max"))
    hh = h.set_index(["person_id", "qa_index"]).join(g, how="left")
    chk("I3_count_eq_depth_eq_n", ((hh.cnt == hh.ranking_depth) & (hh.ranking_depth == hh.n_candidates)).all())
    chk("I4_ranks_contiguous_1_to_N", ((hh.rmin == 1) & (hh.rmax == hh.n_candidates)).all())
    # candidates
    ok_cand, ok_nv, ok_strat = True, True, True
    dsr = {k: E.get_row(*k) for k in keys}
    for (pid, qi), cand, nvp, strat in zip(hk, h.candidate_visit_ids, h.n_visits_patient, h.timeline_sampling_strategy):
        row = dsr[(pid, qi)]
        exp = list(range(len(C[pid]))) if space == "full_timeline" else E.window_visits(row, C)
        ok_cand &= list(cand) == exp
        ok_nv &= nvp == len(C[pid])
        ok_strat &= strat == row["timeline_sampling_strategy"]
    chk("header_candidate_ids_expected", ok_cand)
    chk("header_n_visits_patient", ok_nv)
    chk("header_strategy_matches_dataset", ok_strat)
    ex = h[["person_id", "qa_index", "candidate_visit_ids"]].explode("candidate_visit_ids").rename(
        columns={"candidate_visit_ids": "original_visit_index"})
    ex["original_visit_index"] = ex["original_visit_index"].astype("int64")
    mm = ex.merge(r[["person_id", "qa_index", "original_visit_index"]].astype("int64"), how="outer", indicator=True)
    chk("I5_ranked_set_eq_candidates", (mm["_merge"] == "both").all())
    nvp = r.person_id.map({int(p): len(C[int(p)]) for p in r.person_id.unique()})
    chk("I6_index_in_range", ((r.original_visit_index >= 0) & (r.original_visit_index < nvp)).all())
    dt_ok = all(C[int(p)][int(i)]["visit_datetime"] == d for p, i, d in
                zip(r.person_id.to_numpy(), r.original_visit_index.to_numpy(), r.visit_datetime.to_numpy()))
    chk("I7_datetime_eq_corpus", dt_ok)
    chk("score_non_null_finite", np.isfinite(r.score.to_numpy()).all())
    rs = r.sort_values(["person_id", "qa_index", "rank"])
    same = (rs.person_id.values[1:] == rs.person_id.values[:-1]) & (rs.qa_index.values[1:] == rs.qa_index.values[:-1])
    ds = rs.score.values[1:] - rs.score.values[:-1]
    chk("I8_score_non_increasing", (ds[same] <= 0).all())
    tie = same & (ds == 0)
    n_ties = int(tie.sum())
    if retriever != "hybrid_rrf60":
        dv = rs.original_visit_index.values[1:] - rs.original_visit_index.values[:-1]
        chk("I8_tie_break_visit_index_asc", (dv[tie] > 0).all(), {"n_tied_adjacent_pairs": n_ties})
    # gold
    gold_ok = all(list(gv) == E.gold_visits(dsr[(p, q)], C) and ng == len(gv) >= 1 and set(gv) <= set(cv)
                  for p, q, gv, ng, cv in zip(h.person_id, h.qa_index, h.gold_visit_ids, h.n_gold_visits, h.candidate_visit_ids))
    chk("I10_gold_eq_ehr_data_and_subset_of_candidates", gold_ok)
    gl = {(p, q): set(gv) for p, q, gv in zip(h.person_id, h.qa_index, h.gold_visit_ids)}
    isg = np.array([i in gl[(p, q)] for p, q, i in zip(r.person_id.to_numpy(), r.qa_index.to_numpy(), r.original_visit_index.to_numpy())])
    chk("is_gold_consistent", (isg == r.is_gold.to_numpy()).all())
    chk("score_native_true", h.score_native.all())
    # I9: sampled_window relative order vs full_timeline of the same retriever
    if space == "sampled_window" and retriever in ("bm25", "medcpt", "nvembed_v2"):
        f = pd.read_parquet(_paths(retriever, "full_timeline", smoke)[1], columns=["person_id", "qa_index", "original_visit_index", "rank"])
        mj = r[["person_id", "qa_index", "original_visit_index", "rank"]].merge(
            f, on=["person_id", "qa_index", "original_visit_index"], suffixes=("_w", "_f"))
        mj = mj.sort_values(["person_id", "qa_index", "rank_w"])
        s2 = (mj.person_id.values[1:] == mj.person_id.values[:-1]) & (mj.qa_index.values[1:] == mj.qa_index.values[:-1])
        bad = s2 & (mj.rank_f.values[1:] < mj.rank_f.values[:-1])
        kb = mj.iloc[1:][bad][["person_id", "qa_index"]].drop_duplicates()
        info = {"questions_with_order_different_from_full": int(len(kb)), "of": len(keys)}
        if retriever == "bm25":
            res["I9_bm25_window_order_vs_full_INFO_ONLY"] = {**info, "note": "expected to differ: v1.1 change 2 gives the window its own BM25 index/IDF; the verbatim draft invariant predates change 2"}
        else:
            chk("I9_dense_window_order_eq_full", len(kb) == 0, info)
    # I11: hybrid inputs + exact recomputation
    if retriever == "hybrid_rrf60":
        src = json.loads(h.inputs_source.iloc[0])
        chk("I11_inputs_source_same_all_rows", (h.inputs_source == h.inputs_source.iloc[0]).all())
        chk("I11_inputs_are_v11_files_same_space", all(
            src[x]["path"] == _paths(x, space, smoke)[1][len("/vol/"):] and src[x]["sha256"] == _sha("/vol/" + src[x]["path"])
            and json.load(open(_paths(x, space, smoke)[2]))["protocol_version"] == "1.1" for x in ("bm25", "medcpt")))
        chk("I11_no_forbidden_inputs", not any("archive/pre_fix" in src[x]["path"] or "retrieval_results_p2_top20" in src[x]["path"]
                                               for x in ("bm25", "medcpt")))
        b = pd.read_parquet("/vol/" + src["bm25"]["path"], columns=["person_id", "qa_index", "original_visit_index", "rank"])
        m = pd.read_parquet("/vol/" + src["medcpt"]["path"], columns=["person_id", "qa_index", "original_visit_index", "rank"])
        j = b.merge(m, on=["person_id", "qa_index", "original_visit_index"], suffixes=("_b", "_m"))
        # independent recomputation: python-level per question
        exp_rank = {}
        for (p, q), grp in j.groupby(["person_id", "qa_index"], sort=False):
            items = sorted(((-(1.0 / (60 + rb) + 1.0 / (60 + rmm)), min(rb, rmm), int(i))
                            for i, rb, rmm in zip(grp.original_visit_index, grp.rank_b, grp.rank_m)))
            for k_, it in enumerate(items):
                exp_rank[(int(p), int(q), it[2])] = (k_ + 1, -it[0])
        got = all(exp_rank[(int(p), int(q), int(i))] == (int(k_), float(s_)) for p, q, i, k_, s_ in
                  zip(r.person_id, r.qa_index, r.original_visit_index, r["rank"], r.score))
        chk("I11_exact_recompute_rank_and_score", got and len(exp_rank) == len(r), {"n_tied_adjacent_pairs": n_ties})
    res["gold_rank_summary"] = {"hit_at_5_any_gold_pct": float(100 * r[r["rank"] <= 5].groupby(["person_id", "qa_index"]).is_gold.any().reindex(
        pd.MultiIndex.from_tuples(keys), fill_value=False).mean())}
    res["FAILED"] = fails
    res["PASS"] = not fails
    out = f"{_prefix(smoke)}/validation/{retriever}@{space}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=2, default=str)
    vol.commit()
    return res


def _code_sha():
    return hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()


def _copy_outputs(metas):
    for m in metas:
        for o in m["outputs"].values():
            _copy_back(o["path"])
        mp = o["path"].replace(".ranked.parquet", ".meta.json")
        _copy_back(mp)


@app.local_entrypoint()
def rank(retriever: str = "bm25", smoke: bool = False):
    metas = rank_retriever.remote(retriever, _code_sha(), smoke)
    _copy_outputs(metas)
    for m in metas:
        print("RANKED", m["condition_id"], json.dumps(m["outputs"]))
    for space in SPACES:
        v = validate.remote(retriever, space, smoke)
        _copy_back(f"{RUN_REL}/{'smoke/' if smoke else ''}validation/{retriever}@{space}.json")
        print("VALID", retriever, space, "PASS" if v["PASS"] else f"FAIL {v['FAILED']}", json.dumps(v.get("gold_rank_summary")),
              json.dumps(v.get("I9_bm25_window_order_vs_full_INFO_ONLY") or v.get("I9_dense_window_order_eq_full") or ""))


@app.local_entrypoint()
def hybrid(smoke: bool = False):
    metas = build_hybrid.remote(_code_sha(), smoke)
    _copy_outputs(metas)
    for space in SPACES:
        v = validate.remote("hybrid_rrf60", space, smoke)
        _copy_back(f"{RUN_REL}/{'smoke/' if smoke else ''}validation/hybrid_rrf60@{space}.json")
        print("VALID hybrid_rrf60", space, "PASS" if v["PASS"] else f"FAIL {v['FAILED']}", json.dumps(v.get("gold_rank_summary")))
