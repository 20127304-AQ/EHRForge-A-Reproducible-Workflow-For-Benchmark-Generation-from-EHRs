"""Step 3: build the reader contexts ONCE per condition (protocol v1.3 / context unchanged from v1.0).

For each condition in {bm25,medcpt,nvembed_v2,hybrid_rrf60}@full_timeline and oracle:
  * retrieval: ranked parquet sha256 verified against shared/retrieval_v1.1/MANIFEST_rankings.csv; selection =
    sorted(top-min(5,N)) by rank; header gold_visit_ids must equal ehr_data.gold_visits.
  * oracle: ehr_data.gold_visits (original_visit_index, dedup, sorted).
  * ehr_data.build_context(..., budget=3500) -> (context, base metadata); invariants of protocol context_metadata.
  * per-visit text char spans inside the context (used later for n_visits_reaching_model per reader).
  * leakage report: normalized gold answer verbatim inside normalized context / gold-visit note text / question.
Output (volume): /vol/step3/contexts/{condition}.parquet + {condition}.meta.json (file sha256).
"""
import json
import os
import sys
import time

import modal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/step3code")
from images import extractive_image, WORKDIR  # noqa: E402
from step3_common import VOLUME_NAME, HF_CACHE_VOLUME, CONDITIONS  # noqa: E402

app = modal.App("ehrforge-step3-contexts")
vol = modal.Volume.from_name(VOLUME_NAME)
hf = modal.Volume.from_name(HF_CACHE_VOLUME, create_if_missing=True)


def visit_text_spans(context, headers):
    """char spans [start,end) of each visit's (possibly trimmed) text inside the assembled context."""
    spans, pos = [], 0
    for i, h in enumerate(headers):
        assert context.startswith(h + "\n", pos), (i, pos)
        s = pos + len(h) + 1
        if i + 1 < len(headers):
            nxt = context.find("\n\n" + headers[i + 1] + "\n", s)
            assert nxt >= s
            e = nxt
            pos = nxt + 2
        else:
            e = len(context)
        spans.append([s, e])
    # exact reconstruction check
    rebuilt = "\n\n".join(h + "\n" + context[s:e] for h, (s, e) in zip(headers, spans))
    assert rebuilt == context
    return spans


@app.function(image=extractive_image, cpu=4, memory=16384, timeout=7200,
              volumes={"/vol": vol, "/root/.cache/huggingface": hf})
def build(condition: str, limit: int = 0):
    sys.path.insert(0, "/step3code")
    sys.path.insert(0, "/ehr/shared/canonical")
    import csv
    import hashlib
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import ehr_data as E
    from step3_common import (READER_TOP_K, BUDGET, MANIFEST_SHA, normalize_answer, sha256_text, OUT)
    from images import runtime_versions
    t0 = time.time()
    df = E.load_dataset()
    corpus = E.load_corpus()
    tok = E.budget_tokenizer()
    from huggingface_hub import hf_hub_download
    tok_path = hf_hub_download(E.BUDGET_TOKENIZER_REPO, "tokenizer.json", revision=E.BUDGET_TOKENIZER_SHA)
    assert f"/snapshots/{E.BUDGET_TOKENIZER_SHA}/" in os.path.realpath(tok_path) or \
        f"/snapshots/{E.BUDGET_TOKENIZER_SHA}/" in tok_path, tok_path
    tok_file_sha = E.sha256_file(tok_path)
    ehr_data_sha = E.sha256_file("/ehr/shared/canonical/ehr_data.py")

    keys = sorted(map(tuple, df[["person_id", "qa_index"]].astype("int64").itertuples(index=False, name=None)))
    if limit:
        keys = keys[:limit]
    rows = E._dataset_by_key()

    ranking_info = None
    sel_map, rank_map = {}, {}
    if condition != "oracle":
        path = f"/vol/shared/retrieval_v1.1/rankings/{condition}.ranked.parquet"
        man = {r["condition"] + "|" + r["kind"]: r for r in csv.DictReader(open("/vol/shared/retrieval_v1.1/MANIFEST_rankings.csv"))}
        got = E.sha256_file(path)
        exp = man[condition + "|ranked"]["sha256"]
        assert got == exp == MANIFEST_SHA[condition], (got, exp)
        hpath = f"/vol/shared/retrieval_v1.1/rankings/{condition}.header.parquet"
        hgot = E.sha256_file(hpath)
        assert hgot == man[condition + "|header"]["sha256"], hgot
        ranking_info = {"ranked_path": f"shared/retrieval_v1.1/rankings/{condition}.ranked.parquet", "ranked_sha256": got,
                        "header_path": f"shared/retrieval_v1.1/rankings/{condition}.header.parquet", "header_sha256": hgot}
        t = pq.read_table(path, columns=["person_id", "qa_index", "retriever", "search_space", "original_visit_index", "rank"],
                          filters=[("rank", "<=", READER_TOP_K)]).to_pandas()
        r, ss = condition.split("@")
        assert set(t["retriever"]) == {r} and set(t["search_space"]) == {ss}
        h = pq.read_table(hpath, columns=["person_id", "qa_index", "n_candidates", "gold_visit_ids"]).to_pandas()
        hN = {(int(a), int(b)): int(n) for a, b, n in zip(h.person_id, h.qa_index, h.n_candidates)}
        hG = {(int(a), int(b)): [int(x) for x in g] for a, b, g in zip(h.person_id, h.qa_index, h.gold_visit_ids)}
        assert len(hN) == 10742
        t = t.sort_values(["person_id", "qa_index", "rank"])
        for (pid, qi), g in t.groupby(["person_id", "qa_index"], sort=False):
            k = (int(pid), int(qi))
            ranks = g["rank"].tolist()
            assert ranks == list(range(1, len(ranks) + 1)), (k, ranks)
            assert len(ranks) == min(READER_TOP_K, hN[k]), (k, len(ranks), hN[k])
            rank_map[k] = [int(x) for x in g["original_visit_index"]]
            sel_map[k] = sorted(rank_map[k])

    out_rows = []
    leak = {"answer_in_context": 0, "answer_in_gold_notes": 0, "answer_in_question": 0, "n": 0}
    for i, k in enumerate(keys):
        row = rows.loc[k]
        gold = E.gold_visits(row, corpus)
        if condition == "oracle":
            sel, top_rank_order = list(gold), None
        else:
            assert hG[k] == gold, (k, hG[k], gold)
            sel, top_rank_order = sel_map[k], rank_map[k]
        ctx, meta = E.build_context(k[0], sel, BUDGET, qa_index=k[1], condition=condition, reader=None,
                                    gold_visit_ids=gold, corpus=corpus, tokenizer=tok)
        visits = corpus[k[0]]
        headers = [E.HEADER_FMT.format(idx=v, datetime=visits[v]["visit_datetime"]) for v in meta["selected_visit_ids"]]
        spans = visit_text_spans(ctx, headers)
        # invariants (protocol context_metadata)
        assert meta["selected_visit_ids"] == sorted(sel)
        if condition == "oracle":
            assert meta["selected_visit_ids"] == gold and meta["n_selected_visits"] == len(gold)
        assert meta["total_tokens_after"] <= BUDGET, (k, meta["total_tokens_after"])
        assert meta["truncated"] == (meta["total_tokens_before"] > BUDGET)
        if not meta["truncated"]:
            assert meta["per_visit_tokens_after"] == meta["per_visit_tokens_before"]
        assert all(a <= b for a, b in zip(meta["per_visit_tokens_after"], meta["per_visit_tokens_before"]))
        assert meta["context_sha256"] == sha256_text(ctx)
        if meta["gold_fully_survived"]:
            assert meta["n_gold_visits_surviving"] == meta["n_gold_visits"]
        # leakage (report only)
        na = normalize_answer(str(row["answer"]))
        leak["n"] += 1
        if na and na in normalize_answer(ctx):
            leak["answer_in_context"] += 1
        if na and na in normalize_answer(" ".join(visits[g]["text"] for g in gold)):
            leak["answer_in_gold_notes"] += 1
        if na and na in normalize_answer(str(row["question"])):
            leak["answer_in_question"] += 1
        rec = dict(meta)
        rec.update({"context": ctx, "visit_text_spans": json.dumps(spans), "top_k_rank_order": json.dumps(top_rank_order),
                    "question": str(row["question"]), "budget_tokenizer_file_sha256": tok_file_sha})
        for f in ["gold_visit_ids", "selected_visit_ids", "visit_ids_reaching_model", "per_visit_tokens_before",
                  "per_visit_tokens_after"]:
            rec[f] = json.dumps(rec[f])
        out_rows.append(rec)
        if i % 2000 == 0:
            print(f"[{condition}] {i}/{len(keys)} {time.time()-t0:.0f}s", flush=True)

    o = pd.DataFrame(out_rows)
    assert not o.duplicated(["person_id", "qa_index"]).any()
    os.makedirs(f"{OUT}/contexts", exist_ok=True)
    suffix = "" if not limit else f".limit{limit}"
    p = f"{OUT}/contexts/{condition}{suffix}.parquet"
    pq.write_table(pa.Table.from_pandas(o, preserve_index=False), p, compression="zstd")
    fsha = E.sha256_file(p)
    summ = {
        "condition": condition, "n": len(o), "file": p, "file_sha256": fsha,
        "contexts_concat_sha256": hashlib.sha256("".join(o.context_sha256).encode()).hexdigest(),
        "truncated_rate": float(o.truncated.mean()),
        "gold_fully_survived_rate": float(o.gold_fully_survived.mean()),
        "n_gold_visits_surviving_eq_n_gold_rate": float((o.n_gold_visits_surviving == o.n_gold_visits).mean()),
        "any_gold_surviving_rate": float((o.n_gold_visits_surviving > 0).mean()),
        "mean_total_tokens_after": float(o.total_tokens_after.mean()),
        "max_total_tokens_after": int(o.total_tokens_after.max()),
        "leakage": {k2: (v / leak["n"] if k2 != "n" else v) for k2, v in leak.items()},
        "leakage_counts": leak,
        "ranking_input": ranking_info,
        "budget_tokenizer": {"repo": E.BUDGET_TOKENIZER_REPO, "revision": E.BUDGET_TOKENIZER_SHA,
                             "tokenizer_json_sha256": tok_file_sha, "library": "tokenizers"},
        "ehr_data_py_sha256": ehr_data_sha,
        "dataset_sha256": E.DATASET_SHA256, "corpus_sha256": E.CORPUS_SHA256,
        "versions": runtime_versions(), "seconds": time.time() - t0,
    }
    json.dump(summ, open(f"{OUT}/contexts/{condition}{suffix}.meta.json", "w"), indent=1)
    vol.commit()
    hf.commit()
    print(json.dumps({k2: v for k2, v in summ.items() if k2 != "versions"}, indent=1), flush=True)
    return summ


@app.local_entrypoint()
def main(limit: int = 0, conditions: str = ",".join(CONDITIONS)):
    conds = conditions.split(",")
    res = list(build.starmap([(c, limit) for c in conds]))
    os.makedirs(os.path.join(WORKDIR, "results", "contexts"), exist_ok=True)
    for r in res:
        suffix = "" if not limit else f".limit{limit}"
        json.dump(r, open(os.path.join(WORKDIR, "results", "contexts", f"{r['condition']}{suffix}.meta.json"), "w"), indent=1)
        print(r["condition"], r["n"], r["file_sha256"][:16], "trunc", round(r["truncated_rate"], 4),
              "gold_full", round(r["gold_fully_survived_rate"], 4), "leak", r["leakage"])
