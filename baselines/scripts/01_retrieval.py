"""Run BM25, MedCPT, NV-Embed-v2, and Hybrid RRF retrieval experiments."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import modal

from ehrforge_repro.config import (
    APP_PREFIX,
    CORPUS_PATH,
    DATA_VOLUME_NAME,
    DATASET_PATH,
    DENSE_MAX_LENGTH,
    HF_SECRET_NAME,
    MANIFEST_DIR,
    MODEL_IDS,
    NVEMBED_QUERY_INSTRUCTION,
    RANDOM_SEED,
    RESULTS_VOLUME_NAME,
    RETRIEVAL_DIR,
    RETRIEVAL_RESULTS_PATH,
    RETRIEVAL_TOP_K,
    RRF_K,
)
from ehrforge_repro.data import (
    group_dataset_by_patient,
    load_corpus,
    load_dataset,
    parse_evidence_indices,
)
from ehrforge_repro.io_utils import (
    file_sha256,
    iter_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)


app = modal.App(f"{APP_PREFIX}-retrieval")
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name("huggingface-cache-qa", create_if_missing=True)

common_packages = [
    "numpy<2",
    "pandas>=2.1,<3",
    "tqdm>=4.66,<5",
    "rank-bm25>=0.2.2,<0.3",
]

cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*common_packages)
    .add_local_python_source("ehrforge_repro")
)

small_dense_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11"
    )
    .entrypoint([])
    .pip_install(
        *common_packages,
        "torch==2.4.0",
        "transformers==4.44.2",
        "accelerate>=0.33,<1",
        "sentencepiece>=0.2,<1",
    )
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("ehrforge_repro")
)

nvembed_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11"
    )
    .entrypoint([])
    .apt_install("build-essential", "git")
    .pip_install(
        *common_packages,
        "torch==2.2.0",
        "transformers==4.42.4",
        "sentence-transformers==2.7.0",
        "accelerate>=0.29,<1",
        "einops>=0.8,<1",
        "sentencepiece>=0.2,<1",
        "packaging",
        "ninja",
    )
    .run_commands("pip install flash-attn==2.2.0 --no-build-isolation")
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_python_source("ehrforge_repro")
)

volume_mounts = {
    "/data": data_volume,
    "/results": results_volume,
    "/root/.cache/huggingface": hf_cache,
}


def _method_path(method: str) -> Path:
    return RETRIEVAL_DIR / f"{method}.jsonl"


def _rank_top_k(scores: Any, top_k: int) -> list[int]:
    import numpy as np

    scores_array = np.asarray(scores)
    if scores_array.size == 0:
        return []
    count = min(top_k, scores_array.shape[-1])
    return np.argsort(-scores_array, kind="stable")[:count].astype(int).tolist()


def _load_method_records(method: str) -> dict[tuple[int, int], dict[str, Any]]:
    path = _method_path(method)
    if not path.exists():
        raise FileNotFoundError(f"Missing retrieval stage output: {path}")
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for record in iter_jsonl(path):
        key = (int(record["person_id"]), int(record["qa_index"]))
        if key in records:
            raise ValueError(f"Duplicate {method} record for sample {key}")
        records[key] = record
    return records


def _rrf(rankings: list[list[int]], top_k: int, rrf_k: int) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, document_id in enumerate(ranking, start=1):
            scores[int(document_id)] = scores.get(int(document_id), 0.0) + 1.0 / (
                rrf_k + rank
            )
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [document_id for document_id, _ in ordered[:top_k]]


@app.function(
    image=cpu_image,
    volumes=volume_mounts,
    cpu=8,
    memory=32768,
    timeout=24 * 60 * 60,
)
def run_bm25(overwrite: bool = False) -> dict[str, Any]:
    """Compute BM25 rankings over each patient's visit timeline."""
    import numpy as np
    from rank_bm25 import BM25Okapi
    from tqdm import tqdm

    results_volume.reload()
    output_path = _method_path("bm25")
    if output_path.exists() and not overwrite:
        return {"status": "skipped", "path": str(output_path)}

    frame = load_dataset(DATASET_PATH)
    corpus = load_corpus(CORPUS_PATH)
    grouped = group_dataset_by_patient(frame)

    def tokenize(text: str) -> list[str]:
        import re

        return re.sub(r"[^\w\s]", " ", text.lower()).split()

    output: list[dict[str, Any]] = []
    started = time.time()
    for person_id, rows in tqdm(grouped.items(), desc="BM25 patients"):
        visits = corpus.get(person_id, [])
        if visits:
            index = BM25Okapi([tokenize(str(visit.get("text", ""))) for visit in visits])
        else:
            index = None

        for row in rows:
            if index is None:
                ranking: list[int] = []
            else:
                scores = index.get_scores(tokenize(str(row["question"])))
                ranking = _rank_top_k(np.asarray(scores), RETRIEVAL_TOP_K)
            output.append(
                {
                    "person_id": person_id,
                    "qa_index": int(row["qa_index"]),
                    "row_index": int(row["_row_index"]),
                    "ranking": ranking,
                }
            )

    output.sort(key=lambda item: item["row_index"])
    write_jsonl_atomic(output_path, output)
    write_json_atomic(
        MANIFEST_DIR / "bm25.json",
        {
            "method": "bm25",
            "top_k": RETRIEVAL_TOP_K,
            "tokenizer": "lowercase_alphanumeric_whitespace",
            "dataset_sha256": file_sha256(DATASET_PATH),
            "n_samples": len(output),
            "elapsed_seconds": time.time() - started,
        },
    )
    results_volume.commit()
    return {"status": "completed", "n_samples": len(output), "path": str(output_path)}


@app.function(
    image=small_dense_image,
    volumes=volume_mounts,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    gpu="A100-80GB",
    memory=131072,
    timeout=24 * 60 * 60,
)
def run_medcpt(overwrite: bool = False, batch_size: int = 64) -> dict[str, Any]:
    """Compute MedCPT rankings with the released query and article encoders."""
    import gc

    import numpy as np
    import torch
    from tqdm import tqdm
    from transformers import AutoModel, AutoTokenizer

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    results_volume.reload()
    output_path = _method_path("medcpt")
    if output_path.exists() and not overwrite:
        return {"status": "skipped", "path": str(output_path)}

    frame = load_dataset(DATASET_PATH)
    corpus = load_corpus(CORPUS_PATH)
    grouped = group_dataset_by_patient(frame)

    query_model_id = MODEL_IDS["medcpt_query"]
    article_model_id = MODEL_IDS["medcpt_article"]
    query_tokenizer = AutoTokenizer.from_pretrained(query_model_id)
    article_tokenizer = AutoTokenizer.from_pretrained(article_model_id)
    query_model = AutoModel.from_pretrained(query_model_id).cuda().eval()
    article_model = AutoModel.from_pretrained(article_model_id).cuda().eval()

    def encode(
        texts: list[str], tokenizer: Any, model: Any, local_batch_size: int
    ) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        for start in range(0, len(texts), local_batch_size):
            encoded = tokenizer(
                texts[start : start + local_batch_size],
                padding=True,
                truncation=True,
                max_length=DENSE_MAX_LENGTH,
                return_tensors="pt",
            ).to("cuda")
            with torch.inference_mode():
                embeddings = model(**encoded).last_hidden_state[:, 0, :]
            chunks.append(embeddings.float().cpu())
        return torch.cat(chunks, dim=0) if chunks else torch.empty((0, 768))

    output: list[dict[str, Any]] = []
    started = time.time()
    for person_id, rows in tqdm(grouped.items(), desc="MedCPT patients"):
        visits = corpus.get(person_id, [])
        if not visits:
            for row in rows:
                output.append(
                    {
                        "person_id": person_id,
                        "qa_index": int(row["qa_index"]),
                        "row_index": int(row["_row_index"]),
                        "ranking": [],
                    }
                )
            continue

        article_embeddings = encode(
            [str(visit.get("text", "")) for visit in visits],
            article_tokenizer,
            article_model,
            batch_size,
        )
        query_embeddings = encode(
            [str(row["question"]) for row in rows],
            query_tokenizer,
            query_model,
            batch_size,
        )
        scores = query_embeddings @ article_embeddings.T
        for row, row_scores in zip(rows, scores.numpy()):
            output.append(
                {
                    "person_id": person_id,
                    "qa_index": int(row["qa_index"]),
                    "row_index": int(row["_row_index"]),
                    "ranking": _rank_top_k(row_scores, RETRIEVAL_TOP_K),
                }
            )

    output.sort(key=lambda item: item["row_index"])
    write_jsonl_atomic(output_path, output)
    write_json_atomic(
        MANIFEST_DIR / "medcpt.json",
        {
            "method": "medcpt",
            "query_model": query_model_id,
            "article_model": article_model_id,
            "query_model_revision": getattr(query_model.config, "_commit_hash", None),
            "article_model_revision": getattr(article_model.config, "_commit_hash", None),
            "pooling": "cls_last_hidden_state",
            "max_length": DENSE_MAX_LENGTH,
            "top_k": RETRIEVAL_TOP_K,
            "dataset_sha256": file_sha256(DATASET_PATH),
            "n_samples": len(output),
            "elapsed_seconds": time.time() - started,
        },
    )
    results_volume.commit()

    del query_model, article_model
    torch.cuda.empty_cache()
    gc.collect()
    return {"status": "completed", "n_samples": len(output), "path": str(output_path)}


@app.function(
    image=nvembed_image,
    volumes=volume_mounts,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    gpu="A100-80GB",
    memory=131072,
    timeout=24 * 60 * 60,
)
def run_nvembed(overwrite: bool = False, batch_size: int = 4) -> dict[str, Any]:
    """Compute NV-Embed-v2 rankings without substituting another model."""
    import gc

    import numpy as np
    import torch
    from tqdm import tqdm
    from transformers import AutoModel

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    results_volume.reload()
    output_path = _method_path("nvembed")
    if output_path.exists() and not overwrite:
        return {"status": "skipped", "path": str(output_path)}

    frame = load_dataset(DATASET_PATH)
    corpus = load_corpus(CORPUS_PATH)
    grouped = group_dataset_by_patient(frame)
    model_id = MODEL_IDS["nvembed"]

    try:
        model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
        ).eval()
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load the required NV-Embed-v2 checkpoint {model_id}. "
            "No fallback model is permitted for this experiment."
        ) from exc

    def encode(texts: list[str], instruction: str) -> np.ndarray:
        if not texts:
            return np.empty((0, 4096), dtype=np.float32)
        embeddings = model._do_encode(
            texts,
            batch_size=batch_size,
            instruction=instruction,
            max_length=DENSE_MAX_LENGTH,
            num_workers=0,
            return_numpy=True,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.clip(norms, 1e-12, None)

    output: list[dict[str, Any]] = []
    started = time.time()
    for person_id, rows in tqdm(grouped.items(), desc="NV-Embed-v2 patients"):
        visits = corpus.get(person_id, [])
        if not visits:
            for row in rows:
                output.append(
                    {
                        "person_id": person_id,
                        "qa_index": int(row["qa_index"]),
                        "row_index": int(row["_row_index"]),
                        "ranking": [],
                    }
                )
            continue

        passage_embeddings = encode(
            [str(visit.get("text", "")) for visit in visits], ""
        )
        query_embeddings = encode(
            [str(row["question"]) for row in rows], NVEMBED_QUERY_INSTRUCTION
        )
        scores = query_embeddings @ passage_embeddings.T
        for row, row_scores in zip(rows, scores):
            output.append(
                {
                    "person_id": person_id,
                    "qa_index": int(row["qa_index"]),
                    "row_index": int(row["_row_index"]),
                    "ranking": _rank_top_k(row_scores, RETRIEVAL_TOP_K),
                }
            )

    output.sort(key=lambda item: item["row_index"])
    write_jsonl_atomic(output_path, output)
    write_json_atomic(
        MANIFEST_DIR / "nvembed.json",
        {
            "method": "nvembed",
            "model": model_id,
            "model_revision": getattr(model.config, "_commit_hash", None),
            "query_instruction": NVEMBED_QUERY_INSTRUCTION,
            "passage_instruction": "",
            "max_length": DENSE_MAX_LENGTH,
            "top_k": RETRIEVAL_TOP_K,
            "fallback_used": False,
            "dataset_sha256": file_sha256(DATASET_PATH),
            "n_samples": len(output),
            "elapsed_seconds": time.time() - started,
        },
    )
    results_volume.commit()

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return {"status": "completed", "n_samples": len(output), "path": str(output_path)}


@app.function(
    image=cpu_image,
    volumes=volume_mounts,
    cpu=4,
    memory=32768,
    timeout=4 * 60 * 60,
)
def combine_retrieval(overwrite: bool = False) -> dict[str, Any]:
    """Combine method rankings, compute Hybrid RRF, and attach oracle evidence."""
    results_volume.reload()
    if RETRIEVAL_RESULTS_PATH.exists() and not overwrite:
        return {"status": "skipped", "path": str(RETRIEVAL_RESULTS_PATH)}

    frame = load_dataset(DATASET_PATH)
    bm25 = _load_method_records("bm25")
    medcpt = _load_method_records("medcpt")
    nvembed = _load_method_records("nvembed")

    expected_keys = {
        (int(row.person_id), int(row.qa_index)) for row in frame.itertuples(index=False)
    }
    for method_name, records in (
        ("bm25", bm25),
        ("medcpt", medcpt),
        ("nvembed", nvembed),
    ):
        if set(records) != expected_keys:
            missing = sorted(expected_keys - set(records))[:10]
            extra = sorted(set(records) - expected_keys)[:10]
            raise ValueError(
                f"{method_name} key coverage mismatch; missing={missing}, extra={extra}"
            )

    combined: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        key = (int(row["person_id"]), int(row["qa_index"]))
        bm25_ranking = [int(value) for value in bm25[key]["ranking"]]
        medcpt_ranking = [int(value) for value in medcpt[key]["ranking"]]
        nvembed_ranking = [int(value) for value in nvembed[key]["ranking"]]
        combined.append(
            {
                "person_id": key[0],
                "qa_index": key[1],
                "row_index": int(row["_row_index"]),
                "bm25_top20": bm25_ranking,
                "medcpt_top20": medcpt_ranking,
                "nvembed_top20": nvembed_ranking,
                "hybrid_top20": _rrf(
                    [bm25_ranking, medcpt_ranking], RETRIEVAL_TOP_K, RRF_K
                ),
                "oracle_indices": parse_evidence_indices(row.get("evidence")),
            }
        )

    combined.sort(key=lambda item: item["row_index"])
    write_jsonl_atomic(RETRIEVAL_RESULTS_PATH, combined)
    write_json_atomic(
        MANIFEST_DIR / "retrieval_combined.json",
        {
            "methods": ["bm25", "medcpt", "nvembed", "hybrid", "oracle"],
            "hybrid_method": "reciprocal_rank_fusion",
            "rrf_k": RRF_K,
            "top_k": RETRIEVAL_TOP_K,
            "dataset_sha256": file_sha256(DATASET_PATH),
            "n_samples": len(combined),
            "source_files": {
                method: str(_method_path(method))
                for method in ("bm25", "medcpt", "nvembed")
            },
        },
    )
    results_volume.commit()
    return {
        "status": "completed",
        "n_samples": len(combined),
        "path": str(RETRIEVAL_RESULTS_PATH),
    }


@app.local_entrypoint()
def main(
    phase: str = "all",
    overwrite: bool = False,
    medcpt_batch_size: int = 64,
    nvembed_batch_size: int = 4,
) -> None:
    """Run one retrieval phase or the complete retrieval pipeline."""
    phase = phase.strip().lower()
    valid = {"all", "bm25", "medcpt", "nvembed", "combine"}
    if phase not in valid:
        raise ValueError(f"phase must be one of {sorted(valid)}")

    if phase in {"all", "bm25"}:
        print(run_bm25.remote(overwrite=overwrite))
    if phase in {"all", "medcpt"}:
        print(
            run_medcpt.remote(
                overwrite=overwrite,
                batch_size=medcpt_batch_size,
            )
        )
    if phase in {"all", "nvembed"}:
        print(
            run_nvembed.remote(
                overwrite=overwrite,
                batch_size=nvembed_batch_size,
            )
        )
    if phase in {"all", "combine"}:
        print(combine_retrieval.remote(overwrite=overwrite))

    print(
        "Results are stored in Modal volume "
        f"{RESULTS_VOLUME_NAME} under {RETRIEVAL_DIR}."
    )
