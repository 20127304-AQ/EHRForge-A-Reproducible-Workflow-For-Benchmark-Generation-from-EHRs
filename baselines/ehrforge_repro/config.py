"""Shared configuration for the EHRForge downstream experiments."""

from __future__ import annotations

import os
from pathlib import Path


APP_PREFIX = os.getenv("EHRFORGE_APP_PREFIX", "ehrforge-experiments")
DATA_VOLUME_NAME = os.getenv("EHRFORGE_DATA_VOLUME", "clinical-qa-data")
RESULTS_VOLUME_NAME = os.getenv("EHRFORGE_RESULTS_VOLUME", "ehrforge-experiment-results")
HF_SECRET_NAME = os.getenv("EHRFORGE_HF_SECRET", "huggingface-secret")

DATA_ROOT = Path("/data")
RESULTS_ROOT = Path("/results")
DATASET_PATH = DATA_ROOT / "dataset.csv"
CORPUS_PATH = DATA_ROOT / "corpus.json"

RETRIEVAL_DIR = RESULTS_ROOT / "retrieval"
PREDICTION_DIR = RESULTS_ROOT / "predictions"
PREDICTION_METRICS_DIR = RESULTS_ROOT / "predictions_with_metrics"
METRICS_DIR = RESULTS_ROOT / "metrics"
MANIFEST_DIR = RESULTS_ROOT / "manifests"
VALIDATION_DIR = RESULTS_ROOT / "validation"

RETRIEVAL_RESULTS_PATH = RETRIEVAL_DIR / "retrieval_results.jsonl"
RETRIEVAL_METRICS_PATH = METRICS_DIR / "retrieval_metrics.json"
QA_METRICS_PATH = METRICS_DIR / "qa_metrics.json"

RANDOM_SEED = int(os.getenv("EHRFORGE_RANDOM_SEED", "42"))
RETRIEVAL_TOP_K = int(os.getenv("EHRFORGE_RETRIEVAL_TOP_K", "20"))
READER_TOP_K = int(os.getenv("EHRFORGE_READER_TOP_K", "5"))
RRF_K = int(os.getenv("EHRFORGE_RRF_K", "60"))
DENSE_MAX_LENGTH = int(os.getenv("EHRFORGE_DENSE_MAX_LENGTH", "512"))

CONTEXT_TYPES = ("bm25", "medcpt", "nvembed", "hybrid", "oracle")
RETRIEVER_TYPES = ("bm25", "medcpt", "nvembed", "hybrid")
READER_TYPES = ("clinicalbert", "longformer", "qwen2.5-7b", "qwen2.5-32b")
STRATIFICATION_AXES = ("difficulty", "visit_group", "reasoning_type")
RETRIEVAL_K_VALUES = (5, 10, 20)

MODEL_IDS = {
    "medcpt_query": os.getenv(
        "EHRFORGE_MEDCPT_QUERY_MODEL", "ncbi/MedCPT-Query-Encoder"
    ),
    "medcpt_article": os.getenv(
        "EHRFORGE_MEDCPT_ARTICLE_MODEL", "ncbi/MedCPT-Article-Encoder"
    ),
    "nvembed": os.getenv("EHRFORGE_NVEMBED_MODEL", "nvidia/NV-Embed-v2"),
    "clinicalbert": os.getenv(
        "EHRFORGE_CLINICALBERT_MODEL", "emilyalsentzer/Bio_ClinicalBERT"
    ),
    "longformer": os.getenv(
        "EHRFORGE_LONGFORMER_MODEL",
        "allenai/longformer-large-4096-finetuned-triviaqa",
    ),
    "qwen2.5-7b": os.getenv(
        "EHRFORGE_QWEN7B_MODEL", "Qwen/Qwen2.5-7B-Instruct"
    ),
    "qwen2.5-32b": os.getenv(
        "EHRFORGE_QWEN32B_MODEL", "Qwen/Qwen2.5-32B-Instruct"
    ),
    "bertscore": os.getenv(
        "EHRFORGE_BERTSCORE_MODEL", "emilyalsentzer/Bio_ClinicalBERT"
    ),
}

BERTSCORE_LAYER = int(os.getenv("EHRFORGE_BERTSCORE_LAYER", "9"))
BERTSCORE_BATCH_SIZE = int(os.getenv("EHRFORGE_BERTSCORE_BATCH_SIZE", "64"))
BERTSCORE_CHUNK_SIZE = int(os.getenv("EHRFORGE_BERTSCORE_CHUNK_SIZE", "1024"))

QWEN_MAX_MODEL_LEN = int(os.getenv("EHRFORGE_QWEN_MAX_MODEL_LEN", "6144"))
QWEN_MAX_NEW_TOKENS = int(os.getenv("EHRFORGE_QWEN_MAX_NEW_TOKENS", "256"))
QWEN_CONTEXT_TOKEN_BUDGET = int(
    os.getenv("EHRFORGE_QWEN_CONTEXT_TOKEN_BUDGET", "5200")
)
QWEN_BATCH_SIZE = int(os.getenv("EHRFORGE_QWEN_BATCH_SIZE", "128"))

EXTRACTIVE_BATCH_SIZE = {
    "clinicalbert": int(os.getenv("EHRFORGE_CLINICALBERT_BATCH_SIZE", "32")),
    "longformer": int(os.getenv("EHRFORGE_LONGFORMER_BATCH_SIZE", "2")),
}
EXTRACTIVE_MAX_LENGTH = {
    "clinicalbert": int(os.getenv("EHRFORGE_CLINICALBERT_MAX_LENGTH", "512")),
    "longformer": int(os.getenv("EHRFORGE_LONGFORMER_MAX_LENGTH", "4096")),
}
EXTRACTIVE_MAX_ANSWER_TOKENS = int(
    os.getenv("EHRFORGE_MAX_ANSWER_TOKENS", "64")
)


NVEMBED_QUERY_INSTRUCTION = (
    "Instruct: Given a clinical question, retrieve the most relevant clinical "
    "visit notes.\nQuery: "
)

SYSTEM_PROMPT = (
    "You are a clinical assistant. Answer the question based strictly on the "
    "provided patient visit notes. Be concise and answer in 1-3 sentences."
)

REQUIRED_DATASET_COLUMNS = {
    "person_id",
    "qa_index",
    "question",
    "answer",
    "evidence",
    "reasoning_type",
    "visit_group",
    "difficulty",
}


def prediction_path(context_type: str, reader: str) -> Path:
    """Return the canonical prediction path for one experiment combination."""
    return PREDICTION_DIR / f"{context_type}__{reader}.jsonl"


def prediction_metrics_path(context_type: str, reader: str) -> Path:
    """Return the canonical per-instance metrics path for one combination."""
    return PREDICTION_METRICS_DIR / f"{context_type}__{reader}.jsonl"


def combination_key(context_type: str, reader: str) -> str:
    """Return the canonical retriever-reader key."""
    return f"{context_type}+{reader}"
