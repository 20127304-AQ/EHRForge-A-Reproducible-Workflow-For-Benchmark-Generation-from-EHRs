"""Run ClinicalBERT, Longformer, Qwen2.5-7B, and Qwen2.5-32B readers."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Iterable

import modal

from ehrforge_repro.config import (
    APP_PREFIX,
    CONTEXT_TYPES,
    CORPUS_PATH,
    DATA_VOLUME_NAME,
    DATASET_PATH,
    EXTRACTIVE_BATCH_SIZE,
    EXTRACTIVE_MAX_ANSWER_TOKENS,
    EXTRACTIVE_MAX_LENGTH,
    HF_SECRET_NAME,
    MANIFEST_DIR,
    MODEL_IDS,
    PREDICTION_DIR,
    QWEN_BATCH_SIZE,
    QWEN_CONTEXT_TOKEN_BUDGET,
    QWEN_MAX_MODEL_LEN,
    QWEN_MAX_NEW_TOKENS,
    RANDOM_SEED,
    RESULTS_VOLUME_NAME,
    RETRIEVAL_RESULTS_PATH,
    prediction_path,
)
from ehrforge_repro.contexts import (
    build_context,
    build_context_with_token_budget,
    build_qwen_messages,
    select_visit_indices,
)
from ehrforge_repro.data import load_corpus, load_dataset, load_retrieval_map
from ehrforge_repro.io_utils import (
    append_jsonl,
    file_sha256,
    read_jsonl_deduplicated,
    write_json_atomic,
    write_jsonl_atomic,
)


app = modal.App(f"{APP_PREFIX}-readers")
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False)
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)
hf_cache = modal.Volume.from_name("huggingface-cache-qa", create_if_missing=True)

extractive_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.11"
    )
    .entrypoint([])
    .pip_install(
        "torch==2.4.0",
        "transformers==4.44.2",
        "numpy<2",
        "pandas>=2.1,<3",
        "tqdm>=4.66,<5",
        "sentencepiece>=0.2,<1",
        "accelerate>=0.33,<1",
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

qwen_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04", add_python="3.11"
    )
    .entrypoint([])
    .pip_install(
        "vllm==0.6.3",
        "torch==2.4.0",
        "torchvision==0.19.0",
        "numpy<2",
        "transformers==4.44.2",
        "pandas>=2.1,<3",
        "tqdm>=4.66,<5",
        "sentencepiece>=0.2,<1",
        "accelerate>=0.33,<1",
    )
    .env(
        {
            "HF_HOME": "/root/.cache/huggingface",
            "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
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


def _parse_csv_selection(value: str, valid_values: Iterable[str]) -> list[str]:
    valid = tuple(valid_values)
    if value.strip().lower() == "all":
        return list(valid)
    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [item for item in selected if item not in valid]
    if unknown:
        raise ValueError(f"Unknown values {unknown}; expected a subset of {valid}")
    if not selected:
        raise ValueError("At least one value must be selected")
    return list(dict.fromkeys(selected))


def _record_for_prediction(
    row: dict[str, Any],
    context_type: str,
    reader: str,
    model_id: str,
    prediction: str,
    selected_indices: list[int],
) -> dict[str, Any]:
    return {
        "person_id": int(row["person_id"]),
        "qa_index": int(row["qa_index"]),
        "row_index": int(row["_row_index"]),
        "retriever": context_type,
        "reader": reader,
        "model_id": model_id,
        "prediction": prediction.strip(),
        "gold_answer": str(row.get("answer", "")),
        "reasoning_type": str(row.get("reasoning_type", "")),
        "visit_group": str(row.get("visit_group", "")),
        "difficulty": str(row.get("difficulty", "")),
        "selected_visit_indices": [int(value) for value in selected_indices],
    }


def _best_extractive_spans(
    model: Any,
    tokenizer: Any,
    questions: list[str],
    contexts: list[str],
    max_length: int,
    max_answer_tokens: int,
    longformer: bool,
) -> list[str]:
    import torch

    predictions = [""] * len(questions)
    active_positions = [index for index, context in enumerate(contexts) if context.strip()]
    if not active_positions:
        return predictions

    active_questions = [questions[index] for index in active_positions]
    active_contexts = [contexts[index] for index in active_positions]
    encoded = tokenizer(
        active_questions,
        active_contexts,
        padding=True,
        truncation="only_second",
        max_length=max_length,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    sequence_ids = [encoded.sequence_ids(index) for index in range(len(active_positions))]
    offsets = encoded.pop("offset_mapping").tolist()
    model_inputs = {key: value.to("cuda") for key, value in encoded.items()}

    if longformer:
        global_attention = torch.zeros_like(model_inputs["input_ids"])
        for batch_index, sequence in enumerate(sequence_ids):
            for token_index, sequence_id in enumerate(sequence):
                if sequence_id == 0:
                    global_attention[batch_index, token_index] = 1
            global_attention[batch_index, 0] = 1
        model_inputs["global_attention_mask"] = global_attention

    with torch.inference_mode():
        output = model(**model_inputs)

    start_logits = output.start_logits.detach().cpu()
    end_logits = output.end_logits.detach().cpu()
    attention_mask = model_inputs["attention_mask"].detach().cpu()

    for local_index, original_position in enumerate(active_positions):
        context_positions = [
            token_index
            for token_index, sequence_id in enumerate(sequence_ids[local_index])
            if sequence_id == 1
            and attention_mask[local_index, token_index].item() == 1
            and offsets[local_index][token_index][1]
            > offsets[local_index][token_index][0]
        ]
        if not context_positions:
            continue

        start_candidates = sorted(
            context_positions,
            key=lambda index: float(start_logits[local_index, index]),
            reverse=True,
        )[:20]
        end_candidates = sorted(
            context_positions,
            key=lambda index: float(end_logits[local_index, index]),
            reverse=True,
        )[:20]

        best: tuple[float, int, int] | None = None
        for start_index in start_candidates:
            for end_index in end_candidates:
                if end_index < start_index:
                    continue
                if end_index - start_index + 1 > max_answer_tokens:
                    continue
                score = float(start_logits[local_index, start_index]) + float(
                    end_logits[local_index, end_index]
                )
                if best is None or score > best[0]:
                    best = (score, start_index, end_index)

        if best is None:
            continue
        _, start_index, end_index = best
        start_character = offsets[local_index][start_index][0]
        end_character = offsets[local_index][end_index][1]
        predictions[original_position] = active_contexts[local_index][
            start_character:end_character
        ].strip()

    return predictions


def _prepare_experiment_rows(
    context_type: str,
    frame: Any,
    corpus: dict[int, list[dict[str, Any]]],
    retrieval_map: dict[tuple[int, int], dict[str, Any]],
) -> list[tuple[dict[str, Any], list[int], str]]:
    prepared: list[tuple[dict[str, Any], list[int], str]] = []
    for row in frame.to_dict("records"):
        key = (int(row["person_id"]), int(row["qa_index"]))
        retrieval_record = retrieval_map.get(key)
        if retrieval_record is None:
            raise KeyError(f"Missing retrieval record for sample {key}")
        indices = select_visit_indices(row, retrieval_record, context_type)
        context = build_context(corpus.get(key[0], []), indices)
        prepared.append((row, indices, context))
    return prepared


@app.function(
    image=extractive_image,
    volumes=volume_mounts,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    gpu="A100-80GB",
    memory=131072,
    timeout=24 * 60 * 60,
)
def run_extractive_reader(
    reader: str,
    contexts: str = "all",
    resume: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run one extractive reader over selected retrieval and oracle contexts."""
    import gc
    import random

    import numpy as np
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    if reader not in {"clinicalbert", "longformer"}:
        raise ValueError("reader must be clinicalbert or longformer")

    selected_contexts = _parse_csv_selection(contexts, CONTEXT_TYPES)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    results_volume.reload()
    frame = load_dataset(DATASET_PATH)
    corpus = load_corpus(CORPUS_PATH)
    retrieval_map = load_retrieval_map(RETRIEVAL_RESULTS_PATH)
    model_id = MODEL_IDS[reader]
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model, loading_info = AutoModelForQuestionAnswering.from_pretrained(
        model_id,
        output_loading_info=True,
        ignore_mismatched_sizes=True,
    )
    model = model.cuda().eval()

    max_length = EXTRACTIVE_MAX_LENGTH[reader]
    batch_size = EXTRACTIVE_BATCH_SIZE[reader]
    summary: dict[str, Any] = {}
    started = time.time()

    for context_type in selected_contexts:
        output_path = prediction_path(context_type, reader)
        if overwrite and output_path.exists():
            output_path.unlink()
        existing, invalid_lines, duplicate_lines = read_jsonl_deduplicated(output_path)
        if not resume and existing:
            raise RuntimeError(
                f"Prediction file already exists for {context_type}+{reader}; "
                "use overwrite=True or resume=True"
            )
        completed = {
            (int(record["person_id"]), int(record["qa_index"]))
            for record in existing
        }

        prepared = _prepare_experiment_rows(
            context_type, frame, corpus, retrieval_map
        )
        todo = [
            item
            for item in prepared
            if (int(item[0]["person_id"]), int(item[0]["qa_index"])) not in completed
        ]

        for start in tqdm(
            range(0, len(todo), batch_size),
            desc=f"{context_type}+{reader}",
        ):
            batch = todo[start : start + batch_size]
            questions = [str(row["question"]) for row, _, _ in batch]
            context_strings = [context for _, _, context in batch]
            batch_predictions = _best_extractive_spans(
                model,
                tokenizer,
                questions,
                context_strings,
                max_length=max_length,
                max_answer_tokens=EXTRACTIVE_MAX_ANSWER_TOKENS,
                longformer=reader == "longformer",
            )
            new_records = [
                _record_for_prediction(
                    row,
                    context_type,
                    reader,
                    model_id,
                    prediction,
                    indices,
                )
                for (row, indices, _), prediction in zip(batch, batch_predictions)
            ]
            append_jsonl(output_path, new_records)
            if (start // batch_size + 1) % 10 == 0:
                results_volume.commit()

        final_records, invalid_after, duplicates_after = read_jsonl_deduplicated(
            output_path
        )
        write_jsonl_atomic(output_path, final_records)
        results_volume.commit()
        summary[context_type] = {
            "n_predictions": len(final_records),
            "expected": len(frame),
            "complete": len(final_records) == len(frame),
            "invalid_lines_removed": invalid_lines + invalid_after,
            "duplicate_lines_removed": duplicate_lines + duplicates_after,
            "path": str(output_path),
        }

    manifest = {
        "reader": reader,
        "model": model_id,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "contexts": selected_contexts,
        "max_length": max_length,
        "batch_size": batch_size,
        "max_answer_tokens": EXTRACTIVE_MAX_ANSWER_TOKENS,
        "fallback_used": False,
        "missing_keys": loading_info.get("missing_keys", []),
        "unexpected_keys": loading_info.get("unexpected_keys", []),
        "mismatched_keys": [
            str(item) for item in loading_info.get("mismatched_keys", [])
        ],
        "dataset_sha256": file_sha256(DATASET_PATH),
        "elapsed_seconds": time.time() - started,
        "outputs": summary,
    }
    write_json_atomic(MANIFEST_DIR / f"reader_{reader}.json", manifest)
    results_volume.commit()

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return manifest


def _run_qwen(
    reader: str,
    contexts: str,
    resume: bool,
    overwrite: bool,
    batch_size: int,
) -> dict[str, Any]:
    import gc
    import random

    import numpy as np
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    selected_contexts = _parse_csv_selection(contexts, CONTEXT_TYPES)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    results_volume.reload()
    frame = load_dataset(DATASET_PATH)
    corpus = load_corpus(CORPUS_PATH)
    retrieval_map = load_retrieval_map(RETRIEVAL_RESULTS_PATH)
    model_id = MODEL_IDS[reader]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        max_model_len=QWEN_MAX_MODEL_LEN,
        gpu_memory_utilization=0.92 if reader == "qwen2.5-32b" else 0.90,
        trust_remote_code=True,
        enforce_eager=True,
        tensor_parallel_size=1,
    )
    sampling = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=QWEN_MAX_NEW_TOKENS,
        stop=["<|im_end|>"],
    )

    summary: dict[str, Any] = {}
    started = time.time()
    for context_type in selected_contexts:
        output_path = prediction_path(context_type, reader)
        if overwrite and output_path.exists():
            output_path.unlink()
        existing, invalid_lines, duplicate_lines = read_jsonl_deduplicated(output_path)
        if not resume and existing:
            raise RuntimeError(
                f"Prediction file already exists for {context_type}+{reader}; "
                "use overwrite=True or resume=True"
            )
        completed = {
            (int(record["person_id"]), int(record["qa_index"]))
            for record in existing
        }

        todo: list[tuple[dict[str, Any], list[int], str]] = []
        for row in frame.to_dict("records"):
            key = (int(row["person_id"]), int(row["qa_index"]))
            if key in completed:
                continue
            retrieval_record = retrieval_map.get(key)
            if retrieval_record is None:
                raise KeyError(f"Missing retrieval record for sample {key}")
            indices = select_visit_indices(row, retrieval_record, context_type)
            context = build_context_with_token_budget(
                corpus.get(key[0], []),
                indices,
                tokenizer,
                QWEN_CONTEXT_TOKEN_BUDGET,
            )
            messages = build_qwen_messages(str(row["question"]), context)
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            todo.append((row, indices, prompt))

        for start in tqdm(
            range(0, len(todo), batch_size),
            desc=f"{context_type}+{reader}",
        ):
            batch = todo[start : start + batch_size]
            prompts = [prompt for _, _, prompt in batch]
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} outputs for {len(batch)} prompts"
                )
            new_records = [
                _record_for_prediction(
                    row,
                    context_type,
                    reader,
                    model_id,
                    output.outputs[0].text,
                    indices,
                )
                for (row, indices, _), output in zip(batch, outputs)
            ]
            append_jsonl(output_path, new_records)
            if (start // batch_size + 1) % 4 == 0:
                results_volume.commit()

        final_records, invalid_after, duplicates_after = read_jsonl_deduplicated(
            output_path
        )
        write_jsonl_atomic(output_path, final_records)
        results_volume.commit()
        summary[context_type] = {
            "n_predictions": len(final_records),
            "expected": len(frame),
            "complete": len(final_records) == len(frame),
            "invalid_lines_removed": invalid_lines + invalid_after,
            "duplicate_lines_removed": duplicate_lines + duplicates_after,
            "path": str(output_path),
        }

    manifest = {
        "reader": reader,
        "model": model_id,
        "model_revision": tokenizer.init_kwargs.get("_commit_hash"),
        "contexts": selected_contexts,
        "max_model_len": QWEN_MAX_MODEL_LEN,
        "context_token_budget": QWEN_CONTEXT_TOKEN_BUDGET,
        "max_new_tokens": QWEN_MAX_NEW_TOKENS,
        "temperature": 0.0,
        "batch_size": batch_size,
        "backend": "vllm",
        "fallback_used": False,
        "dataset_sha256": file_sha256(DATASET_PATH),
        "elapsed_seconds": time.time() - started,
        "outputs": summary,
    }
    write_json_atomic(MANIFEST_DIR / f"reader_{reader}.json", manifest)
    results_volume.commit()

    del llm
    torch.cuda.empty_cache()
    gc.collect()
    return manifest


@app.function(
    image=qwen_image,
    volumes=volume_mounts,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    gpu="A100-80GB",
    memory=131072,
    timeout=24 * 60 * 60,
)
def run_qwen7b(
    contexts: str = "all",
    resume: bool = True,
    overwrite: bool = False,
    batch_size: int = QWEN_BATCH_SIZE,
) -> dict[str, Any]:
    """Run Qwen2.5-7B-Instruct over selected contexts."""
    return _run_qwen("qwen2.5-7b", contexts, resume, overwrite, batch_size)


@app.function(
    image=qwen_image,
    volumes=volume_mounts,
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    gpu="A100-80GB",
    memory=131072,
    timeout=24 * 60 * 60,
)
def run_qwen32b(
    contexts: str = "all",
    resume: bool = True,
    overwrite: bool = False,
    batch_size: int = QWEN_BATCH_SIZE,
) -> dict[str, Any]:
    """Run Qwen2.5-32B-Instruct over selected contexts."""
    return _run_qwen("qwen2.5-32b", contexts, resume, overwrite, batch_size)


@app.local_entrypoint()
def main(
    readers: str = "all",
    contexts: str = "all",
    resume: bool = True,
    overwrite: bool = False,
    qwen_batch_size: int = QWEN_BATCH_SIZE,
) -> None:
    """Run selected readers sequentially."""
    selected_readers = _parse_csv_selection(
        readers,
        ("clinicalbert", "longformer", "qwen2.5-7b", "qwen2.5-32b"),
    )
    _parse_csv_selection(contexts, CONTEXT_TYPES)

    for reader in selected_readers:
        if reader in {"clinicalbert", "longformer"}:
            result = run_extractive_reader.remote(
                reader=reader,
                contexts=contexts,
                resume=resume,
                overwrite=overwrite,
            )
        elif reader == "qwen2.5-7b":
            result = run_qwen7b.remote(
                contexts=contexts,
                resume=resume,
                overwrite=overwrite,
                batch_size=qwen_batch_size,
            )
        else:
            result = run_qwen32b.remote(
                contexts=contexts,
                resume=resume,
                overwrite=overwrite,
                batch_size=qwen_batch_size,
            )
        print(result)

    print(
        "Predictions are stored in Modal volume "
        f"{RESULTS_VOLUME_NAME} under {PREDICTION_DIR}."
    )
