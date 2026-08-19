# EHRForge Experiment Reproduction

This package reproduces the retrieval and question-answering experiments for the EHRForge longitudinal clinical QA dataset.

It evaluates:

* Retrieval: BM25, MedCPT, NV-Embed-v2, and Hybrid BM25 + MedCPT.
* Readers: ClinicalBERT, Longformer, Qwen2.5-7B-Instruct, and Qwen2.5-32B-Instruct.
* Contexts: four retrieval outputs and Oracle Evidence.

The full experiment matrix contains 20 retrieval-reader combinations.

## Metrics

The package computes:

* Evidence Recall@5, @10, and @20
* Exact Coverage@5, @10, and @20
* Exact Match
* Token F1
* BERTScore precision, recall, and F1
* Aggregates by difficulty, visit group, and reasoning type

Aggregate metrics are saved as JSON. Per-instance metrics are saved as JSONL.

## Requirements

* Python 3.11
* Modal account
* NVIDIA A100-80GB availability
* Hugging Face access to `nvidia/NV-Embed-v2`
* Modal volumes:

  * `clinical-qa-data`
  * `ehrforge-experiment-results`
* Modal secret:

  * `huggingface-secret` containing `HF_TOKEN`

Install and configure the environment:

```bash
python -m pip install -e .
modal setup

modal volume create clinical-qa-data
modal volume create ehrforge-experiment-results
modal secret create huggingface-secret HF_TOKEN=hf_your_token
```

## Input data

Upload the dataset and corpus:

```bash
modal volume put clinical-qa-data data/dataset.csv /dataset.csv
modal volume put clinical-qa-data data/corpus.json /corpus.json
```

Required dataset columns:

```text
person_id, qa_index, question, answer, evidence,
reasoning_type, visit_group, difficulty
```

Expected corpus format:

```json
{
  "person_id": 1,
  "visits": [
    {
      "visit_datetime": "2020-01-01",
      "text": "..."
    }
  ]
}
```

## Run experiments

Run the complete pipeline:

```bash
bash run_all.sh
```

Equivalent commands:

```bash
modal run scripts/01_retrieval.py --phase all
modal run scripts/02_readers.py --readers all --contexts all
modal run scripts/03_metrics.py --mode all --combinations all
modal run scripts/04_validate.py --strict
```

Prediction jobs resume automatically using `(person_id, qa_index)`. Use `--overwrite` only when existing outputs should be replaced.

## Run selected experiments

Example:

```bash
modal run scripts/01_retrieval.py --phase hybrid

modal run scripts/02_readers.py \
  --readers qwen2.5-7b,qwen2.5-32b \
  --contexts hybrid,oracle

modal run scripts/03_metrics.py \
  --mode qa \
  --combinations hybrid+qwen2.5-7b,hybrid+qwen2.5-32b
```

## Outputs

Results are written to the `ehrforge-experiment-results` Modal volume:

```text
/retrieval/
/predictions/
/predictions_with_metrics/
/metrics/
/manifests/
/validation/
```

Download all results:

```bash
modal volume get ehrforge-experiment-results / ./ehrforge_results
```

## Validation and reproducibility

Strict validation checks that retrieval results, predictions, metrics, model identities, and all 20 experiment combinations are complete and consistent:

```bash
modal run scripts/04_validate.py --strict
```

Configuration values, model IDs, data hashes, metric settings, and output counts are recorded in JSON manifests. Environment-variable overrides are defined in `ehrforge_repro/config.py`.
