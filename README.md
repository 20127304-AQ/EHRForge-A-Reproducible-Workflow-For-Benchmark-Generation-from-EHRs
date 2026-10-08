# EHRForge: A Reproducible Workflow for Benchmark Generation from EHRs

EHRForge builds evidence-grounded temporal question-answering (QA) benchmarks from longitudinal clinical notes, and
evaluates retrieval and reader systems on them. This repository accompanies the revised manuscript and contains:

| Part | Where | What |
|---|---|---|
| Dataset-generation workflow (manuscript Section 3) | repository root: `analyze_dataset.py`, `build_patient_sequences.py`, `generate_temporal_qa.py`, `validate_evidence.py`, `prepare_qa_evaluation.py`, `ehrforge/`, `requirements.txt` | **Unchanged** from commit `e617df19c5a279714166a3937fabe36094a012fc` |
| Evaluation pipeline, protocol v1.4 (manuscript Section 4) | `evaluation/` | Steps 01–05: retrieval, retrieval metrics, readers, scoring, stratified analyses; frozen protocols v1.0–v1.4 |
| Environments | `environment/` | One pinned requirements file per compute image |
| Tests | `tests/` | Synthetic-fixture tests (no patient data) |

> **The old `baselines/` folder has been removed.** It looked up gold evidence with `visit_index` (a position inside
> the sampled chunk) instead of `original_visit_index` (the position in the full patient timeline), and it reported
> readers that did not match the models that actually ran. All numbers produced with it are invalid.

---

## 1. Data access

The raw clinical notes, original patient timelines, document identifiers, patient identifiers, instantiated benchmark
question and answer pairs, and verbatim evidence snippets are not publicly available because of patient privacy,
institutional governance, and data-use restrictions. The benchmark is synthetic at the question, answer, and labeling
levels but remains grounded in patient-derived timelines and evidence.

The patient-derived benchmark is provided through controlled access rather than as an open dataset. Researchers may
request access under the applicable governance framework. Access is subject to institutional approval, licensing and
data-use conditions, and is limited to research and evaluation purposes. The benchmark is not intended for clinical
decision-making. **Access-enquiry contact details and the request procedure will be updated here once the institutional
access route is finalized.** Fully synthetic benchmarks will be made available upon reasonable request.

This repository contains **no** raw clinical notes, patient identifiers, timelines, patient-derived QA pairs, evidence snippets,
reader contexts, predictions or per-question scores.

### Placing the restricted inputs

`evaluation/config.yaml` defines the workspace (default: `<repo>/workspace/`, outputs in `<repo>/out/`; both are
git-ignored). Every key can be overridden with an `EHRFORGE_<KEY>` environment variable.

```bash
mkdir -p workspace/data
cp /secure/location/dataset.csv workspace/data/dataset.csv
cp /secure/location/corpus.json workspace/data/corpus.json
sha256sum workspace/data/dataset.csv workspace/data/corpus.json
```

Expected sha256:

| File | sha256 | Content |
|---|---|---|
| `workspace/data/dataset.csv` | `66a968330bea85bef75f847a6a0c4e8bb8352e0ac363c85c75436b0ee034847f` | 10,742 QA pairs, 829 patients, 24,141 evidence items |
| `workspace/data/corpus.json` | `b77d4b75ff4a8bd6ce8c9138a718ae04c0d76bf3066218b762a2c887d1a5d0a2` | 961 patients, 170,827 visits |

Then populate the canonical files (protocols + shared data layer):

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r environment/requirements-cpu.txt
python evaluation/setup_workspace.py                         # copies evaluation/protocol/*, evaluation/common/* -> workspace/shared/canonical/ (sha-verified)
                                                              # and regenerates workspace/shared/canonical/judge_keys_1000.csv from dataset.csv when needed
```

Gold evidence is always mapped through **`original_visit_index`** with a visit-datetime assertion
(`evaluation/common/ehr_data.py`); rows are joined only on `(person_id, qa_index)`.

---

## 2. Dataset-generation pipeline

Unchanged upstream code. Install with `pip install -r requirements.txt`; QA generation needs `OPENAI_API_KEY` in the environment.

```bash
python analyze_dataset.py --input-csv inter_df_personid.csv --output-dir analysis_outputs
python build_patient_sequences.py --input-csv inter_df_personid.csv --output-jsonl patient_sequences.jsonl
python generate_temporal_qa.py --input-jsonl patient_sequences.jsonl --output-dir outputs --run-name batch_001 \
       --checkpoint-path outputs/processed_patient_ids.txt --batch-size 100 --model gpt-5.4-nano
python validate_evidence.py --original-jsonl patient_sequences.jsonl --generated-jsonl outputs/temporal_qa_batch_001.jsonl \
       --check-output-excel outputs/evidence_check_batch_001.xlsx --qa-flat-excel outputs/temporal_qa_batch_001_flat.xlsx \
       --qa-with-match-excel outputs/temporal_qa_batch_001_with_match.xlsx --min-part-match-ratio 1.0
python prepare_qa_evaluation.py --input-excel outputs/temporal_qa_batch_001_with_match.xlsx --output-jsonl outputs/qa_eval_batch_001.jsonl
```

Stages: corpus analysis → chronological visit-level timelines → timeline-length-aware context sampling →
schema-constrained generation → Round 1 temporal/structural filter → Round 2 exact evidence alignment → export.
In `generate_temporal_qa.py`, `visit_index` is by design the index *inside the prompt timeline*; each evidence item
also stores `original_visit_index` (full-timeline index), which is the only index the evaluation uses.

---

## 3. Evaluation pipeline

Each step asserts the sha256 of its frozen protocol and inputs before running. Run everything from the repository
root. Outputs go to `out/<step>/results` and `out/<step>/logs`. GPU steps run on [Modal](https://modal.com); Modal
object names (volumes, apps, HF secret) are set in `evaluation/config.yaml`.

| Step | Protocol | Compute | Environment / image |
|---|---|---|---|
| 01 retrieval | v1.1 | Modal: 1× A100-80GB (embeddings), CPU (ranking) | `environment/requirements-step1-main-image.txt`; NV-Embed-v2 in its **own** image `environment/requirements-step1-nvembed-image.txt` (transformers 4.42.4) |
| 02 retrieval metrics | v1.2 | local CPU | `environment/requirements-cpu.txt` |
| 03 readers (5 readers × 5 conditions = 25 runs) | v1.3 | Modal: A100-80GB (extractive), H100 80GB (Qwen 7B: 1×, 32B: 2×, TP=2) | `environment/requirements-step3-extractive-image.txt`; `environment/requirements-step3-vllm-0.28.0-image.txt` (official `vllm/vllm-openai:v0.28.0` image, digest pinned) |
| 04 scoring | v1.4 | Modal: 1 GPU (BERTScore), 2× H100 (judge); CPU aggregation | `environment/requirements-step4-bertscore-image.txt`; `environment/requirements-step4-judge-image.txt`; `environment/requirements-cpu.txt` |
| 05 stratified / sensitivity | v1.4 | local CPU | `environment/requirements-cpu.txt` |

Modal prerequisites: `pip install modal && modal setup`; create an HF secret
(`modal secret create huggingface-secret HF_TOKEN=...`). Mirror the workspace onto the step volumes with the same
relative layout (`/data/...`, `/shared/...`), e.g.:

```bash
modal volume create ehrforge-step1-v11
modal volume put ehrforge-step1-v11 workspace/data /data
modal volume put ehrforge-step1-v11 workspace/shared/canonical /shared/canonical
```

### 01 — Retrieval

```bash
modal run evaluation/01_retrieval/retrieval_v11.py::embed --model medcpt
modal run evaluation/01_retrieval/retrieval_v11.py::embed --model nvembed_v2
modal run evaluation/01_retrieval/retrieval_v11.py::rank --retriever bm25
modal run evaluation/01_retrieval/retrieval_v11.py::rank --retriever medcpt
modal run evaluation/01_retrieval/retrieval_v11.py::rank --retriever nvembed_v2
modal run evaluation/01_retrieval/retrieval_v11.py::hybrid          # RRF (k=60) over the FULL bm25 + medcpt rankings
```
Each stage is resumable, validates every protocol invariant and copies results to `out/01_retrieval/`. Add
`--smoke` / `--limit-shards 1` for a cheap test. Place the ranked files and `MANIFEST_rankings.csv` at
`workspace/shared/retrieval_v1.1/`.
`evaluation/01_retrieval/ehr_modal.py` holds the smoke/tokeniser cross-check/benchmark entrypoints.

### 02 — Retrieval metrics

```bash
python evaluation/02_retrieval_metrics/part12_metrics.py     # Hit/Recall/Coverage@k, MRR, exact random baselines, paired bootstrap
```

### 03 — Readers

```bash
modal volume put ehrforge-step3-v13 workspace/data /data
modal volume put ehrforge-step3-v13 workspace/shared/canonical /shared/canonical
modal volume put ehrforge-step3-v13 workspace/shared/retrieval_v1.1 /shared/retrieval_v1.1
modal run evaluation/03_readers/verify_volume_inputs.py
modal run evaluation/03_readers/probe_images.py
modal run evaluation/03_readers/build_contexts.py                        # 3,500 Qwen-token contexts, chronological, visit headers kept
modal run evaluation/03_readers/run_readers.py --mode prechecks          # load checks + SQuAD2-dev sanity check
python evaluation/03_readers/make_pilot_keys.py                          # 50-question pilot (seed 42)
modal run evaluation/03_readers/run_readers.py --mode extractive --reader roberta_base_squad2 --tag full
modal run evaluation/03_readers/run_readers.py --mode extractive --reader biobert_v1_1_pubmed_squad_v2 --tag full
modal run evaluation/03_readers/run_readers.py --mode extractive --reader longformer_squadv2 --tag full
modal run evaluation/03_readers/run_readers.py --mode generative --reader qwen2_5_7b --tag full
modal run evaluation/03_readers/run_readers.py --mode generative --reader qwen2_5_32b --tag full
python evaluation/03_readers/validate_runs.py full 10742
```
Run the pilot first with `--tag pilot --keys-file <pilot keys json>`. Predictions belong at
`workspace/shared/readers_v1.3/predictions/<reader>/<condition>.parquet`.

### 04 — Scoring

```bash
python evaluation/04_scoring/gate.py
python evaluation/04_scoring/build_inputs.py
modal run evaluation/04_scoring/s4_modal.py --mode upload_inputs
modal run evaluation/04_scoring/s4_modal.py --mode download_judge
modal run --detach evaluation/04_scoring/s4_modal.py --mode bertscore
python evaluation/04_scoring/assemble_per_row.py
python evaluation/04_scoring/build_pilot.py --tag pilot1
modal run --detach evaluation/04_scoring/s4_modal.py --mode judge --tag pilot1 --items out/04_scoring/results/pilot/pilot1/items.parquet
python evaluation/04_scoring/analyze_pilot.py --tag pilot1
# full judge input = 1,000 judge keys x 25 runs = 25,000 items; no standalone builder script is included
# the corresponding restricted intermediate must be staged at out/04_scoring/results/judge_full/items.parquet
modal run --detach evaluation/04_scoring/s4_modal.py --mode judge --tag full --items out/04_scoring/results/judge_full/items.parquet
python evaluation/04_scoring/aggregate.py
python evaluation/04_scoring/build_clinician_sheet.py
# after two clinicians fill the sheet:
python evaluation/04_scoring/analyze_clinician_agreement.py --help
```

### 05 — Stratified and sensitivity analyses

```bash
python evaluation/05_stratified/step4b.py
```

`step4b.py` is retained as the source implementation for the manuscript's stratified and sensitivity analyses. It
requires the intermediate outputs from Steps 02–04 to be present in the governed workspace, including
`workspace/shared/retrieval_tables_v1.2/`, `workspace/shared/readers_v1.3/`, and
`workspace/shared/scoring_v1.4/`. These intermediate outputs are not included in this public source-only repository.

---

## 4. Models

| Role | Model | Revision |
|---|---|---|
| Retriever (lexical) | BM25Okapi (`rank_bm25` 0.2.2; k1 1.5, b 0.75; lowercase + punctuation stripped) | — |
| Retriever (dense, biomedical) | `ncbi/MedCPT-Query-Encoder` / `ncbi/MedCPT-Article-Encoder` | `d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc` / `d05a736da4bb84ee4057b7f7999485be6ed85465` |
| Retriever (dense, general) | `nvidia/NV-Embed-v2` (own image, transformers 4.42.4) | `3fa59658547db50a1e8e3346cf057fd0c77ed6ef` |
| Retriever (hybrid) | `hybrid_rrf60`: reciprocal-rank fusion (k=60) of the **full** BM25 and MedCPT rankings | — |
| Reader (extractive) | `deepset/roberta-base-squad2` | `adc3b06f79f797d1c575d5479d6f5efe54a9e3b4` |
| Reader (extractive, biomedical) | `ktrapeznikov/biobert_v1.1_pubmed_squad_v2` — replaces `jon-t/Bio_ClinicalBERT_QA`, which failed the SQuAD2 sanity check (F1 10.4; trained on 100 examples / 18 steps) | `351a8218e59777dcb0a1b454ead77a0c39014bc5` |
| Reader (extractive, long) | `mrm8488/longformer-base-4096-finetuned-squadv2` | `e8039dc570012c25410a22131f83a77edfea3ed4` |
| Reader (generative) | `Qwen/Qwen2.5-7B-Instruct` via vLLM 0.28.0 (also the context-budget tokenizer) | `a09a35458c702b33eeacc393d103063234e8bc28` |
| Reader (generative) | `Qwen/Qwen2.5-32B-Instruct` via vLLM 0.28.0 | `5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd` |
| BERTScore (primary) | `FacebookAI/roberta-large`, layer 17 (bert-score 0.3.13, idf=False, no rescaling, fp32) | `722cf37b1afa9454edce342e7895e588b6ff1d59` |
| BERTScore (secondary) | `emilyalsentzer/Bio_ClinicalBERT`, layer 9 | `d5892b39a4adaed74b92212a44081509db72f87b` |
| LLM judge | `prometheus-eval/prometheus-8x7b-v2.0` (absolute 1–5 grading with reference answer; vLLM 0.28.0, 2× H100, TP 2, bf16, greedy, seed 42; 1,000 fixed judge keys) | `2db013b60e3e91f7a06113e436410899768e8228` |

Readers: stride 128, max answer 64 tokens, no-answer disabled (extractive); Qwen bf16 greedy, seed 42,
max_model_len 8192, 256 new tokens. Conditions: BM25, MedCPT, NV-Embed-v2, hybrid_rrf60 (top-5 from the full
timeline) and Oracle (all gold visits via `original_visit_index`). Confidence intervals: paired bootstrap, 1,000
resamples, seed 42.

---

## 5. Tests

```bash
pip install -r environment/requirements-cpu.txt
python -m pytest tests/ -q        # synthetic fixtures only; 1 test skips if prometheus_eval is not installed
```

## License

Apache License 2.0 (`LICENSE`, unchanged from the original repository). Model weights are subject to their own licenses.
