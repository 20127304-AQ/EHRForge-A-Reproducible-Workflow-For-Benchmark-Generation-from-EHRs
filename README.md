# EHRForge: A reproducible workflow for constructing evidence-grounded synthetic temporal question-answering benchmarks from longitudinal clinical notes

EHRForge is a modular and privacy-conscious research workflow for building temporal question-answering (QA) benchmarks from longitudinal electronic health record (EHR) notes.

The workflow converts raw clinical-note rows into chronological patient timelines, selects context according to timeline length, generates schema-constrained QA candidates, and validates them through temporal filtering and exact evidence alignment.

> This repository contains workflow code and non-sensitive research components. It does not include raw clinical notes, patient identifiers, patient timelines, verbatim evidence, or the restricted institutional benchmark.

---

## Features

* Chronological patient-timeline construction
* Visit-level note aggregation
* Timeline-length-aware context selection
* Schema-constrained temporal QA generation
* Temporal and structural validation
* Exact evidence-to-visit matching
* Resumable batch processing
* Retrieval and reader baselines
* Privacy-conscious data handling

---

## Workflow

```text
Raw clinical-note rows
          │
          ▼
Dataset analysis
          │
          ▼
Chronological patient timelines
          │
          ▼
Context selection
          │
          ▼
Temporal QA generation
          │
          ▼
Temporal and structural filtering
          │
          ▼
Exact evidence alignment
          │
          ▼
Evaluation-ready QA records
```

Generated QA pairs are treated as provisional candidates. They must pass the configured validation stages before being included in an evaluation dataset.

---

## Repository Structure

```text
EHRForge/
├── .env.example
├── README.md
├── requirements.txt
│
├── analyze_dataset.py
├── build_patient_sequences.py
├── generate_temporal_qa.py
├── prepare_qa_evaluation.py
├── validate_evidence.py
│
├── ehrforge/
│   ├── __init__.py
│   ├── config.py
│   ├── io_utils.py
│   └── text_processing.py
│
└── baselines/
    ├── README.md
    ├── pyproject.toml
    ├── run_all.sh
    │
    ├── ehrforge_repro/
    │   ├── __init__.py
    │   ├── config.py
    │   ├── contexts.py
    │   ├── data.py
    │   ├── io_utils.py
    │   └── metrics.py
    │
    ├── scripts/
    │   ├── 01_retrieval.py
    │   ├── 02_readers.py
    │   ├── 03_metrics.py
    │   └── 04_validate.py
    │
    └── tests/
        ├── test_contexts_and_io.py
        └── test_metrics.py
```

### Main Scripts

| File                         | Purpose                                  |
| ---------------------------- | ---------------------------------------- |
| `analyze_dataset.py`         | Analyzes the source dataset              |
| `build_patient_sequences.py` | Builds chronological patient timelines   |
| `generate_temporal_qa.py`    | Generates temporal QA candidates         |
| `validate_evidence.py`       | Validates evidence against source visits |
| `prepare_qa_evaluation.py`   | Prepares evaluation-ready QA records     |

The `baselines` directory contains retrieval, reader, metric, and validation experiments.

---

## Installation

Python 3.10 or later is recommended.

```bash
git clone <repository-url>
cd EHRForge

python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux or macOS
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Install the baseline package when required:

```bash
pip install -e ./baselines
```

---

## Configuration

Review the configuration files before running the workflow:

```text
ehrforge/config.py
baselines/ehrforge_repro/config.py
```

Configuration options include:

* source-column mappings;
* patient eligibility rules;
* context-sampling parameters;
* model settings;
* output paths; and
* evidence-matching thresholds.

Set the OpenAI API key before QA generation:

```powershell
# Windows PowerShell
$env:OPENAI_API_KEY="your_api_key_here"
```

```bash
# Linux or macOS
export OPENAI_API_KEY="your_api_key_here"
```

Never commit API keys or local `.env` files.

---

## Input Data

EHRForge expects a CSV file containing longitudinal clinical-note rows.

The default workflow uses fields such as:

| Field            | Description                            |
| ---------------- | -------------------------------------- |
| `Visit_DateTime` | Clinical-note or visit timestamp       |
| `person_id`      | Internal patient identifier            |
| `doc_id`         | Source-document identifier             |
| `doc_text`       | Clinical-note text                     |
| `URNumber`       | Institution-specific record identifier |

Column names can be changed in the project configuration.

Notes are grouped by patient and visit time, normalized, aggregated when necessary, and sorted chronologically.

---

## Minimal Usage

Display the available arguments for a script:

```bash
python <script_name>.py --help
```

A typical workflow is:

```bash
python analyze_dataset.py \
  --input-csv inter_df_personid.csv \
  --output-dir analysis_outputs
```

```bash
python build_patient_sequences.py \
  --input-csv inter_df_personid.csv \
  --output-jsonl patient_sequences.jsonl
```

```bash
python generate_temporal_qa.py \
  --input-jsonl patient_sequences.jsonl \
  --output-dir outputs \
  --run-name batch_001 \
  --checkpoint-path outputs/processed_patient_ids.txt \
  --batch-size 100 \
  --model gpt-5.4-nano
```

```bash
python validate_evidence.py \
  --original-jsonl patient_sequences.jsonl \
  --generated-jsonl outputs/temporal_qa_batch_001.jsonl \
  --check-output-excel outputs/evidence_check_batch_001.xlsx \
  --qa-flat-excel outputs/temporal_qa_batch_001_flat.xlsx \
  --qa-with-match-excel outputs/temporal_qa_batch_001_with_match.xlsx \
  --min-part-match-ratio 1.0
```

```bash
python prepare_qa_evaluation.py \
  --input-excel outputs/temporal_qa_batch_001_with_match.xlsx \
  --output-jsonl outputs/qa_eval_batch_001.jsonl
```

Use a unique `--run-name` for each batch. Reuse the checkpoint file to avoid processing the same patients again.

---

## Validation

EHRForge applies two validation stages:

1. **Temporal and structural filtering** checks required fields, labels, visit references, evidence order, multi-visit requirements, and duplicates.
2. **Evidence alignment** verifies that every cited evidence snippet occurs in its referenced source visit.

Use `--min-part-match-ratio 1.0` for strict evidence matching.

---

## Case-Study Results

One governed institutional execution produced:

| Stage                                    |  Count |
| ---------------------------------------- | -----: |
| Generated QA candidates                  | 27,558 |
| Passed temporal and structural filtering | 18,867 |
| Fully evidence-aligned QA pairs          | 10,742 |
| Patients represented                     |    829 |

These results describe one restricted case study and may vary across datasets and configurations.

---

## Privacy and Data Availability

This repository does not contain:

* raw clinical notes;
* patient or document identifiers;
* original patient timelines;
* patient-grounded QA pairs; or
* verbatim clinical evidence.

Generated outputs may still contain sensitive patient-derived information and should not be committed to a public repository.

EHRForge is intended for use with appropriately authorized data under applicable institutional governance and privacy requirements. It is not intended for clinical decision-making.

---

## Testing

Run the baseline tests with:

```bash
python -m pytest baselines/tests
```

---

## Citation

When using EHRForge in academic work, please cite:

> Quang An Quoc Tran, Huy Quoc To, and Ming Liu.
> **EHRForge: A Reproducible Workflow for Constructing an Evidence-Grounded Synthetic Temporal QA Benchmark from Longitudinal Clinical Notes.**
> Manuscript submitted for publication, 2026.

Publication details and a BibTeX entry will be added after publication.

---

## License

The EHRForge source code is licensed under the [Apache License 2.0](LICENSE).

The license applies only to the source code and explicitly released public artifacts. It does not apply to restricted clinical data or patient-derived benchmark instances.

---

## Disclaimer

EHRForge is research software for benchmark construction and evaluation. It must not be used for diagnosis, treatment recommendations, clinical decision-making, or direct patient care.
