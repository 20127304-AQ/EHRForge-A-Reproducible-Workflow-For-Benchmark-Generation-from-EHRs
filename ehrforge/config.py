"""Configuration objects for the EHRForge data-generation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class GenerationConfig:
    """Runtime configuration for temporal QA generation."""

    input_jsonl: Path = Path("patient_sequences.jsonl")
    output_dir: Path = Path(".")
    run_name: str = "batch_001"
    checkpoint_path: Path = Path("processed_patient_ids.txt")

    batch_size: int = 100
    random_seed: int = 42
    model_name: str = "gpt-5.4-nano"
    only_visit_group: Optional[str] = None

    max_qas_small_group: int = 10
    max_qas_medium_group: int = 10
    max_qas_per_chunk: int = 5

    max_visits_medium_group: int = 30
    early_visits_medium_group: int = 12
    late_visits_medium_group: int = 18

    chunk_size_long_group: int = 30
    chunk_overlap_long_group: int = 5
    global_early_visits_long_group: int = 15
    global_late_visits_long_group: int = 15
    max_qas_global_chunk: int = 5

    max_chars_per_visit: int = 2500
    sleep_between_calls: float = 0.5
    max_retries: int = 3
    retry_sleep: float = 3.0

    strict_temporal_filter: bool = False
    require_multi_visit_for_temporal: bool = True
    reset_checkpoint: bool = False
    mark_failed_as_processed: bool = True

    def validate(self) -> None:
        """Validate related configuration values before a run starts."""
        allowed_groups = {
            None,
            "2-10 visits",
            "11-100 visits",
            "101-1000 visits",
        }
        if self.only_visit_group not in allowed_groups:
            raise ValueError(
                "only_visit_group must be one of: None, '2-10 visits', "
                "'11-100 visits', or '101-1000 visits'."
            )
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if self.chunk_size_long_group <= 0:
            raise ValueError("chunk_size_long_group must be greater than zero.")
        if not 0 <= self.chunk_overlap_long_group < self.chunk_size_long_group:
            raise ValueError(
                "chunk_overlap_long_group must be non-negative and smaller than "
                "chunk_size_long_group."
            )
        if self.max_retries <= 0:
            raise ValueError("max_retries must be greater than zero.")

    @property
    def output_jsonl(self) -> Path:
        return self.output_dir / f"temporal_qa_{self.run_name}.jsonl"

    @property
    def output_flat_excel(self) -> Path:
        return self.output_dir / f"temporal_qa_{self.run_name}_flat.xlsx"

    @property
    def output_run_errors_jsonl(self) -> Path:
        return self.output_dir / f"temporal_qa_{self.run_name}_run_errors.jsonl"

    @property
    def output_qa_rejections_jsonl(self) -> Path:
        return self.output_dir / f"temporal_qa_{self.run_name}_qa_rejections.jsonl"

    @property
    def output_stats_json(self) -> Path:
        return self.output_dir / f"temporal_qa_{self.run_name}_stats.json"
