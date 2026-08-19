#!/usr/bin/env python3
"""Run corpus-level exploratory analysis for the EHRForge source CSV."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ehrforge.text_processing import (
    clean_text_for_eda,
    download_nltk_resources,
    lemmatize_tokens,
    tokenize_without_stopwords,
)


REQUIRED_COLUMNS = {"person_id", "Visit_DateTime", "doc_text"}


def load_source_data(csv_path: Path) -> pd.DataFrame:
    """Load the source CSV and prepare fields used by exploratory analysis."""
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV was not found: {csv_path}")

    dataframe = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    dataframe = dataframe.copy()
    dataframe["Visit_DateTime"] = pd.to_datetime(
        dataframe["Visit_DateTime"], unit="ms", errors="coerce"
    )
    if dataframe["Visit_DateTime"].isna().any():
        invalid_count = int(dataframe["Visit_DateTime"].isna().sum())
        raise ValueError(f"Found {invalid_count} invalid Visit_DateTime values.")

    dataframe["clean_text"] = dataframe["doc_text"].apply(clean_text_for_eda)
    dataframe["doc_length"] = dataframe["clean_text"].str.split().str.len()
    dataframe["year"] = dataframe["Visit_DateTime"].dt.year
    return dataframe.sort_values(
        ["person_id", "Visit_DateTime"], kind="stable"
    ).reset_index(drop=True)


def build_visit_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Aggregate note fragments recorded at the same patient timestamp."""
    aggregation: dict[str, Any] = {"clean_text": " ".join}
    if "doc_id" in dataframe.columns:
        aggregation["doc_id"] = list

    visit_dataframe = (
        dataframe.groupby(["person_id", "Visit_DateTime"], sort=True)
        .agg(aggregation)
        .reset_index()
    )
    visit_dataframe["visit_length"] = (
        visit_dataframe["clean_text"].str.split().str.len()
    )
    return visit_dataframe


def compute_statistics(
    dataframe: pd.DataFrame, visit_dataframe: pd.DataFrame
) -> dict[str, Any]:
    """Compute the descriptive statistics reported by the notebook."""
    visits_per_patient = visit_dataframe.groupby("person_id").size()
    docs_per_visit = dataframe.groupby(["person_id", "Visit_DateTime"]).size()
    docs_per_patient = dataframe.groupby("person_id").size()

    return {
        "source_rows": int(len(dataframe)),
        "unique_patients": int(dataframe["person_id"].nunique()),
        "missing_doc_text": int(dataframe["doc_text"].isna().sum()),
        "exact_duplicate_rows": int(dataframe.duplicated().sum()),
        "aggregated_visits": int(len(visit_dataframe)),
        "visits_per_patient": {
            "mean": float(visits_per_patient.mean()),
            "median": float(visits_per_patient.median()),
            "minimum": int(visits_per_patient.min()),
            "maximum": int(visits_per_patient.max()),
        },
        "documents_per_visit": {
            "mean": float(docs_per_visit.mean()),
            "median": float(docs_per_visit.median()),
            "maximum": int(docs_per_visit.max()),
        },
        "documents_per_patient": {
            "mean": float(docs_per_patient.mean()),
            "median": float(docs_per_patient.median()),
            "maximum": int(docs_per_patient.max()),
        },
        "document_length_words": {
            "mean": float(dataframe["doc_length"].mean()),
            "median": float(dataframe["doc_length"].median()),
            "maximum": int(dataframe["doc_length"].max()),
        },
        "visit_length_words": {
            "mean": float(visit_dataframe["visit_length"].mean()),
            "median": float(visit_dataframe["visit_length"].median()),
            "maximum": int(visit_dataframe["visit_length"].max()),
        },
        "notes_by_year": {
            str(int(year)): int(count)
            for year, count in dataframe["year"].value_counts().sort_index().items()
        },
    }


def save_histogram(
    values: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    bins: int = 50,
) -> None:
    """Save one histogram to disk."""
    figure = plt.figure()
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def save_year_bar_chart(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Save the yearly note-count distribution."""
    figure = plt.figure()
    dataframe["year"].value_counts().sort_index().plot(kind="bar")
    plt.title("Clinical-note timestamps by year")
    plt.xlabel("Year")
    plt.ylabel("Number of notes")
    plt.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def run_text_analysis(
    dataframe: pd.DataFrame,
    output_dir: Path,
    top_n: int,
    create_wordcloud: bool,
) -> None:
    """Run token-frequency, lemmatization, and optional word-cloud analysis."""
    download_nltk_resources()

    tokens = dataframe["clean_text"].apply(tokenize_without_stopwords)
    lemmas = tokens.apply(lemmatize_tokens)
    all_tokens = [token for row_tokens in tokens for token in row_tokens]
    most_common = Counter(all_tokens).most_common(top_n)

    frequency_path = output_dir / "top_words.json"
    frequency_path.write_text(
        json.dumps(
            [{"word": word, "count": count} for word, count in most_common],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if create_wordcloud:
        from wordcloud import WordCloud

        all_lemmas = " ".join(" ".join(row_lemmas) for row_lemmas in lemmas)
        cloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
        ).generate(all_lemmas)
        figure = plt.figure(figsize=(10, 5))
        plt.imshow(cloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud - All Documents")
        plt.tight_layout()
        figure.savefig(output_dir / "word_cloud.png", dpi=200)
        plt.close(figure)


def export_patient_notes(
    visit_dataframe: pd.DataFrame, person_id: int, output_path: Path
) -> None:
    """Export one patient's chronological visit notes to a text file."""
    patient_visits = visit_dataframe[
        visit_dataframe["person_id"].astype(int) == int(person_id)
    ]
    if patient_visits.empty:
        raise ValueError(f"No visits were found for person_id={person_id}.")

    with output_path.open("w", encoding="utf-8") as handle:
        for visit_number, row in enumerate(patient_visits.itertuples(), start=1):
            handle.write(
                f"--- Visit {visit_number} | {row.Visit_DateTime} ---\n"
            )
            handle.write(f"{row.clean_text}\n\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and visualize the row-level clinical-note corpus."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("inter_df_personid.csv"),
        help="Path to the source clinical-note CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs"),
        help="Directory for statistics and figures.",
    )
    parser.add_argument(
        "--text-analysis",
        action="store_true",
        help="Run NLTK tokenization, stopword removal, and lemmatization.",
    )
    parser.add_argument(
        "--wordcloud",
        action="store_true",
        help="Create a word cloud; requires --text-analysis.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of frequent tokens to save during text analysis.",
    )
    parser.add_argument(
        "--export-person-id",
        type=int,
        default=None,
        help="Optional patient identifier whose chronological notes should be exported.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = load_source_data(args.input_csv)
    visit_dataframe = build_visit_dataframe(dataframe)
    statistics = compute_statistics(dataframe, visit_dataframe)

    statistics_path = args.output_dir / "dataset_statistics.json"
    statistics_path.write_text(
        json.dumps(statistics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    visits_per_patient = visit_dataframe.groupby("person_id").size()
    save_histogram(
        visits_per_patient,
        "Distribution of visits per patient",
        "Number of visits per patient",
        "Number of patients",
        args.output_dir / "visits_per_patient.png",
    )
    save_histogram(
        dataframe["doc_length"],
        "Document length distribution",
        "Document length in words",
        "Number of documents",
        args.output_dir / "document_length_distribution.png",
    )
    save_year_bar_chart(dataframe, args.output_dir / "notes_by_year.png")

    if args.text_analysis:
        run_text_analysis(
            dataframe=dataframe,
            output_dir=args.output_dir,
            top_n=args.top_n,
            create_wordcloud=args.wordcloud,
        )
    elif args.wordcloud:
        raise ValueError("--wordcloud requires --text-analysis.")

    if args.export_person_id is not None:
        export_patient_notes(
            visit_dataframe,
            args.export_person_id,
            args.output_dir / f"patient_{args.export_person_id}_notes.txt",
        )

    print(json.dumps(statistics, indent=2, ensure_ascii=False))
    print(f"\nSaved analysis outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
