"""Text-normalization helpers for clinical notes and exploratory analysis."""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd


def clean_clinical_text(text: object) -> str:
    """Apply light normalization while preserving clinical punctuation and casing."""
    if pd.isna(text):
        return ""
    normalized = str(text)
    normalized = re.sub(r"\n+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def clean_text_for_eda(text: object) -> str:
    """Normalize text for token-frequency and word-cloud analysis."""
    if pd.isna(text):
        return ""
    normalized = str(text).lower()
    normalized = re.sub(r"\n+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^a-zA-Z0-9 ]", "", normalized)
    return normalized.strip()


def download_nltk_resources() -> None:
    """Download the NLTK resources used by the exploratory-analysis script."""
    import nltk

    for resource in ("stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"):
        nltk.download(resource, quiet=True)


def tokenize_without_stopwords(text: str) -> list[str]:
    """Tokenize text and remove English stopwords."""
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    stop_words = set(stopwords.words("english"))
    return [token for token in word_tokenize(text) if token not in stop_words]


def lemmatize_tokens(tokens: Iterable[str]) -> list[str]:
    """Lemmatize a sequence of tokens with WordNet."""
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]
