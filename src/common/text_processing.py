"""Common text-processing helpers reused across datasets."""
from __future__ import annotations

import re
from typing import Iterable, Set

import nltk
from nltk.corpus import stopwords

def load_english_stopwords(additional: Iterable[str] | None = None) -> Set[str]:
    """Return a reusable stopword set, downloading resources if required."""
    try:
        base = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        base = set(stopwords.words("english"))

    if additional:
        base.update(additional)
    return base

def remove_subject_prefix(text: str) -> str:
    """Remove common email subject prefixes."""
    return re.sub(r"^(subject|re|fw)\s*:\s*", "", text, flags=re.IGNORECASE)

def normalise_whitespace(text: str) -> str:
    """Collapse repeated whitespace and strip edges."""
    return re.sub(r"\s+", " ", text).strip()

def keep_alpha_characters(text: str) -> str:
    """Strip everything except lowercase alphabetic characters and spaces."""
    return re.sub(r"[^a-z\s]", " ", text)

def remove_digits_and_punct(text: str) -> str:
    """Remove digits and punctuation, retaining word separators."""
    text = re.sub(r"\d+", " ", text)
    return re.sub(r"[^\w\s]", " ", text)

def drop_tokens(text: str, tokens: Iterable[str]) -> str:
    """Remove unwanted tokens from whitespace-separated text."""
    unwanted = {token.lower() for token in tokens}
    filtered = [word for word in text.split() if word.lower() not in unwanted]
    return " ".join(filtered)


