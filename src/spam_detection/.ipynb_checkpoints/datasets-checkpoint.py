"""Data loading and cleaning utilities for the spam datasets."""
from __future__ import annotations

from typing import Dict

import pandas as pd

from ..common.paths import ensure_parent_dir, spam_processed_path, spam_raw_path
from ..common.text_processing import (
    drop_tokens,
    keep_alpha_characters,
    load_english_stopwords,
    normalise_whitespace,
    remove_digits_and_punct,
    remove_subject_prefix,
)

# Reuse a single stopword set across datasets
STOPWORDS = load_english_stopwords()
UNWANTED_MERGED_TOKENS = {"-", "_", "e", "cc", "j", "subject", "vince", "hou", "enron", "ect", "com", "http", "www"}


def _preprocess_dataset1(text: str) -> str:
    text = text.lower()
    text = remove_subject_prefix(text)
    text = text.replace("_", " ")
    text = remove_digits_and_punct(text)
    text = normalise_whitespace(text)
    tokens = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(tokens)

def _preprocess_dataset2(text: str) -> str:
    text = str(text).lower()
    text = remove_subject_prefix(text)
    text = text.replace("\n", " ")
    text = remove_digits_and_punct(text)
    text = normalise_whitespace(text)
    tokens = [word for word in text.split() if word not in STOPWORDS]
    return " ".join(tokens)

def _preprocess_dataset3(text: str) -> str:
    text = str(text).lower()
    text = remove_subject_prefix(text)
    text = keep_alpha_characters(text)
    text = normalise_whitespace(text)
    return text

def process_dataset1() -> pd.DataFrame:
    df = pd.read_csv(spam_raw_path("spam_ds1.csv"), usecols=[0, 1])
    df.columns = ["text", "label"]
    df = df.dropna().drop_duplicates()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[df["label"].isin({"0", "1"})]
    df["label"] = df["label"].astype(int)
    df["clean_text"] = df["text"].apply(_preprocess_dataset1)
    return df[["clean_text", "label", "text"]]

def process_dataset2() -> pd.DataFrame:
    df = pd.read_csv(spam_raw_path("spam_ds2.csv"))
    df.columns = ["extra1", "extra2", "text", "label"]
    df = df[["text", "label"]]
    df = df.dropna().drop_duplicates()
    df["label"] = df["label"].astype(int)
    df["clean_text"] = df["text"].apply(_preprocess_dataset2)
    return df[["clean_text", "label", "text"]]

def process_dataset3() -> pd.DataFrame:
    df = pd.read_csv(spam_raw_path("spam_ds3.csv"))
    df = df.rename(columns={"message": "text", "label": "label"})
    df["label"] = df["label"].map({"ham": 0, "spam": 1}).astype(int)
    df["clean_text"] = df["text"].apply(_preprocess_dataset3)
    df = df.dropna().drop_duplicates()
    return df[["clean_text", "label", "text"]]

def save_clean_dataset(df: pd.DataFrame, filename: str) -> None:
    path = spam_processed_path(filename)
    ensure_parent_dir(path)
    df.to_csv(path, index=False)

def merge_datasets(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged = pd.concat([df[["clean_text", "label"]] for df in datasets.values()], ignore_index=True)
    merged = merged.dropna().drop_duplicates()
    merged["clean_text"] = merged["clean_text"].astype(str)
    merged["clean_text"] = merged["clean_text"].str.lower()
    merged["clean_text"] = merged["clean_text"].apply(lambda text: drop_tokens(text, UNWANTED_MERGED_TOKENS))
    merged["clean_text"] = merged["clean_text"].apply(normalise_whitespace)
    return merged

def run_full_processing_pipeline() -> Dict[str, pd.DataFrame]:
    datasets = {
        "spam_ds1": process_dataset1(),
        "spam_ds2": process_dataset2(),
        "spam_ds3": process_dataset3(),
    }

    save_clean_dataset(datasets["spam_ds1"][ ["clean_text", "label"] ], "cleaned_spam_ds1.csv")
    save_clean_dataset(datasets["spam_ds2"][ ["clean_text", "label"] ], "cleaned_spam_ds2.csv")
    save_clean_dataset(datasets["spam_ds3"][ ["clean_text", "label"] ], "cleaned_spam_ds3.csv")

    merged = merge_datasets(datasets)
    save_clean_dataset(merged, "spam_merged.csv")
    datasets["spam_merged"] = merged
    return datasets


