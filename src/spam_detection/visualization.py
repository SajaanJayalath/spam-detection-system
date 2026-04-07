"""Visualisation helpers for spam datasets."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

from ..common.paths import ensure_parent_dir

sns.set_style("whitegrid")

def _save_or_show(path: Path | None) -> None:
    if path is None:
        plt.show()
    else:
        ensure_parent_dir(path)
        plt.savefig(path, bbox_inches="tight")
    plt.close()

def plot_label_distribution(df: pd.DataFrame, label_col: str, title: str, path: Path | None = None) -> None:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=label_col, data=df, hue=label_col, legend=False, palette="Set2")
    plt.title(title)
    plt.xlabel("Label (0 = Ham, 1 = Spam)")
    plt.ylabel("Count")
    _save_or_show(path)

def plot_length_distribution(df: pd.DataFrame, text_col: str, label_col: str, title: str, path: Path | None = None) -> None:
    temp = df.copy()
    temp["text_length"] = temp[text_col].astype(str).apply(len)
    plt.figure(figsize=(8, 5))
    sns.histplot(data=temp, x="text_length", hue=label_col, bins=50, kde=True, palette="Set2")
    plt.title(title)
    plt.xlabel("Email Length (characters)")
    plt.ylabel("Frequency")
    _save_or_show(path)

def plot_top_words(texts: Iterable[str], title: str, palette: str, path: Path | None = None, top_n: int = 20) -> None:
    tokens = " ".join(texts).split()
    frequencies = Counter(tokens).most_common(top_n)
    if not frequencies:
        return
    words, counts = zip(*frequencies)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(counts), y=list(words), hue=list(words), dodge=False, palette=palette, legend=False)
    plt.title(title)
    plt.xlabel("Count")
    plt.ylabel("Word")
    _save_or_show(path)

def plot_wordcloud(texts: Iterable[str], title: str, path: Path | None = None, width: int = 600, height: int = 400) -> None:
    combined = " ".join(texts)
    if not combined.strip():
        return
    wc = WordCloud(width=width, height=height, background_color="black").generate(combined)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    _save_or_show(path)
