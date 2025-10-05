"""Shared evaluation plotting utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from .paths import ensure_parent_dir


_DEF_CMAP = "Blues"


def save_confusion_matrix(
    estimator,
    X,
    y,
    *,
    title: str,
    path: Path,
    normalize: str | None = None,
    labels: Sequence[str] | None = None,
) -> None:
    """Render and save a confusion matrix."""
    ensure_parent_dir(path)
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ConfusionMatrixDisplay.from_estimator(
        estimator,
        X,
        y,
        display_labels=labels,
        cmap=_DEF_CMAP,
        normalize=normalize,
        ax=ax,
        colorbar=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_roc_curve(
    estimator,
    X,
    y,
    *,
    title: str,
    path: Path,
) -> None:
    """Render and save a ROC curve."""
    ensure_parent_dir(path)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    RocCurveDisplay.from_estimator(estimator, X, y, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_precision_recall_curve(
    estimator,
    X,
    y,
    *,
    title: str,
    path: Path,
) -> None:
    """Render and save a precision-recall curve."""
    ensure_parent_dir(path)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    PrecisionRecallDisplay.from_estimator(estimator, X, y, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def save_regression_diagnostics(
    estimator,
    X,
    y,
    *,
    prefix: Path,
    title: str,
) -> None:
    """Save predicted-vs-actual and residual plots for a regressor."""
    ensure_parent_dir(prefix)
    y_pred = estimator.predict(X)
    residuals = y - y_pred

    # Predicted vs Actual
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.scatter(y, y_pred, alpha=0.4, edgecolor="none")
    limits = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    ax.plot(limits, limits, linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{title} – Predicted vs Actual")
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_pred_vs_actual.png"), dpi=200)
    plt.close(fig)

    # Residual distribution
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.hist(residuals, bins=40, color="#4C72B0", alpha=0.8)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (actual − predicted)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"{title} – Residual Distribution")
    fig.tight_layout()
    fig.savefig(prefix.with_name(prefix.name + "_residuals.png"), dpi=200)
    plt.close(fig)


def save_cluster_size_bar(
    clusters: Iterable[int],
    *,
    title: str,
    path: Path,
) -> None:
    """Plot and save cluster counts."""
    ensure_parent_dir(path)
    counts = pd.Series(list(clusters)).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    counts.plot(kind="bar", color="#55A868", ax=ax)
    ax.set_xlabel("Cluster ID")
    ax.set_ylabel("Count")
    ax.set_title(title)
    for idx, value in enumerate(counts):
        ax.text(idx, value, f"{value:,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)




def save_metric_bar_chart(
    df: pd.DataFrame,
    *,
    domain: str,
    metric: str,
    path: Path,
    types: Iterable[str] | None = None,
) -> None:
    """Plot a bar chart comparing a metric across models for a domain."""
    ensure_parent_dir(path)
    subset = df[df['domain'] == domain].copy()
    if types is not None:
        subset = subset[subset['type'].isin(types)]
    subset = subset[subset[metric].notna()].sort_values(metric, ascending=False)
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.barh(subset['model'], subset[metric], color="#C44E52")
    ax.invert_yaxis()
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_title(f"{domain.title()} Models - {metric.replace('_', ' ').title()}")
    for idx, value in enumerate(subset[metric]):
        ax.text(value, idx, f"{value:.3f}", va='center', ha='left', fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

