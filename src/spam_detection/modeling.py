"""Model training utilities for spam detection."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    silhouette_score,
    adjusted_rand_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.linear_model import ElasticNet

from ..common.paths import (
    ensure_parent_dir,
    reports_spam_path,
    spam_model_path,
    spam_processed_path,
)


@dataclass
class DatasetSplit:
    """Container for train/test splits."""

    X_train: pd.Series
    X_test: pd.Series
    y_train: pd.Series
    y_test: pd.Series


def load_merged_spam_dataset() -> pd.DataFrame:
    """Load the merged spam dataset used for downstream modelling."""
    dataset_path = spam_processed_path("spam_merged.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Merged spam dataset not found. Run scripts/process_spam.py first."
        )
    df = pd.read_csv(dataset_path)
    required_cols = {"clean_text", "label"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    return df


def make_train_test_split(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    """Create a stratified train/test split for spam modelling."""
    X = df["clean_text"].astype(str)
    y = df["label"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return DatasetSplit(X_train, X_test, y_train, y_test)


def _save_metrics(metrics: Dict[str, Any], filename: str) -> None:
    path = reports_spam_path(filename)
    ensure_parent_dir(path)
    path.write_text(json.dumps(metrics, indent=2))


def _persist_model(pipeline: Pipeline, filename: str) -> None:
    path = spam_model_path(filename)
    ensure_parent_dir(path)
    joblib.dump(pipeline, path)


def train_multinomial_nb(
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    min_df: int = 2,
    max_features: int | None = 50000,
    ngram_range: Tuple[int, int] = (1, 2),
    alpha: float = 0.1,
) -> Dict[str, Any]:
    """Train and evaluate a Multinomial Naive Bayes spam classifier."""
    df = load_merged_spam_dataset()
    split = make_train_test_split(df, test_size=test_size, random_state=random_state)

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    min_df=min_df,
                    max_features=max_features,
                    ngram_range=ngram_range,
                    lowercase=True,
                ),
            ),
            ("clf", MultinomialNB(alpha=alpha)),
        ]
    )

    pipeline.fit(split.X_train, split.y_train)
    y_pred = pipeline.predict(split.X_test)
    y_proba = pipeline.predict_proba(split.X_test)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        split.y_test,
        y_pred,
        average="binary",
        zero_division=0,
    )

    metrics: Dict[str, Any] = {
        "model": "multinomial_nb",
        "vectorizer": {
            "min_df": min_df,
            "max_features": max_features,
            "ngram_range": list(ngram_range),
        },
        "alpha": alpha,
        "test_size": test_size,
        "random_state": random_state,
        "n_train": int(split.y_train.shape[0]),
        "n_test": int(split.y_test.shape[0]),
        "n_features": int(len(pipeline.named_steps["tfidf"].vocabulary_)),
        "accuracy": accuracy_score(split.y_test, y_pred),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(split.y_test, y_proba),
        "confusion_matrix": confusion_matrix(split.y_test, y_pred).tolist(),
        "classification_report": classification_report(
            split.y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }

    _save_metrics(metrics, "metrics/multinomial_nb_metrics.json")
    _persist_model(pipeline, "multinomial_nb.joblib")

    return metrics

def build_dense_feature_pipeline(
    *,
    min_df: int = 2,
    max_features: int | None = 60000,
    ngram_range: Tuple[int, int] = (1, 2),
    n_components: int = 300,
    random_state: int = 42,
) -> Pipeline:
    """Return a TF-IDF -> SVD -> scaling pipeline for dense features."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    min_df=min_df,
                    max_features=max_features,
                    ngram_range=ngram_range,
                    lowercase=True,
                ),
            ),
            (
                "svd",
                TruncatedSVD(n_components=n_components, random_state=random_state),
            ),
            ("scaler", StandardScaler()),
        ]
    )


def build_dense_features(
    *,
    min_df: int = 2,
    max_features: int | None = 60000,
    ngram_range: Tuple[int, int] = (1, 2),
    n_components: int = 300,
    random_state: int = 42,
) -> Tuple[Pipeline, np.ndarray, np.ndarray]:
    """Fit the dense feature pipeline on the merged dataset and return features."""
    df = load_merged_spam_dataset()
    pipeline = build_dense_feature_pipeline(
        min_df=min_df,
        max_features=max_features,
        ngram_range=ngram_range,
        n_components=n_components,
        random_state=random_state,
    )
    texts = df["clean_text"].astype(str)
    features = pipeline.fit_transform(texts)
    return pipeline, np.asarray(features), df["label"].astype(int).to_numpy()


def _cluster_assignment_accuracy(y_true: np.ndarray, clusters: np.ndarray) -> Tuple[Dict[int, int], float]:
    """Map clusters to labels using majority vote and compute accuracy."""
    contingency = pd.crosstab(clusters, y_true)
    mapping: Dict[int, int] = {}
    for cluster, row in contingency.iterrows():
        mapping[int(cluster)] = int(row.idxmax())
    mapped = np.vectorize(mapping.get, otypes=[int])(clusters)
    return mapping, accuracy_score(y_true, mapped)


def run_kmeans_clustering(
    *,
    n_clusters: int = 2,
    min_df: int = 2,
    max_features: int | None = 60000,
    ngram_range: Tuple[int, int] = (1, 2),
    n_components: int = 300,
    random_state: int = 42,
    n_init: int = 10,
) -> Dict[str, Any]:
    """Run KMeans clustering on dense spam features and evaluate alignment."""
    pipeline, features, labels = build_dense_features(
        min_df=min_df,
        max_features=max_features,
        ngram_range=ngram_range,
        n_components=n_components,
        random_state=random_state,
    )

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    cluster_labels = kmeans.fit_predict(features)

    try:
        silhouette = silhouette_score(features, cluster_labels)
    except ValueError:
        silhouette = float("nan")

    ari = adjusted_rand_score(labels, cluster_labels)
    mapping, mapped_accuracy = _cluster_assignment_accuracy(labels, cluster_labels)

    metrics: Dict[str, Any] = {
        "model": "kmeans",
        "n_clusters": n_clusters,
        "vectorizer": {
            "min_df": min_df,
            "max_features": max_features,
            "ngram_range": list(ngram_range),
        },
        "svd": {
            "n_components": n_components,
        },
        "random_state": random_state,
        "n_samples": int(labels.shape[0]),
        "silhouette": silhouette,
        "adjusted_rand_index": ari,
        "cluster_label_mapping": {str(k): v for k, v in mapping.items()},
        "mapped_accuracy": mapped_accuracy,
        "cluster_sizes": (
            pd.Series(cluster_labels)
            .value_counts()
            .sort_index()
            .to_dict()
        ),
    }

    _save_metrics(metrics, "metrics/kmeans_clustering_metrics.json")
    _persist_model(kmeans, "kmeans_clusters.joblib")

    assignments = pd.DataFrame(
        {
            "clean_text": load_merged_spam_dataset()["clean_text"],
            "label": labels,
            "cluster": cluster_labels,
        }
    )
    assignment_path = reports_spam_path("kmeans_assignments.csv")
    ensure_parent_dir(assignment_path)
    assignments.to_csv(assignment_path, index=False)

    return metrics

def train_elasticnet_regressor(
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    min_df: int = 2,
    max_features: int | None = 60000,
    ngram_range: Tuple[int, int] = (1, 2),
    n_components: int = 300,
    alpha: float = 0.001,
    l1_ratio: float = 0.15,
    max_iter: int = 2000,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Train an ElasticNet regressor and evaluate both regression and classification views."""
    df = load_merged_spam_dataset()
    split = make_train_test_split(df, test_size=test_size, random_state=random_state)

    feature_pipeline = build_dense_feature_pipeline(
        min_df=min_df,
        max_features=max_features,
        ngram_range=ngram_range,
        n_components=n_components,
        random_state=random_state,
    )

    full_pipeline = Pipeline(
        steps=feature_pipeline.steps
        + [
            (
                "reg",
                ElasticNet(
                    alpha=alpha,
                    l1_ratio=l1_ratio,
                    max_iter=max_iter,
                    random_state=random_state,
                ),
            )
        ]
    )

    full_pipeline.fit(split.X_train, split.y_train)
    y_pred_continuous = full_pipeline.predict(split.X_test)
    y_pred_binary = (y_pred_continuous >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        split.y_test,
        y_pred_binary,
        average="binary",
        zero_division=0,
    )

    mse = mean_squared_error(split.y_test, y_pred_continuous)
    rmse = float(mse ** 0.5)
    mae = mean_absolute_error(split.y_test, y_pred_continuous)
    r2 = r2_score(split.y_test, y_pred_continuous)

    metrics: Dict[str, Any] = {
        "model": "elasticnet_regressor",
        "vectorizer": {
            "min_df": min_df,
            "max_features": max_features,
            "ngram_range": list(ngram_range),
        },
        "svd": {"n_components": n_components},
        "elasticnet": {
            "alpha": alpha,
            "l1_ratio": l1_ratio,
            "max_iter": max_iter,
            "threshold": threshold,
        },
        "test_size": test_size,
        "random_state": random_state,
        "n_train": int(split.y_train.shape[0]),
        "n_test": int(split.y_test.shape[0]),
        "rmse": rmse,
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "roc_auc": roc_auc_score(split.y_test, y_pred_continuous),
        "classification_view": {
            "accuracy": accuracy_score(split.y_test, y_pred_binary),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": confusion_matrix(split.y_test, y_pred_binary).tolist(),
            "classification_report": classification_report(
                split.y_test,
                y_pred_binary,
                output_dict=True,
                zero_division=0,
            ),
        },
    }

    _save_metrics(metrics, "metrics/elasticnet_regression_metrics.json")
    _persist_model(full_pipeline, "elasticnet_regressor.joblib")

    return metrics

