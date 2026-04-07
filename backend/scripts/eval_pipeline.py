from __future__ import annotations

"""
Quick evaluator for a scikit-learn Pipeline/estimator saved via joblib.

Usage:
  python -m backend.scripts.eval_pipeline --model path/to/model.joblib --data path/to/test.csv

CSV requirements:
  - columns: text,label
  - label values should be 'spam' or 'ham' (case-insensitive) or 1/0

Outputs accuracy, precision, recall, F1, and ROC-AUC (if possible).
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score


def normalize_label(v) -> int:
    s = str(v).strip().lower()
    if s in {"1", "spam", "true", "yes"}:
        return 1
    return 0


def load_data(path: Path, text_col: str, label_col: str) -> Tuple[list[str], np.ndarray]:
    df = pd.read_csv(path)
    cols = set(df.columns.str.lower())
    tcol = text_col
    lcol = label_col

    # Allow some common aliases if exact names not present
    if tcol not in df.columns:
        for cand in [text_col, "text", "message", "clean_text", "content", "body"]:
            if cand in df.columns:
                tcol = cand
                break
            # case-insensitive match
            if cand.lower() in cols:
                tcol = next(c for c in df.columns if c.lower() == cand.lower())
                break

    if lcol not in df.columns:
        for cand in [label_col, "label", "target", "y", "class"]:
            if cand in df.columns:
                lcol = cand
                break
            if cand.lower() in cols:
                lcol = next(c for c in df.columns if c.lower() == cand.lower())
                break

    if tcol not in df.columns or lcol not in df.columns:
        raise SystemExit(
            f"CSV must contain columns matching text='{text_col}' and label='{label_col}' (aliases supported). "
            f"Found: {list(df.columns)}"
        )

    X = df[tcol].astype(str).tolist()
    y = df[lcol].map(normalize_label).to_numpy()
    return X, y


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path, help="Path to joblib pipeline/model")
    parser.add_argument("--data", required=True, type=Path, help="CSV with text/label columns")
    parser.add_argument("--text-col", default="text", help="Name of the text column (default: text)")
    parser.add_argument("--label-col", default="label", help="Name of the label column (default: label)")
    args = parser.parse_args()

    model = joblib.load(args.model)
    X, y_true = load_data(args.data, args.text_col, args.label_col)

    y_pred = model.predict(X)

    # Try to compute proba for ROC-AUC
    y_score = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 2:
            y_score = proba[:, 1]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    if y_score is not None:
        try:
            auc = roc_auc_score(y_true, y_score)
            print(f"ROC-AUC:   {auc:.4f}")
        except Exception:
            pass

    print("\nClassification report:\n")
    print(classification_report(y_true, y_pred, target_names=["ham", "spam"]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
