from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

METRICS_DIRS = {
    "spam": Path("reports/spam/metrics"),
    "malware": Path("reports/malware/metrics"),
}

CLASSIFICATION_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
REGRESSION_KEYS = ["rmse", "mae", "r2", "roc_auc"]
CLUSTERING_KEYS = ["silhouette", "adjusted_rand_index", "mapped_accuracy"]


def _flat_metrics(domain: str, payload: Dict[str, Any], source: Path) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "domain": domain,
        "source_file": source.name,
        "dataset": payload.get("dataset", payload.get("model")),
        "model": payload.get("model", payload.get("model_name", source.stem)),
    }

    if all(key in payload for key in CLASSIFICATION_KEYS):
        record.update({key: payload.get(key) for key in CLASSIFICATION_KEYS})
        record["type"] = "classification"
    elif "classification_view" in payload:
        record.update({key: payload["classification_view"].get(key) for key in CLASSIFICATION_KEYS})
        record["rmse"] = payload.get("rmse")
        record["mae"] = payload.get("mae")
        record["r2"] = payload.get("r2")
        record["type"] = "regression"
    elif any(key in payload for key in CLUSTERING_KEYS):
        record.update({key: payload.get(key) for key in CLUSTERING_KEYS})
        record["type"] = "clustering"
    else:
        # fallback to flatten all numeric metrics
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                record[key] = value
        record["type"] = payload.get("type", "unknown")
    return record


def load_metrics() -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for domain, directory in METRICS_DIRS.items():
        if not directory.exists():
            continue
        for file in sorted(directory.glob("*.json")):
            try:
                payload = json.loads(file.read_text())
            except json.JSONDecodeError as exc:
                print(f"Skipping {file}: {exc}")
                continue
            record = _flat_metrics(domain, payload, file)
            records.append(record)
    return pd.DataFrame(records)


def main() -> None:
    df = load_metrics()
    if df.empty:
        print("No metrics found.")
        return

    output_path = Path("reports/metrics_summary.csv")
    df_sorted = df.sort_values(["domain", "type", "model"])
    df_sorted.to_csv(output_path, index=False)
    print("Aggregated metrics saved to", output_path)
    print()
    with pd.option_context("display.max_columns", None, "display.width", 120):
        print(df_sorted.fillna("").to_string(index=False))


if __name__ == "__main__":
    main()
