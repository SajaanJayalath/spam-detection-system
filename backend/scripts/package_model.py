from __future__ import annotations

"""
Copy a chosen model/pipeline into backend/models for packaging with the backend.

Usage:
  python -m backend.scripts.package_model --src models/spam/logreg_tfidf.joblib [--name spam_pipeline.joblib]
"""

import argparse
import shutil
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path, help="Path to model pipeline (.joblib) to package")
    p.add_argument("--name", default="spam_pipeline.joblib", help="Destination filename inside backend/models/")
    args = p.parse_args()

    dest_dir = Path(__file__).resolve().parents[1] / "models"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / args.name
    shutil.copy2(args.src, dest)
    print(f"Packaged model to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

