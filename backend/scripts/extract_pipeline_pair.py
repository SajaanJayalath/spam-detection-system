from __future__ import annotations

"""
Extract a standalone model + vectorizer from a scikit-learn Pipeline (.joblib)
and save them under backend/models so the backend reports has_model=true and
has_vectorizer=true.

Usage:
  python -m backend.scripts.extract_pipeline_pair --src path/to/pipeline.joblib \
      [--model-name spam_model.pkl] [--vect-name spam_vectorizer.pkl]
"""

import argparse
from pathlib import Path
import joblib


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, type=Path, help="Path to a joblib pipeline")
    p.add_argument("--model-name", dest="model_name", default="spam_model.pkl", help="Output model filename")
    p.add_argument("--vect-name", dest="vect_name", default="spam_vectorizer.pkl", help="Output vectorizer filename")
    args = p.parse_args()

    pipe = joblib.load(args.src)

    # Try common step names
    vectorizer = None
    model = None

    # Pipeline may expose named_steps
    named = getattr(pipe, "named_steps", None)
    if isinstance(named, dict):
        for key, val in named.items():
            cls = val.__class__.__name__.lower()
            if vectorizer is None and any(k in cls for k in ["tfidf", "countvectorizer", "vectorizer"]):
                vectorizer = val
            elif model is None and any(k in cls for k in ["nb", "bayes", "logistic", "svm", "forest", "regress", "classifier", "model"]):
                model = val

    # Fallback: assume last step is estimator, first is vectorizer
    if vectorizer is None:
        try:
            vectorizer = pipe.steps[0][1]
        except Exception:  # noqa: BLE001
            pass
    if model is None:
        try:
            model = pipe.steps[-1][1]
        except Exception:  # noqa: BLE001
            pass

    if vectorizer is None or model is None:
        raise SystemExit("Could not infer vectorizer and model from the pipeline. Provide a compatible pipeline.")

    out_dir = Path(__file__).resolve().parents[1] / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / args.model_name)
    joblib.dump(vectorizer, out_dir / args.vect_name)
    print(f"Saved model to {out_dir / args.model_name}")
    print(f"Saved vectorizer to {out_dir / args.vect_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
