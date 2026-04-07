from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Any, Dict
import warnings

import joblib
from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

from backend.utils.model_service import MODEL_SERVICE
from backend.utils.error_handler import validation_exception_handler, model_not_loaded_handler, ModelNotLoadedError
from backend.utils.logger import configure_logging
from backend.utils.middleware import RequestContextMiddleware
from backend.routers.spam_routes import router as spam_router
from backend.routers.feedback_routes import router as feedback_router

# Load environment variables from a .env file if present
load_dotenv()  # looks for .env in CWD or parents

# Logging setup with rotation
LOG_DIR = Path(__file__).resolve().parent / "logs"
logger = configure_logging(LOG_DIR)

app = FastAPI(title="Spam Detection API", version="1.0.0")

# CORS for React
origins = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request logging middleware
app.add_middleware(RequestContextMiddleware)

# Exception handlers
from fastapi.exceptions import RequestValidationError  # noqa: E402
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(ModelNotLoadedError, model_not_loaded_handler)

# Include routers
app.include_router(spam_router)
app.include_router(feedback_router)


def _try_load(path: Path) -> Optional[Any]:
    try:
        if path.exists():
            return joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed loading artifact %s: %s", path, exc)
    return None


@app.on_event("startup")
async def load_models() -> None:
    base = Path(__file__).resolve().parent

    # Preferred locations per spec
    model_path = base / "models" / "spam_model.pkl"
    vect_path = base / "models" / "spam_vectorizer.pkl"
    backend_models_dir = base / "models"

    # Fallbacks to existing project artifacts (pipelines)
    project_root = base.parent
    preferred_pipeline_env = os.getenv("SPAM_PIPELINE_PATH", "").strip()
    pipeline_logreg = project_root / "models" / "spam" / "logreg_tfidf.joblib"
    pipeline_nb = project_root / "models" / "spam" / "multinomial_nb.joblib"

    # Optionally silence sklearn version mismatch warnings
    try:
        from sklearn.exceptions import InconsistentVersionWarning  # type: ignore
        warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
    except Exception:
        pass

    model = _try_load(model_path)
    vectorizer = _try_load(vect_path)

    if model is not None and vectorizer is not None:
        MODEL_SERVICE.model = model
        MODEL_SERVICE.vectorizer = vectorizer
        MODEL_SERVICE.classes_ = getattr(model, "classes_", None)
        try:
            MODEL_SERVICE.source = f"{model_path.name} + {vect_path.name} in backend/models"
        except Exception:
            MODEL_SERVICE.source = str(model_path)
        logger.info("Loaded standalone model + vectorizer from backend/models")
        return

    # 1) If there is a packaged pipeline inside backend/models, prefer that
    packaged_candidates = list(backend_models_dir.glob("*.joblib"))
    if packaged_candidates:
        pipeline = _try_load(packaged_candidates[0])
        if pipeline is not None:
            MODEL_SERVICE.pipeline = pipeline
            MODEL_SERVICE.classes_ = getattr(pipeline, "classes_", None)
            MODEL_SERVICE.source = str(packaged_candidates[0])
            logger.info("Loaded pipeline packaged under backend/models: %s", packaged_candidates[0])
            return

    # 2) Allow explicit override via env
    if preferred_pipeline_env:
        pipeline = _try_load(Path(preferred_pipeline_env))
        if pipeline is not None:
            MODEL_SERVICE.pipeline = pipeline
            MODEL_SERVICE.classes_ = getattr(pipeline, "classes_", None)
            MODEL_SERVICE.source = preferred_pipeline_env
            logger.info("Loaded pipeline from SPAM_PIPELINE_PATH=%s", preferred_pipeline_env)
            return

    # 3) Prefer tuned Logistic Regression pipeline if available
    pipeline = _try_load(pipeline_logreg)
    if pipeline is not None:
        MODEL_SERVICE.pipeline = pipeline
        MODEL_SERVICE.classes_ = getattr(pipeline, "classes_", None)
        MODEL_SERVICE.source = str(pipeline_logreg)
        logger.info("Loaded pipeline from %s", pipeline_logreg)
        return

    # 4) Fallback to Multinomial NB pipeline
    pipeline = _try_load(pipeline_nb)
    if pipeline is not None:
        MODEL_SERVICE.pipeline = pipeline
        MODEL_SERVICE.classes_ = getattr(pipeline, "classes_", None)
        MODEL_SERVICE.source = str(pipeline_nb)
        logger.info("Loaded pipeline fallback from %s", pipeline_nb)
        return

    logger.error(
        "No model artifacts found. Place spam_model.pkl(+vectorizer) or a pipeline at %s or %s.",
        pipeline_logreg,
        pipeline_nb,
    )


# Health endpoint
@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok" if MODEL_SERVICE.loaded() else "model_not_loaded",
        "has_pipeline": MODEL_SERVICE.pipeline is not None,
        "has_model": MODEL_SERVICE.model is not None,
        "has_vectorizer": MODEL_SERVICE.vectorizer is not None,
        "model_source": MODEL_SERVICE.source,
    }


@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Spam Detection API. See /docs for usage."}
