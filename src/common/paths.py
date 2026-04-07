"""Utility helpers for working with project filesystem paths."""
from __future__ import annotations

from pathlib import Path


# Repository root resolved dynamically so scripts work from any CWD
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = REPO_ROOT / "reports"
MODELS_DIR = REPO_ROOT / "models"


def spam_raw_path(filename: str) -> Path:
    """Return the path to a raw spam dataset."""
    return RAW_DATA_DIR / "spam" / filename

def spam_processed_path(filename: str) -> Path:
    """Return the path to a processed spam artefact."""
    return PROCESSED_DATA_DIR / "spam" / filename

def reports_spam_path(filename: str) -> Path:
    """Return the path for saving spam-related figures."""
    return REPORTS_DIR / "spam" / filename

def ensure_parent_dir(path: Path) -> None:
    """Create parent directories for ``path`` if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)

def malware_raw_path(filename: str) -> Path:
    """Return the path to a raw malware dataset."""
    return RAW_DATA_DIR / "malware" / filename

def malware_processed_path(filename: str) -> Path:
    """Return the path to a processed malware artefact."""
    return PROCESSED_DATA_DIR / "malware" / filename


def spam_model_path(filename: str) -> Path:
    """Return the path for saving spam models."""
    return MODELS_DIR / "spam" / filename

def malware_model_path(filename: str) -> Path:
    """Return the path for saving malware models."""
    return MODELS_DIR / "malware" / filename

def reports_malware_path(filename: str) -> Path:
    """Return the path for saving malware-related figures."""
    return REPORTS_DIR / "malware" / filename
