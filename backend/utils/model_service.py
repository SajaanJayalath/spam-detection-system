from __future__ import annotations
from typing import Optional, Any


class ModelService:
    def __init__(self) -> None:
        self.pipeline: Optional[Any] = None
        self.model: Optional[Any] = None
        self.vectorizer: Optional[Any] = None
        self.classes_: Optional[Any] = None
        self.source: Optional[str] = None  # human-readable path/label of loaded model

    def loaded(self) -> bool:
        return self.pipeline is not None or (self.model is not None and self.vectorizer is not None)


MODEL_SERVICE = ModelService()
