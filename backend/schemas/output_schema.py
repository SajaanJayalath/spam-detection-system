from __future__ import annotations

from datetime import datetime
from typing import Dict, Literal

from pydantic import BaseModel, Field


class SpamPredictionResponse(BaseModel):
    prediction: Literal["spam", "ham"]
    confidence: float = Field(ge=0.0, le=1.0)
    class_probs: Dict[str, float]
    timestamp: datetime
    id: str | None = None


class ErrorResponse(BaseModel):
    error: str
    message: str
