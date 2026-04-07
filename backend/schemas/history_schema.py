from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class HistoryItem(BaseModel):
    id: str
    text: str
    prediction: Literal["spam", "ham"]
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime


class HistoryUpdate(BaseModel):
    prediction: Optional[Literal["spam", "ham"]] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

