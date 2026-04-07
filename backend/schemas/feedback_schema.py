from __future__ import annotations

from pydantic import BaseModel, Field, EmailStr


class FeedbackInput(BaseModel):
    name: str = Field(..., min_length=2, max_length=120, description="Your full name")
    email: EmailStr = Field(..., description="Your email address")
    subject: str = Field(..., min_length=2, max_length=200, description="Subject of feedback")
    message: str = Field(..., min_length=5, max_length=5000, description="Feedback message body")


class FeedbackResponse(BaseModel):
    status: str
    delivered: bool
    message: str

