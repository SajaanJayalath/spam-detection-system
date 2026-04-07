from __future__ import annotations

import logging
from fastapi import APIRouter

from backend.schemas.feedback_schema import FeedbackInput, FeedbackResponse
from backend.utils.emailer import send_feedback_email

logger = logging.getLogger("spam_app")

router = APIRouter(prefix="/feedback", tags=["feedback"])


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(payload: FeedbackInput) -> FeedbackResponse:  # noqa: D401
    """Accept user feedback and forward via email (or file fallback)."""
    delivered = send_feedback_email(
        name=payload.name,
        email=payload.email,
        subject=payload.subject,
        message=payload.message,
    )
    status = "accepted"
    msg = "Feedback delivered via email." if delivered else "Feedback recorded; email not configured."
    return FeedbackResponse(status=status, delivered=delivered, message=msg)

