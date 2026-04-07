from pydantic import BaseModel, Field


class SpamInput(BaseModel):
    text: str = Field(
        ...,
        min_length=2,
        max_length=10000,
        description="Message text to classify (2-10000 characters)",
    )
