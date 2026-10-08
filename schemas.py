
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


SentimentLabel = Literal["positive", "negative", "neutral"]


def validate_english_text(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Text must be a string.")

    value = value.strip()

    if not value:
        raise ValueError("Text cannot be empty.")

    words = value.split()

    if len(words) > 500:
        raise ValueError("Text cannot exceed 500 words.")

    # Basic input screening, not a full language detector.
    if not re.search(r"[A-Za-z]", value):
        raise ValueError("Please provide English text.")

    return value


class AnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="English text containing 1 to 500 words."
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        return validate_english_text(value)


class BatchRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="Provide between 1 and 10 texts."
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, values: list[str]) -> list[str]:
        return [validate_english_text(value) for value in values]


class SentimentResult(BaseModel):
    text: str
    sentiment: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0)


class BatchResponse(BaseModel):
    results: list[SentimentResult]


class HealthResponse(BaseModel):
    status: str
    model: str
    model_loaded: bool