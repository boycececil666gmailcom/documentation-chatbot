# region Agent Schemas
from typing import Literal

from pydantic import BaseModel, Field


class RAGResponseSchema(BaseModel):
    """Deterministic output schema for RAG QA synthesis."""

    answer: str = Field(
        description="Strictly grounded answer text synthesized from retrieved document context"
    )
    citations: list[str] = Field(
        default_factory=list,
        description="List of referenced topic titles or breadcrumbs cited from retrieved context",
    )


class CritiqueResultSchema(BaseModel):
    """Deterministic output schema for critique node evaluation."""

    is_passed: bool = Field(
        description="True if response passes critique evaluation, False otherwise"
    )
    feedback: str | None = Field(
        default=None, description="Detailed explanation if is_passed is False"
    )


class HyDESchema(BaseModel):
    """Deterministic output schema for HyDE hypothetical document generation."""

    passage: str = Field(
        description="Short, plausible documentation excerpt answering user query"
    )


class ClassifierSchema(BaseModel):
    """Deterministic output schema for domain query classification."""

    category: Literal["pass", "refuse"] = Field(
        description="Category 'pass' if on-topic, 'refuse' if off-topic"
    )
    reason: str | None = Field(
        default=None, description="Reason for refusal if off-topic"
    )


# endregion
