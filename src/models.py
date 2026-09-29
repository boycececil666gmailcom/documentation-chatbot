# region Agent Schemas
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


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


class HyDEDecisionSchema(BaseModel):
    """Deterministic output schema for HyDE decision."""

    should_hyde: bool = Field(
        description="True if query is abstract/non-technical and benefits from HyDE"
    )
    reason: str = Field(description="Reason for HyDE decision")


# endregion


# region API Schemas
class MessageSchema(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str = Field(min_length=1, description="Message content cannot be empty")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, description="Query string cannot be empty")
    history: list[MessageSchema] = Field(
        default_factory=list, description="Chat history messages"
    )


class QueryResponse(BaseModel):
    response: str
    citations: list[str] = Field(default_factory=list)
    tool_calls_executed: list[str] = Field(default_factory=list)
    should_hyde: bool | None = None
    hyde_reason: str | None = None
    hyde_content: str | None = None
    retrieved_documents: str | None = None
    history: list[MessageSchema] | None = None


# endregion


# region Tool Schemas
class ToolQueryArgs(BaseModel):
    query: str

    @model_validator(mode="before")
    @classmethod
    def parse_args(cls, data: Any) -> dict[str, str]:
        if isinstance(data, dict):
            q_val = (
                data.get("query")
                or data.get("input")
                or (next(iter(data.values())) if data else "")
            )
            return {"query": str(q_val)}
        return {"query": str(data)}


# endregion
