# region State
from typing import Literal, NotRequired, TypedDict

from langchain_core.documents import Document


class InputState(TypedDict):
    """User input payload schema required for workflow execution."""

    query: str


class AgentState(TypedDict):
    """Internal state schema passed across LangGraph nodes."""

    query: str
    routing_decision: NotRequired[
        Literal["refuse", "keyword", "general", "vague"]
    ]
    hypothetical_doc: NotRequired[str | None]
    bm25_query: NotRequired[str | None]
    search_query: NotRequired[str | None]
    retrieved_docs: NotRequired[list[Document]]
    ranked_docs: NotRequired[list[Document]]
    retrieved_context: NotRequired[str | None]
    draft_response: NotRequired[str]
    citations: NotRequired[list[str]]
    is_critique_passed: NotRequired[bool]
    critique_feedback: NotRequired[str | None]
    retry_count: NotRequired[int]


# endregion
