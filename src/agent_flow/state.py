# region Imports
from typing import Literal, NotRequired, TypedDict

from langchain_core.documents import Document

# endregion


# region State Definitions
class InputState(TypedDict):
    """User input payload schema required for workflow execution."""

    query: str


class AgentState(TypedDict):
    """Internal state schema passed across LangGraph nodes."""

    query: str
    routing_decision: NotRequired[Literal["refuse", "bm25", "hyde_bm25", "hyde"]]
    hypothetical_doc: NotRequired[str | None]
    bm25_query: NotRequired[str | None]
    search_query: NotRequired[str | None]
    bm25_docs: NotRequired[list[Document]]
    hyde_docs: NotRequired[list[Document]]
    expanded_docs: NotRequired[list[Document]]
    ranked_docs: NotRequired[list[Document]]
    retrieved_context: NotRequired[str | None]
    draft_response: NotRequired[str]
    citations: NotRequired[list[str]]
    is_critique_passed: NotRequired[bool]
    critique_feedback: NotRequired[str | None]
    retry_count: NotRequired[int]


# endregion
