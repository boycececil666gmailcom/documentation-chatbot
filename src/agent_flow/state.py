# region State
from typing import Literal, NotRequired, TypedDict


class InputState(TypedDict):
    """User input payload schema required for workflow execution."""

    query: str
    history: NotRequired[list[dict]]


class AgentState(TypedDict):
    """Internal state schema passed across LangGraph nodes."""

    query: str
    history: NotRequired[list[dict]]
    should_answer: NotRequired[Literal["pass", "refuse"]]
    should_hyde: NotRequired[bool]
    hyde_reason: NotRequired[str | None]
    hyde_content: NotRequired[str | None]
    retrieved_documents: NotRequired[str | None]
    final_response: NotRequired[str]
    citations: NotRequired[list[str]]
    critique_feedback: NotRequired[str | None]
    attempt_count: NotRequired[int]


# endregion
