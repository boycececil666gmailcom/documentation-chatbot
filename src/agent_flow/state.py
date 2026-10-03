# region State
from typing import Literal, NotRequired, TypedDict


class InputState(TypedDict):
    """User input payload schema required for workflow execution."""

    query: str


class AgentState(TypedDict):
    """Internal state schema passed across LangGraph nodes."""

    query: str
    should_answer: NotRequired[Literal["pass", "refuse"]]
    should_hyde: NotRequired[bool]
    hyde_content: NotRequired[str | None]
    retrieved_documents: NotRequired[str | None]
    final_response: NotRequired[str]
    citations: NotRequired[list[str]]
    critique_passed: NotRequired[bool]
    critique_feedback: NotRequired[str | None]
    attempt_count: NotRequired[int]


# endregion
