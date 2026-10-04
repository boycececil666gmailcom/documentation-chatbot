# region State
from typing import Literal, NotRequired, TypedDict


class InputState(TypedDict):
    """User input payload schema required for workflow execution."""

    query: str


class AgentState(TypedDict):
    """Internal state schema passed across LangGraph nodes."""

    query: str
    domain_route: NotRequired[Literal["pass", "refuse"]]
    use_hyde: NotRequired[bool]
    hypothetical_doc: NotRequired[str | None]
    retrieved_context: NotRequired[str | None]
    draft_response: NotRequired[str]
    citations: NotRequired[list[str]]
    is_critique_passed: NotRequired[bool]
    critique_feedback: NotRequired[str | None]
    retry_count: NotRequired[int]


# endregion
