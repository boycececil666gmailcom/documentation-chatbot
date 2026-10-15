# region Imports
from typing import Annotated, Any, NotRequired, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from ..models import PlanStep

# endregion


# region Types
class InputState(TypedDict):
    """User input payload schema required for workflow execution."""

    query: str
    user_id: NotRequired[str]


class SubagentResult(TypedDict):
    """Consolidated outcome from a specialized subagent execution."""

    agent_name: str
    findings: str
    sources: list[str]


class AgentState(TypedDict):
    """Internal state schema passed across LangGraph nodes in the Agentic RAG pipeline."""

    # Short-Term Memory: Conversational message buffer
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    user_id: NotRequired[str]

    # Long-Term Memory: Profile, past insights, and domain knowledge
    long_term_memory: NotRequired[dict[str, Any]]

    # Planning: Structured decomposition
    intent_analysis: NotRequired[str]
    plan_steps: NotRequired[list[PlanStep]]
    subagents_required: NotRequired[list[str]]
    current_step_index: NotRequired[int]

    # Subagent Execution Outputs
    agent_local_files_result: NotRequired[SubagentResult | None]
    agent_database_result: NotRequired[SubagentResult | None]
    agent_web_search_result: NotRequired[SubagentResult | None]

    # Aggregated Context & Synthesis
    aggregated_context: NotRequired[str]
    draft_response: NotRequired[str]
    citations: NotRequired[list[str]]

    # System 2 Critique & Groundedness Audit
    is_critique_passed: NotRequired[bool]
    critique_feedback: NotRequired[str | None]
    retry_count: NotRequired[int]


# endregion
