# region Imports
from .state import AgentState

# endregion


# region Edges
def route_to_subagents(state: AgentState) -> list[str]:
    """Routes execution from planning/orchestrator to one or more specialized subagents concurrently."""
    required = state.get("subagents_required") or []
    valid_subagents = {"agent_local_files", "agent_database", "agent_web_search"}
    selected = [agent for agent in required if agent in valid_subagents]

    # Resilient fallback: engage local files and web search if none specified
    if not selected:
        selected = ["agent_local_files", "agent_web_search"]

    return selected


def route_after_critique(state: AgentState) -> str:
    """Routes to 'approved' (END) or 'rejected' (loop back to orchestrator) based on critique evaluation."""
    is_critique_passed = state.get("is_critique_passed", True)
    retry_count = state.get("retry_count", 0)
    return "approved" if is_critique_passed or retry_count >= 2 else "rejected"


# endregion
