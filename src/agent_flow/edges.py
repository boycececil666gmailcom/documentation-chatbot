# region Imports
from .state import AgentState

# endregion


# region Edges
def route_by_category(state: AgentState) -> str:
    """Routes state based on classification should_answer ('pass' vs 'refuse')."""
    return state.get("should_answer", "refuse")


def route_by_hyde_decision(state: AgentState) -> str:
    """Routes to 'enable' (hyde_node) or 'skip' (retrieve_node)."""
    return "enable" if state.get("should_hyde", True) else "skip"


def route_after_critique(state: AgentState) -> str:
    """Routes to 'approved' (END) or 'rejected' (loop back) based on critique feedback."""
    feedback = state.get("critique_feedback")
    attempt_count = state.get("attempt_count", 0)
    return "approved" if feedback == "PASS" or attempt_count >= 3 else "rejected"


# endregion
