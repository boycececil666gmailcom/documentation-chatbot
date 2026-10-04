# region Imports
from .state import AgentState

# endregion


# region Edges
def route_from_router(state: AgentState) -> str:
    """Routes state from router node to dedicated processing branches ('refuse', 'bm25', 'hyde_bm25', 'hyde')."""
    decision = state.get("routing_decision", "general")
    if decision == "refuse":
        return "refuse"
    elif decision == "keyword":
        return "bm25"
    elif decision == "general":
        return "hyde_bm25"
    else:  # "vague"
        return "hyde"


def route_after_critique(state: AgentState) -> str:
    """Routes to 'approved' (END) or 'rejected' (loop back to router) based on critique evaluation."""
    is_critique_passed = state.get("is_critique_passed", True)
    retry_count = state.get("retry_count", 0)
    return "approved" if is_critique_passed or retry_count >= 3 else "rejected"


# endregion
