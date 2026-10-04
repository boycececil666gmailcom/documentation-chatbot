# region Imports
from .state import AgentState

# endregion


# region Edges
def route_from_router(state: AgentState) -> str:
    """Routes state from router node directly to matching node ('refuse', 'bm25', 'hyde_bm25', 'hyde')."""
    return state.get("routing_decision", "hyde_bm25")


def route_after_critique(state: AgentState) -> str:
    """Routes to 'approved' (END) or 'rejected' (loop back to router) based on critique evaluation."""
    is_critique_passed = state.get("is_critique_passed", True)
    retry_count = state.get("retry_count", 0)
    return "approved" if is_critique_passed or retry_count >= 3 else "rejected"


# endregion
