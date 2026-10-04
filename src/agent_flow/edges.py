# region Imports
from .state import AgentState

# endregion


# region Edges
def route_from_router(state: AgentState) -> str | list[str]:
    """Routes state from router: directly to 'refuse', 'bm25', 'hyde', or concurrent ['bm25', 'hyde']."""
    decision = state.get("routing_decision", "hyde_bm25")
    if decision == "refuse":
        return "refuse"
    if decision == "bm25":
        return "bm25"
    if decision == "hyde":
        return "hyde"
    # Concurrent Fan-out: triggers both bm25 and hyde in the same Pregel Superstep
    return ["bm25", "hyde"]


def route_after_critique(state: AgentState) -> str:
    """Routes to 'approved' (END) or 'rejected' (loop back to router) based on critique evaluation."""
    is_critique_passed = state.get("is_critique_passed", True)
    retry_count = state.get("retry_count", 0)
    return "approved" if is_critique_passed or retry_count >= 3 else "rejected"


# endregion
