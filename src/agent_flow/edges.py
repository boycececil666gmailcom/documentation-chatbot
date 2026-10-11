# region Imports
from .state import AgentState

# endregion


# region Edges
def route_from_router(state: AgentState) -> str | list[str]:
    """Routes state from router: directly to 'refuse', 'local_slm_keyword_extraction', 'hyde', or concurrent ['local_slm_keyword_extraction', 'hyde']."""
    decision = state.get("routing_decision", "hyde_bm25")
    if decision == "refuse":
        return "refuse"
    if decision == "bm25":
        return "local_slm_keyword_extraction"
    if decision == "hyde":
        return "hyde"
    # Concurrent Fan-out: triggers both local_slm_keyword_extraction and hyde in the same Pregel Superstep
    return ["local_slm_keyword_extraction", "hyde"]


def route_after_critique(state: AgentState) -> str:
    """Routes to 'approved' (END) or 'rejected' (loop back to router) based on critique evaluation."""
    is_critique_passed = state.get("is_critique_passed", True)
    retry_count = state.get("retry_count", 0)
    return "approved" if is_critique_passed or retry_count >= 3 else "rejected"


# endregion
