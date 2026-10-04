# region Imports
from .state import AgentState

# endregion


# region Edges
def route_by_category(state: AgentState) -> str:
    """Routes state based on classification domain_route ('pass' vs 'refuse')."""
    return state.get("domain_route", "refuse")


def route_by_hyde_decision(state: AgentState) -> str:
    """Routes to 'enable' (hyde_node) or 'skip' (retrieve_node)."""
    return "enable" if state.get("use_hyde", True) else "skip"


def route_after_critique(state: AgentState) -> str:
    """Routes to 'approved' (END) or 'rejected' (loop back) based on critique evaluation."""
    is_critique_passed = state.get("is_critique_passed", True)
    retry_count = state.get("retry_count", 0)
    return "approved" if is_critique_passed or retry_count >= 3 else "rejected"


# endregion
