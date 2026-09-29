# region HyDE Edge
from ..state import AgentState


def route_by_hyde_decision(state: AgentState) -> str:
    """Routes to 'enable' (hyde_node) or 'skip' (retrieve_node)."""
    return "enable" if state.get("should_hyde", True) else "skip"


# endregion
