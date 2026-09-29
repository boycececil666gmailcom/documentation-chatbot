# region Category Edge
from ..state import AgentState


def route_by_category(state: AgentState) -> str:
    """Routes state based on classification should_answer ('pass' vs 'refuse')."""
    return state.get("should_answer", "refuse")


# endregion
