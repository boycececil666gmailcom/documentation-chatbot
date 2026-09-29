# region Critique Edge
from ..state import AgentState


def route_after_critique(state: AgentState) -> str:
    """Routes to 'approved' (END) or 'rejected' (loop back) based on critique feedback."""
    feedback = state.get("critique_feedback")
    attempt_count = state.get("attempt_count", 0)
    return "approved" if feedback == "PASS" or attempt_count >= 3 else "rejected"


# endregion
