# region HyDE Decision
import re

from ..state import AgentState


def hyde_decision_node(state: AgentState) -> dict:
    """Decides whether HyDE expansion is beneficial for the user query."""
    query = state["query"].strip()

    # Skip HyDE for technical codes, versions, or error patterns
    error_pattern = r"(error|err|code|uuid|v\d+\.\d+|\b[A-Z]{2,}-\d+\b|\b\d{3,5}\b)"
    if re.search(error_pattern, query, re.IGNORECASE):
        return {
            "should_hyde": False,
            "hyde_reason": "Query contains specific identifier or error pattern",
        }

    # Skip HyDE for detailed long queries
    if len(query) > 80 or len(query.split()) >= 12:
        return {
            "should_hyde": False,
            "hyde_reason": "Query is already specific and detailed",
        }

    # Enable HyDE for short or abstract queries
    return {
        "should_hyde": True,
        "hyde_reason": "Abstract or short query benefits from hypothetical expansion",
    }


# endregion
