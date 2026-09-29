# region Retrieval Node
from ...tools import retrieve_VDB
from ..state import AgentState


def retrieve_node(state: AgentState) -> dict:
    """Retrieves document context from vector database using HyDE passage or user query."""
    query = state["query"]
    hypo_doc = state.get("hyde_content")

    retrieved_documents = state.get("retrieved_documents")
    if not retrieved_documents:
        search_target = hypo_doc if hypo_doc else query
        retrieved_documents = retrieve_VDB.invoke(search_target)

    return {"retrieved_documents": retrieved_documents}


# endregion
