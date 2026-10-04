# region Imports
from langgraph.graph import END, StateGraph

from .edges import route_after_critique, route_from_router
from .nodes import (
    bm25_node,
    critique_node,
    generate_node,
    hyde_node,
    refuse_node,
    rerank_node,
    retrieve_node,
    router_node,
)
from .state import AgentState, InputState

# endregion


# region Graph Definition
workflow = StateGraph(AgentState, input=InputState)

# Add Nodes
workflow.add_node("router", router_node)
workflow.add_node("bm25", bm25_node)
workflow.add_node("hyde", hyde_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("rerank", rerank_node)
workflow.add_node("generate", generate_node)
workflow.add_node("refuse", refuse_node)
workflow.add_node("critique", critique_node)

# Set Entry Point and Conditional Transitions
workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    route_from_router,
    ["refuse", "bm25", "hyde"],
)

workflow.add_edge("bm25", "retrieve")
workflow.add_edge("hyde", "retrieve")
workflow.add_edge("retrieve", "rerank")
workflow.add_edge("rerank", "generate")
workflow.add_edge("generate", "critique")
workflow.add_edge("refuse", "critique")

workflow.add_conditional_edges(
    "critique",
    route_after_critique,
    {"approved": END, "rejected": "router"},
)

agent_graph = workflow.compile(name="KanziDocumentationAgent")
# endregion


# region Direct Runner
if __name__ == "__main__":
    import asyncio

    test_input = {
        "query": "What is Kanzi fundamentals?",
    }
    print("[Agent-Graph] Executing workflow directly...")
    result = asyncio.run(agent_graph.ainvoke(test_input))
    print("\n[Agent-Graph] === RESULT ===")
    print(f"Response:\n{result.get('draft_response', '')}")
    print(f"Citations: {result.get('citations', [])}")
# endregion
