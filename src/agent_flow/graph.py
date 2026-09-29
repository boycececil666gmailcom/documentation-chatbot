# region Graph Definition
from langgraph.graph import END, StateGraph

from .edges import route_after_critique, route_by_category, route_by_hyde_decision
from .nodes import (
    classifier_node,
    critique_node,
    generate_node,
    hyde_decision_node,
    hyde_node,
    refuse_node,
    retrieve_node,
)
from .state import AgentState, InputState

# Initialize Workflow Graph
workflow = StateGraph(AgentState, input_schema=InputState)

# Add Nodes
workflow.add_node("classifier", classifier_node)
workflow.add_node("hyde_decision", hyde_decision_node)
workflow.add_node("hyde", hyde_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("refuse", refuse_node)
workflow.add_node("critique", critique_node)

# Set Entry Point and Conditional Transitions
workflow.set_entry_point("classifier")

workflow.add_conditional_edges(
    "classifier",
    route_by_category,
    {"pass": "hyde_decision", "refuse": "refuse"},
)

workflow.add_conditional_edges(
    "hyde_decision",
    route_by_hyde_decision,
    {"enable": "hyde", "skip": "retrieve"},
)

workflow.add_edge("hyde", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "critique")
workflow.add_edge("refuse", "critique")

workflow.add_conditional_edges(
    "critique",
    route_after_critique,
    {"approved": END, "rejected": "classifier"},
)

agent_graph = workflow.compile()
# endregion
