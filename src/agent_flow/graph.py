# region Imports
import asyncio

from langgraph.graph import END, StateGraph

from .edges import route_after_critique, route_to_subagents
from .nodes import (
    agent_database_node,
    agent_local_files_node,
    agent_web_search_node,
    critique_node,
    orchestrator_node,
    planning_node,
    synthesize_node,
)
from .state import AgentState, InputState

# endregion


# region Graph
workflow = StateGraph(AgentState, input_schema=InputState)

# Planning & Orchestration
workflow.add_node("planning", planning_node)
workflow.add_node("orchestrator", orchestrator_node)

# Specialized Worker Subagents (Agent 1, Agent 2, Agent 3)
workflow.add_node("agent_local_files", agent_local_files_node)
workflow.add_node("agent_database", agent_database_node)
workflow.add_node("agent_web_search", agent_web_search_node)

# Synthesis & System 2 Quality Verification
workflow.add_node("synthesize", synthesize_node)
workflow.add_node("critique", critique_node)

# Orchestration Transitions: Entry Point is direct Planning
workflow.set_entry_point("planning")
workflow.add_edge("planning", "orchestrator")

# Dynamic Fan-Out from Orchestrator to Specialized Subagents
workflow.add_conditional_edges(
    "orchestrator",
    route_to_subagents,
    ["agent_local_files", "agent_database", "agent_web_search"],
)

# Fan-In: All active subagents converge into the Synthesis Node
workflow.add_edge("agent_local_files", "synthesize")
workflow.add_edge("agent_database", "synthesize")
workflow.add_edge("agent_web_search", "synthesize")

# Synthesis flows into Critique
workflow.add_edge("synthesize", "critique")

# Reflection Loop: Approved -> Complete, Rejected -> Loop back to Orchestrator
workflow.add_conditional_edges(
    "critique",
    route_after_critique,
    {"approved": END, "rejected": "orchestrator"},
)

agent_graph = workflow.compile(
    name="AgenticRAGOrchestrator",
)
# endregion


# region Main
async def _run_cli_test() -> None:
    import sys

    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    test_query = "What is Kanzi UI and how are 3D nodes structured?"
    config = {"configurable": {"thread_id": "session_test_001"}}
    input_payload: InputState = {
        "query": test_query,
    }
    print(
        f"\n[AgenticRAG-Main] Initiating test run with query: '{test_query}'",
        flush=True,
    )
    result = await agent_graph.ainvoke(input_payload, config=config)

    print("\n[AgenticRAG-Main] === INTENT ANALYSIS ===", flush=True)
    print(result.get("intent_analysis"), flush=True)

    print("\n[AgenticRAG-Main] === SUBAGENTS ENGAGED ===", flush=True)
    print(result.get("subagents_required"), flush=True)

    print("\n[AgenticRAG-Main] === SYNTHESIZED RESPONSE ===", flush=True)
    print(result.get("draft_response"), flush=True)

    print("\n[AgenticRAG-Main] === CITATIONS ===", flush=True)
    print(result.get("citations"), flush=True)


if __name__ == "__main__":
    asyncio.run(_run_cli_test())
# endregion
