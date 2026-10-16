# region Imports
import re
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..config import CHATBOT_THEME
from ..llm_client import llm
from ..models import (
    CritiqueResultSchema,
    PlanSchema,
    RAGResponseSchema,
    SubagentResultSchema,
)
from .state import AgentState, SubagentResult
from .tools import (
    get_database_schema,
    query_database,
    search_documentation,
    search_web,
)

# endregion


# region Types
# Schemas imported from src.models and src.agent_flow.state
# endregion


# region Scaffolding
def _format_context_from_subagents(state: AgentState) -> str:
    """Formats findings from all active subagents into consolidated evidence context."""
    blocks: list[str] = []

    res1 = state.get("agent_local_files_result")
    if res1:
        blocks.append(
            f"=== EVIDENCE FROM {res1['agent_name']} ===\n"
            f"Sources: {', '.join(res1.get('sources', []))}\n"
            f"Findings:\n{res1['findings']}"
        )

    res2 = state.get("agent_database_result")
    if res2:
        blocks.append(
            f"=== EVIDENCE FROM {res2['agent_name']} ===\n"
            f"Sources: {', '.join(res2.get('sources', []))}\n"
            f"Findings:\n{res2['findings']}"
        )

    res3 = state.get("agent_web_search_result")
    if res3:
        blocks.append(
            f"=== EVIDENCE FROM {res3['agent_name']} ===\n"
            f"Sources: {', '.join(res3.get('sources', []))}\n"
            f"Findings:\n{res3['findings']}"
        )

    return "\n\n".join(blocks) if blocks else "No subagent findings collected."


def _extract_recent_history(messages: list[Any], max_turns: int = 4) -> str:
    """Formats the last N conversation turns into structured context."""
    if not messages:
        return "No previous conversation turns."
    recent = messages[-max_turns:]
    formatted_turns: list[str] = []
    for msg in recent:
        role = (
            "User"
            if isinstance(msg, HumanMessage)
            else ("Assistant" if isinstance(msg, AIMessage) else "System")
        )
        content_preview = str(msg.content)[:200].replace("\n", " ")
        formatted_turns.append(f"{role}: {content_preview}")
    return "\n".join(formatted_turns)


# endregion


# region Planning
def planning_node(state: AgentState) -> dict:
    """Analyzes user query and conversation context to generate a structured multi-step execution plan."""
    messages = state.get("messages") or []
    query = state.get("query") or (str(messages[-1].content) if messages else "")
    st_history = _extract_recent_history(messages)

    system_prompt = (
        f"You are the Lead Planning Agent for '{CHATBOT_THEME}'.\n"
        "Analyze the user inquiry and conversational context.\n"
        "Devise a concrete investigation plan delegating tasks to specialized subagents:\n"
        "- 'agent_local_files': In-depth inspection of local documentation files, APIs, and guides.\n"
        "- 'agent_database': Relational database inspection, metadata queries, and chunk relations.\n"
        "- 'agent_web_search': External search for live documentation, community practices, or external APIs.\n\n"
        f"Conversational Context:\n{st_history}"
    )

    try:
        structured_llm = llm.with_structured_output(PlanSchema)
        plan = cast(
            PlanSchema,
            structured_llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"Plan investigation for query: '{query}'"),
                ]
            ),
        )
        return {
            "query": query,
            "intent_analysis": plan.intent_analysis,
            "plan_steps": plan.steps,
            "subagents_required": plan.subagents_required,
        }
    except Exception as exc:
        print(
            f"[Planning-planning_node] Structured plan failed ({exc}), using default multi-agent plan..."
        )
        from ..models import PlanStep

        default_steps = [
            PlanStep(
                step_number=1,
                subagent="agent_local_files",
                instruction=f"Search local documentation for '{query}'",
            ),
            PlanStep(
                step_number=2,
                subagent="agent_web_search",
                instruction=f"Search web for latest '{query}' documentation",
            ),
        ]
        return {
            "query": query,
            "intent_analysis": f"Investigation for: {query}",
            "plan_steps": default_steps,
            "subagents_required": ["agent_local_files", "agent_web_search"],
        }


# endregion


# region Orchestrator
def orchestrator_node(state: AgentState) -> dict:
    """Coordinates execution across specialized subagents based on the active plan."""
    return {}


# endregion


# region Subagents
def agent_local_files_node(state: AgentState) -> dict:
    """Agent 1: Specializes in querying local documentation via standard search_documentation tool."""
    query = state["query"]
    plan_steps = state.get("plan_steps", [])

    instruction = query
    for step in plan_steps:
        if step.subagent == "agent_local_files":
            instruction = step.instruction
            break

    search_query = f"{instruction} {query}".strip()
    raw_evidence = search_documentation.invoke({"query": search_query})

    summary_prompt = (
        f"You are Agent 1 (Local Documentation Specialist) for '{CHATBOT_THEME}'.\n"
        f"Task Instruction: {instruction}\n\n"
        f"Raw Local Documentation Evidence:\n{raw_evidence}\n\n"
        "Synthesize key technical insights, API definitions, and code samples discovered."
    )

    try:
        structured_llm = llm.with_structured_output(SubagentResultSchema)
        res = cast(
            SubagentResultSchema,
            structured_llm.invoke([HumanMessage(content=summary_prompt)]),
        )
        findings = res.findings
        sources = res.sources or ["Local Documentation"]
    except Exception:
        findings = raw_evidence[:1500]
        sources = ["Local Documentation"]

    result_payload: SubagentResult = {
        "agent_name": "Agent 1 (Local Files & Documentation)",
        "findings": findings,
        "sources": sources,
    }
    return {"agent_local_files_result": result_payload}


def agent_database_node(state: AgentState) -> dict:
    """Agent 2: Specializes in querying PostgreSQL tables, schemas, and metadata via standard tools."""
    query = state["query"]
    plan_steps = state.get("plan_steps", [])

    instruction = query
    for step in plan_steps:
        if step.subagent == "agent_database":
            instruction = step.instruction
            break

    findings_list: list[str] = [f"Assigned database task: {instruction}"]
    sources: list[str] = ["PostgreSQL: langchain_pg_embedding"]

    try:
        schema_info = get_database_schema.invoke(
            {"table_name": "langchain_pg_embedding"}
        )
        findings_list.append(schema_info)

        raw_tokens = [
            t
            for t in re.findall(r"\w+", f"{instruction} {query}")
            if len(t) >= 2
            and t.lower()
            not in {"search", "kanzi", "database", "query", "what", "how", "the", "and"}
        ]
        search_term = raw_tokens[0] if raw_tokens else "node"
        clean_term = search_term.replace("'", "''")

        sql = (
            f"SELECT id, cmetadata->>'title' as title, cmetadata->>'breadcrumb' as breadcrumb, "
            f"substring(document from 1 for 350) as snippet "
            f"FROM langchain_pg_embedding "
            f"WHERE document ILIKE '%{clean_term}%' OR cmetadata->>'title' ILIKE '%{clean_term}%' "
            f"LIMIT 4;"
        )
        sql_rows = query_database.invoke({"sql_query": sql})
        findings_list.append(f"Records:\n{sql_rows}")
    except Exception as exc:
        findings_list.append(f"Database query note: {exc}")

    findings_text = (
        "\n\n".join(findings_list)
        if findings_list
        else "Database inspected with no relevant records."
    )
    result_payload: SubagentResult = {
        "agent_name": "Agent 2 (Relational Database)",
        "findings": findings_text,
        "sources": sources,
    }
    return {"agent_database_result": result_payload}


def agent_web_search_node(state: AgentState) -> dict:
    """Agent 3: Specializes in querying live web search APIs via standard search_web tool."""
    query = state["query"]
    plan_steps = state.get("plan_steps", [])

    instruction = query
    for step in plan_steps:
        if step.subagent == "agent_web_search":
            instruction = step.instruction
            break

    search_query = f"{CHATBOT_THEME} {instruction}"
    raw_web = search_web.invoke({"query": search_query})

    result_payload: SubagentResult = {
        "agent_name": "Agent 3 (Web Search & External APIs)",
        "findings": raw_web,
        "sources": ["Live Web Search (DDGS)"],
    }
    return {"agent_web_search_result": result_payload}


# endregion


# region Synthesis
def synthesize_node(state: AgentState) -> dict:
    """Synthesizes the grounded final answer using aggregated evidence from subagents."""
    query = state["query"]
    context = _format_context_from_subagents(state)
    feedback = state.get("critique_feedback")

    system_prompt = (
        f"You are the Primary Synthesis Orchestrator for '{CHATBOT_THEME}'.\n"
        "Synthesize an authoritative, highly technical response directly answering the user query.\n\n"
        "CRITICAL RULES:\n"
        "1. STRICT FIDELITY: Rely strictly on the aggregated evidence provided by the subagents.\n"
        "2. INLINE CITATIONS: Immediately attach an inline citation [Topic / Source Name] to every factual claim, API reference, or guideline.\n"
        "3. CITATIONS ARRAY: Populate the 'citations' array with the distinct topics/sources referenced inline.\n"
    )
    if feedback:
        system_prompt += (
            f"\n4. PREVIOUS CRITIQUE FEEDBACK (Address and rectify): {feedback}\n"
        )

    user_prompt = f"User Query: {query}\n\nAggregated Subagent Evidence:\n{context}"

    try:
        structured_llm = llm.with_structured_output(RAGResponseSchema)
        rag_res = cast(
            RAGResponseSchema,
            structured_llm.invoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_prompt),
                ]
            ),
        )
        answer = rag_res.answer
        citations = rag_res.citations
    except Exception as exc:
        print(f"[Synthesis-synthesize_node] Structured output fallback ({exc})")
        raw_res = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
        )
        answer = str(raw_res.content)
        citations = []

    return {
        "aggregated_context": context,
        "draft_response": answer,
        "citations": citations,
        "messages": [AIMessage(content=answer)],
    }


# endregion


# region Critique
def critique_node(state: AgentState) -> dict:
    """Evaluates groundedness, inline citation validity, and goal completeness."""
    draft = state.get("draft_response", "")
    query = state["query"]
    context = state.get("aggregated_context", "")
    retry_count = state.get("retry_count", 0)

    prompt = (
        f"You are a strict quality control auditor for '{CHATBOT_THEME}'.\n"
        f"User Query: {query}\n\n"
        f"Aggregated Subagent Context:\n{context[:3000]}\n\n"
        f"Draft Response:\n{draft}\n\n"
        "Evaluation Rules:\n"
        "1. Groundedness: Are claims supported by the aggregated subagent context?\n"
        "2. Citations: Are inline citations present and reflective of actual sources?\n\n"
        "Return valid JSON matching CritiqueResultSchema:\n"
        "- is_passed: true if accurate and grounded, false otherwise\n"
        "- feedback: explanation if false, otherwise null"
    )

    try:
        structured_llm = llm.with_structured_output(CritiqueResultSchema)
        eval_result = cast(
            CritiqueResultSchema,
            structured_llm.invoke([HumanMessage(content=prompt)]),
        )
        is_passed = eval_result.is_passed if eval_result else True
        feedback = eval_result.feedback if eval_result and not is_passed else None
    except Exception as exc:
        print(f"[Critique-critique_node] Warning during critique: {exc}")
        is_passed = True
        feedback = None

    print(
        f"[Critique-critique_node] System 2 critique: is_passed={is_passed}, feedback={feedback}"
    )

    return {
        "is_critique_passed": is_passed,
        "critique_feedback": feedback,
        "retry_count": retry_count if is_passed else retry_count + 1,
    }


# endregion
