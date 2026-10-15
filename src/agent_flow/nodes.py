# region Imports
import json
import re
from typing import cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..config import CHATBOT_THEME
from ..llm_client import llm
from ..models import (
    CritiqueResultSchema,
    PlanSchema,
    RAGResponseSchema,
    SubagentResultSchema,
)
from .mcp_servers import (
    DatabaseServer,
    KnowledgeStoreServer,
    LocalFilesServer,
    SearchAPIServer,
)
from .memory import LongTermMemoryManager, ShortTermMemoryManager
from .state import AgentState, SubagentResult

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


# endregion


# region Memory
def memory_node(state: AgentState) -> dict:
    """Loads long-term user profile and extracts recent conversation history from short-term memory."""
    user_id = state.get("user_id") or "default_user"
    messages = state.get("messages") or []

    lt_memory = LongTermMemoryManager.load_user_memory(user_id)
    query = state.get("query") or (messages[-1].content if messages else "")

    return {
        "user_id": user_id,
        "query": str(query),
        "long_term_memory": lt_memory,
    }


# endregion


# region Planning
def planning_node(state: AgentState) -> dict:
    """Analyzes user query and memory context to generate a structured multi-step execution plan."""
    query = state["query"]
    lt_memory = state.get("long_term_memory", {})
    st_history = ShortTermMemoryManager.extract_recent_history(state.get("messages", []))

    system_prompt = (
        f"You are the Lead Planning Agent for '{CHATBOT_THEME}'.\n"
        "Analyze the user inquiry, short-term conversational context, and long-term user profile.\n"
        "Devise a concrete investigation plan delegating tasks to specialized subagents:\n"
        "- 'agent_local_files': In-depth inspection of local documentation files, APIs, and guides.\n"
        "- 'agent_database': Relational database inspection, metadata queries, and chunk relations.\n"
        "- 'agent_web_search': External search for live documentation, community practices, or external APIs.\n\n"
        f"Conversational Context:\n{st_history}\n\n"
        f"User Profile & Preferences:\n{json.dumps(lt_memory, ensure_ascii=False)}"
    )

    try:
        structured_llm = llm.with_structured_output(PlanSchema)
        plan = cast(
            PlanSchema,
            structured_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Plan investigation for query: '{query}'"),
            ]),
        )
        return {
            "intent_analysis": plan.intent_analysis,
            "plan_steps": plan.steps,
            "subagents_required": plan.subagents_required,
        }
    except Exception as exc:
        print(f"[Planning-planning_node] Structured plan failed ({exc}), using default multi-agent plan...")
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
    """Agent 1: Specializes in querying and reading local documentation files via MCP tools."""
    query = state["query"]
    plan_steps = state.get("plan_steps", [])

    instruction = query
    for step in plan_steps:
        if step.subagent == "agent_local_files":
            instruction = step.instruction
            break

    # MCP Tool Calls
    search_query = f"{instruction} {query}".strip()
    sources: list[str] = []
    excerpts: list[str] = []

    # Priority 1: High-fidelity Hybrid Retrieval (PGVector + BM25 + FlashRank Reranker)
    hybrid_matches = KnowledgeStoreServer.hybrid_search(search_query, top_k=6)
    if hybrid_matches:
        for m in hybrid_matches:
            bc = m.get("breadcrumb") or m.get("title") or "Documentation Passage"
            sources.append(bc)
            excerpts.append(f"### {m.get('title')} [{bc}]\n{m.get('content', '')[:1200]}")
    else:
        # Priority 2: Fallback to local structured JSON file scanner
        search_results = LocalFilesServer.search_local_documents(search_query, max_results=5)
        for item in search_results:
            sources.append(str(item.get("breadcrumb") or item.get("title")))
            doc_detail = LocalFilesServer.read_document_section(item["id"])
            if doc_detail:
                excerpts.append(
                    f"### {doc_detail.get('title')} ({doc_detail.get('breadcrumb')})\n"
                    f"{doc_detail.get('content', '')[:1200]}"
                )
            else:
                excerpts.append(f"### {item.get('title')}\n{item.get('excerpt', '')}")

    raw_evidence = "\n\n".join(excerpts) if excerpts else "No matching local documentation sections found."

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
        all_sources = list(set(sources + res.sources))
    except Exception:
        findings = raw_evidence[:1500]
        all_sources = sources

    result_payload: SubagentResult = {
        "agent_name": "Agent 1 (Local Files & Documentation)",
        "findings": findings,
        "sources": all_sources,
    }
    return {"agent_local_files_result": result_payload}


def agent_database_node(state: AgentState) -> dict:
    """Agent 2: Specializes in querying PostgreSQL tables, schemas, and metadata via MCP tools."""
    query = state["query"]
    plan_steps = state.get("plan_steps", [])

    instruction = query
    for step in plan_steps:
        if step.subagent == "agent_database":
            instruction = step.instruction
            break

    findings_list: list[str] = [f"Assigned database task: {instruction}"]
    sources: list[str] = []

    try:
        # MCP Tool Call: Inspect database schema
        schema_info = DatabaseServer.get_table_schema("langchain_pg_embedding")
        if schema_info:
            cols = [col["column"] for col in schema_info]
            findings_list.append(f"Database Table 'langchain_pg_embedding' available columns: {', '.join(cols)}")
            sources.append("PostgreSQL: langchain_pg_embedding")

        # MCP Tool Call: Structured query for document records matching key terms
        raw_tokens = [t for t in re.findall(r"\w+", f"{instruction} {query}") if len(t) >= 2 and t.lower() not in {"search", "kanzi", "database", "query", "what", "how", "the", "and"}]
        search_term = raw_tokens[0] if raw_tokens else "node"
        clean_term = search_term.replace("'", "''")

        rows = DatabaseServer.execute_query(
            f"SELECT id, cmetadata->>'title' as title, cmetadata->>'breadcrumb' as breadcrumb, "
            f"substring(document from 1 for 350) as snippet "
            f"FROM langchain_pg_embedding "
            f"WHERE document ILIKE '%{clean_term}%' OR cmetadata->>'title' ILIKE '%{clean_term}%' "
            f"LIMIT 4;"
        )
        if not rows:
            rows = DatabaseServer.execute_query(
                "SELECT id, cmetadata->>'title' as title, cmetadata->>'breadcrumb' as breadcrumb, "
                "substring(document from 1 for 350) as snippet "
                "FROM langchain_pg_embedding LIMIT 4;"
            )

        if rows:
            findings_list.append("Relational Database Records Found:")
            for r in rows:
                t = r.get("title") or "PostgreSQL Record"
                bc = r.get("breadcrumb") or t
                snip = r.get("snippet", "").strip()
                findings_list.append(f"- {t} [{bc}]: {snip[:200]}...")
                sources.append(f"DB: {bc}")
    except Exception as exc:
        findings_list.append(f"Database query note: {exc}")

    findings_text = "\n\n".join(findings_list) if findings_list else "Database inspected with no relevant records."
    result_payload: SubagentResult = {
        "agent_name": "Agent 2 (Relational Database)",
        "findings": findings_text,
        "sources": sources or ["PostgreSQL Documentation Store"],
    }
    return {"agent_database_result": result_payload}


def agent_web_search_node(state: AgentState) -> dict:
    """Agent 3: Specializes in querying live web search APIs and external documentation via MCP tools."""
    query = state["query"]
    plan_steps = state.get("plan_steps", [])

    instruction = query
    for step in plan_steps:
        if step.subagent == "agent_web_search":
            instruction = step.instruction
            break

    # MCP Tool Call: Web search
    search_query = f"{CHATBOT_THEME} {instruction}"
    web_results = SearchAPIServer.search_web(search_query, max_results=3)

    sources: list[str] = []
    summaries: list[str] = []

    for r in web_results:
        url = r.get("url", "")
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        if url:
            sources.append(url)
        summaries.append(f"Title: {title}\nURL: {url}\nSnippet: {snippet}")

    raw_web = "\n\n".join(summaries) if summaries else "No external web search results found."
    result_payload: SubagentResult = {
        "agent_name": "Agent 3 (Web Search & External APIs)",
        "findings": raw_web,
        "sources": sources,
    }
    return {"agent_web_search_result": result_payload}


# endregion


# region Synthesis
def synthesize_node(state: AgentState) -> dict:
    """Synthesizes the grounded final answer using aggregated evidence and user profile preferences."""
    query = state["query"]
    context = _format_context_from_subagents(state)
    lt_memory = state.get("long_term_memory", {})
    feedback = state.get("critique_feedback")

    system_prompt = (
        f"You are the Primary Synthesis Orchestrator for '{CHATBOT_THEME}'.\n"
        "Synthesize an authoritative, highly technical response directly answering the user query.\n\n"
        "CRITICAL RULES:\n"
        "1. STRICT FIDELITY: Rely strictly on the aggregated evidence provided by the subagents.\n"
        "2. INLINE CITATIONS: Immediately attach an inline citation [Topic / Source Name] to every factual claim, API reference, or guideline.\n"
        "3. CITATIONS ARRAY: Populate the 'citations' array with the distinct topics/sources referenced inline.\n"
        f"4. USER PROFILE PREFERENCES: Incorporate user depth and preferences: {json.dumps(lt_memory.get('preferences', {}))}\n"
    )
    if feedback:
        system_prompt += f"\n5. PREVIOUS CRITIQUE FEEDBACK (Address and rectify): {feedback}\n"

    user_prompt = f"User Query: {query}\n\nAggregated Subagent Evidence:\n{context}"

    try:
        structured_llm = llm.with_structured_output(RAGResponseSchema)
        rag_res = cast(
            RAGResponseSchema,
            structured_llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]),
        )
        answer = rag_res.answer
        citations = rag_res.citations
    except Exception as exc:
        print(f"[Synthesis-synthesize_node] Structured output fallback ({exc})")
        raw_res = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        answer = str(raw_res.content)
        citations = []

    # Record verified summary fact in long-term memory
    user_id = state.get("user_id") or "default_user"
    LongTermMemoryManager.record_fact(
        user_id,
        f"Inquiry on '{query[:50]}': answered with {len(citations)} citations.",
    )

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

    print(f"[Critique-critique_node] System 2 critique: is_passed={is_passed}, feedback={feedback}")

    return {
        "is_critique_passed": is_passed,
        "critique_feedback": feedback,
        "retry_count": retry_count if is_passed else retry_count + 1,
    }


# endregion
