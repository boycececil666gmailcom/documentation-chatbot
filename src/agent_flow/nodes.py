# region Imports
from typing import cast

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .. import vector_db as db
from ..config import CHATBOT_THEME
from ..llm_client import call_jev_decisions, hyde_llm, llm, reranker
from ..models import CritiqueResultSchema, HyDESchema, RAGResponseSchema
from .state import AgentState

# endregion


# region Router Node
def router_node(state: AgentState) -> dict:
    """Classifies domain scope and selects retrieval strategy in a single Jev decision."""
    query = state["query"].strip()

    state_payload = {"user_query": query}
    questions = {
        "routing_strategy": {
            "type": "choice",
            "instructions": (
                f"Allowed Domain/Theme: '{CHATBOT_THEME}'. "
                "Classify domain relevance and determine the optimal retrieval strategy:\n"
                "- 'refuse': Completely off-topic or unrelated query (e.g., cooking, sports, medicine).\n"
                "- 'bm25': Query contains explicit keywords, exact API names, error codes, or identifiers (direct BM25 search).\n"
                "- 'hyde_bm25': General technical question, how-to, or conceptual workflow within domain (hybrid BM25 + HyDE).\n"
                "- 'hyde': Very short, abstract, or ambiguous query lacking specific technical keywords (HyDE expansion)."
            ),
            "criteria": {
                "refuse": "Off-topic query completely outside domain",
                "bm25": "Contains specific identifiers, exact APIs, or error codes",
                "hyde_bm25": "General technical question or workflow within domain",
                "hyde": "Abstract, short, or ambiguous query lacking keywords",
            },
        }
    }

    answers = call_jev_decisions(
        state_payload,
        questions,
        min_confidence=0.20,
        fallback="hyde_bm25",
    )
    decision = answers.get("routing_strategy", {}).get("choice", "hyde_bm25")
    return {
        "routing_decision": decision,
        "bm25_docs": [],
        "hyde_docs": [],
    }


# endregion


# region HyDE Node
def generate_hypothetical_document(query: str) -> str:
    """Generates a domain-injected hypothetical document passage for query expansion."""
    system_prompt = (
        f"You are a senior technical documentation author for '{CHATBOT_THEME}'.\n"
        "Write a concise, realistic 2-3 sentence documentation excerpt that directly answers the user's query.\n"
        "Include relevant domain-specific concepts, APIs, and tool terminology if applicable. Output only the excerpt."
    )
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]
        structured_llm = hyde_llm.with_structured_output(HyDESchema)
        res = cast(HyDESchema, structured_llm.invoke(messages))
        if res and res.passage:
            return res.passage.strip()
    except Exception:
        pass
    return query


def hyde_node(state: AgentState) -> dict:
    """Generates hypothetical passage and retrieves candidate chunks using dense vector similarity."""
    query = state["query"]
    print(f"[HyDE-hyde_node] Generating hypothetical doc for: '{query[:60]}...'")
    hypo_doc = generate_hypothetical_document(query)
    print(f"[HyDE-hyde_node] Hypothetical doc prepared: '{hypo_doc[:60]}...'")
    print("[HyDE-hyde_node] Executing dense vector retrieval for hypothetical passage...")
    docs = db.retrieve_collapsed_tree(query=hypo_doc, top_k=10)
    print(f"[HyDE-hyde_node] Retrieved {len(docs)} dense candidate chunks")
    return {
        "hypothetical_doc": hypo_doc,
        "hyde_docs": docs,
    }


# endregion


# region BM25 Node
def bm25_node(state: AgentState) -> dict:
    """Retrieves candidate document chunks using pure BM25 sparse keyword matching."""
    query = state["query"].strip()
    print(f"[BM25-bm25_node] Executing BM25 keyword retrieval for: '{query[:60]}...'")
    docs = db.retrieve_bm25(query=query, top_k=10)
    print(f"[BM25-bm25_node] Retrieved {len(docs)} BM25 candidate chunks")
    return {
        "bm25_query": query,
        "bm25_docs": docs,
    }


# endregion


# region Rerank Node
def get_chunk_key(doc: Document) -> str:
    """Extract unique topic identifier from document metadata."""
    return str(doc.metadata.get("breadcrumb") or doc.metadata.get("title") or "Unknown Topic")


def format_docs_context(docs: list[Document]) -> str:
    """Formats ranked documents into a structured prompt context block."""
    if not docs:
        return "No matching vector documents found."
    return "\n\n".join(
        f"--- [Rank {i}] Topic: [{get_chunk_key(doc)}] (Relevance Score: {doc.metadata.get('relevance_score', 0.0):.4f}) ---\n"
        f"{doc.metadata.get('big') or doc.page_content}"
        for i, doc in enumerate(docs, 1)
    )


def rerank_node(state: AgentState) -> dict:
    """Merges and scores candidate documents from BM25 and HyDE using FlashRank cross-encoder."""
    bm25_docs = state.get("bm25_docs") or []
    hyde_docs = state.get("hyde_docs") or []
    legacy_docs = state.get("retrieved_docs") or []

    # Merge and deduplicate candidates across both parallel retrieval branches
    seen: set[str] = set()
    candidate_docs: list[Document] = []
    for doc in bm25_docs + hyde_docs + legacy_docs:
        key = str(doc.metadata.get("breadcrumb") or doc.metadata.get("title") or doc.page_content[:60])
        if key not in seen:
            seen.add(key)
            candidate_docs.append(doc)

    bm25_query = state.get("bm25_query")
    hypo_doc = state.get("hypothetical_doc")
    query = state["query"]

    if bm25_query and hypo_doc:
        search_query = f"{bm25_query}\n{hypo_doc}"
    elif hypo_doc:
        search_query = hypo_doc
    elif bm25_query:
        search_query = bm25_query
    else:
        search_query = query

    ranked_docs = reranker.compress_documents(candidate_docs, search_query) if candidate_docs else []
    print(
        f"[Rerank-rerank_node] Merged {len(candidate_docs)} candidate docs -> {len(ranked_docs)} ranked docs"
    )
    return {
        "retrieved_docs": candidate_docs,
        "ranked_docs": ranked_docs,
        "search_query": search_query,
    }


# endregion


# region Generation Node
def generate_node(state: AgentState) -> dict:
    """Synthesizes strictly grounded response prioritized by document relevance ranks."""
    query = state["query"]
    ranked_docs = state.get("ranked_docs", [])
    retrieved_context = format_docs_context(ranked_docs)

    system_prompt = (
        f"Retrieved Document Context (Ordered by Cross-Encoder Relevance):\n{retrieved_context}\n\n"
        "CRITICAL RULES:\n"
        "1. RANK PRIORITY: The context documents are strictly sorted by relevance (Rank 1 is the primary and most authoritative source). "
        "Synthesize your answer primarily from the highest-ranked documents (Rank 1 and Rank 2). "
        "Do not allow lower-ranked or tangential details to contradict or dilute information from higher-ranked documents.\n"
        "2. GROUNDEDNESS: Your answer must be strictly grounded in the retrieved document context. Never invent facts.\n"
        "3. INLINE CITATIONS: For every factual claim, guideline, or step in your answer, immediately attach an inline citation specifying the exact source topic in brackets (e.g., 'To reduce draw calls, batch static meshes [Performance > Meshes].'). Place citations directly on the relevant sentence or bullet point, NOT as a vague generic dump at the end.\n"
        "4. CITATIONS ARRAY: In the 'citations' field, include only the topic names that you actively cited inline in the answer.\n"
        "5. MISSING INFO: If the context does not contain the answer, state 'Information not available in documentation' and return an empty citations list."
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]

    # Append retry instruction if previous attempt failed critique
    prev_draft = state.get("draft_response")
    feedback = state.get("critique_feedback")
    if state.get("is_critique_passed") is False and prev_draft:
        messages.append(AIMessage(content=prev_draft))
        critique_msg = (
            f"CRITIQUE FEEDBACK: Previous draft was rejected because: {feedback}\n"
            if feedback
            else "CRITIQUE: Previous draft was rejected due to lack of strict groundedness in retrieved documents.\n"
        )
        messages.append(
            HumanMessage(
                content=(
                    f"{critique_msg}"
                    "Revise your answer to strictly ground every claim with precise inline citations [Topic Name] matching the source chunks."
                )
            )
        )

    structured_llm = llm.with_structured_output(RAGResponseSchema)
    response = cast(RAGResponseSchema, structured_llm.invoke(messages))
    final_text = (
        response.answer.strip()
        if response and response.answer
        else "Information not available in documentation."
    )
    raw_citations = [
        c.strip(" []") for c in (response.citations if response else []) if c.strip()
    ]
    unique_citations = (
        list(dict.fromkeys(raw_citations))
        if final_text != "Information not available in documentation."
        else []
    )

    # Validate citations against actual topics present in ranked_docs
    valid_topics = {get_chunk_key(doc) for doc in ranked_docs}
    validated_citations = (
        [c for c in unique_citations if c in valid_topics]
        if valid_topics and unique_citations
        else unique_citations
    )

    return {
        "draft_response": final_text,
        "citations": validated_citations,
    }


# endregion


# region Refusal Node
def refuse_node(state: AgentState) -> dict:
    """Generates polite refusal for off-theme queries."""
    system_prompt = (
        f"You are a customer service assistant bound to the theme '{CHATBOT_THEME}'.\n"
        f"Politely explain that you can only assist with questions related to '{CHATBOT_THEME}', "
        f"and decline to answer this query."
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["query"]),
    ]
    structured_llm = llm.with_structured_output(RAGResponseSchema)
    response: RAGResponseSchema = structured_llm.invoke(messages)
    refusal_text = (
        response.answer.strip()
        if response and response.answer
        else f"I can only assist with questions related to '{CHATBOT_THEME}'."
    )

    return {
        "draft_response": refusal_text,
        "citations": [],
    }


# endregion


# region Critique Node
def critique_node(state: AgentState) -> dict:
    """Evaluates draft answer quality and groundedness using System 2 LLM (DeepSeek)."""
    routing_decision = state.get("routing_decision")
    draft = state.get("draft_response", "")
    docs = format_docs_context(state.get("ranked_docs", []))
    query = state["query"]
    retry_count = state.get("retry_count", 0)

    if routing_decision == "refuse":
        prompt = (
            f"You are a strict quality control evaluator.\n"
            f"Verify if the draft response is a polite and clear refusal to answer a query outside the theme: '{CHATBOT_THEME}'.\n"
            f"User Query: {query}\n"
            f"Draft Response: {draft}\n\n"
            "Return valid JSON matching CritiqueResultSchema:\n"
            '- is_passed: true if polite refusal, false otherwise\n'
            '- feedback: explanation string if false, otherwise null'
        )
    else:
        prompt = (
            f"You are a quality control auditor verifying documentation answers.\n"
            f"User Query: {query}\n\n"
            f"Retrieved Documentation Context:\n{docs}\n\n"
            f"Draft Response:\n{draft}\n\n"
            "Evaluation Rules:\n"
            "1. Groundedness: Is the answer supported by the retrieved documentation without unverified extrapolation?\n"
            "2. Inline Citations: Do inline citations accurately reference source topics?\n\n"
            "Return valid JSON matching CritiqueResultSchema:\n"
            '- is_passed: true if grounded and accurate, false otherwise\n'
            '- feedback: concise explanation of what claim or citation failed if false, otherwise null'
        )

    try:
        structured_llm = llm.with_structured_output(CritiqueResultSchema)
        eval_result = cast(
            CritiqueResultSchema,
            structured_llm.invoke([HumanMessage(content=prompt)]),
        )
        is_passed = eval_result.is_passed if eval_result else True
        feedback = (
            eval_result.feedback
            if (eval_result and not is_passed)
            else ("Failed groundedness validation" if not is_passed else None)
        )
    except Exception as exc:
        print(f"[Critique-critique_node] Warning during critique evaluation: {exc}")
        is_passed = True
        feedback = None

    print(
        f"[Critique-critique_node] System 2 critique: is_passed={is_passed}, "
        f"feedback={feedback}"
    )

    return {
        "is_critique_passed": is_passed,
        "critique_feedback": feedback,
        "retry_count": retry_count if is_passed else retry_count + 1,
    }


# endregion
