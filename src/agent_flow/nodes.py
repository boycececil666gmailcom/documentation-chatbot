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
                "- 'keyword': Query contains explicit keywords, exact API names, error codes, or identifiers (direct BM25 search).\n"
                "- 'general': General technical question, how-to, or conceptual workflow within domain (hybrid BM25 + HyDE).\n"
                "- 'vague': Very short, abstract, or ambiguous query lacking specific technical keywords (HyDE expansion)."
            ),
            "criteria": {
                "refuse": "Off-topic query completely outside domain",
                "keyword": "Contains specific identifiers, exact APIs, or error codes",
                "general": "General technical question or workflow within domain",
                "vague": "Abstract, short, or ambiguous query lacking keywords",
            },
        }
    }

    answers = call_jev_decisions(
        state_payload,
        questions,
        min_confidence=0.20,
        fallback="general",
    )
    decision = answers.get("routing_strategy", {}).get("choice", "general")
    return {"routing_decision": decision}


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
    """Generates hypothetical document passage for vague or keyword-lacking queries."""
    hypo_doc = generate_hypothetical_document(state["query"])
    print(f"[HyDE-hyde_node] Hypothetical doc prepared: '{hypo_doc[:60]}...'")
    return {
        "hypothetical_doc": hypo_doc,
        "search_query": hypo_doc,
    }


# endregion


# region BM25 Node
def bm25_node(state: AgentState) -> dict:
    """Extracts and normalizes keywords for pure BM25 sparse keyword matching."""
    query = state["query"].strip()
    print(f"[BM25-bm25_node] Pure BM25 keyword query: '{query[:60]}...'")
    return {
        "bm25_query": query,
        "search_query": query,
    }


# endregion


# region HyDE + BM25 Node
def hyde_bm25_node(state: AgentState) -> dict:
    """Generates HyDE passage and combines it with BM25 keywords by composing hyde_node and bm25_node."""
    hyde_res = hyde_node(state)
    bm25_res = bm25_node(state)
    combined_query = f"{bm25_res['bm25_query']}\n{hyde_res['hypothetical_doc']}"
    print(f"[HyDE_BM25-hyde_bm25_node] Hybrid query prepared: '{combined_query[:60]}...'")
    return {
        **hyde_res,
        **bm25_res,
        "search_query": combined_query,
    }


# endregion


# region Retrieval Node
def fetch_ranked_documents(query: str) -> list[Document]:
    """Retrieve and rerank document chunks for a query."""
    docs = db.retrieve_collapsed_tree(query=query, top_k=10, max_tokens=4000)
    return reranker.compress_documents(docs, query)


def get_chunk_key(doc: Document) -> str:
    """Extract unique topic identifier from document metadata."""
    return str(doc.metadata.get("breadcrumb") or doc.metadata.get("title") or "Unknown Topic")


def retrieve_node(state: AgentState) -> dict:
    """Retrieves document context from vector database using selected strategy."""
    search_target = state.get("search_query") or state["query"]
    ranked_docs = fetch_ranked_documents(search_target)
    formatted_chunks = [
        f"[{get_chunk_key(doc)}] (Score: {doc.metadata.get('relevance_score', 0.0):.3f})\n"
        f"{doc.metadata.get('big') or doc.page_content}"
        for doc in ranked_docs
    ]

    retrieved_context = (
        "=== VECTOR DATABASE CONTEXT ===\n" + "\n\n".join(formatted_chunks)
        if formatted_chunks
        else "No matching vector documents found."
    )

    return {"retrieved_context": retrieved_context}


# endregion


# region Generation Node
def generate_node(state: AgentState) -> dict:
    """Synthesizes strictly grounded response based on retrieved documents and query."""
    query = state["query"]
    retrieved_context = state.get("retrieved_context", "")

    system_prompt = (
        f"Retrieved Document Context:\n{retrieved_context}\n\n"
        "CRITICAL RULES:\n"
        "1. GROUNDEDNESS: Your answer must be strictly grounded in the retrieved document context. Never invent facts.\n"
        "2. INLINE CITATIONS: For every factual claim, guideline, or step in your answer, immediately attach an inline citation specifying the exact source topic in brackets (e.g., 'To reduce draw calls, batch static meshes [Performance > Meshes].'). Place citations directly on the relevant sentence or bullet point, NOT as a vague generic dump at the end.\n"
        "3. CITATIONS ARRAY: In the 'citations' field, include only the topic names that you actively cited inline in the answer.\n"
        "4. MISSING INFO: If the context does not contain the answer, state 'Information not available in documentation' and return an empty citations list."
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

    return {
        "draft_response": final_text,
        "citations": unique_citations,
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
    docs = state.get("retrieved_context", "")
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
