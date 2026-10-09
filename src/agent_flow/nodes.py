# region Imports
from typing import cast

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .. import vector_db as db
from ..config import CHATBOT_THEME
from ..llm_client import (
    call_jev_decisions,
    hyde_llm,
    llm,
    local_slm,
    reranker,
)
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
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]
    try:
        structured_llm = hyde_llm.with_structured_output(HyDESchema)
        res = cast(HyDESchema, structured_llm.invoke(messages))
        if res and res.passage:
            return res.passage.strip()
    except Exception as exc:
        print(
            f"[HyDE-generate_hypothetical_document] Structured invoke failed ({exc}), falling back to direct invoke..."
        )
        try:
            raw_res = hyde_llm.invoke(messages)
            if raw_res and raw_res.content:
                text_content = str(raw_res.content).strip()
                if text_content:
                    return text_content
        except Exception as inner_exc:
            print(
                f"[HyDE-generate_hypothetical_document] Direct invoke failed ({inner_exc})"
            )
    return query


def hyde_node(state: AgentState) -> dict:
    """Generates hypothetical passage and retrieves candidate chunks using dense vector similarity."""
    query = state["query"]
    print(f"[HyDE-hyde_node] Generating hypothetical doc for: '{query[:60]}...'")
    hypo_doc = generate_hypothetical_document(query)
    print(f"[HyDE-hyde_node] Hypothetical doc prepared: '{hypo_doc[:60]}...'")
    print(
        "[HyDE-hyde_node] Executing dense vector retrieval for hypothetical passage..."
    )
    docs = db.retrieve_collapsed_tree(query=hypo_doc, top_k=10)
    print(f"[HyDE-hyde_node] Retrieved {len(docs)} dense candidate chunks")
    return {
        "hypothetical_doc": hypo_doc,
        "hyde_docs": docs,
    }


# endregion


# region BM25 Node
def extract_bm25_keywords(query: str) -> str:
    """Transforms conversational query into dense technical search tokens using local SLM."""
    # Fast path: short queries (<= 3 words) are already clean search tokens
    if len(query.split()) <= 3:
        return query

    system_prompt = (
        "Extract only key technical terms, identifiers, API names, and core keywords from the user query for sparse keyword search.\n"
        "Remove conversational filler, questions, and stop words.\n"
        "Output ONLY space-separated keywords without punctuation or quotes."
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query),
    ]
    try:
        response = local_slm.invoke(messages)
        raw_text = str(response.content).strip().replace("\n", " ").replace(",", " ")
        cleaned = " ".join(raw_text.split())
        if cleaned and len(cleaned) > 2:
            return cleaned
    except Exception as exc:
        print(f"[BM25-extract_keywords] Local SLM extraction notice: {exc}")
    return query


def bm25_node(state: AgentState) -> dict:
    """Retrieves candidate document chunks using sanitized BM25 sparse keyword matching."""
    raw_query = state["query"].strip()
    sparse_query = extract_bm25_keywords(raw_query)
    print(
        f"[BM25-bm25_node] Query: '{raw_query[:50]}' -> Extracted keywords: '{sparse_query}'"
    )
    docs = db.retrieve_bm25(query=sparse_query, top_k=10)
    print(f"[BM25-bm25_node] Retrieved {len(docs)} BM25 candidate chunks")
    return {
        "bm25_query": sparse_query,
        "bm25_docs": docs,
    }


# endregion


# region Rerank Node
def get_chunk_key(doc: Document) -> str:
    """Extract unique topic identifier from document metadata."""
    return str(
        doc.metadata.get("breadcrumb") or doc.metadata.get("title") or "Unknown Topic"
    )


def format_docs_context(docs: list[Document]) -> str:
    """Formats ranked documents into a structured prompt context block enriched with RAPTOR hierarchy."""
    if not docs:
        return "No matching vector documents found."

    context_blocks: list[str] = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        topic_key = get_chunk_key(doc)
        relevance_score = meta.get("relevance_score", 0.0)
        layer = meta.get("raptor_layer", 2)

        header = f"--- [Rank {i}] Topic: [{topic_key}] (RAPTOR Layer: {layer}, Relevance Score: {relevance_score:.4f}) ---"

        # RAPTOR Hierarchical Context Expansion (Ancestors & Parent Scope)
        hierarchy_lines: list[str] = []
        ancestors = meta.get("raptor_ancestors", [])
        if ancestors:
            for anc in reversed(ancestors):
                anc_layer = anc.get("raptor_layer", 0)
                anc_title = anc.get("title", "")
                anc_lead = anc.get("lead_content", "")
                if anc_lead:
                    hierarchy_lines.append(
                        f"- [RAPTOR Level {anc_layer} Overview ({anc_title})]: {anc_lead}"
                    )

        children_topics = meta.get("raptor_children_topics", [])
        if children_topics:
            sub_titles = [c.get("title", "") for c in children_topics if c.get("title")]
            if sub_titles:
                hierarchy_lines.append(
                    f"- [RAPTOR Sub-topics in Branch]: {', '.join(sub_titles)}"
                )

        block_components = [header]
        if hierarchy_lines:
            block_components.append(
                "[RAPTOR Hierarchical Context]:\n" + "\n".join(hierarchy_lines)
            )

        raw_content = doc.page_content
        block_components.append(f"[Detailed Source Content]:\n{raw_content}")
        context_blocks.append("\n".join(block_components))

    return "\n\n".join(context_blocks)


# region Hierarchy Context Expansion Node
def context_expansion_by_hierarchy_node(state: AgentState) -> dict:
    """Merges retrieval candidates and enriches each chunk with parent document context before reranking."""
    bm25_docs = state.get("bm25_docs") or []
    hyde_docs = state.get("hyde_docs") or []

    # Merge and deduplicate candidates across parallel retrieval branches
    seen: set[str] = set()
    initial_candidates: list[Document] = []
    for doc in bm25_docs + hyde_docs:
        key = str(
            doc.metadata.get("doc_id")
            or doc.metadata.get("breadcrumb")
            or doc.metadata.get("title")
            or doc.page_content[:60]
        )
        if key not in seen:
            seen.add(key)
            initial_candidates.append(doc)

    # 1. RAPTOR Top-Down Candidate Expansion: Inject child chunks for matched summary nodes
    expanded_candidates = db.expand_raptor_candidates(
        initial_candidates, max_children_per_parent=2
    )

    # 2. Parent Context Expansion: Enrich candidates with parent document title, lead overview, and breadcrumbs
    parent_enriched_candidates = db.expand_parent_context(expanded_candidates)

    print(
        f"[HierarchyExpansion-context_expansion_by_hierarchy_node] Initial candidates {len(initial_candidates)} -> expanded to {len(expanded_candidates)} -> hierarchy-enriched {len(parent_enriched_candidates)} docs"
    )
    return {
        "expanded_docs": parent_enriched_candidates,
    }


# endregion


# region Rerank Node
def rerank_node(state: AgentState) -> dict:
    """Scores parent-expanded candidate documents using FlashRank cross-encoder."""
    expanded_docs = state.get("expanded_docs") or []

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

    ranked_docs = (
        reranker.compress_documents(expanded_docs, search_query)
        if expanded_docs
        else []
    )

    # RAPTOR Bottom-Up Context Expansion: Attach ancestor scope and summaries to top ranked docs
    enriched_ranked_docs = db.expand_raptor_context(ranked_docs)

    print(
        f"[Rerank-rerank_node] Ranked {len(expanded_docs)} parent-enriched candidates -> {len(enriched_ranked_docs)} top ranked docs"
    )
    return {
        "ranked_docs": enriched_ranked_docs,
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
        "5. MISSING INFO: If the context does not contain the answer, state 'Information not available in documentation' and return an empty citations list.\n"
        "6. HIERARCHICAL CONTEXT: Utilize the [RAPTOR Hierarchical Context] to understand the architectural domain and high-level concepts, while synthesizing specific technical details, APIs, and instructions from [Detailed Source Content]."
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
    """Returns static polite refusal for off-theme queries without LLM invocation."""
    query = state["query"]
    print(
        f"[Refusal-refuse_node] Generating static refusal for off-theme query: '{query[:60]}...'"
    )
    return {
        "draft_response": (
            f"I am a specialized technical assistant dedicated to {CHATBOT_THEME}. "
            f"I can only assist with questions directly related to {CHATBOT_THEME} documentation and workflows."
        ),
        "citations": [],
    }


# endregion


# region Critique Node
def critique_node(state: AgentState) -> dict:
    """Evaluates draft answer quality and groundedness using System 2 LLM (DeepSeek)."""
    routing_decision = state.get("routing_decision")
    draft = state.get("draft_response", "")
    query = state["query"]
    retry_count = state.get("retry_count", 0)

    if routing_decision == "refuse":
        prompt = (
            f"You are a strict quality control auditor for '{CHATBOT_THEME}'.\n"
            f"Verify if refusing the query is appropriate because it is outside the scope of '{CHATBOT_THEME}'.\n"
            f"If the user query is actually related to '{CHATBOT_THEME}', reject the refusal so the agent can route properly.\n"
            f"User Query: {query}\n"
            f"Draft Refusal: {draft}\n\n"
            "Return valid JSON matching CritiqueResultSchema:\n"
            "- is_passed: true if query is off-topic and refusal is appropriate, false if the query was on-topic\n"
            "- feedback: concise explanation if false, otherwise null"
        )
    else:
        docs = format_docs_context(state.get("ranked_docs", []))
        prompt = (
            f"You are a quality control auditor verifying documentation answers.\n"
            f"User Query: {query}\n\n"
            f"Retrieved Documentation Context:\n{docs}\n\n"
            f"Draft Response:\n{draft}\n\n"
            "Evaluation Rules:\n"
            "1. Groundedness: Is the answer supported by the retrieved documentation without unverified extrapolation?\n"
            "2. Inline Citations: Do inline citations accurately reference source topics?\n\n"
            "Return valid JSON matching CritiqueResultSchema:\n"
            "- is_passed: true if grounded and accurate, false otherwise\n"
            "- feedback: concise explanation of what claim or citation failed if false, otherwise null"
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
