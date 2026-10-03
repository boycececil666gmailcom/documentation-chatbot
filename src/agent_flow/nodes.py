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


# region Classifier Node
def classifier_node(state: AgentState) -> dict:
    """Classifies if query aligns with configured chatbot theme using TypeSafe Jev."""
    query = state["query"]

    state_payload = {"user_query": query}
    questions = {
        "domain_scope": {
            "type": "choice",
            "instructions": (
                f"Allowed Domain/Theme: '{CHATBOT_THEME}'. "
                "Determine if the user query is relevant to this domain or general technical questions/greetings related to it. "
                "Assume the user is already working within this domain unless completely unrelated (cooking, sports, medicine, etc.)."
            ),
            "criteria": {
                "pass": "Relevant technical question, concept, or greeting within domain",
                "refuse": "Completely unrelated off-topic query",
            },
        }
    }

    answers = call_jev_decisions(state_payload, questions)
    choice = answers.get("domain_scope", {}).get("choice", "pass")
    return {"should_answer": choice}


# endregion


# region HyDE Nodes
def hyde_decision_node(state: AgentState) -> dict:
    """Decides whether HyDE expansion is beneficial for the user query using TypeSafe Jev."""
    query = state["query"].strip()

    state_payload = {"user_query": query}
    questions = {
        "hyde_necessity": {
            "type": "choice",
            "instructions": (
                "Evaluate whether hypothetical document expansion (HyDE) is needed for semantic vector retrieval. "
                "- Choose 'expand' for short, conceptual, or abstract questions lacking explicit identifiers or technical keywords. "
                "- Choose 'skip' for queries that already contain specific error codes, API names, function signatures, version numbers, or detailed context."
            ),
            "criteria": {
                "expand": "Short or abstract query that benefits from hypothetical passage generation",
                "skip": "Query has specific keywords, identifiers, error codes, or is already detailed",
            },
        }
    }

    # Centralized low-confidence fallback handling (Policy 2: default to expand on confidence < 0.20)
    answers = call_jev_decisions(
        state_payload,
        questions,
        min_confidence=0.20,
        fallback="expand",
    )
    decision = answers.get("hyde_necessity", {})
    should_hyde = decision.get("choice") == "expand"
    return {"should_hyde": should_hyde}


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
    """Generates hypothetical document passage and updates agent state."""
    return {"hyde_content": generate_hypothetical_document(state["query"])}


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
    """Retrieves document context from vector database, deduplicating against already loaded chunks."""
    query = state["query"]
    hypo_doc = state.get("hyde_content")
    search_target = hypo_doc if hypo_doc else query

    retrieved_documents = state.get("retrieved_documents")
    loaded_keys = list(state.get("loaded_doc_keys", []))

    if not retrieved_documents:
        ranked_docs = fetch_ranked_documents(search_target)
        new_chunks: list[str] = []
        new_keys: list[str] = []

        for doc in ranked_docs:
            key = get_chunk_key(doc)
            new_keys.append(key)
            content = doc.metadata.get("big") or doc.page_content
            score = doc.metadata.get("relevance_score", 0.0)
            new_chunks.append(f"[{key}] (Score: {score:.3f})\n{content}")

        if new_chunks:
            retrieved_documents = "=== NEW VECTOR CONTEXT ===\n" + "\n\n".join(new_chunks)
            loaded_keys = new_keys
        else:
            retrieved_documents = "No matching vector documents found."
    else:
        # Multi-turn branch: filter out already loaded chunks to avoid duplicate context
        ranked_docs = fetch_ranked_documents(search_target)
        new_chunks: list[str] = []
        new_keys: list[str] = []

        for doc in ranked_docs:
            key = get_chunk_key(doc)
            if key not in loaded_keys:
                new_keys.append(key)
                content = doc.metadata.get("big") or doc.page_content
                score = doc.metadata.get("relevance_score", 0.0)
                new_chunks.append(f"[{key}] (Score: {score:.3f})\n{content}")

        if new_chunks:
            retrieved_documents = "=== NEW VECTOR CONTEXT ===\n" + "\n\n".join(new_chunks)
            loaded_keys.extend(new_keys)
        else:
            retrieved_documents = "All relevant document context is already loaded in previous conversation turns."

    return {
        "retrieved_documents": retrieved_documents,
        "loaded_doc_keys": loaded_keys,
    }


# endregion


# region Generation Node
def generate_node(state: AgentState) -> dict:
    """Synthesizes strictly grounded response based on retrieved documents and conversation history."""
    query = state["query"]
    history = state.get("history", [])
    retrieved_documents = state.get("retrieved_documents", "")

    system_prompt = (
        f"Retrieved Document Context:\n{retrieved_documents}\n\n"
        "CRITICAL RULES:\n"
        "1. GROUNDEDNESS: Your answer must be strictly grounded in the retrieved document context. Never invent facts.\n"
        "2. INLINE CITATIONS: For every factual claim, guideline, or step in your answer, immediately attach an inline citation specifying the exact source topic in brackets (e.g., 'To reduce draw calls, batch static meshes [Performance > Meshes].'). Place citations directly on the relevant sentence or bullet point, NOT as a vague generic dump at the end.\n"
        "3. CITATIONS ARRAY: In the 'citations' field, include only the topic names that you actively cited inline in the answer.\n"
        "4. MISSING INFO: If the context does not contain the answer, state 'Information not available in documentation' and return an empty citations list."
    )

    messages = [SystemMessage(content=system_prompt)]

    # 1. Replay established conversational history
    for msg in history:
        role, content = msg.get("role"), msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    # 2. Append turn instruction: if fresh turn, append query; if retry, append critique feedback
    critique_passed = state.get("critique_passed")
    feedback = state.get("critique_feedback")
    if critique_passed is False:
        critique_msg = (
            f"CRITIQUE FEEDBACK: Previous draft was rejected because: {feedback}\n"
            if feedback
            else "CRITIQUE: Previous draft was rejected due to lack of strict groundedness in retrieved documents.\n"
        )
        turn_user_msg = (
            f"{critique_msg}"
            "Revise your answer to strictly ground every claim with precise inline citations [Topic Name] matching the source chunks."
        )
    else:
        turn_user_msg = query

    messages.append(HumanMessage(content=turn_user_msg))

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

    # 3.  (KV Cache Priority): Maintain exact token prefix sequence across turns and retries
    updated_history = list(history) + [
        {"role": "user", "content": turn_user_msg},
        {"role": "assistant", "content": final_text},
    ]

    return {
        "final_response": final_text,
        "citations": unique_citations,
        "history": updated_history,
        "loaded_doc_keys": state.get("loaded_doc_keys", []),
    }


# endregion


# region Refusal Node
def refuse_node(state: AgentState) -> dict:
    """Generates polite refusal for off-theme queries and preserves conversation history."""
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

    history = state.get("history", [])
    updated_history = list(history) + [
        {"role": "user", "content": state["query"]},
        {"role": "assistant", "content": refusal_text},
    ]

    return {
        "final_response": refusal_text,
        "citations": [],
        "history": updated_history,
        "loaded_doc_keys": state.get("loaded_doc_keys", []),
    }


# endregion


# region Critique Node
def critique_node(state: AgentState) -> dict:
    """Evaluates draft answer quality and groundedness using System 2 LLM (DeepSeek)."""
    should_answer = state.get("should_answer")
    draft = state.get("final_response", "")
    docs = state.get("retrieved_documents", "")
    query = state["query"]
    attempt_count = state.get("attempt_count", 0)

    if should_answer == "refuse":
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
        "critique_passed": is_passed,
        "critique_feedback": feedback,
        "attempt_count": attempt_count if is_passed else attempt_count + 1,
    }


# endregion
