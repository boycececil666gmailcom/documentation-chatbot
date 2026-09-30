# region Imports
import re
from typing import cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..config import CHATBOT_THEME
from ..llm_client import hyde_llm, llm
from ..models import (
    ClassifierSchema,
    CritiqueResultSchema,
    HyDESchema,
    RAGResponseSchema,
)
from ..state import AgentState
from ..tools import retrieve_VDB

# endregion


# region Classifier Node
def classifier_node(state: AgentState) -> dict:
    """Classifies if query aligns with configured chatbot theme using structured output."""
    system_prompt = (
        f"You are a domain intent classifier for a technical support assistant.\n"
        f"Allowed Domain/Theme: '{CHATBOT_THEME}'.\n\n"
        "Determine if the user query is relevant to this domain or is general greetings/technical questions related to it.\n"
        "Guideline: Users often ask implicit technical questions without explicitly repeating the product name (e.g., asking about logs, environment variables, build configs, UI layouts, shaders, performance, plugins).\n"
        "Assume the user is already working within this domain context unless the query is clearly and completely unrelated (e.g., cooking recipes, sports, medical advice).\n\n"
        "Respond with a JSON object matching this schema:\n"
        '- "category": "pass" if potentially on-topic or implicit technical query, "refuse" if completely off-topic\n'
        '- "reason": optional explanation string'
    )

    structured_llm = llm.with_structured_output(ClassifierSchema)
    result = cast(
        ClassifierSchema,
        structured_llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=state["query"]),
            ]
        ),
    )

    return {"should_answer": result.category if result else "refuse"}


# endregion


# region HyDE Nodes
def hyde_decision_node(state: AgentState) -> dict:
    """Decides whether HyDE expansion is beneficial for the user query."""
    query = state["query"].strip()

    # Skip HyDE for technical codes, versions, or error patterns
    error_pattern = r"(error|err|code|uuid|v\d+\.\d+|\b[A-Z]{2,}-\d+\b|\b\d{3,5}\b)"
    if re.search(error_pattern, query, re.IGNORECASE):
        return {
            "should_hyde": False,
            "hyde_reason": "Query contains specific identifier or error pattern",
        }

    # Skip HyDE for detailed long queries
    if len(query) > 80 or len(query.split()) >= 12:
        return {
            "should_hyde": False,
            "hyde_reason": "Query is already specific and detailed",
        }

    # Enable HyDE for short or abstract queries
    return {
        "should_hyde": True,
        "hyde_reason": "Abstract or short query benefits from hypothetical expansion",
    }


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

    # Append conversational history
    for msg in history:
        role, content = msg.get("role"), msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=query))

    # Append critique feedback for retry loops if present
    feedback = state.get("critique_feedback")
    prev_draft = state.get("final_response")
    if feedback and prev_draft:
        messages.append(AIMessage(content=prev_draft))
        messages.append(
            HumanMessage(
                content=(
                    f"CRITIQUE FEEDBACK: Previous draft was rejected because: {feedback}\n"
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

    updated_history = list(history) + [
        {"role": "user", "content": query},
        {"role": "assistant", "content": final_text},
    ]

    return {
        "final_response": final_text,
        "citations": unique_citations,
        "history": updated_history,
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
    return {"final_response": response.answer, "citations": []}


# endregion


# region Critique Node
def critique_node(state: AgentState) -> dict:
    """Evaluates draft answer quality and groundedness against retrieved context or refusal rules."""
    should_answer = state.get("should_answer")
    draft = state.get("final_response")
    docs = state.get("retrieved_documents")
    query = state["query"]
    attempt_count = state.get("attempt_count", 0)
    hypo_doc = state.get("hyde_content")

    if should_answer == "refuse":
        prompt = (
            f"You are a strict quality control evaluator.\n"
            f"Verify if the draft response is a polite refusal to answer a query outside the theme: '{CHATBOT_THEME}'.\n"
            f"User Query: {query}\n"
            f"Draft Response: {draft}\n\n"
            "Return a valid JSON object matching this schema:\n"
            '- "is_passed": true if the draft is a polite refusal, false otherwise\n'
            '- "feedback": explanation string if is_passed is false, otherwise null'
        )
    else:
        prompt = (
            f"You are a strict quality control evaluator.\n"
            f"Verify if the draft response is fully grounded in the retrieved documents context and that all inline citations [Topic Name] accurately correspond to the specific facts cited from retrieved topics.\n"
            f"STRICT CHECK: No extrapolated or invented numbers/facts.\n"
            f"User Query: {query}\n"
            f"HyDE Passage: {hypo_doc or 'N/A'}\n"
            f"Retrieved Context:\n{docs}\n"
            f"Draft Response: {draft}\n\n"
            "Return a valid JSON object matching this schema:\n"
            '- "is_passed": true if fully grounded with zero hallucination, false otherwise\n'
            '- "feedback": explanation string if is_passed is false, otherwise null'
        )

    try:
        structured_llm = llm.with_structured_output(CritiqueResultSchema)
        eval_result = cast(
            CritiqueResultSchema,
            structured_llm.invoke([HumanMessage(content=prompt)]),
        )

        if eval_result and eval_result.is_passed:
            return {"critique_feedback": "PASS"}

        return {
            "critique_feedback": (eval_result.feedback if eval_result else None)
            or "Failed groundedness validation",
            "attempt_count": attempt_count + 1,
        }
    except Exception:
        # Fallback to PASS on upstream schema validation errors to prevent pipeline crash
        return {"critique_feedback": "PASS"}


# endregion
