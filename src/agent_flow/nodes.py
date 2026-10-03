# region Imports
from typing import cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ..config import CHATBOT_THEME
from ..llm_client import call_jev_decisions, hyde_llm, llm
from ..models import HyDESchema, RAGResponseSchema
from ..tools import retrieve_VDB
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
    print(f"[Classifier-classifier_node] Jev classification choice: {choice}")
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

    answers = call_jev_decisions(state_payload, questions)
    choice = answers.get("hyde_necessity", {}).get("choice", "skip")
    should_hyde = choice == "expand"
    reason = (
        "Abstract or short query benefits from hypothetical expansion (Jev Decision)"
        if should_hyde
        else "Query contains specific terms or details; skipping HyDE (Jev Decision)"
    )
    print(
        f"[HyDE-hyde_decision_node] Jev decision: should_hyde={should_hyde}, reason={reason}"
    )
    return {
        "should_hyde": should_hyde,
        "hyde_reason": reason,
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
    """Evaluates draft answer quality and groundedness using TypeSafe Jev."""
    should_answer = state.get("should_answer")
    draft = state.get("final_response", "")
    docs = state.get("retrieved_documents", [])
    query = state["query"]
    attempt_count = state.get("attempt_count", 0)

    if should_answer == "refuse":
        state_payload = {
            "user_query": query,
            "draft_response": draft,
            "theme": CHATBOT_THEME,
        }
        questions = {
            "is_polite_refusal": {
                "type": "choice",
                "instructions": (
                    f"Verify if the draft response is a polite and clear refusal to answer a query outside '{CHATBOT_THEME}'."
                ),
                "criteria": {
                    "pass": "The draft politely declines to answer the off-topic query",
                    "fail": "The draft is impolite or improperly attempts to answer an off-topic query",
                },
            }
        }
        answers = call_jev_decisions(state_payload, questions)
        choice = answers.get("is_polite_refusal", {}).get("choice", "pass")
    else:
        doc_snippets = []
        if isinstance(docs, list):
            for d in docs[:5]:
                content = getattr(d, "page_content", str(d))
                doc_snippets.append(content[:500])
        context_str = "\n---\n".join(doc_snippets) if doc_snippets else str(docs)[:2000]

        state_payload = {
            "user_query": query,
            "retrieved_context": context_str,
            "draft_response": draft,
        }
        questions = {
            "groundedness": {
                "type": "choice",
                "instructions": (
                    "Strict quality evaluation: verify if the draft response is fully grounded in the retrieved documentation context. "
                    "Ensure there are no invented facts or numbers, and that claims align with the provided context."
                ),
                "criteria": {
                    "pass": "Draft is completely grounded in retrieved context with zero hallucination",
                    "fail": "Draft contains unsupported facts, hallucinations, or contradicts retrieved context",
                },
            }
        }
        answers = call_jev_decisions(state_payload, questions)
        choice = answers.get("groundedness", {}).get("choice", "pass")

    print(f"[Critique-critique_node] Jev critique decision: choice={choice}")
    if choice == "pass":
        return {"critique_feedback": "PASS"}

    return {
        "critique_feedback": "Draft response failed groundedness validation (Jev Decision)",
        "attempt_count": attempt_count + 1,
    }


# endregion
