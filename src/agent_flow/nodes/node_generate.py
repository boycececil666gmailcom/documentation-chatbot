# region Generation Node
from typing import cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from ...llm_client import llm
from ...models import RAGResponseSchema
from ..state import AgentState


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
        response.answer.strip() if response and response.answer else "Information not available in documentation."
    )
    raw_citations = [c.strip(" []") for c in (response.citations if response else []) if c.strip()]
    unique_citations = list(dict.fromkeys(raw_citations)) if final_text != "Information not available in documentation." else []

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
