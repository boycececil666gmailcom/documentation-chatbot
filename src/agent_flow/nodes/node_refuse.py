# region Refusal Node
from langchain_core.messages import HumanMessage, SystemMessage

from ...config import CHATBOT_THEME
from ...llm_client import llm
from ...models import RAGResponseSchema
from ..state import AgentState


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
