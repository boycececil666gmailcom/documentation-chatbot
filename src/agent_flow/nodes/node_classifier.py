# region Classifier Node
from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage

from ...config import CHATBOT_THEME
from ...llm_client import llm
from ...models import ClassifierSchema
from ..state import AgentState


def classifier_node(state: AgentState) -> dict:
    """Classifies if query aligns with configured chatbot theme using Pydantic structured output."""
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
