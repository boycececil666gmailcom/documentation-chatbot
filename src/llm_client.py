# region Imports
from collections.abc import Callable
from typing import Any

import httpx
from flashrank import Ranker
from langchain_community.document_compressors.flashrank_rerank import (
    FlashrankRerank,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langsmith import traceable

from .config import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_DECISIONS_URL,
    OPENROUTER_EMBED_MODEL,
    OPENROUTER_JEV_MODEL,
    OPENROUTER_MODEL,
    OPENROUTER_PROVIDER_IGNORE,
    OPENROUTER_PROVIDER_SORT,
    OPENROUTER_TEMPERATURE,
)

# endregion


# region Generative Clients
_provider_config = {"allow_fallbacks": True}
if OPENROUTER_PROVIDER_SORT:
    _provider_config["sort"] = OPENROUTER_PROVIDER_SORT
if OPENROUTER_PROVIDER_IGNORE:
    _provider_config["ignore"] = OPENROUTER_PROVIDER_IGNORE

_extra_body = {"provider": _provider_config} if _provider_config else {}

# Primary LLM instance for standard generation (DeepSeek via OpenRouter)
llm = ChatOpenAI(
    model=OPENROUTER_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    temperature=OPENROUTER_TEMPERATURE,
    extra_body=_extra_body if _extra_body else None,
)

# High-fidelity API LLM instance for HyDE passage generation (reverted to DeepSeek)
hyde_llm = ChatOpenAI(
    model=OPENROUTER_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    temperature=0.0,
    extra_body=_extra_body if _extra_body else None,
)

# Dedicated local Ollama SLM client (preserved for local routing, classification, or SFT evaluation)
local_slm = ChatOpenAI(
    model=OLLAMA_MODEL,
    api_key="ollama",
    base_url=OLLAMA_BASE_URL,
    temperature=0.0,
)

# Shared embeddings client for vector store and theme similarity
embeddings = OpenAIEmbeddings(
    model=OPENROUTER_EMBED_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    check_embedding_ctx_length=False,
    model_kwargs={"encoding_format": "float"},
)

# Shared Cross-Encoder reranker instance
reranker = FlashrankRerank(client=Ranker(), top_n=5)
# endregion


# region Jev Decisions
@traceable(run_type="llm", name="typesafe/jev")
def call_jev_decisions(
    state_payload: dict[str, Any],
    questions: dict[str, Any],
    min_confidence: float = 0.0,
    fallback: str | Callable[[float, dict[str, float]], str] | None = None,
) -> dict[str, Any]:
    """Invokes OpenRouter Decisions API with TypeSafe Jev model.

    If min_confidence > 0 and confidence is below threshold, overrides choice
    with fallback value or result of fallback callback function.
    """
    model = (
        "typesafe/jev-1.13"
        if OPENROUTER_JEV_MODEL == "typesafe/jev-latest"
        else OPENROUTER_JEV_MODEL
    )

    payload = {
        "model": model,
        "state": state_payload,
        "questions": questions,
    }
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/boyce/documentation-chatbot",
        "X-Title": "Documentation Chatbot",
    }

    with httpx.Client(timeout=15.0) as client:
        response = client.post(OPENROUTER_DECISIONS_URL, json=payload, headers=headers)
        if response.status_code != 200:
            print(
                f"[Jev-call_jev_decisions] HTTP Error {response.status_code}: {response.text}"
            )
            raise RuntimeError(
                f"Jev API returned HTTP {response.status_code}: {response.text}"
            )

        data = response.json()
        answers: dict[str, Any] = data.get("answers", {})

        for q_key, ans in answers.items():
            conf = float(ans.get("confidence", 0.0))
            probs = ans.get("probabilities", {})
            choice = ans.get("choice")
            print(
                f"[Jev-call_jev_decisions] Question '{q_key}': choice='{choice}', "
                f"confidence={conf:.2f}, probabilities={probs}"
            )

            # Centralized low-confidence fallback handling
            if min_confidence > 0 and conf < min_confidence and fallback is not None:
                resolved_choice = (
                    fallback(conf, probs) if callable(fallback) else fallback
                )
                print(
                    f"[Jev-call_jev_decisions] Low confidence ({conf:.2f} < {min_confidence:.2f}) on '{q_key}'. "
                    f"Overriding choice '{choice}' -> '{resolved_choice}' via fallback."
                )
                ans["choice"] = resolved_choice
                ans["fallback_applied"] = True

        return answers


# endregion
