# region Imports & Setup
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
# endregion


# region Model Factory
def get_eval_models(temperature: float = 0.0, is_generator: bool = False):
    """Initializes LLM and Embeddings wrappers for Ragas evaluation and testset generation."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "[EvalClient-init] OPENROUTER_API_KEY is not set in environment."
        )

    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4-flash-0731")
    embed_model = os.getenv("OPENROUTER_EMBED_MODEL", "nvidia/nemotron-3-embed-1b:free")
    provider_sort = os.getenv("OPENROUTER_PROVIDER_SORT", "throughput")
    provider_ignore_raw = os.getenv("OPENROUTER_PROVIDER_IGNORE", "wafer")
    provider_ignore = [p.strip() for p in provider_ignore_raw.split(",") if p.strip()]

    provider_config = {"allow_fallbacks": True}
    if provider_sort:
        provider_config["sort"] = provider_sort
    if provider_ignore:
        provider_config["ignore"] = provider_ignore

    extra_body = {
        "provider": provider_config,
        "reasoning": {
            "effort": "medium",
        },
    }

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=16384,
        request_timeout=240.0,
        extra_body=extra_body,
    )
    embeddings = OpenAIEmbeddings(
        model=embed_model,
        api_key=api_key,
        base_url=base_url,
        check_embedding_ctx_length=False,
        model_kwargs={"encoding_format": "float"},
    )

    llm_wrapper = (
        LangchainLLMWrapper(llm, is_finished_parser=lambda _: True)
        if is_generator
        else LangchainLLMWrapper(llm)
    )
    return llm_wrapper, LangchainEmbeddingsWrapper(embeddings)


# endregion
