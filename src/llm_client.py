# region LLM Clients
from flashrank import Ranker
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_EMBED_MODEL,
    OPENROUTER_MODEL,
    OPENROUTER_PROVIDER_IGNORE,
    OPENROUTER_PROVIDER_SORT,
    OPENROUTER_TEMPERATURE,
)

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

# LLM instance configured with lower temperature for HyDE passage generation
hyde_llm = ChatOpenAI(
    model=OPENROUTER_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    temperature=0.0,
    extra_body=_extra_body if _extra_body else None,
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
