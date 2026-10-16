# region Imports
from typing import Any

from flashrank import Ranker
from langchain_community.document_compressors.flashrank_rerank import (
    FlashrankRerank,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from .config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_EMBED_MODEL,
    OPENROUTER_MODEL,
    OPENROUTER_PROVIDER_IGNORE,
    OPENROUTER_PROVIDER_SORT,
    OPENROUTER_TEMPERATURE,
    RERANKER_TOP_N,
)

# endregion


# region Types
class LazyReranker:
    """Lazy cross-encoder reranker wrapper to eliminate import and model loading latency."""

    def __init__(self, top_n: int = RERANKER_TOP_N) -> None:
        self._top_n = top_n
        self._instance: FlashrankRerank | None = None

    def _get_instance(self) -> FlashrankRerank:
        if self._instance is None:
            self._instance = FlashrankRerank(client=Ranker(), top_n=self._top_n)
        return self._instance

    def compress_documents(self, documents: Any, query: str, **kwargs: Any) -> Any:
        return self._get_instance().compress_documents(documents, query, **kwargs)


# endregion


# region Scaffolding
_provider_config = {"allow_fallbacks": True}
if OPENROUTER_PROVIDER_SORT:
    _provider_config["sort"] = OPENROUTER_PROVIDER_SORT
if OPENROUTER_PROVIDER_IGNORE:
    _provider_config["ignore"] = OPENROUTER_PROVIDER_IGNORE

_extra_body = {"provider": _provider_config} if _provider_config else {}
# endregion


# region Clients
# Primary generative LLM for planning, worker subagents, synthesis, and critique
llm = ChatOpenAI(
    model=OPENROUTER_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    temperature=OPENROUTER_TEMPERATURE,
    extra_body=_extra_body if _extra_body else None,
)

# Shared embeddings client for PGVector dense vector search
embeddings = OpenAIEmbeddings(
    model=OPENROUTER_EMBED_MODEL,
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL,
    check_embedding_ctx_length=False,
    model_kwargs={"encoding_format": "float"},
)

# Shared FlashRank reranker instance
reranker = LazyReranker(top_n=RERANKER_TOP_N)
# endregion
