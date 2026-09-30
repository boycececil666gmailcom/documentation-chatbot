# region Vector Stores
from functools import lru_cache

from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector

from .config import PGVECTOR_URL
from .llm_client import embeddings


@lru_cache(maxsize=1)
def get_vector_store(collection_name: str = "raptor_chunks") -> PGVector:
    return PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=PGVECTOR_URL,
        use_jsonb=True,
    )


# endregion


# region Collapsed Tree Retrieval
def retrieve_collapsed_tree(
    query: str,
    top_k: int = 10,
    max_tokens: int = 4000,
) -> list[Document]:
    """Retrieves chunks flatly across the entire collapsed tree based on similarity up to a token limit."""
    store = get_vector_store("raptor_chunks")
    candidate_docs = store.similarity_search(query=query, k=top_k)

    # Accumulate chunks up to the token budget (approx 4 chars/token)
    selected_docs: list[Document] = []
    current_tokens = 0

    for doc in candidate_docs:
        content = doc.metadata.get("big") or doc.page_content
        approx_tokens = max(1, len(content) // 4)
        if current_tokens + approx_tokens > max_tokens and selected_docs:
            break
        selected_docs.append(doc)
        current_tokens += approx_tokens

    return selected_docs


# endregion
