# region Vector Stores
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector

from .config import PGVECTOR_COLLECTION_NAME, PGVECTOR_URL
from .llm_client import embeddings

_vector_store: PGVector | None = None


def get_vector_store() -> PGVector:
    """Lazily initialize and return the singleton PGVector instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = PGVector(
            embeddings=embeddings,
            collection_name=PGVECTOR_COLLECTION_NAME,
            connection=PGVECTOR_URL,
            use_jsonb=True,
        )
    return _vector_store


# endregion


# region Collapsed Tree Retrieval
def retrieve_collapsed_tree(
    query: str,
    top_k: int = 10,
    max_tokens: int = 4000,
) -> list[Document]:
    """Retrieves chunks flatly across the entire collapsed tree based on similarity up to a token limit."""
    store = get_vector_store()
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
