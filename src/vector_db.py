# region Vector Stores
from functools import lru_cache

from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode

from .config import QDRANT_API_KEY, QDRANT_URL
from .llm_client import embeddings


@lru_cache(maxsize=1)
def get_sparse_embeddings() -> FastEmbedSparse | None:
    try:
        return FastEmbedSparse(model_name="Qdrant/bm25")
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_vector_store(collection_name: str = "raptor_chunks") -> QdrantVectorStore:
    return QdrantVectorStore.from_existing_collection(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=collection_name,
        content_payload_key="small",
        embedding=embeddings,
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
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
