# region Retrieval Tool
from langchain_core.tools import tool

from . import vector_db as db
from .llm_client import reranker


@tool
def retrieve_VDB(query: str) -> str:
    """Retrieve semantically relevant document chunks using Collapsed Tree retrieval with FlashRank reranking."""
    docs = db.retrieve_collapsed_tree(query=query, top_k=10, max_tokens=4000)
    ranked_docs = reranker.compress_documents(docs, query)

    formatted_chunks = [
        f"[{doc.metadata.get('breadcrumb') or doc.metadata.get('title', '')}] (Score: {doc.metadata.get('relevance_score', 0.0):.3f})\n"
        f"{doc.metadata.get('big') or doc.page_content}"
        for doc in ranked_docs
    ]

    return (
        "=== VECTOR DATABASE CONTEXT ===\n" + "\n\n".join(formatted_chunks)
        if formatted_chunks
        else "No matching vector documents found."
    )


# endregion
