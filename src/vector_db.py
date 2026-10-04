# region Imports
import json
from typing import cast

import psycopg
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector

from .config import PGVECTOR_COLLECTION_NAME, PGVECTOR_URL
from .llm_client import embeddings

# endregion


# region Vector & BM25 Stores
_vector_store: PGVector | None = None
_bm25_retriever: BM25Retriever | None = None


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


def get_bm25_retriever() -> BM25Retriever:
    """Lazily initialize and return the in-memory BM25Retriever from Postgres chunks."""
    global _bm25_retriever
    if _bm25_retriever is None:
        conn_str = PGVECTOR_URL.replace("+psycopg", "")
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT document, cmetadata FROM langchain_pg_embedding")
                rows = cur.fetchall()

        docs = [
            Document(
                page_content=str(meta.get("big") or row[0]),
                metadata=cast(dict, meta),
            )
            for row in rows
            for meta in [row[1] if isinstance(row[1], dict) else json.loads(row[1])]
        ]
        retriever = BM25Retriever.from_documents(docs)
        retriever.k = 10
        _bm25_retriever = retriever
    return _bm25_retriever


# endregion


# region Retrieval Strategies
def retrieve_collapsed_tree(
    query: str,
    top_k: int = 10,
    max_tokens: int = 4000,
) -> list[Document]:
    """Retrieves chunks flatly across the entire collapsed tree based on similarity up to a token limit."""
    store = get_vector_store()
    candidate_docs = store.similarity_search(query=query, k=top_k)

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


def retrieve_bm25(query: str, top_k: int = 10) -> list[Document]:
    """Retrieves document chunks using pure BM25 sparse keyword matching."""
    retriever = get_bm25_retriever()
    retriever.k = top_k
    return retriever.invoke(query)


def retrieve_hybrid(
    dense_query: str,
    sparse_query: str,
    top_k: int = 10,
) -> list[Document]:
    """Retrieves and merges document chunks from both dense semantic and BM25 sparse search."""
    dense_docs = retrieve_collapsed_tree(query=dense_query, top_k=top_k)
    sparse_docs = retrieve_bm25(query=sparse_query, top_k=top_k)

    seen_keys: set[str] = set()
    merged: list[Document] = []

    max_len = max(len(dense_docs), len(sparse_docs))
    for i in range(max_len):
        if i < len(dense_docs):
            d = dense_docs[i]
            doc_key = str(d.metadata.get("breadcrumb") or d.metadata.get("title") or d.page_content[:40])
            if doc_key not in seen_keys:
                seen_keys.add(doc_key)
                merged.append(d)
        if i < len(sparse_docs):
            s = sparse_docs[i]
            doc_key = str(s.metadata.get("breadcrumb") or s.metadata.get("title") or s.page_content[:40])
            if doc_key not in seen_keys:
                seen_keys.add(doc_key)
                merged.append(s)

    return merged[:top_k]


# endregion
