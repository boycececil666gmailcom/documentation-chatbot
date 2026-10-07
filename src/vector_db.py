# region Imports
import json
from typing import Any

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
_hierarchy_map: dict[str, dict] | None = None
_children_map: dict[str, list[dict]] | None = None


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


def get_hierarchy_index() -> tuple[dict[str, dict], dict[str, list[dict]]]:
    """Lazily initialize and return the RAPTOR chunk hierarchy and children index from langchain_pg_embedding."""
    global _hierarchy_map, _children_map
    if _hierarchy_map is None or _children_map is None:
        conn_str = PGVECTOR_URL.replace("+psycopg", "")
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, document, cmetadata FROM langchain_pg_embedding"
                )
                rows = cur.fetchall()

        h_map: dict[str, dict] = {}
        c_map: dict[str, list[dict]] = {}
        for row in rows:
            chunk_id = str(row[0])
            raw_doc = str(row[1])
            meta = row[2] if isinstance(row[2], dict) else json.loads(row[2])
            meta_dict = dict(meta)
            meta_dict["chunk_id"] = chunk_id
            meta_dict["document"] = raw_doc
            h_map[chunk_id] = meta_dict

            # Index by doc_id (Part 1 or earliest chunk represents the document)
            doc_id = meta_dict.get("doc_id")
            if doc_id and (meta_dict.get("chunk_index", 1) == 1 or doc_id not in h_map):
                h_map[doc_id] = meta_dict

            parent_id = meta_dict.get("parent_id")
            if parent_id:
                c_map.setdefault(parent_id, []).append(meta_dict)

        _hierarchy_map = h_map
        _children_map = c_map
        print(
            f"[VectorDB-get_hierarchy_index] Indexed {len(_hierarchy_map)} RAPTOR nodes across {len(_children_map)} branches"
        )
    return _hierarchy_map, _children_map


def fetch_document_by_id(node_id: str | None) -> dict[str, Any] | None:
    """Real-time fetch of document or chunk record by ID using memory cache from langchain_pg_embedding."""
    if not node_id:
        return None
    h_map, _ = get_hierarchy_index()
    return h_map.get(str(node_id))


def fetch_child_documents(child_ids: list[str]) -> list[dict[str, Any]]:
    """Real-time fetch of child documents by their IDs."""
    if not child_ids:
        return []
    h_map, _ = get_hierarchy_index()
    return [h_map[cid] for cid in child_ids if cid in h_map]


def get_bm25_retriever() -> BM25Retriever:
    """Lazily initialize and return the in-memory BM25Retriever from Postgres chunks enriched with RAPTOR hierarchy."""
    global _bm25_retriever
    if _bm25_retriever is None:
        conn_str = PGVECTOR_URL.replace("+psycopg", "")
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, document, cmetadata FROM langchain_pg_embedding"
                )
                rows = cur.fetchall()

        docs = []
        for row in rows:
            meta = row[2] if isinstance(row[2], dict) else json.loads(row[2])
            meta_dict = dict(meta)
            meta_dict["chunk_id"] = str(row[0])

            # Enrich BM25 searchable text with RAPTOR breadcrumb hierarchy and original text
            breadcrumb = meta_dict.get("breadcrumb", "")
            raw_content = str(row[1] or "")
            searchable_text = f"Topic: {breadcrumb}\n\n{raw_content}" if breadcrumb else raw_content

            docs.append(
                Document(
                    id=str(row[0]),
                    page_content=searchable_text,
                    metadata=meta_dict,
                )
            )
        retriever = BM25Retriever.from_documents(docs)
        retriever.k = 10
        _bm25_retriever = retriever
    return _bm25_retriever


# endregion


# region RAPTOR Expansion
def get_ancestors(parent_id: str | None, max_depth: int = 3) -> list[dict]:
    """Traverse parent_id up the tree to retrieve ancestor metadata list [parent, grandparent, ...]."""
    if not parent_id:
        return []
    h_map, _ = get_hierarchy_index()
    ancestors: list[dict] = []
    curr_id = parent_id
    depth = 0
    while curr_id and curr_id in h_map and depth < max_depth:
        p_meta = h_map[curr_id]
        p_lead = p_meta.get("document", "")
        lead_excerpt = p_lead[:300] + "..." if len(p_lead) > 300 else p_lead
        ancestors.append(
            {
                "chunk_id": curr_id,
                "title": p_meta.get("title", ""),
                "lead_content": lead_excerpt,
                "breadcrumb": p_meta.get("breadcrumb", ""),
                "raptor_layer": p_meta.get("raptor_layer", 0),
            }
        )
        curr_id = p_meta.get("parent_id")
        depth += 1
    return ancestors


def expand_parent_context(docs: list[Document]) -> list[Document]:
    """Enriches candidate chunks with real-time fetched parent document context and child subtopics before reranking."""
    if not docs:
        return []
    enriched: list[Document] = []

    for doc in docs:
        meta = dict(doc.metadata)
        doc_id = meta.get("doc_id")
        parent_id = meta.get("parent_id")

        # Real-time fetch of parent document and ancestor section by ID
        parent_doc = fetch_document_by_id(doc_id)
        ancestor_doc = fetch_document_by_id(parent_id) if parent_id else None

        preamble_parts: list[str] = []

        # Hierarchy Path (Breadcrumb)
        bc = (
            (parent_doc.get("breadcrumb") if parent_doc else None)
            or meta.get("breadcrumb")
            or meta.get("title", "")
        )
        if bc:
            preamble_parts.append(f"[Path]: {bc}")

        # Parent Document Overview
        if parent_doc:
            lead_raw = parent_doc.get("lead_content") or parent_doc.get("document", "")
            lead = lead_raw[:350].strip()
            if lead and not doc.page_content.startswith(lead[:60]):
                preamble_parts.append(f"[Parent Overview]: {lead}")

            # Child Subtopics (if parent document has child nodes)
            child_ids = parent_doc.get("child_ids", [])
            if child_ids:
                children = fetch_child_documents(child_ids[:3])
                child_titles = [c.get("title", "") for c in children if c.get("title")]
                if child_titles:
                    preamble_parts.append(f"[Related Subtopics]: {', '.join(child_titles)}")

        elif ancestor_doc:
            lead_raw = ancestor_doc.get("lead_content") or ancestor_doc.get("document", "")
            lead = lead_raw[:350].strip()
            if lead and not doc.page_content.startswith(lead[:60]):
                preamble_parts.append(f"[Parent Overview]: {lead}")

        if preamble_parts:
            enriched_content = f"{chr(10).join(preamble_parts)}\n\n[Section Content]:\n{doc.page_content}"
        else:
            enriched_content = doc.page_content

        enriched.append(
            Document(
                page_content=enriched_content,
                metadata=meta,
                id=doc.id if hasattr(doc, "id") else None,
            )
        )

    return enriched


def expand_raptor_context(docs: list[Document]) -> list[Document]:
    """Enriches Document objects with RAPTOR ancestor lead preambles and hierarchical context."""
    if not docs:
        return []
    h_map, c_map = get_hierarchy_index()

    for doc in docs:
        meta = doc.metadata
        parent_id = meta.get("parent_id")
        chunk_id = meta.get("chunk_id") or getattr(doc, "id", None)

        # 1. Ancestor chain expansion (Leaf / Child -> Parent Overview)
        if parent_id and "raptor_ancestors" not in meta:
            meta["raptor_ancestors"] = get_ancestors(parent_id)

        # 2. Children expansion (if current node is a high-level node Layer <= 1)
        layer = meta.get("raptor_layer", 2)
        if layer < 2 and chunk_id and "raptor_children_topics" not in meta:
            children = c_map.get(str(chunk_id), [])
            if children:
                meta["raptor_children_topics"] = [
                    {"title": c.get("title", ""), "breadcrumb": c.get("breadcrumb", "")}
                    for c in children[:5]
                ]

    return docs


def expand_raptor_candidates(
    docs: list[Document], max_children_per_parent: int = 2
) -> list[Document]:
    """Top-down expansion: If summary chunks (Layer <= 1) are retrieved, fetch representative child chunks."""
    if not docs:
        return []
    h_map, c_map = get_hierarchy_index()
    expanded: list[Document] = list(docs)
    seen_breadcrumbs: set[str] = {
        str(d.metadata.get("breadcrumb") or d.metadata.get("title") or "") for d in docs
    }

    for doc in docs:
        layer = doc.metadata.get("raptor_layer", 2)
        chunk_id = doc.metadata.get("chunk_id") or getattr(doc, "id", None)
        if layer <= 1 and chunk_id and str(chunk_id) in c_map:
            children = c_map[str(chunk_id)][:max_children_per_parent]
            for child in children:
                bc = str(child.get("breadcrumb") or child.get("title") or "")
                if bc and bc not in seen_breadcrumbs:
                    seen_breadcrumbs.add(bc)
                    child_doc = Document(
                        id=child.get("chunk_id"),
                        page_content=str(child.get("document", "")),
                        metadata=child,
                    )
                    expanded.append(child_doc)

    return expanded


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
        content = doc.page_content
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
            doc_key = str(
                d.metadata.get("breadcrumb")
                or d.metadata.get("title")
                or d.page_content[:40]
            )
            if doc_key not in seen_keys:
                seen_keys.add(doc_key)
                merged.append(d)
        if i < len(sparse_docs):
            s = sparse_docs[i]
            doc_key = str(
                s.metadata.get("breadcrumb")
                or s.metadata.get("title")
                or s.page_content[:40]
            )
            if doc_key not in seen_keys:
                seen_keys.add(doc_key)
                merged.append(s)

    return merged[:top_k]


# endregion
