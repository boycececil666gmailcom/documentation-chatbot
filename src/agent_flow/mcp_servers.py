# region Imports
import json
import re
from pathlib import Path
from typing import Any

import httpx
import psycopg

from ..config import PGVECTOR_URL

# endregion


# region Types
class MCPServerResponse:
    """Standardized response envelope from MCP tool calls."""

    def __init__(self, success: bool, data: Any, message: str = "") -> None:
        self.success = success
        self.data = data
        self.message = message

    def to_dict(self) -> dict[str, Any]:
        return {"success": self.success, "data": self.data, "message": self.message}


# endregion


# region Scaffolding
_DATASET_CACHE: list[dict[str, Any]] | None = None


def _resolve_dataset_path() -> Path | None:
    """Resolves documentation dataset path across local dev and Docker container environments."""
    candidates = [
        Path("dataset/crawl/3.split_chunks.json"),
        Path("/app/dataset/crawl/3.split_chunks.json"),
        Path("dataset/crawl/2.structure_hierarchy.json"),
        Path("/app/dataset/crawl/2.structure_hierarchy.json"),
        Path(__file__).resolve().parent.parent.parent / "dataset" / "crawl" / "3.split_chunks.json",
        Path(__file__).resolve().parent.parent.parent / "dataset" / "crawl" / "2.structure_hierarchy.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _load_local_documents() -> list[dict[str, Any]]:
    """Loads and caches local hierarchical documentation dataset."""
    global _DATASET_CACHE
    if _DATASET_CACHE is None:
        path = _resolve_dataset_path()
        if path and path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    _DATASET_CACHE = json.load(f)
            except Exception as exc:
                print(f"[MCP-LocalFiles] Failed to load dataset from {path}: {exc}")
                _DATASET_CACHE = []
        else:
            _DATASET_CACHE = []
    return _DATASET_CACHE


# endregion


# region Files
class LocalFilesServer:
    """MCP Server providing tools for inspecting and reading local documentation files."""

    @staticmethod
    def search_local_documents(query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Searches local structured documentation files by keyword matching across titles, breadcrumbs, and content."""
        docs = _load_local_documents()
        if not docs:
            return []

        stop_words = {
            "what", "is", "and", "how", "are", "the", "for", "with", "this",
            "that", "from", "search", "documentation", "kanzi", "user", "using",
        }
        raw_tokens = re.findall(r"[a-zA-Z0-9_]+", query)
        tokens = [t.lower() for t in raw_tokens if len(t) >= 2 and t.lower() not in stop_words]
        if not tokens:
            tokens = [t.lower() for t in raw_tokens if len(t) >= 2] or [query.lower().strip()]

        scored_docs: list[tuple[float, dict[str, Any]]] = []
        for doc in docs:
            meta = doc.get("metadata", {})
            title = str(doc.get("title") or meta.get("title") or "").lower()
            breadcrumb = str(meta.get("breadcrumb", "")).lower()
            content = str(doc.get("content", ""))
            content_lower = content.lower()

            score = 0.0
            for token in tokens:
                if token in title:
                    score += 8.0
                if token in breadcrumb:
                    score += 5.0
                count = content_lower.count(token)
                if count > 0:
                    score += min(count * 1.0, 10.0)

            if score > 0:
                doc_title = doc.get("title") or meta.get("title") or meta.get("breadcrumb") or "Documentation Section"
                scored_docs.append((
                    score,
                    {
                        "id": str(doc.get("id")),
                        "title": doc_title,
                        "breadcrumb": meta.get("breadcrumb", doc_title),
                        "excerpt": content[:450].strip(),
                        "url": meta.get("url") or doc.get("url"),
                        "score": round(score, 2),
                    },
                ))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_docs[:max_results]]

    @staticmethod
    def read_document_section(doc_id: str) -> dict[str, Any] | None:
        """Reads complete content and hierarchy metadata for a specific document ID."""
        docs = _load_local_documents()
        for doc in docs:
            if str(doc.get("id")) == str(doc_id):
                meta = doc.get("metadata", {})
                title = doc.get("title") or meta.get("title") or meta.get("breadcrumb") or "Documentation Section"
                return {
                    "id": str(doc.get("id")),
                    "title": title,
                    "breadcrumb": meta.get("breadcrumb", title),
                    "url": doc.get("url") or meta.get("url"),
                    "content": doc.get("content"),
                    "parent_id": meta.get("parent_id"),
                    "child_ids": meta.get("child_ids", []),
                }
        return None

    @staticmethod
    def list_topics(limit: int = 15) -> list[str]:
        """Returns sample of top-level technical topics and breadcrumbs from local documentation."""
        docs = _load_local_documents()
        topics: list[str] = []
        for doc in docs:
            bc = doc.get("metadata", {}).get("breadcrumb") or doc.get("title")
            if bc and bc not in topics:
                topics.append(bc)
            if len(topics) >= limit:
                break
        return topics


# endregion


# region Database
class DatabaseServer:
    """MCP Server providing tools for inspecting relational database records and schemas."""

    @staticmethod
    def execute_query(sql_query: str, max_rows: int = 10) -> list[dict[str, Any]]:
        """Executes safe, read-only SQL queries against PostgreSQL documentation tables."""
        clean_sql = sql_query.strip()
        upper_sql = clean_sql.upper()

        # Read-only enforcement
        if not upper_sql.startswith("SELECT"):
            raise ValueError("Security violation: Only SELECT queries are permitted.")
        for forbidden in ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE", "TRUNCATE", "GRANT"]:
            if re.search(rf"\b{forbidden}\b", upper_sql):
                raise ValueError(f"Security violation: Keyword '{forbidden}' is forbidden.")

        conn_str = PGVECTOR_URL.replace("+psycopg", "")
        with psycopg.connect(conn_str, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute(clean_sql)
                rows = cur.fetchmany(max_rows)
                colnames = [desc[0] for desc in cur.description] if cur.description else []
                return [dict(zip(colnames, row, strict=False)) for row in rows]

    @staticmethod
    def get_table_schema(table_name: str = "langchain_pg_embedding") -> list[dict[str, str]]:
        """Returns column names and data types for the specified database table."""
        conn_str = PGVECTOR_URL.replace("+psycopg", "")
        sql = """
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            ORDER BY ordinal_position;
        """
        with psycopg.connect(conn_str, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (table_name,))
                rows = cur.fetchall()
                return [
                    {"column": row[0], "type": row[1], "nullable": row[2]}
                    for row in rows
                ]


# endregion


# region Search
class SearchAPIServer:
    """MCP Server providing tools for real-time web search and external technical references."""

    @staticmethod
    def search_web(query: str, max_results: int = 5) -> list[dict[str, Any]]:
        """Performs live web search using modern DDGS engine with resilience fallbacks."""
        try:
            from ddgs import DDGS

            with DDGS() as ddgs_client:
                results = list(ddgs_client.text(query, max_results=max_results))
                return [
                    {
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r.get("href", ""),
                    }
                    for r in results
                ]
        except Exception as exc:
            print(f"[MCP-SearchAPI] Web search warning: {exc}")
            return [
                {
                    "title": f"Search fallback for: {query}",
                    "snippet": "External web search service temporarily degraded or offline.",
                    "url": "https://docs.kanzi.com",
                }
            ]


# endregion


# region External
class ExternalAPIServer:
    """MCP Server providing tools for querying external REST APIs and cloud endpoints."""

    @staticmethod
    def fetch_api_endpoint(url: str, timeout: float = 8.0) -> dict[str, Any]:
        """Fetches data from an external documentation endpoint or REST service."""
        try:
            with httpx.Client(timeout=timeout) as client:
                res = client.get(url)
                return {
                    "status_code": res.status_code,
                    "content_type": res.headers.get("content-type", ""),
                    "text_excerpt": res.text[:500],
                }
        except Exception as exc:
            return {"error": str(exc), "status_code": 500}


# endregion


# region Knowledge
class KnowledgeStoreServer:
    """MCP Server exposing hybrid PGVector dense retrieval, BM25 keyword matching, and FlashRank neural reranking."""

    @staticmethod
    def hybrid_search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Executes unified hybrid retrieval: fetches dense vector chunks and sparse BM25 matches, enriches with parent hierarchy, and neural reranks with FlashRank."""
        from ..llm_client import reranker
        from ..vector_db import (
            expand_parent_context,
            expand_raptor_candidates,
            get_bm25_retriever,
            retrieve_collapsed_tree,
        )

        dense_docs = []
        try:
            dense_docs = retrieve_collapsed_tree(query=query, top_k=top_k)
        except Exception as exc:
            print(f"[KnowledgeStoreServer] Dense search notice: {exc}")

        sparse_docs = []
        try:
            bm25_retriever = get_bm25_retriever()
            sparse_docs = bm25_retriever.invoke(query)
        except Exception as exc:
            print(f"[KnowledgeStoreServer] BM25 search notice: {exc}")

        # Merge and deduplicate candidates across dense and sparse retrieval
        seen: set[str] = set()
        candidates = []
        for doc in dense_docs + sparse_docs:
            doc_id = str(doc.metadata.get("chunk_id") or doc.metadata.get("doc_id") or doc.page_content[:60])
            if doc_id not in seen:
                seen.add(doc_id)
                candidates.append(doc)

        if not candidates:
            return []

        # Hierarchy expansion (RAPTOR child links & parent document lead overview)
        expanded = expand_raptor_candidates(candidates)
        enriched = expand_parent_context(expanded)

        # FlashRank Cross-Encoder reranking
        ranked_docs = reranker.compress_documents(enriched, query)

        return [
            {
                "id": str(d.metadata.get("chunk_id") or d.metadata.get("doc_id") or "chunk"),
                "title": d.metadata.get("title") or d.metadata.get("breadcrumb") or "Documentation Passage",
                "breadcrumb": d.metadata.get("breadcrumb", ""),
                "content": d.page_content,
                "url": d.metadata.get("url", ""),
            }
            for d in ranked_docs
        ]


# endregion


# region Core
def call_mcp_tool(server_name: str, tool_name: str, **kwargs: Any) -> Any:
    """Primary declarative MCP dispatcher delegating calls to registered servers."""
    if server_name == "knowledge_store":
        server = KnowledgeStoreServer
    elif server_name == "local_files":
        server = LocalFilesServer
    elif server_name == "database":
        server = DatabaseServer
    elif server_name == "search_api":
        server = SearchAPIServer
    elif server_name == "external_api":
        server = ExternalAPIServer
    else:
        raise ValueError(f"Unknown MCP Server '{server_name}'")

    if not hasattr(server, tool_name):
        raise AttributeError(f"Server '{server_name}' does not implement tool '{tool_name}'")

    method = getattr(server, tool_name)
    return method(**kwargs)


# endregion
