# region Imports
import re
from typing import Any

import psycopg
from langchain_core.tools import tool

from ..config import PGVECTOR_URL

# endregion


# region Types
# Standard Python return types (str, list, dict) utilized by LangChain tools
# endregion


# region Scaffolding
def _check_readonly_sql(sql_query: str) -> str:
    """Enforces strict read-only execution on SQL queries."""
    clean_sql = sql_query.strip()
    upper_sql = clean_sql.upper()

    if not upper_sql.startswith("SELECT"):
        raise ValueError("Security violation: Only SELECT queries are permitted.")

    forbidden_keywords = [
        "DROP",
        "DELETE",
        "UPDATE",
        "INSERT",
        "ALTER",
        "CREATE",
        "TRUNCATE",
        "GRANT",
        "REVOKE",
    ]
    for forbidden in forbidden_keywords:
        if re.search(rf"\b{forbidden}\b", upper_sql):
            raise ValueError(f"Security violation: Keyword '{forbidden}' is forbidden.")

    return clean_sql


# endregion


# region Tools
@tool
def search_documentation(query: str) -> str:
    """Searches the local Kanzi documentation using hybrid PGVector dense retrieval, BM25 keyword matching, and FlashRank neural reranking."""
    from ..llm_client import reranker
    from ..vector_db import (
        expand_parent_context,
        expand_raptor_candidates,
        get_bm25_retriever,
        retrieve_collapsed_tree,
    )

    dense_docs = []
    try:
        dense_docs = retrieve_collapsed_tree(query=query, top_k=6)
    except Exception as exc:
        print(f"[Tools-search_documentation] Dense search notice: {exc}")

    sparse_docs = []
    try:
        bm25_retriever = get_bm25_retriever()
        sparse_docs = bm25_retriever.invoke(query)
    except Exception as exc:
        print(f"[Tools-search_documentation] BM25 search notice: {exc}")

    seen: set[str] = set()
    candidates = []
    for doc in dense_docs + sparse_docs:
        doc_id = str(
            doc.metadata.get("chunk_id")
            or doc.metadata.get("doc_id")
            or doc.page_content[:60]
        )
        if doc_id not in seen:
            seen.add(doc_id)
            candidates.append(doc)

    if not candidates:
        return "No matching local documentation sections found."

    expanded = expand_raptor_candidates(candidates)
    enriched = expand_parent_context(expanded)
    ranked_docs = reranker.compress_documents(enriched, query)

    sections: list[str] = []
    for d in ranked_docs[:6]:
        title = (
            d.metadata.get("title")
            or d.metadata.get("breadcrumb")
            or "Documentation Passage"
        )
        breadcrumb = d.metadata.get("breadcrumb", "")
        content = d.page_content[:1200]
        sections.append(f"### {title} [{breadcrumb}]\n{content}")

    return (
        "\n\n".join(sections)
        if sections
        else "No matching local documentation sections found."
    )


@tool
def query_database(sql_query: str) -> str:
    """Executes a safe, read-only SQL SELECT query against the PostgreSQL metadata and chunk tables."""
    clean_sql = _check_readonly_sql(sql_query)
    conn_str = PGVECTOR_URL.replace("+psycopg", "")

    with psycopg.connect(conn_str, connect_timeout=3) as conn:
        with conn.cursor() as cur:
            cur.execute(clean_sql)
            rows = cur.fetchmany(5)
            colnames = [desc[0] for desc in cur.description] if cur.description else []
            records = [dict(zip(colnames, row, strict=False)) for row in rows]

    if not records:
        return "Query executed successfully. 0 records found."

    formatted: list[str] = []
    for r in records:
        formatted.append(str(r))
    return "\n".join(formatted)


@tool
def get_database_schema(table_name: str = "langchain_pg_embedding") -> str:
    """Returns column names and data types for the specified PostgreSQL table."""
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

    columns = [f"{row[0]} ({row[1]}, nullable={row[2]})" for row in rows]
    return f"Table '{table_name}' columns:\n" + "\n".join(f"- {col}" for col in columns)


@tool
def search_web(query: str) -> str:
    """Performs live web search for external documentation, community discussions, or API updates."""
    try:
        from ddgs import DDGS

        with DDGS() as ddgs_client:
            results = list(ddgs_client.text(query, max_results=3))
            if not results:
                return "No web search results found."

            entries = [
                f"Title: {r.get('title', '')}\nURL: {r.get('href', '')}\nSnippet: {r.get('body', '')}"
                for r in results
            ]
            return "\n\n".join(entries)
    except Exception as exc:
        return f"Web search service unavailable ({exc}). Fallback URL: https://docs.kanzi.com"


# endregion


# region Core
ALL_TOOLS = [search_documentation, query_database, get_database_schema, search_web]


def get_all_tools() -> list[Any]:
    """Returns the primary registered tool suite for LLM agent binding."""
    return ALL_TOOLS


# endregion
