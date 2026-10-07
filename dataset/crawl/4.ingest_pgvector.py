# region Ingestion
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector

from src.config import OPENROUTER_EMBED_MODEL, PGVECTOR_COLLECTION_NAME, PGVECTOR_URL
from src.llm_client import embeddings

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(dotenv_path=ROOT_DIR / ".env")

CONFIG_PATH = CURRENT_DIR / "crawler_config.json"
with open(CONFIG_PATH, encoding="utf-8") as f:
    config = json.load(f)

ingest_cfg = config.get("ingest", {})
INPUT_FILE = CURRENT_DIR / ingest_cfg.get("input_path", "3.split_chunks.json")
OUTPUT_FILE = CURRENT_DIR / ingest_cfg.get("output_path", "4.ingest_pgvector.json")


def query_database_status() -> dict[str, Any]:
    """Inspects PostgreSQL table to check row count and layer distribution."""
    conn_str = PGVECTOR_URL.replace("+psycopg", "")
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT count(*) FROM langchain_pg_embedding
                WHERE collection_id IN (
                    SELECT uuid FROM langchain_pg_collection WHERE name = %s
                )
                """,
                (PGVECTOR_COLLECTION_NAME,),
            )
            total_db_rows = cur.fetchone()[0]

            cur.execute(
                """
                SELECT cmetadata->>'raptor_layer', count(*)
                FROM langchain_pg_embedding
                WHERE collection_id IN (
                    SELECT uuid FROM langchain_pg_collection WHERE name = %s
                )
                GROUP BY cmetadata->>'raptor_layer'
                ORDER BY 1
                """,
                (PGVECTOR_COLLECTION_NAME,),
            )
            rows = cur.fetchall()
            layers = {f"layer_{r[0]}": r[1] for r in rows if r[0] is not None}

            cur.execute(
                """
                SELECT document FROM langchain_pg_embedding
                WHERE collection_id IN (
                    SELECT uuid FROM langchain_pg_collection WHERE name = %s
                )
                LIMIT 1
                """,
                (PGVECTOR_COLLECTION_NAME,),
            )
            sample_doc = cur.fetchone()
            preview = sample_doc[0][:100] if sample_doc else ""

    return {
        "total_db_rows": total_db_rows,
        "layers": layers,
        "sample_preview": preview,
    }


def wipe_database(collection_only: bool = True) -> int:
    """Explicitly deletes stale records from PGVector database.

    If collection_only is True, purges rows belonging to PGVECTOR_COLLECTION_NAME.
    If False, truncates langchain_pg_embedding completely.
    """
    conn_str = PGVECTOR_URL.replace("+psycopg", "")
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            if collection_only:
                cur.execute(
                    """
                    DELETE FROM langchain_pg_embedding
                    WHERE collection_id IN (
                        SELECT uuid FROM langchain_pg_collection WHERE name = %s
                    )
                    """,
                    (PGVECTOR_COLLECTION_NAME,),
                )
            else:
                cur.execute("TRUNCATE TABLE langchain_pg_embedding CASCADE")
            deleted_count = cur.rowcount
            conn.commit()
    print(
        f"[Ingestion-wipe_database] Purged {deleted_count} stale rows for collection '{PGVECTOR_COLLECTION_NAME}'."
    )
    return deleted_count


def batch_ingest(
    chunks_data: list[dict[str, Any]],
    batch_size: int = 64,
    delay: float = 3.2,
    wipe: bool = True,
) -> dict[str, Any]:
    """Ingests chunks into PGVector where page_content is stored in document column and cmetadata is clean."""
    if wipe:
        wipe_database(collection_only=True)

    total = len(chunks_data)
    start_time = time.time()
    total_batches = math.ceil(total / batch_size)
    print(
        f"[Ingestion-batch_ingest] Target: collection '{PGVECTOR_COLLECTION_NAME}' | Total Chunks: {total} | Batches: {total_batches}"
    )

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=PGVECTOR_COLLECTION_NAME,
        connection=PGVECTOR_URL,
        pre_delete_collection=False,
        use_jsonb=True,
    )

    ingested_count = 0
    for start in range(0, total, batch_size):
        batch = chunks_data[start : start + batch_size]
        batch_docs = []
        batch_ids = []
        seen_batch_ids: set[str] = set()
        for d in batch:
            doc_id = str(d["id"])
            if doc_id in seen_batch_ids:
                continue
            seen_batch_ids.add(doc_id)

            text = (d.get("content") or d.get("metadata", {}).get("big", "")).strip()[:12000]
            if not text:
                continue
            batch_docs.append(
                Document(
                    page_content=text,
                    metadata={k: v for k, v in d.get("metadata", {}).items() if k != "big"},
                )
            )
            batch_ids.append(doc_id)

        if not batch_docs:
            continue

        max_retries = 5
        for attempt in range(max_retries):
            try:
                vector_store.add_documents(documents=batch_docs, ids=batch_ids)
                ingested_count += len(batch_docs)
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_sec = 15 * (attempt + 1)
                    print(
                        f"[Ingestion-batch_ingest] Rate limit (429). Backing off {wait_sec}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(wait_sec)
                else:
                    raise

        print(
            f"[Ingestion-batch_ingest] Ingested {min(start + batch_size, total)}/{total} chunks..."
        )
        time.sleep(delay)

    elapsed_sec = round(time.time() - start_time, 2)
    db_status = query_database_status()

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "collection_name": PGVECTOR_COLLECTION_NAME,
        "status": "synchronized" if db_status["total_db_rows"] == total else "outdated_mismatch",
        "source_file": INPUT_FILE.name,
        "target_search_field": "document (canonical content column)",
        "embedding_model": OPENROUTER_EMBED_MODEL,
        "total_source_chunks": total,
        "ingested_chunks": ingested_count,
        "database_rows": db_status["total_db_rows"],
        "batch_size": batch_size,
        "total_batches": total_batches,
        "layer_distribution": db_status["layers"],
        "elapsed_seconds": elapsed_sec,
        "integrity_check": "passed" if db_status["total_db_rows"] == total else "mismatch",
    }
    return summary


def main() -> None:
    """Purges stale collection records and ingests all chunks into PGVector."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input chunks file not found: {INPUT_FILE}")

    with open(INPUT_FILE, encoding="utf-8") as f:
        records: list[dict[str, Any]] = json.load(f)

    # Clean up VDB and ingest new chunks with metadata into PGVector
    summary = batch_ingest(
        chunks_data=records,
        batch_size=ingest_cfg.get("batch_size", 64),
        delay=ingest_cfg.get("rate_limit_delay", 3.2),
        wipe=True,
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"[Ingestion-main] Database rows: {summary['database_rows']} | Source chunks: {summary['total_source_chunks']} | Integrity: {summary['integrity_check']}"
    )
    print(f"[Ingestion-main] Saved ingestion report JSON to '{OUTPUT_FILE.name}'")


if __name__ == "__main__":
    main()
# endregion
