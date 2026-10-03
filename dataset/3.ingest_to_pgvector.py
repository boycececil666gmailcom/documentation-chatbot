# region Imports
import json
import os
import sys
import time
from pathlib import Path

_CURRENT_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _CURRENT_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector

from src.llm_client import embeddings

# endregion

# region Configuration
load_dotenv(dotenv_path=_ROOT_DIR / ".env")

PGVECTOR_URL = os.getenv(
    "PGVECTOR_URL",
    "postgresql+psycopg://postgres:postgrespassword123@localhost:5432/documentation_chatbot",
)
PGVECTOR_COLLECTION_NAME = os.getenv("PGVECTOR_COLLECTION_NAME", "raptor_chunks")

BATCH_SIZE = 64
INPUT_JSON_PATH = _CURRENT_DIR / "2.raptor_chunks.json"
# endregion


# region Ingestion Logic
def ingest_collapsed_tree(
    chunks: list[dict],
    collection_name: str = PGVECTOR_COLLECTION_NAME,
) -> None:
    """Wipes and batch-ingests all chunks flatly into a single unified PGVector collection ('raptor_chunks') for Collapsed Tree retrieval."""
    total = len(chunks)
    if total == 0:
        return

    print(f"\n[Ingestion] Collapsed Tree: '{collection_name}' ({total} total chunks)")

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=collection_name,
        connection=PGVECTOR_URL,
        pre_delete_collection=True,
        use_jsonb=True,
    )

    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        batch_docs = [
            Document(
                page_content=d.get("metadata", {}).get("summary")
                or d.get("metadata", {}).get("title")
                or d.get("metadata", {}).get("big", "document"),
                metadata=d.get("metadata", {}),
            )
            for d in batch
        ]
        batch_ids = [str(d["id"]) for d in batch]
        max_retries = 5
        for attempt in range(max_retries):
            try:
                vector_store.add_documents(documents=batch_docs, ids=batch_ids)
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait_sec = 15 * (attempt + 1)
                    print(
                        f"[Ingestion] Rate limit (429) encountered. Backing off {wait_sec}s (attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(wait_sec)
                else:
                    raise

        print(
            f"[Ingestion] Processed {min(start + BATCH_SIZE, total)}/{total} chunks..."
        )
        time.sleep(3.2)  # Respect OpenRouter free-tier rate limit (<= 20 req/min)

    print(
        f"[Ingestion] Successfully ingested {total} chunks into PGVector collection '{collection_name}'."
    )


def ingest_chunks_to_pgvector() -> None:
    """Loads 2.raptor_chunks.json and ingests all chunks into PGVector collection for Collapsed Tree."""
    if not INPUT_JSON_PATH.exists():
        raise FileNotFoundError(
            f"Input file '{INPUT_JSON_PATH.name}' not found. Run 2.raptor_tree_pipeline.py first."
        )

    print(f"[Ingestion] Connecting to PGVector at: {PGVECTOR_URL}")

    with open(INPUT_JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    ingest_collapsed_tree(data)


if __name__ == "__main__":
    ingest_chunks_to_pgvector()
# endregion
