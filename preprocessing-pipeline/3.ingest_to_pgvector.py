# region Imports
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector

from llm_client import embeddings

# endregion

# region Configuration
_CURRENT_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _CURRENT_DIR.parent
load_dotenv(dotenv_path=_ROOT_DIR / ".env")

_user = os.getenv("POSTGRES_USER", "postgres")
_pass = os.getenv("POSTGRES_PASSWORD", "postgrespassword123")
_host = os.getenv("POSTGRES_HOST", "localhost")
_port = os.getenv("POSTGRES_PORT", "5432")
_db = os.getenv("POSTGRES_DB", "documentation_chatbot")

_default_url = f"postgresql+psycopg://{_user}:{_pass}@{_host}:{_port}/{_db}"
PGVECTOR_URL = os.getenv("PGVECTOR_URL", _default_url)

BATCH_SIZE = 64
INPUT_JSON_PATH = _CURRENT_DIR / "2.raptor_chunks.json"
# endregion


# region Ingestion Logic
def ingest_collapsed_tree(
    chunks: list[dict],
    collection_name: str = "raptor_chunks",
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
                page_content=d.get("small")
                or d.get("metadata", {}).get("summary")
                or d.get("metadata", {}).get("title")
                or d.get("metadata", {}).get("big", "document"),
                metadata=d.get("metadata", {}),
            )
            for d in batch
        ]
        batch_ids = [str(d["id"]) for d in batch]
        vector_store.add_documents(documents=batch_docs, ids=batch_ids)
        print(
            f"[Ingestion] Processed {min(start + BATCH_SIZE, total)}/{total} chunks..."
        )

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
