# region Imports & Setup
import json
import sys
import time
from pathlib import Path

from langchain_core.documents import Document

# Bootstrap project root
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.vector_db import get_vector_store

# endregion


# region Full Ingestion Script
def run_full_ingestion(batch_size: int = 100):
    print("\n\033[1;96m========================================================\033[0m")
    print(
        "\033[1;92m>>> Starting Full Reset & Re-Ingestion of All Chunks into PGVector\033[0m"
    )
    print("\033[1;96m========================================================\033[0m\n")

    store = get_vector_store("raptor_chunks")

    # 1. Load rag_chunks.json
    chunks_path = project_root / "preprocessing-pipeline" / "rag_chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {chunks_path}")

    with open(chunks_path, encoding="utf-8") as f:
        data = json.load(f)

    raw_chunks = data.get("chunks", []) if isinstance(data, dict) else data
    print(f"Loaded {len(raw_chunks):,} chunks from {chunks_path.name}.")

    # 2. Load eval_dataset.json
    eval_path = project_root / "eval" / "eval_dataset.json"
    eval_chunks = []
    if eval_path.exists():
        with open(eval_path, encoding="utf-8") as ef:
            eval_data = json.load(ef)
        for item in eval_data:
            text = item.get("ground_truth") or item.get("content") or ""
            if text:
                eval_chunks.append(
                    {
                        "content": text,
                        "metadata": {
                            "page_title": item.get("page_title", ""),
                            "source_url": item.get("source_url", ""),
                            "id": item.get("id", ""),
                        },
                    }
                )
        print(f"Loaded {len(eval_chunks):,} evaluation chunks from {eval_path.name}.")

    all_raw = raw_chunks + eval_chunks
    total_chunks = len(all_raw)
    print(f"Total Combined Chunks to Ingest: {total_chunks:,}")

    # 3. Batch Ingestion
    documents_batch = []
    success_count = 0
    start_time = time.time()

    for idx, item in enumerate(all_raw, start=1):
        text = (
            item.get("child_content") or item.get("content") or item.get("text") or ""
        )
        if not text:
            continue

        raw_meta = item.get("metadata") or {}
        metadata = {
            k: json.dumps(v, ensure_ascii=False)
            if isinstance(v, (list, dict))
            else str(v)
            for k, v in raw_meta.items()
            if v is not None
        }

        documents_batch.append(Document(page_content=text, metadata=metadata))

        if len(documents_batch) >= batch_size or idx == total_chunks:
            try:
                store.add_documents(documents_batch)
                success_count += len(documents_batch)
            except Exception as e:
                print(f"Error ingesting batch ending at index {idx}: {e}")

            documents_batch.clear()

            if idx % 1000 == 0 or idx == total_chunks:
                elapsed = time.time() - start_time
                rate = success_count / elapsed if elapsed > 0 else 0
                print(
                    f"Progress: [{idx:,}/{total_chunks:,}] ({success_count:,} ingested, {rate:.1f} chunks/sec)"
                )

    print("\n\033[1;96m========================================================\033[0m")
    print(
        f"\033[1;92m>>> Full Re-Ingestion Complete: {success_count:,}/{total_chunks:,} Chunks Successfully Stored in PGVector!\033[0m"
    )
    print("\033[1;96m========================================================\033[0m\n")


if __name__ == "__main__":
    run_full_ingestion()
# endregion
