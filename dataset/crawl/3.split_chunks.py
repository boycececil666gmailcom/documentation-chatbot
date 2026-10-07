# region Chunker
import json
import math
import sys
import uuid
from pathlib import Path
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

CONFIG_PATH = CURRENT_DIR / "crawler_config.json"
with open(CONFIG_PATH, encoding="utf-8") as f:
    config = json.load(f)

chunker_cfg = config.get("chunker", {})
INPUT_FILE = CURRENT_DIR / chunker_cfg.get("input_path", "2.structure_hierarchy.json")
OUTPUT_FILE = CURRENT_DIR / chunker_cfg.get("output_path", "3.split_chunks.json")
CHUNK_SIZE = chunker_cfg.get("chunk_size", 3000)
CHUNK_OVERLAP = chunker_cfg.get("chunk_overlap", 300)
THRESHOLD = chunker_cfg.get("max_doc_length_threshold", 3500)

NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "],
)


def split_structured_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Splits oversized structured documents into logical embeddable chunks."""
    all_chunks: list[dict[str, Any]] = []

    for doc in documents:
        content = doc.get("content", "").strip()
        if not content:
            continue

        doc_id = doc["id"]
        title = doc["title"]
        url = doc["url"]
        meta = doc.get("metadata", {})
        parent_id = meta.get("parent_id", "")
        layer = meta.get("raptor_layer", 2)
        bc = meta.get("breadcrumb", title)

        if len(content) > THRESHOLD:
            splits = splitter.split_text(content)
            total_parts = len(splits)
            for idx, part_text in enumerate(splits):
                chunk_id = str(uuid.uuid5(NAMESPACE, f"chunk_{doc_id}_{idx}"))
                part_title = f"{title} (Part {idx + 1}/{total_parts})"
                all_chunks.append({
                    "id": chunk_id,
                    "content": part_text,
                    "metadata": {
                        "parent_id": parent_id,
                        "doc_id": doc_id,
                        "title": part_title,
                        "url": url,
                        "raptor_layer": layer,
                        "breadcrumb": bc,
                        "chunk_index": idx + 1,
                        "total_chunks": total_parts,
                        "char_count": len(part_text),
                    },
                })
        else:
            chunk_id = str(uuid.uuid5(NAMESPACE, f"chunk_{doc_id}_0"))
            all_chunks.append({
                "id": chunk_id,
                "content": content,
                "metadata": {
                    "parent_id": parent_id,
                    "doc_id": doc_id,
                    "title": title,
                    "url": url,
                    "raptor_layer": layer,
                    "breadcrumb": bc,
                    "chunk_index": 1,
                    "total_chunks": 1,
                    "char_count": len(content),
                },
            })

    return all_chunks


def main() -> None:
    """Reads structured documents, applies text chunking, and exports chunks artifact."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Structured documents file not found: {INPUT_FILE}")

    print(f"[Chunker-main] Reading structured documents from '{INPUT_FILE.name}'...")
    with open(INPUT_FILE, encoding="utf-8") as f:
        documents: list[dict[str, Any]] = json.load(f)

    chunks = split_structured_documents(documents)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    chunk_lengths = [c["metadata"]["char_count"] for c in chunks]
    avg_chunk = sum(chunk_lengths) // len(chunk_lengths) if chunk_lengths else 0
    single_chunks = sum(1 for c in chunks if c["metadata"]["total_chunks"] == 1)
    split_chunks = len(chunks) - single_chunks

    print(f"[Chunker-main] Successfully produced {len(chunks)} embeddable chunks from {len(documents)} documents.")
    print(f"[Chunker-main] Single-part documents: {single_chunks} | Split-part chunks: {split_chunks}")
    print(f"[Chunker-main] Chunk length stats: min={min(chunk_lengths, default=0)}, avg={avg_chunk}, max={max(chunk_lengths, default=0)}")
    print(f"[Chunker-main] Saved chunks artifact to '{OUTPUT_FILE.name}'")


if __name__ == "__main__":
    main()
# endregion
