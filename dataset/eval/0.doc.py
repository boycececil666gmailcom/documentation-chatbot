# region Sampler
import json
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(dotenv_path=ROOT_DIR / ".env")

CONFIG_PATH = CURRENT_DIR / "eval_config.json"
with open(CONFIG_PATH, encoding="utf-8") as f:
    eval_config = json.load(f)

gen_cfg = eval_config.get("generator", {})
CHUNKS_SOURCE = ROOT_DIR / "dataset" / "crawl" / "2.structure_chunks.json"
OUTPUT_FILE = CURRENT_DIR / gen_cfg.get("doc_path", "0.doc.json")


def sample_representative_documents(
    source_chunks_path: Path,
    max_chunks: int = 20,
) -> list[dict[str, Any]]:
    """Samples diverse, high-value chunks across root and section layers."""
    if not source_chunks_path.exists():
        raise FileNotFoundError(f"Source chunks file not found: {source_chunks_path}")

    with open(source_chunks_path, encoding="utf-8") as f:
        chunks: list[dict[str, Any]] = json.load(f)

    print(
        f"[Sampler-sample] Inspecting {len(chunks)} chunks from {source_chunks_path.name}..."
    )

    layer_0_chunks = [c for c in chunks if c.get("metadata", {}).get("raptor_layer") == 0]
    layer_1_chunks = [c for c in chunks if c.get("metadata", {}).get("raptor_layer") == 1]
    layer_2_chunks = [c for c in chunks if c.get("metadata", {}).get("raptor_layer") == 2]

    # Select representative chunks across all levels
    selected: list[dict[str, Any]] = []

    # 1. Take all root chunks (up to 3)
    selected.extend(layer_0_chunks[:3])

    # 2. Take substantial layer 1 section overviews
    for c in layer_1_chunks:
        if len(selected) >= max_chunks:
            break
        text = c.get("content") or c.get("metadata", {}).get("big", "")
        if len(text) > 200:
            selected.append(c)

    # 3. Fill remaining with detailed leaf chunks if needed
    for c in layer_2_chunks:
        if len(selected) >= max_chunks:
            break
        text = c.get("content") or c.get("metadata", {}).get("big", "")
        if len(text) > 300:
            selected.append(c)

    # Reformat to clean eval doc schema with canonical content
    eval_docs = [
        {
            "id": c.get("id"),
            "content": c.get("content") or c.get("metadata", {}).get("big", ""),
            "metadata": {
                "title": c.get("metadata", {}).get("title", ""),
                "breadcrumb": c.get("metadata", {}).get("breadcrumb", ""),
                "level": c.get("metadata", {}).get("raptor_layer", 0),
            },
        }
        for c in selected[:max_chunks]
    ]

    return eval_docs


def main() -> None:
    """Execute sampling and export 0.doc.json intermediate artifact."""
    max_chunks = gen_cfg.get("max_chunks", 20)
    docs = sample_representative_documents(CHUNKS_SOURCE, max_chunks=max_chunks)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    print(f"[Sampler-main] Exported {len(docs)} representative documents to '{OUTPUT_FILE.name}'")


if __name__ == "__main__":
    main()
# endregion
