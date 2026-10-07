# region Hierarchy
import json
import sys
import uuid
from pathlib import Path
from typing import Any

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

CONFIG_PATH = CURRENT_DIR / "crawler_config.json"
with open(CONFIG_PATH, encoding="utf-8") as f:
    config = json.load(f)

hierarchy_cfg = config.get("hierarchy", {})
INPUT_FILE = CURRENT_DIR / hierarchy_cfg.get("input_path", "1.crawler.json")
OUTPUT_FILE = CURRENT_DIR / hierarchy_cfg.get("output_path", "2.structure_hierarchy.json")

NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def flatten_hierarchy_tree(raw_trees: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flattens nested crawler document trees into page-level structured documents without chunking."""
    flat_docs: list[dict[str, Any]] = []

    def _traverse(node: dict[str, Any], parent_id: str, breadcrumb: str, depth: int) -> None:
        title = node.get("title", "Untitled").strip()
        url = node.get("url", "").strip()
        content = node.get("markdown_content", "").strip()
        sub_docs = node.get("sub_documents", [])
        bc = f"{breadcrumb} > {title}" if breadcrumb else title

        # Layer mapping: depth 1 -> layer 0 (root), depth 2 -> layer 1 (category), depth >= 3 -> layer 2 (leaf)
        layer = min(depth - 1, 2)
        doc_id = str(uuid.uuid5(NAMESPACE, f"doc_{parent_id}_{url}_{title}"))

        doc_record = {
            "id": doc_id,
            "title": title,
            "url": url,
            "content": content,
            "metadata": {
                "parent_id": parent_id,
                "raptor_layer": layer,
                "breadcrumb": bc,
                "char_count": len(content),
                "has_sub_documents": len(sub_docs) > 0,
                "sub_documents_count": len(sub_docs),
            },
        }
        flat_docs.append(doc_record)

        for child in sub_docs:
            _traverse(child, doc_id, bc, depth + 1)

    for tree in raw_trees:
        _traverse(tree, parent_id="", breadcrumb="", depth=1)

    return flat_docs


def main() -> None:
    """Executes tree hierarchy flattening and exports page-level structured documents."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input crawler file not found: {INPUT_FILE}")

    print(f"[Hierarchy-main] Reading raw crawler tree from '{INPUT_FILE.name}'...")
    with open(INPUT_FILE, encoding="utf-8") as f:
        raw_trees: list[dict[str, Any]] = json.load(f)

    flat_docs = flatten_hierarchy_tree(raw_trees)

    # Observability metrics
    layer_counts: dict[int, int] = {}
    char_lengths = []
    for d in flat_docs:
        layer = d["metadata"]["raptor_layer"]
        layer_counts[layer] = layer_counts.get(layer, 0) + 1
        char_lengths.append(d["metadata"]["char_count"])

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(flat_docs, f, ensure_ascii=False, indent=2)

    avg_chars = sum(char_lengths) // len(char_lengths) if char_lengths else 0
    large_docs = sum(1 for c in char_lengths if c > 3500)

    print(
        f"[Hierarchy-main] Extracted {len(flat_docs)} page-level documents across layers: {layer_counts}"
    )
    print(
        f"[Hierarchy-main] Character stats: min={min(char_lengths, default=0)}, avg={avg_chars}, max={max(char_lengths, default=0)}"
    )
    print(
        f"[Hierarchy-main] Documents exceeding 3,500 chars (to be split in Step 3): {large_docs}/{len(flat_docs)}"
    )
    print(f"[Hierarchy-main] Saved page-level structured documents to '{OUTPUT_FILE.name}'")


if __name__ == "__main__":
    main()
# endregion
