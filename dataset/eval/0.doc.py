# region Imports
import json
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
# endregion

# region Types
class SampledDoc:
    """Represents a sanitized, representative documentation chunk for evaluation."""

    def __init__(
        self,
        doc_id: str,
        content: str,
        title: str,
        breadcrumb: str,
        raptor_layer: int,
        url: str = "",
        parent_id: str = "",
        child_ids: list[str] | None = None,
        char_count: int = 0,
    ) -> None:
        self.doc_id = doc_id
        self.content = content
        self.title = title
        self.breadcrumb = breadcrumb
        self.raptor_layer = raptor_layer
        self.url = url
        self.parent_id = parent_id
        self.child_ids = child_ids or []
        self.char_count = char_count or len(content)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.doc_id,
            "content": self.content,
            "metadata": {
                "title": self.title,
                "breadcrumb": self.breadcrumb,
                "level": self.raptor_layer,
                "raptor_layer": self.raptor_layer,
                "url": self.url,
                "parent_id": self.parent_id,
                "child_ids": self.child_ids,
                "char_count": self.char_count,
            },
        }
# endregion

# region Scaffolding
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(dotenv_path=ROOT_DIR / ".env")

CONFIG_PATH = CURRENT_DIR / "eval_config.json"
with open(CONFIG_PATH, encoding="utf-8") as f:
    eval_config = json.load(f)

gen_cfg = eval_config.get("generator", {})
CHUNKS_SOURCE = ROOT_DIR / "dataset" / "crawl" / "3.split_chunks.json"
if not CHUNKS_SOURCE.exists():
    CHUNKS_SOURCE = ROOT_DIR / "dataset" / "crawl" / "3.chunks.json"
OUTPUT_FILE = CURRENT_DIR / gen_cfg.get("doc_path", "0.doc.json")
# endregion

# region Sampler
def sanitize_content(raw_text: str) -> str:
    """Removes crawler artifacts such as heading anchor links and copy badges."""
    text = re.sub(r"\[\s*¶\s*\]\([^)]+\)", "", raw_text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)\s*(?:Copy\s+(?:page|link))?", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sample_representative_documents(
    source_chunks_path: Path,
    max_chunks: int = 20,
) -> list[dict[str, Any]]:
    """Performs stratified, domain-diverse sampling across RAPTOR hierarchy layers."""
    if not source_chunks_path.exists():
        raise FileNotFoundError(f"Source chunks file not found: {source_chunks_path}")

    with open(source_chunks_path, encoding="utf-8") as f:
        chunks: list[dict[str, Any]] = json.load(f)

    print(f"[Sampler-sample] Inspecting {len(chunks)} chunks from {source_chunks_path.name}...")

    layer_0_chunks = [c for c in chunks if c.get("metadata", {}).get("raptor_layer") == 0]
    layer_1_chunks = [c for c in chunks if c.get("metadata", {}).get("raptor_layer") == 1]
    layer_2_chunks = [c for c in chunks if c.get("metadata", {}).get("raptor_layer") == 2]

    selected: list[dict[str, Any]] = []

    # 1. Layer 0: Diverse root chapters (up to 3 distinct top-level categories)
    seen_roots: set[str] = set()
    for c in layer_0_chunks:
        root_name = c["metadata"].get("title", "").split(" (Part ")[0]
        if root_name not in seen_roots:
            seen_roots.add(root_name)
            selected.append(c)
        if len(selected) >= 3:
            break

    # 2. Layer 1: Subsystem overviews with distinct breadcrumb paths
    target_l1 = 10
    l1_selected = 0
    seen_l1_bcs: set[str] = set()
    for c in layer_1_chunks:
        bc = c["metadata"].get("breadcrumb", "")
        text = sanitize_content(c.get("content") or c.get("metadata", {}).get("big", ""))
        if bc not in seen_l1_bcs and len(text) > 350:
            seen_l1_bcs.add(bc)
            selected.append(c)
            l1_selected += 1
        if l1_selected >= target_l1:
            break

    # 3. Layer 2: Concrete technical details/APIs from distinct feature families
    seen_l2_bcs: set[str] = set()
    l2_candidates = [
        c for c in layer_2_chunks
        if any(dom in c["metadata"].get("breadcrumb", "") for dom in ["Working with", "References"])
    ]
    for c in l2_candidates:
        if len(selected) >= max_chunks:
            break
        bc = c["metadata"].get("breadcrumb", "")
        text = sanitize_content(c.get("content") or c.get("metadata", {}).get("big", ""))
        if bc not in seen_l2_bcs and len(text) > 400:
            seen_l2_bcs.add(bc)
            selected.append(c)

    eval_docs: list[dict[str, Any]] = []
    for c in selected[:max_chunks]:
        meta = c.get("metadata", {})
        cleaned_text = sanitize_content(c.get("content") or meta.get("big", ""))
        sampled_obj = SampledDoc(
            doc_id=c["id"],
            content=cleaned_text,
            title=meta.get("title", ""),
            breadcrumb=meta.get("breadcrumb", ""),
            raptor_layer=meta.get("raptor_layer", 0),
            url=meta.get("url", ""),
            parent_id=meta.get("parent_id", ""),
            child_ids=meta.get("child_ids", []),
            char_count=len(cleaned_text),
        )
        eval_docs.append(sampled_obj.to_dict())

    return eval_docs
# endregion

# region Core
def run_sampling_pipeline(
    source_path: Path = CHUNKS_SOURCE,
    output_path: Path = OUTPUT_FILE,
    max_chunks: int = 20,
) -> list[dict[str, Any]]:
    """Declarative core execution for sampling and persisting representative docs."""
    docs = sample_representative_documents(source_path, max_chunks=max_chunks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)
    return docs
# endregion

# region Main
def main() -> None:
    """CLI runtime harness."""
    max_chunks = gen_cfg.get("max_chunks", 20)
    docs = run_sampling_pipeline(CHUNKS_SOURCE, OUTPUT_FILE, max_chunks=max_chunks)
    print(f"[Sampler-main] Successfully exported {len(docs)} synchronized documents to '{OUTPUT_FILE.name}'")


if __name__ == "__main__":
    main()
# endregion
