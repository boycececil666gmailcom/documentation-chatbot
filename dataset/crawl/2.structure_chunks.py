# region Chunker
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(dotenv_path=ROOT_DIR / ".env")

CONFIG_PATH = CURRENT_DIR / "crawler_config.json"
with open(CONFIG_PATH, encoding="utf-8") as f:
    config = json.load(f)

raptor_cfg = config.get("raptor", {})
INPUT_FILE = CURRENT_DIR / raptor_cfg.get("input_path", "1.crawler.json")
OUTPUT_FILE = CURRENT_DIR / raptor_cfg.get("output_path", "2.structure_chunks.json")

NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")],
    strip_headers=False,
)
char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=3000,
    chunk_overlap=150,
)


def clean_markdown(text: str) -> str:
    """Removes UI artifacts, download badges, and markdown boilerplate."""
    if not text:
        return ""
    text = re.sub(r"\[\!\[.*?\]\(.*?\)\]\(.*?\)", "", text)
    text = re.sub(r"\!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    boilerplate_pat = (
        r"^(Copy page|Copy this page as Markdown|View as Markdown|Ask Claude|"
        r"Ask ChatGPT|Connect to VS Code|Download and run the Claude|Copy to clipboard)"
    )
    cleaned = [
        line
        for line in text.splitlines()
        if not re.search(boilerplate_pat, line.strip(), re.IGNORECASE)
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


def process_document_tree(raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Structures hierarchical crawl tree into a bounded 3-layer RAPTOR dataset.

    Guarantees that parent nodes contain strictly unique overview text
    not duplicated in any child leaf chunk, skips empty chunks, and normalizes schema.
    """
    flat_nodes: list[dict[str, Any]] = []

    def _add_node(
        node_id: str,
        title: str,
        layer: int,
        url: str,
        markdown: str,
        parent_id: str,
        breadcrumb: str,
    ) -> None:
        cleaned = clean_markdown(markdown)
        if not cleaned or not cleaned.strip():
            return
        flat_nodes.append(
            {
                "id": node_id,
                "content": cleaned,
                "metadata": {
                    "parent_id": parent_id or "",
                    "title": title,
                    "url": url,
                    "raptor_layer": layer,
                    "breadcrumb": breadcrumb or title,
                },
            }
        )

    def _split_and_add_leaves(
        text: str, base_title: str, url: str, parent_id: str, breadcrumb: str
    ) -> None:
        splits = md_splitter.split_text(text) if text else []
        if not splits:
            splits = [text]

        for s_idx, split in enumerate(splits):
            s_text = split.page_content if hasattr(split, "page_content") else str(split)
            title = base_title
            for line in s_text.splitlines():
                if line.strip().startswith("#"):
                    title = re.sub(r"\[.*?\]", "", line.strip().lstrip("#")).strip()
                    break

            bc = f"{breadcrumb} > {title}" if title != base_title else breadcrumb

            # Guard against overly large sub-sections (> 3500 chars)
            if len(s_text) > 3500:
                sub_splits = char_splitter.split_text(s_text)
                for sub_idx, sub_part in enumerate(sub_splits):
                    node_id = str(
                        uuid.uuid5(NAMESPACE, f"leaf_{url}_{base_title}_{s_idx}_{sub_idx}")
                    )
                    part_title = f"{title} (Part {sub_idx + 1})" if len(sub_splits) > 1 else title
                    _add_node(node_id, part_title, 2, url, sub_part, parent_id, bc)
            else:
                node_id = str(uuid.uuid5(NAMESPACE, f"leaf_{url}_{base_title}_{s_idx}"))
                _add_node(node_id, title, 2, url, s_text, parent_id, bc)

    def _handle_section(sec: dict[str, Any], parent_id: str, breadcrumb: str) -> None:
        title = sec.get("title", "Section")
        sec_bc = f"{breadcrumb} > {title}"
        url = sec.get("url", "")
        sub_docs = sec.get("sub_documents", [])
        content = sec.get("markdown_content", "").strip()

        sec_id = str(uuid.uuid5(NAMESPACE, f"layer1_{url}_{title}"))

        # Process section's own markdown content
        splits = md_splitter.split_text(content) if content else []
        if len(splits) > 1:
            p_split = splits[0]
            p_text = p_split.page_content if hasattr(p_split, "page_content") else str(p_split)
            if len(p_text) > 3500:
                p_text = p_text[:3500]
            _add_node(sec_id, title, 1, url, p_text, parent_id, sec_bc)

            # Remaining splits within the same section page become Layer 2 leaves under sec_id
            for idx, split in enumerate(splits[1:], start=1):
                text = split.page_content if hasattr(split, "page_content") else str(split)
                sub_title = title
                for line in text.splitlines():
                    if line.strip().startswith("#"):
                        sub_title = re.sub(r"\[.*?\]", "", line.strip().lstrip("#")).strip()
                        break
                bc = f"{sec_bc} > {sub_title}" if sub_title != title else sec_bc

                if len(text) > 3500:
                    sub_splits = char_splitter.split_text(text)
                    for sub_idx, sub_part in enumerate(sub_splits):
                        node_id = str(uuid.uuid5(NAMESPACE, f"leaf_{url}_{title}_{idx}_{sub_idx}"))
                        part_title = f"{sub_title} (Part {sub_idx + 1})"
                        _add_node(node_id, part_title, 2, url, sub_part, sec_id, bc)
                else:
                    node_id = str(uuid.uuid5(NAMESPACE, f"leaf_{url}_{title}_{idx}"))
                    _add_node(node_id, sub_title, 2, url, text, sec_id, bc)
        else:
            p_text = content[:3500] if len(content) > 3500 else content
            _add_node(sec_id, title, 1, url, p_text, parent_id, sec_bc)

        # Linked sub_documents also become Layer 2 leaves under sec_id
        for leaf in sub_docs:
            l_content = leaf.get("markdown_content", "").strip()
            l_url = leaf.get("url", "")
            l_title = leaf.get("title", "Detail Chunk")
            _split_and_add_leaves(l_content, l_title, l_url, sec_id, f"{sec_bc} > {l_title}")

    def _handle_root(root: dict[str, Any]) -> None:
        title = root.get("title", "Root Document")
        url = root.get("url", "")
        root_id = str(uuid.uuid5(NAMESPACE, f"layer0_{url}_{title}"))
        content = root.get("markdown_content", "")
        p_text = content[:3500] if len(content) > 3500 else content
        _add_node(root_id, title, 0, url, p_text, "", title)

        for sec in root.get("sub_documents", []):
            _handle_section(sec, root_id, title)

    print(
        f"[Chunker-process_document_tree] Structuring bounded 3-layer RAPTOR hierarchy across {len(raw_data)} roots..."
    )
    for r in raw_data:
        _handle_root(r)
    return flat_nodes


def main() -> None:
    """Execute markdown structuring, cleaning, and export structured chunks artifact."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input crawler file not found: {INPUT_FILE}")

    with open(INPUT_FILE, encoding="utf-8") as f:
        tree_input = json.load(f)

    structured_chunks = process_document_tree(tree_input)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(structured_chunks, f, ensure_ascii=False, indent=2)
    print(f"[Chunker-main] Saved {len(structured_chunks)} structured chunks to '{OUTPUT_FILE.name}'")


if __name__ == "__main__":
    main()
# endregion
