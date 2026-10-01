# region Imports
import asyncio
import json
import re
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import MarkdownHeaderTextSplitter
from pydantic import BaseModel, Field

from llm_client import llm

# endregion

# region Configuration
_CURRENT_DIR = Path(__file__).resolve().parent
INPUT_JSON_PATH = _CURRENT_DIR / "1.scrape.json"
OUTPUT_JSON_PATH = _CURRENT_DIR / "2.raptor_chunks.json"
_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
MAX_CONCURRENCY = 20
# endregion


# region Helpers
def clean_markdown(text: str) -> str:
    """Cleans raw markdown by removing links, images, anchors, and boilerplate."""
    if not text:
        return ""
    text = re.sub(r"\[\!\[.*?\]\(.*?\)\]\(.*?\)", "", text)  # Image link
    text = re.sub(r"\!\[.*?\]\(.*?\)", "", text)  # Standalone image
    text = re.sub(r"\[¶\]\(.*?\)", "", text)  # Heading anchor
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)  # Text link to plain text
    # Filter UI boilerplate lines
    boilerplate_pat = r"^(Copy page|Copy this page as Markdown|View as Markdown|Ask Claude|Ask ChatGPT|Connect to VS Code|Download and run the Claude|Copy to clipboard)"
    cleaned_lines = [
        line
        for line in text.splitlines()
        if not re.search(boilerplate_pat, line.strip(), re.IGNORECASE)
    ]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines)).strip()


# endregion


# region Models
class ChunkSummaryResult(BaseModel):
    """Pydantic schema for structured documentation summary and keyword extraction."""

    summary: str = Field(
        description="Concise 2-3 sentence summary of the documentation chunk."
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="3-5 technical keywords relevant to the content.",
    )


class RaptorNode:
    """Represents a node in the 3-tier RAPTOR tree (0: Root, 1: Section, 2: Leaf)."""

    def __init__(
        self,
        node_id: str,
        title: str,
        layer: int,
        summary: str,
        url: str = "",
        markdown: str = "",
        keywords: list[str] | None = None,
        parent_id: str | None = None,
        breadcrumb: str = "",
        links: dict[str, list[str]] | None = None,
    ):
        self.node_id = node_id
        self.title = title
        self.layer = layer
        self.summary = summary
        self.url = url
        self.markdown = clean_markdown(markdown)
        self.keywords = keywords or []
        self.parent_id = parent_id or ""
        self.breadcrumb = breadcrumb
        self.links = links or {"internal": [], "external": []}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "metadata": {
                "parent_id": self.parent_id,
                "title": self.title,
                "summary": self.summary,
                "url": self.url,
                "raptor_layer": self.layer,
                "breadcrumb": self.breadcrumb,
                "keywords": self.keywords,
                "links": self.links,
                "big": self.markdown,
            },
        }


# endregion


# region Tree Builder
class RaptorTreeBuilder:
    """Builds 3-layer RAPTOR structured dataset (0: Root, 1: Section, 2: Leaf)."""

    def __init__(self, raw_json_path: Path):
        self.raw_json_path = raw_json_path
        self.nodes: list[RaptorNode] = []
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")],
            strip_headers=False,
        )
        self.llm = (
            llm.with_structured_output(ChunkSummaryResult, method="json_mode")
            if llm
            else None
        )
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    def _uuid(self, seed: str) -> str:
        return str(uuid.uuid5(_NAMESPACE, seed))

    async def _summarize_chunk(self, title: str, text: str) -> tuple[str, list[str]]:
        cleaned = clean_markdown(text)
        clean_lines = [
            line.strip()
            for line in cleaned.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        fallback_summary = " ".join(clean_lines)[:250] if clean_lines else title

        if not self.llm or not cleaned:
            return fallback_summary, []

        prompt = (
            f"Document Section: {title}\n\nContent:\n{cleaned[:3000]}\n\n"
            "Return a JSON object with exactly two keys:\n"
            '- "summary": concise 2-3 sentence summary\n'
            '- "keywords": array of 3-5 technical keywords'
        )

        async with self.semaphore:
            try:
                result: ChunkSummaryResult = await self.llm.ainvoke(
                    [
                        SystemMessage(
                            content="You are an expert technical documentation summarizer. You MUST output valid JSON matching the requested schema."
                        ),
                        HumanMessage(content=prompt),
                    ]
                )
                return result.summary or fallback_summary, result.keywords or []
            except Exception:
                return fallback_summary, []

    async def build(self) -> list[RaptorNode]:
        with open(self.raw_json_path, encoding="utf-8") as f:
            data = json.load(f)

        await asyncio.gather(*[self._process_root(root) for root in data])
        return self.nodes

    async def _create_node(
        self,
        item: dict[str, Any],
        layer: int,
        parent_id: str = "",
        breadcrumb: str = "",
    ) -> RaptorNode:
        title = item.get("title", f"Layer {layer} Doc")
        url = item.get("url", "")
        markdown = item.get("markdown_content", "")
        summary, keywords = await self._summarize_chunk(title, markdown)
        return RaptorNode(
            node_id=self._uuid(f"layer{layer}_{url}_{title}"),
            title=title,
            layer=layer,
            summary=summary,
            url=url,
            markdown=markdown,
            keywords=keywords,
            parent_id=parent_id,
            breadcrumb=breadcrumb or title,
            links=item.get("links", {"internal": [], "external": []}),
        )

    async def _process_root(self, item: dict[str, Any]):
        root_node = await self._create_node(item, layer=0)
        self.nodes.append(root_node)

        await asyncio.gather(
            *[
                self._process_section(
                    sec, parent_id=root_node.node_id, breadcrumb=root_node.title
                )
                for sec in item.get("sub_documents", [])
            ]
        )

    async def _process_section(
        self, item: dict[str, Any], parent_id: str, breadcrumb: str
    ):
        title = item.get("title", "Section")
        sec_bc = f"{breadcrumb} > {title}"
        sec_node = await self._create_node(
            item, layer=1, parent_id=parent_id, breadcrumb=sec_bc
        )
        self.nodes.append(sec_node)

        sub_docs = item.get("sub_documents", []) or [item]
        await asyncio.gather(
            *[
                self._chunk_leaf(
                    leaf,
                    parent_id=sec_node.node_id,
                    breadcrumb=f"{sec_bc} > {leaf.get('title', '')}",
                )
                for leaf in sub_docs
            ]
        )

    async def _chunk_leaf(self, doc: dict[str, Any], parent_id: str, breadcrumb: str):
        markdown = doc.get("markdown_content", "").strip()
        if not markdown:
            return

        splits = self.md_splitter.split_text(markdown)
        url = doc.get("url", "")
        base_title = doc.get("title", "Detail Chunk")
        links = doc.get("links", {"internal": [], "external": []})

        async def _build_leaf(idx: int, split: Any) -> RaptorNode:
            content = (
                split.page_content if hasattr(split, "page_content") else str(split)
            )
            title = base_title
            for line in content.splitlines():
                if line.strip().startswith("#"):
                    title = line.strip().lstrip("#").split("[¶]")[0].strip()
                    break

            summary, keywords = await self._summarize_chunk(title, content)
            return RaptorNode(
                node_id=self._uuid(f"leaf_{url}_{base_title}_{idx}"),
                title=title,
                layer=2,
                summary=summary,
                url=url,
                markdown=content,
                keywords=keywords,
                parent_id=parent_id,
                breadcrumb=f"{breadcrumb} > {title}"
                if title != base_title
                else breadcrumb,
                links=links,
            )

        leaves = await asyncio.gather(
            *[_build_leaf(i, s) for i, s in enumerate(splits)]
        )
        self.nodes.extend(leaves)


# endregion


# region Pipeline Execution
async def run_formatting_pipeline_async():
    """Builds RAPTOR chunks and formats them into standard JSON."""
    if not INPUT_JSON_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_JSON_PATH}")

    print(
        f"[RAPTOR] Building tree and generating chunk summaries from: {INPUT_JSON_PATH.name}..."
    )
    builder = RaptorTreeBuilder(INPUT_JSON_PATH)
    nodes = await builder.build()
    formatted = [n.to_dict() for n in nodes]

    counts = {
        i: sum(1 for n in formatted if n["metadata"]["raptor_layer"] == i)
        for i in range(3)
    }
    print(
        f"[RAPTOR] Formatted {len(formatted)} chunks: Layer 0={counts[0]}, Layer 1={counts[1]}, Layer 2={counts[2]}"
    )

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(formatted, f, ensure_ascii=False, indent=2)

    print(f"[RAPTOR] Saved dataset to: {OUTPUT_JSON_PATH.name}")


def run_formatting_pipeline():
    asyncio.run(run_formatting_pipeline_async())


if __name__ == "__main__":
    run_formatting_pipeline()
# endregion
