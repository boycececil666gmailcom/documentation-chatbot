# region Generator
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.synthesizers.multi_hop import (
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.transforms.engine import Parallel
from ragas.testset.transforms.extractors import EmbeddingExtractor, SummaryExtractor
from ragas.testset.transforms.extractors.llm_based import NERExtractor, ThemesExtractor
from ragas.testset.transforms.filters import CustomNodeFilter
from ragas.testset.transforms.relationship_builders import (
    CosineSimilarityBuilder,
    OverlapScoreBuilder,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(dotenv_path=ROOT_DIR / ".env")

CONFIG_PATH = CURRENT_DIR / "eval_config.json"
with open(CONFIG_PATH, encoding="utf-8") as f:
    eval_config = json.load(f)

gen_cfg = eval_config.get("generator", {})
INPUT_FILE = CURRENT_DIR / gen_cfg.get("doc_path", "0.doc.json")
OUTPUT_FILE = CURRENT_DIR / gen_cfg.get("dataset_path", "1.dataset.json")


def get_eval_models() -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    """Initializes LLM and Embeddings wrappers for RAGAS evaluation."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("[Generator-init] OPENROUTER_API_KEY is not set in environment.")

    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-v4.1-flash")
    embed_model = os.getenv("OPENROUTER_EMBED_MODEL", "nvidia/nemotron-3-embed-1b:free")

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
    )
    embeddings = OpenAIEmbeddings(
        model=embed_model,
        api_key=api_key,
        base_url=base_url,
        check_embedding_ctx_length=False,
    )

    wrapped_llm = LangchainLLMWrapper(llm, is_finished_parser=lambda _: True)
    wrapped_embeddings = LangchainEmbeddingsWrapper(embeddings)
    return wrapped_llm, wrapped_embeddings


def load_input_documents(doc_path: Path, max_chunks: int = 20) -> list[Document]:
    """Loads documents from sampled 0.doc.json."""
    if not doc_path.exists():
        raise FileNotFoundError(f"[Generator-load] Target document not found: {doc_path}")

    items = json.loads(doc_path.read_text(encoding="utf-8"))
    items = items if isinstance(items, list) else [items]
    docs: list[Document] = []
    for item in items:
        meta = item.get("metadata", {})
        content = (item.get("content") or meta.get("big") or meta.get("summary") or "").strip()
        if len(content) > 50:
            docs.append(Document(page_content=content, metadata={"id": item.get("id", "")}))
        if 0 < max_chunks <= len(docs):
            break
    return docs


def build_query_distribution(ragas_llm: Any, weights: dict[str, float]) -> list[tuple]:
    """Builds a normalized 4-path query distribution matrix."""
    synthesizers = {
        "single_specific": SingleHopSpecificQuerySynthesizer(
            llm=ragas_llm, property_name="entities", name="single_hop_specific"
        ),
        "single_abstract": SingleHopSpecificQuerySynthesizer(
            llm=ragas_llm, property_name="themes", name="single_hop_abstract"
        ),
        "multi_specific": MultiHopSpecificQuerySynthesizer(
            llm=ragas_llm, name="multi_hop_specific"
        ),
        "multi_abstract": MultiHopAbstractQuerySynthesizer(
            llm=ragas_llm, name="multi_hop_abstract"
        ),
    }
    raw = [(synthesizers[k], max(0.0, weights.get(k, 0.25))) for k in synthesizers]
    total = sum(w for _, w in raw) or 1.0
    return [(s, w / total) for s, w in raw if w > 0]


def generate_eval_dataset(
    doc_path: Path,
    output_path: Path,
    test_size: int = 30,
    max_chunks: int = 20,
    weights: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Synthesizes evaluation test dataset and saves 1.dataset.json."""
    print(f"[Generator-load] Loading up to {max_chunks} chunks from: {doc_path.name}")
    docs = load_input_documents(doc_path, max_chunks=max_chunks)
    if not docs:
        raise ValueError(f"[Generator-load] No readable document content found in {doc_path}")

    ragas_llm, ragas_embeddings = get_eval_models()
    transforms = [
        SummaryExtractor(llm=ragas_llm),
        CustomNodeFilter(llm=ragas_llm),
        Parallel(
            EmbeddingExtractor(
                embedding_model=ragas_embeddings,
                property_name="summary_embedding",
                embed_property_name="summary",
            ),
            ThemesExtractor(llm=ragas_llm),
            NERExtractor(llm=ragas_llm),
        ),
        Parallel(
            CosineSimilarityBuilder(
                property_name="summary_embedding",
                new_property_name="summary_similarity",
                threshold=0.1,
            ),
            OverlapScoreBuilder(threshold=0.01),
        ),
    ]

    distribution = build_query_distribution(ragas_llm, weights or {})
    summary_str = ", ".join(f"{s.name}: {w:.0%}" for s, w in distribution)
    print(
        f"[Generator-generate] Synthesizing {test_size} samples ({summary_str}) across {len(docs)} chunks..."
    )

    generator = TestsetGenerator(
        llm=ragas_llm,
        embedding_model=ragas_embeddings,
        knowledge_graph=KnowledgeGraph(),
    )
    run_config = RunConfig(max_workers=4, max_retries=5, timeout=240)
    dataset = generator.generate_with_langchain_docs(
        documents=docs,
        testset_size=test_size,
        transforms=transforms,
        query_distribution=distribution,
        run_config=run_config,
    )

    samples = [
        {
            "id": f"sample-{idx + 1:02d}",
            "question": row.get("user_input") or row.get("question") or "",
            "ground_truth": row.get("reference") or row.get("ground_truth") or "",
            "ground_truth_contexts": list(row.get("reference_contexts") or []),
            "synthesizer": row.get("synthesizer_name") or "",
        }
        for idx, row in dataset.to_pandas().iterrows()
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[Generator-save] Saved {len(samples)} samples to '{output_path.name}'")
    return samples


def main() -> None:
    """Execute dataset synthesis and export 1.dataset.json intermediate artifact."""
    test_size = gen_cfg.get("test_size", 30)
    max_chunks = gen_cfg.get("max_chunks", 20)
    weights = gen_cfg.get("weights")

    generate_eval_dataset(
        doc_path=INPUT_FILE,
        output_path=OUTPUT_FILE,
        test_size=test_size,
        max_chunks=max_chunks,
        weights=weights,
    )


if __name__ == "__main__":
    main()
# endregion
