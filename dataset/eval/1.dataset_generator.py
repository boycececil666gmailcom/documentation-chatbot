# region Imports & Setup
import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
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

try:
    from .llm_client import get_eval_models
except ImportError:
    from llm_client import get_eval_models
# endregion


# region Document Loader
def load_input_documents(doc_path: Path, max_chunks: int = 20) -> list[Document]:
    """Loads documents from text files or sampled JSON chunks."""
    if not doc_path.exists():
        raise FileNotFoundError(
            f"[DatasetGenerator-load] Target document not found: {doc_path}"
        )

    if doc_path.suffix.lower() == ".json":
        items = json.loads(doc_path.read_text(encoding="utf-8"))
        items = items if isinstance(items, list) else [items]
        docs = []
        for item in items:
            meta = item.get("metadata", {})
            content = (
                meta.get("big") or meta.get("summary") or item.get("small") or ""
            ).strip()
            if len(content) > 50:
                docs.append(
                    Document(page_content=content, metadata={"id": item.get("id", "")})
                )
            if 0 < max_chunks <= len(docs):
                break
        return docs

    return TextLoader(str(doc_path), encoding="utf-8").load()


# endregion


# region Query Distribution
def build_query_distribution(ragas_llm, weights: dict[str, float]) -> list[tuple]:
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


# endregion


# region Dataset Generator
def generate_eval_dataset_from_docs(
    doc_path: Path,
    output_path: Path,
    test_size: int = 30,
    max_chunks: int = 20,
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """Synthesizes evaluation test samples across 4 distinct query synthesis paths."""
    print(f"[DatasetGenerator-load] Loading up to {max_chunks} chunks from: {doc_path}")
    docs = load_input_documents(doc_path, max_chunks=max_chunks)
    if not docs:
        raise ValueError(
            f"[DatasetGenerator-load] No readable document content found in {doc_path}"
        )

    ragas_llm, ragas_embeddings = get_eval_models(is_generator=True)
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
        f"[DatasetGenerator-generate] Synthesizing {test_size} samples ({summary_str}) across {len(docs)} chunks..."
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
            "question": row.get("user_input") or row.get("question") or "",
            "right_answer": row.get("reference") or row.get("ground_truth") or "",
            "metadata": {
                "id": f"sample-{idx + 1:02d}",
                "ground_truth_contexts": list(row.get("reference_contexts") or []),
                "synthesizer": row.get("synthesizer_name") or "",
            },
        }
        for idx, row in dataset.to_pandas().iterrows()
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[DatasetGenerator-save] Saved {len(samples)} samples to {output_path}")
    return samples


# endregion


# region CLI Interface
def main():
    default_doc = Path(__file__).resolve().parent / "0.chunks.json"
    default_output = Path(__file__).resolve().parent / "1.eval_dataset.json"

    parser = argparse.ArgumentParser(
        description="Synthesize evaluation dataset from sampled chunks via OpenRouter."
    )
    parser.add_argument(
        "--doc",
        "-d",
        type=str,
        default=str(default_doc),
        help="Input document or JSON path",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=str(default_output), help="Output JSON path"
    )
    parser.add_argument(
        "--size",
        "-s",
        type=int,
        default=30,
        help="Number of test samples (default: 30)",
    )
    parser.add_argument(
        "--chunks",
        "-c",
        type=int,
        default=20,
        help="Number of chunks to sample from JSON (default: 20)",
    )
    parser.add_argument(
        "--single-specific",
        type=float,
        default=0.25,
        help="Weight for Single-Hop Specific (default: 0.25)",
    )
    parser.add_argument(
        "--single-abstract",
        type=float,
        default=0.25,
        help="Weight for Single-Hop Abstract (default: 0.25)",
    )
    parser.add_argument(
        "--multi-specific",
        type=float,
        default=0.25,
        help="Weight for Multi-Hop Specific (default: 0.25)",
    )
    parser.add_argument(
        "--multi-abstract",
        type=float,
        default=0.25,
        help="Weight for Multi-Hop Abstract (default: 0.25)",
    )
    args = parser.parse_args()

    weights = {
        "single_specific": args.single_specific,
        "single_abstract": args.single_abstract,
        "multi_specific": args.multi_specific,
        "multi_abstract": args.multi_abstract,
    }
    generate_eval_dataset_from_docs(
        doc_path=Path(args.doc).resolve(),
        output_path=Path(args.output).resolve(),
        test_size=args.size,
        max_chunks=args.chunks,
        weights=weights,
    )


if __name__ == "__main__":
    main()
# endregion
