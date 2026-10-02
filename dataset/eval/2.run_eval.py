# region Imports & Setup
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import httpx
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

try:
    from .llm_client import get_eval_models
except ImportError:
    from llm_client import get_eval_models
# endregion


# region RAG Fetcher
def fetch_rag_responses(dataset_path: Path, endpoint_url: str) -> list[dict]:
    """Queries the running RAG endpoint for each test question in the dataset."""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"[EvalRunner-fetch] Dataset file not found: {dataset_path}"
        )

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    print(
        f"[EvalRunner-fetch] Querying RAG endpoint ({endpoint_url}) for {len(dataset)} questions..."
    )

    samples = []
    with httpx.Client(timeout=90.0) as client:
        for idx, item in enumerate(dataset, 1):
            q = item.get("question", "").strip()
            if not q:
                continue

            print(f"[EvalRunner-fetch] [{idx}/{len(dataset)}] Query: '{q[:60]}...'")
            answer, contexts = "Error querying RAG endpoint.", ["Error"]
            for attempt in range(1, 4):
                try:
                    resp = client.post(endpoint_url, json={"query": q, "history": []})
                    resp.raise_for_status()
                    data = resp.json()
                    answer = data.get("final_response") or data.get("response") or ""
                    raw_ctx = (
                        data.get("retrieved_documents") or data.get("contexts") or []
                    )
                    contexts = (
                        [str(c) for c in raw_ctx]
                        if isinstance(raw_ctx, list)
                        else [str(raw_ctx)]
                    )
                    break
                except Exception as err:
                    print(
                        f"[EvalRunner-fetch] Attempt {attempt}/3 failed for '{q[:40]}': {err}"
                    )
                    if attempt < 3:
                        import time

                        time.sleep(2.0 * attempt)

            meta = item.get("metadata", {})
            gt_contexts = meta.get("ground_truth_contexts")

            samples.append(
                {
                    "question": q,
                    "answer": answer,
                    "contexts": contexts,
                    "right_answer": item.get("right_answer", ""),
                    "reference_contexts": gt_contexts,
                }
            )

    return samples


# endregion


# region Metric Evaluation
def run_ragas_evaluation(samples: list[dict], output_dir: Path):
    """Executes RAGAS evaluation on collected samples and exports results to CSV."""
    if not samples:
        raise ValueError("[EvalRunner-eval] No samples to evaluate.")

    eval_llm, eval_embeddings = get_eval_models()
    eval_dataset = Dataset.from_dict(
        {
            "question": [s["question"] for s in samples],
            "answer": [s["answer"] for s in samples],
            "contexts": [s["contexts"] for s in samples],
            "ground_truth": [s["right_answer"] for s in samples],
            "reference_contexts": [s["reference_contexts"] for s in samples],
        }
    )

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    print(f"[EvalRunner-eval] Evaluating {len(samples)} samples with RAGAS metrics...")
    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=RunConfig(max_workers=2, max_retries=5, timeout=240),
    )

    df: pd.DataFrame = result.to_pandas()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"2.run_eval_{timestamp}.csv"

    avg_row = {col: "-" for col in df.columns}
    avg_row["user_input"] = "[AVERAGE SUMMARY]"
    for m in metrics:
        if m.name in df.columns:
            avg_row[m.name] = round(float(df[m.name].mean()), 4)

    df_out = pd.concat([pd.DataFrame([avg_row]), df], ignore_index=True)
    df_out.to_csv(csv_path, index=False)

    print(f"\n[EvalRunner-save] CSV saved to: {csv_path}\n")


# endregion


# region CLI Interface
def main():
    default_dataset = Path(__file__).resolve().parent / "1.eval_dataset.json"
    default_output = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Run RAGAS evaluation on RAG pipeline."
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default=str(default_dataset),
        help="Dataset JSON path",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=str(default_output),
        help="Output directory",
    )
    parser.add_argument(
        "--endpoint",
        "-e",
        type=str,
        default=os.getenv("RAG_ENDPOINT", "http://localhost:8000/query"),
        help="RAG API Endpoint",
    )
    args = parser.parse_args()

    samples = fetch_rag_responses(
        dataset_path=Path(args.dataset).resolve(),
        endpoint_url=args.endpoint,
    )
    run_ragas_evaluation(samples=samples, output_dir=Path(args.output_dir).resolve())


if __name__ == "__main__":
    main()
# endregion
