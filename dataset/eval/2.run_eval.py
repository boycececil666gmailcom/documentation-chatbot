# region Evaluation
import asyncio
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(dotenv_path=ROOT_DIR / ".env")

CONFIG_PATH = CURRENT_DIR / "eval_config.json"
with open(CONFIG_PATH, encoding="utf-8") as f:
    eval_config = json.load(f)

query_cfg = eval_config.get("query", {})
eval_cfg = eval_config.get("evaluation", {})

INPUT_FILE = CURRENT_DIR / query_cfg.get("dataset_path", "1.dataset.json")
OUTPUT_DIR = CURRENT_DIR / eval_cfg.get("output_dir", ".")
OUTPUT_PREFIX = eval_cfg.get("output_prefix", "2.run_eval_")


def get_eval_models() -> tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    """Initializes LLM and Embeddings wrappers for RAGAS evaluation."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("[EvalRunner-init] OPENROUTER_API_KEY is not set in environment.")

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


async def fetch_single_response(
    query: str,
    mode: str = "direct",
    endpoint_url: str = "http://localhost:2024",
    client: httpx.AsyncClient | None = None,
) -> tuple[str, list[str]]:
    """Fetches answer and retrieved contexts from LangGraph agent."""
    if mode == "direct":
        from src.agent_flow.graph import agent_graph

        state = await agent_graph.ainvoke({"query": query})
        answer = state.get("draft_response") or ""
        docs = state.get("ranked_docs") or state.get("expanded_docs") or []
        contexts = [
            str(getattr(d, "page_content", "") or d.metadata.get("big", ""))
            for d in docs
            if d
        ]
        return answer, contexts

    # HTTP endpoint mode
    payload = {"input": {"query": query}}
    http_client = client or httpx.AsyncClient(timeout=90.0)
    try:
        resp = await http_client.post(f"{endpoint_url}/invoke", json=payload)
        resp.raise_for_status()
        data = resp.json().get("output", {})
        answer = data.get("draft_response", "")
        docs = data.get("ranked_docs", []) or data.get("expanded_docs", [])
        contexts = [d.get("page_content", "") or d.get("metadata", {}).get("big", "") for d in docs]
        return answer, contexts
    finally:
        if not client:
            await http_client.aclose()


async def collect_agent_responses(
    samples: list[dict[str, Any]],
    mode: str = "direct",
    endpoint_url: str = "http://localhost:2024",
    max_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Queries LangGraph agent sequentially for all questions in testset."""
    target_samples = samples[:max_samples] if max_samples else samples
    total = len(target_samples)
    print(f"[EvalRunner-fetch] Querying LangGraph agent for {total} questions (mode: '{mode}')...")

    evaluated_records: list[dict[str, Any]] = []
    for idx, item in enumerate(target_samples, start=1):
        question = item["question"]
        print(f"[{idx}/{total}] Q: {question[:70]}...")
        t0 = time.time()
        try:
            answer, contexts = await fetch_single_response(
                query=question,
                mode=mode,
                endpoint_url=endpoint_url,
            )
            latency = round(time.time() - t0, 2)
            print(f"       A: {answer[:60]}... ({latency}s, {len(contexts)} contexts)")
            evaluated_records.append(
                {
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": item.get("ground_truth", ""),
                    "ground_truth_contexts": item.get("ground_truth_contexts", []),
                    "latency": latency,
                }
            )
        except Exception as e:
            print(f"       ERROR: Failed to query agent: {e}")

    return evaluated_records


def run_ragas_evaluation(
    records: list[dict[str, Any]],
    output_dir: Path,
    output_prefix: str = "2.run_eval_",
    max_workers: int = 2,
    timeout: int = 240,
    max_retries: int = 5,
) -> pd.DataFrame:
    """Executes RAGAS evaluation on collected samples and exports results to CSV."""
    if not records:
        raise ValueError("[EvalRunner-eval] No records provided for evaluation.")

    eval_llm, eval_embeddings = get_eval_models()
    eval_dataset = Dataset.from_dict(
        {
            "question": [r["question"] for r in records],
            "answer": [r["answer"] for r in records],
            "contexts": [r["contexts"] for r in records],
            "ground_truth": [r["ground_truth"] for r in records],
        }
    )

    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    print(f"[EvalRunner-eval] Running RAGAS benchmark on {len(records)} samples...")

    result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=RunConfig(max_workers=max_workers, max_retries=max_retries, timeout=timeout),
    )

    df: pd.DataFrame = result.to_pandas()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_csv = output_dir / f"{output_prefix}{timestamp}.csv"
    canonical_csv = output_dir / "2.run_eval.csv"

    avg_row = {col: "-" for col in df.columns}
    avg_row["user_input"] = "[AVERAGE SUMMARY]"
    for m in metrics:
        if m.name in df.columns:
            avg_row[m.name] = round(float(df[m.name].mean()), 4)

    df_out = pd.concat([pd.DataFrame([avg_row]), df], ignore_index=True)
    df_out.to_csv(timestamp_csv, index=False)
    shutil.copy2(timestamp_csv, canonical_csv)

    print(f"\n[EvalRunner-save] CSV saved to: {timestamp_csv.name} and {canonical_csv.name}")
    print("\n### Mean Evaluation Scores:")
    for m in metrics:
        if m.name in avg_row:
            print(f"- {m.name:20s}: {avg_row[m.name]}")

    return df_out


async def main_async() -> None:
    """Async main routine."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input test dataset file not found: {INPUT_FILE}")

    with open(INPUT_FILE, encoding="utf-8") as f:
        samples: list[dict[str, Any]] = json.load(f)

    mode = query_cfg.get("mode", "direct")
    endpoint_url = query_cfg.get("endpoint_url", "http://localhost:2024")
    max_samples = query_cfg.get("max_samples")

    records = await collect_agent_responses(
        samples=samples,
        mode=mode,
        endpoint_url=endpoint_url,
        max_samples=max_samples,
    )

    max_workers = eval_cfg.get("max_workers", 2)
    timeout = eval_cfg.get("timeout", 240)
    max_retries = eval_cfg.get("max_retries", 5)

    run_ragas_evaluation(
        records=records,
        output_dir=OUTPUT_DIR,
        output_prefix=OUTPUT_PREFIX,
        max_workers=max_workers,
        timeout=timeout,
        max_retries=max_retries,
    )


def main() -> None:
    """Execute evaluation benchmark."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
# endregion
