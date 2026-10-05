# region Imports
import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client
# endregion


# region Helpers
def load_eval_config(config_path: Path) -> dict:
    """Loads JSON configuration file for evaluation."""
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as err:
        print(f"[LangSmith-config] Warning: Failed to parse {config_path.name}: {err}")
        return {}
# endregion


# region Uploader
def upload_dataset_to_langsmith(
    dataset_path: Path,
    dataset_name: str,
    description: str,
) -> str | None:
    """Uploads clean local dataset to LangSmith with dynamic inputs/outputs mapping."""
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key or api_key.strip() in ("", "mock_key"):
        print("[LangSmith-upload] Error: LANGSMITH_API_KEY is not configured in environment.")
        return None

    if not dataset_path.exists():
        raise FileNotFoundError(f"[LangSmith-upload] Dataset file not found: {dataset_path}")

    raw_samples = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(raw_samples, list):
        raw_samples = [raw_samples]

    client = Client()

    # Check or create target dataset
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
        print(f"[LangSmith-upload] Found existing dataset '{dataset_name}' (ID: {dataset.id})")
    else:
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=description,
        )
        print(f"[LangSmith-upload] Created new dataset '{dataset_name}' (ID: {dataset.id})")

    # Dynamic transformation into LangSmith standard inputs / outputs
    inputs = [
        {"query": s.get("question") or s.get("inputs", {}).get("query", "")}
        for s in raw_samples
    ]
    outputs = [
        {"ground_truth": s.get("ground_truth") or s.get("outputs", {}).get("ground_truth") or ""}
        for s in raw_samples
    ]
    metadata = [
        {
            "id": s.get("id") or s.get("metadata", {}).get("id", f"sample-{idx + 1:02d}"),
            "ground_truth_contexts": s.get("ground_truth_contexts") or s.get("metadata", {}).get("ground_truth_contexts", []),
            "synthesizer": s.get("synthesizer") or s.get("metadata", {}).get("synthesizer", ""),
        }
        for idx, s in enumerate(raw_samples)
    ]

    print(f"[LangSmith-upload] Ingesting {len(raw_samples)} examples into LangSmith dataset '{dataset_name}'...")
    client.create_examples(
        inputs=inputs,
        outputs=outputs,
        metadata=metadata,
        dataset_id=dataset.id,
    )
    print(f"[LangSmith-upload] Successfully uploaded {len(raw_samples)} examples to LangSmith!")
    return str(dataset.id)
# endregion


# region Main
def main():
    """CLI entrypoint for standalone LangSmith dataset ingestion."""
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parents[1]
    load_dotenv(root_dir / ".env")

    config_path = current_dir / "eval_config.json"
    cfg = load_eval_config(config_path)
    ls_cfg = cfg.get("langsmith", {})

    parser = argparse.ArgumentParser(description="Upload evaluation dataset to LangSmith.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(current_dir / ls_cfg.get("dataset_path", "1.dataset.json")),
        help="Path to evaluation dataset JSON file",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=ls_cfg.get("dataset_name", "kanzi-documentation-eval"),
        help="LangSmith dataset name",
    )
    parser.add_argument(
        "--description",
        type=str,
        default=ls_cfg.get("description", "Kanzi QA RAG evaluation test set synthesized via KnowledgeGraph"),
        help="Description for the LangSmith dataset",
    )

    args = parser.parse_args()
    dataset_file = Path(args.dataset)

    upload_dataset_to_langsmith(
        dataset_path=dataset_file,
        dataset_name=args.name,
        description=args.description,
    )


if __name__ == "__main__":
    main()
# endregion
