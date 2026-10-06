# region LangSmithUploader
import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langsmith import Client

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

load_dotenv(dotenv_path=ROOT_DIR / ".env")

CONFIG_PATH = CURRENT_DIR / "eval_config.json"
OUTPUT_FILE = CURRENT_DIR / "2.5.upload_langsmith.json"


def load_eval_config(config_path: Path) -> dict[str, Any]:
    """Loads JSON configuration file for evaluation."""
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as err:
        print(f"[LangSmith-config] Warning: Failed to parse {config_path.name}: {err}")
        return {}


def upload_dataset_to_langsmith(
    dataset_path: Path,
    dataset_name: str,
    description: str,
) -> dict[str, Any]:
    """Uploads clean local dataset to LangSmith with dynamic inputs/outputs mapping.

    Serves as Stage 2.5: Cloud alternative to local RAGAS evaluation (Stage 2).
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key or api_key.strip() in ("", "mock_key"):
        print("[LangSmith-upload] Error: LANGSMITH_API_KEY is not configured in environment.")
        return {
            "status": "error",
            "message": "LANGSMITH_API_KEY is not configured in environment",
        }

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
            "ground_truth_contexts": s.get("ground_truth_contexts")
            or s.get("metadata", {}).get("ground_truth_contexts", []),
            "synthesizer": s.get("synthesizer") or s.get("metadata", {}).get("synthesizer", ""),
        }
        for idx, s in enumerate(raw_samples)
    ]

    print(
        f"[LangSmith-upload] Ingesting {len(raw_samples)} examples into LangSmith dataset '{dataset_name}'..."
    )
    client.create_examples(
        inputs=inputs,
        outputs=outputs,
        metadata=metadata,
        dataset_id=dataset.id,
    )
    print(
        f"[LangSmith-upload] Successfully uploaded {len(raw_samples)} examples to LangSmith!"
    )

    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "dataset_name": dataset_name,
        "dataset_id": str(dataset.id),
        "source_dataset_file": dataset_path.name,
        "examples_uploaded": len(raw_samples),
        "status": "success",
    }
    return summary


def main() -> None:
    """CLI entrypoint for Stage 2.5 LangSmith dataset ingestion."""
    cfg = load_eval_config(CONFIG_PATH)
    ls_cfg = cfg.get("langsmith", {})

    parser = argparse.ArgumentParser(
        description="Upload evaluation dataset to LangSmith (Stage 2.5)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(CURRENT_DIR / ls_cfg.get("dataset_path", "1.dataset.json")),
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
        default=ls_cfg.get(
            "description",
            "Kanzi QA RAG evaluation test set synthesized via KnowledgeGraph",
        ),
        help="Description for the LangSmith dataset",
    )

    args = parser.parse_args()
    dataset_file = Path(args.dataset)

    summary = upload_dataset_to_langsmith(
        dataset_path=dataset_file,
        dataset_name=args.name,
        description=args.description,
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[LangSmith-main] Saved upload artifact to '{OUTPUT_FILE.name}'")


if __name__ == "__main__":
    main()
# endregion
