# region Imports & Configuration
import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import httpx

# endregion


# region Data Ingestion Logic
def load_chunks(file_path: Path) -> list[dict]:
    """Reads chunk records from a JSON or JSON.GZ file."""
    print("\n\033[1;96m========================================================\033[0m")
    print(
        f"\033[1;92m>>> [1/3] [{os.path.basename(__file__)}] Reading structured JSON dataset: {file_path.name}\033[0m"
    )
    print("\033[1;96m========================================================\033[0m\n")

    opener = gzip.open if file_path.name.endswith(".gz") else open
    with opener(file_path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    chunks = (
        data.get("chunks", [])
        if isinstance(data, dict)
        else (data if isinstance(data, list) else [])
    )
    print(f"Successfully loaded {len(chunks):,} chunk records from dataset.")
    return chunks


def ingest_chunks(chunks: list[dict], endpoint_url: str, batch_size: int = 50):
    """Posts document chunks to the Vector DB ingestion HTTP endpoint."""
    print("\n\033[1;96m========================================================\033[0m")
    print(
        f"\033[1;92m>>> [2/3] [{os.path.basename(__file__)}] Connecting to Vector DB Ingestion Endpoint: {endpoint_url}\033[0m"
    )
    print("\033[1;96m========================================================\033[0m\n")

    success, failures = 0, 0
    total = len(chunks)

    with httpx.Client(timeout=60.0) as client:
        for idx, chunk in enumerate(chunks, start=1):
            text = (
                chunk.get("child_content")
                or chunk.get("content")
                or chunk.get("text")
                or ""
            )
            if not text:
                continue

            metadata = {
                k: json.dumps(v, ensure_ascii=False)
                if isinstance(v, (list, dict))
                else str(v)
                for k, v in (chunk.get("metadata") or {}).items()
                if v is not None
            }

            try:
                res = client.post(
                    endpoint_url, json={"text": text, "metadata": metadata}
                )
                if res.status_code == 200:
                    success += 1
                else:
                    failures += 1
                    print(
                        f"Failed chunk {idx}/{total}: HTTP {res.status_code} - {res.text}"
                    )
            except Exception as e:
                failures += 1
                print(f"Error sending chunk {idx}/{total}: {e}")

            if idx % batch_size == 0 or idx == total:
                print(
                    f"Progress: [{idx}/{total}] chunks processed (Success: {success}, Failures: {failures})"
                )

    print("\n\033[1;96m========================================================\033[0m")
    print(
        f"\033[1;92m>>> [3/3] [{os.path.basename(__file__)}] Ingestion Complete: {success}/{total} Chunks Successfully Ingested into VDB\033[0m"
    )
    print("\033[1;96m========================================================\033[0m\n")


# endregion


# region CLI Entry Point
def main():
    try:
        parser = argparse.ArgumentParser(
            description="Ingest JSON chunks into Qdrant Vector DB."
        )
        parser.add_argument(
            "--input",
            "-i",
            default="preprocessing-pipeline/rag_chunks.json",
            help="Input dataset path",
        )
        parser.add_argument(
            "--endpoint",
            "-e",
            default=os.getenv("INGEST_ENDPOINT", "http://localhost:8000/ingest/vector"),
            help="VDB Ingest Endpoint",
        )
        parser.add_argument(
            "--batch-size",
            "-b",
            type=int,
            default=50,
            help="Batch progress logging size",
        )
        args = parser.parse_args()

        input_path = Path(args.input)
        chunks = load_chunks(input_path)
        ingest_chunks(chunks, args.endpoint, args.batch_size)
    except Exception as err:
        print(f"Execution Error: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
# endregion
