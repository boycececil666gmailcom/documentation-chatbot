#region Synthesizer
import json
import random
import re
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config import DATA_DIR, PROJECT_ROOT, SYSTEM_PROMPT, get_logger

logger = get_logger("DatasetSynth", "1.synth_dataset.log")

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "if", "in", "into", "is",
    "isn't", "it", "it's", "its", "itself", "just", "me", "more", "most", "mustn't",
    "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "she", "should", "shouldn't", "so", "some", "such", "than", "that", "that's",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "were", "weren't", "what", "what's", "when",
    "where", "which", "while", "who", "whom", "why", "with", "won't", "would",
    "wouldn't", "you", "your", "yours", "yourself", "yourselves", "tell", "please",
    "want", "know", "explain", "help", "need", "like",
    # Procedural and Action Stop Words
    "using", "use", "creates", "creating", "create", "adds", "adding", "add",
    "manages", "managing", "manage", "imports", "importing", "import",
    "deploys", "deploying", "deploy", "sets", "setting", "set", "gets",
    "getting", "get", "applies", "applying", "apply"
}


def clean_to_keywords(text: str) -> str:
    """Extracts strictly valid technical keywords from a text line."""
    # Split merged camelCase tokens like theSorting -> the Sorting
    normalized = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    tokens = re.findall(r"\b[A-Za-z0-9_#::\.-]+\b", normalized)
    filtered = [t for t in tokens if t.lower() not in STOP_WORDS and len(t) > 1]
    
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for t in filtered:
        t_clean = t.strip(".,;:?!-")
        if t_clean.lower() not in seen and len(t_clean) > 1:
            seen.add(t_clean.lower())
            unique.append(t_clean)
    return " ".join(unique)


def build_samples_from_eval_dataset(dataset_path: Path) -> list[dict]:
    """Generates distillation pairs from existing evaluation queries."""
    if not dataset_path.exists():
        logger.warning(f"[DatasetSynth-eval] Warning: {dataset_path} not found.")
        return []

    with open(dataset_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    samples = []
    for item in records:
        q = item.get("question", "").strip()
        if not q:
            continue
        kw = clean_to_keywords(q)
        if kw and len(kw.split()) >= 2:
            samples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": kw},
                ]
            })
    logger.info(f"[DatasetSynth-eval] Extracted {len(samples)} samples from eval dataset.")
    return samples


def build_samples_from_hierarchy(hierarchy_path: Path) -> list[dict]:
    """Generates synthetic question-keyword pairs from document hierarchy nodes."""
    if not hierarchy_path.exists():
        logger.warning(f"[DatasetSynth-hierarchy] Warning: {hierarchy_path} not found.")
        return []

    with open(hierarchy_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    templates = [
        "How do I use {} in Kanzi?",
        "Can you explain how to configure {}?",
        "Where can I find the documentation for {}?",
        "What are the best practices for {} in Kanzi Studio?",
        "How does {} affect application performance?",
        "Could you please tell me how to implement {}?",
        "How to set up and manage {}?",
    ]

    samples = []
    for node in nodes:
        title = node.get("title", "").strip()
        # Clean title noise (e.g. "Part 1/4", links)
        title_clean = re.sub(r"\(Part \d+/\d+\)", "", title).strip()
        if not title_clean or len(title_clean) < 3 or title_clean.startswith("Untitled"):
            continue

        target_kw = clean_to_keywords(title_clean)
        if not target_kw:
            continue

        for tmpl in templates:
            user_q = tmpl.format(title_clean)
            samples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_q},
                    {"role": "assistant", "content": target_kw},
                ]
            })

    logger.info(f"[DatasetSynth-hierarchy] Synthesized {len(samples)} samples from hierarchy nodes.")
    return samples


def main() -> None:
    """Compiles training and validation datasets into JSONL files."""
    eval_path = PROJECT_ROOT / "dataset" / "eval" / "1.dataset.json"
    hier_path = PROJECT_ROOT / "dataset" / "crawl" / "2.structure_hierarchy.json"

    samples = []
    samples.extend(build_samples_from_eval_dataset(eval_path))
    samples.extend(build_samples_from_hierarchy(hier_path))

    # Deduplicate by user query
    deduped = {}
    for s in samples:
        q = s["messages"][1]["content"]
        if q not in deduped:
            deduped[q] = s
    all_samples = list(deduped.values())
    random.seed(42)
    random.shuffle(all_samples)

    if not all_samples:
        raise ValueError("[DatasetSynth-main] No training samples could be synthesized.")

    # Split 85% train, 15% validation
    split_idx = int(len(all_samples) * 0.85)
    train_data = all_samples[:split_idx]
    val_data = all_samples[split_idx:]

    train_file = DATA_DIR / "train.jsonl"
    val_file = DATA_DIR / "val.jsonl"

    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_file, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"[DatasetSynth-main] Exported {len(train_data)} train samples to {train_file}")
    logger.info(f"[DatasetSynth-main] Exported {len(val_data)} val samples to {val_file}")


if __name__ == "__main__":
    main()
# endregion
