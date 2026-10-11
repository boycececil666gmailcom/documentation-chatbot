# region Evaluation and Benchmarking
import json
import re
import sys
import time
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import (
    BASE_MODEL_NAME,
    DATA_DIR,
    OUTPUT_ADAPTER_DIR,
    SYSTEM_PROMPT,
    get_logger,
)

logger = get_logger("EvalBenchmark", "eval_benchmark.log")


def calculate_keyword_metrics(predicted: str, target: str) -> tuple[float, float]:
    """Calculates recall and precision against target keywords."""
    pred_tokens = set(re.findall(r"\b[A-Za-z0-9_#::\.-]+\b", predicted.lower()))
    target_tokens = set(re.findall(r"\b[A-Za-z0-9_#::\.-]+\b", target.lower()))

    if not target_tokens:
        return 1.0, 0.0

    hits = len(pred_tokens.intersection(target_tokens))
    recall = hits / len(target_tokens)
    precision = hits / len(pred_tokens) if pred_tokens else 0.0
    return recall, precision


def run_benchmark(max_eval_samples: int = 50):
    """Runs automated evaluation on validation set measuring recall, precision, and latency."""
    val_file = DATA_DIR / "val.jsonl"
    if not val_file.exists():
        raise FileNotFoundError(f"Validation dataset not found at {val_file}")

    logger.info(f"[Benchmark-init] Loading base model: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if OUTPUT_ADAPTER_DIR.exists() and (OUTPUT_ADAPTER_DIR / "adapter_config.json").exists():
        logger.info(f"[Benchmark-init] Loading fine-tuned LoRA adapter from: {OUTPUT_ADAPTER_DIR}")
        model = PeftModel.from_pretrained(base_model, str(OUTPUT_ADAPTER_DIR))
    else:
        logger.info("[Benchmark-init] No adapter found. Evaluating zero-shot base model as baseline.")
        model = base_model

    model.eval()

    with open(val_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f][:max_eval_samples]

    recalls, precisions, latencies = [], [], []

    logger.info(f"[Benchmark-eval] Running evaluation on {len(records)} test samples...")
    for i, item in enumerate(records, 1):
        user_msg = item["messages"][1]["content"]
        target_kw = item["messages"][2]["content"]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        start_t = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - start_t

        gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        rec, prec = calculate_keyword_metrics(pred_text, target_kw)
        recalls.append(rec)
        precisions.append(prec)
        latencies.append(elapsed)

        if i <= 5:
            logger.info(
                f"[Benchmark-sample] Sample {i} | Time: {elapsed:.3f}s | "
                f"Rec: {rec:.2f} | Prec: {prec:.2f} | Query: {user_msg} | Target: {target_kw} | Pred: {pred_text}"
            )

    avg_recall = sum(recalls) / len(recalls)
    avg_precision = sum(precisions) / len(precisions)
    avg_latency = sum(latencies) / len(latencies)

    logger.info("================ BENCHMARK RESULTS ================")
    logger.info(f"Total Evaluated: {len(records)} samples")
    logger.info(f"Average Recall:    {avg_recall * 100:.2f}%")
    logger.info(f"Average Precision: {avg_precision * 100:.2f}%")
    logger.info(f"Average Latency:   {avg_latency * 1000:.1f} ms / query")
    logger.info("====================================================")


if __name__ == "__main__":
    run_benchmark()
# endregion
