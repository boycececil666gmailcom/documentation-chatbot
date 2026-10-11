# region Export and Model Merge
import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    BASE_MODEL_NAME,
    OUTPUT_ADAPTER_DIR,
    OUTPUT_MERGED_DIR,
    SYSTEM_PROMPT,
    get_logger,
)

logger = get_logger("ExportMerge", "export_merge.log")


def merge_and_export():
    """Merges trained LoRA adapter weights into base model and generates Ollama Modelfile."""
    if not OUTPUT_ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"LoRA adapter directory not found at {OUTPUT_ADAPTER_DIR}. Train first using 2.train_qlora.py."
        )

    logger.info(f"[Export-merge] Loading base model: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)

    # Load base model in float16 for clean weight merging
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cpu",  # CPU merge avoids VRAM spikes during full model un-quantization
        trust_remote_code=True,
    )

    logger.info(f"[Export-merge] Applying LoRA adapter from: {OUTPUT_ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base_model, str(OUTPUT_ADAPTER_DIR))

    logger.info("[Export-merge] Merging weights into base model...")
    merged_model = model.merge_and_unload()

    OUTPUT_MERGED_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"[Export-merge] Saving merged 16-bit model to: {OUTPUT_MERGED_DIR}")
    merged_model.save_pretrained(str(OUTPUT_MERGED_DIR), safe_serialization=True)
    tokenizer.save_pretrained(str(OUTPUT_MERGED_DIR))

    # Generate Ollama Modelfile
    modelfile_path = OUTPUT_MERGED_DIR / "Modelfile"
    modelfile_content = f"""# Ollama Modelfile for Fine-Tuned Kanzi Keyword Distillation SLM
FROM {OUTPUT_MERGED_DIR}

PARAMETER temperature 0.0
PARAMETER top_p 0.95
PARAMETER num_predict 32
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

SYSTEM "{SYSTEM_PROMPT}"
"""
    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    logger.info(f"[Export-merge] Ollama Modelfile generated at: {modelfile_path}")
    logger.info(f"[Export-merge] Register command: ollama create kanzi-slm-extractor -f {modelfile_path}")


if __name__ == "__main__":
    merge_and_export()
# endregion
