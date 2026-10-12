# region Configuration
import os
from pathlib import Path

from dotenv import load_dotenv

# Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"

# Load environment variables from project root .env
load_dotenv(PROJECT_ROOT / ".env")

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Model Settings
# Configurable base model name on Hugging Face or local path
BASE_MODEL_NAME = os.getenv("SLM_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
OUTPUT_ADAPTER_DIR = OUTPUT_DIR / "qlora_adapter"
OUTPUT_MERGED_DIR = OUTPUT_DIR / "merged_model"

# Quantization & LoRA Hyperparameters
LORA_R = int(os.getenv("LORA_R", "16"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# Training Hyperparameters
BATCH_SIZE = int(os.getenv("SLM_BATCH_SIZE", "4"))
GRAD_ACCUMULATION_STEPS = int(os.getenv("SLM_GRAD_ACCUM", "4"))
LEARNING_RATE = float(os.getenv("SLM_LR", "2e-4"))
NUM_EPOCHS = int(os.getenv("SLM_EPOCHS", "3"))
MAX_SEQ_LENGTH = int(os.getenv("SLM_MAX_SEQ_LEN", "512"))
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01

# System Prompt for Distillation Task
SYSTEM_PROMPT = (
    "Extract only key technical terms, identifiers, API names, and core keywords "
    "from the user query for sparse keyword search. "
    "Remove conversational filler, questions, and stop words. "
    "Output ONLY space-separated keywords without punctuation or quotes."
)


def get_logger(module_name: str, log_filename: str):
    """Initializes and returns a logger with dual console and file outputs."""
    import logging

    logger = logging.getLogger(module_name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(LOGS_DIR / log_filename, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger
# endregion
