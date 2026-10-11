# region QLoRA Training Engine
import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTTrainer

from config import (
    BASE_MODEL_NAME,
    BATCH_SIZE,
    DATA_DIR,
    GRAD_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LOGS_DIR,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_SEQ_LENGTH,
    NUM_EPOCHS,
    OUTPUT_ADAPTER_DIR,
    TARGET_MODULES,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    get_logger,
)

logger = get_logger("QLoRA-Train", "train_qlora.log")


class TrainingLogCallback(TrainerCallback):
    """Logs training steps and metrics to both console and file."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            loss = logs.get("loss", "N/A")
            eval_loss = logs.get("eval_loss", "N/A")
            lr = logs.get("learning_rate", "N/A")
            logger.info(
                f"[QLoRA-train] Step {step} | Loss: {loss} | Eval Loss: {eval_loss} | LR: {lr}"
            )


def train():
    """Executes 4-bit QLoRA fine-tuning for technical keyword extraction."""
    train_file = DATA_DIR / "train.jsonl"
    val_file = DATA_DIR / "val.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {train_file}. Run 1.synth_dataset.py first."
        )

    logger.info(f"[QLoRA-train] Target base model: {BASE_MODEL_NAME}")
    logger.info(f"[QLoRA-train] Device available: CUDA={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"[QLoRA-train] GPU device: {torch.cuda.get_device_name(0)}")

    # 1. 4-bit Quantization Configuration (BitsAndBytes)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # 2. Tokenizer & Model Initialization
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA Configuration
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Dataset Loading & Preprocessing
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "validation": str(val_file),
        },
    )

    def formatting_prompts_func(example):
        output_texts = []
        for msgs in example["messages"]:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            output_texts.append(text)
        return output_texts

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_ADAPTER_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        logging_dir=str(LOGS_DIR / "runs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=not torch.cuda.is_bf16_supported() and torch.cuda.is_available(),
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        gradient_checkpointing=True,
        report_to="none",
    )

    # 6. SFTTrainer Execution
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora_config,
        formatting_func=formatting_prompts_func,
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[TrainingLogCallback()],
    )

    logger.info("[QLoRA-train] Starting model training...")
    trainer.train()

    # 7. Save Adapter & Tokenizer
    OUTPUT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(OUTPUT_ADAPTER_DIR))
    tokenizer.save_pretrained(str(OUTPUT_ADAPTER_DIR))
    logger.info(f"[QLoRA-train] Fine-tuned LoRA adapter successfully saved to: {OUTPUT_ADAPTER_DIR}")


if __name__ == "__main__":
    train()
# endregion
