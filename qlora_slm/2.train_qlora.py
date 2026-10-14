#region Imports
import os
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from config import (
    BASE_MODEL_NAME,
    BATCH_SIZE,
    DATA_DIR,
    GRAD_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    MAX_SEQ_LENGTH,
    NUM_EPOCHS,
    OUTPUT_ADAPTER_DIR,
    TARGET_MODULES,
    WEIGHT_DECAY,
    get_logger,
)
#endregion

#region Scaffolding
logger = get_logger("QLoRA-Train", "2.train_qlora.log")


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


def create_training_args(output_dir: Path) -> TrainingArguments:
    """Encapsulates training argument scaffolding and optimization policies."""
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_steps=20,
        weight_decay=WEIGHT_DECAY,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=True,
        gradient_checkpointing=True,
        report_to="none",
    )
#endregion

#region Model
def setup_qlora_model(model_name: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Configures 4-bit NF4 quantization and attaches trainable LoRA adapters."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    kbit_model = prepare_model_for_kbit_training(base_model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(kbit_model, lora_config)
    peft_model.print_trainable_parameters()

    return peft_model, tokenizer
#endregion

#region Dataset
def prepare_datasets(tokenizer: PreTrainedTokenizerBase) -> tuple[Dataset, Dataset]:
    """Loads and tokenizes dataset with prompt token masking (-100 labels)."""
    train_file = DATA_DIR / "train.jsonl"
    val_file = DATA_DIR / "val.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {train_file}. Run 1.synth_dataset.py first."
        )

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "validation": str(val_file),
        },
    )

    def mask_prompt_labels(example: dict) -> dict:
        messages = example["messages"]
        prompt_text = tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
        )["input_ids"]

        # Mask prompt token loss so backprop penalizes only extracted keyword tokens
        labels = list(full_ids)
        for idx in range(min(len(prompt_ids), len(labels))):
            labels[idx] = -100

        return {"input_ids": full_ids, "labels": labels}

    tokenized_train = dataset["train"].map(
        mask_prompt_labels,
        remove_columns=dataset["train"].column_names,
    )
    tokenized_val = dataset["validation"].map(
        mask_prompt_labels,
        remove_columns=dataset["validation"].column_names,
    )

    return tokenized_train, tokenized_val
#endregion

#region Core
def train() -> None:
    """Executes end-to-end 4-bit QLoRA training workflow."""
    logger.info(
        f"[QLoRA-train] Starting training for {BASE_MODEL_NAME} "
        f"(CUDA={torch.cuda.is_available()})"
    )

    # 1. Model & Tokenizer Setup
    model, tokenizer = setup_qlora_model(BASE_MODEL_NAME)

    # 2. Dataset Tokenization & Prompt Masking
    tokenized_train, tokenized_val = prepare_datasets(tokenizer)

    # 3. Trainer Initialization
    trainer = Trainer(
        model=model,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8),
        args=create_training_args(OUTPUT_ADAPTER_DIR),
        callbacks=[TrainingLogCallback()],
    )

    # 4. Training Run & Checkpoint Serialization
    trainer.train()

    OUTPUT_ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(OUTPUT_ADAPTER_DIR))
    tokenizer.save_pretrained(str(OUTPUT_ADAPTER_DIR))
    logger.info(f"[QLoRA-train] Fine-tuned LoRA adapter saved to: {OUTPUT_ADAPTER_DIR}")
#endregion

#region Main
if __name__ == "__main__":
    train()
#endregion
