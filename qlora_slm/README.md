# QLoRA Technical Keyword Distillation SLM

## 1. Overview
This module houses the end-to-end QLoRA (4-bit quantized Low-Rank Adaptation) fine-tuning pipeline for the Local SLM (`kanzi-slm-extractor` / `Qwen/Qwen2.5-3B-Instruct`).
The dedicated downstream task is **Extractive Keyword Distillation for BM25 Sparse Search**:
- **Input**: Conversational/complex user query (e.g., "Kanziで3Dノードのプロパティを変更するC++ APIコードを教えてください")
- **Task**: Strip syntax noise, questions, and conversational fillers without hallucinating terms.
- **Output**: Space-separated pure technical keywords (e.g., `Kanzi 3D Node Property C++ API`).

---

## 2. Hardware Budget (8GB VRAM Constraint)

| Stage | Memory Allocation | Technique |
| :--- | :--- | :--- |
| **Base Model (4-bit NF4)** | ~2.2 - 2.5 GB | `bitsandbytes` 4-bit NF4 double quantization |
| **LoRA Trainable Parameters** | ~0.3 GB | Rank $r=16$, $\alpha=32$ across all linear projections |
| **Activation & Optimizer** | ~4.5 GB | `paged_adamw_8bit` + Gradient Checkpointing + Batch Size 4 |
| **Total Peak Training VRAM** | **~7.1 GB** | Fully contained within **8GB GPU memory** (RTX 4060 Laptop) |

---

## 3. Directory Structure

```text
qlora_slm/
├── config.py                 # Central hyperparameter & path configurations
├── 1.synth_dataset.py        # Dataset synthesizer (Train: 2,651 / Val: 468 samples)
├── 2.train_qlora.py          # 4-bit NF4 QLoRA training engine with transformers.Trainer
├── 3.eval_and_benchmark.py   # Benchmark measuring Recall, Precision, and Token Latency
├── 4.export_and_merge.py     # LoRA adapter merging (FP16) & Modelfile generation
├── Modelfile                 # Version-controlled Ollama registration template
├── data/
│   ├── train.jsonl           # ChatML formatted training data
│   └── val.jsonl             # ChatML formatted validation data
├── outputs/
│   ├── kanzi_slm.gguf        # Quantized Q8_0 GGUF binary (3.28 GB)
│   ├── Modelfile.gguf        # Ollama container configuration
│   ├── qlora_adapter/        # Trained LoRA checkpoints and adapter config
│   └── merged_model/         # Merged full FP16 model (6.17 GB)
└── logs/                     # Training & benchmark runtime logs
```

---

## 4. Execution Workflow

### Step 1: Synthesize High-Quality Technical Dataset
Extracts technical terms, Kanzi C++ API patterns, and query pairs from existing project datasets (`dataset/eval/1.dataset.json` and `dataset/crawl/2.structure_hierarchy.json`):
```bash
uv run python qlora_slm/1.synth_dataset.py
```

### Step 2: Run 4-bit QLoRA Training
Trains adapter weights while freezing 4-bit base model parameters (Triton-safe for Windows):
```bash
uv run python qlora_slm/2.train_qlora.py
```

### Step 3: Evaluate Precision, Recall & Latency
Evaluates keyword retention rate and inference speed:
```bash
uv run python qlora_slm/3.eval_and_benchmark.py
```

### Step 4: Merge Adapter & Register to Ollama
Merges LoRA adapter into FP16 model, converts to GGUF, and registers with Ollama container:
```bash
uv run python qlora_slm/4.export_and_merge.py
docker exec -i documentation-chatbot-ollama ollama create kanzi-slm-extractor -f /outputs/Modelfile.gguf
```
