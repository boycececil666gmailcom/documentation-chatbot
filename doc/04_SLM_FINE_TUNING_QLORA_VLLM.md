# SLM Adaptation & Serving: 4-Bit QLoRA & vLLM Architecture

This document covers the end-to-end small language model (SLM) pipeline: fine-tuning domain keyword distillation models with 4-bit QLoRA and serving them at scale using vLLM's PagedAttention engine.

---

## 1. Why Fine-Tune an SLM for Keyword Distillation?

BM25 relies on exact term frequency and inverse document frequency (TF-IDF). Passing conversational user inputs (e.g., *"Could you please show me how to bind button events in Kanzi?"*) directly into BM25 degrades retrieval quality because conversational stop-words (`could`, `you`, `please`, `show`) dilute scoring weights.

General-purpose proprietary LLMs (e.g., GPT-4 or Claude) can extract keywords, but introduce drawbacks:
1. **Conversational Formatting Pollution**: General LLMs frequently output preamble text (*"Here are your keywords:"*) and markdown formatting, breaking automated parser pipelines.
2. **Cost and Latency**: Paying cloud API round-trips ($500\text{ms}–1200\text{ms}$) for a trivial keyword extraction task is cost-prohibitive.

The solution in this repository is an in-house fine-tuned SLM (`kanzi-slm-extractor`) based on Qwen2.5, trained to perform zero-noise keyword distillation deterministically.

---

## 2. QLoRA 4-Bit Adaptation Pipeline (`qlora_slm/`)

### Mathematical Foundation
Standard LoRA freezes the pre-trained weights $W_0 \in \mathbb{R}^{d \times k}$ and injects trainable rank-decomposition matrices:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and rank $r \ll \min(d, k)$.

**QLoRA** introduces three memory optimizations:
1. **NF4 Quantization (NormalFloat 4)**: An information-theoretically optimal quantile quantization data type for normally distributed neural network weights.
2. **Double Quantization (DQ)**: Quantizes the quantization constants themselves, saving $\approx 0.37$ bits per parameter.
3. **Paged Optimizers**: Uses CUDA Unified Memory to prevent out-of-memory errors during gradient checkpointing peaks.

### Training Configuration (`qlora_slm/config.py`)
- **Base Model**: `Qwen/Qwen2.5-3B-Instruct` (or 1.5B)
- **LoRA Parameters**: $r = 16$, $\alpha = 32$, $\text{Dropout} = 0.05$
- **Target Projections**: All linear attention and MLP layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Quantization**: 4-bit NF4 with bfloat16 computation dtype

### Pipeline Stages
1. `1.synth_dataset.py`: Generates paired datasets mapping raw conversational queries to ground-truth space-separated technical keywords.
2. `2.train_qlora.py`: Trains PEFT adapter using HuggingFace SFTTrainer with gradient accumulation.
3. `3.eval_and_benchmark.py`: Evaluates keyword token recall, precision, and generation latency.
4. `4.export_and_merge.py`: Merges adapter weights back into base 16-bit float weights and exports to GGUF format for local container deployment.

---

## 3. High-Throughput Serving with vLLM

### The Serving Bottleneck: Memory Fragmentation
Standard HuggingFace and PyTorch inference pipelines allocate contiguous memory for Key-Value (KV) caches based on the maximum sequence length. This leads to:
- **Internal Fragmentation**: Pre-allocated memory for tokens that are never generated.
- **External Fragmentation**: Memory gaps between requests prevent batch size scaling.

### PagedAttention
vLLM resolves memory fragmentation by managing the KV cache in the same way an operating system manages virtual memory via paging:

```text
Logical KV Cache:     [ Block 0 ] [ Block 1 ] [ Block 2 ]
                           │           │           │
                           ▼           ▼           ▼
Physical GPU Memory:  [ Frame 14] [ Frame 02] [ Frame 89] (Non-contiguous)
```

- Blocks are allocated dynamically on-demand during autoregressive token generation.
- Reduces memory waste from $>60\%$ to under $4\%$, enabling up to $4\times$ larger concurrent batch sizes.

### Continuous Batching (Iteration-Level Scheduling)
Traditional serving batches requests at the request level, meaning the entire batch must wait until the slowest sequence finishes generating. 

vLLM implements continuous batching, scheduling execution at the **iteration (token) level**. When one sequence finishes, its memory blocks are released immediately, and a new incoming prompt is merged into the running batch without pausing other requests.

This ensures the local keyword extractor maintains sub-200ms latency even under concurrent multi-user load.
