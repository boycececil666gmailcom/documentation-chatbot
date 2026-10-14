# Evaluation & LLMOps: Ragas Metrics & LangSmith Observability

This document details the evaluation methodology, automated benchmark pipelines, and distributed observability stack governing system quality and hallucination prevention.

---

## 1. The Need for Automated Quantitative Evaluation

Subjective "vibe checks" are insufficient for enterprise RAG deployments. Without automated quantitative benchmarking:
- Changes to chunking algorithms or prompt templates risk silent regressions in answer grounding.
- Performance gains in one area (e.g., higher recall) often introduce hallucinations in another.
- Engineers cannot pinpoint whether a failure originated in retrieval, reranking, or generation.

To resolve this, the system implements automated testing using **Ragas** and continuous distributed tracing via **LangSmith**.

---

## 2. Ragas Evaluation Metrics Suite

Implemented in `dataset/eval/pipeline.ipynb` using reference datasets (`dataset/eval/1.dataset.json`), the Ragas suite scores four orthogonal dimensions:

```mermaid
quadrantChart
    title RAG Quality Evaluation Quadrants
    x-axis Low Retrieval Precision --> High Retrieval Precision
    y-axis Low Generation Faithfulness --> High Generation Faithfulness
    quadrant-1 Production Ready (Grounded & Accurate)
    quadrant-2 Hallucination Risk (Irrelevant Context Generated Well)
    quadrant-3 Catastrophic Failure (Missed Context & Hallucination)
    quadrant-4 Underutilized Context (Good Retrieval, Weak Generation)
```

### 1. Faithfulness (Groundedness Score)
- **Definition**: Measures the factual consistency of the generated response against the retrieved context passages.
- **Formula**:
  $$\text{Faithfulness} = \frac{|\text{Number of Claims Supported by Context}|}{|\text{Total Claims Made in Response}|}$$
- **Current Milestone**: Reached **`0.9652`** (96.5% of claims factually verified) via System 2 critique gating.

### 2. Answer Relevancy
- **Definition**: Evaluates whether the generated response directly addresses the user query without rambling or introducing off-topic tangents.
- **Formula**: Computes the mean cosine similarity between the original user query and synthetic queries generated backward from the response:
  $$\text{Answer Relevancy} = \frac{1}{N} \sum_{i=1}^N \text{Sim}(q, q_i^{synth})$$

### 3. Context Precision
- **Definition**: Evaluates whether relevant signal chunks are ranked at the top of the context window rather than buried at the bottom.
- **Milestone Evolution**: Rose from `0.4937` to **`0.9280`** after introducing the FlashRank cross-encoder.

### 4. Context Recall
- **Definition**: Measures whether all ground-truth facts required to answer the question were successfully retrieved from the knowledge store.
- **Formula**:
  $$\text{Context Recall} = \frac{|\text{Ground-Truth Sentences Attributed to Retrieved Context}|}{|\text{Total Sentences in Ground-Truth Answer}|}$$

---

## 3. LangSmith Distributed Tracing & Observability

### Distributed Tracing Pipeline
All graph transitions and model invocations are instrumented with LangSmith tracing (`@traceable` in `src/llm_client.py`):

```text
Run ID: [f8a1...]
├── router (TypeSafe Jev: "hyde_bm25", conf=0.88)
├── [Branch A] hyde (Structured passage generation) -> pgvector_search
├── [Branch B] local_slm_keyword_extraction -> bm25_search
├── context_expansion_by_hierarchy (Parent/Child join)
├── rerank (FlashRank cross-encoder: 15 docs -> 5 docs)
├── generate (Rank-prioritized synthesis with inline citations)
└── critique (System 2 check: passed=True)
```

### Auditing Failure Trajectories
When an evaluation question scores below threshold during automated benchmarks:
1. The test runner tags the execution run ID in LangSmith.
2. Engineers inspect the exact execution span to isolate the root cause:
   - *Was the ground truth chunk missed by BM25 and PGVector?* -> Adjust crawler chunk split size or embedding model.
   - *Was the chunk retrieved but dropped by FlashRank?* -> Adjust cross-encoder top-N threshold.
   - *Did the generator fail to cite the retrieved context?* -> Refine generation prompt constraints.

### Targeted Prompt Engineering Iterations
Rather than modifying prompts arbitrarily, prompt refinements are driven strictly by failure logs:
- Observed recurring formatting errors when generating citations -> Added explicit few-shot citation schemas in `src/agent_flow/nodes.py`.
- Observed excessive conversational preambles in local SLM output -> Formulated targeted QLoRA synthetic training data to enforce clean space-separated tokens.
