# RAG Evaluation Journal

---

## 1. Benchmark Evolution Summary

| Date / Timestamp | Key Milestones & Technologies | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **2026-07-20** (`14:30:00`) | [Baseline] Naive Dense Retrieval + Fixed-length Chunking | `0.4267` | `0.3392` | `0.4937` | `0.3508` |
| **2026-07-28** (`10:15:00`) | [RAPTOR] Heading Hierarchy & Small-to-Big Chunking | `0.5564` | `0.4191` | `0.6244` | `0.4411` |
| **2026-08-05** (`16:45:00`) | [Hybrid & Rerank] Qdrant Hybrid + FlashRank Reranker | `0.6595` | `0.4737` | `0.7255` | `0.5532` |
| **2026-08-12** (`09:30:00`) | [Agentic Flow] LangGraph Autonomous Self-Critique Loop | `0.7589` | `0.5141` | `0.8347` | `0.6391` |
| **2026-08-19** (`01:00:25`) | [Reasoning & Intent] Reasoning Model Optimization + Implicit Intent Routing | `0.8331` | `0.5552` | **`0.9062`** | **`0.7214`** |
| **2026-08-19** (`12:47:35`) | [Domain HyDE & Classifier Fix] Domain-Injected HyDE & False Rejection Guard | **`0.9648`** | **`0.5597`** | `0.8125` | `0.6471` |

---

## 2. Iteration Log

### Week 1 (2026-07-20) — Baseline Setup
- **Architecture**: Naive dense retrieval (top-3) with fixed-length chunking (500 chars).
- **Bottlenecks**: Context boundary severance; low recall (`0.35`) and high hallucination rate (`Faithfulness: 0.43`).

### Week 2 (2026-07-28) — RAPTOR & Small-to-Big Chunking
- **Architecture**: Semantic Markdown chunking, Small-to-Big payload separation, GMM summary trees (RAPTOR).
- **Impact**: Faithfulness `0.43` -> `0.56` (+13.0%), Context Precision `0.49` -> `0.62` (+13.1%).

### Week 3 (2026-08-05) — Hybrid Search & FlashRank
- **Architecture**: Dense + FastEmbed sparse hybrid retrieval with FlashRank cross-encoder reranking (top-10 -> top-5).
- **Impact**: Context Precision `0.62` -> `0.73` (+10.1%), Context Recall `0.44` -> `0.55` (+11.2%).

### Week 4 (2026-08-12) — Agentic Critique Loop
- **Architecture**: Stateful LangGraph workflow with a self-critique reflection node and automated retry loop.
- **Impact**: Faithfulness `0.66` -> `0.76` (+9.9%), Context Precision `0.73` -> `0.83` (+10.9%).

### Week 5 (2026-08-19) — Reasoning & Classification Tuning
- **Architecture**: Reasoning token allocation optimization (`effort: medium`, `max_tokens: 16384`) and intent classifier prompt refinement for implicit queries.
- **Impact**: Context Precision `0.83` -> `0.91` (+7.2%), Context Recall `0.64` -> `0.72` (+8.2%).

### Week 5.1 (2026-08-19) — Domain HyDE & Evaluation Pipeline
- **Architecture**: Domain-injected HyDE for concise queries, false-rejection safeguards in classifier, and automated 32-sample RAGAS evaluation runner ([`pipeline.ipynb`](file:///c:/Users/boyce/OneDrive/Desktop/documentation-chatbot/dataset/eval/pipeline.ipynb)).
- **Impact**: Faithfulness surged to `0.96` (+13.2%, 75% perfect `1.0`); Context Precision at `0.81`.
