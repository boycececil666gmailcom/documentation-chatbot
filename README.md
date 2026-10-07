# documentation-chatbot

> Modular, production-grade documentation chatbot and RAG backend engine powered by LangGraph multi-agent orchestration, TypeSafe Jev intent routing, hybrid PGVector & BM25 retrieval, and automated System 2 self-critique reflection.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2.50+-1C3C3C?style=flat&logo=langchain&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=flat&logo=langchain&logoColor=white)
![OpenRouter](https://img.shields.io/badge/OpenRouter-DeepSeek_V4.1_Flash-6366F1?style=flat)
![TypeSafe Jev](https://img.shields.io/badge/TypeSafe-Jev_Decisions-blue?style=flat)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-PGVector_pg16-336791?style=flat&logo=postgresql&logoColor=white)
![FlashRank](https://img.shields.io/badge/FlashRank-Cross--Encoder-FF6F00?style=flat)
![Docker](https://img.shields.io/badge/Docker-Compose_Ready-2496ED?style=flat&logo=docker&logoColor=white)
![CloudBeaver](https://img.shields.io/badge/CloudBeaver-Web_GUI-1B72BA?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## End-to-End Pipeline Overview

```mermaid
---
config:
  theme: neutral
---
flowchart LR
    A["1. Data Ingestion<br/>(Crawl4AI & RAPTOR)"] --> B[("2. Knowledge Store<br/>(PGVector & BM25)")]
    B --> C["3. Intent Routing<br/>(TypeSafe Jev Decisions)"]
    C --> D["4. Parallel Retrieval<br/>(HyDE Dense + BM25 Sparse)"]
    D --> E["5. Neural Rerank<br/>(FlashRank Cross-Encoder)"]
    E --> F["6. Grounded Generation<br/>& System 2 Critique"]
    F --> G["7. Continuous LLMOps<br/>(LangSmith & Ragas)"]
```

---

## 1. Core Purpose & Business Value

Documentation-heavy platforms often struggle with escalating support ticket volumes, ungrounded assistant responses, and inaccurate technical guidance that compromises user trust. The **documentation-chatbot** engine transforms static product documentation into an autonomous, reliable conversational knowledge companion that answers user queries with strict fidelity to verified documentation.

- **Zero Hallucination & Brand Integrity**: Answers are strictly confined to verified internal documentation, ensuring customers receive dependable guidance without invented facts.
- **Domain Boundary Governance**: Automatically screens incoming inquiries to reject off-topic questions, keeping support conversations focused strictly on authorized business domains.
- **Self-Correcting Quality Assurance**: Employs an automated reflection loop that reviews draft responses before delivery, catching and revising incomplete or inaccurate answers before users see them.
- **Multi-Angle Knowledge Discovery**: Seamlessly balances exact terminology lookups with conceptual understanding, allowing users to find answers whether they know specific technical terms or describe their issue broadly.
- **Support Team Productivity**: Resolves repetitive customer inquiries autonomously around the clock, enabling technical support teams to dedicate time to high-value customer relationships.
- **Actionable Documentation Insights**: Identifies documentation gaps and user query trends through automated tracking, giving technical writers clear guidance on what content to improve.

---

## 2. Functional & Non-Functional Requirements

### Functional Requirements (FR)

- **FR-1 (Ingestion & Knowledge Indexing)**: Scrape web documentation using Crawl4AI, structure content into hierarchical heading-aware chunks (4-stage observable pipeline with `parent_id` & `child_ids`), and persist dual representations in PostgreSQL pgvector and an in-memory sparse keyword index.
- **FR-2 (Semantic Intent Routing)**: Evaluate user query intent and domain boundaries using the TypeSafe Jev decisions API with confidence thresholding (`min_confidence=0.20`), routing queries deterministically to `refuse`, `bm25`, `hyde`, or concurrent `hyde_bm25`.
- **FR-3 (Multi-Strategy Retrieval & Hierarchical Reranking)**: Perform parallel sparse keyword retrieval (BM25) and hypothetical document dense search (HyDE + Nemotron embeddings), dynamically expand parent context and child subtopics in the `context_expansion_by_hierarchy` node, re-score candidates with FlashRank cross-encoder, and enrich top ranked contexts with RAPTOR ancestor summaries (Layer 0/1 scope).
- **FR-4 (Grounded Synthesis & System 2 Critique Loop)**: Synthesize responses using DeepSeek models strictly bound to ranked contexts with inline topic citations `[Topic Name]`, followed by an automated reflection critique node that triggers stateful re-routing and revisions if ungrounded claims are detected (up to 3 retry attempts).

### Non-Functional Requirements (NFR)

- **NFR-1 (Performance & Latency)**: End-to-end query completion p95 latency <= 3.8s for standard queries; cached BM25 candidate lookup p99 latency <= 15ms.
- **NFR-2 (Reliability & Groundedness Accuracy)**: Evaluation benchmark faithfulness >= 0.96 (tested via Ragas suite with 75% zero-hallucination rate); 99.9% uptime for vector search and agent API.
- **NFR-3 (Security & Boundary Isolation)**: Environment-driven credential isolation via strict runtime validation, internal container network isolation, and stateless agent workflow execution.
- **NFR-4 (Scalability & Concurrency)**: Containerized deployment supporting concurrent Pregel execution in LangGraph, headless vector search across 5,000+ document chunks, and horizontal replica expansion.

---

## 3. LangGraph Agent Workflow Architecture

The core chatbot engine is implemented as a stateful, compiled LangGraph `StateGraph` (`src/agent_flow/graph.py`). The workflow handles query intake, domain governance, multi-strategy retrieval, neural reranking, grounded response generation, and an automated System 2 self-critique loop.

```mermaid
---
config:
  theme: neutral
---
flowchart TB
    Start(["__start__"]) --> Router["router<br/>(TypeSafe Jev Intent Routing)"]

    Router -.->|"routing_decision: hyde"| Hyde["hyde<br/>(Domain HyDE & Dense Search)"]
    Router -.->|"routing_decision: bm25"| BM25["bm25<br/>(Sparse Keyword Match)"]
    Router -.->|"routing_decision: refuse"| Refuse["refuse<br/>(Domain Boundary Refusal)"]

    Hyde --> ContextExpansion["context_expansion_by_hierarchy<br/>(Context Expansion by Hierarchy)"]
    BM25 --> ContextExpansion

    ContextExpansion --> Rerank["rerank<br/>(FlashRank Cross-Encoder)"]
    Rerank --> Generate["generate<br/>(Grounded Answer Synthesis)"]
    Generate --> Critique["critique<br/>(System 2 Quality Evaluation)"]
    Refuse --> Critique

    Critique -.->|"approved<br/>(is_critique_passed: true or retry >= 3)"| End(["__end__"])
    Critique -.->|"rejected<br/>(reflection feedback revision)"| Router

    classDef startEnd fill:#E2E8F0,stroke:#64748B,stroke-width:2px,color:#0F172A;
    classDef routerNode fill:#EDE9FE,stroke:#8B5CF6,stroke-width:2px,color:#4C1D95;
    classDef hydeNode fill:#ECFCCB,stroke:#84CC16,stroke-width:2px,color:#365314;
    classDef bm25Node fill:#EDE9FE,stroke:#8B5CF6,stroke-width:2px,color:#4C1D95;
    classDef refuseNode fill:#CFFAFE,stroke:#06B6D4,stroke-width:2px,color:#164E63;
    classDef contextExpansionNode fill:#FEF3C7,stroke:#F59E0B,stroke-width:2px,color:#78350F;
    classDef rerankNode fill:#EDE9FE,stroke:#8B5CF6,stroke-width:2px,color:#4C1D95;
    classDef generateNode fill:#CFFAFE,stroke:#06B6D4,stroke-width:2px,color:#164E63;
    classDef critiqueNode fill:#DBEAFE,stroke:#3B82F6,stroke-width:2px,color:#1E3A8A;

    class Start,End startEnd;
    class Router routerNode;
    class Hyde hydeNode;
    class BM25 bm25Node;
    class Refuse refuseNode;
    class ContextExpansion contextExpansionNode;
    class Rerank rerankNode;
    class Generate generateNode;
    class Critique critiqueNode;
```

### Node Responsibilities & Execution Mechanics

| Node | Execution Role & Technology | Input State Attributes | Output State Updates |
| :--- | :--- | :--- | :--- |
| `router` | **Intent Classification & Routing**<br/>Invokes TypeSafe Jev Decisions API (`typesafe/jev-1.13`) on OpenRouter to evaluate question intent and select optimal retrieval strategy (`refuse`, `bm25`, `hyde`, `hyde_bm25`). | `query` | `routing_decision`<br/>`bm25_docs: []`<br/>`hyde_docs: []` |
| `hyde` | **Hypothetical Document Expansion & Dense Search**<br/>Generates domain-injected 2-3 sentence hypothetical documentation excerpt using `hyde_llm`, then performs vector similarity search against PGVector (`raptor_chunks`). | `query` | `hypothetical_doc`<br/>`hyde_docs` |
| `bm25` | **Sparse Keyword Retrieval**<br/>Executes exact keyword matching against in-memory `BM25Retriever` constructed from PostgreSQL document chunks to reliably locate API names and identifiers. | `query` | `bm25_query`<br/>`bm25_docs` |
| `context_expansion_by_hierarchy` | **Context Expansion by Hierarchy**<br/>Merges sparse and dense candidates, deduplicates them, and executes real-time resolution of parent document overviews and related subtopics via `parent_id` and `child_ids` before FlashRank reranking. | `bm25_docs`<br/>`hyde_docs` | `expanded_docs` |
| `rerank` | **Neural Reranking & Context Enrichment**<br/>Computes semantic relevance scores of hierarchy-enriched chunks against query using FlashRank cross-encoder, and enriches top docs bottom-up with RAPTOR ancestor summaries. | `expanded_docs`<br/>`query` | `ranked_docs`<br/>`search_query` |
| `generate` | **Rank-Prioritized Grounded Synthesis**<br/>Synthesizes answers strictly grounded in ranked context chunks and macro-level RAPTOR parent summaries with inline citations `[Topic Name]`. Validates citations against actual retrieved topic keys. | `ranked_docs`<br/>`query`<br/>`critique_feedback` | `draft_response`<br/>`citations` |
| `refuse` | **Domain Boundary Enforcement**<br/>Generates a polite refusal response when the router classifies a query as completely off-topic relative to `CHATBOT_THEME`. | `query` | `draft_response`<br/>`citations: []` |
| `critique` | **System 2 Reflection & Groundedness Audit**<br/>Audit evaluator (DeepSeek) validating whether claims are factually supported by documentation context and citations are accurate. Sets `is_critique_passed`. | `draft_response`<br/>`ranked_docs`<br/>`query` | `is_critique_passed`<br/>`critique_feedback`<br/>`retry_count` |

---

## 4. Repository Structure

```text
documentation-chatbot/
├── dataset/
│   ├── crawl/                         # Web scraping & 4-stage observable ingestion pipeline
│   │   ├── 1.crawler.py & .json       # Scraped documentation source trees (Crawl4AI)
│   │   ├── 2.structure_hierarchy.py & .json # Page-level structured hierarchy (parent_id & child_ids)
│   │   ├── 3.split_chunks.py & .json  # Embeddable chunks (>3500 chars split) with parent/child links
│   │   ├── 4.ingest_pgvector.py & .json # PGVector collection wipe and batch embedding ingestion
│   │   └── crawler_config.json        # Crawling depth, chunk threshold, and rate limit configs
│   └── eval/                          # RAGAS automated benchmarking suite & test datasets
│       ├── 0.doc.json                 # Reference documentation excerpts
│       ├── 1.dataset.json             # Ground-truth Q&A evaluation dataset
│       ├── 2.run_eval_*.csv           # Historical benchmark run records across iterations
│       ├── JOURNAL.md                 # Evaluation milestones & metrics tracking journal
│       ├── eval_config.json           # Evaluation models, weights, and criteria thresholds
│       ├── pipeline.ipynb             # Automated Ragas evaluation benchmark runner
│       └── upload_langsmith.py        # LangSmith benchmark synchronization utility
├── src/
│   ├── agent_flow/                    # LangGraph autonomous workflow engine
│   │   ├── __init__.py
│   │   ├── edges.py                   # Conditional routing & retry reflection loop logic
│   │   ├── graph.py                   # StateGraph assembly, compilation, and direct execution
│   │   ├── nodes.py                   # Router, BM25, HyDE, Rerank, Generate, Refuse, Critique nodes
│   │   └── state.py                   # InputState and AgentState TypedDict definitions
│   ├── __init__.py
│   ├── config.py                      # Environment variable loader & strict validator
│   ├── llm_client.py                  # OpenRouter clients (DeepSeek, Nemotron, TypeSafe Jev, FlashRank)
│   ├── models.py                      # Pydantic structured schemas (RAGResponse, CritiqueResult, HyDE)
│   └── vector_db.py                   # PGVector store, in-memory BM25 index & hybrid retrieval
├── .env.example                       # Environment configuration template
├── docker-compose.yml                 # Multi-container stack (LangGraph, PGVector, CloudBeaver)
├── Dockerfile                         # Production multi-stage container image with uv
├── langgraph.json                     # LangGraph CLI server deployment configuration
├── pgadmin-servers.json               # Database client connection configuration
├── pyproject.toml                     # Dependencies, Ruff, Hatchling, and Pytest configuration
├── README.md                          # Enterprise documentation and architecture guide
└── uv.lock                            # Deterministic uv dependency lockfile
```

---

## 5. Getting Started & Operations

### Prerequisites

- [Docker](https://www.docker.com/) & Docker Compose
- [Python 3.11+](https://www.python.org/)
- [uv](https://docs.astral.sh/uv/) (recommended for local development)
- [OpenRouter API Key](https://openrouter.ai/)
- [LangSmith API Key](https://smith.langchain.com/)

### Environment Configuration

Copy the example environment file and supply your API credentials:

```bash
cp .env.example .env
```

| Environment Variable | Description | Default / Example Value |
| :--- | :--- | :--- |
| `OPENROUTER_API_KEY` | OpenRouter access token | `sk-or-v1-...` |
| `OPENROUTER_MODEL` | Primary text generation model | `deepseek/deepseek-v4.1-flash` |
| `OPENROUTER_JEV_MODEL` | TypeSafe Jev decisions model | `typesafe/jev-1.13` |
| `OPENROUTER_EMBED_MODEL` | Dense embedding model | `nvidia/nemotron-3-embed-1b:free` |
| `PGVECTOR_URL` | PostgreSQL connection string | `postgresql+psycopg://postgres:postgrespassword123@localhost:5432/documentation_chatbot` |
| `PGVECTOR_COLLECTION_NAME` | Target vector collection name | `raptor_chunks` |
| `CHATBOT_THEME` | Enforced documentation domain | `Kanzi QA Chatbot` |
| `LANGSMITH_TRACING` | Enable distributed trace collection | `true` |
| `LANGSMITH_API_KEY` | LangSmith observability token | `lsv2_pt_...` |
| `LANGSMITH_PROJECT` | LangSmith telemetry project name | `Enterprise-RAG-Engine` |

### Running via Docker Compose

Launch the complete containerized stack:

```bash
docker compose up -d
```

Service endpoints:
- **LangGraph Agent Server**: `http://localhost:2024`
- **CloudBeaver Database GUI**: `http://localhost:8978`
- **PGVector PostgreSQL**: `localhost:5432`

Inspect container status and logs:

```bash
docker compose ps
docker compose logs -f langgraph
```

### Local Development Workflow

Install dependencies with `uv`:

```bash
uv sync --all-groups
```

Run the LangGraph workflow directly via CLI:

```bash
python -m src.agent_flow.graph
```

Or start the LangGraph development server locally:

```bash
uv run langgraph dev --host 0.0.0.0 --port 2024
```

---

## 6. Benchmark Evolution & Quality Assurance

The system's retrieval and answer quality are continuously benchmarked against an authoritative evaluation dataset using Ragas metrics, tracked systematically in [`dataset/eval/JOURNAL.md`](file:///c:/Users/boyce/OneDrive/Desktop/documentation-chatbot/dataset/eval/JOURNAL.md).

| Iteration / Milestone | Key Architectural Enhancements | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Baseline** | Naive Dense Retrieval (top-3) + Fixed Chunking (500 chars) | `0.4267` | `0.3392` | `0.4937` | `0.3508` |
| **RAPTOR & Small-to-Big** | Markdown Heading Trees & Hierarchy Preservation | `0.5564` | `0.4191` | `0.6244` | `0.4411` |
| **Hybrid & FlashRank** | PGVector Dense + BM25 Sparse + FlashRank Reranker | `0.6595` | `0.4737` | `0.7255` | `0.5532` |
| **Agentic Critique Loop** | Stateful LangGraph Reflection Node & Retry Loop | `0.7589` | `0.5141` | `0.8347` | `0.6391` |
| **Reasoning & Intent Routing** | TypeSafe Jev Intent Routing & Token Allocation | `0.8331` | `0.5552` | **`0.9062`** | **`0.7214`** |
| **Domain HyDE & Guards** | Domain-Injected HyDE Expansion & False Rejection Guard | **`0.9648`** | **`0.5597`** | `0.8125` | `0.6471` |

To execute the benchmark suite and reproduce metrics:

```bash
# Open and run the evaluation notebook:
jupyter notebook dataset/eval/pipeline.ipynb
```

---

## 7. References & Technical Citations

- **Meta KDD Cup 2024 CRAG Challenge 1st Place Solution**:
  - Paper: [Winning Solution For Meta KDD Cup' 24 (arXiv:2410.00005)](https://arxiv.org/pdf/2410.00005)
  - Authors: Yanzhao Zhang, et al.
  - Core Insights: High-precision retrieval framework, tuned LLMs for hallucination minimization, and structured query routing in the Comprehensive RAG Benchmark Challenge.
- **NTT DOCOMO Engineering Blog - CRAG Top Solutions Analysis**:
  - Article: [RAG精度向上を目指したCRAGコンペ上位解法の紹介 - ドコモ開発者ブログ](https://nttdocomo-developers.jp/entry/2025/04/21/090000)
  - Author: 鈴木明作 (NTT DOCOMO R&D)
  - Core Insights: In-depth technical breakdown of the top-ranking solutions from the KDD Cup 2024 CRAG competition, covering multi-stage retrieval, reranking mechanisms, and self-critique answer grounding.
