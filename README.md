# documentation-chatbot

> Modular, enterprise-grade Documentation Chatbot and Retrieval-Augmented Generation (RAG) backend engine template with multi-agent orchestration, hybrid vector search, and GraphRAG entity-relationship reasoning.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=flat&logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-1C3C3C?style=flat&logo=langchain&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3+-1C3C3C?style=flat&logo=langchain&logoColor=white)
![OpenRouter](https://img.shields.io/badge/OpenRouter-DeepSeek_V4_Flash-6366F1?style=flat)
![LLMLingua-2](https://img.shields.io/badge/LLMLingua--2-Prompt_Compression-8A2BE2?style=flat)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-PGVector-336791?style=flat&logo=postgresql&logoColor=white)
![Neo4j](https://img.shields.io/badge/Neo4j-Graph_DB-008CC1?style=flat&logo=neo4j&logoColor=white)
![Terraform](https://img.shields.io/badge/Terraform-IaC-7B42BC?style=flat&logo=terraform&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=flat&logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-Orchestration-326CE5?style=flat&logo=kubernetes&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## End-to-End Pipeline Architecture

```mermaid
---
config:
  theme: neutral
---
flowchart LR
    A["1. Data Ingestion<br/>(Crawl4AI & RAPTOR)"] --> B[("2. Hybrid Knowledge Store<br/>(PGVector & Neo4j)")]
    B --> C["3. Multi-Agent Retrieval<br/>(HyDE + Dense + BM25)"]
    C --> D["4. Neural Rerank<br/>(FlashRank Cross-Encoder)"]
    D --> E["5. Context Compression<br/>(LLMLingua-2 Compactor)"]
    E --> F["6. Self-Critique Loop<br/>& Answer Generation"]
    F --> G["7. LLMOps Evaluation<br/>(LangSmith & Ragas)"]
```

---

## 1. Core Purpose & Business Value

The **documentation-chatbot** eliminates the guesswork from AI-powered documentation assistance and customer support by restricting every generated answer to verified, company-owned knowledge — making hallucination structurally impossible.

- **Zero Hallucination, 100% Accuracy**: Every customer response is grounded in the company's own documentation and knowledge base, guaranteeing factual accuracy with no invented information.
- **Domain Boundary Enforcement**: The engine automatically rejects off-topic inquiries, keeping support conversations strictly within defined business domains (e.g., Fintech SaaS product documentation).
- **Comprehensive Knowledge Coverage**: Combines traditional document search with intelligent entity-relationship mapping, surfacing connections between products, pricing tiers, and organizational hierarchies that simple search cannot find.
- **Self-Correcting Quality Assurance**: An automated review loop verifies every draft answer against source material before delivery, catching errors before customers ever see them.
- **Prompt Token & Latency Optimization**: Integrates LLMLingua-2 context compression to compact retrieved passages, reducing LLM token consumption and latency while preserving vital semantic facts.
- **Plug-and-Play Backend Template**: The streamlined architecture means this engine can be connected to any customer portal, mobile app, or internal tool with minimal integration effort.
- **Intelligent Query Enhancement**: Automatically improves vague or abstract questions via hypothetical document expansion, maximizing the chance of surfacing the most relevant knowledge for complex queries.

---

## 2. System Architecture & Technical Execution

The core RAG engine (`src/`) exposes a unified FastAPI API interface directly to clients. The backend runs a stateful multi-node LangGraph agent that performs domain classification, optional HyDE query expansion, hybrid vector + graph retrieval, neural reranking (FlashRank), context compression (LLMLingua-2), and a self-critique quality loop before returning a response.

### Core Concept & Phased Execution Sequence

```mermaid
---
config:
  theme: neutral
---
sequenceDiagram
    autonumber
    actor Client as Client App / End User
    participant BE as RAG Backend (port 8000)
    participant AG as LangGraph Agent
    participant VDB as PGVector DB (port 5432)
    participant GDB as Neo4j Graph DB (port 7687)

    Note over Client, BE: Phase 1: Query Submission
    Client->>BE: POST /query (message, history)

    Note over BE, AG: Phase 2: Agent Execution Loop
    BE->>AG: Invoke StateGraph (AgentState)
    AG->>AG: node_classifier - Classify domain scope

    alt Query within business domain
        rect rgb(240, 243, 246)
            AG->>AG: node_hyde_decision - Evaluate HyDE necessity
            AG->>AG: node_hyde_generator - Generate hypothetical document (if enabled)
            AG->>VDB: Dense vector similarity search (PGVector + Gemini embeddings)
            AG->>GDB: Cypher graph query - extract entity relationships (Neo4j Bolt)
            AG->>AG: node_retrieve - Retrieve context & rerank (PGVector + FlashRank)
            AG->>AG: node_retrieve - LLMLingua-2 context compression
            AG->>AG: node_generate - Synthesize grounded answer from context
            AG->>AG: node_critique - Self-critique quality check
        end
    else Query outside business domain
        rect rgb(250, 235, 235)
            AG->>AG: node_refuse - Generate polite refusal message
        end
    end

    Note over AG, BE: Phase 3: Response Delivery
    AG-->>BE: Return final AgentState (agent_response)
    BE-->>Client: QueryResponse (response, citations, tool_calls_executed, retrieved_documents)
```

---

### High-Level Target Production Architecture

```mermaid
---
config:
  layout: elk
  theme: neutral
---
flowchart TB

    subgraph Client["Client"]
        User["Browser / Mobile App / API Consumer"]
    end

    subgraph Edge["Edge Layer"]
        CDN["CDN (Cloudflare / AWS CloudFront)"]
        LB["Load Balancer (Nginx / HAProxy)"]
        Ingress["Kubernetes Nginx Ingress Controller"]
    end

    subgraph BackendSvc["RAG Backend Service (theme-based-rag-backend)"]
        BE["Backend Handler (FastAPI + Uvicorn, port 8000)"]
        subgraph AgentGraph["LangGraph Agent StateGraph"]
            Classifier["node_classifier"]
            HyDEDecision["node_hyde_decision"]
            HyDEGen["node_hyde_generator"]
            Retrieve["node_retrieve<br/>(FlashRank Rerank + LLMLingua-2)"]
            Generate["node_generate"]
            Critique["node_critique"]
            Refuse["node_refuse"]
        end
    end

    subgraph VectorStore["PGVector DB (StatefulSet)"]
        PGVectorPort["PostgreSQL (port 5432)"]
        PGVectorPVC[("PVC: pgvector-data 5Gi")]
    end

    subgraph GraphStore["Neo4j Graph DB (StatefulSet)"]
        Neo4jBolt["Bolt Protocol (port 7687)"]
        Neo4jHTTP["HTTP Browser (port 7474)"]
        Neo4jPVC[("PVC: neo4j-data 5Gi")]
    end

    subgraph Observability["Observability"]
        LangSmith["LangSmith Tracing (api.smith.langchain.com)"]
    end

    subgraph SecretsLayer["Kubernetes Secrets"]
        PostgresSec["postgres-secrets (POSTGRES_DB / USER / PASSWORD)"]
        GeminiSec["gemini-secrets (GEMINI_API_KEY)"]
        LangchainSec["langchain-secrets (LANGSMITH_API_KEY)"]
        Neo4jSec["neo4j-secrets (NEO4J_USERNAME / PASSWORD)"]
    end

    User --> CDN
    CDN --> LB
    LB --> Ingress
    Ingress --> BE

    BE --> AgentGraph
    AgentGraph --> PGVectorPort
    AgentGraph --> Neo4jBolt
    AgentGraph --> LangSmith

    PGVectorPort --> PGVectorPVC
    Neo4jBolt --> Neo4jPVC

    BE --> PostgresSec
    BE --> GeminiSec
    BE --> LangchainSec
    BE --> Neo4jSec
```

---

### Kubernetes Network & Service Isolation Design

```mermaid
---
config:
  layout: elk
  theme: neutral
---
flowchart TB

    subgraph Outside["Outside World"]
        ExternalClient["curl / Browser / Frontend App / pytest"]
    end

    subgraph Internal["Kubernetes Internal Network (rag-engine namespace) - not reachable from outside"]

        subgraph BackendCtr["theme-based-rag-backend (FastAPI + Uvicorn, ClusterIP port 80 -> 8000)"]
            direction TB
            BEQueryH["POST /query - invoke LangGraph StateGraph"]
            BEHealthH["GET /health - ping vector store"]
        end

        subgraph PGVectorCtr["pgvector (StatefulSet, Headless Service port 5432)"]
            direction TB
            PGVectorPort["PostgreSQL: port 5432"]
            PGVectorData[("collection: raptor_chunks<br/>dense: gemini-embedding-001<br/>PVC: pgvector-data 5Gi")]
        end

        subgraph Neo4jCtr["neo4j (StatefulSet, Headless Service port 7474/7687)"]
            direction TB
            Neo4jBrowserPort["HTTP Browser: port 7474"]
            Neo4jBoltPort["Bolt: port 7687"]
            Neo4jData[("nodes: Entity<br/>relationships: RELATED_TO<br/>PVC: neo4j-data 5Gi")]
        end

        subgraph SecretsCtr["Kubernetes Opaque Secrets"]
            direction TB
            S0["postgres-secrets: POSTGRES_DB / USER / PASSWORD"]
            S1["gemini-secrets: GEMINI_API_KEY"]
            S2["neo4j-secrets: NEO4J_USERNAME / NEO4J_PASSWORD / NEO4J_AUTH"]
            S3["langchain-secrets: LANGSMITH_API_KEY"]
        end

    end

    ExternalClient -->|"NodePort / Nginx Ingress - exposed entry point"| BackendCtr
    BEQueryH -->|"vector similarity search"| PGVectorPort
    BEQueryH -->|"Cypher MATCH query via Bolt"| Neo4jBoltPort
    BackendCtr -->|"env injection from secrets"| SecretsCtr
```

---

## 3. Repository Structure

```text
Enterprise-RAG-Engine/
├── Dockerfile                         # Multi-stage production container image
├── infra/
│   ├── TF/
│   │   ├── backend.tf                 # Backend Deployment + ClusterIP Service
│   │   ├── ingress.tf                 # Nginx Ingress routing rule
│   │   ├── neo4j.tf                   # Neo4j StatefulSet + Headless Service + PVC
│   │   ├── pgvector.tf                # PGVector StatefulSet + Headless Service + PVC
│   │   ├── secrets.tf                 # Kubernetes Opaque Secrets
│   │   ├── namespace.tf               # Kubernetes namespace definition
│   │   ├── providers.tf               # Terraform provider configuration
│   │   ├── variables.tf               # Input variable declarations
│   │   ├── outputs.tf                 # Terraform output definitions
│   │   └── terraform.tfvars.example   # Example variable values (safe to commit)
│   ├── build-image.sh                 # Docker image build script
│   └── docker-compose.yml             # Local multi-container Docker Compose stack
├── src/
│   ├── agent_flow/                    # LangGraph StateGraph nodes and edges
│   ├── script/                        # Vector DB batch ingestion scripts
│   ├── config.py                      # Environment variable configuration
│   ├── graph_db.py                    # Neo4j driver, entity extraction, Cypher queries
│   ├── vector_db.py                   # PGVector search, embedding pipeline
│   ├── tools.py                       # LangGraph tool: retrieve_VDB
│   ├── models.py                      # Pydantic request/response schemas
│   └── main.py                        # FastAPI app: /query, /health
├── pyproject.toml                     # Project metadata, dependencies, ruff + pytest config
├── langgraph.json                     # LangGraph API server configuration
└── README.md
```
