# 🤖 FAIForge

> A production-ready AI boilerplate with unified adapter patterns for LLMs, RAG, and intelligent routing

![Status](https://img.shields.io/badge/status-production--ready-green)
![Version](https://img.shields.io/badge/version-2.5.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Build AI applications faster.** FAIForge provides a complete foundation with multi-provider LLM support, RAG pipelines, streaming, function calling, and intelligent model routing—all with built-in observability and Docker deployment.

---

## 💡 Why FAIForge?

I built this while exploring different LLM providers and found myself repeatedly solving the same infrastructure problems:

- **Provider switching** - Rewriting code every time I wanted to test a different model
- **Cost tracking** - No visibility into per-request costs across providers
- **Observability** - Difficulty debugging AI interactions without proper logging
- **Deployment** - Setting up Docker, security headers, CORS each time
- **RAG complexity** - Implementing chunking, embeddings, vector stores from scratch
- **Reliability** - No automatic failover when providers go down

The adapter pattern solves this. Now I can compare GPT-4o vs Claude with just a config change, implement RAG with pluggable components, and have automatic failover when providers fail.

**Quick Links:** [Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#️-architecture) • [API Docs](#-api-reference) • [RAG System](#-rag-system) • [Deployment](#-deployment)

---

## ✨ Features

### 🔌 Multi-Provider LLM Architecture
- **OpenAI** (GPT-4o, GPT-4o-mini)
- **Anthropic** (Claude Opus 4, Claude Sonnet 4.5)
- **Google Gemini** (gemini-2.0-flash, gemini-1.5-pro/flash)
- **Cohere** (Command R+, Command R)
- **Ollama** (Local models - Llama 3, Mistral, Phi-3, any Ollama model)
- **vLLM** (GPU-accelerated local serving - any HuggingFace model)
- Unified adapter pattern - switch providers with one line

### 🚀 Advanced LLM Capabilities
- **Streaming (SSE)** - Real-time token streaming for responsive UX
- **Function Calling** - Native tool use support for OpenAI & Anthropic
- **Structured Outputs** - JSON mode for reliable parsing
- **Model Fallbacks** - Automatic failover when providers fail
- **Smart Routing** - Route requests based on complexity/cost rules
- **Circuit Breaker** - Auto-disable unhealthy providers
- **Semantic Caching** - Cache similar queries to reduce costs & latency

### 📚 RAG (Retrieval-Augmented Generation)
- **4 Vector Databases** - Pinecone, Weaviate, Qdrant, ChromaDB
- **2 Embedding Providers** - OpenAI, HuggingFace (local)
- **4 Chunking Strategies** - Recursive, semantic, token-based, fixed-size
- **Hybrid Search** - BM25 + semantic with RRF/weighted fusion
- **Cross-Encoder Reranker** - Post-retrieval reranking (`use_rerank=true`)
- **RAG-embedded Chat** - `use_rag=true` injects context into any chat completion
- **Document Management** - List and delete ingested chunks via API
- **Async Ingestion** - `background=true` returns job ID immediately

### 🧪 Evals & Observability
- **LLM Judge** - 0–10 AI scoring of any response
- **RAG Metrics** - Faithfulness, answer relevancy, context precision/recall (Ragas-first, LLM-judge fallback)
- **Prompt A/B Testing** - ABRouter with round-robin and weighted-random routing
- **"Rate This" UI** - Thumbs up/down + AI judge button on every chat message
- **Prometheus metrics** - `/metrics` endpoint for Grafana/alerting
- **Structured JSON logging** - Machine-parseable logs with correlation IDs
- **Automatic cost tracking** - Per-request pricing for all providers
- **Provider health checks** - Circuit breaker status monitoring

### 🔒 Enterprise Middleware
- **API Key Auth** - Bearer token authentication; set `FAIFORGE_API_KEYS` to enable
- **Rate Limiting** - Per-key sliding window; 429 + `Retry-After` header
- **Zero-config dev mode** - Auth/rate limiting auto-disabled when keys not set

### ⚙️ Configuration-Driven
- **YAML-based config** - No hardcoded values
- **Environment overrides** - Different configs for dev/staging/prod
- **Fallback chains** - Configurable provider failover sequences
- **Routing rules** - Define model selection logic in config
- **12-factor app** compliant

### 🐳 Docker Deployment
- **One-command setup** - `docker-compose up`
- **Multi-stage builds** - Optimized image sizes (~350MB total)
- **Health monitoring** - Auto-restart on failure
- **Production-ready** - Non-root user, security headers

### 🎨 Full-Stack Ready
- **FastAPI backend** - Modern, async Python
- **React + TypeScript frontend** - 4-tab UI (Chat, RAG, Stats, Evals)
  - **Chat tab** - Multi-model selector, streaming, RAG toggle, collapsible source citations, "Rate This" AI judge panel
  - **RAG tab** - Document ingest, hybrid/semantic/BM25 search, document browser with per-chunk deletion
  - **Stats tab** - Live cache stats, RAG corpus stats, cache clear
  - **Evals tab** - Feedback log, A/B experiment stats
- **Nginx reverse proxy** - Production-grade serving
- **API documentation** - Auto-generated OpenAPI/Swagger

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API key (required)
- Anthropic, Gemini, or Cohere API keys (optional)

### 1. Clone & Setup
```bash
git clone https://github.com/fiddyrod/faiforge.git
cd faiforge

# Add your API keys
cp backend/.env.example backend/.env
nano backend/.env  # Add your keys (see Environment Variables section)
```

### 2. Start Everything
```bash
docker-compose up -d
```

That's it! 🎉

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

### 3. Test It
```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "model": "gpt-4o-mini"
  }'

# Streaming response
curl -X POST "http://localhost:8000/v1/chat/completions?stream=true" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "model": "gpt-4o-mini"
  }'
```

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                         FAIForge v2.5                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐                                                │
│  │   Browser   │                                                │
│  └──────┬──────┘                                                │
│         │ HTTP                                                  │
│         ↓                                                       │
│  ┌─────────────────────────────────┐                           │
│  │  Frontend (React + Nginx)       │                           │
│  │  Port: 3000  (4 tabs)           │                           │
│  │  Chat │ RAG │ Stats │ Evals     │                           │
│  └──────┬──────────────────────────┘                           │
│         │ Proxy /v1/* → backend:8000                           │
│         ↓                                                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Backend (FastAPI) - Port: 8000                             ││
│  │  ┌─────────────────────────────────────────────────────┐   ││
│  │  │  Middleware                                          │   ││
│  │  │  • API Key Auth (Bearer)  • Rate Limiting (per-key) │   ││
│  │  └───────────────────────────┬─────────────────────────┘   ││
│  │                              ↓                              ││
│  │  ┌─────────────────────────────────────────────────────┐   ││
│  │  │  API Layer                                           │   ││
│  │  │  • Streaming (SSE)    • Function Calling            │   ││
│  │  │  • Structured Outputs • RAG-embedded Chat           │   ││
│  │  └───────────────────────────┬─────────────────────────┘   ││
│  │                              ↓                              ││
│  │  ┌─────────────────────────────────────────────────────┐   ││
│  │  │  Smart Router + Fallback Adapter                     │   ││
│  │  │  • Query-based routing  • Circuit breaker           │   ││
│  │  │  • Retry with backoff   • Provider failover         │   ││
│  │  └───────────────────────────┬─────────────────────────┘   ││
│  │                              ↓                              ││
│  │  ┌─────────────────────────────────────────────────────┐   ││
│  │  │  LLM Adapters (6 providers)                          │   ││
│  │  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │   ││
│  │  │  │OpenAI  │ │Anthropic│ │Gemini  │ │Cohere  │        │   ││
│  │  │  └────────┘ └────────┘ └────────┘ └────────┘        │   ││
│  │  │  ┌────────┐ ┌────────┐                               │   ││
│  │  │  │Ollama  │ │ vLLM   │                               │   ││
│  │  │  │(local) │ │ (GPU)  │                               │   ││
│  │  │  └────────┘ └────────┘                               │   ││
│  │  └─────────────────────────────────────────────────────┘   ││
│  │                                                             ││
│  │  ┌─────────────────────────────────────────────────────┐   ││
│  │  │  RAG Pipeline                                        │   ││
│  │  │  Embeddings → Chunking → Vector Store               │   ││
│  │  │  + Hybrid Search (BM25+semantic) + Reranker         │   ││
│  │  │  Pinecone │ Weaviate │ Qdrant │ ChromaDB            │   ││
│  │  └─────────────────────────────────────────────────────┘   ││
│  │                                                             ││
│  │  ┌─────────────────────────────────────────────────────┐   ││
│  │  │  Evals                                               │   ││
│  │  │  LLM Judge • RAG Metrics • A/B Testing              │   ││
│  │  └─────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  Observability: Prometheus /metrics • JSON logs • Cost tracking│
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

**Frontend (`/frontend`)**
- React 18 + TypeScript
- Tailwind CSS styling
- Vite build system
- Nginx production serving

**Backend (`/backend`)**
- FastAPI async framework
- Pydantic validation
- Multi-provider adapters
- Structured logging

**LLM Adapters (`/backend/core/inference/adapters`)**
- Unified interface for all providers
- Streaming support (SSE)
- Function/tool calling
- Structured outputs (JSON mode)

**Fallback & Routing (`/backend/core/inference`)**
- `fallback.py` - FallbackAdapter with circuit breaker
- `registry.py` - Model registry with routing support
- Config-driven fallback chains

**Semantic Cache (`/backend/core/cache`)**
- `semantic.py` - SemanticCache with embedding similarity
- `backends/` - Memory (dev) and Redis (production) backends
- Configurable similarity threshold and TTL

**RAG System (`/backend/core/rag`)**
- Embedding adapters (OpenAI, HuggingFace)
- Chunking strategies (recursive, semantic, token, fixed)
- Vector store adapters (Pinecone, Weaviate, Qdrant, Chroma)
- Pipeline orchestration

**Configuration (`/backend/core/config`)**
- `app.yaml` - Application settings
- `models.yaml` - Model definitions
- `routing.yaml` - Fallback chains & routing rules
- `rag.yaml` - RAG pipeline configuration

---

## ⚙️ Configuration

### Model Config (`backend/core/config/models.yaml`)
```yaml
models:
  gpt-4o-mini:
    adapter: openai
    model: gpt-4o-mini

  claude-sonnet:
    adapter: anthropic
    model: claude-sonnet-4-5-20250929

  tiny-llama:
    adapter: vllm
    model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

### Fallback & Routing Config (`backend/core/config/routing.yaml`)
```yaml
fallback_chains:
  default:
    models: [gpt-4o-mini, claude-sonnet, ollama/phi3]  # local as final fallback
    retry:
      max_retries: 3
      base_delay: 1.0
    circuit_breaker:
      threshold: 5
      recovery: 300

  high_quality:
    models: [gpt-4o, claude-opus, gpt-4o-mini]

  local_only:
    models: [ollama/llama3, ollama/mistral, ollama/phi3]

routing:
  enabled: true
  default_chain: default
  rules:
    - condition: has_tools
      fallback_chain: high_quality
    - condition: max_tokens > 2000
      fallback_chain: high_quality

# Semantic caching (reduces API costs)
cache:
  enabled: true
  backend: memory           # "memory" for dev, "redis" for prod
  similarity_threshold: 0.95
  ttl: 3600                 # 1 hour
  redis:
    url: "redis://localhost:6379"
```

### RAG Config (`backend/core/config/rag.yaml`)
```yaml
embeddings:
  default: openai
  adapters:
    openai:
      model: text-embedding-3-small
    huggingface:
      model: all-MiniLM-L6-v2

chunking:
  default: recursive
  adapters:
    recursive:
      chunk_size: 512
      overlap: 50

vector_stores:
  default: chroma
  adapters:
    chroma:
      collection: documents
    pinecone:
      index: faiforge
```

### Environment Variables
```bash
# Required
OPENAI_API_KEY=sk-...

# LLM Providers (optional — models skipped if key not set)
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...
COHERE_API_KEY=...

# Enterprise (optional — disabled when not set)
FAIFORGE_API_KEYS=key1,key2,key3   # Bearer token auth
FAIFORGE_RATE_LIMIT_REQUESTS=60    # requests per window (default: 60)
FAIFORGE_RATE_LIMIT_WINDOW=60      # window in seconds (default: 60)

# Infrastructure
ENV=production              # development | production
LOAD_VLLM=false            # Enable GPU local models
PINECONE_API_KEY=...       # For Pinecone vector store
```

---

## 📡 API Reference

### Health Check
```bash
GET /health
# Response: {"status": "healthy", "models_loaded": 4}
```

### Provider Health (Circuit Breaker Status)
```bash
GET /v1/health/providers
# Response: {"status": "ok", "providers": {...circuit breaker status...}}
```

### Cache Stats
```bash
GET /v1/cache/stats
# Response: {"enabled": true, "backend": "memory", "hit_rate": 0.85, "cache_size": 150}
```

### Clear Cache
```bash
POST /v1/cache/clear
# Response: {"status": "ok", "entries_cleared": 150}
```

### List Models
```bash
GET /v1/models
# Response: {"models": [...], "fallback_chains": [...], "all": [...]}
```

### Chat Completion
```bash
POST /v1/chat/completions
Content-Type: application/json

{
  "messages": [{"role": "user", "content": "Hello!"}],
  "model": "gpt-4o-mini",
  "temperature": 0.7,
  "max_tokens": 500
}
```

### Streaming Completion
```bash
POST /v1/chat/completions?stream=true
# Returns: Server-Sent Events (SSE) stream
```

### Function Calling
```bash
POST /v1/chat/completions
{
  "messages": [{"role": "user", "content": "What's the weather?"}],
  "model": "gpt-4o-mini",
  "tools": [{
    "type": "function",
    "function": {
      "name": "get_weather",
      "description": "Get weather for a city",
      "parameters": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  }]
}
```

### Structured Output (JSON Mode)
```bash
POST /v1/chat/completions
{
  "messages": [{"role": "user", "content": "Extract: John is 25"}],
  "model": "gpt-4o-mini",
  "response_format": {"type": "json_object"}
}
```

### RAG - Ingest Documents
```bash
POST /v1/rag/ingest
Content-Type: application/json

{
  "documents": [
    {"text": "Your document content here...", "source": "my-doc.txt"}
  ]
}
# Response: {"status": "ok", "chunks_created": 12, "documents_ingested": 1}
```

### RAG - Query
```bash
POST /v1/rag/query
Content-Type: application/json

{
  "query": "What is the main topic?",
  "top_k": 5,
  "search_mode": "hybrid"   # "hybrid" | "semantic" | "bm25"
}
# Response: {"results": [{"text": "...", "source": "...", "score": 0.92, "semantic_score": 0.89, "bm25_score": 0.74}]}
```

### RAG - Stats
```bash
GET /v1/rag/stats
# Response: {"total_chunks": 42, "total_documents": 3, "hybrid_search": {"enabled": true, "fusion_method": "rrf"}}
```

### RAG - List / Delete Documents
```bash
GET /v1/rag/documents
DELETE /v1/rag/documents/{chunk_id}
```

### RAG - Async Ingestion
```bash
# Start in background
POST /v1/rag/ingest?background=true
# Response: {"job_id": "uuid", "status": "pending"}

# Poll status
GET /v1/rag/jobs/{job_id}
# Response: {"job_id": "...", "status": "done", "result": {...}}
```

### RAG-embedded Chat
```bash
POST /v1/chat/completions?use_rag=true
# Retrieves top-5 relevant chunks, injects as system context
# Response includes: {"content": "...", "sources": [{"content": "...", "score": 0.92}]}
```

### Evals
```bash
# LLM Judge score
POST /v1/evals/judge
{"messages": [...], "response": "...", "criteria": "helpfulness"}

# Submit feedback
POST /v1/evals/feedback
{"message_id": "...", "rating": 1, "comment": "..."}

# Get feedback log
GET /v1/evals/feedback

# Run RAG metrics
POST /v1/evals/rag
{"question": "...", "answer": "...", "contexts": [...], "ground_truth": "..."}

# A/B experiments
POST /v1/evals/ab/experiments
GET /v1/evals/ab/experiments/{id}/stats
```

### Prometheus Metrics
```bash
GET /metrics
# Returns Prometheus-format metrics: request counts, latency, token usage, costs
```

**Full API docs:** http://localhost:8000/docs

---

## 📚 RAG System

FAIForge includes a complete RAG (Retrieval-Augmented Generation) system with pluggable components.

### Components

| Component | Options |
|-----------|---------|
| **Embeddings** | OpenAI (text-embedding-3-small/large), HuggingFace (all-MiniLM-L6-v2, local) |
| **Chunking** | Recursive (smart splitting), Semantic (similarity-based), Token-based (LLM-aware), Fixed-size |
| **Vector Stores** | ChromaDB (embedded), Pinecone (managed), Qdrant (self-hosted), Weaviate (open-source) |

### Search Modes

| Mode | Description |
|------|-------------|
| `semantic` | Dense vector search via embeddings - best for conceptual similarity |
| `bm25` | Sparse keyword retrieval (Okapi BM25) - best for exact term matching |
| `hybrid` | BM25 + semantic fused with RRF (Reciprocal Rank Fusion) - best overall |

### Python Usage

```python
from core.rag import RAGPipeline, RAGRegistry, SearchMode, HybridConfig

# Initialize with hybrid search enabled
registry = RAGRegistry()
pipeline = RAGPipeline(
    embedding_adapter=registry.get_embedding("openai"),
    chunking_adapter=registry.get_chunking("recursive"),
    vector_store=registry.get_vector_store("chroma"),
    hybrid_config=HybridConfig(enabled=True, semantic_weight=0.6, bm25_weight=0.4)
)

# Ingest documents
await pipeline.ingest_documents(documents)

# Query with hybrid search (default)
results = await pipeline.query("What is the main topic?", top_k=5)

# Query with specific search mode
results = await pipeline.query("exact keyword match", top_k=5, search_mode=SearchMode.BM25)
```

### REST API

The RAG system is also exposed via REST endpoints - see the [API Reference](#-api-reference) section for `/v1/rag/ingest`, `/v1/rag/query`, and `/v1/rag/stats`.

---

## 🔌 Extending FAIForge

### Adding New LLM Providers

1. Create adapter class inheriting from `LLMAdapter`
2. Implement `complete()` and `complete_stream()` methods
3. Handle `tools` and `response_format` parameters
4. Register in `registry.py` and configure in `models.yaml`

**Currently supported:** OpenAI, Anthropic, Google Gemini, Cohere, Ollama, vLLM
**Easy to add:** Mistral AI, any OpenAI-compatible API

### Adding New Vector Stores

1. Create adapter inheriting from `VectorStoreAdapter`
2. Implement `add()`, `search()`, `delete()` methods
3. Register in RAG registry

**Currently supported:** ChromaDB, Pinecone, Qdrant, Weaviate

---

## 🚀 Deployment

### Docker Compose (Recommended)
```bash
docker-compose up -d
docker-compose logs -f
docker-compose down
```

### Cloud Platforms

**AWS ECS / Fargate** - Use task definitions with environment variables
**Google Cloud Run** - Deploy as separate services
**Railway / Render / Fly.io** - One-click deployment with GitHub

---

## 🛣️ Roadmap

### Complete Roadmap Table

| Category | Feature | Status | Description |
|----------|---------|--------|-------------|
| **Foundation** | Streaming (SSE) | ✅ Done | Real-time token streaming |
| **Foundation** | Function Calling | ✅ Done | Native tool use (OpenAI/Anthropic) |
| **Foundation** | Structured Outputs | ✅ Done | JSON mode for reliable parsing |
| **Infra** | Model Fallbacks | ✅ Done | Auto-failover with circuit breaker |
| **Infra** | Smart Routing | ✅ Done | Query-based model selection |
| **Infra** | Semantic Caching | ✅ Done | Redis + In-Memory backends |
| **RAG** | Vector Databases | ✅ Done | Pinecone, Weaviate, Qdrant, ChromaDB |
| **RAG** | Embeddings | ✅ Done | OpenAI, HuggingFace (local) |
| **RAG** | Chunking | ✅ Done | Recursive, semantic, token, fixed |
| **RAG** | Hybrid Search | ✅ Done | BM25 + semantic search with RRF fusion |
| **Edge AI** | Ollama Adapter | ✅ Done | Local inference (Llama3, Mistral, Phi-3) |
| **Evals** | RAG Evaluation | ✅ Done | Ragas + LLM-judge fallback (faithfulness, relevancy, precision, recall) |
| **Evals** | LLM Response Eval | ✅ Done | LLMJudge (0-10 scorer) + "Rate This" UI |
| **Evals** | Prompt A/B Testing | ✅ Done | ABRouter with round-robin & weighted-random |
| **Observability** | Prometheus | ✅ Done | `/metrics` endpoint, Prometheus-compatible |
| **Adapters** | Gemini Adapter | ✅ Done | gemini-2.0-flash, gemini-1.5-pro/flash |
| **Adapters** | Cohere Adapter | ✅ Done | Command R+, Command R |
| **Enterprise** | API Key Auth | ✅ Done | Bearer token; disabled when keys unset |
| **Enterprise** | Rate Limiting | ✅ Done | Per-key sliding window, 429 + Retry-After |
| **Enterprise** | Async Ingestion | ✅ Done | `background=true` + job polling |
| **Agentic** | MCP Integration | 🔜 Planned | Model Context Protocol |
| **Agentic** | Agent Framework | 🔜 Planned | LangGraph/tool-using agents |
| **Agentic** | Human-in-the-loop | 🔜 Planned | Approval workflows |
| **Observability** | LangSmith/W&B | 🔜 Planned | Tracing integration |
| **Edge AI** | llama.cpp/ONNX | 🔜 Planned | Quantized model support |
| **Multimodal** | Vision Support | 🔜 Planned | GPT-4o, Gemini Vision |
| **Multimodal** | Audio Processing | 🔜 Planned | Speech/audio input |
| **Governance** | Prompt Injection Defense | 🔜 Planned | Security hardening |
| **Governance** | PII Masking | 🔜 Planned | Data privacy |
| **Governance** | Content Moderation | 🔜 Planned | Guardrails |

---

## 📝 License & Usage

MIT License - see [LICENSE](LICENSE) for details.

This project is open-source and available for use in personal or commercial projects. Feel free to fork and adapt for your own needs!

---

## 🙏 Acknowledgments

Built with: **FastAPI**, **React**, **Ollama**, **vLLM**, **Docker**, **Tailwind CSS**, **Pydantic**, **Nginx**, **Prometheus**

Vector stores: **ChromaDB**, **Pinecone**, **Qdrant**, **Weaviate**

---

**⭐ Star this repo if you find it useful!**

---

*FAIForge v2.5 - Production-ready AI infrastructure for modern applications* 🚀
