# 🤖 FAIForge

> A production-ready AI boilerplate with unified adapter patterns for LLMs, RAG, and intelligent routing

![Status](https://img.shields.io/badge/status-production--ready-green)
![Version](https://img.shields.io/badge/version-2.0.0-blue)
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
- **vLLM** (Local models - TinyLlama, any HuggingFace model)
- Unified adapter pattern - switch providers with one line

### 🚀 Advanced LLM Capabilities
- **Streaming (SSE)** - Real-time token streaming for responsive UX
- **Function Calling** - Native tool use support for OpenAI & Anthropic
- **Structured Outputs** - JSON mode for reliable parsing
- **Model Fallbacks** - Automatic failover when providers fail
- **Smart Routing** - Route requests based on complexity/cost rules
- **Circuit Breaker** - Auto-disable unhealthy providers

### 📚 RAG (Retrieval-Augmented Generation)
- **4 Vector Databases** - Pinecone, Weaviate, Qdrant, ChromaDB
- **2 Embedding Providers** - OpenAI, HuggingFace (local)
- **4 Chunking Strategies** - Recursive, semantic, token-based, fixed-size
- **Pipeline Orchestration** - End-to-end document ingestion & retrieval

### 📊 Production Observability
- **Structured JSON logging** - Machine-parseable logs
- **Request correlation IDs** - Trace requests end-to-end
- **Automatic cost tracking** - Per-request pricing for all providers
- **Performance monitoring** - Latency, token counts, error rates
- **Provider health checks** - Circuit breaker status monitoring

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
- **React + TypeScript frontend** - Beautiful chat UI
- **Nginx reverse proxy** - Production-grade serving
- **API documentation** - Auto-generated OpenAPI/Swagger

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API key
- Anthropic API key (optional)

### 1. Clone & Setup
```bash
git clone https://github.com/fiddyrod/faiforge.git
cd faiforge

# Add your API keys
cp backend/.env.example backend/.env
nano backend/.env  # Add your OPENAI_API_KEY and ANTHROPIC_API_KEY
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
│                         FAIForge v2.0                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐                                                │
│  │   Browser   │                                                │
│  └──────┬──────┘                                                │
│         │ HTTP                                                  │
│         ↓                                                       │
│  ┌─────────────────────────────────┐                           │
│  │  Frontend (React + Nginx)       │                           │
│  │  Port: 3000                     │                           │
│  └──────┬──────────────────────────┘                           │
│         │ Proxy /v1/* → backend:8000                           │
│         ↓                                                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Backend (FastAPI) - Port: 8000                             ││
│  │  ┌─────────────────────────────────────────────────────┐   ││
│  │  │  API Layer                                           │   ││
│  │  │  • Streaming (SSE)    • Function Calling            │   ││
│  │  │  • Structured Outputs • Provider Health             │   ││
│  │  └───────────────────────────┬─────────────────────────┘   ││
│  │                              ↓                              ││
│  │  ┌─────────────────────────────────────────────────────┐   ││
│  │  │  Smart Router                                        │   ││
│  │  │  • Query-based routing  • Complexity detection      │   ││
│  │  └───────────────────────────┬─────────────────────────┘   ││
│  │                              ↓                              ││
│  │  ┌─────────────────────────────────────────────────────┐   ││
│  │  │  Fallback Adapter                                    │   ││
│  │  │  • Retry with backoff   • Circuit breaker           │   ││
│  │  │  • Provider failover    • Health tracking           │   ││
│  │  └───────────────────────────┬─────────────────────────┘   ││
│  │                              ↓                              ││
│  │  ┌─────────────────────────────────────────────────────┐   ││
│  │  │  LLM Adapters                                        │   ││
│  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │   ││
│  │  │  │ OpenAI   │ │Anthropic │ │  vLLM    │            │   ││
│  │  │  │ GPT-4o   │ │ Claude   │ │ (local)  │            │   ││
│  │  │  └────┬─────┘ └────┬─────┘ └──────────┘            │   ││
│  │  │       │            │                                │   ││
│  │  └───────┼────────────┼────────────────────────────────┘   ││
│  │          ↓            ↓                                     ││
│  │    api.openai.com   api.anthropic.com                      ││
│  │                                                             ││
│  │  ┌─────────────────────────────────────────────────────┐   ││
│  │  │  RAG Pipeline                                        │   ││
│  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐            │   ││
│  │  │  │Embeddings│→│ Chunking │→│  Vector  │            │   ││
│  │  │  │OpenAI/HF │ │4 methods │ │  Store   │            │   ││
│  │  │  └──────────┘ └──────────┘ └──────────┘            │   ││
│  │  │                             ↓                       │   ││
│  │  │                      Pinecone│Weaviate│Qdrant│Chroma│   ││
│  │  └─────────────────────────────────────────────────────┘   ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  Observability: JSON logs • Correlation IDs • Cost tracking    │
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
    models: [gpt-4o-mini, claude-sonnet, tinyllama]
    retry:
      max_retries: 3
      base_delay: 1.0
    circuit_breaker:
      threshold: 5
      recovery: 300

  high_quality:
    models: [gpt-4o, claude-opus, gpt-4o-mini]

routing:
  enabled: true
  default_chain: default
  rules:
    - condition: has_tools
      fallback_chain: high_quality
    - condition: max_tokens > 2000
      fallback_chain: high_quality
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
ANTHROPIC_API_KEY=sk-ant-...

# Optional
ENV=production              # development | production
LOAD_VLLM=false            # Enable local models
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

### Usage

```python
from core.rag import RAGPipeline, RAGRegistry

# Initialize
registry = RAGRegistry()
pipeline = RAGPipeline(
    embedding_adapter=registry.get_embedding("openai"),
    chunking_adapter=registry.get_chunking("recursive"),
    vector_store=registry.get_vector_store("chroma")
)

# Ingest documents
await pipeline.ingest(documents)

# Query
results = await pipeline.query("What is the main topic?", top_k=5)
```

---

## 🔌 Extending FAIForge

### Adding New LLM Providers

1. Create adapter class inheriting from `LLMAdapter`
2. Implement `complete()` and `complete_stream()` methods
3. Handle `tools` and `response_format` parameters
4. Register in `registry.py` and configure in `models.yaml`

**Currently supported:** OpenAI, Anthropic, vLLM
**Easy to add:** Cohere, Google Gemini, Mistral AI, any OpenAI-compatible API

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
| **Infra** | Semantic Caching | 🔜 Planned | Redis/GPTCache (after RAG merge) |
| **RAG** | Vector Databases | ✅ Done | Pinecone, Weaviate, Qdrant, ChromaDB |
| **RAG** | Embeddings | ✅ Done | OpenAI, HuggingFace (local) |
| **RAG** | Chunking | ✅ Done | Recursive, semantic, token, fixed |
| **RAG** | Hybrid Search | 🔜 Planned | BM25 + semantic search |
| **Evals** | RAG Evaluation | 🔜 Planned | Ragas/Arize integration |
| **Evals** | LLM Response Eval | 🔜 Planned | Unit testing for AI |
| **Evals** | Prompt A/B Testing | 🔜 Planned | Systematic prompt optimization |
| **Agentic** | MCP Integration | 🔜 Planned | Model Context Protocol |
| **Agentic** | Agent Framework | 🔜 Planned | LangGraph/tool-using agents |
| **Agentic** | Human-in-the-loop | 🔜 Planned | Approval workflows |
| **Observability** | Prometheus | 🔜 Planned | Metrics exporter |
| **Observability** | LangSmith/W&B | 🔜 Planned | Tracing integration |
| **Edge AI** | Ollama Adapter | 🔜 Planned | Local model inference |
| **Edge AI** | llama.cpp/ONNX | 🔜 Planned | Quantized model support |
| **Multimodal** | Vision Support | 🔜 Planned | GPT-4o, Gemini Vision |
| **Multimodal** | Audio Processing | 🔜 Planned | Speech/audio input |
| **Governance** | Prompt Injection Defense | 🔜 Planned | Security hardening |
| **Governance** | PII Masking | 🔜 Planned | Data privacy |
| **Governance** | Content Moderation | 🔜 Planned | Guardrails |

### Persistence & Auth (Future)

| Feature | Status | Description |
|---------|--------|-------------|
| Conversation persistence | 🔜 Planned | SQLite/PostgreSQL |
| User authentication | 🔜 Planned | Sessions & auth |
| Rate limiting | 🔜 Planned | Request throttling |
| Redis caching | 🔜 Planned | Response caching |
| Admin dashboard | 🔜 Planned | Management UI |

---

## 📝 License & Usage

MIT License - see [LICENSE](LICENSE) for details.

This project is open-source and available for use in personal or commercial projects. Feel free to fork and adapt for your own needs!

---

## 🙏 Acknowledgments

Built with: **FastAPI**, **React**, **vLLM**, **Docker**, **Tailwind CSS**, **Pydantic**, **Nginx**

Vector stores: **ChromaDB**, **Pinecone**, **Qdrant**, **Weaviate**

---

**⭐ Star this repo if you find it useful!**

---

*FAIForge v2.0 - Production-ready AI infrastructure for modern applications* 🚀
