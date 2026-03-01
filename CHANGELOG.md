# Changelog

All notable changes to FAIForge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-22

### Added

**Multi-Provider LLM Support**
- OpenAI adapter (GPT-4o, GPT-4o-mini) with automatic cost tracking
- Anthropic adapter (Claude Opus 4, Claude Sonnet 4.5)
- vLLM adapter for local model deployment (TinyLlama, HuggingFace models)
- Unified adapter pattern for easy provider switching

**Production Observability**
- Structured JSON logging with correlation IDs
- Request tracing end-to-end across all components
- Automatic per-request cost calculation for all providers
- Performance monitoring (latency, token counts, error rates)
- Health check endpoints with model availability status

**Security Features**
- Comprehensive input validation with Pydantic constraints
  - Role validation (user, assistant, system)
  - Content length limits (1-32,000 characters)
  - Message count limits (1-50 messages)
  - Temperature range validation (0.0-2.0)
  - Token limits (1-4,000)
- API request timeouts (60s default for OpenAI and Anthropic)
- Security headers in nginx (XSS, CSP, Frame Options, Content-Type)
- Non-root Docker user configuration
- Environment-based CORS with configurable origins

**Infrastructure & Deployment**
- Docker Compose setup with multi-stage builds
- Optimized container images (~350MB total footprint)
- FastAPI backend with async request handling
- React + TypeScript frontend with Vite build system
- Nginx reverse proxy with production-grade configuration
- Health monitoring with automatic container restart
- One-command deployment (`docker-compose up`)

**Configuration Management**
- YAML-based configuration system (app.yaml, models.yaml)
- Environment-specific overrides (development.yaml, production.yaml)
- Runtime configuration via environment variables
- 12-factor app compliance

**Developer Experience**
- Complete test suite with pytest (12 tests covering API, config, validation)
- Test fixtures and mocks for adapters
- Comprehensive documentation
  - Quick start guide
  - Architecture documentation
  - Adapter tutorial with working Cohere example
  - Security deployment checklist
  - Contributing guidelines
- Auto-generated OpenAPI/Swagger documentation
- Local development support (with and without Docker)

### Documentation

- README with professional tone and clear structure
- Architecture diagrams showing component interactions
- Step-by-step tutorial for adding new LLM providers
- Security checklist for production deployments
- Contributing guidelines with code standards
- Design principles section explaining architectural decisions

### Testing

- API endpoint tests (health, models, chat completions)
- Configuration loading and override tests
- Input validation tests (empty messages, invalid roles, parameter ranges)
- Test coverage for error handling
- Mock fixtures for adapter testing

---

## [2.5.0] - 2026-03-01

### Added

**New LLM Adapters**
- `GeminiAdapter` — Google Gemini (gemini-2.0-flash, gemini-1.5-pro/flash) with streaming, function calling, JSON mode, and cost tracking
- `CohereAdapter` — Cohere (Command R+, Command R) via `AsyncClientV2` with streaming and JSON-via-preamble mode
- `models.yaml` entries for `gemini-flash`, `gemini-pro`, `command-r-plus`, `command-r`
- `GEMINI_API_KEY` and `COHERE_API_KEY` environment variable support

**Enterprise Middleware**
- `APIKeyMiddleware` — Bearer token authentication (`Authorization: Bearer <key>`); automatically disabled when `FAIFORGE_API_KEYS` env var is not set (zero-config dev mode)
- `RateLimitMiddleware` — per-key in-memory sliding window rate limiter; returns 429 with `Retry-After` header when limit exceeded; configurable via `FAIFORGE_RATE_LIMIT_REQUESTS` / `FAIFORGE_RATE_LIMIT_WINDOW`

**Async Ingestion**
- `POST /v1/rag/ingest?background=true` — returns 202 with `job_id` immediately, ingests in background
- `GET /v1/rag/jobs/{job_id}` — poll job status (`pending` → `running` → `done` | `error`)

---

## [2.4.0] - 2026-02-15

### Added

**Evals Framework**
- `LLMJudge` — generic 0–10 AI scorer using any registered `LLMAdapter`; JSON output with regex fallback
- RAG eval metrics (Ragas-first, LLM-judge fallback): `FaithfulnessMetric`, `AnswerRelevancyMetric`, `ContextPrecisionMetric`, `ContextRecallMetric`
- `RAGEvalPipeline` — runs multiple metrics concurrently with `asyncio.gather`, graceful per-metric error handling
- `InMemoryEvalStore` — feedback storage with `EvalStoreBackend` Protocol for easy swap-out (SQLite, Postgres, etc.)
- `ABRouter` — round-robin and weighted-random A/B variant routing with per-variant stats tracking

**Evals API endpoints**
- `POST /v1/evals/judge` — score any response with LLM judge
- `POST /v1/evals/feedback` — store thumbs up/down feedback
- `GET /v1/evals/feedback` — retrieve feedback log
- `POST /v1/evals/rag` — run RAG evaluation metrics
- `POST /v1/evals/ab/experiments` — create A/B experiment
- `GET /v1/evals/ab/experiments/{id}/stats` — view per-variant stats

**Frontend (Evals tab)**
- `RateThisPanel` — thumbs up/down + AI judge button under each assistant message; score badge with color coding (green ≥7, amber 4–7, red <4)
- `EvalsTab` — feedback log + A/B experiment stats panel
- `RAGTab Documents` panel — browse ingested chunks, delete individual chunks

---

## [2.3.0] - 2026-01-20

### Added

**RAG-embedded Chat**
- `use_rag=true` query param on `POST /v1/chat/completions` — retrieves top-5 relevant chunks and injects them as system context
- `sources` field on `CompletionResponse` — array of source chunks with content, score, and metadata
- Frontend RAG toggle button (purple) in Chat tab; collapsible sources panel under each assistant message

**Cross-Encoder Reranker**
- `CrossEncoderReranker` — post-retrieval reranking with lazy model loading (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `use_rerank=true` query param on RAG query endpoint

**Document Management API**
- `GET /v1/rag/documents` — list all ingested document chunks with metadata
- `DELETE /v1/rag/documents/{chunk_id}` — delete a specific chunk

**Cache bypass for RAG**
- Cache lookup skipped when `use_rag=true` to ensure fresh retrieval on every request

---

## [2.2.0] - 2025-12-10

### Added

**Hybrid Search**
- BM25 sparse retrieval combined with semantic dense retrieval
- RRF (Reciprocal Rank Fusion) and weighted fusion methods
- `search_mode` parameter on RAG query: `semantic` | `bm25` | `hybrid`
- Configurable `semantic_weight` / `bm25_weight`

**Ollama Adapter**
- Local inference via Ollama (`ollama/llama3`, `ollama/mistral`, `ollama/phi3`, any Ollama model)
- Streaming support, configurable base URL

**RAG Frontend UI**
- RAG tab: document upload, paste text, search mode toggle, scored results display
- Stats tab: live cache stats, RAG corpus stats, cache clear button

---

## [Unreleased]

### Planned

**Agentic**
- MCP (Model Context Protocol) integration
- Agent framework (LangGraph / tool-using agents)
- Human-in-the-loop approval workflows

**Observability**
- LangSmith / Weights & Biases tracing integration

**Edge AI**
- llama.cpp / ONNX quantized model support

**Multimodal**
- Vision support (GPT-4o Vision, Gemini Vision)
- Audio / speech input processing

**Governance**
- Prompt injection defense
- PII masking
- Content moderation guardrails

---

## Version History

### Versioning Strategy

FAIForge follows [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for new functionality (backwards compatible)
- **PATCH** version for backwards compatible bug fixes

### Release Notes

Detailed release notes are available in [docs/RELEASE_NOTES_v1.0.0.md](docs/RELEASE_NOTES_v1.0.0.md)

---


[2.5.0]: https://github.com/fiddyrod/faiforge/releases/tag/v2.5.0
[2.4.0]: https://github.com/fiddyrod/faiforge/releases/tag/v2.4.0
[2.3.0]: https://github.com/fiddyrod/faiforge/releases/tag/v2.3.0
[2.2.0]: https://github.com/fiddyrod/faiforge/releases/tag/v2.2.0
[1.0.0]: https://github.com/fiddyrod/faiforge/releases/tag/v1.0.0
