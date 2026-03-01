import json
import traceback
import os
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

import yaml

from ..inference.registry import load_registry
from ..inference.adapters import (
    Message,
    Tool,
    FunctionDef,
    ToolCall,
    FunctionCallResult,
    ResponseFormat,
)
from ..cache import (
    SemanticCache,
    CacheConfig,
    create_cache_backend,
)
from ..config import AppConfig
from ..observability import RequestLoggingMiddleware, get_logger, log_with_context
from ..observability.metrics import (
    PrometheusMiddleware,
    metrics_response,
    llm_requests_total,
    llm_request_duration_seconds,
    llm_tokens_total,
    llm_cost_usd_total,
    cache_requests_total,
    rag_requests_total,
    rag_documents_ingested_total,
    rag_chunks_created_total,
    models_loaded as models_loaded_gauge,
)

# RAG imports (optional - graceful fallback if deps missing)
try:
    from ..rag import (
        RAGPipeline,
        SearchMode,
        HybridConfig,
    )
    from ..rag.embeddings.openai_embedding import OpenAIEmbedding
    from ..rag.chunking.recursive_chunker import RecursiveChunker
    from ..rag.vector_stores.chroma_store import ChromaStore
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# Reranker import (optional - requires sentence-transformers)
try:
    from ..rag.reranking import CrossEncoderReranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False


# =============================================================================
# Pydantic Models for API
# =============================================================================

class ChatMessage(BaseModel):
    """Chat message in request"""
    role: str = Field(..., pattern="^(user|assistant|system|tool)$", description="Message role")
    content: Optional[str] = Field(None, max_length=32000, description="Message content")
    tool_call_id: Optional[str] = Field(None, description="Tool call ID for tool responses")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls from assistant")


class FunctionDefRequest(BaseModel):
    """Function definition in request"""
    name: str = Field(..., min_length=1, max_length=64)
    description: str = Field(..., min_length=1, max_length=1024)
    parameters: Dict[str, Any] = Field(..., description="JSON Schema for parameters")


class ToolRequest(BaseModel):
    """Tool definition in request"""
    type: str = Field(default="function")
    function: FunctionDefRequest


class ResponseFormatRequest(BaseModel):
    """Response format specification"""
    type: str = Field(default="text", pattern="^(text|json_object|json_schema)$")
    json_schema: Optional[Dict[str, Any]] = None


class CompletionRequest(BaseModel):
    """Chat completion request"""
    messages: List[ChatMessage] = Field(..., min_length=1, max_length=50)
    model: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=500, ge=1, le=4000)
    tools: Optional[List[ToolRequest]] = None
    tool_choice: Optional[str | Dict[str, Any]] = None
    response_format: Optional[ResponseFormatRequest] = None


class ToolCallResponse(BaseModel):
    """Tool call in response"""
    id: str
    type: str
    function: Dict[str, str]


class CompletionResponse(BaseModel):
    """Chat completion response"""
    content: Optional[str]
    model: str
    usage: Dict[str, int]
    cost_usd: float
    latency_ms: float
    tool_calls: Optional[List[ToolCallResponse]] = None
    finish_reason: str = "stop"
    sources: Optional[List[Dict[str, Any]]] = None  # RAG source chunks when use_rag=true


# =============================================================================
# RAG API Models
# =============================================================================

class RAGDocument(BaseModel):
    """Document for RAG ingestion"""
    content: str = Field(..., min_length=1, max_length=100000)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGIngestRequest(BaseModel):
    """Request to ingest documents into RAG"""
    documents: List[RAGDocument] = Field(..., min_length=1, max_length=100)


class RAGIngestResponse(BaseModel):
    """Response from RAG ingestion"""
    documents_processed: int
    chunks_created: int
    embeddings_generated: int
    bm25_indexed: int
    total_latency_ms: float


class RAGQueryRequest(BaseModel):
    """Request to query RAG system"""
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    search_mode: str = Field(default="hybrid", pattern="^(semantic|bm25|hybrid)$")
    filters: Optional[Dict[str, Any]] = None


class RAGResult(BaseModel):
    """Single RAG search result"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    semantic_score: Optional[float] = None
    bm25_score: Optional[float] = None


class RAGQueryResponse(BaseModel):
    """Response from RAG query"""
    query: str
    results: List[RAGResult]
    total_results: int
    search_mode: str
    latency_ms: float


class RAGDocumentListItem(BaseModel):
    """A single document chunk in the list response"""
    id: str
    content: str
    metadata: Dict[str, Any]


class RAGDocumentListResponse(BaseModel):
    """Response from listing RAG documents"""
    documents: List[RAGDocumentListItem]
    total: int
    limit: int
    offset: int


class RAGDeleteBySourceRequest(BaseModel):
    """Delete documents by source label"""
    source: str = Field(..., min_length=1)


# =============================================================================
# Helper Functions
# =============================================================================

def convert_request_messages(messages: List[ChatMessage]) -> List[Message]:
    """Convert API messages to adapter Message format"""
    result = []
    for msg in messages:
        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc["id"],
                    type=tc.get("type", "function"),
                    function=FunctionCallResult(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"]
                    )
                )
                for tc in msg.tool_calls
            ]

        result.append(Message(
            role=msg.role,
            content=msg.content,
            tool_call_id=msg.tool_call_id,
            tool_calls=tool_calls
        ))
    return result


def convert_request_tools(tools: List[ToolRequest]) -> List[Tool]:
    """Convert API tools to adapter Tool format"""
    return [
        Tool(
            type=t.type,
            function=FunctionDef(
                name=t.function.name,
                description=t.function.description,
                parameters=t.function.parameters
            )
        )
        for t in tools
    ]


def convert_response_format(rf: ResponseFormatRequest) -> ResponseFormat:
    """Convert API response format to adapter format"""
    return ResponseFormat(
        type=rf.type,
        json_schema=rf.json_schema
    )


def format_tool_calls_response(tool_calls) -> Optional[List[ToolCallResponse]]:
    """Convert adapter tool calls to API response format"""
    if not tool_calls:
        return None
    return [
        ToolCallResponse(
            id=tc.id,
            type=tc.type,
            function={
                "name": tc.function.name,
                "arguments": tc.function.arguments
            }
        )
        for tc in tool_calls
    ]


async def stream_generator(
    adapter, messages, temperature, max_tokens, tools, tool_choice, response_format,
    rag_sources: Optional[List[Dict[str, Any]]] = None
):
    """Generate SSE stream from adapter.

    If rag_sources is provided (use_rag=True), a final 'sources' event is emitted
    after [DONE] so the client can display citations without waiting.
    """
    try:
        async for chunk in adapter.complete_stream(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format
        ):
            # Format as SSE
            data = {
                "content": chunk.content,
                "finish_reason": chunk.finish_reason,
            }

            # Include tool calls in final chunk
            if chunk.tool_calls:
                data["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in chunk.tool_calls
                ]

            # Include usage in final chunk
            if chunk.input_tokens is not None:
                data["usage"] = {
                    "prompt_tokens": chunk.input_tokens,
                    "completion_tokens": chunk.output_tokens,
                    "total_tokens": (chunk.input_tokens or 0) + (chunk.output_tokens or 0)
                }

            yield f"data: {json.dumps(data)}\n\n"

        # Send done signal
        yield "data: [DONE]\n\n"

        # Emit RAG sources as a final event after DONE
        if rag_sources:
            yield f"data: {json.dumps({'sources': rag_sources})}\n\n"

    except Exception as e:
        error_data = {"error": str(e), "type": type(e).__name__}
        yield f"data: {json.dumps(error_data)}\n\n"


# =============================================================================
# App Factory
# =============================================================================

def create_app(
    config: AppConfig,
    openai_api_key: str,
    anthropic_api_key: str = None,
    embedding_fn=None,
    routing_config_path: str = "core/config/routing.yaml"
) -> FastAPI:
    """Create and configure FastAPI application"""

    logger = get_logger()

    app = FastAPI(
        title="FAIForge API",
        version="2.3.0",
        description="Production-ready AI boilerplate with streaming, function calling, structured outputs, semantic caching, hybrid RAG, and cross-encoder reranking"
    )

    # =============================================================================
    # Initialize Semantic Cache
    # =============================================================================
    semantic_cache: Optional[SemanticCache] = None
    cache_config: Optional[CacheConfig] = None

    try:
        if os.path.exists(routing_config_path):
            with open(routing_config_path, 'r') as f:
                routing_yaml = yaml.safe_load(f)

            if routing_yaml.get('cache', {}).get('enabled', False):
                cache_config = CacheConfig.from_dict(routing_yaml.get('cache', {}))
                backend = create_cache_backend(cache_config)

                # Use provided embedding function or create placeholder
                if embedding_fn:
                    semantic_cache = SemanticCache(
                        backend=backend,
                        embedding_fn=embedding_fn,
                        similarity_threshold=cache_config.similarity_threshold,
                        config=cache_config
                    )
                    logger.info(
                        f"Semantic cache enabled: backend={cache_config.backend}, "
                        f"threshold={cache_config.similarity_threshold}"
                    )
                else:
                    logger.warning("Cache enabled but no embedding function provided - cache disabled")
    except Exception as e:
        logger.warning(f"Failed to initialize semantic cache: {e}")

    # =============================================================================
    # RAG Pipeline + Reranker (initialized on startup)
    # =============================================================================
    rag_pipeline: Optional["RAGPipeline"] = None
    reranker: Optional["CrossEncoderReranker"] = None

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        log_with_context(
            logger,
            "error",
            f"Unhandled exception: {str(exc)}",
            event="unhandled_exception",
            path=request.url.path,
            method=request.method,
            exception_type=type(exc).__name__,
            traceback=traceback.format_exc()
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error_type": type(exc).__name__}
        )

    # HTTP exception handler
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        log_with_context(
            logger,
            "warning" if exc.status_code < 500 else "error",
            f"HTTP exception: {exc.detail}",
            event="http_exception",
            path=request.url.path,
            method=request.method,
            status_code=exc.status_code,
            detail=exc.detail
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )

    # CORS middleware
    if config.cors.enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors.origins,
            allow_credentials=config.cors.allow_credentials,
            allow_methods=config.cors.allow_methods,
            allow_headers=config.cors.allow_headers,
        )
        logger.info(f"CORS enabled for origins: {config.cors.origins}")

    # Request logging middleware
    app.add_middleware(RequestLoggingMiddleware)
    logger.info("Request logging middleware enabled")

    # Prometheus metrics middleware
    app.add_middleware(PrometheusMiddleware)

    # Load model registry
    logger.info(f"Loading models from {config.models.config_path}")
    registry = load_registry(
        config.models.config_path,
        openai_api_key,
        anthropic_api_key,
        load_vllm=config.models.load_vllm
    )
    logger.info(f"Loaded {len(registry.list())} models: {', '.join(registry.list())}")

    # Update default model in request
    CompletionRequest.model_fields['model'].default = config.defaults.model
    CompletionRequest.model_fields['temperature'].default = config.defaults.temperature
    CompletionRequest.model_fields['max_tokens'].default = config.defaults.max_tokens

    # =============================================================================
    # Routes
    # =============================================================================

    @app.on_event("startup")
    async def startup_event():
        nonlocal rag_pipeline, reranker

        features = ["streaming", "function_calling", "structured_outputs", "fallback_routing"]
        if semantic_cache:
            features.append("semantic_caching")

        # Initialize RAG pipeline
        if RAG_AVAILABLE and openai_api_key:
            try:
                embedding_adapter = OpenAIEmbedding(
                    api_key=openai_api_key,
                    model="text-embedding-3-small"
                )
                chunking_adapter = RecursiveChunker(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                vector_store = ChromaStore(
                    collection_name="faiforge_documents",
                    persist_directory="./data/chroma_db"
                )
                await vector_store.initialize()

                rag_pipeline = RAGPipeline(
                    embedding_adapter=embedding_adapter,
                    chunking_adapter=chunking_adapter,
                    vector_store_adapter=vector_store,
                    hybrid_config=HybridConfig(
                        enabled=True,
                        semantic_weight=0.5,
                        bm25_weight=0.5,
                        fusion_method="rrf"
                    )
                )
                features.append("rag_hybrid_search")
                logger.info("RAG pipeline initialized with hybrid search")

                # Initialize reranker (lazy — model downloads on first use)
                if RERANKER_AVAILABLE:
                    try:
                        reranker = CrossEncoderReranker()
                        features.append("reranking")
                        logger.info("Cross-encoder reranker initialized (model loads on first use)")
                    except Exception as re:
                        logger.warning(f"Reranker init failed: {re}")
            except Exception as e:
                logger.warning(f"RAG pipeline init failed: {e}")

        log_with_context(
            logger,
            "info",
            "FAIForge API started",
            event="app_startup",
            version="2.3.0",
            models_loaded=len(registry.list()),
            models=registry.list(),
            features=features,
            cache_enabled=semantic_cache is not None,
            rag_enabled=rag_pipeline is not None
        )
        models_loaded_gauge.set(len(registry.list()))

    @app.on_event("shutdown")
    async def shutdown_event():
        if semantic_cache:
            await semantic_cache.close()
        log_with_context(logger, "info", "FAIForge API shutting down", event="app_shutdown")

    @app.get("/")
    async def root():
        features = ["streaming", "function_calling", "structured_outputs", "fallback_routing"]
        if semantic_cache:
            features.append("semantic_caching")

        return {
            "name": "FAIForge API",
            "version": "2.3.0",
            "status": "ok",
            "features": features
        }

    @app.get("/health")
    async def health():
        models_count = len(registry.list())
        log_with_context(
            logger, "debug", "Health check requested",
            event="health_check", models_loaded=models_count
        )
        return {"status": "healthy", "models_loaded": models_count}

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics():
        """Prometheus metrics endpoint (scrape target for prometheus.yml)."""
        return metrics_response()

    @app.get("/v1/models")
    async def list_models():
        """List all available models and fallback chains"""
        return {
            "models": registry.list_models(),
            "fallback_chains": registry.list_chains(),
            "all": registry.list()
        }

    @app.get("/v1/health/providers")
    async def provider_health():
        """
        Get health status of all providers in fallback chains.

        Returns circuit breaker status, consecutive failures, and last error for each provider.
        """
        health_status = registry.get_health_status()
        log_with_context(
            logger, "debug", "Provider health check requested",
            event="provider_health_check"
        )
        return {
            "status": "ok",
            "providers": health_status
        }

    # =============================================================================
    # Cache Endpoints
    # =============================================================================

    @app.get("/v1/cache/stats")
    async def cache_stats():
        """
        Get semantic cache statistics.

        Returns hit rate, total queries, cache size, and average similarity on hits.
        """
        if not semantic_cache:
            return {
                "enabled": False,
                "message": "Semantic cache is not enabled"
            }

        stats = await semantic_cache.get_stats()
        return {
            "enabled": True,
            "backend": cache_config.backend if cache_config else "unknown",
            "similarity_threshold": semantic_cache.similarity_threshold,
            **stats
        }

    @app.post("/v1/cache/clear")
    async def cache_clear():
        """
        Clear all semantic cache entries.

        Returns the number of entries that were cleared.
        """
        if not semantic_cache:
            raise HTTPException(
                status_code=400,
                detail="Semantic cache is not enabled"
            )

        count = await semantic_cache.clear()
        log_with_context(
            logger, "info",
            f"Cache cleared: {count} entries removed",
            event="cache_clear",
            entries_cleared=count
        )
        return {
            "status": "ok",
            "entries_cleared": count
        }

    @app.post("/v1/chat/completions")
    async def chat_completion(
        request: CompletionRequest,
        stream: bool = Query(default=False, description="Enable streaming response"),
        use_cache: bool = Query(default=True, description="Use semantic cache if available"),
        use_rag: bool = Query(default=False, description="Retrieve context from RAG pipeline and inject into prompt"),
        use_rerank: bool = Query(default=False, description="Apply cross-encoder reranking to RAG results"),
    ):
        """
        Create a chat completion.

        Supports:
        - Standard completion (stream=false)
        - Streaming via SSE (stream=true)
        - Function/tool calling (tools parameter)
        - Structured outputs (response_format parameter)
        - Semantic caching (use_cache=true, default)
        - RAG context injection (use_rag=true)
        """

        # Validate model exists
        if request.model not in registry.list():
            log_with_context(
                logger, "warning",
                f"Invalid model requested: {request.model}",
                event="invalid_model",
                requested_model=request.model,
                available_models=registry.list()
            )
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' not found. Available: {', '.join(registry.list())}"
            )

        # Convert request to adapter format
        messages = convert_request_messages(request.messages)
        tools = convert_request_tools(request.tools) if request.tools else None
        response_format = convert_response_format(request.response_format) if request.response_format else None

        # Get adapter
        adapter = registry.get(request.model)

        # RAG context injection
        rag_sources: List[Dict[str, Any]] = []
        if use_rag and rag_pipeline:
            user_query = next(
                (m.content for m in reversed(messages) if m.role == "user"), None
            )
            if user_query:
                try:
                    # Fetch more candidates when reranking so we can trim to top-5 after
                    retrieval_top_k = 10 if (use_rerank and reranker) else 5
                    rag_resp = await rag_pipeline.query(query_text=user_query, top_k=retrieval_top_k)
                    if rag_resp.results:
                        final_results = rag_resp.results

                        # Rerank if requested and available
                        if use_rerank and reranker:
                            try:
                                reranked = await reranker.rerank(
                                    query=user_query,
                                    results=rag_resp.results,
                                    top_k=5,
                                )
                                # Convert RerankResult back to a duck-typed list
                                final_results = reranked
                                log_with_context(
                                    logger, "info",
                                    "Reranking applied to RAG results",
                                    event="reranker_applied",
                                    candidates=len(rag_resp.results),
                                    top_k=len(final_results),
                                )
                            except Exception as re:
                                logger.warning(f"Reranking failed (using original order): {re}")

                        context_blocks = "\n\n---\n\n".join(
                            f"[Source {i + 1}]\n{r.content}"
                            for i, r in enumerate(final_results)
                        )
                        system_content = (
                            "Answer the user's question using the context below. "
                            "If the context is not relevant, answer from your own knowledge.\n\n"
                            f"Context:\n{context_blocks}"
                        )
                        # Prepend system message (replaces any existing system message)
                        messages = [Message(role="system", content=system_content)] + [
                            m for m in messages if m.role != "system"
                        ]
                        rag_sources = [
                            {
                                "content": r.content[:300],
                                "score": r.score,
                                "metadata": {k: v for k, v in r.metadata.items()
                                             if not k.startswith("_")},
                            }
                            for r in final_results
                        ]
                        rag_requests_total.labels(operation="query", status="success").inc()
                        log_with_context(
                            logger, "info",
                            f"RAG retrieved {len(rag_sources)} sources for chat",
                            event="rag_chat_retrieval",
                            sources=len(rag_sources),
                            reranked=use_rerank and reranker is not None,
                            model=request.model
                        )
                except Exception as e:
                    logger.warning(f"RAG retrieval failed (continuing without context): {e}")

        # Generate cache key from last user message (for cache lookup)
        # Skip cache when use_rag=True — RAG responses are context-dependent and must not be served stale
        cache_query = None
        if semantic_cache and use_cache and not stream and not tools and not use_rag:
            # Only cache simple queries without tools
            user_messages = [m for m in request.messages if m.role == "user"]
            if user_messages:
                cache_query = user_messages[-1].content

        # Check cache for hit
        if cache_query:
            try:
                cache_result = await semantic_cache.get(
                    cache_query,
                    metadata_filter={"model": request.model}
                )
                if cache_result:
                    cached_response, similarity = cache_result
                    log_with_context(
                        logger, "info",
                        f"Cache hit: similarity={similarity:.4f}",
                        event="cache_hit",
                        similarity=similarity,
                        model=request.model
                    )
                    cache_requests_total.labels(result="hit").inc()
                    return CompletionResponse(
                        content=cached_response.get("content"),
                        model=cached_response.get("model", request.model),
                        usage=cached_response.get("usage", {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0
                        }),
                        cost_usd=0.0,  # No cost for cached response
                        latency_ms=0.0,  # Near-instant
                        tool_calls=cached_response.get("tool_calls"),
                        finish_reason=cached_response.get("finish_reason", "stop")
                    )
                else:
                    cache_requests_total.labels(result="miss").inc()
            except Exception as e:
                logger.warning(f"Cache lookup failed: {e}")

        # Handle streaming (not cached)
        if stream:
            return StreamingResponse(
                stream_generator(
                    adapter=adapter,
                    messages=messages,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    tools=tools,
                    tool_choice=request.tool_choice,
                    response_format=response_format,
                    rag_sources=rag_sources if rag_sources else None,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no"  # Disable nginx buffering
                }
            )

        # Non-streaming completion
        try:
            response = await adapter.complete(
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tools=tools,
                tool_choice=request.tool_choice,
                response_format=response_format
            )

            llm_requests_total.labels(model=request.model, status="success").inc()
            llm_request_duration_seconds.labels(model=request.model).observe(response.latency_ms / 1000)
            llm_tokens_total.labels(model=request.model, token_type="prompt").inc(response.input_tokens)
            llm_tokens_total.labels(model=request.model, token_type="completion").inc(response.output_tokens)
            llm_cost_usd_total.labels(model=request.model).inc(response.cost_usd)

            response_data = CompletionResponse(
                content=response.content,
                model=response.model,
                usage={
                    "prompt_tokens": response.input_tokens,
                    "completion_tokens": response.output_tokens,
                    "total_tokens": response.input_tokens + response.output_tokens
                },
                cost_usd=response.cost_usd,
                latency_ms=response.latency_ms,
                tool_calls=format_tool_calls_response(response.tool_calls),
                finish_reason=response.finish_reason,
                sources=rag_sources if rag_sources else None,
            )

            # Store in cache on miss (only for simple queries)
            if cache_query and semantic_cache:
                try:
                    await semantic_cache.set(
                        cache_query,
                        {
                            "content": response.content,
                            "model": response.model,
                            "usage": {
                                "prompt_tokens": response.input_tokens,
                                "completion_tokens": response.output_tokens,
                                "total_tokens": response.input_tokens + response.output_tokens
                            },
                            "finish_reason": response.finish_reason
                        },
                        metadata={"model": request.model}
                    )
                except Exception as e:
                    logger.warning(f"Cache store failed: {e}")

            return response_data

        except Exception as e:
            llm_requests_total.labels(model=request.model, status="error").inc()
            log_with_context(
                logger, "error",
                f"Chat completion failed: {str(e)}",
                event="chat_completion_error",
                model=request.model,
                error=str(e),
                error_type=type(e).__name__
            )
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    # =============================================================================
    # RAG Endpoints
    # =============================================================================

    @app.post("/v1/rag/ingest", response_model=RAGIngestResponse)
    async def rag_ingest(request: RAGIngestRequest):
        """
        Ingest documents into the RAG system.

        Chunks, embeds, stores in vector DB, and indexes for BM25.
        """
        if not rag_pipeline:
            raise HTTPException(
                status_code=503,
                detail="RAG pipeline not available. Check OpenAI API key and chromadb installation."
            )

        try:
            docs = [
                {"content": doc.content, "metadata": doc.metadata}
                for doc in request.documents
            ]
            result = await rag_pipeline.ingest_documents(docs)

            rag_requests_total.labels(operation="ingest", status="success").inc()
            rag_documents_ingested_total.inc(result["documents_processed"])
            rag_chunks_created_total.inc(result["chunks_created"])

            return RAGIngestResponse(
                documents_processed=result["documents_processed"],
                chunks_created=result["chunks_created"],
                embeddings_generated=result["embeddings_generated"],
                bm25_indexed=result.get("bm25_indexed", 0),
                total_latency_ms=result["total_latency_ms"]
            )
        except Exception as e:
            rag_requests_total.labels(operation="ingest", status="error").inc()
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

    @app.post("/v1/rag/query", response_model=RAGQueryResponse)
    async def rag_query(request: RAGQueryRequest):
        """
        Query the RAG system.

        Supports semantic, BM25, and hybrid search modes.
        """
        if not rag_pipeline:
            raise HTTPException(
                status_code=503,
                detail="RAG pipeline not available."
            )

        try:
            mode_map = {
                "semantic": SearchMode.SEMANTIC,
                "bm25": SearchMode.BM25,
                "hybrid": SearchMode.HYBRID,
            }
            search_mode = mode_map.get(request.search_mode, SearchMode.HYBRID)

            response = await rag_pipeline.query(
                query_text=request.query,
                top_k=request.top_k,
                filters=request.filters,
                search_mode=search_mode
            )

            results = [
                RAGResult(
                    id=r.id,
                    content=r.content,
                    score=r.score,
                    metadata={k: v for k, v in r.metadata.items()
                               if not k.startswith("_")},
                    semantic_score=r.metadata.get("_semantic_score"),
                    bm25_score=r.metadata.get("_bm25_score")
                )
                for r in response.results
            ]

            rag_requests_total.labels(operation="query", status="success").inc()
            return RAGQueryResponse(
                query=request.query,
                results=results,
                total_results=len(results),
                search_mode=request.search_mode,
                latency_ms=response.latency_ms
            )
        except Exception as e:
            rag_requests_total.labels(operation="query", status="error").inc()
            raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    @app.get("/v1/rag/stats")
    async def rag_stats():
        """Get RAG system statistics."""
        if not rag_pipeline:
            return {"enabled": False, "reason": "RAG pipeline not initialized"}

        try:
            stats = await rag_pipeline.get_stats()
            stats["enabled"] = True
            return stats
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get RAG stats: {str(e)}")

    # =========================================================================
    # Document Management Endpoints
    # =========================================================================

    @app.get("/v1/rag/documents", response_model=RAGDocumentListResponse)
    async def list_rag_documents(
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        source: Optional[str] = Query(default=None, description="Filter by source label"),
    ):
        """List ingested document chunks with metadata."""
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not available.")
        try:
            filters = {"source": source} if source else None
            result = await rag_pipeline.vector_store.list_documents(
                limit=limit, offset=offset, filters=filters
            )
            return RAGDocumentListResponse(
                documents=[RAGDocumentListItem(**d) for d in result["documents"]],
                total=result["total"],
                limit=result["limit"],
                offset=result["offset"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

    @app.delete("/v1/rag/documents/{doc_id}")
    async def delete_rag_document(doc_id: str):
        """Delete a single document chunk by its ID."""
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not available.")
        try:
            result = await rag_pipeline.vector_store.delete(ids=[doc_id])
            log_with_context(
                logger, "info", f"Deleted RAG document: {doc_id}",
                event="rag_document_delete", doc_id=doc_id
            )
            return {"status": "ok", "deleted_id": doc_id, **result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

    @app.delete("/v1/rag/documents")
    async def delete_rag_documents_by_source(request: RAGDeleteBySourceRequest):
        """Delete all chunks that share a source label."""
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not available.")
        try:
            result = await rag_pipeline.vector_store.delete(
                filters={"source": request.source}
            )
            log_with_context(
                logger, "info", f"Deleted RAG documents by source: {request.source}",
                event="rag_document_delete_by_source", source=request.source
            )
            return {"status": "ok", "source": request.source, **result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

    @app.post("/v1/rag/documents/clear")
    async def clear_rag_documents():
        """Delete ALL document chunks from the vector store."""
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not available.")
        try:
            result = await rag_pipeline.vector_store.clear_collection()
            log_with_context(
                logger, "info", "RAG collection cleared",
                event="rag_collection_clear", **result
            )
            return {"status": "ok", **result}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")

    return app
