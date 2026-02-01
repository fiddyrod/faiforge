import json
import traceback
import os
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
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


async def stream_generator(adapter, messages, temperature, max_tokens, tools, tool_choice, response_format):
    """Generate SSE stream from adapter"""
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
        version="2.0.0",
        description="Production-ready AI boilerplate with streaming, function calling, structured outputs, and semantic caching"
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
        features = ["streaming", "function_calling", "structured_outputs", "fallback_routing"]
        if semantic_cache:
            features.append("semantic_caching")

        log_with_context(
            logger,
            "info",
            "FAIForge API started",
            event="app_startup",
            version="2.0.0",
            models_loaded=len(registry.list()),
            models=registry.list(),
            features=features,
            cache_enabled=semantic_cache is not None
        )

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
            "version": "2.0.0",
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
        use_cache: bool = Query(default=True, description="Use semantic cache if available")
    ):
        """
        Create a chat completion.

        Supports:
        - Standard completion (stream=false)
        - Streaming via SSE (stream=true)
        - Function/tool calling (tools parameter)
        - Structured outputs (response_format parameter)
        - Semantic caching (use_cache=true, default)
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

        # Generate cache key from last user message (for cache lookup)
        cache_query = None
        if semantic_cache and use_cache and not stream and not tools:
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
                    response_format=response_format
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
                finish_reason=response.finish_reason
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
            log_with_context(
                logger, "error",
                f"Chat completion failed: {str(e)}",
                event="chat_completion_error",
                model=request.model,
                error=str(e),
                error_type=type(e).__name__
            )
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

    return app
