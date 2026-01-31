import json
import traceback
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from ..inference.registry import load_registry
from ..inference.adapters import (
    Message,
    Tool,
    FunctionDef,
    ToolCall,
    FunctionCallResult,
    ResponseFormat,
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
    anthropic_api_key: str = None
) -> FastAPI:
    """Create and configure FastAPI application"""

    logger = get_logger()

    app = FastAPI(
        title="FAIForge API",
        version="1.0.0",
        description="Production-ready AI boilerplate with streaming, function calling, and structured outputs"
    )

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
        log_with_context(
            logger,
            "info",
            "FAIForge API started",
            event="app_startup",
            version="1.0.0",
            models_loaded=len(registry.list()),
            models=registry.list(),
            features=["streaming", "function_calling", "structured_outputs"]
        )

    @app.on_event("shutdown")
    async def shutdown_event():
        log_with_context(logger, "info", "FAIForge API shutting down", event="app_shutdown")

    @app.get("/")
    async def root():
        return {
            "name": "FAIForge API",
            "version": "1.0.0",
            "status": "ok",
            "features": ["streaming", "function_calling", "structured_outputs"]
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

    @app.post("/v1/chat/completions")
    async def chat_completion(
        request: CompletionRequest,
        stream: bool = Query(default=False, description="Enable streaming response")
    ):
        """
        Create a chat completion.

        Supports:
        - Standard completion (stream=false)
        - Streaming via SSE (stream=true)
        - Function/tool calling (tools parameter)
        - Structured outputs (response_format parameter)
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

        # Handle streaming
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

            return CompletionResponse(
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
