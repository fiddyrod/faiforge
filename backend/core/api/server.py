from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import traceback

from ..inference.registry import load_registry
from ..inference.adapters import Message
from ..config import AppConfig
from ..observability import RequestLoggingMiddleware, get_logger, log_with_context

def create_app(
    config: AppConfig,
    openai_api_key: str,
    anthropic_api_key: str = None
) -> FastAPI:
    """Create and configure FastAPI application"""
    
    logger = get_logger()
    
    app = FastAPI(
        title="FAIForge API",
        version="0.4.0",
        description="Production-ready AI boilerplate with observability"
    )
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Handle all unhandled exceptions"""
        
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
            content={
                "detail": "Internal server error",
                "error_type": type(exc).__name__
            }
        )
    
    # HTTP exception handler (for proper error responses)
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with logging"""
        
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
        logger.info("CORS enabled", extra={
            'extra_fields': {'origins': config.cors.origins}
        })
    
    # Request logging middleware 
    app.add_middleware(RequestLoggingMiddleware)
    logger.info("Request logging middleware enabled")

    
    # Load model registry with config
    print(f"🔄 Loading models from {config.models.config_path}...")
    registry = load_registry(
        config.models.config_path,
        openai_api_key,
        anthropic_api_key,
        load_vllm=config.models.load_vllm
    )
    print(f"✅ Loaded {len(registry.list())} models: {', '.join(registry.list())}")
    
    # Request/Response models - defaults from config!
    class ChatMessage(BaseModel):
        role: str
        content: str
    
    class CompletionRequest(BaseModel):
        messages: List[ChatMessage]
        model: str = config.defaults.model  
        temperature: float = config.defaults.temperature  
        max_tokens: int = config.defaults.max_tokens  
    
    class CompletionResponse(BaseModel):
        content: str
        model: str
        usage: dict
        cost_usd: float
        latency_ms: float

    # Startup event
    @app.on_event("startup")
    async def startup_event():
        """Log application startup"""
        log_with_context(
            logger,
            "info",
            "FAIForge API started",
            event="app_startup",
            version="0.4.0",
            models_loaded=len(registry.list()),
            models=registry.list()
        )
    
    # Shutdown event
    @app.on_event("shutdown")
    async def shutdown_event():
        """Log application shutdown"""
        log_with_context(
            logger,
            "info",
            "FAIForge API shutting down",
            event="app_shutdown"
        )
    
    # Routes
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "name": "GenAI Boilerplate",
            "version": "0.1.0",
            "status": "ok"
        }
    
    @app.get("/health")
    async def health():
        """Health check endpoint"""
        
        models_count = len(registry.list())
        
        # Only log health checks at DEBUG level to avoid spam
        log_with_context(
            logger,
            "debug",
            "Health check requested",
            event="health_check",
            models_loaded=models_count
        )
        
        return {
            "status": "healthy",
            "models_loaded": models_count
        }
    
    @app.get("/v1/models")
    async def list_models():
        """List all available models"""
        return {"models": registry.list()}
    
    @app.post("/v1/chat/completions", response_model=CompletionResponse)
    async def chat_completion(request: CompletionRequest):
        """
        Create a chat completion
        
        This endpoint accepts a list of messages and returns a completion
        from the specified model.
        """
        
        # Validate model exists
        if request.model not in registry.list():
            log_with_context(
                logger,
                "warning",
                f"Invalid model requested: {request.model}",
                event="invalid_model",
                requested_model=request.model,
                available_models=registry.list()
            )
            
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' not found. Available: {', '.join(registry.list())}"
            )
        
        # Convert request messages to adapter format
        messages = [
            Message(role=msg.role, content=msg.content)
            for msg in request.messages
        ]
        
        # Get adapter and generate completion
        try:
            adapter = registry.get(request.model)
            response = await adapter.complete(
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
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
                latency_ms=response.latency_ms
            )
        
        except Exception as e:
            # Log the error (already logged in adapter, but log at API level too)
            log_with_context(
                logger,
                "error",
                f"Chat completion failed: {str(e)}",
                event="chat_completion_error",
                model=request.model,
                error=str(e),
                error_type=type(e).__name__
            )
            
            # Re-raise as HTTP exception
            raise HTTPException(
                status_code=500,
                detail=f"Internal error: {str(e)}"
            )
    
    return app