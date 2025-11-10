from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from ..inference.registry import load_registry
from ..inference.adapters import Message
from ..config import AppConfig


def create_app(
    config: AppConfig,
    openai_api_key: str,
    anthropic_api_key: str = None
) -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="FAIForge API",
        version="0.3.0",  # Bumped version!
        description="Production-ready AI boilerplate - Configuration-driven"
    )
    
    # CORS - now from config!
    if config.cors.enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors.origins,
            allow_credentials=config.cors.allow_credentials,
            allow_methods=config.cors.allow_methods,
            allow_headers=config.cors.allow_headers,
        )
        print(f"✅ CORS enabled for origins: {config.cors.origins}")
    
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
        model: str = config.defaults.model  # ← From config!
        temperature: float = config.defaults.temperature  # ← From config!
        max_tokens: int = config.defaults.max_tokens  # ← From config!
    
    class CompletionResponse(BaseModel):
        content: str
        model: str
        usage: dict
        cost_usd: float
        latency_ms: float
    
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
        return {
            "status": "healthy",
            "models_loaded": len(registry.list())
        }
    
    @app.get("/v1/models")
    async def list_models():
        """List all available models"""
        return {"models": registry.list()}
    
    @app.post("/v1/chat/completions", response_model=CompletionResponse)
    async def create_completion(request: CompletionRequest):
        """
        Generate a chat completion.
        
        This is the main endpoint for generating LLM responses.
        """
        try:
            # Get the appropriate adapter
            adapter = registry.get(request.model)
            
            # Convert to internal message format
            messages = [
                Message(role=msg.role, content=msg.content)
                for msg in request.messages
            ]
            
            # Generate completion
            response = await adapter.complete(
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            # Return response with metadata
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
        
        except ValueError as e:
            # Model not found
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            # Other errors
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
    return app