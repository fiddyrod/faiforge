from .base import (
    LLMAdapter,
    Message,
    Response,
    StreamChunk,
    Tool,
    FunctionDef,
    ToolCall,
    FunctionCallResult,
    ResponseFormat,
)
from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter

# Optional vLLM import (requires GPU)
try:
    from .vllm_adapter import VLLMAdapter
    VLLM_AVAILABLE = True
except ImportError:
    VLLMAdapter = None
    VLLM_AVAILABLE = False

__all__ = [
    # Base classes and types
    "LLMAdapter",
    "Message",
    "Response",
    "StreamChunk",
    # Tool/Function calling
    "Tool",
    "FunctionDef",
    "ToolCall",
    "FunctionCallResult",
    # Structured outputs
    "ResponseFormat",
    # Adapters
    "OpenAIAdapter",
    "AnthropicAdapter",
    "VLLMAdapter",
    "VLLM_AVAILABLE",
]