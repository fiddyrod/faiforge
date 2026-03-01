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
from .ollama_adapter import OllamaAdapter

# Optional vLLM import (requires GPU)
try:
    from .vllm_adapter import VLLMAdapter
    VLLM_AVAILABLE = True
except ImportError:
    VLLMAdapter = None
    VLLM_AVAILABLE = False

# Optional Gemini import (requires google-generativeai)
try:
    from .gemini_adapter import GeminiAdapter
    GEMINI_AVAILABLE = True
except ImportError:
    GeminiAdapter = None
    GEMINI_AVAILABLE = False

# Optional Cohere import (requires cohere)
try:
    from .cohere_adapter import CohereAdapter
    COHERE_AVAILABLE = True
except ImportError:
    CohereAdapter = None
    COHERE_AVAILABLE = False

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
    "OllamaAdapter",
    "VLLMAdapter",
    "VLLM_AVAILABLE",
    "GeminiAdapter",
    "GEMINI_AVAILABLE",
    "CohereAdapter",
    "COHERE_AVAILABLE",
]