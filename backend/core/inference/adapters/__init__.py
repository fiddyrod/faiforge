from .base import LLMAdapter, Message, Response
from .openai_adapter import OpenAIAdapter
from .anthropic_adapter import AnthropicAdapter

# Optional vLLM import (requires GPU)
try:
    from .vllm_adapter import VLLMAdapter
    VLLM_AVAILABLE = True
except ImportError:
    VLLMAdapter = None
    VLLM_AVAILABLE = False

__all__ = ["LLMAdapter", "Message", "Response", "OpenAIAdapter", "AnthropicAdapter", "VLLMAdapter"]