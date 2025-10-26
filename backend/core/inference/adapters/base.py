from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass


@dataclass
class Message:
    """Single message in a conversation"""
    role: str  # 'user', 'assistant', or 'system'
    content: str


@dataclass
class Response:
    """LLM response with metadata"""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float


class LLMAdapter(ABC):
    """Base class for all LLM adapters"""
    
    @abstractmethod
    async def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Response:
        """
        Generate a completion for the given messages.
        
        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Response object with content and metadata
        """
        pass