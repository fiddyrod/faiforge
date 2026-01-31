from abc import ABC, abstractmethod
from typing import List, Optional, AsyncGenerator, Dict, Any
from dataclasses import dataclass, field


# =============================================================================
# Message Types
# =============================================================================

@dataclass
class Message:
    """Single message in a conversation"""
    role: str  # 'user', 'assistant', 'system', or 'tool'
    content: Optional[str] = None
    tool_call_id: Optional[str] = None  # For tool response messages
    tool_calls: Optional[List['ToolCall']] = None  # For assistant messages with tool calls


# =============================================================================
# Tool/Function Calling Types
# =============================================================================

@dataclass
class FunctionDef:
    """Definition of a callable function"""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema


@dataclass
class Tool:
    """Tool that can be called by the LLM"""
    type: str = "function"
    function: Optional[FunctionDef] = None


@dataclass
class FunctionCallResult:
    """Result of a function call request from LLM"""
    name: str
    arguments: str  # JSON string of arguments


@dataclass
class ToolCall:
    """Tool call requested by the LLM"""
    id: str
    type: str = "function"
    function: Optional[FunctionCallResult] = None


# =============================================================================
# Structured Output Types
# =============================================================================

@dataclass
class ResponseFormat:
    """
    Response format specification for structured outputs.

    Types:
        - "text": Default, unstructured text response
        - "json_object": Force valid JSON output
        - "json_schema": Force JSON matching specific schema
    """
    type: str = "text"  # "text", "json_object", or "json_schema"
    json_schema: Optional[Dict[str, Any]] = None  # Required when type="json_schema"


# =============================================================================
# Response Types
# =============================================================================

@dataclass
class Response:
    """LLM response with metadata"""
    content: Optional[str]  # Can be None if tool_calls present
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: str = "stop"  # "stop", "tool_calls", "length", "content_filter"


@dataclass
class StreamChunk:
    """
    Single chunk from a streaming response.

    During streaming:
        - content: Text delta (partial content)
        - tool_calls: Partial tool call data (accumulated)
        - finish_reason: None until stream ends

    Final chunk:
        - finish_reason: "stop", "tool_calls", etc.
    """
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    finish_reason: Optional[str] = None

    # Metadata (populated on final chunk or periodically)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None


# =============================================================================
# Base Adapter
# =============================================================================

class LLMAdapter(ABC):
    """
    Base class for all LLM adapters.

    Provides unified interface for:
        - Text completion (complete)
        - Streaming completion (complete_stream)
        - Function/tool calling
        - Structured outputs (JSON mode)
    """

    @abstractmethod
    async def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 500,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[ResponseFormat] = None
    ) -> Response:
        """
        Generate a completion for the given messages.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            tools: List of tools/functions the LLM can call
            tool_choice: Control tool usage - "auto", "none", "required",
                        or {"type": "function", "function": {"name": "func_name"}}
            response_format: Force structured output format (JSON mode)

        Returns:
            Response object with content and/or tool_calls
        """
        pass

    @abstractmethod
    async def complete_stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 500,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[ResponseFormat] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Generate a streaming completion for the given messages.

        Args:
            messages: List of conversation messages
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            tools: List of tools/functions the LLM can call
            tool_choice: Control tool usage
            response_format: Force structured output format

        Yields:
            StreamChunk objects with partial content as it arrives
        """
        pass
