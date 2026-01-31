import time
import json
from typing import List, Optional, Dict, Any, AsyncGenerator
from openai import AsyncOpenAI

from .base import (
    LLMAdapter,
    Message,
    Response,
    StreamChunk,
    Tool,
    ToolCall,
    FunctionCallResult,
    ResponseFormat,
)
from ...observability import get_logger, log_with_context


class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI API with streaming, function calling, and structured outputs"""

    # Pricing per 1M tokens (January 2025)
    PRICING = {
        "gpt-4o": {
            "input": 2.50 / 1_000_000,
            "output": 10.00 / 1_000_000
        },
        "gpt-4o-mini": {
            "input": 0.150 / 1_000_000,
            "output": 0.600 / 1_000_000
        },
    }

    def __init__(self, api_key: str, model: str, timeout: float = 60.0):
        """
        Initialize OpenAI adapter.

        Args:
            api_key: OpenAI API key
            model: Model name (e.g., 'gpt-4o-mini')
            timeout: Request timeout in seconds (default: 60)
        """
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        self.model = model
        self.logger = get_logger()

        if model not in self.PRICING:
            raise ValueError(f"Unknown model: {model}. Available: {list(self.PRICING.keys())}")

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert FAIForge messages to OpenAI format"""
        openai_messages = []

        for msg in messages:
            openai_msg: Dict[str, Any] = {"role": msg.role}

            # Handle content
            if msg.content is not None:
                openai_msg["content"] = msg.content

            # Handle tool call responses
            if msg.role == "tool" and msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id

            # Handle assistant messages with tool calls
            if msg.tool_calls:
                openai_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in msg.tool_calls
                ]

            openai_messages.append(openai_msg)

        return openai_messages

    def _convert_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """Convert FAIForge tools to OpenAI format"""
        return [
            {
                "type": tool.type,
                "function": {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": tool.function.parameters
                }
            }
            for tool in tools
            if tool.function is not None
        ]

    def _convert_response_format(self, response_format: ResponseFormat) -> Dict[str, Any]:
        """Convert FAIForge response format to OpenAI format"""
        if response_format.type == "json_object":
            return {"type": "json_object"}
        elif response_format.type == "json_schema" and response_format.json_schema:
            return {
                "type": "json_schema",
                "json_schema": response_format.json_schema
            }
        return {"type": "text"}

    def _parse_tool_calls(self, openai_tool_calls) -> List[ToolCall]:
        """Parse OpenAI tool calls to FAIForge format"""
        if not openai_tool_calls:
            return None

        return [
            ToolCall(
                id=tc.id,
                type=tc.type,
                function=FunctionCallResult(
                    name=tc.function.name,
                    arguments=tc.function.arguments
                )
            )
            for tc in openai_tool_calls
        ]

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD"""
        pricing = self.PRICING.get(self.model, {"input": 0, "output": 0})
        return input_tokens * pricing["input"] + output_tokens * pricing["output"]

    async def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 500,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[ResponseFormat] = None
    ) -> Response:
        """Generate completion using OpenAI API"""

        start_time = time.time()

        # Log request start
        log_with_context(
            self.logger,
            "info",
            "Starting OpenAI request",
            event="llm_request_start",
            provider="openai",
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            message_count=len(messages),
            has_tools=tools is not None,
            has_response_format=response_format is not None
        )

        try:
            # Build request kwargs
            kwargs = {
                "model": self.model,
                "messages": self._convert_messages(messages),
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            # Add tools if provided
            if tools:
                kwargs["tools"] = self._convert_tools(tools)
                if tool_choice:
                    kwargs["tool_choice"] = tool_choice

            # Add response format if provided
            if response_format and response_format.type != "text":
                kwargs["response_format"] = self._convert_response_format(response_format)

            # Call OpenAI API
            response = await self.client.chat.completions.create(**kwargs)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Extract response data
            choice = response.choices[0]
            message = choice.message
            usage = response.usage

            # Calculate cost
            cost_usd = self._calculate_cost(usage.prompt_tokens, usage.completion_tokens)

            # Parse tool calls if present
            tool_calls = self._parse_tool_calls(message.tool_calls)

            # Determine finish reason
            finish_reason = choice.finish_reason or "stop"
            if tool_calls:
                finish_reason = "tool_calls"

            # Log successful completion
            log_with_context(
                self.logger,
                "info",
                "OpenAI request completed successfully",
                event="llm_request_complete",
                provider="openai",
                model=self.model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.prompt_tokens + usage.completion_tokens,
                cost_usd=round(cost_usd, 6),
                latency_ms=round(latency_ms, 2),
                finish_reason=finish_reason,
                has_tool_calls=tool_calls is not None,
                status="success"
            )

            return Response(
                content=message.content,
                model=self.model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                cost_usd=round(cost_usd, 6),
                latency_ms=round(latency_ms, 2),
                tool_calls=tool_calls,
                finish_reason=finish_reason
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            log_with_context(
                self.logger,
                "error",
                f"OpenAI request failed: {str(e)}",
                event="llm_request_error",
                provider="openai",
                model=self.model,
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round(latency_ms, 2),
                status="error"
            )
            raise

    async def complete_stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 500,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[ResponseFormat] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """Generate streaming completion using OpenAI API"""

        start_time = time.time()

        log_with_context(
            self.logger,
            "info",
            "Starting OpenAI streaming request",
            event="llm_stream_start",
            provider="openai",
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            message_count=len(messages),
            has_tools=tools is not None
        )

        try:
            # Build request kwargs
            kwargs = {
                "model": self.model,
                "messages": self._convert_messages(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True}  # Get token counts at end
            }

            # Add tools if provided
            if tools:
                kwargs["tools"] = self._convert_tools(tools)
                if tool_choice:
                    kwargs["tool_choice"] = tool_choice

            # Add response format if provided
            if response_format and response_format.type != "text":
                kwargs["response_format"] = self._convert_response_format(response_format)

            # Create streaming response
            stream = await self.client.chat.completions.create(**kwargs)

            # Track tool calls across chunks (they come in pieces)
            accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}
            finish_reason = None
            input_tokens = None
            output_tokens = None

            async for chunk in stream:
                # Check for usage info (comes at the end)
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens

                # Process choices
                if chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta

                    # Check finish reason
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

                    # Handle content delta
                    content = delta.content if hasattr(delta, 'content') else None

                    # Handle tool call deltas
                    if hasattr(delta, 'tool_calls') and delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            idx = tc_delta.index

                            if idx not in accumulated_tool_calls:
                                accumulated_tool_calls[idx] = {
                                    "id": tc_delta.id or "",
                                    "type": tc_delta.type or "function",
                                    "function": {"name": "", "arguments": ""}
                                }

                            if tc_delta.id:
                                accumulated_tool_calls[idx]["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    accumulated_tool_calls[idx]["function"]["name"] = tc_delta.function.name
                                if tc_delta.function.arguments:
                                    accumulated_tool_calls[idx]["function"]["arguments"] += tc_delta.function.arguments

                    # Yield chunk if there's content
                    if content:
                        yield StreamChunk(content=content)

            # Parse accumulated tool calls
            final_tool_calls = None
            if accumulated_tool_calls:
                final_tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        type=tc["type"],
                        function=FunctionCallResult(
                            name=tc["function"]["name"],
                            arguments=tc["function"]["arguments"]
                        )
                    )
                    for tc in accumulated_tool_calls.values()
                ]
                finish_reason = "tool_calls"

            # Yield final chunk with metadata
            latency_ms = (time.time() - start_time) * 1000

            log_with_context(
                self.logger,
                "info",
                "OpenAI streaming completed",
                event="llm_stream_complete",
                provider="openai",
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=round(latency_ms, 2),
                finish_reason=finish_reason,
                has_tool_calls=final_tool_calls is not None,
                status="success"
            )

            yield StreamChunk(
                finish_reason=finish_reason or "stop",
                tool_calls=final_tool_calls,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            log_with_context(
                self.logger,
                "error",
                f"OpenAI streaming failed: {str(e)}",
                event="llm_stream_error",
                provider="openai",
                model=self.model,
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round(latency_ms, 2),
                status="error"
            )
            raise
