import time
import json
from typing import List, Optional, Dict, Any, AsyncGenerator
from anthropic import AsyncAnthropic

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


class AnthropicAdapter(LLMAdapter):
    """Adapter for Anthropic Claude API with streaming, function calling, and structured outputs"""

    # Pricing per 1M tokens (January 2025)
    PRICING = {
        "claude-sonnet-4-5-20250929": {
            "input": 3.00 / 1_000_000,
            "output": 15.00 / 1_000_000
        },
        "claude-opus-4-20250514": {
            "input": 15.00 / 1_000_000,
            "output": 75.00 / 1_000_000
        },
    }

    def __init__(self, api_key: str, model: str, timeout: float = 60.0):
        """
        Initialize Anthropic adapter.

        Args:
            api_key: Anthropic API key
            model: Model name (e.g., 'claude-sonnet-4-5-20250929')
            timeout: Request timeout in seconds (default: 60)
        """
        self.client = AsyncAnthropic(api_key=api_key, timeout=timeout)
        self.model = model
        self.logger = get_logger()

        if model not in self.PRICING:
            self.logger.warning(f"Unknown model '{model}', cost calculation may be inaccurate")

    def _convert_messages(self, messages: List[Message]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert FAIForge messages to Anthropic format.

        Returns:
            Tuple of (system_message, anthropic_messages)
        """
        system_message = None
        anthropic_messages = []

        for msg in messages:
            # Extract system message (Anthropic handles separately)
            if msg.role == "system":
                system_message = msg.content
                continue

            # Handle tool result messages
            if msg.role == "tool" and msg.tool_call_id:
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id,
                        "content": msg.content
                    }]
                })
                continue

            # Handle assistant messages with tool calls
            if msg.role == "assistant" and msg.tool_calls:
                content_blocks = []

                # Add text content if present
                if msg.content:
                    content_blocks.append({
                        "type": "text",
                        "text": msg.content
                    })

                # Add tool use blocks
                for tc in msg.tool_calls:
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.function.name,
                        "input": json.loads(tc.function.arguments)
                    })

                anthropic_messages.append({
                    "role": "assistant",
                    "content": content_blocks
                })
                continue

            # Handle regular messages
            anthropic_messages.append({
                "role": msg.role,
                "content": msg.content
            })

        return system_message, anthropic_messages

    def _convert_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """Convert FAIForge tools to Anthropic format"""
        return [
            {
                "name": tool.function.name,
                "description": tool.function.description,
                "input_schema": tool.function.parameters
            }
            for tool in tools
            if tool.function is not None
        ]

    def _convert_tool_choice(self, tool_choice: str | Dict[str, Any]) -> Dict[str, Any]:
        """Convert tool_choice to Anthropic format"""
        if tool_choice == "auto":
            return {"type": "auto"}
        elif tool_choice == "none":
            return {"type": "none"}
        elif tool_choice == "required":
            return {"type": "any"}
        elif isinstance(tool_choice, dict):
            # Specific tool: {"type": "function", "function": {"name": "func_name"}}
            if "function" in tool_choice and "name" in tool_choice["function"]:
                return {"type": "tool", "name": tool_choice["function"]["name"]}
        return {"type": "auto"}

    def _parse_tool_calls(self, content_blocks) -> tuple[Optional[str], Optional[List[ToolCall]]]:
        """
        Parse Anthropic response content blocks.

        Returns:
            Tuple of (text_content, tool_calls)
        """
        text_content = None
        tool_calls = []

        for block in content_blocks:
            if block.type == "text":
                text_content = block.text
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    type="function",
                    function=FunctionCallResult(
                        name=block.name,
                        arguments=json.dumps(block.input)
                    )
                ))

        return text_content, tool_calls if tool_calls else None

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD"""
        pricing = self.PRICING.get(self.model, {"input": 0, "output": 0})
        return input_tokens * pricing["input"] + output_tokens * pricing["output"]

    def _build_system_prompt(
        self,
        base_system: Optional[str],
        response_format: Optional[ResponseFormat]
    ) -> Optional[str]:
        """Build system prompt with optional JSON mode instructions"""
        parts = []

        if base_system:
            parts.append(base_system)

        # Anthropic doesn't have native JSON mode - use system prompt
        if response_format and response_format.type in ("json_object", "json_schema"):
            json_instruction = "\n\nIMPORTANT: You must respond with valid JSON only. No additional text or explanation."

            if response_format.type == "json_schema" and response_format.json_schema:
                schema_str = json.dumps(response_format.json_schema, indent=2)
                json_instruction += f"\n\nYour response must conform to this JSON schema:\n{schema_str}"

            parts.append(json_instruction)

        return "\n".join(parts) if parts else None

    async def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 500,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[ResponseFormat] = None
    ) -> Response:
        """Generate completion using Anthropic API"""

        start_time = time.time()

        log_with_context(
            self.logger,
            "info",
            "Starting Anthropic request",
            event="llm_request_start",
            provider="anthropic",
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            message_count=len(messages),
            has_tools=tools is not None,
            has_response_format=response_format is not None
        )

        try:
            # Convert messages
            system_message, anthropic_messages = self._convert_messages(messages)

            # Build system prompt (include JSON mode if needed)
            system_prompt = self._build_system_prompt(system_message, response_format)

            # Build request kwargs
            kwargs = {
                "model": self.model,
                "messages": anthropic_messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            # Add tools if provided
            if tools:
                kwargs["tools"] = self._convert_tools(tools)
                if tool_choice:
                    kwargs["tool_choice"] = self._convert_tool_choice(tool_choice)

            # Call Anthropic API
            response = await self.client.messages.create(**kwargs)

            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000

            # Parse response content
            text_content, tool_calls = self._parse_tool_calls(response.content)

            # Get token usage
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            # Calculate cost
            cost_usd = self._calculate_cost(input_tokens, output_tokens)

            # Determine finish reason
            finish_reason = response.stop_reason or "stop"
            if finish_reason == "tool_use":
                finish_reason = "tool_calls"
            elif finish_reason == "end_turn":
                finish_reason = "stop"

            log_with_context(
                self.logger,
                "info",
                "Anthropic request completed",
                event="llm_request_complete",
                provider="anthropic",
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=round(cost_usd, 6),
                latency_ms=round(latency_ms, 2),
                finish_reason=finish_reason,
                has_tool_calls=tool_calls is not None,
                status="success"
            )

            return Response(
                content=text_content,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
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
                f"Anthropic request failed: {str(e)}",
                event="llm_request_error",
                provider="anthropic",
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
        """Generate streaming completion using Anthropic API"""

        start_time = time.time()

        log_with_context(
            self.logger,
            "info",
            "Starting Anthropic streaming request",
            event="llm_stream_start",
            provider="anthropic",
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            message_count=len(messages),
            has_tools=tools is not None
        )

        try:
            # Convert messages
            system_message, anthropic_messages = self._convert_messages(messages)
            system_prompt = self._build_system_prompt(system_message, response_format)

            # Build request kwargs
            kwargs = {
                "model": self.model,
                "messages": anthropic_messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            if system_prompt:
                kwargs["system"] = system_prompt

            if tools:
                kwargs["tools"] = self._convert_tools(tools)
                if tool_choice:
                    kwargs["tool_choice"] = self._convert_tool_choice(tool_choice)

            # Track state across stream events
            accumulated_tool_calls: Dict[str, Dict[str, Any]] = {}
            current_tool_id = None
            input_tokens = None
            output_tokens = None
            finish_reason = None

            # Create streaming response
            async with self.client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    # Handle different event types
                    if event.type == "message_start":
                        if hasattr(event.message, 'usage'):
                            input_tokens = event.message.usage.input_tokens

                    elif event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            current_tool_id = event.content_block.id
                            accumulated_tool_calls[current_tool_id] = {
                                "id": current_tool_id,
                                "name": event.content_block.name,
                                "arguments": ""
                            }

                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            yield StreamChunk(content=event.delta.text)
                        elif event.delta.type == "input_json_delta":
                            if current_tool_id and current_tool_id in accumulated_tool_calls:
                                accumulated_tool_calls[current_tool_id]["arguments"] += event.delta.partial_json

                    elif event.type == "message_delta":
                        if hasattr(event, 'usage'):
                            output_tokens = event.usage.output_tokens
                        if hasattr(event.delta, 'stop_reason'):
                            finish_reason = event.delta.stop_reason

            # Parse accumulated tool calls
            final_tool_calls = None
            if accumulated_tool_calls:
                final_tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        type="function",
                        function=FunctionCallResult(
                            name=tc["name"],
                            arguments=tc["arguments"]
                        )
                    )
                    for tc in accumulated_tool_calls.values()
                ]

            # Map finish reason
            if finish_reason == "tool_use":
                finish_reason = "tool_calls"
            elif finish_reason == "end_turn":
                finish_reason = "stop"

            latency_ms = (time.time() - start_time) * 1000

            log_with_context(
                self.logger,
                "info",
                "Anthropic streaming completed",
                event="llm_stream_complete",
                provider="anthropic",
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
                f"Anthropic streaming failed: {str(e)}",
                event="llm_stream_error",
                provider="anthropic",
                model=self.model,
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round(latency_ms, 2),
                status="error"
            )
            raise
