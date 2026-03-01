import time
import json
from typing import List, Optional, Dict, Any, AsyncGenerator

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

try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    cohere = None
    COHERE_AVAILABLE = False


class CohereAdapter(LLMAdapter):
    """Adapter for Cohere API with streaming and function calling.

    Supports:
        - command-r-plus  (highest capability, 128k context)
        - command-r       (balanced, 128k context)
        - command-light   (fast, low cost)

    Note: Cohere uses its own chat message format (role: "USER" / "CHATBOT").
    JSON mode is achieved via a system preamble instruction.

    Requires: pip install cohere
    """

    # Pricing per 1M tokens (March 2025)
    PRICING: Dict[str, Dict[str, float]] = {
        "command-r-plus": {
            "input": 2.50 / 1_000_000,
            "output": 10.00 / 1_000_000,
        },
        "command-r": {
            "input": 0.15 / 1_000_000,
            "output": 0.60 / 1_000_000,
        },
        "command-light": {
            "input": 0.30 / 1_000_000,
            "output": 0.60 / 1_000_000,
        },
    }

    def __init__(self, api_key: str, model: str, timeout: float = 60.0):
        if not COHERE_AVAILABLE:
            raise ImportError(
                "cohere not installed. Install with: pip install cohere"
            )
        self.client = cohere.AsyncClientV2(api_key=api_key, timeout=timeout)
        self.model = model
        self.logger = get_logger()

        if model not in self.PRICING:
            self.logger.warning(f"Unknown Cohere model '{model}', cost calculation may be inaccurate")

    def _convert_messages(
        self, messages: List[Message]
    ) -> tuple[Optional[str], List[Dict[str, str]]]:
        """Convert FAIForge messages to Cohere v2 chat format.

        Returns:
            (preamble, cohere_messages)
        """
        preamble = None
        cohere_messages = []

        for msg in messages:
            if msg.role == "system":
                preamble = msg.content
                continue
            # Cohere v2 uses "user" / "assistant"
            cohere_messages.append({
                "role": msg.role,
                "content": msg.content or "",
            })

        return preamble, cohere_messages

    def _convert_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """Convert FAIForge tools to Cohere v2 tool format."""
        cohere_tools = []
        for tool in tools:
            if tool.function is None:
                continue
            cohere_tools.append({
                "type": "function",
                "function": {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": tool.function.parameters,
                },
            })
        return cohere_tools

    def _parse_tool_calls(self, tool_calls_raw) -> Optional[List[ToolCall]]:
        """Parse Cohere tool call objects."""
        if not tool_calls_raw:
            return None
        result = []
        for tc in tool_calls_raw:
            result.append(ToolCall(
                id=tc.id if hasattr(tc, "id") else f"call_{tc.function.name}",
                type="function",
                function=FunctionCallResult(
                    name=tc.function.name,
                    arguments=tc.function.arguments
                    if isinstance(tc.function.arguments, str)
                    else json.dumps(tc.function.arguments),
                ),
            ))
        return result if result else None

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = self.PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        return input_tokens * pricing["input"] + output_tokens * pricing["output"]

    def _build_preamble(
        self,
        base_preamble: Optional[str],
        response_format: Optional[ResponseFormat],
    ) -> Optional[str]:
        """Cohere has no native JSON mode — instruct via preamble."""
        parts = []
        if base_preamble:
            parts.append(base_preamble)
        if response_format and response_format.type in ("json_object", "json_schema"):
            instruction = "IMPORTANT: Respond with valid JSON only. No additional text."
            if response_format.type == "json_schema" and response_format.json_schema:
                instruction += f"\n\nJSON schema:\n{json.dumps(response_format.json_schema, indent=2)}"
            parts.append(instruction)
        return "\n\n".join(parts) if parts else None

    async def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 500,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[ResponseFormat] = None,
    ) -> Response:
        start_time = time.time()

        log_with_context(
            self.logger, "info", "Starting Cohere request",
            event="llm_request_start", provider="cohere", model=self.model,
            temperature=temperature, max_tokens=max_tokens,
            message_count=len(messages), has_tools=tools is not None,
        )

        try:
            preamble, cohere_messages = self._convert_messages(messages)
            effective_preamble = self._build_preamble(preamble, response_format)

            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": cohere_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if effective_preamble:
                kwargs["preamble"] = effective_preamble
            if tools:
                kwargs["tools"] = self._convert_tools(tools)

            response = await self.client.chat(**kwargs)
            latency_ms = (time.time() - start_time) * 1000

            # Extract content
            text_content = None
            tool_calls = None
            message = response.message

            for block in (message.content or []):
                if hasattr(block, "text"):
                    text_content = block.text

            if hasattr(message, "tool_calls") and message.tool_calls:
                tool_calls = self._parse_tool_calls(message.tool_calls)

            input_tokens = response.usage.billed_units.input_tokens if response.usage else 0
            output_tokens = response.usage.billed_units.output_tokens if response.usage else 0
            cost_usd = self._calculate_cost(int(input_tokens), int(output_tokens))

            finish_reason = "stop"
            if response.finish_reason == "TOOL_CALL":
                finish_reason = "tool_calls"
            elif response.finish_reason == "MAX_TOKENS":
                finish_reason = "length"

            log_with_context(
                self.logger, "info", "Cohere request completed",
                event="llm_request_complete", provider="cohere", model=self.model,
                input_tokens=input_tokens, output_tokens=output_tokens,
                cost_usd=round(cost_usd, 6), latency_ms=round(latency_ms, 2),
                finish_reason=finish_reason, status="success",
            )

            return Response(
                content=text_content,
                model=self.model,
                input_tokens=int(input_tokens),
                output_tokens=int(output_tokens),
                cost_usd=round(cost_usd, 6),
                latency_ms=round(latency_ms, 2),
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            log_with_context(
                self.logger, "error", f"Cohere request failed: {e}",
                event="llm_request_error", provider="cohere", model=self.model,
                error=str(e), error_type=type(e).__name__,
                latency_ms=round(latency_ms, 2), status="error",
            )
            raise

    async def complete_stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 500,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[ResponseFormat] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        start_time = time.time()

        log_with_context(
            self.logger, "info", "Starting Cohere streaming request",
            event="llm_stream_start", provider="cohere", model=self.model,
        )

        try:
            preamble, cohere_messages = self._convert_messages(messages)
            effective_preamble = self._build_preamble(preamble, response_format)

            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": cohere_messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if effective_preamble:
                kwargs["preamble"] = effective_preamble
            if tools:
                kwargs["tools"] = self._convert_tools(tools)

            input_tokens = None
            output_tokens = None

            async with self.client.chat_stream(**kwargs) as stream:
                async for event in stream:
                    event_type = type(event).__name__

                    if event_type == "StreamedChatResponseV2" and hasattr(event, "delta"):
                        delta = event.delta
                        if hasattr(delta, "message") and delta.message:
                            msg = delta.message
                            if hasattr(msg, "content") and msg.content:
                                for block in msg.content:
                                    if hasattr(block, "text") and block.text:
                                        yield StreamChunk(content=block.text)

                    # Capture usage from finish event
                    if hasattr(event, "type") and event.type == "message-end":
                        if hasattr(event, "delta") and hasattr(event.delta, "usage"):
                            usage = event.delta.usage
                            if hasattr(usage, "billed_units"):
                                input_tokens = int(usage.billed_units.input_tokens or 0)
                                output_tokens = int(usage.billed_units.output_tokens or 0)

            latency_ms = (time.time() - start_time) * 1000

            log_with_context(
                self.logger, "info", "Cohere streaming completed",
                event="llm_stream_complete", provider="cohere", model=self.model,
                input_tokens=input_tokens, output_tokens=output_tokens,
                latency_ms=round(latency_ms, 2), status="success",
            )

            yield StreamChunk(
                finish_reason="stop",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            log_with_context(
                self.logger, "error", f"Cohere streaming failed: {e}",
                event="llm_stream_error", provider="cohere", model=self.model,
                error=str(e), error_type=type(e).__name__,
                latency_ms=round(latency_ms, 2), status="error",
            )
            raise
