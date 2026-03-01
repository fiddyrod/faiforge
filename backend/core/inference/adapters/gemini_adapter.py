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
    import google.generativeai as genai
    from google.generativeai.types import GenerateContentResponse
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None
    GEMINI_AVAILABLE = False


class GeminiAdapter(LLMAdapter):
    """Adapter for Google Gemini API with streaming, function calling, and JSON mode.

    Supports:
        - gemini-2.0-flash  (fast, low cost)
        - gemini-1.5-pro    (large context)
        - gemini-1.5-flash  (balanced)

    Requires: pip install google-generativeai
    """

    # Pricing per 1M tokens (March 2025)
    PRICING: Dict[str, Dict[str, float]] = {
        "gemini-2.0-flash": {
            "input": 0.10 / 1_000_000,
            "output": 0.40 / 1_000_000,
        },
        "gemini-1.5-pro": {
            "input": 1.25 / 1_000_000,
            "output": 5.00 / 1_000_000,
        },
        "gemini-1.5-flash": {
            "input": 0.075 / 1_000_000,
            "output": 0.30 / 1_000_000,
        },
        "gemini-1.5-flash-8b": {
            "input": 0.0375 / 1_000_000,
            "output": 0.15 / 1_000_000,
        },
    }

    def __init__(self, api_key: str, model: str, timeout: float = 60.0):
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai not installed. Install with: pip install google-generativeai"
            )
        genai.configure(api_key=api_key)
        self.model = model
        self.timeout = timeout
        self.logger = get_logger()
        self._client = genai.GenerativeModel(model)

        if model not in self.PRICING:
            self.logger.warning(f"Unknown Gemini model '{model}', cost calculation may be inaccurate")

    def _convert_messages(
        self, messages: List[Message]
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """Convert FAIForge messages to Gemini format.

        Returns:
            (system_instruction, gemini_contents)
        """
        system_instruction = None
        contents = []

        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
                continue

            # Map roles: "user" → "user", "assistant" → "model"
            role = "model" if msg.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg.content or ""}]})

        return system_instruction, contents

    def _convert_tools(self, tools: List[Tool]) -> List[Dict[str, Any]]:
        """Convert FAIForge tools to Gemini function declarations."""
        declarations = []
        for tool in tools:
            if tool.function is None:
                continue
            declarations.append({
                "name": tool.function.name,
                "description": tool.function.description,
                "parameters": tool.function.parameters,
            })
        return [{"function_declarations": declarations}]

    def _parse_tool_calls(self, response) -> tuple[Optional[str], Optional[List[ToolCall]]]:
        """Extract text and tool calls from a Gemini response."""
        text_content = None
        tool_calls = []

        for part in response.parts:
            if hasattr(part, "text") and part.text:
                text_content = part.text
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                tool_calls.append(ToolCall(
                    id=f"call_{fc.name}",
                    type="function",
                    function=FunctionCallResult(
                        name=fc.name,
                        arguments=json.dumps(dict(fc.args)),
                    ),
                ))

        return text_content, tool_calls if tool_calls else None

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        pricing = self.PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        return input_tokens * pricing["input"] + output_tokens * pricing["output"]

    def _generation_config(
        self,
        temperature: float,
        max_tokens: int,
        response_format: Optional[ResponseFormat],
    ) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if response_format and response_format.type in ("json_object", "json_schema"):
            cfg["response_mime_type"] = "application/json"
        return cfg

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
            self.logger, "info", "Starting Gemini request",
            event="llm_request_start", provider="gemini", model=self.model,
            temperature=temperature, max_tokens=max_tokens,
            message_count=len(messages), has_tools=tools is not None,
        )

        try:
            system_instruction, contents = self._convert_messages(messages)
            gen_cfg = self._generation_config(temperature, max_tokens, response_format)

            # Rebuild client with system instruction if present
            client = genai.GenerativeModel(
                self.model,
                system_instruction=system_instruction,
                generation_config=gen_cfg,
                tools=self._convert_tools(tools) if tools else None,
            )

            response = await client.generate_content_async(contents)
            latency_ms = (time.time() - start_time) * 1000

            text_content, tool_calls = self._parse_tool_calls(response.candidates[0].content)

            input_tokens = response.usage_metadata.prompt_token_count or 0
            output_tokens = response.usage_metadata.candidates_token_count or 0
            cost_usd = self._calculate_cost(input_tokens, output_tokens)

            finish_reason = "stop"
            if response.candidates[0].finish_reason.name == "STOP":
                finish_reason = "stop"
            elif response.candidates[0].finish_reason.name == "MAX_TOKENS":
                finish_reason = "length"

            log_with_context(
                self.logger, "info", "Gemini request completed",
                event="llm_request_complete", provider="gemini", model=self.model,
                input_tokens=input_tokens, output_tokens=output_tokens,
                cost_usd=round(cost_usd, 6), latency_ms=round(latency_ms, 2),
                finish_reason=finish_reason, status="success",
            )

            return Response(
                content=text_content,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=round(cost_usd, 6),
                latency_ms=round(latency_ms, 2),
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            log_with_context(
                self.logger, "error", f"Gemini request failed: {e}",
                event="llm_request_error", provider="gemini", model=self.model,
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
            self.logger, "info", "Starting Gemini streaming request",
            event="llm_stream_start", provider="gemini", model=self.model,
        )

        try:
            system_instruction, contents = self._convert_messages(messages)
            gen_cfg = self._generation_config(temperature, max_tokens, response_format)

            client = genai.GenerativeModel(
                self.model,
                system_instruction=system_instruction,
                generation_config=gen_cfg,
                tools=self._convert_tools(tools) if tools else None,
            )

            response = await client.generate_content_async(contents, stream=True)

            input_tokens = None
            output_tokens = None

            async for chunk in response:
                if not chunk.candidates:
                    continue
                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        yield StreamChunk(content=part.text)

                # Capture usage from final chunk
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    if chunk.usage_metadata.prompt_token_count:
                        input_tokens = chunk.usage_metadata.prompt_token_count
                    if chunk.usage_metadata.candidates_token_count:
                        output_tokens = chunk.usage_metadata.candidates_token_count

            latency_ms = (time.time() - start_time) * 1000

            log_with_context(
                self.logger, "info", "Gemini streaming completed",
                event="llm_stream_complete", provider="gemini", model=self.model,
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
                self.logger, "error", f"Gemini streaming failed: {e}",
                event="llm_stream_error", provider="gemini", model=self.model,
                error=str(e), error_type=type(e).__name__,
                latency_ms=round(latency_ms, 2), status="error",
            )
            raise
