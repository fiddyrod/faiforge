import time
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


class OllamaAdapter(LLMAdapter):
    """
    Adapter for Ollama local model serving.

    Ollama exposes an OpenAI-compatible API at http://localhost:11434/v1,
    so this adapter reuses the OpenAI client with a custom base URL.

    Setup:
        1. Install Ollama: https://ollama.ai
        2. Pull a model: ollama pull llama3
        3. Ollama runs automatically at localhost:11434

    Popular models:
        - llama3          (8B, good quality)
        - llama3:70b      (70B, best quality, needs high VRAM)
        - mistral         (7B, fast and capable)
        - phi3            (3.8B, great for low VRAM)
        - phi3:mini       (3.8B mini, very fast)
        - tinyllama       (1.1B, ultra fast, low quality)
        - codellama       (7B, code-focused)
        - nomic-embed-text (embedding model)

    Example:
        adapter = OllamaAdapter(model="llama3")
        response = await adapter.complete([
            Message(role="user", content="Hello!")
        ])
    """

    DEFAULT_BASE_URL = "http://localhost:11434/v1"

    def __init__(
        self,
        model: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,  # Longer timeout for local models
    ):
        """
        Initialize Ollama adapter.

        Args:
            model: Ollama model name (e.g., 'llama3', 'mistral', 'phi3')
            base_url: Ollama API base URL (default: http://localhost:11434/v1)
            timeout: Request timeout in seconds (default: 120 - local models are slower)
        """
        self.model = model
        self.base_url = base_url
        self.logger = get_logger()

        # Ollama's OpenAI-compatible API doesn't need a real API key
        self.client = AsyncOpenAI(
            api_key="ollama",
            base_url=base_url,
            timeout=timeout
        )

        log_with_context(
            self.logger,
            "info",
            "Ollama adapter initialized",
            event="ollama_init",
            model=model,
            base_url=base_url
        )

    def _convert_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert FAIForge messages to OpenAI-compatible format."""
        openai_messages = []

        for msg in messages:
            openai_msg: Dict[str, Any] = {"role": msg.role}

            if msg.content is not None:
                openai_msg["content"] = msg.content

            if msg.role == "tool" and msg.tool_call_id:
                openai_msg["tool_call_id"] = msg.tool_call_id

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
        """Convert FAIForge tools to OpenAI-compatible format."""
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

    def _parse_tool_calls(self, tool_calls) -> Optional[List[ToolCall]]:
        """Parse tool calls from response."""
        if not tool_calls:
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
            for tc in tool_calls
        ]

    async def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 500,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[ResponseFormat] = None
    ) -> Response:
        """Generate completion using Ollama."""

        start_time = time.time()

        log_with_context(
            self.logger,
            "info",
            "Starting Ollama request",
            event="llm_request_start",
            provider="ollama",
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            message_count=len(messages),
            has_tools=tools is not None
        )

        try:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": self._convert_messages(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            if tools:
                kwargs["tools"] = self._convert_tools(tools)
                if tool_choice:
                    kwargs["tool_choice"] = tool_choice

            # Ollama supports json mode via response_format
            if response_format and response_format.type == "json_object":
                kwargs["response_format"] = {"type": "json_object"}

            response = await self.client.chat.completions.create(**kwargs)

            latency_ms = (time.time() - start_time) * 1000
            choice = response.choices[0]
            message = choice.message
            usage = response.usage

            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0

            tool_calls = self._parse_tool_calls(message.tool_calls)
            finish_reason = choice.finish_reason or "stop"
            if tool_calls:
                finish_reason = "tool_calls"

            log_with_context(
                self.logger,
                "info",
                "Ollama request completed",
                event="llm_request_complete",
                provider="ollama",
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=0.0,  # Local inference is free
                latency_ms=round(latency_ms, 2),
                finish_reason=finish_reason,
                status="success"
            )

            return Response(
                content=message.content,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=0.0,
                latency_ms=round(latency_ms, 2),
                tool_calls=tool_calls,
                finish_reason=finish_reason
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            log_with_context(
                self.logger,
                "error",
                f"Ollama request failed: {str(e)}",
                event="llm_request_error",
                provider="ollama",
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
        """Generate streaming completion using Ollama."""

        start_time = time.time()

        log_with_context(
            self.logger,
            "info",
            "Starting Ollama streaming request",
            event="llm_stream_start",
            provider="ollama",
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            message_count=len(messages)
        )

        try:
            kwargs: Dict[str, Any] = {
                "model": self.model,
                "messages": self._convert_messages(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }

            if tools:
                kwargs["tools"] = self._convert_tools(tools)
                if tool_choice:
                    kwargs["tool_choice"] = tool_choice

            if response_format and response_format.type == "json_object":
                kwargs["response_format"] = {"type": "json_object"}

            stream = await self.client.chat.completions.create(**kwargs)

            accumulated_tool_calls: Dict[int, Dict[str, Any]] = {}
            finish_reason = None
            input_tokens = 0
            output_tokens = 0

            async for chunk in stream:
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens

                if chunk.choices:
                    choice = chunk.choices[0]
                    delta = choice.delta

                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

                    content = delta.content if hasattr(delta, 'content') else None

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

                    if content:
                        yield StreamChunk(content=content)

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

            latency_ms = (time.time() - start_time) * 1000

            log_with_context(
                self.logger,
                "info",
                "Ollama streaming completed",
                event="llm_stream_complete",
                provider="ollama",
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=round(latency_ms, 2),
                finish_reason=finish_reason,
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
                f"Ollama streaming failed: {str(e)}",
                event="llm_stream_error",
                provider="ollama",
                model=self.model,
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round(latency_ms, 2),
                status="error"
            )
            raise

    @staticmethod
    async def list_models(base_url: str = DEFAULT_BASE_URL) -> List[str]:
        """
        List available Ollama models.

        Args:
            base_url: Ollama API base URL

        Returns:
            List of available model names
        """
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                # Ollama's native API for listing models
                ollama_url = base_url.replace("/v1", "")
                response = await client.get(f"{ollama_url}/api/tags", timeout=10.0)
                response.raise_for_status()
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []
