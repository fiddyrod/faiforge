import time
from typing import List, Optional, Dict, Any, AsyncGenerator
from vllm import LLM, SamplingParams

from .base import (
    LLMAdapter,
    Message,
    Response,
    StreamChunk,
    Tool,
    ResponseFormat,
)
from ...observability import get_logger, log_with_context


class VLLMAdapter(LLMAdapter):
    """
    Adapter for vLLM local model serving.

    Note: vLLM has limited support for function calling and structured outputs.
    These features will log warnings and fall back to standard completion.
    """

    def __init__(
        self,
        model: str,
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.5
    ):
        """Initialize vLLM adapter"""
        self.model = model
        self.logger = get_logger()

        self.logger.info(f"Loading vLLM model: {model}")
        self.logger.info(f"GPU memory utilization: {gpu_memory_utilization * 100}%")
        self.logger.info("This may take a few minutes on first run (downloading weights)")

        try:
            self.llm = LLM(
                model=model,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True
            )
            self.logger.info(f"Model {model} loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load {model}: {e}")
            raise

    def _format_messages(self, messages: List[Message]) -> str:
        """Convert messages to a prompt string"""
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n\n"
            elif msg.role == "tool":
                # Basic tool result formatting
                prompt += f"Tool Result: {msg.content}\n\n"

        prompt += "Assistant: "
        return prompt

    def _format_json_instruction(self, response_format: ResponseFormat) -> str:
        """Add JSON instruction to prompt if needed"""
        if response_format.type == "json_object":
            return "\n\nRespond with valid JSON only."
        elif response_format.type == "json_schema" and response_format.json_schema:
            import json
            schema_str = json.dumps(response_format.json_schema, indent=2)
            return f"\n\nRespond with valid JSON matching this schema:\n{schema_str}"
        return ""

    async def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 500,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[ResponseFormat] = None
    ) -> Response:
        """Generate completion using vLLM"""

        start_time = time.time()

        # Warn about unsupported features
        if tools:
            self.logger.warning("vLLM adapter does not support function calling. Tools will be ignored.")

        log_with_context(
            self.logger,
            "info",
            "Starting vLLM request",
            event="llm_request_start",
            provider="vllm",
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            message_count=len(messages),
            has_tools=tools is not None,
            has_response_format=response_format is not None
        )

        try:
            # Format messages into prompt
            prompt = self._format_messages(messages)

            # Add JSON instruction if needed
            if response_format and response_format.type != "text":
                prompt += self._format_json_instruction(response_format)

            # Configure sampling
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95
            )

            # Generate (vLLM's generate is synchronous, wrap it)
            outputs = self.llm.generate([prompt], sampling_params)
            output = outputs[0].outputs[0]

            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000

            # Get token counts
            input_tokens = len(outputs[0].prompt_token_ids)
            output_tokens = len(output.token_ids)

            # Determine finish reason
            finish_reason = "stop"
            if output.finish_reason == "length":
                finish_reason = "length"

            log_with_context(
                self.logger,
                "info",
                "vLLM request completed",
                event="llm_request_complete",
                provider="vllm",
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                cost_usd=0.0,
                latency_ms=round(latency_ms, 2),
                finish_reason=finish_reason,
                status="success"
            )

            return Response(
                content=output.text,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=0.0,
                latency_ms=round(latency_ms, 2),
                tool_calls=None,  # vLLM doesn't support tool calls
                finish_reason=finish_reason
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            log_with_context(
                self.logger,
                "error",
                f"vLLM request failed: {str(e)}",
                event="llm_request_error",
                provider="vllm",
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
        """
        Generate streaming completion using vLLM.

        Note: vLLM streaming requires the AsyncLLMEngine. This implementation
        falls back to returning the full response as a single chunk for simplicity.
        For true streaming, consider using vLLM's AsyncLLMEngine directly.
        """

        start_time = time.time()

        if tools:
            self.logger.warning("vLLM adapter does not support function calling. Tools will be ignored.")

        log_with_context(
            self.logger,
            "info",
            "Starting vLLM streaming request",
            event="llm_stream_start",
            provider="vllm",
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            message_count=len(messages)
        )

        try:
            # Format messages into prompt
            prompt = self._format_messages(messages)

            if response_format and response_format.type != "text":
                prompt += self._format_json_instruction(response_format)

            # Configure sampling
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95
            )

            # Generate (fall back to non-streaming for simplicity)
            # True streaming would require AsyncLLMEngine
            outputs = self.llm.generate([prompt], sampling_params)
            output = outputs[0].outputs[0]

            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            input_tokens = len(outputs[0].prompt_token_ids)
            output_tokens = len(output.token_ids)

            # Simulate streaming by yielding content in chunks
            content = output.text
            chunk_size = 10  # characters per chunk

            for i in range(0, len(content), chunk_size):
                chunk_content = content[i:i + chunk_size]
                yield StreamChunk(content=chunk_content)

            # Determine finish reason
            finish_reason = "stop"
            if output.finish_reason == "length":
                finish_reason = "length"

            log_with_context(
                self.logger,
                "info",
                "vLLM streaming completed",
                event="llm_stream_complete",
                provider="vllm",
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=round(latency_ms, 2),
                finish_reason=finish_reason,
                status="success"
            )

            # Yield final chunk with metadata
            yield StreamChunk(
                finish_reason=finish_reason,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            log_with_context(
                self.logger,
                "error",
                f"vLLM streaming failed: {str(e)}",
                event="llm_stream_error",
                provider="vllm",
                model=self.model,
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round(latency_ms, 2),
                status="error"
            )
            raise
