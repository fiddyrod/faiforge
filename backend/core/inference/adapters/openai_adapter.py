import time
from typing import List
from openai import AsyncOpenAI

from .base import LLMAdapter, Message, Response
from ...observability import get_logger, log_with_context

class OpenAIAdapter(LLMAdapter):
    """Adapter for OpenAI API"""
    
    # Pricing per 1M tokens (October 2024)
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
        
    
    async def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Response:
        """Generate completion using OpenAI API"""

        start_time = time.time()

        # Log request start
        log_with_context(
            self.logger,
            "info",
            f"Starting OpenAI request",
            event="llm_request_start",
            provider="openai",
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            message_count=len(messages)
        )
        
        try:
            # Convert messages to OpenAI format
            openai_messages = [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ]
            
            # Call OpenAI API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Calculate cost
            usage = response.usage
            pricing = self.PRICING[self.model]
            cost_usd = (
                usage.prompt_tokens * pricing["input"] +
                usage.completion_tokens * pricing["output"]
            )

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
                status="success"
            )

            return Response(
                content=response.choices[0].message.content,
                model=self.model,
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                cost_usd=round(cost_usd, 6),
                latency_ms=round(latency_ms, 2)
            )
        except Exception as e:
            # Log error
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
            
            raise e