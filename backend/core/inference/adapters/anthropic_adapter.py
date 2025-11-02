import time
from typing import List
from anthropic import AsyncAnthropic
from .base import LLMAdapter, Message, Response


class AnthropicAdapter(LLMAdapter):
    """Adapter for Anthropic Claude API"""
    
    # Pricing per 1M tokens (November 2024)
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
    
    def __init__(self, api_key: str, model: str):
        """
        Initialize Anthropic adapter.
        
        Args:
            api_key: Anthropic API key
            model: Model name (e.g., 'claude-sonnet-4.5-20250929')
        """
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        
        if model not in self.PRICING:
            # Allow model but warn if pricing unknown
            print(f"Warning: Unknown model '{model}', cost calculation may be inaccurate")
    
    async def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Response:
        """Generate completion using Anthropic API"""
        start_time = time.time()
        
        # Anthropic requires system messages separate
        system_message = None
        anthropic_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        # Call Anthropic API
        kwargs = {
            "model": self.model,
            "messages": anthropic_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if system_message:
            kwargs["system"] = system_message
        
        response = await self.client.messages.create(**kwargs)
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Calculate cost
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        
        if self.model in self.PRICING:
            pricing = self.PRICING[self.model]
            cost_usd = (
                input_tokens * pricing["input"] +
                output_tokens * pricing["output"]
            )
        else:
            cost_usd = 0.0  # Unknown pricing
        
        return Response(
            content=response.content[0].text,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost_usd, 6),
            latency_ms=round(latency_ms, 2)
        )