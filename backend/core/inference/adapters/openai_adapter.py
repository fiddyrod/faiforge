import time
from typing import List
from openai import AsyncOpenAI
from .base import LLMAdapter, Message, Response


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
    
    def __init__(self, api_key: str, model: str):
        """
        Initialize OpenAI adapter.
        
        Args:
            api_key: OpenAI API key
            model: Model name (e.g., 'gpt-4o-mini')
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        
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
        
        return Response(
            content=response.choices[0].message.content,
            model=self.model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost_usd=round(cost_usd, 6),
            latency_ms=round(latency_ms, 2)
        )