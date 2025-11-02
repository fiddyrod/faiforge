import time
from typing import List, Optional
from vllm import LLM, SamplingParams
from .base import LLMAdapter, Message, Response


class VLLMAdapter(LLMAdapter):
    """Adapter for vLLM local model serving"""
    
    def __init__(
        self,
        model: str,
        max_model_len: int = 2048,
        gpu_memory_utilization: float = 0.9
    ):
        """
        Initialize vLLM adapter.
        
        Args:
            model: HuggingFace model name
            max_model_len: Maximum context length
            gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0)
        """
        print(f"Loading vLLM model: {model}...")
        print("This may take a few minutes on first run (downloading weights)...")
        
        self.model_name = model
        self.llm = LLM(
            model=model,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True  # Required for some models
        )
        
        print(f"✅ Model {model} loaded successfully!")
    
    def _format_messages(self, messages: List[Message]) -> str:
        """
        Convert messages to a prompt string.
        Different models have different chat formats.
        This is a simple version - you'd customize per model.
        """
        prompt = ""
        for msg in messages:
            if msg.role == "system":
                prompt += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                prompt += f"User: {msg.content}\n\n"
            elif msg.role == "assistant":
                prompt += f"Assistant: {msg.content}\n\n"
        
        prompt += "Assistant: "
        return prompt
    
    async def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Response:
        """Generate completion using vLLM"""
        start_time = time.time()
        
        # Format messages into prompt
        prompt = self._format_messages(messages)
        
        # Configure sampling
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95
        )
        
        # Generate (vLLM is synchronous, but we're in async function)
        outputs = self.llm.generate([prompt], sampling_params)
        output = outputs[0].outputs[0]
        
        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000
        
        # Estimate token counts (vLLM provides token IDs)
        input_tokens = len(outputs[0].prompt_token_ids)
        output_tokens = len(output.token_ids)
        
        return Response(
            content=output.text,
            model=self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=0.0,  # Local models are free!
            latency_ms=round(latency_ms, 2)
        )