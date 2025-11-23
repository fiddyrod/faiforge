import time
from typing import List
from vllm import LLM, SamplingParams

from .base import LLMAdapter, Message, Response
from ...observability import get_logger, log_with_context


class VLLMAdapter(LLMAdapter):
    """Adapter for vLLM local model serving"""
    
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
        
        # Log request start
        log_with_context(
            self.logger,
            "info",
            f"Starting vLLM request",
            event="llm_request_start",
            provider="vllm",
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            message_count=len(messages)
        )
        
        try:
            # Format messages into prompt
            prompt = self._format_messages(messages)
            
            # Configure sampling
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.95
            )
            
            # Generate
            outputs = self.llm.generate([prompt], sampling_params)
            output = outputs[0].outputs[0]
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            
            # Estimate token counts
            input_tokens = len(outputs[0].prompt_token_ids)
            output_tokens = len(output.token_ids)
            total_tokens = input_tokens + output_tokens
            
            # Log successful completion
            log_with_context(
                self.logger,
                "info",
                f"vLLM request completed",
                event="llm_request_complete",
                provider="vllm",
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                cost_usd=0.0,
                latency_ms=round(latency_ms, 2),
                status="success"
            )
            
            return Response(
                content=output.text,
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=0.0,
                latency_ms=round(latency_ms, 2)
            )
        
        except Exception as e:
            # Calculate latency even on error
            latency_ms = (time.time() - start_time) * 1000
            
            # Log error
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
            
            # Re-raise
            raise