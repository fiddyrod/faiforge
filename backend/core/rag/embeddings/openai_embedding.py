import time
from typing import List
from openai import AsyncOpenAI

from .base import EmbeddingAdapter, EmbeddingResult
from ...observability import get_logger, log_with_context


class OpenAIEmbedding(EmbeddingAdapter):
    """Adapter for OpenAI Embeddings API"""

    # Model configurations with dimensions and pricing per 1M tokens (January 2025)
    MODEL_CONFIG = {
        "text-embedding-3-small": {
            "dimensions": 1536,
            "pricing": 0.020 / 1_000_000  # $0.020 per 1M tokens
        },
        "text-embedding-3-large": {
            "dimensions": 3072,
            "pricing": 0.130 / 1_000_000  # $0.130 per 1M tokens
        },
        "text-embedding-ada-002": {
            "dimensions": 1536,
            "pricing": 0.100 / 1_000_000  # $0.100 per 1M tokens
        }
    }

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        timeout: float = 60.0
    ):
        """
        Initialize OpenAI embedding adapter.

        Args:
            api_key: OpenAI API key
            model: Embedding model name
            timeout: Request timeout in seconds
        """
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout)
        self._model_name = model
        self.logger = get_logger()

        if model not in self.MODEL_CONFIG:
            raise ValueError(
                f"Unknown model: {model}. Available: {list(self.MODEL_CONFIG.keys())}"
            )

        self._dimensions = self.MODEL_CONFIG[model]["dimensions"]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_documents(
        self,
        texts: List[str]
    ) -> EmbeddingResult:
        """Generate embeddings for multiple documents"""

        start_time = time.time()

        # Log request start
        log_with_context(
            self.logger,
            "info",
            "Starting OpenAI embedding request",
            event="embedding_request_start",
            provider="openai",
            model=self._model_name,
            document_count=len(texts)
        )

        try:
            # Call OpenAI embeddings API
            response = await self.client.embeddings.create(
                model=self._model_name,
                input=texts
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract embeddings
            embeddings = [item.embedding for item in response.data]
            total_tokens = response.usage.total_tokens

            # Log successful completion
            log_with_context(
                self.logger,
                "info",
                "OpenAI embedding request completed",
                event="embedding_request_complete",
                provider="openai",
                model=self._model_name,
                document_count=len(texts),
                total_tokens=total_tokens,
                latency_ms=round(latency_ms, 2),
                status="success"
            )

            return EmbeddingResult(
                embeddings=embeddings,
                dimensions=self._dimensions,
                model=self._model_name,
                total_tokens=total_tokens,
                latency_ms=round(latency_ms, 2)
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            # Log error
            log_with_context(
                self.logger,
                "error",
                f"OpenAI embedding request failed: {str(e)}",
                event="embedding_request_error",
                provider="openai",
                model=self._model_name,
                error=str(e),
                error_type=type(e).__name__,
                latency_ms=round(latency_ms, 2),
                status="error"
            )

            raise e

    async def embed_query(
        self,
        text: str
    ) -> EmbeddingResult:
        """Generate embedding for a single query"""
        return await self.embed_documents([text])
