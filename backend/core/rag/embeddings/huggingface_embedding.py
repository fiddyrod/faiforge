import time
import asyncio
from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer

from .base import EmbeddingAdapter, EmbeddingResult
from ...observability import get_logger, log_with_context


class HuggingFaceEmbedding(EmbeddingAdapter):
    """Adapter for HuggingFace sentence-transformers embeddings"""

    # Model configurations
    MODEL_CONFIG = {
        "all-MiniLM-L6-v2": {
            "dimensions": 384,
            "max_seq_length": 256
        },
        "all-mpnet-base-v2": {
            "dimensions": 768,
            "max_seq_length": 384
        }
    }

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize HuggingFace embedding adapter.

        Args:
            model: Model name from sentence-transformers
            device: Device to use ('cpu', 'cuda', or None for auto-detection)
        """
        self._model_name = model
        self.logger = get_logger()

        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Log model loading
        self.logger.info(f"Loading HuggingFace model: {model} on {device}")

        try:
            self.model = SentenceTransformer(model, device=device)
            self._dimensions = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"Model {model} loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load {model}: {e}")
            raise

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
            "Starting HuggingFace embedding request",
            event="embedding_request_start",
            provider="huggingface",
            model=self._model_name,
            device=self.device,
            document_count=len(texts)
        )

        try:
            # Run encoding in thread pool since sentence-transformers is sync
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self.model.encode(
                    texts,
                    convert_to_numpy=False,
                    show_progress_bar=False
                )
            )

            latency_ms = (time.time() - start_time) * 1000

            # Convert to list of lists
            embeddings_list = [emb.tolist() for emb in embeddings]

            # Estimate tokens (rough approximation: words * 1.3)
            estimated_tokens = sum(len(text.split()) * 1.3 for text in texts)

            # Log successful completion
            log_with_context(
                self.logger,
                "info",
                "HuggingFace embedding request completed",
                event="embedding_request_complete",
                provider="huggingface",
                model=self._model_name,
                device=self.device,
                document_count=len(texts),
                estimated_tokens=int(estimated_tokens),
                latency_ms=round(latency_ms, 2),
                status="success"
            )

            return EmbeddingResult(
                embeddings=embeddings_list,
                dimensions=self._dimensions,
                model=self._model_name,
                total_tokens=int(estimated_tokens),
                latency_ms=round(latency_ms, 2)
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            # Log error
            log_with_context(
                self.logger,
                "error",
                f"HuggingFace embedding request failed: {str(e)}",
                event="embedding_request_error",
                provider="huggingface",
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
