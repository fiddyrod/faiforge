from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    """Result from embedding operation with metadata"""
    embeddings: List[List[float]]  # List of embedding vectors
    dimensions: int                 # Embedding dimension size
    model: str                      # Model name
    total_tokens: int               # Total tokens processed
    latency_ms: float               # Processing time in milliseconds


class EmbeddingAdapter(ABC):
    """Base class for all embedding adapters"""

    @abstractmethod
    async def embed_documents(
        self,
        texts: List[str]
    ) -> EmbeddingResult:
        """
        Generate embeddings for multiple documents/chunks.

        Args:
            texts: List of text strings to embed

        Returns:
            EmbeddingResult with embeddings and metadata
        """
        pass

    @abstractmethod
    async def embed_query(
        self,
        text: str
    ) -> EmbeddingResult:
        """
        Generate embedding for a single query string.

        Args:
            text: Query text to embed

        Returns:
            EmbeddingResult with single embedding and metadata
        """
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimension size"""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier"""
        pass
