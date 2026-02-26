"""
Base interfaces for search adapters.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class FusionMethod(str, Enum):
    """Score fusion methods for hybrid search."""
    WEIGHTED_SUM = "weighted_sum"  # Linear combination of normalized scores
    RRF = "rrf"  # Reciprocal Rank Fusion
    MAX = "max"  # Take maximum score
    MIN = "min"  # Take minimum score


@dataclass
class ScoredResult:
    """A search result with score and metadata."""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Source tracking for hybrid search
    semantic_score: Optional[float] = None
    bm25_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
            "semantic_score": self.semantic_score,
            "bm25_score": self.bm25_score,
        }


class SearchAdapter(ABC):
    """Base class for search adapters."""

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ScoredResult]:
        """
        Search for documents matching the query.

        Args:
            query: Search query string
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of scored results
        """
        pass

    @abstractmethod
    async def index_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Index documents for search.

        Args:
            documents: List of documents with 'id', 'content', and optional 'metadata'

        Returns:
            Indexing statistics
        """
        pass
