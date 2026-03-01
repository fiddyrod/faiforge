from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class RerankResult:
    id: str
    content: str
    score: float          # cross-encoder score (higher = more relevant)
    original_score: float # score from initial retrieval
    metadata: dict


class RerankerAdapter(ABC):
    """Base class for reranking retrieved chunks."""

    @abstractmethod
    async def rerank(self, query: str, results: list, top_k: int) -> List[RerankResult]:
        """Rerank a list of search results for a given query.

        Args:
            query: The user query.
            results: List of SearchResult or similar objects with .id, .content, .score, .metadata.
            top_k: Number of results to return after reranking.

        Returns:
            List of RerankResult sorted by cross-encoder score descending.
        """
        ...
