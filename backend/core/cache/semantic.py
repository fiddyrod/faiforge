"""
Semantic caching layer for LLM responses.

Uses embedding similarity to find cached responses for semantically
similar queries, reducing API costs and latency.
"""

import hashlib
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from .backends.base import CacheBackend
from .models import CacheConfig, CacheEntry, CacheStats

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Semantic caching layer for LLM responses.

    Uses embedding similarity to find cached responses for semantically
    similar queries, reducing API costs and latency.

    Example:
        cache = SemanticCache(
            backend=MemoryCacheBackend(),
            embedding_fn=my_embedding_function,
            similarity_threshold=0.95
        )

        # Check cache
        result = await cache.get("What is Python?")
        if result:
            response, similarity = result
            print(f"Cache hit with {similarity:.2%} similarity")

        # Store in cache
        await cache.set("What is Python?", {"content": "Python is..."})
    """

    def __init__(
        self,
        backend: CacheBackend,
        embedding_fn: Callable[[str], Awaitable[List[float]]],
        similarity_threshold: float = 0.95,
        config: Optional[CacheConfig] = None
    ):
        """
        Initialize semantic cache.

        Args:
            backend: Cache storage backend (Memory or Redis)
            embedding_fn: Async function to compute embeddings
            similarity_threshold: Minimum cosine similarity for cache hit (0-1)
            config: Optional cache configuration
        """
        self.backend = backend
        self.embedding_fn = embedding_fn
        self.similarity_threshold = similarity_threshold
        self.config = config or CacheConfig()
        self._stats = CacheStats()

    @staticmethod
    def _compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        if NUMPY_AVAILABLE:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(np.dot(vec1, vec2) / (norm1 * norm2))
        else:
            # Pure Python fallback
            dot = sum(a * b for a, b in zip(embedding1, embedding2))
            norm1 = sum(a * a for a in embedding1) ** 0.5
            norm2 = sum(b * b for b in embedding2) ** 0.5

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot / (norm1 * norm2)

    @staticmethod
    def _generate_key(query: str) -> str:
        """Generate cache key from query."""
        return hashlib.sha256(query.encode('utf-8')).hexdigest()[:16]

    async def get(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Optional[Tuple[Dict[str, Any], float]]:
        """
        Find cached response for semantically similar query.

        Args:
            query: The query string to look up
            metadata_filter: Optional filter to match against entry metadata

        Returns:
            Tuple of (cached_response, similarity_score) or None if no match
        """
        self._stats.total_queries += 1

        # Compute query embedding
        query_embedding = await self.embedding_fn(query)

        # Get all entries and find best match
        entries = await self.backend.get_all_entries()

        best_match: Optional[CacheEntry] = None
        best_similarity: float = 0.0

        for entry in entries:
            # Apply metadata filter if provided
            if metadata_filter:
                if not all(entry.metadata.get(k) == v for k, v in metadata_filter.items()):
                    continue

            similarity = self._compute_similarity(query_embedding, entry.embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry

        # Check if best match meets threshold
        if best_match and best_similarity >= self.similarity_threshold:
            self._stats.hits += 1
            # Update running average
            n = self._stats.hits
            self._stats.avg_similarity_on_hit = (
                (self._stats.avg_similarity_on_hit * (n - 1) + best_similarity) / n
            )

            logger.debug(
                f"Cache hit: similarity={best_similarity:.4f}, "
                f"query='{query[:50]}...'"
            )
            return best_match.response, best_similarity

        self._stats.misses += 1
        return None

    async def set(
        self,
        query: str,
        response: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[int] = None
    ) -> str:
        """
        Cache a query-response pair.

        Args:
            query: The query string
            response: The response to cache
            metadata: Optional metadata for filtering
            ttl: Optional TTL override (seconds)

        Returns:
            The cache key
        """
        # Compute embedding
        embedding = await self.embedding_fn(query)

        # Create entry
        entry = CacheEntry(
            query=query,
            response=response,
            embedding=embedding,
            created_at=time.time(),
            ttl=ttl or self.config.ttl,
            metadata=metadata or {}
        )

        # Generate key and store
        key = self._generate_key(query)
        await self.backend.set(key, entry)

        logger.debug(f"Cached response: key={key}, query='{query[:50]}...'")
        return key

    async def invalidate(self, query: str) -> bool:
        """
        Invalidate cache entry for exact query match.

        Args:
            query: The query to invalidate

        Returns:
            True if entry was found and deleted
        """
        key = self._generate_key(query)
        return await self.backend.delete(key)

    async def clear(self) -> int:
        """
        Clear all cache entries.

        Returns:
            Number of entries cleared
        """
        count = await self.backend.clear()
        self._stats = CacheStats()  # Reset stats
        logger.info(f"Cache cleared: {count} entries removed")
        return count

    @property
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics as dictionary."""
        self._stats.cache_size = await self.backend.size()
        return self._stats.to_dict()

    async def close(self) -> None:
        """Clean up resources."""
        await self.backend.close()
