"""
Cache Module - Semantic caching for LLM responses.

Reduces API costs and latency by caching semantically similar queries.
Supports Redis (production) and In-Memory (development) backends.

Example usage:
    from core.cache import SemanticCache, MemoryCacheBackend, CacheConfig

    # Create backend
    backend = MemoryCacheBackend(max_entries=10000, ttl=3600)

    # Create cache with your embedding function
    cache = SemanticCache(
        backend=backend,
        embedding_fn=my_embedding_function,
        similarity_threshold=0.95
    )

    # Check cache before calling LLM
    result = await cache.get(query)
    if result:
        response, similarity = result
        return response

    # On cache miss, call LLM and store result
    response = await llm.complete(messages)
    await cache.set(query, response)
"""

from .models import (
    CacheEntry,
    CacheStats,
    CacheConfig,
)
from .backends import (
    CacheBackend,
    MemoryCacheBackend,
    RedisCacheBackend,
)
from .semantic import SemanticCache


def create_cache_backend(config: CacheConfig) -> CacheBackend:
    """
    Create cache backend from configuration.

    Args:
        config: Cache configuration

    Returns:
        Configured cache backend
    """
    if config.backend == "redis":
        return RedisCacheBackend(
            url=config.redis_url,
            db=config.redis_db,
            ttl=config.ttl,
            namespace=config.namespace
        )
    else:
        return MemoryCacheBackend(
            max_entries=config.max_entries,
            ttl=config.ttl
        )


async def create_semantic_cache(
    config: CacheConfig,
    embedding_fn
) -> SemanticCache:
    """
    Create semantic cache from configuration.

    Args:
        config: Cache configuration
        embedding_fn: Async function to compute embeddings

    Returns:
        Configured semantic cache
    """
    backend = create_cache_backend(config)

    return SemanticCache(
        backend=backend,
        embedding_fn=embedding_fn,
        similarity_threshold=config.similarity_threshold,
        config=config
    )


__all__ = [
    # Models
    "CacheEntry",
    "CacheStats",
    "CacheConfig",
    # Backends
    "CacheBackend",
    "MemoryCacheBackend",
    "RedisCacheBackend",
    # Semantic Cache
    "SemanticCache",
    # Factory functions
    "create_cache_backend",
    "create_semantic_cache",
]
