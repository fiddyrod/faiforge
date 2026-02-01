"""
Cache data models and configuration.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CacheEntry:
    """A cached query-response pair with embedding."""
    query: str
    response: Dict[str, Any]
    embedding: List[float]
    created_at: float
    ttl: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired based on TTL."""
        if self.ttl is None:
            return False
        return time.time() > (self.created_at + self.ttl)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "query": self.query,
            "response": self.response,
            "embedding": self.embedding,
            "created_at": self.created_at,
            "ttl": self.ttl,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheEntry":
        """Deserialize from dictionary."""
        return cls(
            query=data["query"],
            response=data["response"],
            embedding=data["embedding"],
            created_at=data["created_at"],
            ttl=data.get("ttl"),
            metadata=data.get("metadata", {})
        )


@dataclass
class CacheStats:
    """Statistics for cache performance monitoring."""
    hits: int = 0
    misses: int = 0
    total_queries: int = 0
    avg_similarity_on_hit: float = 0.0
    cache_size: int = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.total_queries == 0:
            return 0.0
        return self.hits / self.total_queries

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_queries": self.total_queries,
            "hit_rate": round(self.hit_rate, 4),
            "avg_similarity_on_hit": round(self.avg_similarity_on_hit, 4),
            "cache_size": self.cache_size
        }


@dataclass
class CacheConfig:
    """Configuration for semantic cache."""
    enabled: bool = True
    backend: str = "memory"  # "memory" or "redis"
    similarity_threshold: float = 0.95
    max_entries: int = 10000
    ttl: int = 3600  # seconds
    namespace: str = "faiforge:cache"

    # Redis-specific config
    redis_url: str = "redis://localhost:6379"
    redis_db: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheConfig":
        """Create config from dictionary."""
        return cls(
            enabled=data.get("enabled", True),
            backend=data.get("backend", "memory"),
            similarity_threshold=data.get("similarity_threshold", 0.95),
            max_entries=data.get("max_entries", 10000),
            ttl=data.get("ttl", 3600),
            namespace=data.get("namespace", "faiforge:cache"),
            redis_url=data.get("redis", {}).get("url", "redis://localhost:6379"),
            redis_db=data.get("redis", {}).get("db", 0)
        )
