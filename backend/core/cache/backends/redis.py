"""
Redis cache backend for production use.
"""

import json
import logging
from typing import List, Optional

from .base import CacheBackend
from ..models import CacheEntry

logger = logging.getLogger(__name__)


class RedisCacheBackend(CacheBackend):
    """
    Redis cache backend for production use.

    Features:
    - Persistent storage
    - Distributed access
    - Native TTL support
    - Scalable to millions of entries

    Requires: redis[hiredis] package
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        db: int = 0,
        ttl: int = 3600,
        namespace: str = "faiforge:cache"
    ):
        self.url = url
        self.db = db
        self.default_ttl = ttl
        self.namespace = namespace
        self._client = None
        self._initialized = False

    async def _ensure_connected(self) -> None:
        """Lazy initialization of Redis connection."""
        if self._initialized:
            return

        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError(
                "Redis backend requires 'redis' package. "
                "Install with: pip install redis[hiredis]"
            )

        self._client = redis.from_url(
            self.url,
            db=self.db,
            decode_responses=False  # We handle encoding ourselves
        )
        self._initialized = True
        logger.info(f"Connected to Redis at {self.url}")

    def _make_key(self, key: str) -> str:
        """Create namespaced Redis key."""
        return f"{self.namespace}:{key}"

    def _index_key(self) -> str:
        """Key for the set of all cache keys."""
        return f"{self.namespace}:_index"

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve entry by key."""
        await self._ensure_connected()

        redis_key = self._make_key(key)
        data = await self._client.get(redis_key)

        if data is None:
            return None

        try:
            entry_dict = json.loads(data.decode('utf-8'))
            return CacheEntry.from_dict(entry_dict)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to deserialize cache entry: {e}")
            await self._client.delete(redis_key)
            return None

    async def set(self, key: str, entry: CacheEntry) -> None:
        """Store entry with key."""
        await self._ensure_connected()

        redis_key = self._make_key(key)
        data = json.dumps(entry.to_dict()).encode('utf-8')

        ttl = entry.ttl or self.default_ttl

        # Use pipeline for atomic operation
        async with self._client.pipeline() as pipe:
            pipe.set(redis_key, data, ex=ttl)
            pipe.sadd(self._index_key(), key)
            await pipe.execute()

    async def get_all_entries(self) -> List[CacheEntry]:
        """Get all cache entries for similarity search."""
        await self._ensure_connected()

        # Get all keys from index
        keys = await self._client.smembers(self._index_key())
        if not keys:
            return []

        # Fetch all entries
        entries = []
        expired_keys = []

        redis_keys = [self._make_key(k.decode('utf-8') if isinstance(k, bytes) else k) for k in keys]
        values = await self._client.mget(redis_keys)

        for key, value in zip(keys, values):
            key_str = key.decode('utf-8') if isinstance(key, bytes) else key
            if value is None:
                expired_keys.append(key_str)
                continue

            try:
                entry_dict = json.loads(value.decode('utf-8'))
                entries.append(CacheEntry.from_dict(entry_dict))
            except (json.JSONDecodeError, KeyError):
                expired_keys.append(key_str)

        # Clean up expired keys from index
        if expired_keys:
            await self._client.srem(self._index_key(), *expired_keys)

        return entries

    async def delete(self, key: str) -> bool:
        """Delete entry by key."""
        await self._ensure_connected()

        redis_key = self._make_key(key)

        async with self._client.pipeline() as pipe:
            pipe.delete(redis_key)
            pipe.srem(self._index_key(), key)
            results = await pipe.execute()

        return results[0] > 0

    async def clear(self) -> int:
        """Clear all entries in namespace."""
        await self._ensure_connected()

        # Get all keys
        keys = await self._client.smembers(self._index_key())
        if not keys:
            return 0

        count = len(keys)
        redis_keys = [self._make_key(k.decode('utf-8') if isinstance(k, bytes) else k) for k in keys]

        # Delete all keys and index
        async with self._client.pipeline() as pipe:
            pipe.delete(*redis_keys)
            pipe.delete(self._index_key())
            await pipe.execute()

        return count

    async def size(self) -> int:
        """Get number of entries."""
        await self._ensure_connected()
        return await self._client.scard(self._index_key())

    async def close(self) -> None:
        """Close Redis connection."""
        if self._client:
            await self._client.close()
            self._initialized = False
