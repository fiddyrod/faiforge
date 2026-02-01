"""
In-memory cache backend for development and testing.
"""

import asyncio
from typing import Dict, List, Optional

from .base import CacheBackend
from ..models import CacheEntry


class MemoryCacheBackend(CacheBackend):
    """
    In-memory cache backend for development and testing.

    Features:
    - Fast access with O(1) key lookup
    - Automatic TTL expiration
    - LRU eviction when max_entries reached
    """

    def __init__(
        self,
        max_entries: int = 10000,
        ttl: int = 3600
    ):
        self.max_entries = max_entries
        self.default_ttl = ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: List[str] = []  # For LRU eviction
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve entry by key."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            # Check expiration
            if entry.is_expired:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return None

            # Update access order for LRU
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            return entry

    async def set(self, key: str, entry: CacheEntry) -> None:
        """Store entry with key."""
        async with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.max_entries and self._access_order:
                oldest_key = self._access_order.pop(0)
                self._cache.pop(oldest_key, None)

            self._cache[key] = entry
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

    async def get_all_entries(self) -> List[CacheEntry]:
        """Get all non-expired cache entries."""
        async with self._lock:
            entries = []
            expired_keys = []

            for key, entry in self._cache.items():
                if entry.is_expired:
                    expired_keys.append(key)
                else:
                    entries.append(entry)

            # Clean up expired entries
            for key in expired_keys:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)

            return entries

    async def delete(self, key: str) -> bool:
        """Delete entry by key."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return True
            return False

    async def clear(self) -> int:
        """Clear all entries."""
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            self._access_order.clear()
            return count

    async def size(self) -> int:
        """Get number of entries."""
        return len(self._cache)

    async def close(self) -> None:
        """No cleanup needed for in-memory backend."""
        pass
