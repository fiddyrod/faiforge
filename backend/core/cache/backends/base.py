"""
Base cache backend interface.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..models import CacheEntry


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Retrieve entry by key."""
        pass

    @abstractmethod
    async def set(self, key: str, entry: CacheEntry) -> None:
        """Store entry with key."""
        pass

    @abstractmethod
    async def get_all_entries(self) -> List[CacheEntry]:
        """Get all cache entries for similarity search."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete entry by key."""
        pass

    @abstractmethod
    async def clear(self) -> int:
        """Clear all entries. Returns count of deleted entries."""
        pass

    @abstractmethod
    async def size(self) -> int:
        """Get number of entries in cache."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        pass
