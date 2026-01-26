from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class ChunkingStrategy(str, Enum):
    """Chunking strategy types"""
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    FIXED_SIZE = "fixed_size"
    TOKEN = "token"


@dataclass
class Chunk:
    """A single text chunk with metadata"""
    content: str                                    # Chunk text content
    chunk_id: str                                   # Unique chunk identifier
    metadata: Dict[str, Any] = field(default_factory=dict)  # Arbitrary metadata
    start_index: Optional[int] = None               # Start position in original text
    end_index: Optional[int] = None                 # End position in original text
    token_count: Optional[int] = None               # Token count (if available)


@dataclass
class ChunkingResult:
    """Result from chunking operation"""
    chunks: List[Chunk]           # List of chunks
    total_chunks: int             # Total number of chunks
    strategy: str                 # Strategy used
    latency_ms: float             # Processing time


class ChunkingAdapter(ABC):
    """Base class for all chunking adapters"""

    @abstractmethod
    async def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChunkingResult:
        """
        Split text into chunks.

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach to all chunks

        Returns:
            ChunkingResult with chunks and metadata
        """
        pass

    @abstractmethod
    async def chunk_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> ChunkingResult:
        """
        Chunk multiple documents.

        Args:
            documents: List of dicts with 'content' and optional 'metadata'

        Returns:
            ChunkingResult with all chunks
        """
        pass

    @property
    @abstractmethod
    def strategy(self) -> ChunkingStrategy:
        """Return the chunking strategy type"""
        pass
