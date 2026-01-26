import time
import uuid
from typing import List, Dict, Any, Optional

from .base import ChunkingAdapter, Chunk, ChunkingResult, ChunkingStrategy
from ...observability import get_logger, log_with_context


class FixedSizeChunker(ChunkingAdapter):
    """Simple fixed-size chunking with sliding window and overlap"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100
    ):
        """
        Initialize fixed-size chunker.

        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = get_logger()

        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )

    @property
    def strategy(self) -> ChunkingStrategy:
        return ChunkingStrategy.FIXED_SIZE

    async def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChunkingResult:
        """Chunk text using fixed-size sliding window"""

        start_time = time.time()

        log_with_context(
            self.logger,
            "info",
            "Starting fixed-size chunking",
            event="chunking_start",
            strategy="fixed_size",
            text_length=len(text),
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        chunks = []
        start = 0
        metadata = metadata or {}

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_content = text[start:end]

            chunk = Chunk(
                content=chunk_content,
                chunk_id=str(uuid.uuid4()),
                metadata=metadata.copy(),
                start_index=start,
                end_index=end
            )
            chunks.append(chunk)

            # Move start forward (chunk_size - overlap)
            start += self.chunk_size - self.chunk_overlap

        latency_ms = (time.time() - start_time) * 1000

        log_with_context(
            self.logger,
            "info",
            "Fixed-size chunking completed",
            event="chunking_complete",
            strategy="fixed_size",
            total_chunks=len(chunks),
            latency_ms=round(latency_ms, 2)
        )

        return ChunkingResult(
            chunks=chunks,
            total_chunks=len(chunks),
            strategy="fixed_size",
            latency_ms=round(latency_ms, 2)
        )

    async def chunk_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> ChunkingResult:
        """Chunk multiple documents"""

        start_time = time.time()
        all_chunks = []

        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            result = await self.chunk_text(content, metadata)
            all_chunks.extend(result.chunks)

        latency_ms = (time.time() - start_time) * 1000

        return ChunkingResult(
            chunks=all_chunks,
            total_chunks=len(all_chunks),
            strategy="fixed_size",
            latency_ms=round(latency_ms, 2)
        )
