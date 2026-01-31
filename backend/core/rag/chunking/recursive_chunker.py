import time
import uuid
import re
from typing import List, Dict, Any, Optional

from .base import ChunkingAdapter, Chunk, ChunkingResult, ChunkingStrategy
from ...observability import get_logger, log_with_context


class RecursiveChunker(ChunkingAdapter):
    """Recursive text chunking with hierarchical separators"""

    DEFAULT_SEPARATORS = [
        "\n\n\n",  # Multiple newlines (paragraphs)
        "\n\n",    # Double newlines
        "\n",      # Single newlines
        ". ",      # Sentences
        "! ",      # Exclamations
        "? ",      # Questions
        " ",       # Words
        ""         # Characters (last resort)
    ]

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True
    ):
        """
        Initialize recursive chunker.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            separators: Hierarchical list of separators (most important first)
            keep_separator: Whether to keep separators in chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.keep_separator = keep_separator
        self.logger = get_logger()

    @property
    def strategy(self) -> ChunkingStrategy:
        return ChunkingStrategy.RECURSIVE

    async def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChunkingResult:
        """Chunk text using recursive splitting"""

        start_time = time.time()

        log_with_context(
            self.logger,
            "info",
            "Starting recursive chunking",
            event="chunking_start",
            strategy="recursive",
            text_length=len(text),
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        chunks = self._split_text_recursive(text, metadata or {})
        latency_ms = (time.time() - start_time) * 1000

        log_with_context(
            self.logger,
            "info",
            "Recursive chunking completed",
            event="chunking_complete",
            strategy="recursive",
            total_chunks=len(chunks),
            latency_ms=round(latency_ms, 2)
        )

        return ChunkingResult(
            chunks=chunks,
            total_chunks=len(chunks),
            strategy="recursive",
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
            strategy="recursive",
            latency_ms=round(latency_ms, 2)
        )

    def _split_text_recursive(
        self,
        text: str,
        metadata: Dict[str, Any],
        separator_index: int = 0
    ) -> List[Chunk]:
        """Recursively split text using separators"""

        # If text is small enough, return as single chunk
        if len(text) <= self.chunk_size:
            return [Chunk(
                content=text,
                chunk_id=str(uuid.uuid4()),
                metadata=metadata.copy(),
                start_index=0,
                end_index=len(text)
            )]

        # If we've exhausted all separators, use character-level split
        if separator_index >= len(self.separators):
            return self._create_fixed_chunks(text, metadata)

        separator = self.separators[separator_index]

        # Empty separator means character-level split
        if separator == "":
            return self._create_fixed_chunks(text, metadata)

        # Split by current separator
        if self.keep_separator:
            # Split but keep the separator
            splits = re.split(f"({re.escape(separator)})", text)
        else:
            splits = text.split(separator)

        # Merge splits into chunks
        chunks = []
        current_chunk = ""

        for i, split in enumerate(splits):
            # Skip empty splits
            if not split:
                continue

            # If this split would make chunk too large, finalize current chunk
            if len(current_chunk) + len(split) > self.chunk_size:
                if current_chunk:
                    # Recursively process current chunk with next separator
                    sub_chunks = self._split_text_recursive(
                        current_chunk,
                        metadata,
                        separator_index + 1
                    )
                    chunks.extend(sub_chunks)
                current_chunk = split
            else:
                current_chunk += split

        # Process remaining chunk
        if current_chunk:
            sub_chunks = self._split_text_recursive(
                current_chunk,
                metadata,
                separator_index + 1
            )
            chunks.extend(sub_chunks)

        return chunks

    def _create_fixed_chunks(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Create fixed-size chunks with overlap"""

        chunks = []
        start = 0

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

            # Move start forward with overlap
            start += self.chunk_size - self.chunk_overlap

            # Prevent infinite loop if overlap >= chunk_size
            if self.chunk_overlap >= self.chunk_size:
                start = end

        return chunks
