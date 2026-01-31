import time
import uuid
from typing import List, Dict, Any, Optional
import tiktoken

from .base import ChunkingAdapter, Chunk, ChunkingResult, ChunkingStrategy
from ...observability import get_logger, log_with_context


class TokenChunker(ChunkingAdapter):
    """Token-based chunking using tiktoken"""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize token-based chunker.

        Args:
            chunk_size: Number of tokens per chunk
            chunk_overlap: Number of overlapping tokens between chunks
            encoding_name: Tiktoken encoding name (e.g., 'cl100k_base' for GPT-4)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding_name = encoding_name
        self.logger = get_logger()

        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
            )

        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            self.logger.error(f"Failed to load tiktoken encoding {encoding_name}: {e}")
            raise

    @property
    def strategy(self) -> ChunkingStrategy:
        return ChunkingStrategy.TOKEN

    async def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChunkingResult:
        """Chunk text by token count"""

        start_time = time.time()

        log_with_context(
            self.logger,
            "info",
            "Starting token-based chunking",
            event="chunking_start",
            strategy="token",
            text_length=len(text),
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            encoding=self.encoding_name
        )

        # Tokenize the entire text
        tokens = self.encoding.encode(text)
        total_tokens = len(tokens)

        chunks = []
        start_idx = 0
        metadata = metadata or {}

        while start_idx < total_tokens:
            # Get token slice for this chunk
            end_idx = min(start_idx + self.chunk_size, total_tokens)
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode tokens back to text
            chunk_content = self.encoding.decode(chunk_tokens)

            chunk = Chunk(
                content=chunk_content,
                chunk_id=str(uuid.uuid4()),
                metadata=metadata.copy(),
                start_index=start_idx,  # Token indices, not character indices
                end_index=end_idx,
                token_count=len(chunk_tokens)
            )
            chunks.append(chunk)

            # Move start forward (chunk_size - overlap)
            start_idx += self.chunk_size - self.chunk_overlap

        latency_ms = (time.time() - start_time) * 1000

        log_with_context(
            self.logger,
            "info",
            "Token-based chunking completed",
            event="chunking_complete",
            strategy="token",
            total_chunks=len(chunks),
            total_tokens=total_tokens,
            latency_ms=round(latency_ms, 2)
        )

        return ChunkingResult(
            chunks=chunks,
            total_chunks=len(chunks),
            strategy="token",
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
            strategy="token",
            latency_ms=round(latency_ms, 2)
        )
