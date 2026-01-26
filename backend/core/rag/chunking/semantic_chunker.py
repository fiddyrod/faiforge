import time
import uuid
import re
from typing import List, Dict, Any, Optional
import numpy as np

from .base import ChunkingAdapter, Chunk, ChunkingResult, ChunkingStrategy
from ..embeddings.base import EmbeddingAdapter
from ...observability import get_logger, log_with_context


class SemanticChunker(ChunkingAdapter):
    """Semantic chunking using embedding similarity"""

    def __init__(
        self,
        embedding_adapter: EmbeddingAdapter,
        similarity_threshold: float = 0.5,
        max_chunk_size: int = 1500,
        min_chunk_size: int = 100
    ):
        """
        Initialize semantic chunker.

        Args:
            embedding_adapter: Embedding adapter for sentence embeddings
            similarity_threshold: Threshold for semantic boundary detection (0-1)
            max_chunk_size: Maximum chunk size in characters
            min_chunk_size: Minimum chunk size in characters
        """
        self.embedding_adapter = embedding_adapter
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.logger = get_logger()

        if not 0 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")

    @property
    def strategy(self) -> ChunkingStrategy:
        return ChunkingStrategy.SEMANTIC

    async def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChunkingResult:
        """Chunk text using semantic boundary detection"""

        start_time = time.time()

        log_with_context(
            self.logger,
            "info",
            "Starting semantic chunking",
            event="chunking_start",
            strategy="semantic",
            text_length=len(text),
            similarity_threshold=self.similarity_threshold,
            max_chunk_size=self.max_chunk_size
        )

        # Split into sentences
        sentences = self._split_sentences(text)

        if len(sentences) <= 1:
            # Not enough sentences to chunk semantically
            chunk = Chunk(
                content=text,
                chunk_id=str(uuid.uuid4()),
                metadata=(metadata or {}).copy(),
                start_index=0,
                end_index=len(text)
            )
            latency_ms = (time.time() - start_time) * 1000
            return ChunkingResult(
                chunks=[chunk],
                total_chunks=1,
                strategy="semantic",
                latency_ms=round(latency_ms, 2)
            )

        # Generate embeddings for sentences
        embedding_result = await self.embedding_adapter.embed_documents(sentences)
        embeddings = embedding_result.embeddings

        # Calculate similarity between adjacent sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Find chunk boundaries where similarity drops below threshold
        boundaries = [0]  # Start with first sentence
        for i, sim in enumerate(similarities):
            if sim < self.similarity_threshold:
                boundaries.append(i + 1)
        boundaries.append(len(sentences))  # End with last sentence

        # Create chunks from boundaries
        chunks = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            chunk_sentences = sentences[start_idx:end_idx]
            chunk_content = " ".join(chunk_sentences)

            # Split large chunks
            if len(chunk_content) > self.max_chunk_size:
                # Fall back to character-based splitting for large semantic chunks
                sub_chunks = self._split_large_chunk(chunk_content, metadata or {})
                chunks.extend(sub_chunks)
            else:
                chunk = Chunk(
                    content=chunk_content,
                    chunk_id=str(uuid.uuid4()),
                    metadata=(metadata or {}).copy(),
                    start_index=start_idx,
                    end_index=end_idx
                )
                chunks.append(chunk)

        # Merge small chunks
        chunks = self._merge_small_chunks(chunks)

        latency_ms = (time.time() - start_time) * 1000

        log_with_context(
            self.logger,
            "info",
            "Semantic chunking completed",
            event="chunking_complete",
            strategy="semantic",
            total_chunks=len(chunks),
            sentences_processed=len(sentences),
            latency_ms=round(latency_ms, 2)
        )

        return ChunkingResult(
            chunks=chunks,
            total_chunks=len(chunks),
            strategy="semantic",
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
            strategy="semantic",
            latency_ms=round(latency_ms, 2)
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter using regex
        # Matches periods, exclamation marks, question marks followed by space or newline
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter out empty sentences
        return [s.strip() for s in sentences if s.strip()]

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def _split_large_chunk(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[Chunk]:
        """Split a large semantic chunk into smaller chunks"""
        chunks = []
        start = 0

        while start < len(text):
            end = min(start + self.max_chunk_size, len(text))
            chunk_content = text[start:end]

            chunk = Chunk(
                content=chunk_content,
                chunk_id=str(uuid.uuid4()),
                metadata=metadata.copy(),
                start_index=start,
                end_index=end
            )
            chunks.append(chunk)
            start = end

        return chunks

    def _merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge chunks smaller than min_chunk_size"""
        if not chunks:
            return chunks

        merged = []
        current_chunk = chunks[0]

        for i in range(1, len(chunks)):
            next_chunk = chunks[i]

            # If current chunk is too small and merging won't exceed max size
            if (len(current_chunk.content) < self.min_chunk_size and
                len(current_chunk.content) + len(next_chunk.content) <= self.max_chunk_size):
                # Merge with next chunk
                current_chunk = Chunk(
                    content=current_chunk.content + " " + next_chunk.content,
                    chunk_id=str(uuid.uuid4()),
                    metadata=current_chunk.metadata.copy(),
                    start_index=current_chunk.start_index,
                    end_index=next_chunk.end_index
                )
            else:
                # Keep current chunk and move to next
                merged.append(current_chunk)
                current_chunk = next_chunk

        # Add the last chunk
        merged.append(current_chunk)

        return merged
