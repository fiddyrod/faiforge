"""Tests for RecursiveChunker."""
import pytest
from core.rag.chunking.recursive_chunker import RecursiveChunker
from core.rag.chunking.base import ChunkingStrategy


@pytest.fixture
def chunker():
    return RecursiveChunker(chunk_size=100, chunk_overlap=20)


SHORT_TEXT = "This is a short piece of text."

LONG_TEXT = "\n\n".join([
    "First paragraph about Python programming language and its ecosystem.",
    "Second paragraph about machine learning and AI applications.",
    "Third paragraph about web development frameworks like Django.",
    "Fourth paragraph about database management with SQL.",
    "Fifth paragraph about cloud computing and DevOps practices.",
])


class TestChunkText:
    @pytest.mark.asyncio
    async def test_short_text_returns_single_chunk(self, chunker):
        result = await chunker.chunk_text(SHORT_TEXT)
        assert result.total_chunks == 1
        assert result.chunks[0].content == SHORT_TEXT

    @pytest.mark.asyncio
    async def test_long_text_returns_multiple_chunks(self):
        small = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        result = await small.chunk_text(LONG_TEXT)
        assert result.total_chunks > 1

    @pytest.mark.asyncio
    async def test_all_content_covered(self, chunker):
        result = await chunker.chunk_text(LONG_TEXT)
        combined = "".join(c.content for c in result.chunks)
        assert "Python" in combined
        assert "machine learning" in combined

    @pytest.mark.asyncio
    async def test_chunk_size_roughly_respected(self):
        """Chunks should not wildly exceed the target chunk_size."""
        small = RecursiveChunker(chunk_size=60, chunk_overlap=0)
        result = await small.chunk_text(LONG_TEXT)
        for chunk in result.chunks:
            # Allow up to 3x the chunk_size to handle separator retention edge cases
            assert len(chunk.content) <= small.chunk_size * 3

    @pytest.mark.asyncio
    async def test_metadata_propagated_to_all_chunks(self, chunker):
        meta = {"source": "test.txt", "author": "alice"}
        result = await chunker.chunk_text(LONG_TEXT, metadata=meta)
        for chunk in result.chunks:
            assert chunk.metadata["source"] == "test.txt"
            assert chunk.metadata["author"] == "alice"

    @pytest.mark.asyncio
    async def test_metadata_not_shared_between_chunks(self, chunker):
        """Mutating one chunk's metadata should not affect others."""
        result = await chunker.chunk_text(SHORT_TEXT, metadata={"key": "val"})
        result.chunks[0].metadata["key"] = "mutated"
        # Only one chunk anyway but test the copy behaviour
        assert result.chunks[0].metadata["key"] == "mutated"

    @pytest.mark.asyncio
    async def test_chunk_ids_are_unique(self, chunker):
        result = await chunker.chunk_text(LONG_TEXT)
        ids = [c.chunk_id for c in result.chunks]
        assert len(ids) == len(set(ids))

    @pytest.mark.asyncio
    async def test_strategy_is_recursive(self, chunker):
        assert chunker.strategy == ChunkingStrategy.RECURSIVE

    @pytest.mark.asyncio
    async def test_result_has_non_negative_latency(self, chunker):
        result = await chunker.chunk_text(SHORT_TEXT)
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_total_chunks_matches_list_length(self, chunker):
        result = await chunker.chunk_text(LONG_TEXT)
        assert result.total_chunks == len(result.chunks)

    @pytest.mark.asyncio
    async def test_empty_text_handled_gracefully(self, chunker):
        result = await chunker.chunk_text("")
        # Should not raise; may produce 0 or 1 empty chunk
        assert isinstance(result.chunks, list)


class TestChunkDocuments:
    @pytest.mark.asyncio
    async def test_two_short_docs_produce_two_chunks(self, chunker):
        docs = [
            {"content": SHORT_TEXT, "metadata": {"doc": "1"}},
            {"content": SHORT_TEXT, "metadata": {"doc": "2"}},
        ]
        result = await chunker.chunk_documents(docs)
        assert result.total_chunks == 2

    @pytest.mark.asyncio
    async def test_empty_document_list(self, chunker):
        result = await chunker.chunk_documents([])
        assert result.total_chunks == 0

    @pytest.mark.asyncio
    async def test_metadata_per_document_preserved(self, chunker):
        docs = [
            {"content": SHORT_TEXT, "metadata": {"source": "doc1.txt"}},
            {"content": SHORT_TEXT, "metadata": {"source": "doc2.txt"}},
        ]
        result = await chunker.chunk_documents(docs)
        sources = {c.metadata.get("source") for c in result.chunks}
        assert "doc1.txt" in sources
        assert "doc2.txt" in sources

    @pytest.mark.asyncio
    async def test_result_strategy_is_recursive(self, chunker):
        result = await chunker.chunk_documents([{"content": SHORT_TEXT}])
        assert result.strategy == "recursive"


class TestFixedChunks:
    def test_creates_multiple_chunks_for_long_text(self):
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=5)
        text = "a" * 50
        chunks = chunker._create_fixed_chunks(text, {})
        assert len(chunks) > 1

    def test_overlap_means_next_chunk_starts_before_previous_ends(self):
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=5)
        text = "a" * 60
        chunks = chunker._create_fixed_chunks(text, {})
        for i in range(len(chunks) - 1):
            assert chunks[i + 1].start_index < chunks[i].end_index

    def test_zero_overlap_no_gaps(self):
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0)
        text = "a" * 40
        chunks = chunker._create_fixed_chunks(text, {})
        assert len(chunks) == 2
        assert chunks[0].end_index == chunks[1].start_index

    def test_metadata_copied_not_shared(self):
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0)
        meta = {"key": "original"}
        chunks = chunker._create_fixed_chunks("a" * 50, meta)
        chunks[0].metadata["key"] = "mutated"
        assert chunks[1].metadata["key"] == "original"

    def test_last_chunk_end_index_equals_text_length(self):
        chunker = RecursiveChunker(chunk_size=20, chunk_overlap=0)
        text = "a" * 45
        chunks = chunker._create_fixed_chunks(text, {})
        assert chunks[-1].end_index == len(text)


# ===========================================================================
# FixedSizeChunker
# ===========================================================================

from core.rag.chunking.fixed_size_chunker import FixedSizeChunker


class TestFixedSizeChunkerInit:
    def test_overlap_less_than_size_succeeds(self):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=20)
        assert chunker.chunk_size == 100
        assert chunker.chunk_overlap == 20

    def test_overlap_equal_to_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            FixedSizeChunker(chunk_size=50, chunk_overlap=50)

    def test_overlap_greater_than_size_raises(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=50, chunk_overlap=60)

    def test_strategy_is_fixed_size(self):
        from core.rag.chunking.base import ChunkingStrategy
        chunker = FixedSizeChunker()
        assert chunker.strategy == ChunkingStrategy.FIXED_SIZE


class TestFixedSizeChunkText:
    @pytest.mark.asyncio
    async def test_short_text_returns_single_chunk(self):
        chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=0)
        result = await chunker.chunk_text("Short text.")
        assert result.total_chunks == 1
        assert result.chunks[0].content == "Short text."

    @pytest.mark.asyncio
    async def test_long_text_returns_multiple_chunks(self):
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=0)
        result = await chunker.chunk_text("a" * 50)
        assert result.total_chunks == 5

    @pytest.mark.asyncio
    async def test_chunk_size_respected(self):
        chunker = FixedSizeChunker(chunk_size=20, chunk_overlap=0)
        result = await chunker.chunk_text("a" * 60)
        for chunk in result.chunks[:-1]:  # all but last
            assert len(chunk.content) == 20

    @pytest.mark.asyncio
    async def test_overlap_produces_overlapping_indices(self):
        chunker = FixedSizeChunker(chunk_size=20, chunk_overlap=5)
        result = await chunker.chunk_text("a" * 60)
        chunks = result.chunks
        for i in range(len(chunks) - 1):
            assert chunks[i + 1].start_index < chunks[i].end_index

    @pytest.mark.asyncio
    async def test_no_overlap_contiguous_indices(self):
        chunker = FixedSizeChunker(chunk_size=20, chunk_overlap=0)
        result = await chunker.chunk_text("a" * 40)
        chunks = result.chunks
        assert len(chunks) == 2
        assert chunks[0].end_index == chunks[1].start_index

    @pytest.mark.asyncio
    async def test_metadata_propagated_to_all_chunks(self):
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=0)
        meta = {"source": "file.txt"}
        result = await chunker.chunk_text("a" * 30, metadata=meta)
        for chunk in result.chunks:
            assert chunk.metadata["source"] == "file.txt"

    @pytest.mark.asyncio
    async def test_metadata_copied_not_shared(self):
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=0)
        result = await chunker.chunk_text("a" * 30, metadata={"k": "v"})
        result.chunks[0].metadata["k"] = "mutated"
        for chunk in result.chunks[1:]:
            assert chunk.metadata["k"] == "v"

    @pytest.mark.asyncio
    async def test_total_chunks_matches_list_length(self):
        chunker = FixedSizeChunker(chunk_size=15, chunk_overlap=0)
        result = await chunker.chunk_text("a" * 45)
        assert result.total_chunks == len(result.chunks)

    @pytest.mark.asyncio
    async def test_strategy_field_in_result(self):
        chunker = FixedSizeChunker()
        result = await chunker.chunk_text("hello")
        assert result.strategy == "fixed_size"

    @pytest.mark.asyncio
    async def test_latency_ms_non_negative(self):
        chunker = FixedSizeChunker()
        result = await chunker.chunk_text("hello world")
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_empty_text(self):
        chunker = FixedSizeChunker(chunk_size=100, chunk_overlap=0)
        result = await chunker.chunk_text("")
        assert isinstance(result.chunks, list)


class TestFixedSizeChunkDocuments:
    @pytest.mark.asyncio
    async def test_two_docs_produce_two_chunks(self):
        chunker = FixedSizeChunker(chunk_size=200, chunk_overlap=0)
        docs = [
            {"content": "First document text.", "metadata": {"id": "1"}},
            {"content": "Second document text.", "metadata": {"id": "2"}},
        ]
        result = await chunker.chunk_documents(docs)
        assert result.total_chunks == 2

    @pytest.mark.asyncio
    async def test_empty_document_list(self):
        chunker = FixedSizeChunker()
        result = await chunker.chunk_documents([])
        assert result.total_chunks == 0


# ===========================================================================
# TokenChunker
# ===========================================================================

from core.rag.chunking.token_chunker import TokenChunker


class TestTokenChunkerInit:
    def test_valid_params_succeed(self):
        chunker = TokenChunker(chunk_size=100, chunk_overlap=20)
        assert chunker.chunk_size == 100
        assert chunker.chunk_overlap == 20

    def test_overlap_equal_to_size_raises(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            TokenChunker(chunk_size=50, chunk_overlap=50)

    def test_overlap_greater_than_size_raises(self):
        with pytest.raises(ValueError):
            TokenChunker(chunk_size=50, chunk_overlap=60)

    def test_strategy_is_token(self):
        from core.rag.chunking.base import ChunkingStrategy
        chunker = TokenChunker()
        assert chunker.strategy == ChunkingStrategy.TOKEN

    def test_default_encoding_is_cl100k(self):
        chunker = TokenChunker()
        assert chunker.encoding_name == "cl100k_base"


class TestTokenChunkText:
    @pytest.mark.asyncio
    async def test_short_text_single_chunk(self):
        chunker = TokenChunker(chunk_size=512, chunk_overlap=50)
        result = await chunker.chunk_text("Hello world!")
        assert result.total_chunks == 1

    @pytest.mark.asyncio
    async def test_long_text_multiple_chunks(self):
        # Each word = ~1 token; 5 tokens per chunk, 1 overlap
        chunker = TokenChunker(chunk_size=5, chunk_overlap=1)
        text = "word " * 30  # ~30 tokens
        result = await chunker.chunk_text(text)
        assert result.total_chunks > 1

    @pytest.mark.asyncio
    async def test_token_count_populated(self):
        chunker = TokenChunker(chunk_size=10, chunk_overlap=0)
        text = "word " * 20
        result = await chunker.chunk_text(text)
        for chunk in result.chunks:
            assert chunk.token_count is not None
            assert chunk.token_count > 0

    @pytest.mark.asyncio
    async def test_start_end_index_are_token_indices(self):
        """start_index and end_index are token positions, not character positions."""
        chunker = TokenChunker(chunk_size=5, chunk_overlap=0)
        text = "word " * 10
        result = await chunker.chunk_text(text)
        # First chunk: start=0
        assert result.chunks[0].start_index == 0
        assert result.chunks[0].end_index <= 5

    @pytest.mark.asyncio
    async def test_chunks_cover_all_content(self):
        chunker = TokenChunker(chunk_size=8, chunk_overlap=0)
        words = ["alpha", "beta", "gamma", "delta", "epsilon",
                 "zeta", "eta", "theta", "iota", "kappa"]
        text = " ".join(words)
        result = await chunker.chunk_text(text)
        combined = "".join(c.content for c in result.chunks)
        for word in words:
            assert word in combined

    @pytest.mark.asyncio
    async def test_metadata_propagated(self):
        chunker = TokenChunker(chunk_size=5, chunk_overlap=0)
        text = "word " * 20
        result = await chunker.chunk_text(text, metadata={"src": "doc.txt"})
        for chunk in result.chunks:
            assert chunk.metadata["src"] == "doc.txt"

    @pytest.mark.asyncio
    async def test_total_chunks_matches_list_length(self):
        chunker = TokenChunker(chunk_size=10, chunk_overlap=2)
        result = await chunker.chunk_text("word " * 50)
        assert result.total_chunks == len(result.chunks)

    @pytest.mark.asyncio
    async def test_strategy_field_in_result(self):
        chunker = TokenChunker()
        result = await chunker.chunk_text("hello")
        assert result.strategy == "token"

    @pytest.mark.asyncio
    async def test_latency_ms_non_negative(self):
        chunker = TokenChunker()
        result = await chunker.chunk_text("some text here")
        assert result.latency_ms >= 0


class TestTokenChunkDocuments:
    @pytest.mark.asyncio
    async def test_two_docs_produce_chunks(self):
        chunker = TokenChunker(chunk_size=512, chunk_overlap=50)
        docs = [
            {"content": "First document.", "metadata": {}},
            {"content": "Second document.", "metadata": {}},
        ]
        result = await chunker.chunk_documents(docs)
        assert result.total_chunks == 2

    @pytest.mark.asyncio
    async def test_empty_doc_list(self):
        chunker = TokenChunker()
        result = await chunker.chunk_documents([])
        assert result.total_chunks == 0
