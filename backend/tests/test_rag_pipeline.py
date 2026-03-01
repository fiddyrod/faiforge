"""Tests for RAGPipeline with fully mocked components."""
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

from core.rag.pipeline import RAGPipeline, SearchMode, HybridConfig
from core.rag.chunking.base import Chunk, ChunkingResult
from core.rag.vector_stores.base import SearchResponse, SearchResult


# ---------------------------------------------------------------------------
# Mock factory helpers
# ---------------------------------------------------------------------------

def make_embedding_adapter(dims=8):
    adapter = MagicMock()
    adapter.model_name = "text-embedding-3-small"
    adapter.dimensions = dims
    adapter.embed_documents = AsyncMock(
        return_value=MagicMock(embeddings=[[0.1] * dims])
    )
    adapter.embed_query = AsyncMock(
        return_value=MagicMock(embeddings=[[0.1] * dims])
    )
    return adapter


def make_chunker(chunks_per_doc=2):
    adapter = MagicMock()
    adapter.strategy = MagicMock(value="recursive")

    def _fake_chunk(text, metadata=None):
        chunks = [
            Chunk(
                content=f"chunk-{i}: {text[:20]}",
                chunk_id=str(uuid.uuid4()),
                metadata=dict(metadata or {}),
                start_index=0,
                end_index=min(20, len(text)),
            )
            for i in range(chunks_per_doc)
        ]
        return ChunkingResult(chunks=chunks, total_chunks=len(chunks), strategy="recursive", latency_ms=1.0)

    adapter.chunk_text = AsyncMock(side_effect=lambda text, metadata=None: _fake_chunk(text, metadata))
    return adapter


def make_vector_store(n_results=1):
    store = MagicMock()
    store.add_documents = AsyncMock(return_value={"documents_added": n_results * 2})
    store.search = AsyncMock(return_value=SearchResponse(
        results=[
            SearchResult(id=f"chunk-{i}", content=f"Result {i}", score=0.9 - i * 0.1, metadata={})
            for i in range(n_results)
        ],
        query_embedding=[0.1] * 8,
        total_results=n_results,
        latency_ms=5.0,
    ))
    store.get_collection_info = AsyncMock(return_value={
        "provider": "chroma", "total_documents": 4, "collection": "test",
    })
    store.delete = AsyncMock(return_value={"deleted": 1})
    return store


DOCS = [
    {"content": "Python is a programming language.", "metadata": {"source": "python.txt"}},
    {"content": "Machine learning with TensorFlow.", "metadata": {"source": "ml.txt"}},
]


@pytest.fixture
def pipeline():
    return RAGPipeline(
        embedding_adapter=make_embedding_adapter(),
        chunking_adapter=make_chunker(),
        vector_store_adapter=make_vector_store(),
    )


@pytest.fixture
def hybrid_pipeline():
    return RAGPipeline(
        embedding_adapter=make_embedding_adapter(),
        chunking_adapter=make_chunker(),
        vector_store_adapter=make_vector_store(),
        hybrid_config=HybridConfig(enabled=True, semantic_weight=0.6, bm25_weight=0.4),
    )


# ---------------------------------------------------------------------------
# ingest_documents
# ---------------------------------------------------------------------------

class TestIngestDocuments:
    @pytest.mark.asyncio
    async def test_returns_expected_keys(self, pipeline):
        result = await pipeline.ingest_documents(DOCS)
        for key in ("documents_processed", "chunks_created", "embeddings_generated",
                    "vector_store_documents", "total_latency_ms",
                    "chunking_latency_ms", "embedding_latency_ms", "storage_latency_ms"):
            assert key in result

    @pytest.mark.asyncio
    async def test_documents_processed_count_matches_input(self, pipeline):
        result = await pipeline.ingest_documents(DOCS)
        assert result["documents_processed"] == len(DOCS)

    @pytest.mark.asyncio
    async def test_chunks_created_positive(self, pipeline):
        result = await pipeline.ingest_documents(DOCS)
        assert result["chunks_created"] > 0

    @pytest.mark.asyncio
    async def test_chunker_called_once_per_document(self, pipeline):
        await pipeline.ingest_documents(DOCS)
        assert pipeline.chunking_adapter.chunk_text.call_count == len(DOCS)

    @pytest.mark.asyncio
    async def test_embed_documents_called_once(self, pipeline):
        await pipeline.ingest_documents(DOCS)
        pipeline.embedding_adapter.embed_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_vector_store_add_called_once(self, pipeline):
        await pipeline.ingest_documents(DOCS)
        pipeline.vector_store.add_documents.assert_called_once()

    @pytest.mark.asyncio
    async def test_bm25_not_indexed_when_hybrid_disabled(self, pipeline):
        result = await pipeline.ingest_documents(DOCS)
        assert result["bm25_indexed"] == 0

    @pytest.mark.asyncio
    async def test_bm25_indexed_when_hybrid_enabled(self, hybrid_pipeline):
        result = await hybrid_pipeline.ingest_documents(DOCS)
        assert result["bm25_indexed"] > 0

    @pytest.mark.asyncio
    async def test_error_in_chunker_propagates(self, pipeline):
        pipeline.chunking_adapter.chunk_text = AsyncMock(side_effect=RuntimeError("chunking failed"))
        with pytest.raises(RuntimeError, match="chunking failed"):
            await pipeline.ingest_documents(DOCS)

    @pytest.mark.asyncio
    async def test_error_in_embedding_propagates(self, pipeline):
        pipeline.embedding_adapter.embed_documents = AsyncMock(side_effect=ValueError("embed error"))
        with pytest.raises(ValueError, match="embed error"):
            await pipeline.ingest_documents(DOCS)

    @pytest.mark.asyncio
    async def test_empty_doc_list_returns_zero_counts(self, pipeline):
        result = await pipeline.ingest_documents([])
        assert result["documents_processed"] == 0
        assert result["chunks_created"] == 0


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

class TestQuery:
    @pytest.mark.asyncio
    async def test_semantic_query_returns_search_response(self, pipeline):
        result = await pipeline.query("python", search_mode=SearchMode.SEMANTIC)
        assert isinstance(result, SearchResponse)
        assert len(result.results) > 0

    @pytest.mark.asyncio
    async def test_semantic_mode_calls_embed_query(self, pipeline):
        await pipeline.query("python", search_mode=SearchMode.SEMANTIC)
        pipeline.embedding_adapter.embed_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_semantic_mode_calls_vector_store_search(self, pipeline):
        await pipeline.query("python", search_mode=SearchMode.SEMANTIC)
        pipeline.vector_store.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_default_mode_is_semantic_when_hybrid_disabled(self, pipeline):
        await pipeline.query("query")
        pipeline.embedding_adapter.embed_query.assert_called()

    @pytest.mark.asyncio
    async def test_bm25_mode_without_bm25_index_falls_back_to_semantic(self, pipeline):
        """pipeline has no hybrid enabled, so BM25 mode falls back to semantic."""
        await pipeline.query("query", search_mode=SearchMode.BM25)
        pipeline.embedding_adapter.embed_query.assert_called()

    @pytest.mark.asyncio
    async def test_hybrid_mode_used_when_hybrid_enabled(self, hybrid_pipeline):
        await hybrid_pipeline.ingest_documents(DOCS)
        result = await hybrid_pipeline.query("python", search_mode=SearchMode.HYBRID)
        assert isinstance(result, SearchResponse)

    @pytest.mark.asyncio
    async def test_default_mode_is_hybrid_when_hybrid_enabled(self, hybrid_pipeline):
        await hybrid_pipeline.ingest_documents(DOCS)
        # Default should be HYBRID since hybrid_config.enabled=True
        result = await hybrid_pipeline.query("python")
        assert result is not None

    @pytest.mark.asyncio
    async def test_top_k_passed_to_vector_store(self, pipeline):
        await pipeline.query("python", top_k=3, search_mode=SearchMode.SEMANTIC)
        call_kwargs = pipeline.vector_store.search.call_args[1]
        assert call_kwargs.get("limit") == 3

    @pytest.mark.asyncio
    async def test_embed_error_propagates(self, pipeline):
        pipeline.embedding_adapter.embed_query = AsyncMock(side_effect=ConnectionError("embed failed"))
        with pytest.raises(ConnectionError, match="embed failed"):
            await pipeline.query("test")

    @pytest.mark.asyncio
    async def test_result_has_latency(self, pipeline):
        result = await pipeline.query("python", search_mode=SearchMode.SEMANTIC)
        assert result.latency_ms >= 0


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    @pytest.mark.asyncio
    async def test_stats_contains_required_fields(self, pipeline):
        stats = await pipeline.get_stats()
        for key in ("embedding_model", "embedding_dimensions", "vector_store", "hybrid_search"):
            assert key in stats

    @pytest.mark.asyncio
    async def test_hybrid_disabled_in_stats(self, pipeline):
        stats = await pipeline.get_stats()
        assert stats["hybrid_search"]["enabled"] is False

    @pytest.mark.asyncio
    async def test_hybrid_enabled_in_stats(self, hybrid_pipeline):
        stats = await hybrid_pipeline.get_stats()
        assert stats["hybrid_search"]["enabled"] is True

    @pytest.mark.asyncio
    async def test_embedding_model_name_correct(self, pipeline):
        stats = await pipeline.get_stats()
        assert stats["embedding_model"] == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_vector_store_info_included(self, pipeline):
        stats = await pipeline.get_stats()
        assert stats["vector_store"]["provider"] == "chroma"


# ---------------------------------------------------------------------------
# HybridConfig
# ---------------------------------------------------------------------------

class TestHybridConfig:
    def test_hybrid_searcher_created_when_enabled(self, hybrid_pipeline):
        assert hybrid_pipeline._hybrid_searcher is not None

    def test_bm25_retriever_created_when_enabled(self, hybrid_pipeline):
        assert hybrid_pipeline._bm25_retriever is not None

    def test_no_hybrid_searcher_when_disabled(self, pipeline):
        assert pipeline._hybrid_searcher is None
        assert pipeline._bm25_retriever is None

    def test_fusion_method_rrf_by_default(self, hybrid_pipeline):
        from core.rag.search.base import FusionMethod
        assert hybrid_pipeline._hybrid_searcher.config.fusion_method == FusionMethod.RRF

    def test_weights_respected(self, hybrid_pipeline):
        config = hybrid_pipeline._hybrid_searcher.config
        assert config.semantic_weight == 0.6
        assert config.bm25_weight == 0.4
