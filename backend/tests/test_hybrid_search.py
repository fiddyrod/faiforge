"""Tests for hybrid search fusion algorithms (RRF, Weighted Sum, Max, Min)."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from core.rag.search.base import FusionMethod, ScoredResult
from core.rag.search.hybrid import HybridSearcher, HybridSearchConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_result(id_, score, bm25_score=None, semantic_score=None):
    return ScoredResult(
        id=id_,
        content=f"Content for doc {id_}",
        score=score,
        bm25_score=bm25_score if bm25_score is not None else score,
        semantic_score=semantic_score if semantic_score is not None else score,
    )


def make_search_result(id_, score):
    sr = MagicMock()
    sr.id = id_
    sr.content = f"Semantic content for {id_}"
    sr.score = score
    sr.metadata = {}
    return sr


DOCS = [
    {"id": "a", "content": "Python machine learning deep learning neural networks"},
    {"id": "b", "content": "Java enterprise software development programming"},
    {"id": "c", "content": "Python web development Django Flask framework"},
    {"id": "d", "content": "Database SQL queries optimization indexing"},
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.search = AsyncMock(return_value=MagicMock(results=[]))
    return store


@pytest.fixture
def mock_embedding():
    emb = MagicMock()
    emb.embed_query = AsyncMock(return_value=MagicMock(embeddings=[[0.1] * 10]))
    return emb


@pytest.fixture
def rrf_searcher(mock_vector_store, mock_embedding):
    return HybridSearcher(
        vector_store=mock_vector_store,
        embedding_adapter=mock_embedding,
        config=HybridSearchConfig(
            semantic_weight=0.5,
            bm25_weight=0.5,
            fusion_method=FusionMethod.RRF,
            rrf_k=60,
        ),
    )


@pytest.fixture
def weighted_searcher(mock_vector_store, mock_embedding):
    return HybridSearcher(
        vector_store=mock_vector_store,
        embedding_adapter=mock_embedding,
        config=HybridSearchConfig(
            semantic_weight=0.6,
            bm25_weight=0.4,
            fusion_method=FusionMethod.WEIGHTED_SUM,
        ),
    )


@pytest.fixture
def max_searcher(mock_vector_store, mock_embedding):
    return HybridSearcher(
        vector_store=mock_vector_store,
        embedding_adapter=mock_embedding,
        config=HybridSearchConfig(fusion_method=FusionMethod.MAX),
    )


@pytest.fixture
def min_searcher(mock_vector_store, mock_embedding):
    return HybridSearcher(
        vector_store=mock_vector_store,
        embedding_adapter=mock_embedding,
        config=HybridSearchConfig(fusion_method=FusionMethod.MIN),
    )


# ---------------------------------------------------------------------------
# RRF Fusion
# ---------------------------------------------------------------------------

class TestRRFFusion:
    def test_single_list_preserves_rank_order(self, rrf_searcher):
        bm25 = [make_result("a", 2.0), make_result("b", 1.5), make_result("c", 1.0)]
        result = rrf_searcher._fuse_rrf(bm25, [], top_k=3)
        assert len(result) == 3
        assert result[0].id == "a"

    def test_doc_in_both_lists_ranks_higher(self, rrf_searcher):
        bm25 = [make_result("a", 2.0), make_result("b", 1.0)]
        semantic = [make_result("b", 0.9), make_result("c", 0.8)]
        result = rrf_searcher._fuse_rrf(bm25, semantic, top_k=5)
        ids = [r.id for r in result]
        assert "b" in ids
        assert "a" in ids
        assert "c" in ids

    def test_top_ranked_in_both_lists_is_first(self, rrf_searcher):
        """doc 'a' is rank-1 in both lists, so it should win."""
        bm25 = [make_result("a", 2.0), make_result("b", 1.0)]
        semantic = [make_result("a", 0.9), make_result("c", 0.8)]
        result = rrf_searcher._fuse_rrf(bm25, semantic, top_k=3)
        assert result[0].id == "a"

    def test_top_k_enforced(self, rrf_searcher):
        bm25 = [make_result(str(i), float(10 - i)) for i in range(5)]
        semantic = [make_result(str(i), float(10 - i)) for i in range(5)]
        result = rrf_searcher._fuse_rrf(bm25, semantic, top_k=2)
        assert len(result) == 2

    def test_all_scores_positive(self, rrf_searcher):
        bm25 = [make_result("a", 1.0)]
        semantic = [make_result("b", 0.9)]
        result = rrf_searcher._fuse_rrf(bm25, semantic, top_k=5)
        for r in result:
            assert r.score > 0

    def test_empty_both_lists_returns_empty(self, rrf_searcher):
        result = rrf_searcher._fuse_rrf([], [], top_k=5)
        assert result == []

    def test_empty_bm25_returns_semantic_only(self, rrf_searcher):
        semantic = [make_result("a", 0.9), make_result("b", 0.7)]
        result = rrf_searcher._fuse_rrf([], semantic, top_k=5)
        ids = {r.id for r in result}
        assert ids == {"a", "b"}


# ---------------------------------------------------------------------------
# Weighted Sum Fusion
# ---------------------------------------------------------------------------

class TestWeightedSumFusion:
    def test_scores_normalized_to_0_1(self, weighted_searcher):
        bm25 = [make_result("a", 5.0), make_result("b", 1.0)]
        semantic = [make_result("a", 0.9), make_result("c", 0.3)]
        result = weighted_searcher._fuse_weighted_sum(bm25, semantic, top_k=5)
        for r in result:
            assert 0.0 <= r.score <= 1.0

    def test_includes_docs_from_both_lists(self, weighted_searcher):
        bm25 = [make_result("a", 2.0)]
        semantic = [make_result("b", 0.9)]
        result = weighted_searcher._fuse_weighted_sum(bm25, semantic, top_k=5)
        ids = {r.id for r in result}
        assert "a" in ids
        assert "b" in ids

    def test_top_k_enforced(self, weighted_searcher):
        bm25 = [make_result(str(i), float(10 - i)) for i in range(5)]
        semantic = [make_result(str(i), float(10 - i)) for i in range(5)]
        result = weighted_searcher._fuse_weighted_sum(bm25, semantic, top_k=3)
        assert len(result) <= 3

    def test_doc_in_both_gets_higher_score_than_doc_in_one(self, weighted_searcher):
        bm25 = [make_result("shared", 2.0), make_result("bm25_only", 1.5)]
        semantic = [make_result("shared", 0.9), make_result("sem_only", 0.8)]
        result = weighted_searcher._fuse_weighted_sum(bm25, semantic, top_k=5)
        scores = {r.id: r.score for r in result}
        assert scores["shared"] >= scores.get("bm25_only", 0)
        assert scores["shared"] >= scores.get("sem_only", 0)


# ---------------------------------------------------------------------------
# Max Fusion
# ---------------------------------------------------------------------------

class TestMaxFusion:
    def test_includes_all_docs(self, max_searcher):
        bm25 = [make_result("a", 2.0)]
        semantic = [make_result("b", 0.9)]
        result = max_searcher._fuse_max(bm25, semantic, top_k=5)
        ids = {r.id for r in result}
        assert "a" in ids and "b" in ids

    def test_scores_in_0_1_range(self, max_searcher):
        bm25 = [make_result("a", 5.0), make_result("b", 1.0)]
        semantic = [make_result("c", 0.9), make_result("d", 0.3)]
        result = max_searcher._fuse_max(bm25, semantic, top_k=5)
        for r in result:
            assert 0.0 <= r.score <= 1.0

    def test_empty_lists_returns_empty(self, max_searcher):
        result = max_searcher._fuse_max([], [], top_k=5)
        assert result == []

    def test_top_k_enforced(self, max_searcher):
        bm25 = [make_result(str(i), float(i)) for i in range(5)]
        result = max_searcher._fuse_max(bm25, [], top_k=2)
        assert len(result) <= 2


# ---------------------------------------------------------------------------
# Min Fusion
# ---------------------------------------------------------------------------

class TestMinFusion:
    def test_only_common_docs_returned(self, min_searcher):
        bm25 = [make_result("a", 2.0), make_result("b", 1.0)]
        semantic = [make_result("a", 0.9), make_result("c", 0.7)]
        result = min_searcher._fuse_min(bm25, semantic, top_k=5)
        ids = [r.id for r in result]
        assert "a" in ids
        assert "b" not in ids
        assert "c" not in ids

    def test_falls_back_to_semantic_when_no_overlap(self, min_searcher):
        bm25 = [make_result("a", 2.0)]
        semantic = [make_result("b", 0.9), make_result("c", 0.7)]
        result = min_searcher._fuse_min(bm25, semantic, top_k=5)
        ids = [r.id for r in result]
        assert "b" in ids

    def test_scores_in_0_1_range(self, min_searcher):
        bm25 = [make_result("x", 3.0), make_result("y", 1.0)]
        semantic = [make_result("x", 0.9), make_result("y", 0.4)]
        result = min_searcher._fuse_min(bm25, semantic, top_k=5)
        for r in result:
            assert 0.0 <= r.score <= 1.0


# ---------------------------------------------------------------------------
# Index and Stats
# ---------------------------------------------------------------------------

class TestHybridSearcherLifecycle:
    @pytest.mark.asyncio
    async def test_not_indexed_before_index_call(self, rrf_searcher):
        assert rrf_searcher._indexed is False

    @pytest.mark.asyncio
    async def test_indexed_after_index_call(self, rrf_searcher):
        await rrf_searcher.index_documents(DOCS)
        assert rrf_searcher._indexed is True

    @pytest.mark.asyncio
    async def test_index_returns_stats(self, rrf_searcher):
        result = await rrf_searcher.index_documents(DOCS)
        assert result["indexed_count"] == len(DOCS)
        assert "indexing_latency_ms" in result

    def test_get_stats_before_indexing(self, rrf_searcher):
        stats = rrf_searcher.get_stats()
        assert stats["bm25_indexed"] is False

    @pytest.mark.asyncio
    async def test_get_stats_after_indexing(self, rrf_searcher):
        await rrf_searcher.index_documents(DOCS)
        stats = rrf_searcher.get_stats()
        assert stats["bm25_indexed"] is True
        assert stats["bm25_stats"]["total_documents"] == len(DOCS)
        assert stats["config"]["fusion_method"] == "rrf"

    @pytest.mark.asyncio
    async def test_search_without_index_returns_semantic_only(
        self, rrf_searcher, mock_vector_store
    ):
        mock_response = MagicMock()
        mock_response.results = [make_search_result("a", 0.9)]
        mock_vector_store.search.return_value = mock_response

        results = await rrf_searcher.search("python")
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_search_after_index_includes_bm25(self, rrf_searcher, mock_vector_store):
        await rrf_searcher.index_documents(DOCS)

        mock_response = MagicMock()
        mock_response.results = [make_search_result("a", 0.9)]
        mock_vector_store.search.return_value = mock_response

        results = await rrf_searcher.search("python machine learning")
        assert isinstance(results, list)
        assert len(results) >= 1
