"""Tests for semantic caching layer and in-memory backend."""
import time
import pytest
from unittest.mock import AsyncMock

from core.cache.semantic import SemanticCache
from core.cache.backends.memory import MemoryCacheBackend
from core.cache.models import CacheEntry, CacheStats


# ---------------------------------------------------------------------------
# Fixed embeddings for deterministic tests
# ---------------------------------------------------------------------------

IDENTICAL  = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
SIMILAR    = [0.995, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # cosine ~0.995
DIFFERENT  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]     # cosine ~0.0


def make_cache(embedding=IDENTICAL, threshold=0.95):
    backend = MemoryCacheBackend()
    embedding_fn = AsyncMock(return_value=embedding)
    return SemanticCache(
        backend=backend,
        embedding_fn=embedding_fn,
        similarity_threshold=threshold,
    ), embedding_fn


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_return_1(self):
        sim = SemanticCache._compute_similarity([1.0, 0.0], [1.0, 0.0])
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors_return_0(self):
        sim = SemanticCache._compute_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(sim) < 1e-6

    def test_opposite_vectors_return_minus_1(self):
        sim = SemanticCache._compute_similarity([1.0, 0.0], [-1.0, 0.0])
        assert abs(sim - (-1.0)) < 1e-6

    def test_zero_vector_returns_0(self):
        sim = SemanticCache._compute_similarity([0.0, 0.0], [1.0, 0.0])
        assert sim == 0.0

    def test_similar_vectors_above_threshold(self):
        sim = SemanticCache._compute_similarity(IDENTICAL, SIMILAR)
        assert sim > 0.95


# ---------------------------------------------------------------------------
# Cache key generation
# ---------------------------------------------------------------------------

class TestCacheKeyGeneration:
    def test_same_query_produces_same_key(self):
        k1 = SemanticCache._generate_key("What is Python?")
        k2 = SemanticCache._generate_key("What is Python?")
        assert k1 == k2

    def test_different_queries_produce_different_keys(self):
        k1 = SemanticCache._generate_key("What is Python?")
        k2 = SemanticCache._generate_key("What is Java?")
        assert k1 != k2

    def test_key_is_16_chars(self):
        k = SemanticCache._generate_key("some query")
        assert len(k) == 16


# ---------------------------------------------------------------------------
# Set and Get
# ---------------------------------------------------------------------------

class TestSetAndGet:
    @pytest.mark.asyncio
    async def test_miss_on_empty_cache(self):
        cache, _ = make_cache()
        result = await cache.get("What is Python?")
        assert result is None

    @pytest.mark.asyncio
    async def test_hit_after_set_same_embedding(self):
        cache, _ = make_cache(embedding=IDENTICAL)
        await cache.set("What is Python?", {"content": "Python is a language"})
        result = await cache.get("What is Python?")
        assert result is not None
        response, similarity = result
        assert response == {"content": "Python is a language"}
        assert similarity >= 0.95

    @pytest.mark.asyncio
    async def test_miss_when_embedding_too_different(self):
        backend = MemoryCacheBackend()
        embedding_fn = AsyncMock()

        cache = SemanticCache(
            backend=backend,
            embedding_fn=embedding_fn,
            similarity_threshold=0.95,
        )

        embedding_fn.return_value = IDENTICAL
        await cache.set("question one", {"content": "answer one"})

        embedding_fn.return_value = DIFFERENT
        result = await cache.get("completely different query")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_best_match_among_multiple_entries(self):
        backend = MemoryCacheBackend()
        embedding_fn = AsyncMock()
        cache = SemanticCache(backend=backend, embedding_fn=embedding_fn, similarity_threshold=0.95)

        embedding_fn.return_value = IDENTICAL
        await cache.set("question one", {"content": "answer one"})

        embedding_fn.return_value = SIMILAR
        await cache.set("question two", {"content": "answer two"})

        # Query with IDENTICAL embedding — should match "question one"
        embedding_fn.return_value = IDENTICAL
        result = await cache.get("query")
        assert result is not None


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestCacheStats:
    @pytest.mark.asyncio
    async def test_initial_stats_all_zero(self):
        cache, _ = make_cache()
        stats = await cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["total_queries"] == 0
        assert stats["hit_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_miss_increments_misses_and_total(self):
        cache, _ = make_cache()
        await cache.get("unknown query")
        assert cache.stats.misses == 1
        assert cache.stats.total_queries == 1

    @pytest.mark.asyncio
    async def test_hit_increments_hits_and_total(self):
        cache, _ = make_cache()
        await cache.set("python question", {"content": "answer"})
        await cache.get("python question")
        assert cache.stats.hits == 1
        assert cache.stats.total_queries == 1

    @pytest.mark.asyncio
    async def test_hit_rate_is_50_percent(self):
        backend = MemoryCacheBackend()
        embedding_fn = AsyncMock()
        cache = SemanticCache(backend=backend, embedding_fn=embedding_fn, similarity_threshold=0.95)

        # Store with IDENTICAL embedding
        embedding_fn.return_value = IDENTICAL
        await cache.set("q1", {"content": "a1"})

        # Query 1: same embedding -> hit
        embedding_fn.return_value = IDENTICAL
        await cache.get("q1")

        # Query 2: orthogonal embedding -> miss (cosine similarity ~0.0)
        embedding_fn.return_value = DIFFERENT
        await cache.get("completely different query")

        stats = await cache.get_stats()
        assert stats["hit_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_avg_similarity_on_hit_updated(self):
        cache, _ = make_cache()
        await cache.set("q1", {"content": "a1"})
        await cache.get("q1")
        assert cache.stats.avg_similarity_on_hit > 0.0

    @pytest.mark.asyncio
    async def test_cache_size_in_stats(self):
        cache, _ = make_cache()
        await cache.set("q1", {"content": "a1"})
        await cache.set("q2", {"content": "a2"})
        stats = await cache.get_stats()
        assert stats["cache_size"] >= 2


# ---------------------------------------------------------------------------
# Metadata filtering
# ---------------------------------------------------------------------------

class TestMetadataFilter:
    @pytest.mark.asyncio
    async def test_filter_matches_correct_entry(self):
        cache, _ = make_cache()
        await cache.set("question", {"content": "answer"}, metadata={"model": "gpt-4"})
        result = await cache.get("question", metadata_filter={"model": "gpt-4"})
        assert result is not None

    @pytest.mark.asyncio
    async def test_filter_blocks_wrong_metadata(self):
        cache, _ = make_cache()
        await cache.set("question", {"content": "answer"}, metadata={"model": "gpt-4"})
        result = await cache.get("question", metadata_filter={"model": "claude"})
        assert result is None


# ---------------------------------------------------------------------------
# Clear and invalidate
# ---------------------------------------------------------------------------

class TestClearAndInvalidate:
    @pytest.mark.asyncio
    async def test_clear_removes_all_entries(self):
        cache, _ = make_cache()
        await cache.set("q1", {"content": "a1"})
        await cache.set("q2", {"content": "a2"})
        cleared = await cache.clear()
        assert cleared >= 2
        assert await cache.get("q1") is None

    @pytest.mark.asyncio
    async def test_clear_resets_stats(self):
        cache, _ = make_cache()
        await cache.set("q1", {"content": "a1"})
        await cache.get("q1")   # hit
        await cache.clear()
        assert cache.stats.hits == 0
        assert cache.stats.misses == 0

    @pytest.mark.asyncio
    async def test_invalidate_specific_entry(self):
        cache, _ = make_cache()
        await cache.set("q1", {"content": "a1"})
        removed = await cache.invalidate("q1")
        assert removed is True
        result = await cache.get("q1")
        assert result is None

    @pytest.mark.asyncio
    async def test_invalidate_nonexistent_returns_false(self):
        cache, _ = make_cache()
        removed = await cache.invalidate("does_not_exist")
        assert removed is False


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------

class TestTTL:
    @pytest.mark.asyncio
    async def test_ttl_override_stored_in_entry(self):
        backend = MemoryCacheBackend()
        embedding_fn = AsyncMock(return_value=IDENTICAL)
        cache = SemanticCache(backend=backend, embedding_fn=embedding_fn, similarity_threshold=0.95)
        key = await cache.set("q", {"content": "a"}, ttl=7200)
        entries = await backend.get_all_entries()
        assert entries[0].ttl == 7200


# ---------------------------------------------------------------------------
# In-memory cache backend (standalone)
# ---------------------------------------------------------------------------

class TestMemoryCacheBackend:
    @pytest.mark.asyncio
    async def test_set_and_get(self):
        backend = MemoryCacheBackend()
        entry = CacheEntry(
            query="test", response={"content": "answer"},
            embedding=[0.1, 0.2], created_at=time.time(), ttl=3600,
        )
        await backend.set("key1", entry)
        result = await backend.get("key1")
        assert result is not None
        assert result.query == "test"

    @pytest.mark.asyncio
    async def test_missing_key_returns_none(self):
        backend = MemoryCacheBackend()
        assert await backend.get("nonexistent") is None

    @pytest.mark.asyncio
    async def test_delete_returns_true_and_removes_entry(self):
        backend = MemoryCacheBackend()
        entry = CacheEntry(query="q", response={}, embedding=[0.1], created_at=time.time(), ttl=3600)
        await backend.set("key1", entry)
        assert await backend.delete("key1") is True
        assert await backend.get("key1") is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(self):
        backend = MemoryCacheBackend()
        assert await backend.delete("nope") is False

    @pytest.mark.asyncio
    async def test_clear_returns_count_and_empties_store(self):
        backend = MemoryCacheBackend()
        for i in range(5):
            entry = CacheEntry(query=f"q{i}", response={}, embedding=[0.1], created_at=time.time(), ttl=3600)
            await backend.set(f"key{i}", entry)
        cleared = await backend.clear()
        assert cleared == 5
        assert await backend.size() == 0

    @pytest.mark.asyncio
    async def test_expired_entry_returns_none(self):
        backend = MemoryCacheBackend()
        entry = CacheEntry(
            query="q", response={}, embedding=[0.1],
            created_at=time.time() - 100,  # 100 seconds ago
            ttl=10,                          # expired 90 seconds ago
        )
        await backend.set("key1", entry)
        assert await backend.get("key1") is None

    @pytest.mark.asyncio
    async def test_get_all_entries_excludes_expired(self):
        backend = MemoryCacheBackend()
        valid = CacheEntry(query="valid", response={}, embedding=[0.1], created_at=time.time(), ttl=3600)
        expired = CacheEntry(query="expired", response={}, embedding=[0.1], created_at=time.time() - 200, ttl=10)
        await backend.set("valid", valid)
        await backend.set("expired", expired)
        entries = await backend.get_all_entries()
        assert len(entries) == 1
        assert entries[0].query == "valid"

    @pytest.mark.asyncio
    async def test_lru_eviction_at_capacity(self):
        backend = MemoryCacheBackend(max_entries=3)
        for i in range(4):
            entry = CacheEntry(query=f"q{i}", response={}, embedding=[0.1], created_at=time.time(), ttl=3600)
            await backend.set(f"key{i}", entry)
        assert await backend.size() == 3

    @pytest.mark.asyncio
    async def test_size_reflects_current_count(self):
        backend = MemoryCacheBackend()
        assert await backend.size() == 0
        entry = CacheEntry(query="q", response={}, embedding=[0.1], created_at=time.time(), ttl=3600)
        await backend.set("k", entry)
        assert await backend.size() == 1


# ---------------------------------------------------------------------------
# CacheEntry model
# ---------------------------------------------------------------------------

class TestCacheEntry:
    def test_not_expired_when_ttl_none(self):
        entry = CacheEntry(query="q", response={}, embedding=[], created_at=time.time(), ttl=None)
        assert entry.is_expired is False

    def test_expired_when_past_ttl(self):
        entry = CacheEntry(query="q", response={}, embedding=[], created_at=time.time() - 100, ttl=10)
        assert entry.is_expired is True

    def test_not_expired_when_within_ttl(self):
        entry = CacheEntry(query="q", response={}, embedding=[], created_at=time.time(), ttl=3600)
        assert entry.is_expired is False

    def test_to_dict_round_trip(self):
        entry = CacheEntry(
            query="test query", response={"content": "answer"},
            embedding=[0.1, 0.2], created_at=1234567890.0, ttl=3600,
            metadata={"model": "gpt-4"},
        )
        d = entry.to_dict()
        restored = CacheEntry.from_dict(d)
        assert restored.query == entry.query
        assert restored.response == entry.response
        assert restored.ttl == entry.ttl
        assert restored.metadata == entry.metadata
