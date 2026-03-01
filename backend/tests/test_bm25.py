"""Tests for BM25 keyword search retriever."""
import pytest
from core.rag.search.bm25 import BM25Retriever, BM25Config


@pytest.fixture
def retriever():
    return BM25Retriever()


@pytest.fixture
def retriever_no_stopwords():
    config = BM25Config(k1=1.2, b=0.5, stopwords=set())
    return BM25Retriever(config=config)


SAMPLE_DOCS = [
    {"id": "1", "content": "Python is a programming language", "metadata": {"lang": "python"}},
    {"id": "2", "content": "Java is also a programming language used for enterprise", "metadata": {"lang": "java"}},
    {"id": "3", "content": "Machine learning with Python and TensorFlow", "metadata": {"lang": "python"}},
    {"id": "4", "content": "Database management with SQL queries", "metadata": {"lang": "sql"}},
    {"id": "5", "content": "Web development with JavaScript and React", "metadata": {"lang": "javascript"}},
]


class TestTokenization:
    def test_lowercase(self, retriever):
        tokens = retriever._tokenize("Hello World FOO")
        assert all(t == t.lower() for t in tokens)
        assert "hello" in tokens
        assert "world" in tokens

    def test_removes_punctuation(self, retriever):
        tokens = retriever._tokenize("Hello, world! How are you?")
        for t in tokens:
            assert "," not in t and "!" not in t and "?" not in t

    def test_filters_stopwords(self, retriever):
        tokens = retriever._tokenize("the quick brown fox")
        assert "the" not in tokens
        assert "brown" in tokens
        assert "quick" in tokens

    def test_min_token_length(self, retriever):
        tokens = retriever._tokenize("a bb ccc dddd")
        assert "a" not in tokens  # length 1, filtered
        assert "bb" in tokens     # length 2, kept

    def test_empty_text_returns_empty(self, retriever):
        assert retriever._tokenize("") == []

    def test_all_stopwords_returns_empty(self, retriever):
        assert retriever._tokenize("the and or but") == []

    def test_no_stopwords_config(self, retriever_no_stopwords):
        tokens = retriever_no_stopwords._tokenize("the fox")
        assert "the" in tokens


class TestIndexing:
    @pytest.mark.asyncio
    async def test_index_single_doc(self, retriever):
        result = await retriever.index_documents([SAMPLE_DOCS[0]])
        assert result["indexed_count"] == 1
        assert result["total_documents"] == 1
        assert result["vocabulary_size"] > 0

    @pytest.mark.asyncio
    async def test_index_multiple_docs(self, retriever):
        result = await retriever.index_documents(SAMPLE_DOCS)
        assert result["indexed_count"] == len(SAMPLE_DOCS)
        assert result["total_documents"] == len(SAMPLE_DOCS)

    @pytest.mark.asyncio
    async def test_skip_doc_missing_id(self, retriever):
        result = await retriever.index_documents([{"content": "no id here"}])
        assert result["skipped_count"] == 1
        assert result["indexed_count"] == 0

    @pytest.mark.asyncio
    async def test_skip_doc_empty_content(self, retriever):
        result = await retriever.index_documents([{"id": "x", "content": ""}])
        assert result["skipped_count"] == 1

    @pytest.mark.asyncio
    async def test_update_existing_doc_keeps_total(self, retriever):
        await retriever.index_documents([{"id": "1", "content": "Python language"}])
        await retriever.index_documents([{"id": "1", "content": "Java language updated"}])
        assert retriever._total_docs == 1

    @pytest.mark.asyncio
    async def test_avg_doc_length_updated_after_indexing(self, retriever):
        await retriever.index_documents(SAMPLE_DOCS)
        assert retriever._avg_doc_length > 0


class TestSearch:
    @pytest.mark.asyncio
    async def test_basic_search_returns_results(self, retriever):
        await retriever.index_documents(SAMPLE_DOCS)
        results = await retriever.search("python programming")
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_relevant_doc_ranked_first(self, retriever):
        await retriever.index_documents(SAMPLE_DOCS)
        results = await retriever.search("python")
        top_ids = [r.id for r in results[:2]]
        assert "1" in top_ids or "3" in top_ids

    @pytest.mark.asyncio
    async def test_scores_descending(self, retriever):
        await retriever.index_documents(SAMPLE_DOCS)
        results = await retriever.search("programming language")
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    @pytest.mark.asyncio
    async def test_top_k_limit(self, retriever):
        await retriever.index_documents(SAMPLE_DOCS)
        results = await retriever.search("language", top_k=2)
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_unknown_term_returns_empty(self, retriever):
        await retriever.index_documents(SAMPLE_DOCS)
        results = await retriever.search("xyzzyqwerty12345")
        assert results == []

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self, retriever):
        await retriever.index_documents(SAMPLE_DOCS)
        results = await retriever.search("")
        assert results == []

    @pytest.mark.asyncio
    async def test_stopword_only_query_returns_empty(self, retriever):
        await retriever.index_documents(SAMPLE_DOCS)
        results = await retriever.search("the and or")
        assert results == []

    @pytest.mark.asyncio
    async def test_metadata_filter_restricts_results(self, retriever):
        await retriever.index_documents(SAMPLE_DOCS)
        results = await retriever.search("language", filters={"lang": "python"})
        for r in results:
            assert r.metadata.get("lang") == "python"

    @pytest.mark.asyncio
    async def test_empty_index_returns_empty(self, retriever):
        results = await retriever.search("python")
        assert results == []

    @pytest.mark.asyncio
    async def test_bm25_score_field_populated(self, retriever):
        await retriever.index_documents(SAMPLE_DOCS)
        results = await retriever.search("python")
        for r in results:
            assert r.bm25_score is not None
            assert r.bm25_score > 0


class TestRemoveDocument:
    @pytest.mark.asyncio
    async def test_remove_decrements_total(self, retriever):
        await retriever.index_documents(SAMPLE_DOCS)
        retriever.remove_document("1")
        assert retriever._total_docs == len(SAMPLE_DOCS) - 1

    @pytest.mark.asyncio
    async def test_remove_nonexistent_returns_false(self, retriever):
        assert retriever.remove_document("doesnotexist") is False

    @pytest.mark.asyncio
    async def test_removed_doc_absent_from_search(self, retriever):
        await retriever.index_documents(SAMPLE_DOCS)
        retriever.remove_document("1")
        results = await retriever.search("python")
        ids = [r.id for r in results]
        assert "1" not in ids

    @pytest.mark.asyncio
    async def test_remove_cleans_inverted_index(self, retriever):
        await retriever.index_documents([{"id": "solo", "content": "uniquewordxyz"}])
        retriever.remove_document("solo")
        assert "uniquewordxyz" not in retriever._inverted_index


class TestClear:
    @pytest.mark.asyncio
    async def test_clear_resets_all_state(self, retriever):
        await retriever.index_documents(SAMPLE_DOCS)
        retriever.clear()
        assert retriever._total_docs == 0
        assert retriever._avg_doc_length == 0.0
        assert len(retriever._documents) == 0
        assert len(retriever._inverted_index) == 0

    @pytest.mark.asyncio
    async def test_search_empty_after_clear(self, retriever):
        await retriever.index_documents(SAMPLE_DOCS)
        retriever.clear()
        results = await retriever.search("python")
        assert results == []


class TestGetStats:
    def test_stats_empty_index(self, retriever):
        stats = retriever.get_stats()
        assert stats["total_documents"] == 0
        assert stats["vocabulary_size"] == 0

    @pytest.mark.asyncio
    async def test_stats_after_indexing(self, retriever):
        await retriever.index_documents(SAMPLE_DOCS)
        stats = retriever.get_stats()
        assert stats["total_documents"] == len(SAMPLE_DOCS)
        assert stats["vocabulary_size"] > 0
        assert stats["avg_document_length"] > 0
        assert stats["config"]["k1"] == retriever.config.k1
