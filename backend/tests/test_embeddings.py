"""Tests for OpenAI Embedding adapter."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.rag.embeddings.base import EmbeddingResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_openai_embedding_response(embeddings, total_tokens=20):
    """Build a mock openai.embeddings.create() response."""
    resp = MagicMock()
    resp.data = [MagicMock(embedding=e) for e in embeddings]
    resp.usage = MagicMock(total_tokens=total_tokens)
    return resp


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestOpenAIEmbeddingInit:
    def test_known_model_small_succeeds(self):
        from core.rag.embeddings.openai_embedding import OpenAIEmbedding
        emb = OpenAIEmbedding(api_key="key", model="text-embedding-3-small")
        assert emb.model_name == "text-embedding-3-small"
        assert emb.dimensions == 1536

    def test_known_model_large_succeeds(self):
        from core.rag.embeddings.openai_embedding import OpenAIEmbedding
        emb = OpenAIEmbedding(api_key="key", model="text-embedding-3-large")
        assert emb.dimensions == 3072

    def test_ada_002_succeeds(self):
        from core.rag.embeddings.openai_embedding import OpenAIEmbedding
        emb = OpenAIEmbedding(api_key="key", model="text-embedding-ada-002")
        assert emb.dimensions == 1536

    def test_unknown_model_raises_value_error(self):
        from core.rag.embeddings.openai_embedding import OpenAIEmbedding
        with pytest.raises(ValueError, match="Unknown model"):
            OpenAIEmbedding(api_key="key", model="text-embedding-unknown-9000")

    def test_default_model_is_small(self):
        from core.rag.embeddings.openai_embedding import OpenAIEmbedding
        emb = OpenAIEmbedding(api_key="key")
        assert emb.model_name == "text-embedding-3-small"


# ---------------------------------------------------------------------------
# embed_documents
# ---------------------------------------------------------------------------

class TestEmbedDocuments:
    @pytest.fixture
    def embedding(self):
        from core.rag.embeddings.openai_embedding import OpenAIEmbedding
        return OpenAIEmbedding(api_key="test-key", model="text-embedding-3-small")

    @pytest.mark.asyncio
    async def test_returns_embedding_result(self, embedding):
        vectors = [[0.1] * 1536, [0.2] * 1536]
        mock_resp = make_openai_embedding_response(vectors, total_tokens=10)
        with patch.object(embedding.client.embeddings, "create",
                          new=AsyncMock(return_value=mock_resp)):
            result = await embedding.embed_documents(["doc1", "doc2"])
        assert isinstance(result, EmbeddingResult)

    @pytest.mark.asyncio
    async def test_correct_number_of_embeddings(self, embedding):
        vectors = [[0.1] * 1536, [0.2] * 1536, [0.3] * 1536]
        mock_resp = make_openai_embedding_response(vectors)
        with patch.object(embedding.client.embeddings, "create",
                          new=AsyncMock(return_value=mock_resp)):
            result = await embedding.embed_documents(["a", "b", "c"])
        assert len(result.embeddings) == 3

    @pytest.mark.asyncio
    async def test_dimensions_match_model(self, embedding):
        vectors = [[0.1] * 1536]
        mock_resp = make_openai_embedding_response(vectors)
        with patch.object(embedding.client.embeddings, "create",
                          new=AsyncMock(return_value=mock_resp)):
            result = await embedding.embed_documents(["text"])
        assert result.dimensions == 1536

    @pytest.mark.asyncio
    async def test_model_name_in_result(self, embedding):
        mock_resp = make_openai_embedding_response([[0.1] * 1536])
        with patch.object(embedding.client.embeddings, "create",
                          new=AsyncMock(return_value=mock_resp)):
            result = await embedding.embed_documents(["text"])
        assert result.model == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_total_tokens_returned(self, embedding):
        mock_resp = make_openai_embedding_response([[0.1] * 1536], total_tokens=42)
        with patch.object(embedding.client.embeddings, "create",
                          new=AsyncMock(return_value=mock_resp)):
            result = await embedding.embed_documents(["text"])
        assert result.total_tokens == 42

    @pytest.mark.asyncio
    async def test_latency_ms_non_negative(self, embedding):
        mock_resp = make_openai_embedding_response([[0.1] * 1536])
        with patch.object(embedding.client.embeddings, "create",
                          new=AsyncMock(return_value=mock_resp)):
            result = await embedding.embed_documents(["text"])
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_api_called_with_correct_model(self, embedding):
        mock_resp = make_openai_embedding_response([[0.1] * 1536])
        captured = {}

        async def mock_create(**kwargs):
            captured.update(kwargs)
            return mock_resp

        with patch.object(embedding.client.embeddings, "create", new=mock_create):
            await embedding.embed_documents(["hello world"])

        assert captured["model"] == "text-embedding-3-small"
        assert captured["input"] == ["hello world"]

    @pytest.mark.asyncio
    async def test_exception_propagates(self, embedding):
        with patch.object(embedding.client.embeddings, "create",
                          new=AsyncMock(side_effect=RuntimeError("API timeout"))):
            with pytest.raises(RuntimeError, match="API timeout"):
                await embedding.embed_documents(["text"])


# ---------------------------------------------------------------------------
# embed_query (delegates to embed_documents)
# ---------------------------------------------------------------------------

class TestEmbedQuery:
    @pytest.fixture
    def embedding(self):
        from core.rag.embeddings.openai_embedding import OpenAIEmbedding
        return OpenAIEmbedding(api_key="test-key", model="text-embedding-3-small")

    @pytest.mark.asyncio
    async def test_returns_embedding_result(self, embedding):
        mock_resp = make_openai_embedding_response([[0.5] * 1536])
        with patch.object(embedding.client.embeddings, "create",
                          new=AsyncMock(return_value=mock_resp)):
            result = await embedding.embed_query("search query")
        assert isinstance(result, EmbeddingResult)

    @pytest.mark.asyncio
    async def test_called_with_single_item_list(self, embedding):
        """embed_query wraps the text in a list and calls embed_documents."""
        mock_resp = make_openai_embedding_response([[0.1] * 1536])
        captured = {}

        async def mock_create(**kwargs):
            captured.update(kwargs)
            return mock_resp

        with patch.object(embedding.client.embeddings, "create", new=mock_create):
            await embedding.embed_query("my query")

        assert captured["input"] == ["my query"]

    @pytest.mark.asyncio
    async def test_result_has_one_embedding(self, embedding):
        mock_resp = make_openai_embedding_response([[0.1] * 1536])
        with patch.object(embedding.client.embeddings, "create",
                          new=AsyncMock(return_value=mock_resp)):
            result = await embedding.embed_query("query")
        assert len(result.embeddings) == 1


# ---------------------------------------------------------------------------
# Large embedding model (3072 dimensions)
# ---------------------------------------------------------------------------

class TestLargeEmbeddingModel:
    @pytest.fixture
    def large_embedding(self):
        from core.rag.embeddings.openai_embedding import OpenAIEmbedding
        return OpenAIEmbedding(api_key="key", model="text-embedding-3-large")

    @pytest.mark.asyncio
    async def test_dimensions_are_3072(self, large_embedding):
        mock_resp = make_openai_embedding_response([[0.1] * 3072])
        with patch.object(large_embedding.client.embeddings, "create",
                          new=AsyncMock(return_value=mock_resp)):
            result = await large_embedding.embed_documents(["text"])
        assert result.dimensions == 3072

    def test_model_config_has_correct_dims(self):
        from core.rag.embeddings.openai_embedding import OpenAIEmbedding
        assert OpenAIEmbedding.MODEL_CONFIG["text-embedding-3-large"]["dimensions"] == 3072
        assert OpenAIEmbedding.MODEL_CONFIG["text-embedding-3-small"]["dimensions"] == 1536
