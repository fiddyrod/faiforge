"""Comprehensive tests for all FAIForge API endpoints.

Uses mocked registry and adapters so no real API calls are made.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock, patch

from core.api.server import create_app
from core.config import (
    AppConfig, APIConfig, CORSConfig, DefaultsConfig,
    ModelsConfig, ObservabilityConfig, CacheConfig, RateLimitConfig, RAGConfig
)
from core.inference.registry import ModelRegistry
from core.inference.adapters.base import (
    Response, StreamChunk, ToolCall, FunctionCallResult
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_config():
    return AppConfig(
        api=APIConfig(host="0.0.0.0", port=8000, reload=False, workers=1),
        cors=CORSConfig(
            enabled=False,
            origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"]
        ),
        defaults=DefaultsConfig(model="gpt-4o-mini", temperature=0.7, max_tokens=500),
        models=ModelsConfig(config_path="core/config/models.yaml", load_vllm=False),
        observability=ObservabilityConfig(log_level="INFO", log_format="json",
                                          request_logging=True),
        cache=CacheConfig(enabled=False, backend="memory", ttl_seconds=3600),
        rate_limit=RateLimitConfig(enabled=False, requests_per_minute=60),
        rag=RAGConfig()
    )


def _make_mock_adapter(content="Mock response"):
    adapter = MagicMock()
    adapter.complete = AsyncMock(return_value=Response(
        content=content,
        model="gpt-4o-mini",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.000015,
        latency_ms=50.0,
        finish_reason="stop",
        tool_calls=None,
    ))

    async def _stream_gen(*args, **kwargs):
        yield StreamChunk(content="Hello")
        yield StreamChunk(finish_reason="stop", input_tokens=5, output_tokens=3)

    adapter.complete_stream = MagicMock(side_effect=_stream_gen)
    return adapter


def _make_registry(adapter=None):
    registry = ModelRegistry()
    adapter = adapter or _make_mock_adapter()
    registry.register("gpt-4o-mini", adapter)
    return registry, adapter


@pytest.fixture
def client_and_adapter():
    """TestClient with mocked registry (no real API calls).

    We pass openai_api_key="" so the startup event skips RAG initialization
    (which requires a valid key to embed documents).
    """
    config = _make_config()
    registry, adapter = _make_registry()

    with patch("core.api.server.load_registry", return_value=registry):
        # Empty string is falsy → startup skips RAG init
        app = create_app(config, "",
                         routing_config_path="/nonexistent/routing.yaml")
        with TestClient(app) as c:
            yield c, adapter


@pytest.fixture
def client(client_and_adapter):
    c, _ = client_and_adapter
    return c


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

class TestRootEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_response_has_name(self, client):
        data = client.get("/").json()
        assert data["name"] == "FAIForge API"

    def test_response_has_status_ok(self, client):
        data = client.get("/").json()
        assert data["status"] == "ok"

    def test_response_has_features(self, client):
        data = client.get("/").json()
        assert "features" in data
        assert isinstance(data["features"], list)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_status_is_healthy(self, client):
        data = client.get("/health").json()
        assert data["status"] == "healthy"

    def test_models_loaded_present(self, client):
        data = client.get("/health").json()
        assert "models_loaded" in data
        assert data["models_loaded"] >= 1


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------

class TestListModelsEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200

    def test_response_has_models_field(self, client):
        data = client.get("/v1/models").json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_response_has_fallback_chains(self, client):
        data = client.get("/v1/models").json()
        assert "fallback_chains" in data

    def test_response_has_all_field(self, client):
        data = client.get("/v1/models").json()
        assert "all" in data
        assert "gpt-4o-mini" in data["all"]


# ---------------------------------------------------------------------------
# GET /v1/health/providers
# ---------------------------------------------------------------------------

class TestProviderHealthEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/v1/health/providers")
        assert resp.status_code == 200

    def test_response_has_status_ok(self, client):
        data = client.get("/v1/health/providers").json()
        assert data["status"] == "ok"

    def test_providers_field_present(self, client):
        data = client.get("/v1/health/providers").json()
        assert "providers" in data


# ---------------------------------------------------------------------------
# GET /v1/cache/stats  (cache disabled)
# ---------------------------------------------------------------------------

class TestCacheStatsEndpoint:
    def test_returns_200_when_cache_disabled(self, client):
        resp = client.get("/v1/cache/stats")
        assert resp.status_code == 200

    def test_enabled_is_false_when_cache_disabled(self, client):
        data = client.get("/v1/cache/stats").json()
        assert data["enabled"] is False


# ---------------------------------------------------------------------------
# POST /v1/cache/clear  (cache disabled)
# ---------------------------------------------------------------------------

class TestCacheClearEndpoint:
    def test_returns_400_when_cache_disabled(self, client):
        resp = client.post("/v1/cache/clear")
        assert resp.status_code == 400

    def test_error_message_mentions_cache(self, client):
        data = client.post("/v1/cache/clear").json()
        assert "cache" in data["detail"].lower()


# ---------------------------------------------------------------------------
# POST /v1/chat/completions  — validation
# ---------------------------------------------------------------------------

class TestChatCompletionValidation:
    def test_empty_messages_returns_422(self, client):
        resp = client.post("/v1/chat/completions",
                           json={"messages": [], "model": "gpt-4o-mini"})
        assert resp.status_code == 422

    def test_invalid_role_returns_422(self, client):
        resp = client.post("/v1/chat/completions",
                           json={"messages": [{"role": "robot", "content": "hi"}],
                                 "model": "gpt-4o-mini"})
        assert resp.status_code == 422

    def test_temperature_too_high_returns_422(self, client):
        resp = client.post("/v1/chat/completions",
                           json={"messages": [{"role": "user", "content": "hi"}],
                                 "model": "gpt-4o-mini",
                                 "temperature": 3.0})
        assert resp.status_code == 422

    def test_temperature_too_low_returns_422(self, client):
        resp = client.post("/v1/chat/completions",
                           json={"messages": [{"role": "user", "content": "hi"}],
                                 "model": "gpt-4o-mini",
                                 "temperature": -0.1})
        assert resp.status_code == 422

    def test_max_tokens_zero_returns_422(self, client):
        resp = client.post("/v1/chat/completions",
                           json={"messages": [{"role": "user", "content": "hi"}],
                                 "model": "gpt-4o-mini",
                                 "max_tokens": 0})
        assert resp.status_code == 422

    def test_max_tokens_too_large_returns_422(self, client):
        resp = client.post("/v1/chat/completions",
                           json={"messages": [{"role": "user", "content": "hi"}],
                                 "model": "gpt-4o-mini",
                                 "max_tokens": 99999})
        assert resp.status_code == 422

    def test_invalid_model_returns_400(self, client):
        resp = client.post("/v1/chat/completions",
                           json={"messages": [{"role": "user", "content": "hi"}],
                                 "model": "gpt-999-nonexistent"})
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"].lower()

    def test_message_too_long_returns_422(self, client):
        long_content = "x" * 100_001
        resp = client.post("/v1/chat/completions",
                           json={"messages": [{"role": "user", "content": long_content}],
                                 "model": "gpt-4o-mini"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /v1/chat/completions  — successful completion
# ---------------------------------------------------------------------------

class TestChatCompletionSuccess:
    def test_returns_200(self, client_and_adapter):
        client, _ = client_and_adapter
        resp = client.post("/v1/chat/completions",
                           json={"messages": [{"role": "user", "content": "Hi"}],
                                 "model": "gpt-4o-mini"})
        assert resp.status_code == 200

    def test_response_has_content(self, client_and_adapter):
        client, _ = client_and_adapter
        data = client.post("/v1/chat/completions",
                           json={"messages": [{"role": "user", "content": "Hi"}],
                                 "model": "gpt-4o-mini"}).json()
        assert data["content"] == "Mock response"

    def test_response_has_usage(self, client_and_adapter):
        client, _ = client_and_adapter
        data = client.post("/v1/chat/completions",
                           json={"messages": [{"role": "user", "content": "Hi"}],
                                 "model": "gpt-4o-mini"}).json()
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] == 10
        assert data["usage"]["completion_tokens"] == 5

    def test_response_has_cost(self, client_and_adapter):
        client, _ = client_and_adapter
        data = client.post("/v1/chat/completions",
                           json={"messages": [{"role": "user", "content": "Hi"}],
                                 "model": "gpt-4o-mini"}).json()
        assert "cost_usd" in data
        assert data["cost_usd"] >= 0

    def test_response_has_finish_reason(self, client_and_adapter):
        client, _ = client_and_adapter
        data = client.post("/v1/chat/completions",
                           json={"messages": [{"role": "user", "content": "Hi"}],
                                 "model": "gpt-4o-mini"}).json()
        assert data["finish_reason"] == "stop"

    def test_adapter_complete_called(self, client_and_adapter):
        client, adapter = client_and_adapter
        adapter.complete.reset_mock()
        client.post("/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "Hi"}],
                          "model": "gpt-4o-mini"})
        adapter.complete.assert_called_once()

    def test_system_message_in_request(self, client_and_adapter):
        client, adapter = client_and_adapter
        adapter.complete.reset_mock()
        client.post("/v1/chat/completions",
                    json={"messages": [
                              {"role": "system", "content": "You are helpful"},
                              {"role": "user", "content": "Hi"},
                          ],
                          "model": "gpt-4o-mini"})
        adapter.complete.assert_called_once()

    def test_adapter_error_returns_500(self, client_and_adapter):
        client, adapter = client_and_adapter
        adapter.complete = AsyncMock(side_effect=RuntimeError("upstream failure"))
        resp = client.post("/v1/chat/completions",
                           json={"messages": [{"role": "user", "content": "Hi"}],
                                 "model": "gpt-4o-mini"})
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# POST /v1/chat/completions  — streaming
# ---------------------------------------------------------------------------

class TestChatCompletionStreaming:
    def test_streaming_returns_200(self, client_and_adapter):
        client, _ = client_and_adapter
        resp = client.post("/v1/chat/completions",
                           params={"stream": "true"},
                           json={"messages": [{"role": "user", "content": "Hi"}],
                                 "model": "gpt-4o-mini"})
        assert resp.status_code == 200

    def test_streaming_content_type_is_event_stream(self, client_and_adapter):
        client, _ = client_and_adapter
        resp = client.post("/v1/chat/completions",
                           params={"stream": "true"},
                           json={"messages": [{"role": "user", "content": "Hi"}],
                                 "model": "gpt-4o-mini"})
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_streaming_response_contains_done(self, client_and_adapter):
        client, _ = client_and_adapter
        resp = client.post("/v1/chat/completions",
                           params={"stream": "true"},
                           json={"messages": [{"role": "user", "content": "Hi"}],
                                 "model": "gpt-4o-mini"})
        assert "[DONE]" in resp.text


# ---------------------------------------------------------------------------
# Tool calls in response
# ---------------------------------------------------------------------------

class TestChatCompletionWithToolCalls:
    def test_tool_calls_in_response(self, client_and_adapter):
        client, adapter = client_and_adapter
        adapter.complete = AsyncMock(return_value=Response(
            content=None,
            model="gpt-4o-mini",
            input_tokens=20,
            output_tokens=10,
            cost_usd=0.0,
            latency_ms=50.0,
            finish_reason="tool_calls",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    type="function",
                    function=FunctionCallResult(name="get_weather",
                                               arguments='{"city":"NYC"}')
                )
            ],
        ))
        data = client.post("/v1/chat/completions",
                           json={"messages": [{"role": "user", "content": "Weather?"}],
                                 "model": "gpt-4o-mini"}).json()
        assert data["finish_reason"] == "tool_calls"
        assert data["tool_calls"] is not None
        assert data["tool_calls"][0]["function"]["name"] == "get_weather"


# ---------------------------------------------------------------------------
# GET /v1/rag/stats  (RAG not initialized in test env)
# ---------------------------------------------------------------------------

class TestRAGStatsEndpoint:
    def test_returns_200(self, client):
        resp = client.get("/v1/rag/stats")
        assert resp.status_code == 200

    def test_enabled_is_false_without_pipeline(self, client):
        data = client.get("/v1/rag/stats").json()
        assert data["enabled"] is False


# ---------------------------------------------------------------------------
# POST /v1/rag/ingest  (RAG not initialized)
# ---------------------------------------------------------------------------

class TestRAGIngestEndpoint:
    def test_returns_503_without_pipeline(self, client):
        resp = client.post("/v1/rag/ingest",
                           json={"documents": [
                               {"content": "Some document text here.", "metadata": {}}
                           ]})
        assert resp.status_code == 503

    def test_error_mentions_rag(self, client):
        data = client.post("/v1/rag/ingest",
                           json={"documents": [
                               {"content": "Text.", "metadata": {}}
                           ]}).json()
        assert "rag" in data["detail"].lower() or "pipeline" in data["detail"].lower()

    def test_validation_rejects_empty_content(self, client):
        resp = client.post("/v1/rag/ingest",
                           json={"documents": [{"content": "", "metadata": {}}]})
        assert resp.status_code == 422

    def test_validation_rejects_empty_documents_list(self, client):
        resp = client.post("/v1/rag/ingest",
                           json={"documents": []})
        assert resp.status_code in (422, 503)


# ---------------------------------------------------------------------------
# POST /v1/rag/query  (RAG not initialized)
# ---------------------------------------------------------------------------

class TestRAGQueryEndpoint:
    def test_returns_503_without_pipeline(self, client):
        resp = client.post("/v1/rag/query",
                           json={"query": "What is Python?"})
        assert resp.status_code == 503

    def test_validation_invalid_search_mode(self, client):
        resp = client.post("/v1/rag/query",
                           json={"query": "hi", "search_mode": "turbo"})
        assert resp.status_code == 422

    def test_validation_empty_query(self, client):
        resp = client.post("/v1/rag/query", json={"query": ""})
        assert resp.status_code == 422

    def test_valid_search_modes_accepted(self, client):
        for mode in ("semantic", "bm25", "hybrid"):
            resp = client.post("/v1/rag/query",
                               json={"query": "test query", "search_mode": mode})
            # 503 because RAG not available, not 422
            assert resp.status_code == 503

    def test_top_k_out_of_range_rejected(self, client):
        resp = client.post("/v1/rag/query",
                           json={"query": "test", "top_k": 0})
        assert resp.status_code == 422

    def test_top_k_too_large_rejected(self, client):
        resp = client.post("/v1/rag/query",
                           json={"query": "test", "top_k": 100})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Request conversion helpers (unit tests — no HTTP)
# ---------------------------------------------------------------------------

class TestConvertRequestMessages:
    def test_converts_simple_user_message(self):
        from core.api.server import convert_request_messages
        from core.api.server import ChatMessage

        msgs = [ChatMessage(role="user", content="Hello")]
        result = convert_request_messages(msgs)
        assert result[0].role == "user"
        assert result[0].content == "Hello"

    def test_converts_tool_call_in_message(self):
        from core.api.server import convert_request_messages
        from core.api.server import ChatMessage

        tool_call_data = [{
            "id": "call-1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"NYC"}'}
        }]
        msgs = [ChatMessage(role="assistant", content=None, tool_calls=tool_call_data)]
        result = convert_request_messages(msgs)
        assert result[0].tool_calls is not None
        assert result[0].tool_calls[0].id == "call-1"


class TestConvertRequestTools:
    def test_converts_tool(self):
        from core.api.server import convert_request_tools
        from core.api.server import ToolRequest, FunctionDefRequest

        tools = [ToolRequest(
            type="function",
            function=FunctionDefRequest(
                name="fn", description="desc",
                parameters={"type": "object"}
            )
        )]
        result = convert_request_tools(tools)
        assert result[0].function.name == "fn"


class TestFormatToolCallsResponse:
    def test_none_returns_none(self):
        from core.api.server import format_tool_calls_response
        assert format_tool_calls_response(None) is None

    def test_empty_list_returns_none(self):
        from core.api.server import format_tool_calls_response
        assert format_tool_calls_response([]) is None

    def test_tool_calls_converted(self):
        from core.api.server import format_tool_calls_response
        tc = ToolCall(
            id="call-1",
            type="function",
            function=FunctionCallResult(name="fn", arguments='{"k":"v"}')
        )
        result = format_tool_calls_response([tc])
        assert result is not None
        assert result[0].id == "call-1"
        assert result[0].function["name"] == "fn"
