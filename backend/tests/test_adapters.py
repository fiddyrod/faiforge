"""Tests for LLM adapters (Ollama, OpenAI, Anthropic) using mocked HTTP clients."""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.inference.adapters.base import (
    Message,
    Response,
    StreamChunk,
    Tool,
    FunctionDef,
    ResponseFormat,
    ToolCall,
    FunctionCallResult,
)

MESSAGES = [Message(role="user", content="Hello!")]


# ---------------------------------------------------------------------------
# Helpers shared across adapters
# ---------------------------------------------------------------------------

def make_openai_complete_response(content="Test response", finish_reason="stop",
                                   prompt_tokens=10, completion_tokens=5,
                                   tool_calls=None):
    """Build a mock OpenAI completions.create() response."""
    resp = MagicMock()
    resp.choices = [MagicMock(
        message=MagicMock(content=content, tool_calls=tool_calls),
        finish_reason=finish_reason,
    )]
    resp.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return resp


def make_openai_stream_chunk(content=None, finish_reason=None,
                              prompt_tokens=None, completion_tokens=None,
                              tool_calls=None):
    """Build a mock SSE chunk from OpenAI streaming API."""
    chunk = MagicMock()
    if prompt_tokens is not None:
        chunk.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    else:
        chunk.usage = None
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk.choices = [choice]
    return chunk


class MockAnthropicStream:
    """Async context manager that yields pre-baked Anthropic stream events."""
    def __init__(self, events):
        self._events = events

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for e in self._events:
            yield e


def make_anthropic_event(event_type, **attrs):
    """Create a MagicMock Anthropic stream event with `.type` set."""
    e = MagicMock()
    e.type = event_type
    for k, v in attrs.items():
        setattr(e, k, v)
    return e


# ---------------------------------------------------------------------------
# OllamaAdapter
# ---------------------------------------------------------------------------

class TestOllamaAdapterInit:
    def test_default_base_url(self):
        from core.inference.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter(model="phi3")
        assert adapter.base_url == "http://localhost:11434/v1"
        assert adapter.model == "phi3"

    def test_custom_base_url(self):
        from core.inference.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter(model="llama3", base_url="http://remote:11434/v1")
        assert adapter.base_url == "http://remote:11434/v1"

    def test_model_name_stored(self):
        from core.inference.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter(model="mistral")
        assert adapter.model == "mistral"


class TestOllamaAdapterComplete:
    def _make_mock_response(self, content="Hello there!", tool_calls=None,
                            finish_reason="stop", prompt_tokens=10, completion_tokens=5):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(
            message=MagicMock(content=content, tool_calls=tool_calls),
            finish_reason=finish_reason,
        )]
        mock_resp.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        return mock_resp

    @pytest.mark.asyncio
    async def test_complete_returns_response_object(self):
        from core.inference.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter(model="phi3")
        mock_resp = self._make_mock_response(content="Hi!")
        with patch.object(adapter.client.chat.completions, "create", new=AsyncMock(return_value=mock_resp)):
            result = await adapter.complete(MESSAGES)
        assert isinstance(result, Response)
        assert result.content == "Hi!"

    @pytest.mark.asyncio
    async def test_cost_is_always_zero(self):
        from core.inference.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter(model="phi3")
        mock_resp = self._make_mock_response()
        with patch.object(adapter.client.chat.completions, "create", new=AsyncMock(return_value=mock_resp)):
            result = await adapter.complete(MESSAGES)
        assert result.cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_token_counts_from_usage(self):
        from core.inference.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter(model="phi3")
        mock_resp = self._make_mock_response(prompt_tokens=20, completion_tokens=8)
        with patch.object(adapter.client.chat.completions, "create", new=AsyncMock(return_value=mock_resp)):
            result = await adapter.complete(MESSAGES)
        assert result.input_tokens == 20
        assert result.output_tokens == 8

    @pytest.mark.asyncio
    async def test_json_format_passed_to_client(self):
        from core.inference.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter(model="phi3")
        mock_resp = self._make_mock_response(content='{"key": "value"}')
        captured = {}

        async def mock_create(**kwargs):
            captured.update(kwargs)
            return mock_resp

        with patch.object(adapter.client.chat.completions, "create", new=mock_create):
            await adapter.complete(MESSAGES, response_format=ResponseFormat(type="json_object"))

        assert captured.get("response_format") == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_exception_propagates(self):
        from core.inference.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter(model="phi3")
        with patch.object(
            adapter.client.chat.completions, "create",
            new=AsyncMock(side_effect=ConnectionError("Connection refused")),
        ):
            with pytest.raises(ConnectionError):
                await adapter.complete(MESSAGES)

    @pytest.mark.asyncio
    async def test_tool_calls_detected_in_finish_reason(self):
        from core.inference.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter(model="phi3")

        mock_tc = MagicMock()
        mock_tc.id = "call-1"
        mock_tc.type = "function"
        mock_tc.function = MagicMock(name="get_weather", arguments='{"city": "NYC"}')

        mock_resp = self._make_mock_response(content=None, tool_calls=[mock_tc], finish_reason="stop")
        with patch.object(adapter.client.chat.completions, "create", new=AsyncMock(return_value=mock_resp)):
            result = await adapter.complete(MESSAGES)
        assert result.finish_reason == "tool_calls"
        assert result.tool_calls is not None
        assert len(result.tool_calls) == 1

    @pytest.mark.asyncio
    async def test_no_usage_defaults_to_zero_tokens(self):
        from core.inference.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter(model="phi3")
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock(
            message=MagicMock(content="ok", tool_calls=None),
            finish_reason="stop",
        )]
        mock_resp.usage = None
        with patch.object(adapter.client.chat.completions, "create", new=AsyncMock(return_value=mock_resp)):
            result = await adapter.complete(MESSAGES)
        assert result.input_tokens == 0
        assert result.output_tokens == 0


class TestOllamaAdapterMessageConversion:
    def test_converts_user_message(self):
        from core.inference.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter(model="phi3")
        msgs = [Message(role="user", content="Hello")]
        converted = adapter._convert_messages(msgs)
        assert len(converted) == 1
        assert converted[0]["role"] == "user"
        assert converted[0]["content"] == "Hello"

    def test_converts_system_message(self):
        from core.inference.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter(model="phi3")
        msgs = [Message(role="system", content="Be helpful")]
        converted = adapter._convert_messages(msgs)
        assert converted[0]["role"] == "system"

    def test_converts_tool_response_message(self):
        from core.inference.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter(model="phi3")
        msgs = [Message(role="tool", content="result", tool_call_id="call-1")]
        converted = adapter._convert_messages(msgs)
        assert converted[0]["tool_call_id"] == "call-1"

    def test_converts_assistant_message_with_tool_calls(self):
        from core.inference.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter(model="phi3")
        tc = ToolCall(
            id="call-1",
            type="function",
            function=FunctionCallResult(name="get_weather", arguments='{"city":"NYC"}'),
        )
        msgs = [Message(role="assistant", content=None, tool_calls=[tc])]
        converted = adapter._convert_messages(msgs)
        assert "tool_calls" in converted[0]
        assert converted[0]["tool_calls"][0]["id"] == "call-1"

    def test_converts_multiple_messages_in_order(self):
        from core.inference.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter(model="phi3")
        msgs = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
        ]
        converted = adapter._convert_messages(msgs)
        assert len(converted) == 3
        assert [m["role"] for m in converted] == ["system", "user", "assistant"]


class TestOllamaAdapterToolConversion:
    def test_converts_single_tool(self):
        from core.inference.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter(model="phi3")
        tools = [Tool(
            type="function",
            function=FunctionDef(
                name="get_weather",
                description="Get weather for a city",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            ),
        )]
        converted = adapter._convert_tools(tools)
        assert len(converted) == 1
        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "get_weather"
        assert converted[0]["function"]["description"] == "Get weather for a city"

    def test_skips_tool_without_function_def(self):
        from core.inference.adapters.ollama_adapter import OllamaAdapter
        adapter = OllamaAdapter(model="phi3")
        tools = [Tool(type="function", function=None)]
        converted = adapter._convert_tools(tools)
        assert len(converted) == 0


class TestOllamaAdapterListModels:
    @pytest.mark.asyncio
    async def test_returns_model_names(self):
        from core.inference.adapters.ollama_adapter import OllamaAdapter
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": [{"name": "llama3"}, {"name": "phi3"}]}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            models = await OllamaAdapter.list_models()

        assert "llama3" in models
        assert "phi3" in models

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_connection_error(self):
        from core.inference.adapters.ollama_adapter import OllamaAdapter
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=ConnectionError("refused")
            )
            models = await OllamaAdapter.list_models()
        assert models == []

    @pytest.mark.asyncio
    async def test_custom_base_url_used(self):
        from core.inference.adapters.ollama_adapter import OllamaAdapter
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": []}
        mock_response.raise_for_status = MagicMock()
        captured_url = []

        async def mock_get(url, **kwargs):
            captured_url.append(url)
            return mock_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = mock_get
            await OllamaAdapter.list_models(base_url="http://remote:11434/v1")

        assert len(captured_url) == 1
        assert "remote" in captured_url[0]


# ===========================================================================
# OpenAIAdapter
# ===========================================================================

class TestOpenAIAdapterInit:
    def test_known_model_succeeds(self):
        from core.inference.adapters.openai_adapter import OpenAIAdapter
        adapter = OpenAIAdapter(api_key="test-key", model="gpt-4o-mini")
        assert adapter.model == "gpt-4o-mini"

    def test_unknown_model_raises_value_error(self):
        from core.inference.adapters.openai_adapter import OpenAIAdapter
        with pytest.raises(ValueError, match="Unknown model"):
            OpenAIAdapter(api_key="test-key", model="gpt-99-super")

    def test_all_known_models_accepted(self):
        from core.inference.adapters.openai_adapter import OpenAIAdapter
        for model in OpenAIAdapter.PRICING:
            adapter = OpenAIAdapter(api_key="k", model=model)
            assert adapter.model == model


class TestOpenAIMessageConversion:
    @pytest.fixture
    def adapter(self):
        from core.inference.adapters.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(api_key="test-key", model="gpt-4o-mini")

    def test_user_message(self, adapter):
        msgs = [Message(role="user", content="Hello")]
        result = adapter._convert_messages(msgs)
        assert result[0] == {"role": "user", "content": "Hello"}

    def test_system_message(self, adapter):
        msgs = [Message(role="system", content="Be helpful")]
        result = adapter._convert_messages(msgs)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Be helpful"

    def test_tool_response_message_includes_tool_call_id(self, adapter):
        msgs = [Message(role="tool", content="42°F", tool_call_id="call-abc")]
        result = adapter._convert_messages(msgs)
        assert result[0]["tool_call_id"] == "call-abc"
        assert result[0]["content"] == "42°F"

    def test_assistant_message_with_tool_calls(self, adapter):
        tc = ToolCall(
            id="call-1",
            type="function",
            function=FunctionCallResult(name="get_weather", arguments='{"city":"NYC"}'),
        )
        msgs = [Message(role="assistant", content=None, tool_calls=[tc])]
        result = adapter._convert_messages(msgs)
        assert "tool_calls" in result[0]
        assert result[0]["tool_calls"][0]["id"] == "call-1"
        assert result[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_message_without_content_excluded(self, adapter):
        """Content=None should not inject a 'content' key."""
        msgs = [Message(role="assistant", content=None)]
        result = adapter._convert_messages(msgs)
        assert "content" not in result[0]


class TestOpenAIToolConversion:
    @pytest.fixture
    def adapter(self):
        from core.inference.adapters.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(api_key="test-key", model="gpt-4o-mini")

    def test_converts_tool_to_openai_format(self, adapter):
        tools = [Tool(
            type="function",
            function=FunctionDef(name="fn", description="desc", parameters={}),
        )]
        result = adapter._convert_tools(tools)
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "fn"
        assert result[0]["function"]["description"] == "desc"

    def test_skips_tool_with_no_function(self, adapter):
        tools = [Tool(type="function", function=None)]
        assert adapter._convert_tools(tools) == []


class TestOpenAIResponseFormat:
    @pytest.fixture
    def adapter(self):
        from core.inference.adapters.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(api_key="test-key", model="gpt-4o-mini")

    def test_json_object_format(self, adapter):
        rf = ResponseFormat(type="json_object")
        result = adapter._convert_response_format(rf)
        assert result == {"type": "json_object"}

    def test_json_schema_format(self, adapter):
        schema = {"name": "response", "schema": {"type": "object"}}
        rf = ResponseFormat(type="json_schema", json_schema=schema)
        result = adapter._convert_response_format(rf)
        assert result["type"] == "json_schema"
        assert result["json_schema"] == schema

    def test_text_format_fallback(self, adapter):
        rf = ResponseFormat(type="text")
        result = adapter._convert_response_format(rf)
        assert result == {"type": "text"}


class TestOpenAICostCalculation:
    @pytest.fixture
    def adapter(self):
        from core.inference.adapters.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(api_key="test-key", model="gpt-4o-mini")

    def test_zero_tokens_zero_cost(self, adapter):
        assert adapter._calculate_cost(0, 0) == 0.0

    def test_cost_is_positive_for_nonzero_tokens(self, adapter):
        cost = adapter._calculate_cost(1000, 500)
        assert cost > 0

    def test_output_tokens_cost_more_than_input(self, adapter):
        input_cost = adapter._calculate_cost(1000, 0)
        output_cost = adapter._calculate_cost(0, 1000)
        assert output_cost > input_cost


class TestOpenAIComplete:
    @pytest.fixture
    def adapter(self):
        from core.inference.adapters.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(api_key="test-key", model="gpt-4o-mini")

    @pytest.mark.asyncio
    async def test_returns_response_with_content(self, adapter):
        mock_resp = make_openai_complete_response(content="Hello world!")
        with patch.object(adapter.client.chat.completions, "create",
                          new=AsyncMock(return_value=mock_resp)):
            result = await adapter.complete(MESSAGES)
        assert isinstance(result, Response)
        assert result.content == "Hello world!"

    @pytest.mark.asyncio
    async def test_token_counts_correct(self, adapter):
        mock_resp = make_openai_complete_response(prompt_tokens=50, completion_tokens=20)
        with patch.object(adapter.client.chat.completions, "create",
                          new=AsyncMock(return_value=mock_resp)):
            result = await adapter.complete(MESSAGES)
        assert result.input_tokens == 50
        assert result.output_tokens == 20

    @pytest.mark.asyncio
    async def test_cost_calculated_and_rounded(self, adapter):
        mock_resp = make_openai_complete_response(prompt_tokens=100, completion_tokens=50)
        with patch.object(adapter.client.chat.completions, "create",
                          new=AsyncMock(return_value=mock_resp)):
            result = await adapter.complete(MESSAGES)
        assert isinstance(result.cost_usd, float)
        assert result.cost_usd >= 0

    @pytest.mark.asyncio
    async def test_finish_reason_stop(self, adapter):
        mock_resp = make_openai_complete_response(finish_reason="stop")
        with patch.object(adapter.client.chat.completions, "create",
                          new=AsyncMock(return_value=mock_resp)):
            result = await adapter.complete(MESSAGES)
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_tool_calls_sets_finish_reason_to_tool_calls(self, adapter):
        tc_mock = MagicMock()
        tc_mock.id = "call-1"
        tc_mock.type = "function"
        tc_mock.function = MagicMock(name="fn", arguments='{}')
        mock_resp = make_openai_complete_response(
            content=None, tool_calls=[tc_mock], finish_reason="tool_calls"
        )
        with patch.object(adapter.client.chat.completions, "create",
                          new=AsyncMock(return_value=mock_resp)):
            result = await adapter.complete(MESSAGES)
        assert result.finish_reason == "tool_calls"
        assert result.tool_calls is not None

    @pytest.mark.asyncio
    async def test_exception_propagates(self, adapter):
        with patch.object(adapter.client.chat.completions, "create",
                          new=AsyncMock(side_effect=RuntimeError("API down"))):
            with pytest.raises(RuntimeError, match="API down"):
                await adapter.complete(MESSAGES)

    @pytest.mark.asyncio
    async def test_latency_ms_non_negative(self, adapter):
        mock_resp = make_openai_complete_response()
        with patch.object(adapter.client.chat.completions, "create",
                          new=AsyncMock(return_value=mock_resp)):
            result = await adapter.complete(MESSAGES)
        assert result.latency_ms >= 0


class TestOpenAICompleteStream:
    @pytest.fixture
    def adapter(self):
        from core.inference.adapters.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(api_key="test-key", model="gpt-4o-mini")

    @pytest.mark.asyncio
    async def test_yields_content_chunks_then_final_chunk(self, adapter):
        chunks_in = [
            make_openai_stream_chunk(content="Hello"),
            make_openai_stream_chunk(content=" world"),
            make_openai_stream_chunk(
                content=None, finish_reason="stop",
                prompt_tokens=10, completion_tokens=5
            ),
        ]

        async def mock_create(**kwargs):
            async def gen():
                for c in chunks_in:
                    yield c
            return gen()

        with patch.object(adapter.client.chat.completions, "create", new=mock_create):
            out = []
            async for chunk in adapter.complete_stream(MESSAGES):
                out.append(chunk)

        content_chunks = [c for c in out if c.content]
        assert len(content_chunks) >= 1
        # Final chunk has finish_reason and token counts
        final = out[-1]
        assert final.finish_reason is not None
        assert final.input_tokens == 10
        assert final.output_tokens == 5

    @pytest.mark.asyncio
    async def test_empty_stream_yields_final_chunk(self, adapter):
        async def mock_create(**kwargs):
            async def gen():
                if False:
                    yield
            return gen()

        with patch.object(adapter.client.chat.completions, "create", new=mock_create):
            out = []
            async for chunk in adapter.complete_stream(MESSAGES):
                out.append(chunk)

        assert len(out) == 1
        assert out[0].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_exception_propagates_in_stream(self, adapter):
        async def mock_create(**kwargs):
            raise RuntimeError("stream broken")

        with patch.object(adapter.client.chat.completions, "create", new=mock_create):
            with pytest.raises(RuntimeError, match="stream broken"):
                async for _ in adapter.complete_stream(MESSAGES):
                    pass


# ===========================================================================
# AnthropicAdapter
# ===========================================================================

class TestAnthropicAdapterInit:
    def test_known_model_stored(self):
        from core.inference.adapters.anthropic_adapter import AnthropicAdapter
        adapter = AnthropicAdapter(api_key="test-key",
                                   model="claude-sonnet-4-5-20250929")
        assert adapter.model == "claude-sonnet-4-5-20250929"

    def test_unknown_model_does_not_raise(self):
        """Anthropic adapter just warns on unknown models, doesn't raise."""
        from core.inference.adapters.anthropic_adapter import AnthropicAdapter
        adapter = AnthropicAdapter(api_key="test-key", model="claude-unknown-model")
        assert adapter.model == "claude-unknown-model"


class TestAnthropicMessageConversion:
    @pytest.fixture
    def adapter(self):
        from core.inference.adapters.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(api_key="test-key", model="claude-sonnet-4-5-20250929")

    def test_system_message_extracted(self, adapter):
        msgs = [
            Message(role="system", content="Be concise"),
            Message(role="user", content="Hello"),
        ]
        system, converted = adapter._convert_messages(msgs)
        assert system == "Be concise"
        assert len(converted) == 1
        assert converted[0]["role"] == "user"

    def test_no_system_returns_none(self, adapter):
        msgs = [Message(role="user", content="Hi")]
        system, converted = adapter._convert_messages(msgs)
        assert system is None

    def test_tool_result_wrapped_in_user_message(self, adapter):
        msgs = [Message(role="tool", content="result data", tool_call_id="call-1")]
        _, converted = adapter._convert_messages(msgs)
        assert converted[0]["role"] == "user"
        block = converted[0]["content"][0]
        assert block["type"] == "tool_result"
        assert block["tool_use_id"] == "call-1"
        assert block["content"] == "result data"

    def test_assistant_with_tool_calls_uses_tool_use_blocks(self, adapter):
        tc = ToolCall(
            id="call-1",
            type="function",
            function=FunctionCallResult(name="get_weather", arguments='{"city": "NYC"}'),
        )
        msgs = [Message(role="assistant", content="Checking...", tool_calls=[tc])]
        _, converted = adapter._convert_messages(msgs)
        assert converted[0]["role"] == "assistant"
        content_blocks = converted[0]["content"]
        types = [b["type"] for b in content_blocks]
        assert "tool_use" in types
        tool_block = next(b for b in content_blocks if b["type"] == "tool_use")
        assert tool_block["id"] == "call-1"
        assert tool_block["name"] == "get_weather"
        assert tool_block["input"] == {"city": "NYC"}

    def test_regular_user_message(self, adapter):
        msgs = [Message(role="user", content="Hello!")]
        _, converted = adapter._convert_messages(msgs)
        assert converted[0] == {"role": "user", "content": "Hello!"}


class TestAnthropicToolConversion:
    @pytest.fixture
    def adapter(self):
        from core.inference.adapters.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(api_key="test-key", model="claude-sonnet-4-5-20250929")

    def test_uses_input_schema_not_parameters(self, adapter):
        tools = [Tool(
            type="function",
            function=FunctionDef(name="fn", description="desc",
                                 parameters={"type": "object"}),
        )]
        result = adapter._convert_tools(tools)
        assert "input_schema" in result[0]
        assert "parameters" not in result[0]
        assert result[0]["name"] == "fn"

    def test_skips_tool_with_no_function(self, adapter):
        assert adapter._convert_tools([Tool(type="function", function=None)]) == []


class TestAnthropicToolChoiceConversion:
    @pytest.fixture
    def adapter(self):
        from core.inference.adapters.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(api_key="test-key", model="claude-sonnet-4-5-20250929")

    def test_auto(self, adapter):
        assert adapter._convert_tool_choice("auto") == {"type": "auto"}

    def test_none(self, adapter):
        assert adapter._convert_tool_choice("none") == {"type": "none"}

    def test_required_maps_to_any(self, adapter):
        assert adapter._convert_tool_choice("required") == {"type": "any"}

    def test_specific_tool_dict(self, adapter):
        tc = {"type": "function", "function": {"name": "my_fn"}}
        result = adapter._convert_tool_choice(tc)
        assert result == {"type": "tool", "name": "my_fn"}

    def test_unknown_falls_back_to_auto(self, adapter):
        assert adapter._convert_tool_choice("unknown_value") == {"type": "auto"}


class TestAnthropicParseToolCalls:
    @pytest.fixture
    def adapter(self):
        from core.inference.adapters.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(api_key="test-key", model="claude-sonnet-4-5-20250929")

    def _make_text_block(self, text):
        b = MagicMock()
        b.type = "text"
        b.text = text
        return b

    def _make_tool_use_block(self, id_, name, input_dict):
        b = MagicMock()
        b.type = "tool_use"
        b.id = id_
        b.name = name
        b.input = input_dict
        return b

    def test_text_block_returns_text_content(self, adapter):
        blocks = [self._make_text_block("Hello!")]
        text, tools = adapter._parse_tool_calls(blocks)
        assert text == "Hello!"
        assert tools is None

    def test_tool_use_block_returns_tool_call(self, adapter):
        blocks = [self._make_tool_use_block("call-1", "get_weather", {"city": "NYC"})]
        text, tools = adapter._parse_tool_calls(blocks)
        assert tools is not None
        assert len(tools) == 1
        assert tools[0].id == "call-1"
        assert tools[0].function.name == "get_weather"
        assert json.loads(tools[0].function.arguments) == {"city": "NYC"}

    def test_mixed_blocks(self, adapter):
        blocks = [
            self._make_text_block("Let me check"),
            self._make_tool_use_block("call-2", "get_price", {"item": "apple"}),
        ]
        text, tools = adapter._parse_tool_calls(blocks)
        assert text == "Let me check"
        assert tools is not None

    def test_no_blocks_returns_nones(self, adapter):
        text, tools = adapter._parse_tool_calls([])
        assert text is None
        assert tools is None


class TestAnthropicBuildSystemPrompt:
    @pytest.fixture
    def adapter(self):
        from core.inference.adapters.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(api_key="test-key", model="claude-sonnet-4-5-20250929")

    def test_no_system_no_format_returns_none(self, adapter):
        assert adapter._build_system_prompt(None, None) is None

    def test_base_system_only(self, adapter):
        result = adapter._build_system_prompt("Be helpful", None)
        assert result == "Be helpful"

    def test_json_object_appends_instruction(self, adapter):
        rf = ResponseFormat(type="json_object")
        result = adapter._build_system_prompt(None, rf)
        assert result is not None
        assert "JSON" in result

    def test_json_schema_includes_schema(self, adapter):
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        rf = ResponseFormat(type="json_schema", json_schema=schema)
        result = adapter._build_system_prompt("Base", rf)
        assert "Base" in result
        assert "schema" in result.lower() or "Schema" in result


class TestAnthropicCostCalculation:
    @pytest.fixture
    def adapter(self):
        from core.inference.adapters.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(api_key="test-key", model="claude-sonnet-4-5-20250929")

    def test_zero_tokens_zero_cost(self, adapter):
        assert adapter._calculate_cost(0, 0) == 0.0

    def test_positive_cost_for_tokens(self, adapter):
        cost = adapter._calculate_cost(1000, 500)
        assert cost > 0

    def test_unknown_model_returns_zero_cost(self):
        from core.inference.adapters.anthropic_adapter import AnthropicAdapter
        adapter = AnthropicAdapter(api_key="k", model="claude-unknown-xyz")
        assert adapter._calculate_cost(1000, 500) == 0.0


class TestAnthropicComplete:
    @pytest.fixture
    def adapter(self):
        from core.inference.adapters.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(api_key="test-key", model="claude-sonnet-4-5-20250929")

    def _make_anthropic_response(self, text="Hello!", stop_reason="end_turn",
                                  input_tokens=10, output_tokens=5):
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = text
        resp = MagicMock()
        resp.content = [text_block]
        resp.usage = MagicMock(input_tokens=input_tokens, output_tokens=output_tokens)
        resp.stop_reason = stop_reason
        return resp

    @pytest.mark.asyncio
    async def test_returns_response_with_content(self, adapter):
        mock_resp = self._make_anthropic_response(text="Hi there!")
        with patch.object(adapter.client.messages, "create",
                          new=AsyncMock(return_value=mock_resp)):
            result = await adapter.complete(MESSAGES)
        assert isinstance(result, Response)
        assert result.content == "Hi there!"

    @pytest.mark.asyncio
    async def test_end_turn_maps_to_stop(self, adapter):
        mock_resp = self._make_anthropic_response(stop_reason="end_turn")
        with patch.object(adapter.client.messages, "create",
                          new=AsyncMock(return_value=mock_resp)):
            result = await adapter.complete(MESSAGES)
        assert result.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_tool_use_stop_reason_maps_to_tool_calls(self, adapter):
        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.id = "call-1"
        tool_block.name = "get_weather"
        tool_block.input = {"city": "NYC"}
        resp = MagicMock()
        resp.content = [tool_block]
        resp.usage = MagicMock(input_tokens=10, output_tokens=5)
        resp.stop_reason = "tool_use"
        with patch.object(adapter.client.messages, "create",
                          new=AsyncMock(return_value=resp)):
            result = await adapter.complete(MESSAGES)
        assert result.finish_reason == "tool_calls"
        assert result.tool_calls is not None

    @pytest.mark.asyncio
    async def test_system_message_extracted_and_sent(self, adapter):
        msgs = [
            Message(role="system", content="Be brief"),
            Message(role="user", content="Hi"),
        ]
        mock_resp = self._make_anthropic_response()
        captured = {}

        async def mock_create(**kwargs):
            captured.update(kwargs)
            return mock_resp

        with patch.object(adapter.client.messages, "create", new=mock_create):
            await adapter.complete(msgs)

        assert "system" in captured
        assert "Be brief" in captured["system"]

    @pytest.mark.asyncio
    async def test_exception_propagates(self, adapter):
        with patch.object(adapter.client.messages, "create",
                          new=AsyncMock(side_effect=ValueError("bad request"))):
            with pytest.raises(ValueError, match="bad request"):
                await adapter.complete(MESSAGES)

    @pytest.mark.asyncio
    async def test_token_counts_and_cost(self, adapter):
        mock_resp = self._make_anthropic_response(input_tokens=100, output_tokens=50)
        with patch.object(adapter.client.messages, "create",
                          new=AsyncMock(return_value=mock_resp)):
            result = await adapter.complete(MESSAGES)
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.cost_usd >= 0


class TestAnthropicCompleteStream:
    @pytest.fixture
    def adapter(self):
        from core.inference.adapters.anthropic_adapter import AnthropicAdapter
        return AnthropicAdapter(api_key="test-key", model="claude-sonnet-4-5-20250929")

    def _make_events(self):
        """Build a realistic sequence of Anthropic stream events."""
        # message_start
        e_start = MagicMock()
        e_start.type = "message_start"
        e_start.message = MagicMock()
        e_start.message.usage = MagicMock(input_tokens=10)

        # content_block_delta (text)
        e_text = MagicMock()
        e_text.type = "content_block_delta"
        e_text.delta = MagicMock()
        e_text.delta.type = "text_delta"
        e_text.delta.text = "Hello!"

        # message_delta (end)
        e_end = MagicMock()
        e_end.type = "message_delta"
        e_end.usage = MagicMock(output_tokens=5)
        e_end.delta = MagicMock()
        e_end.delta.stop_reason = "end_turn"

        return [e_start, e_text, e_end]

    @pytest.mark.asyncio
    async def test_yields_text_content_chunks(self, adapter):
        events = self._make_events()
        adapter.client.messages.stream = MagicMock(
            return_value=MockAnthropicStream(events)
        )
        out = []
        async for chunk in adapter.complete_stream(MESSAGES):
            out.append(chunk)

        content_chunks = [c for c in out if c.content == "Hello!"]
        assert len(content_chunks) == 1

    @pytest.mark.asyncio
    async def test_final_chunk_has_token_counts(self, adapter):
        events = self._make_events()
        adapter.client.messages.stream = MagicMock(
            return_value=MockAnthropicStream(events)
        )
        out = []
        async for chunk in adapter.complete_stream(MESSAGES):
            out.append(chunk)

        final = out[-1]
        assert final.input_tokens == 10
        assert final.output_tokens == 5

    @pytest.mark.asyncio
    async def test_end_turn_maps_to_stop_finish_reason(self, adapter):
        events = self._make_events()
        adapter.client.messages.stream = MagicMock(
            return_value=MockAnthropicStream(events)
        )
        out = []
        async for chunk in adapter.complete_stream(MESSAGES):
            out.append(chunk)

        assert out[-1].finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_exception_propagates_in_stream(self, adapter):
        adapter.client.messages.stream = MagicMock(side_effect=RuntimeError("network error"))
        with pytest.raises(RuntimeError, match="network error"):
            async for _ in adapter.complete_stream(MESSAGES):
                pass
