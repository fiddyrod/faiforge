"""Tests for ModelRegistry and registry helper functions."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from core.inference.registry import (
    ModelRegistry,
    _create_adapter,
    _load_fallback_chains,
    _load_router,
)
from core.inference.fallback import FallbackAdapter, FallbackConfig, RetryConfig
from core.inference.adapters.base import Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_adapter(name="mock"):
    adapter = MagicMock()
    adapter.complete = AsyncMock(return_value=MagicMock(
        content="ok", model=name, input_tokens=5, output_tokens=3,
        cost_usd=0.0, latency_ms=10.0, finish_reason="stop", tool_calls=None
    ))
    return adapter


MESSAGES = [Message(role="user", content="Hi")]


# ---------------------------------------------------------------------------
# ModelRegistry: basic CRUD
# ---------------------------------------------------------------------------

class TestModelRegistryRegister:
    def test_register_single_model(self):
        reg = ModelRegistry()
        reg.register("gpt-4o-mini", make_mock_adapter())
        assert "gpt-4o-mini" in reg.list()

    def test_register_multiple_models(self):
        reg = ModelRegistry()
        reg.register("model-a", make_mock_adapter())
        reg.register("model-b", make_mock_adapter())
        assert "model-a" in reg.list()
        assert "model-b" in reg.list()

    def test_register_overrides_existing(self):
        reg = ModelRegistry()
        a1 = make_mock_adapter("a1")
        a2 = make_mock_adapter("a2")
        reg.register("model-a", a1)
        reg.register("model-a", a2)
        assert reg.get("model-a") is a2


class TestModelRegistryGet:
    def test_get_individual_model(self):
        reg = ModelRegistry()
        adapter = make_mock_adapter()
        reg.register("model-a", adapter)
        assert reg.get("model-a") is adapter

    def test_get_missing_model_raises_value_error(self):
        reg = ModelRegistry()
        with pytest.raises(ValueError, match="not found"):
            reg.get("nonexistent")

    def test_get_prefers_fallback_chain_over_model(self):
        """If a fallback chain and a model share the same name, chain wins."""
        reg = ModelRegistry()
        individual = make_mock_adapter()
        reg.register("premium", individual)

        chain_adapter = make_mock_adapter()
        config = FallbackConfig(
            chain=["premium"],
            retry=RetryConfig(max_retries=1, base_delay=0.001, max_delay=0.01),
            circuit_breaker_threshold=3,
        )
        fa = FallbackAdapter(adapters={"premium": chain_adapter}, config=config)
        reg.register_fallback_chain("premium", fa)

        assert reg.get("premium") is fa

    def test_get_fallback_chain_by_name(self):
        reg = ModelRegistry()
        chain_mock = MagicMock()
        reg.register_fallback_chain("my_chain", chain_mock)
        assert reg.get("my_chain") is chain_mock


class TestModelRegistryList:
    def test_list_includes_models_and_chains(self):
        reg = ModelRegistry()
        reg.register("model-a", make_mock_adapter())
        reg.register_fallback_chain("chain-b", MagicMock())
        names = reg.list()
        assert "model-a" in names
        assert "chain-b" in names

    def test_list_models_only(self):
        reg = ModelRegistry()
        reg.register("model-a", make_mock_adapter())
        reg.register_fallback_chain("chain-b", MagicMock())
        assert reg.list_models() == ["model-a"]
        assert "chain-b" not in reg.list_models()

    def test_list_chains_only(self):
        reg = ModelRegistry()
        reg.register("model-a", make_mock_adapter())
        reg.register_fallback_chain("chain-b", MagicMock())
        assert reg.list_chains() == ["chain-b"]
        assert "model-a" not in reg.list_chains()

    def test_empty_registry_returns_empty_list(self):
        reg = ModelRegistry()
        assert reg.list() == []


class TestModelRegistryGetRouted:
    def test_uses_router_when_set(self):
        reg = ModelRegistry()
        adapter = make_mock_adapter()
        reg.register("model-a", adapter)

        router = MagicMock()
        router.route = MagicMock(return_value=adapter)
        reg.set_router(router)

        result = reg.get_routed(MESSAGES, max_tokens=100)
        assert result is adapter
        router.route.assert_called_once()

    def test_fallback_to_first_chain_without_router(self):
        reg = ModelRegistry()
        chain = MagicMock()
        reg.register_fallback_chain("my_chain", chain)
        result = reg.get_routed(MESSAGES)
        assert result is chain

    def test_fallback_to_first_model_without_router_or_chains(self):
        reg = ModelRegistry()
        adapter = make_mock_adapter()
        reg.register("model-a", adapter)
        result = reg.get_routed(MESSAGES)
        assert result is adapter


class TestModelRegistryHealthStatus:
    def test_health_status_empty_when_no_chains(self):
        reg = ModelRegistry()
        assert reg.get_health_status() == {}

    def test_health_status_includes_all_chains(self):
        reg = ModelRegistry()
        chain_a = MagicMock()
        chain_a.get_health_status = MagicMock(return_value={"model-a": {"status": "healthy"}})
        chain_b = MagicMock()
        chain_b.get_health_status = MagicMock(return_value={"model-b": {"status": "unknown"}})

        reg.register_fallback_chain("chain-a", chain_a)
        reg.register_fallback_chain("chain-b", chain_b)

        status = reg.get_health_status()
        assert "chain-a" in status
        assert "chain-b" in status


# ---------------------------------------------------------------------------
# _create_adapter
# ---------------------------------------------------------------------------

class TestCreateAdapter:
    def test_openai_adapter_created(self):
        with patch("core.inference.registry.OpenAIAdapter") as MockOpenAI:
            MockOpenAI.return_value = MagicMock()
            result = _create_adapter(
                "openai",
                {"model": "gpt-4o-mini"},
                "test-key", None, False
            )
        MockOpenAI.assert_called_once_with(api_key="test-key", model="gpt-4o-mini")
        assert result is not None

    def test_anthropic_without_key_returns_none(self):
        result = _create_adapter(
            "anthropic",
            {"model": "claude-sonnet-4-5-20250929"},
            "openai-key", None, False
        )
        assert result is None

    def test_anthropic_with_key_created(self):
        with patch("core.inference.registry.AnthropicAdapter") as MockAnthropic:
            MockAnthropic.return_value = MagicMock()
            result = _create_adapter(
                "anthropic",
                {"model": "claude-sonnet-4-5-20250929"},
                "openai-key", "anthropic-key", False
            )
        MockAnthropic.assert_called_once()
        assert result is not None

    def test_vllm_with_load_vllm_false_returns_none(self):
        result = _create_adapter(
            "vllm",
            {"model": "llama3"},
            "key", None, load_vllm=False
        )
        assert result is None

    def test_unknown_adapter_type_returns_none(self):
        result = _create_adapter(
            "unknown_type",
            {"model": "some-model"},
            "key", None, False
        )
        assert result is None


# ---------------------------------------------------------------------------
# _load_fallback_chains
# ---------------------------------------------------------------------------

class TestLoadFallbackChains:
    def _make_config(self, models, retry=None, circuit_breaker=None):
        chain_config = {"models": models}
        if retry:
            chain_config["retry"] = retry
        if circuit_breaker:
            chain_config["circuit_breaker"] = circuit_breaker
        return {"fallback_chains": {"test_chain": chain_config}}

    def test_chain_created_for_available_models(self):
        adapters = {"model-a": make_mock_adapter(), "model-b": make_mock_adapter()}
        config = self._make_config(["model-a", "model-b"])
        chains = _load_fallback_chains(config, adapters)
        assert "test_chain" in chains
        assert isinstance(chains["test_chain"], FallbackAdapter)

    def test_skips_chain_when_no_models_available(self):
        adapters = {}  # empty - no models
        config = self._make_config(["model-a", "model-b"])
        chains = _load_fallback_chains(config, adapters)
        assert "test_chain" not in chains

    def test_partial_models_available(self):
        """Chain is created with only available models from the list."""
        adapters = {"model-a": make_mock_adapter()}  # model-b missing
        config = self._make_config(["model-a", "model-b"])
        chains = _load_fallback_chains(config, adapters)
        assert "test_chain" in chains

    def test_custom_retry_config_applied(self):
        adapters = {"model-a": make_mock_adapter()}
        config = self._make_config(
            ["model-a"],
            retry={"max_retries": 5, "base_delay": 0.5, "max_delay": 10.0}
        )
        chains = _load_fallback_chains(config, adapters)
        assert "test_chain" in chains

    def test_empty_fallback_chains_config(self):
        adapters = {"model-a": make_mock_adapter()}
        config = {"fallback_chains": {}}
        chains = _load_fallback_chains(config, adapters)
        assert chains == {}

    def test_multiple_chains_loaded(self):
        adapters = {
            "model-a": make_mock_adapter(),
            "model-b": make_mock_adapter(),
        }
        config = {
            "fallback_chains": {
                "chain-1": {"models": ["model-a"]},
                "chain-2": {"models": ["model-b"]},
            }
        }
        chains = _load_fallback_chains(config, adapters)
        assert "chain-1" in chains
        assert "chain-2" in chains


# ---------------------------------------------------------------------------
# _load_router
# ---------------------------------------------------------------------------

class TestLoadRouter:
    def test_returns_none_when_routing_disabled(self):
        config = {"routing": {"enabled": False}}
        result = _load_router(config, {}, {})
        assert result is None

    def test_returns_none_when_routing_key_absent(self):
        config = {}
        result = _load_router(config, {}, {})
        assert result is None

    def test_returns_smart_router_when_enabled(self):
        from core.inference.fallback import SmartRouter
        adapters = {"model-a": make_mock_adapter()}
        config = {
            "routing": {
                "enabled": True,
                "default_chain": "model-a",
                "rules": []
            }
        }
        router = _load_router(config, adapters, {})
        assert isinstance(router, SmartRouter)

    def test_rules_converted_to_router_format(self):
        from core.inference.fallback import SmartRouter
        adapters = {
            "fast": make_mock_adapter(),
            "smart": make_mock_adapter(),
        }
        chains = {"premium": MagicMock()}
        config = {
            "routing": {
                "enabled": True,
                "default_chain": "fast",
                "rules": [
                    {"condition": "has_tools", "fallback_chain": "premium"},
                    {"condition": "max_tokens > 2000", "model": "smart"},
                ]
            }
        }
        router = _load_router(config, adapters, chains)
        assert isinstance(router, SmartRouter)


# ---------------------------------------------------------------------------
# load_registry integration (uses real models.yaml, no API calls)
# ---------------------------------------------------------------------------

class TestLoadRegistry:
    def test_load_registry_returns_model_registry(self):
        from core.inference.registry import load_registry
        registry = load_registry("core/config/models.yaml", "fake-openai-key")
        assert isinstance(registry, ModelRegistry)

    def test_load_registry_has_at_least_one_model(self):
        from core.inference.registry import load_registry
        registry = load_registry("core/config/models.yaml", "fake-openai-key")
        assert len(registry.list()) > 0

    def test_load_registry_raises_for_missing_config(self):
        from core.inference.registry import load_registry
        with pytest.raises(FileNotFoundError):
            load_registry("/nonexistent/path/models.yaml", "key")

    def test_load_registry_without_anthropic_key_skips_anthropic_models(self):
        from core.inference.registry import load_registry
        registry = load_registry(
            "core/config/models.yaml",
            "fake-openai-key",
            anthropic_api_key=None
        )
        # Anthropic models should not be registered
        for name in registry.list_models():
            assert "claude" not in name.lower()
