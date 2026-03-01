"""Tests for FallbackAdapter (circuit breaker, retry) and SmartRouter."""
import time
import pytest
from unittest.mock import AsyncMock, MagicMock

from core.inference.fallback import (
    FallbackAdapter,
    FallbackConfig,
    RetryConfig,
    SmartRouter,
    ProviderStatus,
)
from core.inference.adapters.base import (
    Message,
    Response,
    StreamChunk,
    Tool,
    FunctionDef,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MESSAGES = [Message(role="user", content="Hello!")]


def make_response(content="Test response", model="test-model"):
    return Response(
        content=content,
        model=model,
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.001,
        latency_ms=100.0,
    )


def make_adapter(content="Test response", side_effect=None):
    adapter = MagicMock()
    if side_effect:
        adapter.complete = AsyncMock(side_effect=side_effect)
    else:
        adapter.complete = AsyncMock(return_value=make_response(content=content))

    async def _stream_gen():
        yield StreamChunk(content="token1")
        yield StreamChunk(finish_reason="stop", input_tokens=5, output_tokens=3)

    adapter.complete_stream = MagicMock(return_value=_stream_gen())
    return adapter


@pytest.fixture
def fast_retry():
    return RetryConfig(max_retries=2, base_delay=0.001, max_delay=0.01, exponential_base=2.0)


@pytest.fixture
def fallback_config(fast_retry):
    return FallbackConfig(
        chain=["model-a", "model-b"],
        retry=fast_retry,
        circuit_breaker_threshold=3,
        circuit_breaker_recovery=1.0,
    )


def make_fallback(config, adapters):
    return FallbackAdapter(adapters=adapters, config=config)


# ---------------------------------------------------------------------------
# RetryConfig
# ---------------------------------------------------------------------------

class TestRetryConfig:
    def test_delay_increases_exponentially(self):
        cfg = RetryConfig(max_retries=4, base_delay=1.0, max_delay=30.0, exponential_base=2.0)
        assert cfg.get_delay(0) == 1.0
        assert cfg.get_delay(1) == 2.0
        assert cfg.get_delay(2) == 4.0

    def test_delay_capped_at_max_delay(self):
        cfg = RetryConfig(max_retries=10, base_delay=1.0, max_delay=5.0, exponential_base=2.0)
        assert cfg.get_delay(10) == 5.0

    def test_zero_attempt_returns_base_delay(self):
        cfg = RetryConfig(base_delay=2.5)
        assert cfg.get_delay(0) == 2.5


# ---------------------------------------------------------------------------
# FallbackAdapter initialisation
# ---------------------------------------------------------------------------

class TestFallbackAdapterInit:
    def test_init_succeeds_with_all_adapters_present(self, fallback_config):
        adapters = {"model-a": make_adapter(), "model-b": make_adapter()}
        fa = make_fallback(fallback_config, adapters)
        assert fa.name == "fallback"
        assert set(fa._health.keys()) == {"model-a", "model-b"}

    def test_init_raises_for_missing_adapter_in_chain(self, fallback_config):
        adapters = {"model-a": make_adapter()}   # model-b is missing
        with pytest.raises(ValueError, match="model-b"):
            make_fallback(fallback_config, adapters)

    def test_initial_health_status_unknown(self, fallback_config):
        adapters = {"model-a": make_adapter(), "model-b": make_adapter()}
        fa = make_fallback(fallback_config, adapters)
        assert fa._health["model-a"].status == ProviderStatus.UNKNOWN


# ---------------------------------------------------------------------------
# Health recording
# ---------------------------------------------------------------------------

class TestHealthRecording:
    def test_record_success_marks_healthy_and_resets_failures(self, fallback_config):
        adapters = {"model-a": make_adapter(), "model-b": make_adapter()}
        fa = make_fallback(fallback_config, adapters)
        fa._record_failure("model-a", "err")
        fa._record_failure("model-a", "err")
        fa._record_success("model-a")
        assert fa._health["model-a"].status == ProviderStatus.HEALTHY
        assert fa._health["model-a"].consecutive_failures == 0
        assert fa._health["model-a"].error_message is None

    def test_two_failures_marks_degraded(self, fallback_config):
        adapters = {"model-a": make_adapter(), "model-b": make_adapter()}
        fa = make_fallback(fallback_config, adapters)
        fa._record_failure("model-a", "err")
        fa._record_failure("model-a", "err")
        assert fa._health["model-a"].status == ProviderStatus.DEGRADED

    def test_circuit_breaker_trips_at_threshold(self, fallback_config):
        adapters = {"model-a": make_adapter(), "model-b": make_adapter()}
        fa = make_fallback(fallback_config, adapters)
        for _ in range(fallback_config.circuit_breaker_threshold):
            fa._record_failure("model-a", "err")
        assert fa._health["model-a"].status == ProviderStatus.UNHEALTHY

    def test_error_message_stored(self, fallback_config):
        adapters = {"model-a": make_adapter(), "model-b": make_adapter()}
        fa = make_fallback(fallback_config, adapters)
        fa._record_failure("model-a", "timeout error")
        assert fa._health["model-a"].error_message == "timeout error"


# ---------------------------------------------------------------------------
# Provider availability
# ---------------------------------------------------------------------------

class TestProviderAvailability:
    def test_unknown_status_is_available(self, fallback_config):
        adapters = {"model-a": make_adapter(), "model-b": make_adapter()}
        fa = make_fallback(fallback_config, adapters)
        assert fa._is_provider_available("model-a") is True

    def test_healthy_status_is_available(self, fallback_config):
        adapters = {"model-a": make_adapter(), "model-b": make_adapter()}
        fa = make_fallback(fallback_config, adapters)
        fa._record_success("model-a")
        assert fa._is_provider_available("model-a") is True

    def test_degraded_status_is_available(self, fallback_config):
        adapters = {"model-a": make_adapter(), "model-b": make_adapter()}
        fa = make_fallback(fallback_config, adapters)
        fa._record_failure("model-a", "err")
        fa._record_failure("model-a", "err")  # -> degraded
        assert fa._is_provider_available("model-a") is True

    def test_unhealthy_within_recovery_is_unavailable(self, fallback_config):
        adapters = {"model-a": make_adapter(), "model-b": make_adapter()}
        fa = make_fallback(fallback_config, adapters)
        for _ in range(fallback_config.circuit_breaker_threshold):
            fa._record_failure("model-a", "err")
        assert fa._is_provider_available("model-a") is False

    def test_unhealthy_after_recovery_window_becomes_available(self, fallback_config):
        adapters = {"model-a": make_adapter(), "model-b": make_adapter()}
        fa = make_fallback(fallback_config, adapters)
        for _ in range(fallback_config.circuit_breaker_threshold):
            fa._record_failure("model-a", "err")
        # Push last_failure into the past past the recovery window
        fa._health["model-a"].last_failure = time.time() - fallback_config.circuit_breaker_recovery - 1
        assert fa._is_provider_available("model-a") is True


# ---------------------------------------------------------------------------
# complete()
# ---------------------------------------------------------------------------

class TestComplete:
    @pytest.mark.asyncio
    async def test_uses_primary_adapter_first(self, fallback_config):
        primary = make_adapter(content="primary")
        secondary = make_adapter(content="secondary")
        fa = FallbackAdapter(
            adapters={"model-a": primary, "model-b": secondary},
            config=fallback_config,
        )
        result = await fa.complete(MESSAGES)
        assert result.content == "primary"
        secondary.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_when_primary_raises(self, fallback_config):
        primary = make_adapter(side_effect=Exception("API error"))
        secondary = make_adapter(content="secondary")
        fa = FallbackAdapter(
            adapters={"model-a": primary, "model-b": secondary},
            config=fallback_config,
        )
        result = await fa.complete(MESSAGES)
        assert result.content == "secondary"

    @pytest.mark.asyncio
    async def test_raises_runtime_error_when_all_fail(self, fallback_config):
        fa = FallbackAdapter(
            adapters={
                "model-a": make_adapter(side_effect=Exception("err-a")),
                "model-b": make_adapter(side_effect=Exception("err-b")),
            },
            config=fallback_config,
        )
        with pytest.raises(RuntimeError, match="All providers failed"):
            await fa.complete(MESSAGES)

    @pytest.mark.asyncio
    async def test_skips_unhealthy_provider(self, fallback_config):
        primary = make_adapter(content="primary")
        secondary = make_adapter(content="secondary")
        fa = FallbackAdapter(
            adapters={"model-a": primary, "model-b": secondary},
            config=fallback_config,
        )
        for _ in range(fallback_config.circuit_breaker_threshold):
            fa._record_failure("model-a", "err")

        result = await fa.complete(MESSAGES)
        assert result.content == "secondary"
        primary.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_failure_recorded_for_failing_adapter(self, fallback_config):
        fa = FallbackAdapter(
            adapters={
                "model-a": make_adapter(side_effect=Exception("fail")),
                "model-b": make_adapter(content="ok"),
            },
            config=fallback_config,
        )
        await fa.complete(MESSAGES)
        assert fa._health["model-a"].consecutive_failures > 0

    @pytest.mark.asyncio
    async def test_success_recorded_for_successful_adapter(self, fallback_config):
        fa = FallbackAdapter(
            adapters={"model-a": make_adapter(), "model-b": make_adapter()},
            config=fallback_config,
        )
        await fa.complete(MESSAGES)
        assert fa._health["model-a"].status == ProviderStatus.HEALTHY


# ---------------------------------------------------------------------------
# get_health_status()
# ---------------------------------------------------------------------------

class TestGetHealthStatus:
    def test_all_providers_in_status_dict(self, fallback_config):
        adapters = {"model-a": make_adapter(), "model-b": make_adapter()}
        fa = make_fallback(fallback_config, adapters)
        status = fa.get_health_status()
        assert "model-a" in status
        assert "model-b" in status

    def test_status_fields_present(self, fallback_config):
        adapters = {"model-a": make_adapter(), "model-b": make_adapter()}
        fa = make_fallback(fallback_config, adapters)
        s = fa.get_health_status()["model-a"]
        assert "status" in s
        assert "consecutive_failures" in s
        assert "last_success" in s
        assert "last_failure" in s

    @pytest.mark.asyncio
    async def test_status_updated_to_healthy_after_success(self, fallback_config):
        adapters = {"model-a": make_adapter(), "model-b": make_adapter()}
        fa = make_fallback(fallback_config, adapters)
        await fa.complete(MESSAGES)
        assert fa.get_health_status()["model-a"]["status"] == "healthy"


# ---------------------------------------------------------------------------
# SmartRouter
# ---------------------------------------------------------------------------

class TestSmartRouter:
    @pytest.fixture
    def router_setup(self):
        adapters = {
            "model-a": make_adapter(),
            "fast-model": make_adapter(),
        }
        hq_group = MagicMock()
        router = SmartRouter(
            adapters=adapters,
            fallback_adapters={"high_quality": hq_group},
            default_model="model-a",
            rules=[
                {"condition": "has_tools", "fallback_group": "high_quality"},
                {"condition": "max_tokens > 2000", "fallback_group": "high_quality"},
                {"condition": "max_tokens < 100", "model": "fast-model"},
                {"condition": "message_count > 10", "fallback_group": "high_quality"},
                {"condition": "long_context", "fallback_group": "high_quality"},
            ],
        )
        return router, adapters, hq_group

    def test_routes_to_default_when_no_rule_matches(self, router_setup):
        router, adapters, _ = router_setup
        result = router.route(MESSAGES, max_tokens=500)
        assert result is adapters["model-a"]

    def test_routes_to_high_quality_when_has_tools(self, router_setup):
        router, _, hq_group = router_setup
        tool = Tool(type="function", function=FunctionDef(
            name="get_weather", description="Get weather", parameters={},
        ))
        result = router.route(MESSAGES, tools=[tool])
        assert result is hq_group

    def test_routes_to_high_quality_for_large_max_tokens(self, router_setup):
        router, _, hq_group = router_setup
        result = router.route(MESSAGES, max_tokens=3000)
        assert result is hq_group

    def test_routes_to_fast_model_for_small_max_tokens(self, router_setup):
        router, adapters, _ = router_setup
        result = router.route(MESSAGES, max_tokens=50)
        assert result is adapters["fast-model"]

    def test_routes_to_high_quality_for_many_messages(self, router_setup):
        router, _, hq_group = router_setup
        many = [Message(role="user", content="msg") for _ in range(15)]
        result = router.route(many, max_tokens=500)
        assert result is hq_group

    def test_routes_to_high_quality_for_long_context(self, router_setup):
        router, _, hq_group = router_setup
        long_msg = Message(role="user", content="x" * 5000)
        result = router.route([long_msg], max_tokens=500)
        assert result is hq_group

    def test_first_matching_rule_wins(self, router_setup):
        """has_tools rule comes before max_tokens > 2000 in our list; both match here."""
        router, _, hq_group = router_setup
        tool = Tool(type="function", function=FunctionDef(name="fn", description="d", parameters={}))
        result = router.route(MESSAGES, max_tokens=5000, tools=[tool])
        assert result is hq_group

    def test_no_rules_returns_default(self):
        adapter = make_adapter()
        router = SmartRouter(
            adapters={"model-a": adapter},
            fallback_adapters={},
            default_model="model-a",
            rules=[],
        )
        result = router.route(MESSAGES, max_tokens=500)
        assert result is adapter
