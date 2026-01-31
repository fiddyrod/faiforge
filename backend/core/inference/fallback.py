"""
Model Fallback & Smart Routing System

Provides:
- Automatic failover when primary model is unavailable
- Configurable fallback chains
- Health checks for provider availability
- Retry with exponential backoff
- Optional query-based routing
"""

import asyncio
import time
from typing import List, Optional, Dict, Any, AsyncGenerator, Callable
from dataclasses import dataclass, field
from enum import Enum

from .adapters import (
    LLMAdapter,
    Message,
    Response,
    StreamChunk,
    Tool,
    ResponseFormat,
)
from ..observability import get_logger, log_with_context


class ProviderStatus(Enum):
    """Health status of a provider"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ProviderHealth:
    """Tracks health status of a provider"""
    status: ProviderStatus = ProviderStatus.UNKNOWN
    last_check: float = 0.0
    last_success: float = 0.0
    last_failure: float = 0.0
    consecutive_failures: int = 0
    error_message: Optional[str] = None


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 30.0  # seconds
    exponential_base: float = 2.0

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt number (0-indexed)"""
        delay = self.base_delay * (self.exponential_base ** attempt)
        return min(delay, self.max_delay)


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior"""
    chain: List[str] = field(default_factory=list)  # Model names in order
    retry: RetryConfig = field(default_factory=RetryConfig)
    health_check_interval: float = 60.0  # seconds
    circuit_breaker_threshold: int = 5  # failures before marking unhealthy
    circuit_breaker_recovery: float = 300.0  # seconds before retrying unhealthy provider


class FallbackAdapter(LLMAdapter):
    """
    Adapter that wraps multiple adapters with fallback logic.

    Features:
    - Tries primary adapter first
    - Falls back to secondary adapters on failure
    - Implements retry with exponential backoff
    - Tracks provider health
    - Circuit breaker pattern for unhealthy providers
    """

    def __init__(
        self,
        adapters: Dict[str, LLMAdapter],
        config: FallbackConfig,
        name: str = "fallback"
    ):
        """
        Initialize fallback adapter.

        Args:
            adapters: Dict of model_name -> adapter
            config: Fallback configuration
            name: Name for this fallback group (for logging)
        """
        self.adapters = adapters
        self.config = config
        self.name = name
        self.logger = get_logger()

        # Track health of each provider
        self._health: Dict[str, ProviderHealth] = {
            name: ProviderHealth() for name in config.chain
        }

        # Validate chain
        for model_name in config.chain:
            if model_name not in adapters:
                raise ValueError(f"Model '{model_name}' in fallback chain not found in adapters")

    def _is_provider_available(self, model_name: str) -> bool:
        """Check if provider should be attempted based on health status"""
        health = self._health.get(model_name)
        if not health:
            return False

        # Unknown status - try it
        if health.status == ProviderStatus.UNKNOWN:
            return True

        # Healthy - always available
        if health.status == ProviderStatus.HEALTHY:
            return True

        # Degraded - available but monitor closely
        if health.status == ProviderStatus.DEGRADED:
            return True

        # Unhealthy - check if recovery period has passed
        if health.status == ProviderStatus.UNHEALTHY:
            time_since_failure = time.time() - health.last_failure
            if time_since_failure > self.config.circuit_breaker_recovery:
                log_with_context(
                    self.logger, "info",
                    f"Circuit breaker recovery: attempting {model_name}",
                    event="circuit_breaker_recovery",
                    model=model_name,
                    time_since_failure=time_since_failure
                )
                return True
            return False

        return True

    def _record_success(self, model_name: str):
        """Record successful request"""
        health = self._health.get(model_name)
        if health:
            health.status = ProviderStatus.HEALTHY
            health.last_success = time.time()
            health.last_check = time.time()
            health.consecutive_failures = 0
            health.error_message = None

    def _record_failure(self, model_name: str, error: str):
        """Record failed request"""
        health = self._health.get(model_name)
        if health:
            health.last_failure = time.time()
            health.last_check = time.time()
            health.consecutive_failures += 1
            health.error_message = error

            # Update status based on consecutive failures
            if health.consecutive_failures >= self.config.circuit_breaker_threshold:
                health.status = ProviderStatus.UNHEALTHY
                log_with_context(
                    self.logger, "warning",
                    f"Circuit breaker tripped for {model_name}",
                    event="circuit_breaker_tripped",
                    model=model_name,
                    consecutive_failures=health.consecutive_failures
                )
            elif health.consecutive_failures >= 2:
                health.status = ProviderStatus.DEGRADED

    async def _try_adapter(
        self,
        model_name: str,
        method: Callable,
        **kwargs
    ) -> tuple[bool, Any, Optional[str]]:
        """
        Try calling an adapter with retries.

        Returns:
            Tuple of (success, result, error_message)
        """
        adapter = self.adapters.get(model_name)
        if not adapter:
            return False, None, f"Adapter not found: {model_name}"

        last_error = None

        for attempt in range(self.config.retry.max_retries):
            try:
                result = await method(adapter, **kwargs)
                self._record_success(model_name)
                return True, result, None

            except Exception as e:
                last_error = str(e)

                log_with_context(
                    self.logger, "warning",
                    f"Attempt {attempt + 1}/{self.config.retry.max_retries} failed for {model_name}",
                    event="fallback_retry",
                    model=model_name,
                    attempt=attempt + 1,
                    max_retries=self.config.retry.max_retries,
                    error=last_error
                )

                # Don't retry on last attempt
                if attempt < self.config.retry.max_retries - 1:
                    delay = self.config.retry.get_delay(attempt)
                    await asyncio.sleep(delay)

        self._record_failure(model_name, last_error)
        return False, None, last_error

    async def complete(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 500,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[ResponseFormat] = None
    ) -> Response:
        """
        Generate completion with automatic fallback.

        Tries each adapter in the fallback chain until one succeeds.
        """
        errors = []

        for model_name in self.config.chain:
            # Skip unavailable providers
            if not self._is_provider_available(model_name):
                log_with_context(
                    self.logger, "debug",
                    f"Skipping unavailable provider: {model_name}",
                    event="fallback_skip",
                    model=model_name,
                    status=self._health[model_name].status.value
                )
                continue

            log_with_context(
                self.logger, "info",
                f"Attempting completion with {model_name}",
                event="fallback_attempt",
                model=model_name,
                fallback_group=self.name
            )

            async def call_complete(adapter, **kw):
                return await adapter.complete(**kw)

            success, result, error = await self._try_adapter(
                model_name,
                call_complete,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
                tool_choice=tool_choice,
                response_format=response_format
            )

            if success:
                log_with_context(
                    self.logger, "info",
                    f"Completion successful with {model_name}",
                    event="fallback_success",
                    model=model_name,
                    fallback_group=self.name
                )
                return result

            errors.append(f"{model_name}: {error}")

        # All providers failed
        error_summary = "; ".join(errors)
        log_with_context(
            self.logger, "error",
            f"All providers in fallback chain failed",
            event="fallback_exhausted",
            fallback_group=self.name,
            chain=self.config.chain,
            errors=errors
        )
        raise RuntimeError(f"All providers failed: {error_summary}")

    async def complete_stream(
        self,
        messages: List[Message],
        temperature: float = 0.7,
        max_tokens: int = 500,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[str | Dict[str, Any]] = None,
        response_format: Optional[ResponseFormat] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Generate streaming completion with automatic fallback.

        Note: Streaming fallback is more complex - we can't retry mid-stream.
        We attempt each provider once for streaming.
        """
        errors = []

        for model_name in self.config.chain:
            if not self._is_provider_available(model_name):
                continue

            adapter = self.adapters.get(model_name)
            if not adapter:
                continue

            log_with_context(
                self.logger, "info",
                f"Attempting streaming with {model_name}",
                event="fallback_stream_attempt",
                model=model_name,
                fallback_group=self.name
            )

            try:
                # Try to get the first chunk to verify connection works
                stream = adapter.complete_stream(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format
                )

                # Yield all chunks from successful stream
                async for chunk in stream:
                    yield chunk

                self._record_success(model_name)
                return  # Success - don't try other providers

            except Exception as e:
                error = str(e)
                self._record_failure(model_name, error)
                errors.append(f"{model_name}: {error}")

                log_with_context(
                    self.logger, "warning",
                    f"Streaming failed for {model_name}, trying next",
                    event="fallback_stream_failure",
                    model=model_name,
                    error=error
                )
                continue

        # All providers failed
        error_summary = "; ".join(errors)
        log_with_context(
            self.logger, "error",
            f"All streaming providers failed",
            event="fallback_stream_exhausted",
            fallback_group=self.name,
            errors=errors
        )
        raise RuntimeError(f"All streaming providers failed: {error_summary}")

    def get_health_status(self) -> Dict[str, Dict[str, Any]]:
        """Get health status of all providers in chain"""
        return {
            name: {
                "status": health.status.value,
                "consecutive_failures": health.consecutive_failures,
                "last_success": health.last_success,
                "last_failure": health.last_failure,
                "error_message": health.error_message
            }
            for name, health in self._health.items()
        }


class SmartRouter:
    """
    Routes requests to appropriate models based on rules.

    Can be combined with FallbackAdapter for routing + fallback.
    """

    def __init__(
        self,
        adapters: Dict[str, LLMAdapter],
        fallback_adapters: Dict[str, FallbackAdapter],
        default_model: str,
        rules: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize smart router.

        Args:
            adapters: Direct model adapters
            fallback_adapters: Fallback-wrapped adapters
            default_model: Default model/fallback group to use
            rules: List of routing rules
        """
        self.adapters = adapters
        self.fallback_adapters = fallback_adapters
        self.default_model = default_model
        self.rules = rules or []
        self.logger = get_logger()

    def _evaluate_rule(
        self,
        rule: Dict[str, Any],
        messages: List[Message],
        max_tokens: int,
        tools: Optional[List[Tool]]
    ) -> bool:
        """Evaluate if a routing rule matches the request"""
        condition = rule.get("condition", "")

        # Simple rule evaluation
        if condition == "has_tools" and tools:
            return True

        if condition.startswith("max_tokens >"):
            threshold = int(condition.split(">")[1].strip())
            return max_tokens > threshold

        if condition.startswith("max_tokens <"):
            threshold = int(condition.split("<")[1].strip())
            return max_tokens < threshold

        if condition.startswith("message_count >"):
            threshold = int(condition.split(">")[1].strip())
            return len(messages) > threshold

        # Check for long messages (complex queries)
        if condition == "long_context":
            total_content = sum(len(m.content or "") for m in messages)
            return total_content > 4000

        return False

    def route(
        self,
        messages: List[Message],
        max_tokens: int = 500,
        tools: Optional[List[Tool]] = None
    ) -> LLMAdapter:
        """
        Route request to appropriate adapter based on rules.

        Returns:
            The adapter to use for this request
        """
        # Evaluate rules in order
        for rule in self.rules:
            if self._evaluate_rule(rule, messages, max_tokens, tools):
                target = rule.get("model") or rule.get("fallback_group")

                log_with_context(
                    self.logger, "info",
                    f"Routing rule matched: {rule.get('condition')} -> {target}",
                    event="smart_route_match",
                    condition=rule.get("condition"),
                    target=target
                )

                # Check fallback groups first, then direct adapters
                if target in self.fallback_adapters:
                    return self.fallback_adapters[target]
                if target in self.adapters:
                    return self.adapters[target]

        # Use default
        log_with_context(
            self.logger, "debug",
            f"No routing rule matched, using default: {self.default_model}",
            event="smart_route_default",
            default=self.default_model
        )

        if self.default_model in self.fallback_adapters:
            return self.fallback_adapters[self.default_model]
        return self.adapters[self.default_model]
