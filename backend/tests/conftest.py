"""Pytest configuration and fixtures for FAIForge tests"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, AsyncMock
from core.config import AppConfig, APIConfig, CORSConfig, DefaultsConfig, ModelsConfig, ObservabilityConfig, CacheConfig, RateLimitConfig
from core.inference.registry import ModelRegistry


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing"""
    return AppConfig(
        api=APIConfig(host="0.0.0.0", port=8000, reload=False, workers=1),
        cors=CORSConfig(
            enabled=True,
            origins=["http://localhost:3000"],
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"]
        ),
        defaults=DefaultsConfig(model="gpt-4o-mini", temperature=0.7, max_tokens=500),
        models=ModelsConfig(config_path="core/config/models.yaml", load_vllm=False),
        observability=ObservabilityConfig(log_level="INFO", log_format="json", request_logging=True),
        cache=CacheConfig(enabled=False, backend="redis", ttl_seconds=3600),
        rate_limit=RateLimitConfig(enabled=False, requests_per_minute=60)
    )


@pytest.fixture
def mock_registry():
    """Create a mock model registry for testing"""
    registry = ModelRegistry()

    # Mock adapter
    mock_adapter = Mock()
    mock_adapter.complete = AsyncMock(return_value=Mock(
        content="Test response",
        model="gpt-4o-mini",
        input_tokens=10,
        output_tokens=5,
        cost_usd=0.00001,
        latency_ms=100.0
    ))

    registry.register("gpt-4o-mini", mock_adapter)
    return registry


@pytest.fixture
def api_keys():
    """Provide test API keys"""
    return {
        "openai": "test-openai-key",
        "anthropic": "test-anthropic-key"
    }
