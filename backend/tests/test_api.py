"""Tests for API endpoints"""
import pytest
from fastapi.testclient import TestClient
from core.api.server import create_app


@pytest.mark.asyncio
class TestAPIEndpoints:
    """Test suite for API endpoints"""

    def test_root_endpoint(self, mock_config, api_keys):
        """Test root endpoint returns correct response"""
        app = create_app(mock_config, api_keys["openai"], api_keys["anthropic"])
        client = TestClient(app)

        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "FAIForge API"
        assert data["version"] == "2.3.0"
        assert data["status"] == "ok"

    def test_health_endpoint(self, mock_config, api_keys):
        """Test health check endpoint"""
        app = create_app(mock_config, api_keys["openai"], api_keys["anthropic"])
        client = TestClient(app)

        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "models_loaded" in data

    def test_list_models_endpoint(self, mock_config, api_keys):
        """Test list models endpoint"""
        app = create_app(mock_config, api_keys["openai"], api_keys["anthropic"])
        client = TestClient(app)

        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)

    def test_chat_completion_invalid_model(self, mock_config, api_keys):
        """Test chat completion with invalid model returns 400"""
        app = create_app(mock_config, api_keys["openai"], api_keys["anthropic"])
        client = TestClient(app)

        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "invalid-model"
        }

        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 400
        assert "not found" in response.json()["detail"].lower()

    def test_chat_completion_validation_empty_messages(self, mock_config, api_keys):
        """Test chat completion validates empty messages list"""
        app = create_app(mock_config, api_keys["openai"], api_keys["anthropic"])
        client = TestClient(app)

        payload = {
            "messages": [],
            "model": "gpt-4o-mini"
        }

        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 422  # Validation error

    def test_chat_completion_validation_invalid_role(self, mock_config, api_keys):
        """Test chat completion validates message role"""
        app = create_app(mock_config, api_keys["openai"], api_keys["anthropic"])
        client = TestClient(app)

        payload = {
            "messages": [{"role": "invalid_role", "content": "Hello"}],
            "model": "gpt-4o-mini"
        }

        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 422  # Validation error

    def test_chat_completion_validation_temperature_range(self, mock_config, api_keys):
        """Test chat completion validates temperature range"""
        app = create_app(mock_config, api_keys["openai"], api_keys["anthropic"])
        client = TestClient(app)

        # Test temperature too high
        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4o-mini",
            "temperature": 3.0  # Invalid: > 2.0
        }

        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 422

    def test_chat_completion_validation_max_tokens(self, mock_config, api_keys):
        """Test chat completion validates max_tokens"""
        app = create_app(mock_config, api_keys["openai"], api_keys["anthropic"])
        client = TestClient(app)

        # Test max_tokens too low
        payload = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4o-mini",
            "max_tokens": 0  # Invalid: < 1
        }

        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 422
