"""Tests for configuration management"""
import pytest
import os
import tempfile
import yaml
from pathlib import Path
from core.config import load_config, AppConfig, APIConfig


class TestConfigLoading:
    """Test suite for configuration loading"""

    def test_load_default_config(self):
        """Test loading default configuration"""
        config = load_config()

        assert isinstance(config, AppConfig)
        assert isinstance(config.api, APIConfig)
        assert config.api.port == 8000
        assert config.defaults.model == "gpt-4o-mini"

    def test_environment_variable_override(self):
        """Test that environment variables override config"""
        # Set environment variable
        os.environ["FAIFORGE_API_PORT"] = "9000"

        try:
            config = load_config()
            assert config.api.port == 9000
        finally:
            # Clean up
            if "FAIFORGE_API_PORT" in os.environ:
                del os.environ["FAIFORGE_API_PORT"]

    def test_cors_config(self):
        """Test CORS configuration is loaded correctly"""
        config = load_config()

        assert config.cors.enabled is True
        assert isinstance(config.cors.origins, list)
        assert "http://localhost:3000" in config.cors.origins

    def test_observability_config(self):
        """Test observability configuration"""
        config = load_config()

        assert config.observability.log_level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
        assert config.observability.log_format in ["json", "text"]
        assert config.observability.request_logging is True
