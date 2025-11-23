import os
import yaml
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    workers: int = 1


@dataclass
class CORSConfig:
    """CORS configuration"""
    enabled: bool = True
    origins: List[str] = field(default_factory=lambda: ["http://localhost:3000"])
    allow_credentials: bool = True
    allow_methods: List[str] = field(default_factory=lambda: ["*"])
    allow_headers: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class DefaultsConfig:
    """Default model settings"""
    model: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 500


@dataclass
class ModelsConfig:
    """Model registry configuration"""
    config_path: str = "core/config/models.yaml"
    load_vllm: bool = True


@dataclass
class ObservabilityConfig:
    """Observability configuration"""
    log_level: str = "INFO"
    log_format: str = "json"
    request_logging: bool = True


@dataclass
class CacheConfig:
    """Cache configuration"""
    enabled: bool = False
    backend: str = "redis"
    ttl_seconds: int = 3600


@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    enabled: bool = False
    requests_per_minute: int = 60


@dataclass
class AppConfig:
    """Complete application configuration"""
    api: APIConfig
    cors: CORSConfig
    defaults: DefaultsConfig
    models: ModelsConfig
    observability: ObservabilityConfig
    cache: CacheConfig
    rate_limit: RateLimitConfig


def load_yaml(file_path: str) -> dict:
    """Load YAML file and return as dict"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    
    with open(path) as f:
        return yaml.safe_load(f) or {}


def merge_configs(base: dict, override: dict) -> dict:
    """Recursively merge override config into base config"""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def load_config(
    base_config_path: str = "core/config/app.yaml",
    environment: Optional[str] = None
) -> AppConfig:
    """
    Load application configuration.
    
    Args:
        base_config_path: Path to base config file
        environment: Environment name (development, production, etc.)
                    If None, uses ENV environment variable or defaults to 'development'
    
    Returns:
        AppConfig instance with loaded configuration
    """
    # Determine environment
    if environment is None:
        environment = os.getenv("ENV", "development")

    logger.info(f"Loading configuration for environment: {environment}")

    # Load base config
    base_config = load_yaml(base_config_path)

    # Load environment-specific overrides
    env_config_path = f"core/config/environments/{environment}.yaml"
    if Path(env_config_path).exists():
        env_config = load_yaml(env_config_path)
        logger.debug(f"Env config API port: {env_config.get('api', {}).get('port', 'NOT SET')}")
        config_dict = merge_configs(base_config, env_config)
        logger.debug(f"After merge API port: {config_dict.get('api', {}).get('port', 'NOT SET')}")
        logger.info(f"Loaded environment overrides from {env_config_path}")
    else:
        config_dict = base_config
        logger.warning(f"No environment config found at {env_config_path}, using base config")

    # Apply environment variable overrides
    config_dict = apply_env_overrides(config_dict)
    logger.debug(f"After env vars API port: {config_dict.get('api', {}).get('port', 'NOT SET')}")

    # Build config objects
    return AppConfig(
        api=APIConfig(**config_dict.get("api", {})),
        cors=CORSConfig(**config_dict.get("cors", {})),
        defaults=DefaultsConfig(**config_dict.get("defaults", {})),
        models=ModelsConfig(**config_dict.get("models", {})),
        observability=ObservabilityConfig(**config_dict.get("observability", {})),
        cache=CacheConfig(**config_dict.get("cache", {})),
        rate_limit=RateLimitConfig(**config_dict.get("rate_limit", {}))
    )


def apply_env_overrides(config: dict) -> dict:
    """
    Apply environment variable overrides.
    
    Environment variables follow pattern: FAIFORGE_SECTION_KEY
    Example: FAIFORGE_API_PORT=8080
    """
    overrides = {}
    prefix = "FAIFORGE_"

    logger.debug(f"Scanning for {prefix}* environment variables")

    for key, value in os.environ.items():
        if key.startswith(prefix):
            logger.debug(f"Found env var: {key} = {value}")

            # Remove prefix and split into section and key
            parts = key[len(prefix):].lower().split("_", 1)
            if len(parts) == 2:
                section, config_key = parts

                if section not in overrides:
                    overrides[section] = {}

                # Convert value to appropriate type
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False
                elif value.isdigit():
                    value = int(value)

                overrides[section][config_key] = value
                logger.info(f"Config override: {section}.{config_key} = {value}")

    if not overrides:
        logger.debug(f"No {prefix}* variables found")
    
    # Merge overrides into config
    for section, values in overrides.items():
        if section in config:
            if isinstance(config[section], dict):
                config[section].update(values)
    
    return config


# Convenience function
def get_config() -> AppConfig:
    """Get application configuration (cached)"""
    if not hasattr(get_config, "_config"):
        get_config._config = load_config()
    return get_config._config