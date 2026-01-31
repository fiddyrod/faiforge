"""
Model Registry with Fallback Support

Provides:
- Loading models from YAML configuration
- Fallback chain setup
- Smart routing integration
"""

import yaml
from typing import Dict, Optional, List, Any
from pathlib import Path
import logging

from .adapters import (
    LLMAdapter,
    OpenAIAdapter,
    AnthropicAdapter,
    VLLMAdapter,
    VLLM_AVAILABLE
)
from .fallback import (
    FallbackAdapter,
    FallbackConfig,
    RetryConfig,
    SmartRouter
)

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for managing available models and fallback chains"""

    def __init__(self):
        self._models: Dict[str, LLMAdapter] = {}
        self._fallback_chains: Dict[str, FallbackAdapter] = {}
        self._router: Optional[SmartRouter] = None

    def register(self, name: str, adapter: LLMAdapter):
        """Register a model with the registry"""
        self._models[name] = adapter

    def register_fallback_chain(self, name: str, adapter: FallbackAdapter):
        """Register a fallback chain"""
        self._fallback_chains[name] = adapter

    def set_router(self, router: SmartRouter):
        """Set the smart router"""
        self._router = router

    def get(self, name: str) -> LLMAdapter:
        """
        Get a model adapter by name.

        Checks fallback chains first, then individual models.
        """
        # Check fallback chains first
        if name in self._fallback_chains:
            return self._fallback_chains[name]

        # Then check individual models
        if name in self._models:
            return self._models[name]

        available = list(self._models.keys()) + list(self._fallback_chains.keys())
        raise ValueError(f"Model '{name}' not found. Available: {', '.join(available)}")

    def get_routed(
        self,
        messages: List[Any],
        max_tokens: int = 500,
        tools: Optional[List[Any]] = None
    ) -> LLMAdapter:
        """
        Get adapter based on smart routing rules.

        Falls back to default model if no router configured.
        """
        if self._router:
            return self._router.route(messages, max_tokens, tools)

        # No router - return first fallback chain or first model
        if self._fallback_chains:
            return list(self._fallback_chains.values())[0]
        return list(self._models.values())[0]

    def list(self) -> list:
        """List all available model and chain names"""
        return list(self._models.keys()) + list(self._fallback_chains.keys())

    def list_models(self) -> list:
        """List individual model names only"""
        return list(self._models.keys())

    def list_chains(self) -> list:
        """List fallback chain names only"""
        return list(self._fallback_chains.keys())

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all fallback chains"""
        return {
            name: chain.get_health_status()
            for name, chain in self._fallback_chains.items()
        }


def _create_adapter(
    adapter_type: str,
    model_config: Dict[str, Any],
    openai_api_key: str,
    anthropic_api_key: Optional[str],
    load_vllm: bool
) -> Optional[LLMAdapter]:
    """Create a single adapter based on type"""

    if adapter_type == "openai":
        return OpenAIAdapter(
            api_key=openai_api_key,
            model=model_config["model"]
        )

    elif adapter_type == "anthropic":
        if not anthropic_api_key:
            logger.warning(f"Skipping Anthropic model - ANTHROPIC_API_KEY not set")
            return None
        return AnthropicAdapter(
            api_key=anthropic_api_key,
            model=model_config["model"]
        )

    elif adapter_type == "vllm":
        if not VLLM_AVAILABLE:
            logger.warning(f"Skipping vLLM model - vLLM not installed")
            return None
        if not load_vllm:
            logger.info(f"Skipping vLLM model - vLLM loading disabled")
            return None

        return VLLMAdapter(
            model=model_config["model"],
            max_model_len=model_config.get("max_model_len", 2048),
            gpu_memory_utilization=model_config.get("gpu_memory_utilization", 0.9)
        )

    else:
        logger.warning(f"Unknown adapter type: {adapter_type}")
        return None


def _load_fallback_chains(
    routing_config: Dict[str, Any],
    adapters: Dict[str, LLMAdapter]
) -> Dict[str, FallbackAdapter]:
    """Load fallback chains from routing config"""

    chains = {}
    chain_configs = routing_config.get("fallback_chains", {})

    for chain_name, chain_config in chain_configs.items():
        model_names = chain_config.get("models", [])

        # Filter to available models only
        available_models = {
            name: adapters[name]
            for name in model_names
            if name in adapters
        }

        if not available_models:
            logger.warning(f"Skipping fallback chain '{chain_name}' - no available models")
            continue

        # Build retry config
        retry_cfg = chain_config.get("retry", {})
        retry_config = RetryConfig(
            max_retries=retry_cfg.get("max_retries", 3),
            base_delay=retry_cfg.get("base_delay", 1.0),
            max_delay=retry_cfg.get("max_delay", 30.0)
        )

        # Build circuit breaker config
        cb_cfg = chain_config.get("circuit_breaker", {})
        fallback_config = FallbackConfig(
            chain=[n for n in model_names if n in adapters],
            retry=retry_config,
            circuit_breaker_threshold=cb_cfg.get("threshold", 5),
            circuit_breaker_recovery=cb_cfg.get("recovery", 300.0)
        )

        # Create fallback adapter
        fallback_adapter = FallbackAdapter(
            adapters=available_models,
            config=fallback_config,
            name=chain_name
        )

        chains[chain_name] = fallback_adapter
        logger.info(f"Loaded fallback chain '{chain_name}': {fallback_config.chain}")

    return chains


def _load_router(
    routing_config: Dict[str, Any],
    adapters: Dict[str, LLMAdapter],
    fallback_chains: Dict[str, FallbackAdapter]
) -> Optional[SmartRouter]:
    """Load smart router from config"""

    routing_settings = routing_config.get("routing", {})

    if not routing_settings.get("enabled", False):
        return None

    default_chain = routing_settings.get("default_chain", "default")
    rules = routing_settings.get("rules", [])

    # Convert rules to router format
    router_rules = []
    for rule in rules:
        router_rules.append({
            "condition": rule.get("condition"),
            "fallback_group": rule.get("fallback_chain"),
            "model": rule.get("model")
        })

    router = SmartRouter(
        adapters=adapters,
        fallback_adapters=fallback_chains,
        default_model=default_chain,
        rules=router_rules
    )

    logger.info(f"Smart router enabled with {len(router_rules)} rules, default: {default_chain}")
    return router


def load_registry(
    config_path: str,
    openai_api_key: str,
    anthropic_api_key: Optional[str] = None,
    load_vllm: bool = True,
    routing_config_path: Optional[str] = None
) -> ModelRegistry:
    """
    Load models from YAML configuration with optional fallback/routing support.

    Args:
        config_path: Path to models.yaml
        openai_api_key: OpenAI API key
        anthropic_api_key: Anthropic API key (optional)
        load_vllm: Whether to load vLLM models
        routing_config_path: Path to routing.yaml (optional)
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    registry = ModelRegistry()

    # Load individual models
    for model_name, model_config in config.get("models", {}).items():
        adapter_type = model_config.get("adapter")

        adapter = _create_adapter(
            adapter_type=adapter_type,
            model_config=model_config,
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            load_vllm=load_vllm
        )

        if adapter:
            registry.register(model_name, adapter)
            logger.info(f"Registered model: {model_name}")

    # Load routing config if provided
    if routing_config_path:
        routing_file = Path(routing_config_path)
        if routing_file.exists():
            with open(routing_file) as f:
                routing_config = yaml.safe_load(f)

            # Load fallback chains
            fallback_chains = _load_fallback_chains(routing_config, registry._models)
            for name, chain in fallback_chains.items():
                registry.register_fallback_chain(name, chain)

            # Load smart router
            router = _load_router(routing_config, registry._models, fallback_chains)
            if router:
                registry.set_router(router)
        else:
            logger.warning(f"Routing config not found: {routing_config_path}")

    return registry
