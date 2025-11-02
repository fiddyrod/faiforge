import yaml
from typing import Dict
from pathlib import Path
from .adapters import LLMAdapter, OpenAIAdapter, AnthropicAdapter


class ModelRegistry:
    """Registry for managing available models"""
    
    def __init__(self):
        self._models: Dict[str, LLMAdapter] = {}
    
    def register(self, name: str, adapter: LLMAdapter):
        """Register a model with the registry"""
        self._models[name] = adapter
    
    def get(self, name: str) -> LLMAdapter:
        """Get a model adapter by name"""
        if name not in self._models:
            available = ", ".join(self._models.keys())
            raise ValueError(f"Model '{name}' not found. Available: {available}")
        return self._models[name]
    
    def list(self) -> list:
        """List all available model names"""
        return list(self._models.keys())


def load_registry(
    config_path: str,
    openai_api_key: str,
    anthropic_api_key: str = None
) -> ModelRegistry:
    """
    Load models from YAML configuration.
    
    Args:
        config_path: Path to models.yaml
        openai_api_key: OpenAI API key
        anthropic_api_key: Anthropic API key (optional)
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    registry = ModelRegistry()
    
    for model_name, model_config in config["models"].items():
        adapter_type = model_config["adapter"]
        
        if adapter_type == "openai":
            adapter = OpenAIAdapter(
                api_key=openai_api_key,
                model=model_config["model"]
            )
            registry.register(model_name, adapter)
        
        elif adapter_type == "anthropic":
            if not anthropic_api_key:
                print(f"Warning: Skipping {model_name} - ANTHROPIC_API_KEY not set")
                continue
            adapter = AnthropicAdapter(
                api_key=anthropic_api_key,
                model=model_config["model"]
            )
            registry.register(model_name, adapter)
        
        else:
            print(f"Warning: Unknown adapter type '{adapter_type}' for model '{model_name}'")
    
    return registry