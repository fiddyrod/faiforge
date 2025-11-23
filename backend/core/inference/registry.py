import yaml
from typing import Dict, Optional
from pathlib import Path

# Import adapters (VLLMAdapter may be None if vllm not installed)
from .adapters import (
    LLMAdapter, 
    OpenAIAdapter, 
    AnthropicAdapter, 
    VLLMAdapter,
    VLLM_AVAILABLE
)

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
    anthropic_api_key: Optional[str] = None,
    load_vllm: bool = True
) -> ModelRegistry:
    """
    Load models from YAML configuration.
    
    Args:
        config_path: Path to models.yaml
        openai_api_key: OpenAI API key
        anthropic_api_key: Anthropic API key (optional)
        load_vllm: Whether to load vLLM models (can be slow on startup)
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
                print(f"⚠️: Skipping {model_name} - ANTHROPIC_API_KEY not set")
                continue
            adapter = AnthropicAdapter(
                api_key=anthropic_api_key,
                model=model_config["model"]
            )
            registry.register(model_name, adapter)

        elif adapter_type == "vllm":
            # Check if vLLM is installed
            if not VLLM_AVAILABLE:
                print(f"⚠️  Skipping {model_name} - vLLM not installed")
                continue
            
            # Check if vLLM loading is enabled
            if not load_vllm:
                print(f"⚠️  Skipping {model_name} - vLLM loading disabled")
                continue
            
            print(f"🔄 Loading vLLM model: {model_name}...")
            print(f"📋 Config: {model_config}")
            
            adapter = VLLMAdapter(
                model=model_config["model"],
                max_model_len=model_config.get("max_model_len", 2048),
                gpu_memory_utilization=model_config.get("gpu_memory_utilization", 0.9)
            )
            registry.register(model_name, adapter)
        
        else:
            print(f"⚠️: Unknown adapter type '{adapter_type}' for model '{model_name}'")
    
    return registry