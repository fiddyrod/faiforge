"""
Basic Usage Example for FAIForge

This example shows how to:
1. Initialize an adapter
2. Send a simple completion request
3. Handle the response

Run this:
    python examples/basic_usage.py
"""

import asyncio
import os
from pathlib import Path
import sys

# Add parent directory to path so we can import core
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from core.inference.registry import get_adapter
from core.config.loader import load_config


async def main():
    """Basic completion example."""
    
    print("🔥 FAIForge - Basic Usage Example\n")
    
    # Load configuration
    config = load_config()
    
    # Get OpenAI adapter (you can use "anthropic" or "vllm" too)
    print("📡 Initializing OpenAI adapter...")
    adapter = get_adapter("openai", config["providers"]["openai"])
    
    # Create a simple request
    request = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "Say hello in a creative way!"}
        ],
        "temperature": 0.7,
        "max_tokens": 100,
        "stream": False  # Non-streaming for simplicity
    }
    
    print(f"💬 Sending request to {request['model']}...")
    print(f"   Prompt: '{request['messages'][0]['content']}'\n")
    
    # Make the request
    response = await adapter.complete(request)
    
    # Display the response
    print("✅ Response received!\n")
    print("━" * 60)
    print(f"📝 Content:\n{response['content']}\n")
    print("━" * 60)
    print(f"\n📊 Metadata:")
    print(f"   Model: {response['model']}")
    print(f"   Tokens: {response['usage']['total_tokens']} "
          f"({response['usage']['prompt_tokens']} prompt + "
          f"{response['usage']['completion_tokens']} completion)")
    print(f"   Cost: ${response['cost']:.6f}")
    
    print("\n✨ Done! That's how easy it is.\n")


if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Add it to your .env file or export it:")
        print("   export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    asyncio.run(main())
