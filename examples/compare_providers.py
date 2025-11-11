"""
Multi-Provider Comparison Example for FAIForge

This example shows how to:
1. Use multiple providers with the same code
2. Compare responses across providers
3. Analyze cost and performance differences

This demonstrates the power of the adapter pattern!

Run this:
    python examples/compare_providers.py
"""

import asyncio
import os
from pathlib import Path
import sys
import time
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from core.inference.registry import get_adapter
from core.config.loader import load_config


async def compare_providers(prompt: str, providers: List[str]):
    """
    Run the same prompt across multiple providers and compare.
    
    This is the magic of the adapter pattern: same code works
    with any provider!
    """
    
    print(f"\n{'=' * 70}")
    print(f"🔬 Comparing Providers")
    print(f"{'=' * 70}\n")
    
    print(f"📝 Prompt: \"{prompt}\"\n")
    
    config = load_config()
    results = []
    
    for provider_name in providers:
        print(f"{'─' * 70}")
        print(f"🔌 Testing {provider_name.upper()}...")
        print(f"{'─' * 70}\n")
        
        try:
            # Get adapter for this provider
            adapter = get_adapter(
                provider_name, 
                config["providers"][provider_name]
            )
            
            # Determine best model for this provider
            models = {
                "openai": "gpt-4o-mini",
                "anthropic": "claude-3-5-sonnet-20241022",
                "vllm": "meta-llama/Llama-3-8B"
            }
            
            model = models.get(provider_name, "default")
            
            # Make request (same code for all providers!)
            start_time = time.time()
            
            response = await adapter.complete({
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 200,
                "stream": False
            })
            
            elapsed_time = time.time() - start_time
            
            # Store results
            results.append({
                "provider": provider_name,
                "model": response["model"],
                "content": response["content"],
                "tokens": response["usage"]["total_tokens"],
                "cost": response["cost"],
                "latency": elapsed_time
            })
            
            # Display response
            print(f"✅ Response from {provider_name}:\n")
            print(f"{response['content']}\n")
            print(f"📊 Stats:")
            print(f"   • Latency: {elapsed_time:.2f}s")
            print(f"   • Tokens: {response['usage']['total_tokens']}")
            print(f"   • Cost: ${response['cost']:.6f}\n")
            
        except Exception as e:
            print(f"❌ Error with {provider_name}: {str(e)}\n")
            continue
    
    # Display comparison summary
    if len(results) > 1:
        print(f"\n{'=' * 70}")
        print(f"📊 COMPARISON SUMMARY")
        print(f"{'=' * 70}\n")
        
        # Latency comparison
        print("⚡ Latency:")
        for r in sorted(results, key=lambda x: x["latency"]):
            print(f"   {r['provider']:12} {r['latency']:6.2f}s {'🏆' if r == results[0] else ''}")
        
        # Cost comparison
        print("\n💰 Cost:")
        for r in sorted(results, key=lambda x: x["cost"]):
            cost_str = f"${r['cost']:.6f}" if r['cost'] > 0 else "FREE"
            print(f"   {r['provider']:12} {cost_str:>12} {'🏆' if r['cost'] == 0 else ''}")
        
        # Token usage
        print("\n📝 Tokens:")
        for r in sorted(results, key=lambda x: x["tokens"]):
            print(f"   {r['provider']:12} {r['tokens']:>6} tokens")
        
        print()


async def test_adapter_pattern():
    """
    Demonstrate the adapter pattern with a single function
    that works with any provider.
    """
    
    print(f"\n{'=' * 70}")
    print(f"🎯 Adapter Pattern Demo")
    print(f"{'=' * 70}\n")
    
    print("The power of FAIForge: Write once, run anywhere!\n")
    
    config = load_config()
    
    # This function works with ANY provider
    async def get_completion(provider: str, model: str, prompt: str) -> str:
        """
        Generic completion function.
        Works with OpenAI, Anthropic, vLLM, or any future provider!
        """
        adapter = get_adapter(provider, config["providers"][provider])
        
        response = await adapter.complete({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        })
        
        return response["content"]
    
    # Use it with different providers - same code!
    prompt = "Say 'Hello World' in a creative way"
    
    print(f"📝 Prompt: \"{prompt}\"\n")
    
    # Test with OpenAI
    print("1️⃣  Using OpenAI:")
    result = await get_completion("openai", "gpt-4o-mini", prompt)
    print(f"   {result}\n")
    
    # Test with Anthropic (if configured)
    if os.getenv("ANTHROPIC_API_KEY"):
        print("2️⃣  Using Anthropic:")
        result = await get_completion(
            "anthropic", 
            "claude-3-5-sonnet-20241022", 
            prompt
        )
        print(f"   {result}\n")
    
    # Test with local model (if running)
    try:
        print("3️⃣  Using Local Model (vLLM):")
        result = await get_completion("vllm", "meta-llama/Llama-3-8B", prompt)
        print(f"   {result}\n")
    except:
        print("   ⚠️  Local model not running (skip)\n")
    
    print("✨ Same function, different providers. That's the adapter pattern!\n")


async def cost_optimization_example():
    """
    Show how to optimize costs by choosing the right provider/model.
    """
    
    print(f"\n{'=' * 70}")
    print(f"💰 Cost Optimization Example")
    print(f"{'=' * 70}\n")
    
    config = load_config()
    
    # Simple task that doesn't need GPT-4
    prompt = "What is 2+2?"
    
    print(f"📝 Task: \"{prompt}\"")
    print("💭 Question: Which model should we use?\n")
    
    # Test with expensive model
    print("1️⃣  Using GPT-4o (expensive):")
    adapter = get_adapter("openai", config["providers"]["openai"])
    
    response = await adapter.complete({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    })
    
    print(f"   Response: {response['content']}")
    print(f"   Cost: ${response['cost']:.6f}\n")
    
    expensive_cost = response['cost']
    
    # Test with cheap model
    print("2️⃣  Using GPT-4o-mini (cheap):")
    
    response = await adapter.complete({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    })
    
    print(f"   Response: {response['content']}")
    print(f"   Cost: ${response['cost']:.6f}\n")
    
    cheap_cost = response['cost']
    
    # Analysis
    savings = ((expensive_cost - cheap_cost) / expensive_cost) * 100
    
    print("📊 Analysis:")
    print(f"   • Both got the answer right")
    print(f"   • GPT-4o-mini saved {savings:.1f}% on cost")
    print(f"   • For 1M requests: save ${(expensive_cost - cheap_cost) * 1_000_000:.2f}")
    print("\n💡 Lesson: Use the right tool for the job!\n")


async def main():
    """Run all examples."""
    
    print("\n" + "🔥" * 35)
    print("🔥  FAIForge Multi-Provider Examples  🔥")
    print("🔥" * 35)
    
    # Determine which providers to test
    providers_to_test = []
    
    if os.getenv("OPENAI_API_KEY"):
        providers_to_test.append("openai")
    
    if os.getenv("ANTHROPIC_API_KEY"):
        providers_to_test.append("anthropic")
    
    # Check if local vLLM is running (optional)
    # providers_to_test.append("vllm")
    
    if not providers_to_test:
        print("\n❌ No API keys found!")
        print("   Set OPENAI_API_KEY or ANTHROPIC_API_KEY in your .env")
        return
    
    # Run examples
    await compare_providers(
        prompt="Explain recursion in one sentence",
        providers=providers_to_test
    )
    
    await test_adapter_pattern()
    
    if "openai" in providers_to_test:
        await cost_optimization_example()
    
    print("\n✨ All examples complete!\n")


if __name__ == "__main__":
    asyncio.run(main())
