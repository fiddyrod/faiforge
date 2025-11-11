"""
Streaming Example for FAIForge

This example shows how to:
1. Stream responses in real-time
2. Display content as it arrives
3. Handle completion metadata

Run this:
    python examples/streaming.py
"""

import asyncio
import os
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from core.inference.registry import get_adapter
from core.config.loader import load_config


async def stream_example():
    """Streaming completion example."""
    
    print("🔥 FAIForge - Streaming Example\n")
    
    # Load configuration
    config = load_config()
    
    # Get adapter
    print("📡 Initializing OpenAI adapter...")
    adapter = get_adapter("openai", config["providers"]["openai"])
    
    # Create streaming request
    request = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user", 
                "content": "Write a short poem about artificial intelligence (3-4 lines)"
            }
        ],
        "temperature": 0.8,
        "max_tokens": 200,
        "stream": True  # ← Enable streaming
    }
    
    print(f"💬 Streaming from {request['model']}...")
    print(f"   Prompt: '{request['messages'][0]['content']}'\n")
    print("━" * 60)
    print("📝 Response (streaming):\n")
    
    # Track metrics
    start_time = time.time()
    first_token_time = None
    chunk_count = 0
    total_content = ""
    
    # Stream the response
    async for chunk in adapter.complete_stream(request):
        # Record time to first token
        if first_token_time is None and chunk.get("content"):
            first_token_time = time.time() - start_time
        
        # Print content as it arrives (no newline)
        if chunk.get("content"):
            print(chunk["content"], end="", flush=True)
            total_content += chunk["content"]
            chunk_count += 1
        
        # Check if streaming is done
        if chunk.get("finish_reason"):
            print("\n")
            break
    
    total_time = time.time() - start_time
    
    # Display metadata
    print("━" * 60)
    print(f"\n📊 Streaming Metrics:")
    print(f"   Chunks received: {chunk_count}")
    print(f"   Time to first token: {first_token_time:.2f}s")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Characters: {len(total_content)}")
    
    print("\n✨ Done! Notice how the response appeared in real-time.\n")


async def compare_streaming_vs_non_streaming():
    """Compare streaming vs non-streaming performance."""
    
    print("\n" + "=" * 60)
    print("🏎️  Streaming vs Non-Streaming Comparison")
    print("=" * 60 + "\n")
    
    config = load_config()
    adapter = get_adapter("openai", config["providers"]["openai"])
    
    prompt = "Explain async/await in Python in one paragraph"
    
    # Non-streaming request
    print("1️⃣  Non-Streaming Request:\n")
    start = time.time()
    
    response = await adapter.complete({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False
    })
    
    non_streaming_time = time.time() - start
    
    print(f"   ⏱️  Total time: {non_streaming_time:.2f}s")
    print(f"   📝 Response length: {len(response['content'])} chars")
    print(f"   💭 User experience: Wait {non_streaming_time:.2f}s, then see full response\n")
    
    # Streaming request
    print("2️⃣  Streaming Request:\n")
    start = time.time()
    first_token = None
    
    async for chunk in adapter.complete_stream({
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }):
        if first_token is None and chunk.get("content"):
            first_token = time.time() - start
        
        if chunk.get("finish_reason"):
            break
    
    streaming_time = time.time() - start
    
    print(f"   ⏱️  Time to first token: {first_token:.2f}s")
    print(f"   ⏱️  Total time: {streaming_time:.2f}s")
    print(f"   💭 User experience: See response after {first_token:.2f}s\n")
    
    # Analysis
    print("📊 Analysis:")
    print(f"   • Streaming feels {non_streaming_time/first_token:.1f}x faster")
    print(f"   • User sees content {non_streaming_time - first_token:.2f}s earlier")
    print(f"   • Total time similar (~{abs(streaming_time - non_streaming_time):.2f}s difference)")
    print("\n💡 Conclusion: Streaming dramatically improves perceived performance!\n")


async def main():
    """Run examples."""
    
    # Basic streaming example
    await stream_example()
    
    # Comparison
    await compare_streaming_vs_non_streaming()


if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("   Add it to your .env file or export it:")
        print("   export OPENAI_API_KEY='your-key-here'")
        sys.exit(1)
    
    asyncio.run(main())
