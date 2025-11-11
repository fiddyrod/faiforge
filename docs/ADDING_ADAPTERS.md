# 🔌 Adding New Model Providers to FAIForge

This guide walks you through adding a new LLM provider to FAIForge. You'll learn by example, see working code, and understand the patterns.

**Time to add a new provider:** ~30 minutes once you understand the pattern.

---

## 📋 Table of Contents

1. [Quick Overview](#quick-overview)
2. [The Adapter Pattern](#the-adapter-pattern)
3. [Step-by-Step Tutorial](#step-by-step-tutorial)
4. [Real Examples](#real-examples)
5. [Testing Your Adapter](#testing-your-adapter)
6. [Common Patterns](#common-patterns)
7. [Troubleshooting](#troubleshooting)

---

## 🎯 Quick Overview

Adding a new provider involves 4 files:

```
1. core/inference/adapters/your_provider.py  ← Your adapter
2. core/inference/registry.py                ← Register it
3. config/models.yaml                        ← Configure it
4. tests/test_your_provider.py               ← Test it
```

**That's it.** No changes to the router, API layer, or core logic.

---

## 🏗️ The Adapter Pattern

### **What is it?**

The adapter pattern lets you add new providers without changing existing code. Each provider implements the same interface (`BaseAdapter`), so the rest of the system doesn't care which one it's using.

### **Why use it?**

**Without adapters:**
```python
# BAD: Router needs to know about every provider
if provider == "openai":
    response = call_openai(...)
elif provider == "anthropic":
    response = call_anthropic(...)
elif provider == "cohere":
    response = call_cohere(...)
# Need to modify this code every time you add a provider!
```

**With adapters:**
```python
# GOOD: Router just calls the adapter
adapter = get_adapter(provider)
response = await adapter.complete(request)
# Adding a new provider? No changes needed here!
```

---

## 📚 Step-by-Step Tutorial

Let's add **Cohere** as a new provider. Follow along!

### **Step 1: Create the Adapter File**

Create `core/inference/adapters/cohere_adapter.py`:

```python
"""
Cohere adapter for FAIForge.

Implements the BaseAdapter interface for Cohere's API.
Handles streaming, error handling, and cost tracking.
"""

import httpx
from typing import Dict, Any, AsyncIterator
from .base import BaseAdapter, CompletionRequest, CompletionResponse


class CohereAdapter(BaseAdapter):
    """
    Adapter for Cohere API.
    
    Cohere uses a different API structure than OpenAI:
    - Endpoint: /v1/chat
    - Streaming: different SSE format
    - Models: command-r, command-r-plus, etc.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Cohere adapter.
        
        Args:
            config: Configuration dict with:
                - api_key: Cohere API key
                - base_url: API base URL (default: https://api.cohere.ai)
                - timeout: Request timeout in seconds
                - max_retries: Number of retry attempts
        """
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://api.cohere.ai")
        self.timeout = config.get("timeout", 30)
        self.max_retries = config.get("max_retries", 3)
        
        if not self.api_key:
            raise ValueError("Cohere API key is required")
    
    async def complete(
        self, 
        request: CompletionRequest
    ) -> CompletionResponse:
        """
        Send a completion request to Cohere.
        
        This is the main entry point. It handles:
        1. Request transformation (our format → Cohere format)
        2. API call with retries
        3. Response transformation (Cohere format → our format)
        4. Cost calculation
        
        Args:
            request: Standardized completion request
            
        Returns:
            Standardized completion response
        """
        # Transform request to Cohere format
        cohere_request = self._transform_request(request)
        
        # Make API call with retry logic
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await self._make_request(
                client=client,
                endpoint="/v1/chat",
                payload=cohere_request
            )
        
        # Transform response to our standard format
        return self._transform_response(response, request)
    
    async def complete_stream(
        self, 
        request: CompletionRequest
    ) -> AsyncIterator[CompletionResponse]:
        """
        Stream completion responses from Cohere.
        
        Cohere's streaming format:
        - Server-sent events (SSE)
        - Each chunk: {"text": "...", "is_finished": false}
        - Final chunk: {"text": "", "is_finished": true, "meta": {...}}
        
        Yields:
            Standardized completion response chunks
        """
        cohere_request = self._transform_request(request)
        cohere_request["stream"] = True
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat",
                json=cohere_request,
                headers=self._get_headers()
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunk = self._parse_sse_chunk(line)
                        if chunk:
                            yield self._transform_stream_chunk(chunk)
    
    def _transform_request(self, request: CompletionRequest) -> Dict[str, Any]:
        """
        Transform our standard request to Cohere's format.
        
        Our format:
            {
                "model": "command-r",
                "messages": [{"role": "user", "content": "..."}],
                "temperature": 0.7,
                "max_tokens": 1000
            }
        
        Cohere format:
            {
                "model": "command-r",
                "message": "...",  # Last user message only
                "chat_history": [...],  # Previous messages
                "temperature": 0.7,
                "max_tokens": 1000
            }
        """
        # Extract last user message
        last_message = request["messages"][-1]["content"]
        
        # Extract chat history (all except last message)
        chat_history = []
        for msg in request["messages"][:-1]:
            chat_history.append({
                "role": "USER" if msg["role"] == "user" else "CHATBOT",
                "message": msg["content"]
            })
        
        return {
            "model": request["model"],
            "message": last_message,
            "chat_history": chat_history,
            "temperature": request.get("temperature", 0.7),
            "max_tokens": request.get("max_tokens", 1000),
        }
    
    def _transform_response(
        self, 
        response: Dict[str, Any],
        original_request: CompletionRequest
    ) -> CompletionResponse:
        """
        Transform Cohere's response to our standard format.
        
        Cohere response:
            {
                "text": "...",
                "meta": {
                    "tokens": {"input_tokens": 10, "output_tokens": 50}
                }
            }
        
        Our format:
            {
                "content": "...",
                "model": "command-r",
                "usage": {"prompt_tokens": 10, "completion_tokens": 50},
                "cost": 0.0015
            }
        """
        usage = response.get("meta", {}).get("tokens", {})
        
        return {
            "content": response["text"],
            "model": original_request["model"],
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": (
                    usage.get("input_tokens", 0) + 
                    usage.get("output_tokens", 0)
                )
            },
            "cost": self._calculate_cost(usage, original_request["model"])
        }
    
    def _transform_stream_chunk(
        self, 
        chunk: Dict[str, Any]
    ) -> CompletionResponse:
        """Transform a streaming chunk to our format."""
        return {
            "content": chunk.get("text", ""),
            "finish_reason": "stop" if chunk.get("is_finished") else None
        }
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for Cohere API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    async def _make_request(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        payload: Dict[str, Any],
        retry_count: int = 0
    ) -> Dict[str, Any]:
        """
        Make HTTP request with exponential backoff retry.
        
        Handles transient errors (rate limits, timeouts) by retrying.
        Permanent errors (auth, invalid request) fail immediately.
        """
        try:
            response = await client.post(
                f"{self.base_url}{endpoint}",
                json=payload,
                headers=self._get_headers()
            )
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPStatusError as e:
            # Rate limit (429) or server error (5xx) - retry
            if e.response.status_code in [429, 500, 502, 503, 504]:
                if retry_count < self.max_retries:
                    wait_time = 2 ** retry_count  # Exponential backoff
                    await asyncio.sleep(wait_time)
                    return await self._make_request(
                        client, endpoint, payload, retry_count + 1
                    )
            # Other errors - fail immediately
            raise
        
        except httpx.TimeoutException:
            # Timeout - retry
            if retry_count < self.max_retries:
                return await self._make_request(
                    client, endpoint, payload, retry_count + 1
                )
            raise
    
    def _calculate_cost(
        self, 
        usage: Dict[str, int], 
        model: str
    ) -> float:
        """
        Calculate cost based on token usage.
        
        Cohere pricing (as of Dec 2024):
        - command-r: $0.50 per 1M input, $1.50 per 1M output
        - command-r-plus: $3.00 per 1M input, $15.00 per 1M output
        """
        pricing = {
            "command-r": {
                "input": 0.50 / 1_000_000,
                "output": 1.50 / 1_000_000
            },
            "command-r-plus": {
                "input": 3.00 / 1_000_000,
                "output": 15.00 / 1_000_000
            }
        }
        
        if model not in pricing:
            return 0.0
        
        input_cost = usage.get("input_tokens", 0) * pricing[model]["input"]
        output_cost = usage.get("output_tokens", 0) * pricing[model]["output"]
        
        return input_cost + output_cost
    
    def _parse_sse_chunk(self, line: str) -> Dict[str, Any]:
        """Parse a server-sent event line."""
        if not line.startswith("data: "):
            return None
        
        try:
            data = line[6:]  # Remove "data: " prefix
            return json.loads(data)
        except json.JSONDecodeError:
            return None
```

---

### **Step 2: Register the Adapter**

Edit `core/inference/registry.py`:

```python
from .adapters.openai_adapter import OpenAIAdapter
from .adapters.anthropic_adapter import AnthropicAdapter
from .adapters.vllm_adapter import VLLMAdapter
from .adapters.cohere_adapter import CohereAdapter  # ← Add this

# Adapter registry
ADAPTERS = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "vllm": VLLMAdapter,
    "cohere": CohereAdapter,  # ← Add this
}

def get_adapter(provider: str, config: dict):
    """Get adapter instance for provider."""
    if provider not in ADAPTERS:
        raise ValueError(f"Unknown provider: {provider}")
    
    return ADAPTERS[provider](config)
```

---

### **Step 3: Add Configuration**

Edit `config/models.yaml`:

```yaml
providers:
  # ... existing providers ...
  
  cohere:
    api_key: ${COHERE_API_KEY}
    base_url: "https://api.cohere.ai"
    timeout: 30
    max_retries: 3
    
    models:
      - name: "command-r"
        context_window: 128000
        cost_per_1k_input: 0.0005
        cost_per_1k_output: 0.0015
      
      - name: "command-r-plus"
        context_window: 128000
        cost_per_1k_input: 0.003
        cost_per_1k_output: 0.015
```

Add to `.env`:
```bash
COHERE_API_KEY=your_key_here
```

---

### **Step 4: Test Your Adapter**

Create `tests/test_cohere_adapter.py`:

```python
import pytest
from core.inference.adapters.cohere_adapter import CohereAdapter

@pytest.mark.asyncio
async def test_cohere_basic_completion():
    """Test basic completion request."""
    config = {
        "api_key": "test_key",
        "base_url": "https://api.cohere.ai"
    }
    
    adapter = CohereAdapter(config)
    
    request = {
        "model": "command-r",
        "messages": [
            {"role": "user", "content": "Say hello"}
        ]
    }
    
    # This will call the real API
    # For unit tests, you'd mock the HTTP client
    response = await adapter.complete(request)
    
    assert "content" in response
    assert response["model"] == "command-r"
    assert "usage" in response
    assert "cost" in response

@pytest.mark.asyncio
async def test_cohere_streaming():
    """Test streaming completion."""
    # Similar to above, but test streaming
    pass
```

---

## 🔍 Real Examples

FAIForge includes 3 working adapters you can learn from:

### **1. OpenAI Adapter** (`openai_adapter.py`)
**Learn from this for:**
- Standard REST API structure
- Retry logic with exponential backoff
- Cost calculation
- Error handling

**Key patterns:**
```python
# Error handling with retries
async def _make_request(self, retry_count=0):
    try:
        response = await client.post(...)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:  # Rate limit
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count
                await asyncio.sleep(wait_time)
                return await self._make_request(retry_count + 1)
        raise
```

### **2. Anthropic Adapter** (`anthropic_adapter.py`)
**Learn from this for:**
- Different message format (system vs user/assistant)
- Different streaming format
- Provider-specific quirks

**Key differences from OpenAI:**
```python
# Anthropic requires separate system prompt
def _transform_request(self, request):
    system_message = None
    messages = []
    
    for msg in request["messages"]:
        if msg["role"] == "system":
            system_message = msg["content"]
        else:
            messages.append(msg)
    
    return {
        "model": request["model"],
        "system": system_message,  # ← Separate field
        "messages": messages
    }
```

### **3. vLLM Adapter** (`vllm_adapter.py`)
**Learn from this for:**
- Local model handling
- No API key needed
- Different performance characteristics

**Key differences:**
```python
# Local models have different endpoints
self.base_url = config.get("base_url", "http://localhost:8000")

# No cost calculation for local models
def _calculate_cost(self, usage, model):
    return 0.0  # Local models are free!
```

---

## 🛠️ Common Patterns

### **Pattern 1: Request Transformation**

Most providers have different request formats. The pattern:

```python
def _transform_request(self, request: CompletionRequest) -> Dict:
    """
    Convert our standard format to provider-specific format.
    
    This isolates provider differences from the rest of the system.
    """
    # Extract what you need from request
    # Transform to provider format
    # Return transformed request
```

### **Pattern 2: Response Transformation**

Similarly for responses:

```python
def _transform_response(self, response: Dict) -> CompletionResponse:
    """
    Convert provider response to our standard format.
    
    All adapters return the same format, regardless of provider.
    """
    return {
        "content": response["text"],
        "model": response["model"],
        "usage": {...},
        "cost": self._calculate_cost(...)
    }
```

### **Pattern 3: Error Handling**

Consistent error handling across providers:

```python
async def _make_request(self, retry_count=0):
    try:
        response = await client.post(...)
        response.raise_for_status()
        return response.json()
    
    except httpx.HTTPStatusError as e:
        # Retry on transient errors
        if should_retry(e.response.status_code):
            if retry_count < self.max_retries:
                await asyncio.sleep(2 ** retry_count)
                return await self._make_request(retry_count + 1)
        raise
    
    except httpx.TimeoutException:
        # Retry on timeout
        if retry_count < self.max_retries:
            return await self._make_request(retry_count + 1)
        raise
```

### **Pattern 4: Streaming**

Streaming responses are async iterators:

```python
async def complete_stream(self, request) -> AsyncIterator:
    """
    Stream responses chunk by chunk.
    
    Use async generators for efficient streaming.
    """
    async with client.stream("POST", ...) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                chunk = parse_chunk(line)
                yield self._transform_stream_chunk(chunk)
```

---

## 🐛 Troubleshooting

### **Issue: "Adapter not found"**
**Cause:** Forgot to register in `registry.py`

**Solution:**
```python
# In registry.py
ADAPTERS = {
    "your_provider": YourProviderAdapter,  # ← Add this
}
```

### **Issue: "API key not found"**
**Cause:** Environment variable not set

**Solution:**
```bash
# In .env
YOUR_PROVIDER_API_KEY=your_key_here

# In config/models.yaml
providers:
  your_provider:
    api_key: ${YOUR_PROVIDER_API_KEY}  # ← Use ${VAR} syntax
```

### **Issue: "Streaming not working"**
**Cause:** Not using async iterator correctly

**Solution:**
```python
# Wrong
async def complete_stream(self):
    chunks = []
    async for chunk in ...:
        chunks.append(chunk)
    return chunks  # ❌ Returns list, not iterator

# Right
async def complete_stream(self):
    async for chunk in ...:
        yield chunk  # ✅ Yields chunks as they arrive
```

### **Issue: "Timeout errors"**
**Cause:** Default timeout too short for large responses

**Solution:**
```python
# In your adapter __init__
self.timeout = config.get("timeout", 60)  # Increase default

# Or in config/models.yaml
providers:
  your_provider:
    timeout: 120  # 2 minutes
```

---

## 📝 Checklist

Before submitting your adapter:

- [ ] Inherits from `BaseAdapter`
- [ ] Implements `complete()` method
- [ ] Implements `complete_stream()` method (if supported)
- [ ] Request/response transformation
- [ ] Error handling with retries
- [ ] Cost calculation
- [ ] Registered in `registry.py`
- [ ] Configuration in `models.yaml`
- [ ] Environment variables documented
- [ ] Tests written
- [ ] Documentation added

---

## 🎯 Summary

Adding a new provider is straightforward:

1. **Create adapter** - Implement `BaseAdapter` interface
2. **Register it** - Add to `registry.py`
3. **Configure it** - Add to `config/models.yaml`
4. **Test it** - Write tests

**The adapter pattern keeps your codebase clean.** New providers don't touch existing code.

---

## 🤝 Need Help?

- Check existing adapters for examples
- Open a GitHub issue
- Ask in discussions

**Happy coding!** 🚀
