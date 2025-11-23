# Adding LLM Providers

Quick guide to adding new LLM providers using the adapter pattern.

**Time needed:** ~30 minutes

---

## Quick Start

Four steps to add a provider:

1. Create adapter class in `core/inference/adapters/`
2. Register in `core/inference/registry.py`
3. Configure in `config/models.yaml`
4. Add API key to `.env`

---

## Example: Adding Cohere

### 1. Create Adapter

`core/inference/adapters/cohere_adapter.py`:

```python
from .base import BaseAdapter
import httpx

class CohereAdapter(BaseAdapter):
    def __init__(self, api_key: str, model: str, timeout: float = 60.0):
        self.client = httpx.AsyncClient(
            base_url="https://api.cohere.ai",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout
        )
        self.model = model

    async def complete(self, messages, temperature=0.7, max_tokens=500):
        # Transform to Cohere format
        response = await self.client.post("/v1/chat", json={
            "model": self.model,
            "message": messages[-1]["content"],
            "temperature": temperature,
            "max_tokens": max_tokens
        })

        data = response.json()

        # Transform to standard format
        return {
            "content": data["text"],
            "usage": {
                "prompt_tokens": data["meta"]["tokens"]["input_tokens"],
                "completion_tokens": data["meta"]["tokens"]["output_tokens"]
            }
        }
```

### 2. Register

`core/inference/registry.py`:

```python
from .adapters.cohere_adapter import CohereAdapter

ADAPTERS = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "cohere": CohereAdapter,  # Add this
}
```

### 3. Configure

`config/models.yaml`:

```yaml
models:
  command-r:
    adapter: cohere
    model: command-r
```

### 4. Add API Key

`.env`:

```bash
COHERE_API_KEY=your-key-here
```

Done! Your provider is ready to use.

---

## Key Patterns

### Request Transformation

Each provider has different API formats. Transform in your adapter:

```python
# Our format → Provider format
def _to_provider_format(self, messages):
    return {"message": messages[-1]["content"]}
```

### Response Transformation

All adapters return the same format:

```python
# Provider format → Our format
return {
    "content": "...",
    "usage": {"prompt_tokens": 10, "completion_tokens": 20}
}
```

### Error Handling

Retry on transient errors:

```python
try:
    response = await client.post(...)
except httpx.HTTPStatusError as e:
    if e.response.status_code == 429:  # Rate limit
        await asyncio.sleep(2)
        # Retry logic
```

---

## Reference Existing Adapters

- **`openai_adapter.py`** - Standard REST API with retries
- **`anthropic_adapter.py`** - Different message format handling
- **`vllm_adapter.py`** - Local models (no API key)

---

## Troubleshooting

**Adapter not found?**
→ Check it's registered in `registry.py`

**API key not found?**
→ Check `.env` file has the key

**Timeout errors?**
→ Increase timeout in adapter init

---

That's it. The adapter pattern keeps things simple.
