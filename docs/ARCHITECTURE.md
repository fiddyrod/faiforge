# 🏗️ FAIForge Architecture

This document explains FAIForge's architecture, design decisions, and patterns.

---

## 📐 System Overview

```
┌─────────────────────────────────────────────────────┐
│                Application Layer                     │
│         (Your code using FAIForge)                   │
└────────────────────┬────────────────────────────────┘
                     │ HTTP/REST
┌────────────────────▼────────────────────────────────┐
│               FastAPI Server                         │
│                  (Port 8000)                         │
│  ┌───────────────────────────────────────────────┐  │
│  │           API Routes Layer                    │  │
│  │  • /v1/completions                            │  │
│  │  • /health                                    │  │
│  │  • /models                                    │  │
│  └─────────────────────┬─────────────────────────┘  │
│                        │                             │
│  ┌─────────────────────▼─────────────────────────┐  │
│  │         Inference Router                      │  │
│  │  • Request validation (Pydantic)              │  │
│  │  • Provider selection                         │  │
│  │  • Error handling                             │  │
│  │  • Logging & tracing                          │  │
│  └─────────────────────┬─────────────────────────┘  │
│                        │                             │
│  ┌─────────────────────▼─────────────────────────┐  │
│  │          Adapter Registry                     │  │
│  │  • Provider lookup                            │  │
│  │  • Adapter instantiation                      │  │
│  │  • Configuration injection                    │  │
│  └─────────────────────┬─────────────────────────┘  │
│                        │                             │
│  ┌────────┬───────────┴───────────┬────────────┐   │
│  │        │                       │            │   │
│  ▼        ▼                       ▼            ▼   │
│ ┌────┐  ┌────┐                ┌────┐      ┌────┐  │
│ │OpenAI│ │Anthropic│           │vLLM│      │Future│ │
│ │Adapter│ │ Adapter │           │Adapter│    │Adapters│ │
│ └────┘  └────┘                └────┘      └────┘  │
└──────┬──────┬───────────────────┬────────────┬─────┘
       │      │                   │            │
   ┌───▼──┐ ┌─▼────┐         ┌───▼───┐    ┌───▼───┐
   │OpenAI│ │Anthropic│        │ Local │    │Future │
   │  API │ │  API  │        │ Model │    │Provider│
   └──────┘ └───────┘         └───────┘    └───────┘
```

---

## 🎯 Design Principles

### **1. Adapter Pattern**
**Problem:** Different LLM providers have different APIs.  
**Solution:** Each provider implements a common interface (`BaseAdapter`).

**Benefits:**
- ✅ Add new providers without modifying existing code
- ✅ Swap providers easily
- ✅ Test providers independently
- ✅ Consistent behavior across providers

**Example:**
```python
# Application code stays the same
adapter = get_adapter(provider_name, config)
response = await adapter.complete(request)

# Works with any provider!
```

### **2. Config-Driven Architecture**
**Problem:** Hardcoded values make systems inflexible.  
**Solution:** YAML configuration with environment overrides.

**Benefits:**
- ✅ Change behavior without code changes
- ✅ Environment-specific settings (dev/staging/prod)
- ✅ Version control configuration
- ✅ Easy to understand and modify

**Structure:**
```yaml
# config/default.yaml - Base configuration
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    models: [...]

# config/production.yaml - Overrides
defaults:
  model: "gpt-4o"  # Use better model in production
```

### **3. Async-First**
**Problem:** AI API calls are I/O-bound and can take seconds.  
**Solution:** Use async/await for non-blocking I/O.

**Benefits:**
- ✅ Handle multiple requests concurrently
- ✅ Don't block on slow API calls
- ✅ Better resource utilization
- ✅ Improved throughput

**Why it matters:**
```python
# Synchronous (blocking)
def complete(request):
    response = requests.post(...)  # Blocks entire thread
    return response

# 10 requests = 10 seconds (sequential)

# Asynchronous (non-blocking)
async def complete(request):
    response = await httpx.post(...)  # Doesn't block
    return response

# 10 requests = ~1 second (concurrent)
```

### **4. Separation of Concerns**
Each layer has a single responsibility:

| Layer | Responsibility |
|-------|----------------|
| **API Routes** | HTTP handling, request/response formatting |
| **Router** | Business logic, validation, orchestration |
| **Registry** | Provider management, adapter instantiation |
| **Adapters** | Provider-specific API integration |
| **Config** | System configuration |

**Benefits:**
- ✅ Easy to test each layer independently
- ✅ Changes don't ripple through system
- ✅ Clear where to add new features

### **5. Fail Gracefully**
**Problem:** External APIs can fail unpredictably.  
**Solution:** Comprehensive error handling with retries.

**Strategies:**
```python
# 1. Exponential backoff for transient errors
wait_time = 2 ** retry_count  # 1s, 2s, 4s, 8s...

# 2. Distinguish permanent vs transient errors
if status_code == 429:  # Rate limit - retry
    retry()
elif status_code == 401:  # Auth error - fail immediately
    raise AuthError()

# 3. Timeouts to prevent hanging
timeout = config.get("timeout", 30)

# 4. Fallback providers (future feature)
try:
    response = await openai_adapter.complete(request)
except ProviderError:
    response = await anthropic_adapter.complete(request)
```

---

## 🔧 Core Components

### **BaseAdapter (Abstract Interface)**

Every provider adapter implements this interface:

```python
class BaseAdapter(ABC):
    """
    Base class for all LLM provider adapters.
    
    Why abstract? Ensures all adapters have the same interface,
    making them interchangeable.
    """
    
    @abstractmethod
    async def complete(
        self, 
        request: CompletionRequest
    ) -> CompletionResponse:
        """
        Send a completion request.
        
        This is the main entry point. All adapters must implement this.
        
        Args:
            request: Standardized request format
            
        Returns:
            Standardized response format
        """
        pass
    
    @abstractmethod
    async def complete_stream(
        self, 
        request: CompletionRequest
    ) -> AsyncIterator[CompletionResponse]:
        """
        Stream a completion request.
        
        Returns chunks as they arrive from the provider.
        
        Yields:
            Response chunks
        """
        pass
```

**Key decisions:**
- **Async methods:** All I/O is async for performance
- **Standardized types:** `CompletionRequest` and `CompletionResponse` hide provider differences
- **Iterator pattern:** Streaming uses async generators for efficiency

### **Adapter Registry**

Manages adapter instances:

```python
# registry.py
ADAPTERS = {
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "vllm": VLLMAdapter,
}

def get_adapter(provider: str, config: dict) -> BaseAdapter:
    """
    Factory function to get adapter instances.
    
    Why a registry? Centralized provider management.
    Adding a new provider = adding one line.
    """
    if provider not in ADAPTERS:
        raise ValueError(f"Unknown provider: {provider}")
    
    adapter_class = ADAPTERS[provider]
    return adapter_class(config)
```

**Benefits:**
- Single source of truth for providers
- Easy to add new providers
- Configuration injection
- Type safety

### **Configuration System**

Hierarchical YAML configuration:

```python
# 1. Load base config
config = yaml.load("config/default.yaml")

# 2. Load environment-specific overrides
env = os.getenv("APP_ENV", "development")
overrides = yaml.load(f"config/{env}.yaml")

# 3. Merge configurations
config = merge(config, overrides)

# 4. Substitute environment variables
config = substitute_env_vars(config)
```

**Pattern:**
```yaml
# Use ${ENV_VAR} for environment variables
api_key: ${OPENAI_API_KEY}

# Default values
${OPENAI_API_KEY:default_value}
```

### **Request/Response Models**

Pydantic models for validation:

```python
class CompletionRequest(BaseModel):
    """
    Standardized request format.
    
    Why Pydantic? Automatic validation, type checking,
    and serialization.
    """
    model: str
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    stream: bool = False
    
    @validator("temperature")
    def validate_temperature(cls, v):
        if not 0 <= v <= 2:
            raise ValueError("Temperature must be 0-2")
        return v

class CompletionResponse(BaseModel):
    """Standardized response format."""
    content: str
    model: str
    usage: TokenUsage
    cost: float
    finish_reason: Optional[str] = None
```

**Benefits:**
- Automatic validation
- Type safety
- Clear contracts
- Documentation

---

## 🚦 Request Flow

### **1. Request Arrives**
```python
# API route receives request
@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    ...
```

### **2. Validation**
```python
# Pydantic validates request
# - Required fields present?
# - Correct types?
# - Valid values?
```

### **3. Provider Selection**
```python
# Determine which provider to use
provider = request.get("provider", config.defaults.provider)
```

### **4. Adapter Retrieval**
```python
# Get adapter from registry
adapter = get_adapter(provider, config.providers[provider])
```

### **5. Request Transformation**
```python
# Adapter transforms to provider-specific format
provider_request = adapter._transform_request(request)
```

### **6. API Call**
```python
# Make async HTTP request to provider
response = await client.post(provider_url, json=provider_request)
```

### **7. Response Transformation**
```python
# Transform provider response to standard format
standard_response = adapter._transform_response(response)
```

### **8. Return to Client**
```python
# Return standardized response
return standard_response
```

---

## 📊 Data Transformations

### **Request Transformation Example**

**Our standard format:**
```json
{
  "model": "gpt-4o",
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello"}
  ],
  "temperature": 0.7
}
```

**OpenAI format (same!):**
```json
{
  "model": "gpt-4o",
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello"}
  ],
  "temperature": 0.7
}
```

**Anthropic format (different!):**
```json
{
  "model": "claude-3-5-sonnet-20241022",
  "system": "You are helpful",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "temperature": 0.7
}
```

**Adapter handles this:**
```python
def _transform_request(self, request):
    # Extract system message separately for Anthropic
    system = next(
        (m["content"] for m in request["messages"] if m["role"] == "system"),
        None
    )
    
    # Filter out system messages
    messages = [
        m for m in request["messages"] 
        if m["role"] != "system"
    ]
    
    return {
        "model": request["model"],
        "system": system,
        "messages": messages,
        "temperature": request["temperature"]
    }
```

---

## 🔒 Error Handling Strategy

### **Error Categories**

| Error Type | Status Code | Action |
|------------|-------------|--------|
| **Validation** | 400 | Return immediately |
| **Authentication** | 401 | Return immediately |
| **Rate Limit** | 429 | Retry with backoff |
| **Server Error** | 5xx | Retry with backoff |
| **Timeout** | - | Retry |
| **Unknown** | - | Log and return 500 |

### **Retry Logic**

```python
async def _make_request(self, retry_count=0):
    """Make request with exponential backoff."""
    try:
        return await self._do_request()
        
    except httpx.HTTPStatusError as e:
        # Transient errors - retry
        if e.response.status_code in [429, 500, 502, 503, 504]:
            if retry_count < self.max_retries:
                wait_time = 2 ** retry_count  # 1s, 2s, 4s
                await asyncio.sleep(wait_time)
                return await self._make_request(retry_count + 1)
        
        # Permanent errors - fail
        raise
    
    except httpx.TimeoutException:
        # Timeout - retry
        if retry_count < self.max_retries:
            return await self._make_request(retry_count + 1)
        raise
```

### **Error Propagation**

```python
# Adapter Layer
try:
    response = await self._make_request()
except httpx.HTTPError as e:
    raise AdapterError(f"Request failed: {e}")

# Router Layer
try:
    response = await adapter.complete(request)
except AdapterError as e:
    logger.error(f"Adapter error: {e}")
    raise HTTPException(status_code=502, detail=str(e))

# API Layer
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )
```

---

## 🎨 Design Patterns Used

### **1. Factory Pattern** (Adapter Registry)
Creates adapter instances based on provider name.

### **2. Strategy Pattern** (Adapters)
Different strategies (adapters) for different providers, same interface.

### **3. Template Method** (BaseAdapter)
Base class defines algorithm structure, subclasses implement steps.

### **4. Singleton** (Config Loader)
Configuration loaded once and reused.

### **5. Iterator Pattern** (Streaming)
Async generators for streaming responses.

---

## 🚀 Performance Considerations

### **Async Concurrency**
```python
# Can handle multiple requests concurrently
# Each request doesn't block others
async def handle_requests(requests):
    tasks = [adapter.complete(req) for req in requests]
    return await asyncio.gather(*tasks)
```

### **Connection Pooling**
```python
# Reuse HTTP connections
async with httpx.AsyncClient() as client:
    # Connection pooling handled automatically
    response = await client.post(...)
```

### **Streaming for Large Responses**
```python
# Don't wait for full response
# Start processing immediately
async for chunk in adapter.complete_stream(request):
    process(chunk)  # Process as it arrives
```

---

## 🔮 Future Architecture

### **v2.0: Orchestration Layer**
```
React UI → Node.js Gateway → Python Service → Providers
```

### **v3.0: Advanced Features**
- **Multi-tenancy:** Separate namespaces per customer
- **RAG:** Vector store integration
- **Caching:** Redis for response caching
- **Event Bus:** Async communication between services
- **Queue System:** Job processing for long-running tasks

---

## 🤔 Design Decisions & Trade-offs

### **Why Python for Core?**
**Pros:**
- Strong AI/ML ecosystem
- Async support (asyncio)
- Type hints for safety
- Widely used in AI community

**Cons:**
- Slower than Go/Rust
- GIL limitations (mitigated by async I/O)

**Decision:** Python's ecosystem wins.

### **Why FastAPI?**
**Pros:**
- Native async support
- Automatic API documentation
- Type validation with Pydantic
- Modern and fast

**Cons:**
- Newer than Flask/Django
- Smaller community

**Decision:** Async support is critical for our use case.

### **Why YAML for Config?**
**Pros:**
- Human-readable
- Supports comments
- Good for hierarchical data

**Cons:**
- Not as powerful as Python
- Parsing can be slow (not an issue for us)

**Decision:** Readability and maintainability win.

### **Why Adapter Pattern?**
**Pros:**
- Easy to add providers
- Clean separation
- Testable

**Cons:**
- More code upfront
- Slight performance overhead

**Decision:** Flexibility and maintainability win.

---

## 📚 Further Reading

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Python Async Patterns](https://realpython.com/async-io-python/)
- [Design Patterns](https://refactoring.guru/design-patterns)
- [12-Factor App](https://12factor.net/)

---

## 🎯 Summary

FAIForge's architecture prioritizes:

1. **Flexibility** - Easy to extend
2. **Maintainability** - Clear structure
3. **Performance** - Async everything
4. **Reliability** - Comprehensive error handling
5. **Developer Experience** - Clean APIs, good docs

**The result:** A production-ready foundation for AI applications.
