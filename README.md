# 🔥 FAIForge

> Production-ready Python foundation for AI applications. Clean adapter pattern, full observability, Docker deployment.

**The boilerplate your AI product should have started with.**

Build AI features with any LLM provider. Fork it, add your models, ship in days not weeks.

---

## ✨ Why FAIForge?

Most teams building AI features face the same problems:
- 🔧 Reinventing infrastructure (routing, config, error handling)
- 💸 No cost visibility until the bill arrives
- 🐛 Debugging black-box API calls
- 🔌 Locked into one provider
- 🚀 Weeks of setup before shipping features

**FAIForge solves this.** It's a production-ready foundation with patterns you can trust.

---

## 🎯 Features

### **Core**
- ✅ **Multi-Provider Adapters** - OpenAI, Anthropic, local models (vLLM)
- ✅ **Async/Streaming** - Non-blocking I/O, real-time responses
- ✅ **Config-Driven** - YAML-based configuration with environment overrides
- ✅ **Error Handling** - Retry logic, exponential backoff, graceful failures
- ✅ **Cost Tracking** - Token usage and cost per request

### **Production-Ready** *(Coming in Next 2 Weeks)*
- 🚧 **Observability** - Structured logging, request tracing, health checks
- 🚧 **Docker Deployment** - One-command local setup, AWS/GCP guides
- 🚧 **Testing** - Unit and integration tests included

### **Extensible**
- 📦 **Adapter Pattern** - Add new providers in ~30 minutes
- 🔧 **Clear Architecture** - Easy to understand, easy to modify
- 📚 **Well-Documented** - Examples and guides included

---

## 🚀 Quick Start (5 Minutes)

### **Prerequisites**
- Python 3.9+
- OpenAI API key (or Anthropic, or run local models)

### **Installation**

```bash
# Clone the repo
git clone https://github.com/yourusername/faiforge.git
cd faiforge/backend

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your API keys
```

### **Run It**

```bash
# Start the service
python run.py

# In another terminal, test it
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

**That's it.** You're running a production-ready AI service.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│           Your Application                   │
└──────────────────┬──────────────────────────┘
                   │ HTTP/REST
┌──────────────────▼──────────────────────────┐
│         FastAPI Service (Port 8000)          │
│  ┌──────────────────────────────────────┐   │
│  │   Inference Router                   │   │
│  │   - Request validation               │   │
│  │   - Provider selection               │   │
│  │   - Error handling                   │   │
│  └──────────────┬───────────────────────┘   │
│                 │                             │
│  ┌──────────────▼───────────────────────┐   │
│  │      Adapter Registry                │   │
│  │  ┌────────┐ ┌────────┐ ┌─────────┐  │   │
│  │  │ OpenAI │ │Anthropic│ │  vLLM   │  │   │
│  │  │Adapter │ │ Adapter │ │ Adapter │  │   │
│  │  └────┬───┘ └────┬────┘ └────┬────┘  │   │
│  └───────┼──────────┼───────────┼───────┘   │
└──────────┼──────────┼───────────┼───────────┘
           │          │           │
      ┌────▼────┐ ┌───▼────┐ ┌───▼─────┐
      │ OpenAI  │ │Anthropic│ │  Local  │
      │   API   │ │   API   │ │  Model  │
      └─────────┘ └─────────┘ └─────────┘
```

### **Key Concepts**

**Adapter Pattern:**
Each provider (OpenAI, Anthropic, vLLM) implements the same `BaseAdapter` interface. Your application code doesn't need to know which provider it's using—just call `adapter.complete()`.

**Config-Driven:**
All settings (models, endpoints, defaults) are in YAML files. Change providers without touching code.

**Async-First:**
AI calls are I/O-bound. Async/await means you can handle multiple requests without blocking threads.

---

## 📁 Project Structure

```
faiforge/
├── backend/
│   ├── core/
│   │   ├── inference/
│   │   │   ├── adapters/
│   │   │   │   ├── base.py              # Base adapter interface
│   │   │   │   ├── openai_adapter.py    # OpenAI implementation
│   │   │   │   ├── anthropic_adapter.py # Anthropic implementation
│   │   │   │   └── vllm_adapter.py      # Local model implementation
│   │   │   ├── registry.py              # Adapter registration
│   │   │   └── router.py                # Request routing logic
│   │   ├── api/
│   │   │   └── routes.py                # FastAPI endpoints
│   │   └── config/
│   │       ├── loader.py                # YAML config loader
│   │       └── models.py                # Pydantic models
│   ├── config/
│   │   ├── default.yaml                 # Default configuration
│   │   └── production.yaml              # Production overrides
│   ├── requirements.txt
│   ├── run.py                           # Entry point
│   └── .env.example
├── examples/                            # Usage examples
│   ├── basic_usage.py
│   ├── streaming.py
│   └── custom_adapter/
├── docs/
│   ├── ADDING_ADAPTERS.md              # How to extend
│   └── ARCHITECTURE.md                  # Design decisions
├── tests/
│   ├── test_adapters.py
│   └── test_integration.py
└── README.md
```

---

## 🔌 Adding Your Own Model Provider

**FAIForge is designed to be extended.** Adding a new provider takes ~30 minutes.

See [docs/ADDING_ADAPTERS.md](docs/ADDING_ADAPTERS.md) for the complete guide.

**Quick overview:**

1. **Create adapter file** (`core/inference/adapters/your_provider.py`)
2. **Implement `BaseAdapter`** interface
3. **Register in `registry.py`**
4. **Add config in `config/models.yaml`**
5. **Test it**

You have 3 working examples to learn from:
- `openai_adapter.py` - Cloud API with retry logic
- `anthropic_adapter.py` - Different API structure
- `vllm_adapter.py` - Local model handling

---

## 📚 Examples

Check the `examples/` directory for working code:

### **Basic Usage**
```python
# examples/basic_usage.py
from core.inference.registry import get_adapter

# Get adapter for OpenAI
adapter = get_adapter("openai")

# Make a request
response = await adapter.complete({
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": False
})

print(response["content"])
```

### **Streaming Responses**
```python
# examples/streaming.py
async for chunk in adapter.complete_stream({
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": True
}):
    print(chunk["content"], end="", flush=True)
```

### **Compare Models**
```python
# examples/compare_models.py
# Run the same prompt across multiple providers
# and compare latency, cost, and quality
```

---

## ⚙️ Configuration

FAIForge uses YAML configuration with environment-specific overrides.

**`config/default.yaml`:**
```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}  # From environment variable
    base_url: "https://api.openai.com/v1"
    models:
      - name: "gpt-4o"
        context_window: 128000
        cost_per_1k_input: 0.005
        cost_per_1k_output: 0.015
      - name: "gpt-4o-mini"
        context_window: 128000
        cost_per_1k_input: 0.00015
        cost_per_1k_output: 0.0006

defaults:
  model: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 4096
  stream: true
```

**Environment-specific overrides:**
```yaml
# config/production.yaml
defaults:
  model: "gpt-4o"  # Use better model in production
  max_tokens: 2048  # More conservative limits
```

**Set environment:**
```bash
export APP_ENV=production  # Uses production.yaml overrides
python run.py
```

---

## 🧪 Testing

```bash
# Run unit tests
pytest tests/test_adapters.py

# Run integration tests
pytest tests/test_integration.py

# Run all tests
pytest
```

---

## 🚀 Deployment

### **Docker** *(Coming Soon)*
```bash
docker-compose up
```

### **Manual Deployment**
1. Set production environment variables
2. Use gunicorn or uvicorn for production ASGI server
3. Put behind nginx for SSL and rate limiting
4. Monitor with your observability stack

**Full deployment guides coming in v1.0 (2 weeks)**

---

## 🛣️ Roadmap

### **v1.0 - Python Foundation** *(Current - Launching Dec 2024)*
- [x] Multi-provider adapters (OpenAI, Anthropic, vLLM)
- [x] Config system (YAML with env overrides)
- [x] Async/streaming support
- [x] Error handling and retries
- [ ] Observability (structured logs, tracing)
- [ ] Docker deployment
- [ ] Testing suite
- [ ] Complete documentation

### **v2.0 - Full-Stack Platform** *(Jan 2025)*
- [ ] Node.js orchestration layer
- [ ] React dashboard UI
- [ ] Metrics visualization
- [ ] Multi-model playground
- [ ] Cost tracking dashboard

### **v3.0 - Advanced Features** *(Q1 2025)*
- [ ] RAG capabilities (document Q&A)
- [ ] Agent workflows (tool calling)
- [ ] Multi-modal support (vision, audio)
- [ ] Fine-tuning pipelines
- [ ] Multi-tenancy

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Report bugs** - Open an issue with details
2. **Request features** - Share your use case
3. **Add adapters** - Contribute new provider integrations
4. **Improve docs** - Fix typos, add examples
5. **Write tests** - Help us maintain quality

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 💡 Use Cases

**FAIForge is perfect for:**

✅ **Startups** building AI features from scratch  
✅ **Dev teams** tired of reinventing infrastructure  
✅ **Agencies** deploying custom AI solutions for clients  
✅ **Learners** wanting to understand production AI patterns  
✅ **Companies** needing provider flexibility  

**Not ideal for:**
❌ Simple projects (just use OpenAI SDK directly)  
❌ No-code solutions (this requires development)  

---

## 📝 License

MIT License - Use it however you want!

---

## 🙏 Acknowledgments

Built to scratch my own itch: "Why am I rebuilding the same AI infrastructure for every project?"

Inspired by the need for production-ready patterns that don't require a PhD to understand.

Special thanks to:
- The FastAPI team for an amazing framework
- OpenAI, Anthropic, and the vLLM team for building great APIs
- Every developer who's had to debug a black-box AI integration

---

## 📞 Support

- 📧 **Issues**: [GitHub Issues](https://github.com/yourusername/faiforge/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/faiforge/discussions)
- 🐦 **Twitter**: [@yourusername](https://twitter.com/yourusername)

---

**Built with ❤️ for the AI developer community**

⭐ **Star this repo** if you find it useful!  
🍴 **Fork it** and build something amazing!  
📢 **Share it** with your team!

---

> *"The best code is code you don't have to write." - FAIForge gives you the foundation so you can focus on your unique value.*
