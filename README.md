# 🤖 FAIForge

> A clean foundation for exploring LLM providers with a unified adapter pattern

![Status](https://img.shields.io/badge/status-production--ready-green)
![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Switch between OpenAI, Anthropic, and local models with one line of config.** FAIForge provides a simple adapter pattern for multi-provider LLM development with built-in observability and Docker deployment.

---

## 💡 Why FAIForge?

I built this while exploring different LLM providers and found myself repeatedly solving the same infrastructure problems:

- **Provider switching** - Rewriting code every time I wanted to test a different model
- **Cost tracking** - No visibility into per-request costs across providers
- **Observability** - Difficulty debugging AI interactions without proper logging
- **Deployment** - Setting up Docker, security headers, CORS each time

The adapter pattern solves this. Now I can compare GPT-4o vs Claude with just a config change, and all the observability/deployment infrastructure comes for free.

**Quick Links:** [Features](#-features) • [Quick Start](#-quick-start) • [Architecture](#️-architecture) • [Extending](#-extending-faiforge) • [API Docs](#-api-reference) • [Deployment](#-deployment)

---

## ✨ Features

### 🔌 Multi-Provider Architecture
- **OpenAI** (GPT-4o, GPT-4o-mini)
- **Anthropic** (Claude Opus 4, Claude Sonnet 4.5)
- **vLLM** (Local models - TinyLlama, any HuggingFace model)
- Unified adapter pattern - switch providers with one line

### 📊 Production Observability
- **Structured JSON logging** - Machine-parseable logs
- **Request correlation IDs** - Trace requests end-to-end
- **Automatic cost tracking** - Per-request pricing for all providers
- **Performance monitoring** - Latency, token counts, error rates
- **Health checks** - Built-in monitoring endpoints

### ⚙️ Configuration-Driven
- **YAML-based config** - No hardcoded values
- **Environment overrides** - Different configs for dev/staging/prod
- **Runtime config** - Override via environment variables
- **12-factor app** compliant

### 🐳 Docker Deployment
- **One-command setup** - `docker-compose up`
- **Multi-stage builds** - Optimized image sizes (~350MB total)
- **Health monitoring** - Auto-restart on failure
- **Production-ready** - Non-root user, security headers

### 🎨 Full-Stack Ready
- **FastAPI backend** - Modern, async Python
- **React + TypeScript frontend** - Beautiful chat UI
- **Nginx reverse proxy** - Production-grade serving
- **API documentation** - Auto-generated OpenAPI/Swagger

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API key
- Anthropic API key (optional)

### 1. Clone & Setup
```bash
git clone https://github.com/fiddyrod/faiforge.git
cd faiforge

# Add your API keys
cp backend/.env.example backend/.env
nano backend/.env  # Add your OPENAI_API_KEY and ANTHROPIC_API_KEY
```

### 2. Start Everything
```bash
docker-compose up -d
```

That's it! 🎉

- **Frontend:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

### 3. Test It
```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "model": "gpt-4o-mini"
  }'
```

---

## 🏗️ Architecture
```
┌─────────────┐
│   Browser   │
└──────┬──────┘
       │ HTTP
       ↓
┌─────────────────────────────────┐
│  Frontend (React + Nginx)       │
│  Port: 3000                     │
└──────┬──────────────────────────┘
       │ Proxy /v1/* → backend:8000
       ↓
┌─────────────────────────────────┐
│  Backend (FastAPI)              │
│  Port: 8000                     │
│  ┌─────────────────────────┐   │
│  │  Request Middleware     │   │
│  │  - Logging              │   │
│  │  - Correlation IDs      │   │
│  │  - Error handling       │   │
│  └───────────┬─────────────┘   │
│              ↓                  │
│  ┌─────────────────────────┐   │
│  │  Model Registry         │   │
│  │  - Load configs         │   │
│  │  - Initialize adapters  │   │
│  └───────────┬─────────────┘   │
│              ↓                  │
│  ┌─────────────────────────┐   │
│  │  LLM Adapters           │   │
│  │  ┌──────────┐           │   │
│  │  │ OpenAI   │───────────┼───┼─→ api.openai.com
│  │  │ Anthropic│───────────┼───┼─→ api.anthropic.com
│  │  │ vLLM     │ (local)   │   │
│  │  └──────────┘           │   │
│  └─────────────────────────┘   │
└─────────────────────────────────┘

Observability:
- JSON logs → stdout → Docker logs
- Request traces → Correlation IDs
- Metrics → Cost, latency, tokens
```

### Key Components

**Frontend (`/frontend`)**
- React 18 + TypeScript
- Tailwind CSS styling
- Vite build system
- Nginx production serving

**Backend (`/backend`)**
- FastAPI async framework
- Pydantic validation
- Multi-provider adapters
- Structured logging

**Configuration (`/backend/core/config`)**
- YAML-based config files
- Environment-specific overrides
- Runtime env var support

**Adapters (`/backend/core/inference/adapters`)**
- Unified interface
- Provider-specific implementations
- Automatic cost calculation
- Error handling

---

## ⚙️ Configuration

### Application Config (`backend/core/config/app.yaml`)
```yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 1

cors:
  enabled: true
  origins:
    - "http://localhost:3000"
  allow_methods: ["GET", "POST", "OPTIONS"]

defaults:
  model: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 500

observability:
  log_level: "INFO"
  log_format: "json"
```

### Model Config (`backend/core/config/models.yaml`)
```yaml
models:
  gpt-4o-mini:
    adapter: openai
    model: gpt-4o-mini
  
  claude-sonnet:
    adapter: anthropic
    model: claude-sonnet-4-5-20250929
  
  tiny-llama:
    adapter: vllm
    model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
    gpu_memory_utilization: 0.5
```

### Environment Variables
```bash
# Required
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Optional
ENV=production              # development | production
LOAD_VLLM=false            # Enable local models
FAIFORGE_API_PORT=9000     # Override port
```

---

## 📡 API Reference

### Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": 4
}
```

### List Models
```bash
GET /v1/models
```

**Response:**
```json
{
  "models": ["gpt-4o-mini", "gpt-4o", "claude-sonnet", "claude-opus"]
}
```

### Chat Completion
```bash
POST /v1/chat/completions
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "model": "gpt-4o-mini",
  "temperature": 0.7,
  "max_tokens": 500
}
```

**Response:**
```json
{
  "content": "Hello! How can I help you?",
  "model": "gpt-4o-mini",
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 9,
    "total_tokens": 17
  },
  "cost_usd": 0.000005,
  "latency_ms": 234.5
}
```

**Full API docs:** http://localhost:8000/docs

---

## 🚀 Deployment

### Docker Compose (Recommended)
```bash
# Production deployment
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Cloud Platforms

**AWS ECS / Fargate**
- Use `docker-compose.yml` as base
- Configure task definitions with environment variables
- Set up Application Load Balancer
- Use AWS Secrets Manager for API keys

**Google Cloud Run**
- Deploy backend and frontend as separate services
- Configure service URLs in environment
- Use Secret Manager for API keys
- Set up Cloud Load Balancing

**Railway / Render / Fly.io**
- Connect GitHub repository
- Automatic HTTPS
- Built-in monitoring
- One-click deployment

---

## 🔌 Extending FAIForge - The Adapter Pattern

### Why Adapters Matter

The **adapter pattern** is the core of FAIForge. It's what makes provider switching painless and keeps your code clean as you add more models.

Each LLM provider has a different API format. Without adapters, you'd have provider-specific logic scattered everywhere. With adapters, you write it once and all providers work the same way.

### Adding New Providers

**Time to add:** ~30 minutes

**What you'll do:**
1. Create adapter class inheriting from `BaseAdapter`
2. Implement `complete()` and `complete_stream()` methods
3. Transform requests/responses to match the provider's API
4. Register in `registry.py` and configure in `models.yaml`

**Currently supported:**
- OpenAI (GPT-4o, GPT-4o-mini)
- Anthropic (Claude Opus 4, Sonnet 4.5)
- vLLM (local models - TinyLlama, any HuggingFace model)

**Easy to add:**
- Cohere (Command R, Command R+)
- Google Gemini (Pro, Ultra)
- Mistral AI (Mistral Large, Mixtral)
- Any OpenAI-compatible API

**Complete tutorial:** See [docs/ADDING_ADAPTERS.md](docs/ADDING_ADAPTERS.md) for step-by-step guide with working Cohere example.

### What Adapters Give You

- **Isolation** - Provider changes don't affect other code
- **Consistency** - All providers return the same response format
- **Testability** - Easy to mock and test each provider
- **Observability** - Unified logging and cost tracking across all providers

Once you understand this pattern, adding providers becomes routine.

---

## 💻 Development

### Local Development (Without Docker)

**Backend:**
```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run server
python run.py
```

**Frontend:**
```bash
cd frontend

# Install dependencies
npm install

# Run dev server
npm run dev
```

### Project Structure
```
faiforge/
├── backend/
│   ├── core/
│   │   ├── api/              # FastAPI routes & server
│   │   │   └── server.py
│   │   ├── inference/        # LLM adapters
│   │   │   ├── adapters/
│   │   │   │   ├── base.py
│   │   │   │   ├── openai_adapter.py
│   │   │   │   ├── anthropic_adapter.py
│   │   │   │   └── vllm_adapter.py
│   │   │   └── registry.py
│   │   ├── config/           # Configuration management
│   │   │   ├── __init__.py
│   │   │   ├── app.yaml
│   │   │   ├── models.yaml
│   │   │   └── environments/
│   │   │       ├── development.yaml
│   │   │       └── production.yaml
│   │   └── observability/    # Logging & monitoring
│   │       ├── __init__.py
│   │       └── middleware.py
│   ├── main.py
│   ├── run.py
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── public/
│   ├── package.json
│   ├── nginx.conf
│   └── Dockerfile
├── docker-compose.yml
├── README.md
└── LICENSE
```

---

## 🎯 Use Cases

### 1. AI Product Development
Build your AI application on top of FAIForge instead of starting from scratch. Focus on your product logic while leveraging production-ready infrastructure.

### 2. Cost Optimization
- **Development:** Test with TinyLlama (local, free)
- **Staging:** Use GPT-4o-mini (cheap, fast)
- **Production:** Deploy with Claude Opus (quality)

### 3. Multi-Model Comparison
Run the same prompt across different models and compare:
- Response quality
- Speed/latency
- Cost per request
- Token usage

### 4. Learning & Experimentation
Study production-ready patterns:
- Adapter pattern implementation
- Observability setup
- Docker multi-stage builds
- Configuration management

---

## 🔧 Troubleshooting

### "CORS error in browser"
Make sure Docker containers are running:
```bash
docker-compose ps
```
Both services should show "Up" status.

### "ModuleNotFoundError: vllm"
vLLM is not included in base Docker image (requires GPU). Set `LOAD_VLLM=false` in your `.env` file.

### "OPENAI_API_KEY not found"
- Check `.env` file exists in `backend/` directory
- Ensure no quotes around values: `OPENAI_API_KEY=sk-...` not `"sk-..."`
- Restart containers after changing `.env`

### "Container keeps restarting"
Check logs for errors:
```bash
docker-compose logs backend
docker-compose logs frontend
```

### "Empty response from backend"
Backend might be crashing. Check:
```bash
docker-compose logs backend | tail -50
```

---

## ❓ FAQ

**Q: Can I use this in production?**  
A: Yes! It's designed with production patterns (observability, error handling, Docker), but always test thoroughly with your specific use case first.

**Q: Do I need a GPU?**  
A: Only if you want to run local models via vLLM. Cloud providers (OpenAI, Anthropic) work without GPU.

**Q: How much does it cost to run?**  
A: Docker hosting is cheap (~$5-20/month). LLM costs depend on usage - OpenAI/Anthropic charge per token. Monitor in their dashboards.

**Q: Can I add more LLM providers?**
A: Yes! The adapter pattern makes this straightforward. Adding a new provider (Cohere, Gemini, Mistral, etc.) takes ~30 minutes. See [docs/ADDING_ADAPTERS.md](docs/ADDING_ADAPTERS.md) for a complete tutorial with working examples.

**Q: Is this better than LangChain?**
A: Different goals. LangChain excels at complex chains and agents. FAIForge focuses on production infrastructure, observability, and multi-provider management. They can complement each other - use FAIForge as your API layer with LangChain for orchestration if needed.

**Q: How do I update to new model versions?**  
A: Update `backend/core/config/models.yaml` with new model IDs. No code changes needed.

---

## 🛣️ Roadmap

The following features are planned for future releases:

### Planned Features

**Authentication & Persistence:**
- [ ] Conversation persistence (SQLite/PostgreSQL)
- [ ] User authentication & sessions
- [ ] Rate limiting implementation
- [ ] Caching layer (Redis)

**Advanced AI Capabilities:**
- [ ] RAG module (document Q&A)
- [ ] Vector database integration (Pinecone/Weaviate)
- [ ] Streaming responses
- [ ] Conversation search

**Enterprise Features:**
- [ ] Agent framework
- [ ] Tool calling & function execution
- [ ] Multi-agent orchestration
- [ ] Multi-modal support (vision, audio)

**Developer Experience:**
- [ ] Admin dashboard
- [ ] Model evaluation suite
- [ ] Fine-tuning pipeline
- [ ] Advanced UI components

---

## 📝 License & Usage

MIT License - see [LICENSE](LICENSE) for details.

This project is open-source and available for use in personal or commercial projects. While the code is available for reference and learning, this is currently a personal project not actively seeking external contributions.

Feel free to fork and adapt for your own needs!

---

## 🙏 Acknowledgments

Built with these amazing tools:

- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern Python web framework
- **[React](https://react.dev/)** - UI library
- **[vLLM](https://github.com/vllm-project/vllm)** - High-performance local model serving
- **[Docker](https://www.docker.com/)** - Containerization platform
- **[Tailwind CSS](https://tailwindcss.com/)** - Utility-first CSS
- **[Pydantic](https://docs.pydantic.dev/)** - Data validation
- **[Nginx](https://nginx.org/)** - Web server

Special thanks to the open-source community for these incredible tools!

---

## 🎓 Design Principles

This project prioritizes:

**Production-Ready Over Feature-Rich**
- Security, observability, and error handling come first
- Clean architecture patterns that scale
- Comprehensive input validation and timeout handling

**Simplicity Over Complexity**
- No unnecessary abstractions
- Clear separation of concerns
- Easy to understand and modify

**Developer Experience**
- One-command deployment
- Clear documentation
- Extensible adapter pattern for adding providers

Simple, focused, and practical.

---

## 👨‍💻 About

I built FAIForge as a personal project for exploring different LLM providers and testing prompt strategies across models. The adapter pattern made it easy to swap providers, and the observability features helped me understand costs and performance.

**Use it as a foundation**: If you're building AI applications and want to avoid reinventing provider management, cost tracking, and deployment infrastructure, feel free to fork this and build on top of it.

It's saved me considerable time - hopefully it helps you too!

---

## ⚠️ Important Notes

**API Keys:** Never commit your `.env` file. API keys should be environment variables only. The `.env` file is in `.gitignore` for safety.

**Costs:** OpenAI and Anthropic charge per token. Always monitor usage in their dashboards. Set up billing alerts!

**Local Models:** vLLM requires NVIDIA GPU with CUDA support. CPU inference is extremely slow and not recommended.

**Security:** Input validation, timeouts, and security headers are included. For public deployments, add authentication and rate limiting (configurations provided). Review [SECURITY_CHECKLIST.md](SECURITY_CHECKLIST.md) before production deployment.

---

## 📧 Contact & Support

- **Issues:** Open an issue in your repository's Issues tab
- **Discussions:** Use GitHub Discussions for questions and ideas

Found a bug? Have a feature request? Open an issue!

Want to chat about AI development? Start a discussion!

---

**⭐ Star this repo if you find it useful!**

---

*FAIForge - The AI boilerplate your product should have started with* 🚀
