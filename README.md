# 🤖 FAIForge

> Production-ready AI boilerplate with multi-provider support and built-in observability

![Status](https://img.shields.io/badge/status-production--ready-green)
![Version](https://img.shields.io/badge/version-1.0.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**Stop reinventing the wheel.** FAIForge is a production-ready foundation for building AI applications with multiple LLM providers, complete observability, and Docker deployment.

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
git clone https://github.com/yourusername/faiforge.git
cd faiforge

# Add your API keys
cp backend/.env.example backend/.env
nano backend/.env  # Add your keys
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
A: Absolutely! Follow the adapter pattern in `core/inference/adapters/`. See existing adapters as examples.

**Q: Is this better than LangChain?**  
A: Different purpose. LangChain is for complex chains and agents. FAIForge is a clean, production-ready foundation to build on.

**Q: How do I update to new model versions?**  
A: Update `backend/core/config/models.yaml` with new model IDs. No code changes needed.

---

## 🛣️ Roadmap

### v1.1 (Next 2-4 Weeks)
- [ ] Conversation persistence (SQLite/PostgreSQL)
- [ ] User authentication & sessions
- [ ] Rate limiting
- [ ] Caching layer (Redis)

### v1.2 (1-2 Months)
- [ ] RAG module (document Q&A)
- [ ] Vector database integration (Pinecone/Weaviate)
- [ ] Streaming responses
- [ ] Conversation search

### v1.3 (2-3 Months)
- [ ] Agent framework
- [ ] Tool calling & function execution
- [ ] Multi-agent orchestration
- [ ] Advanced UI components

### v2.0 (Future)
- [ ] Multi-modal support (vision, audio)
- [ ] Fine-tuning pipeline
- [ ] Model evaluation suite
- [ ] Admin dashboard

---

## 🤝 Contributing

Contributions welcome! This project is both a learning resource and production foundation.

**How to contribute:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

**Areas needing help:**
- Additional LLM providers (Gemini, Cohere, Mistral, Llama)
- UI/UX improvements
- Documentation enhancements
- Test coverage
- Deployment guides for specific platforms
- Performance optimizations

**Code Style:**
- Backend: Follow PEP 8, use type hints
- Frontend: TypeScript strict mode, ESLint
- Commits: Conventional Commits format

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

You're free to use this for personal or commercial projects. Attribution appreciated but not required!

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

## 👨‍💻 About

Built by a developer learning modern AI development. This project represents ~60 hours of focused weekend work, implementing production patterns learned from experience and research.

Started as a learning exercise to transition from PHP to full-stack Python/React development. Along the way, I discovered best practices for observability, configuration management, and Docker deployment.

**If you're building AI products or learning AI development, I hope this saves you time!**

---

## ⚠️ Important Notes

**API Keys:** Never commit your `.env` file. API keys should be environment variables only. The `.env` file is in `.gitignore` for safety.

**Costs:** OpenAI and Anthropic charge per token. Always monitor usage in their dashboards. Set up billing alerts!

**Local Models:** vLLM requires NVIDIA GPU with CUDA support. CPU inference is extremely slow and not recommended.

**Security:** This is designed for development/internal use. For public deployments, add authentication, rate limiting, and input validation.

---

## 📧 Contact & Support

- **Issues:** [GitHub Issues](https://github.com/yourusername/faiforge/issues)
- **Discussions:** [GitHub Discussions](https://github.com/yourusername/faiforge/discussions)
- **Twitter:** [@yourhandle](https://twitter.com/yourhandle) (optional)
- **Email:** your.email@example.com (optional)

Found a bug? Have a feature request? Open an issue!

Want to chat about AI development? Start a discussion!

---

**⭐ Star this repo if you find it useful!**

---

*FAIForge - The AI boilerplate your product should have started with* 🚀
