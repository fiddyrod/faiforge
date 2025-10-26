# 🔥 FAIForge

> Production-ready boilerplate for building GenAI applications

Build full-stack AI applications with any LLM provider. One interface, infinite possibilities.

![FAIForge Demo](https://github.com/user-attachments/assets/4b0f8982-b96a-4dad-8928-ecf2d444dd50)


## ✨ Features

- 🔌 **Multi-provider support** - OpenAI, Anthropic (coming), vLLM (coming)
- 🎨 **Beautiful UI** - React + TypeScript + Tailwind CSS
- 📊 **Built-in metrics** - Real-time cost tracking & latency monitoring
- 🔧 **Extensible** - Add new providers in minutes via adapter pattern
- 🚀 **Production-ready** - Error handling, CORS, async/await

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- OpenAI API key

### Backend Setup
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Add your OPENAI_API_KEY to .env
python run.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Visit http://localhost:5173 🎉

## 🏗️ Architecture
```
React UI → FastAPI → Adapter Layer → LLM Providers
                                   ├─ OpenAI ✅
                                   ├─ Anthropic (Week 2)
                                   └─ vLLM (Week 2)
```

## 📁 Project Structure
```
faiforge/
├── backend/
│   ├── core/
│   │   ├── inference/      # LLM adapters
│   │   ├── api/            # FastAPI server
│   │   └── config/         # Configuration
│   └── main.py
└── frontend/
    └── src/
        └── App.tsx         # Chat UI
```

## 🛣️ Roadmap

### ✅ Week 1 (Complete)
- [x] OpenAI integration
- [x] React chat UI with cost tracking
- [x] Model switching
- [x] Error handling

### 🚧 Week 2 (Next)
- [ ] Anthropic Claude adapter
- [ ] vLLM for local models
- [ ] YAML configuration system
- [ ] Structured logging

### 🔮 Future
- [ ] RAG module (document Q&A)
- [ ] Agent framework (tool calling)
- [ ] Multi-modal support (vision, audio)
- [ ] Fine-tuning pipelines
- [ ] Docker deployment

## 💡 Why FAIForge?

Building GenAI applications shouldn't require reinventing the wheel. FAIForge provides:

- **Adapter Pattern** - Switch providers without changing your code
- **Cost Transparency** - See exactly what each request costs
- **Type Safety** - TypeScript + Python type hints
- **Developer Experience** - Auto-reload, clear errors, great docs

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

MIT

## 🙏 Acknowledgments

Built during a weekend sprint to learn production AI patterns. Inspired by the need for better GenAI boilerplates.

---

**Built with ❤️ for the AI developer community**

⭐ Star this repo if you find it useful!