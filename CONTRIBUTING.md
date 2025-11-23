# 🤝 Contributing to FAIForge

**Note:** This project is currently a personal/professional project and not actively seeking external contributions. This document is maintained as a reference for code standards and development guidelines.

If you're interested in using FAIForge, feel free to fork the repository and adapt it for your own needs under the MIT License.

---

## 📖 Purpose of This Document

This guide documents the development standards, patterns, and best practices used in FAIForge. It serves as:
- Reference documentation for the codebase architecture
- Guide for maintaining consistency when forking
- Educational resource for understanding the project's design decisions

---

## 🚀 Development Setup

If you're forking this project, here's how to set up your development environment:

### **1. Clone Your Fork**
```bash
git clone https://github.com/YOUR_USERNAME/faiforge.git
cd faiforge
```

### **2. Set Up Development Environment**
```bash
# Backend
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Dev dependencies

# Configure
cp .env.example .env
# Add your API keys
```

### **3. Run Tests**
```bash
pytest tests/
pytest -v                # Verbose output
pytest --cov            # With coverage
```

---

## 📋 Development Guidelines

### **Code Style**

**Python:**
```python
# Use type hints
def complete(self, request: CompletionRequest) -> CompletionResponse:
    pass

# Document functions
"""
Brief description.

Args:
    param: Description

Returns:
    Description
"""

# Use descriptive variable names
user_message = request["messages"][-1]  # ✅ Good
msg = req["messages"][-1]               # ❌ Bad

# Follow PEP 8
# Use black for formatting: black backend/
```

**Async Best Practices:**
```python
# Always await async functions
response = await adapter.complete(request)  # ✅
response = adapter.complete(request)        # ❌

# Use async with for context managers
async with httpx.AsyncClient() as client:   # ✅
    ...

# Don't mix sync and async
def sync_function():
    await async_call()  # ❌ Won't work
```

### **Testing**

Every new feature should have tests:

```python
# tests/test_your_feature.py
import pytest

@pytest.mark.asyncio
async def test_your_feature():
    """Test description."""
    # Arrange
    adapter = YourAdapter(config)
    request = {...}
    
    # Act
    response = await adapter.complete(request)
    
    # Assert
    assert response["content"]
    assert response["usage"]["total_tokens"] > 0
```

**Run tests:**
```bash
pytest tests/
pytest tests/test_specific_file.py  # Single file
pytest -v                            # Verbose
pytest --cov                         # With coverage
```

### **Documentation**

- **Code comments:** Explain WHY, not WHAT
  ```python
  # ✅ Good
  # Use exponential backoff to avoid overwhelming the API during outages
  wait_time = 2 ** retry_count
  
  # ❌ Bad
  # Calculate wait time
  wait_time = 2 ** retry_count
  ```

- **Docstrings:** Required for public functions/classes
- **README updates:** If you change functionality
- **Examples:** Add example if introducing new feature

---

## 🔌 Adding a New Adapter

The most common contribution! See [ADDING_ADAPTERS.md](ADDING_ADAPTERS.md) for detailed guide.

**Checklist:**
- [ ] Create `adapters/your_provider.py`
- [ ] Inherit from `BaseAdapter`
- [ ] Implement `complete()` and `complete_stream()`
- [ ] Add to `registry.py`
- [ ] Add config to `config/models.yaml`
- [ ] Write tests in `tests/test_your_provider.py`
- [ ] Add example in `examples/`
- [ ] Update main README

**Example PR title:**
```
feat: add Cohere adapter with streaming support
```

---


## 🏗️ Architecture Principles

The codebase follows these principles:

### **1. Adapter Pattern First**
New providers should use the adapter pattern. Don't modify the core router.

### **2. Config Over Code**
Prefer YAML configuration over hardcoded values.

```python
# ✅ Good
timeout = config.get("timeout", 30)

# ❌ Bad
timeout = 30
```

### **3. Fail Gracefully**
Handle errors, don't let them crash the system.

```python
# ✅ Good
try:
    response = await client.post(...)
except httpx.HTTPError as e:
    logger.error(f"Request failed: {e}")
    raise AdapterError("Failed to complete request")

# ❌ Bad
response = await client.post(...)  # Unhandled error
```

### **4. Async Everywhere**
Use async/await for I/O operations.

### **5. Type Hints**
Use type hints for better code clarity.

### **6. Testing**
All new code should have tests.

---

## 📝 Commit Message Guidelines

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```bash
feat: add new feature
fix: fix bug
docs: update documentation
test: add tests
refactor: refactor code
style: formatting changes
chore: maintenance tasks
```

**Examples:**
```bash
feat: add Cohere adapter with streaming support
fix: handle timeout errors in OpenAI adapter
docs: improve ADDING_ADAPTERS.md with more examples
test: add integration tests for Anthropic adapter
refactor: simplify request transformation logic
```

**Bad examples:**
```bash
Update stuff            # ❌ Too vague
Fixed it                # ❌ What did you fix?
Added code              # ❌ What code?
```

---

---

## ✅ Code Quality Standards

When working with the codebase, maintain these standards:

- ✅ **Functionality** - Code works as intended
- ✅ **Tests** - All new code has tests
- ✅ **Documentation** - Public APIs are documented
- ✅ **Patterns** - Follows established architecture
- ✅ **Clarity** - Code is readable and maintainable

---

## 📜 License

Any modifications you make to your fork will be under the MIT License.

---

## ❓ Questions About the Code?

This project is well-documented:

- 📖 **[README.md](README.md)** - Main project documentation
- 🏗️ **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design
- 🔌 **[ADDING_ADAPTERS.md](docs/ADDING_ADAPTERS.md)** - Adapter tutorial
- 🔒 **[SECURITY_CHECKLIST.md](SECURITY_CHECKLIST.md)** - Production deployment

---

**FAIForge - A clean foundation for multi-provider LLM development** 🚀
