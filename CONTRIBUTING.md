# 🤝 Contributing to FAIForge

First off, thank you for considering contributing to FAIForge! It's people like you that make FAIForge a great tool for the AI developer community.

---

## 🎯 Ways to Contribute

There are many ways to contribute:

- 🐛 **Report bugs** - Found something broken? Let us know!
- 💡 **Suggest features** - Have an idea? Share it!
- 🔌 **Add adapters** - Support for new LLM providers
- 📝 **Improve docs** - Fix typos, add examples, clarify explanations
- 🧪 **Write tests** - Help us maintain quality
- 🎨 **UI/UX improvements** - Make FAIForge more beautiful
- 💬 **Answer questions** - Help others in discussions

---

## 🚀 Quick Start for Contributors

### **1. Fork and Clone**
```bash
# Fork the repo on GitHub, then:
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

### **3. Create a Branch**
```bash
git checkout -b feature/your-feature-name
# Or: fix/bug-description
# Or: docs/what-youre-documenting
```

### **4. Make Your Changes**
- Write code
- Add tests
- Update docs
- Test locally

### **5. Commit and Push**
```bash
git add .
git commit -m "feat: add support for X"
git push origin feature/your-feature-name
```

### **6. Open a Pull Request**
- Go to GitHub and open a PR
- Fill out the PR template
- Wait for review

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

## 🐛 Reporting Bugs

**Before reporting:**
1. Check existing issues
2. Try with latest version
3. Test with minimal example

**What to include:**
```markdown
## Bug Description
Clear description of what's wrong

## To Reproduce
Steps to reproduce:
1. Configure X
2. Run Y
3. See error

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## Environment
- FAIForge version: v1.0.0
- Python version: 3.11
- OS: Ubuntu 22.04
- Provider: OpenAI

## Error Logs
```python
# Paste full error traceback
```

## Additional Context
Any other relevant information
```

---

## 💡 Suggesting Features

We love new ideas! Before suggesting:

1. **Check existing issues** - Someone might have suggested it
2. **Consider scope** - Does it fit FAIForge's purpose?
3. **Think about implementation** - How would it work?

**Use this template:**
```markdown
## Feature Description
What feature do you want?

## Use Case
Who would use this and why?

## Proposed Solution
How could this be implemented?

## Alternatives Considered
What other approaches did you think about?

## Additional Context
Screenshots, examples, etc.
```

---

## 🏗️ Architecture Principles

When contributing, keep these principles in mind:

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

## 🔍 Code Review Process

1. **Automated checks** run on your PR (tests, linting)
2. **Maintainer review** - We'll review your code
3. **Feedback** - We might request changes
4. **Approval** - Once approved, we'll merge!

**What we look for:**
- ✅ Does it work?
- ✅ Are there tests?
- ✅ Is it documented?
- ✅ Does it follow our patterns?
- ✅ Is the code clear?

**Response time:**
- We aim to respond within 48 hours
- Complex PRs might take longer

---

## 🎯 Good First Issues

New to the project? Look for issues labeled:
- `good first issue` - Easy to tackle
- `help wanted` - We need help on these
- `documentation` - Improve docs

---

## 💬 Communication

### **GitHub Issues**
For bugs, features, and technical questions.

### **GitHub Discussions**
For:
- General questions
- Ideas and brainstorming
- Show and tell
- Community chat

### **Pull Requests**
For:
- Code contributions
- Documentation updates

---

## 🚫 What NOT to Contribute

To keep FAIForge focused, please avoid:

- ❌ **Unrelated features** - Stay within scope
- ❌ **Dependencies without reason** - Keep it lightweight
- ❌ **Breaking changes** - Without discussion first
- ❌ **Duplicates** - Check existing PRs/issues
- ❌ **Code without tests** - Tests are required

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## 🙏 Recognition

Contributors are recognized in:
- GitHub contributors page
- Release notes
- Special thanks in README

---

## ❓ Questions?

Not sure about something? Ask!

- 💬 **Discussions:** For general questions
- 📧 **Issues:** For specific problems
- 📖 **Docs:** Check existing documentation

---

**Thank you for contributing to FAIForge!** 🚀

Every contribution, no matter how small, makes FAIForge better for everyone. We appreciate your time and effort!
