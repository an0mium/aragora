# Contributing to Aragora

Guidelines for contributing to the Aragora multi-agent debate framework.

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- At least one API key: `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`

### Setup

```bash
# Clone the repository
git clone https://github.com/aragora/aragora.git
cd aragora

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### Verify Installation

```bash
# Run tests
pytest tests/ -v --timeout=60

# Run linter
python -m ruff check aragora/

# Run type checker
python -m mypy aragora/ --show-error-codes
```

## Code Standards

### Style

- **Formatter**: Black (line length 88)
- **Linter**: Ruff
- **Type Checker**: Mypy (strict mode for core modules)

```bash
# Format code
python -m black aragora/ tests/

# Check linting
python -m ruff check aragora/

# Fix auto-fixable issues
python -m ruff check aragora/ --fix

# Type check
python -m mypy aragora/ --show-error-codes
```

### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Files | snake_case | `user_store.py` |
| Classes | PascalCase | `DebateOrchestrator` |
| Functions | snake_case | `get_debate_result` |
| Constants | UPPER_SNAKE | `MAX_RETRY_COUNT` |
| Private | Leading underscore | `_internal_method` |

### Type Annotations

All new code must include type annotations:

```python
# Good
def process_debate(debate_id: str, options: dict[str, Any]) -> DebateResult:
    ...

# Bad - no type hints
def process_debate(debate_id, options):
    ...
```

Core modules under strict mypy (`pyproject.toml [tool.mypy]`):
- `aragora/debate/convergence.py`
- `aragora/debate/phases/*.py`
- `aragora/server/handlers/base.py`
- See `pyproject.toml` for full list

### Error Handling

Use project-specific exceptions from `aragora/exceptions.py`:

```python
from aragora.exceptions import DebateError, ValidationError

# Good - specific exception
if not debate_id:
    raise ValidationError("debate_id is required")

# Avoid - generic exception
if not debate_id:
    raise Exception("missing id")
```

### Logging

Use module-level loggers:

```python
import logging

logger = logging.getLogger(__name__)

def my_function():
    logger.info("Processing started")
    logger.debug("Debug details: %s", details)
    logger.warning("Potential issue detected")
    logger.error("Operation failed: %s", error)
```

## Testing

### Requirements

- All new features must have tests
- All bug fixes must have regression tests
- Target coverage: 70%+ for new code

### Running Tests

```bash
# All tests
pytest tests/ -v

# Fast tests only (skip slow/load)
scripts/test_tiers.sh fast

# CI-equivalent run
scripts/test_tiers.sh ci

# With coverage
pytest tests/ --cov=aragora --cov-report=html
```

### Writing Tests

```python
import pytest
from aragora.debate import Arena

class TestDebateOrchestration:
    """Test group for debate orchestration."""

    def test_arena_creates_valid_debate(self, mock_agents):
        """Arena should create debate with provided agents."""
        arena = Arena(agents=mock_agents)
        assert arena.agent_count == len(mock_agents)

    @pytest.mark.asyncio
    async def test_arena_runs_to_completion(self, mock_agents, mock_environment):
        """Arena should complete debate rounds."""
        arena = Arena(agents=mock_agents, environment=mock_environment)
        result = await arena.run()
        assert result.status == "completed"

    @pytest.mark.slow
    def test_arena_handles_large_agent_pool(self):
        """Tests with many agents (>5s runtime)."""
        ...
```

### Test Markers

| Marker | When to Use |
|--------|-------------|
| `@pytest.mark.slow` | Test takes >5 seconds |
| `@pytest.mark.load` | Stress/load testing |
| `@pytest.mark.integration` | Integration tests |
| `@pytest.mark.e2e` | End-to-end tests |
| `@pytest.mark.asyncio` | Async tests |

### Common Fixtures

From `tests/conftest.py`:

- `temp_db` - Temporary SQLite database
- `mock_storage` - Mock DebateStorage
- `mock_agents` - List of 3 mock agents
- `mock_api_keys` - Sets mock API keys
- `handler_context` - Complete handler context

## Pull Request Process

### Before Submitting

1. **Ensure tests pass**
   ```bash
   pytest tests/ -v --timeout=60
   ```

2. **Run linting**
   ```bash
   python -m ruff check aragora/
   python -m black --check aragora/
   ```

3. **Run type check**
   ```bash
   python -m mypy aragora/ --show-error-codes
   ```

4. **Security scan**
   ```bash
   python -m bandit -r aragora/ -ll
   ```

### Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation
- `test` - Tests
- `refactor` - Code refactoring
- `perf` - Performance improvement
- `chore` - Maintenance

**Examples:**
```
feat(debate): add consensus timeout configuration
fix(server): handle WebSocket disconnection gracefully
docs(api): update endpoint documentation
test(handlers): add rate limiting tests
refactor(memory): extract cache logic into separate module
```

### PR Description Template

```markdown
## Summary
Brief description of changes (1-3 sentences)

## Changes
- Change 1
- Change 2

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests (if applicable)
- [ ] Manual testing performed

## Checklist
- [ ] Tests pass locally
- [ ] Linting passes
- [ ] Type checking passes
- [ ] Documentation updated (if needed)
```

### Review Process

1. Create PR against `main` branch
2. Ensure CI checks pass
3. Request review from maintainer
4. Address feedback
5. Squash and merge after approval

## Architecture Guidelines

### Module Organization

```
aragora/
├── debate/           # Core debate logic
├── agents/           # Agent implementations
├── server/           # HTTP/WebSocket API
│   └── handlers/     # Request handlers
├── memory/           # Persistence layer
├── ranking/          # ELO system
└── exceptions.py     # All exceptions
```

### Adding New Handlers

1. Create handler in `aragora/server/handlers/`
2. Inherit from `BaseHandler`
3. Register in `handlers/__init__.py`
4. Add tests in `tests/test_handlers_*.py`

```python
from aragora.server.handlers.base import BaseHandler, json_response

class MyHandler(BaseHandler):
    routes = ["GET /api/my-endpoint"]

    @staticmethod
    def can_handle(path: str) -> bool:
        return path.startswith("/api/my-endpoint")

    def handle(self, path: str, query_params: dict, handler: Any):
        return json_response({"status": "ok"})
```

### Adding New Agents

1. Create agent in `aragora/agents/api_agents/`
2. Inherit from base agent class
3. Implement required methods
4. Register in agent factory
5. Add tests

See `docs/CUSTOM_AGENTS.md` for detailed guide.

## Protected Files

These files require explicit maintainer approval before modification:

- `CLAUDE.md` - AI assistant instructions
- `aragora/core.py` - Core dataclasses
- `aragora/__init__.py` - Package exports
- `scripts/nomic_loop.py` - Self-improvement loop

## Getting Help

- **Documentation**: `docs/` directory
- **Issues**: GitHub Issues
- **Testing Guide**: `docs/TESTING.md`
- **API Reference**: `docs/API_REFERENCE.md`

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
