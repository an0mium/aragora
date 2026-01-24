# Development Guide

This guide helps new contributors get set up and productive with Aragora development.

If you just want to run a debate (not contribute), start at [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

## Quick Start

```bash
# Clone the repository
git clone https://github.com/an0mium/aragora.git
cd aragora

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"

# Run tests to verify setup
pytest tests/ -v --timeout=30

# Start the development server
python -m aragora.server --api-port 8080 --ws-port 8765
```

## Prerequisites

- **Python 3.10+** (tested on 3.10, 3.11, 3.12)
- **Node.js 18+** (for frontend development)
- **Git** (obviously)

### Optional Dependencies

| Feature | Install Command | Notes |
|---------|-----------------|-------|
| PostgreSQL | `pip install psycopg2-binary` | Production database |
| Redis | `pip install redis` | Rate limiting, caching |
| Monitoring | `pip install -e ".[monitoring]"` | Prometheus, Sentry |
| Observability | `pip install -e ".[observability]"` | OpenTelemetry |
| Broadcast | `pip install -e ".[broadcast]"` | TTS, audio features |

## Project Structure

```
aragora/
├── agents/           # AI agent implementations (Claude, GPT, Gemini, etc.)
├── cli/              # Command-line interface
├── config/           # Configuration management
├── debate/           # Core debate orchestration
│   ├── orchestrator.py   # Main Arena class
│   ├── consensus.py      # Consensus detection
│   └── convergence.py    # Semantic similarity
├── memory/           # Learning and persistence
├── server/           # HTTP/WebSocket API
│   ├── handlers/     # Route handlers
│   └── unified_server.py  # Main server
├── storage/          # Database backends
└── verification/     # Formal proof generation

tests/                # Test suite (pytest)
scripts/              # Utility scripts
docs/                 # Documentation
```

## Environment Setup

### API Keys

Create a `.env` file in the project root:

```bash
# Required: At least one AI provider
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Optional: Additional providers
GEMINI_API_KEY=...
XAI_API_KEY=...
OPENROUTER_API_KEY=...  # Fallback for quota errors

# Optional: Server configuration
ARAGORA_API_TOKEN=your-secret-token
ARAGORA_ALLOWED_ORIGINS=http://localhost:3000
```

See `docs/ENVIRONMENT.md` for full reference.

Optional but recommended for local dev:
```bash
ARAGORA_DATA_DIR=.nomic
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_orchestrator.py -v

# Run tests matching pattern
pytest tests/ -k "consensus" -v

# Run with coverage
pytest tests/ --cov=aragora --cov-report=html

# Run only fast tests (skip slow integration tests)
pytest tests/ -v --timeout=10

# Run mutation testing
mutmut run --paths-to-mutate=aragora/debate/
```

### Test Organization

| Directory | Description |
|-----------|-------------|
| `tests/test_*.py` | Unit tests (fast, mocked dependencies) |
| `tests/integrations/` | Integration tests (may use real APIs) |
| `tests/e2e/` | End-to-end tests (full system) |

## Code Style

We use **Black** for formatting and **Ruff** for linting:

```bash
# Format code
black aragora/ tests/

# Lint code
ruff check aragora/ tests/

# Type checking
mypy aragora/

# Security scan
bandit -r aragora/
```

### Pre-commit Hooks (Recommended)

```bash
pip install pre-commit
pre-commit install

# Hooks run automatically on commit
# Manual run:
pre-commit run --all-files
```

## Making Changes

### Branch Naming

- `feature/short-description` - New features
- `fix/issue-number-description` - Bug fixes
- `refactor/area` - Code refactoring
- `docs/topic` - Documentation updates

### Commit Messages

Follow conventional commits:

```
type(scope): short description

Longer description if needed.

Closes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Pull Request Checklist

- [ ] Tests pass locally (`pytest tests/ -v`)
- [ ] Code is formatted (`black .`)
- [ ] Linting passes (`ruff check .`)
- [ ] New code has tests
- [ ] Documentation updated if needed
- [ ] No secrets committed (`.env` is gitignored)

## Common Development Tasks

### Adding a New Agent

1. Create agent class in `aragora/agents/`:
   ```python
   from aragora.agents.base import BaseAgent, AgentResponse

   class MyAgent(BaseAgent):
       AGENT_ID = "my-agent"

       async def respond(self, context: str, **kwargs) -> AgentResponse:
           # Implementation
           pass
   ```

2. Register in `aragora/agents/__init__.py`

3. Add tests in `tests/test_agents.py`

### Adding a New API Endpoint

1. Create handler in `aragora/server/handlers/`:
   ```python
   from .base import BaseHandler, json_response

   class MyHandler(BaseHandler):
       ROUTES = ["/api/my-endpoint"]

       def handle(self, path, query_params, handler):
           return json_response({"status": "ok"})
   ```

2. Register in `aragora/server/unified_server.py`

3. Add tests in `tests/test_handlers_*.py`

4. Document in `docs/API_REFERENCE.md`

### Adding a New Storage Backend

1. Extend `SQLiteStore` in `aragora/storage/`:
   ```python
   from aragora.storage.base_store import SQLiteStore

   class MyStore(SQLiteStore):
       SCHEMA_NAME = "my_store"
       SCHEMA_VERSION = 1
       INITIAL_SCHEMA = """
           CREATE TABLE IF NOT EXISTS my_table (...);
       """
   ```

2. Add tests in `tests/test_storage_*.py`

## Frontend Development

The frontend is a Next.js app in `aragora/live/`. See [docs/FRONTEND_DEVELOPMENT.md](docs/FRONTEND_DEVELOPMENT.md) for the full guide. The TypeScript SDK lives in `aragora-js/` and is not a UI.

```bash
cd aragora/live

# Install dependencies
npm install

# Development server
npm run dev

# Build for production
npm run build

# Run tests
npm test

# Lint
npm run lint
```

### Frontend Structure

```
aragora/live/
├── src/
│   ├── app/           # Next.js app router pages
│   ├── components/    # React components
│   ├── hooks/         # Custom React hooks
│   └── lib/           # Utilities
├── public/            # Static assets
└── package.json
```

## Debugging

### Server Debugging

```bash
# Enable debug logging
ARAGORA_LOG_LEVEL=DEBUG python -m aragora.server --api-port 8080 --ws-port 8765

# Profile a debate
python -c "
from aragora import Arena, Environment, DebateProtocol
import cProfile
cProfile.run('...')
"
```

### Common Issues

| Issue | Solution |
|-------|----------|
| Import errors | Run `pip install -e .` to reinstall |
| Database locked | Check for zombie processes: `lsof .nomic/aragora.db` |
| API rate limits | Add `OPENROUTER_API_KEY` for fallback |
| WebSocket disconnects | Check CORS settings in `.env` |

## Getting Help

- **Documentation**: `docs/` directory
- **Issues**: GitHub Issues
- **Architecture**: See `docs/ARCHITECTURE.md`
- **API Reference**: See `docs/API_REFERENCE.md`

## License

MIT License - see LICENSE file for details.
