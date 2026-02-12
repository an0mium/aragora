# Local Development Setup

Get Aragora running locally for development, testing, and debugging.

## Prerequisites

- **Python 3.10+** (3.11 recommended; 3.10-3.13 supported)
- **Git**
- **pip** (bundled with Python)
- At least one LLM API key (see [Environment Variables](#environment-variables))

## Clone and Install

```bash
git clone https://github.com/an0mium/aragora.git
cd aragora

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

For full test coverage (includes optional extras like Redis, OpenAI, Z3, etc.):

```bash
pip install -e ".[test]"
```

To install everything (all optional extras):

```bash
pip install -e ".[all]"
```

## Environment Variables

Create a `.env` file in the project root (never commit this file):

```bash
# Required -- at least one LLM provider key
ANTHROPIC_API_KEY=sk-ant-...        # Claude
OPENAI_API_KEY=sk-...               # GPT / Whisper

# Recommended -- automatic fallback on 429 quota errors
OPENROUTER_API_KEY=sk-or-...

# Optional providers
MISTRAL_API_KEY=...
GEMINI_API_KEY=...
XAI_API_KEY=...
```

No database setup is needed for local dev. Aragora defaults to SQLite when no
Postgres/Supabase credentials are configured.

## Running the Server

### Quick start (offline mode)

Offline mode uses SQLite and in-memory stores with demo data, requiring no
external services or API keys:

```bash
python -m aragora.server --offline
```

This sets `ARAGORA_OFFLINE=true`, `ARAGORA_DEMO_MODE=true`, and
`ARAGORA_DB_BACKEND=sqlite` automatically.

### Standard local server

```bash
# HTTP API on :8080, WebSocket on :8765
python -m aragora.server --port 8765 --http-port 8080

# Bind to all interfaces (for testing from other devices)
python -m aragora.server --host 0.0.0.0

# Multi-worker production mode
python -m aragora.server --workers 4 --host 0.0.0.0
```

The HTTP API is available at `http://localhost:8080/api/` and WebSocket
streaming at `ws://localhost:8765/ws`.

## Running Tests

The test suite uses pytest with `asyncio_mode = "auto"`.

```bash
# Run full test suite
pytest tests/ -v

# Run a specific test file
pytest tests/debate/test_orchestrator.py -v

# Run tests matching a keyword pattern
pytest tests/ -k "test_consensus" -v

# Ignore known-broken connector tests
pytest tests/ --ignore=tests/connectors -v

# Run only fast unit tests
pytest tests/ -m unit

# Skip slow and integration tests
pytest tests/ -m "not slow and not integration"

# Parallel execution (requires pytest-xdist)
pytest tests/ -n auto

# With coverage
pytest tests/ --cov=aragora --cov-report=html
```

### Common pytest flags

| Flag | Purpose |
|------|---------|
| `-k "pattern"` | Run tests matching keyword expression |
| `--ignore=path` | Skip a directory (e.g., `--ignore=tests/connectors`) |
| `-m marker` | Run tests with a specific marker (`unit`, `slow`, `integration`, etc.) |
| `-n auto` | Parallel execution across CPU cores |
| `-x` | Stop on first failure |
| `--tb=short` | Shorter tracebacks |
| `-p no:randomly` | Disable random test ordering |

## Linting and Type Checking

```bash
# Ruff linter (fast)
ruff check aragora/

# Ruff with auto-fix
ruff check aragora/ --fix

# Type checking
mypy aragora/

# Security linting
bandit -r aragora/ -c pyproject.toml
```

Line length is 100 characters. See `pyproject.toml` `[tool.ruff]` and
`[tool.mypy]` sections for full configuration.

## Running Your First Debate

```python
import asyncio
from aragora import Arena, Environment, DebateProtocol

async def main():
    env = Environment(task="Should we use microservices or a monolith?")
    protocol = DebateProtocol(rounds=3, consensus="majority")
    # agents are auto-selected from available API keys
    arena = Arena(env, agents=None, protocol=protocol)
    result = await arena.run()
    print(result.summary)

asyncio.run(main())
```

## CLI Usage

Aragora installs a CLI entry point:

```bash
# Review a file with the Gauntlet
aragora review path/to/file.py

# Scan a skill for malware patterns
aragora skills scan path/to/skill.py

# Run the Nomic Loop (self-improvement)
python scripts/nomic_loop.py
```

## Troubleshooting

### Stale `.pyc` cache causing `NameError`

If you see `NameError: name 'field' is not defined` or similar import errors
after refactoring, clear the bytecode cache:

```bash
find . -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
```

### Tests fail with `DATA_DIR` or path issues

Tests that need to override the data directory must patch the function, not
the constant:

```python
# Correct
mocker.patch("aragora.persistence.paths.get_default_data_dir", return_value=tmp_path)

# Wrong -- resolve_db_path calls the function, not the constant
mocker.patch("aragora.persistence.paths.DATA_DIR", tmp_path)
```

### Import errors for optional dependencies

Many modules use graceful degradation. If you see `ModuleNotFoundError` for
packages like `redis`, `z3`, or `sentence-transformers`, install the relevant
extras:

```bash
pip install -e ".[redis]"          # Redis support
pip install -e ".[ml]"             # scikit-learn, sentence-transformers
pip install -e ".[observability]"  # OpenTelemetry, Prometheus
pip install -e ".[documents]"     # PDF, DOCX, XLSX parsing
```

### Telegram/connector collection errors

The Telegram connector test file has a pre-existing collection error. Use
`--ignore=tests/connectors` to skip it during general test runs.

### Server won't start without API keys

Use `--offline` mode for local development without any API keys configured.
This provides demo data and SQLite storage.
