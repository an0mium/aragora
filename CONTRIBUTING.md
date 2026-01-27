# Contributing to Aragora

Thank you for your interest in contributing to Aragora! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.10+
- Node.js 18+ (for frontend development)
- Docker (optional, for containerized development)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/aragora-ai/aragora.git
cd aragora

# Install development dependencies
make dev

# Run tests to verify setup
make test-fast

# Start the development server
make serve
```

### Using VS Code Dev Container

1. Install the "Dev Containers" extension in VS Code
2. Open the project folder
3. Click "Reopen in Container" when prompted
4. Wait for the container to build and dependencies to install

## Development Workflow

### Running Tests

```bash
# Run all tests
make test

# Run fast tests only (excludes slow, e2e, load tests)
make test-fast

# Run tests with coverage
make test-cov

# Run specific test file
pytest tests/debate/test_orchestrator.py -v
```

### Code Quality

```bash
# Run linter
make lint

# Format code
make format

# Type checking
make typecheck

# Run all checks
make check
```

### Development Server

```bash
# Start API server
make serve

# Interactive debate REPL
make repl

# System health check
make doctor
```

## Code Style

### Python

- Follow PEP 8 with a line length of 88 characters
- Use type hints for all function signatures
- Use `ruff` for linting and formatting
- Use `mypy` for type checking

### Commit Messages

Follow conventional commits format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(marketplace): add template rating system
fix(debate): handle timeout in consensus phase
docs(api): update authentication guide
```

### Pull Request Process

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes
4. Run tests and checks: `make check && make test`
5. Commit with a descriptive message
6. Push to your fork: `git push origin feat/my-feature`
7. Open a Pull Request

## Architecture Overview

Aragora is a **control plane for multi-agent robust decisionmaking**. Here's how the key systems fit together:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              User Request                                │
│                    (CLI, HTTP API, WebSocket, Chat)                      │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Server Layer (unified_server.py)                 │
│  • 275+ HTTP endpoints  • WebSocket streaming  • Handler registry        │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
             ┌──────────┐   ┌──────────┐   ┌──────────┐
             │  Arena   │   │ Knowledge│   │ Control  │
             │ (Debate) │   │  Mound   │   │  Plane   │
             └────┬─────┘   └────┬─────┘   └────┬─────┘
                  │              │              │
    ┌─────────────┼─────────────┐│              │
    ▼             ▼             ▼▼              ▼
┌───────┐   ┌─────────┐   ┌──────────┐   ┌──────────┐
│Agents │   │ Memory  │   │ Evidence │   │  RBAC    │
│(15+)  │   │Continuum│   │ & Pulse  │   │ & Audit  │
└───────┘   └─────────┘   └──────────┘   └──────────┘
```

### Core Components

| Component | Location | Purpose |
|-----------|----------|---------|
| **Arena** | `aragora/debate/orchestrator.py` | Multi-agent debate orchestration with phases, consensus, and convergence |
| **Agents** | `aragora/agents/` | 15+ AI model integrations (Claude, GPT, Gemini, Mistral, etc.) |
| **Memory** | `aragora/memory/continuum.py` | 4-tier memory system (fast/medium/slow/glacial) |
| **Knowledge Mound** | `aragora/knowledge/mound/` | Organizational knowledge with semantic search |
| **Server** | `aragora/server/` | HTTP/WebSocket API with 70+ handlers |
| **Control Plane** | `aragora/control_plane/` | Agent registry, scheduling, policy governance |

### Key Patterns

- **Protocol-based composition**: Features like calibration, rhetorical analysis, and trickster detection are enabled via `DebateProtocol` flags
- **Circuit breaker resilience**: All external calls use `aragora/resilience.py` for fault tolerance
- **Adapter pattern**: Knowledge Mound uses 14 adapters to integrate with subsystems
- **Event-driven streaming**: WebSocket events for real-time debate updates

## Project Structure

```
aragora/
├── aragora/           # Main package
│   ├── agents/        # Agent implementations (API + CLI)
│   ├── debate/        # Debate orchestration (Arena, phases, consensus)
│   ├── memory/        # Memory systems (Continuum, 4-tier)
│   ├── knowledge/     # Knowledge Mound with 14 adapters
│   ├── server/        # HTTP/WebSocket API (275+ endpoints)
│   ├── control_plane/ # Enterprise orchestration
│   ├── connectors/    # 130+ external integrations
│   ├── cli/           # CLI commands
│   └── rbac/          # Role-based access control
├── tests/             # 50,000+ tests across 1,600+ files
├── docs/              # Documentation (212 files)
└── sdk/typescript/    # TypeScript SDK
```

## Package Naming

### Python Packages

| Package | Purpose | Install | Import |
|---------|---------|---------|--------|
| `aragora` | Full control plane + SDK | `pip install aragora` | `from aragora.client import AragoraClient` |
| `aragora-client` | Lightweight async SDK | `pip install aragora-client` | `from aragora_client import AragoraClient` |
| `aragora-sdk` | Deprecated | _avoid_ | _deprecated_ |

```python
# Full control plane package
from aragora.client import AragoraClient

# Lightweight SDK (async)
from aragora_client import AragoraClient as AragoraAsyncClient
```

### TypeScript Packages

Two npm packages exist, but `@aragora/sdk` is the official TypeScript SDK:

| Package | Purpose | Install |
|---------|---------|---------|
| `@aragora/sdk` | Official SDK (recommended) - workflows, explainability, marketplace | `npm install @aragora/sdk` |
| `@aragora/client` | **Deprecated** legacy client - `/api/v1` compatibility | `npm install @aragora/client` |

```typescript
// For application developers
import { createClient } from '@aragora/sdk';

// Legacy client (deprecated)
import { AragoraClient } from '@aragora/client';
```

See [aragora-js/README.md](aragora-js/README.md) and [sdk/typescript/README.md](sdk/typescript/README.md) for detailed feature comparison.

### Version Synchronization

All packages maintain version parity. The following must stay in sync:

| File | Version Location |
|------|------------------|
| `pyproject.toml` | `project.version` |
| `aragora/__version__.py` | `__version__` |
| `sdk/typescript/package.json` | `version` |
| `aragora-js/package.json` | `version` |
| `aragora/live/package.json` | `version` |

CI automatically validates version parity on every build. To check locally:

```bash
# Verify all versions match
python -c "
import json, tomllib
py = tomllib.load(open('pyproject.toml', 'rb'))['project']['version']
sdk = json.load(open('sdk/typescript/package.json'))['version']
client = json.load(open('aragora-js/package.json'))['version']
print(f'Python: {py}, SDK: {sdk}, Client: {client}')
assert py == sdk == client, 'Version mismatch!'
"
```

> **Note**: The TypeScript packages (`@aragora/sdk` and `@aragora/client`) will be consolidated in v3.0.0. See [docs/SDK_CONSOLIDATION.md](docs/SDK_CONSOLIDATION.md) for the roadmap.

## Adding New Features

### Adding a New Agent

1. Create agent file in `aragora/agents/api_agents/`
2. Implement the `Agent` protocol
3. Register in `aragora/agents/__init__.py`
4. Add tests in `tests/agents/`

### Adding a CLI Command

1. Create command file in `aragora/cli/`
2. Register in `aragora/cli/main.py`
3. Add tests in `tests/cli/`

### Adding a Server Handler

1. Create handler in `aragora/server/handlers/`
2. Register routes in server startup
3. Add tests in `tests/server/handlers/`

## Testing Guidelines

### Test Organization

Tests mirror the source structure:
- `tests/debate/` → `aragora/debate/`
- `tests/server/handlers/` → `aragora/server/handlers/`

### Running Specific Tests

```bash
# Run tests for a specific module
pytest tests/debate/test_orchestrator.py -v

# Run tests matching a pattern
pytest tests/ -k "consensus" -v

# Run tests with markers
pytest tests/ -m "not slow and not e2e" -v

# Run with coverage for specific module
pytest tests/debate/ --cov=aragora/debate --cov-report=term-missing
```

### Test Markers

| Marker | Purpose | Example |
|--------|---------|---------|
| `@pytest.mark.slow` | Tests taking >10s | Integration tests |
| `@pytest.mark.e2e` | End-to-end tests | Full server tests |
| `@pytest.mark.load` | Load/stress tests | Performance tests |
| `@pytest.mark.serial` | Must run sequentially | Singleton tests |

### Writing Good Tests

```python
# Use descriptive names
async def test_arena_reaches_consensus_with_majority_vote():
    ...

# Use fixtures for common setup
@pytest.fixture
def mock_agent():
    return AsyncMock(spec=Agent)

# Test edge cases
async def test_arena_handles_agent_timeout_gracefully():
    ...
```

## Debugging Tips

### Common Issues

**Import Errors**
```bash
# Check if module exists
python -c "import aragora.debate.orchestrator"

# Check for circular imports
python -c "import aragora" 2>&1 | grep -i "circular"
```

**Test Failures**
```bash
# Run with verbose output
pytest tests/path/to/test.py -v --tb=long

# Run single test with debugging
pytest tests/path/to/test.py::test_name -v -s --capture=no

# Check for async issues
pytest tests/path/to/test.py --asyncio-mode=auto -v
```

**Server Issues**
```bash
# Check server health
curl http://localhost:8080/api/health | jq

# View server logs
ARAGORA_LOG_LEVEL=DEBUG python -m aragora.server.unified_server

# Check WebSocket connection
websocat ws://localhost:8765/ws
```

**Memory/Database Issues**
```bash
# Check database integrity
python -c "
from aragora.memory.continuum import ContinuumMemory
cm = ContinuumMemory()
print(cm.get_stats())
"

# Clear test database
rm -f ~/.aragora/test_*.db
```

### Debugging Debates

```python
# Enable verbose debate logging
from aragora import Arena, Environment, DebateProtocol

protocol = DebateProtocol(
    rounds=3,
    enable_debug_logging=True,  # Verbose output
)

# Run with tracing
arena = Arena(env, agents, protocol, enable_tracing=True)
result = await arena.run()
print(result.trace)  # Full execution trace
```

### Performance Profiling

```bash
# Profile a specific test
python -m cProfile -o profile.out -m pytest tests/debate/test_orchestrator.py::test_arena_run -v
python -c "import pstats; p = pstats.Stats('profile.out'); p.sort_stats('cumulative').print_stats(20)"

# Memory profiling
pip install memory_profiler
python -m memory_profiler your_script.py
```

## Getting Help

- Check existing issues for similar problems
- Open a new issue with a clear description
- Join our community discussions
- Read the relevant `docs/*.md` files for your area

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
