# Aragora Test Infrastructure

This document describes the test infrastructure, mocking patterns, and how to run tests without external dependencies.

## Quick Start

```bash
# Run all tests (fast, no external deps needed)
pytest tests/ -m "not integration and not network and not slow"

# Run with coverage
pytest tests/ --cov=aragora --cov-report=term-missing

# Run specific test file
pytest tests/debate/test_orchestrator.py -v
```

## Test Markers

| Marker | Description | Requires |
|--------|-------------|----------|
| `@pytest.mark.unit` | Fast unit tests | Nothing |
| `@pytest.mark.slow` | Long-running tests (>30s) | Real ML models |
| `@pytest.mark.integration` | External service tests | Redis, PostgreSQL |
| `@pytest.mark.network` | Real API calls | API keys |
| `@pytest.mark.e2e` | End-to-end tests | Running server |
| `@pytest.mark.serial` | Must run serially | Nothing |
| `@pytest.mark.knowledge` | Knowledge Mound tests | Nothing |

## Running Without API Keys

All unit tests run **without API keys** by default. The `tests/conftest.py` provides comprehensive mocking that is automatically applied.

### Automatic Mocking (autouse=True)

These fixtures are applied automatically to all tests (except those marked `integration` or `network`):

| Fixture | What It Mocks | Behavior |
|---------|---------------|----------|
| `mock_external_apis` | OpenAI, Anthropic, HTTPX clients | Deterministic responses based on input hash |
| `mock_sentence_transformers` | SentenceTransformer, CrossEncoder | Fast embeddings without model download |
| `fast_convergence_backend` | Convergence detection | Uses Jaccard instead of ML |
| `reset_circuit_breakers` | Circuit breaker state | Clean state per test |

### Mock Response Patterns

Mocks return **deterministic responses** based on input hash:
- Same input always produces same output
- Different inputs produce different outputs
- Both sync and async APIs are supported

```python
# Example: MockOpenAI returns consistent responses
response1 = await client.chat.completions.create(messages=[{"content": "hello"}])
response2 = await client.chat.completions.create(messages=[{"content": "hello"}])
assert response1.choices[0].message.content == response2.choices[0].message.content
```

## Optional Dependency Handling

Tests gracefully skip when optional dependencies are missing:

```python
# In conftest.py - available as HAS_* constants
HAS_Z3 = _check_import("z3")
HAS_REDIS = _check_import("redis")
HAS_ASYNCPG = _check_import("asyncpg")
HAS_SUPABASE = _check_import("supabase")
HAS_HTTPX = _check_import("httpx")
HAS_PYJWT = _check_import("jwt")
HAS_SKLEARN = _check_import("sklearn")
HAS_SENTENCE_TRANSFORMERS = _check_import("sentence_transformers")
HAS_MCP = _check_import("mcp")

# In tests - use skipif decorators
@pytest.mark.skipif(not HAS_Z3, reason="Z3 not installed")
def test_formal_verification():
    ...
```

## Running Integration Tests

Integration tests require external services:

```bash
# Start services
docker compose -f docker-compose.test.yml up -d

# Run integration tests
pytest tests/integration/ -v

# Stop services
docker compose -f docker-compose.test.yml down
```

### Required Services

| Service | Port | Environment Variable |
|---------|------|---------------------|
| Redis | 6379 | `REDIS_URL=redis://localhost:6379` |
| PostgreSQL | 5432 | `DATABASE_URL=postgresql://user:pass@localhost/test` |

## Test Fixtures

### Mock Agents (tests/integration/conftest.py)

| Fixture | Description |
|---------|-------------|
| `MockAgent` | Configurable responses, call tracking |
| `FailingAgent` | Fails after N successful calls |
| `SlowAgent` | Configurable response delay |
| `mock_agents()` | Standard set of debate agents |
| `consensus_agents()` | Agents configured to reach consensus |
| `split_vote_agents()` | Agents that produce split votes |

### Database Fixtures

| Fixture | Description |
|---------|-------------|
| `temp_db_path` | Temporary SQLite database |
| `critique_store` | CritiqueStore with temp DB |
| `memory_store` | ContinuumMemory with temp storage |
| `elo_system` | ELO ranking system with temp DB |

## Skip Marker Audit

The test suite has automated skip tracking:

```bash
# Run skip audit
python scripts/audit_test_skips.py

# Update documentation
python scripts/audit_test_skips.py --update-docs

# Check current baseline
cat tests/.skip_baseline
```

See [SKIP_AUDIT.md](SKIP_AUDIT.md) for the full skip marker report.

## CI Configuration

| Workflow | Purpose | Runs On |
|----------|---------|---------|
| `test.yml` | Fast PR checks | Every PR |
| `integration.yml` | Full integration suite | Nightly, manual |
| `e2e.yml` | End-to-end tests | Nightly, manual |

### Test Categories in CI

```yaml
# Fast tests (PR checks)
pytest tests/ -m "not slow and not load and not e2e"

# Minimal integration baseline (PR checks)
pytest tests/integration/ -m "integration_minimal"

# Integration tests (nightly)
pytest tests/integration/ -m "integration or not slow"

# E2E tests (nightly)
pytest tests/e2e/ -v --timeout=180
```

## Writing New Tests

### Best Practices

1. **Use fixtures** - Don't create test data manually
2. **Avoid real API calls** - Use mock fixtures or mark with `@pytest.mark.network`
3. **Clean up state** - Use `tmp_path` or `temp_db_path` fixtures
4. **Mark appropriately** - Use markers for slow/integration/e2e tests

### Example Test

```python
import pytest
from aragora.debate import Arena, Environment, DebateProtocol

@pytest.mark.asyncio
async def test_debate_reaches_consensus(mock_agents, consensus_agents):
    """Test that debate with consensus agents reaches agreement."""
    env = Environment(task="Test question")
    protocol = DebateProtocol(rounds=3)
    arena = Arena(env, consensus_agents, protocol)

    result = await arena.run()

    assert result.consensus is not None
    assert result.consensus.strength in ("strong", "unanimous")
```

### Opting Out of Mocking

For tests that need real behavior:

```python
@pytest.mark.network  # Allows real network calls
@pytest.mark.slow     # Uses real ML models
async def test_real_api_integration():
    # This test makes real API calls
    ...
```

## Troubleshooting

### Tests Timeout

```bash
# Increase timeout
pytest tests/ --timeout=120

# Run with verbose output
pytest tests/ -v --tb=long
```

### Missing Dependencies

```bash
# Install test dependencies
pip install -e ".[test]"

# Check available optional deps
python -c "from tests.conftest import *; print([k for k, v in globals().items() if k.startswith('HAS_')])"
```

### Flaky Tests

```bash
# Run multiple times to detect flakiness
pytest tests/path/to/test.py --count=5

# Run with random seed
pytest tests/ -p randomly --randomly-seed=12345
```
