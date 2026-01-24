---
title: Testing Guide
description: Testing Guide
---

# Testing Guide

Comprehensive guide for testing the Aragora codebase.

## Quick Start

```bash
# Run all tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Run specific test file
pytest tests/test_debate_convergence_comprehensive.py

# Run specific test class
pytest tests/test_debate_convergence_comprehensive.py::TestJaccardBackend

# Run specific test
pytest tests/test_debate_convergence_comprehensive.py::TestJaccardBackend::test_identical_texts
```

## Test Tiers

Use `scripts/test_tiers.sh` for common tiers:

| Tier | Command | Notes |
|------|---------|-------|
| `fast` | `scripts/test_tiers.sh fast` | Skip slow/load/e2e for rapid feedback |
| `ci` | `scripts/test_tiers.sh ci` | Mirrors the main CI test run |
| `lint` | `scripts/test_tiers.sh lint` | Black + Ruff checks |
| `typecheck` | `scripts/test_tiers.sh typecheck` | Mypy checks |
| `frontend` | `scripts/test_tiers.sh frontend` | Jest/RTL in `aragora/live` |
| `e2e` | `scripts/test_tiers.sh e2e` | Playwright E2E in `aragora/live` |

## CI Mapping

CI workflows and what they cover:

| Workflow | Purpose |
|----------|---------|
| `.github/workflows/test.yml` | Pytest matrix + smoke tests + frontend build |
| `.github/workflows/lint.yml` | Black, Ruff, mypy, ESLint, Bandit |
| `.github/workflows/e2e.yml` | Full Playwright E2E + Python E2E harness tests |
| `.github/workflows/integration.yml` | E2E harness, integration, and control plane tests |
| `.github/workflows/load-tests.yml` | Scheduled load tests and memory checks |

## Test Organization

```
tests/
├── conftest.py                    # Shared fixtures and setup
├── test_*.py                      # Main test files (400+)
├── benchmarks/                    # Performance benchmarks
├── e2e/                           # End-to-end tests with test harness
│   ├── conftest.py               # E2E-specific fixtures
│   ├── harness.py                # E2E test harness implementation
│   └── test_full_flow.py         # Full system flow tests
├── integration/                   # Integration tests
│   ├── conftest.py               # Integration-specific fixtures
│   ├── test_api_workflow.py      # API workflow tests
│   ├── test_debate_lifecycle.py  # Full debate lifecycle
│   └── test_websocket_events.py  # WebSocket event tests
├── security/                      # Security tests
│   ├── test_auth_boundaries.py   # Auth boundary tests
│   ├── test_cors.py              # CORS handling
│   ├── test_csrf_protection.py   # CSRF protection
│   ├── test_input_validation.py  # Input validation
│   ├── test_rate_limit_enforcement.py
│   └── test_sql_injection.py     # SQL injection prevention
└── storage/                       # Storage layer tests
    └── test_*.py                  # Database and persistence tests
```

## Running Tests

### Basic Commands

```bash
# Run all tests with coverage
pytest tests/ --cov=aragora --cov-report=html

# Run tests excluding slow tests
pytest tests/ -m "not slow"

# Run only integration tests
pytest tests/integration/ -m integration

# Run with specific timeout
pytest tests/ --timeout=30

# Run in parallel (requires pytest-xdist)
pytest tests/ -n auto

# Run with output capture disabled (see prints)
pytest tests/ -s
```

### Test Markers

Tests use markers to categorize them:

| Marker | Description | Usage |
|--------|-------------|-------|
| `slow` | Tests that take >5 seconds | `-m "not slow"` to skip |
| `load` | Load/stress tests | `-m load` to run only |
| `integration` | Integration tests | `-m integration` |
| `e2e` | End-to-end tests | `-m e2e` |

```bash
# Run only fast tests
pytest tests/ -m "not slow and not load"

# Run integration tests only
pytest tests/ -m integration

# Run everything except e2e
pytest tests/ -m "not e2e"
```

### Environment Variables

Some tests require specific environment configuration:

```bash
# Force specific similarity backend (avoid slow model loading)
ARAGORA_CONVERGENCE_BACKEND=jaccard pytest tests/test_debate_convergence_comprehensive.py

# Set test API keys (use mock values for unit tests)
ANTHROPIC_API_KEY=test-key pytest tests/

# Enable debug logging
ARAGORA_DEBUG=1 pytest tests/ -v
```

## Writing Tests

### Test File Naming

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test methods: `test_<behavior_description>`

```python
# tests/test_debate_convergence.py
class TestJaccardBackend:
    def test_identical_texts_return_similarity_of_one(self):
        ...
```

### Using Fixtures

Common fixtures from `conftest.py`:

```python
import pytest

def test_with_temp_database(temp_db):
    """Uses temporary SQLite database that's cleaned up after test."""
    from aragora.ranking.elo import EloSystem
    elo = EloSystem(db_path=temp_db)
    # ... test code

def test_with_mock_storage(mock_storage):
    """Uses pre-configured mock DebateStorage."""
    debates = mock_storage.list_debates()
    assert len(debates) == 2

def test_with_mock_agents(mock_agents):
    """Uses list of 3 mock agents: claude, gemini, gpt4."""
    assert len(mock_agents) == 3

def test_requiring_api_keys(mock_api_keys):
    """Sets mock ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY."""
    import os
    assert os.getenv("ANTHROPIC_API_KEY") == "test-anthropic-key"

def test_with_clean_environment(clean_env):
    """All API keys removed for clean slate testing."""
    import os
    assert os.getenv("ANTHROPIC_API_KEY") is None
```

### Async Tests

Use `pytest-asyncio` for async tests:

```python
import pytest

@pytest.mark.asyncio
async def test_async_debate():
    """Async tests are automatically detected."""
    result = await some_async_function()
    assert result is not None

class TestAsyncClass:
    @pytest.mark.asyncio
    async def test_async_method(self):
        """Async methods in classes work too."""
        pass
```

### Mocking External Services

```python
from unittest.mock import Mock, patch, AsyncMock

def test_api_call_mocked():
    """Mock external API calls."""
    with patch("aragora.agents.anthropic.AnthropicAgent._call_api") as mock:
        mock.return_value = {"content": "mocked response"}
        # ... test code

@pytest.mark.asyncio
async def test_async_api_mocked():
    """Mock async API calls."""
    with patch("module.async_function", new=AsyncMock(return_value="result")):
        result = await some_function()
        assert result == "result"
```

### Testing Error Handling

```python
import pytest

def test_raises_on_invalid_input():
    """Test that exceptions are raised correctly."""
    with pytest.raises(ValueError, match="Invalid input"):
        function_that_should_raise("bad input")

def test_custom_exception():
    """Test custom exception types."""
    from aragora.exceptions import DebateError
    with pytest.raises(DebateError):
        start_invalid_debate()
```

## Fixtures Reference

### Auto-Used Fixtures

These run automatically for every test:

| Fixture | Purpose |
|---------|---------|
| `reset_circuit_breakers` | Resets CircuitBreaker state |
| `clear_handler_cache` | Clears handler TTL cache |
| `reset_supabase_env` | Clears Supabase env vars |
| `reset_lazy_globals` | Resets module-level lazy globals |

### Available Fixtures

| Fixture | Description |
|---------|-------------|
| `temp_db` | Temporary SQLite database path |
| `temp_dir` | Temporary directory (Path) |
| `temp_nomic_dir` | Temporary nomic state directory |
| `mock_storage` | Mock DebateStorage with sample data |
| `mock_elo_system` | Mock EloSystem with sample rankings |
| `mock_agent` | Single mock agent |
| `mock_agents` | List of 3 mock agents |
| `mock_environment` | Mock Environment for arena testing |
| `mock_emitter` | Mock event emitter |
| `mock_auth_config` | Mock AuthConfig |
| `handler_context` | Complete handler context dict |
| `elo_system` | Real EloSystem with temp database |
| `continuum_memory` | Real ContinuumMemory with temp database |
| `clean_env` | Clears all API key env vars |
| `mock_api_keys` | Sets mock API keys |
| `sample_debate_messages` | Sample debate message list |
| `sample_critique` | Sample critique dict |

## Coverage

### Running Coverage

```bash
# Basic coverage
pytest tests/ --cov=aragora

# HTML report
pytest tests/ --cov=aragora --cov-report=html
open htmlcov/index.html

# Terminal report
pytest tests/ --cov=aragora --cov-report=term-missing

# Fail if below threshold
pytest tests/ --cov=aragora --cov-fail-under=70

# Coverage for specific modules
pytest tests/ --cov=aragora/debate --cov=aragora/server
```

### Current Coverage Targets

| Module | Target | Priority |
|--------|--------|----------|
| `aragora/debate/` | 80% | Critical |
| `aragora/server/handlers/` | 70% | High |
| `aragora/agents/` | 60% | Medium |
| `aragora/memory/` | 70% | High |
| `aragora/billing/` | 80% | Critical |

## E2E Test Harness

The E2E test harness provides a complete integration testing environment for testing full system workflows including the control plane, task scheduling, and debate orchestration.

### Harness Overview

The harness (`tests/e2e/harness.py`) spins up:
- ControlPlaneCoordinator
- TaskScheduler
- Mock agents with configurable behaviors
- Optional Redis/PostgreSQL connections
- Metrics and tracing support

### Running E2E Tests Locally

```bash
# Run all E2E tests
pytest tests/e2e/ -v

# Run with Redis (requires Redis running locally)
REDIS_URL=redis://localhost:6379 pytest tests/e2e/ -v

# Run with verbose CI-style logging
ARAGORA_CI=true pytest tests/e2e/ -v

# Run specific test class
pytest tests/e2e/test_full_flow.py::TestDebateIntegration -v
```

### Using the Harness in Tests

Basic usage with context manager:

```python
import pytest
from tests.e2e.harness import e2e_environment, E2ETestConfig

@pytest.mark.asyncio
async def test_task_workflow():
    async with e2e_environment() as harness:
        # Submit a task
        task_id = await harness.submit_task(
            task_type="analysis",
            payload={"input": "test data"},
            required_capabilities=["analysis"],
        )

        # Wait for completion (harness auto-processes with mock agents)
        result = await harness.wait_for_task(task_id)

        assert result is not None
        assert result.status.value == "completed"
```

Using fixtures from conftest.py:

```python
@pytest.mark.asyncio
async def test_with_fixture(e2e_harness):
    # Fixture provides pre-configured harness with 3 agents
    task_id = await e2e_harness.submit_task("test", {"data": "value"})
    result = await e2e_harness.wait_for_task(task_id)
    assert result is not None
```

### Running Debates Through the Harness

```python
@pytest.mark.asyncio
async def test_debate():
    async with e2e_environment() as harness:
        # Run a debate directly
        result = await harness.run_debate(
            topic="Should we use microservices?",
            rounds=3,
        )

        assert result is not None

        # Or run via control plane (full task lifecycle)
        result = await harness.run_debate_via_control_plane(
            topic="API design best practices",
            rounds=2,
        )

        assert result["consensus_reached"] is True
```

### Custom Agent Configuration

```python
from tests.e2e.harness import E2ETestConfig, MockAgentConfig, e2e_environment

@pytest.mark.asyncio
async def test_custom_agents():
    config = E2ETestConfig(
        num_agents=5,
        agent_capabilities=["code", "review"],
        fail_rate=0.1,  # 10% simulated failures
    )

    async with e2e_environment(config) as harness:
        # Create additional specialized agent
        agent = await harness.create_agent(
            agent_id="specialist",
            capabilities=["security", "audit"],
        )

        # Test with specialized capability
        task_id = await harness.submit_task(
            task_type="audit",
            payload={},
            required_capabilities=["security"],
        )

        result = await harness.wait_for_task(task_id)
```

### Available E2E Fixtures

| Fixture | Description |
|---------|-------------|
| `e2e_harness` | Basic harness with 3 agents, in-memory storage |
| `e2e_harness_with_redis` | Harness with Redis backend |
| `debate_harness` | Debate-focused harness with 4 agents |
| `load_test_harness` | Load testing harness with 10 agents |
| `harness_config` | Customizable configuration object |
| `mock_agent_factory` | Factory for creating mock agents |

### CI Integration

E2E tests run in CI via `.github/workflows/integration.yml`:

```bash
# What CI runs
pytest tests/e2e/ -v --tb=short --timeout=180

# With environment
REDIS_URL=redis://localhost:6379
ARAGORA_CI=true
```

The harness automatically adjusts timeouts when running in CI:
- Default timeout: 30s (local) / 60s (CI)
- Task timeout: 10s (local) / 30s (CI)

### Specialized Harnesses

**DebateTestHarness** - For debate-focused testing:

```python
from tests.e2e.harness import DebateTestHarness

harness = DebateTestHarness()
await harness.start()

# Run tracked debates
await harness.run_debate_with_tracking("Topic 1", rounds=2)
await harness.run_debate_with_tracking("Topic 2", rounds=2)

# Get metrics
rate = harness.get_consensus_rate()
results = harness.get_debate_results()

await harness.stop()
```

**LoadTestHarness** - For load/performance testing:

```python
from tests.e2e.harness import LoadTestHarness

harness = LoadTestHarness()
await harness.start()

# Submit many tasks concurrently
task_ids = await harness.submit_concurrent_tasks(count=100)

# Measure throughput
metrics = await harness.measure_throughput(task_count=50)
print(f"Tasks/second: {metrics['tasks_per_second']}")

await harness.stop()
```

## CI/CD Integration

Tests run automatically on CI:

```yaml
# .github/workflows/test.yml (simplified)
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev,research]" && pip install pytest-cov pytest-timeout
      - run: pytest tests/ -v --timeout=60 --cov=aragora --cov-report=xml --cov-report=term-missing -x --tb=short
```

### Test Commands in CI

```bash
# Full test suite
pytest tests/ -v --timeout=60 --cov=aragora --cov-report=xml --cov-report=term-missing -x --tb=short

# CLI smoke (demo)
aragora ask "Smoke test a demo debate" --demo --rounds 1

# Server smoke (non-default ports to avoid collisions)
aragora serve --api-port 8090 --ws-port 8766 --host 127.0.0.1
```

## Debugging Tests

### Using pdb

```bash
# Drop into debugger on failure
pytest tests/test_file.py --pdb

# Drop into debugger on first failure
pytest tests/test_file.py --pdb -x

# Start debugger at specific line
# Add: import pdb; pdb.set_trace() in code
pytest tests/test_file.py -s
```

### Verbose Output

```bash
# Show test names as they run
pytest tests/ -v

# Show full diff on assertion failures
pytest tests/ -vv

# Show local variables in tracebacks
pytest tests/ --tb=long

# Show only the first N failures
pytest tests/ --maxfail=3
```

### Logging

```python
import logging

def test_with_logging(caplog):
    """Capture log output during tests."""
    with caplog.at_level(logging.DEBUG):
        function_that_logs()
    assert "expected message" in caplog.text
```

## Performance Testing

### Benchmarks

```bash
# Run benchmarks
pytest tests/benchmarks/ -v

# Run with timing info
pytest tests/ --durations=10

# Profile slow tests
pytest tests/ --profile
```

### Load Tests

```bash
# Run load tests only
pytest tests/ -m load -v

# Example load test
pytest tests/integration/test_server_under_load.py -v
```

## Best Practices

### Test Isolation

1. **Use fixtures for setup/teardown** - Don't rely on test order
2. **Mock external dependencies** - Don't make real API calls
3. **Use temporary databases** - Clean state for each test
4. **Reset global state** - Use `autouse` fixtures

### Test Quality

1. **One assertion per test** (when practical)
2. **Descriptive test names** - `test_<action>_<expected_result>`
3. **Test edge cases** - Empty inputs, None values, boundaries
4. **Test error paths** - Exceptions, invalid inputs

### Performance

1. **Mock slow operations** - API calls, model loading
2. **Use `@pytest.mark.slow`** - For tests >5 seconds
3. **Parallelize with `-n auto`** - When tests are isolated
4. **Cache expensive fixtures** - Use `scope="session"` when safe

## Troubleshooting

### Common Issues

**Tests hang loading models:**
```bash
# Force fast backend
ARAGORA_CONVERGENCE_BACKEND=jaccard pytest tests/
```

**Tests fail with missing env vars:**
```bash
# Use mock API keys fixture
@pytest.fixture(autouse=True)
def setup_env(mock_api_keys):
    yield
```

**Tests pollute each other:**
```python
# Add to conftest.py
@pytest.fixture(autouse=True)
def reset_state():
    yield
    # Reset after each test
    clear_global_state()
```

**Async tests timeout:**
```python
@pytest.mark.asyncio
@pytest.mark.timeout(30)  # Explicit timeout
async def test_slow_async():
    ...
```

### Getting Help

- Check existing test files for patterns
- Review `conftest.py` for available fixtures
- Run with `-v --tb=long` for detailed errors
- Use `--pdb` to drop into debugger
