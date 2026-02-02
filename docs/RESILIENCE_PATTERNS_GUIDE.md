# Resilience Patterns Guide

This guide documents Aragora's unified resilience strategy for fault-tolerant operations across agents, connectors, and integrations.

## Overview

Aragora implements a layered resilience strategy:

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                      │
├──────────────┬──────────────┬──────────────┬────────────┤
│    Retry     │   Timeout    │   Circuit    │   Health   │
│   Strategy   │   Control    │   Breaker    │   Check    │
├──────────────┴──────────────┴──────────────┴────────────┤
│              Persistence & Metrics Layer                  │
└─────────────────────────────────────────────────────────┘
```

## Core Patterns

### 1. Circuit Breaker

Prevents cascading failures by stopping requests to failing services.

**States:**
- **CLOSED**: Normal operation, requests pass through
- **OPEN**: Requests fail immediately without calling the service
- **HALF_OPEN**: Limited requests allowed to test recovery

**Configuration:**

```python
from aragora.resilience import CircuitBreaker, get_circuit_breaker

# Using the registry (recommended)
cb = get_circuit_breaker(
    name="anthropic-api",
    provider="anthropic",
    failure_threshold=5,      # Failures before opening
    cooldown_seconds=60.0,    # Time in OPEN state
    success_threshold=2,      # Successes to close from half-open
)

# Direct instantiation
cb = CircuitBreaker(
    failure_threshold=5,
    cooldown_seconds=30.0,
    half_open_max_calls=3,
)
```

**Usage:**

```python
# Async context manager
async with cb:
    result = await call_api()

# Or with decorator
from aragora.resilience import with_resilience

@with_resilience(circuit_name="my-service")
async def call_external_service():
    ...
```

### 2. Retry Strategy

Automatically retries failed operations with configurable backoff.

**Strategies:**
- `EXPONENTIAL`: Exponential backoff with jitter
- `FIXED`: Fixed delay between retries
- `LINEAR`: Linearly increasing delay

```python
from aragora.resilience import RetryConfig, with_retry

# Configure retry behavior
config = RetryConfig(
    max_retries=3,
    strategy=RetryStrategy.EXPONENTIAL,
    base_delay=1.0,
    max_delay=30.0,
    jitter=True,
    retry_on=(ConnectionError, TimeoutError),
)

# Apply to async function
@with_retry(config)
async def fetch_data():
    ...

# Or use inline
result = await with_retry(config)(lambda: api.call())()
```

**Exponential Backoff Calculation:**

```python
from aragora.resilience import calculate_backoff_delay

delay = calculate_backoff_delay(
    attempt=2,              # Current attempt number
    base_delay=1.0,         # Starting delay
    max_delay=60.0,         # Maximum delay cap
    jitter_factor=0.1,      # Random jitter percentage
)
# Result: ~4.0 seconds (2^2 * 1.0 + jitter)
```

### 3. Timeout Control

Prevents operations from hanging indefinitely.

```python
from aragora.resilience import TimeoutConfig, with_timeout

# Configure timeouts
config = TimeoutConfig(
    default_timeout=30.0,
    connect_timeout=10.0,
    read_timeout=60.0,
)

# Apply to async function
@with_timeout(30.0)
async def slow_operation():
    ...

# Context manager
from aragora.resilience import timeout_context

async with timeout_context(30.0):
    await slow_operation()
```

### 4. Health Checking

Monitors service health and reports status.

```python
from aragora.resilience import HealthChecker, HealthStatus

class APIHealthChecker(HealthChecker):
    async def check_health(self) -> HealthReport:
        try:
            await self.client.ping()
            return HealthReport(
                status=HealthStatus.HEALTHY,
                message="API responding",
                latency_ms=15.2,
            )
        except Exception as e:
            return HealthReport(
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )
```

## Per-Connector Configuration

Different connectors have different reliability characteristics. Configure resilience per-connector:

### Agent Connectors

```python
# anthropic - Generally reliable, use moderate settings
ANTHROPIC_CONFIG = CircuitBreakerConfig(
    failure_threshold=5,
    timeout_seconds=60.0,
)

# openai - Rate limits common, use lenient settings
OPENAI_CONFIG = CircuitBreakerConfig(
    failure_threshold=10,
    timeout_seconds=30.0,
)

# openrouter - Used as fallback, strict settings
OPENROUTER_CONFIG = CircuitBreakerConfig(
    failure_threshold=3,
    timeout_seconds=120.0,
)
```

### Chat Connectors

```python
# Slack - High availability, moderate settings
SLACK_RESILIENCE = {
    "failure_threshold": 5,
    "timeout_seconds": 30.0,
    "retry_max": 3,
}

# Email - Lower expectations, lenient settings
EMAIL_RESILIENCE = {
    "failure_threshold": 10,
    "timeout_seconds": 120.0,
    "retry_max": 5,
}
```

### Storage Connectors

```python
# Supabase - Critical path, strict monitoring
SUPABASE_CONFIG = CircuitBreakerConfig(
    failure_threshold=3,
    timeout_seconds=10.0,
    success_threshold=1,
)
```

## Cascading Failure Prevention

### Bulkhead Pattern

Isolate failures by using separate circuit breakers per service:

```python
# Each agent has its own circuit breaker
anthropic_cb = get_circuit_breaker("anthropic-agent", provider="anthropic")
openai_cb = get_circuit_breaker("openai-agent", provider="openai")
gemini_cb = get_circuit_breaker("gemini-agent", provider="google")

# Failure in one doesn't affect others
async def run_debate():
    tasks = [
        call_with_circuit(anthropic_cb, anthropic_call),
        call_with_circuit(openai_cb, openai_call),
        call_with_circuit(gemini_cb, gemini_call),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
```

### Fallback Chain

Graceful degradation with fallback providers:

```python
async def call_with_fallback(prompt: str) -> str:
    providers = [
        ("anthropic", anthropic_cb),
        ("openai", openai_cb),
        ("openrouter", openrouter_cb),  # Fallback
    ]

    for name, cb in providers:
        try:
            async with cb:
                return await call_provider(name, prompt)
        except CircuitOpenError:
            logger.warning(f"{name} circuit open, trying next")
            continue

    raise AllProvidersFailedError()
```

## Metrics and Monitoring

### Built-in Metrics

```python
from aragora.resilience import (
    get_circuit_breaker_status,
    get_circuit_breaker_metrics,
    get_all_circuit_breakers_status,
)

# Single circuit status
status = get_circuit_breaker_status("anthropic-api")
# Returns: {"state": "CLOSED", "failure_count": 0, ...}

# All circuits summary
summary = get_all_circuit_breakers_status()
# Returns: {"total": 5, "open": 1, "closed": 3, "half_open": 1}

# Detailed metrics
metrics = get_circuit_breaker_metrics("anthropic-api")
# Returns: {"success_count": 150, "failure_count": 3, "last_failure": ...}
```

### Prometheus Integration

```python
from aragora.resilience import set_metrics_callback

def prometheus_callback(event: str, circuit_name: str, data: dict):
    circuit_state_gauge.labels(circuit=circuit_name).set(
        1 if data.get("state") == "OPEN" else 0
    )
    failure_counter.labels(circuit=circuit_name).inc()

set_metrics_callback(prometheus_callback)
```

## Persistence

Circuit breaker state persists across restarts:

```python
from aragora.resilience import (
    init_circuit_breaker_persistence,
    persist_all_circuit_breakers,
    load_circuit_breakers,
)

# Initialize persistence (SQLite by default)
init_circuit_breaker_persistence()

# Load persisted state on startup
await load_circuit_breakers()

# Persist state periodically or on shutdown
await persist_all_circuit_breakers()
```

## Best Practices

### 1. Configure Per-Service

Don't use global defaults. Each service has different characteristics:

```python
# Good: Per-service configuration
anthropic_cb = get_circuit_breaker("anthropic", failure_threshold=5)
flaky_api_cb = get_circuit_breaker("flaky-api", failure_threshold=2)

# Bad: Same config for everything
cb = get_circuit_breaker("service")  # Uses defaults
```

### 2. Combine Patterns

Use multiple patterns together for comprehensive resilience:

```python
@with_retry(RetryConfig(max_retries=3))
@with_timeout(30.0)
@with_resilience(circuit_name="api")
async def call_api():
    ...
```

### 3. Monitor and Alert

Set up alerting on circuit breaker state changes:

```python
def on_circuit_change(event: str, name: str, data: dict):
    if event == "circuit_opened":
        send_alert(f"Circuit {name} opened! Check service health.")

set_metrics_callback(on_circuit_change)
```

### 4. Test Failure Scenarios

Include resilience testing in your test suite:

```python
async def test_circuit_breaker_opens():
    cb = CircuitBreaker(failure_threshold=2)

    # Force failures
    for _ in range(3):
        try:
            async with cb:
                raise ConnectionError()
        except:
            pass

    # Circuit should be open
    assert cb.state == CircuitState.OPEN

    # Requests should fail fast
    with pytest.raises(CircuitOpenError):
        async with cb:
            await api_call()
```

## Environment Variables

Override resilience settings via environment:

```bash
# Circuit breaker defaults
CIRCUIT_BREAKER_FAILURE_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT_SECONDS=60

# Per-provider overrides
ANTHROPIC_FAILURE_THRESHOLD=3
OPENAI_FAILURE_THRESHOLD=10

# Retry configuration
RETRY_MAX_ATTEMPTS=3
RETRY_BASE_DELAY=1.0

# Timeout configuration
DEFAULT_TIMEOUT_SECONDS=30
CONNECT_TIMEOUT_SECONDS=10
```

## Related Documentation

- [Agent Configuration](./AGENTS.md)
- [Connector Setup](./CONNECTORS.md)
- [Monitoring Guide](./MONITORING.md)
- [Disaster Recovery](./DISASTER_RECOVERY.md)
