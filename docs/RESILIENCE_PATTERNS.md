# Resilience Patterns Guide

This guide documents the resilience patterns available in Aragora for building fault-tolerant systems.

## Overview

Aragora provides a comprehensive resilience package (`aragora.resilience`) with patterns for:

- **Circuit Breaker**: Prevents cascading failures by stopping requests to failing services
- **Retry**: Automatic retry with configurable backoff strategies
- **Timeout**: Enforces time limits on operations
- **Health Checking**: Monitors service health and availability

## Quick Start

```python
from aragora.resilience import (
    CircuitBreaker,
    get_circuit_breaker,
    with_resilience,
    with_retry,
    with_timeout,
)

# Using the global registry (recommended)
cb = get_circuit_breaker("anthropic-api", provider="anthropic")

# Or create directly
cb = CircuitBreaker(failure_threshold=5, cooldown_seconds=30)

# Execute with circuit breaker
result = await cb.execute(api_call)
```

---

## Circuit Breaker

### Concept

The circuit breaker pattern prevents an application from repeatedly trying to execute an operation that's likely to fail. It has three states:

| State | Description | Behavior |
|-------|-------------|----------|
| **CLOSED** | Normal operation | Requests pass through, failures tracked |
| **OPEN** | Failures exceeded threshold | Requests fail immediately (CircuitOpenError) |
| **HALF_OPEN** | Testing recovery | Limited requests allowed to test service |

### Basic Usage

```python
from aragora.resilience import CircuitBreaker, CircuitOpenError

cb = CircuitBreaker(
    failure_threshold=5,     # Open after 5 failures
    cooldown_seconds=30,     # Wait 30s before testing recovery
    half_open_max_calls=3,   # Allow 3 test calls in half-open state
)

try:
    result = await cb.execute(my_async_function, arg1, arg2)
except CircuitOpenError:
    # Circuit is open - use fallback
    result = get_fallback_result()
except Exception as e:
    # Operation failed (circuit may have opened)
    handle_error(e)
```

### Using the Registry

For applications with multiple services, use the global registry:

```python
from aragora.resilience import get_circuit_breaker, reset_all_circuit_breakers

# Get or create a circuit breaker by name
cb = get_circuit_breaker(
    name="openai-api",
    provider="openai",  # Optional metadata
    failure_threshold=3,
    cooldown_seconds=60,
)

# Get status
status = cb.get_status()
print(f"State: {status['state']}, Failures: {status['failure_count']}")

# Reset all circuit breakers (useful for testing)
reset_all_circuit_breakers()
```

### Decorator Pattern

```python
from aragora.resilience import with_resilience

@with_resilience(circuit_name="external-api")
async def call_external_api(endpoint: str, data: dict):
    async with aiohttp.ClientSession() as session:
        async with session.post(endpoint, json=data) as resp:
            return await resp.json()
```

### Monitoring and Metrics

```python
from aragora.resilience import (
    get_circuit_breaker_status,
    get_circuit_breaker_metrics,
    get_all_circuit_breakers_status,
    set_metrics_callback,
)

# Get status for a specific circuit
status = get_circuit_breaker_status("openai-api")
# Returns: {"state": "closed", "failure_count": 2, "last_failure": "..."}

# Get metrics
metrics = get_circuit_breaker_metrics("openai-api")
# Returns: {"total_calls": 100, "failures": 5, "success_rate": 0.95}

# Get all circuit breakers
all_status = get_all_circuit_breakers_status()

# Custom metrics callback (for Prometheus, etc.)
def my_metrics_callback(event_type: str, circuit_name: str, details: dict):
    prometheus_counter.labels(event_type, circuit_name).inc()

set_metrics_callback(my_metrics_callback)
```

### Persistence

Circuit breaker state can be persisted to survive restarts:

```python
from aragora.resilience import (
    init_circuit_breaker_persistence,
    persist_all_circuit_breakers,
    load_circuit_breakers,
)

# Initialize persistence (call at startup)
init_circuit_breaker_persistence(db_path="data/circuit_breakers.db")

# Load persisted state
load_circuit_breakers()

# Persist current state (call periodically or on shutdown)
persist_all_circuit_breakers()
```

---

## Retry Pattern

### Basic Usage

```python
from aragora.resilience import with_retry, RetryConfig

@with_retry(max_attempts=3, delay_seconds=1.0, exponential_backoff=True)
async def unreliable_operation():
    return await external_service.call()

# Or with config object
config = RetryConfig(
    max_attempts=5,
    initial_delay=0.5,
    max_delay=30.0,
    exponential_base=2.0,
    jitter=True,  # Add randomness to prevent thundering herd
)

@with_retry(config=config)
async def my_function():
    ...
```

### Retry with Exception Filtering

```python
from aragora.resilience import with_retry

# Only retry on specific exceptions
@with_retry(
    max_attempts=3,
    retry_on=(ConnectionError, TimeoutError),
    dont_retry_on=(ValueError, KeyError),
)
async def api_call():
    ...
```

### Synchronous Retry

```python
from aragora.resilience import with_retry_sync

@with_retry_sync(max_attempts=3)
def sync_operation():
    return requests.get("https://api.example.com/data")
```

---

## Timeout Pattern

### Basic Usage

```python
from aragora.resilience import with_timeout, TimeoutConfig
import asyncio

@with_timeout(seconds=10.0)
async def slow_operation():
    return await external_service.call()

# Using config
config = TimeoutConfig(
    seconds=30.0,
    on_timeout="raise",  # or "return_none" or "return_default"
    default_value=None,
)

@with_timeout(config=config)
async def my_function():
    ...
```

### Timeout with Fallback

```python
@with_timeout(seconds=5.0, on_timeout="return_default", default_value={"error": "timeout"})
async def api_with_fallback():
    return await slow_api.call()
```

### Synchronous Timeout

```python
from aragora.resilience import with_timeout_sync

@with_timeout_sync(seconds=30.0)
def blocking_operation():
    return requests.get("https://slow-api.example.com")
```

---

## Health Checking

### Basic Usage

```python
from aragora.resilience import HealthChecker, HealthStatus, HealthReport

class MyServiceHealthChecker(HealthChecker):
    async def check(self) -> HealthReport:
        try:
            # Perform health check
            is_healthy = await self.service.ping()
            return HealthReport(
                status=HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY,
                message="Service is operational" if is_healthy else "Service unreachable",
                details={"latency_ms": 50},
            )
        except Exception as e:
            return HealthReport(
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )
```

### Composite Health Checks

```python
from aragora.resilience import CompositeHealthChecker

checker = CompositeHealthChecker([
    DatabaseHealthChecker(db),
    RedisHealthChecker(redis),
    ExternalAPIHealthChecker(api),
])

report = await checker.check()
# Returns aggregated health status
```

---

## Combined Patterns

### Circuit Breaker + Retry + Timeout

```python
from aragora.resilience import with_resilience

@with_resilience(
    circuit_name="external-api",
    circuit_failure_threshold=5,
    circuit_cooldown_seconds=30,
    retry_max_attempts=3,
    retry_delay_seconds=1.0,
    timeout_seconds=10.0,
)
async def robust_api_call(endpoint: str):
    """
    This call will:
    1. Timeout after 10 seconds
    2. Retry up to 3 times with 1s delay
    3. Trip circuit breaker after 5 failures
    """
    return await http_client.get(endpoint)
```

---

## Integration with Aragora Components

### Agent Resilience

```python
from aragora.agents.airlock import AirlockProxy

# Wrap agent with resilience (automatic circuit breaker + retry)
airlock = AirlockProxy(agent, config=AirlockConfig(
    circuit_breaker_threshold=3,
    retry_on_rate_limit=True,
    fallback_response="I'm currently unavailable. Please try again later.",
))
```

### Arena Circuit Breaker

The debate arena uses circuit breakers for agent failure handling:

```python
from aragora import Arena

arena = Arena(
    environment=env,
    agents=agents,
    circuit_breaker=CircuitBreaker(failure_threshold=3),
)
```

### HTTP Client Resilience

```python
from aragora.resilience.http_client import ResilientHTTPClient

client = ResilientHTTPClient(
    base_url="https://api.example.com",
    timeout=30.0,
    retry_config=RetryConfig(max_attempts=3),
    circuit_breaker_config=CircuitBreakerConfig(failure_threshold=5),
)

response = await client.get("/endpoint")
```

---

## Best Practices

### 1. Choose Appropriate Thresholds

| Service Type | Failure Threshold | Cooldown | Retry Attempts |
|--------------|-------------------|----------|----------------|
| External API | 5-10 | 30-60s | 3-5 |
| Database | 3-5 | 10-30s | 2-3 |
| Message Queue | 5-10 | 15-30s | 3-5 |
| AI Provider | 3-5 | 60-120s | 2-3 |

### 2. Use Exponential Backoff

```python
# Good: exponential backoff with jitter
@with_retry(max_attempts=5, exponential_backoff=True, jitter=True)
async def my_call():
    ...

# Avoid: fixed delay can cause thundering herd
@with_retry(max_attempts=5, delay_seconds=1.0)  # Less ideal
async def my_call():
    ...
```

### 3. Implement Fallbacks

```python
from aragora.resilience import CircuitOpenError

async def get_data():
    try:
        return await circuit_breaker.execute(fetch_from_primary)
    except CircuitOpenError:
        # Primary is down, use cached/fallback
        return get_cached_data()
    except Exception:
        return get_default_data()
```

### 4. Monitor Circuit State

```python
# Expose metrics endpoint
@app.get("/health/circuits")
async def circuit_health():
    return get_all_circuit_breakers_status()
```

### 5. Test Failure Scenarios

```python
def test_circuit_breaker_opens():
    cb = CircuitBreaker(failure_threshold=2)

    # Simulate failures
    for _ in range(2):
        with pytest.raises(ValueError):
            await cb.execute(failing_function)

    # Circuit should be open
    with pytest.raises(CircuitOpenError):
        await cb.execute(any_function)
```

---

## Configuration Reference

### CircuitBreaker

| Parameter | Default | Description |
|-----------|---------|-------------|
| `failure_threshold` | 5 | Failures before opening circuit |
| `cooldown_seconds` | 30 | Time to wait before testing recovery |
| `half_open_max_calls` | 3 | Max calls allowed in half-open state |
| `success_threshold` | 2 | Successes needed to close circuit |

### RetryConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_attempts` | 3 | Maximum retry attempts |
| `initial_delay` | 1.0 | Initial delay between retries (seconds) |
| `max_delay` | 60.0 | Maximum delay cap |
| `exponential_base` | 2.0 | Base for exponential backoff |
| `jitter` | True | Add randomness to delays |

### TimeoutConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `seconds` | 30.0 | Timeout duration |
| `on_timeout` | "raise" | Action on timeout: "raise", "return_none", "return_default" |
| `default_value` | None | Value to return if `on_timeout="return_default"` |

---

## File Locations

| Component | Location |
|-----------|----------|
| Circuit Breaker | `aragora/resilience/circuit_breaker.py` |
| Registry | `aragora/resilience/registry.py` |
| Retry | `aragora/resilience/retry.py` |
| Timeout | `aragora/resilience/timeout.py` |
| Health | `aragora/resilience/health.py` |
| HTTP Client | `aragora/resilience/http_client.py` |
| Metrics | `aragora/resilience/metrics.py` |
| Persistence | `aragora/resilience/persistence.py` |
| Decorator | `aragora/resilience/decorator.py` |

---

*Last updated: 2026-02-01*
