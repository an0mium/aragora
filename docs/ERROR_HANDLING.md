# Error Handling Guide

Comprehensive guide to error handling patterns in Aragora, covering the exception hierarchy,
circuit breakers, retry strategies, fallback chains, and middleware-level error processing.

## Table of Contents

- [Exception Hierarchy](#exception-hierarchy)
- [Error Categories and HTTP Mapping](#error-categories-and-http-mapping)
- [Circuit Breakers](#circuit-breakers)
- [Retry Strategies](#retry-strategies)
- [Fallback Chains](#fallback-chains)
- [Agent Error Handling](#agent-error-handling)
- [Server Middleware](#server-middleware)
- [Custom Error Types](#custom-error-types)
- [Best Practices](#best-practices)

---

## Exception Hierarchy

All Aragora exceptions inherit from `AragoraError`, enabling unified error handling
across the entire codebase. The hierarchy is defined in `aragora/exceptions.py`:

```
AragoraError (base)
├── DebateError
│   ├── DebateNotFoundError
│   ├── DebateConfigurationError
│   ├── ConsensusError
│   ├── ConsensusTimeoutError
│   ├── PhaseExecutionError
│   ├── EarlyStopError
│   └── RoundLimitExceededError
├── ValidationError
│   ├── InputValidationError
│   ├── SchemaValidationError
│   └── JSONParseError
├── StorageError
│   ├── DatabaseError
│   ├── DatabaseConnectionError
│   └── RecordNotFoundError
├── MemoryError
│   ├── MemoryRetrievalError
│   ├── MemoryStorageError
│   ├── TierTransitionError
│   └── EmbeddingError
├── AuthError
│   ├── AuthenticationError
│   ├── AuthorizationError
│   ├── TokenExpiredError
│   └── RateLimitExceededError
├── InfrastructureError
│   ├── RedisUnavailableError
│   ├── ExternalServiceError
│   └── CircuitBreakerError
├── AgentError (aragora.agents.errors)
│   ├── AgentConnectionError
│   ├── AgentTimeoutError
│   ├── AgentRateLimitError
│   ├── AgentAPIError
│   ├── AgentResponseError
│   ├── AgentStreamError
│   ├── AgentCircuitOpenError
│   └── CLIAgentError
├── ConnectorError (aragora.connectors.exceptions)
│   ├── ConnectorTimeoutError
│   ├── ConnectorRateLimitError
│   └── ConnectorNetworkError
├── NomicError
│   ├── NomicCycleError
│   ├── NomicPhaseError
│   └── NomicTimeoutError
└── StreamingError
    ├── WebSocketError
    └── StreamConnectionError
```

Every `AragoraError` carries a `message` and a `details` dict for structured context:

```python
from aragora.exceptions import AragoraError

try:
    await some_operation()
except AragoraError as e:
    logger.error(f"Operation failed: {e.message}", extra=e.details)
```

---

## Error Categories and HTTP Mapping

The exception handler middleware (`aragora/server/middleware/exception_handler.py`)
maps every exception type to an HTTP status code. The full mapping is in the
`EXCEPTION_STATUS_MAP` dictionary.

### Key Mappings

| Category | Exception Types | HTTP Status |
|----------|----------------|-------------|
| **Client Errors** | `ValueError`, `ValidationError`, `InputValidationError` | 400 |
| **Not Found** | `FileNotFoundError`, `DebateNotFoundError`, `RecordNotFoundError` | 404 |
| **Authentication** | `AuthenticationError`, `TokenExpiredError`, `APIKeyError` | 401 |
| **Authorization** | `PermissionError`, `AuthorizationError` | 403 |
| **Rate Limiting** | `RateLimitExceededError`, `AgentRateLimitError` | 429 |
| **Server Errors** | `RuntimeError`, `DatabaseError`, `DebateError` | 500 |
| **Service Unavailable** | `DatabaseConnectionError`, `AgentCircuitOpenError` | 503 |
| **Timeout** | `TimeoutError`, `AgentTimeoutError`, `ConsensusTimeoutError` | 504 |
| **Graceful Stop** | `EarlyStopError`, `RoundLimitExceededError` | 200 |

### Utility Functions

```python
from aragora.server.middleware.exception_handler import (
    is_client_error,
    is_server_error,
    is_retryable,
    is_authentication_error,
    map_exception_to_status,
)

# Check error category
if is_retryable(exc):
    # Status 429, 502, 503, or 504 - safe to retry
    await retry_operation()
elif is_client_error(exc):
    # Status 4xx - do not retry, fix the request
    return error_response(exc)
```

---

## Circuit Breakers

Circuit breakers prevent cascading failures by temporarily blocking calls to
failing services. The implementation lives in `aragora/resilience/circuit_breaker.py`.

### States

| State | Behavior |
|-------|----------|
| **CLOSED** | Normal operation. Requests flow through. Failures are counted. |
| **OPEN** | After `failure_threshold` consecutive failures. All requests are blocked. |
| **HALF-OPEN** | After `cooldown_seconds` elapse. Trial requests are allowed; successes close the circuit. |

### Basic Usage

```python
from aragora.resilience.circuit_breaker import CircuitBreaker

# Single-entity mode
breaker = CircuitBreaker(
    name="my-service",
    failure_threshold=3,      # Open after 3 failures
    cooldown_seconds=60.0,    # Wait 60s before retrying
)

if breaker.can_proceed():
    try:
        result = await call_api()
        breaker.record_success()
    except Exception:
        breaker.record_failure()
```

### Multi-Entity Mode

Track circuit state per provider or agent independently:

```python
breaker = CircuitBreaker(
    name="agents",
    failure_threshold=3,
    cooldown_seconds=60.0,
    half_open_success_threshold=2,  # 2 successes to fully close
)

# Each agent has independent state
if breaker.is_available("claude"):
    try:
        result = await claude_agent.generate(prompt)
        breaker.record_success("claude")
    except Exception:
        breaker.record_failure("claude")

# Check which agents are available
available = breaker.get_available_providers()
```

### Protected Call Context Manager

The recommended pattern uses the `protected_call` context manager:

```python
from aragora.resilience.circuit_breaker import CircuitBreaker, CircuitOpenError

breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=30.0)

try:
    async with breaker.protected_call(entity="openai"):
        result = await openai_agent.generate(prompt)
        # Success is automatically recorded
except CircuitOpenError as e:
    logger.warning(f"Circuit open: {e.circuit_name}, retry in {e.cooldown_remaining:.1f}s")
except Exception:
    # Failure is automatically recorded by the context manager
    pass
```

### Per-Provider Configuration

Each AI provider has tuned circuit breaker defaults in `aragora/resilience_config.py`:

| Provider | Failure Threshold | Cooldown (s) | Success Threshold | Half-Open Max Calls |
|----------|:-----------------:|:------------:|:-----------------:|:-------------------:|
| Anthropic | 3 | 30 | 2 | 2 |
| OpenAI | 5 | 60 | 2 | 3 |
| Mistral | 4 | 45 | 2 | 2 |
| OpenRouter | 5 | 90 | 3 | 2 |
| xAI/Grok | 3 | 60 | 2 | 2 |
| Gemini | 4 | 45 | 2 | 3 |
| Default | 5 | 60 | 2 | 3 |

```python
from aragora.resilience_config import get_circuit_breaker_config, CircuitBreakerConfig
from aragora.resilience.circuit_breaker import CircuitBreaker

# Get provider-specific config
config = get_circuit_breaker_config(provider="anthropic")
breaker = CircuitBreaker.from_config(config, name="anthropic-cb")

# Register custom agent-level config
from aragora.resilience_config import register_agent_config

register_agent_config(
    "claude-sonnet",
    CircuitBreakerConfig(failure_threshold=10, timeout_seconds=120)
)
```

### Environment Variable Overrides

Override circuit breaker settings globally via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_CB_FAILURE_THRESHOLD` | (per-provider) | Failures before opening |
| `ARAGORA_CB_SUCCESS_THRESHOLD` | (per-provider) | Successes to close in half-open |
| `ARAGORA_CB_TIMEOUT_SECONDS` | (per-provider) | Cooldown duration |
| `ARAGORA_CB_HALF_OPEN_MAX_CALLS` | (per-provider) | Max calls in half-open state |

---

## Retry Strategies

Retry logic is built into the agent error decorators (`aragora/agents/errors/decorators.py`).

### Exponential Backoff with Jitter

The `calculate_retry_delay_with_jitter` function computes retry delays:

```python
from aragora.agents.errors.decorators import calculate_retry_delay_with_jitter

# Attempt 0: ~1.0s (± 30% jitter)
# Attempt 1: ~2.0s (± 30% jitter)
# Attempt 2: ~4.0s (± 30% jitter)
# Attempt 3: capped at max_delay

delay = calculate_retry_delay_with_jitter(
    attempt=2,          # 0-indexed attempt number
    base_delay=1.0,     # Initial delay in seconds
    max_delay=30.0,     # Maximum delay cap
    jitter_factor=0.3,  # ±30% randomization
)
```

Jitter prevents thundering herd problems when multiple clients recover simultaneously.

### Agent Error Decorator

The `@handle_agent_errors` decorator provides retry + circuit breaker integration:

```python
from aragora.agents.errors.decorators import handle_agent_errors

class MyAPIAgent:
    @handle_agent_errors(
        max_retries=3,
        retry_delay=1.0,
        retry_backoff=2.0,
        max_delay=30.0,
        retryable_exceptions=(AgentConnectionError, AgentTimeoutError, AgentRateLimitError),
        circuit_breaker_attr="_circuit_breaker",
    )
    async def generate(self, prompt: str) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(self.url, json={"prompt": prompt}) as resp:
                return await resp.text()
```

### Retryable vs Non-Retryable Errors

| Error Type | Retryable | Reason |
|------------|:---------:|--------|
| `AgentConnectionError` | Yes | Network issues are transient |
| `AgentTimeoutError` | Yes | Server may have been temporarily slow |
| `AgentRateLimitError` | Yes | Wait and retry after backoff |
| `AgentStreamError` | Yes | Streaming interruptions are transient |
| `AgentAPIError` (4xx) | No | Bad request, fix the input |
| `AgentAPIError` (5xx) | Yes | Server error, may recover |
| `AgentResponseError` | No | Response parsing failed, won't change on retry |
| `AgentCircuitOpenError` | Yes | Wait for cooldown, then retry |

### Rate Limit Handling with Retry-After

When a provider returns HTTP 429 with a `Retry-After` header, the system
respects the provider's requested wait time:

```python
# Handled automatically by _handle_response_error in decorators.py
# 1. Parse Retry-After header
# 2. Cap at max_delay
# 3. Add 10% jitter to prevent synchronized retries
# 4. Use as override delay instead of exponential backoff
```

---

## Fallback Chains

When a primary provider fails, Aragora can automatically route to alternative
providers. The implementation is in `aragora/agents/fallback.py`.

### QuotaFallbackMixin

The simplest fallback pattern routes to OpenRouter when the primary provider
hits rate limits or quota errors:

```python
from aragora.agents.fallback import QuotaFallbackMixin

class MyAgent(APIAgent, QuotaFallbackMixin):
    OPENROUTER_MODEL_MAP = {
        "gpt-4o": "openai/gpt-4o",
        "gpt-4": "openai/gpt-4",
    }
    DEFAULT_FALLBACK_MODEL = "openai/gpt-4o"

    async def generate(self, prompt, context=None):
        try:
            return await self._call_primary_api(prompt)
        except APIError as e:
            if self.is_quota_error(e.status_code, str(e)):
                result = await self.fallback_generate(prompt, context, e.status_code)
                if result is not None:
                    return result
            raise
```

### Quota Error Detection

The `is_quota_error` method detects these conditions:

| HTTP Status | Condition |
|:-----------:|-----------|
| 429 | Rate limit (all providers) |
| 403 | Quota exceeded (with keyword match) |
| 400 | Billing/credit exhaustion (with keyword match) |
| 408, 504, 524 | Timeout errors |

Keywords checked: `rate limit`, `quota`, `exceeded`, `billing`, `credit balance`,
`insufficient`, `timeout`, `timed out`.

### AgentFallbackChain

For multi-provider sequencing with full circuit breaker integration:

```python
from aragora.agents.fallback import AgentFallbackChain
from aragora.resilience.circuit_breaker import CircuitBreaker

chain = AgentFallbackChain(
    providers=["openai", "openrouter", "anthropic"],
    circuit_breaker=CircuitBreaker(failure_threshold=3, cooldown_seconds=60),
    max_retries=3,            # Try at most 3 providers
    max_fallback_time=30.0,   # Give up after 30 seconds total
)

# Register provider factories
chain.register_provider("openai", lambda: OpenAIAPIAgent(model="gpt-4o"))
chain.register_provider("openrouter", lambda: OpenRouterAgent(model="openai/gpt-4o"))
chain.register_provider("anthropic", lambda: AnthropicAPIAgent(model="claude-sonnet-4"))

# Generate with automatic fallback
result = await chain.generate(prompt, context)

# Monitor health
status = chain.get_status()
print(f"Fallback rate: {status['metrics']['fallback_rate']}")
print(f"Available: {status['available_providers']}")
```

### Including Local LLMs in Fallback

```python
from aragora.agents.fallback import build_fallback_chain_with_local

# Default: OpenAI -> OpenRouter -> Ollama/LM Studio -> Anthropic
providers = build_fallback_chain_with_local(
    primary_providers=["openai", "openrouter", "anthropic"],
    include_local=True,
)

# Priority local: OpenAI -> Ollama/LM Studio -> OpenRouter -> Anthropic
providers = build_fallback_chain_with_local(
    primary_providers=["openai", "openrouter", "anthropic"],
    include_local=True,
    local_priority=True,
)
```

### Fallback Error Types

| Error | Raised When |
|-------|-------------|
| `AllProvidersExhaustedError` | Every provider in the chain failed |
| `FallbackTimeoutError` | `max_fallback_time` exceeded before a provider succeeded |

### Enabling Fallback

Fallback is opt-in by default to prevent unexpected billing:

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_OPENROUTER_FALLBACK_ENABLED` | `false` | Enable OpenRouter fallback |
| `OPENROUTER_API_KEY` | (none) | Required for OpenRouter fallback |

---

## Agent Error Handling

The agent error hierarchy (`aragora/agents/errors/exceptions.py`) adds
agent-specific context to errors:

```python
from aragora.agents.errors.exceptions import (
    AgentError,
    AgentTimeoutError,
    AgentRateLimitError,
    AgentCircuitOpenError,
)

try:
    result = await agent.generate(prompt)
except AgentCircuitOpenError as e:
    # Circuit breaker is protecting this agent
    logger.warning(f"Agent {e.agent_name} circuit open, cooldown: {e.cooldown_seconds}s")
except AgentRateLimitError as e:
    # Provider rate limit hit
    if e.retry_after:
        await asyncio.sleep(e.retry_after)
except AgentTimeoutError as e:
    # Agent took too long
    if e.partial_content:
        # Use partial response if available
        result = e.partial_content
except AgentError as e:
    # Any agent error
    if e.recoverable:
        # Safe to retry
        pass
    logger.error(f"Agent error: {e}", extra={"cause": e.cause})
```

---

## Server Middleware

The exception handler middleware provides three usage patterns.

### Decorator Style

```python
from aragora.server.middleware.exception_handler import (
    handle_exceptions,
    async_handle_exceptions,
)

# Sync handler
@handle_exceptions("leaderboard retrieval")
def get_leaderboard(self, query_params):
    return self.db.get_leaderboard()

# Async handler
@async_handle_exceptions("agent generation")
async def generate_response(self, prompt):
    return await self.agent.generate(prompt)
```

### Context Manager Style

```python
from aragora.server.middleware.exception_handler import (
    ExceptionHandler,
    async_exception_handler,
)

# Sync context manager
with ExceptionHandler("debate creation") as ctx:
    result = create_debate()
    ctx.success(result)

if ctx.error:
    return ctx.error_response  # Sanitized error dict

# Async context manager
async with async_exception_handler("agent generation") as ctx:
    result = await agent.generate(prompt)
    ctx.success(result)
```

### Error Response Format

All error responses follow a consistent structure with trace IDs for debugging:

```json
{
    "error": "Failed to create debate: invalid configuration",
    "status": 400,
    "trace_id": "a1b2c3d4",
    "error_type": "DebateConfigurationError",
    "context": "debate creation"
}
```

The `X-Trace-Id` header is also set in the HTTP response for correlation.

---

## Custom Error Types

When creating new error types, follow these conventions:

### For Domain Errors

Inherit from `AragoraError` and include structured details:

```python
from aragora.exceptions import AragoraError

class MyDomainError(AragoraError):
    """Raised when my domain operation fails."""

    def __init__(self, resource_id: str, reason: str):
        super().__init__(
            f"Operation failed for {resource_id}: {reason}",
            {"resource_id": resource_id, "reason": reason},
        )
        self.resource_id = resource_id
        self.reason = reason
```

### For Agent Errors

Inherit from `AgentError` and set the `recoverable` flag:

```python
from aragora.agents.errors.exceptions import AgentError

class MyAgentError(AgentError):
    def __init__(self, message: str, agent_name: str, recoverable: bool = True):
        super().__init__(message, agent_name=agent_name, recoverable=recoverable)
```

### Register HTTP Status Mapping

Add new exceptions to the exception handler middleware:

```python
# In aragora/server/middleware/exception_handler.py
EXCEPTION_STATUS_MAP["MyDomainError"] = 422  # Unprocessable Entity
```

---

## Best Practices

1. **Use specific exception types.** Catch `AgentTimeoutError` rather than bare `Exception`.
   This enables proper retry logic and HTTP status mapping.

2. **Always include context.** Pass `details` dicts to `AragoraError` for structured logging.

3. **Respect the `recoverable` flag.** Agent errors with `recoverable=True` are safe to retry.
   Errors with `recoverable=False` indicate permanent failures (bad input, auth issues).

4. **Use circuit breakers for external calls.** Any call to an AI provider, database, or
   external service should be wrapped with a circuit breaker.

5. **Configure per-provider thresholds.** Use `get_circuit_breaker_config(provider="...")` to
   get tuned defaults rather than hardcoding values.

6. **Enable fallback chains for production.** Set `ARAGORA_OPENROUTER_FALLBACK_ENABLED=true`
   and provide `OPENROUTER_API_KEY` to prevent single-provider outages from blocking debates.

7. **Use the middleware decorators in handlers.** Wrap all HTTP handlers with
   `@handle_exceptions` or `@async_handle_exceptions` for consistent error responses.

8. **Log with trace IDs.** The `ExceptionHandler` context manager generates trace IDs
   automatically. Include them in all error logs for debugging.

9. **Never expose internal errors to clients.** The middleware uses `safe_error_message()`
   to sanitize error messages before sending them to clients.

10. **Test error paths.** Use `CircuitBreaker.from_dict()` and `CircuitBreaker.to_dict()`
    to simulate and verify circuit breaker state transitions in tests.
