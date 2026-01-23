---
slug: 012-agent-fallback-strategy
title: "ADR-012: Agent Fallback Strategy"
description: "ADR-012: Agent Fallback Strategy"
---

# ADR-012: Agent Fallback Strategy

## Status
Accepted

## Context
Multi-agent debates require reliable agent responses, but:
- API providers have rate limits (429 errors)
- Providers experience outages
- Some agents may be temporarily unavailable
- Cost varies significantly across providers

We needed a strategy that:
- Maximizes debate completion rates
- Handles transient failures gracefully
- Respects rate limits automatically
- Optimizes for cost when possible

## Decision
We implemented a **three-layer fallback strategy**:

### Layer 1: Circuit Breaker
Located in `aragora/resilience.py`:

```python
class CircuitBreaker:
    states: CLOSED -> OPEN -> HALF_OPEN
    failure_threshold: 5 consecutive failures
    recovery_timeout: 60 seconds
```

- Tracks failures per agent/provider
- Opens circuit to prevent cascading failures
- Half-open state tests recovery

### Layer 2: OpenRouter Fallback
Located in `aragora/agents/fallback.py`:

On 429 (rate limit) errors:
1. Detect quota exhaustion
2. Automatically route to OpenRouter
3. Use equivalent model mapping:
   - Claude -> claude-3-opus (via OpenRouter)
   - GPT-4 -> gpt-4-turbo (via OpenRouter)

### Layer 3: Airlock Proxy
Located in `aragora/agents/airlock.py`:

- Wraps agents with retry logic
- Implements exponential backoff
- Isolates failures from debate flow
- Configurable via `ArenaConfig.use_airlock`

### Provider Mapping
```python
OPENROUTER_FALLBACKS = {
    "anthropic": "anthropic/claude-3-opus",
    "openai": "openai/gpt-4-turbo",
    "google": "google/gemini-pro",
}
```

## Consequences
**Positive:**
- High debate completion rates (>99%)
- Automatic handling of transient failures
- Cost optimization via routing
- No manual intervention required

**Negative:**
- OpenRouter dependency for fallback
- Slightly higher latency on fallback
- Cost may increase when using fallbacks
- Model equivalence is approximate

## References
- `aragora/resilience.py` - CircuitBreaker
- `aragora/agents/fallback.py` - OpenRouter fallback
- `aragora/agents/airlock.py` - AirlockProxy
- Environment: `OPENROUTER_API_KEY`
