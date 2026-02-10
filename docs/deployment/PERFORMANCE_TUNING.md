# Performance Tuning Guide

How to optimize Aragora debate performance through concurrency settings, memory tier
configuration, caching strategies, timeout management, and monitoring.

## Table of Contents

- [Debate Configuration Matrix](#debate-configuration-matrix)
- [Concurrency Tuning](#concurrency-tuning)
- [Timeout Configuration](#timeout-configuration)
- [Memory Tier Optimization](#memory-tier-optimization)
- [Caching Strategies](#caching-strategies)
- [Provider Rate Limits](#provider-rate-limits)
- [Cost Optimization](#cost-optimization)
- [Monitoring Performance](#monitoring-performance)
- [Configuration Quick Reference](#configuration-quick-reference)

---

## Debate Configuration Matrix

The `DebateProtocol` (`aragora/debate/protocol.py`) controls debate behavior.
Choose a configuration profile based on your requirements:

### Pre-Built Profiles

| Profile | Rounds | Consensus | Early Stop | Timeout | Best For |
|---------|:------:|-----------|:----------:|:-------:|----------|
| **Default** | 9 | judge | threshold=0.85 | 1200s | Full analysis |
| **Quick** | 3 | majority | disabled | 300s | Fast decisions |
| **High-Assurance** | 9 | supermajority (0.8) | threshold=0.95 | 1800s | Critical decisions |
| **Cost-Optimized** | 5 | judge | threshold=0.7 | 600s | Budget-conscious |
| **Light** | 5 | judge | threshold=0.85 | 600s | aragora.ai light mode |

### Configuration Examples

```python
from aragora.debate.protocol import DebateProtocol

# Default: Full 9-round structured debate
default = DebateProtocol()

# Quick: Fast decisions with minimal rounds
quick = DebateProtocol(
    rounds=3,
    consensus="majority",
    use_structured_phases=False,
    early_stopping=False,
    timeout_seconds=300,
)

# High-Assurance: Maximum rigor
high_assurance = DebateProtocol(
    consensus="supermajority",
    consensus_threshold=0.8,
    formal_verification_enabled=True,
    enable_trickster=True,
    trickster_sensitivity=0.7,
    early_stop_threshold=0.95,
    timeout_seconds=1800,
)

# Cost-Optimized: Minimize API calls
cost_optimized = DebateProtocol(
    rounds=5,
    early_stopping=True,
    early_stop_threshold=0.7,
    min_rounds_before_early_stop=2,
    convergence_detection=True,
    convergence_threshold=0.85,
    timeout_seconds=600,
)
```

### Topology Impact on Performance

The debate topology controls how agents communicate and directly affects
the number of API calls per round:

| Topology | API Calls per Round | Best For |
|----------|:-------------------:|----------|
| `all-to-all` | O(n^2) | Thorough analysis (default) |
| `ring` | O(n) | Efficient sequential review |
| `star` | O(n) | Hub-and-spoke decisions |
| `sparse` | O(n * sparsity) | Balanced throughput/quality |
| `round-robin` | O(n) | Sequential, deterministic |

```python
# Reduce API calls with sparse topology
protocol = DebateProtocol(
    topology="sparse",
    topology_sparsity=0.3,  # Only 30% of possible connections
)
```

### Consensus Mechanism Performance

| Mechanism | Rounds to Converge | API Overhead | Reliability |
|-----------|:------------------:|:------------:|:-----------:|
| `any` | 1 | Lowest | Low |
| `majority` | 2-4 | Low | Medium |
| `judge` | Full cycle | Medium | High |
| `supermajority` | 4-7 | Medium-High | Very High |
| `unanimous` | 5-9+ | High | Highest |
| `byzantine` | 3-5 | High | Fault-tolerant |

---

## Concurrency Tuning

Concurrency settings (`aragora/config/settings.py`, `ConcurrencySettings`) control
how many parallel API calls happen during each debate phase.

### Concurrency Parameters

| Setting | Env Variable | Default | Range | Description |
|---------|-------------|:-------:|:-----:|-------------|
| Proposals | `ARAGORA_MAX_CONCURRENT_PROPOSALS` | 5 | 1-50 | Parallel proposal generations |
| Critiques | `ARAGORA_MAX_CONCURRENT_CRITIQUES` | 10 | 1-100 | Parallel critique generations |
| Revisions | `ARAGORA_MAX_CONCURRENT_REVISIONS` | 5 | 1-50 | Parallel revision generations |
| Streaming | `ARAGORA_MAX_CONCURRENT_STREAMING` | 3 | 1-20 | Parallel streaming connections |
| Stagger | `ARAGORA_PROPOSAL_STAGGER_SECONDS` | 0.0 | 0-30 | Legacy delay between proposals |

### How Concurrency Works

Each debate phase uses `asyncio.Semaphore` to limit parallel API calls:

```python
# In aragora/debate/phases/critique_generator.py
critique_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CRITIQUES)

async def generate_critique(agent):
    async with critique_semaphore:
        return await agent.generate(prompt)

# All agents run concurrently, but at most MAX_CONCURRENT_CRITIQUES at a time
results = await asyncio.gather(*[generate_critique(a) for a in agents])
```

### Tuning Guidelines

**High-throughput (more API keys, generous rate limits):**
```bash
export ARAGORA_MAX_CONCURRENT_PROPOSALS=10
export ARAGORA_MAX_CONCURRENT_CRITIQUES=20
export ARAGORA_MAX_CONCURRENT_REVISIONS=10
```

**Conservative (single API key, strict rate limits):**
```bash
export ARAGORA_MAX_CONCURRENT_PROPOSALS=2
export ARAGORA_MAX_CONCURRENT_CRITIQUES=3
export ARAGORA_MAX_CONCURRENT_REVISIONS=2
```

**Single-provider mode (e.g., only Anthropic):**
```bash
export ARAGORA_MAX_CONCURRENT_PROPOSALS=3
export ARAGORA_MAX_CONCURRENT_CRITIQUES=5
export ARAGORA_MAX_CONCURRENT_REVISIONS=3
```

### Phase Duration Estimation

The debate engine calculates phase timeouts dynamically based on concurrency:

```python
# From aragora/debate/phases/debate_rounds.py
# Critique phase timeout calculation:
timeout = (num_agents / MAX_CONCURRENT_CRITIQUES) * AGENT_TIMEOUT_SECONDS + 60.0

# Example: 10 agents, 5 concurrent, 240s agent timeout
# = (10 / 5) * 240 + 60 = 540s
```

---

## Timeout Configuration

Timeouts prevent runaway operations. They are configured at multiple levels:

### Timeout Hierarchy

| Level | Setting | Default | Env Variable |
|-------|---------|:-------:|-------------|
| **Agent Call** | `agent_timeout_seconds` | 240s | `ARAGORA_AGENT_TIMEOUT` |
| **Per Round** | `round_timeout_seconds` | max(90s, agent_timeout + 60) | (protocol) |
| **Debate Rounds Phase** | `debate_rounds_timeout_seconds` | 420s (7 min) | (protocol) |
| **Full Debate** | `timeout_seconds` | max(1200s, DEBATE_TIMEOUT) | `ARAGORA_DEBATE_TIMEOUT` |
| **Heartbeat** | `heartbeat_interval_seconds` | 15s | `ARAGORA_HEARTBEAT_INTERVAL` |

### Configuring Timeouts

```python
protocol = DebateProtocol(
    timeout_seconds=1800,                   # 30 minutes for entire debate
    round_timeout_seconds=180,              # 3 minutes per round
    debate_rounds_timeout_seconds=600,      # 10 minutes for all debate rounds
)
```

### Timeout Best Practices

- **Agent timeout** should be at least 120s for complex prompts (LLMs can take time).
- **Round timeout** must exceed agent timeout to allow all parallel agents to complete.
- **Debate timeout** should be at least `rounds * round_timeout + 180` (for voting/synthesis).
- Set `ARAGORA_HEARTBEAT_INTERVAL` lower (5-10s) for real-time UI updates.

---

## Memory Tier Optimization

The Continuum Memory system (`aragora/memory/tier_manager.py`) uses four tiers with
different update frequencies and retention characteristics.

### Tier Configuration

| Tier | Half-Life | Update Frequency | Learning Rate | Decay Rate | Use Case |
|------|:---------:|:----------------:|:-------------:|:----------:|----------|
| **FAST** | 1 hour | Every event | 0.30 | 0.95 | Immediate context, active debate |
| **MEDIUM** | 24 hours | Per debate round | 0.10 | 0.99 | Session memory, recent debates |
| **SLOW** | 7 days | Per nomic cycle | 0.03 | 0.999 | Cross-session learning, evidence |
| **GLACIAL** | 30 days | Monthly | 0.01 | 0.9999 | Organizational culture, long-term |

### Tier Transitions

Memory entries automatically promote or demote based on surprise scores:

| Transition | Condition | Threshold |
|------------|-----------|:---------:|
| GLACIAL -> SLOW | Surprise score above threshold | 0.5 |
| SLOW -> MEDIUM | Surprise score above threshold | 0.6 |
| MEDIUM -> FAST | Surprise score above threshold | 0.7 |
| FAST -> MEDIUM | Stability score above threshold | 0.2 |
| MEDIUM -> SLOW | Stability score above threshold | 0.3 |
| SLOW -> GLACIAL | Stability score above threshold | 0.4 |

### Optimizing Tier Behavior

For **high-frequency debate environments** (many debates per day):
- Lower FAST tier promotion threshold to capture more transient insights
- Increase MEDIUM tier half-life for longer session awareness

For **low-frequency environments** (weekly debates):
- Higher promotion thresholds to reduce noise
- Shorter FAST tier half-life to keep memory focused

```python
from aragora.memory.tier_manager import TierConfig, TierManager, MemoryTier

# Custom tier configuration
custom_configs = {
    MemoryTier.FAST: TierConfig(
        name="fast",
        half_life_hours=2,          # Extended from 1 hour
        update_frequency="event",
        base_learning_rate=0.25,    # Slightly lower learning rate
        decay_rate=0.95,
        promotion_threshold=1.0,
        demotion_threshold=0.3,     # Slower demotion
    ),
    # ... configure other tiers similarly
}

manager = TierManager(configs=custom_configs)
```

---

## Caching Strategies

Aragora provides multiple caching layers to reduce latency and API costs.

### Request-Scoped Query Cache

The `RequestScopedCache` (`aragora/knowledge/mound/query_cache.py`) prevents
repeated identical queries within a single request:

```python
from aragora.knowledge.mound.query_cache import request_cache_context, get_or_compute

with request_cache_context():
    # First call: executes the lambda
    node = get_or_compute("node:123", lambda: db.get_node("123"))

    # Second call: returns cached result
    node_again = get_or_compute("node:123", lambda: db.get_node("123"))
    assert node is node_again  # Same object, no DB hit
```

| Setting | Default | Description |
|---------|---------|-------------|
| `ARAGORA_QUERY_CACHE_ENABLED` | `true` | Enable request-scoped cache |
| `ARAGORA_QUERY_CACHE_MAX_SIZE` | `1000` | Max entries per request |

### Redis Cache (Knowledge Mound)

The `RedisCache` (`aragora/knowledge/mound/redis_cache.py`) provides distributed
caching for knowledge queries:

```python
from aragora.knowledge.mound.redis_cache import RedisCache

cache = RedisCache(
    url="redis://localhost:6379",
    default_ttl=300,       # 5 minutes for query results
    culture_ttl=3600,      # 1 hour for culture patterns
    max_entries=10_000,    # LRU eviction after 10K entries
    prefix="aragora:km",   # Redis key prefix
)
await cache.connect()
```

Cache key structure:
- `aragora:km:{workspace}:node:{node_id}` - Individual knowledge nodes
- `aragora:km:{workspace}:query:{hash}` - Query result sets
- `aragora:km:{workspace}:culture` - Organization culture profiles
- `aragora:km:staleness:pending` - Sorted set of stale nodes
- `aragora:km:_entry_tracker` - LRU tracking sorted set

### Caching Recommendations

| Scenario | Strategy | TTL |
|----------|----------|:---:|
| Active debate | Request-scoped cache only | Request lifetime |
| Knowledge queries | Redis + request cache | 5 min (query), 1 hr (culture) |
| ELO ratings | Redis cache | 10 min |
| Agent availability | In-memory (circuit breaker state) | Real-time |
| Staleness checks | Redis sorted set | 30 min |

---

## Provider Rate Limits

Configure per-provider rate limits in `aragora/config/settings.py`
(`ProviderRateLimitSettings`):

| Provider | Env Variable | Default RPM |
|----------|-------------|:-----------:|
| Anthropic | `ARAGORA_PROVIDER_ANTHROPIC_RPM` | 1000 |
| OpenAI | `ARAGORA_PROVIDER_OPENAI_RPM` | 500 |
| Mistral | `ARAGORA_PROVIDER_MISTRAL_RPM` | 300 |
| Gemini | `ARAGORA_PROVIDER_GEMINI_RPM` | 60 |
| Grok | `ARAGORA_PROVIDER_GROK_RPM` | 500 |
| DeepSeek | `ARAGORA_PROVIDER_DEEPSEEK_RPM` | 200 |
| OpenRouter | `ARAGORA_PROVIDER_OPENROUTER_RPM` | 500 |

### Balancing Concurrency and Rate Limits

The total requests per minute for a debate is approximately:

```
Total RPM = rounds * agents * phases_per_round * (1 / avg_response_time_minutes)
```

For a 9-round debate with 5 agents (proposal + critique + revision per round):
```
Total calls = 9 * 5 * 3 = 135 API calls
At 240s per call average: ~135 / (240/60) = ~34 RPM peak
```

If using a single provider with a 60 RPM limit:
```bash
# Reduce concurrency to stay under rate limit
export ARAGORA_MAX_CONCURRENT_PROPOSALS=2
export ARAGORA_MAX_CONCURRENT_CRITIQUES=3
export ARAGORA_MAX_CONCURRENT_REVISIONS=2
```

---

## Cost Optimization

### Early Stopping

Early stopping can save 40-70% in API costs by ending debates when agents agree:

```python
protocol = DebateProtocol(
    early_stopping=True,
    early_stop_threshold=0.7,              # 70% of agents agree to stop
    min_rounds_before_early_stop=2,        # Minimum 2 rounds first
    convergence_detection=True,            # Auto-detect agreement
    convergence_threshold=0.85,            # Semantic similarity threshold
)
```

### Reducing Agent Count

For simpler decisions, fewer agents reduce costs proportionally:

```python
# 3 agents instead of 5+
arena = Arena(
    env=environment,
    agents=agents[:3],       # Fewer agents = fewer API calls
    protocol=protocol,
)
```

### Topology Selection

Use cheaper topologies for non-critical debates:

```python
# Ring topology: n critiques instead of n^2
protocol = DebateProtocol(topology="ring")

# Sparse topology: fraction of all-to-all
protocol = DebateProtocol(topology="sparse", topology_sparsity=0.3)
```

### Agent Agreement Intensity

Higher agreement intensity leads to faster convergence, fewer rounds:

```python
# agreement_intensity=7 -> agents tend to agree more -> faster convergence
protocol = DebateProtocol(agreement_intensity=7)

# agreement_intensity=2 -> more disagreement -> better exploration but more rounds
protocol = DebateProtocol(agreement_intensity=2)
```

---

## Monitoring Performance

### Agent Performance Monitor

The `AgentPerformanceMonitor` (`aragora/agents/performance_monitor.py`) tracks
per-agent metrics:

```python
from aragora.agents.performance_monitor import AgentPerformanceMonitor, AgentMetric

monitor = AgentPerformanceMonitor()

# Record a metric
metric = AgentMetric(
    agent_name="claude-sonnet",
    operation="generate",
    start_time=start,
    end_time=end,
    duration_ms=duration,
    success=True,
    response_length=len(response),
    phase="proposal",
    round_num=3,
)
monitor.record(metric)

# Get aggregated stats
stats = monitor.get_stats("claude-sonnet")
print(f"Success rate: {stats.success_rate:.1%}")
print(f"Avg latency: {stats.avg_duration_ms:.0f}ms")
print(f"Timeouts: {stats.timeout_calls}")
```

### Fallback Chain Metrics

Monitor fallback behavior via `FallbackMetrics`:

```python
chain = AgentFallbackChain(providers=[...])

# After running
status = chain.get_status()
metrics = status["metrics"]
print(f"Primary attempts: {metrics['primary_attempts']}")
print(f"Fallback rate: {metrics['fallback_rate']}")
print(f"Overall success: {metrics['success_rate']}")
print(f"Providers used: {metrics['providers_used']}")
```

### Circuit Breaker Status

```python
from aragora.resilience.circuit_breaker import CircuitBreaker

breaker = CircuitBreaker(name="agents")

# Check all entity statuses
all_status = breaker.get_all_status()
for entity, info in all_status.items():
    print(f"{entity}: {info['status']} (failures: {info['failures']})")

# Serialize for persistence/monitoring
state = breaker.to_dict()
```

### Key Metrics to Watch

| Metric | Healthy Range | Action if Exceeded |
|--------|:------------:|-------------------|
| Agent success rate | > 95% | Check API keys, rate limits |
| Agent avg latency | < 30s | Reduce prompt size, check provider |
| Fallback rate | < 10% | Review primary provider health |
| Circuit breaker open count | 0 | Investigate provider outages |
| Debate duration | < 15 min | Reduce rounds or enable early stopping |
| Cache hit rate | > 60% | Increase cache TTL or size |
| Memory tier promotions/day | < 100 | Adjust promotion thresholds |

---

## Configuration Quick Reference

### Environment Variables

```bash
# Concurrency
export ARAGORA_MAX_CONCURRENT_PROPOSALS=5
export ARAGORA_MAX_CONCURRENT_CRITIQUES=10
export ARAGORA_MAX_CONCURRENT_REVISIONS=5
export ARAGORA_MAX_CONCURRENT_STREAMING=3

# Timeouts
export ARAGORA_AGENT_TIMEOUT=240
export ARAGORA_DEBATE_TIMEOUT=1200
export ARAGORA_HEARTBEAT_INTERVAL=15

# Circuit Breaker (global overrides)
export ARAGORA_CB_FAILURE_THRESHOLD=5
export ARAGORA_CB_TIMEOUT_SECONDS=60
export ARAGORA_CB_SUCCESS_THRESHOLD=2

# Fallback
export ARAGORA_OPENROUTER_FALLBACK_ENABLED=true
export OPENROUTER_API_KEY=sk-or-...

# Provider Rate Limits (RPM)
export ARAGORA_PROVIDER_ANTHROPIC_RPM=1000
export ARAGORA_PROVIDER_OPENAI_RPM=500
export ARAGORA_PROVIDER_GEMINI_RPM=60

# Caching
export ARAGORA_QUERY_CACHE_ENABLED=true
export ARAGORA_QUERY_CACHE_MAX_SIZE=1000
```

### DebateProtocol Quick Settings

```python
from aragora.debate.protocol import DebateProtocol

# Fastest possible debate
fastest = DebateProtocol(
    rounds=1,
    consensus="any",
    use_structured_phases=False,
    early_stopping=False,
    enable_calibration=False,
    enable_rhetorical_observer=False,
    enable_trickster=False,
    enable_evolution=False,
    timeout_seconds=120,
)

# Most thorough debate
thorough = DebateProtocol(
    rounds=9,
    consensus="supermajority",
    consensus_threshold=0.8,
    use_structured_phases=True,
    early_stopping=True,
    early_stop_threshold=0.95,
    formal_verification_enabled=True,
    enable_calibration=True,
    enable_rhetorical_observer=True,
    enable_trickster=True,
    trickster_sensitivity=0.5,
    enable_evolution=True,
    convergence_detection=True,
    convergence_threshold=0.9,
    timeout_seconds=2400,
)
```
