# Performance Targets and Scaling Guide

Reference document for capacity planning, hardware sizing, and tuning Aragora for production workloads.

## Hardware Profiles

### Minimum (Development / Evaluation)

| Resource | Specification |
|----------|---------------|
| CPU | 2 cores |
| RAM | 4 GB |
| Disk | 20 GB SSD |
| Network | 100 Mbps |
| Database | SQLite (default) |

Supports: 1-2 concurrent debates, 3-5 agents per debate, single-user evaluation.

### Recommended (Small Team / SMB)

| Resource | Specification |
|----------|---------------|
| CPU | 4 cores |
| RAM | 8 GB |
| Disk | 100 GB SSD |
| Network | 1 Gbps |
| Database | PostgreSQL 15+ |
| Cache | Redis 7+ (optional) |

Supports: 5-10 concurrent debates, up to 10 agents per debate, 10-50 concurrent users.

### Enterprise (Production)

| Resource | Specification |
|----------|---------------|
| CPU | 8+ cores |
| RAM | 16-32 GB |
| Disk | 500 GB+ NVMe SSD |
| Network | 10 Gbps |
| Database | PostgreSQL 15+ (HA replica pair) |
| Cache | Redis 7+ cluster (3 nodes) |
| Load Balancer | Nginx / HAProxy / cloud ALB |

Supports: 10-100 concurrent debates, 50+ agents, 100-1000+ concurrent users, WebSocket streaming.

## Throughput Expectations

### Debate Orchestration

| Metric | Min | Recommended | Enterprise |
|--------|-----|-------------|------------|
| Concurrent debates | 2 | 10 | 100 |
| Agents per debate | 5 | 10 | 50 |
| Debates completed/hour | 5-10 | 30-60 | 200+ |
| Rounds per debate | 3-9 | 9 (default) | 12 (max) |

Throughput is primarily bounded by upstream LLM API rate limits, not Aragora compute. Each debate round requires one API call per agent for proposals, critiques, and revisions.

### API Server

| Metric | Target |
|--------|--------|
| HTTP requests/sec (p50) | 500+ |
| HTTP requests/sec (p99) | 200+ |
| WebSocket connections | 1,000+ per node |
| Static file serving | 2,000+ req/sec |
| JSON payload limit | 10 MB (`MAX_JSON_CONTENT_LENGTH`) |
| Upload limit | 100 MB (`MAX_CONTENT_LENGTH`) |

### Database

| Backend | Read throughput | Write throughput | Connection pool |
|---------|----------------|------------------|-----------------|
| SQLite | 5,000 reads/sec | 100 writes/sec (WAL) | Single connection |
| PostgreSQL | 10,000+ reads/sec | 2,000+ writes/sec | 20 base + 15 overflow = 35 total |

## Latency Targets

### SLO Defaults

These are the built-in SLO targets from `aragora/observability/slo.py`:

| SLO | Target | Environment Variable |
|-----|--------|---------------------|
| Availability | 99.9% | `SLO_AVAILABILITY_TARGET` |
| p99 Latency | 500 ms | `SLO_LATENCY_P99_TARGET_MS` |
| Debate Success Rate | 95% | `SLO_DEBATE_SUCCESS_TARGET` |

### Component Latency Breakdown

| Operation | p50 | p95 | p99 |
|-----------|-----|-----|-----|
| API request routing | <5 ms | <15 ms | <50 ms |
| Database query (SQLite) | <10 ms | <50 ms | <100 ms |
| Database query (PostgreSQL) | <5 ms | <20 ms | <50 ms |
| WebSocket message delivery | <10 ms | <30 ms | <100 ms |
| Memory tier lookup (fast) | <5 ms | <10 ms | <25 ms |
| Memory tier lookup (slow/glacial) | <20 ms | <50 ms | <150 ms |
| Agent API call (LLM) | 2-10 s | 15-30 s | 60-240 s |
| Full debate (3 rounds, 5 agents) | 2-5 min | 8-12 min | 15 min |
| Consensus detection | <100 ms | <300 ms | <500 ms |

Agent API call latency dominates overall debate time. Aragora internal overhead is typically <1% of total debate duration.

## Memory System Profiles

### Continuum Memory Tiers

| Tier | TTL | Half-Life | Decay Rate | Purpose |
|------|-----|-----------|------------|---------|
| Fast | 1 minute | 1 hour | 24.0 | Immediate debate context |
| Medium | 1 hour | 24 hours | 1.0 | Session memory |
| Slow | 1 day | 7 days | 0.14 | Cross-session learning |
| Glacial | 1 week | 30 days | 0.03 | Long-term institutional patterns |

### Cache TTLs

| Cache | TTL | Max Size |
|-------|-----|----------|
| KM similarity | 5 min | 1,000 entries |
| Consensus queries | 5 min | 2,000 entries |
| Dissent queries | 5 min | 2,000 entries |
| Embedding cache | Configurable (`CACHE_TTL_EMBEDDINGS`) | 1,000 entries |

## Concurrency Tuning Knobs

All values are configurable via environment variables. Defaults are tuned for the Recommended hardware profile.

### Debate Phase Concurrency

| Setting | Default | Range | Environment Variable | Notes |
|---------|---------|-------|---------------------|-------|
| Max concurrent debates | 10 | 1-100 | `ARAGORA_MAX_CONCURRENT_DEBATES` | Per-server limit |
| Max concurrent proposals | 5 | 1-50 | `ARAGORA_MAX_CONCURRENT_PROPOSALS` | Per-debate semaphore |
| Max concurrent critiques | 10 | 1-100 | `ARAGORA_MAX_CONCURRENT_CRITIQUES` | Per-debate semaphore |
| Max concurrent revisions | 5 | 1-50 | `ARAGORA_MAX_CONCURRENT_REVISIONS` | Per-debate semaphore |
| Max concurrent streaming | 3 | 1-20 | `ARAGORA_MAX_CONCURRENT_STREAMING` | WebSocket stream chains |
| Max concurrent branches | 3 | - | `ARAGORA_MAX_CONCURRENT_BRANCHES` | Debate forking |
| Max agents per debate | 10 | 2-50 | `ARAGORA_MAX_AGENTS_PER_DEBATE` | >20 may cause performance issues |

### Timeouts

| Setting | Default | Range | Environment Variable |
|---------|---------|-------|---------------------|
| Debate timeout | 900 s (15 min) | 30-7200 s | `ARAGORA_DEBATE_TIMEOUT` |
| Agent call timeout | 240 s (4 min) | 30-1200 s | `ARAGORA_AGENT_TIMEOUT` |
| Heartbeat interval | 15 s | 5-120 s | `ARAGORA_HEARTBEAT_INTERVAL` |
| DB query timeout | 30 s | 1-300 s | `ARAGORA_DB_TIMEOUT` |
| DB command timeout | 60 s | 1-600 s | `ARAGORA_DB_COMMAND_TIMEOUT` |
| DB statement timeout | 60 s | 1-600 s | `ARAGORA_DB_STATEMENT_TIMEOUT` |

### Database Connection Pool

| Setting | Default | Range | Environment Variable |
|---------|---------|-------|---------------------|
| Pool size | 20 | 1-100 | `ARAGORA_DB_POOL_SIZE` |
| Pool max overflow | 15 | 0-100 | `ARAGORA_DB_POOL_OVERFLOW` |
| Pool timeout | 30 s | 1-300 s | `ARAGORA_DB_POOL_TIMEOUT` |
| Pool recycle | 1800 s (30 min) | 60-7200 s | `ARAGORA_DB_POOL_RECYCLE` |
| Supabase pool size | 10 | 1-50 | `SUPABASE_POOL_SIZE` |
| Supabase pool overflow | 5 | 0-20 | `SUPABASE_POOL_OVERFLOW` |

### Rate Limiting

| Setting | Default | Range | Environment Variable |
|---------|---------|-------|---------------------|
| API rate limit | 60 req/min | 1-10000 | `ARAGORA_RATE_LIMIT` |
| IP rate limit | 120 req/min | 1-10000 | `ARAGORA_IP_RATE_LIMIT` |
| Burst multiplier | 2.0x | 1.0-10.0 | `ARAGORA_BURST_MULTIPLIER` |
| Upload rate (per min) | 5 | - | `MAX_UPLOADS_PER_MINUTE` |
| Upload rate (per hour) | 30 | - | `MAX_UPLOADS_PER_HOUR` |

### Provider Rate Limits (requests/min)

| Provider | Default RPM | Environment Variable |
|----------|-------------|---------------------|
| Anthropic | 1,000 | `ARAGORA_PROVIDER_ANTHROPIC_RPM` |
| OpenAI | 500 | `ARAGORA_PROVIDER_OPENAI_RPM` |
| Grok (xAI) | 500 | `ARAGORA_PROVIDER_GROK_RPM` |
| OpenRouter | 500 | `ARAGORA_PROVIDER_OPENROUTER_RPM` |
| Mistral | 300 | `ARAGORA_PROVIDER_MISTRAL_RPM` |
| DeepSeek | 200 | `ARAGORA_PROVIDER_DEEPSEEK_RPM` |
| Gemini | 60 | `ARAGORA_PROVIDER_GEMINI_RPM` |

### Inter-Request Delays

| Setting | Default | Environment Variable |
|---------|---------|---------------------|
| General stagger | 1.5 s | `ARAGORA_INTER_REQUEST_DELAY` |
| OpenRouter stagger | 2.0 s | `ARAGORA_OPENROUTER_INTER_REQUEST_DELAY` |
| Proposal stagger (legacy) | 0.0 s (disabled) | `ARAGORA_PROPOSAL_STAGGER_SECONDS` |

### WebSocket

| Setting | Default | Range | Environment Variable |
|---------|---------|-------|---------------------|
| Max message size | 64 KB | 1 KB - 10 MB | `ARAGORA_WS_MAX_MESSAGE_SIZE` |
| Heartbeat interval | 30 s | 5-300 s | `ARAGORA_WS_HEARTBEAT` |
| Stream batch size | 10 tokens | - | `ARAGORA_STREAM_BATCH_SIZE` |
| Stream drain interval | 5 ms | - | `ARAGORA_STREAM_DRAIN_INTERVAL_MS` |
| Event queue size | 10,000 | 100-100,000 | `ARAGORA_USER_EVENT_QUEUE_SIZE` |

### Redis (when enabled)

| Setting | Default | Environment Variable |
|---------|---------|---------------------|
| Max connections | 50 | `ARAGORA_REDIS_MAX_CONNECTIONS` |
| Rate limit key TTL | 120 s | `ARAGORA_REDIS_TTL` |
| Key prefix | `aragora:ratelimit:` | `ARAGORA_REDIS_KEY_PREFIX` |

## Resilience Configuration

### Circuit Breaker

Default settings from `aragora/resilience/circuit_breaker.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| Failure threshold | 3 | Consecutive failures before circuit opens |
| Cooldown | 60 s | Time before half-open recovery attempt |
| Half-open success threshold | 2 | Successes needed to close circuit |
| Half-open max calls | 3 | Max concurrent trial calls in half-open |
| Registry max size | 1,000 | Max circuit breakers before pruning |
| Stale threshold | 24 hours | Idle time before circuit breaker pruning |

Configurable per-provider via `CircuitBreakerConfig`:
```python
from aragora.resilience_config import CircuitBreakerConfig
config = CircuitBreakerConfig(failure_threshold=10, timeout_seconds=120)
```

### Retry Policies

Default retry configuration from `aragora/resilience/retry.py`:

| Setting | Default |
|---------|---------|
| Max retries | 3 |
| Base delay | 0.1 s |
| Max delay | 30 s |
| Strategy | Exponential backoff |
| Jitter | Multiplicative (+/- 25%) |

Provider-specific overrides:

| Provider | Max Retries | Base Delay | Max Delay | Notes |
|----------|-------------|------------|-----------|-------|
| Anthropic | 3 | 2.0 s | 120 s | Conservative (strict rate limits) |
| OpenAI | 3 | 1.0 s | 60 s | Respects Retry-After headers |
| Others | 3 | 1.0 s | 60 s | Standard defaults |

## Scaling Characteristics

### Vertical Scaling

Aragora is async Python (asyncio), so it benefits from:
- **More CPU cores**: Parallel debate orchestration, concurrent agent calls
- **More RAM**: Larger memory tier caches, more concurrent WebSocket connections, bigger DB connection pools
- **Faster storage**: Lower latency for SQLite WAL, PostgreSQL queries, and Knowledge Mound operations

Recommended vertical scaling path:
1. Start with 4 cores / 8 GB (Recommended profile)
2. Scale to 8 cores / 16 GB when hitting 10+ concurrent debates
3. Scale to 16 cores / 32 GB for enterprise workloads with 50+ concurrent debates

### Horizontal Scaling

Aragora supports horizontal scaling with stateless HTTP/WebSocket servers behind a load balancer:

**Stateless components** (scale freely):
- HTTP API server (`aragora serve`)
- WebSocket streaming servers
- Worker nodes for background processing

**Stateful components** (scale with care):
- PostgreSQL (primary + read replicas)
- Redis (cluster mode for rate limiting and caching)
- Knowledge Mound storage

**Kubernetes HPA example:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: aragora-api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: aragora-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: aragora_active_debates
      target:
        type: AverageValue
        averageValue: "5"
```

**Scaling guidelines:**
- 1 pod per 5-10 concurrent debates (CPU-bound on API call management)
- WebSocket connections are sticky; use session affinity on the load balancer
- PostgreSQL connection pool should be sized: `pool_size * num_pods <= max_connections`
- Redis connections: `max_connections * num_pods <= Redis maxclients`

### Load Balancer Configuration

For WebSocket support, configure your load balancer for:
- Connection upgrade (HTTP -> WebSocket)
- Session affinity (IP hash or cookie-based)
- Health check: `GET /api/v1/health` (returns 200 when ready)
- Readiness check: `GET /readyz` (returns 503 until startup completes)
- Idle timeout: >= 300 s (to keep WebSocket connections alive through heartbeats)

## Monitoring

### Key Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `aragora_active_debates` | Gauge | Currently running debates |
| `aragora_debate_duration_seconds` | Histogram | Debate completion time |
| `aragora_agent_call_duration_seconds` | Histogram | LLM API call latency |
| `aragora_circuit_breaker_state` | Gauge | Circuit breaker state (0=closed, 1=open, 2=half-open) |
| `aragora_slo_compliance` | Gauge | SLO compliance (1.0 = compliant) |
| `aragora_slo_error_budget` | Gauge | Remaining error budget percentage |
| `aragora_slo_burn_rate` | Gauge | Error budget consumption rate |

### Alerting Thresholds

| Alert | Threshold | Severity |
|-------|-----------|----------|
| High debate latency | p99 > 500 ms | Warning |
| Low availability | < 99.9% over 5 min | Critical |
| Circuit breaker open | Any provider open > 5 min | Warning |
| DB pool exhaustion | Available connections < 5 | Critical |
| Memory usage | > 80% RSS | Warning |
| Debate failure rate | > 5% over 15 min | Critical |

## Tuning Recommendations by Profile

### Development
```bash
export ARAGORA_MAX_CONCURRENT_DEBATES=2
export ARAGORA_MAX_CONCURRENT_PROPOSALS=2
export ARAGORA_MAX_CONCURRENT_CRITIQUES=3
export ARAGORA_AGENT_TIMEOUT=120
export ARAGORA_RATE_LIMIT=30
```

### SMB Production
```bash
# Use defaults (already tuned for this profile)
export ARAGORA_DB_BACKEND=postgresql
export DATABASE_URL=postgresql://user:pass@localhost:5432/aragora
export ARAGORA_REDIS_URL=redis://localhost:6379/0
```

### Enterprise Production
```bash
export ARAGORA_MAX_CONCURRENT_DEBATES=50
export ARAGORA_MAX_CONCURRENT_PROPOSALS=10
export ARAGORA_MAX_CONCURRENT_CRITIQUES=20
export ARAGORA_MAX_CONCURRENT_REVISIONS=10
export ARAGORA_MAX_CONCURRENT_STREAMING=10
export ARAGORA_DB_POOL_SIZE=50
export ARAGORA_DB_POOL_OVERFLOW=30
export ARAGORA_REDIS_MAX_CONNECTIONS=100
export ARAGORA_RATE_LIMIT=500
export ARAGORA_IP_RATE_LIMIT=1000
export ARAGORA_BURST_MULTIPLIER=3.0
```
