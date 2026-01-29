# Performance Tuning Guide

This guide covers practical configuration for optimizing Aragora's performance across
debate orchestration, memory, API rate limiting, knowledge queries, and server deployment.

All settings are configured via environment variables with the `ARAGORA_` prefix and
managed through `aragora/config/settings.py`.

---

## 1. Debate Optimization

### Concurrency Settings

Debate phases (propose, critique, revise) execute agent calls in parallel, controlled
by semaphore-based concurrency limits. Higher values increase throughput but risk
hitting provider rate limits.

| Setting | Env Var | Default | Range | Description |
|---------|---------|---------|-------|-------------|
| Proposals | `ARAGORA_MAX_CONCURRENT_PROPOSALS` | 5 | 1--50 | Parallel proposal generations per round |
| Critiques | `ARAGORA_MAX_CONCURRENT_CRITIQUES` | 10 | 1--100 | Parallel critique generations per round |
| Revisions | `ARAGORA_MAX_CONCURRENT_REVISIONS` | 5 | 1--50 | Parallel revision generations per round |
| Streaming | `ARAGORA_MAX_CONCURRENT_STREAMING` | 3 | 1--20 | Parallel streaming connections |
| Agent Timeout | `ARAGORA_AGENT_TIMEOUT` | 240s | 30--1200 | Per-agent call timeout |
| Heartbeat | `ARAGORA_HEARTBEAT_INTERVAL` | 15s | 5--120 | Heartbeat interval for long operations |

**Recommendations:**

- **Low-latency (3--5 agents):** Set proposals=3, critiques=5, revisions=3. Keeps round
  times under 15 seconds at the P90 SLO.
- **High-throughput (8+ agents):** Use defaults or increase critiques to 15. Critiques
  are lighter-weight than proposals, so a higher limit is safe.
- **Rate-limit constrained:** Reduce all values to 2--3. Enable
  `ARAGORA_PROPOSAL_STAGGER_SECONDS=1.0` for legacy stagger mode if semaphore-based
  concurrency still triggers 429 errors.

### Round Count Tuning

| Setting | Env Var | Default | Range |
|---------|---------|---------|-------|
| Default rounds | `ARAGORA_DEFAULT_ROUNDS` | 9 | 1--20 |
| Max rounds | `ARAGORA_MAX_ROUNDS` | 12 | 1--50 |
| Debate timeout | `ARAGORA_DEBATE_TIMEOUT` | 600s | 30--7200 |

Round count is the primary latency lever. Each round executes a full propose-critique-revise
cycle. The debate round SLO targets P50=5s, P90=15s, P99=30s per round, with a 120-second
hard timeout per round.

**Recommendations:**

- **Quick decisions:** 3--5 rounds. Suitable for well-defined, low-ambiguity tasks.
- **Balanced (default):** 7--9 rounds. Good trade-off between quality and speed.
- **High-stakes decisions:** 10--12 rounds. Use for complex or contentious topics
  where thorough deliberation matters.

### Agent Selection

| Setting | Env Var | Default | Range |
|---------|---------|---------|-------|
| Max agents per debate | `ARAGORA_MAX_AGENTS_PER_DEBATE` | 10 | 2--50 |
| Max concurrent debates | `ARAGORA_MAX_CONCURRENT_DEBATES` | 10 | 1--100 |
| Default agents | `ARAGORA_DEFAULT_AGENTS` | grok,anthropic-api,openai-api,deepseek,mistral,gemini,qwen,kimi | -- |

Fewer agents means faster rounds but less diverse perspectives. The total API cost scales
linearly with agent count multiplied by round count.

**Recommendations:**

- **Fast mode (3 agents):** Pick the top 3 by ELO for the domain. Set
  `ARAGORA_DEFAULT_AGENTS=anthropic-api,openai-api,grok`.
- **Balanced (5--6 agents):** Good diversity without excessive latency. The default
  set minus 2--3 niche agents.
- **Maximum diversity (8+ agents):** Use all defaults. Best for exploratory or
  high-stakes debates.

### Early Stopping and Convergence

| Setting | Env Var | Default | Range |
|---------|---------|---------|-------|
| Similarity threshold | `ARAGORA_CONSENSUS_SIMILARITY` | 0.85 | 0.5--1.0 |
| Min vote ratio | `ARAGORA_CONSENSUS_MIN_VOTES` | 0.6 | 0.5--1.0 |
| Early exit threshold | `ARAGORA_CONSENSUS_EARLY_EXIT` | 0.95 | 0.8--1.0 |
| Supermajority threshold | `ARAGORA_CONSENSUS_SUPERMAJORITY` | 0.67 | 0.5--1.0 |

Convergence detection uses a 3-tier similarity backend (SentenceTransformer > TF-IDF > Jaccard).
Override with `ARAGORA_CONVERGENCE_BACKEND=tfidf` or `jaccard` for lighter-weight detection.

**Recommendations:**

- **Aggressive early stopping:** Lower `CONSENSUS_SIMILARITY` to 0.75 and
  `CONSENSUS_EARLY_EXIT` to 0.90. Saves 2--4 rounds on average.
- **Thorough deliberation:** Raise `CONSENSUS_SIMILARITY` to 0.92. Agents must
  genuinely agree before the debate ends early.
- **Consensus mode tuning:** Use `ARAGORA_DEFAULT_CONSENSUS=majority` for speed,
  `judge` for quality, `unanimous` for maximum rigor.

---

## 2. Memory Tier Configuration

ContinuumMemory uses four tiers with exponentially increasing half-lives. Each tier
has a maximum item count and a half-life that controls decay.

| Tier | Env Var (Max Items) | Default Max | Env Var (Half-Life) | Default Half-Life | Purpose |
|------|---------------------|-------------|---------------------|-------------------|---------|
| Fast | `ARAGORA_MEMORY_FAST_MAX` | 1,000 | `ARAGORA_MEMORY_FAST_TTL` | 1h | Immediate context, current debate |
| Medium | `ARAGORA_MEMORY_MEDIUM_MAX` | 5,000 | `ARAGORA_MEMORY_MEDIUM_TTL` | 24h | Session memory, tactical patterns |
| Slow | `ARAGORA_MEMORY_SLOW_MAX` | 10,000 | `ARAGORA_MEMORY_SLOW_TTL` | 168h (7d) | Cross-session learning |
| Glacial | `ARAGORA_MEMORY_GLACIAL_MAX` | 50,000 | `ARAGORA_MEMORY_GLACIAL_TTL` | 720h (30d) | Foundational knowledge |

Additional consolidation settings:

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| Promotion cooldown | `ARAGORA_MEMORY_PROMOTION_COOLDOWN` | 1h | Minimum time between tier promotions |
| Consolidation threshold | `ARAGORA_MEMORY_CONSOLIDATION_THRESHOLD` | 3.0 | Access count to trigger promotion |
| Retention multiplier | `ARAGORA_MEMORY_RETENTION_MULTIPLIER` | 2.0 | Multiplier for high-value item retention |

**Recommendations:**

- **Memory-constrained environments:** Reduce Fast to 500, Medium to 2,000, Slow to 5,000.
  Lower max items reduces memory footprint proportionally.
- **Long-running sessions:** Increase `MEMORY_FAST_TTL` to 4h and `MEMORY_MEDIUM_TTL` to
  72h so context persists across extended work sessions.
- **High-volume deployments:** Increase Glacial max to 100,000 for richer institutional
  memory. Increase consolidation threshold to 5.0 to be more selective about promotions.

Memory operation SLOs for reference:

| Operation | P50 | P90 | P99 | Timeout |
|-----------|-----|-----|-----|---------|
| Store | 20ms | 80ms | 300ms | 2s |
| Recall | 30ms | 100ms | 400ms | 3s |

---

## 3. API Rate Limiting

### Per-Provider Rate Limits

Each provider has configurable requests-per-minute (RPM) and tokens-per-minute (TPM) limits.

| Provider | RPM Env Var | Default RPM | TPM Env Var | Default TPM |
|----------|-------------|-------------|-------------|-------------|
| Anthropic | `ARAGORA_PROVIDER_ANTHROPIC_RPM` | 1,000 | `ARAGORA_PROVIDER_ANTHROPIC_TPM` | 100,000 |
| OpenAI | `ARAGORA_PROVIDER_OPENAI_RPM` | 500 | `ARAGORA_PROVIDER_OPENAI_TPM` | 90,000 |
| Grok | `ARAGORA_PROVIDER_GROK_RPM` | 500 | `ARAGORA_PROVIDER_GROK_TPM` | 100,000 |
| Mistral | `ARAGORA_PROVIDER_MISTRAL_RPM` | 300 | `ARAGORA_PROVIDER_MISTRAL_TPM` | 50,000 |
| DeepSeek | `ARAGORA_PROVIDER_DEEPSEEK_RPM` | 200 | -- | -- |
| Gemini | `ARAGORA_PROVIDER_GEMINI_RPM` | 60 | `ARAGORA_PROVIDER_GEMINI_TPM` | 30,000 |
| OpenRouter | `ARAGORA_PROVIDER_OPENROUTER_RPM` | 500 | -- | -- |

### OpenRouter Fallback

When a primary provider returns a 429 (rate limit) or quota error, Aragora can
automatically retry via OpenRouter. This is opt-in to prevent unexpected billing.

| Setting | Env Var | Default |
|---------|---------|---------|
| Enable fallback | `ARAGORA_OPENROUTER_FALLBACK_ENABLED` | false |
| Local LLM fallback | `ARAGORA_LOCAL_FALLBACK_ENABLED` | false |
| Local LLM priority | `ARAGORA_LOCAL_FALLBACK_PRIORITY` | false |

Set `ARAGORA_OPENROUTER_FALLBACK_ENABLED=true` and ensure `OPENROUTER_API_KEY` is set.
For air-gapped environments, enable local fallback with Ollama or LM Studio instead.

### Circuit Breaker Configuration

Circuit breakers prevent cascading failures when providers go down. The global defaults
are defined in `aragora/resilience_config.py`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `failure_threshold` | 5 | Consecutive failures before opening the circuit |
| `success_threshold` | 2 | Successes in half-open state before closing |
| `timeout_seconds` | 60.0 | Time circuit stays open before transitioning to half-open |
| `half_open_max_calls` | 3 | Max concurrent calls allowed in half-open state |

Global availability SLO: 99.9% uptime, max 5 consecutive failures, 30s recovery timeout.

**Recommendations:**

- **Unreliable providers:** Lower `failure_threshold` to 2--3 and increase
  `timeout_seconds` to 120s. This trips the breaker faster and waits longer
  before retrying.
- **Stable providers:** Raise `failure_threshold` to 8--10 and lower
  `timeout_seconds` to 30s. Tolerates occasional blips without tripping.
- **Registry pruning:** The circuit breaker registry auto-prunes after 1,000 entries.
  Stale breakers (not accessed in 24 hours) are removed automatically.

### HTTP Rate Limiting

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| Default limit | `ARAGORA_RATE_LIMIT` | 60 req/min | Per-user rate limit |
| IP rate limit | `ARAGORA_IP_RATE_LIMIT` | 120 req/min | Per-IP rate limit |
| Burst multiplier | `ARAGORA_BURST_MULTIPLIER` | 2.0x | Burst allowance above base limit |
| Redis URL | `ARAGORA_REDIS_URL` | -- | Redis for distributed rate limiting |
| Redis TTL | `ARAGORA_REDIS_TTL` | 120s | Rate limit window TTL in Redis |

---

## 4. Knowledge Mound Query Optimization

### Search and Ingestion SLOs

| Operation | P50 | P90 | P99 | Timeout |
|-----------|-----|-----|-----|---------|
| Query | 50ms | 150ms | 500ms | 5s |
| Ingestion | 100ms | 300ms | 1s | 10s |
| Checkpoint | 500ms | 2s | 5s | 30s |
| Semantic search | 100ms | 300ms | 1s | 5s |

### Cache Settings

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| Embeddings cache | `ARAGORA_CACHE_EMBEDDINGS` | 3,600s | Cache TTL for embedding vectors |
| Consensus cache | `ARAGORA_CACHE_CONSENSUS` | 240s | Debate consensus results |
| Analytics cache | `ARAGORA_CACHE_ANALYTICS` | 600s | Analytics aggregations |
| Agent profile cache | `ARAGORA_CACHE_AGENT_PROFILE` | 600s | Agent metadata |
| Query default cache | `ARAGORA_CACHE_QUERY` | 60s | Generic query cache |
| Method default cache | `ARAGORA_CACHE_METHOD` | 300s | Generic method cache |

**Recommendations:**

- **Read-heavy workloads:** Increase `CACHE_EMBEDDINGS` to 7,200s and `CACHE_CONSENSUS`
  to 600s. Reduces redundant embedding computations.
- **Write-heavy workloads:** Lower `CACHE_CONSENSUS` to 120s so fresh debate results
  propagate faster. Keep embeddings cache high since embeddings rarely change.
- **Memory-constrained:** Reduce all cache TTLs by 50%. Entries expire sooner, freeing memory.

### Adapter Sync Tuning

Knowledge Mound adapters sync data between subsystems and the mound. Each direction
has its own SLO.

| Operation | P50 | P90 | P99 | Timeout |
|-----------|-----|-----|-----|---------|
| Full sync | 300ms | 800ms | 2s | 15s |
| Forward sync (source to KM) | 100ms | 300ms | 800ms | 5s |
| Reverse query (KM to source) | 50ms | 150ms | 500ms | 3s |
| Validation feedback | 200ms | 500ms | 1.5s | 10s |

The ingestion threshold (`ARAGORA_INTEGRATION_KNOWLEDGE_THRESHOLD`, default 0.85) controls
which debate outcomes are ingested into the mound. Lower it to 0.7 for more aggressive
knowledge capture; raise to 0.95 for higher-confidence-only ingestion.

---

## 5. Server Performance

### Worker and Connection Configuration

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| Max content length | `ARAGORA_MAX_CONTENT_LENGTH` | 100MB | Maximum request body size |
| Max question length | `ARAGORA_MAX_QUESTION_LENGTH` | 10,000 chars | Maximum debate question length |
| Default pagination | `ARAGORA_DEFAULT_PAGINATION` | 20 | Default page size for list endpoints |
| Max API limit | `ARAGORA_MAX_API_LIMIT` | 100 | Maximum items per API response |

### WebSocket Configuration

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| Max message size | `ARAGORA_WS_MAX_MESSAGE_SIZE` | 64KB | Maximum WebSocket message size |
| Heartbeat interval | `ARAGORA_WS_HEARTBEAT` | 30s | WebSocket heartbeat interval |
| User event queue | `ARAGORA_USER_EVENT_QUEUE_SIZE` | 10,000 | Max queued events per user |

WebSocket SLOs:

| Operation | P50 | P90 | P99 | Timeout |
|-----------|-----|-----|-----|---------|
| Connection | 50ms | 150ms | 500ms | 5s |
| Message delivery | 10ms | 30ms | 100ms | 1s |
| Broadcast | 50ms | 150ms | 500ms | 5s |

**Recommendations:**

- **High concurrency (100+ WebSocket clients):** Increase `USER_EVENT_QUEUE_SIZE` to
  50,000 to prevent dropped events during peak debate activity. Monitor broadcast P99.
- **Low-bandwidth clients:** Reduce `WS_MAX_MESSAGE_SIZE` to 16KB and increase
  `WS_HEARTBEAT` to 60s to reduce overhead.

### Database Connection Pool

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| Pool size | `ARAGORA_DB_POOL_SIZE` | 20 | Base connection pool size |
| Pool overflow | `ARAGORA_DB_POOL_OVERFLOW` | 15 | Extra connections above pool size |
| Pool timeout | `ARAGORA_DB_POOL_TIMEOUT` | 30s | Wait time for a connection from pool |
| Command timeout | `ARAGORA_DB_COMMAND_TIMEOUT` | 60s | Max time for a single query |
| Statement timeout | `ARAGORA_DB_STATEMENT_TIMEOUT` | 60s | PostgreSQL query cancellation |
| Pool recycle | `ARAGORA_DB_POOL_RECYCLE` | 1,800s | Idle connection recycle interval |

The pool should accommodate `max_concurrent_debates * 2` (read + write). With the
default 10 concurrent debates, the default pool of 20 + 15 overflow = 35 total
connections is appropriate.

**Recommendations:**

- **High-concurrency (20+ concurrent debates):** Set `DB_POOL_SIZE=40` and
  `DB_POOL_OVERFLOW=20`. Ensure your PostgreSQL instance supports 60+ connections.
- **Supabase deployments:** Use `SUPABASE_POOL_SIZE=10` and `SUPABASE_POOL_OVERFLOW=5`
  (lower limits due to Supabase connection pooling).

### Redis for Distributed Deployments

Set `ARAGORA_REDIS_URL` to enable distributed rate limiting and caching across
multiple server instances. Configure `ARAGORA_REDIS_KEY_PREFIX` (default:
`aragora:ratelimit:`) to namespace keys if sharing a Redis instance.

### Streaming Configuration

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| Stream buffer | `ARAGORA_STREAM_BUFFER_SIZE` | 10MB | Max buffer for streaming responses |
| Chunk timeout | `ARAGORA_STREAM_CHUNK_TIMEOUT` | 180s | Timeout between stream chunks |
| Max context chars | `ARAGORA_MAX_CONTEXT_CHARS` | 100,000 | Max characters for context/history |
| Max message chars | `ARAGORA_MAX_MESSAGE_CHARS` | 50,000 | Max characters per message |

The chunk timeout must be less than the agent timeout (240s default) to prevent
zombie streams. If you increase agent timeout, increase chunk timeout proportionally.

---

## 6. Performance SLO Reference

All SLOs are defined in `aragora/config/performance_slos.py`. Use the `check_latency_slo`
function to validate measurements programmatically:

```python
from aragora.config.performance_slos import check_latency_slo

within, message = check_latency_slo("km_query", latency_ms=120.0, percentile="p90")
# within=True, message="km_query latency 120.0ms within p90 SLO (150ms)"
```

### Global Availability Target

| Metric | Value |
|--------|-------|
| Uptime target | 99.9% |
| Max consecutive failures | 5 |
| Recovery timeout | 30s |

### Bot Platform SLOs

Slack requires webhook acknowledgment within 3 seconds. The bot webhook SLO reflects this:

| Operation | P50 | P90 | P99 | Timeout |
|-----------|-----|-----|-----|---------|
| Bot response | 500ms | 1.5s | 3s | 30s |
| Webhook ack | 100ms | 500ms | 2.5s | 3s |

### API Endpoint SLO

| Percentile | Target |
|------------|--------|
| P50 | 100ms |
| P90 | 500ms |
| P99 | 2,000ms |
| Timeout | 30s |

---

## 7. Quick-Start Profiles

### Low-Latency Profile

Optimize for fastest possible debate resolution.

```bash
export ARAGORA_DEFAULT_ROUNDS=3
export ARAGORA_MAX_AGENTS_PER_DEBATE=3
export ARAGORA_DEFAULT_AGENTS=anthropic-api,openai-api,grok
export ARAGORA_MAX_CONCURRENT_PROPOSALS=3
export ARAGORA_MAX_CONCURRENT_CRITIQUES=5
export ARAGORA_CONSENSUS_SIMILARITY=0.75
export ARAGORA_CONSENSUS_EARLY_EXIT=0.90
export ARAGORA_DEFAULT_CONSENSUS=majority
```

### High-Quality Profile

Optimize for thorough, well-deliberated outcomes.

```bash
export ARAGORA_DEFAULT_ROUNDS=12
export ARAGORA_MAX_AGENTS_PER_DEBATE=8
export ARAGORA_MAX_CONCURRENT_PROPOSALS=5
export ARAGORA_MAX_CONCURRENT_CRITIQUES=10
export ARAGORA_CONSENSUS_SIMILARITY=0.92
export ARAGORA_CONSENSUS_EARLY_EXIT=0.98
export ARAGORA_DEFAULT_CONSENSUS=judge
```

### Cost-Optimized Profile

Minimize API spend while maintaining acceptable quality.

```bash
export ARAGORA_DEFAULT_ROUNDS=5
export ARAGORA_MAX_AGENTS_PER_DEBATE=4
export ARAGORA_DEFAULT_AGENTS=deepseek,mistral,qwen,kimi
export ARAGORA_MAX_CONCURRENT_PROPOSALS=2
export ARAGORA_MAX_CONCURRENT_CRITIQUES=4
export ARAGORA_OPENROUTER_FALLBACK_ENABLED=false
export ARAGORA_CONSENSUS_SIMILARITY=0.80
```
