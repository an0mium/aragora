# Knowledge Mound

**The "Termite Mound" Enterprise Knowledge Architecture**

The Knowledge Mound implements a shared knowledge superstructure where all agents contribute to and query from a unified knowledge base. Like a termite mound, it's an emergent structure built from thousands of small contributions that together form a robust, well-organized whole.

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Storage Backends](#storage-backends)
4. [Adapters](#adapters)
5. [Semantic Search](#semantic-search)
6. [Staleness Detection and Revalidation](#staleness-detection-and-revalidation)
7. [Confidence Decay](#confidence-decay)
8. [Resilience](#resilience)
9. [Federation](#federation)
10. [Configuration](#configuration)
11. [Examples](#examples)

---

## Overview

The Knowledge Mound is the central knowledge repository for the Aragora multi-agent control plane. It provides:

- **Unified API** over multiple storage backends (SQLite, PostgreSQL, Redis)
- **Cross-system queries** across ContinuumMemory, ConsensusMemory, FactStore, and more
- **Provenance tracking** for audit and compliance
- **Staleness detection** with automatic revalidation scheduling
- **Culture accumulation** for organizational learning
- **Multi-tenant workspace isolation**
- **Semantic search** with mandatory embeddings
- **Federation** for distributed knowledge synchronization

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Knowledge Item** | A unit of knowledge with content, confidence, source, and metadata |
| **Workspace** | Tenant isolation boundary for multi-org deployments |
| **Adapter** | Bridge that syncs data between subsystems and the Knowledge Mound |
| **Staleness** | Score indicating how likely knowledge needs revalidation |
| **Confidence Decay** | Time-based reduction of knowledge confidence |
| **Federation** | Cross-region synchronization of knowledge |

---

## Architecture

```
                                    +-------------------+
                                    |   Applications    |
                                    +--------+----------+
                                             |
                                    +--------v----------+
                                    |  KnowledgeMound   |
                                    |     (Facade)      |
                                    +--------+----------+
                                             |
              +------------------------------+------------------------------+
              |                              |                              |
    +---------v---------+        +-----------v-----------+        +--------v--------+
    |   Query Layer     |        |    Operations Layer   |        |  Storage Layer  |
    +-------------------+        +-----------------------+        +-----------------+
    | - Semantic Search |        | - Staleness Detection |        | - PostgreSQL    |
    | - Graph Traversal |        | - Confidence Decay    |        | - SQLite        |
    | - Federated Query |        | - Contradiction Det.  |        | - Redis Cache   |
    | - RLM Integration |        | - Knowledge Extraction|        | - Weaviate      |
    +-------------------+        | - Culture Accumulator |        +-----------------+
                                 | - Dedup & Pruning     |
                                 | - Auto-Curation       |
                                 +-----------------------+
                                             |
              +------------------------------+------------------------------+
              |              |               |               |              |
    +---------v----+ +------v------+ +------v------+ +------v------+ +-----v------+
    |  Continuum   | |  Consensus  | |  Evidence   | |   Belief    | |   Pulse    |
    |   Adapter    | |   Adapter   | |   Adapter   | |   Adapter   | |  Adapter   |
    +--------------+ +-------------+ +-------------+ +-------------+ +------------+
           |               |               |               |              |
    +------v------+ +------v------+ +------v------+ +------v------+ +-----v------+
    | Continuum   | | Consensus   | |  Evidence   | |   Belief    | |   Pulse    |
    |   Memory    | |   Memory    | |    Store    | |   Network   | |  Service   |
    +-------------+ +-------------+ +-------------+ +-------------+ +------------+
```

### Multi-Tier Storage

```
+------------------------------------------------------------------+
|                        Knowledge Mound                            |
|                                                                   |
|  +--------------------+  +--------------------+  +--------------+ |
|  |    Fast Tier       |  |   Medium Tier      |  |  Slow Tier   | |
|  |   (TTL: 1 min)     |  |   (TTL: 1 hour)    |  | (TTL: 7 days)| |
|  | - Immediate ctx    |  | - Session memory   |  | - Cross-sess | |
|  | - Hot knowledge    |  | - Recent debates   |  | - Validated  | |
|  +--------------------+  +--------------------+  +--------------+ |
|                                                                   |
|  +--------------------+  +--------------------+  +--------------+ |
|  |   Glacial Tier     |  |   Semantic Index   |  |    Cache     | |
|  |  (TTL: 30 days)    |  | (Embeddings Store) |  |   (Redis)    | |
|  | - Long-term learn  |  | - Vector search    |  | - Query cache| |
|  | - Culture patterns |  | - Deduplication    |  | - Hot data   | |
|  +--------------------+  +--------------------+  +--------------+ |
+------------------------------------------------------------------+
```

### Component Breakdown

| Component | File | Purpose |
|-----------|------|---------|
| Facade | `facade.py` | Main entry point, composes all mixins |
| Core | `core.py` | Initialization, storage adapters, lifecycle |
| Types | `types.py` | Data classes (KnowledgeItem, MoundConfig, etc.) |
| Semantic Store | `semantic_store.py` | Embedding-based semantic search |
| Graph Store | `graph_store.py` | Relationship traversal and lineage |
| PostgreSQL Store | `postgres_store.py` | Production-grade persistence |
| Redis Cache | `redis_cache.py` | Query caching layer |

---

## Storage Backends

The Knowledge Mound supports three backend configurations:

### SQLite (Development)

Default backend for local development and testing. Zero configuration required.

```python
from aragora.knowledge.mound import KnowledgeMound, MoundConfig, MoundBackend

config = MoundConfig(
    backend=MoundBackend.SQLITE,
    sqlite_path="/path/to/mound.db",  # Optional, defaults to DB_KNOWLEDGE_PATH
)
mound = KnowledgeMound(config)
```

### PostgreSQL (Production)

Production-grade backend with connection pooling, full-text search, and async operations.

```python
config = MoundConfig(
    backend=MoundBackend.POSTGRES,
    postgres_url="postgresql://user:pass@localhost:5432/aragora",
    postgres_pool_size=10,
    postgres_pool_max_overflow=5,
)
mound = KnowledgeMound(config)
```

**Features:**
- Connection pooling via asyncpg
- Full-text search with `ts_rank`
- JSON/JSONB metadata storage
- Cursor-based pagination for large result sets

### Hybrid (Production + Caching)

PostgreSQL for persistence plus Redis for query caching.

```python
config = MoundConfig(
    backend=MoundBackend.HYBRID,
    postgres_url="postgresql://user:pass@localhost:5432/aragora",
    redis_url="redis://localhost:6379",
    redis_cache_ttl=300,  # 5 minutes for query results
    redis_culture_ttl=3600,  # 1 hour for culture patterns
)
mound = KnowledgeMound(config)
```

**Benefits:**
- Reduced database load for repeated queries
- Sub-millisecond cache hits
- Automatic cache invalidation via event bus

---

## Adapters

Adapters bridge existing memory systems to the Knowledge Mound, enabling bidirectional data flow.

### Adapter Architecture

```
+------------------+     Forward Sync (IN)      +------------------+
|                  | --------------------------> |                  |
|  Source System   |                             |  Knowledge Mound |
|  (e.g., ELO)     | <-------------------------- |                  |
+------------------+     Reverse Sync (OUT)      +------------------+
```

### Core Memory Adapters

| Adapter | Source | What It Syncs |
|---------|--------|---------------|
| **ContinuumAdapter** | ContinuumMemory | Multi-tier memories (fast/medium/slow/glacial) |
| **ConsensusAdapter** | ConsensusMemory | Debate outcomes and agreements |
| **CritiqueAdapter** | CritiqueStore | Critique patterns and feedback |

### Bidirectional Adapters

| Adapter | Source | What It Syncs |
|---------|--------|---------------|
| **EvidenceAdapter** | EvidenceStore | Evidence snippets with quality scores |
| **BeliefAdapter** | BeliefNetwork | Belief nodes, cruxes, and claim provenance |
| **InsightsAdapter** | InsightsEngine | Debate insights and Trickster flip detections |
| **EloAdapter** | ELORankings | Agent ratings and calibration data |
| **PulseAdapter** | PulseService | Trending topics and scheduled debates |
| **CostAdapter** | CostTracker | Budget alerts and cost patterns (opt-in) |
| **RlmAdapter** | RLM | Compression patterns and context priorities |
| **CultureAdapter** | CultureAccumulator | Organizational patterns and norms |

### Control Plane Adapters

| Adapter | Source | What It Syncs |
|---------|--------|---------------|
| **ControlPlaneAdapter** | ControlPlane | Task outcomes, agent capabilities |
| **ReceiptAdapter** | GauntletReceipts | Decision audit trails |
| **FabricAdapter** | AgentFabric | Pool snapshots, scheduling outcomes |
| **WorkspaceAdapter** | WorkspaceManager | Rig snapshots, convoy outcomes |
| **GatewayAdapter** | LocalGateway | Message routing, channel performance |
| **ComputerUseAdapter** | CU Orchestrator | Task execution, policy blocks |

### External Adapters

| Adapter | Purpose |
|---------|---------|
| **ExtractionAdapter** | Knowledge graph extraction from debates |
| **ERC8004Adapter** | Blockchain knowledge verification |
| **SupermemoryAdapter** | External memory persistence |
| **OpenClawAdapter** | OpenClaw action pattern learning |

### Using Adapters

```python
from aragora.knowledge.mound.adapters import EvidenceAdapter

# Create adapter with event callback for real-time updates
adapter = EvidenceAdapter(
    evidence_store=my_evidence_store,
    knowledge_mound=mound,
    enable_dual_write=True,  # Write to both systems during migration
    event_callback=my_websocket_callback,
)

# Forward sync: Evidence -> Knowledge Mound
result = await adapter.sync_to_km()

# Reverse sync: Knowledge Mound validations -> Evidence scores
await adapter.apply_validations_to_source()
```

---

## Semantic Search

The Knowledge Mound requires semantic embeddings for all stored knowledge, enabling intelligent similarity-based retrieval.

### How It Works

```
1. Store Request
   +-------------+      +-----------------+      +------------------+
   | Content     | ---> | Embedding       | ---> | SemanticStore    |
   | "Contract   |      | Provider        |      | (SQLite index)   |
   |  notice..." |      | (OpenAI/Gemini/ |      | - content_hash   |
   +-------------+      |  Ollama)        |      | - embedding blob |
                        +-----------------+      | - metadata       |
                                                 +------------------+

2. Search Request
   +-------------+      +-----------------+      +------------------+
   | Query       | ---> | Embed Query     | ---> | Cosine Similarity|
   | "contract   |      |                 |      | Search           |
   |  terms"     |      +-----------------+      +------------------+
   +-------------+                                       |
                                                         v
                                               +------------------+
                                               | Ranked Results   |
                                               | (by similarity)  |
                                               +------------------+
```

### Embedding Providers

The semantic store auto-detects the best available provider:

1. **OpenAI** (if `OPENAI_API_KEY` set) - text-embedding-3-small
2. **Gemini** (if `GEMINI_API_KEY` set) - text-embedding-004
3. **Ollama** (if running locally) - nomic-embed-text
4. **Fallback** - Hash-based embeddings (limited semantic capability)

### Deduplication

Content deduplication happens automatically via SHA-256 content hashing:

```python
# First store
result1 = await mound.store(IngestionRequest(
    content="Contracts require 90-day notice",
    ...
))  # Creates new node

# Duplicate store
result2 = await mound.store(IngestionRequest(
    content="Contracts require 90-day notice",
    ...
))  # Returns existing node ID, increments retrieval count
```

---

## Staleness Detection and Revalidation

Knowledge becomes stale over time. The staleness system identifies outdated knowledge and schedules revalidation.

### Staleness Score Computation

The staleness score (0.0 = fresh, 1.0 = stale) is computed from four factors:

| Factor | Weight | Description |
|--------|--------|-------------|
| **Age** | 40% | Time since last update relative to tier threshold |
| **Contradictions** | 30% | Number of contradicting items added since validation |
| **New Evidence** | 20% | Relevant evidence added that may affect validity |
| **Consensus Change** | 10% | Changes in related debate outcomes |

### Tier-Based Age Thresholds

| Tier | Default Threshold | Use Case |
|------|-------------------|----------|
| Fast | 1 hour | Immediate context, session data |
| Medium | 1 day | Recent conversations, short-term memory |
| Slow | 7 days | Validated facts, cross-session knowledge |
| Glacial | 30 days | Long-term patterns, organizational learning |

### Revalidation Workflow

```
1. Detection
   +------------------+     staleness > 0.8      +------------------+
   | StalenessDetector| -----------------------→ | Emit Event       |
   | (periodic check) |                          | KNOWLEDGE_STALE  |
   +------------------+                          +------------------+
                                                         |
2. Scheduling                                            v
   +------------------+     queue task           +------------------+
   | Revalidation     | <----------------------- | Event Handler    |
   | Scheduler        |                          +------------------+
   +------------------+
           |
3. Execution
           v
   +------------------+     validate             +------------------+
   | Debate/Agent     | -----------------------→ | Update Knowledge |
   | Revalidation     |                          | confidence/status|
   +------------------+                          +------------------+
```

### Usage

```python
# Get stale knowledge
stale_items = await mound.get_stale_knowledge(
    threshold=0.7,  # Items with staleness >= 0.7
    workspace_id="my_team",
    limit=50,
)

# Mark as validated after revalidation
await mound.mark_validated(node_id, validated_by="debate_456")

# Schedule for revalidation
await mound.schedule_revalidation(node_id, priority="high")
```

---

## Confidence Decay

Confidence decay automatically reduces trust in aging knowledge that hasn't been revalidated or accessed.

### Decay Models

| Model | Formula | Use Case |
|-------|---------|----------|
| **Exponential** | `C(t) = C0 * e^(-lambda * t)` | Default, natural decay |
| **Linear** | `C(t) = C0 - (rate * t)` | Predictable, uniform decay |
| **Step** | `C(t) = C0 - (step * floor(t/interval))` | Periodic review cycles |

### Decay Configuration

```python
from aragora.knowledge.mound.ops.confidence_decay import DecayConfig, DecayModel

config = DecayConfig(
    model=DecayModel.EXPONENTIAL,
    decay_rate=0.1,  # 10% decay per period
    min_confidence=0.1,  # Floor value
    decay_period_days=30,  # How often decay is applied
)
```

### Scheduler

The `ConfidenceDecayScheduler` runs as a background task:

```python
from aragora.knowledge.mound import start_decay_scheduler, stop_decay_scheduler

# Start scheduler (typically at server startup)
scheduler = await start_decay_scheduler(
    knowledge_mound=mound,
    decay_interval_hours=24,  # Run daily
    workspaces=["team_a", "team_b"],  # Or None for all
)

# Get scheduler stats
stats = scheduler.get_stats()

# Trigger immediate decay
reports = await scheduler.trigger_decay_now(workspace_id="team_a")

# Stop scheduler (at shutdown)
await stop_decay_scheduler()
```

### Confidence Boosts

Recent access or validation can boost confidence:

```python
# Record positive event (access, citation, validation)
await mound.record_confidence_event(
    node_id=node_id,
    event_type="validation",
    boost=0.1,  # +10% confidence
)
```

---

## Resilience

The resilience module provides production-hardening capabilities for reliable operation.

### Components

```
+------------------------------------------------------------------+
|                     Resilience Layer                              |
|                                                                   |
|  +------------------+  +------------------+  +------------------+ |
|  |   Retry Logic    |  | Circuit Breaker  |  |    Bulkhead      | |
|  | - Exponential    |  | - Failure thresh |  | - Concurrency    | |
|  | - Linear         |  | - Recovery time  |  |   limits         | |
|  | - Jitter         |  | - Health-aware   |  | - Isolation      | |
|  +------------------+  +------------------+  +------------------+ |
|                                                                   |
|  +------------------+  +------------------+  +------------------+ |
|  | Health Monitor   |  | Cache Invalid.   |  |  Transactions    | |
|  | - Liveness probe |  | - Event bus      |  | - Isolation lvls | |
|  | - Latency track  |  | - Pub/Sub        |  | - Timeout        | |
|  +------------------+  +------------------+  +------------------+ |
+------------------------------------------------------------------+
```

### Retry with Exponential Backoff

```python
from aragora.knowledge.mound.resilience import with_retry, RetryConfig, RetryStrategy

@with_retry(RetryConfig(
    max_retries=3,
    base_delay=0.1,
    max_delay=10.0,
    strategy=RetryStrategy.EXPONENTIAL,
    jitter=True,  # Prevent thundering herd
    retryable_exceptions=(ConnectionError, TimeoutError),
))
async def save_knowledge(data):
    await store.save(data)
```

### Circuit Breaker

Prevents cascade failures when a dependency is unhealthy:

```python
from aragora.knowledge.mound.resilience import (
    AdapterCircuitBreaker,
    AdapterCircuitBreakerConfig,
)

breaker = AdapterCircuitBreaker(AdapterCircuitBreakerConfig(
    failure_threshold=5,  # Open after 5 failures
    recovery_timeout=30.0,  # Try again after 30s
    half_open_max_calls=3,  # Test calls when half-open
))

# Circuit states: CLOSED -> OPEN -> HALF_OPEN -> CLOSED
```

### Health Monitoring

```python
from aragora.knowledge.mound.resilience import ConnectionHealthMonitor

monitor = ConnectionHealthMonitor(
    pool=db_pool,
    failure_threshold=5,
    health_check_interval=10.0,
)

await monitor.start()
status = await monitor.check_health()
# HealthStatus(healthy=True, latency_ms=2.3, consecutive_failures=0)
```

### Cache Invalidation Bus

Distributed cache invalidation via pub/sub:

```python
from aragora.knowledge.mound.resilience import (
    CacheInvalidationBus,
    CacheInvalidationEvent,
)

bus = CacheInvalidationBus(redis_client)

# Publish invalidation
await bus.publish(CacheInvalidationEvent(
    cache_key="km:query:abc123",
    reason="item_updated",
    item_id="km_abc123",
))

# Subscribe to invalidations
async for event in bus.subscribe():
    await local_cache.delete(event.cache_key)
```

### ResilientPostgresStore

The `ResilientPostgresStore` wraps the base store with all resilience features:

```python
from aragora.knowledge.mound.resilience import ResilientPostgresStore

store = ResilientPostgresStore(
    store=base_postgres_store,
    retry_config=RetryConfig(max_retries=3),
    transaction_config=TransactionConfig(timeout_seconds=30),
)

# All operations now have retry, circuit breaker, and health monitoring
await store.save_node_async(data)
```

---

## Federation

Federation enables knowledge synchronization across distributed Knowledge Mound instances.

### Federation Modes

| Mode | Description |
|------|-------------|
| **Hub-Spoke** | Central hub syncs with regional spokes |
| **Mesh** | Peer-to-peer sync between all regions |
| **Hierarchical** | Parent-child relationships between regions |

### Federated Query

The `FederatedQueryAggregator` queries multiple adapters in parallel:

```python
from aragora.knowledge.mound import FederatedQueryAggregator, QuerySource

aggregator = FederatedQueryAggregator(
    parallel=True,
    timeout_seconds=10.0,
    deduplicate=True,
)

# Register adapters
aggregator.register_adapter(QuerySource.EVIDENCE, evidence_adapter)
aggregator.register_adapter(QuerySource.BELIEF, belief_adapter)
aggregator.register_adapter(QuerySource.PULSE, pulse_adapter)

# Query across all sources
result = await aggregator.query(
    query="climate change impacts",
    sources=[QuerySource.EVIDENCE, QuerySource.BELIEF],
    limit=20,
    min_relevance=0.5,
)

# Result contains aggregated, deduplicated, ranked items
for item in result.results:
    print(f"{item.source}: {item.content[:100]}... (score: {item.relevance_score})")
```

### Cross-Region Sync

```python
# Register a federated region
await mound.register_federated_region(FederatedRegion(
    region_id="us-west",
    endpoint_url="https://km-us-west.example.com",
    capabilities=["full_sync", "incremental"],
))

# Push knowledge to region
await mound.sync_to_region(
    region_id="us-west",
    scope=SyncScope.WORKSPACE,
    workspace_id="shared_team",
)

# Pull knowledge from region
await mound.pull_from_region(
    region_id="us-west",
    since=last_sync_timestamp,
)
```

---

## Configuration

### MoundConfig Reference

```python
@dataclass
class MoundConfig:
    # Backend selection
    backend: MoundBackend = MoundBackend.SQLITE

    # PostgreSQL settings (production)
    postgres_url: str | None = None
    postgres_pool_size: int = 10
    postgres_pool_max_overflow: int = 5

    # Redis settings (caching layer)
    redis_url: str | None = None
    redis_cache_ttl: int = 300  # 5 minutes
    redis_culture_ttl: int = 3600  # 1 hour

    # SQLite settings (development)
    sqlite_path: str | None = None

    # Weaviate settings (vector search)
    weaviate_url: str | None = None
    weaviate_collection: str = "KnowledgeMound"

    # Feature flags
    enable_staleness_detection: bool = True
    enable_culture_accumulator: bool = True
    enable_auto_revalidation: bool = False
    enable_deduplication: bool = True
    enable_provenance_tracking: bool = True

    # Adapter flags
    enable_evidence_adapter: bool = True
    enable_pulse_adapter: bool = True
    enable_insights_adapter: bool = True
    enable_elo_adapter: bool = True
    enable_belief_adapter: bool = True
    enable_cost_adapter: bool = False  # Opt-in (sensitive)

    # Confidence thresholds
    evidence_min_reliability: float = 0.6
    pulse_min_quality: float = 0.6
    insight_min_confidence: float = 0.7

    # Query settings
    default_query_limit: int = 20
    max_query_limit: int = 100
    parallel_queries: bool = True

    # Staleness settings
    staleness_check_interval: timedelta = timedelta(hours=1)
    staleness_age_threshold: timedelta = timedelta(days=7)
    staleness_revalidation_threshold: float = 0.8

    # Resilience settings
    enable_resilience: bool = True
    enable_integrity_checks: bool = True
    enable_health_monitoring: bool = True
    enable_cache_invalidation_events: bool = True
    retry_max_attempts: int = 3
    retry_base_delay: float = 0.1
    transaction_timeout: float = 30.0
```

### Global Configuration

Set configuration before first use:

```python
from aragora.knowledge.mound import set_mound_config, get_knowledge_mound

# Set config once at startup
set_mound_config(MoundConfig(
    backend=MoundBackend.HYBRID,
    postgres_url=os.environ["DATABASE_URL"],
    redis_url=os.environ["REDIS_URL"],
))

# Get singleton instance anywhere
mound = get_knowledge_mound(workspace_id="my_team")
```

---

## Examples

### Basic Usage

```python
from aragora.knowledge.mound import (
    KnowledgeMound,
    MoundConfig,
    MoundBackend,
    IngestionRequest,
    KnowledgeSource,
)

# Create and initialize
config = MoundConfig(backend=MoundBackend.SQLITE)
mound = KnowledgeMound(config, workspace_id="my_team")
await mound.initialize()

# Store knowledge
result = await mound.store(IngestionRequest(
    content="Contracts require 90-day notice for termination",
    source_type=KnowledgeSource.DEBATE,
    debate_id="debate_123",
    confidence=0.95,
    workspace_id="my_team",
    topics=["contracts", "legal", "termination"],
))
print(f"Stored: {result.node_id}")

# Query semantically
results = await mound.query(
    query="contract notice requirements",
    limit=10,
)
for item in results.items:
    print(f"- {item.content[:80]}... (confidence: {item.confidence})")

# Get stale knowledge
stale = await mound.get_stale_knowledge(threshold=0.7)
print(f"Found {len(stale)} stale items needing revalidation")

# Close when done
await mound.close()
```

### Production Setup with Context Manager

```python
from aragora.knowledge.mound import KnowledgeMound, MoundConfig, MoundBackend

config = MoundConfig(
    backend=MoundBackend.HYBRID,
    postgres_url="postgresql://user:pass@localhost/aragora",
    redis_url="redis://localhost:6379",
    enable_resilience=True,
)

async with KnowledgeMound(config, workspace_id="prod").session() as mound:
    # Mound is initialized, use it
    stats = await mound.get_stats()
    print(f"Total nodes: {stats.total_nodes}")

    # Query with graph expansion
    result = await mound.query_graph(
        start_id="km_abc123",
        relationship_types=[RelationshipType.SUPPORTS, RelationshipType.DERIVED_FROM],
        depth=2,
    )
    print(f"Found {result.total_nodes} related nodes")
# Mound is automatically closed
```

### Adapter Integration

```python
from aragora.knowledge.mound import KnowledgeMound
from aragora.knowledge.mound.adapters import (
    AdapterFactory,
    EvidenceAdapter,
    BeliefAdapter,
)

# Auto-create adapters from Arena subsystems
factory = AdapterFactory(knowledge_mound=mound)
adapters = factory.create_from_arena(arena)

# Or create specific adapter
evidence_adapter = EvidenceAdapter(
    evidence_store=arena.evidence_store,
    knowledge_mound=mound,
    event_callback=lambda event_type, data: ws.broadcast(event_type, data),
)

# Sync evidence to Knowledge Mound
result = await evidence_adapter.sync_to_km()
print(f"Synced {result.nodes_synced} evidence items")

# Search via adapter
results = await evidence_adapter.search_by_topic(
    topic="climate policy",
    min_reliability=0.7,
    limit=20,
)
```

### Culture Profile

```python
# Get organizational culture profile
profile = await mound.get_culture_profile(workspace_id="my_team")

print(f"Decision style: {profile.dominant_traits.get('decision_style')}")
print(f"Risk tolerance: {profile.dominant_traits.get('risk_tolerance')}")
print(f"Total observations: {profile.total_observations}")

# Recommend agents based on culture
recommendations = await mound.recommend_agents(
    task_type="technical_analysis",
    workspace_id="my_team",
)
for agent, score in recommendations:
    print(f"- {agent}: {score:.2f}")
```

---

## Further Reading

- [CLAUDE.md](../../../CLAUDE.md) - Project overview and architecture
- [docs/STATUS.md](../../../docs/STATUS.md) - Feature implementation status
- [docs/API_REFERENCE.md](../../../docs/API_REFERENCE.md) - REST API documentation

---

## Test Coverage

The Knowledge Mound has 950+ tests covering:
- Core CRUD operations
- Semantic search and deduplication
- Staleness detection and decay
- All 14+ adapters
- Federation and synchronization
- Resilience patterns
- Multi-tenant isolation

Run tests:
```bash
pytest tests/knowledge/mound/ -v
```
