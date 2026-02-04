# Knowledge Module

Unified knowledge management with the Knowledge Mound architecture.

## Overview

The knowledge module implements a "termite mound" architecture - an emergent shared knowledge superstructure built from agent contributions. It provides semantic search, graph relationships, staleness detection, and 32 adapters that bridge existing memory systems to a unified knowledge graph.

## Quick Start

```python
from aragora.knowledge.mound import KnowledgeMound, MoundConfig, IngestionRequest

# Initialize
config = MoundConfig(backend=MoundBackend.SQLITE)
mound = KnowledgeMound(config, workspace_id="team_a")
await mound.initialize()

# Store knowledge
result = await mound.store(IngestionRequest(
    content="Contracts require 90-day notice for termination",
    source_type=KnowledgeSource.DEBATE,
    debate_id="debate_123",
    confidence=0.95,
))

# Query semantically
results = await mound.query("contract notice requirements", limit=10)
for item in results.items:
    print(f"{item.content[:80]}... (confidence: {item.confidence})")

await mound.close()
```

## Key Files

| File | Purpose |
|------|---------|
| `bridges.py` | Integration bridges (MetaLearner, Evidence, Pattern) |
| `mound_core.py` | Core types (KnowledgeNode, ProvenanceChain) |
| `mound/facade.py` | Main entry point (composed from 17 mixins) |
| `mound/semantic_store.py` | Vector embedding-based semantic search |
| `mound/graph_store.py` | Relationship traversal and lineage |
| `mound/postgres_store.py` | Production PostgreSQL backend |
| `mound/redis_cache.py` | Query caching layer |
| `mound/adapters/` | 32 adapters for system integration |

## Architecture

Three-layer design:

| Layer | Purpose | Components |
|-------|---------|------------|
| **Facade** | Main API | KnowledgeMound class (17 mixins) |
| **Operations** | Business logic | CRUD, Query, Staleness, Culture, Sync |
| **Storage** | Persistence | PostgreSQL, SQLite, Redis, Weaviate |

### 17 Operational Mixins

1. **CRUDOperationsMixin** - Create, read, update, delete nodes
2. **QueryOperationsMixin** - Semantic search, filtering, pagination
3. **StalenessOperationsMixin** - Staleness detection and revalidation
4. **CultureOperationsMixin** - Organizational pattern accumulation
5. **SyncOperationsMixin** - Cross-system synchronization
6. **GlobalKnowledgeMixin** - Global knowledge sharing
7. **KnowledgeFederationMixin** - Cross-region knowledge sync
8. **ContradictionOperationsMixin** - Contradiction detection
9. **ConfidenceDecayMixin** - Time-based confidence reduction
10. **GovernanceMixin** - RBAC and audit trails
11. **AnalyticsMixin** - Coverage, usage, quality analytics
12. **ExtractionMixin** - Knowledge extraction from debates

## Adapters System (32 Adapters)

Adapters bridge existing memory systems to the Knowledge Mound:

**Core Memory Adapters:**
- `ContinuumAdapter` - Multi-tier memory (fast/medium/slow/glacial)
- `ConsensusAdapter` - Debate outcomes, agreements, dissent
- `CritiqueAdapter` - Critique patterns and feedback

**Integration Adapters:**
- `EvidenceAdapter` - Evidence with reliability scoring
- `BeliefAdapter` - Belief network nodes, cruxes, claims
- `InsightsAdapter` - Debate insights, Trickster flip detection
- `PerformanceAdapter` - Agent ratings, expertise, calibration
- `PulseAdapter` - Trending topics, scheduled debates
- `CostAdapter` - Budget alerts, cost patterns

**Control Plane Adapters:**
- `ControlPlaneAdapter` - Task outcomes, agent capabilities
- `ReceiptAdapter` - Gauntlet decision audit trails
- `WorkspaceAdapter` - Rig snapshots, convoy outcomes
- `GatewayAdapter` - Message routing, channel performance

### Using the Adapter Factory

```python
from aragora.knowledge.mound.adapters import AdapterFactory

factory = AdapterFactory()
adapters = factory.create_from_config(
    elo_system=arena.elo_system,
    continuum_memory=arena.continuum_memory,
    evidence_store=arena.evidence_collector,
)
factory.register_with_coordinator(coordinator, adapters)
```

## Integration Bridges

### KnowledgeBridgeHub

```python
from aragora.knowledge import KnowledgeBridgeHub

hub = KnowledgeBridgeHub(mound)

# MetaLearner integration
await hub.meta_learner.capture_adjustment(
    metrics=learning_metrics,
    adjustments=hyperparams,
    cycle_number=5,
)

# Evidence storage
await hub.evidence.store_evidence(
    content="User data shows 40% prefer dark mode",
    source="analytics_report",
    evidence_type="statistical",
    strength=0.85,
)

# Pattern learning
await hub.patterns.store_pattern(
    pattern_type="critique_style",
    description="Agent identifies logical inconsistencies",
    occurrences=15,
    confidence=0.85,
)
```

## Core Concepts

### KnowledgeNode

```python
@dataclass
class KnowledgeNode:
    node_type: Literal["fact", "claim", "memory", "evidence", "consensus"]
    content: str
    confidence: float = 0.5
    provenance: ProvenanceChain
    tier: MemoryTier  # FAST, MEDIUM, SLOW, GLACIAL
    workspace_id: str

    # Graph relationships
    supports: list[str]        # Node IDs this supports
    contradicts: list[str]     # Node IDs this contradicts
    derived_from: list[str]    # Source node IDs
```

### ProvenanceChain

```python
@dataclass
class ProvenanceChain:
    source_type: ProvenanceType  # DOCUMENT, DEBATE, USER, AGENT, INFERENCE
    source_id: str
    agent_id: str | None
    debate_id: str | None
    transformations: list[dict]
    created_at: datetime
```

## Key Features

### Staleness Detection

Four-factor staleness score (0.0 = fresh, 1.0 = stale):
- Age (40%): Time since update vs tier threshold
- Contradictions (30%): Conflicting items added
- New Evidence (20%): Relevant evidence added
- Consensus Change (10%): Changes in related outcomes

```python
stale_items = await mound.get_stale_knowledge(threshold=0.7, limit=50)
for item in stale_items:
    await mound.schedule_revalidation(item.id, priority="high")
```

### Confidence Decay

Time-based confidence reduction:
- Exponential: `C(t) = C0 * e^(-lambda * t)`
- Linear: `C(t) = C0 - (rate * t)`
- Step: Periodic review cycles

### Graph Traversal

```python
result = await mound.query_graph(
    start_id="km_abc123",
    relationship_types=["supports", "derived_from"],
    depth=2,
)
```

### Culture Profile

```python
profile = await mound.get_culture_profile(workspace_id="team_a")
print(f"Decision style: {profile.dominant_traits['decision_style']}")
print(f"Risk tolerance: {profile.dominant_traits['risk_tolerance']}")
```

## Configuration

```python
config = MoundConfig(
    backend=MoundBackend.POSTGRES,
    postgres_url="postgresql://user:pass@localhost/aragora",
    postgres_pool_size=10,
    redis_url="redis://localhost:6379",
    redis_cache_ttl=300,  # 5 minutes
)
```

## Storage Backends

| Backend | Use Case |
|---------|----------|
| SQLite | Development, testing |
| PostgreSQL | Production (connection pooling, full-text search) |
| Hybrid | Enterprise (PostgreSQL + Redis caching) |

## Test Coverage

950+ tests covering:
- Core CRUD operations
- Semantic search and deduplication
- Staleness detection and decay
- All 32 adapters
- Federation and synchronization
- Resilience patterns

```bash
pytest tests/knowledge/mound/ -v
```

## Related Modules

- `aragora.memory` - Memory systems (continuum, consensus)
- `aragora.debate` - Uses knowledge during debates
- `aragora.knowledge.mound/README.md` - Detailed KM documentation
