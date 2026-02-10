# Knowledge Integration Guide

How to integrate external knowledge sources with Aragora's Knowledge Mound, create
custom adapters, and manage knowledge flow through debates.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Knowledge Mound Basics](#knowledge-mound-basics)
- [Knowledge Bridges](#knowledge-bridges)
- [Adapter System](#adapter-system)
- [Creating Custom Adapters](#creating-custom-adapters)
- [Adapter Factory](#adapter-factory)
- [Bidirectional Coordination](#bidirectional-coordination)
- [Knowledge Flow Through Debates](#knowledge-flow-through-debates)
- [Configuration Reference](#configuration-reference)
- [Common Patterns](#common-patterns)

---

## Architecture Overview

The Knowledge Mound is Aragora's unified knowledge storage layer, following a
"termite mound" architecture where all agents contribute to and query from a
shared knowledge superstructure.

```
                    ┌─────────────────────────┐
                    │     Knowledge Mound      │
                    │  (Unified Knowledge API)  │
                    └─────────┬───────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
    ┌───────▼───────┐ ┌──────▼──────┐  ┌──────▼───────┐
    │   Bridges     │ │  Adapters   │  │  Direct API  │
    │ (Meta, Evid,  │ │ (20+ built- │  │  (store,     │
    │  Pattern)     │ │  in types)  │  │   query)     │
    └───────┬───────┘ └──────┬──────┘  └──────┬───────┘
            │                │                 │
    ┌───────▼───────┐ ┌──────▼──────┐  ┌──────▼───────┐
    │ MetaLearner   │ │ Continuum   │  │   Arena      │
    │ Evidence      │ │ Consensus   │  │   (Debate    │
    │ Patterns      │ │ ELO, Pulse  │  │    Engine)   │
    └───────────────┘ └─────────────┘  └──────────────┘
```

Key modules:
- `aragora/knowledge/mound/` - Core mound implementation and types
- `aragora/knowledge/mound/facade.py` - `KnowledgeMound` main class
- `aragora/knowledge/mound/adapters/` - 20+ built-in adapters
- `aragora/knowledge/mound/adapters/factory.py` - Automatic adapter creation
- `aragora/knowledge/bridges.py` - High-level integration bridges
- `aragora/knowledge/mound/bidirectional_coordinator.py` - Sync coordination

---

## Knowledge Mound Basics

### Initialization

```python
from aragora.knowledge.mound import KnowledgeMound, MoundConfig, MoundBackend

# Development: SQLite backend (default)
mound = KnowledgeMound(workspace_id="my_team")
await mound.initialize()

# Production: PostgreSQL + Redis caching
config = MoundConfig(
    backend=MoundBackend.HYBRID,
    postgres_url="postgresql://user:pass@host/db",
    redis_url="redis://localhost:6379",
)
mound = KnowledgeMound(config, workspace_id="enterprise")
await mound.initialize()
```

### Storing Knowledge

```python
from aragora.knowledge.mound import IngestionRequest, KnowledgeSource

result = await mound.store(IngestionRequest(
    content="Contracts require 90-day notice for termination",
    source_type=KnowledgeSource.DEBATE,
    debate_id="debate_123",
    workspace_id="my_team",
))
print(f"Stored as node: {result.node_id}")
```

### Querying Knowledge

```python
# Text search
results = await mound.query("contract notice requirements")

# Check for stale knowledge
stale_items = await mound.get_stale_knowledge(threshold=0.7)
```

### Knowledge Node Types

The mound stores knowledge as `KnowledgeNode` objects with these types:

| Node Type | Description | Typical Source |
|-----------|-------------|----------------|
| `fact` | Established facts and patterns | Debates, meta-learning |
| `evidence` | Supporting evidence with citations | Evidence collector |
| `claim` | Claims that need verification | Agent proposals |
| `opinion` | Agent perspectives | Debate rounds |

### Provenance Tracking

Every node carries a `ProvenanceChain` for audit:

```python
from aragora.knowledge.mound import ProvenanceChain, ProvenanceType

provenance = ProvenanceChain(
    source_type=ProvenanceType.DEBATE,      # DEBATE, AGENT, DOCUMENT, INFERENCE, USER
    source_id="debate_456",
    transformations=[
        {
            "type": "consensus",
            "details": {"round": 5, "agreement": 0.85},
            "timestamp": "2025-01-15T10:30:00Z",
        }
    ],
)
```

---

## Knowledge Bridges

Bridges (`aragora/knowledge/bridges.py`) provide high-level connectors between
Aragora subsystems and the Knowledge Mound. Use the `KnowledgeBridgeHub` for
centralized access.

### KnowledgeBridgeHub

```python
from aragora.knowledge.bridges import KnowledgeBridgeHub

hub = KnowledgeBridgeHub(mound)

# All bridges are lazily initialized
hub.meta_learner   # MetaLearnerBridge
hub.evidence       # EvidenceBridge
hub.patterns       # PatternBridge
```

### MetaLearnerBridge

Captures hyperparameter adjustments from the meta-learning system:

```python
# Capture a hyperparameter adjustment
node_id = await hub.meta_learner.capture_adjustment(
    metrics=learning_metrics,        # LearningMetrics dataclass
    adjustments={"decay_rate": 0.98},
    hyperparams=hyperparams_state,
    cycle_number=42,
)

# Capture a learning summary
node_id = await hub.meta_learner.capture_learning_summary(summary)
```

Adjustments are stored as `fact` nodes with `MemoryTier.MEDIUM` (24-hour half-life).
Summaries are stored with `MemoryTier.SLOW` (7-day half-life).

### EvidenceBridge

Stores external evidence as knowledge nodes:

```python
# Store evidence directly
node_id = await hub.evidence.store_evidence(
    content="Study shows 90-day notice reduces disputes by 40%",
    source="https://example.com/study",
    evidence_type="citation",    # citation, data, tool_output
    supports_claim=True,
    strength=0.8,
    metadata={"year": 2024},
)

# Store from Evidence Collector output
node_id = await hub.evidence.store_from_collector_evidence(
    evidence=collector_evidence,
    claim_node_id="node_abc",    # Link to the claim this supports
)
```

Evidence nodes automatically track support/contradiction relationships to claims.

### PatternBridge

Stores detected patterns from debate analysis:

```python
# Store a generic pattern
node_id = await hub.patterns.store_pattern(
    pattern_type="debate",
    description="Teams with 5+ agents reach consensus 20% faster",
    occurrences=15,
    confidence=0.75,
    source_ids=["debate_1", "debate_2"],
)

# Store a critique pattern
node_id = await hub.patterns.store_critique_pattern(
    pattern_description="Consistently identifies logical fallacies",
    agent_name="claude-sonnet",
    frequency=8,
    effectiveness=0.9,
)

# Store a debate pattern
node_id = await hub.patterns.store_debate_pattern(
    pattern_description="Early convergence on technical topics",
    debate_ids=["d1", "d2", "d3"],
    consensus_rate=0.85,
)
```

Patterns are automatically tiered by frequency:
- 1-2 occurrences: `MemoryTier.FAST` (1-hour half-life)
- 3-9 occurrences: `MemoryTier.MEDIUM` (24-hour half-life)
- 10+ occurrences: `MemoryTier.SLOW` (7-day half-life)

---

## Adapter System

Adapters provide bidirectional synchronization between Arena subsystems and the
Knowledge Mound. There are 20+ built-in adapters defined in
`aragora/knowledge/mound/adapters/factory.py`.

### Built-in Adapters

| Adapter | Priority | Source Subsystem | Forward Sync | Reverse Sync |
|---------|:--------:|------------------|--------------|--------------|
| `continuum` | 100 | ContinuumMemory | `store` | `sync_validations_to_continuum` |
| `consensus` | 90 | ConsensusMemory | `sync_to_km` | `sync_validations_from_km` |
| `control_plane` | 85 | ControlPlane | `sync_from_control_plane` | `get_policy_recommendations` |
| `critique` | 80 | CritiqueStore | `store` | `sync_validations_from_km` |
| `provenance` | 75 | ProvenanceStore | `ingest_provenance` | (none, append-only) |
| `evidence` | 70 | EvidenceStore | `store` | `update_reliability_from_km` |
| `receipt` | 65 | ReceiptStore | `ingest_receipt` | `find_related_decisions` |
| `belief` | 60 | BeliefNetwork | `store_converged_belief` | `sync_validations_from_km` |
| `calibration_fusion` | 55 | CalibrationTracker | `sync_from_calibration` | `get_calibration_insights` |
| `insights` | 50 | InsightStore | `store_insight` | `sync_validations_from_km` |
| `performance` | 45 | EloSystem | `store_match` | `sync_km_to_elo` |
| `elo` | 40 | EloSystem | `store_match` | `sync_km_to_elo` |
| `fabric` | 35 | AgentFabric | `sync_from_fabric` | `get_pool_recommendations` |
| `workspace` | 34 | WorkspaceManager | `sync_from_workspace` | `get_rig_recommendations` |
| `computer_use` | 33 | ComputerUseOrch. | `sync_from_orchestrator` | `get_similar_tasks` |
| `gateway` | 32 | LocalGateway | `sync_from_gateway` | `get_routing_recommendations` |
| `pulse` | 30 | PulseManager | `store_trending_topic` | `sync_validations_from_km` |
| `culture` | 25 | CultureManager | `sync_to_mound` | `load_from_mound` |
| `rlm` | 20 | RLM Compressor | `sync_to_mound` | `load_from_mound` |
| `cost` | 10 | CostTracker | `store_anomaly` | `sync_validations_from_km` |

Higher priority adapters sync first during bidirectional cycles.

### Adapter Specification

Each adapter is defined by an `AdapterSpec`:

```python
from aragora.knowledge.mound.adapters.factory import AdapterSpec

spec = AdapterSpec(
    name="continuum",                          # Unique identifier
    adapter_class=ContinuumAdapter,            # Python class
    required_deps=["continuum_memory"],         # Required subsystem dependencies
    forward_method="store",                     # Source -> KM method
    reverse_method="sync_validations_to_continuum",  # KM -> Source method
    priority=100,                               # Sync order (higher = first)
    enabled_by_default=True,                    # Auto-enable on creation
    config_key="km_continuum_adapter",          # ArenaConfig key for explicit adapter
)
```

---

## Creating Custom Adapters

### Step 1: Define the Adapter Class

Create a class that implements the forward and reverse sync methods:

```python
# aragora/knowledge/mound/adapters/my_adapter.py

from __future__ import annotations
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class MyCustomAdapter:
    """Adapter for syncing MySubsystem data to/from Knowledge Mound."""

    def __init__(
        self,
        my_subsystem: Any,
        event_callback: Optional[Callable[[str, dict], None]] = None,
    ):
        self._subsystem = my_subsystem
        self._event_callback = event_callback

    async def sync_to_km(self, mound: Any) -> dict:
        """Forward sync: Push data from MySubsystem into the Knowledge Mound.

        Returns:
            Dict with sync statistics (items_processed, items_updated, errors).
        """
        items = self._subsystem.get_pending_items()
        processed = 0
        errors = []

        for item in items:
            try:
                from aragora.knowledge.mound import KnowledgeNode, ProvenanceChain, ProvenanceType
                from aragora.memory.tier_manager import MemoryTier

                node = KnowledgeNode(
                    node_type="fact",
                    content=item.content,
                    confidence=item.confidence,
                    provenance=ProvenanceChain(
                        source_type=ProvenanceType.AGENT,
                        source_id=f"my_subsystem_{item.id}",
                    ),
                    tier=MemoryTier.MEDIUM,
                    workspace_id=mound.workspace_id,
                )
                await mound.add_node(node)
                processed += 1
            except Exception as e:
                errors.append(str(e))

        return {"items_processed": processed, "errors": errors}

    async def sync_from_km(self, mound: Any) -> dict:
        """Reverse sync: Pull KM validations back into MySubsystem.

        Returns:
            Dict with sync statistics.
        """
        # Query relevant nodes from KM
        results = await mound.query("my_subsystem_related")
        updated = 0

        for result in results:
            if hasattr(result, "confidence"):
                self._subsystem.update_confidence(result.node_id, result.confidence)
                updated += 1

        return {"items_updated": updated}
```

### Step 2: Register the Adapter Spec

Add the adapter to the factory registry:

```python
from aragora.knowledge.mound.adapters.factory import AdapterSpec, register_adapter_spec

register_adapter_spec(
    AdapterSpec(
        name="my_custom",
        adapter_class=MyCustomAdapter,
        required_deps=["my_subsystem"],         # Must match kwarg name in create_from_subsystems
        forward_method="sync_to_km",
        reverse_method="sync_from_km",
        priority=50,                             # Medium priority
        enabled_by_default=True,
        config_key="km_my_custom_adapter",
    )
)
```

### Step 3: Use with AdapterFactory

```python
from aragora.knowledge.mound.adapters.factory import AdapterFactory

factory = AdapterFactory(event_callback=my_event_callback)

# Create from explicit subsystems
adapters = factory.create_from_subsystems(
    my_subsystem=my_subsystem_instance,
    continuum_memory=continuum,
    elo_system=elo,
)

# Or create from ArenaConfig
adapters = factory.create_from_arena_config(
    config=arena_config,
    subsystems={"my_subsystem": my_subsystem_instance},
)
```

### Step 4: Register with Coordinator

```python
from aragora.knowledge.mound.bidirectional_coordinator import BidirectionalCoordinator

coordinator = BidirectionalCoordinator()
registered_count = factory.register_with_coordinator(coordinator, adapters)
print(f"Registered {registered_count} adapters")
```

---

## Adapter Factory

The `AdapterFactory` (`aragora/knowledge/mound/adapters/factory.py`) automates
adapter creation based on available subsystem dependencies.

### Automatic Creation

The factory checks which subsystem dependencies are available and creates
only the adapters whose requirements are satisfied:

```python
factory = AdapterFactory()

# Only adapters whose required_deps are present will be created
adapters = factory.create_from_subsystems(
    continuum_memory=my_continuum,   # Enables: continuum adapter
    elo_system=my_elo,               # Enables: elo adapter
    # evidence_store not provided    # Skips: evidence adapter
)
```

### ArenaConfig Integration

When using `create_from_arena_config`, explicitly configured adapters take
precedence over auto-created ones:

```python
adapters = factory.create_from_arena_config(
    config=arena_config,              # Checks config attributes for explicit adapters
    subsystems={                      # Fallback for auto-creation
        "continuum_memory": continuum,
        "evidence_store": evidence,
    },
)
```

### Listing Available Adapters

```python
specs = factory.get_available_adapter_specs()
for name, spec in specs.items():
    print(f"{name}: requires {spec.required_deps}, priority={spec.priority}")
```

---

## Bidirectional Coordination

The `BidirectionalCoordinator` (`aragora/knowledge/mound/bidirectional_coordinator.py`)
manages the sync lifecycle across all adapters.

### Setup

```python
from aragora.knowledge.mound.bidirectional_coordinator import BidirectionalCoordinator

coordinator = BidirectionalCoordinator()

# Register adapters manually
coordinator.register_adapter(
    name="continuum",
    adapter=continuum_adapter,
    forward_method="store",
    reverse_method="sync_validations_to_continuum",
    priority=100,
)

# Or use the factory (preferred)
factory.register_with_coordinator(coordinator, adapters)
```

### Running Sync

```python
# Full bidirectional sync
report = await coordinator.run_bidirectional_sync()

print(f"Forward: {report.successful_forward}/{report.total_adapters}")
print(f"Reverse: {report.successful_reverse}/{report.total_adapters}")
print(f"Errors: {report.total_errors}")
print(f"Duration: {report.total_duration_ms}ms")
```

### Adapter Lifecycle

```python
# Disable an adapter temporarily
coordinator.disable_adapter("cost")

# Re-enable
coordinator.enable_adapter("cost")
```

Adapters disabled by default (like `cost` and `performance`) must be explicitly
enabled:

```python
coordinator.enable_adapter("cost")  # Opt-in adapter
```

---

## Knowledge Flow Through Debates

Knowledge flows through the debate system in a cycle:

```
                     ┌─────────────┐
                     │  Pre-Debate  │
                     │  (Research)  │
                     └──────┬──────┘
                            │ Query KM for context
                            ▼
                     ┌─────────────┐
                     │   Debate     │
                     │  (Arena)     │
                     │  Rounds 0-8  │
                     └──────┬──────┘
                            │ Store results via adapters
                            ▼
                     ┌─────────────┐
                     │ Post-Debate  │
                     │ (Adapters)   │
                     └──────┬──────┘
                            │ Forward sync to KM
                            ▼
                     ┌─────────────┐
                     │ Revalidation │
                     │ (Scheduler)  │
                     └──────┬──────┘
                            │ Reverse sync from KM
                            ▼
                     ┌─────────────┐
                     │ Next Debate  │
                     │ (Enriched)   │
                     └─────────────┘
```

### Pre-Debate Knowledge Injection

Before a debate starts, relevant knowledge is queried from the mound:

```python
# The Arena queries KM during the context gathering phase (Round 0)
# Knowledge nodes matching the debate topic are injected into agent prompts

context_knowledge = await mound.query(
    debate_topic,
    filters=QueryFilters(
        workspace_id="my_team",
        min_confidence=0.6,
    ),
)
```

### Post-Debate Knowledge Capture

After a debate completes, results flow back through adapters:

1. **Consensus decisions** are stored via the `consensus` adapter
2. **Agent critiques** are stored via the `critique` adapter
3. **ELO ratings** are updated via the `elo` adapter
4. **Evidence gathered** is stored via the `evidence` adapter
5. **Decision receipts** are stored via the `receipt` adapter

### Cross-Debate Learning

The `culture` adapter accumulates organizational patterns over time:

```python
# Culture patterns emerge from repeated debate outcomes
# These are stored with MemoryTier.GLACIAL (30-day half-life)
# and used to inform future debates about organizational preferences
```

---

## Configuration Reference

### MoundConfig Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `MoundBackend` | `SQLITE` | Storage backend (`SQLITE`, `POSTGRES`, `HYBRID`) |
| `postgres_url` | `str` | `None` | PostgreSQL connection URL |
| `redis_url` | `str` | `None` | Redis URL for caching |
| `default_workspace_id` | `str` | `"default"` | Default workspace for queries |

### Redis Cache Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `default_ttl` | 300 (5 min) | Default cache TTL for items |
| `culture_ttl` | 3600 (1 hour) | TTL for culture patterns |
| `max_entries` | 10,000 | Maximum cached entries before LRU eviction |

### Query Cache Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ARAGORA_QUERY_CACHE_ENABLED` | `true` | Enable request-scoped query cache |
| `ARAGORA_QUERY_CACHE_MAX_SIZE` | `1000` | Max entries per request cache |

### Adapter Config Keys

Each adapter can be explicitly configured via `ArenaConfig` attributes:

| Config Key | Adapter |
|------------|---------|
| `km_continuum_adapter` | Continuum |
| `km_consensus_adapter` | Consensus |
| `km_critique_adapter` | Critique |
| `km_evidence_adapter` | Evidence |
| `km_belief_adapter` | Belief |
| `km_insights_adapter` | Insights |
| `km_elo_bridge` | ELO |
| `km_performance_adapter` | Performance |
| `km_pulse_adapter` | Pulse |
| `km_cost_adapter` | Cost |
| `km_provenance_adapter` | Provenance |
| `km_control_plane_adapter` | Control Plane |
| `km_receipt_adapter` | Receipt |
| `km_culture_adapter` | Culture |
| `km_rlm_adapter` | RLM |
| `km_calibration_fusion_adapter` | Calibration Fusion |

---

## Common Patterns

### Storing Debate Outcomes as Knowledge

```python
from aragora.knowledge.mound import KnowledgeNode, ProvenanceChain, ProvenanceType
from aragora.memory.tier_manager import MemoryTier

# After consensus is reached
node = KnowledgeNode(
    node_type="fact",
    content=consensus_text,
    confidence=consensus_score,
    provenance=ProvenanceChain(
        source_type=ProvenanceType.DEBATE,
        source_id=debate_id,
        transformations=[{
            "type": "consensus",
            "details": {"round": final_round, "method": "judge"},
            "timestamp": datetime.now().isoformat(),
        }],
    ),
    tier=MemoryTier.SLOW,       # Consensus decisions are long-term
    workspace_id=workspace_id,
)
await mound.add_node(node)
```

### Querying Knowledge for Agent Context

```python
# During prompt building, inject relevant knowledge
results = await mound.query(topic)
knowledge_context = "\n".join(
    f"- [{r.confidence:.0%}] {r.content}" for r in results
)
prompt = f"Context from organizational knowledge:\n{knowledge_context}\n\nQuestion: {topic}"
```

### Event-Driven Adapter Updates

```python
# Adapters can emit WebSocket events for real-time UI updates
factory = AdapterFactory(
    event_callback=lambda event_type, data: ws_server.broadcast(event_type, data)
)
```

### Workspace Isolation

All knowledge operations are scoped to a workspace:

```python
# Each team/tenant has isolated knowledge
team_a_mound = KnowledgeMound(workspace_id="team_a")
team_b_mound = KnowledgeMound(workspace_id="team_b")

# Cross-workspace queries require explicit federation
# See aragora/knowledge/mound/federated_query.py
```
