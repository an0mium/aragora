# Memory Module

Multi-tier learning and persistence system implementing nested learning with catastrophic forgetting prevention.

## Overview

The memory module provides a four-tier memory system (fast/medium/slow/glacial) inspired by Google Research's Nested Learning paradigm. It coordinates atomic writes across multiple storage systems and integrates with the Knowledge Mound for unified knowledge management.

## Quick Start

```python
from aragora.memory import ContinuumMemory, ConsensusMemory

# Initialize memory systems
continuum = ContinuumMemory()
consensus = ConsensusMemory()

# Store a memory
await continuum.store(
    key="rate_limiter_design",
    content="Token bucket with sliding window",
    tier=MemoryTier.MEDIUM,
    importance=0.8,
)

# Query memories
results = await continuum.query("rate limiting patterns", limit=10)

# Store consensus outcome
await consensus.store_outcome(
    debate_id="debate-123",
    decision="Use token bucket algorithm",
    confidence=0.92,
    dissenting_views=["Fixed window advocate"],
)
```

## Key Files

| File | Purpose |
|------|---------|
| `continuum.py` | PostgreSQL-backed multi-tier memory system |
| `consensus.py` | Debate outcome storage, dissent tracking |
| `coordinator.py` | Atomic transaction coordinator with rollback |
| `tier_manager.py` | Tier lifecycle, promotion/demotion logic |
| `store.py` | CritiqueStore for patterns and reputation |
| `embeddings.py` | Semantic retrieval (OpenAI, Gemini, Ollama) |
| `hybrid_search.py` | Combined keyword + semantic search |
| `cross_debate_rlm.py` | RLM integration for institutional memory |

## Memory Tier System

Four nested tiers with different learning rates and retention:

| Tier | Half-Life | Learning Rate | Use Case |
|------|-----------|---------------|----------|
| **FAST** | 1 hour | 30% | Immediate patterns, event-based decisions |
| **MEDIUM** | 24 hours | 10% | Tactical learning, debate round outcomes |
| **SLOW** | 7 days | 3% | Strategic lessons, stable patterns |
| **GLACIAL** | 30 days | 1% | Foundational knowledge, irreversible decisions |

### Tier Transitions

**Promotion** (to faster tier):
- High surprise score (>0.7) triggers promotion
- Novel patterns move to fast tier for rapid learning

**Demotion** (to slower tier):
- High stability score (consolidation >0.8, surprise <0.2)
- Established patterns move to glacial for long-term retention

**Red Lines:**
```python
# Mark critical memories as non-deletable
await continuum.mark_red_line(memory_id, promote_to_glacial=True)
```

## Memory During Debates

### Context Gathering

```python
# Automatic retrieval during debate initialization
memories = await continuum.query(
    query=debate_task,
    limit=20,
    min_importance=0.5,
)
# Memories injected into agent prompts
```

### Post-Debate Storage

The `MemoryCoordinator` performs atomic writes:

```python
from aragora.memory import MemoryCoordinator

coordinator = MemoryCoordinator(
    continuum_memory=continuum,
    consensus_memory=consensus,
    critique_store=critique_store,
    knowledge_mound=mound,
)

await coordinator.commit_debate_outcome(
    debate_id="debate-123",
    outcome=result,
    rollback_on_failure=True,
)
```

Systems updated atomically:
- **ContinuumMemory** - Debate outcome with importance scoring
- **ConsensusMemory** - Agreement strength, dissenting views
- **CritiqueStore** - Critique patterns, agent reputation
- **KnowledgeMound** - Unified knowledge graph

## Configuration

### Arena Integration

```python
arena = Arena(
    memory_config=MemoryConfig(
        enable_knowledge_retrieval=True,
        enable_knowledge_ingestion=True,
        enable_continuum_memory=True,
        enable_cross_debate_memory=True,
    )
)
```

### Coordinator Options

```python
coordinator = MemoryCoordinator(
    write_continuum=True,
    write_consensus=True,
    write_critique=True,
    write_mound=True,
    rollback_on_failure=True,
    parallel_writes=False,  # Sequential for safety
    min_confidence_for_mound=0.7,
    timeout_seconds=30.0,
)
```

### Tier Manager

```python
tier_manager = TierManager(
    promotion_cooldown_hours=24.0,
    min_updates_for_demotion=10,
)

# Automatic glacial promotion scheduler
scheduler = GlacialPromotionScheduler(
    interval_hours=24.0,
    min_consolidation=0.8,
    max_surprise=0.2,
    min_update_count=20,
)
```

## Knowledge Mound Integration

Bidirectional sync with KnowledgeMound:

```python
# Query similar entries from KM
similar = await continuum.query_km_for_similar(query, limit=10)

# Pre-warm cache for common queries
await continuum.prewarm_for_query(query_pattern)

# Invalidate stale cross-references
await continuum.invalidate_reference(km_node_id)
```

Cache strategy:
- KM similarity queries: 5 min TTL, 1000 entries max
- Consensus queries: 5 min TTL, 2000 entries max

## Surprise-Based Learning

Memories are scored by surprise (novelty):

```python
surprise_score = (
    0.3 * success_rate_novelty +
    0.3 * semantic_surprisal +
    0.2 * timing_surprise +
    0.2 * agent_prediction_error
)
```

High surprise → promote to faster tier for rapid learning.
Low surprise → demote to slower tier for archival.

## Storage Backends

| Backend | Use Case |
|---------|----------|
| SQLite | Development, testing |
| PostgreSQL | Production (async connection pooling) |
| Redis | Caching layer |

Default paths:
- Continuum: `.aragora_beads/continuum_memory.db`
- Consensus: `.aragora_beads/consensus_memory.db`

## Related Modules

- `aragora.knowledge` - Knowledge Mound integration
- `aragora.debate` - Uses memory during debates
- `aragora.memory.backends` - Storage implementations
