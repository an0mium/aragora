# Memory Strategy Architecture

This document describes the multi-tier memory system, promotion/demotion algorithms, and the HOPE-inspired design principles.

## Overview

Aragora implements a tiered memory architecture inspired by the HOPE (Hierarchical Optimization of Past Experiences) pattern, where memories are organized by timescale and importance.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MEMORY TIER HIERARCHY                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    ┌─────────────────────────────────────────────────────────────────────┐  │
│    │                    FAST TIER (Event-Level)                          │  │
│    │    TTL: 1 minute    │    Max: 1000 items    │    Updates: Per event │  │
│    │    Importance ≥ 0.8                                                 │  │
│    └─────────────────────────────────────────────────────────────────────┘  │
│                                    ▼ decay                                   │
│    ┌─────────────────────────────────────────────────────────────────────┐  │
│    │                   MEDIUM TIER (Round-Level)                         │  │
│    │    TTL: 1 hour      │    Max: 500 items     │    Updates: Per round │  │
│    │    Importance ≥ 0.5                                                 │  │
│    └─────────────────────────────────────────────────────────────────────┘  │
│                                    ▼ decay                                   │
│    ┌─────────────────────────────────────────────────────────────────────┐  │
│    │                    SLOW TIER (Cycle-Level)                          │  │
│    │    TTL: 1 day       │    Max: 200 items     │    Updates: Per cycle │  │
│    │    Importance ≥ 0.3                                                 │  │
│    └─────────────────────────────────────────────────────────────────────┘  │
│                                    ▼ consolidate                             │
│    ┌─────────────────────────────────────────────────────────────────────┐  │
│    │                  GLACIAL TIER (Archival)                            │  │
│    │    TTL: 1 week      │    Max: 100 items     │    Updates: Monthly   │  │
│    │    Importance < 0.3 or explicitly archived                          │  │
│    └─────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Memory Types

### ContinuumMemory

**Module:** `aragora/memory/continuum.py`

The primary multi-tier memory system for storing and retrieving debate-related memories.

```python
class ContinuumMemory:
    """Multi-tier memory with automatic promotion/demotion."""

    def store(
        self,
        content: str,
        tier: MemoryTier = MemoryTier.SLOW,
        importance: float = 0.5,
        source: str = "",
        topic: str = "",
        metadata: dict = None,
    ) -> MemoryEntry

    def retrieve(
        self,
        query: str,
        tiers: list[MemoryTier] = None,
        limit: int = 5,
        min_similarity: float = 0.3,
    ) -> list[MemoryEntry]
```

### ConsensusMemory

**Module:** `aragora/memory/consensus.py`

Specialized storage for debate outcomes and dissenting views.

```python
class ConsensusMemory:
    """Persistent storage for consensus and dissent."""

    def store_consensus(self, record: ConsensusRecord) -> str
    def store_dissent(self, record: DissentRecord) -> str
    def find_similar(self, topic: str, limit: int = 5) -> list[SimilarDebate]
    def get_dissent_for_topic(self, topic_hash: str) -> list[DissentRecord]
```

### MemoryStream

**Module:** `aragora/memory/streams.py`

Per-agent episodic memory with temporal ordering.

```python
class MemoryStream:
    """Agent-specific memory timeline."""

    def record(self, event: MemoryEvent) -> None
    def recall(self, query: str, limit: int = 10) -> list[MemoryEvent]
    def summarize(self, time_range: tuple) -> str
```

## Tier Configuration

```python
TIER_CONFIGS = {
    MemoryTier.FAST: TierConfig(
        ttl_seconds=60,           # 1 minute
        max_items=1000,
        importance_threshold=0.8,
        update_frequency="per_event",
    ),
    MemoryTier.MEDIUM: TierConfig(
        ttl_seconds=3600,         # 1 hour
        max_items=500,
        importance_threshold=0.5,
        update_frequency="per_round",
    ),
    MemoryTier.SLOW: TierConfig(
        ttl_seconds=86400,        # 1 day
        max_items=200,
        importance_threshold=0.3,
        update_frequency="per_cycle",
    ),
    MemoryTier.GLACIAL: TierConfig(
        ttl_seconds=604800,       # 1 week
        max_items=100,
        importance_threshold=0.0,
        update_frequency="monthly",
    ),
}
```

## Promotion/Demotion Algorithm

### Promotion Criteria

Memories are promoted to a higher tier when they demonstrate sustained relevance:

```python
def should_promote(entry: MemoryEntry) -> bool:
    """Determine if memory should be promoted to higher tier."""

    # Recently accessed frequently
    if entry.access_count >= PROMOTION_ACCESS_THRESHOLD:
        return True

    # High surprise score (unexpectedly relevant)
    if entry.surprise_score >= SURPRISE_THRESHOLD:
        return True

    # Referenced in successful debate outcomes
    if entry.success_references >= SUCCESS_REFERENCE_THRESHOLD:
        return True

    return False
```

### Demotion Criteria

Memories decay to lower tiers over time:

```python
def should_demote(entry: MemoryEntry) -> bool:
    """Determine if memory should be demoted to lower tier."""

    # Hasn't been accessed recently
    if entry.last_access_age > TIER_CONFIGS[entry.tier].ttl_seconds:
        return True

    # Importance has decayed below threshold
    if entry.decayed_importance < TIER_CONFIGS[entry.tier].importance_threshold:
        return True

    return False
```

### Importance Decay Function

```python
def decay_importance(entry: MemoryEntry, current_time: float) -> float:
    """Calculate decayed importance based on age."""
    age_hours = (current_time - entry.created_at) / 3600

    # Exponential decay with half-life based on tier
    half_life = TIER_HALF_LIVES[entry.tier]
    decay_factor = 0.5 ** (age_hours / half_life)

    return entry.base_importance * decay_factor
```

## Surprise Score Mechanism

The surprise score measures how unexpectedly relevant a memory is when retrieved:

```python
def update_surprise_score(entry: MemoryEntry, retrieval_context: dict) -> None:
    """Update surprise score after retrieval."""

    # Low prior relevance prediction, high actual usefulness = surprise
    predicted_relevance = retrieval_context.get("predicted_relevance", 0.5)
    actual_usefulness = retrieval_context.get("actual_usefulness", 0.5)

    surprise = max(0, actual_usefulness - predicted_relevance)

    # Exponential moving average
    alpha = 0.3
    entry.surprise_score = alpha * surprise + (1 - alpha) * entry.surprise_score
```

High surprise scores indicate memories that proved valuable in unexpected contexts, triggering promotion.

## Consolidation Process

Periodically, memories are consolidated from lower tiers into higher ones:

```python
async def consolidate_memories(self) -> ConsolidationResult:
    """Consolidate and summarize memories across tiers."""

    # Process tier pairs from bottom to top
    tier_pairs = [
        (MemoryTier.GLACIAL, MemoryTier.SLOW),
        (MemoryTier.SLOW, MemoryTier.MEDIUM),
        (MemoryTier.MEDIUM, MemoryTier.FAST),
    ]

    for source_tier, target_tier in tier_pairs:
        candidates = self._get_promotion_candidates(source_tier)

        for entry in candidates:
            if self.should_promote(entry):
                self._move_to_tier(entry, target_tier)

        # Summarize remaining entries
        if len(self._get_entries(source_tier)) > TIER_CONFIGS[source_tier].max_items:
            self._summarize_and_evict(source_tier)
```

## Retrieval Strategy

### Multi-Tier Search

Retrieval searches across tiers with configurable strategy:

```python
def retrieve(
    self,
    query: str,
    tiers: list[MemoryTier] = None,
    limit: int = 5,
    strategy: str = "weighted",
) -> list[MemoryEntry]:
    """
    Retrieve memories across tiers.

    Strategies:
    - "fast_first": Search FAST, fall through if insufficient
    - "weighted": Weight results by tier (FAST > MEDIUM > SLOW)
    - "comprehensive": Search all tiers, merge results
    """
```

### Embedding-Based Similarity

Semantic search uses embeddings for relevance:

```python
def _compute_similarity(self, query: str, entry: MemoryEntry) -> float:
    """Compute semantic similarity between query and memory."""
    query_embedding = self._embed(query)
    entry_embedding = entry.embedding

    return cosine_similarity(query_embedding, entry_embedding)
```

### Hybrid Search (Vector + Keyword)

For higher recall, hybrid search combines vector similarity with keyword
matches using Reciprocal Rank Fusion (RRF). This is useful for exact terms and
acronyms that embeddings may miss.

```python
from aragora.memory import get_hybrid_memory_search

search = get_hybrid_memory_search(continuum_memory)
results = await search.search("SLA breach", limit=10)
```

## Integration with Debate Phases

### Phase 0 (Context Init)
- Retrieves relevant memories from SLOW and MEDIUM tiers
- Injects historical context into debate

### Phase 2 (Debate Rounds)
- Stores critique patterns in FAST tier
- Updates surprise scores on retrieval

### Phase 7 (Feedback)
- Stores debate outcome in SLOW tier
- Triggers consolidation if needed
- Updates memory outcomes for referenced entries

## Storage Backend

All memory tiers use SQLite with WAL mode for concurrent access:

```python
def get_wal_connection(db_path: Path, timeout: float = 30.0) -> Connection:
    """Get SQLite connection with WAL mode enabled."""
    conn = sqlite3.connect(db_path, timeout=timeout)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(f"PRAGMA busy_timeout = {int(timeout * 1000)}")
    return conn
```

## Monitoring and Metrics

```python
def get_memory_stats(self) -> dict:
    """Get memory system statistics."""
    return {
        "tier_counts": {tier: len(entries) for tier, entries in self._tiers.items()},
        "promotions_last_hour": self._promotion_count,
        "demotions_last_hour": self._demotion_count,
        "hit_rate": self._cache_hits / (self._cache_hits + self._cache_misses),
        "avg_surprise_score": self._compute_avg_surprise(),
    }
```

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) - System overview
- [DEBATE_PHASES.md](DEBATE_PHASES.md) - Debate execution phases
- [DATABASE.md](DATABASE.md) - Database strategy
