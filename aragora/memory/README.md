# Memory Module

Multi-tier learning and persistence for organizational knowledge.

## Quick Start

```python
from aragora.memory import ContinuumMemory, MemoryTier

# Create multi-tier memory
memory = ContinuumMemory(db_path="aragora_memory.db")

# Store with tier selection
await memory.store("key", "value", tier=MemoryTier.FAST)

# Retrieve with tier-aware lookup
value = await memory.retrieve("key")
```

## Key Components

| Component | Purpose |
|-----------|---------|
| `ContinuumMemory` | Multi-tier learning with nested update frequencies |
| `ConsensusMemory` | Debate outcomes and historical decisions |
| `MemoryCoordinator` | Atomic transactions across memory systems |
| `CritiqueStore` | Pattern storage for critique feedback |
| `InMemoryBackend` | Fast, non-persistent storage for testing |

## Memory Tiers

| Tier | Half-Life | Update Frequency | Purpose |
|------|-----------|------------------|---------|
| FAST | 1 hour | Every event | Immediate patterns |
| MEDIUM | 24 hours | Per debate round | Tactical learning |
| SLOW | 7 days | Per nomic cycle | Strategic lessons |
| GLACIAL | 30 days | Monthly | Foundational knowledge |

## Architecture

```
memory/
├── continuum.py        # ContinuumMemory multi-tier system
├── consensus.py        # ConsensusMemory debate outcomes
├── coordinator.py      # MemoryCoordinator atomic transactions
├── store.py            # CritiqueStore pattern storage
├── tier_manager.py     # Memory tier lifecycle management
├── embeddings.py       # Semantic retrieval
├── hybrid_search.py    # Keyword + semantic search
├── surprise.py         # Surprise-based memorization
├── backends/           # Storage backends
│   ├── in_memory.py    # InMemoryBackend for testing
│   ├── postgres_*.py   # PostgreSQL backends
│   └── protocols.py    # Backend protocol definitions
└── tier_analytics.py   # Performance metrics
```

## Key Patterns

- **Nested Learning**: Catastrophic forgetting prevention
- **Tier Promotion**: Surprise-based importance elevation
- **Hybrid Search**: Combined keyword + semantic retrieval
- **Atomic Writes**: Coordinated cross-system transactions

## Related Documentation

- [CLAUDE.md](../../CLAUDE.md) - Project overview
- [docs/STATUS.md](../../docs/STATUS.md) - Feature status
