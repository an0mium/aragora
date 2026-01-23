---
slug: 003-multi-tier-memory-system
title: "ADR-003: Multi-Tier Memory System"
description: "ADR-003: Multi-Tier Memory System"
---

# ADR-003: Multi-Tier Memory System

## Status
Accepted

## Context

Debate agents need access to historical context:
- Recent debate outcomes
- Successful argument patterns
- Agent performance history
- Cross-session learning

A single memory store doesn't fit all access patterns:
- Some data needs sub-second access (current debate)
- Some data accessed hourly (session context)
- Some data rarely accessed (historical patterns)

Memory costs and performance vary significantly by tier.

## Decision

Implement a four-tier memory system with automatic promotion/demotion:

```
┌─────────────────────────────────────────────────────────────┐
│                    ContinuumMemory                          │
├─────────────┬─────────────┬─────────────┬──────────────────┤
│    FAST     │   MEDIUM    │    SLOW     │     GLACIAL      │
│   (1 min)   │  (1 hour)   │   (1 day)   │    (1 week)      │
├─────────────┼─────────────┼─────────────┼──────────────────┤
│ In-memory   │ In-memory   │ SQLite/     │ Supabase/        │
│ dict        │ LRU cache   │ Redis       │ PostgreSQL       │
└─────────────┴─────────────┴─────────────┴──────────────────┘
```

### Tier Characteristics

| Tier | TTL | Storage | Use Case |
|------|-----|---------|----------|
| Fast | 1 min | Dict | Current debate context |
| Medium | 1 hour | LRU Cache | Session memories |
| Slow | 1 day | SQLite/Redis | Cross-session patterns |
| Glacial | 1 week | PostgreSQL | Long-term learning |

### Access Patterns

```python
memory = ContinuumMemory()

# Write to fast tier (auto-promotes on access)
await memory.store("key", value, tier="fast")

# Read checks all tiers, promotes on hit
value = await memory.retrieve("key")

# Async sweeper demotes expired entries
await memory.sweep()
```

## Consequences

### Positive
- **Performance**: Hot data in fast tiers
- **Cost efficiency**: Cold data in cheap storage
- **Automatic management**: Promotion/demotion handled
- **Flexibility**: Tiers can be configured per deployment

### Negative
- **Complexity**: Four storage backends to maintain
- **Consistency**: Cross-tier operations need care
- **Configuration**: Tier thresholds need tuning

### Neutral
- Optional tiers (can run with just fast/medium)
- Metrics track tier hit rates for optimization

## Related
- `aragora/memory/continuum.py` - ContinuumMemory implementation
- `aragora/memory/tier_manager.py` - Tier promotion/demotion
- `aragora/memory/consensus.py` - Consensus-specific memory
