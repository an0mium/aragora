# ADR-002: Multi-Tier Memory System

## Status
Accepted

## Context
Aragora debates generate significant context that needs to be retained across:
- Individual turns (immediate context)
- Full debate sessions (session memory)
- Cross-debate learning (long-term patterns)
- Historical outcomes (consensus memory)

A single-tier approach would either:
- Be too expensive (keeping everything)
- Lose valuable context (aggressive pruning)

## Decision
We implemented a **four-tier memory continuum** with differentiated TTLs:

### Tier Structure
Located in `aragora/memory/continuum.py`:

| Tier | TTL | Purpose | Storage |
|------|-----|---------|---------|
| Fast | 1 min | Immediate turn context | In-memory |
| Medium | 1 hour | Current session | In-memory + optional Redis |
| Slow | 1 day | Cross-session patterns | SQLite/Supabase |
| Glacial | 1 week | Long-term learning | SQLite/Supabase |

### Memory Types

**CritiqueStore** (`aragora/memory/store.py`):
- Stores agent critiques with embeddings
- Enables semantic search for relevant past critiques
- Supports embedding-based retrieval

**ConsensusMemory** (`aragora/memory/consensus.py`):
- Historical debate outcomes
- Pattern detection for similar topics
- Informs future consensus strategies

**StreamingMemory** (`aragora/memory/streams.py`):
- Event-sourced memory updates
- Supports real-time streaming to clients
- Redis Streams backend for scalability

### Promotion/Demotion
- Entries promote based on access frequency
- Time-based decay demotes stale entries
- Explicit importance markers prevent demotion

## Consequences
**Positive:**
- Efficient resource usage (most data in lower tiers)
- Fast access for frequently-used context
- Long-term learning without infinite storage
- Semantic search enables relevant recall

**Negative:**
- Complexity in tier management
- Potential context loss on demotion
- Redis dependency for Medium tier at scale

## References
- `aragora/memory/continuum.py` - Tier implementation
- `aragora/memory/store.py` - CritiqueStore
- `aragora/memory/consensus.py` - ConsensusMemory
- `docs/MEMORY_TIERS.md` - Tier documentation
