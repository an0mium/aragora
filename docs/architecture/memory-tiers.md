# Memory Tiers

Aragora uses a four-tier memory architecture (`ContinuumMemory`) to balance
speed, cost, and retention. Each tier serves a distinct temporal purpose, and
data flows from faster tiers to slower ones as it proves durably useful.

## Tier Architecture

```mermaid
flowchart TD
    Input([New Memory Entry]) --> Fast

    subgraph Fast["Fast Tier"]
        F_TTL["TTL: 1 minute"]
        F_Purpose["Immediate debate context"]
        F_Store["In-memory cache"]
    end

    Fast -->|Promoted on reuse| Medium

    subgraph Medium["Medium Tier"]
        M_TTL["TTL: 1 hour"]
        M_Purpose["Session memory"]
        M_Store["Local cache with eviction"]
    end

    Medium -->|Promoted on persistence| Slow

    subgraph Slow["Slow Tier"]
        S_TTL["TTL: 1 day"]
        S_Purpose["Cross-session learning"]
        S_Store["Persistent store (Postgres)"]
    end

    Slow -->|Promoted on pattern detection| Glacial

    subgraph Glacial["Glacial Tier"]
        G_TTL["TTL: 1 week"]
        G_Purpose["Long-term patterns"]
        G_Store["Persistent store (Postgres)"]
    end

    Glacial --> KM["Knowledge Mound (permanent)"]
```

## Data Flow Between Tiers

```mermaid
sequenceDiagram
    participant D as Debate Engine
    participant F as Fast (1 min)
    participant M as Medium (1 hr)
    participant S as Slow (1 day)
    participant G as Glacial (1 week)
    participant KM as Knowledge Mound

    D->>F: Write immediate context
    Note over F: Available during current round

    F->>M: Promote on repeated access
    Note over M: Available during current session

    M->>S: Promote on cross-session relevance
    Note over S: Available across sessions today

    S->>G: Promote on pattern detection
    Note over G: Available for long-term learning

    G->>KM: Promote to permanent knowledge
    Note over KM: Federated organizational knowledge

    D->>F: Read (cache hit, fastest)
    D->>M: Read (fallback)
    D->>S: Read (fallback)
    D->>G: Read (fallback, slowest)
```

## Tier Summary

| Tier | TTL | Storage | Purpose | Access Pattern |
|------|-----|---------|---------|----------------|
| Fast | 1 min | In-memory | Current round context | Hot path, every read |
| Medium | 1 hr | Local cache | Session continuity | Within-session lookups |
| Slow | 1 day | Postgres | Cross-session learning | Start-of-session load |
| Glacial | 1 week | Postgres | Long-term patterns | Background analysis |

## Promotion Criteria

Entries move to a slower (longer-lived) tier when they demonstrate lasting value:

- **Fast to Medium** -- Entry is accessed more than once within its TTL.
- **Medium to Slow** -- Entry is referenced across multiple debate rounds or sessions.
- **Slow to Glacial** -- Pattern detection identifies the entry as part of a recurring theme.
- **Glacial to Knowledge Mound** -- Entry achieves sufficient confidence and is validated.

## Memory Coordinator

The `MemoryCoordinator` (`memory/coordinator.py`) provides atomic cross-system
writes when `enable_coordinated_writes` is active. This ensures that a debate
outcome is written to all relevant tiers and subsystems (memory, consensus store,
critique store, Knowledge Mound) as a single logical operation.

```mermaid
flowchart LR
    Coordinator["Memory Coordinator"]
    Coordinator --> Fast
    Coordinator --> Medium
    Coordinator --> Slow
    Coordinator --> Glacial
    Coordinator --> ConsensusStore["Consensus Store"]
    Coordinator --> CritiqueStore["Critique Store"]
    Coordinator --> KnowledgeMound["Knowledge Mound"]

    style Coordinator fill:#e8f0fe,stroke:#4285f4
```

## Cross-Debate Memory

When `enable_cross_debate_memory` is set, the system injects institutional
knowledge from previous debates into new ones. This pulls from the Slow and
Glacial tiers to seed context, allowing agents to build on prior decisions
rather than starting from scratch.

## Related Modules

| Module | Path | Role |
|--------|------|------|
| ContinuumMemory | `aragora/memory/continuum.py` | Four-tier implementation |
| MemoryCoordinator | `aragora/memory/coordinator.py` | Atomic writes |
| ConsensusMemory | `aragora/memory/consensus.py` | Historical outcomes |
| MemoryManager | `aragora/debate/memory_manager.py` | Debate-level coordination |
