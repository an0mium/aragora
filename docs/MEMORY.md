# Memory Systems Overview

Aragora’s control plane uses layered memory to keep robust decisionmaking grounded,
persistent, and improvable across debates, workflows, and channels.

## Core Components

- **ContinuumMemory**: Tiered memory with promotion/demotion rules and
  retention controls. (`aragora/memory/continuum.py`)
- **ConsensusMemory**: Stores consensus outcomes, dissent, and similar-topic
  retrieval. (`aragora/memory/consensus.py`)
- **MemoryStream**: Per-agent event timelines used for episodic recall.
  (`aragora/memory/streams.py`)
- **Knowledge Mound**: Unified knowledge facade across multiple stores,
  with provenance and staleness tracking. (`aragora/knowledge/mound/core.py`)

## Lifecycle at a Glance

1. Debate/workflow events are recorded into MemoryStream and ContinuumMemory.
2. Consensus results and receipts are recorded into ConsensusMemory.
3. The Knowledge Mound syncs across memory stores to provide unified query and
   governance controls.

## Storage and Backends

Memory defaults to local persistence for development. Production deployments
should use durable backends (PostgreSQL/Redis/Weaviate) via Knowledge Mound
configuration for multi-instance consistency.

## Related Documentation

- [Memory tiers](MEMORY_TIERS.md)
- [Memory strategy](MEMORY_STRATEGY.md)
- [Knowledge Mound](KNOWLEDGE_MOUND.md)
- [Memory analytics](MEMORY_ANALYTICS.md)
