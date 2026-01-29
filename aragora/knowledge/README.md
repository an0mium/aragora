# Knowledge Module

Unified enterprise knowledge management with semantic search and bidirectional adapters.

## Quick Start

```python
from aragora.knowledge import KnowledgeBridgeHub, KnowledgeMound

# Access via bridge hub
hub = KnowledgeBridgeHub()
await hub.sync_all()

# Direct mound access
mound = KnowledgeMound(config)
results = await mound.semantic_search("rate limiting patterns", limit=10)
```

## Key Components

| Component | Purpose |
|-----------|---------|
| `KnowledgeBridgeHub` | Central access for MetaLearner, Evidence, Pattern bridges |
| `KnowledgeMound` | Enterprise storage with semantic search |
| `AdapterFactory` | Auto-create adapters from Arena subsystems |

## Adapters

| Adapter | Purpose |
|---------|---------|
| `ContinuumAdapter` | Multi-tier memory sync |
| `ConsensusAdapter` | Debate outcomes with confidence decay |
| `CritiqueAdapter` | Critique patterns and feedback |
| `EvidenceAdapter` | External evidence with quality scores |
| `BeliefAdapter` | Belief network nodes and cruxes |
| `EloAdapter` | Agent rankings and calibration |
| `PulseAdapter` | Trending topics injection |
| `ControlPlaneAdapter` | Cross-workspace capabilities |
| `ReceiptAdapter` | Cryptographic audit trails |
| `CultureAdapter` | Organizational patterns via stigmergy |

## Architecture

```
knowledge/
├── bridges.py           # KnowledgeBridgeHub integration
├── mound/               # Core knowledge mound
│   ├── facade.py        # Unified API
│   ├── core.py          # Foundation
│   ├── types.py         # Data types
│   ├── adapters/        # 14 bidirectional adapters
│   │   ├── _base.py     # Adapter base class
│   │   ├── continuum.py # Memory adapter
│   │   ├── consensus.py # Debate outcomes
│   │   ├── evidence.py  # External evidence
│   │   └── ...          # 10+ more adapters
│   ├── api/             # CRUD and query operations
│   ├── ops/             # Enterprise operations
│   │   ├── staleness.py # Revalidation
│   │   ├── sync.py      # Bidirectional sync
│   │   ├── contradiction.py # Detection
│   │   └── ...          # 15+ more operations
│   └── resilience.py    # Retry, health, integrity
├── culture/             # Organizational learning
│   └── stigmergy.py     # Stigmergic patterns
└── verticals/           # Industry-specific models
```

## Key Patterns

- **Bidirectional Flow**: KM ↔ Source systems for feedback loops
- **Confidence Decay**: Time-based confidence reduction
- **Contradiction Detection**: Byzantine consensus validation
- **Dual-Write Migration**: Gradual system transitions
- **Semantic Search**: Vector-based retrieval with relevance scoring

## Related Documentation

- [CLAUDE.md](../../CLAUDE.md) - Project overview
- [docs/STATUS.md](../../docs/STATUS.md) - Feature status
