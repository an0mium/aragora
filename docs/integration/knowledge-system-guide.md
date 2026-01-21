# Knowledge System Guide

This guide covers the Knowledge Mound system for document ingestion, semantic search, and fact management.

## Overview

The Knowledge System provides:
- **Document Pipeline**: Ingest, chunk, and embed documents
- **Vector Storage**: Semantic search across knowledge
- **Fact Registry**: Extract and track facts with staleness detection
- **Vertical Knowledge**: Industry-specific knowledge bases

## Quick Start

### Basic Document Processing

```python
from aragora.knowledge import KnowledgePipeline, PipelineConfig

# Configure pipeline
config = PipelineConfig(
    workspace_id="my-workspace",
    use_weaviate=False,  # Use local storage
    extract_facts=True,
)

# Create and start pipeline
pipeline = KnowledgePipeline(config=config)
await pipeline.start()

# Process a document
result = await pipeline.process_document(
    content=document_bytes,
    filename="contract.pdf",
    tags=["legal", "contracts"],
)

print(f"Processed: {result.chunk_count} chunks created")

# Search
results = await pipeline.search("indemnification clause", limit=5)

await pipeline.stop()
```

### Knowledge Mound

```python
from aragora.knowledge.mound import KnowledgeMound, MoundConfig, MoundBackend

config = MoundConfig(
    backend=MoundBackend.SQLITE,
    sqlite_path="knowledge.db",
    enable_deduplication=True,
)

async with KnowledgeMound(config=config).session() as mound:
    # Add knowledge
    node_id = await mound.add(
        content="API keys should never be committed to version control",
        metadata={"vertical": "software", "confidence": 0.95},
        node_type="fact",
    )

    # Query
    result = await mound.query("security best practices", limit=10)

    for item in result.items:
        print(f"- {item.content[:100]}...")
```

## Document Pipeline

### Supported Formats

- PDF (`.pdf`)
- Word (`.docx`, `.doc`)
- Markdown (`.md`)
- Plain text (`.txt`)
- HTML (`.html`)
- JSON (`.json`)
- CSV (`.csv`)

### Pipeline Configuration

```python
config = PipelineConfig(
    workspace_id="workspace-001",

    # Vector storage
    use_weaviate=True,  # Use Weaviate (production)
    weaviate_url="http://localhost:8080",

    # Chunking
    chunk_size=512,
    chunk_overlap=50,

    # Embedding
    embedding_model="text-embedding-3-small",

    # Features
    extract_facts=True,
    detect_entities=True,

    # Storage
    fact_db_path="facts.db",
)
```

### Batch Processing

```python
files = [
    (doc1_bytes, "doc1.pdf"),
    (doc2_bytes, "doc2.pdf"),
    (doc3_bytes, "doc3.md"),
]

results = await pipeline.process_batch(files)

for result in results:
    if result.success:
        print(f"Processed {result.filename}: {result.chunk_count} chunks")
    else:
        print(f"Failed {result.filename}: {result.error}")
```

### Progress Tracking

```python
def on_progress(doc_id: str, progress: float, message: str):
    print(f"[{doc_id}] {progress*100:.0f}% - {message}")

pipeline.set_progress_callback(on_progress)
await pipeline.process_document(content, "large_doc.pdf")
```

## Vector Storage

### Backend Options

#### In-Memory (Development)

```python
from aragora.knowledge.mound.vector_abstraction import (
    VectorStoreConfig,
    VectorBackend,
    VectorStoreFactory,
)

config = VectorStoreConfig(
    backend=VectorBackend.MEMORY,
    collection_name="test_knowledge",
    embedding_dimensions=1536,
)

store = VectorStoreFactory.create(config)
await store.connect()
```

#### Weaviate (Production)

```python
config = VectorStoreConfig(
    backend=VectorBackend.WEAVIATE,
    weaviate_url="http://weaviate:8080",
    collection_name="knowledge_mound",
    embedding_dimensions=1536,
)
```

### Vector Operations

```python
# Upsert
await store.upsert(
    id="doc-001",
    embedding=[0.1, 0.2, ...],  # 1536 dimensions
    content="Document content here",
    metadata={"source": "upload", "vertical": "legal"},
)

# Batch upsert
items = [
    {"id": "doc-002", "embedding": [...], "content": "..."},
    {"id": "doc-003", "embedding": [...], "content": "..."},
]
await store.upsert_batch(items)

# Search
results = await store.search(
    embedding=query_embedding,
    limit=10,
    filters={"vertical": "legal"},
    min_score=0.7,
)

# Hybrid search (vector + keyword)
results = await store.hybrid_search(
    query="contract termination",
    embedding=query_embedding,
    limit=10,
    alpha=0.5,  # Balance between keyword and semantic
)
```

### Namespace Isolation

```python
# Store in namespace
await store.upsert(
    id="doc-001",
    embedding=[...],
    content="...",
    namespace="tenant-123",
)

# Search within namespace
results = await store.search(
    embedding=[...],
    namespace="tenant-123",
)
```

## Fact Registry

### Registering Facts

```python
from aragora.knowledge.mound import FactRegistry

registry = FactRegistry(
    vector_store=store,
    embedding_service=embedding_service,
)
await registry.initialize()

# Register a fact
fact = await registry.register(
    statement="HIPAA requires encryption of PHI at rest",
    vertical="healthcare",
    category="compliance",
    confidence=0.95,
    workspace_id="workspace-001",
)
```

### Fact Staleness

Facts have time-based confidence decay:

```python
# Check fact staleness
print(f"Staleness: {fact.staleness_days} days")
print(f"Current confidence: {fact.current_confidence}")
print(f"Needs reverification: {fact.needs_reverification}")

# Refresh a fact
fact.refresh(new_confidence=0.98)

# Get stale facts for reverification
stale_facts = await registry.get_stale_facts()
```

### Vertical-Specific Decay Rates

Different categories have different decay rates:

| Vertical | Category | Decay Rate |
|----------|----------|------------|
| Software | Vulnerability | 0.05/day |
| Software | Best practice | 0.01/day |
| Legal | Regulation | 0.001/day |
| Legal | Case law | 0.005/day |
| Healthcare | Treatment | 0.02/day |

### Querying Facts

```python
results = await registry.query(
    query="encryption requirements",
    verticals=["healthcare", "software"],
    workspace_id="workspace-001",
    min_confidence=0.7,
)

for fact in results:
    print(f"[{fact.vertical}] {fact.statement}")
    print(f"  Confidence: {fact.current_confidence:.2f}")
```

## Vertical Knowledge

### Available Verticals

- Software
- Legal
- Healthcare
- Accounting
- Research
- Regulatory

### Using Vertical Knowledge

```python
from aragora.knowledge.mound.verticals import VerticalRegistry

# Get a vertical module
software = VerticalRegistry.get("software")

# Check capabilities
caps = VerticalRegistry.get_capabilities("software")
# {'pattern_detection': True, 'compliance_check': True, ...}

# List all verticals
verticals = VerticalRegistry.list_all()
```

### Vertical-Specific Processing

```python
# Process with vertical context
node_id = await mound.add(
    content="Implement rate limiting for API endpoints",
    metadata={
        "vertical": "software",
        "category": "security",
    },
)

# Query with vertical filter
result = await mound.query(
    query="API security",
    filters={"vertical": "software"},
)
```

## Integration with Debates

### Knowledge-Augmented Debates

```python
from aragora import Arena, Environment

# Retrieve relevant knowledge
knowledge = await mound.query(
    query=debate_topic,
    filters={"vertical": "legal"},
    limit=10,
)

# Include in debate context
env = Environment(
    task=debate_topic,
    context={
        "knowledge": [k.content for k in knowledge.items],
    },
)

result = await arena.run()
```

### Workflow Integration

```python
from aragora.workflow import StepDefinition

# Knowledge pipeline step
StepDefinition(
    id="gather_knowledge",
    step_type="knowledge_pipeline",
    config={
        "action": "search",
        "query": "{{context.topic}}",
        "verticals": ["legal"],
        "limit": 10,
    },
    next_steps=["debate"],
)
```

## Security & Compliance

### Document Access Control

```python
node_id = await mound.add(
    content="Confidential information",
    metadata={
        "access_level": "confidential",
        "allowed_roles": ["legal", "executive"],
    },
)

# Query respects access control
result = await mound.query(
    query="...",
    filters={"access_level": {"$in": user_access_levels}},
)
```

### GDPR Compliant Deletion

```python
# Delete with archive
await mound.delete(node_id, archive=True)

# Hard delete (GDPR erasure)
await mound.delete(node_id, archive=False, hard_delete=True)
```

### Audit Logging

All retrievals are logged for audit:

```python
# Logs include:
# - User ID
# - Query
# - Retrieved document IDs
# - Timestamp
# - Access reason (if provided)
```

## Bidirectional Integration Architecture

The Knowledge Mound integrates bidirectionally with all major subsystems, enabling organizational learning across debates, memory, beliefs, and agent performance.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          KNOWLEDGE MOUND (Central Hub)                      │
│                                                                             │
│   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│   │ Beliefs    │  │ Insights   │  │ Expertise  │  │ Compression│           │
│   │ & Cruxes   │  │ & Flips    │  │ Profiles   │  │ Patterns   │           │
│   └────────────┘  └────────────┘  └────────────┘  └────────────┘           │
└───────┬───────────────┬───────────────┬───────────────┬────────────────────┘
        │               │               │               │
        ▼               ▼               ▼               ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│ BeliefAdapter │ │InsightsAdapter│ │RankingAdapter │ │  RlmAdapter   │
│               │ │               │ │               │ │               │
│ - beliefs     │ │ - insights    │ │ - expertise   │ │ - patterns    │
│ - cruxes      │ │ - flips       │ │ - domains     │ │ - priorities  │
│ - provenance  │ │ - patterns    │ │ - history     │ │ - markers     │
└───────┬───────┘ └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
        │               │               │               │
        ▼               ▼               ▼               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CrossSubscriberManager (Event Router)                   │
│                                                                             │
│  Inbound (→KM)           Outbound (KM→)                                     │
│  ┌──────────────────┐    ┌──────────────────┐                               │
│  │ memory_to_mound  │    │ mound_to_memory  │                               │
│  │ belief_to_mound  │    │ mound_to_belief  │                               │
│  │ rlm_to_mound     │    │ mound_to_rlm     │                               │
│  │ elo_to_mound     │    │ mound_to_elo     │                               │
│  │ insight_to_mound │    │ mound_to_trickster│                              │
│  │ consensus_to_    │    │ culture_to_debate │                              │
│  │   mound          │    │ staleness_to_    │                               │
│  └──────────────────┘    │   debate         │                               │
│                          └──────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────┘
        │               │               │               │
        ▼               ▼               ▼               ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│BeliefNetwork  │ │ Trickster/    │ │  EloSystem/   │ │     RLM       │
│               │ │ Insights      │ │  TeamSelector │ │   Compressor  │
└───────────────┘ └───────────────┘ └───────────────┘ └───────────────┘
        │               │               │               │
        └───────────────┴───────────────┴───────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │    ContinuumMemory    │
                    │  (also bidirectional) │
                    └───────────────────────┘
```

### Data Flow Summary

| Integration | Inbound (→KM) | Outbound (KM→) |
|-------------|---------------|----------------|
| Memory | High-importance memories (≥0.7) | Pre-warm cache on KM query |
| Beliefs | Converged beliefs, cruxes | Initialize debate priors |
| RLM | High-value compression patterns | Update compression priorities |
| ELO | Significant expertise changes | Domain expert recommendations |
| Insights | High-confidence insights, ALL flips | Agent flip history for Trickster |
| Consensus | Full consensus with dissent | N/A (stored for future queries) |
| Culture | Emerging patterns | Inform debate protocol |
| Staleness | N/A | Warn active debates |
| Provenance | Verified chains | Historical verification |

### Cross-Subscriber Handlers

```python
from aragora.events.cross_subscribers import get_cross_subscriber_manager
from aragora.events.types import StreamEventType

# Get the manager
manager = get_cross_subscriber_manager()

# Register a custom handler
@manager.subscribe(StreamEventType.CONSENSUS)
def on_consensus(event):
    # Process consensus event
    print(f"Consensus reached: {event.data.get('topic')}")

# Check handler stats
stats = manager.get_stats()
print(f"Events processed: {stats['consensus_to_mound'].events_processed}")
```

### Observability Endpoints

```bash
# Get cross-pollination statistics
curl http://localhost:8080/api/cross-pollination/stats

# List all registered subscribers
curl http://localhost:8080/api/cross-pollination/subscribers

# Get Arena event bridge status
curl http://localhost:8080/api/cross-pollination/bridge
```

### Performance SLOs

| Operation | P50 | P90 | P99 | Timeout |
|-----------|-----|-----|-----|---------|
| KM Query | 50ms | 150ms | 500ms | 5s |
| KM Ingestion | 100ms | 300ms | 1000ms | 10s |
| Consensus Ingestion | 200ms | 500ms | 1500ms | 10s |
| Adapter Sync | 300ms | 800ms | 2000ms | 15s |
| Event Dispatch | 10ms | 50ms | 200ms | 5s |

### Enhanced Consensus Ingestion

Consensus events are enriched before storage:

1. **Dissent Tracking**: Dissenting views stored as separate nodes linked to consensus
2. **Evolution Detection**: Automatic detection of similar prior consensus (supersedes)
3. **Claim Linking**: Key claims stored with parent consensus reference
4. **Evidence Linking**: Supporting evidence attached to consensus node

```python
# Example consensus event data
{
    "debate_id": "debate_001",
    "consensus_reached": True,
    "topic": "API authentication strategy",
    "conclusion": "OAuth 2.0 with PKCE recommended",
    "strength": "strong",
    "dissents": [
        {
            "agent_id": "mistral",
            "type": "risk_warning",
            "content": "PKCE adds complexity for simple use cases",
            "acknowledged": True,
        }
    ],
    "supersedes": "old_consensus_auth_001",  # Optional
}
```

## Bidirectional Adapter Integration

The Knowledge Mound uses adapters to enable bidirectional data flow between subsystems:

### Available Adapters

| Adapter | Source System | Forward Flow | Reverse Flow |
|---------|--------------|--------------|--------------|
| `EvidenceAdapter` | EvidenceStore | Evidence → KM | KM → Similar evidence |
| `ContinuumAdapter` | ContinuumMemory | Memories → KM | KM → Similar memories |
| `ConsensusAdapter` | ConsensusMemory | Consensus → KM | KM → Past debates |
| `BeliefAdapter` | BeliefNetwork | Beliefs/cruxes → KM | KM → Prior beliefs |
| `InsightsAdapter` | InsightStore | Insights/flips → KM | KM → Pattern matching |
| `EloAdapter` | EloSystem | Ratings → KM | KM → Skill history |
| `CostAdapter` | CostTracker | Alerts → KM | KM → Cost patterns |

### Automatic Wiring

Adapters are automatically wired at server initialization:

```python
# In aragora/server/initialization.py
def init_continuum_memory(nomic_dir: Path):
    memory = ContinuumMemory(base_dir=str(nomic_dir))

    # Adapter is automatically created and wired
    adapter = ContinuumAdapter(continuum=memory, enable_dual_write=True)
    memory.set_km_adapter(adapter)

    return memory
```

### Manual Adapter Configuration

For custom setups, adapters can be manually configured:

```python
from aragora.memory.continuum import ContinuumMemory
from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

# Create memory system
memory = ContinuumMemory()

# Create and wire adapter with event callback
adapter = ContinuumAdapter(
    continuum=memory,
    enable_dual_write=True,
    event_callback=my_event_handler,
)
memory.set_km_adapter(adapter)

# Now memory operations sync bidirectionally with KM
```

### Reverse Flow Queries

Query KM from source systems for context before creating new data:

```python
# Check KM for similar memories before storing
similar = memory.query_km_for_similar(
    content="Error handling patterns",
    limit=5,
    min_similarity=0.7,
)

if similar:
    # Found existing knowledge - avoid duplication
    print(f"Found {len(similar)} similar entries in KM")
```

### Prometheus Metrics

All adapters emit Prometheus metrics for operation tracking:

```
# Operations total
aragora_km_operations_total{operation="search", status="success"}

# Operation latency
aragora_km_operation_latency_seconds{operation="search"}

# Adapter sync tracking
aragora_km_adapter_syncs_total{adapter="continuum", direction="forward", status="success"}
```

### Dashboard Events

Adapters emit WebSocket events for real-time dashboard updates:

- `km_adapter_forward_sync` - Data synced to KM
- `km_adapter_reverse_query` - Reverse flow query executed
- `km_adapter_validation` - KM validation feedback received

## API Reference

Core modules:
- `aragora/knowledge/pipeline.py` - Document processing pipeline
- `aragora/knowledge/mound/facade.py` - Knowledge Mound API
- `aragora/knowledge/mound/fact_registry.py` - Fact management
- `aragora/knowledge/mound/vector_abstraction/` - Vector backends
- `aragora/knowledge/mound/verticals/` - Vertical knowledge modules
- `aragora/knowledge/mound/adapters/` - Bidirectional adapters
- `aragora/events/cross_subscribers.py` - Cross-subsystem event handlers
- `aragora/config/performance_slos.py` - Performance SLO definitions
