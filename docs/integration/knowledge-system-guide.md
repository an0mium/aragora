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

## API Reference

Core modules:
- `aragora/knowledge/pipeline.py` - Document processing pipeline
- `aragora/knowledge/mound/facade.py` - Knowledge Mound API
- `aragora/knowledge/mound/fact_registry.py` - Fact management
- `aragora/knowledge/mound/vector_abstraction/` - Vector backends
- `aragora/knowledge/mound/verticals/` - Vertical knowledge modules
