# Knowledge Mound

Unified enterprise knowledge storage implementing the "termite mound" architecture where all agents contribute to and query from a shared knowledge superstructure.

## Overview

The Knowledge Mound system provides:

- **Unified API** over multiple storage backends (SQLite, PostgreSQL, Redis)
- **Cross-System Queries** across ContinuumMemory, ConsensusMemory, FactStore, EvidenceStore
- **Provenance Tracking** for audit and compliance
- **Staleness Detection** with automatic revalidation scheduling
- **Culture Accumulation** for organizational learning
- **Multi-Tenant Workspace Isolation**

## Architecture

```
                    ┌───────────────────────────────────────────┐
                    │             Knowledge Mound               │
                    │         (Unified Knowledge Facade)        │
                    └───────────────┬───────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐          ┌───────────────┐          ┌───────────────┐
│ Meta Store    │          │ Vector Store  │          │ Redis Cache   │
│ (SQLite/PG)   │          │ (Weaviate)    │          │               │
└───────────────┘          └───────────────┘          └───────────────┘
        │
        ├── ContinuumMemory (multi-tier temporal memory)
        ├── ConsensusMemory (debate outcomes)
        ├── FactStore (verified facts)
        ├── EvidenceStore (supporting evidence)
        └── CritiqueStore (critique patterns)
```

## Quick Start

### Basic Usage (SQLite)

```python
from aragora.knowledge.mound import KnowledgeMound, IngestionRequest, KnowledgeSource

# Create mound with default SQLite backend
mound = KnowledgeMound(workspace_id="my_team")
await mound.initialize()

# Store knowledge
result = await mound.store(IngestionRequest(
    content="Contracts require 90-day notice periods",
    source_type=KnowledgeSource.DEBATE,
    debate_id="debate_123",
    workspace_id="my_team",
    confidence=0.95,
))

# Query semantically
results = await mound.query("contract notice requirements", limit=10)

# Check staleness
stale = await mound.get_stale_knowledge(threshold=0.7)

# Get culture profile
culture = await mound.get_culture_profile()
```

### Production Usage (PostgreSQL + Redis)

```python
from aragora.knowledge.mound import KnowledgeMound, MoundConfig, MoundBackend

config = MoundConfig(
    backend=MoundBackend.HYBRID,
    postgres_url="postgresql://user:pass@host/db",
    redis_url="redis://localhost:6379",
    weaviate_url="http://localhost:8080",
    enable_staleness_detection=True,
    enable_culture_accumulator=True,
)

mound = KnowledgeMound(config, workspace_id="enterprise")
await mound.initialize()
```

---

## Core Components

### 1. Knowledge Mound Facade

The main entry point providing a unified API across all storage backends.

```python
from aragora.knowledge.mound import KnowledgeMound

mound = KnowledgeMound(workspace_id="default")
await mound.initialize()

# Store knowledge
result = await mound.store(ingestion_request)

# Query knowledge
results = await mound.query("query text", filters=QueryFilters(...))

# Get statistics
stats = await mound.get_stats()

# Sync with connected memory systems
sync_result = await mound.sync_all()
```

### 2. Semantic Store

Provides mandatory embedding-based semantic search.

```python
from aragora.knowledge.mound import SemanticStore

semantic = SemanticStore(embedding_model="text-embedding-3-small")
await semantic.index(content="...", node_id="node_123")
results = await semantic.search("query", limit=10)
```

### 3. Knowledge Graph Store

Manages relationships and lineage between knowledge nodes.

```python
from aragora.knowledge.mound import KnowledgeGraphStore, GraphLink

graph = KnowledgeGraphStore()
await graph.add_link(GraphLink(
    source_id="fact_1",
    target_id="fact_2",
    relationship="SUPPORTS",
    weight=0.9,
))
lineage = await graph.get_lineage("node_id")
```

### 4. Domain Taxonomy

Hierarchical organization of knowledge by domain.

```python
from aragora.knowledge.mound import DomainTaxonomy, DEFAULT_TAXONOMY

taxonomy = DomainTaxonomy(DEFAULT_TAXONOMY)
domain = taxonomy.classify("This contract requires HIPAA compliance")
# Returns: "legal/compliance/healthcare"
```

### 5. Meta-Learner

Cross-memory optimization and tier balancing.

```python
from aragora.knowledge.mound import KnowledgeMoundMetaLearner

learner = KnowledgeMoundMetaLearner(mound)
metrics = await learner.compute_retrieval_metrics()
recommendations = await learner.suggest_tier_optimizations()
coalesce_result = await learner.coalesce_duplicates()
```

---

## Configuration

### MoundConfig Options

| Option | Default | Description |
|--------|---------|-------------|
| `backend` | `SQLITE` | Storage backend: `SQLITE`, `POSTGRES`, `HYBRID` |
| `postgres_url` | `None` | PostgreSQL connection URL |
| `postgres_pool_size` | `10` | Connection pool size |
| `redis_url` | `None` | Redis connection URL |
| `redis_cache_ttl` | `300` | Query cache TTL (seconds) |
| `weaviate_url` | `None` | Weaviate vector store URL |
| `enable_staleness_detection` | `True` | Track knowledge freshness |
| `enable_culture_accumulator` | `True` | Track organizational patterns |
| `enable_auto_revalidation` | `False` | Auto-schedule stale knowledge review |
| `enable_deduplication` | `True` | Deduplicate similar content |
| `enable_provenance_tracking` | `True` | Track knowledge sources |
| `default_workspace_id` | `"default"` | Default multi-tenant workspace |
| `staleness_age_threshold` | `7 days` | Age after which knowledge is stale |

### Backend Options

| Backend | Use Case | Requirements |
|---------|----------|--------------|
| `SQLITE` | Development, testing, single-node | None |
| `POSTGRES` | Production, multi-node | PostgreSQL 14+ |
| `HYBRID` | Production with caching | PostgreSQL + Redis |

---

## Knowledge Types

### KnowledgeSource

| Source | Description |
|--------|-------------|
| `DEBATE` | From debate consensus or agent statements |
| `DOCUMENT` | From ingested documents |
| `FACT` | Extracted and verified facts |
| `EVIDENCE` | Supporting evidence |
| `USER` | User-provided knowledge |
| `EXTERNAL` | External API or system |

### Node Types

| Type | Description |
|------|-------------|
| `fact` | Verified factual statement |
| `claim` | Unverified claim |
| `memory` | Agent memory |
| `evidence` | Supporting evidence |
| `consensus` | Debate consensus |
| `entity` | Named entity |

### Relationship Types

| Relationship | Description |
|--------------|-------------|
| `SUPPORTS` | Source supports target claim |
| `CONTRADICTS` | Source contradicts target |
| `DERIVED_FROM` | Source derived from target |
| `RELATED_TO` | General relationship |
| `SUPERSEDES` | Source supersedes target |
| `CITES` | Source cites target |

---

## Staleness Detection

The Knowledge Mound tracks knowledge freshness and schedules revalidation.

```python
# Check for stale knowledge
stale_items = await mound.get_stale_knowledge(threshold=0.7)

for item in stale_items:
    print(f"{item.node_id}: {item.reason} (staleness: {item.staleness_score})")
```

### Staleness Reasons

| Reason | Description |
|--------|-------------|
| `AGE` | Knowledge exceeds age threshold |
| `CONTRADICTION` | New knowledge contradicts this item |
| `NEW_EVIDENCE` | New evidence affects validity |
| `CONSENSUS_CHANGE` | Debate consensus has changed |
| `SCHEDULED` | Scheduled for periodic review |
| `MANUAL` | Manually marked for review |

---

## Culture Accumulation

Tracks organizational patterns and preferences over time.

```python
# Get organization culture profile
culture = await mound.get_culture_profile()

print(f"Decision Style: {culture.decision_style}")
print(f"Risk Tolerance: {culture.risk_tolerance}")
print(f"Top Domains: {culture.domain_expertise}")
print(f"Agent Preferences: {culture.agent_preferences}")
```

### Culture Pattern Types

| Pattern | Description |
|---------|-------------|
| `DECISION_STYLE` | How decisions are typically made |
| `RISK_TOLERANCE` | Organization's risk appetite |
| `DOMAIN_EXPERTISE` | Areas of frequent expertise |
| `AGENT_PREFERENCES` | Preferred agents by domain |
| `DEBATE_DYNAMICS` | Typical debate patterns |
| `RESOLUTION_PATTERNS` | How conflicts are resolved |

---

## API Endpoints

### Facts API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/knowledge/query` | Natural language query |
| GET | `/api/knowledge/facts` | List facts with filtering |
| GET | `/api/knowledge/facts/:id` | Get specific fact |
| POST | `/api/knowledge/facts` | Add a new fact |
| PUT | `/api/knowledge/facts/:id` | Update a fact |
| DELETE | `/api/knowledge/facts/:id` | Delete a fact |
| POST | `/api/knowledge/facts/:id/verify` | Verify fact with agents |
| GET | `/api/knowledge/facts/:id/contradictions` | Get contradicting facts |
| GET | `/api/knowledge/facts/:id/relations` | Get fact relations |
| POST | `/api/knowledge/facts/relations` | Add relation between facts |
| GET | `/api/knowledge/search` | Search chunks via embeddings |
| GET | `/api/knowledge/stats` | Get knowledge base statistics |

### Knowledge Mound API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/knowledge/mound/query` | Semantic query |
| POST | `/api/knowledge/mound/nodes` | Add knowledge node |
| GET | `/api/knowledge/mound/nodes/:id` | Get specific node |
| GET | `/api/knowledge/mound/nodes` | List/filter nodes |
| GET | `/api/knowledge/mound/nodes/:id/relationships` | Get node relationships |
| POST | `/api/knowledge/mound/relationships` | Add relationship |
| GET | `/api/knowledge/mound/graph/:id` | Graph traversal from node |
| GET | `/api/knowledge/mound/stats` | Mound statistics |
| POST | `/api/knowledge/mound/index/repository` | Index a repository |

### Example Requests

**Query Knowledge:**
```bash
curl -X POST http://localhost:8080/api/knowledge/mound/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "contract notice requirements",
    "workspace_id": "enterprise",
    "limit": 10,
    "filters": {
      "source_types": ["debate", "document"],
      "min_confidence": 0.7
    }
  }'
```

**Add Knowledge Node:**
```bash
curl -X POST http://localhost:8080/api/knowledge/mound/nodes \
  -H "Content-Type: application/json" \
  -d '{
    "content": "All contracts require 90-day notice periods",
    "workspace_id": "enterprise",
    "source_type": "debate",
    "debate_id": "debate_123",
    "confidence": 0.95,
    "topics": ["legal", "contracts"]
  }'
```

---

## CLI Commands

The `aragora knowledge` command provides CLI access to the Knowledge Mound.

```bash
# Query knowledge base
aragora knowledge query "What are the payment terms?"

# List facts
aragora knowledge facts --workspace default

# Search document chunks
aragora knowledge search "contract expiration"

# Process and ingest documents
aragora knowledge process document.pdf

# List processing jobs
aragora knowledge jobs
```

See [CLI Reference](CLI_REFERENCE.md#knowledge---knowledge-base-operations) for full command documentation.

---

## Integration with Memory Systems

The Knowledge Mound integrates with existing memory systems:

### ContinuumMemory Adapter

```python
from aragora.knowledge.mound.adapters import ContinuumAdapter
from aragora.memory.continuum import ContinuumMemory

continuum = ContinuumMemory()
adapter = ContinuumAdapter(mound, continuum)
await adapter.sync()  # Sync continuum memories to mound
```

### ConsensusMemory Adapter

```python
from aragora.knowledge.mound.adapters import ConsensusAdapter
from aragora.memory.consensus import ConsensusMemory

consensus = ConsensusMemory()
adapter = ConsensusAdapter(mound, consensus)
await adapter.sync()  # Sync consensus outcomes to mound
```

### CritiqueStore Adapter

```python
from aragora.knowledge.mound.adapters import CritiqueAdapter
from aragora.memory.store import CritiqueStore

critique = CritiqueStore()
adapter = CritiqueAdapter(mound, critique)
await adapter.sync()  # Sync critique patterns to mound
```

---

## Vertical Knowledge

Domain-specific knowledge bases for vertical specialists.

```python
from aragora.knowledge.mound.verticals import VerticalKnowledgeStore

# Create vertical-specific store
legal_knowledge = VerticalKnowledgeStore(
    vertical_id="legal",
    mound=mound,
)

# Load compliance frameworks
await legal_knowledge.load_framework("GDPR")
await legal_knowledge.load_framework("CCPA")

# Query with vertical context
results = await legal_knowledge.query("data processing requirements")
```

### Available Vertical Knowledge

| Vertical | Frameworks | Domain Terms |
|----------|------------|--------------|
| `software` | OWASP, CWE | Security, architecture |
| `legal` | GDPR, CCPA, HIPAA | Contracts, compliance |
| `healthcare` | HIPAA, HITECH, FDA 21 CFR 11 | Clinical, PHI |
| `accounting` | SOX, GAAP, PCAOB | Financial, audit |
| `research` | IRB, CONSORT, PRISMA | Methodology, ethics |

---

## Vector Store Backends

### Weaviate (Production)

```python
config = MoundConfig(
    weaviate_url="http://localhost:8080",
    weaviate_collection="KnowledgeMound",
    weaviate_api_key="your-api-key",
)
```

### Qdrant

```python
from aragora.knowledge.mound.vector_abstraction import QdrantStore

vector_store = QdrantStore(
    url="http://localhost:6333",
    collection_name="knowledge_mound",
)
```

### ChromaDB

```python
from aragora.knowledge.mound.vector_abstraction import ChromaStore

vector_store = ChromaStore(
    persist_directory="./chroma_data",
    collection_name="knowledge_mound",
)
```

### In-Memory (Testing)

```python
from aragora.knowledge.mound.vector_abstraction import MemoryStore

vector_store = MemoryStore()  # Ephemeral, for testing
```

---

## Visibility & Access Control

The Knowledge Mound supports fine-grained visibility levels for controlling who can access knowledge items.

### Visibility Levels

| Level | Description | Use Case |
|-------|-------------|----------|
| `private` | Creator and explicit grantees only | Personal notes, drafts |
| `workspace` | All members of the workspace (default) | Team knowledge |
| `organization` | All members of the organization | Cross-team standards |
| `public` | Anyone with access to the system | Open documentation |
| `system` | Global verified facts (admin only) | Canonical knowledge |

### Setting Visibility

```python
from aragora.knowledge.mound import VisibilityLevel

# Set visibility when storing
result = await mound.store(IngestionRequest(
    content="Internal API guidelines",
    workspace_id="engineering",
    visibility=VisibilityLevel.ORGANIZATION,
))

# Update visibility of existing item
await mound.set_visibility(
    item_id="node_123",
    visibility=VisibilityLevel.WORKSPACE,
    set_by="user_456",
)
```

### API Endpoints for Visibility

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/knowledge/mound/nodes/:id/visibility` | Get item visibility |
| PUT | `/api/knowledge/mound/nodes/:id/visibility` | Set item visibility |
| GET | `/api/knowledge/mound/nodes/:id/grants` | List access grants |
| POST | `/api/knowledge/mound/nodes/:id/grants` | Add access grant |
| DELETE | `/api/knowledge/mound/nodes/:id/grants/:grantId` | Revoke grant |

---

## Cross-Workspace Sharing

Share knowledge items with other workspaces or users using explicit access grants.

### Grant Types

| Type | Description |
|------|-------------|
| `user` | Grant to specific user |
| `workspace` | Grant to all workspace members |
| `organization` | Grant to all org members |
| `role` | Grant to users with specific role |

### Sharing Knowledge

```python
from aragora.knowledge.mound import AccessGrant, AccessGrantType

# Share with another workspace
grant = await mound.share_with_workspace(
    item_id="node_123",
    from_workspace_id="team_a",
    to_workspace_id="team_b",
    shared_by="user_456",
    permissions=["read"],
    expires_at=datetime.now() + timedelta(days=30),
)

# Get items shared with me
shared_items = await mound.get_shared_with_me(
    workspace_id="team_b",
    limit=50,
)

# Revoke a share
await mound.revoke_share(
    item_id="node_123",
    grantee_id="team_b",
    revoked_by="user_456",
)
```

### API Endpoints for Sharing

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/knowledge/mound/share` | Share item with workspace/user |
| GET | `/api/knowledge/mound/shared-with-me` | Items shared with current user |
| DELETE | `/api/knowledge/mound/share/:itemId/:granteeId` | Revoke share |

---

## Global Knowledge

The system workspace (`__system__`) contains verified facts accessible to all users. Global knowledge is ideal for canonical information that should be universally available.

### Storing Global Knowledge

```python
# Store a verified fact (admin only)
fact_id = await mound.store_verified_fact(
    content="HIPAA requires 6-year retention of PHI records",
    source="45 CFR 164.530(j)",
    confidence=1.0,
    verified_by="admin_user",
)

# Query global knowledge
results = await mound.query_global_knowledge(
    query="HIPAA retention requirements",
    limit=10,
)

# Promote workspace knowledge to global (admin only)
global_id = await mound.promote_to_global(
    item_id="node_123",
    workspace_id="legal_team",
    promoted_by="admin_user",
    reason="Verified through external audit",
)
```

### API Endpoints for Global Knowledge

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/knowledge/mound/global/facts` | Store verified fact (admin) |
| GET | `/api/knowledge/mound/global/query` | Query global knowledge |
| POST | `/api/knowledge/mound/global/promote` | Promote to global (admin) |

---

## Knowledge Federation

Synchronize knowledge across multiple Aragora deployments or regions.

### Federation Modes

| Mode | Description |
|------|-------------|
| `push` | Send knowledge to remote region |
| `pull` | Receive knowledge from remote region |
| `bidirectional` | Two-way sync |
| `none` | Federation disabled |

### Sync Scopes

| Scope | Description |
|-------|-------------|
| `full` | Complete knowledge items |
| `metadata` | Metadata only (titles, dates, confidence) |
| `summary` | AI-generated summaries |

### Configuring Federation

```python
from aragora.knowledge.mound import FederationPolicy, SharingScope

# Register a federated region (admin only)
region = await mound.register_federated_region(
    region_id="us-west",
    endpoint_url="https://us-west.aragora.example.com/api",
    api_key="federated-api-key",
    sync_policy=FederationPolicy(
        mode="bidirectional",
        scope=SharingScope.SUMMARY,
    ),
)

# Sync to a region
result = await mound.sync_to_region(
    region_id="us-west",
    workspace_id="enterprise",
    since=datetime.now() - timedelta(hours=24),
    scope=SharingScope.SUMMARY,
)

# Pull from a region
result = await mound.pull_from_region(
    region_id="us-west",
    workspace_id="enterprise",
)

# Check federation status
status = await mound.get_federation_status()
for region in status.regions:
    print(f"{region.name}: {region.health} (last sync: {region.last_sync_at})")
```

### API Endpoints for Federation

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/knowledge/mound/federation/regions` | Register region (admin) |
| GET | `/api/knowledge/mound/federation/regions` | List federated regions |
| POST | `/api/knowledge/mound/federation/sync/push` | Push to region |
| POST | `/api/knowledge/mound/federation/sync/pull` | Pull from region |
| GET | `/api/knowledge/mound/federation/status` | Federation health status |
| PUT | `/api/knowledge/mound/federation/regions/:id` | Update region settings |
| DELETE | `/api/knowledge/mound/federation/regions/:id` | Remove region |

---

## Security & Compliance

### Document-Level Access Control

```python
# Workspace isolation
result = await mound.query(
    "sensitive data",
    workspace_id="team_a",  # Only returns team_a knowledge
)

# Visibility-aware queries automatically filter by access
result = await mound.query(
    "contract terms",
    workspace_id="team_a",
    actor_id="user_123",  # Filters by user's access grants
)
```

### Audit Logging

```python
# All retrievals are logged
# Check audit logs for compliance
audit_logs = await mound.get_audit_log(
    start_time=datetime.now() - timedelta(days=7),
    action_types=["query", "store"],
)
```

### GDPR/CCPA Compliant Deletion

```python
# Right to erasure
await mound.delete_user_knowledge(user_id="user_123")

# Document-level deletion
await mound.delete_document_knowledge(document_id="doc_456")
```

---

## Best Practices

1. **Use Workspaces for Isolation** - Always specify `workspace_id` for multi-tenant deployments
2. **Enable Provenance Tracking** - Keep `enable_provenance_tracking=True` for audit trails
3. **Configure Staleness Thresholds** - Adjust `staleness_age_threshold` based on domain freshness requirements
4. **Sync Regularly** - Call `mound.sync_all()` periodically to keep knowledge current
5. **Monitor Culture Patterns** - Use culture profiles to understand organizational learning

---

## See Also

- [ADR-014: Knowledge Mound Architecture](ADR/014-knowledge-mound-architecture.md)
- [Verticals Documentation](VERTICALS.md)
- [Memory Systems](MEMORY.md)
- [Evidence Collection](EVIDENCE.md)
- [API Reference](API_REFERENCE.md)
