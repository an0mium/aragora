# ADR-014: Knowledge Mound Architecture

## Status
Accepted

## Context
Enterprise deployments require a unified knowledge management system that:
- Supports multiple document formats
- Enables semantic search across documents
- Integrates with vertical-specific knowledge bases
- Provides fact extraction and verification
- Maintains provenance and audit trails

## Decision
We implemented the **Knowledge Mound** system with the following architecture:

### Core Components

**Knowledge Pipeline** (`aragora/knowledge/pipeline.py`):
```
Document Ingestion -> Chunking -> Embedding -> Vector Storage -> Retrieval
```

Pipeline stages:
1. **Ingestion**: Multiple format support (PDF, DOCX, MD, etc.)
2. **Chunking**: Semantic-aware text splitting
3. **Embedding**: OpenAI/local embedding models
4. **Storage**: Weaviate/Pinecone/local vector store
5. **Retrieval**: Hybrid semantic + keyword search

### Vector Store Integration
Located in `aragora/documents/indexing/`:

**Weaviate Store** (`weaviate_store.py`):
- Primary production backend
- Supports hybrid search
- Multi-tenant isolation

**Local Store**:
- Development/testing
- SQLite + numpy for embeddings

### Fact Registry
Located in `aragora/knowledge/`:

**FactRegistry**:
- Extracted facts from documents
- Confidence scoring
- Source attribution

**VerticalKnowledge**:
- Industry-specific knowledge bases
- Pre-loaded compliance frameworks
- Domain terminology

### Integration Points

**Debate Integration**:
```python
# In debate context
knowledge = await knowledge_mound.retrieve(
    query=debate_topic,
    filters={"vertical": "legal"},
    limit=10
)
```

**Workflow Integration**:
- `knowledge_pipeline` node type in workflows
- Automatic document processing in workflows

### Security & Compliance
- Document-level access control
- Encryption at rest
- Audit logging for all retrievals
- GDPR/CCPA compliant deletion

## Consequences
**Positive:**
- Unified knowledge access across verticals
- Semantic search improves relevance
- Provenance enables audit trails
- Scalable vector storage

**Negative:**
- Vector store dependency (Weaviate/Pinecone)
- Embedding costs for large document sets
- Chunk size tuning required per use case
- Complex multi-tenant isolation

## References
- `aragora/knowledge/pipeline.py` - Main pipeline
- `aragora/documents/indexing/weaviate_store.py` - Vector store
- `aragora/knowledge/` - Knowledge module (40K lines)
- `docs/EVIDENCE.md` - Evidence/knowledge documentation
