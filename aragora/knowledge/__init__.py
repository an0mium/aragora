"""
Knowledge Base module for enterprise document auditing.

Provides:
- FactStore: SQLite-based storage for agreed-upon facts
- DatasetQueryEngine: Natural language query interface
- WeaviateEmbeddingService: Vector search integration
- FactExtractor: Multi-agent fact extraction from documents

The knowledge base builds on top of the Evidence system to create
a queryable layer of verified facts that have been agreed upon
through multi-agent debate and Byzantine consensus.

Usage:
    from aragora.knowledge import FactStore, Fact, ValidationStatus

    store = FactStore()
    fact = await store.add_fact(
        statement="Contract expires on 2025-12-31",
        evidence_ids=["ev_123", "ev_456"],
        source_documents=["doc_789"],
    )

    # Query facts
    facts = await store.query_facts(
        query="contract expiration",
        workspace_id="ws_abc",
    )
"""

from aragora.knowledge.types import (
    Fact,
    FactFilters,
    FactRelation,
    FactRelationType,
    QueryResult,
    ValidationStatus,
    VerificationResult,
)
from aragora.knowledge.fact_store import FactStore, InMemoryFactStore
from aragora.knowledge.embeddings import (
    ChunkMatch,
    EmbeddingConfig,
    InMemoryEmbeddingService,
    WeaviateEmbeddingService,
)
from aragora.knowledge.query_engine import (
    DatasetQueryEngine,
    QueryOptions,
    SimpleQueryEngine,
)
from aragora.knowledge.pipeline import (
    KnowledgePipeline,
    PipelineConfig,
    ProcessingResult,
    create_pipeline,
)
from aragora.knowledge.fact_extractor import (
    ExtractionConfig,
    ExtractionResult,
    ExtractedFact,
    FactExtractor,
    create_fact_extractor,
)

__all__ = [
    # Core types
    "Fact",
    "FactFilters",
    "FactRelation",
    "FactRelationType",
    "QueryResult",
    "ValidationStatus",
    "VerificationResult",
    # Stores
    "FactStore",
    "InMemoryFactStore",
    # Embeddings
    "ChunkMatch",
    "EmbeddingConfig",
    "InMemoryEmbeddingService",
    "WeaviateEmbeddingService",
    # Query Engine
    "DatasetQueryEngine",
    "QueryOptions",
    "SimpleQueryEngine",
    # Pipeline
    "KnowledgePipeline",
    "PipelineConfig",
    "ProcessingResult",
    "create_pipeline",
    # Fact Extractor
    "ExtractionConfig",
    "ExtractionResult",
    "ExtractedFact",
    "FactExtractor",
    "create_fact_extractor",
]
