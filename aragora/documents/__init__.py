"""
Enterprise Document Processing for Aragora.

Provides document ingestion, chunking, indexing, and auditing capabilities
for large-scale document analysis using multi-agent debate.

Key components:
- ingestion: Batch upload, parsing (Unstructured.io), streaming
- chunking: Token counting, semantic/sliding/recursive strategies
- indexing: Vector storage (Weaviate), hybrid search

Usage:
    from aragora.documents import (
        DocumentChunk,
        IngestedDocument,
        TokenCounter,
        ChunkingStrategy,
        BatchProcessor,
    )
"""

from aragora.documents.models import (
    ChunkType,
    DocumentChunk,
    DocumentStatus,
    IngestedDocument,
    get_model_token_limit,
    MODEL_TOKEN_LIMITS,
)
from aragora.documents.chunking import (
    TokenCounter,
    get_token_counter,
    ChunkingStrategy,
    ChunkingConfig,
    SemanticChunking,
    SlidingWindowChunking,
    RecursiveChunking,
    FixedSizeChunking,
    get_chunking_strategy,
    auto_select_strategy,
    ContextManager,
    ContextWindow,
    ContextConfig,
    ContextStrategy,
    get_context_manager,
)

__all__ = [
    # Models
    "ChunkType",
    "DocumentChunk",
    "DocumentStatus",
    "IngestedDocument",
    "get_model_token_limit",
    "MODEL_TOKEN_LIMITS",
    # Token counting
    "TokenCounter",
    "get_token_counter",
    # Chunking strategies
    "ChunkingStrategy",
    "ChunkingConfig",
    "SemanticChunking",
    "SlidingWindowChunking",
    "RecursiveChunking",
    "FixedSizeChunking",
    "get_chunking_strategy",
    "auto_select_strategy",
    # Context management
    "ContextManager",
    "ContextWindow",
    "ContextConfig",
    "ContextStrategy",
    "get_context_manager",
]
