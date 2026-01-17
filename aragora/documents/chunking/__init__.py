"""
Document chunking: token counting and chunking strategies.

Provides semantic, sliding window, and recursive chunking
with per-model token accounting.
"""

from aragora.documents.chunking.token_counter import (
    TokenCounter,
    get_token_counter,
)
from aragora.documents.chunking.strategies import (
    ChunkingStrategy,
    ChunkingConfig,
    SemanticChunking,
    SlidingWindowChunking,
    RecursiveChunking,
    FixedSizeChunking,
    get_chunking_strategy,
    auto_select_strategy,
)
from aragora.documents.chunking.context_manager import (
    ContextManager,
    ContextWindow,
    ContextConfig,
    ContextStrategy,
    get_context_manager,
)

__all__ = [
    # Token counter
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
    # Context manager
    "ContextManager",
    "ContextWindow",
    "ContextConfig",
    "ContextStrategy",
    "get_context_manager",
]
