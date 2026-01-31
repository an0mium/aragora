"""
Memory Backend Implementations.

Provides pluggable storage backends for the memory system:
- InMemoryBackend: Testing and development
- SQLiteBackend: Local file-based storage (default)
- (Future) PostgresBackend: Production PostgreSQL storage
- (Future) RedisBackend: High-speed caching backend

Vector Index:
- VectorIndex: FAISS-accelerated similarity search with numpy fallback
"""

from aragora.memory.backends.in_memory import InMemoryBackend
from aragora.memory.backends.vector_index import (
    VectorIndex,
    VectorIndexConfig,
    SearchResult,
    HAS_FAISS,
    HAS_NUMPY,
)

__all__ = [
    "InMemoryBackend",
    "VectorIndex",
    "VectorIndexConfig",
    "SearchResult",
    "HAS_FAISS",
    "HAS_NUMPY",
]
