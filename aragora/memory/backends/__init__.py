"""
Memory Backend Implementations.

Provides pluggable storage backends for the memory system:
- InMemoryBackend: Testing and development
- SQLiteBackend: Local file-based storage (default)
- (Future) PostgresBackend: Production PostgreSQL storage
- (Future) RedisBackend: High-speed caching backend
"""

from aragora.memory.backends.in_memory import InMemoryBackend

__all__ = ["InMemoryBackend"]
