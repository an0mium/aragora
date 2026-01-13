"""
Aragora Repository Pattern Implementation.

Provides database abstraction layer for data access, separating persistence
logic from business logic. Each repository handles a single entity type.

Usage:
    from aragora.persistence.repositories import DebateRepository, MemoryRepository

    # Debate repository
    repo = DebateRepository()
    debate = repo.get("debate-123")
    repo.save(debate)

    # Memory repository
    mem_repo = MemoryRepository()
    memory = mem_repo.add_memory("claude", "Learned something", memory_type="insight")
    memories = mem_repo.retrieve("claude", query="testing", limit=5)

Benefits:
- Testable: Repositories can be mocked or use in-memory databases
- Consistent: Common patterns for CRUD operations
- Decoupled: Business logic doesn't know about SQLite internals
- Type-safe: Entity classes with validation
"""

from .base import BaseRepository, EntityNotFoundError, RepositoryError
from .debate import DebateEntity, DebateMetadata, DebateRepository
from .elo import EloRepository, LeaderboardEntry, MatchEntity, RatingEntity
from .memory import MemoryEntity, MemoryRepository, ReflectionSchedule, RetrievedMemory

__all__ = [
    # Base
    "BaseRepository",
    "RepositoryError",
    "EntityNotFoundError",
    # Debate
    "DebateRepository",
    "DebateEntity",
    "DebateMetadata",
    # ELO
    "EloRepository",
    "RatingEntity",
    "MatchEntity",
    "LeaderboardEntry",
    # Memory
    "MemoryRepository",
    "MemoryEntity",
    "RetrievedMemory",
    "ReflectionSchedule",
]
