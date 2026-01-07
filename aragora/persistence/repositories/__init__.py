"""
Aragora Repository Pattern Implementation.

Provides database abstraction layer for data access, separating persistence
logic from business logic. Each repository handles a single entity type.

Usage:
    from aragora.persistence.repositories import DebateRepository

    repo = DebateRepository()
    debate = repo.get("debate-123")
    repo.save(debate)

Benefits:
- Testable: Repositories can be mocked or use in-memory databases
- Consistent: Common patterns for CRUD operations
- Decoupled: Business logic doesn't know about SQLite internals
"""

from .base import BaseRepository, RepositoryError, EntityNotFoundError
from .debate import DebateRepository, DebateEntity, DebateMetadata
from .elo import EloRepository, RatingEntity, MatchEntity, LeaderboardEntry

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
]
