"""
Aragora Repository Pattern Implementation.

Provides database abstraction layer for data access, separating persistence
logic from business logic. Each repository handles a single entity type.

Usage:
    from aragora.persistence.repositories import DebateRepository

    repo = DebateRepository()
    debate = repo.get("debate-123")
    repo.save(debate)

Unit of Work for transactions:
    from aragora.persistence.repositories import UnitOfWork

    with UnitOfWork.sync("/path/to/db.db") as uow:
        result = uow.execute_one("SELECT * FROM debates WHERE id = ?", (id,))
        uow.execute("UPDATE debates SET view_count = ? WHERE id = ?", (10, id))
        # Auto-commits on success, rolls back on exception

Benefits:
- Testable: Repositories can be mocked or use in-memory databases
- Consistent: Common patterns for CRUD operations
- Decoupled: Business logic doesn't know about SQLite internals
- Observable: Query metrics and slow query detection
"""

from .base import BaseRepository, RepositoryError, EntityNotFoundError
from .debate import DebateRepository, DebateEntity, DebateMetadata
from .elo import EloRepository, RatingEntity, MatchEntity, LeaderboardEntry
from .unit_of_work import (
    UnitOfWork,
    TransactionMetrics,
    QueryMetrics,
    InstrumentedConnection,
    batch_insert,
    batch_update,
)

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
    # Unit of Work
    "UnitOfWork",
    "TransactionMetrics",
    "QueryMetrics",
    "InstrumentedConnection",
    "batch_insert",
    "batch_update",
]
