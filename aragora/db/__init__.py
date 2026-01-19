"""
Database abstraction layer for Aragora.

Provides a unified interface for SQLite (development) and PostgreSQL (production).

Usage:
    from aragora.db import get_database, DatabaseBackend

    # Get the configured database backend
    db = get_database()

    with db.connection() as conn:
        result = conn.execute("SELECT * FROM debates WHERE id = ?", (debate_id,))
        row = result.fetchone()
"""

from aragora.db.backends import (
    DatabaseBackend,
    DatabaseConfig,
    PostgresBackend,
    SQLiteBackend,
    configure_database,
    get_database,
)
from aragora.db.profiling import (
    QueryProfiler,
    QueryProfile,
    profile_queries,
    profile_function,
    instrument_sqlite_connection,
    get_index_recommendations,
    apply_recommended_indexes,
)

__all__ = [
    "DatabaseBackend",
    "SQLiteBackend",
    "PostgresBackend",
    "get_database",
    "configure_database",
    "DatabaseConfig",
    # Profiling utilities
    "QueryProfiler",
    "QueryProfile",
    "profile_queries",
    "profile_function",
    "instrument_sqlite_connection",
    "get_index_recommendations",
    "apply_recommended_indexes",
]
