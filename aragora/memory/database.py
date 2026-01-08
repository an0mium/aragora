"""
Database abstraction for the memory module.

Provides thread-safe database access by inheriting from BaseDatabase,
which delegates to DatabaseManager with per-operation connections
for concurrent access patterns.
"""

from aragora.storage import BaseDatabase


class MemoryDatabase(BaseDatabase):
    """
    Database wrapper for memory system operations.

    Inherits thread-safe access via BaseDatabase, which uses
    DatabaseManager.fresh_connection() for per-operation connections.
    Uses WAL mode for better concurrent read/write performance.

    Usage:
        db = MemoryDatabase("/path/to/memory.db")

        # Context manager with auto-commit/rollback
        with db.connection() as conn:
            conn.execute("INSERT INTO ...")

        # Convenience methods
        row = db.fetch_one("SELECT * FROM memories WHERE id = ?", ("123",))
        rows = db.fetch_all("SELECT * FROM memories ORDER BY timestamp DESC")
    """

    pass
