"""
Database abstraction for the ELO ranking system.

Provides thread-safe database access by inheriting from BaseDatabase,
which delegates to DatabaseManager with per-operation connections
for concurrent access patterns.
"""

from aragora.storage import BaseDatabase


class EloDatabase(BaseDatabase):
    """
    Database wrapper for ELO system operations.

    Inherits thread-safe access via BaseDatabase, which uses
    DatabaseManager.fresh_connection() for per-operation connections.
    Uses WAL mode for better concurrent read/write performance.

    Usage:
        db = EloDatabase("/path/to/elo.db")

        # Context manager with auto-commit/rollback
        with db.connection() as conn:
            conn.execute("INSERT INTO ...")

        # Convenience methods
        row = db.fetch_one("SELECT * FROM ratings WHERE agent_name = ?", ("claude",))
        rows = db.fetch_all("SELECT * FROM ratings ORDER BY elo DESC LIMIT ?", (10,))
    """

    pass
