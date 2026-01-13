"""
Database abstraction for the tournaments module.

Provides thread-safe database access by inheriting from SQLiteStore,
which provides standardized schema management and connection handling.
"""

from aragora.config import DB_TIMEOUT_SECONDS
from aragora.storage.base_store import SQLiteStore


class TournamentDatabase(SQLiteStore):
    """
    Database wrapper for tournament system operations.

    Inherits from SQLiteStore for standardized schema management.
    Uses WAL mode for better concurrent read/write performance.

    Usage:
        db = TournamentDatabase("/path/to/tournaments.db")

        # Context manager with auto-commit/rollback
        with db.connection() as conn:
            conn.execute("INSERT INTO ...")

        # Convenience methods
        row = db.fetch_one("SELECT * FROM tournaments WHERE id = ?", ("123",))
        rows = db.fetch_all("SELECT * FROM tournaments ORDER BY created_at DESC")
    """

    SCHEMA_NAME = "tournaments"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS tournaments (
            tournament_id TEXT PRIMARY KEY,
            name TEXT,
            format TEXT,
            agents TEXT,
            tasks TEXT,
            standings TEXT,
            champion TEXT,
            started_at TEXT,
            completed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS tournament_matches (
            match_id TEXT PRIMARY KEY,
            tournament_id TEXT,
            round_num INTEGER,
            participants TEXT,
            task_id TEXT,
            scores TEXT,
            winner TEXT,
            started_at TEXT,
            completed_at TEXT
        );
    """

    def __init__(self, db_path: str = "aragora_tournaments.db"):
        """Initialize tournament database."""
        super().__init__(db_path, timeout=DB_TIMEOUT_SECONDS)
