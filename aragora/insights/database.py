"""
Database abstraction for the insights module.

Provides standardized schema management by inheriting from SQLiteStore.
"""

from aragora.config import DB_TIMEOUT_SECONDS, resolve_db_path
from aragora.storage.base_store import SQLiteStore


class InsightsDatabase(SQLiteStore):
    """
    Database wrapper for insights system operations.

    Inherits from SQLiteStore for standardized schema management.
    Uses WAL mode for better concurrent read/write performance.

    Note: This is a generic database wrapper. Individual stores
    (PositionTracker, PersonaLaboratory) define their own schemas.

    Usage:
        db = InsightsDatabase("/path/to/insights.db")

        # Context manager with auto-commit/rollback
        with db.connection() as conn:
            conn.execute("INSERT INTO ...")
    """

    SCHEMA_NAME = "insights"
    SCHEMA_VERSION = 1

    # Position tracking tables (shared by PositionTracker)
    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS position_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            debate_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            position_type TEXT NOT NULL,
            position_text TEXT NOT NULL,
            round_num INTEGER DEFAULT 0,
            confidence REAL DEFAULT 0.5,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            was_winning_position INTEGER DEFAULT NULL,
            verified_correct INTEGER DEFAULT NULL,
            UNIQUE(debate_id, agent_name, position_type, round_num)
        );

        CREATE TABLE IF NOT EXISTS debate_outcomes (
            debate_id TEXT PRIMARY KEY,
            winning_agent TEXT,
            winning_position TEXT,
            consensus_confidence REAL,
            verified_at TEXT DEFAULT NULL,
            verification_result INTEGER DEFAULT NULL,
            verification_source TEXT DEFAULT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_position_agent
        ON position_history(agent_name);

        CREATE INDEX IF NOT EXISTS idx_position_debate
        ON position_history(debate_id);

        -- Laboratory tables (shared by PersonaLaboratory)
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id TEXT PRIMARY KEY,
            agent_name TEXT NOT NULL,
            control_persona TEXT NOT NULL,
            variant_persona TEXT NOT NULL,
            hypothesis TEXT,
            status TEXT DEFAULT 'running',
            control_successes INTEGER DEFAULT 0,
            control_trials INTEGER DEFAULT 0,
            variant_successes INTEGER DEFAULT 0,
            variant_trials INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            completed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS emergent_traits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trait_name TEXT NOT NULL,
            source_agents TEXT NOT NULL,
            supporting_evidence TEXT,
            confidence REAL DEFAULT 0.5,
            first_detected TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS trait_transfers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_agent TEXT NOT NULL,
            to_agent TEXT NOT NULL,
            trait TEXT NOT NULL,
            expertise_domain TEXT,
            success_rate_before REAL,
            success_rate_after REAL,
            transferred_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS evolution_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            mutation_type TEXT NOT NULL,
            before_state TEXT NOT NULL,
            after_state TEXT NOT NULL,
            reason TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """

    def __init__(self, db_path: str = "aragora_insights.db"):
        """Initialize insights database."""
        super().__init__(resolve_db_path(db_path), timeout=DB_TIMEOUT_SECONDS)
