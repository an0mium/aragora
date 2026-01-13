"""
Database abstraction for the genesis module.

Provides standardized schema management by inheriting from SQLiteStore.
"""

from aragora.config import DB_TIMEOUT_SECONDS
from aragora.storage.base_store import SQLiteStore


class GenesisDatabase(SQLiteStore):
    """
    Database wrapper for genesis system operations.

    Inherits from SQLiteStore for standardized schema management.
    Uses WAL mode for better concurrent read/write performance.

    Usage:
        db = GenesisDatabase("/path/to/genesis.db")

        # Context manager with auto-commit/rollback
        with db.connection() as conn:
            conn.execute("INSERT INTO ...")
    """

    SCHEMA_NAME = "genesis"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        -- Genesis events for ledger tracking
        CREATE TABLE IF NOT EXISTS genesis_events (
            event_id TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            parent_event_id TEXT,
            content_hash TEXT NOT NULL,
            data TEXT,
            FOREIGN KEY (parent_event_id) REFERENCES genesis_events(event_id)
        );

        CREATE INDEX IF NOT EXISTS idx_events_type
        ON genesis_events(event_type);

        CREATE INDEX IF NOT EXISTS idx_events_timestamp
        ON genesis_events(timestamp);

        -- Genomes for agent evolution
        CREATE TABLE IF NOT EXISTS genomes (
            genome_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            traits TEXT,
            expertise TEXT,
            model_preference TEXT,
            parent_genomes TEXT,
            generation INTEGER DEFAULT 0,
            fitness_score REAL DEFAULT 0.5,
            birth_debate_id TEXT,
            created_at TEXT,
            updated_at TEXT,
            consensus_contributions INTEGER DEFAULT 0,
            critiques_accepted INTEGER DEFAULT 0,
            predictions_correct INTEGER DEFAULT 0,
            debates_participated INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_genomes_fitness
        ON genomes(fitness_score DESC);

        CREATE INDEX IF NOT EXISTS idx_genomes_generation
        ON genomes(generation);

        -- Populations for breeding system
        CREATE TABLE IF NOT EXISTS populations (
            population_id TEXT PRIMARY KEY,
            genome_ids TEXT,
            generation INTEGER DEFAULT 0,
            created_at TEXT,
            debate_history TEXT
        );

        CREATE TABLE IF NOT EXISTS active_population (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            population_id TEXT
        );
    """

    def __init__(self, db_path: str = ".nomic/genesis.db"):
        """Initialize genesis database."""
        super().__init__(db_path, timeout=DB_TIMEOUT_SECONDS)
