"""
Database abstraction for the ELO ranking system.

Provides schema management and thread-safe database access for ELO operations.
"""

from __future__ import annotations

__all__ = [
    "EloDatabase",
    "ELO_SCHEMA_VERSION",
]

import sqlite3

from aragora.config import resolve_db_path
from aragora.storage.base_store import SQLiteStore
from aragora.storage.schema import SchemaManager, safe_add_column

ELO_SCHEMA_VERSION = 2

ELO_INITIAL_SCHEMA = """
    -- Agent ratings
    CREATE TABLE IF NOT EXISTS ratings (
        agent_name TEXT PRIMARY KEY,
        elo REAL DEFAULT 1500,
        domain_elos TEXT,
        wins INTEGER DEFAULT 0,
        losses INTEGER DEFAULT 0,
        draws INTEGER DEFAULT 0,
        debates_count INTEGER DEFAULT 0,
        critiques_accepted INTEGER DEFAULT 0,
        critiques_total INTEGER DEFAULT 0,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    -- Match history
    CREATE TABLE IF NOT EXISTS matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        debate_id TEXT UNIQUE,
        winner TEXT,
        participants TEXT,
        domain TEXT,
        scores TEXT,
        elo_changes TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    -- ELO history for tracking progression
    CREATE TABLE IF NOT EXISTS elo_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        agent_name TEXT NOT NULL,
        elo REAL NOT NULL,
        debate_id TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    -- Calibration predictions table
    CREATE TABLE IF NOT EXISTS calibration_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tournament_id TEXT NOT NULL,
        predictor_agent TEXT NOT NULL,
        predicted_winner TEXT NOT NULL,
        confidence REAL NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(tournament_id, predictor_agent)
    );

    -- Domain-specific calibration tracking
    CREATE TABLE IF NOT EXISTS domain_calibration (
        agent_name TEXT NOT NULL,
        domain TEXT NOT NULL,
        total_predictions INTEGER DEFAULT 0,
        total_correct INTEGER DEFAULT 0,
        brier_sum REAL DEFAULT 0.0,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (agent_name, domain)
    );

    -- Calibration by confidence bucket
    CREATE TABLE IF NOT EXISTS calibration_buckets (
        agent_name TEXT NOT NULL,
        domain TEXT NOT NULL,
        bucket_key TEXT NOT NULL,
        predictions INTEGER DEFAULT 0,
        correct INTEGER DEFAULT 0,
        brier_sum REAL DEFAULT 0.0,
        PRIMARY KEY (agent_name, domain, bucket_key)
    );

    -- Agent relationships tracking
    CREATE TABLE IF NOT EXISTS agent_relationships (
        agent_a TEXT NOT NULL,
        agent_b TEXT NOT NULL,
        debate_count INTEGER DEFAULT 0,
        agreement_count INTEGER DEFAULT 0,
        critique_count_a_to_b INTEGER DEFAULT 0,
        critique_count_b_to_a INTEGER DEFAULT 0,
        critique_accepted_a_to_b INTEGER DEFAULT 0,
        critique_accepted_b_to_a INTEGER DEFAULT 0,
        position_changes_a_after_b INTEGER DEFAULT 0,
        position_changes_b_after_a INTEGER DEFAULT 0,
        a_wins_over_b INTEGER DEFAULT 0,
        b_wins_over_a INTEGER DEFAULT 0,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (agent_a, agent_b),
        CHECK (agent_a < agent_b)
    );

    -- Performance indexes
    CREATE INDEX IF NOT EXISTS idx_elo_history_agent ON elo_history(agent_name);
    CREATE INDEX IF NOT EXISTS idx_elo_history_created ON elo_history(created_at);
    CREATE INDEX IF NOT EXISTS idx_elo_history_debate ON elo_history(debate_id);
    CREATE INDEX IF NOT EXISTS idx_matches_winner ON matches(winner);
    CREATE INDEX IF NOT EXISTS idx_matches_created ON matches(created_at);
    CREATE INDEX IF NOT EXISTS idx_matches_domain ON matches(domain);
    CREATE INDEX IF NOT EXISTS idx_domain_cal_agent ON domain_calibration(agent_name);
    CREATE INDEX IF NOT EXISTS idx_domain_cal_agent_domain ON domain_calibration(agent_name, domain);
    CREATE INDEX IF NOT EXISTS idx_relationships_a ON agent_relationships(agent_a);
    CREATE INDEX IF NOT EXISTS idx_relationships_b ON agent_relationships(agent_b);
    CREATE INDEX IF NOT EXISTS idx_calibration_pred_tournament ON calibration_predictions(tournament_id);
    CREATE INDEX IF NOT EXISTS idx_ratings_agent ON ratings(agent_name);
"""


class EloDatabase(SQLiteStore):
    """
    SQLite-backed ELO store with schema management.
    """

    SCHEMA_NAME = "elo"
    SCHEMA_VERSION = ELO_SCHEMA_VERSION
    INITIAL_SCHEMA = ELO_INITIAL_SCHEMA

    def __init__(self, db_path: str):
        super().__init__(resolve_db_path(db_path))

    def register_migrations(self, manager: SchemaManager) -> None:
        """Register schema migrations for ELO database."""
        manager.register_migration(
            from_version=1,
            to_version=2,
            function=self._migrate_v1_to_v2,
            description="Add calibration columns to ratings table",
        )

    def _migrate_v1_to_v2(self, conn: sqlite3.Connection) -> None:
        """Migration: Add calibration columns to ratings table."""
        safe_add_column(conn, "ratings", "calibration_correct", "INTEGER", default="0")
        safe_add_column(conn, "ratings", "calibration_total", "INTEGER", default="0")
        safe_add_column(conn, "ratings", "calibration_brier_sum", "REAL", default="0.0")
