"""Tests for the ranking database module.

Tests cover:
- EloDatabase class
- ELO schema definition
- Schema migrations
"""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from aragora.ranking.database import (
    EloDatabase,
    ELO_SCHEMA_VERSION,
    ELO_INITIAL_SCHEMA,
)


class TestEloSchemaConstants:
    """Tests for schema constants."""

    def test_schema_version_defined(self):
        """Schema version should be defined."""
        assert ELO_SCHEMA_VERSION >= 1

    def test_initial_schema_defined(self):
        """Initial schema should be defined."""
        assert ELO_INITIAL_SCHEMA is not None
        assert len(ELO_INITIAL_SCHEMA) > 0

    def test_initial_schema_contains_tables(self):
        """Initial schema should define expected tables."""
        assert "ratings" in ELO_INITIAL_SCHEMA
        assert "matches" in ELO_INITIAL_SCHEMA
        assert "elo_history" in ELO_INITIAL_SCHEMA
        assert "calibration_predictions" in ELO_INITIAL_SCHEMA
        assert "domain_calibration" in ELO_INITIAL_SCHEMA
        assert "calibration_buckets" in ELO_INITIAL_SCHEMA
        assert "agent_relationships" in ELO_INITIAL_SCHEMA


class TestEloDatabaseInit:
    """Tests for EloDatabase initialization."""

    def test_database_class_exists(self):
        """EloDatabase class should exist."""
        assert EloDatabase is not None

    def test_schema_name(self):
        """Should have correct schema name."""
        assert EloDatabase.SCHEMA_NAME == "elo"

    def test_schema_version(self):
        """Should have correct schema version."""
        assert EloDatabase.SCHEMA_VERSION == ELO_SCHEMA_VERSION

    def test_initial_schema(self):
        """Should have initial schema defined."""
        assert EloDatabase.INITIAL_SCHEMA == ELO_INITIAL_SCHEMA


class TestEloDatabaseCreation:
    """Tests for EloDatabase creation and usage."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield str(Path(tmpdir) / "test_elo.db")

    def test_create_database(self, temp_db_path):
        """Should create database file."""
        db = EloDatabase(temp_db_path)
        # Database should be created
        assert db is not None

    def test_database_creates_tables(self, temp_db_path):
        """Should create expected tables."""
        db = EloDatabase(temp_db_path)

        # Check tables exist by querying sqlite_master
        with db.connection() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}

        # Verify key tables exist
        assert "ratings" in tables
        assert "matches" in tables
        assert "elo_history" in tables

    def test_database_connection_context_manager(self, temp_db_path):
        """Should support context manager for connections."""
        db = EloDatabase(temp_db_path)

        with db.connection() as conn:
            # Connection should be valid
            assert conn is not None
            # Should be able to execute queries
            conn.execute("SELECT 1")


class TestEloDatabaseMigrations:
    """Tests for EloDatabase migrations."""

    def test_has_register_migrations_method(self):
        """Should have register_migrations method."""
        assert hasattr(EloDatabase, "register_migrations")
        assert callable(getattr(EloDatabase, "register_migrations"))

    def test_has_migrate_v1_to_v2(self):
        """Should have v1 to v2 migration method."""
        assert hasattr(EloDatabase, "_migrate_v1_to_v2")


class TestRatingsTable:
    """Tests for the ratings table schema."""

    @pytest.fixture
    def db_with_tables(self):
        """Create database with tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_elo.db")
            db = EloDatabase(db_path)
            yield db

    def test_ratings_table_columns(self, db_with_tables):
        """Ratings table should have expected columns."""
        with db_with_tables.connection() as conn:
            cursor = conn.execute("PRAGMA table_info(ratings)")
            columns = {row[1] for row in cursor.fetchall()}

        expected_columns = {
            "agent_name",
            "elo",
            "domain_elos",
            "wins",
            "losses",
            "draws",
            "debates_count",
            "critiques_accepted",
            "critiques_total",
            "updated_at",
        }
        assert expected_columns.issubset(columns)

    def test_insert_rating(self, db_with_tables):
        """Should be able to insert rating."""
        with db_with_tables.connection() as conn:
            conn.execute(
                "INSERT INTO ratings (agent_name, elo) VALUES (?, ?)",
                ("test-agent", 1500.0),
            )
            cursor = conn.execute(
                "SELECT agent_name, elo FROM ratings WHERE agent_name = ?",
                ("test-agent",),
            )
            row = cursor.fetchone()

        assert row is not None
        assert row[0] == "test-agent"
        assert row[1] == 1500.0


class TestMatchesTable:
    """Tests for the matches table schema."""

    @pytest.fixture
    def db_with_tables(self):
        """Create database with tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_elo.db")
            db = EloDatabase(db_path)
            yield db

    def test_matches_table_columns(self, db_with_tables):
        """Matches table should have expected columns."""
        with db_with_tables.connection() as conn:
            cursor = conn.execute("PRAGMA table_info(matches)")
            columns = {row[1] for row in cursor.fetchall()}

        expected_columns = {
            "id",
            "debate_id",
            "winner",
            "participants",
            "domain",
            "scores",
            "elo_changes",
            "created_at",
        }
        assert expected_columns.issubset(columns)

    def test_insert_match(self, db_with_tables):
        """Should be able to insert match."""
        with db_with_tables.connection() as conn:
            conn.execute(
                """
                INSERT INTO matches (debate_id, winner, participants, domain)
                VALUES (?, ?, ?, ?)
                """,
                ("debate-123", "agent-a", '["agent-a", "agent-b"]', "logic"),
            )
            cursor = conn.execute(
                "SELECT debate_id, winner FROM matches WHERE debate_id = ?",
                ("debate-123",),
            )
            row = cursor.fetchone()

        assert row is not None
        assert row[0] == "debate-123"
        assert row[1] == "agent-a"


class TestEloHistoryTable:
    """Tests for the elo_history table schema."""

    @pytest.fixture
    def db_with_tables(self):
        """Create database with tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_elo.db")
            db = EloDatabase(db_path)
            yield db

    def test_elo_history_table_columns(self, db_with_tables):
        """Elo history table should have expected columns."""
        with db_with_tables.connection() as conn:
            cursor = conn.execute("PRAGMA table_info(elo_history)")
            columns = {row[1] for row in cursor.fetchall()}

        expected_columns = {
            "id",
            "agent_name",
            "elo",
            "debate_id",
            "created_at",
        }
        assert expected_columns.issubset(columns)

    def test_elo_history_indexes(self, db_with_tables):
        """Should have performance indexes."""
        with db_with_tables.connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='elo_history'"
            )
            indexes = {row[0] for row in cursor.fetchall()}

        assert "idx_elo_history_agent" in indexes
        assert "idx_elo_history_created" in indexes


class TestAgentRelationshipsTable:
    """Tests for the agent_relationships table schema."""

    @pytest.fixture
    def db_with_tables(self):
        """Create database with tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test_elo.db")
            db = EloDatabase(db_path)
            yield db

    def test_relationships_table_columns(self, db_with_tables):
        """Agent relationships table should have expected columns."""
        with db_with_tables.connection() as conn:
            cursor = conn.execute("PRAGMA table_info(agent_relationships)")
            columns = {row[1] for row in cursor.fetchall()}

        expected_columns = {
            "agent_a",
            "agent_b",
            "debate_count",
            "agreement_count",
            "a_wins_over_b",
            "b_wins_over_a",
        }
        assert expected_columns.issubset(columns)

    def test_relationships_check_constraint(self, db_with_tables):
        """Should enforce agent_a < agent_b ordering."""
        with db_with_tables.connection() as conn:
            # This should work (a < b)
            conn.execute(
                """
                INSERT INTO agent_relationships (agent_a, agent_b, debate_count)
                VALUES (?, ?, ?)
                """,
                ("alice", "bob", 1),
            )

            # This should fail (b > a) due to CHECK constraint
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    """
                    INSERT INTO agent_relationships (agent_a, agent_b, debate_count)
                    VALUES (?, ?, ?)
                    """,
                    ("charlie", "alice", 1),
                )
