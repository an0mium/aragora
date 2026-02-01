"""
Tests for new migration files.

Covers:
- Forward migration execution
- Rollback migration execution
- Migration idempotency
- Schema verification after migration

Run with:
    python -m pytest tests/migrations/test_new_migrations.py -v --noconftest --timeout=60
"""

from __future__ import annotations

import sqlite3
from typing import Optional

import pytest


# ---------------------------------------------------------------------------
# Helpers: lightweight in-memory SQLite backend
# ---------------------------------------------------------------------------


class InMemorySQLiteBackend:
    """Minimal DatabaseBackend implementation for testing."""

    backend_type = "sqlite"

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute("PRAGMA journal_mode=WAL")

    def execute_write(self, sql: str, params: tuple = ()) -> None:
        self._conn.execute(sql, params)
        self._conn.commit()

    def fetch_all(self, sql: str, params: tuple = ()) -> list[tuple]:
        cursor = self._conn.execute(sql, params)
        return cursor.fetchall()

    def fetch_one(self, sql: str, params: tuple = ()) -> Optional[tuple]:
        cursor = self._conn.execute(sql, params)
        return cursor.fetchone()

    def close(self) -> None:
        self._conn.close()

    def table_exists(self, table: str) -> bool:
        rows = self.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        return len(rows) > 0

    def index_exists(self, index: str) -> bool:
        rows = self.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='index' AND name=?",
            (index,),
        )
        return len(rows) > 0

    def get_columns(self, table: str) -> set[str]:
        cols = self.fetch_all(f"PRAGMA table_info({table})")
        return {row[1] for row in cols}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def backend():
    """Provide a fresh in-memory SQLite backend per test."""
    b = InMemorySQLiteBackend()
    yield b
    b.close()


@pytest.fixture()
def runner(backend):
    """Provide a MigrationRunner wired to the in-memory backend."""
    from aragora.migrations.runner import MigrationRunner

    return MigrationRunner(backend=backend)


# ---------------------------------------------------------------------------
# Test Debate Metrics Indexes Migration
# ---------------------------------------------------------------------------


class TestDebateMetricsIndexesMigration:
    """Tests for v20260201000000_add_debate_metrics_indexes."""

    def test_import_migration(self):
        """Verify migration module can be imported."""
        from aragora.migrations.versions import v20260201000000_add_debate_metrics_indexes as mod

        assert hasattr(mod, "migration")
        assert hasattr(mod, "up_fn")
        assert hasattr(mod, "down_fn")

    def test_migration_metadata(self):
        """Verify migration has correct metadata."""
        from aragora.migrations.versions.v20260201000000_add_debate_metrics_indexes import migration

        assert migration.version == 20260201000000
        assert "index" in migration.name.lower()
        assert migration.up_fn is not None
        assert migration.down_fn is not None

    def test_forward_migration_with_gauntlet_table(self, backend):
        """Test forward migration creates indexes on existing gauntlet_results table."""
        from aragora.migrations.versions.v20260201000000_add_debate_metrics_indexes import up_fn

        # Create prerequisite table
        backend.execute_write("""
            CREATE TABLE gauntlet_results (
                gauntlet_id TEXT PRIMARY KEY,
                verdict TEXT,
                confidence REAL,
                robustness_score REAL,
                created_at TIMESTAMP
            )
        """)

        # Run migration
        up_fn(backend)

        # Verify indexes created
        assert backend.index_exists("idx_gauntlet_results_verdict_created")
        assert backend.index_exists("idx_gauntlet_results_confidence")
        assert backend.index_exists("idx_gauntlet_results_robustness")

    def test_rollback_migration(self, backend):
        """Test rollback removes indexes."""
        from aragora.migrations.versions.v20260201000000_add_debate_metrics_indexes import (
            up_fn,
            down_fn,
        )

        # Create prerequisite table
        backend.execute_write("""
            CREATE TABLE gauntlet_results (
                gauntlet_id TEXT PRIMARY KEY,
                verdict TEXT,
                confidence REAL,
                robustness_score REAL,
                created_at TIMESTAMP
            )
        """)

        # Run forward then rollback
        up_fn(backend)
        down_fn(backend)

        # Verify indexes removed
        assert not backend.index_exists("idx_gauntlet_results_verdict_created")
        assert not backend.index_exists("idx_gauntlet_results_confidence")

    def test_idempotent_forward_migration(self, backend):
        """Test migration can be run multiple times without error."""
        from aragora.migrations.versions.v20260201000000_add_debate_metrics_indexes import up_fn

        # Create prerequisite table
        backend.execute_write("""
            CREATE TABLE gauntlet_results (
                gauntlet_id TEXT PRIMARY KEY,
                verdict TEXT,
                confidence REAL,
                robustness_score REAL,
                created_at TIMESTAMP
            )
        """)

        # Run twice - should not raise
        up_fn(backend)
        up_fn(backend)

    def test_migration_without_prerequisite_tables(self, backend):
        """Test migration handles missing prerequisite tables gracefully."""
        from aragora.migrations.versions.v20260201000000_add_debate_metrics_indexes import up_fn

        # Run without creating tables - should not raise
        up_fn(backend)


# ---------------------------------------------------------------------------
# Test Agent Performance Tracking Migration
# ---------------------------------------------------------------------------


class TestAgentPerformanceTrackingMigration:
    """Tests for v20260201000100_add_agent_performance_tracking."""

    def test_import_migration(self):
        """Verify migration module can be imported."""
        from aragora.migrations.versions import (
            v20260201000100_add_agent_performance_tracking as mod,
        )

        assert hasattr(mod, "migration")

    def test_migration_metadata(self):
        """Verify migration has correct metadata."""
        from aragora.migrations.versions.v20260201000100_add_agent_performance_tracking import (
            migration,
        )

        assert migration.version == 20260201000100
        assert "agent" in migration.name.lower()
        assert "performance" in migration.name.lower()

    def test_forward_migration_creates_table(self, backend):
        """Test forward migration creates agent_performance table."""
        from aragora.migrations.versions.v20260201000100_add_agent_performance_tracking import (
            up_fn,
        )

        up_fn(backend)

        assert backend.table_exists("agent_performance")

        # Verify columns
        cols = backend.get_columns("agent_performance")
        assert "id" in cols
        assert "agent_id" in cols
        assert "agent_type" in cols
        assert "duration_ms" in cols
        assert "input_tokens" in cols
        assert "output_tokens" in cols
        assert "elo_rating" in cols

    def test_forward_migration_creates_indexes(self, backend):
        """Test forward migration creates indexes."""
        from aragora.migrations.versions.v20260201000100_add_agent_performance_tracking import (
            up_fn,
        )

        up_fn(backend)

        assert backend.index_exists("idx_agent_perf_agent_time")
        assert backend.index_exists("idx_agent_perf_type")
        assert backend.index_exists("idx_agent_perf_debate")

    def test_rollback_migration(self, backend):
        """Test rollback removes table and indexes."""
        from aragora.migrations.versions.v20260201000100_add_agent_performance_tracking import (
            up_fn,
            down_fn,
        )

        up_fn(backend)
        assert backend.table_exists("agent_performance")

        down_fn(backend)
        assert not backend.table_exists("agent_performance")

    def test_idempotent_forward_migration(self, backend):
        """Test migration can be run multiple times without error."""
        from aragora.migrations.versions.v20260201000100_add_agent_performance_tracking import (
            up_fn,
        )

        # Run twice - should not raise
        up_fn(backend)
        up_fn(backend)

        assert backend.table_exists("agent_performance")


# ---------------------------------------------------------------------------
# Test Decision Receipts Migration
# ---------------------------------------------------------------------------


class TestDecisionReceiptsMigration:
    """Tests for v20260201000200_add_decision_receipts_table."""

    def test_import_migration(self):
        """Verify migration module can be imported."""
        from aragora.migrations.versions import v20260201000200_add_decision_receipts_table as mod

        assert hasattr(mod, "migration")

    def test_migration_metadata(self):
        """Verify migration has correct metadata."""
        from aragora.migrations.versions.v20260201000200_add_decision_receipts_table import (
            migration,
        )

        assert migration.version == 20260201000200
        assert "receipt" in migration.name.lower()

    def test_forward_migration_creates_table(self, backend):
        """Test forward migration creates decision_receipts table."""
        from aragora.migrations.versions.v20260201000200_add_decision_receipts_table import up_fn

        up_fn(backend)

        assert backend.table_exists("decision_receipts")

        # Verify columns
        cols = backend.get_columns("decision_receipts")
        assert "receipt_id" in cols
        assert "debate_id" in cols
        assert "receipt_hash" in cols
        assert "hash_algorithm" in cols
        assert "decision_summary" in cols
        assert "confidence_score" in cols
        assert "chain_hash" in cols

    def test_forward_migration_creates_indexes(self, backend):
        """Test forward migration creates indexes."""
        from aragora.migrations.versions.v20260201000200_add_decision_receipts_table import up_fn

        up_fn(backend)

        assert backend.index_exists("idx_receipts_debate")
        assert backend.index_exists("idx_receipts_hash")
        assert backend.index_exists("idx_receipts_chain")

    def test_rollback_migration(self, backend):
        """Test rollback removes table and indexes."""
        from aragora.migrations.versions.v20260201000200_add_decision_receipts_table import (
            up_fn,
            down_fn,
        )

        up_fn(backend)
        assert backend.table_exists("decision_receipts")

        down_fn(backend)
        assert not backend.table_exists("decision_receipts")

    def test_insert_and_query_receipt(self, backend):
        """Test table can store and retrieve data."""
        from aragora.migrations.versions.v20260201000200_add_decision_receipts_table import up_fn

        up_fn(backend)

        # Insert test data
        backend.execute_write("""
            INSERT INTO decision_receipts (receipt_id, debate_id, receipt_hash, decision_summary)
            VALUES ('r1', 'd1', 'abc123', 'Test decision')
        """)

        # Query it back
        row = backend.fetch_one(
            "SELECT receipt_id, debate_id, receipt_hash FROM decision_receipts WHERE receipt_id = ?",
            ("r1",),
        )
        assert row is not None
        assert row[0] == "r1"
        assert row[1] == "d1"
        assert row[2] == "abc123"


# ---------------------------------------------------------------------------
# Test Rate Limit Tracking Migration
# ---------------------------------------------------------------------------


class TestRateLimitTrackingMigration:
    """Tests for v20260201000300_add_rate_limit_tracking."""

    def test_import_migration(self):
        """Verify migration module can be imported."""
        from aragora.migrations.versions import v20260201000300_add_rate_limit_tracking as mod

        assert hasattr(mod, "migration")

    def test_migration_metadata(self):
        """Verify migration has correct metadata."""
        from aragora.migrations.versions.v20260201000300_add_rate_limit_tracking import migration

        assert migration.version == 20260201000300
        assert "rate" in migration.name.lower()

    def test_forward_migration_creates_tables(self, backend):
        """Test forward migration creates rate limit tables."""
        from aragora.migrations.versions.v20260201000300_add_rate_limit_tracking import up_fn

        up_fn(backend)

        assert backend.table_exists("rate_limit_entries")
        assert backend.table_exists("rate_limit_violations")

    def test_rate_limit_entries_schema(self, backend):
        """Test rate_limit_entries table schema."""
        from aragora.migrations.versions.v20260201000300_add_rate_limit_tracking import up_fn

        up_fn(backend)

        cols = backend.get_columns("rate_limit_entries")
        assert "key" in cols
        assert "bucket_type" in cols
        assert "tokens_remaining" in cols
        assert "expires_at" in cols

    def test_rate_limit_violations_schema(self, backend):
        """Test rate_limit_violations table schema."""
        from aragora.migrations.versions.v20260201000300_add_rate_limit_tracking import up_fn

        up_fn(backend)

        cols = backend.get_columns("rate_limit_violations")
        assert "key" in cols
        assert "bucket_type" in cols
        assert "client_ip" in cols
        assert "user_id" in cols

    def test_rollback_migration(self, backend):
        """Test rollback removes both tables."""
        from aragora.migrations.versions.v20260201000300_add_rate_limit_tracking import (
            up_fn,
            down_fn,
        )

        up_fn(backend)
        assert backend.table_exists("rate_limit_entries")
        assert backend.table_exists("rate_limit_violations")

        down_fn(backend)
        assert not backend.table_exists("rate_limit_entries")
        assert not backend.table_exists("rate_limit_violations")


# ---------------------------------------------------------------------------
# Test Session Management Migration
# ---------------------------------------------------------------------------


class TestSessionManagementMigration:
    """Tests for v20260201000400_add_session_management."""

    def test_import_migration(self):
        """Verify migration module can be imported."""
        from aragora.migrations.versions import v20260201000400_add_session_management as mod

        assert hasattr(mod, "migration")

    def test_migration_metadata(self):
        """Verify migration has correct metadata."""
        from aragora.migrations.versions.v20260201000400_add_session_management import migration

        assert migration.version == 20260201000400
        assert "session" in migration.name.lower()

    def test_forward_migration_creates_tables(self, backend):
        """Test forward migration creates session tables."""
        from aragora.migrations.versions.v20260201000400_add_session_management import up_fn

        up_fn(backend)

        assert backend.table_exists("user_sessions")
        assert backend.table_exists("session_events")

    def test_user_sessions_schema(self, backend):
        """Test user_sessions table schema."""
        from aragora.migrations.versions.v20260201000400_add_session_management import up_fn

        up_fn(backend)

        cols = backend.get_columns("user_sessions")
        assert "session_id" in cols
        assert "user_id" in cols
        assert "expires_at" in cols
        assert "ip_address" in cols
        assert "auth_method" in cols
        assert "mfa_verified" in cols

    def test_session_events_schema(self, backend):
        """Test session_events table schema."""
        from aragora.migrations.versions.v20260201000400_add_session_management import up_fn

        up_fn(backend)

        cols = backend.get_columns("session_events")
        assert "session_id" in cols
        assert "user_id" in cols
        assert "event_type" in cols
        assert "success" in cols

    def test_rollback_migration(self, backend):
        """Test rollback removes both tables."""
        from aragora.migrations.versions.v20260201000400_add_session_management import (
            up_fn,
            down_fn,
        )

        up_fn(backend)
        assert backend.table_exists("user_sessions")
        assert backend.table_exists("session_events")

        down_fn(backend)
        assert not backend.table_exists("user_sessions")
        assert not backend.table_exists("session_events")


# ---------------------------------------------------------------------------
# Full Lifecycle Tests with Runner
# ---------------------------------------------------------------------------


class TestMigrationRunnerIntegration:
    """Test migrations work correctly with the MigrationRunner."""

    def test_all_new_migrations_register(self, runner):
        """Test all new migrations can be registered."""
        from aragora.migrations.runner import _load_migrations

        _load_migrations(runner)

        # Check our new migrations are loaded
        versions = [m.version for m in runner._migrations]
        assert 20260201000000 in versions  # Debate metrics indexes
        assert 20260201000100 in versions  # Agent performance
        assert 20260201000200 in versions  # Decision receipts
        assert 20260201000300 in versions  # Rate limit tracking
        assert 20260201000400 in versions  # Session management

    def test_upgrade_applies_new_migrations(self, runner, backend):
        """Test upgrade applies our new migrations."""
        from aragora.migrations.runner import _load_migrations

        _load_migrations(runner)

        # Apply all migrations
        applied = runner.upgrade()

        # Should have applied our new migrations
        applied_versions = {m.version for m in applied}
        assert 20260201000100 in applied_versions  # Agent performance
        assert 20260201000200 in applied_versions  # Decision receipts

        # Verify tables exist
        assert backend.table_exists("agent_performance")
        assert backend.table_exists("decision_receipts")
        assert backend.table_exists("rate_limit_entries")
        assert backend.table_exists("user_sessions")

    def test_downgrade_rolls_back_new_migrations(self, runner, backend):
        """Test downgrade rolls back our new migrations."""
        from aragora.migrations.runner import _load_migrations

        _load_migrations(runner)

        # Apply all
        runner.upgrade()

        # Tables should exist
        assert backend.table_exists("agent_performance")

        # Rollback several steps
        runner.rollback_steps(steps=5)

        # Tables should be gone
        assert not backend.table_exists("agent_performance")
        assert not backend.table_exists("decision_receipts")

    def test_upgrade_is_idempotent(self, runner, backend):
        """Test upgrade called multiple times is idempotent."""
        from aragora.migrations.runner import _load_migrations

        _load_migrations(runner)

        # Apply twice
        first_applied = runner.upgrade()
        second_applied = runner.upgrade()

        # Second call should return empty (nothing new to apply)
        assert len(second_applied) == 0

        # Tables should still exist
        assert backend.table_exists("agent_performance")


# ---------------------------------------------------------------------------
# Edge Cases and Error Handling
# ---------------------------------------------------------------------------


class TestMigrationEdgeCases:
    """Test edge cases and error handling."""

    def test_migration_with_data_preservation(self, backend):
        """Test that migrations preserve existing data."""
        from aragora.migrations.versions.v20260201000100_add_agent_performance_tracking import (
            up_fn,
        )

        up_fn(backend)

        # Insert some test data
        backend.execute_write("""
            INSERT INTO agent_performance (id, agent_id, agent_type, operation, started_at)
            VALUES ('test1', 'agent1', 'claude', 'propose', '2026-01-01 00:00:00')
        """)

        # Run migration again (idempotent)
        up_fn(backend)

        # Data should still exist
        row = backend.fetch_one(
            "SELECT id, agent_id FROM agent_performance WHERE id = ?", ("test1",)
        )
        assert row is not None
        assert row[0] == "test1"
        assert row[1] == "agent1"

    def test_migrations_in_correct_order(self, runner):
        """Test migrations are sorted by version."""
        from aragora.migrations.runner import _load_migrations

        _load_migrations(runner)

        versions = [m.version for m in runner._migrations]

        # Should be sorted
        assert versions == sorted(versions)

        # Our migrations should be in correct order
        our_versions = [v for v in versions if v >= 20260201000000 and v <= 20260201000400]
        expected_order = [
            20260201000000,  # Indexes first (no deps)
            20260201000100,  # Agent performance
            20260201000200,  # Decision receipts
            20260201000300,  # Rate limiting
            20260201000400,  # Sessions
        ]
        assert our_versions == expected_order
