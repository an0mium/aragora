"""
E2E tests for persistence and data integrity.

Tests data persistence scenarios:
1. Debate survives server restart (conceptual)
2. Memory tier migrations (fast -> medium -> slow)
3. Transaction rollback on error
4. Concurrent write handling
5. Data integrity after migration
6. Database connection resilience
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from aragora.resilience import (
    CircuitBreaker,
    get_circuit_breaker,
    reset_all_circuit_breakers,
    init_circuit_breaker_persistence,
    persist_circuit_breaker,
    load_circuit_breakers,
    persist_all_circuit_breakers,
    cleanup_stale_persisted,
)


# =============================================================================
# Test Helpers
# =============================================================================


def create_test_db(db_path: Path) -> sqlite3.Connection:
    """Create a test SQLite database with basic schema."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS debates (
            id TEXT PRIMARY KEY,
            topic TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            workspace_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            debate_id TEXT,
            content TEXT,
            tier TEXT DEFAULT 'fast',
            importance REAL DEFAULT 0.5,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (debate_id) REFERENCES debates(id)
        )
    """
    )
    conn.commit()
    return conn


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_persistence.db"


@pytest.fixture
def test_db(temp_db_path):
    """Create a test database with schema."""
    conn = create_test_db(temp_db_path)
    yield conn
    conn.close()


@pytest.fixture
def cb_db_path():
    """Create a temporary path for circuit breaker persistence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "circuit_breaker.db"


@pytest.fixture(autouse=True)
def reset_circuit_breakers():
    """Reset circuit breakers before and after each test."""
    reset_all_circuit_breakers()
    yield
    reset_all_circuit_breakers()


# =============================================================================
# Database Transaction Tests
# =============================================================================


class TestDatabaseTransactions:
    """Tests for database transaction handling."""

    def test_transaction_commit_persists_data(self, test_db, temp_db_path):
        """E2E: Committed transaction should persist data."""
        # Insert within transaction
        test_db.execute(
            "INSERT INTO debates (id, topic) VALUES (?, ?)",
            ("debate-1", "Test Topic"),
        )
        test_db.commit()

        # Verify data persists in new connection
        new_conn = sqlite3.connect(str(temp_db_path))
        cursor = new_conn.execute("SELECT topic FROM debates WHERE id = ?", ("debate-1",))
        result = cursor.fetchone()
        new_conn.close()

        assert result is not None
        assert result[0] == "Test Topic"

    def test_transaction_rollback_discards_data(self, test_db, temp_db_path):
        """E2E: Rolled back transaction should not persist data."""
        # Insert without commit
        test_db.execute(
            "INSERT INTO debates (id, topic) VALUES (?, ?)",
            ("debate-rollback", "Rollback Topic"),
        )
        test_db.rollback()

        # Verify data was not persisted
        cursor = test_db.execute(
            "SELECT * FROM debates WHERE id = ?", ("debate-rollback",)
        )
        result = cursor.fetchone()

        assert result is None

    def test_foreign_key_constraint_on_memory(self, test_db):
        """E2E: Foreign key constraints should be enforced on memories."""
        # SQLite foreign keys are disabled by default
        test_db.execute("PRAGMA foreign_keys = ON")

        # This should work - debate exists
        test_db.execute(
            "INSERT INTO debates (id, topic) VALUES (?, ?)",
            ("debate-fk", "FK Test"),
        )
        test_db.execute(
            "INSERT INTO memories (id, debate_id, content) VALUES (?, ?, ?)",
            ("memory-1", "debate-fk", "Test content"),
        )
        test_db.commit()

        # Verify memory was inserted
        cursor = test_db.execute("SELECT * FROM memories WHERE id = ?", ("memory-1",))
        assert cursor.fetchone() is not None


# =============================================================================
# Concurrent Write Tests
# =============================================================================


class TestConcurrentWrites:
    """Tests for concurrent database write handling."""

    def test_concurrent_inserts_all_succeed(self, temp_db_path):
        """E2E: Concurrent inserts should all succeed with WAL mode."""
        # Create database with WAL mode
        conn = create_test_db(temp_db_path)
        conn.close()

        results = []
        errors = []

        def insert_debate(debate_id: str):
            try:
                # Each thread gets its own connection
                thread_conn = sqlite3.connect(str(temp_db_path), timeout=30.0)
                thread_conn.execute("PRAGMA journal_mode=WAL;")
                thread_conn.execute(
                    "INSERT INTO debates (id, topic) VALUES (?, ?)",
                    (debate_id, f"Topic {debate_id}"),
                )
                thread_conn.commit()
                thread_conn.close()
                results.append(debate_id)
            except Exception as e:
                errors.append((debate_id, str(e)))

        # Run concurrent inserts
        threads = []
        for i in range(10):
            t = threading.Thread(target=insert_debate, args=(f"concurrent-{i}",))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All inserts should succeed
        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 10

        # Verify all data persisted
        verify_conn = sqlite3.connect(str(temp_db_path))
        cursor = verify_conn.execute("SELECT COUNT(*) FROM debates")
        count = cursor.fetchone()[0]
        verify_conn.close()

        assert count == 10

    def test_concurrent_updates_serialize_correctly(self, temp_db_path):
        """E2E: Concurrent updates should serialize without data loss."""
        # Create database and initial record
        conn = create_test_db(temp_db_path)
        conn.execute(
            "INSERT INTO debates (id, topic, status) VALUES (?, ?, ?)",
            ("update-test", "Update Test", "pending"),
        )
        conn.commit()
        conn.close()

        update_count = {"value": 0}
        lock = threading.Lock()

        def update_debate(new_status: str):
            thread_conn = sqlite3.connect(str(temp_db_path), timeout=30.0)
            thread_conn.execute("PRAGMA journal_mode=WAL;")
            thread_conn.execute(
                "UPDATE debates SET status = ? WHERE id = ?",
                (new_status, "update-test"),
            )
            thread_conn.commit()
            thread_conn.close()
            with lock:
                update_count["value"] += 1

        # Run concurrent updates
        statuses = ["running", "completed", "archived", "failed", "pending"]
        threads = []
        for status in statuses:
            t = threading.Thread(target=update_debate, args=(status,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All updates should complete
        assert update_count["value"] == 5

        # Final status should be one of the valid statuses
        verify_conn = sqlite3.connect(str(temp_db_path))
        cursor = verify_conn.execute(
            "SELECT status FROM debates WHERE id = ?", ("update-test",)
        )
        final_status = cursor.fetchone()[0]
        verify_conn.close()

        assert final_status in statuses


# =============================================================================
# Memory Tier Tests
# =============================================================================


class TestMemoryTiers:
    """Tests for memory tier functionality."""

    def test_memory_tier_values(self, test_db):
        """E2E: Memory should store tier information correctly."""
        tiers = ["fast", "medium", "slow", "glacial"]

        for i, tier in enumerate(tiers):
            test_db.execute(
                "INSERT INTO memories (id, debate_id, content, tier, importance) VALUES (?, ?, ?, ?, ?)",
                (f"memory-tier-{i}", None, f"Content {tier}", tier, 0.5 + i * 0.1),
            )
        test_db.commit()

        # Query by tier
        for tier in tiers:
            cursor = test_db.execute(
                "SELECT content FROM memories WHERE tier = ?", (tier,)
            )
            result = cursor.fetchone()
            assert result is not None
            assert tier in result[0]

    def test_importance_ordering(self, test_db):
        """E2E: Memories should be retrievable ordered by importance."""
        # Insert memories with varying importance
        for i in range(5):
            test_db.execute(
                "INSERT INTO memories (id, content, importance) VALUES (?, ?, ?)",
                (f"mem-importance-{i}", f"Content {i}", i * 0.2),
            )
        test_db.commit()

        # Query ordered by importance (highest first)
        cursor = test_db.execute(
            "SELECT id, importance FROM memories ORDER BY importance DESC"
        )
        results = cursor.fetchall()

        # Verify descending order
        importances = [r[1] for r in results]
        assert importances == sorted(importances, reverse=True)


# =============================================================================
# Circuit Breaker Persistence Tests
# =============================================================================


class TestCircuitBreakerPersistence:
    """Tests for circuit breaker state persistence."""

    def test_init_creates_database(self, cb_db_path):
        """E2E: init_circuit_breaker_persistence should create database."""
        init_circuit_breaker_persistence(str(cb_db_path))

        assert cb_db_path.exists()

        # Verify table exists
        conn = sqlite3.connect(str(cb_db_path))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='circuit_breakers'"
        )
        result = cursor.fetchone()
        conn.close()

        assert result is not None

    def test_persist_and_load_circuit_breaker(self, cb_db_path):
        """E2E: Circuit breaker state should persist and load correctly."""
        init_circuit_breaker_persistence(str(cb_db_path))

        # Create and configure a circuit breaker
        cb = get_circuit_breaker("test-service", failure_threshold=5, cooldown_seconds=30.0)

        # Record some failures
        for _ in range(3):
            cb.record_failure()

        # Persist
        persist_circuit_breaker("test-service", cb)

        # Reset and reload
        reset_all_circuit_breakers()
        count = load_circuit_breakers()

        assert count >= 1

        # Get the loaded circuit breaker
        loaded_cb = get_circuit_breaker("test-service")

        # State should be preserved
        assert loaded_cb.failures == 3

    def test_persist_all_circuit_breakers(self, cb_db_path):
        """E2E: persist_all should save all registered circuit breakers."""
        init_circuit_breaker_persistence(str(cb_db_path))

        # Create multiple circuit breakers
        cb1 = get_circuit_breaker("service-1")
        cb2 = get_circuit_breaker("service-2")
        cb3 = get_circuit_breaker("service-3")

        cb1.record_failure()
        cb2.record_failure()
        cb2.record_failure()

        # Persist all
        count = persist_all_circuit_breakers()
        # At least our 3 services should be persisted
        assert count >= 3

        # Verify in database
        conn = sqlite3.connect(str(cb_db_path))
        cursor = conn.execute("SELECT COUNT(*) FROM circuit_breakers")
        db_count = cursor.fetchone()[0]
        conn.close()

        # At least our services should be in DB
        assert db_count >= 3

    def test_cleanup_stale_entries(self, cb_db_path):
        """E2E: cleanup should remove stale circuit breaker entries."""
        init_circuit_breaker_persistence(str(cb_db_path))

        # Insert an old entry directly
        conn = sqlite3.connect(str(cb_db_path))
        conn.execute(
            """
            INSERT INTO circuit_breakers (name, state_json, failure_threshold, cooldown_seconds, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """,
            ("stale-service", "{}", 3, 60.0, "2020-01-01T00:00:00"),
        )
        conn.commit()
        conn.close()

        # Cleanup with short max age
        deleted = cleanup_stale_persisted(max_age_hours=0.001)

        assert deleted >= 1


# =============================================================================
# Data Integrity Tests
# =============================================================================


class TestDataIntegrity:
    """Tests for data integrity guarantees."""

    def test_unique_constraint_enforced(self, test_db):
        """E2E: Primary key uniqueness should be enforced."""
        test_db.execute(
            "INSERT INTO debates (id, topic) VALUES (?, ?)",
            ("unique-test", "First Topic"),
        )
        test_db.commit()

        # Duplicate insert should fail
        with pytest.raises(sqlite3.IntegrityError):
            test_db.execute(
                "INSERT INTO debates (id, topic) VALUES (?, ?)",
                ("unique-test", "Duplicate Topic"),
            )
            test_db.commit()

    def test_not_null_constraint_enforced(self, test_db):
        """E2E: NOT NULL constraints should be enforced."""
        with pytest.raises(sqlite3.IntegrityError):
            test_db.execute(
                "INSERT INTO debates (id, topic) VALUES (?, ?)",
                ("null-test", None),  # topic is NOT NULL
            )
            test_db.commit()

    def test_default_values_applied(self, test_db):
        """E2E: Default values should be applied correctly."""
        test_db.execute(
            "INSERT INTO debates (id, topic) VALUES (?, ?)",
            ("default-test", "Default Test"),
        )
        test_db.commit()

        cursor = test_db.execute(
            "SELECT status, created_at FROM debates WHERE id = ?", ("default-test",)
        )
        result = cursor.fetchone()

        assert result[0] == "pending"  # Default status
        assert result[1] is not None  # Default timestamp


# =============================================================================
# Connection Resilience Tests
# =============================================================================


class TestConnectionResilience:
    """Tests for database connection resilience."""

    def test_reconnect_after_close(self, temp_db_path):
        """E2E: Should be able to reconnect after connection close."""
        # Create and close connection
        conn = create_test_db(temp_db_path)
        conn.execute(
            "INSERT INTO debates (id, topic) VALUES (?, ?)",
            ("reconnect-test", "Reconnect Test"),
        )
        conn.commit()
        conn.close()

        # Reconnect and verify data
        new_conn = sqlite3.connect(str(temp_db_path))
        cursor = new_conn.execute(
            "SELECT topic FROM debates WHERE id = ?", ("reconnect-test",)
        )
        result = cursor.fetchone()
        new_conn.close()

        assert result is not None
        assert result[0] == "Reconnect Test"

    def test_busy_timeout_prevents_lock_errors(self, temp_db_path):
        """E2E: Busy timeout should prevent immediate lock errors."""
        # Create database
        conn1 = create_test_db(temp_db_path)

        # Start a transaction without committing
        conn1.execute(
            "INSERT INTO debates (id, topic) VALUES (?, ?)",
            ("lock-test", "Lock Test"),
        )

        # Second connection with timeout should wait
        conn2 = sqlite3.connect(str(temp_db_path), timeout=5.0)
        conn2.execute("PRAGMA journal_mode=WAL;")

        # Commit first transaction
        conn1.commit()

        # Now second connection should work
        cursor = conn2.execute("SELECT COUNT(*) FROM debates")
        result = cursor.fetchone()

        conn1.close()
        conn2.close()

        assert result[0] >= 1

    def test_wal_mode_enables_concurrent_reads(self, temp_db_path):
        """E2E: WAL mode should allow concurrent reads during writes."""
        # Create database with WAL mode
        conn = create_test_db(temp_db_path)
        conn.execute(
            "INSERT INTO debates (id, topic) VALUES (?, ?)",
            ("wal-test", "WAL Test"),
        )
        conn.commit()

        # Start a long-running write transaction
        conn.execute(
            "INSERT INTO debates (id, topic) VALUES (?, ?)",
            ("wal-write", "WAL Write"),
        )
        # Note: Not committed yet

        # Concurrent read should still work in WAL mode
        read_conn = sqlite3.connect(str(temp_db_path))
        cursor = read_conn.execute("SELECT COUNT(*) FROM debates")
        result = cursor.fetchone()
        read_conn.close()

        # Should see committed data
        assert result[0] >= 1

        conn.commit()
        conn.close()
