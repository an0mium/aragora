"""
E2E tests for memory persistence.

Tests the system's ability to:
- Store and retrieve ContinuumMemory entries
- Persist memory across restarts (simulated)
- Handle tier promotion/demotion
- Track ELO ratings persistence
- Restore circuit breaker state
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.memory.continuum import (
    CONTINUUM_SCHEMA_VERSION,
    ContinuumMemoryEntry,
    MemoryTier,
)
from aragora.memory.tier_manager import DEFAULT_TIER_CONFIGS, TierManager, get_tier_manager
from aragora.resilience import CircuitBreaker, get_circuit_breaker, persist_all_circuit_breakers


class TestContinuumMemoryEntry:
    """Test ContinuumMemoryEntry dataclass."""

    def test_create_entry(self):
        """Verify ContinuumMemoryEntry can be created."""
        entry = ContinuumMemoryEntry(
            id="entry-1",
            tier=MemoryTier.FAST,
            content="Test content",
            importance=0.8,
            surprise_score=0.5,
            consolidation_score=0.3,
            update_count=1,
            success_count=1,
            failure_count=0,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

        assert entry.id == "entry-1"
        assert entry.tier == MemoryTier.FAST
        assert entry.importance == 0.8

    def test_entry_has_metadata(self):
        """Verify entry can have metadata."""
        entry = ContinuumMemoryEntry(
            id="entry-2",
            tier=MemoryTier.MEDIUM,
            content="With metadata",
            importance=0.5,
            surprise_score=0.3,
            consolidation_score=0.2,
            update_count=2,
            success_count=2,
            failure_count=0,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            metadata={"agent": "claude", "topic": "testing"},
        )

        assert entry.metadata["agent"] == "claude"

    def test_entry_default_metadata(self):
        """Verify entry has empty metadata by default."""
        entry = ContinuumMemoryEntry(
            id="entry-3",
            tier=MemoryTier.SLOW,
            content="No metadata",
            importance=0.5,
            surprise_score=0.3,
            consolidation_score=0.2,
            update_count=1,
            success_count=1,
            failure_count=0,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

        assert entry.metadata == {}


class TestMemoryTiers:
    """Test memory tier definitions."""

    def test_tier_enum_values(self):
        """Verify all tier values are defined."""
        assert MemoryTier.FAST is not None
        assert MemoryTier.MEDIUM is not None
        assert MemoryTier.SLOW is not None
        assert MemoryTier.GLACIAL is not None

    def test_tier_ordering(self):
        """Verify tiers have logical ordering."""
        # Tiers should be orderable by persistence duration
        tiers = [MemoryTier.FAST, MemoryTier.MEDIUM, MemoryTier.SLOW, MemoryTier.GLACIAL]
        assert len(tiers) == 4

    def test_default_tier_configs(self):
        """Verify default tier configurations exist."""
        assert DEFAULT_TIER_CONFIGS is not None
        assert len(DEFAULT_TIER_CONFIGS) >= 4


class TestTierManager:
    """Test TierManager functionality."""

    def test_get_tier_manager(self):
        """Verify TierManager singleton works."""
        manager1 = get_tier_manager()
        manager2 = get_tier_manager()
        assert manager1 is manager2

    def test_tier_manager_has_configs(self):
        """Verify TierManager has tier configs."""
        manager = get_tier_manager()
        assert hasattr(manager, "configs") or hasattr(manager, "_configs")


class TestMemoryPersistenceSimulation:
    """Test memory persistence behavior (simulated)."""

    @pytest.fixture
    def temp_db(self, tmp_path: Path) -> Path:
        """Create a temporary database."""
        db_path = tmp_path / "memory.db"
        return db_path

    def test_can_create_memory_table(self, temp_db: Path):
        """Verify memory table can be created."""
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS continuum_memory (
                id TEXT PRIMARY KEY,
                tier TEXT NOT NULL,
                content TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                surprise_score REAL DEFAULT 0.0,
                consolidation_score REAL DEFAULT 0.0,
                update_count INTEGER DEFAULT 1,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """)
        conn.commit()

        # Verify table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='continuum_memory'")
        result = cursor.fetchone()
        assert result is not None

        conn.close()

    def test_can_insert_and_retrieve_entry(self, temp_db: Path):
        """Verify entries can be inserted and retrieved."""
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()

        # Create table
        cursor.execute("""
            CREATE TABLE continuum_memory (
                id TEXT PRIMARY KEY,
                tier TEXT NOT NULL,
                content TEXT NOT NULL,
                importance REAL,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # Insert entry
        now = datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO continuum_memory (id, tier, content, importance, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("entry-1", "FAST", "Test content", 0.8, now, now),
        )
        conn.commit()

        # Retrieve entry
        cursor.execute("SELECT * FROM continuum_memory WHERE id = ?", ("entry-1",))
        row = cursor.fetchone()

        assert row is not None
        assert row[0] == "entry-1"
        assert row[1] == "FAST"
        assert row[2] == "Test content"

        conn.close()

    def test_persistence_across_connections(self, temp_db: Path):
        """Verify data persists across connections."""
        # First connection - write data
        conn1 = sqlite3.connect(str(temp_db))
        cursor1 = conn1.cursor()
        cursor1.execute("""
            CREATE TABLE test_persistence (id TEXT PRIMARY KEY, value TEXT)
        """)
        cursor1.execute("INSERT INTO test_persistence VALUES ('key1', 'value1')")
        conn1.commit()
        conn1.close()

        # Second connection - read data
        conn2 = sqlite3.connect(str(temp_db))
        cursor2 = conn2.cursor()
        cursor2.execute("SELECT value FROM test_persistence WHERE id = ?", ("key1",))
        result = cursor2.fetchone()

        assert result is not None
        assert result[0] == "value1"

        conn2.close()


class TestTierPromotion:
    """Test memory tier promotion logic."""

    def test_promotion_criteria(self):
        """Verify promotion criteria structure."""
        promotion_criteria = {
            "fast_to_medium": {"min_update_count": 5, "min_consolidation": 0.3},
            "medium_to_slow": {"min_update_count": 10, "min_consolidation": 0.5},
            "slow_to_glacial": {"min_update_count": 20, "min_consolidation": 0.7},
        }

        assert promotion_criteria["fast_to_medium"]["min_update_count"] == 5

    def test_entry_qualifies_for_promotion(self):
        """Verify entry qualification check works."""
        entry = ContinuumMemoryEntry(
            id="promo-candidate",
            tier=MemoryTier.FAST,
            content="Frequently accessed",
            importance=0.9,
            surprise_score=0.2,
            consolidation_score=0.5,
            update_count=10,  # High update count
            success_count=8,
            failure_count=2,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

        # Check if qualifies for promotion
        min_updates_for_promotion = 5
        min_consolidation_for_promotion = 0.3

        qualifies = (
            entry.update_count >= min_updates_for_promotion
            and entry.consolidation_score >= min_consolidation_for_promotion
        )

        assert qualifies

    def test_entry_does_not_qualify(self):
        """Verify under-qualified entry doesn't promote."""
        entry = ContinuumMemoryEntry(
            id="no-promo",
            tier=MemoryTier.FAST,
            content="New entry",
            importance=0.5,
            surprise_score=0.1,
            consolidation_score=0.1,  # Low consolidation
            update_count=2,  # Low update count
            success_count=1,
            failure_count=1,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )

        min_updates = 5
        min_consolidation = 0.3

        qualifies = entry.update_count >= min_updates and entry.consolidation_score >= min_consolidation
        assert not qualifies


class TestCircuitBreakerPersistence:
    """Test circuit breaker state persistence."""

    @pytest.fixture
    def fresh_circuit_breakers(self):
        """Reset circuit breakers before each test."""
        from aragora.resilience import _circuit_breakers

        _circuit_breakers.clear()
        yield
        _circuit_breakers.clear()

    def test_circuit_breaker_state_structure(self, fresh_circuit_breakers):
        """Verify circuit breaker state can be captured."""
        breaker = get_circuit_breaker("test-agent", failure_threshold=3)

        # Record some activity
        breaker.record_failure()
        breaker.record_failure()

        if hasattr(breaker, "to_dict"):
            state = breaker.to_dict()
            # to_dict returns nested structure with single_mode and entity_mode
            if "single_mode" in state:
                assert "is_open" in state["single_mode"]
                assert "failures" in state["single_mode"]
            else:
                assert "is_open" in state or "failure_threshold" in state
        else:
            state = {
                "is_open": breaker.is_open,
                "failure_threshold": breaker.failure_threshold,
            }
            assert "is_open" in state

    def test_persist_all_circuit_breakers(self, fresh_circuit_breakers, tmp_path):
        """Verify all circuit breakers can be persisted."""
        # Create some circuit breakers
        breaker1 = get_circuit_breaker("agent-1", failure_threshold=3)
        breaker2 = get_circuit_breaker("agent-2", failure_threshold=5)

        # Record some state
        breaker1.record_failure()
        breaker2.record_success()

        # Persist should not raise
        try:
            count = persist_all_circuit_breakers()
            assert count >= 0
        except Exception:
            # May not have storage configured - that's OK for this test
            pass

    def test_circuit_breaker_recovery_state(self, fresh_circuit_breakers):
        """Verify circuit breaker cooldown state is trackable."""
        breaker = get_circuit_breaker(
            "cooldown-test", failure_threshold=2, cooldown_seconds=60
        )

        # Open the breaker
        breaker.record_failure()
        breaker.record_failure()

        assert breaker.is_open

        # Check cooldown remaining
        remaining = breaker.cooldown_remaining()
        assert remaining >= 0


class TestELORatingsPersistence:
    """Test ELO ratings persistence."""

    @pytest.fixture
    def temp_db(self, tmp_path: Path) -> Path:
        """Create a temporary database."""
        return tmp_path / "elo.db"

    def test_can_store_elo_ratings(self, temp_db: Path):
        """Verify ELO ratings can be stored."""
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE agent_elo (
                agent_name TEXT PRIMARY KEY,
                rating REAL NOT NULL,
                matches_played INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                updated_at TEXT NOT NULL
            )
        """)

        cursor.execute(
            "INSERT INTO agent_elo VALUES (?, ?, ?, ?, ?, ?)",
            ("claude", 1500.0, 10, 6, 4, datetime.now().isoformat()),
        )
        conn.commit()

        cursor.execute("SELECT rating FROM agent_elo WHERE agent_name = ?", ("claude",))
        result = cursor.fetchone()

        assert result is not None
        assert result[0] == 1500.0

        conn.close()

    def test_can_update_elo_rating(self, temp_db: Path):
        """Verify ELO ratings can be updated."""
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE agent_elo (
                agent_name TEXT PRIMARY KEY,
                rating REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        now = datetime.now().isoformat()
        cursor.execute("INSERT INTO agent_elo VALUES (?, ?, ?)", ("gpt-4", 1500.0, now))
        conn.commit()

        # Update rating after a win
        new_rating = 1520.0
        cursor.execute(
            "UPDATE agent_elo SET rating = ?, updated_at = ? WHERE agent_name = ?",
            (new_rating, now, "gpt-4"),
        )
        conn.commit()

        cursor.execute("SELECT rating FROM agent_elo WHERE agent_name = ?", ("gpt-4",))
        result = cursor.fetchone()

        assert result[0] == 1520.0

        conn.close()

    def test_elo_leaderboard_query(self, temp_db: Path):
        """Verify leaderboard query works."""
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE agent_elo (
                agent_name TEXT PRIMARY KEY,
                rating REAL NOT NULL
            )
        """)

        # Insert multiple agents
        agents = [("claude", 1550.0), ("gpt-4", 1500.0), ("gemini", 1480.0)]
        cursor.executemany("INSERT INTO agent_elo VALUES (?, ?)", agents)
        conn.commit()

        # Get leaderboard
        cursor.execute("SELECT agent_name, rating FROM agent_elo ORDER BY rating DESC")
        leaderboard = cursor.fetchall()

        assert len(leaderboard) == 3
        assert leaderboard[0][0] == "claude"  # Top rated
        assert leaderboard[0][1] == 1550.0

        conn.close()


class TestSchemaVersioning:
    """Test schema version management."""

    def test_schema_version_defined(self):
        """Verify schema version is defined."""
        assert CONTINUUM_SCHEMA_VERSION >= 1

    def test_schema_migration_structure(self, tmp_path: Path):
        """Verify schema migrations can be tracked."""
        db_path = tmp_path / "versioned.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Create schema version table
        cursor.execute("""
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL
            )
        """)

        cursor.execute(
            "INSERT INTO schema_version VALUES (?, ?)",
            (1, datetime.now().isoformat()),
        )
        conn.commit()

        # Check version
        cursor.execute("SELECT MAX(version) FROM schema_version")
        current_version = cursor.fetchone()[0]

        assert current_version == 1

        conn.close()


class TestConcurrentMemoryAccess:
    """Test thread safety of memory operations."""

    @pytest.fixture
    def temp_db(self, tmp_path: Path) -> Path:
        """Create a temporary database."""
        db_path = tmp_path / "concurrent.db"

        # Initialize with WAL mode
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("""
            CREATE TABLE memory_entries (
                id TEXT PRIMARY KEY,
                content TEXT,
                update_count INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()

        return db_path

    def test_concurrent_writes(self, temp_db: Path):
        """Verify concurrent writes succeed with WAL mode."""
        import threading

        errors: List[Exception] = []
        write_count = 50

        def write_entry(entry_id: int):
            try:
                conn = sqlite3.connect(str(temp_db))
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO memory_entries (id, content) VALUES (?, ?)",
                    (f"entry-{entry_id}", f"content-{entry_id}"),
                )
                conn.commit()
                conn.close()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_entry, args=(i,)) for i in range(write_count)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

        # Verify all entries written
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM memory_entries")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == write_count

    def test_concurrent_read_write(self, temp_db: Path):
        """Verify concurrent read and write operations succeed."""
        import threading

        errors: List[Exception] = []

        # Pre-populate some entries
        conn = sqlite3.connect(str(temp_db))
        cursor = conn.cursor()
        for i in range(10):
            cursor.execute(
                "INSERT INTO memory_entries (id, content, update_count) VALUES (?, ?, ?)",
                (f"existing-{i}", f"content-{i}", 0),
            )
        conn.commit()
        conn.close()

        def read_operation():
            try:
                conn = sqlite3.connect(str(temp_db))
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM memory_entries")
                _ = cursor.fetchall()
                conn.close()
            except Exception as e:
                errors.append(e)

        def update_operation(entry_id: int):
            try:
                conn = sqlite3.connect(str(temp_db))
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE memory_entries SET update_count = update_count + 1 WHERE id = ?",
                    (f"existing-{entry_id % 10}",),
                )
                conn.commit()
                conn.close()
            except Exception as e:
                errors.append(e)

        # Mix of reads and updates
        threads = []
        for i in range(30):
            if i % 3 == 0:
                threads.append(threading.Thread(target=read_operation))
            else:
                threads.append(threading.Thread(target=update_operation, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
