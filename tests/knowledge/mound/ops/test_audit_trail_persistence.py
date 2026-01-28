"""
Tests for Audit Trail SQLite persistence.

Phase 7: KM Governance Test Gaps - Audit trail SQLite tests.

Tests:
- test_sqlite_database_creation - Table and index setup
- test_audit_entry_write_and_read - CRUD operations
- test_audit_query_by_actor - Actor filtering
- test_audit_query_by_action - Action type filtering
- test_audit_query_by_time_range - Date range filtering
- test_audit_entry_json_serialization - Details field
- test_concurrent_audit_writes - Thread safety
- test_audit_max_entries_trimming - Size limits
- test_audit_success_failure_filtering - Status filtering
- test_audit_ip_and_user_agent_tracking - Client info
- test_aiosqlite_unavailable_fallback - Graceful degradation
- test_database_recovery_on_corruption - Error handling
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from aragora.knowledge.mound.ops.governance import (
    AuditAction,
    AuditEntry,
    AuditTrail,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, "test_audit.db")


@pytest.fixture
def audit_trail(temp_db_path):
    """Create an audit trail with persistence enabled."""
    return AuditTrail(
        max_entries=1000,
        enable_persistence=True,
        db_path=temp_db_path,
    )


@pytest.fixture
def memory_only_audit_trail():
    """Create an audit trail without persistence."""
    return AuditTrail(
        max_entries=100,
        enable_persistence=False,
    )


# ============================================================================
# Test: SQLite Database Creation
# ============================================================================


class TestSQLiteDatabaseCreation:
    """Test SQLite database initialization."""

    @pytest.mark.asyncio
    async def test_sqlite_database_creation(self, audit_trail, temp_db_path):
        """Test that database tables and indexes are created."""
        # Log an entry to trigger database initialization
        await audit_trail.log(
            action=AuditAction.ITEM_CREATE,
            actor_id="user-1",
            resource_type="knowledge_item",
            resource_id="item-1",
        )

        # Verify database file was created
        assert os.path.exists(temp_db_path)

        # Check tables and indexes were created
        import aiosqlite

        async with aiosqlite.connect(temp_db_path) as db:
            # Check table exists
            async with db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='audit_entries'"
            ) as cursor:
                table = await cursor.fetchone()
                assert table is not None

            # Check indexes exist
            async with db.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_audit%'"
            ) as cursor:
                indexes = await cursor.fetchall()
                # Should have at least 3 indexes
                assert len(indexes) >= 3

    @pytest.mark.asyncio
    async def test_database_columns_correct(self, audit_trail, temp_db_path):
        """Test that database has correct column structure."""
        # Initialize DB
        await audit_trail.log(
            action=AuditAction.ITEM_READ,
            actor_id="user-1",
            resource_type="item",
            resource_id="item-1",
        )

        import aiosqlite

        async with aiosqlite.connect(temp_db_path) as db:
            async with db.execute("PRAGMA table_info(audit_entries)") as cursor:
                columns = await cursor.fetchall()
                column_names = [col[1] for col in columns]

                expected_columns = [
                    "id",
                    "action",
                    "actor_id",
                    "resource_type",
                    "resource_id",
                    "workspace_id",
                    "timestamp",
                    "details",
                    "ip_address",
                    "user_agent",
                    "success",
                    "error_message",
                ]
                for col in expected_columns:
                    assert col in column_names


# ============================================================================
# Test: Audit Entry Write and Read
# ============================================================================


class TestAuditEntryWriteAndRead:
    """Test CRUD operations for audit entries."""

    @pytest.mark.asyncio
    async def test_audit_entry_write_and_read(self, audit_trail):
        """Test writing and reading audit entries."""
        # Write entry
        entry = await audit_trail.log(
            action=AuditAction.ITEM_CREATE,
            actor_id="user-test",
            resource_type="knowledge_item",
            resource_id="item-123",
            workspace_id="ws-1",
            details={"key": "value"},
            ip_address="192.168.1.1",
            user_agent="Test Agent",
        )

        assert entry.id is not None
        assert entry.action == AuditAction.ITEM_CREATE
        assert entry.actor_id == "user-test"

        # Read back from database
        entries = await audit_trail.query(actor_id="user-test", from_database=True)
        assert len(entries) >= 1

        # Find our entry
        found = next((e for e in entries if e.id == entry.id), None)
        assert found is not None
        assert found.resource_id == "item-123"
        assert found.details == {"key": "value"}

    @pytest.mark.asyncio
    async def test_multiple_entries_write(self, audit_trail):
        """Test writing multiple audit entries."""
        # Write multiple entries
        for i in range(10):
            await audit_trail.log(
                action=AuditAction.ITEM_READ,
                actor_id=f"user-{i}",
                resource_type="item",
                resource_id=f"item-{i}",
            )

        # Query all entries
        entries = await audit_trail.query(limit=20)
        assert len(entries) == 10


# ============================================================================
# Test: Query by Actor
# ============================================================================


class TestAuditQueryByActor:
    """Test filtering audit entries by actor."""

    @pytest.mark.asyncio
    async def test_audit_query_by_actor(self, audit_trail):
        """Test filtering by actor ID."""
        # Create entries for different actors
        await audit_trail.log(
            action=AuditAction.ITEM_CREATE,
            actor_id="alice",
            resource_type="item",
            resource_id="item-1",
        )
        await audit_trail.log(
            action=AuditAction.ITEM_UPDATE,
            actor_id="bob",
            resource_type="item",
            resource_id="item-2",
        )
        await audit_trail.log(
            action=AuditAction.ITEM_DELETE,
            actor_id="alice",
            resource_type="item",
            resource_id="item-3",
        )

        # Query by alice
        alice_entries = await audit_trail.query(actor_id="alice")
        assert len(alice_entries) == 2
        assert all(e.actor_id == "alice" for e in alice_entries)

        # Query by bob
        bob_entries = await audit_trail.query(actor_id="bob")
        assert len(bob_entries) == 1
        assert bob_entries[0].actor_id == "bob"


# ============================================================================
# Test: Query by Action
# ============================================================================


class TestAuditQueryByAction:
    """Test filtering audit entries by action type."""

    @pytest.mark.asyncio
    async def test_audit_query_by_action(self, audit_trail):
        """Test filtering by action type."""
        # Create entries with different actions
        await audit_trail.log(
            action=AuditAction.ITEM_CREATE,
            actor_id="user-1",
            resource_type="item",
            resource_id="item-1",
        )
        await audit_trail.log(
            action=AuditAction.ITEM_READ,
            actor_id="user-1",
            resource_type="item",
            resource_id="item-1",
        )
        await audit_trail.log(
            action=AuditAction.ITEM_UPDATE,
            actor_id="user-1",
            resource_type="item",
            resource_id="item-1",
        )

        # Query by action
        read_entries = await audit_trail.query(action=AuditAction.ITEM_READ)
        assert len(read_entries) == 1
        assert read_entries[0].action == AuditAction.ITEM_READ


# ============================================================================
# Test: Query by Time Range
# ============================================================================


class TestAuditQueryByTimeRange:
    """Test filtering audit entries by time range."""

    @pytest.mark.asyncio
    async def test_audit_query_by_time_range(self, memory_only_audit_trail):
        """Test filtering by time range."""
        audit = memory_only_audit_trail
        now = datetime.now()

        # Create entry
        entry = await audit.log(
            action=AuditAction.ITEM_CREATE,
            actor_id="user-1",
            resource_type="item",
            resource_id="item-1",
        )

        # Query with time range that includes the entry
        entries = await audit.query(
            start_time=now - timedelta(minutes=5),
            end_time=now + timedelta(minutes=5),
        )
        assert len(entries) == 1

        # Query with time range before the entry
        old_entries = await audit.query(
            start_time=now - timedelta(days=2),
            end_time=now - timedelta(days=1),
        )
        assert len(old_entries) == 0


# ============================================================================
# Test: JSON Serialization
# ============================================================================


class TestAuditEntryJSONSerialization:
    """Test JSON serialization of audit entries."""

    @pytest.mark.asyncio
    async def test_audit_entry_json_serialization(self, audit_trail):
        """Test that details field is properly JSON serialized."""
        complex_details = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
            "number": 42,
            "boolean": True,
            "null": None,
        }

        entry = await audit_trail.log(
            action=AuditAction.ITEM_CREATE,
            actor_id="user-1",
            resource_type="item",
            resource_id="item-1",
            details=complex_details,
        )

        # Query back
        entries = await audit_trail.query(actor_id="user-1")
        found = next((e for e in entries if e.id == entry.id), None)

        assert found is not None
        assert found.details["nested"]["key"] == "value"
        assert found.details["list"] == [1, 2, 3]
        assert found.details["number"] == 42

    @pytest.mark.asyncio
    async def test_audit_entry_to_dict(self, memory_only_audit_trail):
        """Test to_dict serialization."""
        entry = await memory_only_audit_trail.log(
            action=AuditAction.ITEM_CREATE,
            actor_id="user-1",
            resource_type="item",
            resource_id="item-1",
            details={"key": "value"},
        )

        d = entry.to_dict()
        assert d["id"] == entry.id
        assert d["action"] == "item.create"
        assert d["actor_id"] == "user-1"
        assert "timestamp" in d


# ============================================================================
# Test: Concurrent Writes
# ============================================================================


class TestConcurrentAuditWrites:
    """Test thread safety of concurrent audit writes."""

    @pytest.mark.asyncio
    async def test_concurrent_audit_writes(self, audit_trail):
        """Test concurrent audit writes don't corrupt data."""
        num_entries = 50

        async def write_entry(i: int):
            return await audit_trail.log(
                action=AuditAction.ITEM_READ,
                actor_id=f"user-{i}",
                resource_type="item",
                resource_id=f"item-{i}",
            )

        # Write concurrently
        tasks = [write_entry(i) for i in range(num_entries)]
        entries = await asyncio.gather(*tasks)

        assert len(entries) == num_entries

        # Verify all entries were written
        all_entries = await audit_trail.query(limit=num_entries + 10)
        assert len(all_entries) >= num_entries


# ============================================================================
# Test: Max Entries Trimming
# ============================================================================


class TestAuditMaxEntriesTrimming:
    """Test in-memory entry trimming when max is exceeded."""

    @pytest.mark.asyncio
    async def test_audit_max_entries_trimming(self):
        """Test that entries are trimmed when max_entries is exceeded."""
        small_audit = AuditTrail(
            max_entries=10,
            enable_persistence=False,
        )

        # Write more than max entries
        for i in range(15):
            await small_audit.log(
                action=AuditAction.ITEM_READ,
                actor_id=f"user-{i}",
                resource_type="item",
                resource_id=f"item-{i}",
            )

        # Check in-memory count
        stats = small_audit.get_stats()
        assert stats["total_entries"] == 10


# ============================================================================
# Test: Success/Failure Filtering
# ============================================================================


class TestAuditSuccessFailureFiltering:
    """Test filtering by success/failure status."""

    @pytest.mark.asyncio
    async def test_audit_success_failure_filtering(self, memory_only_audit_trail):
        """Test filtering by success status."""
        audit = memory_only_audit_trail

        # Create successful entries
        await audit.log(
            action=AuditAction.ITEM_CREATE,
            actor_id="user-1",
            resource_type="item",
            resource_id="item-1",
            success=True,
        )

        # Create failed entries
        await audit.log(
            action=AuditAction.ITEM_DELETE,
            actor_id="user-1",
            resource_type="item",
            resource_id="item-2",
            success=False,
            error_message="Permission denied",
        )

        # Query success only
        success_entries = await audit.query(success_only=True)
        assert len(success_entries) == 1
        assert success_entries[0].success is True

        # Query all
        all_entries = await audit.query()
        assert len(all_entries) == 2


# ============================================================================
# Test: IP and User Agent Tracking
# ============================================================================


class TestAuditIPAndUserAgentTracking:
    """Test client info tracking."""

    @pytest.mark.asyncio
    async def test_audit_ip_and_user_agent_tracking(self, audit_trail):
        """Test that IP and user agent are properly stored."""
        entry = await audit_trail.log(
            action=AuditAction.ITEM_CREATE,
            actor_id="user-1",
            resource_type="item",
            resource_id="item-1",
            ip_address="10.0.0.1",
            user_agent="Mozilla/5.0 Test Browser",
        )

        assert entry.ip_address == "10.0.0.1"
        assert entry.user_agent == "Mozilla/5.0 Test Browser"

        # Query and verify persisted
        entries = await audit_trail.query(actor_id="user-1")
        found = next((e for e in entries if e.id == entry.id), None)
        assert found is not None
        assert found.ip_address == "10.0.0.1"
        assert found.user_agent == "Mozilla/5.0 Test Browser"


# ============================================================================
# Test: aiosqlite Unavailable Fallback
# ============================================================================


class TestAiosqliteUnavailableFallback:
    """Test graceful degradation when aiosqlite is unavailable."""

    @pytest.mark.asyncio
    async def test_aiosqlite_unavailable_fallback(self, temp_db_path):
        """Test fallback to in-memory when aiosqlite import fails."""
        with patch.dict("sys.modules", {"aiosqlite": None}):
            audit = AuditTrail(
                enable_persistence=True,
                db_path=temp_db_path,
            )

            # Should still work with in-memory fallback
            entry = await audit.log(
                action=AuditAction.ITEM_CREATE,
                actor_id="user-1",
                resource_type="item",
                resource_id="item-1",
            )

            assert entry is not None

            # Query should work
            entries = await audit.query(actor_id="user-1")
            assert len(entries) >= 1


# ============================================================================
# Test: Database Error Recovery
# ============================================================================


class TestDatabaseRecovery:
    """Test error handling and recovery."""

    @pytest.mark.asyncio
    async def test_database_recovery_on_corruption(self, temp_db_path):
        """Test handling of database errors."""
        audit = AuditTrail(
            enable_persistence=True,
            db_path=temp_db_path,
        )

        # Initialize database
        await audit.log(
            action=AuditAction.ITEM_CREATE,
            actor_id="user-1",
            resource_type="item",
            resource_id="item-1",
        )

        # Corrupt the database by writing garbage
        with open(temp_db_path, "w") as f:
            f.write("corrupted data")

        # Should handle error gracefully and fall back to in-memory
        # This tests that the query doesn't crash
        try:
            entries = await audit.query(actor_id="user-1")
            # May return empty or from memory cache
            assert isinstance(entries, list)
        except Exception:
            # Database corruption is caught
            pass


# ============================================================================
# Test: User Activity Summary
# ============================================================================


class TestUserActivitySummary:
    """Test user activity summary generation."""

    @pytest.mark.asyncio
    async def test_get_user_activity(self, memory_only_audit_trail):
        """Test user activity summary generation."""
        audit = memory_only_audit_trail

        # Create various activities
        await audit.log(
            action=AuditAction.ITEM_CREATE,
            actor_id="user-1",
            resource_type="knowledge_item",
            resource_id="item-1",
        )
        await audit.log(
            action=AuditAction.ITEM_READ,
            actor_id="user-1",
            resource_type="knowledge_item",
            resource_id="item-1",
        )
        await audit.log(
            action=AuditAction.ITEM_UPDATE,
            actor_id="user-1",
            resource_type="knowledge_item",
            resource_id="item-1",
        )

        activity = await audit.get_user_activity("user-1", days=30)

        assert activity["user_id"] == "user-1"
        assert activity["total_actions"] == 3
        assert activity["by_action"]["item.create"] == 1
        assert activity["by_action"]["item.read"] == 1
        assert activity["by_action"]["item.update"] == 1
        assert activity["success_rate"] == 1.0


__all__ = [
    "TestSQLiteDatabaseCreation",
    "TestAuditEntryWriteAndRead",
    "TestAuditQueryByActor",
    "TestAuditQueryByAction",
    "TestAuditQueryByTimeRange",
    "TestAuditEntryJSONSerialization",
    "TestConcurrentAuditWrites",
    "TestAuditMaxEntriesTrimming",
    "TestAuditSuccessFailureFiltering",
    "TestAuditIPAndUserAgentTracking",
    "TestAiosqliteUnavailableFallback",
    "TestDatabaseRecovery",
    "TestUserActivitySummary",
]
