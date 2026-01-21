"""
Tests for Decision Result Store.

Tests cover:
- Basic CRUD operations
- TTL expiration
- LRU eviction
- In-memory caching
- Persistence across restarts
"""

import pytest
import time
from pathlib import Path

from aragora.storage.decision_result_store import (
    DecisionResultStore,
    DecisionResultEntry,
    get_decision_result_store,
    reset_decision_result_store,
)


class TestDecisionResultEntry:
    """Tests for DecisionResultEntry dataclass."""

    def test_create_entry(self):
        """Should create entry with defaults."""
        entry = DecisionResultEntry(
            request_id="req-123",
            status="completed",
            result={"answer": "test"},
        )

        assert entry.request_id == "req-123"
        assert entry.status == "completed"
        assert entry.result == {"answer": "test"}
        assert entry.created_at > 0

    def test_entry_to_dict(self):
        """Should convert to dictionary."""
        entry = DecisionResultEntry(
            request_id="req-123",
            status="completed",
            result={"answer": "test"},
            completed_at="2024-01-01T00:00:00",
        )

        data = entry.to_dict()
        assert data["request_id"] == "req-123"
        assert data["status"] == "completed"
        assert data["completed_at"] == "2024-01-01T00:00:00"

    def test_entry_from_dict(self):
        """Should create from dictionary."""
        data = {
            "request_id": "req-456",
            "status": "failed",
            "result": {"error": "test"},
            "error": "Something went wrong",
        }

        entry = DecisionResultEntry.from_dict(data)
        assert entry.request_id == "req-456"
        assert entry.status == "failed"
        assert entry.error == "Something went wrong"

    def test_entry_expiration(self):
        """Should detect expired entries."""
        # Create entry with 0 TTL (already expired)
        entry = DecisionResultEntry(
            request_id="req-123",
            status="completed",
            result={},
            ttl_seconds=0,
            created_at=time.time() - 1,
        )

        assert entry.is_expired is True

        # Create entry with long TTL
        entry2 = DecisionResultEntry(
            request_id="req-456",
            status="completed",
            result={},
            ttl_seconds=3600,
        )

        assert entry2.is_expired is False


class TestDecisionResultStore:
    """Tests for DecisionResultStore."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary database path."""
        return tmp_path / "test_decisions.db"

    @pytest.fixture
    def store(self, temp_db):
        """Create a fresh store instance."""
        return DecisionResultStore(
            db_path=temp_db,
            ttl_seconds=3600,
            max_entries=100,
            cache_size=10,
        )

    def test_save_and_get(self, store):
        """Should save and retrieve decision results."""
        store.save("req-123", {
            "status": "completed",
            "result": {"answer": "42"},
            "completed_at": "2024-01-01T00:00:00",
        })

        result = store.get("req-123")
        assert result is not None
        assert result["request_id"] == "req-123"
        assert result["status"] == "completed"
        assert result["result"]["answer"] == "42"

    def test_get_nonexistent(self, store):
        """Should return None for nonexistent entries."""
        result = store.get("nonexistent")
        assert result is None

    def test_get_status(self, store):
        """Should return status for polling."""
        store.save("req-123", {
            "status": "completed",
            "completed_at": "2024-01-01T00:00:00",
        })

        status = store.get_status("req-123")
        assert status["request_id"] == "req-123"
        assert status["status"] == "completed"
        assert status["completed_at"] == "2024-01-01T00:00:00"

    def test_get_status_not_found(self, store):
        """Should return not_found status for missing entries."""
        status = store.get_status("nonexistent")
        assert status["status"] == "not_found"

    def test_list_recent(self, store):
        """Should list recent decisions."""
        for i in range(5):
            store.save(f"req-{i}", {"status": "completed"})

        recent = store.list_recent(limit=3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0]["request_id"] == "req-4"

    def test_count(self, store):
        """Should count entries."""
        assert store.count() == 0

        for i in range(3):
            store.save(f"req-{i}", {"status": "completed"})

        assert store.count() == 3

    def test_delete(self, store):
        """Should delete entries."""
        store.save("req-123", {"status": "completed"})
        assert store.get("req-123") is not None

        deleted = store.delete("req-123")
        assert deleted is True
        assert store.get("req-123") is None

    def test_delete_nonexistent(self, store):
        """Should return False when deleting nonexistent entry."""
        deleted = store.delete("nonexistent")
        assert deleted is False

    def test_ttl_expiration(self, tmp_path):
        """Should expire entries after TTL."""
        store = DecisionResultStore(
            db_path=tmp_path / "ttl_test.db",
            ttl_seconds=1,  # 1 second TTL
        )

        store.save("req-123", {"status": "completed"})
        assert store.get("req-123") is not None

        # Wait for expiration
        time.sleep(1.5)

        # Should be expired now
        assert store.get("req-123") is None

    def test_lru_eviction(self, tmp_path):
        """Should evict oldest entries when max reached."""
        store = DecisionResultStore(
            db_path=tmp_path / "lru_test.db",
            max_entries=5,
            cache_size=5,
        )

        # Add more than max entries
        for i in range(10):
            store.save(f"req-{i}", {"status": "completed"})
            time.sleep(0.01)  # Small delay to ensure ordering

        # Should have at most max_entries
        assert store.count() <= 5

        # Oldest entries should be evicted, newest should remain
        assert store.get("req-9") is not None

    def test_cache_hit(self, store):
        """Should serve from cache on subsequent reads."""
        store.save("req-123", {"status": "completed"})

        # First read populates cache
        result1 = store.get("req-123")
        # Second read should hit cache
        result2 = store.get("req-123")

        assert result1 == result2

    def test_update_existing(self, store):
        """Should update existing entries."""
        store.save("req-123", {"status": "pending"})
        store.save("req-123", {"status": "completed", "result": {"answer": "done"}})

        result = store.get("req-123")
        assert result["status"] == "completed"
        assert result["result"]["answer"] == "done"

    def test_persistence_across_instances(self, temp_db):
        """Should persist data across store instances."""
        # First instance
        store1 = DecisionResultStore(db_path=temp_db)
        store1.save("req-123", {"status": "completed", "result": {"data": "test"}})

        # Second instance (simulates restart)
        store2 = DecisionResultStore(db_path=temp_db)
        result = store2.get("req-123")

        assert result is not None
        assert result["status"] == "completed"
        assert result["result"]["data"] == "test"

    def test_get_metrics(self, store):
        """Should return store metrics."""
        for i in range(5):
            store.save(f"req-{i}", {"status": "completed"})

        metrics = store.get_metrics()

        assert metrics["total_entries"] == 5
        assert "cache_size" in metrics
        assert "max_entries" in metrics
        assert "ttl_seconds" in metrics


class TestGlobalStore:
    """Tests for global store functions."""

    def setup_method(self):
        """Reset global store before each test."""
        reset_decision_result_store()

    def test_get_decision_result_store_singleton(self, tmp_path):
        """Should return same instance."""
        import os
        os.environ["ARAGORA_DECISION_RESULTS_DB"] = str(tmp_path / "singleton.db")

        try:
            store1 = get_decision_result_store()
            store2 = get_decision_result_store()
            assert store1 is store2
        finally:
            del os.environ["ARAGORA_DECISION_RESULTS_DB"]
            reset_decision_result_store()

    def test_reset_decision_result_store(self, tmp_path):
        """Should reset global instance."""
        import os
        os.environ["ARAGORA_DECISION_RESULTS_DB"] = str(tmp_path / "reset.db")

        try:
            store1 = get_decision_result_store()
            reset_decision_result_store()
            store2 = get_decision_result_store()

            # Should be different instances after reset
            assert store1 is not store2
        finally:
            del os.environ["ARAGORA_DECISION_RESULTS_DB"]
            reset_decision_result_store()


class TestConcurrency:
    """Tests for concurrent access."""

    def test_concurrent_writes(self, tmp_path):
        """Should handle concurrent writes safely."""
        import concurrent.futures

        store = DecisionResultStore(db_path=tmp_path / "concurrent.db")

        def write_entry(i):
            store.save(f"req-{i}", {"status": "completed", "index": i})
            return i

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(write_entry, i) for i in range(100)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 100
        assert store.count() == 100

    def test_concurrent_reads(self, tmp_path):
        """Should handle concurrent reads safely."""
        import concurrent.futures

        store = DecisionResultStore(db_path=tmp_path / "concurrent_read.db")

        # Populate store
        for i in range(100):
            store.save(f"req-{i}", {"status": "completed", "index": i})

        def read_entry(i):
            return store.get(f"req-{i}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(read_entry, i) for i in range(100)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 100
        assert all(r is not None for r in results)
