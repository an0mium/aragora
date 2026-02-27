"""Tests for DebateResultStore — SQLite persistence for playground debates."""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import pytest

from aragora.storage.debate_store import DebateResultStore, get_debate_store


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Create a fresh DebateResultStore backed by a temp directory."""
    monkeypatch.setenv("ARAGORA_DATA_DIR", str(tmp_path))
    return DebateResultStore("test_debates.db")


class TestDebateResultStore:
    def test_save_and_get(self, store):
        result = {"topic": "Test", "rounds": [{"round": 1}], "id": "abc123"}
        store.save("abc123", "Test topic", result)

        retrieved = store.get("abc123")
        assert retrieved is not None
        assert retrieved["topic"] == "Test"
        assert retrieved["id"] == "abc123"

    def test_get_nonexistent_returns_none(self, store):
        assert store.get("nonexistent") is None

    def test_get_expired_returns_none(self, store):
        result = {"topic": "Expired", "id": "exp1"}
        store.save("exp1", "Expired topic", result, ttl_days=0)

        # TTL of 0 means it expires immediately (expires_at ~ now)
        time.sleep(0.01)
        assert store.get("exp1") is None

    def test_save_with_custom_source(self, store):
        result = {"id": "src1", "data": "test"}
        store.save("src1", "Source test", result, source="oracle")

        recent = store.list_recent()
        assert len(recent) == 1
        assert recent[0]["source"] == "oracle"

    def test_save_replaces_existing(self, store):
        result1 = {"id": "dup1", "version": 1}
        result2 = {"id": "dup1", "version": 2}

        store.save("dup1", "Original", result1)
        store.save("dup1", "Updated", result2)

        retrieved = store.get("dup1")
        assert retrieved is not None
        assert retrieved["version"] == 2

    def test_list_recent_ordering(self, store):
        for i in range(5):
            store.save(f"id{i}", f"Topic {i}", {"id": f"id{i}", "n": i})

        recent = store.list_recent(limit=3)
        assert len(recent) == 3
        # Most recent first
        assert recent[0]["id"] == "id4"
        assert recent[1]["id"] == "id3"
        assert recent[2]["id"] == "id2"

    def test_list_recent_excludes_expired(self, store):
        store.save("fresh", "Fresh", {"id": "fresh"}, ttl_days=30)
        store.save("stale", "Stale", {"id": "stale"}, ttl_days=0)

        time.sleep(0.01)
        recent = store.list_recent()
        assert len(recent) == 1
        assert recent[0]["id"] == "fresh"

    def test_list_recent_metadata_only(self, store):
        big_result = {"id": "meta1", "huge_field": "x" * 10000}
        store.save("meta1", "Meta test", big_result)

        recent = store.list_recent()
        assert len(recent) == 1
        entry = recent[0]
        assert "id" in entry
        assert "topic" in entry
        assert "source" in entry
        assert "created_at" in entry
        # Full result data should NOT be in the listing
        assert "huge_field" not in entry

    def test_cleanup_expired(self, store):
        store.save("keep", "Keep me", {"id": "keep"}, ttl_days=30)
        store.save("expire1", "Expire 1", {"id": "expire1"}, ttl_days=0)
        store.save("expire2", "Expire 2", {"id": "expire2"}, ttl_days=0)

        time.sleep(0.01)
        deleted = store.cleanup_expired()
        assert deleted == 2

        assert store.get("keep") is not None
        assert store.get("expire1") is None
        assert store.get("expire2") is None

    def test_cleanup_returns_zero_when_nothing_expired(self, store):
        store.save("fresh", "Fresh", {"id": "fresh"}, ttl_days=30)
        deleted = store.cleanup_expired()
        assert deleted == 0

    def test_handles_corrupt_json(self, store):
        """If result_json is corrupt, get() returns None instead of crashing."""
        now = time.time()
        with store.connection() as conn:
            conn.execute(
                """
                INSERT INTO debate_results
                    (id, topic, result_json, source, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("corrupt1", "Test", "not valid json{{{", "test", now, now + 86400),
            )

        assert store.get("corrupt1") is None

    def test_save_with_non_serializable_values(self, store):
        """Non-JSON-serializable values should be converted via default=str."""
        from datetime import datetime, timezone

        result = {
            "id": "dt1",
            "timestamp": datetime.now(timezone.utc),
            "data": {"nested": True},
        }
        store.save("dt1", "Datetime test", result)

        retrieved = store.get("dt1")
        assert retrieved is not None
        assert isinstance(retrieved["timestamp"], str)

    def test_default_source_is_playground(self, store):
        store.save("pg1", "Playground", {"id": "pg1"})
        recent = store.list_recent()
        assert recent[0]["source"] == "playground"


class TestGetDebateStore:
    def test_returns_instance(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ARAGORA_DATA_DIR", str(tmp_path))
        # Reset singleton
        import aragora.storage.debate_store as mod

        monkeypatch.setattr(mod, "_store", None)

        store = get_debate_store()
        assert isinstance(store, DebateResultStore)

    def test_returns_same_instance(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ARAGORA_DATA_DIR", str(tmp_path))
        import aragora.storage.debate_store as mod

        monkeypatch.setattr(mod, "_store", None)

        s1 = get_debate_store()
        s2 = get_debate_store()
        assert s1 is s2
