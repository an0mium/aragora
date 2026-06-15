"""
Tests for ReplaysHandler endpoints.

Endpoints tested:
- GET /api/replays - List available replays
- GET /api/replays/{replay_id} - Get specific replay with events
- GET /api/learning/evolution - Get meta-learning patterns
"""

import json
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from aragora.server.handlers.replays import ReplaysHandler, _replays_limiter
from aragora.server.errors import safe_error_message as _safe_error_message
from aragora.server.handlers.base import clear_cache


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset the rate limiter before each test to avoid cross-test pollution."""
    _replays_limiter.clear()
    yield
    _replays_limiter.clear()


@pytest.fixture
def replays_handler(tmp_path):
    """Create a ReplaysHandler with tmp_path as nomic_dir."""
    ctx = {"nomic_dir": tmp_path}
    return ReplaysHandler(ctx)


@pytest.fixture
def setup_replays_dir(tmp_path):
    """Create sample replays directory structure."""
    replays_dir = tmp_path / "replays"
    replays_dir.mkdir()

    # Create replay 1
    replay1 = replays_dir / "replay-001"
    replay1.mkdir()
    (replay1 / "meta.json").write_text(
        json.dumps(
            {
                "topic": "AI Safety Discussion",
                "agents": [{"name": "claude"}, {"name": "gpt4"}],
                "schema_version": "1.1",
            }
        )
    )
    (replay1 / "events.jsonl").write_text(
        '{"type": "start", "timestamp": "2024-01-01T10:00:00Z"}\n'
        '{"type": "message", "agent": "claude", "content": "Hello"}\n'
        '{"type": "end", "timestamp": "2024-01-01T12:00:00Z"}\n'
    )

    # Create replay 2
    replay2 = replays_dir / "replay-002"
    replay2.mkdir()
    (replay2 / "meta.json").write_text(
        json.dumps(
            {
                "topic": "Code Review Best Practices",
                "agents": [{"name": "gemini"}],
                "schema_version": "1.0",
            }
        )
    )
    (replay2 / "events.jsonl").write_text(
        '{"type": "message", "agent": "gemini", "content": "Review..."}\n'
    )

    return replays_dir


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear caches before and after each test."""
    clear_cache()
    yield
    clear_cache()


# ============================================================================
# Route Matching Tests
# ============================================================================


class TestReplaysHandlerRouting:
    """Tests for route matching."""

    def test_can_handle_replays_list(self, replays_handler):
        """Should handle /api/replays."""
        assert replays_handler.can_handle("/api/v1/replays") is True

    def test_can_handle_specific_replay(self, replays_handler):
        """Should handle /api/replays/{replay_id}."""
        assert replays_handler.can_handle("/api/v1/replays/replay-001") is True

    def test_can_handle_learning_evolution(self, replays_handler):
        """Should handle /api/learning/evolution."""
        assert replays_handler.can_handle("/api/v1/learning/evolution") is True

    def test_cannot_handle_nested_replay_path(self, replays_handler):
        """Should not handle deeply nested replay paths."""
        assert replays_handler.can_handle("/api/v1/replays/replay-001/events") is False

    def test_cannot_handle_unknown_routes(self, replays_handler):
        """Should not handle unknown routes."""
        assert replays_handler.can_handle("/api/v1/unknown") is False
        assert replays_handler.can_handle("/api/v1/debates") is False

    def test_handle_returns_none_for_unknown(self, replays_handler):
        """Should return None for unknown paths."""
        result = replays_handler.handle("/api/unknown", {}, None)
        assert result is None


# ============================================================================
# List Replays Tests
# ============================================================================


class TestListReplaysEndpoint:
    """Tests for /api/replays endpoint."""

    def test_returns_replay_list(self, replays_handler, setup_replays_dir):
        """Should return list of replays."""
        result = replays_handler.handle("/api/replays", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert isinstance(data, list)
        assert len(data) == 2

    def test_includes_replay_metadata(self, replays_handler, setup_replays_dir):
        """Should include replay metadata."""
        result = replays_handler.handle("/api/replays", {}, None)

        data = json.loads(result.body)
        replay = data[0]  # Should be replay-002 (sorted reverse)
        assert "id" in replay
        assert "topic" in replay
        assert "agents" in replay

    def test_sorts_replays_by_id_descending(self, replays_handler, setup_replays_dir):
        """Should sort replays by ID descending."""
        result = replays_handler.handle("/api/replays", {}, None)

        data = json.loads(result.body)
        assert data[0]["id"] == "replay-002"
        assert data[1]["id"] == "replay-001"

    def test_returns_empty_without_replays_dir(self, replays_handler):
        """Should return empty list when replays dir doesn't exist."""
        result = replays_handler.handle("/api/replays", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data == []

    def test_returns_empty_without_nomic_dir(self):
        """Should return empty list without nomic_dir."""
        handler = ReplaysHandler({})
        result = handler.handle("/api/replays", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data == []

    def test_skips_malformed_meta_files(self, replays_handler, tmp_path):
        """Should skip replays with malformed meta.json."""
        replays_dir = tmp_path / "replays"
        replays_dir.mkdir()

        # Create replay with bad meta
        bad_replay = replays_dir / "bad-replay"
        bad_replay.mkdir()
        (bad_replay / "meta.json").write_text("not valid json {")

        result = replays_handler.handle("/api/replays", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        # Bad replay should be skipped
        assert len(data) == 0

    def test_skips_non_directories(self, replays_handler, tmp_path):
        """Should skip non-directory entries."""
        replays_dir = tmp_path / "replays"
        replays_dir.mkdir()

        # Create a file instead of directory
        (replays_dir / "not-a-replay.txt").write_text("just a file")

        result = replays_handler.handle("/api/replays", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert len(data) == 0


# ============================================================================
# Get Replay Tests
# ============================================================================


class TestGetReplayEndpoint:
    """Tests for /api/replays/{replay_id} endpoint."""

    def test_returns_replay_details(self, replays_handler, setup_replays_dir):
        """Should return replay details."""
        result = replays_handler.handle("/api/replays/replay-001", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["id"] == "replay-001"
        assert "meta" in data
        assert "events" in data

    def test_returns_meta_data(self, replays_handler, setup_replays_dir):
        """Should return meta data."""
        result = replays_handler.handle("/api/replays/replay-001", {}, None)

        data = json.loads(result.body)
        assert data["meta"]["topic"] == "AI Safety Discussion"
        assert len(data["meta"]["agents"]) == 2

    def test_returns_events(self, replays_handler, setup_replays_dir):
        """Should return events from events.jsonl."""
        result = replays_handler.handle("/api/replays/replay-001", {}, None)

        data = json.loads(result.body)
        assert len(data["events"]) == 3
        assert data["events"][0]["type"] == "start"

    def test_returns_event_count(self, replays_handler, setup_replays_dir):
        """Should return event count."""
        result = replays_handler.handle("/api/replays/replay-001", {}, None)

        data = json.loads(result.body)
        assert data["event_count"] == 3

    def test_returns_404_for_nonexistent(self, replays_handler, setup_replays_dir):
        """Should return 404 for non-existent replay."""
        result = replays_handler.handle("/api/replays/nonexistent", {}, None)

        assert result.status_code == 404

    def test_returns_503_without_nomic_dir(self):
        """Should return 503 without nomic_dir."""
        handler = ReplaysHandler({})
        result = handler.handle("/api/replays/test", {}, None)

        assert result.status_code == 503

    def test_handles_missing_events_file(self, replays_handler, tmp_path):
        """Should handle missing events.jsonl."""
        replays_dir = tmp_path / "replays"
        replays_dir.mkdir()
        replay = replays_dir / "no-events"
        replay.mkdir()
        (replay / "meta.json").write_text('{"topic": "Test"}')
        # No events.jsonl

        result = replays_handler.handle("/api/replays/no-events", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["events"] == []
        assert data["event_count"] == 0

    def test_handles_malformed_events(self, replays_handler, tmp_path):
        """Should skip malformed event lines."""
        replays_dir = tmp_path / "replays"
        replays_dir.mkdir()
        replay = replays_dir / "bad-events"
        replay.mkdir()
        (replay / "meta.json").write_text('{"topic": "Test"}')
        (replay / "events.jsonl").write_text(
            '{"type": "good"}\nnot valid json\n{"type": "also_good"}\n'
        )

        result = replays_handler.handle("/api/replays/bad-events", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        # Should skip the invalid line
        assert len(data["events"]) == 2


# ============================================================================
# Security Tests
# ============================================================================


class TestReplaysSecurity:
    """Tests for security measures."""

    def test_blocks_path_traversal(self, replays_handler):
        """Should block path traversal attempts."""
        result = replays_handler.handle("/api/replays/../../../etc/passwd", {}, None)

        assert result.status_code == 400

    def test_blocks_dotdot_in_path(self, replays_handler):
        """Should block .. in replay ID."""
        result = replays_handler.handle("/api/replays/..%2F..%2Fetc", {}, None)

        assert result.status_code == 400

    def test_validates_replay_id_format(self, replays_handler):
        """Should validate replay ID format."""
        # Special characters should be rejected
        result = replays_handler.handle("/api/replays/test;rm", {}, None)
        assert result.status_code == 400

        result = replays_handler.handle("/api/replays/test<script>", {}, None)
        assert result.status_code == 400

    def test_accepts_valid_replay_ids(self, replays_handler, setup_replays_dir):
        """Should accept valid replay IDs."""
        valid_ids = ["replay-001", "test_replay", "Replay123", "a-b_c"]

        for replay_id in valid_ids:
            result = replays_handler.handle(f"/api/replays/{replay_id}", {}, None)
            # Should not return 400 for valid IDs (404 is OK if not found)
            assert result.status_code != 400, f"Should accept: {replay_id}"

    def test_rejects_empty_replay_id(self, replays_handler):
        """Should reject empty replay ID."""
        result = replays_handler.handle("/api/replays/", {}, None)

        # Empty ID should be rejected or not matched
        assert result is None or result.status_code == 400


# ============================================================================
# Learning Evolution Tests
# ============================================================================


class TestLearningEvolutionEndpoint:
    """Tests for /api/learning/evolution endpoint."""

    def test_returns_patterns(self, replays_handler, tmp_path):
        """Should return learning patterns."""
        # Create database with patterns
        db_path = tmp_path / "meta_learning.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE meta_patterns (
                id INTEGER PRIMARY KEY,
                pattern_type TEXT,
                data TEXT,
                created_at TEXT
            )
        """
        )
        conn.execute(
            """
            INSERT INTO meta_patterns (pattern_type, data, created_at)
            VALUES ('improvement', '{"score": 0.8}', '2024-01-01T10:00:00Z')
        """
        )
        conn.commit()
        conn.close()

        result = replays_handler.handle("/api/learning/evolution", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert len(data["patterns"]) == 1
        assert data["patterns_count"] == 1

    def test_respects_limit(self, replays_handler, tmp_path):
        """Should respect limit parameter."""
        db_path = tmp_path / "meta_learning.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE meta_patterns (
                id INTEGER PRIMARY KEY,
                pattern_type TEXT,
                created_at TEXT
            )
        """
        )
        for i in range(50):
            conn.execute(
                """
                INSERT INTO meta_patterns (pattern_type, created_at)
                VALUES (?, ?)
            """,
                (f"pattern_{i}", f"2024-01-{i + 1:02d}T10:00:00Z"),
            )
        conn.commit()
        conn.close()

        result = replays_handler.handle("/api/learning/evolution", {"limit": "10"}, None)

        data = json.loads(result.body)
        assert len(data["patterns"]) == 10

    def test_caps_limit_at_100(self, replays_handler, tmp_path):
        """Should cap limit at 100."""
        db_path = tmp_path / "meta_learning.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE meta_patterns (
                id INTEGER PRIMARY KEY,
                pattern_type TEXT,
                created_at TEXT
            )
        """
        )
        for i in range(150):
            conn.execute(
                """
                INSERT INTO meta_patterns (pattern_type, created_at)
                VALUES (?, datetime('now'))
            """,
                (f"pattern_{i}",),
            )
        conn.commit()
        conn.close()

        result = replays_handler.handle("/api/learning/evolution", {"limit": "500"}, None)

        data = json.loads(result.body)
        assert len(data["patterns"]) <= 100

    def test_returns_empty_without_db(self, replays_handler):
        """Should return empty when database doesn't exist."""
        result = replays_handler.handle("/api/learning/evolution", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["patterns"] == []
        assert data["patterns_count"] == 0

    def test_returns_empty_without_nomic_dir(self):
        """Should return empty without nomic_dir."""
        handler = ReplaysHandler({})
        result = handler.handle("/api/learning/evolution", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["patterns"] == []

    def test_handles_missing_table(self, replays_handler, tmp_path):
        """Should handle missing table gracefully."""
        db_path = tmp_path / "meta_learning.db"
        conn = sqlite3.connect(str(db_path))
        conn.close()  # Empty database

        result = replays_handler.handle("/api/learning/evolution", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["patterns"] == []


# ============================================================================
# Error Message Helper Tests
# ============================================================================


class TestSafeErrorMessage:
    """Tests for _safe_error_message helper."""

    def test_file_not_found_error(self):
        """Should sanitize FileNotFoundError."""
        msg = _safe_error_message(FileNotFoundError("secret/path"), "test")
        assert "secret/path" not in msg
        assert "not found" in msg.lower()

    def test_os_error(self):
        """Should sanitize OSError."""
        msg = _safe_error_message(OSError("permission denied"), "test")
        assert "permission" not in msg.lower()
        assert "not found" in msg.lower()

    def test_json_decode_error(self):
        """Should sanitize JSONDecodeError."""
        try:
            json.loads("invalid")
        except json.JSONDecodeError as e:
            msg = _safe_error_message(e, "test")
            # JSONDecodeError type name is "JSONDecodeError" not "json.JSONDecodeError"
            # so it falls through to generic handler
            assert "error" in msg.lower()

    def test_generic_error(self):
        """Should return generic message for unknown errors."""
        msg = _safe_error_message(RuntimeError("sensitive info"), "test")
        assert "sensitive info" not in msg
        assert "error" in msg.lower()


# ============================================================================
# Caching Tests
# ============================================================================


class TestReplaysCaching:
    """Tests for caching behavior."""

    def test_list_replays_caches(self, replays_handler, setup_replays_dir):
        """Should cache replay list."""
        # First call
        result1 = replays_handler.handle("/api/replays", {}, None)
        # Second call should use cache
        result2 = replays_handler.handle("/api/replays", {}, None)

        assert result1.status_code == 200
        assert result2.status_code == 200


# ============================================================================
# Edge Cases
# ============================================================================


class TestReplaysEdgeCases:
    """Tests for edge cases."""

    def test_empty_events_file(self, replays_handler, tmp_path):
        """Should handle empty events file."""
        replays_dir = tmp_path / "replays"
        replays_dir.mkdir()
        replay = replays_dir / "empty-events"
        replay.mkdir()
        (replay / "meta.json").write_text('{"topic": "Test"}')
        (replay / "events.jsonl").write_text("")

        result = replays_handler.handle("/api/replays/empty-events", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["events"] == []

    def test_replay_without_meta(self, replays_handler, tmp_path):
        """Should handle replay without meta.json."""
        replays_dir = tmp_path / "replays"
        replays_dir.mkdir()
        replay = replays_dir / "no-meta"
        replay.mkdir()
        (replay / "events.jsonl").write_text('{"type": "event"}')

        result = replays_handler.handle("/api/replays/no-meta", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["meta"] == {}

    def test_replay_with_malformed_meta(self, replays_handler, tmp_path):
        """Should handle malformed meta.json in replay."""
        replays_dir = tmp_path / "replays"
        replays_dir.mkdir()
        replay = replays_dir / "bad-meta"
        replay.mkdir()
        (replay / "meta.json").write_text("not valid json")

        result = replays_handler.handle("/api/replays/bad-meta", {}, None)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert "error" in data["meta"]

    def test_agent_names_extraction(self, replays_handler, tmp_path):
        """Should extract agent names correctly."""
        replays_dir = tmp_path / "replays"
        replays_dir.mkdir()
        replay = replays_dir / "agents-test"
        replay.mkdir()
        (replay / "meta.json").write_text(
            json.dumps(
                {
                    "topic": "Test",
                    "agents": [
                        {"name": "claude", "provider": "anthropic"},
                        {"name": "gpt4", "provider": "openai"},
                    ],
                }
            )
        )

        result = replays_handler.handle("/api/replays", {}, None)

        data = json.loads(result.body)
        replay_data = data[0]
        assert replay_data["agents"] == ["claude", "gpt4"]
