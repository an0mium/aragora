"""Comprehensive tests for the ReplaysHandler REST endpoints.

Covers all routes and behavior of aragora/server/handlers/replays.py (524 lines):

Routing:
- can_handle() for all declared ROUTES and dynamic replay ID matching
- Versioned path handling (/api/v1/replays/...)

GET endpoints:
- /api/replays - List available replays (with/without nomic dir, limit, pagination, malformed meta)
- /api/replays/:replay_id - Get specific replay with events (pagination, not found, no dir, malformed)
- /api/learning/evolution - Learning evolution data (patterns, agents, debates, empty states)
- /api/meta-learning/stats - Meta-learning hyperparams and efficiency (trends, empty, no db, no table)

Auth:
- GET requests skip auth (public dashboard data)
- Non-GET methods require auth and debates:write permission

Rate limiting:
- Rate limit exceeded returns 429

Error handling:
- Missing nomic_dir returns empty/503
- Missing files return empty responses
- Malformed JSON is gracefully handled
- SQLite OperationalError for missing tables
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.replays import ReplaysHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict | list:
    """Extract JSON body from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class MockHTTPHandler:
    """Lightweight mock for the HTTP handler passed to ReplaysHandler methods."""

    def __init__(
        self,
        method: str = "GET",
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ):
        self.command = method
        self.headers: dict[str, str] = headers or {"User-Agent": "test-agent"}
        self.rfile = MagicMock()
        self.client_address = ("127.0.0.1", 12345)

        if body is not None:
            raw = json.dumps(body).encode()
            self.rfile.read.return_value = raw
            self.headers["Content-Length"] = str(len(raw))
            self.headers["Content-Type"] = "application/json"
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def nomic_dir(tmp_path):
    """Create a temporary nomic directory."""
    d = tmp_path / "nomic_session"
    d.mkdir()
    return d


@pytest.fixture
def handler(nomic_dir):
    """Create a ReplaysHandler with a temporary nomic directory."""
    return ReplaysHandler({"nomic_dir": nomic_dir})


@pytest.fixture
def handler_no_dir():
    """Create a ReplaysHandler without nomic_dir."""
    return ReplaysHandler({})


@pytest.fixture(autouse=True)
def _bypass_rate_limit(monkeypatch):
    """Bypass rate limiting for all tests by default."""
    monkeypatch.setattr(
        "aragora.server.handlers.replays._replays_limiter",
        MagicMock(is_allowed=MagicMock(return_value=True)),
    )


@pytest.fixture(autouse=True)
def _clear_caches():
    """Clear TTL caches before each test to avoid stale state."""
    from aragora.server.handlers.admin.cache import clear_cache

    clear_cache()
    yield
    clear_cache()


def _create_replay_dir(
    replays_dir: Path,
    replay_id: str,
    topic: str = "Test topic",
    agents: list[dict] | None = None,
    events: list[dict] | None = None,
    schema_version: str = "1.0",
    meta_override: dict | None = None,
    skip_meta: bool = False,
    malformed_meta: bool = False,
) -> Path:
    """Helper to create a replay directory with meta.json and events.jsonl."""
    replay_path = replays_dir / replay_id
    replay_path.mkdir(parents=True, exist_ok=True)

    if not skip_meta:
        if malformed_meta:
            (replay_path / "meta.json").write_text("{invalid json here")
        else:
            meta = meta_override or {
                "topic": topic,
                "agents": agents or [{"name": "claude"}, {"name": "gpt-4"}],
                "schema_version": schema_version,
            }
            (replay_path / "meta.json").write_text(json.dumps(meta))

    if events is not None:
        with open(replay_path / "events.jsonl", "w") as f:
            for event in events:
                f.write(json.dumps(event) + "\n")

    return replay_path


# ---------------------------------------------------------------------------
# can_handle() routing
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Tests for route matching via can_handle()."""

    def test_replays_list(self, handler):
        assert handler.can_handle("/api/replays")

    def test_replays_list_versioned(self, handler):
        assert handler.can_handle("/api/v1/replays")

    def test_replays_specific(self, handler):
        assert handler.can_handle("/api/replays/abc123")

    def test_replays_specific_versioned(self, handler):
        assert handler.can_handle("/api/v1/replays/debate_001")

    def test_learning_evolution(self, handler):
        assert handler.can_handle("/api/learning/evolution")

    def test_learning_evolution_versioned(self, handler):
        assert handler.can_handle("/api/v1/learning/evolution")

    def test_meta_learning_stats(self, handler):
        assert handler.can_handle("/api/meta-learning/stats")

    def test_meta_learning_stats_versioned(self, handler):
        assert handler.can_handle("/api/v1/meta-learning/stats")

    def test_unrelated_path_rejected(self, handler):
        assert not handler.can_handle("/api/debates")

    def test_replays_nested_too_deep_rejected(self, handler):
        """Paths with more than 4 segments after split are rejected."""
        assert not handler.can_handle("/api/replays/abc/extra")

    def test_unrelated_learning_rejected(self, handler):
        assert not handler.can_handle("/api/learning/other")

    def test_empty_path_rejected(self, handler):
        assert not handler.can_handle("/")

    def test_routes_count(self, handler):
        """All 9 routes are declared in ROUTES."""
        assert len(handler.ROUTES) == 9


# ---------------------------------------------------------------------------
# handle() routing dispatch
# ---------------------------------------------------------------------------


class TestHandleRouting:
    """Tests for the main handle() method dispatching."""

    def test_unmatched_path_returns_none(self, handler):
        result = handler.handle("/api/unknown", {}, MockHTTPHandler())
        assert result is None

    def test_get_skips_auth(self, handler, nomic_dir):
        """GET requests should not require auth."""
        result = handler.handle("/api/replays", {}, MockHTTPHandler(method="GET"))
        assert result is not None
        assert _status(result) == 200

    def test_replays_list_route(self, handler, nomic_dir):
        result = handler.handle("/api/replays", {}, MockHTTPHandler())
        assert _status(result) == 200

    def test_learning_evolution_route(self, handler, nomic_dir):
        result = handler.handle("/api/learning/evolution", {}, MockHTTPHandler())
        assert _status(result) == 200

    def test_meta_learning_route(self, handler, nomic_dir):
        result = handler.handle("/api/meta-learning/stats", {}, MockHTTPHandler())
        assert _status(result) == 200

    def test_specific_replay_route(self, handler, nomic_dir):
        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir()
        _create_replay_dir(replays_dir, "test-001")
        result = handler.handle("/api/replays/test-001", {}, MockHTTPHandler())
        assert _status(result) == 200

    def test_versioned_replays_list(self, handler, nomic_dir):
        result = handler.handle("/api/v1/replays", {}, MockHTTPHandler())
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for rate limit enforcement."""

    def test_rate_limit_exceeded(self, handler, nomic_dir, monkeypatch):
        """When rate limit is exceeded, return 429."""
        monkeypatch.setattr(
            "aragora.server.handlers.replays._replays_limiter",
            MagicMock(is_allowed=MagicMock(return_value=False)),
        )
        result = handler.handle("/api/replays", {}, MockHTTPHandler())
        assert _status(result) == 429
        assert "Rate limit" in _body(result).get("error", "")

    def test_rate_limit_exceeded_learning(self, handler, nomic_dir, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.replays._replays_limiter",
            MagicMock(is_allowed=MagicMock(return_value=False)),
        )
        result = handler.handle("/api/learning/evolution", {}, MockHTTPHandler())
        assert _status(result) == 429

    def test_rate_limit_exceeded_meta_learning(self, handler, nomic_dir, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.replays._replays_limiter",
            MagicMock(is_allowed=MagicMock(return_value=False)),
        )
        result = handler.handle("/api/meta-learning/stats", {}, MockHTTPHandler())
        assert _status(result) == 429

    def test_rate_limit_exceeded_specific_replay(self, handler, nomic_dir, monkeypatch):
        monkeypatch.setattr(
            "aragora.server.handlers.replays._replays_limiter",
            MagicMock(is_allowed=MagicMock(return_value=False)),
        )
        result = handler.handle("/api/replays/test-001", {}, MockHTTPHandler())
        assert _status(result) == 429


# ---------------------------------------------------------------------------
# GET /api/replays - List replays
# ---------------------------------------------------------------------------


class TestListReplays:
    """Tests for the replay listing endpoint."""

    def test_no_nomic_dir(self, handler_no_dir):
        result = handler_no_dir.handle("/api/replays", {}, MockHTTPHandler())
        assert _status(result) == 200
        assert _body(result) == []

    def test_no_replays_dir(self, handler, nomic_dir):
        """When replays directory does not exist, return empty list."""
        result = handler.handle("/api/replays", {}, MockHTTPHandler())
        assert _status(result) == 200
        assert _body(result) == []

    def test_empty_replays_dir(self, handler, nomic_dir):
        """When replays directory exists but is empty."""
        (nomic_dir / "replays").mkdir()
        result = handler.handle("/api/replays", {}, MockHTTPHandler())
        assert _status(result) == 200
        assert _body(result) == []

    def test_list_replays_with_data(self, handler, nomic_dir):
        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir()
        _create_replay_dir(replays_dir, "debate-001", topic="Rate limiters")
        _create_replay_dir(replays_dir, "debate-002", topic="Circuit breakers")

        result = handler.handle("/api/replays", {}, MockHTTPHandler())
        assert _status(result) == 200
        body = _body(result)
        assert len(body) == 2
        topics = {r["topic"] for r in body}
        assert "Rate limiters" in topics
        assert "Circuit breakers" in topics

    def test_replay_entry_fields(self, handler, nomic_dir):
        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir()
        _create_replay_dir(
            replays_dir,
            "debate-x",
            topic="My topic",
            agents=[{"name": "a1"}, {"name": "a2"}],
            schema_version="2.0",
        )

        result = handler.handle("/api/replays", {}, MockHTTPHandler())
        body = _body(result)
        assert len(body) == 1
        entry = body[0]
        assert entry["id"] == "debate-x"
        assert entry["topic"] == "My topic"
        assert entry["agents"] == ["a1", "a2"]
        assert entry["schema_version"] == "2.0"

    def test_limit_param(self, handler, nomic_dir):
        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir()
        for i in range(5):
            _create_replay_dir(replays_dir, f"d-{i}", topic=f"Topic {i}")

        result = handler.handle("/api/replays", {"limit": "2"}, MockHTTPHandler())
        body = _body(result)
        assert len(body) == 2

    def test_limit_clamped_min(self, handler, nomic_dir):
        """Limit is clamped to at least 1."""
        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir()
        _create_replay_dir(replays_dir, "d-0")

        result = handler.handle("/api/replays", {"limit": "0"}, MockHTTPHandler())
        body = _body(result)
        assert len(body) == 1

    def test_limit_clamped_max(self, handler, nomic_dir):
        """Limit is clamped to at most 500."""
        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir()
        _create_replay_dir(replays_dir, "d-0")

        # Should not error with limit=999
        result = handler.handle("/api/replays", {"limit": "999"}, MockHTTPHandler())
        assert _status(result) == 200

    def test_skips_malformed_meta(self, handler, nomic_dir):
        """Replays with malformed meta.json are silently skipped."""
        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir()
        _create_replay_dir(replays_dir, "good-001", topic="Good")
        _create_replay_dir(replays_dir, "bad-001", malformed_meta=True)

        result = handler.handle("/api/replays", {}, MockHTTPHandler())
        body = _body(result)
        assert len(body) == 1
        assert body[0]["topic"] == "Good"

    def test_skips_dirs_without_meta(self, handler, nomic_dir):
        """Replay directories without meta.json are skipped."""
        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir()
        _create_replay_dir(replays_dir, "no-meta", skip_meta=True)
        _create_replay_dir(replays_dir, "with-meta", topic="Present")

        result = handler.handle("/api/replays", {}, MockHTTPHandler())
        body = _body(result)
        assert len(body) == 1
        assert body[0]["topic"] == "Present"

    def test_skips_files_in_replays_dir(self, handler, nomic_dir):
        """Non-directory entries in replays/ are ignored."""
        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir()
        (replays_dir / "stray_file.txt").write_text("not a replay")
        _create_replay_dir(replays_dir, "real-replay", topic="Real")

        result = handler.handle("/api/replays", {}, MockHTTPHandler())
        body = _body(result)
        assert len(body) == 1

    def test_agents_default_empty(self, handler, nomic_dir):
        """When meta has no agents, return empty list."""
        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir()
        _create_replay_dir(replays_dir, "no-agents", meta_override={"topic": "T"})

        result = handler.handle("/api/replays", {}, MockHTTPHandler())
        body = _body(result)
        assert body[0]["agents"] == []

    def test_topic_defaults_to_dir_name(self, handler, nomic_dir):
        """When meta has no topic field, use directory name."""
        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir()
        _create_replay_dir(replays_dir, "my-dir", meta_override={"agents": []})

        result = handler.handle("/api/replays", {}, MockHTTPHandler())
        body = _body(result)
        assert body[0]["topic"] == "my-dir"


# ---------------------------------------------------------------------------
# GET /api/replays/:replay_id - Get specific replay
# ---------------------------------------------------------------------------


class TestGetReplay:
    """Tests for getting a specific replay."""

    def test_no_nomic_dir(self, handler_no_dir):
        result = handler_no_dir.handle("/api/replays/test-001", {}, MockHTTPHandler())
        assert _status(result) == 503
        assert "not configured" in _body(result).get("error", "").lower()

    def test_replay_not_found(self, handler, nomic_dir):
        (nomic_dir / "replays").mkdir()
        result = handler.handle("/api/replays/nonexistent", {}, MockHTTPHandler())
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    def test_replay_with_meta_and_events(self, handler, nomic_dir):
        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir()
        events = [
            {"type": "start", "ts": "2026-01-01"},
            {"type": "message", "agent": "claude", "content": "Hello"},
            {"type": "end", "ts": "2026-01-01"},
        ]
        _create_replay_dir(replays_dir, "r-001", topic="My debate", events=events)

        result = handler.handle("/api/replays/r-001", {}, MockHTTPHandler())
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "r-001"
        assert body["meta"]["topic"] == "My debate"
        assert len(body["events"]) == 3
        assert body["event_count"] == 3
        assert body["total_events"] == 3
        assert body["offset"] == 0
        assert body["has_more"] is False

    def test_replay_pagination_offset(self, handler, nomic_dir):
        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir()
        events = [{"type": f"event_{i}"} for i in range(10)]
        _create_replay_dir(replays_dir, "r-page", events=events)

        result = handler.handle(
            "/api/replays/r-page", {"offset": "3", "limit": "4"}, MockHTTPHandler()
        )
        body = _body(result)
        assert body["offset"] == 3
        assert body["limit"] == 4
        assert body["event_count"] == 4
        assert body["total_events"] == 10
        assert body["has_more"] is True
        assert body["events"][0]["type"] == "event_3"

    def test_replay_pagination_last_page(self, handler, nomic_dir):
        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir()
        events = [{"type": f"event_{i}"} for i in range(5)]
        _create_replay_dir(replays_dir, "r-last", events=events)

        result = handler.handle(
            "/api/replays/r-last", {"offset": "3", "limit": "10"}, MockHTTPHandler()
        )
        body = _body(result)
        assert body["event_count"] == 2
        assert body["total_events"] == 5
        assert body["has_more"] is False

    def test_replay_offset_beyond_total(self, handler, nomic_dir):
        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir()
        events = [{"type": "e"}]
        _create_replay_dir(replays_dir, "r-off", events=events)

        result = handler.handle("/api/replays/r-off", {"offset": "100"}, MockHTTPHandler())
        body = _body(result)
        assert body["event_count"] == 0
        assert body["total_events"] == 1
        assert body["has_more"] is False

    def test_replay_no_events_file(self, handler, nomic_dir):
        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir()
        _create_replay_dir(replays_dir, "r-no-events", topic="No events")

        result = handler.handle("/api/replays/r-no-events", {}, MockHTTPHandler())
        body = _body(result)
        assert body["events"] == []
        assert body["total_events"] == 0

    def test_replay_malformed_meta(self, handler, nomic_dir):
        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir()
        _create_replay_dir(replays_dir, "r-bad-meta", malformed_meta=True)

        result = handler.handle("/api/replays/r-bad-meta", {}, MockHTTPHandler())
        body = _body(result)
        assert _status(result) == 200
        assert "error" in body["meta"]

    def test_replay_malformed_event_lines_skipped(self, handler, nomic_dir):
        """Malformed JSONL lines are silently skipped."""
        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir()
        replay_dir = replays_dir / "r-bad-events"
        replay_dir.mkdir()
        (replay_dir / "meta.json").write_text('{"topic": "test"}')
        with open(replay_dir / "events.jsonl", "w") as f:
            f.write('{"type": "good"}\n')
            f.write("not valid json\n")
            f.write('{"type": "also_good"}\n')

        result = handler.handle("/api/replays/r-bad-events", {}, MockHTTPHandler())
        body = _body(result)
        # 3 lines total (including malformed), but only 2 successfully parsed
        assert body["total_events"] == 3
        assert body["event_count"] == 2

    def test_replay_empty_lines_skipped(self, handler, nomic_dir):
        """Empty lines in events file are skipped."""
        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir()
        replay_dir = replays_dir / "r-empty-lines"
        replay_dir.mkdir()
        (replay_dir / "meta.json").write_text('{"topic": "test"}')
        with open(replay_dir / "events.jsonl", "w") as f:
            f.write('{"type": "a"}\n')
            f.write("\n")
            f.write('{"type": "b"}\n')

        result = handler.handle("/api/replays/r-empty-lines", {}, MockHTTPHandler())
        body = _body(result)
        assert body["total_events"] == 3
        # Empty line after strip is skipped
        assert body["event_count"] == 2

    def test_replay_limit_clamped_min(self, handler, nomic_dir):
        """Limit 0 is clamped to 1."""
        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir()
        events = [{"t": i} for i in range(5)]
        _create_replay_dir(replays_dir, "r-clamp", events=events)

        result = handler.handle("/api/replays/r-clamp", {"limit": "0"}, MockHTTPHandler())
        body = _body(result)
        assert body["limit"] == 1
        assert body["event_count"] == 1

    def test_replay_offset_clamped_min(self, handler, nomic_dir):
        """Negative offset is clamped to 0."""
        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir()
        events = [{"t": i} for i in range(3)]
        _create_replay_dir(replays_dir, "r-neg-off", events=events)

        result = handler.handle("/api/replays/r-neg-off", {"offset": "-5"}, MockHTTPHandler())
        body = _body(result)
        assert body["offset"] == 0

    def test_replay_versioned_path(self, handler, nomic_dir):
        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir()
        _create_replay_dir(replays_dir, "v1-test", topic="Versioned")

        result = handler.handle("/api/v1/replays/v1-test", {}, MockHTTPHandler())
        assert _status(result) == 200
        body = _body(result)
        assert body["id"] == "v1-test"


# ---------------------------------------------------------------------------
# GET /api/learning/evolution
# ---------------------------------------------------------------------------


class TestLearningEvolution:
    """Tests for learning/evolution endpoint."""

    def test_no_nomic_dir(self, handler_no_dir):
        result = handler_no_dir.handle("/api/learning/evolution", {}, MockHTTPHandler())
        assert _status(result) == 200
        body = _body(result)
        assert body["patterns"] == []
        assert body["agents"] == []
        assert body["debates"] == []
        assert body["patterns_count"] == 0
        assert body["agents_count"] == 0
        assert body["debates_count"] == 0

    def test_no_db_no_elo_no_state(self, handler, nomic_dir):
        """When none of the data files exist, return empty response."""
        result = handler.handle("/api/learning/evolution", {}, MockHTTPHandler())
        body = _body(result)
        assert body["patterns"] == []
        assert body["agents"] == []
        assert body["debates"] == []

    def test_patterns_from_db(self, handler, nomic_dir):
        """Patterns are read from meta_learning.db."""
        db_path = nomic_dir / "meta_learning.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE meta_patterns (
                id INTEGER PRIMARY KEY,
                pattern_type TEXT,
                success_rate REAL,
                occurrence_count INTEGER,
                created_at TEXT
            )
        """)
        conn.execute(
            "INSERT INTO meta_patterns (pattern_type, success_rate, occurrence_count, created_at) VALUES (?, ?, ?, ?)",
            ("refactoring", 0.85, 12, "2026-01-15T10:00:00"),
        )
        conn.execute(
            "INSERT INTO meta_patterns (pattern_type, success_rate, occurrence_count, created_at) VALUES (?, ?, ?, ?)",
            ("bug_fix", 0.92, 8, "2026-01-16T10:00:00"),
        )
        conn.commit()
        conn.close()

        result = handler.handle("/api/learning/evolution", {}, MockHTTPHandler())
        body = _body(result)
        assert body["patterns_count"] == 2
        # Most recent first
        assert body["patterns"][0]["issue_type"] == "bug_fix"
        assert body["patterns"][0]["date"] == "2026-01-16"
        assert body["patterns"][0]["success_rate"] == 0.92
        assert body["patterns"][0]["pattern_count"] == 8
        assert body["patterns"][1]["issue_type"] == "refactoring"

    def test_patterns_limit(self, handler, nomic_dir):
        """Limit parameter controls how many patterns are returned."""
        db_path = nomic_dir / "meta_learning.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE meta_patterns (
                id INTEGER PRIMARY KEY, pattern_type TEXT,
                success_rate REAL, occurrence_count INTEGER, created_at TEXT
            )
        """)
        for i in range(10):
            conn.execute(
                "INSERT INTO meta_patterns VALUES (?, ?, ?, ?, ?)",
                (i, f"type_{i}", 0.5, 1, f"2026-01-{i + 1:02d}T00:00:00"),
            )
        conn.commit()
        conn.close()

        result = handler.handle("/api/learning/evolution", {"limit": "3"}, MockHTTPHandler())
        body = _body(result)
        assert body["patterns_count"] == 3

    def test_patterns_no_such_table(self, handler, nomic_dir):
        """When meta_patterns table doesn't exist, return empty patterns."""
        db_path = nomic_dir / "meta_learning.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE other_table (id INTEGER)")
        conn.commit()
        conn.close()

        result = handler.handle("/api/learning/evolution", {}, MockHTTPHandler())
        body = _body(result)
        assert body["patterns"] == []

    def test_agents_from_elo_snapshot(self, handler, nomic_dir):
        """Agent data is read from elo_snapshot.json."""
        elo_data = {
            "timestamp": "2026-02-01T12:00:00",
            "ratings": {
                "claude": {
                    "elo": 1500,
                    "games": 20,
                    "wins": 15,
                    "calibration_score": 0.8,
                },
                "gpt-4": {
                    "elo": 1200,
                    "games": 10,
                    "wins": 3,
                    "calibration_score": 0.6,
                },
            },
        }
        (nomic_dir / "elo_snapshot.json").write_text(json.dumps(elo_data))

        result = handler.handle("/api/learning/evolution", {}, MockHTTPHandler())
        body = _body(result)
        assert body["agents_count"] == 2

        agents_by_name = {a["agent"]: a for a in body["agents"]}
        claude = agents_by_name["claude"]
        assert claude["date"] == "2026-02-01"
        assert claude["acceptance_rate"] == 15 / 20
        assert claude["critique_quality"] == 0.8
        assert claude["reputation_score"] == 1500 / 2000

        gpt4 = agents_by_name["gpt-4"]
        assert gpt4["acceptance_rate"] == 3 / 10

    def test_agents_zero_games(self, handler, nomic_dir):
        """Agent with 0 games gets default 0.5 acceptance rate."""
        elo_data = {
            "timestamp": "2026-02-01T12:00:00",
            "ratings": {
                "newbie": {"elo": 1000, "games": 0, "wins": 0},
            },
        }
        (nomic_dir / "elo_snapshot.json").write_text(json.dumps(elo_data))

        result = handler.handle("/api/learning/evolution", {}, MockHTTPHandler())
        body = _body(result)
        assert body["agents"][0]["acceptance_rate"] == 0.5

    def test_agents_elo_normalized(self, handler, nomic_dir):
        """ELO above 2000 is capped at 1.0 reputation."""
        elo_data = {
            "timestamp": "2026-02-01T12:00:00",
            "ratings": {
                "top": {"elo": 2500, "games": 100, "wins": 90},
            },
        }
        (nomic_dir / "elo_snapshot.json").write_text(json.dumps(elo_data))

        result = handler.handle("/api/learning/evolution", {}, MockHTTPHandler())
        body = _body(result)
        assert body["agents"][0]["reputation_score"] == 1.0

    def test_agents_malformed_elo(self, handler, nomic_dir):
        """Malformed ELO JSON is gracefully handled."""
        (nomic_dir / "elo_snapshot.json").write_text("{invalid json")

        result = handler.handle("/api/learning/evolution", {}, MockHTTPHandler())
        body = _body(result)
        assert body["agents"] == []

    def test_agents_no_timestamp(self, handler, nomic_dir):
        """Missing timestamp in ELO snapshot yields empty date."""
        elo_data = {
            "ratings": {
                "agent1": {"elo": 1000, "games": 5, "wins": 3},
            },
        }
        (nomic_dir / "elo_snapshot.json").write_text(json.dumps(elo_data))

        result = handler.handle("/api/learning/evolution", {}, MockHTTPHandler())
        body = _body(result)
        assert body["agents"][0]["date"] == ""

    def test_debates_from_nomic_state(self, handler, nomic_dir):
        """Debate data comes from nomic_state.json debate_history."""
        state = {
            "debate_history": [
                {
                    "timestamp": "2026-01-10T10:00:00",
                    "consensus_reached": True,
                    "confidence": 0.9,
                    "rounds": 3,
                    "duration_seconds": 120,
                },
                {
                    "timestamp": "2026-01-10T11:00:00",
                    "consensus_reached": False,
                    "confidence": 0.4,
                    "rounds": 5,
                    "duration_seconds": 300,
                },
                {
                    "timestamp": "2026-01-11T09:00:00",
                    "consensus_reached": True,
                    "confidence": 0.95,
                    "rounds": 2,
                    "duration_seconds": 60,
                },
            ],
        }
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        result = handler.handle("/api/learning/evolution", {}, MockHTTPHandler())
        body = _body(result)
        assert body["debates_count"] == 2  # Two distinct dates

        by_date = {d["date"]: d for d in body["debates"]}

        jan10 = by_date["2026-01-10"]
        assert jan10["total_debates"] == 2
        assert jan10["consensus_rate"] == 0.5
        assert jan10["avg_confidence"] == pytest.approx((0.9 + 0.4) / 2)
        assert jan10["avg_rounds"] == pytest.approx((3 + 5) / 2)
        assert jan10["avg_duration"] == pytest.approx((120 + 300) / 2)

        jan11 = by_date["2026-01-11"]
        assert jan11["total_debates"] == 1
        assert jan11["consensus_rate"] == 1.0

    def test_debates_empty_history(self, handler, nomic_dir):
        """Empty debate_history returns no debates."""
        state = {"debate_history": []}
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        result = handler.handle("/api/learning/evolution", {}, MockHTTPHandler())
        body = _body(result)
        assert body["debates"] == []

    def test_debates_malformed_state(self, handler, nomic_dir):
        """Malformed state JSON is gracefully handled."""
        (nomic_dir / "nomic_state.json").write_text("{bad json")

        result = handler.handle("/api/learning/evolution", {}, MockHTTPHandler())
        body = _body(result)
        assert body["debates"] == []

    def test_debates_missing_fields_use_defaults(self, handler, nomic_dir):
        """Debates with missing fields use default values."""
        from aragora.config import DEFAULT_ROUNDS

        state = {
            "debate_history": [
                {"timestamp": "2026-03-01T00:00:00"},
            ],
        }
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        result = handler.handle("/api/learning/evolution", {}, MockHTTPHandler())
        body = _body(result)
        assert body["debates_count"] == 1
        d = body["debates"][0]
        assert d["consensus_rate"] == 0  # consensus_reached defaults to False
        assert d["avg_confidence"] == 0.5  # confidence default
        assert d["avg_rounds"] == DEFAULT_ROUNDS
        assert d["avg_duration"] == 60  # duration_seconds default

    def test_debates_limit(self, handler, nomic_dir):
        """Limit parameter restricts debate history entries scanned."""
        state = {
            "debate_history": [
                {"timestamp": f"2026-01-{i + 1:02d}T00:00:00", "consensus_reached": True}
                for i in range(30)
            ],
        }
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        result = handler.handle("/api/learning/evolution", {"limit": "5"}, MockHTTPHandler())
        body = _body(result)
        # limit=5 means last 5 debate entries are used from history
        assert body["debates_count"] <= 5

    def test_versioned_learning_evolution(self, handler, nomic_dir):
        result = handler.handle("/api/v1/learning/evolution", {}, MockHTTPHandler())
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# GET /api/meta-learning/stats
# ---------------------------------------------------------------------------


class TestMetaLearningStats:
    """Tests for meta-learning stats endpoint."""

    def test_no_nomic_dir(self, handler_no_dir):
        result = handler_no_dir.handle("/api/meta-learning/stats", {}, MockHTTPHandler())
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "no_data"
        assert body["current_hyperparams"] == {}
        assert body["adjustment_history"] == []
        assert body["efficiency_log"] == []

    def test_no_database(self, handler, nomic_dir):
        """When meta_learning.db doesn't exist."""
        result = handler.handle("/api/meta-learning/stats", {}, MockHTTPHandler())
        body = _body(result)
        assert body["status"] == "no_database"

    def _create_meta_learning_db(self, db_path: Path):
        """Helper to create a meta_learning.db with required tables."""
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE meta_hyperparams (
                id INTEGER PRIMARY KEY,
                hyperparams TEXT,
                metrics TEXT,
                adjustment_reason TEXT,
                created_at TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE meta_efficiency_log (
                id INTEGER PRIMARY KEY,
                cycle_number INTEGER,
                metrics TEXT,
                created_at TEXT
            )
        """)
        conn.commit()
        return conn

    def test_tables_not_initialized(self, handler, nomic_dir):
        """When DB exists but tables are missing."""
        db_path = nomic_dir / "meta_learning.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE something_else (id INTEGER)")
        conn.commit()
        conn.close()

        result = handler.handle("/api/meta-learning/stats", {}, MockHTTPHandler())
        body = _body(result)
        assert body["status"] == "tables_not_initialized"

    def test_empty_tables(self, handler, nomic_dir):
        """When tables exist but are empty."""
        db_path = nomic_dir / "meta_learning.db"
        conn = self._create_meta_learning_db(db_path)
        conn.close()

        result = handler.handle("/api/meta-learning/stats", {}, MockHTTPHandler())
        body = _body(result)
        assert body["status"] == "ok"
        assert body["current_hyperparams"] == {}
        assert body["adjustment_history"] == []
        assert body["efficiency_log"] == []
        assert body["trend"] == "insufficient_data"
        assert body["evaluations"] == 0

    def test_hyperparams_and_history(self, handler, nomic_dir):
        """Test current hyperparams and adjustment history retrieval."""
        db_path = nomic_dir / "meta_learning.db"
        conn = self._create_meta_learning_db(db_path)
        conn.execute(
            "INSERT INTO meta_hyperparams VALUES (?, ?, ?, ?, ?)",
            (
                1,
                json.dumps({"lr": 0.01, "batch": 32}),
                json.dumps({"loss": 0.5}),
                "initial",
                "2026-01-01T00:00:00",
            ),
        )
        conn.execute(
            "INSERT INTO meta_hyperparams VALUES (?, ?, ?, ?, ?)",
            (
                2,
                json.dumps({"lr": 0.005, "batch": 64}),
                json.dumps({"loss": 0.3}),
                "reduced lr",
                "2026-01-02T00:00:00",
            ),
        )
        conn.commit()
        conn.close()

        result = handler.handle("/api/meta-learning/stats", {}, MockHTTPHandler())
        body = _body(result)
        assert body["status"] == "ok"
        # Current = most recent
        assert body["current_hyperparams"] == {"lr": 0.005, "batch": 64}
        assert len(body["adjustment_history"]) == 2
        # Most recent first
        assert body["adjustment_history"][0]["reason"] == "reduced lr"
        assert body["adjustment_history"][0]["hyperparams"] == {"lr": 0.005, "batch": 64}
        assert body["adjustment_history"][1]["reason"] == "initial"

    def test_efficiency_log(self, handler, nomic_dir):
        """Test efficiency log retrieval."""
        db_path = nomic_dir / "meta_learning.db"
        conn = self._create_meta_learning_db(db_path)
        for i in range(3):
            conn.execute(
                "INSERT INTO meta_efficiency_log VALUES (?, ?, ?, ?)",
                (
                    i,
                    i + 1,
                    json.dumps({"pattern_retention_rate": 0.5 + i * 0.1}),
                    f"2026-01-{i + 1:02d}T00:00:00",
                ),
            )
        conn.commit()
        conn.close()

        result = handler.handle("/api/meta-learning/stats", {}, MockHTTPHandler())
        body = _body(result)
        assert body["evaluations"] == 3
        assert len(body["efficiency_log"]) == 3
        # Most recent first
        assert body["efficiency_log"][0]["cycle"] == 3

    def test_trend_improving(self, handler, nomic_dir):
        """Trend is 'improving' when newer entries have higher retention.

        SQL returns DESC by created_at, so efficiency_log = [newest, ..., oldest].
        second_half = [:mid] = newer entries, first_half = [mid:] = older entries.
        Improving means second_retention > first_retention + 0.05.
        """
        db_path = nomic_dir / "meta_learning.db"
        conn = self._create_meta_learning_db(db_path)
        # Inserted by ascending created_at; SQL DESC reverses to [jan04,jan03,jan02,jan01]
        # So: second_half (newer) = [0.9, 0.85] avg=0.875
        #     first_half (older)  = [0.5, 0.45] avg=0.475
        # 0.875 > 0.475 + 0.05 => improving
        data = [
            (0, 0.45, "2026-01-01"),  # oldest, low retention
            (1, 0.5, "2026-01-02"),
            (2, 0.85, "2026-01-03"),
            (3, 0.9, "2026-01-04"),  # newest, high retention
        ]
        for i, ret, date in data:
            conn.execute(
                "INSERT INTO meta_efficiency_log VALUES (?, ?, ?, ?)",
                (i, i + 1, json.dumps({"pattern_retention_rate": ret}), f"{date}T00:00:00"),
            )
        conn.commit()
        conn.close()

        result = handler.handle("/api/meta-learning/stats", {}, MockHTTPHandler())
        body = _body(result)
        assert body["trend"] == "improving"

    def test_trend_declining(self, handler, nomic_dir):
        """Trend is 'declining' when newer entries have lower retention.

        After DESC: [newest=low, ..., oldest=high].
        second_half (newer) avg < first_half (older) avg - 0.05 => declining.
        """
        db_path = nomic_dir / "meta_learning.db"
        conn = self._create_meta_learning_db(db_path)
        # DESC order: [jan04=0.3, jan03=0.35, jan02=0.85, jan01=0.9]
        # second_half (newer) = [0.3, 0.35] avg=0.325
        # first_half (older)  = [0.85, 0.9] avg=0.875
        # 0.325 < 0.875 - 0.05 => declining
        data = [
            (0, 0.9, "2026-01-01"),  # oldest, high retention
            (1, 0.85, "2026-01-02"),
            (2, 0.35, "2026-01-03"),
            (3, 0.3, "2026-01-04"),  # newest, low retention
        ]
        for i, ret, date in data:
            conn.execute(
                "INSERT INTO meta_efficiency_log VALUES (?, ?, ?, ?)",
                (i, i + 1, json.dumps({"pattern_retention_rate": ret}), f"{date}T00:00:00"),
            )
        conn.commit()
        conn.close()

        result = handler.handle("/api/meta-learning/stats", {}, MockHTTPHandler())
        body = _body(result)
        assert body["trend"] == "declining"

    def test_trend_stable(self, handler, nomic_dir):
        """Trend is 'stable' when retention doesn't change much (within 0.05)."""
        db_path = nomic_dir / "meta_learning.db"
        conn = self._create_meta_learning_db(db_path)
        # All entries have similar retention
        for i in range(4):
            conn.execute(
                "INSERT INTO meta_efficiency_log VALUES (?, ?, ?, ?)",
                (
                    i,
                    i + 1,
                    json.dumps({"pattern_retention_rate": 0.7}),
                    f"2026-01-{i + 1:02d}T00:00:00",
                ),
            )
        conn.commit()
        conn.close()

        result = handler.handle("/api/meta-learning/stats", {}, MockHTTPHandler())
        body = _body(result)
        assert body["trend"] == "stable"

    def test_trend_insufficient_data(self, handler, nomic_dir):
        """Fewer than 4 efficiency entries yields 'insufficient_data'."""
        db_path = nomic_dir / "meta_learning.db"
        conn = self._create_meta_learning_db(db_path)
        for i in range(3):
            conn.execute(
                "INSERT INTO meta_efficiency_log VALUES (?, ?, ?, ?)",
                (i, i + 1, json.dumps({}), f"2026-01-{i + 1:02d}T00:00:00"),
            )
        conn.commit()
        conn.close()

        result = handler.handle("/api/meta-learning/stats", {}, MockHTTPHandler())
        body = _body(result)
        assert body["trend"] == "insufficient_data"

    def test_limit_param(self, handler, nomic_dir):
        """Limit controls how many records are returned."""
        db_path = nomic_dir / "meta_learning.db"
        conn = self._create_meta_learning_db(db_path)
        for i in range(10):
            conn.execute(
                "INSERT INTO meta_hyperparams VALUES (?, ?, ?, ?, ?)",
                (i, "{}", "{}", f"reason_{i}", f"2026-01-{i + 1:02d}T00:00:00"),
            )
            conn.execute(
                "INSERT INTO meta_efficiency_log VALUES (?, ?, ?, ?)",
                (i, i + 1, "{}", f"2026-01-{i + 1:02d}T00:00:00"),
            )
        conn.commit()
        conn.close()

        result = handler.handle("/api/meta-learning/stats", {"limit": "3"}, MockHTTPHandler())
        body = _body(result)
        assert len(body["adjustment_history"]) == 3
        assert len(body["efficiency_log"]) == 3

    def test_limit_clamped(self, handler, nomic_dir):
        """Limit is clamped between 1 and 50."""
        db_path = nomic_dir / "meta_learning.db"
        conn = self._create_meta_learning_db(db_path)
        conn.close()

        # Limit 0 clamped to 1
        result = handler.handle("/api/meta-learning/stats", {"limit": "0"}, MockHTTPHandler())
        assert _status(result) == 200

        # Limit 999 clamped to 50
        result = handler.handle("/api/meta-learning/stats", {"limit": "999"}, MockHTTPHandler())
        assert _status(result) == 200

    def test_hyperparams_malformed_json(self, handler, nomic_dir):
        """Malformed JSON in hyperparams column uses fallback."""
        db_path = nomic_dir / "meta_learning.db"
        conn = self._create_meta_learning_db(db_path)
        conn.execute(
            "INSERT INTO meta_hyperparams VALUES (?, ?, ?, ?, ?)",
            (1, "not json", "also bad", "test", "2026-01-01T00:00:00"),
        )
        conn.commit()
        conn.close()

        result = handler.handle("/api/meta-learning/stats", {}, MockHTTPHandler())
        body = _body(result)
        assert body["status"] == "ok"
        # safe_json_parse returns {} for malformed hyperparams
        assert body["current_hyperparams"] == {}

    def test_versioned_meta_learning(self, handler, nomic_dir):
        result = handler.handle("/api/v1/meta-learning/stats", {}, MockHTTPHandler())
        assert _status(result) == 200


# ---------------------------------------------------------------------------
# Auth enforcement on non-GET methods
# ---------------------------------------------------------------------------


class TestAuthEnforcement:
    """Tests for auth/permission checks on write operations."""

    @pytest.mark.no_auto_auth
    def test_post_requires_auth(self, nomic_dir):
        """Non-GET methods should require authentication."""
        handler = ReplaysHandler({"nomic_dir": nomic_dir})
        mock_http = MockHTTPHandler(method="POST")
        result = handler.handle("/api/replays", {}, mock_http)
        # Should return 401 since auth is not mocked
        assert result is not None
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_put_requires_auth(self, nomic_dir):
        handler = ReplaysHandler({"nomic_dir": nomic_dir})
        mock_http = MockHTTPHandler(method="PUT")
        result = handler.handle("/api/replays", {}, mock_http)
        assert result is not None
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_delete_requires_auth(self, nomic_dir):
        handler = ReplaysHandler({"nomic_dir": nomic_dir})
        mock_http = MockHTTPHandler(method="DELETE")
        result = handler.handle("/api/replays", {}, mock_http)
        assert result is not None
        assert _status(result) == 401


# ---------------------------------------------------------------------------
# Edge cases and integration
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case and integration tests."""

    def test_handler_none_handler_arg(self, nomic_dir):
        """When handler is None, method defaults to GET."""
        h = ReplaysHandler({"nomic_dir": nomic_dir})
        result = h.handle("/api/replays", {}, None)
        assert result is not None
        assert _status(result) == 200

    def test_handler_without_command(self, nomic_dir):
        """Handler without command attribute defaults to GET."""
        h = ReplaysHandler({"nomic_dir": nomic_dir})
        mock = MagicMock(spec=[])  # No attributes
        mock.client_address = ("127.0.0.1", 12345)
        mock.headers = {}
        result = h.handle("/api/replays", {}, mock)
        assert result is not None
        assert _status(result) == 200

    def test_combined_data_sources(self, handler, nomic_dir):
        """Learning evolution combines data from all three sources."""
        # Create DB with patterns
        db_path = nomic_dir / "meta_learning.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE meta_patterns (
                id INTEGER PRIMARY KEY, pattern_type TEXT,
                success_rate REAL, occurrence_count INTEGER, created_at TEXT
            )
        """)
        conn.execute("INSERT INTO meta_patterns VALUES (1, 'test', 0.8, 5, '2026-01-01T00:00:00')")
        conn.commit()
        conn.close()

        # Create ELO snapshot
        elo = {
            "timestamp": "2026-01-01T00:00:00",
            "ratings": {"agent1": {"elo": 1200, "games": 10, "wins": 7}},
        }
        (nomic_dir / "elo_snapshot.json").write_text(json.dumps(elo))

        # Create state with debate history
        state = {
            "debate_history": [{"timestamp": "2026-01-01T00:00:00", "consensus_reached": True}]
        }
        (nomic_dir / "nomic_state.json").write_text(json.dumps(state))

        result = handler.handle("/api/learning/evolution", {}, MockHTTPHandler())
        body = _body(result)
        assert body["patterns_count"] == 1
        assert body["agents_count"] == 1
        assert body["debates_count"] == 1

    def test_ctx_defaults_to_empty(self):
        """Handler with None ctx uses empty dict."""
        h = ReplaysHandler(None)
        assert h.ctx == {}

    def test_nomic_dir_as_path_object(self, tmp_path):
        """nomic_dir can be a Path object."""
        h = ReplaysHandler({"nomic_dir": tmp_path})
        result = h.handle("/api/replays", {}, MockHTTPHandler())
        assert _status(result) == 200

    def test_replays_sorted_by_mtime(self, handler, nomic_dir):
        """Replays are sorted by modification time (newest first)."""
        import time

        replays_dir = nomic_dir / "replays"
        replays_dir.mkdir()

        # Create replays with distinct modification times
        _create_replay_dir(replays_dir, "old-replay", topic="Old")
        time.sleep(0.05)
        _create_replay_dir(replays_dir, "new-replay", topic="New")

        result = handler.handle("/api/replays", {}, MockHTTPHandler())
        body = _body(result)
        assert len(body) == 2
        assert body[0]["topic"] == "New"
        assert body[1]["topic"] == "Old"
