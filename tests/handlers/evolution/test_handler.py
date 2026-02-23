"""
Tests for EvolutionHandler (aragora/server/handlers/evolution/handler.py).

Covers all routes and behavior:
- GET /api/evolution - Get evolution summary (root redirect)
- GET /api/evolution/patterns - Get top patterns across all agents
- GET /api/evolution/summary - Get evolution summary statistics
- GET /api/evolution/{agent}/history - Get prompt evolution history for an agent
- GET /api/evolution/{agent}/prompt - Get current/specific prompt version for an agent
- Versioned variants (/api/v1/...)
- can_handle() routing
- Module unavailable (503) responses
- Nomic dir not configured (503) responses
- Input validation and error paths
- Rate limiting
- Authentication on non-GET methods
- Path parameter extraction and validation
- Internal error handling (500)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import aragora.server.handlers.evolution.handler as _handler_mod
from aragora.server.handlers.evolution.handler import EvolutionHandler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract the body dict from a HandlerResult."""
    if result is None:
        return {}
    if hasattr(result, "body"):
        raw = result.body
        if isinstance(raw, bytes):
            return json.loads(raw)
        return raw
    return {}


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if hasattr(result, "status_code"):
        return result.status_code
    return 0


# ---------------------------------------------------------------------------
# Mock HTTP handler
# ---------------------------------------------------------------------------


class MockHTTPHandler:
    """Mock HTTP handler for BaseHandler and rate limiting."""

    def __init__(self, body: dict | None = None, method: str = "GET"):
        self.rfile = MagicMock()
        self._body = body
        self.command = method
        if body is not None:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {
                "Content-Length": str(len(body_bytes)),
                "Content-Type": "application/json",
            }
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {
                "Content-Length": "2",
                "Content-Type": "application/json",
            }
        self.client_address = ("127.0.0.1", 54321)


# ---------------------------------------------------------------------------
# Mock PromptVersion for _get_prompt_version
# ---------------------------------------------------------------------------


class MockPromptVersion:
    """Mock prompt version object returned by PromptEvolver."""

    def __init__(
        self,
        version: int = 1,
        prompt: str = "You are a helpful assistant.",
        performance_score: float = 0.85,
        debates_count: int = 10,
        consensus_rate: float = 0.75,
        metadata: dict | None = None,
        created_at: str = "2026-01-01T00:00:00",
    ):
        self.version = version
        self.prompt = prompt
        self.performance_score = performance_score
        self.debates_count = debates_count
        self.consensus_rate = consensus_rate
        self.metadata = metadata or {}
        self.created_at = created_at


# ---------------------------------------------------------------------------
# Mock PromptEvolver
# ---------------------------------------------------------------------------


class MockPromptEvolver:
    """Mock PromptEvolver for testing."""

    def __init__(self, db_path: str = "test.db"):
        self.db_path = db_path
        self._history = []
        self._patterns = []
        self._prompt_version = None
        self._conn = MagicMock()

    def get_evolution_history(self, agent: str, limit: int = 10) -> list:
        return self._history[:limit]

    def get_top_patterns(self, pattern_type: str | None = None, limit: int = 10) -> list:
        if pattern_type:
            return [p for p in self._patterns if p.get("type") == pattern_type][:limit]
        return self._patterns[:limit]

    def get_prompt_version(
        self, agent: str, version: int | None = None
    ) -> MockPromptVersion | None:
        return self._prompt_version

    def connection(self):
        return self._conn


# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------

_PROMPT_EVOLVER_PATCH = "aragora.server.handlers.evolution.handler.PromptEvolver"
_GET_DB_PATH_PATCH = "aragora.server.handlers.evolution.handler.get_db_path"
_GET_CLIENT_IP_PATCH = "aragora.server.handlers.evolution.handler.get_client_ip"

# Save originals to restore
_ORIG_EVOLUTION_AVAILABLE = _handler_mod.EVOLUTION_AVAILABLE
_ORIG_PROMPT_EVOLVER = _handler_mod.PromptEvolver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _restore_module_globals():
    """Restore module-level globals after each test."""
    yield
    _handler_mod.EVOLUTION_AVAILABLE = _ORIG_EVOLUTION_AVAILABLE
    _handler_mod.PromptEvolver = _ORIG_PROMPT_EVOLVER


@pytest.fixture(autouse=True)
def _clear_rate_limiter():
    """Clear rate limiter buckets between tests."""
    _handler_mod._evolution_limiter._buckets.clear()
    yield
    _handler_mod._evolution_limiter._buckets.clear()


@pytest.fixture
def handler():
    """Create EvolutionHandler with EVOLUTION_AVAILABLE=True."""
    _handler_mod.EVOLUTION_AVAILABLE = True
    _handler_mod.PromptEvolver = MockPromptEvolver
    h = EvolutionHandler(ctx={"nomic_dir": Path("/tmp/nomic")})
    return h


@pytest.fixture
def handler_unavailable():
    """Handler with EVOLUTION_AVAILABLE = False."""
    _handler_mod.EVOLUTION_AVAILABLE = False
    _handler_mod.PromptEvolver = None
    return EvolutionHandler(ctx={})


@pytest.fixture
def handler_no_nomic():
    """Handler with EVOLUTION_AVAILABLE True but no nomic_dir."""
    _handler_mod.EVOLUTION_AVAILABLE = True
    _handler_mod.PromptEvolver = MockPromptEvolver
    return EvolutionHandler(ctx={})


@pytest.fixture
def http():
    """Create a default GET mock HTTP handler."""
    return MockHTTPHandler()


@pytest.fixture
def http_post():
    """Create a POST mock HTTP handler."""
    return MockHTTPHandler(method="POST")


# ============================================================================
# can_handle() routing tests
# ============================================================================


class TestCanHandle:
    """Test can_handle routing logic."""

    def test_root_evolution_path(self, handler):
        assert handler.can_handle("/api/evolution") is True

    def test_patterns_path(self, handler):
        assert handler.can_handle("/api/evolution/patterns") is True

    def test_summary_path(self, handler):
        assert handler.can_handle("/api/evolution/summary") is True

    def test_history_path(self, handler):
        assert handler.can_handle("/api/evolution/claude/history") is True

    def test_prompt_path(self, handler):
        assert handler.can_handle("/api/evolution/claude/prompt") is True

    def test_versioned_root(self, handler):
        assert handler.can_handle("/api/v1/evolution") is True

    def test_versioned_patterns(self, handler):
        assert handler.can_handle("/api/v1/evolution/patterns") is True

    def test_versioned_summary(self, handler):
        assert handler.can_handle("/api/v1/evolution/summary") is True

    def test_versioned_history(self, handler):
        assert handler.can_handle("/api/v1/evolution/claude/history") is True

    def test_versioned_prompt(self, handler):
        assert handler.can_handle("/api/v1/evolution/claude/prompt") is True

    def test_unrelated_path(self, handler):
        assert handler.can_handle("/api/debates") is False

    def test_partial_path(self, handler):
        assert handler.can_handle("/api/evol") is False

    def test_different_prefix(self, handler):
        assert handler.can_handle("/foo/evolution/patterns") is False

    def test_history_with_different_agent(self, handler):
        assert handler.can_handle("/api/evolution/gpt4/history") is True

    def test_prompt_with_different_agent(self, handler):
        assert handler.can_handle("/api/evolution/gemini/prompt") is True


# ============================================================================
# Root endpoint (/api/evolution)
# ============================================================================


class TestRootEndpoint:
    """Test GET /api/evolution returns summary."""

    def test_root_returns_summary_503_when_unavailable(self, handler_unavailable, http):
        result = handler_unavailable.handle("/api/evolution", {}, http)
        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "")

    def test_root_returns_503_no_nomic_dir(self, handler_no_nomic, http):
        result = handler_no_nomic.handle("/api/evolution", {}, http)
        assert _status(result) == 503
        assert "not configured" in _body(result).get("error", "")

    def test_root_returns_summary_on_success(self, handler, http):
        mock_evolver = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Setup cursor results for each query
        mock_cursor.fetchone.side_effect = [(5,), (3,), (12,)]
        mock_cursor.fetchall.side_effect = [
            [("structural", 8), ("semantic", 4)],
            [("claude", 0.95, 3), ("gpt4", 0.88, 2)],
            [("claude", "mutation", "2026-01-01")],
        ]
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_evolver.connection.return_value = mock_conn

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution", {}, http)

        assert _status(result) == 200
        body = _body(result)
        assert body["total_prompt_versions"] == 5
        assert body["total_agents"] == 3
        assert body["total_patterns"] == 12
        assert body["pattern_distribution"] == {"structural": 8, "semantic": 4}
        assert len(body["top_agents"]) == 2
        assert body["top_agents"][0]["agent"] == "claude"
        assert len(body["recent_activity"]) == 1

    def test_root_versioned_path(self, handler, http):
        mock_evolver = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [(0,), (0,), (0,)]
        mock_cursor.fetchall.side_effect = [[], [], []]
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_evolver.connection.return_value = mock_conn

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/v1/evolution", {}, http)

        assert _status(result) == 200


# ============================================================================
# Summary endpoint (/api/evolution/summary)
# ============================================================================


class TestSummaryEndpoint:
    """Test GET /api/evolution/summary."""

    def test_summary_returns_503_when_unavailable(self, handler_unavailable, http):
        result = handler_unavailable.handle("/api/evolution/summary", {}, http)
        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "")

    def test_summary_returns_503_no_nomic_dir(self, handler_no_nomic, http):
        result = handler_no_nomic.handle("/api/evolution/summary", {}, http)
        assert _status(result) == 503
        assert "not configured" in _body(result).get("error", "")

    def test_summary_returns_stats_on_success(self, handler, http):
        mock_evolver = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [(10,), (5,), (20,)]
        mock_cursor.fetchall.side_effect = [
            [("structural", 12), ("behavioral", 8)],
            [("claude", 0.92, 4), ("gpt4", 0.85, 3)],
            [("gpt4", "crossover", "2026-01-02")],
        ]
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_evolver.connection.return_value = mock_conn

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/summary", {}, http)

        assert _status(result) == 200
        body = _body(result)
        assert body["total_prompt_versions"] == 10
        assert body["total_agents"] == 5
        assert body["total_patterns"] == 20
        assert body["pattern_distribution"]["structural"] == 12
        assert body["pattern_distribution"]["behavioral"] == 8
        assert body["top_agents"][0]["best_score"] == 0.92
        assert body["recent_activity"][0]["strategy"] == "crossover"

    def test_summary_versioned_path(self, handler, http):
        mock_evolver = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [(0,), (0,), (0,)]
        mock_cursor.fetchall.side_effect = [[], [], []]
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_evolver.connection.return_value = mock_conn

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/v1/evolution/summary", {}, http)

        assert _status(result) == 200

    def test_summary_handles_internal_error(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.connection.side_effect = OSError("db error")

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/summary", {}, http)

        assert _status(result) == 500
        assert "Failed" in _body(result).get("error", "")

    def test_summary_handles_key_error(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.connection.side_effect = KeyError("missing key")

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/summary", {}, http)

        assert _status(result) == 500

    def test_summary_handles_value_error(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.connection.side_effect = ValueError("bad value")

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/summary", {}, http)

        assert _status(result) == 500

    def test_summary_handles_type_error(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.connection.side_effect = TypeError("bad type")

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/summary", {}, http)

        assert _status(result) == 500

    def test_summary_handles_attribute_error(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.connection.side_effect = AttributeError("no attr")

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/summary", {}, http)

        assert _status(result) == 500

    def test_summary_empty_results(self, handler, http):
        mock_evolver = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.side_effect = [(0,), (0,), (0,)]
        mock_cursor.fetchall.side_effect = [[], [], []]
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_evolver.connection.return_value = mock_conn

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/summary", {}, http)

        assert _status(result) == 200
        body = _body(result)
        assert body["total_prompt_versions"] == 0
        assert body["total_agents"] == 0
        assert body["total_patterns"] == 0
        assert body["pattern_distribution"] == {}
        assert body["top_agents"] == []
        assert body["recent_activity"] == []


# ============================================================================
# Patterns endpoint (/api/evolution/patterns)
# ============================================================================


class TestPatternsEndpoint:
    """Test GET /api/evolution/patterns."""

    def test_patterns_returns_503_when_unavailable(self, handler_unavailable, http):
        result = handler_unavailable.handle("/api/evolution/patterns", {}, http)
        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "")

    def test_patterns_returns_503_no_nomic_dir(self, handler_no_nomic, http):
        result = handler_no_nomic.handle("/api/evolution/patterns", {}, http)
        assert _status(result) == 503
        assert "not configured" in _body(result).get("error", "")

    def test_patterns_success_no_filter(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_top_patterns.return_value = [
            {"type": "structural", "name": "loop unroll", "count": 5},
            {"type": "semantic", "name": "tone shift", "count": 3},
        ]

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/patterns", {}, http)

        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 2
        assert len(body["patterns"]) == 2
        assert body["filter"] is None

    def test_patterns_with_type_filter(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_top_patterns.return_value = [
            {"type": "structural", "name": "loop unroll", "count": 5},
        ]

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/patterns", {"type": "structural"}, http)

        assert _status(result) == 200
        body = _body(result)
        assert body["filter"] == "structural"
        mock_evolver.get_top_patterns.assert_called_once_with(pattern_type="structural", limit=10)

    def test_patterns_with_custom_limit(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_top_patterns.return_value = []

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/patterns", {"limit": "5"}, http)

        assert _status(result) == 200
        mock_evolver.get_top_patterns.assert_called_once_with(pattern_type=None, limit=5)

    def test_patterns_limit_clamped_min(self, handler, http):
        """Limit below 1 is clamped to 1."""
        mock_evolver = MagicMock()
        mock_evolver.get_top_patterns.return_value = []

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/patterns", {"limit": "0"}, http)

        assert _status(result) == 200
        mock_evolver.get_top_patterns.assert_called_once_with(pattern_type=None, limit=1)

    def test_patterns_limit_clamped_max(self, handler, http):
        """Limit above 50 is clamped to 50."""
        mock_evolver = MagicMock()
        mock_evolver.get_top_patterns.return_value = []

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/patterns", {"limit": "100"}, http)

        assert _status(result) == 200
        mock_evolver.get_top_patterns.assert_called_once_with(pattern_type=None, limit=50)

    def test_patterns_limit_negative_clamped(self, handler, http):
        """Negative limit is clamped to 1."""
        mock_evolver = MagicMock()
        mock_evolver.get_top_patterns.return_value = []

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/patterns", {"limit": "-5"}, http)

        assert _status(result) == 200
        mock_evolver.get_top_patterns.assert_called_once_with(pattern_type=None, limit=1)

    def test_patterns_versioned_path(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_top_patterns.return_value = []

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/v1/evolution/patterns", {}, http)

        assert _status(result) == 200

    def test_patterns_internal_error(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_top_patterns.side_effect = OSError("db failure")

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/patterns", {}, http)

        assert _status(result) == 500
        assert "Failed" in _body(result).get("error", "")

    def test_patterns_key_error(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_top_patterns.side_effect = KeyError("bad key")

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/patterns", {}, http)

        assert _status(result) == 500

    def test_patterns_value_error(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_top_patterns.side_effect = ValueError("bad")

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/patterns", {}, http)

        assert _status(result) == 500

    def test_patterns_type_error(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_top_patterns.side_effect = TypeError("wrong type")

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/patterns", {}, http)

        assert _status(result) == 500

    def test_patterns_attribute_error(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_top_patterns.side_effect = AttributeError("missing")

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/patterns", {}, http)

        assert _status(result) == 500

    def test_patterns_empty_results(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_top_patterns.return_value = []

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/patterns", {}, http)

        assert _status(result) == 200
        body = _body(result)
        assert body["count"] == 0
        assert body["patterns"] == []


# ============================================================================
# History endpoint (/api/evolution/{agent}/history)
# ============================================================================


class TestHistoryEndpoint:
    """Test GET /api/evolution/{agent}/history."""

    def test_history_returns_503_when_unavailable(self, handler_unavailable, http):
        result = handler_unavailable.handle("/api/evolution/claude/history", {}, http)
        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "")

    def test_history_returns_503_no_nomic_dir(self, handler_no_nomic, http):
        result = handler_no_nomic.handle("/api/evolution/claude/history", {}, http)
        assert _status(result) == 503
        assert "not configured" in _body(result).get("error", "")

    def test_history_success(self, handler, http):
        history_data = [
            {"version": 1, "score": 0.8, "strategy": "mutation"},
            {"version": 2, "score": 0.85, "strategy": "crossover"},
        ]
        mock_evolver = MagicMock()
        mock_evolver.get_evolution_history.return_value = history_data

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/claude/history", {}, http)

        assert _status(result) == 200
        body = _body(result)
        assert body["agent"] == "claude"
        assert body["count"] == 2
        assert body["history"] == history_data

    def test_history_with_limit(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_evolution_history.return_value = [{"version": 1}]

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/claude/history", {"limit": "5"}, http)

        assert _status(result) == 200
        mock_evolver.get_evolution_history.assert_called_once_with("claude", limit=5)

    def test_history_limit_clamped_min(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_evolution_history.return_value = []

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/claude/history", {"limit": "0"}, http)

        assert _status(result) == 200
        mock_evolver.get_evolution_history.assert_called_once_with("claude", limit=1)

    def test_history_limit_clamped_max(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_evolution_history.return_value = []

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/claude/history", {"limit": "100"}, http)

        assert _status(result) == 200
        mock_evolver.get_evolution_history.assert_called_once_with("claude", limit=50)

    def test_history_different_agent(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_evolution_history.return_value = []

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/gpt4/history", {}, http)

        assert _status(result) == 200
        body = _body(result)
        assert body["agent"] == "gpt4"

    def test_history_versioned_path(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_evolution_history.return_value = []

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/v1/evolution/claude/history", {}, http)

        assert _status(result) == 200

    def test_history_invalid_agent_name(self, handler, http):
        """Agent names with special chars should be rejected."""
        result = handler.handle("/api/evolution/../../etc/history", {}, http)
        assert _status(result) == 400

    def test_history_empty_agent_name(self, handler, http):
        """Empty agent segment should be rejected."""
        result = handler.handle("/api/evolution//history", {}, http)
        # The path split produces an empty segment at index 3
        assert _status(result) == 400

    def test_history_internal_error(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_evolution_history.side_effect = OSError("db down")

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/claude/history", {}, http)

        assert _status(result) == 500
        assert "Failed" in _body(result).get("error", "")

    def test_history_key_error(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_evolution_history.side_effect = KeyError("no key")

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/claude/history", {}, http)

        assert _status(result) == 500

    def test_history_value_error(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_evolution_history.side_effect = ValueError("bad")

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/claude/history", {}, http)

        assert _status(result) == 500

    def test_history_agent_with_underscores(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_evolution_history.return_value = []

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/my_agent/history", {}, http)

        assert _status(result) == 200
        assert _body(result)["agent"] == "my_agent"

    def test_history_agent_with_dashes(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_evolution_history.return_value = []

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/my-agent/history", {}, http)

        assert _status(result) == 200
        assert _body(result)["agent"] == "my-agent"

    def test_history_agent_name_too_long(self, handler, http):
        """Agent names longer than 32 chars should be rejected."""
        long_name = "a" * 33
        result = handler.handle(f"/api/evolution/{long_name}/history", {}, http)
        assert _status(result) == 400


# ============================================================================
# Prompt endpoint (/api/evolution/{agent}/prompt)
# ============================================================================


class TestPromptEndpoint:
    """Test GET /api/evolution/{agent}/prompt."""

    def test_prompt_returns_503_when_unavailable(self, handler_unavailable, http):
        result = handler_unavailable.handle("/api/evolution/claude/prompt", {}, http)
        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "")

    def test_prompt_returns_503_no_nomic_dir(self, handler_no_nomic, http):
        result = handler_no_nomic.handle("/api/evolution/claude/prompt", {}, http)
        assert _status(result) == 503
        assert "not configured" in _body(result).get("error", "")

    def test_prompt_success_current_version(self, handler, http):
        mock_pv = MockPromptVersion(
            version=3,
            prompt="You are a debate expert.",
            performance_score=0.92,
            debates_count=25,
            consensus_rate=0.80,
            metadata={"strategy": "mutation"},
            created_at="2026-02-01T12:00:00",
        )
        mock_evolver = MagicMock()
        mock_evolver.get_prompt_version.return_value = mock_pv

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/claude/prompt", {}, http)

        assert _status(result) == 200
        body = _body(result)
        assert body["agent"] == "claude"
        assert body["version"] == 3
        assert body["prompt"] == "You are a debate expert."
        assert body["performance_score"] == 0.92
        assert body["debates_count"] == 25
        assert body["consensus_rate"] == 0.80
        assert body["metadata"] == {"strategy": "mutation"}
        assert body["created_at"] == "2026-02-01T12:00:00"

    def test_prompt_with_specific_version(self, handler, http):
        mock_pv = MockPromptVersion(version=2)
        mock_evolver = MagicMock()
        mock_evolver.get_prompt_version.return_value = mock_pv

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/claude/prompt", {"version": "2"}, http)

        assert _status(result) == 200
        mock_evolver.get_prompt_version.assert_called_once_with("claude", 2)

    def test_prompt_no_version_param(self, handler, http):
        """When no version param is provided, get_int_param returns 0."""
        mock_pv = MockPromptVersion()
        mock_evolver = MagicMock()
        mock_evolver.get_prompt_version.return_value = mock_pv

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/claude/prompt", {}, http)

        assert _status(result) == 200
        # get_int_param returns 0 as default when key is absent
        mock_evolver.get_prompt_version.assert_called_once_with("claude", 0)

    def test_prompt_not_found_returns_404(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_prompt_version.return_value = None

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/claude/prompt", {}, http)

        assert _status(result) == 404
        assert "No prompt version found" in _body(result).get("error", "")

    def test_prompt_not_found_includes_agent_name(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_prompt_version.return_value = None

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/gpt4/prompt", {}, http)

        assert _status(result) == 404
        assert "gpt4" in _body(result).get("error", "")

    def test_prompt_versioned_path(self, handler, http):
        mock_pv = MockPromptVersion()
        mock_evolver = MagicMock()
        mock_evolver.get_prompt_version.return_value = mock_pv

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/v1/evolution/claude/prompt", {}, http)

        assert _status(result) == 200

    def test_prompt_invalid_agent_name(self, handler, http):
        result = handler.handle("/api/evolution/bad!agent/prompt", {}, http)
        assert _status(result) == 400

    def test_prompt_internal_error(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_prompt_version.side_effect = OSError("db error")

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/claude/prompt", {}, http)

        assert _status(result) == 500
        assert "Failed" in _body(result).get("error", "")

    def test_prompt_key_error(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_prompt_version.side_effect = KeyError("no key")

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/claude/prompt", {}, http)

        assert _status(result) == 500

    def test_prompt_value_error(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_prompt_version.side_effect = ValueError("bad")

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/claude/prompt", {}, http)

        assert _status(result) == 500

    def test_prompt_type_error(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_prompt_version.side_effect = TypeError("wrong")

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/claude/prompt", {}, http)

        assert _status(result) == 500

    def test_prompt_different_agent(self, handler, http):
        mock_pv = MockPromptVersion()
        mock_evolver = MagicMock()
        mock_evolver.get_prompt_version.return_value = mock_pv

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/gemini/prompt", {}, http)

        assert _status(result) == 200
        assert _body(result)["agent"] == "gemini"

    def test_prompt_agent_with_numbers(self, handler, http):
        mock_pv = MockPromptVersion()
        mock_evolver = MagicMock()
        mock_evolver.get_prompt_version.return_value = mock_pv

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/gpt4o/prompt", {}, http)

        assert _status(result) == 200
        assert _body(result)["agent"] == "gpt4o"

    def test_prompt_agent_name_too_long(self, handler, http):
        long_name = "x" * 33
        result = handler.handle(f"/api/evolution/{long_name}/prompt", {}, http)
        assert _status(result) == 400


# ============================================================================
# Rate limiting
# ============================================================================


class TestRateLimiting:
    """Test rate limiting behavior."""

    def test_rate_limit_exceeded(self, handler, http):
        with patch(_GET_CLIENT_IP_PATCH, return_value="10.0.0.1"):
            # Fill up the rate limiter
            with patch.object(_handler_mod._evolution_limiter, "is_allowed", return_value=False):
                result = handler.handle("/api/evolution/patterns", {}, http)

        assert _status(result) == 429
        assert "Rate limit" in _body(result).get("error", "")

    def test_rate_limit_not_exceeded(self, handler, http):
        mock_evolver = MagicMock()
        mock_evolver.get_top_patterns.return_value = []

        with (
            patch(_GET_CLIENT_IP_PATCH, return_value="10.0.0.2"),
            patch.object(_handler_mod._evolution_limiter, "is_allowed", return_value=True),
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/patterns", {}, http)

        assert _status(result) == 200

    def test_rate_limit_on_summary(self, handler, http):
        with (
            patch(_GET_CLIENT_IP_PATCH, return_value="10.0.0.3"),
            patch.object(_handler_mod._evolution_limiter, "is_allowed", return_value=False),
        ):
            result = handler.handle("/api/evolution/summary", {}, http)

        assert _status(result) == 429

    def test_rate_limit_on_history(self, handler, http):
        with (
            patch(_GET_CLIENT_IP_PATCH, return_value="10.0.0.4"),
            patch.object(_handler_mod._evolution_limiter, "is_allowed", return_value=False),
        ):
            result = handler.handle("/api/evolution/claude/history", {}, http)

        assert _status(result) == 429

    def test_rate_limit_on_prompt(self, handler, http):
        with (
            patch(_GET_CLIENT_IP_PATCH, return_value="10.0.0.5"),
            patch.object(_handler_mod._evolution_limiter, "is_allowed", return_value=False),
        ):
            result = handler.handle("/api/evolution/claude/prompt", {}, http)

        assert _status(result) == 429

    def test_rate_limit_on_root(self, handler, http):
        with (
            patch(_GET_CLIENT_IP_PATCH, return_value="10.0.0.6"),
            patch.object(_handler_mod._evolution_limiter, "is_allowed", return_value=False),
        ):
            result = handler.handle("/api/evolution", {}, http)

        assert _status(result) == 429


# ============================================================================
# Authentication on non-GET methods
# ============================================================================


class TestAuthentication:
    """Test authentication enforcement on non-GET methods."""

    @pytest.mark.no_auto_auth
    def test_post_requires_auth(self, http_post):
        """POST to evolution endpoints requires authentication."""
        _handler_mod.EVOLUTION_AVAILABLE = True
        _handler_mod.PromptEvolver = MockPromptEvolver
        h = EvolutionHandler(ctx={"nomic_dir": Path("/tmp/nomic")})
        result = h.handle("/api/evolution/patterns", {}, http_post)
        # Should get 401 because not authenticated
        assert _status(result) == 401

    def test_get_skips_auth(self, handler, http):
        """GET requests skip authentication."""
        mock_evolver = MagicMock()
        mock_evolver.get_top_patterns.return_value = []

        with (
            patch(_PROMPT_EVOLVER_PATCH, return_value=mock_evolver),
            patch(_GET_DB_PATH_PATCH, return_value="/tmp/evo.db"),
        ):
            result = handler.handle("/api/evolution/patterns", {}, http)

        # Should succeed with 200, not 401
        assert _status(result) == 200


# ============================================================================
# Routing dispatch (handle method returns None for non-matching paths)
# ============================================================================


class TestRouting:
    """Test that handle() returns None for non-matching paths."""

    def test_non_matching_path_returns_none(self, handler, http):
        result = handler.handle("/api/debates", {}, http)
        assert result is None

    def test_non_matching_evolution_prefix(self, handler, http):
        result = handler.handle("/api/evolve", {}, http)
        assert result is None

    def test_completely_different_path(self, handler, http):
        result = handler.handle("/api/agents/list", {}, http)
        assert result is None

    def test_handle_with_no_handler(self, handler):
        """Handle with no HTTP handler still works for GET."""
        _handler_mod.EVOLUTION_AVAILABLE = False
        result = handler.handle("/api/evolution", {}, None)
        assert _status(result) == 503


# ============================================================================
# Constructor and context
# ============================================================================


class TestConstructor:
    """Test handler construction."""

    def test_default_context(self):
        h = EvolutionHandler()
        assert h.ctx == {}

    def test_custom_context(self):
        ctx = {"nomic_dir": Path("/custom")}
        h = EvolutionHandler(ctx=ctx)
        assert h.ctx == ctx

    def test_none_context_becomes_empty_dict(self):
        h = EvolutionHandler(ctx=None)
        assert h.ctx == {}


# ============================================================================
# ROUTES class attribute
# ============================================================================


class TestRoutesAttribute:
    """Test ROUTES class attribute contains expected routes."""

    def test_routes_contains_base_paths(self, handler):
        assert "/api/evolution" in handler.ROUTES
        assert "/api/evolution/patterns" in handler.ROUTES
        assert "/api/evolution/summary" in handler.ROUTES

    def test_routes_contains_wildcard_paths(self, handler):
        assert "/api/evolution/*/history" in handler.ROUTES
        assert "/api/evolution/*/prompt" in handler.ROUTES

    def test_routes_contains_versioned_paths(self, handler):
        assert "/api/v1/evolution" in handler.ROUTES
        assert "/api/v1/evolution/patterns" in handler.ROUTES
        assert "/api/v1/evolution/summary" in handler.ROUTES
        assert "/api/v1/evolution/*/history" in handler.ROUTES
        assert "/api/v1/evolution/*/prompt" in handler.ROUTES

    def test_routes_count(self, handler):
        assert len(handler.ROUTES) == 10


# ============================================================================
# PromptEvolver unavailability (module-level flag)
# ============================================================================


class TestEvolutionUnavailable:
    """Test behavior when EVOLUTION_AVAILABLE is False or PromptEvolver is None."""

    def test_history_503_when_flag_false(self, handler_unavailable, http):
        result = handler_unavailable.handle("/api/evolution/claude/history", {}, http)
        assert _status(result) == 503

    def test_prompt_503_when_flag_false(self, handler_unavailable, http):
        result = handler_unavailable.handle("/api/evolution/claude/prompt", {}, http)
        assert _status(result) == 503

    def test_patterns_503_when_flag_false(self, handler_unavailable, http):
        result = handler_unavailable.handle("/api/evolution/patterns", {}, http)
        assert _status(result) == 503

    def test_summary_503_when_flag_false(self, handler_unavailable, http):
        result = handler_unavailable.handle("/api/evolution/summary", {}, http)
        assert _status(result) == 503

    def test_root_503_when_flag_false(self, handler_unavailable, http):
        result = handler_unavailable.handle("/api/evolution", {}, http)
        assert _status(result) == 503

    def test_history_503_when_evolver_none(self, http):
        _handler_mod.EVOLUTION_AVAILABLE = True
        _handler_mod.PromptEvolver = None
        h = EvolutionHandler(ctx={"nomic_dir": Path("/tmp/nomic")})
        result = h.handle("/api/evolution/claude/history", {}, http)
        assert _status(result) == 503

    def test_patterns_503_when_evolver_none(self, http):
        _handler_mod.EVOLUTION_AVAILABLE = True
        _handler_mod.PromptEvolver = None
        h = EvolutionHandler(ctx={"nomic_dir": Path("/tmp/nomic")})
        result = h.handle("/api/evolution/patterns", {}, http)
        assert _status(result) == 503


# ============================================================================
# Nomic dir not configured
# ============================================================================


class TestNomicDirNotConfigured:
    """Test behavior when nomic_dir is not in context."""

    def test_history_503_no_nomic_dir(self, handler_no_nomic, http):
        result = handler_no_nomic.handle("/api/evolution/claude/history", {}, http)
        assert _status(result) == 503
        assert "not configured" in _body(result).get("error", "")

    def test_patterns_503_no_nomic_dir(self, handler_no_nomic, http):
        result = handler_no_nomic.handle("/api/evolution/patterns", {}, http)
        assert _status(result) == 503

    def test_prompt_503_no_nomic_dir(self, handler_no_nomic, http):
        result = handler_no_nomic.handle("/api/evolution/claude/prompt", {}, http)
        assert _status(result) == 503

    def test_summary_503_no_nomic_dir(self, handler_no_nomic, http):
        result = handler_no_nomic.handle("/api/evolution/summary", {}, http)
        assert _status(result) == 503
