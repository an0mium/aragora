"""Tests for the prompt evolution endpoint handler.

Tests cover:
- Pattern listing with filters and limits
- Agent prompt history retrieval
- Prompt version retrieval
- Evolution summary statistics
- Graceful degradation when EVOLUTION_AVAILABLE=False
- Rate limiting
- Path validation and routing
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Pre-mock broken transitive imports to allow importing evolution handler
# without triggering the full handlers.__init__ chain
if "aragora.server.handlers.social._slack_impl" not in sys.modules:
    sys.modules["aragora.server.handlers.social._slack_impl"] = MagicMock()

import pytest


@pytest.fixture(autouse=True)
def _clear_global_state():
    """Reset rate limiter and module-level globals between tests.

    Saves and restores EVOLUTION_AVAILABLE and PromptEvolver so that
    cross-test pollution (e.g. from a prior test that sets them to mocks
    and fails before restoring) cannot affect subsequent tests.

    The rate limiter is reset via its ``clear()`` method rather than by
    replacing ``_buckets`` with a new dict, which would break the
    backward-compatible ``_requests`` alias created in ``__init__``.
    """
    import aragora.server.handlers.evolution.handler as mod

    # Reset rate limiter buckets (use .clear() to preserve the _requests alias)
    mod._evolution_limiter.clear()

    # Save original module-level globals
    orig_avail = mod.EVOLUTION_AVAILABLE
    orig_evolver = mod.PromptEvolver

    yield

    # Restore module-level globals to prevent cross-test pollution
    mod.EVOLUTION_AVAILABLE = orig_avail
    mod.PromptEvolver = orig_evolver

    # Also clear rate limiter on teardown
    mod._evolution_limiter.clear()


def _make_handler_instance(ctx=None):
    """Create an EvolutionHandler with a mock context."""
    from aragora.server.handlers.evolution.handler import EvolutionHandler

    if ctx is None:
        ctx = {"nomic_dir": "/tmp/test_nomic"}
    return EvolutionHandler(ctx)


def _mock_http_handler(client_ip="127.0.0.1"):
    """Create a mock HTTP request handler."""
    handler = MagicMock()
    handler.client_address = (client_ip, 12345)
    handler.headers = {}
    return handler


def _parse_result(result):
    """Parse a HandlerResult into (status_code, data_dict)."""
    return result.status_code, json.loads(result.body)


# ---------------------------------------------------------------------------
# Routing / can_handle tests
# ---------------------------------------------------------------------------


class TestCanHandle:
    def test_patterns_path(self):
        h = _make_handler_instance()
        assert h.can_handle("/api/evolution/patterns") is True

    def test_summary_path(self):
        h = _make_handler_instance()
        assert h.can_handle("/api/evolution/summary") is True

    def test_history_path(self):
        h = _make_handler_instance()
        assert h.can_handle("/api/evolution/claude/history") is True

    def test_prompt_path(self):
        h = _make_handler_instance()
        assert h.can_handle("/api/evolution/claude/prompt") is True

    def test_v1_patterns_path(self):
        h = _make_handler_instance()
        assert h.can_handle("/api/v1/evolution/patterns") is True

    def test_v1_history_path(self):
        h = _make_handler_instance()
        assert h.can_handle("/api/v1/evolution/claude/history") is True

    def test_unrelated_path(self):
        h = _make_handler_instance()
        assert h.can_handle("/api/debates") is False

    def test_partial_path(self):
        h = _make_handler_instance()
        assert h.can_handle("/api/evolution") is False


# ---------------------------------------------------------------------------
# Handle routing tests
# ---------------------------------------------------------------------------


class TestHandleRouting:
    def test_non_evolution_path_returns_none(self):
        h = _make_handler_instance()
        result = h.handle("/api/debates", {}, _mock_http_handler())
        assert result is None

    def test_routes_to_patterns(self):
        h = _make_handler_instance()
        with patch.object(h, "_get_patterns", return_value=MagicMock(status_code=200)) as mock:
            h.handle("/api/evolution/patterns", {}, _mock_http_handler())
            mock.assert_called_once()

    def test_routes_to_summary(self):
        h = _make_handler_instance()
        with patch.object(h, "_get_summary", return_value=MagicMock(status_code=200)) as mock:
            h.handle("/api/evolution/summary", {}, _mock_http_handler())
            mock.assert_called_once()

    def test_routes_to_history(self):
        h = _make_handler_instance()
        with patch.object(
            h, "_get_evolution_history", return_value=MagicMock(status_code=200)
        ) as mock:
            h.handle("/api/evolution/claude/history", {"limit": "5"}, _mock_http_handler())
            mock.assert_called_once()

    def test_routes_to_prompt(self):
        h = _make_handler_instance()
        with patch.object(
            h, "_get_prompt_version", return_value=MagicMock(status_code=200)
        ) as mock:
            h.handle("/api/evolution/claude/prompt", {}, _mock_http_handler())
            mock.assert_called_once()


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    def test_rate_limit_exceeded(self):
        import aragora.server.handlers.evolution.handler as mod

        h = _make_handler_instance()
        mock_handler = _mock_http_handler()
        # Exhaust rate limiter
        with patch.object(mod._evolution_limiter, "is_allowed", return_value=False):
            result = h.handle("/api/evolution/patterns", {}, mock_handler)
            status, data = _parse_result(result)
            assert status == 429
            assert "Rate limit" in data["error"]


# ---------------------------------------------------------------------------
# EVOLUTION_AVAILABLE=False degradation
# ---------------------------------------------------------------------------


class TestEvolutionUnavailable:
    def test_patterns_returns_503(self):
        h = _make_handler_instance()
        with patch("aragora.server.handlers.evolution.handler.EVOLUTION_AVAILABLE", False):
            result = h._get_patterns(None, 10)
        status, data = _parse_result(result)
        assert status == 503
        assert "not available" in data["error"]

    def test_history_returns_503(self):
        h = _make_handler_instance()
        with patch("aragora.server.handlers.evolution.handler.EVOLUTION_AVAILABLE", False):
            result = h._get_evolution_history("claude", 10)
        status, data = _parse_result(result)
        assert status == 503

    def test_prompt_version_returns_503(self):
        h = _make_handler_instance()
        with patch("aragora.server.handlers.evolution.handler.EVOLUTION_AVAILABLE", False):
            result = h._get_prompt_version("claude", None)
        status, data = _parse_result(result)
        assert status == 503

    def test_summary_returns_503(self):
        h = _make_handler_instance()
        with patch("aragora.server.handlers.evolution.handler.EVOLUTION_AVAILABLE", False):
            result = h._get_summary()
        status, data = _parse_result(result)
        assert status == 503


# ---------------------------------------------------------------------------
# Nomic dir not configured
# ---------------------------------------------------------------------------


class TestNomicDirNotConfigured:
    def test_patterns_no_nomic_dir(self):
        h = _make_handler_instance(ctx={})
        with (
            patch("aragora.server.handlers.evolution.handler.EVOLUTION_AVAILABLE", True),
            patch("aragora.server.handlers.evolution.handler.PromptEvolver", MagicMock()),
        ):
            result = h._get_patterns(None, 10)
        status, data = _parse_result(result)
        assert status == 503
        assert "not configured" in data["error"]

    def test_history_no_nomic_dir(self):
        h = _make_handler_instance(ctx={})
        with (
            patch("aragora.server.handlers.evolution.handler.EVOLUTION_AVAILABLE", True),
            patch("aragora.server.handlers.evolution.handler.PromptEvolver", MagicMock()),
        ):
            result = h._get_evolution_history("claude", 10)
        status, data = _parse_result(result)
        assert status == 503

    def test_prompt_version_no_nomic_dir(self):
        h = _make_handler_instance(ctx={})
        with (
            patch("aragora.server.handlers.evolution.handler.EVOLUTION_AVAILABLE", True),
            patch("aragora.server.handlers.evolution.handler.PromptEvolver", MagicMock()),
        ):
            result = h._get_prompt_version("claude", None)
        status, data = _parse_result(result)
        assert status == 503

    def test_summary_no_nomic_dir(self):
        h = _make_handler_instance(ctx={})
        with (
            patch("aragora.server.handlers.evolution.handler.EVOLUTION_AVAILABLE", True),
            patch("aragora.server.handlers.evolution.handler.PromptEvolver", MagicMock()),
        ):
            result = h._get_summary()
        status, data = _parse_result(result)
        assert status == 503


# ---------------------------------------------------------------------------
# Successful data retrieval (mocked PromptEvolver)
# ---------------------------------------------------------------------------


class TestGetPatterns:
    def test_returns_patterns(self):
        mock_evolver_cls = MagicMock()
        mock_evolver_inst = MagicMock()
        mock_evolver_inst.get_top_patterns.return_value = [
            {"pattern": "chain_of_thought", "frequency": 5},
        ]
        mock_evolver_cls.return_value = mock_evolver_inst

        h = _make_handler_instance()
        with (
            patch("aragora.server.handlers.evolution.handler.EVOLUTION_AVAILABLE", True),
            patch("aragora.server.handlers.evolution.handler.PromptEvolver", mock_evolver_cls),
            patch(
                "aragora.server.handlers.evolution.handler.get_db_path", return_value="/tmp/test.db"
            ),
        ):
            result = h._get_patterns("strategy", 10)

        status, data = _parse_result(result)
        assert status == 200
        assert data["count"] == 1
        assert data["filter"] == "strategy"
        assert len(data["patterns"]) == 1

    def test_patterns_exception_returns_500(self):
        mock_evolver_cls = MagicMock(side_effect=Exception("db error"))

        h = _make_handler_instance()
        with (
            patch("aragora.server.handlers.evolution.handler.EVOLUTION_AVAILABLE", True),
            patch("aragora.server.handlers.evolution.handler.PromptEvolver", mock_evolver_cls),
            patch(
                "aragora.server.handlers.evolution.handler.get_db_path", return_value="/tmp/test.db"
            ),
        ):
            result = h._get_patterns(None, 10)

        status, data = _parse_result(result)
        assert status == 500


class TestGetEvolutionHistory:
    def test_returns_history(self):
        mock_evolver_cls = MagicMock()
        mock_evolver_inst = MagicMock()
        mock_evolver_inst.get_evolution_history.return_value = [
            {"version": 1, "strategy": "mutation"},
            {"version": 2, "strategy": "crossover"},
        ]
        mock_evolver_cls.return_value = mock_evolver_inst

        h = _make_handler_instance()
        with (
            patch("aragora.server.handlers.evolution.handler.EVOLUTION_AVAILABLE", True),
            patch("aragora.server.handlers.evolution.handler.PromptEvolver", mock_evolver_cls),
            patch(
                "aragora.server.handlers.evolution.handler.get_db_path", return_value="/tmp/test.db"
            ),
        ):
            result = h._get_evolution_history("claude", 10)

        status, data = _parse_result(result)
        assert status == 200
        assert data["agent"] == "claude"
        assert data["count"] == 2

    def test_history_exception_returns_500(self):
        mock_evolver_cls = MagicMock(side_effect=RuntimeError("broken"))

        h = _make_handler_instance()
        with (
            patch("aragora.server.handlers.evolution.handler.EVOLUTION_AVAILABLE", True),
            patch("aragora.server.handlers.evolution.handler.PromptEvolver", mock_evolver_cls),
            patch(
                "aragora.server.handlers.evolution.handler.get_db_path", return_value="/tmp/test.db"
            ),
        ):
            result = h._get_evolution_history("claude", 5)

        status, _ = _parse_result(result)
        assert status == 500


class TestGetPromptVersion:
    def test_returns_prompt(self):
        mock_evolver_cls = MagicMock()
        mock_evolver_inst = MagicMock()
        mock_version = SimpleNamespace(
            version=3,
            prompt="You are a helpful assistant.",
            performance_score=0.85,
            debates_count=10,
            consensus_rate=0.7,
            metadata={"strategy": "mutation"},
            created_at="2025-01-01T00:00:00",
        )
        mock_evolver_inst.get_prompt_version.return_value = mock_version
        mock_evolver_cls.return_value = mock_evolver_inst

        h = _make_handler_instance()
        with (
            patch("aragora.server.handlers.evolution.handler.EVOLUTION_AVAILABLE", True),
            patch("aragora.server.handlers.evolution.handler.PromptEvolver", mock_evolver_cls),
            patch(
                "aragora.server.handlers.evolution.handler.get_db_path", return_value="/tmp/test.db"
            ),
        ):
            result = h._get_prompt_version("claude", 3)

        status, data = _parse_result(result)
        assert status == 200
        assert data["agent"] == "claude"
        assert data["version"] == 3
        assert data["performance_score"] == 0.85
        assert data["prompt"] == "You are a helpful assistant."

    def test_prompt_not_found_returns_404(self):
        mock_evolver_cls = MagicMock()
        mock_evolver_inst = MagicMock()
        mock_evolver_inst.get_prompt_version.return_value = None
        mock_evolver_cls.return_value = mock_evolver_inst

        h = _make_handler_instance()
        with (
            patch("aragora.server.handlers.evolution.handler.EVOLUTION_AVAILABLE", True),
            patch("aragora.server.handlers.evolution.handler.PromptEvolver", mock_evolver_cls),
            patch(
                "aragora.server.handlers.evolution.handler.get_db_path", return_value="/tmp/test.db"
            ),
        ):
            result = h._get_prompt_version("unknown_agent", None)

        status, data = _parse_result(result)
        assert status == 404

    def test_prompt_exception_returns_500(self):
        mock_evolver_cls = MagicMock(side_effect=Exception("boom"))

        h = _make_handler_instance()
        with (
            patch("aragora.server.handlers.evolution.handler.EVOLUTION_AVAILABLE", True),
            patch("aragora.server.handlers.evolution.handler.PromptEvolver", mock_evolver_cls),
            patch(
                "aragora.server.handlers.evolution.handler.get_db_path", return_value="/tmp/test.db"
            ),
        ):
            result = h._get_prompt_version("claude", 1)

        status, _ = _parse_result(result)
        assert status == 500


class TestGetSummary:
    """Test the _get_summary method.

    Uses ``patch`` context managers to isolate EVOLUTION_AVAILABLE and
    PromptEvolver from cross-test pollution instead of direct module
    global mutation with try/finally.
    """

    def test_returns_summary(self):
        # Mock evolver with connection context manager
        mock_cursor = MagicMock()
        # Sequence of fetchone/fetchall calls in _get_summary:
        # 1) COUNT(*) FROM prompt_versions -> (5,)
        # 2) COUNT(DISTINCT agent_name) -> (2,)
        # 3) COUNT(*) FROM extracted_patterns -> (8,)
        # 4) pattern_type distribution -> [("strategy", 5), ("structure", 3)]
        # 5) top agents -> [("claude", 0.9, 3), ("gpt4", 0.8, 2)]
        # 6) recent activity -> [("claude", "mutation", "2025-01-01")]
        mock_cursor.fetchone.side_effect = [(5,), (2,), (8,)]
        mock_cursor.fetchall.side_effect = [
            [("strategy", 5), ("structure", 3)],
            [("claude", 0.9, 3), ("gpt4", 0.8, 2)],
            [("claude", "mutation", "2025-01-01")],
        ]

        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        mock_evolver_cls = MagicMock()
        mock_evolver_inst = MagicMock()
        mock_evolver_inst.connection.return_value = mock_conn
        mock_evolver_cls.return_value = mock_evolver_inst

        h = _make_handler_instance()
        with (
            patch("aragora.server.handlers.evolution.handler.EVOLUTION_AVAILABLE", True),
            patch("aragora.server.handlers.evolution.handler.PromptEvolver", mock_evolver_cls),
            patch(
                "aragora.server.handlers.evolution.handler.get_db_path", return_value="/tmp/test.db"
            ),
        ):
            result = h._get_summary()

        status, data = _parse_result(result)
        assert status == 200
        assert data["total_prompt_versions"] == 5
        assert data["total_agents"] == 2
        assert data["total_patterns"] == 8
        assert data["pattern_distribution"] == {"strategy": 5, "structure": 3}
        assert len(data["top_agents"]) == 2
        assert len(data["recent_activity"]) == 1

    def test_summary_exception_returns_500(self):
        mock_evolver_cls = MagicMock(side_effect=Exception("db crash"))

        h = _make_handler_instance()
        with (
            patch("aragora.server.handlers.evolution.handler.EVOLUTION_AVAILABLE", True),
            patch("aragora.server.handlers.evolution.handler.PromptEvolver", mock_evolver_cls),
            patch(
                "aragora.server.handlers.evolution.handler.get_db_path", return_value="/tmp/test.db"
            ),
        ):
            result = h._get_summary()

        status, _ = _parse_result(result)
        assert status == 500


# ---------------------------------------------------------------------------
# Query parameter clamping
# ---------------------------------------------------------------------------


class TestQueryParamClamping:
    def test_limit_clamped_to_min(self):
        """Limit below 1 should be clamped to 1."""
        h = _make_handler_instance()
        with patch.object(h, "_get_patterns", return_value=MagicMock(status_code=200)) as mock:
            h.handle("/api/evolution/patterns", {"limit": "0"}, _mock_http_handler())
            # limit = min(max(0, 1), 50) = 1
            mock.assert_called_once_with(None, 1)

    def test_limit_clamped_to_max(self):
        """Limit above 50 should be clamped to 50."""
        h = _make_handler_instance()
        with patch.object(h, "_get_patterns", return_value=MagicMock(status_code=200)) as mock:
            h.handle("/api/evolution/patterns", {"limit": "100"}, _mock_http_handler())
            mock.assert_called_once_with(None, 50)

    def test_type_filter_passed(self):
        """Pattern type filter is passed through."""
        h = _make_handler_instance()
        with patch.object(h, "_get_patterns", return_value=MagicMock(status_code=200)) as mock:
            h.handle(
                "/api/evolution/patterns", {"type": "strategy", "limit": "5"}, _mock_http_handler()
            )
            mock.assert_called_once_with("strategy", 5)
