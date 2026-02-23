"""Comprehensive tests for AgentFlipsMixin (aragora/server/handlers/agents/agent_flips.py).

Tests cover all 3 endpoint methods provided by the mixin:
- GET /api/agent/{name}/flips - Get agent position flips
- GET /api/flips/recent - Get recent flips across all agents
- GET /api/flips/summary - Get flip summary for dashboard

Each endpoint is tested for:
- Happy path with valid data
- No nomic_dir (fallback defaults)
- Edge cases (empty data, large limits)
- Input validation (limit capping)
- Error handling (exceptions in FlipDetector)
- Caching behavior
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_caches():
    """Reset caches and rate limiters before each test."""
    try:
        from aragora.server.handlers.admin.cache import clear_cache
        clear_cache()
    except ImportError:
        pass

    try:
        from aragora.server.middleware.rate_limit.registry import reset_rate_limiters
        reset_rate_limiters()
    except ImportError:
        pass

    try:
        from aragora.server.handlers.agents import agents as agents_mod
        agents_mod._agent_limiter = agents_mod.RateLimiter(requests_per_minute=60)
    except (ImportError, AttributeError):
        pass

    try:
        from aragora.server.handlers.utils import rate_limit as rl_mod
        with rl_mod._limiters_lock:
            rl_mod._limiters.clear()
    except (ImportError, AttributeError):
        pass

    yield

    try:
        from aragora.server.handlers.admin.cache import clear_cache
        clear_cache()
    except ImportError:
        pass

    try:
        from aragora.server.middleware.rate_limit.registry import reset_rate_limiters
        reset_rate_limiters()
    except ImportError:
        pass


@pytest.fixture
def handler():
    """Create an AgentsHandler with empty server context."""
    from aragora.server.handlers.agents.agents import AgentsHandler
    return AgentsHandler(server_context={})


@pytest.fixture
def mock_http_handler():
    """Create mock HTTP handler with client address and empty headers."""
    h = MagicMock()
    h.client_address = ("127.0.0.1", 54321)
    h.headers = {}
    return h


def _make_flip_event(**overrides) -> MagicMock:
    """Create a mock FlipEvent with to_dict support."""
    flip = MagicMock()
    flip_data = {
        "id": overrides.get("id", "flip-001"),
        "agent_name": overrides.get("agent_name", "claude"),
        "original_claim": overrides.get("original_claim", "X is true"),
        "new_claim": overrides.get("new_claim", "X is false"),
        "original_confidence": overrides.get("original_confidence", 0.9),
        "new_confidence": overrides.get("new_confidence", 0.8),
        "original_debate_id": overrides.get("original_debate_id", "debate-1"),
        "new_debate_id": overrides.get("new_debate_id", "debate-2"),
        "similarity_score": overrides.get("similarity_score", 0.85),
        "flip_type": overrides.get("flip_type", "contradiction"),
        "domain": overrides.get("domain", "technical"),
        "detected_at": overrides.get("detected_at", "2026-02-23T12:00:00"),
    }
    flip.to_dict.return_value = flip_data
    return flip


def _make_consistency_score(**overrides) -> MagicMock:
    """Create a mock AgentConsistencyScore with to_dict support."""
    score = MagicMock()
    score_data = {
        "agent_name": overrides.get("agent_name", "claude"),
        "total_positions": overrides.get("total_positions", 20),
        "total_flips": overrides.get("total_flips", 3),
        "contradictions": overrides.get("contradictions", 1),
        "refinements": overrides.get("refinements", 1),
        "retractions": overrides.get("retractions", 0),
        "qualifications": overrides.get("qualifications", 1),
        "consistency_score": overrides.get("consistency_score", 0.85),
        "flip_rate": overrides.get("flip_rate", 0.15),
        "avg_confidence_on_flip": overrides.get("avg_confidence_on_flip", 0.7),
        "domains_with_flips": overrides.get("domains_with_flips", ["technical"]),
    }
    score.to_dict.return_value = score_data
    return score


# ===========================================================================
# _get_agent_flips: GET /api/agent/{name}/flips
# ===========================================================================

class TestGetAgentFlips:
    """Tests for the _get_agent_flips endpoint."""

    def test_happy_path_with_nomic_dir(self, handler):
        """Returns flips and consistency when nomic_dir is available."""
        mock_flip = _make_flip_event(agent_name="claude")
        mock_consistency = _make_consistency_score(agent_name="claude")
        mock_detector = MagicMock()
        mock_detector.detect_flips_for_agent.return_value = [mock_flip]
        mock_detector.get_agent_consistency.return_value = mock_consistency

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result = handler._get_agent_flips("claude", 20)
                    assert _status(result) == 200
                    body = _body(result)
                    assert body["agent"] == "claude"
                    assert len(body["flips"]) == 1
                    assert body["count"] == 1
                    assert "consistency" in body

    def test_no_nomic_dir_returns_empty_defaults(self, handler):
        """Returns empty defaults when nomic_dir is None."""
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = handler._get_agent_flips("claude", 20)
            assert _status(result) == 200
            body = _body(result)
            assert body["agent"] == "claude"
            assert body["flips"] == []
            assert body["count"] == 0
            assert body["consistency"]["agent_name"] == "claude"
            assert body["consistency"]["total_positions"] == 0
            assert body["consistency"]["total_flips"] == 0
            assert body["consistency"]["consistency_score"] == 1.0

    def test_limit_capped_at_100(self, handler):
        """Limit is capped at 100 via min(limit, 100)."""
        mock_detector = MagicMock()
        mock_detector.detect_flips_for_agent.return_value = []
        mock_detector.get_agent_consistency.return_value = _make_consistency_score()

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    handler._get_agent_flips("claude", 200)
                    mock_detector.detect_flips_for_agent.assert_called_once_with(
                        "claude", lookback_positions=100
                    )

    def test_limit_below_100_passed_through(self, handler):
        """Limits below 100 are used as-is."""
        mock_detector = MagicMock()
        mock_detector.detect_flips_for_agent.return_value = []
        mock_detector.get_agent_consistency.return_value = _make_consistency_score()

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    handler._get_agent_flips("gpt4", 50)
                    mock_detector.detect_flips_for_agent.assert_called_once_with(
                        "gpt4", lookback_positions=50
                    )

    def test_limit_exactly_100(self, handler):
        """Limit of exactly 100 is passed through."""
        mock_detector = MagicMock()
        mock_detector.detect_flips_for_agent.return_value = []
        mock_detector.get_agent_consistency.return_value = _make_consistency_score()

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    handler._get_agent_flips("claude", 100)
                    mock_detector.detect_flips_for_agent.assert_called_once_with(
                        "claude", lookback_positions=100
                    )

    def test_multiple_flips_returned(self, handler):
        """Multiple flips are returned in the response."""
        flips = [
            _make_flip_event(id=f"flip-{i}", agent_name="claude")
            for i in range(5)
        ]
        mock_detector = MagicMock()
        mock_detector.detect_flips_for_agent.return_value = flips
        mock_detector.get_agent_consistency.return_value = _make_consistency_score()

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result = handler._get_agent_flips("claude", 20)
                    body = _body(result)
                    assert body["count"] == 5
                    assert len(body["flips"]) == 5

    def test_empty_flips_with_nomic_dir(self, handler):
        """When FlipDetector finds no flips, returns empty list with consistency."""
        mock_detector = MagicMock()
        mock_detector.detect_flips_for_agent.return_value = []
        mock_detector.get_agent_consistency.return_value = _make_consistency_score(
            total_flips=0, consistency_score=1.0
        )

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result = handler._get_agent_flips("claude", 20)
                    assert _status(result) == 200
                    body = _body(result)
                    assert body["flips"] == []
                    assert body["count"] == 0

    def test_flip_detector_exception_returns_500(self, handler):
        """When FlipDetector raises, handle_errors returns 500."""
        mock_detector = MagicMock()
        mock_detector.detect_flips_for_agent.side_effect = RuntimeError("DB corrupt")

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result = handler._get_agent_flips("claude", 20)
                    assert _status(result) == 500

    def test_consistency_exception_returns_400(self, handler):
        """When get_agent_consistency raises ValueError, handle_errors returns 400."""
        mock_detector = MagicMock()
        mock_detector.detect_flips_for_agent.return_value = []
        mock_detector.get_agent_consistency.side_effect = ValueError("No data")

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result = handler._get_agent_flips("claude", 20)
                    assert _status(result) == 400

    def test_os_error_returns_500(self, handler):
        """OSError in FlipDetector returns 500."""
        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                side_effect=OSError("No such file"),
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result = handler._get_agent_flips("claude", 20)
                    assert _status(result) == 500

    def test_different_agent_names(self, handler):
        """Works with different agent names."""
        for agent_name in ["gpt4", "gemini-pro", "llama3"]:
            with patch.object(handler, "get_nomic_dir", return_value=None):
                result = handler._get_agent_flips(agent_name, 10)
                assert _status(result) == 200
                body = _body(result)
                assert body["agent"] == agent_name

    def test_consistency_data_passed_through(self, handler):
        """Consistency data from FlipDetector is serialized via to_dict."""
        consistency_data = {
            "agent_name": "claude",
            "total_positions": 50,
            "total_flips": 8,
            "consistency_score": 0.72,
            "flip_rate": 0.16,
        }
        mock_consistency = MagicMock()
        mock_consistency.to_dict.return_value = consistency_data
        mock_detector = MagicMock()
        mock_detector.detect_flips_for_agent.return_value = []
        mock_detector.get_agent_consistency.return_value = mock_consistency

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result = handler._get_agent_flips("claude", 20)
                    body = _body(result)
                    assert body["consistency"] == consistency_data


# ===========================================================================
# _get_agent_flips via handle(): routed through /api/agent/{name}/flips
# ===========================================================================

class TestAgentFlipsViaHandle:
    """Tests for flips accessed through the handle() routing method."""

    @pytest.mark.asyncio
    async def test_route_via_handle(self, handler, mock_http_handler):
        """GET /api/agent/claude/flips routes to _get_agent_flips."""
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = await handler.handle(
                "/api/agent/claude/flips", {}, mock_http_handler
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["agent"] == "claude"
            assert body["flips"] == []

    @pytest.mark.asyncio
    async def test_route_versioned_path(self, handler, mock_http_handler):
        """GET /api/v1/agent/claude/flips also works."""
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = await handler.handle(
                "/api/v1/agent/claude/flips", {}, mock_http_handler
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["agent"] == "claude"

    @pytest.mark.asyncio
    async def test_route_with_limit_param(self, handler, mock_http_handler):
        """Limit query param is passed through to _get_agent_flips."""
        mock_detector = MagicMock()
        mock_detector.detect_flips_for_agent.return_value = []
        mock_detector.get_agent_consistency.return_value = _make_consistency_score()

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result = await handler.handle(
                        "/api/agent/claude/flips",
                        {"limit": "30"},
                        mock_http_handler,
                    )
                    assert _status(result) == 200
                    mock_detector.detect_flips_for_agent.assert_called_once_with(
                        "claude", lookback_positions=30
                    )

    @pytest.mark.asyncio
    async def test_route_default_limit_is_20(self, handler, mock_http_handler):
        """Default limit when not provided is 20."""
        mock_detector = MagicMock()
        mock_detector.detect_flips_for_agent.return_value = []
        mock_detector.get_agent_consistency.return_value = _make_consistency_score()

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    await handler.handle(
                        "/api/agent/claude/flips", {}, mock_http_handler
                    )
                    mock_detector.detect_flips_for_agent.assert_called_once_with(
                        "claude", lookback_positions=20
                    )


# ===========================================================================
# _get_recent_flips: GET /api/flips/recent
# ===========================================================================

class TestGetRecentFlips:
    """Tests for the _get_recent_flips endpoint."""

    def test_happy_path_with_nomic_dir(self, handler):
        """Returns flips and summary when nomic_dir is available."""
        mock_flips = [
            _make_flip_event(id="flip-1", agent_name="claude"),
            _make_flip_event(id="flip-2", agent_name="gpt4"),
        ]
        mock_summary = {
            "total_flips": 10,
            "by_type": {"contradiction": 5, "refinement": 5},
            "by_agent": {"claude": 3, "gpt4": 7},
            "recent_24h": 2,
        }
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = mock_flips
        mock_detector.get_flip_summary.return_value = mock_summary

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result = handler._get_recent_flips(20)
                    assert _status(result) == 200
                    body = _body(result)
                    assert len(body["flips"]) == 2
                    assert body["count"] == 2
                    assert body["summary"] == mock_summary

    def test_no_nomic_dir_returns_empty_defaults(self, handler):
        """Returns empty defaults when nomic_dir is None."""
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = handler._get_recent_flips(20)
            assert _status(result) == 200
            body = _body(result)
            assert body["flips"] == []
            assert body["count"] == 0
            assert body["summary"]["total_flips"] == 0
            assert body["summary"]["by_type"] == {}
            assert body["summary"]["by_agent"] == {}
            assert body["summary"]["recent_24h"] == 0

    def test_limit_capped_at_100(self, handler):
        """Limit is capped at 100 via min(limit, 100)."""
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = []
        mock_detector.get_flip_summary.return_value = {}

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    handler._get_recent_flips(500)
                    mock_detector.get_recent_flips.assert_called_once_with(limit=100)

    def test_limit_below_100_passed_through(self, handler):
        """Limits below 100 are passed as-is."""
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = []
        mock_detector.get_flip_summary.return_value = {}

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    handler._get_recent_flips(15)
                    mock_detector.get_recent_flips.assert_called_once_with(limit=15)

    def test_detector_exception_returns_500(self, handler):
        """When FlipDetector.get_recent_flips raises, returns 500."""
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.side_effect = RuntimeError("DB error")

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result = handler._get_recent_flips(20)
                    assert _status(result) == 500

    def test_summary_exception_returns_500(self, handler):
        """When get_flip_summary raises, returns 500."""
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = []
        mock_detector.get_flip_summary.side_effect = OSError("Disk failure")

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result = handler._get_recent_flips(20)
                    assert _status(result) == 500

    def test_empty_flips_with_nomic_dir(self, handler):
        """Empty flips from FlipDetector returns count=0."""
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = []
        mock_detector.get_flip_summary.return_value = {"total_flips": 0}

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result = handler._get_recent_flips(20)
                    assert _status(result) == 200
                    body = _body(result)
                    assert body["count"] == 0
                    assert body["flips"] == []

    def test_flip_constructor_value_error_returns_400(self, handler):
        """When FlipDetector constructor raises ValueError, returns 400."""
        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                side_effect=ValueError("Bad DB path"),
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result = handler._get_recent_flips(20)
                    assert _status(result) == 400


# ===========================================================================
# _get_recent_flips via handle(): routed through /api/flips/recent
# ===========================================================================

class TestRecentFlipsViaHandle:
    """Tests for recent flips accessed through handle() routing."""

    @pytest.mark.asyncio
    async def test_route_via_handle(self, handler, mock_http_handler):
        """GET /api/flips/recent routes to _get_recent_flips."""
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = await handler.handle(
                "/api/flips/recent", {}, mock_http_handler
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["flips"] == []
            assert body["count"] == 0

    @pytest.mark.asyncio
    async def test_route_versioned_path(self, handler, mock_http_handler):
        """GET /api/v1/flips/recent also works."""
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = await handler.handle(
                "/api/v1/flips/recent", {}, mock_http_handler
            )
            assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_route_with_limit_param(self, handler, mock_http_handler):
        """Limit query param is passed through."""
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = []
        mock_detector.get_flip_summary.return_value = {}

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result = await handler.handle(
                        "/api/flips/recent", {"limit": "10"}, mock_http_handler
                    )
                    assert _status(result) == 200
                    mock_detector.get_recent_flips.assert_called_once_with(limit=10)

    @pytest.mark.asyncio
    async def test_route_default_limit(self, handler, mock_http_handler):
        """Default limit for /api/flips/recent is 20."""
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = []
        mock_detector.get_flip_summary.return_value = {}

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    await handler.handle(
                        "/api/flips/recent", {}, mock_http_handler
                    )
                    mock_detector.get_recent_flips.assert_called_once_with(limit=20)


# ===========================================================================
# _get_flip_summary: GET /api/flips/summary
# ===========================================================================

class TestGetFlipSummary:
    """Tests for the _get_flip_summary endpoint."""

    def test_happy_path_with_nomic_dir(self, handler):
        """Returns summary from FlipDetector when nomic_dir is available."""
        mock_summary = {
            "total_flips": 42,
            "by_type": {"contradiction": 20, "refinement": 15, "retraction": 7},
            "by_agent": {"claude": 15, "gpt4": 12, "gemini": 15},
            "recent_24h": 5,
        }
        mock_detector = MagicMock()
        mock_detector.get_flip_summary.return_value = mock_summary

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result = handler._get_flip_summary()
                    assert _status(result) == 200
                    body = _body(result)
                    assert body["total_flips"] == 42
                    assert body["by_type"]["contradiction"] == 20
                    assert body["by_agent"]["claude"] == 15
                    assert body["recent_24h"] == 5

    def test_no_nomic_dir_returns_empty_defaults(self, handler):
        """Returns empty defaults when nomic_dir is None."""
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = handler._get_flip_summary()
            assert _status(result) == 200
            body = _body(result)
            assert body["total_flips"] == 0
            assert body["by_type"] == {}
            assert body["by_agent"] == {}
            assert body["recent_24h"] == 0

    def test_detector_exception_returns_500(self, handler):
        """When get_flip_summary raises, returns 500."""
        mock_detector = MagicMock()
        mock_detector.get_flip_summary.side_effect = RuntimeError("DB error")

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result = handler._get_flip_summary()
                    assert _status(result) == 500

    def test_value_error_returns_400(self, handler):
        """ValueError from FlipDetector returns 400."""
        mock_detector = MagicMock()
        mock_detector.get_flip_summary.side_effect = ValueError("Invalid data")

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result = handler._get_flip_summary()
                    assert _status(result) == 400

    def test_os_error_returns_500(self, handler):
        """OSError from FlipDetector returns 500."""
        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                side_effect=OSError("No access"),
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result = handler._get_flip_summary()
                    assert _status(result) == 500

    def test_summary_empty_dict(self, handler):
        """When FlipDetector returns empty dict, it is passed through."""
        mock_detector = MagicMock()
        mock_detector.get_flip_summary.return_value = {}

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result = handler._get_flip_summary()
                    assert _status(result) == 200
                    body = _body(result)
                    assert body == {}


# ===========================================================================
# _get_flip_summary via handle(): routed through /api/flips/summary
# ===========================================================================

class TestFlipSummaryViaHandle:
    """Tests for flip summary accessed through handle() routing."""

    @pytest.mark.asyncio
    async def test_route_via_handle(self, handler, mock_http_handler):
        """GET /api/flips/summary routes to _get_flip_summary."""
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = await handler.handle(
                "/api/flips/summary", {}, mock_http_handler
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["total_flips"] == 0

    @pytest.mark.asyncio
    async def test_route_versioned_path(self, handler, mock_http_handler):
        """GET /api/v1/flips/summary also works."""
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = await handler.handle(
                "/api/v1/flips/summary", {}, mock_http_handler
            )
            assert _status(result) == 200


# ===========================================================================
# can_handle tests for flip-related paths
# ===========================================================================

class TestCanHandle:
    """Tests for can_handle on flip-related paths."""

    def test_agent_flips_path(self, handler):
        assert handler.can_handle("/api/agent/claude/flips")

    def test_agent_flips_versioned(self, handler):
        assert handler.can_handle("/api/v1/agent/claude/flips")

    def test_flips_recent_path(self, handler):
        assert handler.can_handle("/api/flips/recent")

    def test_flips_recent_versioned(self, handler):
        assert handler.can_handle("/api/v1/flips/recent")

    def test_flips_summary_path(self, handler):
        assert handler.can_handle("/api/flips/summary")

    def test_flips_summary_versioned(self, handler):
        assert handler.can_handle("/api/v1/flips/summary")

    def test_unrelated_path(self, handler):
        assert not handler.can_handle("/api/debates")


# ===========================================================================
# Caching behavior
# ===========================================================================

class TestCaching:
    """Tests that caching decorators work correctly for flip endpoints."""

    def test_agent_flips_cached_on_second_call(self, handler):
        """Second call to _get_agent_flips returns cached result."""
        mock_detector = MagicMock()
        mock_detector.detect_flips_for_agent.return_value = [_make_flip_event()]
        mock_detector.get_agent_consistency.return_value = _make_consistency_score()

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result1 = handler._get_agent_flips("claude", 20)
                    result2 = handler._get_agent_flips("claude", 20)
                    assert _status(result1) == 200
                    assert _status(result2) == 200
                    # FlipDetector should only be called once due to caching
                    assert mock_detector.detect_flips_for_agent.call_count == 1

    def test_agent_flips_different_agents_not_cached(self, handler):
        """Different agent names get different cache entries."""
        mock_detector = MagicMock()
        mock_detector.detect_flips_for_agent.return_value = []
        mock_detector.get_agent_consistency.return_value = _make_consistency_score()

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    handler._get_agent_flips("claude", 20)
                    handler._get_agent_flips("gpt4", 20)
                    assert mock_detector.detect_flips_for_agent.call_count == 2

    def test_recent_flips_cached_on_second_call(self, handler):
        """Second call to _get_recent_flips returns cached result."""
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = []
        mock_detector.get_flip_summary.return_value = {"total_flips": 0}

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    handler._get_recent_flips(20)
                    handler._get_recent_flips(20)
                    assert mock_detector.get_recent_flips.call_count == 1

    def test_flip_summary_cached_on_second_call(self, handler):
        """Second call to _get_flip_summary returns cached result."""
        mock_detector = MagicMock()
        mock_detector.get_flip_summary.return_value = {"total_flips": 0}

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    handler._get_flip_summary()
                    handler._get_flip_summary()
                    assert mock_detector.get_flip_summary.call_count == 1

    def test_recent_flips_different_limits_not_cached(self, handler):
        """Different limits get different cache entries."""
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.return_value = []
        mock_detector.get_flip_summary.return_value = {}

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    handler._get_recent_flips(10)
                    handler._get_recent_flips(50)
                    assert mock_detector.get_recent_flips.call_count == 2


# ===========================================================================
# Edge cases
# ===========================================================================

class TestEdgeCases:
    """Miscellaneous edge case tests."""

    def test_agent_flips_limit_zero(self, handler):
        """Limit of zero works without error."""
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = handler._get_agent_flips("claude", 0)
            assert _status(result) == 200

    def test_agent_flips_negative_limit(self, handler):
        """Negative limit is handled (min(-5, 100) = -5, up to FlipDetector)."""
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = handler._get_agent_flips("claude", -5)
            assert _status(result) == 200

    def test_recent_flips_limit_zero(self, handler):
        """Limit of zero works without error."""
        with patch.object(handler, "get_nomic_dir", return_value=None):
            result = handler._get_recent_flips(0)
            assert _status(result) == 200

    def test_agent_flips_key_error_returns_404(self, handler):
        """KeyError in FlipDetector returns 404."""
        mock_detector = MagicMock()
        mock_detector.detect_flips_for_agent.side_effect = KeyError("missing")

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result = handler._get_agent_flips("claude", 20)
                    assert _status(result) == 404

    def test_recent_flips_key_error_returns_404(self, handler):
        """KeyError in FlipDetector.get_recent_flips returns 404."""
        mock_detector = MagicMock()
        mock_detector.get_recent_flips.side_effect = KeyError("no key")

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result = handler._get_recent_flips(20)
                    assert _status(result) == 404

    def test_flip_summary_key_error_returns_404(self, handler):
        """KeyError in FlipDetector.get_flip_summary returns 404."""
        mock_detector = MagicMock()
        mock_detector.get_flip_summary.side_effect = KeyError("missing")

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/nomic/positions.db",
                ):
                    result = handler._get_flip_summary()
                    assert _status(result) == 404

    def test_get_db_path_called_with_correct_args(self, handler):
        """get_db_path is called with DatabaseType.POSITIONS and nomic_dir."""
        from aragora.persistence.db_config import DatabaseType

        mock_detector = MagicMock()
        mock_detector.detect_flips_for_agent.return_value = []
        mock_detector.get_agent_consistency.return_value = _make_consistency_score()

        nomic_path = Path("/tmp/test_nomic")
        with patch.object(handler, "get_nomic_dir", return_value=nomic_path):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ):
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value="/tmp/test_nomic/positions.db",
                ) as mock_get_db_path:
                    handler._get_agent_flips("claude", 20)
                    mock_get_db_path.assert_called_once_with(
                        DatabaseType.POSITIONS, nomic_path
                    )

    def test_flip_detector_receives_db_path_string(self, handler):
        """FlipDetector is instantiated with string from get_db_path."""
        mock_detector = MagicMock()
        mock_detector.detect_flips_for_agent.return_value = []
        mock_detector.get_agent_consistency.return_value = _make_consistency_score()

        with patch.object(handler, "get_nomic_dir", return_value=Path("/tmp/nomic")):
            with patch(
                "aragora.insights.flip_detector.FlipDetector",
                return_value=mock_detector,
            ) as mock_cls:
                with patch(
                    "aragora.server.handlers.agents.agent_flips.get_db_path",
                    return_value=Path("/tmp/nomic/positions.db"),
                ):
                    handler._get_agent_flips("claude", 20)
                    mock_cls.assert_called_once_with(
                        str(Path("/tmp/nomic/positions.db"))
                    )
