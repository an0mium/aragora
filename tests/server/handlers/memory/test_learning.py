"""Tests for LearningHandler - cross-cycle learning analytics endpoints."""

from __future__ import annotations

import json
import sqlite3
import sys
import types as _types_mod
from collections import defaultdict
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

import pytest

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

from aragora.server.handlers.memory.learning import (
    LearningHandler,
    MEMORY_READ_PERMISSION,
    MEMORY_WRITE_PERMISSION,
    _learning_limiter,
)
from aragora.server.handlers.secure import ForbiddenError, UnauthorizedError


# =============================================================================
# Helpers
# =============================================================================


def parse_response(result) -> dict:
    """Parse HandlerResult body to dict."""
    return json.loads(result.body.decode("utf-8"))


class MockAuthContext:
    """Mock authorization context."""

    def __init__(self, user_id: str = "user-123", permissions: list | None = None):
        self.user_id = user_id
        self.permissions = permissions or [MEMORY_READ_PERMISSION]

    def has_permission(self, perm: str) -> bool:
        return perm in self.permissions


class MockHandler:
    """Mock HTTP handler for testing."""

    def __init__(self, client_ip: str = "127.0.0.1"):
        self.headers = {"X-Forwarded-For": client_ip, "Content-Length": "0"}
        self.client_address = (client_ip, 12345)
        self.rfile = MagicMock()
        self.rfile.read.return_value = b""


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_rate_limiters():
    """Reset rate limiter state between tests."""
    _learning_limiter._buckets.clear()
    yield


@pytest.fixture
def nomic_dir(tmp_path):
    """Create a temporary nomic directory."""
    nomic = tmp_path / "nomic"
    nomic.mkdir()
    return nomic


@pytest.fixture
def handler(nomic_dir):
    """Create a LearningHandler with a configured nomic_dir."""
    return LearningHandler(ctx={"nomic_dir": str(nomic_dir)})


@pytest.fixture
def handler_no_nomic():
    """Create a LearningHandler without nomic_dir configured."""
    return LearningHandler(ctx={})


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler."""
    return MockHandler()


@pytest.fixture
def auth_context():
    """Create a mock auth context with memory:read permission."""
    return MockAuthContext()


@pytest.fixture
def auth_context_no_perms():
    """Create a mock auth context without permissions."""
    return MockAuthContext(permissions=[])


# =============================================================================
# Test Handler Routing
# =============================================================================


class TestHandlerRouting:
    """Tests for can_handle and route dispatch."""

    def test_can_handle_cycles(self, handler):
        assert handler.can_handle("/api/v1/learning/cycles") is True

    def test_can_handle_patterns(self, handler):
        assert handler.can_handle("/api/v1/learning/patterns") is True

    def test_can_handle_agent_evolution(self, handler):
        assert handler.can_handle("/api/v1/learning/agent-evolution") is True

    def test_can_handle_insights(self, handler):
        assert handler.can_handle("/api/v1/learning/insights") is True

    def test_cannot_handle_unknown_route(self, handler):
        assert handler.can_handle("/api/v1/learning/unknown") is False

    def test_cannot_handle_partial_route(self, handler):
        assert handler.can_handle("/api/v1/learning") is False

    def test_cannot_handle_other_api(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_routes_list_matches(self, handler):
        """All ROUTES entries should be handled."""
        for route in LearningHandler.ROUTES:
            assert handler.can_handle(route) is True


# =============================================================================
# Test Authentication and RBAC
# =============================================================================


class TestAuthenticationAndRBAC:
    """Tests for authentication and permission checks on the handle() method."""

    @pytest.mark.asyncio
    async def test_unauthenticated_returns_401(self, handler, mock_http_handler):
        """Unauthenticated requests should get 401."""
        with patch.object(handler, "get_auth_context", side_effect=UnauthorizedError("No token")):
            result = await handler.handle("/api/v1/learning/cycles", {}, mock_http_handler)
        assert result.status_code == 401
        data = parse_response(result)
        assert "Authentication required" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_forbidden_returns_403(self, handler, mock_http_handler):
        """Users without memory:read permission should get 403."""
        mock_ctx = MockAuthContext(permissions=[])
        with (
            patch.object(handler, "get_auth_context", return_value=mock_ctx),
            patch.object(
                handler, "check_permission", side_effect=ForbiddenError("Permission denied")
            ),
        ):
            result = await handler.handle("/api/v1/learning/patterns", {}, mock_http_handler)
        assert result.status_code == 403
        data = parse_response(result)
        assert "Permission denied" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_authenticated_request_proceeds(self, handler, mock_http_handler):
        """Authenticated users with permissions should get 200."""
        mock_ctx = MockAuthContext()
        with (
            patch.object(handler, "get_auth_context", return_value=mock_ctx),
            patch.object(handler, "check_permission", return_value=True),
        ):
            result = await handler.handle("/api/v1/learning/cycles", {}, mock_http_handler)
        # Should return data (even if empty), not an auth error
        assert result.status_code in (200, 503)

    @pytest.mark.asyncio
    async def test_unknown_route_returns_none(self, handler, mock_http_handler):
        """Unknown routes should return None (not handled)."""
        mock_ctx = MockAuthContext()
        with (
            patch.object(handler, "get_auth_context", return_value=mock_ctx),
            patch.object(handler, "check_permission", return_value=True),
        ):
            result = await handler.handle("/api/v1/unknown", {}, mock_http_handler)
        assert result is None


# =============================================================================
# Test Rate Limiting
# =============================================================================


class TestRateLimiting:
    """Tests for rate limiting on the handle() method."""

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded_returns_429(self, handler, mock_http_handler):
        """When rate limit is exceeded, return 429."""
        with patch.object(_learning_limiter, "is_allowed", return_value=False):
            result = await handler.handle("/api/v1/learning/cycles", {}, mock_http_handler)
        assert result.status_code == 429
        data = parse_response(result)
        assert "Rate limit" in data.get("error", "")

    @pytest.mark.asyncio
    async def test_rate_limit_allowed_proceeds(self, handler, mock_http_handler):
        """When rate limit is OK, proceed with normal handling."""
        mock_ctx = MockAuthContext()
        with (
            patch.object(_learning_limiter, "is_allowed", return_value=True),
            patch.object(handler, "get_auth_context", return_value=mock_ctx),
            patch.object(handler, "check_permission", return_value=True),
        ):
            result = await handler.handle("/api/v1/learning/cycles", {}, mock_http_handler)
        assert result is not None
        assert result.status_code != 429


# =============================================================================
# Test Cycle Summaries Endpoint
# =============================================================================


class TestGetCycleSummaries:
    """Tests for _get_cycle_summaries."""

    def test_no_nomic_dir_returns_503(self, handler_no_nomic):
        result = handler_no_nomic._get_cycle_summaries(20)
        assert result.status_code == 503
        data = parse_response(result)
        assert "not configured" in data.get("error", "")

    def test_no_replays_dir_returns_empty(self, handler, nomic_dir):
        """If replays dir doesn't exist, return empty list."""
        result = handler._get_cycle_summaries(20)
        assert result.status_code == 200
        data = parse_response(result)
        assert data["cycles"] == []
        assert data["count"] == 0

    def test_empty_replays_dir(self, handler, nomic_dir):
        """If replays dir exists but is empty, return empty list."""
        (nomic_dir / "replays").mkdir()
        result = handler._get_cycle_summaries(20)
        assert result.status_code == 200
        data = parse_response(result)
        assert data["cycles"] == []
        assert data["count"] == 0

    def test_single_cycle(self, handler, nomic_dir):
        """Should parse a single valid cycle directory."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        cycle_dir = replays / "nomic-cycle-1"
        cycle_dir.mkdir()

        meta = {
            "debate_id": "debate-001",
            "topic": "Improve test coverage",
            "agents": [{"name": "claude"}, {"name": "gpt4"}],
            "started_at": "2024-01-01T00:00:00Z",
            "ended_at": "2024-01-01T00:05:00Z",
            "duration_ms": 300000,
            "status": "completed",
            "final_verdict": "Increase unit tests by 30%",
            "event_count": 42,
        }
        (cycle_dir / "meta.json").write_text(json.dumps(meta))

        result = handler._get_cycle_summaries(20)
        assert result.status_code == 200
        data = parse_response(result)
        assert data["count"] == 1
        assert data["cycles"][0]["cycle"] == 1
        assert data["cycles"][0]["debate_id"] == "debate-001"
        assert data["cycles"][0]["topic"] == "Improve test coverage"
        assert data["cycles"][0]["agents"] == ["claude", "gpt4"]
        assert data["cycles"][0]["success"] is True

    def test_multiple_cycles_sorted_reverse(self, handler, nomic_dir):
        """Cycles should be sorted in reverse order (most recent first)."""
        replays = nomic_dir / "replays"
        replays.mkdir()

        for i in range(1, 4):
            cycle_dir = replays / f"nomic-cycle-{i}"
            cycle_dir.mkdir()
            meta = {
                "debate_id": f"debate-{i}",
                "topic": f"Topic {i}",
                "agents": [],
                "status": "completed",
                "final_verdict": f"Verdict {i}",
                "event_count": i * 10,
            }
            (cycle_dir / "meta.json").write_text(json.dumps(meta))

        result = handler._get_cycle_summaries(20)
        data = parse_response(result)
        assert data["count"] == 3
        # Reverse sorted: 3, 2, 1
        assert data["cycles"][0]["cycle"] == 3
        assert data["cycles"][1]["cycle"] == 2
        assert data["cycles"][2]["cycle"] == 1

    def test_limit_parameter(self, handler, nomic_dir):
        """Limit parameter should cap the number of cycles returned."""
        replays = nomic_dir / "replays"
        replays.mkdir()

        for i in range(1, 6):
            cycle_dir = replays / f"nomic-cycle-{i}"
            cycle_dir.mkdir()
            meta = {"debate_id": f"d-{i}", "topic": f"T{i}", "agents": [], "status": "completed"}
            (cycle_dir / "meta.json").write_text(json.dumps(meta))

        result = handler._get_cycle_summaries(2)
        data = parse_response(result)
        assert data["count"] == 2
        assert data["has_more"] is True

    def test_skips_non_cycle_directories(self, handler, nomic_dir):
        """Directories not matching nomic-cycle-* pattern should be skipped."""
        replays = nomic_dir / "replays"
        replays.mkdir()

        # Valid cycle
        cycle_dir = replays / "nomic-cycle-1"
        cycle_dir.mkdir()
        (cycle_dir / "meta.json").write_text(
            json.dumps(
                {
                    "debate_id": "d-1",
                    "topic": "T1",
                    "agents": [],
                    "status": "completed",
                }
            )
        )

        # Invalid directory names
        (replays / "random-dir").mkdir()
        (replays / "cycle-2").mkdir()

        result = handler._get_cycle_summaries(20)
        data = parse_response(result)
        assert data["count"] == 1

    def test_skips_cycle_without_meta(self, handler, nomic_dir):
        """Cycle dirs without meta.json should be skipped."""
        replays = nomic_dir / "replays"
        replays.mkdir()

        (replays / "nomic-cycle-1").mkdir()  # No meta.json

        result = handler._get_cycle_summaries(20)
        data = parse_response(result)
        assert data["count"] == 0

    def test_skips_invalid_json(self, handler, nomic_dir):
        """Cycles with invalid JSON in meta.json should be skipped."""
        replays = nomic_dir / "replays"
        replays.mkdir()

        cycle_dir = replays / "nomic-cycle-1"
        cycle_dir.mkdir()
        (cycle_dir / "meta.json").write_text("not valid json {{{")

        result = handler._get_cycle_summaries(20)
        data = parse_response(result)
        assert data["count"] == 0

    def test_failed_cycle_success_flag(self, handler, nomic_dir):
        """Cycle without final_verdict should have success=False."""
        replays = nomic_dir / "replays"
        replays.mkdir()

        cycle_dir = replays / "nomic-cycle-1"
        cycle_dir.mkdir()
        meta = {"debate_id": "d-1", "topic": "T1", "agents": [], "status": "failed"}
        (cycle_dir / "meta.json").write_text(json.dumps(meta))

        result = handler._get_cycle_summaries(20)
        data = parse_response(result)
        assert data["cycles"][0]["success"] is False

    def test_completed_without_verdict_success_false(self, handler, nomic_dir):
        """Completed status but no final_verdict should be success=False."""
        replays = nomic_dir / "replays"
        replays.mkdir()

        cycle_dir = replays / "nomic-cycle-1"
        cycle_dir.mkdir()
        meta = {
            "debate_id": "d-1",
            "topic": "T1",
            "agents": [],
            "status": "completed",
            "final_verdict": None,
        }
        (cycle_dir / "meta.json").write_text(json.dumps(meta))

        result = handler._get_cycle_summaries(20)
        data = parse_response(result)
        assert data["cycles"][0]["success"] is False


# =============================================================================
# Test Learned Patterns Endpoint
# =============================================================================


class TestGetLearnedPatterns:
    """Tests for _get_learned_patterns."""

    def test_no_nomic_dir_returns_503(self, handler_no_nomic):
        result = handler_no_nomic._get_learned_patterns()
        assert result.status_code == 503

    def test_empty_nomic_dir(self, handler, nomic_dir):
        """No risk_register or replays should return empty patterns."""
        result = handler._get_learned_patterns()
        assert result.status_code == 200
        data = parse_response(result)
        assert data["successful_patterns"] == []
        assert data["failed_patterns"] == []
        assert data["recurring_themes"] == []
        assert data["agent_specializations"] == {}

    def test_risk_register_parsing(self, handler, nomic_dir):
        """Should parse risk_register.jsonl for patterns."""
        entries = [
            {
                "cycle": 1,
                "phase": "implement",
                "confidence": 0.2,
                "task": "Bad task",
                "error": "Failed",
            },
            {"cycle": 2, "phase": "verify", "confidence": 0.8},
            {"cycle": 3, "phase": "debate", "confidence": 0.9},
        ]
        risk_file = nomic_dir / "risk_register.jsonl"
        risk_file.write_text("\n".join(json.dumps(e) for e in entries))

        result = handler._get_learned_patterns()
        data = parse_response(result)
        assert len(data["failed_patterns"]) == 1
        assert data["failed_patterns"][0]["cycle"] == 1
        assert len(data["successful_patterns"]) == 2

    def test_risk_register_limits_to_10(self, handler, nomic_dir):
        """Should limit patterns to last 10 entries."""
        entries = [{"cycle": i, "phase": "test", "confidence": 0.1} for i in range(20)]
        risk_file = nomic_dir / "risk_register.jsonl"
        risk_file.write_text("\n".join(json.dumps(e) for e in entries))

        result = handler._get_learned_patterns()
        data = parse_response(result)
        assert len(data["failed_patterns"]) <= 10

    def test_risk_register_skips_empty_lines(self, handler, nomic_dir):
        """Empty lines in risk_register.jsonl should be skipped."""
        content = json.dumps({"cycle": 1, "phase": "test", "confidence": 0.5}) + "\n\n\n"
        (nomic_dir / "risk_register.jsonl").write_text(content)

        result = handler._get_learned_patterns()
        data = parse_response(result)
        assert len(data["successful_patterns"]) == 1

    def test_risk_register_skips_invalid_json_lines(self, handler, nomic_dir):
        """Invalid JSON lines should be skipped gracefully."""
        content = json.dumps({"cycle": 1, "phase": "test", "confidence": 0.5}) + "\nnot json\n"
        (nomic_dir / "risk_register.jsonl").write_text(content)

        result = handler._get_learned_patterns()
        data = parse_response(result)
        assert len(data["successful_patterns"]) == 1

    def test_recurring_themes_from_replays(self, handler, nomic_dir):
        """Should detect keyword themes from replay topics."""
        replays = nomic_dir / "replays"
        replays.mkdir()

        topics = [
            "Fix security vulnerability",
            "Performance testing",
            "Security audit",
            "API feature",
        ]
        for i, topic in enumerate(topics):
            cycle_dir = replays / f"nomic-cycle-{i}"
            cycle_dir.mkdir()
            (cycle_dir / "meta.json").write_text(json.dumps({"topic": topic}))

        result = handler._get_learned_patterns()
        data = parse_response(result)
        themes = {t["theme"]: t["count"] for t in data["recurring_themes"]}
        assert themes.get("security", 0) == 2
        assert themes.get("testing", 0) == 1
        assert themes.get("api", 0) == 1
        assert themes.get("feature", 0) == 1

    def test_agent_specializations_from_winners(self, handler, nomic_dir):
        """Should track winning agents as specializations."""
        replays = nomic_dir / "replays"
        replays.mkdir()

        for i in range(3):
            cycle_dir = replays / f"nomic-cycle-{i}"
            cycle_dir.mkdir()
            winner = "claude" if i < 2 else "gpt4"
            (cycle_dir / "meta.json").write_text(json.dumps({"topic": "test", "winner": winner}))

        result = handler._get_learned_patterns()
        data = parse_response(result)
        assert data["agent_specializations"]["claude"] == 2
        assert data["agent_specializations"]["gpt4"] == 1

    def test_replays_with_invalid_meta_skipped(self, handler, nomic_dir):
        """Replay directories with invalid meta.json should be skipped."""
        replays = nomic_dir / "replays"
        replays.mkdir()

        cycle_dir = replays / "nomic-cycle-0"
        cycle_dir.mkdir()
        (cycle_dir / "meta.json").write_text("bad json")

        result = handler._get_learned_patterns()
        data = parse_response(result)
        assert data["recurring_themes"] == []

    def test_replays_skips_files_not_dirs(self, handler, nomic_dir):
        """Non-directory entries in replays should be skipped."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        (replays / "some_file.txt").write_text("not a dir")

        result = handler._get_learned_patterns()
        data = parse_response(result)
        assert data["recurring_themes"] == []


# =============================================================================
# Test Agent Evolution Endpoint
# =============================================================================


class TestGetAgentEvolution:
    """Tests for _get_agent_evolution."""

    def test_no_nomic_dir_returns_503(self, handler_no_nomic):
        result = handler_no_nomic._get_agent_evolution()
        assert result.status_code == 503

    def test_no_replays_dir(self, handler, nomic_dir):
        """No replays directory should return empty results."""
        result = handler._get_agent_evolution()
        assert result.status_code == 200
        data = parse_response(result)
        assert data["agents"] == {}
        assert data["total_cycles_analyzed"] == 0

    def test_empty_replays(self, handler, nomic_dir):
        """Empty replays directory should return empty results."""
        (nomic_dir / "replays").mkdir()
        result = handler._get_agent_evolution()
        data = parse_response(result)
        assert data["agents"] == {}
        assert data["total_cycles_analyzed"] == 0

    def test_single_cycle_single_agent(self, handler, nomic_dir):
        """Track a single agent across one cycle."""
        replays = nomic_dir / "replays"
        replays.mkdir()

        cycle_dir = replays / "nomic-cycle-1"
        cycle_dir.mkdir()
        meta = {
            "agents": [{"name": "claude"}],
            "vote_tally": {"claude": 5},
            "winner": "claude",
            "status": "completed",
        }
        (cycle_dir / "meta.json").write_text(json.dumps(meta))

        result = handler._get_agent_evolution()
        data = parse_response(result)
        assert "claude" in data["agents"]
        assert data["agents"]["claude"]["total_wins"] == 1
        assert data["agents"]["claude"]["total_cycles"] == 1
        assert data["total_cycles_analyzed"] == 1

    def test_multiple_cycles_trend_improving(self, handler, nomic_dir):
        """Agent winning most recent cycles should have 'improving' trend."""
        replays = nomic_dir / "replays"
        replays.mkdir()

        for i in range(1, 6):
            cycle_dir = replays / f"nomic-cycle-{i}"
            cycle_dir.mkdir()
            meta = {
                "agents": [{"name": "claude"}, {"name": "gpt4"}],
                "vote_tally": {"claude": 3, "gpt4": 2},
                "winner": "claude",
                "status": "completed",
            }
            (cycle_dir / "meta.json").write_text(json.dumps(meta))

        result = handler._get_agent_evolution()
        data = parse_response(result)
        assert data["agents"]["claude"]["trend"] == "improving"

    def test_trend_declining(self, handler, nomic_dir):
        """Agent never winning should have 'declining' trend."""
        replays = nomic_dir / "replays"
        replays.mkdir()

        for i in range(1, 6):
            cycle_dir = replays / f"nomic-cycle-{i}"
            cycle_dir.mkdir()
            meta = {
                "agents": [{"name": "claude"}, {"name": "gpt4"}],
                "vote_tally": {"claude": 3, "gpt4": 2},
                "winner": "gpt4",
                "status": "completed",
            }
            (cycle_dir / "meta.json").write_text(json.dumps(meta))

        result = handler._get_agent_evolution()
        data = parse_response(result)
        # claude never wins -> declining
        assert data["agents"]["claude"]["trend"] == "declining"

    def test_trend_stable_with_one_data_point(self, handler, nomic_dir):
        """Agent with only one data point should be 'stable'."""
        replays = nomic_dir / "replays"
        replays.mkdir()

        cycle_dir = replays / "nomic-cycle-1"
        cycle_dir.mkdir()
        meta = {"agents": [{"name": "claude"}], "vote_tally": {}, "status": "completed"}
        (cycle_dir / "meta.json").write_text(json.dumps(meta))

        result = handler._get_agent_evolution()
        data = parse_response(result)
        assert data["agents"]["claude"]["trend"] == "stable"

    def test_data_points_capped_at_20(self, handler, nomic_dir):
        """Data points should be capped at last 20."""
        replays = nomic_dir / "replays"
        replays.mkdir()

        for i in range(1, 30):
            cycle_dir = replays / f"nomic-cycle-{i}"
            cycle_dir.mkdir()
            meta = {
                "agents": [{"name": "claude"}],
                "vote_tally": {"claude": 1},
                "winner": "claude" if i % 2 == 0 else None,
                "status": "completed",
            }
            (cycle_dir / "meta.json").write_text(json.dumps(meta))

        result = handler._get_agent_evolution()
        data = parse_response(result)
        assert len(data["agents"]["claude"]["data_points"]) <= 20
        assert data["agents"]["claude"]["total_cycles"] == 29

    def test_skips_invalid_cycle_names(self, handler, nomic_dir):
        """Directories not matching nomic-cycle-N pattern should be skipped."""
        replays = nomic_dir / "replays"
        replays.mkdir()

        (replays / "nomic-cycle-abc").mkdir()
        (replays / "random-dir").mkdir()

        result = handler._get_agent_evolution()
        data = parse_response(result)
        assert data["agents"] == {}
        assert data["total_cycles_analyzed"] == 0

    def test_skips_invalid_meta_json(self, handler, nomic_dir):
        """Cycles with invalid meta.json should be skipped."""
        replays = nomic_dir / "replays"
        replays.mkdir()

        cycle_dir = replays / "nomic-cycle-1"
        cycle_dir.mkdir()
        (cycle_dir / "meta.json").write_text("{invalid")

        result = handler._get_agent_evolution()
        data = parse_response(result)
        assert data["total_cycles_analyzed"] == 0

    def test_agent_not_winner(self, handler, nomic_dir):
        """Agent that participates but doesn't win should have is_winner=False."""
        replays = nomic_dir / "replays"
        replays.mkdir()

        cycle_dir = replays / "nomic-cycle-1"
        cycle_dir.mkdir()
        meta = {
            "agents": [{"name": "claude"}, {"name": "gpt4"}],
            "vote_tally": {"claude": 1, "gpt4": 5},
            "winner": "gpt4",
            "status": "completed",
        }
        (cycle_dir / "meta.json").write_text(json.dumps(meta))

        result = handler._get_agent_evolution()
        data = parse_response(result)
        assert data["agents"]["claude"]["data_points"][0]["is_winner"] is False
        assert data["agents"]["gpt4"]["data_points"][0]["is_winner"] is True


# =============================================================================
# Test Aggregated Insights Endpoint
# =============================================================================


class TestGetAggregatedInsights:
    """Tests for _get_aggregated_insights."""

    def test_no_nomic_dir_returns_503(self, handler_no_nomic):
        result = handler_no_nomic._get_aggregated_insights(50)
        assert result.status_code == 503

    def test_no_insights_db(self, handler, nomic_dir):
        """If insights.db doesn't exist, return empty insights."""
        result = handler._get_aggregated_insights(50)
        assert result.status_code == 200
        data = parse_response(result)
        assert data["insights"] == []
        assert data["count"] == 0
        assert data["by_category"] == {}

    def test_insights_from_database(self, handler, nomic_dir):
        """Should load insights from insights.db."""
        db_path = nomic_dir / "insights.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE insights (
                insight_id TEXT,
                debate_id TEXT,
                category TEXT,
                content TEXT,
                confidence REAL,
                created_at TEXT
            )
        """)
        conn.execute(
            "INSERT INTO insights VALUES (?, ?, ?, ?, ?, ?)",
            ("i-1", "d-1", "performance", "Latency improved", 0.9, "2024-01-01"),
        )
        conn.execute(
            "INSERT INTO insights VALUES (?, ?, ?, ?, ?, ?)",
            ("i-2", "d-2", "security", "Vulnerability found", 0.8, "2024-01-02"),
        )
        conn.execute(
            "INSERT INTO insights VALUES (?, ?, ?, ?, ?, ?)",
            ("i-3", "d-3", "performance", "CPU usage reduced", 0.7, "2024-01-03"),
        )
        conn.commit()
        conn.close()

        with patch("aragora.server.handlers.memory.learning.get_db_connection") as mock_db:
            # Use a real sqlite connection as context manager
            real_conn = sqlite3.connect(str(db_path))
            mock_db.return_value.__enter__ = MagicMock(return_value=real_conn)
            mock_db.return_value.__exit__ = MagicMock(return_value=False)

            result = handler._get_aggregated_insights(50)

        data = parse_response(result)
        assert data["count"] == 3
        assert len(data["insights"]) == 3
        assert data["by_category"]["performance"] == 2
        assert data["by_category"]["security"] == 1
        real_conn.close()

    def test_insights_limit_parameter(self, handler, nomic_dir):
        """Limit parameter should cap the number of insights returned."""
        db_path = nomic_dir / "insights.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE insights (
                insight_id TEXT, debate_id TEXT, category TEXT,
                content TEXT, confidence REAL, created_at TEXT
            )
        """)
        for i in range(10):
            conn.execute(
                "INSERT INTO insights VALUES (?, ?, ?, ?, ?, ?)",
                (f"i-{i}", f"d-{i}", "general", f"Insight {i}", 0.5, f"2024-01-{i + 1:02d}"),
            )
        conn.commit()
        conn.close()

        with patch("aragora.server.handlers.memory.learning.get_db_connection") as mock_db:
            real_conn = sqlite3.connect(str(db_path))
            mock_db.return_value.__enter__ = MagicMock(return_value=real_conn)
            mock_db.return_value.__exit__ = MagicMock(return_value=False)

            result = handler._get_aggregated_insights(3)

        data = parse_response(result)
        assert data["count"] == 3
        real_conn.close()

    def test_insights_db_error_handled(self, handler, nomic_dir):
        """Database errors should be handled gracefully, returning empty results."""
        db_path = nomic_dir / "insights.db"
        db_path.write_text("")  # Create an empty file (invalid DB)

        with patch("aragora.server.handlers.memory.learning.get_db_connection") as mock_db:
            mock_db.return_value.__enter__ = MagicMock(side_effect=ValueError("DB error"))
            mock_db.return_value.__exit__ = MagicMock(return_value=False)

            result = handler._get_aggregated_insights(50)

        assert result.status_code == 200
        data = parse_response(result)
        assert data["insights"] == []
        assert data["count"] == 0

    def test_category_aggregation(self, handler, nomic_dir):
        """Should correctly aggregate insights by category."""
        db_path = nomic_dir / "insights.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE insights (
                insight_id TEXT, debate_id TEXT, category TEXT,
                content TEXT, confidence REAL, created_at TEXT
            )
        """)
        categories = ["security", "security", "performance", "testing", "testing", "testing"]
        for i, cat in enumerate(categories):
            conn.execute(
                "INSERT INTO insights VALUES (?, ?, ?, ?, ?, ?)",
                (f"i-{i}", f"d-{i}", cat, f"Insight {i}", 0.5, f"2024-01-{i + 1:02d}"),
            )
        conn.commit()
        conn.close()

        with patch("aragora.server.handlers.memory.learning.get_db_connection") as mock_db:
            real_conn = sqlite3.connect(str(db_path))
            mock_db.return_value.__enter__ = MagicMock(return_value=real_conn)
            mock_db.return_value.__exit__ = MagicMock(return_value=False)

            result = handler._get_aggregated_insights(50)

        data = parse_response(result)
        assert data["by_category"]["security"] == 2
        assert data["by_category"]["performance"] == 1
        assert data["by_category"]["testing"] == 3
        real_conn.close()


# =============================================================================
# Test Input Validation (Pagination / Limits)
# =============================================================================


class TestInputValidation:
    """Tests for input validation via handle() method."""

    @pytest.mark.asyncio
    async def test_cycles_default_limit(self, handler, mock_http_handler, nomic_dir):
        """Cycles endpoint should use default limit of 20 when not specified."""
        (nomic_dir / "replays").mkdir()
        mock_ctx = MockAuthContext()
        with (
            patch.object(handler, "get_auth_context", return_value=mock_ctx),
            patch.object(handler, "check_permission", return_value=True),
        ):
            result = await handler.handle("/api/v1/learning/cycles", {}, mock_http_handler)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_insights_default_limit(self, handler, mock_http_handler, nomic_dir):
        """Insights endpoint should use default limit of 50 when not specified."""
        mock_ctx = MockAuthContext()
        with (
            patch.object(handler, "get_auth_context", return_value=mock_ctx),
            patch.object(handler, "check_permission", return_value=True),
        ):
            result = await handler.handle("/api/v1/learning/insights", {}, mock_http_handler)
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_cycles_custom_limit(self, handler, mock_http_handler, nomic_dir):
        """Cycles endpoint should accept custom limit parameter."""
        (nomic_dir / "replays").mkdir()
        mock_ctx = MockAuthContext()
        with (
            patch.object(handler, "get_auth_context", return_value=mock_ctx),
            patch.object(handler, "check_permission", return_value=True),
        ):
            result = await handler.handle(
                "/api/v1/learning/cycles", {"limit": "5"}, mock_http_handler
            )
        assert result.status_code == 200


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""

    def test_handler_init_default_ctx(self):
        """Handler should work with default (None) context."""
        h = LearningHandler()
        assert h.ctx == {}

    def test_handler_init_custom_ctx(self):
        """Handler should accept custom context."""
        ctx = {"nomic_dir": "/tmp/test"}
        h = LearningHandler(ctx=ctx)
        assert h.ctx == ctx

    def test_get_nomic_dir_returns_path(self, handler, nomic_dir):
        """_get_nomic_dir should return a Path object."""
        result = handler._get_nomic_dir()
        assert isinstance(result, Path)
        assert str(result) == str(nomic_dir)

    def test_get_nomic_dir_returns_none_when_missing(self, handler_no_nomic):
        """_get_nomic_dir should return None when not configured."""
        result = handler_no_nomic._get_nomic_dir()
        assert result is None

    def test_cycle_with_missing_optional_fields(self, handler, nomic_dir):
        """Cycles with missing optional fields should still work."""
        replays = nomic_dir / "replays"
        replays.mkdir()

        cycle_dir = replays / "nomic-cycle-1"
        cycle_dir.mkdir()
        # Minimal meta with almost nothing
        (cycle_dir / "meta.json").write_text(json.dumps({}))

        result = handler._get_cycle_summaries(20)
        data = parse_response(result)
        assert data["count"] == 1
        cycle = data["cycles"][0]
        assert cycle["debate_id"] == ""
        assert cycle["topic"] == ""
        assert cycle["agents"] == []
        assert cycle["status"] == "unknown"
        assert cycle["success"] is False

    def test_risk_register_task_truncated(self, handler, nomic_dir):
        """Long task descriptions in risk_register should be truncated to 100 chars."""
        long_task = "x" * 200
        entries = [
            {"cycle": 1, "phase": "test", "confidence": 0.1, "task": long_task, "error": "fail"}
        ]
        (nomic_dir / "risk_register.jsonl").write_text(json.dumps(entries[0]))

        result = handler._get_learned_patterns()
        data = parse_response(result)
        if data["failed_patterns"]:
            assert len(data["failed_patterns"][0]["task"]) <= 100

    def test_risk_register_error_truncated(self, handler, nomic_dir):
        """Long error messages in risk_register should be truncated to 200 chars."""
        long_error = "e" * 300
        entries = [
            {"cycle": 1, "phase": "test", "confidence": 0.1, "task": "t", "error": long_error}
        ]
        (nomic_dir / "risk_register.jsonl").write_text(json.dumps(entries[0]))

        result = handler._get_learned_patterns()
        data = parse_response(result)
        if data["failed_patterns"]:
            assert len(data["failed_patterns"][0]["error"]) <= 200

    def test_evolution_no_vote_tally(self, handler, nomic_dir):
        """Evolution should handle missing vote_tally gracefully."""
        replays = nomic_dir / "replays"
        replays.mkdir()

        cycle_dir = replays / "nomic-cycle-1"
        cycle_dir.mkdir()
        meta = {"agents": [{"name": "claude"}], "status": "completed"}
        (cycle_dir / "meta.json").write_text(json.dumps(meta))

        result = handler._get_agent_evolution()
        data = parse_response(result)
        assert data["agents"]["claude"]["data_points"][0]["votes"] == 0

    def test_evolution_no_winner(self, handler, nomic_dir):
        """Evolution should handle missing winner gracefully."""
        replays = nomic_dir / "replays"
        replays.mkdir()

        cycle_dir = replays / "nomic-cycle-1"
        cycle_dir.mkdir()
        meta = {"agents": [{"name": "claude"}], "vote_tally": {}, "status": "completed"}
        (cycle_dir / "meta.json").write_text(json.dumps(meta))

        result = handler._get_agent_evolution()
        data = parse_response(result)
        assert data["agents"]["claude"]["data_points"][0]["is_winner"] is False

    def test_patterns_no_winner_in_meta(self, handler, nomic_dir):
        """Patterns should handle cycles without a winner."""
        replays = nomic_dir / "replays"
        replays.mkdir()

        cycle_dir = replays / "nomic-cycle-1"
        cycle_dir.mkdir()
        (cycle_dir / "meta.json").write_text(json.dumps({"topic": "test"}))

        result = handler._get_learned_patterns()
        data = parse_response(result)
        assert data["agent_specializations"] == {}


# =============================================================================
# Test Permission Constants
# =============================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_memory_read_permission(self):
        assert MEMORY_READ_PERMISSION == "memory:read"

    def test_memory_write_permission(self):
        assert MEMORY_WRITE_PERMISSION == "memory:write"

    def test_routes_defined(self):
        assert len(LearningHandler.ROUTES) == 4
        assert "/api/v1/learning/cycles" in LearningHandler.ROUTES
        assert "/api/v1/learning/patterns" in LearningHandler.ROUTES
        assert "/api/v1/learning/agent-evolution" in LearningHandler.ROUTES
        assert "/api/v1/learning/insights" in LearningHandler.ROUTES
