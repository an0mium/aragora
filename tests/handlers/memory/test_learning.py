"""Tests for LearningHandler (aragora/server/handlers/memory/learning.py).

Covers all routes and behaviour of the LearningHandler class:
- can_handle() routing for all ROUTES
- GET /api/v1/learning/cycles
- GET /api/v1/learning/patterns
- GET /api/v1/learning/agent-evolution
- GET /api/v1/learning/insights
- Rate limiting (429 responses)
- Nomic dir not configured (503 responses)
- Missing replays / risk_register / insights.db
- Malformed JSON in meta.json / risk_register.jsonl
- Cycle limit clamping via query_params
- Agent evolution trend calculations (improving / declining / stable)
- Unknown path returns None
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
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


class MockHTTPHandler:
    """Mock HTTP request handler for LearningHandler tests."""

    def __init__(
        self,
        body: dict | None = None,
        method: str = "GET",
    ):
        self.command = method
        self.client_address = ("127.0.0.1", 12345)
        self.headers: dict[str, str] = {"User-Agent": "test-agent"}
        self.rfile = MagicMock()
        self._request_body = body

        if body:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers["Content-Length"] = str(len(body_bytes))
        else:
            self.rfile.read.return_value = b"{}"
            self.headers["Content-Length"] = "2"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def nomic_dir(tmp_path: Path) -> Path:
    """Create a temporary nomic directory with standard structure."""
    nomic = tmp_path / "nomic"
    nomic.mkdir()
    return nomic


@pytest.fixture()
def handler(nomic_dir: Path):
    """Create a LearningHandler with a nomic_dir context."""
    from aragora.server.handlers.memory.learning import LearningHandler

    return LearningHandler(ctx={"nomic_dir": str(nomic_dir)})


@pytest.fixture()
def handler_no_nomic():
    """Create a LearningHandler with no nomic_dir configured."""
    from aragora.server.handlers.memory.learning import LearningHandler

    return LearningHandler(ctx={})


@pytest.fixture()
def mock_http():
    """Create a default MockHTTPHandler."""
    return MockHTTPHandler()


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    """Reset rate limiters before each test so earlier tests don't pollute."""
    from aragora.server.handlers.memory.learning import _learning_limiter

    _learning_limiter._buckets.clear()
    yield
    _learning_limiter._buckets.clear()


# ---------------------------------------------------------------------------
# Helpers to create filesystem fixtures
# ---------------------------------------------------------------------------


def _create_cycle(
    replays_dir: Path,
    cycle_num: int,
    *,
    topic: str = "Improve testing",
    agents: list[dict] | None = None,  # Pass [] for empty; None = default agents
    winner: str | None = None,
    status: str = "completed",
    vote_tally: dict | None = None,
    final_verdict: str | None = "approved",
    extra: dict | None = None,
) -> Path:
    """Create a cycle directory with meta.json inside replays_dir."""
    cycle_dir = replays_dir / f"nomic-cycle-{cycle_num}"
    cycle_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "debate_id": f"debate-{cycle_num}",
        "topic": topic,
        "agents": [{"name": "claude"}, {"name": "grok"}] if agents is None else agents,
        "started_at": "2026-01-01T00:00:00",
        "ended_at": "2026-01-01T01:00:00",
        "duration_ms": 3600000,
        "status": status,
        "final_verdict": final_verdict,
        "event_count": 42,
        "winner": winner,
        "vote_tally": vote_tally or {},
    }
    if extra:
        meta.update(extra)

    (cycle_dir / "meta.json").write_text(json.dumps(meta))
    return cycle_dir


def _create_risk_register(nomic_dir: Path, entries: list[dict]) -> Path:
    """Create a risk_register.jsonl file."""
    risk_file = nomic_dir / "risk_register.jsonl"
    lines = [json.dumps(e) for e in entries]
    risk_file.write_text("\n".join(lines) + "\n")
    return risk_file


def _create_insights_db(nomic_dir: Path, rows: list[tuple]) -> Path:
    """Create an insights.db with populated insights table."""
    db_path = nomic_dir / "insights.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE insights (
            insight_id TEXT,
            debate_id TEXT,
            category TEXT,
            content TEXT,
            confidence REAL,
            created_at TEXT
        )
    """
    )
    for row in rows:
        conn.execute(
            "INSERT INTO insights VALUES (?, ?, ?, ?, ?, ?)",
            row,
        )
    conn.commit()
    conn.close()
    return db_path


# ===========================================================================
# can_handle
# ===========================================================================


class TestCanHandle:
    """Test LearningHandler.can_handle routing."""

    def test_cycles_route(self, handler):
        assert handler.can_handle("/api/v1/learning/cycles") is True

    def test_patterns_route(self, handler):
        assert handler.can_handle("/api/v1/learning/patterns") is True

    def test_agent_evolution_route(self, handler):
        assert handler.can_handle("/api/v1/learning/agent-evolution") is True

    def test_insights_route(self, handler):
        assert handler.can_handle("/api/v1/learning/insights") is True

    def test_unknown_route(self, handler):
        assert handler.can_handle("/api/v1/learning/unknown") is False

    def test_unversioned_route(self, handler):
        assert handler.can_handle("/api/learning/cycles") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False


# ===========================================================================
# GET /api/v1/learning/cycles
# ===========================================================================


class TestGetCycleSummaries:
    """Test the cycles endpoint."""

    @pytest.mark.asyncio
    async def test_no_replays_dir(self, handler, nomic_dir, mock_http):
        """When replays dir doesn't exist, return empty list."""
        result = await handler.handle("/api/v1/learning/cycles", {}, mock_http)
        body = _body(result)
        assert _status(result) == 200
        assert body["cycles"] == []
        assert body["count"] == 0

    @pytest.mark.asyncio
    async def test_empty_replays_dir(self, handler, nomic_dir, mock_http):
        """When replays dir exists but is empty, return empty list."""
        (nomic_dir / "replays").mkdir()
        result = await handler.handle("/api/v1/learning/cycles", {}, mock_http)
        body = _body(result)
        assert _status(result) == 200
        assert body["cycles"] == []
        assert body["count"] == 0

    @pytest.mark.asyncio
    async def test_single_cycle(self, handler, nomic_dir, mock_http):
        """Return a single cycle summary."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        _create_cycle(replays, 1, topic="Add rate limiting")

        result = await handler.handle("/api/v1/learning/cycles", {}, mock_http)
        body = _body(result)
        assert _status(result) == 200
        assert body["count"] == 1
        cycle = body["cycles"][0]
        assert cycle["cycle"] == 1
        assert cycle["topic"] == "Add rate limiting"
        assert cycle["debate_id"] == "debate-1"
        assert cycle["agents"] == ["claude", "grok"]
        assert cycle["status"] == "completed"
        assert cycle["success"] is True

    @pytest.mark.asyncio
    async def test_multiple_cycles_sorted_reverse(self, handler, nomic_dir, mock_http):
        """Cycles should be returned sorted by descending cycle number."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        _create_cycle(replays, 1, topic="First")
        _create_cycle(replays, 3, topic="Third")
        _create_cycle(replays, 2, topic="Second")

        result = await handler.handle("/api/v1/learning/cycles", {}, mock_http)
        body = _body(result)
        assert body["count"] == 3
        cycle_nums = [c["cycle"] for c in body["cycles"]]
        assert cycle_nums == [3, 2, 1]

    @pytest.mark.asyncio
    async def test_limit_param(self, handler, nomic_dir, mock_http):
        """Respect the limit query parameter."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        for i in range(5):
            _create_cycle(replays, i + 1)

        result = await handler.handle(
            "/api/v1/learning/cycles", {"limit": "2"}, mock_http
        )
        body = _body(result)
        assert body["count"] == 2
        assert body["has_more"] is True

    @pytest.mark.asyncio
    async def test_default_limit(self, handler, nomic_dir, mock_http):
        """Default limit is 20."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        # Create fewer than default limit to test has_more=False
        for i in range(3):
            _create_cycle(replays, i + 1)

        result = await handler.handle("/api/v1/learning/cycles", {}, mock_http)
        body = _body(result)
        assert body["count"] == 3
        assert body["has_more"] is False

    @pytest.mark.asyncio
    async def test_cycle_without_meta(self, handler, nomic_dir, mock_http):
        """Cycle dirs without meta.json should be silently skipped."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        # Valid cycle
        _create_cycle(replays, 1)
        # Cycle dir without meta.json
        (replays / "nomic-cycle-2").mkdir()

        result = await handler.handle("/api/v1/learning/cycles", {}, mock_http)
        body = _body(result)
        assert body["count"] == 1

    @pytest.mark.asyncio
    async def test_malformed_meta_json(self, handler, nomic_dir, mock_http):
        """Malformed meta.json should be skipped gracefully."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        cycle_dir = replays / "nomic-cycle-1"
        cycle_dir.mkdir()
        (cycle_dir / "meta.json").write_text("NOT VALID JSON {{{")

        _create_cycle(replays, 2, topic="Valid cycle")

        result = await handler.handle("/api/v1/learning/cycles", {}, mock_http)
        body = _body(result)
        assert body["count"] == 1
        assert body["cycles"][0]["cycle"] == 2

    @pytest.mark.asyncio
    async def test_non_cycle_dirs_skipped(self, handler, nomic_dir, mock_http):
        """Directories not matching nomic-cycle-* pattern are skipped."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        (replays / "some-other-dir").mkdir()
        (replays / "readme.txt").touch()
        _create_cycle(replays, 1)

        result = await handler.handle("/api/v1/learning/cycles", {}, mock_http)
        body = _body(result)
        assert body["count"] == 1

    @pytest.mark.asyncio
    async def test_success_flag_when_no_verdict(self, handler, nomic_dir, mock_http):
        """success should be False when final_verdict is None."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        _create_cycle(replays, 1, status="completed", final_verdict=None)

        result = await handler.handle("/api/v1/learning/cycles", {}, mock_http)
        body = _body(result)
        assert body["cycles"][0]["success"] is False

    @pytest.mark.asyncio
    async def test_success_flag_when_not_completed(self, handler, nomic_dir, mock_http):
        """success should be False when status is not 'completed'."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        _create_cycle(replays, 1, status="failed", final_verdict="rejected")

        result = await handler.handle("/api/v1/learning/cycles", {}, mock_http)
        body = _body(result)
        assert body["cycles"][0]["success"] is False

    @pytest.mark.asyncio
    async def test_nomic_dir_not_configured(self, handler_no_nomic, mock_http):
        """Return 503 when nomic dir is not configured."""
        result = await handler_no_nomic.handle(
            "/api/v1/learning/cycles", {}, mock_http
        )
        assert _status(result) == 503
        assert "not configured" in _body(result).get("error", "").lower()


# ===========================================================================
# GET /api/v1/learning/patterns
# ===========================================================================


class TestGetLearnedPatterns:
    """Test the patterns endpoint."""

    @pytest.mark.asyncio
    async def test_no_data(self, handler, nomic_dir, mock_http):
        """Return empty patterns when no risk register or replays exist."""
        result = await handler.handle("/api/v1/learning/patterns", {}, mock_http)
        body = _body(result)
        assert _status(result) == 200
        assert body["successful_patterns"] == []
        assert body["failed_patterns"] == []
        assert body["recurring_themes"] == []
        assert body["agent_specializations"] == {}

    @pytest.mark.asyncio
    async def test_risk_register_with_failed_entries(
        self, handler, nomic_dir, mock_http
    ):
        """Low-confidence entries should appear as failed patterns."""
        _create_risk_register(
            nomic_dir,
            [
                {
                    "cycle": 1,
                    "phase": "implement",
                    "task": "Fix memory leak",
                    "error": "Segfault in module X",
                    "confidence": 0.1,
                },
                {
                    "cycle": 2,
                    "phase": "verify",
                    "task": "Run tests",
                    "error": "Timeout",
                    "confidence": 0.2,
                },
            ],
        )

        result = await handler.handle("/api/v1/learning/patterns", {}, mock_http)
        body = _body(result)
        assert len(body["failed_patterns"]) == 2
        assert body["failed_patterns"][0]["cycle"] == 1
        assert body["failed_patterns"][0]["phase"] == "implement"

    @pytest.mark.asyncio
    async def test_risk_register_with_successful_entries(
        self, handler, nomic_dir, mock_http
    ):
        """High-confidence entries should appear as successful patterns."""
        _create_risk_register(
            nomic_dir,
            [
                {"cycle": 1, "phase": "debate", "confidence": 0.9},
                {"cycle": 2, "phase": "design", "confidence": 0.7},
            ],
        )

        result = await handler.handle("/api/v1/learning/patterns", {}, mock_http)
        body = _body(result)
        assert len(body["successful_patterns"]) == 2
        assert body["successful_patterns"][0]["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_risk_register_mixed(self, handler, nomic_dir, mock_http):
        """Mix of low and high confidence entries sorts correctly."""
        _create_risk_register(
            nomic_dir,
            [
                {"cycle": 1, "phase": "debate", "confidence": 0.1, "task": "x", "error": "y"},
                {"cycle": 2, "phase": "design", "confidence": 0.8},
            ],
        )

        result = await handler.handle("/api/v1/learning/patterns", {}, mock_http)
        body = _body(result)
        assert len(body["failed_patterns"]) == 1
        assert len(body["successful_patterns"]) == 1

    @pytest.mark.asyncio
    async def test_risk_register_limits_to_last_10(
        self, handler, nomic_dir, mock_http
    ):
        """Both pattern lists should be capped at last 10 entries."""
        entries = [
            {"cycle": i, "phase": "test", "confidence": 0.1, "task": f"task-{i}", "error": "err"}
            for i in range(15)
        ]
        _create_risk_register(nomic_dir, entries)

        result = await handler.handle("/api/v1/learning/patterns", {}, mock_http)
        body = _body(result)
        assert len(body["failed_patterns"]) == 10

    @pytest.mark.asyncio
    async def test_risk_register_malformed_line(self, handler, nomic_dir, mock_http):
        """Malformed lines in risk register are skipped."""
        risk_file = nomic_dir / "risk_register.jsonl"
        risk_file.write_text(
            '{"cycle": 1, "phase": "debate", "confidence": 0.9}\n'
            "NOT JSON\n"
            '{"cycle": 2, "phase": "design", "confidence": 0.8}\n'
        )

        result = await handler.handle("/api/v1/learning/patterns", {}, mock_http)
        body = _body(result)
        assert len(body["successful_patterns"]) == 2

    @pytest.mark.asyncio
    async def test_risk_register_empty_lines(self, handler, nomic_dir, mock_http):
        """Empty lines in risk register are skipped."""
        risk_file = nomic_dir / "risk_register.jsonl"
        risk_file.write_text(
            '{"cycle": 1, "phase": "debate", "confidence": 0.9}\n'
            "\n"
            "\n"
            '{"cycle": 2, "phase": "design", "confidence": 0.8}\n'
        )

        result = await handler.handle("/api/v1/learning/patterns", {}, mock_http)
        body = _body(result)
        assert len(body["successful_patterns"]) == 2

    @pytest.mark.asyncio
    async def test_recurring_themes(self, handler, nomic_dir, mock_http):
        """Topics containing keywords should be counted as themes."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        _create_cycle(replays, 1, topic="Improve security posture")
        _create_cycle(replays, 2, topic="Fix security vulnerability")
        _create_cycle(replays, 3, topic="Add new feature flag")

        result = await handler.handle("/api/v1/learning/patterns", {}, mock_http)
        body = _body(result)
        themes = {t["theme"]: t["count"] for t in body["recurring_themes"]}
        assert themes.get("security") == 2
        assert themes.get("feature") == 1
        assert themes.get("fix") == 1

    @pytest.mark.asyncio
    async def test_agent_specializations(self, handler, nomic_dir, mock_http):
        """Agent wins from meta.json should be tracked."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        _create_cycle(replays, 1, winner="claude")
        _create_cycle(replays, 2, winner="claude")
        _create_cycle(replays, 3, winner="grok")

        result = await handler.handle("/api/v1/learning/patterns", {}, mock_http)
        body = _body(result)
        specs = body["agent_specializations"]
        assert specs["claude"] == 2
        assert specs["grok"] == 1

    @pytest.mark.asyncio
    async def test_nomic_dir_not_configured(self, handler_no_nomic, mock_http):
        """Return 503 when nomic dir is not configured."""
        result = await handler_no_nomic.handle(
            "/api/v1/learning/patterns", {}, mock_http
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_themes_sorted_by_count(self, handler, nomic_dir, mock_http):
        """Recurring themes should be sorted by count descending."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        # 3x testing, 2x security, 1x feature
        _create_cycle(replays, 1, topic="Testing new module")
        _create_cycle(replays, 2, topic="Testing edge cases")
        _create_cycle(replays, 3, topic="Testing security layer")
        _create_cycle(replays, 4, topic="Security audit")
        _create_cycle(replays, 5, topic="New feature rollout")

        result = await handler.handle("/api/v1/learning/patterns", {}, mock_http)
        body = _body(result)
        theme_names = [t["theme"] for t in body["recurring_themes"]]
        # testing=3 should come first, then security=2
        assert theme_names[0] == "testing"
        assert theme_names[1] == "security"


# ===========================================================================
# GET /api/v1/learning/agent-evolution
# ===========================================================================


class TestGetAgentEvolution:
    """Test the agent-evolution endpoint."""

    @pytest.mark.asyncio
    async def test_no_replays(self, handler, nomic_dir, mock_http):
        """Return empty data when replays dir doesn't exist."""
        result = await handler.handle(
            "/api/v1/learning/agent-evolution", {}, mock_http
        )
        body = _body(result)
        assert _status(result) == 200
        assert body["agents"] == {}
        assert body["total_cycles_analyzed"] == 0

    @pytest.mark.asyncio
    async def test_single_agent_single_cycle(self, handler, nomic_dir, mock_http):
        """Single agent with single cycle should have trend 'stable'."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        _create_cycle(
            replays,
            1,
            agents=[{"name": "claude"}],
            winner="claude",
            vote_tally={"claude": 5},
        )

        result = await handler.handle(
            "/api/v1/learning/agent-evolution", {}, mock_http
        )
        body = _body(result)
        assert "claude" in body["agents"]
        claude = body["agents"]["claude"]
        assert claude["trend"] == "stable"
        assert claude["total_wins"] == 1
        assert claude["total_cycles"] == 1

    @pytest.mark.asyncio
    async def test_improving_trend(self, handler, nomic_dir, mock_http):
        """Agent winning > 50% of recent cycles should have 'improving' trend."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        # 4 cycles where claude wins 3 of the last 3
        _create_cycle(replays, 1, agents=[{"name": "claude"}], winner="grok")
        _create_cycle(replays, 2, agents=[{"name": "claude"}], winner="claude")
        _create_cycle(replays, 3, agents=[{"name": "claude"}], winner="claude")
        _create_cycle(replays, 4, agents=[{"name": "claude"}], winner="claude")

        result = await handler.handle(
            "/api/v1/learning/agent-evolution", {}, mock_http
        )
        body = _body(result)
        assert body["agents"]["claude"]["trend"] == "improving"

    @pytest.mark.asyncio
    async def test_declining_trend(self, handler, nomic_dir, mock_http):
        """Agent winning < 20% of recent cycles should have 'declining' trend."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        # 5 cycles where claude wins 0 of the last 3
        for i in range(1, 6):
            _create_cycle(
                replays,
                i,
                agents=[{"name": "claude"}],
                winner="grok",
                vote_tally={"claude": 0, "grok": 5},
            )

        result = await handler.handle(
            "/api/v1/learning/agent-evolution", {}, mock_http
        )
        body = _body(result)
        assert body["agents"]["claude"]["trend"] == "declining"

    @pytest.mark.asyncio
    async def test_stable_trend(self, handler, nomic_dir, mock_http):
        """Agent winning between 20-50% of recent cycles should be 'stable'."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        # 3 cycles: 1 win out of 3 = 33% -> stable
        _create_cycle(replays, 1, agents=[{"name": "claude"}], winner="claude")
        _create_cycle(replays, 2, agents=[{"name": "claude"}], winner="grok")
        _create_cycle(replays, 3, agents=[{"name": "claude"}], winner="grok")

        result = await handler.handle(
            "/api/v1/learning/agent-evolution", {}, mock_http
        )
        body = _body(result)
        assert body["agents"]["claude"]["trend"] == "stable"

    @pytest.mark.asyncio
    async def test_multiple_agents(self, handler, nomic_dir, mock_http):
        """Multiple agents are tracked independently."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        _create_cycle(
            replays,
            1,
            agents=[{"name": "claude"}, {"name": "grok"}],
            winner="claude",
            vote_tally={"claude": 5, "grok": 2},
        )

        result = await handler.handle(
            "/api/v1/learning/agent-evolution", {}, mock_http
        )
        body = _body(result)
        assert "claude" in body["agents"]
        assert "grok" in body["agents"]
        assert body["total_cycles_analyzed"] == 1

    @pytest.mark.asyncio
    async def test_data_points_limited_to_20(self, handler, nomic_dir, mock_http):
        """Only last 20 data points should be returned per agent."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        for i in range(1, 26):
            _create_cycle(replays, i, agents=[{"name": "claude"}])

        result = await handler.handle(
            "/api/v1/learning/agent-evolution", {}, mock_http
        )
        body = _body(result)
        assert len(body["agents"]["claude"]["data_points"]) == 20
        assert body["agents"]["claude"]["total_cycles"] == 25

    @pytest.mark.asyncio
    async def test_malformed_cycle_dir_name(self, handler, nomic_dir, mock_http):
        """Cycle dirs with non-integer suffix should be skipped."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        (replays / "nomic-cycle-abc").mkdir()
        _create_cycle(replays, 1)

        result = await handler.handle(
            "/api/v1/learning/agent-evolution", {}, mock_http
        )
        body = _body(result)
        assert body["total_cycles_analyzed"] == 1

    @pytest.mark.asyncio
    async def test_nomic_dir_not_configured(self, handler_no_nomic, mock_http):
        """Return 503 when nomic dir is not configured."""
        result = await handler_no_nomic.handle(
            "/api/v1/learning/agent-evolution", {}, mock_http
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_malformed_meta_json(self, handler, nomic_dir, mock_http):
        """Cycles with invalid meta.json should be skipped."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        bad_dir = replays / "nomic-cycle-1"
        bad_dir.mkdir()
        (bad_dir / "meta.json").write_text("{invalid json")
        _create_cycle(replays, 2, agents=[{"name": "claude"}])

        result = await handler.handle(
            "/api/v1/learning/agent-evolution", {}, mock_http
        )
        body = _body(result)
        assert body["total_cycles_analyzed"] == 1

    @pytest.mark.asyncio
    async def test_vote_tally_tracking(self, handler, nomic_dir, mock_http):
        """Vote tallies from meta.json should be reflected in data points."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        _create_cycle(
            replays,
            1,
            agents=[{"name": "claude"}],
            winner="claude",
            vote_tally={"claude": 7},
        )

        result = await handler.handle(
            "/api/v1/learning/agent-evolution", {}, mock_http
        )
        body = _body(result)
        dp = body["agents"]["claude"]["data_points"][0]
        assert dp["votes"] == 7
        assert dp["is_winner"] is True
        assert dp["participated"] is True


# ===========================================================================
# GET /api/v1/learning/insights
# ===========================================================================


class TestGetAggregatedInsights:
    """Test the insights endpoint."""

    @pytest.mark.asyncio
    async def test_no_insights_db(self, handler, nomic_dir, mock_http):
        """Return empty insights when db doesn't exist."""
        result = await handler.handle("/api/v1/learning/insights", {}, mock_http)
        body = _body(result)
        assert _status(result) == 200
        assert body["insights"] == []
        assert body["count"] == 0
        assert body["by_category"] == {}

    @pytest.mark.asyncio
    async def test_insights_from_db(self, handler, nomic_dir, mock_http):
        """Retrieve insights from the SQLite database."""
        _create_insights_db(
            nomic_dir,
            [
                ("ins-1", "debate-1", "security", "Use TLS", 0.9, "2026-01-01"),
                ("ins-2", "debate-2", "performance", "Cache more", 0.8, "2026-01-02"),
            ],
        )

        result = await handler.handle("/api/v1/learning/insights", {}, mock_http)
        body = _body(result)
        assert body["count"] == 2
        assert body["by_category"]["security"] == 1
        assert body["by_category"]["performance"] == 1

    @pytest.mark.asyncio
    async def test_insights_ordered_desc(self, handler, nomic_dir, mock_http):
        """Insights should be ordered by created_at descending."""
        _create_insights_db(
            nomic_dir,
            [
                ("ins-1", "d-1", "general", "Old", 0.5, "2026-01-01"),
                ("ins-2", "d-2", "general", "New", 0.5, "2026-01-10"),
            ],
        )

        result = await handler.handle("/api/v1/learning/insights", {}, mock_http)
        body = _body(result)
        assert body["insights"][0]["insight_id"] == "ins-2"
        assert body["insights"][1]["insight_id"] == "ins-1"

    @pytest.mark.asyncio
    async def test_insights_limit_param(self, handler, nomic_dir, mock_http):
        """Respect the limit query parameter."""
        rows = [
            (f"ins-{i}", f"d-{i}", "general", f"Content {i}", 0.5, f"2026-01-{i+1:02d}")
            for i in range(10)
        ]
        _create_insights_db(nomic_dir, rows)

        result = await handler.handle(
            "/api/v1/learning/insights", {"limit": "3"}, mock_http
        )
        body = _body(result)
        assert body["count"] == 3

    @pytest.mark.asyncio
    async def test_insights_default_limit(self, handler, nomic_dir, mock_http):
        """Default limit is 50."""
        rows = [
            (f"ins-{i}", f"d-{i}", "general", f"Content {i}", 0.5, f"2026-01-01")
            for i in range(5)
        ]
        _create_insights_db(nomic_dir, rows)

        result = await handler.handle("/api/v1/learning/insights", {}, mock_http)
        body = _body(result)
        # All 5 returned (less than default limit of 50)
        assert body["count"] == 5

    @pytest.mark.asyncio
    async def test_insights_category_aggregation(self, handler, nomic_dir, mock_http):
        """by_category should aggregate insight counts."""
        _create_insights_db(
            nomic_dir,
            [
                ("ins-1", "d-1", "security", "A", 0.9, "2026-01-01"),
                ("ins-2", "d-2", "security", "B", 0.8, "2026-01-02"),
                ("ins-3", "d-3", "performance", "C", 0.7, "2026-01-03"),
            ],
        )

        result = await handler.handle("/api/v1/learning/insights", {}, mock_http)
        body = _body(result)
        assert body["by_category"]["security"] == 2
        assert body["by_category"]["performance"] == 1

    @pytest.mark.asyncio
    async def test_insights_fields(self, handler, nomic_dir, mock_http):
        """Each insight should contain all expected fields."""
        _create_insights_db(
            nomic_dir,
            [("ins-1", "debate-1", "security", "Use TLS", 0.9, "2026-01-01")],
        )

        result = await handler.handle("/api/v1/learning/insights", {}, mock_http)
        body = _body(result)
        insight = body["insights"][0]
        assert insight["insight_id"] == "ins-1"
        assert insight["debate_id"] == "debate-1"
        assert insight["category"] == "security"
        assert insight["content"] == "Use TLS"
        assert insight["confidence"] == 0.9
        assert insight["created_at"] == "2026-01-01"

    @pytest.mark.asyncio
    async def test_nomic_dir_not_configured(self, handler_no_nomic, mock_http):
        """Return 503 when nomic dir is not configured."""
        result = await handler_no_nomic.handle(
            "/api/v1/learning/insights", {}, mock_http
        )
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_corrupted_insights_db(self, handler, nomic_dir, mock_http):
        """Corrupted insights.db should be handled gracefully."""
        db_path = nomic_dir / "insights.db"
        db_path.write_text("this is not a database")

        result = await handler.handle("/api/v1/learning/insights", {}, mock_http)
        body = _body(result)
        # Should degrade gracefully to empty or return 500 from handle_errors
        # The handler catches RuntimeError/OSError so it should return empty
        assert body.get("count", 0) == 0 or "error" in body


# ===========================================================================
# Rate Limiting
# ===========================================================================


class TestRateLimiting:
    """Test rate limiting on the learning handler."""

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self, handler, mock_http):
        """Return 429 when rate limit is exceeded."""
        from aragora.server.handlers.memory.learning import _learning_limiter

        # Exhaust the rate limit
        for _ in range(31):
            _learning_limiter.is_allowed("127.0.0.1")

        result = await handler.handle("/api/v1/learning/cycles", {}, mock_http)
        assert _status(result) == 429
        assert "rate limit" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_different_ips_independent(self, handler, nomic_dir, mock_http):
        """Different IPs should have independent rate limits."""
        from aragora.server.handlers.memory.learning import _learning_limiter

        # Exhaust limit for one IP
        for _ in range(31):
            _learning_limiter.is_allowed("10.0.0.1")

        # Different IP (the mock uses 127.0.0.1) should still work
        (nomic_dir / "replays").mkdir()
        result = await handler.handle("/api/v1/learning/cycles", {}, mock_http)
        assert _status(result) == 200


# ===========================================================================
# Unknown Path
# ===========================================================================


class TestUnknownPath:
    """Test handler returns None for unknown paths."""

    @pytest.mark.asyncio
    async def test_unknown_path_returns_none(self, handler, mock_http):
        """Unknown paths should return None to allow fallthrough."""
        result = await handler.handle(
            "/api/v1/learning/nonexistent", {}, mock_http
        )
        assert result is None


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_handler_with_none_ctx(self, mock_http):
        """Handler with None ctx should default to empty dict."""
        from aragora.server.handlers.memory.learning import LearningHandler

        h = LearningHandler(ctx=None)
        result = await h.handle("/api/v1/learning/cycles", {}, mock_http)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_risk_register_boundary_confidence(
        self, handler, nomic_dir, mock_http
    ):
        """Test confidence boundary at 0.3 exactly."""
        _create_risk_register(
            nomic_dir,
            [
                # Exactly 0.3 should be a successful pattern (not < 0.3)
                {"cycle": 1, "phase": "test", "confidence": 0.3},
                # Just below 0.3 should be failed
                {"cycle": 2, "phase": "test", "confidence": 0.29, "task": "x", "error": "y"},
            ],
        )

        result = await handler.handle("/api/v1/learning/patterns", {}, mock_http)
        body = _body(result)
        assert len(body["successful_patterns"]) == 1
        assert len(body["failed_patterns"]) == 1

    @pytest.mark.asyncio
    async def test_task_truncation_in_risk_register(
        self, handler, nomic_dir, mock_http
    ):
        """Long task strings should be truncated to 100 chars."""
        long_task = "x" * 200
        _create_risk_register(
            nomic_dir,
            [{"cycle": 1, "phase": "test", "confidence": 0.1, "task": long_task, "error": "e"}],
        )

        result = await handler.handle("/api/v1/learning/patterns", {}, mock_http)
        body = _body(result)
        assert len(body["failed_patterns"][0]["task"]) == 100

    @pytest.mark.asyncio
    async def test_error_truncation_in_risk_register(
        self, handler, nomic_dir, mock_http
    ):
        """Long error strings should be truncated to 200 chars."""
        long_error = "e" * 300
        _create_risk_register(
            nomic_dir,
            [{"cycle": 1, "phase": "test", "confidence": 0.1, "task": "t", "error": long_error}],
        )

        result = await handler.handle("/api/v1/learning/patterns", {}, mock_http)
        body = _body(result)
        assert len(body["failed_patterns"][0]["error"]) == 200

    @pytest.mark.asyncio
    async def test_no_winner_in_patterns(self, handler, nomic_dir, mock_http):
        """Cycles without a winner should not add to agent_specializations."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        _create_cycle(replays, 1, winner=None)

        result = await handler.handle("/api/v1/learning/patterns", {}, mock_http)
        body = _body(result)
        assert body["agent_specializations"] == {}

    @pytest.mark.asyncio
    async def test_agent_evolution_non_cycle_dirs_skipped(
        self, handler, nomic_dir, mock_http
    ):
        """Non-cycle directories should be skipped in agent evolution."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        (replays / "some-other-dir").mkdir()
        _create_cycle(replays, 1, agents=[{"name": "claude"}])

        result = await handler.handle(
            "/api/v1/learning/agent-evolution", {}, mock_http
        )
        body = _body(result)
        assert body["total_cycles_analyzed"] == 1

    @pytest.mark.asyncio
    async def test_cycle_with_missing_optional_fields(
        self, handler, nomic_dir, mock_http
    ):
        """Cycle meta.json with minimal fields should not crash."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        cycle_dir = replays / "nomic-cycle-1"
        cycle_dir.mkdir()
        # Minimal meta.json - only required for parsing
        (cycle_dir / "meta.json").write_text("{}")

        result = await handler.handle("/api/v1/learning/cycles", {}, mock_http)
        body = _body(result)
        assert body["count"] == 1
        cycle = body["cycles"][0]
        assert cycle["debate_id"] == ""
        assert cycle["topic"] == ""
        assert cycle["agents"] == []
        assert cycle["status"] == "unknown"

    @pytest.mark.asyncio
    async def test_evolution_with_empty_agents_list(
        self, handler, nomic_dir, mock_http
    ):
        """Cycle with no agents should be counted but produce no evolution data."""
        replays = nomic_dir / "replays"
        replays.mkdir()
        _create_cycle(replays, 1, agents=[])

        result = await handler.handle(
            "/api/v1/learning/agent-evolution", {}, mock_http
        )
        body = _body(result)
        assert body["total_cycles_analyzed"] == 1
        assert body["agents"] == {}
