"""Tests for System Intelligence Dashboard handler.

Covers all routes and behaviour of the SystemIntelligenceHandler class:
- GET /api/v1/system-intelligence/overview          - High-level system stats
- GET /api/v1/system-intelligence/agent-performance  - ELO, calibration, win rates
- GET /api/v1/system-intelligence/institutional-memory - Cross-debate injection stats
- GET /api/v1/system-intelligence/improvement-queue   - Queue contents + breakdown
"""

from __future__ import annotations

import json
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.system_intelligence import SystemIntelligenceHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_CDM_MODULE = "aragora.memory.cross_debate_rlm"


@contextmanager
def _fake_cross_debate_module(cdm_instance=None):
    """Inject a fake ``aragora.memory.cross_debate`` module into sys.modules.

    The handler does ``from aragora.memory.cross_debate import CrossDebateMemory``
    which requires the module to exist in ``sys.modules``.

    If *cdm_instance* is provided, ``CrossDebateMemory()`` returns it.
    If *cdm_instance* is ``None``, the module is removed so the import fails
    with ``ImportError``.
    """
    had_module = _CDM_MODULE in sys.modules
    old_module = sys.modules.get(_CDM_MODULE)

    if cdm_instance is None:
        # Make the import fail
        sys.modules.pop(_CDM_MODULE, None)
        try:
            yield
        finally:
            if had_module:
                sys.modules[_CDM_MODULE] = old_module  # type: ignore[assignment]
            else:
                sys.modules.pop(_CDM_MODULE, None)
        return

    fake_mod = types.ModuleType(_CDM_MODULE)
    fake_mod.CrossDebateMemory = lambda: cdm_instance  # type: ignore[attr-defined]
    sys.modules[_CDM_MODULE] = fake_mod
    try:
        yield
    finally:
        if had_module:
            sys.modules[_CDM_MODULE] = old_module  # type: ignore[assignment]
        else:
            sys.modules.pop(_CDM_MODULE, None)


def _parse_response(result) -> dict:
    """Extract data from json_response HandlerResult."""
    if hasattr(result, "body"):
        body = result.body
        if isinstance(body, (bytes, bytearray)):
            body = body.decode("utf-8")
        if isinstance(body, str):
            body = json.loads(body)
        if isinstance(body, dict):
            return body
    if isinstance(result, tuple):
        body = result[0] if len(result) > 0 else {}
        if isinstance(body, str):
            body = json.loads(body)
        return body
    if isinstance(result, dict):
        return result
    return {}


def _get_data(result) -> dict:
    """Extract the 'data' envelope from a response."""
    body = _parse_response(result)
    if isinstance(body, dict) and "data" in body:
        return body["data"]
    return body


# ---------------------------------------------------------------------------
# Mock data classes
# ---------------------------------------------------------------------------


@dataclass
class MockCycleEntry:
    """Mock Nomic cycle entry."""

    success: bool = True
    cycle_id: str = "cycle-001"


@dataclass
class MockLeaderboardEntry:
    """Mock ELO leaderboard entry as dict-like."""

    agent_name: str = "claude"
    rating: float = 1650.0
    wins: int = 40
    losses: int = 10


@dataclass
class MockHistoryEntry:
    """Mock ELO history entry."""

    timestamp: str = "2026-02-01T00:00:00Z"
    rating: float = 1600.0


@dataclass
class MockAgentState:
    """Mock SelectionFeedbackLoop agent state."""

    domain_wins: dict = field(default_factory=lambda: {"tech": 5, "finance": 3})


@dataclass
class MockImprovementItem:
    """Mock improvement queue item."""

    debate_id: str = "debate-001"
    task: str = "Improve test coverage for module X"
    confidence: float = 0.85
    category: str = "testing"
    created_at: str = "2026-02-01T12:00:00Z"


@dataclass
class MockConfidenceDecayEntry:
    """Mock KM confidence decay stats entry."""

    topic: str = "api-design"
    initial_confidence: float = 0.9
    current_confidence: float = 0.7


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a SystemIntelligenceHandler instance."""
    return SystemIntelligenceHandler(server_context={})


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler for passing to handle()."""
    h = MagicMock()
    h.rfile = MagicMock()
    h.rfile.read.return_value = b"{}"
    h.headers = {"Content-Length": "2"}
    return h


# ---------------------------------------------------------------------------
# ROUTES / can_handle
# ---------------------------------------------------------------------------


class TestRoutes:
    """Test ROUTES class attribute and can_handle method."""

    def test_routes_contains_all_endpoints(self):
        expected = [
            "/api/system-intelligence/overview",
            "/api/system-intelligence/agent-performance",
            "/api/system-intelligence/institutional-memory",
            "/api/system-intelligence/improvement-queue",
        ]
        for route in expected:
            assert route in SystemIntelligenceHandler.ROUTES, f"Missing route: {route}"

    def test_can_handle_with_version_prefix(self, handler):
        """Handler uses strip_version_prefix, so versioned paths should work."""
        versioned_paths = [
            "/api/v1/system-intelligence/overview",
            "/api/v1/system-intelligence/agent-performance",
            "/api/v1/system-intelligence/institutional-memory",
            "/api/v1/system-intelligence/improvement-queue",
        ]
        for path in versioned_paths:
            assert handler.can_handle(path), f"Should handle versioned: {path}"

    def test_can_handle_without_version_prefix(self, handler):
        """Handler should also work with unversioned paths."""
        for route in SystemIntelligenceHandler.ROUTES:
            assert handler.can_handle(route), f"Should handle: {route}"

    def test_can_handle_rejects_unknown_paths(self, handler):
        assert not handler.can_handle("/api/v1/system-intelligence/unknown")
        assert not handler.can_handle("/api/v1/other")
        assert not handler.can_handle("/api/v1/system-intelligence")


# ---------------------------------------------------------------------------
# Route dispatch
# ---------------------------------------------------------------------------


class TestRouteDispatch:
    """Test that handle() routes to the correct method."""

    @pytest.mark.asyncio
    async def test_dispatch_overview(self, handler, mock_http_handler):
        with _patch_all_overview_deps():
            result = await handler.handle(
                "/api/v1/system-intelligence/overview", {}, mock_http_handler
            )
        data = _get_data(result)
        assert "totalCycles" in data
        assert "successRate" in data

    @pytest.mark.asyncio
    async def test_dispatch_agent_performance(self, handler, mock_http_handler):
        with _patch_all_agent_perf_deps():
            result = await handler.handle(
                "/api/v1/system-intelligence/agent-performance", {}, mock_http_handler
            )
        data = _get_data(result)
        assert "agents" in data

    @pytest.mark.asyncio
    async def test_dispatch_institutional_memory(self, handler, mock_http_handler):
        with _patch_all_inst_memory_deps():
            result = await handler.handle(
                "/api/v1/system-intelligence/institutional-memory", {}, mock_http_handler
            )
        data = _get_data(result)
        assert "totalInjections" in data

    @pytest.mark.asyncio
    async def test_dispatch_improvement_queue(self, handler, mock_http_handler):
        with _patch_improvement_queue_deps():
            result = await handler.handle(
                "/api/v1/system-intelligence/improvement-queue", {}, mock_http_handler
            )
        data = _get_data(result)
        assert "items" in data

    @pytest.mark.asyncio
    async def test_dispatch_unknown_returns_none(self, handler, mock_http_handler):
        result = await handler.handle(
            "/api/v1/system-intelligence/nonexistent", {}, mock_http_handler
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_dispatch_unversioned_path(self, handler, mock_http_handler):
        with _patch_all_overview_deps():
            result = await handler.handle(
                "/api/system-intelligence/overview", {}, mock_http_handler
            )
        data = _get_data(result)
        assert "totalCycles" in data


# ---------------------------------------------------------------------------
# GET /api/v1/system-intelligence/overview
# ---------------------------------------------------------------------------


class TestOverview:
    """Test the overview endpoint."""

    @pytest.mark.asyncio
    async def test_overview_full_data(self, handler):
        cycle_store = MagicMock()
        cycle_store.get_recent_cycles.return_value = [
            {"success": True},
            {"success": True},
            {"success": False},
        ]

        elo = MagicMock()
        elo.get_leaderboard.return_value = [
            {"agent_name": "claude", "rating": 1650, "wins": 40},
            {"agent_name": "gpt4", "rating": 1600, "wins": 35},
        ]

        km = MagicMock()
        km.get_stats.return_value = {"total_items": 500}

        queue = MagicMock()
        queue.peek.return_value = [
            MockImprovementItem(debate_id="d1", task="Fix bug", category="bugs"),
        ]

        with (
            patch("aragora.nomic.cycle_store.get_cycle_store", return_value=cycle_store),
            patch("aragora.ranking.elo.EloSystem", return_value=elo),
            patch("aragora.knowledge.mound.core.KnowledgeMoundCore", return_value=km),
            patch("aragora.nomic.improvement_queue.get_improvement_queue", return_value=queue),
        ):
            result = await handler._get_overview()

        data = _get_data(result)
        assert data["totalCycles"] == 3
        assert data["successRate"] == round(2 / 3, 4)
        assert data["activeAgents"] == 2
        assert data["knowledgeItems"] == 500
        assert len(data["topAgents"]) == 2
        assert len(data["recentImprovements"]) == 1

    @pytest.mark.asyncio
    async def test_overview_top_agents_dict_entries(self, handler):
        elo = MagicMock()
        elo.get_leaderboard.return_value = [
            {"agent_name": "claude", "rating": 1700, "wins": 50},
        ]

        with (
            _patch_cycle_store_unavailable(),
            patch("aragora.ranking.elo.EloSystem", return_value=elo),
            _patch_km_unavailable(),
            _patch_improvement_queue_unavailable(),
        ):
            result = await handler._get_overview()

        data = _get_data(result)
        agent = data["topAgents"][0]
        assert agent["id"] == "claude"
        assert agent["elo"] == 1700
        assert agent["wins"] == 50

    @pytest.mark.asyncio
    async def test_overview_top_agents_object_entries(self, handler):
        entry = MockLeaderboardEntry(agent_name="gemini", rating=1550, wins=20)
        elo = MagicMock()
        elo.get_leaderboard.return_value = [entry]

        with (
            _patch_cycle_store_unavailable(),
            patch("aragora.ranking.elo.EloSystem", return_value=elo),
            _patch_km_unavailable(),
            _patch_improvement_queue_unavailable(),
        ):
            result = await handler._get_overview()

        data = _get_data(result)
        agent = data["topAgents"][0]
        assert agent["id"] == "gemini"
        assert agent["elo"] == 1550

    @pytest.mark.asyncio
    async def test_overview_no_cycles(self, handler):
        cycle_store = MagicMock()
        cycle_store.get_recent_cycles.return_value = []

        with (
            patch("aragora.nomic.cycle_store.get_cycle_store", return_value=cycle_store),
            _patch_elo_unavailable(),
            _patch_km_unavailable(),
            _patch_improvement_queue_unavailable(),
        ):
            result = await handler._get_overview()

        data = _get_data(result)
        assert data["totalCycles"] == 0
        assert data["successRate"] == 0.0

    @pytest.mark.asyncio
    async def test_overview_all_deps_unavailable(self, handler):
        with (
            _patch_cycle_store_unavailable(),
            _patch_elo_unavailable(),
            _patch_km_unavailable(),
            _patch_improvement_queue_unavailable(),
        ):
            result = await handler._get_overview()

        data = _get_data(result)
        assert data["totalCycles"] == 0
        assert data["successRate"] == 0.0
        assert data["activeAgents"] == 0
        assert data["knowledgeItems"] == 0
        assert data["topAgents"] == []
        assert data["recentImprovements"] == []

    @pytest.mark.asyncio
    async def test_overview_km_non_dict_stats(self, handler):
        km = MagicMock()
        km.get_stats.return_value = "not-a-dict"

        with (
            _patch_cycle_store_unavailable(),
            _patch_elo_unavailable(),
            patch("aragora.knowledge.mound.core.KnowledgeMoundCore", return_value=km),
            _patch_improvement_queue_unavailable(),
        ):
            result = await handler._get_overview()

        data = _get_data(result)
        assert data["knowledgeItems"] == 0

    @pytest.mark.asyncio
    async def test_overview_response_shape(self, handler):
        with _patch_all_overview_deps():
            result = await handler._get_overview()

        body = _parse_response(result)
        assert "data" in body
        data = body["data"]
        expected_keys = [
            "totalCycles",
            "successRate",
            "activeAgents",
            "knowledgeItems",
            "topAgents",
            "recentImprovements",
        ]
        for key in expected_keys:
            assert key in data, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# GET /api/v1/system-intelligence/agent-performance
# ---------------------------------------------------------------------------


class TestAgentPerformance:
    """Test the agent performance endpoint."""

    @pytest.mark.asyncio
    async def test_agent_performance_with_dict_entries(self, handler):
        elo = MagicMock()
        elo.get_leaderboard.return_value = [
            {"agent_name": "claude", "rating": 1650, "wins": 40, "losses": 10},
        ]
        elo.get_elo_history.return_value = [
            {"timestamp": "2026-01-01T00:00:00Z", "rating": 1600},
            {"timestamp": "2026-02-01T00:00:00Z", "rating": 1650},
        ]
        elo.get_agent_calibration_history.return_value = [{"score": 0.85}]

        feedback = MagicMock()
        feedback.get_agent_state.return_value = MockAgentState()

        with (
            patch("aragora.ranking.elo.EloSystem", return_value=elo),
            patch(
                "aragora.debate.selection_feedback.SelectionFeedbackLoop",
                return_value=feedback,
            ),
        ):
            result = await handler._get_agent_performance()

        data = _get_data(result)
        assert len(data["agents"]) == 1
        agent = data["agents"][0]
        assert agent["id"] == "claude"
        assert agent["name"] == "claude"
        assert agent["elo"] == 1650
        assert agent["winRate"] == 0.8  # 40/(40+10)
        assert agent["calibration"] == 0.85
        assert len(agent["eloHistory"]) == 2
        assert "tech" in agent["domains"]
        assert "finance" in agent["domains"]

    @pytest.mark.asyncio
    async def test_agent_performance_with_object_entries(self, handler):
        entry = MockLeaderboardEntry(agent_name="gpt4", rating=1600, wins=30, losses=20)
        elo = MagicMock()
        elo.get_leaderboard.return_value = [entry]
        elo.get_elo_history.return_value = []
        elo.get_agent_calibration_history.return_value = [{"score": 0.75}]

        with (
            patch("aragora.ranking.elo.EloSystem", return_value=elo),
            patch(
                "aragora.debate.selection_feedback.SelectionFeedbackLoop",
                side_effect=ImportError("not available"),
            ),
        ):
            result = await handler._get_agent_performance()

        data = _get_data(result)
        agent = data["agents"][0]
        assert agent["name"] == "gpt4"
        assert agent["elo"] == 1600
        assert agent["winRate"] == 0.6  # 30/50
        assert agent["calibration"] == 0.75
        assert agent["domains"] == []

    @pytest.mark.asyncio
    async def test_agent_performance_win_rate_zero_games(self, handler):
        elo = MagicMock()
        elo.get_leaderboard.return_value = [
            {"agent_name": "newbie", "rating": 1500, "wins": 0, "losses": 0},
        ]
        elo.get_elo_history.return_value = []
        elo.get_agent_calibration_history.return_value = [0.0]

        with (
            patch("aragora.ranking.elo.EloSystem", return_value=elo),
            patch(
                "aragora.debate.selection_feedback.SelectionFeedbackLoop",
                side_effect=ImportError("not available"),
            ),
        ):
            result = await handler._get_agent_performance()

        data = _get_data(result)
        assert data["agents"][0]["winRate"] == 0.0

    @pytest.mark.asyncio
    async def test_agent_performance_elo_unavailable(self, handler):
        with _patch_elo_unavailable():
            result = await handler._get_agent_performance()

        data = _get_data(result)
        assert data["agents"] == []

    @pytest.mark.asyncio
    async def test_agent_performance_history_with_object_entries(self, handler):
        elo = MagicMock()
        elo.get_leaderboard.return_value = [
            {"agent_name": "claude", "rating": 1650, "wins": 10, "losses": 5},
        ]
        hist_entry = MockHistoryEntry(timestamp="2026-01-15", rating=1620)
        elo.get_elo_history.return_value = [hist_entry]
        elo.get_agent_calibration_history.return_value = [{"score": 0.9}]

        with (
            patch("aragora.ranking.elo.EloSystem", return_value=elo),
            patch(
                "aragora.debate.selection_feedback.SelectionFeedbackLoop",
                side_effect=ImportError("not available"),
            ),
        ):
            result = await handler._get_agent_performance()

        data = _get_data(result)
        assert data["agents"][0]["eloHistory"][0]["date"] == "2026-01-15"
        assert data["agents"][0]["eloHistory"][0]["elo"] == 1620

    @pytest.mark.asyncio
    async def test_agent_performance_calibration_numeric(self, handler):
        elo = MagicMock()
        elo.get_leaderboard.return_value = [
            {"agent_name": "claude", "rating": 1650, "wins": 10, "losses": 5},
        ]
        elo.get_elo_history.return_value = []
        elo.get_agent_calibration_history.return_value = [42]  # int

        with (
            patch("aragora.ranking.elo.EloSystem", return_value=elo),
            patch(
                "aragora.debate.selection_feedback.SelectionFeedbackLoop",
                side_effect=ImportError("not available"),
            ),
        ):
            result = await handler._get_agent_performance()

        data = _get_data(result)
        assert data["agents"][0]["calibration"] == 42.0

    @pytest.mark.asyncio
    async def test_agent_performance_calibration_error(self, handler):
        elo = MagicMock()
        elo.get_leaderboard.return_value = [
            {"agent_name": "claude", "rating": 1650, "wins": 10, "losses": 5},
        ]
        elo.get_elo_history.return_value = []
        elo.get_agent_calibration_history.side_effect = AttributeError("no calibration")

        with (
            patch("aragora.ranking.elo.EloSystem", return_value=elo),
            patch(
                "aragora.debate.selection_feedback.SelectionFeedbackLoop",
                side_effect=ImportError("not available"),
            ),
        ):
            result = await handler._get_agent_performance()

        data = _get_data(result)
        assert data["agents"][0]["calibration"] == 0.0

    @pytest.mark.asyncio
    async def test_agent_performance_response_shape(self, handler):
        with _patch_elo_unavailable():
            result = await handler._get_agent_performance()

        body = _parse_response(result)
        assert "data" in body
        assert "agents" in body["data"]

    @pytest.mark.asyncio
    async def test_agent_performance_entry_shape(self, handler):
        elo = MagicMock()
        elo.get_leaderboard.return_value = [
            {"agent_name": "claude", "rating": 1650, "wins": 10, "losses": 5},
        ]
        elo.get_elo_history.return_value = []
        elo.get_agent_calibration_history.return_value = [{"score": 0.8}]

        with (
            patch("aragora.ranking.elo.EloSystem", return_value=elo),
            patch(
                "aragora.debate.selection_feedback.SelectionFeedbackLoop",
                side_effect=ImportError("not available"),
            ),
        ):
            result = await handler._get_agent_performance()

        data = _get_data(result)
        agent = data["agents"][0]
        expected_keys = [
            "id",
            "name",
            "elo",
            "eloHistory",
            "calibration",
            "winRate",
            "domains",
        ]
        for key in expected_keys:
            assert key in agent, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# GET /api/v1/system-intelligence/institutional-memory
# ---------------------------------------------------------------------------


class TestInstitutionalMemory:
    """Test the institutional memory endpoint."""

    @pytest.mark.asyncio
    async def test_institutional_memory_full_data(self, handler):
        adapter = AsyncMock()
        adapter.find_high_roi_goal_types = AsyncMock(
            return_value=[
                {"pattern": "testing", "cycle_count": 10, "avg_improvement_score": 0.85},
                {"pattern": "refactor", "cycle_count": 5, "avg_improvement_score": 0.70},
            ]
        )

        cdm = MagicMock()
        cdm.get_statistics.return_value = {
            "total_injections": 150,
            "retrieval_count": 400,
        }

        km = MagicMock()
        km.get_confidence_decay_stats.return_value = [
            {"topic": "api-design", "initial_confidence": 0.9, "current_confidence": 0.7},
        ]

        with (
            patch(
                "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
                return_value=adapter,
            ),
            _fake_cross_debate_module(cdm),
            patch("aragora.knowledge.mound.core.KnowledgeMoundCore", return_value=km),
        ):
            result = await handler._get_institutional_memory()

        data = _get_data(result)
        assert data["totalInjections"] == 150
        assert data["retrievalCount"] == 400
        assert len(data["topPatterns"]) == 2
        assert data["topPatterns"][0]["pattern"] == "testing"
        assert data["topPatterns"][0]["frequency"] == 10
        assert data["topPatterns"][0]["confidence"] == 0.85
        assert len(data["confidenceChanges"]) == 1
        assert data["confidenceChanges"][0]["topic"] == "api-design"

    @pytest.mark.asyncio
    async def test_institutional_memory_all_deps_unavailable(self, handler):
        with (
            patch(
                "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
                side_effect=ImportError("not available"),
            ),
            _fake_cross_debate_module(None),
            patch(
                "aragora.knowledge.mound.core.KnowledgeMoundCore",
                side_effect=ImportError("not available"),
            ),
        ):
            result = await handler._get_institutional_memory()

        data = _get_data(result)
        assert data["totalInjections"] == 0
        assert data["retrievalCount"] == 0
        assert data["topPatterns"] == []
        assert data["confidenceChanges"] == []

    @pytest.mark.asyncio
    async def test_institutional_memory_adapter_roi_error(self, handler):
        """ROI data fetch fails but other data sources still work."""
        adapter = AsyncMock()
        adapter.find_high_roi_goal_types = AsyncMock(side_effect=RuntimeError("failed"))

        cdm = MagicMock()
        cdm.get_statistics.return_value = {
            "total_injections": 50,
            "retrieval_count": 100,
        }

        with (
            patch(
                "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
                return_value=adapter,
            ),
            _fake_cross_debate_module(cdm),
            patch(
                "aragora.knowledge.mound.core.KnowledgeMoundCore",
                side_effect=ImportError("not available"),
            ),
        ):
            result = await handler._get_institutional_memory()

        data = _get_data(result)
        assert data["totalInjections"] == 50
        assert data["topPatterns"] == []

    @pytest.mark.asyncio
    async def test_institutional_memory_cdm_non_dict(self, handler):
        """Non-dict stats from CrossDebateMemory should default to 0."""
        cdm = MagicMock()
        cdm.get_statistics.return_value = "not-a-dict"

        with (
            patch(
                "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
                side_effect=ImportError("not available"),
            ),
            _fake_cross_debate_module(cdm),
            patch(
                "aragora.knowledge.mound.core.KnowledgeMoundCore",
                side_effect=ImportError("not available"),
            ),
        ):
            result = await handler._get_institutional_memory()

        data = _get_data(result)
        assert data["totalInjections"] == 0
        assert data["retrievalCount"] == 0

    @pytest.mark.asyncio
    async def test_institutional_memory_confidence_decay_non_list(self, handler):
        """Non-list decay stats should be ignored."""
        km = MagicMock()
        km.get_confidence_decay_stats.return_value = {"not": "a-list"}

        with (
            patch(
                "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
                side_effect=ImportError("not available"),
            ),
            _fake_cross_debate_module(None),
            patch("aragora.knowledge.mound.core.KnowledgeMoundCore", return_value=km),
        ):
            result = await handler._get_institutional_memory()

        data = _get_data(result)
        assert data["confidenceChanges"] == []

    @pytest.mark.asyncio
    async def test_institutional_memory_confidence_decay_truncated(self, handler):
        """Only first 10 decay entries should be included."""
        km = MagicMock()
        entries = [
            {"topic": f"topic-{i}", "initial_confidence": 0.9, "current_confidence": 0.5}
            for i in range(20)
        ]
        km.get_confidence_decay_stats.return_value = entries

        with (
            patch(
                "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
                side_effect=ImportError("not available"),
            ),
            _fake_cross_debate_module(None),
            patch("aragora.knowledge.mound.core.KnowledgeMoundCore", return_value=km),
        ):
            result = await handler._get_institutional_memory()

        data = _get_data(result)
        assert len(data["confidenceChanges"]) == 10

    @pytest.mark.asyncio
    async def test_institutional_memory_response_shape(self, handler):
        with _patch_all_inst_memory_deps():
            result = await handler._get_institutional_memory()

        body = _parse_response(result)
        assert "data" in body
        data = body["data"]
        expected_keys = [
            "totalInjections",
            "retrievalCount",
            "topPatterns",
            "confidenceChanges",
        ]
        for key in expected_keys:
            assert key in data, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# GET /api/v1/system-intelligence/improvement-queue
# ---------------------------------------------------------------------------


class TestImprovementQueue:
    """Test the improvement queue endpoint."""

    @pytest.mark.asyncio
    async def test_improvement_queue_with_items(self, handler):
        queue = MagicMock()
        queue.__len__ = MagicMock(return_value=3)
        queue.peek.return_value = [
            MockImprovementItem(debate_id="d1", task="Fix bug A", confidence=0.9, category="bugs"),
            MockImprovementItem(
                debate_id="d2", task="Add tests", confidence=0.7, category="testing"
            ),
            MockImprovementItem(
                debate_id="d3", task="Refactor X", confidence=0.5, category="refactor"
            ),
        ]

        with patch(
            "aragora.nomic.improvement_queue.get_improvement_queue",
            return_value=queue,
        ):
            result = await handler._get_improvement_queue()

        data = _get_data(result)
        assert data["totalSize"] == 3
        assert len(data["items"]) == 3
        assert data["sourceBreakdown"]["bugs"] == 1
        assert data["sourceBreakdown"]["testing"] == 1
        assert data["sourceBreakdown"]["refactor"] == 1

    @pytest.mark.asyncio
    async def test_improvement_queue_item_shape(self, handler):
        queue = MagicMock()
        queue.__len__ = MagicMock(return_value=1)
        queue.peek.return_value = [
            MockImprovementItem(
                debate_id="d1",
                task="Fix something important",
                confidence=0.85,
                category="fix",
                created_at="2026-02-01T12:00:00Z",
            ),
        ]

        with patch(
            "aragora.nomic.improvement_queue.get_improvement_queue",
            return_value=queue,
        ):
            result = await handler._get_improvement_queue()

        data = _get_data(result)
        item = data["items"][0]
        assert item["id"] == "d1"
        assert item["goal"] == "Fix something important"
        assert item["priority"] == 85  # int(0.85 * 100)
        assert item["source"] == "fix"
        assert item["status"] == "pending"
        assert item["createdAt"] == "2026-02-01T12:00:00Z"

    @pytest.mark.asyncio
    async def test_improvement_queue_task_truncated(self, handler):
        """Tasks longer than 200 chars should be truncated."""
        long_task = "x" * 500
        queue = MagicMock()
        queue.__len__ = MagicMock(return_value=1)
        queue.peek.return_value = [
            MockImprovementItem(debate_id="d1", task=long_task, confidence=0.5, category="fix"),
        ]

        with patch(
            "aragora.nomic.improvement_queue.get_improvement_queue",
            return_value=queue,
        ):
            result = await handler._get_improvement_queue()

        data = _get_data(result)
        assert len(data["items"][0]["goal"]) == 200

    @pytest.mark.asyncio
    async def test_improvement_queue_empty(self, handler):
        queue = MagicMock()
        queue.__len__ = MagicMock(return_value=0)
        queue.peek.return_value = []

        with patch(
            "aragora.nomic.improvement_queue.get_improvement_queue",
            return_value=queue,
        ):
            result = await handler._get_improvement_queue()

        data = _get_data(result)
        assert data["totalSize"] == 0
        assert data["items"] == []
        assert data["sourceBreakdown"] == {}

    @pytest.mark.asyncio
    async def test_improvement_queue_unavailable(self, handler):
        with _patch_improvement_queue_unavailable():
            result = await handler._get_improvement_queue()

        data = _get_data(result)
        assert data["totalSize"] == 0
        assert data["items"] == []
        assert data["sourceBreakdown"] == {}

    @pytest.mark.asyncio
    async def test_improvement_queue_source_breakdown_aggregation(self, handler):
        """Multiple items from same source should be aggregated."""
        queue = MagicMock()
        queue.__len__ = MagicMock(return_value=4)
        queue.peek.return_value = [
            MockImprovementItem(debate_id="d1", category="bugs"),
            MockImprovementItem(debate_id="d2", category="bugs"),
            MockImprovementItem(debate_id="d3", category="testing"),
            MockImprovementItem(debate_id="d4", category="bugs"),
        ]

        with patch(
            "aragora.nomic.improvement_queue.get_improvement_queue",
            return_value=queue,
        ):
            result = await handler._get_improvement_queue()

        data = _get_data(result)
        assert data["sourceBreakdown"]["bugs"] == 3
        assert data["sourceBreakdown"]["testing"] == 1

    @pytest.mark.asyncio
    async def test_improvement_queue_response_shape(self, handler):
        with _patch_improvement_queue_unavailable():
            result = await handler._get_improvement_queue()

        body = _parse_response(result)
        assert "data" in body
        data = body["data"]
        expected_keys = ["items", "totalSize", "sourceBreakdown"]
        for key in expected_keys:
            assert key in data, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Error handling / graceful degradation
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Test graceful degradation when dependencies fail."""

    @pytest.mark.asyncio
    async def test_overview_graceful_on_all_failures(self, handler):
        """Overview should return zero-state when everything fails."""
        with (
            _patch_cycle_store_unavailable(),
            _patch_elo_unavailable(),
            _patch_km_unavailable(),
            _patch_improvement_queue_unavailable(),
        ):
            result = await handler._get_overview()

        data = _get_data(result)
        assert data["totalCycles"] == 0
        assert data["topAgents"] == []

    @pytest.mark.asyncio
    async def test_agent_performance_graceful_on_elo_failure(self, handler):
        with _patch_elo_unavailable():
            result = await handler._get_agent_performance()

        data = _get_data(result)
        assert data["agents"] == []

    @pytest.mark.asyncio
    async def test_institutional_memory_graceful_on_all_failures(self, handler):
        with _patch_all_inst_memory_deps_unavailable():
            result = await handler._get_institutional_memory()

        data = _get_data(result)
        assert data["totalInjections"] == 0
        assert data["topPatterns"] == []
        assert data["confidenceChanges"] == []

    @pytest.mark.asyncio
    async def test_improvement_queue_graceful_on_runtime_error(self, handler):
        with patch(
            "aragora.nomic.improvement_queue.get_improvement_queue",
            side_effect=RuntimeError("queue broken"),
        ):
            result = await handler._get_improvement_queue()

        data = _get_data(result)
        assert data["totalSize"] == 0
        assert data["items"] == []


# ---------------------------------------------------------------------------
# Patch helpers (context managers)
# ---------------------------------------------------------------------------


def _patch_cycle_store_unavailable():
    return patch(
        "aragora.nomic.cycle_store.get_cycle_store",
        side_effect=ImportError("not available"),
    )


def _patch_elo_unavailable():
    return patch(
        "aragora.ranking.elo.EloSystem",
        side_effect=ImportError("not available"),
    )


def _patch_km_unavailable():
    return patch(
        "aragora.knowledge.mound.core.KnowledgeMoundCore",
        side_effect=ImportError("not available"),
    )


def _patch_improvement_queue_unavailable():
    return patch(
        "aragora.nomic.improvement_queue.get_improvement_queue",
        side_effect=ImportError("not available"),
    )


class _patch_all_overview_deps:
    """Context manager that patches all overview dependencies to return empty data."""

    def __init__(self):
        self._patches = []

    def __enter__(self):
        cycle_store = MagicMock()
        cycle_store.get_recent_cycles.return_value = []
        elo = MagicMock()
        elo.get_leaderboard.return_value = []
        km = MagicMock()
        km.get_stats.return_value = {"total_items": 0}
        queue = MagicMock()
        queue.peek.return_value = []

        self._patches = [
            patch("aragora.nomic.cycle_store.get_cycle_store", return_value=cycle_store),
            patch("aragora.ranking.elo.EloSystem", return_value=elo),
            patch("aragora.knowledge.mound.core.KnowledgeMoundCore", return_value=km),
            patch("aragora.nomic.improvement_queue.get_improvement_queue", return_value=queue),
        ]
        for p in self._patches:
            p.__enter__()
        return self

    def __exit__(self, *args):
        for p in reversed(self._patches):
            p.__exit__(*args)


class _patch_all_agent_perf_deps:
    """Context manager that patches all agent performance dependencies."""

    def __init__(self):
        self._patches = []

    def __enter__(self):
        elo = MagicMock()
        elo.get_leaderboard.return_value = []

        self._patches = [
            patch("aragora.ranking.elo.EloSystem", return_value=elo),
        ]
        for p in self._patches:
            p.__enter__()
        return self

    def __exit__(self, *args):
        for p in reversed(self._patches):
            p.__exit__(*args)


class _patch_all_inst_memory_deps:
    """Context manager that patches all institutional memory dependencies to return data."""

    def __init__(self):
        self._patches = []
        self._cdm_ctx = None

    def __enter__(self):
        adapter = AsyncMock()
        adapter.find_high_roi_goal_types = AsyncMock(return_value=[])

        cdm = MagicMock()
        cdm.get_statistics.return_value = {"total_injections": 0, "retrieval_count": 0}

        km = MagicMock()
        km.get_confidence_decay_stats.return_value = []

        self._patches = [
            patch(
                "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
                return_value=adapter,
            ),
            patch("aragora.knowledge.mound.core.KnowledgeMoundCore", return_value=km),
        ]
        self._cdm_ctx = _fake_cross_debate_module(cdm)
        self._cdm_ctx.__enter__()
        for p in self._patches:
            p.__enter__()
        return self

    def __exit__(self, *args):
        for p in reversed(self._patches):
            p.__exit__(*args)
        if self._cdm_ctx:
            self._cdm_ctx.__exit__(*args)


class _patch_all_inst_memory_deps_unavailable:
    """Context manager that makes all institutional memory deps unavailable."""

    def __init__(self):
        self._patches = []
        self._cdm_ctx = None

    def __enter__(self):
        self._patches = [
            patch(
                "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
                side_effect=ImportError("not available"),
            ),
            patch(
                "aragora.knowledge.mound.core.KnowledgeMoundCore",
                side_effect=ImportError("not available"),
            ),
        ]
        self._cdm_ctx = _fake_cross_debate_module(None)
        self._cdm_ctx.__enter__()
        for p in self._patches:
            p.__enter__()
        return self

    def __exit__(self, *args):
        for p in reversed(self._patches):
            p.__exit__(*args)
        if self._cdm_ctx:
            self._cdm_ctx.__exit__(*args)


def _patch_improvement_queue_deps():
    """Context manager for improvement queue with empty data."""

    class _ctx:
        def __init__(self):
            self._patches = []

        def __enter__(self):
            queue = MagicMock()
            queue.__len__ = MagicMock(return_value=0)
            queue.peek.return_value = []
            self._patches = [
                patch(
                    "aragora.nomic.improvement_queue.get_improvement_queue",
                    return_value=queue,
                ),
            ]
            for p in self._patches:
                p.__enter__()
            return self

        def __exit__(self, *args):
            for p in reversed(self._patches):
                p.__exit__(*args)

    return _ctx()
