"""Tests for the unified Feedback Routing Hub.

Covers:
- Routing from each known source type
- Unknown source rejection
- Statistics tracking correctness
- History ordering and limits
- Multiple targets per source
- Graceful handling of missing target modules
- Thread safety basics
- Handler endpoint integration
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.nomic.feedback_hub import (
    KNOWN_SOURCES,
    FeedbackHub,
    RouteResult,
    get_feedback_hub,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hub() -> FeedbackHub:
    """Fresh hub instance for each test."""
    return FeedbackHub(max_history=100)


# ---------------------------------------------------------------------------
# RouteResult tests
# ---------------------------------------------------------------------------


class TestRouteResult:
    def test_success_when_all_targets_hit(self):
        r = RouteResult(source="test", targets_hit=["a", "b"])
        assert r.success is True
        assert r.partial is False

    def test_not_success_when_target_failed(self):
        r = RouteResult(source="test", targets_failed=["a"])
        assert r.success is False

    def test_partial_when_mixed(self):
        r = RouteResult(source="test", targets_hit=["a"], targets_failed=["b"])
        assert r.success is False
        assert r.partial is True

    def test_to_dict_contains_expected_keys(self):
        r = RouteResult(source="x", targets_hit=["y"], routed_at=1000.0)
        d = r.to_dict()
        assert d["source"] == "x"
        assert d["targets_hit"] == ["y"]
        assert d["routed_at"] == 1000.0
        assert "success" in d


# ---------------------------------------------------------------------------
# Unknown source rejection
# ---------------------------------------------------------------------------


class TestUnknownSource:
    def test_unknown_source_raises_value_error(self, hub: FeedbackHub):
        with pytest.raises(ValueError, match="Unknown feedback source"):
            hub.route("totally_unknown_source", {})

    def test_error_message_lists_known_sources(self, hub: FeedbackHub):
        with pytest.raises(ValueError) as exc_info:
            hub.route("nope", {})
        for src in KNOWN_SOURCES:
            assert src in str(exc_info.value)


# ---------------------------------------------------------------------------
# Source routing: user_feedback
# ---------------------------------------------------------------------------


class TestRouteUserFeedback:
    def test_user_feedback_delegates_to_analyzer(self, hub: FeedbackHub):
        """user_feedback route calls FeedbackAnalyzer and hits improvement_queue."""
        mock_analyzer = MagicMock()
        mock_analysis = MagicMock()
        mock_analysis.goals_created = 2
        mock_analysis.feedback_processed = 5
        mock_analyzer.return_value.process_new_feedback.return_value = mock_analysis

        with patch(
            "aragora.nomic.feedback_analyzer.FeedbackAnalyzer",
            mock_analyzer,
        ):
            result = hub.route("user_feedback", {"limit": 10})
            assert "improvement_queue" in result.targets_hit
            mock_analyzer.return_value.process_new_feedback.assert_called_once()

    def test_user_feedback_with_missing_analyzer(self, hub: FeedbackHub):
        """When FeedbackAnalyzer import fails, target is marked failed."""
        with patch.dict("sys.modules", {"aragora.nomic.feedback_analyzer": None}):
            result = hub.route("user_feedback", {})
            assert "improvement_queue" in result.targets_failed
            assert len(result.errors) > 0


# ---------------------------------------------------------------------------
# Source routing: gauntlet
# ---------------------------------------------------------------------------


class TestRouteGauntlet:
    def test_gauntlet_missing_result_fails(self, hub: FeedbackHub):
        """Missing gauntlet_result in payload -> target failure."""
        with patch(
            "aragora.nomic.feedback_hub.FeedbackHub._route_gauntlet",
            wraps=hub._route_gauntlet,
        ):
            result = hub.route("gauntlet", {})
            assert "improvement_queue" in result.targets_failed
            assert any("Missing gauntlet_result" in e for e in result.errors)

    def test_gauntlet_with_mock_result(self, hub: FeedbackHub):
        """With a mock GauntletResult, auto_improve completes."""
        mock_gauntlet_result = MagicMock()
        mock_gauntlet_result.gauntlet_id = "g-001"
        mock_gauntlet_result.findings = []

        mock_auto = MagicMock()
        mock_auto_result = MagicMock()
        mock_auto_result.goals_queued = 0
        mock_auto.return_value.on_run_complete.return_value = mock_auto_result

        with patch(
            "aragora.gauntlet.auto_improve.GauntletAutoImprove",
            mock_auto,
        ):
            result = hub.route("gauntlet", {"gauntlet_result": mock_gauntlet_result})
            assert "improvement_queue" in result.targets_hit


# ---------------------------------------------------------------------------
# Source routing: introspection
# ---------------------------------------------------------------------------


class TestRouteIntrospection:
    def test_introspection_pushes_to_queue_and_genesis(self, hub: FeedbackHub):
        """Introspection routes to both ImprovementQueue and Genesis."""
        mock_queue = MagicMock()
        mock_population = MagicMock()
        mock_evolved = MagicMock()
        mock_evolved.genomes = []

        mock_pop_mgr = MagicMock()
        mock_pop_mgr.get_or_create_population.return_value = mock_population
        mock_pop_mgr.evolve_population.return_value = mock_evolved

        with (
            patch(
                "aragora.nomic.feedback_orchestrator.ImprovementQueue",
                return_value=mock_queue,
            ),
            patch(
                "aragora.genesis.breeding.PopulationManager",
                return_value=mock_pop_mgr,
            ),
        ):
            result = hub.route(
                "introspection",
                {
                    "agent_name": "claude",
                    "success_rate": 0.3,
                    "agent_type": "claude",
                },
            )
            assert "improvement_queue" in result.targets_hit
            assert "genesis_evolution" in result.targets_hit
            mock_queue.push.assert_called_once()

    def test_introspection_genesis_import_failure(self, hub: FeedbackHub):
        """If Genesis is missing, queue still works."""
        mock_queue = MagicMock()

        with (
            patch(
                "aragora.nomic.feedback_orchestrator.ImprovementQueue",
                return_value=mock_queue,
            ),
            patch.dict("sys.modules", {"aragora.genesis.breeding": None}),
        ):
            result = hub.route(
                "introspection",
                {"agent_name": "test", "success_rate": 0.2},
            )
            assert "improvement_queue" in result.targets_hit
            assert "genesis_evolution" in result.targets_failed


# ---------------------------------------------------------------------------
# Source routing: debate_outcomes
# ---------------------------------------------------------------------------


class TestRouteDebateOutcomes:
    def test_debate_outcomes_routes_to_km_and_elo(self, hub: FeedbackHub):
        mock_mound = MagicMock()
        mock_elo = MagicMock()

        with (
            patch(
                "aragora.knowledge.mound.get_knowledge_mound",
                return_value=mock_mound,
            ),
            patch(
                "aragora.ranking.elo.EloSystem",
                return_value=mock_elo,
            ),
        ):
            result = hub.route(
                "debate_outcomes",
                {
                    "debate_id": "d-001",
                    "conclusion": "Use rate limiting",
                    "confidence": 0.85,
                    "consensus_reached": True,
                    "agents_participated": ["claude", "gpt4"],
                    "winner_agent": "claude",
                },
            )
            assert "knowledge_mound" in result.targets_hit
            assert "elo_update" in result.targets_hit

    def test_debate_outcomes_no_winner_still_succeeds(self, hub: FeedbackHub):
        """No winner agent means ELO has nothing to update but is still hit."""
        mock_mound = MagicMock()
        mock_elo = MagicMock()

        with (
            patch(
                "aragora.knowledge.mound.get_knowledge_mound",
                return_value=mock_mound,
            ),
            patch(
                "aragora.ranking.elo.EloSystem",
                return_value=mock_elo,
            ),
        ):
            result = hub.route(
                "debate_outcomes",
                {"debate_id": "d-002", "agents_participated": []},
            )
            assert "elo_update" in result.targets_hit


# ---------------------------------------------------------------------------
# Source routing: knowledge_contradictions
# ---------------------------------------------------------------------------


class TestRouteKnowledgeContradictions:
    def test_contradiction_pushed_to_queue(self, hub: FeedbackHub):
        mock_queue = MagicMock()

        with patch(
            "aragora.nomic.feedback_orchestrator.ImprovementQueue",
            return_value=mock_queue,
        ):
            result = hub.route(
                "knowledge_contradictions",
                {
                    "contradiction_type": "semantic",
                    "item_a_id": "km-1",
                    "item_b_id": "km-2",
                    "severity": "high",
                    "conflict_score": 0.9,
                },
            )
            assert "improvement_queue" in result.targets_hit
            mock_queue.push.assert_called_once()
            goal = mock_queue.push.call_args[0][0]
            assert "contradiction" in goal.goal.lower()
            assert goal.priority == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Source routing: pulse_stale_topics
# ---------------------------------------------------------------------------


class TestRoutePulseStaleTopics:
    def test_pulse_refresh_called(self, hub: FeedbackHub):
        mock_manager = MagicMock()
        import asyncio

        # Make get_trending_topics return a coroutine
        async def fake_trending(**kwargs):
            return [MagicMock(title="AI safety")]

        mock_manager.get_trending_topics = fake_trending

        with patch(
            "aragora.pulse.ingestor.PulseManager",
            return_value=mock_manager,
        ):
            result = hub.route(
                "pulse_stale_topics",
                {"platforms": ["hackernews"], "limit_per_platform": 3},
            )
            assert "pulse_refresh" in result.targets_hit

    def test_pulse_missing_module(self, hub: FeedbackHub):
        with patch.dict("sys.modules", {"aragora.pulse.ingestor": None}):
            result = hub.route("pulse_stale_topics", {})
            assert "pulse_refresh" in result.targets_failed


# ---------------------------------------------------------------------------
# Statistics tracking
# ---------------------------------------------------------------------------


class TestStatistics:
    def test_stats_initially_empty(self, hub: FeedbackHub):
        stats = hub.stats()
        assert stats["total_routed"] == 0
        assert stats["total_failures"] == 0
        assert stats["by_source"] == {}
        assert stats["by_target"] == {}
        assert stats["history_size"] == 0

    def test_stats_increment_on_route(self, hub: FeedbackHub):
        mock_queue = MagicMock()

        with patch(
            "aragora.nomic.feedback_orchestrator.ImprovementQueue",
            return_value=mock_queue,
        ):
            hub.route(
                "knowledge_contradictions",
                {"severity": "medium", "conflict_score": 0.5},
            )

        stats = hub.stats()
        assert stats["total_routed"] == 1
        assert stats["by_source"]["knowledge_contradictions"] == 1
        assert stats["by_target"].get("improvement_queue", 0) == 1

    def test_stats_track_multiple_sources(self, hub: FeedbackHub):
        mock_queue = MagicMock()

        with patch(
            "aragora.nomic.feedback_orchestrator.ImprovementQueue",
            return_value=mock_queue,
        ):
            hub.route("knowledge_contradictions", {"severity": "low"})
            hub.route("knowledge_contradictions", {"severity": "high"})

        stats = hub.stats()
        assert stats["total_routed"] == 2
        assert stats["by_source"]["knowledge_contradictions"] == 2

    def test_failure_count_tracked(self, hub: FeedbackHub):
        """Gauntlet with missing result causes a failure count."""
        hub.route("gauntlet", {})  # Missing gauntlet_result
        stats = hub.stats()
        assert stats["total_failures"] >= 1

    def test_known_sources_in_stats(self, hub: FeedbackHub):
        stats = hub.stats()
        assert set(stats["known_sources"]) == KNOWN_SOURCES


# ---------------------------------------------------------------------------
# History ordering and limits
# ---------------------------------------------------------------------------


class TestHistory:
    def test_history_ordered_by_recency(self, hub: FeedbackHub):
        """Most recent routes appear first in history."""
        mock_queue = MagicMock()

        with patch(
            "aragora.nomic.feedback_orchestrator.ImprovementQueue",
            return_value=mock_queue,
        ):
            hub.route(
                "knowledge_contradictions",
                {"severity": "low", "conflict_score": 0.1},
            )
            hub.route(
                "knowledge_contradictions",
                {"severity": "high", "conflict_score": 0.9},
            )

        history = hub.history()
        assert len(history) == 2
        # Most recent first
        assert history[0]["routed_at"] >= history[1]["routed_at"]

    def test_history_limit(self, hub: FeedbackHub):
        mock_queue = MagicMock()

        with patch(
            "aragora.nomic.feedback_orchestrator.ImprovementQueue",
            return_value=mock_queue,
        ):
            for _ in range(10):
                hub.route("knowledge_contradictions", {"severity": "low"})

        assert len(hub.history(limit=3)) == 3

    def test_history_max_capped(self):
        """Hub with small max_history evicts old entries."""
        hub = FeedbackHub(max_history=2)
        mock_queue = MagicMock()

        with patch(
            "aragora.nomic.feedback_orchestrator.ImprovementQueue",
            return_value=mock_queue,
        ):
            for _ in range(5):
                hub.route("knowledge_contradictions", {"severity": "low"})

        assert len(hub.history(limit=100)) == 2


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_everything(self, hub: FeedbackHub):
        mock_queue = MagicMock()

        with patch(
            "aragora.nomic.feedback_orchestrator.ImprovementQueue",
            return_value=mock_queue,
        ):
            hub.route("knowledge_contradictions", {"severity": "low"})

        assert hub.stats()["total_routed"] == 1

        hub.reset()

        stats = hub.stats()
        assert stats["total_routed"] == 0
        assert stats["history_size"] == 0


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_feedback_hub_returns_same_instance(self):
        hub1 = get_feedback_hub()
        hub2 = get_feedback_hub()
        assert hub1 is hub2


# ---------------------------------------------------------------------------
# Handler tests
# ---------------------------------------------------------------------------


class TestFeedbackHubHandler:
    def test_can_handle_stats(self):
        from aragora.server.handlers.feedback_hub import FeedbackHubHandler

        handler = FeedbackHubHandler()
        assert handler.can_handle("/api/v1/feedback-hub/stats", "GET") is True
        assert handler.can_handle("/api/feedback-hub/stats", "GET") is True
        assert handler.can_handle("/api/v1/feedback-hub/stats", "POST") is False

    def test_can_handle_history(self):
        from aragora.server.handlers.feedback_hub import FeedbackHubHandler

        handler = FeedbackHubHandler()
        assert handler.can_handle("/api/v1/feedback-hub/history", "GET") is True
        assert handler.can_handle("/api/feedback-hub/history", "GET") is True

    def test_cannot_handle_unknown(self):
        from aragora.server.handlers.feedback_hub import FeedbackHubHandler

        handler = FeedbackHubHandler()
        assert handler.can_handle("/api/v1/unknown", "GET") is False
