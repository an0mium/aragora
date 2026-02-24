"""Comprehensive tests for the FeedbackHub -- deep coverage of routing,
ELO updates, KM dispatch, error handling, thread safety, and edge cases.

This supplements the existing test_feedback_hub.py with additional
scenarios focused on:
- ELO record_match API correctness (winner=/loser= kwargs)
- KM feedback dispatch with store_dict path
- Error propagation and graceful degradation for each route
- Thread-safe counter updates under concurrent routing
- History eviction semantics
- Priority mapping for knowledge contradictions
- Introspection priority clamping
- Singleton reset / thread safety
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch, call

import pytest

from aragora.nomic.feedback_hub import (
    KNOWN_SOURCES,
    FeedbackHub,
    RouteResult,
    get_feedback_hub,
    _MAX_HISTORY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hub() -> FeedbackHub:
    """Fresh hub for each test."""
    return FeedbackHub(max_history=100)


# ---------------------------------------------------------------------------
# RouteResult edge cases
# ---------------------------------------------------------------------------


class TestRouteResultEdgeCases:
    def test_empty_result_is_not_success(self):
        """A result with no targets hit or failed is not success."""
        r = RouteResult(source="test")
        assert r.success is False
        assert r.partial is False

    def test_routed_at_populated_by_default(self):
        """routed_at should be populated automatically."""
        before = time.time()
        r = RouteResult(source="test")
        after = time.time()
        assert before <= r.routed_at <= after

    def test_to_dict_includes_errors(self):
        """to_dict should include error list."""
        r = RouteResult(source="test", errors=["err1", "err2"])
        d = r.to_dict()
        assert d["errors"] == ["err1", "err2"]

    def test_multiple_targets_all_hit(self):
        """Multiple targets all hit should be success."""
        r = RouteResult(source="test", targets_hit=["a", "b", "c"])
        assert r.success is True
        assert r.partial is False

    def test_multiple_targets_all_failed(self):
        """Multiple targets all failed should not be success or partial."""
        r = RouteResult(source="test", targets_failed=["a", "b"])
        assert r.success is False
        assert r.partial is False


# ---------------------------------------------------------------------------
# ELO update path (debate_outcomes)
# ---------------------------------------------------------------------------


class TestEloUpdatePath:
    def test_elo_record_match_called_with_winner_loser_kwargs(self, hub: FeedbackHub):
        """ELO record_match should be called with winner= and loser= keyword args."""
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
                    "debate_id": "d-100",
                    "agents_participated": ["claude", "gpt4", "gemini"],
                    "winner_agent": "claude",
                },
            )

        assert "elo_update" in result.targets_hit
        # record_match should be called for each non-winner agent
        assert mock_elo.record_match.call_count == 2
        # Verify winner= and loser= kwargs
        for c in mock_elo.record_match.call_args_list:
            assert c.kwargs["winner"] == "claude"
            assert c.kwargs["loser"] in ("gpt4", "gemini")

    def test_elo_single_participant_no_match_recorded(self, hub: FeedbackHub):
        """If only one agent participated and is the winner, no matches recorded."""
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
                    "debate_id": "d-101",
                    "agents_participated": ["claude"],
                    "winner_agent": "claude",
                },
            )

        assert "elo_update" in result.targets_hit
        mock_elo.record_match.assert_not_called()

    def test_elo_import_failure_marks_target_failed(self, hub: FeedbackHub):
        """If EloSystem can't be imported, elo_update target fails."""
        mock_mound = MagicMock()

        with (
            patch(
                "aragora.knowledge.mound.get_knowledge_mound",
                return_value=mock_mound,
            ),
            patch.dict("sys.modules", {"aragora.ranking.elo": None}),
        ):
            result = hub.route(
                "debate_outcomes",
                {
                    "debate_id": "d-102",
                    "agents_participated": ["claude", "gpt4"],
                    "winner_agent": "claude",
                },
            )

        assert "knowledge_mound" in result.targets_hit
        assert "elo_update" in result.targets_failed
        assert any("EloSystem not available" in e for e in result.errors)

    def test_elo_runtime_error_gracefully_handled(self, hub: FeedbackHub):
        """RuntimeError from record_match is caught and target fails."""
        mock_mound = MagicMock()
        mock_elo = MagicMock()
        mock_elo.record_match.side_effect = RuntimeError("ELO computation failed")

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
                    "debate_id": "d-103",
                    "agents_participated": ["a", "b"],
                    "winner_agent": "a",
                },
            )

        assert "knowledge_mound" in result.targets_hit
        assert "elo_update" in result.targets_failed
        assert any("RuntimeError" in e for e in result.errors)


# ---------------------------------------------------------------------------
# KM feedback dispatch (debate_outcomes)
# ---------------------------------------------------------------------------


class TestKmFeedbackDispatch:
    def test_km_store_dict_called_with_debate_data(self, hub: FeedbackHub):
        """KnowledgeMound.store_dict should receive debate outcome data."""
        mock_mound = MagicMock()
        mock_mound.store_dict = MagicMock()
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
            hub.route(
                "debate_outcomes",
                {
                    "debate_id": "d-200",
                    "conclusion": "Rate limiting is essential",
                    "consensus_reached": True,
                    "confidence": 0.9,
                },
            )

        mock_mound.store_dict.assert_called_once()
        stored = mock_mound.store_dict.call_args[0][0]
        assert stored["type"] == "debate_outcome"
        assert stored["debate_id"] == "d-200"
        assert stored["conclusion"] == "Rate limiting is essential"
        assert stored["consensus"] is True
        assert stored["confidence"] == 0.9

    def test_km_without_store_dict_still_hits_target(self, hub: FeedbackHub):
        """If mound has no store_dict, knowledge_mound target is still hit."""
        mock_mound = MagicMock(spec=[])  # no store_dict
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
                {"debate_id": "d-201"},
            )

        assert "knowledge_mound" in result.targets_hit

    def test_km_import_failure(self, hub: FeedbackHub):
        """If KnowledgeMound import fails, knowledge_mound target fails."""
        mock_elo = MagicMock()

        with (
            patch.dict("sys.modules", {"aragora.knowledge.mound": None}),
            patch(
                "aragora.ranking.elo.EloSystem",
                return_value=mock_elo,
            ),
        ):
            result = hub.route(
                "debate_outcomes",
                {"debate_id": "d-202", "agents_participated": ["a"], "winner_agent": "a"},
            )

        assert "knowledge_mound" in result.targets_failed
        assert "elo_update" in result.targets_hit

    def test_km_store_dict_raises_type_error(self, hub: FeedbackHub):
        """TypeError from store_dict is caught and target fails."""
        mock_mound = MagicMock()
        mock_mound.store_dict.side_effect = TypeError("Bad data")
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
                {"debate_id": "d-203"},
            )

        assert "knowledge_mound" in result.targets_failed
        assert any("TypeError" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Error handling for each route
# ---------------------------------------------------------------------------


class TestErrorHandlingPaths:
    def test_user_feedback_runtime_error(self, hub: FeedbackHub):
        """RuntimeError in FeedbackAnalyzer is caught."""
        mock_cls = MagicMock()
        mock_cls.return_value.process_new_feedback.side_effect = RuntimeError("boom")

        with patch(
            "aragora.nomic.feedback_analyzer.FeedbackAnalyzer",
            mock_cls,
        ):
            result = hub.route("user_feedback", {})

        assert "improvement_queue" in result.targets_failed
        assert any("RuntimeError" in e for e in result.errors)

    def test_gauntlet_auto_improve_attribute_error(self, hub: FeedbackHub):
        """AttributeError from GauntletAutoImprove is caught."""
        mock_cls = MagicMock()
        mock_cls.return_value.on_run_complete.side_effect = AttributeError("missing attr")

        with patch(
            "aragora.gauntlet.auto_improve.GauntletAutoImprove",
            mock_cls,
        ):
            result = hub.route(
                "gauntlet",
                {"gauntlet_result": MagicMock()},
            )

        assert "improvement_queue" in result.targets_failed

    def test_introspection_queue_runtime_error(self, hub: FeedbackHub):
        """RuntimeError from ImprovementQueue is caught, Genesis may still work."""
        mock_queue_cls = MagicMock()
        mock_queue_cls.return_value.push.side_effect = RuntimeError("queue broken")

        mock_pop_mgr = MagicMock()
        mock_evolved = MagicMock()
        mock_evolved.genomes = [1, 2]
        mock_pop_mgr.evolve_population.return_value = mock_evolved

        with (
            patch(
                "aragora.nomic.feedback_orchestrator.ImprovementQueue",
                mock_queue_cls,
            ),
            patch(
                "aragora.genesis.breeding.PopulationManager",
                return_value=mock_pop_mgr,
            ),
        ):
            result = hub.route(
                "introspection",
                {"agent_name": "test", "success_rate": 0.1, "agent_type": "claude"},
            )

        assert "improvement_queue" in result.targets_failed
        assert "genesis_evolution" in result.targets_hit

    def test_knowledge_contradictions_value_error(self, hub: FeedbackHub):
        """ValueError from ImprovementQueue push is caught."""
        mock_cls = MagicMock()
        mock_cls.return_value.push.side_effect = ValueError("bad goal")

        with patch(
            "aragora.nomic.feedback_orchestrator.ImprovementQueue",
            mock_cls,
        ):
            result = hub.route(
                "knowledge_contradictions",
                {"severity": "critical"},
            )

        assert "improvement_queue" in result.targets_failed

    def test_pulse_timeout_error(self, hub: FeedbackHub):
        """TimeoutError from PulseManager is caught."""
        mock_mgr = MagicMock()

        async def slow_trending(**kwargs):
            import asyncio

            raise TimeoutError("Too slow")

        mock_mgr.get_trending_topics = slow_trending

        with patch(
            "aragora.pulse.ingestor.PulseManager",
            return_value=mock_mgr,
        ):
            result = hub.route("pulse_stale_topics", {})

        assert "pulse_refresh" in result.targets_failed


# ---------------------------------------------------------------------------
# Introspection priority clamping
# ---------------------------------------------------------------------------


class TestIntrospectionPriorityClamping:
    def test_priority_clamped_to_zero_for_perfect_agent(self, hub: FeedbackHub):
        """success_rate=1.0 -> priority=0.0."""
        mock_queue = MagicMock()

        with (
            patch(
                "aragora.nomic.feedback_orchestrator.ImprovementQueue",
                return_value=mock_queue,
            ),
            patch.dict("sys.modules", {"aragora.genesis.breeding": None}),
        ):
            hub.route(
                "introspection",
                {"agent_name": "perfect", "success_rate": 1.0},
            )

        goal = mock_queue.push.call_args[0][0]
        assert goal.priority == pytest.approx(0.0)

    def test_priority_clamped_to_one_for_zero_success(self, hub: FeedbackHub):
        """success_rate=0.0 -> priority=1.0."""
        mock_queue = MagicMock()

        with (
            patch(
                "aragora.nomic.feedback_orchestrator.ImprovementQueue",
                return_value=mock_queue,
            ),
            patch.dict("sys.modules", {"aragora.genesis.breeding": None}),
        ):
            hub.route(
                "introspection",
                {"agent_name": "worst", "success_rate": 0.0},
            )

        goal = mock_queue.push.call_args[0][0]
        assert goal.priority == pytest.approx(1.0)

    def test_priority_clamped_when_negative_success_rate(self, hub: FeedbackHub):
        """Negative success_rate should still clamp priority to [0, 1]."""
        mock_queue = MagicMock()

        with (
            patch(
                "aragora.nomic.feedback_orchestrator.ImprovementQueue",
                return_value=mock_queue,
            ),
            patch.dict("sys.modules", {"aragora.genesis.breeding": None}),
        ):
            hub.route(
                "introspection",
                {"agent_name": "glitched", "success_rate": -0.5},
            )

        goal = mock_queue.push.call_args[0][0]
        assert 0.0 <= goal.priority <= 1.0


# ---------------------------------------------------------------------------
# Knowledge contradiction priority mapping
# ---------------------------------------------------------------------------


class TestContradictionPriorityMapping:
    @pytest.mark.parametrize(
        "severity,expected_priority",
        [
            ("critical", 0.95),
            ("high", 0.8),
            ("medium", 0.6),
            ("low", 0.3),
            ("unknown_severity", 0.5),
        ],
    )
    def test_severity_maps_to_correct_priority(
        self, hub: FeedbackHub, severity: str, expected_priority: float
    ):
        """Each severity level should map to its defined priority."""
        mock_queue = MagicMock()

        with patch(
            "aragora.nomic.feedback_orchestrator.ImprovementQueue",
            return_value=mock_queue,
        ):
            hub.route(
                "knowledge_contradictions",
                {
                    "severity": severity,
                    "conflict_score": 0.7,
                    "contradiction_type": "semantic",
                    "item_a_id": "a",
                    "item_b_id": "b",
                },
            )

        goal = mock_queue.push.call_args[0][0]
        assert goal.priority == pytest.approx(expected_priority)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_routing_updates_counters_correctly(self, hub: FeedbackHub):
        """Concurrent routes from multiple threads should produce correct totals."""
        mock_queue = MagicMock()
        num_threads = 10
        routes_per_thread = 20
        errors = []

        with patch(
            "aragora.nomic.feedback_orchestrator.ImprovementQueue",
            return_value=mock_queue,
        ):

            def worker():
                try:
                    for _ in range(routes_per_thread):
                        hub.route("knowledge_contradictions", {"severity": "low"})
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker) for _ in range(num_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

        assert len(errors) == 0, f"Thread errors: {errors}"
        stats = hub.stats()
        assert stats["total_routed"] == num_threads * routes_per_thread


# ---------------------------------------------------------------------------
# History semantics
# ---------------------------------------------------------------------------


class TestHistorySemantics:
    def test_history_returns_dicts_not_dataclasses(self, hub: FeedbackHub):
        """history() should return plain dicts."""
        mock_queue = MagicMock()

        with patch(
            "aragora.nomic.feedback_orchestrator.ImprovementQueue",
            return_value=mock_queue,
        ):
            hub.route("knowledge_contradictions", {"severity": "low"})

        entries = hub.history()
        assert len(entries) == 1
        assert isinstance(entries[0], dict)

    def test_history_limit_zero_returns_empty(self, hub: FeedbackHub):
        """history(limit=0) should return empty list."""
        mock_queue = MagicMock()

        with patch(
            "aragora.nomic.feedback_orchestrator.ImprovementQueue",
            return_value=mock_queue,
        ):
            hub.route("knowledge_contradictions", {"severity": "low"})

        assert hub.history(limit=0) == []

    def test_default_max_history_is_500(self):
        """Default max_history should be 500."""
        assert _MAX_HISTORY == 500
        hub = FeedbackHub()
        assert hub._max_history == 500


# ---------------------------------------------------------------------------
# Gauntlet edge cases
# ---------------------------------------------------------------------------


class TestGauntletEdgeCases:
    def test_gauntlet_custom_max_goals(self, hub: FeedbackHub):
        """max_goals_per_run in payload should be forwarded."""
        mock_cls = MagicMock()
        mock_result = MagicMock()
        mock_result.goals_queued = 3
        mock_cls.return_value.on_run_complete.return_value = mock_result

        with patch(
            "aragora.gauntlet.auto_improve.GauntletAutoImprove",
            mock_cls,
        ):
            hub.route(
                "gauntlet",
                {
                    "gauntlet_result": MagicMock(),
                    "max_goals_per_run": 10,
                },
            )

        # GauntletAutoImprove should have been created with max_goals_per_run=10
        mock_cls.assert_called_once_with(enabled=True, max_goals_per_run=10)

    def test_gauntlet_import_failure(self, hub: FeedbackHub):
        """Import failure for GauntletAutoImprove should be handled."""
        with patch.dict("sys.modules", {"aragora.gauntlet.auto_improve": None}):
            result = hub.route(
                "gauntlet",
                {"gauntlet_result": MagicMock()},
            )

        assert "improvement_queue" in result.targets_failed
        assert any("GauntletAutoImprove not available" in e for e in result.errors)
