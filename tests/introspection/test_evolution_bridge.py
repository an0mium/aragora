"""Tests for the IntrospectionEvolutionBridge.

Validates that introspection data is correctly analyzed to produce
targeted evolution recommendations, routed to ImprovementQueue, and
optionally fed into Genesis breeding configs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.introspection.active import (
    AgentPerformanceSummary,
    RoundMetrics,
)
from aragora.introspection.evolution_bridge import (
    EvolutionRecommendation,
    IntrospectionEvolutionBridge,
)
from aragora.introspection.types import IntrospectionSnapshot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_summary(
    agent_name: str = "claude",
    rounds: int = 5,
    proposals: int = 10,
    accepted: int = 2,
    critiques: int = 8,
    critiques_effective: int = 1,
    influence: float = 0.1,
    round_influences: list[float] | None = None,
) -> AgentPerformanceSummary:
    """Build an AgentPerformanceSummary with configurable metrics."""
    summary = AgentPerformanceSummary(
        agent_name=agent_name,
        rounds_completed=rounds,
        total_proposals=proposals,
        total_accepted=accepted,
        total_critiques=critiques,
        total_critiques_effective=critiques_effective,
        total_argument_influence=influence * rounds,
    )
    # Populate round history for influence-drop detection
    if round_influences is not None:
        for i, inf in enumerate(round_influences):
            summary.round_history.append(
                RoundMetrics(round_number=i + 1, argument_influence=inf)
            )
    return summary


def _make_snapshot(
    agent_name: str = "gemini",
    reputation: float = 0.2,
    debate_count: int = 10,
    proposals_made: int = 20,
    proposals_accepted: int = 3,
    calibration: float = 0.2,
) -> IntrospectionSnapshot:
    """Build an IntrospectionSnapshot with configurable metrics."""
    return IntrospectionSnapshot(
        agent_name=agent_name,
        reputation_score=reputation,
        debate_count=debate_count,
        proposals_made=proposals_made,
        proposals_accepted=proposals_accepted,
        calibration_score=calibration,
    )


# ---------------------------------------------------------------------------
# Tests: Underperforming agents generate recommendations
# ---------------------------------------------------------------------------


class TestUnderperformingAgents:
    """Agents below thresholds produce actionable recommendations."""

    def test_low_proposal_acceptance(self) -> None:
        """Agent with low proposal acceptance triggers a recommendation."""
        bridge = IntrospectionEvolutionBridge()
        summary = _make_summary(proposals=10, accepted=1)  # 10% acceptance
        recs = bridge.analyze_summaries({"claude": summary})

        proposal_recs = [r for r in recs if r.metric_name == "proposal_acceptance_rate"]
        assert len(proposal_recs) >= 1
        rec = proposal_recs[0]
        assert "proposal acceptance" in rec.recommendation.lower()
        assert rec.severity in ("high", "medium")
        assert rec.metric_value == pytest.approx(0.1)
        assert rec.source == "introspection"

    def test_low_critique_effectiveness(self) -> None:
        """Agent with low critique effectiveness triggers a recommendation."""
        bridge = IntrospectionEvolutionBridge()
        summary = _make_summary(critiques=10, critiques_effective=1)  # 10%
        recs = bridge.analyze_summaries({"grok": summary})

        crit_recs = [r for r in recs if r.metric_name == "critique_effectiveness"]
        assert len(crit_recs) >= 1
        assert "critique" in crit_recs[0].recommendation.lower()

    def test_low_average_influence(self) -> None:
        """Agent with low average influence triggers a recommendation."""
        bridge = IntrospectionEvolutionBridge()
        summary = _make_summary(influence=0.05)  # well below 0.15 threshold
        recs = bridge.analyze_summaries({"codex": summary})

        inf_recs = [r for r in recs if r.metric_name == "average_influence"]
        assert len(inf_recs) >= 1
        assert "influence" in inf_recs[0].recommendation.lower()

    def test_influence_drop_detected(self) -> None:
        """Agent whose influence drops significantly triggers a recommendation."""
        bridge = IntrospectionEvolutionBridge()
        # First 4 rounds high influence, last 4 rounds very low
        influences = [0.8, 0.7, 0.75, 0.8, 0.1, 0.15, 0.1, 0.05]
        summary = _make_summary(
            rounds=8,
            influence=0.4,
            round_influences=influences,
            proposals=10,
            accepted=8,  # high acceptance to avoid proposal rec noise
            critiques=0,
        )
        recs = bridge.analyze_summaries({"claude": summary})

        drop_recs = [r for r in recs if r.metric_name == "influence_drop_pct"]
        assert len(drop_recs) >= 1
        assert "dropped" in drop_recs[0].recommendation.lower()
        assert drop_recs[0].severity == "high"

    def test_low_reputation_snapshot(self) -> None:
        """Historical snapshot with low reputation triggers a recommendation."""
        bridge = IntrospectionEvolutionBridge()
        snapshot = _make_snapshot(reputation=0.15, debate_count=10)
        recs = bridge.analyze_snapshots({"gemini": snapshot})

        rep_recs = [r for r in recs if r.metric_name == "reputation_score"]
        assert len(rep_recs) >= 1
        assert "reputation" in rep_recs[0].recommendation.lower()
        assert rep_recs[0].severity == "high"

    def test_low_calibration_snapshot(self) -> None:
        """Snapshot with low calibration score triggers a recommendation."""
        bridge = IntrospectionEvolutionBridge()
        snapshot = _make_snapshot(calibration=0.15)
        recs = bridge.analyze_snapshots({"gemini": snapshot})

        cal_recs = [r for r in recs if r.metric_name == "calibration_score"]
        assert len(cal_recs) >= 1
        assert "calibration" in cal_recs[0].recommendation.lower()


# ---------------------------------------------------------------------------
# Tests: High-performing agents
# ---------------------------------------------------------------------------


class TestHighPerformingAgents:
    """Agents above all thresholds should not trigger recommendations."""

    def test_high_performer_no_recommendations(self) -> None:
        """An agent performing well produces zero recommendations."""
        bridge = IntrospectionEvolutionBridge()
        summary = _make_summary(
            proposals=10,
            accepted=8,  # 80%
            critiques=10,
            critiques_effective=7,  # 70%
            influence=0.7,
        )
        recs = bridge.analyze_summaries({"claude": summary})
        assert recs == []

    def test_high_performer_snapshot_no_recommendations(self) -> None:
        """A high-reputation snapshot produces zero recommendations."""
        bridge = IntrospectionEvolutionBridge()
        snapshot = _make_snapshot(
            reputation=0.9,
            proposals_made=20,
            proposals_accepted=15,
            calibration=0.8,
        )
        recs = bridge.analyze_snapshots({"gemini": snapshot})
        assert recs == []


# ---------------------------------------------------------------------------
# Tests: Formatting and serialization
# ---------------------------------------------------------------------------


class TestRecommendationFormatting:
    """Recommendations are properly formatted and serializable."""

    def test_recommendation_to_dict(self) -> None:
        """EvolutionRecommendation.to_dict() round-trips correctly."""
        rec = EvolutionRecommendation(
            agent_name="claude",
            recommendation="Test recommendation",
            severity="high",
            metric_name="proposal_acceptance_rate",
            metric_value=0.1,
            threshold=0.25,
            context={"extra": "data"},
        )
        d = rec.to_dict()
        assert d["agent_name"] == "claude"
        assert d["severity"] == "high"
        assert d["source"] == "introspection"
        assert d["context"]["extra"] == "data"
        assert isinstance(d["timestamp"], float)

    def test_recommendation_contains_agent_name(self) -> None:
        """Recommendation text includes the agent name."""
        bridge = IntrospectionEvolutionBridge()
        summary = _make_summary(agent_name="grok-v2", proposals=10, accepted=0)
        recs = bridge.analyze_summaries({"grok-v2": summary})
        assert any("grok-v2" in r.recommendation for r in recs)


# ---------------------------------------------------------------------------
# Tests: Empty / edge-case data
# ---------------------------------------------------------------------------


class TestEmptyData:
    """Graceful handling of missing or insufficient data."""

    def test_empty_summaries(self) -> None:
        """Empty summaries dict produces no recommendations."""
        bridge = IntrospectionEvolutionBridge()
        assert bridge.analyze_summaries({}) == []

    def test_empty_snapshots(self) -> None:
        """Empty snapshots dict produces no recommendations."""
        bridge = IntrospectionEvolutionBridge()
        assert bridge.analyze_snapshots({}) == []

    def test_analyze_with_nothing(self) -> None:
        """analyze() with no arguments returns empty list."""
        bridge = IntrospectionEvolutionBridge()
        assert bridge.analyze() == []

    def test_insufficient_rounds_skipped(self) -> None:
        """Agent with too few rounds is not evaluated."""
        bridge = IntrospectionEvolutionBridge()
        summary = _make_summary(rounds=1, proposals=10, accepted=0)
        recs = bridge.analyze_summaries({"claude": summary})
        assert recs == []

    def test_insufficient_debate_count_skipped(self) -> None:
        """Snapshot with too few debates is not evaluated."""
        bridge = IntrospectionEvolutionBridge()
        snapshot = _make_snapshot(debate_count=1, reputation=0.1)
        recs = bridge.analyze_snapshots({"gemini": snapshot})
        assert recs == []


# ---------------------------------------------------------------------------
# Tests: ImprovementQueue integration
# ---------------------------------------------------------------------------


class TestQueueIntegration:
    """Recommendations route correctly to ImprovementQueue."""

    def test_route_to_queue_pushes_goals(self) -> None:
        """Recommendations are converted to ImprovementGoals and pushed."""
        mock_queue = MagicMock()
        bridge = IntrospectionEvolutionBridge(queue=mock_queue)

        recs = [
            EvolutionRecommendation(
                agent_name="claude",
                recommendation="Fix low acceptance",
                severity="high",
                metric_name="proposal_acceptance_rate",
                metric_value=0.1,
                threshold=0.25,
            ),
            EvolutionRecommendation(
                agent_name="grok",
                recommendation="Improve critique quality",
                severity="medium",
                metric_name="critique_effectiveness",
                metric_value=0.15,
                threshold=0.20,
            ),
        ]
        pushed = bridge.route_to_queue(recs)

        assert pushed == 2
        assert mock_queue.push.call_count == 2

        # Verify the first pushed goal
        first_goal = mock_queue.push.call_args_list[0][0][0]
        assert first_goal.goal == "Fix low acceptance"
        assert first_goal.source == "introspection"
        assert first_goal.priority == pytest.approx(0.85)
        assert first_goal.context["agent_name"] == "claude"

    def test_route_empty_recommendations(self) -> None:
        """Routing empty recommendations returns 0 and does not touch queue."""
        mock_queue = MagicMock()
        bridge = IntrospectionEvolutionBridge(queue=mock_queue)
        assert bridge.route_to_queue([]) == 0
        mock_queue.push.assert_not_called()

    def test_route_when_queue_unavailable(self) -> None:
        """Routing gracefully returns 0 when queue cannot be constructed."""
        bridge = IntrospectionEvolutionBridge()
        # Patch lazy queue creation to return None
        bridge._get_queue = MagicMock(return_value=None)

        recs = [
            EvolutionRecommendation(
                agent_name="claude",
                recommendation="Test",
            )
        ]
        assert bridge.route_to_queue(recs) == 0


# ---------------------------------------------------------------------------
# Tests: Genesis breeding integration
# ---------------------------------------------------------------------------


class TestGenesisIntegration:
    """Recommendations optionally feed into Genesis breeding configs."""

    @patch("aragora.introspection.evolution_bridge.PopulationManager", autospec=False)
    def test_feed_genesis_adjusts_fitness(
        self,
        mock_pm_cls: MagicMock,
    ) -> None:
        """High-severity recommendations lower genome fitness."""
        # Set up mock PopulationManager
        mock_manager = MagicMock()
        mock_pm_cls.return_value = mock_manager

        mock_genome = MagicMock()
        mock_genome.genome_id = "genome-abc"
        mock_manager.genome_store.get_by_name.return_value = mock_genome

        bridge = IntrospectionEvolutionBridge()
        recs = [
            EvolutionRecommendation(
                agent_name="claude",
                recommendation="Fix acceptance",
                severity="high",
            ),
        ]

        # We need to patch the import inside feed_genesis
        with patch.dict(
            "sys.modules",
            {"aragora.genesis.breeding": MagicMock(PopulationManager=mock_pm_cls)},
        ):
            adjusted = bridge.feed_genesis(recs)

        assert adjusted == 1
        mock_manager.update_fitness.assert_called_once_with(
            "genome-abc", fitness_delta=-0.1
        )

    def test_feed_genesis_skips_low_severity(self) -> None:
        """Low-severity recommendations do not adjust genesis fitness."""
        bridge = IntrospectionEvolutionBridge()
        recs = [
            EvolutionRecommendation(
                agent_name="claude",
                recommendation="Minor issue",
                severity="low",
            ),
        ]
        # feed_genesis should return 0 because severity=low is skipped
        # before even attempting to import genesis
        adjusted = bridge.feed_genesis(recs)
        assert adjusted == 0

    def test_feed_genesis_empty_recommendations(self) -> None:
        """Empty recommendations returns 0 without touching genesis."""
        bridge = IntrospectionEvolutionBridge()
        assert bridge.feed_genesis([]) == 0

    @patch(
        "aragora.introspection.evolution_bridge.PopulationManager",
        side_effect=ImportError("genesis not available"),
    )
    def test_feed_genesis_handles_missing_module(
        self,
        mock_pm_cls: MagicMock,
    ) -> None:
        """Gracefully handles ImportError when genesis module is unavailable."""
        bridge = IntrospectionEvolutionBridge()
        recs = [
            EvolutionRecommendation(
                agent_name="claude",
                recommendation="Fix it",
                severity="high",
            ),
        ]
        # The patch makes the import raise ImportError
        with patch.dict("sys.modules", {"aragora.genesis.breeding": None}):
            adjusted = bridge.feed_genesis(recs)
        assert adjusted == 0


# ---------------------------------------------------------------------------
# Tests: Custom thresholds
# ---------------------------------------------------------------------------


class TestCustomThresholds:
    """Users can override thresholds to tune sensitivity."""

    def test_custom_threshold_changes_behavior(self) -> None:
        """A stricter threshold catches agents that default thresholds skip."""
        # With default threshold (0.25), 30% acceptance is fine
        bridge_default = IntrospectionEvolutionBridge()
        summary = _make_summary(proposals=10, accepted=3)  # 30%
        recs = bridge_default.analyze_summaries({"claude": summary})
        proposal_recs = [r for r in recs if r.metric_name == "proposal_acceptance_rate"]
        assert len(proposal_recs) == 0

        # With stricter threshold (0.50), 30% triggers a rec
        bridge_strict = IntrospectionEvolutionBridge(
            thresholds={"proposal_acceptance_rate": 0.50}
        )
        recs = bridge_strict.analyze_summaries({"claude": summary})
        proposal_recs = [r for r in recs if r.metric_name == "proposal_acceptance_rate"]
        assert len(proposal_recs) == 1


# ---------------------------------------------------------------------------
# Tests: Combined analyze
# ---------------------------------------------------------------------------


class TestCombinedAnalyze:
    """The convenience analyze() method works with both data sources."""

    def test_analyze_combines_summaries_and_snapshots(self) -> None:
        """analyze() with both sources returns recommendations from both."""
        bridge = IntrospectionEvolutionBridge()

        summary = _make_summary(proposals=10, accepted=0, influence=0.05)
        snapshot = _make_snapshot(reputation=0.1, debate_count=10)

        recs = bridge.analyze(
            summaries={"claude": summary},
            snapshots={"gemini": snapshot},
        )

        agents_mentioned = {r.agent_name for r in recs}
        assert "claude" in agents_mentioned
        assert "gemini" in agents_mentioned
