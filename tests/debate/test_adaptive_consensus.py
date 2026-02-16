"""Tests for adaptive consensus thresholds based on voter calibration quality."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import MagicMock

import pytest

from aragora.debate.adaptive_consensus import (
    NEUTRAL_BRIER,
    AdaptiveConsensus,
    AdaptiveConsensusConfig,
)


# ---------------------------------------------------------------------------
# Test helpers / fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeAgent:
    """Minimal agent stub with a ``name`` attribute."""

    name: str


@dataclass
class _FakeCalibrationSummary:
    """Mimics ``CalibrationSummary`` from ``aragora.agents.calibration``."""

    agent: str
    total_predictions: int = 0
    brier_score: float = 0.0


@dataclass
class _FakeAgentRating:
    """Mimics ``AgentRating`` from ``aragora.ranking.elo``."""

    agent_name: str
    calibration_total: int = 0
    calibration_brier_sum: float = 0.0

    @property
    def calibration_brier_score(self) -> float:
        if self.calibration_total == 0:
            return 1.0
        return self.calibration_brier_sum / self.calibration_total


def _make_calibration_tracker(
    agent_data: dict[str, _FakeCalibrationSummary],
) -> MagicMock:
    """Build a mock CalibrationTracker that returns summaries from *agent_data*."""
    tracker = MagicMock()

    def _get_summary(agent_name: str) -> _FakeCalibrationSummary:
        if agent_name in agent_data:
            return agent_data[agent_name]
        raise KeyError(f"No calibration data for {agent_name}")

    tracker.get_calibration_summary = MagicMock(side_effect=_get_summary)
    return tracker


def _make_elo_system(agent_data: dict[str, _FakeAgentRating]) -> MagicMock:
    """Build a mock EloSystem that returns ratings from *agent_data*."""
    elo = MagicMock()

    def _get_rating(agent_name: str) -> _FakeAgentRating:
        if agent_name in agent_data:
            return agent_data[agent_name]
        raise KeyError(f"No rating for {agent_name}")

    elo.get_rating = MagicMock(side_effect=_get_rating)
    return elo


# ---------------------------------------------------------------------------
# 1. Default threshold when no calibration data is available
# ---------------------------------------------------------------------------


class TestDefaultThreshold:
    """When no calibration data exists, the base threshold is returned."""

    def test_no_calibration_data_returns_base(self) -> None:
        ac = AdaptiveConsensus()
        agents = [_FakeAgent("alice"), _FakeAgent("bob")]
        threshold = ac.compute_threshold(agents)
        assert threshold == pytest.approx(0.6)

    def test_no_sources_returns_base(self) -> None:
        ac = AdaptiveConsensus()
        agents = [_FakeAgent("alice")]
        threshold = ac.compute_threshold(agents, elo_system=None, calibration_tracker=None)
        assert threshold == pytest.approx(0.6)

    def test_empty_agent_list_returns_base(self) -> None:
        ac = AdaptiveConsensus()
        threshold = ac.compute_threshold([])
        assert threshold == pytest.approx(0.6)

    def test_custom_base_threshold(self) -> None:
        config = AdaptiveConsensusConfig(base_threshold=0.7)
        ac = AdaptiveConsensus(config)
        agents = [_FakeAgent("a")]
        assert ac.compute_threshold(agents) == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# 2. Lower threshold when all agents are well-calibrated (Brier < 0.2)
# ---------------------------------------------------------------------------


class TestWellCalibratedPool:
    """Well-calibrated agents (low Brier) should lower the threshold."""

    def test_low_brier_lowers_threshold(self) -> None:
        ac = AdaptiveConsensus()
        agents = [_FakeAgent("a1"), _FakeAgent("a2"), _FakeAgent("a3")]

        # All agents have Brier = 0.10 (well below NEUTRAL_BRIER=0.25)
        tracker = _make_calibration_tracker({
            "a1": _FakeCalibrationSummary("a1", total_predictions=20, brier_score=0.10),
            "a2": _FakeCalibrationSummary("a2", total_predictions=20, brier_score=0.10),
            "a3": _FakeCalibrationSummary("a3", total_predictions=20, brier_score=0.10),
        })

        threshold = ac.compute_threshold(agents, calibration_tracker=tracker)

        # Expected: 0.6 + 0.3 * (0.10 - 0.25) = 0.6 + 0.3 * (-0.15) = 0.6 - 0.045 = 0.555
        assert threshold == pytest.approx(0.555)
        assert threshold < 0.6

    def test_very_low_brier_further_lowers(self) -> None:
        ac = AdaptiveConsensus()
        agents = [_FakeAgent("a1"), _FakeAgent("a2")]

        tracker = _make_calibration_tracker({
            "a1": _FakeCalibrationSummary("a1", total_predictions=50, brier_score=0.05),
            "a2": _FakeCalibrationSummary("a2", total_predictions=50, brier_score=0.05),
        })

        threshold = ac.compute_threshold(agents, calibration_tracker=tracker)

        # Expected: 0.6 + 0.3 * (0.05 - 0.25) = 0.6 - 0.06 = 0.54
        assert threshold == pytest.approx(0.54)

    def test_perfect_brier_zero(self) -> None:
        ac = AdaptiveConsensus()
        agents = [_FakeAgent("perfect")]

        tracker = _make_calibration_tracker({
            "perfect": _FakeCalibrationSummary("perfect", total_predictions=100, brier_score=0.0),
        })

        threshold = ac.compute_threshold(agents, calibration_tracker=tracker)

        # Expected: 0.6 + 0.3 * (0.0 - 0.25) = 0.6 - 0.075 = 0.525
        assert threshold == pytest.approx(0.525)
        assert threshold < 0.6


# ---------------------------------------------------------------------------
# 3. Higher threshold when agents are poorly calibrated (Brier > 0.4)
# ---------------------------------------------------------------------------


class TestPoorlyCalibratedPool:
    """Poorly-calibrated agents (high Brier) should raise the threshold."""

    def test_high_brier_raises_threshold(self) -> None:
        ac = AdaptiveConsensus()
        agents = [_FakeAgent("bad1"), _FakeAgent("bad2")]

        tracker = _make_calibration_tracker({
            "bad1": _FakeCalibrationSummary("bad1", total_predictions=30, brier_score=0.45),
            "bad2": _FakeCalibrationSummary("bad2", total_predictions=30, brier_score=0.45),
        })

        threshold = ac.compute_threshold(agents, calibration_tracker=tracker)

        # Expected: 0.6 + 0.3 * (0.45 - 0.25) = 0.6 + 0.3 * 0.20 = 0.6 + 0.06 = 0.66
        assert threshold == pytest.approx(0.66)
        assert threshold > 0.6

    def test_very_high_brier_raises_more(self) -> None:
        ac = AdaptiveConsensus()
        agents = [_FakeAgent("terrible")]

        tracker = _make_calibration_tracker({
            "terrible": _FakeCalibrationSummary("terrible", total_predictions=20, brier_score=0.8),
        })

        threshold = ac.compute_threshold(agents, calibration_tracker=tracker)

        # Expected: 0.6 + 0.3 * (0.8 - 0.25) = 0.6 + 0.165 = 0.765
        assert threshold == pytest.approx(0.765)

    def test_mixed_pool_averages(self) -> None:
        """One well-calibrated agent + one poorly-calibrated -> average Brier."""
        ac = AdaptiveConsensus()
        agents = [_FakeAgent("good"), _FakeAgent("bad")]

        tracker = _make_calibration_tracker({
            "good": _FakeCalibrationSummary("good", total_predictions=20, brier_score=0.10),
            "bad": _FakeCalibrationSummary("bad", total_predictions=20, brier_score=0.50),
        })

        threshold = ac.compute_threshold(agents, calibration_tracker=tracker)

        # avg_brier = (0.10 + 0.50) / 2 = 0.30
        # Expected: 0.6 + 0.3 * (0.30 - 0.25) = 0.6 + 0.015 = 0.615
        assert threshold == pytest.approx(0.615)


# ---------------------------------------------------------------------------
# 4. Threshold clamped to min/max bounds
# ---------------------------------------------------------------------------


class TestClamping:
    """Threshold is always clamped to [min_threshold, max_threshold]."""

    def test_clamped_to_min(self) -> None:
        """Extremely well-calibrated pool should not drop below min_threshold."""
        config = AdaptiveConsensusConfig(
            base_threshold=0.6,
            min_threshold=0.45,
            calibration_impact=2.0,  # Very high impact to force below min
        )
        ac = AdaptiveConsensus(config)
        agents = [_FakeAgent("a")]

        tracker = _make_calibration_tracker({
            "a": _FakeCalibrationSummary("a", total_predictions=100, brier_score=0.0),
        })

        threshold = ac.compute_threshold(agents, calibration_tracker=tracker)

        # Raw: 0.6 + 2.0 * (0.0 - 0.25) = 0.6 - 0.5 = 0.1
        # Clamped to 0.45
        assert threshold == pytest.approx(0.45)

    def test_clamped_to_max(self) -> None:
        """Extremely poorly-calibrated pool should not exceed max_threshold."""
        config = AdaptiveConsensusConfig(
            base_threshold=0.6,
            max_threshold=0.85,
            calibration_impact=2.0,
        )
        ac = AdaptiveConsensus(config)
        agents = [_FakeAgent("a")]

        tracker = _make_calibration_tracker({
            "a": _FakeCalibrationSummary("a", total_predictions=50, brier_score=1.0),
        })

        threshold = ac.compute_threshold(agents, calibration_tracker=tracker)

        # Raw: 0.6 + 2.0 * (1.0 - 0.25) = 0.6 + 1.5 = 2.1
        # Clamped to 0.85
        assert threshold == pytest.approx(0.85)

    def test_within_bounds_not_clamped(self) -> None:
        """Normal case: threshold stays within bounds and is not clamped."""
        ac = AdaptiveConsensus()
        agents = [_FakeAgent("a")]

        tracker = _make_calibration_tracker({
            "a": _FakeCalibrationSummary("a", total_predictions=20, brier_score=0.20),
        })

        threshold = ac.compute_threshold(agents, calibration_tracker=tracker)

        # Raw: 0.6 + 0.3 * (0.20 - 0.25) = 0.6 - 0.015 = 0.585
        assert 0.45 < threshold < 0.85
        assert threshold == pytest.approx(0.585)


# ---------------------------------------------------------------------------
# 5. Ignores agents with insufficient calibration samples
# ---------------------------------------------------------------------------


class TestInsufficientSamples:
    """Agents below min_calibration_samples are excluded from the average."""

    def test_below_min_samples_ignored(self) -> None:
        config = AdaptiveConsensusConfig(min_calibration_samples=10)
        ac = AdaptiveConsensus(config)
        agents = [_FakeAgent("enough"), _FakeAgent("not_enough")]

        tracker = _make_calibration_tracker({
            "enough": _FakeCalibrationSummary("enough", total_predictions=15, brier_score=0.10),
            "not_enough": _FakeCalibrationSummary("not_enough", total_predictions=3, brier_score=0.90),
        })

        threshold = ac.compute_threshold(agents, calibration_tracker=tracker)

        # Only "enough" contributes (brier=0.10)
        # Expected: 0.6 + 0.3 * (0.10 - 0.25) = 0.555
        assert threshold == pytest.approx(0.555)

    def test_all_below_min_returns_base(self) -> None:
        config = AdaptiveConsensusConfig(min_calibration_samples=50)
        ac = AdaptiveConsensus(config)
        agents = [_FakeAgent("a"), _FakeAgent("b")]

        tracker = _make_calibration_tracker({
            "a": _FakeCalibrationSummary("a", total_predictions=10, brier_score=0.1),
            "b": _FakeCalibrationSummary("b", total_predictions=5, brier_score=0.9),
        })

        threshold = ac.compute_threshold(agents, calibration_tracker=tracker)
        assert threshold == pytest.approx(0.6)  # base threshold

    def test_default_min_samples_is_five(self) -> None:
        """Default min_calibration_samples=5; agent with 5 predictions is included."""
        ac = AdaptiveConsensus()
        agents = [_FakeAgent("a")]

        tracker = _make_calibration_tracker({
            "a": _FakeCalibrationSummary("a", total_predictions=5, brier_score=0.10),
        })

        threshold = ac.compute_threshold(agents, calibration_tracker=tracker)
        # Should NOT be the base threshold; agent has exactly min_samples
        assert threshold != pytest.approx(0.6)
        assert threshold == pytest.approx(0.555)

    def test_agent_with_four_predictions_excluded(self) -> None:
        """Agent with 4 predictions (below default 5) is excluded."""
        ac = AdaptiveConsensus()
        agents = [_FakeAgent("a")]

        tracker = _make_calibration_tracker({
            "a": _FakeCalibrationSummary("a", total_predictions=4, brier_score=0.10),
        })

        threshold = ac.compute_threshold(agents, calibration_tracker=tracker)
        assert threshold == pytest.approx(0.6)  # falls back to base


# ---------------------------------------------------------------------------
# 6. Explanation string includes calibration details
# ---------------------------------------------------------------------------


class TestExplanation:
    """compute_threshold_with_explanation returns informative audit text."""

    def test_explanation_when_no_data(self) -> None:
        ac = AdaptiveConsensus()
        agents = [_FakeAgent("a")]
        threshold, explanation = ac.compute_threshold_with_explanation(agents)

        assert threshold == pytest.approx(0.6)
        assert "No calibration data" in explanation
        assert "base threshold" in explanation

    def test_explanation_includes_per_agent_detail(self) -> None:
        ac = AdaptiveConsensus()
        agents = [_FakeAgent("alice"), _FakeAgent("bob")]

        tracker = _make_calibration_tracker({
            "alice": _FakeCalibrationSummary("alice", total_predictions=20, brier_score=0.15),
            "bob": _FakeCalibrationSummary("bob", total_predictions=30, brier_score=0.20),
        })

        threshold, explanation = ac.compute_threshold_with_explanation(
            agents, calibration_tracker=tracker
        )

        assert "alice" in explanation
        assert "bob" in explanation
        assert "brier=" in explanation
        assert "samples=" in explanation
        assert "calibration_tracker" in explanation

    def test_explanation_includes_formula(self) -> None:
        ac = AdaptiveConsensus()
        agents = [_FakeAgent("a")]

        tracker = _make_calibration_tracker({
            "a": _FakeCalibrationSummary("a", total_predictions=10, brier_score=0.30),
        })

        threshold, explanation = ac.compute_threshold_with_explanation(
            agents, calibration_tracker=tracker
        )

        assert "Formula" in explanation
        assert "Average Brier score" in explanation

    def test_explanation_notes_clamping(self) -> None:
        config = AdaptiveConsensusConfig(calibration_impact=5.0)
        ac = AdaptiveConsensus(config)
        agents = [_FakeAgent("a")]

        tracker = _make_calibration_tracker({
            "a": _FakeCalibrationSummary("a", total_predictions=50, brier_score=0.0),
        })

        threshold, explanation = ac.compute_threshold_with_explanation(
            agents, calibration_tracker=tracker
        )

        assert "clamped" in explanation

    def test_explanation_no_clamp_note_when_within_bounds(self) -> None:
        ac = AdaptiveConsensus()
        agents = [_FakeAgent("a")]

        tracker = _make_calibration_tracker({
            "a": _FakeCalibrationSummary("a", total_predictions=20, brier_score=NEUTRAL_BRIER),
        })

        threshold, explanation = ac.compute_threshold_with_explanation(
            agents, calibration_tracker=tracker
        )

        assert "clamped" not in explanation

    def test_explanation_threshold_matches_compute_threshold(self) -> None:
        ac = AdaptiveConsensus()
        agents = [_FakeAgent("x"), _FakeAgent("y")]

        tracker = _make_calibration_tracker({
            "x": _FakeCalibrationSummary("x", total_predictions=25, brier_score=0.12),
            "y": _FakeCalibrationSummary("y", total_predictions=40, brier_score=0.18),
        })

        threshold_plain = ac.compute_threshold(agents, calibration_tracker=tracker)
        threshold_expl, _ = ac.compute_threshold_with_explanation(
            agents, calibration_tracker=tracker
        )

        assert threshold_plain == pytest.approx(threshold_expl)


# ---------------------------------------------------------------------------
# 7. Works with both CalibrationTracker and ELO system fallback
# ---------------------------------------------------------------------------


class TestEloFallback:
    """When CalibrationTracker is unavailable, fall back to ELO system."""

    def test_elo_system_fallback(self) -> None:
        ac = AdaptiveConsensus()
        agents = [_FakeAgent("a1"), _FakeAgent("a2")]

        elo = _make_elo_system({
            "a1": _FakeAgentRating("a1", calibration_total=20, calibration_brier_sum=2.0),
            "a2": _FakeAgentRating("a2", calibration_total=20, calibration_brier_sum=4.0),
        })

        threshold = ac.compute_threshold(agents, elo_system=elo)

        # a1 brier = 2.0/20 = 0.10, a2 brier = 4.0/20 = 0.20
        # avg_brier = 0.15
        # Expected: 0.6 + 0.3 * (0.15 - 0.25) = 0.6 - 0.03 = 0.57
        assert threshold == pytest.approx(0.57)

    def test_calibration_tracker_preferred_over_elo(self) -> None:
        """When both are provided, CalibrationTracker should be used."""
        ac = AdaptiveConsensus()
        agents = [_FakeAgent("a1")]

        tracker = _make_calibration_tracker({
            "a1": _FakeCalibrationSummary("a1", total_predictions=20, brier_score=0.10),
        })

        # ELO has different (worse) data -- should be ignored
        elo = _make_elo_system({
            "a1": _FakeAgentRating("a1", calibration_total=20, calibration_brier_sum=16.0),
        })

        threshold = ac.compute_threshold(
            agents, elo_system=elo, calibration_tracker=tracker
        )

        # Should use tracker brier=0.10, not elo brier=0.80
        assert threshold == pytest.approx(0.555)

    def test_elo_used_when_tracker_has_no_data_for_agent(self) -> None:
        """If tracker raises for an agent, fall back to ELO for that agent."""
        ac = AdaptiveConsensus()
        agents = [_FakeAgent("tracked"), _FakeAgent("elo_only")]

        tracker = _make_calibration_tracker({
            "tracked": _FakeCalibrationSummary("tracked", total_predictions=20, brier_score=0.10),
            # "elo_only" not in tracker -> KeyError
        })

        elo = _make_elo_system({
            "elo_only": _FakeAgentRating("elo_only", calibration_total=20, calibration_brier_sum=4.0),
        })

        threshold = ac.compute_threshold(
            agents, elo_system=elo, calibration_tracker=tracker
        )

        # tracked -> tracker brier=0.10
        # elo_only -> elo brier=4.0/20=0.20
        # avg = 0.15
        # Expected: 0.6 + 0.3 * (0.15 - 0.25) = 0.57
        assert threshold == pytest.approx(0.57)

    def test_elo_insufficient_samples_ignored(self) -> None:
        ac = AdaptiveConsensus()
        agents = [_FakeAgent("a")]

        elo = _make_elo_system({
            "a": _FakeAgentRating("a", calibration_total=2, calibration_brier_sum=0.2),
        })

        threshold = ac.compute_threshold(agents, elo_system=elo)
        assert threshold == pytest.approx(0.6)  # base, because 2 < 5

    def test_explanation_shows_elo_source(self) -> None:
        ac = AdaptiveConsensus()
        agents = [_FakeAgent("a")]

        elo = _make_elo_system({
            "a": _FakeAgentRating("a", calibration_total=10, calibration_brier_sum=2.0),
        })

        _, explanation = ac.compute_threshold_with_explanation(agents, elo_system=elo)
        assert "elo_system" in explanation


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Miscellaneous edge cases."""

    def test_neutral_brier_returns_base(self) -> None:
        """Brier == NEUTRAL_BRIER should give exactly the base threshold."""
        ac = AdaptiveConsensus()
        agents = [_FakeAgent("a")]

        tracker = _make_calibration_tracker({
            "a": _FakeCalibrationSummary("a", total_predictions=20, brier_score=NEUTRAL_BRIER),
        })

        threshold = ac.compute_threshold(agents, calibration_tracker=tracker)
        assert threshold == pytest.approx(0.6)

    def test_single_agent_pool(self) -> None:
        ac = AdaptiveConsensus()
        agents = [_FakeAgent("solo")]

        tracker = _make_calibration_tracker({
            "solo": _FakeCalibrationSummary("solo", total_predictions=50, brier_score=0.15),
        })

        threshold = ac.compute_threshold(agents, calibration_tracker=tracker)
        # 0.6 + 0.3 * (0.15 - 0.25) = 0.6 - 0.03 = 0.57
        assert threshold == pytest.approx(0.57)

    def test_config_is_optional(self) -> None:
        """AdaptiveConsensus() with no config should use defaults."""
        ac = AdaptiveConsensus()
        assert ac.config.base_threshold == 0.6
        assert ac.config.min_threshold == 0.45
        assert ac.config.max_threshold == 0.85
        assert ac.config.calibration_impact == 0.3
        assert ac.config.min_calibration_samples == 5

    def test_tracker_raises_attribute_error(self) -> None:
        """If tracker raises AttributeError, agent is skipped gracefully."""
        ac = AdaptiveConsensus()
        agents = [_FakeAgent("a")]

        tracker = MagicMock()
        tracker.get_calibration_summary.side_effect = AttributeError("broken")

        threshold = ac.compute_threshold(agents, calibration_tracker=tracker)
        assert threshold == pytest.approx(0.6)

    def test_elo_raises_value_error(self) -> None:
        """If ELO raises ValueError, agent is skipped gracefully."""
        ac = AdaptiveConsensus()
        agents = [_FakeAgent("a")]

        elo = MagicMock()
        elo.get_rating.side_effect = ValueError("bad agent name")

        threshold = ac.compute_threshold(agents, elo_system=elo)
        assert threshold == pytest.approx(0.6)

    def test_agent_without_name_attribute(self) -> None:
        """Agents without ``name`` fall back to str() representation."""
        ac = AdaptiveConsensus()

        # Use a plain string as an "agent" -- getattr(agent, "name", str(agent))
        agents = ["agent_string"]

        tracker = _make_calibration_tracker({
            "agent_string": _FakeCalibrationSummary(
                "agent_string", total_predictions=10, brier_score=0.20
            ),
        })

        threshold = ac.compute_threshold(agents, calibration_tracker=tracker)
        # 0.6 + 0.3 * (0.20 - 0.25) = 0.585
        assert threshold == pytest.approx(0.585)

    def test_large_pool_averages_correctly(self) -> None:
        """Verify averaging works correctly with many agents."""
        ac = AdaptiveConsensus()
        n = 20
        agents = [_FakeAgent(f"agent_{i}") for i in range(n)]

        # Half have brier=0.10, half have brier=0.40
        data = {}
        for i in range(n):
            brier = 0.10 if i < n // 2 else 0.40
            data[f"agent_{i}"] = _FakeCalibrationSummary(
                f"agent_{i}", total_predictions=30, brier_score=brier
            )

        tracker = _make_calibration_tracker(data)
        threshold = ac.compute_threshold(agents, calibration_tracker=tracker)

        # avg_brier = (10*0.10 + 10*0.40) / 20 = 5.0/20 = 0.25 = NEUTRAL
        assert threshold == pytest.approx(0.6)
