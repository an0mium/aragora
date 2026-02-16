"""Tests for NomicOutcomeTracker - debate quality regression detection."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from aragora.nomic.outcome_tracker import (
    DebateMetrics,
    DebateScenario,
    NomicOutcomeTracker,
    OutcomeComparison,
)


# --- DebateMetrics ---


class TestDebateMetrics:
    def test_defaults(self):
        m = DebateMetrics()
        assert m.consensus_rate == 0.0
        assert m.avg_rounds == 0.0
        assert m.avg_tokens == 0
        assert m.calibration_spread == 0.0
        assert m.timestamp > 0

    def test_round_trip(self):
        m = DebateMetrics(
            consensus_rate=0.8,
            avg_rounds=2.5,
            avg_tokens=1500,
            calibration_spread=0.05,
            timestamp=1000.0,
        )
        d = m.to_dict()
        m2 = DebateMetrics.from_dict(d)
        assert m2.consensus_rate == 0.8
        assert m2.avg_rounds == 2.5
        assert m2.avg_tokens == 1500
        assert m2.calibration_spread == 0.05
        assert m2.timestamp == 1000.0


# --- OutcomeComparison ---


class TestOutcomeComparison:
    def test_round_trip(self):
        baseline = DebateMetrics(consensus_rate=0.7, avg_rounds=3.0, avg_tokens=2000)
        after = DebateMetrics(consensus_rate=0.9, avg_rounds=2.0, avg_tokens=1500)
        c = OutcomeComparison(
            baseline=baseline,
            after=after,
            improved=True,
            metrics_delta={"consensus_rate": 0.2},
            recommendation="keep",
        )
        d = c.to_dict()
        c2 = OutcomeComparison.from_dict(d)
        assert c2.improved is True
        assert c2.recommendation == "keep"
        assert c2.metrics_delta["consensus_rate"] == 0.2
        assert c2.baseline.consensus_rate == 0.7
        assert c2.after.consensus_rate == 0.9


# --- NomicOutcomeTracker ---


@pytest.fixture
def mock_runner():
    """Returns an async runner that yields controllable debate results."""

    results: list[dict] = []

    async def runner(topic: str, agent_count: int, expected_rounds: int) -> dict:
        if results:
            return results.pop(0)
        return {
            "consensus_reached": True,
            "rounds": expected_rounds,
            "tokens_used": expected_rounds * agent_count * 500,
            "brier_scores": [0.2, 0.25, 0.3][:agent_count],
        }

    runner._results = results  # type: ignore[attr-defined]
    return runner


class TestCaptureMetrics:
    @pytest.mark.asyncio
    async def test_capture_baseline_with_default_runner(self):
        tracker = NomicOutcomeTracker()
        metrics = await tracker.capture_baseline()
        # Default runner returns consensus for all 3 scenarios
        assert metrics.consensus_rate == 1.0
        assert metrics.avg_rounds > 0
        assert metrics.avg_tokens > 0

    @pytest.mark.asyncio
    async def test_capture_custom_runner(self, mock_runner):
        mock_runner._results.extend(
            [
                {"consensus_reached": True, "rounds": 2, "tokens_used": 1000, "brier_scores": [0.1, 0.2]},
                {"consensus_reached": False, "rounds": 5, "tokens_used": 3000, "brier_scores": [0.3, 0.4]},
                {"consensus_reached": True, "rounds": 3, "tokens_used": 2000, "brier_scores": [0.2, 0.2]},
            ]
        )
        tracker = NomicOutcomeTracker(scenario_runner=mock_runner)
        metrics = await tracker.capture_baseline()
        # 2 out of 3 reached consensus
        assert abs(metrics.consensus_rate - 2.0 / 3.0) < 1e-9
        assert abs(metrics.avg_rounds - (2 + 5 + 3) / 3.0) < 1e-9
        assert metrics.avg_tokens == (1000 + 3000 + 2000) // 3
        assert metrics.calibration_spread > 0

    @pytest.mark.asyncio
    async def test_capture_handles_scenario_failure(self):
        async def failing_runner(topic, agent_count, expected_rounds):
            raise RuntimeError("API unavailable")

        tracker = NomicOutcomeTracker(scenario_runner=failing_runner)
        # Should not raise, just log warnings and return zero metrics
        metrics = await tracker.capture_baseline()
        assert metrics.consensus_rate == 0.0

    @pytest.mark.asyncio
    async def test_capture_with_empty_scenarios(self):
        tracker = NomicOutcomeTracker(scenarios=[])
        metrics = await tracker.capture_baseline()
        assert metrics.consensus_rate == 0.0
        assert metrics.avg_rounds == 0.0


class TestCompare:
    def test_improvement_detected(self):
        tracker = NomicOutcomeTracker()
        baseline = DebateMetrics(
            consensus_rate=0.6, avg_rounds=4.0, avg_tokens=3000, calibration_spread=0.15
        )
        after = DebateMetrics(
            consensus_rate=0.9, avg_rounds=2.5, avg_tokens=2000, calibration_spread=0.08
        )
        result = tracker.compare(baseline, after)
        assert result.improved is True
        assert result.recommendation == "keep"
        assert result.metrics_delta["consensus_rate"] > 0
        assert result.metrics_delta["avg_rounds"] < 0
        assert result.metrics_delta["avg_tokens"] < 0
        assert result.metrics_delta["calibration_spread"] < 0

    def test_regression_detected_single(self):
        tracker = NomicOutcomeTracker()
        baseline = DebateMetrics(
            consensus_rate=0.9, avg_rounds=2.0, avg_tokens=1000, calibration_spread=0.05
        )
        # Only consensus_rate degrades significantly (>5%)
        after = DebateMetrics(
            consensus_rate=0.7, avg_rounds=2.0, avg_tokens=1000, calibration_spread=0.05
        )
        result = tracker.compare(baseline, after)
        assert result.improved is False
        # Single regression => "review", not "revert"
        assert result.recommendation == "review"

    def test_regression_detected_multiple(self):
        tracker = NomicOutcomeTracker()
        baseline = DebateMetrics(
            consensus_rate=0.9, avg_rounds=2.0, avg_tokens=1000, calibration_spread=0.05
        )
        # Two metrics degrade significantly
        after = DebateMetrics(
            consensus_rate=0.7, avg_rounds=3.0, avg_tokens=2000, calibration_spread=0.10
        )
        result = tracker.compare(baseline, after)
        assert result.improved is False
        assert result.recommendation == "revert"

    def test_no_change_is_keep(self):
        tracker = NomicOutcomeTracker()
        baseline = DebateMetrics(
            consensus_rate=0.8, avg_rounds=3.0, avg_tokens=2000, calibration_spread=0.10
        )
        after = DebateMetrics(
            consensus_rate=0.8, avg_rounds=3.0, avg_tokens=2000, calibration_spread=0.10
        )
        result = tracker.compare(baseline, after)
        assert result.improved is True
        assert result.recommendation == "keep"

    def test_small_degradation_within_threshold(self):
        """Degradation under 5% should not count as regression."""
        tracker = NomicOutcomeTracker()
        baseline = DebateMetrics(
            consensus_rate=1.0, avg_rounds=3.0, avg_tokens=2000, calibration_spread=0.10
        )
        # 4% degradation in consensus_rate (within 5% threshold)
        after = DebateMetrics(
            consensus_rate=0.96, avg_rounds=3.0, avg_tokens=2000, calibration_spread=0.10
        )
        result = tracker.compare(baseline, after)
        # No regression because 4% < 5% threshold
        assert result.improved is True
        assert result.recommendation == "keep"

    def test_custom_threshold(self):
        # Strict 1% threshold
        tracker = NomicOutcomeTracker(degradation_threshold=0.01)
        baseline = DebateMetrics(consensus_rate=1.0, avg_rounds=3.0)
        after = DebateMetrics(consensus_rate=0.97, avg_rounds=3.0)
        result = tracker.compare(baseline, after)
        # 3% > 1% threshold => regression
        assert result.improved is False

    def test_zero_baseline_no_division_error(self):
        """Metrics starting at zero should not cause division-by-zero."""
        tracker = NomicOutcomeTracker()
        baseline = DebateMetrics(
            consensus_rate=0.0, avg_rounds=0.0, avg_tokens=0, calibration_spread=0.0
        )
        after = DebateMetrics(
            consensus_rate=0.5, avg_rounds=2.0, avg_tokens=1000, calibration_spread=0.05
        )
        result = tracker.compare(baseline, after)
        assert result.improved is True
        assert result.recommendation == "keep"


class TestRecordCycleOutcome:
    def test_records_to_cycle_store(self):
        mock_store = MagicMock()
        mock_record = MagicMock()
        mock_record.evidence_quality_scores = {}
        mock_store.load_cycle.return_value = mock_record

        tracker = NomicOutcomeTracker(cycle_store=mock_store)
        comparison = OutcomeComparison(
            baseline=DebateMetrics(consensus_rate=0.7),
            after=DebateMetrics(consensus_rate=0.9),
            improved=True,
            metrics_delta={"consensus_rate": 0.2},
            recommendation="keep",
        )

        tracker.record_cycle_outcome("cycle_1", comparison)

        mock_store.load_cycle.assert_called_once_with("cycle_1")
        mock_record.add_pattern_reinforcement.assert_called_once()
        mock_store.save_cycle.assert_called_once_with(mock_record)

        # Verify the metric delta was stored
        assert "outcome_consensus_rate_delta" in mock_record.evidence_quality_scores

    def test_skips_when_cycle_not_found(self):
        mock_store = MagicMock()
        mock_store.load_cycle.return_value = None

        tracker = NomicOutcomeTracker(cycle_store=mock_store)
        comparison = OutcomeComparison(
            baseline=DebateMetrics(),
            after=DebateMetrics(),
            improved=True,
            recommendation="keep",
        )

        # Should not raise
        tracker.record_cycle_outcome("nonexistent", comparison)
        mock_store.save_cycle.assert_not_called()

    def test_skips_when_no_store_and_import_fails(self, monkeypatch):
        tracker = NomicOutcomeTracker(cycle_store=None)

        # Make the fallback import fail
        import aragora.nomic.outcome_tracker as ot_mod

        def mock_get_cycle_store():
            raise ImportError("no store")

        monkeypatch.setattr(
            "aragora.nomic.outcome_tracker.logger",
            ot_mod.logger,
        )

        comparison = OutcomeComparison(
            baseline=DebateMetrics(),
            after=DebateMetrics(),
            improved=True,
            recommendation="keep",
        )

        # Patch the import path
        monkeypatch.setattr(
            "aragora.nomic.cycle_store.get_cycle_store",
            mock_get_cycle_store,
        )

        # Should not raise even when store import fails
        tracker.record_cycle_outcome("cycle_x", comparison)


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Simulate a complete baseline -> change -> after -> compare cycle."""
        call_count = 0

        async def improving_runner(topic, agent_count, expected_rounds):
            nonlocal call_count
            call_count += 1
            # First 3 calls are baseline (worse metrics)
            # Next 3 calls are "after" (better metrics)
            if call_count <= 3:
                return {
                    "consensus_reached": call_count != 2,  # 2/3 consensus
                    "rounds": 4,
                    "tokens_used": 3000,
                    "brier_scores": [0.3, 0.35, 0.4],
                }
            else:
                return {
                    "consensus_reached": True,  # 3/3 consensus
                    "rounds": 2,
                    "tokens_used": 1500,
                    "brier_scores": [0.15, 0.18, 0.20],
                }

        tracker = NomicOutcomeTracker(scenario_runner=improving_runner)
        baseline = await tracker.capture_baseline()
        after = await tracker.capture_after()
        comparison = tracker.compare(baseline, after)

        assert comparison.improved is True
        assert comparison.recommendation == "keep"
        assert comparison.metrics_delta["consensus_rate"] > 0
        assert comparison.metrics_delta["avg_rounds"] < 0
        assert comparison.metrics_delta["avg_tokens"] < 0

    @pytest.mark.asyncio
    async def test_full_workflow_with_regression(self):
        """Simulate a cycle that causes debate quality regression."""
        call_count = 0

        async def regressing_runner(topic, agent_count, expected_rounds):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return {
                    "consensus_reached": True,
                    "rounds": 2,
                    "tokens_used": 1000,
                    "brier_scores": [0.15, 0.18],
                }
            else:
                return {
                    "consensus_reached": call_count != 5,  # 2/3 consensus
                    "rounds": 5,
                    "tokens_used": 4000,
                    "brier_scores": [0.35, 0.45],
                }

        tracker = NomicOutcomeTracker(scenario_runner=regressing_runner)
        baseline = await tracker.capture_baseline()
        after = await tracker.capture_after()
        comparison = tracker.compare(baseline, after)

        assert comparison.improved is False
        assert comparison.recommendation in ("revert", "review")
