"""Tests for NomicOutcomeTracker - debate quality regression detection."""

from __future__ import annotations

import time
import unittest.mock
from unittest.mock import MagicMock

import pytest

from aragora.nomic.outcome_tracker import (
    DebateMetrics,
    DebateScenario,
    NomicOutcomeTracker,
    OutcomeComparison,
    _default_scenario_runner,
    _lightweight_debate_runner,
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


# --- get_regression_history ---


class TestGetRegressionHistory:
    def test_returns_empty_when_no_store(self):
        """Should return empty list when cycle store import fails."""
        import sys
        from unittest.mock import patch

        # Block the import so the lazy import inside get_regression_history fails
        with patch.dict(sys.modules, {"aragora.nomic.cycle_store": None}):
            result = NomicOutcomeTracker.get_regression_history()
        assert result == []

    def test_returns_empty_for_no_cycles(self, monkeypatch):
        """Should return empty list when no cycles exist."""
        mock_store = MagicMock()
        mock_store.get_recent_cycles.return_value = []
        monkeypatch.setattr(
            "aragora.nomic.cycle_store.get_cycle_store",
            lambda: mock_store,
        )
        result = NomicOutcomeTracker.get_regression_history()
        assert result == []

    def test_filters_for_regressions(self, monkeypatch):
        """Should return only cycles with negative outcome deltas."""
        from aragora.nomic.cycle_record import NomicCycleRecord

        # Cycle with regression (consensus_rate went down, avg_tokens went up)
        regressed_cycle = NomicCycleRecord(
            cycle_id="cycle_regressed_abc",
            started_at=1000.0,
        )
        regressed_cycle.evidence_quality_scores = {
            "outcome_consensus_rate_delta": -0.15,
            "outcome_avg_tokens_delta": 500.0,
            "outcome_avg_rounds_delta": -1.0,  # improved (lower is better)
        }

        # Cycle with improvement (all deltas positive for higher-is-better)
        improved_cycle = NomicCycleRecord(
            cycle_id="cycle_improved_xyz",
            started_at=2000.0,
        )
        improved_cycle.evidence_quality_scores = {
            "outcome_consensus_rate_delta": 0.2,
            "outcome_avg_tokens_delta": -300.0,
            "outcome_avg_rounds_delta": -0.5,
        }

        # Cycle with no outcome data at all
        empty_cycle = NomicCycleRecord(
            cycle_id="cycle_empty",
            started_at=3000.0,
        )

        mock_store = MagicMock()
        mock_store.get_recent_cycles.return_value = [
            regressed_cycle,
            improved_cycle,
            empty_cycle,
        ]
        monkeypatch.setattr(
            "aragora.nomic.cycle_store.get_cycle_store",
            lambda: mock_store,
        )

        result = NomicOutcomeTracker.get_regression_history(limit=10)

        assert len(result) == 1
        assert result[0]["cycle_id"] == "cycle_regressed_abc"
        assert "consensus_rate" in result[0]["regressed_metrics"]
        assert "avg_tokens" in result[0]["regressed_metrics"]
        # avg_rounds improved (delta < 0 for lower-is-better), so should NOT be in regressed
        assert "avg_rounds" not in result[0]["regressed_metrics"]

    def test_limit_parameter(self, monkeypatch):
        """Should pass limit to the cycle store."""
        mock_store = MagicMock()
        mock_store.get_recent_cycles.return_value = []
        monkeypatch.setattr(
            "aragora.nomic.cycle_store.get_cycle_store",
            lambda: mock_store,
        )

        NomicOutcomeTracker.get_regression_history(limit=3)
        mock_store.get_recent_cycles.assert_called_once_with(3)

    def test_recommendation_from_reinforcement(self, monkeypatch):
        """Should derive recommendation from pattern reinforcements."""
        from aragora.nomic.cycle_record import NomicCycleRecord, PatternReinforcement

        cycle = NomicCycleRecord(cycle_id="cycle_revert", started_at=1000.0)
        cycle.evidence_quality_scores = {
            "outcome_consensus_rate_delta": -0.2,
            "outcome_avg_tokens_delta": 1000.0,
        }
        cycle.pattern_reinforcements.append(
            PatternReinforcement(
                pattern_type="outcome_tracking",
                description="Debate quality degraded: recommendation=revert",
                success=False,
                confidence=0.3,
            )
        )

        mock_store = MagicMock()
        mock_store.get_recent_cycles.return_value = [cycle]
        monkeypatch.setattr(
            "aragora.nomic.cycle_store.get_cycle_store",
            lambda: mock_store,
        )

        result = NomicOutcomeTracker.get_regression_history()
        assert len(result) == 1
        assert result[0]["recommendation"] == "revert"

    def test_ignores_non_outcome_scores(self, monkeypatch):
        """Should ignore evidence_quality_scores that are not outcome deltas."""
        from aragora.nomic.cycle_record import NomicCycleRecord

        cycle = NomicCycleRecord(cycle_id="cycle_other", started_at=1000.0)
        cycle.evidence_quality_scores = {
            "some_other_metric": -0.5,
            "evidence_relevance": 0.9,
        }

        mock_store = MagicMock()
        mock_store.get_recent_cycles.return_value = [cycle]
        monkeypatch.setattr(
            "aragora.nomic.cycle_store.get_cycle_store",
            lambda: mock_store,
        )

        result = NomicOutcomeTracker.get_regression_history()
        assert result == []


# --- _lightweight_debate_runner ---


class TestLightweightDebateRunner:
    @pytest.mark.asyncio
    async def test_falls_back_to_simulation(self):
        """When Arena/agents are unavailable, should fall back to _default_scenario_runner."""
        with unittest.mock.patch(
            "aragora.nomic.outcome_tracker.Arena",
            side_effect=ImportError("no arena"),
            create=True,
        ), unittest.mock.patch(
            "aragora.agents.base.create_agent",
            side_effect=ImportError("no agents"),
        ):
            result = await _lightweight_debate_runner("test topic", 2, 3)
        assert "consensus_reached" in result
        assert "rounds" in result
        assert "tokens_used" in result
        assert "brier_scores" in result

    @pytest.mark.asyncio
    async def test_returns_correct_keys(self):
        """Result dict should always have the four standard keys."""
        with unittest.mock.patch(
            "aragora.agents.base.create_agent",
            side_effect=ImportError("no agents"),
        ):
            result = await _lightweight_debate_runner("analyze caching strategy", 3, 2)
        for key in ("consensus_reached", "rounds", "tokens_used", "brier_scores"):
            assert key in result, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_simulation_fallback_values_match_default(self):
        """When falling back, values should match _default_scenario_runner output."""
        with unittest.mock.patch(
            "aragora.agents.base.create_agent",
            side_effect=ImportError("no agents"),
        ):
            lightweight = await _lightweight_debate_runner("topic", 2, 3)
        default = await _default_scenario_runner("topic", 2, 3)
        # Both should produce valid results (may differ if Arena is available)
        assert isinstance(lightweight["consensus_reached"], bool)
        assert isinstance(default["consensus_reached"], bool)
        assert isinstance(lightweight["rounds"], int)
        assert isinstance(lightweight["tokens_used"], int)


# --- create_with_real_debates ---


class TestCreateWithRealDebates:
    def test_factory_returns_tracker(self):
        tracker = NomicOutcomeTracker.create_with_real_debates()
        assert isinstance(tracker, NomicOutcomeTracker)
        # Uses the lightweight runner, not the default simulated one
        assert tracker._runner is _lightweight_debate_runner
        assert tracker._runner is not _default_scenario_runner

    def test_factory_accepts_threshold(self):
        tracker = NomicOutcomeTracker.create_with_real_debates(degradation_threshold=0.10)
        assert tracker.degradation_threshold == 0.10

    def test_factory_accepts_cycle_store(self):
        store = MagicMock()
        tracker = NomicOutcomeTracker.create_with_real_debates(cycle_store=store)
        assert tracker._cycle_store is store

    @pytest.mark.asyncio
    async def test_captures_metrics(self):
        """Tracker from factory should be able to capture baseline metrics."""
        tracker = NomicOutcomeTracker.create_with_real_debates()
        with unittest.mock.patch(
            "aragora.agents.base.create_agent",
            side_effect=ImportError("no agents"),
        ):
            metrics = await tracker.capture_baseline()
        # Should always return valid metrics (falls back to simulation if needed)
        assert metrics.consensus_rate >= 0.0
        assert metrics.avg_rounds >= 0.0


# --- verify_diff ---


class TestVerifyDiff:
    @pytest.mark.asyncio
    async def test_verify_empty_diff(self):
        """Empty diff should still return a result dict."""
        tracker = NomicOutcomeTracker()
        with unittest.mock.patch.dict(
            "sys.modules",
            {"aragora.compat.openclaw.pr_review_runner": None},
        ):
            result = await tracker.verify_diff("")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_verify_returns_passed_key(self):
        """Result should always include the 'passed' key."""
        tracker = NomicOutcomeTracker()
        with unittest.mock.patch.dict(
            "sys.modules",
            {"aragora.compat.openclaw.pr_review_runner": None},
        ):
            result = await tracker.verify_diff(
                "--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-old\n+new"
            )
        assert "passed" in result

    @pytest.mark.asyncio
    async def test_verify_diff_with_mock_runner(self, monkeypatch):
        """When PRReviewRunner is mocked, verify_diff should use it."""
        from unittest.mock import AsyncMock

        mock_review = MagicMock()
        mock_review.findings = []
        mock_review.error = None
        mock_review.agreement_score = 0.85

        mock_runner_cls = MagicMock()
        mock_runner_instance = MagicMock()
        mock_runner_instance.review_diff = AsyncMock(return_value=mock_review)
        mock_runner_cls.return_value = mock_runner_instance

        monkeypatch.setattr(
            "aragora.compat.openclaw.pr_review_runner.PRReviewRunner",
            mock_runner_cls,
        )

        tracker = NomicOutcomeTracker()
        result = await tracker.verify_diff("--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b")

        assert result["passed"] is True
        assert result["findings_count"] == 0
        assert result["agreement_score"] == 0.85

    @pytest.mark.asyncio
    async def test_verify_diff_with_critical_findings(self, monkeypatch):
        """Should report passed=False when review contains critical findings."""
        from unittest.mock import AsyncMock

        mock_finding = MagicMock()
        mock_finding.severity = "critical"

        mock_review = MagicMock()
        mock_review.findings = [mock_finding]
        mock_review.error = None
        mock_review.agreement_score = 0.6

        mock_runner_cls = MagicMock()
        mock_runner_instance = MagicMock()
        mock_runner_instance.review_diff = AsyncMock(return_value=mock_review)
        mock_runner_cls.return_value = mock_runner_instance

        monkeypatch.setattr(
            "aragora.compat.openclaw.pr_review_runner.PRReviewRunner",
            mock_runner_cls,
        )

        tracker = NomicOutcomeTracker()
        result = await tracker.verify_diff("--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b")

        assert result["passed"] is False
        assert result["findings_count"] == 1

    @pytest.mark.asyncio
    async def test_verify_diff_handles_runtime_error(self, monkeypatch):
        """Should return passed=False with error message on RuntimeError."""
        from unittest.mock import AsyncMock

        mock_runner_cls = MagicMock()
        mock_runner_instance = MagicMock()
        mock_runner_instance.review_diff = AsyncMock(side_effect=RuntimeError("API down"))
        mock_runner_cls.return_value = mock_runner_instance

        monkeypatch.setattr(
            "aragora.compat.openclaw.pr_review_runner.PRReviewRunner",
            mock_runner_cls,
        )

        tracker = NomicOutcomeTracker()
        result = await tracker.verify_diff("some diff")

        assert result["passed"] is False
        assert "API down" in result["error"]

    @pytest.mark.asyncio
    async def test_verify_diff_import_error_skips(self, monkeypatch):
        """When PRReviewRunner cannot be imported, should return skipped=True."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "pr_review_runner" in name:
                raise ImportError("not available")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        tracker = NomicOutcomeTracker()
        result = await tracker.verify_diff("some diff")

        assert result["passed"] is True
        assert result.get("skipped") is True
