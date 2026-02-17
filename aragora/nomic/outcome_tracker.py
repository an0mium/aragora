"""Nomic Loop Outcome Tracker - measures whether improvements helped debate quality.

Captures debate quality metrics before and after a Nomic Loop cycle to detect
silent regressions. If an improvement cycle makes code changes that degrade
consensus rates, increase token costs, or widen calibration spread, the tracker
flags the regression and recommends a revert.

Usage:
    tracker = NomicOutcomeTracker()

    baseline = await tracker.capture_baseline()
    # ... run improvement cycle ...
    after = await tracker.capture_after()

    comparison = tracker.compare(baseline, after)
    if not comparison.improved:
        print(f"Regression detected: {comparison.recommendation}")

    # Persist to cycle store
    tracker.record_cycle_outcome("cycle_123", comparison)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable, Awaitable

logger = logging.getLogger(__name__)

# Default degradation threshold: a metric must degrade by more than 5%
# relative to its baseline to count as a regression.
DEGRADATION_THRESHOLD = 0.05

# Default number of test scenarios to run for metric capture.
DEFAULT_SCENARIO_COUNT = 3


@dataclass
class DebateMetrics:
    """Point-in-time snapshot of debate quality metrics."""

    consensus_rate: float = 0.0  # 0-1, fraction of debates reaching consensus
    avg_rounds: float = 0.0  # average rounds to reach consensus
    avg_tokens: int = 0  # average token cost per debate
    calibration_spread: float = 0.0  # std dev of agent Brier scores (lower = better)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "consensus_rate": self.consensus_rate,
            "avg_rounds": self.avg_rounds,
            "avg_tokens": self.avg_tokens,
            "calibration_spread": self.calibration_spread,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DebateMetrics:
        return cls(
            consensus_rate=data.get("consensus_rate", 0.0),
            avg_rounds=data.get("avg_rounds", 0.0),
            avg_tokens=data.get("avg_tokens", 0),
            calibration_spread=data.get("calibration_spread", 0.0),
            timestamp=data.get("timestamp", 0.0),
        )


@dataclass
class OutcomeComparison:
    """Result of comparing baseline vs post-improvement debate metrics."""

    baseline: DebateMetrics
    after: DebateMetrics
    improved: bool = False
    metrics_delta: dict[str, float] = field(default_factory=dict)
    recommendation: str = "review"  # "keep" | "revert" | "review"

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "after": self.after.to_dict(),
            "improved": self.improved,
            "metrics_delta": self.metrics_delta,
            "recommendation": self.recommendation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutcomeComparison:
        return cls(
            baseline=DebateMetrics.from_dict(data.get("baseline", {})),
            after=DebateMetrics.from_dict(data.get("after", {})),
            improved=data.get("improved", False),
            metrics_delta=data.get("metrics_delta", {}),
            recommendation=data.get("recommendation", "review"),
        )


@dataclass
class DebateScenario:
    """A lightweight test debate scenario for metric capture.

    Instead of running real API-backed debates, scenarios provide
    deterministic or simulated debate outcomes for measurement.
    """

    topic: str
    expected_rounds: int = 3
    agent_count: int = 3

    # Callable that returns simulated debate results.
    # Signature: async (topic, agent_count, expected_rounds) -> dict
    runner: Callable[..., Awaitable[dict[str, Any]]] | None = None


# A default set of lightweight scenarios for baseline/after capture.
DEFAULT_SCENARIOS: list[DebateScenario] = [
    DebateScenario(topic="Should we add rate limiting to the API?", expected_rounds=3),
    DebateScenario(topic="Is the current test coverage sufficient?", expected_rounds=2),
    DebateScenario(topic="Should we refactor the memory subsystem?", expected_rounds=4),
]


async def _default_scenario_runner(
    topic: str,
    agent_count: int,
    expected_rounds: int,
) -> dict[str, Any]:
    """Default no-op scenario runner that returns plausible simulated metrics.

    In production, this would be replaced with actual Arena.run() calls
    or a dedicated lightweight debate runner. For testing and offline use,
    it returns deterministic values based on the scenario parameters.
    """
    return {
        "consensus_reached": True,
        "rounds": expected_rounds,
        "tokens_used": expected_rounds * agent_count * 500,
        "brier_scores": [0.2, 0.25, 0.3][:agent_count],
    }


class NomicOutcomeTracker:
    """Tracks whether Nomic Loop improvements actually help debate quality.

    Runs a fixed set of test debate scenarios before and after a code change,
    then compares the metrics to detect regressions.

    Args:
        scenarios: List of debate scenarios to run. Defaults to DEFAULT_SCENARIOS.
        scenario_runner: Async callable to execute each scenario. If None, uses
            the default simulated runner.
        degradation_threshold: Maximum allowed relative degradation per metric
            before it counts as a regression. Default: 0.05 (5%).
        cycle_store: Optional CycleLearningStore for persisting outcomes.
    """

    def __init__(
        self,
        scenarios: list[DebateScenario] | None = None,
        scenario_runner: Callable[..., Awaitable[dict[str, Any]]] | None = None,
        degradation_threshold: float = DEGRADATION_THRESHOLD,
        cycle_store: Any | None = None,
    ):
        self.scenarios = list(DEFAULT_SCENARIOS) if scenarios is None else scenarios
        self._runner = scenario_runner or _default_scenario_runner
        self.degradation_threshold = degradation_threshold
        self._cycle_store = cycle_store

    async def capture_baseline(self) -> DebateMetrics:
        """Run test debates and capture pre-improvement metrics."""
        return await self._capture("baseline")

    async def capture_after(self) -> DebateMetrics:
        """Run test debates and capture post-improvement metrics."""
        return await self._capture("after")

    def compare(
        self,
        baseline: DebateMetrics,
        after: DebateMetrics,
    ) -> OutcomeComparison:
        """Compare baseline and after metrics to determine outcome.

        A metric is considered degraded if it worsens by more than
        ``degradation_threshold`` relative to the baseline value.

        For consensus_rate: higher is better (degradation = decrease).
        For avg_rounds: lower is better (degradation = increase).
        For avg_tokens: lower is better (degradation = increase).
        For calibration_spread: lower is better (degradation = increase).

        Returns:
            OutcomeComparison with improvement status and recommendation.
        """
        deltas: dict[str, float] = {}
        regressions: list[str] = []
        improvements: list[str] = []

        # consensus_rate: higher is better
        cr_delta = after.consensus_rate - baseline.consensus_rate
        deltas["consensus_rate"] = cr_delta
        if baseline.consensus_rate > 0 and cr_delta < 0:
            if abs(cr_delta) / baseline.consensus_rate > self.degradation_threshold:
                regressions.append("consensus_rate")
        if cr_delta > 0:
            improvements.append("consensus_rate")

        # avg_rounds: lower is better
        rounds_delta = after.avg_rounds - baseline.avg_rounds
        deltas["avg_rounds"] = rounds_delta
        if baseline.avg_rounds > 0 and rounds_delta > 0:
            if rounds_delta / baseline.avg_rounds > self.degradation_threshold:
                regressions.append("avg_rounds")
        if rounds_delta < 0:
            improvements.append("avg_rounds")

        # avg_tokens: lower is better
        tokens_delta = after.avg_tokens - baseline.avg_tokens
        deltas["avg_tokens"] = float(tokens_delta)
        if baseline.avg_tokens > 0 and tokens_delta > 0:
            if tokens_delta / baseline.avg_tokens > self.degradation_threshold:
                regressions.append("avg_tokens")
        if tokens_delta < 0:
            improvements.append("avg_tokens")

        # calibration_spread: lower is better
        cal_delta = after.calibration_spread - baseline.calibration_spread
        deltas["calibration_spread"] = cal_delta
        if baseline.calibration_spread > 0 and cal_delta > 0:
            if cal_delta / baseline.calibration_spread > self.degradation_threshold:
                regressions.append("calibration_spread")
        if cal_delta < 0:
            improvements.append("calibration_spread")

        # Determine recommendation
        if regressions:
            improved = False
            recommendation = "revert" if len(regressions) >= 2 else "review"
        elif improvements:
            improved = True
            recommendation = "keep"
        else:
            # No change
            improved = True
            recommendation = "keep"

        comparison = OutcomeComparison(
            baseline=baseline,
            after=after,
            improved=improved,
            metrics_delta=deltas,
            recommendation=recommendation,
        )

        logger.info(
            "outcome_comparison recommendation=%s improved=%s regressions=%s improvements=%s",
            recommendation,
            improved,
            regressions,
            improvements,
        )

        return comparison

    def record_cycle_outcome(
        self,
        cycle_id: str,
        comparison: OutcomeComparison,
    ) -> None:
        """Persist comparison results to the cycle store.

        If a cycle_store was provided at init time, updates the cycle record
        with the outcome comparison data. Otherwise, logs a warning and no-ops.

        Args:
            cycle_id: The Nomic cycle ID to update.
            comparison: The outcome comparison to record.
        """
        if self._cycle_store is None:
            try:
                from aragora.nomic.cycle_store import get_cycle_store

                self._cycle_store = get_cycle_store()
            except (ImportError, OSError) as e:
                logger.warning(
                    "outcome_record_skipped cycle_id=%s reason=no_cycle_store error=%s",
                    cycle_id,
                    e,
                )
                return

        record = self._cycle_store.load_cycle(cycle_id)
        if record is None:
            logger.warning(
                "outcome_record_skipped cycle_id=%s reason=cycle_not_found",
                cycle_id,
            )
            return

        # Store outcome data as a pattern reinforcement
        record.add_pattern_reinforcement(
            pattern_type="outcome_tracking",
            description=(
                f"Debate quality {'improved' if comparison.improved else 'degraded'}: "
                f"recommendation={comparison.recommendation}"
            ),
            success=comparison.improved,
            confidence=0.8 if comparison.improved else 0.3,
        )

        # Store detailed metrics in evidence_quality_scores
        for metric_name, delta_value in comparison.metrics_delta.items():
            record.evidence_quality_scores[f"outcome_{metric_name}_delta"] = delta_value

        self._cycle_store.save_cycle(record)

        logger.info(
            "outcome_recorded cycle_id=%s improved=%s recommendation=%s",
            cycle_id,
            comparison.improved,
            comparison.recommendation,
        )

    # --- Internal ---

    async def _capture(self, label: str) -> DebateMetrics:
        """Run all scenarios and aggregate metrics."""
        consensus_count = 0
        total_rounds = 0
        total_tokens = 0
        all_brier: list[float] = []

        for scenario in self.scenarios:
            runner = scenario.runner or self._runner
            try:
                result = await runner(
                    scenario.topic,
                    scenario.agent_count,
                    scenario.expected_rounds,
                )

                if result.get("consensus_reached", False):
                    consensus_count += 1

                total_rounds += result.get("rounds", scenario.expected_rounds)
                total_tokens += result.get("tokens_used", 0)

                brier_scores = result.get("brier_scores", [])
                all_brier.extend(brier_scores)

            except (RuntimeError, OSError, ValueError, TypeError) as e:
                logger.warning(
                    "scenario_failed label=%s topic=%s error=%s",
                    label,
                    scenario.topic[:50],
                    e,
                )

        n = len(self.scenarios)
        if n == 0:
            return DebateMetrics()

        # Calibration spread = std dev of Brier scores
        cal_spread = 0.0
        if all_brier:
            mean_brier = sum(all_brier) / len(all_brier)
            variance = sum((b - mean_brier) ** 2 for b in all_brier) / len(all_brier)
            cal_spread = variance ** 0.5

        metrics = DebateMetrics(
            consensus_rate=consensus_count / n,
            avg_rounds=total_rounds / n,
            avg_tokens=total_tokens // n,
            calibration_spread=cal_spread,
        )

        logger.info(
            "metrics_captured label=%s consensus_rate=%.2f avg_rounds=%.1f "
            "avg_tokens=%d calibration_spread=%.3f",
            label,
            metrics.consensus_rate,
            metrics.avg_rounds,
            metrics.avg_tokens,
            metrics.calibration_spread,
        )

        return metrics
