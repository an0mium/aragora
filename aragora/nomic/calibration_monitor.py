"""Cross-cycle calibration drift detection.

Monitors calibration scores across Nomic Loop cycles and detects
three types of drift: stagnation, regression, and confidence inflation.

Usage:
    from aragora.nomic.calibration_monitor import CalibrationDriftDetector

    detector = CalibrationDriftDetector(window_size=10)
    detector.record_cycle("cycle_1", {"claude": 0.8, "gemini": 0.7})
    detector.record_cycle("cycle_2", {"claude": 0.79, "gemini": 0.68})
    warnings = detector.detect_drift()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class DriftWarning:
    """A calibration drift warning for an agent."""

    type: str  # "stagnation", "regression", "inflation"
    agent_name: str
    score_history: list[float]
    severity: str  # "low", "medium", "high"
    message: str


@dataclass
class CalibrationDriftDetector:
    """Detect calibration drift across Nomic Loop cycles.

    Tracks calibration scores per agent over a sliding window and
    identifies problematic patterns: stagnation, regression, and
    confidence inflation.

    Args:
        window_size: Number of recent cycles to consider.
        stagnation_threshold: Variance below which scores are considered stagnant.
        regression_threshold: Minimum total decrease to trigger regression warning.
    """

    window_size: int = 10
    stagnation_threshold: float = 0.01
    regression_threshold: float = 0.05
    _cycles: list[tuple[str, dict[str, float]]] = field(default_factory=list, repr=False)

    def record_cycle(self, cycle_id: str, scores: dict[str, float]) -> None:
        """Record calibration scores for a cycle.

        Args:
            cycle_id: Unique identifier for the cycle.
            scores: Mapping of agent_name to calibration score (0-1).
        """
        self._cycles.append((cycle_id, dict(scores)))
        # Trim to window
        if len(self._cycles) > self.window_size:
            self._cycles = self._cycles[-self.window_size :]
        logger.debug(
            "calibration_monitor_recorded cycle=%s agents=%d total_cycles=%d",
            cycle_id,
            len(scores),
            len(self._cycles),
        )

    def detect_drift(self) -> list[DriftWarning]:
        """Analyze recent cycles for calibration drift issues.

        Returns:
            List of DriftWarning objects for detected issues.
        """
        warnings: list[DriftWarning] = []

        if len(self._cycles) < 2:
            return warnings

        # Collect per-agent score histories
        agent_histories = self._build_agent_histories()

        for agent_name, history in agent_histories.items():
            if len(history) < 2:
                continue

            # Check stagnation
            stagnation = self._check_stagnation(agent_name, history)
            if stagnation:
                warnings.append(stagnation)

            # Check regression
            regression = self._check_regression(agent_name, history)
            if regression:
                warnings.append(regression)

            # Check inflation
            inflation = self._check_inflation(agent_name, history)
            if inflation:
                warnings.append(inflation)

        if warnings:
            logger.info(
                "calibration_drift_detected warnings=%d agents=%s",
                len(warnings),
                list({w.agent_name for w in warnings}),
            )
            self._emit_warnings(warnings)
            self._persist_to_km(warnings)

        return warnings

    def _build_agent_histories(self) -> dict[str, list[float]]:
        """Build per-agent score histories from recorded cycles."""
        histories: dict[str, list[float]] = {}
        for _, scores in self._cycles:
            for agent, score in scores.items():
                histories.setdefault(agent, []).append(score)
        return histories

    def _check_stagnation(self, agent_name: str, history: list[float]) -> DriftWarning | None:
        """Detect score stagnation: low variance over the window."""
        if len(history) < 3:
            return None

        mean = sum(history) / len(history)
        variance = sum((s - mean) ** 2 for s in history) / len(history)

        if variance < self.stagnation_threshold:
            severity = "high" if variance < self.stagnation_threshold / 2 else "medium"
            return DriftWarning(
                type="stagnation",
                agent_name=agent_name,
                score_history=history,
                severity=severity,
                message=(
                    f"Agent {agent_name} calibration stagnant "
                    f"(variance={variance:.4f} < {self.stagnation_threshold}) "
                    f"over {len(history)} cycles"
                ),
            )
        return None

    def _check_regression(self, agent_name: str, history: list[float]) -> DriftWarning | None:
        """Detect monotonically decreasing scores over 3+ cycles."""
        if len(history) < 3:
            return None

        # Check last 3+ entries for monotonic decrease
        tail = history[-min(len(history), self.window_size) :]
        decreasing_run = 0
        for i in range(1, len(tail)):
            if tail[i] < tail[i - 1]:
                decreasing_run += 1
            else:
                decreasing_run = 0

        total_drop = tail[0] - tail[-1] if tail else 0

        if decreasing_run >= 2 and total_drop >= self.regression_threshold:
            severity = "high" if total_drop >= self.regression_threshold * 2 else "medium"
            return DriftWarning(
                type="regression",
                agent_name=agent_name,
                score_history=history,
                severity=severity,
                message=(
                    f"Agent {agent_name} calibration regressing "
                    f"(dropped {total_drop:.3f} over {decreasing_run + 1} cycles)"
                ),
            )
        return None

    def _check_inflation(self, agent_name: str, history: list[float]) -> DriftWarning | None:
        """Detect confidence inflation: consistently high scores (> 0.95)."""
        if len(history) < 3:
            return None

        recent = history[-min(len(history), self.window_size) :]
        inflated_count = sum(1 for s in recent if s > 0.95)
        inflated_ratio = inflated_count / len(recent)

        if inflated_ratio >= 0.7:
            severity = "high" if inflated_ratio >= 0.9 else "medium"
            return DriftWarning(
                type="inflation",
                agent_name=agent_name,
                score_history=history,
                severity=severity,
                message=(
                    f"Agent {agent_name} shows confidence inflation "
                    f"({inflated_count}/{len(recent)} scores > 0.95)"
                ),
            )
        return None

    def _emit_warnings(self, warnings: list[DriftWarning]) -> None:
        """Emit drift warnings via streaming interface if available."""
        try:
            from aragora.spectate.stream import SpectatorStream

            stream = SpectatorStream()
            for warning in warnings:
                stream.emit(
                    event="calibration_drift",
                    details={
                        "type": warning.type,
                        "agent": warning.agent_name,
                        "severity": warning.severity,
                        "message": warning.message,
                    },
                )
        except ImportError:
            pass
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.debug("Failed to emit drift warning: %s", e)

    def _persist_to_km(self, warnings: list[DriftWarning]) -> None:
        """Store drift metrics in KM via lightweight ingestion."""
        try:
            from aragora.knowledge.mound.core import KnowledgeItem
            from aragora.knowledge.mound.adapters.receipt_adapter import ReceiptAdapter

            adapter = ReceiptAdapter()
            import asyncio

            async def _ingest() -> None:
                for warning in warnings:
                    try:
                        item = KnowledgeItem(
                            content=warning.message,
                            source="calibration_drift_detector",
                            tags=[
                                "calibration_drift",
                                warning.type,
                                warning.agent_name,
                                warning.severity,
                            ],
                        )
                        await adapter.ingest(item)
                    except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
                        logger.debug("KM drift ingestion failed: %s", exc)

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_ingest())
            except RuntimeError:
                pass

        except ImportError:
            logger.debug("KM not available for drift persistence")
        except (RuntimeError, ValueError, TypeError, AttributeError) as e:
            logger.debug("Drift KM persistence failed: %s", e)


__all__ = [
    "CalibrationDriftDetector",
    "DriftWarning",
]
