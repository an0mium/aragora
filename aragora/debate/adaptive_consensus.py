"""
Adaptive Consensus Thresholds based on voter calibration quality.

When all voters are well-calibrated (low Brier scores), a lower consensus
threshold is safe because votes are trustworthy. When voters are poorly
calibrated, a higher threshold compensates for unreliable voting.

The existing consensus phase in ``consensus_phase.py`` uses a static
``consensus_threshold`` (default 0.6) from ``DebateProtocol``. Calibration-
weighted *voting* already exists (agents with better calibration get higher
vote weights via ``WeightCalculator``), but the *threshold itself* does not
adapt. This module closes that gap.

Formula:
    threshold = base + calibration_impact * (avg_brier - NEUTRAL_BRIER)

where ``NEUTRAL_BRIER`` (0.25) represents a Brier score that neither raises
nor lowers the threshold. The result is clamped to
``[min_threshold, max_threshold]``.
"""

from __future__ import annotations

__all__ = [
    "AdaptiveConsensusConfig",
    "AdaptiveConsensus",
]

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# A Brier score of 0.25 is the "break-even" point: it corresponds to
# a uniform-random binary predictor (always predicting 0.5). Scores below
# this indicate skill; scores above indicate anti-skill.
NEUTRAL_BRIER: float = 0.25


@dataclass
class AdaptiveConsensusConfig:
    """Configuration for adaptive consensus thresholds.

    Attributes:
        base_threshold: Default consensus threshold (mirrors DebateProtocol default).
        min_threshold: Floor -- the threshold will never drop below this.
        max_threshold: Ceiling -- the threshold will never rise above this.
        calibration_impact: Scaling factor that controls how strongly
            calibration quality shifts the threshold.
        min_calibration_samples: Minimum number of calibration predictions
            an agent must have before its Brier score is included in the
            pool average.  Agents with fewer samples are ignored.
    """

    base_threshold: float = 0.6
    min_threshold: float = 0.45
    max_threshold: float = 0.85
    calibration_impact: float = 0.3
    min_calibration_samples: int = 5


@dataclass
class _AgentBrierResult:
    """Internal: Brier score retrieval result for a single agent."""

    agent_name: str
    brier_score: float
    sample_count: int
    source: str  # "calibration_tracker" or "elo_system"


class AdaptiveConsensus:
    """Dynamically adjusts consensus thresholds based on voter calibration quality.

    This class examines the calibration track record of the agents in the
    voter pool and computes an adjusted consensus threshold:

    * **Well-calibrated pool** (low average Brier) -> lower threshold,
      because votes are more trustworthy.
    * **Poorly-calibrated pool** (high average Brier) -> higher threshold,
      requiring stronger agreement to compensate for noise.

    Usage::

        adaptive = AdaptiveConsensus(config)
        threshold = adaptive.compute_threshold(
            agents, elo_system=elo, calibration_tracker=tracker,
        )
    """

    def __init__(self, config: AdaptiveConsensusConfig | None = None) -> None:
        self.config = config or AdaptiveConsensusConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_threshold(
        self,
        agents: list[Any],
        elo_system: Any | None = None,
        calibration_tracker: Any | None = None,
    ) -> float:
        """Compute an adaptive consensus threshold for this voter pool.

        Args:
            agents: List of agent objects (must have a ``name`` attribute).
            elo_system: Optional ``EloSystem`` instance used as fallback when
                ``calibration_tracker`` is unavailable.
            calibration_tracker: Optional ``CalibrationTracker`` instance
                (preferred source of Brier scores).

        Returns:
            The adjusted consensus threshold, clamped to
            ``[config.min_threshold, config.max_threshold]``.
        """
        brier_results = self._collect_brier_scores(
            agents, elo_system, calibration_tracker
        )

        if not brier_results:
            logger.debug(
                "adaptive_consensus_no_calibration_data agents=%d "
                "returning_base_threshold=%.3f",
                len(agents),
                self.config.base_threshold,
            )
            return self.config.base_threshold

        avg_brier = sum(r.brier_score for r in brier_results) / len(brier_results)
        raw_threshold = (
            self.config.base_threshold
            + self.config.calibration_impact * (avg_brier - NEUTRAL_BRIER)
        )
        threshold = max(
            self.config.min_threshold,
            min(self.config.max_threshold, raw_threshold),
        )

        logger.info(
            "adaptive_consensus_threshold avg_brier=%.4f agents_with_data=%d/%d "
            "raw=%.4f clamped=%.4f",
            avg_brier,
            len(brier_results),
            len(agents),
            raw_threshold,
            threshold,
        )

        return threshold

    def compute_threshold_with_explanation(
        self,
        agents: list[Any],
        elo_system: Any | None = None,
        calibration_tracker: Any | None = None,
    ) -> tuple[float, str]:
        """Compute threshold and return an audit-friendly explanation.

        Returns:
            A ``(threshold, explanation)`` tuple. The explanation string
            contains calibration details suitable for inclusion in
            decision receipts / audit trails.
        """
        brier_results = self._collect_brier_scores(
            agents, elo_system, calibration_tracker
        )

        if not brier_results:
            explanation = (
                f"No calibration data available for {len(agents)} agent(s). "
                f"Using base threshold {self.config.base_threshold:.2f}."
            )
            return self.config.base_threshold, explanation

        avg_brier = sum(r.brier_score for r in brier_results) / len(brier_results)
        raw_threshold = (
            self.config.base_threshold
            + self.config.calibration_impact * (avg_brier - NEUTRAL_BRIER)
        )
        threshold = max(
            self.config.min_threshold,
            min(self.config.max_threshold, raw_threshold),
        )

        # Build per-agent detail lines
        agent_lines: list[str] = []
        for r in brier_results:
            agent_lines.append(
                f"  - {r.agent_name}: brier={r.brier_score:.4f}, "
                f"samples={r.sample_count}, source={r.source}"
            )

        was_clamped = raw_threshold != threshold
        clamp_note = ""
        if was_clamped:
            clamp_note = (
                f" (clamped from {raw_threshold:.4f} to "
                f"[{self.config.min_threshold:.2f}, {self.config.max_threshold:.2f}])"
            )

        explanation = (
            f"Adaptive consensus threshold: {threshold:.4f}{clamp_note}. "
            f"Average Brier score: {avg_brier:.4f} across "
            f"{len(brier_results)}/{len(agents)} agent(s) with sufficient data. "
            f"Formula: {self.config.base_threshold:.2f} + "
            f"{self.config.calibration_impact:.2f} * "
            f"({avg_brier:.4f} - {NEUTRAL_BRIER}). "
            f"Per-agent calibration:\n" + "\n".join(agent_lines)
        )

        return threshold, explanation

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_brier_scores(
        self,
        agents: list[Any],
        elo_system: Any | None,
        calibration_tracker: Any | None,
    ) -> list[_AgentBrierResult]:
        """Gather per-agent Brier scores from available data sources.

        Prefers ``calibration_tracker`` (richer data) over ``elo_system``.
        Agents with fewer than ``config.min_calibration_samples`` predictions
        are excluded.
        """
        results: list[_AgentBrierResult] = []

        for agent in agents:
            agent_name = getattr(agent, "name", str(agent))
            result = self._get_agent_brier(
                agent_name, elo_system, calibration_tracker
            )
            if result is not None:
                results.append(result)

        return results

    def _get_agent_brier(
        self,
        agent_name: str,
        elo_system: Any | None,
        calibration_tracker: Any | None,
    ) -> _AgentBrierResult | None:
        """Retrieve Brier score for a single agent.

        Tries ``calibration_tracker`` first, then falls back to
        ``elo_system.get_rating()``.
        """
        # --- Try CalibrationTracker first ---
        if calibration_tracker is not None:
            try:
                summary = calibration_tracker.get_calibration_summary(agent_name)
                if summary.total_predictions >= self.config.min_calibration_samples:
                    return _AgentBrierResult(
                        agent_name=agent_name,
                        brier_score=summary.brier_score,
                        sample_count=summary.total_predictions,
                        source="calibration_tracker",
                    )
                logger.debug(
                    "adaptive_consensus_skip_agent agent=%s "
                    "samples=%d min_required=%d source=calibration_tracker",
                    agent_name,
                    summary.total_predictions,
                    self.config.min_calibration_samples,
                )
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                logger.debug(
                    "adaptive_consensus_calibration_error agent=%s error=%s",
                    agent_name,
                    exc,
                )

        # --- Fallback to ELO system ---
        if elo_system is not None:
            try:
                rating = elo_system.get_rating(agent_name)
                if rating.calibration_total >= self.config.min_calibration_samples:
                    return _AgentBrierResult(
                        agent_name=agent_name,
                        brier_score=rating.calibration_brier_score,
                        sample_count=rating.calibration_total,
                        source="elo_system",
                    )
                logger.debug(
                    "adaptive_consensus_skip_agent agent=%s "
                    "samples=%d min_required=%d source=elo_system",
                    agent_name,
                    rating.calibration_total,
                    self.config.min_calibration_samples,
                )
            except (ValueError, KeyError, TypeError, AttributeError) as exc:
                logger.debug(
                    "adaptive_consensus_elo_error agent=%s error=%s",
                    agent_name,
                    exc,
                )

        return None
