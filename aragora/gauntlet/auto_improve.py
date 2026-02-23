"""Gauntlet Auto-Improve: Close the security-to-improvement feedback loop.

Automatically triggers the Nomic Loop when Gauntlet security tests complete
by converting high/critical findings into self-improvement goals.

Flow:
    Gauntlet run completes
    -> GauntletAutoImprove.on_run_complete(result)
    -> improvement_bridge.findings_to_goals() converts findings
    -> Filters to high/critical severity only
    -> Rate-limits to max N goals per run
    -> Pushes to ImprovementQueue for the next Nomic Loop cycle

Usage:
    from aragora.gauntlet.auto_improve import GauntletAutoImprove

    auto = GauntletAutoImprove(enabled=True, max_goals_per_run=5)
    queued = auto.on_run_complete(gauntlet_result)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_MAX_GOALS_PER_RUN = 5
DEFAULT_MIN_SEVERITY = "high"  # Only high and critical
DEFAULT_COOLDOWN_SECONDS = 60.0  # Min time between auto-improve triggers


@dataclass
class AutoImproveResult:
    """Result of an auto-improve trigger."""

    goals_queued: int = 0
    goals_skipped_severity: int = 0
    goals_skipped_duplicate: int = 0
    goals_skipped_rate_limit: int = 0
    gauntlet_id: str = ""
    error: str | None = None
    goal_descriptions: list[str] = field(default_factory=list)

    @property
    def total_findings_processed(self) -> int:
        return (
            self.goals_queued
            + self.goals_skipped_severity
            + self.goals_skipped_duplicate
            + self.goals_skipped_rate_limit
        )


class GauntletAutoImprove:
    """Automatically bridges Gauntlet findings to the Nomic Loop improvement queue.

    This class listens for gauntlet run completions and pushes high-severity
    findings as improvement goals, closing the security-to-improvement feedback
    loop.

    Args:
        enabled: Whether auto-improvement is active. Defaults to False (opt-in).
        max_goals_per_run: Maximum goals to queue per gauntlet run.
        min_severity: Minimum severity level to include ("critical" or "high").
        cooldown_seconds: Minimum seconds between auto-improve triggers.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        max_goals_per_run: int = DEFAULT_MAX_GOALS_PER_RUN,
        min_severity: str = DEFAULT_MIN_SEVERITY,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
    ):
        self.enabled = enabled
        self.max_goals_per_run = max_goals_per_run
        self.min_severity = min_severity
        self.cooldown_seconds = cooldown_seconds

        # Track previously queued finding IDs to avoid duplicates
        self._queued_finding_ids: set[str] = set()

        # Rate limiting: track last trigger time
        self._last_trigger_time: float = 0.0

    def on_run_complete(self, gauntlet_result: Any) -> AutoImproveResult:
        """Handle a completed gauntlet run.

        This is the main entry point, called after a GauntletRunner.run()
        completes. It converts findings into improvement goals and pushes
        them to the ImprovementQueue.

        Args:
            gauntlet_result: The GauntletResult from the completed run.

        Returns:
            AutoImproveResult with details about what was queued.
        """
        result = AutoImproveResult()

        if not self.enabled:
            return result

        # Extract gauntlet ID for logging
        gauntlet_id = getattr(gauntlet_result, "gauntlet_id", "unknown")
        result.gauntlet_id = gauntlet_id

        # Rate limit check
        now = time.monotonic()
        elapsed = now - self._last_trigger_time
        if self._last_trigger_time > 0 and elapsed < self.cooldown_seconds:
            logger.debug(
                "auto_improve_cooldown gauntlet=%s elapsed=%.1fs cooldown=%.1fs",
                gauntlet_id,
                elapsed,
                self.cooldown_seconds,
            )
            result.error = "cooldown_active"
            return result

        self._last_trigger_time = now

        # Convert findings to goals using the existing bridge
        try:
            from aragora.gauntlet.improvement_bridge import findings_to_goals
        except ImportError:
            logger.warning("auto_improve_bridge_unavailable gauntlet=%s", gauntlet_id)
            result.error = "improvement_bridge_import_failed"
            return result

        try:
            all_goals = findings_to_goals(
                gauntlet_result,
                max_goals=self.max_goals_per_run * 2,  # Over-fetch to allow filtering
                min_severity=self.min_severity,
            )
        except (RuntimeError, ValueError, TypeError, AttributeError) as exc:
            logger.warning(
                "auto_improve_bridge_error gauntlet=%s error=%s",
                gauntlet_id,
                type(exc).__name__,
            )
            result.error = f"findings_to_goals_failed: {type(exc).__name__}"
            return result

        if not all_goals:
            logger.debug("auto_improve_no_goals gauntlet=%s", gauntlet_id)
            return result

        # Filter out already-queued findings (deduplication)
        filtered_goals = []
        for goal in all_goals:
            if goal.source_finding_id in self._queued_finding_ids:
                result.goals_skipped_duplicate += 1
            else:
                filtered_goals.append(goal)

        # Apply rate limit (max goals per run)
        goals_to_queue = filtered_goals[: self.max_goals_per_run]
        result.goals_skipped_rate_limit = max(
            0, len(filtered_goals) - self.max_goals_per_run
        )

        if not goals_to_queue:
            logger.debug(
                "auto_improve_all_filtered gauntlet=%s duplicates=%d",
                gauntlet_id,
                result.goals_skipped_duplicate,
            )
            return result

        # Push goals to the ImprovementQueue
        try:
            from aragora.nomic.feedback_orchestrator import (
                ImprovementGoal,
                ImprovementQueue,
            )

            queue = ImprovementQueue()

            for goal in goals_to_queue:
                # Map bridge priority (1=highest int) to queue priority (1.0=highest float)
                float_priority = max(0.0, min(1.0, 1.0 - (goal.priority - 1) * 0.2))

                queue.push(
                    ImprovementGoal(
                        goal=goal.description,
                        source="gauntlet_auto_improve",
                        priority=float_priority,
                        context={
                            "gauntlet_id": gauntlet_id,
                            "finding_id": goal.source_finding_id,
                            "severity": goal.severity,
                            "category": goal.category,
                            "track": goal.track,
                            "file_hints": goal.file_hints,
                        },
                    )
                )

                # Track as queued
                self._queued_finding_ids.add(goal.source_finding_id)
                result.goals_queued += 1
                result.goal_descriptions.append(goal.description)

        except ImportError:
            logger.warning(
                "auto_improve_queue_unavailable gauntlet=%s", gauntlet_id
            )
            result.error = "improvement_queue_import_failed"
            return result
        except (RuntimeError, ValueError, OSError) as exc:
            logger.warning(
                "auto_improve_queue_error gauntlet=%s error=%s",
                gauntlet_id,
                type(exc).__name__,
            )
            result.error = f"queue_push_failed: {type(exc).__name__}"
            return result

        logger.info(
            "auto_improve_complete gauntlet=%s queued=%d skipped_dup=%d "
            "skipped_rate=%d severity_filter=%s",
            gauntlet_id,
            result.goals_queued,
            result.goals_skipped_duplicate,
            result.goals_skipped_rate_limit,
            self.min_severity,
        )

        return result

    def reset(self) -> None:
        """Reset internal state (for testing)."""
        self._queued_finding_ids.clear()
        self._last_trigger_time = 0.0


__all__ = [
    "AutoImproveResult",
    "GauntletAutoImprove",
]
