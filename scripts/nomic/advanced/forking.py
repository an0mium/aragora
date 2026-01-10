"""
Counterfactual branching for nomic loop debates.

Provides debate forking when disagreement is detected, allowing
parallel exploration of alternative paths before merging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ForkOutcome:
    """Record of a fork outcome for learning."""

    fork_point_id: str
    winning_branch_id: str
    branches_explored: int
    merge_confidence: float
    timestamp: str = ""


class ForkingRunner:
    """
    Manages debate forking for counterfactual exploration.

    When significant disagreement is detected, forks the debate into
    parallel branches to explore alternatives before merging.

    Usage:
        runner = ForkingRunner(
            enabled=True,
            available=DEBATE_FORKER_AVAILABLE,
            fork_detector_class=ForkDetector,
            log_fn=loop._log,
        )

        # Check if debate should fork
        decision = runner.check_should_fork(messages, round_num, agents)
        if decision and decision.should_fork:
            result = await runner.run_forked_debate(decision, base_context)
    """

    def __init__(
        self,
        enabled: bool = False,
        available: bool = False,
        fork_detector_class: Any = None,
        log_fn: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize forking runner.

        Args:
            enabled: Whether forking is enabled
            available: Whether DebateForker is available
            fork_detector_class: ForkDetector class
            log_fn: Optional logging function
        """
        self._enabled = enabled
        self._available = available
        self._fork_detector_class = fork_detector_class
        self._log = log_fn or (lambda msg: logger.info(msg))

        # Track outcomes for learning
        self._outcomes: list[ForkOutcome] = []

    @property
    def is_enabled(self) -> bool:
        """Whether forking is enabled and available."""
        return self._enabled and self._available and self._fork_detector_class is not None

    def check_should_fork(
        self,
        messages: list[Any],
        round_num: int,
        agents: list[Any],
    ) -> Optional[Any]:
        """
        Check if debate should fork.

        Args:
            messages: Debate messages so far
            round_num: Current round number
            agents: List of participating agents

        Returns:
            ForkDecision if fork should happen, None otherwise
        """
        if not self.is_enabled:
            return None

        try:
            detector = self._fork_detector_class()
            decision = detector.should_fork(messages, round_num, agents)

            if decision and hasattr(decision, "should_fork") and decision.should_fork:
                reason = getattr(decision, "reason", "unknown")
                self._log(f"  [forking] Fork triggered: {reason}")

            return decision

        except Exception as e:
            self._log(f"  [forking] Check error: {e}")
            return None

    async def run_forked_debate(
        self,
        fork_decision: Any,
        base_context: str,
    ) -> Optional[Any]:
        """
        Run forked parallel debates.

        Note: This feature requires proper Environment, agents, and run_debate_fn
        to be passed to run_branches. Currently disabled until full integration.

        Args:
            fork_decision: ForkDecision from check_should_fork
            base_context: Base context for the forked debates

        Returns:
            MergeResult if successful, None otherwise
        """
        if not self.is_enabled or not fork_decision:
            return None

        try:
            branches = getattr(fork_decision, "branches", [])
            if not branches:
                self._log("  [forking] No branches in fork decision")
                return None

            self._log(f"  [forking] Fork detected with {len(branches)} branches")

            # TODO: Full forking requires Environment, agents list, and run_debate_fn
            # For now, log the fork but don't execute parallel branches
            self._log("  [forking] Skipping parallel execution (integration pending)")
            return None

        except Exception as e:
            self._log(f"  [forking] Run error: {e}")
            return None

    def record_outcome(
        self,
        fork_point: Any,
        merge_result: Any,
    ) -> None:
        """
        Record fork outcome for learning.

        Args:
            fork_point: ForkPoint from the fork
            merge_result: MergeResult from merging branches
        """
        if not fork_point or not merge_result:
            return

        try:
            from datetime import datetime

            outcome = ForkOutcome(
                fork_point_id=getattr(fork_point, "id", "unknown"),
                winning_branch_id=getattr(merge_result, "winning_branch_id", "unknown"),
                branches_explored=len(getattr(fork_point, "branches", [])),
                merge_confidence=getattr(merge_result, "confidence", 0.0),
                timestamp=datetime.now().isoformat(),
            )
            self._outcomes.append(outcome)
            self._log(f"  [forking] Recorded outcome: {outcome.winning_branch_id}")

        except Exception as e:
            self._log(f"  [forking] Record error: {e}")

    def get_outcomes(self) -> list[ForkOutcome]:
        """Get all recorded fork outcomes."""
        return self._outcomes.copy()


__all__ = [
    "ForkOutcome",
    "ForkingRunner",
]
