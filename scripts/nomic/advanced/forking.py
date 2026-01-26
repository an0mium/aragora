"""
Counterfactual branching for nomic loop debates.

Provides debate forking when disagreement is detected, allowing
parallel exploration of alternative paths before merging.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Optional, Callable, TYPE_CHECKING

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
        *,
        env: Optional[Any] = None,
        agents: Optional[list[Any]] = None,
        run_debate_fn: Optional[Callable[..., Awaitable[Any]]] = None,
    ) -> Optional[Any]:
        """
        Run forked parallel debates.

        This method requires all parameters to execute forked debates. If required
        parameters are missing, it raises NotImplementedError with guidance.

        Args:
            fork_decision: ForkDecision from check_should_fork
            base_context: Base context for the forked debates
            env: Environment for debates (required for execution)
            agents: List of agents (required for execution)
            run_debate_fn: Async function to run each debate branch (required)

        Returns:
            MergeResult if successful, None otherwise

        Raises:
            NotImplementedError: If required parameters not provided
        """
        if not self.is_enabled or not fork_decision:
            return None

        try:
            branches = getattr(fork_decision, "branches", [])
            if not branches:
                self._log("  [forking] No branches in fork decision")
                return None

            self._log(f"  [forking] Fork detected with {len(branches)} branches")

            # Validate required parameters for execution
            if env is None or agents is None or run_debate_fn is None:
                missing = []
                if env is None:
                    missing.append("env")
                if agents is None:
                    missing.append("agents")
                if run_debate_fn is None:
                    missing.append("run_debate_fn")

                self._log(
                    f"  [forking] Missing required parameters: {', '.join(missing)}. "
                    "Use DebateForker directly for full functionality."
                )
                raise NotImplementedError(
                    f"ForkingRunner.run_forked_debate requires {', '.join(missing)}. "
                    "Use aragora.debate.forking.DebateForker.run_branches() directly "
                    "for full forking functionality."
                )

            # Full implementation when all parameters provided
            try:
                from aragora.debate.forking import DebateForker

                forker = DebateForker()
                fork_branches = forker.fork(
                    parent_debate_id=base_context[:8] if base_context else "fork",
                    fork_round=0,
                    messages_so_far=[],
                    decision=fork_decision,
                )

                completed = await forker.run_branches(
                    branches=fork_branches,
                    env=env,
                    agents=agents,
                    run_debate_fn=run_debate_fn,
                )

                if completed:
                    result = forker.merge(completed)
                    self._log(f"  [forking] Merged {len(completed)} branches")
                    return result
                return None

            except ImportError:
                self._log("  [forking] DebateForker not available")
                raise NotImplementedError(
                    "DebateForker module not available. Ensure aragora.debate.forking is installed."
                )

        except NotImplementedError:
            raise  # Re-raise NotImplementedError as-is
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
