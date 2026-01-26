"""
Deadlock detection and resolution for the nomic loop.

Detects various types of cycle stalls:
- Repeated failures: Same error 3+ times in a row
- Oscillating patterns: A-B-A-B alternating failures
- Phase stalls: No progress within a single phase

Resolution strategies:
- Clear cached state for fresh attempts
- Counterfactual resolution via belief network
- Adaptive consensus threshold decay
- Force judge consensus mode
- Skip after max retries
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class DeadlockState:
    """State container for deadlock tracking."""

    cycle_history: list[dict] = field(default_factory=list)
    max_history: int = 10
    phase_progress: dict[str, list[dict]] = field(default_factory=dict)
    deadlock_count: int = 0
    consensus_threshold_decay: int = 0
    force_judge_consensus: bool = False
    cached_cruxes: list = field(default_factory=list)

    def reset_for_cycle(self) -> None:
        """Reset per-cycle state."""
        self.phase_progress = {}
        self.force_judge_consensus = False


@dataclass
class DeadlockResolution:
    """Result of handling a deadlock."""

    action: str  # "continue", "retry_with_reset", "force_judge", "skip"
    message: str = ""
    cleared_state: bool = False
    threshold_lowered: bool = False
    new_threshold: Optional[float] = None


class DeadlockManager:
    """
    Detects and resolves deadlock patterns in the nomic loop.

    Tracks cycle outcomes and phase progress to identify when the loop
    is stuck in unproductive patterns.

    Usage:
        manager = DeadlockManager(log_fn=loop._log)

        # Record each cycle outcome
        manager.record_outcome(cycle_num, "design_no_consensus", {"votes": 2})

        # Check for deadlocks
        deadlock = manager.detect_deadlock()
        if deadlock:
            resolution = await manager.handle_deadlock(
                deadlock,
                belief_network=nomic_integration._belief_network
            )
            if resolution.action == "skip":
                continue
    """

    # Consensus decay levels
    THRESHOLD_DECAY = [0.6, 0.5, 0.4]  # 60% -> 50% -> 40%

    # Max deadlocks before skip
    MAX_DEADLOCK_RETRIES = 3

    def __init__(
        self,
        max_history: int = 10,
        log_fn: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize deadlock manager.

        Args:
            max_history: Maximum cycle history to retain
            log_fn: Optional logging function (defaults to logger.info)
        """
        self.state = DeadlockState(max_history=max_history)
        self._log = log_fn or (lambda msg: logger.info(msg))

    def record_outcome(
        self,
        cycle: int,
        outcome: str,
        details: Optional[dict] = None,
    ) -> None:
        """
        Track cycle outcome for deadlock detection.

        Args:
            cycle: Current cycle number
            outcome: Outcome type (e.g., "success", "design_no_consensus")
            details: Optional additional details
        """
        self.state.cycle_history.append(
            {
                "cycle": cycle,
                "outcome": outcome,
                "timestamp": datetime.now().isoformat(),
                "details": details or {},
            }
        )

        # Trim history
        if len(self.state.cycle_history) > self.state.max_history:
            self.state.cycle_history.pop(0)

    def detect_deadlock(self) -> str:
        """
        Detect if we're stuck in a cycle pattern.

        Returns:
            Deadlock type description, or empty string if none detected
        """
        history = self.state.cycle_history

        if len(history) < 3:
            return ""

        # Check for repeated same outcome (e.g., design_no_consensus 3 times)
        recent = [h["outcome"] for h in history[-3:]]
        if len(set(recent)) == 1 and recent[0] != "success":
            return f"Repeated failure: {recent[0]} for 3 cycles"

        # Check for oscillating pattern (A-B-A-B)
        if len(history) >= 4:
            last4 = [h["outcome"] for h in history[-4:]]
            if last4[0] == last4[2] and last4[1] == last4[3] and last4[0] != last4[1]:
                return f"Oscillating pattern: {last4[0]} <-> {last4[1]}"

        return ""

    def track_phase_progress(
        self,
        cycle: int,
        phase: str,
        round_num: int,
        consensus: float,
        changed: bool,
    ) -> bool:
        """
        Track progress within a phase to detect stalls.

        Args:
            cycle: Current cycle number
            phase: Phase name (e.g., "debate", "design")
            round_num: Round within the phase
            consensus: Current consensus level (0-1)
            changed: Whether any positions changed this round

        Returns:
            True if phase is stalled, False otherwise
        """
        key = f"{cycle}_{phase}"
        if key not in self.state.phase_progress:
            self.state.phase_progress[key] = []

        self.state.phase_progress[key].append(
            {
                "round": round_num,
                "consensus": consensus,
                "changed": changed,
                "timestamp": datetime.now(),
            }
        )

        # Detect stall: 3+ rounds with <5% consensus change and no position changes
        history = self.state.phase_progress[key]
        if len(history) >= 3:
            recent = history[-3:]
            consensus_change = abs(recent[-1]["consensus"] - recent[0]["consensus"])
            any_changed = any(r["changed"] for r in recent)

            if consensus_change < 0.05 and not any_changed:
                self._log(
                    f"  [STALL] {phase} stuck for 3 rounds "
                    f"(consensus: {recent[-1]['consensus']:.0%})"
                )
                return True

        return False

    async def handle_deadlock(
        self,
        deadlock_type: str,
        belief_network: Any = None,
    ) -> DeadlockResolution:
        """
        Handle detected deadlock with appropriate action.

        Args:
            deadlock_type: Description of the deadlock
            belief_network: Optional belief network for counterfactual resolution

        Returns:
            DeadlockResolution with action to take
        """
        self._log(f"  [DEADLOCK] Detected: {deadlock_type}")

        resolution = DeadlockResolution(action="continue")

        if "Repeated failure" in deadlock_type:
            # Clear cached state
            self.state.cached_cruxes = []
            self.state.phase_progress = {}
            resolution.cleared_state = True
            self._log("  [DEADLOCK] Cleared cached state for fresh attempt")

            # Try counterfactual resolution via belief network
            if belief_network is not None:
                try:
                    self._log(
                        "  [DEADLOCK] Attempting counterfactual resolution via belief network..."
                    )
                    contested = belief_network.get_contested_claims()
                    if contested:
                        self._log(
                            f"  [DEADLOCK] Found {len(contested)} contested claims for resolution"
                        )
                        self.state.cached_cruxes = contested
                except Exception as e:
                    self._log(f"  [DEADLOCK] Counterfactual resolution failed: {e}")

            # Rotate agent roles after multiple deadlocks
            if self.state.deadlock_count >= 2:
                self._log("  [DEADLOCK] Will rotate agent roles for fresh perspective")
                self.state.force_judge_consensus = True

            # Lower consensus threshold
            if self.state.consensus_threshold_decay < len(self.THRESHOLD_DECAY) - 1:
                self.state.consensus_threshold_decay += 1
                new_threshold = self.get_adaptive_threshold()
                self._log(f"  [DEADLOCK] Lowered consensus threshold to {new_threshold:.0%}")
                resolution.threshold_lowered = True
                resolution.new_threshold = new_threshold

            self.state.deadlock_count += 1
            resolution.action = "retry_with_reset"
            resolution.message = "Cleared state and adjusted thresholds"

        elif "Oscillating" in deadlock_type:
            # Force judge consensus to break oscillation
            self._log("  [DEADLOCK] Forcing judge consensus mode to break oscillation")
            self.state.force_judge_consensus = True
            resolution.action = "force_judge"
            resolution.message = "Forcing judge to break tie"

        elif self.state.deadlock_count >= self.MAX_DEADLOCK_RETRIES:
            # After max retries, skip
            self._log(
                f"  [DEADLOCK] Max retries ({self.MAX_DEADLOCK_RETRIES}) reached, "
                "skipping this improvement"
            )
            resolution.action = "skip"
            resolution.message = "Max retries exceeded"

        return resolution

    def get_adaptive_threshold(self) -> float:
        """
        Get consensus threshold adjusted for repeated failures.

        Decay path: 0.6 -> 0.5 -> 0.4 (after consecutive no-consensus cycles)
        This allows the system to break deadlocks by accepting lower agreement.

        Returns:
            Adjusted consensus threshold
        """
        idx = min(
            self.state.consensus_threshold_decay,
            len(self.THRESHOLD_DECAY) - 1,
        )
        threshold = self.THRESHOLD_DECAY[idx]

        if threshold < self.THRESHOLD_DECAY[0]:
            self._log(
                f"  [consensus] Using adaptive threshold: {threshold:.0%} "
                f"(decay level {self.state.consensus_threshold_decay})"
            )

        return threshold

    def reset_for_cycle(self) -> None:
        """Reset per-cycle state at the start of each cycle."""
        self.state.reset_for_cycle()

    def reset_all(self) -> None:
        """Fully reset deadlock tracking (e.g., after successful cycle)."""
        self.state = DeadlockState(max_history=self.state.max_history)

    @property
    def should_force_judge(self) -> bool:
        """Whether judge should be forced to break ties."""
        return self.state.force_judge_consensus

    @property
    def cached_cruxes(self) -> list:
        """Contested claims cached for next debate."""
        return self.state.cached_cruxes

    @cached_cruxes.setter
    def cached_cruxes(self, value: list) -> None:
        self.state.cached_cruxes = value


__all__ = [
    "DeadlockState",
    "DeadlockResolution",
    "DeadlockManager",
]
