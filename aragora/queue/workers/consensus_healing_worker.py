"""
Consensus Healing Background Worker.

Monitors debates that failed to reach consensus and attempts healing actions:
- Analyzes patterns in failed consensus attempts
- Triggers re-debates with modified parameters
- Notifies users of stuck debates
- Tracks healing metrics

Usage:
    from aragora.queue.workers.consensus_healing_worker import ConsensusHealingWorker

    worker = ConsensusHealingWorker()
    await worker.start()  # Starts background healing loop
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Job type constants
JOB_TYPE_CONSENSUS_HEAL = "consensus_heal"


class HealingAction(str, Enum):
    """Types of healing actions."""

    RE_DEBATE = "re_debate"  # Trigger a new debate
    EXTEND_ROUNDS = "extend_rounds"  # Continue with more rounds
    ADD_MEDIATOR = "add_mediator"  # Add a mediating agent
    NOTIFY_USER = "notify_user"  # Notify user of stuck debate
    ARCHIVE = "archive"  # Archive as unresolved
    ESCALATE = "escalate"  # Escalate to human review


class HealingReason(str, Enum):
    """Reasons for healing intervention."""

    NO_CONSENSUS = "no_consensus"  # Debate ended without consensus
    STALLED = "stalled"  # Debate stalled (no progress)
    DIVERGING = "diverging"  # Agents diverging rather than converging
    TIMEOUT = "timeout"  # Debate timed out
    ERROR = "error"  # Debate ended in error
    LOW_QUALITY = "low_quality"  # Low quality consensus


@dataclass
class HealingCandidate:
    """A debate candidate for healing."""

    debate_id: str
    task: str
    reason: HealingReason
    created_at: float
    completed_at: Optional[float]
    rounds_completed: int
    agent_count: int
    consensus_probability: float = 0.0
    convergence_trend: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealingResult:
    """Result of a healing attempt."""

    debate_id: str
    action: HealingAction
    success: bool
    message: str
    new_debate_id: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "debate_id": self.debate_id,
            "action": self.action.value,
            "success": self.success,
            "message": self.message,
            "new_debate_id": self.new_debate_id,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }


@dataclass
class HealingConfig:
    """Configuration for consensus healing."""

    # Timing
    scan_interval_seconds: float = 300  # 5 minutes
    min_age_for_healing_seconds: float = 3600  # 1 hour
    max_age_for_healing_seconds: float = 86400 * 7  # 7 days

    # Thresholds
    consensus_threshold: float = 0.6  # Below this = needs healing
    min_rounds_for_redebate: int = 3
    max_healing_attempts: int = 3

    # Actions
    auto_redebate_enabled: bool = False  # Require user approval by default
    notify_on_stuck: bool = True
    archive_after_max_attempts: bool = True


class ConsensusHealingWorker:
    """
    Background worker for consensus healing.

    Features:
    - Periodically scans for debates needing healing
    - Analyzes consensus failure patterns
    - Takes appropriate healing actions
    - Tracks healing metrics
    """

    def __init__(
        self,
        worker_id: Optional[str] = None,
        config: Optional[HealingConfig] = None,
        on_healing_needed: Optional[Callable[[HealingCandidate], None]] = None,
        on_healing_complete: Optional[Callable[[HealingResult], None]] = None,
    ):
        """
        Initialize consensus healing worker.

        Args:
            worker_id: Unique worker identifier
            config: Healing configuration
            on_healing_needed: Callback when candidate found
            on_healing_complete: Callback when healing completes
        """
        self.worker_id = worker_id or f"consensus-healer-{os.getpid()}"
        self.config = config or HealingConfig()
        self.on_healing_needed = on_healing_needed
        self.on_healing_complete = on_healing_complete

        self._running = False
        self._healing_history: List[HealingResult] = []
        self._candidates: Dict[str, HealingCandidate] = {}

        # Metrics
        self._scans_completed = 0
        self._candidates_found = 0
        self._healings_attempted = 0
        self._healings_succeeded = 0

    async def start(self) -> None:
        """Start the healing worker loop."""
        self._running = True
        logger.info(f"[{self.worker_id}] Starting consensus healing worker")

        while self._running:
            try:
                await self._scan_for_candidates()
                await self._process_candidates()
                await asyncio.sleep(self.config.scan_interval_seconds)
            except asyncio.CancelledError:
                logger.info(f"[{self.worker_id}] Worker cancelled")
                break
            except Exception as e:
                logger.error(f"[{self.worker_id}] Error in healing loop: {e}")
                await asyncio.sleep(30)  # Back off on errors

    async def stop(self) -> None:
        """Stop the healing worker."""
        logger.info(f"[{self.worker_id}] Stopping consensus healing worker")
        self._running = False

    async def _scan_for_candidates(self) -> None:
        """Scan for debates needing consensus healing."""
        try:
            candidates = await self._find_healing_candidates()
            self._scans_completed += 1

            for candidate in candidates:
                if candidate.debate_id not in self._candidates:
                    self._candidates[candidate.debate_id] = candidate
                    self._candidates_found += 1

                    if self.on_healing_needed:
                        try:
                            self.on_healing_needed(candidate)
                        except Exception as e:
                            logger.error(f"Error in on_healing_needed callback: {e}")

            logger.debug(f"[{self.worker_id}] Scan complete: {len(candidates)} candidates found")
        except Exception as e:
            logger.error(f"[{self.worker_id}] Error scanning for candidates: {e}")

    async def _find_healing_candidates(self) -> List[HealingCandidate]:
        """Find debates that need consensus healing."""
        candidates = []
        now = time.time()

        try:
            # Try to use consensus memory if available
            from aragora.memory.consensus import get_consensus_memory  # type: ignore[attr-defined]

            memory = get_consensus_memory()

            # Find debates without consensus
            # This is a simplified implementation - in production, query the database
            stale_debates = await self._query_stale_debates(memory)

            for debate_info in stale_debates:
                age = now - debate_info.get("created_at", now)

                # Check age bounds
                if age < self.config.min_age_for_healing_seconds:
                    continue
                if age > self.config.max_age_for_healing_seconds:
                    continue

                # Check if already being healed
                healing_attempts = debate_info.get("healing_attempts", 0)
                if healing_attempts >= self.config.max_healing_attempts:
                    continue

                candidate = HealingCandidate(
                    debate_id=debate_info.get("id", "unknown"),
                    task=debate_info.get("task", ""),
                    reason=self._determine_reason(debate_info),
                    created_at=debate_info.get("created_at", now),
                    completed_at=debate_info.get("completed_at"),
                    rounds_completed=debate_info.get("rounds_completed", 0),
                    agent_count=debate_info.get("agent_count", 0),
                    consensus_probability=debate_info.get("consensus_probability", 0.0),
                    convergence_trend=debate_info.get("convergence_trend", "unknown"),
                    metadata=debate_info,
                )
                candidates.append(candidate)

        except ImportError:
            logger.debug("ConsensusMemory not available, skipping scan")
        except Exception as e:
            logger.error(f"Error finding healing candidates: {e}")

        return candidates

    async def _query_stale_debates(self, memory: Any) -> List[Dict[str, Any]]:
        """Query for debates that might need healing."""
        # This would query the consensus memory for debates without strong consensus
        # For now, return empty list - real implementation would use memory.query()
        try:
            # Try to get recent debates without consensus
            if hasattr(memory, "query_unconverged"):
                return await memory.query_unconverged(
                    min_age_seconds=self.config.min_age_for_healing_seconds,
                    max_results=100,
                )
            return []
        except Exception:
            return []

    def _determine_reason(self, debate_info: Dict[str, Any]) -> HealingReason:
        """Determine the reason a debate needs healing."""
        consensus_prob = debate_info.get("consensus_probability", 0.0)
        convergence = debate_info.get("convergence_trend", "unknown")
        error = debate_info.get("error")

        if error:
            return HealingReason.ERROR
        if convergence == "diverging":
            return HealingReason.DIVERGING
        if consensus_prob < 0.3:
            return HealingReason.NO_CONSENSUS
        if debate_info.get("timed_out"):
            return HealingReason.TIMEOUT
        if consensus_prob < self.config.consensus_threshold:
            return HealingReason.LOW_QUALITY

        return HealingReason.STALLED

    async def _process_candidates(self) -> None:
        """Process healing candidates."""
        # Process in batches to avoid overwhelming the system
        to_process = list(self._candidates.values())[:10]

        for candidate in to_process:
            try:
                result = await self._heal_candidate(candidate)
                self._healings_attempted += 1

                if result.success:
                    self._healings_succeeded += 1

                self._healing_history.append(result)

                # Keep history bounded
                if len(self._healing_history) > 1000:
                    self._healing_history = self._healing_history[-500:]

                # Remove processed candidate
                self._candidates.pop(candidate.debate_id, None)

                if self.on_healing_complete:
                    try:
                        self.on_healing_complete(result)
                    except Exception as e:
                        logger.error(f"Error in on_healing_complete callback: {e}")

            except Exception as e:
                logger.error(f"Error healing candidate {candidate.debate_id}: {e}")

    async def _heal_candidate(self, candidate: HealingCandidate) -> HealingResult:
        """Attempt to heal a single candidate."""
        action = self._determine_action(candidate)

        logger.info(
            f"[{self.worker_id}] Healing {candidate.debate_id}: "
            f"reason={candidate.reason.value}, action={action.value}"
        )

        try:
            if action == HealingAction.RE_DEBATE:
                return await self._action_redebate(candidate)
            elif action == HealingAction.EXTEND_ROUNDS:
                return await self._action_extend_rounds(candidate)
            elif action == HealingAction.ADD_MEDIATOR:
                return await self._action_add_mediator(candidate)
            elif action == HealingAction.NOTIFY_USER:
                return await self._action_notify_user(candidate)
            elif action == HealingAction.ARCHIVE:
                return await self._action_archive(candidate)
            elif action == HealingAction.ESCALATE:
                return await self._action_escalate(candidate)
            else:
                return HealingResult(
                    debate_id=candidate.debate_id,
                    action=action,
                    success=False,
                    message=f"Unknown action: {action}",
                )
        except Exception as e:
            return HealingResult(
                debate_id=candidate.debate_id,
                action=action,
                success=False,
                message=f"Error: {e}",
            )

    def _determine_action(self, candidate: HealingCandidate) -> HealingAction:
        """Determine appropriate healing action for a candidate."""
        # Get healing attempts from metadata
        attempts = candidate.metadata.get("healing_attempts", 0)

        # After max attempts, archive
        if attempts >= self.config.max_healing_attempts - 1:
            if self.config.archive_after_max_attempts:
                return HealingAction.ARCHIVE
            return HealingAction.ESCALATE

        # Determine based on reason
        if candidate.reason == HealingReason.ERROR:
            return HealingAction.NOTIFY_USER

        if candidate.reason == HealingReason.DIVERGING:
            return HealingAction.ADD_MEDIATOR

        if candidate.reason == HealingReason.TIMEOUT:
            if candidate.rounds_completed >= self.config.min_rounds_for_redebate:
                return (
                    HealingAction.RE_DEBATE
                    if self.config.auto_redebate_enabled
                    else HealingAction.NOTIFY_USER
                )
            return HealingAction.EXTEND_ROUNDS

        if candidate.reason == HealingReason.NO_CONSENSUS:
            if self.config.auto_redebate_enabled:
                return HealingAction.RE_DEBATE
            return HealingAction.NOTIFY_USER

        if candidate.reason == HealingReason.LOW_QUALITY:
            return HealingAction.EXTEND_ROUNDS

        # Default: notify user
        return HealingAction.NOTIFY_USER

    async def _action_redebate(self, candidate: HealingCandidate) -> HealingResult:
        """Trigger a new debate with modified parameters."""
        # In a real implementation, this would create a new debate job
        logger.info(f"Would trigger re-debate for {candidate.debate_id}")

        return HealingResult(
            debate_id=candidate.debate_id,
            action=HealingAction.RE_DEBATE,
            success=True,
            message="Re-debate scheduled (simulation)",
            metrics={"original_rounds": candidate.rounds_completed},
        )

    async def _action_extend_rounds(self, candidate: HealingCandidate) -> HealingResult:
        """Extend the debate with additional rounds."""
        logger.info(f"Would extend rounds for {candidate.debate_id}")

        return HealingResult(
            debate_id=candidate.debate_id,
            action=HealingAction.EXTEND_ROUNDS,
            success=True,
            message="Rounds extended (simulation)",
            metrics={"additional_rounds": 3},
        )

    async def _action_add_mediator(self, candidate: HealingCandidate) -> HealingResult:
        """Add a mediating agent to help reach consensus."""
        logger.info(f"Would add mediator for {candidate.debate_id}")

        return HealingResult(
            debate_id=candidate.debate_id,
            action=HealingAction.ADD_MEDIATOR,
            success=True,
            message="Mediator added (simulation)",
        )

    async def _action_notify_user(self, candidate: HealingCandidate) -> HealingResult:
        """Notify user about stuck debate."""
        if not self.config.notify_on_stuck:
            return HealingResult(
                debate_id=candidate.debate_id,
                action=HealingAction.NOTIFY_USER,
                success=False,
                message="Notifications disabled",
            )

        logger.info(f"Would notify user about {candidate.debate_id}")

        return HealingResult(
            debate_id=candidate.debate_id,
            action=HealingAction.NOTIFY_USER,
            success=True,
            message="User notified (simulation)",
        )

    async def _action_archive(self, candidate: HealingCandidate) -> HealingResult:
        """Archive debate as unresolved."""
        logger.info(f"Archiving unresolved debate {candidate.debate_id}")

        return HealingResult(
            debate_id=candidate.debate_id,
            action=HealingAction.ARCHIVE,
            success=True,
            message="Debate archived as unresolved",
        )

    async def _action_escalate(self, candidate: HealingCandidate) -> HealingResult:
        """Escalate debate for human review."""
        logger.info(f"Escalating debate {candidate.debate_id} for human review")

        return HealingResult(
            debate_id=candidate.debate_id,
            action=HealingAction.ESCALATE,
            success=True,
            message="Escalated for human review",
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get healing worker metrics."""
        return {
            "worker_id": self.worker_id,
            "running": self._running,
            "scans_completed": self._scans_completed,
            "candidates_found": self._candidates_found,
            "candidates_pending": len(self._candidates),
            "healings_attempted": self._healings_attempted,
            "healings_succeeded": self._healings_succeeded,
            "success_rate": (
                self._healings_succeeded / self._healings_attempted
                if self._healings_attempted > 0
                else 0.0
            ),
            "recent_healings": [h.to_dict() for h in self._healing_history[-10:]],
        }

    def get_status(self) -> Dict[str, Any]:
        """Get worker status."""
        return {
            "worker_id": self.worker_id,
            "running": self._running,
            "config": {
                "scan_interval_seconds": self.config.scan_interval_seconds,
                "auto_redebate_enabled": self.config.auto_redebate_enabled,
                "max_healing_attempts": self.config.max_healing_attempts,
            },
            "metrics": self.get_metrics(),
        }


# Global worker instance
_global_worker: Optional[ConsensusHealingWorker] = None


def get_consensus_healing_worker() -> ConsensusHealingWorker:
    """Get or create global consensus healing worker."""
    global _global_worker
    if _global_worker is None:
        _global_worker = ConsensusHealingWorker()
    return _global_worker


async def start_consensus_healing() -> ConsensusHealingWorker:
    """Start the global consensus healing worker."""
    worker = get_consensus_healing_worker()
    asyncio.create_task(worker.start())
    return worker


async def stop_consensus_healing() -> None:
    """Stop the global consensus healing worker."""
    global _global_worker
    if _global_worker:
        await _global_worker.stop()


__all__ = [
    "ConsensusHealingWorker",
    "HealingConfig",
    "HealingAction",
    "HealingReason",
    "HealingCandidate",
    "HealingResult",
    "get_consensus_healing_worker",
    "start_consensus_healing",
    "stop_consensus_healing",
]
