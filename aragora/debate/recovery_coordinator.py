"""
Recovery Coordinator for Multi-Agent Debates.

Coordinates recovery actions based on witness observations and deadlock detection:
- Agent replacement on persistent failures
- Deadlock resolution strategies
- Progress intervention tactics
- Integration with circuit breakers and protocol messages

Inspired by gastown's recovery coordination patterns.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from aragora.debate.deadlock_detector import Deadlock, DeadlockDetector, DeadlockType
from aragora.debate.witness import DebateWitness, StallEvent, StallReason, ProgressStatus
from aragora.debate.protocol_messages import (
    ProtocolMessage,
    ProtocolMessageType,
)
from aragora.debate.protocol_messages.messages import agent_event_message

logger = logging.getLogger(__name__)


class RecoveryAction(str, Enum):
    """Types of recovery actions."""

    NONE = "none"
    NUDGE = "nudge"  # Send reminder/prompt to agent
    REPLACE = "replace"  # Replace agent with another
    SKIP = "skip"  # Skip agent's turn
    RESET_ROUND = "reset_round"  # Reset the current round
    FORCE_VOTE = "force_vote"  # Force early voting
    INJECT_MEDIATOR = "inject_mediator"  # Add mediating agent
    ESCALATE = "escalate"  # Escalate to human operator
    ABORT = "abort"  # Abort the debate


@dataclass
class RecoveryDecision:
    """A decision on how to recover from an issue."""

    action: RecoveryAction
    target_agent_id: Optional[str] = None
    replacement_agent_id: Optional[str] = None
    reason: str = ""
    confidence: float = 1.0
    requires_approval: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryEvent:
    """Record of a recovery action taken."""

    id: str
    debate_id: str
    action: RecoveryAction
    trigger_type: str  # "stall", "deadlock", "failure"
    trigger_id: Optional[str] = None
    target_agent_id: Optional[str] = None
    result: str = "pending"  # pending, success, failed
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryConfig:
    """Configuration for recovery coordination."""

    # Agent replacement settings
    max_agent_failures: int = 3
    max_agent_stalls: int = 2
    replacement_pool: List[str] = field(default_factory=list)  # Available replacement agent IDs

    # Deadlock handling
    cycle_resolution_strategy: str = "inject_mediator"  # or "force_vote", "escalate"
    mutual_block_strategy: str = "nudge"  # or "inject_mediator", "skip"
    semantic_loop_strategy: str = "nudge"  # or "skip", "force_vote"
    convergence_failure_strategy: str = "force_vote"  # or "escalate"

    # Timing
    nudge_before_replace: bool = True
    approval_required_for_replace: bool = False
    approval_required_for_abort: bool = True

    # Callbacks
    emit_protocol_messages: bool = True


class RecoveryCoordinator:
    """
    Coordinates recovery actions for debate issues.

    The coordinator receives observations from the witness and deadlock
    detector, decides on recovery actions, and executes them.

    Usage:
        coordinator = RecoveryCoordinator(
            debate_id="debate-123",
            witness=witness,
            config=RecoveryConfig(replacement_pool=["backup-claude", "backup-gpt"]),
        )

        # Handle a stall
        recovery = await coordinator.handle_stall(stall_event)

        # Handle a deadlock
        recovery = await coordinator.handle_deadlock(deadlock)

        # Process all pending issues
        recoveries = await coordinator.process_pending_issues()
    """

    def __init__(
        self,
        debate_id: str,
        witness: Optional[DebateWitness] = None,
        config: Optional[RecoveryConfig] = None,
        on_action: Optional[Callable[[RecoveryEvent], Any]] = None,
        on_message: Optional[Callable[[ProtocolMessage], Any]] = None,
    ):
        """
        Initialize the recovery coordinator.

        Args:
            debate_id: ID of the debate
            witness: Witness instance for progress tracking
            config: Recovery configuration
            on_action: Callback when action is taken
            on_message: Callback for protocol messages
        """
        self.debate_id = debate_id
        self.witness = witness
        self.config = config or RecoveryConfig()
        self.on_action = on_action
        self.on_message = on_message

        # State tracking
        self._recovery_history: List[RecoveryEvent] = []
        self._pending_stalls: List[StallEvent] = []
        self._pending_deadlocks: List[Deadlock] = []
        self._replaced_agents: Set[str] = set()
        self._agent_nudge_counts: Dict[str, int] = {}
        self._event_counter = 0
        self._lock = asyncio.Lock()

    async def handle_stall(self, stall: StallEvent) -> Optional[RecoveryEvent]:
        """
        Handle a detected stall event.

        Args:
            stall: The stall event from the witness

        Returns:
            Recovery event if action was taken
        """
        decision = self._decide_stall_recovery(stall)
        if decision.action == RecoveryAction.NONE:
            return None

        return await self._execute_recovery(
            decision=decision,
            trigger_type="stall",
            trigger_id=f"stall-{stall.agent_id}-{stall.round_number}",
        )

    async def handle_deadlock(self, deadlock: Deadlock) -> Optional[RecoveryEvent]:
        """
        Handle a detected deadlock.

        Args:
            deadlock: The deadlock from the detector

        Returns:
            Recovery event if action was taken
        """
        decision = self._decide_deadlock_recovery(deadlock)
        if decision.action == RecoveryAction.NONE:
            return None

        return await self._execute_recovery(
            decision=decision,
            trigger_type="deadlock",
            trigger_id=deadlock.id,
        )

    async def handle_agent_failure(self, agent_id: str, error: str) -> Optional[RecoveryEvent]:
        """
        Handle an agent failure.

        Args:
            agent_id: ID of the failed agent
            error: Error message

        Returns:
            Recovery event if action was taken
        """
        # Track failure in witness if available
        if self.witness:
            agent = self.witness.get_agent_progress(agent_id)
            if agent:
                agent.failure_count += 1
                agent.status = ProgressStatus.FAILED

        decision = self._decide_failure_recovery(agent_id, error)
        if decision.action == RecoveryAction.NONE:
            return None

        return await self._execute_recovery(
            decision=decision,
            trigger_type="failure",
            trigger_id=f"failure-{agent_id}",
            details={"error": error},
        )

    def _decide_stall_recovery(self, stall: StallEvent) -> RecoveryDecision:
        """Decide how to recover from a stall."""
        agent_id = stall.agent_id

        # Check if agent has been nudged already
        nudge_count = self._agent_nudge_counts.get(agent_id, 0)

        # Determine action based on stall reason
        if stall.reason == StallReason.TIMEOUT:
            if self.config.nudge_before_replace and nudge_count < self.config.max_agent_stalls:
                self._agent_nudge_counts[agent_id] = nudge_count + 1
                return RecoveryDecision(
                    action=RecoveryAction.NUDGE,
                    target_agent_id=agent_id,
                    reason=f"Agent timed out (nudge {nudge_count + 1}/{self.config.max_agent_stalls})",
                )
            else:
                return self._decide_replacement(agent_id, "repeated stalls")

        elif stall.reason == StallReason.REPEATED_CONTENT:
            return RecoveryDecision(
                action=RecoveryAction.NUDGE,
                target_agent_id=agent_id,
                reason="Agent repeating previous content",
                metadata={"nudge_type": "request_novel_perspective"},
            )

        elif stall.reason == StallReason.AGENT_FAILURE:
            return self._decide_replacement(agent_id, "agent failure")

        elif stall.reason == StallReason.NO_PROGRESS:
            return RecoveryDecision(
                action=RecoveryAction.SKIP,
                target_agent_id=agent_id,
                reason="No progress in round",
            )

        return RecoveryDecision(action=RecoveryAction.NONE)

    def _decide_deadlock_recovery(self, deadlock: Deadlock) -> RecoveryDecision:
        """Decide how to recover from a deadlock."""
        strategy_map = {
            DeadlockType.CYCLE: self.config.cycle_resolution_strategy,
            DeadlockType.MUTUAL_BLOCK: self.config.mutual_block_strategy,
            DeadlockType.SEMANTIC_LOOP: self.config.semantic_loop_strategy,
            DeadlockType.CONVERGENCE_FAILURE: self.config.convergence_failure_strategy,
        }

        strategy = strategy_map.get(deadlock.deadlock_type, "escalate")
        action = (
            RecoveryAction(strategy)
            if strategy in [a.value for a in RecoveryAction]
            else RecoveryAction.ESCALATE
        )

        decision = RecoveryDecision(
            action=action,
            reason=f"Deadlock resolution: {deadlock.description}",
            metadata={
                "deadlock_type": deadlock.deadlock_type.value,
                "severity": deadlock.severity,
            },
        )

        # Set target agent for agent-specific actions
        if action == RecoveryAction.NUDGE and deadlock.involved_agents:
            decision.target_agent_id = deadlock.involved_agents[0]

        # High severity or abort requires approval
        if deadlock.severity in ("high", "critical") or action == RecoveryAction.ABORT:
            decision.requires_approval = True

        return decision

    def _decide_failure_recovery(self, agent_id: str, error: str) -> RecoveryDecision:
        """Decide how to recover from an agent failure."""
        # Check failure count
        failure_count = 0
        if self.witness:
            agent = self.witness.get_agent_progress(agent_id)
            if agent:
                failure_count = agent.failure_count

        if failure_count >= self.config.max_agent_failures:
            return self._decide_replacement(agent_id, f"too many failures ({failure_count})")

        # Transient errors can be retried
        transient_patterns = ["timeout", "rate limit", "503", "429", "connection"]
        if any(p in error.lower() for p in transient_patterns):
            return RecoveryDecision(
                action=RecoveryAction.NUDGE,
                target_agent_id=agent_id,
                reason=f"Transient error: {error[:100]}",
                metadata={"retry": True},
            )

        # Permanent errors should replace
        return self._decide_replacement(agent_id, f"error: {error[:100]}")

    def _decide_replacement(self, agent_id: str, reason: str) -> RecoveryDecision:
        """Decide on agent replacement."""
        if agent_id in self._replaced_agents:
            return RecoveryDecision(
                action=RecoveryAction.SKIP,
                target_agent_id=agent_id,
                reason=f"Agent already replaced, skipping ({reason})",
            )

        if not self.config.replacement_pool:
            return RecoveryDecision(
                action=RecoveryAction.ESCALATE,
                target_agent_id=agent_id,
                reason=f"No replacement available for {agent_id} ({reason})",
                requires_approval=True,
            )

        # Find available replacement
        replacement = None
        for candidate in self.config.replacement_pool:
            if candidate not in self._replaced_agents and candidate != agent_id:
                replacement = candidate
                break

        if not replacement:
            return RecoveryDecision(
                action=RecoveryAction.ESCALATE,
                target_agent_id=agent_id,
                reason=f"All replacements exhausted ({reason})",
                requires_approval=True,
            )

        return RecoveryDecision(
            action=RecoveryAction.REPLACE,
            target_agent_id=agent_id,
            replacement_agent_id=replacement,
            reason=reason,
            requires_approval=self.config.approval_required_for_replace,
        )

    async def _execute_recovery(
        self,
        decision: RecoveryDecision,
        trigger_type: str,
        trigger_id: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> RecoveryEvent:
        """Execute a recovery decision."""
        async with self._lock:
            event = RecoveryEvent(
                id=self._generate_event_id(),
                debate_id=self.debate_id,
                action=decision.action,
                trigger_type=trigger_type,
                trigger_id=trigger_id,
                target_agent_id=decision.target_agent_id,
                details={
                    "reason": decision.reason,
                    "confidence": decision.confidence,
                    "requires_approval": decision.requires_approval,
                    **(decision.metadata or {}),
                    **(details or {}),
                },
            )

            # Check if approval needed
            if decision.requires_approval:
                event.result = "pending_approval"
                logger.info(
                    f"Recovery {event.id} requires approval: {decision.action.value} "
                    f"for {decision.target_agent_id}"
                )
            else:
                # Execute the action
                success = await self._execute_action(decision)
                event.result = "success" if success else "failed"
                event.completed_at = datetime.now(timezone.utc)

            self._recovery_history.append(event)

            # Emit protocol message if configured
            if self.config.emit_protocol_messages and self.on_message:
                message = self._create_protocol_message(decision, event)
                if message:
                    await self._emit_message(message)

            # Notify callback
            if self.on_action:
                try:
                    result = self.on_action(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Recovery callback error: {e}")

            return event

    async def _execute_action(self, decision: RecoveryDecision) -> bool:
        """Execute the actual recovery action."""
        try:
            if decision.action == RecoveryAction.NUDGE:
                logger.info(f"Nudging agent {decision.target_agent_id}: {decision.reason}")
                return True

            elif decision.action == RecoveryAction.REPLACE:
                logger.info(
                    f"Replacing {decision.target_agent_id} with {decision.replacement_agent_id}"
                )
                if decision.target_agent_id:
                    self._replaced_agents.add(decision.target_agent_id)
                return True

            elif decision.action == RecoveryAction.SKIP:
                logger.info(f"Skipping agent {decision.target_agent_id}")
                return True

            elif decision.action == RecoveryAction.RESET_ROUND:
                logger.info("Resetting current round")
                return True

            elif decision.action == RecoveryAction.FORCE_VOTE:
                logger.info("Forcing early voting")
                return True

            elif decision.action == RecoveryAction.INJECT_MEDIATOR:
                logger.info("Injecting mediator agent")
                return True

            elif decision.action == RecoveryAction.ESCALATE:
                logger.warning(f"Escalating: {decision.reason}")
                return True

            elif decision.action == RecoveryAction.ABORT:
                logger.error(f"Aborting debate: {decision.reason}")
                return True

            return False

        except Exception as e:
            logger.error(f"Recovery execution failed: {e}")
            return False

    def _create_protocol_message(
        self, decision: RecoveryDecision, event: RecoveryEvent
    ) -> Optional[ProtocolMessage]:
        """Create a protocol message for the recovery action."""
        if decision.action == RecoveryAction.REPLACE:
            return agent_event_message(
                debate_id=self.debate_id,
                agent_id=decision.target_agent_id or "unknown",
                agent_name=decision.target_agent_id or "unknown",
                model="unknown",
                role="participant",
                event_type=ProtocolMessageType.AGENT_REPLACED,
                reason=decision.reason,
                replacement_id=decision.replacement_agent_id,
            )

        elif decision.action in (RecoveryAction.ABORT, RecoveryAction.ESCALATE):
            return ProtocolMessage(
                message_type=ProtocolMessageType.RECOVERY_INITIATED,
                debate_id=self.debate_id,
                metadata={
                    "action": decision.action.value,
                    "reason": decision.reason,
                    "event_id": event.id,
                },
            )

        return None

    async def _emit_message(self, message: ProtocolMessage) -> None:
        """Emit a protocol message."""
        if self.on_message:
            try:
                result = self.on_message(message)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Protocol message emission failed: {e}")

    def _generate_event_id(self) -> str:
        """Generate a unique event ID."""
        self._event_counter += 1
        return f"recovery-{self.debate_id[:8]}-{self._event_counter:04d}"

    async def process_pending_issues(self) -> List[RecoveryEvent]:
        """
        Process all pending stalls and deadlocks.

        Returns:
            List of recovery events from processing
        """
        events = []

        for stall in self._pending_stalls:
            event = await self.handle_stall(stall)
            if event:
                events.append(event)
        self._pending_stalls.clear()

        for deadlock in self._pending_deadlocks:
            event = await self.handle_deadlock(deadlock)
            if event:
                events.append(event)
        self._pending_deadlocks.clear()

        return events

    def queue_stall(self, stall: StallEvent) -> None:
        """Queue a stall for later processing."""
        self._pending_stalls.append(stall)

    def queue_deadlock(self, deadlock: Deadlock) -> None:
        """Queue a deadlock for later processing."""
        self._pending_deadlocks.append(deadlock)

    async def approve_recovery(self, event_id: str, approved: bool) -> bool:
        """
        Approve or reject a pending recovery action.

        Args:
            event_id: ID of the recovery event
            approved: Whether to approve the action

        Returns:
            True if event was found and updated
        """
        for event in self._recovery_history:
            if event.id == event_id and event.result == "pending_approval":
                if approved:
                    # Re-execute with approval
                    decision = RecoveryDecision(
                        action=event.action,
                        target_agent_id=event.target_agent_id,
                        replacement_agent_id=event.details.get("replacement_agent_id"),
                        reason=event.details.get("reason", ""),
                    )
                    success = await self._execute_action(decision)
                    event.result = "success" if success else "failed"
                else:
                    event.result = "rejected"
                event.completed_at = datetime.now(timezone.utc)
                return True
        return False

    def get_recovery_history(
        self, limit: int = 100, include_pending: bool = True
    ) -> List[Dict[str, Any]]:
        """Get recovery action history."""
        events = self._recovery_history
        if not include_pending:
            events = [e for e in events if e.result not in ("pending", "pending_approval")]

        return [
            {
                "id": e.id,
                "debate_id": e.debate_id,
                "action": e.action.value,
                "trigger_type": e.trigger_type,
                "trigger_id": e.trigger_id,
                "target_agent_id": e.target_agent_id,
                "result": e.result,
                "timestamp": e.timestamp.isoformat(),
                "completed_at": e.completed_at.isoformat() if e.completed_at else None,
                "details": e.details,
            }
            for e in events[-limit:]
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        return {
            "debate_id": self.debate_id,
            "total_recoveries": len(self._recovery_history),
            "successful_recoveries": len(
                [e for e in self._recovery_history if e.result == "success"]
            ),
            "failed_recoveries": len([e for e in self._recovery_history if e.result == "failed"]),
            "pending_approvals": len(
                [e for e in self._recovery_history if e.result == "pending_approval"]
            ),
            "replaced_agents": list(self._replaced_agents),
            "pending_stalls": len(self._pending_stalls),
            "pending_deadlocks": len(self._pending_deadlocks),
            "actions_by_type": {
                action.value: len([e for e in self._recovery_history if e.action == action])
                for action in RecoveryAction
            },
        }


# Factory function for integrated witness + recovery setup
async def create_debate_observer(
    debate_id: str,
    agents: List[str],
    replacement_pool: Optional[List[str]] = None,
    on_protocol_message: Optional[Callable[[ProtocolMessage], Any]] = None,
) -> tuple[DebateWitness, DeadlockDetector, RecoveryCoordinator]:
    """
    Create an integrated observer system for a debate.

    Args:
        debate_id: ID of the debate
        agents: List of agent IDs participating
        replacement_pool: Optional list of replacement agent IDs
        on_protocol_message: Optional callback for protocol messages

    Returns:
        Tuple of (witness, deadlock_detector, recovery_coordinator)
    """
    # Create witness
    witness = DebateWitness(debate_id=debate_id)
    for agent_id in agents:
        witness.register_agent(agent_id)

    # Create deadlock detector
    detector = DeadlockDetector(debate_id=debate_id)

    # Create recovery coordinator
    config = RecoveryConfig(
        replacement_pool=replacement_pool or [],
        emit_protocol_messages=on_protocol_message is not None,
    )

    coordinator = RecoveryCoordinator(
        debate_id=debate_id,
        witness=witness,
        config=config,
        on_message=on_protocol_message,
    )

    # Wire up callbacks
    witness.on_stall = coordinator.queue_stall

    return witness, detector, coordinator
