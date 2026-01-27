"""
Stuck Detector: Identify and Recover Stuck Work.

This module identifies work that has stalled and triggers recovery actions.
Uses a traffic light system (GREEN/YELLOW/RED) for monitoring work age.

Key concepts:
- StuckDetector: Monitors work age thresholds
- WorkAge: Classification (GREEN, YELLOW, RED) based on time since update
- RecoveryAction: Actions to take for stuck work (reassign, escalate, notify)
- AutoRecovery: Automatic reassignment to available agents

Usage:
    from aragora.nomic.stuck_detector import StuckDetector

    detector = StuckDetector(
        bead_store=bead_store,
        coordinator=coordinator,
        escalation_store=escalation_store,
    )
    await detector.initialize()

    # Check all work
    stuck_items = await detector.detect_stuck_work()

    # Get work health summary
    summary = await detector.get_health_summary()

    # Start continuous monitoring
    await detector.start_monitoring()
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.nomic.beads import BeadStore
    from aragora.nomic.convoys import ConvoyManager
    from aragora.nomic.convoy_coordinator import ConvoyCoordinator
    from aragora.nomic.escalation_store import EscalationStore

logger = logging.getLogger(__name__)


class WorkAge(str, Enum):
    """
    Traffic light classification for work age.

    Based on time since last update:
    - GREEN: < 2 minutes - normal operation
    - YELLOW: 2-5 minutes - attention needed
    - RED: > 5 minutes - likely stuck
    """

    GREEN = "green"  # Normal, work progressing
    YELLOW = "yellow"  # Attention needed
    RED = "red"  # Likely stuck


class RecoveryAction(str, Enum):
    """Actions that can be taken for stuck work."""

    NONE = "none"  # No action needed
    NOTIFY = "notify"  # Send notification
    ESCALATE = "escalate"  # Escalate to supervisor
    REASSIGN = "reassign"  # Reassign to different agent
    CANCEL = "cancel"  # Cancel the work
    RETRY = "retry"  # Retry from checkpoint


@dataclass
class StuckWorkItem:
    """Information about a potentially stuck work item."""

    id: str
    work_type: str  # bead, convoy, assignment
    title: str
    agent_id: Optional[str]
    age: WorkAge
    last_update: datetime
    time_since_update: timedelta
    expected_duration: Optional[timedelta] = None
    previous_recoveries: int = 0
    recommended_action: RecoveryAction = RecoveryAction.NONE
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def age_minutes(self) -> float:
        """Get age in minutes."""
        return self.time_since_update.total_seconds() / 60

    @property
    def is_stuck(self) -> bool:
        """Check if work is considered stuck."""
        return self.age == WorkAge.RED


@dataclass
class StuckDetectorConfig:
    """Configuration for stuck detection thresholds."""

    # Age thresholds in minutes
    green_threshold_minutes: float = 2.0
    yellow_threshold_minutes: float = 5.0
    red_threshold_minutes: float = 10.0

    # Recovery settings
    max_recoveries: int = 3
    auto_reassign_on_red: bool = True
    escalate_after_recoveries: int = 2
    cancel_after_recoveries: int = 3

    # Monitoring settings
    check_interval_seconds: int = 30
    batch_size: int = 100

    # Notifications
    notify_on_yellow: bool = True
    notify_on_red: bool = True
    notify_on_recovery: bool = True


@dataclass
class HealthSummary:
    """Summary of work health across the system."""

    total_items: int
    green_count: int
    yellow_count: int
    red_count: int
    recovered_count: int
    failed_recoveries: int
    by_agent: Dict[str, Dict[str, int]]
    by_convoy: Dict[str, Dict[str, int]]
    oldest_stuck: Optional[StuckWorkItem] = None

    @property
    def health_percentage(self) -> float:
        """Get overall health as percentage of green items."""
        if self.total_items == 0:
            return 100.0
        return (self.green_count / self.total_items) * 100

    @property
    def stuck_count(self) -> int:
        """Get total stuck (red) items."""
        return self.red_count


class StuckDetector:
    """
    Monitors work items for stalls and triggers recovery.

    Uses configurable thresholds to classify work by age and
    takes automatic or manual recovery actions.
    """

    def __init__(
        self,
        bead_store: Optional["BeadStore"] = None,
        convoy_manager: Optional["ConvoyManager"] = None,
        coordinator: Optional["ConvoyCoordinator"] = None,
        escalation_store: Optional["EscalationStore"] = None,
        config: Optional[StuckDetectorConfig] = None,
    ):
        """
        Initialize the stuck detector.

        Args:
            bead_store: Store for bead operations
            convoy_manager: Manager for convoy operations
            coordinator: Coordinator for bead assignments
            escalation_store: Store for escalations
            config: Detector configuration
        """
        self.bead_store = bead_store
        self.convoy_manager = convoy_manager
        self.coordinator = coordinator
        self.escalation_store = escalation_store
        self.config = config or StuckDetectorConfig()

        self._recovery_counts: Dict[str, int] = {}  # item_id -> recovery count
        self._last_check: Dict[str, datetime] = {}  # item_id -> last check time
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        self._callbacks: List[Callable] = []
        self._lock = asyncio.Lock()

    def register_callback(self, callback: Callable) -> None:
        """Register a callback for stuck work events."""
        self._callbacks.append(callback)

    async def initialize(self) -> None:
        """Initialize the detector."""
        logger.info("StuckDetector initialized")

    async def start_monitoring(self) -> None:
        """Start continuous monitoring."""
        if self._running:
            return

        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("StuckDetector monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
        logger.info("StuckDetector monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                stuck_items = await self.detect_stuck_work()

                for item in stuck_items:
                    if item.is_stuck:
                        await self._handle_stuck_item(item)

                await asyncio.sleep(self.config.check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)

    def _classify_age(self, time_since_update: timedelta) -> WorkAge:
        """Classify work age into traffic light category."""
        minutes = time_since_update.total_seconds() / 60

        if minutes < self.config.green_threshold_minutes:
            return WorkAge.GREEN
        elif minutes < self.config.yellow_threshold_minutes:
            return WorkAge.YELLOW
        else:
            return WorkAge.RED

    def _determine_action(self, item: StuckWorkItem) -> RecoveryAction:
        """Determine recommended recovery action."""
        recovery_count = self._recovery_counts.get(item.id, 0)

        if item.age == WorkAge.GREEN:
            return RecoveryAction.NONE

        if item.age == WorkAge.YELLOW:
            return RecoveryAction.NOTIFY

        # RED status
        if recovery_count >= self.config.cancel_after_recoveries:
            return RecoveryAction.CANCEL
        elif recovery_count >= self.config.escalate_after_recoveries:
            return RecoveryAction.ESCALATE
        elif self.config.auto_reassign_on_red:
            return RecoveryAction.REASSIGN
        else:
            return RecoveryAction.ESCALATE

    async def detect_stuck_work(self) -> List[StuckWorkItem]:
        """
        Detect all stuck work items.

        Returns:
            List of stuck work items with age classification
        """
        stuck_items = []
        now = datetime.now(timezone.utc)

        # Check beads
        if self.bead_store:
            stuck_items.extend(await self._check_beads(now))

        # Check convoy assignments
        if self.coordinator:
            stuck_items.extend(await self._check_assignments(now))

        return stuck_items

    async def _check_beads(self, now: datetime) -> List[StuckWorkItem]:
        """Check beads for stuck work."""
        items = []

        from aragora.nomic.beads import BeadStatus

        # Get running beads
        running_beads = await self.bead_store.list_beads(status=BeadStatus.RUNNING)

        for bead in running_beads:
            last_update = bead.updated_at or bead.created_at
            time_since = now - last_update
            age = self._classify_age(time_since)

            item = StuckWorkItem(
                id=bead.id,
                work_type="bead",
                title=bead.title,
                agent_id=bead.assigned_to,
                age=age,
                last_update=last_update,
                time_since_update=time_since,
                previous_recoveries=self._recovery_counts.get(bead.id, 0),
                metadata={"bead_type": bead.bead_type.value},
            )
            item.recommended_action = self._determine_action(item)
            items.append(item)

        return items

    async def _check_assignments(self, now: datetime) -> List[StuckWorkItem]:
        """Check bead assignments for stuck work."""
        items = []

        from aragora.nomic.convoy_coordinator import AssignmentStatus

        # Get all active assignments
        for assignment in list(self.coordinator._assignments.values()):
            if assignment.status != AssignmentStatus.ACTIVE:
                continue

            last_update = assignment.updated_at or assignment.assigned_at
            time_since = now - last_update
            age = self._classify_age(time_since)

            item = StuckWorkItem(
                id=assignment.id,
                work_type="assignment",
                title=f"Assignment for bead {assignment.bead_id}",
                agent_id=assignment.agent_id,
                age=age,
                last_update=last_update,
                time_since_update=time_since,
                expected_duration=timedelta(minutes=assignment.estimated_duration_minutes),
                previous_recoveries=len(assignment.previous_agents),
                metadata={
                    "bead_id": assignment.bead_id,
                    "convoy_id": assignment.convoy_id,
                },
            )
            item.recommended_action = self._determine_action(item)
            items.append(item)

        return items

    async def _handle_stuck_item(self, item: StuckWorkItem) -> bool:
        """
        Handle a stuck work item.

        Args:
            item: The stuck work item

        Returns:
            True if recovery action was taken
        """
        action = item.recommended_action

        if action == RecoveryAction.NONE:
            return False

        async with self._lock:
            # Update recovery count
            self._recovery_counts[item.id] = item.previous_recoveries + 1

            # Notify callbacks
            for callback in self._callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(item, action)
                    else:
                        callback(item, action)
                except Exception as e:
                    logger.error(f"Callback error: {e}")

            # Take action
            if action == RecoveryAction.NOTIFY:
                logger.warning(
                    f"Work item {item.id} ({item.work_type}) is yellow: "
                    f"{item.age_minutes:.1f} minutes since update"
                )
                return True

            elif action == RecoveryAction.REASSIGN:
                return await self._do_reassign(item)

            elif action == RecoveryAction.ESCALATE:
                return await self._do_escalate(item)

            elif action == RecoveryAction.CANCEL:
                return await self._do_cancel(item)

            elif action == RecoveryAction.RETRY:
                return await self._do_retry(item)

        return False

    async def _do_reassign(self, item: StuckWorkItem) -> bool:
        """Reassign stuck work to a different agent."""
        if item.work_type == "assignment" and self.coordinator:
            bead_id = item.metadata.get("bead_id")
            if bead_id:
                from aragora.nomic.convoy_coordinator import RebalanceReason

                assignment = await self.coordinator.get_assignment(bead_id)
                if assignment:
                    new_assignment = await self.coordinator._reassign_bead(
                        assignment, RebalanceReason.PROGRESS_STALLED
                    )
                    if new_assignment:
                        logger.info(
                            f"Reassigned stuck bead {bead_id} from {item.agent_id} "
                            f"to {new_assignment.agent_id}"
                        )
                        return True

        elif item.work_type == "bead" and self.bead_store:
            # Mark bead as pending for re-claiming
            bead = await self.bead_store.get(item.id)
            if bead:
                from aragora.nomic.beads import BeadStatus

                bead.status = BeadStatus.PENDING
                bead.assigned_to = None
                await self.bead_store.update(bead)
                logger.info(f"Reset stuck bead {item.id} to pending")
                return True

        return False

    async def _do_escalate(self, item: StuckWorkItem) -> bool:
        """Escalate stuck work."""
        if self.escalation_store:
            await self.escalation_store.create_chain(
                source="stuck_detector",
                target=item.id,
                reason=f"Work stuck for {item.age_minutes:.1f} minutes after {item.previous_recoveries} recovery attempts",
                metadata={
                    "work_type": item.work_type,
                    "agent_id": item.agent_id,
                    "title": item.title,
                },
            )
            logger.info(f"Escalated stuck work {item.id}")
            return True

        logger.warning(f"Would escalate {item.id} but no escalation store configured")
        return False

    async def _do_cancel(self, item: StuckWorkItem) -> bool:
        """Cancel stuck work."""
        if item.work_type == "bead" and self.bead_store:
            bead = await self.bead_store.get(item.id)
            if bead:
                from aragora.nomic.beads import BeadStatus

                bead.status = BeadStatus.CANCELLED
                bead.error_message = (
                    f"Cancelled after {item.previous_recoveries} failed recovery attempts"
                )
                await self.bead_store.update(bead)
                logger.warning(f"Cancelled stuck bead {item.id}")
                return True

        elif item.work_type == "assignment" and self.coordinator:
            from aragora.nomic.convoy_coordinator import AssignmentStatus

            assignment = self.coordinator._assignments.get(item.id)
            if assignment:
                await self.coordinator.update_assignment_status(
                    assignment.bead_id,
                    AssignmentStatus.FAILED,
                    f"Failed after {item.previous_recoveries} recovery attempts",
                )
                logger.warning(f"Cancelled stuck assignment {item.id}")
                return True

        return False

    async def _do_retry(self, item: StuckWorkItem) -> bool:
        """Retry stuck work from checkpoint."""
        # This would integrate with molecule checkpointing
        logger.info(f"Would retry {item.id} from checkpoint")
        return False

    async def check_item(self, item_id: str, work_type: str) -> Optional[StuckWorkItem]:
        """
        Check a specific work item.

        Args:
            item_id: ID of the item
            work_type: Type of work (bead, assignment)

        Returns:
            StuckWorkItem if found, None otherwise
        """
        now = datetime.now(timezone.utc)

        if work_type == "bead" and self.bead_store:
            bead = await self.bead_store.get(item_id)
            if bead:
                last_update = bead.updated_at or bead.created_at
                time_since = now - last_update
                age = self._classify_age(time_since)

                item = StuckWorkItem(
                    id=bead.id,
                    work_type="bead",
                    title=bead.title,
                    agent_id=bead.assigned_to,
                    age=age,
                    last_update=last_update,
                    time_since_update=time_since,
                    previous_recoveries=self._recovery_counts.get(bead.id, 0),
                )
                item.recommended_action = self._determine_action(item)
                return item

        elif work_type == "assignment" and self.coordinator:
            assignment = self.coordinator._assignments.get(item_id)
            if assignment:
                last_update = assignment.updated_at or assignment.assigned_at
                time_since = now - last_update
                age = self._classify_age(time_since)

                item = StuckWorkItem(
                    id=assignment.id,
                    work_type="assignment",
                    title=f"Assignment for bead {assignment.bead_id}",
                    agent_id=assignment.agent_id,
                    age=age,
                    last_update=last_update,
                    time_since_update=time_since,
                    previous_recoveries=len(assignment.previous_agents),
                )
                item.recommended_action = self._determine_action(item)
                return item

        return None

    async def get_health_summary(self) -> HealthSummary:
        """
        Get a summary of work health.

        Returns:
            HealthSummary with statistics
        """
        items = await self.detect_stuck_work()

        green_count = len([i for i in items if i.age == WorkAge.GREEN])
        yellow_count = len([i for i in items if i.age == WorkAge.YELLOW])
        red_count = len([i for i in items if i.age == WorkAge.RED])

        # Group by agent
        by_agent: Dict[str, Dict[str, int]] = {}
        for item in items:
            if item.agent_id:
                if item.agent_id not in by_agent:
                    by_agent[item.agent_id] = {"green": 0, "yellow": 0, "red": 0}
                by_agent[item.agent_id][item.age.value] += 1

        # Group by convoy
        by_convoy: Dict[str, Dict[str, int]] = {}
        for item in items:
            convoy_id = item.metadata.get("convoy_id")
            if convoy_id:
                if convoy_id not in by_convoy:
                    by_convoy[convoy_id] = {"green": 0, "yellow": 0, "red": 0}
                by_convoy[convoy_id][item.age.value] += 1

        # Find oldest stuck
        stuck_items = [i for i in items if i.is_stuck]
        oldest = None
        if stuck_items:
            oldest = max(stuck_items, key=lambda i: i.time_since_update)

        return HealthSummary(
            total_items=len(items),
            green_count=green_count,
            yellow_count=yellow_count,
            red_count=red_count,
            recovered_count=sum(self._recovery_counts.values()),
            failed_recoveries=len([k for k, v in self._recovery_counts.items() if v > 1]),
            by_agent=by_agent,
            by_convoy=by_convoy,
            oldest_stuck=oldest,
        )

    async def get_statistics(self) -> Dict[str, Any]:
        """Get detector statistics."""
        summary = await self.get_health_summary()
        return {
            "total_items": summary.total_items,
            "healthy_percentage": summary.health_percentage,
            "stuck_count": summary.stuck_count,
            "recovered_count": summary.recovered_count,
            "monitoring_active": self._running,
        }


# Singleton instance
_default_detector: Optional[StuckDetector] = None


async def get_stuck_detector(
    bead_store: Optional["BeadStore"] = None,
    coordinator: Optional["ConvoyCoordinator"] = None,
    escalation_store: Optional["EscalationStore"] = None,
) -> StuckDetector:
    """Get the default stuck detector instance."""
    global _default_detector
    if _default_detector is None:
        _default_detector = StuckDetector(
            bead_store=bead_store,
            coordinator=coordinator,
            escalation_store=escalation_store,
        )
        await _default_detector.initialize()
    return _default_detector


def reset_stuck_detector() -> None:
    """Reset the default detector (for testing)."""
    global _default_detector
    _default_detector = None
