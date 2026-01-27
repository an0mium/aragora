"""
Witness Behavior: Active Monitoring for the WITNESS Role.

This module implements the WITNESS role behavior with active monitoring
of agent health, bead progress, and convoy completion rates.

Key concepts:
- WitnessBehavior: Patrol loop for monitoring agents and work
- HealthReport: Structured health status for agents and convoys
- AlertTrigger: When and how to escalate issues to MAYOR
- HeartbeatMonitor: Track agent liveness

Usage:
    from aragora.nomic.witness_behavior import WitnessBehavior

    witness = WitnessBehavior(
        hierarchy=agent_hierarchy,
        convoy_manager=convoy_manager,
        escalation_store=escalation_store,
    )
    await witness.initialize()

    # Start the patrol loop
    await witness.start_patrol()

    # Generate a health report
    report = await witness.generate_health_report()

    # Check specific agent
    is_healthy = await witness.check_agent_health("agent-001")
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.nomic.convoy_coordinator import ConvoyCoordinator
    from aragora.nomic.convoys import ConvoyManager
    from aragora.nomic.escalation_store import EscalationStore

from aragora.nomic.agent_roles import (
    AgentHierarchy,
    AgentRole,
    RoleAssignment,
)

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status levels."""

    HEALTHY = "healthy"  # All systems normal
    DEGRADED = "degraded"  # Some issues but functional
    UNHEALTHY = "unhealthy"  # Significant issues
    CRITICAL = "critical"  # Immediate attention needed
    UNKNOWN = "unknown"  # Cannot determine status


class AlertSeverity(str, Enum):
    """Severity levels for alerts."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AgentHealthCheck:
    """Health check result for an agent."""

    agent_id: str
    status: HealthStatus
    checked_at: datetime
    last_heartbeat: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    active_beads: int = 0
    completed_beads: int = 0
    failed_beads: int = 0
    avg_response_time_ms: Optional[float] = None
    error_rate: float = 0.0
    issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_responsive(self) -> bool:
        """Check if agent has responded recently."""
        if not self.last_heartbeat:
            return False
        age = datetime.now(timezone.utc) - self.last_heartbeat
        return age < timedelta(minutes=5)

    @property
    def is_stuck(self) -> bool:
        """Check if agent appears stuck."""
        if self.active_beads == 0:
            return False
        if not self.last_activity:
            return True
        age = datetime.now(timezone.utc) - self.last_activity
        return age > timedelta(minutes=10)


@dataclass
class ConvoyHealthCheck:
    """Health check result for a convoy."""

    convoy_id: str
    status: HealthStatus
    checked_at: datetime
    total_beads: int
    completed_beads: int
    failed_beads: int
    stuck_beads: int
    completion_rate: float
    avg_bead_duration_minutes: Optional[float] = None
    estimated_completion: Optional[datetime] = None
    issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def progress_percentage(self) -> float:
        """Get completion percentage."""
        if self.total_beads == 0:
            return 0.0
        return (self.completed_beads / self.total_beads) * 100


@dataclass
class HealthReport:
    """
    Comprehensive health report from the witness.

    Aggregates health information across agents and convoys.
    """

    report_id: str
    generated_at: datetime
    overall_status: HealthStatus
    agent_checks: List[AgentHealthCheck]
    convoy_checks: List[ConvoyHealthCheck]
    alerts: List["Alert"]
    statistics: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "overall_status": self.overall_status.value,
            "agent_checks": [
                {
                    "agent_id": c.agent_id,
                    "status": c.status.value,
                    "checked_at": c.checked_at.isoformat(),
                    "active_beads": c.active_beads,
                    "issues": c.issues,
                }
                for c in self.agent_checks
            ],
            "convoy_checks": [
                {
                    "convoy_id": c.convoy_id,
                    "status": c.status.value,
                    "progress": c.progress_percentage,
                    "issues": c.issues,
                }
                for c in self.convoy_checks
            ],
            "alerts": [a.to_dict() for a in self.alerts],
            "statistics": self.statistics,
            "recommendations": self.recommendations,
        }


@dataclass
class Alert:
    """An alert generated by the witness."""

    id: str
    severity: AlertSeverity
    source: str  # What generated the alert
    target: str  # What the alert is about
    message: str
    timestamp: datetime
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "severity": self.severity.value,
            "source": self.source,
            "target": self.target,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "metadata": self.metadata,
        }


@dataclass
class WitnessConfig:
    """Configuration for witness behavior."""

    # Patrol timing
    patrol_interval_seconds: int = 30
    health_check_interval_seconds: int = 60
    report_interval_seconds: int = 300

    # Thresholds
    heartbeat_timeout_seconds: int = 120
    stuck_threshold_minutes: int = 10
    error_rate_threshold: float = 0.2
    completion_rate_threshold: float = 0.5

    # Alert settings
    max_alerts_per_target: int = 5
    alert_cooldown_seconds: int = 300
    auto_escalate_critical: bool = True

    # Behavior
    notify_mayor_on_critical: bool = True
    auto_reassign_stuck_beads: bool = False
    spawn_polecats_on_overload: bool = True


class HeartbeatMonitor:
    """Monitors agent heartbeats for liveness detection."""

    def __init__(self, timeout_seconds: int = 120):
        """
        Initialize the heartbeat monitor.

        Args:
            timeout_seconds: Time before agent considered unresponsive
        """
        self.timeout_seconds = timeout_seconds
        self._heartbeats: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    async def record_heartbeat(self, agent_id: str) -> None:
        """Record a heartbeat from an agent."""
        async with self._lock:
            self._heartbeats[agent_id] = datetime.now(timezone.utc)

    async def get_last_heartbeat(self, agent_id: str) -> Optional[datetime]:
        """Get the last heartbeat time for an agent."""
        return self._heartbeats.get(agent_id)

    async def is_responsive(self, agent_id: str) -> bool:
        """Check if an agent is responsive."""
        last_heartbeat = self._heartbeats.get(agent_id)
        if not last_heartbeat:
            return False
        age = datetime.now(timezone.utc) - last_heartbeat
        return age.total_seconds() < self.timeout_seconds

    async def get_unresponsive_agents(self) -> List[str]:
        """Get list of unresponsive agent IDs."""
        now = datetime.now(timezone.utc)
        unresponsive = []
        for agent_id, last_heartbeat in self._heartbeats.items():
            if (now - last_heartbeat).total_seconds() > self.timeout_seconds:
                unresponsive.append(agent_id)
        return unresponsive


class WitnessBehavior:
    """
    Implements the WITNESS role patrol behavior.

    Monitors agent health, bead progress, and convoy completion,
    generating alerts and escalations as needed.
    """

    def __init__(
        self,
        hierarchy: AgentHierarchy,
        convoy_manager: Optional["ConvoyManager"] = None,
        coordinator: Optional["ConvoyCoordinator"] = None,
        escalation_store: Optional["EscalationStore"] = None,
        config: Optional[WitnessConfig] = None,
    ):
        """
        Initialize the witness behavior.

        Args:
            hierarchy: Agent role hierarchy
            convoy_manager: Optional convoy manager for convoy monitoring
            coordinator: Optional convoy coordinator for bead assignments
            escalation_store: Optional escalation store for alerts
            config: Witness configuration
        """
        self.hierarchy = hierarchy
        self.convoy_manager = convoy_manager
        self.coordinator = coordinator
        self.escalation_store = escalation_store
        self.config = config or WitnessConfig()

        self.heartbeat_monitor = HeartbeatMonitor(
            timeout_seconds=self.config.heartbeat_timeout_seconds
        )

        self._alerts: Dict[str, Alert] = {}
        self._alert_counts: Dict[str, int] = {}  # target -> count
        self._alert_cooldowns: Dict[str, datetime] = {}  # target -> cooldown until
        self._reports: List[HealthReport] = []
        self._patrol_task: Optional[asyncio.Task] = None
        self._running = False
        self._lock = asyncio.Lock()
        self._callbacks: List[Callable] = []

    def register_callback(self, callback: Callable) -> None:
        """Register a callback for alerts."""
        self._callbacks.append(callback)

    async def start_patrol(self) -> None:
        """Start the patrol loop."""
        if self._running:
            return

        self._running = True
        self._patrol_task = asyncio.create_task(self._patrol_loop())
        logger.info("Witness patrol started")

    async def stop_patrol(self) -> None:
        """Stop the patrol loop."""
        self._running = False
        if self._patrol_task:
            self._patrol_task.cancel()
            try:
                await self._patrol_task
            except asyncio.CancelledError:
                pass
            self._patrol_task = None
        logger.info("Witness patrol stopped")

    async def _patrol_loop(self) -> None:
        """Main patrol loop."""
        while self._running:
            try:
                await self._run_patrol_cycle()
                await asyncio.sleep(self.config.patrol_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Patrol cycle error: {e}")
                await asyncio.sleep(5)  # Brief pause on error

    async def _run_patrol_cycle(self) -> None:
        """Run a single patrol cycle."""
        # Check agent health
        agents = await self.hierarchy.get_agents_by_role(AgentRole.CREW)
        agents.extend(await self.hierarchy.get_agents_by_role(AgentRole.POLECAT))

        for agent in agents:
            health = await self.check_agent_health(agent.agent_id)
            if health.status in (HealthStatus.UNHEALTHY, HealthStatus.CRITICAL):
                await self._handle_unhealthy_agent(agent, health)

        # Check convoy progress
        if self.convoy_manager:
            await self._check_convoy_progress()

        # Process escalation auto-escalations
        if self.escalation_store:
            await self.escalation_store.process_auto_escalations()

        # Cleanup expired polecats
        await self.hierarchy.cleanup_expired_polecats()

    async def check_agent_health(self, agent_id: str) -> AgentHealthCheck:
        """
        Check health of a specific agent.

        Args:
            agent_id: ID of agent to check

        Returns:
            AgentHealthCheck with status details
        """
        now = datetime.now(timezone.utc)
        issues = []
        status = HealthStatus.HEALTHY

        # Get agent assignment
        assignment = await self.hierarchy.get_assignment(agent_id)
        if not assignment:
            return AgentHealthCheck(
                agent_id=agent_id,
                status=HealthStatus.UNKNOWN,
                checked_at=now,
                issues=["Agent not found in hierarchy"],
            )

        # Check heartbeat
        last_heartbeat = await self.heartbeat_monitor.get_last_heartbeat(agent_id)
        is_responsive = await self.heartbeat_monitor.is_responsive(agent_id)

        if not is_responsive:
            issues.append("No recent heartbeat")
            status = HealthStatus.UNHEALTHY

        # Check bead assignments if coordinator available
        active_beads = 0
        completed_beads = 0
        failed_beads = 0
        last_activity = None

        if self.coordinator:
            from aragora.nomic.convoy_coordinator import AssignmentStatus

            assignments = await self.coordinator.get_agent_assignments(agent_id)
            for asg in assignments:
                if asg.status == AssignmentStatus.ACTIVE:
                    active_beads += 1
                    if asg.started_at and (not last_activity or asg.started_at > last_activity):
                        last_activity = asg.started_at
                elif asg.status == AssignmentStatus.COMPLETED:
                    completed_beads += 1
                    if asg.completed_at and (not last_activity or asg.completed_at > last_activity):
                        last_activity = asg.completed_at
                elif asg.status == AssignmentStatus.FAILED:
                    failed_beads += 1

        # Check for stuck work
        if active_beads > 0 and last_activity:
            age = now - last_activity
            if age > timedelta(minutes=self.config.stuck_threshold_minutes):
                issues.append(f"No activity for {age.total_seconds() / 60:.1f} minutes")
                status = HealthStatus.DEGRADED

        # Check error rate
        total = completed_beads + failed_beads
        error_rate = failed_beads / total if total > 0 else 0.0
        if error_rate > self.config.error_rate_threshold:
            issues.append(f"High error rate: {error_rate:.1%}")
            if status == HealthStatus.HEALTHY:
                status = HealthStatus.DEGRADED

        # Determine final status
        if len(issues) >= 3:
            status = HealthStatus.CRITICAL
        elif len(issues) >= 2 and status == HealthStatus.DEGRADED:
            status = HealthStatus.UNHEALTHY

        return AgentHealthCheck(
            agent_id=agent_id,
            status=status,
            checked_at=now,
            last_heartbeat=last_heartbeat,
            last_activity=last_activity,
            active_beads=active_beads,
            completed_beads=completed_beads,
            failed_beads=failed_beads,
            error_rate=error_rate,
            issues=issues,
        )

    async def _handle_unhealthy_agent(
        self,
        assignment: RoleAssignment,
        health: AgentHealthCheck,
    ) -> None:
        """Handle an unhealthy agent."""
        agent_id = assignment.agent_id

        # Create alert
        severity = (
            AlertSeverity.CRITICAL
            if health.status == HealthStatus.CRITICAL
            else AlertSeverity.ERROR
        )
        message = f"Agent {agent_id} is {health.status.value}: {', '.join(health.issues)}"

        alert = await self._create_alert(
            severity=severity,
            source="witness_patrol",
            target=agent_id,
            message=message,
        )

        if alert and self.config.auto_escalate_critical and health.status == HealthStatus.CRITICAL:
            # Escalate to mayor
            if self.escalation_store:
                await self.escalation_store.create_chain(
                    source="witness",
                    target=agent_id,
                    reason=message,
                )

        # Auto-reassign stuck beads if configured
        if self.config.auto_reassign_stuck_beads and self.coordinator and health.is_stuck:
            await self.coordinator.handle_agent_failure(agent_id)

    async def _check_convoy_progress(self) -> None:
        """Check progress of all active convoys."""
        if not self.convoy_manager:
            return

        from aragora.nomic.convoys import ConvoyStatus

        convoys = await self.convoy_manager.list_convoys(status=ConvoyStatus.ACTIVE)

        for convoy in convoys:
            progress = await self.convoy_manager.get_convoy_progress(convoy.id)

            # Check for stalled convoys
            completion_rate = progress.completion_percentage / 100.0

            if completion_rate < self.config.completion_rate_threshold:
                if progress.running_beads == 0 and progress.pending_beads > 0:
                    # Convoy is stalled
                    await self._create_alert(
                        severity=AlertSeverity.WARNING,
                        source="witness_patrol",
                        target=convoy.id,
                        message=f"Convoy {convoy.title} is stalled: {progress.pending_beads} pending beads",
                    )

    async def _create_alert(
        self,
        severity: AlertSeverity,
        source: str,
        target: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Alert]:
        """
        Create an alert if within limits.

        Args:
            severity: Alert severity
            source: Alert source
            target: Alert target
            message: Alert message
            metadata: Optional metadata

        Returns:
            Alert if created, None if suppressed
        """
        import uuid

        async with self._lock:
            # Check cooldown
            cooldown_until = self._alert_cooldowns.get(target)
            if cooldown_until and datetime.now(timezone.utc) < cooldown_until:
                logger.debug(f"Alert for {target} suppressed by cooldown")
                return None

            # Check alert count
            count = self._alert_counts.get(target, 0)
            if count >= self.config.max_alerts_per_target:
                logger.warning(f"Alert limit reached for {target}")
                return None

            # Create alert
            alert = Alert(
                id=str(uuid.uuid4()),
                severity=severity,
                source=source,
                target=target,
                message=message,
                timestamp=datetime.now(timezone.utc),
                metadata=metadata or {},
            )

            self._alerts[alert.id] = alert
            self._alert_counts[target] = count + 1
            self._alert_cooldowns[target] = datetime.now(timezone.utc) + timedelta(
                seconds=self.config.alert_cooldown_seconds
            )

            logger.info(f"Alert created: [{severity.value}] {message}")

            # Notify callbacks
            for callback in self._callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback error: {e}")

            # Notify mayor if critical
            if severity == AlertSeverity.CRITICAL and self.config.notify_mayor_on_critical:
                await self._notify_mayor(alert)

            return alert

    async def _notify_mayor(self, alert: Alert) -> None:
        """Notify the mayor of a critical alert."""
        from aragora.nomic.agent_roles import RoleBasedRouter

        router = RoleBasedRouter(self.hierarchy)
        mayor_id = await router.route_to_mayor()

        if mayor_id:
            logger.info(f"Notifying mayor {mayor_id} of critical alert: {alert.message}")
            # In a real implementation, this would send a message to the mayor agent

    async def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
    ) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: ID of alert to acknowledge
            acknowledged_by: Who is acknowledging

        Returns:
            True if acknowledged, False if not found
        """
        alert = self._alerts.get(alert_id)
        if not alert:
            return False

        alert.acknowledged = True
        alert.acknowledged_at = datetime.now(timezone.utc)
        alert.acknowledged_by = acknowledged_by

        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return True

    async def generate_health_report(self) -> HealthReport:
        """
        Generate a comprehensive health report.

        Returns:
            HealthReport with all health information
        """
        import uuid

        now = datetime.now(timezone.utc)
        agent_checks = []
        convoy_checks = []
        recommendations = []

        # Check all agents
        for role in [AgentRole.CREW, AgentRole.POLECAT, AgentRole.WITNESS]:
            agents = await self.hierarchy.get_agents_by_role(role)
            for agent in agents:
                health = await self.check_agent_health(agent.agent_id)
                agent_checks.append(health)

                if health.status == HealthStatus.UNHEALTHY:
                    recommendations.append(
                        f"Investigate agent {agent.agent_id}: {', '.join(health.issues)}"
                    )

        # Check all active convoys
        if self.convoy_manager:
            from aragora.nomic.convoys import ConvoyStatus

            convoys = await self.convoy_manager.list_convoys(status=ConvoyStatus.ACTIVE)
            for convoy in convoys:
                progress = await self.convoy_manager.get_convoy_progress(convoy.id)

                status = HealthStatus.HEALTHY
                issues = []

                if progress.failed_beads > 0:
                    issues.append(f"{progress.failed_beads} failed beads")
                    status = HealthStatus.DEGRADED

                if progress.running_beads == 0 and progress.pending_beads > 0:
                    issues.append("Convoy stalled")
                    status = HealthStatus.UNHEALTHY

                convoy_checks.append(
                    ConvoyHealthCheck(
                        convoy_id=convoy.id,
                        status=status,
                        checked_at=now,
                        total_beads=progress.total_beads,
                        completed_beads=progress.completed_beads,
                        failed_beads=progress.failed_beads,
                        stuck_beads=0,  # Would need per-bead tracking
                        completion_rate=progress.completion_percentage / 100.0,
                        issues=issues,
                    )
                )

        # Determine overall status
        critical_agents = len([c for c in agent_checks if c.status == HealthStatus.CRITICAL])
        unhealthy_agents = len([c for c in agent_checks if c.status == HealthStatus.UNHEALTHY])
        unhealthy_convoys = len([c for c in convoy_checks if c.status == HealthStatus.UNHEALTHY])

        if critical_agents > 0:
            overall_status = HealthStatus.CRITICAL
        elif unhealthy_agents > 0 or unhealthy_convoys > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif len([c for c in agent_checks if c.status == HealthStatus.DEGRADED]) > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        # Get recent alerts
        recent_alerts = [
            a for a in self._alerts.values() if (now - a.timestamp) < timedelta(hours=1)
        ]

        # Generate statistics
        statistics = {
            "total_agents": len(agent_checks),
            "healthy_agents": len([c for c in agent_checks if c.status == HealthStatus.HEALTHY]),
            "unhealthy_agents": unhealthy_agents,
            "critical_agents": critical_agents,
            "total_convoys": len(convoy_checks),
            "healthy_convoys": len([c for c in convoy_checks if c.status == HealthStatus.HEALTHY]),
            "alerts_last_hour": len(recent_alerts),
            "unacknowledged_alerts": len([a for a in recent_alerts if not a.acknowledged]),
        }

        report = HealthReport(
            report_id=str(uuid.uuid4()),
            generated_at=now,
            overall_status=overall_status,
            agent_checks=agent_checks,
            convoy_checks=convoy_checks,
            alerts=list(recent_alerts),
            statistics=statistics,
            recommendations=recommendations,
        )

        self._reports.append(report)

        # Keep only recent reports
        if len(self._reports) > 100:
            self._reports = self._reports[-100:]

        return report

    async def get_recent_alerts(
        self,
        hours: int = 24,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Alert]:
        """
        Get recent alerts.

        Args:
            hours: How far back to look
            severity: Optional severity filter

        Returns:
            List of alerts
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=hours)

        alerts = [a for a in self._alerts.values() if a.timestamp > cutoff]

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)

    async def get_statistics(self) -> Dict[str, Any]:
        """Get witness behavior statistics."""
        return {
            "total_alerts": len(self._alerts),
            "unacknowledged_alerts": len([a for a in self._alerts.values() if not a.acknowledged]),
            "reports_generated": len(self._reports),
            "patrol_running": self._running,
        }


# Singleton instance
_default_witness: Optional[WitnessBehavior] = None


async def get_witness_behavior(
    hierarchy: AgentHierarchy,
    convoy_manager: Optional["ConvoyManager"] = None,
    coordinator: Optional["ConvoyCoordinator"] = None,
    escalation_store: Optional["EscalationStore"] = None,
) -> WitnessBehavior:
    """Get the default witness behavior instance."""
    global _default_witness
    if _default_witness is None:
        _default_witness = WitnessBehavior(
            hierarchy=hierarchy,
            convoy_manager=convoy_manager,
            coordinator=coordinator,
            escalation_store=escalation_store,
        )
    return _default_witness


def reset_witness_behavior() -> None:
    """Reset the default witness (for testing)."""
    global _default_witness
    _default_witness = None
