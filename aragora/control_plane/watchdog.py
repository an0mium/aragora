"""
Three-Tier Watchdog System.

Implements the Gastown three-tier watchdog pattern for multi-level
agent monitoring and escalation:

Tier 1 - MECHANICAL: Heartbeat, memory, circuit breaker checks
Tier 2 - BOOT_AGENT: Response quality, latency, semantic checks
Tier 3 - DEACON: SLA compliance, cross-agent coordination, global policy

Each tier monitors different aspects and can escalate to the next
tier when issues exceed local resolution capabilities.

Usage:
    watchdog = ThreeTierWatchdog()

    # Configure tiers
    watchdog.configure_tier(WatchdogConfig(
        tier=WatchdogTier.MECHANICAL,
        heartbeat_interval=5.0,
        timeout_threshold=30.0,
    ))

    # Start monitoring
    await watchdog.start()

    # Escalate issues
    await watchdog.escalate(
        source_tier=WatchdogTier.MECHANICAL,
        issue=WatchdogIssue(
            severity=IssueSeverity.WARNING,
            agent="claude-opus",
            message="Heartbeat timeout",
        ),
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class WatchdogTier(str, Enum):
    """The three tiers of the watchdog system."""

    MECHANICAL = "mechanical"  # Tier 1: Heartbeat, memory, circuit breaker
    BOOT_AGENT = "boot_agent"  # Tier 2: Response quality, latency
    DEACON = "deacon"  # Tier 3: SLA, cross-agent coordination


class IssueSeverity(IntEnum):
    """Severity levels for watchdog issues."""

    INFO = 0  # Informational, no action needed
    WARNING = 1  # Potential issue, monitor closely
    ERROR = 2  # Active issue, needs attention
    CRITICAL = 3  # Severe issue, immediate action required


class IssueCategory(str, Enum):
    """Categories of issues detected by the watchdog."""

    # Mechanical (Tier 1)
    HEARTBEAT_MISSING = "heartbeat_missing"
    MEMORY_EXCEEDED = "memory_exceeded"
    CIRCUIT_OPEN = "circuit_open"
    RESOURCE_EXHAUSTED = "resource_exhausted"

    # Boot Agent (Tier 2)
    LATENCY_EXCEEDED = "latency_exceeded"
    RESPONSE_QUALITY_LOW = "response_quality_low"
    ERROR_RATE_HIGH = "error_rate_high"
    SEMANTIC_DRIFT = "semantic_drift"

    # Deacon (Tier 3)
    SLA_VIOLATION = "sla_violation"
    COORDINATION_FAILURE = "coordination_failure"
    POLICY_VIOLATION = "policy_violation"
    CONSENSUS_BLOCKED = "consensus_blocked"


@dataclass
class WatchdogIssue:
    """An issue detected by the watchdog."""

    severity: IssueSeverity
    category: IssueCategory
    agent: Optional[str]
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    # Tracking
    id: str = field(default_factory=lambda: f"issue-{int(time.time() * 1000) % 1000000}")
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    detected_by: Optional[WatchdogTier] = None

    # Resolution tracking
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "severity": self.severity.name,
            "category": self.category.value,
            "agent": self.agent,
            "message": self.message,
            "details": self.details,
            "detected_at": self.detected_at.isoformat(),
            "detected_by": self.detected_by.value if self.detected_by else None,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_notes": self.resolution_notes,
        }


@dataclass
class EscalationResult:
    """Result of an escalation attempt."""

    issue_id: str
    escalated_to: WatchdogTier
    accepted: bool
    action_taken: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class WatchdogConfig:
    """Configuration for a watchdog tier."""

    tier: WatchdogTier

    # Timing
    check_interval_seconds: float = 5.0
    heartbeat_timeout_seconds: float = 30.0

    # Thresholds
    memory_warning_mb: float = 1024.0
    memory_critical_mb: float = 2048.0
    latency_warning_ms: float = 5000.0
    latency_critical_ms: float = 15000.0
    error_rate_warning: float = 0.1  # 10%
    error_rate_critical: float = 0.3  # 30%

    # Escalation
    auto_escalate: bool = True
    escalation_threshold: int = 3  # Issues before escalating

    # SLA (Deacon tier)
    sla_response_time_ms: float = 10000.0
    sla_availability_pct: float = 99.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "check_interval_seconds": self.check_interval_seconds,
            "heartbeat_timeout_seconds": self.heartbeat_timeout_seconds,
            "memory_warning_mb": self.memory_warning_mb,
            "memory_critical_mb": self.memory_critical_mb,
            "latency_warning_ms": self.latency_warning_ms,
            "latency_critical_ms": self.latency_critical_ms,
            "error_rate_warning": self.error_rate_warning,
            "error_rate_critical": self.error_rate_critical,
            "auto_escalate": self.auto_escalate,
            "escalation_threshold": self.escalation_threshold,
        }


@dataclass
class AgentHealth:
    """Health state tracked for an agent."""

    agent_name: str
    last_heartbeat: Optional[datetime] = None
    consecutive_failures: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    memory_usage_mb: float = 0.0
    circuit_breaker_state: str = "closed"
    active_issues: List[WatchdogIssue] = field(default_factory=list)

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def average_latency_ms(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    def record_request(self, success: bool, latency_ms: float) -> None:
        self.total_requests += 1
        self.total_latency_ms += latency_ms
        if not success:
            self.failed_requests += 1
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0

    def record_heartbeat(self) -> None:
        self.last_heartbeat = datetime.now(timezone.utc)


class ThreeTierWatchdog:
    """
    Three-tier watchdog for multi-level agent monitoring.

    Tier 1 (MECHANICAL): Low-level health checks
        - Heartbeat monitoring
        - Memory usage tracking
        - Circuit breaker state

    Tier 2 (BOOT_AGENT): Quality and performance
        - Response latency
        - Error rates
        - Semantic quality scoring

    Tier 3 (DEACON): Business-level oversight
        - SLA compliance
        - Cross-agent coordination
        - Global policy enforcement
    """

    def __init__(self):
        """Initialize the three-tier watchdog."""
        self._configs: Dict[WatchdogTier, WatchdogConfig] = {
            WatchdogTier.MECHANICAL: WatchdogConfig(tier=WatchdogTier.MECHANICAL),
            WatchdogTier.BOOT_AGENT: WatchdogConfig(tier=WatchdogTier.BOOT_AGENT),
            WatchdogTier.DEACON: WatchdogConfig(tier=WatchdogTier.DEACON),
        }
        self._agent_health: Dict[str, AgentHealth] = {}
        self._active_issues: Dict[str, WatchdogIssue] = {}
        self._escalation_counts: Dict[WatchdogTier, int] = defaultdict(int)
        self._handlers: Dict[WatchdogTier, List[Callable]] = defaultdict(list)
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "issues_detected": 0,
            "issues_resolved": 0,
            "escalations": 0,
            "tier_checks": {tier.value: 0 for tier in WatchdogTier},
        }

    def configure_tier(self, config: WatchdogConfig) -> None:
        """
        Configure a watchdog tier.

        Args:
            config: Configuration for the tier
        """
        self._configs[config.tier] = config
        logger.info(f"Configured watchdog tier: {config.tier.value}")

    def register_agent(self, agent_name: str) -> None:
        """Register an agent for monitoring."""
        if agent_name not in self._agent_health:
            self._agent_health[agent_name] = AgentHealth(agent_name=agent_name)
            logger.debug(f"Registered agent for watchdog monitoring: {agent_name}")

    def unregister_agent(self, agent_name: str) -> None:
        """Unregister an agent from monitoring."""
        if agent_name in self._agent_health:
            del self._agent_health[agent_name]

    def register_handler(
        self,
        tier: WatchdogTier,
        handler: Callable[[WatchdogIssue], Any],
    ) -> Callable[[], None]:
        """
        Register an issue handler for a tier.

        Args:
            tier: The tier to handle issues from
            handler: Callback function for issues

        Returns:
            Unregister function
        """
        self._handlers[tier].append(handler)

        def unregister():
            if handler in self._handlers[tier]:
                self._handlers[tier].remove(handler)

        return unregister

    async def start(self) -> None:
        """Start the watchdog monitoring loops."""
        if self._running:
            return

        self._running = True

        # Start tier monitoring tasks
        for tier in WatchdogTier:
            config = self._configs[tier]
            task = asyncio.create_task(
                self._run_tier_loop(tier, config),
                name=f"watchdog_{tier.value}",
            )
            self._tasks.append(task)

        logger.info("Three-tier watchdog started")

    async def stop(self) -> None:
        """Stop the watchdog monitoring loops."""
        self._running = False

        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        logger.info("Three-tier watchdog stopped")

    async def _run_tier_loop(
        self,
        tier: WatchdogTier,
        config: WatchdogConfig,
    ) -> None:
        """Run the monitoring loop for a tier."""
        while self._running:
            try:
                await self._check_tier(tier, config)
                self._stats["tier_checks"][tier.value] += 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Watchdog tier {tier.value} check failed: {e}")

            await asyncio.sleep(config.check_interval_seconds)

    async def _check_tier(
        self,
        tier: WatchdogTier,
        config: WatchdogConfig,
    ) -> List[WatchdogIssue]:
        """
        Run checks for a specific tier.

        Args:
            tier: The tier to check
            config: Tier configuration

        Returns:
            List of detected issues
        """
        issues: List[WatchdogIssue] = []

        if tier == WatchdogTier.MECHANICAL:
            issues.extend(await self._check_mechanical(config))
        elif tier == WatchdogTier.BOOT_AGENT:
            issues.extend(await self._check_boot_agent(config))
        elif tier == WatchdogTier.DEACON:
            issues.extend(await self._check_deacon(config))

        # Process detected issues
        for issue in issues:
            issue.detected_by = tier
            await self._handle_issue(tier, issue)

        return issues

    async def _check_mechanical(
        self,
        config: WatchdogConfig,
    ) -> List[WatchdogIssue]:
        """Tier 1: Mechanical checks."""
        issues: List[WatchdogIssue] = []
        now = datetime.now(timezone.utc)

        for agent_name, health in self._agent_health.items():
            # Check heartbeat
            if health.last_heartbeat:
                elapsed = (now - health.last_heartbeat).total_seconds()
                if elapsed > config.heartbeat_timeout_seconds:
                    issues.append(
                        WatchdogIssue(
                            severity=IssueSeverity.ERROR,
                            category=IssueCategory.HEARTBEAT_MISSING,
                            agent=agent_name,
                            message=f"No heartbeat for {elapsed:.1f}s",
                            details={"elapsed_seconds": elapsed},
                        )
                    )

            # Check memory
            if health.memory_usage_mb > config.memory_critical_mb:
                issues.append(
                    WatchdogIssue(
                        severity=IssueSeverity.CRITICAL,
                        category=IssueCategory.MEMORY_EXCEEDED,
                        agent=agent_name,
                        message=f"Memory critical: {health.memory_usage_mb:.1f}MB",
                        details={"memory_mb": health.memory_usage_mb},
                    )
                )
            elif health.memory_usage_mb > config.memory_warning_mb:
                issues.append(
                    WatchdogIssue(
                        severity=IssueSeverity.WARNING,
                        category=IssueCategory.MEMORY_EXCEEDED,
                        agent=agent_name,
                        message=f"Memory warning: {health.memory_usage_mb:.1f}MB",
                        details={"memory_mb": health.memory_usage_mb},
                    )
                )

            # Check circuit breaker
            if health.circuit_breaker_state == "open":
                issues.append(
                    WatchdogIssue(
                        severity=IssueSeverity.ERROR,
                        category=IssueCategory.CIRCUIT_OPEN,
                        agent=agent_name,
                        message="Circuit breaker is open",
                    )
                )

        return issues

    async def _check_boot_agent(
        self,
        config: WatchdogConfig,
    ) -> List[WatchdogIssue]:
        """Tier 2: Boot agent quality checks."""
        issues: List[WatchdogIssue] = []

        for agent_name, health in self._agent_health.items():
            # Check latency
            avg_latency = health.average_latency_ms
            if avg_latency > config.latency_critical_ms:
                issues.append(
                    WatchdogIssue(
                        severity=IssueSeverity.CRITICAL,
                        category=IssueCategory.LATENCY_EXCEEDED,
                        agent=agent_name,
                        message=f"Latency critical: {avg_latency:.0f}ms",
                        details={"latency_ms": avg_latency},
                    )
                )
            elif avg_latency > config.latency_warning_ms:
                issues.append(
                    WatchdogIssue(
                        severity=IssueSeverity.WARNING,
                        category=IssueCategory.LATENCY_EXCEEDED,
                        agent=agent_name,
                        message=f"Latency warning: {avg_latency:.0f}ms",
                        details={"latency_ms": avg_latency},
                    )
                )

            # Check error rate
            error_rate = health.error_rate
            if error_rate > config.error_rate_critical:
                issues.append(
                    WatchdogIssue(
                        severity=IssueSeverity.CRITICAL,
                        category=IssueCategory.ERROR_RATE_HIGH,
                        agent=agent_name,
                        message=f"Error rate critical: {error_rate * 100:.1f}%",
                        details={"error_rate": error_rate},
                    )
                )
            elif error_rate > config.error_rate_warning:
                issues.append(
                    WatchdogIssue(
                        severity=IssueSeverity.WARNING,
                        category=IssueCategory.ERROR_RATE_HIGH,
                        agent=agent_name,
                        message=f"Error rate warning: {error_rate * 100:.1f}%",
                        details={"error_rate": error_rate},
                    )
                )

        return issues

    async def _check_deacon(
        self,
        config: WatchdogConfig,
    ) -> List[WatchdogIssue]:
        """Tier 3: Deacon SLA and coordination checks."""
        issues: List[WatchdogIssue] = []

        # Check overall SLA compliance
        total_requests = sum(h.total_requests for h in self._agent_health.values())
        total_failures = sum(h.failed_requests for h in self._agent_health.values())

        if total_requests > 0:
            success_rate = (total_requests - total_failures) / total_requests * 100
            if success_rate < config.sla_availability_pct:
                issues.append(
                    WatchdogIssue(
                        severity=IssueSeverity.CRITICAL,
                        category=IssueCategory.SLA_VIOLATION,
                        agent=None,
                        message=f"SLA availability violated: {success_rate:.1f}%",
                        details={
                            "availability_pct": success_rate,
                            "sla_target_pct": config.sla_availability_pct,
                        },
                    )
                )

        # Check for coordination issues
        critical_agents = [
            name
            for name, health in self._agent_health.items()
            if health.circuit_breaker_state == "open" or health.consecutive_failures >= 5
        ]

        if len(critical_agents) > len(self._agent_health) / 2:
            issues.append(
                WatchdogIssue(
                    severity=IssueSeverity.CRITICAL,
                    category=IssueCategory.COORDINATION_FAILURE,
                    agent=None,
                    message=f"Majority of agents unhealthy: {critical_agents}",
                    details={"unhealthy_agents": critical_agents},
                )
            )

        return issues

    async def _handle_issue(
        self,
        tier: WatchdogTier,
        issue: WatchdogIssue,
    ) -> None:
        """Handle a detected issue."""
        async with self._lock:
            self._active_issues[issue.id] = issue
            self._stats["issues_detected"] += 1

            # Get agent health and add to active issues
            if issue.agent and issue.agent in self._agent_health:
                self._agent_health[issue.agent].active_issues.append(issue)

        # Call registered handlers
        for handler in self._handlers[tier]:
            try:
                result = handler(issue)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"Issue handler failed: {e}")

        # Auto-escalate if configured
        config = self._configs[tier]
        if config.auto_escalate and issue.severity >= IssueSeverity.ERROR:
            self._escalation_counts[tier] += 1

            if self._escalation_counts[tier] >= config.escalation_threshold:
                await self.escalate(tier, issue)
                self._escalation_counts[tier] = 0

        logger.info(
            f"Watchdog issue detected [{tier.value}]: "
            f"{issue.severity.name} - {issue.category.value}: {issue.message}"
        )

    async def escalate(
        self,
        source_tier: WatchdogTier,
        issue: WatchdogIssue,
    ) -> EscalationResult:
        """
        Escalate an issue to the next tier.

        Args:
            source_tier: The tier escalating the issue
            issue: The issue to escalate

        Returns:
            EscalationResult with outcome
        """
        # Determine target tier
        tier_order = [WatchdogTier.MECHANICAL, WatchdogTier.BOOT_AGENT, WatchdogTier.DEACON]
        current_index = tier_order.index(source_tier)

        if current_index >= len(tier_order) - 1:
            # Already at highest tier
            logger.warning(f"Cannot escalate from {source_tier.value} - already at highest tier")
            return EscalationResult(
                issue_id=issue.id,
                escalated_to=source_tier,
                accepted=False,
                error_message="Already at highest tier",
            )

        target_tier = tier_order[current_index + 1]
        self._stats["escalations"] += 1

        logger.info(f"Escalating issue {issue.id} from {source_tier.value} to {target_tier.value}")

        # Call target tier handlers
        for handler in self._handlers[target_tier]:
            try:
                result = handler(issue)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.warning(f"Escalation handler failed: {e}")

        return EscalationResult(
            issue_id=issue.id,
            escalated_to=target_tier,
            accepted=True,
            action_taken=f"Escalated to {target_tier.value} handlers",
        )

    def resolve_issue(
        self,
        issue_id: str,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Mark an issue as resolved.

        Args:
            issue_id: The issue ID
            notes: Optional resolution notes

        Returns:
            True if issue was found and resolved
        """
        if issue_id not in self._active_issues:
            return False

        issue = self._active_issues[issue_id]
        issue.resolved = True
        issue.resolved_at = datetime.now(timezone.utc)
        issue.resolution_notes = notes

        self._stats["issues_resolved"] += 1

        # Remove from agent's active issues
        if issue.agent and issue.agent in self._agent_health:
            health = self._agent_health[issue.agent]
            health.active_issues = [i for i in health.active_issues if i.id != issue_id]

        logger.info(f"Issue resolved: {issue_id}")
        return True

    def record_heartbeat(self, agent_name: str) -> None:
        """Record a heartbeat from an agent."""
        if agent_name not in self._agent_health:
            self.register_agent(agent_name)
        self._agent_health[agent_name].record_heartbeat()

    def record_request(
        self,
        agent_name: str,
        success: bool,
        latency_ms: float,
    ) -> None:
        """Record a request to an agent."""
        if agent_name not in self._agent_health:
            self.register_agent(agent_name)
        self._agent_health[agent_name].record_request(success, latency_ms)

    def update_memory_usage(self, agent_name: str, memory_mb: float) -> None:
        """Update memory usage for an agent."""
        if agent_name not in self._agent_health:
            self.register_agent(agent_name)
        self._agent_health[agent_name].memory_usage_mb = memory_mb

    def update_circuit_breaker(self, agent_name: str, state: str) -> None:
        """Update circuit breaker state for an agent."""
        if agent_name not in self._agent_health:
            self.register_agent(agent_name)
        self._agent_health[agent_name].circuit_breaker_state = state

    def get_agent_health(self, agent_name: str) -> Optional[AgentHealth]:
        """Get health state for an agent."""
        return self._agent_health.get(agent_name)

    def get_all_health(self) -> Dict[str, AgentHealth]:
        """Get health state for all agents."""
        return dict(self._agent_health)

    def get_active_issues(
        self,
        severity: Optional[IssueSeverity] = None,
        agent: Optional[str] = None,
    ) -> List[WatchdogIssue]:
        """
        Get active (unresolved) issues.

        Args:
            severity: Filter by minimum severity
            agent: Filter by agent name

        Returns:
            List of active issues
        """
        issues = [i for i in self._active_issues.values() if not i.resolved]

        if severity is not None:
            issues = [i for i in issues if i.severity >= severity]

        if agent is not None:
            issues = [i for i in issues if i.agent == agent]

        return issues

    def get_stats(self) -> Dict[str, Any]:
        """Get watchdog statistics."""
        return {
            **self._stats,
            "active_issues": len([i for i in self._active_issues.values() if not i.resolved]),
            "monitored_agents": len(self._agent_health),
            "is_running": self._running,
        }


# Global watchdog singleton
_default_watchdog: Optional[ThreeTierWatchdog] = None


def get_watchdog() -> ThreeTierWatchdog:
    """Get the default watchdog instance."""
    global _default_watchdog
    if _default_watchdog is None:
        _default_watchdog = ThreeTierWatchdog()
    return _default_watchdog


def reset_watchdog() -> None:
    """Reset the default watchdog (for testing)."""
    global _default_watchdog
    if _default_watchdog:
        asyncio.create_task(_default_watchdog.stop())
    _default_watchdog = None
