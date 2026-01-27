"""
End-to-End Integration Tests: Watchdog ↔ Control Plane.

Tests the integration between the Three-Tier Watchdog and Control Plane:
1. Agent registration triggers watchdog registration
2. Heartbeat forwarding and monitoring
3. Issue escalation and handling
4. Statistics aggregation
"""

import pytest
import asyncio
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.control_plane.watchdog import (
    ThreeTierWatchdog,
    WatchdogConfig,
    WatchdogTier,
    WatchdogIssue,
    IssueSeverity,
    IssueCategory,
    AgentHealth,
    reset_watchdog,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def watchdog():
    """Create a fresh watchdog for each test."""
    reset_watchdog()
    wd = ThreeTierWatchdog()
    yield wd
    # Cleanup
    asyncio.get_event_loop().run_until_complete(wd.stop())


@pytest.fixture
def strict_config():
    """Create strict monitoring configuration."""
    return {
        WatchdogTier.MECHANICAL: WatchdogConfig(
            tier=WatchdogTier.MECHANICAL,
            check_interval_seconds=1.0,
            heartbeat_timeout_seconds=5.0,
            memory_warning_mb=512.0,
            memory_critical_mb=1024.0,
            auto_escalate=True,
            escalation_threshold=2,
        ),
        WatchdogTier.BOOT_AGENT: WatchdogConfig(
            tier=WatchdogTier.BOOT_AGENT,
            check_interval_seconds=2.0,
            latency_warning_ms=1000.0,
            latency_critical_ms=3000.0,
            error_rate_warning=0.05,
            error_rate_critical=0.15,
        ),
        WatchdogTier.DEACON: WatchdogConfig(
            tier=WatchdogTier.DEACON,
            check_interval_seconds=5.0,
            sla_availability_pct=99.0,
            sla_response_time_ms=5000.0,
        ),
    }


@pytest.fixture
def mock_control_plane():
    """Create a mock control plane coordinator."""
    coordinator = MagicMock()
    coordinator._agents = {}
    coordinator._tasks = {}

    def mock_register(agent_name, **kwargs):
        coordinator._agents[agent_name] = {
            "name": agent_name,
            "registered_at": datetime.now(timezone.utc),
            "status": "healthy",
            **kwargs,
        }
        return True

    def mock_unregister(agent_name):
        if agent_name in coordinator._agents:
            del coordinator._agents[agent_name]
            return True
        return False

    coordinator.register_agent = MagicMock(side_effect=mock_register)
    coordinator.unregister_agent = MagicMock(side_effect=mock_unregister)
    coordinator.get_agent = lambda name: coordinator._agents.get(name)
    coordinator.list_agents = lambda: list(coordinator._agents.values())

    return coordinator


# ============================================================================
# Integration Tests
# ============================================================================


class TestWatchdogAgentRegistration:
    """Tests for watchdog agent registration."""

    def test_agent_registration(self, watchdog):
        """Test that agents can be registered with watchdog."""
        watchdog.register_agent("claude-opus")
        watchdog.register_agent("gpt-4o")
        watchdog.register_agent("gemini-pro")

        health = watchdog.get_all_health()
        assert len(health) == 3
        assert "claude-opus" in health
        assert "gpt-4o" in health
        assert "gemini-pro" in health

    def test_agent_unregistration(self, watchdog):
        """Test that agents can be unregistered."""
        watchdog.register_agent("claude-opus")
        watchdog.register_agent("gpt-4o")

        watchdog.unregister_agent("claude-opus")

        health = watchdog.get_all_health()
        assert len(health) == 1
        assert "gpt-4o" in health
        assert "claude-opus" not in health

    def test_control_plane_triggers_watchdog_registration(self, watchdog, mock_control_plane):
        """Test that control plane registration triggers watchdog registration."""

        # Simulate control plane + watchdog integration
        def on_agent_registered(agent_name):
            watchdog.register_agent(agent_name)

        # Register agents via control plane
        mock_control_plane.register_agent("claude-opus")
        on_agent_registered("claude-opus")

        mock_control_plane.register_agent("gpt-4o")
        on_agent_registered("gpt-4o")

        # Both should be in watchdog
        assert watchdog.get_agent_health("claude-opus") is not None
        assert watchdog.get_agent_health("gpt-4o") is not None


class TestWatchdogHeartbeatMonitoring:
    """Tests for heartbeat monitoring."""

    def test_heartbeat_recording(self, watchdog):
        """Test that heartbeats are recorded."""
        watchdog.register_agent("claude-opus")

        # Record heartbeat
        watchdog.record_heartbeat("claude-opus")

        health = watchdog.get_agent_health("claude-opus")
        assert health.last_heartbeat is not None
        assert (datetime.now(timezone.utc) - health.last_heartbeat).seconds < 2

    @pytest.mark.asyncio
    async def test_missing_heartbeat_detection(self, watchdog, strict_config):
        """Test detection of missing heartbeats."""
        watchdog.configure_tier(strict_config[WatchdogTier.MECHANICAL])
        watchdog.register_agent("claude-opus")

        # Record heartbeat, then wait
        watchdog.record_heartbeat("claude-opus")

        # Manually set last heartbeat to old time
        health = watchdog.get_agent_health("claude-opus")
        health.last_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=10)

        # Run mechanical check
        config = watchdog._configs[WatchdogTier.MECHANICAL]
        issues = await watchdog._check_mechanical(config)

        # Should detect missing heartbeat
        heartbeat_issues = [i for i in issues if i.category == IssueCategory.HEARTBEAT_MISSING]
        assert len(heartbeat_issues) == 1
        assert heartbeat_issues[0].agent == "claude-opus"

    def test_auto_registration_on_heartbeat(self, watchdog):
        """Test that heartbeat auto-registers unknown agents."""
        # Don't pre-register
        watchdog.record_heartbeat("new-agent")

        health = watchdog.get_agent_health("new-agent")
        assert health is not None
        assert health.last_heartbeat is not None


class TestWatchdogRequestTracking:
    """Tests for request tracking and metrics."""

    def test_request_recording(self, watchdog):
        """Test that requests are recorded correctly."""
        watchdog.register_agent("claude-opus")

        # Record successful requests
        watchdog.record_request("claude-opus", success=True, latency_ms=150.0)
        watchdog.record_request("claude-opus", success=True, latency_ms=200.0)
        watchdog.record_request("claude-opus", success=False, latency_ms=500.0)

        health = watchdog.get_agent_health("claude-opus")
        assert health.total_requests == 3
        assert health.failed_requests == 1
        assert health.error_rate == pytest.approx(1 / 3, rel=0.01)
        assert health.average_latency_ms == pytest.approx(283.33, rel=0.01)

    def test_consecutive_failure_tracking(self, watchdog):
        """Test that consecutive failures are tracked."""
        watchdog.register_agent("claude-opus")

        # Successful requests reset consecutive failures
        watchdog.record_request("claude-opus", success=True, latency_ms=100.0)
        watchdog.record_request("claude-opus", success=True, latency_ms=100.0)

        health = watchdog.get_agent_health("claude-opus")
        assert health.consecutive_failures == 0

        # Failed requests increment
        watchdog.record_request("claude-opus", success=False, latency_ms=100.0)
        watchdog.record_request("claude-opus", success=False, latency_ms=100.0)

        health = watchdog.get_agent_health("claude-opus")
        assert health.consecutive_failures == 2

        # Success resets
        watchdog.record_request("claude-opus", success=True, latency_ms=100.0)

        health = watchdog.get_agent_health("claude-opus")
        assert health.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_high_error_rate_detection(self, watchdog, strict_config):
        """Test detection of high error rates."""
        watchdog.configure_tier(strict_config[WatchdogTier.BOOT_AGENT])
        watchdog.register_agent("claude-opus")

        # Record high error rate (30% > 15% critical threshold)
        for _ in range(7):
            watchdog.record_request("claude-opus", success=True, latency_ms=100.0)
        for _ in range(3):
            watchdog.record_request("claude-opus", success=False, latency_ms=100.0)

        config = watchdog._configs[WatchdogTier.BOOT_AGENT]
        issues = await watchdog._check_boot_agent(config)

        error_issues = [i for i in issues if i.category == IssueCategory.ERROR_RATE_HIGH]
        assert len(error_issues) == 1
        assert error_issues[0].severity == IssueSeverity.CRITICAL


class TestWatchdogMemoryMonitoring:
    """Tests for memory monitoring."""

    def test_memory_usage_update(self, watchdog):
        """Test updating memory usage."""
        watchdog.register_agent("claude-opus")

        watchdog.update_memory_usage("claude-opus", memory_mb=512.0)

        health = watchdog.get_agent_health("claude-opus")
        assert health.memory_usage_mb == 512.0

    @pytest.mark.asyncio
    async def test_memory_threshold_detection(self, watchdog, strict_config):
        """Test detection of memory threshold violations."""
        watchdog.configure_tier(strict_config[WatchdogTier.MECHANICAL])
        watchdog.register_agent("claude-opus")
        watchdog.register_agent("gpt-4o")

        # Set memory usage
        watchdog.update_memory_usage("claude-opus", memory_mb=600.0)  # Warning
        watchdog.update_memory_usage("gpt-4o", memory_mb=1200.0)  # Critical

        config = watchdog._configs[WatchdogTier.MECHANICAL]
        issues = await watchdog._check_mechanical(config)

        memory_issues = [i for i in issues if i.category == IssueCategory.MEMORY_EXCEEDED]

        # Should have warning for claude and critical for gpt
        warning_issues = [i for i in memory_issues if i.severity == IssueSeverity.WARNING]
        critical_issues = [i for i in memory_issues if i.severity == IssueSeverity.CRITICAL]

        assert len(warning_issues) == 1
        assert warning_issues[0].agent == "claude-opus"
        assert len(critical_issues) == 1
        assert critical_issues[0].agent == "gpt-4o"


class TestWatchdogCircuitBreaker:
    """Tests for circuit breaker state tracking."""

    def test_circuit_breaker_update(self, watchdog):
        """Test updating circuit breaker state."""
        watchdog.register_agent("claude-opus")

        watchdog.update_circuit_breaker("claude-opus", state="closed")
        health = watchdog.get_agent_health("claude-opus")
        assert health.circuit_breaker_state == "closed"

        watchdog.update_circuit_breaker("claude-opus", state="open")
        health = watchdog.get_agent_health("claude-opus")
        assert health.circuit_breaker_state == "open"

    @pytest.mark.asyncio
    async def test_open_circuit_detection(self, watchdog, strict_config):
        """Test detection of open circuit breakers."""
        watchdog.configure_tier(strict_config[WatchdogTier.MECHANICAL])
        watchdog.register_agent("claude-opus")

        watchdog.update_circuit_breaker("claude-opus", state="open")

        config = watchdog._configs[WatchdogTier.MECHANICAL]
        issues = await watchdog._check_mechanical(config)

        circuit_issues = [i for i in issues if i.category == IssueCategory.CIRCUIT_OPEN]
        assert len(circuit_issues) == 1
        assert circuit_issues[0].agent == "claude-opus"


class TestWatchdogEscalation:
    """Tests for issue escalation between tiers."""

    @pytest.mark.asyncio
    async def test_escalation_to_next_tier(self, watchdog):
        """Test that issues can be escalated to the next tier."""
        escalated_issues = []

        def boot_agent_handler(issue):
            escalated_issues.append(issue)

        watchdog.register_handler(WatchdogTier.BOOT_AGENT, boot_agent_handler)

        issue = WatchdogIssue(
            severity=IssueSeverity.ERROR,
            category=IssueCategory.HEARTBEAT_MISSING,
            agent="claude-opus",
            message="Agent unresponsive",
        )

        result = await watchdog.escalate(WatchdogTier.MECHANICAL, issue)

        assert result.accepted
        assert result.escalated_to == WatchdogTier.BOOT_AGENT
        assert len(escalated_issues) == 1

    @pytest.mark.asyncio
    async def test_escalation_from_highest_tier_fails(self, watchdog):
        """Test that escalation from DEACON (highest) tier is rejected."""
        issue = WatchdogIssue(
            severity=IssueSeverity.CRITICAL,
            category=IssueCategory.SLA_VIOLATION,
            agent=None,
            message="SLA violated",
        )

        result = await watchdog.escalate(WatchdogTier.DEACON, issue)

        assert not result.accepted
        assert "highest tier" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_multi_tier_escalation(self, watchdog):
        """Test escalation through multiple tiers."""
        tier_chain = []

        def mechanical_handler(issue):
            tier_chain.append("mechanical")

        def boot_agent_handler(issue):
            tier_chain.append("boot_agent")

        def deacon_handler(issue):
            tier_chain.append("deacon")

        watchdog.register_handler(WatchdogTier.MECHANICAL, mechanical_handler)
        watchdog.register_handler(WatchdogTier.BOOT_AGENT, boot_agent_handler)
        watchdog.register_handler(WatchdogTier.DEACON, deacon_handler)

        issue = WatchdogIssue(
            severity=IssueSeverity.CRITICAL,
            category=IssueCategory.HEARTBEAT_MISSING,
            agent="claude-opus",
            message="Critical issue",
        )

        # Escalate MECHANICAL -> BOOT_AGENT
        await watchdog.escalate(WatchdogTier.MECHANICAL, issue)

        # Escalate BOOT_AGENT -> DEACON
        await watchdog.escalate(WatchdogTier.BOOT_AGENT, issue)

        assert tier_chain == ["boot_agent", "deacon"]


class TestWatchdogIssueManagement:
    """Tests for issue tracking and resolution."""

    @pytest.mark.asyncio
    async def test_issue_detection_and_tracking(self, watchdog, strict_config):
        """Test that issues are detected and tracked."""
        watchdog.configure_tier(strict_config[WatchdogTier.MECHANICAL])
        watchdog.register_agent("claude-opus")

        # Create condition for issue
        health = watchdog.get_agent_health("claude-opus")
        health.last_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=10)

        # Handle the issue
        async def issue_handler(issue):
            pass

        watchdog.register_handler(WatchdogTier.MECHANICAL, issue_handler)

        config = watchdog._configs[WatchdogTier.MECHANICAL]
        issues = await watchdog._check_mechanical(config)

        for issue in issues:
            await watchdog._handle_issue(WatchdogTier.MECHANICAL, issue)

        # Should have active issue
        active = watchdog.get_active_issues()
        assert len(active) > 0

    def test_issue_resolution(self, watchdog):
        """Test marking issues as resolved."""
        watchdog.register_agent("claude-opus")

        # Create a test issue
        issue = WatchdogIssue(
            severity=IssueSeverity.ERROR,
            category=IssueCategory.HEARTBEAT_MISSING,
            agent="claude-opus",
            message="Test issue",
        )

        # Add to active issues
        watchdog._active_issues[issue.id] = issue

        # Resolve it
        result = watchdog.resolve_issue(issue.id, notes="Agent restarted")

        assert result
        assert issue.resolved
        assert issue.resolution_notes == "Agent restarted"
        assert issue.resolved_at is not None

    def test_get_active_issues_filtered(self, watchdog):
        """Test filtering active issues."""
        # Create various issues
        issues = [
            WatchdogIssue(
                severity=IssueSeverity.WARNING,
                category=IssueCategory.MEMORY_EXCEEDED,
                agent="claude-opus",
                message="Memory warning",
            ),
            WatchdogIssue(
                severity=IssueSeverity.ERROR,
                category=IssueCategory.HEARTBEAT_MISSING,
                agent="claude-opus",
                message="Heartbeat missing",
            ),
            WatchdogIssue(
                severity=IssueSeverity.CRITICAL,
                category=IssueCategory.SLA_VIOLATION,
                agent=None,
                message="SLA violation",
            ),
        ]

        for issue in issues:
            watchdog._active_issues[issue.id] = issue

        # Filter by severity
        critical_only = watchdog.get_active_issues(severity=IssueSeverity.CRITICAL)
        assert len(critical_only) == 1
        assert critical_only[0].severity == IssueSeverity.CRITICAL

        # Filter by agent
        claude_issues = watchdog.get_active_issues(agent="claude-opus")
        assert len(claude_issues) == 2
        assert all(i.agent == "claude-opus" for i in claude_issues)


class TestWatchdogSLAMonitoring:
    """Tests for SLA monitoring (DEACON tier)."""

    @pytest.mark.asyncio
    async def test_sla_violation_detection(self, watchdog, strict_config):
        """Test detection of SLA violations."""
        watchdog.configure_tier(strict_config[WatchdogTier.DEACON])

        # Register agents with poor success rates
        watchdog.register_agent("claude-opus")
        watchdog.register_agent("gpt-4o")

        # Record failures to bring success rate below SLA (99%)
        for _ in range(95):
            watchdog.record_request("claude-opus", success=True, latency_ms=100.0)
        for _ in range(10):
            watchdog.record_request("claude-opus", success=False, latency_ms=100.0)

        config = watchdog._configs[WatchdogTier.DEACON]
        issues = await watchdog._check_deacon(config)

        sla_issues = [i for i in issues if i.category == IssueCategory.SLA_VIOLATION]
        assert len(sla_issues) == 1
        assert sla_issues[0].severity == IssueSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_coordination_failure_detection(self, watchdog, strict_config):
        """Test detection of coordination failures."""
        watchdog.configure_tier(strict_config[WatchdogTier.DEACON])

        # Register multiple agents
        for name in ["agent1", "agent2", "agent3", "agent4"]:
            watchdog.register_agent(name)

        # Make majority unhealthy
        watchdog.update_circuit_breaker("agent1", state="open")
        watchdog.update_circuit_breaker("agent2", state="open")
        watchdog.update_circuit_breaker("agent3", state="open")
        # agent4 remains healthy

        config = watchdog._configs[WatchdogTier.DEACON]
        issues = await watchdog._check_deacon(config)

        coord_issues = [i for i in issues if i.category == IssueCategory.COORDINATION_FAILURE]
        assert len(coord_issues) == 1


class TestWatchdogStatistics:
    """Tests for watchdog statistics."""

    def test_statistics_tracking(self, watchdog):
        """Test that statistics are tracked correctly."""
        watchdog.register_agent("claude-opus")
        watchdog.register_agent("gpt-4o")

        stats = watchdog.get_stats()

        assert stats["monitored_agents"] == 2
        assert stats["issues_detected"] == 0
        assert stats["issues_resolved"] == 0
        assert stats["escalations"] == 0

    @pytest.mark.asyncio
    async def test_statistics_after_issues(self, watchdog):
        """Test statistics after issue detection and resolution."""
        # Create and track an issue
        issue = WatchdogIssue(
            severity=IssueSeverity.ERROR,
            category=IssueCategory.HEARTBEAT_MISSING,
            agent="claude-opus",
            message="Test",
        )

        await watchdog._handle_issue(WatchdogTier.MECHANICAL, issue)

        stats = watchdog.get_stats()
        assert stats["issues_detected"] == 1

        # Resolve
        watchdog.resolve_issue(issue.id)

        stats = watchdog.get_stats()
        assert stats["issues_resolved"] == 1
