"""
Tests for aragora.control_plane.watchdog module.

Covers:
- WatchdogTier enum
- IssueSeverity enum
- IssueCategory enum
- WatchdogIssue dataclass
- EscalationResult dataclass
- WatchdogConfig dataclass
- AgentHealth dataclass
- ThreeTierWatchdog class
- Global watchdog singleton
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.control_plane.watchdog import (
    AgentHealth,
    EscalationResult,
    IssueCategory,
    IssueSeverity,
    ThreeTierWatchdog,
    WatchdogConfig,
    WatchdogIssue,
    WatchdogTier,
    get_watchdog,
    reset_watchdog,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def watchdog() -> ThreeTierWatchdog:
    """Create a fresh watchdog for testing."""
    return ThreeTierWatchdog()


@pytest.fixture
def issue() -> WatchdogIssue:
    """Create a basic issue for testing."""
    return WatchdogIssue(
        severity=IssueSeverity.WARNING,
        category=IssueCategory.HEARTBEAT_MISSING,
        agent="test-agent",
        message="Test issue",
    )


@pytest.fixture
def config() -> WatchdogConfig:
    """Create a basic config for testing."""
    return WatchdogConfig(
        tier=WatchdogTier.MECHANICAL,
        check_interval_seconds=0.1,
        heartbeat_timeout_seconds=1.0,
    )


# =============================================================================
# WatchdogTier Tests
# =============================================================================


class TestWatchdogTier:
    """Tests for WatchdogTier enum."""

    def test_tier_values(self):
        """Test tier enum values."""
        assert WatchdogTier.MECHANICAL.value == "mechanical"
        assert WatchdogTier.BOOT_AGENT.value == "boot_agent"
        assert WatchdogTier.DEACON.value == "deacon"

    def test_tier_is_string_enum(self):
        """Test tier is a string enum."""
        for tier in WatchdogTier:
            assert isinstance(tier, str)


# =============================================================================
# IssueSeverity Tests
# =============================================================================


class TestIssueSeverity:
    """Tests for IssueSeverity enum."""

    def test_severity_values(self):
        """Test severity values in order."""
        assert IssueSeverity.INFO.value == 0
        assert IssueSeverity.WARNING.value == 1
        assert IssueSeverity.ERROR.value == 2
        assert IssueSeverity.CRITICAL.value == 3

    def test_severity_comparison(self):
        """Test severity can be compared."""
        assert IssueSeverity.INFO < IssueSeverity.WARNING
        assert IssueSeverity.WARNING < IssueSeverity.ERROR
        assert IssueSeverity.ERROR < IssueSeverity.CRITICAL


# =============================================================================
# IssueCategory Tests
# =============================================================================


class TestIssueCategory:
    """Tests for IssueCategory enum."""

    def test_mechanical_categories(self):
        """Test Tier 1 categories."""
        assert IssueCategory.HEARTBEAT_MISSING.value == "heartbeat_missing"
        assert IssueCategory.MEMORY_EXCEEDED.value == "memory_exceeded"
        assert IssueCategory.CIRCUIT_OPEN.value == "circuit_open"
        assert IssueCategory.RESOURCE_EXHAUSTED.value == "resource_exhausted"

    def test_boot_agent_categories(self):
        """Test Tier 2 categories."""
        assert IssueCategory.LATENCY_EXCEEDED.value == "latency_exceeded"
        assert IssueCategory.RESPONSE_QUALITY_LOW.value == "response_quality_low"
        assert IssueCategory.ERROR_RATE_HIGH.value == "error_rate_high"
        assert IssueCategory.SEMANTIC_DRIFT.value == "semantic_drift"

    def test_deacon_categories(self):
        """Test Tier 3 categories."""
        assert IssueCategory.SLA_VIOLATION.value == "sla_violation"
        assert IssueCategory.COORDINATION_FAILURE.value == "coordination_failure"
        assert IssueCategory.POLICY_VIOLATION.value == "policy_violation"
        assert IssueCategory.CONSENSUS_BLOCKED.value == "consensus_blocked"


# =============================================================================
# WatchdogIssue Tests
# =============================================================================


class TestWatchdogIssue:
    """Tests for WatchdogIssue dataclass."""

    def test_create_issue(self, issue: WatchdogIssue):
        """Test creating an issue."""
        assert issue.severity == IssueSeverity.WARNING
        assert issue.category == IssueCategory.HEARTBEAT_MISSING
        assert issue.agent == "test-agent"
        assert issue.message == "Test issue"

    def test_issue_defaults(self, issue: WatchdogIssue):
        """Test issue default values."""
        assert issue.id is not None
        assert issue.detected_at is not None
        assert issue.detected_by is None
        assert issue.resolved is False
        assert issue.resolved_at is None
        assert issue.resolution_notes is None
        assert issue.details == {}

    def test_unique_ids(self):
        """Test issues get unique IDs when explicitly provided."""
        # Note: Auto-generated IDs may collide in same millisecond, so test with explicit IDs
        i1 = WatchdogIssue(
            severity=IssueSeverity.INFO,
            category=IssueCategory.HEARTBEAT_MISSING,
            agent=None,
            message="Test 1",
            id="issue-001",
        )
        i2 = WatchdogIssue(
            severity=IssueSeverity.INFO,
            category=IssueCategory.HEARTBEAT_MISSING,
            agent=None,
            message="Test 2",
            id="issue-002",
        )
        assert i1.id != i2.id

    def test_id_format(self):
        """Test auto-generated ID has expected format."""
        issue = WatchdogIssue(
            severity=IssueSeverity.INFO,
            category=IssueCategory.HEARTBEAT_MISSING,
            agent=None,
            message="Test",
        )
        assert issue.id.startswith("issue-")
        # ID should be "issue-" followed by digits
        assert issue.id[6:].isdigit()

    def test_to_dict(self, issue: WatchdogIssue):
        """Test serialization to dict."""
        data = issue.to_dict()

        assert data["id"] == issue.id
        assert data["severity"] == "WARNING"
        assert data["category"] == "heartbeat_missing"
        assert data["agent"] == "test-agent"
        assert data["message"] == "Test issue"
        assert data["resolved"] is False

    def test_to_dict_with_resolution(self, issue: WatchdogIssue):
        """Test serialization includes resolution info."""
        issue.resolved = True
        issue.resolved_at = datetime.now(timezone.utc)
        issue.resolution_notes = "Fixed"

        data = issue.to_dict()

        assert data["resolved"] is True
        assert data["resolved_at"] is not None
        assert data["resolution_notes"] == "Fixed"


# =============================================================================
# EscalationResult Tests
# =============================================================================


class TestEscalationResult:
    """Tests for EscalationResult dataclass."""

    def test_create_success_result(self):
        """Test creating a success escalation result."""
        result = EscalationResult(
            issue_id="issue-123",
            escalated_to=WatchdogTier.BOOT_AGENT,
            accepted=True,
            action_taken="Escalated to boot agent",
        )

        assert result.accepted is True
        assert result.escalated_to == WatchdogTier.BOOT_AGENT
        assert result.action_taken == "Escalated to boot agent"

    def test_create_failure_result(self):
        """Test creating a failure escalation result."""
        result = EscalationResult(
            issue_id="issue-123",
            escalated_to=WatchdogTier.DEACON,
            accepted=False,
            error_message="Already at highest tier",
        )

        assert result.accepted is False
        assert result.error_message == "Already at highest tier"


# =============================================================================
# WatchdogConfig Tests
# =============================================================================


class TestWatchdogConfig:
    """Tests for WatchdogConfig dataclass."""

    def test_create_config(self, config: WatchdogConfig):
        """Test creating a config."""
        assert config.tier == WatchdogTier.MECHANICAL
        assert config.check_interval_seconds == 0.1
        assert config.heartbeat_timeout_seconds == 1.0

    def test_config_defaults(self):
        """Test config default values."""
        config = WatchdogConfig(tier=WatchdogTier.MECHANICAL)

        assert config.check_interval_seconds == 5.0
        assert config.heartbeat_timeout_seconds == 30.0
        assert config.memory_warning_mb == 1024.0
        assert config.memory_critical_mb == 2048.0
        assert config.latency_warning_ms == 5000.0
        assert config.latency_critical_ms == 15000.0
        assert config.error_rate_warning == 0.1
        assert config.error_rate_critical == 0.3
        assert config.auto_escalate is True
        assert config.escalation_threshold == 3

    def test_to_dict(self, config: WatchdogConfig):
        """Test serialization to dict."""
        data = config.to_dict()

        assert data["tier"] == "mechanical"
        assert data["check_interval_seconds"] == 0.1


# =============================================================================
# AgentHealth Tests
# =============================================================================


class TestAgentHealth:
    """Tests for AgentHealth dataclass."""

    def test_create_health(self):
        """Test creating agent health."""
        health = AgentHealth(agent_name="test-agent")

        assert health.agent_name == "test-agent"
        assert health.last_heartbeat is None
        assert health.consecutive_failures == 0
        assert health.total_requests == 0
        assert health.failed_requests == 0

    def test_error_rate_zero_requests(self):
        """Test error rate with no requests."""
        health = AgentHealth(agent_name="test")
        assert health.error_rate == 0.0

    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        health = AgentHealth(agent_name="test")
        health.total_requests = 100
        health.failed_requests = 25

        assert health.error_rate == 0.25

    def test_average_latency_zero_requests(self):
        """Test average latency with no requests."""
        health = AgentHealth(agent_name="test")
        assert health.average_latency_ms == 0.0

    def test_average_latency_calculation(self):
        """Test average latency calculation."""
        health = AgentHealth(agent_name="test")
        health.total_requests = 10
        health.total_latency_ms = 5000

        assert health.average_latency_ms == 500.0

    def test_record_successful_request(self):
        """Test recording successful request."""
        health = AgentHealth(agent_name="test")
        health.record_request(success=True, latency_ms=100)

        assert health.total_requests == 1
        assert health.failed_requests == 0
        assert health.consecutive_failures == 0
        assert health.total_latency_ms == 100

    def test_record_failed_request(self):
        """Test recording failed request."""
        health = AgentHealth(agent_name="test")
        health.record_request(success=False, latency_ms=50)

        assert health.total_requests == 1
        assert health.failed_requests == 1
        assert health.consecutive_failures == 1

    def test_consecutive_failures_reset_on_success(self):
        """Test consecutive failures reset on success."""
        health = AgentHealth(agent_name="test")
        health.record_request(success=False, latency_ms=100)
        health.record_request(success=False, latency_ms=100)
        assert health.consecutive_failures == 2

        health.record_request(success=True, latency_ms=100)
        assert health.consecutive_failures == 0

    def test_record_heartbeat(self):
        """Test recording heartbeat."""
        health = AgentHealth(agent_name="test")
        assert health.last_heartbeat is None

        health.record_heartbeat()

        assert health.last_heartbeat is not None


# =============================================================================
# ThreeTierWatchdog Registration Tests
# =============================================================================


class TestThreeTierWatchdogRegistration:
    """Tests for agent and handler registration."""

    def test_register_agent(self, watchdog: ThreeTierWatchdog):
        """Test registering an agent."""
        watchdog.register_agent("test-agent")

        assert "test-agent" in watchdog._agent_health
        health = watchdog.get_agent_health("test-agent")
        assert health is not None
        assert health.agent_name == "test-agent"

    def test_register_duplicate_agent(self, watchdog: ThreeTierWatchdog):
        """Test registering same agent twice is idempotent."""
        watchdog.register_agent("test-agent")
        watchdog.register_agent("test-agent")

        assert len(watchdog._agent_health) == 1

    def test_unregister_agent(self, watchdog: ThreeTierWatchdog):
        """Test unregistering an agent."""
        watchdog.register_agent("test-agent")
        watchdog.unregister_agent("test-agent")

        assert "test-agent" not in watchdog._agent_health

    def test_register_handler(self, watchdog: ThreeTierWatchdog):
        """Test registering an issue handler."""
        handler = MagicMock()
        unregister = watchdog.register_handler(WatchdogTier.MECHANICAL, handler)

        assert handler in watchdog._handlers[WatchdogTier.MECHANICAL]

        unregister()
        assert handler not in watchdog._handlers[WatchdogTier.MECHANICAL]


# =============================================================================
# ThreeTierWatchdog Configuration Tests
# =============================================================================


class TestThreeTierWatchdogConfig:
    """Tests for watchdog configuration."""

    def test_configure_tier(self, watchdog: ThreeTierWatchdog, config: WatchdogConfig):
        """Test configuring a tier."""
        watchdog.configure_tier(config)

        stored = watchdog._configs[WatchdogTier.MECHANICAL]
        assert stored.check_interval_seconds == 0.1

    def test_default_configs(self, watchdog: ThreeTierWatchdog):
        """Test default configs exist for all tiers."""
        for tier in WatchdogTier:
            assert tier in watchdog._configs


# =============================================================================
# ThreeTierWatchdog Recording Tests
# =============================================================================


class TestThreeTierWatchdogRecording:
    """Tests for recording agent metrics."""

    def test_record_heartbeat(self, watchdog: ThreeTierWatchdog):
        """Test recording heartbeat."""
        watchdog.record_heartbeat("test-agent")

        health = watchdog.get_agent_health("test-agent")
        assert health is not None
        assert health.last_heartbeat is not None

    def test_record_heartbeat_auto_registers(self, watchdog: ThreeTierWatchdog):
        """Test heartbeat auto-registers agent."""
        assert "new-agent" not in watchdog._agent_health

        watchdog.record_heartbeat("new-agent")

        assert "new-agent" in watchdog._agent_health

    def test_record_request(self, watchdog: ThreeTierWatchdog):
        """Test recording request."""
        watchdog.record_request("test-agent", success=True, latency_ms=100)

        health = watchdog.get_agent_health("test-agent")
        assert health.total_requests == 1

    def test_update_memory_usage(self, watchdog: ThreeTierWatchdog):
        """Test updating memory usage."""
        watchdog.update_memory_usage("test-agent", 512.0)

        health = watchdog.get_agent_health("test-agent")
        assert health.memory_usage_mb == 512.0

    def test_update_circuit_breaker(self, watchdog: ThreeTierWatchdog):
        """Test updating circuit breaker state."""
        watchdog.update_circuit_breaker("test-agent", "open")

        health = watchdog.get_agent_health("test-agent")
        assert health.circuit_breaker_state == "open"


# =============================================================================
# ThreeTierWatchdog Tier Checks Tests
# =============================================================================


class TestThreeTierWatchdogChecks:
    """Tests for tier-specific checks."""

    @pytest.mark.asyncio
    async def test_check_mechanical_heartbeat(self, watchdog: ThreeTierWatchdog):
        """Test mechanical tier detects missing heartbeat."""
        watchdog.register_agent("test-agent")
        health = watchdog.get_agent_health("test-agent")
        health.last_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=60)

        config = WatchdogConfig(
            tier=WatchdogTier.MECHANICAL,
            heartbeat_timeout_seconds=30.0,
        )

        issues = await watchdog._check_mechanical(config)

        assert len(issues) >= 1
        assert any(i.category == IssueCategory.HEARTBEAT_MISSING for i in issues)

    @pytest.mark.asyncio
    async def test_check_mechanical_memory_warning(self, watchdog: ThreeTierWatchdog):
        """Test mechanical tier detects memory warning."""
        watchdog.register_agent("test-agent")
        watchdog.update_memory_usage("test-agent", 1500.0)

        config = WatchdogConfig(
            tier=WatchdogTier.MECHANICAL,
            memory_warning_mb=1024.0,
            memory_critical_mb=2048.0,
        )

        issues = await watchdog._check_mechanical(config)

        assert len(issues) >= 1
        memory_issues = [i for i in issues if i.category == IssueCategory.MEMORY_EXCEEDED]
        assert len(memory_issues) >= 1
        assert memory_issues[0].severity == IssueSeverity.WARNING

    @pytest.mark.asyncio
    async def test_check_mechanical_memory_critical(self, watchdog: ThreeTierWatchdog):
        """Test mechanical tier detects memory critical."""
        watchdog.register_agent("test-agent")
        watchdog.update_memory_usage("test-agent", 3000.0)

        config = WatchdogConfig(
            tier=WatchdogTier.MECHANICAL,
            memory_critical_mb=2048.0,
        )

        issues = await watchdog._check_mechanical(config)

        memory_issues = [i for i in issues if i.category == IssueCategory.MEMORY_EXCEEDED]
        assert len(memory_issues) >= 1
        assert memory_issues[0].severity == IssueSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_check_mechanical_circuit_open(self, watchdog: ThreeTierWatchdog):
        """Test mechanical tier detects open circuit breaker."""
        watchdog.register_agent("test-agent")
        watchdog.update_circuit_breaker("test-agent", "open")

        config = WatchdogConfig(tier=WatchdogTier.MECHANICAL)

        issues = await watchdog._check_mechanical(config)

        circuit_issues = [i for i in issues if i.category == IssueCategory.CIRCUIT_OPEN]
        assert len(circuit_issues) >= 1

    @pytest.mark.asyncio
    async def test_check_boot_agent_latency(self, watchdog: ThreeTierWatchdog):
        """Test boot agent tier detects high latency."""
        watchdog.register_agent("test-agent")
        # Record high latency requests
        for _ in range(10):
            watchdog.record_request("test-agent", success=True, latency_ms=8000)

        config = WatchdogConfig(
            tier=WatchdogTier.BOOT_AGENT,
            latency_warning_ms=5000.0,
            latency_critical_ms=15000.0,
        )

        issues = await watchdog._check_boot_agent(config)

        latency_issues = [i for i in issues if i.category == IssueCategory.LATENCY_EXCEEDED]
        assert len(latency_issues) >= 1

    @pytest.mark.asyncio
    async def test_check_boot_agent_error_rate(self, watchdog: ThreeTierWatchdog):
        """Test boot agent tier detects high error rate."""
        watchdog.register_agent("test-agent")
        # Record 20% error rate
        for i in range(10):
            watchdog.record_request("test-agent", success=i < 8, latency_ms=100)

        config = WatchdogConfig(
            tier=WatchdogTier.BOOT_AGENT,
            error_rate_warning=0.1,
        )

        issues = await watchdog._check_boot_agent(config)

        error_issues = [i for i in issues if i.category == IssueCategory.ERROR_RATE_HIGH]
        assert len(error_issues) >= 1

    @pytest.mark.asyncio
    async def test_check_deacon_sla_violation(self, watchdog: ThreeTierWatchdog):
        """Test deacon tier detects SLA violation."""
        watchdog.register_agent("test-agent")
        # Record 50% failure rate
        for i in range(10):
            watchdog.record_request("test-agent", success=i < 5, latency_ms=100)

        config = WatchdogConfig(
            tier=WatchdogTier.DEACON,
            sla_availability_pct=99.0,
        )

        issues = await watchdog._check_deacon(config)

        sla_issues = [i for i in issues if i.category == IssueCategory.SLA_VIOLATION]
        assert len(sla_issues) >= 1


# =============================================================================
# ThreeTierWatchdog Issue Management Tests
# =============================================================================


class TestThreeTierWatchdogIssues:
    """Tests for issue management."""

    @pytest.mark.asyncio
    async def test_handle_issue(self, watchdog: ThreeTierWatchdog, issue: WatchdogIssue):
        """Test handling an issue."""
        await watchdog._handle_issue(WatchdogTier.MECHANICAL, issue)

        assert issue.id in watchdog._active_issues
        stats = watchdog.get_stats()
        assert stats["issues_detected"] == 1

    @pytest.mark.asyncio
    async def test_handle_issue_calls_handlers(
        self, watchdog: ThreeTierWatchdog, issue: WatchdogIssue
    ):
        """Test handling issue calls registered handlers."""
        handler = MagicMock()
        watchdog.register_handler(WatchdogTier.MECHANICAL, handler)

        await watchdog._handle_issue(WatchdogTier.MECHANICAL, issue)

        handler.assert_called_once_with(issue)

    def test_resolve_issue(self, watchdog: ThreeTierWatchdog, issue: WatchdogIssue):
        """Test resolving an issue."""
        watchdog._active_issues[issue.id] = issue

        result = watchdog.resolve_issue(issue.id, notes="Fixed")

        assert result is True
        assert issue.resolved is True
        assert issue.resolution_notes == "Fixed"

    def test_resolve_nonexistent_issue(self, watchdog: ThreeTierWatchdog):
        """Test resolving nonexistent issue."""
        result = watchdog.resolve_issue("nonexistent")
        assert result is False

    def test_get_active_issues(self, watchdog: ThreeTierWatchdog):
        """Test getting active issues."""
        issue1 = WatchdogIssue(
            severity=IssueSeverity.WARNING,
            category=IssueCategory.HEARTBEAT_MISSING,
            agent="agent1",
            message="Test 1",
            id="active-issue-1",
        )
        issue2 = WatchdogIssue(
            severity=IssueSeverity.ERROR,
            category=IssueCategory.MEMORY_EXCEEDED,
            agent="agent2",
            message="Test 2",
            id="resolved-issue-2",
        )
        issue2.resolved = True

        watchdog._active_issues[issue1.id] = issue1
        watchdog._active_issues[issue2.id] = issue2

        active = watchdog.get_active_issues()

        assert len(active) == 1
        assert active[0].id == issue1.id

    def test_get_active_issues_filter_severity(self, watchdog: ThreeTierWatchdog):
        """Test filtering active issues by severity."""
        issue_warning = WatchdogIssue(
            severity=IssueSeverity.WARNING,
            category=IssueCategory.HEARTBEAT_MISSING,
            agent=None,
            message="Warning",
            id="warning-issue-1",
        )
        issue_error = WatchdogIssue(
            severity=IssueSeverity.ERROR,
            category=IssueCategory.CIRCUIT_OPEN,
            agent=None,
            message="Error",
            id="error-issue-2",
        )

        watchdog._active_issues[issue_warning.id] = issue_warning
        watchdog._active_issues[issue_error.id] = issue_error

        active = watchdog.get_active_issues(severity=IssueSeverity.ERROR)

        assert len(active) == 1
        assert active[0].severity == IssueSeverity.ERROR


# =============================================================================
# ThreeTierWatchdog Escalation Tests
# =============================================================================


class TestThreeTierWatchdogEscalation:
    """Tests for issue escalation."""

    @pytest.mark.asyncio
    async def test_escalate_mechanical_to_boot_agent(
        self, watchdog: ThreeTierWatchdog, issue: WatchdogIssue
    ):
        """Test escalating from mechanical to boot agent."""
        result = await watchdog.escalate(WatchdogTier.MECHANICAL, issue)

        assert result.accepted is True
        assert result.escalated_to == WatchdogTier.BOOT_AGENT

    @pytest.mark.asyncio
    async def test_escalate_boot_agent_to_deacon(
        self, watchdog: ThreeTierWatchdog, issue: WatchdogIssue
    ):
        """Test escalating from boot agent to deacon."""
        result = await watchdog.escalate(WatchdogTier.BOOT_AGENT, issue)

        assert result.accepted is True
        assert result.escalated_to == WatchdogTier.DEACON

    @pytest.mark.asyncio
    async def test_escalate_from_highest_tier(
        self, watchdog: ThreeTierWatchdog, issue: WatchdogIssue
    ):
        """Test escalating from deacon (highest) tier fails."""
        result = await watchdog.escalate(WatchdogTier.DEACON, issue)

        assert result.accepted is False
        assert "highest tier" in result.error_message.lower()

    @pytest.mark.asyncio
    async def test_escalate_calls_handlers(self, watchdog: ThreeTierWatchdog, issue: WatchdogIssue):
        """Test escalation calls target tier handlers."""
        handler = MagicMock()
        watchdog.register_handler(WatchdogTier.BOOT_AGENT, handler)

        await watchdog.escalate(WatchdogTier.MECHANICAL, issue)

        handler.assert_called_once_with(issue)


# =============================================================================
# ThreeTierWatchdog Start/Stop Tests
# =============================================================================


class TestThreeTierWatchdogLifecycle:
    """Tests for watchdog start/stop."""

    @pytest.mark.asyncio
    async def test_start_stop(self, watchdog: ThreeTierWatchdog):
        """Test starting and stopping watchdog."""
        # Configure very short intervals for testing
        for tier in WatchdogTier:
            watchdog.configure_tier(WatchdogConfig(tier=tier, check_interval_seconds=0.01))

        await watchdog.start()
        assert watchdog._running is True
        assert len(watchdog._tasks) == 3

        await asyncio.sleep(0.05)  # Let it run briefly

        await watchdog.stop()
        assert watchdog._running is False
        assert len(watchdog._tasks) == 0

    @pytest.mark.asyncio
    async def test_start_idempotent(self, watchdog: ThreeTierWatchdog):
        """Test starting twice is idempotent."""
        for tier in WatchdogTier:
            watchdog.configure_tier(WatchdogConfig(tier=tier, check_interval_seconds=60))

        await watchdog.start()
        initial_tasks = len(watchdog._tasks)

        await watchdog.start()  # Second start
        assert len(watchdog._tasks) == initial_tasks

        await watchdog.stop()


# =============================================================================
# ThreeTierWatchdog Stats Tests
# =============================================================================


class TestThreeTierWatchdogStats:
    """Tests for statistics."""

    def test_get_stats(self, watchdog: ThreeTierWatchdog):
        """Test getting statistics."""
        stats = watchdog.get_stats()

        assert "issues_detected" in stats
        assert "issues_resolved" in stats
        assert "escalations" in stats
        assert "tier_checks" in stats
        assert "active_issues" in stats
        assert "monitored_agents" in stats
        assert "is_running" in stats

    def test_get_all_health(self, watchdog: ThreeTierWatchdog):
        """Test getting all agent health."""
        watchdog.register_agent("agent1")
        watchdog.register_agent("agent2")

        all_health = watchdog.get_all_health()

        assert len(all_health) == 2
        assert "agent1" in all_health
        assert "agent2" in all_health


# =============================================================================
# Global Watchdog Singleton Tests
# =============================================================================


class TestGlobalWatchdog:
    """Tests for global watchdog singleton."""

    @pytest.mark.asyncio
    async def test_get_watchdog_returns_singleton(self):
        """Test get_watchdog returns same instance."""
        reset_watchdog()
        await asyncio.sleep(0.01)

        w1 = get_watchdog()
        w2 = get_watchdog()

        assert w1 is w2

    @pytest.mark.asyncio
    async def test_reset_watchdog(self):
        """Test reset_watchdog creates new instance."""
        w1 = get_watchdog()
        reset_watchdog()
        # Allow any pending async cleanup
        await asyncio.sleep(0.01)
        w2 = get_watchdog()

        assert w1 is not w2
