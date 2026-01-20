"""
Tests for the Control Plane HealthMonitor.

These tests verify health probing, status tracking, and circuit breaker integration.
"""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from aragora.control_plane.health import (
    HealthCheck,
    HealthMonitor,
    HealthStatus,
)


class TestHealthCheck:
    """Tests for HealthCheck dataclass."""

    def test_creation(self):
        """Test basic HealthCheck creation."""
        check = HealthCheck(
            agent_id="claude-3",
            status=HealthStatus.HEALTHY,
            latency_ms=15.5,
        )

        assert check.agent_id == "claude-3"
        assert check.status == HealthStatus.HEALTHY
        assert check.latency_ms == 15.5
        assert check.error is None

    def test_creation_with_error(self):
        """Test HealthCheck with error."""
        check = HealthCheck(
            agent_id="agent-1",
            status=HealthStatus.UNHEALTHY,
            latency_ms=5000.0,
            error="Connection timeout",
            metadata={"attempts": 3},
        )

        assert check.status == HealthStatus.UNHEALTHY
        assert check.error == "Connection timeout"
        assert check.metadata == {"attempts": 3}

    def test_serialization(self):
        """Test to_dict."""
        check = HealthCheck(
            agent_id="agent-1",
            status=HealthStatus.DEGRADED,
            latency_ms=100.0,
            metadata={"load": 0.8},
        )

        data = check.to_dict()

        assert data["agent_id"] == "agent-1"
        assert data["status"] == "degraded"
        assert data["latency_ms"] == 100.0
        assert data["metadata"] == {"load": 0.8}


class TestHealthMonitor:
    """Tests for HealthMonitor."""

    @pytest.fixture
    def monitor(self):
        """Create a health monitor."""
        return HealthMonitor(
            registry=None,
            probe_interval=1.0,
            probe_timeout=0.5,
            unhealthy_threshold=2,
            recovery_threshold=2,
        )

    def test_register_probe(self, monitor):
        """Test probe registration."""
        probe = MagicMock(return_value=True)

        monitor.register_probe("agent-1", probe)

        assert "agent-1" in monitor._probes
        assert "agent-1" in monitor._circuit_breakers
        assert monitor._failure_counts["agent-1"] == 0

    def test_unregister_probe(self, monitor):
        """Test probe unregistration."""
        probe = MagicMock(return_value=True)
        monitor.register_probe("agent-1", probe)

        monitor.unregister_probe("agent-1")

        assert "agent-1" not in monitor._probes

    @pytest.mark.asyncio
    async def test_probe_healthy(self, monitor):
        """Test probing a healthy agent via run_probes."""
        probe = MagicMock(return_value=True)
        monitor.register_probe("agent-1", probe)

        # Run probes which internally calls _probe_agent
        await monitor._probe_all_agents()

        check = monitor.get_agent_health("agent-1")
        assert check is not None
        assert check.status == HealthStatus.HEALTHY
        probe.assert_called()

    @pytest.mark.asyncio
    async def test_probe_unhealthy(self, monitor):
        """Test probing an unhealthy agent."""
        probe = MagicMock(return_value=False)
        monitor.register_probe("agent-1", probe)

        await monitor._probe_all_agents()

        check = monitor.get_agent_health("agent-1")
        assert check is not None
        # Status can be UNHEALTHY or DEGRADED based on failure count
        assert check.status in (HealthStatus.UNHEALTHY, HealthStatus.DEGRADED)

    @pytest.mark.asyncio
    async def test_probe_exception(self, monitor):
        """Test probing an agent that throws exception."""
        probe = MagicMock(side_effect=Exception("Connection error"))
        monitor.register_probe("agent-1", probe)

        await monitor._probe_all_agents()

        check = monitor.get_agent_health("agent-1")
        assert check is not None
        # Exception causes degraded or unhealthy status with error message
        assert check.status in (
            HealthStatus.CRITICAL,
            HealthStatus.DEGRADED,
            HealthStatus.UNHEALTHY,
        )
        assert check.error is not None
        assert "Connection error" in check.error

    @pytest.mark.asyncio
    async def test_start_stop(self, monitor):
        """Test starting and stopping monitoring."""
        probe = MagicMock(return_value=True)
        monitor.register_probe("agent-1", probe)

        await monitor.start()
        assert monitor._running

        # Let it run briefly
        await asyncio.sleep(0.1)

        await monitor.stop()
        assert not monitor._running

    def test_get_agent_health(self, monitor):
        """Test getting agent health status."""
        # Set up health check
        monitor._health_checks["agent-1"] = HealthCheck(
            agent_id="agent-1",
            status=HealthStatus.HEALTHY,
            latency_ms=10.0,
        )

        check = monitor.get_agent_health("agent-1")

        assert check is not None
        assert check.status == HealthStatus.HEALTHY

    def test_get_agent_health_unknown(self, monitor):
        """Test getting health for unknown agent."""
        check = monitor.get_agent_health("unknown-agent")
        assert check is None

    def test_get_all_health(self, monitor):
        """Test getting all agent health statuses."""
        monitor._health_checks["agent-1"] = HealthCheck(
            agent_id="agent-1",
            status=HealthStatus.HEALTHY,
            latency_ms=10.0,
        )
        monitor._health_checks["agent-2"] = HealthCheck(
            agent_id="agent-2",
            status=HealthStatus.DEGRADED,
            latency_ms=500.0,
        )

        all_health = monitor.get_all_health()

        assert len(all_health) == 2
        assert "agent-1" in all_health
        assert "agent-2" in all_health

    def test_is_agent_available_healthy(self, monitor):
        """Test availability check for healthy agent."""
        monitor._health_checks["agent-1"] = HealthCheck(
            agent_id="agent-1",
            status=HealthStatus.HEALTHY,
            latency_ms=10.0,
        )

        assert monitor.is_agent_available("agent-1")

    def test_is_agent_available_degraded(self, monitor):
        """Test availability check for degraded agent."""
        monitor._health_checks["agent-1"] = HealthCheck(
            agent_id="agent-1",
            status=HealthStatus.DEGRADED,
            latency_ms=100.0,
        )

        # Degraded agents are still available
        assert monitor.is_agent_available("agent-1")

    def test_is_agent_available_unhealthy(self, monitor):
        """Test availability check for unhealthy agent."""
        monitor._health_checks["agent-1"] = HealthCheck(
            agent_id="agent-1",
            status=HealthStatus.UNHEALTHY,
            latency_ms=5000.0,
        )

        assert not monitor.is_agent_available("agent-1")

    def test_is_agent_available_unknown(self, monitor):
        """Test availability check for unknown agent."""
        # Unknown agents are considered available (no evidence they're not)
        assert monitor.is_agent_available("unknown-agent")

    def test_get_system_health_all_healthy(self, monitor):
        """Test system health when all agents are healthy."""
        monitor._health_checks["agent-1"] = HealthCheck(
            agent_id="agent-1",
            status=HealthStatus.HEALTHY,
            latency_ms=10.0,
        )
        monitor._health_checks["agent-2"] = HealthCheck(
            agent_id="agent-2",
            status=HealthStatus.HEALTHY,
            latency_ms=15.0,
        )

        status = monitor.get_system_health()
        assert status == HealthStatus.HEALTHY

    def test_get_system_health_some_degraded(self, monitor):
        """Test system health when some agents are degraded."""
        monitor._health_checks["agent-1"] = HealthCheck(
            agent_id="agent-1",
            status=HealthStatus.HEALTHY,
            latency_ms=10.0,
        )
        monitor._health_checks["agent-2"] = HealthCheck(
            agent_id="agent-2",
            status=HealthStatus.DEGRADED,
            latency_ms=500.0,
        )

        status = monitor.get_system_health()
        assert status == HealthStatus.DEGRADED

    def test_get_system_health_some_unhealthy(self, monitor):
        """Test system health when some agents are unhealthy."""
        monitor._health_checks["agent-1"] = HealthCheck(
            agent_id="agent-1",
            status=HealthStatus.HEALTHY,
            latency_ms=10.0,
        )
        monitor._health_checks["agent-2"] = HealthCheck(
            agent_id="agent-2",
            status=HealthStatus.UNHEALTHY,
            latency_ms=5000.0,
        )

        status = monitor.get_system_health()
        assert status == HealthStatus.UNHEALTHY

    def test_get_system_health_all_critical(self, monitor):
        """Test system health when all agents are critical."""
        monitor._health_checks["agent-1"] = HealthCheck(
            agent_id="agent-1",
            status=HealthStatus.CRITICAL,
            latency_ms=0.0,
            error="Agent offline",
        )

        status = monitor.get_system_health()
        assert status == HealthStatus.CRITICAL

    def test_get_system_health_no_agents(self, monitor):
        """Test system health with no monitored agents."""
        status = monitor.get_system_health()
        assert status == HealthStatus.HEALTHY

    def test_get_stats(self, monitor):
        """Test statistics retrieval."""
        probe1 = MagicMock(return_value=True)
        probe2 = MagicMock(return_value=True)
        monitor.register_probe("agent-1", probe1)
        monitor.register_probe("agent-2", probe2)

        monitor._health_checks["agent-1"] = HealthCheck(
            agent_id="agent-1",
            status=HealthStatus.HEALTHY,
            latency_ms=10.0,
        )
        monitor._health_checks["agent-2"] = HealthCheck(
            agent_id="agent-2",
            status=HealthStatus.UNHEALTHY,
            latency_ms=5000.0,
        )

        stats = monitor.get_stats()

        assert stats["monitored_agents"] == 2
        assert "by_status" in stats

    @pytest.mark.asyncio
    async def test_probe_updates_health_check(self, monitor):
        """Test that probing updates the health check."""
        probe = MagicMock(return_value=False)
        monitor.register_probe("agent-1", probe)

        # Initially no health check
        assert monitor.get_agent_health("agent-1") is None

        # Run probes
        await monitor._probe_all_agents()

        # Now should have health check
        check = monitor.get_agent_health("agent-1")
        assert check is not None
        assert check.agent_id == "agent-1"

    @pytest.mark.asyncio
    async def test_failure_tracking(self, monitor):
        """Test that consecutive failures are tracked."""
        probe = MagicMock(return_value=False)
        monitor.register_probe("agent-1", probe)

        # Run probes multiple times
        for _ in range(3):
            await monitor._probe_all_agents()

        # Failure count should increase
        assert monitor._failure_counts.get("agent-1", 0) >= 1
