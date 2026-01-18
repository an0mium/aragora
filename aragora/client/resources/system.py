"""
System API for the Aragora Python SDK.

Provides access to system health, status, and configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from aragora.client.client import AragoraClient


@dataclass
class HealthStatus:
    """System health status."""

    status: str
    version: str
    uptime_seconds: float
    checks: Dict[str, bool]
    timestamp: str

    @property
    def is_healthy(self) -> bool:
        """Check if all health checks pass."""
        return self.status == "healthy" and all(self.checks.values())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HealthStatus":
        return cls(
            status=data.get("status", "unknown"),
            version=data.get("version", "unknown"),
            uptime_seconds=data.get("uptime_seconds", 0.0),
            checks=data.get("checks", {}),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class SystemInfo:
    """System information."""

    version: str
    environment: str
    python_version: str
    platform: str
    agents_available: List[str]
    features_enabled: List[str]
    memory_mb: float = 0.0
    cpu_percent: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemInfo":
        return cls(
            version=data.get("version", "unknown"),
            environment=data.get("environment", "production"),
            python_version=data.get("python_version", ""),
            platform=data.get("platform", ""),
            agents_available=data.get("agents_available", []),
            features_enabled=data.get("features_enabled", []),
            memory_mb=data.get("memory_mb", 0.0),
            cpu_percent=data.get("cpu_percent", 0.0),
        )


@dataclass
class SystemStats:
    """System statistics."""

    total_debates: int
    total_agents: int
    active_debates: int
    debates_today: int
    debates_this_week: int
    avg_debate_duration_seconds: float
    memory_entries: int
    consensus_rate: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemStats":
        return cls(
            total_debates=data.get("total_debates", 0),
            total_agents=data.get("total_agents", 0),
            active_debates=data.get("active_debates", 0),
            debates_today=data.get("debates_today", 0),
            debates_this_week=data.get("debates_this_week", 0),
            avg_debate_duration_seconds=data.get("avg_debate_duration_seconds", 0.0),
            memory_entries=data.get("memory_entries", 0),
            consensus_rate=data.get("consensus_rate", 0.0),
        )


@dataclass
class CircuitBreakerStatus:
    """Circuit breaker status for an agent."""

    agent_id: str
    state: str
    failure_count: int
    success_count: int
    last_failure: Optional[str] = None
    last_success: Optional[str] = None

    @property
    def is_open(self) -> bool:
        """Check if circuit breaker is open (blocking requests)."""
        return self.state == "open"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CircuitBreakerStatus":
        return cls(
            agent_id=data.get("agent_id", ""),
            state=data.get("state", "closed"),
            failure_count=data.get("failure_count", 0),
            success_count=data.get("success_count", 0),
            last_failure=data.get("last_failure"),
            last_success=data.get("last_success"),
        )


class SystemAPI:
    """
    API interface for system health and status.

    Provides access to health checks, system info, and statistics.

    Example:
        # Check system health
        health = client.system.health()
        if health.is_healthy:
            print("System is healthy")
        else:
            print(f"Unhealthy checks: {[k for k, v in health.checks.items() if not v]}")

        # Get system stats
        stats = client.system.stats()
        print(f"Total debates: {stats.total_debates}")
        print(f"Consensus rate: {stats.consensus_rate:.1%}")

        # Get available agents
        info = client.system.info()
        print(f"Available agents: {', '.join(info.agents_available)}")
    """

    def __init__(self, client: "AragoraClient"):
        self._client = client

    def health(self) -> HealthStatus:
        """
        Get system health status.

        Returns:
            HealthStatus with component checks
        """
        response = self._client._get("/api/health")
        return HealthStatus.from_dict(response)

    async def health_async(self) -> HealthStatus:
        """Async version of health."""
        response = await self._client._get_async("/api/health")
        return HealthStatus.from_dict(response)

    def info(self) -> SystemInfo:
        """
        Get system information.

        Returns:
            SystemInfo with version, platform, and capabilities
        """
        response = self._client._get("/api/system/info")
        return SystemInfo.from_dict(response)

    async def info_async(self) -> SystemInfo:
        """Async version of info."""
        response = await self._client._get_async("/api/system/info")
        return SystemInfo.from_dict(response)

    def stats(self) -> SystemStats:
        """
        Get system statistics.

        Returns:
            SystemStats with aggregate metrics
        """
        response = self._client._get("/api/system/stats")
        return SystemStats.from_dict(response)

    async def stats_async(self) -> SystemStats:
        """Async version of stats."""
        response = await self._client._get_async("/api/system/stats")
        return SystemStats.from_dict(response)

    def circuit_breakers(self) -> List[CircuitBreakerStatus]:
        """
        Get circuit breaker status for all agents.

        Returns:
            List of CircuitBreakerStatus objects
        """
        response = self._client._get("/api/system/circuit-breakers")
        breakers = response.get("breakers", [])
        return [CircuitBreakerStatus.from_dict(b) for b in breakers]

    async def circuit_breakers_async(self) -> List[CircuitBreakerStatus]:
        """Async version of circuit_breakers."""
        response = await self._client._get_async("/api/system/circuit-breakers")
        breakers = response.get("breakers", [])
        return [CircuitBreakerStatus.from_dict(b) for b in breakers]

    def reset_circuit_breaker(self, agent_id: str) -> bool:
        """
        Reset a circuit breaker for an agent.

        Args:
            agent_id: The agent ID

        Returns:
            True if reset successfully
        """
        response = self._client._post(f"/api/system/circuit-breakers/{agent_id}/reset", {})
        return response.get("reset", False)

    async def reset_circuit_breaker_async(self, agent_id: str) -> bool:
        """Async version of reset_circuit_breaker."""
        response = await self._client._post_async(
            f"/api/system/circuit-breakers/{agent_id}/reset", {}
        )
        return response.get("reset", False)

    def modes(self) -> Dict[str, Any]:
        """
        Get current system modes.

        Returns:
            Dict with mode settings
        """
        return self._client._get("/api/system/modes")

    async def modes_async(self) -> Dict[str, Any]:
        """Async version of modes."""
        return await self._client._get_async("/api/system/modes")
