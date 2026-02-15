"""Tests for SystemAPI client resource."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.client.client import AragoraClient
from aragora.client.resources.system import (
    CircuitBreakerStatus,
    HealthStatus,
    SystemAPI,
    SystemInfo,
    SystemStats,
)


@pytest.fixture
def mock_client() -> AragoraClient:
    client = MagicMock(spec=AragoraClient)
    return client


@pytest.fixture
def api(mock_client: AragoraClient) -> SystemAPI:
    return SystemAPI(mock_client)


# ---------------------------------------------------------------------------
# Sample response payloads
# ---------------------------------------------------------------------------

SAMPLE_HEALTH = {
    "status": "healthy",
    "version": "2.1.0",
    "uptime_seconds": 86400.5,
    "checks": {
        "database": True,
        "redis": True,
        "agents": True,
    },
    "timestamp": "2026-02-12T10:00:00Z",
}

SAMPLE_INFO = {
    "version": "2.1.0",
    "environment": "staging",
    "python_version": "3.12.1",
    "platform": "linux",
    "agents_available": ["claude", "gpt-4", "gemini"],
    "features_enabled": ["debate", "memory", "pulse"],
    "memory_mb": 512.4,
    "cpu_percent": 23.7,
}

SAMPLE_STATS = {
    "total_debates": 1500,
    "total_agents": 12,
    "active_debates": 3,
    "debates_today": 45,
    "debates_this_week": 280,
    "avg_debate_duration_seconds": 120.5,
    "memory_entries": 98000,
    "consensus_rate": 0.87,
}

SAMPLE_BREAKERS = {
    "breakers": [
        {
            "agent_id": "agent-1",
            "state": "closed",
            "failure_count": 2,
            "success_count": 150,
            "last_failure": "2026-02-12T09:00:00Z",
            "last_success": "2026-02-12T09:59:00Z",
        },
        {
            "agent_id": "agent-2",
            "state": "open",
            "failure_count": 10,
            "success_count": 5,
            "last_failure": "2026-02-12T09:58:00Z",
            "last_success": "2026-02-12T08:00:00Z",
        },
    ]
}

SAMPLE_MODES = {
    "maintenance": False,
    "read_only": False,
    "debug": True,
    "offline": False,
}


# ---------------------------------------------------------------------------
# HealthStatus dataclass tests
# ---------------------------------------------------------------------------


class TestHealthStatusDataclass:
    def test_construction(self) -> None:
        h = HealthStatus(
            status="healthy",
            version="1.0.0",
            uptime_seconds=100.0,
            checks={"db": True},
            timestamp="2026-01-01T00:00:00Z",
        )
        assert h.status == "healthy"
        assert h.version == "1.0.0"
        assert h.uptime_seconds == 100.0
        assert h.checks == {"db": True}
        assert h.timestamp == "2026-01-01T00:00:00Z"

    def test_is_healthy_all_pass(self) -> None:
        h = HealthStatus(
            status="healthy",
            version="1.0",
            uptime_seconds=1.0,
            checks={"db": True, "redis": True},
            timestamp="",
        )
        assert h.is_healthy is True

    def test_is_healthy_status_not_healthy(self) -> None:
        h = HealthStatus(
            status="degraded",
            version="1.0",
            uptime_seconds=1.0,
            checks={"db": True},
            timestamp="",
        )
        assert h.is_healthy is False

    def test_is_healthy_check_failing(self) -> None:
        h = HealthStatus(
            status="healthy",
            version="1.0",
            uptime_seconds=1.0,
            checks={"db": True, "redis": False},
            timestamp="",
        )
        assert h.is_healthy is False

    def test_is_healthy_empty_checks(self) -> None:
        h = HealthStatus(
            status="healthy",
            version="1.0",
            uptime_seconds=0.0,
            checks={},
            timestamp="",
        )
        # all({}.values()) is True (vacuous truth), so status="healthy" + empty checks = healthy
        assert h.is_healthy is True

    def test_from_dict_full(self) -> None:
        h = HealthStatus.from_dict(SAMPLE_HEALTH)
        assert h.status == "healthy"
        assert h.version == "2.1.0"
        assert h.uptime_seconds == 86400.5
        assert h.checks == {"database": True, "redis": True, "agents": True}
        assert h.timestamp == "2026-02-12T10:00:00Z"

    def test_from_dict_empty(self) -> None:
        h = HealthStatus.from_dict({})
        assert h.status == "unknown"
        assert h.version == "unknown"
        assert h.uptime_seconds == 0.0
        assert h.checks == {}
        assert h.timestamp == ""

    def test_from_dict_partial(self) -> None:
        h = HealthStatus.from_dict({"status": "degraded", "version": "3.0"})
        assert h.status == "degraded"
        assert h.version == "3.0"
        assert h.uptime_seconds == 0.0
        assert h.checks == {}


# ---------------------------------------------------------------------------
# SystemInfo dataclass tests
# ---------------------------------------------------------------------------


class TestSystemInfoDataclass:
    def test_construction_with_defaults(self) -> None:
        info = SystemInfo(
            version="1.0",
            environment="production",
            python_version="3.12",
            platform="linux",
            agents_available=["a"],
            features_enabled=["b"],
        )
        assert info.memory_mb == 0.0
        assert info.cpu_percent == 0.0

    def test_construction_full(self) -> None:
        info = SystemInfo(
            version="1.0",
            environment="dev",
            python_version="3.11",
            platform="darwin",
            agents_available=["claude"],
            features_enabled=["debate"],
            memory_mb=256.0,
            cpu_percent=45.2,
        )
        assert info.memory_mb == 256.0
        assert info.cpu_percent == 45.2

    def test_from_dict_full(self) -> None:
        info = SystemInfo.from_dict(SAMPLE_INFO)
        assert info.version == "2.1.0"
        assert info.environment == "staging"
        assert info.python_version == "3.12.1"
        assert info.platform == "linux"
        assert info.agents_available == ["claude", "gpt-4", "gemini"]
        assert info.features_enabled == ["debate", "memory", "pulse"]
        assert info.memory_mb == 512.4
        assert info.cpu_percent == 23.7

    def test_from_dict_empty(self) -> None:
        info = SystemInfo.from_dict({})
        assert info.version == "unknown"
        assert info.environment == "production"
        assert info.python_version == ""
        assert info.platform == ""
        assert info.agents_available == []
        assert info.features_enabled == []
        assert info.memory_mb == 0.0
        assert info.cpu_percent == 0.0

    def test_from_dict_partial(self) -> None:
        info = SystemInfo.from_dict({"version": "5.0", "agents_available": ["grok"]})
        assert info.version == "5.0"
        assert info.agents_available == ["grok"]
        assert info.environment == "production"


# ---------------------------------------------------------------------------
# SystemStats dataclass tests
# ---------------------------------------------------------------------------


class TestSystemStatsDataclass:
    def test_from_dict_full(self) -> None:
        stats = SystemStats.from_dict(SAMPLE_STATS)
        assert stats.total_debates == 1500
        assert stats.total_agents == 12
        assert stats.active_debates == 3
        assert stats.debates_today == 45
        assert stats.debates_this_week == 280
        assert stats.avg_debate_duration_seconds == 120.5
        assert stats.memory_entries == 98000
        assert stats.consensus_rate == 0.87

    def test_from_dict_empty(self) -> None:
        stats = SystemStats.from_dict({})
        assert stats.total_debates == 0
        assert stats.total_agents == 0
        assert stats.active_debates == 0
        assert stats.debates_today == 0
        assert stats.debates_this_week == 0
        assert stats.avg_debate_duration_seconds == 0.0
        assert stats.memory_entries == 0
        assert stats.consensus_rate == 0.0

    def test_from_dict_partial(self) -> None:
        stats = SystemStats.from_dict({"total_debates": 42, "consensus_rate": 0.95})
        assert stats.total_debates == 42
        assert stats.consensus_rate == 0.95
        assert stats.total_agents == 0

    def test_construction(self) -> None:
        stats = SystemStats(
            total_debates=10,
            total_agents=3,
            active_debates=1,
            debates_today=2,
            debates_this_week=7,
            avg_debate_duration_seconds=60.0,
            memory_entries=500,
            consensus_rate=0.5,
        )
        assert stats.total_debates == 10
        assert stats.consensus_rate == 0.5


# ---------------------------------------------------------------------------
# CircuitBreakerStatus dataclass tests
# ---------------------------------------------------------------------------


class TestCircuitBreakerStatusDataclass:
    def test_construction_minimal(self) -> None:
        cb = CircuitBreakerStatus(
            agent_id="agent-x",
            state="closed",
            failure_count=0,
            success_count=100,
        )
        assert cb.agent_id == "agent-x"
        assert cb.last_failure is None
        assert cb.last_success is None

    def test_construction_full(self) -> None:
        cb = CircuitBreakerStatus(
            agent_id="agent-y",
            state="open",
            failure_count=5,
            success_count=10,
            last_failure="2026-02-12T09:00:00Z",
            last_success="2026-02-12T08:00:00Z",
        )
        assert cb.last_failure == "2026-02-12T09:00:00Z"
        assert cb.last_success == "2026-02-12T08:00:00Z"

    def test_is_open_true(self) -> None:
        cb = CircuitBreakerStatus(agent_id="a", state="open", failure_count=5, success_count=0)
        assert cb.is_open is True

    def test_is_open_false_closed(self) -> None:
        cb = CircuitBreakerStatus(agent_id="a", state="closed", failure_count=0, success_count=10)
        assert cb.is_open is False

    def test_is_open_false_half_open(self) -> None:
        cb = CircuitBreakerStatus(agent_id="a", state="half-open", failure_count=3, success_count=1)
        assert cb.is_open is False

    def test_from_dict_full(self) -> None:
        data = SAMPLE_BREAKERS["breakers"][0]
        cb = CircuitBreakerStatus.from_dict(data)
        assert cb.agent_id == "agent-1"
        assert cb.state == "closed"
        assert cb.failure_count == 2
        assert cb.success_count == 150
        assert cb.last_failure == "2026-02-12T09:00:00Z"
        assert cb.last_success == "2026-02-12T09:59:00Z"

    def test_from_dict_empty(self) -> None:
        cb = CircuitBreakerStatus.from_dict({})
        assert cb.agent_id == ""
        assert cb.state == "closed"
        assert cb.failure_count == 0
        assert cb.success_count == 0
        assert cb.last_failure is None
        assert cb.last_success is None

    def test_from_dict_missing_optional_fields(self) -> None:
        cb = CircuitBreakerStatus.from_dict(
            {"agent_id": "z", "state": "open", "failure_count": 7, "success_count": 0}
        )
        assert cb.agent_id == "z"
        assert cb.is_open is True
        assert cb.last_failure is None
        assert cb.last_success is None


# ---------------------------------------------------------------------------
# SystemAPI.health / health_async
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_HEALTH
        result = api.health()
        assert isinstance(result, HealthStatus)
        assert result.status == "healthy"
        assert result.is_healthy is True
        mock_client._get.assert_called_once_with("/api/health")

    def test_health_unhealthy(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {
            "status": "unhealthy",
            "version": "2.0",
            "uptime_seconds": 10.0,
            "checks": {"database": False, "redis": True},
            "timestamp": "2026-02-12T10:00:00Z",
        }
        result = api.health()
        assert result.is_healthy is False

    @pytest.mark.asyncio
    async def test_health_async(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_HEALTH)
        result = await api.health_async()
        assert isinstance(result, HealthStatus)
        assert result.status == "healthy"
        assert result.version == "2.1.0"
        mock_client._get_async.assert_called_once_with("/api/health")

    def test_health_empty_response(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {}
        result = api.health()
        assert result.status == "unknown"
        assert result.version == "unknown"
        assert result.uptime_seconds == 0.0


# ---------------------------------------------------------------------------
# SystemAPI.info / info_async
# ---------------------------------------------------------------------------


class TestInfo:
    def test_info(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_INFO
        result = api.info()
        assert isinstance(result, SystemInfo)
        assert result.version == "2.1.0"
        assert result.environment == "staging"
        assert "claude" in result.agents_available
        assert result.memory_mb == 512.4
        mock_client._get.assert_called_once_with("/api/system/info")

    @pytest.mark.asyncio
    async def test_info_async(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_INFO)
        result = await api.info_async()
        assert isinstance(result, SystemInfo)
        assert result.platform == "linux"
        mock_client._get_async.assert_called_once_with("/api/system/info")

    def test_info_empty_response(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {}
        result = api.info()
        assert result.version == "unknown"
        assert result.agents_available == []
        assert result.features_enabled == []

    def test_info_no_agents(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"version": "1.0", "environment": "test"}
        result = api.info()
        assert result.agents_available == []
        assert result.features_enabled == []


# ---------------------------------------------------------------------------
# SystemAPI.stats / stats_async
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_STATS
        result = api.stats()
        assert isinstance(result, SystemStats)
        assert result.total_debates == 1500
        assert result.consensus_rate == 0.87
        mock_client._get.assert_called_once_with("/api/system/stats")

    @pytest.mark.asyncio
    async def test_stats_async(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_STATS)
        result = await api.stats_async()
        assert isinstance(result, SystemStats)
        assert result.active_debates == 3
        mock_client._get_async.assert_called_once_with("/api/system/stats")

    def test_stats_empty_response(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {}
        result = api.stats()
        assert result.total_debates == 0
        assert result.consensus_rate == 0.0

    def test_stats_partial_response(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"total_debates": 7}
        result = api.stats()
        assert result.total_debates == 7
        assert result.total_agents == 0


# ---------------------------------------------------------------------------
# SystemAPI.circuit_breakers / circuit_breakers_async
# ---------------------------------------------------------------------------


class TestCircuitBreakers:
    def test_circuit_breakers(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_BREAKERS
        result = api.circuit_breakers()
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(b, CircuitBreakerStatus) for b in result)
        assert result[0].agent_id == "agent-1"
        assert result[0].is_open is False
        assert result[1].agent_id == "agent-2"
        assert result[1].is_open is True
        mock_client._get.assert_called_once_with("/api/system/circuit-breakers")

    @pytest.mark.asyncio
    async def test_circuit_breakers_async(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_BREAKERS)
        result = await api.circuit_breakers_async()
        assert len(result) == 2
        assert result[1].state == "open"
        mock_client._get_async.assert_called_once_with("/api/system/circuit-breakers")

    def test_circuit_breakers_empty(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"breakers": []}
        result = api.circuit_breakers()
        assert result == []

    def test_circuit_breakers_missing_key(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        """When response lacks 'breakers' key, should return empty list."""
        mock_client._get.return_value = {}
        result = api.circuit_breakers()
        assert result == []

    def test_circuit_breakers_single(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {
            "breakers": [
                {
                    "agent_id": "solo",
                    "state": "half-open",
                    "failure_count": 3,
                    "success_count": 1,
                }
            ]
        }
        result = api.circuit_breakers()
        assert len(result) == 1
        assert result[0].state == "half-open"
        assert result[0].is_open is False


# ---------------------------------------------------------------------------
# SystemAPI.reset_circuit_breaker / reset_circuit_breaker_async
# ---------------------------------------------------------------------------


class TestResetCircuitBreaker:
    def test_reset_success(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"reset": True}
        result = api.reset_circuit_breaker("agent-2")
        assert result is True
        mock_client._post.assert_called_once_with("/api/system/circuit-breakers/agent-2/reset", {})

    def test_reset_failure(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"reset": False}
        result = api.reset_circuit_breaker("agent-unknown")
        assert result is False

    def test_reset_missing_key(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        """When response lacks 'reset' key, should default to False."""
        mock_client._post.return_value = {}
        result = api.reset_circuit_breaker("agent-1")
        assert result is False

    @pytest.mark.asyncio
    async def test_reset_async(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._post_async = AsyncMock(return_value={"reset": True})
        result = await api.reset_circuit_breaker_async("agent-2")
        assert result is True
        mock_client._post_async.assert_called_once_with(
            "/api/system/circuit-breakers/agent-2/reset", {}
        )

    @pytest.mark.asyncio
    async def test_reset_async_failure(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._post_async = AsyncMock(return_value={"reset": False})
        result = await api.reset_circuit_breaker_async("nonexistent")
        assert result is False

    def test_reset_agent_id_with_special_chars(
        self, api: SystemAPI, mock_client: AragoraClient
    ) -> None:
        """Verify agent_id is interpolated into URL path."""
        mock_client._post.return_value = {"reset": True}
        api.reset_circuit_breaker("agent-with-dashes-123")
        called_path = mock_client._post.call_args[0][0]
        assert called_path == "/api/system/circuit-breakers/agent-with-dashes-123/reset"


# ---------------------------------------------------------------------------
# SystemAPI.modes / modes_async
# ---------------------------------------------------------------------------


class TestModes:
    def test_modes(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_MODES
        result = api.modes()
        assert isinstance(result, dict)
        assert result["maintenance"] is False
        assert result["debug"] is True
        mock_client._get.assert_called_once_with("/api/system/modes")

    @pytest.mark.asyncio
    async def test_modes_async(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_MODES)
        result = await api.modes_async()
        assert result["read_only"] is False
        mock_client._get_async.assert_called_once_with("/api/system/modes")

    def test_modes_empty_response(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {}
        result = api.modes()
        assert result == {}

    def test_modes_returns_raw_dict(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        """modes() returns the raw dict from the client, no dataclass wrapping."""
        raw = {"custom_mode": True, "experimental": "beta"}
        mock_client._get.return_value = raw
        result = api.modes()
        assert result is raw


# ---------------------------------------------------------------------------
# Integration-like workflow tests
# ---------------------------------------------------------------------------


class TestWorkflowIntegration:
    """Tests that combine multiple SystemAPI calls in realistic sequences."""

    def test_health_check_workflow(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        """Check health, then inspect breakers if unhealthy."""
        mock_client._get.side_effect = [
            {
                "status": "degraded",
                "version": "2.1.0",
                "uptime_seconds": 3600.0,
                "checks": {"database": True, "agents": False},
                "timestamp": "2026-02-12T10:00:00Z",
            },
            {
                "breakers": [
                    {
                        "agent_id": "claude",
                        "state": "open",
                        "failure_count": 15,
                        "success_count": 0,
                    }
                ]
            },
        ]

        health = api.health()
        assert health.is_healthy is False
        failing = [k for k, v in health.checks.items() if not v]
        assert "agents" in failing

        breakers = api.circuit_breakers()
        assert len(breakers) == 1
        assert breakers[0].is_open is True
        assert breakers[0].agent_id == "claude"

    def test_health_then_reset_breaker_workflow(
        self, api: SystemAPI, mock_client: AragoraClient
    ) -> None:
        """Detect open breaker, reset it, verify it closed."""
        mock_client._get.side_effect = [
            {
                "breakers": [
                    {
                        "agent_id": "gpt-4",
                        "state": "open",
                        "failure_count": 8,
                        "success_count": 2,
                    }
                ]
            },
            {
                "breakers": [
                    {
                        "agent_id": "gpt-4",
                        "state": "closed",
                        "failure_count": 0,
                        "success_count": 0,
                    }
                ]
            },
        ]
        mock_client._post.return_value = {"reset": True}

        # Step 1: List breakers and find open one
        breakers = api.circuit_breakers()
        open_breakers = [b for b in breakers if b.is_open]
        assert len(open_breakers) == 1

        # Step 2: Reset it
        success = api.reset_circuit_breaker(open_breakers[0].agent_id)
        assert success is True

        # Step 3: Verify it is now closed
        breakers_after = api.circuit_breakers()
        assert all(not b.is_open for b in breakers_after)

    def test_full_system_overview_workflow(
        self, api: SystemAPI, mock_client: AragoraClient
    ) -> None:
        """Fetch health, info, stats, modes -- a dashboard overview."""
        mock_client._get.side_effect = [
            SAMPLE_HEALTH,
            SAMPLE_INFO,
            SAMPLE_STATS,
            SAMPLE_MODES,
        ]

        health = api.health()
        info = api.info()
        stats = api.stats()
        modes = api.modes()

        assert health.is_healthy is True
        assert info.version == health.version  # versions should match
        assert stats.total_agents == 12
        assert modes["debug"] is True

        # Verify 4 GET calls were made in sequence
        assert mock_client._get.call_count == 4
        paths = [call.args[0] for call in mock_client._get.call_args_list]
        assert paths == [
            "/api/health",
            "/api/system/info",
            "/api/system/stats",
            "/api/system/modes",
        ]

    @pytest.mark.asyncio
    async def test_async_health_check_workflow(
        self, api: SystemAPI, mock_client: AragoraClient
    ) -> None:
        """Async version: check health, then get info."""
        mock_client._get_async = AsyncMock(side_effect=[SAMPLE_HEALTH, SAMPLE_INFO])
        health = await api.health_async()
        assert health.is_healthy is True

        info = await api.info_async()
        assert info.version == "2.1.0"
        assert mock_client._get_async.call_count == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_health_extra_keys_ignored(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        """from_dict should tolerate extra keys in the response."""
        mock_client._get.return_value = {
            **SAMPLE_HEALTH,
            "extra_field": "should_be_ignored",
            "nested": {"deep": True},
        }
        result = api.health()
        assert result.status == "healthy"
        # Extra fields should not cause errors

    def test_info_agents_empty_list(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {
            "version": "1.0",
            "agents_available": [],
            "features_enabled": [],
        }
        result = api.info()
        assert result.agents_available == []
        assert result.features_enabled == []

    def test_stats_zero_values(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {
            "total_debates": 0,
            "total_agents": 0,
            "active_debates": 0,
            "debates_today": 0,
            "debates_this_week": 0,
            "avg_debate_duration_seconds": 0.0,
            "memory_entries": 0,
            "consensus_rate": 0.0,
        }
        result = api.stats()
        assert result.total_debates == 0
        assert result.consensus_rate == 0.0

    def test_circuit_breaker_from_dict_none_timestamps(self) -> None:
        """last_failure and last_success can be explicitly None."""
        cb = CircuitBreakerStatus.from_dict(
            {
                "agent_id": "a",
                "state": "closed",
                "failure_count": 0,
                "success_count": 5,
                "last_failure": None,
                "last_success": None,
            }
        )
        assert cb.last_failure is None
        assert cb.last_success is None

    def test_health_status_checks_all_false(self) -> None:
        h = HealthStatus.from_dict(
            {
                "status": "healthy",
                "checks": {"a": False, "b": False, "c": False},
            }
        )
        assert h.is_healthy is False

    def test_system_api_stores_client_reference(self, mock_client: AragoraClient) -> None:
        api = SystemAPI(mock_client)
        assert api._client is mock_client

    def test_stats_large_numbers(self, api: SystemAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {
            "total_debates": 999_999_999,
            "total_agents": 500,
            "active_debates": 10_000,
            "debates_today": 50_000,
            "debates_this_week": 300_000,
            "avg_debate_duration_seconds": 9999.99,
            "memory_entries": 100_000_000,
            "consensus_rate": 1.0,
        }
        result = api.stats()
        assert result.total_debates == 999_999_999
        assert result.consensus_rate == 1.0
