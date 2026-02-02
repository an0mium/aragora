"""
Tests for GatewayHealthHandler - Gateway health monitoring HTTP endpoints.

Tests cover:
- Path routing (can_handle)
- Overall gateway health (healthy, degraded, unhealthy, no agents)
- Individual agent health checks
- Gateway unavailable handling
- Vault status inclusion
- Timestamp inclusion
- Response time tracking
- Agent health check timeouts
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.gateway_health_handler import (
    GatewayHealthHandler,
    _get_vault_status,
)


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


class MockExternalAgent:
    """Mock external framework agent for testing."""

    def __init__(
        self,
        name: str = "test-agent",
        available: bool = True,
        agent_type: str = "crewai",
        base_url: str = "https://example.com",
        delay: float = 0.0,
        raise_on_check: bool = False,
    ):
        self.name = name
        self.agent_type = agent_type
        self.base_url = base_url
        self._available = available
        self._delay = delay
        self._raise_on_check = raise_on_check

    async def is_available(self) -> bool:
        if self._raise_on_check:
            raise ConnectionError("Connection timed out")
        if self._delay > 0:
            import asyncio

            await asyncio.sleep(self._delay)
        return self._available


@pytest.fixture
def server_context():
    """Create a server context dict with empty external agents."""
    return {"external_agents": {}}


@pytest.fixture
def handler(server_context):
    """Create a GatewayHealthHandler with the given server context."""
    return GatewayHealthHandler(server_context)


@pytest.fixture
def mock_handler():
    """Create a mock HTTP request handler."""
    h = MagicMock()
    h.headers = {}
    h.client_address = ("127.0.0.1", 8080)
    return h


def _parse_result(result):
    """Parse a HandlerResult into (body_dict, status_code)."""
    assert result is not None, "Expected a result, got None"
    status_code = result.status_code
    body = result.body
    if isinstance(body, (bytes, str)):
        data = json.loads(body)
    else:
        data = body
    return data, status_code


# ===========================================================================
# Test: can_handle
# ===========================================================================


class TestCanHandle:
    """Test path routing via can_handle."""

    def test_can_handle_health_paths(self, handler):
        """can_handle returns True for valid health paths."""
        assert handler.can_handle("/api/v1/gateway/health") is True
        assert handler.can_handle("/api/v1/gateway/agents/crewai-agent/health") is True
        assert handler.can_handle("/api/v1/gateway/agents/my_agent/health") is True

    def test_cannot_handle_other_paths(self, handler):
        """can_handle returns False for non-health paths."""
        assert handler.can_handle("/api/v1/gateway/devices") is False
        assert handler.can_handle("/api/v1/gateway") is False
        assert handler.can_handle("/api/v1/debates") is False
        assert handler.can_handle("/api/v1/gateway/agents") is False
        assert handler.can_handle("/api/v1/gateway/health/extra") is False
        assert handler.can_handle("/api/v1/gateway/agents/foo/bar/health") is False

    def test_health_empty_path_returns_none(self, handler, mock_handler):
        """Non-matching path returns None from handle."""
        result = handler.handle("/api/v1/debates", {}, mock_handler)
        assert result is None


# ===========================================================================
# Test: Overall health endpoint
# ===========================================================================


class TestOverallHealth:
    """Test GET /api/v1/gateway/health."""

    @patch(
        "aragora.server.handlers.gateway_health_handler.GATEWAY_AVAILABLE",
        True,
    )
    def test_get_overall_health_healthy(self, server_context, mock_handler):
        """All agents healthy returns 'healthy' status."""
        agent_a = MockExternalAgent(name="agent-a", available=True, agent_type="crewai")
        agent_b = MockExternalAgent(name="agent-b", available=True, agent_type="langchain")
        server_context["external_agents"] = {
            "agent-a": agent_a,
            "agent-b": agent_b,
        }
        h = GatewayHealthHandler(server_context)

        result = h._handle_overall_health(mock_handler)
        data, status = _parse_result(result)

        assert status == 200
        assert data["status"] == "healthy"
        assert data["agents"]["agent-a"]["status"] == "healthy"
        assert data["agents"]["agent-b"]["status"] == "healthy"
        assert data["gateway"]["external_agents_available"] is True
        assert data["gateway"]["active_executions"] == 0

    @patch(
        "aragora.server.handlers.gateway_health_handler.GATEWAY_AVAILABLE",
        True,
    )
    def test_get_overall_health_degraded(self, server_context, mock_handler):
        """Some agents unhealthy returns 'degraded' status."""
        agent_ok = MockExternalAgent(name="ok", available=True)
        agent_bad = MockExternalAgent(name="bad", available=False)
        server_context["external_agents"] = {
            "ok-agent": agent_ok,
            "bad-agent": agent_bad,
        }
        h = GatewayHealthHandler(server_context)

        result = h._handle_overall_health(mock_handler)
        data, status = _parse_result(result)

        assert status == 200
        assert data["status"] == "degraded"
        assert data["agents"]["ok-agent"]["status"] == "healthy"
        assert data["agents"]["bad-agent"]["status"] == "unhealthy"

    @patch(
        "aragora.server.handlers.gateway_health_handler.GATEWAY_AVAILABLE",
        True,
    )
    def test_get_overall_health_unhealthy(self, server_context, mock_handler):
        """All agents unhealthy returns 'unhealthy' status."""
        agent_a = MockExternalAgent(name="a", available=False)
        agent_b = MockExternalAgent(name="b", available=False)
        server_context["external_agents"] = {
            "agent-a": agent_a,
            "agent-b": agent_b,
        }
        h = GatewayHealthHandler(server_context)

        result = h._handle_overall_health(mock_handler)
        data, status = _parse_result(result)

        assert status == 200
        assert data["status"] == "unhealthy"

    @patch(
        "aragora.server.handlers.gateway_health_handler.GATEWAY_AVAILABLE",
        True,
    )
    def test_get_overall_health_no_agents(self, server_context, mock_handler):
        """No agents registered returns 'healthy' status with empty agents dict."""
        server_context["external_agents"] = {}
        h = GatewayHealthHandler(server_context)

        result = h._handle_overall_health(mock_handler)
        data, status = _parse_result(result)

        assert status == 200
        assert data["status"] == "healthy"
        assert data["agents"] == {}
        assert data["gateway"]["external_agents_available"] is True

    @patch(
        "aragora.server.handlers.gateway_health_handler.GATEWAY_AVAILABLE",
        False,
    )
    def test_get_overall_health_gateway_unavailable(self, server_context, mock_handler):
        """Returns 503 if gateway module is not available."""
        h = GatewayHealthHandler(server_context)

        result = h._handle_overall_health(mock_handler)
        data, status = _parse_result(result)

        assert status == 503

    @patch(
        "aragora.server.handlers.gateway_health_handler.GATEWAY_AVAILABLE",
        True,
    )
    def test_health_includes_vault_status(self, server_context, mock_handler):
        """Response includes credential vault status."""
        server_context["external_agents"] = {}
        h = GatewayHealthHandler(server_context)

        result = h._handle_overall_health(mock_handler)
        data, status = _parse_result(result)

        assert status == 200
        assert "credential_vault_status" in data["gateway"]
        assert data["gateway"]["credential_vault_status"] in (
            "sealed",
            "unsealed",
            "unavailable",
        )

    @patch(
        "aragora.server.handlers.gateway_health_handler.GATEWAY_AVAILABLE",
        True,
    )
    def test_health_includes_timestamp(self, server_context, mock_handler):
        """Response includes ISO timestamp."""
        server_context["external_agents"] = {}
        h = GatewayHealthHandler(server_context)

        result = h._handle_overall_health(mock_handler)
        data, status = _parse_result(result)

        assert status == 200
        assert "timestamp" in data
        # Verify it is a parseable ISO timestamp
        ts = data["timestamp"]
        # ISO format should contain 'T' and be parseable
        assert "T" in ts
        # Should be roughly now (within last 5 seconds)
        parsed = datetime.fromisoformat(ts)
        now = datetime.now(timezone.utc)
        diff = abs((now - parsed).total_seconds())
        assert diff < 5, f"Timestamp drift too large: {diff}s"


# ===========================================================================
# Test: Individual agent health endpoint
# ===========================================================================


class TestAgentHealth:
    """Test GET /api/v1/gateway/agents/{name}/health."""

    @patch(
        "aragora.server.handlers.gateway_health_handler.GATEWAY_AVAILABLE",
        True,
    )
    def test_get_agent_health_healthy(self, server_context, mock_handler):
        """Individual agent that is healthy returns correct status."""
        agent = MockExternalAgent(
            name="my-agent",
            available=True,
            agent_type="crewai",
            base_url="https://agent.example.com",
        )
        server_context["external_agents"] = {"my-agent": agent}
        h = GatewayHealthHandler(server_context)

        result = h._handle_agent_health("my-agent", mock_handler)
        data, status = _parse_result(result)

        assert status == 200
        assert data["name"] == "my-agent"
        assert data["status"] == "healthy"
        assert data["framework"] == "crewai"
        assert data["base_url"] == "https://agent.example.com"
        assert "last_check" in data
        assert "response_time_ms" in data

    @patch(
        "aragora.server.handlers.gateway_health_handler.GATEWAY_AVAILABLE",
        True,
    )
    def test_get_agent_health_unhealthy(self, server_context, mock_handler):
        """Individual agent that is not available returns 'unhealthy' status."""
        agent = MockExternalAgent(name="bad-agent", available=False)
        server_context["external_agents"] = {"bad-agent": agent}
        h = GatewayHealthHandler(server_context)

        result = h._handle_agent_health("bad-agent", mock_handler)
        data, status = _parse_result(result)

        assert status == 200
        assert data["name"] == "bad-agent"
        assert data["status"] == "unhealthy"

    @patch(
        "aragora.server.handlers.gateway_health_handler.GATEWAY_AVAILABLE",
        True,
    )
    def test_get_agent_health_not_found(self, server_context, mock_handler):
        """Agent name not in registry returns 404."""
        server_context["external_agents"] = {}
        h = GatewayHealthHandler(server_context)

        result = h._handle_agent_health("nonexistent", mock_handler)
        data, status = _parse_result(result)

        assert status == 404

    @patch(
        "aragora.server.handlers.gateway_health_handler.GATEWAY_AVAILABLE",
        True,
    )
    def test_get_agent_health_response_time(self, server_context, mock_handler):
        """Response includes response_time_ms field with a numeric value."""
        agent = MockExternalAgent(name="timed-agent", available=True)
        server_context["external_agents"] = {"timed-agent": agent}
        h = GatewayHealthHandler(server_context)

        result = h._handle_agent_health("timed-agent", mock_handler)
        data, status = _parse_result(result)

        assert status == 200
        assert "response_time_ms" in data
        assert isinstance(data["response_time_ms"], (int, float))
        assert data["response_time_ms"] >= 0

    @patch(
        "aragora.server.handlers.gateway_health_handler.GATEWAY_AVAILABLE",
        True,
    )
    def test_get_agent_health_timeout(self, server_context, mock_handler):
        """Agent health check that raises a connection error returns 'unavailable' status.

        Note: The handler distinguishes between:
        - 'unknown' for configuration errors (AttributeError, TypeError, ValueError)
        - 'unavailable' for runtime errors (RuntimeError, TimeoutError, OSError, ConnectionError)
        """
        agent = MockExternalAgent(
            name="timeout-agent",
            available=False,
            raise_on_check=True,  # Raises ConnectionError
        )
        server_context["external_agents"] = {"timeout-agent": agent}
        h = GatewayHealthHandler(server_context)

        result = h._handle_agent_health("timeout-agent", mock_handler)
        data, status = _parse_result(result)

        assert status == 200
        assert data["name"] == "timeout-agent"
        assert data["status"] == "unavailable"  # Runtime error -> unavailable
        assert "response_time_ms" in data

    @patch(
        "aragora.server.handlers.gateway_health_handler.GATEWAY_AVAILABLE",
        False,
    )
    def test_get_agent_health_gateway_unavailable(self, server_context, mock_handler):
        """Returns 503 if gateway module is not available for agent health."""
        h = GatewayHealthHandler(server_context)

        result = h._handle_agent_health("any-agent", mock_handler)
        data, status = _parse_result(result)

        assert status == 503


# ===========================================================================
# Test: Vault status helper
# ===========================================================================


class TestVaultStatus:
    """Test the _get_vault_status helper."""

    def test_vault_available(self):
        """When credential vault module is importable, returns 'sealed'."""
        with patch(
            "aragora.server.handlers.gateway_health_handler._get_vault_status",
            return_value="sealed",
        ):
            # Re-call to check the return
            assert _get_vault_status() in ("sealed", "unsealed", "unavailable")

    def test_vault_unavailable(self):
        """When credential vault module import fails, returns 'unavailable'."""
        import importlib

        with patch.dict("sys.modules", {"aragora.gateway.security.credential_vault": None}):
            # Force import failure
            with patch(
                "builtins.__import__",
                side_effect=lambda name, *args, **kwargs: (
                    (_ for _ in ()).throw(ImportError("no module"))
                    if "credential_vault" in name
                    else importlib.__import__(name, *args, **kwargs)
                ),
            ):
                result = _get_vault_status()
                assert result == "unavailable"


# ===========================================================================
# Test: Circuit Breaker Integration
# ===========================================================================


class TestCircuitBreaker:
    """Tests for circuit breaker integration."""

    def test_get_gateway_health_circuit_breaker(self):
        """Test getting circuit breaker instance."""
        from aragora.server.handlers.gateway_health_handler import (
            get_gateway_health_circuit_breaker,
        )

        cb = get_gateway_health_circuit_breaker()
        assert cb is not None
        assert cb.name == "gateway_health_handler"

    def test_get_gateway_health_circuit_breaker_status(self):
        """Test getting circuit breaker status dict."""
        from aragora.server.handlers.gateway_health_handler import (
            get_gateway_health_circuit_breaker_status,
        )

        status = get_gateway_health_circuit_breaker_status()
        assert isinstance(status, dict)
        # Check that status contains expected keys from to_dict()
        assert "config" in status
        assert "single_mode" in status
        assert status["config"]["failure_threshold"] == 5

    def test_circuit_breaker_is_singleton(self):
        """Test circuit breaker returns same instance."""
        from aragora.server.handlers.gateway_health_handler import (
            get_gateway_health_circuit_breaker,
        )

        cb1 = get_gateway_health_circuit_breaker()
        cb2 = get_gateway_health_circuit_breaker()
        assert cb1 is cb2

    def test_circuit_breaker_has_correct_config(self):
        """Test circuit breaker has expected configuration."""
        from aragora.server.handlers.gateway_health_handler import (
            get_gateway_health_circuit_breaker,
        )

        cb = get_gateway_health_circuit_breaker()
        assert cb.failure_threshold == 5
        assert cb.cooldown_seconds == 30.0


# ===========================================================================
# Test: Rate Limiting
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting on handler methods."""

    def test_handle_has_rate_limit_decorator(self, handler):
        """Test that handle method has rate limit decorator."""
        # Check if the method has the _rate_limited attribute set by the decorator
        assert hasattr(handler.handle, "_rate_limited") or hasattr(
            GatewayHealthHandler.handle, "__wrapped__"
        )


# ===========================================================================
# Test: Handler Initialization
# ===========================================================================


class TestHandlerInit:
    """Tests for handler initialization."""

    def test_init_with_none_context(self):
        """Test handler initialization with None context."""
        handler = GatewayHealthHandler(None)
        assert handler.ctx == {}

    def test_init_with_context(self):
        """Test handler initialization with context dict."""
        ctx = {"external_agents": {"agent-1": MagicMock()}}
        handler = GatewayHealthHandler(ctx)
        assert handler.ctx == ctx

    def test_routes_defined(self):
        """Test that ROUTES class attribute is defined."""
        assert hasattr(GatewayHealthHandler, "ROUTES")
        assert "/api/v1/gateway/health" in GatewayHealthHandler.ROUTES


# ===========================================================================
# Test: Agent Check with Sync is_available
# ===========================================================================


class TestSyncAgentCheck:
    """Tests for agents with synchronous is_available method."""

    class SyncAgent:
        """Agent with sync is_available method."""

        def __init__(self, available: bool = True):
            self._available = available
            self.agent_type = "sync_agent"
            self.base_url = "https://sync.example.com"

        def is_available(self) -> bool:
            return self._available

    @patch(
        "aragora.server.handlers.gateway_health_handler.GATEWAY_AVAILABLE",
        True,
    )
    def test_sync_agent_healthy(self, server_context, mock_handler):
        """Sync agent returning True shows as healthy."""
        agent = self.SyncAgent(available=True)
        server_context["external_agents"] = {"sync-agent": agent}
        h = GatewayHealthHandler(server_context)

        result = h._handle_agent_health("sync-agent", mock_handler)
        data, status = _parse_result(result)

        assert status == 200
        assert data["status"] == "healthy"

    @patch(
        "aragora.server.handlers.gateway_health_handler.GATEWAY_AVAILABLE",
        True,
    )
    def test_sync_agent_unhealthy(self, server_context, mock_handler):
        """Sync agent returning False shows as unhealthy."""
        agent = self.SyncAgent(available=False)
        server_context["external_agents"] = {"sync-agent": agent}
        h = GatewayHealthHandler(server_context)

        result = h._handle_agent_health("sync-agent", mock_handler)
        data, status = _parse_result(result)

        assert status == 200
        assert data["status"] == "unhealthy"


# ===========================================================================
# Test: Agent Without is_available
# ===========================================================================


class TestAgentWithoutIsAvailable:
    """Tests for agents without is_available method."""

    class AgentNoMethod:
        """Agent without is_available method."""

        def __init__(self):
            self.agent_type = "no_method_agent"

    @patch(
        "aragora.server.handlers.gateway_health_handler.GATEWAY_AVAILABLE",
        True,
    )
    def test_agent_without_is_available(self, server_context, mock_handler):
        """Agent without is_available method is treated as unavailable."""
        agent = self.AgentNoMethod()
        server_context["external_agents"] = {"no-method-agent": agent}
        h = GatewayHealthHandler(server_context)

        result = h._handle_agent_health("no-method-agent", mock_handler)
        data, status = _parse_result(result)

        assert status == 200
        assert data["status"] == "unhealthy"


# ===========================================================================
# Test: Error Types in Agent Health Check
# ===========================================================================


class TestAgentHealthCheckErrors:
    """Tests for different error types in agent health checks."""

    class AgentTypeError:
        """Agent that raises TypeError."""

        def __init__(self):
            self.agent_type = "type_error_agent"

        def is_available(self):
            raise TypeError("Invalid type")

    class AgentValueError:
        """Agent that raises ValueError."""

        def __init__(self):
            self.agent_type = "value_error_agent"

        def is_available(self):
            raise ValueError("Invalid value")

    class AgentAttributeError:
        """Agent that raises AttributeError."""

        def __init__(self):
            self.agent_type = "attr_error_agent"

        def is_available(self):
            raise AttributeError("Missing attribute")

    class AgentOSError:
        """Agent that raises OSError."""

        def __init__(self):
            self.agent_type = "os_error_agent"

        def is_available(self):
            raise OSError("Network error")

    @patch(
        "aragora.server.handlers.gateway_health_handler.GATEWAY_AVAILABLE",
        True,
    )
    def test_type_error_returns_unknown(self, server_context, mock_handler):
        """TypeError in is_available returns 'unknown' status."""
        agent = self.AgentTypeError()
        server_context["external_agents"] = {"type-error-agent": agent}
        h = GatewayHealthHandler(server_context)

        result = h._handle_agent_health("type-error-agent", mock_handler)
        data, status = _parse_result(result)

        assert status == 200
        assert data["status"] == "unknown"

    @patch(
        "aragora.server.handlers.gateway_health_handler.GATEWAY_AVAILABLE",
        True,
    )
    def test_value_error_returns_unknown(self, server_context, mock_handler):
        """ValueError in is_available returns 'unknown' status."""
        agent = self.AgentValueError()
        server_context["external_agents"] = {"value-error-agent": agent}
        h = GatewayHealthHandler(server_context)

        result = h._handle_agent_health("value-error-agent", mock_handler)
        data, status = _parse_result(result)

        assert status == 200
        assert data["status"] == "unknown"

    @patch(
        "aragora.server.handlers.gateway_health_handler.GATEWAY_AVAILABLE",
        True,
    )
    def test_attribute_error_returns_unknown(self, server_context, mock_handler):
        """AttributeError in is_available returns 'unknown' status."""
        agent = self.AgentAttributeError()
        server_context["external_agents"] = {"attr-error-agent": agent}
        h = GatewayHealthHandler(server_context)

        result = h._handle_agent_health("attr-error-agent", mock_handler)
        data, status = _parse_result(result)

        assert status == 200
        assert data["status"] == "unknown"

    @patch(
        "aragora.server.handlers.gateway_health_handler.GATEWAY_AVAILABLE",
        True,
    )
    def test_os_error_returns_unavailable(self, server_context, mock_handler):
        """OSError in is_available returns 'unavailable' status."""
        agent = self.AgentOSError()
        server_context["external_agents"] = {"os-error-agent": agent}
        h = GatewayHealthHandler(server_context)

        result = h._handle_agent_health("os-error-agent", mock_handler)
        data, status = _parse_result(result)

        assert status == 200
        assert data["status"] == "unavailable"


# ===========================================================================
# Test: Overall Health with Mixed Agent Errors
# ===========================================================================


class TestOverallHealthMixedErrors:
    """Tests for overall health with various agent error conditions."""

    @patch(
        "aragora.server.handlers.gateway_health_handler.GATEWAY_AVAILABLE",
        True,
    )
    def test_overall_health_with_unknown_agent(self, server_context, mock_handler):
        """Overall health includes agents with 'unknown' status."""

        class ErrorAgent:
            agent_type = "error_agent"

            def is_available(self):
                raise ValueError("Config error")

        agent_good = MockExternalAgent(name="good", available=True)
        agent_error = ErrorAgent()
        server_context["external_agents"] = {
            "good-agent": agent_good,
            "error-agent": agent_error,
        }
        h = GatewayHealthHandler(server_context)

        result = h._handle_overall_health(mock_handler)
        data, status = _parse_result(result)

        assert status == 200
        assert data["status"] == "degraded"  # One unknown = degraded
        assert data["agents"]["good-agent"]["status"] == "healthy"
        assert data["agents"]["error-agent"]["status"] == "unknown"


# ===========================================================================
# Test: External Agents Context Handling
# ===========================================================================


class TestExternalAgentsContext:
    """Tests for external_agents context handling."""

    @patch(
        "aragora.server.handlers.gateway_health_handler.GATEWAY_AVAILABLE",
        True,
    )
    def test_none_external_agents(self, mock_handler):
        """Handler with None external_agents works correctly."""
        ctx = {"external_agents": None}
        h = GatewayHealthHandler(ctx)

        result = h._handle_overall_health(mock_handler)
        data, status = _parse_result(result)

        assert status == 200
        assert data["status"] == "healthy"
        assert data["agents"] == {}

    @patch(
        "aragora.server.handlers.gateway_health_handler.GATEWAY_AVAILABLE",
        True,
    )
    def test_missing_external_agents_key(self, mock_handler):
        """Handler with missing external_agents key works correctly."""
        ctx = {}
        h = GatewayHealthHandler(ctx)

        result = h._handle_overall_health(mock_handler)
        data, status = _parse_result(result)

        assert status == 200
        assert data["status"] == "healthy"
        assert data["agents"] == {}
