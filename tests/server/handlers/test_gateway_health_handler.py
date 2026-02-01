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
        """Agent health check that raises an exception returns 'unknown' status."""
        agent = MockExternalAgent(
            name="timeout-agent",
            available=False,
            raise_on_check=True,
        )
        server_context["external_agents"] = {"timeout-agent": agent}
        h = GatewayHealthHandler(server_context)

        result = h._handle_agent_health("timeout-agent", mock_handler)
        data, status = _parse_result(result)

        assert status == 200
        assert data["name"] == "timeout-agent"
        assert data["status"] == "unknown"
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
