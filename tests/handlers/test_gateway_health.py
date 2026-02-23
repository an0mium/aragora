"""Tests for gateway health handler (aragora/server/handlers/gateway_health_handler.py).

Covers all routes and behavior of the GatewayHealthHandler class:
- can_handle() routing for all ROUTES and non-matching paths
- GET /api/v1/gateway/health                - Overall gateway health status
- GET /api/v1/gateway/agents/{name}/health  - Individual agent health check
- Gateway unavailable (503) responses when GATEWAY_AVAILABLE is False
- Overall health status calculation (healthy, degraded, unhealthy)
- Agent health check with sync/async is_available()
- Agent health check exception handling (AttributeError, TypeError, ValueError,
  RuntimeError, TimeoutError, OSError)
- Credential vault status detection
- Circuit breaker helper functions
- Module exports
- Security tests (path traversal, injection)
- Edge cases for path parsing
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.gateway_health_handler import (
    GatewayHealthHandler,
    get_gateway_health_circuit_breaker,
    get_gateway_health_circuit_breaker_status,
    _get_vault_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract the body dict from a HandlerResult."""
    if hasattr(result, "to_dict"):
        d = result.to_dict()
        return d.get("body", d)
    if isinstance(result, dict):
        return result.get("body", result)
    try:
        body, status, _ = result
        return body if isinstance(body, dict) else {}
    except (TypeError, ValueError):
        return {}


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if hasattr(result, "status_code"):
        return result.status_code
    if isinstance(result, dict):
        return result.get("status_code", result.get("status", 200))
    try:
        _, status, _ = result
        return status
    except (TypeError, ValueError):
        return 200


class MockHTTPHandler:
    """Mock HTTP handler used by BaseHandler.read_json_body."""

    def __init__(self, body: dict | None = None):
        self.rfile = MagicMock()
        self._body = body
        if body is not None:
            body_bytes = json.dumps(body).encode()
            self.rfile.read.return_value = body_bytes
            self.headers = {
                "Content-Length": str(len(body_bytes)),
                "Content-Type": "application/json",
            }
        else:
            self.rfile.read.return_value = b"{}"
            self.headers = {
                "Content-Length": "2",
                "Content-Type": "application/json",
            }
        self.client_address = ("127.0.0.1", 54321)


# ---------------------------------------------------------------------------
# Mock agent classes
# ---------------------------------------------------------------------------


class MockAgent:
    """Mock external agent with is_available method."""

    def __init__(
        self,
        available: bool = True,
        agent_type: str = "crewai",
        base_url: str | None = "https://agent.example.com",
    ):
        self._available = available
        self.agent_type = agent_type
        self.base_url = base_url

    def is_available(self) -> bool:
        return self._available


class MockAgentNoIsAvailable:
    """Mock agent without is_available method."""

    agent_type = "custom"
    base_url = "https://custom.example.com"


class MockAgentRaisesAttributeError:
    """Mock agent whose is_available raises AttributeError."""

    agent_type = "faulty"

    def is_available(self) -> bool:
        raise AttributeError("No connection attribute")


class MockAgentRaisesTypeError:
    """Mock agent whose is_available raises TypeError."""

    agent_type = "broken"

    def is_available(self) -> bool:
        raise TypeError("Invalid type in config")


class MockAgentRaisesValueError:
    """Mock agent whose is_available raises ValueError."""

    agent_type = "misconfigured"

    def is_available(self) -> bool:
        raise ValueError("Invalid config value")


class MockAgentRaisesRuntimeError:
    """Mock agent whose is_available raises RuntimeError."""

    agent_type = "runtime_fail"

    def is_available(self) -> bool:
        raise RuntimeError("Agent runtime failure")


class MockAgentRaisesTimeoutError:
    """Mock agent whose is_available raises TimeoutError."""

    agent_type = "timeout_agent"

    def is_available(self) -> bool:
        raise TimeoutError("Connection timed out")


class MockAgentRaisesOSError:
    """Mock agent whose is_available raises OSError."""

    agent_type = "network_fail"

    def is_available(self) -> bool:
        raise OSError("Network unreachable")


class MockAgentAsyncAvailable:
    """Mock agent with async is_available method."""

    agent_type = "async_agent"
    base_url = "https://async.example.com"

    async def is_available(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_gateway_available():
    """Ensure GATEWAY_AVAILABLE is True by default for most tests."""
    with patch(
        "aragora.server.handlers.gateway_health_handler.GATEWAY_AVAILABLE", True
    ):
        yield


@pytest.fixture(autouse=True)
def _patch_rate_limit():
    """Bypass rate limiting for handler tests."""
    with patch(
        "aragora.server.handlers.gateway_health_handler.rate_limit",
        lambda **kwargs: lambda fn: fn,
    ):
        yield


@pytest.fixture
def handler():
    """Create a GatewayHealthHandler instance with empty context."""
    return GatewayHealthHandler(ctx={})


@pytest.fixture
def handler_with_agents():
    """Create a handler with pre-registered healthy agents."""
    return GatewayHealthHandler(ctx={
        "external_agents": {
            "agent-1": MockAgent(available=True, agent_type="crewai"),
            "agent-2": MockAgent(available=True, agent_type="langgraph"),
        }
    })


@pytest.fixture
def handler_mixed_agents():
    """Create a handler with a mix of healthy and unhealthy agents."""
    return GatewayHealthHandler(ctx={
        "external_agents": {
            "healthy": MockAgent(available=True, agent_type="crewai"),
            "unhealthy": MockAgent(available=False, agent_type="langgraph"),
        }
    })


@pytest.fixture
def handler_all_unhealthy():
    """Create a handler with only unhealthy agents."""
    return GatewayHealthHandler(ctx={
        "external_agents": {
            "down-1": MockAgent(available=False, agent_type="crewai"),
            "down-2": MockAgent(available=False, agent_type="langgraph"),
        }
    })


# ===========================================================================
# can_handle routing tests
# ===========================================================================


class TestCanHandle:
    """Test the can_handle path routing."""

    def test_handles_gateway_health(self, handler):
        assert handler.can_handle("/api/v1/gateway/health") is True

    def test_handles_agent_health(self, handler):
        assert handler.can_handle("/api/v1/gateway/agents/my-agent/health") is True

    def test_handles_agent_health_with_underscores(self, handler):
        assert handler.can_handle("/api/v1/gateway/agents/my_agent/health") is True

    def test_rejects_non_gateway_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_rejects_root(self, handler):
        assert handler.can_handle("/") is False

    def test_rejects_partial_prefix(self, handler):
        assert handler.can_handle("/api/v1/gateway/heal") is False

    def test_rejects_gateway_agents_without_health(self, handler):
        assert handler.can_handle("/api/v1/gateway/agents/my-agent") is False

    def test_rejects_gateway_health_with_extra_segment(self, handler):
        assert handler.can_handle("/api/v1/gateway/health/extra") is False

    def test_rejects_agents_health_wrong_segment_count(self, handler):
        """Path with wrong number of segments should be rejected."""
        # 7 segments: /api/v1/gateway/agents/name/extra/health
        assert handler.can_handle("/api/v1/gateway/agents/name/extra/health") is False

    def test_rejects_agents_base_path(self, handler):
        assert handler.can_handle("/api/v1/gateway/agents") is False

    def test_rejects_empty_string(self, handler):
        assert handler.can_handle("") is False

    def test_rejects_gateway_devices(self, handler):
        assert handler.can_handle("/api/v1/gateway/devices") is False

    def test_handles_agent_health_numeric_name(self, handler):
        assert handler.can_handle("/api/v1/gateway/agents/123/health") is True


# ===========================================================================
# ROUTES class attribute
# ===========================================================================


class TestRoutesAttribute:
    """Verify the ROUTES class attribute lists expected patterns."""

    def test_routes_contains_gateway_health(self):
        assert "/api/v1/gateway/health" in GatewayHealthHandler.ROUTES

    def test_routes_contains_agent_health_wildcard(self):
        assert "/api/v1/gateway/agents/*/health" in GatewayHealthHandler.ROUTES

    def test_routes_count(self):
        assert len(GatewayHealthHandler.ROUTES) == 2


# ===========================================================================
# Handler initialization
# ===========================================================================


class TestInitialization:
    """Test handler initialization."""

    def test_default_ctx(self):
        h = GatewayHealthHandler()
        assert h.ctx == {}

    def test_custom_ctx(self):
        ctx = {"key": "value"}
        h = GatewayHealthHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_none_ctx_becomes_empty_dict(self):
        h = GatewayHealthHandler(ctx=None)
        assert h.ctx == {}


# ===========================================================================
# GET /api/v1/gateway/health - Overall health
# ===========================================================================


class TestOverallHealth:
    """Test GET /api/v1/gateway/health."""

    def test_healthy_with_no_agents(self, handler):
        """No agents registered => healthy status."""
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "healthy"
        assert body["agents"] == {}
        assert "timestamp" in body
        assert body["gateway"]["external_agents_available"] is True
        assert body["gateway"]["active_executions"] == 0

    def test_healthy_with_all_agents_up(self, handler_with_agents):
        """All agents available => healthy."""
        http = MockHTTPHandler()
        result = handler_with_agents.handle("/api/v1/gateway/health", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "healthy"
        assert len(body["agents"]) == 2
        for name, info in body["agents"].items():
            assert info["status"] == "healthy"
            assert "framework" in info
            assert "last_check" in info
            assert "latency_ms" in info

    def test_degraded_with_mixed_agents(self, handler_mixed_agents):
        """Some agents down => degraded."""
        http = MockHTTPHandler()
        result = handler_mixed_agents.handle("/api/v1/gateway/health", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "degraded"
        assert body["agents"]["healthy"]["status"] == "healthy"
        assert body["agents"]["unhealthy"]["status"] == "unhealthy"

    def test_unhealthy_with_all_agents_down(self, handler_all_unhealthy):
        """All agents down => unhealthy."""
        http = MockHTTPHandler()
        result = handler_all_unhealthy.handle("/api/v1/gateway/health", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "unhealthy"

    def test_agent_without_is_available(self, handler):
        """Agent without is_available => unhealthy (returns False)."""
        handler.ctx["external_agents"] = {
            "no-check": MockAgentNoIsAvailable(),
        }
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["agents"]["no-check"]["status"] == "unhealthy"

    def test_agent_raises_attribute_error(self, handler):
        """Agent raising AttributeError => unknown status with error type."""
        handler.ctx["external_agents"] = {
            "faulty": MockAgentRaisesAttributeError(),
        }
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["agents"]["faulty"]["status"] == "unknown"
        assert body["agents"]["faulty"]["error"] == "AttributeError"
        assert body["status"] == "unhealthy"

    def test_agent_raises_type_error(self, handler):
        """Agent raising TypeError => unknown status."""
        handler.ctx["external_agents"] = {
            "broken": MockAgentRaisesTypeError(),
        }
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["agents"]["broken"]["status"] == "unknown"
        assert body["agents"]["broken"]["error"] == "TypeError"

    def test_agent_raises_value_error(self, handler):
        """Agent raising ValueError => unknown status."""
        handler.ctx["external_agents"] = {
            "bad-val": MockAgentRaisesValueError(),
        }
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["agents"]["bad-val"]["status"] == "unknown"
        assert body["agents"]["bad-val"]["error"] == "ValueError"

    def test_agent_raises_runtime_error(self, handler):
        """Agent raising RuntimeError => unavailable status."""
        handler.ctx["external_agents"] = {
            "crashed": MockAgentRaisesRuntimeError(),
        }
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["agents"]["crashed"]["status"] == "unavailable"
        assert body["agents"]["crashed"]["error"] == "RuntimeError"

    def test_agent_raises_timeout_error(self, handler):
        """Agent raising TimeoutError => unavailable status."""
        handler.ctx["external_agents"] = {
            "timeout": MockAgentRaisesTimeoutError(),
        }
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["agents"]["timeout"]["status"] == "unavailable"
        assert body["agents"]["timeout"]["error"] == "TimeoutError"

    def test_agent_raises_os_error(self, handler):
        """Agent raising OSError => unavailable status."""
        handler.ctx["external_agents"] = {
            "network": MockAgentRaisesOSError(),
        }
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["agents"]["network"]["status"] == "unavailable"
        assert body["agents"]["network"]["error"] == "OSError"

    def test_mixed_error_types(self, handler):
        """Mixed agent errors: some unknown, some unavailable, some healthy."""
        handler.ctx["external_agents"] = {
            "good": MockAgent(available=True),
            "attr-err": MockAgentRaisesAttributeError(),
            "runtime-err": MockAgentRaisesRuntimeError(),
        }
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "degraded"
        assert body["agents"]["good"]["status"] == "healthy"
        assert body["agents"]["attr-err"]["status"] == "unknown"
        assert body["agents"]["runtime-err"]["status"] == "unavailable"

    def test_framework_type_from_agent(self, handler_with_agents):
        """Framework field should come from agent.agent_type."""
        http = MockHTTPHandler()
        result = handler_with_agents.handle("/api/v1/gateway/health", {}, http)
        body = _body(result)
        assert body["agents"]["agent-1"]["framework"] == "crewai"
        assert body["agents"]["agent-2"]["framework"] == "langgraph"

    def test_framework_unknown_when_no_agent_type(self, handler):
        """Agent without agent_type => framework 'unknown'."""
        agent = MagicMock(spec=[])  # no attributes
        agent.is_available = MagicMock(return_value=True)
        handler.ctx["external_agents"] = {"bare": agent}
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health", {}, http)
        body = _body(result)
        assert body["agents"]["bare"]["framework"] == "unknown"

    def test_vault_status_included(self, handler):
        """Response should include credential vault status."""
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health", {}, http)
        body = _body(result)
        assert "credential_vault_status" in body["gateway"]

    def test_timestamp_is_iso_format(self, handler):
        """Timestamp should be in ISO format."""
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health", {}, http)
        body = _body(result)
        ts = body["timestamp"]
        assert "T" in ts
        assert "+" in ts or "Z" in ts or ts.endswith("+00:00")

    def test_ctx_external_agents_none(self, handler):
        """ctx with external_agents=None => treated as empty."""
        handler.ctx["external_agents"] = None
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "healthy"
        assert body["agents"] == {}

    def test_latency_ms_non_negative(self, handler_with_agents):
        """Latency should be a non-negative number."""
        http = MockHTTPHandler()
        result = handler_with_agents.handle("/api/v1/gateway/health", {}, http)
        body = _body(result)
        for name, info in body["agents"].items():
            assert info["latency_ms"] >= 0

    def test_many_agents_all_healthy(self, handler):
        """Health check with many agents all healthy."""
        handler.ctx["external_agents"] = {
            f"agent-{i}": MockAgent(available=True, agent_type=f"type-{i}")
            for i in range(20)
        }
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "healthy"
        assert len(body["agents"]) == 20

    def test_one_unhealthy_of_many(self, handler):
        """One unhealthy agent among many => degraded."""
        agents = {
            f"agent-{i}": MockAgent(available=True)
            for i in range(9)
        }
        agents["sick"] = MockAgent(available=False)
        handler.ctx["external_agents"] = agents
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "degraded"


# ===========================================================================
# GET /api/v1/gateway/agents/{name}/health - Individual agent health
# ===========================================================================


class TestAgentHealth:
    """Test GET /api/v1/gateway/agents/{name}/health."""

    def test_healthy_agent(self, handler_with_agents):
        """Agent that is available => healthy status."""
        http = MockHTTPHandler()
        result = handler_with_agents.handle(
            "/api/v1/gateway/agents/agent-1/health", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["name"] == "agent-1"
        assert body["status"] == "healthy"
        assert body["framework"] == "crewai"
        assert "last_check" in body
        assert "response_time_ms" in body
        assert body["response_time_ms"] >= 0

    def test_unhealthy_agent(self):
        """Agent that is not available => unhealthy status."""
        h = GatewayHealthHandler(ctx={
            "external_agents": {
                "down": MockAgent(available=False, agent_type="crewai"),
            }
        })
        http = MockHTTPHandler()
        result = h.handle("/api/v1/gateway/agents/down/health", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["name"] == "down"
        assert body["status"] == "unhealthy"

    def test_agent_not_found(self, handler):
        """Non-existent agent => 404."""
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/nonexistent/health", {}, http
        )
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    def test_agent_not_found_with_agents_registered(self, handler_with_agents):
        """Agent not in registry => 404 even when other agents exist."""
        http = MockHTTPHandler()
        result = handler_with_agents.handle(
            "/api/v1/gateway/agents/missing/health", {}, http
        )
        assert _status(result) == 404

    def test_agent_without_is_available(self, handler):
        """Agent without is_available method => unhealthy."""
        handler.ctx["external_agents"] = {
            "no-check": MockAgentNoIsAvailable(),
        }
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/no-check/health", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "unhealthy"

    def test_agent_raises_attribute_error(self, handler):
        """Agent raising AttributeError => unknown status."""
        handler.ctx["external_agents"] = {
            "faulty": MockAgentRaisesAttributeError(),
        }
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/faulty/health", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "unknown"

    def test_agent_raises_type_error(self, handler):
        """Agent raising TypeError => unknown status."""
        handler.ctx["external_agents"] = {
            "broken": MockAgentRaisesTypeError(),
        }
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/broken/health", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "unknown"

    def test_agent_raises_value_error(self, handler):
        """Agent raising ValueError => unknown status."""
        handler.ctx["external_agents"] = {
            "bad-val": MockAgentRaisesValueError(),
        }
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/bad-val/health", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "unknown"

    def test_agent_raises_runtime_error(self, handler):
        """Agent raising RuntimeError => unavailable status."""
        handler.ctx["external_agents"] = {
            "crashed": MockAgentRaisesRuntimeError(),
        }
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/crashed/health", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "unavailable"

    def test_agent_raises_timeout_error(self, handler):
        """Agent raising TimeoutError => unavailable status."""
        handler.ctx["external_agents"] = {
            "timeout": MockAgentRaisesTimeoutError(),
        }
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/timeout/health", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "unavailable"

    def test_agent_raises_os_error(self, handler):
        """Agent raising OSError => unavailable status."""
        handler.ctx["external_agents"] = {
            "network": MockAgentRaisesOSError(),
        }
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/network/health", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "unavailable"

    def test_base_url_included(self, handler):
        """Response includes base_url from agent."""
        handler.ctx["external_agents"] = {
            "with-url": MockAgent(base_url="https://agent.example.com"),
        }
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/with-url/health", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["base_url"] == "https://agent.example.com"

    def test_base_url_none_when_not_set(self, handler):
        """Response has base_url=None when agent has no base_url."""
        agent = MagicMock(spec=["is_available", "agent_type"])
        agent.is_available.return_value = True
        agent.agent_type = "custom"
        handler.ctx["external_agents"] = {"no-url": agent}
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/no-url/health", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["base_url"] is None

    def test_framework_from_agent_type(self, handler_with_agents):
        """Framework field comes from agent.agent_type."""
        http = MockHTTPHandler()
        result = handler_with_agents.handle(
            "/api/v1/gateway/agents/agent-1/health", {}, http
        )
        body = _body(result)
        assert body["framework"] == "crewai"

    def test_framework_unknown_when_no_attr(self, handler):
        """Agent without agent_type => framework='unknown'."""
        agent = MagicMock(spec=["is_available"])
        agent.is_available.return_value = True
        handler.ctx["external_agents"] = {"bare": agent}
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/bare/health", {}, http
        )
        body = _body(result)
        assert body["framework"] == "unknown"

    def test_response_time_ms_is_number(self, handler_with_agents):
        """response_time_ms should be a float."""
        http = MockHTTPHandler()
        result = handler_with_agents.handle(
            "/api/v1/gateway/agents/agent-1/health", {}, http
        )
        body = _body(result)
        assert isinstance(body["response_time_ms"], (int, float))

    def test_ctx_external_agents_none(self, handler):
        """ctx with external_agents=None => agent not found."""
        handler.ctx["external_agents"] = None
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/anything/health", {}, http
        )
        assert _status(result) == 404


# ===========================================================================
# _check_agent_available method
# ===========================================================================


class TestCheckAgentAvailable:
    """Test the _check_agent_available helper method."""

    def test_available_returns_true(self, handler):
        agent = MockAgent(available=True)
        assert handler._check_agent_available(agent) is True

    def test_unavailable_returns_false(self, handler):
        agent = MockAgent(available=False)
        assert handler._check_agent_available(agent) is False

    def test_no_is_available_method_returns_false(self, handler):
        agent = MockAgentNoIsAvailable()
        assert handler._check_agent_available(agent) is False

    def test_truthy_return_value(self, handler):
        """Non-bool truthy return value should be coerced to True."""
        agent = MagicMock()
        agent.is_available.return_value = 1
        assert handler._check_agent_available(agent) is True

    def test_falsy_return_value(self, handler):
        """Non-bool falsy return value should be coerced to False."""
        agent = MagicMock()
        agent.is_available.return_value = 0
        assert handler._check_agent_available(agent) is False

    def test_none_return_value(self, handler):
        """None return value should be coerced to False."""
        agent = MagicMock()
        agent.is_available.return_value = None
        assert handler._check_agent_available(agent) is False

    def test_async_is_available(self, handler):
        """Async is_available should be handled via run_async."""
        agent = MockAgentAsyncAvailable()
        with patch(
            "aragora.server.http_utils.run_async",
            return_value=True,
        ) as mock_run:
            result = handler._check_agent_available(agent)
        assert result is True
        mock_run.assert_called_once()


# ===========================================================================
# Gateway unavailable (503) for all routes
# ===========================================================================


class TestGatewayUnavailable:
    """Test 503 responses when GATEWAY_AVAILABLE is False."""

    @pytest.fixture(autouse=True)
    def _disable_gateway(self):
        with patch(
            "aragora.server.handlers.gateway_health_handler.GATEWAY_AVAILABLE", False
        ):
            yield

    def test_overall_health_503(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health", {}, http)
        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "").lower()

    def test_agent_health_503(self, handler):
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/some-agent/health", {}, http
        )
        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "").lower()


# ===========================================================================
# Unknown routes return None
# ===========================================================================


class TestUnknownRoutes:
    """Test that unknown routes return None."""

    def test_handle_unknown_path(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/other/endpoint", {}, http)
        assert result is None

    def test_handle_gateway_base_path(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway", {}, http)
        assert result is None

    def test_handle_gateway_agents_without_health(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/agents/name", {}, http)
        assert result is None

    def test_handle_wrong_version(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v2/gateway/health", {}, http)
        assert result is None

    def test_handle_health_trailing_slash(self, handler):
        """Path with trailing slash does not match exact path."""
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health/", {}, http)
        assert result is None


# ===========================================================================
# Vault status
# ===========================================================================


class TestVaultStatus:
    """Test _get_vault_status helper function."""

    def test_vault_unavailable_when_module_not_found(self):
        """When credential_vault module is not available, return 'unavailable'."""
        with patch("importlib.util.find_spec", return_value=None):
            assert _get_vault_status() == "unavailable"

    def test_vault_sealed_when_module_available(self):
        """When credential_vault module is available, return 'sealed'."""
        with patch("importlib.util.find_spec", return_value=MagicMock()):
            assert _get_vault_status() == "sealed"


# ===========================================================================
# Circuit breaker helper functions
# ===========================================================================


class TestCircuitBreaker:
    """Test the module-level circuit breaker functions."""

    def test_get_circuit_breaker_returns_instance(self):
        cb = get_gateway_health_circuit_breaker()
        assert cb is not None
        assert cb.name == "gateway_health_handler"

    def test_get_circuit_breaker_status_returns_dict(self):
        status = get_gateway_health_circuit_breaker_status()
        assert isinstance(status, dict)

    def test_circuit_breaker_config(self):
        cb = get_gateway_health_circuit_breaker()
        assert cb.failure_threshold == 5
        assert cb.cooldown_seconds == 30.0

    def test_circuit_breaker_same_instance(self):
        """get_gateway_health_circuit_breaker returns same singleton."""
        cb1 = get_gateway_health_circuit_breaker()
        cb2 = get_gateway_health_circuit_breaker()
        assert cb1 is cb2


# ===========================================================================
# Module exports
# ===========================================================================


class TestModuleExports:
    """Test that __all__ exports are correct."""

    def test_all_exports(self):
        from aragora.server.handlers import gateway_health_handler

        assert "GatewayHealthHandler" in gateway_health_handler.__all__

    def test_handler_class_importable(self):
        from aragora.server.handlers.gateway_health_handler import GatewayHealthHandler
        assert GatewayHealthHandler is not None


# ===========================================================================
# Security tests
# ===========================================================================


class TestSecurity:
    """Security-focused tests for input validation."""

    def test_path_traversal_in_agent_name(self, handler):
        """Path traversal in agent name should not cause issues."""
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/../../etc/passwd/health", {}, http
        )
        # Either None (not matched due to segment count) or 404 (not found)
        if result is not None:
            assert _status(result) in (404, 503)

    def test_null_byte_in_agent_name(self, handler):
        """Null byte in agent name should not cause crashes."""
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/agent\x00evil/health", {}, http
        )
        if result is not None:
            assert _status(result) in (404, 503)

    def test_sql_injection_in_agent_name(self, handler):
        """SQL injection attempt in agent name should not cause issues."""
        handler.ctx["external_agents"] = {}
        http = MockHTTPHandler()
        # The slash in the SQL injection changes segment count
        result = handler.handle(
            "/api/v1/gateway/agents/'; DROP TABLE agents;--/health", {}, http
        )
        if result is not None:
            assert _status(result) in (404, 503)

    def test_xss_in_agent_name(self, handler):
        """XSS attempt in agent name should not cause issues."""
        handler.ctx["external_agents"] = {}
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/<script>alert(1)</script>/health", {}, http
        )
        if result is not None:
            assert _status(result) in (404, 503)

    def test_very_long_agent_name(self, handler):
        """Very long agent name should not cause resource exhaustion."""
        handler.ctx["external_agents"] = {}
        long_name = "a" * 10000
        http = MockHTTPHandler()
        result = handler.handle(
            f"/api/v1/gateway/agents/{long_name}/health", {}, http
        )
        if result is not None:
            assert _status(result) == 404

    def test_unicode_agent_name(self, handler):
        """Unicode in agent name should be handled gracefully."""
        handler.ctx["external_agents"] = {}
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/\u00e9l\u00e8ve/health", {}, http
        )
        if result is not None:
            assert _status(result) == 404


# ===========================================================================
# Path parsing edge cases
# ===========================================================================


class TestPathParsing:
    """Test path parsing edge cases in handle()."""

    def test_agent_name_extraction(self, handler):
        """Agent name is correctly extracted from path."""
        handler.ctx["external_agents"] = {
            "test-agent": MockAgent(available=True),
        }
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/test-agent/health", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["name"] == "test-agent"

    def test_agent_name_with_dashes(self, handler):
        """Agent name with dashes is correctly extracted."""
        handler.ctx["external_agents"] = {
            "my-cool-agent": MockAgent(available=True),
        }
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/my-cool-agent/health", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["name"] == "my-cool-agent"

    def test_agent_name_with_numbers(self, handler):
        """Agent name with numbers is correctly extracted."""
        handler.ctx["external_agents"] = {
            "agent42": MockAgent(available=True),
        }
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/agent42/health", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["name"] == "agent42"

    def test_path_segment_count_exactly_6(self, handler):
        """Path with exactly 6 segments after strip should match agent health."""
        # /api/v1/gateway/agents/name/health => 6 segments
        handler.ctx["external_agents"] = {
            "name": MockAgent(available=True),
        }
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/name/health", {}, http
        )
        assert _status(result) == 200

    def test_path_too_few_segments(self, handler):
        """Path with too few segments should not match agent health route."""
        http = MockHTTPHandler()
        # 5 segments: api/v1/gateway/agents/health
        result = handler.handle("/api/v1/gateway/agents/health", {}, http)
        # This matches can_handle check for /api/v1/gateway/agents/health/health
        # With count==5, it won't match the 6-segment check in handle()
        # can_handle: starts with /api/v1/gateway/agents/, ends with /health, count==5 (not 6)
        # so can_handle returns False, handle returns None
        assert result is None

    def test_case_sensitive_health_path(self, handler):
        """Path matching is case-sensitive: /Health should not match."""
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/name/Health", {}, http
        )
        assert result is None

    def test_case_sensitive_gateway_path(self, handler):
        """Path matching is case-sensitive: /Gateway should not match."""
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/Gateway/health", {}, http)
        assert result is None


# ===========================================================================
# Overall health status edge cases
# ===========================================================================


class TestOverallHealthStatusLogic:
    """Test the overall health status determination logic."""

    def test_empty_external_agents_dict(self, handler):
        """Empty dict of external agents => healthy."""
        handler.ctx["external_agents"] = {}
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "healthy"

    def test_single_healthy_agent(self, handler):
        """Single healthy agent => healthy."""
        handler.ctx["external_agents"] = {
            "solo": MockAgent(available=True),
        }
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health", {}, http)
        body = _body(result)
        assert body["status"] == "healthy"

    def test_single_unhealthy_agent(self, handler):
        """Single unhealthy agent => unhealthy."""
        handler.ctx["external_agents"] = {
            "solo": MockAgent(available=False),
        }
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health", {}, http)
        body = _body(result)
        assert body["status"] == "unhealthy"

    def test_all_agents_exception_is_unhealthy(self, handler):
        """All agents throwing exceptions => unhealthy."""
        handler.ctx["external_agents"] = {
            "err-1": MockAgentRaisesRuntimeError(),
            "err-2": MockAgentRaisesAttributeError(),
        }
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health", {}, http)
        body = _body(result)
        assert body["status"] == "unhealthy"

    def test_one_exception_one_healthy_is_degraded(self, handler):
        """Mix of exception and healthy agents => degraded."""
        handler.ctx["external_agents"] = {
            "good": MockAgent(available=True),
            "error": MockAgentRaisesTimeoutError(),
        }
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health", {}, http)
        body = _body(result)
        assert body["status"] == "degraded"


# ===========================================================================
# Agent health path extraction in handle()
# ===========================================================================


class TestHandleDispatch:
    """Test the handle method dispatch logic."""

    def test_overall_health_dispatch(self, handler):
        """handle() dispatches to _handle_overall_health for /api/v1/gateway/health."""
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert "status" in body
        assert "gateway" in body
        assert "agents" in body
        assert "timestamp" in body

    def test_agent_health_dispatch(self, handler):
        """handle() dispatches to _handle_agent_health for /api/v1/gateway/agents/{name}/health."""
        handler.ctx["external_agents"] = {
            "my-agent": MockAgent(available=True),
        }
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/my-agent/health", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["name"] == "my-agent"
        assert "status" in body
        assert "framework" in body
        assert "response_time_ms" in body

    def test_non_matching_returns_none(self, handler):
        """handle() returns None for paths that don't match any route."""
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/unrelated", {}, http)
        assert result is None


# ===========================================================================
# Error handler framework in _handle_agent_health
# ===========================================================================


class TestAgentHealthErrorFramework:
    """Test that agent_type is preserved in error responses."""

    def test_framework_preserved_on_attribute_error(self, handler):
        handler.ctx["external_agents"] = {
            "err-agent": MockAgentRaisesAttributeError(),
        }
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/err-agent/health", {}, http
        )
        body = _body(result)
        assert body["framework"] == "faulty"

    def test_framework_preserved_on_runtime_error(self, handler):
        handler.ctx["external_agents"] = {
            "runtime-agent": MockAgentRaisesRuntimeError(),
        }
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/runtime-agent/health", {}, http
        )
        body = _body(result)
        assert body["framework"] == "runtime_fail"

    def test_framework_preserved_on_timeout_error(self, handler):
        handler.ctx["external_agents"] = {
            "to-agent": MockAgentRaisesTimeoutError(),
        }
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/to-agent/health", {}, http
        )
        body = _body(result)
        assert body["framework"] == "timeout_agent"

    def test_framework_preserved_on_os_error(self, handler):
        handler.ctx["external_agents"] = {
            "os-agent": MockAgentRaisesOSError(),
        }
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/os-agent/health", {}, http
        )
        body = _body(result)
        assert body["framework"] == "network_fail"


# ===========================================================================
# Response structure validation
# ===========================================================================


class TestResponseStructure:
    """Test response body structure for completeness."""

    def test_overall_health_response_keys(self, handler):
        """Overall health response contains all required keys."""
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health", {}, http)
        body = _body(result)
        assert set(body.keys()) == {"status", "gateway", "agents", "timestamp"}

    def test_overall_health_gateway_keys(self, handler):
        """Gateway sub-object has all required keys."""
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health", {}, http)
        body = _body(result)
        gw = body["gateway"]
        assert "external_agents_available" in gw
        assert "credential_vault_status" in gw
        assert "active_executions" in gw

    def test_agent_health_info_keys_healthy(self, handler_with_agents):
        """Per-agent health info contains all required keys for healthy agent."""
        http = MockHTTPHandler()
        result = handler_with_agents.handle("/api/v1/gateway/health", {}, http)
        body = _body(result)
        info = body["agents"]["agent-1"]
        assert set(info.keys()) == {"status", "framework", "last_check", "latency_ms"}

    def test_agent_health_info_keys_error(self, handler):
        """Per-agent health info contains error field on exception."""
        handler.ctx["external_agents"] = {
            "err": MockAgentRaisesAttributeError(),
        }
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/health", {}, http)
        body = _body(result)
        info = body["agents"]["err"]
        assert "error" in info
        assert "status" in info
        assert "framework" in info
        assert "last_check" in info

    def test_individual_health_response_keys(self, handler_with_agents):
        """Individual agent health response contains all required keys."""
        http = MockHTTPHandler()
        result = handler_with_agents.handle(
            "/api/v1/gateway/agents/agent-1/health", {}, http
        )
        body = _body(result)
        assert set(body.keys()) == {
            "name", "status", "framework", "base_url",
            "last_check", "response_time_ms",
        }


# ===========================================================================
# Multiple calls / idempotency
# ===========================================================================


class TestIdempotency:
    """Test that repeated calls produce consistent results."""

    def test_overall_health_idempotent(self, handler_with_agents):
        """Two consecutive calls should return same status."""
        http1 = MockHTTPHandler()
        result1 = handler_with_agents.handle("/api/v1/gateway/health", {}, http1)
        http2 = MockHTTPHandler()
        result2 = handler_with_agents.handle("/api/v1/gateway/health", {}, http2)
        assert _status(result1) == _status(result2)
        assert _body(result1)["status"] == _body(result2)["status"]

    def test_agent_health_idempotent(self, handler_with_agents):
        """Two consecutive agent health calls should return same status."""
        http1 = MockHTTPHandler()
        result1 = handler_with_agents.handle(
            "/api/v1/gateway/agents/agent-1/health", {}, http1
        )
        http2 = MockHTTPHandler()
        result2 = handler_with_agents.handle(
            "/api/v1/gateway/agents/agent-1/health", {}, http2
        )
        assert _body(result1)["status"] == _body(result2)["status"]
        assert _body(result1)["name"] == _body(result2)["name"]

    def test_overall_then_individual(self, handler_with_agents):
        """Overall health and individual agent health should be consistent."""
        http1 = MockHTTPHandler()
        overall = handler_with_agents.handle("/api/v1/gateway/health", {}, http1)
        http2 = MockHTTPHandler()
        individual = handler_with_agents.handle(
            "/api/v1/gateway/agents/agent-1/health", {}, http2
        )
        overall_agent = _body(overall)["agents"]["agent-1"]
        individual_body = _body(individual)
        assert overall_agent["status"] == individual_body["status"]
        assert overall_agent["framework"] == individual_body["framework"]


# ===========================================================================
# can_handle edge cases
# ===========================================================================


class TestCanHandleEdgeCases:
    """Additional edge cases for can_handle."""

    def test_health_path_with_query_like_segment(self, handler):
        """Path with query-like characters should not confuse routing."""
        assert handler.can_handle("/api/v1/gateway/health?format=json") is False

    def test_agents_health_with_dots_in_name(self, handler):
        """Agent name with dots still has correct segment count."""
        assert handler.can_handle("/api/v1/gateway/agents/agent.v2/health") is True

    def test_agents_health_with_underscores_and_numbers(self, handler):
        assert handler.can_handle("/api/v1/gateway/agents/agent_v2_123/health") is True

    def test_slash_count_boundary_5(self, handler):
        """Path with 5 slashes should not match agent health (needs 6)."""
        assert handler.can_handle("/api/v1/gateway/agents/health") is False

    def test_slash_count_boundary_7(self, handler):
        """Path with 7 slashes should not match agent health."""
        assert handler.can_handle("/api/v1/gateway/agents/a/b/health") is False


# ===========================================================================
# Agent health response_time_ms on error
# ===========================================================================


class TestAgentHealthResponseTime:
    """Test that response_time_ms is included even when agent check fails."""

    def test_response_time_on_attribute_error(self, handler):
        handler.ctx["external_agents"] = {
            "err": MockAgentRaisesAttributeError(),
        }
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/err/health", {}, http
        )
        body = _body(result)
        assert "response_time_ms" in body
        assert isinstance(body["response_time_ms"], (int, float))
        assert body["response_time_ms"] >= 0

    def test_response_time_on_runtime_error(self, handler):
        handler.ctx["external_agents"] = {
            "err": MockAgentRaisesRuntimeError(),
        }
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/err/health", {}, http
        )
        body = _body(result)
        assert "response_time_ms" in body
        assert body["response_time_ms"] >= 0

    def test_response_time_on_timeout_error(self, handler):
        handler.ctx["external_agents"] = {
            "err": MockAgentRaisesTimeoutError(),
        }
        http = MockHTTPHandler()
        result = handler.handle(
            "/api/v1/gateway/agents/err/health", {}, http
        )
        body = _body(result)
        assert "response_time_ms" in body
        assert body["response_time_ms"] >= 0
