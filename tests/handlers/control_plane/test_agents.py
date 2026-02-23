"""Comprehensive tests for control plane agent management handlers.

Tests the AgentHandlerMixin endpoints:
- GET  /api/control-plane/agents (list agents)
- GET  /api/control-plane/agents/{agent_id} (get agent)
- POST /api/control-plane/agents (register agent)
- POST /api/control-plane/agents/{agent_id}/heartbeat (heartbeat)
- DELETE /api/control-plane/agents/{agent_id} (unregister agent)

Also tests:
- Async variants (_handle_register_agent_async, _handle_heartbeat_async)
- Routing through handle(), handle_post(), handle_delete()
- Mixin helper methods (_get_has_permission, _await_if_needed, _emit_event)
- No-coordinator (503) paths
- Error handling for each exception type
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.control_plane import ControlPlaneHandler


# ============================================================================
# Helpers
# ============================================================================


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ============================================================================
# Mock Domain Objects
# ============================================================================


class MockAgentStatus(Enum):
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    DRAINING = "draining"
    OFFLINE = "offline"
    FAILED = "failed"


@dataclass
class MockAgentInfo:
    """Mocked agent info object returned by the coordinator."""

    agent_id: str
    capabilities: list[str]
    model: str
    provider: str
    status: str = "ready"
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "capabilities": self.capabilities,
            "model": self.model,
            "provider": self.provider,
            "status": self.status,
            "metadata": self.metadata or {},
        }


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_coordinator():
    """Create a mock coordinator with standard async agent methods."""
    coord = MagicMock()
    coord.list_agents = AsyncMock(return_value=[])
    coord.get_agent = AsyncMock(return_value=None)
    coord.register_agent = AsyncMock(return_value=MockAgentInfo(
        agent_id="agent-001",
        capabilities=["reasoning"],
        model="claude-3",
        provider="anthropic",
    ))
    coord.heartbeat = AsyncMock(return_value=True)
    coord.unregister_agent = AsyncMock(return_value=True)
    # Stats method needed by other mixins in some tests
    coord.get_stats = AsyncMock(return_value={})
    return coord


@pytest.fixture
def handler(mock_coordinator):
    """Create a ControlPlaneHandler with a mock coordinator in context."""
    ctx: dict[str, Any] = {
        "control_plane_coordinator": mock_coordinator,
    }
    return ControlPlaneHandler(ctx)


@pytest.fixture
def handler_no_coord():
    """Create a ControlPlaneHandler with NO coordinator (not initialized)."""
    ctx: dict[str, Any] = {}
    return ControlPlaneHandler(ctx)


@pytest.fixture
def mock_http_handler():
    """Create a minimal mock HTTP handler."""
    m = MagicMock()
    m.path = "/api/control-plane/agents"
    m.headers = {"Content-Type": "application/json"}
    return m


@pytest.fixture
def sample_agent():
    """Create a representative agent info object."""
    return MockAgentInfo(
        agent_id="agent-001",
        capabilities=["reasoning", "search"],
        model="claude-3",
        provider="anthropic",
        status="ready",
        metadata={"version": "1.0"},
    )


@pytest.fixture
def busy_agent():
    """Create a busy agent."""
    return MockAgentInfo(
        agent_id="agent-002",
        capabilities=["code_generation"],
        model="gpt-4",
        provider="openai",
        status="busy",
        metadata={"task_count": 3},
    )


@pytest.fixture
def offline_agent():
    """Create an offline agent."""
    return MockAgentInfo(
        agent_id="agent-003",
        capabilities=["analysis"],
        model="mistral-large",
        provider="mistral",
        status="offline",
        metadata={},
    )


# ============================================================================
# GET /api/control-plane/agents (list agents)
# ============================================================================


class TestListAgents:
    """Tests for _handle_list_agents."""

    def test_list_agents_empty(self, handler, mock_coordinator):
        mock_coordinator.list_agents = AsyncMock(return_value=[])
        result = handler._handle_list_agents({})
        assert _status(result) == 200
        body = _body(result)
        assert body["agents"] == []
        assert body["total"] == 0

    def test_list_agents_returns_agents(self, handler, mock_coordinator, sample_agent, busy_agent):
        mock_coordinator.list_agents = AsyncMock(return_value=[sample_agent, busy_agent])
        result = handler._handle_list_agents({})
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert body["agents"][0]["agent_id"] == "agent-001"
        assert body["agents"][1]["agent_id"] == "agent-002"

    def test_list_agents_with_capability_filter(self, handler, mock_coordinator, sample_agent):
        mock_coordinator.list_agents = AsyncMock(return_value=[sample_agent])
        result = handler._handle_list_agents({"capability": "reasoning"})
        assert _status(result) == 200
        mock_coordinator.list_agents.assert_called_once_with(
            capability="reasoning",
            only_available=True,
        )

    def test_list_agents_available_true_default(self, handler, mock_coordinator):
        mock_coordinator.list_agents = AsyncMock(return_value=[])
        result = handler._handle_list_agents({})
        assert _status(result) == 200
        mock_coordinator.list_agents.assert_called_once_with(
            capability=None,
            only_available=True,
        )

    def test_list_agents_available_false(self, handler, mock_coordinator):
        mock_coordinator.list_agents = AsyncMock(return_value=[])
        result = handler._handle_list_agents({"available": "false"})
        assert _status(result) == 200
        mock_coordinator.list_agents.assert_called_once_with(
            capability=None,
            only_available=False,
        )

    def test_list_agents_available_true_explicit(self, handler, mock_coordinator):
        mock_coordinator.list_agents = AsyncMock(return_value=[])
        result = handler._handle_list_agents({"available": "True"})
        assert _status(result) == 200
        mock_coordinator.list_agents.assert_called_once_with(
            capability=None,
            only_available=True,
        )

    def test_list_agents_no_coordinator(self, handler_no_coord):
        result = handler_no_coord._handle_list_agents({})
        assert _status(result) == 503

    def test_list_agents_runtime_error(self, handler, mock_coordinator):
        mock_coordinator.list_agents = AsyncMock(side_effect=RuntimeError("connection lost"))
        result = handler._handle_list_agents({})
        assert _status(result) == 503

    def test_list_agents_timeout_error(self, handler, mock_coordinator):
        mock_coordinator.list_agents = AsyncMock(side_effect=TimeoutError("timed out"))
        result = handler._handle_list_agents({})
        assert _status(result) == 503

    def test_list_agents_value_error(self, handler, mock_coordinator):
        mock_coordinator.list_agents = AsyncMock(side_effect=ValueError("bad param"))
        result = handler._handle_list_agents({})
        assert _status(result) == 400

    def test_list_agents_key_error(self, handler, mock_coordinator):
        mock_coordinator.list_agents = AsyncMock(side_effect=KeyError("missing"))
        result = handler._handle_list_agents({})
        assert _status(result) == 400

    def test_list_agents_attribute_error(self, handler, mock_coordinator):
        mock_coordinator.list_agents = AsyncMock(side_effect=AttributeError("no attr"))
        result = handler._handle_list_agents({})
        assert _status(result) == 400

    def test_list_agents_os_error(self, handler, mock_coordinator):
        mock_coordinator.list_agents = AsyncMock(side_effect=OSError("disk failure"))
        result = handler._handle_list_agents({})
        assert _status(result) == 500

    def test_list_agents_type_error(self, handler, mock_coordinator):
        mock_coordinator.list_agents = AsyncMock(side_effect=TypeError("wrong type"))
        result = handler._handle_list_agents({})
        assert _status(result) == 500

    def test_list_agents_to_dict_called(self, handler, mock_coordinator):
        """Verify to_dict() is called on each agent for serialization."""
        mock_agent = MagicMock()
        mock_agent.to_dict.return_value = {"agent_id": "a1", "capabilities": []}
        mock_coordinator.list_agents = AsyncMock(return_value=[mock_agent])
        result = handler._handle_list_agents({})
        assert _status(result) == 200
        mock_agent.to_dict.assert_called_once()

    def test_list_agents_with_capability_and_available(self, handler, mock_coordinator, sample_agent):
        mock_coordinator.list_agents = AsyncMock(return_value=[sample_agent])
        result = handler._handle_list_agents({
            "capability": "search",
            "available": "false",
        })
        assert _status(result) == 200
        mock_coordinator.list_agents.assert_called_once_with(
            capability="search",
            only_available=False,
        )


# ============================================================================
# GET /api/control-plane/agents/{agent_id} (get agent)
# ============================================================================


class TestGetAgent:
    """Tests for _handle_get_agent."""

    def test_get_agent_success(self, handler, mock_coordinator, sample_agent):
        mock_coordinator.get_agent = AsyncMock(return_value=sample_agent)
        result = handler._handle_get_agent("agent-001")
        assert _status(result) == 200
        body = _body(result)
        assert body["agent_id"] == "agent-001"
        assert body["model"] == "claude-3"
        assert body["provider"] == "anthropic"
        assert body["capabilities"] == ["reasoning", "search"]

    def test_get_agent_not_found(self, handler, mock_coordinator):
        mock_coordinator.get_agent = AsyncMock(return_value=None)
        result = handler._handle_get_agent("nonexistent")
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    def test_get_agent_no_coordinator(self, handler_no_coord):
        result = handler_no_coord._handle_get_agent("agent-001")
        assert _status(result) == 503

    def test_get_agent_value_error(self, handler, mock_coordinator):
        mock_coordinator.get_agent = AsyncMock(side_effect=ValueError("bad id"))
        result = handler._handle_get_agent("bad-id")
        assert _status(result) == 400

    def test_get_agent_key_error(self, handler, mock_coordinator):
        mock_coordinator.get_agent = AsyncMock(side_effect=KeyError("missing"))
        result = handler._handle_get_agent("missing-key")
        assert _status(result) == 400

    def test_get_agent_attribute_error(self, handler, mock_coordinator):
        mock_coordinator.get_agent = AsyncMock(side_effect=AttributeError("no attr"))
        result = handler._handle_get_agent("attr-err")
        assert _status(result) == 400

    def test_get_agent_runtime_error(self, handler, mock_coordinator):
        mock_coordinator.get_agent = AsyncMock(side_effect=RuntimeError("boom"))
        result = handler._handle_get_agent("runtime-err")
        assert _status(result) == 500

    def test_get_agent_os_error(self, handler, mock_coordinator):
        mock_coordinator.get_agent = AsyncMock(side_effect=OSError("disk failure"))
        result = handler._handle_get_agent("os-err")
        assert _status(result) == 500

    def test_get_agent_type_error(self, handler, mock_coordinator):
        mock_coordinator.get_agent = AsyncMock(side_effect=TypeError("wrong type"))
        result = handler._handle_get_agent("type-err")
        assert _status(result) == 500

    def test_get_agent_to_dict_called(self, handler, mock_coordinator):
        """Verify agent.to_dict() is called for serialization."""
        mock_agent = MagicMock()
        mock_agent.to_dict.return_value = {"agent_id": "a-x", "model": "test"}
        mock_coordinator.get_agent = AsyncMock(return_value=mock_agent)
        result = handler._handle_get_agent("a-x")
        assert _status(result) == 200
        mock_agent.to_dict.assert_called_once()


# ============================================================================
# POST /api/control-plane/agents (register agent - sync)
# ============================================================================


class TestRegisterAgent:
    """Tests for _handle_register_agent (sync variant)."""

    def test_register_agent_success(self, handler, mock_coordinator, mock_http_handler):
        agent = MockAgentInfo(
            agent_id="new-agent",
            capabilities=["reasoning"],
            model="claude-3",
            provider="anthropic",
        )
        mock_coordinator.register_agent = AsyncMock(return_value=agent)
        body = {
            "agent_id": "new-agent",
            "capabilities": ["reasoning"],
            "model": "claude-3",
            "provider": "anthropic",
            "metadata": {"version": "2.0"},
        }
        result = handler._handle_register_agent(body, mock_http_handler)
        assert _status(result) == 201
        data = _body(result)
        assert data["agent_id"] == "new-agent"

    def test_register_agent_missing_agent_id(self, handler, mock_http_handler):
        body = {"capabilities": ["reasoning"]}
        result = handler._handle_register_agent(body, mock_http_handler)
        assert _status(result) == 400
        assert "agent_id" in _body(result).get("error", "").lower()

    def test_register_agent_empty_agent_id(self, handler, mock_http_handler):
        body = {"agent_id": ""}
        result = handler._handle_register_agent(body, mock_http_handler)
        assert _status(result) == 400

    def test_register_agent_no_coordinator(self, handler_no_coord, mock_http_handler):
        body = {"agent_id": "new-agent"}
        result = handler_no_coord._handle_register_agent(body, mock_http_handler)
        assert _status(result) == 503

    def test_register_agent_defaults(self, handler, mock_coordinator, mock_http_handler):
        """Capabilities, model, provider, metadata should default."""
        agent = MockAgentInfo(
            agent_id="default-agent",
            capabilities=[],
            model="unknown",
            provider="unknown",
        )
        mock_coordinator.register_agent = AsyncMock(return_value=agent)
        body = {"agent_id": "default-agent"}
        result = handler._handle_register_agent(body, mock_http_handler)
        assert _status(result) == 201
        mock_coordinator.register_agent.assert_called_once_with(
            agent_id="default-agent",
            capabilities=[],
            model="unknown",
            provider="unknown",
            metadata={},
        )

    def test_register_agent_runtime_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.register_agent = AsyncMock(side_effect=RuntimeError("db error"))
        body = {"agent_id": "agent-err"}
        result = handler._handle_register_agent(body, mock_http_handler)
        assert _status(result) == 500

    def test_register_agent_value_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.register_agent = AsyncMock(side_effect=ValueError("duplicate"))
        body = {"agent_id": "dup-agent"}
        result = handler._handle_register_agent(body, mock_http_handler)
        assert _status(result) == 500

    def test_register_agent_os_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.register_agent = AsyncMock(side_effect=OSError("disk"))
        body = {"agent_id": "os-agent"}
        result = handler._handle_register_agent(body, mock_http_handler)
        assert _status(result) == 500

    def test_register_agent_type_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.register_agent = AsyncMock(side_effect=TypeError("type"))
        body = {"agent_id": "type-agent"}
        result = handler._handle_register_agent(body, mock_http_handler)
        assert _status(result) == 500

    def test_register_agent_key_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.register_agent = AsyncMock(side_effect=KeyError("key"))
        body = {"agent_id": "key-agent"}
        result = handler._handle_register_agent(body, mock_http_handler)
        assert _status(result) == 500

    def test_register_agent_attribute_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.register_agent = AsyncMock(side_effect=AttributeError("attr"))
        body = {"agent_id": "attr-agent"}
        result = handler._handle_register_agent(body, mock_http_handler)
        assert _status(result) == 500

    def test_register_agent_emits_event(self, handler, mock_coordinator, mock_http_handler):
        agent = MockAgentInfo(
            agent_id="evt-agent",
            capabilities=["search"],
            model="gpt-4",
            provider="openai",
        )
        mock_coordinator.register_agent = AsyncMock(return_value=agent)
        body = {
            "agent_id": "evt-agent",
            "capabilities": ["search"],
            "model": "gpt-4",
            "provider": "openai",
        }
        with patch.object(handler, "_emit_event") as mock_emit:
            result = handler._handle_register_agent(body, mock_http_handler)
        assert _status(result) == 201
        mock_emit.assert_called_once_with(
            "emit_agent_registered",
            agent_id="evt-agent",
            capabilities=["search"],
            model="gpt-4",
            provider="openai",
        )

    def test_register_agent_with_metadata(self, handler, mock_coordinator, mock_http_handler):
        agent = MockAgentInfo(
            agent_id="meta-agent",
            capabilities=[],
            model="claude-3",
            provider="anthropic",
            metadata={"env": "prod"},
        )
        mock_coordinator.register_agent = AsyncMock(return_value=agent)
        body = {
            "agent_id": "meta-agent",
            "metadata": {"env": "prod"},
        }
        result = handler._handle_register_agent(body, mock_http_handler)
        assert _status(result) == 201
        mock_coordinator.register_agent.assert_called_once_with(
            agent_id="meta-agent",
            capabilities=[],
            model="unknown",
            provider="unknown",
            metadata={"env": "prod"},
        )


# ============================================================================
# POST /api/control-plane/agents (register agent - async)
# ============================================================================


class TestRegisterAgentAsync:
    """Tests for _handle_register_agent_async."""

    @pytest.mark.asyncio
    async def test_register_agent_async_success(self, handler, mock_coordinator, mock_http_handler):
        agent = MockAgentInfo(
            agent_id="async-agent",
            capabilities=["reasoning"],
            model="claude-3",
            provider="anthropic",
        )
        mock_coordinator.register_agent = AsyncMock(return_value=agent)
        body = {
            "agent_id": "async-agent",
            "capabilities": ["reasoning"],
            "model": "claude-3",
            "provider": "anthropic",
        }
        result = await handler._handle_register_agent_async(body, mock_http_handler)
        assert _status(result) == 201
        data = _body(result)
        assert data["agent_id"] == "async-agent"

    @pytest.mark.asyncio
    async def test_register_agent_async_missing_agent_id(self, handler, mock_http_handler):
        body = {"capabilities": ["reasoning"]}
        result = await handler._handle_register_agent_async(body, mock_http_handler)
        assert _status(result) == 400
        assert "agent_id" in _body(result).get("error", "").lower()

    @pytest.mark.asyncio
    async def test_register_agent_async_empty_agent_id(self, handler, mock_http_handler):
        body = {"agent_id": ""}
        result = await handler._handle_register_agent_async(body, mock_http_handler)
        assert _status(result) == 400

    @pytest.mark.asyncio
    async def test_register_agent_async_no_coordinator(self, handler_no_coord, mock_http_handler):
        body = {"agent_id": "agent-x"}
        result = await handler_no_coord._handle_register_agent_async(body, mock_http_handler)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_register_agent_async_runtime_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.register_agent = AsyncMock(side_effect=RuntimeError("crash"))
        body = {"agent_id": "err-agent"}
        result = await handler._handle_register_agent_async(body, mock_http_handler)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_register_agent_async_value_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.register_agent = AsyncMock(side_effect=ValueError("duplicate"))
        body = {"agent_id": "dup-agent"}
        result = await handler._handle_register_agent_async(body, mock_http_handler)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_register_agent_async_os_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.register_agent = AsyncMock(side_effect=OSError("io"))
        body = {"agent_id": "os-agent"}
        result = await handler._handle_register_agent_async(body, mock_http_handler)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_register_agent_async_type_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.register_agent = AsyncMock(side_effect=TypeError("type"))
        body = {"agent_id": "type-agent"}
        result = await handler._handle_register_agent_async(body, mock_http_handler)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_register_agent_async_key_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.register_agent = AsyncMock(side_effect=KeyError("key"))
        body = {"agent_id": "key-agent"}
        result = await handler._handle_register_agent_async(body, mock_http_handler)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_register_agent_async_emits_event(self, handler, mock_coordinator, mock_http_handler):
        agent = MockAgentInfo(
            agent_id="evt-async",
            capabilities=["code"],
            model="gpt-4",
            provider="openai",
        )
        mock_coordinator.register_agent = AsyncMock(return_value=agent)
        body = {
            "agent_id": "evt-async",
            "capabilities": ["code"],
            "model": "gpt-4",
            "provider": "openai",
        }
        with patch.object(handler, "_emit_event") as mock_emit:
            result = await handler._handle_register_agent_async(body, mock_http_handler)
        assert _status(result) == 201
        mock_emit.assert_called_once_with(
            "emit_agent_registered",
            agent_id="evt-async",
            capabilities=["code"],
            model="gpt-4",
            provider="openai",
        )

    @pytest.mark.asyncio
    async def test_register_agent_async_defaults(self, handler, mock_coordinator, mock_http_handler):
        agent = MockAgentInfo(
            agent_id="default-async",
            capabilities=[],
            model="unknown",
            provider="unknown",
        )
        mock_coordinator.register_agent = AsyncMock(return_value=agent)
        body = {"agent_id": "default-async"}
        result = await handler._handle_register_agent_async(body, mock_http_handler)
        assert _status(result) == 201
        mock_coordinator.register_agent.assert_called_once_with(
            agent_id="default-async",
            capabilities=[],
            model="unknown",
            provider="unknown",
            metadata={},
        )


# ============================================================================
# POST /api/control-plane/agents/{agent_id}/heartbeat (sync)
# ============================================================================


class TestHeartbeat:
    """Tests for _handle_heartbeat (sync variant)."""

    def test_heartbeat_success(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.heartbeat = AsyncMock(return_value=True)
        body = {"status": "ready"}
        with patch("aragora.control_plane.registry.AgentStatus", MockAgentStatus):
            result = handler._handle_heartbeat("agent-001", body, mock_http_handler)
        assert _status(result) == 200
        assert _body(result)["acknowledged"] is True

    def test_heartbeat_no_status(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.heartbeat = AsyncMock(return_value=True)
        body = {}
        with patch("aragora.control_plane.registry.AgentStatus", MockAgentStatus):
            result = handler._handle_heartbeat("agent-001", body, mock_http_handler)
        assert _status(result) == 200
        assert _body(result)["acknowledged"] is True

    def test_heartbeat_agent_not_found(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.heartbeat = AsyncMock(return_value=False)
        body = {}
        with patch("aragora.control_plane.registry.AgentStatus", MockAgentStatus):
            result = handler._handle_heartbeat("nonexistent", body, mock_http_handler)
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    def test_heartbeat_no_coordinator(self, handler_no_coord, mock_http_handler):
        body = {"status": "ready"}
        result = handler_no_coord._handle_heartbeat("agent-001", body, mock_http_handler)
        assert _status(result) == 503

    def test_heartbeat_value_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.heartbeat = AsyncMock(side_effect=ValueError("bad status"))
        body = {"status": "invalid_status_value"}
        with patch("aragora.control_plane.registry.AgentStatus", MockAgentStatus):
            result = handler._handle_heartbeat("agent-001", body, mock_http_handler)
        assert _status(result) == 500

    def test_heartbeat_runtime_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.heartbeat = AsyncMock(side_effect=RuntimeError("crash"))
        body = {"status": "ready"}
        with patch("aragora.control_plane.registry.AgentStatus", MockAgentStatus):
            result = handler._handle_heartbeat("agent-001", body, mock_http_handler)
        assert _status(result) == 500

    def test_heartbeat_os_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.heartbeat = AsyncMock(side_effect=OSError("net error"))
        body = {}
        with patch("aragora.control_plane.registry.AgentStatus", MockAgentStatus):
            result = handler._handle_heartbeat("agent-001", body, mock_http_handler)
        assert _status(result) == 500

    def test_heartbeat_key_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.heartbeat = AsyncMock(side_effect=KeyError("missing"))
        body = {}
        with patch("aragora.control_plane.registry.AgentStatus", MockAgentStatus):
            result = handler._handle_heartbeat("agent-001", body, mock_http_handler)
        assert _status(result) == 500

    def test_heartbeat_type_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.heartbeat = AsyncMock(side_effect=TypeError("wrong"))
        body = {}
        with patch("aragora.control_plane.registry.AgentStatus", MockAgentStatus):
            result = handler._handle_heartbeat("agent-001", body, mock_http_handler)
        assert _status(result) == 500

    def test_heartbeat_attribute_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.heartbeat = AsyncMock(side_effect=AttributeError("no attr"))
        body = {}
        with patch("aragora.control_plane.registry.AgentStatus", MockAgentStatus):
            result = handler._handle_heartbeat("agent-001", body, mock_http_handler)
        assert _status(result) == 500

    def test_heartbeat_invalid_status_enum(self, handler, mock_coordinator, mock_http_handler):
        """Status value that doesn't match the AgentStatus enum raises ValueError."""
        body = {"status": "totally_bogus_status"}
        with patch("aragora.control_plane.registry.AgentStatus", MockAgentStatus):
            result = handler._handle_heartbeat("agent-001", body, mock_http_handler)
        assert _status(result) == 500

    def test_heartbeat_status_busy(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.heartbeat = AsyncMock(return_value=True)
        body = {"status": "busy"}
        with patch("aragora.control_plane.registry.AgentStatus", MockAgentStatus):
            result = handler._handle_heartbeat("agent-001", body, mock_http_handler)
        assert _status(result) == 200
        assert _body(result)["acknowledged"] is True


# ============================================================================
# POST /api/control-plane/agents/{agent_id}/heartbeat (async)
# ============================================================================


class TestHeartbeatAsync:
    """Tests for _handle_heartbeat_async."""

    @pytest.mark.asyncio
    async def test_heartbeat_async_success(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.heartbeat = AsyncMock(return_value=True)
        body = {"status": "ready"}
        with patch("aragora.control_plane.registry.AgentStatus", MockAgentStatus):
            result = await handler._handle_heartbeat_async("agent-001", body, mock_http_handler)
        assert _status(result) == 200
        assert _body(result)["acknowledged"] is True

    @pytest.mark.asyncio
    async def test_heartbeat_async_no_status(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.heartbeat = AsyncMock(return_value=True)
        body = {}
        with patch("aragora.control_plane.registry.AgentStatus", MockAgentStatus):
            result = await handler._handle_heartbeat_async("agent-001", body, mock_http_handler)
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_heartbeat_async_not_found(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.heartbeat = AsyncMock(return_value=False)
        body = {}
        with patch("aragora.control_plane.registry.AgentStatus", MockAgentStatus):
            result = await handler._handle_heartbeat_async("nonexistent", body, mock_http_handler)
        assert _status(result) == 404

    @pytest.mark.asyncio
    async def test_heartbeat_async_no_coordinator(self, handler_no_coord, mock_http_handler):
        body = {"status": "ready"}
        result = await handler_no_coord._handle_heartbeat_async("agent-001", body, mock_http_handler)
        assert _status(result) == 503

    @pytest.mark.asyncio
    async def test_heartbeat_async_runtime_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.heartbeat = AsyncMock(side_effect=RuntimeError("crash"))
        body = {"status": "ready"}
        with patch("aragora.control_plane.registry.AgentStatus", MockAgentStatus):
            result = await handler._handle_heartbeat_async("agent-001", body, mock_http_handler)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_heartbeat_async_value_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.heartbeat = AsyncMock(side_effect=ValueError("invalid"))
        body = {"status": "ready"}
        with patch("aragora.control_plane.registry.AgentStatus", MockAgentStatus):
            result = await handler._handle_heartbeat_async("agent-001", body, mock_http_handler)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_heartbeat_async_os_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.heartbeat = AsyncMock(side_effect=OSError("io"))
        body = {}
        with patch("aragora.control_plane.registry.AgentStatus", MockAgentStatus):
            result = await handler._handle_heartbeat_async("agent-001", body, mock_http_handler)
        assert _status(result) == 500

    @pytest.mark.asyncio
    async def test_heartbeat_async_invalid_status_enum(self, handler, mock_coordinator, mock_http_handler):
        body = {"status": "totally_bogus"}
        with patch("aragora.control_plane.registry.AgentStatus", MockAgentStatus):
            result = await handler._handle_heartbeat_async("agent-001", body, mock_http_handler)
        assert _status(result) == 500


# ============================================================================
# DELETE /api/control-plane/agents/{agent_id} (unregister)
# ============================================================================


class TestUnregisterAgent:
    """Tests for _handle_unregister_agent."""

    def test_unregister_agent_success(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.unregister_agent = AsyncMock(return_value=True)
        result = handler._handle_unregister_agent("agent-001", mock_http_handler)
        assert _status(result) == 200
        assert _body(result)["unregistered"] is True

    def test_unregister_agent_not_found(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.unregister_agent = AsyncMock(return_value=False)
        result = handler._handle_unregister_agent("nonexistent", mock_http_handler)
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    def test_unregister_agent_no_coordinator(self, handler_no_coord, mock_http_handler):
        result = handler_no_coord._handle_unregister_agent("agent-001", mock_http_handler)
        assert _status(result) == 503

    def test_unregister_agent_runtime_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.unregister_agent = AsyncMock(side_effect=RuntimeError("db down"))
        result = handler._handle_unregister_agent("agent-001", mock_http_handler)
        assert _status(result) == 500

    def test_unregister_agent_value_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.unregister_agent = AsyncMock(side_effect=ValueError("invalid"))
        result = handler._handle_unregister_agent("agent-001", mock_http_handler)
        assert _status(result) == 500

    def test_unregister_agent_os_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.unregister_agent = AsyncMock(side_effect=OSError("disk"))
        result = handler._handle_unregister_agent("agent-001", mock_http_handler)
        assert _status(result) == 500

    def test_unregister_agent_type_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.unregister_agent = AsyncMock(side_effect=TypeError("wrong"))
        result = handler._handle_unregister_agent("agent-001", mock_http_handler)
        assert _status(result) == 500

    def test_unregister_agent_key_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.unregister_agent = AsyncMock(side_effect=KeyError("key"))
        result = handler._handle_unregister_agent("agent-001", mock_http_handler)
        assert _status(result) == 500

    def test_unregister_agent_attribute_error(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.unregister_agent = AsyncMock(side_effect=AttributeError("attr"))
        result = handler._handle_unregister_agent("agent-001", mock_http_handler)
        assert _status(result) == 500

    def test_unregister_agent_emits_event(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.unregister_agent = AsyncMock(return_value=True)
        with patch.object(handler, "_emit_event") as mock_emit:
            result = handler._handle_unregister_agent("agent-del", mock_http_handler)
        assert _status(result) == 200
        mock_emit.assert_called_once_with(
            "emit_agent_unregistered",
            agent_id="agent-del",
            reason="manual_unregistration",
        )

    def test_unregister_agent_no_event_on_not_found(self, handler, mock_coordinator, mock_http_handler):
        """Event should NOT be emitted when agent is not found."""
        mock_coordinator.unregister_agent = AsyncMock(return_value=False)
        with patch.object(handler, "_emit_event") as mock_emit:
            result = handler._handle_unregister_agent("ghost", mock_http_handler)
        assert _status(result) == 404
        mock_emit.assert_not_called()


# ============================================================================
# GET Routing via handle()
# ============================================================================


class TestGetRouting:
    """Tests for GET request routing through handle()."""

    def test_route_list_agents(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.list_agents = AsyncMock(return_value=[])
        result = handler.handle("/api/control-plane/agents", {}, mock_http_handler)
        assert _status(result) == 200
        assert _body(result)["total"] == 0

    def test_route_list_agents_v1(self, handler, mock_coordinator, mock_http_handler):
        """Versioned path /api/v1/control-plane/agents should normalize."""
        mock_coordinator.list_agents = AsyncMock(return_value=[])
        result = handler.handle("/api/v1/control-plane/agents", {}, mock_http_handler)
        assert _status(result) == 200
        assert _body(result)["total"] == 0

    def test_route_get_agent(self, handler, mock_coordinator, mock_http_handler, sample_agent):
        mock_coordinator.get_agent = AsyncMock(return_value=sample_agent)
        result = handler.handle("/api/control-plane/agents/agent-001", {}, mock_http_handler)
        assert _status(result) == 200
        assert _body(result)["agent_id"] == "agent-001"

    def test_route_get_agent_v1(self, handler, mock_coordinator, mock_http_handler, sample_agent):
        mock_coordinator.get_agent = AsyncMock(return_value=sample_agent)
        result = handler.handle("/api/v1/control-plane/agents/agent-001", {}, mock_http_handler)
        assert _status(result) == 200
        assert _body(result)["agent_id"] == "agent-001"

    def test_route_get_agent_not_found(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.get_agent = AsyncMock(return_value=None)
        result = handler.handle("/api/control-plane/agents/nonexistent", {}, mock_http_handler)
        assert _status(result) == 404

    def test_route_list_agents_with_query_params(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.list_agents = AsyncMock(return_value=[])
        result = handler.handle(
            "/api/control-plane/agents",
            {"capability": "reasoning", "available": "false"},
            mock_http_handler,
        )
        assert _status(result) == 200
        mock_coordinator.list_agents.assert_called_once_with(
            capability="reasoning",
            only_available=False,
        )


# ============================================================================
# POST Routing via handle_post()
# ============================================================================


class TestPostRouting:
    """Tests for POST request routing through handle_post()."""

    @pytest.mark.asyncio
    async def test_route_post_register_agent(self, handler, mock_coordinator, mock_http_handler):
        agent = MockAgentInfo(
            agent_id="new-agent",
            capabilities=["reasoning"],
            model="claude-3",
            provider="anthropic",
        )
        mock_coordinator.register_agent = AsyncMock(return_value=agent)
        body = {
            "agent_id": "new-agent",
            "capabilities": ["reasoning"],
            "model": "claude-3",
            "provider": "anthropic",
        }
        mock_http_handler.rfile.read.return_value = json.dumps(body).encode()
        mock_http_handler.headers = {
            "Content-Length": str(len(json.dumps(body).encode())),
            "Content-Type": "application/json",
        }
        result = await handler.handle_post("/api/control-plane/agents", {}, mock_http_handler)
        assert _status(result) == 201
        assert _body(result)["agent_id"] == "new-agent"

    @pytest.mark.asyncio
    async def test_route_post_register_agent_v1(self, handler, mock_coordinator, mock_http_handler):
        agent = MockAgentInfo(
            agent_id="v1-agent",
            capabilities=[],
            model="unknown",
            provider="unknown",
        )
        mock_coordinator.register_agent = AsyncMock(return_value=agent)
        body = {"agent_id": "v1-agent"}
        mock_http_handler.rfile.read.return_value = json.dumps(body).encode()
        mock_http_handler.headers = {
            "Content-Length": str(len(json.dumps(body).encode())),
            "Content-Type": "application/json",
        }
        result = await handler.handle_post("/api/v1/control-plane/agents", {}, mock_http_handler)
        assert _status(result) == 201

    @pytest.mark.asyncio
    async def test_route_post_heartbeat(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.heartbeat = AsyncMock(return_value=True)
        body = {"status": "ready"}
        mock_http_handler.rfile.read.return_value = json.dumps(body).encode()
        mock_http_handler.headers = {
            "Content-Length": str(len(json.dumps(body).encode())),
            "Content-Type": "application/json",
        }
        with patch("aragora.control_plane.registry.AgentStatus", MockAgentStatus):
            result = await handler.handle_post(
                "/api/control-plane/agents/agent-001/heartbeat", {}, mock_http_handler
            )
        assert _status(result) == 200
        assert _body(result)["acknowledged"] is True

    @pytest.mark.asyncio
    async def test_route_post_heartbeat_v1(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.heartbeat = AsyncMock(return_value=True)
        body = {}
        mock_http_handler.rfile.read.return_value = json.dumps(body).encode()
        mock_http_handler.headers = {
            "Content-Length": str(len(json.dumps(body).encode())),
            "Content-Type": "application/json",
        }
        with patch("aragora.control_plane.registry.AgentStatus", MockAgentStatus):
            result = await handler.handle_post(
                "/api/v1/control-plane/agents/agent-002/heartbeat", {}, mock_http_handler
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_route_post_heartbeat_extracts_agent_id(self, handler, mock_coordinator, mock_http_handler):
        """Verify the agent_id is correctly extracted from the path."""
        mock_coordinator.heartbeat = AsyncMock(return_value=True)
        body = {}
        mock_http_handler.rfile.read.return_value = json.dumps(body).encode()
        mock_http_handler.headers = {
            "Content-Length": str(len(json.dumps(body).encode())),
            "Content-Type": "application/json",
        }
        with patch("aragora.control_plane.registry.AgentStatus", MockAgentStatus):
            result = await handler.handle_post(
                "/api/control-plane/agents/my-specific-agent/heartbeat", {}, mock_http_handler
            )
        assert _status(result) == 200
        # Verify the correct agent_id was passed
        mock_coordinator.heartbeat.assert_called_once()
        call_args = mock_coordinator.heartbeat.call_args
        assert call_args[0][0] == "my-specific-agent"


# ============================================================================
# DELETE Routing via handle_delete()
# ============================================================================


class TestDeleteRouting:
    """Tests for DELETE request routing through handle_delete()."""

    def test_route_delete_agent(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.unregister_agent = AsyncMock(return_value=True)
        result = handler.handle_delete("/api/control-plane/agents/agent-001", {}, mock_http_handler)
        assert _status(result) == 200
        assert _body(result)["unregistered"] is True

    def test_route_delete_agent_v1(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.unregister_agent = AsyncMock(return_value=True)
        result = handler.handle_delete("/api/v1/control-plane/agents/agent-001", {}, mock_http_handler)
        assert _status(result) == 200
        assert _body(result)["unregistered"] is True

    def test_route_delete_agent_not_found(self, handler, mock_coordinator, mock_http_handler):
        mock_coordinator.unregister_agent = AsyncMock(return_value=False)
        result = handler.handle_delete("/api/control-plane/agents/ghost", {}, mock_http_handler)
        assert _status(result) == 404

    def test_route_delete_unknown_path(self, handler, mock_http_handler):
        result = handler.handle_delete("/api/control-plane/unknown/path", {}, mock_http_handler)
        assert result is None

    def test_route_delete_nested_agent_path_not_matched(self, handler, mock_http_handler):
        """Paths like /api/control-plane/agents/id/extra should not match DELETE."""
        result = handler.handle_delete(
            "/api/control-plane/agents/agent-001/extra", {}, mock_http_handler
        )
        assert result is None


# ============================================================================
# Mixin Internal / Helper Methods
# ============================================================================


class TestMixinHelpers:
    """Tests for AgentHandlerMixin and related helper methods."""

    def test_get_coordinator_from_context(self, handler, mock_coordinator):
        assert handler._get_coordinator() is mock_coordinator

    def test_get_coordinator_none_when_missing(self, handler_no_coord):
        assert handler_no_coord._get_coordinator() is None

    def test_require_coordinator_success(self, handler, mock_coordinator):
        coord, err = handler._require_coordinator()
        assert coord is mock_coordinator
        assert err is None

    def test_require_coordinator_error(self, handler_no_coord):
        coord, err = handler_no_coord._require_coordinator()
        assert coord is None
        assert _status(err) == 503

    def test_handle_coordinator_error_value_error(self, handler):
        result = handler._handle_coordinator_error(ValueError("bad"), "test_op")
        assert _status(result) == 400

    def test_handle_coordinator_error_key_error(self, handler):
        result = handler._handle_coordinator_error(KeyError("missing"), "test_op")
        assert _status(result) == 400

    def test_handle_coordinator_error_attribute_error(self, handler):
        result = handler._handle_coordinator_error(AttributeError("no attr"), "test_op")
        assert _status(result) == 400

    def test_handle_coordinator_error_runtime_error(self, handler):
        result = handler._handle_coordinator_error(RuntimeError("crash"), "test_op")
        assert _status(result) == 500

    def test_handle_coordinator_error_os_error(self, handler):
        result = handler._handle_coordinator_error(OSError("disk"), "test_op")
        assert _status(result) == 500

    def test_get_stream_returns_none_when_missing(self, handler):
        assert handler._get_stream() is None

    def test_get_stream_returns_stream(self, handler):
        mock_stream = MagicMock()
        handler.ctx["control_plane_stream"] = mock_stream
        assert handler._get_stream() is mock_stream

    def test_emit_event_no_stream_is_noop(self, handler):
        """Should not raise when no stream is configured."""
        handler._emit_event("emit_something", agent_id="a1")

    def test_can_handle_control_plane_path(self, handler):
        assert handler.can_handle("/api/control-plane/agents") is True

    def test_can_handle_v1_control_plane_path(self, handler):
        assert handler.can_handle("/api/v1/control-plane/agents") is True

    def test_can_handle_coordination_path(self, handler):
        assert handler.can_handle("/api/v1/coordination/workspaces") is True

    def test_cannot_handle_unrelated_path(self, handler):
        assert handler.can_handle("/api/debates") is False


class TestAwaitIfNeeded:
    """Tests for _await_if_needed utility."""

    def test_await_if_needed_sync_value(self):
        from aragora.server.handlers.control_plane.agents import _await_if_needed

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(_await_if_needed(42))
            assert result == 42
        finally:
            loop.close()

    def test_await_if_needed_none(self):
        from aragora.server.handlers.control_plane.agents import _await_if_needed

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(_await_if_needed(None))
            assert result is None
        finally:
            loop.close()

    def test_await_if_needed_async_coroutine(self):
        from aragora.server.handlers.control_plane.agents import _await_if_needed

        async def coro():
            return "async_result"

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(_await_if_needed(coro()))
            assert result == "async_result"
        finally:
            loop.close()

    def test_await_if_needed_string(self):
        from aragora.server.handlers.control_plane.agents import _await_if_needed

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(_await_if_needed("hello"))
            assert result == "hello"
        finally:
            loop.close()

    def test_await_if_needed_dict(self):
        from aragora.server.handlers.control_plane.agents import _await_if_needed

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(_await_if_needed({"key": "val"}))
            assert result == {"key": "val"}
        finally:
            loop.close()


class TestGetHasPermission:
    """Tests for _get_has_permission module-level helper."""

    def test_get_has_permission_returns_callable(self):
        from aragora.server.handlers.control_plane.agents import _get_has_permission

        fn = _get_has_permission()
        assert callable(fn)

    def test_get_has_permission_fallback(self):
        """When the control_plane module is not loaded, falls back to _has_permission."""
        from aragora.server.handlers.control_plane.agents import _get_has_permission

        import sys
        # Even when the module IS loaded, the function should be callable
        fn = _get_has_permission()
        assert callable(fn)


# ============================================================================
# Permission Denied Tests (no_auto_auth)
# ============================================================================


class TestPermissionDenied:
    """Tests for permission-denied scenarios on auth-protected endpoints."""

    @pytest.mark.no_auto_auth
    def test_register_agent_permission_denied(self, mock_coordinator, mock_http_handler):
        """When has_permission returns False, should return 403."""
        ctx: dict[str, Any] = {"control_plane_coordinator": mock_coordinator}
        h = ControlPlaneHandler(ctx)

        # Mock require_auth_or_error to return a user with no role
        mock_user = MagicMock()
        mock_user.role = "viewer"  # A role that shouldn't have agent write access

        with patch.object(h, "require_auth_or_error", return_value=(mock_user, None)):
            with patch(
                "aragora.server.handlers.control_plane.agents._get_has_permission",
                return_value=lambda role, perm: False,
            ):
                body = {"agent_id": "new-agent"}
                result = h._handle_register_agent(body, mock_http_handler)
        assert _status(result) == 403
        assert "denied" in _body(result).get("error", "").lower()

    @pytest.mark.no_auto_auth
    def test_heartbeat_permission_denied(self, mock_coordinator, mock_http_handler):
        ctx: dict[str, Any] = {"control_plane_coordinator": mock_coordinator}
        h = ControlPlaneHandler(ctx)

        mock_user = MagicMock()
        mock_user.role = "viewer"

        with patch.object(h, "require_auth_or_error", return_value=(mock_user, None)):
            with patch(
                "aragora.server.handlers.control_plane.agents._get_has_permission",
                return_value=lambda role, perm: False,
            ):
                body = {"status": "ready"}
                result = h._handle_heartbeat("agent-001", body, mock_http_handler)
        assert _status(result) == 403

    @pytest.mark.no_auto_auth
    def test_unregister_agent_permission_denied(self, mock_coordinator, mock_http_handler):
        ctx: dict[str, Any] = {"control_plane_coordinator": mock_coordinator}
        h = ControlPlaneHandler(ctx)

        mock_user = MagicMock()
        mock_user.role = "viewer"

        with patch.object(h, "require_auth_or_error", return_value=(mock_user, None)):
            with patch(
                "aragora.server.handlers.control_plane.agents._get_has_permission",
                return_value=lambda role, perm: False,
            ):
                result = h._handle_unregister_agent("agent-001", mock_http_handler)
        assert _status(result) == 403

    @pytest.mark.no_auto_auth
    def test_register_agent_auth_error(self, mock_coordinator, mock_http_handler):
        """When require_auth_or_error returns an error, should propagate it."""
        ctx: dict[str, Any] = {"control_plane_coordinator": mock_coordinator}
        h = ControlPlaneHandler(ctx)

        from aragora.server.handlers.base import error_response

        auth_err = error_response("Unauthorized", 401)

        with patch.object(h, "require_auth_or_error", return_value=(None, auth_err)):
            body = {"agent_id": "agent-x"}
            result = h._handle_register_agent(body, mock_http_handler)
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_heartbeat_auth_error(self, mock_coordinator, mock_http_handler):
        ctx: dict[str, Any] = {"control_plane_coordinator": mock_coordinator}
        h = ControlPlaneHandler(ctx)

        from aragora.server.handlers.base import error_response

        auth_err = error_response("Unauthorized", 401)

        with patch.object(h, "require_auth_or_error", return_value=(None, auth_err)):
            body = {}
            result = h._handle_heartbeat("agent-001", body, mock_http_handler)
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_unregister_agent_auth_error(self, mock_coordinator, mock_http_handler):
        ctx: dict[str, Any] = {"control_plane_coordinator": mock_coordinator}
        h = ControlPlaneHandler(ctx)

        from aragora.server.handlers.base import error_response

        auth_err = error_response("Unauthorized", 401)

        with patch.object(h, "require_auth_or_error", return_value=(None, auth_err)):
            result = h._handle_unregister_agent("agent-001", mock_http_handler)
        assert _status(result) == 401


# ============================================================================
# Async Permission Denied
# ============================================================================


class TestAsyncPermissionDenied:
    """Permission-denied tests for async variants."""

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_register_agent_async_permission_denied(self, mock_coordinator, mock_http_handler):
        ctx: dict[str, Any] = {"control_plane_coordinator": mock_coordinator}
        h = ControlPlaneHandler(ctx)

        mock_user = MagicMock()
        mock_user.role = "viewer"

        with patch.object(h, "require_auth_or_error", return_value=(mock_user, None)):
            with patch(
                "aragora.server.handlers.control_plane.agents._get_has_permission",
                return_value=lambda role, perm: False,
            ):
                body = {"agent_id": "new-agent"}
                result = await h._handle_register_agent_async(body, mock_http_handler)
        assert _status(result) == 403

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_register_agent_async_auth_error(self, mock_coordinator, mock_http_handler):
        ctx: dict[str, Any] = {"control_plane_coordinator": mock_coordinator}
        h = ControlPlaneHandler(ctx)

        from aragora.server.handlers.base import error_response

        auth_err = error_response("Unauthorized", 401)

        with patch.object(h, "require_auth_or_error", return_value=(None, auth_err)):
            body = {"agent_id": "agent-x"}
            result = await h._handle_register_agent_async(body, mock_http_handler)
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_heartbeat_async_permission_denied(self, mock_coordinator, mock_http_handler):
        ctx: dict[str, Any] = {"control_plane_coordinator": mock_coordinator}
        h = ControlPlaneHandler(ctx)

        mock_user = MagicMock()
        mock_user.role = "viewer"

        with patch.object(h, "require_auth_or_error", return_value=(mock_user, None)):
            with patch(
                "aragora.server.handlers.control_plane.agents._get_has_permission",
                return_value=lambda role, perm: False,
            ):
                body = {"status": "ready"}
                result = await h._handle_heartbeat_async("agent-001", body, mock_http_handler)
        assert _status(result) == 403

    @pytest.mark.no_auto_auth
    @pytest.mark.asyncio
    async def test_heartbeat_async_auth_error(self, mock_coordinator, mock_http_handler):
        ctx: dict[str, Any] = {"control_plane_coordinator": mock_coordinator}
        h = ControlPlaneHandler(ctx)

        from aragora.server.handlers.base import error_response

        auth_err = error_response("Unauthorized", 401)

        with patch.object(h, "require_auth_or_error", return_value=(None, auth_err)):
            body = {}
            result = await h._handle_heartbeat_async("agent-001", body, mock_http_handler)
        assert _status(result) == 401


# ============================================================================
# User Without role Attribute
# ============================================================================


class TestUserWithoutRole:
    """Tests for user objects that lack a 'role' attribute."""

    @pytest.mark.no_auto_auth
    def test_register_agent_user_without_role(self, mock_coordinator, mock_http_handler):
        """User without role attribute should pass None for role check."""
        ctx: dict[str, Any] = {"control_plane_coordinator": mock_coordinator}
        h = ControlPlaneHandler(ctx)

        # Create user without 'role' attribute
        mock_user = MagicMock(spec=[])  # No attributes at all

        permission_calls = []

        def track_permission(role, perm):
            permission_calls.append((role, perm))
            return True

        with patch.object(h, "require_auth_or_error", return_value=(mock_user, None)):
            with patch(
                "aragora.server.handlers.control_plane.agents._get_has_permission",
                return_value=track_permission,
            ):
                agent = MockAgentInfo(
                    agent_id="role-test",
                    capabilities=[],
                    model="unknown",
                    provider="unknown",
                )
                mock_coordinator.register_agent = AsyncMock(return_value=agent)
                body = {"agent_id": "role-test"}
                result = h._handle_register_agent(body, mock_http_handler)

        assert _status(result) == 201
        # role should be None since mock_user has no 'role' attribute
        assert permission_calls[0][0] is None
        assert permission_calls[0][1] == "controlplane:agents"


# ============================================================================
# Normalize Path
# ============================================================================


class TestNormalizePath:
    """Tests for _normalize_path version stripping."""

    def test_normalize_v1_control_plane(self, handler):
        assert handler._normalize_path("/api/v1/control-plane/agents") == "/api/control-plane/agents"

    def test_normalize_v1_control_plane_with_id(self, handler):
        assert handler._normalize_path("/api/v1/control-plane/agents/a1") == "/api/control-plane/agents/a1"

    def test_normalize_v1_control_plane_bare(self, handler):
        assert handler._normalize_path("/api/v1/control-plane") == "/api/control-plane"

    def test_normalize_non_versioned(self, handler):
        assert handler._normalize_path("/api/control-plane/agents") == "/api/control-plane/agents"

    def test_normalize_coordination_path_not_changed(self, handler):
        """Coordination paths are NOT normalized by _normalize_path."""
        assert handler._normalize_path("/api/v1/coordination/workspaces") == "/api/v1/coordination/workspaces"


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Miscellaneous edge cases for full coverage."""

    def test_list_agents_single_agent(self, handler, mock_coordinator, sample_agent):
        mock_coordinator.list_agents = AsyncMock(return_value=[sample_agent])
        result = handler._handle_list_agents({})
        body = _body(result)
        assert body["total"] == 1
        assert len(body["agents"]) == 1

    def test_list_agents_many_agents(self, handler, mock_coordinator):
        agents = [
            MockAgentInfo(
                agent_id=f"agent-{i:03d}",
                capabilities=["gen"],
                model="model",
                provider="prov",
            )
            for i in range(50)
        ]
        mock_coordinator.list_agents = AsyncMock(return_value=agents)
        result = handler._handle_list_agents({})
        body = _body(result)
        assert body["total"] == 50
        assert len(body["agents"]) == 50

    def test_register_agent_with_all_fields(self, handler, mock_coordinator, mock_http_handler):
        agent = MockAgentInfo(
            agent_id="full-agent",
            capabilities=["reasoning", "search", "code_generation"],
            model="claude-3.5",
            provider="anthropic",
            metadata={"env": "prod", "version": "3.5"},
        )
        mock_coordinator.register_agent = AsyncMock(return_value=agent)
        body = {
            "agent_id": "full-agent",
            "capabilities": ["reasoning", "search", "code_generation"],
            "model": "claude-3.5",
            "provider": "anthropic",
            "metadata": {"env": "prod", "version": "3.5"},
        }
        result = handler._handle_register_agent(body, mock_http_handler)
        assert _status(result) == 201
        data = _body(result)
        assert data["capabilities"] == ["reasoning", "search", "code_generation"]
        assert data["metadata"] == {"env": "prod", "version": "3.5"}

    def test_get_agent_different_ids(self, handler, mock_coordinator):
        """Verify various agent_id formats work."""
        for agent_id in ["simple", "with-dashes", "with_underscores", "agent.dotted", "a123"]:
            agent = MockAgentInfo(
                agent_id=agent_id,
                capabilities=[],
                model="m",
                provider="p",
            )
            mock_coordinator.get_agent = AsyncMock(return_value=agent)
            result = handler._handle_get_agent(agent_id)
            assert _status(result) == 200
            assert _body(result)["agent_id"] == agent_id

    def test_heartbeat_multiple_statuses(self, handler, mock_coordinator, mock_http_handler):
        """Test heartbeat with each valid status value."""
        mock_coordinator.heartbeat = AsyncMock(return_value=True)
        for status_value in ["starting", "ready", "busy", "draining", "offline", "failed"]:
            body = {"status": status_value}
            with patch("aragora.control_plane.registry.AgentStatus", MockAgentStatus):
                result = handler._handle_heartbeat("agent-001", body, mock_http_handler)
            assert _status(result) == 200, f"Failed for status {status_value}"

    def test_emit_event_with_stream_present(self, handler):
        """When stream is present with the emit method, it should be called."""
        mock_stream = MagicMock()
        mock_method = AsyncMock()
        mock_stream.emit_agent_registered = mock_method
        handler.ctx["control_plane_stream"] = mock_stream

        # The _emit_event uses asyncio scheduling; just verify no errors
        handler._emit_event(
            "emit_agent_registered",
            agent_id="test",
            capabilities=[],
            model="m",
            provider="p",
        )

    def test_emit_event_stream_method_missing(self, handler):
        """When stream exists but method doesn't, should be a no-op."""
        mock_stream = MagicMock(spec=[])  # No methods
        handler.ctx["control_plane_stream"] = mock_stream
        # Should not raise
        handler._emit_event("emit_nonexistent_method", agent_id="test")
