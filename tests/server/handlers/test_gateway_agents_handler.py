"""
Tests for GatewayAgentsHandler - External agent registration HTTP endpoints.

Tests cover:
- Agent registration with validation
- Agent listing and retrieval
- Agent deletion
- SSRF protection on base_url
- Input validation (name, framework_type, base_url)
- Duplicate detection
- Gateway unavailability handling
- Path routing
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.gateway_agents_handler import (
    AGENT_NAME_PATTERN,
    ALLOWED_FRAMEWORKS,
    GatewayAgentsHandler,
)


# ===========================================================================
# Test Fixtures and Helpers
# ===========================================================================


def make_mock_handler(body: dict[str, Any] | None = None) -> MagicMock:
    """Create a mock HTTP request handler with optional JSON body."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 8080)

    if body is not None:
        body_bytes = json.dumps(body).encode()
        handler.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body_bytes)),
        }
        handler.rfile = MagicMock()
        handler.rfile.read.return_value = body_bytes
    else:
        handler.headers = {"Content-Type": "application/json", "Content-Length": "0"}
        handler.rfile = MagicMock()
        handler.rfile.read.return_value = b""

    return handler


def valid_agent_body(
    name: str = "my-crewai-agent",
    framework_type: str = "crewai",
    base_url: str = "https://external-service.example.com",
    **overrides: Any,
) -> dict[str, Any]:
    """Create a valid agent registration body."""
    body: dict[str, Any] = {
        "name": name,
        "framework_type": framework_type,
        "base_url": base_url,
    }
    body.update(overrides)
    return body


@pytest.fixture
def server_context() -> dict[str, Any]:
    """Create server context with external_agents dict."""
    return {"external_agents": {}}


@pytest.fixture
def handler(server_context: dict[str, Any]) -> GatewayAgentsHandler:
    """Create handler with clean context."""
    return GatewayAgentsHandler(server_context)


# ===========================================================================
# Path Routing Tests
# ===========================================================================


class TestPathRouting:
    """Test path matching and routing."""

    def test_can_handle_agent_paths(self, handler: GatewayAgentsHandler) -> None:
        """Handler should match /api/v1/gateway/agents paths."""
        assert handler.can_handle("/api/v1/gateway/agents")
        assert handler.can_handle("/api/v1/gateway/agents/")
        assert handler.can_handle("/api/v1/gateway/agents/my-agent")
        assert handler.can_handle("/api/v1/gateway/agents/some-agent-name")

    def test_cannot_handle_other_paths(self, handler: GatewayAgentsHandler) -> None:
        """Handler should not match non-agent paths."""
        assert not handler.can_handle("/api/v1/gateway/devices")
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/v1/gateway/channels")
        assert not handler.can_handle("/api/gateway/agents")  # Missing v1

    def test_non_matching_path_returns_none(self, handler: GatewayAgentsHandler) -> None:
        """Unmatched path should return None from handle methods."""
        mock_handler = make_mock_handler()

        result = handler.handle("/api/v1/debates", {}, mock_handler)
        assert result is None

        result = handler.handle_post("/api/v1/debates", {}, mock_handler)
        assert result is None

        result = handler.handle_delete("/api/v1/debates", {}, mock_handler)
        assert result is None


# ===========================================================================
# Agent Registration Tests
# ===========================================================================


class TestRegisterAgent:
    """Test POST /api/v1/gateway/agents."""

    def test_register_agent_success(self, handler: GatewayAgentsHandler) -> None:
        """Successful registration returns 201 with agent info."""
        body = valid_agent_body()
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/agents", {}, mock_handler)

        assert result is not None
        assert result.status_code == 201

        data = json.loads(result.body)
        assert data["name"] == "my-crewai-agent"
        assert data["framework_type"] == "crewai"
        assert data["base_url"] == "https://external-service.example.com"
        assert data["registered"] is True
        assert "message" in data

    def test_register_agent_missing_name(self, handler: GatewayAgentsHandler) -> None:
        """Registration without name returns 400."""
        body = valid_agent_body()
        del body["name"]
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/agents", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400

    def test_register_agent_missing_base_url(self, handler: GatewayAgentsHandler) -> None:
        """Registration without base_url returns 400."""
        body = valid_agent_body()
        del body["base_url"]
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/agents", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400

    def test_register_agent_missing_framework(self, handler: GatewayAgentsHandler) -> None:
        """Registration without framework_type returns 400."""
        body = valid_agent_body()
        del body["framework_type"]
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/agents", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400

    def test_register_agent_invalid_name(self, handler: GatewayAgentsHandler) -> None:
        """Registration with invalid name returns 400."""
        body = valid_agent_body(name="invalid name with spaces!")
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/agents", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400

    def test_register_agent_name_too_long(self, handler: GatewayAgentsHandler) -> None:
        """Registration with name over 64 characters returns 400."""
        body = valid_agent_body(name="a" * 65)
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/agents", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400

    def test_register_agent_invalid_framework(self, handler: GatewayAgentsHandler) -> None:
        """Registration with unknown framework returns 400."""
        body = valid_agent_body(framework_type="unknown_framework")
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/agents", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "framework_type" in data["error"]

    def test_register_agent_duplicate_name(
        self, handler: GatewayAgentsHandler, server_context: dict[str, Any]
    ) -> None:
        """Registration with existing name returns 409 Conflict."""
        # Pre-populate an agent
        server_context["external_agents"]["my-crewai-agent"] = {
            "name": "my-crewai-agent",
            "framework_type": "crewai",
            "base_url": "https://existing.example.com",
        }

        body = valid_agent_body()
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/agents", {}, mock_handler)

        assert result is not None
        assert result.status_code == 409

    def test_register_agent_invalid_json(self, handler: GatewayAgentsHandler) -> None:
        """Registration with invalid JSON body returns 400."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 8080)
        mock_handler.headers = {
            "Content-Type": "application/json",
            "Content-Length": "12",
        }
        mock_handler.rfile = MagicMock()
        mock_handler.rfile.read.return_value = b"not valid json"

        result = handler.handle_post("/api/v1/gateway/agents", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400

    def test_register_agent_with_config(self, handler: GatewayAgentsHandler) -> None:
        """Registration stores config field properly."""
        config = {"model": "gpt-4", "temperature": 0.7}
        body = valid_agent_body(config=config)
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/agents", {}, mock_handler)

        assert result is not None
        assert result.status_code == 201

        # Verify stored config
        agents = handler._get_external_agents()
        assert "my-crewai-agent" in agents
        assert agents["my-crewai-agent"]["config"] == config

    def test_register_agent_default_timeout(self, handler: GatewayAgentsHandler) -> None:
        """Registration without timeout uses default of 30."""
        body = valid_agent_body()
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/agents", {}, mock_handler)

        assert result is not None
        assert result.status_code == 201

        agents = handler._get_external_agents()
        assert agents["my-crewai-agent"]["timeout"] == 30


# ===========================================================================
# SSRF Validation Tests
# ===========================================================================


class TestSSRFValidation:
    """Test SSRF protection on base_url."""

    def test_register_agent_ssrf_blocked(self, handler: GatewayAgentsHandler) -> None:
        """Registration with private IP base_url is rejected."""
        body = valid_agent_body(base_url="https://192.168.1.100/api")
        mock_handler = make_mock_handler(body)

        # Mock SSRF validation to return unsafe result
        with (
            patch("aragora.server.handlers.gateway_agents_handler.SSRF_AVAILABLE", True),
            patch("aragora.server.handlers.gateway_agents_handler.validate_url") as mock_validate,
        ):
            mock_result = MagicMock()
            mock_result.is_safe = False
            mock_result.error = "URL resolves to private IP address"
            mock_validate.return_value = mock_result

            result = handler.handle_post("/api/v1/gateway/agents", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "security validation" in data["error"]

    def test_register_agent_ssrf_localhost_blocked(self, handler: GatewayAgentsHandler) -> None:
        """Registration with localhost base_url is rejected."""
        body = valid_agent_body(base_url="https://localhost:8080/api")
        mock_handler = make_mock_handler(body)

        with (
            patch("aragora.server.handlers.gateway_agents_handler.SSRF_AVAILABLE", True),
            patch("aragora.server.handlers.gateway_agents_handler.validate_url") as mock_validate,
        ):
            mock_result = MagicMock()
            mock_result.is_safe = False
            mock_result.error = "URL hostname resolves to localhost"
            mock_validate.return_value = mock_result

            result = handler.handle_post("/api/v1/gateway/agents", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "security validation" in data["error"]

    def test_register_agent_http_rejected(self, handler: GatewayAgentsHandler) -> None:
        """Registration with HTTP base_url is rejected without env var."""
        body = valid_agent_body(base_url="http://external-service.example.com")
        mock_handler = make_mock_handler(body)

        with patch.dict("os.environ", {}, clear=False):
            # Ensure ARAGORA_ALLOW_INSECURE_AGENTS is not set
            env = dict(
                **{
                    k: v
                    for k, v in __import__("os").environ.items()
                    if k != "ARAGORA_ALLOW_INSECURE_AGENTS"
                }
            )
            with patch.dict("os.environ", env, clear=True):
                result = handler.handle_post("/api/v1/gateway/agents", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "HTTPS" in data["error"]


# ===========================================================================
# List Agents Tests
# ===========================================================================


class TestListAgents:
    """Test GET /api/v1/gateway/agents."""

    def test_list_agents_empty(self, handler: GatewayAgentsHandler) -> None:
        """Listing with no agents returns empty list."""
        mock_handler = make_mock_handler()

        result = handler.handle("/api/v1/gateway/agents", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200

        data = json.loads(result.body)
        assert data["agents"] == []
        assert data["total"] == 0

    def test_list_agents_populated(
        self, handler: GatewayAgentsHandler, server_context: dict[str, Any]
    ) -> None:
        """Listing returns all registered agents."""
        server_context["external_agents"] = {
            "agent-a": {
                "name": "agent-a",
                "framework_type": "crewai",
                "base_url": "https://a.example.com",
                "timeout": 30,
            },
            "agent-b": {
                "name": "agent-b",
                "framework_type": "langgraph",
                "base_url": "https://b.example.com",
                "timeout": 60,
            },
        }

        mock_handler = make_mock_handler()

        result = handler.handle("/api/v1/gateway/agents", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200

        data = json.loads(result.body)
        assert data["total"] == 2
        assert len(data["agents"]) == 2

        names = {a["name"] for a in data["agents"]}
        assert names == {"agent-a", "agent-b"}

        for agent in data["agents"]:
            assert "framework_type" in agent
            assert "base_url" in agent
            assert "timeout" in agent
            assert agent["status"] == "registered"


# ===========================================================================
# Get Agent Details Tests
# ===========================================================================


class TestGetAgentDetails:
    """Test GET /api/v1/gateway/agents/{name}."""

    def test_get_agent_details(
        self, handler: GatewayAgentsHandler, server_context: dict[str, Any]
    ) -> None:
        """Getting an existing agent returns full details."""
        server_context["external_agents"]["my-agent"] = {
            "name": "my-agent",
            "framework_type": "crewai",
            "base_url": "https://service.example.com",
            "timeout": 45,
            "config": {"model": "gpt-4"},
        }

        mock_handler = make_mock_handler()

        result = handler.handle("/api/v1/gateway/agents/my-agent", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200

        data = json.loads(result.body)
        assert data["name"] == "my-agent"
        assert data["framework_type"] == "crewai"
        assert data["base_url"] == "https://service.example.com"
        assert data["timeout"] == 45
        assert data["status"] == "registered"
        assert data["config"] == {"model": "gpt-4"}

    def test_get_agent_not_found(self, handler: GatewayAgentsHandler) -> None:
        """Getting a non-existent agent returns 404."""
        mock_handler = make_mock_handler()

        result = handler.handle("/api/v1/gateway/agents/nonexistent", {}, mock_handler)

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Delete Agent Tests
# ===========================================================================


class TestDeleteAgent:
    """Test DELETE /api/v1/gateway/agents/{name}."""

    def test_delete_agent_success(
        self, handler: GatewayAgentsHandler, server_context: dict[str, Any]
    ) -> None:
        """Deleting an existing agent removes it and returns 200."""
        server_context["external_agents"]["my-agent"] = {
            "name": "my-agent",
            "framework_type": "crewai",
            "base_url": "https://service.example.com",
        }

        mock_handler = make_mock_handler()

        result = handler.handle_delete("/api/v1/gateway/agents/my-agent", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200

        data = json.loads(result.body)
        assert data["name"] == "my-agent"
        assert "message" in data

        # Verify agent was removed
        assert "my-agent" not in server_context["external_agents"]

    def test_delete_agent_not_found(self, handler: GatewayAgentsHandler) -> None:
        """Deleting a non-existent agent returns 404."""
        mock_handler = make_mock_handler()

        result = handler.handle_delete("/api/v1/gateway/agents/nonexistent", {}, mock_handler)

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Gateway Unavailable Tests
# ===========================================================================


class TestGatewayUnavailable:
    """Test behavior when gateway module is not available."""

    def test_gateway_unavailable_register(self, server_context: dict[str, Any]) -> None:
        """POST returns 503 when gateway not available."""
        with patch("aragora.server.handlers.gateway_agents_handler.GATEWAY_AVAILABLE", False):
            h = GatewayAgentsHandler(server_context)
            body = valid_agent_body()
            mock_handler = make_mock_handler(body)

            result = h.handle_post("/api/v1/gateway/agents", {}, mock_handler)

            assert result is not None
            assert result.status_code == 503

    def test_gateway_unavailable_list(self, server_context: dict[str, Any]) -> None:
        """GET returns 503 when gateway not available."""
        with patch("aragora.server.handlers.gateway_agents_handler.GATEWAY_AVAILABLE", False):
            h = GatewayAgentsHandler(server_context)
            mock_handler = make_mock_handler()

            result = h.handle("/api/v1/gateway/agents", {}, mock_handler)

            assert result is not None
            assert result.status_code == 503
