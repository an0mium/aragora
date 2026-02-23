"""Tests for gateway agents handler (aragora/server/handlers/gateway_agents_handler.py).

Covers all routes and behavior of the GatewayAgentsHandler class:
- can_handle() routing for all ROUTES and non-matching paths
- POST   /api/v1/gateway/agents          - Register a new external agent
- GET    /api/v1/gateway/agents          - List registered agents
- GET    /api/v1/gateway/agents/{name}   - Get agent details
- DELETE /api/v1/gateway/agents/{name}   - Unregister an agent
- Gateway unavailable (503) responses when GATEWAY_AVAILABLE is False
- Input validation (missing fields, invalid names, invalid frameworks, invalid URLs)
- SSRF protection for base_url
- Duplicate agent registration (409)
- HTTPS enforcement for base_url
- Agent name pattern validation
- Circuit breaker helper functions
- Module exports
- Security tests (path traversal, injection)
- End-to-end lifecycle tests
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.gateway_agents_handler import (
    AGENT_NAME_PATTERN,
    ALLOWED_FRAMEWORKS,
    DEFAULT_TIMEOUT,
    GatewayAgentsHandler,
    get_gateway_agents_circuit_breaker,
    get_gateway_agents_circuit_breaker_status,
    reset_gateway_agents_circuit_breaker,
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


class MockHTTPHandlerInvalidJSON:
    """Mock HTTP handler returning invalid JSON."""

    def __init__(self):
        self.rfile = MagicMock()
        self.rfile.read.return_value = b"NOT-JSON"
        self.headers = {
            "Content-Length": "8",
            "Content-Type": "application/json",
        }
        self.client_address = ("127.0.0.1", 54321)


class MockHTTPHandlerNoBody:
    """Mock HTTP handler with no body content."""

    def __init__(self):
        self.rfile = MagicMock()
        self.rfile.read.return_value = b""
        self.headers = {
            "Content-Length": "0",
            "Content-Type": "application/json",
        }
        self.client_address = ("127.0.0.1", 54321)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_gateway_available():
    """Ensure GATEWAY_AVAILABLE is True by default for most tests."""
    with patch(
        "aragora.server.handlers.gateway_agents_handler.GATEWAY_AVAILABLE", True
    ):
        yield


@pytest.fixture(autouse=True)
def _reset_circuit_breaker():
    """Reset circuit breaker state between tests."""
    reset_gateway_agents_circuit_breaker()
    yield
    reset_gateway_agents_circuit_breaker()


@pytest.fixture
def handler():
    """Create a GatewayAgentsHandler instance with empty context."""
    return GatewayAgentsHandler(ctx={})


@pytest.fixture
def handler_with_agents():
    """Create a handler with pre-registered agents."""
    h = GatewayAgentsHandler(ctx={
        "external_agents": {
            "test-agent-1": {
                "name": "test-agent-1",
                "framework_type": "crewai",
                "base_url": "https://agent1.example.com",
                "timeout": 30,
                "config": {"key": "value"},
            },
            "test-agent-2": {
                "name": "test-agent-2",
                "framework_type": "langgraph",
                "base_url": "https://agent2.example.com",
                "timeout": 60,
                "config": {},
            },
        }
    })
    return h


# ===========================================================================
# can_handle routing tests
# ===========================================================================


class TestCanHandle:
    """Test the can_handle path routing."""

    def test_handles_gateway_agents_base(self, handler):
        assert handler.can_handle("/api/v1/gateway/agents") is True

    def test_handles_gateway_agents_with_name(self, handler):
        assert handler.can_handle("/api/v1/gateway/agents/my-agent") is True

    def test_handles_gateway_agents_trailing_slash(self, handler):
        assert handler.can_handle("/api/v1/gateway/agents/") is True

    def test_rejects_non_gateway_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_rejects_root(self, handler):
        assert handler.can_handle("/") is False

    def test_rejects_partial_prefix(self, handler):
        assert handler.can_handle("/api/v1/gateway/age") is False

    def test_rejects_gateway_devices(self, handler):
        assert handler.can_handle("/api/v1/gateway/devices") is False

    def test_rejects_gateway_credentials(self, handler):
        assert handler.can_handle("/api/v1/gateway/credentials") is False

    def test_handles_deep_nested_path(self, handler):
        assert handler.can_handle("/api/v1/gateway/agents/name/extra") is True


# ===========================================================================
# ROUTES class attribute
# ===========================================================================


class TestRoutesAttribute:
    """Verify the ROUTES class attribute lists expected patterns."""

    def test_routes_contains_agents_base(self):
        assert "/api/v1/gateway/agents" in GatewayAgentsHandler.ROUTES

    def test_routes_contains_agents_wildcard(self):
        assert "/api/v1/gateway/agents/*" in GatewayAgentsHandler.ROUTES

    def test_routes_count(self):
        assert len(GatewayAgentsHandler.ROUTES) == 2


# ===========================================================================
# _extract_agent_name path parsing
# ===========================================================================


class TestExtractAgentName:
    """Test the _extract_agent_name helper."""

    def test_extract_name_from_valid_path(self, handler):
        assert handler._extract_agent_name("/api/v1/gateway/agents/my-agent") == "my-agent"

    def test_extract_name_with_trailing_slash(self, handler):
        assert handler._extract_agent_name("/api/v1/gateway/agents/my-agent/") == "my-agent"

    def test_extract_name_returns_none_for_base_path(self, handler):
        assert handler._extract_agent_name("/api/v1/gateway/agents") is None

    def test_extract_name_returns_none_for_agents_only(self, handler):
        """Path where the name segment equals 'agents' should return None."""
        assert handler._extract_agent_name("/api/v1/gateway/agents/agents") is None

    def test_extract_name_with_dashes_and_underscores(self, handler):
        assert handler._extract_agent_name("/api/v1/gateway/agents/my_agent-v2") == "my_agent-v2"

    def test_extract_name_short_path_returns_none(self, handler):
        assert handler._extract_agent_name("/api/v1/gateway") is None

    def test_extract_name_empty_path_returns_none(self, handler):
        assert handler._extract_agent_name("") is None


# ===========================================================================
# GET /api/v1/gateway/agents (list)
# ===========================================================================


class TestListAgents:
    """Test GET /api/v1/gateway/agents."""

    def test_list_agents_empty(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["agents"] == []
        assert body["total"] == 0

    def test_list_agents_returns_registered_agents(self, handler_with_agents):
        http = MockHTTPHandler()
        result = handler_with_agents.handle("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        assert len(body["agents"]) == 2

        names = {a["name"] for a in body["agents"]}
        assert "test-agent-1" in names
        assert "test-agent-2" in names

    def test_list_agents_includes_correct_fields(self, handler_with_agents):
        http = MockHTTPHandler()
        result = handler_with_agents.handle("/api/v1/gateway/agents", {}, http)
        body = _body(result)

        for agent in body["agents"]:
            assert "name" in agent
            assert "framework_type" in agent
            assert "base_url" in agent
            assert "timeout" in agent
            assert agent["status"] == "registered"

    def test_list_agents_with_trailing_slash(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/agents/", {}, http)
        # Trailing slash on base path — the handler strips and matches
        # _extract_agent_name returns None for empty segment, so it falls through
        # to list check: path.rstrip("/") == "/api/v1/gateway/agents"
        assert _status(result) == 200

    def test_list_agents_default_framework_type(self, handler):
        """Agent without framework_type should default to 'custom'."""
        handler.ctx["external_agents"] = {
            "bare-agent": {
                "name": "bare-agent",
            }
        }
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/agents", {}, http)
        body = _body(result)
        assert body["agents"][0]["framework_type"] == "custom"

    def test_list_agents_default_timeout(self, handler):
        """Agent without timeout should default to DEFAULT_TIMEOUT."""
        handler.ctx["external_agents"] = {
            "bare-agent": {
                "name": "bare-agent",
            }
        }
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/agents", {}, http)
        body = _body(result)
        assert body["agents"][0]["timeout"] == DEFAULT_TIMEOUT

    def test_list_non_matching_path_returns_none(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/debates", {}, http)
        assert result is None


# ===========================================================================
# GET /api/v1/gateway/agents/{name} (get single)
# ===========================================================================


class TestGetAgent:
    """Test GET /api/v1/gateway/agents/{name}."""

    def test_get_agent_found(self, handler_with_agents):
        http = MockHTTPHandler()
        result = handler_with_agents.handle("/api/v1/gateway/agents/test-agent-1", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["name"] == "test-agent-1"
        assert body["framework_type"] == "crewai"
        assert body["base_url"] == "https://agent1.example.com"
        assert body["timeout"] == 30
        assert body["status"] == "registered"
        assert body["config"] == {"key": "value"}

    def test_get_agent_not_found(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/agents/nonexistent", {}, http)
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    def test_get_agent_default_fields(self, handler):
        """Agent with minimal fields returns defaults."""
        handler.ctx["external_agents"] = {
            "min-agent": {"name": "min-agent"}
        }
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/agents/min-agent", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["framework_type"] == "custom"
        assert body["base_url"] == ""
        assert body["timeout"] == DEFAULT_TIMEOUT
        assert body["config"] == {}

    def test_get_agent_with_config(self, handler_with_agents):
        http = MockHTTPHandler()
        result = handler_with_agents.handle("/api/v1/gateway/agents/test-agent-1", {}, http)
        body = _body(result)
        assert body["config"] == {"key": "value"}


# ===========================================================================
# POST /api/v1/gateway/agents (register)
# ===========================================================================


class TestRegisterAgent:
    """Test POST /api/v1/gateway/agents."""

    def test_register_agent_success(self, handler):
        http = MockHTTPHandler(body={
            "name": "my-agent",
            "framework_type": "crewai",
            "base_url": "https://agent.example.com",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 201
        body = _body(result)
        assert body["name"] == "my-agent"
        assert body["framework_type"] == "crewai"
        assert body["base_url"] == "https://agent.example.com"
        assert body["registered"] is True
        assert "successfully" in body["message"].lower()

    def test_register_agent_stored_in_context(self, handler):
        http = MockHTTPHandler(body={
            "name": "stored-agent",
            "framework_type": "autogen",
            "base_url": "https://stored.example.com",
        })
        handler.handle_post("/api/v1/gateway/agents", {}, http)
        agents = handler.ctx["external_agents"]
        assert "stored-agent" in agents
        assert agents["stored-agent"]["framework_type"] == "autogen"

    def test_register_agent_with_optional_fields(self, handler):
        http = MockHTTPHandler(body={
            "name": "full-agent",
            "framework_type": "langgraph",
            "base_url": "https://full.example.com",
            "timeout": 120,
            "config": {"model": "gpt-4"},
            "api_key_env": "MY_API_KEY",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 201
        agents = handler.ctx["external_agents"]
        info = agents["full-agent"]
        assert info["timeout"] == 120
        assert info["config"] == {"model": "gpt-4"}
        assert info["api_key_env"] == "MY_API_KEY"

    def test_register_agent_default_timeout(self, handler):
        http = MockHTTPHandler(body={
            "name": "default-timeout",
            "framework_type": "custom",
            "base_url": "https://default.example.com",
        })
        handler.handle_post("/api/v1/gateway/agents", {}, http)
        agents = handler.ctx["external_agents"]
        assert agents["default-timeout"]["timeout"] == DEFAULT_TIMEOUT

    def test_register_agent_no_api_key_env(self, handler):
        """When api_key_env is not provided, it should not appear in stored info."""
        http = MockHTTPHandler(body={
            "name": "no-key",
            "framework_type": "custom",
            "base_url": "https://nokey.example.com",
        })
        handler.handle_post("/api/v1/gateway/agents", {}, http)
        agents = handler.ctx["external_agents"]
        assert "api_key_env" not in agents["no-key"]

    def test_register_agent_missing_name(self, handler):
        http = MockHTTPHandler(body={
            "framework_type": "crewai",
            "base_url": "https://agent.example.com",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400
        assert "name" in _body(result).get("error", "").lower()

    def test_register_agent_empty_name(self, handler):
        http = MockHTTPHandler(body={
            "name": "",
            "framework_type": "crewai",
            "base_url": "https://agent.example.com",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400

    def test_register_agent_missing_base_url(self, handler):
        http = MockHTTPHandler(body={
            "name": "my-agent",
            "framework_type": "crewai",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400
        assert "base_url" in _body(result).get("error", "").lower()

    def test_register_agent_empty_base_url(self, handler):
        http = MockHTTPHandler(body={
            "name": "my-agent",
            "framework_type": "crewai",
            "base_url": "",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400

    def test_register_agent_missing_framework_type(self, handler):
        http = MockHTTPHandler(body={
            "name": "my-agent",
            "base_url": "https://agent.example.com",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400
        assert "framework_type" in _body(result).get("error", "").lower()

    def test_register_agent_empty_framework_type(self, handler):
        http = MockHTTPHandler(body={
            "name": "my-agent",
            "framework_type": "",
            "base_url": "https://agent.example.com",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400

    def test_register_agent_invalid_framework_type(self, handler):
        http = MockHTTPHandler(body={
            "name": "my-agent",
            "framework_type": "invalid_framework",
            "base_url": "https://agent.example.com",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400
        error = _body(result).get("error", "")
        assert "Invalid framework_type" in error

    def test_register_agent_all_valid_frameworks(self, handler):
        """Each allowed framework should register successfully."""
        for fw in sorted(ALLOWED_FRAMEWORKS):
            h = GatewayAgentsHandler(ctx={})
            http = MockHTTPHandler(body={
                "name": f"agent-{fw}",
                "framework_type": fw,
                "base_url": f"https://{fw}.example.com",
            })
            with patch(
                "aragora.server.handlers.gateway_agents_handler.GATEWAY_AVAILABLE", True
            ):
                result = h.handle_post("/api/v1/gateway/agents", {}, http)
            assert _status(result) == 201, f"Failed for framework: {fw}"

    def test_register_agent_invalid_json(self, handler):
        http = MockHTTPHandlerInvalidJSON()
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400

    def test_register_agent_empty_body(self, handler):
        http = MockHTTPHandlerNoBody()
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400

    def test_register_agent_duplicate_name(self, handler_with_agents):
        http = MockHTTPHandler(body={
            "name": "test-agent-1",
            "framework_type": "custom",
            "base_url": "https://duplicate.example.com",
        })
        result = handler_with_agents.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 409
        assert "already exists" in _body(result).get("error", "").lower()

    def test_register_post_on_non_matching_path_returns_none(self, handler):
        http = MockHTTPHandler(body={
            "name": "agent",
            "framework_type": "custom",
            "base_url": "https://example.com",
        })
        result = handler.handle_post("/api/v1/debates", {}, http)
        assert result is None

    def test_register_post_on_agent_name_path_returns_none(self, handler):
        """POST on /api/v1/gateway/agents/some-name should return None (only base path accepts POST)."""
        http = MockHTTPHandler(body={
            "name": "agent",
            "framework_type": "custom",
            "base_url": "https://example.com",
        })
        result = handler.handle_post("/api/v1/gateway/agents/some-name", {}, http)
        assert result is None


# ===========================================================================
# Agent name validation
# ===========================================================================


class TestAgentNameValidation:
    """Test agent name pattern validation during registration."""

    def test_valid_alphanumeric_name(self, handler):
        http = MockHTTPHandler(body={
            "name": "agent123",
            "framework_type": "custom",
            "base_url": "https://example.com",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 201

    def test_valid_name_with_hyphens(self, handler):
        http = MockHTTPHandler(body={
            "name": "my-agent-v2",
            "framework_type": "custom",
            "base_url": "https://example.com",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 201

    def test_valid_name_with_underscores(self, handler):
        http = MockHTTPHandler(body={
            "name": "my_agent_v2",
            "framework_type": "custom",
            "base_url": "https://example.com",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 201

    def test_invalid_name_starts_with_hyphen(self, handler):
        http = MockHTTPHandler(body={
            "name": "-bad-name",
            "framework_type": "custom",
            "base_url": "https://example.com",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400
        assert "Invalid agent name" in _body(result).get("error", "")

    def test_invalid_name_starts_with_underscore(self, handler):
        http = MockHTTPHandler(body={
            "name": "_bad_name",
            "framework_type": "custom",
            "base_url": "https://example.com",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400

    def test_invalid_name_with_spaces(self, handler):
        http = MockHTTPHandler(body={
            "name": "bad name",
            "framework_type": "custom",
            "base_url": "https://example.com",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400

    def test_invalid_name_with_special_chars(self, handler):
        http = MockHTTPHandler(body={
            "name": "agent@!#",
            "framework_type": "custom",
            "base_url": "https://example.com",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400

    def test_invalid_name_too_long(self, handler):
        """Name longer than 64 characters should be rejected."""
        http = MockHTTPHandler(body={
            "name": "a" * 65,
            "framework_type": "custom",
            "base_url": "https://example.com",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400

    def test_valid_name_max_length(self, handler):
        """Name at exactly 64 characters should be accepted."""
        http = MockHTTPHandler(body={
            "name": "a" * 64,
            "framework_type": "custom",
            "base_url": "https://example.com",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 201

    def test_valid_single_char_name(self, handler):
        http = MockHTTPHandler(body={
            "name": "a",
            "framework_type": "custom",
            "base_url": "https://example.com",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 201


# ===========================================================================
# base_url validation
# ===========================================================================


class TestBaseUrlValidation:
    """Test base_url validation logic."""

    def test_https_url_accepted(self, handler):
        result = handler._validate_base_url("https://example.com")
        assert result is None

    def test_http_url_rejected_by_default(self, handler):
        result = handler._validate_base_url("http://example.com")
        assert result is not None
        assert _status(result) == 400
        assert "HTTPS" in _body(result).get("error", "")

    def test_http_url_accepted_with_insecure_flag(self, handler):
        with patch.dict("os.environ", {"ARAGORA_ALLOW_INSECURE_AGENTS": "true"}):
            result = handler._validate_base_url("http://example.com")
        assert result is None

    def test_http_url_accepted_with_insecure_flag_1(self, handler):
        with patch.dict("os.environ", {"ARAGORA_ALLOW_INSECURE_AGENTS": "1"}):
            result = handler._validate_base_url("http://example.com")
        assert result is None

    def test_http_url_accepted_with_insecure_flag_yes(self, handler):
        with patch.dict("os.environ", {"ARAGORA_ALLOW_INSECURE_AGENTS": "yes"}):
            result = handler._validate_base_url("http://example.com")
        assert result is None

    def test_ftp_url_rejected(self, handler):
        result = handler._validate_base_url("ftp://example.com")
        assert result is not None
        assert _status(result) == 400

    def test_ftp_url_rejected_even_with_insecure_flag(self, handler):
        with patch.dict("os.environ", {"ARAGORA_ALLOW_INSECURE_AGENTS": "true"}):
            result = handler._validate_base_url("ftp://example.com")
        assert result is not None
        assert _status(result) == 400

    def test_empty_url_rejected(self, handler):
        result = handler._validate_base_url("")
        assert result is not None
        assert _status(result) == 400

    def test_ssrf_validation_when_available(self, handler):
        """When SSRF protection is available and URL is unsafe, return 400."""
        mock_result = MagicMock()
        mock_result.is_safe = False
        mock_result.error = "Internal IP detected"
        with patch(
            "aragora.server.handlers.gateway_agents_handler.SSRF_AVAILABLE", True
        ), patch(
            "aragora.security.ssrf_protection.validate_url",
            return_value=mock_result,
        ):
            result = handler._validate_base_url("https://internal.example.com")
        assert result is not None
        assert _status(result) == 400
        assert "security validation" in _body(result).get("error", "").lower()

    def test_ssrf_validation_passes_safe_url(self, handler):
        """When SSRF protection is available and URL is safe, return None."""
        mock_result = MagicMock()
        mock_result.is_safe = True
        with patch(
            "aragora.server.handlers.gateway_agents_handler.SSRF_AVAILABLE", True
        ), patch(
            "aragora.security.ssrf_protection.validate_url",
            return_value=mock_result,
        ):
            result = handler._validate_base_url("https://safe.example.com")
        assert result is None


# ===========================================================================
# DELETE /api/v1/gateway/agents/{name}
# ===========================================================================


class TestDeleteAgent:
    """Test DELETE /api/v1/gateway/agents/{name}."""

    def test_delete_agent_success(self, handler_with_agents):
        http = MockHTTPHandler()
        result = handler_with_agents.handle_delete(
            "/api/v1/gateway/agents/test-agent-1", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["name"] == "test-agent-1"
        assert "unregistered" in body["message"].lower()
        # Verify removed from registry
        assert "test-agent-1" not in handler_with_agents.ctx["external_agents"]

    def test_delete_agent_not_found(self, handler):
        http = MockHTTPHandler()
        result = handler.handle_delete(
            "/api/v1/gateway/agents/nonexistent", {}, http
        )
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    def test_delete_no_name_returns_none(self, handler):
        """DELETE on /api/v1/gateway/agents (no name segment) returns None."""
        http = MockHTTPHandler()
        result = handler.handle_delete("/api/v1/gateway/agents", {}, http)
        assert result is None

    def test_delete_non_matching_path_returns_none(self, handler):
        http = MockHTTPHandler()
        result = handler.handle_delete("/api/v1/debates", {}, http)
        assert result is None

    def test_delete_preserves_other_agents(self, handler_with_agents):
        http = MockHTTPHandler()
        handler_with_agents.handle_delete(
            "/api/v1/gateway/agents/test-agent-1", {}, http
        )
        assert "test-agent-2" in handler_with_agents.ctx["external_agents"]


# ===========================================================================
# Gateway unavailable (503) for all methods
# ===========================================================================


class TestGatewayUnavailable:
    """Test 503 responses when GATEWAY_AVAILABLE is False."""

    @pytest.fixture(autouse=True)
    def _disable_gateway(self):
        with patch(
            "aragora.server.handlers.gateway_agents_handler.GATEWAY_AVAILABLE", False
        ):
            yield

    def test_handle_get_503(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "").lower()

    def test_handle_post_503(self, handler):
        http = MockHTTPHandler(body={
            "name": "agent",
            "framework_type": "custom",
            "base_url": "https://example.com",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 503

    def test_handle_delete_503(self, handler):
        http = MockHTTPHandler()
        result = handler.handle_delete("/api/v1/gateway/agents/test-agent", {}, http)
        assert _status(result) == 503

    def test_handle_get_single_503(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/agents/test-agent", {}, http)
        assert _status(result) == 503


# ===========================================================================
# Unknown routes return None
# ===========================================================================


class TestUnknownRoutes:
    """Test that unknown routes return None."""

    def test_handle_unknown_path(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/other/endpoint", {}, http)
        assert result is None

    def test_handle_post_unknown_path(self, handler):
        http = MockHTTPHandler(body={})
        result = handler.handle_post("/api/v1/other/endpoint", {}, http)
        assert result is None

    def test_handle_delete_unknown_path(self, handler):
        http = MockHTTPHandler()
        result = handler.handle_delete("/api/v1/other/endpoint", {}, http)
        assert result is None


# ===========================================================================
# Circuit breaker helper functions
# ===========================================================================


class TestCircuitBreaker:
    """Test the module-level circuit breaker functions."""

    def test_get_circuit_breaker_returns_instance(self):
        cb = get_gateway_agents_circuit_breaker()
        assert cb is not None
        assert cb.name == "gateway_agents_handler"

    def test_get_circuit_breaker_status_returns_dict(self):
        status = get_gateway_agents_circuit_breaker_status()
        assert isinstance(status, dict)

    def test_reset_circuit_breaker(self):
        cb = get_gateway_agents_circuit_breaker()
        # Simulate some failures
        cb._single_failures = 3
        cb._single_open_at = 100.0
        cb._single_successes = 1
        cb._single_half_open_calls = 2
        reset_gateway_agents_circuit_breaker()
        assert cb._single_failures == 0
        assert cb._single_open_at == 0.0
        assert cb._single_successes == 0
        assert cb._single_half_open_calls == 0

    def test_circuit_breaker_config(self):
        cb = get_gateway_agents_circuit_breaker()
        assert cb.failure_threshold == 5
        assert cb.cooldown_seconds == 30.0


# ===========================================================================
# Module exports
# ===========================================================================


class TestModuleExports:
    """Test that __all__ exports are correct."""

    def test_all_exports(self):
        from aragora.server.handlers import gateway_agents_handler

        assert "GatewayAgentsHandler" in gateway_agents_handler.__all__
        assert "get_gateway_agents_circuit_breaker" in gateway_agents_handler.__all__
        assert "get_gateway_agents_circuit_breaker_status" in gateway_agents_handler.__all__
        assert "reset_gateway_agents_circuit_breaker" in gateway_agents_handler.__all__

    def test_all_exports_count(self):
        from aragora.server.handlers import gateway_agents_handler

        assert len(gateway_agents_handler.__all__) == 4


# ===========================================================================
# Constants
# ===========================================================================


class TestConstants:
    """Test module-level constants."""

    def test_default_timeout_value(self):
        assert DEFAULT_TIMEOUT == 30

    def test_allowed_frameworks(self):
        assert ALLOWED_FRAMEWORKS == {"openclaw", "crewai", "autogen", "langgraph", "custom"}

    def test_agent_name_pattern_valid(self):
        valid = ["a", "abc", "a1", "A1", "my-agent", "my_agent", "agent123", "A" * 64]
        for name in valid:
            assert AGENT_NAME_PATTERN.match(name) is not None, f"Should match: {name}"

    def test_agent_name_pattern_invalid(self):
        invalid = [
            "",
            "-start",
            "_start",
            "a" * 65,
            "bad name",
            "agent@!",
            ".dot",
        ]
        for name in invalid:
            assert AGENT_NAME_PATTERN.match(name) is None, f"Should not match: {name}"


# ===========================================================================
# External agents registry management
# ===========================================================================


class TestExternalAgentsRegistry:
    """Test the _get_external_agents helper."""

    def test_get_external_agents_creates_if_missing(self, handler):
        assert "external_agents" not in handler.ctx
        agents = handler._get_external_agents()
        assert isinstance(agents, dict)
        assert handler.ctx["external_agents"] is agents

    def test_get_external_agents_returns_existing(self, handler):
        handler.ctx["external_agents"] = {"existing": {"name": "existing"}}
        agents = handler._get_external_agents()
        assert "existing" in agents

    def test_get_external_agents_is_mutable(self, handler):
        agents = handler._get_external_agents()
        agents["new"] = {"name": "new"}
        assert "new" in handler.ctx["external_agents"]


# ===========================================================================
# Security tests
# ===========================================================================


class TestSecurity:
    """Security-focused tests for input validation."""

    def test_path_traversal_in_name(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/agents/../../etc/passwd", {}, http)
        # Handler should not crash; _extract_agent_name will extract a segment
        # Either returns 404 (not found) or handles gracefully
        if result is not None:
            assert _status(result) in (400, 404)

    def test_register_agent_name_with_dot_dot(self, handler):
        http = MockHTTPHandler(body={
            "name": "../../../etc/passwd",
            "framework_type": "custom",
            "base_url": "https://example.com",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400
        assert "Invalid agent name" in _body(result).get("error", "")

    def test_register_agent_name_with_null_byte(self, handler):
        http = MockHTTPHandler(body={
            "name": "agent\x00evil",
            "framework_type": "custom",
            "base_url": "https://example.com",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400

    def test_register_agent_name_with_unicode(self, handler):
        http = MockHTTPHandler(body={
            "name": "agent\u00e9",
            "framework_type": "custom",
            "base_url": "https://example.com",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400

    def test_register_agent_sql_injection_in_name(self, handler):
        http = MockHTTPHandler(body={
            "name": "'; DROP TABLE agents;--",
            "framework_type": "custom",
            "base_url": "https://example.com",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400

    def test_register_agent_xss_in_name(self, handler):
        http = MockHTTPHandler(body={
            "name": "<script>alert('xss')</script>",
            "framework_type": "custom",
            "base_url": "https://example.com",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400

    def test_register_agent_base_url_with_javascript_protocol(self, handler):
        http = MockHTTPHandler(body={
            "name": "agent1",
            "framework_type": "custom",
            "base_url": "javascript:alert(1)",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400

    def test_register_agent_base_url_with_data_protocol(self, handler):
        http = MockHTTPHandler(body={
            "name": "agent2",
            "framework_type": "custom",
            "base_url": "data:text/html,<script>alert(1)</script>",
        })
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400


# ===========================================================================
# End-to-end lifecycle tests
# ===========================================================================


class TestEndToEnd:
    """Integration-style tests exercising the full lifecycle."""

    def test_register_list_get_delete(self, handler):
        # 1. Register an agent
        reg_http = MockHTTPHandler(body={
            "name": "lifecycle-agent",
            "framework_type": "openclaw",
            "base_url": "https://lifecycle.example.com",
            "timeout": 45,
            "config": {"version": "1.0"},
        })
        reg_result = handler.handle_post("/api/v1/gateway/agents", {}, reg_http)
        assert _status(reg_result) == 201

        # 2. List agents
        list_http = MockHTTPHandler()
        list_result = handler.handle("/api/v1/gateway/agents", {}, list_http)
        assert _status(list_result) == 200
        assert _body(list_result)["total"] == 1
        assert _body(list_result)["agents"][0]["name"] == "lifecycle-agent"

        # 3. Get agent by name
        get_http = MockHTTPHandler()
        get_result = handler.handle("/api/v1/gateway/agents/lifecycle-agent", {}, get_http)
        assert _status(get_result) == 200
        body = _body(get_result)
        assert body["name"] == "lifecycle-agent"
        assert body["framework_type"] == "openclaw"
        assert body["timeout"] == 45
        assert body["config"] == {"version": "1.0"}

        # 4. Delete agent
        del_http = MockHTTPHandler()
        del_result = handler.handle_delete(
            "/api/v1/gateway/agents/lifecycle-agent", {}, del_http
        )
        assert _status(del_result) == 200

        # 5. List should be empty
        list_http2 = MockHTTPHandler()
        list_result2 = handler.handle("/api/v1/gateway/agents", {}, list_http2)
        assert _body(list_result2)["total"] == 0

        # 6. Get deleted agent returns 404
        get_http2 = MockHTTPHandler()
        get_result2 = handler.handle("/api/v1/gateway/agents/lifecycle-agent", {}, get_http2)
        assert _status(get_result2) == 404

    def test_register_multiple_agents(self, handler):
        """Register multiple agents and verify all are listed."""
        for i in range(5):
            http = MockHTTPHandler(body={
                "name": f"agent-{i}",
                "framework_type": "custom",
                "base_url": f"https://agent{i}.example.com",
            })
            result = handler.handle_post("/api/v1/gateway/agents", {}, http)
            assert _status(result) == 201

        list_http = MockHTTPHandler()
        list_result = handler.handle("/api/v1/gateway/agents", {}, list_http)
        assert _body(list_result)["total"] == 5
        names = {a["name"] for a in _body(list_result)["agents"]}
        for i in range(5):
            assert f"agent-{i}" in names

    def test_register_then_duplicate_rejected(self, handler):
        """After registering, a second registration with same name is rejected."""
        http1 = MockHTTPHandler(body={
            "name": "unique-agent",
            "framework_type": "crewai",
            "base_url": "https://unique.example.com",
        })
        result1 = handler.handle_post("/api/v1/gateway/agents", {}, http1)
        assert _status(result1) == 201

        http2 = MockHTTPHandler(body={
            "name": "unique-agent",
            "framework_type": "autogen",
            "base_url": "https://other.example.com",
        })
        result2 = handler.handle_post("/api/v1/gateway/agents", {}, http2)
        assert _status(result2) == 409


# ===========================================================================
# Handler initialization
# ===========================================================================


class TestInitialization:
    """Test handler initialization."""

    def test_default_ctx(self):
        h = GatewayAgentsHandler()
        assert h.ctx == {}

    def test_custom_ctx(self):
        ctx = {"key": "value"}
        h = GatewayAgentsHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_none_ctx_becomes_empty_dict(self):
        h = GatewayAgentsHandler(ctx=None)
        assert h.ctx == {}


# ===========================================================================
# Edge cases for GET handle() routing
# ===========================================================================


class TestHandleRoutingEdgeCases:
    """Test edge cases in handle() path dispatch."""

    def test_get_on_agents_subpath_with_extra_segments(self, handler_with_agents):
        """GET on /api/v1/gateway/agents/test-agent-1/extra tries to get agent named 'test-agent-1'."""
        http = MockHTTPHandler()
        # _extract_agent_name will return "test-agent-1" since parts[5] = "test-agent-1"
        result = handler_with_agents.handle(
            "/api/v1/gateway/agents/test-agent-1/extra", {}, http
        )
        assert _status(result) == 200
        assert _body(result)["name"] == "test-agent-1"

    def test_get_agents_path_case_sensitive(self, handler):
        """Path matching is case-sensitive: /api/v1/gateway/Agents should not match."""
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/Agents", {}, http)
        assert result is None

    def test_handle_returns_none_for_non_gateway_prefix(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v2/gateway/agents", {}, http)
        assert result is None


# ===========================================================================
# Insecure agents environment variable edge cases
# ===========================================================================


class TestInsecureAgentsEnvVar:
    """Test ARAGORA_ALLOW_INSECURE_AGENTS environment variable behavior."""

    def test_insecure_flag_false_does_not_allow_http(self, handler):
        with patch.dict("os.environ", {"ARAGORA_ALLOW_INSECURE_AGENTS": "false"}):
            result = handler._validate_base_url("http://example.com")
        assert result is not None
        assert _status(result) == 400

    def test_insecure_flag_empty_does_not_allow_http(self, handler):
        with patch.dict("os.environ", {"ARAGORA_ALLOW_INSECURE_AGENTS": ""}):
            result = handler._validate_base_url("http://example.com")
        assert result is not None
        assert _status(result) == 400

    def test_insecure_flag_TRUE_uppercase(self, handler):
        with patch.dict("os.environ", {"ARAGORA_ALLOW_INSECURE_AGENTS": "TRUE"}):
            result = handler._validate_base_url("http://example.com")
        assert result is None

    def test_insecure_flag_YES_uppercase(self, handler):
        with patch.dict("os.environ", {"ARAGORA_ALLOW_INSECURE_AGENTS": "YES"}):
            result = handler._validate_base_url("http://example.com")
        assert result is None

    def test_https_always_accepted_regardless_of_flag(self, handler):
        with patch.dict("os.environ", {"ARAGORA_ALLOW_INSECURE_AGENTS": "false"}):
            result = handler._validate_base_url("https://example.com")
        assert result is None


# ===========================================================================
# Multiple agents — delete and re-register
# ===========================================================================


class TestDeleteAndReRegister:
    """Test re-registering an agent after deletion."""

    def test_re_register_after_delete(self, handler):
        # Register
        http1 = MockHTTPHandler(body={
            "name": "reuse-agent",
            "framework_type": "custom",
            "base_url": "https://reuse.example.com",
        })
        result1 = handler.handle_post("/api/v1/gateway/agents", {}, http1)
        assert _status(result1) == 201

        # Delete
        del_http = MockHTTPHandler()
        del_result = handler.handle_delete("/api/v1/gateway/agents/reuse-agent", {}, del_http)
        assert _status(del_result) == 200

        # Re-register with same name but different framework
        http2 = MockHTTPHandler(body={
            "name": "reuse-agent",
            "framework_type": "crewai",
            "base_url": "https://reuse-v2.example.com",
        })
        result2 = handler.handle_post("/api/v1/gateway/agents", {}, http2)
        assert _status(result2) == 201

        # Verify new registration
        get_http = MockHTTPHandler()
        get_result = handler.handle("/api/v1/gateway/agents/reuse-agent", {}, get_http)
        assert _status(get_result) == 200
        assert _body(get_result)["framework_type"] == "crewai"
        assert _body(get_result)["base_url"] == "https://reuse-v2.example.com"
