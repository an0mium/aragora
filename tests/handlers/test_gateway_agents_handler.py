"""Tests for GatewayAgentsHandler (aragora/server/handlers/gateway_agents_handler.py).

Covers all routes and methods:
    POST   /api/v1/gateway/agents          - Register a new external agent
    GET    /api/v1/gateway/agents          - List registered agents
    GET    /api/v1/gateway/agents/{name}   - Get agent details
    DELETE /api/v1/gateway/agents/{name}   - Unregister an agent

Tests include:
- ROUTES class attribute verification
- can_handle() for matching and non-matching paths
- _extract_agent_name() path parsing
- _get_external_agents() registry management
- _validate_base_url() with HTTPS enforcement, insecure flag, SSRF
- GET list (empty, populated, defaults, trailing slash)
- GET single agent (found, not found, defaults, config)
- POST register (success, optional fields, missing fields, invalid name,
  invalid framework, duplicate, invalid JSON, no body)
- DELETE (success, not found, no name segment, preserves other agents)
- 503 responses when GATEWAY_AVAILABLE is False
- Circuit breaker helper functions
- Module __all__ exports
- Constants (AGENT_NAME_PATTERN, ALLOWED_FRAMEWORKS, DEFAULT_TIMEOUT)
- Security (path traversal, SQL injection, XSS, protocol injection)
- End-to-end lifecycle (register -> list -> get -> delete -> verify)
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
    """Extract body dict from a HandlerResult."""
    if result is None:
        return {}
    if hasattr(result, "body"):
        try:
            return json.loads(result.body)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError):
            return {}
    if isinstance(result, dict):
        return result
    try:
        body, _status, _ = result
        return body if isinstance(body, dict) else {}
    except (TypeError, ValueError):
        return {}


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
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
    """Mock HTTP request handler for BaseHandler.read_json_body."""

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


class MockHTTPHandlerBadJSON:
    """Mock HTTP handler that returns unparseable JSON."""

    def __init__(self):
        self.rfile = MagicMock()
        self.rfile.read.return_value = b"{{NOT JSON}}"
        self.headers = {
            "Content-Length": "12",
            "Content-Type": "application/json",
        }
        self.client_address = ("127.0.0.1", 54321)


class MockHTTPHandlerEmptyBody:
    """Mock HTTP handler with Content-Length 0."""

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
    """Default GATEWAY_AVAILABLE to True for all tests."""
    with patch("aragora.server.handlers.gateway_agents_handler.GATEWAY_AVAILABLE", True):
        yield


@pytest.fixture(autouse=True)
def _reset_cb():
    """Reset circuit breaker state between tests."""
    reset_gateway_agents_circuit_breaker()
    yield
    reset_gateway_agents_circuit_breaker()


@pytest.fixture
def handler():
    """Fresh handler with empty context."""
    return GatewayAgentsHandler(ctx={})


@pytest.fixture
def handler_with_agents():
    """Handler pre-loaded with two registered agents."""
    return GatewayAgentsHandler(
        ctx={
            "external_agents": {
                "alpha": {
                    "name": "alpha",
                    "framework_type": "crewai",
                    "base_url": "https://alpha.example.com",
                    "timeout": 30,
                    "config": {"env": "prod"},
                },
                "beta": {
                    "name": "beta",
                    "framework_type": "langgraph",
                    "base_url": "https://beta.example.com",
                    "timeout": 60,
                    "config": {},
                },
            }
        }
    )


# ===========================================================================
# ROUTES class attribute
# ===========================================================================


class TestRoutes:
    """Verify ROUTES lists the expected URL patterns."""

    def test_routes_length(self):
        assert len(GatewayAgentsHandler.ROUTES) == 2

    def test_routes_contains_base(self):
        assert "/api/v1/gateway/agents" in GatewayAgentsHandler.ROUTES

    def test_routes_contains_wildcard(self):
        assert "/api/v1/gateway/agents/*" in GatewayAgentsHandler.ROUTES


# ===========================================================================
# can_handle
# ===========================================================================


class TestCanHandle:
    """Test path routing via can_handle()."""

    def test_base_path(self, handler):
        assert handler.can_handle("/api/v1/gateway/agents") is True

    def test_named_agent_path(self, handler):
        assert handler.can_handle("/api/v1/gateway/agents/my-agent") is True

    def test_trailing_slash(self, handler):
        assert handler.can_handle("/api/v1/gateway/agents/") is True

    def test_deep_nested(self, handler):
        assert handler.can_handle("/api/v1/gateway/agents/name/sub") is True

    def test_rejects_debates(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_rejects_root(self, handler):
        assert handler.can_handle("/") is False

    def test_rejects_partial_match(self, handler):
        assert handler.can_handle("/api/v1/gateway/age") is False

    def test_rejects_gateway_other(self, handler):
        assert handler.can_handle("/api/v1/gateway/health") is False

    def test_case_sensitive(self, handler):
        assert handler.can_handle("/api/v1/gateway/Agents") is False


# ===========================================================================
# _extract_agent_name
# ===========================================================================


class TestExtractAgentName:
    """Test path segment extraction."""

    def test_extracts_name(self, handler):
        assert handler._extract_agent_name("/api/v1/gateway/agents/myagent") == "myagent"

    def test_trailing_slash_stripped(self, handler):
        assert handler._extract_agent_name("/api/v1/gateway/agents/myagent/") == "myagent"

    def test_base_path_returns_none(self, handler):
        assert handler._extract_agent_name("/api/v1/gateway/agents") is None

    def test_name_equals_agents_returns_none(self, handler):
        assert handler._extract_agent_name("/api/v1/gateway/agents/agents") is None

    def test_hyphens_and_underscores(self, handler):
        assert handler._extract_agent_name("/api/v1/gateway/agents/my_agent-v2") == "my_agent-v2"

    def test_short_path_returns_none(self, handler):
        assert handler._extract_agent_name("/api/v1/gateway") is None

    def test_empty_path_returns_none(self, handler):
        assert handler._extract_agent_name("") is None

    def test_extra_segments_ignored(self, handler):
        # parts[5] is still the agent name
        assert handler._extract_agent_name("/api/v1/gateway/agents/foo/bar") == "foo"


# ===========================================================================
# _get_external_agents
# ===========================================================================


class TestGetExternalAgents:
    """Test the external agents registry helper."""

    def test_creates_dict_if_missing(self, handler):
        assert "external_agents" not in handler.ctx
        agents = handler._get_external_agents()
        assert isinstance(agents, dict)
        assert handler.ctx["external_agents"] is agents

    def test_returns_existing_dict(self, handler_with_agents):
        agents = handler_with_agents._get_external_agents()
        assert "alpha" in agents
        assert "beta" in agents

    def test_mutations_are_persistent(self, handler):
        agents = handler._get_external_agents()
        agents["new"] = {"name": "new"}
        assert "new" in handler.ctx["external_agents"]


# ===========================================================================
# _validate_base_url
# ===========================================================================


class TestValidateBaseUrl:
    """Test base_url validation logic."""

    def test_https_accepted(self, handler):
        assert handler._validate_base_url("https://example.com") is None

    def test_http_rejected_by_default(self, handler):
        result = handler._validate_base_url("http://example.com")
        assert result is not None
        assert _status(result) == 400
        assert "HTTPS" in _body(result).get("error", "")

    def test_http_accepted_when_insecure_true(self, handler):
        with patch.dict("os.environ", {"ARAGORA_ALLOW_INSECURE_AGENTS": "true"}):
            assert handler._validate_base_url("http://example.com") is None

    def test_http_accepted_when_insecure_1(self, handler):
        with patch.dict("os.environ", {"ARAGORA_ALLOW_INSECURE_AGENTS": "1"}):
            assert handler._validate_base_url("http://example.com") is None

    def test_http_accepted_when_insecure_yes(self, handler):
        with patch.dict("os.environ", {"ARAGORA_ALLOW_INSECURE_AGENTS": "yes"}):
            assert handler._validate_base_url("http://example.com") is None

    def test_ftp_rejected_strict_mode(self, handler):
        result = handler._validate_base_url("ftp://example.com")
        assert result is not None
        assert _status(result) == 400

    def test_ftp_rejected_even_insecure(self, handler):
        with patch.dict("os.environ", {"ARAGORA_ALLOW_INSECURE_AGENTS": "true"}):
            result = handler._validate_base_url("ftp://example.com")
        assert result is not None
        assert _status(result) == 400

    def test_empty_url_rejected(self, handler):
        result = handler._validate_base_url("")
        assert result is not None
        assert _status(result) == 400

    def test_insecure_flag_false_blocks_http(self, handler):
        with patch.dict("os.environ", {"ARAGORA_ALLOW_INSECURE_AGENTS": "false"}):
            result = handler._validate_base_url("http://example.com")
        assert _status(result) == 400

    def test_insecure_flag_empty_blocks_http(self, handler):
        with patch.dict("os.environ", {"ARAGORA_ALLOW_INSECURE_AGENTS": ""}):
            result = handler._validate_base_url("http://example.com")
        assert _status(result) == 400

    def test_insecure_TRUE_uppercase(self, handler):
        with patch.dict("os.environ", {"ARAGORA_ALLOW_INSECURE_AGENTS": "TRUE"}):
            assert handler._validate_base_url("http://example.com") is None

    def test_ssrf_blocks_unsafe_url(self, handler):
        mock_result = MagicMock()
        mock_result.is_safe = False
        mock_result.error = "Internal IP"
        with (
            patch("aragora.server.handlers.gateway_agents_handler.SSRF_AVAILABLE", True),
            patch(
                "aragora.security.ssrf_protection.validate_url",
                return_value=mock_result,
            ),
        ):
            result = handler._validate_base_url("https://internal.corp")
        assert _status(result) == 400
        assert "security validation" in _body(result).get("error", "").lower()

    def test_ssrf_allows_safe_url(self, handler):
        mock_result = MagicMock()
        mock_result.is_safe = True
        with (
            patch("aragora.server.handlers.gateway_agents_handler.SSRF_AVAILABLE", True),
            patch(
                "aragora.security.ssrf_protection.validate_url",
                return_value=mock_result,
            ),
        ):
            assert handler._validate_base_url("https://safe.example.com") is None


# ===========================================================================
# GET /api/v1/gateway/agents  (list)
# ===========================================================================


class TestListAgents:
    """Test GET list agents endpoint."""

    def test_list_empty(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["agents"] == []
        assert body["total"] == 0

    def test_list_populated(self, handler_with_agents):
        http = MockHTTPHandler()
        result = handler_with_agents.handle("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 2
        names = {a["name"] for a in body["agents"]}
        assert names == {"alpha", "beta"}

    def test_list_agent_fields(self, handler_with_agents):
        http = MockHTTPHandler()
        result = handler_with_agents.handle("/api/v1/gateway/agents", {}, http)
        body = _body(result)
        for agent in body["agents"]:
            assert "name" in agent
            assert "framework_type" in agent
            assert "base_url" in agent
            assert "timeout" in agent
            assert agent["status"] == "registered"

    def test_list_trailing_slash(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/agents/", {}, http)
        assert _status(result) == 200

    def test_list_default_framework(self, handler):
        handler.ctx["external_agents"] = {"bare": {"name": "bare"}}
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/agents", {}, http)
        body = _body(result)
        assert body["agents"][0]["framework_type"] == "custom"

    def test_list_default_timeout(self, handler):
        handler.ctx["external_agents"] = {"bare": {"name": "bare"}}
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/agents", {}, http)
        body = _body(result)
        assert body["agents"][0]["timeout"] == DEFAULT_TIMEOUT

    def test_non_matching_returns_none(self, handler):
        http = MockHTTPHandler()
        assert handler.handle("/api/v1/debates", {}, http) is None


# ===========================================================================
# GET /api/v1/gateway/agents/{name}  (get single)
# ===========================================================================


class TestGetAgent:
    """Test GET single agent endpoint."""

    def test_agent_found(self, handler_with_agents):
        http = MockHTTPHandler()
        result = handler_with_agents.handle("/api/v1/gateway/agents/alpha", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["name"] == "alpha"
        assert body["framework_type"] == "crewai"
        assert body["base_url"] == "https://alpha.example.com"
        assert body["timeout"] == 30
        assert body["status"] == "registered"
        assert body["config"] == {"env": "prod"}

    def test_agent_not_found(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/agents/nonexistent", {}, http)
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    def test_agent_defaults(self, handler):
        handler.ctx["external_agents"] = {"min": {"name": "min"}}
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/agents/min", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["framework_type"] == "custom"
        assert body["base_url"] == ""
        assert body["timeout"] == DEFAULT_TIMEOUT
        assert body["config"] == {}

    def test_get_with_extra_segments(self, handler_with_agents):
        """Extra path segments after agent name are ignored; name is parts[5]."""
        http = MockHTTPHandler()
        result = handler_with_agents.handle("/api/v1/gateway/agents/alpha/extra", {}, http)
        assert _status(result) == 200
        assert _body(result)["name"] == "alpha"


# ===========================================================================
# POST /api/v1/gateway/agents  (register)
# ===========================================================================


class TestRegisterAgent:
    """Test POST register agent endpoint."""

    def test_register_success(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "new-agent",
                "framework_type": "crewai",
                "base_url": "https://new.example.com",
            }
        )
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 201
        body = _body(result)
        assert body["name"] == "new-agent"
        assert body["framework_type"] == "crewai"
        assert body["base_url"] == "https://new.example.com"
        assert body["registered"] is True
        assert "successfully" in body["message"].lower()

    def test_register_stored_in_ctx(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "stored",
                "framework_type": "autogen",
                "base_url": "https://stored.example.com",
            }
        )
        handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert "stored" in handler.ctx["external_agents"]
        assert handler.ctx["external_agents"]["stored"]["framework_type"] == "autogen"

    def test_register_optional_fields(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "full",
                "framework_type": "langgraph",
                "base_url": "https://full.example.com",
                "timeout": 120,
                "config": {"model": "gpt-4"},
                "api_key_env": "MY_KEY",
            }
        )
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 201
        info = handler.ctx["external_agents"]["full"]
        assert info["timeout"] == 120
        assert info["config"] == {"model": "gpt-4"}
        assert info["api_key_env"] == "MY_KEY"

    def test_register_default_timeout(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "dtimeout",
                "framework_type": "custom",
                "base_url": "https://dt.example.com",
            }
        )
        handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert handler.ctx["external_agents"]["dtimeout"]["timeout"] == DEFAULT_TIMEOUT

    def test_register_no_api_key_env(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "nokey",
                "framework_type": "custom",
                "base_url": "https://nokey.example.com",
            }
        )
        handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert "api_key_env" not in handler.ctx["external_agents"]["nokey"]

    def test_register_missing_name(self, handler):
        http = MockHTTPHandler(
            body={
                "framework_type": "crewai",
                "base_url": "https://example.com",
            }
        )
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400
        assert "name" in _body(result).get("error", "").lower()

    def test_register_empty_name(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "",
                "framework_type": "crewai",
                "base_url": "https://example.com",
            }
        )
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400

    def test_register_missing_base_url(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "agent",
                "framework_type": "crewai",
            }
        )
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400
        assert "base_url" in _body(result).get("error", "").lower()

    def test_register_empty_base_url(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "agent",
                "framework_type": "crewai",
                "base_url": "",
            }
        )
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400

    def test_register_missing_framework_type(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "agent",
                "base_url": "https://example.com",
            }
        )
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400
        assert "framework_type" in _body(result).get("error", "").lower()

    def test_register_empty_framework_type(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "agent",
                "framework_type": "",
                "base_url": "https://example.com",
            }
        )
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400

    def test_register_invalid_framework_type(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "agent",
                "framework_type": "unknown_framework",
                "base_url": "https://example.com",
            }
        )
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400
        assert "Invalid framework_type" in _body(result).get("error", "")

    def test_register_all_valid_frameworks(self, handler):
        for fw in sorted(ALLOWED_FRAMEWORKS):
            h = GatewayAgentsHandler(ctx={})
            http = MockHTTPHandler(
                body={
                    "name": f"agent-{fw}",
                    "framework_type": fw,
                    "base_url": f"https://{fw}.example.com",
                }
            )
            with patch(
                "aragora.server.handlers.gateway_agents_handler.GATEWAY_AVAILABLE",
                True,
            ):
                result = h.handle_post("/api/v1/gateway/agents", {}, http)
            assert _status(result) == 201, f"Failed for framework: {fw}"

    def test_register_invalid_json(self, handler):
        http = MockHTTPHandlerBadJSON()
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400

    def test_register_empty_body(self, handler):
        http = MockHTTPHandlerEmptyBody()
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400

    def test_register_duplicate(self, handler_with_agents):
        http = MockHTTPHandler(
            body={
                "name": "alpha",
                "framework_type": "custom",
                "base_url": "https://dup.example.com",
            }
        )
        result = handler_with_agents.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 409
        assert "already exists" in _body(result).get("error", "").lower()

    def test_post_non_matching_path_returns_none(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "agent",
                "framework_type": "custom",
                "base_url": "https://example.com",
            }
        )
        assert handler.handle_post("/api/v1/debates", {}, http) is None

    def test_post_on_named_path_returns_none(self, handler):
        """POST on /agents/{name} should return None (only base path accepts POST)."""
        http = MockHTTPHandler(
            body={
                "name": "agent",
                "framework_type": "custom",
                "base_url": "https://example.com",
            }
        )
        result = handler.handle_post("/api/v1/gateway/agents/some-name", {}, http)
        assert result is None


# ===========================================================================
# Agent name pattern validation
# ===========================================================================


class TestAgentNameValidation:
    """Test AGENT_NAME_PATTERN used during registration."""

    def test_alphanumeric(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "agent123",
                "framework_type": "custom",
                "base_url": "https://example.com",
            }
        )
        assert _status(handler.handle_post("/api/v1/gateway/agents", {}, http)) == 201

    def test_hyphens(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "my-agent-v2",
                "framework_type": "custom",
                "base_url": "https://example.com",
            }
        )
        assert _status(handler.handle_post("/api/v1/gateway/agents", {}, http)) == 201

    def test_underscores(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "my_agent_v2",
                "framework_type": "custom",
                "base_url": "https://example.com",
            }
        )
        assert _status(handler.handle_post("/api/v1/gateway/agents", {}, http)) == 201

    def test_single_char(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "x",
                "framework_type": "custom",
                "base_url": "https://example.com",
            }
        )
        assert _status(handler.handle_post("/api/v1/gateway/agents", {}, http)) == 201

    def test_max_length_64(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "a" * 64,
                "framework_type": "custom",
                "base_url": "https://example.com",
            }
        )
        assert _status(handler.handle_post("/api/v1/gateway/agents", {}, http)) == 201

    def test_too_long_65(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "a" * 65,
                "framework_type": "custom",
                "base_url": "https://example.com",
            }
        )
        assert _status(handler.handle_post("/api/v1/gateway/agents", {}, http)) == 400

    def test_starts_with_hyphen(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "-bad",
                "framework_type": "custom",
                "base_url": "https://example.com",
            }
        )
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400
        assert "Invalid agent name" in _body(result).get("error", "")

    def test_starts_with_underscore(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "_bad",
                "framework_type": "custom",
                "base_url": "https://example.com",
            }
        )
        assert _status(handler.handle_post("/api/v1/gateway/agents", {}, http)) == 400

    def test_spaces_rejected(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "bad name",
                "framework_type": "custom",
                "base_url": "https://example.com",
            }
        )
        assert _status(handler.handle_post("/api/v1/gateway/agents", {}, http)) == 400

    def test_special_chars_rejected(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "agent@!#",
                "framework_type": "custom",
                "base_url": "https://example.com",
            }
        )
        assert _status(handler.handle_post("/api/v1/gateway/agents", {}, http)) == 400

    def test_pattern_regex_valid(self):
        for name in ["a", "abc", "A1", "my-agent", "my_agent", "z" * 64]:
            assert AGENT_NAME_PATTERN.match(name), f"Should match: {name}"

    def test_pattern_regex_invalid(self):
        for name in ["", "-x", "_x", "a" * 65, "bad name", ".dot"]:
            assert AGENT_NAME_PATTERN.match(name) is None, f"Should not match: {name}"


# ===========================================================================
# DELETE /api/v1/gateway/agents/{name}
# ===========================================================================


class TestDeleteAgent:
    """Test DELETE agent endpoint."""

    def test_delete_success(self, handler_with_agents):
        http = MockHTTPHandler()
        result = handler_with_agents.handle_delete("/api/v1/gateway/agents/alpha", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["name"] == "alpha"
        assert "unregistered" in body["message"].lower()
        assert "alpha" not in handler_with_agents.ctx["external_agents"]

    def test_delete_not_found(self, handler):
        http = MockHTTPHandler()
        result = handler.handle_delete("/api/v1/gateway/agents/ghost", {}, http)
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    def test_delete_no_name_returns_none(self, handler):
        http = MockHTTPHandler()
        result = handler.handle_delete("/api/v1/gateway/agents", {}, http)
        assert result is None

    def test_delete_non_matching_returns_none(self, handler):
        http = MockHTTPHandler()
        assert handler.handle_delete("/api/v1/debates/x", {}, http) is None

    def test_delete_preserves_others(self, handler_with_agents):
        http = MockHTTPHandler()
        handler_with_agents.handle_delete("/api/v1/gateway/agents/alpha", {}, http)
        assert "beta" in handler_with_agents.ctx["external_agents"]


# ===========================================================================
# Gateway unavailable (503)
# ===========================================================================


class TestGatewayUnavailable:
    """All methods should return 503 when GATEWAY_AVAILABLE is False."""

    @pytest.fixture(autouse=True)
    def _disable_gateway(self):
        with patch("aragora.server.handlers.gateway_agents_handler.GATEWAY_AVAILABLE", False):
            yield

    def test_get_list_503(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 503
        assert "not available" in _body(result).get("error", "").lower()

    def test_get_single_503(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/agents/x", {}, http)
        assert _status(result) == 503

    def test_post_503(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "a",
                "framework_type": "custom",
                "base_url": "https://example.com",
            }
        )
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 503

    def test_delete_503(self, handler):
        http = MockHTTPHandler()
        result = handler.handle_delete("/api/v1/gateway/agents/x", {}, http)
        assert _status(result) == 503


# ===========================================================================
# Unknown routes return None
# ===========================================================================


class TestUnknownRoutes:
    """Non-matching paths return None for all methods."""

    def test_handle_unknown(self, handler):
        http = MockHTTPHandler()
        assert handler.handle("/api/v1/unknown", {}, http) is None

    def test_handle_post_unknown(self, handler):
        http = MockHTTPHandler(body={})
        assert handler.handle_post("/api/v1/unknown", {}, http) is None

    def test_handle_delete_unknown(self, handler):
        http = MockHTTPHandler()
        assert handler.handle_delete("/api/v1/unknown", {}, http) is None

    def test_handle_v2_path(self, handler):
        http = MockHTTPHandler()
        assert handler.handle("/api/v2/gateway/agents", {}, http) is None


# ===========================================================================
# Circuit breaker helpers
# ===========================================================================


class TestCircuitBreakerHelpers:
    """Test module-level circuit breaker functions."""

    def test_get_returns_instance(self):
        cb = get_gateway_agents_circuit_breaker()
        assert cb is not None
        assert cb.name == "gateway_agents_handler"

    def test_status_returns_dict(self):
        status = get_gateway_agents_circuit_breaker_status()
        assert isinstance(status, dict)

    def test_reset(self):
        cb = get_gateway_agents_circuit_breaker()
        cb._single_failures = 3
        cb._single_open_at = 99.0
        cb._single_successes = 1
        cb._single_half_open_calls = 2
        reset_gateway_agents_circuit_breaker()
        assert cb._single_failures == 0
        assert cb._single_open_at == 0.0
        assert cb._single_successes == 0
        assert cb._single_half_open_calls == 0

    def test_config_thresholds(self):
        cb = get_gateway_agents_circuit_breaker()
        assert cb.failure_threshold == 5
        assert cb.cooldown_seconds == 30.0


# ===========================================================================
# Module exports (__all__)
# ===========================================================================


class TestModuleExports:
    """Verify __all__ contains the expected symbols."""

    def test_all_has_handler(self):
        from aragora.server.handlers import gateway_agents_handler as mod

        assert "GatewayAgentsHandler" in mod.__all__

    def test_all_has_get_cb(self):
        from aragora.server.handlers import gateway_agents_handler as mod

        assert "get_gateway_agents_circuit_breaker" in mod.__all__

    def test_all_has_get_cb_status(self):
        from aragora.server.handlers import gateway_agents_handler as mod

        assert "get_gateway_agents_circuit_breaker_status" in mod.__all__

    def test_all_has_reset_cb(self):
        from aragora.server.handlers import gateway_agents_handler as mod

        assert "reset_gateway_agents_circuit_breaker" in mod.__all__

    def test_all_count(self):
        from aragora.server.handlers import gateway_agents_handler as mod

        assert len(mod.__all__) == 4


# ===========================================================================
# Constants
# ===========================================================================


class TestConstants:
    """Verify module-level constants."""

    def test_default_timeout(self):
        assert DEFAULT_TIMEOUT == 30

    def test_allowed_frameworks_set(self):
        assert ALLOWED_FRAMEWORKS == {"openclaw", "crewai", "autogen", "langgraph", "custom"}


# ===========================================================================
# Handler initialization
# ===========================================================================


class TestInit:
    """Test constructor behavior."""

    def test_default_ctx_empty(self):
        h = GatewayAgentsHandler()
        assert h.ctx == {}

    def test_custom_ctx(self):
        ctx = {"key": "val"}
        h = GatewayAgentsHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_none_ctx_becomes_empty(self):
        h = GatewayAgentsHandler(ctx=None)
        assert h.ctx == {}


# ===========================================================================
# Security tests
# ===========================================================================


class TestSecurity:
    """Security-focused input validation tests."""

    def test_path_traversal_in_get(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/agents/../../etc/passwd", {}, http)
        if result is not None:
            assert _status(result) in (400, 404)

    def test_dotdot_name_rejected(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "../etc/passwd",
                "framework_type": "custom",
                "base_url": "https://example.com",
            }
        )
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400

    def test_null_byte_in_name(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "agent\x00evil",
                "framework_type": "custom",
                "base_url": "https://example.com",
            }
        )
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400

    def test_unicode_in_name(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "agent\u00e9",
                "framework_type": "custom",
                "base_url": "https://example.com",
            }
        )
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400

    def test_sql_injection_in_name(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "'; DROP TABLE agents;--",
                "framework_type": "custom",
                "base_url": "https://example.com",
            }
        )
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400

    def test_xss_in_name(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "<script>alert(1)</script>",
                "framework_type": "custom",
                "base_url": "https://example.com",
            }
        )
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400

    def test_javascript_protocol_url(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "jsagent",
                "framework_type": "custom",
                "base_url": "javascript:alert(1)",
            }
        )
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400

    def test_data_protocol_url(self, handler):
        http = MockHTTPHandler(
            body={
                "name": "dataagent",
                "framework_type": "custom",
                "base_url": "data:text/html,<h1>evil</h1>",
            }
        )
        result = handler.handle_post("/api/v1/gateway/agents", {}, http)
        assert _status(result) == 400


# ===========================================================================
# End-to-end lifecycle
# ===========================================================================


class TestEndToEndLifecycle:
    """Integration-style tests exercising the full register/list/get/delete flow."""

    def test_register_list_get_delete(self, handler):
        # Register
        reg_http = MockHTTPHandler(
            body={
                "name": "lifecycle",
                "framework_type": "openclaw",
                "base_url": "https://lifecycle.example.com",
                "timeout": 45,
                "config": {"v": "1"},
            }
        )
        assert _status(handler.handle_post("/api/v1/gateway/agents", {}, reg_http)) == 201

        # List
        list_http = MockHTTPHandler()
        list_result = handler.handle("/api/v1/gateway/agents", {}, list_http)
        assert _body(list_result)["total"] == 1
        assert _body(list_result)["agents"][0]["name"] == "lifecycle"

        # Get
        get_http = MockHTTPHandler()
        get_result = handler.handle("/api/v1/gateway/agents/lifecycle", {}, get_http)
        assert _status(get_result) == 200
        body = _body(get_result)
        assert body["framework_type"] == "openclaw"
        assert body["timeout"] == 45
        assert body["config"] == {"v": "1"}

        # Delete
        del_http = MockHTTPHandler()
        del_result = handler.handle_delete("/api/v1/gateway/agents/lifecycle", {}, del_http)
        assert _status(del_result) == 200

        # Verify gone
        list_http2 = MockHTTPHandler()
        assert _body(handler.handle("/api/v1/gateway/agents", {}, list_http2))["total"] == 0

        get_http2 = MockHTTPHandler()
        assert _status(handler.handle("/api/v1/gateway/agents/lifecycle", {}, get_http2)) == 404

    def test_register_multiple(self, handler):
        for i in range(4):
            http = MockHTTPHandler(
                body={
                    "name": f"m{i}",
                    "framework_type": "custom",
                    "base_url": f"https://m{i}.example.com",
                }
            )
            assert _status(handler.handle_post("/api/v1/gateway/agents", {}, http)) == 201

        list_http = MockHTTPHandler()
        assert _body(handler.handle("/api/v1/gateway/agents", {}, list_http))["total"] == 4

    def test_delete_and_reregister(self, handler):
        # Register
        http1 = MockHTTPHandler(
            body={
                "name": "reuse",
                "framework_type": "custom",
                "base_url": "https://reuse.example.com",
            }
        )
        handler.handle_post("/api/v1/gateway/agents", {}, http1)

        # Delete
        del_http = MockHTTPHandler()
        handler.handle_delete("/api/v1/gateway/agents/reuse", {}, del_http)

        # Re-register with different framework
        http2 = MockHTTPHandler(
            body={
                "name": "reuse",
                "framework_type": "crewai",
                "base_url": "https://reuse-v2.example.com",
            }
        )
        result = handler.handle_post("/api/v1/gateway/agents", {}, http2)
        assert _status(result) == 201

        get_http = MockHTTPHandler()
        get_result = handler.handle("/api/v1/gateway/agents/reuse", {}, get_http)
        assert _body(get_result)["framework_type"] == "crewai"
        assert _body(get_result)["base_url"] == "https://reuse-v2.example.com"

    def test_duplicate_rejected_after_register(self, handler):
        http1 = MockHTTPHandler(
            body={
                "name": "uniq",
                "framework_type": "crewai",
                "base_url": "https://uniq.example.com",
            }
        )
        assert _status(handler.handle_post("/api/v1/gateway/agents", {}, http1)) == 201

        http2 = MockHTTPHandler(
            body={
                "name": "uniq",
                "framework_type": "autogen",
                "base_url": "https://other.example.com",
            }
        )
        assert _status(handler.handle_post("/api/v1/gateway/agents", {}, http2)) == 409
