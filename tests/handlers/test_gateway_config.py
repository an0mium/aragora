"""Tests for gateway configuration handler.

Covers all routes and behavior of the GatewayConfigHandler class:
- can_handle() routing for matched and non-matched paths
- GET    /api/v1/gateway/config          - Get current gateway configuration
- POST   /api/v1/gateway/config          - Update gateway configuration
- GET    /api/v1/gateway/config/defaults - Get default configuration values
- Validation of all 10 config keys (boundaries, types)
- Unknown keys are silently ignored
- No-change POST returns empty changes list
- Partial updates preserve existing config values
- Invalid JSON body handling
- Multiple simultaneous config changes
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from aragora.server.handlers.gateway_config_handler import (
    CONFIG_VALIDATION_MESSAGES,
    CONFIG_VALIDATORS,
    GatewayConfigHandler,
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
    """Mock HTTP handler with no body content (Content-Length: 0)."""

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


@pytest.fixture
def handler():
    """Create a GatewayConfigHandler with empty server context."""
    return GatewayConfigHandler(server_context={})


@pytest.fixture
def handler_with_config():
    """Create a handler with pre-loaded custom config."""
    ctx = {
        "gateway_config": {
            "agent_timeout": 60,
            "max_concurrent_agents": 20,
            "consensus_threshold": 0.8,
            "min_verification_quorum": 3,
            "rate_limit_requests_per_minute": 100,
            "credential_cache_ttl_seconds": 600,
            "circuit_breaker_failure_threshold": 10,
            "circuit_breaker_recovery_timeout": 120,
            "allow_http_agents": True,
            "require_ssrf_validation": False,
        },
        "gateway_config_updated_at": "2026-01-15T10:00:00+00:00",
    }
    return GatewayConfigHandler(server_context=ctx)


# ===========================================================================
# can_handle routing tests
# ===========================================================================


class TestCanHandle:
    """Test the can_handle path routing."""

    def test_handles_config_path(self, handler):
        assert handler.can_handle("/api/v1/gateway/config") is True

    def test_handles_defaults_path(self, handler):
        assert handler.can_handle("/api/v1/gateway/config/defaults") is True

    def test_rejects_non_matching_path(self, handler):
        assert handler.can_handle("/api/v1/gateway/devices") is False

    def test_rejects_root(self, handler):
        assert handler.can_handle("/") is False

    def test_rejects_partial_prefix(self, handler):
        assert handler.can_handle("/api/v1/gateway/con") is False

    def test_rejects_debates_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_rejects_extra_suffix(self, handler):
        assert handler.can_handle("/api/v1/gateway/config/defaults/extra") is False

    def test_rejects_config_subpath(self, handler):
        assert handler.can_handle("/api/v1/gateway/config/something") is False

    def test_rejects_empty_string(self, handler):
        assert handler.can_handle("") is False


# ===========================================================================
# GET /api/v1/gateway/config
# ===========================================================================


class TestGetConfig:
    """Test GET /api/v1/gateway/config."""

    def test_returns_default_config_on_fresh_handler(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/config", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert "config" in body
        assert "updated_at" in body
        config = body["config"]
        assert config["agent_timeout"] == 30
        assert config["max_concurrent_agents"] == 10
        assert config["consensus_threshold"] == 0.7
        assert config["min_verification_quorum"] == 2
        assert config["rate_limit_requests_per_minute"] == 60
        assert config["credential_cache_ttl_seconds"] == 300
        assert config["circuit_breaker_failure_threshold"] == 5
        assert config["circuit_breaker_recovery_timeout"] == 60
        assert config["allow_http_agents"] is False
        assert config["require_ssrf_validation"] is True

    def test_returns_custom_config(self, handler_with_config):
        http = MockHTTPHandler()
        result = handler_with_config.handle("/api/v1/gateway/config", {}, http)
        assert _status(result) == 200
        body = _body(result)
        config = body["config"]
        assert config["agent_timeout"] == 60
        assert config["max_concurrent_agents"] == 20
        assert config["consensus_threshold"] == 0.8
        assert config["allow_http_agents"] is True
        assert config["require_ssrf_validation"] is False

    def test_returns_updated_at_timestamp(self, handler_with_config):
        http = MockHTTPHandler()
        result = handler_with_config.handle("/api/v1/gateway/config", {}, http)
        body = _body(result)
        assert body["updated_at"] == "2026-01-15T10:00:00+00:00"

    def test_config_has_all_default_keys(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/config", {}, http)
        body = _body(result)
        config = body["config"]
        for key in GatewayConfigHandler.DEFAULT_CONFIG:
            assert key in config, f"Missing key: {key}"

    def test_returns_none_for_unhandled_path(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/other", {}, http)
        assert result is None

    def test_query_params_are_ignored(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/config", {"foo": "bar", "limit": "10"}, http)
        assert _status(result) == 200


# ===========================================================================
# GET /api/v1/gateway/config/defaults
# ===========================================================================


class TestGetDefaults:
    """Test GET /api/v1/gateway/config/defaults."""

    def test_returns_defaults(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/config/defaults", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert "defaults" in body
        defaults = body["defaults"]
        assert defaults == GatewayConfigHandler.DEFAULT_CONFIG

    def test_defaults_not_affected_by_config_changes(self, handler_with_config):
        """Defaults should always return the class-level DEFAULT_CONFIG, not current config."""
        http = MockHTTPHandler()
        result = handler_with_config.handle("/api/v1/gateway/config/defaults", {}, http)
        assert _status(result) == 200
        body = _body(result)
        defaults = body["defaults"]
        # Defaults should be the class-level defaults, not the custom config
        assert defaults["agent_timeout"] == 30
        assert defaults["max_concurrent_agents"] == 10

    def test_defaults_contain_all_keys(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/config/defaults", {}, http)
        body = _body(result)
        defaults = body["defaults"]
        assert len(defaults) == 10
        expected_keys = {
            "agent_timeout",
            "max_concurrent_agents",
            "consensus_threshold",
            "min_verification_quorum",
            "rate_limit_requests_per_minute",
            "credential_cache_ttl_seconds",
            "circuit_breaker_failure_threshold",
            "circuit_breaker_recovery_timeout",
            "allow_http_agents",
            "require_ssrf_validation",
        }
        assert set(defaults.keys()) == expected_keys


# ===========================================================================
# POST /api/v1/gateway/config - Happy path updates
# ===========================================================================


class TestUpdateConfigHappyPath:
    """Test POST /api/v1/gateway/config - valid updates."""

    def test_update_single_int_field(self, handler):
        http = MockHTTPHandler(body={"agent_timeout": 60})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["config"]["agent_timeout"] == 60
        assert body["message"] == "Configuration updated successfully"
        assert len(body["changes"]) == 1
        assert "agent_timeout" in body["changes"][0]

    def test_update_single_float_field(self, handler):
        http = MockHTTPHandler(body={"consensus_threshold": 0.9})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["config"]["consensus_threshold"] == 0.9

    def test_update_boolean_field_true(self, handler):
        http = MockHTTPHandler(body={"allow_http_agents": True})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["config"]["allow_http_agents"] is True
        assert len(body["changes"]) == 1

    def test_update_boolean_field_false(self, handler):
        """require_ssrf_validation defaults to True; set to False."""
        http = MockHTTPHandler(body={"require_ssrf_validation": False})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["config"]["require_ssrf_validation"] is False

    def test_update_multiple_fields(self, handler):
        http = MockHTTPHandler(
            body={
                "agent_timeout": 120,
                "max_concurrent_agents": 50,
                "consensus_threshold": 0.85,
            }
        )
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["config"]["agent_timeout"] == 120
        assert body["config"]["max_concurrent_agents"] == 50
        assert body["config"]["consensus_threshold"] == 0.85
        assert len(body["changes"]) == 3

    def test_update_all_fields(self, handler):
        new_config = {
            "agent_timeout": 100,
            "max_concurrent_agents": 50,
            "consensus_threshold": 0.5,
            "min_verification_quorum": 5,
            "rate_limit_requests_per_minute": 200,
            "credential_cache_ttl_seconds": 900,
            "circuit_breaker_failure_threshold": 10,
            "circuit_breaker_recovery_timeout": 300,
            "allow_http_agents": True,
            "require_ssrf_validation": False,
        }
        http = MockHTTPHandler(body=new_config)
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 200
        body = _body(result)
        for key, value in new_config.items():
            assert body["config"][key] == value
        assert len(body["changes"]) == 10

    def test_response_includes_updated_at(self, handler):
        http = MockHTTPHandler(body={"agent_timeout": 45})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert "updated_at" in body
        assert isinstance(body["updated_at"], str)
        assert len(body["updated_at"]) > 0

    def test_config_persists_across_reads(self, handler):
        """Updated config should be returned by subsequent GET."""
        # POST to update
        http_post = MockHTTPHandler(body={"agent_timeout": 99})
        result = handler.handle_post("/api/v1/gateway/config", {}, http_post)
        assert _status(result) == 200

        # GET to verify persistence
        http_get = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/config", {}, http_get)
        assert _status(result) == 200
        body = _body(result)
        assert body["config"]["agent_timeout"] == 99

    def test_unchanged_fields_preserved(self, handler):
        """Fields not included in POST should retain their prior values."""
        http = MockHTTPHandler(body={"agent_timeout": 45})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 200
        body = _body(result)
        # Other fields should remain default
        assert body["config"]["max_concurrent_agents"] == 10
        assert body["config"]["consensus_threshold"] == 0.7
        assert body["config"]["require_ssrf_validation"] is True


# ===========================================================================
# POST /api/v1/gateway/config - No-change and empty body
# ===========================================================================


class TestUpdateConfigNoChange:
    """Test POST /api/v1/gateway/config with no effective changes."""

    def test_same_value_results_in_no_changes(self, handler):
        """Posting the same value as default should produce empty changes list."""
        http = MockHTTPHandler(body={"agent_timeout": 30})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["changes"] == []
        assert body["config"]["agent_timeout"] == 30

    def test_empty_body_produces_no_changes(self, handler):
        http = MockHTTPHandler(body={})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["changes"] == []

    def test_no_body_content_length_zero(self, handler):
        """Content-Length: 0 means read_json_body returns {} (empty dict)."""
        http = MockHTTPHandlerNoBody()
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["changes"] == []


# ===========================================================================
# POST /api/v1/gateway/config - Unknown keys
# ===========================================================================


class TestUpdateConfigUnknownKeys:
    """Test that unknown config keys are silently ignored."""

    def test_unknown_key_only(self, handler):
        http = MockHTTPHandler(body={"nonexistent_key": "value"})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["changes"] == []
        assert "nonexistent_key" not in body["config"]

    def test_mixed_known_and_unknown_keys(self, handler):
        http = MockHTTPHandler(body={"agent_timeout": 45, "unknown_field": True})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["config"]["agent_timeout"] == 45
        assert "unknown_field" not in body["config"]
        assert len(body["changes"]) == 1

    def test_multiple_unknown_keys(self, handler):
        http = MockHTTPHandler(body={"foo": 1, "bar": "baz", "qux": True})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["changes"] == []


# ===========================================================================
# POST /api/v1/gateway/config - Validation errors
# ===========================================================================


class TestUpdateConfigValidation:
    """Test config value validation."""

    # --- agent_timeout ---
    def test_agent_timeout_below_min(self, handler):
        http = MockHTTPHandler(body={"agent_timeout": 0})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400
        body = _body(result)
        assert "agent_timeout" in body.get("error", "")

    def test_agent_timeout_above_max(self, handler):
        http = MockHTTPHandler(body={"agent_timeout": 301})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    def test_agent_timeout_wrong_type(self, handler):
        http = MockHTTPHandler(body={"agent_timeout": "fast"})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    def test_agent_timeout_at_min_boundary(self, handler):
        http = MockHTTPHandler(body={"agent_timeout": 1})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 200

    def test_agent_timeout_at_max_boundary(self, handler):
        http = MockHTTPHandler(body={"agent_timeout": 300})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 200

    def test_agent_timeout_float_valid(self, handler):
        http = MockHTTPHandler(body={"agent_timeout": 15.5})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 200

    def test_agent_timeout_negative(self, handler):
        http = MockHTTPHandler(body={"agent_timeout": -1})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    # --- max_concurrent_agents ---
    def test_max_concurrent_agents_below_min(self, handler):
        http = MockHTTPHandler(body={"max_concurrent_agents": 0})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    def test_max_concurrent_agents_above_max(self, handler):
        http = MockHTTPHandler(body={"max_concurrent_agents": 101})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    def test_max_concurrent_agents_float_rejected(self, handler):
        """max_concurrent_agents requires int, not float."""
        http = MockHTTPHandler(body={"max_concurrent_agents": 10.5})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    def test_max_concurrent_agents_at_boundaries(self, handler):
        for val in [1, 100]:
            http = MockHTTPHandler(body={"max_concurrent_agents": val})
            result = handler.handle_post("/api/v1/gateway/config", {}, http)
            assert _status(result) == 200, f"Failed for value {val}"

    # --- consensus_threshold ---
    def test_consensus_threshold_below_min(self, handler):
        http = MockHTTPHandler(body={"consensus_threshold": -0.1})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    def test_consensus_threshold_above_max(self, handler):
        http = MockHTTPHandler(body={"consensus_threshold": 1.1})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    def test_consensus_threshold_at_boundaries(self, handler):
        for val in [0.0, 1.0]:
            http = MockHTTPHandler(body={"consensus_threshold": val})
            result = handler.handle_post("/api/v1/gateway/config", {}, http)
            assert _status(result) == 200, f"Failed for value {val}"

    def test_consensus_threshold_string_rejected(self, handler):
        http = MockHTTPHandler(body={"consensus_threshold": "high"})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    # --- min_verification_quorum ---
    def test_min_verification_quorum_below_min(self, handler):
        http = MockHTTPHandler(body={"min_verification_quorum": 0})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    def test_min_verification_quorum_above_max(self, handler):
        http = MockHTTPHandler(body={"min_verification_quorum": 11})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    def test_min_verification_quorum_at_boundaries(self, handler):
        for val in [1, 10]:
            http = MockHTTPHandler(body={"min_verification_quorum": val})
            result = handler.handle_post("/api/v1/gateway/config", {}, http)
            assert _status(result) == 200, f"Failed for value {val}"

    def test_min_verification_quorum_float_rejected(self, handler):
        http = MockHTTPHandler(body={"min_verification_quorum": 2.5})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    # --- rate_limit_requests_per_minute ---
    def test_rate_limit_rpm_below_min(self, handler):
        http = MockHTTPHandler(body={"rate_limit_requests_per_minute": 0})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    def test_rate_limit_rpm_above_max(self, handler):
        http = MockHTTPHandler(body={"rate_limit_requests_per_minute": 1001})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    def test_rate_limit_rpm_at_boundaries(self, handler):
        for val in [1, 1000]:
            http = MockHTTPHandler(body={"rate_limit_requests_per_minute": val})
            result = handler.handle_post("/api/v1/gateway/config", {}, http)
            assert _status(result) == 200, f"Failed for value {val}"

    # --- credential_cache_ttl_seconds ---
    def test_credential_cache_ttl_below_min(self, handler):
        http = MockHTTPHandler(body={"credential_cache_ttl_seconds": -1})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    def test_credential_cache_ttl_above_max(self, handler):
        http = MockHTTPHandler(body={"credential_cache_ttl_seconds": 3601})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    def test_credential_cache_ttl_at_boundaries(self, handler):
        for val in [0, 3600]:
            http = MockHTTPHandler(body={"credential_cache_ttl_seconds": val})
            result = handler.handle_post("/api/v1/gateway/config", {}, http)
            assert _status(result) == 200, f"Failed for value {val}"

    # --- circuit_breaker_failure_threshold ---
    def test_cb_failure_threshold_below_min(self, handler):
        http = MockHTTPHandler(body={"circuit_breaker_failure_threshold": 0})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    def test_cb_failure_threshold_above_max(self, handler):
        http = MockHTTPHandler(body={"circuit_breaker_failure_threshold": 21})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    def test_cb_failure_threshold_at_boundaries(self, handler):
        for val in [1, 20]:
            http = MockHTTPHandler(body={"circuit_breaker_failure_threshold": val})
            result = handler.handle_post("/api/v1/gateway/config", {}, http)
            assert _status(result) == 200, f"Failed for value {val}"

    # --- circuit_breaker_recovery_timeout ---
    def test_cb_recovery_timeout_below_min(self, handler):
        http = MockHTTPHandler(body={"circuit_breaker_recovery_timeout": 0})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    def test_cb_recovery_timeout_above_max(self, handler):
        http = MockHTTPHandler(body={"circuit_breaker_recovery_timeout": 601})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    def test_cb_recovery_timeout_at_boundaries(self, handler):
        for val in [1, 600]:
            http = MockHTTPHandler(body={"circuit_breaker_recovery_timeout": val})
            result = handler.handle_post("/api/v1/gateway/config", {}, http)
            assert _status(result) == 200, f"Failed for value {val}"

    # --- allow_http_agents (boolean) ---
    def test_allow_http_agents_string_rejected(self, handler):
        http = MockHTTPHandler(body={"allow_http_agents": "true"})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    def test_allow_http_agents_int_rejected(self, handler):
        http = MockHTTPHandler(body={"allow_http_agents": 1})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    def test_allow_http_agents_null_rejected(self, handler):
        http = MockHTTPHandler(body={"allow_http_agents": None})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    # --- require_ssrf_validation (boolean) ---
    def test_require_ssrf_validation_string_rejected(self, handler):
        http = MockHTTPHandler(body={"require_ssrf_validation": "false"})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    def test_require_ssrf_validation_int_rejected(self, handler):
        http = MockHTTPHandler(body={"require_ssrf_validation": 0})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    # --- Validation stops at first invalid field ---
    def test_validation_stops_on_first_invalid(self, handler):
        """If an invalid field is encountered, the response is 400 immediately."""
        http = MockHTTPHandler(body={"agent_timeout": -1, "max_concurrent_agents": 50})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400

    # --- Error messages ---
    def test_error_message_for_agent_timeout(self, handler):
        http = MockHTTPHandler(body={"agent_timeout": 999})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400
        body = _body(result)
        assert "agent_timeout" in body.get("error", "")


# ===========================================================================
# POST /api/v1/gateway/config - Invalid JSON
# ===========================================================================


class TestUpdateConfigInvalidJSON:
    """Test POST with invalid JSON body."""

    def test_invalid_json_returns_400(self, handler):
        http = MockHTTPHandlerInvalidJSON()
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 400
        body = _body(result)
        assert "error" in body


# ===========================================================================
# POST /api/v1/gateway/config - Path routing
# ===========================================================================


class TestHandlePostRouting:
    """Test handle_post only accepts the config path."""

    def test_post_returns_none_for_defaults_path(self, handler):
        http = MockHTTPHandler(body={})
        result = handler.handle_post("/api/v1/gateway/config/defaults", {}, http)
        assert result is None

    def test_post_returns_none_for_unrelated_path(self, handler):
        http = MockHTTPHandler(body={})
        result = handler.handle_post("/api/v1/gateway/devices", {}, http)
        assert result is None

    def test_post_returns_none_for_root(self, handler):
        http = MockHTTPHandler(body={})
        result = handler.handle_post("/", {}, http)
        assert result is None


# ===========================================================================
# Initialization and context management
# ===========================================================================


class TestInitialization:
    """Test handler initialization and server context setup."""

    def test_init_populates_gateway_config_in_context(self):
        ctx: dict[str, Any] = {}
        handler = GatewayConfigHandler(server_context=ctx)
        assert "gateway_config" in ctx
        assert ctx["gateway_config"] == GatewayConfigHandler.DEFAULT_CONFIG

    def test_init_populates_updated_at_in_context(self):
        ctx: dict[str, Any] = {}
        handler = GatewayConfigHandler(server_context=ctx)
        assert "gateway_config_updated_at" in ctx
        assert isinstance(ctx["gateway_config_updated_at"], str)

    def test_init_does_not_overwrite_existing_config(self):
        custom_config = {"agent_timeout": 99}
        ctx: dict[str, Any] = {"gateway_config": custom_config}
        handler = GatewayConfigHandler(server_context=ctx)
        assert ctx["gateway_config"] is custom_config
        assert ctx["gateway_config"]["agent_timeout"] == 99

    def test_init_does_not_overwrite_existing_updated_at(self):
        ctx: dict[str, Any] = {"gateway_config_updated_at": "2020-01-01T00:00:00+00:00"}
        handler = GatewayConfigHandler(server_context=ctx)
        assert ctx["gateway_config_updated_at"] == "2020-01-01T00:00:00+00:00"

    def test_routes_class_attribute(self):
        assert "/api/v1/gateway/config" in GatewayConfigHandler.ROUTES
        assert "/api/v1/gateway/config/defaults" in GatewayConfigHandler.ROUTES

    def test_default_config_class_attribute(self):
        dc = GatewayConfigHandler.DEFAULT_CONFIG
        assert isinstance(dc, dict)
        assert len(dc) == 10


# ===========================================================================
# Sequential operations (multi-step scenarios)
# ===========================================================================


class TestMultiStepScenarios:
    """Test sequential update and read operations."""

    def test_two_sequential_updates(self, handler):
        """First update, then a second update should stack on the first."""
        http1 = MockHTTPHandler(body={"agent_timeout": 60})
        result1 = handler.handle_post("/api/v1/gateway/config", {}, http1)
        assert _status(result1) == 200

        http2 = MockHTTPHandler(body={"max_concurrent_agents": 25})
        result2 = handler.handle_post("/api/v1/gateway/config", {}, http2)
        assert _status(result2) == 200
        body2 = _body(result2)
        # Both changes should be reflected
        assert body2["config"]["agent_timeout"] == 60
        assert body2["config"]["max_concurrent_agents"] == 25

    def test_overwrite_same_field_twice(self, handler):
        http1 = MockHTTPHandler(body={"agent_timeout": 60})
        handler.handle_post("/api/v1/gateway/config", {}, http1)

        http2 = MockHTTPHandler(body={"agent_timeout": 120})
        result = handler.handle_post("/api/v1/gateway/config", {}, http2)
        assert _status(result) == 200
        body = _body(result)
        assert body["config"]["agent_timeout"] == 120
        assert len(body["changes"]) == 1
        assert "60 -> 120" in body["changes"][0]

    def test_update_then_get_then_defaults(self, handler):
        """Full lifecycle: update, get, and defaults should all be consistent."""
        # Update
        http_post = MockHTTPHandler(body={"consensus_threshold": 0.95})
        handler.handle_post("/api/v1/gateway/config", {}, http_post)

        # Get current config
        http_get = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/config", {}, http_get)
        body = _body(result)
        assert body["config"]["consensus_threshold"] == 0.95

        # Get defaults (should be unaffected)
        http_defaults = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/config/defaults", {}, http_defaults)
        body = _body(result)
        assert body["defaults"]["consensus_threshold"] == 0.7

    def test_failed_update_does_not_change_config(self, handler):
        """An invalid update should not modify the config at all."""
        # First, confirm defaults
        http_get1 = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/config", {}, http_get1)
        original_config = _body(result)["config"].copy()

        # Try invalid update
        http_bad = MockHTTPHandler(body={"agent_timeout": -100})
        result = handler.handle_post("/api/v1/gateway/config", {}, http_bad)
        assert _status(result) == 400

        # Confirm config unchanged
        http_get2 = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/config", {}, http_get2)
        current_config = _body(result)["config"]
        assert current_config == original_config


# ===========================================================================
# CONFIG_VALIDATORS and CONFIG_VALIDATION_MESSAGES consistency
# ===========================================================================


class TestValidatorConsistency:
    """Test that validators and messages are properly defined."""

    def test_every_validator_has_a_message(self):
        for key in CONFIG_VALIDATORS:
            assert key in CONFIG_VALIDATION_MESSAGES, f"Missing validation message for: {key}"

    def test_every_message_has_a_validator(self):
        for key in CONFIG_VALIDATION_MESSAGES:
            assert key in CONFIG_VALIDATORS, f"Orphan validation message for: {key}"

    def test_validators_match_default_config_keys(self):
        assert set(CONFIG_VALIDATORS.keys()) == set(GatewayConfigHandler.DEFAULT_CONFIG.keys())

    def test_all_default_values_pass_validation(self):
        """Every default value should pass its own validator."""
        for key, value in GatewayConfigHandler.DEFAULT_CONFIG.items():
            validator = CONFIG_VALIDATORS[key]
            assert validator(value), f"Default value for {key} ({value!r}) fails its validator"


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Miscellaneous edge cases."""

    def test_consensus_threshold_integer_zero(self, handler):
        """Integer 0 should be valid for consensus_threshold (0 <= 0 <= 1.0)."""
        http = MockHTTPHandler(body={"consensus_threshold": 0})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 200

    def test_consensus_threshold_integer_one(self, handler):
        """Integer 1 should be valid for consensus_threshold (0 <= 1 <= 1.0)."""
        http = MockHTTPHandler(body={"consensus_threshold": 1})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 200

    def test_agent_timeout_float_at_boundary(self, handler):
        """Float 300.0 should be valid for agent_timeout."""
        http = MockHTTPHandler(body={"agent_timeout": 300.0})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 200

    def test_credential_cache_ttl_zero_is_valid(self, handler):
        """Zero TTL means no caching, which is valid."""
        http = MockHTTPHandler(body={"credential_cache_ttl_seconds": 0})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        assert _status(result) == 200

    def test_shared_server_context(self):
        """Two handler instances sharing the same context see the same config."""
        ctx: dict[str, Any] = {}
        h1 = GatewayConfigHandler(server_context=ctx)
        h2 = GatewayConfigHandler(server_context=ctx)

        # Update via h1
        http_post = MockHTTPHandler(body={"agent_timeout": 77})
        h1.handle_post("/api/v1/gateway/config", {}, http_post)

        # Read via h2
        http_get = MockHTTPHandler()
        result = h2.handle("/api/v1/gateway/config", {}, http_get)
        body = _body(result)
        assert body["config"]["agent_timeout"] == 77

    def test_handle_returns_none_for_unhandled_path_via_can_handle(self, handler):
        """handle() should return None for paths not in can_handle."""
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/something/else", {}, http)
        assert result is None

    def test_changes_list_format(self, handler):
        """Each change entry should show 'key: old -> new' format."""
        http = MockHTTPHandler(body={"agent_timeout": 99})
        result = handler.handle_post("/api/v1/gateway/config", {}, http)
        body = _body(result)
        assert len(body["changes"]) == 1
        change = body["changes"][0]
        assert "agent_timeout" in change
        assert "30" in change  # old value
        assert "99" in change  # new value
        assert "->" in change

    def test_post_config_path_only(self, handler):
        """handle_post returns None for defaults path."""
        http = MockHTTPHandler(body={})
        result = handler.handle_post("/api/v1/gateway/config/defaults", {}, http)
        assert result is None
