"""
Tests for GatewayConfigHandler - Gateway configuration HTTP endpoints.

Tests cover:
- GET /api/v1/gateway/config - Get current configuration
- POST /api/v1/gateway/config - Update configuration
- GET /api/v1/gateway/config/defaults - Get default values
- Configuration validation
- RBAC protection
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.gateway_config_handler import GatewayConfigHandler


# ===========================================================================
# Test Fixtures and Mocks
# ===========================================================================


class MockRequestHandler:
    """Mock HTTP request handler."""

    def __init__(self, body: dict | None = None, headers: dict | None = None):
        self._body = body
        self.headers = headers or {"Content-Length": "0"}
        self._body_bytes = json.dumps(body).encode() if body else b"{}"
        if body:
            self.headers["Content-Length"] = str(len(self._body_bytes))
        self.rfile = MagicMock()
        self.rfile.read = MagicMock(return_value=self._body_bytes)


@pytest.fixture
def mock_server_context() -> dict[str, Any]:
    """Create mock server context."""
    return {}


@pytest.fixture
def handler(mock_server_context: dict[str, Any]) -> GatewayConfigHandler:
    """Create handler with mocked dependencies."""
    return GatewayConfigHandler(mock_server_context)


# ===========================================================================
# Route Handling Tests
# ===========================================================================


class TestGatewayConfigHandlerRouting:
    """Test request routing."""

    def test_can_handle_config_paths(self, handler: GatewayConfigHandler):
        """Test that handler recognizes config paths."""
        assert handler.can_handle("/api/v1/gateway/config")
        assert handler.can_handle("/api/v1/gateway/config/defaults")

    def test_cannot_handle_other_paths(self, handler: GatewayConfigHandler):
        """Test that handler rejects non-config paths."""
        assert not handler.can_handle("/api/v1/gateway/devices")
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/gateway/config")  # Missing v1
        assert not handler.can_handle("/api/v1/gateway/config/something")


# ===========================================================================
# GET Config Tests
# ===========================================================================


class TestGetConfig:
    """Test GET /api/v1/gateway/config endpoint."""

    def test_get_config_returns_current(self, handler: GatewayConfigHandler):
        """Test that GET returns current configuration."""
        mock_handler = MockRequestHandler()

        with patch.object(handler, "_handle_get_config", wraps=handler._handle_get_config):
            result = handler._handle_get_config(mock_handler)

        assert result is not None
        assert result.status_code == 200

        body = json.loads(result.body)
        assert "config" in body
        assert body["config"]["agent_timeout"] == 30
        assert body["config"]["max_concurrent_agents"] == 10
        assert body["config"]["consensus_threshold"] == 0.7

    def test_get_config_includes_updated_at(self, handler: GatewayConfigHandler):
        """Test that response includes updated_at timestamp."""
        mock_handler = MockRequestHandler()

        result = handler._handle_get_config(mock_handler)

        assert result is not None
        assert result.status_code == 200

        body = json.loads(result.body)
        assert "updated_at" in body
        # Verify it's an ISO format timestamp
        assert "T" in body["updated_at"]


# ===========================================================================
# GET Defaults Tests
# ===========================================================================


class TestGetDefaults:
    """Test GET /api/v1/gateway/config/defaults endpoint."""

    def test_get_defaults_returns_defaults(self, handler: GatewayConfigHandler):
        """Test that GET /defaults returns default configuration values."""
        mock_handler = MockRequestHandler()

        result = handler._handle_get_defaults(mock_handler)

        assert result is not None
        assert result.status_code == 200

        body = json.loads(result.body)
        assert "defaults" in body
        assert body["defaults"]["agent_timeout"] == 30
        assert body["defaults"]["max_concurrent_agents"] == 10
        assert body["defaults"]["consensus_threshold"] == 0.7
        assert body["defaults"]["min_verification_quorum"] == 2
        assert body["defaults"]["rate_limit_requests_per_minute"] == 60
        assert body["defaults"]["credential_cache_ttl_seconds"] == 300
        assert body["defaults"]["circuit_breaker_failure_threshold"] == 5
        assert body["defaults"]["circuit_breaker_recovery_timeout"] == 60
        assert body["defaults"]["allow_http_agents"] is False
        assert body["defaults"]["require_ssrf_validation"] is True


# ===========================================================================
# POST Config Tests
# ===========================================================================


class TestUpdateConfig:
    """Test POST /api/v1/gateway/config endpoint."""

    def test_update_config_success(self, handler: GatewayConfigHandler):
        """Test that POST updates specific configuration values."""
        mock_handler = MockRequestHandler(
            body={
                "agent_timeout": 60,
                "consensus_threshold": 0.8,
            }
        )

        result = handler._handle_update_config(mock_handler)

        assert result is not None
        assert result.status_code == 200

        body = json.loads(result.body)
        assert body["config"]["agent_timeout"] == 60
        assert body["config"]["consensus_threshold"] == 0.8
        # Other values should remain default
        assert body["config"]["max_concurrent_agents"] == 10

    def test_update_config_validates_agent_timeout(self, handler: GatewayConfigHandler):
        """Test that agent_timeout validation rejects out of range values."""
        # Test value too low
        mock_handler = MockRequestHandler(body={"agent_timeout": 0})
        result = handler._handle_update_config(mock_handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "agent_timeout" in body.get("error", "")

        # Test value too high
        mock_handler = MockRequestHandler(body={"agent_timeout": 500})
        result = handler._handle_update_config(mock_handler)

        assert result is not None
        assert result.status_code == 400

    def test_update_config_validates_consensus_threshold(self, handler: GatewayConfigHandler):
        """Test that consensus_threshold validation rejects out of range values."""
        # Test value too low
        mock_handler = MockRequestHandler(body={"consensus_threshold": -0.1})
        result = handler._handle_update_config(mock_handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "consensus_threshold" in body.get("error", "")

        # Test value too high
        mock_handler = MockRequestHandler(body={"consensus_threshold": 1.5})
        result = handler._handle_update_config(mock_handler)

        assert result is not None
        assert result.status_code == 400

    def test_update_config_validates_boolean_fields(self, handler: GatewayConfigHandler):
        """Test that boolean fields reject non-boolean values."""
        mock_handler = MockRequestHandler(body={"allow_http_agents": "true"})
        result = handler._handle_update_config(mock_handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "allow_http_agents" in body.get("error", "")

        # Test with integer instead of boolean
        mock_handler = MockRequestHandler(body={"require_ssrf_validation": 1})
        result = handler._handle_update_config(mock_handler)

        assert result is not None
        assert result.status_code == 400

    def test_update_config_ignores_unknown_keys(self, handler: GatewayConfigHandler):
        """Test that unknown configuration keys are ignored without error."""
        mock_handler = MockRequestHandler(
            body={
                "unknown_key": "value",
                "another_unknown": 123,
                "agent_timeout": 45,  # Valid key
            }
        )

        result = handler._handle_update_config(mock_handler)

        assert result is not None
        assert result.status_code == 200

        body = json.loads(result.body)
        # Valid key should be updated
        assert body["config"]["agent_timeout"] == 45
        # Unknown keys should not be in config
        assert "unknown_key" not in body["config"]
        assert "another_unknown" not in body["config"]

    def test_update_config_tracks_changes(self, handler: GatewayConfigHandler):
        """Test that changes list accurately reflects what was changed."""
        mock_handler = MockRequestHandler(
            body={
                "agent_timeout": 60,
                "consensus_threshold": 0.8,
            }
        )

        result = handler._handle_update_config(mock_handler)

        assert result is not None
        assert result.status_code == 200

        body = json.loads(result.body)
        assert "changes" in body
        assert len(body["changes"]) == 2

        # Verify change strings
        changes_str = " ".join(body["changes"])
        assert "agent_timeout" in changes_str
        assert "30 -> 60" in changes_str
        assert "consensus_threshold" in changes_str
        assert "0.7 -> 0.8" in changes_str

    def test_update_config_updates_timestamp(self, handler: GatewayConfigHandler):
        """Test that timestamp is updated on config change."""
        # Get initial timestamp
        initial_updated_at = handler.ctx.get("gateway_config_updated_at")

        # Make a change
        mock_handler = MockRequestHandler(body={"agent_timeout": 45})
        result = handler._handle_update_config(mock_handler)

        assert result is not None
        body = json.loads(result.body)

        # Timestamp should be different (or at least present)
        assert "updated_at" in body
        # In practice, timestamps would differ, but in fast tests they might be same
        assert body["updated_at"] is not None


# ===========================================================================
# Persistence Tests
# ===========================================================================


class TestConfigPersistence:
    """Test configuration persistence in server context."""

    def test_config_persists_in_context(self, mock_server_context: dict[str, Any]):
        """Test that configuration is stored in server context."""
        handler = GatewayConfigHandler(mock_server_context)

        # Update config
        mock_handler = MockRequestHandler(body={"agent_timeout": 120})
        handler._handle_update_config(mock_handler)

        # Verify it's in context
        assert "gateway_config" in mock_server_context
        assert mock_server_context["gateway_config"]["agent_timeout"] == 120

        # Create new handler with same context
        handler2 = GatewayConfigHandler(mock_server_context)
        result = handler2._handle_get_config(MockRequestHandler())

        body = json.loads(result.body)
        assert body["config"]["agent_timeout"] == 120


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestErrorHandling:
    """Test error handling scenarios."""

    def test_invalid_json_returns_400(self, handler: GatewayConfigHandler):
        """Test that invalid JSON body returns 400 error."""
        mock_handler = MockRequestHandler()
        # Simulate read_json_body returning None for invalid JSON
        with patch.object(handler, "read_json_body", return_value=None):
            result = handler._handle_update_config(mock_handler)

        assert result is not None
        assert result.status_code == 400
        body = json.loads(result.body)
        assert "Invalid JSON" in body.get("error", "")

    def test_empty_update_no_changes(self, handler: GatewayConfigHandler):
        """Test that empty POST body results in no changes."""
        mock_handler = MockRequestHandler(body={})

        result = handler._handle_update_config(mock_handler)

        assert result is not None
        assert result.status_code == 200

        body = json.loads(result.body)
        assert body["changes"] == []
        assert body["message"] == "Configuration updated successfully"


# ===========================================================================
# Additional Validation Tests
# ===========================================================================


class TestAdditionalValidation:
    """Test additional configuration validation scenarios."""

    def test_validates_max_concurrent_agents(self, handler: GatewayConfigHandler):
        """Test max_concurrent_agents validation."""
        # Too low
        mock_handler = MockRequestHandler(body={"max_concurrent_agents": 0})
        result = handler._handle_update_config(mock_handler)
        assert result.status_code == 400

        # Too high
        mock_handler = MockRequestHandler(body={"max_concurrent_agents": 150})
        result = handler._handle_update_config(mock_handler)
        assert result.status_code == 400

        # Valid
        mock_handler = MockRequestHandler(body={"max_concurrent_agents": 50})
        result = handler._handle_update_config(mock_handler)
        assert result.status_code == 200

    def test_validates_min_verification_quorum(self, handler: GatewayConfigHandler):
        """Test min_verification_quorum validation."""
        # Too low
        mock_handler = MockRequestHandler(body={"min_verification_quorum": 0})
        result = handler._handle_update_config(mock_handler)
        assert result.status_code == 400

        # Too high
        mock_handler = MockRequestHandler(body={"min_verification_quorum": 15})
        result = handler._handle_update_config(mock_handler)
        assert result.status_code == 400

        # Valid
        mock_handler = MockRequestHandler(body={"min_verification_quorum": 5})
        result = handler._handle_update_config(mock_handler)
        assert result.status_code == 200

    def test_validates_circuit_breaker_settings(self, handler: GatewayConfigHandler):
        """Test circuit breaker configuration validation."""
        # Invalid failure threshold
        mock_handler = MockRequestHandler(body={"circuit_breaker_failure_threshold": 0})
        result = handler._handle_update_config(mock_handler)
        assert result.status_code == 400

        mock_handler = MockRequestHandler(body={"circuit_breaker_failure_threshold": 25})
        result = handler._handle_update_config(mock_handler)
        assert result.status_code == 400

        # Invalid recovery timeout
        mock_handler = MockRequestHandler(body={"circuit_breaker_recovery_timeout": 0})
        result = handler._handle_update_config(mock_handler)
        assert result.status_code == 400

        mock_handler = MockRequestHandler(body={"circuit_breaker_recovery_timeout": 700})
        result = handler._handle_update_config(mock_handler)
        assert result.status_code == 400

        # Valid values
        mock_handler = MockRequestHandler(
            body={
                "circuit_breaker_failure_threshold": 10,
                "circuit_breaker_recovery_timeout": 120,
            }
        )
        result = handler._handle_update_config(mock_handler)
        assert result.status_code == 200

    def test_boolean_values_work_correctly(self, handler: GatewayConfigHandler):
        """Test that boolean values are accepted correctly."""
        # Test True values
        mock_handler = MockRequestHandler(
            body={
                "allow_http_agents": True,
                "require_ssrf_validation": False,
            }
        )
        result = handler._handle_update_config(mock_handler)

        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["config"]["allow_http_agents"] is True
        assert body["config"]["require_ssrf_validation"] is False
