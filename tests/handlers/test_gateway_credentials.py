"""Tests for gateway credentials handler.

Covers all routes and behavior of the GatewayCredentialsHandler class:
- can_handle() routing for all ROUTES and non-matching paths
- POST   /api/v1/gateway/credentials              - Store a new credential
- GET    /api/v1/gateway/credentials               - List credentials (metadata only)
- GET    /api/v1/gateway/credentials/{id}          - Get credential metadata
- DELETE /api/v1/gateway/credentials/{id}          - Delete a credential
- POST   /api/v1/gateway/credentials/{id}/rotate   - Rotate a credential
- Input validation (missing fields, invalid types, bad service names)
- Credential proxy integration (store, delete, rotate)
- Proxy failure handling
- Security: credential values are NEVER returned in responses
- Circuit breaker helper functions
- Module exports
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.gateway_credentials_handler import (
    GatewayCredentialsHandler,
    SERVICE_NAME_PATTERN,
    VALID_CREDENTIAL_TYPES,
    get_gateway_credentials_circuit_breaker,
    get_gateway_credentials_circuit_breaker_status,
    reset_gateway_credentials_circuit_breaker,
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


@pytest.fixture
def handler():
    """Create a GatewayCredentialsHandler instance with empty server context."""
    return GatewayCredentialsHandler(server_context={})


@pytest.fixture
def mock_proxy():
    """Create a mock CredentialProxy."""
    proxy = MagicMock()
    proxy.store = MagicMock()
    proxy.delete = MagicMock()
    return proxy


@pytest.fixture
def handler_with_proxy(handler, mock_proxy):
    """Handler with credential proxy available in server context."""
    handler.ctx = {"credential_proxy": mock_proxy}
    return handler


@pytest.fixture
def handler_with_credential(handler):
    """Handler with a pre-stored credential in memory."""
    handler._credentials["cred_abc123"] = {
        "credential_id": "cred_abc123",
        "service_name": "my-service",
        "credential_type": "api_key",
        "status": "active",
        "created_at": "2026-01-01T00:00:00+00:00",
        "tenant_id": "tenant-1",
        "scopes": ["read", "write"],
        "expires_at": "2027-01-01T00:00:00+00:00",
        "metadata": {"env": "production"},
    }
    return handler


@pytest.fixture
def handler_with_proxy_and_credential(handler_with_proxy):
    """Handler with proxy AND a pre-stored credential."""
    handler_with_proxy._credentials["cred_abc123"] = {
        "credential_id": "cred_abc123",
        "service_name": "my-service",
        "credential_type": "api_key",
        "status": "active",
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    return handler_with_proxy


@pytest.fixture(autouse=True)
def _reset_circuit_breaker():
    """Reset circuit breaker state between tests."""
    reset_gateway_credentials_circuit_breaker()
    yield
    reset_gateway_credentials_circuit_breaker()


# ===========================================================================
# can_handle routing tests
# ===========================================================================


class TestCanHandle:
    """Test the can_handle path routing."""

    def test_handles_credentials_base(self, handler):
        assert handler.can_handle("/api/v1/gateway/credentials") is True

    def test_handles_credentials_with_id(self, handler):
        assert handler.can_handle("/api/v1/gateway/credentials/cred_abc123") is True

    def test_handles_credentials_rotate(self, handler):
        assert handler.can_handle("/api/v1/gateway/credentials/cred_abc123/rotate") is True

    def test_rejects_non_gateway_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_rejects_root(self, handler):
        assert handler.can_handle("/") is False

    def test_rejects_partial_prefix(self, handler):
        assert handler.can_handle("/api/v1/gateway/cred") is False

    def test_rejects_gateway_devices(self, handler):
        assert handler.can_handle("/api/v1/gateway/devices") is False

    def test_handles_trailing_slash(self, handler):
        assert handler.can_handle("/api/v1/gateway/credentials/") is True


# ===========================================================================
# ROUTES class attribute
# ===========================================================================


class TestRoutesAttribute:
    """Verify the ROUTES class attribute lists expected patterns."""

    def test_routes_contains_credentials_base(self):
        assert "/api/v1/gateway/credentials" in GatewayCredentialsHandler.ROUTES

    def test_routes_contains_credentials_wildcard(self):
        assert "/api/v1/gateway/credentials/*" in GatewayCredentialsHandler.ROUTES

    def test_routes_count(self):
        assert len(GatewayCredentialsHandler.ROUTES) == 2


# ===========================================================================
# Path extraction helpers
# ===========================================================================


class TestPathExtraction:
    """Test _extract_credential_id and _is_rotate_path."""

    def test_extract_id_from_valid_path(self, handler):
        assert (
            handler._extract_credential_id("/api/v1/gateway/credentials/cred_abc123")
            == "cred_abc123"
        )

    def test_extract_id_with_trailing_slash(self, handler):
        assert (
            handler._extract_credential_id("/api/v1/gateway/credentials/cred_abc123/")
            == "cred_abc123"
        )

    def test_extract_id_from_rotate_path(self, handler):
        assert (
            handler._extract_credential_id("/api/v1/gateway/credentials/cred_abc123/rotate")
            == "cred_abc123"
        )

    def test_extract_id_from_base_path_returns_none(self, handler):
        assert handler._extract_credential_id("/api/v1/gateway/credentials") is None

    def test_extract_id_empty_segment_returns_none(self, handler):
        # Path with trailing slash and no actual id
        assert handler._extract_credential_id("/api/v1/gateway/credentials/") is None

    def test_is_rotate_path_true(self, handler):
        assert handler._is_rotate_path("/api/v1/gateway/credentials/cred_abc123/rotate") is True

    def test_is_rotate_path_false_no_rotate(self, handler):
        assert handler._is_rotate_path("/api/v1/gateway/credentials/cred_abc123") is False

    def test_is_rotate_path_false_base(self, handler):
        assert handler._is_rotate_path("/api/v1/gateway/credentials") is False

    def test_is_rotate_path_with_trailing_slash(self, handler):
        assert handler._is_rotate_path("/api/v1/gateway/credentials/cred_abc123/rotate/") is True


# ===========================================================================
# Validation helpers
# ===========================================================================


class TestValidation:
    """Test _validate_service_name and _validate_credential_type."""

    def test_validate_service_name_valid(self, handler):
        assert handler._validate_service_name("my-service") is None

    def test_validate_service_name_alphanumeric(self, handler):
        assert handler._validate_service_name("MyService123") is None

    def test_validate_service_name_empty(self, handler):
        assert handler._validate_service_name("") == "service_name is required"

    def test_validate_service_name_starts_with_hyphen(self, handler):
        result = handler._validate_service_name("-bad-name")
        assert result is not None
        assert "Invalid service_name" in result

    def test_validate_service_name_special_chars(self, handler):
        result = handler._validate_service_name("bad name!")
        assert result is not None

    def test_validate_service_name_too_long(self, handler):
        result = handler._validate_service_name("a" * 129)
        assert result is not None

    def test_validate_service_name_max_length(self, handler):
        assert handler._validate_service_name("a" * 128) is None

    def test_validate_credential_type_api_key(self, handler):
        assert handler._validate_credential_type("api_key") is None

    def test_validate_credential_type_oauth_token(self, handler):
        assert handler._validate_credential_type("oauth_token") is None

    def test_validate_credential_type_bearer_token(self, handler):
        assert handler._validate_credential_type("bearer_token") is None

    def test_validate_credential_type_basic_auth(self, handler):
        assert handler._validate_credential_type("basic_auth") is None

    def test_validate_credential_type_custom(self, handler):
        assert handler._validate_credential_type("custom") is None

    def test_validate_credential_type_empty(self, handler):
        assert handler._validate_credential_type("") == "credential_type is required"

    def test_validate_credential_type_invalid(self, handler):
        result = handler._validate_credential_type("password")
        assert result is not None
        assert "Invalid credential_type" in result

    def test_valid_credential_types_constant(self):
        assert VALID_CREDENTIAL_TYPES == {
            "api_key",
            "oauth_token",
            "bearer_token",
            "basic_auth",
            "custom",
        }

    def test_service_name_pattern_rejects_unicode(self, handler):
        result = handler._validate_service_name("service\u00e9")
        assert result is not None


# ===========================================================================
# POST /api/v1/gateway/credentials (store)
# ===========================================================================


class TestStoreCredential:
    """Test POST /api/v1/gateway/credentials."""

    def test_store_credential_success(self, handler):
        http = MockHTTPHandler(
            body={
                "service_name": "my-service",
                "credential_type": "api_key",
                "value": "sk-abc123secret",
            }
        )
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 201
        body = _body(result)
        assert body["credential_id"].startswith("cred_")
        assert body["service_name"] == "my-service"
        assert body["credential_type"] == "api_key"
        assert body["message"] == "Credential stored successfully"
        assert "created_at" in body
        # CRITICAL: value must NEVER be in response
        assert "value" not in body
        assert "sk-abc123secret" not in json.dumps(body)

    def test_store_credential_with_optional_fields(self, handler):
        http = MockHTTPHandler(
            body={
                "service_name": "my-service",
                "credential_type": "oauth_token",
                "value": "oauth-token-value",
                "tenant_id": "tenant-1",
                "scopes": ["read", "write"],
                "expires_at": "2027-01-01T00:00:00+00:00",
                "metadata": {"env": "staging"},
            }
        )
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 201
        body = _body(result)
        assert body["tenant_id"] == "tenant-1"
        assert body["scopes"] == ["read", "write"]
        assert body["expires_at"] == "2027-01-01T00:00:00+00:00"
        # value must NEVER be in response
        assert "value" not in body

    def test_store_credential_stores_in_memory(self, handler):
        http = MockHTTPHandler(
            body={
                "service_name": "svc-a",
                "credential_type": "api_key",
                "value": "secret",
            }
        )
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 201
        cred_id = _body(result)["credential_id"]
        assert cred_id in handler._credentials
        meta = handler._credentials[cred_id]
        assert meta["service_name"] == "svc-a"
        assert meta["credential_type"] == "api_key"
        # value must NEVER be in memory metadata
        assert "value" not in meta

    def test_store_credential_via_proxy(self, handler_with_proxy, mock_proxy):
        http = MockHTTPHandler(
            body={
                "service_name": "svc-proxy",
                "credential_type": "bearer_token",
                "value": "my-bearer-token",
            }
        )
        result = handler_with_proxy.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 201
        mock_proxy.store.assert_called_once()
        call_kwargs = mock_proxy.store.call_args
        assert (
            call_kwargs[1]["service_name"] == "svc-proxy"
            or call_kwargs.kwargs.get("service_name") == "svc-proxy"
        )

    def test_store_credential_proxy_failure(self, handler_with_proxy, mock_proxy):
        mock_proxy.store.side_effect = RuntimeError("Connection refused")
        http = MockHTTPHandler(
            body={
                "service_name": "svc-fail",
                "credential_type": "api_key",
                "value": "my-secret",
            }
        )
        result = handler_with_proxy.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 500
        assert "Failed to store credential" in _body(result).get("error", "")

    def test_store_credential_proxy_os_error(self, handler_with_proxy, mock_proxy):
        mock_proxy.store.side_effect = OSError("Disk full")
        http = MockHTTPHandler(
            body={
                "service_name": "svc-fail",
                "credential_type": "api_key",
                "value": "my-secret",
            }
        )
        result = handler_with_proxy.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 500

    def test_store_credential_missing_service_name(self, handler):
        http = MockHTTPHandler(
            body={
                "credential_type": "api_key",
                "value": "secret",
            }
        )
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 400
        assert "service_name" in _body(result).get("error", "").lower()

    def test_store_credential_missing_credential_type(self, handler):
        http = MockHTTPHandler(
            body={
                "service_name": "svc-a",
                "value": "secret",
            }
        )
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 400
        assert "credential_type" in _body(result).get("error", "").lower()

    def test_store_credential_missing_value(self, handler):
        http = MockHTTPHandler(
            body={
                "service_name": "svc-a",
                "credential_type": "api_key",
            }
        )
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 400
        assert "value" in _body(result).get("error", "").lower()

    def test_store_credential_empty_value(self, handler):
        http = MockHTTPHandler(
            body={
                "service_name": "svc-a",
                "credential_type": "api_key",
                "value": "",
            }
        )
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 400

    def test_store_credential_whitespace_value(self, handler):
        http = MockHTTPHandler(
            body={
                "service_name": "svc-a",
                "credential_type": "api_key",
                "value": "   ",
            }
        )
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 400

    def test_store_credential_non_string_value(self, handler):
        http = MockHTTPHandler(
            body={
                "service_name": "svc-a",
                "credential_type": "api_key",
                "value": 12345,
            }
        )
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 400

    def test_store_credential_invalid_service_name(self, handler):
        http = MockHTTPHandler(
            body={
                "service_name": "-bad-name",
                "credential_type": "api_key",
                "value": "secret",
            }
        )
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 400
        assert "Invalid service_name" in _body(result).get("error", "")

    def test_store_credential_invalid_credential_type(self, handler):
        http = MockHTTPHandler(
            body={
                "service_name": "svc-a",
                "credential_type": "password",
                "value": "secret",
            }
        )
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 400
        assert "Invalid credential_type" in _body(result).get("error", "")

    def test_store_credential_invalid_json(self, handler):
        http = MockHTTPHandlerInvalidJSON()
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 400

    def test_store_credential_empty_body(self, handler):
        http = MockHTTPHandlerNoBody()
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 400

    def test_store_credential_non_matching_path_returns_none(self, handler):
        http = MockHTTPHandler(
            body={
                "service_name": "svc-a",
                "credential_type": "api_key",
                "value": "secret",
            }
        )
        result = handler.handle_post("/api/v1/debates", {}, http)
        assert result is None


# ===========================================================================
# GET /api/v1/gateway/credentials (list)
# ===========================================================================


class TestListCredentials:
    """Test GET /api/v1/gateway/credentials."""

    def test_list_credentials_empty(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["credentials"] == []
        assert body["total"] == 0

    def test_list_credentials_returns_metadata(self, handler_with_credential):
        http = MockHTTPHandler()
        result = handler_with_credential.handle("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        cred = body["credentials"][0]
        assert cred["credential_id"] == "cred_abc123"
        assert cred["service_name"] == "my-service"
        assert cred["credential_type"] == "api_key"
        assert cred["status"] == "active"
        assert cred["created_at"] == "2026-01-01T00:00:00+00:00"
        # Optional fields in list response
        assert cred["tenant_id"] == "tenant-1"
        assert cred["expires_at"] == "2027-01-01T00:00:00+00:00"
        # CRITICAL: value must NEVER appear
        assert "value" not in cred

    def test_list_credentials_multiple(self, handler):
        for i in range(3):
            handler._credentials[f"cred_{i}"] = {
                "credential_id": f"cred_{i}",
                "service_name": f"svc-{i}",
                "credential_type": "api_key",
                "status": "active",
                "created_at": "2026-01-01T00:00:00+00:00",
            }
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 3
        assert len(body["credentials"]) == 3

    def test_list_does_not_return_scopes_or_metadata(self, handler_with_credential):
        """List endpoint should NOT include scopes or metadata (only get does)."""
        http = MockHTTPHandler()
        result = handler_with_credential.handle("/api/v1/gateway/credentials", {}, http)
        body = _body(result)
        cred = body["credentials"][0]
        # scopes and metadata are NOT included in list
        assert "scopes" not in cred
        assert "metadata" not in cred

    def test_list_non_matching_path_returns_none(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/debates", {}, http)
        assert result is None


# ===========================================================================
# GET /api/v1/gateway/credentials/{id} (get single)
# ===========================================================================


class TestGetCredential:
    """Test GET /api/v1/gateway/credentials/{id}."""

    def test_get_credential_found(self, handler_with_credential):
        http = MockHTTPHandler()
        result = handler_with_credential.handle("/api/v1/gateway/credentials/cred_abc123", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["credential_id"] == "cred_abc123"
        assert body["service_name"] == "my-service"
        assert body["credential_type"] == "api_key"
        assert body["status"] == "active"
        assert body["created_at"] == "2026-01-01T00:00:00+00:00"
        assert body["tenant_id"] == "tenant-1"
        assert body["scopes"] == ["read", "write"]
        assert body["expires_at"] == "2027-01-01T00:00:00+00:00"
        assert body["metadata"] == {"env": "production"}
        # CRITICAL: value must NEVER appear
        assert "value" not in body

    def test_get_credential_not_found(self, handler):
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/credentials/cred_nonexist", {}, http)
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    def test_get_credential_minimal_metadata(self, handler):
        """Credential with only required fields."""
        handler._credentials["cred_min"] = {
            "credential_id": "cred_min",
            "service_name": "svc-min",
            "credential_type": "custom",
            "status": "active",
            "created_at": "2026-02-01T00:00:00+00:00",
        }
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/credentials/cred_min", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["credential_id"] == "cred_min"
        assert "tenant_id" not in body
        assert "scopes" not in body
        assert "expires_at" not in body
        assert "metadata" not in body


# ===========================================================================
# DELETE /api/v1/gateway/credentials/{id}
# ===========================================================================


class TestDeleteCredential:
    """Test DELETE /api/v1/gateway/credentials/{id}."""

    def test_delete_credential_success(self, handler_with_credential):
        http = MockHTTPHandler()
        result = handler_with_credential.handle_delete(
            "/api/v1/gateway/credentials/cred_abc123", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["credential_id"] == "cred_abc123"
        assert body["message"] == "Credential deleted successfully"
        # Verify removed from memory
        assert "cred_abc123" not in handler_with_credential._credentials

    def test_delete_credential_not_found(self, handler):
        http = MockHTTPHandler()
        result = handler.handle_delete("/api/v1/gateway/credentials/cred_nonexist", {}, http)
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    def test_delete_credential_via_proxy(self, handler_with_proxy_and_credential, mock_proxy):
        http = MockHTTPHandler()
        result = handler_with_proxy_and_credential.handle_delete(
            "/api/v1/gateway/credentials/cred_abc123", {}, http
        )
        assert _status(result) == 200
        mock_proxy.delete.assert_called_once_with("cred_abc123")

    def test_delete_credential_proxy_failure_still_deletes(
        self, handler_with_proxy_and_credential, mock_proxy
    ):
        """Proxy failure during delete still removes from in-memory store."""
        mock_proxy.delete.side_effect = RuntimeError("Connection lost")
        http = MockHTTPHandler()
        result = handler_with_proxy_and_credential.handle_delete(
            "/api/v1/gateway/credentials/cred_abc123", {}, http
        )
        # Delete still succeeds (proxy error is logged but not fatal)
        assert _status(result) == 200
        assert "cred_abc123" not in handler_with_proxy_and_credential._credentials

    def test_delete_credential_proxy_os_error(self, handler_with_proxy_and_credential, mock_proxy):
        mock_proxy.delete.side_effect = OSError("Disk error")
        http = MockHTTPHandler()
        result = handler_with_proxy_and_credential.handle_delete(
            "/api/v1/gateway/credentials/cred_abc123", {}, http
        )
        assert _status(result) == 200

    def test_delete_no_id_returns_none(self, handler):
        http = MockHTTPHandler()
        result = handler.handle_delete("/api/v1/gateway/credentials", {}, http)
        assert result is None

    def test_delete_non_matching_path_returns_none(self, handler):
        http = MockHTTPHandler()
        result = handler.handle_delete("/api/v1/debates", {}, http)
        assert result is None


# ===========================================================================
# POST /api/v1/gateway/credentials/{id}/rotate
# ===========================================================================


class TestRotateCredential:
    """Test POST /api/v1/gateway/credentials/{id}/rotate."""

    def test_rotate_credential_success(self, handler_with_credential):
        http = MockHTTPHandler(body={"value": "new-secret-value"})
        result = handler_with_credential.handle_post(
            "/api/v1/gateway/credentials/cred_abc123/rotate", {}, http
        )
        assert _status(result) == 201
        body = _body(result)
        assert body["credential_id"].startswith("cred_")
        assert body["credential_id"] != "cred_abc123"
        assert body["previous_credential_id"] == "cred_abc123"
        assert body["service_name"] == "my-service"
        assert body["credential_type"] == "api_key"
        assert body["status"] == "active"
        assert body["message"] == "Credential rotated successfully"
        assert "created_at" in body
        # CRITICAL: value must NEVER appear
        assert "value" not in body or body.get("value") is None
        assert "new-secret-value" not in json.dumps(body)

    def test_rotate_marks_old_as_rotated(self, handler_with_credential):
        http = MockHTTPHandler(body={"value": "new-secret"})
        handler_with_credential.handle_post(
            "/api/v1/gateway/credentials/cred_abc123/rotate", {}, http
        )
        old_meta = handler_with_credential._credentials["cred_abc123"]
        assert old_meta["status"] == "rotated"

    def test_rotate_creates_new_credential_in_memory(self, handler_with_credential):
        http = MockHTTPHandler(body={"value": "new-secret"})
        result = handler_with_credential.handle_post(
            "/api/v1/gateway/credentials/cred_abc123/rotate", {}, http
        )
        new_id = _body(result)["credential_id"]
        assert new_id in handler_with_credential._credentials
        new_meta = handler_with_credential._credentials[new_id]
        assert new_meta["service_name"] == "my-service"
        assert new_meta["credential_type"] == "api_key"
        # value must NEVER be in memory
        assert "value" not in new_meta

    def test_rotate_credential_not_found(self, handler):
        http = MockHTTPHandler(body={"value": "new-value"})
        result = handler.handle_post("/api/v1/gateway/credentials/cred_nonexist/rotate", {}, http)
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    def test_rotate_missing_value(self, handler_with_credential):
        http = MockHTTPHandler(body={})
        result = handler_with_credential.handle_post(
            "/api/v1/gateway/credentials/cred_abc123/rotate", {}, http
        )
        assert _status(result) == 400
        assert "value" in _body(result).get("error", "").lower()

    def test_rotate_empty_value(self, handler_with_credential):
        http = MockHTTPHandler(body={"value": ""})
        result = handler_with_credential.handle_post(
            "/api/v1/gateway/credentials/cred_abc123/rotate", {}, http
        )
        assert _status(result) == 400

    def test_rotate_whitespace_value(self, handler_with_credential):
        http = MockHTTPHandler(body={"value": "   "})
        result = handler_with_credential.handle_post(
            "/api/v1/gateway/credentials/cred_abc123/rotate", {}, http
        )
        assert _status(result) == 400

    def test_rotate_non_string_value(self, handler_with_credential):
        http = MockHTTPHandler(body={"value": 999})
        result = handler_with_credential.handle_post(
            "/api/v1/gateway/credentials/cred_abc123/rotate", {}, http
        )
        assert _status(result) == 400

    def test_rotate_invalid_json(self, handler_with_credential):
        http = MockHTTPHandlerInvalidJSON()
        result = handler_with_credential.handle_post(
            "/api/v1/gateway/credentials/cred_abc123/rotate", {}, http
        )
        assert _status(result) == 400

    def test_rotate_via_proxy(self, handler_with_proxy_and_credential, mock_proxy):
        http = MockHTTPHandler(body={"value": "rotated-secret"})
        result = handler_with_proxy_and_credential.handle_post(
            "/api/v1/gateway/credentials/cred_abc123/rotate", {}, http
        )
        assert _status(result) == 201
        mock_proxy.store.assert_called_once()

    def test_rotate_proxy_failure_reverts_status(
        self, handler_with_proxy_and_credential, mock_proxy
    ):
        """When proxy fails during rotation, old credential status is reverted."""
        mock_proxy.store.side_effect = RuntimeError("Storage failure")
        http = MockHTTPHandler(body={"value": "rotated-secret"})
        result = handler_with_proxy_and_credential.handle_post(
            "/api/v1/gateway/credentials/cred_abc123/rotate", {}, http
        )
        assert _status(result) == 500
        assert "Failed to rotate credential" in _body(result).get("error", "")
        # Old credential should have status reverted back to active
        old_meta = handler_with_proxy_and_credential._credentials["cred_abc123"]
        assert old_meta["status"] == "active"

    def test_rotate_proxy_type_error(self, handler_with_proxy_and_credential, mock_proxy):
        mock_proxy.store.side_effect = TypeError("Bad type")
        http = MockHTTPHandler(body={"value": "new-val"})
        result = handler_with_proxy_and_credential.handle_post(
            "/api/v1/gateway/credentials/cred_abc123/rotate", {}, http
        )
        assert _status(result) == 500
        # Status should be reverted
        old_meta = handler_with_proxy_and_credential._credentials["cred_abc123"]
        assert old_meta["status"] == "active"

    def test_rotate_preserves_optional_fields(self, handler_with_credential):
        """Rotation should carry over tenant_id, scopes, expires_at, metadata."""
        http = MockHTTPHandler(body={"value": "new-secret"})
        result = handler_with_credential.handle_post(
            "/api/v1/gateway/credentials/cred_abc123/rotate", {}, http
        )
        new_id = _body(result)["credential_id"]
        new_meta = handler_with_credential._credentials[new_id]
        assert new_meta.get("tenant_id") == "tenant-1"
        assert new_meta.get("scopes") == ["read", "write"]
        assert new_meta.get("expires_at") == "2027-01-01T00:00:00+00:00"
        assert new_meta.get("metadata") == {"env": "production"}


# ===========================================================================
# Unknown routes return None
# ===========================================================================


class TestUnknownRoutes:
    """Test that unrecognized paths return None."""

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

    def test_handle_get_credentials_with_extra_segments(self, handler):
        """GET on /api/v1/gateway/credentials/{id}/extra returns None (not a valid route)."""
        http = MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/credentials/cred_abc/extra/segment", {}, http)
        # The id extracted would be "cred_abc", so it should try get_credential
        # which returns 404 since it's not in memory
        assert result is not None  # It does try to handle it
        assert _status(result) == 404


# ===========================================================================
# Credential metadata construction
# ===========================================================================


class TestMakeCredentialMetadata:
    """Test _make_credential_metadata helper."""

    def test_basic_metadata(self, handler):
        meta = handler._make_credential_metadata(
            credential_id="cred_test",
            service_name="svc",
            credential_type="api_key",
        )
        assert meta["credential_id"] == "cred_test"
        assert meta["service_name"] == "svc"
        assert meta["credential_type"] == "api_key"
        assert meta["status"] == "active"
        assert "created_at" in meta
        # No optional fields
        assert "tenant_id" not in meta
        assert "scopes" not in meta
        assert "expires_at" not in meta
        assert "metadata" not in meta

    def test_metadata_with_all_optional_fields(self, handler):
        meta = handler._make_credential_metadata(
            credential_id="cred_full",
            service_name="svc",
            credential_type="bearer_token",
            tenant_id="t-1",
            scopes=["read"],
            expires_at="2027-01-01T00:00:00Z",
            metadata={"key": "val"},
            status="rotated",
        )
        assert meta["tenant_id"] == "t-1"
        assert meta["scopes"] == ["read"]
        assert meta["expires_at"] == "2027-01-01T00:00:00Z"
        assert meta["metadata"] == {"key": "val"}
        assert meta["status"] == "rotated"

    def test_metadata_never_contains_value(self, handler):
        """The value field must NEVER appear in metadata."""
        meta = handler._make_credential_metadata(
            credential_id="cred_val",
            service_name="svc",
            credential_type="api_key",
        )
        assert "value" not in meta


# ===========================================================================
# Circuit breaker helper functions
# ===========================================================================


class TestCircuitBreaker:
    """Test the module-level circuit breaker functions."""

    def test_get_circuit_breaker_returns_instance(self):
        cb = get_gateway_credentials_circuit_breaker()
        assert cb is not None
        assert cb.name == "gateway_credentials_handler"

    def test_get_circuit_breaker_status_returns_dict(self):
        status = get_gateway_credentials_circuit_breaker_status()
        assert isinstance(status, dict)

    def test_reset_circuit_breaker(self):
        cb = get_gateway_credentials_circuit_breaker()
        # Simulate some failures
        cb._single_failures = 3
        cb._single_open_at = 100.0
        cb._single_successes = 1
        cb._single_half_open_calls = 2
        reset_gateway_credentials_circuit_breaker()
        assert cb._single_failures == 0
        assert cb._single_open_at == 0.0
        assert cb._single_successes == 0
        assert cb._single_half_open_calls == 0

    def test_circuit_breaker_config(self):
        cb = get_gateway_credentials_circuit_breaker()
        assert cb.failure_threshold == 5
        assert cb.cooldown_seconds == 30.0


# ===========================================================================
# Module exports
# ===========================================================================


class TestModuleExports:
    """Test that __all__ exports are correct."""

    def test_all_exports(self):
        from aragora.server.handlers import gateway_credentials_handler

        assert "GatewayCredentialsHandler" in gateway_credentials_handler.__all__
        assert "get_gateway_credentials_circuit_breaker" in gateway_credentials_handler.__all__
        assert (
            "get_gateway_credentials_circuit_breaker_status" in gateway_credentials_handler.__all__
        )
        assert "reset_gateway_credentials_circuit_breaker" in gateway_credentials_handler.__all__

    def test_all_exports_count(self):
        from aragora.server.handlers import gateway_credentials_handler

        assert len(gateway_credentials_handler.__all__) == 4


# ===========================================================================
# End-to-end: store then list then get then rotate then delete
# ===========================================================================


class TestEndToEnd:
    """Integration-style tests exercising the full lifecycle."""

    def test_store_list_get_rotate_delete(self, handler):
        # 1. Store a credential
        store_http = MockHTTPHandler(
            body={
                "service_name": "lifecycle-svc",
                "credential_type": "api_key",
                "value": "initial-secret",
            }
        )
        store_result = handler.handle_post("/api/v1/gateway/credentials", {}, store_http)
        assert _status(store_result) == 201
        cred_id = _body(store_result)["credential_id"]

        # 2. List credentials
        list_http = MockHTTPHandler()
        list_result = handler.handle("/api/v1/gateway/credentials", {}, list_http)
        assert _status(list_result) == 200
        assert _body(list_result)["total"] == 1

        # 3. Get credential by ID
        get_http = MockHTTPHandler()
        get_result = handler.handle(f"/api/v1/gateway/credentials/{cred_id}", {}, get_http)
        assert _status(get_result) == 200
        assert _body(get_result)["credential_id"] == cred_id
        assert _body(get_result)["service_name"] == "lifecycle-svc"

        # 4. Rotate credential
        rotate_http = MockHTTPHandler(body={"value": "rotated-secret"})
        rotate_result = handler.handle_post(
            f"/api/v1/gateway/credentials/{cred_id}/rotate", {}, rotate_http
        )
        assert _status(rotate_result) == 201
        new_cred_id = _body(rotate_result)["credential_id"]
        assert new_cred_id != cred_id

        # Old credential is marked rotated
        assert handler._credentials[cred_id]["status"] == "rotated"

        # 5. List should now show 2 credentials
        list_http2 = MockHTTPHandler()
        list_result2 = handler.handle("/api/v1/gateway/credentials", {}, list_http2)
        assert _body(list_result2)["total"] == 2

        # 6. Delete old credential
        del_http = MockHTTPHandler()
        del_result = handler.handle_delete(f"/api/v1/gateway/credentials/{cred_id}", {}, del_http)
        assert _status(del_result) == 200

        # 7. List should now show 1 credential
        list_http3 = MockHTTPHandler()
        list_result3 = handler.handle("/api/v1/gateway/credentials", {}, list_http3)
        assert _body(list_result3)["total"] == 1
        assert _body(list_result3)["credentials"][0]["credential_id"] == new_cred_id

    def test_value_never_leaked(self, handler):
        """Verify that credential value never leaks through any response."""
        secret = "super-secret-api-key-12345"

        # Store
        store_http = MockHTTPHandler(
            body={
                "service_name": "leak-test",
                "credential_type": "api_key",
                "value": secret,
            }
        )
        store_result = handler.handle_post("/api/v1/gateway/credentials", {}, store_http)
        cred_id = _body(store_result)["credential_id"]
        assert secret not in json.dumps(_body(store_result))

        # List
        list_http = MockHTTPHandler()
        list_result = handler.handle("/api/v1/gateway/credentials", {}, list_http)
        assert secret not in json.dumps(_body(list_result))

        # Get
        get_http = MockHTTPHandler()
        get_result = handler.handle(f"/api/v1/gateway/credentials/{cred_id}", {}, get_http)
        assert secret not in json.dumps(_body(get_result))

        # Rotate
        new_secret = "new-super-secret-67890"
        rotate_http = MockHTTPHandler(body={"value": new_secret})
        rotate_result = handler.handle_post(
            f"/api/v1/gateway/credentials/{cred_id}/rotate", {}, rotate_http
        )
        assert secret not in json.dumps(_body(rotate_result))
        assert new_secret not in json.dumps(_body(rotate_result))

        # Also verify in-memory store never contains values
        for meta in handler._credentials.values():
            assert "value" not in meta
            assert secret not in json.dumps(meta)
            assert new_secret not in json.dumps(meta)


# ===========================================================================
# Credential store accessor
# ===========================================================================


class TestCredentialStoreAccessor:
    """Test _get_credential_store and _get_credentials."""

    def test_get_credential_store_none(self, handler):
        assert handler._get_credential_store() is None

    def test_get_credential_store_from_context(self, handler_with_proxy, mock_proxy):
        assert handler_with_proxy._get_credential_store() is mock_proxy

    def test_get_credentials_returns_dict(self, handler):
        assert isinstance(handler._get_credentials(), dict)
        assert handler._get_credentials() is handler._credentials


# ===========================================================================
# Service name pattern
# ===========================================================================


class TestServiceNamePattern:
    """Test the SERVICE_NAME_PATTERN regex."""

    def test_valid_names(self):
        valid = ["a", "abc", "a-b-c", "A1", "my-service-123", "X" * 128]
        for name in valid:
            assert SERVICE_NAME_PATTERN.match(name) is not None, f"Should match: {name}"

    def test_invalid_names(self):
        invalid = ["-start", "", " space", "a" * 129, "special!char", "under_score"]
        for name in invalid:
            assert SERVICE_NAME_PATTERN.match(name) is None, f"Should not match: {name}"
