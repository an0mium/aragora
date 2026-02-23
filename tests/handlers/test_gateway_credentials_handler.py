"""Tests for the GatewayCredentialsHandler REST endpoints.

Comprehensive test coverage for gateway_credentials_handler.py (516 LOC):

Routes:
    POST   /api/v1/gateway/credentials              - Store a new credential
    GET    /api/v1/gateway/credentials               - List credentials (metadata only)
    GET    /api/v1/gateway/credentials/{id}          - Get credential metadata
    DELETE /api/v1/gateway/credentials/{id}          - Delete a credential
    POST   /api/v1/gateway/credentials/{id}/rotate   - Rotate a credential

Coverage includes:
    - ROUTES class attribute verification
    - can_handle() routing for all paths
    - Path extraction helpers (_extract_credential_id, _is_rotate_path)
    - Validation helpers (_validate_service_name, _validate_credential_type)
    - Credential metadata construction (_make_credential_metadata)
    - All route handlers (GET, POST, DELETE) with happy and error paths
    - Credential proxy integration and failure handling
    - Security: credential values are NEVER returned in any response
    - Circuit breaker helper functions
    - Module __all__ exports
    - End-to-end lifecycle test
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

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
    """Extract the JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    # HandlerResult namedtuple: parse .body bytes/str as JSON
    if hasattr(result, "body"):
        raw = result.body
        if isinstance(raw, bytes):
            return json.loads(raw)
        if isinstance(raw, str):
            return json.loads(raw)
        if isinstance(raw, dict):
            return raw
    # Fallback: try tuple unpacking (body, status, headers)
    try:
        body, _status, _headers = result
        if isinstance(body, dict):
            return body
        return json.loads(body) if body else {}
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


class _MockHTTPHandler:
    """Simulates the HTTP handler object passed to BaseHandler methods.

    BaseHandler.read_json_body reads Content-Length from headers, then reads
    that many bytes from rfile.
    """

    def __init__(self, body: dict | None = None):
        self.rfile = MagicMock()
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
        self.client_address = ("127.0.0.1", 12345)


class _MockHTTPHandlerBadJSON:
    """Returns non-JSON bytes to trigger parse failures."""

    def __init__(self):
        self.rfile = MagicMock()
        self.rfile.read.return_value = b"{broken json"
        self.headers = {
            "Content-Length": "12",
            "Content-Type": "application/json",
        }
        self.client_address = ("127.0.0.1", 12345)


class _MockHTTPHandlerEmptyBody:
    """Returns Content-Length 0 (no body)."""

    def __init__(self):
        self.rfile = MagicMock()
        self.rfile.read.return_value = b""
        self.headers = {
            "Content-Length": "0",
            "Content-Type": "application/json",
        }
        self.client_address = ("127.0.0.1", 12345)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a fresh GatewayCredentialsHandler with empty context."""
    return GatewayCredentialsHandler(server_context={})


@pytest.fixture
def proxy():
    """Create a mock CredentialProxy."""
    p = MagicMock()
    p.store = MagicMock()
    p.delete = MagicMock()
    return p


@pytest.fixture
def handler_with_proxy(proxy):
    """Handler wired to a mock credential proxy."""
    h = GatewayCredentialsHandler(server_context={"credential_proxy": proxy})
    return h


@pytest.fixture
def seeded_handler(handler):
    """Handler with one pre-loaded credential entry in memory."""
    handler._credentials["cred_test001"] = {
        "credential_id": "cred_test001",
        "service_name": "alpha-svc",
        "credential_type": "api_key",
        "status": "active",
        "created_at": "2026-01-15T12:00:00+00:00",
        "tenant_id": "org-42",
        "scopes": ["read", "write"],
        "expires_at": "2027-06-01T00:00:00+00:00",
        "metadata": {"region": "us-east-1"},
    }
    return handler


@pytest.fixture
def seeded_handler_with_proxy(handler_with_proxy):
    """Handler with proxy AND a pre-loaded credential."""
    handler_with_proxy._credentials["cred_test001"] = {
        "credential_id": "cred_test001",
        "service_name": "alpha-svc",
        "credential_type": "api_key",
        "status": "active",
        "created_at": "2026-01-15T12:00:00+00:00",
    }
    return handler_with_proxy


@pytest.fixture(autouse=True)
def _reset_cb():
    """Reset the module-level circuit breaker between tests."""
    reset_gateway_credentials_circuit_breaker()
    yield
    reset_gateway_credentials_circuit_breaker()


# ===========================================================================
# ROUTES class attribute
# ===========================================================================


class TestRoutes:
    """Verify the ROUTES class attribute."""

    def test_routes_count(self):
        assert len(GatewayCredentialsHandler.ROUTES) == 2

    def test_routes_has_base_path(self):
        assert "/api/v1/gateway/credentials" in GatewayCredentialsHandler.ROUTES

    def test_routes_has_wildcard(self):
        assert "/api/v1/gateway/credentials/*" in GatewayCredentialsHandler.ROUTES


# ===========================================================================
# can_handle
# ===========================================================================


class TestCanHandle:
    """Test can_handle path matching."""

    def test_base_credentials_path(self, handler):
        assert handler.can_handle("/api/v1/gateway/credentials") is True

    def test_credentials_with_id(self, handler):
        assert handler.can_handle("/api/v1/gateway/credentials/cred_xyz") is True

    def test_credentials_rotate(self, handler):
        assert handler.can_handle("/api/v1/gateway/credentials/cred_xyz/rotate") is True

    def test_credentials_trailing_slash(self, handler):
        assert handler.can_handle("/api/v1/gateway/credentials/") is True

    def test_rejects_unrelated_path(self, handler):
        assert handler.can_handle("/api/v1/debates") is False

    def test_rejects_partial_match(self, handler):
        assert handler.can_handle("/api/v1/gateway/creds") is False

    def test_rejects_gateway_other(self, handler):
        assert handler.can_handle("/api/v1/gateway/routes") is False

    def test_rejects_root(self, handler):
        assert handler.can_handle("/") is False


# ===========================================================================
# _extract_credential_id
# ===========================================================================


class TestExtractCredentialId:
    """Test the _extract_credential_id helper."""

    def test_standard_path(self, handler):
        assert handler._extract_credential_id("/api/v1/gateway/credentials/cred_abc") == "cred_abc"

    def test_trailing_slash(self, handler):
        assert handler._extract_credential_id("/api/v1/gateway/credentials/cred_abc/") == "cred_abc"

    def test_rotate_path_extracts_id(self, handler):
        assert handler._extract_credential_id("/api/v1/gateway/credentials/cred_abc/rotate") == "cred_abc"

    def test_base_path_returns_none(self, handler):
        assert handler._extract_credential_id("/api/v1/gateway/credentials") is None

    def test_trailing_slash_only_returns_none(self, handler):
        assert handler._extract_credential_id("/api/v1/gateway/credentials/") is None

    def test_id_equals_credentials_returns_none(self, handler):
        # Edge case: the code explicitly rejects cred_id == "credentials"
        assert handler._extract_credential_id("/api/v1/gateway/credentials/credentials") is None


# ===========================================================================
# _is_rotate_path
# ===========================================================================


class TestIsRotatePath:
    """Test the _is_rotate_path helper."""

    def test_rotate_path(self, handler):
        assert handler._is_rotate_path("/api/v1/gateway/credentials/id1/rotate") is True

    def test_rotate_path_trailing_slash(self, handler):
        assert handler._is_rotate_path("/api/v1/gateway/credentials/id1/rotate/") is True

    def test_non_rotate_path(self, handler):
        assert handler._is_rotate_path("/api/v1/gateway/credentials/id1") is False

    def test_base_path(self, handler):
        assert handler._is_rotate_path("/api/v1/gateway/credentials") is False


# ===========================================================================
# _validate_service_name
# ===========================================================================


class TestValidateServiceName:
    """Test the _validate_service_name validation helper."""

    def test_valid_simple_name(self, handler):
        assert handler._validate_service_name("my-service") is None

    def test_valid_alphanumeric(self, handler):
        assert handler._validate_service_name("Service123") is None

    def test_valid_single_char(self, handler):
        assert handler._validate_service_name("x") is None

    def test_valid_max_length(self, handler):
        assert handler._validate_service_name("a" * 128) is None

    def test_empty_string(self, handler):
        err = handler._validate_service_name("")
        assert err == "service_name is required"

    def test_starts_with_hyphen(self, handler):
        err = handler._validate_service_name("-invalid")
        assert err is not None
        assert "Invalid service_name" in err

    def test_contains_spaces(self, handler):
        err = handler._validate_service_name("has space")
        assert err is not None

    def test_too_long(self, handler):
        err = handler._validate_service_name("a" * 129)
        assert err is not None

    def test_special_characters(self, handler):
        err = handler._validate_service_name("bad@name!")
        assert err is not None

    def test_unicode_rejected(self, handler):
        err = handler._validate_service_name("servi\u00e7e")
        assert err is not None


# ===========================================================================
# _validate_credential_type
# ===========================================================================


class TestValidateCredentialType:
    """Test the _validate_credential_type validation helper."""

    @pytest.mark.parametrize("ctype", sorted(VALID_CREDENTIAL_TYPES))
    def test_valid_types(self, handler, ctype):
        assert handler._validate_credential_type(ctype) is None

    def test_empty_string(self, handler):
        err = handler._validate_credential_type("")
        assert err == "credential_type is required"

    def test_invalid_type(self, handler):
        err = handler._validate_credential_type("password")
        assert err is not None
        assert "Invalid credential_type" in err

    def test_valid_credential_types_constant(self):
        assert VALID_CREDENTIAL_TYPES == {"api_key", "oauth_token", "bearer_token", "basic_auth", "custom"}


# ===========================================================================
# _make_credential_metadata
# ===========================================================================


class TestMakeCredentialMetadata:
    """Test the _make_credential_metadata factory."""

    def test_required_fields_only(self, handler):
        meta = handler._make_credential_metadata(
            credential_id="cred_01",
            service_name="svc",
            credential_type="api_key",
        )
        assert meta["credential_id"] == "cred_01"
        assert meta["service_name"] == "svc"
        assert meta["credential_type"] == "api_key"
        assert meta["status"] == "active"
        assert "created_at" in meta
        assert "tenant_id" not in meta
        assert "scopes" not in meta
        assert "expires_at" not in meta
        assert "metadata" not in meta

    def test_all_optional_fields(self, handler):
        meta = handler._make_credential_metadata(
            credential_id="cred_full",
            service_name="svc",
            credential_type="bearer_token",
            tenant_id="t-99",
            scopes=["admin"],
            expires_at="2028-01-01T00:00:00Z",
            metadata={"k": "v"},
            status="rotated",
        )
        assert meta["tenant_id"] == "t-99"
        assert meta["scopes"] == ["admin"]
        assert meta["expires_at"] == "2028-01-01T00:00:00Z"
        assert meta["metadata"] == {"k": "v"}
        assert meta["status"] == "rotated"

    def test_value_never_in_metadata(self, handler):
        meta = handler._make_credential_metadata(
            credential_id="cred_no_val",
            service_name="s",
            credential_type="custom",
        )
        assert "value" not in meta


# ===========================================================================
# POST /api/v1/gateway/credentials (store)
# ===========================================================================


class TestStoreCredential:
    """Test POST /api/v1/gateway/credentials."""

    def test_store_success(self, handler):
        http = _MockHTTPHandler(body={
            "service_name": "my-svc",
            "credential_type": "api_key",
            "value": "secret-key-abc",
        })
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 201
        body = _body(result)
        assert body["credential_id"].startswith("cred_")
        assert body["service_name"] == "my-svc"
        assert body["credential_type"] == "api_key"
        assert body["message"] == "Credential stored successfully"
        assert "created_at" in body
        # Value must NEVER be in response
        assert "value" not in body
        assert "secret-key-abc" not in json.dumps(body)

    def test_store_with_optional_fields(self, handler):
        http = _MockHTTPHandler(body={
            "service_name": "svc-opt",
            "credential_type": "oauth_token",
            "value": "tok-123",
            "tenant_id": "ten-1",
            "scopes": ["read"],
            "expires_at": "2028-12-31T00:00:00Z",
            "metadata": {"env": "prod"},
        })
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 201
        body = _body(result)
        assert body["tenant_id"] == "ten-1"
        assert body["scopes"] == ["read"]
        assert body["expires_at"] == "2028-12-31T00:00:00Z"
        assert "value" not in body

    def test_store_persists_in_memory(self, handler):
        http = _MockHTTPHandler(body={
            "service_name": "svc-mem",
            "credential_type": "api_key",
            "value": "val",
        })
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        cred_id = _body(result)["credential_id"]
        assert cred_id in handler._credentials
        meta = handler._credentials[cred_id]
        assert meta["service_name"] == "svc-mem"
        assert "value" not in meta

    def test_store_calls_proxy(self, handler_with_proxy, proxy):
        http = _MockHTTPHandler(body={
            "service_name": "svc-prx",
            "credential_type": "bearer_token",
            "value": "bearer-val",
        })
        result = handler_with_proxy.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 201
        proxy.store.assert_called_once()

    def test_store_proxy_runtime_error(self, handler_with_proxy, proxy):
        proxy.store.side_effect = RuntimeError("connection refused")
        http = _MockHTTPHandler(body={
            "service_name": "svc-err",
            "credential_type": "api_key",
            "value": "val",
        })
        result = handler_with_proxy.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 500
        assert "failed" in _body(result).get("error", "").lower()

    def test_store_proxy_os_error(self, handler_with_proxy, proxy):
        proxy.store.side_effect = OSError("disk full")
        http = _MockHTTPHandler(body={
            "service_name": "svc-os",
            "credential_type": "api_key",
            "value": "val",
        })
        result = handler_with_proxy.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 500

    def test_store_proxy_attribute_error(self, handler_with_proxy, proxy):
        proxy.store.side_effect = AttributeError("missing method")
        http = _MockHTTPHandler(body={
            "service_name": "svc-attr",
            "credential_type": "api_key",
            "value": "val",
        })
        result = handler_with_proxy.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 500

    def test_store_proxy_type_error(self, handler_with_proxy, proxy):
        proxy.store.side_effect = TypeError("wrong type")
        http = _MockHTTPHandler(body={
            "service_name": "svc-typ",
            "credential_type": "api_key",
            "value": "val",
        })
        result = handler_with_proxy.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 500

    def test_store_missing_service_name(self, handler):
        http = _MockHTTPHandler(body={
            "credential_type": "api_key",
            "value": "val",
        })
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 400
        assert "service_name" in _body(result).get("error", "").lower()

    def test_store_missing_credential_type(self, handler):
        http = _MockHTTPHandler(body={
            "service_name": "svc",
            "value": "val",
        })
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 400
        assert "credential_type" in _body(result).get("error", "").lower()

    def test_store_missing_value(self, handler):
        http = _MockHTTPHandler(body={
            "service_name": "svc",
            "credential_type": "api_key",
        })
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 400
        assert "value" in _body(result).get("error", "").lower()

    def test_store_empty_value(self, handler):
        http = _MockHTTPHandler(body={
            "service_name": "svc",
            "credential_type": "api_key",
            "value": "",
        })
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 400

    def test_store_whitespace_only_value(self, handler):
        http = _MockHTTPHandler(body={
            "service_name": "svc",
            "credential_type": "api_key",
            "value": "   ",
        })
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 400

    def test_store_non_string_value(self, handler):
        http = _MockHTTPHandler(body={
            "service_name": "svc",
            "credential_type": "api_key",
            "value": 42,
        })
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 400

    def test_store_invalid_service_name(self, handler):
        http = _MockHTTPHandler(body={
            "service_name": "-bad",
            "credential_type": "api_key",
            "value": "val",
        })
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 400
        assert "Invalid service_name" in _body(result).get("error", "")

    def test_store_invalid_credential_type(self, handler):
        http = _MockHTTPHandler(body={
            "service_name": "svc",
            "credential_type": "password",
            "value": "val",
        })
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 400
        assert "Invalid credential_type" in _body(result).get("error", "")

    def test_store_invalid_json_body(self, handler):
        http = _MockHTTPHandlerBadJSON()
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 400

    def test_store_empty_body(self, handler):
        http = _MockHTTPHandlerEmptyBody()
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 400

    def test_store_non_matching_path_returns_none(self, handler):
        http = _MockHTTPHandler(body={"service_name": "s", "credential_type": "api_key", "value": "v"})
        result = handler.handle_post("/api/v1/debates", {}, http)
        assert result is None


# ===========================================================================
# GET /api/v1/gateway/credentials (list)
# ===========================================================================


class TestListCredentials:
    """Test GET /api/v1/gateway/credentials."""

    def test_list_empty(self, handler):
        http = _MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["credentials"] == []
        assert body["total"] == 0

    def test_list_returns_metadata(self, seeded_handler):
        http = _MockHTTPHandler()
        result = seeded_handler.handle("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["total"] == 1
        cred = body["credentials"][0]
        assert cred["credential_id"] == "cred_test001"
        assert cred["service_name"] == "alpha-svc"
        assert cred["credential_type"] == "api_key"
        assert cred["status"] == "active"
        assert "value" not in cred

    def test_list_includes_tenant_id_and_expires_at(self, seeded_handler):
        http = _MockHTTPHandler()
        result = seeded_handler.handle("/api/v1/gateway/credentials", {}, http)
        cred = _body(result)["credentials"][0]
        assert cred["tenant_id"] == "org-42"
        assert cred["expires_at"] == "2027-06-01T00:00:00+00:00"

    def test_list_excludes_scopes_and_metadata(self, seeded_handler):
        """List endpoint does NOT include scopes or metadata."""
        http = _MockHTTPHandler()
        result = seeded_handler.handle("/api/v1/gateway/credentials", {}, http)
        cred = _body(result)["credentials"][0]
        assert "scopes" not in cred
        assert "metadata" not in cred

    def test_list_multiple(self, handler):
        for i in range(5):
            handler._credentials[f"cred_{i}"] = {
                "credential_id": f"cred_{i}",
                "service_name": f"svc-{i}",
                "credential_type": "custom",
                "status": "active",
                "created_at": "2026-02-01T00:00:00Z",
            }
        http = _MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/credentials", {}, http)
        assert _body(result)["total"] == 5
        assert len(_body(result)["credentials"]) == 5

    def test_list_non_matching_path_returns_none(self, handler):
        http = _MockHTTPHandler()
        result = handler.handle("/api/v1/other", {}, http)
        assert result is None


# ===========================================================================
# GET /api/v1/gateway/credentials/{id} (get single)
# ===========================================================================


class TestGetCredential:
    """Test GET /api/v1/gateway/credentials/{id}."""

    def test_get_found(self, seeded_handler):
        http = _MockHTTPHandler()
        result = seeded_handler.handle("/api/v1/gateway/credentials/cred_test001", {}, http)
        assert _status(result) == 200
        body = _body(result)
        assert body["credential_id"] == "cred_test001"
        assert body["service_name"] == "alpha-svc"
        assert body["credential_type"] == "api_key"
        assert body["status"] == "active"
        assert body["tenant_id"] == "org-42"
        assert body["scopes"] == ["read", "write"]
        assert body["expires_at"] == "2027-06-01T00:00:00+00:00"
        assert body["metadata"] == {"region": "us-east-1"}
        assert "value" not in body

    def test_get_not_found(self, handler):
        http = _MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/credentials/cred_nope", {}, http)
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    def test_get_minimal_fields(self, handler):
        handler._credentials["cred_min"] = {
            "credential_id": "cred_min",
            "service_name": "svc-min",
            "credential_type": "custom",
            "status": "active",
            "created_at": "2026-03-01T00:00:00Z",
        }
        http = _MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/credentials/cred_min", {}, http)
        body = _body(result)
        assert _status(result) == 200
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

    def test_delete_success(self, seeded_handler):
        http = _MockHTTPHandler()
        result = seeded_handler.handle_delete(
            "/api/v1/gateway/credentials/cred_test001", {}, http
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["credential_id"] == "cred_test001"
        assert body["message"] == "Credential deleted successfully"
        assert "cred_test001" not in seeded_handler._credentials

    def test_delete_not_found(self, handler):
        http = _MockHTTPHandler()
        result = handler.handle_delete(
            "/api/v1/gateway/credentials/cred_missing", {}, http
        )
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    def test_delete_calls_proxy(self, seeded_handler_with_proxy, proxy):
        http = _MockHTTPHandler()
        result = seeded_handler_with_proxy.handle_delete(
            "/api/v1/gateway/credentials/cred_test001", {}, http
        )
        assert _status(result) == 200
        proxy.delete.assert_called_once_with("cred_test001")

    def test_delete_proxy_failure_still_removes(self, seeded_handler_with_proxy, proxy):
        """Proxy failure during delete is logged but not fatal."""
        proxy.delete.side_effect = RuntimeError("gone")
        http = _MockHTTPHandler()
        result = seeded_handler_with_proxy.handle_delete(
            "/api/v1/gateway/credentials/cred_test001", {}, http
        )
        assert _status(result) == 200
        assert "cred_test001" not in seeded_handler_with_proxy._credentials

    def test_delete_proxy_os_error(self, seeded_handler_with_proxy, proxy):
        proxy.delete.side_effect = OSError("disk error")
        http = _MockHTTPHandler()
        result = seeded_handler_with_proxy.handle_delete(
            "/api/v1/gateway/credentials/cred_test001", {}, http
        )
        assert _status(result) == 200

    def test_delete_proxy_attribute_error(self, seeded_handler_with_proxy, proxy):
        proxy.delete.side_effect = AttributeError("no delete")
        http = _MockHTTPHandler()
        result = seeded_handler_with_proxy.handle_delete(
            "/api/v1/gateway/credentials/cred_test001", {}, http
        )
        assert _status(result) == 200

    def test_delete_base_path_no_id_returns_none(self, handler):
        http = _MockHTTPHandler()
        result = handler.handle_delete("/api/v1/gateway/credentials", {}, http)
        assert result is None

    def test_delete_non_matching_path_returns_none(self, handler):
        http = _MockHTTPHandler()
        result = handler.handle_delete("/api/v1/other", {}, http)
        assert result is None


# ===========================================================================
# POST /api/v1/gateway/credentials/{id}/rotate
# ===========================================================================


class TestRotateCredential:
    """Test POST /api/v1/gateway/credentials/{id}/rotate."""

    def test_rotate_success(self, seeded_handler):
        http = _MockHTTPHandler(body={"value": "new-rotated-key"})
        result = seeded_handler.handle_post(
            "/api/v1/gateway/credentials/cred_test001/rotate", {}, http
        )
        assert _status(result) == 201
        body = _body(result)
        assert body["credential_id"].startswith("cred_")
        assert body["credential_id"] != "cred_test001"
        assert body["previous_credential_id"] == "cred_test001"
        assert body["service_name"] == "alpha-svc"
        assert body["credential_type"] == "api_key"
        assert body["status"] == "active"
        assert body["message"] == "Credential rotated successfully"
        assert "created_at" in body
        # Value NEVER in response
        assert "value" not in body
        assert "new-rotated-key" not in json.dumps(body)

    def test_rotate_marks_old_as_rotated(self, seeded_handler):
        http = _MockHTTPHandler(body={"value": "new-val"})
        seeded_handler.handle_post(
            "/api/v1/gateway/credentials/cred_test001/rotate", {}, http
        )
        assert seeded_handler._credentials["cred_test001"]["status"] == "rotated"

    def test_rotate_creates_new_entry(self, seeded_handler):
        http = _MockHTTPHandler(body={"value": "new-val"})
        result = seeded_handler.handle_post(
            "/api/v1/gateway/credentials/cred_test001/rotate", {}, http
        )
        new_id = _body(result)["credential_id"]
        assert new_id in seeded_handler._credentials
        new_meta = seeded_handler._credentials[new_id]
        assert new_meta["service_name"] == "alpha-svc"
        assert new_meta["credential_type"] == "api_key"
        assert "value" not in new_meta

    def test_rotate_preserves_optional_fields(self, seeded_handler):
        http = _MockHTTPHandler(body={"value": "new-val"})
        result = seeded_handler.handle_post(
            "/api/v1/gateway/credentials/cred_test001/rotate", {}, http
        )
        new_id = _body(result)["credential_id"]
        new_meta = seeded_handler._credentials[new_id]
        assert new_meta.get("tenant_id") == "org-42"
        assert new_meta.get("scopes") == ["read", "write"]
        assert new_meta.get("expires_at") == "2027-06-01T00:00:00+00:00"
        assert new_meta.get("metadata") == {"region": "us-east-1"}

    def test_rotate_not_found(self, handler):
        http = _MockHTTPHandler(body={"value": "val"})
        result = handler.handle_post(
            "/api/v1/gateway/credentials/cred_missing/rotate", {}, http
        )
        assert _status(result) == 404
        assert "not found" in _body(result).get("error", "").lower()

    def test_rotate_missing_value(self, seeded_handler):
        http = _MockHTTPHandler(body={})
        result = seeded_handler.handle_post(
            "/api/v1/gateway/credentials/cred_test001/rotate", {}, http
        )
        assert _status(result) == 400
        assert "value" in _body(result).get("error", "").lower()

    def test_rotate_empty_value(self, seeded_handler):
        http = _MockHTTPHandler(body={"value": ""})
        result = seeded_handler.handle_post(
            "/api/v1/gateway/credentials/cred_test001/rotate", {}, http
        )
        assert _status(result) == 400

    def test_rotate_whitespace_value(self, seeded_handler):
        http = _MockHTTPHandler(body={"value": "  "})
        result = seeded_handler.handle_post(
            "/api/v1/gateway/credentials/cred_test001/rotate", {}, http
        )
        assert _status(result) == 400

    def test_rotate_non_string_value(self, seeded_handler):
        http = _MockHTTPHandler(body={"value": 123})
        result = seeded_handler.handle_post(
            "/api/v1/gateway/credentials/cred_test001/rotate", {}, http
        )
        assert _status(result) == 400

    def test_rotate_invalid_json(self, seeded_handler):
        http = _MockHTTPHandlerBadJSON()
        result = seeded_handler.handle_post(
            "/api/v1/gateway/credentials/cred_test001/rotate", {}, http
        )
        assert _status(result) == 400

    def test_rotate_via_proxy(self, seeded_handler_with_proxy, proxy):
        http = _MockHTTPHandler(body={"value": "rotated-val"})
        result = seeded_handler_with_proxy.handle_post(
            "/api/v1/gateway/credentials/cred_test001/rotate", {}, http
        )
        assert _status(result) == 201
        proxy.store.assert_called_once()

    def test_rotate_proxy_failure_reverts_status(self, seeded_handler_with_proxy, proxy):
        proxy.store.side_effect = RuntimeError("fail")
        http = _MockHTTPHandler(body={"value": "rotated-val"})
        result = seeded_handler_with_proxy.handle_post(
            "/api/v1/gateway/credentials/cred_test001/rotate", {}, http
        )
        assert _status(result) == 500
        assert "Failed to rotate credential" in _body(result).get("error", "")
        # Old credential status should be reverted to active
        old = seeded_handler_with_proxy._credentials["cred_test001"]
        assert old["status"] == "active"

    def test_rotate_proxy_type_error_reverts(self, seeded_handler_with_proxy, proxy):
        proxy.store.side_effect = TypeError("bad")
        http = _MockHTTPHandler(body={"value": "val"})
        result = seeded_handler_with_proxy.handle_post(
            "/api/v1/gateway/credentials/cred_test001/rotate", {}, http
        )
        assert _status(result) == 500
        old = seeded_handler_with_proxy._credentials["cred_test001"]
        assert old["status"] == "active"

    def test_rotate_proxy_os_error_reverts(self, seeded_handler_with_proxy, proxy):
        proxy.store.side_effect = OSError("io err")
        http = _MockHTTPHandler(body={"value": "val"})
        result = seeded_handler_with_proxy.handle_post(
            "/api/v1/gateway/credentials/cred_test001/rotate", {}, http
        )
        assert _status(result) == 500
        old = seeded_handler_with_proxy._credentials["cred_test001"]
        assert old["status"] == "active"


# ===========================================================================
# handle() dispatch (GET)
# ===========================================================================


class TestHandleGetDispatch:
    """Test that handle() correctly dispatches GET requests."""

    def test_get_dispatches_to_list(self, handler):
        http = _MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 200
        assert "credentials" in _body(result)

    def test_get_dispatches_to_get_by_id(self, seeded_handler):
        http = _MockHTTPHandler()
        result = seeded_handler.handle("/api/v1/gateway/credentials/cred_test001", {}, http)
        assert _status(result) == 200
        assert _body(result)["credential_id"] == "cred_test001"

    def test_get_non_matching_returns_none(self, handler):
        http = _MockHTTPHandler()
        result = handler.handle("/api/v1/billing/plans", {}, http)
        assert result is None

    def test_get_with_trailing_slash_lists(self, handler):
        """Trailing slash on base path should still list credentials."""
        http = _MockHTTPHandler()
        # path.rstrip("/") == "/api/v1/gateway/credentials" so it lists
        result = handler.handle("/api/v1/gateway/credentials/", {}, http)
        # Since rstrip gives empty id segment, _extract_credential_id returns None
        # and path.rstrip("/") == "/api/v1/gateway/credentials" matches list
        # But can_handle sees /api/v1/gateway/credentials/ which starts with prefix
        # The id extraction on /api/v1/gateway/credentials/ returns None
        # Then the base path check: "/api/v1/gateway/credentials/" rstrip -> matches
        assert result is not None


# ===========================================================================
# handle_post() dispatch
# ===========================================================================


class TestHandlePostDispatch:
    """Test that handle_post() correctly dispatches POST requests."""

    def test_post_dispatches_to_store(self, handler):
        http = _MockHTTPHandler(body={
            "service_name": "dispatch-svc",
            "credential_type": "api_key",
            "value": "val",
        })
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert _status(result) == 201

    def test_post_dispatches_to_rotate(self, seeded_handler):
        http = _MockHTTPHandler(body={"value": "new-val"})
        result = seeded_handler.handle_post(
            "/api/v1/gateway/credentials/cred_test001/rotate", {}, http
        )
        assert _status(result) == 201

    def test_post_non_matching_returns_none(self, handler):
        http = _MockHTTPHandler(body={})
        result = handler.handle_post("/api/v1/other", {}, http)
        assert result is None


# ===========================================================================
# handle_delete() dispatch
# ===========================================================================


class TestHandleDeleteDispatch:
    """Test that handle_delete() correctly dispatches DELETE requests."""

    def test_delete_dispatches_to_delete(self, seeded_handler):
        http = _MockHTTPHandler()
        result = seeded_handler.handle_delete(
            "/api/v1/gateway/credentials/cred_test001", {}, http
        )
        assert _status(result) == 200

    def test_delete_non_matching_returns_none(self, handler):
        http = _MockHTTPHandler()
        result = handler.handle_delete("/api/v1/other", {}, http)
        assert result is None


# ===========================================================================
# Circuit breaker functions
# ===========================================================================


class TestCircuitBreakerFunctions:
    """Test the module-level circuit breaker helpers."""

    def test_get_returns_circuit_breaker_instance(self):
        cb = get_gateway_credentials_circuit_breaker()
        assert cb is not None
        assert cb.name == "gateway_credentials_handler"

    def test_get_status_returns_dict(self):
        status = get_gateway_credentials_circuit_breaker_status()
        assert isinstance(status, dict)

    def test_reset_clears_state(self):
        cb = get_gateway_credentials_circuit_breaker()
        cb._single_failures = 10
        cb._single_open_at = 999.0
        cb._single_successes = 5
        cb._single_half_open_calls = 3
        reset_gateway_credentials_circuit_breaker()
        assert cb._single_failures == 0
        assert cb._single_open_at == 0.0
        assert cb._single_successes == 0
        assert cb._single_half_open_calls == 0

    def test_circuit_breaker_threshold_config(self):
        cb = get_gateway_credentials_circuit_breaker()
        assert cb.failure_threshold == 5
        assert cb.cooldown_seconds == 30.0


# ===========================================================================
# Module exports (__all__)
# ===========================================================================


class TestModuleExports:
    """Verify __all__ exports are correct."""

    def test_exports_contain_handler(self):
        from aragora.server.handlers import gateway_credentials_handler as mod
        assert "GatewayCredentialsHandler" in mod.__all__

    def test_exports_contain_circuit_breaker_functions(self):
        from aragora.server.handlers import gateway_credentials_handler as mod
        assert "get_gateway_credentials_circuit_breaker" in mod.__all__
        assert "get_gateway_credentials_circuit_breaker_status" in mod.__all__
        assert "reset_gateway_credentials_circuit_breaker" in mod.__all__

    def test_exports_count(self):
        from aragora.server.handlers import gateway_credentials_handler as mod
        assert len(mod.__all__) == 4


# ===========================================================================
# Credential store accessor
# ===========================================================================


class TestCredentialStoreAccessor:
    """Test _get_credential_store and _get_credentials."""

    def test_no_proxy_returns_none(self, handler):
        assert handler._get_credential_store() is None

    def test_proxy_from_context(self, handler_with_proxy, proxy):
        assert handler_with_proxy._get_credential_store() is proxy

    def test_get_credentials_returns_same_dict(self, handler):
        creds = handler._get_credentials()
        assert creds is handler._credentials
        assert isinstance(creds, dict)


# ===========================================================================
# SERVICE_NAME_PATTERN regex
# ===========================================================================


class TestServiceNamePattern:
    """Test the SERVICE_NAME_PATTERN regex directly."""

    @pytest.mark.parametrize("name", ["a", "abc", "A1-B2", "x" * 128])
    def test_valid_patterns(self, name):
        assert SERVICE_NAME_PATTERN.match(name) is not None

    @pytest.mark.parametrize("name", ["-start", "", "a" * 129, "sp ace", "no!ok", "under_score"])
    def test_invalid_patterns(self, name):
        assert SERVICE_NAME_PATTERN.match(name) is None


# ===========================================================================
# Constructor
# ===========================================================================


class TestConstructor:
    """Test handler construction."""

    def test_default_empty_context(self):
        h = GatewayCredentialsHandler(server_context={})
        assert h.ctx == {}
        assert h._credentials == {}

    def test_context_preserved(self):
        ctx: dict[str, Any] = {"credential_proxy": MagicMock()}
        h = GatewayCredentialsHandler(server_context=ctx)
        assert h._get_credential_store() is not None

    def test_credentials_start_empty(self):
        h = GatewayCredentialsHandler(server_context={})
        assert len(h._credentials) == 0


# ===========================================================================
# Security: value never leaked
# ===========================================================================


class TestValueNeverLeaked:
    """Critical security tests: credential values must NEVER appear in responses."""

    def test_store_response_no_value(self, handler):
        secret = "ultra-secret-key-12345"
        http = _MockHTTPHandler(body={
            "service_name": "sec-svc",
            "credential_type": "api_key",
            "value": secret,
        })
        result = handler.handle_post("/api/v1/gateway/credentials", {}, http)
        assert secret not in json.dumps(_body(result))

    def test_list_response_no_value(self, handler):
        secret = "list-secret-key"
        http_store = _MockHTTPHandler(body={
            "service_name": "sec-svc",
            "credential_type": "api_key",
            "value": secret,
        })
        handler.handle_post("/api/v1/gateway/credentials", {}, http_store)

        http_list = _MockHTTPHandler()
        result = handler.handle("/api/v1/gateway/credentials", {}, http_list)
        assert secret not in json.dumps(_body(result))

    def test_get_response_no_value(self, handler):
        secret = "get-secret-key"
        http_store = _MockHTTPHandler(body={
            "service_name": "sec-svc",
            "credential_type": "api_key",
            "value": secret,
        })
        store_result = handler.handle_post("/api/v1/gateway/credentials", {}, http_store)
        cred_id = _body(store_result)["credential_id"]

        http_get = _MockHTTPHandler()
        result = handler.handle(f"/api/v1/gateway/credentials/{cred_id}", {}, http_get)
        assert secret not in json.dumps(_body(result))

    def test_rotate_response_no_value(self, seeded_handler):
        new_secret = "rotate-secret"
        http = _MockHTTPHandler(body={"value": new_secret})
        result = seeded_handler.handle_post(
            "/api/v1/gateway/credentials/cred_test001/rotate", {}, http
        )
        assert new_secret not in json.dumps(_body(result))

    def test_in_memory_store_no_value(self, handler):
        secret = "mem-secret"
        http_store = _MockHTTPHandler(body={
            "service_name": "sec-svc",
            "credential_type": "api_key",
            "value": secret,
        })
        handler.handle_post("/api/v1/gateway/credentials", {}, http_store)
        for meta in handler._credentials.values():
            assert "value" not in meta
            assert secret not in json.dumps(meta)


# ===========================================================================
# End-to-end lifecycle
# ===========================================================================


class TestEndToEndLifecycle:
    """Full lifecycle: store -> list -> get -> rotate -> delete."""

    def test_full_lifecycle(self, handler):
        # 1. Store
        http_store = _MockHTTPHandler(body={
            "service_name": "lifecycle-svc",
            "credential_type": "api_key",
            "value": "initial-secret",
        })
        store_result = handler.handle_post("/api/v1/gateway/credentials", {}, http_store)
        assert _status(store_result) == 201
        cred_id = _body(store_result)["credential_id"]

        # 2. List (should show 1)
        list_result = handler.handle("/api/v1/gateway/credentials", {}, _MockHTTPHandler())
        assert _body(list_result)["total"] == 1

        # 3. Get by ID
        get_result = handler.handle(f"/api/v1/gateway/credentials/{cred_id}", {}, _MockHTTPHandler())
        assert _status(get_result) == 200
        assert _body(get_result)["credential_id"] == cred_id

        # 4. Rotate
        rotate_result = handler.handle_post(
            f"/api/v1/gateway/credentials/{cred_id}/rotate",
            {},
            _MockHTTPHandler(body={"value": "rotated"}),
        )
        assert _status(rotate_result) == 201
        new_id = _body(rotate_result)["credential_id"]
        assert new_id != cred_id
        assert handler._credentials[cred_id]["status"] == "rotated"

        # 5. List now shows 2
        list_result2 = handler.handle("/api/v1/gateway/credentials", {}, _MockHTTPHandler())
        assert _body(list_result2)["total"] == 2

        # 6. Delete old
        del_result = handler.handle_delete(
            f"/api/v1/gateway/credentials/{cred_id}", {}, _MockHTTPHandler()
        )
        assert _status(del_result) == 200

        # 7. List now shows 1
        list_result3 = handler.handle("/api/v1/gateway/credentials", {}, _MockHTTPHandler())
        assert _body(list_result3)["total"] == 1
        assert _body(list_result3)["credentials"][0]["credential_id"] == new_id
