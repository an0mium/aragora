"""
Tests for GatewayCredentialsHandler - Gateway credential management HTTP endpoints.

Tests cover:
- Path routing (can_handle)
- Credential storage with validation
- Credential listing (metadata only, NEVER values)
- Credential retrieval by ID
- Credential deletion
- Credential rotation
- Input validation (service_name, credential_type, value)
- Security: credential values never leaked in responses
- Fallback to in-memory storage when proxy unavailable
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


def valid_credential_body(
    service_name: str = "openai-prod",
    credential_type: str = "api_key",
    value: str = "sk-test-secret-key-12345",
    **overrides: Any,
) -> dict[str, Any]:
    """Create a valid credential storage body."""
    body: dict[str, Any] = {
        "service_name": service_name,
        "credential_type": credential_type,
        "value": value,
    }
    body.update(overrides)
    return body


def store_credential(
    handler_instance: GatewayCredentialsHandler,
    body: dict[str, Any] | None = None,
) -> tuple[Any, str | None]:
    """Helper to store a credential and return (result, credential_id)."""
    if body is None:
        body = valid_credential_body()
    mock = make_mock_handler(body)
    result = handler_instance.handle_post("/api/v1/gateway/credentials", {}, mock)
    cred_id = None
    if result is not None and result.status_code == 201:
        data = json.loads(result.body)
        cred_id = data.get("credential_id")
    return result, cred_id


@pytest.fixture
def server_context() -> dict[str, Any]:
    """Create empty server context."""
    return {}


@pytest.fixture
def handler(server_context: dict[str, Any]) -> GatewayCredentialsHandler:
    """Create handler with clean context."""
    return GatewayCredentialsHandler(server_context)


# ===========================================================================
# Path Routing Tests
# ===========================================================================


class TestPathRouting:
    """Test path matching and routing."""

    def test_can_handle_credential_paths(self, handler: GatewayCredentialsHandler) -> None:
        """Handler should match /api/v1/gateway/credentials paths."""
        assert handler.can_handle("/api/v1/gateway/credentials")
        assert handler.can_handle("/api/v1/gateway/credentials/")
        assert handler.can_handle("/api/v1/gateway/credentials/cred_abc123")
        assert handler.can_handle("/api/v1/gateway/credentials/cred_abc123/rotate")

    def test_cannot_handle_other_paths(self, handler: GatewayCredentialsHandler) -> None:
        """Handler should not match non-credential paths."""
        assert not handler.can_handle("/api/v1/gateway/agents")
        assert not handler.can_handle("/api/v1/debates")
        assert not handler.can_handle("/api/v1/gateway/channels")
        assert not handler.can_handle("/api/gateway/credentials")  # Missing v1


# ===========================================================================
# Store Credential Tests
# ===========================================================================


class TestStoreCredential:
    """Test POST /api/v1/gateway/credentials."""

    def test_store_credential_success(self, handler: GatewayCredentialsHandler) -> None:
        """Successful storage returns 201 with metadata (no value)."""
        body = valid_credential_body()
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 201

        data = json.loads(result.body)
        assert data["service_name"] == "openai-prod"
        assert data["credential_type"] == "api_key"
        assert "credential_id" in data
        assert "created_at" in data
        assert "message" in data
        # CRITICAL: value must NEVER appear in response
        assert "value" not in data

    def test_store_credential_missing_service_name(
        self, handler: GatewayCredentialsHandler
    ) -> None:
        """Storage without service_name returns 400."""
        body = valid_credential_body()
        del body["service_name"]
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400

    def test_store_credential_missing_value(self, handler: GatewayCredentialsHandler) -> None:
        """Storage without value returns 400."""
        body = valid_credential_body()
        del body["value"]
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400

    def test_store_credential_missing_type(self, handler: GatewayCredentialsHandler) -> None:
        """Storage without credential_type returns 400."""
        body = valid_credential_body()
        del body["credential_type"]
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400

    def test_store_credential_invalid_service_name(
        self, handler: GatewayCredentialsHandler
    ) -> None:
        """Storage with invalid service_name returns 400."""
        body = valid_credential_body(service_name="invalid name with spaces!")
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400

    def test_store_credential_invalid_type(self, handler: GatewayCredentialsHandler) -> None:
        """Storage with unknown credential_type returns 400."""
        body = valid_credential_body(credential_type="unknown_type")
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400
        data = json.loads(result.body)
        assert "credential_type" in data["error"]

    def test_store_credential_empty_value(self, handler: GatewayCredentialsHandler) -> None:
        """Storage with empty value string returns 400."""
        body = valid_credential_body(value="")
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400

    def test_store_credential_invalid_json(self, handler: GatewayCredentialsHandler) -> None:
        """Storage with invalid JSON body returns 400."""
        mock_handler = MagicMock()
        mock_handler.client_address = ("127.0.0.1", 8080)
        mock_handler.headers = {
            "Content-Type": "application/json",
            "Content-Length": "12",
        }
        mock_handler.rfile = MagicMock()
        mock_handler.rfile.read.return_value = b"not valid json"

        result = handler.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# List Credentials Tests
# ===========================================================================


class TestListCredentials:
    """Test GET /api/v1/gateway/credentials."""

    def test_list_credentials_empty(self, handler: GatewayCredentialsHandler) -> None:
        """Listing with no credentials returns empty list."""
        mock_handler = make_mock_handler()

        result = handler.handle("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200

        data = json.loads(result.body)
        assert data["credentials"] == []
        assert data["total"] == 0

    def test_list_credentials_populated(self, handler: GatewayCredentialsHandler) -> None:
        """Listing returns metadata for stored credentials."""
        # Store two credentials
        store_credential(handler, valid_credential_body(service_name="service-a"))
        store_credential(handler, valid_credential_body(service_name="service-b"))

        mock_handler = make_mock_handler()
        result = handler.handle("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200

        data = json.loads(result.body)
        assert data["total"] == 2
        assert len(data["credentials"]) == 2

        service_names = {c["service_name"] for c in data["credentials"]}
        assert "service-a" in service_names
        assert "service-b" in service_names

    def test_list_credentials_never_returns_values(
        self, handler: GatewayCredentialsHandler
    ) -> None:
        """Verify no 'value' field appears in any credential in the list."""
        store_credential(handler, valid_credential_body(service_name="svc-1"))
        store_credential(handler, valid_credential_body(service_name="svc-2"))

        mock_handler = make_mock_handler()
        result = handler.handle("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        data = json.loads(result.body)

        for cred in data["credentials"]:
            assert "value" not in cred, "Credential value must NEVER appear in list response"

        # Also check the raw body for the secret
        raw_body = result.body.decode("utf-8")
        assert "sk-test-secret-key-12345" not in raw_body


# ===========================================================================
# Get Credential Metadata Tests
# ===========================================================================


class TestGetCredentialMetadata:
    """Test GET /api/v1/gateway/credentials/{id}."""

    def test_get_credential_metadata(self, handler: GatewayCredentialsHandler) -> None:
        """Getting an existing credential returns metadata."""
        _, cred_id = store_credential(handler)
        assert cred_id is not None

        mock_handler = make_mock_handler()
        result = handler.handle(f"/api/v1/gateway/credentials/{cred_id}", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200

        data = json.loads(result.body)
        assert data["credential_id"] == cred_id
        assert data["service_name"] == "openai-prod"
        assert data["credential_type"] == "api_key"
        assert data["status"] == "active"
        assert "created_at" in data

    def test_get_credential_not_found(self, handler: GatewayCredentialsHandler) -> None:
        """Getting a non-existent credential returns 404."""
        mock_handler = make_mock_handler()
        result = handler.handle("/api/v1/gateway/credentials/cred_nonexistent", {}, mock_handler)

        assert result is not None
        assert result.status_code == 404

    def test_get_credential_never_returns_value(self, handler: GatewayCredentialsHandler) -> None:
        """Verify no 'value' field in single credential response."""
        _, cred_id = store_credential(handler)
        assert cred_id is not None

        mock_handler = make_mock_handler()
        result = handler.handle(f"/api/v1/gateway/credentials/{cred_id}", {}, mock_handler)

        assert result is not None
        data = json.loads(result.body)
        assert "value" not in data, "Credential value must NEVER appear in get response"

        raw_body = result.body.decode("utf-8")
        assert "sk-test-secret-key-12345" not in raw_body


# ===========================================================================
# Delete Credential Tests
# ===========================================================================


class TestDeleteCredential:
    """Test DELETE /api/v1/gateway/credentials/{id}."""

    def test_delete_credential_success(self, handler: GatewayCredentialsHandler) -> None:
        """Deleting an existing credential removes it and returns 200."""
        _, cred_id = store_credential(handler)
        assert cred_id is not None

        mock_handler = make_mock_handler()
        result = handler.handle_delete(f"/api/v1/gateway/credentials/{cred_id}", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200

        data = json.loads(result.body)
        assert data["credential_id"] == cred_id
        assert "message" in data

        # Verify credential was removed
        assert cred_id not in handler._get_credentials()

    def test_delete_credential_not_found(self, handler: GatewayCredentialsHandler) -> None:
        """Deleting a non-existent credential returns 404."""
        mock_handler = make_mock_handler()
        result = handler.handle_delete(
            "/api/v1/gateway/credentials/cred_nonexistent", {}, mock_handler
        )

        assert result is not None
        assert result.status_code == 404


# ===========================================================================
# Rotate Credential Tests
# ===========================================================================


class TestRotateCredential:
    """Test POST /api/v1/gateway/credentials/{id}/rotate."""

    def test_rotate_credential_success(self, handler: GatewayCredentialsHandler) -> None:
        """Rotating a credential creates a new one and marks old as rotated."""
        _, old_cred_id = store_credential(handler)
        assert old_cred_id is not None

        rotate_body = {"value": "sk-new-rotated-secret-key"}
        mock_handler = make_mock_handler(rotate_body)

        result = handler.handle_post(
            f"/api/v1/gateway/credentials/{old_cred_id}/rotate",
            {},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 201

        data = json.loads(result.body)
        assert "credential_id" in data
        assert data["credential_id"] != old_cred_id
        assert data["previous_credential_id"] == old_cred_id
        assert data["service_name"] == "openai-prod"
        assert data["status"] == "active"
        assert "message" in data
        # CRITICAL: value must NEVER appear in response
        assert "value" not in data

        # Verify old credential is marked as rotated
        credentials = handler._get_credentials()
        assert credentials[old_cred_id]["status"] == "rotated"

        # Verify new credential exists
        new_cred_id = data["credential_id"]
        assert new_cred_id in credentials
        assert credentials[new_cred_id]["status"] == "active"

    def test_rotate_credential_not_found(self, handler: GatewayCredentialsHandler) -> None:
        """Rotating a non-existent credential returns 404."""
        rotate_body = {"value": "sk-new-secret"}
        mock_handler = make_mock_handler(rotate_body)

        result = handler.handle_post(
            "/api/v1/gateway/credentials/cred_nonexistent/rotate",
            {},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 404

    def test_rotate_credential_missing_value(self, handler: GatewayCredentialsHandler) -> None:
        """Rotating without a new value returns 400."""
        _, old_cred_id = store_credential(handler)
        assert old_cred_id is not None

        rotate_body: dict[str, Any] = {}
        mock_handler = make_mock_handler(rotate_body)

        result = handler.handle_post(
            f"/api/v1/gateway/credentials/{old_cred_id}/rotate",
            {},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Credential ID and Timestamps Tests
# ===========================================================================


class TestCredentialProperties:
    """Test credential ID generation and timestamps."""

    def test_credential_has_generated_id(self, handler: GatewayCredentialsHandler) -> None:
        """Generated credential ID starts with 'cred_'."""
        _, cred_id = store_credential(handler)
        assert cred_id is not None
        assert cred_id.startswith("cred_")
        # cred_ prefix + 12 hex chars
        assert len(cred_id) == 5 + 12

    def test_credential_has_timestamps(self, handler: GatewayCredentialsHandler) -> None:
        """Stored credential has created_at timestamp."""
        result, cred_id = store_credential(handler)
        assert result is not None
        assert result.status_code == 201

        data = json.loads(result.body)
        assert "created_at" in data
        # Should be an ISO format timestamp
        assert "T" in data["created_at"]

    def test_store_credential_with_scopes(self, handler: GatewayCredentialsHandler) -> None:
        """Scopes are stored and returned correctly."""
        body = valid_credential_body(scopes=["read", "write"])
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 201

        data = json.loads(result.body)
        assert data["scopes"] == ["read", "write"]

        # Also verify scopes in the stored metadata
        cred_id = data["credential_id"]
        meta = handler._get_credentials()[cred_id]
        assert meta.get("scopes") == ["read", "write"]


# ===========================================================================
# Proxy Unavailable Fallback Tests
# ===========================================================================


class TestProxyFallback:
    """Test behavior when credential proxy is not available."""

    def test_credential_proxy_unavailable(self, server_context: dict[str, Any]) -> None:
        """Handler falls back to in-memory storage when proxy is not in context."""
        # No credential_proxy in server_context
        h = GatewayCredentialsHandler(server_context)

        body = valid_credential_body()
        mock_handler = make_mock_handler(body)

        result = h.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 201

        data = json.loads(result.body)
        cred_id = data["credential_id"]

        # Verify stored in in-memory fallback
        credentials = h._get_credentials()
        assert cred_id in credentials
        assert credentials[cred_id]["service_name"] == "openai-prod"

        # CRITICAL: no value stored in metadata
        assert "value" not in credentials[cred_id]
