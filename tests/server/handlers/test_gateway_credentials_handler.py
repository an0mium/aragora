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


# ===========================================================================
# Circuit Breaker Tests
# ===========================================================================


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_exists(self) -> None:
        """Test that circuit breaker can be retrieved."""
        from aragora.server.handlers.gateway_credentials_handler import (
            get_gateway_credentials_circuit_breaker,
        )

        cb = get_gateway_credentials_circuit_breaker()
        assert cb is not None
        assert cb.name == "gateway_credentials_handler"

    def test_circuit_breaker_status(self) -> None:
        """Test that circuit breaker status can be retrieved."""
        from aragora.server.handlers.gateway_credentials_handler import (
            get_gateway_credentials_circuit_breaker_status,
        )

        status = get_gateway_credentials_circuit_breaker_status()
        assert isinstance(status, dict)
        # Status dict has config, entity_mode, and single_mode keys
        assert "config" in status or "single_mode" in status

    def test_circuit_breaker_reset(self) -> None:
        """Test that circuit breaker can be reset."""
        from aragora.server.handlers.gateway_credentials_handler import (
            get_gateway_credentials_circuit_breaker,
            reset_gateway_credentials_circuit_breaker,
        )

        cb = get_gateway_credentials_circuit_breaker()
        reset_gateway_credentials_circuit_breaker()
        assert cb._single_failures == 0

    def test_circuit_breaker_threshold(self) -> None:
        """Test circuit breaker has correct failure threshold."""
        from aragora.server.handlers.gateway_credentials_handler import (
            get_gateway_credentials_circuit_breaker,
        )

        cb = get_gateway_credentials_circuit_breaker()
        assert cb.failure_threshold == 5

    def test_circuit_breaker_cooldown(self) -> None:
        """Test circuit breaker has correct cooldown."""
        from aragora.server.handlers.gateway_credentials_handler import (
            get_gateway_credentials_circuit_breaker,
        )

        cb = get_gateway_credentials_circuit_breaker()
        assert cb.cooldown_seconds == 30.0


# ===========================================================================
# Service Name Validation Tests
# ===========================================================================


class TestServiceNameValidation:
    """Test service_name validation edge cases."""

    def test_service_name_with_numbers(self, handler: GatewayCredentialsHandler) -> None:
        """Service name with numbers should be valid."""
        body = valid_credential_body(service_name="service123")
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 201

    def test_service_name_with_hyphens(self, handler: GatewayCredentialsHandler) -> None:
        """Service name with hyphens should be valid."""
        body = valid_credential_body(service_name="my-service-name")
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 201

    def test_service_name_starting_with_hyphen(self, handler: GatewayCredentialsHandler) -> None:
        """Service name starting with hyphen should be invalid."""
        body = valid_credential_body(service_name="-invalid")
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400

    def test_service_name_max_length(self, handler: GatewayCredentialsHandler) -> None:
        """Service name at max length (128) should be valid."""
        body = valid_credential_body(service_name="a" * 128)
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 201

    def test_service_name_exceeds_max_length(self, handler: GatewayCredentialsHandler) -> None:
        """Service name exceeding max length should be invalid."""
        body = valid_credential_body(service_name="a" * 129)
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Credential Type Tests
# ===========================================================================


class TestCredentialTypeValidation:
    """Test credential_type validation."""

    def test_oauth_token_type(self, handler: GatewayCredentialsHandler) -> None:
        """oauth_token credential type should be valid."""
        body = valid_credential_body(credential_type="oauth_token")
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 201

    def test_bearer_token_type(self, handler: GatewayCredentialsHandler) -> None:
        """bearer_token credential type should be valid."""
        body = valid_credential_body(credential_type="bearer_token")
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 201

    def test_basic_auth_type(self, handler: GatewayCredentialsHandler) -> None:
        """basic_auth credential type should be valid."""
        body = valid_credential_body(credential_type="basic_auth")
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 201

    def test_custom_type(self, handler: GatewayCredentialsHandler) -> None:
        """custom credential type should be valid."""
        body = valid_credential_body(credential_type="custom")
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 201

    def test_all_valid_types(self, handler: GatewayCredentialsHandler) -> None:
        """All valid credential types should be accepted."""
        for cred_type in VALID_CREDENTIAL_TYPES:
            h = GatewayCredentialsHandler({})  # Fresh handler for each
            # Service name must be alphanumeric with hyphens only (no underscores)
            safe_name = cred_type.replace("_", "-")
            body = valid_credential_body(credential_type=cred_type, service_name=f"svc-{safe_name}")
            mock_handler = make_mock_handler(body)

            result = h.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

            assert result is not None, f"Failed for type: {cred_type}"
            assert result.status_code == 201, f"Failed for type: {cred_type}"


# ===========================================================================
# Value Validation Tests
# ===========================================================================


class TestValueValidation:
    """Test credential value validation."""

    def test_whitespace_only_value(self, handler: GatewayCredentialsHandler) -> None:
        """Whitespace-only value should be rejected."""
        body = valid_credential_body(value="   ")
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400

    def test_non_string_value(self, handler: GatewayCredentialsHandler) -> None:
        """Non-string value should be rejected."""
        body = valid_credential_body()
        body["value"] = 12345  # Integer instead of string
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Optional Fields Tests
# ===========================================================================


class TestOptionalFields:
    """Test optional credential fields."""

    def test_store_with_tenant_id(self, handler: GatewayCredentialsHandler) -> None:
        """Credential with tenant_id should be stored correctly."""
        body = valid_credential_body(tenant_id="tenant-123")
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 201

        data = json.loads(result.body)
        assert data["tenant_id"] == "tenant-123"

    def test_store_with_expires_at(self, handler: GatewayCredentialsHandler) -> None:
        """Credential with expires_at should be stored correctly."""
        body = valid_credential_body(expires_at="2026-12-31T23:59:59Z")
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 201

        data = json.loads(result.body)
        assert data["expires_at"] == "2026-12-31T23:59:59Z"

    def test_store_with_metadata(self, handler: GatewayCredentialsHandler) -> None:
        """Credential with metadata should be stored correctly."""
        body = valid_credential_body(metadata={"env": "production", "region": "us-east"})
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        assert result.status_code == 201

        # Check metadata is stored in internal store
        data = json.loads(result.body)
        cred_id = data["credential_id"]
        meta = handler._get_credentials()[cred_id]
        assert meta["metadata"] == {"env": "production", "region": "us-east"}

    def test_get_credential_returns_optional_fields(
        self, handler: GatewayCredentialsHandler
    ) -> None:
        """Getting credential should return all optional fields."""
        body = valid_credential_body(
            tenant_id="tenant-456",
            scopes=["read", "write"],
            expires_at="2027-01-01T00:00:00Z",
            metadata={"key": "value"},
        )
        _, cred_id = store_credential(handler, body)
        assert cred_id is not None

        mock_handler = make_mock_handler()
        result = handler.handle(f"/api/v1/gateway/credentials/{cred_id}", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200

        data = json.loads(result.body)
        assert data["tenant_id"] == "tenant-456"
        assert data["scopes"] == ["read", "write"]
        assert data["expires_at"] == "2027-01-01T00:00:00Z"
        assert data["metadata"] == {"key": "value"}


# ===========================================================================
# Rotation Edge Cases Tests
# ===========================================================================


class TestRotationEdgeCases:
    """Test credential rotation edge cases."""

    def test_rotate_preserves_service_name(self, handler: GatewayCredentialsHandler) -> None:
        """Rotated credential preserves original service_name."""
        _, old_cred_id = store_credential(handler, valid_credential_body(service_name="my-service"))
        assert old_cred_id is not None

        rotate_body = {"value": "sk-new-rotated-value"}
        mock_handler = make_mock_handler(rotate_body)

        result = handler.handle_post(
            f"/api/v1/gateway/credentials/{old_cred_id}/rotate",
            {},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 201

        data = json.loads(result.body)
        assert data["service_name"] == "my-service"

    def test_rotate_preserves_credential_type(self, handler: GatewayCredentialsHandler) -> None:
        """Rotated credential preserves original credential_type."""
        _, old_cred_id = store_credential(
            handler, valid_credential_body(credential_type="oauth_token")
        )
        assert old_cred_id is not None

        rotate_body = {"value": "new-oauth-token"}
        mock_handler = make_mock_handler(rotate_body)

        result = handler.handle_post(
            f"/api/v1/gateway/credentials/{old_cred_id}/rotate",
            {},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 201

        data = json.loads(result.body)
        assert data["credential_type"] == "oauth_token"

    def test_rotate_empty_value_rejected(self, handler: GatewayCredentialsHandler) -> None:
        """Rotation with empty value should be rejected."""
        _, old_cred_id = store_credential(handler)
        assert old_cred_id is not None

        rotate_body = {"value": ""}
        mock_handler = make_mock_handler(rotate_body)

        result = handler.handle_post(
            f"/api/v1/gateway/credentials/{old_cred_id}/rotate",
            {},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 400

    def test_rotate_whitespace_value_rejected(self, handler: GatewayCredentialsHandler) -> None:
        """Rotation with whitespace-only value should be rejected."""
        _, old_cred_id = store_credential(handler)
        assert old_cred_id is not None

        rotate_body = {"value": "   "}
        mock_handler = make_mock_handler(rotate_body)

        result = handler.handle_post(
            f"/api/v1/gateway/credentials/{old_cred_id}/rotate",
            {},
            mock_handler,
        )

        assert result is not None
        assert result.status_code == 400


# ===========================================================================
# Path Extraction Tests
# ===========================================================================


class TestPathExtraction:
    """Test credential ID extraction from paths."""

    def test_extract_credential_id_basic(self, handler: GatewayCredentialsHandler) -> None:
        """Test basic credential ID extraction."""
        cred_id = handler._extract_credential_id("/api/v1/gateway/credentials/cred_abc123")
        assert cred_id == "cred_abc123"

    def test_extract_credential_id_with_trailing_slash(
        self, handler: GatewayCredentialsHandler
    ) -> None:
        """Test credential ID extraction with trailing slash."""
        cred_id = handler._extract_credential_id("/api/v1/gateway/credentials/cred_abc123/")
        assert cred_id == "cred_abc123"

    def test_extract_credential_id_from_rotate_path(
        self, handler: GatewayCredentialsHandler
    ) -> None:
        """Test credential ID extraction from rotate path."""
        cred_id = handler._extract_credential_id("/api/v1/gateway/credentials/cred_abc123/rotate")
        assert cred_id == "cred_abc123"

    def test_is_rotate_path_positive(self, handler: GatewayCredentialsHandler) -> None:
        """Test rotate path detection - positive case."""
        assert handler._is_rotate_path("/api/v1/gateway/credentials/cred_abc123/rotate")

    def test_is_rotate_path_negative(self, handler: GatewayCredentialsHandler) -> None:
        """Test rotate path detection - negative case."""
        assert not handler._is_rotate_path("/api/v1/gateway/credentials/cred_abc123")


# ===========================================================================
# Handler Method Routing Tests
# ===========================================================================


class TestHandlerMethodRouting:
    """Test that handlers correctly route to sub-methods."""

    def test_handle_returns_none_for_invalid_path(self, handler: GatewayCredentialsHandler) -> None:
        """Test handle returns None for non-credential paths."""
        mock_handler = make_mock_handler()
        result = handler.handle("/api/v1/debates", {}, mock_handler)
        assert result is None

    def test_handle_post_returns_none_for_invalid_path(
        self, handler: GatewayCredentialsHandler
    ) -> None:
        """Test handle_post returns None for non-credential paths."""
        mock_handler = make_mock_handler(valid_credential_body())
        result = handler.handle_post("/api/v1/debates", {}, mock_handler)
        assert result is None

    def test_handle_delete_returns_none_for_invalid_path(
        self, handler: GatewayCredentialsHandler
    ) -> None:
        """Test handle_delete returns None for non-credential paths."""
        mock_handler = make_mock_handler()
        result = handler.handle_delete("/api/v1/debates", {}, mock_handler)
        assert result is None

    def test_handle_delete_returns_none_for_base_path(
        self, handler: GatewayCredentialsHandler
    ) -> None:
        """Test handle_delete returns None for base credentials path."""
        mock_handler = make_mock_handler()
        result = handler.handle_delete("/api/v1/gateway/credentials", {}, mock_handler)
        assert result is None


# ===========================================================================
# Credential Metadata Helper Tests
# ===========================================================================


class TestCredentialMetadataHelper:
    """Test credential metadata helper function."""

    def test_make_credential_metadata_basic(self, handler: GatewayCredentialsHandler) -> None:
        """Test basic metadata creation."""
        meta = handler._make_credential_metadata(
            credential_id="cred_test",
            service_name="test-service",
            credential_type="api_key",
        )

        assert meta["credential_id"] == "cred_test"
        assert meta["service_name"] == "test-service"
        assert meta["credential_type"] == "api_key"
        assert meta["status"] == "active"
        assert "created_at" in meta

    def test_make_credential_metadata_with_optionals(
        self, handler: GatewayCredentialsHandler
    ) -> None:
        """Test metadata creation with optional fields."""
        meta = handler._make_credential_metadata(
            credential_id="cred_test",
            service_name="test-service",
            credential_type="api_key",
            tenant_id="tenant-1",
            scopes=["read"],
            expires_at="2027-01-01T00:00:00Z",
            metadata={"env": "test"},
            status="rotated",
        )

        assert meta["tenant_id"] == "tenant-1"
        assert meta["scopes"] == ["read"]
        assert meta["expires_at"] == "2027-01-01T00:00:00Z"
        assert meta["metadata"] == {"env": "test"}
        assert meta["status"] == "rotated"


# ===========================================================================
# Security Tests - Value Never Leaked
# ===========================================================================


class TestSecurityValueNeverLeaked:
    """Critical security tests to ensure credential values are never leaked."""

    def test_store_response_no_value(self, handler: GatewayCredentialsHandler) -> None:
        """Store response must never contain value field."""
        body = valid_credential_body(value="super-secret-key-12345")
        mock_handler = make_mock_handler(body)

        result = handler.handle_post("/api/v1/gateway/credentials", {}, mock_handler)

        assert result is not None
        raw_body = result.body.decode("utf-8") if isinstance(result.body, bytes) else result.body
        assert "super-secret-key-12345" not in raw_body

    def test_rotate_response_no_value(self, handler: GatewayCredentialsHandler) -> None:
        """Rotate response must never contain value field."""
        _, old_cred_id = store_credential(handler)
        assert old_cred_id is not None

        rotate_body = {"value": "new-super-secret-key-67890"}
        mock_handler = make_mock_handler(rotate_body)

        result = handler.handle_post(
            f"/api/v1/gateway/credentials/{old_cred_id}/rotate",
            {},
            mock_handler,
        )

        assert result is not None
        raw_body = result.body.decode("utf-8") if isinstance(result.body, bytes) else result.body
        assert "new-super-secret-key-67890" not in raw_body

    def test_internal_storage_no_value(self, handler: GatewayCredentialsHandler) -> None:
        """Internal metadata storage must never contain value field."""
        body = valid_credential_body(value="stored-secret-value")
        _, cred_id = store_credential(handler, body)
        assert cred_id is not None

        credentials = handler._get_credentials()
        meta = credentials[cred_id]

        # Check all keys in metadata
        assert "value" not in meta
        # Also check no value string in any field
        for key, val in meta.items():
            if isinstance(val, str):
                assert "stored-secret-value" not in val
