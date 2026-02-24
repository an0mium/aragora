"""Tests for the data classification HTTP handler.

Tests cover:
- GET /api/v1/data-classification/policy - Full active policy retrieval
- GET /api/v1/data-classification/policy?level=<level> - Single-level policy
- POST /api/v1/data-classification/classify - Data classification
- POST /api/v1/data-classification/validate - Handling validation
- POST /api/v1/data-classification/enforce - Cross-context enforcement
- Error handling for invalid inputs
- Response envelope structure ({"data": ...})
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.compliance.data_classification import (
    DataClassification,
    DataClassifier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(command: str = "GET", body: dict | None = None):
    """Create a mock HTTP handler with command and optional JSON body."""
    handler = MagicMock()
    handler.command = command
    if body is not None:
        raw = json.dumps(body).encode()
        handler.rfile = MagicMock()
        handler.rfile.read.return_value = raw
        handler.headers = {"Content-Length": str(len(raw))}
    else:
        handler.headers = {}
    return handler


def _parse_data(result):
    """Extract the data payload from a HandlerResult tuple."""
    # HandlerResult is (body_bytes, status_code, content_type)
    # or a dict-like with body/status
    if isinstance(result, tuple):
        body = result[0]
        if isinstance(body, bytes):
            body = body.decode()
        return json.loads(body)
    if isinstance(result, dict):
        body = result.get("body", result)
        if isinstance(body, bytes):
            body = body.decode()
        if isinstance(body, str):
            return json.loads(body)
        return body
    # Try to access as object
    body = getattr(result, "body", result)
    if isinstance(body, bytes):
        body = body.decode()
    if isinstance(body, str):
        return json.loads(body)
    return body


# ---------------------------------------------------------------------------
# Handler import with RBAC bypass
# ---------------------------------------------------------------------------


@pytest.fixture
def handler_instance():
    """Create a DataClassificationHandler with RBAC bypassed."""
    # Reset module-level singletons
    import aragora.server.handlers.data_classification_handler as mod

    mod._classifier = None
    mod._enforcer = None

    from aragora.server.handlers.data_classification_handler import (
        DataClassificationHandler,
    )

    return DataClassificationHandler({})


# =============================================================================
# GET /api/v1/data-classification/policy Tests
# =============================================================================


class TestGetPolicy:
    """Test GET /api/v1/data-classification/policy endpoint."""

    @pytest.mark.asyncio
    async def test_get_full_policy(self, handler_instance):
        """Full policy returns all classification levels and rules."""
        with patch.object(type(handler_instance), "handle", handler_instance.handle.__func__):
            # Call _get_policy directly to bypass RBAC
            result = await handler_instance._get_policy({})

        body = _parse_data(result)
        assert "data" in body
        data = body["data"]
        assert "version" in data
        assert data["version"] == "1.0"
        assert "levels" in data
        assert "public" in data["levels"]
        assert "restricted" in data["levels"]
        assert "policies" in data
        assert "keywords" in data
        assert "sensitivity_order" in data

    @pytest.mark.asyncio
    async def test_get_policy_for_level(self, handler_instance):
        """Querying with ?level=restricted returns only that level's policy."""
        result = await handler_instance._get_policy({"level": "restricted"})

        body = _parse_data(result)
        assert "data" in body
        data = body["data"]
        assert data["classification"] == "restricted"
        assert data["encryption_required"] is True
        assert data["audit_logging"] is True

    @pytest.mark.asyncio
    async def test_get_policy_for_public(self, handler_instance):
        """Querying with ?level=public returns the public policy."""
        result = await handler_instance._get_policy({"level": "public"})

        body = _parse_data(result)
        data = body["data"]
        assert data["classification"] == "public"
        assert data["encryption_required"] is False

    @pytest.mark.asyncio
    async def test_get_policy_invalid_level(self, handler_instance):
        """Invalid level returns 400."""
        result = await handler_instance._get_policy({"level": "invalid_level"})
        body = _parse_data(result)
        # Should be an error response
        if isinstance(result, tuple):
            assert result[1] == 400
        else:
            status = getattr(result, "status", None) or result.get("status", 200)
            assert status == 400

    @pytest.mark.asyncio
    async def test_full_policy_has_all_levels(self, handler_instance):
        """Full policy includes all five classification levels."""
        result = await handler_instance._get_policy({})

        body = _parse_data(result)
        data = body["data"]
        assert len(data["levels"]) == 5
        for level in DataClassification:
            assert level.value in data["levels"]
            assert level.value in data["policies"]


# =============================================================================
# POST /api/v1/data-classification/classify Tests
# =============================================================================


class TestClassifyEndpoint:
    """Test POST /api/v1/data-classification/classify endpoint."""

    @pytest.mark.asyncio
    async def test_classify_public_data(self, handler_instance):
        result = await handler_instance._classify_data({"data": {"title": "Hello World"}})
        body = _parse_data(result)
        assert "data" in body
        assert body["data"]["classification"] == "public"
        assert body["data"]["pii_detected"] is False

    @pytest.mark.asyncio
    async def test_classify_pii_data(self, handler_instance):
        result = await handler_instance._classify_data(
            {"data": {"note": "Contact user@example.com"}}
        )
        body = _parse_data(result)
        assert body["data"]["classification"] == "pii"
        assert body["data"]["pii_detected"] is True
        assert "email" in body["data"]["pii_types"]

    @pytest.mark.asyncio
    async def test_classify_restricted_data(self, handler_instance):
        result = await handler_instance._classify_data({"data": {"api_key": "sk-secret-abc"}})
        body = _parse_data(result)
        assert body["data"]["classification"] == "restricted"

    @pytest.mark.asyncio
    async def test_classify_with_context(self, handler_instance):
        result = await handler_instance._classify_data(
            {"data": {"report": "Q1"}, "context": "financial"}
        )
        body = _parse_data(result)
        assert body["data"]["classification"] == "confidential"
        assert body["data"]["context"] == "financial"

    @pytest.mark.asyncio
    async def test_classify_missing_data(self, handler_instance):
        result = await handler_instance._classify_data({})
        body = _parse_data(result)
        # Should be error
        if isinstance(result, tuple):
            assert result[1] == 400

    @pytest.mark.asyncio
    async def test_classify_invalid_data_type(self, handler_instance):
        result = await handler_instance._classify_data({"data": "not_a_dict"})
        if isinstance(result, tuple):
            assert result[1] == 400

    @pytest.mark.asyncio
    async def test_classify_has_timestamp(self, handler_instance):
        result = await handler_instance._classify_data({"data": {"title": "test"}})
        body = _parse_data(result)
        assert "classified_at" in body["data"]


# =============================================================================
# POST /api/v1/data-classification/validate Tests
# =============================================================================


class TestValidateEndpoint:
    """Test POST /api/v1/data-classification/validate endpoint."""

    @pytest.mark.asyncio
    async def test_validate_public_read_allowed(self, handler_instance):
        result = await handler_instance._validate_handling(
            {
                "data": {"title": "Hello"},
                "classification": "public",
                "operation": "read",
            }
        )
        body = _parse_data(result)
        assert body["data"]["allowed"] is True

    @pytest.mark.asyncio
    async def test_validate_restricted_export_blocked(self, handler_instance):
        result = await handler_instance._validate_handling(
            {
                "data": {"secret": "value"},
                "classification": "restricted",
                "operation": "export",
                "is_encrypted": True,
                "has_consent": True,
            }
        )
        body = _parse_data(result)
        assert body["data"]["allowed"] is False
        assert any("export" in v.lower() for v in body["data"]["violations"])

    @pytest.mark.asyncio
    async def test_validate_missing_data(self, handler_instance):
        result = await handler_instance._validate_handling(
            {
                "classification": "public",
                "operation": "read",
            }
        )
        if isinstance(result, tuple):
            assert result[1] == 400

    @pytest.mark.asyncio
    async def test_validate_missing_classification(self, handler_instance):
        result = await handler_instance._validate_handling(
            {
                "data": {"title": "Hello"},
                "operation": "read",
            }
        )
        if isinstance(result, tuple):
            assert result[1] == 400

    @pytest.mark.asyncio
    async def test_validate_invalid_classification(self, handler_instance):
        result = await handler_instance._validate_handling(
            {
                "data": {"title": "Hello"},
                "classification": "invalid",
                "operation": "read",
            }
        )
        if isinstance(result, tuple):
            assert result[1] == 400

    @pytest.mark.asyncio
    async def test_validate_missing_operation(self, handler_instance):
        result = await handler_instance._validate_handling(
            {
                "data": {"title": "Hello"},
                "classification": "public",
            }
        )
        if isinstance(result, tuple):
            assert result[1] == 400

    @pytest.mark.asyncio
    async def test_validate_encryption_violation(self, handler_instance):
        result = await handler_instance._validate_handling(
            {
                "data": {"secret": "value"},
                "classification": "restricted",
                "operation": "read",
                "is_encrypted": False,
                "has_consent": True,
            }
        )
        body = _parse_data(result)
        assert body["data"]["allowed"] is False
        assert any("Encryption" in v for v in body["data"]["violations"])

    @pytest.mark.asyncio
    async def test_validate_region_violation(self, handler_instance):
        result = await handler_instance._validate_handling(
            {
                "data": {"financial": "data"},
                "classification": "confidential",
                "operation": "read",
                "is_encrypted": True,
                "region": "cn",
            }
        )
        body = _parse_data(result)
        assert body["data"]["allowed"] is False
        assert any("Region" in v for v in body["data"]["violations"])


# =============================================================================
# POST /api/v1/data-classification/enforce Tests
# =============================================================================


class TestEnforceEndpoint:
    """Test POST /api/v1/data-classification/enforce endpoint."""

    @pytest.mark.asyncio
    async def test_enforce_same_level_allowed(self, handler_instance):
        result = await handler_instance._enforce_access(
            {
                "data": {"info": "test"},
                "source_classification": "internal",
                "target_classification": "internal",
            }
        )
        body = _parse_data(result)
        assert body["data"]["allowed"] is True

    @pytest.mark.asyncio
    async def test_enforce_restricted_to_public_blocked(self, handler_instance):
        result = await handler_instance._enforce_access(
            {
                "data": {"secret": "value"},
                "source_classification": "restricted",
                "target_classification": "public",
                "is_encrypted": True,
                "has_consent": True,
            }
        )
        body = _parse_data(result)
        assert body["data"]["allowed"] is False
        assert body["data"]["source_classification"] == "restricted"
        assert body["data"]["target_classification"] == "public"

    @pytest.mark.asyncio
    async def test_enforce_public_to_restricted_allowed(self, handler_instance):
        result = await handler_instance._enforce_access(
            {
                "data": {"title": "Hello"},
                "source_classification": "public",
                "target_classification": "restricted",
            }
        )
        body = _parse_data(result)
        assert body["data"]["allowed"] is True

    @pytest.mark.asyncio
    async def test_enforce_missing_source(self, handler_instance):
        result = await handler_instance._enforce_access(
            {
                "data": {"info": "test"},
                "target_classification": "public",
            }
        )
        if isinstance(result, tuple):
            assert result[1] == 400

    @pytest.mark.asyncio
    async def test_enforce_missing_target(self, handler_instance):
        result = await handler_instance._enforce_access(
            {
                "data": {"info": "test"},
                "source_classification": "internal",
            }
        )
        if isinstance(result, tuple):
            assert result[1] == 400

    @pytest.mark.asyncio
    async def test_enforce_invalid_classification(self, handler_instance):
        result = await handler_instance._enforce_access(
            {
                "data": {"info": "test"},
                "source_classification": "invalid",
                "target_classification": "public",
            }
        )
        if isinstance(result, tuple):
            assert result[1] == 400

    @pytest.mark.asyncio
    async def test_enforce_missing_data(self, handler_instance):
        result = await handler_instance._enforce_access(
            {
                "source_classification": "internal",
                "target_classification": "public",
            }
        )
        if isinstance(result, tuple):
            assert result[1] == 400

    @pytest.mark.asyncio
    async def test_enforce_pii_to_public_blocked(self, handler_instance):
        result = await handler_instance._enforce_access(
            {
                "data": {"email": "user@example.com"},
                "source_classification": "pii",
                "target_classification": "public",
                "is_encrypted": True,
                "has_consent": True,
            }
        )
        body = _parse_data(result)
        assert body["data"]["allowed"] is False


# =============================================================================
# Handler Routing Tests
# =============================================================================


class TestHandlerRouting:
    """Test can_handle and route dispatch."""

    def test_can_handle_policy_path(self, handler_instance):
        assert handler_instance.can_handle("/api/v1/data-classification/policy") is True

    def test_can_handle_classify_path(self, handler_instance):
        assert handler_instance.can_handle("/api/v1/data-classification/classify") is True

    def test_can_handle_validate_path(self, handler_instance):
        assert handler_instance.can_handle("/api/v1/data-classification/validate") is True

    def test_can_handle_enforce_path(self, handler_instance):
        assert handler_instance.can_handle("/api/v1/data-classification/enforce") is True

    def test_cannot_handle_other_path(self, handler_instance):
        assert handler_instance.can_handle("/api/v1/costs") is False

    def test_cannot_handle_partial_prefix(self, handler_instance):
        assert handler_instance.can_handle("/api/v1/data") is False

    def test_routes_list(self, handler_instance):
        assert "/api/v1/data-classification/policy" in handler_instance.ROUTES
        assert "/api/v1/data-classification/classify" in handler_instance.ROUTES
        assert "/api/v1/data-classification/validate" in handler_instance.ROUTES
        assert "/api/v1/data-classification/enforce" in handler_instance.ROUTES
