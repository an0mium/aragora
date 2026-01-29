"""Tests for decorators module."""

from __future__ import annotations

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import json
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.utils.decorators import (
    generate_trace_id,
    map_exception_to_status,
    validate_params,
    handle_errors,
    auto_error_response,
    log_request,
    has_permission,
    require_storage,
    require_feature,
    with_error_recovery,
    deprecated_endpoint,
    PERMISSION_MATRIX,
)


# =============================================================================
# Test generate_trace_id
# =============================================================================


class TestGenerateTraceId:
    """Tests for generate_trace_id function."""

    def test_generates_string(self):
        """Should generate a string trace ID."""
        trace_id = generate_trace_id()
        assert isinstance(trace_id, str)

    def test_generates_8_char_id(self):
        """Should generate 8 character trace ID."""
        trace_id = generate_trace_id()
        assert len(trace_id) == 8

    def test_generates_unique_ids(self):
        """Should generate unique IDs."""
        ids = {generate_trace_id() for _ in range(100)}
        assert len(ids) == 100


# =============================================================================
# Test map_exception_to_status
# =============================================================================


class TestMapExceptionToStatus:
    """Tests for map_exception_to_status function."""

    def test_maps_file_not_found(self):
        """Should map FileNotFoundError to 404."""
        assert map_exception_to_status(FileNotFoundError()) == 404

    def test_maps_key_error(self):
        """Should map KeyError to 404."""
        assert map_exception_to_status(KeyError()) == 404

    def test_maps_value_error(self):
        """Should map ValueError to 400."""
        assert map_exception_to_status(ValueError()) == 400

    def test_maps_type_error(self):
        """Should map TypeError to 400."""
        assert map_exception_to_status(TypeError()) == 400

    def test_maps_permission_error(self):
        """Should map PermissionError to 403."""
        assert map_exception_to_status(PermissionError()) == 403

    def test_maps_timeout_error(self):
        """Should map TimeoutError to 504."""
        assert map_exception_to_status(TimeoutError()) == 504

    def test_maps_connection_error(self):
        """Should map ConnectionError to 502."""
        assert map_exception_to_status(ConnectionError()) == 502

    def test_returns_default_for_unknown(self):
        """Should return default for unknown exceptions."""

        class CustomError(Exception):
            pass

        assert map_exception_to_status(CustomError(), default=500) == 500


# =============================================================================
# Test validate_params decorator
# =============================================================================


class TestValidateParams:
    """Tests for @validate_params decorator."""

    def test_extracts_int_param(self):
        """Should extract and validate int parameters."""
        extracted = {}

        @validate_params({"limit": (int, 10, 1, 100)})
        def handler(query_params=None, limit=None):
            extracted["limit"] = limit
            return {"limit": limit}

        handler(query_params={"limit": "50"})
        assert extracted["limit"] == 50

    def test_clamps_int_to_min(self):
        """Should clamp int to minimum value."""

        @validate_params({"limit": (int, 10, 5, 100)})
        def handler(query_params=None, limit=None):
            return {"limit": limit}

        result = handler(query_params={"limit": "1"})
        assert result["limit"] == 5

    def test_clamps_int_to_max(self):
        """Should clamp int to maximum value."""

        @validate_params({"limit": (int, 10, 1, 50)})
        def handler(query_params=None, limit=None):
            return {"limit": limit}

        result = handler(query_params={"limit": "100"})
        assert result["limit"] == 50

    def test_extracts_float_param(self):
        """Should extract and validate float parameters."""

        @validate_params({"threshold": (float, 0.5, 0.0, 1.0)})
        def handler(query_params=None, threshold=None):
            return {"threshold": threshold}

        result = handler(query_params={"threshold": "0.75"})
        assert result["threshold"] == 0.75

    def test_extracts_bool_param(self):
        """Should extract and validate bool parameters."""

        @validate_params({"active": (bool, False, None, None)})
        def handler(query_params=None, active=None):
            return {"active": active}

        result = handler(query_params={"active": "true"})
        assert result["active"] is True

    def test_extracts_string_param(self):
        """Should extract and validate string parameters."""

        @validate_params({"name": (str, None, None, 50)})
        def handler(query_params=None, name=None):
            return {"name": name}

        result = handler(query_params={"name": "test"})
        assert result["name"] == "test"

    def test_truncates_string_to_max_length(self):
        """Should truncate string to max length."""

        @validate_params({"name": (str, None, None, 5)})
        def handler(query_params=None, name=None):
            return {"name": name}

        result = handler(query_params={"name": "long_string"})
        assert result["name"] == "long_"

    def test_uses_default_when_missing(self):
        """Should use default when parameter is missing."""

        @validate_params({"limit": (int, 25, 1, 100)})
        def handler(query_params=None, limit=None):
            return {"limit": limit}

        result = handler(query_params={})
        assert result["limit"] == 25


# =============================================================================
# Test handle_errors decorator
# =============================================================================


class TestHandleErrors:
    """Tests for @handle_errors decorator."""

    def test_returns_result_on_success(self):
        """Should return result on success."""

        @handle_errors("test operation")
        def handler():
            return {"success": True}

        result = handler()
        assert result == {"success": True}

    def test_returns_error_response_on_exception(self):
        """Should return error response on exception."""

        @handle_errors("test operation")
        def handler():
            raise ValueError("Invalid input")

        result = handler()
        assert result.status_code == 400

    def test_adds_trace_id_header(self):
        """Should add trace ID header on error."""

        @handle_errors("test operation")
        def handler():
            raise RuntimeError("Failed")

        result = handler()
        assert "X-Trace-Id" in result.headers

    def test_maps_exception_to_status(self):
        """Should map exception to appropriate status."""

        @handle_errors("test operation")
        def handler():
            raise FileNotFoundError("Not found")

        result = handler()
        assert result.status_code == 404


# =============================================================================
# Test auto_error_response decorator
# =============================================================================


class TestAutoErrorResponse:
    """Tests for @auto_error_response decorator."""

    def test_returns_result_on_success(self):
        """Should return result on success."""

        @auto_error_response("test")
        def handler():
            return {"ok": True}

        result = handler()
        assert result == {"ok": True}

    def test_returns_400_on_value_error(self):
        """Should return 400 on ValueError."""

        @auto_error_response("test")
        def handler():
            raise ValueError("Invalid")

        result = handler()
        assert result.status_code == 400

    def test_returns_403_on_permission_error(self):
        """Should return 403 on PermissionError."""

        @auto_error_response("test")
        def handler():
            raise PermissionError("Denied")

        result = handler()
        assert result.status_code == 403


# =============================================================================
# Test log_request decorator
# =============================================================================


class TestLogRequest:
    """Tests for @log_request decorator."""

    def test_returns_result(self):
        """Should return handler result."""

        @log_request("test request")
        def handler():
            return {"data": "test"}

        result = handler()
        assert result == {"data": "test"}

    def test_handles_exceptions(self):
        """Should re-raise exceptions after logging."""

        @log_request("test request")
        def handler():
            raise RuntimeError("Failed")

        with pytest.raises(RuntimeError):
            handler()


# =============================================================================
# Test has_permission
# =============================================================================


class TestHasPermission:
    """Tests for has_permission function."""

    def test_grants_direct_permission(self):
        """Should grant directly assigned permission."""
        assert has_permission("member", "debates:read") is True
        assert has_permission("admin", "debates:delete") is True

    def test_denies_missing_permission(self):
        """Should deny permission not in role."""
        assert has_permission("member", "org:delete") is False

    def test_grants_wildcard_permission(self):
        """Should grant through wildcard permission."""
        assert has_permission("owner", "admin:anything") is True

    def test_returns_false_for_empty_role(self):
        """Should return False for empty role."""
        assert has_permission("", "debates:read") is False
        assert has_permission(None, "debates:read") is False

    def test_returns_false_for_empty_permission(self):
        """Should return False for empty permission."""
        assert has_permission("admin", "") is False
        assert has_permission("admin", None) is False


# =============================================================================
# Test require_storage decorator
# =============================================================================


class TestRequireStorage:
    """Tests for @require_storage decorator."""

    def test_allows_when_storage_available(self):
        """Should allow when storage is available."""

        class Handler:
            def get_storage(self):
                return MagicMock()

            @require_storage
            def action(self):
                return {"success": True}

        handler = Handler()
        result = handler.action()
        assert result == {"success": True}

    def test_returns_503_when_storage_unavailable(self):
        """Should return 503 when storage is unavailable."""

        class Handler:
            def get_storage(self):
                return None

            @require_storage
            def action(self):
                return {"success": True}

        handler = Handler()
        result = handler.action()
        assert result.status_code == 503


# =============================================================================
# Test require_feature decorator
# =============================================================================


class TestRequireFeature:
    """Tests for @require_feature decorator."""

    def test_allows_when_feature_available(self):
        """Should allow when feature check passes."""

        @require_feature(lambda: True, "test feature")
        def handler():
            return {"success": True}

        result = handler()
        assert result == {"success": True}

    def test_returns_error_when_unavailable(self):
        """Should return error when feature unavailable."""

        @require_feature(lambda: False, "test feature", status_code=503)
        def handler():
            return {"success": True}

        result = handler()
        assert result.status_code == 503


# =============================================================================
# Test with_error_recovery decorator
# =============================================================================


class TestWithErrorRecovery:
    """Tests for @with_error_recovery decorator."""

    def test_returns_value_on_success(self):
        """Should return value on success."""

        @with_error_recovery(fallback_value=None)
        def handler():
            return {"data": "test"}

        result = handler()
        assert result == {"data": "test"}

    def test_returns_fallback_on_error(self):
        """Should return fallback on error."""

        @with_error_recovery(fallback_value={"default": True})
        def handler():
            raise RuntimeError("Failed")

        result = handler()
        assert result == {"default": True}


# =============================================================================
# Test deprecated_endpoint decorator
# =============================================================================


class TestDeprecatedEndpoint:
    """Tests for @deprecated_endpoint decorator."""

    def test_returns_result(self):
        """Should return handler result with deprecation headers."""

        @deprecated_endpoint()
        def handler():
            return {"result": "data"}

        result = handler()
        assert result["result"] == "data"
        # Deprecated endpoint adds headers to dict results
        assert "headers" in result

    def test_adds_deprecation_header(self):
        """Should add Deprecation header."""

        @deprecated_endpoint()
        def handler():
            return {"result": "data", "headers": {}}

        result = handler()
        assert result["headers"]["Deprecation"] == "true"

    def test_adds_link_header_for_replacement(self):
        """Should add Link header when replacement specified."""

        @deprecated_endpoint(replacement="/api/v2/resource")
        def handler():
            return {"result": "data", "headers": {}}

        result = handler()
        assert 'rel="successor-version"' in result["headers"]["Link"]


# =============================================================================
# Test PERMISSION_MATRIX constant
# =============================================================================


class TestPermissionMatrix:
    """Tests for PERMISSION_MATRIX constant."""

    def test_is_dict(self):
        """Should be a dictionary."""
        assert isinstance(PERMISSION_MATRIX, dict)

    def test_contains_debate_permissions(self):
        """Should contain debate permissions."""
        assert "debates:read" in PERMISSION_MATRIX
        assert "debates:create" in PERMISSION_MATRIX
        assert "debates:delete" in PERMISSION_MATRIX

    def test_contains_admin_permissions(self):
        """Should contain admin permissions."""
        assert "admin:*" in PERMISSION_MATRIX

    def test_permission_values_are_lists(self):
        """Should have lists as values."""
        for perm, roles in PERMISSION_MATRIX.items():
            assert isinstance(roles, list), f"Permission {perm} should have list value"
