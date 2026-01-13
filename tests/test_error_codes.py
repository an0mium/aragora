"""
Tests for error codes and standardized error responses.

Tests:
- ErrorCode constants
- ERROR_CODE_STATUS_MAP mapping
- get_status_for_code helper
- Integration with error_response
"""

import pytest

from aragora.server.error_codes import (
    ErrorCode,
    ERROR_CODE_STATUS_MAP,
    get_status_for_code,
)
from aragora.server.handlers.utils.responses import error_response


class TestErrorCodeConstants:
    """Tests for ErrorCode class constants."""

    def test_auth_codes_exist(self):
        """Test authentication error codes are defined."""
        assert ErrorCode.AUTH_REQUIRED == "AUTH_REQUIRED"
        assert ErrorCode.AUTH_INVALID_TOKEN == "AUTH_INVALID_TOKEN"
        assert ErrorCode.AUTH_EXPIRED == "AUTH_EXPIRED"
        assert ErrorCode.AUTH_INVALID_CREDENTIALS == "AUTH_INVALID_CREDENTIALS"
        assert ErrorCode.AUTH_ACCOUNT_LOCKED == "AUTH_ACCOUNT_LOCKED"
        assert ErrorCode.AUTH_MFA_REQUIRED == "AUTH_MFA_REQUIRED"
        assert ErrorCode.AUTH_MFA_INVALID == "AUTH_MFA_INVALID"

    def test_authorization_codes_exist(self):
        """Test authorization error codes are defined."""
        assert ErrorCode.FORBIDDEN == "FORBIDDEN"
        assert ErrorCode.PERMISSION_DENIED == "PERMISSION_DENIED"
        assert ErrorCode.ORG_ACCESS_DENIED == "ORG_ACCESS_DENIED"
        assert ErrorCode.ROLE_INSUFFICIENT == "ROLE_INSUFFICIENT"

    def test_validation_codes_exist(self):
        """Test validation error codes are defined."""
        assert ErrorCode.VALIDATION_ERROR == "VALIDATION_ERROR"
        assert ErrorCode.INVALID_REQUEST == "INVALID_REQUEST"
        assert ErrorCode.INVALID_JSON == "INVALID_JSON"
        assert ErrorCode.MISSING_FIELD == "MISSING_FIELD"
        assert ErrorCode.INVALID_FIELD == "INVALID_FIELD"
        assert ErrorCode.INVALID_QUERY_PARAM == "INVALID_QUERY_PARAM"
        assert ErrorCode.INVALID_PATH == "INVALID_PATH"

    def test_resource_codes_exist(self):
        """Test resource error codes are defined."""
        assert ErrorCode.NOT_FOUND == "NOT_FOUND"
        assert ErrorCode.DEBATE_NOT_FOUND == "DEBATE_NOT_FOUND"
        assert ErrorCode.AGENT_NOT_FOUND == "AGENT_NOT_FOUND"
        assert ErrorCode.USER_NOT_FOUND == "USER_NOT_FOUND"
        assert ErrorCode.ORG_NOT_FOUND == "ORG_NOT_FOUND"
        assert ErrorCode.CONFLICT == "CONFLICT"
        assert ErrorCode.ALREADY_EXISTS == "ALREADY_EXISTS"
        assert ErrorCode.GONE == "GONE"

    def test_rate_limit_codes_exist(self):
        """Test rate limiting error codes are defined."""
        assert ErrorCode.RATE_LIMITED == "RATE_LIMITED"
        assert ErrorCode.QUOTA_EXCEEDED == "QUOTA_EXCEEDED"
        assert ErrorCode.QUOTA_INSUFFICIENT == "QUOTA_INSUFFICIENT"

    def test_server_error_codes_exist(self):
        """Test server error codes are defined."""
        assert ErrorCode.INTERNAL_ERROR == "INTERNAL_ERROR"
        assert ErrorCode.DATABASE_ERROR == "DATABASE_ERROR"
        assert ErrorCode.EXTERNAL_SERVICE_ERROR == "EXTERNAL_SERVICE_ERROR"
        assert ErrorCode.SERVICE_UNAVAILABLE == "SERVICE_UNAVAILABLE"
        assert ErrorCode.FEATURE_UNAVAILABLE == "FEATURE_UNAVAILABLE"
        assert ErrorCode.MAINTENANCE == "MAINTENANCE"

    def test_content_error_codes_exist(self):
        """Test content error codes are defined."""
        assert ErrorCode.PAYLOAD_TOO_LARGE == "PAYLOAD_TOO_LARGE"
        assert ErrorCode.UNSUPPORTED_MEDIA_TYPE == "UNSUPPORTED_MEDIA_TYPE"
        assert ErrorCode.UNPROCESSABLE_ENTITY == "UNPROCESSABLE_ENTITY"


class TestErrorCodeStatusMap:
    """Tests for ERROR_CODE_STATUS_MAP."""

    def test_auth_errors_map_to_401(self):
        """Test authentication errors map to 401."""
        assert ERROR_CODE_STATUS_MAP[ErrorCode.AUTH_REQUIRED] == 401
        assert ERROR_CODE_STATUS_MAP[ErrorCode.AUTH_INVALID_TOKEN] == 401
        assert ERROR_CODE_STATUS_MAP[ErrorCode.AUTH_EXPIRED] == 401
        assert ERROR_CODE_STATUS_MAP[ErrorCode.AUTH_INVALID_CREDENTIALS] == 401

    def test_authorization_errors_map_to_403(self):
        """Test authorization errors map to 403."""
        assert ERROR_CODE_STATUS_MAP[ErrorCode.FORBIDDEN] == 403
        assert ERROR_CODE_STATUS_MAP[ErrorCode.PERMISSION_DENIED] == 403
        assert ERROR_CODE_STATUS_MAP[ErrorCode.ORG_ACCESS_DENIED] == 403
        assert ERROR_CODE_STATUS_MAP[ErrorCode.ROLE_INSUFFICIENT] == 403

    def test_validation_errors_map_to_400(self):
        """Test validation errors map to 400."""
        assert ERROR_CODE_STATUS_MAP[ErrorCode.VALIDATION_ERROR] == 400
        assert ERROR_CODE_STATUS_MAP[ErrorCode.INVALID_REQUEST] == 400
        assert ERROR_CODE_STATUS_MAP[ErrorCode.INVALID_JSON] == 400
        assert ERROR_CODE_STATUS_MAP[ErrorCode.MISSING_FIELD] == 400

    def test_not_found_errors_map_to_404(self):
        """Test not found errors map to 404."""
        assert ERROR_CODE_STATUS_MAP[ErrorCode.NOT_FOUND] == 404
        assert ERROR_CODE_STATUS_MAP[ErrorCode.DEBATE_NOT_FOUND] == 404
        assert ERROR_CODE_STATUS_MAP[ErrorCode.AGENT_NOT_FOUND] == 404
        assert ERROR_CODE_STATUS_MAP[ErrorCode.USER_NOT_FOUND] == 404

    def test_conflict_errors_map_to_409(self):
        """Test conflict errors map to 409."""
        assert ERROR_CODE_STATUS_MAP[ErrorCode.CONFLICT] == 409
        assert ERROR_CODE_STATUS_MAP[ErrorCode.ALREADY_EXISTS] == 409

    def test_rate_limit_errors_map_to_429(self):
        """Test rate limit errors map to 429."""
        assert ERROR_CODE_STATUS_MAP[ErrorCode.RATE_LIMITED] == 429
        assert ERROR_CODE_STATUS_MAP[ErrorCode.QUOTA_EXCEEDED] == 429
        assert ERROR_CODE_STATUS_MAP[ErrorCode.QUOTA_INSUFFICIENT] == 429

    def test_server_errors_map_to_500_or_503(self):
        """Test server errors map to 500 or 503."""
        assert ERROR_CODE_STATUS_MAP[ErrorCode.INTERNAL_ERROR] == 500
        assert ERROR_CODE_STATUS_MAP[ErrorCode.DATABASE_ERROR] == 500
        assert ERROR_CODE_STATUS_MAP[ErrorCode.SERVICE_UNAVAILABLE] == 503
        assert ERROR_CODE_STATUS_MAP[ErrorCode.FEATURE_UNAVAILABLE] == 503


class TestGetStatusForCode:
    """Tests for get_status_for_code helper."""

    def test_returns_correct_status_for_known_code(self):
        """Test returns mapped status for known codes."""
        assert get_status_for_code(ErrorCode.AUTH_REQUIRED) == 401
        assert get_status_for_code(ErrorCode.FORBIDDEN) == 403
        assert get_status_for_code(ErrorCode.NOT_FOUND) == 404
        assert get_status_for_code(ErrorCode.RATE_LIMITED) == 429
        assert get_status_for_code(ErrorCode.INTERNAL_ERROR) == 500

    def test_returns_400_for_unknown_code(self):
        """Test returns 400 for unknown error codes."""
        assert get_status_for_code("UNKNOWN_ERROR") == 400
        assert get_status_for_code("CUSTOM_ERROR") == 400

    def test_handles_none_gracefully(self):
        """Test handles None gracefully."""
        # Should not raise, returns default
        result = get_status_for_code(None)
        assert result == 400


class TestErrorResponseIntegration:
    """Tests for error_response with error codes."""

    def test_simple_error_format(self):
        """Test simple error format without code."""
        result = error_response("Something went wrong", 400)

        import json

        body = json.loads(result.body)

        assert result.status_code == 400
        assert body == {"error": "Something went wrong"}

    def test_structured_error_with_code(self):
        """Test structured error format with code."""
        result = error_response(
            "Invalid email format",
            status=400,
            code=ErrorCode.VALIDATION_ERROR,
        )

        import json

        body = json.loads(result.body)

        assert result.status_code == 400
        assert "error" in body
        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert body["error"]["message"] == "Invalid email format"

    def test_structured_error_with_details(self):
        """Test structured error format with details."""
        result = error_response(
            "Field validation failed",
            status=400,
            code=ErrorCode.INVALID_FIELD,
            details={"field": "email", "reason": "invalid format"},
        )

        import json

        body = json.loads(result.body)

        assert body["error"]["code"] == "INVALID_FIELD"
        assert body["error"]["details"]["field"] == "email"
        assert body["error"]["details"]["reason"] == "invalid format"

    def test_structured_error_with_trace_id(self):
        """Test structured error format with trace ID."""
        result = error_response(
            "Internal error",
            status=500,
            code=ErrorCode.INTERNAL_ERROR,
            trace_id="trace-abc-123",
        )

        import json

        body = json.loads(result.body)

        assert body["error"]["code"] == "INTERNAL_ERROR"
        assert body["error"]["trace_id"] == "trace-abc-123"

    def test_structured_error_with_suggestion(self):
        """Test structured error format with suggestion."""
        result = error_response(
            "Rate limit exceeded",
            status=429,
            code=ErrorCode.RATE_LIMITED,
            suggestion="Wait 60 seconds before retrying",
        )

        import json

        body = json.loads(result.body)

        assert body["error"]["code"] == "RATE_LIMITED"
        assert body["error"]["suggestion"] == "Wait 60 seconds before retrying"

    def test_all_fields_combined(self):
        """Test structured error with all fields."""
        result = error_response(
            "Quota exceeded",
            status=429,
            code=ErrorCode.QUOTA_EXCEEDED,
            trace_id="trace-xyz",
            suggestion="Upgrade your plan",
            details={"limit": 100, "used": 100, "remaining": 0},
        )

        import json

        body = json.loads(result.body)

        error_obj = body["error"]
        assert error_obj["code"] == "QUOTA_EXCEEDED"
        assert error_obj["message"] == "Quota exceeded"
        assert error_obj["trace_id"] == "trace-xyz"
        assert error_obj["suggestion"] == "Upgrade your plan"
        assert error_obj["details"]["limit"] == 100
        assert error_obj["details"]["remaining"] == 0


class TestErrorCodeConsistency:
    """Tests for error code consistency and naming conventions."""

    def test_all_codes_are_uppercase(self):
        """Test that all error codes are uppercase."""
        for attr_name in dir(ErrorCode):
            if not attr_name.startswith("_"):
                value = getattr(ErrorCode, attr_name)
                if isinstance(value, str):
                    assert value == value.upper(), f"{attr_name} is not uppercase"

    def test_all_codes_use_underscores(self):
        """Test that all error codes use underscores (not hyphens)."""
        for attr_name in dir(ErrorCode):
            if not attr_name.startswith("_"):
                value = getattr(ErrorCode, attr_name)
                if isinstance(value, str):
                    assert "-" not in value, f"{attr_name} contains hyphen"

    def test_all_mapped_codes_are_defined(self):
        """Test that all codes in map are defined in ErrorCode."""
        for code in ERROR_CODE_STATUS_MAP.keys():
            # Check code is defined as a constant
            found = False
            for attr_name in dir(ErrorCode):
                if not attr_name.startswith("_"):
                    if getattr(ErrorCode, attr_name) == code:
                        found = True
                        break
            assert found, f"Code {code} in map but not defined in ErrorCode"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
