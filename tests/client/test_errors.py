"""
Tests for client SDK error classes.

Tests cover:
- AragoraAPIError base class
- RateLimitError (HTTP 429)
- AuthenticationError (HTTP 401)
- NotFoundError (HTTP 404)
- QuotaExceededError (HTTP 402)
- ValidationError (HTTP 400)
- Error message formatting with suggestions
- Error inheritance hierarchy
"""

from __future__ import annotations

import pytest

from aragora.client.errors import (
    AragoraAPIError,
    AuthenticationError,
    NotFoundError,
    QuotaExceededError,
    RateLimitError,
    ValidationError,
)
from aragora.exceptions import AragoraError


# ============================================================================
# AragoraAPIError (Base Class) Tests
# ============================================================================


class TestAragoraAPIErrorBase:
    """Tests for AragoraAPIError base class."""

    def test_inherits_from_aragora_error(self):
        """Test AragoraAPIError inherits from AragoraError."""
        error = AragoraAPIError("Test error")
        assert isinstance(error, AragoraError)
        assert isinstance(error, Exception)

    def test_default_code(self):
        """Test default error code is UNKNOWN."""
        error = AragoraAPIError("Test error")
        assert error.code == "UNKNOWN"

    def test_default_status_code(self):
        """Test default status code is 500."""
        error = AragoraAPIError("Test error")
        assert error.status_code == 500

    def test_custom_message(self):
        """Test custom error message."""
        error = AragoraAPIError("Custom message")
        assert "Custom message" in str(error)

    def test_custom_code(self):
        """Test custom error code."""
        error = AragoraAPIError("Test", code="CUSTOM_CODE")
        assert error.code == "CUSTOM_CODE"

    def test_custom_status_code(self):
        """Test custom status code."""
        error = AragoraAPIError("Test", status_code=418)
        assert error.status_code == 418

    def test_suggestion_none_by_default(self):
        """Test suggestion is None by default."""
        error = AragoraAPIError("Test error")
        assert error.suggestion is None

    def test_suggestion_appended_to_message(self):
        """Test suggestion is appended to error message."""
        error = AragoraAPIError("Test error", suggestion="Try again later")
        assert "Suggestion: Try again later" in str(error)

    def test_message_without_suggestion(self):
        """Test message format without suggestion."""
        error = AragoraAPIError("Test error")
        assert str(error) == "Test error"

    def test_message_with_suggestion(self):
        """Test message format with suggestion."""
        error = AragoraAPIError("Test error", suggestion="Fix it")
        assert str(error) == "Test error. Suggestion: Fix it"


class TestAragoraAPIErrorAttributes:
    """Tests for AragoraAPIError attribute storage."""

    def test_stores_all_parameters(self):
        """Test all parameters are stored correctly."""
        error = AragoraAPIError(
            "Error message",
            code="TEST_CODE",
            status_code=422,
            suggestion="Test suggestion",
        )

        assert error._base_message == "Error message"
        assert error.code == "TEST_CODE"
        assert error.status_code == 422
        assert error.suggestion == "Test suggestion"

    def test_base_message_preserved(self):
        """Test base message is preserved without suggestion."""
        error = AragoraAPIError("Base message", suggestion="Extra info")
        assert error._base_message == "Base message"


# ============================================================================
# RateLimitError Tests
# ============================================================================


class TestRateLimitError:
    """Tests for RateLimitError (HTTP 429)."""

    def test_inherits_from_api_error(self):
        """Test RateLimitError inherits from AragoraAPIError."""
        error = RateLimitError()
        assert isinstance(error, AragoraAPIError)

    def test_default_message(self):
        """Test default error message."""
        error = RateLimitError()
        assert "Rate limit exceeded" in str(error)

    def test_custom_message(self):
        """Test custom error message."""
        error = RateLimitError("Too many requests")
        assert "Too many requests" in str(error)

    def test_status_code_429(self):
        """Test status code is 429."""
        error = RateLimitError()
        assert error.status_code == 429

    def test_code_rate_limited(self):
        """Test error code is RATE_LIMITED."""
        error = RateLimitError()
        assert error.code == "RATE_LIMITED"

    def test_includes_retry_suggestion(self):
        """Test error includes retry suggestion."""
        error = RateLimitError()
        assert "RetryConfig" in str(error) or "backoff" in str(error)

    def test_retry_after_none_by_default(self):
        """Test retry_after is None by default."""
        error = RateLimitError()
        assert error.retry_after is None

    def test_retry_after_custom_value(self):
        """Test retry_after can be set."""
        error = RateLimitError(retry_after=30.0)
        assert error.retry_after == 30.0

    def test_retry_after_zero(self):
        """Test retry_after can be zero."""
        error = RateLimitError(retry_after=0)
        assert error.retry_after == 0


# ============================================================================
# AuthenticationError Tests
# ============================================================================


class TestAuthenticationError:
    """Tests for AuthenticationError (HTTP 401)."""

    def test_inherits_from_api_error(self):
        """Test AuthenticationError inherits from AragoraAPIError."""
        error = AuthenticationError()
        assert isinstance(error, AragoraAPIError)

    def test_default_message(self):
        """Test default error message."""
        error = AuthenticationError()
        assert "Authentication failed" in str(error)

    def test_custom_message(self):
        """Test custom error message."""
        error = AuthenticationError("Invalid API key")
        assert "Invalid API key" in str(error)

    def test_status_code_401(self):
        """Test status code is 401."""
        error = AuthenticationError()
        assert error.status_code == 401

    def test_code_unauthorized(self):
        """Test error code is UNAUTHORIZED."""
        error = AuthenticationError()
        assert error.code == "UNAUTHORIZED"

    def test_includes_token_suggestion(self):
        """Test error includes API token suggestion."""
        error = AuthenticationError()
        assert "ARAGORA_API_TOKEN" in str(error)


# ============================================================================
# NotFoundError Tests
# ============================================================================


class TestNotFoundError:
    """Tests for NotFoundError (HTTP 404)."""

    def test_inherits_from_api_error(self):
        """Test NotFoundError inherits from AragoraAPIError."""
        error = NotFoundError()
        assert isinstance(error, AragoraAPIError)

    def test_default_message(self):
        """Test default error message."""
        error = NotFoundError()
        assert "not found" in str(error).lower()

    def test_custom_message(self):
        """Test custom error message."""
        error = NotFoundError("Debate abc123 not found")
        assert "Debate abc123 not found" in str(error)

    def test_status_code_404(self):
        """Test status code is 404."""
        error = NotFoundError()
        assert error.status_code == 404

    def test_code_not_found(self):
        """Test error code is NOT_FOUND."""
        error = NotFoundError()
        assert error.code == "NOT_FOUND"

    def test_default_resource_type(self):
        """Test default resource_type is 'resource'."""
        error = NotFoundError()
        assert error.resource_type == "resource"

    def test_custom_resource_type(self):
        """Test custom resource_type."""
        error = NotFoundError("Not found", resource_type="debate")
        assert error.resource_type == "debate"

    def test_suggestion_includes_resource_type(self):
        """Test suggestion includes resource type."""
        error = NotFoundError("Not found", resource_type="debate")
        assert "debate" in str(error)


# ============================================================================
# QuotaExceededError Tests
# ============================================================================


class TestQuotaExceededError:
    """Tests for QuotaExceededError (HTTP 402)."""

    def test_inherits_from_api_error(self):
        """Test QuotaExceededError inherits from AragoraAPIError."""
        error = QuotaExceededError()
        assert isinstance(error, AragoraAPIError)

    def test_default_message(self):
        """Test default error message."""
        error = QuotaExceededError()
        assert "Quota exceeded" in str(error)

    def test_custom_message(self):
        """Test custom error message."""
        error = QuotaExceededError("API quota exhausted")
        assert "API quota exhausted" in str(error)

    def test_status_code_402(self):
        """Test status code is 402."""
        error = QuotaExceededError()
        assert error.status_code == 402

    def test_code_quota_exceeded(self):
        """Test error code is QUOTA_EXCEEDED."""
        error = QuotaExceededError()
        assert error.code == "QUOTA_EXCEEDED"

    def test_includes_upgrade_suggestion(self):
        """Test error includes upgrade suggestion."""
        error = QuotaExceededError()
        assert "plan" in str(error).lower() or "quota" in str(error).lower()


# ============================================================================
# ValidationError Tests
# ============================================================================


class TestValidationError:
    """Tests for ValidationError (HTTP 400)."""

    def test_inherits_from_api_error(self):
        """Test ValidationError inherits from AragoraAPIError."""
        error = ValidationError()
        assert isinstance(error, AragoraAPIError)

    def test_default_message(self):
        """Test default error message."""
        error = ValidationError()
        assert "Validation error" in str(error)

    def test_custom_message(self):
        """Test custom error message."""
        error = ValidationError("Invalid email format")
        assert "Invalid email format" in str(error)

    def test_status_code_400(self):
        """Test status code is 400."""
        error = ValidationError()
        assert error.status_code == 400

    def test_code_validation_error(self):
        """Test error code is VALIDATION_ERROR."""
        error = ValidationError()
        assert error.code == "VALIDATION_ERROR"

    def test_field_none_by_default(self):
        """Test field is None by default."""
        error = ValidationError()
        assert error.field is None

    def test_custom_field(self):
        """Test custom field parameter."""
        error = ValidationError("Invalid value", field="email")
        assert error.field == "email"

    def test_suggestion_includes_field(self):
        """Test suggestion includes field name when provided."""
        error = ValidationError("Invalid value", field="email")
        assert "email" in str(error)

    def test_suggestion_generic_when_no_field(self):
        """Test suggestion is generic when no field provided."""
        error = ValidationError("Invalid value")
        assert "parameters" in str(error)


# ============================================================================
# Error Hierarchy Tests
# ============================================================================


class TestErrorHierarchy:
    """Tests for error class hierarchy."""

    def test_all_errors_catchable_by_aragora_error(self):
        """Test all SDK errors can be caught by AragoraError."""
        errors = [
            AragoraAPIError("test"),
            RateLimitError(),
            AuthenticationError(),
            NotFoundError(),
            QuotaExceededError(),
            ValidationError(),
        ]

        for error in errors:
            try:
                raise error
            except AragoraError:
                pass  # Expected
            except Exception as e:
                pytest.fail(f"{type(error).__name__} not caught by AragoraError: {e}")

    def test_all_errors_catchable_by_api_error(self):
        """Test all specific errors can be caught by AragoraAPIError."""
        errors = [
            RateLimitError(),
            AuthenticationError(),
            NotFoundError(),
            QuotaExceededError(),
            ValidationError(),
        ]

        for error in errors:
            try:
                raise error
            except AragoraAPIError:
                pass  # Expected
            except Exception as e:
                pytest.fail(f"{type(error).__name__} not caught by AragoraAPIError: {e}")

    def test_specific_errors_catchable_individually(self):
        """Test specific errors can be caught individually."""
        error_types = [
            (RateLimitError, RateLimitError()),
            (AuthenticationError, AuthenticationError()),
            (NotFoundError, NotFoundError()),
            (QuotaExceededError, QuotaExceededError()),
            (ValidationError, ValidationError()),
        ]

        for error_class, error_instance in error_types:
            try:
                raise error_instance
            except error_class:
                pass  # Expected
            except Exception as e:
                pytest.fail(f"{error_class.__name__} not caught: {e}")


# ============================================================================
# Error Status Code Mapping Tests
# ============================================================================


class TestErrorStatusCodeMapping:
    """Tests to verify error classes match expected HTTP status codes."""

    @pytest.mark.parametrize(
        "error_class,expected_status",
        [
            (RateLimitError, 429),
            (AuthenticationError, 401),
            (NotFoundError, 404),
            (QuotaExceededError, 402),
            (ValidationError, 400),
        ],
    )
    def test_error_status_codes(self, error_class, expected_status):
        """Test each error class has correct status code."""
        error = error_class()
        assert error.status_code == expected_status

    @pytest.mark.parametrize(
        "error_class,expected_code",
        [
            (RateLimitError, "RATE_LIMITED"),
            (AuthenticationError, "UNAUTHORIZED"),
            (NotFoundError, "NOT_FOUND"),
            (QuotaExceededError, "QUOTA_EXCEEDED"),
            (ValidationError, "VALIDATION_ERROR"),
        ],
    )
    def test_error_codes(self, error_class, expected_code):
        """Test each error class has correct error code."""
        error = error_class()
        assert error.code == expected_code


# ============================================================================
# Error Message Formatting Tests
# ============================================================================


class TestErrorMessageFormatting:
    """Tests for error message string formatting."""

    def test_str_returns_full_message(self):
        """Test __str__ returns full message with suggestion."""
        error = AragoraAPIError("Base", suggestion="Hint")
        result = str(error)
        assert "Base" in result
        assert "Hint" in result

    def test_repr_available(self):
        """Test __repr__ is available for debugging."""
        error = AragoraAPIError("Test error", code="TEST")
        repr_str = repr(error)
        # Should be a valid repr (either default or custom)
        assert repr_str is not None

    def test_error_in_exception_chain(self):
        """Test error works in exception chain."""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise AragoraAPIError("Wrapped error") from e
        except AragoraAPIError as e:
            assert "Wrapped error" in str(e)
            assert e.__cause__ is not None
