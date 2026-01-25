"""Tests for Aragora SDK exceptions."""

from __future__ import annotations

import pytest

from aragora_client.exceptions import (
    AragoraAuthenticationError,
    AragoraConnectionError,
    AragoraError,
    AragoraNotFoundError,
    AragoraTimeoutError,
    AragoraValidationError,
)


class TestAragoraError:
    """Tests for the base AragoraError exception."""

    def test_basic_initialization(self) -> None:
        """Test basic error creation with message only."""
        error = AragoraError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.code is None
        assert error.status is None
        assert error.details == {}

    def test_full_initialization(self) -> None:
        """Test error creation with all parameters."""
        details = {"field": "email", "reason": "invalid format"}
        error = AragoraError(
            "Validation failed",
            code="VALIDATION_ERROR",
            status=400,
            details=details,
        )
        assert error.message == "Validation failed"
        assert error.code == "VALIDATION_ERROR"
        assert error.status == 400
        assert error.details == details

    def test_repr(self) -> None:
        """Test string representation of error."""
        error = AragoraError("Test error", code="TEST_CODE", status=500)
        expected = "AragoraError('Test error', code='TEST_CODE', status=500)"
        assert repr(error) == expected

    def test_repr_without_code_or_status(self) -> None:
        """Test repr when code and status are None."""
        error = AragoraError("Test error")
        expected = "AragoraError('Test error', code=None, status=None)"
        assert repr(error) == expected

    def test_is_exception(self) -> None:
        """Test that AragoraError is a proper exception."""
        error = AragoraError("Test")
        assert isinstance(error, Exception)
        with pytest.raises(AragoraError):
            raise error

    def test_details_default_empty_dict(self) -> None:
        """Test that details defaults to empty dict not None."""
        error = AragoraError("Test", details=None)
        assert error.details == {}
        assert isinstance(error.details, dict)


class TestAragoraConnectionError:
    """Tests for AragoraConnectionError."""

    def test_default_message(self) -> None:
        """Test default error message."""
        error = AragoraConnectionError()
        assert error.message == "Failed to connect to Aragora server"
        assert error.code == "CONNECTION_ERROR"
        assert error.status is None

    def test_custom_message(self) -> None:
        """Test custom error message."""
        error = AragoraConnectionError("Network unreachable")
        assert error.message == "Network unreachable"
        assert error.code == "CONNECTION_ERROR"

    def test_inheritance(self) -> None:
        """Test that it inherits from AragoraError."""
        error = AragoraConnectionError()
        assert isinstance(error, AragoraError)
        assert isinstance(error, Exception)


class TestAragoraAuthenticationError:
    """Tests for AragoraAuthenticationError."""

    def test_default_message(self) -> None:
        """Test default error message."""
        error = AragoraAuthenticationError()
        assert error.message == "Authentication failed"
        assert error.code == "AUTHENTICATION_ERROR"
        assert error.status == 401

    def test_custom_message(self) -> None:
        """Test custom error message."""
        error = AragoraAuthenticationError("Invalid API key")
        assert error.message == "Invalid API key"
        assert error.code == "AUTHENTICATION_ERROR"
        assert error.status == 401

    def test_inheritance(self) -> None:
        """Test that it inherits from AragoraError."""
        error = AragoraAuthenticationError()
        assert isinstance(error, AragoraError)


class TestAragoraNotFoundError:
    """Tests for AragoraNotFoundError."""

    def test_initialization(self) -> None:
        """Test error creation with resource info."""
        error = AragoraNotFoundError("Debate", "debate-123")
        assert error.message == "Debate not found: debate-123"
        assert error.resource == "Debate"
        assert error.resource_id == "debate-123"
        assert error.code == "NOT_FOUND"
        assert error.status == 404

    def test_different_resources(self) -> None:
        """Test with different resource types."""
        user_error = AragoraNotFoundError("User", "user-abc")
        assert user_error.message == "User not found: user-abc"
        assert user_error.resource == "User"
        assert user_error.resource_id == "user-abc"

        workflow_error = AragoraNotFoundError("Workflow", "wf-xyz")
        assert workflow_error.message == "Workflow not found: wf-xyz"

    def test_inheritance(self) -> None:
        """Test that it inherits from AragoraError."""
        error = AragoraNotFoundError("Debate", "123")
        assert isinstance(error, AragoraError)


class TestAragoraValidationError:
    """Tests for AragoraValidationError."""

    def test_message_only(self) -> None:
        """Test error with message only."""
        error = AragoraValidationError("Invalid input")
        assert error.message == "Invalid input"
        assert error.code == "VALIDATION_ERROR"
        assert error.status == 400
        assert error.details == {}

    def test_with_details(self) -> None:
        """Test error with validation details."""
        details = {
            "errors": [
                {"field": "email", "message": "Invalid email format"},
                {"field": "name", "message": "Name is required"},
            ]
        }
        error = AragoraValidationError("Validation failed", details=details)
        assert error.message == "Validation failed"
        assert error.details == details
        assert len(error.details["errors"]) == 2

    def test_inheritance(self) -> None:
        """Test that it inherits from AragoraError."""
        error = AragoraValidationError("Test")
        assert isinstance(error, AragoraError)


class TestAragoraTimeoutError:
    """Tests for AragoraTimeoutError."""

    def test_default_message(self) -> None:
        """Test default error message."""
        error = AragoraTimeoutError()
        assert error.message == "Request timed out"
        assert error.code == "TIMEOUT"
        assert error.status is None

    def test_custom_message(self) -> None:
        """Test custom error message."""
        error = AragoraTimeoutError("Debate timed out after 300s")
        assert error.message == "Debate timed out after 300s"
        assert error.code == "TIMEOUT"

    def test_inheritance(self) -> None:
        """Test that it inherits from AragoraError."""
        error = AragoraTimeoutError()
        assert isinstance(error, AragoraError)


class TestExceptionCatching:
    """Test exception catching patterns."""

    def test_catch_specific_error(self) -> None:
        """Test catching specific exception types."""
        with pytest.raises(AragoraAuthenticationError):
            raise AragoraAuthenticationError("Invalid token")

    def test_catch_base_error(self) -> None:
        """Test catching specific errors as base class."""
        with pytest.raises(AragoraError):
            raise AragoraNotFoundError("Debate", "123")

    def test_catch_as_exception(self) -> None:
        """Test catching as general Exception."""
        with pytest.raises(Exception):
            raise AragoraValidationError("Bad input")

    def test_exception_chaining(self) -> None:
        """Test exception chaining with __cause__."""
        original = ValueError("Original error")
        try:
            raise AragoraConnectionError("Connection failed") from original
        except AragoraConnectionError as e:
            assert e.__cause__ is original
