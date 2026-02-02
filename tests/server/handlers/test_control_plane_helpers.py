"""
Tests for Control Plane handler helper methods.

Tests cover the helper methods introduced to reduce code duplication:
- _require_coordinator: Returns coordinator or error response
- _handle_coordinator_error: Unified error handling for operations
"""

import pytest
from unittest.mock import MagicMock, patch

from aragora.server.handlers.control_plane import ControlPlaneHandler


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def handler():
    """Create a ControlPlaneHandler instance for testing."""
    ctx = {"control_plane_coordinator": None}
    return ControlPlaneHandler(ctx)


@pytest.fixture
def handler_with_coordinator():
    """Create a ControlPlaneHandler instance with a mock coordinator."""
    mock_coordinator = MagicMock()
    ctx = {"control_plane_coordinator": mock_coordinator}
    handler = ControlPlaneHandler(ctx)
    return handler, mock_coordinator


# ============================================================================
# Tests for _require_coordinator
# ============================================================================


class TestRequireCoordinator:
    """Tests for the _require_coordinator helper method."""

    def test_returns_error_when_not_initialized(self, handler):
        """Test that _require_coordinator returns error when coordinator is None."""
        coordinator, err = handler._require_coordinator()

        assert coordinator is None
        assert err is not None
        # HandlerResult is a dataclass with status_code attribute
        assert err.status_code == 503
        assert b"Control plane not initialized" in err.body

    def test_returns_coordinator_when_initialized(self, handler_with_coordinator):
        """Test that _require_coordinator returns coordinator when available."""
        handler, mock_coordinator = handler_with_coordinator

        coordinator, err = handler._require_coordinator()

        assert coordinator is mock_coordinator
        assert err is None

    def test_returns_class_level_coordinator_if_set(self, handler):
        """Test that _require_coordinator returns class-level coordinator if set."""
        mock_coordinator = MagicMock()
        original = ControlPlaneHandler.coordinator
        try:
            ControlPlaneHandler.coordinator = mock_coordinator

            coordinator, err = handler._require_coordinator()

            assert coordinator is mock_coordinator
            assert err is None
        finally:
            ControlPlaneHandler.coordinator = original


# ============================================================================
# Tests for _handle_coordinator_error
# ============================================================================


class TestHandleCoordinatorError:
    """Tests for the _handle_coordinator_error helper method."""

    def test_returns_400_for_value_error(self, handler):
        """Test that ValueError returns 400 status code."""
        error = ValueError("Invalid value provided")

        result = handler._handle_coordinator_error(error, "test_operation")

        assert result.status_code == 400

    def test_returns_400_for_key_error(self, handler):
        """Test that KeyError returns 400 status code."""
        error = KeyError("missing_key")

        result = handler._handle_coordinator_error(error, "test_operation")

        assert result.status_code == 400

    def test_returns_400_for_attribute_error(self, handler):
        """Test that AttributeError returns 400 status code."""
        error = AttributeError("Object has no attribute 'foo'")

        result = handler._handle_coordinator_error(error, "test_operation")

        assert result.status_code == 400

    def test_returns_500_for_runtime_error(self, handler):
        """Test that RuntimeError returns 500 status code."""
        error = RuntimeError("Something went wrong")

        result = handler._handle_coordinator_error(error, "test_operation")

        assert result.status_code == 500

    def test_returns_500_for_generic_exception(self, handler):
        """Test that generic Exception returns 500 status code."""
        error = Exception("Unknown error")

        result = handler._handle_coordinator_error(error, "test_operation")

        assert result.status_code == 500

    def test_returns_500_for_type_error(self, handler):
        """Test that TypeError (not in 400 list) returns 500 status code."""
        error = TypeError("Expected int, got str")

        result = handler._handle_coordinator_error(error, "test_operation")

        assert result.status_code == 500

    def test_logs_warning_for_data_errors(self, handler):
        """Test that data errors (400) are logged as warnings."""
        error = ValueError("Invalid data")

        with patch("aragora.server.handlers.control_plane.logger") as mock_logger:
            handler._handle_coordinator_error(error, "test_op")
            mock_logger.warning.assert_called_once()
            assert "Data error" in str(mock_logger.warning.call_args)
            assert "test_op" in str(mock_logger.warning.call_args)

    def test_logs_error_for_other_exceptions(self, handler):
        """Test that other errors (500) are logged as errors."""
        error = RuntimeError("System failure")

        with patch("aragora.server.handlers.control_plane.logger") as mock_logger:
            handler._handle_coordinator_error(error, "test_op")
            mock_logger.error.assert_called_once()
            assert "test_op" in str(mock_logger.error.call_args)

    def test_includes_operation_in_log_message(self, handler):
        """Test that the operation name is included in log messages."""
        error = ValueError("Test error")

        with patch("aragora.server.handlers.control_plane.logger") as mock_logger:
            handler._handle_coordinator_error(error, "my_custom_operation")
            call_args = str(mock_logger.warning.call_args)
            assert "my_custom_operation" in call_args


# ============================================================================
# Integration Tests - Helper usage in handlers
# ============================================================================


class TestHelperIntegration:
    """Test that helpers are correctly integrated into handler methods."""

    def test_handler_uses_require_coordinator(self, handler):
        """Test that handler methods correctly use _require_coordinator."""
        # Without coordinator, _handle_system_health should return 503
        result = handler._handle_system_health()
        assert result.status_code == 503
        assert b"Control plane not initialized" in result.body

    def test_handler_uses_error_handler_on_exception(self, handler_with_coordinator):
        """Test that handler methods use _handle_coordinator_error on exceptions."""
        handler, mock_coordinator = handler_with_coordinator

        # Make the coordinator raise an exception
        mock_coordinator.get_system_health.side_effect = ValueError("Test error")

        result = handler._handle_system_health()

        # Should get a 400 response because ValueError is a data error
        assert result.status_code == 400
