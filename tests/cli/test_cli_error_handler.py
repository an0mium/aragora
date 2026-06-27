"""Tests for CLI error handler with recovery suggestions."""

import sys
import pytest
from io import StringIO

from aragora.cli.error_handler import (
    CLIError,
    CLIErrorHandler,
    ErrorCategory,
    RecoverySuggestion,
    api_key_error,
    cli_error_handler,
    handle_cli_error,
    rate_limit_error,
    server_unavailable_error,
)


class TestErrorCategory:
    """Tests for ErrorCategory enum."""

    def test_all_categories_exist(self):
        """Verify all expected categories exist."""
        expected = [
            "API_KEY",
            "NETWORK",
            "SERVER",
            "AGENT",
            "CONFIG",
            "FILE",
            "VALIDATION",
            "PERMISSION",
            "UNKNOWN",
        ]
        for cat in expected:
            assert hasattr(ErrorCategory, cat)

    def test_category_values(self):
        """Verify category values are strings."""
        assert ErrorCategory.API_KEY.value == "api_key"
        assert ErrorCategory.NETWORK.value == "network"
        assert ErrorCategory.UNKNOWN.value == "unknown"


class TestRecoverySuggestion:
    """Tests for RecoverySuggestion dataclass."""

    def test_basic_creation(self):
        """Create a basic recovery suggestion."""
        suggestion = RecoverySuggestion(
            title="Fix the issue",
            steps=["Step 1", "Step 2"],
        )
        assert suggestion.title == "Fix the issue"
        assert len(suggestion.steps) == 2
        assert suggestion.command is None

    def test_with_command(self):
        """Create a suggestion with a command."""
        suggestion = RecoverySuggestion(
            title="Run diagnostics",
            steps=["Check your setup"],
            command="aragora doctor",
        )
        assert suggestion.command == "aragora doctor"


class TestCLIError:
    """Tests for CLIError dataclass."""

    def test_basic_creation(self):
        """Create a basic CLI error."""
        error = CLIError(
            message="Something went wrong",
            category=ErrorCategory.UNKNOWN,
        )
        assert error.message == "Something went wrong"
        assert error.category == ErrorCategory.UNKNOWN
        assert error.suggestions == []
        assert error.details is None
        assert error.original_error is None

    def test_with_all_fields(self):
        """Create a CLI error with all fields."""
        original = ValueError("Original error")
        suggestion = RecoverySuggestion(title="Fix it", steps=["Do this"])
        error = CLIError(
            message="Something went wrong",
            category=ErrorCategory.VALIDATION,
            suggestions=[suggestion],
            details="More info here",
            original_error=original,
        )
        assert error.suggestions == [suggestion]
        assert error.details == "More info here"
        assert error.original_error is original

    def test_format_basic(self):
        """Format a basic error."""
        error = CLIError(
            message="Test error",
            category=ErrorCategory.UNKNOWN,
        )
        formatted = error.format()
        assert "[ERROR] Test error" in formatted

    def test_format_with_details(self):
        """Format an error with details."""
        error = CLIError(
            message="Test error",
            category=ErrorCategory.UNKNOWN,
            details="Additional context",
        )
        formatted = error.format()
        assert "Details: Additional context" in formatted

    def test_format_with_suggestions(self):
        """Format an error with suggestions."""
        suggestion = RecoverySuggestion(
            title="Try this",
            steps=["First step", "Second step"],
            command="aragora fix",
        )
        error = CLIError(
            message="Test error",
            category=ErrorCategory.UNKNOWN,
            suggestions=[suggestion],
        )
        formatted = error.format()
        assert "To fix this:" in formatted
        assert "1. Try this" in formatted
        assert "First step" in formatted
        assert "$ aragora fix" in formatted

    def test_format_verbose_with_traceback(self):
        """Format an error with verbose traceback."""
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            error = CLIError(
                message="Test error",
                category=ErrorCategory.UNKNOWN,
                original_error=e,
            )
            formatted = error.format(verbose=True)
            assert "--- Traceback ---" in formatted
            assert "ValueError" in formatted


class TestCLIErrorHandler:
    """Tests for CLIErrorHandler class."""

    def test_classify_api_key_error(self):
        """Classify an API key error."""
        errors = [
            Exception("Missing ANTHROPIC_API_KEY"),
            Exception("401 Unauthorized"),
            Exception("Invalid authentication"),
            Exception("api key required"),
        ]
        for error in errors:
            category = CLIErrorHandler.classify_error(error)
            assert category == ErrorCategory.API_KEY, f"Failed for: {error}"

    def test_classify_network_error(self):
        """Classify a network error."""
        errors = [
            Exception("Connection refused"),
            Exception("Request timeout"),
            Exception("DNS resolution failed"),
            Exception("Could not connect to host"),
        ]
        for error in errors:
            category = CLIErrorHandler.classify_error(error)
            assert category == ErrorCategory.NETWORK, f"Failed for: {error}"

    def test_classify_rate_limit_as_network(self):
        """Rate limit errors are classified as network errors."""
        errors = [
            Exception("Rate limit exceeded"),
            Exception("429 Too Many Requests"),
            Exception("API quota exceeded"),
        ]
        for error in errors:
            category = CLIErrorHandler.classify_error(error)
            assert category == ErrorCategory.NETWORK, f"Failed for: {error}"

    def test_classify_server_error(self):
        """Classify a server error."""
        errors = [
            Exception("500 Internal Server Error"),
            Exception("503 Service Unavailable"),
            Exception("Server error occurred"),
        ]
        for error in errors:
            category = CLIErrorHandler.classify_error(error)
            assert category == ErrorCategory.SERVER, f"Failed for: {error}"

    def test_classify_permission_error(self):
        """Classify a permission error."""
        errors = [
            Exception("Permission denied"),
            Exception("Access denied to resource"),
            Exception("403 Forbidden"),
        ]
        for error in errors:
            category = CLIErrorHandler.classify_error(error)
            assert category == ErrorCategory.PERMISSION, f"Failed for: {error}"

    def test_classify_file_error(self):
        """Classify a file not found error."""
        error = FileNotFoundError("No such file or directory")
        category = CLIErrorHandler.classify_error(error)
        assert category == ErrorCategory.FILE

    def test_classify_validation_error(self):
        """Classify a validation error."""
        error = Exception("Invalid input provided")
        category = CLIErrorHandler.classify_error(error)
        assert category == ErrorCategory.VALIDATION

    def test_classify_unknown_error(self):
        """Unknown errors get UNKNOWN category."""
        error = Exception("Some random error without patterns")
        category = CLIErrorHandler.classify_error(error)
        assert category == ErrorCategory.UNKNOWN

    def test_get_suggestions_api_key(self):
        """Get suggestions for API key errors."""
        suggestions = CLIErrorHandler.get_suggestions(
            ErrorCategory.API_KEY, Exception("Missing key")
        )
        assert len(suggestions) >= 1
        # Should suggest setting API key
        titles = [s.title for s in suggestions]
        assert any("API key" in t for t in titles)

    def test_get_suggestions_network(self):
        """Get suggestions for network errors."""
        suggestions = CLIErrorHandler.get_suggestions(
            ErrorCategory.NETWORK, Exception("Connection failed")
        )
        assert len(suggestions) >= 1
        # Should suggest checking network
        titles = [s.title for s in suggestions]
        assert any("network" in t.lower() or "connection" in t.lower() for t in titles)

    def test_get_suggestions_rate_limit(self):
        """Get suggestions for rate limit errors."""
        suggestions = CLIErrorHandler.get_suggestions(
            ErrorCategory.NETWORK, Exception("Rate limit exceeded")
        )
        assert len(suggestions) >= 1
        # Should suggest fallback provider
        all_steps = " ".join(step for s in suggestions for step in s.steps)
        assert "fallback" in all_steps.lower() or "wait" in all_steps.lower()

    def test_get_suggestions_server(self):
        """Get suggestions for server errors."""
        suggestions = CLIErrorHandler.get_suggestions(
            ErrorCategory.SERVER, Exception("Server unavailable")
        )
        assert len(suggestions) >= 1
        # Should suggest starting server or demo mode
        all_steps = " ".join(step for s in suggestions for step in s.steps)
        assert "server" in all_steps.lower() or "demo" in all_steps.lower()

    def test_create_error(self):
        """Create a structured CLI error from an exception."""
        exception = ValueError("Missing ANTHROPIC_API_KEY in environment")
        cli_error = CLIErrorHandler.create_error(exception)

        assert cli_error.category == ErrorCategory.API_KEY
        assert len(cli_error.suggestions) >= 1
        assert cli_error.original_error is exception

    def test_create_error_truncates_long_message(self):
        """Long error messages are truncated."""
        long_msg = "A" * 300
        exception = Exception(long_msg)
        cli_error = CLIErrorHandler.create_error(exception)

        assert len(cli_error.message) <= 203  # 200 + "..."
        assert cli_error.message.endswith("...")


class TestHandleCLIError:
    """Tests for handle_cli_error function."""

    def test_returns_cli_error_without_exit(self, capsys):
        """Handle error without exiting."""
        error = ValueError("Test error")
        result = handle_cli_error(error, exit_on_error=False)

        assert isinstance(result, CLIError)
        captured = capsys.readouterr()
        assert "[ERROR]" in captured.err

    def test_exits_by_default(self):
        """Handle error exits by default."""
        error = ValueError("Test error")
        with pytest.raises(SystemExit) as exc_info:
            handle_cli_error(error, exit_code=42)
        assert exc_info.value.code == 42

    def test_verbose_shows_traceback(self, capsys):
        """Verbose mode shows traceback."""
        try:
            raise ValueError("Traceback test")
        except ValueError as e:
            handle_cli_error(e, verbose=True, exit_on_error=False)

        captured = capsys.readouterr()
        assert "Traceback" in captured.err
        assert "ValueError" in captured.err


class TestCLIErrorHandlerDecorator:
    """Tests for cli_error_handler decorator."""

    def test_decorator_passes_through_success(self):
        """Decorated function works normally on success."""

        @cli_error_handler()
        def successful_func():
            return "success"

        result = successful_func()
        assert result == "success"

    def test_decorator_handles_exception(self):
        """Decorated function handles exceptions."""

        @cli_error_handler()
        def failing_func():
            raise ValueError("Test failure")

        with pytest.raises(SystemExit):
            failing_func()

    def test_decorator_handles_keyboard_interrupt(self):
        """Decorated function handles KeyboardInterrupt."""

        @cli_error_handler()
        def interrupted_func():
            raise KeyboardInterrupt()

        with pytest.raises(SystemExit) as exc_info:
            interrupted_func()
        assert exc_info.value.code == 130

    def test_decorator_passes_system_exit(self):
        """Decorated function passes through SystemExit."""

        @cli_error_handler()
        def exiting_func():
            sys.exit(5)

        with pytest.raises(SystemExit) as exc_info:
            exiting_func()
        assert exc_info.value.code == 5


class TestErrorShortcuts:
    """Tests for error shortcut functions."""

    def test_api_key_error(self):
        """Create an API key error."""
        error = api_key_error("Anthropic")
        assert error.category == ErrorCategory.API_KEY
        assert "Anthropic" in error.message
        assert len(error.suggestions) >= 1

    def test_server_unavailable_error(self):
        """Create a server unavailable error."""
        error = server_unavailable_error("http://localhost:9000")
        assert error.category == ErrorCategory.SERVER
        assert "localhost:9000" in error.message
        assert len(error.suggestions) >= 1

    def test_rate_limit_error(self):
        """Create a rate limit error."""
        error = rate_limit_error("OpenAI")
        assert error.category == ErrorCategory.NETWORK
        assert "OpenAI" in error.message
        assert len(error.suggestions) >= 1
