"""Tests for CLI error handler."""

import pytest
from unittest.mock import patch

from aragora.cli.error_handler import (
    CLIError,
    CLIErrorHandler,
    ErrorCategory,
    RecoverySuggestion,
    handle_cli_error,
    api_key_error,
    server_unavailable_error,
    rate_limit_error,
)


class TestErrorCategory:
    """Test ErrorCategory enum."""

    def test_all_categories_exist(self):
        """Verify all expected categories are defined."""
        expected = [
            "api_key",
            "network",
            "server",
            "agent",
            "config",
            "file",
            "validation",
            "permission",
            "unknown",
        ]
        for category in expected:
            assert ErrorCategory(category) is not None

    def test_category_is_string_enum(self):
        """Categories should be string enums."""
        assert ErrorCategory.API_KEY == "api_key"
        assert ErrorCategory.NETWORK.value == "network"


class TestRecoverySuggestion:
    """Test RecoverySuggestion dataclass."""

    def test_basic_suggestion(self):
        """Test creating a basic suggestion."""
        suggestion = RecoverySuggestion(
            title="Check your API key",
            steps=["Step 1", "Step 2"],
        )
        assert suggestion.title == "Check your API key"
        assert len(suggestion.steps) == 2
        assert suggestion.command is None

    def test_suggestion_with_command(self):
        """Test suggestion with a command."""
        suggestion = RecoverySuggestion(
            title="Run diagnostics",
            steps=["Check setup"],
            command="aragora doctor",
        )
        assert suggestion.command == "aragora doctor"


class TestCLIError:
    """Test CLIError dataclass."""

    def test_basic_error(self):
        """Test creating a basic error."""
        error = CLIError(
            message="Something went wrong",
            category=ErrorCategory.UNKNOWN,
        )
        assert error.message == "Something went wrong"
        assert error.category == ErrorCategory.UNKNOWN
        assert error.suggestions == []
        assert error.details is None

    def test_error_with_suggestions(self):
        """Test error with suggestions."""
        suggestions = [
            RecoverySuggestion(title="Fix it", steps=["Do this"]),
        ]
        error = CLIError(
            message="Test error",
            category=ErrorCategory.CONFIG,
            suggestions=suggestions,
        )
        assert len(error.suggestions) == 1

    def test_format_basic(self):
        """Test basic error formatting."""
        error = CLIError(
            message="Test error message",
            category=ErrorCategory.UNKNOWN,
        )
        formatted = error.format()
        assert "[ERROR]" in formatted
        assert "Test error message" in formatted

    def test_format_with_details(self):
        """Test formatting with details."""
        error = CLIError(
            message="Main error",
            category=ErrorCategory.UNKNOWN,
            details="Additional context here",
        )
        formatted = error.format()
        assert "Details:" in formatted
        assert "Additional context here" in formatted

    def test_format_with_suggestions(self):
        """Test formatting with suggestions."""
        error = CLIError(
            message="Error",
            category=ErrorCategory.UNKNOWN,
            suggestions=[
                RecoverySuggestion(
                    title="Fix this",
                    steps=["Step 1", "Step 2"],
                    command="run fix",
                )
            ],
        )
        formatted = error.format()
        assert "To fix this:" in formatted
        assert "Fix this" in formatted
        assert "Step 1" in formatted
        assert "$ run fix" in formatted

    def test_format_verbose_with_traceback(self):
        """Test verbose format includes traceback."""
        original = ValueError("Original error")
        error = CLIError(
            message="Wrapper",
            category=ErrorCategory.UNKNOWN,
            original_error=original,
        )
        formatted = error.format(verbose=True)
        assert "Traceback" in formatted
        assert "ValueError" in formatted


class TestCLIErrorHandlerClassify:
    """Test error classification."""

    def test_classify_api_key_errors(self):
        """Test classification of API key errors."""
        api_key_messages = [
            "Missing ANTHROPIC_API_KEY",
            "No api_key provided",
            "Authentication failed",
            "Unauthorized (401)",
            "sk-ant-xxx is invalid",
        ]
        for msg in api_key_messages:
            error = Exception(msg)
            category = CLIErrorHandler.classify_error(error)
            assert category == ErrorCategory.API_KEY, f"Failed for: {msg}"

    def test_classify_network_errors(self):
        """Test classification of network errors."""
        network_messages = [
            "Connection refused",
            "Request timeout",
            "Network unreachable",
            "Could not connect to server",
            "errno 61",
        ]
        for msg in network_messages:
            error = Exception(msg)
            category = CLIErrorHandler.classify_error(error)
            assert category == ErrorCategory.NETWORK, f"Failed for: {msg}"

    def test_classify_rate_limit_errors(self):
        """Test rate limit errors classify as network."""
        rate_limit_messages = [
            "Rate limit exceeded",
            "HTTP 429 Too Many Requests",
            "Quota exceeded",
        ]
        for msg in rate_limit_messages:
            error = Exception(msg)
            category = CLIErrorHandler.classify_error(error)
            assert category == ErrorCategory.NETWORK, f"Failed for: {msg}"

    def test_classify_server_errors(self):
        """Test classification of server errors."""
        server_messages = [
            "HTTP 500 Internal Server Error",
            "502 Bad Gateway",
            "Service unavailable (503)",
        ]
        for msg in server_messages:
            error = Exception(msg)
            category = CLIErrorHandler.classify_error(error)
            assert category == ErrorCategory.SERVER, f"Failed for: {msg}"

    def test_classify_permission_errors(self):
        """Test classification of permission errors."""
        permission_messages = [
            "Permission denied",
            "Access denied to file",
            "Forbidden resource",
            "errno 13",
        ]
        for msg in permission_messages:
            error = Exception(msg)
            category = CLIErrorHandler.classify_error(error)
            assert category == ErrorCategory.PERMISSION, f"Failed for: {msg}"

    def test_classify_config_errors(self):
        """Test classification of config errors."""
        # Config in message
        error = Exception("Invalid config format")
        assert CLIErrorHandler.classify_error(error) == ErrorCategory.CONFIG

    def test_classify_file_errors(self):
        """Test classification of file errors."""
        error = Exception("No such file or directory")
        assert CLIErrorHandler.classify_error(error) == ErrorCategory.FILE

    def test_classify_validation_errors(self):
        """Test classification of validation errors."""
        error = Exception("Invalid value provided")
        assert CLIErrorHandler.classify_error(error) == ErrorCategory.VALIDATION

    def test_classify_unknown_errors(self):
        """Test unknown error classification."""
        error = Exception("Something completely random happened")
        category = CLIErrorHandler.classify_error(error)
        assert category == ErrorCategory.UNKNOWN


class TestCLIErrorHandlerSuggestions:
    """Test recovery suggestion generation."""

    def test_api_key_suggestions(self):
        """Test suggestions for API key errors."""
        suggestions = CLIErrorHandler.get_suggestions(
            ErrorCategory.API_KEY, Exception("Missing key")
        )
        assert len(suggestions) >= 1
        titles = [s.title for s in suggestions]
        assert any("API key" in t or "environment" in t.lower() for t in titles)

    def test_network_rate_limit_suggestions(self):
        """Test suggestions for rate limit errors."""
        error = Exception("Rate limit exceeded")
        suggestions = CLIErrorHandler.get_suggestions(ErrorCategory.NETWORK, error)
        assert len(suggestions) >= 1
        all_steps = " ".join(step for s in suggestions for step in s.steps)
        assert "provider" in all_steps.lower() or "wait" in all_steps.lower()

    def test_server_suggestions(self):
        """Test suggestions for server errors."""
        suggestions = CLIErrorHandler.get_suggestions(
            ErrorCategory.SERVER, Exception("Server down")
        )
        assert len(suggestions) >= 1
        commands = [s.command for s in suggestions if s.command]
        assert any("serve" in cmd or "demo" in cmd for cmd in commands)

    def test_agent_suggestions(self):
        """Test suggestions for agent errors."""
        suggestions = CLIErrorHandler.get_suggestions(
            ErrorCategory.AGENT, Exception("Agent failed")
        )
        assert len(suggestions) >= 1

    def test_config_suggestions(self):
        """Test suggestions for config errors."""
        suggestions = CLIErrorHandler.get_suggestions(ErrorCategory.CONFIG, Exception("Bad config"))
        assert len(suggestions) >= 1

    def test_unknown_suggestions(self):
        """Test suggestions for unknown errors."""
        suggestions = CLIErrorHandler.get_suggestions(ErrorCategory.UNKNOWN, Exception("???"))
        assert len(suggestions) >= 1
        # Should suggest diagnostics
        all_text = " ".join(str(s) for s in suggestions)
        assert "doctor" in all_text.lower() or "help" in all_text.lower()


class TestCLIErrorHandlerCreate:
    """Test error creation."""

    def test_create_error(self):
        """Test creating a structured CLI error."""
        original = ValueError("Invalid input provided")
        cli_error = CLIErrorHandler.create_error(original)

        assert cli_error.message == "Invalid input provided"
        assert cli_error.category == ErrorCategory.VALIDATION  # "invalid" in message
        assert cli_error.original_error == original
        assert len(cli_error.suggestions) > 0

    def test_create_error_truncates_long_messages(self):
        """Test that long messages are truncated."""
        long_msg = "x" * 500
        original = Exception(long_msg)
        cli_error = CLIErrorHandler.create_error(original)

        assert len(cli_error.message) <= 203  # 200 + "..."
        assert cli_error.message.endswith("...")


class TestHandleCLIError:
    """Test the handle_cli_error function."""

    def test_handle_error_no_exit(self):
        """Test handling error without exiting."""
        error = Exception("Test error")
        result = handle_cli_error(error, exit_on_error=False)

        assert isinstance(result, CLIError)
        assert "Test error" in result.message

    def test_handle_error_exits(self):
        """Test that handle_cli_error exits by default."""
        error = Exception("Test error")
        with pytest.raises(SystemExit) as exc_info:
            handle_cli_error(error)
        assert exc_info.value.code == 1

    def test_handle_error_custom_exit_code(self):
        """Test custom exit code."""
        error = Exception("Test error")
        with pytest.raises(SystemExit) as exc_info:
            handle_cli_error(error, exit_code=42)
        assert exc_info.value.code == 42


class TestErrorShortcuts:
    """Test error shortcut functions."""

    def test_api_key_error(self):
        """Test API key error shortcut."""
        error = api_key_error("Anthropic")
        assert error.category == ErrorCategory.API_KEY
        assert "Anthropic" in error.message
        assert len(error.suggestions) > 0

    def test_server_unavailable_error(self):
        """Test server unavailable error shortcut."""
        error = server_unavailable_error("http://test:8080")
        assert error.category == ErrorCategory.SERVER
        assert "http://test:8080" in error.message
        assert len(error.suggestions) > 0

    def test_rate_limit_error(self):
        """Test rate limit error shortcut."""
        error = rate_limit_error("OpenAI")
        assert error.category == ErrorCategory.NETWORK
        assert "OpenAI" in error.message
        assert len(error.suggestions) > 0
