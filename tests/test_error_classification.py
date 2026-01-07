"""
Tests for error classification in agent operations.

Tests the classify_cli_error function and error hierarchy for proper
categorization of CLI subprocess failures, rate limits, and timeouts.
"""

import json
import pytest

from aragora.agents.errors import (
    AgentError,
    AgentConnectionError,
    AgentTimeoutError,
    AgentRateLimitError,
    AgentAPIError,
    AgentResponseError,
    AgentStreamError,
    AgentCircuitOpenError,
    CLIAgentError,
    CLIParseError,
    CLITimeoutError,
    CLISubprocessError,
    CLINotFoundError,
    classify_cli_error,
)


class TestAgentErrorHierarchy:
    """Test the agent error class hierarchy."""

    def test_agent_error_base_class(self) -> None:
        """AgentError is the base for all agent errors."""
        error = AgentError("Test error", agent_name="test-agent")

        assert str(error) == "[test-agent] Test error"
        assert error.agent_name == "test-agent"
        assert error.recoverable is True

    def test_agent_error_with_cause(self) -> None:
        """AgentError captures root cause exception."""
        root_cause = ValueError("Root cause")
        error = AgentError("Wrapped error", cause=root_cause)

        assert error.cause is root_cause
        assert "ValueError" in str(error)
        assert "Root cause" in str(error)

    def test_connection_error_is_recoverable(self) -> None:
        """Connection errors are recoverable."""
        error = AgentConnectionError("Connection failed", status_code=503)

        assert error.recoverable is True
        assert error.status_code == 503

    def test_timeout_error_captures_timeout(self) -> None:
        """Timeout error captures the timeout value."""
        error = AgentTimeoutError(
            "Timed out",
            agent_name="slow-agent",
            timeout_seconds=30.0,
        )

        assert error.timeout_seconds == 30.0
        assert error.recoverable is True

    def test_rate_limit_error_captures_retry_after(self) -> None:
        """Rate limit error captures Retry-After header."""
        error = AgentRateLimitError(
            "Rate limited",
            agent_name="busy-agent",
            retry_after=60.0,
        )

        assert error.retry_after == 60.0
        assert error.recoverable is True

    def test_api_error_4xx_not_recoverable(self) -> None:
        """4xx API errors are not recoverable."""
        error = AgentAPIError(
            "Bad request",
            status_code=400,
            error_type="invalid_request",
        )

        assert error.recoverable is False
        assert error.status_code == 400
        assert error.error_type == "invalid_request"

    def test_api_error_5xx_is_recoverable(self) -> None:
        """5xx API errors are recoverable."""
        error = AgentAPIError("Server error", status_code=500)

        assert error.recoverable is True

    def test_response_error_not_recoverable(self) -> None:
        """Response parse errors are not recoverable."""
        error = AgentResponseError(
            "Invalid JSON",
            response_data={"malformed": True},
        )

        assert error.recoverable is False
        assert error.response_data == {"malformed": True}

    def test_stream_error_captures_partial_content(self) -> None:
        """Stream error captures partial content received."""
        error = AgentStreamError(
            "Stream interrupted",
            partial_content="Partial response so far...",
        )

        assert error.partial_content == "Partial response so far..."
        assert error.recoverable is True

    def test_circuit_open_error(self) -> None:
        """Circuit open error captures cooldown time."""
        error = AgentCircuitOpenError(
            "Circuit open",
            agent_name="failing-agent",
            cooldown_seconds=60.0,
        )

        assert error.cooldown_seconds == 60.0
        assert error.recoverable is True


class TestCLIAgentErrors:
    """Test CLI-specific error classes."""

    def test_cli_error_captures_returncode(self) -> None:
        """CLI error captures subprocess return code."""
        error = CLIAgentError(
            "Command failed",
            agent_name="codex",
            returncode=1,
            stderr="Error output",
        )

        assert error.returncode == 1
        assert error.stderr == "Error output"

    def test_cli_parse_error_captures_raw_output(self) -> None:
        """CLI parse error captures raw output for debugging."""
        error = CLIParseError(
            "Invalid JSON",
            raw_output="Not valid JSON {",
            returncode=0,
        )

        assert error.raw_output == "Not valid JSON {"
        assert error.recoverable is False

    def test_cli_timeout_error(self) -> None:
        """CLI timeout error has correct returncode."""
        error = CLITimeoutError(
            "Timed out",
            agent_name="slow-cli",
            timeout_seconds=30.0,
        )

        assert error.returncode == -9  # SIGKILL
        assert error.timeout_seconds == 30.0
        assert error.recoverable is True

    def test_cli_not_found_error(self) -> None:
        """CLI not found error is not recoverable."""
        error = CLINotFoundError(
            "Command not found",
            cli_name="missing-tool",
        )

        assert error.returncode == 127
        assert error.cli_name == "missing-tool"
        assert error.recoverable is False


class TestClassifyCLIError:
    """Test the classify_cli_error function."""

    def test_rate_limit_detection_from_stderr(self) -> None:
        """Detects rate limit from stderr message."""
        error = classify_cli_error(
            returncode=1,
            stderr="Error: rate limit exceeded",
            stdout="",
            agent_name="test-agent",
        )

        assert isinstance(error, CLIAgentError)
        assert error.recoverable is True
        assert "rate limit" in str(error).lower()

    def test_rate_limit_detection_429(self) -> None:
        """Detects rate limit from HTTP 429 in stderr."""
        error = classify_cli_error(
            returncode=1,
            stderr="HTTP Error 429: Too Many Requests",
            stdout="",
        )

        assert error.recoverable is True

    def test_timeout_detection_sigkill(self) -> None:
        """Detects timeout from SIGKILL return code."""
        error = classify_cli_error(
            returncode=-9,
            stderr="",
            stdout="",
            timeout_seconds=30.0,
        )

        assert isinstance(error, CLITimeoutError)
        assert error.timeout_seconds == 30.0

    def test_timeout_detection_from_stderr(self) -> None:
        """Detects timeout from stderr message."""
        error = classify_cli_error(
            returncode=1,
            stderr="Error: connection timed out",
            stdout="",
        )

        assert isinstance(error, CLITimeoutError)

    def test_command_not_found(self) -> None:
        """Detects command not found error."""
        error = classify_cli_error(
            returncode=127,
            stderr="bash: claude: command not found",
            stdout="",
        )

        assert isinstance(error, CLINotFoundError)
        assert error.recoverable is False

    def test_permission_denied(self) -> None:
        """Detects permission denied error."""
        error = classify_cli_error(
            returncode=126,
            stderr="Permission denied",
            stdout="",
        )

        assert isinstance(error, CLISubprocessError)
        assert error.returncode == 126

    def test_empty_response_with_success_code(self) -> None:
        """Empty response with success code classified as subprocess error."""
        error = classify_cli_error(
            returncode=0,
            stderr="",
            stdout="",
        )

        # Empty response with return code 0 is generic subprocess error
        assert isinstance(error, CLISubprocessError)

    def test_whitespace_only_response_parse_error(self) -> None:
        """Whitespace-only response classified as parse error."""
        error = classify_cli_error(
            returncode=0,
            stderr="",
            stdout="   \n\t  ",
        )

        assert isinstance(error, CLIParseError)
        assert error.recoverable is False

    def test_json_error_in_stdout(self) -> None:
        """Detects error in JSON response."""
        json_response = json.dumps({"error": "API key invalid"})

        error = classify_cli_error(
            returncode=0,
            stderr="",
            stdout=json_response,
        )

        assert isinstance(error, CLIAgentError)
        assert "API key invalid" in str(error)

    def test_invalid_json_parse_error(self) -> None:
        """Invalid JSON in stdout classified as parse error."""
        error = classify_cli_error(
            returncode=0,
            stderr="",
            stdout="{invalid json",
        )

        assert isinstance(error, CLIParseError)
        assert error.raw_output == "{invalid json"[:200]

    def test_generic_subprocess_error(self) -> None:
        """Unknown errors classified as generic subprocess error."""
        error = classify_cli_error(
            returncode=42,
            stderr="Some unknown error",
            stdout="",
        )

        assert isinstance(error, CLISubprocessError)
        assert error.returncode == 42
        assert "42" in str(error)

    def test_truncates_long_stderr(self) -> None:
        """Long stderr is truncated in error."""
        long_stderr = "x" * 1000

        error = classify_cli_error(
            returncode=1,
            stderr=long_stderr,
            stdout="",
        )

        # Should be truncated to 500 chars
        assert len(error.stderr) <= 500


class TestErrorPatternMatching:
    """Test pattern matching for various error types."""

    @pytest.mark.parametrize("pattern", [
        "rate limit exceeded",
        "rate_limit_error",
        "HTTP 429",
        "too many requests",
        "throttled by provider",
    ])
    def test_rate_limit_patterns(self, pattern: str) -> None:
        """Various rate limit patterns are detected."""
        error = classify_cli_error(
            returncode=1,
            stderr=f"Error: {pattern}",
            stdout="",
        )

        assert error.recoverable is True

    @pytest.mark.parametrize("pattern", [
        "timeout waiting for response",
        "request timed out",
        "connection timed out",
    ])
    def test_timeout_patterns(self, pattern: str) -> None:
        """Various timeout patterns are detected."""
        error = classify_cli_error(
            returncode=1,
            stderr=f"Error: {pattern}",
            stdout="",
        )

        assert isinstance(error, CLITimeoutError)

    @pytest.mark.parametrize("returncode,stderr", [
        (127, "command not found"),
        (127, "not found"),
    ])
    def test_not_found_patterns(self, returncode: int, stderr: str) -> None:
        """Various not-found patterns are detected."""
        error = classify_cli_error(
            returncode=returncode,
            stderr=stderr,
            stdout="",
        )

        assert isinstance(error, CLINotFoundError)


class TestErrorContextPreservation:
    """Test that error context is preserved for debugging."""

    def test_agent_name_in_all_errors(self) -> None:
        """Agent name is preserved in classified errors."""
        error = classify_cli_error(
            returncode=1,
            stderr="Some error",
            stdout="",
            agent_name="my-agent",
        )

        assert error.agent_name == "my-agent"

    def test_stderr_preserved_in_error(self) -> None:
        """Original stderr is preserved for debugging."""
        error = classify_cli_error(
            returncode=1,
            stderr="Detailed error message with context",
            stdout="",
        )

        assert "Detailed error message" in (error.stderr or "")

    def test_timeout_value_preserved(self) -> None:
        """Timeout value is preserved in timeout errors."""
        error = classify_cli_error(
            returncode=-9,
            stderr="",
            stdout="",
            timeout_seconds=45.0,
        )

        assert isinstance(error, CLITimeoutError)
        assert error.timeout_seconds == 45.0


class TestErrorClassifier:
    """Test the centralized ErrorClassifier utility class."""

    def test_is_rate_limit_detects_common_patterns(self) -> None:
        """ErrorClassifier.is_rate_limit detects rate limit patterns."""
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.is_rate_limit("rate limit exceeded") is True
        assert ErrorClassifier.is_rate_limit("HTTP 429 Too Many Requests") is True
        assert ErrorClassifier.is_rate_limit("quota exceeded") is True
        assert ErrorClassifier.is_rate_limit("You have been throttled") is True
        assert ErrorClassifier.is_rate_limit("billing issue") is True

    def test_is_rate_limit_negative_cases(self) -> None:
        """ErrorClassifier.is_rate_limit returns False for non-rate-limit errors."""
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.is_rate_limit("invalid input") is False
        assert ErrorClassifier.is_rate_limit("syntax error") is False
        assert ErrorClassifier.is_rate_limit("") is False

    def test_is_network_error_detects_connection_issues(self) -> None:
        """ErrorClassifier.is_network_error detects network patterns."""
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.is_network_error("connection refused") is True
        assert ErrorClassifier.is_network_error("503 service unavailable") is True
        assert ErrorClassifier.is_network_error("request timed out") is True
        assert ErrorClassifier.is_network_error("network is unreachable") is True

    def test_is_cli_error_detects_cli_patterns(self) -> None:
        """ErrorClassifier.is_cli_error detects CLI-specific patterns."""
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.is_cli_error("command not found") is True
        assert ErrorClassifier.is_cli_error("permission denied") is True
        assert ErrorClassifier.is_cli_error("model not found") is True
        assert ErrorClassifier.is_cli_error("unauthorized") is True

    def test_should_fallback_with_timeout_errors(self) -> None:
        """ErrorClassifier.should_fallback returns True for timeout errors."""
        import asyncio
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.should_fallback(TimeoutError("timed out")) is True
        assert ErrorClassifier.should_fallback(asyncio.TimeoutError()) is True

    def test_should_fallback_with_connection_errors(self) -> None:
        """ErrorClassifier.should_fallback returns True for connection errors."""
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.should_fallback(ConnectionError("refused")) is True
        assert ErrorClassifier.should_fallback(ConnectionRefusedError()) is True
        assert ErrorClassifier.should_fallback(ConnectionResetError()) is True
        assert ErrorClassifier.should_fallback(BrokenPipeError()) is True

    def test_should_fallback_with_rate_limit_message(self) -> None:
        """ErrorClassifier.should_fallback returns True for rate limit messages."""
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.should_fallback(Exception("rate limit exceeded")) is True
        assert ErrorClassifier.should_fallback(Exception("HTTP 429")) is True

    def test_should_fallback_with_os_errors(self) -> None:
        """ErrorClassifier.should_fallback returns True for specific OS errors."""
        from aragora.agents.errors import ErrorClassifier

        # ECONNREFUSED = 111
        error = OSError(111, "Connection refused")
        assert ErrorClassifier.should_fallback(error) is True

        # ETIMEDOUT = 110
        error = OSError(110, "Connection timed out")
        assert ErrorClassifier.should_fallback(error) is True

    def test_should_fallback_with_subprocess_errors(self) -> None:
        """ErrorClassifier.should_fallback returns True for subprocess errors."""
        import subprocess
        from aragora.agents.errors import ErrorClassifier

        error = subprocess.SubprocessError("Process failed")
        assert ErrorClassifier.should_fallback(error) is True

    def test_should_fallback_negative_cases(self) -> None:
        """ErrorClassifier.should_fallback returns False for regular errors."""
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.should_fallback(ValueError("invalid")) is False
        assert ErrorClassifier.should_fallback(TypeError("wrong type")) is False
        assert ErrorClassifier.should_fallback(KeyError("missing")) is False

    def test_get_error_category_timeout(self) -> None:
        """ErrorClassifier.get_error_category identifies timeout errors."""
        import asyncio
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.get_error_category(TimeoutError()) == "timeout"
        assert ErrorClassifier.get_error_category(asyncio.TimeoutError()) == "timeout"

    def test_get_error_category_rate_limit(self) -> None:
        """ErrorClassifier.get_error_category identifies rate limit errors."""
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.get_error_category(Exception("rate limit")) == "rate_limit"
        assert ErrorClassifier.get_error_category(Exception("429")) == "rate_limit"

    def test_get_error_category_network(self) -> None:
        """ErrorClassifier.get_error_category identifies network errors."""
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.get_error_category(ConnectionError()) == "network"
        assert ErrorClassifier.get_error_category(Exception("connection refused")) == "network"

    def test_get_error_category_cli(self) -> None:
        """ErrorClassifier.get_error_category identifies CLI errors."""
        import subprocess
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.get_error_category(subprocess.SubprocessError()) == "cli"
        assert ErrorClassifier.get_error_category(Exception("command not found")) == "cli"

    def test_get_error_category_unknown(self) -> None:
        """ErrorClassifier.get_error_category returns unknown for unclassified errors."""
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.get_error_category(ValueError("bad value")) == "unknown"
        assert ErrorClassifier.get_error_category(Exception("some error")) == "unknown"
