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

    @pytest.mark.parametrize(
        "stderr,expected_recoverable",
        [
            ("Error: rate limit exceeded", True),
            ("HTTP Error 429: Too Many Requests", True),
            ("quota exceeded for today", True),
        ],
    )
    def test_rate_limit_detection(self, stderr: str, expected_recoverable: bool) -> None:
        """Detects rate limit from various stderr patterns."""
        error = classify_cli_error(
            returncode=1,
            stderr=stderr,
            stdout="",
            agent_name="test-agent",
        )

        assert error.recoverable is expected_recoverable

    @pytest.mark.parametrize(
        "returncode,stderr,timeout_seconds,expected_type",
        [
            (-9, "", 30.0, CLITimeoutError),
            (1, "Error: connection timed out", None, CLITimeoutError),
            (1, "request timed out", None, CLITimeoutError),
        ],
    )
    def test_timeout_detection(
        self, returncode: int, stderr: str, timeout_seconds: float | None, expected_type: type
    ) -> None:
        """Detects timeout from various sources."""
        error = classify_cli_error(
            returncode=returncode,
            stderr=stderr,
            stdout="",
            timeout_seconds=timeout_seconds,
        )

        assert isinstance(error, expected_type)
        if timeout_seconds is not None:
            assert error.timeout_seconds == timeout_seconds

    @pytest.mark.parametrize(
        "returncode,stderr,expected_type,expected_recoverable",
        [
            (127, "bash: claude: command not found", CLINotFoundError, False),
            (126, "Permission denied", CLISubprocessError, True),
            (42, "Some unknown error", CLISubprocessError, True),
        ],
    )
    def test_subprocess_error_types(
        self, returncode: int, stderr: str, expected_type: type, expected_recoverable: bool
    ) -> None:
        """Detects various subprocess error types."""
        error = classify_cli_error(
            returncode=returncode,
            stderr=stderr,
            stdout="",
        )

        assert isinstance(error, expected_type)
        assert error.returncode == returncode

    @pytest.mark.parametrize(
        "returncode,stdout,expected_type",
        [
            (0, "", CLISubprocessError),  # Empty response
            (0, "   \n\t  ", CLIParseError),  # Whitespace only
            (0, "{invalid json", CLIParseError),  # Invalid JSON
        ],
    )
    def test_parse_errors(self, returncode: int, stdout: str, expected_type: type) -> None:
        """Detects parse errors from various stdout patterns."""
        error = classify_cli_error(
            returncode=returncode,
            stderr="",
            stdout=stdout,
        )

        assert isinstance(error, expected_type)

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

    @pytest.mark.parametrize(
        "pattern",
        [
            "rate limit exceeded",
            "rate_limit_error",
            "HTTP 429",
            "too many requests",
            "throttled by provider",
        ],
    )
    def test_rate_limit_patterns(self, pattern: str) -> None:
        """Various rate limit patterns are detected."""
        error = classify_cli_error(
            returncode=1,
            stderr=f"Error: {pattern}",
            stdout="",
        )

        assert error.recoverable is True

    @pytest.mark.parametrize(
        "pattern",
        [
            "timeout waiting for response",
            "request timed out",
            "connection timed out",
        ],
    )
    def test_timeout_patterns(self, pattern: str) -> None:
        """Various timeout patterns are detected."""
        error = classify_cli_error(
            returncode=1,
            stderr=f"Error: {pattern}",
            stdout="",
        )

        assert isinstance(error, CLITimeoutError)

    @pytest.mark.parametrize(
        "returncode,stderr",
        [
            (127, "command not found"),
            (127, "not found"),
        ],
    )
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

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("rate limit exceeded", True),
            ("HTTP 429 Too Many Requests", True),
            ("quota exceeded", True),
            ("You have been throttled", True),
            ("billing issue", True),
            ("invalid input", False),
            ("syntax error", False),
            ("", False),
        ],
    )
    def test_is_rate_limit(self, message: str, expected: bool) -> None:
        """ErrorClassifier.is_rate_limit detects rate limit patterns."""
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.is_rate_limit(message) is expected

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("connection refused", True),
            ("503 service unavailable", True),
            ("request timed out", True),
            ("network is unreachable", True),
            ("normal response", False),
        ],
    )
    def test_is_network_error(self, message: str, expected: bool) -> None:
        """ErrorClassifier.is_network_error detects network patterns."""
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.is_network_error(message) is expected

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("command not found", True),
            ("permission denied", True),
            ("broken pipe", True),
            ("argument list too long", True),
            ("normal output", False),
        ],
    )
    def test_is_cli_error(self, message: str, expected: bool) -> None:
        """ErrorClassifier.is_cli_error detects CLI-specific patterns."""
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.is_cli_error(message) is expected

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("model not found", True),
            ("model unavailable", True),
            ("model is currently overloaded", True),
            ("valid response", False),
        ],
    )
    def test_is_model_error(self, message: str, expected: bool) -> None:
        """ErrorClassifier.is_model_error detects model-specific patterns."""
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.is_model_error(message) is expected

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("unauthorized", True),
            ("invalid api key", True),
            ("401", True),
            ("forbidden", True),
            ("access granted", False),
        ],
    )
    def test_is_auth_error(self, message: str, expected: bool) -> None:
        """ErrorClassifier.is_auth_error detects authentication patterns."""
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.is_auth_error(message) is expected

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("content policy violation", True),
            ("flagged by moderation", True),
            ("refused to generate", True),
            ("generated successfully", False),
        ],
    )
    def test_is_content_policy_error(self, message: str, expected: bool) -> None:
        """ErrorClassifier.is_content_policy_error detects content policy patterns."""
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.is_content_policy_error(message) is expected

    @pytest.mark.parametrize(
        "message,expected",
        [
            ("context length exceeded", True),
            ("invalid input", True),
            ("bad request 400", True),
            ("valid request", False),
        ],
    )
    def test_is_validation_error(self, message: str, expected: bool) -> None:
        """ErrorClassifier.is_validation_error detects input validation patterns."""
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.is_validation_error(message) is expected

    @pytest.mark.parametrize(
        "error_factory,expected",
        [
            (lambda: TimeoutError("timed out"), True),
            (lambda: ConnectionError("refused"), True),
            (lambda: ConnectionRefusedError(), True),
            (lambda: ConnectionResetError(), True),
            (lambda: BrokenPipeError(), True),
            (lambda: Exception("rate limit exceeded"), True),
            (lambda: Exception("HTTP 429"), True),
            (lambda: OSError(111, "Connection refused"), True),
            (lambda: OSError(110, "Connection timed out"), True),
            (lambda: ValueError("invalid"), False),
            (lambda: TypeError("wrong type"), False),
            (lambda: KeyError("missing"), False),
        ],
    )
    def test_should_fallback(self, error_factory, expected: bool) -> None:
        """ErrorClassifier.should_fallback returns correct result for various error types."""
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.should_fallback(error_factory()) is expected

    def test_should_fallback_with_asyncio_timeout(self) -> None:
        """ErrorClassifier.should_fallback returns True for asyncio timeout errors."""
        import asyncio
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.should_fallback(asyncio.TimeoutError()) is True

    def test_should_fallback_with_subprocess_errors(self) -> None:
        """ErrorClassifier.should_fallback returns True for subprocess errors."""
        import subprocess
        from aragora.agents.errors import ErrorClassifier

        error = subprocess.SubprocessError("Process failed")
        assert ErrorClassifier.should_fallback(error) is True

    @pytest.mark.parametrize(
        "error_factory,expected_category",
        [
            (lambda: TimeoutError(), "timeout"),
            (lambda: ConnectionError(), "network"),
            (lambda: Exception("rate limit"), "rate_limit"),
            (lambda: Exception("429"), "rate_limit"),
            (lambda: Exception("connection refused"), "network"),
            (lambda: Exception("unauthorized"), "auth"),
            (lambda: Exception("model not found"), "model"),
            (lambda: Exception("content policy"), "content_policy"),
            (lambda: Exception("context length exceeded"), "validation"),
            (lambda: ValueError("bad value"), "unknown"),
            (lambda: Exception("some error"), "unknown"),
        ],
    )
    def test_get_error_category(self, error_factory, expected_category: str) -> None:
        """ErrorClassifier.get_error_category identifies error categories."""
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.get_error_category(error_factory()) == expected_category

    def test_get_error_category_asyncio(self) -> None:
        """ErrorClassifier.get_error_category identifies asyncio timeout errors."""
        import asyncio
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.get_error_category(asyncio.TimeoutError()) == "timeout"

    def test_get_error_category_subprocess(self) -> None:
        """ErrorClassifier.get_error_category identifies subprocess errors."""
        import subprocess
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.get_error_category(subprocess.SubprocessError()) == "cli"
        assert ErrorClassifier.get_error_category(Exception("command not found")) == "cli"


class TestNewPatterns:
    """Test newly added error patterns."""

    @pytest.mark.parametrize(
        "message",
        [
            "ssl error: certificate verify failed",
            "SSL handshake failed",
            "certificate expired",
            "cert verify failed",
            "proxy error: connection failed",
            "HTTP 407 Proxy Authentication Required",
            "tunnel connection failed",
            "dns resolution failed",
            "getaddrinfo failed",
            "nodename nor servname provided",
            "HTTP 500 Internal Server Error",
            "internal server error",
        ],
    )
    def test_network_error_patterns(self, message: str) -> None:
        """Network-related errors (SSL, proxy, DNS, HTTP 500) are detected."""
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.is_network_error(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "context window exceeded",
            "maximum context limit",
            "token limit exceeded",
            "input too large",
            "prompt too long",
            "string_above_max_length",
            "please reduce your prompt",
            "reduce the length of your input",
            "exceeds the model's limit",
            "exceeds maximum allowed",
        ],
    )
    def test_validation_error_patterns(self, message: str) -> None:
        """Context window and validation errors are detected."""
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.is_validation_error(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "model does not exist",
            "does not support this feature",
            "model_access_denied",
            "this model has been decommissioned",
            "not available in your region",
        ],
    )
    def test_model_access_error_patterns(self, message: str) -> None:
        """Model access and availability errors are detected."""
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.is_model_error(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "I cannot assist with that",
            "I'm unable to generate",
            "output blocked by safety",
            "response blocked",
            "violates ethical guidelines",
        ],
    )
    def test_content_policy_patterns(self, message: str) -> None:
        """Natural language content policy responses are detected."""
        from aragora.agents.errors import ErrorClassifier

        assert ErrorClassifier.is_content_policy_error(message) is True

    @pytest.mark.parametrize(
        "message,expected_category,expected_fallback",
        [
            ("SSL certificate verify failed", "NETWORK", True),
            ("context window exceeded", "VALIDATION", False),
            ("model_access_denied", "MODEL", True),
        ],
    )
    def test_classify_full_with_new_patterns(
        self, message: str, expected_category: str, expected_fallback: bool
    ) -> None:
        """classify_full correctly handles new patterns."""
        from aragora.agents.errors import ErrorClassifier, ErrorCategory

        result = ErrorClassifier.classify_full(Exception(message))
        assert result.category == getattr(ErrorCategory, expected_category)
        assert result.should_fallback is expected_fallback


class TestClassifyFull:
    """Test the comprehensive classify_full method."""

    def test_classify_full_returns_classified_error(self) -> None:
        """classify_full returns a ClassifiedError instance."""
        from aragora.agents.errors import ErrorClassifier, ClassifiedError

        result = ErrorClassifier.classify_full(TimeoutError("timed out"))

        assert isinstance(result, ClassifiedError)
        assert result.message == "timed out"

    @pytest.mark.parametrize(
        "error_factory,category,severity,action,fallback",
        [
            (lambda: TimeoutError("request timed out"), "TIMEOUT", "WARNING", "RETRY", True),
            (lambda: Exception("rate limit exceeded"), "RATE_LIMIT", "INFO", "WAIT", True),
            (lambda: Exception("401 Unauthorized"), "AUTH", "CRITICAL", "ESCALATE", True),
            (
                lambda: Exception("Content policy violation"),
                "CONTENT_POLICY",
                "ERROR",
                "ABORT",
                False,
            ),
            (lambda: Exception("Context length exceeded"), "VALIDATION", "ERROR", "ABORT", False),
            (lambda: Exception("model not found"), "MODEL", "WARNING", "FALLBACK", True),
            (lambda: ConnectionError("connection refused"), "NETWORK", "WARNING", "RETRY", True),
        ],
    )
    def test_classify_full_error_types(
        self, error_factory, category: str, severity: str, action: str, fallback: bool
    ) -> None:
        """classify_full correctly classifies various error types."""
        from aragora.agents.errors import (
            ErrorClassifier,
            ErrorCategory,
            ErrorSeverity,
            RecoveryAction,
        )

        result = ErrorClassifier.classify_full(error_factory())

        assert result.category == getattr(ErrorCategory, category)
        assert result.severity == getattr(ErrorSeverity, severity)
        assert result.action == getattr(RecoveryAction, action)
        assert result.should_fallback is fallback

    def test_classify_full_timeout_is_recoverable(self) -> None:
        """classify_full marks timeout errors as recoverable."""
        from aragora.agents.errors import ErrorClassifier

        result = ErrorClassifier.classify_full(TimeoutError("request timed out"))
        assert result.is_recoverable is True

    def test_classify_full_rate_limit_has_retry_after(self) -> None:
        """classify_full sets retry_after for rate limit errors."""
        from aragora.agents.errors import ErrorClassifier

        result = ErrorClassifier.classify_full(Exception("rate limit exceeded"))
        assert result.retry_after is not None

    def test_classify_full_rate_limit_extracts_retry_after(self) -> None:
        """classify_full extracts retry-after from error message."""
        from aragora.agents.errors import ErrorClassifier

        result = ErrorClassifier.classify_full(Exception("Rate limit: retry after 120 seconds"))
        assert result.retry_after == 120.0

    def test_classify_full_content_policy_not_recoverable(self) -> None:
        """classify_full marks content policy errors as non-recoverable."""
        from aragora.agents.errors import ErrorClassifier

        result = ErrorClassifier.classify_full(Exception("Content policy violation"))
        assert result.is_recoverable is False

    def test_classified_error_category_str_property(self) -> None:
        """ClassifiedError.category_str returns string for backward compat."""
        from aragora.agents.errors import ErrorClassifier

        result = ErrorClassifier.classify_full(TimeoutError())

        assert result.category_str == "timeout"
        assert isinstance(result.category_str, str)
