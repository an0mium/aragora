"""
Tests for error classification module.

Tests cover:
- ErrorCategory and ErrorSeverity enums
- Pattern-based error detection
- ErrorClassifier methods
- ClassifiedError dataclass
- CLI error classification
- Recovery action recommendations
"""

from __future__ import annotations

import asyncio
import subprocess

import pytest

from aragora.agents.errors.classifier import (
    ALL_FALLBACK_PATTERNS,
    AUTH_ERROR_PATTERNS,
    CLI_ERROR_PATTERNS,
    CONTENT_POLICY_PATTERNS,
    MODEL_ERROR_PATTERNS,
    NETWORK_ERROR_PATTERNS,
    RATE_LIMIT_PATTERNS,
    VALIDATION_ERROR_PATTERNS,
    ClassifiedError,
    ErrorCategory,
    ErrorClassifier,
    ErrorSeverity,
    RecoveryAction,
    classify_cli_error,
)
from aragora.agents.errors.exceptions import (
    CLIAgentError,
    CLINotFoundError,
    CLIParseError,
    CLISubprocessError,
    CLITimeoutError,
)


# ============================================================================
# Enum Tests
# ============================================================================


class TestErrorCategory:
    """Tests for ErrorCategory enum."""

    def test_all_categories_exist(self):
        """Test all expected categories exist."""
        expected = [
            "TIMEOUT",
            "RATE_LIMIT",
            "NETWORK",
            "CLI",
            "AUTH",
            "VALIDATION",
            "MODEL",
            "CONTENT_POLICY",
            "UNKNOWN",
        ]

        for name in expected:
            assert hasattr(ErrorCategory, name)
            assert getattr(ErrorCategory, name).value == name.lower()

    def test_category_values(self):
        """Test category values are lowercase strings."""
        assert ErrorCategory.TIMEOUT.value == "timeout"
        assert ErrorCategory.RATE_LIMIT.value == "rate_limit"
        assert ErrorCategory.NETWORK.value == "network"
        assert ErrorCategory.UNKNOWN.value == "unknown"


class TestErrorSeverity:
    """Tests for ErrorSeverity enum."""

    def test_all_severities_exist(self):
        """Test all expected severities exist."""
        assert ErrorSeverity.CRITICAL.value == "critical"
        assert ErrorSeverity.ERROR.value == "error"
        assert ErrorSeverity.WARNING.value == "warning"
        assert ErrorSeverity.INFO.value == "info"


class TestRecoveryAction:
    """Tests for RecoveryAction enum."""

    def test_all_actions_exist(self):
        """Test all expected actions exist."""
        assert RecoveryAction.RETRY.value == "retry"
        assert RecoveryAction.RETRY_IMMEDIATE.value == "retry_immediate"
        assert RecoveryAction.FALLBACK.value == "fallback"
        assert RecoveryAction.WAIT.value == "wait"
        assert RecoveryAction.ABORT.value == "abort"
        assert RecoveryAction.ESCALATE.value == "escalate"


# ============================================================================
# Pattern Constants Tests
# ============================================================================


class TestPatternConstants:
    """Tests for error pattern constants."""

    def test_rate_limit_patterns_include_common_errors(self):
        """Test rate limit patterns include common error messages."""
        patterns = RATE_LIMIT_PATTERNS

        assert "rate limit" in patterns
        assert "429" in patterns
        assert "quota exceeded" in patterns
        assert "too many requests" in patterns

    def test_network_patterns_include_common_errors(self):
        """Test network patterns include common error messages."""
        patterns = NETWORK_ERROR_PATTERNS

        assert "503" in patterns
        assert "connection refused" in patterns
        assert "timeout" in patterns
        assert "ssl error" in patterns

    def test_auth_patterns_include_common_errors(self):
        """Test auth patterns include common error messages."""
        patterns = AUTH_ERROR_PATTERNS

        assert "401" in patterns
        assert "unauthorized" in patterns
        assert "invalid api key" in patterns
        assert "forbidden" in patterns

    def test_validation_patterns_include_common_errors(self):
        """Test validation patterns include common error messages."""
        patterns = VALIDATION_ERROR_PATTERNS

        assert "context length" in patterns
        assert "too long" in patterns
        assert "bad request" in patterns
        assert "400" in patterns

    def test_model_patterns_include_common_errors(self):
        """Test model patterns include common error messages."""
        patterns = MODEL_ERROR_PATTERNS

        assert "model not found" in patterns
        assert "model overloaded" in patterns
        assert "unsupported model" in patterns

    def test_content_policy_patterns_include_common_errors(self):
        """Test content policy patterns include common error messages."""
        patterns = CONTENT_POLICY_PATTERNS

        assert "content policy" in patterns
        assert "content filter" in patterns
        assert "moderation" in patterns

    def test_cli_patterns_include_common_errors(self):
        """Test CLI patterns include common error messages."""
        patterns = CLI_ERROR_PATTERNS

        assert "command not found" in patterns
        assert "permission denied" in patterns

    def test_all_fallback_patterns_is_combined(self):
        """Test ALL_FALLBACK_PATTERNS combines expected patterns."""
        # Should include rate limit, network, cli, auth, model
        assert all(p in ALL_FALLBACK_PATTERNS for p in RATE_LIMIT_PATTERNS)
        assert all(p in ALL_FALLBACK_PATTERNS for p in NETWORK_ERROR_PATTERNS)
        assert all(p in ALL_FALLBACK_PATTERNS for p in AUTH_ERROR_PATTERNS)

        # Should NOT include content policy or validation (can't recover)
        assert not any(p in ALL_FALLBACK_PATTERNS for p in CONTENT_POLICY_PATTERNS)


# ============================================================================
# ClassifiedError Tests
# ============================================================================


class TestClassifiedError:
    """Tests for ClassifiedError dataclass."""

    def test_create_classified_error(self):
        """Test creating a classified error."""
        error = ClassifiedError(
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.WARNING,
            action=RecoveryAction.RETRY,
            should_fallback=True,
            message="Request timed out",
        )

        assert error.category == ErrorCategory.TIMEOUT
        assert error.severity == ErrorSeverity.WARNING
        assert error.action == RecoveryAction.RETRY
        assert error.should_fallback is True
        assert error.message == "Request timed out"

    def test_is_recoverable_for_retry(self):
        """Test is_recoverable for recoverable actions."""
        error = ClassifiedError(
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.WARNING,
            action=RecoveryAction.RETRY,
            should_fallback=True,
        )

        assert error.is_recoverable is True

    def test_is_recoverable_for_abort(self):
        """Test is_recoverable for non-recoverable actions."""
        error = ClassifiedError(
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.ERROR,
            action=RecoveryAction.ABORT,
            should_fallback=False,
        )

        assert error.is_recoverable is False

    def test_is_recoverable_for_escalate(self):
        """Test is_recoverable for escalate actions."""
        error = ClassifiedError(
            category=ErrorCategory.AUTH,
            severity=ErrorSeverity.CRITICAL,
            action=RecoveryAction.ESCALATE,
            should_fallback=True,
        )

        assert error.is_recoverable is False

    def test_category_str_property(self):
        """Test category_str returns string value."""
        error = ClassifiedError(
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.INFO,
            action=RecoveryAction.WAIT,
            should_fallback=True,
        )

        assert error.category_str == "rate_limit"

    def test_retry_after_optional(self):
        """Test retry_after is optional."""
        error = ClassifiedError(
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.INFO,
            action=RecoveryAction.WAIT,
            should_fallback=True,
            retry_after=60.0,
        )

        assert error.retry_after == 60.0

    def test_details_default_empty(self):
        """Test details defaults to empty dict."""
        error = ClassifiedError(
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.WARNING,
            action=RecoveryAction.RETRY,
            should_fallback=True,
        )

        assert error.details == {}


# ============================================================================
# ErrorClassifier Pattern Detection Tests
# ============================================================================


class TestErrorClassifierPatternDetection:
    """Tests for ErrorClassifier pattern detection methods."""

    def test_is_rate_limit_positive(self):
        """Test rate limit detection for positive cases."""
        assert ErrorClassifier.is_rate_limit("Error 429: Too many requests")
        assert ErrorClassifier.is_rate_limit("Rate limit exceeded")
        assert ErrorClassifier.is_rate_limit("Quota exceeded for model")
        assert ErrorClassifier.is_rate_limit("Resource exhausted")
        assert ErrorClassifier.is_rate_limit("insufficient_quota error")  # Pattern uses underscore

    def test_is_rate_limit_negative(self):
        """Test rate limit detection for negative cases."""
        assert not ErrorClassifier.is_rate_limit("Connection refused")
        assert not ErrorClassifier.is_rate_limit("Invalid input")
        assert not ErrorClassifier.is_rate_limit("Model not found")

    def test_is_network_error_positive(self):
        """Test network error detection for positive cases."""
        assert ErrorClassifier.is_network_error("503 Service Unavailable")
        assert ErrorClassifier.is_network_error("Connection refused")
        assert ErrorClassifier.is_network_error("Request timed out")
        assert ErrorClassifier.is_network_error("SSL error occurred")
        assert ErrorClassifier.is_network_error("Bad gateway 502")

    def test_is_network_error_negative(self):
        """Test network error detection for negative cases."""
        assert not ErrorClassifier.is_network_error("Invalid API key")
        assert not ErrorClassifier.is_network_error("Content policy violation")

    def test_is_auth_error_positive(self):
        """Test auth error detection for positive cases."""
        assert ErrorClassifier.is_auth_error("401 Unauthorized")
        assert ErrorClassifier.is_auth_error("Invalid API key")
        assert ErrorClassifier.is_auth_error("Authentication failed")
        assert ErrorClassifier.is_auth_error("403 Forbidden")
        assert ErrorClassifier.is_auth_error("Token expired")

    def test_is_auth_error_negative(self):
        """Test auth error detection for negative cases."""
        assert not ErrorClassifier.is_auth_error("Connection refused")
        assert not ErrorClassifier.is_auth_error("Model overloaded")

    def test_is_validation_error_positive(self):
        """Test validation error detection for positive cases."""
        assert ErrorClassifier.is_validation_error("Context length exceeded")
        assert ErrorClassifier.is_validation_error("Input too large")
        assert ErrorClassifier.is_validation_error("400 Bad Request")
        assert ErrorClassifier.is_validation_error("Prompt too long")
        assert ErrorClassifier.is_validation_error("Missing required field")

    def test_is_validation_error_negative(self):
        """Test validation error detection for negative cases."""
        assert not ErrorClassifier.is_validation_error("Rate limit exceeded")
        assert not ErrorClassifier.is_validation_error("Connection timeout")

    def test_is_model_error_positive(self):
        """Test model error detection for positive cases."""
        assert ErrorClassifier.is_model_error("Model not found")
        assert ErrorClassifier.is_model_error("Model is currently overloaded")
        assert ErrorClassifier.is_model_error("Unsupported model")
        assert ErrorClassifier.is_model_error("Model deprecated")

    def test_is_model_error_negative(self):
        """Test model error detection for negative cases."""
        assert not ErrorClassifier.is_model_error("Rate limit exceeded")
        assert not ErrorClassifier.is_model_error("Invalid input")

    def test_is_content_policy_error_positive(self):
        """Test content policy error detection for positive cases."""
        assert ErrorClassifier.is_content_policy_error("Content policy violation")
        assert ErrorClassifier.is_content_policy_error("Content filter triggered")
        assert ErrorClassifier.is_content_policy_error("Flagged for moderation")
        assert ErrorClassifier.is_content_policy_error("Safety filter activated")

    def test_is_content_policy_error_negative(self):
        """Test content policy error detection for negative cases."""
        assert not ErrorClassifier.is_content_policy_error("Rate limit exceeded")
        assert not ErrorClassifier.is_content_policy_error("Model not found")

    def test_is_cli_error_positive(self):
        """Test CLI error detection for positive cases."""
        assert ErrorClassifier.is_cli_error("Command not found")
        assert ErrorClassifier.is_cli_error("Permission denied")
        assert ErrorClassifier.is_cli_error("No such file or directory")

    def test_is_cli_error_negative(self):
        """Test CLI error detection for negative cases."""
        assert not ErrorClassifier.is_cli_error("Rate limit exceeded")


# ============================================================================
# ErrorClassifier should_fallback Tests
# ============================================================================


class TestErrorClassifierShouldFallback:
    """Tests for ErrorClassifier.should_fallback method."""

    def test_timeout_error_should_fallback(self):
        """Test TimeoutError triggers fallback."""
        error = TimeoutError("Request timed out")
        assert ErrorClassifier.should_fallback(error) is True

    def test_asyncio_timeout_error_should_fallback(self):
        """Test asyncio.TimeoutError triggers fallback."""
        error = asyncio.TimeoutError()
        assert ErrorClassifier.should_fallback(error) is True

    def test_connection_error_should_fallback(self):
        """Test ConnectionError triggers fallback."""
        error = ConnectionError("Connection refused")
        assert ErrorClassifier.should_fallback(error) is True

    def test_connection_refused_error_should_fallback(self):
        """Test ConnectionRefusedError triggers fallback."""
        error = ConnectionRefusedError("Connection refused")
        assert ErrorClassifier.should_fallback(error) is True

    def test_broken_pipe_error_should_fallback(self):
        """Test BrokenPipeError triggers fallback."""
        error = BrokenPipeError("Broken pipe")
        assert ErrorClassifier.should_fallback(error) is True

    def test_rate_limit_message_should_fallback(self):
        """Test rate limit in message triggers fallback."""
        error = RuntimeError("Error 429: Rate limit exceeded")
        assert ErrorClassifier.should_fallback(error) is True

    def test_network_message_should_fallback(self):
        """Test network error in message triggers fallback."""
        error = RuntimeError("503 Service Unavailable")
        assert ErrorClassifier.should_fallback(error) is True

    def test_os_error_with_network_errno_should_fallback(self):
        """Test OSError with network errno triggers fallback."""
        error = OSError(111, "Connection refused")  # ECONNREFUSED
        assert ErrorClassifier.should_fallback(error) is True

    def test_subprocess_error_should_fallback(self):
        """Test subprocess error triggers fallback."""
        error = subprocess.SubprocessError("Command failed")
        assert ErrorClassifier.should_fallback(error) is True

    def test_runtime_error_with_cli_should_fallback(self):
        """Test RuntimeError with CLI in message triggers fallback."""
        error = RuntimeError("CLI command failed")
        assert ErrorClassifier.should_fallback(error) is True

    def test_runtime_error_with_api_error_should_fallback(self):
        """Test RuntimeError with API error triggers fallback."""
        error = RuntimeError("API error: 500 Internal Server Error")
        assert ErrorClassifier.should_fallback(error) is True

    def test_generic_value_error_should_not_fallback(self):
        """Test generic ValueError doesn't trigger fallback."""
        error = ValueError("Invalid value provided")
        assert ErrorClassifier.should_fallback(error) is False


# ============================================================================
# ErrorClassifier get_error_category Tests
# ============================================================================


class TestErrorClassifierGetCategory:
    """Tests for ErrorClassifier.get_error_category method."""

    def test_timeout_category(self):
        """Test timeout errors get correct category."""
        error = TimeoutError("Request timed out")
        assert ErrorClassifier.get_error_category(error) == "timeout"

    def test_asyncio_timeout_category(self):
        """Test asyncio timeout errors get correct category."""
        error = asyncio.TimeoutError()
        assert ErrorClassifier.get_error_category(error) == "timeout"

    def test_rate_limit_category(self):
        """Test rate limit errors get correct category."""
        error = RuntimeError("429 Rate limit exceeded")
        assert ErrorClassifier.get_error_category(error) == "rate_limit"

    def test_network_category_from_message(self):
        """Test network errors from message get correct category."""
        error = RuntimeError("503 Service Unavailable")
        assert ErrorClassifier.get_error_category(error) == "network"

    def test_network_category_from_exception_type(self):
        """Test network errors from exception type get correct category."""
        error = ConnectionRefusedError("Connection refused")
        assert ErrorClassifier.get_error_category(error) == "network"

    def test_auth_category(self):
        """Test auth errors get correct category."""
        error = RuntimeError("401 Unauthorized")
        assert ErrorClassifier.get_error_category(error) == "auth"

    def test_content_policy_category(self):
        """Test content policy errors get correct category."""
        error = RuntimeError("Content policy violation")
        assert ErrorClassifier.get_error_category(error) == "content_policy"

    def test_model_category(self):
        """Test model errors get correct category."""
        error = RuntimeError("Model not found")
        assert ErrorClassifier.get_error_category(error) == "model"

    def test_validation_category(self):
        """Test validation errors get correct category."""
        error = RuntimeError("Context length exceeded")
        assert ErrorClassifier.get_error_category(error) == "validation"

    def test_cli_category_from_message(self):
        """Test CLI errors from message get correct category."""
        error = RuntimeError("Command not found: claude")
        assert ErrorClassifier.get_error_category(error) == "cli"

    def test_cli_category_from_exception_type(self):
        """Test CLI errors from exception type get correct category."""
        error = subprocess.SubprocessError("Subprocess failed")
        assert ErrorClassifier.get_error_category(error) == "cli"

    def test_unknown_category(self):
        """Test unknown errors get correct category."""
        error = RuntimeError("Something went wrong")
        assert ErrorClassifier.get_error_category(error) == "unknown"


# ============================================================================
# ErrorClassifier classify_error Tests
# ============================================================================


class TestErrorClassifierClassifyError:
    """Tests for ErrorClassifier.classify_error method."""

    def test_classify_timeout(self):
        """Test timeout error classification."""
        error = TimeoutError("Request timed out")
        should_fallback, category = ErrorClassifier.classify_error(error)

        assert should_fallback is True
        assert category == "timeout"

    def test_classify_rate_limit(self):
        """Test rate limit error classification."""
        error = RuntimeError("429 Rate limit exceeded")
        should_fallback, category = ErrorClassifier.classify_error(error)

        assert should_fallback is True
        assert category == "rate_limit"

    def test_classify_network(self):
        """Test network error classification."""
        error = ConnectionRefusedError("Connection refused")
        should_fallback, category = ErrorClassifier.classify_error(error)

        assert should_fallback is True
        assert category == "network"

    def test_classify_content_policy_no_fallback(self):
        """Test content policy errors don't trigger fallback."""
        error = RuntimeError("Content policy violation")
        should_fallback, category = ErrorClassifier.classify_error(error)

        assert should_fallback is False
        assert category == "content_policy"

    def test_classify_validation_no_fallback(self):
        """Test validation errors don't trigger fallback."""
        error = RuntimeError("Context length exceeded")
        should_fallback, category = ErrorClassifier.classify_error(error)

        assert should_fallback is False
        assert category == "validation"

    def test_classify_auth_with_fallback(self):
        """Test auth errors trigger fallback."""
        error = RuntimeError("401 Unauthorized")
        should_fallback, category = ErrorClassifier.classify_error(error)

        assert should_fallback is True
        assert category == "auth"


# ============================================================================
# ErrorClassifier classify_full Tests
# ============================================================================


class TestErrorClassifierClassifyFull:
    """Tests for ErrorClassifier.classify_full method."""

    def test_classify_full_timeout(self):
        """Test full timeout error classification."""
        error = TimeoutError("Request timed out")
        result = ErrorClassifier.classify_full(error)

        assert result.category == ErrorCategory.TIMEOUT
        assert result.severity == ErrorSeverity.WARNING
        assert result.action == RecoveryAction.RETRY
        assert result.should_fallback is True
        assert "timed out" in result.message

    def test_classify_full_rate_limit_with_retry_after(self):
        """Test rate limit classification extracts retry-after."""
        error = RuntimeError("Rate limit exceeded. Retry after 30 seconds.")
        result = ErrorClassifier.classify_full(error)

        assert result.category == ErrorCategory.RATE_LIMIT
        assert result.severity == ErrorSeverity.INFO
        assert result.action == RecoveryAction.WAIT
        assert result.retry_after == 30.0

    def test_classify_full_rate_limit_default_retry_after(self):
        """Test rate limit classification uses default retry-after."""
        error = RuntimeError("429 Too many requests")
        result = ErrorClassifier.classify_full(error)

        assert result.category == ErrorCategory.RATE_LIMIT
        assert result.retry_after == 60.0  # Default

    def test_classify_full_network(self):
        """Test full network error classification."""
        error = ConnectionRefusedError("Connection refused")
        result = ErrorClassifier.classify_full(error)

        assert result.category == ErrorCategory.NETWORK
        assert result.severity == ErrorSeverity.WARNING
        assert result.action == RecoveryAction.RETRY
        assert result.should_fallback is True

    def test_classify_full_auth(self):
        """Test full auth error classification."""
        error = RuntimeError("401 Unauthorized")
        result = ErrorClassifier.classify_full(error)

        assert result.category == ErrorCategory.AUTH
        assert result.severity == ErrorSeverity.CRITICAL
        assert result.action == RecoveryAction.ESCALATE
        assert result.should_fallback is True

    def test_classify_full_content_policy(self):
        """Test full content policy error classification."""
        error = RuntimeError("Content policy violation")
        result = ErrorClassifier.classify_full(error)

        assert result.category == ErrorCategory.CONTENT_POLICY
        assert result.severity == ErrorSeverity.ERROR
        assert result.action == RecoveryAction.ABORT
        assert result.should_fallback is False

    def test_classify_full_model(self):
        """Test full model error classification."""
        error = RuntimeError("Model not found: gpt-5")
        result = ErrorClassifier.classify_full(error)

        assert result.category == ErrorCategory.MODEL
        assert result.severity == ErrorSeverity.WARNING
        assert result.action == RecoveryAction.FALLBACK
        assert result.should_fallback is True

    def test_classify_full_validation(self):
        """Test full validation error classification."""
        error = RuntimeError("Context length exceeded")
        result = ErrorClassifier.classify_full(error)

        assert result.category == ErrorCategory.VALIDATION
        assert result.severity == ErrorSeverity.ERROR
        assert result.action == RecoveryAction.ABORT
        assert result.should_fallback is False

    def test_classify_full_cli(self):
        """Test full CLI error classification."""
        error = subprocess.SubprocessError("Command not found")
        result = ErrorClassifier.classify_full(error)

        assert result.category == ErrorCategory.CLI
        assert result.severity == ErrorSeverity.WARNING
        assert result.action == RecoveryAction.FALLBACK
        assert result.should_fallback is True

    def test_classify_full_os_error_with_errno(self):
        """Test OS error with network errno gets classified as network error.

        Note: Python auto-converts some errno values to specific exception types
        (e.g., errno 32 becomes BrokenPipeError). The errno check path is only
        reached if no pattern matches first, so we use a message that won't
        match any error patterns.
        """
        # Use errno 7 (E2BIG) which is in NETWORK_ERRNO but use a generic message
        # that won't match any patterns (E2BIG's normal message "Argument list too long"
        # is in CLI_ERROR_PATTERNS)
        error = OSError(7, "System limit reached")
        result = ErrorClassifier.classify_full(error)

        assert result.category == ErrorCategory.NETWORK
        assert result.details.get("errno") == 7

    def test_classify_full_unknown(self):
        """Test unknown error classification."""
        error = RuntimeError("Something unexpected happened")
        result = ErrorClassifier.classify_full(error)

        assert result.category == ErrorCategory.UNKNOWN
        assert result.severity == ErrorSeverity.WARNING
        assert result.action == RecoveryAction.RETRY


# ============================================================================
# classify_cli_error Tests
# ============================================================================


class TestClassifyCLIError:
    """Tests for classify_cli_error function."""

    def test_classify_rate_limit(self):
        """Test CLI rate limit error classification."""
        error = classify_cli_error(
            returncode=1, stderr="Error: 429 Rate limit exceeded", stdout="", agent_name="claude"
        )

        assert isinstance(error, CLIAgentError)
        assert "Rate limit" in error.message
        assert error.agent_name == "claude"
        assert error.recoverable is True

    def test_classify_timeout_by_signal(self):
        """Test CLI timeout error classification by signal."""
        error = classify_cli_error(
            returncode=-9,  # SIGKILL
            stderr="",
            stdout="",
            agent_name="claude",
            timeout_seconds=30.0,
        )

        assert isinstance(error, CLITimeoutError)
        assert "30" in error.message
        assert error.timeout_seconds == 30.0

    def test_classify_timeout_by_message(self):
        """Test CLI timeout error classification by message."""
        error = classify_cli_error(
            returncode=1, stderr="Request timed out", stdout="", agent_name="claude"
        )

        assert isinstance(error, CLITimeoutError)

    def test_classify_command_not_found(self):
        """Test CLI command not found error classification."""
        error = classify_cli_error(
            returncode=127, stderr="bash: claude: command not found", stdout=""
        )

        assert isinstance(error, CLINotFoundError)
        assert "not found" in error.message.lower()

    def test_classify_permission_denied(self):
        """Test CLI permission denied error classification."""
        error = classify_cli_error(
            returncode=126, stderr="Permission denied", stdout="", agent_name="claude"
        )

        assert isinstance(error, CLISubprocessError)
        assert error.returncode == 126

    def test_classify_empty_response(self):
        """Test CLI empty response error classification."""
        error = classify_cli_error(
            returncode=0, stderr="", stdout="   ", agent_name="claude"  # Empty/whitespace
        )

        assert isinstance(error, CLIParseError)
        assert "Empty" in error.message

    def test_classify_json_error_response(self):
        """Test CLI JSON error response classification."""
        error = classify_cli_error(
            returncode=1, stderr="", stdout='{"error": "Something went wrong"}', agent_name="claude"
        )

        assert isinstance(error, CLIAgentError)
        assert "Something went wrong" in error.message
        assert error.recoverable is True

    def test_classify_invalid_json_response(self):
        """Test CLI invalid JSON response classification."""
        error = classify_cli_error(
            returncode=0, stderr="", stdout="{invalid json", agent_name="claude"
        )

        assert isinstance(error, CLIParseError)
        assert "Invalid JSON" in error.message

    def test_classify_generic_subprocess_error(self):
        """Test CLI generic subprocess error classification."""
        error = classify_cli_error(
            returncode=1, stderr="Unknown error occurred", stdout="", agent_name="claude"
        )

        assert isinstance(error, CLISubprocessError)
        assert error.returncode == 1
        assert "Unknown error" in error.stderr


# ============================================================================
# Network Errno Tests
# ============================================================================


class TestNetworkErrno:
    """Tests for network errno detection."""

    def test_network_errno_constants(self):
        """Test NETWORK_ERRNO contains expected values."""
        errno_set = ErrorClassifier.NETWORK_ERRNO

        assert 7 in errno_set  # E2BIG
        assert 32 in errno_set  # EPIPE
        assert 104 in errno_set  # ECONNRESET
        assert 110 in errno_set  # ETIMEDOUT
        assert 111 in errno_set  # ECONNREFUSED
        assert 113 in errno_set  # EHOSTUNREACH

    def test_os_error_with_network_errno_triggers_fallback(self):
        """Test OSError with network errno triggers fallback."""
        for errno in [104, 110, 111, 113]:  # ECONNRESET, ETIMEDOUT, ECONNREFUSED, EHOSTUNREACH
            error = OSError(errno, f"Error {errno}")
            assert ErrorClassifier.should_fallback(error) is True

    def test_os_error_without_network_errno_no_fallback(self):
        """Test OSError without network errno and no matching pattern doesn't trigger fallback."""
        # Use errno not in NETWORK_ERRNO and message not in any pattern
        error = OSError(1, "Generic operation error")  # EPERM with non-matching message
        assert ErrorClassifier.should_fallback(error) is False
