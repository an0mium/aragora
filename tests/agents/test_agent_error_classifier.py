"""
Tests for agent error classification.

Tests error pattern matching, classification, recovery actions, and fallback decisions.
"""

from __future__ import annotations

import asyncio
import subprocess
from unittest.mock import MagicMock

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
    ErrorAction,
    ErrorCategory,
    ErrorClassifier,
    ErrorContext,
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


# =============================================================================
# Pattern Constants Tests
# =============================================================================


class TestPatternConstants:
    """Tests for error pattern constant tuples."""

    def test_rate_limit_patterns_non_empty(self):
        """Test that rate limit patterns are defined."""
        assert len(RATE_LIMIT_PATTERNS) > 0
        assert "rate limit" in RATE_LIMIT_PATTERNS
        assert "429" in RATE_LIMIT_PATTERNS

    def test_network_error_patterns_non_empty(self):
        """Test that network error patterns are defined."""
        assert len(NETWORK_ERROR_PATTERNS) > 0
        assert "503" in NETWORK_ERROR_PATTERNS
        assert "connection refused" in NETWORK_ERROR_PATTERNS

    def test_cli_error_patterns_non_empty(self):
        """Test that CLI error patterns are defined."""
        assert len(CLI_ERROR_PATTERNS) > 0
        assert "command not found" in CLI_ERROR_PATTERNS

    def test_auth_error_patterns_non_empty(self):
        """Test that auth error patterns are defined."""
        assert len(AUTH_ERROR_PATTERNS) > 0
        assert "unauthorized" in AUTH_ERROR_PATTERNS
        assert "401" in AUTH_ERROR_PATTERNS

    def test_validation_error_patterns_non_empty(self):
        """Test that validation error patterns are defined."""
        assert len(VALIDATION_ERROR_PATTERNS) > 0
        assert "context length" in VALIDATION_ERROR_PATTERNS

    def test_model_error_patterns_non_empty(self):
        """Test that model error patterns are defined."""
        assert len(MODEL_ERROR_PATTERNS) > 0
        assert "model not found" in MODEL_ERROR_PATTERNS

    def test_content_policy_patterns_non_empty(self):
        """Test that content policy patterns are defined."""
        assert len(CONTENT_POLICY_PATTERNS) > 0
        assert "content policy" in CONTENT_POLICY_PATTERNS

    def test_all_fallback_patterns_includes_rate_limit(self):
        """Test that ALL_FALLBACK_PATTERNS includes rate limit patterns."""
        for pattern in RATE_LIMIT_PATTERNS:
            assert pattern in ALL_FALLBACK_PATTERNS

    def test_all_fallback_patterns_includes_network(self):
        """Test that ALL_FALLBACK_PATTERNS includes network patterns."""
        for pattern in NETWORK_ERROR_PATTERNS:
            assert pattern in ALL_FALLBACK_PATTERNS


# =============================================================================
# Enum Tests
# =============================================================================


class TestErrorEnums:
    """Tests for error classification enums."""

    def test_error_category_values(self):
        """Test ErrorCategory enum values."""
        assert ErrorCategory.TIMEOUT.value == "timeout"
        assert ErrorCategory.RATE_LIMIT.value == "rate_limit"
        assert ErrorCategory.NETWORK.value == "network"
        assert ErrorCategory.CLI.value == "cli"
        assert ErrorCategory.AUTH.value == "auth"
        assert ErrorCategory.VALIDATION.value == "validation"
        assert ErrorCategory.MODEL.value == "model"
        assert ErrorCategory.CONTENT_POLICY.value == "content_policy"
        assert ErrorCategory.UNKNOWN.value == "unknown"

    def test_error_severity_values(self):
        """Test ErrorSeverity enum values."""
        assert ErrorSeverity.CRITICAL.value == "critical"
        assert ErrorSeverity.ERROR.value == "error"
        assert ErrorSeverity.WARNING.value == "warning"
        assert ErrorSeverity.INFO.value == "info"

    def test_recovery_action_values(self):
        """Test RecoveryAction enum values."""
        assert RecoveryAction.RETRY.value == "retry"
        assert RecoveryAction.RETRY_IMMEDIATE.value == "retry_immediate"
        assert RecoveryAction.FALLBACK.value == "fallback"
        assert RecoveryAction.WAIT.value == "wait"
        assert RecoveryAction.ABORT.value == "abort"
        assert RecoveryAction.ESCALATE.value == "escalate"


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestClassifiedError:
    """Tests for ClassifiedError dataclass."""

    def test_create_classified_error(self):
        """Test creating a ClassifiedError."""
        err = ClassifiedError(
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.INFO,
            action=RecoveryAction.WAIT,
            should_fallback=True,
            message="Rate limit exceeded",
            retry_after=60.0,
        )
        assert err.category == ErrorCategory.RATE_LIMIT
        assert err.severity == ErrorSeverity.INFO
        assert err.action == RecoveryAction.WAIT
        assert err.should_fallback is True
        assert err.retry_after == 60.0

    def test_is_recoverable_true(self):
        """Test is_recoverable for recoverable actions."""
        err = ClassifiedError(
            category=ErrorCategory.TIMEOUT,
            severity=ErrorSeverity.WARNING,
            action=RecoveryAction.RETRY,
            should_fallback=True,
        )
        assert err.is_recoverable is True

    def test_is_recoverable_false_for_abort(self):
        """Test is_recoverable for ABORT action."""
        err = ClassifiedError(
            category=ErrorCategory.CONTENT_POLICY,
            severity=ErrorSeverity.ERROR,
            action=RecoveryAction.ABORT,
            should_fallback=False,
        )
        assert err.is_recoverable is False

    def test_is_recoverable_false_for_escalate(self):
        """Test is_recoverable for ESCALATE action."""
        err = ClassifiedError(
            category=ErrorCategory.AUTH,
            severity=ErrorSeverity.CRITICAL,
            action=RecoveryAction.ESCALATE,
            should_fallback=True,
        )
        assert err.is_recoverable is False

    def test_category_str_property(self):
        """Test category_str backward compatibility property."""
        err = ClassifiedError(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.WARNING,
            action=RecoveryAction.RETRY,
            should_fallback=True,
        )
        assert err.category_str == "network"

    def test_details_default_empty(self):
        """Test that details defaults to empty dict."""
        err = ClassifiedError(
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.WARNING,
            action=RecoveryAction.RETRY,
            should_fallback=False,
        )
        assert err.details == {}


class TestErrorContext:
    """Tests for ErrorContext dataclass."""

    def test_create_error_context(self):
        """Test creating an ErrorContext."""
        ctx = ErrorContext(
            agent_name="claude",
            attempt=2,
            max_retries=3,
            retry_delay=1.0,
            max_delay=30.0,
            timeout=60.0,
        )
        assert ctx.agent_name == "claude"
        assert ctx.attempt == 2
        assert ctx.max_retries == 3
        assert ctx.timeout == 60.0


class TestErrorAction:
    """Tests for ErrorAction dataclass (deprecated but maintained)."""

    def test_create_error_action(self):
        """Test creating an ErrorAction."""
        mock_error = MagicMock()
        action = ErrorAction(
            error=mock_error,
            should_retry=True,
            delay_seconds=5.0,
            log_level="warning",
        )
        assert action.error is mock_error
        assert action.should_retry is True
        assert action.delay_seconds == 5.0


# =============================================================================
# ErrorClassifier Pattern Matching Tests
# =============================================================================


class TestErrorClassifierPatterns:
    """Tests for ErrorClassifier pattern matching methods."""

    # Rate limit detection
    def test_is_rate_limit_429(self):
        """Test detecting 429 rate limit."""
        assert ErrorClassifier.is_rate_limit("Error: 429 Too Many Requests") is True

    def test_is_rate_limit_quota(self):
        """Test detecting quota exceeded."""
        assert ErrorClassifier.is_rate_limit("Error: quota exceeded") is True

    def test_is_rate_limit_throttled(self):
        """Test detecting throttled requests."""
        assert ErrorClassifier.is_rate_limit("Request was throttled") is True

    def test_is_rate_limit_billing(self):
        """Test detecting billing errors."""
        assert ErrorClassifier.is_rate_limit("Billing issue: insufficient credits") is True

    def test_is_rate_limit_false(self):
        """Test that normal errors aren't classified as rate limit."""
        assert ErrorClassifier.is_rate_limit("Connection failed") is False

    # Network error detection
    def test_is_network_error_503(self):
        """Test detecting 503 service unavailable."""
        assert ErrorClassifier.is_network_error("503 Service Unavailable") is True

    def test_is_network_error_connection_refused(self):
        """Test detecting connection refused."""
        assert ErrorClassifier.is_network_error("Connection refused") is True

    def test_is_network_error_dns(self):
        """Test detecting DNS failures."""
        assert ErrorClassifier.is_network_error("DNS resolution failed") is True

    def test_is_network_error_ssl(self):
        """Test detecting SSL errors."""
        assert ErrorClassifier.is_network_error("SSL certificate verify failed") is True

    def test_is_network_error_timeout(self):
        """Test detecting timeout errors."""
        assert ErrorClassifier.is_network_error("Request timed out") is True

    def test_is_network_error_false(self):
        """Test that normal errors aren't classified as network errors."""
        assert ErrorClassifier.is_network_error("Invalid input") is False

    # CLI error detection
    def test_is_cli_error_command_not_found(self):
        """Test detecting command not found."""
        assert ErrorClassifier.is_cli_error("bash: claude: command not found") is True

    def test_is_cli_error_permission_denied(self):
        """Test detecting permission denied."""
        assert ErrorClassifier.is_cli_error("Permission denied") is True

    def test_is_cli_error_broken_pipe(self):
        """Test detecting broken pipe."""
        assert ErrorClassifier.is_cli_error("Broken pipe") is True

    def test_is_cli_error_false(self):
        """Test that normal errors aren't classified as CLI errors."""
        assert ErrorClassifier.is_cli_error("API error") is False

    # Auth error detection
    def test_is_auth_error_401(self):
        """Test detecting 401 unauthorized."""
        assert ErrorClassifier.is_auth_error("HTTP 401 Unauthorized") is True

    def test_is_auth_error_403(self):
        """Test detecting 403 forbidden."""
        assert ErrorClassifier.is_auth_error("403 Forbidden") is True

    def test_is_auth_error_invalid_key(self):
        """Test detecting invalid API key."""
        assert ErrorClassifier.is_auth_error("Error: invalid_api_key") is True

    def test_is_auth_error_token_expired(self):
        """Test detecting expired token."""
        assert ErrorClassifier.is_auth_error("Token expired") is True

    def test_is_auth_error_false(self):
        """Test that normal errors aren't classified as auth errors."""
        assert ErrorClassifier.is_auth_error("Server error") is False

    # Validation error detection
    def test_is_validation_error_context_length(self):
        """Test detecting context length errors."""
        assert ErrorClassifier.is_validation_error("Error: context_length exceeded") is True

    def test_is_validation_error_too_long(self):
        """Test detecting prompt too long."""
        assert ErrorClassifier.is_validation_error("Prompt too long") is True

    def test_is_validation_error_400(self):
        """Test detecting 400 bad request."""
        assert ErrorClassifier.is_validation_error("400 Bad Request: Invalid input") is True

    def test_is_validation_error_false(self):
        """Test that normal errors aren't classified as validation errors."""
        assert ErrorClassifier.is_validation_error("Timeout") is False

    # Model error detection
    def test_is_model_error_not_found(self):
        """Test detecting model not found."""
        assert ErrorClassifier.is_model_error("model_not_found: gpt-5") is True

    def test_is_model_error_overloaded(self):
        """Test detecting model overloaded."""
        assert ErrorClassifier.is_model_error("Model is currently overloaded") is True

    def test_is_model_error_deprecated(self):
        """Test detecting deprecated model."""
        assert ErrorClassifier.is_model_error("Model deprecated") is True

    def test_is_model_error_false(self):
        """Test that normal errors aren't classified as model errors."""
        assert ErrorClassifier.is_model_error("Rate limit") is False

    # Content policy detection
    def test_is_content_policy_error_policy(self):
        """Test detecting content policy violation."""
        assert ErrorClassifier.is_content_policy_error("Content policy violation") is True

    def test_is_content_policy_error_moderation(self):
        """Test detecting moderation flag."""
        assert ErrorClassifier.is_content_policy_error("Content flagged by moderation") is True

    def test_is_content_policy_error_safety(self):
        """Test detecting safety filter."""
        assert ErrorClassifier.is_content_policy_error("Blocked by safety filter") is True

    def test_is_content_policy_error_false(self):
        """Test that normal errors aren't classified as content policy."""
        assert ErrorClassifier.is_content_policy_error("Network error") is False


# =============================================================================
# ErrorClassifier Fallback Decision Tests
# =============================================================================


class TestErrorClassifierFallback:
    """Tests for ErrorClassifier.should_fallback method."""

    def test_should_fallback_timeout_error(self):
        """Test fallback triggered by TimeoutError."""
        error = TimeoutError("Request timed out")
        assert ErrorClassifier.should_fallback(error) is True

    def test_should_fallback_asyncio_timeout(self):
        """Test fallback triggered by asyncio.TimeoutError."""
        error = asyncio.TimeoutError()
        assert ErrorClassifier.should_fallback(error) is True

    def test_should_fallback_connection_error(self):
        """Test fallback triggered by ConnectionError."""
        error = ConnectionError("Connection failed")
        assert ErrorClassifier.should_fallback(error) is True

    def test_should_fallback_connection_refused(self):
        """Test fallback triggered by ConnectionRefusedError."""
        error = ConnectionRefusedError("Connection refused")
        assert ErrorClassifier.should_fallback(error) is True

    def test_should_fallback_broken_pipe(self):
        """Test fallback triggered by BrokenPipeError."""
        error = BrokenPipeError("Broken pipe")
        assert ErrorClassifier.should_fallback(error) is True

    def test_should_fallback_rate_limit_message(self):
        """Test fallback triggered by rate limit in message."""
        error = Exception("Error: 429 Too Many Requests")
        assert ErrorClassifier.should_fallback(error) is True

    def test_should_fallback_network_error_message(self):
        """Test fallback triggered by network error in message."""
        error = Exception("503 Service Unavailable")
        assert ErrorClassifier.should_fallback(error) is True

    def test_should_fallback_os_error_errno(self):
        """Test fallback triggered by OSError with network errno."""
        error = OSError(111, "Connection refused")
        assert ErrorClassifier.should_fallback(error) is True

    def test_should_fallback_runtime_cli_error(self):
        """Test fallback triggered by RuntimeError with CLI message."""
        error = RuntimeError("CLI command failed with return code 1")
        assert ErrorClassifier.should_fallback(error) is True

    def test_should_fallback_subprocess_error(self):
        """Test fallback triggered by SubprocessError."""
        error = subprocess.SubprocessError("Subprocess failed")
        assert ErrorClassifier.should_fallback(error) is True

    def test_should_not_fallback_generic(self):
        """Test no fallback for generic errors."""
        error = Exception("Generic error")
        assert ErrorClassifier.should_fallback(error) is False

    def test_should_not_fallback_content_policy(self):
        """Test no fallback for content policy via should_fallback (basic check)."""
        # Note: should_fallback uses ALL_FALLBACK_PATTERNS which doesn't include content policy
        error = Exception("Content policy violation")
        # This checks the pattern, but content_policy is not in ALL_FALLBACK_PATTERNS
        assert ErrorClassifier.should_fallback(error) is False


# =============================================================================
# ErrorClassifier Category Tests
# =============================================================================


class TestErrorClassifierCategory:
    """Tests for ErrorClassifier.get_error_category method."""

    def test_category_timeout(self):
        """Test category for timeout errors."""
        error = TimeoutError("Timed out")
        assert ErrorClassifier.get_error_category(error) == "timeout"

    def test_category_rate_limit(self):
        """Test category for rate limit errors."""
        error = Exception("429 Too Many Requests")
        assert ErrorClassifier.get_error_category(error) == "rate_limit"

    def test_category_network(self):
        """Test category for network errors."""
        error = ConnectionError("Connection failed")
        assert ErrorClassifier.get_error_category(error) == "network"

    def test_category_auth(self):
        """Test category for auth errors."""
        error = Exception("401 Unauthorized")
        assert ErrorClassifier.get_error_category(error) == "auth"

    def test_category_content_policy(self):
        """Test category for content policy errors."""
        error = Exception("Content policy violation")
        assert ErrorClassifier.get_error_category(error) == "content_policy"

    def test_category_model(self):
        """Test category for model errors."""
        error = Exception("Model not found")
        assert ErrorClassifier.get_error_category(error) == "model"

    def test_category_validation(self):
        """Test category for validation errors."""
        error = Exception("Context length exceeded")
        assert ErrorClassifier.get_error_category(error) == "validation"

    def test_category_cli(self):
        """Test category for CLI errors."""
        error = subprocess.SubprocessError("Failed")
        assert ErrorClassifier.get_error_category(error) == "cli"

    def test_category_unknown(self):
        """Test category for unknown errors."""
        error = Exception("Something unexpected happened")
        assert ErrorClassifier.get_error_category(error) == "unknown"


# =============================================================================
# ErrorClassifier.classify_error Tests
# =============================================================================


class TestErrorClassifierClassify:
    """Tests for ErrorClassifier.classify_error method."""

    def test_classify_timeout(self):
        """Test classification of timeout errors."""
        error = TimeoutError("Timed out")
        should_fallback, category = ErrorClassifier.classify_error(error)
        assert should_fallback is True
        assert category == "timeout"

    def test_classify_rate_limit(self):
        """Test classification of rate limit errors."""
        error = Exception("Error: 429 rate limit exceeded")
        should_fallback, category = ErrorClassifier.classify_error(error)
        assert should_fallback is True
        assert category == "rate_limit"

    def test_classify_network(self):
        """Test classification of network errors."""
        error = ConnectionRefusedError("Refused")
        should_fallback, category = ErrorClassifier.classify_error(error)
        assert should_fallback is True
        assert category == "network"

    def test_classify_auth(self):
        """Test classification of auth errors."""
        error = Exception("401 Unauthorized: Invalid API key")
        should_fallback, category = ErrorClassifier.classify_error(error)
        assert should_fallback is True
        assert category == "auth"

    def test_classify_model(self):
        """Test classification of model errors."""
        error = Exception("Error: model_not_found for gpt-5")
        should_fallback, category = ErrorClassifier.classify_error(error)
        assert should_fallback is True
        assert category == "model"

    def test_classify_content_policy_no_fallback(self):
        """Test that content policy errors don't trigger fallback."""
        error = Exception("Content blocked by safety filter")
        should_fallback, category = ErrorClassifier.classify_error(error)
        assert should_fallback is False
        assert category == "content_policy"

    def test_classify_validation_no_fallback(self):
        """Test that validation errors don't trigger fallback."""
        error = Exception("Prompt too long: context_length exceeded")
        should_fallback, category = ErrorClassifier.classify_error(error)
        assert should_fallback is False
        assert category == "validation"

    def test_classify_cli(self):
        """Test classification of CLI errors."""
        error = subprocess.SubprocessError("CLI failed")
        should_fallback, category = ErrorClassifier.classify_error(error)
        assert should_fallback is True
        assert category == "cli"


# =============================================================================
# ErrorClassifier.classify_full Tests
# =============================================================================


class TestErrorClassifierClassifyFull:
    """Tests for ErrorClassifier.classify_full method."""

    def test_classify_full_timeout(self):
        """Test full classification of timeout errors."""
        error = TimeoutError("Request timed out after 30s")
        result = ErrorClassifier.classify_full(error)

        assert result.category == ErrorCategory.TIMEOUT
        assert result.severity == ErrorSeverity.WARNING
        assert result.action == RecoveryAction.RETRY
        assert result.should_fallback is True
        assert result.is_recoverable is True

    def test_classify_full_rate_limit_with_retry_after(self):
        """Test full classification extracts retry-after hint."""
        error = Exception("429 Rate limit exceeded. Retry after 60 seconds")
        result = ErrorClassifier.classify_full(error)

        assert result.category == ErrorCategory.RATE_LIMIT
        assert result.severity == ErrorSeverity.INFO
        assert result.action == RecoveryAction.WAIT
        assert result.retry_after == 60.0

    def test_classify_full_rate_limit_default_retry(self):
        """Test rate limit defaults to 60s retry."""
        error = Exception("429 Rate limit exceeded")
        result = ErrorClassifier.classify_full(error)

        assert result.category == ErrorCategory.RATE_LIMIT
        assert result.retry_after == 60.0

    def test_classify_full_network(self):
        """Test full classification of network errors."""
        error = ConnectionResetError("Connection reset by peer")
        result = ErrorClassifier.classify_full(error)

        assert result.category == ErrorCategory.NETWORK
        assert result.severity == ErrorSeverity.WARNING
        assert result.action == RecoveryAction.RETRY
        assert result.should_fallback is True

    def test_classify_full_os_error_with_errno(self):
        """Test full classification of OSError with network errno."""
        error = OSError(111, "Connection refused")
        result = ErrorClassifier.classify_full(error)

        # OSError with network errno is classified as NETWORK
        assert result.category == ErrorCategory.NETWORK
        assert result.should_fallback is True
        assert result.action == RecoveryAction.RETRY

    def test_classify_full_auth_critical(self):
        """Test auth errors are classified as critical."""
        error = Exception("401 Unauthorized: invalid_api_key")
        result = ErrorClassifier.classify_full(error)

        assert result.category == ErrorCategory.AUTH
        assert result.severity == ErrorSeverity.CRITICAL
        assert result.action == RecoveryAction.ESCALATE
        assert result.should_fallback is True  # Try different agent
        assert result.is_recoverable is False

    def test_classify_full_content_policy_abort(self):
        """Test content policy triggers ABORT."""
        error = Exception("Content blocked by content policy filter")
        result = ErrorClassifier.classify_full(error)

        assert result.category == ErrorCategory.CONTENT_POLICY
        assert result.severity == ErrorSeverity.ERROR
        assert result.action == RecoveryAction.ABORT
        assert result.should_fallback is False
        assert result.is_recoverable is False

    def test_classify_full_model(self):
        """Test full classification of model errors."""
        error = Exception("Error: model_not_found - gpt-5 does not exist")
        result = ErrorClassifier.classify_full(error)

        assert result.category == ErrorCategory.MODEL
        assert result.action == RecoveryAction.FALLBACK
        assert result.should_fallback is True

    def test_classify_full_validation_abort(self):
        """Test validation errors trigger ABORT."""
        error = Exception("Prompt too long: max_tokens exceeded")
        result = ErrorClassifier.classify_full(error)

        assert result.category == ErrorCategory.VALIDATION
        assert result.action == RecoveryAction.ABORT
        assert result.should_fallback is False
        assert result.is_recoverable is False

    def test_classify_full_cli(self):
        """Test full classification of CLI errors."""
        error = subprocess.SubprocessError("Process exited with code 1")
        result = ErrorClassifier.classify_full(error)

        assert result.category == ErrorCategory.CLI
        assert result.action == RecoveryAction.FALLBACK

    def test_classify_full_runtime_cli(self):
        """Test RuntimeError with CLI message."""
        error = RuntimeError("CLI command failed with return code 127")
        result = ErrorClassifier.classify_full(error)

        assert result.category == ErrorCategory.CLI
        assert result.action == RecoveryAction.FALLBACK

    def test_classify_full_unknown(self):
        """Test unknown errors default classification."""
        error = Exception("Something weird happened")
        result = ErrorClassifier.classify_full(error)

        assert result.category == ErrorCategory.UNKNOWN
        assert result.severity == ErrorSeverity.WARNING
        assert result.action == RecoveryAction.RETRY


# =============================================================================
# classify_cli_error Function Tests
# =============================================================================


class TestClassifyCLIError:
    """Tests for classify_cli_error function."""

    def test_classify_rate_limit(self):
        """Test CLI error classified as rate limit."""
        error = classify_cli_error(
            returncode=1,
            stderr="Error: 429 rate limit exceeded",
            stdout="",
            agent_name="claude",
        )
        assert isinstance(error, CLIAgentError)
        assert "Rate limit" in str(error)
        assert error.recoverable is True

    def test_classify_timeout(self):
        """Test CLI error classified as timeout from return code."""
        error = classify_cli_error(
            returncode=-9,  # SIGKILL
            stderr="",
            stdout="",
            agent_name="claude",
            timeout_seconds=30.0,
        )
        assert isinstance(error, CLITimeoutError)
        assert "30" in str(error)  # Accepts both 30s and 30.0s

    def test_classify_timeout_from_stderr(self):
        """Test CLI error classified as timeout from stderr."""
        error = classify_cli_error(
            returncode=1,
            stderr="Command timed out",
            stdout="",
            agent_name="claude",
        )
        assert isinstance(error, CLITimeoutError)

    def test_classify_command_not_found_127(self):
        """Test command not found from exit code 127."""
        error = classify_cli_error(
            returncode=127,
            stderr="bash: claude: command not found",
            stdout="",
            agent_name="claude",
        )
        assert isinstance(error, CLINotFoundError)

    def test_classify_command_not_found_message(self):
        """Test command not found from stderr message."""
        error = classify_cli_error(
            returncode=1,
            stderr="not found: claude",
            stdout="",
            agent_name="claude",
        )
        assert isinstance(error, CLINotFoundError)

    def test_classify_permission_denied_126(self):
        """Test permission denied from exit code 126."""
        error = classify_cli_error(
            returncode=126,
            stderr="Permission denied",
            stdout="",
            agent_name="claude",
        )
        assert isinstance(error, CLISubprocessError)
        assert "Permission denied" in str(error)

    def test_classify_empty_response(self):
        """Test empty response classified as parse error."""
        error = classify_cli_error(
            returncode=0,
            stderr="",
            stdout="   ",  # Whitespace only
            agent_name="claude",
        )
        assert isinstance(error, CLIParseError)
        assert "Empty response" in str(error)

    def test_classify_invalid_json(self):
        """Test invalid JSON classified as parse error."""
        error = classify_cli_error(
            returncode=0,
            stderr="",
            stdout="{invalid json",
            agent_name="claude",
        )
        assert isinstance(error, CLIParseError)
        assert "Invalid JSON" in str(error)

    def test_classify_json_error_response(self):
        """Test JSON with error field."""
        error = classify_cli_error(
            returncode=0,
            stderr="",
            stdout='{"error": "API Error: Rate limit exceeded"}',
            agent_name="claude",
        )
        assert isinstance(error, CLIAgentError)
        assert error.recoverable is True

    def test_classify_generic_failure(self):
        """Test generic subprocess failure."""
        error = classify_cli_error(
            returncode=1,
            stderr="Some error occurred",
            stdout="",
            agent_name="claude",
        )
        assert isinstance(error, CLISubprocessError)
        assert "code 1" in str(error)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_error_message(self):
        """Test classification of empty error message."""
        assert ErrorClassifier.is_rate_limit("") is False
        assert ErrorClassifier.is_network_error("") is False
        assert ErrorClassifier.get_error_category(Exception("")) == "unknown"

    def test_none_handling_in_patterns(self):
        """Test that pattern matching handles None-like cases."""
        # Should not raise
        error = Exception("")
        result = ErrorClassifier.classify_full(error)
        assert result.category == ErrorCategory.UNKNOWN

    def test_case_insensitive_matching(self):
        """Test that pattern matching is case-insensitive."""
        assert ErrorClassifier.is_rate_limit("RATE LIMIT EXCEEDED") is True
        assert ErrorClassifier.is_network_error("CONNECTION REFUSED") is True
        assert ErrorClassifier.is_auth_error("UNAUTHORIZED") is True

    def test_partial_pattern_matching(self):
        """Test that partial patterns are matched."""
        # "throttl" should match "throttled" and "throttling"
        assert ErrorClassifier.is_rate_limit("Request was throttled") is True
        assert ErrorClassifier.is_rate_limit("Throttling in effect") is True

    def test_combined_error_message(self):
        """Test classification when error contains multiple patterns."""
        # Rate limit takes priority
        error = Exception("Connection failed due to rate limit")
        _, category = ErrorClassifier.classify_error(error)
        assert category == "rate_limit"

    def test_network_errno_set(self):
        """Test that NETWORK_ERRNO contains expected values."""
        assert 111 in ErrorClassifier.NETWORK_ERRNO  # ECONNREFUSED
        assert 110 in ErrorClassifier.NETWORK_ERRNO  # ETIMEDOUT
        assert 104 in ErrorClassifier.NETWORK_ERRNO  # ECONNRESET

    def test_classify_cli_error_truncates_stderr(self):
        """Test that classify_cli_error truncates long stderr."""
        long_stderr = "x" * 1000
        error = classify_cli_error(
            returncode=1,
            stderr=long_stderr,
            stdout="",
            agent_name="claude",
        )
        # stderr should be truncated to 500 chars in the error
        assert len(error.stderr) <= 500

    def test_classify_cli_error_no_agent_name(self):
        """Test classify_cli_error works without agent name."""
        error = classify_cli_error(
            returncode=1,
            stderr="Error",
            stdout="",
            agent_name=None,
        )
        assert isinstance(error, CLISubprocessError)
