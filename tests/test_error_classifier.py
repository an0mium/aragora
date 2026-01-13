"""
Tests for error classification in aragora.agents.errors.classifier.

Tests error pattern matching, classification, and recovery recommendations.
"""

import pytest
import subprocess
from unittest.mock import MagicMock

from aragora.agents.errors.classifier import (
    ErrorCategory,
    ErrorSeverity,
    RecoveryAction,
    ErrorClassifier,
    ClassifiedError,
    classify_cli_error,
    RATE_LIMIT_PATTERNS,
    NETWORK_ERROR_PATTERNS,
    CLI_ERROR_PATTERNS,
    AUTH_ERROR_PATTERNS,
    VALIDATION_ERROR_PATTERNS,
    MODEL_ERROR_PATTERNS,
    CONTENT_POLICY_PATTERNS,
)
from aragora.agents.errors.exceptions import (
    AgentError,
    CLIAgentError,
    CLITimeoutError,
    CLISubprocessError,
    CLIParseError,
)


# =============================================================================
# ErrorCategory Enum Tests
# =============================================================================


class TestErrorCategory:
    """Tests for ErrorCategory enum."""

    def test_all_categories_defined(self):
        """All expected categories should be defined."""
        categories = [e.value for e in ErrorCategory]
        assert "timeout" in categories
        assert "rate_limit" in categories
        assert "network" in categories
        assert "cli" in categories
        assert "auth" in categories
        assert "validation" in categories
        assert "model" in categories
        assert "content_policy" in categories
        assert "unknown" in categories


class TestErrorSeverity:
    """Tests for ErrorSeverity enum."""

    def test_all_severities_defined(self):
        """All expected severities should be defined."""
        severities = [e.value for e in ErrorSeverity]
        assert "critical" in severities
        assert "error" in severities
        assert "warning" in severities
        assert "info" in severities


class TestRecoveryAction:
    """Tests for RecoveryAction enum."""

    def test_all_actions_defined(self):
        """All expected recovery actions should be defined."""
        actions = [e.value for e in RecoveryAction]
        assert "retry" in actions
        assert "retry_immediate" in actions
        assert "fallback" in actions
        assert "wait" in actions
        assert "abort" in actions
        assert "escalate" in actions


# =============================================================================
# Rate Limit Pattern Tests
# =============================================================================


class TestIsRateLimit:
    """Tests for ErrorClassifier.is_rate_limit."""

    @pytest.mark.parametrize(
        "message",
        [
            "rate limit exceeded",
            "Rate Limit Error",
            "error 429: too many requests",
            "quota exceeded for model",
            "resource_exhausted: quota limit",
            "billing issue - payment required",
        ],
    )
    def test_recognizes_rate_limit_patterns(self, message):
        """Should recognize rate limit patterns."""
        assert ErrorClassifier.is_rate_limit(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "invalid input",
            "model not found",
            "authentication failed",
            "network timeout",
        ],
    )
    def test_rejects_non_rate_limit_messages(self, message):
        """Should not match non-rate-limit messages."""
        assert ErrorClassifier.is_rate_limit(message) is False


# =============================================================================
# Network Error Pattern Tests
# =============================================================================


class TestIsNetworkError:
    """Tests for ErrorClassifier.is_network_error."""

    @pytest.mark.parametrize(
        "message",
        [
            "503 service unavailable",
            "connection refused",
            "connection timed out",
            "server overloaded",
            "DNS resolution failed",
        ],
    )
    def test_recognizes_network_patterns(self, message):
        """Should recognize network error patterns."""
        assert ErrorClassifier.is_network_error(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "invalid API key",
            "model not found",
            "rate limit exceeded",
        ],
    )
    def test_rejects_non_network_messages(self, message):
        """Should not match non-network messages."""
        assert ErrorClassifier.is_network_error(message) is False


# =============================================================================
# CLI Error Pattern Tests
# =============================================================================


class TestIsCLIError:
    """Tests for ErrorClassifier.is_cli_error."""

    @pytest.mark.parametrize(
        "message",
        [
            "command not found",
            "no such file or directory",
            "permission denied",
            "broken pipe",
        ],
    )
    def test_recognizes_cli_patterns(self, message):
        """Should recognize CLI error patterns."""
        assert ErrorClassifier.is_cli_error(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "invalid API key",
            "network timeout",
        ],
    )
    def test_rejects_non_cli_messages(self, message):
        """Should not match non-CLI messages."""
        assert ErrorClassifier.is_cli_error(message) is False


# =============================================================================
# Auth Error Pattern Tests
# =============================================================================


class TestIsAuthError:
    """Tests for ErrorClassifier.is_auth_error."""

    @pytest.mark.parametrize(
        "message",
        [
            "authentication failed",
            "invalid API key",
            "401 unauthorized",
            "403 forbidden",
            "permission denied",
        ],
    )
    def test_recognizes_auth_patterns(self, message):
        """Should recognize auth error patterns."""
        assert ErrorClassifier.is_auth_error(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "model not found",
            "network timeout",
            "rate limit exceeded",
        ],
    )
    def test_rejects_non_auth_messages(self, message):
        """Should not match non-auth messages."""
        assert ErrorClassifier.is_auth_error(message) is False


# =============================================================================
# Validation Error Pattern Tests
# =============================================================================


class TestIsValidationError:
    """Tests for ErrorClassifier.is_validation_error."""

    @pytest.mark.parametrize(
        "message",
        [
            "context length exceeded",
            "context_length error",
            "max_tokens exceeded",
            "invalid input format",
        ],
    )
    def test_recognizes_validation_patterns(self, message):
        """Should recognize validation error patterns."""
        assert ErrorClassifier.is_validation_error(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "model not found",
            "network timeout",
        ],
    )
    def test_rejects_non_validation_messages(self, message):
        """Should not match non-validation messages."""
        assert ErrorClassifier.is_validation_error(message) is False


# =============================================================================
# Model Error Pattern Tests
# =============================================================================


class TestIsModelError:
    """Tests for ErrorClassifier.is_model_error."""

    @pytest.mark.parametrize(
        "message",
        [
            "model not found",
            "model unavailable",
            "404 model does not exist",
        ],
    )
    def test_recognizes_model_patterns(self, message):
        """Should recognize model error patterns."""
        assert ErrorClassifier.is_model_error(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "network timeout",
            "rate limit exceeded",
        ],
    )
    def test_rejects_non_model_messages(self, message):
        """Should not match non-model messages."""
        assert ErrorClassifier.is_model_error(message) is False


# =============================================================================
# Content Policy Error Pattern Tests
# =============================================================================


class TestIsContentPolicyError:
    """Tests for ErrorClassifier.is_content_policy_error."""

    @pytest.mark.parametrize(
        "message",
        [
            "content policy violation",
            "safety filter triggered",
            "content blocked by moderation",
            "harmful content detected",
        ],
    )
    def test_recognizes_content_policy_patterns(self, message):
        """Should recognize content policy error patterns."""
        assert ErrorClassifier.is_content_policy_error(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            "network timeout",
            "rate limit exceeded",
            "model not found",
        ],
    )
    def test_rejects_non_content_policy_messages(self, message):
        """Should not match non-content-policy messages."""
        assert ErrorClassifier.is_content_policy_error(message) is False


# =============================================================================
# Should Fallback Tests
# =============================================================================


class TestShouldFallback:
    """Tests for ErrorClassifier.should_fallback."""

    def test_fallback_on_rate_limit(self):
        """Should trigger fallback on rate limit error."""
        error = Exception("Error 429: rate limit exceeded")
        assert ErrorClassifier.should_fallback(error) is True

    def test_fallback_on_network_error(self):
        """Should trigger fallback on network error."""
        error = Exception("Connection refused")
        assert ErrorClassifier.should_fallback(error) is True

    def test_no_fallback_on_validation_error(self):
        """Should not fallback on validation error (user must fix)."""
        error = Exception("Context length exceeded maximum")
        assert ErrorClassifier.should_fallback(error) is False

    def test_no_fallback_on_content_policy(self):
        """Should not fallback on content policy (would fail elsewhere too)."""
        error = Exception("Content policy violation")
        assert ErrorClassifier.should_fallback(error) is False

    def test_fallback_on_cli_error(self):
        """Should trigger fallback on CLI error with fallback pattern."""
        # CLIAgentError with a pattern from CLI_ERROR_PATTERNS
        error = CLIAgentError("command not found")
        assert ErrorClassifier.should_fallback(error) is True


# =============================================================================
# Get Error Category Tests
# =============================================================================


class TestGetErrorCategory:
    """Tests for ErrorClassifier.get_error_category."""

    def test_categorizes_rate_limit(self):
        """Should categorize rate limit errors."""
        error = Exception("Error 429: rate limit exceeded")
        category = ErrorClassifier.get_error_category(error)
        assert category == "rate_limit"

    def test_categorizes_network_error(self):
        """Should categorize network errors."""
        error = Exception("Connection refused")
        category = ErrorClassifier.get_error_category(error)
        assert category == "network"

    def test_categorizes_timeout(self):
        """Should categorize timeout errors."""
        error = TimeoutError("Request timed out")
        category = ErrorClassifier.get_error_category(error)
        assert category == "timeout"

    def test_categorizes_unknown(self):
        """Should categorize unknown errors."""
        error = Exception("Some unknown error")
        category = ErrorClassifier.get_error_category(error)
        assert category == "unknown"


# =============================================================================
# Classify Error Tests
# =============================================================================


class TestClassifyError:
    """Tests for ErrorClassifier.classify_error."""

    def test_classifies_rate_limit(self):
        """Should classify rate limit with fallback=True."""
        error = Exception("Error 429: rate limit exceeded")
        should_fallback, category = ErrorClassifier.classify_error(error)
        assert should_fallback is True
        assert category == "rate_limit"

    def test_classifies_validation(self):
        """Should classify validation with fallback=False."""
        error = Exception("Context length exceeded")
        should_fallback, category = ErrorClassifier.classify_error(error)
        assert should_fallback is False
        assert category == "validation"


# =============================================================================
# Classify Full Tests
# =============================================================================


class TestClassifyFull:
    """Tests for ErrorClassifier.classify_full."""

    def test_returns_classified_error(self):
        """Should return ClassifiedError with all fields."""
        error = Exception("Error 429: rate limit exceeded")
        classified = ErrorClassifier.classify_full(error)

        assert isinstance(classified, ClassifiedError)
        assert classified.category == ErrorCategory.RATE_LIMIT
        assert classified.should_fallback is True
        assert classified.action == RecoveryAction.WAIT

    def test_network_error_classification(self):
        """Should classify network errors correctly."""
        error = Exception("503 service unavailable")
        classified = ErrorClassifier.classify_full(error)

        assert classified.category == ErrorCategory.NETWORK
        assert classified.should_fallback is True
        assert classified.action == RecoveryAction.RETRY

    def test_auth_error_classification(self):
        """Should classify auth errors correctly."""
        error = Exception("401 unauthorized")
        classified = ErrorClassifier.classify_full(error)

        assert classified.category == ErrorCategory.AUTH
        assert classified.should_fallback is True  # Auth errors can fallback to other providers
        assert classified.action == RecoveryAction.ESCALATE

    def test_validation_error_classification(self):
        """Should classify validation errors correctly."""
        error = Exception("context length exceeded")
        classified = ErrorClassifier.classify_full(error)

        assert classified.category == ErrorCategory.VALIDATION
        assert classified.should_fallback is False
        assert classified.action == RecoveryAction.ABORT


# =============================================================================
# ClassifiedError Tests
# =============================================================================


class TestClassifiedError:
    """Tests for ClassifiedError dataclass."""

    def test_is_recoverable_true(self):
        """Should return True for recoverable errors."""
        classified = ClassifiedError(
            category=ErrorCategory.RATE_LIMIT,
            severity=ErrorSeverity.WARNING,
            action=RecoveryAction.WAIT,
            should_fallback=True,
            message="Rate limit exceeded",
        )
        assert classified.is_recoverable is True

    def test_is_recoverable_false(self):
        """Should return False for non-recoverable errors."""
        classified = ClassifiedError(
            category=ErrorCategory.AUTH,
            severity=ErrorSeverity.ERROR,
            action=RecoveryAction.ABORT,
            should_fallback=False,
            message="Invalid API key",
        )
        assert classified.is_recoverable is False

    def test_category_str(self):
        """Should return category string value."""
        classified = ClassifiedError(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.WARNING,
            action=RecoveryAction.RETRY,
            should_fallback=True,
            message="Connection refused",
        )
        assert classified.category_str == "network"


# =============================================================================
# CLI Error Classification Tests
# =============================================================================


class TestClassifyCLIError:
    """Tests for classify_cli_error function."""

    def test_classifies_timeout(self):
        """Should return CLITimeoutError for timeout."""
        error = classify_cli_error(
            returncode=-9,  # SIGKILL
            stdout="",
            stderr="Command timed out",
        )
        assert isinstance(error, CLITimeoutError)

    def test_classifies_not_found(self):
        """Should return CLIAgentError for command not found."""
        from aragora.agents.errors.exceptions import CLINotFoundError

        error = classify_cli_error(
            returncode=127,
            stdout="",
            stderr="command not found",
        )
        assert isinstance(error, CLINotFoundError)

    def test_classifies_rate_limit_in_output(self):
        """Should detect rate limit in CLI output."""
        error = classify_cli_error(
            returncode=1,
            stdout="",
            stderr="Error: rate limit exceeded",
        )
        assert isinstance(error, CLIAgentError)
        assert "rate limit" in str(error).lower()

    def test_classifies_subprocess_error(self):
        """Should classify subprocess errors."""
        error = classify_cli_error(
            returncode=1,
            stdout="",
            stderr="Some error occurred",
        )
        assert isinstance(error, (CLIAgentError, CLISubprocessError))


# =============================================================================
# Pattern Constants Tests
# =============================================================================


class TestPatternConstants:
    """Tests for error pattern constants."""

    def test_rate_limit_patterns_not_empty(self):
        """Rate limit patterns should be defined."""
        assert len(RATE_LIMIT_PATTERNS) > 0
        assert "rate limit" in RATE_LIMIT_PATTERNS

    def test_network_patterns_not_empty(self):
        """Network patterns should be defined."""
        assert len(NETWORK_ERROR_PATTERNS) > 0
        assert "503" in NETWORK_ERROR_PATTERNS

    def test_cli_patterns_not_empty(self):
        """CLI patterns should be defined."""
        assert len(CLI_ERROR_PATTERNS) > 0

    def test_auth_patterns_not_empty(self):
        """Auth patterns should be defined."""
        assert len(AUTH_ERROR_PATTERNS) > 0

    def test_validation_patterns_not_empty(self):
        """Validation patterns should be defined."""
        assert len(VALIDATION_ERROR_PATTERNS) > 0

    def test_model_patterns_not_empty(self):
        """Model patterns should be defined."""
        assert len(MODEL_ERROR_PATTERNS) > 0

    def test_content_policy_patterns_not_empty(self):
        """Content policy patterns should be defined."""
        assert len(CONTENT_POLICY_PATTERNS) > 0
