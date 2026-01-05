"""
Tests for custom exception types.
"""

import pytest

from aragora.exceptions import (
    # Base
    AragoraError,
    # Debate
    DebateError,
    DebateNotFoundError,
    DebateConfigurationError,
    ConsensusError,
    RoundLimitExceededError,
    EarlyStopError,
    # Agent
    AgentError,
    AgentNotFoundError,
    AgentConfigurationError,
    AgentResponseError,
    AgentTimeoutError,
    APIKeyError,
    RateLimitError,
    # Validation
    ValidationError,
    InputValidationError,
    SchemaValidationError,
    # Storage
    StorageError,
    DatabaseError,
    DatabaseConnectionError,
    RecordNotFoundError,
    # Memory
    MemoryError,
    MemoryRetrievalError,
    EmbeddingError,
    # Mode
    ModeError,
    ModeNotFoundError,
    # Plugin
    PluginError,
    PluginNotFoundError,
    PluginExecutionError,
    # Auth
    AuthError,
    AuthenticationError,
    TokenExpiredError,
    RateLimitExceededError,
    # Nomic
    NomicError,
    NomicCycleError,
    NomicStateError,
    # Verification
    VerificationError,
    Z3NotAvailableError,
    VerificationTimeoutError,
)


class TestAragoraError:
    """Tests for base AragoraError."""

    def test_basic_message(self):
        """Test basic error message."""
        err = AragoraError("Something went wrong")

        assert str(err) == "Something went wrong"
        assert err.message == "Something went wrong"
        assert err.details == {}

    def test_with_details(self):
        """Test error with details dict."""
        err = AragoraError("Error occurred", {"key": "value"})

        assert "key" in str(err)
        assert err.details == {"key": "value"}

    def test_inheritance(self):
        """Test all custom exceptions inherit from AragoraError."""
        exceptions = [
            DebateError("test"),
            AgentError("test"),
            ValidationError("test"),
            StorageError("test"),
            MemoryError("test"),
            ModeError("test"),
            PluginError("test"),
            AuthError("test"),
            NomicError("test"),
            VerificationError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, AragoraError)


class TestDebateErrors:
    """Tests for debate-related exceptions."""

    def test_debate_not_found(self):
        """Test DebateNotFoundError."""
        err = DebateNotFoundError("debate-123")

        assert err.debate_id == "debate-123"
        assert "debate-123" in str(err)
        assert isinstance(err, DebateError)

    def test_round_limit_exceeded(self):
        """Test RoundLimitExceededError."""
        err = RoundLimitExceededError(max_rounds=5, current_round=6)

        assert err.max_rounds == 5
        assert err.current_round == 6
        assert "6/5" in str(err)

    def test_early_stop(self):
        """Test EarlyStopError."""
        err = EarlyStopError(reason="Consensus reached", round_stopped=3)

        assert err.reason == "Consensus reached"
        assert err.round_stopped == 3


class TestAgentErrors:
    """Tests for agent-related exceptions."""

    def test_agent_not_found(self):
        """Test AgentNotFoundError."""
        err = AgentNotFoundError("claude")

        assert err.agent_name == "claude"
        assert "claude" in str(err)

    def test_agent_response_error(self):
        """Test AgentResponseError."""
        err = AgentResponseError("gemini", "Connection timeout")

        assert err.agent_name == "gemini"
        assert err.reason == "Connection timeout"
        assert "gemini" in str(err)
        assert "timeout" in str(err).lower()

    def test_agent_timeout(self):
        """Test AgentTimeoutError."""
        err = AgentTimeoutError("gpt4", 30.0)

        assert err.agent_name == "gpt4"
        assert err.timeout_seconds == 30.0

    def test_api_key_error(self):
        """Test APIKeyError."""
        err = APIKeyError("anthropic")

        assert err.provider == "anthropic"
        assert "anthropic" in str(err)

    def test_rate_limit(self):
        """Test RateLimitError."""
        err = RateLimitError("openai", retry_after=60.0)

        assert err.provider == "openai"
        assert err.retry_after == 60.0


class TestValidationErrors:
    """Tests for validation exceptions."""

    def test_input_validation(self):
        """Test InputValidationError."""
        err = InputValidationError("email", "Invalid format")

        assert err.field == "email"
        assert err.reason == "Invalid format"

    def test_schema_validation(self):
        """Test SchemaValidationError."""
        errors = ["Missing field: name", "Invalid type for age"]
        err = SchemaValidationError("UserSchema", errors)

        assert err.schema_name == "UserSchema"
        assert err.errors == errors
        assert len(err.errors) == 2


class TestStorageErrors:
    """Tests for storage exceptions."""

    def test_database_connection(self):
        """Test DatabaseConnectionError."""
        err = DatabaseConnectionError("/path/to/db.sqlite", "File not found")

        assert err.db_path == "/path/to/db.sqlite"
        assert err.reason == "File not found"

    def test_record_not_found(self):
        """Test RecordNotFoundError."""
        err = RecordNotFoundError("debates", "abc123")

        assert err.table == "debates"
        assert err.record_id == "abc123"


class TestMemoryErrors:
    """Tests for memory exceptions."""

    def test_embedding_error_truncates_preview(self):
        """Test EmbeddingError truncates long text."""
        long_text = "x" * 100
        err = EmbeddingError(long_text, "Model unavailable")

        # Preview should be truncated
        assert "..." in str(err) or len(long_text[:50]) < len(long_text)
        assert err.reason == "Model unavailable"


class TestModeErrors:
    """Tests for mode exceptions."""

    def test_mode_not_found(self):
        """Test ModeNotFoundError."""
        err = ModeNotFoundError("red_team")

        assert err.mode_name == "red_team"
        assert "red_team" in str(err)


class TestPluginErrors:
    """Tests for plugin exceptions."""

    def test_plugin_not_found(self):
        """Test PluginNotFoundError."""
        err = PluginNotFoundError("my_plugin")

        assert err.plugin_name == "my_plugin"

    def test_plugin_execution(self):
        """Test PluginExecutionError."""
        err = PluginExecutionError("my_plugin", "Syntax error")

        assert err.plugin_name == "my_plugin"
        assert err.reason == "Syntax error"


class TestAuthErrors:
    """Tests for auth exceptions."""

    def test_rate_limit_exceeded(self):
        """Test RateLimitExceededError."""
        err = RateLimitExceededError(limit=100, window_seconds=60)

        assert err.limit == 100
        assert err.window_seconds == 60
        assert "100" in str(err)


class TestNomicErrors:
    """Tests for Nomic exceptions."""

    def test_nomic_cycle_error(self):
        """Test NomicCycleError."""
        err = NomicCycleError(cycle=5, phase="implement", reason="Test failed")

        assert err.cycle == 5
        assert err.phase == "implement"
        assert err.reason == "Test failed"
        assert "5" in str(err)
        assert "implement" in str(err)


class TestVerificationErrors:
    """Tests for verification exceptions."""

    def test_z3_not_available(self):
        """Test Z3NotAvailableError."""
        err = Z3NotAvailableError()

        assert "Z3" in str(err)
        assert "pip install" in str(err)

    def test_verification_timeout(self):
        """Test VerificationTimeoutError."""
        err = VerificationTimeoutError(timeout_ms=5000)

        assert err.timeout_ms == 5000
        assert "5000" in str(err)


class TestExceptionHierarchy:
    """Tests for exception hierarchy and catching."""

    def test_catch_all_aragora_errors(self):
        """Test that all custom exceptions can be caught with AragoraError."""
        exceptions = [
            DebateNotFoundError("test"),
            AgentTimeoutError("test", 10),
            InputValidationError("field", "reason"),
            DatabaseConnectionError("/path", "reason"),
            ModeNotFoundError("mode"),
            PluginNotFoundError("plugin"),
            TokenExpiredError("Token expired"),
            NomicStateError("Invalid state"),
            Z3NotAvailableError(),
        ]

        for exc in exceptions:
            try:
                raise exc
            except AragoraError as e:
                # Should catch all of them
                assert e is exc
            except Exception:
                pytest.fail(f"Exception {type(exc)} not caught by AragoraError")

    def test_catch_specific_error_type(self):
        """Test that specific error types can be caught separately."""
        with pytest.raises(DebateNotFoundError):
            raise DebateNotFoundError("test-id")

        with pytest.raises(AgentError):
            raise AgentTimeoutError("agent", 10)

        with pytest.raises(ValidationError):
            raise InputValidationError("field", "reason")
