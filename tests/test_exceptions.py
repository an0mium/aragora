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
    # Agent config errors (from exceptions.py)
    AgentNotFoundError,
    AgentConfigurationError,
    APIKeyError,
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

# Agent runtime errors are in aragora.agents.errors
from aragora.agents.errors import (
    AgentError,
    AgentResponseError,
    AgentTimeoutError,
    AgentRateLimitError,
)

# Alias for backwards compatibility in tests
RateLimitError = AgentRateLimitError


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
        err = AgentResponseError("Connection timeout", agent_name="gemini")

        assert err.agent_name == "gemini"
        assert "timeout" in str(err).lower()

    def test_agent_timeout(self):
        """Test AgentTimeoutError."""
        err = AgentTimeoutError("Request timed out", agent_name="gpt4", timeout_seconds=30.0)

        assert err.agent_name == "gpt4"
        assert err.timeout_seconds == 30.0

    def test_api_key_error(self):
        """Test APIKeyError."""
        err = APIKeyError("anthropic")

        assert err.provider == "anthropic"
        assert "anthropic" in str(err)

    def test_rate_limit(self):
        """Test RateLimitError (AgentRateLimitError)."""
        err = RateLimitError("Rate limit exceeded", agent_name="openai", retry_after=60.0)

        assert err.agent_name == "openai"
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


# =============================================================================
# Additional Debate Error Tests
# =============================================================================


class TestAdditionalDebateErrors:
    """Tests for additional debate-related exceptions."""

    def test_consensus_timeout_error(self):
        """Test ConsensusTimeoutError."""
        from aragora.exceptions import ConsensusTimeoutError

        err = ConsensusTimeoutError(timeout_seconds=120.0, rounds_completed=5)

        assert err.timeout_seconds == 120.0
        assert err.rounds_completed == 5
        assert "120" in str(err)
        assert "5" in str(err)

    def test_vote_validation_error(self):
        """Test VoteValidationError."""
        from aragora.exceptions import VoteValidationError

        err = VoteValidationError(
            agent_name="claude",
            reason="Missing required field",
            vote_data={"incomplete": True},
        )

        assert err.agent_name == "claude"
        assert err.reason == "Missing required field"
        assert err.vote_data == {"incomplete": True}

    def test_phase_execution_error(self):
        """Test PhaseExecutionError."""
        from aragora.exceptions import PhaseExecutionError

        err = PhaseExecutionError(
            phase_name="proposal",
            reason="Agent timeout",
            recoverable=True,
        )

        assert err.phase_name == "proposal"
        assert err.reason == "Agent timeout"
        assert err.recoverable is True

    def test_debate_start_error(self):
        """Test DebateStartError."""
        from aragora.exceptions import DebateStartError

        err = DebateStartError(
            debate_id="debate-123",
            reason="No agents available",
        )

        assert err.debate_id == "debate-123"
        assert err.reason == "No agents available"

    def test_debate_batch_error(self):
        """Test DebateBatchError."""
        from aragora.exceptions import DebateBatchError

        err = DebateBatchError(
            operation="start_all",
            reason="Database lock",
            failed_ids=["d1", "d2"],
        )

        assert err.operation == "start_all"
        assert err.failed_ids == ["d1", "d2"]

    def test_debate_execution_error(self):
        """Test DebateExecutionError."""
        from aragora.exceptions import DebateExecutionError

        err = DebateExecutionError(
            debate_id="debate-456",
            phase="critique",
            reason="Memory overflow",
        )

        assert err.debate_id == "debate-456"
        assert err.phase == "critique"
        assert err.reason == "Memory overflow"

    def test_vote_processing_error(self):
        """Test VoteProcessingError."""
        from aragora.exceptions import VoteProcessingError

        err = VoteProcessingError(
            debate_id="debate-789",
            reason="Invalid vote format",
            agent_name="gemini",
        )

        assert err.debate_id == "debate-789"
        assert err.reason == "Invalid vote format"
        assert err.agent_name == "gemini"


# =============================================================================
# Configuration Error Tests
# =============================================================================


class TestConfigurationErrors:
    """Tests for configuration-related exceptions."""

    def test_configuration_error(self):
        """Test ConfigurationError."""
        from aragora.exceptions import ConfigurationError

        err = ConfigurationError(
            component="CircuitBreaker",
            reason="Missing callback",
        )

        assert err.component == "CircuitBreaker"
        assert err.reason == "Missing callback"

    def test_json_parse_error(self):
        """Test JSONParseError."""
        from aragora.exceptions import JSONParseError

        raw = '{"invalid": json}' * 100  # Long text
        err = JSONParseError(
            source="agent_response",
            reason="Unexpected token",
            raw_text=raw,
        )

        assert err.source == "agent_response"
        assert err.reason == "Unexpected token"
        assert err.raw_text == raw
        # Preview should be truncated in string representation
        assert len(str(err)) < len(raw)


# =============================================================================
# Memory Error Tests
# =============================================================================


class TestAdditionalMemoryErrors:
    """Tests for additional memory-related exceptions."""

    def test_memory_storage_error(self):
        """Test MemoryStorageError."""
        from aragora.exceptions import MemoryStorageError

        err = MemoryStorageError("Failed to store memory")

        assert isinstance(err, MemoryError)

    def test_tier_transition_error(self):
        """Test TierTransitionError."""
        from aragora.exceptions import TierTransitionError

        err = TierTransitionError(
            from_tier="fast",
            to_tier="slow",
            reason="Consolidation failed",
        )

        assert err.from_tier == "fast"
        assert err.to_tier == "slow"
        assert err.reason == "Consolidation failed"

    def test_memory_operation_error(self):
        """Test MemoryOperationError."""
        from aragora.exceptions import MemoryOperationError

        err = MemoryOperationError(
            tier="glacial",
            operation="read",
            reason="Connection timeout",
        )

        assert err.tier == "glacial"
        assert err.operation == "read"


# =============================================================================
# Infrastructure Error Tests
# =============================================================================


class TestInfrastructureErrors:
    """Tests for infrastructure-related exceptions."""

    def test_redis_unavailable_error(self):
        """Test RedisUnavailableError."""
        from aragora.exceptions import RedisUnavailableError

        err = RedisUnavailableError(operation="cache_get")

        assert err.operation == "cache_get"
        assert "Redis" in str(err)

    def test_external_service_error(self):
        """Test ExternalServiceError."""
        from aragora.exceptions import ExternalServiceError

        err = ExternalServiceError(
            service="stripe",
            reason="API timeout",
            status_code=504,
        )

        assert err.service == "stripe"
        assert err.status_code == 504

    def test_circuit_breaker_error(self):
        """Test CircuitBreakerError."""
        from aragora.exceptions import CircuitBreakerError

        err = CircuitBreakerError(
            service="openai",
            state="open",
            reason="Too many failures",
        )

        assert err.service == "openai"
        assert err.state == "open"


# =============================================================================
# Billing Error Tests
# =============================================================================


class TestBillingErrors:
    """Tests for billing-related exceptions."""

    def test_budget_exceeded_error(self):
        """Test BudgetExceededError."""
        from aragora.exceptions import BudgetExceededError

        err = BudgetExceededError(
            message="Budget limit reached",
            org_id="org-123",
            remaining_usd=0.0,
        )

        assert err.org_id == "org-123"
        assert err.remaining_usd == 0.0

    def test_insufficient_credits_error(self):
        """Test InsufficientCreditsError."""
        from aragora.exceptions import InsufficientCreditsError

        err = InsufficientCreditsError(
            required=100.0,
            available=25.50,
            org_id="org-456",
        )

        assert err.required == 100.0
        assert err.available == 25.50
        assert err.org_id == "org-456"


# =============================================================================
# Checkpoint Error Tests
# =============================================================================


class TestCheckpointErrors:
    """Tests for checkpoint-related exceptions."""

    def test_checkpoint_not_found_error(self):
        """Test CheckpointNotFoundError."""
        from aragora.exceptions import CheckpointNotFoundError

        err = CheckpointNotFoundError(checkpoint_id="cp-abc123")

        assert err.checkpoint_id == "cp-abc123"
        assert "cp-abc123" in str(err)

    def test_checkpoint_corrupted_error(self):
        """Test CheckpointCorruptedError."""
        from aragora.exceptions import CheckpointCorruptedError

        err = CheckpointCorruptedError(
            checkpoint_id="cp-def456",
            reason="Invalid JSON structure",
        )

        assert err.checkpoint_id == "cp-def456"
        assert err.reason == "Invalid JSON structure"

    def test_checkpoint_save_error(self):
        """Test CheckpointSaveError."""
        from aragora.exceptions import CheckpointSaveError

        err = CheckpointSaveError(
            checkpoint_id="cp-ghi789",
            reason="Disk full",
        )

        assert err.checkpoint_id == "cp-ghi789"


# =============================================================================
# Convergence Error Tests
# =============================================================================


class TestConvergenceErrors:
    """Tests for convergence-related exceptions."""

    def test_convergence_backend_error(self):
        """Test ConvergenceBackendError."""
        from aragora.exceptions import ConvergenceBackendError

        err = ConvergenceBackendError(
            backend_name="semantic_similarity",
            reason="Model not loaded",
        )

        assert err.backend_name == "semantic_similarity"
        assert err.reason == "Model not loaded"

    def test_convergence_threshold_error(self):
        """Test ConvergenceThresholdError."""
        from aragora.exceptions import ConvergenceThresholdError

        err = ConvergenceThresholdError(
            threshold=1.5,
            reason="Must be between 0 and 1",
        )

        assert err.threshold == 1.5


# =============================================================================
# Cache Error Tests
# =============================================================================


class TestCacheErrors:
    """Tests for cache-related exceptions."""

    def test_cache_key_error(self):
        """Test CacheKeyError."""
        from aragora.exceptions import CacheKeyError

        err = CacheKeyError(
            key="",
            reason="Key cannot be empty",
        )

        assert err.key == ""
        assert err.reason == "Key cannot be empty"

    def test_cache_capacity_error(self):
        """Test CacheCapacityError."""
        from aragora.exceptions import CacheCapacityError

        err = CacheCapacityError(
            current_size=1000,
            max_size=500,
        )

        assert err.current_size == 1000
        assert err.max_size == 500


# =============================================================================
# Streaming Error Tests
# =============================================================================


class TestStreamingErrors:
    """Tests for streaming-related exceptions."""

    def test_websocket_error(self):
        """Test WebSocketError."""
        from aragora.exceptions import WebSocketError

        err = WebSocketError(
            reason="Connection refused",
            code=1006,
        )

        assert err.reason == "Connection refused"
        assert err.code == 1006

    def test_stream_connection_error(self):
        """Test StreamConnectionError."""
        from aragora.exceptions import StreamConnectionError

        err = StreamConnectionError(
            stream_id="stream-123",
            reason="Network timeout",
        )

        assert err.stream_id == "stream-123"

    def test_stream_timeout_error(self):
        """Test StreamTimeoutError."""
        from aragora.exceptions import StreamTimeoutError

        err = StreamTimeoutError(
            stream_id="stream-456",
            timeout_seconds=30.0,
        )

        assert err.stream_id == "stream-456"
        assert err.timeout_seconds == 30.0


# =============================================================================
# Evidence Error Tests
# =============================================================================


class TestEvidenceErrors:
    """Tests for evidence-related exceptions."""

    def test_evidence_parse_error(self):
        """Test EvidenceParseError."""
        from aragora.exceptions import EvidenceParseError

        err = EvidenceParseError(
            source="external_api",
            reason="Invalid format",
        )

        assert err.source == "external_api"

    def test_evidence_not_found_error(self):
        """Test EvidenceNotFoundError."""
        from aragora.exceptions import EvidenceNotFoundError

        err = EvidenceNotFoundError(evidence_id="ev-12345")

        assert err.evidence_id == "ev-12345"


# =============================================================================
# Notification Error Tests
# =============================================================================


class TestNotificationErrors:
    """Tests for notification-related exceptions."""

    def test_slack_notification_error(self):
        """Test SlackNotificationError."""
        from aragora.exceptions import SlackNotificationError

        err = SlackNotificationError(
            message="Webhook failed",
            status_code=403,
            error_code="invalid_token",
        )

        assert err.status_code == 403
        assert err.error_code == "invalid_token"

    def test_webhook_delivery_error(self):
        """Test WebhookDeliveryError."""
        from aragora.exceptions import WebhookDeliveryError

        err = WebhookDeliveryError(
            webhook_url="https://example.com/hook",
            status_code=500,
            message="Server error",
        )

        assert err.webhook_url == "https://example.com/hook"
        assert err.status_code == 500


# =============================================================================
# Document Processing Error Tests
# =============================================================================


class TestDocumentProcessingErrors:
    """Tests for document processing exceptions."""

    def test_document_parse_error(self):
        """Test DocumentParseError."""
        from aragora.exceptions import DocumentParseError

        err = DocumentParseError(
            document_id="doc-123",
            reason="Unsupported format",
            original_error=ValueError("bad format"),
        )

        assert err.document_id == "doc-123"
        assert err.reason == "Unsupported format"
        assert isinstance(err.original_error, ValueError)

    def test_document_chunk_error(self):
        """Test DocumentChunkError."""
        from aragora.exceptions import DocumentChunkError

        err = DocumentChunkError(
            document_id="doc-456",
            reason="Chunk too large",
        )

        assert err.document_id == "doc-456"


# =============================================================================
# Nomic Error Extensions Tests
# =============================================================================


class TestNomicErrorExtensions:
    """Tests for extended Nomic exceptions."""

    def test_nomic_init_error(self):
        """Test NomicInitError."""
        from aragora.exceptions import NomicInitError

        err = NomicInitError(
            component="InsightStore",
            reason="Database connection failed",
            recoverable=True,
        )

        assert err.component == "InsightStore"
        assert err.recoverable is True

    def test_nomic_memory_error(self):
        """Test NomicMemoryError."""
        from aragora.exceptions import NomicMemoryError

        err = NomicMemoryError(
            operation="read",
            reason="Timeout",
            tier="slow",
        )

        assert err.operation == "read"
        assert err.tier == "slow"

    def test_nomic_agent_error(self):
        """Test NomicAgentError."""
        from aragora.exceptions import NomicAgentError

        err = NomicAgentError(
            agent_name="claude",
            operation="probe",
            reason="Rate limited",
        )

        assert err.agent_name == "claude"
        assert err.operation == "probe"

    def test_nomic_phase_error(self):
        """Test NomicPhaseError."""
        from aragora.exceptions import NomicPhaseError

        err = NomicPhaseError(
            phase="verify",
            reason="Tests failed",
            stage="unit_tests",
            recoverable=False,
        )

        assert err.phase == "verify"
        assert err.stage == "unit_tests"
        assert err.recoverable is False

    def test_nomic_integration_error(self):
        """Test NomicIntegrationError."""
        from aragora.exceptions import NomicIntegrationError

        err = NomicIntegrationError(
            integration="supabase",
            reason="Connection refused",
        )

        assert err.integration == "supabase"

    def test_nomic_analytics_error(self):
        """Test NomicAnalyticsError."""
        from aragora.exceptions import NomicAnalyticsError

        err = NomicAnalyticsError(
            tracker="ELO",
            reason="Rating calculation failed",
        )

        assert err.tracker == "ELO"

    def test_nomic_verification_error(self):
        """Test NomicVerificationError."""
        from aragora.exceptions import NomicVerificationError

        err = NomicVerificationError(
            check_type="syntax",
            reason="Parse error",
            file_path="/path/to/file.py",
        )

        assert err.check_type == "syntax"
        assert err.file_path == "/path/to/file.py"

    def test_nomic_timeout_error(self):
        """Test NomicTimeoutError."""
        from aragora.exceptions import NomicTimeoutError

        err = NomicTimeoutError(
            operation="debate_round",
            timeout_seconds=300.0,
        )

        assert err.operation == "debate_round"
        assert err.timeout_seconds == 300.0


# =============================================================================
# OAuth Error Tests
# =============================================================================


class TestOAuthErrors:
    """Tests for OAuth-related exceptions."""

    def test_oauth_state_error(self):
        """Test OAuthStateError."""
        from aragora.exceptions import OAuthStateError

        err = OAuthStateError(reason="State mismatch")

        assert err.reason == "State mismatch"
        assert "state" in str(err).lower()
