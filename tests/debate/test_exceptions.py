"""Tests for aragora.debate.exceptions."""

from __future__ import annotations

import pytest

from aragora.debate.exceptions import (
    AgentCircuitOpenError,
    AgentResponseError,
    AragoraError,
    CheckpointError,
    CheckpointNotFoundError,
    ConsensusError,
    ConsensusTimeoutError,
    CritiqueGenerationError,
    DebateBatchError,
    DebateConfigurationError,
    DebateError,
    DebateExecutionError,
    DebateNotFoundError,
    DebateStartError,
    DebateTimeoutError,
    EarlyStopError,
    PhaseExecutionError,
    RevisionGenerationError,
    RoundLimitExceededError,
    SynthesisGenerationError,
    VoteProcessingError,
    VoteValidationError,
)


class TestReExports:
    """All re-exported exception classes are importable."""

    @pytest.mark.parametrize(
        "exc_cls",
        [
            AragoraError,
            ConsensusError,
            ConsensusTimeoutError,
            DebateBatchError,
            DebateConfigurationError,
            DebateError,
            DebateExecutionError,
            DebateNotFoundError,
            DebateStartError,
            EarlyStopError,
            PhaseExecutionError,
            RoundLimitExceededError,
            VoteProcessingError,
            VoteValidationError,
        ],
    )
    def test_re_export_is_exception(self, exc_cls):
        assert issubclass(exc_cls, Exception)


class TestDebateTimeoutError:
    """DebateTimeoutError message formatting and attributes."""

    def test_default_message(self):
        exc = DebateTimeoutError()
        assert "timed out" in str(exc).lower()

    def test_with_debate_id(self):
        exc = DebateTimeoutError(debate_id="d-123")
        assert "d-123" in str(exc)
        assert exc.debate_id == "d-123"

    def test_with_timeout_seconds(self):
        exc = DebateTimeoutError(timeout_seconds=30.0)
        assert "30" in str(exc)
        assert exc.timeout_seconds == 30.0

    def test_with_both(self):
        exc = DebateTimeoutError(debate_id="d-1", timeout_seconds=60)
        assert "d-1" in str(exc)
        assert "60" in str(exc)

    def test_is_debate_error(self):
        assert issubclass(DebateTimeoutError, DebateError)


class TestAgentCircuitOpenError:
    """AgentCircuitOpenError message formatting and attributes."""

    def test_default_message(self):
        exc = AgentCircuitOpenError()
        assert "circuit breaker" in str(exc).lower()

    def test_with_agent_name(self):
        exc = AgentCircuitOpenError(agent_name="gpt-4")
        assert "gpt-4" in str(exc)
        assert exc.agent_name == "gpt-4"

    def test_is_debate_error(self):
        assert issubclass(AgentCircuitOpenError, DebateError)


class TestCheckpointErrors:
    """CheckpointError and CheckpointNotFoundError hierarchy."""

    def test_checkpoint_error_is_aragora_error(self):
        assert issubclass(CheckpointError, AragoraError)

    def test_checkpoint_not_found_is_checkpoint_error(self):
        assert issubclass(CheckpointNotFoundError, CheckpointError)

    def test_checkpoint_error_instantiable(self):
        exc = CheckpointError("bad checkpoint")
        assert "bad checkpoint" in str(exc)

    def test_checkpoint_not_found_instantiable(self):
        exc = CheckpointNotFoundError("ckpt-42")
        assert "ckpt-42" in str(exc)


class TestAgentResponseError:
    """AgentResponseError message formatting and attributes."""

    def test_default_message(self):
        exc = AgentResponseError()
        assert "response error" in str(exc).lower()

    def test_with_agent_name(self):
        exc = AgentResponseError(agent_name="claude")
        assert "claude" in str(exc)

    def test_with_phase(self):
        exc = AgentResponseError(phase="proposal")
        assert "proposal" in str(exc)

    def test_with_cause(self):
        cause = ValueError("bad input")
        exc = AgentResponseError(cause=cause)
        assert exc.__cause__ is cause
        assert "bad input" in str(exc)

    def test_with_all_params(self):
        cause = RuntimeError("fail")
        exc = AgentResponseError(agent_name="gpt", phase="critique", cause=cause)
        msg = str(exc)
        assert "gpt" in msg
        assert "critique" in msg
        assert "fail" in msg

    def test_is_debate_execution_error(self):
        assert issubclass(AgentResponseError, DebateExecutionError)


class TestCritiqueGenerationError:
    """CritiqueGenerationError sets phase to 'critique'."""

    def test_phase_is_critique(self):
        exc = CritiqueGenerationError(agent_name="a1")
        assert "critique" in str(exc)

    def test_is_agent_response_error(self):
        assert issubclass(CritiqueGenerationError, AgentResponseError)

    def test_with_cause(self):
        cause = TypeError("oops")
        exc = CritiqueGenerationError(agent_name="a1", cause=cause)
        assert exc.__cause__ is cause


class TestRevisionGenerationError:
    """RevisionGenerationError sets phase to 'revision'."""

    def test_phase_is_revision(self):
        exc = RevisionGenerationError(agent_name="a2")
        assert "revision" in str(exc)

    def test_is_agent_response_error(self):
        assert issubclass(RevisionGenerationError, AgentResponseError)


class TestSynthesisGenerationError:
    """SynthesisGenerationError sets phase to 'synthesis'."""

    def test_phase_is_synthesis(self):
        exc = SynthesisGenerationError(agent_name="a3")
        assert "synthesis" in str(exc)

    def test_is_agent_response_error(self):
        assert issubclass(SynthesisGenerationError, AgentResponseError)


class TestAllExported:
    """__all__ contains all expected exception names."""

    def test_all_list_complete(self):
        from aragora.debate import exceptions

        expected = {
            "AragoraError",
            "ConsensusError",
            "ConsensusTimeoutError",
            "DebateBatchError",
            "DebateConfigurationError",
            "DebateError",
            "DebateExecutionError",
            "DebateNotFoundError",
            "DebateStartError",
            "EarlyStopError",
            "PhaseExecutionError",
            "RoundLimitExceededError",
            "VoteProcessingError",
            "VoteValidationError",
            "DebateTimeoutError",
            "AgentCircuitOpenError",
            "CheckpointError",
            "CheckpointNotFoundError",
            "AgentResponseError",
            "CritiqueGenerationError",
            "RevisionGenerationError",
            "SynthesisGenerationError",
        }
        assert set(exceptions.__all__) == expected
