"""Tests for formal verification event emission.

Tests cover:
- FORMAL_VERIFICATION_RESULT event emitted after attempt_formal_verification
- Event contains correct status, is_verified, is_high_confidence fields
- Claim is truncated to 200 chars in event data
- Event emission graceful when events module unavailable
- Protocol flag enable_formal_verification is False by default
- enable_formal_verification activates formal verification in consensus phase
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.verification.formal import (
    FormalLanguage,
    FormalProofResult,
    FormalProofStatus,
    FormalVerificationManager,
)


# =============================================================================
# Event emission after attempt_formal_verification
# =============================================================================


class TestFormalVerificationEventEmission:
    """Tests for FORMAL_VERIFICATION_RESULT event emission."""

    @pytest.mark.asyncio
    async def test_event_emitted_after_verification(self):
        """FORMAL_VERIFICATION_RESULT event is emitted after attempt_formal_verification."""
        emitted_events = []

        def capture_event(event):
            emitted_events.append(event)

        manager = FormalVerificationManager(event_callback=capture_event)

        # Mock a backend that can verify the claim
        mock_backend = MagicMock()
        mock_backend.is_available = True
        mock_backend.can_verify.return_value = True
        mock_backend.language = FormalLanguage.Z3_SMT
        mock_backend.translate = AsyncMock(return_value="(assert (> x 0))\n(check-sat)")
        mock_backend.prove = AsyncMock(
            return_value=FormalProofResult(
                status=FormalProofStatus.PROOF_FOUND,
                language=FormalLanguage.Z3_SMT,
                formal_statement="(assert (> x 0))\n(check-sat)",
                proof_text="QED",
                translation_confidence=0.9,
                semantic_match_verified=True,
            )
        )
        manager.backends = [mock_backend]

        result = await manager.attempt_formal_verification(
            claim="x is greater than 0",
            claim_type="LOGICAL",
        )

        assert result.status == FormalProofStatus.PROOF_FOUND
        assert len(emitted_events) == 1

        event = emitted_events[0]
        from aragora.events.types import StreamEventType

        assert event.type == StreamEventType.FORMAL_VERIFICATION_RESULT
        assert event.data["claim"] == "x is greater than 0"
        assert event.data["status"] == "proof_found"
        assert event.data["is_verified"] is True

    @pytest.mark.asyncio
    async def test_event_contains_correct_fields(self):
        """Event data contains status, is_verified, is_high_confidence, backend, formal_language."""
        emitted_events = []

        manager = FormalVerificationManager(event_callback=lambda e: emitted_events.append(e))

        mock_backend = MagicMock()
        mock_backend.is_available = True
        mock_backend.can_verify.return_value = True
        mock_backend.language = FormalLanguage.LEAN4
        mock_backend.translate = AsyncMock(return_value="theorem x : True := trivial")
        mock_backend.prove = AsyncMock(
            return_value=FormalProofResult(
                status=FormalProofStatus.PROOF_FOUND,
                language=FormalLanguage.LEAN4,
                formal_statement="theorem x : True := trivial",
                translation_confidence=0.95,
                semantic_match_verified=True,
            )
        )
        manager.backends = [mock_backend]

        await manager.attempt_formal_verification(claim="True is provable")

        assert len(emitted_events) == 1
        data = emitted_events[0].data
        assert data["status"] == "proof_found"
        assert data["is_verified"] is True
        assert data["is_high_confidence"] is True
        assert data["backend"] == "lean4"
        assert data["formal_language"] == "lean4"

    @pytest.mark.asyncio
    async def test_event_fields_for_failed_proof(self):
        """Event correctly reports failed verification status."""
        emitted_events = []

        manager = FormalVerificationManager(event_callback=lambda e: emitted_events.append(e))

        mock_backend = MagicMock()
        mock_backend.is_available = True
        mock_backend.can_verify.return_value = True
        mock_backend.language = FormalLanguage.Z3_SMT
        mock_backend.translate = AsyncMock(return_value="(assert (= 1 2))\n(check-sat)")
        mock_backend.prove = AsyncMock(
            return_value=FormalProofResult(
                status=FormalProofStatus.PROOF_FAILED,
                language=FormalLanguage.Z3_SMT,
                formal_statement="(assert (= 1 2))\n(check-sat)",
                error_message="Claim is false - counterexample found",
            )
        )
        manager.backends = [mock_backend]

        result = await manager.attempt_formal_verification(claim="1 equals 2")

        assert result.status == FormalProofStatus.PROOF_FAILED
        assert len(emitted_events) == 1

        data = emitted_events[0].data
        assert data["status"] == "proof_failed"
        assert data["is_verified"] is False
        assert data["is_high_confidence"] is False

    @pytest.mark.asyncio
    async def test_claim_truncated_to_200_chars(self):
        """Claims longer than 200 characters are truncated in event data."""
        emitted_events = []

        manager = FormalVerificationManager(event_callback=lambda e: emitted_events.append(e))

        mock_backend = MagicMock()
        mock_backend.is_available = True
        mock_backend.can_verify.return_value = True
        mock_backend.language = FormalLanguage.Z3_SMT
        mock_backend.translate = AsyncMock(return_value="(check-sat)")
        mock_backend.prove = AsyncMock(
            return_value=FormalProofResult(
                status=FormalProofStatus.PROOF_FOUND,
                language=FormalLanguage.Z3_SMT,
            )
        )
        manager.backends = [mock_backend]

        long_claim = "A" * 500
        await manager.attempt_formal_verification(claim=long_claim)

        assert len(emitted_events) == 1
        assert len(emitted_events[0].data["claim"]) == 200
        assert emitted_events[0].data["claim"] == "A" * 200

    @pytest.mark.asyncio
    async def test_no_event_when_no_backend_available(self):
        """No event emitted when no backend can verify the claim."""
        emitted_events = []

        manager = FormalVerificationManager(event_callback=lambda e: emitted_events.append(e))
        # No backends available
        manager.backends = []

        result = await manager.attempt_formal_verification(claim="some claim")

        assert result.status == FormalProofStatus.NOT_SUPPORTED
        assert len(emitted_events) == 0

    @pytest.mark.asyncio
    async def test_no_event_when_translation_fails(self):
        """No event emitted when claim translation fails."""
        emitted_events = []

        manager = FormalVerificationManager(event_callback=lambda e: emitted_events.append(e))

        mock_backend = MagicMock()
        mock_backend.is_available = True
        mock_backend.can_verify.return_value = True
        mock_backend.language = FormalLanguage.Z3_SMT
        mock_backend.translate = AsyncMock(return_value=None)
        manager.backends = [mock_backend]

        result = await manager.attempt_formal_verification(claim="untranslatable claim")

        assert result.status == FormalProofStatus.TRANSLATION_FAILED
        # No event because we return before reaching the prove step
        assert len(emitted_events) == 0

    @pytest.mark.asyncio
    async def test_event_emission_graceful_when_events_module_unavailable(self):
        """Event emission degrades gracefully when aragora.events.types is unavailable."""
        manager = FormalVerificationManager()  # No callback

        mock_backend = MagicMock()
        mock_backend.is_available = True
        mock_backend.can_verify.return_value = True
        mock_backend.language = FormalLanguage.Z3_SMT
        mock_backend.translate = AsyncMock(return_value="(check-sat)")
        mock_backend.prove = AsyncMock(
            return_value=FormalProofResult(
                status=FormalProofStatus.PROOF_FOUND,
                language=FormalLanguage.Z3_SMT,
            )
        )
        manager.backends = [mock_backend]

        # Patch the events import to simulate unavailability
        with patch.dict("sys.modules", {"aragora.events.types": None}):
            # Should not raise
            result = await manager.attempt_formal_verification(claim="x > 0")

        assert result.status == FormalProofStatus.PROOF_FOUND

    @pytest.mark.asyncio
    async def test_event_emission_graceful_when_callback_raises(self):
        """Event emission degrades gracefully when callback raises RuntimeError."""

        def bad_callback(event):
            raise RuntimeError("callback broken")

        manager = FormalVerificationManager(event_callback=bad_callback)

        mock_backend = MagicMock()
        mock_backend.is_available = True
        mock_backend.can_verify.return_value = True
        mock_backend.language = FormalLanguage.Z3_SMT
        mock_backend.translate = AsyncMock(return_value="(check-sat)")
        mock_backend.prove = AsyncMock(
            return_value=FormalProofResult(
                status=FormalProofStatus.PROOF_FOUND,
                language=FormalLanguage.Z3_SMT,
            )
        )
        manager.backends = [mock_backend]

        # Should not raise despite callback error
        result = await manager.attempt_formal_verification(claim="x > 0")
        assert result.status == FormalProofStatus.PROOF_FOUND

    @pytest.mark.asyncio
    async def test_no_callback_no_error(self):
        """No error when event_callback is None (default)."""
        manager = FormalVerificationManager()  # event_callback=None

        mock_backend = MagicMock()
        mock_backend.is_available = True
        mock_backend.can_verify.return_value = True
        mock_backend.language = FormalLanguage.Z3_SMT
        mock_backend.translate = AsyncMock(return_value="(check-sat)")
        mock_backend.prove = AsyncMock(
            return_value=FormalProofResult(
                status=FormalProofStatus.PROOF_FOUND,
                language=FormalLanguage.Z3_SMT,
            )
        )
        manager.backends = [mock_backend]

        result = await manager.attempt_formal_verification(claim="x > 0")
        assert result.status == FormalProofStatus.PROOF_FOUND


# =============================================================================
# Protocol flag tests
# =============================================================================


class TestEnableFormalVerificationFlag:
    """Tests for the enable_formal_verification protocol flag."""

    def test_default_is_false(self):
        """enable_formal_verification defaults to False in DebateProtocol."""
        from aragora.debate.protocol import DebateProtocol

        protocol = DebateProtocol()
        assert protocol.enable_formal_verification is False

    def test_can_be_set_to_true(self):
        """enable_formal_verification can be explicitly enabled."""
        from aragora.debate.protocol import DebateProtocol

        protocol = DebateProtocol(enable_formal_verification=True)
        assert protocol.enable_formal_verification is True

    def test_independent_of_formal_verification_enabled(self):
        """enable_formal_verification is independent of formal_verification_enabled."""
        from aragora.debate.protocol import DebateProtocol

        protocol = DebateProtocol(
            enable_formal_verification=True,
            formal_verification_enabled=False,
        )
        assert protocol.enable_formal_verification is True
        assert protocol.formal_verification_enabled is False

    def test_both_flags_can_be_true(self):
        """Both flags can be set simultaneously."""
        from aragora.debate.protocol import DebateProtocol

        protocol = DebateProtocol(
            enable_formal_verification=True,
            formal_verification_enabled=True,
        )
        assert protocol.enable_formal_verification is True
        assert protocol.formal_verification_enabled is True


# =============================================================================
# Consensus phase wiring tests
# =============================================================================


class TestConsensusPhaseFormalVerificationWiring:
    """Tests that enable_formal_verification activates verification in consensus phase."""

    @pytest.mark.asyncio
    async def test_enable_formal_verification_activates_verify(self):
        """Setting enable_formal_verification=True activates _verify_consensus_formally."""
        from aragora.debate.phases.consensus_phase import ConsensusPhase

        # Build minimal dependencies
        protocol = MagicMock()
        protocol.formal_verification_enabled = False
        protocol.enable_formal_verification = True
        protocol.formal_verification_timeout = 5.0
        protocol.formal_verification_languages = ["z3_smt"]
        protocol.enable_hilbert_proofing = False

        # Create a minimal ctx
        ctx = MagicMock()
        ctx.result = MagicMock()
        ctx.result.final_answer = "The sum of 2 and 2 is 4"
        ctx.result.formal_verification = None
        ctx.event_emitter = None
        ctx.loop_id = "test-loop"
        ctx.debate_id = "test-debate"
        ctx.env = MagicMock()
        ctx.env.task = "test task"

        phase = ConsensusPhase()
        phase.protocol = protocol

        # The method uses lazy import inside the try block. Patch the source module.
        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {"any_available": False}

        with patch(
            "aragora.verification.formal.get_formal_verification_manager",
            return_value=mock_manager,
        ):
            await phase._verify_consensus_formally(ctx)

        # Verify it reached the verification logic (didn't return early)
        # Since no backends available, it should set status to "skipped"
        assert ctx.result.formal_verification is not None
        assert ctx.result.formal_verification.get("status") == "skipped"

    @pytest.mark.asyncio
    async def test_neither_flag_enabled_skips_verification(self):
        """When both flags are False, verification is skipped entirely."""
        from aragora.debate.phases.consensus_phase import ConsensusPhase

        protocol = MagicMock()
        protocol.formal_verification_enabled = False
        protocol.enable_formal_verification = False

        ctx = MagicMock()
        ctx.result = MagicMock()
        ctx.result.final_answer = "Something"
        ctx.result.formal_verification = None

        phase = ConsensusPhase()
        phase.protocol = protocol

        await phase._verify_consensus_formally(ctx)

        # Should remain None -- verification was skipped
        assert ctx.result.formal_verification is None

    @pytest.mark.asyncio
    async def test_formal_verification_enabled_still_works(self):
        """The existing formal_verification_enabled flag still activates verification."""
        from aragora.debate.phases.consensus_phase import ConsensusPhase

        protocol = MagicMock()
        protocol.formal_verification_enabled = True
        protocol.enable_formal_verification = False
        protocol.formal_verification_timeout = 5.0
        protocol.formal_verification_languages = ["z3_smt"]
        protocol.enable_hilbert_proofing = False

        ctx = MagicMock()
        ctx.result = MagicMock()
        ctx.result.final_answer = "2 + 2 = 4"
        ctx.result.formal_verification = None
        ctx.event_emitter = None
        ctx.loop_id = "test"
        ctx.debate_id = "test"
        ctx.env = MagicMock()
        ctx.env.task = "test"

        phase = ConsensusPhase()
        phase.protocol = protocol

        # Mock manager with no available backends
        mock_manager = MagicMock()
        mock_manager.status_report.return_value = {"any_available": False}

        with patch(
            "aragora.verification.formal.get_formal_verification_manager",
            return_value=mock_manager,
        ):
            await phase._verify_consensus_formally(ctx)

        # Should have attempted verification (set some result, not None)
        # Since no backends available, it should be set to "skipped"
        assert ctx.result.formal_verification is not None
        assert ctx.result.formal_verification.get("status") == "skipped"
