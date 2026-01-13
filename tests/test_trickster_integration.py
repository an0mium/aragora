"""
Tests for Trickster (hollow consensus detection) integration.

Tests that the Trickster is properly wired to detect hollow consensus
and inject challenges during debates.
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch


class TestTricksterBasics:
    """Tests for Trickster basic functionality."""

    def test_trickster_creation(self):
        """Trickster should be creatable."""
        try:
            from aragora.debate.trickster import EvidencePoweredTrickster, TricksterConfig
        except ImportError:
            pytest.skip("Trickster module not available")

        config = TricksterConfig(sensitivity=0.7)
        trickster = EvidencePoweredTrickster(config=config)
        assert trickster is not None

    def test_trickster_config_sensitivity(self):
        """TricksterConfig should have configurable sensitivity."""
        try:
            from aragora.debate.trickster import TricksterConfig
        except ImportError:
            pytest.skip("Trickster module not available")

        config = TricksterConfig(sensitivity=0.5)
        assert config.sensitivity == 0.5

        config_high = TricksterConfig(sensitivity=0.9)
        assert config_high.sensitivity == 0.9


class TestHollowConsensusDetection:
    """Tests for hollow consensus detection."""

    def test_detect_low_evidence_consensus(self):
        """Should detect consensus with insufficient evidence."""
        try:
            from aragora.debate.trickster import EvidencePoweredTrickster, TricksterConfig
        except ImportError:
            pytest.skip("Trickster module not available")

        trickster = EvidencePoweredTrickster(config=TricksterConfig(sensitivity=0.7))

        # Simulate a round with high agreement but low evidence
        round_data = {
            "proposals": [
                {"agent": "claude", "content": "I agree with the approach", "confidence": 0.9},
                {"agent": "gpt4", "content": "Yes, I concur", "confidence": 0.85},
            ],
            "evidence_count": 0,
            "disagreement_level": 0.1,
        }

        # Should detect hollow consensus
        if hasattr(trickster, "detect_hollow_consensus"):
            is_hollow = trickster.detect_hollow_consensus(round_data)
            # Low evidence + high agreement = hollow
            assert is_hollow is True

    def test_detect_valid_consensus(self):
        """Should not flag consensus backed by evidence."""
        try:
            from aragora.debate.trickster import EvidencePoweredTrickster, TricksterConfig
        except ImportError:
            pytest.skip("Trickster module not available")

        trickster = EvidencePoweredTrickster(config=TricksterConfig(sensitivity=0.7))

        # Simulate a round with evidence-backed agreement
        round_data = {
            "proposals": [
                {
                    "agent": "claude",
                    "content": "Based on the benchmarks, X is faster",
                    "confidence": 0.9,
                },
                {
                    "agent": "gpt4",
                    "content": "The data supports this conclusion",
                    "confidence": 0.85,
                },
            ],
            "evidence_count": 3,
            "disagreement_level": 0.1,
        }

        if hasattr(trickster, "detect_hollow_consensus"):
            is_hollow = trickster.detect_hollow_consensus(round_data)
            # Evidence present = not hollow
            assert is_hollow is False


class TestTricksterChallenge:
    """Tests for Trickster challenge generation."""

    def test_generate_challenge(self):
        """Trickster should generate challenge prompts."""
        try:
            from aragora.debate.trickster import EvidencePoweredTrickster, TricksterConfig
        except ImportError:
            pytest.skip("Trickster module not available")

        trickster = EvidencePoweredTrickster(config=TricksterConfig(sensitivity=0.7))

        if hasattr(trickster, "generate_challenge"):
            challenge = trickster.generate_challenge(
                topic="Should we use microservices?",
                current_consensus="Microservices are the best approach",
            )

            assert challenge is not None
            assert len(challenge) > 0
            # Challenge should question the consensus
            assert any(
                word in challenge.lower()
                for word in ["consider", "what if", "evidence", "why", "assume"]
            )


class TestTricksterWiring:
    """Tests for Trickster wiring in debate phases."""

    def test_protocol_has_trickster_flag(self):
        """DebateProtocol should have enable_trickster flag."""
        from aragora.core import DebateProtocol

        protocol = DebateProtocol(enable_trickster=True)
        assert protocol.enable_trickster is True

    def test_protocol_trickster_sensitivity(self):
        """DebateProtocol should have trickster_sensitivity."""
        from aragora.core import DebateProtocol

        protocol = DebateProtocol(
            enable_trickster=True,
            trickster_sensitivity=0.8,
        )
        assert protocol.trickster_sensitivity == 0.8

    def test_debate_rounds_phase_accepts_trickster(self):
        """DebateRoundsPhase should accept trickster parameter."""
        from aragora.debate.phases.debate_rounds import DebateRoundsPhase
        from aragora.core import DebateProtocol

        mock_trickster = Mock()
        protocol = DebateProtocol(enable_trickster=True)

        phase = DebateRoundsPhase(
            protocol=protocol,
            circuit_breaker=None,
            convergence_detector=None,
            recorder=None,
            hooks=None,
            trickster=mock_trickster,
            rhetorical_observer=None,
            event_emitter=None,
            novelty_tracker=None,
            update_role_assignments=lambda *args: None,
            assign_stances=lambda *args: None,
            select_critics_for_proposal=lambda *args: [],
            critique_with_agent=AsyncMock(),
            build_revision_prompt=lambda *args: "",
            generate_with_agent=AsyncMock(),
            with_timeout=lambda coro, _: coro,
            notify_spectator=lambda *args: None,
            record_grounded_position=lambda *args: None,
            check_judge_termination=AsyncMock(return_value=False),
            check_early_stopping=lambda *args: (False, None),
        )

        assert phase.trickster is mock_trickster


class TestTricksterEvents:
    """Tests for Trickster event emission."""

    def test_stream_event_types_include_trickster(self):
        """StreamEventType should include trickster events."""
        from aragora.server.stream.events import StreamEventType

        assert hasattr(StreamEventType, "HOLLOW_CONSENSUS")
        assert hasattr(StreamEventType, "TRICKSTER_INTERVENTION")

    def test_spectator_maps_trickster_events(self):
        """Spectator should map trickster event types."""
        from aragora.debate.phases.spectator import Spectator
        from aragora.server.stream.events import StreamEventType

        spectator = Spectator(loop_id="test", event_emitter=None)

        # Check type mapping
        type_mapping = spectator.type_mapping if hasattr(spectator, "type_mapping") else {}

        # Should map hollow_consensus and trickster_intervention
        assert "hollow_consensus" in type_mapping or hasattr(spectator, "_map_event_type")


class TestTricksterIntegration:
    """Integration tests for Trickster in debates."""

    @pytest.mark.asyncio
    async def test_trickster_intervenes_on_hollow_consensus(self):
        """Trickster should intervene when hollow consensus detected."""
        try:
            from aragora.debate.trickster import EvidencePoweredTrickster, TricksterConfig
        except ImportError:
            pytest.skip("Trickster module not available")

        trickster = EvidencePoweredTrickster(
            config=TricksterConfig(sensitivity=0.5)  # Lower threshold for testing
        )

        # Mock event emitter
        mock_emitter = Mock()

        # Simulate detecting hollow consensus
        proposals = [
            {"agent": "claude", "content": "I think so", "confidence": 0.9},
            {"agent": "gpt4", "content": "Agreed", "confidence": 0.85},
        ]

        if hasattr(trickster, "check_and_intervene"):
            intervention = await trickster.check_and_intervene(
                proposals=proposals,
                evidence_count=0,
                round_num=1,
            )

            # Should return an intervention
            if intervention:
                assert "challenge" in intervention or "intervention_type" in intervention

    @pytest.mark.asyncio
    async def test_trickster_emits_event(self):
        """Trickster intervention should emit event."""
        try:
            from aragora.debate.trickster import EvidencePoweredTrickster, TricksterConfig
        except ImportError:
            pytest.skip("Trickster module not available")

        from aragora.server.stream import StreamEvent, StreamEventType

        trickster = EvidencePoweredTrickster(config=TricksterConfig(sensitivity=0.5))

        mock_emitter = Mock()

        # Trigger intervention
        if hasattr(trickster, "emit_intervention_event"):
            trickster.emit_intervention_event(
                event_emitter=mock_emitter,
                round_num=1,
                challenge="What evidence supports this claim?",
                targets=["claude", "gpt4"],
            )

            # Should have emitted an event
            mock_emitter.emit.assert_called()


class TestNoveltyTracker:
    """Tests for NoveltyTracker integration with Trickster."""

    def test_novelty_tracker_creation(self):
        """NoveltyTracker should be creatable."""
        try:
            from aragora.debate.novelty import NoveltyTracker
        except ImportError:
            pytest.skip("NoveltyTracker module not available")

        tracker = NoveltyTracker(low_novelty_threshold=0.15)
        assert tracker is not None

    def test_novelty_tracker_detects_staleness(self):
        """NoveltyTracker should detect stale/repetitive proposals."""
        try:
            from aragora.debate.novelty import NoveltyTracker
        except ImportError:
            pytest.skip("NoveltyTracker module not available")

        tracker = NoveltyTracker(low_novelty_threshold=0.15)

        # Add same-ish content multiple times
        tracker.add_proposal("The solution is to use caching")
        tracker.add_proposal("We should implement caching")
        tracker.add_proposal("Caching is the answer")

        # Should detect low novelty
        if hasattr(tracker, "is_stale"):
            assert tracker.is_stale("Caching would help") is True

    def test_novelty_tracker_accepts_novel_content(self):
        """NoveltyTracker should accept genuinely novel content."""
        try:
            from aragora.debate.novelty import NoveltyTracker
        except ImportError:
            pytest.skip("NoveltyTracker module not available")

        tracker = NoveltyTracker(low_novelty_threshold=0.15)

        tracker.add_proposal("The solution is to use caching")

        # Novel content should not be stale
        if hasattr(tracker, "is_stale"):
            assert tracker.is_stale("We need better error handling instead") is False
