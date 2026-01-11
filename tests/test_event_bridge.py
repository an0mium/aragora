"""
Tests for aragora.debate.event_bridge module.

Covers:
- EventEmitterBridge initialization
- Event type mapping
- Spectator emission
- WebSocket emission
- Cartographer updates
- Moment event emission
- Error handling
"""

from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.event_bridge import EventEmitterBridge


class TestEventEmitterBridgeInit:
    """Tests for EventEmitterBridge initialization."""

    def test_init_with_no_components(self):
        """Bridge should initialize with no components."""
        bridge = EventEmitterBridge()

        assert bridge.spectator is None
        assert bridge.event_emitter is None
        assert bridge.cartographer is None
        assert bridge.loop_id == ""

    def test_init_with_all_components(self):
        """Bridge should initialize with all components."""
        spectator = MagicMock()
        emitter = MagicMock()
        cartographer = MagicMock()

        bridge = EventEmitterBridge(
            spectator=spectator,
            event_emitter=emitter,
            cartographer=cartographer,
            loop_id="test-loop-123",
        )

        assert bridge.spectator is spectator
        assert bridge.event_emitter is emitter
        assert bridge.cartographer is cartographer
        assert bridge.loop_id == "test-loop-123"


class TestEventTypeMapping:
    """Tests for event type mapping constants."""

    def test_debate_lifecycle_events_mapped(self):
        """Debate lifecycle events should be mapped."""
        mapping = EventEmitterBridge.EVENT_TYPE_MAPPING

        assert mapping["debate_start"] == "DEBATE_START"
        assert mapping["debate_end"] == "DEBATE_END"
        assert mapping["round_start"] == "ROUND_START"
        assert mapping["round"] == "ROUND_START"

    def test_agent_action_events_mapped(self):
        """Agent action events should be mapped."""
        mapping = EventEmitterBridge.EVENT_TYPE_MAPPING

        assert mapping["propose"] == "AGENT_MESSAGE"
        assert mapping["proposal"] == "AGENT_MESSAGE"
        assert mapping["critique"] == "CRITIQUE"
        assert mapping["vote"] == "VOTE"
        assert mapping["judge"] == "AGENT_MESSAGE"

    def test_consensus_events_mapped(self):
        """Consensus events should be mapped."""
        mapping = EventEmitterBridge.EVENT_TYPE_MAPPING

        assert mapping["consensus"] == "CONSENSUS"
        assert mapping["convergence"] == "CONSENSUS"

    def test_memory_events_mapped(self):
        """Memory events should be mapped."""
        mapping = EventEmitterBridge.EVENT_TYPE_MAPPING

        assert mapping["memory_recall"] == "MEMORY_RECALL"
        assert mapping["memory_tier_promotion"] == "MEMORY_TIER_PROMOTION"
        assert mapping["memory_tier_demotion"] == "MEMORY_TIER_DEMOTION"

    def test_audience_events_mapped(self):
        """Audience events should be mapped."""
        mapping = EventEmitterBridge.EVENT_TYPE_MAPPING

        assert mapping["audience_drain"] == "AUDIENCE_DRAIN"
        assert mapping["audience_summary"] == "AUDIENCE_SUMMARY"

    def test_token_events_mapped(self):
        """Token streaming events should be mapped."""
        mapping = EventEmitterBridge.EVENT_TYPE_MAPPING

        assert mapping["token_start"] == "TOKEN_START"
        assert mapping["token_delta"] == "TOKEN_DELTA"
        assert mapping["token_end"] == "TOKEN_END"


class TestNotifyMethod:
    """Tests for the notify method."""

    def test_notify_with_spectator(self):
        """notify should emit to spectator when available."""
        spectator = MagicMock()
        bridge = EventEmitterBridge(spectator=spectator)

        bridge.notify("proposal", agent="agent1", details="Test proposal")

        spectator.emit.assert_called_once_with(
            "proposal",
            agent="agent1",
            details="Test proposal",
        )

    def test_notify_without_spectator(self):
        """notify should not fail without spectator."""
        bridge = EventEmitterBridge()

        # Should not raise
        bridge.notify("proposal", agent="agent1")

    def test_notify_with_event_emitter(self):
        """notify should emit to WebSocket when emitter available."""
        emitter = MagicMock()
        bridge = EventEmitterBridge(event_emitter=emitter, loop_id="test-loop")

        with patch.object(bridge, "_emit_to_websocket") as mock_emit:
            bridge.notify("proposal", agent="agent1", details="Test")
            mock_emit.assert_called_once_with("proposal", agent="agent1", details="Test")

    def test_notify_with_both_spectator_and_emitter(self):
        """notify should emit to both spectator and WebSocket."""
        spectator = MagicMock()
        emitter = MagicMock()
        bridge = EventEmitterBridge(spectator=spectator, event_emitter=emitter)

        with patch.object(bridge, "_emit_to_websocket") as mock_emit:
            bridge.notify("vote", agent="agent1", details="Accept")

            spectator.emit.assert_called_once()
            mock_emit.assert_called_once()


class TestEmitToWebSocket:
    """Tests for WebSocket emission."""

    def test_emit_mapped_event_type(self):
        """Should emit StreamEvent for mapped event types."""
        emitter = MagicMock()
        bridge = EventEmitterBridge(event_emitter=emitter, loop_id="test-loop")

        with patch("aragora.server.stream.StreamEvent") as MockStreamEvent:
            with patch("aragora.server.stream.StreamEventType") as MockStreamEventType:
                MockStreamEventType.AGENT_MESSAGE = "AGENT_MESSAGE"

                bridge._emit_to_websocket(
                    "proposal",
                    agent="agent1",
                    details="My proposal",
                    round_number=2,
                )

                MockStreamEvent.assert_called_once()
                emitter.emit.assert_called_once()

    def test_skip_unmapped_event_type(self):
        """Should skip events not in mapping."""
        emitter = MagicMock()
        bridge = EventEmitterBridge(event_emitter=emitter)

        bridge._emit_to_websocket("unknown_event_type", agent="agent1")

        # Should not call emit for unmapped types
        emitter.emit.assert_not_called()

    def test_emit_handles_exception(self):
        """Should handle exceptions gracefully."""
        emitter = MagicMock()
        emitter.emit.side_effect = Exception("Emit failed")
        bridge = EventEmitterBridge(event_emitter=emitter, loop_id="test")

        # Should not raise
        bridge._emit_to_websocket("proposal", agent="agent1")

    def test_emit_updates_cartographer(self):
        """Should update cartographer after WebSocket emit."""
        emitter = MagicMock()
        cartographer = MagicMock()
        bridge = EventEmitterBridge(
            event_emitter=emitter,
            cartographer=cartographer,
            loop_id="test",
        )

        with patch.object(bridge, "_update_cartographer") as mock_update:
            bridge._emit_to_websocket("proposal", agent="agent1", details="Test")
            mock_update.assert_called_once()


class TestUpdateCartographer:
    """Tests for cartographer updates."""

    @pytest.fixture
    def bridge_with_cartographer(self):
        """Create bridge with mock cartographer."""
        cartographer = MagicMock()
        bridge = EventEmitterBridge(cartographer=cartographer)
        return bridge, cartographer

    def test_update_without_cartographer(self):
        """Should not fail without cartographer."""
        bridge = EventEmitterBridge()

        # Should not raise
        bridge._update_cartographer("proposal", agent="agent1")

    def test_update_from_proposal(self, bridge_with_cartographer):
        """Should update cartographer for proposal events."""
        bridge, cartographer = bridge_with_cartographer

        bridge._update_cartographer(
            "proposal",
            agent="agent1",
            details="My proposal content",
            round_number=1,
        )

        cartographer.update_from_message.assert_called_once_with(
            agent="agent1",
            content="My proposal content",
            role="proposer",
            round_num=1,
        )

    def test_update_from_propose(self, bridge_with_cartographer):
        """Should handle 'propose' alias."""
        bridge, cartographer = bridge_with_cartographer

        bridge._update_cartographer("propose", agent="agent1", details="Content")

        cartographer.update_from_message.assert_called_once()

    def test_update_from_critique(self, bridge_with_cartographer):
        """Should update cartographer for critique events."""
        bridge, cartographer = bridge_with_cartographer

        bridge._update_cartographer(
            "critique",
            agent="critic1",
            details="Critiqued agent2: This is wrong",
            metric=0.8,
            round_number=2,
        )

        cartographer.update_from_critique.assert_called_once_with(
            critic_agent="critic1",
            target_agent="agent2",
            severity=0.8,
            round_num=2,
            critique_text="Critiqued agent2: This is wrong",
        )

    def test_update_from_critique_default_severity(self, bridge_with_cartographer):
        """Should use default severity when not provided."""
        bridge, cartographer = bridge_with_cartographer

        bridge._update_cartographer("critique", agent="critic1", details="Test")

        call_args = cartographer.update_from_critique.call_args
        assert call_args.kwargs["severity"] == 0.5

    def test_update_from_vote(self, bridge_with_cartographer):
        """Should update cartographer for vote events."""
        bridge, cartographer = bridge_with_cartographer

        bridge._update_cartographer(
            "vote",
            agent="voter1",
            details="Voted: Accept",
            round_number=3,
        )

        cartographer.update_from_vote.assert_called_once_with(
            agent="voter1",
            vote_value="Accept",
            round_num=3,
        )

    def test_update_from_consensus(self, bridge_with_cartographer):
        """Should update cartographer for consensus events."""
        bridge, cartographer = bridge_with_cartographer

        bridge._update_cartographer(
            "consensus",
            details="Consensus reached: Proposal A",
            round_number=4,
        )

        cartographer.update_from_consensus.assert_called_once_with(
            result="Proposal A",
            round_num=4,
        )

    def test_update_handles_exception(self, bridge_with_cartographer):
        """Should handle cartographer exceptions gracefully."""
        bridge, cartographer = bridge_with_cartographer
        cartographer.update_from_message.side_effect = Exception("Update failed")

        # Should not raise
        bridge._update_cartographer("proposal", agent="agent1", details="Test")

    def test_update_ignores_unknown_event_types(self, bridge_with_cartographer):
        """Should ignore event types without cartographer logic."""
        bridge, cartographer = bridge_with_cartographer

        bridge._update_cartographer("unknown_type", agent="agent1")

        # No cartographer methods should be called
        cartographer.update_from_message.assert_not_called()
        cartographer.update_from_critique.assert_not_called()
        cartographer.update_from_vote.assert_not_called()
        cartographer.update_from_consensus.assert_not_called()


class TestExtractCritiqueTarget:
    """Tests for critique target extraction."""

    def test_extract_target_from_standard_format(self):
        """Should extract target from 'Critiqued X: ...' format."""
        details = "Critiqued agent2: This approach is flawed"

        target = EventEmitterBridge._extract_critique_target(details)

        assert target == "agent2"

    def test_extract_target_empty_when_not_found(self):
        """Should return empty string when target not found."""
        details = "General critique of the approach"

        target = EventEmitterBridge._extract_critique_target(details)

        assert target == ""

    def test_extract_target_with_complex_agent_name(self):
        """Should handle complex agent names."""
        details = "Critiqued anthropic-api-claude: Need more detail"

        target = EventEmitterBridge._extract_critique_target(details)

        assert target == "anthropic-api-claude"


class TestEmitMoment:
    """Tests for moment event emission."""

    def test_emit_moment_without_emitter(self):
        """Should not fail without event emitter."""
        bridge = EventEmitterBridge()
        moment = MagicMock()

        # Should not raise
        bridge.emit_moment(moment)

    def test_emit_moment_with_emitter(self):
        """Should emit moment event to WebSocket."""
        emitter = MagicMock()
        bridge = EventEmitterBridge(event_emitter=emitter, loop_id="test-loop")

        moment = MagicMock()
        moment.to_dict.return_value = {"type": "insight", "content": "Test"}
        moment.moment_type = "insight"
        moment.agent_name = "agent1"

        with patch("aragora.server.stream.StreamEvent") as MockStreamEvent:
            with patch("aragora.server.stream.StreamEventType") as MockStreamEventType:
                MockStreamEventType.MOMENT_DETECTED = "MOMENT_DETECTED"

                bridge.emit_moment(moment)

                MockStreamEvent.assert_called_once()
                emitter.emit.assert_called_once()

    def test_emit_moment_handles_exception(self):
        """Should handle emission exceptions gracefully."""
        emitter = MagicMock()
        emitter.emit.side_effect = Exception("Emit failed")
        bridge = EventEmitterBridge(event_emitter=emitter, loop_id="test")

        moment = MagicMock()
        moment.to_dict.return_value = {}

        # Should not raise
        bridge.emit_moment(moment)


class TestIntegration:
    """Integration tests for EventEmitterBridge."""

    def test_full_debate_flow(self):
        """Test a typical debate event flow."""
        spectator = MagicMock()
        emitter = MagicMock()
        cartographer = MagicMock()

        bridge = EventEmitterBridge(
            spectator=spectator,
            event_emitter=emitter,
            cartographer=cartographer,
            loop_id="debate-123",
        )

        # Simulate debate flow
        bridge.notify("debate_start", details="Starting debate")
        bridge.notify("round_start", round_number=1)
        bridge.notify("proposal", agent="agent1", details="My proposal", round_number=1)
        bridge.notify("critique", agent="agent2", details="Critiqued agent1: Issue", round_number=1)
        bridge.notify("vote", agent="agent1", details="Accept", round_number=1)
        bridge.notify("consensus", details="Reached: Proposal accepted", round_number=1)
        bridge.notify("debate_end", details="Debate complete")

        # Verify spectator received all events
        assert spectator.emit.call_count == 7

        # Verify cartographer received relevant updates
        cartographer.update_from_message.assert_called()
        cartographer.update_from_critique.assert_called()
        cartographer.update_from_vote.assert_called()
        cartographer.update_from_consensus.assert_called()

    def test_resilient_to_emitter_and_cartographer_failures(self):
        """Bridge should continue even if emitter/cartographer fail."""
        spectator = MagicMock()
        emitter = MagicMock()
        emitter.emit.side_effect = Exception("Emitter error")
        cartographer = MagicMock()
        cartographer.update_from_message.side_effect = Exception("Cartographer error")

        bridge = EventEmitterBridge(
            spectator=spectator,
            event_emitter=emitter,
            cartographer=cartographer,
            loop_id="test",
        )

        # Should not raise despite emitter/cartographer failures
        # (spectator errors propagate as they're not caught in notify())
        bridge.notify("proposal", agent="agent1", details="Test")

        # Verify spectator was still called
        spectator.emit.assert_called_once()
