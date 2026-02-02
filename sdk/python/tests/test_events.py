"""
Tests for Aragora SDK events module.

Tests cover:
- Event class instantiation with defaults
- Event class field assignment
- Handling of unexpected fields
- EVENT_CLASS_MAP completeness and correctness
- Specific event scenarios
"""

import pytest

from aragora_sdk.events import (
    EVENT_CLASS_MAP,
    AgentMessageEvent,
    AudienceSuggestionEvent,
    ConnectedEvent,
    ConsensusEvent,
    ConsensusReachedEvent,
    CritiqueEvent,
    DebateEndEvent,
    DebateStartEvent,
    DisconnectedEvent,
    ErrorEvent,
    MessageEvent,
    PhaseChangeEvent,
    ProposeEvent,
    RevisionEvent,
    RoundStartEvent,
    SynthesisEvent,
    UserVoteEvent,
    VoteEvent,
    WarningEvent,
)


class TestEventClassInstantiationWithDefaults:
    """Test that each event class can be instantiated with no arguments."""

    def test_connected_event_defaults(self):
        event = ConnectedEvent()
        assert event.server_version == ""

    def test_disconnected_event_defaults(self):
        event = DisconnectedEvent()
        assert event.code == 1000
        assert event.reason == ""

    def test_error_event_defaults(self):
        event = ErrorEvent()
        assert event.error == ""
        assert event.code == ""

    def test_debate_start_event_defaults(self):
        event = DebateStartEvent()
        assert event.debate_id == ""
        assert event.task == ""
        assert event.agents == []
        assert event.total_rounds == 0
        assert event.protocol == ""

    def test_round_start_event_defaults(self):
        event = RoundStartEvent()
        assert event.debate_id == ""
        assert event.round_number == 0
        assert event.total_rounds == 0

    def test_agent_message_event_defaults(self):
        event = AgentMessageEvent()
        assert event.debate_id == ""
        assert event.round_number == 0
        assert event.agent == ""
        assert event.content == ""
        assert event.confidence is None
        assert event.role == ""

    def test_propose_event_defaults(self):
        event = ProposeEvent()
        assert event.debate_id == ""
        assert event.round_number == 0
        assert event.agent == ""
        assert event.content == ""
        assert event.confidence is None

    def test_critique_event_defaults(self):
        event = CritiqueEvent()
        assert event.debate_id == ""
        assert event.round_number == 0
        assert event.agent == ""
        assert event.target_agent == ""
        assert event.content == ""
        assert event.score is None

    def test_revision_event_defaults(self):
        event = RevisionEvent()
        assert event.debate_id == ""
        assert event.round_number == 0
        assert event.agent == ""
        assert event.content == ""
        assert event.original_content == ""
        assert event.confidence is None

    def test_synthesis_event_defaults(self):
        event = SynthesisEvent()
        assert event.debate_id == ""
        assert event.round_number == 0
        assert event.agent == ""
        assert event.content == ""
        assert event.sources == []

    def test_vote_event_defaults(self):
        event = VoteEvent()
        assert event.debate_id == ""
        assert event.round_number == 0
        assert event.agent == ""
        assert event.choice == ""
        assert event.reasoning == ""
        assert event.confidence is None

    def test_consensus_event_defaults(self):
        event = ConsensusEvent()
        assert event.debate_id == ""
        assert event.round_number == 0
        assert event.result == ""
        assert event.confidence is None
        assert event.method == ""

    def test_consensus_reached_event_defaults(self):
        event = ConsensusReachedEvent()
        assert event.debate_id == ""
        assert event.result == ""
        assert event.confidence is None
        assert event.final_round == 0

    def test_debate_end_event_defaults(self):
        event = DebateEndEvent()
        assert event.debate_id == ""
        assert event.result == ""
        assert event.total_rounds == 0
        assert event.consensus_reached is False
        assert event.duration_seconds is None

    def test_phase_change_event_defaults(self):
        event = PhaseChangeEvent()
        assert event.debate_id == ""
        assert event.from_phase == ""
        assert event.to_phase == ""
        assert event.round_number == 0

    def test_audience_suggestion_event_defaults(self):
        event = AudienceSuggestionEvent()
        assert event.debate_id == ""
        assert event.user_id == ""
        assert event.content == ""
        assert event.round_number == 0

    def test_user_vote_event_defaults(self):
        event = UserVoteEvent()
        assert event.debate_id == ""
        assert event.user_id == ""
        assert event.choice == ""
        assert event.round_number == 0

    def test_warning_event_defaults(self):
        event = WarningEvent()
        assert event.debate_id == ""
        assert event.message == ""
        assert event.severity == "warning"

    def test_message_event_defaults(self):
        event = MessageEvent()
        assert event.content == ""
        assert event.sender == ""
        assert event.metadata == {}


class TestEventClassFieldAssignment:
    """Test that each event class accepts all expected fields."""

    def test_connected_event_with_fields(self):
        event = ConnectedEvent(server_version="1.2.3")
        assert event.server_version == "1.2.3"

    def test_disconnected_event_with_fields(self):
        event = DisconnectedEvent(code=1001, reason="Going away")
        assert event.code == 1001
        assert event.reason == "Going away"

    def test_error_event_with_fields(self):
        event = ErrorEvent(error="Connection failed", code="ERR_CONN")
        assert event.error == "Connection failed"
        assert event.code == "ERR_CONN"

    def test_debate_start_event_with_full_data(self):
        event = DebateStartEvent(
            debate_id="debate-123",
            task="Discuss AI ethics",
            agents=["claude", "gpt", "gemini"],
            total_rounds=5,
            protocol="majority",
        )
        assert event.debate_id == "debate-123"
        assert event.task == "Discuss AI ethics"
        assert event.agents == ["claude", "gpt", "gemini"]
        assert event.total_rounds == 5
        assert event.protocol == "majority"

    def test_agent_message_event_with_confidence(self):
        event = AgentMessageEvent(
            debate_id="debate-456",
            round_number=2,
            agent="claude",
            content="My analysis suggests...",
            confidence=0.85,
            role="proposer",
        )
        assert event.debate_id == "debate-456"
        assert event.round_number == 2
        assert event.agent == "claude"
        assert event.content == "My analysis suggests..."
        assert event.confidence == 0.85
        assert event.role == "proposer"

    def test_consensus_event_with_method(self):
        event = ConsensusEvent(
            debate_id="debate-789",
            round_number=3,
            result="Agreement on option A",
            confidence=0.92,
            method="supermajority",
        )
        assert event.debate_id == "debate-789"
        assert event.round_number == 3
        assert event.result == "Agreement on option A"
        assert event.confidence == 0.92
        assert event.method == "supermajority"

    def test_debate_end_event_with_duration(self):
        event = DebateEndEvent(
            debate_id="debate-final",
            result="Consensus reached on policy X",
            total_rounds=4,
            consensus_reached=True,
            duration_seconds=125.5,
        )
        assert event.debate_id == "debate-final"
        assert event.result == "Consensus reached on policy X"
        assert event.total_rounds == 4
        assert event.consensus_reached is True
        assert event.duration_seconds == 125.5


class TestUnexpectedFieldsHandling:
    """Test that events handle unexpected fields gracefully."""

    def test_connected_event_ignores_unexpected_fields(self):
        # Dataclasses will raise TypeError for unexpected kwargs
        with pytest.raises(TypeError):
            ConnectedEvent(server_version="1.0", unexpected_field="value")

    def test_debate_start_event_ignores_unexpected_fields(self):
        with pytest.raises(TypeError):
            DebateStartEvent(debate_id="123", extra_field="ignored")

    def test_message_event_ignores_unexpected_fields(self):
        with pytest.raises(TypeError):
            MessageEvent(content="hello", unknown="value")


class TestEventClassMap:
    """Test EVENT_CLASS_MAP completeness and correctness."""

    def test_event_class_map_has_19_event_types(self):
        assert len(EVENT_CLASS_MAP) == 19

    def test_event_class_map_contains_all_expected_types(self):
        expected_types = [
            "connected",
            "disconnected",
            "error",
            "debate_start",
            "round_start",
            "agent_message",
            "propose",
            "critique",
            "revision",
            "synthesis",
            "vote",
            "consensus",
            "consensus_reached",
            "debate_end",
            "phase_change",
            "audience_suggestion",
            "user_vote",
            "warning",
            "message",
        ]
        for event_type in expected_types:
            assert event_type in EVENT_CLASS_MAP, f"Missing event type: {event_type}"

    def test_event_class_map_values_are_correct_types(self):
        expected_mappings = {
            "connected": ConnectedEvent,
            "disconnected": DisconnectedEvent,
            "error": ErrorEvent,
            "debate_start": DebateStartEvent,
            "round_start": RoundStartEvent,
            "agent_message": AgentMessageEvent,
            "propose": ProposeEvent,
            "critique": CritiqueEvent,
            "revision": RevisionEvent,
            "synthesis": SynthesisEvent,
            "vote": VoteEvent,
            "consensus": ConsensusEvent,
            "consensus_reached": ConsensusReachedEvent,
            "debate_end": DebateEndEvent,
            "phase_change": PhaseChangeEvent,
            "audience_suggestion": AudienceSuggestionEvent,
            "user_vote": UserVoteEvent,
            "warning": WarningEvent,
            "message": MessageEvent,
        }
        for event_type, event_class in expected_mappings.items():
            assert EVENT_CLASS_MAP[event_type] is event_class

    def test_event_class_map_values_are_instantiable(self):
        for _event_type, event_class in EVENT_CLASS_MAP.items():
            # Each class should be instantiable with no arguments
            instance = event_class()
            assert instance is not None


class TestSpecificEventScenarios:
    """Test specific event scenarios with realistic data."""

    def test_critique_event_with_score(self):
        event = CritiqueEvent(
            debate_id="debate-crit",
            round_number=2,
            agent="gpt",
            target_agent="claude",
            content="The argument lacks supporting evidence",
            score=0.65,
        )
        assert event.agent == "gpt"
        assert event.target_agent == "claude"
        assert event.score == 0.65

    def test_revision_event_with_original_content(self):
        event = RevisionEvent(
            debate_id="debate-rev",
            round_number=3,
            agent="claude",
            content="Updated position with evidence",
            original_content="Initial position",
            confidence=0.78,
        )
        assert event.original_content == "Initial position"
        assert event.content == "Updated position with evidence"

    def test_synthesis_event_with_sources(self):
        event = SynthesisEvent(
            debate_id="debate-syn",
            round_number=4,
            agent="gemini",
            content="Combined viewpoints",
            sources=["claude", "gpt", "mistral"],
        )
        assert event.sources == ["claude", "gpt", "mistral"]
        assert len(event.sources) == 3

    def test_phase_change_event_transitions(self):
        event = PhaseChangeEvent(
            debate_id="debate-phase",
            from_phase="proposal",
            to_phase="critique",
            round_number=1,
        )
        assert event.from_phase == "proposal"
        assert event.to_phase == "critique"

    def test_message_event_with_metadata(self):
        event = MessageEvent(
            content="System notification",
            sender="system",
            metadata={"priority": "high", "timestamp": 1234567890},
        )
        assert event.metadata["priority"] == "high"
        assert event.metadata["timestamp"] == 1234567890

    def test_warning_event_with_custom_severity(self):
        event = WarningEvent(
            debate_id="debate-warn",
            message="Rate limit approaching",
            severity="info",
        )
        assert event.severity == "info"
        assert event.message == "Rate limit approaching"
