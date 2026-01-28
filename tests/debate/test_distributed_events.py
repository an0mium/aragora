"""
Tests for distributed debate event types and data structures.
"""

import pytest
import time
from dataclasses import asdict

from aragora.debate.distributed_events import (
    DistributedDebateEventType,
    DistributedDebateEvent,
    AgentProposal,
    AgentCritique,
    ConsensusVote,
    DistributedDebateState,
)


class TestDistributedDebateEventType:
    """Tests for DistributedDebateEventType enum."""

    def test_debate_lifecycle_events(self):
        """Test debate lifecycle event types exist."""
        assert DistributedDebateEventType.DEBATE_CREATED.value == "debate.created"
        assert DistributedDebateEventType.DEBATE_STARTED.value == "debate.started"
        assert DistributedDebateEventType.DEBATE_COMPLETED.value == "debate.completed"
        assert DistributedDebateEventType.DEBATE_FAILED.value == "debate.failed"

    def test_round_events(self):
        """Test round event types exist."""
        assert DistributedDebateEventType.ROUND_STARTED.value == "round.started"
        assert DistributedDebateEventType.ROUND_COMPLETED.value == "round.completed"

    def test_agent_events(self):
        """Test agent participation event types exist."""
        assert DistributedDebateEventType.AGENT_JOINED.value == "agent.joined"
        assert DistributedDebateEventType.AGENT_LEFT.value == "agent.left"
        assert DistributedDebateEventType.AGENT_PROPOSAL.value == "agent.proposal"
        assert DistributedDebateEventType.AGENT_CRITIQUE.value == "agent.critique"

    def test_consensus_events(self):
        """Test consensus event types exist."""
        assert DistributedDebateEventType.CONSENSUS_CHECK.value == "consensus.check"
        assert DistributedDebateEventType.CONSENSUS_REACHED.value == "consensus.reached"
        assert DistributedDebateEventType.CONSENSUS_VOTE.value == "consensus.vote"


class TestDistributedDebateEvent:
    """Tests for DistributedDebateEvent dataclass."""

    def test_event_creation(self):
        """Test creating a distributed debate event."""
        event = DistributedDebateEvent(
            event_type=DistributedDebateEventType.DEBATE_STARTED,
            debate_id="debate-123",
            source_instance="instance-1",
        )

        assert event.event_type == DistributedDebateEventType.DEBATE_STARTED
        assert event.debate_id == "debate-123"
        assert event.source_instance == "instance-1"
        assert event.round_number == 0
        assert event.agent_id is None
        assert event.data == {}
        assert event.version == 1

    def test_event_with_data(self):
        """Test event with additional data."""
        event = DistributedDebateEvent(
            event_type=DistributedDebateEventType.AGENT_PROPOSAL,
            debate_id="debate-123",
            source_instance="instance-1",
            round_number=2,
            agent_id="claude-3",
            data={"content": "My proposal", "confidence": 0.85},
        )

        assert event.round_number == 2
        assert event.agent_id == "claude-3"
        assert event.data["content"] == "My proposal"
        assert event.data["confidence"] == 0.85

    def test_event_timestamp(self):
        """Test event timestamp is set automatically."""
        before = time.time()
        event = DistributedDebateEvent(
            event_type=DistributedDebateEventType.DEBATE_CREATED,
            debate_id="debate-123",
            source_instance="instance-1",
        )
        after = time.time()

        assert before <= event.timestamp <= after

    def test_event_serialization(self):
        """Test event serialization to dict."""
        event = DistributedDebateEvent(
            event_type=DistributedDebateEventType.CONSENSUS_REACHED,
            debate_id="debate-123",
            source_instance="instance-1",
            round_number=3,
            data={"winner": "claude-3"},
        )

        data = event.to_dict()

        assert data["event_type"] == "consensus.reached"
        assert data["debate_id"] == "debate-123"
        assert data["source_instance"] == "instance-1"
        assert data["round_number"] == 3
        assert data["data"]["winner"] == "claude-3"

    def test_event_deserialization(self):
        """Test event deserialization from dict."""
        data = {
            "event_type": "debate.completed",
            "debate_id": "debate-456",
            "source_instance": "instance-2",
            "timestamp": 1700000000.0,
            "round_number": 5,
            "agent_id": None,
            "data": {"result": "success"},
            "version": 1,
        }

        event = DistributedDebateEvent.from_dict(data)

        assert event.event_type == DistributedDebateEventType.DEBATE_COMPLETED
        assert event.debate_id == "debate-456"
        assert event.source_instance == "instance-2"
        assert event.timestamp == 1700000000.0
        assert event.round_number == 5
        assert event.data["result"] == "success"

    def test_event_roundtrip(self):
        """Test event serialization roundtrip."""
        original = DistributedDebateEvent(
            event_type=DistributedDebateEventType.AGENT_CRITIQUE,
            debate_id="debate-789",
            source_instance="instance-3",
            round_number=2,
            agent_id="gpt-4",
            data={"target": "claude-3", "rating": 0.7},
        )

        data = original.to_dict()
        restored = DistributedDebateEvent.from_dict(data)

        assert restored.event_type == original.event_type
        assert restored.debate_id == original.debate_id
        assert restored.source_instance == original.source_instance
        assert restored.round_number == original.round_number
        assert restored.agent_id == original.agent_id
        assert restored.data == original.data


class TestAgentProposal:
    """Tests for AgentProposal dataclass."""

    def test_proposal_creation(self):
        """Test creating an agent proposal."""
        proposal = AgentProposal(
            agent_id="claude-3",
            instance_id="instance-1",
            content="We should use PostgreSQL for the database.",
            round_number=1,
            confidence=0.85,
            reasoning="PostgreSQL offers better ACID compliance.",
        )

        assert proposal.agent_id == "claude-3"
        assert proposal.instance_id == "instance-1"
        assert proposal.content == "We should use PostgreSQL for the database."
        assert proposal.round_number == 1
        assert proposal.confidence == 0.85
        assert proposal.reasoning == "PostgreSQL offers better ACID compliance."

    def test_proposal_serialization(self):
        """Test proposal serialization."""
        proposal = AgentProposal(
            agent_id="gpt-4",
            instance_id="instance-2",
            content="Consider MongoDB for flexibility.",
            round_number=1,
        )

        data = proposal.to_dict()

        assert data["agent_id"] == "gpt-4"
        assert data["instance_id"] == "instance-2"
        assert data["content"] == "Consider MongoDB for flexibility."
        assert data["round_number"] == 1


class TestAgentCritique:
    """Tests for AgentCritique dataclass."""

    def test_critique_creation(self):
        """Test creating an agent critique."""
        critique = AgentCritique(
            agent_id="gpt-4",
            instance_id="instance-1",
            target_agent_id="claude-3",
            content="Good analysis but missing cost considerations.",
            round_number=1,
            rating=0.7,
            strengths=["Clear reasoning", "Good examples"],
            weaknesses=["No cost analysis", "Missing scalability discussion"],
        )

        assert critique.agent_id == "gpt-4"
        assert critique.target_agent_id == "claude-3"
        assert critique.rating == 0.7
        assert len(critique.strengths) == 2
        assert len(critique.weaknesses) == 2

    def test_critique_serialization(self):
        """Test critique serialization."""
        critique = AgentCritique(
            agent_id="gemini",
            instance_id="instance-2",
            target_agent_id="gpt-4",
            content="Interesting perspective.",
            round_number=2,
        )

        data = critique.to_dict()

        assert data["agent_id"] == "gemini"
        assert data["target_agent_id"] == "gpt-4"
        assert data["round_number"] == 2


class TestConsensusVote:
    """Tests for ConsensusVote dataclass."""

    def test_vote_creation(self):
        """Test creating a consensus vote."""
        vote = ConsensusVote(
            agent_id="gpt-4",
            instance_id="instance-1",
            proposal_agent_id="claude-3",
            vote="support",
            round_number=3,
            confidence=0.9,
            reasoning="Strong technical argument.",
        )

        assert vote.agent_id == "gpt-4"
        assert vote.proposal_agent_id == "claude-3"
        assert vote.vote == "support"
        assert vote.confidence == 0.9

    def test_vote_types(self):
        """Test different vote types."""
        votes = [
            ConsensusVote(
                agent_id="agent-1",
                instance_id="i1",
                proposal_agent_id="p1",
                vote="support",
                round_number=1,
            ),
            ConsensusVote(
                agent_id="agent-2",
                instance_id="i1",
                proposal_agent_id="p1",
                vote="oppose",
                round_number=1,
            ),
            ConsensusVote(
                agent_id="agent-3",
                instance_id="i1",
                proposal_agent_id="p1",
                vote="abstain",
                round_number=1,
            ),
        ]

        assert votes[0].vote == "support"
        assert votes[1].vote == "oppose"
        assert votes[2].vote == "abstain"


class TestDistributedDebateState:
    """Tests for DistributedDebateState dataclass."""

    def test_state_creation(self):
        """Test creating debate state."""
        state = DistributedDebateState(
            debate_id="debate-123",
            task="What database should we use?",
            coordinator_instance="instance-1",
        )

        assert state.debate_id == "debate-123"
        assert state.task == "What database should we use?"
        assert state.coordinator_instance == "instance-1"
        assert state.status == "created"
        assert state.current_round == 0
        assert state.max_rounds == 5

    def test_state_with_participants(self):
        """Test state with participants."""
        state = DistributedDebateState(
            debate_id="debate-456",
            task="Architecture review",
            coordinator_instance="instance-1",
            instances={"instance-1": {"joined_at": 1700000000}},
            agents={"claude-3": {"status": "active"}},
        )

        assert "instance-1" in state.instances
        assert "claude-3" in state.agents

    def test_state_serialization(self):
        """Test state serialization."""
        proposal = AgentProposal(
            agent_id="claude-3",
            instance_id="instance-1",
            content="Use microservices",
            round_number=1,
        )

        state = DistributedDebateState(
            debate_id="debate-789",
            task="Architecture decision",
            coordinator_instance="instance-1",
            current_round=2,
            proposals=[proposal],
            consensus_reached=True,
            final_answer="We should use microservices",
            winning_agent="claude-3",
            confidence=0.88,
        )

        data = state.to_dict()

        assert data["debate_id"] == "debate-789"
        assert data["current_round"] == 2
        assert len(data["proposals"]) == 1
        assert data["consensus_reached"] is True
        assert data["final_answer"] == "We should use microservices"
        assert data["winning_agent"] == "claude-3"
        assert data["confidence"] == 0.88

    def test_state_lifecycle(self):
        """Test state status transitions."""
        state = DistributedDebateState(
            debate_id="debate-lifecycle",
            task="Test task",
            coordinator_instance="instance-1",
        )

        assert state.status == "created"
        assert state.started_at is None
        assert state.completed_at is None

        # Simulate start
        state.status = "running"
        state.started_at = time.time()
        assert state.status == "running"
        assert state.started_at is not None

        # Simulate completion
        state.status = "completed"
        state.completed_at = time.time()
        assert state.status == "completed"
        assert state.completed_at is not None
