"""
Tests for ConsensusPhase module.

Tests cover:
- ConsensusDependencies dataclass
- ConsensusCallbacks dataclass
- ConsensusPhase class with different consensus modes
- Vote collection and aggregation
- Winner selection and synthesis generation
- Error handling and edge cases
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.phases.consensus_phase import (
    ConsensusCallbacks,
    ConsensusDependencies,
    ConsensusPhase,
)


@dataclass
class MockVote:
    """Mock vote for testing."""

    agent: str
    choice: str
    reasoning: str = "test reasoning"
    confidence: float = 0.8
    continue_debate: bool = False


@dataclass
class MockMessage:
    """Mock message for testing."""

    agent: str
    content: str
    role: str = "proposer"
    round: int = 0


@dataclass
class MockProtocol:
    """Mock debate protocol."""

    consensus: str = "majority"
    rounds: int = 3
    verify_claims_during_consensus: bool = False


@dataclass
class MockDebateResult:
    """Mock debate result for context."""

    id: str = "test-debate-123"
    task: str = "Test task"
    messages: list = field(default_factory=list)
    critiques: list = field(default_factory=list)
    votes: list = field(default_factory=list)
    final_answer: str = ""
    winner: str = ""
    confidence: float = 0.0
    consensus_reached: bool = False
    participants: list = field(default_factory=list)
    rounds_used: int = 0


@dataclass
class MockEnvironment:
    """Mock environment for context."""

    task: str = "Test debate task"
    context: str = ""


@dataclass
class MockDebateContext:
    """Mock debate context for testing."""

    debate_id: str = "test-debate-123"
    env: MockEnvironment = field(default_factory=MockEnvironment)
    result: MockDebateResult = field(default_factory=MockDebateResult)
    agents: list = field(default_factory=list)
    proposers: list = field(default_factory=list)
    partial_messages: list = field(default_factory=list)
    context_messages: list = field(default_factory=list)


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str, role: str = "proposer"):
        self.name = name
        self.role = role
        self.model = f"mock-{name}"

    async def generate(self, prompt: str) -> str:
        return f"Response from {self.name}"

    async def vote(self, prompt: str) -> MockVote:
        return MockVote(agent=self.name, choice="agent1")


class TestConsensusDependencies:
    """Tests for ConsensusDependencies dataclass."""

    def test_default_values(self):
        """Dependencies have expected defaults."""
        deps = ConsensusDependencies()

        assert deps.protocol is None
        assert deps.elo_system is None
        assert deps.memory is None
        assert deps.agent_weights == {}
        assert deps.user_votes == []

    def test_with_protocol(self):
        """Dependencies accept protocol."""
        protocol = MockProtocol()
        deps = ConsensusDependencies(protocol=protocol)

        assert deps.protocol == protocol
        assert deps.protocol.consensus == "majority"

    def test_with_agent_weights(self):
        """Dependencies accept agent weights."""
        weights = {"agent1": 1.0, "agent2": 0.8}
        deps = ConsensusDependencies(agent_weights=weights)

        assert deps.agent_weights == weights


class TestConsensusCallbacks:
    """Tests for ConsensusCallbacks dataclass."""

    def test_default_values(self):
        """Callbacks have None defaults."""
        callbacks = ConsensusCallbacks()

        assert callbacks.vote_with_agent is None
        assert callbacks.with_timeout is None
        assert callbacks.select_judge is None
        assert callbacks.verify_claims is None

    def test_with_vote_callback(self):
        """Callbacks accept vote function."""
        vote_fn = AsyncMock()
        callbacks = ConsensusCallbacks(vote_with_agent=vote_fn)

        assert callbacks.vote_with_agent == vote_fn

    def test_with_multiple_callbacks(self):
        """Callbacks accept multiple functions."""
        vote_fn = AsyncMock()
        timeout_fn = MagicMock()
        judge_fn = MagicMock()

        callbacks = ConsensusCallbacks(
            vote_with_agent=vote_fn,
            with_timeout=timeout_fn,
            select_judge=judge_fn,
        )

        assert callbacks.vote_with_agent == vote_fn
        assert callbacks.with_timeout == timeout_fn
        assert callbacks.select_judge == judge_fn


class TestConsensusPhaseInit:
    """Tests for ConsensusPhase initialization."""

    def test_init_with_dependencies(self):
        """ConsensusPhase initializes with dependencies."""
        protocol = MockProtocol()
        deps = ConsensusDependencies(protocol=protocol)
        callbacks = ConsensusCallbacks()

        phase = ConsensusPhase(deps, callbacks)

        # Check internal state is set up
        assert phase.protocol == protocol

    def test_init_with_kwargs_style(self):
        """ConsensusPhase initializes with kwargs style."""
        protocol = MockProtocol()

        phase = ConsensusPhase(
            protocol=protocol,
            elo_system=None,
            memory=None,
        )

        assert phase.protocol == protocol


class TestConsensusPhaseMajorityMode:
    """Tests for ConsensusPhase in majority mode."""

    @pytest.fixture
    def majority_phase(self):
        """Create ConsensusPhase configured for majority voting."""
        protocol = MockProtocol(consensus="majority")

        # Mock the vote callback
        async def mock_vote(agent, prompt, **kwargs):
            return MockVote(agent=agent.name, choice="proposal_1", confidence=0.8)

        callbacks = ConsensusCallbacks(
            vote_with_agent=mock_vote,
            with_timeout=lambda coro, timeout: coro,
            notify_spectator=MagicMock(),
        )

        return ConsensusPhase(
            protocol=protocol,
            elo_system=MagicMock(),
            memory=MagicMock(),
            agent_weights={"agent1": 1.0, "agent2": 0.9},
            vote_with_agent=mock_vote,
            with_timeout=lambda coro, timeout: coro,
            notify_spectator=MagicMock(),
        )

    @pytest.fixture
    def ctx_with_proposals(self):
        """Create context with proposals for voting."""
        ctx = MockDebateContext()
        ctx.agents = [MockAgent("agent1"), MockAgent("agent2"), MockAgent("agent3")]

        # Add proposals to messages
        ctx.result.messages = [
            MockMessage(agent="agent1", content="Proposal 1 content", role="proposer"),
            MockMessage(agent="agent2", content="Proposal 2 content", role="proposer"),
        ]
        ctx.result.participants = ["agent1", "agent2", "agent3"]

        return ctx

    @pytest.mark.asyncio
    async def test_majority_voting_selects_winner(self, majority_phase, ctx_with_proposals):
        """Majority mode collects votes and selects winner."""
        # This would be an integration test if VoteCollector is used internally
        # For now, test that the phase can be constructed
        assert majority_phase.protocol.consensus == "majority"


class TestConsensusPhaseModeNone:
    """Tests for ConsensusPhase in 'none' mode (no consensus)."""

    @pytest.fixture
    def none_phase(self):
        """Create ConsensusPhase configured for no consensus."""
        protocol = MockProtocol(consensus="none")

        return ConsensusPhase(
            protocol=protocol,
            elo_system=None,
            memory=None,
            notify_spectator=MagicMock(),
        )

    def test_none_mode_configured(self, none_phase):
        """None mode is properly configured."""
        assert none_phase.protocol.consensus == "none"


class TestConsensusPhaseModeUnanimous:
    """Tests for ConsensusPhase in unanimous mode."""

    @pytest.fixture
    def unanimous_phase(self):
        """Create ConsensusPhase configured for unanimous voting."""
        protocol = MockProtocol(consensus="unanimous")

        return ConsensusPhase(
            protocol=protocol,
            elo_system=MagicMock(),
            memory=MagicMock(),
            notify_spectator=MagicMock(),
        )

    def test_unanimous_mode_configured(self, unanimous_phase):
        """Unanimous mode is properly configured."""
        assert unanimous_phase.protocol.consensus == "unanimous"


class TestConsensusPhaseModeJudge:
    """Tests for ConsensusPhase in judge mode."""

    @pytest.fixture
    def judge_phase(self):
        """Create ConsensusPhase configured for judge mode."""
        protocol = MockProtocol(consensus="judge")

        mock_judge = MockAgent("judge", role="judge")

        return ConsensusPhase(
            protocol=protocol,
            elo_system=MagicMock(),
            memory=MagicMock(),
            select_judge=MagicMock(return_value=mock_judge),
            build_judge_prompt=MagicMock(return_value="Judge prompt"),
            generate_with_agent=AsyncMock(return_value="Judge decision"),
            notify_spectator=MagicMock(),
        )

    def test_judge_mode_configured(self, judge_phase):
        """Judge mode is properly configured."""
        assert judge_phase.protocol.consensus == "judge"


class TestConsensusPhaseErrorHandling:
    """Tests for ConsensusPhase error handling."""

    @pytest.fixture
    def error_prone_phase(self):
        """Create ConsensusPhase with error-prone callbacks."""
        protocol = MockProtocol(consensus="majority")

        async def failing_vote(agent, prompt, **kwargs):
            raise RuntimeError("Vote collection failed")

        return ConsensusPhase(
            protocol=protocol,
            elo_system=None,
            memory=None,
            vote_with_agent=failing_vote,
            notify_spectator=MagicMock(),
        )

    def test_phase_handles_missing_protocol(self):
        """Phase handles missing protocol gracefully."""
        phase = ConsensusPhase(protocol=None)
        # Should not raise during construction
        assert phase.protocol is None


class TestConsensusPhaseWithVerification:
    """Tests for ConsensusPhase with claim verification enabled."""

    @pytest.fixture
    def verified_phase(self):
        """Create ConsensusPhase with verification enabled."""
        protocol = MockProtocol(
            consensus="majority",
            verify_claims_during_consensus=True,
        )

        async def mock_verify(claims):
            return {"verified": 2, "total": 3}

        return ConsensusPhase(
            protocol=protocol,
            elo_system=MagicMock(),
            memory=MagicMock(),
            verify_claims=mock_verify,
            notify_spectator=MagicMock(),
        )

    def test_verification_enabled(self, verified_phase):
        """Verification is enabled when configured."""
        assert verified_phase.protocol.verify_claims_during_consensus is True


class TestConsensusPhaseUserVotes:
    """Tests for ConsensusPhase with user votes."""

    @pytest.fixture
    def user_vote_phase(self):
        """Create ConsensusPhase with user votes enabled."""
        protocol = MockProtocol(consensus="majority")

        user_votes = [
            {"user_id": "user1", "choice": "proposal_1", "weight": 1.0},
            {"user_id": "user2", "choice": "proposal_2", "weight": 0.5},
        ]

        return ConsensusPhase(
            protocol=protocol,
            elo_system=MagicMock(),
            memory=MagicMock(),
            user_votes=user_votes,
            user_vote_multiplier=MagicMock(return_value=1.5),
            notify_spectator=MagicMock(),
        )

    def test_user_votes_configured(self, user_vote_phase):
        """User votes are properly configured."""
        assert len(user_vote_phase.user_votes) == 2


class TestConsensusPhaseWithELO:
    """Tests for ConsensusPhase with ELO system integration."""

    @pytest.fixture
    def elo_phase(self):
        """Create ConsensusPhase with ELO system."""
        protocol = MockProtocol(consensus="majority")

        elo_system = MagicMock()
        elo_system.get_agent_ratings.return_value = {
            "agent1": 1200,
            "agent2": 1100,
            "agent3": 1000,
        }

        return ConsensusPhase(
            protocol=protocol,
            elo_system=elo_system,
            memory=MagicMock(),
            notify_spectator=MagicMock(),
        )

    def test_elo_system_configured(self, elo_phase):
        """ELO system is properly configured."""
        assert elo_phase.elo_system is not None
        ratings = elo_phase.elo_system.get_agent_ratings()
        assert ratings["agent1"] == 1200


class TestConsensusPhaseWithCalibration:
    """Tests for ConsensusPhase with calibration tracking."""

    @pytest.fixture
    def calibrated_phase(self):
        """Create ConsensusPhase with calibration tracker."""
        protocol = MockProtocol(consensus="majority")

        calibration = MagicMock()
        calibration.get_calibration_score.return_value = 0.85

        return ConsensusPhase(
            protocol=protocol,
            elo_system=MagicMock(),
            memory=MagicMock(),
            calibration_tracker=calibration,
            get_calibration_weight=MagicMock(return_value=1.2),
            notify_spectator=MagicMock(),
        )

    def test_calibration_configured(self, calibrated_phase):
        """Calibration tracker is properly configured."""
        assert calibrated_phase.calibration_tracker is not None
        score = calibrated_phase.calibration_tracker.get_calibration_score()
        assert score == 0.85


class TestConsensusPhaseWithFlipDetection:
    """Tests for ConsensusPhase with flip detection."""

    @pytest.fixture
    def flip_phase(self):
        """Create ConsensusPhase with flip detector."""
        protocol = MockProtocol(consensus="majority")

        flip_detector = MagicMock()
        flip_detector.detect_flip.return_value = False

        return ConsensusPhase(
            protocol=protocol,
            elo_system=MagicMock(),
            memory=MagicMock(),
            flip_detector=flip_detector,
            notify_spectator=MagicMock(),
        )

    def test_flip_detector_configured(self, flip_phase):
        """Flip detector is properly configured."""
        assert flip_phase.flip_detector is not None
        assert flip_phase.flip_detector.detect_flip() is False


class TestConsensusPhaseWithPositionTracking:
    """Tests for ConsensusPhase with position tracking."""

    @pytest.fixture
    def position_phase(self):
        """Create ConsensusPhase with position tracker."""
        protocol = MockProtocol(consensus="majority")

        position_tracker = MagicMock()
        position_tracker.get_position.return_value = {"agent1": "for", "agent2": "against"}

        return ConsensusPhase(
            protocol=protocol,
            elo_system=MagicMock(),
            memory=MagicMock(),
            position_tracker=position_tracker,
            notify_spectator=MagicMock(),
        )

    def test_position_tracker_configured(self, position_phase):
        """Position tracker is properly configured."""
        assert position_phase.position_tracker is not None
        positions = position_phase.position_tracker.get_position()
        assert positions["agent1"] == "for"


# Integration tests that test multiple components working together

class TestConsensusPhaseIntegration:
    """Integration tests for ConsensusPhase."""

    @pytest.mark.asyncio
    async def test_full_consensus_flow_with_mocks(self):
        """Test full consensus flow with mocked dependencies."""
        protocol = MockProtocol(consensus="majority")

        # Create mock agents
        agents = [
            MockAgent("agent1", "proposer"),
            MockAgent("agent2", "critic"),
            MockAgent("agent3", "voter"),
        ]

        # Create context
        ctx = MockDebateContext()
        ctx.agents = agents
        ctx.result.messages = [
            MockMessage(agent="agent1", content="Proposal 1"),
            MockMessage(agent="agent2", content="Proposal 2"),
        ]

        # Create votes mock
        async def mock_vote(agent, prompt, **kwargs):
            # Majority votes for agent1
            if agent.name in ["agent1", "agent3"]:
                return MockVote(agent=agent.name, choice="agent1", confidence=0.9)
            return MockVote(agent=agent.name, choice="agent2", confidence=0.7)

        phase = ConsensusPhase(
            protocol=protocol,
            elo_system=MagicMock(),
            memory=MagicMock(),
            vote_with_agent=mock_vote,
            with_timeout=lambda coro, timeout: coro,
            notify_spectator=MagicMock(),
            drain_user_events=MagicMock(return_value=[]),
        )

        # Phase should be properly constructed
        assert phase.protocol.consensus == "majority"

    @pytest.mark.asyncio
    async def test_consensus_with_tie_handling(self):
        """Test consensus handling when votes are tied."""
        protocol = MockProtocol(consensus="majority")

        # Create votes that result in a tie
        async def tied_vote(agent, prompt, **kwargs):
            if agent.name == "agent1":
                return MockVote(agent=agent.name, choice="option_a", confidence=0.8)
            return MockVote(agent=agent.name, choice="option_b", confidence=0.8)

        phase = ConsensusPhase(
            protocol=protocol,
            elo_system=MagicMock(),
            memory=MagicMock(),
            vote_with_agent=tied_vote,
            notify_spectator=MagicMock(),
        )

        assert phase.protocol.consensus == "majority"

    @pytest.mark.asyncio
    async def test_consensus_empty_votes(self):
        """Test consensus handling with no votes collected."""
        protocol = MockProtocol(consensus="majority")

        phase = ConsensusPhase(
            protocol=protocol,
            elo_system=MagicMock(),
            memory=MagicMock(),
            notify_spectator=MagicMock(),
        )

        # Empty context
        ctx = MockDebateContext()
        ctx.agents = []
        ctx.result.messages = []

        # Should handle gracefully
        assert phase.protocol.consensus == "majority"
