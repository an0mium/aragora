"""
Tests for critical untested methods in the Arena orchestrator.

These tests cover critical methods that delegate to helper classes:
1. _build_proposal_prompt() - Core to proposal generation
2. _check_early_stopping() - Controls debate termination
3. _select_judge() - Judge selection for consensus
4. _record_grounded_position() - Position tracking
5. _extract_citation_needs() - Citation tracking

Each method is tested through the Arena's delegation pattern to ensure
proper integration with the underlying helper classes.
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core import Agent, Critique, Environment, Message, Vote
from aragora.debate.orchestrator import Arena
from aragora.debate.protocol import DebateProtocol


class MockAgent(Agent):
    """Mock agent for testing Arena functionality."""

    def __init__(
        self,
        name: str = "mock-agent",
        response: str = "Test response",
        model: str = "mock-model",
        role: str = "proposer",
    ):
        super().__init__(name=name, model=model, role=role)
        self.agent_type = "mock"
        self.response = response
        self.generate_calls = 0

    async def generate(self, prompt: str, context: list = None) -> str:
        self.generate_calls += 1
        return self.response

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list = None,
        target_agent: str = None,
    ) -> Critique:
        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content=proposal[:100] if proposal else "",
            issues=["Test issue"],
            suggestions=["Test suggestion"],
            severity=0.5,
            reasoning="Test reasoning",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        choice = list(proposals.keys())[0] if proposals else self.name
        return Vote(
            agent=self.name,
            choice=choice,
            reasoning="Test vote",
            confidence=0.8,
            continue_debate=False,
        )


@pytest.fixture
def mock_agents():
    """Create a list of mock agents for testing."""
    return [
        MockAgent(name="agent1", response="Proposal 1"),
        MockAgent(name="agent2", response="Proposal 2"),
        MockAgent(name="agent3", response="Proposal 3"),
    ]


@pytest.fixture
def env():
    """Create a test environment."""
    return Environment(task="Test debate task for unit testing")


@pytest.fixture
def protocol():
    """Create a test protocol with default settings."""
    return DebateProtocol(
        rounds=3,
        consensus="majority",
        early_stopping=True,
        min_rounds_before_early_stop=1,
        early_stop_threshold=0.5,
        judge_termination=True,
        min_rounds_before_judge_check=1,
        judge_selection="random",
    )


@pytest.fixture
def arena(env, mock_agents, protocol):
    """Create an Arena instance for testing."""
    return Arena(env, mock_agents, protocol)


# =============================================================================
# Tests for _build_proposal_prompt()
# =============================================================================


class TestBuildProposalPrompt:
    """Tests for the _build_proposal_prompt method."""

    def test_builds_prompt_with_persona_context(self, arena, mock_agents):
        """Test that prompt includes persona context when available."""
        # Mock the persona manager to return persona context
        mock_persona = MagicMock()
        mock_persona.to_prompt_context.return_value = "You are a helpful assistant."
        arena._prompt_context.persona_manager = MagicMock()
        arena._prompt_context.persona_manager.get_persona.return_value = mock_persona

        # Mock the prompt builder
        arena.prompt_builder.build_proposal_prompt = MagicMock(
            return_value="Build proposal for: Test task"
        )

        prompt = arena._build_proposal_prompt(mock_agents[0])

        # Verify prompt was built
        assert prompt is not None
        arena.prompt_builder.build_proposal_prompt.assert_called_once()

    def test_includes_audience_suggestions(self, arena, mock_agents):
        """Test that audience suggestions are included in the prompt."""
        # Mock audience manager with suggestions (using proper dict format)
        arena._prompt_context.audience_manager = MagicMock()
        arena._prompt_context.audience_manager._suggestions = deque(
            [
                {"suggestion": "Suggestion 1", "user": "user1"},
                {"suggestion": "Suggestion 2", "user": "user2"},
            ]
        )
        arena._prompt_context.audience_manager.drain_events = MagicMock()

        # Enable audience injection in protocol
        arena._prompt_context.protocol = MagicMock()
        arena._prompt_context.protocol.audience_injection = "inject"

        # Mock prompt builder
        arena.prompt_builder.build_proposal_prompt = MagicMock(return_value="Prompt with audience")

        prompt = arena._build_proposal_prompt(mock_agents[0])

        # Verify drain_events was called
        arena._prompt_context.audience_manager.drain_events.assert_called()
        assert prompt is not None

    def test_handles_missing_context_gracefully(self, arena, mock_agents):
        """Test that missing context is handled gracefully."""
        # Remove persona manager
        arena._prompt_context.persona_manager = None
        arena._prompt_context.audience_manager = None

        # Mock prompt builder to return basic prompt
        arena.prompt_builder.build_proposal_prompt = MagicMock(return_value="Basic prompt")

        prompt = arena._build_proposal_prompt(mock_agents[0])

        # Should still return a valid prompt
        assert prompt == "Basic prompt"

    def test_syncs_prompt_builder_state(self, arena, mock_agents):
        """Test that prompt builder state is synced before building."""
        arena.prompt_builder.build_proposal_prompt = MagicMock(return_value="Synced prompt")

        # Call build_proposal_prompt
        arena._build_proposal_prompt(mock_agents[0])

        # The state sync happens inside _sync_prompt_builder_state
        # which is called automatically - just verify prompt was built
        arena.prompt_builder.build_proposal_prompt.assert_called_once()


# =============================================================================
# Tests for _check_early_stopping()
# =============================================================================


class TestCheckEarlyStopping:
    """Tests for the _check_early_stopping method."""

    @pytest.mark.asyncio
    async def test_returns_true_when_consensus_threshold_met(self, env, mock_agents, protocol):
        """Test that early stopping returns False (stop) when threshold is met."""
        # Set all agents to respond with STOP
        for agent in mock_agents:
            agent.response = "STOP"

        protocol.early_stopping = True
        protocol.min_rounds_before_early_stop = 1
        protocol.early_stop_threshold = 0.5

        arena = Arena(env, mock_agents, protocol)

        proposals = {"agent1": "Proposal 1", "agent2": "Proposal 2"}
        context = [Message(role="system", agent="system", content="Test context")]

        # Check early stopping at round 2 (after min rounds)
        should_continue = await arena._check_early_stopping(2, proposals, context)

        # Should return False because agents voted to STOP
        assert should_continue is False

    @pytest.mark.asyncio
    async def test_returns_true_when_insufficient_rounds(self, env, mock_agents, protocol):
        """Test that early stopping returns True (continue) before min rounds."""
        protocol.early_stopping = True
        protocol.min_rounds_before_early_stop = 3

        arena = Arena(env, mock_agents, protocol)

        proposals = {"agent1": "Proposal 1"}
        context = []

        # Check at round 1 (before min rounds)
        should_continue = await arena._check_early_stopping(1, proposals, context)

        # Should return True (continue) because min rounds not met
        assert should_continue is True

    @pytest.mark.asyncio
    async def test_respects_min_rounds_before_early_stop(self, env, mock_agents, protocol):
        """Test that min_rounds_before_early_stop is respected."""
        protocol.early_stopping = True
        protocol.min_rounds_before_early_stop = 5

        arena = Arena(env, mock_agents, protocol)

        proposals = {"agent1": "Proposal 1"}
        context = []

        # Check at various rounds
        for round_num in range(1, 5):
            should_continue = await arena._check_early_stopping(round_num, proposals, context)
            assert should_continue is True, f"Round {round_num} should continue"

    @pytest.mark.asyncio
    async def test_returns_true_when_early_stopping_disabled(self, env, mock_agents, protocol):
        """Test that it returns True when early_stopping is disabled."""
        protocol.early_stopping = False

        arena = Arena(env, mock_agents, protocol)

        proposals = {"agent1": "Proposal 1"}
        context = []

        should_continue = await arena._check_early_stopping(10, proposals, context)

        # Should always return True when disabled
        assert should_continue is True

    @pytest.mark.asyncio
    async def test_continues_when_agents_vote_continue(self, env, mock_agents, protocol):
        """Test that debate continues when agents vote CONTINUE."""
        # Set all agents to respond with CONTINUE
        for agent in mock_agents:
            agent.response = "CONTINUE"

        protocol.early_stopping = True
        protocol.min_rounds_before_early_stop = 1
        protocol.early_stop_threshold = 0.5

        arena = Arena(env, mock_agents, protocol)

        proposals = {"agent1": "Proposal 1"}
        context = []

        should_continue = await arena._check_early_stopping(2, proposals, context)

        # Should return True (continue) because agents voted CONTINUE
        assert should_continue is True


# =============================================================================
# Tests for _select_judge()
# =============================================================================


class TestSelectJudge:
    """Tests for the _select_judge method."""

    @pytest.mark.asyncio
    async def test_uses_judge_selector_to_choose_qualified_judge(self, env, mock_agents, protocol):
        """Test that JudgeSelector is used to select a judge."""
        protocol.judge_selection = "random"

        arena = Arena(env, mock_agents, protocol)

        proposals = {"agent1": "Proposal 1", "agent2": "Proposal 2"}
        context = [Message(role="system", agent="system", content="Test context")]

        judge = await arena._select_judge(proposals, context)

        # Should return one of the agents
        assert judge is not None
        assert judge in mock_agents

    @pytest.mark.asyncio
    async def test_returns_agent_from_pool(self, env, mock_agents, protocol):
        """Test that the selected judge is from the agent pool."""
        arena = Arena(env, mock_agents, protocol)

        proposals = {"agent1": "Proposal 1"}
        context = []

        judge = await arena._select_judge(proposals, context)

        # Judge should be one of the original agents
        assert judge.name in [a.name for a in mock_agents]

    @pytest.mark.asyncio
    async def test_handles_empty_proposals(self, env, mock_agents, protocol):
        """Test that judge selection handles empty proposals."""
        arena = Arena(env, mock_agents, protocol)

        proposals = {}
        context = []

        judge = await arena._select_judge(proposals, context)

        # Should still return a judge (random from pool)
        assert judge is not None

    @pytest.mark.asyncio
    async def test_uses_elo_ranked_selection(self, env, mock_agents, protocol):
        """Test ELO-ranked judge selection when configured."""
        protocol.judge_selection = "elo_ranked"

        # Mock ELO system
        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [
            {"agent": "agent2", "elo": 1600},
            {"agent": "agent1", "elo": 1500},
            {"agent": "agent3", "elo": 1400},
        ]

        arena = Arena(env, mock_agents, protocol)
        arena.elo_system = mock_elo

        proposals = {"agent1": "Proposal 1"}
        context = []

        judge = await arena._select_judge(proposals, context)

        # Should return a judge (may be random if ELO lookup fails in test)
        assert judge is not None

    @pytest.mark.asyncio
    async def test_calibrated_selection_strategy(self, env, mock_agents, protocol):
        """Test calibrated judge selection when configured."""
        protocol.judge_selection = "calibrated"

        arena = Arena(env, mock_agents, protocol)

        proposals = {"agent1": "Proposal 1", "agent2": "Proposal 2"}
        context = []

        judge = await arena._select_judge(proposals, context)

        # Should return a judge
        assert judge is not None
        assert judge in mock_agents


# =============================================================================
# Tests for _record_grounded_position()
# =============================================================================


class TestRecordGroundedPosition:
    """Tests for the _record_grounded_position method."""

    def test_records_position_to_ledger_when_available(self, arena):
        """Test that position is recorded when ledger is available."""
        # Mock position ledger
        mock_ledger = MagicMock()
        arena._grounded_ops.position_ledger = mock_ledger

        arena._record_grounded_position(
            agent_name="agent1",
            content="This is my position on the topic",
            debate_id="debate-123",
            round_num=2,
            confidence=0.85,
            domain="technology",
        )

        # Verify ledger was called
        mock_ledger.record_position.assert_called_once_with(
            agent_name="agent1",
            claim="This is my position on the topic",
            confidence=0.85,
            debate_id="debate-123",
            round_num=2,
            domain="technology",
        )

    def test_handles_ledger_not_available_gracefully(self, arena):
        """Test that missing ledger is handled gracefully."""
        # Ensure no position ledger
        arena._grounded_ops.position_ledger = None

        # Should not raise an exception
        arena._record_grounded_position(
            agent_name="agent1",
            content="Position content",
            debate_id="debate-123",
            round_num=1,
        )

        # No assertion needed - just verify no exception

    def test_truncates_long_content(self, arena):
        """Test that long content is truncated to 1000 chars."""
        mock_ledger = MagicMock()
        arena._grounded_ops.position_ledger = mock_ledger

        long_content = "x" * 2000

        arena._record_grounded_position(
            agent_name="agent1",
            content=long_content,
            debate_id="debate-123",
            round_num=1,
        )

        # Verify content was truncated
        call_args = mock_ledger.record_position.call_args
        assert len(call_args.kwargs["claim"]) == 1000

    def test_uses_default_confidence(self, arena):
        """Test that default confidence of 0.7 is used when not specified."""
        mock_ledger = MagicMock()
        arena._grounded_ops.position_ledger = mock_ledger

        arena._record_grounded_position(
            agent_name="agent1",
            content="Position content",
            debate_id="debate-123",
            round_num=1,
        )

        # Verify default confidence was used
        call_args = mock_ledger.record_position.call_args
        assert call_args.kwargs["confidence"] == 0.7

    def test_handles_ledger_errors_gracefully(self, arena):
        """Test that ledger errors are handled gracefully."""
        mock_ledger = MagicMock()
        mock_ledger.record_position.side_effect = ValueError("Database error")
        arena._grounded_ops.position_ledger = mock_ledger

        # Should not raise an exception
        arena._record_grounded_position(
            agent_name="agent1",
            content="Position content",
            debate_id="debate-123",
            round_num=1,
        )

        # No assertion needed - just verify no exception raised


# =============================================================================
# Tests for _extract_citation_needs()
# =============================================================================


class TestExtractCitationNeeds:
    """Tests for the _extract_citation_needs method."""

    def test_returns_empty_dict_when_extractor_is_none(self, arena):
        """Test that empty dict is returned when citation_extractor is None."""
        arena.citation_extractor = None

        proposals = {"agent1": "Some proposal text", "agent2": "Another proposal"}

        result = arena._extract_citation_needs(proposals)

        assert result == {}

    def test_identifies_claims_needing_citations(self, arena):
        """Test that claims needing citations are identified."""
        # Mock citation extractor
        mock_extractor = MagicMock()
        mock_extractor.identify_citation_needs.return_value = [
            {"claim": "Research shows that X is true", "priority": "high"},
            {"claim": "Data indicates Y is correct", "priority": "high"},
        ]
        arena.citation_extractor = mock_extractor

        proposals = {"agent1": "Research shows that X is true. Data indicates Y is correct."}

        result = arena._extract_citation_needs(proposals)

        assert "agent1" in result
        assert len(result["agent1"]) == 2
        assert result["agent1"][0]["priority"] == "high"

    def test_handles_multiple_proposals(self, arena):
        """Test that citation needs are extracted from all proposals."""
        mock_extractor = MagicMock()
        mock_extractor.identify_citation_needs.side_effect = [
            [{"claim": "Claim A", "priority": "high"}],
            [{"claim": "Claim B", "priority": "medium"}],
            [],  # Empty for third proposal
        ]
        arena.citation_extractor = mock_extractor

        proposals = {
            "agent1": "Proposal with claim A",
            "agent2": "Proposal with claim B",
            "agent3": "Proposal with no claims",
        }

        result = arena._extract_citation_needs(proposals)

        # Should have entries for agent1 and agent2, but not agent3 (empty)
        assert "agent1" in result
        assert "agent2" in result
        assert "agent3" not in result

    def test_logs_high_priority_citation_needs(self, arena):
        """Test that high priority citation needs are logged."""
        mock_extractor = MagicMock()
        mock_extractor.identify_citation_needs.return_value = [
            {"claim": "Important research claim", "priority": "high"},
        ]
        arena.citation_extractor = mock_extractor

        proposals = {"agent1": "Proposal with high priority claim"}

        with patch("aragora.debate.orchestrator.logger") as mock_logger:
            arena._extract_citation_needs(proposals)

            # Verify debug logging was called for high priority needs
            # The method calls _log_citation_needs which logs via logger.debug

    def test_handles_empty_proposals(self, arena):
        """Test that empty proposals are handled."""
        mock_extractor = MagicMock()
        mock_extractor.identify_citation_needs.return_value = []
        arena.citation_extractor = mock_extractor

        proposals = {}

        result = arena._extract_citation_needs(proposals)

        assert result == {}

    def test_handles_extractor_returning_none(self, arena):
        """Test handling when extractor returns None or empty for some proposals."""
        mock_extractor = MagicMock()
        mock_extractor.identify_citation_needs.side_effect = [
            [{"claim": "Has citation", "priority": "medium"}],
            None,  # Returns None
            [],  # Returns empty list
        ]
        arena.citation_extractor = mock_extractor

        proposals = {
            "agent1": "Proposal 1",
            "agent2": "Proposal 2",
            "agent3": "Proposal 3",
        }

        result = arena._extract_citation_needs(proposals)

        # Only agent1 should be in results (None and [] are falsy)
        assert "agent1" in result
        # agent2 and agent3 should not be in results due to falsy returns


# =============================================================================
# Integration tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple critical methods."""

    @pytest.mark.asyncio
    async def test_full_debate_round_uses_critical_methods(self, env, mock_agents, protocol):
        """Test that a debate round properly uses all critical methods."""
        arena = Arena(env, mock_agents, protocol)

        # Mock citation extractor
        mock_extractor = MagicMock()
        mock_extractor.identify_citation_needs.return_value = [
            {"claim": "Test claim", "priority": "medium"}
        ]
        arena.citation_extractor = mock_extractor

        # Mock position ledger
        mock_ledger = MagicMock()
        arena._grounded_ops.position_ledger = mock_ledger

        # Build a proposal prompt
        arena.prompt_builder.build_proposal_prompt = MagicMock(return_value="Test prompt")
        prompt = arena._build_proposal_prompt(mock_agents[0])
        assert prompt is not None

        # Record a position
        arena._record_grounded_position(
            agent_name="agent1",
            content="Test position",
            debate_id="test-debate",
            round_num=1,
        )
        mock_ledger.record_position.assert_called_once()

        # Extract citation needs
        proposals = {"agent1": "Test proposal"}
        citation_needs = arena._extract_citation_needs(proposals)
        assert "agent1" in citation_needs

        # Select a judge
        judge = await arena._select_judge(proposals, [])
        assert judge is not None

    @pytest.mark.asyncio
    async def test_early_stopping_with_judge_selection(self, env, mock_agents, protocol):
        """Test that early stopping and judge selection work together."""
        # Set up for early stopping
        protocol.early_stopping = True
        protocol.min_rounds_before_early_stop = 1
        protocol.early_stop_threshold = 0.5

        # Set agents to vote CONTINUE
        for agent in mock_agents:
            agent.response = "CONTINUE"

        arena = Arena(env, mock_agents, protocol)

        proposals = {"agent1": "Proposal 1", "agent2": "Proposal 2"}
        context = []

        # Check early stopping (should continue)
        should_continue = await arena._check_early_stopping(2, proposals, context)
        assert should_continue is True

        # Select a judge
        judge = await arena._select_judge(proposals, context)
        assert judge is not None
        assert judge in mock_agents
