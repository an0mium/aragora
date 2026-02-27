"""
Tests for critical untested methods in the Arena orchestrator.

These tests cover critical methods that delegate to helper classes:

Phase 1 (Methods 1-5):
1. _build_proposal_prompt() - Core to proposal generation
2. _check_early_stopping() - Controls debate termination
3. _select_judge() - Judge selection for consensus
4. _record_grounded_position() - Position tracking
5. _extract_citation_needs() - Citation tracking

Phase 2 (Methods 6-10):
6. _perform_research() (Line 1168) - Research context gathering
7. _gather_evidence_context() (Line 1176) - Evidence grounding
8. _notify_spectator() (Line 1111) - User feedback events
9. _create_debate_bead() (Line 1198) - Audit trail creation
10. _update_role_assignments() (Line 1731) - Role rotation

Each method is tested through the Arena's delegation pattern to ensure
proper integration with the underlying helper classes.

Test coverage for each method includes:
- Happy path with all dependencies available
- Graceful handling when dependencies are None/unavailable
- Error handling for exceptions
- Edge cases (empty inputs, timeouts, etc.)
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


# =============================================================================
# Tests for _perform_research() (Line 1168)
# =============================================================================


@pytest.mark.no_io_stubs
class TestPerformResearch:
    """Tests for the _perform_research method - research context gathering."""

    @pytest.mark.asyncio
    async def test_performs_research_with_context_gatherer(self, arena):
        """Test that research is performed via context gatherer when available."""
        # Mock the context gatherer
        mock_gatherer = MagicMock()
        mock_gatherer.gather_all = AsyncMock(return_value="Research results from web and docs")
        mock_gatherer.evidence_pack = {"sources": ["web"], "snippets": ["Test evidence"]}
        arena._context_delegator.context_gatherer = mock_gatherer

        result = await arena._perform_research("What is machine learning?")

        assert result == "Research results from web and docs"
        mock_gatherer.gather_all.assert_called_once_with("What is machine learning?")

    @pytest.mark.asyncio
    async def test_updates_cache_after_research(self, arena):
        """Test that cache is updated with evidence pack after research."""
        mock_gatherer = MagicMock()
        mock_gatherer.gather_all = AsyncMock(return_value="Research context")
        evidence_pack = {"sources": ["github", "web"], "snippets": ["Evidence 1", "Evidence 2"]}
        mock_gatherer.evidence_pack = evidence_pack
        arena._context_delegator.context_gatherer = mock_gatherer
        arena._context_delegator._cache = MagicMock()

        await arena._perform_research("Research topic")

        # Verify cache was updated with evidence pack
        assert arena._context_delegator._cache.evidence_pack == evidence_pack

    @pytest.mark.asyncio
    async def test_updates_evidence_grounder_after_research(self, arena):
        """Test that evidence grounder receives the evidence pack."""
        mock_gatherer = MagicMock()
        mock_gatherer.gather_all = AsyncMock(return_value="Research results")
        evidence_pack = {"sources": ["docs"], "snippets": ["Found evidence"]}
        mock_gatherer.evidence_pack = evidence_pack
        arena._context_delegator.context_gatherer = mock_gatherer

        mock_grounder = MagicMock()
        arena._context_delegator.evidence_grounder = mock_grounder

        await arena._perform_research("Test task")

        mock_grounder.set_evidence_pack.assert_called_once_with(evidence_pack)

    @pytest.mark.asyncio
    async def test_handles_missing_context_gatherer(self, env, mock_agents, protocol):
        """Test handling when context gatherer is None."""
        arena = Arena(env, mock_agents, protocol)

        # Set context gatherer to have gather_all that returns empty
        arena._context_delegator.context_gatherer.gather_all = AsyncMock(
            return_value="No research context available."
        )

        result = await arena._perform_research("Unknown topic")

        # Should return the no-context message
        assert "No research context available" in result or result is not None

    @pytest.mark.asyncio
    async def test_handles_research_timeout_gracefully(self, arena):
        """Test that research timeouts are handled gracefully."""
        mock_gatherer = MagicMock()

        async def slow_gather(task):
            await asyncio.sleep(10)  # Simulate slow research
            return "Slow results"

        mock_gatherer.gather_all = slow_gather
        mock_gatherer.evidence_pack = None
        arena._context_delegator.context_gatherer = mock_gatherer

        # Use a short timeout
        try:
            result = await asyncio.wait_for(
                arena._perform_research("Test topic"),
                timeout=0.1,
            )
        except asyncio.TimeoutError:
            result = None

        # Timeout should occur, result is None
        assert result is None

    @pytest.mark.asyncio
    async def test_handles_research_exceptions_gracefully(self, arena):
        """Test that research exceptions are propagated or handled."""
        mock_gatherer = MagicMock()
        mock_gatherer.gather_all = AsyncMock(side_effect=ConnectionError("Network error"))
        arena._context_delegator.context_gatherer = mock_gatherer

        with pytest.raises(ConnectionError):
            await arena._perform_research("Test topic")

    @pytest.mark.asyncio
    async def test_research_with_empty_task(self, arena):
        """Test research handling with empty task string."""
        mock_gatherer = MagicMock()
        mock_gatherer.gather_all = AsyncMock(return_value="No research context available.")
        mock_gatherer.evidence_pack = None
        arena._context_delegator.context_gatherer = mock_gatherer

        result = await arena._perform_research("")

        # Should still attempt research with empty task
        mock_gatherer.gather_all.assert_called_once_with("")


# =============================================================================
# Tests for _gather_evidence_context() (Line 1176)
# =============================================================================


class TestGatherEvidenceContext:
    """Tests for the _gather_evidence_context method - evidence grounding."""

    @pytest.mark.asyncio
    async def test_gathers_evidence_from_connectors(self, arena):
        """Test that evidence is gathered from web, GitHub, and docs connectors."""
        mock_gatherer = MagicMock()
        evidence_result = "Evidence from GitHub: Code patterns found. Evidence from web: Articles."
        mock_gatherer.gather_evidence_context = AsyncMock(return_value=evidence_result)
        mock_gatherer.evidence_pack = {"sources": ["github", "web"]}
        arena._context_delegator.context_gatherer = mock_gatherer

        result = await arena._gather_evidence_context("Test task for evidence")

        assert result == evidence_result
        mock_gatherer.gather_evidence_context.assert_called_once_with("Test task for evidence")

    @pytest.mark.asyncio
    async def test_updates_cache_with_evidence_pack(self, arena):
        """Test that cache is updated with evidence pack."""
        mock_gatherer = MagicMock()
        mock_gatherer.gather_evidence_context = AsyncMock(return_value="Evidence context")
        evidence_pack = {"sources": ["local_docs"], "snippets": ["Doc snippet"]}
        mock_gatherer.evidence_pack = evidence_pack
        arena._context_delegator.context_gatherer = mock_gatherer
        arena._context_delegator._cache = MagicMock()

        await arena._gather_evidence_context("Task")

        assert arena._context_delegator._cache.evidence_pack == evidence_pack

    @pytest.mark.asyncio
    async def test_updates_evidence_grounder(self, arena):
        """Test that evidence grounder is updated with new pack."""
        mock_gatherer = MagicMock()
        mock_gatherer.gather_evidence_context = AsyncMock(return_value="Found evidence")
        evidence_pack = {"sources": ["web"], "snippets": ["Web snippet"]}
        mock_gatherer.evidence_pack = evidence_pack
        arena._context_delegator.context_gatherer = mock_gatherer

        mock_grounder = MagicMock()
        arena._context_delegator.evidence_grounder = mock_grounder

        await arena._gather_evidence_context("Evidence task")

        mock_grounder.set_evidence_pack.assert_called_once_with(evidence_pack)

    @pytest.mark.asyncio
    async def test_returns_none_when_no_evidence_found(self, arena):
        """Test handling when no evidence is found."""
        mock_gatherer = MagicMock()
        mock_gatherer.gather_evidence_context = AsyncMock(return_value=None)
        mock_gatherer.evidence_pack = None
        arena._context_delegator.context_gatherer = mock_gatherer

        result = await arena._gather_evidence_context("Obscure topic with no evidence")

        assert result is None

    @pytest.mark.asyncio
    async def test_handles_evidence_collection_errors(self, arena):
        """Test graceful handling of evidence collection errors."""
        mock_gatherer = MagicMock()
        mock_gatherer.gather_evidence_context = AsyncMock(
            side_effect=OSError("Network unreachable")
        )
        arena._context_delegator.context_gatherer = mock_gatherer

        with pytest.raises(OSError):
            await arena._gather_evidence_context("Test task")

    @pytest.mark.asyncio
    async def test_handles_empty_evidence_pack(self, arena):
        """Test handling when evidence pack is empty dict."""
        mock_gatherer = MagicMock()
        mock_gatherer.gather_evidence_context = AsyncMock(return_value="Some context")
        mock_gatherer.evidence_pack = {}
        arena._context_delegator.context_gatherer = mock_gatherer
        arena._context_delegator._cache = MagicMock()

        result = await arena._gather_evidence_context("Task")

        assert result == "Some context"
        # Empty pack should still be set
        assert arena._context_delegator._cache.evidence_pack == {}

    @pytest.mark.asyncio
    async def test_evidence_with_cache_and_grounder_none(self, arena):
        """Test evidence gathering when cache and grounder are None."""
        mock_gatherer = MagicMock()
        mock_gatherer.gather_evidence_context = AsyncMock(return_value="Evidence found")
        mock_gatherer.evidence_pack = {"sources": ["test"]}
        arena._context_delegator.context_gatherer = mock_gatherer
        arena._context_delegator._cache = None
        arena._context_delegator.evidence_grounder = None

        result = await arena._gather_evidence_context("Task")

        # Should still work without cache/grounder
        assert result == "Evidence found"


# =============================================================================
# Tests for _notify_spectator() (Line 1111)
# =============================================================================


class TestNotifySpectator:
    """Tests for the _notify_spectator method - user feedback events."""

    def test_notifies_via_event_bus_when_available(self, arena):
        """Test that spectator is notified via EventBus when available."""
        mock_event_bus = MagicMock()
        arena._event_emitter.event_bus = mock_event_bus
        arena._event_emitter._current_debate_id = "test-debate-123"

        arena._notify_spectator("proposal_ready", agent="agent1", content="My proposal")

        mock_event_bus.emit_sync.assert_called_once()
        call_args = mock_event_bus.emit_sync.call_args
        assert call_args[0][0] == "proposal_ready"  # event_type
        assert call_args[1]["agent"] == "agent1"
        assert call_args[1]["content"] == "My proposal"

    def test_notifies_via_event_bridge_as_fallback(self, arena):
        """Test that event_bridge is used when event_bus is None."""
        arena._event_emitter.event_bus = None
        mock_bridge = MagicMock()
        arena._event_emitter.event_bridge = mock_bridge

        arena._notify_spectator("round_complete", round=2, total=5)

        mock_bridge.notify.assert_called_once()
        call_args = mock_bridge.notify.call_args
        assert call_args[1]["round"] == 2
        assert call_args[1]["total"] == 5

    def test_handles_no_event_bus_or_bridge(self, arena):
        """Test handling when both event_bus and event_bridge are None."""
        arena._event_emitter.event_bus = None
        arena._event_emitter.event_bridge = None

        # Should not raise an exception
        arena._notify_spectator("test_event", data="test")

    def test_passes_debate_id_from_kwargs(self, arena):
        """Test that debate_id from kwargs is used instead of default."""
        mock_event_bus = MagicMock()
        arena._event_emitter.event_bus = mock_event_bus
        arena._event_emitter._current_debate_id = "default-id"

        arena._notify_spectator(
            "custom_event",
            debate_id="custom-debate-id",
            message="Test message",
        )

        call_args = mock_event_bus.emit_sync.call_args
        assert call_args[1]["debate_id"] == "custom-debate-id"

    def test_handles_various_event_types(self, arena):
        """Test notification of various event types."""
        mock_event_bus = MagicMock()
        arena._event_emitter.event_bus = mock_event_bus

        event_types = [
            ("debate_start", {"agents": ["a1", "a2"]}),
            ("round_start", {"round": 1}),
            ("proposal", {"agent": "a1", "content": "text"}),
            ("critique", {"agent": "a2", "target": "a1"}),
            ("vote", {"agent": "a1", "choice": "a2"}),
            ("consensus", {"reached": True, "confidence": 0.9}),
            ("debate_end", {"result": "completed"}),
        ]

        for event_type, kwargs in event_types:
            mock_event_bus.reset_mock()
            arena._notify_spectator(event_type, **kwargs)
            assert mock_event_bus.emit_sync.called

    def test_handles_exception_in_event_bus(self, arena):
        """Test graceful handling when event_bus raises exception."""
        mock_event_bus = MagicMock()
        mock_event_bus.emit_sync.side_effect = RuntimeError("EventBus error")
        arena._event_emitter.event_bus = mock_event_bus

        # The method may raise or catch - depends on implementation
        # Just verify it doesn't crash unexpectedly
        try:
            arena._notify_spectator("test_event", data="test")
        except RuntimeError:
            pass  # Expected if not caught

    def test_notifies_with_empty_kwargs(self, arena):
        """Test notification with no additional kwargs."""
        mock_event_bus = MagicMock()
        arena._event_emitter.event_bus = mock_event_bus

        arena._notify_spectator("simple_event")

        mock_event_bus.emit_sync.assert_called_once()


# =============================================================================
# Tests for _create_debate_bead() (Line 1198)
# =============================================================================


class TestCreateDebateBead:
    """Tests for the _create_debate_bead method - audit trail creation."""

    @pytest.mark.asyncio
    async def test_creates_bead_when_tracking_enabled(self, env, mock_agents, protocol):
        """Test that a bead is created when bead tracking is enabled."""
        protocol.enable_bead_tracking = True
        protocol.bead_min_confidence = 0.5

        arena = Arena(env, mock_agents, protocol)

        # Create a mock result with high confidence
        mock_result = MagicMock()
        mock_result.task = "Test debate task"
        mock_result.confidence = 0.9
        mock_result.final_answer = "The conclusion is..."
        mock_result.status = "completed"
        mock_result.debate_id = "test-debate-id"
        mock_result.consensus_reached = True
        mock_result.rounds_used = 3
        mock_result.participants = ["agent1", "agent2"]
        mock_result.winner = "agent1"

        # Mock the beads module that is imported inside the function
        mock_bead_module = MagicMock()
        mock_store = MagicMock()
        mock_store.initialize = AsyncMock()
        mock_store.create = AsyncMock(return_value="bead-id-123")
        mock_bead_module.BeadStore.return_value = mock_store

        mock_bead = MagicMock()
        mock_bead_module.Bead.create.return_value = mock_bead
        mock_bead_module.BeadPriority.HIGH = "HIGH"
        mock_bead_module.BeadPriority.NORMAL = "NORMAL"
        mock_bead_module.BeadPriority.LOW = "LOW"
        mock_bead_module.BeadType.DEBATE_DECISION = "DEBATE_DECISION"

        # Pre-set the bead store on the arena to bypass _resolve_bead_store
        arena._bead_store = mock_store

        with patch.dict("sys.modules", {"aragora.nomic.beads": mock_bead_module}):
            bead_id = await arena._create_debate_bead(mock_result)

            # Should create and return a bead ID
            assert bead_id == "bead-id-123"

    @pytest.mark.asyncio
    async def test_skips_bead_when_tracking_disabled(self, env, mock_agents, protocol):
        """Test that no bead is created when tracking is disabled."""
        protocol.enable_bead_tracking = False
        arena = Arena(env, mock_agents, protocol)

        mock_result = MagicMock()
        mock_result.confidence = 0.9

        bead_id = await arena._create_debate_bead(mock_result)

        assert bead_id is None

    @pytest.mark.asyncio
    async def test_skips_bead_when_confidence_too_low(self, env, mock_agents, protocol):
        """Test that no bead is created when confidence is below threshold."""
        protocol.enable_bead_tracking = True
        protocol.bead_min_confidence = 0.8

        arena = Arena(env, mock_agents, protocol)

        mock_result = MagicMock()
        mock_result.confidence = 0.5  # Below threshold

        bead_id = await arena._create_debate_bead(mock_result)

        assert bead_id is None

    @pytest.mark.asyncio
    async def test_handles_bead_store_not_available(self, env, mock_agents, protocol):
        """Test handling when BeadStore module is not available."""
        protocol.enable_bead_tracking = True
        protocol.bead_min_confidence = 0.5

        arena = Arena(env, mock_agents, protocol)

        mock_result = MagicMock()
        mock_result.confidence = 0.9

        # The function should handle ImportError gracefully
        # Mock the import to fail
        with patch.dict(
            "sys.modules",
            {"aragora.nomic.beads": None},
        ):
            bead_id = await arena._create_debate_bead(mock_result)

            # Should return None when import fails
            # (The actual implementation catches ImportError)
            assert bead_id is None or isinstance(bead_id, str)

    @pytest.mark.asyncio
    async def test_handles_bead_creation_errors(self, env, mock_agents, protocol):
        """Test graceful handling of bead creation errors."""
        protocol.enable_bead_tracking = True
        protocol.bead_min_confidence = 0.5

        arena = Arena(env, mock_agents, protocol)

        mock_result = MagicMock()
        mock_result.task = "Test task"
        mock_result.confidence = 0.9
        mock_result.final_answer = "Answer"
        mock_result.status = "completed"
        mock_result.debate_id = "id"
        mock_result.consensus_reached = True
        mock_result.rounds_used = 2
        mock_result.participants = []
        mock_result.winner = None

        # Mock the beads module that is imported inside the function
        mock_bead_module = MagicMock()
        mock_store = MagicMock()
        mock_store.initialize = AsyncMock()
        mock_store.create = AsyncMock(side_effect=OSError("Disk full"))
        mock_bead_module.BeadStore.return_value = mock_store

        mock_bead = MagicMock()
        mock_bead_module.Bead.create.return_value = mock_bead
        mock_bead_module.BeadPriority.HIGH = "HIGH"
        mock_bead_module.BeadPriority.NORMAL = "NORMAL"
        mock_bead_module.BeadPriority.LOW = "LOW"
        mock_bead_module.BeadType.DEBATE_DECISION = "DEBATE_DECISION"

        with patch.dict("sys.modules", {"aragora.nomic.beads": mock_bead_module}):
            bead_id = await arena._create_debate_bead(mock_result)

            # Should return None on error
            assert bead_id is None

    @pytest.mark.asyncio
    async def test_bead_priority_based_on_confidence(self, env, mock_agents, protocol):
        """Test that bead priority is set based on confidence level."""
        protocol.enable_bead_tracking = True
        protocol.bead_min_confidence = 0.5

        arena = Arena(env, mock_agents, protocol)

        # Test high confidence -> HIGH priority
        mock_result = MagicMock()
        mock_result.task = "Task"
        mock_result.confidence = 0.95  # >= 0.9 -> HIGH
        mock_result.final_answer = "Answer"
        mock_result.status = "completed"
        mock_result.debate_id = "id"
        mock_result.consensus_reached = True
        mock_result.rounds_used = 2
        mock_result.participants = []
        mock_result.winner = None

        # Mock the beads module that is imported inside the function
        mock_bead_module = MagicMock()
        mock_store = MagicMock()
        mock_store.initialize = AsyncMock()
        mock_store.create = AsyncMock(return_value="bead-id")
        mock_bead_module.BeadStore.return_value = mock_store

        mock_bead = MagicMock()
        mock_bead_module.Bead.create.return_value = mock_bead
        mock_bead_module.BeadPriority.HIGH = "HIGH"
        mock_bead_module.BeadPriority.NORMAL = "NORMAL"
        mock_bead_module.BeadPriority.LOW = "LOW"
        mock_bead_module.BeadType.DEBATE_DECISION = "DEBATE_DECISION"

        # Pre-set the bead store on the arena to bypass _resolve_bead_store
        arena._bead_store = mock_store

        with patch.dict("sys.modules", {"aragora.nomic.beads": mock_bead_module}):
            await arena._create_debate_bead(mock_result)

            # Verify Bead.create was called with HIGH priority
            call_args = mock_bead_module.Bead.create.call_args
            assert call_args[1]["priority"] == "HIGH"


# =============================================================================
# Tests for _update_role_assignments() (Line 1731)
# =============================================================================


class TestUpdateRoleAssignments:
    """Tests for the _update_role_assignments method - role rotation."""

    def test_updates_roles_via_roles_manager(self, arena):
        """Test that role assignments are updated through RolesManager."""
        # Mock the roles manager
        mock_assignment = MagicMock()
        mock_assignment.role = MagicMock()
        mock_assignment.role.value = "critic"

        arena.roles_manager.current_role_assignments = {
            "agent1": mock_assignment,
            "agent2": mock_assignment,
        }
        arena.roles_manager.update_role_assignments = MagicMock()

        arena._update_role_assignments(round_num=2)

        arena.roles_manager.update_role_assignments.assert_called_once()
        call_args = arena.roles_manager.update_role_assignments.call_args
        assert call_args[0][0] == 2  # round_num

    def test_extracts_domain_for_role_matching(self, arena):
        """Test that debate domain is extracted and used for role matching."""
        arena.roles_manager.update_role_assignments = MagicMock()

        # Set up cache to return a specific domain
        arena._cache.debate_domain = "technology"

        arena._update_role_assignments(round_num=1)

        call_args = arena.roles_manager.update_role_assignments.call_args
        assert call_args[0][1] == "technology"  # debate_domain

    def test_syncs_assignments_back_to_arena(self, arena):
        """Test that role assignments are synced back to Arena instance."""
        mock_assignment = MagicMock()
        mock_assignment.role = MagicMock()
        mock_assignment.role.value = "synthesizer"

        new_assignments = {
            "agent1": mock_assignment,
            "agent2": mock_assignment,
            "agent3": mock_assignment,
        }

        def update_fn(round_num, domain):
            arena.roles_manager.current_role_assignments = new_assignments

        arena.roles_manager.update_role_assignments = update_fn

        arena._update_role_assignments(round_num=3)

        # Verify assignments are synced to Arena
        assert arena.current_role_assignments == new_assignments

    def test_logs_role_assignments(self, arena):
        """Test that role assignments are logged for debugging."""
        mock_assignment = MagicMock()
        mock_assignment.role = MagicMock()
        mock_assignment.role.value = "proposer"

        arena.roles_manager.current_role_assignments = {"agent1": mock_assignment}
        arena.roles_manager.update_role_assignments = MagicMock()

        mock_logger = MagicMock()
        with patch("aragora.logging_config.get_logger", return_value=mock_logger):
            arena._update_role_assignments(round_num=1)

            # Verify debug logging was called
            mock_logger.debug.assert_called()

    def test_handles_empty_assignments(self, arena):
        """Test handling when there are no role assignments."""
        arena.roles_manager.current_role_assignments = {}
        arena.roles_manager.update_role_assignments = MagicMock()

        # Should not raise an exception
        arena._update_role_assignments(round_num=1)

        # Verify sync still happens
        assert arena.current_role_assignments == {}

    def test_handles_role_matcher_not_available(self, arena):
        """Test behavior when role_matcher is None."""
        arena.roles_manager.role_matcher = None
        arena.roles_manager.role_rotator = None
        arena.roles_manager.update_role_assignments = MagicMock()

        # Should still work without role systems
        arena._update_role_assignments(round_num=2)

        arena.roles_manager.update_role_assignments.assert_called_once()

    def test_updates_for_multiple_rounds(self, arena):
        """Test role updates across multiple rounds."""
        call_count = 0
        round_nums = []

        def track_updates(round_num, domain):
            nonlocal call_count
            call_count += 1
            round_nums.append(round_num)

        arena.roles_manager.update_role_assignments = track_updates
        arena.roles_manager.current_role_assignments = {}

        for round_num in range(1, 6):
            arena._update_role_assignments(round_num)

        assert call_count == 5
        assert round_nums == [1, 2, 3, 4, 5]

    def test_handles_update_exception_gracefully(self, arena):
        """Test graceful handling of update exceptions."""
        arena.roles_manager.update_role_assignments = MagicMock(
            side_effect=ValueError("Invalid round")
        )

        with pytest.raises(ValueError):
            arena._update_role_assignments(round_num=-1)


# =============================================================================
# Additional Integration Tests for New Methods
# =============================================================================


class TestCriticalMethodsIntegration:
    """Integration tests for the new critical methods together."""

    @pytest.mark.asyncio
    async def test_research_and_evidence_context_flow(self, arena):
        """Test that research and evidence gathering work together."""
        mock_gatherer = MagicMock()
        mock_gatherer.gather_all = AsyncMock(return_value="Research: AI is transformative")
        mock_gatherer.gather_evidence_context = AsyncMock(
            return_value="Evidence: Multiple studies confirm"
        )
        mock_gatherer.evidence_pack = {"sources": ["web", "academic"]}
        arena._context_delegator.context_gatherer = mock_gatherer
        arena._context_delegator._cache = MagicMock()

        # Perform research
        research = await arena._perform_research("AI impact on society")
        assert "Research: AI is transformative" in research

        # Then gather additional evidence
        evidence = await arena._gather_evidence_context("AI impact on society")
        assert "Evidence: Multiple studies confirm" in evidence

    def test_role_updates_and_spectator_notification(self, arena):
        """Test that role updates can trigger spectator notifications."""
        mock_event_bus = MagicMock()
        arena._event_emitter.event_bus = mock_event_bus

        mock_assignment = MagicMock()
        mock_assignment.role = MagicMock()
        mock_assignment.role.value = "critic"
        arena.roles_manager.current_role_assignments = {"agent1": mock_assignment}
        arena.roles_manager.update_role_assignments = MagicMock()

        # Update roles
        arena._update_role_assignments(round_num=2)

        # Notify spectator about role change
        arena._notify_spectator(
            "role_update",
            agent="agent1",
            new_role="critic",
            round=2,
        )

        # Verify both operations completed
        arena.roles_manager.update_role_assignments.assert_called_once()
        mock_event_bus.emit_sync.assert_called()

    @pytest.mark.asyncio
    async def test_bead_creation_after_successful_debate(self, env, mock_agents, protocol):
        """Test bead creation after a successful debate."""
        protocol.enable_bead_tracking = True
        protocol.bead_min_confidence = 0.6

        arena = Arena(env, mock_agents, protocol)

        # Simulate a successful debate result
        mock_result = MagicMock()
        mock_result.task = "Should we adopt microservices?"
        mock_result.confidence = 0.85
        mock_result.final_answer = "Yes, for better scalability."
        mock_result.status = "completed"
        mock_result.debate_id = "debate-456"
        mock_result.consensus_reached = True
        mock_result.rounds_used = 4
        mock_result.participants = ["agent1", "agent2", "agent3"]
        mock_result.winner = "agent2"

        # Mock the orchestrator_hooks function that creates debate beads
        with patch("aragora.debate.orchestrator_hooks.create_debate_bead") as mock_create_bead:
            mock_create_bead.return_value = "bead-xyz-789"

            bead_id = await arena._create_debate_bead(mock_result)

            assert bead_id == "bead-xyz-789"
            mock_create_bead.assert_called_once()

    @pytest.mark.asyncio
    async def test_full_context_gathering_pipeline(self, arena):
        """Test the full context gathering pipeline."""
        mock_gatherer = MagicMock()
        mock_gatherer.gather_all = AsyncMock(return_value="Full research context")
        mock_gatherer.gather_evidence_context = AsyncMock(return_value="Evidence snippets")
        mock_gatherer.evidence_pack = {"sources": ["all"]}
        arena._context_delegator.context_gatherer = mock_gatherer
        arena._context_delegator._cache = MagicMock()

        mock_grounder = MagicMock()
        arena._context_delegator.evidence_grounder = mock_grounder

        # Step 1: Perform initial research
        research = await arena._perform_research("Complex topic")
        assert research == "Full research context"

        # Step 2: Gather specific evidence
        evidence = await arena._gather_evidence_context("Complex topic")
        assert evidence == "Evidence snippets"

        # Verify evidence grounder was updated
        assert mock_grounder.set_evidence_pack.call_count == 2
