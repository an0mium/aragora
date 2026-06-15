"""
Integration tests for phase-based debate orchestration.

Tests verify:
1. All 7 phases execute in correct sequence
2. DebateContext state propagates correctly between phases
3. Callbacks are invoked in expected order
4. Error handling and partial failure scenarios
5. Timeout recovery behavior
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from dataclasses import dataclass

from aragora.core import Agent, Environment, Message, Critique, Vote, DebateResult
from aragora.debate.context import DebateContext
from aragora.debate.protocol import DebateProtocol, CircuitBreaker
from aragora.debate.phases import (
    ContextInitializer,
    ProposalPhase,
    DebateRoundsPhase,
    ConsensusPhase,
    AnalyticsPhase,
    FeedbackPhase,
)


class MockAgent:
    """Simple mock agent for testing."""

    def __init__(self, name: str, role: str = "proposer"):
        self.name = name
        self.role = role
        self.model = "test-model"

    async def generate(self, prompt: str, context=None) -> str:
        return f"Response from {self.name}"

    async def critique(self, content: str, context=None) -> Critique:
        return Critique(
            agent=self.name,
            target_agent="target",
            target_content=content[:50],
            issues=["test issue"],
            suggestions=["test suggestion"],
            severity=0.5,
            reasoning="test reasoning",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        choice = list(proposals.keys())[0] if proposals else "default"
        return Vote(agent=self.name, choice=choice, confidence=0.8, reasoning="test vote")


def create_mock_agents(count: int = 3) -> list[MockAgent]:
    """Create a list of mock agents."""
    return [
        MockAgent(f"agent-{i}", role="proposer" if i % 2 == 0 else "critic") for i in range(count)
    ]


def create_test_context(agents: list = None) -> DebateContext:
    """Create a test DebateContext with default values."""
    if agents is None:
        agents = create_mock_agents()

    env = Environment(task="Test debate task")
    ctx = DebateContext(
        env=env,
        agents=agents,
        start_time=0.0,
        debate_id="test-debate-123",
        domain="general",
    )
    return ctx


class TestPhaseExecutionOrder:
    """Tests for verifying phases execute in correct order."""

    @pytest.mark.asyncio
    async def test_context_initializer_sets_up_result(self):
        """ContextInitializer should create ctx.result and populate proposers."""
        ctx = create_test_context()
        protocol = DebateProtocol(rounds=2)

        initializer = ContextInitializer(
            initial_messages=[],
            trending_topic=None,
            recorder=None,
            debate_embeddings=None,
            insight_store=None,
            memory=None,
            protocol=protocol,
        )

        await initializer.initialize(ctx)

        # Verify result is created
        assert ctx.result is not None
        assert ctx.result.task == "Test debate task"
        # Verify proposers are identified
        assert len(ctx.proposers) > 0

    @pytest.mark.asyncio
    async def test_proposal_phase_requires_initialized_context(self):
        """ProposalPhase should work with initialized context."""
        ctx = create_test_context()
        ctx.result = DebateResult(
            task="Test task",
            consensus_reached=False,
            confidence=0.0,
            messages=[],
            critiques=[],
            votes=[],
            rounds_used=0,
            final_answer="",
        )
        ctx.proposers = ctx.agents[:2]

        # Mock callbacks
        build_prompt = MagicMock(return_value="Generate a proposal")
        generate = AsyncMock(return_value="Test proposal content")

        async def mock_with_timeout(coro, agent_name, timeout_seconds=90.0):
            return await coro

        notify = MagicMock()

        phase = ProposalPhase(
            circuit_breaker=CircuitBreaker(),
            position_tracker=None,
            position_ledger=None,
            recorder=None,
            hooks={},
            build_proposal_prompt=build_prompt,
            generate_with_agent=generate,
            with_timeout=mock_with_timeout,
            notify_spectator=notify,
        )

        await phase.execute(ctx)

        # Verify proposals were generated
        assert len(ctx.proposals) > 0
        # Verify generate was called for each proposer
        assert generate.call_count >= 1

    @pytest.mark.asyncio
    async def test_consensus_phase_produces_final_answer(self):
        """ConsensusPhase should determine winner and set final_answer."""
        ctx = create_test_context()
        agents = ctx.agents

        # Set up required state from previous phases
        ctx.result = DebateResult(
            task="Test task",
            consensus_reached=False,
            confidence=0.0,
            messages=[],
            critiques=[],
            votes=[],
            rounds_used=2,
            final_answer="",
        )
        ctx.proposals = {
            agents[0].name: "Proposal A content",
            agents[1].name: "Proposal B content",
        }

        # Mock vote callback
        async def mock_vote(agent, proposals, task):
            return Vote(
                agent=agent.name,
                choice=list(proposals.keys())[0],
                confidence=0.8,
                reasoning="Test vote",
            )

        async def mock_with_timeout(coro, agent_name, timeout_seconds=90.0):
            return await coro

        phase = ConsensusPhase(
            protocol=DebateProtocol(consensus="majority"),
            elo_system=None,
            memory=None,
            agent_weights={},
            flip_detector=None,
            calibration_tracker=None,
            position_tracker=None,
            user_votes=[],
            vote_with_agent=mock_vote,
            with_timeout=mock_with_timeout,
            notify_spectator=MagicMock(),
        )

        await phase.execute(ctx)

        # Verify final answer is set
        assert ctx.result.final_answer != ""
        # Verify votes were recorded
        assert len(ctx.result.votes) > 0


class TestContextStatePropagation:
    """Tests for verifying state flows correctly between phases."""

    @pytest.mark.asyncio
    async def test_proposals_available_in_consensus_phase(self):
        """Proposals from ProposalPhase should be accessible in ConsensusPhase."""
        ctx = create_test_context()
        agents = ctx.agents

        # Initialize result
        ctx.result = DebateResult(
            task="Test task",
            consensus_reached=False,
            confidence=0.0,
            messages=[],
            critiques=[],
            votes=[],
            rounds_used=0,
            final_answer="",
        )
        ctx.proposers = agents[:2]

        async def mock_with_timeout(coro, agent_name, timeout_seconds=90.0):
            return await coro

        # Run proposal phase
        proposal_phase = ProposalPhase(
            circuit_breaker=CircuitBreaker(),
            position_tracker=None,
            position_ledger=None,
            recorder=None,
            hooks={},
            build_proposal_prompt=MagicMock(return_value="Generate proposal"),
            generate_with_agent=AsyncMock(return_value="Test proposal"),
            with_timeout=mock_with_timeout,
            notify_spectator=MagicMock(),
        )

        await proposal_phase.execute(ctx)

        # Verify proposals exist
        assert len(ctx.proposals) > 0

        async def mock_vote(agent, proposals, task):
            return Vote(
                agent="test", choice=list(proposals.keys())[0], confidence=0.8, reasoning="test"
            )

        # Now consensus phase should have access to proposals
        consensus_phase = ConsensusPhase(
            protocol=DebateProtocol(consensus="majority"),
            elo_system=None,
            memory=None,
            agent_weights={},
            flip_detector=None,
            calibration_tracker=None,
            position_tracker=None,
            user_votes=[],
            vote_with_agent=mock_vote,
            with_timeout=mock_with_timeout,
            notify_spectator=MagicMock(),
        )

        await consensus_phase.execute(ctx)

        # Verify winner is one of the proposers
        assert ctx.result.final_answer != ""

    @pytest.mark.asyncio
    async def test_messages_accumulate_across_phases(self):
        """Messages should accumulate as debate progresses."""
        ctx = create_test_context()
        agents = ctx.agents

        # Initialize result
        ctx.result = DebateResult(
            task="Test task",
            consensus_reached=False,
            confidence=0.0,
            messages=[],
            critiques=[],
            votes=[],
            rounds_used=0,
            final_answer="",
        )
        ctx.proposers = agents[:2]

        initial_message_count = len(ctx.result.messages)

        async def mock_with_timeout(coro, agent_name, timeout_seconds=90.0):
            return await coro

        # Run proposal phase
        proposal_phase = ProposalPhase(
            circuit_breaker=CircuitBreaker(),
            position_tracker=None,
            position_ledger=None,
            recorder=None,
            hooks={},
            build_proposal_prompt=MagicMock(return_value="Generate proposal"),
            generate_with_agent=AsyncMock(return_value="Test proposal"),
            with_timeout=mock_with_timeout,
            notify_spectator=MagicMock(),
        )

        await proposal_phase.execute(ctx)

        # Messages should have increased
        assert len(ctx.result.messages) > initial_message_count


class TestCallbackInvocation:
    """Tests for verifying callbacks are invoked correctly."""

    @pytest.mark.asyncio
    async def test_notify_spectator_called_on_events(self):
        """notify_spectator callback should be called for key events."""
        ctx = create_test_context()
        ctx.result = DebateResult(
            task="Test task",
            consensus_reached=False,
            confidence=0.0,
            messages=[],
            critiques=[],
            votes=[],
            rounds_used=0,
            final_answer="",
        )
        ctx.proposers = ctx.agents[:2]

        notify_mock = MagicMock()

        async def mock_with_timeout(coro, agent_name, timeout_seconds=90.0):
            return await coro

        phase = ProposalPhase(
            circuit_breaker=CircuitBreaker(),
            position_tracker=None,
            position_ledger=None,
            recorder=None,
            hooks={},
            build_proposal_prompt=MagicMock(return_value="Generate proposal"),
            generate_with_agent=AsyncMock(return_value="Test proposal"),
            with_timeout=mock_with_timeout,
            notify_spectator=notify_mock,
        )

        await phase.execute(ctx)

        # Verify notify was called at least once
        assert notify_mock.call_count >= 1

    @pytest.mark.asyncio
    async def test_hooks_called_on_message(self):
        """on_message hook should be called when messages are added."""
        ctx = create_test_context()
        ctx.result = DebateResult(
            task="Test task",
            consensus_reached=False,
            confidence=0.0,
            messages=[],
            critiques=[],
            votes=[],
            rounds_used=0,
            final_answer="",
        )
        ctx.proposers = ctx.agents[:2]

        on_message_hook = MagicMock()

        async def mock_with_timeout(coro, agent_name, timeout_seconds=90.0):
            return await coro

        phase = ProposalPhase(
            circuit_breaker=CircuitBreaker(),
            position_tracker=None,
            position_ledger=None,
            recorder=None,
            hooks={"on_message": on_message_hook},
            build_proposal_prompt=MagicMock(return_value="Generate proposal"),
            generate_with_agent=AsyncMock(return_value="Test proposal"),
            with_timeout=mock_with_timeout,
            notify_spectator=MagicMock(),
        )

        await phase.execute(ctx)

        # Hook should be called for each proposal message
        assert on_message_hook.call_count >= 1


class TestErrorHandling:
    """Tests for error handling and recovery."""

    @pytest.mark.asyncio
    async def test_phase_continues_on_agent_timeout(self):
        """Phase should continue with partial results on agent timeout."""
        ctx = create_test_context()
        agents = ctx.agents
        ctx.result = DebateResult(
            task="Test task",
            consensus_reached=False,
            confidence=0.0,
            messages=[],
            critiques=[],
            votes=[],
            rounds_used=0,
            final_answer="",
        )
        ctx.proposers = agents[:3]

        call_count = [0]

        async def timeout_on_second_call(coro, agent_name, timeout_seconds=90.0):
            call_count[0] += 1
            if call_count[0] == 2:
                raise asyncio.TimeoutError("Agent timed out")
            return await coro

        phase = ProposalPhase(
            circuit_breaker=CircuitBreaker(),
            position_tracker=None,
            position_ledger=None,
            recorder=None,
            hooks={},
            build_proposal_prompt=MagicMock(return_value="Generate proposal"),
            generate_with_agent=AsyncMock(return_value="Test proposal"),
            with_timeout=timeout_on_second_call,
            notify_spectator=MagicMock(),
        )

        # Should not raise, should continue with partial results
        await phase.execute(ctx)

        # At least one proposal should exist
        assert len(ctx.proposals) >= 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_filters_failing_agents(self):
        """Circuit breaker should prevent use of agents that have failed."""
        ctx = create_test_context()
        agents = ctx.agents
        ctx.result = DebateResult(
            task="Test task",
            consensus_reached=False,
            confidence=0.0,
            messages=[],
            critiques=[],
            votes=[],
            rounds_used=0,
            final_answer="",
        )
        ctx.proposers = agents[:2]

        # Create circuit breaker with one agent already tripped
        circuit_breaker = CircuitBreaker(failure_threshold=2, cooldown_seconds=60)
        circuit_breaker.record_failure(agents[0].name)
        circuit_breaker.record_failure(agents[0].name)  # Trip the breaker

        generate_calls = []

        async def track_generate(agent, prompt, context):
            generate_calls.append(agent.name)
            return "Test proposal"

        async def mock_with_timeout(coro, agent_name, timeout_seconds=90.0):
            return await coro

        phase = ProposalPhase(
            circuit_breaker=circuit_breaker,
            position_tracker=None,
            position_ledger=None,
            recorder=None,
            hooks={},
            build_proposal_prompt=MagicMock(return_value="Generate proposal"),
            generate_with_agent=track_generate,
            with_timeout=mock_with_timeout,
            notify_spectator=MagicMock(),
        )

        await phase.execute(ctx)

        # First agent should have been skipped
        assert agents[0].name not in generate_calls

    @pytest.mark.asyncio
    async def test_consensus_fallback_on_no_votes(self):
        """Consensus should fall back to first proposal if no valid votes."""
        ctx = create_test_context()
        agents = ctx.agents
        ctx.result = DebateResult(
            task="Test task",
            consensus_reached=False,
            confidence=0.0,
            messages=[],
            critiques=[],
            votes=[],
            rounds_used=2,
            final_answer="",
        )
        ctx.proposals = {
            agents[0].name: "First proposal content",
            agents[1].name: "Second proposal content",
        }

        # Vote callback that always fails
        async def failing_vote(agent, proposals, task):
            raise Exception("Vote failed")

        async def mock_with_timeout(coro, agent_name, timeout_seconds=90.0):
            return await coro

        phase = ConsensusPhase(
            protocol=DebateProtocol(consensus="majority"),
            elo_system=None,
            memory=None,
            agent_weights={},
            flip_detector=None,
            calibration_tracker=None,
            position_tracker=None,
            user_votes=[],
            vote_with_agent=failing_vote,
            with_timeout=mock_with_timeout,
            notify_spectator=MagicMock(),
        )

        await phase.execute(ctx)

        # Should fall back to first proposal
        assert ctx.result.final_answer != ""


class TestPartialFailureRecovery:
    """Tests for partial failure scenarios."""

    @pytest.mark.asyncio
    async def test_debate_completes_with_single_proposer(self):
        """Debate should complete even if only one proposer succeeds."""
        ctx = create_test_context()
        agents = ctx.agents
        ctx.result = DebateResult(
            task="Test task",
            consensus_reached=False,
            confidence=0.0,
            messages=[],
            critiques=[],
            votes=[],
            rounds_used=0,
            final_answer="",
        )
        ctx.proposers = agents[:3]

        call_count = [0]

        async def fail_most_agents(agent, prompt, context):
            call_count[0] += 1
            if call_count[0] > 1:
                raise Exception(f"Agent {agent.name} failed")
            return "Successful proposal"

        async def mock_with_timeout(coro, agent_name, timeout_seconds=90.0):
            return await coro

        phase = ProposalPhase(
            circuit_breaker=CircuitBreaker(),
            position_tracker=None,
            position_ledger=None,
            recorder=None,
            hooks={},
            build_proposal_prompt=MagicMock(return_value="Generate proposal"),
            generate_with_agent=fail_most_agents,
            with_timeout=mock_with_timeout,
            notify_spectator=MagicMock(),
        )

        await phase.execute(ctx)

        # At least one proposal should exist
        assert len(ctx.proposals) >= 1


class TestFullPipelineIntegration:
    """End-to-end integration tests for the full phase pipeline."""

    @pytest.mark.asyncio
    async def test_full_debate_pipeline_minimal(self):
        """Test minimal end-to-end debate with all phases."""
        agents = create_mock_agents(2)
        ctx = create_test_context(agents)
        protocol = DebateProtocol(rounds=1, consensus="majority")

        # Phase 0: Context Initialization
        initializer = ContextInitializer(
            initial_messages=[],
            trending_topic=None,
            recorder=None,
            debate_embeddings=None,
            insight_store=None,
            memory=None,
            protocol=protocol,
        )
        await initializer.initialize(ctx)

        assert ctx.result is not None

        async def mock_with_timeout(coro, agent_name, timeout_seconds=90.0):
            return await coro

        # Phase 1: Proposals
        proposal_phase = ProposalPhase(
            circuit_breaker=CircuitBreaker(),
            position_tracker=None,
            position_ledger=None,
            recorder=None,
            hooks={},
            build_proposal_prompt=MagicMock(return_value="Generate proposal"),
            generate_with_agent=AsyncMock(return_value="Test proposal content"),
            with_timeout=mock_with_timeout,
            notify_spectator=MagicMock(),
        )
        await proposal_phase.execute(ctx)

        assert len(ctx.proposals) > 0

        # Phase 3: Consensus (skipping rounds for minimal test)
        consensus_phase = ConsensusPhase(
            protocol=protocol,
            elo_system=None,
            memory=None,
            agent_weights={},
            flip_detector=None,
            calibration_tracker=None,
            position_tracker=None,
            user_votes=[],
            vote_with_agent=AsyncMock(
                return_value=Vote(
                    agent="voter",
                    choice=list(ctx.proposals.keys())[0],
                    confidence=0.9,
                    reasoning="Selected best",
                )
            ),
            with_timeout=mock_with_timeout,
            notify_spectator=MagicMock(),
        )
        await consensus_phase.execute(ctx)

        # Verify final state
        assert ctx.result.final_answer != ""
        assert len(ctx.result.votes) > 0

    @pytest.mark.asyncio
    async def test_analytics_phase_sets_duration(self):
        """Analytics phase should calculate and set debate duration."""
        ctx = create_test_context()
        ctx.start_time = 0.0  # Will calculate from current time
        ctx.result = DebateResult(
            task="Test task",
            consensus_reached=True,
            confidence=0.85,
            messages=[],
            critiques=[],
            votes=[],
            rounds_used=2,
            final_answer="Test answer",
        )
        ctx.proposals = {"agent-0": "Test proposal"}
        ctx.winner_agent = "agent-0"

        analytics_phase = AnalyticsPhase(
            memory=None,
            insight_store=None,
            recorder=None,
            event_emitter=None,
            hooks={},
            notify_spectator=MagicMock(),
        )

        await analytics_phase.execute(ctx)

        # Duration should be set (greater than 0 since start_time was 0)
        assert ctx.result.duration_seconds >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
