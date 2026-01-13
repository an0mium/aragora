"""
Tests for aragora.debate.phases.debate_rounds module.

Tests DebateRoundsPhase class and critique/revision loop.
"""

import pytest
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock, AsyncMock

from aragora.debate.context import DebateContext
from aragora.debate.phases.debate_rounds import DebateRoundsPhase


# ============================================================================
# Mock Classes
# ============================================================================


@dataclass
class MockEnvironment:
    """Mock environment for testing."""

    task: str = "Test task"
    context: str = ""


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str = "test_agent"
    role: str = "proposer"
    stance: Optional[str] = None


@dataclass
class MockCritique:
    """Mock critique for testing."""

    target_agent: str = "proposal"
    issues: list = field(default_factory=list)
    severity: float = 0.5

    def to_prompt(self) -> str:
        return f"Critique: {len(self.issues)} issues, severity {self.severity}"


@dataclass
class MockDebateResult:
    """Mock debate result for testing."""

    id: str = "debate_001"
    critiques: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    rounds_used: int = 0
    convergence_status: str = ""
    convergence_similarity: float = 0.0
    per_agent_similarity: dict = field(default_factory=dict)


@dataclass
class MockProtocol:
    """Mock protocol for testing."""

    rounds: int = 2
    asymmetric_stances: bool = False
    rotate_stances: bool = False


@dataclass
class MockConvergence:
    """Mock convergence result."""

    status: str = "converging"
    avg_similarity: float = 0.8
    per_agent_similarity: dict = field(default_factory=dict)
    converged: bool = False


# ============================================================================
# DebateRoundsPhase Construction Tests
# ============================================================================


class TestDebateRoundsPhaseConstruction:
    """Tests for DebateRoundsPhase construction."""

    def test_minimal_construction(self):
        """Should create with no arguments."""
        phase = DebateRoundsPhase()
        assert phase.protocol is None
        assert phase.hooks == {}

    def test_full_construction(self):
        """Should create with all arguments."""
        protocol = MockProtocol()
        cb = MagicMock()
        hooks = {"on_round_start": MagicMock()}

        phase = DebateRoundsPhase(
            protocol=protocol,
            circuit_breaker=cb,
            hooks=hooks,
            recorder=MagicMock(),
        )

        assert phase.protocol is protocol
        assert phase.circuit_breaker is cb
        assert "on_round_start" in phase.hooks


# ============================================================================
# Round Execution Tests
# ============================================================================


class TestRoundExecution:
    """Tests for round execution."""

    @pytest.mark.asyncio
    async def test_execute_rounds(self):
        """Should execute configured number of rounds."""
        protocol = MockProtocol(rounds=2)
        on_round_start = MagicMock()

        phase = DebateRoundsPhase(
            protocol=protocol,
            hooks={"on_round_start": on_round_start},
            critique_with_agent=AsyncMock(return_value=MockCritique()),
            build_revision_prompt=MagicMock(return_value="Revise"),
            generate_with_agent=AsyncMock(return_value="Revised proposal"),
        )

        agents = [MockAgent(name="claude"), MockAgent(name="gpt4")]
        ctx = DebateContext(env=MockEnvironment(), agents=agents, proposers=agents)
        ctx.proposals = {"claude": "Proposal A", "gpt4": "Proposal B"}
        ctx.result = MockDebateResult()

        await phase.execute(ctx)

        assert on_round_start.call_count == 2
        assert ctx.result.rounds_used == 2

    @pytest.mark.asyncio
    async def test_update_role_assignments(self):
        """Should update role assignments each round."""
        protocol = MockProtocol(rounds=1)
        update_roles = MagicMock()

        phase = DebateRoundsPhase(
            protocol=protocol,
            update_role_assignments=update_roles,
            critique_with_agent=AsyncMock(return_value=MockCritique()),
            build_revision_prompt=MagicMock(return_value="Revise"),
            generate_with_agent=AsyncMock(return_value="Revised"),
        )

        agents = [MockAgent(name="claude")]
        ctx = DebateContext(env=MockEnvironment(), agents=agents, proposers=agents)
        ctx.proposals = {"claude": "Proposal A"}
        ctx.result = MockDebateResult()

        await phase.execute(ctx)

        update_roles.assert_called_with(round_num=1)

    @pytest.mark.asyncio
    async def test_notify_spectator_round_start(self):
        """Should notify spectator of round start."""
        protocol = MockProtocol(rounds=1)
        notify = MagicMock()

        phase = DebateRoundsPhase(
            protocol=protocol,
            notify_spectator=notify,
            critique_with_agent=AsyncMock(return_value=MockCritique()),
            build_revision_prompt=MagicMock(return_value="Revise"),
            generate_with_agent=AsyncMock(return_value="Revised"),
        )

        agents = [MockAgent(name="claude")]
        ctx = DebateContext(env=MockEnvironment(), agents=agents, proposers=agents)
        ctx.proposals = {"claude": "Proposal A"}
        ctx.result = MockDebateResult()

        await phase.execute(ctx)

        # Check round notification
        round_calls = [c for c in notify.call_args_list if c[0][0] == "round"]
        assert len(round_calls) >= 1


# ============================================================================
# Critique Phase Tests
# ============================================================================


class TestCritiquePhase:
    """Tests for critique generation."""

    @pytest.mark.asyncio
    async def test_generate_critiques(self):
        """Should generate critiques for proposals."""
        protocol = MockProtocol(rounds=1)

        critique_mock = MockCritique(issues=["issue1"], severity=0.7)

        phase = DebateRoundsPhase(
            protocol=protocol,
            critique_with_agent=AsyncMock(return_value=critique_mock),
            build_revision_prompt=MagicMock(return_value="Revise"),
            generate_with_agent=AsyncMock(return_value="Revised"),
            select_critics_for_proposal=lambda pa, critics: [c for c in critics if c.name != pa],
        )

        agents = [MockAgent(name="claude"), MockAgent(name="gpt4")]
        ctx = DebateContext(env=MockEnvironment(), agents=agents, proposers=agents)
        ctx.proposals = {"claude": "Proposal A", "gpt4": "Proposal B"}
        ctx.result = MockDebateResult()

        await phase.execute(ctx)

        # Should have generated critiques
        assert len(ctx.result.critiques) >= 2

    @pytest.mark.asyncio
    async def test_critique_error_handling(self):
        """Should handle critique errors gracefully."""
        protocol = MockProtocol(rounds=1)
        cb = MagicMock()

        async def critique_with_error(critic, proposal, task, context):
            if critic.name == "error_critic":
                raise Exception("Critique error")
            return MockCritique()

        agents = [MockAgent(name="claude"), MockAgent(name="error_critic")]
        # Mock filter_available_agents to return actual list
        cb.filter_available_agents.return_value = agents

        phase = DebateRoundsPhase(
            protocol=protocol,
            circuit_breaker=cb,
            critique_with_agent=critique_with_error,
            build_revision_prompt=MagicMock(return_value="Revise"),
            generate_with_agent=AsyncMock(return_value="Revised"),
            select_critics_for_proposal=lambda pa, critics: critics,
        )

        ctx = DebateContext(env=MockEnvironment(), agents=agents, proposers=[agents[0]])
        ctx.proposals = {"claude": "Proposal A"}
        ctx.result = MockDebateResult()

        # Should not raise
        await phase.execute(ctx)

        # Should record failure
        cb.record_failure.assert_called_with("error_critic")

    @pytest.mark.asyncio
    async def test_emit_critique_hook(self):
        """Should emit on_critique hook."""
        protocol = MockProtocol(rounds=1)
        on_critique = MagicMock()

        phase = DebateRoundsPhase(
            protocol=protocol,
            hooks={"on_critique": on_critique},
            critique_with_agent=AsyncMock(return_value=MockCritique(issues=["issue1"])),
            build_revision_prompt=MagicMock(return_value="Revise"),
            generate_with_agent=AsyncMock(return_value="Revised"),
            select_critics_for_proposal=lambda pa, critics: [c for c in critics if c.name != pa],
        )

        agents = [MockAgent(name="claude"), MockAgent(name="gpt4")]
        ctx = DebateContext(env=MockEnvironment(), agents=agents, proposers=agents)
        ctx.proposals = {"claude": "Proposal A", "gpt4": "Proposal B"}
        ctx.result = MockDebateResult()

        await phase.execute(ctx)

        on_critique.assert_called()

    @pytest.mark.asyncio
    async def test_critique_circuit_breaker(self):
        """Should filter critics through circuit breaker."""
        protocol = MockProtocol(rounds=1)
        cb = MagicMock()
        cb.filter_available_agents.return_value = [MockAgent(name="claude")]

        phase = DebateRoundsPhase(
            protocol=protocol,
            circuit_breaker=cb,
            critique_with_agent=AsyncMock(return_value=MockCritique()),
            build_revision_prompt=MagicMock(return_value="Revise"),
            generate_with_agent=AsyncMock(return_value="Revised"),
            select_critics_for_proposal=lambda pa, critics: critics,
        )

        agents = [MockAgent(name="claude"), MockAgent(name="gpt4")]
        ctx = DebateContext(env=MockEnvironment(), agents=agents, proposers=agents)
        ctx.proposals = {"claude": "Proposal A"}
        ctx.result = MockDebateResult()

        await phase.execute(ctx)

        cb.filter_available_agents.assert_called()


# ============================================================================
# Revision Phase Tests
# ============================================================================


class TestRevisionPhase:
    """Tests for revision generation."""

    @pytest.mark.asyncio
    async def test_generate_revisions(self):
        """Should generate revisions after critiques."""
        protocol = MockProtocol(rounds=1)

        phase = DebateRoundsPhase(
            protocol=protocol,
            critique_with_agent=AsyncMock(return_value=MockCritique()),
            build_revision_prompt=MagicMock(return_value="Revise"),
            generate_with_agent=AsyncMock(return_value="Revised proposal content"),
            select_critics_for_proposal=lambda pa, critics: [c for c in critics if c.name != pa],
        )

        agents = [MockAgent(name="claude"), MockAgent(name="gpt4")]
        ctx = DebateContext(env=MockEnvironment(), agents=agents, proposers=agents)
        ctx.proposals = {"claude": "Proposal A", "gpt4": "Proposal B"}
        ctx.result = MockDebateResult()

        await phase.execute(ctx)

        # Proposals should be updated with revisions
        assert ctx.proposals["claude"] == "Revised proposal content"

    @pytest.mark.asyncio
    async def test_revision_error_handling(self):
        """Should handle revision errors gracefully."""
        protocol = MockProtocol(rounds=1)
        cb = MagicMock()

        async def generate_with_error(agent, prompt, context):
            if agent.name == "error_agent":
                raise Exception("Revision error")
            return "Revised"

        agents = [MockAgent(name="claude"), MockAgent(name="error_agent")]
        # Mock filter_available_agents to return actual list
        cb.filter_available_agents.return_value = agents

        phase = DebateRoundsPhase(
            protocol=protocol,
            circuit_breaker=cb,
            critique_with_agent=AsyncMock(return_value=MockCritique()),
            build_revision_prompt=MagicMock(return_value="Revise"),
            generate_with_agent=generate_with_error,
            select_critics_for_proposal=lambda pa, critics: [c for c in critics if c.name != pa],
        )

        ctx = DebateContext(env=MockEnvironment(), agents=agents, proposers=agents)
        ctx.proposals = {"claude": "Proposal A", "error_agent": "Proposal B"}
        ctx.result = MockDebateResult()

        # Should not raise
        await phase.execute(ctx)

        # Should record failure for error_agent
        cb.record_failure.assert_called_with("error_agent")

    @pytest.mark.asyncio
    async def test_emit_message_hook_for_revision(self):
        """Should emit on_message hook for revisions."""
        protocol = MockProtocol(rounds=1)
        on_message = MagicMock()

        phase = DebateRoundsPhase(
            protocol=protocol,
            hooks={"on_message": on_message},
            critique_with_agent=AsyncMock(return_value=MockCritique()),
            build_revision_prompt=MagicMock(return_value="Revise"),
            generate_with_agent=AsyncMock(return_value="Revised"),
            select_critics_for_proposal=lambda pa, critics: [c for c in critics if c.name != pa],
        )

        agents = [MockAgent(name="claude"), MockAgent(name="gpt4")]
        ctx = DebateContext(env=MockEnvironment(), agents=agents, proposers=agents)
        ctx.proposals = {"claude": "Proposal A", "gpt4": "Proposal B"}
        ctx.result = MockDebateResult()

        await phase.execute(ctx)

        # Should be called for both critiques and revisions
        assert on_message.call_count >= 4

    @pytest.mark.asyncio
    async def test_record_grounded_position(self):
        """Should record grounded positions after revision."""
        protocol = MockProtocol(rounds=1)
        record_position = MagicMock()

        phase = DebateRoundsPhase(
            protocol=protocol,
            critique_with_agent=AsyncMock(return_value=MockCritique()),
            build_revision_prompt=MagicMock(return_value="Revise"),
            generate_with_agent=AsyncMock(return_value="Revised"),
            record_grounded_position=record_position,
            select_critics_for_proposal=lambda pa, critics: [c for c in critics if c.name != pa],
        )

        agents = [MockAgent(name="claude"), MockAgent(name="gpt4")]
        ctx = DebateContext(env=MockEnvironment(), agents=agents, proposers=agents)
        ctx.proposals = {"claude": "Proposal A", "gpt4": "Proposal B"}
        ctx.result = MockDebateResult()

        await phase.execute(ctx)

        # Should record positions for both agents
        assert record_position.call_count == 2


# ============================================================================
# Convergence Detection Tests
# ============================================================================


class TestConvergenceDetection:
    """Tests for convergence detection."""

    @pytest.mark.asyncio
    async def test_detect_convergence(self):
        """Should detect convergence between rounds."""
        protocol = MockProtocol(rounds=2)

        detector = MagicMock()
        detector.check_convergence.return_value = MockConvergence(
            status="converging", avg_similarity=0.85, converged=False
        )

        phase = DebateRoundsPhase(
            protocol=protocol,
            convergence_detector=detector,
            critique_with_agent=AsyncMock(return_value=MockCritique()),
            build_revision_prompt=MagicMock(return_value="Revise"),
            generate_with_agent=AsyncMock(return_value="Revised"),
            select_critics_for_proposal=lambda pa, critics: [c for c in critics if c.name != pa],
        )

        agents = [MockAgent(name="claude"), MockAgent(name="gpt4")]
        ctx = DebateContext(env=MockEnvironment(), agents=agents, proposers=agents)
        ctx.proposals = {"claude": "Proposal A", "gpt4": "Proposal B"}
        ctx.result = MockDebateResult()

        await phase.execute(ctx)

        # Should have checked convergence
        detector.check_convergence.assert_called()
        assert ctx.result.convergence_status == "converging"

    @pytest.mark.asyncio
    async def test_early_exit_on_convergence(self):
        """Should exit early when fully converged."""
        protocol = MockProtocol(rounds=5)  # Many rounds

        detector = MagicMock()
        detector.check_convergence.return_value = MockConvergence(
            status="converged", avg_similarity=0.99, converged=True
        )

        phase = DebateRoundsPhase(
            protocol=protocol,
            convergence_detector=detector,
            critique_with_agent=AsyncMock(return_value=MockCritique()),
            build_revision_prompt=MagicMock(return_value="Revise"),
            generate_with_agent=AsyncMock(return_value="Revised"),
            select_critics_for_proposal=lambda pa, critics: [c for c in critics if c.name != pa],
        )

        agents = [MockAgent(name="claude"), MockAgent(name="gpt4")]
        ctx = DebateContext(env=MockEnvironment(), agents=agents, proposers=agents)
        ctx.proposals = {"claude": "Proposal A", "gpt4": "Proposal B"}
        ctx.result = MockDebateResult()

        await phase.execute(ctx)

        # Should have exited early (round 2 when convergence detected)
        assert ctx.result.rounds_used < 5

    @pytest.mark.asyncio
    async def test_emit_convergence_hook(self):
        """Should emit on_convergence_check hook."""
        protocol = MockProtocol(rounds=2)
        on_convergence = MagicMock()

        detector = MagicMock()
        detector.check_convergence.return_value = MockConvergence()

        phase = DebateRoundsPhase(
            protocol=protocol,
            convergence_detector=detector,
            hooks={"on_convergence_check": on_convergence},
            critique_with_agent=AsyncMock(return_value=MockCritique()),
            build_revision_prompt=MagicMock(return_value="Revise"),
            generate_with_agent=AsyncMock(return_value="Revised"),
            select_critics_for_proposal=lambda pa, critics: [c for c in critics if c.name != pa],
        )

        agents = [MockAgent(name="claude"), MockAgent(name="gpt4")]
        ctx = DebateContext(env=MockEnvironment(), agents=agents, proposers=agents)
        ctx.proposals = {"claude": "Proposal A", "gpt4": "Proposal B"}
        ctx.result = MockDebateResult()

        await phase.execute(ctx)

        on_convergence.assert_called()


# ============================================================================
# Termination Tests
# ============================================================================


class TestTermination:
    """Tests for early termination."""

    @pytest.mark.asyncio
    async def test_judge_termination(self):
        """Should check judge termination."""
        protocol = MockProtocol(rounds=5)

        async def check_judge_term(round_num, proposals, context):
            return (False, "Judge says stop")  # Should stop

        phase = DebateRoundsPhase(
            protocol=protocol,
            check_judge_termination=check_judge_term,
            critique_with_agent=AsyncMock(return_value=MockCritique()),
            build_revision_prompt=MagicMock(return_value="Revise"),
            generate_with_agent=AsyncMock(return_value="Revised"),
            select_critics_for_proposal=lambda pa, critics: [c for c in critics if c.name != pa],
        )

        agents = [MockAgent(name="claude"), MockAgent(name="gpt4")]
        ctx = DebateContext(env=MockEnvironment(), agents=agents, proposers=agents)
        ctx.proposals = {"claude": "Proposal A", "gpt4": "Proposal B"}
        ctx.result = MockDebateResult()

        await phase.execute(ctx)

        # Should have exited early
        assert ctx.result.rounds_used < 5

    @pytest.mark.asyncio
    async def test_early_stopping(self):
        """Should check early stopping conditions."""
        protocol = MockProtocol(rounds=5)

        async def check_early(round_num, proposals, context):
            return False  # Should stop

        phase = DebateRoundsPhase(
            protocol=protocol,
            check_early_stopping=check_early,
            critique_with_agent=AsyncMock(return_value=MockCritique()),
            build_revision_prompt=MagicMock(return_value="Revise"),
            generate_with_agent=AsyncMock(return_value="Revised"),
            select_critics_for_proposal=lambda pa, critics: [c for c in critics if c.name != pa],
        )

        agents = [MockAgent(name="claude"), MockAgent(name="gpt4")]
        ctx = DebateContext(env=MockEnvironment(), agents=agents, proposers=agents)
        ctx.proposals = {"claude": "Proposal A", "gpt4": "Proposal B"}
        ctx.result = MockDebateResult()

        await phase.execute(ctx)

        # Should have exited early
        assert ctx.result.rounds_used < 5


# ============================================================================
# Stance Rotation Tests
# ============================================================================


class TestStanceRotation:
    """Tests for asymmetric stance rotation."""

    @pytest.mark.asyncio
    async def test_rotate_stances(self):
        """Should rotate stances when configured."""
        protocol = MockProtocol(rounds=2, asymmetric_stances=True, rotate_stances=True)
        assign_stances = MagicMock()

        phase = DebateRoundsPhase(
            protocol=protocol,
            assign_stances=assign_stances,
            critique_with_agent=AsyncMock(return_value=MockCritique()),
            build_revision_prompt=MagicMock(return_value="Revise"),
            generate_with_agent=AsyncMock(return_value="Revised"),
            select_critics_for_proposal=lambda pa, critics: [c for c in critics if c.name != pa],
        )

        agents = [MockAgent(name="claude"), MockAgent(name="gpt4")]
        ctx = DebateContext(env=MockEnvironment(), agents=agents, proposers=agents)
        ctx.proposals = {"claude": "Proposal A", "gpt4": "Proposal B"}
        ctx.result = MockDebateResult()

        await phase.execute(ctx)

        # Should have called assign_stances for each round
        assert assign_stances.call_count == 2


# ============================================================================
# Recorder Tests
# ============================================================================


class TestRecorder:
    """Tests for replay recorder."""

    @pytest.mark.asyncio
    async def test_record_round_start(self):
        """Should record round start."""
        protocol = MockProtocol(rounds=1)
        recorder = MagicMock()

        phase = DebateRoundsPhase(
            protocol=protocol,
            recorder=recorder,
            critique_with_agent=AsyncMock(return_value=MockCritique()),
            build_revision_prompt=MagicMock(return_value="Revise"),
            generate_with_agent=AsyncMock(return_value="Revised"),
            select_critics_for_proposal=lambda pa, critics: [c for c in critics if c.name != pa],
        )

        agents = [MockAgent(name="claude"), MockAgent(name="gpt4")]
        ctx = DebateContext(env=MockEnvironment(), agents=agents, proposers=agents)
        ctx.proposals = {"claude": "Proposal A", "gpt4": "Proposal B"}
        ctx.result = MockDebateResult()

        await phase.execute(ctx)

        recorder.record_phase_change.assert_called()

    @pytest.mark.asyncio
    async def test_record_critiques_and_revisions(self):
        """Should record critiques and revisions."""
        protocol = MockProtocol(rounds=1)
        recorder = MagicMock()

        phase = DebateRoundsPhase(
            protocol=protocol,
            recorder=recorder,
            critique_with_agent=AsyncMock(return_value=MockCritique()),
            build_revision_prompt=MagicMock(return_value="Revise"),
            generate_with_agent=AsyncMock(return_value="Revised"),
            select_critics_for_proposal=lambda pa, critics: [c for c in critics if c.name != pa],
        )

        agents = [MockAgent(name="claude"), MockAgent(name="gpt4")]
        ctx = DebateContext(env=MockEnvironment(), agents=agents, proposers=agents)
        ctx.proposals = {"claude": "Proposal A", "gpt4": "Proposal B"}
        ctx.result = MockDebateResult()

        await phase.execute(ctx)

        # Should have recorded turns for critiques and revisions
        assert recorder.record_turn.call_count >= 4


# ============================================================================
# Partial State Recovery Tests
# ============================================================================


class TestPartialStateRecovery:
    """Tests for partial state recovery."""

    @pytest.mark.asyncio
    async def test_get_partial_messages(self):
        """Should track partial messages for recovery."""
        protocol = MockProtocol(rounds=1)

        phase = DebateRoundsPhase(
            protocol=protocol,
            critique_with_agent=AsyncMock(return_value=MockCritique()),
            build_revision_prompt=MagicMock(return_value="Revise"),
            generate_with_agent=AsyncMock(return_value="Revised"),
            select_critics_for_proposal=lambda pa, critics: [c for c in critics if c.name != pa],
        )

        agents = [MockAgent(name="claude"), MockAgent(name="gpt4")]
        ctx = DebateContext(env=MockEnvironment(), agents=agents, proposers=agents)
        ctx.proposals = {"claude": "Proposal A", "gpt4": "Proposal B"}
        ctx.result = MockDebateResult()

        await phase.execute(ctx)

        # Should have partial messages
        assert len(phase.get_partial_messages()) > 0

    @pytest.mark.asyncio
    async def test_get_partial_critiques(self):
        """Should track partial critiques for recovery."""
        protocol = MockProtocol(rounds=1)

        phase = DebateRoundsPhase(
            protocol=protocol,
            critique_with_agent=AsyncMock(return_value=MockCritique()),
            build_revision_prompt=MagicMock(return_value="Revise"),
            generate_with_agent=AsyncMock(return_value="Revised"),
            select_critics_for_proposal=lambda pa, critics: [c for c in critics if c.name != pa],
        )

        agents = [MockAgent(name="claude"), MockAgent(name="gpt4")]
        ctx = DebateContext(env=MockEnvironment(), agents=agents, proposers=agents)
        ctx.proposals = {"claude": "Proposal A", "gpt4": "Proposal B"}
        ctx.result = MockDebateResult()

        await phase.execute(ctx)

        # Should have partial critiques
        assert len(phase.get_partial_critiques()) > 0


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_proposals(self):
        """Should handle empty proposals."""
        protocol = MockProtocol(rounds=1)

        phase = DebateRoundsPhase(
            protocol=protocol,
            critique_with_agent=AsyncMock(return_value=MockCritique()),
            build_revision_prompt=MagicMock(return_value="Revise"),
            generate_with_agent=AsyncMock(return_value="Revised"),
        )

        ctx = DebateContext(env=MockEnvironment(), agents=[], proposers=[])
        ctx.proposals = {}
        ctx.result = MockDebateResult()

        # Should not raise
        await phase.execute(ctx)

    @pytest.mark.asyncio
    async def test_no_critics_fallback(self):
        """Should use all agents as critics when none designated."""
        protocol = MockProtocol(rounds=1)

        critique_calls = []

        async def critique_mock(critic, proposal, task, context):
            critique_calls.append(critic.name)
            return MockCritique()

        phase = DebateRoundsPhase(
            protocol=protocol,
            critique_with_agent=critique_mock,
            build_revision_prompt=MagicMock(return_value="Revise"),
            generate_with_agent=AsyncMock(return_value="Revised"),
            select_critics_for_proposal=lambda pa, critics: [c for c in critics if c.name != pa],
        )

        # All proposers, no dedicated critics
        agents = [
            MockAgent(name="claude", role="proposer"),
            MockAgent(name="gpt4", role="proposer"),
        ]
        ctx = DebateContext(env=MockEnvironment(), agents=agents, proposers=agents)
        ctx.proposals = {"claude": "Proposal A", "gpt4": "Proposal B"}
        ctx.result = MockDebateResult()

        await phase.execute(ctx)

        # All agents should have critiqued (except self)
        assert "claude" in critique_calls or "gpt4" in critique_calls

    @pytest.mark.asyncio
    async def test_missing_callbacks_warning(self):
        """Should handle missing callbacks gracefully."""
        protocol = MockProtocol(rounds=1)

        phase = DebateRoundsPhase(
            protocol=protocol,
            # No callbacks provided
        )

        agents = [MockAgent(name="claude")]
        ctx = DebateContext(env=MockEnvironment(), agents=agents, proposers=agents)
        ctx.proposals = {"claude": "Proposal A"}
        ctx.result = MockDebateResult()

        # Should not raise
        await phase.execute(ctx)
