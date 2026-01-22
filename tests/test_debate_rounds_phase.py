"""
Tests for DebateRoundsPhase in debate phases.

Tests the debate round loop: critique, revision, convergence, termination.
"""

import asyncio
import pytest
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.debate.phases.debate_rounds import DebateRoundsPhase


# =============================================================================
# Mock Classes
# =============================================================================


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str
    role: Optional[str] = None
    stance: Optional[str] = None


@dataclass
class MockProtocol:
    """Mock protocol for testing."""

    rounds: int = 3
    asymmetric_stances: bool = False
    rotate_stances: bool = False
    use_structured_phases: bool = False


@dataclass
class MockEnvironment:
    """Mock environment for testing."""

    task: str = "Test debate task"


@dataclass
class MockResult:
    """Mock debate result for testing."""

    critiques: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    rounds_used: int = 0
    convergence_status: Optional[str] = None
    convergence_similarity: float = 0.0
    per_agent_similarity: dict = field(default_factory=dict)
    id: str = "test-debate-123"


@dataclass
class MockCritique:
    """Mock critique for testing."""

    issues: list = field(default_factory=list)
    severity: float = 5.0
    # NOTE: target_agent should be the actual agent name (e.g., "agent1", "agent2")
    # not "proposal". Critiques target specific agents, not a generic "proposal".
    target_agent: str = "agent1"

    def to_prompt(self) -> str:
        return f"Critique: {len(self.issues)} issues, severity {self.severity}"


@dataclass
class MockConvergence:
    """Mock convergence result for testing."""

    converged: bool = False
    status: str = "in_progress"
    avg_similarity: float = 0.7
    per_agent_similarity: dict = field(default_factory=dict)


class MockDebateContext:
    """Mock debate context for testing."""

    def __init__(
        self,
        agents: list = None,
        proposals: dict = None,
        env: MockEnvironment = None,
        result: MockResult = None,
    ):
        self.agents = agents or []
        self.proposals = proposals or {}
        self.proposers = [a for a in self.agents if a.role == "proposer"]
        self.env = env or MockEnvironment()
        self.result = result or MockResult()
        self.context_messages = []
        self.cancellation_token = None  # For cancellation support
        self.hook_manager = None  # For pre/post round hooks
        self.per_agent_novelty = {}  # For novelty tracking
        self.avg_novelty = 0.0
        self.low_novelty_agents = []

    def add_message(self, msg):
        self.context_messages.append(msg)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def agents():
    """Three mock agents with roles."""
    return [
        MockAgent(name="agent1", role="proposer"),
        MockAgent(name="agent2", role="critic"),
        MockAgent(name="agent3", role="synthesizer"),
    ]


@pytest.fixture
def protocol():
    """Default protocol."""
    return MockProtocol()


@pytest.fixture
def mock_circuit_breaker():
    """Mock circuit breaker."""
    cb = MagicMock()
    cb.filter_available_agents.side_effect = lambda agents: agents
    cb.record_success = MagicMock()
    cb.record_failure = MagicMock()
    return cb


@pytest.fixture
def mock_convergence_detector():
    """Mock convergence detector."""
    cd = MagicMock()
    cd.check_convergence.return_value = MockConvergence()
    return cd


@pytest.fixture
def mock_recorder():
    """Mock replay recorder."""
    recorder = MagicMock()
    recorder.record_phase_change = MagicMock()
    recorder.record_turn = MagicMock()
    return recorder


# =============================================================================
# Initialization Tests
# =============================================================================


class TestDebateRoundsPhaseInit:
    """Tests for DebateRoundsPhase initialization."""

    def test_init_stores_protocol(self, protocol):
        """Should store protocol reference."""
        phase = DebateRoundsPhase(protocol=protocol)
        assert phase.protocol is protocol

    def test_init_stores_optional_dependencies(
        self, protocol, mock_circuit_breaker, mock_convergence_detector
    ):
        """Should store optional dependencies."""
        phase = DebateRoundsPhase(
            protocol=protocol,
            circuit_breaker=mock_circuit_breaker,
            convergence_detector=mock_convergence_detector,
        )
        assert phase.circuit_breaker is mock_circuit_breaker
        assert phase.convergence_detector is mock_convergence_detector

    def test_init_stores_callbacks(self, protocol):
        """Should store callback functions."""
        update_roles = MagicMock()
        assign_stances = MagicMock()

        phase = DebateRoundsPhase(
            protocol=protocol,
            update_role_assignments=update_roles,
            assign_stances=assign_stances,
        )

        assert phase._update_role_assignments is update_roles
        assert phase._assign_stances is assign_stances

    def test_init_default_hooks(self, protocol):
        """Should default hooks to empty dict."""
        phase = DebateRoundsPhase(protocol=protocol)
        assert phase.hooks == {}

    def test_init_custom_hooks(self, protocol):
        """Should accept custom hooks dict."""
        hooks = {"on_round_start": MagicMock()}
        phase = DebateRoundsPhase(protocol=protocol, hooks=hooks)
        assert phase.hooks == hooks

    def test_init_internal_state(self, protocol):
        """Should initialize internal state correctly."""
        phase = DebateRoundsPhase(protocol=protocol)
        assert phase._partial_messages == []
        assert phase._partial_critiques == []
        assert phase._previous_round_responses == {}


# =============================================================================
# _get_critics Tests
# =============================================================================


class TestGetCritics:
    """Tests for _get_critics method."""

    def test_returns_critics_and_synthesizers(self, protocol, agents):
        """Should return agents with critic or synthesizer role."""
        phase = DebateRoundsPhase(protocol=protocol)
        ctx = MockDebateContext(agents=agents)

        critics = phase._get_critics(ctx)

        assert len(critics) == 2  # critic + synthesizer
        assert any(c.role == "critic" for c in critics)
        assert any(c.role == "synthesizer" for c in critics)

    def test_returns_all_agents_when_no_critics(self, protocol):
        """Should return all agents when none have critic/synthesizer role."""
        agents = [
            MockAgent(name="a1", role="proposer"),
            MockAgent(name="a2", role="proposer"),
        ]
        phase = DebateRoundsPhase(protocol=protocol)
        ctx = MockDebateContext(agents=agents)

        critics = phase._get_critics(ctx)

        assert len(critics) == 2
        assert critics == agents

    def test_filters_through_circuit_breaker(self, protocol, agents, mock_circuit_breaker):
        """Should filter critics through circuit breaker."""
        # Circuit breaker removes agent2
        mock_circuit_breaker.filter_available_agents.side_effect = lambda a: [
            x for x in a if x.name != "agent2"
        ]

        phase = DebateRoundsPhase(protocol=protocol, circuit_breaker=mock_circuit_breaker)
        ctx = MockDebateContext(agents=agents)

        critics = phase._get_critics(ctx)

        assert len(critics) == 1
        assert critics[0].name == "agent3"

    def test_handles_circuit_breaker_error(self, protocol, agents, mock_circuit_breaker, caplog):
        """Should handle circuit breaker errors gracefully."""
        mock_circuit_breaker.filter_available_agents.side_effect = Exception("CB error")

        phase = DebateRoundsPhase(protocol=protocol, circuit_breaker=mock_circuit_breaker)
        ctx = MockDebateContext(agents=agents)

        with caplog.at_level("ERROR"):
            critics = phase._get_critics(ctx)

        # Should return unfiltered critics
        assert len(critics) == 2


# =============================================================================
# _check_convergence Tests
# =============================================================================


class TestCheckConvergence:
    """Tests for _check_convergence method."""

    def test_returns_false_without_detector(self, protocol, agents):
        """Should return False when no convergence detector."""
        phase = DebateRoundsPhase(protocol=protocol)
        ctx = MockDebateContext(
            agents=agents,
            proposals={"agent1": "proposal"},
        )

        result = phase._check_convergence(ctx, round_num=1)

        assert result is False

    def test_returns_false_on_first_round(self, protocol, agents, mock_convergence_detector):
        """Should return False on first round (no previous responses)."""
        phase = DebateRoundsPhase(
            protocol=protocol,
            convergence_detector=mock_convergence_detector,
        )
        ctx = MockDebateContext(
            agents=agents,
            proposals={"agent1": "proposal"},
        )

        result = phase._check_convergence(ctx, round_num=1)

        assert result is False
        # Should NOT call detector on first round
        mock_convergence_detector.check_convergence.assert_not_called()

    def test_returns_true_when_converged(self, protocol, agents, mock_convergence_detector):
        """Should return True when convergence detected."""
        mock_convergence_detector.check_convergence.return_value = MockConvergence(
            converged=True,
            status="converged",
            avg_similarity=0.95,
        )

        phase = DebateRoundsPhase(
            protocol=protocol,
            convergence_detector=mock_convergence_detector,
        )
        phase._previous_round_responses = {"agent1": "old proposal"}

        ctx = MockDebateContext(
            agents=agents,
            proposals={"agent1": "new proposal"},
            result=MockResult(),
        )

        result = phase._check_convergence(ctx, round_num=2)

        assert result is True

    def test_returns_false_when_not_converged(self, protocol, agents, mock_convergence_detector):
        """Should return False when not converged."""
        mock_convergence_detector.check_convergence.return_value = MockConvergence(
            converged=False,
            status="diverging",
            avg_similarity=0.3,
        )

        phase = DebateRoundsPhase(
            protocol=protocol,
            convergence_detector=mock_convergence_detector,
        )
        phase._previous_round_responses = {"agent1": "old proposal"}

        ctx = MockDebateContext(
            agents=agents,
            proposals={"agent1": "new proposal"},
            result=MockResult(),
        )

        result = phase._check_convergence(ctx, round_num=2)

        assert result is False

    def test_updates_result_with_convergence_data(
        self, protocol, agents, mock_convergence_detector
    ):
        """Should update result with convergence data."""
        mock_convergence_detector.check_convergence.return_value = MockConvergence(
            converged=False,
            status="improving",
            avg_similarity=0.75,
            per_agent_similarity={"agent1": 0.75},
        )

        phase = DebateRoundsPhase(
            protocol=protocol,
            convergence_detector=mock_convergence_detector,
        )
        phase._previous_round_responses = {"agent1": "old"}

        result = MockResult()
        ctx = MockDebateContext(
            agents=agents,
            proposals={"agent1": "new"},
            result=result,
        )

        phase._check_convergence(ctx, round_num=2)

        assert result.convergence_status == "improving"
        assert result.convergence_similarity == 0.75

    def test_calls_convergence_hook(self, protocol, agents, mock_convergence_detector):
        """Should call on_convergence_check hook."""
        mock_convergence_detector.check_convergence.return_value = MockConvergence(
            converged=False,
            status="improving",
            avg_similarity=0.8,
        )

        hook = MagicMock()
        phase = DebateRoundsPhase(
            protocol=protocol,
            convergence_detector=mock_convergence_detector,
            hooks={"on_convergence_check": hook},
        )
        phase._previous_round_responses = {"agent1": "old"}

        ctx = MockDebateContext(
            agents=agents,
            proposals={"agent1": "new"},
            result=MockResult(),
        )

        phase._check_convergence(ctx, round_num=2)

        hook.assert_called_once()
        call_kwargs = hook.call_args[1]
        assert call_kwargs["status"] == "improving"
        assert call_kwargs["similarity"] == 0.8
        assert call_kwargs["round_num"] == 2


# =============================================================================
# _should_terminate Tests
# =============================================================================


class TestShouldTerminate:
    """Tests for _should_terminate method."""

    @pytest.mark.asyncio
    async def test_returns_false_without_callbacks(self, protocol, agents):
        """Should return False when no termination callbacks."""
        phase = DebateRoundsPhase(protocol=protocol)
        ctx = MockDebateContext(agents=agents, proposals={})

        result = await phase._should_terminate(ctx, round_num=2)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_on_judge_termination(self, protocol, agents):
        """Should return True when judge says stop."""
        check_judge = AsyncMock(return_value=(False, "Consensus reached"))

        phase = DebateRoundsPhase(
            protocol=protocol,
            check_judge_termination=check_judge,
        )
        ctx = MockDebateContext(agents=agents, proposals={"agent1": "proposal"})

        result = await phase._should_terminate(ctx, round_num=2)

        assert result is True
        check_judge.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_false_when_judge_continues(self, protocol, agents):
        """Should return False when judge says continue."""
        check_judge = AsyncMock(return_value=(True, ""))

        phase = DebateRoundsPhase(
            protocol=protocol,
            check_judge_termination=check_judge,
        )
        ctx = MockDebateContext(agents=agents, proposals={"agent1": "proposal"})

        result = await phase._should_terminate(ctx, round_num=2)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_on_early_stopping(self, protocol, agents):
        """Should return True when early stopping triggered."""
        check_early = AsyncMock(return_value=False)  # False = stop

        phase = DebateRoundsPhase(
            protocol=protocol,
            check_early_stopping=check_early,
        )
        ctx = MockDebateContext(agents=agents, proposals={"agent1": "proposal"})

        result = await phase._should_terminate(ctx, round_num=2)

        assert result is True

    @pytest.mark.asyncio
    async def test_judge_checked_before_early_stopping(self, protocol, agents):
        """Should check judge before early stopping."""
        check_judge = AsyncMock(return_value=(False, "Stop"))  # Stop immediately
        check_early = AsyncMock(return_value=True)  # Would continue

        phase = DebateRoundsPhase(
            protocol=protocol,
            check_judge_termination=check_judge,
            check_early_stopping=check_early,
        )
        ctx = MockDebateContext(agents=agents, proposals={"agent1": "proposal"})

        result = await phase._should_terminate(ctx, round_num=2)

        assert result is True
        # Early stopping should NOT be called since judge already said stop
        check_early.assert_not_called()


# =============================================================================
# get_partial_messages/critiques Tests
# =============================================================================


class TestPartialData:
    """Tests for partial message/critique retrieval."""

    def test_get_partial_messages_empty(self, protocol):
        """Should return empty list initially."""
        phase = DebateRoundsPhase(protocol=protocol)
        assert phase.get_partial_messages() == []

    def test_get_partial_critiques_empty(self, protocol):
        """Should return empty list initially."""
        phase = DebateRoundsPhase(protocol=protocol)
        assert phase.get_partial_critiques() == []


# =============================================================================
# execute Tests (Integration)
# =============================================================================


class TestExecute:
    """Integration tests for execute method."""

    @pytest.mark.asyncio
    async def test_execute_basic_rounds(self, protocol, agents):
        """Should execute multiple rounds."""
        protocol.rounds = 2

        phase = DebateRoundsPhase(protocol=protocol)

        ctx = MockDebateContext(
            agents=agents,
            proposals={"agent1": "initial proposal"},
            result=MockResult(),
        )

        await phase.execute(ctx)

        assert ctx.result.rounds_used == 2

    @pytest.mark.asyncio
    async def test_execute_calls_update_roles(self, protocol, agents):
        """Should call update_role_assignments each round."""
        protocol.rounds = 2
        update_roles = MagicMock()

        phase = DebateRoundsPhase(
            protocol=protocol,
            update_role_assignments=update_roles,
        )

        ctx = MockDebateContext(
            agents=agents,
            proposals={"agent1": "initial"},
            result=MockResult(),
        )

        await phase.execute(ctx)

        assert update_roles.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_rotates_stances(self, agents):
        """Should rotate stances when asymmetric + rotate enabled."""
        protocol = MockProtocol(
            rounds=2,
            asymmetric_stances=True,
            rotate_stances=True,
        )
        assign_stances = MagicMock()

        phase = DebateRoundsPhase(
            protocol=protocol,
            assign_stances=assign_stances,
        )

        ctx = MockDebateContext(
            agents=agents,
            proposals={"agent1": "initial"},
            result=MockResult(),
        )

        await phase.execute(ctx)

        assert assign_stances.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_calls_round_start_hook(self, protocol, agents):
        """Should call on_round_start hook."""
        protocol.rounds = 1
        hook = MagicMock()

        phase = DebateRoundsPhase(
            protocol=protocol,
            hooks={"on_round_start": hook},
        )

        ctx = MockDebateContext(
            agents=agents,
            proposals={"agent1": "initial"},
            result=MockResult(),
        )

        await phase.execute(ctx)

        hook.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_execute_records_phase_change(self, protocol, agents, mock_recorder):
        """Should record phase changes."""
        protocol.rounds = 1

        phase = DebateRoundsPhase(
            protocol=protocol,
            recorder=mock_recorder,
        )

        ctx = MockDebateContext(
            agents=agents,
            proposals={"agent1": "initial"},
            result=MockResult(),
        )

        await phase.execute(ctx)

        mock_recorder.record_phase_change.assert_called_with("round_1_start")

    @pytest.mark.asyncio
    async def test_execute_breaks_on_convergence(self, protocol, agents, mock_convergence_detector):
        """Should break loop when converged."""
        protocol.rounds = 5
        mock_convergence_detector.check_convergence.return_value = MockConvergence(
            converged=True,
            status="converged",
            avg_similarity=0.95,
        )

        phase = DebateRoundsPhase(
            protocol=protocol,
            convergence_detector=mock_convergence_detector,
        )
        # Set previous responses so convergence check runs
        phase._previous_round_responses = {"agent1": "previous"}

        ctx = MockDebateContext(
            agents=agents,
            proposals={"agent1": "current"},
            result=MockResult(),
        )

        await phase.execute(ctx)

        # Should have stopped at round 1 due to convergence
        assert ctx.result.rounds_used == 1

    @pytest.mark.asyncio
    async def test_execute_breaks_on_termination(self, protocol, agents):
        """Should break loop on early termination."""
        protocol.rounds = 5
        check_judge = AsyncMock(return_value=(False, "Done"))

        phase = DebateRoundsPhase(
            protocol=protocol,
            check_judge_termination=check_judge,
        )

        ctx = MockDebateContext(
            agents=agents,
            proposals={"agent1": "initial"},
            result=MockResult(),
        )

        await phase.execute(ctx)

        # Should stop after first round's termination check
        assert ctx.result.rounds_used == 1

    @pytest.mark.asyncio
    async def test_execute_notifies_spectator(self, protocol, agents):
        """Should notify spectator of round start."""
        protocol.rounds = 1
        notify = MagicMock()

        phase = DebateRoundsPhase(
            protocol=protocol,
            notify_spectator=notify,
        )

        ctx = MockDebateContext(
            agents=agents,
            proposals={"agent1": "initial"},
            result=MockResult(),
        )

        await phase.execute(ctx)

        # Should have been called for round start
        notify.assert_any_call(
            "round",
            details="Starting Round 1",
            agent="system",
        )


# =============================================================================
# _critique_phase Tests
# =============================================================================


class TestCritiquePhase:
    """Tests for _critique_phase method."""

    @pytest.mark.asyncio
    async def test_skips_without_callback(self, protocol, agents, caplog):
        """Should skip critique phase without critique_with_agent callback."""
        phase = DebateRoundsPhase(protocol=protocol)

        ctx = MockDebateContext(
            agents=agents,
            proposals={"agent1": "proposal"},
            result=MockResult(),
        )

        critics = [agents[1], agents[2]]  # critic + synthesizer

        with caplog.at_level("WARNING"):
            await phase._critique_phase(ctx, critics, round_num=1)

        assert "No critique_with_agent callback" in caplog.text

    @pytest.mark.asyncio
    async def test_generates_critiques(self, protocol, agents, mock_circuit_breaker):
        """Should generate critiques for proposals."""
        critique_result = MockCritique(
            issues=["issue1", "issue2"],
            severity=6.5,
        )
        critique_fn = AsyncMock(return_value=critique_result)

        phase = DebateRoundsPhase(
            protocol=protocol,
            circuit_breaker=mock_circuit_breaker,
            critique_with_agent=critique_fn,
        )

        result = MockResult()
        ctx = MockDebateContext(
            agents=agents,
            proposals={"agent1": "proposal text"},
            result=result,
        )

        critics = [agents[1]]  # Just critic

        await phase._critique_phase(ctx, critics, round_num=1)

        # Should have one critique
        assert len(result.critiques) == 1
        assert result.critiques[0].severity == 6.5

    @pytest.mark.asyncio
    async def test_records_circuit_breaker_success(self, protocol, agents, mock_circuit_breaker):
        """Should record success in circuit breaker."""
        critique_fn = AsyncMock(return_value=MockCritique())

        phase = DebateRoundsPhase(
            protocol=protocol,
            circuit_breaker=mock_circuit_breaker,
            critique_with_agent=critique_fn,
        )

        ctx = MockDebateContext(
            agents=agents,
            proposals={"agent1": "proposal"},
            result=MockResult(),
        )

        critics = [agents[1]]
        await phase._critique_phase(ctx, critics, round_num=1)

        mock_circuit_breaker.record_success.assert_called()

    @pytest.mark.asyncio
    async def test_records_circuit_breaker_failure(self, protocol, agents, mock_circuit_breaker):
        """Should record failure in circuit breaker on error."""
        critique_fn = AsyncMock(side_effect=Exception("API error"))

        phase = DebateRoundsPhase(
            protocol=protocol,
            circuit_breaker=mock_circuit_breaker,
            critique_with_agent=critique_fn,
        )

        ctx = MockDebateContext(
            agents=agents,
            proposals={"agent1": "proposal"},
            result=MockResult(),
        )

        critics = [agents[1]]
        await phase._critique_phase(ctx, critics, round_num=1)

        mock_circuit_breaker.record_failure.assert_called()


# =============================================================================
# _revision_phase Tests
# =============================================================================


class TestRevisionPhase:
    """Tests for _revision_phase method."""

    @pytest.mark.asyncio
    async def test_skips_without_callbacks(self, protocol, agents, caplog):
        """Should skip revision phase without required callbacks."""
        phase = DebateRoundsPhase(protocol=protocol)

        ctx = MockDebateContext(
            agents=agents,
            proposals={"agent1": "proposal"},
            result=MockResult(critiques=[MockCritique()]),
        )

        critics = [agents[1]]

        with caplog.at_level("WARNING"):
            await phase._revision_phase(ctx, critics, round_num=1)

        assert "Missing callbacks" in caplog.text

    @pytest.mark.asyncio
    async def test_generates_revisions(self, protocol, agents, mock_circuit_breaker):
        """Should generate revisions for proposals."""
        generate_fn = AsyncMock(return_value="revised proposal")
        build_prompt_fn = MagicMock(return_value="revision prompt")

        phase = DebateRoundsPhase(
            protocol=protocol,
            circuit_breaker=mock_circuit_breaker,
            generate_with_agent=generate_fn,
            build_revision_prompt=build_prompt_fn,
        )

        # Critique targeting agent1 specifically (the proposer)
        result = MockResult(critiques=[MockCritique(target_agent="agent1")])
        ctx = MockDebateContext(
            agents=agents,
            proposals={"agent1": "original proposal"},
            result=result,
        )

        critics = [agents[1]]
        await phase._revision_phase(ctx, critics, round_num=1)

        # Should have updated proposal for agent1 since critique targeted agent1
        assert ctx.proposals["agent1"] == "revised proposal"

    @pytest.mark.asyncio
    async def test_handles_revision_errors(self, protocol, agents, mock_circuit_breaker, caplog):
        """Should handle revision errors gracefully."""
        generate_fn = AsyncMock(side_effect=Exception("API error"))
        build_prompt_fn = MagicMock(return_value="prompt")

        phase = DebateRoundsPhase(
            protocol=protocol,
            circuit_breaker=mock_circuit_breaker,
            generate_with_agent=generate_fn,
            build_revision_prompt=build_prompt_fn,
        )

        # Critique targeting agent1 specifically
        result = MockResult(critiques=[MockCritique(target_agent="agent1")])
        ctx = MockDebateContext(
            agents=agents,
            proposals={"agent1": "original"},
            result=result,
        )

        critics = [agents[1]]

        with caplog.at_level("ERROR"):
            await phase._revision_phase(ctx, critics, round_num=1)

        # Should have recorded failure
        mock_circuit_breaker.record_failure.assert_called()

    @pytest.mark.asyncio
    async def test_skips_without_critiques(self, protocol, agents):
        """Should skip revision when no critiques target the agent."""
        generate_fn = AsyncMock(return_value="revised")
        build_prompt_fn = MagicMock(return_value="prompt")

        phase = DebateRoundsPhase(
            protocol=protocol,
            generate_with_agent=generate_fn,
            build_revision_prompt=build_prompt_fn,
        )

        # Critique targets "other_agent", not "agent1" (the proposer)
        # So agent1 should skip revision since no critiques target them
        result = MockResult(critiques=[MockCritique(target_agent="other_agent")])
        ctx = MockDebateContext(
            agents=agents,
            proposals={"agent1": "original"},
            result=result,
        )

        critics = [agents[1]]
        await phase._revision_phase(ctx, critics, round_num=1)

        # Generate should not have been called since no critiques target agent1
        generate_fn.assert_not_called()


# =============================================================================
# RLM Ready Signal Tests
# =============================================================================


class TestRLMReadySignal:
    """Tests for RLM ready signal pattern implementation."""

    def test_parse_ready_signal_html_comment_format(self):
        """Should parse ready signal from HTML comment format."""
        from aragora.debate.phases import parse_ready_signal

        content = """This is my final position on the matter.

<!-- READY_SIGNAL: {"confidence": 0.92, "ready": true, "reasoning": "Fully refined"} -->"""

        signal = parse_ready_signal("agent1", content, round_num=3)

        assert signal.agent == "agent1"
        assert signal.confidence == 0.92
        assert signal.ready is True
        assert signal.reasoning == "Fully refined"
        assert signal.round_num == 3

    def test_parse_ready_signal_json_code_block_format(self):
        """Should parse ready signal from JSON code block format."""
        from aragora.debate.phases import parse_ready_signal

        content = """My analysis is complete.

```ready_signal {"confidence": 0.85, "ready": true, "reasoning": "No further changes needed"}```"""

        signal = parse_ready_signal("agent2", content, round_num=2)

        assert signal.confidence == 0.85
        assert signal.ready is True

    def test_parse_ready_signal_inline_format(self):
        """Should parse ready signal from inline marker format."""
        from aragora.debate.phases import parse_ready_signal

        content = "Final answer provided. [READY: confidence=0.88, ready=true]"

        signal = parse_ready_signal("agent3", content, round_num=4)

        assert signal.confidence == 0.88
        assert signal.ready is True

    def test_parse_ready_signal_natural_language(self):
        """Should detect ready signal from natural language markers."""
        from aragora.debate.phases import parse_ready_signal

        content = "This is my final position. I believe no further refinement is needed."

        signal = parse_ready_signal("agent1", content, round_num=5)

        assert signal.ready is True
        assert signal.confidence == 0.7  # Natural language gets moderate confidence

    def test_parse_ready_signal_no_signal(self):
        """Should return defaults when no ready signal found."""
        from aragora.debate.phases import parse_ready_signal

        content = "This is my proposal but I want to keep iterating."

        signal = parse_ready_signal("agent1", content, round_num=1)

        assert signal.ready is False
        assert signal.confidence == 0.5  # Default

    def test_parse_ready_signal_empty_content(self):
        """Should handle empty content gracefully."""
        from aragora.debate.phases import parse_ready_signal

        signal = parse_ready_signal("agent1", "", round_num=1)

        assert signal.ready is False
        assert signal.confidence == 0.5


class TestAgentReadinessSignal:
    """Tests for AgentReadinessSignal dataclass."""

    def test_is_high_confidence_true(self):
        """Should return True when confidence meets threshold."""
        from aragora.debate.phases.debate_rounds import (
            AgentReadinessSignal,
            RLM_READY_CONFIDENCE_THRESHOLD,
        )

        signal = AgentReadinessSignal(
            agent="test",
            confidence=RLM_READY_CONFIDENCE_THRESHOLD,
            ready=True,
        )

        assert signal.is_high_confidence() is True

    def test_is_high_confidence_false(self):
        """Should return False when confidence below threshold."""
        from aragora.debate.phases.debate_rounds import AgentReadinessSignal

        signal = AgentReadinessSignal(
            agent="test",
            confidence=0.5,
            ready=True,
        )

        assert signal.is_high_confidence() is False

    def test_should_terminate_requires_both(self):
        """Should only terminate when ready AND high confidence."""
        from aragora.debate.phases.debate_rounds import AgentReadinessSignal

        # High confidence but not ready
        signal1 = AgentReadinessSignal(agent="a", confidence=0.9, ready=False)
        assert signal1.should_terminate() is False

        # Ready but low confidence
        signal2 = AgentReadinessSignal(agent="b", confidence=0.5, ready=True)
        assert signal2.should_terminate() is False

        # Both ready and high confidence
        signal3 = AgentReadinessSignal(agent="c", confidence=0.9, ready=True)
        assert signal3.should_terminate() is True


class TestCollectiveReadiness:
    """Tests for CollectiveReadiness tracking."""

    def test_ready_count_tracks_ready_agents(self):
        """Should count agents that should terminate."""
        from aragora.debate.phases.debate_rounds import (
            AgentReadinessSignal,
            CollectiveReadiness,
        )

        readiness = CollectiveReadiness()
        readiness.update(AgentReadinessSignal(agent="a1", confidence=0.9, ready=True))
        readiness.update(AgentReadinessSignal(agent="a2", confidence=0.9, ready=True))
        readiness.update(AgentReadinessSignal(agent="a3", confidence=0.5, ready=False))

        assert readiness.ready_count == 2
        assert readiness.total_count == 3

    def test_avg_confidence_calculation(self):
        """Should calculate average confidence correctly."""
        from aragora.debate.phases.debate_rounds import (
            AgentReadinessSignal,
            CollectiveReadiness,
        )

        readiness = CollectiveReadiness()
        readiness.update(AgentReadinessSignal(agent="a1", confidence=0.8))
        readiness.update(AgentReadinessSignal(agent="a2", confidence=0.6))

        assert readiness.avg_confidence == 0.7

    def test_has_quorum_true(self):
        """Should detect quorum when enough agents ready."""
        from aragora.debate.phases.debate_rounds import (
            AgentReadinessSignal,
            CollectiveReadiness,
        )

        readiness = CollectiveReadiness()
        # 3 out of 4 agents ready (75% quorum)
        readiness.update(AgentReadinessSignal(agent="a1", confidence=0.9, ready=True))
        readiness.update(AgentReadinessSignal(agent="a2", confidence=0.85, ready=True))
        readiness.update(AgentReadinessSignal(agent="a3", confidence=0.82, ready=True))
        readiness.update(AgentReadinessSignal(agent="a4", confidence=0.5, ready=False))

        assert readiness.has_quorum() is True

    def test_has_quorum_false(self):
        """Should return False when quorum not reached."""
        from aragora.debate.phases.debate_rounds import (
            AgentReadinessSignal,
            CollectiveReadiness,
        )

        readiness = CollectiveReadiness()
        # Only 1 out of 4 agents ready (25%)
        readiness.update(AgentReadinessSignal(agent="a1", confidence=0.9, ready=True))
        readiness.update(AgentReadinessSignal(agent="a2", confidence=0.5, ready=False))
        readiness.update(AgentReadinessSignal(agent="a3", confidence=0.5, ready=False))
        readiness.update(AgentReadinessSignal(agent="a4", confidence=0.5, ready=False))

        assert readiness.has_quorum() is False


class TestRLMReadyQuorumCheck:
    """Tests for _check_rlm_ready_quorum method."""

    def test_returns_true_when_quorum_reached(self, protocol):
        """Should return True when enough agents signal ready."""
        phase = DebateRoundsPhase(protocol=protocol)

        ctx = MockDebateContext(
            proposals={
                "agent1": 'Position A <!-- READY_SIGNAL: {"confidence": 0.9, "ready": true} -->',
                "agent2": 'Position B <!-- READY_SIGNAL: {"confidence": 0.85, "ready": true} -->',
                "agent3": 'Position C <!-- READY_SIGNAL: {"confidence": 0.88, "ready": true} -->',
                "agent4": "Still iterating...",
            }
        )

        result = phase._check_rlm_ready_quorum(ctx, round_num=3)

        assert result is True
        assert phase._collective_readiness.ready_count == 3

    def test_returns_false_when_no_quorum(self, protocol):
        """Should return False when not enough agents signal ready."""
        phase = DebateRoundsPhase(protocol=protocol)

        ctx = MockDebateContext(
            proposals={
                "agent1": 'Position A <!-- READY_SIGNAL: {"confidence": 0.9, "ready": true} -->',
                "agent2": "Still working...",
                "agent3": "More iteration needed...",
                "agent4": "Not done yet...",
            }
        )

        result = phase._check_rlm_ready_quorum(ctx, round_num=2)

        assert result is False
        assert phase._collective_readiness.ready_count == 1

    def test_emits_hook_on_quorum(self, protocol):
        """Should emit hook when quorum reached."""
        hook_fn = MagicMock()
        phase = DebateRoundsPhase(
            protocol=protocol,
            hooks={"on_rlm_ready_quorum": hook_fn},
        )

        ctx = MockDebateContext(
            proposals={
                "a1": 'Done <!-- READY_SIGNAL: {"confidence": 0.9, "ready": true} -->',
                "a2": 'Done <!-- READY_SIGNAL: {"confidence": 0.85, "ready": true} -->',
                "a3": 'Done <!-- READY_SIGNAL: {"confidence": 0.88, "ready": true} -->',
            }
        )

        phase._check_rlm_ready_quorum(ctx, round_num=4)

        hook_fn.assert_called_once()
        call_kwargs = hook_fn.call_args.kwargs
        assert call_kwargs["round_num"] == 4
        assert len(call_kwargs["ready_agents"]) == 3

    def test_notifies_spectator_on_quorum(self, protocol):
        """Should notify spectator when quorum reached."""
        notify_fn = MagicMock()
        phase = DebateRoundsPhase(
            protocol=protocol,
            notify_spectator=notify_fn,
        )

        ctx = MockDebateContext(
            proposals={
                "a1": 'Done <!-- READY_SIGNAL: {"confidence": 0.9, "ready": true} -->',
                "a2": 'Done <!-- READY_SIGNAL: {"confidence": 0.9, "ready": true} -->',
                "a3": 'Done <!-- READY_SIGNAL: {"confidence": 0.9, "ready": true} -->',
            }
        )

        phase._check_rlm_ready_quorum(ctx, round_num=3)

        notify_fn.assert_called()
        assert notify_fn.call_args[0][0] == "rlm_ready"


class TestRLMIntegrationWithShouldTerminate:
    """Tests for RLM ready signal integration with _should_terminate."""

    @pytest.mark.asyncio
    async def test_terminates_on_rlm_quorum(self, protocol):
        """Should terminate when RLM quorum is reached."""
        phase = DebateRoundsPhase(protocol=protocol)

        ctx = MockDebateContext(
            proposals={
                "a1": 'Final <!-- READY_SIGNAL: {"confidence": 0.9, "ready": true} -->',
                "a2": 'Final <!-- READY_SIGNAL: {"confidence": 0.85, "ready": true} -->',
                "a3": 'Final <!-- READY_SIGNAL: {"confidence": 0.88, "ready": true} -->',
            }
        )

        result = await phase._should_terminate(ctx, round_num=3)

        assert result is True

    @pytest.mark.asyncio
    async def test_rlm_check_runs_before_judge(self, protocol):
        """RLM check should run before judge-based termination."""
        judge_check = AsyncMock(return_value=(True, "continue"))
        phase = DebateRoundsPhase(
            protocol=protocol,
            check_judge_termination=judge_check,
        )

        ctx = MockDebateContext(
            proposals={
                "a1": 'Final <!-- READY_SIGNAL: {"confidence": 0.9, "ready": true} -->',
                "a2": 'Final <!-- READY_SIGNAL: {"confidence": 0.9, "ready": true} -->',
                "a3": 'Final <!-- READY_SIGNAL: {"confidence": 0.9, "ready": true} -->',
            }
        )

        result = await phase._should_terminate(ctx, round_num=3)

        # Should terminate due to RLM quorum, judge check should not be called
        assert result is True
        judge_check.assert_not_called()
