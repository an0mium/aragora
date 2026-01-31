"""
Tests for debate rounds phase module.

Tests cover:
- _calculate_phase_timeout utility function
- _is_effectively_empty_critique utility function
- _with_callback_timeout utility function
- DebateRoundsPhase initialization
- execute method (full round loop)
- _execute_round method internals
- _get_critics method
- _critique_phase method
- _revision_phase method
- _should_terminate method
- _refresh_evidence_for_round method
- _compress_debate_context method
- _build_final_synthesis_prompt method
- _emit_heartbeat method
- _observe_rhetorical_patterns method
- get_partial_messages / get_partial_critiques accessors
- Error handling and edge cases
"""

import asyncio
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.phases.debate_rounds import (
    DebateRoundsPhase,
    _calculate_phase_timeout,
    _is_effectively_empty_critique,
    _with_callback_timeout,
    REVISION_PHASE_BASE_TIMEOUT,
    DEFAULT_CALLBACK_TIMEOUT,
)


# =============================================================================
# Mock Objects
# =============================================================================


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str = "test-agent"
    provider: str = "test-provider"
    model_type: str = "test-model"
    role: str = "proposer"
    timeout: float = 30.0
    stance: str = ""


@dataclass
class MockCritique:
    """Mock critique for testing."""

    agent: str = "critic-1"
    target_agent: str = "proposer-1"
    target_content: str = "proposal content"
    issues: list = field(default_factory=lambda: ["Issue 1"])
    suggestions: list = field(default_factory=lambda: ["Suggestion 1"])
    severity: float = 5.0
    reasoning: str = "Test reasoning"

    @property
    def target(self) -> str:
        return self.target_agent

    def to_prompt(self) -> str:
        issues_str = "\n".join(f"  - {i}" for i in self.issues)
        suggestions_str = "\n".join(f"  - {s}" for s in self.suggestions)
        return f"Critique from {self.agent} (severity: {self.severity:.1f}):\nIssues:\n{issues_str}\nSuggestions:\n{suggestions_str}\nReasoning: {self.reasoning}"


@dataclass
class MockEnv:
    """Mock environment for testing."""

    task: str = "What is the best approach to testing?"


@dataclass
class MockResult:
    """Mock debate result for testing."""

    id: str = "result-123"
    messages: list = field(default_factory=list)
    critiques: list = field(default_factory=list)
    rounds_used: int = 0
    metadata: Optional[dict] = field(default_factory=dict)


@dataclass
class MockDebateContext:
    """Mock debate context for testing."""

    result: MockResult = field(default_factory=MockResult)
    env: MockEnv = field(default_factory=MockEnv)
    proposals: dict = field(default_factory=dict)
    context_messages: list = field(default_factory=list)
    agents: list = field(default_factory=list)
    proposers: list = field(default_factory=list)
    critics: list = field(default_factory=list)
    debate_id: str = "test-debate-123"
    round_critiques: list = field(default_factory=list)
    evidence_pack: Any = None
    loop_id: str = "test-loop"
    cancellation_token: Any = None
    budget_check_callback: Any = None
    hook_manager: Any = None

    def add_message(self, msg):
        """Add message to context."""
        self.context_messages.append(msg)


@dataclass
class MockProtocol:
    """Mock debate protocol for testing."""

    rounds: int = 3
    asymmetric_stances: bool = False
    rotate_stances: bool = False
    use_structured_phases: bool = False

    def get_round_phase(self, round_number: int):
        return None


@dataclass
class MockConvergenceResult:
    """Mock convergence result."""

    converged: bool = False
    blocked_by_trickster: bool = False
    status: str = "refining"
    similarity: float = 0.5


# A mock context manager for perf monitor
@contextmanager
def _noop_cm(*args, **kwargs):
    yield MagicMock()


# =============================================================================
# _calculate_phase_timeout Tests
# =============================================================================


class TestCalculatePhaseTimeout:
    """Tests for _calculate_phase_timeout utility function."""

    def test_returns_base_timeout_for_few_agents(self):
        """Returns at least REVISION_PHASE_BASE_TIMEOUT for few agents."""
        result = _calculate_phase_timeout(num_agents=1, agent_timeout=30.0)
        assert result >= REVISION_PHASE_BASE_TIMEOUT

    def test_scales_with_agent_count(self):
        """Timeout scales with number of agents."""
        result_few = _calculate_phase_timeout(num_agents=2, agent_timeout=60.0)
        result_many = _calculate_phase_timeout(num_agents=20, agent_timeout=60.0)
        assert result_many > result_few

    def test_includes_buffer(self):
        """Timeout includes a 60-second buffer."""
        # For a scenario where calculated > base, the 60s buffer is included
        result = _calculate_phase_timeout(num_agents=100, agent_timeout=180.0)
        # The formula is (num_agents / MAX_CONCURRENT_REVISIONS) * agent_timeout + 60
        # It should be above base since we have many agents
        assert result > REVISION_PHASE_BASE_TIMEOUT


# =============================================================================
# _is_effectively_empty_critique Tests
# =============================================================================


class TestIsEffectivelyEmptyCritique:
    """Tests for _is_effectively_empty_critique utility function."""

    def test_empty_issues_and_suggestions(self):
        """Returns True when critique has no issues or suggestions."""
        critique = MockCritique(issues=[], suggestions=[])
        assert _is_effectively_empty_critique(critique) is True

    def test_placeholder_issue_no_suggestions(self):
        """Returns True for placeholder 'agent response was empty' issue."""
        critique = MockCritique(issues=["Agent response was empty"], suggestions=[])
        assert _is_effectively_empty_critique(critique) is True

    def test_placeholder_with_whitespace(self):
        """Returns True for placeholder with leading/trailing whitespace."""
        critique = MockCritique(issues=["  agent response was empty  "], suggestions=[])
        assert _is_effectively_empty_critique(critique) is True

    def test_real_issues_returns_false(self):
        """Returns False when critique has real issues."""
        critique = MockCritique(issues=["The argument lacks evidence"], suggestions=["Add data"])
        assert _is_effectively_empty_critique(critique) is False

    def test_placeholder_with_suggestions_returns_false(self):
        """Returns False when placeholder issue has suggestions."""
        critique = MockCritique(
            issues=["Agent response was empty"],
            suggestions=["Try harder"],
        )
        assert _is_effectively_empty_critique(critique) is False

    def test_only_whitespace_issues(self):
        """Returns True when issues are only whitespace."""
        critique = MockCritique(issues=["  ", ""], suggestions=[])
        assert _is_effectively_empty_critique(critique) is True


# =============================================================================
# _with_callback_timeout Tests
# =============================================================================


class TestWithCallbackTimeout:
    """Tests for _with_callback_timeout utility function."""

    @pytest.mark.asyncio
    async def test_returns_result_on_success(self):
        """Returns coroutine result when it completes in time."""

        async def fast_coro():
            return 42

        result = await _with_callback_timeout(fast_coro(), timeout=5.0)
        assert result == 42

    @pytest.mark.asyncio
    async def test_returns_default_on_timeout(self):
        """Returns default value when coroutine times out."""

        async def slow_coro():
            await asyncio.sleep(10)
            return 42

        result = await _with_callback_timeout(slow_coro(), timeout=0.01, default="fallback")
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_uses_none_as_default(self):
        """Uses None as the default when not specified."""

        async def slow_coro():
            await asyncio.sleep(10)

        result = await _with_callback_timeout(slow_coro(), timeout=0.01)
        assert result is None


# =============================================================================
# DebateRoundsPhase Initialization Tests
# =============================================================================


class TestDebateRoundsPhaseInit:
    """Tests for DebateRoundsPhase initialization."""

    def test_init_with_defaults(self):
        """Phase initializes with default values."""
        phase = DebateRoundsPhase()

        assert phase.protocol is None
        assert phase.circuit_breaker is None
        assert phase.convergence_detector is None
        assert phase.recorder is None
        assert phase.hooks == {}
        assert phase.trickster is None
        assert phase.rhetorical_observer is None
        assert phase.event_emitter is None
        assert phase.novelty_tracker is None

    def test_init_stores_callbacks(self):
        """Phase stores all injected callbacks."""
        update_roles = MagicMock()
        assign_stances = MagicMock()
        critique_fn = AsyncMock()
        generate_fn = AsyncMock()

        phase = DebateRoundsPhase(
            update_role_assignments=update_roles,
            assign_stances=assign_stances,
            critique_with_agent=critique_fn,
            generate_with_agent=generate_fn,
        )

        assert phase._update_role_assignments is update_roles
        assert phase._assign_stances is assign_stances
        assert phase._critique_with_agent is critique_fn
        assert phase._generate_with_agent is generate_fn

    def test_init_creates_convergence_tracker(self):
        """Phase creates a DebateConvergenceTracker on init."""
        phase = DebateRoundsPhase()
        assert phase._convergence_tracker is not None

    def test_init_initializes_partial_state(self):
        """Phase initializes empty partial message/critique lists."""
        phase = DebateRoundsPhase()
        assert phase._partial_messages == []
        assert phase._partial_critiques == []


# =============================================================================
# _get_critics Tests
# =============================================================================


class TestGetCritics:
    """Tests for _get_critics method."""

    def test_returns_critics_by_role(self):
        """Returns agents with critic or synthesizer role."""
        agents = [
            MockAgent(name="proposer-1", role="proposer"),
            MockAgent(name="critic-1", role="critic"),
            MockAgent(name="synth-1", role="synthesizer"),
        ]
        ctx = MockDebateContext(agents=agents)

        phase = DebateRoundsPhase()
        critics = phase._get_critics(ctx)

        assert len(critics) == 2
        assert any(c.name == "critic-1" for c in critics)
        assert any(c.name == "synth-1" for c in critics)

    def test_returns_all_agents_when_no_critics(self):
        """Returns all agents when none have critic/synthesizer role."""
        agents = [
            MockAgent(name="agent-1", role="proposer"),
            MockAgent(name="agent-2", role="proposer"),
        ]
        ctx = MockDebateContext(agents=agents)

        phase = DebateRoundsPhase()
        critics = phase._get_critics(ctx)

        assert len(critics) == 2

    def test_filters_through_circuit_breaker(self):
        """Filters critics through circuit breaker when available."""
        agents = [
            MockAgent(name="healthy", role="critic"),
            MockAgent(name="broken", role="critic"),
        ]
        ctx = MockDebateContext(agents=agents)

        cb = MagicMock()
        cb.filter_available_agents.return_value = [agents[0]]

        phase = DebateRoundsPhase(circuit_breaker=cb)
        critics = phase._get_critics(ctx)

        assert len(critics) == 1
        assert critics[0].name == "healthy"


# =============================================================================
# _emit_heartbeat Tests
# =============================================================================


class TestEmitHeartbeat:
    """Tests for _emit_heartbeat method."""

    def test_calls_hook_when_present(self):
        """Calls on_heartbeat hook when registered."""
        hook = MagicMock()
        phase = DebateRoundsPhase(hooks={"on_heartbeat": hook})

        phase._emit_heartbeat("round_1", "alive")

        hook.assert_called_once_with(phase="round_1", status="alive")

    def test_no_error_when_hook_missing(self):
        """No error when on_heartbeat hook is not registered."""
        phase = DebateRoundsPhase(hooks={})

        # Should not raise
        phase._emit_heartbeat("round_1", "alive")

    def test_swallows_hook_exceptions(self):
        """Swallows exceptions from the heartbeat hook."""
        hook = MagicMock(side_effect=RuntimeError("hook failed"))
        phase = DebateRoundsPhase(hooks={"on_heartbeat": hook})

        # Should not raise
        phase._emit_heartbeat("round_1", "alive")


# =============================================================================
# _observe_rhetorical_patterns Tests
# =============================================================================


class TestObserveRhetoricalPatterns:
    """Tests for _observe_rhetorical_patterns method."""

    def test_noop_without_observer(self):
        """Does nothing when no rhetorical observer is set."""
        phase = DebateRoundsPhase()
        # Should not raise
        phase._observe_rhetorical_patterns("agent1", "some content", 1)

    def test_calls_observer_and_emits_event(self):
        """Calls observer and emits events for detected patterns."""
        mock_obs = MagicMock()
        mock_pattern = MagicMock()
        mock_pattern.value = "appeal_to_authority"
        mock_observation = MagicMock()
        mock_observation.pattern = mock_pattern
        mock_observation.confidence = 0.8
        mock_observation.audience_commentary = "Interesting pattern"
        mock_observation.to_dict.return_value = {"pattern": "appeal_to_authority"}
        mock_obs.observe.return_value = [mock_observation]

        mock_emitter = MagicMock()
        hook = MagicMock()
        phase = DebateRoundsPhase(
            rhetorical_observer=mock_obs,
            event_emitter=mock_emitter,
            hooks={"on_rhetorical_observation": hook},
        )

        phase._observe_rhetorical_patterns("agent1", "content", 2, "loop-123")

        mock_obs.observe.assert_called_once()
        mock_emitter.emit_sync.assert_called_once()
        hook.assert_called_once()

    def test_handles_observer_exception(self):
        """Handles exceptions from the observer gracefully."""
        mock_obs = MagicMock()
        mock_obs.observe.side_effect = RuntimeError("observer crashed")

        phase = DebateRoundsPhase(rhetorical_observer=mock_obs)
        # Should not raise
        phase._observe_rhetorical_patterns("agent1", "content", 1)


# =============================================================================
# get_partial_messages / get_partial_critiques Tests
# =============================================================================


class TestPartialAccessors:
    """Tests for get_partial_messages and get_partial_critiques."""

    def test_get_partial_messages_initially_empty(self):
        """Returns empty list initially."""
        phase = DebateRoundsPhase()
        assert phase.get_partial_messages() == []

    def test_get_partial_critiques_initially_empty(self):
        """Returns empty list initially."""
        phase = DebateRoundsPhase()
        assert phase.get_partial_critiques() == []


# =============================================================================
# _build_final_synthesis_prompt Tests
# =============================================================================


class TestBuildFinalSynthesisPrompt:
    """Tests for _build_final_synthesis_prompt method."""

    def test_includes_agent_name(self):
        """Prompt includes the agent's name."""
        phase = DebateRoundsPhase()
        agent = MockAgent(name="claude-opus")

        prompt = phase._build_final_synthesis_prompt(
            agent=agent,
            current_proposal="My proposal",
            all_proposals={"claude-opus": "My proposal", "gpt-4": "Other proposal"},
            critiques=[],
            round_num=7,
        )

        assert "claude-opus" in prompt
        assert "ROUND 7: FINAL SYNTHESIS" in prompt

    def test_includes_other_proposals(self):
        """Prompt includes other agents' proposals."""
        phase = DebateRoundsPhase()
        agent = MockAgent(name="claude")

        prompt = phase._build_final_synthesis_prompt(
            agent=agent,
            current_proposal="My proposal",
            all_proposals={"claude": "My proposal", "gpt": "GPT proposal text"},
            critiques=[],
            round_num=7,
        )

        assert "GPT proposal text" in prompt

    def test_handles_empty_current_proposal(self):
        """Handles empty current proposal gracefully."""
        phase = DebateRoundsPhase()
        agent = MockAgent(name="claude")

        prompt = phase._build_final_synthesis_prompt(
            agent=agent,
            current_proposal="",
            all_proposals={},
            critiques=[],
            round_num=7,
        )

        assert "(No previous proposal)" in prompt


# =============================================================================
# _should_terminate Tests
# =============================================================================


class TestShouldTerminate:
    """Tests for _should_terminate method."""

    @pytest.mark.asyncio
    async def test_returns_false_when_no_callbacks(self):
        """Returns False when no termination callbacks are set."""
        phase = DebateRoundsPhase()
        ctx = MockDebateContext(proposals={"agent1": "proposal"})

        result = await phase._should_terminate(ctx, round_num=1)
        assert result is False

    @pytest.mark.asyncio
    async def test_terminates_on_judge_says_stop(self):
        """Returns True when judge callback says to stop."""

        async def judge_check(round_num, proposals, messages):
            return (False, "Debate has reached conclusion")

        phase = DebateRoundsPhase(check_judge_termination=judge_check)
        ctx = MockDebateContext(proposals={"agent1": "proposal"})

        result = await phase._should_terminate(ctx, round_num=2)
        assert result is True

    @pytest.mark.asyncio
    async def test_terminates_on_early_stopping(self):
        """Returns True when early stopping callback says to stop."""

        async def early_stop(round_num, proposals, messages):
            return False  # False means don't continue

        phase = DebateRoundsPhase(check_early_stopping=early_stop)
        ctx = MockDebateContext(proposals={"agent1": "proposal"})

        result = await phase._should_terminate(ctx, round_num=2)
        assert result is True

    @pytest.mark.asyncio
    async def test_continues_when_judge_says_continue(self):
        """Returns False when judge callback says to continue."""

        async def judge_check(round_num, proposals, messages):
            return (True, "Continue debating")

        phase = DebateRoundsPhase(check_judge_termination=judge_check)
        ctx = MockDebateContext(proposals={"agent1": "proposal"})

        result = await phase._should_terminate(ctx, round_num=2)
        assert result is False


# =============================================================================
# _refresh_evidence_for_round Tests
# =============================================================================


class TestRefreshEvidenceForRound:
    """Tests for _refresh_evidence_for_round method."""

    @pytest.mark.asyncio
    async def test_noop_without_callback(self):
        """Does nothing when no refresh_evidence callback is set."""
        phase = DebateRoundsPhase()
        ctx = MockDebateContext(proposals={"agent1": "proposal"})

        # Should not raise
        await phase._refresh_evidence_for_round(ctx, round_num=1)

    @pytest.mark.asyncio
    async def test_skips_even_rounds(self):
        """Skips evidence refresh on even rounds to avoid API overload."""
        refresh_fn = AsyncMock(return_value=3)
        phase = DebateRoundsPhase(refresh_evidence=refresh_fn)
        ctx = MockDebateContext(proposals={"agent1": "proposal"})

        await phase._refresh_evidence_for_round(ctx, round_num=2)
        refresh_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_calls_refresh_on_odd_rounds(self):
        """Calls refresh callback on odd rounds."""
        refresh_fn = AsyncMock(return_value=5)
        phase = DebateRoundsPhase(refresh_evidence=refresh_fn)
        ctx = MockDebateContext(proposals={"agent1": "Some proposal text"})

        await phase._refresh_evidence_for_round(ctx, round_num=1)
        refresh_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_refresh_exception(self):
        """Handles exceptions in the refresh callback gracefully."""
        refresh_fn = AsyncMock(side_effect=RuntimeError("refresh failed"))
        phase = DebateRoundsPhase(refresh_evidence=refresh_fn)
        ctx = MockDebateContext(proposals={"agent1": "proposal"})

        # Should not raise
        await phase._refresh_evidence_for_round(ctx, round_num=1)


# =============================================================================
# _compress_debate_context Tests
# =============================================================================


class TestCompressDebateContext:
    """Tests for _compress_debate_context method."""

    @pytest.mark.asyncio
    async def test_noop_without_callback(self):
        """Does nothing when no compress_context callback is set."""
        phase = DebateRoundsPhase()
        ctx = MockDebateContext()

        # Should not raise
        await phase._compress_debate_context(ctx, round_num=4)

    @pytest.mark.asyncio
    async def test_skips_when_few_messages(self):
        """Skips compression when context has fewer than 10 messages."""
        compress_fn = AsyncMock()
        phase = DebateRoundsPhase(compress_context=compress_fn)
        ctx = MockDebateContext(context_messages=[MagicMock() for _ in range(5)])

        await phase._compress_debate_context(ctx, round_num=4)
        compress_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_compresses_long_context(self):
        """Compresses context when there are enough messages."""
        original_msgs = [MagicMock() for _ in range(15)]
        compressed_msgs = [MagicMock() for _ in range(5)]
        compressed_crits = []

        compress_fn = AsyncMock(return_value=(compressed_msgs, compressed_crits))
        phase = DebateRoundsPhase(compress_context=compress_fn)
        ctx = MockDebateContext(context_messages=list(original_msgs))

        await phase._compress_debate_context(ctx, round_num=4)

        compress_fn.assert_called_once()
        assert len(ctx.context_messages) == 5


# =============================================================================
# execute Tests (Integration)
# =============================================================================


class TestExecute:
    """Tests for the execute method (main entry point)."""

    @pytest.mark.asyncio
    async def test_execute_runs_rounds(self):
        """Execute runs the specified number of rounds."""
        protocol = MockProtocol(rounds=2)
        convergence_tracker = MagicMock()
        convergence_tracker.check_convergence.return_value = MockConvergenceResult(
            converged=False, blocked_by_trickster=False
        )
        convergence_tracker.track_novelty = MagicMock()
        convergence_tracker.check_rlm_ready_quorum.return_value = False

        critique_fn = AsyncMock(return_value=MockCritique(agent="critic-1", target_agent="agent-1"))
        generate_fn = AsyncMock(return_value="revised proposal")
        build_revision = MagicMock(return_value="revision prompt")

        agent1 = MockAgent(name="agent-1", role="proposer")
        ctx = MockDebateContext(
            agents=[agent1],
            proposers=[agent1],
            proposals={"agent-1": "initial proposal"},
            result=MockResult(critiques=[]),
        )

        phase = DebateRoundsPhase(
            protocol=protocol,
            critique_with_agent=critique_fn,
            generate_with_agent=generate_fn,
            build_revision_prompt=build_revision,
        )
        # Replace convergence tracker
        phase._convergence_tracker = convergence_tracker

        with (
            patch("aragora.debate.phases.debate_rounds.get_debate_monitor") as mock_mon,
            patch("aragora.debate.phases.debate_rounds.get_complexity_governor") as mock_gov,
        ):
            mock_perf = MagicMock()
            mock_perf.track_round = MagicMock(side_effect=lambda *a, **kw: _noop_cm())
            mock_perf.track_phase = MagicMock(side_effect=lambda *a, **kw: _noop_cm())
            mock_perf.slow_round_threshold = 60.0
            mock_mon.return_value = mock_perf
            mock_gov.return_value.get_scaled_timeout.return_value = 30.0

            await phase.execute(ctx)

        assert ctx.result.rounds_used == 2

    @pytest.mark.asyncio
    async def test_execute_exits_on_convergence(self):
        """Execute exits early when convergence is detected."""
        protocol = MockProtocol(rounds=5)
        convergence_tracker = MagicMock()
        # Converge on round 2
        convergence_tracker.check_convergence.side_effect = [
            MockConvergenceResult(converged=True, blocked_by_trickster=False),
        ]
        convergence_tracker.track_novelty = MagicMock()
        convergence_tracker.check_rlm_ready_quorum.return_value = False

        agent1 = MockAgent(name="agent-1", role="proposer")
        ctx = MockDebateContext(
            agents=[agent1],
            proposers=[agent1],
            proposals={"agent-1": "initial proposal"},
            result=MockResult(critiques=[]),
        )

        phase = DebateRoundsPhase(
            protocol=protocol,
            critique_with_agent=AsyncMock(
                return_value=MockCritique(agent="agent-1", target_agent="agent-1")
            ),
            generate_with_agent=AsyncMock(return_value="revised"),
            build_revision_prompt=MagicMock(return_value="prompt"),
        )
        phase._convergence_tracker = convergence_tracker

        with (
            patch("aragora.debate.phases.debate_rounds.get_debate_monitor") as mock_mon,
            patch("aragora.debate.phases.debate_rounds.get_complexity_governor") as mock_gov,
        ):
            mock_perf = MagicMock()
            mock_perf.track_round = MagicMock(side_effect=lambda *a, **kw: _noop_cm())
            mock_perf.track_phase = MagicMock(side_effect=lambda *a, **kw: _noop_cm())
            mock_perf.slow_round_threshold = 60.0
            mock_mon.return_value = mock_perf
            mock_gov.return_value.get_scaled_timeout.return_value = 30.0

            await phase.execute(ctx)

        # Should exit after round 1 since convergence detected
        assert ctx.result.rounds_used == 1

    @pytest.mark.asyncio
    async def test_execute_respects_budget_check(self):
        """Execute stops when budget check callback denies continuation."""
        protocol = MockProtocol(rounds=5)

        agent1 = MockAgent(name="agent-1", role="proposer")
        ctx = MockDebateContext(
            agents=[agent1],
            proposers=[agent1],
            proposals={"agent-1": "initial proposal"},
            result=MockResult(critiques=[]),
            budget_check_callback=lambda round_num: (False, "Budget exceeded"),
        )

        phase = DebateRoundsPhase(protocol=protocol)

        with patch("aragora.debate.phases.debate_rounds.get_debate_monitor") as mock_mon:
            mock_perf = MagicMock()
            mock_perf.track_round = MagicMock(side_effect=lambda *a, **kw: _noop_cm())
            mock_perf.track_phase = MagicMock(side_effect=lambda *a, **kw: _noop_cm())
            mock_perf.slow_round_threshold = 60.0
            mock_mon.return_value = mock_perf

            await phase.execute(ctx)

        # Should not have executed any rounds
        assert ctx.result.rounds_used == 0
        assert ctx.result.metadata.get("budget_pause_reason") == "Budget exceeded"

    @pytest.mark.asyncio
    async def test_execute_with_cancellation_token(self):
        """Execute raises DebateCancelled when cancellation token is set."""
        protocol = MockProtocol(rounds=3)

        cancel_token = MagicMock()
        cancel_token.is_cancelled = True
        cancel_token.reason = "User cancelled"

        ctx = MockDebateContext(
            agents=[MockAgent(name="agent-1")],
            proposers=[MockAgent(name="agent-1")],
            proposals={"agent-1": "initial proposal"},
            result=MockResult(critiques=[]),
            cancellation_token=cancel_token,
        )

        phase = DebateRoundsPhase(protocol=protocol)

        with patch("aragora.debate.phases.debate_rounds.get_debate_monitor") as mock_mon:
            mock_perf = MagicMock()
            mock_perf.track_round = MagicMock(side_effect=lambda *a, **kw: _noop_cm())
            mock_mon.return_value = mock_perf

            with pytest.raises(Exception, match="User cancelled|cancelled"):
                await phase.execute(ctx)

    @pytest.mark.asyncio
    async def test_execute_uses_default_single_round_without_protocol(self):
        """Execute defaults to 1 round when protocol is None."""
        convergence_tracker = MagicMock()
        convergence_tracker.check_convergence.return_value = MockConvergenceResult(
            converged=False, blocked_by_trickster=False
        )
        convergence_tracker.track_novelty = MagicMock()
        convergence_tracker.check_rlm_ready_quorum.return_value = False

        agent1 = MockAgent(name="agent-1", role="proposer")
        ctx = MockDebateContext(
            agents=[agent1],
            proposers=[agent1],
            proposals={"agent-1": "initial proposal"},
            result=MockResult(critiques=[]),
        )

        phase = DebateRoundsPhase(
            protocol=None,  # No protocol
            critique_with_agent=AsyncMock(return_value=None),
        )
        phase._convergence_tracker = convergence_tracker

        with (
            patch("aragora.debate.phases.debate_rounds.get_debate_monitor") as mock_mon,
            patch("aragora.debate.phases.debate_rounds.get_complexity_governor") as mock_gov,
        ):
            mock_perf = MagicMock()
            mock_perf.track_round = MagicMock(side_effect=lambda *a, **kw: _noop_cm())
            mock_perf.track_phase = MagicMock(side_effect=lambda *a, **kw: _noop_cm())
            mock_perf.slow_round_threshold = 60.0
            mock_mon.return_value = mock_perf
            mock_gov.return_value.get_scaled_timeout.return_value = 30.0

            await phase.execute(ctx)

        assert ctx.result.rounds_used == 1


# =============================================================================
# _critique_phase Tests
# =============================================================================


class TestCritiquePhase:
    """Tests for _critique_phase method."""

    @pytest.mark.asyncio
    async def test_skips_when_no_callback(self):
        """Skips critiques when no critique_with_agent callback is set."""
        phase = DebateRoundsPhase()
        ctx = MockDebateContext(
            proposals={"agent-1": "proposal"},
            result=MockResult(),
        )

        # Should not raise
        await phase._critique_phase(ctx, critics=[], round_num=1)

    @pytest.mark.asyncio
    async def test_skips_empty_proposals(self):
        """Skips empty or placeholder proposals."""
        critique_fn = AsyncMock()
        phase = DebateRoundsPhase(critique_with_agent=critique_fn)

        agent1 = MockAgent(name="agent-1", role="critic")
        ctx = MockDebateContext(
            proposals={
                "agent-1": "(Agent produced empty output)",
                "agent-2": "",
            },
            result=MockResult(),
        )

        await phase._critique_phase(ctx, critics=[agent1], round_num=1)
        critique_fn.assert_not_called()


# =============================================================================
# _revision_phase Tests
# =============================================================================


class TestRevisionPhase:
    """Tests for _revision_phase method."""

    @pytest.mark.asyncio
    async def test_skips_when_no_callbacks(self):
        """Skips revisions when required callbacks are missing."""
        phase = DebateRoundsPhase()
        ctx = MockDebateContext(
            proposals={"agent-1": "proposal"},
            result=MockResult(),
        )

        # Should not raise
        await phase._revision_phase(ctx, critics=[], round_num=1)

    @pytest.mark.asyncio
    async def test_skips_when_no_critiques(self):
        """Skips revisions when there are no critiques."""
        phase = DebateRoundsPhase(
            generate_with_agent=AsyncMock(return_value="revised"),
            build_revision_prompt=MagicMock(return_value="prompt"),
        )
        ctx = MockDebateContext(
            proposals={"agent-1": "proposal"},
            result=MockResult(critiques=[]),
            proposers=[MockAgent(name="agent-1")],
        )

        await phase._revision_phase(ctx, critics=[], round_num=1)
        phase._generate_with_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_generates_revisions_for_proposers(self):
        """Generates revisions for proposers who received critiques."""
        generate_fn = AsyncMock(return_value="revised proposal")
        build_fn = MagicMock(return_value="revision prompt")

        phase = DebateRoundsPhase(
            generate_with_agent=generate_fn,
            build_revision_prompt=build_fn,
        )

        agent1 = MockAgent(name="agent-1", role="proposer")
        critique = MockCritique(agent="critic-1", target_agent="agent-1")

        ctx = MockDebateContext(
            proposals={"agent-1": "initial proposal"},
            result=MockResult(critiques=[critique]),
            proposers=[agent1],
            context_messages=[],
        )

        with (
            patch("aragora.debate.phases.debate_rounds.get_complexity_governor") as mock_gov,
            patch("aragora.debate.phases.debate_rounds.AGENT_TIMEOUT_SECONDS", 30.0),
        ):
            mock_gov.return_value.get_scaled_timeout.return_value = 30.0
            await phase._revision_phase(ctx, critics=[], round_num=1)

        # Proposal should be updated
        assert ctx.proposals["agent-1"] == "revised proposal"
        build_fn.assert_called_once()
