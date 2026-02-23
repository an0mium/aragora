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
    metadata: dict | None = field(default_factory=dict)


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


# =============================================================================
# Additional Coverage: Phase Transitions Tests
# =============================================================================


class TestPhaseTransitions:
    """Tests for phase transitions during debate rounds."""

    @pytest.mark.asyncio
    async def test_execute_round_triggers_pre_round_hook(self):
        """Execute round triggers PRE_ROUND hook when hook_manager is present."""
        protocol = MockProtocol(rounds=1)
        hook_manager = AsyncMock()

        agent1 = MockAgent(name="agent-1", role="proposer")
        ctx = MockDebateContext(
            agents=[agent1],
            proposers=[agent1],
            proposals={"agent-1": "proposal"},
            result=MockResult(critiques=[]),
            hook_manager=hook_manager,
        )

        phase = DebateRoundsPhase(
            protocol=protocol,
            critique_with_agent=AsyncMock(return_value=None),
        )

        convergence_tracker = MagicMock()
        convergence_tracker.check_convergence.return_value = MockConvergenceResult(
            converged=False, blocked_by_trickster=False
        )
        convergence_tracker.track_novelty = MagicMock()
        convergence_tracker.check_rlm_ready_quorum.return_value = False
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

        # Hook manager should have been called with pre_round
        hook_manager.trigger.assert_any_call("pre_round", ctx=ctx, round_num=1)

    @pytest.mark.asyncio
    async def test_execute_round_triggers_post_round_hook(self):
        """Execute round triggers POST_ROUND hook when hook_manager is present."""
        protocol = MockProtocol(rounds=1)
        hook_manager = AsyncMock()

        agent1 = MockAgent(name="agent-1", role="proposer")
        ctx = MockDebateContext(
            agents=[agent1],
            proposers=[agent1],
            proposals={"agent-1": "proposal"},
            result=MockResult(critiques=[]),
            hook_manager=hook_manager,
        )

        phase = DebateRoundsPhase(
            protocol=protocol,
            critique_with_agent=AsyncMock(return_value=None),
        )

        convergence_tracker = MagicMock()
        convergence_tracker.check_convergence.return_value = MockConvergenceResult(
            converged=False, blocked_by_trickster=False
        )
        convergence_tracker.track_novelty = MagicMock()
        convergence_tracker.check_rlm_ready_quorum.return_value = False
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

        # Hook manager should have been called with post_round
        call_names = [call[0][0] for call in hook_manager.trigger.call_args_list]
        assert "post_round" in call_names

    @pytest.mark.asyncio
    async def test_execute_calls_checkpoint_callback(self):
        """Execute calls checkpoint callback after each round."""
        protocol = MockProtocol(rounds=2)
        checkpoint_callback = AsyncMock()

        agent1 = MockAgent(name="agent-1", role="proposer")
        ctx = MockDebateContext(
            agents=[agent1],
            proposers=[agent1],
            proposals={"agent-1": "proposal"},
            result=MockResult(critiques=[]),
        )

        phase = DebateRoundsPhase(
            protocol=protocol,
            critique_with_agent=AsyncMock(return_value=None),
            checkpoint_callback=checkpoint_callback,
        )

        convergence_tracker = MagicMock()
        convergence_tracker.check_convergence.return_value = MockConvergenceResult(
            converged=False, blocked_by_trickster=False
        )
        convergence_tracker.track_novelty = MagicMock()
        convergence_tracker.check_rlm_ready_quorum.return_value = False
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

        # Checkpoint should be called after each round
        assert checkpoint_callback.call_count == 2


# =============================================================================
# Additional Coverage: Agent Interaction Tests
# =============================================================================


class TestAgentInteraction:
    """Tests for agent interaction during rounds."""

    @pytest.mark.asyncio
    async def test_critique_phase_with_selected_critics(self):
        """Critique phase uses select_critics_for_proposal callback."""
        critique_fn = AsyncMock(return_value=MockCritique(agent="selected-critic"))
        select_critics_fn = MagicMock(return_value=[MockAgent(name="selected-critic")])

        phase = DebateRoundsPhase(
            critique_with_agent=critique_fn,
            select_critics_for_proposal=select_critics_fn,
        )

        agent1 = MockAgent(name="agent-1")
        critic1 = MockAgent(name="selected-critic", role="critic")
        ctx = MockDebateContext(
            proposals={"agent-1": "proposal content"},
            result=MockResult(),
        )

        await phase._critique_phase(ctx, critics=[critic1], round_num=1)

        select_critics_fn.assert_called_once()
        critique_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_critique_phase_records_circuit_breaker_success(self):
        """Critique phase records success in circuit breaker."""
        cb = MagicMock()
        critique_fn = AsyncMock(return_value=MockCritique(agent="critic-1"))

        phase = DebateRoundsPhase(
            circuit_breaker=cb,
            critique_with_agent=critique_fn,
        )

        critic = MockAgent(name="critic-1", role="critic")
        ctx = MockDebateContext(
            proposals={"agent-1": "proposal"},
            result=MockResult(),
        )

        await phase._critique_phase(ctx, critics=[critic], round_num=1)

        cb.record_success.assert_called_with("critic-1")

    @pytest.mark.asyncio
    async def test_critique_phase_records_circuit_breaker_failure_on_none(self):
        """Critique phase records failure in circuit breaker when critique returns None."""
        cb = MagicMock()
        critique_fn = AsyncMock(return_value=None)

        phase = DebateRoundsPhase(
            circuit_breaker=cb,
            critique_with_agent=critique_fn,
        )

        critic = MockAgent(name="critic-1", role="critic")
        ctx = MockDebateContext(
            proposals={"agent-1": "proposal"},
            result=MockResult(),
        )

        await phase._critique_phase(ctx, critics=[critic], round_num=1)

        cb.record_failure.assert_called_with("critic-1")

    @pytest.mark.asyncio
    async def test_revision_phase_records_circuit_breaker_failure_on_error(self):
        """Revision phase records failure in circuit breaker on exception."""
        cb = MagicMock()
        generate_fn = AsyncMock(side_effect=RuntimeError("Agent error"))
        build_fn = MagicMock(return_value="prompt")

        phase = DebateRoundsPhase(
            circuit_breaker=cb,
            generate_with_agent=generate_fn,
            build_revision_prompt=build_fn,
        )

        agent1 = MockAgent(name="agent-1", role="proposer")
        critique = MockCritique(agent="critic-1", target_agent="agent-1")

        ctx = MockDebateContext(
            proposals={"agent-1": "proposal"},
            result=MockResult(critiques=[critique]),
            proposers=[agent1],
        )

        with (
            patch("aragora.debate.phases.debate_rounds.get_complexity_governor") as mock_gov,
            patch("aragora.debate.phases.debate_rounds.AGENT_TIMEOUT_SECONDS", 30.0),
        ):
            mock_gov.return_value.get_scaled_timeout.return_value = 30.0
            await phase._revision_phase(ctx, critics=[], round_num=1)

        cb.record_failure.assert_called_with("agent-1")


# =============================================================================
# Additional Coverage: Timeout Handling Tests
# =============================================================================


class TestTimeoutHandling:
    """Tests for timeout handling during debate rounds."""

    @pytest.mark.asyncio
    async def test_revision_phase_timeout_handling(self):
        """Revision phase handles phase-level timeout gracefully."""
        generate_fn = AsyncMock(side_effect=asyncio.TimeoutError())
        build_fn = MagicMock(return_value="prompt")

        phase = DebateRoundsPhase(
            generate_with_agent=generate_fn,
            build_revision_prompt=build_fn,
        )

        agent1 = MockAgent(name="agent-1", role="proposer")
        critique = MockCritique(agent="critic-1", target_agent="agent-1")

        ctx = MockDebateContext(
            proposals={"agent-1": "proposal"},
            result=MockResult(critiques=[critique]),
            proposers=[agent1],
        )

        with (
            patch("aragora.debate.phases.debate_rounds.get_complexity_governor") as mock_gov,
            patch("aragora.debate.phases.debate_rounds.AGENT_TIMEOUT_SECONDS", 30.0),
        ):
            mock_gov.return_value.get_scaled_timeout.return_value = 30.0
            await phase._revision_phase(ctx, critics=[], round_num=1)

        # Proposal should remain unchanged
        assert ctx.proposals["agent-1"] == "proposal"

    @pytest.mark.asyncio
    async def test_with_callback_timeout_handles_timeout(self):
        """_with_callback_timeout returns default on timeout."""

        async def slow_coro():
            await asyncio.sleep(10)
            return "completed"

        result = await _with_callback_timeout(slow_coro(), timeout=0.01, default="fallback")
        # TimeoutError triggers the default
        assert result == "fallback"


# =============================================================================
# Additional Coverage: Final Synthesis Tests
# =============================================================================


class TestFinalSynthesis:
    """Tests for final synthesis round execution."""

    @pytest.mark.asyncio
    async def test_execute_final_synthesis_with_no_generate_callback(self):
        """Final synthesis handles missing generate callback gracefully."""
        protocol = MockProtocol(rounds=7, use_structured_phases=True)

        # Mock get_round_phase to return a Final Synthesis phase
        mock_phase = MagicMock()
        mock_phase.name = "Final Synthesis"
        protocol.get_round_phase = MagicMock(return_value=mock_phase)

        agent1 = MockAgent(name="agent-1", role="proposer")
        ctx = MockDebateContext(
            agents=[agent1],
            proposers=[agent1],
            proposals={"agent-1": "proposal"},
            result=MockResult(critiques=[]),
        )

        phase = DebateRoundsPhase(
            protocol=protocol,
            # No generate_with_agent callback
        )

        convergence_tracker = MagicMock()
        convergence_tracker.check_convergence.return_value = MockConvergenceResult(
            converged=False, blocked_by_trickster=False
        )
        convergence_tracker.track_novelty = MagicMock()
        convergence_tracker.check_rlm_ready_quorum.return_value = False
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

        # Should have executed without error
        assert ctx.result.rounds_used == 7

    @pytest.mark.asyncio
    async def test_execute_final_synthesis_with_circuit_breaker(self):
        """Final synthesis filters agents through circuit breaker."""
        protocol = MockProtocol(rounds=7, use_structured_phases=True)
        mock_phase_obj = MagicMock()
        mock_phase_obj.name = "Final Synthesis"
        protocol.get_round_phase = MagicMock(return_value=mock_phase_obj)

        cb = MagicMock()
        generate_fn = AsyncMock(return_value="final synthesis")

        agent1 = MockAgent(name="agent-1", role="proposer")
        agent2 = MockAgent(name="agent-2", role="proposer")
        cb.filter_available_agents.return_value = [agent1]

        ctx = MockDebateContext(
            agents=[agent1, agent2],
            proposers=[agent1, agent2],
            proposals={"agent-1": "proposal1", "agent-2": "proposal2"},
            result=MockResult(critiques=[]),
        )

        phase = DebateRoundsPhase(
            protocol=protocol,
            circuit_breaker=cb,
            generate_with_agent=generate_fn,
        )

        convergence_tracker = MagicMock()
        convergence_tracker.check_convergence.return_value = MockConvergenceResult(
            converged=False, blocked_by_trickster=False
        )
        convergence_tracker.track_novelty = MagicMock()
        convergence_tracker.check_rlm_ready_quorum.return_value = False
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

        # Only agent-1 should have been processed
        cb.filter_available_agents.assert_called()


# =============================================================================
# Additional Coverage: RLM Ready Signal Tests
# =============================================================================


class TestRLMReadySignal:
    """Tests for RLM ready signal quorum checking."""

    @pytest.mark.asyncio
    async def test_should_terminate_on_rlm_ready_quorum(self):
        """Returns True when RLM ready quorum is met."""
        convergence_tracker = MagicMock()
        convergence_tracker.check_rlm_ready_quorum.return_value = True

        phase = DebateRoundsPhase()
        phase._convergence_tracker = convergence_tracker

        ctx = MockDebateContext(proposals={"agent-1": "proposal"})

        result = await phase._should_terminate(ctx, round_num=2)

        assert result is True
        convergence_tracker.check_rlm_ready_quorum.assert_called_once()


# =============================================================================
# Additional Coverage: Strategy-Based Rounds Tests
# =============================================================================


class TestStrategyBasedRounds:
    """Tests for debate strategy-based round estimation."""

    @pytest.mark.asyncio
    async def test_execute_uses_debate_strategy(self):
        """Execute uses debate strategy for round estimation."""
        protocol = MockProtocol(rounds=3)
        strategy = MagicMock()
        strategy_rec = MagicMock()
        strategy_rec.estimated_rounds = 5
        strategy_rec.confidence = 0.8
        strategy_rec.reasoning = "Complex topic requires more rounds"
        strategy_rec.relevant_memories = ["mem1", "mem2"]
        strategy.estimate_rounds_async = AsyncMock(return_value=strategy_rec)

        agent1 = MockAgent(name="agent-1", role="proposer")
        ctx = MockDebateContext(
            agents=[agent1],
            proposers=[agent1],
            proposals={"agent-1": "proposal"},
            result=MockResult(critiques=[], metadata={}),
        )

        convergence_tracker = MagicMock()
        convergence_tracker.check_convergence.return_value = MockConvergenceResult(
            converged=True, blocked_by_trickster=False
        )
        convergence_tracker.track_novelty = MagicMock()
        convergence_tracker.check_rlm_ready_quorum.return_value = False

        phase = DebateRoundsPhase(
            protocol=protocol,
            debate_strategy=strategy,
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

        # Strategy should have been called
        strategy.estimate_rounds_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_handles_strategy_error(self):
        """Execute handles debate strategy errors gracefully."""
        protocol = MockProtocol(rounds=3)
        strategy = MagicMock()
        strategy.estimate_rounds_async = AsyncMock(side_effect=RuntimeError("Strategy error"))

        agent1 = MockAgent(name="agent-1", role="proposer")
        ctx = MockDebateContext(
            agents=[agent1],
            proposers=[agent1],
            proposals={"agent-1": "proposal"},
            result=MockResult(critiques=[]),
        )

        convergence_tracker = MagicMock()
        convergence_tracker.check_convergence.return_value = MockConvergenceResult(
            converged=True, blocked_by_trickster=False
        )
        convergence_tracker.track_novelty = MagicMock()
        convergence_tracker.check_rlm_ready_quorum.return_value = False

        phase = DebateRoundsPhase(
            protocol=protocol,
            debate_strategy=strategy,
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

            # Should not raise
            await phase.execute(ctx)

        # Default rounds should be used
        assert ctx.result.rounds_used == 1


# =============================================================================
# Additional Coverage: Propulsion Events Tests
# =============================================================================


class TestPropulsionEvents:
    """Tests for propulsion event firing."""

    @pytest.mark.asyncio
    async def test_fire_propulsion_event_disabled(self):
        """Propulsion events are not fired when disabled."""
        phase = DebateRoundsPhase(enable_propulsion=False)
        ctx = MockDebateContext()

        # Should not raise
        await phase._fire_propulsion_event("test_event", ctx, 1, {})

    @pytest.mark.asyncio
    async def test_fire_propulsion_event_without_engine(self):
        """Propulsion events are not fired without engine."""
        phase = DebateRoundsPhase(enable_propulsion=True, propulsion_engine=None)
        ctx = MockDebateContext()

        # Should not raise
        await phase._fire_propulsion_event("test_event", ctx, 1, {})


# =============================================================================
# Additional Coverage: Novelty Tracking Tests
# =============================================================================


class TestNoveltyTracking:
    """Tests for novelty tracking during rounds."""

    @pytest.mark.asyncio
    async def test_execute_tracks_initial_novelty(self):
        """Execute tracks novelty for initial proposals."""
        protocol = MockProtocol(rounds=1)
        novelty_tracker = MagicMock()

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
            proposals={"agent-1": "proposal"},
            result=MockResult(critiques=[]),
        )

        phase = DebateRoundsPhase(
            protocol=protocol,
            novelty_tracker=novelty_tracker,
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

        # Novelty should be tracked for round 0 and round 1
        assert convergence_tracker.track_novelty.call_count >= 1


# =============================================================================
# Additional Coverage: Error Recovery Tests
# =============================================================================


class TestErrorRecovery:
    """Tests for error recovery during debate rounds."""

    @pytest.mark.asyncio
    async def test_critique_phase_handles_task_exception(self):
        """Critique phase handles exceptions in critique tasks gracefully."""
        critique_fn = AsyncMock(side_effect=RuntimeError("Critique error"))

        phase = DebateRoundsPhase(critique_with_agent=critique_fn)

        critic = MockAgent(name="critic-1", role="critic")
        ctx = MockDebateContext(
            proposals={"agent-1": "proposal"},
            result=MockResult(),
        )

        # Should not raise
        await phase._critique_phase(ctx, critics=[critic], round_num=1)

    @pytest.mark.asyncio
    async def test_revision_phase_continues_on_single_agent_failure(self):
        """Revision phase continues when single agent fails."""
        call_count = 0

        async def generate_with_count(agent, prompt, messages):
            nonlocal call_count
            call_count += 1
            if agent.name == "agent-1":
                raise RuntimeError("Agent 1 failed")
            return "revised proposal"

        build_fn = MagicMock(return_value="prompt")

        phase = DebateRoundsPhase(
            generate_with_agent=generate_with_count,
            build_revision_prompt=build_fn,
        )

        agent1 = MockAgent(name="agent-1", role="proposer")
        agent2 = MockAgent(name="agent-2", role="proposer")
        critique1 = MockCritique(agent="critic", target_agent="agent-1")
        critique2 = MockCritique(agent="critic", target_agent="agent-2")

        ctx = MockDebateContext(
            proposals={"agent-1": "proposal1", "agent-2": "proposal2"},
            result=MockResult(critiques=[critique1, critique2]),
            proposers=[agent1, agent2],
        )

        with (
            patch("aragora.debate.phases.debate_rounds.get_complexity_governor") as mock_gov,
            patch("aragora.debate.phases.debate_rounds.AGENT_TIMEOUT_SECONDS", 30.0),
        ):
            mock_gov.return_value.get_scaled_timeout.return_value = 30.0
            await phase._revision_phase(ctx, critics=[], round_num=1)

        # Both agents should have been attempted
        assert call_count == 2
        # Agent-2 should have been updated
        assert ctx.proposals["agent-2"] == "revised proposal"

    @pytest.mark.asyncio
    async def test_evidence_refresh_handles_exception(self):
        """Evidence refresh handles exceptions in refresh callback."""
        refresh_fn = AsyncMock(side_effect=RuntimeError("Refresh error"))
        phase = DebateRoundsPhase(refresh_evidence=refresh_fn)

        ctx = MockDebateContext(proposals={"agent-1": "proposal"})

        # Should not raise
        await phase._refresh_evidence_for_round(ctx, round_num=1)

    @pytest.mark.asyncio
    async def test_context_compression_handles_exception(self):
        """Context compression handles exceptions gracefully."""
        compress_fn = AsyncMock(side_effect=RuntimeError("Compression error"))
        phase = DebateRoundsPhase(compress_context=compress_fn)

        ctx = MockDebateContext(context_messages=[MagicMock() for _ in range(15)])

        # Should not raise
        await phase._compress_debate_context(ctx, round_num=4)


# =============================================================================
# Additional Coverage: Hook Emission Tests
# =============================================================================


class TestHookEmission:
    """Tests for hook emission during rounds."""

    def test_emit_heartbeat_with_exception(self):
        """Heartbeat emission handles hook exceptions."""
        hook = MagicMock(side_effect=RuntimeError("Hook failed"))
        phase = DebateRoundsPhase(hooks={"on_heartbeat": hook})

        # Should not raise
        phase._emit_heartbeat("round_1", "testing")

        hook.assert_called_once()

    @pytest.mark.asyncio
    async def test_critique_phase_emits_on_critique_hook(self):
        """Critique phase emits on_critique hook."""
        on_critique = MagicMock()
        critique_fn = AsyncMock(return_value=MockCritique(agent="critic-1"))

        phase = DebateRoundsPhase(
            critique_with_agent=critique_fn,
            hooks={"on_critique": on_critique},
        )

        critic = MockAgent(name="critic-1", role="critic")
        ctx = MockDebateContext(
            proposals={"agent-1": "proposal"},
            result=MockResult(),
        )

        await phase._critique_phase(ctx, critics=[critic], round_num=1)

        on_critique.assert_called_once()

    @pytest.mark.asyncio
    async def test_revision_phase_emits_on_message_hook(self):
        """Revision phase emits on_message hook."""
        on_message = MagicMock()
        generate_fn = AsyncMock(return_value="revised proposal")
        build_fn = MagicMock(return_value="prompt")

        phase = DebateRoundsPhase(
            generate_with_agent=generate_fn,
            build_revision_prompt=build_fn,
            hooks={"on_message": on_message},
        )

        agent1 = MockAgent(name="agent-1", role="proposer")
        critique = MockCritique(agent="critic", target_agent="agent-1")

        ctx = MockDebateContext(
            proposals={"agent-1": "proposal"},
            result=MockResult(critiques=[critique]),
            proposers=[agent1],
        )

        with (
            patch("aragora.debate.phases.debate_rounds.get_complexity_governor") as mock_gov,
            patch("aragora.debate.phases.debate_rounds.AGENT_TIMEOUT_SECONDS", 30.0),
        ):
            mock_gov.return_value.get_scaled_timeout.return_value = 30.0
            await phase._revision_phase(ctx, critics=[], round_num=1)

        on_message.assert_called_once()


# =============================================================================
# Speed Policy and Parallelism Bounds
# =============================================================================


class TestSpeedPolicy:
    """Tests for fast-first routing and protocol-level parallelism bounds."""

    @pytest.mark.asyncio
    async def test_critique_phase_enforces_protocol_parallelism_bound(self):
        """Critique semaphore should respect protocol.max_parallel_critiques."""
        protocol = MockProtocol(rounds=1)
        protocol.max_parallel_critiques = 2

        concurrent = 0
        max_seen = 0

        async def critique_fn(critic, proposal, task, context, target_agent=None):
            nonlocal concurrent, max_seen
            concurrent += 1
            max_seen = max(max_seen, concurrent)
            try:
                await asyncio.sleep(0.01)
                return MockCritique(agent=critic.name, target_agent=target_agent or "unknown")
            finally:
                concurrent -= 1

        phase = DebateRoundsPhase(
            protocol=protocol,
            critique_with_agent=critique_fn,
            select_critics_for_proposal=lambda _proposal_agent, critics: list(critics),
        )

        critics = [MockAgent(name=f"critic-{i}", role="critic") for i in range(4)]
        ctx = MockDebateContext(
            proposals={"agent-1": "proposal-1", "agent-2": "proposal-2"},
            result=MockResult(),
        )

        with patch("aragora.debate.phases.debate_rounds.get_complexity_governor") as mock_gov:
            mock_gov.return_value.get_scaled_timeout.return_value = 30.0
            await phase._critique_phase(ctx, critics=critics, round_num=1)

        assert max_seen <= 2

    @pytest.mark.asyncio
    async def test_revision_phase_enforces_protocol_parallelism_bound(self):
        """Revision semaphore should respect protocol.max_parallel_revisions."""
        protocol = MockProtocol(rounds=1)
        protocol.max_parallel_revisions = 1

        concurrent = 0
        max_seen = 0

        async def generate_fn(agent, prompt, messages):
            nonlocal concurrent, max_seen
            concurrent += 1
            max_seen = max(max_seen, concurrent)
            try:
                await asyncio.sleep(0.01)
                return f"revised-{agent.name}"
            finally:
                concurrent -= 1

        phase = DebateRoundsPhase(
            protocol=protocol,
            generate_with_agent=generate_fn,
            build_revision_prompt=MagicMock(return_value="revision-prompt"),
        )

        proposers = [MockAgent(name=f"agent-{i}", role="proposer") for i in range(3)]
        critiques = [
            MockCritique(agent="critic-a", target_agent="agent-0"),
            MockCritique(agent="critic-b", target_agent="agent-1"),
            MockCritique(agent="critic-c", target_agent="agent-2"),
        ]
        ctx = MockDebateContext(
            proposals={a.name: f"proposal-{a.name}" for a in proposers},
            result=MockResult(critiques=critiques),
            proposers=proposers,
            context_messages=[],
        )

        with (
            patch("aragora.debate.phases.debate_rounds.get_complexity_governor") as mock_gov,
            patch("aragora.debate.phases.debate_rounds.AGENT_TIMEOUT_SECONDS", 30.0),
        ):
            mock_gov.return_value.get_scaled_timeout.return_value = 30.0
            await phase._revision_phase(ctx, critics=[], round_num=1)

        assert max_seen == 1

    @pytest.mark.asyncio
    async def test_fast_first_limits_critics_per_proposal(self):
        """Low-contention fast-first mode should cap critics per proposal."""
        protocol = MockProtocol(rounds=1)
        protocol.fast_first_routing = True
        protocol.fast_first_min_round = 1
        protocol.fast_first_low_contention_agent_threshold = 3
        protocol.fast_first_max_critics_per_proposal = 1

        call_count = 0

        async def critique_fn(critic, proposal, task, context, target_agent=None):
            nonlocal call_count
            call_count += 1
            return MockCritique(agent=critic.name, target_agent=target_agent or "unknown")

        phase = DebateRoundsPhase(
            protocol=protocol,
            critique_with_agent=critique_fn,
            select_critics_for_proposal=lambda _proposal_agent, critics: list(critics),
        )
        critics = [MockAgent(name=f"critic-{i}", role="critic", timeout=10.0 + i) for i in range(3)]
        ctx = MockDebateContext(
            proposals={"agent-1": "proposal-1", "agent-2": "proposal-2"},
            result=MockResult(),
        )

        with patch("aragora.debate.phases.debate_rounds.get_complexity_governor") as mock_gov:
            mock_gov.return_value.get_scaled_timeout.return_value = 30.0
            await phase._critique_phase(ctx, critics=critics, round_num=1)

        assert call_count == 2  # 2 proposals * 1 critic per proposal

    @pytest.mark.asyncio
    async def test_fast_first_early_exit_skips_revision_when_low_contention(self):
        """Fast-first should early-exit before revision when convergence probe is strong."""
        protocol = MockProtocol(rounds=3)
        protocol.fast_first_routing = True
        protocol.fast_first_early_exit = True
        protocol.fast_first_min_round = 1
        protocol.fast_first_low_contention_agent_threshold = 2
        protocol.fast_first_max_total_issues = 0
        protocol.fast_first_max_critique_severity = 0.0
        protocol.fast_first_convergence_threshold = 0.8

        async def critique_fn(critic, proposal, task, context, target_agent=None):
            return MockCritique(
                agent=critic.name,
                target_agent=target_agent or "unknown",
                issues=[],
                severity=0.0,
            )

        generate_fn = AsyncMock(return_value="revised proposal")

        convergence_tracker = MagicMock()
        convergence_tracker.check_convergence.return_value = MockConvergenceResult(
            converged=False,
            blocked_by_trickster=False,
            similarity=0.95,
        )
        convergence_tracker.track_novelty = MagicMock()
        convergence_tracker.check_rlm_ready_quorum.return_value = False

        agent1 = MockAgent(name="agent-1", role="proposer")
        agent2 = MockAgent(name="agent-2", role="proposer")
        ctx = MockDebateContext(
            agents=[agent1, agent2],
            proposers=[agent1, agent2],
            proposals={"agent-1": "proposal-1", "agent-2": "proposal-2"},
            result=MockResult(critiques=[]),
        )

        phase = DebateRoundsPhase(
            protocol=protocol,
            critique_with_agent=critique_fn,
            generate_with_agent=generate_fn,
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

        assert ctx.result.rounds_used == 1
        assert generate_fn.await_count == 0
        assert isinstance(ctx.result.metadata, dict)
        assert "fast_first_early_exit" in ctx.result.metadata


# =============================================================================
# Early Stop Spectator Event Tests
# =============================================================================


class TestEarlyStopSpectatorEvents:
    """Tests for early-stop mechanism emitting spectator events.

    Verifies that when a debate terminates early (via any source: RLM ready,
    judge, agent vote, stability, or convergence), a spectator "early_stop"
    event is emitted and result metadata is enriched with termination details.
    """

    @pytest.mark.asyncio
    async def test_check_termination_conditions_returns_none_when_no_triggers(self):
        """Returns (None, '') when no termination condition is met."""
        convergence_tracker = MagicMock()
        convergence_tracker.check_rlm_ready_quorum.return_value = False

        phase = DebateRoundsPhase()
        phase._convergence_tracker = convergence_tracker
        # No judge or early stopping callbacks
        phase._check_judge_termination = None
        phase._check_early_stopping = None
        phase._stability_detector = None

        ctx = MockDebateContext(proposals={"agent-1": "proposal"})
        source, details = await phase._check_termination_conditions(ctx, round_num=2)

        assert source is None
        assert details == ""

    @pytest.mark.asyncio
    async def test_check_termination_conditions_rlm_ready(self):
        """Returns 'rlm_ready' source when RLM quorum is reached."""
        convergence_tracker = MagicMock()
        convergence_tracker.check_rlm_ready_quorum.return_value = True
        convergence_tracker.collective_readiness.avg_confidence = 0.92

        phase = DebateRoundsPhase()
        phase._convergence_tracker = convergence_tracker

        ctx = MockDebateContext(proposals={"agent-1": "proposal"})
        source, details = await phase._check_termination_conditions(ctx, round_num=3)

        assert source == "rlm_ready"
        assert "0.92" in details

    @pytest.mark.asyncio
    async def test_check_termination_conditions_judge(self):
        """Returns 'judge' source when judge says to stop."""
        convergence_tracker = MagicMock()
        convergence_tracker.check_rlm_ready_quorum.return_value = False

        phase = DebateRoundsPhase()
        phase._convergence_tracker = convergence_tracker
        phase._check_judge_termination = AsyncMock(return_value=(False, "Debate is conclusive"))
        phase._check_early_stopping = None
        phase._stability_detector = None

        ctx = MockDebateContext(proposals={"agent-1": "proposal"})
        source, details = await phase._check_termination_conditions(ctx, round_num=2)

        assert source == "judge"
        assert "conclusive" in details.lower()

    @pytest.mark.asyncio
    async def test_check_termination_conditions_agent_vote(self):
        """Returns 'agent_vote' source when agents vote to stop."""
        convergence_tracker = MagicMock()
        convergence_tracker.check_rlm_ready_quorum.return_value = False

        phase = DebateRoundsPhase()
        phase._convergence_tracker = convergence_tracker
        phase._check_judge_termination = None
        phase._check_early_stopping = AsyncMock(return_value=False)  # False = stop
        phase._stability_detector = None

        ctx = MockDebateContext(proposals={"agent-1": "proposal"})
        source, details = await phase._check_termination_conditions(ctx, round_num=2)

        assert source == "agent_vote"
        assert "voted" in details.lower()

    @pytest.mark.asyncio
    async def test_check_termination_conditions_stability(self):
        """Returns 'stability' source when vote distribution stabilizes."""
        convergence_tracker = MagicMock()
        convergence_tracker.check_rlm_ready_quorum.return_value = False

        stability_detector = MagicMock()
        stability_result = MagicMock()
        stability_result.recommendation = "stop"
        stability_result.stability_score = 0.95
        stability_result.ks_distance = 0.03
        stability_detector.update.return_value = stability_result

        phase = DebateRoundsPhase()
        phase._convergence_tracker = convergence_tracker
        phase._check_judge_termination = None
        phase._check_early_stopping = None
        phase._stability_detector = stability_detector

        ctx = MockDebateContext(proposals={"agent-1": "proposal"})
        ctx.round_votes = {"agent-1": "yes", "agent-2": "yes"}

        source, details = await phase._check_termination_conditions(ctx, round_num=3)

        assert source == "stability"
        assert "0.950" in details
        assert "0.030" in details

    def test_emit_early_stop_event_calls_spectator(self):
        """Emitting early stop event calls notify_spectator callback."""
        notify_spectator = MagicMock()
        protocol = MockProtocol(rounds=5)

        phase = DebateRoundsPhase(
            protocol=protocol,
            notify_spectator=notify_spectator,
        )

        ctx = MockDebateContext(debate_id="test-debate-456")

        phase._emit_early_stop_event(ctx, round_num=3, source="judge", details="Conclusive")

        notify_spectator.assert_called_once()
        call_args = notify_spectator.call_args
        assert call_args[0][0] == "early_stop"  # event_type
        assert "system" in str(call_args)
        assert "round 3/5" in call_args[1]["details"]
        assert "judge" in call_args[1]["details"]
        assert "saved 2 rounds" in call_args[1]["details"]

    def test_emit_early_stop_event_calls_event_emitter(self):
        """Emitting early stop event emits via EventEmitter for WebSocket."""
        event_emitter = MagicMock()
        protocol = MockProtocol(rounds=5)

        phase = DebateRoundsPhase(protocol=protocol, event_emitter=event_emitter)

        ctx = MockDebateContext(debate_id="test-debate-789")

        phase._emit_early_stop_event(ctx, round_num=2, source="stability", details="Stabilized")

        event_emitter.emit_sync.assert_called_once()
        call_kwargs = event_emitter.emit_sync.call_args[1]
        assert call_kwargs["event_type"] == "debate_early_terminated"
        assert call_kwargs["debate_id"] == "test-debate-789"
        assert call_kwargs["round_num"] == 2
        assert call_kwargs["total_rounds"] == 5
        assert call_kwargs["source"] == "stability"
        assert call_kwargs["rounds_saved"] == 3

    def test_emit_early_stop_event_calculates_rounds_saved(self):
        """Rounds saved is correctly computed as total - current."""
        notify_spectator = MagicMock()
        protocol = MockProtocol(rounds=10)

        phase = DebateRoundsPhase(
            protocol=protocol,
            notify_spectator=notify_spectator,
        )

        ctx = MockDebateContext()

        phase._emit_early_stop_event(ctx, round_num=7, source="agent_vote", details="Voted")

        call_args = notify_spectator.call_args
        assert "saved 3 rounds" in call_args[1]["details"]

    def test_emit_early_stop_event_no_crash_without_callbacks(self):
        """Emitting early stop event does not crash when no callbacks set."""
        protocol = MockProtocol(rounds=5)
        phase = DebateRoundsPhase(protocol=protocol)

        ctx = MockDebateContext()

        # Should not raise
        phase._emit_early_stop_event(ctx, round_num=3, source="judge", details="Done")

    @pytest.mark.asyncio
    async def test_execute_round_sets_metadata_on_early_termination(self):
        """Early termination enriches result metadata with source and reason."""
        protocol = MockProtocol(rounds=5)
        notify_spectator = MagicMock()

        convergence_tracker = MagicMock()
        convergence_tracker.check_rlm_ready_quorum.return_value = False
        convergence_tracker.check_convergence.return_value = MockConvergenceResult(
            converged=False, blocked_by_trickster=False
        )

        # Agent vote to stop
        check_early_stopping = AsyncMock(return_value=False)  # False = stop

        agent1 = MockAgent(name="agent-1", role="proposer")
        result = MockResult(metadata={}, critiques=[])
        ctx = MockDebateContext(
            agents=[agent1],
            proposers=[agent1],
            proposals={"agent-1": "proposal"},
            result=result,
        )

        phase = DebateRoundsPhase(
            protocol=protocol,
            notify_spectator=notify_spectator,
            check_early_stopping=check_early_stopping,
            critique_with_agent=AsyncMock(return_value=None),
        )
        phase._convergence_tracker = convergence_tracker

        with (
            patch("aragora.debate.phases.debate_rounds.get_complexity_governor") as mock_gov,
        ):
            mock_gov.return_value.get_scaled_timeout.return_value = 30.0

            perf_monitor = MagicMock()
            perf_monitor.track_phase = MagicMock(side_effect=lambda *a, **kw: _noop_cm())
            perf_monitor.slow_round_threshold = 60.0

            # Execute round 2 out of 5 (not last round, termination checks apply)
            should_continue = await phase._execute_round(
                ctx, perf_monitor, round_num=2, total_rounds=5
            )

        assert should_continue is False
        assert result.metadata["early_termination"] is True
        assert result.metadata["early_termination_source"] == "agent_vote"
        assert result.metadata["early_termination_round"] == 2
        assert "voted" in result.metadata["early_termination_reason"].lower()
        notify_spectator.assert_called()

    @pytest.mark.asyncio
    async def test_convergence_exit_sets_metadata_and_emits_event(self):
        """Convergence-based exit sets metadata and emits early_stop event."""
        protocol = MockProtocol(rounds=5)
        notify_spectator = MagicMock()

        convergence_tracker = MagicMock()
        convergence_tracker.check_convergence.return_value = MockConvergenceResult(
            converged=True,
            blocked_by_trickster=False,
            similarity=0.95,
        )

        agent1 = MockAgent(name="agent-1", role="proposer")
        result = MockResult(metadata={}, critiques=[])
        ctx = MockDebateContext(
            agents=[agent1],
            proposers=[agent1],
            proposals={"agent-1": "proposal"},
            result=result,
        )

        phase = DebateRoundsPhase(
            protocol=protocol,
            notify_spectator=notify_spectator,
            critique_with_agent=AsyncMock(return_value=None),
        )
        phase._convergence_tracker = convergence_tracker
        # Ensure no termination checks run (RLM quorum not reached)
        convergence_tracker.check_rlm_ready_quorum.return_value = False

        with (
            patch("aragora.debate.phases.debate_rounds.get_complexity_governor") as mock_gov,
        ):
            mock_gov.return_value.get_scaled_timeout.return_value = 30.0

            perf_monitor = MagicMock()
            perf_monitor.track_phase = MagicMock(side_effect=lambda *a, **kw: _noop_cm())
            perf_monitor.slow_round_threshold = 60.0

            should_continue = await phase._execute_round(
                ctx, perf_monitor, round_num=2, total_rounds=5
            )

        assert should_continue is False
        assert result.metadata["early_termination"] is True
        assert result.metadata["early_termination_source"] == "convergence"
        assert result.metadata["early_termination_round"] == 2
        assert "0.95" in result.metadata["early_termination_reason"]

        # Verify spectator was notified with early_stop event
        early_stop_calls = [c for c in notify_spectator.call_args_list if c[0][0] == "early_stop"]
        assert len(early_stop_calls) >= 1
        assert "convergence" in early_stop_calls[0][1]["details"].lower()

    @pytest.mark.asyncio
    async def test_should_terminate_backward_compat(self):
        """_should_terminate returns bool for backward compatibility."""
        convergence_tracker = MagicMock()
        convergence_tracker.check_rlm_ready_quorum.return_value = True
        convergence_tracker.collective_readiness.avg_confidence = 0.9

        phase = DebateRoundsPhase()
        phase._convergence_tracker = convergence_tracker

        ctx = MockDebateContext(proposals={"agent-1": "proposal"})
        result = await phase._should_terminate(ctx, round_num=2)

        assert result is True

    @pytest.mark.asyncio
    async def test_should_terminate_returns_false_when_no_triggers(self):
        """_should_terminate returns False when nothing triggers."""
        convergence_tracker = MagicMock()
        convergence_tracker.check_rlm_ready_quorum.return_value = False

        phase = DebateRoundsPhase()
        phase._convergence_tracker = convergence_tracker
        phase._check_judge_termination = None
        phase._check_early_stopping = None
        phase._stability_detector = None

        ctx = MockDebateContext(proposals={"agent-1": "proposal"})
        result = await phase._should_terminate(ctx, round_num=2)

        assert result is False

    @pytest.mark.asyncio
    async def test_check_termination_priority_order(self):
        """RLM ready has highest priority and short-circuits other checks."""
        convergence_tracker = MagicMock()
        convergence_tracker.check_rlm_ready_quorum.return_value = True
        convergence_tracker.collective_readiness.avg_confidence = 0.85

        check_judge = AsyncMock(return_value=(False, "Judge wants stop"))
        check_early = AsyncMock(return_value=False)

        phase = DebateRoundsPhase()
        phase._convergence_tracker = convergence_tracker
        phase._check_judge_termination = check_judge
        phase._check_early_stopping = check_early
        phase._stability_detector = None

        ctx = MockDebateContext(proposals={"agent-1": "proposal"})
        source, _details = await phase._check_termination_conditions(ctx, round_num=3)

        # RLM should win, judge and early stopping should NOT be called
        assert source == "rlm_ready"
        check_judge.assert_not_awaited()
        check_early.assert_not_awaited()
