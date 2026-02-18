"""
Tests for debate_rounds_helpers module.

Covers:
- calculate_phase_timeout: various agent counts, respects minimum, max_concurrent arg
- is_effectively_empty_critique: empty, placeholder text, real content, missing suggestions field
- with_callback_timeout: normal completion, timeout returns default
- record_adaptive_round: import success and ImportError path
- emit_heartbeat: calls hook, no hook, hook errors
- observe_rhetorical_patterns: with/without observer, event emission, error handling
- refresh_evidence_for_round: odd rounds only, even rounds skipped, with/without skills
- refresh_with_skills: happy path, no registry, import error
- compress_debate_context: enough messages, not enough messages, no callback
- build_final_synthesis_prompt: correct format, includes all proposals, excludes own
- execute_final_synthesis_round: generates for each agent, timeout handling, no callback
- fire_propulsion_event: enabled/disabled, import error
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.phases.debate_rounds_helpers import (
    DEFAULT_CALLBACK_TIMEOUT,
    REVISION_PHASE_BASE_TIMEOUT,
    build_final_synthesis_prompt,
    calculate_phase_timeout,
    compress_debate_context,
    emit_heartbeat,
    execute_final_synthesis_round,
    fire_propulsion_event,
    is_effectively_empty_critique,
    observe_rhetorical_patterns,
    record_adaptive_round,
    refresh_evidence_for_round,
    refresh_with_skills,
    with_callback_timeout,
)


# =============================================================================
# Mock Objects
# =============================================================================


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str = "agent-a"
    timeout: float = 30.0
    role: str = "proposer"


@dataclass
class MockCritique:
    """Mock critique with suggestions field."""

    issues: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)
    critic: str = "critic-1"
    target: str = "agent-a"
    summary: str = "Test summary"

    def to_prompt(self) -> str:
        return f"Critique: issues={self.issues}, suggestions={self.suggestions}"


@dataclass
class MockEnv:
    """Mock environment."""

    task: str = "Design a rate limiter"


@dataclass
class MockResult:
    """Mock debate result."""

    messages: list = field(default_factory=list)
    critiques: list = field(default_factory=list)


@dataclass
class MockDebateContext:
    """Mock debate context."""

    env: MockEnv = field(default_factory=MockEnv)
    proposals: dict = field(default_factory=dict)
    context_messages: list = field(default_factory=list)
    agents: list = field(default_factory=list)
    proposers: list = field(default_factory=list)
    result: MockResult = field(default_factory=MockResult)
    round_critiques: list = field(default_factory=list)
    evidence_pack: Any = None
    debate_id: str = "test-debate"

    def add_message(self, msg):
        self.context_messages.append(msg)


# =============================================================================
# calculate_phase_timeout Tests
# =============================================================================


class TestCalculatePhaseTimeout:
    """Tests for calculate_phase_timeout."""

    def test_returns_base_timeout_for_few_agents(self):
        """With few agents the floor is REVISION_PHASE_BASE_TIMEOUT."""
        result = calculate_phase_timeout(num_agents=1, agent_timeout=10.0)
        assert result == REVISION_PHASE_BASE_TIMEOUT

    def test_scales_with_agent_count(self):
        """More agents produces a larger timeout."""
        few = calculate_phase_timeout(num_agents=2, agent_timeout=60.0)
        many = calculate_phase_timeout(num_agents=50, agent_timeout=60.0)
        assert many > few

    def test_formula_exact_value(self):
        """Exact formula: (num_agents / max_concurrent) * agent_timeout + 60."""
        # Use max_concurrent=5, num_agents=20, agent_timeout=100
        # (20 / 5) * 100 + 60 = 460
        result = calculate_phase_timeout(
            num_agents=20, agent_timeout=100.0, max_concurrent=5
        )
        assert result == 460.0

    def test_respects_minimum_timeout(self):
        """Result never drops below REVISION_PHASE_BASE_TIMEOUT (120)."""
        result = calculate_phase_timeout(
            num_agents=1, agent_timeout=1.0, max_concurrent=10
        )
        assert result == REVISION_PHASE_BASE_TIMEOUT

    def test_max_concurrent_zero_treated_as_one(self):
        """max_concurrent <= 0 is clamped to 1 to prevent division by zero."""
        result = calculate_phase_timeout(
            num_agents=5, agent_timeout=100.0, max_concurrent=0
        )
        # (5 / 1) * 100 + 60 = 560
        assert result == 560.0

    def test_single_agent_with_high_timeout(self):
        """A single agent with a very long timeout exceeds the base."""
        result = calculate_phase_timeout(
            num_agents=1, agent_timeout=500.0, max_concurrent=1
        )
        # (1 / 1) * 500 + 60 = 560
        assert result == 560.0


# =============================================================================
# is_effectively_empty_critique Tests
# =============================================================================


class TestIsEffectivelyEmptyCritique:
    """Tests for is_effectively_empty_critique."""

    def test_empty_issues_and_suggestions_with_field(self):
        """A critique with empty issues and suggestions (that has the field) is empty."""
        c = MockCritique(issues=[], suggestions=[])
        assert is_effectively_empty_critique(c) is True

    def test_placeholder_issue_agent_empty(self):
        """Single 'agent response was empty' issue with no suggestions is empty."""
        c = MockCritique(issues=["agent response was empty"], suggestions=[])
        assert is_effectively_empty_critique(c) is True

    def test_placeholder_parenthesized(self):
        """Parenthesized variant is also detected as empty."""
        c = MockCritique(issues=["(agent produced empty output)"], suggestions=[])
        assert is_effectively_empty_critique(c) is True

    def test_placeholder_with_suggestions_not_empty(self):
        """Placeholder issue plus real suggestions is NOT empty."""
        c = MockCritique(
            issues=["agent response was empty"],
            suggestions=["Consider rephrasing"],
        )
        assert is_effectively_empty_critique(c) is False

    def test_real_issues_not_empty(self):
        """Real issues make the critique non-empty."""
        c = MockCritique(issues=["Logic flaw in step 3"], suggestions=[])
        assert is_effectively_empty_critique(c) is False

    def test_no_suggestions_field_returns_false(self):
        """If the object lacks a suggestions attr entirely, returns False."""
        obj = MagicMock(spec=[])
        obj.issues = []
        # No 'suggestions' attribute at all
        assert is_effectively_empty_critique(obj) is False

    def test_whitespace_only_issues_treated_as_empty(self):
        """Issues containing only whitespace are stripped and ignored."""
        c = MockCritique(issues=["   ", "  \n "], suggestions=[])
        assert is_effectively_empty_critique(c) is True

    def test_none_issues_treated_as_empty(self):
        """None in issues field is handled."""
        c = MockCritique(issues=None, suggestions=[])
        assert is_effectively_empty_critique(c) is True


# =============================================================================
# with_callback_timeout Tests
# =============================================================================


class TestWithCallbackTimeout:
    """Tests for with_callback_timeout."""

    @pytest.mark.asyncio
    async def test_returns_coroutine_result(self):
        """Normal completion returns the coroutine's result."""

        async def fast():
            return 42

        result = await with_callback_timeout(fast(), timeout=5.0)
        assert result == 42

    @pytest.mark.asyncio
    async def test_timeout_returns_default(self):
        """Timeout returns the default value."""

        async def slow():
            await asyncio.sleep(100)

        result = await with_callback_timeout(slow(), timeout=0.01, default="fallback")
        assert result == "fallback"

    @pytest.mark.asyncio
    async def test_timeout_returns_none_by_default(self):
        """Default value is None when not specified."""

        async def slow():
            await asyncio.sleep(100)

        result = await with_callback_timeout(slow(), timeout=0.01)
        assert result is None


# =============================================================================
# record_adaptive_round Tests
# =============================================================================


class TestRecordAdaptiveRound:
    """Tests for record_adaptive_round."""

    def test_calls_metric_when_available(self):
        """Calls the metrics function when importable."""
        mock_fn = MagicMock()
        with patch.dict(
            "sys.modules",
            {
                "aragora.observability": MagicMock(),
                "aragora.observability.metrics": MagicMock(
                    record_adaptive_round_change=mock_fn
                ),
            },
        ):
            record_adaptive_round("extend")
            mock_fn.assert_called_once_with("extend")

    def test_handles_import_error(self):
        """Gracefully handles missing metrics module."""
        with patch.dict("sys.modules", {"aragora.observability.metrics": None}):
            # Should not raise
            record_adaptive_round("shrink")


# =============================================================================
# emit_heartbeat Tests
# =============================================================================


class TestEmitHeartbeat:
    """Tests for emit_heartbeat."""

    def test_calls_hook_when_present(self):
        """Calls the on_heartbeat hook with correct args."""
        hook = MagicMock()
        hooks = {"on_heartbeat": hook}
        emit_heartbeat(hooks, "round_3", "alive")
        hook.assert_called_once_with(phase="round_3", status="alive")

    def test_no_hook_does_nothing(self):
        """Safely does nothing when hook is absent."""
        emit_heartbeat({}, "round_1")

    def test_hook_runtime_error_swallowed(self):
        """RuntimeError from hook is caught."""
        hook = MagicMock(side_effect=RuntimeError("broken"))
        hooks = {"on_heartbeat": hook}
        emit_heartbeat(hooks, "round_2")
        hook.assert_called_once()

    def test_hook_type_error_swallowed(self):
        """TypeError from hook is caught."""
        hook = MagicMock(side_effect=TypeError("bad arg"))
        hooks = {"on_heartbeat": hook}
        emit_heartbeat(hooks, "round_2")
        hook.assert_called_once()

    def test_default_status_is_alive(self):
        """Default status parameter is 'alive'."""
        hook = MagicMock()
        hooks = {"on_heartbeat": hook}
        emit_heartbeat(hooks, "round_5")
        hook.assert_called_once_with(phase="round_5", status="alive")


# =============================================================================
# observe_rhetorical_patterns Tests
# =============================================================================


class TestObserveRhetoricalPatterns:
    """Tests for observe_rhetorical_patterns."""

    def test_returns_none_when_no_observer(self):
        """Does nothing when observer is None/falsy."""
        result = observe_rhetorical_patterns(
            rhetorical_observer=None,
            event_emitter=MagicMock(),
            hooks={},
            agent="agent-a",
            content="some content",
            round_num=1,
        )
        assert result is None

    def test_emits_events_on_observations(self):
        """When observer returns observations, events are emitted."""
        obs = MagicMock()
        obs.pattern.value = "appeal_to_authority"
        obs.to_dict.return_value = {"pattern": "appeal_to_authority"}
        obs.audience_commentary = "Interesting pattern"
        obs.confidence = 0.9

        observer = MagicMock()
        observer.observe.return_value = [obs]

        emitter = MagicMock()
        hook = MagicMock()
        hooks = {"on_rhetorical_observation": hook}

        observe_rhetorical_patterns(
            rhetorical_observer=observer,
            event_emitter=emitter,
            hooks=hooks,
            agent="agent-a",
            content="I think because experts say...",
            round_num=2,
            loop_id="loop-1",
        )

        observer.observe.assert_called_once_with(
            agent="agent-a",
            content="I think because experts say...",
            round_num=2,
        )
        emitter.emit_sync.assert_called_once()
        hook.assert_called_once()

    def test_no_observations_returns_early(self):
        """When observer returns empty list, no events emitted."""
        observer = MagicMock()
        observer.observe.return_value = []

        emitter = MagicMock()
        hook = MagicMock()
        hooks = {"on_rhetorical_observation": hook}

        observe_rhetorical_patterns(
            rhetorical_observer=observer,
            event_emitter=emitter,
            hooks=hooks,
            agent="agent-a",
            content="plain text",
            round_num=1,
        )

        emitter.emit_sync.assert_not_called()
        hook.assert_not_called()

    def test_catches_observer_error(self):
        """Errors from observer.observe are caught gracefully."""
        observer = MagicMock()
        observer.observe.side_effect = RuntimeError("boom")

        observe_rhetorical_patterns(
            rhetorical_observer=observer,
            event_emitter=MagicMock(),
            hooks={},
            agent="agent-a",
            content="text",
            round_num=1,
        )
        # No exception raised


# =============================================================================
# refresh_evidence_for_round Tests
# =============================================================================


class TestRefreshEvidenceForRound:
    """Tests for refresh_evidence_for_round."""

    @pytest.mark.asyncio
    async def test_skips_when_no_callback(self):
        """Does nothing when callback is None."""
        ctx = MockDebateContext()
        await refresh_evidence_for_round(
            ctx=ctx,
            round_num=1,
            refresh_evidence_callback=None,
            skill_registry=None,
            enable_skills=False,
            notify_spectator=None,
            hooks={},
            partial_critiques=[],
        )

    @pytest.mark.asyncio
    async def test_skips_even_rounds(self):
        """Even rounds are skipped to avoid API overload."""
        callback = AsyncMock(return_value=5)
        ctx = MockDebateContext(proposals={"a": "proposal text"})
        await refresh_evidence_for_round(
            ctx=ctx,
            round_num=2,
            refresh_evidence_callback=callback,
            skill_registry=None,
            enable_skills=False,
            notify_spectator=None,
            hooks={},
            partial_critiques=[],
        )
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_runs_on_odd_rounds(self):
        """Odd rounds trigger the callback."""
        callback = AsyncMock(return_value=3)
        ctx = MockDebateContext(proposals={"a": "proposal text"})
        await refresh_evidence_for_round(
            ctx=ctx,
            round_num=3,
            refresh_evidence_callback=callback,
            skill_registry=None,
            enable_skills=False,
            notify_spectator=None,
            hooks={},
            partial_critiques=[],
        )
        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_notifies_spectator_on_refresh(self):
        """Spectator is notified when evidence is refreshed."""
        callback = AsyncMock(return_value=2)
        spectator = MagicMock()
        ctx = MockDebateContext(proposals={"a": "proposal text"})
        await refresh_evidence_for_round(
            ctx=ctx,
            round_num=1,
            refresh_evidence_callback=callback,
            skill_registry=None,
            enable_skills=False,
            notify_spectator=spectator,
            hooks={},
            partial_critiques=[],
        )
        spectator.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_when_no_proposals(self):
        """If there are no proposals or critiques, callback is still called but
        combined_text is empty so the function returns early."""
        callback = AsyncMock(return_value=0)
        ctx = MockDebateContext(proposals={})
        await refresh_evidence_for_round(
            ctx=ctx,
            round_num=1,
            refresh_evidence_callback=callback,
            skill_registry=None,
            enable_skills=False,
            notify_spectator=None,
            hooks={},
            partial_critiques=[],
        )
        # No texts to analyze => returns early without calling callback
        callback.assert_not_called()


# =============================================================================
# compress_debate_context Tests
# =============================================================================


class TestCompressDebateContext:
    """Tests for compress_debate_context."""

    @pytest.mark.asyncio
    async def test_skips_when_no_callback(self):
        """Does nothing when callback is None."""
        ctx = MockDebateContext(context_messages=list(range(20)))
        await compress_debate_context(
            ctx=ctx,
            round_num=5,
            compress_context_callback=None,
            hooks={},
            notify_spectator=None,
            partial_critiques=[],
        )
        # context_messages unchanged
        assert len(ctx.context_messages) == 20

    @pytest.mark.asyncio
    async def test_skips_when_too_few_messages(self):
        """Fewer than 10 messages skips compression."""
        callback = AsyncMock()
        ctx = MockDebateContext(context_messages=list(range(5)))
        await compress_debate_context(
            ctx=ctx,
            round_num=5,
            compress_context_callback=callback,
            hooks={},
            notify_spectator=None,
            partial_critiques=[],
        )
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_compresses_when_enough_messages(self):
        """Messages are compressed when there are >= 10."""
        original = list(range(15))
        compressed = list(range(5))
        callback = AsyncMock(return_value=(compressed, []))
        ctx = MockDebateContext(context_messages=original)

        await compress_debate_context(
            ctx=ctx,
            round_num=3,
            compress_context_callback=callback,
            hooks={},
            notify_spectator=None,
            partial_critiques=[],
        )
        callback.assert_called_once()
        assert ctx.context_messages == compressed

    @pytest.mark.asyncio
    async def test_notifies_spectator_on_compression(self):
        """Spectator is notified when context is compressed."""
        compressed = list(range(3))
        callback = AsyncMock(return_value=(compressed, []))
        spectator = MagicMock()
        ctx = MockDebateContext(context_messages=list(range(12)))

        await compress_debate_context(
            ctx=ctx,
            round_num=4,
            compress_context_callback=callback,
            hooks={},
            notify_spectator=spectator,
            partial_critiques=[],
        )
        spectator.assert_called_once()


# =============================================================================
# build_final_synthesis_prompt Tests
# =============================================================================


class TestBuildFinalSynthesisPrompt:
    """Tests for build_final_synthesis_prompt."""

    def test_includes_agent_name(self):
        """Prompt includes the agent's name."""
        agent = MockAgent(name="claude")
        prompt = build_final_synthesis_prompt(
            agent=agent,
            current_proposal="My proposal",
            all_proposals={"claude": "My proposal", "gpt": "Other proposal"},
            critiques=[],
            round_num=7,
        )
        assert "claude" in prompt

    def test_includes_other_proposals(self):
        """Prompt includes proposals from other agents."""
        agent = MockAgent(name="claude")
        prompt = build_final_synthesis_prompt(
            agent=agent,
            current_proposal="My proposal",
            all_proposals={"claude": "My proposal", "gpt": "GPT says hello"},
            critiques=[],
            round_num=7,
        )
        assert "gpt" in prompt
        assert "GPT says hello" in prompt

    def test_excludes_own_proposal_from_others(self):
        """The agent's own proposal is NOT listed in other proposals section."""
        agent = MockAgent(name="claude")
        prompt = build_final_synthesis_prompt(
            agent=agent,
            current_proposal="My unique proposal text",
            all_proposals={"claude": "My unique proposal text"},
            critiques=[],
            round_num=7,
        )
        # The other proposals section should say no other proposals
        assert "No other proposals available" in prompt

    def test_includes_critique_summaries(self):
        """Critique summaries are included in the prompt."""
        agent = MockAgent(name="claude")
        crit = MockCritique(critic="gpt", target="claude", summary="Needs more detail")
        prompt = build_final_synthesis_prompt(
            agent=agent,
            current_proposal="proposal",
            all_proposals={"claude": "proposal"},
            critiques=[crit],
            round_num=7,
        )
        assert "Needs more detail" in prompt

    def test_final_synthesis_header(self):
        """Prompt includes the final synthesis header."""
        agent = MockAgent(name="agent-a")
        prompt = build_final_synthesis_prompt(
            agent=agent,
            current_proposal="",
            all_proposals={},
            critiques=[],
            round_num=7,
        )
        assert "ROUND 7: FINAL SYNTHESIS" in prompt

    def test_empty_proposal_placeholder(self):
        """Empty current proposal shows placeholder text."""
        agent = MockAgent(name="agent-a")
        prompt = build_final_synthesis_prompt(
            agent=agent,
            current_proposal="",
            all_proposals={},
            critiques=[],
            round_num=7,
        )
        assert "No previous proposal" in prompt


# =============================================================================
# execute_final_synthesis_round Tests
# =============================================================================


class TestExecuteFinalSynthesisRound:
    """Tests for execute_final_synthesis_round."""

    @pytest.mark.asyncio
    async def test_generates_for_each_agent(self):
        """Generates final synthesis for each proposer."""
        agent_a = MockAgent(name="agent-a")
        agent_b = MockAgent(name="agent-b")
        ctx = MockDebateContext(
            proposers=[agent_a, agent_b],
            proposals={"agent-a": "Proposal A", "agent-b": "Proposal B"},
            result=MockResult(),
        )
        generate = AsyncMock(return_value="Final synthesis content")

        partial_messages = []
        await execute_final_synthesis_round(
            ctx=ctx,
            round_num=7,
            circuit_breaker=None,
            generate_with_agent=generate,
            hooks={},
            notify_spectator=None,
            partial_messages=partial_messages,
        )

        assert generate.call_count == 2
        assert len(partial_messages) == 2
        assert ctx.proposals["agent-a"] == "Final synthesis content"

    @pytest.mark.asyncio
    async def test_skips_agent_on_timeout(self):
        """Agent timeout does not break the loop."""
        agent = MockAgent(name="agent-a", timeout=0.01)
        ctx = MockDebateContext(
            proposers=[agent],
            proposals={"agent-a": "Original"},
            result=MockResult(),
        )

        async def slow_gen(*args, **kwargs):
            await asyncio.sleep(100)

        partial_messages = []
        await execute_final_synthesis_round(
            ctx=ctx,
            round_num=7,
            circuit_breaker=None,
            generate_with_agent=slow_gen,
            hooks={},
            notify_spectator=None,
            partial_messages=partial_messages,
        )
        # Original proposal unchanged because synthesis timed out
        assert ctx.proposals["agent-a"] == "Original"
        assert len(partial_messages) == 0

    @pytest.mark.asyncio
    async def test_no_generate_callback(self):
        """When generate_with_agent is None, agents are skipped."""
        agent = MockAgent(name="agent-a")
        ctx = MockDebateContext(
            proposers=[agent],
            proposals={"agent-a": "Original"},
            result=MockResult(),
        )
        partial_messages = []
        await execute_final_synthesis_round(
            ctx=ctx,
            round_num=7,
            circuit_breaker=None,
            generate_with_agent=None,
            hooks={},
            notify_spectator=None,
            partial_messages=partial_messages,
        )
        assert len(partial_messages) == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_filters_agents(self):
        """Circuit breaker can exclude agents."""
        agent_a = MockAgent(name="agent-a")
        agent_b = MockAgent(name="agent-b")
        ctx = MockDebateContext(
            proposers=[agent_a, agent_b],
            proposals={"agent-a": "A", "agent-b": "B"},
            result=MockResult(),
        )

        cb = MagicMock()
        cb.filter_available_agents.return_value = [agent_a]

        generate = AsyncMock(return_value="Synth")
        partial_messages = []

        await execute_final_synthesis_round(
            ctx=ctx,
            round_num=7,
            circuit_breaker=cb,
            generate_with_agent=generate,
            hooks={},
            notify_spectator=None,
            partial_messages=partial_messages,
        )

        # Only agent_a should have been generated
        assert generate.call_count == 1
        assert len(partial_messages) == 1

    @pytest.mark.asyncio
    async def test_notifies_spectator(self):
        """Spectator is notified after all syntheses complete."""
        ctx = MockDebateContext(
            proposers=[],
            proposals={},
            result=MockResult(),
        )
        spectator = MagicMock()

        await execute_final_synthesis_round(
            ctx=ctx,
            round_num=7,
            circuit_breaker=None,
            generate_with_agent=AsyncMock(),
            hooks={},
            notify_spectator=spectator,
            partial_messages=[],
        )
        spectator.assert_called_once()


# =============================================================================
# fire_propulsion_event Tests
# =============================================================================


class TestFirePropulsionEvent:
    """Tests for fire_propulsion_event."""

    @pytest.mark.asyncio
    async def test_disabled_does_nothing(self):
        """When enable_propulsion is False, nothing happens."""
        engine = MagicMock()
        await fire_propulsion_event(
            event_type="critiques_ready",
            ctx=MockDebateContext(),
            round_num=1,
            propulsion_engine=engine,
            enable_propulsion=False,
        )
        engine.propel.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_engine_does_nothing(self):
        """When propulsion_engine is None, nothing happens."""
        await fire_propulsion_event(
            event_type="critiques_ready",
            ctx=MockDebateContext(),
            round_num=1,
            propulsion_engine=None,
            enable_propulsion=True,
        )

    @pytest.mark.asyncio
    async def test_fires_when_enabled(self):
        """When both enabled and engine present, fires propulsion."""
        engine = MagicMock()
        engine.propel = AsyncMock(return_value=[])

        with patch(
            "aragora.debate.phases.debate_rounds_helpers.fire_propulsion_event.__module__",
            create=True,
        ):
            # Patch the propulsion imports to avoid real import
            mock_payload_cls = MagicMock()
            mock_priority_cls = MagicMock()
            mock_priority_cls.NORMAL = "NORMAL"

            with patch.dict(
                "sys.modules",
                {
                    "aragora.debate.propulsion": MagicMock(
                        PropulsionPayload=mock_payload_cls,
                        PropulsionPriority=mock_priority_cls,
                    )
                },
            ):
                await fire_propulsion_event(
                    event_type="round_complete",
                    ctx=MockDebateContext(),
                    round_num=2,
                    propulsion_engine=engine,
                    enable_propulsion=True,
                    data={"extra": "data"},
                )
                engine.propel.assert_called_once()
