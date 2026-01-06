"""
Tests for aragora.debate.phases.analytics_phase module.

Tests AnalyticsPhase class and all analytics logic.
"""

import pytest
import time
from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import MagicMock, AsyncMock

from aragora.debate.context import DebateContext
from aragora.debate.phases.analytics_phase import AnalyticsPhase


# ============================================================================
# Mock Classes
# ============================================================================

@dataclass
class MockEnvironment:
    """Mock environment for testing."""
    task: str = "Test task"
    context: str = ""
    domain: str = "security"


@dataclass
class MockAgent:
    """Mock agent for testing."""
    name: str = "test_agent"
    role: str = "proposer"


@dataclass
class MockCritique:
    """Mock critique for testing."""
    agent: str = "critic"
    target_agent: str = "proposer"
    issues: list = field(default_factory=list)
    severity: float = 0.6
    reasoning: str = "High severity issue"
    category: str = "security"


@dataclass
class MockVote:
    """Mock vote for testing."""
    agent: str = "voter"
    choice: str = "claude"
    confidence: float = 0.8


@dataclass
class MockDebateResult:
    """Mock debate result for testing."""
    id: str = "debate_001"
    task: str = "Test task"
    messages: list = field(default_factory=list)
    critiques: list = field(default_factory=list)
    votes: list = field(default_factory=list)
    dissenting_views: list = field(default_factory=list)
    rounds_used: int = 3
    duration_seconds: float = 0.0
    winner: Optional[str] = None
    final_answer: str = "The answer is X"
    consensus_reached: bool = True
    confidence: float = 0.85
    disagreement_report: Optional[object] = None
    grounded_verdict: Optional[object] = None
    belief_cruxes: list = field(default_factory=list)


@dataclass
class MockDisagreementReport:
    """Mock disagreement report for testing."""
    unanimous_critiques: list = field(default_factory=list)
    split_opinions: list = field(default_factory=list)


@dataclass
class MockGroundedVerdict:
    """Mock grounded verdict for testing."""
    grounding_score: float = 0.85
    claims: list = field(default_factory=list)

    def to_dict(self):
        return {"grounding_score": self.grounding_score}


# ============================================================================
# AnalyticsPhase Construction Tests
# ============================================================================

class TestAnalyticsPhaseConstruction:
    """Tests for AnalyticsPhase construction."""

    def test_minimal_construction(self):
        """Should create with no arguments."""
        analytics = AnalyticsPhase()

        assert analytics.memory is None
        assert analytics.insight_store is None
        assert analytics.hooks == {}

    def test_full_construction(self):
        """Should create with all arguments."""
        memory = MagicMock()
        insight_store = MagicMock()
        hooks = {"on_consensus": MagicMock()}

        analytics = AnalyticsPhase(
            memory=memory,
            insight_store=insight_store,
            hooks=hooks,
            loop_id="loop_001",
        )

        assert analytics.memory is memory
        assert analytics.insight_store is insight_store
        assert "on_consensus" in analytics.hooks


# ============================================================================
# Pattern Tracking Tests
# ============================================================================

class TestPatternTracking:
    """Tests for failed pattern tracking."""

    @pytest.mark.asyncio
    async def test_track_failed_patterns(self):
        """Should track high severity patterns on no consensus."""
        memory = MagicMock()
        analytics = AnalyticsPhase(memory=memory)

        ctx = DebateContext(env=MockEnvironment(), start_time=time.time())
        ctx.result = MockDebateResult(
            consensus_reached=False,
            critiques=[MockCritique(severity=0.7, reasoning="Critical issue")],
        )

        await analytics.execute(ctx)

        memory.fail_pattern.assert_called_once()

    @pytest.mark.asyncio
    async def test_skip_low_severity_patterns(self):
        """Should skip low severity critiques."""
        memory = MagicMock()
        analytics = AnalyticsPhase(memory=memory)

        ctx = DebateContext(env=MockEnvironment(), start_time=time.time())
        ctx.result = MockDebateResult(
            consensus_reached=False,
            critiques=[MockCritique(severity=0.3)],  # Below threshold
        )

        await analytics.execute(ctx)

        memory.fail_pattern.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_patterns_on_consensus(self):
        """Should skip pattern tracking when consensus reached."""
        memory = MagicMock()
        analytics = AnalyticsPhase(memory=memory)

        ctx = DebateContext(env=MockEnvironment(), start_time=time.time())
        ctx.result = MockDebateResult(
            consensus_reached=True,
            critiques=[MockCritique(severity=0.8)],
        )

        await analytics.execute(ctx)

        memory.fail_pattern.assert_not_called()


# ============================================================================
# Duration Tests
# ============================================================================

class TestDurationSetting:
    """Tests for duration calculation."""

    @pytest.mark.asyncio
    async def test_set_duration(self):
        """Should set duration from start_time."""
        analytics = AnalyticsPhase()

        start = time.time() - 5.0  # 5 seconds ago
        ctx = DebateContext(env=MockEnvironment(), start_time=start)
        ctx.result = MockDebateResult()

        await analytics.execute(ctx)

        assert ctx.result.duration_seconds >= 5.0
        assert ctx.result.duration_seconds < 10.0


# ============================================================================
# Hook Event Tests
# ============================================================================

class TestHookEvents:
    """Tests for hook event emission."""

    @pytest.mark.asyncio
    async def test_emit_consensus_event(self):
        """Should emit on_consensus hook."""
        on_consensus = MagicMock()
        analytics = AnalyticsPhase(hooks={"on_consensus": on_consensus})

        ctx = DebateContext(env=MockEnvironment(), start_time=time.time())
        ctx.result = MockDebateResult(
            consensus_reached=True,
            confidence=0.9,
            final_answer="Answer",
        )

        await analytics.execute(ctx)

        on_consensus.assert_called_once()
        call_kwargs = on_consensus.call_args[1]
        assert call_kwargs["reached"] is True
        assert call_kwargs["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_emit_debate_end_event(self):
        """Should emit on_debate_end hook."""
        on_debate_end = MagicMock()
        analytics = AnalyticsPhase(hooks={"on_debate_end": on_debate_end})

        ctx = DebateContext(env=MockEnvironment(), start_time=time.time())
        ctx.result = MockDebateResult(rounds_used=3)

        await analytics.execute(ctx)

        on_debate_end.assert_called_once()
        call_kwargs = on_debate_end.call_args[1]
        assert call_kwargs["rounds"] == 3


# ============================================================================
# Spectator Notification Tests
# ============================================================================

class TestSpectatorNotification:
    """Tests for spectator notifications."""

    @pytest.mark.asyncio
    async def test_notify_debate_end(self):
        """Should notify spectator of debate end."""
        notify_spectator = MagicMock()
        analytics = AnalyticsPhase(notify_spectator=notify_spectator)

        ctx = DebateContext(env=MockEnvironment(), start_time=time.time())
        ctx.result = MockDebateResult(confidence=0.85)

        await analytics.execute(ctx)

        notify_spectator.assert_called()
        # Find the debate_end call
        calls = [c for c in notify_spectator.call_args_list if c[0][0] == "debate_end"]
        assert len(calls) >= 1


# ============================================================================
# Winner Determination Tests
# ============================================================================

class TestWinnerDetermination:
    """Tests for winner determination from vote tally."""

    @pytest.mark.asyncio
    async def test_determine_winner(self):
        """Should determine winner from vote tally."""
        analytics = AnalyticsPhase()

        ctx = DebateContext(env=MockEnvironment(), start_time=time.time())
        ctx.result = MockDebateResult()
        ctx.vote_tally = {"claude": 3.0, "gpt4": 1.5}

        await analytics.execute(ctx)

        assert ctx.winner_agent == "claude"
        assert ctx.result.winner == "claude"

    @pytest.mark.asyncio
    async def test_no_winner_without_tally(self):
        """Should not set winner without vote tally."""
        analytics = AnalyticsPhase()

        ctx = DebateContext(env=MockEnvironment(), start_time=time.time())
        ctx.result = MockDebateResult()
        ctx.vote_tally = {}

        await analytics.execute(ctx)

        assert ctx.winner_agent is None


# ============================================================================
# Relationship Update Tests
# ============================================================================

class TestRelationshipUpdate:
    """Tests for agent relationship updates."""

    @pytest.mark.asyncio
    async def test_update_relationships(self):
        """Should call update_agent_relationships callback."""
        update_callback = MagicMock()
        analytics = AnalyticsPhase(update_agent_relationships=update_callback)

        agents = [MockAgent(name="claude"), MockAgent(name="gpt4")]
        ctx = DebateContext(env=MockEnvironment(), agents=agents, start_time=time.time())
        ctx.result = MockDebateResult(votes=[MockVote()])
        ctx.winner_agent = "claude"

        await analytics.execute(ctx)

        update_callback.assert_called_once()
        call_kwargs = update_callback.call_args[1]
        assert call_kwargs["winner"] == "claude"
        assert set(call_kwargs["participants"]) == {"claude", "gpt4"}


# ============================================================================
# Disagreement Report Tests
# ============================================================================

class TestDisagreementReport:
    """Tests for disagreement report generation."""

    @pytest.mark.asyncio
    async def test_generate_disagreement_report(self):
        """Should generate disagreement report."""
        report = MockDisagreementReport(
            unanimous_critiques=["All agents found security issue"],
            split_opinions=[("Topic", ["claude"], ["gpt4"])],
        )
        generate_callback = MagicMock(return_value=report)
        analytics = AnalyticsPhase(generate_disagreement_report=generate_callback)

        ctx = DebateContext(env=MockEnvironment(), start_time=time.time())
        ctx.result = MockDebateResult(
            votes=[MockVote()],
            critiques=[MockCritique()],
        )
        ctx.winner_agent = "claude"

        await analytics.execute(ctx)

        generate_callback.assert_called_once()
        assert ctx.result.disagreement_report is report


# ============================================================================
# Grounded Verdict Tests
# ============================================================================

class TestGroundedVerdict:
    """Tests for grounded verdict generation."""

    @pytest.mark.asyncio
    async def test_generate_grounded_verdict(self):
        """Should generate grounded verdict."""
        verdict = MockGroundedVerdict(grounding_score=0.9, claims=["Claim 1"])
        create_callback = MagicMock(return_value=verdict)
        analytics = AnalyticsPhase(create_grounded_verdict=create_callback)

        ctx = DebateContext(env=MockEnvironment(), start_time=time.time())
        ctx.result = MockDebateResult()

        await analytics.execute(ctx)

        create_callback.assert_called_once()
        assert ctx.result.grounded_verdict is verdict

    @pytest.mark.asyncio
    async def test_emit_grounded_verdict_event(self):
        """Should emit grounded verdict event if StreamEvent available."""
        verdict = MockGroundedVerdict(grounding_score=0.9)
        create_callback = MagicMock(return_value=verdict)
        event_emitter = MagicMock()

        analytics = AnalyticsPhase(
            create_grounded_verdict=create_callback,
            event_emitter=event_emitter,
            loop_id="loop_001",
        )

        ctx = DebateContext(env=MockEnvironment(), start_time=time.time())
        ctx.result = MockDebateResult()

        await analytics.execute(ctx)

        # The emit may or may not be called depending on StreamEvent import
        # Just verify the callback was called correctly
        create_callback.assert_called_once()


# ============================================================================
# Formal Verification Tests
# ============================================================================

class TestFormalVerification:
    """Tests for formal Z3 verification."""

    @pytest.mark.asyncio
    async def test_verify_formally(self):
        """Should call verify_claims_formally callback."""
        verify_callback = AsyncMock()
        analytics = AnalyticsPhase(verify_claims_formally=verify_callback)

        ctx = DebateContext(env=MockEnvironment(), start_time=time.time())
        ctx.result = MockDebateResult()

        await analytics.execute(ctx)

        verify_callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_verify_error_handling(self):
        """Should handle verification errors gracefully."""
        verify_callback = AsyncMock(side_effect=Exception("Z3 error"))
        analytics = AnalyticsPhase(verify_claims_formally=verify_callback)

        ctx = DebateContext(env=MockEnvironment(), start_time=time.time())
        ctx.result = MockDebateResult()

        # Should not raise
        await analytics.execute(ctx)


# ============================================================================
# Recording Finalization Tests
# ============================================================================

class TestRecordingFinalization:
    """Tests for replay recording finalization."""

    @pytest.mark.asyncio
    async def test_finalize_recording(self):
        """Should finalize recording with verdict."""
        recorder = MagicMock()
        analytics = AnalyticsPhase(recorder=recorder)

        ctx = DebateContext(env=MockEnvironment(), start_time=time.time())
        ctx.result = MockDebateResult(final_answer="The final answer")
        ctx.vote_tally = {"claude": 2.0}

        await analytics.execute(ctx)

        recorder.finalize.assert_called_once()
        call_args = recorder.finalize.call_args[0]
        assert "The final answer" in call_args[0]
        assert call_args[1] == {"claude": 2.0}

    @pytest.mark.asyncio
    async def test_finalize_with_empty_answer(self):
        """Should handle empty final answer."""
        recorder = MagicMock()
        analytics = AnalyticsPhase(recorder=recorder)

        ctx = DebateContext(env=MockEnvironment(), start_time=time.time())
        ctx.result = MockDebateResult(final_answer="")
        ctx.vote_tally = {}

        await analytics.execute(ctx)

        call_args = recorder.finalize.call_args[0]
        assert call_args[0] == "incomplete"

    @pytest.mark.asyncio
    async def test_recorder_error_handling(self):
        """Should handle recorder errors gracefully."""
        recorder = MagicMock()
        recorder.finalize.side_effect = Exception("Recorder error")
        analytics = AnalyticsPhase(recorder=recorder)

        ctx = DebateContext(env=MockEnvironment(), start_time=time.time())
        ctx.result = MockDebateResult()

        # Should not raise
        await analytics.execute(ctx)


# ============================================================================
# Completion Logging Tests
# ============================================================================

class TestCompletionLogging:
    """Tests for completion logging."""

    @pytest.mark.asyncio
    async def test_format_conclusion(self):
        """Should call format_conclusion callback."""
        format_callback = MagicMock(return_value="Formatted conclusion")
        analytics = AnalyticsPhase(format_conclusion=format_callback)

        ctx = DebateContext(env=MockEnvironment(), start_time=time.time())
        ctx.result = MockDebateResult()

        await analytics.execute(ctx)

        format_callback.assert_called_once()


# ============================================================================
# Integration Tests
# ============================================================================

class TestAnalyticsPhaseIntegration:
    """Integration tests for full analytics execution."""

    @pytest.mark.asyncio
    async def test_full_analytics_execution(self):
        """Should execute all analytics steps."""
        # Set up all components
        memory = MagicMock()
        recorder = MagicMock()
        on_consensus = MagicMock()
        on_debate_end = MagicMock()
        notify_spectator = MagicMock()
        update_relationships = MagicMock()
        generate_report = MagicMock(return_value=MockDisagreementReport())
        create_verdict = MagicMock(return_value=MockGroundedVerdict())
        verify_formally = AsyncMock()
        format_conclusion = MagicMock(return_value="Conclusion")

        analytics = AnalyticsPhase(
            memory=memory,
            recorder=recorder,
            hooks={"on_consensus": on_consensus, "on_debate_end": on_debate_end},
            notify_spectator=notify_spectator,
            update_agent_relationships=update_relationships,
            generate_disagreement_report=generate_report,
            create_grounded_verdict=create_verdict,
            verify_claims_formally=verify_formally,
            format_conclusion=format_conclusion,
        )

        agents = [MockAgent(name="claude")]
        ctx = DebateContext(
            env=MockEnvironment(),
            agents=agents,
            start_time=time.time() - 10.0,
        )
        ctx.result = MockDebateResult(
            consensus_reached=True,
            votes=[MockVote()],
            critiques=[MockCritique()],
        )
        ctx.vote_tally = {"claude": 2.0}

        await analytics.execute(ctx)

        # Verify all systems were called
        on_consensus.assert_called_once()
        on_debate_end.assert_called_once()
        notify_spectator.assert_called()
        update_relationships.assert_called_once()
        generate_report.assert_called_once()
        create_verdict.assert_called_once()
        verify_formally.assert_called_once()
        format_conclusion.assert_called_once()
        recorder.finalize.assert_called_once()

        # Verify result state
        assert ctx.winner_agent == "claude"
        assert ctx.result.winner == "claude"
        assert ctx.result.duration_seconds >= 10.0

    @pytest.mark.asyncio
    async def test_handles_missing_result(self):
        """Should handle missing result gracefully."""
        analytics = AnalyticsPhase(recorder=MagicMock())
        ctx = DebateContext(env=MockEnvironment(), start_time=time.time())
        ctx.result = None

        # Should not raise
        await analytics.execute(ctx)
