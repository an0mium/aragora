"""Tests for post-debate workflow execution fallback.

Covers:
- Workflow fires when FeedbackPhase was skipped (no trigger flag)
- No double-fire when FeedbackPhase already triggered
- Disabled = skip (enable_post_debate_workflow=False)
- No workflow configured = skip
- Error in workflow doesn't break completion
- FeedbackPhase sets ctx.post_debate_workflow_triggered flag
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.orchestrator_runner import handle_debate_completion


@dataclass
class _FakeResult:
    consensus: str = "Use rate limiting"
    consensus_confidence: float = 0.85
    confidence: float = 0.85
    predictions: dict = field(default_factory=dict)
    bead_id: str | None = None


@dataclass
class _FakeEnv:
    task: str = "test task"


@dataclass
class _FakeCtx:
    debate_id: str = "debate-001"
    agents: list = field(default_factory=list)
    env: _FakeEnv = field(default_factory=_FakeEnv)
    result: _FakeResult | None = field(default_factory=_FakeResult)
    domain: str = "security"


@dataclass
class _FakeState:
    ctx: _FakeCtx = field(default_factory=_FakeCtx)
    debate_id: str = "debate-001"
    debate_status: str = "completed"
    gupp_bead_id: str | None = None
    gupp_hook_entries: list = field(default_factory=list)


def _make_arena(
    enable_workflow: bool = True,
    workflow: MagicMock | None = None,
    threshold: float = 0.0,
) -> MagicMock:
    """Create a mock Arena with workflow attributes."""
    arena = MagicMock()
    arena.enable_post_debate_workflow = enable_workflow
    arena.post_debate_workflow = workflow
    arena.post_debate_workflow_threshold = threshold
    arena._trackers = MagicMock()
    arena.extensions = MagicMock()
    arena._budget_coordinator = MagicMock()
    arena._ingest_debate_outcome = AsyncMock()
    arena._queue_for_supabase_sync = MagicMock()
    return arena


class TestWorkflowFallback:
    """Test post-debate workflow fallback in handle_debate_completion."""

    @pytest.mark.asyncio
    async def test_fires_when_feedback_phase_skipped(self):
        """Workflow runs when ctx.post_debate_workflow_triggered is not set."""
        workflow = MagicMock()
        workflow.execute = AsyncMock()
        arena = _make_arena(enable_workflow=True, workflow=workflow)
        state = _FakeState()
        # ctx does NOT have post_debate_workflow_triggered

        await handle_debate_completion(arena, state)

        # Allow the fire-and-forget task to run
        await asyncio.sleep(0.05)
        workflow.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_double_fire_when_already_triggered(self):
        """Workflow skipped when FeedbackPhase already triggered it."""
        workflow = MagicMock()
        workflow.execute = AsyncMock()
        arena = _make_arena(enable_workflow=True, workflow=workflow)
        state = _FakeState()
        state.ctx.post_debate_workflow_triggered = True  # type: ignore[attr-defined]

        await handle_debate_completion(arena, state)

        await asyncio.sleep(0.05)
        workflow.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_skipped_when_disabled(self):
        """No workflow when enable_post_debate_workflow is False."""
        workflow = MagicMock()
        workflow.execute = AsyncMock()
        arena = _make_arena(enable_workflow=False, workflow=workflow)
        state = _FakeState()

        await handle_debate_completion(arena, state)

        await asyncio.sleep(0.05)
        workflow.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_skipped_when_no_workflow_configured(self):
        """No error when post_debate_workflow is None."""
        arena = _make_arena(enable_workflow=True, workflow=None)
        state = _FakeState()

        await handle_debate_completion(arena, state)
        # Should complete without error

    @pytest.mark.asyncio
    async def test_error_does_not_break_completion(self):
        """Exception in workflow doesn't prevent supabase sync."""
        workflow = MagicMock()
        workflow.execute = AsyncMock(side_effect=RuntimeError("workflow boom"))
        arena = _make_arena(enable_workflow=True, workflow=workflow)
        state = _FakeState()

        await handle_debate_completion(arena, state)

        # Fire-and-forget catches the error
        await asyncio.sleep(0.05)
        # Supabase sync should still have been called
        arena._queue_for_supabase_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_respects_confidence_threshold(self):
        """Workflow only fires if confidence >= threshold."""
        workflow = MagicMock()
        workflow.execute = AsyncMock()
        arena = _make_arena(enable_workflow=True, workflow=workflow, threshold=0.95)
        state = _FakeState()
        # Result confidence is 0.85 < 0.95

        await handle_debate_completion(arena, state)

        await asyncio.sleep(0.05)
        workflow.execute.assert_not_called()


class TestFeedbackPhaseFlag:
    """Test that FeedbackPhase sets post_debate_workflow_triggered."""

    def test_flag_set_on_trigger(self):
        """_maybe_trigger_workflow sets ctx.post_debate_workflow_triggered."""
        from aragora.debate.phases.feedback_phase import FeedbackPhase

        ctx = MagicMock()
        ctx.result = MagicMock()
        ctx.result.confidence = 0.9
        ctx.debate_id = "test-debate"

        phase = FeedbackPhase.__new__(FeedbackPhase)
        phase.post_debate_workflow = MagicMock()
        phase.enable_post_debate_workflow = True
        phase.post_debate_workflow_threshold = 0.5

        # Run _maybe_trigger_workflow in an event loop
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(phase._maybe_trigger_workflow(ctx))
        finally:
            # Clean up pending tasks
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

        assert getattr(ctx, "post_debate_workflow_triggered", False) is True
