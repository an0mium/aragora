"""Tests for explainability auto-generation wiring in ArenaExtensions.

Covers:
- builder.build() is properly awaited (coroutine is consumed, not stored raw)
- The explanation Decision is attached to result.explanation
- Sync callers without an event loop don't crash
- ExplanationBuilder import failure is handled gracefully
- The pending task is stored to prevent garbage collection
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.extensions import ArenaExtensions


# =============================================================================
# Helpers
# =============================================================================


@dataclass
class _FakeCtx:
    """Minimal debate context for testing."""

    debate_id: str = "debate-xyz"
    query: str = "Should we adopt microservices?"
    environment: Any = None


@dataclass
class _FakeEnv:
    """Minimal environment."""

    task: str = "Design a rate limiter"


@dataclass
class _FakeResult:
    """Minimal debate result."""

    final_answer: str = "Yes, adopt microservices with a phased rollout."
    consensus: Any = None
    messages: list[Any] = field(default_factory=list)
    consensus_reached: bool = True
    confidence: float = 0.85
    rounds_used: int = 3
    id: str = "debate-xyz"


@dataclass
class _FakeDecision:
    """Minimal Decision stand-in."""

    decision_id: str = "dec-001"
    conclusion: str = "Adopt microservices"


# =============================================================================
# Tests: builder.build() is properly awaited
# =============================================================================


class TestBuilderBuildIsAwaited:
    """Verify that ExplanationBuilder.build() coroutine is actually consumed."""

    @pytest.mark.asyncio
    async def test_build_coroutine_is_awaited(self):
        """The async build() should be awaited, not stored as a coroutine object."""
        fake_decision = _FakeDecision()
        mock_builder = MagicMock()
        mock_builder.build = AsyncMock(return_value=fake_decision)

        ext = ArenaExtensions(
            auto_explain=True,
            explanation_builder=mock_builder,
        )
        ctx = _FakeCtx()
        result = _FakeResult()

        ext._auto_generate_explanation(ctx, result)

        # The task is scheduled on the loop -- let it complete
        assert hasattr(ext, "_pending_explanation_task")
        await ext._pending_explanation_task

        # build() was called and awaited (AsyncMock tracks this)
        mock_builder.build.assert_awaited_once_with(result, ctx)

    @pytest.mark.asyncio
    async def test_explanation_is_not_a_coroutine(self):
        """The stored explanation must be the Decision, not a coroutine."""
        fake_decision = _FakeDecision()
        mock_builder = MagicMock()
        mock_builder.build = AsyncMock(return_value=fake_decision)

        ext = ArenaExtensions(
            auto_explain=True,
            explanation_builder=mock_builder,
        )
        ctx = _FakeCtx()
        result = _FakeResult()

        ext._auto_generate_explanation(ctx, result)
        await ext._pending_explanation_task

        # The stored value should be the actual Decision, not a coroutine
        assert ext._last_explanation is fake_decision
        assert not asyncio.iscoroutine(ext._last_explanation)


# =============================================================================
# Tests: explanation is attached to result
# =============================================================================


class TestExplanationAttachedToResult:
    """Verify that the Decision is attached to result.explanation."""

    @pytest.mark.asyncio
    async def test_decision_attached_to_result(self):
        """After build completes, result.explanation should be the Decision."""
        fake_decision = _FakeDecision()
        mock_builder = MagicMock()
        mock_builder.build = AsyncMock(return_value=fake_decision)

        ext = ArenaExtensions(
            auto_explain=True,
            explanation_builder=mock_builder,
        )
        ctx = _FakeCtx()
        result = _FakeResult()

        ext._auto_generate_explanation(ctx, result)
        await ext._pending_explanation_task

        assert hasattr(result, "explanation")
        assert result.explanation is fake_decision

    @pytest.mark.asyncio
    async def test_decision_attached_with_env_task_fallback(self):
        """When ctx.query is empty, uses ctx.environment.task."""
        fake_decision = _FakeDecision()
        mock_builder = MagicMock()
        mock_builder.build = AsyncMock(return_value=fake_decision)

        ext = ArenaExtensions(
            auto_explain=True,
            explanation_builder=mock_builder,
        )
        ctx = _FakeCtx(query="", environment=_FakeEnv())
        result = _FakeResult()

        ext._auto_generate_explanation(ctx, result)
        await ext._pending_explanation_task

        assert result.explanation is fake_decision


# =============================================================================
# Tests: sync callers without event loop don't crash
# =============================================================================


class TestSyncCallersNoCrash:
    """Verify that calling from a sync context (no event loop) is safe."""

    def test_no_event_loop_does_not_crash(self):
        """When no event loop is running, method logs and returns gracefully."""
        mock_builder = MagicMock()
        mock_builder.build = AsyncMock(return_value=_FakeDecision())

        ext = ArenaExtensions(
            auto_explain=True,
            explanation_builder=mock_builder,
        )
        ctx = _FakeCtx()
        result = _FakeResult()

        # Call from sync context -- no running loop
        # This should NOT raise
        ext._auto_generate_explanation(ctx, result)

        # build() should NOT have been called (no loop to schedule on)
        mock_builder.build.assert_not_called()

        # No explanation should be attached
        assert not hasattr(result, "explanation")

    def test_on_debate_complete_sync_does_not_crash(self):
        """Full on_debate_complete path doesn't crash from sync context."""
        mock_builder = MagicMock()
        mock_builder.build = AsyncMock(return_value=_FakeDecision())

        ext = ArenaExtensions(
            auto_explain=True,
            explanation_builder=mock_builder,
        )
        ctx = _FakeCtx()
        result = _FakeResult()

        # Should not raise even without event loop
        ext.on_debate_complete(ctx, result, [])


# =============================================================================
# Tests: ExplanationBuilder import failure is handled
# =============================================================================


class TestImportFailureHandled:
    """Verify that ImportError for explainability module is handled."""

    @pytest.mark.asyncio
    async def test_import_error_handled_gracefully(self):
        """When ExplanationBuilder can't be imported, method returns silently."""
        ext = ArenaExtensions(
            auto_explain=True,
            explanation_builder=None,  # Force lazy import path
        )
        ctx = _FakeCtx()
        result = _FakeResult()

        with patch.dict(
            "sys.modules",
            {"aragora.explainability.builder": None},
        ):
            # Should not raise
            ext._auto_generate_explanation(ctx, result)

        # No explanation should be attached
        assert not hasattr(result, "explanation")

    @pytest.mark.asyncio
    async def test_auto_explain_disabled_skips_entirely(self):
        """When auto_explain=False, nothing happens."""
        mock_builder = MagicMock()
        mock_builder.build = AsyncMock(return_value=_FakeDecision())

        ext = ArenaExtensions(
            auto_explain=False,
            explanation_builder=mock_builder,
        )
        ctx = _FakeCtx()
        result = _FakeResult()

        ext._auto_generate_explanation(ctx, result)

        mock_builder.build.assert_not_called()
        assert not hasattr(result, "explanation")


# =============================================================================
# Tests: task reference kept to prevent GC
# =============================================================================


class TestTaskStoredForGC:
    """Verify the asyncio.Task is stored to prevent garbage collection."""

    @pytest.mark.asyncio
    async def test_pending_task_is_stored(self):
        """The scheduled task is saved on self._pending_explanation_task."""
        fake_decision = _FakeDecision()
        mock_builder = MagicMock()
        mock_builder.build = AsyncMock(return_value=fake_decision)

        ext = ArenaExtensions(
            auto_explain=True,
            explanation_builder=mock_builder,
        )
        ctx = _FakeCtx()
        result = _FakeResult()

        ext._auto_generate_explanation(ctx, result)

        task = ext._pending_explanation_task
        assert isinstance(task, asyncio.Task)
        await task
        assert task.done()


# =============================================================================
# Tests: build() failure is handled gracefully via done callback
# =============================================================================


class TestBuildFailureHandled:
    """Verify that if builder.build() raises, it's caught by the callback."""

    @pytest.mark.asyncio
    async def test_build_value_error_does_not_propagate(self):
        """If build() raises ValueError, exception is caught inside the task."""
        mock_builder = MagicMock()
        mock_builder.build = AsyncMock(side_effect=ValueError("build failed"))

        ext = ArenaExtensions(
            auto_explain=True,
            explanation_builder=mock_builder,
        )
        ctx = _FakeCtx()
        result = _FakeResult()

        ext._auto_generate_explanation(ctx, result)

        task = ext._pending_explanation_task
        # Await should not raise -- exception is caught in _build_and_attach
        await task

        # No explanation should be attached
        assert not hasattr(result, "explanation")
        assert ext._last_explanation is None

    @pytest.mark.asyncio
    async def test_build_runtime_error_handled(self):
        """RuntimeError from build() is handled gracefully."""
        mock_builder = MagicMock()
        mock_builder.build = AsyncMock(side_effect=RuntimeError("no model"))

        ext = ArenaExtensions(
            auto_explain=True,
            explanation_builder=mock_builder,
        )
        ctx = _FakeCtx()
        result = _FakeResult()

        ext._auto_generate_explanation(ctx, result)
        await ext._pending_explanation_task

        assert not hasattr(result, "explanation")

    @pytest.mark.asyncio
    async def test_build_type_error_handled(self):
        """TypeError from build() is handled gracefully."""
        mock_builder = MagicMock()
        mock_builder.build = AsyncMock(side_effect=TypeError("bad arg"))

        ext = ArenaExtensions(
            auto_explain=True,
            explanation_builder=mock_builder,
        )
        ctx = _FakeCtx()
        result = _FakeResult()

        ext._auto_generate_explanation(ctx, result)
        await ext._pending_explanation_task

        assert not hasattr(result, "explanation")


# =============================================================================
# Tests: early returns for missing data
# =============================================================================


class TestEarlyReturns:
    """Verify that missing query/answer causes early return."""

    @pytest.mark.asyncio
    async def test_no_query_skips(self):
        """When no query is available, explanation is skipped."""
        mock_builder = MagicMock()
        mock_builder.build = AsyncMock(return_value=_FakeDecision())

        ext = ArenaExtensions(
            auto_explain=True,
            explanation_builder=mock_builder,
        )
        ctx = _FakeCtx(query="", environment=None)
        result = _FakeResult()

        ext._auto_generate_explanation(ctx, result)
        mock_builder.build.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_answer_skips(self):
        """When no answer content is available, explanation is skipped."""
        mock_builder = MagicMock()
        mock_builder.build = AsyncMock(return_value=_FakeDecision())

        ext = ArenaExtensions(
            auto_explain=True,
            explanation_builder=mock_builder,
        )
        ctx = _FakeCtx()
        result = _FakeResult(final_answer="", consensus=None, messages=[])

        ext._auto_generate_explanation(ctx, result)
        mock_builder.build.assert_not_called()


# =============================================================================
# Tests: integration with on_debate_complete
# =============================================================================


class TestOnDebateCompleteIntegration:
    """Verify the fix works through the full on_debate_complete path."""

    @pytest.mark.asyncio
    async def test_on_debate_complete_triggers_async_explanation(self):
        """on_debate_complete schedules the explanation build properly."""
        fake_decision = _FakeDecision()
        mock_builder = MagicMock()
        mock_builder.build = AsyncMock(return_value=fake_decision)

        ext = ArenaExtensions(
            auto_explain=True,
            explanation_builder=mock_builder,
        )
        ctx = _FakeCtx()
        result = _FakeResult()

        ext.on_debate_complete(ctx, result, [])

        # Let the scheduled task run
        assert hasattr(ext, "_pending_explanation_task")
        await ext._pending_explanation_task

        mock_builder.build.assert_awaited_once()
        assert result.explanation is fake_decision

    @pytest.mark.asyncio
    async def test_on_debate_complete_other_extensions_not_affected(self):
        """Other extensions in on_debate_complete still work."""
        fake_decision = _FakeDecision()
        mock_builder = MagicMock()
        mock_builder.build = AsyncMock(return_value=fake_decision)

        ext = ArenaExtensions(
            auto_explain=True,
            explanation_builder=mock_builder,
            auto_evaluate=False,
            auto_broadcast=False,
            auto_export_training=False,
        )
        ctx = _FakeCtx()
        result = _FakeResult()

        # Patch internal methods to track they're still called
        with (
            patch.object(ext, "_record_token_usage") as mock_record,
            patch.object(ext, "_sync_usage_to_stripe") as mock_sync,
        ):
            ext.on_debate_complete(ctx, result, [])
            mock_record.assert_called_once()
            mock_sync.assert_called_once()

        await ext._pending_explanation_task
