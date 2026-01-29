"""Tests for Phase 3 Nomic Loop wiring: ContextPhase + builder integration."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.phases.context import ContextPhase


@contextmanager
def _noop_context(*args, **kwargs):
    yield


def _make_mock_agent(response: str = "agent context output"):
    """Create a mock agent with generate() method."""
    agent = AsyncMock()
    agent.generate = AsyncMock(return_value=response)
    return agent


@pytest.fixture
def mock_claude():
    return _make_mock_agent("Claude exploration result")


@pytest.fixture
def mock_codex():
    return _make_mock_agent("Codex exploration result")


def _patch_streaming():
    """Patch the streaming_task_context used inside _gather_with_agent."""
    return patch(
        "aragora.server.stream.arena_hooks.streaming_task_context",
        _noop_context,
    )


class TestContextPhaseWithBuilder:
    """Test that NomicContextBuilder output is injected into ContextPhase."""

    @pytest.mark.asyncio
    async def test_context_builder_injected(self, mock_claude, mock_codex):
        """Builder output is prepended to combined context."""
        mock_builder = AsyncMock()
        mock_builder.build_debate_context = AsyncMock(
            return_value="## Structured Codebase Map\nKey modules: debate, agents, server"
        )

        phase = ContextPhase(
            aragora_path=Path("."),
            claude_agent=mock_claude,
            codex_agent=mock_codex,
            log_fn=lambda *args, **kwargs: None,
            stream_emit_fn=lambda *args, **kwargs: None,
            context_builder=mock_builder,
        )

        # Disable TRUE RLM path so the prepended header is preserved
        with _patch_streaming(), patch.dict(os.environ, {"ARAGORA_NOMIC_CONTEXT_RLM": "false"}):
            result = await phase.execute()

        assert result["success"]
        summary = result["codebase_summary"]
        # Builder output should be first (prepended with header)
        assert "CODEBASE STRUCTURE MAP" in summary
        assert "Structured Codebase Map" in summary
        mock_builder.build_debate_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_builder_failure_graceful(self, mock_claude, mock_codex):
        """If builder raises, context phase continues without it."""
        mock_builder = AsyncMock()
        mock_builder.build_debate_context = AsyncMock(side_effect=RuntimeError("RLM unavailable"))

        phase = ContextPhase(
            aragora_path=Path("."),
            claude_agent=mock_claude,
            codex_agent=mock_codex,
            log_fn=lambda *args, **kwargs: None,
            stream_emit_fn=lambda *args, **kwargs: None,
            context_builder=mock_builder,
        )

        with _patch_streaming():
            result = await phase.execute()

        # Should still succeed from agent context
        assert result["success"]
        assert "Claude exploration result" in result["codebase_summary"]

    @pytest.mark.asyncio
    async def test_context_builder_none_skipped(self, mock_claude, mock_codex):
        """No builder = no error, normal execution."""
        phase = ContextPhase(
            aragora_path=Path("."),
            claude_agent=mock_claude,
            codex_agent=mock_codex,
            log_fn=lambda *args, **kwargs: None,
            stream_emit_fn=lambda *args, **kwargs: None,
            context_builder=None,
        )

        with _patch_streaming():
            result = await phase.execute()

        assert result["success"]
        assert "CLAUDE" in result["codebase_summary"] or "CODEX" in result["codebase_summary"]

    @pytest.mark.asyncio
    async def test_context_builder_empty_result(self, mock_claude, mock_codex):
        """If builder returns empty string, it's not added."""
        mock_builder = AsyncMock()
        mock_builder.build_debate_context = AsyncMock(return_value="")

        phase = ContextPhase(
            aragora_path=Path("."),
            claude_agent=mock_claude,
            codex_agent=mock_codex,
            log_fn=lambda *args, **kwargs: None,
            stream_emit_fn=lambda *args, **kwargs: None,
            context_builder=mock_builder,
        )

        with _patch_streaming():
            result = await phase.execute()

        assert result["success"]
        # Empty builder result should NOT add the header
        assert "CODEBASE STRUCTURE MAP" not in result["codebase_summary"]
