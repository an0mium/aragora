"""Tests for Phase 3 Nomic Loop wiring: ContextPhase + builder integration."""

from __future__ import annotations

import asyncio
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
    agent.timeout = None  # Prevent MagicMock attribute from breaking asyncio.wait_for
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

        with _patch_streaming(), patch.dict(os.environ, {"ARAGORA_NOMIC_CONTEXT_RLM": "false"}):
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

        with _patch_streaming(), patch.dict(os.environ, {"ARAGORA_NOMIC_CONTEXT_RLM": "false"}):
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


class TestPhaseDataFlow:
    """Test that data flows correctly through the phase pipeline."""

    @pytest.mark.asyncio
    async def test_result_contains_required_fields(self, mock_claude, mock_codex):
        """ContextPhase result should have all required fields."""
        phase = ContextPhase(
            aragora_path=Path("."),
            claude_agent=mock_claude,
            codex_agent=mock_codex,
            log_fn=lambda *args, **kwargs: None,
            stream_emit_fn=lambda *args, **kwargs: None,
        )

        with _patch_streaming(), patch.dict(os.environ, {"ARAGORA_NOMIC_CONTEXT_RLM": "false"}):
            result = await phase.execute()

        assert "success" in result
        assert "codebase_summary" in result
        assert "data" in result
        assert "duration_seconds" in result
        assert "recent_changes" in result
        assert "open_issues" in result

    @pytest.mark.asyncio
    async def test_duration_is_positive(self, mock_claude, mock_codex):
        """Duration should be a positive number."""
        phase = ContextPhase(
            aragora_path=Path("."),
            claude_agent=mock_claude,
            codex_agent=mock_codex,
            log_fn=lambda *args, **kwargs: None,
            stream_emit_fn=lambda *args, **kwargs: None,
        )

        with _patch_streaming(), patch.dict(os.environ, {"ARAGORA_NOMIC_CONTEXT_RLM": "false"}):
            result = await phase.execute()

        assert result["duration_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_agents_succeeded_count(self, mock_claude, mock_codex):
        """data['agents_succeeded'] should reflect how many agents returned valid context."""
        phase = ContextPhase(
            aragora_path=Path("."),
            claude_agent=mock_claude,
            codex_agent=mock_codex,
            log_fn=lambda *args, **kwargs: None,
            stream_emit_fn=lambda *args, **kwargs: None,
        )

        with _patch_streaming(), patch.dict(os.environ, {"ARAGORA_NOMIC_CONTEXT_RLM": "false"}):
            result = await phase.execute()

        assert result["data"]["agents_succeeded"] >= 1

    @pytest.mark.asyncio
    async def test_combined_context_includes_both_agents(self, mock_claude, mock_codex):
        """Both Claude and Codex context should appear in the combined summary."""
        phase = ContextPhase(
            aragora_path=Path("."),
            claude_agent=mock_claude,
            codex_agent=mock_codex,
            log_fn=lambda *args, **kwargs: None,
            stream_emit_fn=lambda *args, **kwargs: None,
        )

        with _patch_streaming(), patch.dict(os.environ, {"ARAGORA_NOMIC_CONTEXT_RLM": "false"}):
            result = await phase.execute()

        summary = result["codebase_summary"]
        assert "CLAUDE" in summary
        assert "CODEX" in summary


class TestPhaseErrorPropagation:
    """Test error handling and propagation through the pipeline."""

    @pytest.mark.asyncio
    async def test_both_agents_fail_uses_fallback(self):
        """When all agents fail, fallback context should be used."""
        failing_claude = _make_mock_agent("Error: connection refused")
        failing_codex = _make_mock_agent("Error: timeout exceeded")

        phase = ContextPhase(
            aragora_path=Path("."),
            claude_agent=failing_claude,
            codex_agent=failing_codex,
            log_fn=lambda *args, **kwargs: None,
            stream_emit_fn=lambda *args, **kwargs: None,
            get_features_fn=lambda: "Fallback: core features list",
        )

        with _patch_streaming(), patch.dict(os.environ, {"ARAGORA_NOMIC_CONTEXT_RLM": "false"}):
            result = await phase.execute()

        assert result["success"]
        assert "Fallback" in result["codebase_summary"]

    @pytest.mark.asyncio
    async def test_one_agent_errors_other_succeeds(self):
        """If one agent errors, the other's context should still be used."""
        good_claude = _make_mock_agent("Claude found 10 modules")
        bad_codex = _make_mock_agent("Error: timeout")

        phase = ContextPhase(
            aragora_path=Path("."),
            claude_agent=good_claude,
            codex_agent=bad_codex,
            log_fn=lambda *args, **kwargs: None,
            stream_emit_fn=lambda *args, **kwargs: None,
        )

        with _patch_streaming(), patch.dict(os.environ, {"ARAGORA_NOMIC_CONTEXT_RLM": "false"}):
            result = await phase.execute()

        assert result["success"]
        assert "Claude found 10 modules" in result["codebase_summary"]

    @pytest.mark.asyncio
    async def test_agent_raises_exception_handled(self):
        """If _gather_with_agent raises an exception, it should be caught."""
        claude = AsyncMock()
        claude.generate = AsyncMock(side_effect=RuntimeError("LLM crash"))
        claude.timeout = None

        codex = _make_mock_agent("Codex result")

        phase = ContextPhase(
            aragora_path=Path("."),
            claude_agent=claude,
            codex_agent=codex,
            log_fn=lambda *args, **kwargs: None,
            stream_emit_fn=lambda *args, **kwargs: None,
        )

        with _patch_streaming(), patch.dict(os.environ, {"ARAGORA_NOMIC_CONTEXT_RLM": "false"}):
            result = await phase.execute()

        assert result["success"]
        # Codex result should still be present
        assert "Codex result" in result["codebase_summary"]

    @pytest.mark.asyncio
    async def test_agent_timeout_handled(self):
        """If agent times out, phase should still complete."""
        claude = AsyncMock()
        claude.generate = AsyncMock(side_effect=asyncio.TimeoutError())
        claude.timeout = 1

        codex = _make_mock_agent("Codex result")

        phase = ContextPhase(
            aragora_path=Path("."),
            claude_agent=claude,
            codex_agent=codex,
            log_fn=lambda *args, **kwargs: None,
            stream_emit_fn=lambda *args, **kwargs: None,
        )

        with _patch_streaming(), patch.dict(os.environ, {"ARAGORA_NOMIC_CONTEXT_RLM": "false"}):
            result = await phase.execute()

        assert result["success"]


class TestPhaseStreamEmit:
    """Test that streaming events are emitted correctly."""

    @pytest.mark.asyncio
    async def test_stream_emit_called_on_start_and_end(self, mock_claude, mock_codex):
        """on_phase_start and on_phase_end should be emitted."""
        emitted = []

        def capture_emit(*args, **kwargs):
            emitted.append(args[0] if args else "unknown")

        phase = ContextPhase(
            aragora_path=Path("."),
            claude_agent=mock_claude,
            codex_agent=mock_codex,
            log_fn=lambda *args, **kwargs: None,
            stream_emit_fn=capture_emit,
        )

        with _patch_streaming(), patch.dict(os.environ, {"ARAGORA_NOMIC_CONTEXT_RLM": "false"}):
            await phase.execute()

        assert "on_phase_start" in emitted
        assert "on_phase_end" in emitted

    @pytest.mark.asyncio
    async def test_log_fn_called(self, mock_claude, mock_codex):
        """log_fn should be called during execution."""
        log_messages = []

        def capture_log(*args, **kwargs):
            log_messages.append(args[0] if args else "")

        phase = ContextPhase(
            aragora_path=Path("."),
            claude_agent=mock_claude,
            codex_agent=mock_codex,
            log_fn=capture_log,
            stream_emit_fn=lambda *args, **kwargs: None,
        )

        with _patch_streaming(), patch.dict(os.environ, {"ARAGORA_NOMIC_CONTEXT_RLM": "false"}):
            await phase.execute()

        assert len(log_messages) > 0


class TestPhaseSkipEnvVars:
    """Test that environment variables control agent skipping."""

    @pytest.mark.asyncio
    async def test_skip_codex_env(self, mock_claude, mock_codex):
        """NOMIC_CONTEXT_SKIP_CODEX=1 should skip Codex."""
        phase = ContextPhase(
            aragora_path=Path("."),
            claude_agent=mock_claude,
            codex_agent=mock_codex,
            log_fn=lambda *args, **kwargs: None,
            stream_emit_fn=lambda *args, **kwargs: None,
        )

        with (
            _patch_streaming(),
            patch.dict(
                os.environ,
                {"NOMIC_CONTEXT_SKIP_CODEX": "1", "ARAGORA_NOMIC_CONTEXT_RLM": "false"},
            ),
        ):
            result = await phase.execute()

        assert result["success"]
        summary = result["codebase_summary"]
        assert "CLAUDE" in summary

    @pytest.mark.asyncio
    async def test_skip_claude_env(self, mock_claude, mock_codex):
        """NOMIC_CONTEXT_SKIP_CLAUDE=1 should skip Claude."""
        phase = ContextPhase(
            aragora_path=Path("."),
            claude_agent=mock_claude,
            codex_agent=mock_codex,
            log_fn=lambda *args, **kwargs: None,
            stream_emit_fn=lambda *args, **kwargs: None,
        )

        with (
            _patch_streaming(),
            patch.dict(
                os.environ,
                {"NOMIC_CONTEXT_SKIP_CLAUDE": "1", "ARAGORA_NOMIC_CONTEXT_RLM": "false"},
            ),
        ):
            result = await phase.execute()

        assert result["success"]
        summary = result["codebase_summary"]
        assert "CODEX" in summary


class TestPhaseKiloCode:
    """Test KiloCode agent integration."""

    @pytest.mark.asyncio
    async def test_kilocode_disabled_by_default(self, mock_claude, mock_codex):
        """Without kilocode_available=True, only Claude + Codex should be used."""
        phase = ContextPhase(
            aragora_path=Path("."),
            claude_agent=mock_claude,
            codex_agent=mock_codex,
            kilocode_available=False,
            log_fn=lambda *args, **kwargs: None,
            stream_emit_fn=lambda *args, **kwargs: None,
        )

        with _patch_streaming(), patch.dict(os.environ, {"ARAGORA_NOMIC_CONTEXT_RLM": "false"}):
            result = await phase.execute()

        assert result["success"]

    @pytest.mark.asyncio
    async def test_kilocode_skip_flag(self, mock_claude, mock_codex):
        """skip_kilocode=True should prevent KiloCode agents even when available."""
        phase = ContextPhase(
            aragora_path=Path("."),
            claude_agent=mock_claude,
            codex_agent=mock_codex,
            kilocode_available=True,
            skip_kilocode=True,
            log_fn=lambda *args, **kwargs: None,
            stream_emit_fn=lambda *args, **kwargs: None,
        )

        with _patch_streaming(), patch.dict(os.environ, {"ARAGORA_NOMIC_CONTEXT_RLM": "false"}):
            result = await phase.execute()

        assert result["success"]


class TestBuildExplorePrompt:
    """Test the exploration prompt generation."""

    def test_prompt_contains_aragora_path(self, mock_claude, mock_codex):
        """The explore prompt should include the aragora_path."""
        phase = ContextPhase(
            aragora_path=Path("/test/path"),
            claude_agent=mock_claude,
            codex_agent=mock_codex,
        )
        prompt = phase._build_explore_prompt()
        assert "/test/path" in prompt

    def test_prompt_includes_critical_instruction(self, mock_claude, mock_codex):
        """The prompt should warn agents to be thorough."""
        phase = ContextPhase(
            aragora_path=Path("."),
            claude_agent=mock_claude,
            codex_agent=mock_codex,
        )
        prompt = phase._build_explore_prompt()
        assert "CRITICAL" in prompt
        assert "EXISTING FEATURES" in prompt
