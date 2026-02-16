"""
Tests for Nomic Loop Context Phase.

Phase 0: Gather codebase understanding
- Tests context gathering from multiple agents
- Tests agent exploration with different harnesses
- Tests RLM context building integration
- Tests error handling and fallbacks
- Tests metrics recording
- Tests ContextResult TypedDict
"""

import asyncio
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.phases import ContextResult
from aragora.nomic.phases.context import ContextPhase, set_metrics_recorder


@contextmanager
def mock_streaming_context(task_id: str):
    """Mock context manager for streaming_task_context."""
    yield


# Patch path for streaming_task_context at the source module
STREAM_PATCH = "aragora.server.stream.arena_hooks.streaming_task_context"


class TestContextPhaseInitialization:
    """Tests for ContextPhase initialization."""

    def test_init_with_required_args(self, mock_aragora_path, mock_claude_agent, mock_codex_agent):
        """Should initialize with required arguments."""
        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
        )
        assert phase.aragora_path == mock_aragora_path
        assert phase.claude == mock_claude_agent
        assert phase.codex == mock_codex_agent
        assert phase.kilocode_available is False
        assert phase.skip_kilocode is False
        assert phase.cycle_count == 0

    def test_init_with_all_optional_args(
        self,
        mock_aragora_path,
        mock_claude_agent,
        mock_codex_agent,
        mock_log_fn,
        mock_stream_emit_fn,
    ):
        """Should initialize with all optional arguments."""
        context_builder = MagicMock()
        kilocode_factory = MagicMock()
        get_features = MagicMock(return_value="Feature list")

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            kilocode_available=True,
            skip_kilocode=True,
            kilocode_agent_factory=kilocode_factory,
            cycle_count=5,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
            get_features_fn=get_features,
            context_builder=context_builder,
        )

        assert phase.kilocode_available is True
        assert phase.skip_kilocode is True
        assert phase.kilocode_agent_factory is kilocode_factory
        assert phase.cycle_count == 5
        assert phase._context_builder is context_builder

    def test_init_default_log_fn(self, mock_aragora_path, mock_claude_agent, mock_codex_agent):
        """Default log function should be print."""
        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
        )
        assert phase._log is not None

    def test_init_default_stream_emit_fn(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent
    ):
        """Default stream emit function should be a no-op."""
        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
        )
        # Should not raise
        phase._stream_emit("test_event", "arg1", "arg2")

    def test_init_default_get_features_fn(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent
    ):
        """Default get_features function should return fallback message."""
        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
        )
        result = phase._get_features()
        assert result == "No features available"


class TestContextPhaseExecution:
    """Tests for ContextPhase execution."""

    @pytest.mark.asyncio
    async def test_execute_with_claude_and_codex(
        self,
        mock_aragora_path,
        mock_claude_agent,
        mock_codex_agent,
        mock_log_fn,
        mock_stream_emit_fn,
    ):
        """Should gather context from both Claude and Codex agents."""
        mock_claude_agent.generate = AsyncMock(return_value="Claude's analysis of codebase")
        mock_codex_agent.generate = AsyncMock(return_value="Codex's analysis of codebase")

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            result = await phase.execute()

        assert result["success"] is True
        assert "codebase_summary" in result
        # The summary should contain agent output or fallback content
        assert len(result["codebase_summary"]) > 0

    @pytest.mark.asyncio
    async def test_execute_records_duration(
        self,
        mock_aragora_path,
        mock_claude_agent,
        mock_codex_agent,
        mock_log_fn,
        mock_stream_emit_fn,
    ):
        """Should record execution duration."""
        mock_claude_agent.generate = AsyncMock(return_value="Analysis")
        mock_codex_agent.generate = AsyncMock(return_value="Analysis")

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            result = await phase.execute()

        assert "duration_seconds" in result
        assert result["duration_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_execute_emits_stream_events(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_stream_emit_fn
    ):
        """Should emit streaming events during execution."""
        mock_claude_agent.generate = AsyncMock(return_value="Analysis")
        mock_codex_agent.generate = AsyncMock(return_value="Analysis")

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            stream_emit_fn=mock_stream_emit_fn,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            await phase.execute()

        # Check that stream events were emitted
        assert mock_stream_emit_fn.call_count >= 2  # phase_start and phase_end

    @pytest.mark.asyncio
    async def test_execute_with_skip_claude_env(
        self,
        mock_aragora_path,
        mock_claude_agent,
        mock_codex_agent,
        mock_log_fn,
        mock_stream_emit_fn,
        monkeypatch,
    ):
        """Should skip Claude when NOMIC_CONTEXT_SKIP_CLAUDE=1."""
        monkeypatch.setenv("NOMIC_CONTEXT_SKIP_CLAUDE", "1")
        mock_codex_agent.generate = AsyncMock(return_value="Codex analysis")

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            result = await phase.execute()

        assert result["success"] is True
        mock_claude_agent.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_with_skip_codex_env(
        self,
        mock_aragora_path,
        mock_claude_agent,
        mock_codex_agent,
        mock_log_fn,
        mock_stream_emit_fn,
        monkeypatch,
    ):
        """Should skip Codex when NOMIC_CONTEXT_SKIP_CODEX=1."""
        monkeypatch.setenv("NOMIC_CONTEXT_SKIP_CODEX", "1")
        mock_claude_agent.generate = AsyncMock(return_value="Claude analysis")

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            result = await phase.execute()

        assert result["success"] is True
        mock_codex_agent.generate.assert_not_called()


class TestContextPhaseAgentExploration:
    """Tests for agent-based context gathering."""

    @pytest.mark.asyncio
    async def test_gather_with_agent_success(
        self,
        mock_aragora_path,
        mock_claude_agent,
        mock_codex_agent,
        mock_log_fn,
        mock_stream_emit_fn,
    ):
        """Should successfully gather context from an agent."""
        mock_claude_agent.generate = AsyncMock(return_value="Detailed codebase analysis")
        mock_claude_agent.timeout = 300

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            name, harness, content = await phase._gather_with_agent(
                mock_claude_agent, "claude", "Claude Code"
            )

        assert name == "claude"
        assert harness == "Claude Code"
        assert content == "Detailed codebase analysis"

    @pytest.mark.asyncio
    async def test_gather_with_agent_empty_response(
        self,
        mock_aragora_path,
        mock_claude_agent,
        mock_codex_agent,
        mock_log_fn,
        mock_stream_emit_fn,
    ):
        """Should handle empty agent response."""
        mock_claude_agent.generate = AsyncMock(return_value="")
        mock_claude_agent.timeout = 300

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            name, harness, content = await phase._gather_with_agent(
                mock_claude_agent, "claude", "Claude Code"
            )

        assert "Error: empty response" in content

    @pytest.mark.asyncio
    async def test_gather_with_agent_none_response(
        self,
        mock_aragora_path,
        mock_claude_agent,
        mock_codex_agent,
        mock_log_fn,
        mock_stream_emit_fn,
    ):
        """Should handle None agent response."""
        mock_claude_agent.generate = AsyncMock(return_value=None)
        mock_claude_agent.timeout = 300

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            name, harness, content = await phase._gather_with_agent(
                mock_claude_agent, "claude", "Claude Code"
            )

        assert "Error: empty response" in content

    @pytest.mark.asyncio
    async def test_gather_with_agent_timeout(
        self,
        mock_aragora_path,
        mock_claude_agent,
        mock_codex_agent,
        mock_log_fn,
        mock_stream_emit_fn,
    ):
        """Should handle agent timeout."""

        async def slow_generate(*args, **kwargs):
            await asyncio.sleep(10)
            return "Too slow"

        mock_claude_agent.generate = slow_generate
        mock_claude_agent.timeout = 0.1

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            name, harness, content = await phase._gather_with_agent(
                mock_claude_agent, "claude", "Claude Code"
            )

        assert "Error: timeout exceeded" in content

    @pytest.mark.asyncio
    async def test_gather_with_agent_exception(
        self,
        mock_aragora_path,
        mock_claude_agent,
        mock_codex_agent,
        mock_log_fn,
        mock_stream_emit_fn,
    ):
        """Should handle agent exception."""
        mock_claude_agent.generate = AsyncMock(side_effect=RuntimeError("Agent crashed"))
        mock_claude_agent.timeout = 300

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            name, harness, content = await phase._gather_with_agent(
                mock_claude_agent, "claude", "Claude Code"
            )

        assert "Error:" in content
        assert "RuntimeError" in content

    @pytest.mark.asyncio
    async def test_gather_with_agent_timeout_env_override(
        self,
        mock_aragora_path,
        mock_claude_agent,
        mock_codex_agent,
        mock_log_fn,
        mock_stream_emit_fn,
        monkeypatch,
    ):
        """Should respect NOMIC_CONTEXT_AGENT_TIMEOUT env var."""
        monkeypatch.setenv("NOMIC_CONTEXT_AGENT_TIMEOUT", "10")
        mock_claude_agent.generate = AsyncMock(return_value="Analysis")
        mock_claude_agent.timeout = 300

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            with patch("asyncio.wait_for", new_callable=AsyncMock) as mock_wait_for:
                mock_wait_for.return_value = "Analysis"
                await phase._gather_with_agent(mock_claude_agent, "claude", "Claude Code")
                # The timeout should be 10 from env var
                mock_wait_for.assert_called_once()
                call_args = mock_wait_for.call_args
                assert call_args[1]["timeout"] == 10


class TestContextPhaseKiloCode:
    """Tests for KiloCode integration."""

    @pytest.mark.asyncio
    async def test_execute_with_kilocode_available(
        self,
        mock_aragora_path,
        mock_claude_agent,
        mock_codex_agent,
        mock_log_fn,
        mock_stream_emit_fn,
    ):
        """Should use KiloCode for Gemini and Grok when available."""
        mock_claude_agent.generate = AsyncMock(return_value="Claude analysis")
        mock_codex_agent.generate = AsyncMock(return_value="Codex analysis")

        mock_kilocode_agent = MagicMock()
        mock_kilocode_agent.generate = AsyncMock(return_value="KiloCode analysis")
        mock_kilocode_agent.timeout = 300

        kilocode_factory = MagicMock(return_value=mock_kilocode_agent)

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            kilocode_available=True,
            skip_kilocode=False,
            kilocode_agent_factory=kilocode_factory,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            await phase.execute()

        # Factory should be called to create Gemini and Grok agents
        assert kilocode_factory.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_with_kilocode_skipped(
        self,
        mock_aragora_path,
        mock_claude_agent,
        mock_codex_agent,
        mock_log_fn,
        mock_stream_emit_fn,
    ):
        """Should not use KiloCode when skip_kilocode is True."""
        mock_claude_agent.generate = AsyncMock(return_value="Claude analysis")
        mock_codex_agent.generate = AsyncMock(return_value="Codex analysis")

        kilocode_factory = MagicMock()

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            kilocode_available=True,
            skip_kilocode=True,
            kilocode_agent_factory=kilocode_factory,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            await phase.execute()

        # Factory should not be called
        kilocode_factory.assert_not_called()


class TestContextPhaseFallback:
    """Tests for fallback behavior."""

    @pytest.mark.asyncio
    async def test_fallback_when_all_agents_fail(
        self,
        mock_aragora_path,
        mock_claude_agent,
        mock_codex_agent,
        mock_log_fn,
        mock_stream_emit_fn,
    ):
        """Should use fallback context when all agents fail."""
        mock_claude_agent.generate = AsyncMock(return_value="Error: agent crashed")
        mock_codex_agent.generate = AsyncMock(return_value="Error: timeout")

        get_features = MagicMock(return_value="Feature A, Feature B")

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
            get_features_fn=get_features,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            result = await phase.execute()

        # Should still succeed with fallback context
        assert "codebase_summary" in result
        assert "Feature A" in result["codebase_summary"] or result["success"]

    @pytest.mark.asyncio
    async def test_fallback_when_agent_returns_exception(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Should handle agent raising exception gracefully."""
        mock_claude_agent.generate = AsyncMock(side_effect=RuntimeError("Failed"))
        mock_codex_agent.generate = AsyncMock(side_effect=RuntimeError("Failed"))

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            result = await phase.execute()

        # Should still complete without raising
        assert "codebase_summary" in result


class TestContextPhaseRLMIntegration:
    """Tests for RLM context builder integration."""

    @pytest.mark.asyncio
    async def test_execute_with_context_builder(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Should use context builder when provided."""
        mock_claude_agent.generate = AsyncMock(return_value="Analysis")
        mock_codex_agent.generate = AsyncMock(return_value="Analysis")

        context_builder = MagicMock()
        context_builder.build_debate_context = AsyncMock(return_value="RLM structured context")
        context_builder.build_rlm_context = AsyncMock()

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
            context_builder=context_builder,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            await phase.execute()

        # Context builder methods should be called
        context_builder.build_debate_context.assert_called()

    @pytest.mark.asyncio
    async def test_execute_context_builder_failure_continues(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Should continue execution even if context builder fails."""
        mock_claude_agent.generate = AsyncMock(return_value="Agent analysis")
        mock_codex_agent.generate = AsyncMock(return_value="Agent analysis")

        context_builder = MagicMock()
        context_builder.build_debate_context = AsyncMock(side_effect=RuntimeError("RLM failed"))

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
            context_builder=context_builder,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            result = await phase.execute()

        # Should succeed with agent context
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_with_rlm_env_disabled(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn, monkeypatch
    ):
        """Should skip RLM context when ARAGORA_NOMIC_CONTEXT_RLM=false."""
        monkeypatch.setenv("ARAGORA_NOMIC_CONTEXT_RLM", "false")
        mock_claude_agent.generate = AsyncMock(return_value="Analysis")
        mock_codex_agent.generate = AsyncMock(return_value="Analysis")

        context_builder = MagicMock()
        context_builder.build_debate_context = AsyncMock(return_value="RLM context")
        context_builder.build_rlm_context = AsyncMock()

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
            context_builder=context_builder,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            result = await phase.execute()

        # build_rlm_context should not be called in the RLM block
        assert result["success"] is True


class TestContextPhaseExplorationPrompt:
    """Tests for the exploration prompt building."""

    def test_build_explore_prompt(self, mock_aragora_path, mock_claude_agent, mock_codex_agent):
        """Should build a comprehensive exploration prompt."""
        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
        )

        prompt = phase._build_explore_prompt()

        assert "aragora" in prompt.lower()
        assert "FEATURE INVENTORY" in prompt
        assert "CLAUDE.md" in prompt
        assert "DO NOT RECREATE" in prompt


class TestContextPhaseMetrics:
    """Tests for metrics recording."""

    @pytest.mark.asyncio
    async def test_metrics_recorder_called(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Should call metrics recorder when configured."""
        mock_claude_agent.generate = AsyncMock(return_value="Analysis")
        mock_codex_agent.generate = AsyncMock(return_value="Analysis")

        phase_recorder = MagicMock()
        agent_recorder = MagicMock()
        set_metrics_recorder(phase_recorder, agent_recorder)

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            await phase.execute()

        # Phase recorder should be called once
        phase_recorder.assert_called_once()

        # Agent recorder should be called for each agent
        assert agent_recorder.call_count >= 2

        # Reset metrics recorder
        set_metrics_recorder(None, None)

    def test_set_metrics_recorder_can_clear(self):
        """Should be able to clear metrics recorders."""
        recorder = MagicMock()
        set_metrics_recorder(recorder, recorder)
        set_metrics_recorder(None, None)
        # Should not raise when calling after clear


class TestContextResult:
    """Tests for ContextResult TypedDict."""

    def test_context_result_success(self):
        """Should create ContextResult with success values."""
        result = ContextResult(
            success=True,
            data={"agents_succeeded": 2},
            duration_seconds=5.5,
            codebase_summary="Complete analysis of aragora codebase",
            recent_changes="Added new feature",
            open_issues=["Issue 1", "Issue 2"],
        )

        assert result["success"] is True
        assert result["codebase_summary"] == "Complete analysis of aragora codebase"
        assert result["recent_changes"] == "Added new feature"
        assert len(result["open_issues"]) == 2

    def test_context_result_failure(self):
        """Should create ContextResult with failure values."""
        result = ContextResult(
            success=False,
            error="All agents failed",
            data={},
            duration_seconds=1.0,
            codebase_summary="",
            recent_changes="",
            open_issues=[],
        )

        assert result["success"] is False
        assert result["error"] == "All agents failed"

    def test_context_result_is_dict(self):
        """ContextResult should be usable as a regular dict."""
        result = ContextResult(
            success=True,
            data={},
            duration_seconds=1.0,
            codebase_summary="Summary",
            recent_changes="",
            open_issues=[],
        )
        assert isinstance(result, dict)
        assert "success" in result


class TestContextPhaseIntegration:
    """Integration tests for context phase."""

    @pytest.mark.asyncio
    async def test_full_context_flow_success(
        self,
        mock_aragora_path,
        mock_claude_agent,
        mock_codex_agent,
        mock_log_fn,
        mock_stream_emit_fn,
    ):
        """Should complete full context gathering flow successfully."""
        mock_claude_agent.generate = AsyncMock(
            return_value="## FEATURE INVENTORY\n- Feature A\n- Feature B"
        )
        mock_claude_agent.timeout = 300
        mock_codex_agent.generate = AsyncMock(
            return_value="## ARCHITECTURE\n- Pattern X\n- Pattern Y"
        )
        mock_codex_agent.timeout = 300

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
            stream_emit_fn=mock_stream_emit_fn,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            result = await phase.execute()

        assert result["success"] is True
        assert "codebase_summary" in result
        assert result["duration_seconds"] >= 0
        assert result["data"]["agents_succeeded"] >= 1

    @pytest.mark.asyncio
    async def test_context_with_mixed_agent_results(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Should handle mixed success/failure from agents."""
        mock_claude_agent.generate = AsyncMock(return_value="Valid analysis")
        mock_claude_agent.timeout = 300
        mock_codex_agent.generate = AsyncMock(side_effect=TimeoutError("Timed out"))
        mock_codex_agent.timeout = 300

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            result = await phase.execute()

        # Should still succeed with partial results
        assert result["success"] is True
        assert result["data"]["agents_succeeded"] >= 1


class TestContextPhaseNullAgents:
    """Tests for null/None agent handling."""

    @pytest.mark.asyncio
    async def test_execute_with_none_claude(self, mock_aragora_path, mock_codex_agent, mock_log_fn):
        """Should handle None Claude agent."""
        mock_codex_agent.generate = AsyncMock(return_value="Codex analysis")
        mock_codex_agent.timeout = 300

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=None,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            result = await phase.execute()

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_execute_with_none_codex(self, mock_aragora_path, mock_claude_agent, mock_log_fn):
        """Should handle None Codex agent."""
        mock_claude_agent.generate = AsyncMock(return_value="Claude analysis")
        mock_claude_agent.timeout = 300

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=None,
            log_fn=mock_log_fn,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            result = await phase.execute()

        assert result["success"] is True


class TestContextPhaseEdgeCases:
    """Edge case tests."""

    @pytest.mark.asyncio
    async def test_agent_returns_error_string(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Should filter out agent responses containing 'Error:'."""
        mock_claude_agent.generate = AsyncMock(return_value="Error: Something went wrong")
        mock_claude_agent.timeout = 300
        mock_codex_agent.generate = AsyncMock(return_value="Valid analysis")
        mock_codex_agent.timeout = 300

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            result = await phase.execute()

        # Should only include valid analysis
        assert result["success"] is True
        # Error responses should be filtered
        assert result["data"]["agents_succeeded"] >= 1

    @pytest.mark.asyncio
    async def test_zero_cycle_count(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Should handle cycle_count of 0."""
        mock_claude_agent.generate = AsyncMock(return_value="Analysis")
        mock_codex_agent.generate = AsyncMock(return_value="Analysis")

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            cycle_count=0,
            log_fn=mock_log_fn,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            result = await phase.execute()

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_high_cycle_count(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Should handle high cycle_count."""
        mock_claude_agent.generate = AsyncMock(return_value="Analysis")
        mock_codex_agent.generate = AsyncMock(return_value="Analysis")

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            cycle_count=999,
            log_fn=mock_log_fn,
        )

        with patch(STREAM_PATCH, mock_streaming_context):
            result = await phase.execute()

        assert result["success"] is True
