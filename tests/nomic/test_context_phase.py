"""
Tests for Nomic Loop Context Phase.

Phase 0: Context gathering
- Tests agent exploration
- Tests context aggregation
- Tests fallback behavior
- Tests error handling
- Tests metrics recording
- Tests ContextResult TypedDict
- Tests environment variable controls
- Tests kilocode / builder interactions
"""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.phases.context import ContextPhase, ContextResult, set_metrics_recorder


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

    def test_init_with_optional_args(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Should initialize with optional arguments."""
        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            kilocode_available=True,
            skip_kilocode=False,
            cycle_count=5,
            log_fn=mock_log_fn,
        )
        assert phase.kilocode_available is True
        assert phase.skip_kilocode is False
        assert phase.cycle_count == 5

    def test_init_defaults(self, mock_aragora_path, mock_claude_agent, mock_codex_agent):
        """Default values for optional arguments."""
        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
        )
        assert phase.kilocode_available is False
        assert phase.skip_kilocode is False
        assert phase.cycle_count == 0
        assert phase._context_builder is None

    def test_init_with_context_builder(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent
    ):
        """Context builder should be stored when provided."""
        builder = MagicMock()
        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            context_builder=builder,
        )
        assert phase._context_builder is builder

    def test_init_with_get_features_fn(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent
    ):
        """Custom get_features_fn should be stored."""

        def fn():
            return "custom features"

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            get_features_fn=fn,
        )
        assert phase._get_features() == "custom features"

    def test_init_default_get_features(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent
    ):
        """Default get_features_fn should return a string."""
        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
        )
        result = phase._get_features()
        assert isinstance(result, str)


class TestContextPhaseExecution:
    """Tests for context phase execution."""

    @pytest.mark.asyncio
    async def test_run_gathers_context_from_agents(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Should gather context from multiple agents."""
        mock_claude_agent.generate = AsyncMock(return_value="Claude context: found 5 modules")
        mock_codex_agent.generate = AsyncMock(return_value="Codex context: found API handlers")

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "_gather_with_agent", new_callable=AsyncMock) as mock_gather:
            mock_gather.return_value = ("claude", "Claude Code", "Found modules")
            result = await phase.execute()

            assert isinstance(result, dict)
            assert "codebase_summary" in result

    @pytest.mark.asyncio
    async def test_run_handles_agent_timeout(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Should handle agent timeout gracefully."""
        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "_gather_with_agent", new_callable=AsyncMock) as mock_gather:
            mock_gather.side_effect = asyncio.TimeoutError("Agent timeout")
            result = await phase.execute()
            assert result is not None

    @pytest.mark.asyncio
    async def test_run_uses_fallback_on_empty_context(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Should use fallback when no context gathered."""
        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
            get_features_fn=lambda: "Fallback features list",
        )

        with patch.object(phase, "_gather_with_agent", new_callable=AsyncMock) as mock_gather:
            mock_gather.return_value = ("claude", "Claude Code", "Error: timeout")
            result = await phase.execute()
            assert result is not None

    @pytest.mark.asyncio
    async def test_run_uses_context_builder_when_enabled(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn, monkeypatch
    ):
        """Should use the provided context builder when RLM context is enabled."""

        class DummyBuilder:
            async def build_debate_context(self):
                return "RLM MAP"

            async def build_rlm_context(self):
                return object()

        monkeypatch.setenv("ARAGORA_NOMIC_CONTEXT_RLM", "true")

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
            context_builder=DummyBuilder(),
        )

        with patch.object(phase, "_gather_with_agent", new_callable=AsyncMock) as mock_gather:
            mock_gather.return_value = ("claude", "Claude Code", "Context gathered")
            result = await phase.execute()

        assert result["codebase_summary"] == "RLM MAP"

    @pytest.mark.asyncio
    async def test_run_returns_success_true(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Successful execution should return success=True."""
        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "_gather_with_agent", new_callable=AsyncMock) as mock_gather:
            mock_gather.return_value = ("claude", "Claude Code", "context data")
            result = await phase.execute()

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_run_records_duration(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Execution should record non-negative duration."""
        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "_gather_with_agent", new_callable=AsyncMock) as mock_gather:
            mock_gather.return_value = ("claude", "Claude Code", "context")
            result = await phase.execute()

        assert result["duration_seconds"] >= 0


class TestContextPhaseMetrics:
    """Tests for context phase metrics recording."""

    @pytest.mark.asyncio
    async def test_records_phase_metrics(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Should record phase execution metrics."""
        metrics_recorded = []

        def mock_phase_recorder(phase: str, outcome: str, duration: float):
            metrics_recorded.append((phase, outcome, duration))

        set_metrics_recorder(phase_recorder=mock_phase_recorder)

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "_gather_with_agent", new_callable=AsyncMock) as mock_gather:
            mock_gather.return_value = ("claude", "Claude Code", "Context gathered")
            await phase.execute()

        set_metrics_recorder(None, None)

        assert len(metrics_recorded) > 0
        assert metrics_recorded[0][0] == "context"

    @pytest.mark.asyncio
    async def test_records_success_outcome(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Metrics should record 'success' outcome on successful execution."""
        metrics_recorded = []

        def recorder(phase, outcome, duration):
            metrics_recorded.append((phase, outcome, duration))

        set_metrics_recorder(phase_recorder=recorder)

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "_gather_with_agent", new_callable=AsyncMock) as mock_gather:
            mock_gather.return_value = ("claude", "Claude Code", "context")
            await phase.execute()

        set_metrics_recorder(None, None)

        assert any(m[1] == "success" for m in metrics_recorded)

    @pytest.mark.asyncio
    async def test_no_metrics_when_recorder_unset(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Without a metrics recorder set, execution should still succeed."""
        set_metrics_recorder(None, None)

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "_gather_with_agent", new_callable=AsyncMock) as mock_gather:
            mock_gather.return_value = ("claude", "Claude Code", "context")
            result = await phase.execute()

        assert result["success"] is True


class TestContextResult:
    """Tests for ContextResult TypedDict."""

    def test_context_result_creation(self):
        """Should create ContextResult with expected fields."""
        result = ContextResult(
            success=True,
            data={"agents_succeeded": 2},
            codebase_summary="Combined: modules and handlers",
            recent_changes="",
            open_issues=[],
            duration_seconds=2.5,
        )

        assert result["success"] is True
        assert result["codebase_summary"] == "Combined: modules and handlers"
        assert result["recent_changes"] == ""
        assert result["duration_seconds"] == 2.5

    def test_context_result_with_empty_values(self):
        """Should allow empty context values."""
        result = ContextResult(
            success=True,
            data={},
            codebase_summary="Fallback context",
            recent_changes="",
            open_issues=[],
            duration_seconds=0.1,
        )

        assert result["codebase_summary"] == "Fallback context"
        assert result["open_issues"] == []

    def test_context_result_with_failed_status(self):
        """ContextResult with success=False should work."""
        result = ContextResult(
            success=False,
            data={"error": "All agents timed out"},
            codebase_summary="",
            recent_changes="",
            open_issues=[],
            duration_seconds=120.0,
        )

        assert result["success"] is False
        assert result["data"]["error"] == "All agents timed out"

    def test_context_result_with_open_issues(self):
        """ContextResult should support a list of open issues."""
        result = ContextResult(
            success=True,
            data={},
            codebase_summary="summary",
            recent_changes="commit abc",
            open_issues=["Issue #1: bug", "Issue #2: feature request"],
            duration_seconds=1.0,
        )

        assert len(result["open_issues"]) == 2
        assert result["recent_changes"] == "commit abc"

    def test_context_result_is_dict(self):
        """ContextResult should be usable as a regular dict."""
        result = ContextResult(
            success=True,
            data={},
            codebase_summary="test",
            recent_changes="",
            open_issues=[],
            duration_seconds=0.0,
        )
        assert isinstance(result, dict)
        assert "success" in result


class TestContextPhaseExplorePrompt:
    """Tests for the exploration prompt builder."""

    def test_build_explore_prompt_contains_path(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent
    ):
        """Prompt should include the aragora path."""
        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
        )
        prompt = phase._build_explore_prompt()
        assert str(mock_aragora_path) in prompt

    def test_build_explore_prompt_sections(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent
    ):
        """Prompt should request specific sections."""
        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
        )
        prompt = phase._build_explore_prompt()
        assert "EXISTING FEATURES" in prompt
        assert "ARCHITECTURE OVERVIEW" in prompt
        assert "GAPS AND OPPORTUNITIES" in prompt

    def test_build_explore_prompt_warns_about_thoroughness(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent
    ):
        """Prompt should warn about being thorough."""
        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
        )
        prompt = phase._build_explore_prompt()
        assert "CRITICAL" in prompt


class TestSetMetricsRecorder:
    """Tests for the set_metrics_recorder function."""

    def test_set_both_recorders(self):
        """Should set both phase and agent recorders."""
        phase_fn = MagicMock()
        agent_fn = MagicMock()
        set_metrics_recorder(phase_recorder=phase_fn, agent_recorder=agent_fn)
        set_metrics_recorder(None, None)

    def test_set_phase_recorder_only(self):
        """Should allow setting just the phase recorder."""
        phase_fn = MagicMock()
        set_metrics_recorder(phase_recorder=phase_fn)
        set_metrics_recorder(None, None)

    def test_reset_recorders(self):
        """Should be able to reset both recorders to None."""
        set_metrics_recorder(phase_recorder=MagicMock(), agent_recorder=MagicMock())
        set_metrics_recorder(None, None)
