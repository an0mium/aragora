"""
Tests for Nomic Loop Context Phase.

Phase 0: Context gathering
- Tests agent exploration
- Tests context aggregation
- Tests fallback behavior
- Tests error handling
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.phases.context import ContextPhase, ContextResult


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


class TestContextPhaseExecution:
    """Tests for context phase execution."""

    @pytest.mark.asyncio
    async def test_run_gathers_context_from_agents(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Should gather context from multiple agents."""
        # Setup mock responses
        mock_claude_agent.generate = AsyncMock(return_value="Claude context: found 5 modules")
        mock_codex_agent.generate = AsyncMock(return_value="Codex context: found API handlers")

        phase = ContextPhase(
            aragora_path=mock_aragora_path,
            claude_agent=mock_claude_agent,
            codex_agent=mock_codex_agent,
            log_fn=mock_log_fn,
        )

        with patch.object(phase, "_gather_with_agent", new_callable=AsyncMock) as mock_gather:
            # Return format: (name, harness, content)
            mock_gather.return_value = ("claude", "Claude Code", "Found modules")

            result = await phase.execute()

            assert isinstance(result, dict)  # ContextResult is a TypedDict
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

            # Should not raise, should handle gracefully
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
            # Return error content to trigger fallback
            mock_gather.return_value = ("claude", "Claude Code", "Error: timeout")

            result = await phase.execute()

            # Should still return a result using fallback
            assert result is not None


class TestContextPhaseMetrics:
    """Tests for context phase metrics recording."""

    @pytest.mark.asyncio
    async def test_records_phase_metrics(
        self, mock_aragora_path, mock_claude_agent, mock_codex_agent, mock_log_fn
    ):
        """Should record phase execution metrics."""
        from aragora.nomic.phases.context import set_metrics_recorder

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
            # Return format: (name, harness, content)
            mock_gather.return_value = ("claude", "Claude Code", "Context gathered")

            await phase.execute()

        # Reset metrics recorder
        set_metrics_recorder(None, None)

        # Check metrics were recorded
        assert len(metrics_recorded) > 0
        assert metrics_recorded[0][0] == "context"


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
