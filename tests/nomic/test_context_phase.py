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

        with patch.object(phase, "_gather_claude_context", new_callable=AsyncMock) as mock_claude:
            with patch.object(phase, "_gather_codex_context", new_callable=AsyncMock) as mock_codex:
                mock_claude.return_value = ("Claude found modules", 1.5)
                mock_codex.return_value = ("Codex found handlers", 1.2)

                result = await phase.run()

                assert isinstance(result, ContextResult)
                mock_claude.assert_called_once()
                mock_codex.assert_called_once()

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

        with patch.object(phase, "_gather_claude_context", new_callable=AsyncMock) as mock_claude:
            with patch.object(phase, "_gather_codex_context", new_callable=AsyncMock) as mock_codex:
                mock_claude.side_effect = asyncio.TimeoutError("Agent timeout")
                mock_codex.return_value = ("Codex found handlers", 1.2)

                # Should not raise, should handle gracefully
                result = await phase.run()
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

        with patch.object(phase, "_gather_claude_context", new_callable=AsyncMock) as mock_claude:
            with patch.object(phase, "_gather_codex_context", new_callable=AsyncMock) as mock_codex:
                mock_claude.return_value = (None, 0)
                mock_codex.return_value = (None, 0)

                result = await phase.run()

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

        with patch.object(phase, "_gather_claude_context", new_callable=AsyncMock) as mock_claude:
            with patch.object(phase, "_gather_codex_context", new_callable=AsyncMock) as mock_codex:
                mock_claude.return_value = ("Context", 1.0)
                mock_codex.return_value = ("Context", 1.0)

                await phase.run()

        # Reset metrics recorder
        set_metrics_recorder(None, None)

        # Check metrics were recorded
        assert len(metrics_recorded) > 0
        assert metrics_recorded[0][0] == "context"


class TestContextResult:
    """Tests for ContextResult dataclass."""

    def test_context_result_creation(self):
        """Should create ContextResult with expected fields."""
        result = ContextResult(
            claude_context="Claude found modules",
            codex_context="Codex found handlers",
            combined_context="Combined: modules and handlers",
            elapsed_seconds=2.5,
        )

        assert result.claude_context == "Claude found modules"
        assert result.codex_context == "Codex found handlers"
        assert result.combined_context == "Combined: modules and handlers"
        assert result.elapsed_seconds == 2.5

    def test_context_result_with_empty_values(self):
        """Should allow empty context values."""
        result = ContextResult(
            claude_context=None,
            codex_context=None,
            combined_context="Fallback context",
            elapsed_seconds=0.1,
        )

        assert result.claude_context is None
        assert result.codex_context is None
        assert result.combined_context == "Fallback context"
