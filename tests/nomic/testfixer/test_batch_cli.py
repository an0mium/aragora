"""Tests for batch mode CLI integration."""

from __future__ import annotations

import argparse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.cli.commands.testfixer import build_parser, _run


class TestBatchCLIArgs:
    """Test that batch CLI arguments are parsed correctly."""

    def _parse(self, *args: str) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        subs = parser.add_subparsers()
        build_parser(subs)
        return parser.parse_args(["testfixer", *args])

    def test_batch_flag(self):
        ns = self._parse("/tmp/repo", "--batch")
        assert ns.batch is True

    def test_max_batch_size(self):
        ns = self._parse("/tmp/repo", "--batch", "--max-batch-size", "10")
        assert ns.max_batch_size == 10

    def test_elo_select(self):
        ns = self._parse("/tmp/repo", "--batch", "--elo-select")
        assert ns.elo_select is True

    def test_elo_fallback_agents(self):
        ns = self._parse(
            "/tmp/repo", "--batch", "--elo-fallback-agents", "anthropic-api,openai-api"
        )
        assert ns.elo_fallback_agents == "anthropic-api,openai-api"

    def test_bug_check_flag(self):
        ns = self._parse("/tmp/repo", "--batch", "--bug-check")
        assert ns.bug_check is True

    def test_impact_analysis_flag(self):
        ns = self._parse("/tmp/repo", "--batch", "--impact-analysis")
        assert ns.impact_analysis is True

    def test_batch_test_command(self):
        ns = self._parse("/tmp/repo", "--batch", "--batch-test-command", "pytest tests/ -v")
        assert ns.batch_test_command == "pytest tests/ -v"

    def test_defaults_without_batch(self):
        ns = self._parse("/tmp/repo")
        assert ns.batch is False
        assert ns.max_batch_size == 5
        assert ns.elo_select is False
        assert ns.bug_check is False
        assert ns.impact_analysis is False


class TestBatchCLIDispatch:
    """Test that --batch dispatches to run_batch_fix_loop."""

    @pytest.mark.asyncio
    async def test_batch_dispatches_to_batch_loop(self):
        ns = argparse.Namespace(
            repo_path="/tmp/repo",
            test_command="pytest tests/ -q --maxfail=1",
            agents="none",
            max_iterations=1,
            min_confidence=0.5,
            min_confidence_auto=0.7,
            timeout_seconds=10.0,
            attempt_store=None,
            require_consensus=False,
            no_revert=False,
            require_approval=False,
            log_file="-",
            log_level="warning",
            run_id="test",
            artifacts_dir=None,
            no_diagnostics=True,
            llm_analyzer=False,
            analysis_agents="",
            analysis_require_consensus=False,
            analysis_consensus_threshold=0.7,
            arena_validate=False,
            arena_agents="",
            arena_rounds=2,
            arena_min_confidence=0.6,
            arena_require_consensus=False,
            arena_consensus_threshold=0.7,
            redteam_validate=False,
            redteam_attackers="",
            redteam_defender="",
            redteam_rounds=2,
            redteam_attacks_per_round=3,
            redteam_min_robustness=0.6,
            pattern_learning=False,
            pattern_store=None,
            generation_timeout_seconds=600.0,
            critique_timeout_seconds=300.0,
            batch=True,
            max_batch_size=5,
            elo_select=False,
            elo_fallback_agents="",
            bug_check=False,
            impact_analysis=False,
            batch_test_command=None,
        )

        mock_result = MagicMock()
        mock_result.summary.return_value = "BatchFixer success: 1/1"
        mock_result.status.value = "success"

        with patch("aragora.cli.commands.testfixer.TestFixerOrchestrator") as MockOrch:
            instance = MockOrch.return_value
            instance.run_batch_fix_loop = AsyncMock(return_value=mock_result)

            exit_code = await _run(ns)

            instance.run_batch_fix_loop.assert_called_once()
            assert exit_code == 0

    @pytest.mark.asyncio
    async def test_non_batch_dispatches_to_fix_loop(self):
        ns = argparse.Namespace(
            repo_path="/tmp/repo",
            test_command="pytest tests/ -q --maxfail=1",
            agents="none",
            max_iterations=1,
            min_confidence=0.5,
            min_confidence_auto=0.7,
            timeout_seconds=10.0,
            attempt_store=None,
            require_consensus=False,
            no_revert=False,
            require_approval=False,
            log_file="-",
            log_level="warning",
            run_id="test",
            artifacts_dir=None,
            no_diagnostics=True,
            llm_analyzer=False,
            analysis_agents="",
            analysis_require_consensus=False,
            analysis_consensus_threshold=0.7,
            arena_validate=False,
            arena_agents="",
            arena_rounds=2,
            arena_min_confidence=0.6,
            arena_require_consensus=False,
            arena_consensus_threshold=0.7,
            redteam_validate=False,
            redteam_attackers="",
            redteam_defender="",
            redteam_rounds=2,
            redteam_attacks_per_round=3,
            redteam_min_robustness=0.6,
            pattern_learning=False,
            pattern_store=None,
            generation_timeout_seconds=600.0,
            critique_timeout_seconds=300.0,
            batch=False,
            max_batch_size=5,
            elo_select=False,
            elo_fallback_agents="",
            bug_check=False,
            impact_analysis=False,
            batch_test_command=None,
        )

        mock_result = MagicMock()
        mock_result.summary.return_value = "TestFixer success: 1/1"
        mock_result.status.value = "success"

        with patch("aragora.cli.commands.testfixer.TestFixerOrchestrator") as MockOrch:
            instance = MockOrch.return_value
            instance.run_fix_loop = AsyncMock(return_value=mock_result)

            exit_code = await _run(ns)

            instance.run_fix_loop.assert_called_once()
            assert exit_code == 0
