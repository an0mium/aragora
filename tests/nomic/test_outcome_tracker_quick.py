"""Tests for quick regression checking in NomicOutcomeTracker.

Tests the ``quick_regression_check`` static method and the ``RegressionResult``
dataclass added for oracle-driven scoped test execution.
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch, MagicMock

import pytest

from aragora.nomic.outcome_tracker import NomicOutcomeTracker, RegressionResult


# =========================================================================
# RegressionResult dataclass tests
# =========================================================================


class TestRegressionResult:
    """Tests for the RegressionResult dataclass."""

    def test_defaults(self) -> None:
        result = RegressionResult()
        assert result.passed == 0
        assert result.failed == 0
        assert result.duration_seconds == 0.0
        assert result.is_regression is False

    def test_regression_detected(self) -> None:
        result = RegressionResult(passed=10, failed=2, duration_seconds=1.5, is_regression=True)
        assert result.is_regression is True
        assert result.failed == 2

    def test_no_regression(self) -> None:
        result = RegressionResult(passed=15, failed=0, duration_seconds=0.8, is_regression=False)
        assert result.is_regression is False


# =========================================================================
# quick_regression_check tests
# =========================================================================


class TestQuickRegressionCheck:
    """Tests for NomicOutcomeTracker.quick_regression_check."""

    @patch("aragora.nomic.outcome_tracker.subprocess.run")
    def test_all_tests_pass(self, mock_run: MagicMock) -> None:
        """When all tests pass, is_regression is False."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="5 passed in 0.42s\n",
            stderr="",
        )
        result = NomicOutcomeTracker.quick_regression_check("/tmp/worktree")
        assert result.passed == 5
        assert result.failed == 0
        assert result.is_regression is False

    @patch("aragora.nomic.outcome_tracker.subprocess.run")
    def test_failures_detected(self, mock_run: MagicMock) -> None:
        """When tests fail, is_regression is True."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="3 passed, 2 failed in 1.23s\n",
            stderr="",
        )
        result = NomicOutcomeTracker.quick_regression_check("/tmp/worktree")
        assert result.passed == 3
        assert result.failed == 2
        assert result.is_regression is True

    @patch("aragora.nomic.outcome_tracker.subprocess.run")
    def test_errors_counted_as_failures(self, mock_run: MagicMock) -> None:
        """Pytest errors (collection errors) are added to the failure count."""
        mock_run.return_value = MagicMock(
            returncode=2,
            stdout="2 passed, 1 failed, 1 error in 0.89s\n",
            stderr="",
        )
        result = NomicOutcomeTracker.quick_regression_check("/tmp/worktree")
        assert result.passed == 2
        assert result.failed == 2  # 1 failed + 1 error
        assert result.is_regression is True

    @patch("aragora.nomic.outcome_tracker.subprocess.run")
    def test_pattern_filter_passed(self, mock_run: MagicMock) -> None:
        """Test pattern is passed to pytest as -k argument."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="2 passed in 0.1s\n",
            stderr="",
        )
        NomicOutcomeTracker.quick_regression_check("/tmp/worktree", test_pattern="test_decomposer")
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "-k" in cmd
        assert "test_decomposer" in cmd

    @patch("aragora.nomic.outcome_tracker.subprocess.run")
    def test_no_pattern_no_k_flag(self, mock_run: MagicMock) -> None:
        """Without a test pattern, -k flag is not added."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="10 passed in 2.1s\n",
            stderr="",
        )
        NomicOutcomeTracker.quick_regression_check("/tmp/worktree")
        call_args = mock_run.call_args
        cmd = call_args[0][0]
        assert "-k" not in cmd

    @patch("aragora.nomic.outcome_tracker.subprocess.run")
    def test_timeout_graceful_degradation(self, mock_run: MagicMock) -> None:
        """Subprocess timeout returns graceful result, not exception."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="pytest", timeout=30)
        result = NomicOutcomeTracker.quick_regression_check("/tmp/worktree")
        assert result.passed == 0
        assert result.failed == 0
        assert result.is_regression is False

    @patch("aragora.nomic.outcome_tracker.subprocess.run")
    def test_os_error_graceful_degradation(self, mock_run: MagicMock) -> None:
        """OSError (e.g. pytest not installed) returns graceful result."""
        mock_run.side_effect = OSError("No such file or directory")
        result = NomicOutcomeTracker.quick_regression_check("/tmp/worktree")
        assert result.passed == 0
        assert result.failed == 0
        assert result.is_regression is False
        assert result.duration_seconds == 0.0

    @patch("aragora.nomic.outcome_tracker.subprocess.run")
    def test_cwd_set_to_worktree(self, mock_run: MagicMock) -> None:
        """The subprocess runs in the specified worktree directory."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="1 passed\n",
            stderr="",
        )
        NomicOutcomeTracker.quick_regression_check("/my/worktree/path")
        call_args = mock_run.call_args
        assert call_args.kwargs.get("cwd") == "/my/worktree/path"
