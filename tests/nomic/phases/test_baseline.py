"""
Tests for BaselineCollector and BaselineSnapshot (Gap 4).

Covers:
- BaselineSnapshot creation and field validation
- compare() for improvement, regression, and no-change scenarios
- _compute_score() range invariants
- BaselineCollector subprocess mocking (success + failure)
- VerifyPhase integration with baseline metrics delta
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.phases.baseline import BaselineCollector, BaselineSnapshot


# ============================================================================
# BaselineSnapshot Tests
# ============================================================================


class TestBaselineSnapshotCreation:
    """Tests for BaselineSnapshot dataclass fields."""

    def test_baseline_snapshot_creation(self):
        """Create a snapshot and verify all fields are stored correctly."""
        ts = time.time()
        snap = BaselineSnapshot(
            tests_passed=100,
            tests_failed=5,
            lint_errors=12,
            test_pass_rate=0.952,
            timestamp=ts,
        )

        assert snap.tests_passed == 100
        assert snap.tests_failed == 5
        assert snap.lint_errors == 12
        assert snap.test_pass_rate == pytest.approx(0.952)
        assert snap.timestamp == ts


class TestBaselineCompare:
    """Tests for BaselineSnapshot.compare() method."""

    def test_compare_improvement(self):
        """After has more passed, fewer lint errors -> improved=True, score>0.5."""
        before = BaselineSnapshot(
            tests_passed=80,
            tests_failed=20,
            lint_errors=30,
            test_pass_rate=0.80,
            timestamp=1000.0,
        )
        after = BaselineSnapshot(
            tests_passed=95,
            tests_failed=5,
            lint_errors=10,
            test_pass_rate=0.95,
            timestamp=2000.0,
        )

        delta = before.compare(after)

        assert delta["tests_passed_delta"] == 15
        assert delta["tests_failed_delta"] == -15
        assert delta["lint_errors_delta"] == -20
        assert delta["test_pass_rate_delta"] == pytest.approx(0.15)
        assert delta["improved"] is True
        assert delta["improvement_score"] > 0.5

    def test_compare_regression(self):
        """After has fewer passed, more lint errors -> improved=False, score<0.5."""
        before = BaselineSnapshot(
            tests_passed=95,
            tests_failed=5,
            lint_errors=5,
            test_pass_rate=0.95,
            timestamp=1000.0,
        )
        after = BaselineSnapshot(
            tests_passed=70,
            tests_failed=30,
            lint_errors=25,
            test_pass_rate=0.70,
            timestamp=2000.0,
        )

        delta = before.compare(after)

        assert delta["tests_passed_delta"] == -25
        assert delta["tests_failed_delta"] == 25
        assert delta["lint_errors_delta"] == 20
        assert delta["test_pass_rate_delta"] == pytest.approx(-0.25)
        assert delta["improved"] is False
        assert delta["improvement_score"] < 0.5

    def test_compare_no_change(self):
        """Same values -> improved=True (>= comparison), score approx 0.5."""
        snap = BaselineSnapshot(
            tests_passed=50,
            tests_failed=10,
            lint_errors=8,
            test_pass_rate=50 / 60,
            timestamp=1000.0,
        )
        same = BaselineSnapshot(
            tests_passed=50,
            tests_failed=10,
            lint_errors=8,
            test_pass_rate=50 / 60,
            timestamp=2000.0,
        )

        delta = snap.compare(same)

        assert delta["tests_passed_delta"] == 0
        assert delta["tests_failed_delta"] == 0
        assert delta["lint_errors_delta"] == 0
        assert delta["test_pass_rate_delta"] == pytest.approx(0.0)
        # >= for pass rate and <= for lint_errors means no-change is "improved"
        assert delta["improved"] is True
        assert delta["improvement_score"] == pytest.approx(0.5, abs=0.01)


# ============================================================================
# Improvement Score Tests
# ============================================================================


class TestImprovementScore:
    """Tests for _compute_score range invariants."""

    @pytest.mark.parametrize(
        "before_rate,after_rate,before_lint,after_lint",
        [
            (0.0, 1.0, 100, 0),  # Maximum improvement
            (1.0, 0.0, 0, 100),  # Maximum regression
            (0.5, 0.5, 10, 10),  # No change
            (0.0, 0.0, 0, 0),  # Zero baseline, zero after
            (1.0, 1.0, 0, 0),  # Perfect baseline, perfect after
            (0.3, 0.7, 50, 20),  # Moderate improvement
            (0.9, 0.1, 5, 50),  # Severe regression
        ],
    )
    def test_improvement_score_range(self, before_rate, after_rate, before_lint, after_lint):
        """Score is always in [0.0, 1.0] regardless of input values."""
        before = BaselineSnapshot(
            tests_passed=int(before_rate * 100),
            tests_failed=int((1 - before_rate) * 100),
            lint_errors=before_lint,
            test_pass_rate=before_rate,
            timestamp=1000.0,
        )
        after = BaselineSnapshot(
            tests_passed=int(after_rate * 100),
            tests_failed=int((1 - after_rate) * 100),
            lint_errors=after_lint,
            test_pass_rate=after_rate,
            timestamp=2000.0,
        )

        score = before._compute_score(after)

        assert 0.0 <= score <= 1.0

    def test_improvement_score_positive(self):
        """Improved metrics -> score > 0.5."""
        before = BaselineSnapshot(
            tests_passed=60,
            tests_failed=40,
            lint_errors=20,
            test_pass_rate=0.60,
            timestamp=1000.0,
        )
        after = BaselineSnapshot(
            tests_passed=90,
            tests_failed=10,
            lint_errors=5,
            test_pass_rate=0.90,
            timestamp=2000.0,
        )

        score = before._compute_score(after)

        assert score > 0.5

    def test_improvement_score_regression(self):
        """Regressed metrics -> score < 0.5."""
        before = BaselineSnapshot(
            tests_passed=90,
            tests_failed=10,
            lint_errors=5,
            test_pass_rate=0.90,
            timestamp=1000.0,
        )
        after = BaselineSnapshot(
            tests_passed=60,
            tests_failed=40,
            lint_errors=20,
            test_pass_rate=0.60,
            timestamp=2000.0,
        )

        score = before._compute_score(after)

        assert score < 0.5


# ============================================================================
# BaselineCollector Tests
# ============================================================================


class TestBaselineCollector:
    """Tests for BaselineCollector subprocess invocation."""

    @pytest.mark.asyncio
    async def test_collector_runs_tests(self, mock_aragora_path):
        """Mock subprocess to verify pytest and ruff are invoked."""
        collector = BaselineCollector(mock_aragora_path)

        exec_calls = []

        async def fake_exec(*args, **kwargs):
            exec_calls.append(args)
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b"10 passed, 2 failed", b""))
            proc.returncode = 0
            return proc

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            snapshot = await collector.collect()

        # Two subprocess calls: one for pytest, one for ruff
        assert len(exec_calls) == 2

        # First call should be pytest
        pytest_args = exec_calls[0]
        assert "pytest" in str(pytest_args)

        # Second call should be ruff
        ruff_args = exec_calls[1]
        assert "ruff" in str(ruff_args)

        # Snapshot should have parsed results
        assert snapshot.tests_passed == 10
        assert snapshot.tests_failed == 2
        assert snapshot.test_pass_rate == pytest.approx(10 / 12)
        assert isinstance(snapshot.timestamp, float)

    @pytest.mark.asyncio
    async def test_collector_graceful_on_failure(self, mock_aragora_path):
        """Mock subprocess to fail; verify defaults (0, 0) and 0 lint errors returned."""
        collector = BaselineCollector(mock_aragora_path)

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=OSError("command not found"),
        ):
            snapshot = await collector.collect()

        # On failure, _collect_test_metrics returns (0, 0)
        assert snapshot.tests_passed == 0
        assert snapshot.tests_failed == 0
        # On failure, _collect_lint_metrics returns 0
        assert snapshot.lint_errors == 0
        # With total=0, rate defaults to 1.0
        assert snapshot.test_pass_rate == pytest.approx(1.0)


# ============================================================================
# VerifyPhase Baseline Integration Test
# ============================================================================


class TestVerifyPhaseBaselineIntegration:
    """Test that VerifyPhase uses baseline to produce metrics_delta."""

    @pytest.mark.asyncio
    async def test_verify_includes_metrics_delta(self, mock_aragora_path):
        """Create VerifyPhase with baseline, verify metrics_delta in result."""
        from aragora.nomic.phases.verify import VerifyPhase

        baseline = BaselineSnapshot(
            tests_passed=80,
            tests_failed=20,
            lint_errors=15,
            test_pass_rate=0.80,
            timestamp=1000.0,
        )

        phase = VerifyPhase(
            aragora_path=mock_aragora_path,
            log_fn=MagicMock(side_effect=lambda *a, **kw: None),
            stream_emit_fn=MagicMock(),
            save_state_fn=MagicMock(),
            baseline=baseline,
        )

        # Mock all verification checks to pass
        with (
            patch.object(phase, "_check_syntax", new_callable=AsyncMock) as mock_syntax,
            patch.object(phase, "_check_imports", new_callable=AsyncMock) as mock_imports,
            patch.object(phase, "_run_tests", new_callable=AsyncMock) as mock_tests,
        ):
            mock_syntax.return_value = {"check": "syntax", "passed": True}
            mock_imports.return_value = {"check": "import", "passed": True}
            mock_tests.return_value = {
                "check": "tests",
                "passed": True,
                "output": "95 passed, 5 failed",
            }

            # Mock the post-implementation BaselineCollector.collect()
            # The import is local inside execute(), so patch at the source module
            post_snapshot = BaselineSnapshot(
                tests_passed=95,
                tests_failed=5,
                lint_errors=5,
                test_pass_rate=0.95,
                timestamp=2000.0,
            )
            with patch("aragora.nomic.phases.baseline.BaselineCollector") as mock_collector_cls:
                mock_collector_instance = MagicMock()
                mock_collector_instance.collect = AsyncMock(return_value=post_snapshot)
                mock_collector_cls.return_value = mock_collector_instance

                result = await phase.execute()

        # Result should include metrics delta
        assert result["success"] is True
        assert result["metrics_delta"] is not None
        assert result["metrics_delta"]["tests_passed_delta"] == 15
        assert result["metrics_delta"]["lint_errors_delta"] == -10
        assert result["metrics_delta"]["improved"] is True
        assert result["improvement_score"] is not None
        assert result["improvement_score"] > 0.5

        # Also in result data
        assert "metrics_delta" in result["data"]
        assert "improvement_score" in result["data"]
