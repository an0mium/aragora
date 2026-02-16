"""Tests for aragora.nomic.metrics_collector.

Covers MetricSnapshot, MetricsDelta, MetricsCollector, and MetricsCollectorConfig.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from aragora.nomic.metrics_collector import (
    MetricsCollector,
    MetricsCollectorConfig,
    MetricsDelta,
    MetricSnapshot,
)


# ---------------------------------------------------------------------------
# MetricSnapshot
# ---------------------------------------------------------------------------


class TestMetricSnapshot:
    """Tests for MetricSnapshot dataclass."""

    def test_defaults(self):
        snap = MetricSnapshot()
        assert snap.timestamp == 0.0
        assert snap.tests_passed == 0
        assert snap.tests_failed == 0
        assert snap.tests_skipped == 0
        assert snap.tests_errors == 0
        assert snap.test_coverage is None
        assert snap.lint_errors == 0
        assert snap.type_errors == 0
        assert snap.files_count == 0
        assert snap.total_lines == 0
        assert snap.custom == {}
        assert snap.collection_errors == []
        assert snap.collection_duration_seconds == 0.0

    def test_tests_total(self):
        snap = MetricSnapshot(tests_passed=10, tests_failed=2, tests_skipped=3, tests_errors=1)
        assert snap.tests_total == 16

    def test_tests_total_all_zeros(self):
        snap = MetricSnapshot()
        assert snap.tests_total == 0

    # -- test_pass_rate --

    def test_pass_rate_basic(self):
        snap = MetricSnapshot(tests_passed=90, tests_failed=10, tests_errors=0)
        assert snap.test_pass_rate == pytest.approx(0.9)

    def test_pass_rate_with_errors(self):
        snap = MetricSnapshot(tests_passed=80, tests_failed=10, tests_errors=10)
        # denominator = 80 + 10 + 10 = 100
        assert snap.test_pass_rate == pytest.approx(0.8)

    def test_pass_rate_excludes_skipped(self):
        """Skipped tests are NOT in the pass rate denominator."""
        snap = MetricSnapshot(tests_passed=50, tests_failed=0, tests_errors=0, tests_skipped=50)
        # denominator = 50 + 0 + 0 = 50
        assert snap.test_pass_rate == pytest.approx(1.0)

    def test_pass_rate_zero_denominator(self):
        snap = MetricSnapshot()
        assert snap.test_pass_rate == 0.0

    def test_pass_rate_all_failed(self):
        snap = MetricSnapshot(tests_passed=0, tests_failed=5, tests_errors=0)
        assert snap.test_pass_rate == 0.0

    # -- serialization --

    def test_to_dict_contains_all_fields(self):
        snap = MetricSnapshot(
            timestamp=1000.0,
            tests_passed=100,
            tests_failed=5,
            tests_skipped=3,
            tests_errors=1,
            test_coverage=0.87,
            lint_errors=12,
            type_errors=4,
            files_count=200,
            total_lines=15000,
            custom={"complexity": 3.5},
            collection_errors=["oops"],
            collection_duration_seconds=2.5,
        )
        d = snap.to_dict()
        assert d["timestamp"] == 1000.0
        assert d["tests_passed"] == 100
        assert d["tests_failed"] == 5
        assert d["tests_skipped"] == 3
        assert d["tests_errors"] == 1
        assert d["test_coverage"] == 0.87
        assert d["lint_errors"] == 12
        assert d["type_errors"] == 4
        assert d["files_count"] == 200
        assert d["total_lines"] == 15000
        assert d["custom"] == {"complexity": 3.5}
        assert d["collection_errors"] == ["oops"]
        assert d["collection_duration_seconds"] == 2.5

    def test_from_dict_roundtrip(self):
        original = MetricSnapshot(
            timestamp=42.0,
            tests_passed=10,
            tests_failed=2,
            tests_skipped=1,
            tests_errors=0,
            test_coverage=0.95,
            lint_errors=3,
            type_errors=1,
            files_count=50,
            total_lines=5000,
            custom={"metric_a": 1.0},
            collection_errors=["err1"],
            collection_duration_seconds=1.1,
        )
        restored = MetricSnapshot.from_dict(original.to_dict())

        assert restored.timestamp == original.timestamp
        assert restored.tests_passed == original.tests_passed
        assert restored.tests_failed == original.tests_failed
        assert restored.tests_skipped == original.tests_skipped
        assert restored.tests_errors == original.tests_errors
        assert restored.test_coverage == original.test_coverage
        assert restored.lint_errors == original.lint_errors
        assert restored.type_errors == original.type_errors
        assert restored.files_count == original.files_count
        assert restored.total_lines == original.total_lines
        assert restored.custom == original.custom
        assert restored.collection_errors == original.collection_errors
        assert restored.collection_duration_seconds == original.collection_duration_seconds

    def test_from_dict_empty(self):
        snap = MetricSnapshot.from_dict({})
        assert snap.timestamp == 0.0
        assert snap.tests_passed == 0
        assert snap.test_coverage is None
        assert snap.custom == {}
        assert snap.collection_errors == []

    def test_from_dict_partial(self):
        snap = MetricSnapshot.from_dict({"tests_passed": 42, "lint_errors": 7})
        assert snap.tests_passed == 42
        assert snap.lint_errors == 7
        assert snap.tests_failed == 0

    def test_to_dict_none_coverage(self):
        snap = MetricSnapshot()
        d = snap.to_dict()
        assert d["test_coverage"] is None


# ---------------------------------------------------------------------------
# MetricsDelta
# ---------------------------------------------------------------------------


class TestMetricsDelta:
    """Tests for MetricsDelta dataclass."""

    def test_defaults(self):
        b = MetricSnapshot()
        a = MetricSnapshot()
        delta = MetricsDelta(baseline=b, after=a)
        assert delta.tests_passed_delta == 0
        assert delta.tests_failed_delta == 0
        assert delta.test_pass_rate_delta == 0.0
        assert delta.test_coverage_delta is None
        assert delta.lint_errors_delta == 0
        assert delta.type_errors_delta == 0
        assert delta.custom_deltas == {}
        assert delta.improved is False
        assert delta.improvement_score == 0.0
        assert delta.summary == ""

    def test_to_dict_includes_nested_snapshots(self):
        b = MetricSnapshot(tests_passed=10)
        a = MetricSnapshot(tests_passed=15)
        delta = MetricsDelta(
            baseline=b,
            after=a,
            tests_passed_delta=5,
            improved=True,
            improvement_score=0.7,
            summary="+5 tests passing",
        )
        d = delta.to_dict()
        assert d["baseline"]["tests_passed"] == 10
        assert d["after"]["tests_passed"] == 15
        assert d["tests_passed_delta"] == 5
        assert d["improved"] is True
        assert d["improvement_score"] == 0.7
        assert d["summary"] == "+5 tests passing"

    def test_from_dict_roundtrip(self):
        b = MetricSnapshot(tests_passed=10, lint_errors=5)
        a = MetricSnapshot(tests_passed=15, lint_errors=2)
        original = MetricsDelta(
            baseline=b,
            after=a,
            tests_passed_delta=5,
            tests_failed_delta=-1,
            test_pass_rate_delta=0.05,
            test_coverage_delta=0.02,
            lint_errors_delta=-3,
            type_errors_delta=0,
            custom_deltas={"x": 1.0},
            improved=True,
            improvement_score=0.8,
            summary="good",
        )
        restored = MetricsDelta.from_dict(original.to_dict())

        assert restored.tests_passed_delta == 5
        assert restored.tests_failed_delta == -1
        assert restored.test_pass_rate_delta == pytest.approx(0.05)
        assert restored.test_coverage_delta == pytest.approx(0.02)
        assert restored.lint_errors_delta == -3
        assert restored.type_errors_delta == 0
        assert restored.custom_deltas == {"x": 1.0}
        assert restored.improved is True
        assert restored.improvement_score == pytest.approx(0.8)
        assert restored.summary == "good"
        assert restored.baseline.tests_passed == 10
        assert restored.after.tests_passed == 15

    def test_from_dict_empty(self):
        delta = MetricsDelta.from_dict({})
        assert delta.baseline.tests_passed == 0
        assert delta.after.tests_passed == 0
        assert delta.improved is False


# ---------------------------------------------------------------------------
# MetricsCollector.compare()
# ---------------------------------------------------------------------------


class TestCompare:
    """Tests for MetricsCollector.compare()."""

    def setup_method(self):
        self.collector = MetricsCollector()

    def test_more_tests_passing_is_improved(self):
        baseline = MetricSnapshot(tests_passed=90, tests_failed=10)
        after = MetricSnapshot(tests_passed=98, tests_failed=2)
        delta = self.collector.compare(baseline, after)

        assert delta.tests_passed_delta == 8
        assert delta.tests_failed_delta == -8
        assert delta.test_pass_rate_delta > 0
        assert delta.improved is True
        assert delta.improvement_score > 0.3
        assert "+8 tests passing" in delta.summary

    def test_fewer_test_failures_is_improved(self):
        baseline = MetricSnapshot(tests_passed=80, tests_failed=20)
        after = MetricSnapshot(tests_passed=90, tests_failed=10)
        delta = self.collector.compare(baseline, after)

        assert delta.tests_failed_delta == -10
        assert delta.improved is True
        assert delta.improvement_score > 0.3

    def test_fewer_lint_errors_is_improved(self):
        baseline = MetricSnapshot(tests_passed=100, tests_failed=0, lint_errors=50)
        after = MetricSnapshot(tests_passed=100, tests_failed=0, lint_errors=10)
        delta = self.collector.compare(baseline, after)

        assert delta.lint_errors_delta == -40
        assert delta.improved is True
        assert "-40 lint errors" in delta.summary

    def test_test_regression_not_improved(self):
        baseline = MetricSnapshot(tests_passed=100, tests_failed=0)
        after = MetricSnapshot(tests_passed=90, tests_failed=10)
        delta = self.collector.compare(baseline, after)

        assert delta.tests_failed_delta == 10
        assert delta.improved is False
        assert "regressions" in delta.summary

    def test_lint_regression_not_improved(self):
        baseline = MetricSnapshot(tests_passed=100, tests_failed=0, lint_errors=5)
        after = MetricSnapshot(tests_passed=100, tests_failed=0, lint_errors=15)
        delta = self.collector.compare(baseline, after)

        assert delta.lint_errors_delta == 10
        assert delta.improved is False

    def test_type_error_regression_not_improved(self):
        baseline = MetricSnapshot(tests_passed=100, tests_failed=0, type_errors=0)
        after = MetricSnapshot(tests_passed=100, tests_failed=0, type_errors=5)
        delta = self.collector.compare(baseline, after)

        assert delta.type_errors_delta == 5
        assert delta.improved is False

    def test_no_change_is_neutral(self):
        baseline = MetricSnapshot(tests_passed=100, tests_failed=0, lint_errors=5)
        after = MetricSnapshot(tests_passed=100, tests_failed=0, lint_errors=5)
        delta = self.collector.compare(baseline, after)

        assert delta.tests_passed_delta == 0
        assert delta.tests_failed_delta == 0
        assert delta.lint_errors_delta == 0
        # tests_total > 0 so pass_rate delta (0.0) contributes a 0.0 signal.
        # With signals=[0.0], score = 0.0.  0.0 is NOT > 0.3, so improved=False.
        assert delta.improvement_score == pytest.approx(0.0)
        assert delta.improved is False
        assert "no regressions detected" in delta.summary

    def test_no_change_empty_baseline_is_neutral(self):
        """With all-zero baseline, no signals are generated -> score defaults to 0.5."""
        baseline = MetricSnapshot()
        after = MetricSnapshot()
        delta = self.collector.compare(baseline, after)

        assert delta.improvement_score == pytest.approx(0.5)
        assert delta.improved is True

    def test_empty_baseline_no_signals(self):
        """All-zero baseline yields no test-based signals -> falls to no_regression default."""
        baseline = MetricSnapshot()
        after = MetricSnapshot()
        delta = self.collector.compare(baseline, after)
        # tests_total=0 so no pass-rate signal added to signals list.
        # signals is empty, no_regression is True -> score = 0.5
        assert delta.improvement_score == pytest.approx(0.5)
        assert delta.improved is True

    def test_coverage_improvement(self):
        baseline = MetricSnapshot(tests_passed=100, tests_failed=0, test_coverage=0.80)
        after = MetricSnapshot(tests_passed=100, tests_failed=0, test_coverage=0.90)
        delta = self.collector.compare(baseline, after)

        assert delta.test_coverage_delta == pytest.approx(0.10)
        assert "+10.0% coverage" in delta.summary

    def test_coverage_delta_none_when_baseline_missing(self):
        baseline = MetricSnapshot(tests_passed=100, tests_failed=0, test_coverage=None)
        after = MetricSnapshot(tests_passed=100, tests_failed=0, test_coverage=0.80)
        delta = self.collector.compare(baseline, after)

        assert delta.test_coverage_delta is None

    def test_coverage_delta_none_when_after_missing(self):
        baseline = MetricSnapshot(tests_passed=100, tests_failed=0, test_coverage=0.80)
        after = MetricSnapshot(tests_passed=100, tests_failed=0, test_coverage=None)
        delta = self.collector.compare(baseline, after)

        assert delta.test_coverage_delta is None

    def test_custom_metric_deltas(self):
        baseline = MetricSnapshot(tests_passed=10, tests_failed=0, custom={"complexity": 5.0, "debt": 10.0})
        after = MetricSnapshot(tests_passed=10, tests_failed=0, custom={"complexity": 3.0, "new_metric": 1.0})
        delta = self.collector.compare(baseline, after)

        assert delta.custom_deltas["complexity"] == pytest.approx(-2.0)
        # 'debt' present only in baseline: after defaults to 0
        assert delta.custom_deltas["debt"] == pytest.approx(-10.0)
        # 'new_metric' present only in after: baseline defaults to 0
        assert delta.custom_deltas["new_metric"] == pytest.approx(1.0)

    def test_improvement_score_capped_at_one(self):
        """Even large improvements should not produce score > 1.0."""
        baseline = MetricSnapshot(tests_passed=10, tests_failed=100, lint_errors=200)
        after = MetricSnapshot(tests_passed=110, tests_failed=0, lint_errors=0)
        delta = self.collector.compare(baseline, after)

        assert 0.0 <= delta.improvement_score <= 1.0
        assert delta.improved is True

    def test_regression_summary_details(self):
        baseline = MetricSnapshot(tests_passed=100, tests_failed=0, lint_errors=0)
        after = MetricSnapshot(tests_passed=95, tests_failed=5, lint_errors=3)
        delta = self.collector.compare(baseline, after)

        assert "regressions" in delta.summary
        assert "+5 test failures" in delta.summary
        assert "+3 lint errors" in delta.summary

    def test_mixed_improvement_and_regression_not_improved(self):
        """Even with some positive signals, regression blocks improved=True."""
        baseline = MetricSnapshot(tests_passed=80, tests_failed=5, lint_errors=10)
        after = MetricSnapshot(tests_passed=90, tests_failed=10, lint_errors=5)
        delta = self.collector.compare(baseline, after)

        # tests_failed went up -> no_regression is False -> improved is False
        assert delta.improved is False


# ---------------------------------------------------------------------------
# MetricsCollector.check_success_criteria()
# ---------------------------------------------------------------------------


class TestCheckSuccessCriteria:
    """Tests for check_success_criteria()."""

    def setup_method(self):
        self.collector = MetricsCollector()

    def test_pass_rate_target_met(self):
        snap = MetricSnapshot(tests_passed=96, tests_failed=4)
        met, unmet = self.collector.check_success_criteria(snap, {"test_pass_rate": ">0.95"})
        assert met is True
        assert unmet == []

    def test_pass_rate_target_not_met(self):
        snap = MetricSnapshot(tests_passed=80, tests_failed=20)
        met, unmet = self.collector.check_success_criteria(snap, {"test_pass_rate": ">0.95"})
        assert met is False
        assert len(unmet) == 1
        assert "test_pass_rate" in unmet[0]
        assert "does not meet target" in unmet[0]

    def test_zero_failures_target_met(self):
        snap = MetricSnapshot(tests_passed=100, tests_failed=0)
        met, unmet = self.collector.check_success_criteria(snap, {"tests_failed": "==0"})
        assert met is True

    def test_zero_failures_target_not_met(self):
        snap = MetricSnapshot(tests_passed=98, tests_failed=2)
        met, unmet = self.collector.check_success_criteria(snap, {"tests_failed": "==0"})
        assert met is False

    def test_lint_errors_le_target(self):
        snap = MetricSnapshot(lint_errors=8)
        met, unmet = self.collector.check_success_criteria(snap, {"lint_errors": "<=10"})
        assert met is True

    def test_lint_errors_le_target_not_met(self):
        snap = MetricSnapshot(lint_errors=15)
        met, unmet = self.collector.check_success_criteria(snap, {"lint_errors": "<=10"})
        assert met is False

    def test_multiple_criteria_all_met(self):
        snap = MetricSnapshot(tests_passed=100, tests_failed=0, lint_errors=0)
        criteria = {
            "test_pass_rate": ">0.99",
            "tests_failed": "==0",
            "lint_errors": "==0",
        }
        met, unmet = self.collector.check_success_criteria(snap, criteria)
        assert met is True
        assert unmet == []

    def test_multiple_criteria_some_unmet(self):
        snap = MetricSnapshot(tests_passed=90, tests_failed=10, lint_errors=5)
        criteria = {
            "test_pass_rate": ">0.95",
            "tests_failed": "==0",
            "lint_errors": "<=10",
        }
        met, unmet = self.collector.check_success_criteria(snap, criteria)
        assert met is False
        assert len(unmet) == 2  # pass_rate and tests_failed

    def test_numeric_target_uses_gte(self):
        """A bare numeric target checks actual >= target."""
        snap = MetricSnapshot(tests_passed=100, tests_failed=0)
        met, unmet = self.collector.check_success_criteria(snap, {"test_pass_rate": 0.95})
        assert met is True

    def test_numeric_target_not_met(self):
        snap = MetricSnapshot(tests_passed=80, tests_failed=20)
        met, unmet = self.collector.check_success_criteria(snap, {"test_pass_rate": 0.95})
        assert met is False

    def test_unavailable_metric(self):
        snap = MetricSnapshot()
        met, unmet = self.collector.check_success_criteria(snap, {"nonexistent_thing": ">0"})
        assert met is False
        assert "metric not available" in unmet[0]

    def test_custom_metric_in_criteria(self):
        snap = MetricSnapshot(custom={"latency": 0.5})
        met, unmet = self.collector.check_success_criteria(snap, {"latency": "<1.0"})
        assert met is True

    def test_coverage_none_is_unavailable(self):
        snap = MetricSnapshot(test_coverage=None)
        met, unmet = self.collector.check_success_criteria(snap, {"test_coverage": ">0.80"})
        assert met is False
        assert "metric not available" in unmet[0]

    def test_coverage_present_is_evaluated(self):
        snap = MetricSnapshot(test_coverage=0.85)
        met, unmet = self.collector.check_success_criteria(snap, {"test_coverage": ">0.80"})
        assert met is True

    def test_non_string_non_numeric_target_skipped(self):
        """Non-string, non-numeric targets are silently skipped (not unmet)."""
        snap = MetricSnapshot(tests_passed=10, tests_failed=0)
        met, unmet = self.collector.check_success_criteria(snap, {"tests_passed": [1, 2, 3]})
        assert met is True
        assert unmet == []


# ---------------------------------------------------------------------------
# MetricsCollector._evaluate_target()
# ---------------------------------------------------------------------------


class TestEvaluateTarget:
    """Tests for the static _evaluate_target() method."""

    @pytest.mark.parametrize(
        "actual, target, expected",
        [
            # Greater than
            (10.0, ">5", True),
            (5.0, ">5", False),
            (3.0, ">5", False),
            # Greater or equal
            (5.0, ">=5", True),
            (6.0, ">=5", True),
            (4.0, ">=5", False),
            # Less than
            (3.0, "<5", True),
            (5.0, "<5", False),
            (7.0, "<5", False),
            # Less or equal
            (5.0, "<=5", True),
            (3.0, "<=5", True),
            (6.0, "<=5", False),
            # Equal
            (0.0, "==0", True),
            (5.0, "==5", True),
            (5.1, "==5", False),
            # Not equal
            (3.0, "!=5", True),
            (5.0, "!=5", False),
            # Floating point targets
            (0.96, ">0.95", True),
            (0.94, ">0.95", False),
            (0.95, ">=0.95", True),
        ],
    )
    def test_operator_evaluation(self, actual, target, expected):
        result = MetricsCollector._evaluate_target(actual, target)
        assert result is expected, f"_evaluate_target({actual}, '{target}') should be {expected}"

    def test_invalid_target_returns_false(self):
        assert MetricsCollector._evaluate_target(5.0, "invalid") is False

    def test_empty_target_returns_false(self):
        assert MetricsCollector._evaluate_target(5.0, "") is False

    def test_unknown_operator_returns_false(self):
        assert MetricsCollector._evaluate_target(5.0, "~5") is False

    def test_target_with_spaces(self):
        """Spaces between operator and value are accepted."""
        assert MetricsCollector._evaluate_target(10.0, "> 5") is True
        assert MetricsCollector._evaluate_target(10.0, "<= 15") is True


# ---------------------------------------------------------------------------
# MetricsCollector._parse_pytest_output()
# ---------------------------------------------------------------------------


class TestParsePytestOutput:
    """Tests for _parse_pytest_output()."""

    def setup_method(self):
        self.collector = MetricsCollector()

    def _parse(self, output: str) -> MetricSnapshot:
        snap = MetricSnapshot()
        self.collector._parse_pytest_output(output, snap)
        return snap

    def test_all_passed(self):
        output = "291 passed in 4.56s"
        snap = self._parse(output)
        assert snap.tests_passed == 291
        assert snap.tests_failed == 0
        assert snap.tests_skipped == 0
        assert snap.tests_errors == 0

    def test_mixed_results(self):
        output = "250 passed, 10 failed, 5 skipped, 2 error in 12.3s"
        snap = self._parse(output)
        assert snap.tests_passed == 250
        assert snap.tests_failed == 10
        assert snap.tests_skipped == 5
        assert snap.tests_errors == 2

    def test_failures_and_errors_only(self):
        output = "5 failed, 3 error in 1.2s"
        snap = self._parse(output)
        assert snap.tests_passed == 0
        assert snap.tests_failed == 5
        assert snap.tests_errors == 3

    def test_with_warnings(self):
        output = "100 passed, 3 warnings in 2.0s"
        snap = self._parse(output)
        assert snap.tests_passed == 100
        assert snap.tests_failed == 0

    def test_coverage_output(self):
        output = (
            "Name                 Stmts   Miss  Cover\n"
            "-------------------------------------------\n"
            "aragora/__init__.py     10      2    80%\n"
            "aragora/core.py         50      5    90%\n"
            "-------------------------------------------\n"
            "TOTAL                   60      7    88%\n"
            "\n"
            "100 passed in 3.5s"
        )
        snap = self._parse(output)
        assert snap.tests_passed == 100
        assert snap.test_coverage == pytest.approx(0.88)

    def test_no_coverage_stays_none(self):
        output = "50 passed in 1.0s"
        snap = self._parse(output)
        assert snap.test_coverage is None

    def test_empty_output(self):
        snap = self._parse("")
        assert snap.tests_passed == 0
        assert snap.tests_failed == 0

    def test_realistic_multiline_output(self):
        output = (
            "tests/test_foo.py ...........\n"
            "tests/test_bar.py .....F..\n"
            "tests/test_baz.py ...s...\n"
            "\n"
            "FAILURES\n"
            "________________________________________\n"
            "test_something - AssertionError\n"
            "________________________________________\n"
            "\n"
            "short test summary info\n"
            "FAILED tests/test_bar.py::test_something\n"
            "== 25 passed, 1 failed, 1 skipped in 2.50s =="
        )
        snap = self._parse(output)
        assert snap.tests_passed == 25
        assert snap.tests_failed == 1
        assert snap.tests_skipped == 1
        assert snap.tests_errors == 0

    def test_only_errors(self):
        output = "1 error in 0.5s"
        snap = self._parse(output)
        assert snap.tests_errors == 1
        assert snap.tests_passed == 0

    def test_100_percent_coverage(self):
        output = "TOTAL                  100      0   100%\n50 passed in 1.0s"
        snap = self._parse(output)
        assert snap.test_coverage == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MetricsCollector -- collect_baseline / collect_after (async, subprocess mocked)
# ---------------------------------------------------------------------------


class TestCollectBaseline:
    """Tests for collect_baseline() and collect_after() with mocked subprocess."""

    @pytest.fixture
    def collector(self, tmp_path):
        config = MetricsCollectorConfig(working_dir=str(tmp_path))
        # Create minimal directory structure for size metrics
        aragora_dir = tmp_path / "aragora"
        aragora_dir.mkdir()
        (aragora_dir / "module.py").write_text("line1\nline2\nline3\n")
        return MetricsCollector(config)

    @pytest.mark.asyncio
    async def test_collect_baseline_parses_pytest(self, collector):
        mock_test_result = MagicMock()
        mock_test_result.stdout = "50 passed, 2 failed in 3.0s"
        mock_test_result.stderr = ""

        mock_lint_result = MagicMock()
        mock_lint_result.stdout = "file.py:1:1: E001 error\nfile.py:2:1: E002 error\n"
        mock_lint_result.stderr = ""

        with patch("subprocess.run", side_effect=[mock_test_result, mock_lint_result]):
            snap = await collector.collect_baseline("improve tests")

        assert snap.tests_passed == 50
        assert snap.tests_failed == 2
        assert snap.lint_errors == 2
        assert snap.timestamp > 0
        assert snap.collection_duration_seconds >= 0
        assert snap.collection_errors == []

    @pytest.mark.asyncio
    async def test_collect_after_same_as_baseline(self, collector):
        """collect_after uses the same underlying _collect()."""
        mock_test_result = MagicMock()
        mock_test_result.stdout = "100 passed in 2.0s"
        mock_test_result.stderr = ""

        mock_lint_result = MagicMock()
        mock_lint_result.stdout = ""
        mock_lint_result.stderr = ""

        with patch("subprocess.run", side_effect=[mock_test_result, mock_lint_result]):
            snap = await collector.collect_after("improve tests")

        assert snap.tests_passed == 100
        assert snap.lint_errors == 0

    @pytest.mark.asyncio
    async def test_subprocess_error_recorded(self, collector):
        """Subprocess failures are recorded in collection_errors, not raised."""
        with patch(
            "subprocess.run",
            side_effect=subprocess.SubprocessError("command not found"),
        ):
            snap = await collector.collect_baseline("test goal")

        assert len(snap.collection_errors) >= 1
        assert "test_collection" in snap.collection_errors[0]

    @pytest.mark.asyncio
    async def test_os_error_recorded(self, collector):
        with patch("subprocess.run", side_effect=OSError("no such file")):
            snap = await collector.collect_baseline("test goal")

        assert len(snap.collection_errors) >= 1

    @pytest.mark.asyncio
    async def test_file_scope_used(self, collector):
        mock_test_result = MagicMock()
        mock_test_result.stdout = "10 passed in 1.0s"
        mock_test_result.stderr = ""

        mock_lint_result = MagicMock()
        mock_lint_result.stdout = ""
        mock_lint_result.stderr = ""

        with patch("subprocess.run", side_effect=[mock_test_result, mock_lint_result]) as mock_run:
            await collector.collect_baseline("fix bug", file_scope=["aragora/module.py"])

        # Verify subprocess was called (tests and lint)
        assert mock_run.call_count == 2

    @pytest.mark.asyncio
    async def test_size_metrics_collected(self, collector):
        mock_test_result = MagicMock()
        mock_test_result.stdout = "5 passed in 0.5s"
        mock_test_result.stderr = ""

        mock_lint_result = MagicMock()
        mock_lint_result.stdout = ""
        mock_lint_result.stderr = ""

        with patch("subprocess.run", side_effect=[mock_test_result, mock_lint_result]):
            snap = await collector.collect_baseline("improve")

        # The fixture created aragora/module.py with 3 lines
        assert snap.files_count == 1
        assert snap.total_lines == 3


# ---------------------------------------------------------------------------
# MetricsCollectorConfig
# ---------------------------------------------------------------------------


class TestMetricsCollectorConfig:
    """Tests for MetricsCollectorConfig defaults."""

    def test_defaults(self):
        cfg = MetricsCollectorConfig()
        assert cfg.test_command == "pytest"
        assert "-x" in cfg.test_args
        assert "-q" in cfg.test_args
        assert "--tb=no" in cfg.test_args
        assert cfg.test_timeout == 300
        assert cfg.collect_coverage is False
        assert cfg.lint_command == "ruff"
        assert "check" in cfg.lint_args
        assert cfg.lint_timeout == 60
        assert cfg.working_dir is None
        assert cfg.test_scope_dirs == []

    def test_custom_config(self):
        cfg = MetricsCollectorConfig(
            test_command="python -m pytest",
            test_args=["-v"],
            test_timeout=600,
            lint_command="flake8",
            lint_args=["--max-line-length=100"],
            working_dir="/tmp/project",
            test_scope_dirs=["tests/unit"],
        )
        assert cfg.test_command == "python -m pytest"
        assert cfg.test_args == ["-v"]
        assert cfg.test_timeout == 600
        assert cfg.lint_command == "flake8"
        assert cfg.working_dir == "/tmp/project"
        assert cfg.test_scope_dirs == ["tests/unit"]


# ---------------------------------------------------------------------------
# MetricsCollector._get_metric_value()
# ---------------------------------------------------------------------------


class TestGetMetricValue:
    """Tests for _get_metric_value()."""

    def setup_method(self):
        self.collector = MetricsCollector()

    def test_standard_metrics(self):
        snap = MetricSnapshot(
            tests_passed=100,
            tests_failed=5,
            lint_errors=10,
            type_errors=2,
            files_count=50,
            total_lines=5000,
            test_coverage=0.85,
        )
        assert self.collector._get_metric_value(snap, "test_pass_rate") == pytest.approx(
            100 / 105  # 100 / (100 + 5 + 0)
        )
        assert self.collector._get_metric_value(snap, "tests_passed") == 100.0
        assert self.collector._get_metric_value(snap, "tests_failed") == 5.0
        assert self.collector._get_metric_value(snap, "test_coverage") == 0.85
        assert self.collector._get_metric_value(snap, "lint_errors") == 10.0
        assert self.collector._get_metric_value(snap, "type_errors") == 2.0
        assert self.collector._get_metric_value(snap, "files_count") == 50.0
        assert self.collector._get_metric_value(snap, "total_lines") == 5000.0

    def test_custom_metric(self):
        snap = MetricSnapshot(custom={"latency": 0.5, "throughput": 100.0})
        assert self.collector._get_metric_value(snap, "latency") == 0.5
        assert self.collector._get_metric_value(snap, "throughput") == 100.0

    def test_unknown_metric_returns_none(self):
        snap = MetricSnapshot()
        assert self.collector._get_metric_value(snap, "nonexistent") is None

    def test_coverage_none_returned(self):
        snap = MetricSnapshot(test_coverage=None)
        assert self.collector._get_metric_value(snap, "test_coverage") is None


# ---------------------------------------------------------------------------
# MetricsCollector._infer_test_path()
# ---------------------------------------------------------------------------


class TestInferTestPath:
    """Tests for _infer_test_path()."""

    def test_valid_aragora_path(self, tmp_path):
        (tmp_path / "tests" / "debate").mkdir(parents=True)
        config = MetricsCollectorConfig(working_dir=str(tmp_path))
        collector = MetricsCollector(config)

        result = collector._infer_test_path("aragora/debate/orchestrator.py")
        assert result == "tests/debate"

    def test_non_aragora_path_returns_none(self, tmp_path):
        config = MetricsCollectorConfig(working_dir=str(tmp_path))
        collector = MetricsCollector(config)
        result = collector._infer_test_path("other/file.py")
        assert result is None

    def test_no_matching_test_dir(self, tmp_path):
        config = MetricsCollectorConfig(working_dir=str(tmp_path))
        collector = MetricsCollector(config)
        result = collector._infer_test_path("aragora/nonexistent/module.py")
        assert result is None

    def test_short_path(self, tmp_path):
        config = MetricsCollectorConfig(working_dir=str(tmp_path))
        collector = MetricsCollector(config)
        # Only one segment, not enough to infer a test dir
        result = collector._infer_test_path("aragora/")
        assert result is None


# ---------------------------------------------------------------------------
# Integration-style: collect + compare full cycle (mocked subprocess)
# ---------------------------------------------------------------------------


class TestFullCycle:
    """End-to-end test of collect_baseline -> collect_after -> compare."""

    @pytest.mark.asyncio
    async def test_improvement_cycle(self, tmp_path):
        aragora_dir = tmp_path / "aragora"
        aragora_dir.mkdir()
        (aragora_dir / "module.py").write_text("x = 1\n")

        config = MetricsCollectorConfig(working_dir=str(tmp_path))
        collector = MetricsCollector(config)

        # Baseline: 90 passed, 10 failed, 5 lint errors
        baseline_test = MagicMock()
        baseline_test.stdout = "90 passed, 10 failed in 5.0s"
        baseline_test.stderr = ""
        baseline_lint = MagicMock()
        baseline_lint.stdout = "e1\ne2\ne3\ne4\ne5"
        baseline_lint.stderr = ""

        with patch("subprocess.run", side_effect=[baseline_test, baseline_lint]):
            baseline = await collector.collect_baseline("fix bugs")

        assert baseline.tests_passed == 90
        assert baseline.tests_failed == 10
        assert baseline.lint_errors == 5

        # After: 98 passed, 2 failed, 1 lint error
        after_test = MagicMock()
        after_test.stdout = "98 passed, 2 failed in 4.0s"
        after_test.stderr = ""
        after_lint = MagicMock()
        after_lint.stdout = "e1"
        after_lint.stderr = ""

        with patch("subprocess.run", side_effect=[after_test, after_lint]):
            after = await collector.collect_after("fix bugs")

        assert after.tests_passed == 98
        assert after.tests_failed == 2
        assert after.lint_errors == 1

        # Compare
        delta = collector.compare(baseline, after)

        assert delta.tests_passed_delta == 8
        assert delta.tests_failed_delta == -8
        assert delta.lint_errors_delta == -4
        assert delta.improved is True
        assert delta.improvement_score > 0.3
        assert "+8 tests passing" in delta.summary

        # Check success criteria
        met, unmet = collector.check_success_criteria(after, {"test_pass_rate": ">0.95"})
        # 98 / (98+2) = 0.98 > 0.95
        assert met is True

    @pytest.mark.asyncio
    async def test_regression_cycle(self, tmp_path):
        aragora_dir = tmp_path / "aragora"
        aragora_dir.mkdir()
        (aragora_dir / "module.py").write_text("x = 1\n")

        config = MetricsCollectorConfig(working_dir=str(tmp_path))
        collector = MetricsCollector(config)

        baseline_test = MagicMock()
        baseline_test.stdout = "100 passed in 5.0s"
        baseline_test.stderr = ""
        baseline_lint = MagicMock()
        baseline_lint.stdout = ""
        baseline_lint.stderr = ""

        with patch("subprocess.run", side_effect=[baseline_test, baseline_lint]):
            baseline = await collector.collect_baseline("refactor")

        after_test = MagicMock()
        after_test.stdout = "90 passed, 10 failed in 5.0s"
        after_test.stderr = ""
        after_lint = MagicMock()
        after_lint.stdout = "err1\nerr2\nerr3"
        after_lint.stderr = ""

        with patch("subprocess.run", side_effect=[after_test, after_lint]):
            after = await collector.collect_after("refactor")

        delta = collector.compare(baseline, after)
        assert delta.improved is False
        assert delta.tests_failed_delta == 10
        assert delta.lint_errors_delta == 3

        met, unmet = collector.check_success_criteria(after, {"tests_failed": "==0"})
        assert met is False
