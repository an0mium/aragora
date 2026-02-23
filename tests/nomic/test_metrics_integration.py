"""Tests for metrics integration in MetaPlanner and SelfImprovePipeline.

Covers:
- MetaPlanner._enrich_context_with_metrics() behavior
- Debate topic formatting with codebase metrics
- Pipeline baseline/after capture and delta computation
- Persistence of metrics_delta into KM cycle outcomes
- Graceful degradation on errors, import failures, and timeouts
"""

from __future__ import annotations

import sys
import time
from types import ModuleType
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.meta_planner import (
    MetaPlanner,
    MetaPlannerConfig,
    PlanningContext,
    Track,
)
from aragora.nomic.self_improve import (
    SelfImproveConfig,
    SelfImprovePipeline,
    SelfImproveResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_collector(
    *,
    files_count: int = 100,
    total_lines: int = 50000,
    lint_errors: int = 0,
    tests_failed: int = 0,
    tests_passed: int = 500,
    test_coverage: float | None = None,
) -> tuple[MagicMock, MagicMock]:
    """Return (mock_module, mock_snapshot) for metrics_collector injection.

    The mock module exposes MetricsCollector, MetricsCollectorConfig, and
    MetricSnapshot -- the three names imported inside
    ``_enrich_context_with_metrics``.
    """
    snapshot = MagicMock()
    snapshot.files_count = files_count
    snapshot.total_lines = total_lines
    snapshot.lint_errors = lint_errors
    snapshot.tests_failed = tests_failed
    snapshot.tests_passed = tests_passed
    snapshot.test_coverage = test_coverage
    snapshot.to_dict.return_value = {
        "timestamp": time.time(),
        "files_count": files_count,
        "total_lines": total_lines,
        "lint_errors": lint_errors,
        "tests_failed": tests_failed,
        "tests_passed": tests_passed,
        "tests_errors": 0,
        "tests_skipped": 0,
        "test_coverage": test_coverage,
        "custom": {},
        "collection_errors": [],
        "collection_duration_seconds": 0.1,
    }

    collector_instance = MagicMock()
    collector_instance._collect_size_metrics = MagicMock()
    collector_instance._collect_lint_metrics = MagicMock()

    mock_module = ModuleType("aragora.nomic.metrics_collector")
    mock_module.MetricsCollector = MagicMock(return_value=collector_instance)  # type: ignore[attr-defined]
    mock_module.MetricsCollectorConfig = MagicMock()  # type: ignore[attr-defined]
    mock_module.MetricSnapshot = MagicMock(return_value=snapshot)  # type: ignore[attr-defined]

    return mock_module, snapshot


def _inject_metrics_module(mock_module: ModuleType) -> dict[str, Any]:
    """Build a sys.modules patch dict that injects the mock metrics_collector."""
    return {"aragora.nomic.metrics_collector": mock_module}


# =========================================================================
# 1. MetaPlanner metrics enrichment (12 tests)
# =========================================================================


class TestEnrichContextWithMetrics:
    """Tests for MetaPlanner._enrich_context_with_metrics()."""

    def test_populates_metric_snapshot(self):
        """metric_snapshot should be populated when collector succeeds."""
        planner = MetaPlanner(MetaPlannerConfig(enable_metrics_collection=True))
        context = PlanningContext()

        mock_module, _snap = _make_mock_collector(files_count=200, total_lines=80000)
        with patch.dict("sys.modules", _inject_metrics_module(mock_module)):
            enriched = planner._enrich_context_with_metrics(context)

        assert enriched.metric_snapshot != {}
        assert enriched.metric_snapshot["files_count"] == 200
        assert enriched.metric_snapshot["total_lines"] == 80000

    def test_returns_context_on_import_error(self):
        """Should return context unchanged when metrics_collector is unavailable."""
        planner = MetaPlanner()
        context = PlanningContext(recent_issues=["pre-existing issue"])

        with patch.dict("sys.modules", {"aragora.nomic.metrics_collector": None}):
            enriched = planner._enrich_context_with_metrics(context)

        assert enriched is context
        assert enriched.metric_snapshot == {}
        assert "pre-existing issue" in enriched.recent_issues

    def test_clean_codebase_no_recent_issues(self):
        """No lint errors or test failures => no entries added to recent_issues."""
        planner = MetaPlanner()
        context = PlanningContext()

        mock_module, _ = _make_mock_collector(lint_errors=0, tests_failed=0)
        with patch.dict("sys.modules", _inject_metrics_module(mock_module)):
            enriched = planner._enrich_context_with_metrics(context)

        metric_issues = [i for i in enriched.recent_issues if "[metrics]" in i]
        assert len(metric_issues) == 0

    def test_lint_errors_produce_recent_issue(self):
        """Lint errors should inject a [metrics] entry into recent_issues."""
        planner = MetaPlanner()
        context = PlanningContext()

        mock_module, _ = _make_mock_collector(lint_errors=42)
        with patch.dict("sys.modules", _inject_metrics_module(mock_module)):
            enriched = planner._enrich_context_with_metrics(context)

        lint_issues = [i for i in enriched.recent_issues if "lint" in i.lower()]
        assert len(lint_issues) == 1
        assert "42" in lint_issues[0]

    def test_test_failures_produce_recent_issue(self):
        """Test failures should inject a [metrics] entry into recent_issues."""
        planner = MetaPlanner()
        context = PlanningContext()

        mock_module, _ = _make_mock_collector(tests_failed=7)
        with patch.dict("sys.modules", _inject_metrics_module(mock_module)):
            enriched = planner._enrich_context_with_metrics(context)

        fail_issues = [i for i in enriched.recent_issues if "test failure" in i.lower()]
        assert len(fail_issues) == 1
        assert "7" in fail_issues[0]

    def test_both_lint_and_test_failures(self):
        """Both lint errors and test failures produce separate entries."""
        planner = MetaPlanner()
        context = PlanningContext()

        mock_module, _ = _make_mock_collector(lint_errors=10, tests_failed=3)
        with patch.dict("sys.modules", _inject_metrics_module(mock_module)):
            enriched = planner._enrich_context_with_metrics(context)

        metric_issues = [i for i in enriched.recent_issues if "[metrics]" in i]
        assert len(metric_issues) == 2

    def test_snapshot_contains_expected_keys(self):
        """metric_snapshot dict should contain all standard keys."""
        planner = MetaPlanner()
        context = PlanningContext()

        mock_module, _ = _make_mock_collector(
            files_count=150,
            total_lines=60000,
            lint_errors=5,
            tests_passed=400,
            tests_failed=2,
            test_coverage=0.87,
        )
        with patch.dict("sys.modules", _inject_metrics_module(mock_module)):
            enriched = planner._enrich_context_with_metrics(context)

        snap = enriched.metric_snapshot
        expected_keys = {
            "files_count",
            "total_lines",
            "lint_errors",
            "tests_passed",
            "tests_failed",
            "tests_errors",
            "tests_skipped",
            "test_coverage",
            "timestamp",
        }
        assert expected_keys.issubset(set(snap.keys()))

    def test_size_collection_oserror_handled(self):
        """OSError in _collect_size_metrics should not crash enrichment."""
        planner = MetaPlanner()
        context = PlanningContext()

        mock_module, snap = _make_mock_collector()
        collector = mock_module.MetricsCollector.return_value
        collector._collect_size_metrics.side_effect = OSError("Permission denied")

        with patch.dict("sys.modules", _inject_metrics_module(mock_module)):
            enriched = planner._enrich_context_with_metrics(context)

        # Should still have a snapshot (lint may have succeeded)
        assert enriched.metric_snapshot is not None

    def test_lint_collection_exception_handled(self):
        """Exception in _collect_lint_metrics should not crash enrichment."""
        planner = MetaPlanner()
        context = PlanningContext()

        mock_module, _ = _make_mock_collector()
        collector = mock_module.MetricsCollector.return_value
        collector._collect_lint_metrics.side_effect = ValueError("ruff not found")

        with patch.dict("sys.modules", _inject_metrics_module(mock_module)):
            enriched = planner._enrich_context_with_metrics(context)

        assert enriched.metric_snapshot is not None

    def test_runtime_error_handled_gracefully(self):
        """RuntimeError during enrichment returns context unchanged."""
        planner = MetaPlanner()
        context = PlanningContext()

        mock_module = ModuleType("aragora.nomic.metrics_collector")
        mock_module.MetricsCollector = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore[attr-defined]
        mock_module.MetricsCollectorConfig = MagicMock()  # type: ignore[attr-defined]

        with patch.dict("sys.modules", _inject_metrics_module(mock_module)):
            enriched = planner._enrich_context_with_metrics(context)

        assert enriched is context

    def test_preserves_existing_recent_issues(self):
        """Enrichment should append to existing recent_issues, not replace."""
        planner = MetaPlanner()
        context = PlanningContext(recent_issues=["existing bug report"])

        mock_module, _ = _make_mock_collector(lint_errors=3)
        with patch.dict("sys.modules", _inject_metrics_module(mock_module)):
            enriched = planner._enrich_context_with_metrics(context)

        assert "existing bug report" in enriched.recent_issues
        assert any("lint" in i.lower() for i in enriched.recent_issues)
        assert len(enriched.recent_issues) >= 2

    def test_metric_snapshot_default_is_empty_dict(self):
        """PlanningContext.metric_snapshot defaults to empty dict."""
        context = PlanningContext()
        assert context.metric_snapshot == {}


# =========================================================================
# 2. Debate topic formatting (8 tests)
# =========================================================================


class TestDebateTopicMetrics:
    """Tests for codebase metrics section in _build_debate_topic."""

    def _build_topic(self, metric_snapshot: dict[str, Any] | None = None) -> str:
        planner = MetaPlanner()
        context = PlanningContext()
        if metric_snapshot is not None:
            context.metric_snapshot = metric_snapshot
        return planner._build_debate_topic(
            objective="Test objective",
            tracks=[Track.QA, Track.DEVELOPER],
            constraints=[],
            context=context,
        )

    def test_metrics_section_present_when_data(self):
        """CODEBASE METRICS section should appear when snapshot has data."""
        topic = self._build_topic({"files_count": 100, "total_lines": 50000})
        assert "CODEBASE METRICS" in topic

    def test_metrics_section_absent_when_empty(self):
        """CODEBASE METRICS section should be absent when snapshot is empty."""
        topic = self._build_topic({})
        assert "CODEBASE METRICS" not in topic

    def test_metrics_section_absent_when_none(self):
        """CODEBASE METRICS section should be absent when snapshot is default (empty dict)."""
        topic = self._build_topic(None)
        assert "CODEBASE METRICS" not in topic

    def test_lines_formatted_with_commas(self):
        """Large line counts should use comma formatting."""
        topic = self._build_topic({"total_lines": 250000})
        assert "250,000" in topic

    def test_pass_rate_formatted_as_percentage(self):
        """Test pass rate should appear as a percentage."""
        topic = self._build_topic(
            {
                "tests_passed": 900,
                "tests_failed": 100,
                "tests_errors": 0,
            }
        )
        assert "90%" in topic

    def test_files_count_in_topic(self):
        """Python files count should appear in the topic."""
        topic = self._build_topic({"files_count": 3000})
        assert "3000" in topic
        assert "Python files" in topic

    def test_lint_errors_in_topic(self):
        """Lint error count should appear in the topic."""
        topic = self._build_topic({"lint_errors": 42})
        assert "42" in topic
        assert "Lint errors" in topic

    def test_coverage_in_topic(self):
        """Test coverage should appear as a percentage."""
        topic = self._build_topic({"test_coverage": 0.87})
        assert "87%" in topic
        assert "coverage" in topic.lower()


# =========================================================================
# 3. Pipeline baseline/after (15 tests)
# =========================================================================


class TestPipelineBaselineAfter:
    """Tests for _capture_baseline, _capture_after, and _compare_metrics."""

    @pytest.mark.asyncio
    async def test_capture_baseline_returns_combined_dict(self):
        """_capture_baseline returns dict with 'debate' and 'codebase' keys."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(enable_codebase_metrics=True, capture_metrics=True)
        )

        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {"files_count": 100}

        mock_collector = MagicMock()
        mock_collector.collect_baseline = AsyncMock(return_value=mock_snapshot)

        mock_module = ModuleType("aragora.nomic.metrics_collector")
        mock_module.MetricsCollector = MagicMock(return_value=mock_collector)  # type: ignore[attr-defined]
        mock_module.MetricsCollectorConfig = MagicMock()  # type: ignore[attr-defined]

        with (
            patch.dict("sys.modules", {"aragora.nomic.metrics_collector": mock_module}),
            patch(
                "aragora.nomic.outcome_tracker.NomicOutcomeTracker",
                side_effect=ImportError("skip"),
            ),
        ):
            result = await pipeline._capture_baseline()

        assert isinstance(result, dict)
        assert "debate" in result
        assert "codebase" in result
        assert result["codebase"] == {"files_count": 100}

    @pytest.mark.asyncio
    async def test_capture_baseline_stashes_collector(self):
        """_capture_baseline stores collector on self._metrics_collector."""
        pipeline = SelfImprovePipeline(SelfImproveConfig(enable_codebase_metrics=True))

        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {"files_count": 50}

        mock_collector = MagicMock()
        mock_collector.collect_baseline = AsyncMock(return_value=mock_snapshot)

        mock_module = ModuleType("aragora.nomic.metrics_collector")
        mock_module.MetricsCollector = MagicMock(return_value=mock_collector)  # type: ignore[attr-defined]
        mock_module.MetricsCollectorConfig = MagicMock()  # type: ignore[attr-defined]

        with (
            patch.dict("sys.modules", {"aragora.nomic.metrics_collector": mock_module}),
            patch(
                "aragora.nomic.outcome_tracker.NomicOutcomeTracker",
                side_effect=ImportError("skip"),
            ),
        ):
            await pipeline._capture_baseline()

        assert hasattr(pipeline, "_metrics_collector")
        assert pipeline._metrics_collector is mock_collector

    @pytest.mark.asyncio
    async def test_capture_after_reuses_collector(self):
        """_capture_after should reuse the collector stashed by _capture_baseline."""
        pipeline = SelfImprovePipeline(SelfImproveConfig(enable_codebase_metrics=True))

        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {"files_count": 50}

        stashed_collector = MagicMock()
        stashed_collector.collect_after = AsyncMock(return_value=mock_snapshot)
        pipeline._metrics_collector = stashed_collector

        with patch(
            "aragora.nomic.outcome_tracker.NomicOutcomeTracker",
            side_effect=ImportError("skip"),
        ):
            result = await pipeline._capture_after()

        stashed_collector.collect_after.assert_called_once_with("self-improve")
        assert result is not None
        assert result["codebase"] == {"files_count": 50}

    @pytest.mark.asyncio
    async def test_capture_after_creates_collector_if_none(self):
        """_capture_after creates a new collector if none was stashed."""
        pipeline = SelfImprovePipeline(SelfImproveConfig(enable_codebase_metrics=True))
        # No _metrics_collector stashed

        mock_snapshot = MagicMock()
        mock_snapshot.to_dict.return_value = {"files_count": 75}

        mock_collector = MagicMock()
        mock_collector.collect_after = AsyncMock(return_value=mock_snapshot)

        mock_module = ModuleType("aragora.nomic.metrics_collector")
        mock_module.MetricsCollector = MagicMock(return_value=mock_collector)  # type: ignore[attr-defined]
        mock_module.MetricsCollectorConfig = MagicMock()  # type: ignore[attr-defined]

        with (
            patch.dict("sys.modules", {"aragora.nomic.metrics_collector": mock_module}),
            patch(
                "aragora.nomic.outcome_tracker.NomicOutcomeTracker",
                side_effect=ImportError("skip"),
            ),
        ):
            result = await pipeline._capture_after()

        assert result is not None
        assert result["codebase"] == {"files_count": 75}

    @pytest.mark.asyncio
    async def test_capture_baseline_returns_none_when_both_fail(self):
        """_capture_baseline returns None when debate and codebase both fail."""
        pipeline = SelfImprovePipeline(SelfImproveConfig(enable_codebase_metrics=True))

        with (
            patch.dict(
                "sys.modules",
                {"aragora.nomic.metrics_collector": None},
            ),
            patch(
                "aragora.nomic.outcome_tracker.NomicOutcomeTracker",
                side_effect=ImportError("skip"),
            ),
        ):
            result = await pipeline._capture_baseline()

        assert result is None

    def test_compare_metrics_produces_improvement_score(self):
        """_compare_metrics should produce improvement_score in result."""
        from aragora.nomic.metrics_collector import MetricSnapshot

        pipeline = SelfImprovePipeline()

        base_snap = MetricSnapshot(
            timestamp=time.time(),
            files_count=100,
            total_lines=50000,
            lint_errors=20,
            tests_passed=400,
            tests_failed=10,
        )
        after_snap = MetricSnapshot(
            timestamp=time.time(),
            files_count=100,
            total_lines=50000,
            lint_errors=5,
            tests_passed=408,
            tests_failed=2,
        )

        baseline = {"debate": None, "codebase": base_snap.to_dict()}
        after = {"debate": None, "codebase": after_snap.to_dict()}

        result = pipeline._compare_metrics(baseline, after)

        assert result is not None
        assert "improvement_score" in result
        assert result["improvement_score"] > 0

    def test_compare_metrics_produces_deltas(self):
        """_compare_metrics should populate deltas.codebase."""
        from aragora.nomic.metrics_collector import MetricSnapshot

        pipeline = SelfImprovePipeline()

        base_snap = MetricSnapshot(
            timestamp=time.time(),
            tests_passed=100,
            tests_failed=5,
            lint_errors=10,
        )
        after_snap = MetricSnapshot(
            timestamp=time.time(),
            tests_passed=105,
            tests_failed=3,
            lint_errors=8,
        )

        baseline = {"debate": None, "codebase": base_snap.to_dict()}
        after = {"debate": None, "codebase": after_snap.to_dict()}

        result = pipeline._compare_metrics(baseline, after)

        assert result is not None
        assert "deltas" in result
        assert "codebase" in result["deltas"]

    def test_compare_metrics_returns_none_for_none_inputs(self):
        """_compare_metrics returns None when baseline or after is None."""
        pipeline = SelfImprovePipeline()
        assert pipeline._compare_metrics(None, {"debate": None}) is None
        assert pipeline._compare_metrics({"debate": None}, None) is None
        assert pipeline._compare_metrics(None, None) is None

    def test_compare_metrics_codebase_improved_flag(self):
        """_compare_metrics sets codebase_improved when metrics improve."""
        from aragora.nomic.metrics_collector import MetricSnapshot

        pipeline = SelfImprovePipeline()

        base_snap = MetricSnapshot(
            timestamp=time.time(),
            lint_errors=20,
            tests_passed=100,
            tests_failed=10,
        )
        after_snap = MetricSnapshot(
            timestamp=time.time(),
            lint_errors=0,
            tests_passed=110,
            tests_failed=0,
        )

        baseline = {"debate": None, "codebase": base_snap.to_dict()}
        after = {"debate": None, "codebase": after_snap.to_dict()}

        result = pipeline._compare_metrics(baseline, after)
        assert result is not None
        assert result.get("codebase_improved") is True

    @pytest.mark.asyncio
    async def test_result_metrics_delta_populated_after_run(self):
        """SelfImproveResult.metrics_delta should be populated after run()."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                use_meta_planner=False,
                use_worktrees=False,
                capture_metrics=True,
                persist_outcomes=False,
                enable_codebase_indexing=False,
            )
        )

        with (
            patch.object(
                pipeline,
                "_capture_baseline",
                new_callable=AsyncMock,
                return_value={"debate": None, "codebase": {"files_count": 100}},
            ),
            patch.object(
                pipeline,
                "_capture_after",
                new_callable=AsyncMock,
                return_value={"debate": None, "codebase": {"files_count": 100}},
            ),
            patch.object(
                pipeline,
                "_compare_metrics",
                return_value={
                    "improved": True,
                    "recommendation": "keep",
                    "deltas": {"codebase": {"lint_delta": -5}},
                    "improvement_score": 0.7,
                    "_outcome_comparison": None,
                },
            ),
        ):
            result = await pipeline.run("Test delta population")

        # metrics_delta includes the codebase deltas plus any goal-achievement keys
        assert result.metrics_delta["codebase"] == {"lint_delta": -5}
        assert result.improvement_score == 0.7

    @pytest.mark.asyncio
    async def test_result_improvement_score_populated(self):
        """SelfImproveResult.improvement_score should reflect comparison."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                use_meta_planner=False,
                use_worktrees=False,
                capture_metrics=True,
                persist_outcomes=False,
                enable_codebase_indexing=False,
            )
        )

        with (
            patch.object(
                pipeline,
                "_capture_baseline",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch.object(
                pipeline,
                "_capture_after",
                new_callable=AsyncMock,
                return_value=MagicMock(),
            ),
            patch.object(
                pipeline,
                "_compare_metrics",
                return_value={
                    "improved": True,
                    "recommendation": "keep",
                    "deltas": {},
                    "improvement_score": 0.85,
                    "_outcome_comparison": None,
                },
            ),
        ):
            result = await pipeline.run("Test score")

        assert result.improvement_score == 0.85

    @pytest.mark.asyncio
    async def test_enable_codebase_metrics_false_skips_collection(self):
        """enable_codebase_metrics=False skips codebase baseline collection."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(enable_codebase_metrics=False, capture_metrics=True)
        )

        with patch(
            "aragora.nomic.outcome_tracker.NomicOutcomeTracker",
            side_effect=ImportError("skip"),
        ):
            result = await pipeline._capture_baseline()

        # debate failed, codebase skipped => None
        assert result is None

    def test_config_enable_codebase_metrics_default(self):
        """SelfImproveConfig.enable_codebase_metrics defaults to True."""
        config = SelfImproveConfig()
        assert config.enable_codebase_metrics is True

    def test_config_metrics_test_scope_default(self):
        """SelfImproveConfig.metrics_test_scope defaults to empty list."""
        config = SelfImproveConfig()
        assert config.metrics_test_scope == []

    def test_config_metrics_test_timeout_default(self):
        """SelfImproveConfig.metrics_test_timeout defaults to 120."""
        config = SelfImproveConfig()
        assert config.metrics_test_timeout == 120


# =========================================================================
# 4. Persistence (8 tests)
# =========================================================================


class TestMetricsPersistence:
    """Tests for metrics_delta flowing into KM cycle outcome."""

    def test_metrics_delta_in_evidence_quality_scores(self):
        """metrics_delta should flow into evidence_quality_scores when present."""
        pipeline = SelfImprovePipeline()
        result = SelfImproveResult(
            cycle_id="cycle_test",
            objective="Test persistence",
            subtasks_completed=2,
            subtasks_failed=0,
            files_changed=["x.py"],
            duration_seconds=10.0,
            metrics_delta={"codebase": {"lint_delta": -5}},
            improvement_score=0.65,
        )

        with patch("aragora.nomic.cycle_store.get_cycle_store") as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = mock_store
            pipeline._persist_outcome("cycle_test", result)

        mock_store.save_cycle.assert_called_once()
        record = mock_store.save_cycle.call_args[0][0]
        assert record.evidence_quality_scores["improvement_score"] == 0.65

    def test_has_metrics_key_set_when_delta_present(self):
        """has_metrics should be 1.0 in evidence_quality_scores when metrics_delta is non-empty."""
        pipeline = SelfImprovePipeline()
        result = SelfImproveResult(
            cycle_id="cycle_m",
            objective="Test",
            subtasks_completed=1,
            duration_seconds=5.0,
            metrics_delta={"codebase": {"lint_delta": -2}},
        )

        with patch("aragora.nomic.cycle_store.get_cycle_store") as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = mock_store
            pipeline._persist_outcome("cycle_m", result)

        record = mock_store.save_cycle.call_args[0][0]
        assert record.evidence_quality_scores.get("has_metrics") == 1.0

    def test_has_metrics_key_absent_when_no_delta(self):
        """has_metrics should NOT be in evidence_quality_scores when metrics_delta is empty."""
        pipeline = SelfImprovePipeline()
        result = SelfImproveResult(
            cycle_id="cycle_nd",
            objective="Test",
            subtasks_completed=1,
            duration_seconds=5.0,
            metrics_delta={},
        )

        with patch("aragora.nomic.cycle_store.get_cycle_store") as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = mock_store
            pipeline._persist_outcome("cycle_nd", result)

        record = mock_store.save_cycle.call_args[0][0]
        assert "has_metrics" not in record.evidence_quality_scores

    def test_improvement_score_persisted_as_float(self):
        """improvement_score should be a float in evidence_quality_scores."""
        pipeline = SelfImprovePipeline()
        result = SelfImproveResult(
            cycle_id="cycle_f",
            objective="Test",
            subtasks_completed=1,
            duration_seconds=1.0,
            improvement_score=0.42,
        )

        with patch("aragora.nomic.cycle_store.get_cycle_store") as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = mock_store
            pipeline._persist_outcome("cycle_f", result)

        record = mock_store.save_cycle.call_args[0][0]
        assert isinstance(record.evidence_quality_scores["improvement_score"], float)
        assert record.evidence_quality_scores["improvement_score"] == 0.42

    def test_persist_stores_subtask_counts(self):
        """evidence_quality_scores should include subtask completion counts."""
        pipeline = SelfImprovePipeline()
        result = SelfImproveResult(
            cycle_id="cycle_sc",
            objective="Test",
            subtasks_completed=3,
            subtasks_failed=1,
            files_changed=["a.py", "b.py"],
            duration_seconds=5.0,
        )

        with patch("aragora.nomic.cycle_store.get_cycle_store") as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = mock_store
            pipeline._persist_outcome("cycle_sc", result)

        record = mock_store.save_cycle.call_args[0][0]
        assert record.evidence_quality_scores["subtasks_completed"] == 3.0
        assert record.evidence_quality_scores["subtasks_failed"] == 1.0
        assert record.evidence_quality_scores["files_changed"] == 2.0

    def test_persist_stores_regressions_flag(self):
        """evidence_quality_scores should include regressions flag."""
        pipeline = SelfImprovePipeline()
        result = SelfImproveResult(
            cycle_id="cycle_r",
            objective="Test",
            regressions_detected=True,
            duration_seconds=1.0,
        )

        with patch("aragora.nomic.cycle_store.get_cycle_store") as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = mock_store
            pipeline._persist_outcome("cycle_r", result)

        record = mock_store.save_cycle.call_args[0][0]
        assert record.evidence_quality_scores["regressions"] == 1.0

    def test_persist_no_regressions_flag(self):
        """regressions should be 0.0 when not detected."""
        pipeline = SelfImprovePipeline()
        result = SelfImproveResult(
            cycle_id="cycle_nr",
            objective="Test",
            regressions_detected=False,
            duration_seconds=1.0,
        )

        with patch("aragora.nomic.cycle_store.get_cycle_store") as mock_get:
            mock_store = MagicMock()
            mock_get.return_value = mock_store
            pipeline._persist_outcome("cycle_nr", result)

        record = mock_store.save_cycle.call_args[0][0]
        assert record.evidence_quality_scores["regressions"] == 0.0

    def test_persist_handles_import_error(self):
        """_persist_outcome should not raise when cycle_store is unavailable."""
        pipeline = SelfImprovePipeline()
        result = SelfImproveResult(
            cycle_id="cycle_ie",
            objective="Test",
            duration_seconds=1.0,
            metrics_delta={"codebase": {}},
        )

        with patch.dict("sys.modules", {"aragora.nomic.cycle_store": None}):
            # Should not raise
            pipeline._persist_outcome("cycle_ie", result)


# =========================================================================
# 5. Graceful degradation (7 tests)
# =========================================================================


class TestGracefulDegradation:
    """Tests for graceful degradation on errors."""

    def test_enrich_survives_subprocess_failure(self):
        """Subprocess failures in size/lint collection should not crash."""
        planner = MetaPlanner()
        context = PlanningContext()

        mock_module, snap = _make_mock_collector()
        collector = mock_module.MetricsCollector.return_value
        collector._collect_size_metrics.side_effect = OSError("No such file or directory")
        collector._collect_lint_metrics.side_effect = OSError("ruff not installed")

        with patch.dict("sys.modules", _inject_metrics_module(mock_module)):
            enriched = planner._enrich_context_with_metrics(context)

        # Should still return a valid context
        assert isinstance(enriched, PlanningContext)

    def test_enrich_survives_import_error(self):
        """ImportError for metrics_collector should be handled silently."""
        planner = MetaPlanner()
        context = PlanningContext(recent_issues=["pre-existing"])

        with patch.dict("sys.modules", {"aragora.nomic.metrics_collector": None}):
            enriched = planner._enrich_context_with_metrics(context)

        assert enriched.metric_snapshot == {}
        assert "pre-existing" in enriched.recent_issues

    def test_enrich_survives_value_error(self):
        """ValueError during collection should be handled gracefully."""
        planner = MetaPlanner()
        context = PlanningContext()

        mock_module = ModuleType("aragora.nomic.metrics_collector")
        mock_module.MetricsCollectorConfig = MagicMock()  # type: ignore[attr-defined]
        mock_module.MetricsCollector = MagicMock(side_effect=ValueError("bad config"))  # type: ignore[attr-defined]

        with patch.dict("sys.modules", _inject_metrics_module(mock_module)):
            enriched = planner._enrich_context_with_metrics(context)

        assert isinstance(enriched, PlanningContext)

    @pytest.mark.asyncio
    async def test_capture_baseline_survives_collector_oserror(self):
        """OSError during baseline collection should not crash the pipeline."""
        pipeline = SelfImprovePipeline(SelfImproveConfig(enable_codebase_metrics=True))

        mock_module = ModuleType("aragora.nomic.metrics_collector")
        mock_module.MetricsCollectorConfig = MagicMock()  # type: ignore[attr-defined]
        mock_collector = MagicMock()
        mock_collector.collect_baseline = AsyncMock(side_effect=OSError("disk full"))
        mock_module.MetricsCollector = MagicMock(return_value=mock_collector)  # type: ignore[attr-defined]

        with (
            patch.dict("sys.modules", {"aragora.nomic.metrics_collector": mock_module}),
            patch(
                "aragora.nomic.outcome_tracker.NomicOutcomeTracker",
                side_effect=ImportError("skip"),
            ),
        ):
            result = await pipeline._capture_baseline()

        # Both debate and codebase failed => None
        assert result is None

    @pytest.mark.asyncio
    async def test_capture_after_survives_import_error(self):
        """ImportError for metrics_collector during after capture should be handled."""
        pipeline = SelfImprovePipeline(SelfImproveConfig(enable_codebase_metrics=True))

        with (
            patch.dict("sys.modules", {"aragora.nomic.metrics_collector": None}),
            patch(
                "aragora.nomic.outcome_tracker.NomicOutcomeTracker",
                side_effect=ImportError("skip"),
            ),
        ):
            result = await pipeline._capture_after()

        assert result is None

    @pytest.mark.asyncio
    async def test_compare_metrics_survives_import_error(self):
        """_compare_metrics should handle ImportError from MetricSnapshot.from_dict."""
        pipeline = SelfImprovePipeline()

        baseline = {"debate": None, "codebase": {"files_count": 10}}
        after = {"debate": None, "codebase": {"files_count": 10}}

        with patch.dict("sys.modules", {"aragora.nomic.metrics_collector": None}):
            result = pipeline._compare_metrics(baseline, after)

        # Should still return a result dict (debate comparison may work)
        assert result is not None
        assert "improved" in result

    @pytest.mark.asyncio
    async def test_full_pipeline_survives_metrics_failure(self):
        """Full pipeline run should not crash even if all metrics fail."""
        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                use_meta_planner=False,
                use_worktrees=False,
                capture_metrics=True,
                persist_outcomes=False,
                enable_codebase_indexing=False,
            )
        )

        with (
            patch.object(
                pipeline,
                "_capture_baseline",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = await pipeline.run("Survive metrics failure")

        assert isinstance(result, SelfImproveResult)
        # Codebase metrics should be empty (baseline was None), but
        # _evaluate_goal may inject goal_achievement / goal_scope_coverage /
        # goal_diff_relevance keys even when capture_metrics fails.
        assert "codebase" not in result.metrics_delta
        assert result.improvement_score == 0.0


# =========================================================================
# 6. SelfImproveResult serialization with metrics fields
# =========================================================================


class TestResultSerialization:
    """Verify metrics fields in SelfImproveResult.to_dict()."""

    def test_to_dict_includes_metrics_delta(self):
        """to_dict() should include metrics_delta."""
        result = SelfImproveResult(
            cycle_id="c1",
            objective="test",
            metrics_delta={"lint_delta": -5},
        )
        d = result.to_dict()
        assert d["metrics_delta"] == {"lint_delta": -5}

    def test_to_dict_includes_improvement_score(self):
        """to_dict() should include improvement_score."""
        result = SelfImproveResult(
            cycle_id="c2",
            objective="test",
            improvement_score=0.75,
        )
        d = result.to_dict()
        assert d["improvement_score"] == 0.75

    def test_to_dict_defaults(self):
        """Default metrics fields should be empty dict and 0.0."""
        result = SelfImproveResult(cycle_id="c3", objective="test")
        d = result.to_dict()
        assert d["metrics_delta"] == {}
        assert d["improvement_score"] == 0.0
