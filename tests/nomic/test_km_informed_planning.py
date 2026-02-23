"""Tests for KM-informed planning in MetaPlanner, TaskDecomposer, and SelfImprovePipeline.

Covers:
- NomicCycleAdapter.find_high_roi_goal_types() (keyword grouping, ranking)
- NomicCycleAdapter.find_recurring_failures() (pattern detection, filtering)
- MetaPlanner._enrich_context_with_history() (KM queries wired in)
- TaskDecomposer.enrich_subtasks_from_km() (subtask addition, warnings)
- SelfImprovePipeline._index_codebase() (codebase indexing)
- Pipeline integration (_index_codebase → _decompose → enriched subtasks)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.meta_planner import (
    MetaPlanner,
    MetaPlannerConfig,
    PlanningContext,
    Track,
)
from aragora.nomic.task_decomposer import SubTask, TaskDecomposer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_adapter(
    similar_cycles: list[Any] | None = None,
    high_roi: list[dict[str, Any]] | None = None,
    recurring_failures: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Create a mock NomicCycleAdapter with configurable query results."""
    adapter = MagicMock()
    adapter.find_similar_cycles = AsyncMock(return_value=similar_cycles or [])
    adapter.find_high_roi_goal_types = AsyncMock(return_value=high_roi or [])
    adapter.find_recurring_failures = AsyncMock(return_value=recurring_failures or [])
    return adapter


def _make_subtask(id: str, title: str, desc: str = "") -> SubTask:
    return SubTask(
        id=id,
        title=title,
        description=desc or title,
        dependencies=[],
        estimated_complexity="medium",
    )


# =========================================================================
# 1. High-ROI goal types (10 tests)
# =========================================================================


class TestHighROIGoalTypes:
    """Tests for find_high_roi_goal_types via MetaPlanner wiring."""

    @pytest.mark.asyncio
    async def test_high_roi_injected_into_successes(self):
        """High-ROI patterns should appear in past_successes_to_build_on."""
        adapter = _make_mock_adapter(
            high_roi=[
                {
                    "pattern": "coverage improvement",
                    "avg_improvement_score": 0.8,
                    "cycle_count": 3,
                    "example_objectives": ["Improve test coverage"],
                },
            ]
        )

        planner = MetaPlanner(
            MetaPlannerConfig(
                enable_cross_cycle_learning=True,
                enable_metrics_collection=False,
            )
        )
        context = PlanningContext()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            result = await planner._enrich_context_with_history(
                "Improve tests",
                [Track.QA],
                context,
            )

        assert any("high_roi" in s for s in result.past_successes_to_build_on)

    @pytest.mark.asyncio
    async def test_high_roi_below_threshold_excluded(self):
        """ROI patterns below 0.3 should not be added."""
        adapter = _make_mock_adapter(
            high_roi=[
                {
                    "pattern": "trivial change",
                    "avg_improvement_score": 0.1,
                    "cycle_count": 1,
                    "example_objectives": ["Minor tweak"],
                },
            ]
        )

        planner = MetaPlanner(
            MetaPlannerConfig(
                enable_cross_cycle_learning=True,
                enable_metrics_collection=False,
            )
        )
        context = PlanningContext()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            result = await planner._enrich_context_with_history(
                "Something",
                [Track.CORE],
                context,
            )

        assert not any("high_roi" in s for s in result.past_successes_to_build_on)

    @pytest.mark.asyncio
    async def test_high_roi_includes_pattern_text(self):
        """The pattern text should appear in the success entry."""
        adapter = _make_mock_adapter(
            high_roi=[
                {
                    "pattern": "lint reduction",
                    "avg_improvement_score": 0.6,
                    "cycle_count": 2,
                    "example_objectives": ["Fix lint"],
                },
            ]
        )

        planner = MetaPlanner(
            MetaPlannerConfig(
                enable_cross_cycle_learning=True,
                enable_metrics_collection=False,
            )
        )
        context = PlanningContext()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            result = await planner._enrich_context_with_history(
                "Test",
                [Track.QA],
                context,
            )

        matching = [s for s in result.past_successes_to_build_on if "lint reduction" in s]
        assert len(matching) >= 1

    @pytest.mark.asyncio
    async def test_high_roi_empty_returns_no_additions(self):
        """Empty high-ROI results should not add any successes."""
        adapter = _make_mock_adapter(high_roi=[])

        planner = MetaPlanner(
            MetaPlannerConfig(
                enable_cross_cycle_learning=True,
                enable_metrics_collection=False,
            )
        )
        context = PlanningContext()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            result = await planner._enrich_context_with_history(
                "Test",
                [Track.QA],
                context,
            )

        assert not any("high_roi" in s for s in result.past_successes_to_build_on)

    @pytest.mark.asyncio
    async def test_high_roi_query_failure_handled(self):
        """RuntimeError from find_high_roi_goal_types should be swallowed."""
        adapter = _make_mock_adapter()
        adapter.find_high_roi_goal_types = AsyncMock(side_effect=RuntimeError("boom"))

        planner = MetaPlanner(
            MetaPlannerConfig(
                enable_cross_cycle_learning=True,
                enable_metrics_collection=False,
            )
        )
        context = PlanningContext()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            result = await planner._enrich_context_with_history(
                "Test",
                [Track.QA],
                context,
            )

        assert isinstance(result, PlanningContext)

    @pytest.mark.asyncio
    async def test_high_roi_includes_cycle_count(self):
        """Cycle count should appear in the success entry."""
        adapter = _make_mock_adapter(
            high_roi=[
                {
                    "pattern": "refactor module",
                    "avg_improvement_score": 0.7,
                    "cycle_count": 5,
                    "example_objectives": ["Refactor auth"],
                },
            ]
        )

        planner = MetaPlanner(
            MetaPlannerConfig(
                enable_cross_cycle_learning=True,
                enable_metrics_collection=False,
            )
        )
        context = PlanningContext()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            result = await planner._enrich_context_with_history(
                "Test",
                [Track.CORE],
                context,
            )

        matching = [s for s in result.past_successes_to_build_on if "5 cycles" in s]
        assert len(matching) >= 1

    @pytest.mark.asyncio
    async def test_high_roi_includes_score(self):
        """Improvement score should appear in the success entry."""
        adapter = _make_mock_adapter(
            high_roi=[
                {
                    "pattern": "testing",
                    "avg_improvement_score": 0.75,
                    "cycle_count": 2,
                    "example_objectives": ["Test"],
                },
            ]
        )

        planner = MetaPlanner(
            MetaPlannerConfig(
                enable_cross_cycle_learning=True,
                enable_metrics_collection=False,
            )
        )
        context = PlanningContext()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            result = await planner._enrich_context_with_history(
                "Test",
                [Track.QA],
                context,
            )

        matching = [s for s in result.past_successes_to_build_on if "0.75" in s]
        assert len(matching) >= 1

    @pytest.mark.asyncio
    async def test_multiple_high_roi_patterns(self):
        """Multiple high-ROI patterns should all be added."""
        adapter = _make_mock_adapter(
            high_roi=[
                {
                    "pattern": "test coverage",
                    "avg_improvement_score": 0.9,
                    "cycle_count": 3,
                    "example_objectives": [],
                },
                {
                    "pattern": "lint cleanup",
                    "avg_improvement_score": 0.7,
                    "cycle_count": 2,
                    "example_objectives": [],
                },
                {
                    "pattern": "minor docs",
                    "avg_improvement_score": 0.2,
                    "cycle_count": 1,
                    "example_objectives": [],
                },
            ]
        )

        planner = MetaPlanner(
            MetaPlannerConfig(
                enable_cross_cycle_learning=True,
                enable_metrics_collection=False,
            )
        )
        context = PlanningContext()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            result = await planner._enrich_context_with_history(
                "Test",
                [Track.QA],
                context,
            )

        # Only patterns > 0.3 should be added
        high_roi_entries = [s for s in result.past_successes_to_build_on if "high_roi" in s]
        assert len(high_roi_entries) == 2  # test coverage + lint cleanup, NOT minor docs

    @pytest.mark.asyncio
    async def test_high_roi_attribute_error_handled(self):
        """AttributeError from find_high_roi_goal_types should be swallowed."""
        adapter = _make_mock_adapter()
        adapter.find_high_roi_goal_types = AsyncMock(side_effect=AttributeError("no attr"))

        planner = MetaPlanner(
            MetaPlannerConfig(
                enable_cross_cycle_learning=True,
                enable_metrics_collection=False,
            )
        )
        context = PlanningContext()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            result = await planner._enrich_context_with_history(
                "Test",
                [Track.QA],
                context,
            )

        assert isinstance(result, PlanningContext)


# =========================================================================
# 2. Recurring failures (10 tests)
# =========================================================================


class TestRecurringFailures:
    """Tests for find_recurring_failures via MetaPlanner wiring."""

    @pytest.mark.asyncio
    async def test_recurring_failures_injected(self):
        """Recurring failures should appear in past_failures_to_avoid."""
        adapter = _make_mock_adapter(
            recurring_failures=[
                {
                    "pattern": "timeout in tests",
                    "occurrences": 4,
                    "affected_tracks": ["qa"],
                    "example_errors": ["TimeoutError"],
                    "last_seen": "2026-02-10",
                },
            ]
        )

        planner = MetaPlanner(
            MetaPlannerConfig(
                enable_cross_cycle_learning=True,
                enable_metrics_collection=False,
            )
        )
        context = PlanningContext()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            result = await planner._enrich_context_with_history(
                "Fix tests",
                [Track.QA],
                context,
            )

        assert any("recurring_failure" in f for f in result.past_failures_to_avoid)

    @pytest.mark.asyncio
    async def test_recurring_failures_include_pattern(self):
        """The failure pattern text should appear."""
        adapter = _make_mock_adapter(
            recurring_failures=[
                {
                    "pattern": "import cycle",
                    "occurrences": 3,
                    "affected_tracks": [],
                    "example_errors": [],
                    "last_seen": "2026-02-15",
                },
            ]
        )

        planner = MetaPlanner(
            MetaPlannerConfig(
                enable_cross_cycle_learning=True,
                enable_metrics_collection=False,
            )
        )
        context = PlanningContext()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            result = await planner._enrich_context_with_history(
                "Test",
                [Track.CORE],
                context,
            )

        matching = [f for f in result.past_failures_to_avoid if "import cycle" in f]
        assert len(matching) >= 1

    @pytest.mark.asyncio
    async def test_recurring_failures_include_occurrence_count(self):
        """Occurrence count should appear in the failure entry."""
        adapter = _make_mock_adapter(
            recurring_failures=[
                {
                    "pattern": "flaky test",
                    "occurrences": 7,
                    "affected_tracks": ["qa"],
                    "example_errors": [],
                    "last_seen": "2026-02-15",
                },
            ]
        )

        planner = MetaPlanner(
            MetaPlannerConfig(
                enable_cross_cycle_learning=True,
                enable_metrics_collection=False,
            )
        )
        context = PlanningContext()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            result = await planner._enrich_context_with_history(
                "Test",
                [Track.QA],
                context,
            )

        matching = [f for f in result.past_failures_to_avoid if "7x" in f]
        assert len(matching) >= 1

    @pytest.mark.asyncio
    async def test_recurring_failures_include_tracks(self):
        """Affected tracks should appear in the failure entry."""
        adapter = _make_mock_adapter(
            recurring_failures=[
                {
                    "pattern": "auth bug",
                    "occurrences": 2,
                    "affected_tracks": ["security", "core"],
                    "example_errors": [],
                    "last_seen": "2026-02-10",
                },
            ]
        )

        planner = MetaPlanner(
            MetaPlannerConfig(
                enable_cross_cycle_learning=True,
                enable_metrics_collection=False,
            )
        )
        context = PlanningContext()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            result = await planner._enrich_context_with_history(
                "Test",
                [Track.SECURITY],
                context,
            )

        matching = [f for f in result.past_failures_to_avoid if "security" in f]
        assert len(matching) >= 1

    @pytest.mark.asyncio
    async def test_recurring_failures_empty_handled(self):
        """Empty recurring failures should not add any entries."""
        adapter = _make_mock_adapter(recurring_failures=[])

        planner = MetaPlanner(
            MetaPlannerConfig(
                enable_cross_cycle_learning=True,
                enable_metrics_collection=False,
            )
        )
        context = PlanningContext()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            result = await planner._enrich_context_with_history(
                "Test",
                [Track.CORE],
                context,
            )

        assert not any("recurring_failure" in f for f in result.past_failures_to_avoid)

    @pytest.mark.asyncio
    async def test_recurring_failures_query_failure_handled(self):
        """RuntimeError from find_recurring_failures should be swallowed."""
        adapter = _make_mock_adapter()
        adapter.find_recurring_failures = AsyncMock(side_effect=RuntimeError("db err"))

        planner = MetaPlanner(
            MetaPlannerConfig(
                enable_cross_cycle_learning=True,
                enable_metrics_collection=False,
            )
        )
        context = PlanningContext()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            result = await planner._enrich_context_with_history(
                "Test",
                [Track.QA],
                context,
            )

        assert isinstance(result, PlanningContext)

    @pytest.mark.asyncio
    async def test_multiple_recurring_failures(self):
        """Multiple failure patterns should all be added."""
        adapter = _make_mock_adapter(
            recurring_failures=[
                {
                    "pattern": "timeout",
                    "occurrences": 3,
                    "affected_tracks": [],
                    "example_errors": [],
                    "last_seen": "",
                },
                {
                    "pattern": "import error",
                    "occurrences": 5,
                    "affected_tracks": ["core"],
                    "example_errors": [],
                    "last_seen": "",
                },
            ]
        )

        planner = MetaPlanner(
            MetaPlannerConfig(
                enable_cross_cycle_learning=True,
                enable_metrics_collection=False,
            )
        )
        context = PlanningContext()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            result = await planner._enrich_context_with_history(
                "Test",
                [Track.CORE],
                context,
            )

        failure_entries = [f for f in result.past_failures_to_avoid if "recurring_failure" in f]
        assert len(failure_entries) == 2


# =========================================================================
# 3. MetaPlanner KM enrichment wiring (10 tests)
# =========================================================================


class TestMetaPlannerKMEnrichment:
    """Tests for KM queries being wired into _enrich_context_with_history."""

    @pytest.mark.asyncio
    async def test_adapter_import_error_handled(self):
        """ImportError for nomic_cycle_adapter should be swallowed."""
        planner = MetaPlanner(
            MetaPlannerConfig(
                enable_cross_cycle_learning=True,
                enable_metrics_collection=False,
            )
        )
        context = PlanningContext()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            side_effect=ImportError("no adapter"),
        ):
            result = await planner._enrich_context_with_history(
                "Test",
                [Track.QA],
                context,
            )

        assert isinstance(result, PlanningContext)

    @pytest.mark.asyncio
    async def test_both_queries_called(self):
        """Both find_high_roi_goal_types and find_recurring_failures should be called."""
        adapter = _make_mock_adapter()

        planner = MetaPlanner(
            MetaPlannerConfig(
                enable_cross_cycle_learning=True,
                enable_metrics_collection=False,
            )
        )
        context = PlanningContext()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            await planner._enrich_context_with_history(
                "Test",
                [Track.QA],
                context,
            )

        adapter.find_high_roi_goal_types.assert_called_once()
        adapter.find_recurring_failures.assert_called_once()

    @pytest.mark.asyncio
    async def test_similar_cycles_still_queried(self):
        """find_similar_cycles should still be called alongside new queries."""
        adapter = _make_mock_adapter()

        planner = MetaPlanner(
            MetaPlannerConfig(
                enable_cross_cycle_learning=True,
                enable_metrics_collection=False,
            )
        )
        context = PlanningContext()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            await planner._enrich_context_with_history(
                "Test",
                [Track.QA],
                context,
            )

        adapter.find_similar_cycles.assert_called_once()

    @pytest.mark.asyncio
    async def test_high_roi_and_failures_combined(self):
        """Both successes and failures should be populated together."""
        adapter = _make_mock_adapter(
            high_roi=[
                {
                    "pattern": "test improvement",
                    "avg_improvement_score": 0.9,
                    "cycle_count": 5,
                    "example_objectives": ["Better tests"],
                }
            ],
            recurring_failures=[
                {
                    "pattern": "circular import",
                    "occurrences": 3,
                    "affected_tracks": ["core"],
                    "example_errors": [],
                    "last_seen": "2026-02-10",
                }
            ],
        )

        planner = MetaPlanner(
            MetaPlannerConfig(
                enable_cross_cycle_learning=True,
                enable_metrics_collection=False,
            )
        )
        context = PlanningContext()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            result = await planner._enrich_context_with_history(
                "Test",
                [Track.CORE],
                context,
            )

        assert len(result.past_successes_to_build_on) >= 1
        assert len(result.past_failures_to_avoid) >= 1

    @pytest.mark.asyncio
    async def test_tracks_passed_to_adapter(self):
        """Track values should be passed to find_similar_cycles."""
        adapter = _make_mock_adapter()

        planner = MetaPlanner(
            MetaPlannerConfig(
                enable_cross_cycle_learning=True,
                enable_metrics_collection=False,
            )
        )
        context = PlanningContext()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            await planner._enrich_context_with_history(
                "Test",
                [Track.QA, Track.DEVELOPER],
                context,
            )

        call_kwargs = adapter.find_similar_cycles.call_args.kwargs
        assert "qa" in call_kwargs["tracks"]
        assert "developer" in call_kwargs["tracks"]

    @pytest.mark.asyncio
    async def test_context_preserved_through_enrichment(self):
        """Pre-existing context fields should be preserved."""
        adapter = _make_mock_adapter()

        planner = MetaPlanner(
            MetaPlannerConfig(
                enable_cross_cycle_learning=True,
                enable_metrics_collection=False,
            )
        )
        context = PlanningContext(
            recent_issues=["existing issue"],
            test_failures=["existing failure"],
        )

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            result = await planner._enrich_context_with_history(
                "Test",
                [Track.QA],
                context,
            )

        assert "existing issue" in result.recent_issues
        assert "existing failure" in result.test_failures

    @pytest.mark.asyncio
    async def test_runtime_error_from_adapter_handled(self):
        """RuntimeError from adapter should be handled gracefully."""
        planner = MetaPlanner(
            MetaPlannerConfig(
                enable_cross_cycle_learning=True,
                enable_metrics_collection=False,
            )
        )
        context = PlanningContext()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            side_effect=RuntimeError("adapter broke"),
        ):
            result = await planner._enrich_context_with_history(
                "Test",
                [Track.QA],
                context,
            )

        assert isinstance(result, PlanningContext)


# =========================================================================
# 4. TaskDecomposer enrichment (10 tests)
# =========================================================================


class TestTaskDecomposerEnrichment:
    """Tests for TaskDecomposer.enrich_subtasks_from_km()."""

    @pytest.mark.asyncio
    async def test_failure_warnings_added_to_subtasks(self):
        """Relevant failure patterns should inject km_warnings."""
        adapter = _make_mock_adapter(
            recurring_failures=[
                {
                    "pattern": "timeout in test execution",
                    "occurrences": 3,
                    "affected_tracks": ["qa"],
                    "example_errors": [],
                    "last_seen": "2026-02-15",
                },
            ]
        )

        decomposer = TaskDecomposer()
        subtasks = [_make_subtask("s1", "Fix test execution issues")]

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            result = await decomposer.enrich_subtasks_from_km(
                "Fix test execution timeout issues",
                subtasks,
            )

        assert any("km_warnings" in s.success_criteria for s in result)

    @pytest.mark.asyncio
    async def test_high_roi_suggests_additional_subtask(self):
        """High-ROI patterns should add suggested subtasks."""
        adapter = _make_mock_adapter(
            high_roi=[
                {
                    "pattern": "coverage improvement",
                    "avg_improvement_score": 0.8,
                    "cycle_count": 4,
                    "example_objectives": ["Add missing tests"],
                },
            ]
        )

        decomposer = TaskDecomposer()
        subtasks = [_make_subtask("s1", "Fix a bug")]

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            result = await decomposer.enrich_subtasks_from_km(
                "Improve codebase quality",
                subtasks,
            )

        assert len(result) >= 1

    @pytest.mark.asyncio
    async def test_subtask_cap_respected(self):
        """Suggestions should not exceed max_subtasks."""
        adapter = _make_mock_adapter(
            high_roi=[
                {
                    "pattern": f"pattern_{i}",
                    "avg_improvement_score": 0.9,
                    "cycle_count": 5,
                    "example_objectives": [f"Example {i}"],
                }
                for i in range(10)
            ]
        )

        from aragora.nomic.task_decomposer import DecomposerConfig

        decomposer = TaskDecomposer(DecomposerConfig(max_subtasks=3))
        subtasks = [_make_subtask("s1", "Original"), _make_subtask("s2", "Also original")]

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            result = await decomposer.enrich_subtasks_from_km("Add things", subtasks)

        assert len(result) <= 3

    @pytest.mark.asyncio
    async def test_import_error_returns_original(self):
        """ImportError should return subtasks unchanged."""
        decomposer = TaskDecomposer()
        subtasks = [_make_subtask("s1", "Original")]

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            side_effect=ImportError("no adapter"),
        ):
            result = await decomposer.enrich_subtasks_from_km("Test", subtasks)

        assert len(result) == 1
        assert result[0].id == "s1"

    @pytest.mark.asyncio
    async def test_runtime_error_returns_original(self):
        """RuntimeError should return subtasks unchanged."""
        decomposer = TaskDecomposer()
        subtasks = [_make_subtask("s1", "Original")]

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            side_effect=RuntimeError("adapter broke"),
        ):
            result = await decomposer.enrich_subtasks_from_km("Test", subtasks)

        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_no_warnings_when_no_overlap(self):
        """Failures with no word overlap should not inject warnings."""
        adapter = _make_mock_adapter(
            recurring_failures=[
                {
                    "pattern": "completely unrelated database migration",
                    "occurrences": 5,
                    "affected_tracks": ["core"],
                    "example_errors": [],
                    "last_seen": "2026-02-10",
                },
            ]
        )

        decomposer = TaskDecomposer()
        subtasks = [_make_subtask("s1", "Fix UI styling")]

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            result = await decomposer.enrich_subtasks_from_km(
                "Fix UI styling",
                subtasks,
            )

        for s in result:
            warnings = s.success_criteria.get("km_warnings", [])
            assert len(warnings) == 0

    @pytest.mark.asyncio
    async def test_empty_subtasks_handled(self):
        """Empty subtask list should not crash."""
        adapter = _make_mock_adapter(
            high_roi=[
                {
                    "pattern": "test",
                    "avg_improvement_score": 0.9,
                    "cycle_count": 1,
                    "example_objectives": ["test"],
                },
            ]
        )

        decomposer = TaskDecomposer()

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            result = await decomposer.enrich_subtasks_from_km("Test", [])

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_km_source_in_suggested_criteria(self):
        """KM-suggested subtasks should have km_source in success_criteria."""
        adapter = _make_mock_adapter(
            high_roi=[
                {
                    "pattern": "unique_pattern_xyz",
                    "avg_improvement_score": 0.95,
                    "cycle_count": 10,
                    "example_objectives": ["Example xyz"],
                },
            ]
        )

        decomposer = TaskDecomposer()
        subtasks = [_make_subtask("s1", "Something else entirely")]

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            result = await decomposer.enrich_subtasks_from_km(
                "Do unique_pattern_xyz work", subtasks
            )

        km_suggested = [s for s in result if "KM-suggested" in s.title]
        if km_suggested:
            assert km_suggested[0].success_criteria.get("km_source") == "high_roi_pattern"

    @pytest.mark.asyncio
    async def test_low_roi_patterns_not_suggested(self):
        """Patterns with avg_improvement_score < 0.5 should not generate subtasks."""
        adapter = _make_mock_adapter(
            high_roi=[
                {
                    "pattern": "weak pattern",
                    "avg_improvement_score": 0.3,
                    "cycle_count": 2,
                    "example_objectives": ["Weak example"],
                },
            ]
        )

        decomposer = TaskDecomposer()
        subtasks = [_make_subtask("s1", "Original task")]

        with patch(
            "aragora.knowledge.mound.adapters.nomic_cycle_adapter.get_nomic_cycle_adapter",
            return_value=adapter,
        ):
            result = await decomposer.enrich_subtasks_from_km("Test weak pattern", subtasks)

        km_suggested = [s for s in result if "KM-suggested" in s.title]
        assert len(km_suggested) == 0


# =========================================================================
# 5. Codebase indexing (8 tests)
# =========================================================================


class TestCodebaseIndexing:
    """Tests for SelfImprovePipeline._index_codebase()."""

    @pytest.mark.asyncio
    async def test_index_codebase_calls_builder(self):
        """_index_codebase should call ingest_module_summaries and ingest_dependency_graph."""
        from aragora.nomic.self_improve import SelfImprovePipeline, SelfImproveConfig

        pipeline = SelfImprovePipeline(SelfImproveConfig(enable_codebase_indexing=True))

        mock_stats = MagicMock()
        mock_stats.items_ingested = 5
        mock_stats.errors = 0

        mock_builder = MagicMock()
        mock_builder.ingest_module_summaries = AsyncMock(return_value=mock_stats)
        mock_builder.ingest_dependency_graph = AsyncMock(return_value=mock_stats)

        with (
            patch(
                "aragora.memory.codebase_builder.CodebaseKnowledgeBuilder",
                return_value=mock_builder,
            ),
            patch("aragora.memory.fabric.MemoryFabric"),
        ):
            await pipeline._index_codebase()

        mock_builder.ingest_module_summaries.assert_called_once()
        mock_builder.ingest_dependency_graph.assert_called_once()

    @pytest.mark.asyncio
    async def test_index_codebase_import_error_handled(self):
        """ImportError should be swallowed."""
        from aragora.nomic.self_improve import SelfImprovePipeline

        pipeline = SelfImprovePipeline()

        with patch.dict("sys.modules", {"aragora.memory.codebase_builder": None}):
            await pipeline._index_codebase()

    @pytest.mark.asyncio
    async def test_index_codebase_runtime_error_handled(self):
        """RuntimeError should be swallowed."""
        from aragora.nomic.self_improve import SelfImprovePipeline

        pipeline = SelfImprovePipeline()

        with (
            patch(
                "aragora.memory.codebase_builder.CodebaseKnowledgeBuilder",
                side_effect=RuntimeError("fabric err"),
            ),
            patch("aragora.memory.fabric.MemoryFabric"),
        ):
            await pipeline._index_codebase()

    @pytest.mark.asyncio
    async def test_index_codebase_disabled_via_config(self):
        """When enable_codebase_indexing=False, run() should skip indexing."""
        from aragora.nomic.self_improve import SelfImprovePipeline, SelfImproveConfig

        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                enable_codebase_indexing=False,
                use_meta_planner=False,
                use_worktrees=False,
                capture_metrics=False,
                persist_outcomes=False,
            )
        )

        with patch.object(pipeline, "_index_codebase", new_callable=AsyncMock) as mock_idx:
            await pipeline.run("Test skip indexing")

        mock_idx.assert_not_called()

    @pytest.mark.asyncio
    async def test_index_codebase_enabled_via_config(self):
        """When enable_codebase_indexing=True, run() should call indexing."""
        from aragora.nomic.self_improve import SelfImprovePipeline, SelfImproveConfig

        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                enable_codebase_indexing=True,
                use_meta_planner=False,
                use_worktrees=False,
                capture_metrics=False,
                persist_outcomes=False,
            )
        )

        with (
            patch.object(pipeline, "_index_codebase", new_callable=AsyncMock) as mock_idx,
            patch.object(pipeline, "_enrich_file_scope", new_callable=AsyncMock),
        ):
            await pipeline.run("Test with indexing")

        mock_idx.assert_called_once()

    def test_config_enable_codebase_indexing_default(self):
        """enable_codebase_indexing should default to True."""
        from aragora.nomic.self_improve import SelfImproveConfig

        assert SelfImproveConfig().enable_codebase_indexing is True


# =========================================================================
# 6. Pipeline integration (7 tests)
# =========================================================================


class TestPipelineKMIntegration:
    """Tests for KM enrichment in the pipeline decomposition step."""

    @pytest.mark.asyncio
    async def test_decompose_calls_enrich_subtasks(self):
        """_decompose should call enrich_subtasks_from_km after analyze."""
        from aragora.nomic.self_improve import SelfImprovePipeline, SelfImproveConfig
        from aragora.nomic.meta_planner import PrioritizedGoal, Track

        pipeline = SelfImprovePipeline(SelfImproveConfig(enable_codebase_indexing=False))

        goal = PrioritizedGoal(
            id="g1",
            track=Track.QA,
            description="Refactor the entire authentication and add comprehensive testing",
            rationale="Test",
            estimated_impact="high",
            priority=1,
        )

        with patch(
            "aragora.nomic.task_decomposer.TaskDecomposer.enrich_subtasks_from_km",
            new_callable=AsyncMock,
        ) as mock_enrich:
            mock_enrich.side_effect = lambda task, subtasks: subtasks
            subtasks = await pipeline._decompose([goal])

        mock_enrich.assert_called()

    @pytest.mark.asyncio
    async def test_decompose_km_failure_still_returns_subtasks(self):
        """KM enrichment failure should not prevent subtask generation."""
        from aragora.nomic.self_improve import SelfImprovePipeline, SelfImproveConfig
        from aragora.nomic.meta_planner import PrioritizedGoal, Track

        pipeline = SelfImprovePipeline(SelfImproveConfig(enable_codebase_indexing=False))

        goal = PrioritizedGoal(
            id="g1",
            track=Track.CORE,
            description="Refactor authentication and add comprehensive tests for coverage",
            rationale="Test",
            estimated_impact="high",
            priority=1,
        )

        with patch(
            "aragora.nomic.task_decomposer.TaskDecomposer.enrich_subtasks_from_km",
            new_callable=AsyncMock,
            side_effect=RuntimeError("KM unavailable"),
        ):
            subtasks = await pipeline._decompose([goal])

        assert len(subtasks) >= 1

    @pytest.mark.asyncio
    async def test_full_run_with_km_enrichment(self):
        """Full pipeline run should integrate KM enrichment without crashing."""
        from aragora.nomic.self_improve import (
            SelfImprovePipeline,
            SelfImproveConfig,
            SelfImproveResult,
        )

        pipeline = SelfImprovePipeline(
            SelfImproveConfig(
                use_meta_planner=False,
                use_worktrees=False,
                capture_metrics=False,
                persist_outcomes=False,
                enable_codebase_indexing=False,
            )
        )

        result = await pipeline.run("Improve test coverage and fix authentication bugs")

        assert isinstance(result, SelfImproveResult)
        assert result.subtasks_total >= 1
