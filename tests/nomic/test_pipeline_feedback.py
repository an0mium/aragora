"""Tests for pipeline feedback loop in MetaPlanner.

Verifies that:
- PlanStore.get_recent_outcomes() returns plan records
- MetaPlanner enriches context with pipeline outcomes
- Successes and failures are correctly categorized
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.pipeline.plan_store import PlanStore
from aragora.pipeline.decision_plan.core import (
    DecisionPlan,
    PlanStatus,
    ApprovalMode,
    BudgetAllocation,
)
from aragora.pipeline.risk_register import RiskLevel


def _make_plan(
    plan_id: str = "plan-001",
    task: str = "Improve dashboard UX",
    status: PlanStatus = PlanStatus.CREATED,
    debate_id: str = "debate-001",
) -> DecisionPlan:
    """Create a test DecisionPlan."""
    return DecisionPlan(
        id=plan_id,
        debate_id=debate_id,
        task=task,
        status=status,
        approval_mode=ApprovalMode.NEVER,
        max_auto_risk=RiskLevel.LOW,
        budget=BudgetAllocation(
            limit_usd=10.0,
            estimated_usd=5.0,
        ),
    )


class TestGetRecentOutcomes:
    """Tests for PlanStore.get_recent_outcomes()."""

    def test_returns_empty_when_no_plans(self):
        """Should return empty list when store is empty."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store = PlanStore(db_path=db_path)
            outcomes = store.get_recent_outcomes()
            assert outcomes == []
        finally:
            os.unlink(db_path)

    def test_returns_completed_plans(self):
        """Should return plans with terminal statuses."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store = PlanStore(db_path=db_path)

            # Create a completed plan
            plan = _make_plan(status=PlanStatus.CREATED)
            store.create(plan)
            store.update_status(plan.id, PlanStatus.COMPLETED)

            outcomes = store.get_recent_outcomes()
            assert len(outcomes) == 1
            assert outcomes[0]["plan_id"] == "plan-001"
            assert outcomes[0]["status"] == "completed"
            assert outcomes[0]["task"] == "Improve dashboard UX"
        finally:
            os.unlink(db_path)

    def test_returns_failed_plans(self):
        """Should include failed plans."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store = PlanStore(db_path=db_path)

            plan = _make_plan(plan_id="plan-fail", status=PlanStatus.CREATED)
            store.create(plan)
            store.update_status(plan.id, PlanStatus.FAILED)

            outcomes = store.get_recent_outcomes()
            assert len(outcomes) == 1
            assert outcomes[0]["status"] == "failed"
        finally:
            os.unlink(db_path)

    def test_excludes_created_and_approved_plans(self):
        """Should not return plans still in progress."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store = PlanStore(db_path=db_path)

            # Plan still in progress
            plan_created = _make_plan(plan_id="p1", status=PlanStatus.CREATED)
            store.create(plan_created)

            plan_approved = _make_plan(plan_id="p2", status=PlanStatus.CREATED)
            store.create(plan_approved)
            store.update_status(plan_approved.id, PlanStatus.APPROVED, approved_by="user")

            outcomes = store.get_recent_outcomes()
            assert len(outcomes) == 0
        finally:
            os.unlink(db_path)

    def test_respects_limit(self):
        """Should respect the limit parameter."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store = PlanStore(db_path=db_path)

            for i in range(5):
                plan = _make_plan(
                    plan_id=f"plan-{i}",
                    task=f"Task {i}",
                    status=PlanStatus.CREATED,
                )
                store.create(plan)
                store.update_status(plan.id, PlanStatus.COMPLETED)

            outcomes = store.get_recent_outcomes(limit=3)
            assert len(outcomes) == 3
        finally:
            os.unlink(db_path)

    def test_includes_execution_records(self):
        """Should join execution records when available."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store = PlanStore(db_path=db_path)

            plan = _make_plan(status=PlanStatus.CREATED)
            store.create(plan)
            store.update_status(plan.id, PlanStatus.FAILED)

            # Create execution record with error
            store.create_execution_record(
                plan_id=plan.id,
                debate_id=plan.debate_id,
                status="failed",
                error={"message": "Test execution failed"},
            )

            outcomes = store.get_recent_outcomes()
            assert len(outcomes) >= 1
            # At least one outcome should have execution data
            has_exec = any(o.get("execution_status") for o in outcomes)
            assert has_exec
        finally:
            os.unlink(db_path)

    def test_includes_refresh_scope_from_nomic_metadata(self):
        """Should surface file/test refresh scope from bounded work orders."""
        from aragora.nomic.pipeline_bridge import NomicPipelineBridge
        from aragora.nomic.task_decomposer import SubTask

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store = PlanStore(db_path=db_path)
            bridge = NomicPipelineBridge()

            plan = _make_plan(status=PlanStatus.CREATED)
            plan.metadata = bridge.build_plan_metadata(
                "Improve pipeline feedback cadence",
                [
                    SubTask(
                        id="sub-1",
                        title="Record refresh scope",
                        description="Persist changed files for next assessment",
                        file_scope=[
                            "aragora/nomic/pipeline_bridge.py",
                            "tests/nomic/test_pipeline_feedback.py",
                        ],
                        success_criteria={
                            "tests": [
                                "python -m pytest tests/nomic/test_pipeline_feedback.py -q",
                            ]
                        },
                    ),
                ],
            )

            store.create(plan)
            store.update_status(plan.id, PlanStatus.COMPLETED)

            outcomes = store.get_recent_outcomes()

            assert len(outcomes) == 1
            assert outcomes[0]["refresh_required"] is True
            assert outcomes[0]["affected_files"] == [
                "aragora/nomic/pipeline_bridge.py",
                "tests/nomic/test_pipeline_feedback.py",
            ]
            assert outcomes[0]["expected_tests"] == [
                "python -m pytest tests/nomic/test_pipeline_feedback.py -q"
            ]
        finally:
            os.unlink(db_path)

    def test_uses_latest_execution_record_per_plan(self):
        """Should avoid duplicate outcomes when a plan has multiple executions."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store = PlanStore(db_path=db_path)

            plan = _make_plan(status=PlanStatus.CREATED)
            store.create(plan)
            store.update_status(plan.id, PlanStatus.FAILED)

            store.create_execution_record(
                plan_id=plan.id,
                debate_id=plan.debate_id,
                status="failed",
                error={"message": "stale failure"},
            )
            store.create_execution_record(
                plan_id=plan.id,
                debate_id=plan.debate_id,
                status="failed",
                error={"message": "latest failure"},
            )

            outcomes = store.get_recent_outcomes()

            assert len(outcomes) == 1
            assert outcomes[0]["execution_error"]["message"] == "latest failure"
        finally:
            os.unlink(db_path)


class TestMetaPlannerPipelineFeedback:
    """Tests for MetaPlanner pipeline feedback integration."""

    @pytest.mark.asyncio
    async def test_enriches_context_with_successes(self):
        """Should add completed plans to past_successes."""
        from aragora.nomic.meta_planner import (
            MetaPlanner,
            MetaPlannerConfig,
            Track,
            PlanningContext,
        )

        mock_outcomes = [
            {
                "plan_id": "p1",
                "task": "Improved dashboard performance",
                "status": "completed",
                "debate_id": "d1",
                "created_at": "2026-02-15T00:00:00",
                "execution_status": "succeeded",
                "execution_error": None,
            },
        ]

        planner = MetaPlanner(
            config=MetaPlannerConfig(
                enable_cross_cycle_learning=False,
            )
        )
        context = PlanningContext()

        with patch("aragora.pipeline.plan_store.get_plan_store") as mock_store_fn:
            mock_store = MagicMock()
            mock_store.get_recent_outcomes.return_value = mock_outcomes
            mock_store_fn.return_value = mock_store

            # Temporarily import and monkey-patch for the test
            result = await planner._enrich_context_with_history("Improve UX", [Track.SME], context)

        assert any("pipeline" in s for s in result.past_successes_to_build_on)

    @pytest.mark.asyncio
    async def test_enriches_context_with_failures(self):
        """Should add failed plans to past_failures."""
        from aragora.nomic.meta_planner import (
            MetaPlanner,
            MetaPlannerConfig,
            Track,
            PlanningContext,
        )

        mock_outcomes = [
            {
                "plan_id": "p2",
                "task": "Migrate database schema",
                "status": "failed",
                "debate_id": "d2",
                "created_at": "2026-02-14T00:00:00",
                "execution_status": "failed",
                "execution_error": {"message": "Migration script timed out"},
            },
        ]

        planner = MetaPlanner(
            config=MetaPlannerConfig(
                enable_cross_cycle_learning=False,
            )
        )
        context = PlanningContext()

        with patch("aragora.pipeline.plan_store.get_plan_store") as mock_store_fn:
            mock_store = MagicMock()
            mock_store.get_recent_outcomes.return_value = mock_outcomes
            mock_store_fn.return_value = mock_store

            result = await planner._enrich_context_with_history("Improve DB", [Track.CORE], context)

        assert any("pipeline:failed" in f for f in result.past_failures_to_avoid)
        assert any("timed out" in f for f in result.past_failures_to_avoid)

    @pytest.mark.asyncio
    async def test_handles_missing_plan_store(self):
        """Should not crash if PlanStore is not available."""
        from aragora.nomic.meta_planner import (
            MetaPlanner,
            MetaPlannerConfig,
            Track,
            PlanningContext,
        )

        planner = MetaPlanner(
            config=MetaPlannerConfig(
                enable_cross_cycle_learning=False,
            )
        )
        context = PlanningContext()

        # Simulate the plan_store module being unavailable by making
        # get_plan_store raise an exception
        with patch(
            "aragora.pipeline.plan_store.get_plan_store",
            side_effect=RuntimeError("store unavailable"),
        ):
            result = await planner._enrich_context_with_history(
                "Do stuff", [Track.DEVELOPER], context
            )

        # Should return context unchanged, not crash
        assert result is context

    @pytest.mark.asyncio
    async def test_rejected_plans_treated_as_failures(self):
        """Rejected plans should appear in failures to avoid."""
        from aragora.nomic.meta_planner import (
            MetaPlanner,
            MetaPlannerConfig,
            Track,
            PlanningContext,
        )

        mock_outcomes = [
            {
                "plan_id": "p3",
                "task": "Risky refactor of core engine",
                "status": "rejected",
                "debate_id": "d3",
                "created_at": "2026-02-13T00:00:00",
                "execution_status": None,
                "execution_error": None,
            },
        ]

        planner = MetaPlanner(
            config=MetaPlannerConfig(
                enable_cross_cycle_learning=False,
            )
        )
        context = PlanningContext()

        with patch("aragora.pipeline.plan_store.get_plan_store") as mock_store_fn:
            mock_store = MagicMock()
            mock_store.get_recent_outcomes.return_value = mock_outcomes
            mock_store_fn.return_value = mock_store

            result = await planner._enrich_context_with_history(
                "Improve core", [Track.CORE], context
            )

        assert any("pipeline:rejected" in f for f in result.past_failures_to_avoid)

    @pytest.mark.asyncio
    async def test_pipeline_refresh_scope_is_added_to_assessment_context(self):
        """Failed pipeline work should seed recent changes and test targets."""
        from aragora.nomic.meta_planner import (
            MetaPlanner,
            MetaPlannerConfig,
            PlanningContext,
            Track,
        )
        from aragora.nomic.meta_planner_utils import build_debate_topic

        mock_outcomes = [
            {
                "plan_id": "p4",
                "task": "Tighten self-assessment cadence",
                "status": "failed",
                "debate_id": "d4",
                "created_at": "2026-02-12T00:00:00",
                "execution_status": "failed",
                "execution_error": {"message": "refresh scope was dropped"},
                "refresh_required": True,
                "affected_files": [
                    "aragora/nomic/pipeline_bridge.py",
                    "aragora/nomic/meta_planner.py",
                ],
                "expected_tests": [
                    "python -m pytest tests/nomic/test_pipeline_feedback.py -q",
                ],
            },
        ]

        planner = MetaPlanner(
            config=MetaPlannerConfig(
                enable_cross_cycle_learning=False,
            )
        )
        context = PlanningContext()

        with patch("aragora.pipeline.plan_store.get_plan_store") as mock_store_fn:
            mock_store = MagicMock()
            mock_store.get_recent_outcomes.return_value = mock_outcomes
            mock_store_fn.return_value = mock_store

            result = await planner._enrich_context_with_history(
                "Improve self-assessment loop", [Track.CORE], context
            )

        assert any("aragora/nomic/pipeline_bridge.py" in change for change in result.recent_changes)
        assert any(
            "python -m pytest tests/nomic/test_pipeline_feedback.py -q" in failure
            for failure in result.test_failures
        )
        assert any("files:" in failure for failure in result.past_failures_to_avoid)

        topic = build_debate_topic(
            "Improve self-assessment loop",
            [Track.CORE],
            [],
            result,
        )
        assert "RECENT CHANGES TO RE-ASSESS" in topic
        assert "aragora/nomic/meta_planner.py" in topic
