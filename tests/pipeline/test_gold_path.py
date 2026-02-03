"""
Gold Path Integration Tests.

Tests the complete decision pipeline end-to-end:
    debate_result -> DecisionPlan -> WorkflowDefinition -> execution -> outcome -> memory

This validates that all the gold-path components connect properly:
1. DecisionPlanFactory creates a plan from DebateResult
2. Plan generates a valid WorkflowDefinition
3. Workflow contains correct step types (implementation, verification, memory_write)
4. ImplementationStep delegates to HybridExecutor
5. PlanExecutor orchestrates the full flow
6. Memory feedback loop records outcomes
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core_types import Critique, DebateResult
from aragora.implement.types import ImplementTask, TaskResult
from aragora.pipeline.decision_plan import (
    ApprovalMode,
    DecisionPlan,
    DecisionPlanFactory,
    PlanOutcome,
    PlanStatus,
    record_plan_outcome,
)
from aragora.pipeline.risk_register import RiskLevel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_debate_result(**overrides) -> DebateResult:
    """Create a DebateResult with sensible defaults for testing."""
    defaults = {
        "debate_id": "gold-path-test-001",
        "task": "Design a rate limiter for the API gateway",
        "final_answer": (
            "1. Implement token bucket algorithm in `rate_limiter.py`\n"
            "2. Add Redis backend for distributed counting in `redis_store.py`\n"
            "3. Create middleware wrapper for Flask routes in `middleware.py`\n"
            "4. Add per-endpoint configuration in `config.py`"
        ),
        "confidence": 0.87,
        "consensus_reached": True,
        "rounds_used": 3,
        "participants": ["claude", "gpt4", "gemini"],
        "total_cost_usd": 0.08,
    }
    defaults.update(overrides)
    return DebateResult(**defaults)


def _make_contested_result() -> DebateResult:
    """DebateResult with low confidence and critiques (requires approval)."""
    return _make_debate_result(
        debate_id="gold-path-contested-001",
        confidence=0.55,
        consensus_reached=False,
        critiques=[
            Critique(
                agent="gpt4",
                target_agent="claude",
                target_content="token bucket",
                issues=[
                    "No handling of burst traffic patterns",
                    "Redis single point of failure risk",
                ],
                suggestions=["Add sliding window fallback"],
                severity=8.5,
                reasoning="Reliability concerns",
            ),
        ],
        dissenting_views=["Consider sliding window algorithm instead"],
        debate_cruxes=[{"claim": "Token bucket is optimal", "sensitivity": 0.8}],
    )


# ---------------------------------------------------------------------------
# Test: Full pipeline - debate result to workflow
# ---------------------------------------------------------------------------


class TestGoldPathPipeline:
    """Tests the complete debate -> plan -> workflow pipeline."""

    def test_debate_result_to_workflow(self):
        """Core gold path: DebateResult -> DecisionPlan -> WorkflowDefinition."""
        result = _make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(
            result,
            budget_limit_usd=5.0,
            approval_mode=ApprovalMode.NEVER,
        )

        # Plan should be auto-approved with NEVER mode
        assert plan.status == PlanStatus.APPROVED
        assert plan.is_approved

        # Generate workflow
        workflow = plan.to_workflow_definition()

        # Verify workflow structure
        assert workflow.id == f"wf-{plan.id}"
        assert len(workflow.steps) >= 3  # impl + verify + memory

        # Check step types are correct
        step_types = {s.step_type for s in workflow.steps}
        assert "implementation" in step_types, (
            f"Expected 'implementation' step type, got: {step_types}"
        )
        assert "memory_write" in step_types

        # Check implementation steps have correct config
        impl_steps = [s for s in workflow.steps if s.step_type == "implementation"]
        assert len(impl_steps) >= 3
        for step in impl_steps:
            assert "task_id" in step.config
            assert "description" in step.config
            assert "files" in step.config
            assert "complexity" in step.config

        # Check workflow validates
        is_valid, errors = workflow.validate()
        assert is_valid, f"Workflow validation failed: {errors}"

    def test_contested_result_gets_approval_checkpoint(self):
        """Contested debates should generate human approval checkpoints."""
        result = _make_contested_result()
        plan = DecisionPlanFactory.from_debate_result(
            result,
            approval_mode=ApprovalMode.RISK_BASED,
        )

        # Should require approval (low confidence + no consensus = HIGH risk)
        assert plan.requires_human_approval
        assert plan.status == PlanStatus.AWAITING_APPROVAL

        workflow = plan.to_workflow_definition()

        # First step should be human checkpoint
        assert workflow.steps[0].step_type == "human_checkpoint"
        assert "Approval" in workflow.steps[0].name

    def test_risk_register_populated(self):
        """Risk analysis should extract risks from debate signals."""
        result = _make_contested_result()
        plan = DecisionPlanFactory.from_debate_result(result)

        assert plan.risk_register is not None
        risks = plan.risk_register.risks

        # Should have risks from: low confidence, no consensus, critiques, dissent, cruxes
        assert len(risks) >= 4

        # No consensus → HIGH risk
        no_consensus = [r for r in risks if "no consensus" in r.title.lower()]
        assert len(no_consensus) == 1
        assert no_consensus[0].level == RiskLevel.HIGH

    def test_verification_plan_from_consensus(self):
        """Verification plan should extract test cases from final answer."""
        result = _make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(result)

        assert plan.verification_plan is not None
        cases = plan.verification_plan.test_cases

        # Should have at least smoke + regression tests
        assert len(cases) >= 2
        smoke = [c for c in cases if c.id == "smoke-1"]
        assert len(smoke) == 1

    def test_budget_tracks_debate_cost(self):
        """Budget should track costs from the source debate."""
        result = _make_debate_result(total_cost_usd=0.15)
        plan = DecisionPlanFactory.from_debate_result(result, budget_limit_usd=10.0)

        assert plan.budget.debate_cost_usd == 0.15
        assert plan.budget.spent_usd == 0.15
        assert plan.budget.remaining_usd == pytest.approx(9.85)
        assert not plan.budget.over_budget

    def test_workflow_metadata_carries_context(self):
        """Workflow metadata should include debate context for traceability."""
        result = _make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(result)
        workflow = plan.to_workflow_definition()

        assert workflow.metadata["debate_id"] == "gold-path-test-001"
        assert workflow.metadata["debate_confidence"] == 0.87
        assert workflow.metadata["decision_plan_id"] == plan.id


# ---------------------------------------------------------------------------
# Test: ImplementationStep execution
# ---------------------------------------------------------------------------


class TestImplementationStepExecution:
    """Tests that ImplementationStep correctly delegates to HybridExecutor."""

    @pytest.mark.asyncio
    async def test_implementation_step_delegates_to_executor(self):
        """ImplementationStep should create ImplementTask and call HybridExecutor."""
        from aragora.workflow.nodes.implementation import ImplementationStep
        from aragora.workflow.step import WorkflowContext

        mock_result = TaskResult(
            task_id="task-1",
            success=True,
            diff="--- a/rate_limiter.py\n+++ b/rate_limiter.py\n+class TokenBucket:",
            model_used="claude",
            duration_seconds=3.5,
        )

        step = ImplementationStep(
            name="Implement: token bucket",
            config={
                "task_id": "task-1",
                "description": "Implement token bucket algorithm",
                "files": ["rate_limiter.py"],
                "complexity": "moderate",
            },
        )

        context = WorkflowContext(
            workflow_id="wf-test",
            definition_id="def-test",
            inputs={},
            step_outputs={},
            state={},
            metadata={},
            current_step_id="step-001",
            current_step_config={},
        )

        with patch("aragora.implement.executor.HybridExecutor") as MockExecutor:
            mock_instance = AsyncMock()
            mock_instance.execute_task.return_value = mock_result
            MockExecutor.return_value = mock_instance

            output = await step.execute(context)

        assert output["success"] is True
        assert output["task_id"] == "task-1"
        assert output["model_used"] == "claude"
        assert "TokenBucket" in output["diff"]

        # Verify HybridExecutor was called with correct ImplementTask
        mock_instance.execute_task.assert_awaited_once()
        called_task = mock_instance.execute_task.call_args[0][0]
        assert isinstance(called_task, ImplementTask)
        assert called_task.id == "task-1"
        assert called_task.description == "Implement token bucket algorithm"

    @pytest.mark.asyncio
    async def test_implementation_step_handles_failure(self):
        """ImplementationStep should handle executor failures gracefully."""
        from aragora.workflow.nodes.implementation import ImplementationStep
        from aragora.workflow.step import WorkflowContext

        mock_result = TaskResult(
            task_id="task-2",
            success=False,
            error="Syntax error in generated code",
            model_used="codex-fallback",
        )

        step = ImplementationStep(
            name="Implement: redis backend",
            config={
                "task_id": "task-2",
                "description": "Add Redis backend",
                "files": ["redis_store.py"],
                "complexity": "complex",
            },
        )

        context = WorkflowContext(
            workflow_id="wf-test",
            definition_id="def-test",
            inputs={},
            step_outputs={},
            state={},
            metadata={},
            current_step_id="step-002",
            current_step_config={},
        )

        with patch("aragora.implement.executor.HybridExecutor") as MockExecutor:
            mock_instance = AsyncMock()
            mock_instance.execute_task.return_value = mock_result
            MockExecutor.return_value = mock_instance

            output = await step.execute(context)

        assert output["success"] is False
        assert output["error"] == "Syntax error in generated code"

    @pytest.mark.asyncio
    async def test_implementation_step_handles_import_error(self):
        """ImplementationStep should degrade gracefully if executor unavailable."""
        from aragora.workflow.nodes.implementation import ImplementationStep
        from aragora.workflow.step import WorkflowContext

        step = ImplementationStep(
            name="Implement: something",
            config={
                "task_id": "task-3",
                "description": "Do something",
                "files": [],
                "complexity": "simple",
            },
        )

        context = WorkflowContext(
            workflow_id="wf-test",
            definition_id="def-test",
            inputs={},
            step_outputs={},
            state={},
            metadata={},
            current_step_id="step-003",
            current_step_config={},
        )

        with patch(
            "aragora.implement.executor.HybridExecutor",
        ) as MockExecutor:
            MockExecutor.side_effect = RuntimeError("Executor init failed")
            # Should not raise
            output = await step.execute(context)

        assert output["success"] is False
        assert "Executor init failed" in output["error"]


# ---------------------------------------------------------------------------
# Test: PlanExecutor orchestration
# ---------------------------------------------------------------------------


class TestPlanExecutorOrchestration:
    """Tests end-to-end PlanExecutor -> WorkflowEngine -> step execution."""

    @pytest.mark.asyncio
    async def test_executor_runs_workflow(self):
        """PlanExecutor should generate and execute workflow from plan."""
        from aragora.pipeline.executor import PlanExecutor

        result = _make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(
            result,
            approval_mode=ApprovalMode.NEVER,
        )

        # Mock the workflow engine execution
        mock_engine_result = MagicMock()
        mock_engine_result.success = True
        mock_engine_result.step_results = {
            "step-001": MagicMock(
                output={
                    "task_id": "task-1",
                    "success": True,
                    "diff": "+implemented",
                    "model_used": "claude",
                },
                status="completed",
            ),
            "step-002": MagicMock(
                output={"success": True, "tests_passed": 3},
                status="completed",
            ),
        }
        mock_engine_result.outputs = {}

        executor = PlanExecutor()

        with patch("aragora.workflow.engine.WorkflowEngine") as MockEngine:
            mock_eng = AsyncMock()
            mock_eng.execute.return_value = mock_engine_result
            MockEngine.return_value = mock_eng

            outcome = await executor.execute(plan)

        assert isinstance(outcome, PlanOutcome)
        assert outcome.success is True
        assert plan.status == PlanStatus.COMPLETED


# ---------------------------------------------------------------------------
# Test: Memory feedback loop
# ---------------------------------------------------------------------------


class TestMemoryFeedbackLoop:
    """Tests the learning stage of the gold path."""

    @pytest.mark.asyncio
    async def test_success_recorded_to_both_stores(self):
        """Successful outcomes should write to both continuum and mound."""
        result = _make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(result)
        plan.status = PlanStatus.EXECUTING

        mock_memory = AsyncMock()
        mock_entry = MagicMock()
        mock_entry.id = "mem-gold-001"
        mock_memory.store.return_value = mock_entry

        mock_mound = AsyncMock()
        mock_mound.store_knowledge.return_value = "km-gold-001"

        outcome = PlanOutcome(
            plan_id=plan.id,
            debate_id=plan.debate_id,
            task=plan.task,
            success=True,
            tasks_completed=4,
            tasks_total=4,
            verification_passed=6,
            verification_total=6,
            total_cost_usd=0.42,
            lessons=["Token bucket algorithm works well for rate limiting"],
        )

        results = await record_plan_outcome(
            plan,
            outcome,
            continuum_memory=mock_memory,
            knowledge_mound=mock_mound,
        )

        # Both stores should be written to
        assert results["continuum_id"] == "mem-gold-001"
        assert results["mound_id"] == "km-gold-001"
        assert results["errors"] == []

        # Plan status should be updated
        assert plan.status == PlanStatus.COMPLETED
        assert plan.memory_written is True
        assert plan.budget.spent_usd == 0.42

    @pytest.mark.asyncio
    async def test_failure_gets_higher_importance(self):
        """Failed implementations should be recorded with higher importance."""
        result = _make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(result)

        mock_memory = AsyncMock()
        mock_entry = MagicMock()
        mock_entry.id = "mem-fail-001"
        mock_memory.store.return_value = mock_entry

        outcome = PlanOutcome(
            plan_id=plan.id,
            debate_id=plan.debate_id,
            task=plan.task,
            success=False,
            error="Redis connection timeout",
            tasks_completed=2,
            tasks_total=4,
            total_cost_usd=0.20,
        )

        await record_plan_outcome(plan, outcome, continuum_memory=mock_memory)

        # Failures get importance=0.8 base, +0.2 for low verification rate (0/0=0.0 < 0.5)
        call_kwargs = mock_memory.store.call_args[1]
        assert call_kwargs["importance"] == 1.0
        assert plan.status == PlanStatus.FAILED
        assert plan.execution_error == "Redis connection timeout"


# ---------------------------------------------------------------------------
# Test: Step type registration
# ---------------------------------------------------------------------------


class TestStepTypeRegistration:
    """Tests that implementation steps are properly registered."""

    def test_implementation_step_registered(self):
        """ImplementationStep should be registered in the workflow engine."""
        from aragora.workflow.engine import WorkflowEngine

        engine = WorkflowEngine()
        assert "implementation" in engine._step_types
        assert "verification" in engine._step_types

    def test_step_types_in_workflow_match_engine(self):
        """Step types in generated workflow should match registered types."""
        from aragora.workflow.engine import WorkflowEngine

        result = _make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(result, approval_mode=ApprovalMode.NEVER)
        workflow = plan.to_workflow_definition()

        engine = WorkflowEngine()
        registered_types = set(engine._step_types.keys())

        for step in workflow.steps:
            assert step.step_type in registered_types, (
                f"Step type '{step.step_type}' not registered in engine. "
                f"Available: {registered_types}"
            )


# ---------------------------------------------------------------------------
# Test: Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerializationRoundTrip:
    """Tests that the full plan serializes and deserializes correctly."""

    def test_plan_to_dict_includes_all_artifacts(self):
        """Full plan serialization should include all gold path artifacts."""
        result = _make_contested_result()
        plan = DecisionPlanFactory.from_debate_result(result, budget_limit_usd=10.0)

        d = plan.to_dict()

        # Core fields
        assert d["debate_id"] == "gold-path-contested-001"
        assert d["status"] == "awaiting_approval"

        # Artifacts present
        assert d["risk_register"] is not None
        assert d["verification_plan"] is not None
        assert d["implement_plan"] is not None

        # Budget
        assert d["budget"]["limit_usd"] == 10.0
        assert d["budget"]["debate_cost_usd"] == 0.08

        # Computed properties
        assert d["has_critical_risks"] is True
        assert d["requires_human_approval"] is True

    def test_summary_human_readable(self):
        """Plan summary should be human-readable with key metrics."""
        result = _make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(result)

        summary = plan.summary()
        assert "Decision Plan" in summary
        assert "rate limiter" in summary.lower()
        assert "87%" in summary
