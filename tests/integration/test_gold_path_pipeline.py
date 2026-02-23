"""
Gold Path integration tests: debate → plan → approve → execute → verify → learn.

Tests the complete decision pipeline:
1. Create a DebateResult (mocked)
2. Build a DecisionPlan via DecisionPlanFactory
3. Approve/reject the plan
4. Execute via PlanExecutor (with mocked WorkflowEngine)
5. Verify outcome recording
6. Confirm memory write-back
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.pipeline.decision_plan import (
    ApprovalMode,
    BudgetAllocation,
    DecisionPlan,
    DecisionPlanFactory,
    PlanOutcome,
    PlanStatus,
    record_plan_outcome,
)
from aragora.pipeline.executor import (
    PlanExecutor,
    get_plan,
    list_plans,
    store_plan,
    _plan_store,
    _plan_outcomes,
)
from aragora.pipeline.risk_register import RiskLevel


# =============================================================================
# Fixtures
# =============================================================================


@dataclass
class MockCritique:
    agent: str = "critic-1"
    severity: float = 5.0
    issues: list[str] = field(default_factory=lambda: ["Minor issue"])
    suggestions: list[str] = field(default_factory=lambda: ["Fix it"])


@dataclass
class MockMessage:
    agent: str = "claude"
    content: str = "Test message"
    role: str = "proposer"
    round: int = 1


def _make_debate_result(**overrides: Any) -> MagicMock:
    """Create a mock DebateResult with sensible defaults."""
    result = MagicMock()
    result.debate_id = overrides.get("debate_id", "debate-test-001")
    result.task = overrides.get("task", "Design a rate limiter for the API")
    result.final_answer = overrides.get(
        "final_answer",
        "1. Create a token bucket algorithm in `api/rate_limiter.py`\n"
        "2. Add middleware to check rate limits on each request\n"
        "3. Use Redis for distributed state\n"
        "4. Add configuration for per-user limits\n",
    )
    result.consensus_reached = overrides.get("consensus_reached", True)
    result.confidence = overrides.get("confidence", 0.85)
    result.messages = overrides.get("messages", [MockMessage()])
    result.critiques = overrides.get("critiques", [MockCritique()])
    result.votes = overrides.get("votes", [])
    result.dissenting_views = overrides.get("dissenting_views", [])
    result.debate_cruxes = overrides.get("debate_cruxes", [])
    result.rounds_used = overrides.get("rounds_used", 3)
    result.participants = overrides.get("participants", ["claude", "gpt4", "gemini"])
    result.total_cost_usd = overrides.get("total_cost_usd", 0.05)
    result.total_tokens = overrides.get("total_tokens", 5000)
    result.per_agent_cost = overrides.get("per_agent_cost", {})
    result.budget_limit_usd = overrides.get("budget_limit_usd")
    result.metadata = overrides.get("metadata", {})
    result.to_dict.return_value = {
        "debate_id": result.debate_id,
        "task": result.task,
        "confidence": result.confidence,
    }
    return result


@pytest.fixture(autouse=True)
def _clear_plan_store():
    """Clear the in-memory plan store between tests."""
    _plan_store.clear()
    _plan_outcomes.clear()
    yield
    _plan_store.clear()
    _plan_outcomes.clear()


# =============================================================================
# Test: DecisionPlanFactory
# =============================================================================


class TestDecisionPlanFactory:
    """Test plan creation from debate results."""

    def test_creates_plan_from_debate_result(self):
        result = _make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(result)

        assert plan.debate_id == "debate-test-001"
        assert plan.task == "Design a rate limiter for the API"
        assert plan.debate_result is result
        assert plan.risk_register is not None
        assert plan.verification_plan is not None
        assert plan.implement_plan is not None

    def test_high_confidence_auto_approves(self):
        result = _make_debate_result(confidence=0.95, consensus_reached=True)
        plan = DecisionPlanFactory.from_debate_result(
            result,
            approval_mode=ApprovalMode.RISK_BASED,
            max_auto_risk=RiskLevel.LOW,
        )
        # High confidence + consensus = low risk = auto-approved
        assert plan.status == PlanStatus.APPROVED

    def test_low_confidence_requires_approval(self):
        result = _make_debate_result(confidence=0.4, consensus_reached=False)
        plan = DecisionPlanFactory.from_debate_result(
            result,
            approval_mode=ApprovalMode.RISK_BASED,
        )
        # Low confidence + no consensus = high risk = needs approval
        assert plan.status == PlanStatus.AWAITING_APPROVAL
        assert plan.requires_human_approval is True

    def test_always_approval_mode(self):
        result = _make_debate_result(confidence=0.99)
        plan = DecisionPlanFactory.from_debate_result(
            result,
            approval_mode=ApprovalMode.ALWAYS,
        )
        assert plan.status == PlanStatus.AWAITING_APPROVAL
        assert plan.requires_human_approval is True

    def test_never_approval_mode(self):
        result = _make_debate_result(confidence=0.3, consensus_reached=False)
        plan = DecisionPlanFactory.from_debate_result(
            result,
            approval_mode=ApprovalMode.NEVER,
        )
        assert plan.status == PlanStatus.APPROVED
        assert plan.requires_human_approval is False

    def test_budget_allocation(self):
        result = _make_debate_result(total_cost_usd=0.12)
        plan = DecisionPlanFactory.from_debate_result(
            result,
            budget_limit_usd=5.00,
        )
        assert plan.budget.limit_usd == 5.00
        assert plan.budget.debate_cost_usd == 0.12
        assert plan.budget.spent_usd == 0.12
        assert plan.budget.over_budget is False

    def test_risk_register_populated(self):
        result = _make_debate_result(
            confidence=0.4,
            consensus_reached=False,
            critiques=[MockCritique(severity=9.0, issues=["Critical security flaw"])],
        )
        plan = DecisionPlanFactory.from_debate_result(result)

        assert plan.risk_register is not None
        assert len(plan.risk_register.risks) > 0
        assert plan.has_critical_risks or plan.highest_risk_level in (
            RiskLevel.HIGH,
            RiskLevel.MEDIUM,
        )

    def test_implement_plan_extracts_tasks(self):
        result = _make_debate_result(
            final_answer=(
                "1. Create the token bucket implementation in `rate_limiter.py`\n"
                "2. Add Redis integration for distributed counters\n"
                "3. Write middleware that checks limits per endpoint\n"
            ),
        )
        plan = DecisionPlanFactory.from_debate_result(result)

        assert plan.implement_plan is not None
        assert len(plan.implement_plan.tasks) >= 3

    def test_verification_plan_has_smoke_test(self):
        result = _make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(result)

        assert plan.verification_plan is not None
        assert len(plan.verification_plan.test_cases) > 0
        # Should always include a smoke test
        ids = [tc.id for tc in plan.verification_plan.test_cases]
        assert "smoke-1" in ids


# =============================================================================
# Test: Plan Lifecycle (approve/reject)
# =============================================================================


class TestPlanLifecycle:
    """Test plan approval and rejection flow."""

    def test_approve_plan(self):
        result = _make_debate_result(confidence=0.3, consensus_reached=False)
        plan = DecisionPlanFactory.from_debate_result(result)
        assert plan.status == PlanStatus.AWAITING_APPROVAL

        plan.approve(approver_id="user-123", reason="Looks good")
        assert plan.status == PlanStatus.APPROVED
        assert plan.is_approved is True
        assert plan.approval_record is not None
        assert plan.approval_record.approver_id == "user-123"

    def test_reject_plan(self):
        result = _make_debate_result(confidence=0.3, consensus_reached=False)
        plan = DecisionPlanFactory.from_debate_result(result)

        plan.reject(approver_id="user-456", reason="Too risky")
        assert plan.status == PlanStatus.REJECTED
        assert plan.approval_record.approved is False

    def test_approve_with_conditions(self):
        result = _make_debate_result(confidence=0.5, consensus_reached=False)
        plan = DecisionPlanFactory.from_debate_result(result)

        plan.approve(
            approver_id="user-789",
            reason="Approved with caveats",
            conditions=["Run in staging first", "Monitor for 24h"],
        )
        assert plan.approval_record.conditions == [
            "Run in staging first",
            "Monitor for 24h",
        ]


# =============================================================================
# Test: Workflow Generation
# =============================================================================


class TestWorkflowGeneration:
    """Test that DecisionPlan generates valid WorkflowDefinitions."""

    def test_generates_workflow_definition(self):
        result = _make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(result)

        workflow = plan.to_workflow_definition()

        assert workflow is not None
        assert workflow.id.startswith("wf-dp-")
        assert len(workflow.steps) > 0
        assert plan.workflow_id == workflow.id

    def test_workflow_includes_memory_writeback(self):
        result = _make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(result)

        workflow = plan.to_workflow_definition()

        step_types = [s.step_type for s in workflow.steps]
        assert "memory_write" in step_types

    def test_workflow_includes_verification(self):
        result = _make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(result)

        workflow = plan.to_workflow_definition()

        step_names = [s.name for s in workflow.steps]
        assert any("Verification" in name for name in step_names)

    def test_high_risk_plan_includes_approval_checkpoint(self):
        result = _make_debate_result(confidence=0.3, consensus_reached=False)
        plan = DecisionPlanFactory.from_debate_result(
            result,
            approval_mode=ApprovalMode.ALWAYS,
        )

        workflow = plan.to_workflow_definition()

        step_types = [s.step_type for s in workflow.steps]
        assert "human_checkpoint" in step_types


# =============================================================================
# Test: Plan Store
# =============================================================================


class TestPlanStore:
    """Test in-memory plan store operations."""

    def test_store_and_retrieve(self):
        result = _make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(result)

        store_plan(plan)
        retrieved = get_plan(plan.id)

        assert retrieved is plan

    def test_list_plans_all(self):
        for i in range(3):
            result = _make_debate_result(debate_id=f"debate-{i}")
            plan = DecisionPlanFactory.from_debate_result(result)
            store_plan(plan)

        plans = list_plans()
        assert len(plans) == 3

    def test_list_plans_by_status(self):
        # Create one approved, one awaiting
        r1 = _make_debate_result(debate_id="d1", confidence=0.95)
        p1 = DecisionPlanFactory.from_debate_result(r1)
        store_plan(p1)

        r2 = _make_debate_result(debate_id="d2", confidence=0.3, consensus_reached=False)
        p2 = DecisionPlanFactory.from_debate_result(r2)
        store_plan(p2)

        approved = list_plans(status=PlanStatus.APPROVED)
        awaiting = list_plans(status=PlanStatus.AWAITING_APPROVAL)

        assert len(approved) >= 1
        assert len(awaiting) >= 1

    def test_get_nonexistent_returns_none(self):
        assert get_plan("nonexistent") is None


# =============================================================================
# Test: Plan Executor
# =============================================================================


class TestPlanExecutor:
    """Test PlanExecutor lifecycle with mocked WorkflowEngine."""

    @pytest.mark.asyncio
    async def test_execute_approved_plan(self):
        result = _make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(result, approval_mode=ApprovalMode.NEVER)
        store_plan(plan)
        assert plan.status == PlanStatus.APPROVED

        mock_wf_result = MagicMock()
        mock_wf_result.success = True
        mock_wf_result.error = None
        mock_wf_result.step_results = []

        with patch("aragora.workflow.engine.WorkflowEngine") as MockEngine:
            engine_instance = AsyncMock()
            engine_instance.execute.return_value = mock_wf_result
            MockEngine.return_value = engine_instance

            executor = PlanExecutor()
            outcome = await executor.execute(plan)

        assert outcome.success is True
        assert plan.status in (PlanStatus.COMPLETED, PlanStatus.FAILED)

    @pytest.mark.asyncio
    async def test_execute_rejected_plan_raises(self):
        result = _make_debate_result(confidence=0.3, consensus_reached=False)
        plan = DecisionPlanFactory.from_debate_result(result)
        plan.reject(approver_id="user-1", reason="No")
        store_plan(plan)

        executor = PlanExecutor()
        with pytest.raises(ValueError, match="rejected"):
            await executor.execute(plan)

    @pytest.mark.asyncio
    async def test_execute_unapproved_plan_raises(self):
        result = _make_debate_result(confidence=0.3, consensus_reached=False)
        plan = DecisionPlanFactory.from_debate_result(result)
        assert plan.status == PlanStatus.AWAITING_APPROVAL
        store_plan(plan)

        executor = PlanExecutor()
        with pytest.raises(ValueError, match="requires approval"):
            await executor.execute(plan)

    @pytest.mark.asyncio
    async def test_execute_records_outcome(self):
        result = _make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(result, approval_mode=ApprovalMode.NEVER)
        store_plan(plan)

        mock_wf_result = MagicMock()
        mock_wf_result.success = True
        mock_wf_result.error = None
        mock_wf_result.step_results = []

        with patch("aragora.workflow.engine.WorkflowEngine") as MockEngine:
            engine_instance = AsyncMock()
            engine_instance.execute.return_value = mock_wf_result
            MockEngine.return_value = engine_instance

            executor = PlanExecutor()
            outcome = await executor.execute(plan)

        from aragora.pipeline.executor import get_outcome

        stored_outcome = get_outcome(plan.id)
        assert stored_outcome is outcome
        assert stored_outcome.plan_id == plan.id

    @pytest.mark.asyncio
    async def test_execute_handles_workflow_failure(self):
        result = _make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(result, approval_mode=ApprovalMode.NEVER)
        store_plan(plan)

        with patch("aragora.workflow.engine.WorkflowEngine") as MockEngine:
            engine_instance = AsyncMock()
            engine_instance.execute.side_effect = RuntimeError("Workflow crashed")
            MockEngine.return_value = engine_instance

            executor = PlanExecutor()
            outcome = await executor.execute(plan)

        assert outcome.success is False
        assert "Workflow crashed" in outcome.error


# =============================================================================
# Test: Memory Write-back
# =============================================================================


class TestMemoryWriteback:
    """Test that plan outcomes are recorded to organizational memory."""

    @pytest.mark.asyncio
    async def test_record_outcome_to_continuum(self):
        result = _make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(result, approval_mode=ApprovalMode.NEVER)

        outcome = PlanOutcome(
            plan_id=plan.id,
            debate_id=plan.debate_id,
            task=plan.task,
            success=True,
            tasks_completed=3,
            tasks_total=3,
            verification_passed=5,
            verification_total=5,
            total_cost_usd=0.15,
            duration_seconds=10.0,
        )

        mock_memory = AsyncMock()
        mock_entry = MagicMock()
        mock_entry.id = "mem-001"
        mock_memory.store.return_value = mock_entry

        results = await record_plan_outcome(plan, outcome, continuum_memory=mock_memory)

        assert results["continuum_id"] == "mem-001"
        assert plan.status == PlanStatus.COMPLETED
        assert plan.memory_written is True

    @pytest.mark.asyncio
    async def test_record_outcome_failure_updates_status(self):
        result = _make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(result, approval_mode=ApprovalMode.NEVER)

        outcome = PlanOutcome(
            plan_id=plan.id,
            debate_id=plan.debate_id,
            task=plan.task,
            success=False,
            error="Verification failed",
        )

        results = await record_plan_outcome(plan, outcome)

        assert plan.status == PlanStatus.FAILED
        assert plan.execution_error == "Verification failed"

    @pytest.mark.asyncio
    async def test_record_outcome_to_knowledge_mound(self):
        result = _make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(result, approval_mode=ApprovalMode.NEVER)

        outcome = PlanOutcome(
            plan_id=plan.id,
            debate_id=plan.debate_id,
            task=plan.task,
            success=True,
            tasks_completed=2,
            tasks_total=2,
            total_cost_usd=0.08,
        )

        mock_mound = AsyncMock()
        mock_mound.store_knowledge.return_value = "km-001"

        results = await record_plan_outcome(plan, outcome, knowledge_mound=mock_mound)

        assert results["mound_id"] == "km-001"


# =============================================================================
# Test: Serialization
# =============================================================================


class TestPlanSerialization:
    """Test plan serialization for API responses."""

    def test_to_dict_roundtrip(self):
        result = _make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(result)

        d = plan.to_dict()

        assert d["id"] == plan.id
        assert d["debate_id"] == "debate-test-001"
        assert d["status"] == plan.status.value
        assert d["approval_mode"] == "risk_based"
        assert "budget" in d
        assert "risk_register" in d
        assert isinstance(d["has_critical_risks"], bool)
        assert isinstance(d["requires_human_approval"], bool)

    def test_summary_is_readable(self):
        result = _make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(result)

        summary = plan.summary()
        assert "Decision Plan" in summary
        assert plan.id in summary
        assert "Status:" in summary

    def test_outcome_to_dict(self):
        outcome = PlanOutcome(
            plan_id="dp-123",
            debate_id="debate-001",
            task="Test task",
            success=True,
            tasks_completed=3,
            tasks_total=4,
            verification_passed=2,
            verification_total=3,
            total_cost_usd=0.05,
            duration_seconds=5.0,
            lessons=["Lesson 1"],
        )

        d = outcome.to_dict()
        assert d["success"] is True
        assert d["completion_rate"] == 0.75
        assert d["verification_rate"] == pytest.approx(2 / 3)
        assert d["lessons"] == ["Lesson 1"]

    def test_outcome_to_memory_content(self):
        outcome = PlanOutcome(
            plan_id="dp-123",
            debate_id="debate-001",
            task="Build rate limiter",
            success=False,
            error="Redis unavailable",
            lessons=["Need Redis health check"],
        )

        content = outcome.to_memory_content()
        assert "FAILURE" in content
        assert "Redis unavailable" in content
        assert "Need Redis health check" in content
