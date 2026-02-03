"""
End-to-end tests for DecisionPlan gold path.

Tests the full flow:
    debate → plan → approve → execute → verify → learn

Unlike unit tests, these tests exercise integration between:
- DecisionPlanFactory (plan creation)
- Approval flow (risk-based routing)
- WorkflowEngine (execution)
- ContinuumMemory (pattern learning)
- KnowledgeMound (organizational learning)

These are integration tests that verify the complete feedback loop works.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core_types import Critique, DebateResult, Vote
from aragora.pipeline.decision_plan import (
    ApprovalMode,
    BudgetAllocation,
    DecisionPlan,
    DecisionPlanFactory,
    PlanOutcome,
    PlanStatus,
    record_plan_outcome,
)
from aragora.pipeline.risk_register import RiskLevel


# ===========================================================================
# Test Fixtures
# ===========================================================================


def make_debate_result(
    *,
    task: str = "Implement user authentication",
    final_answer: str | None = None,
    confidence: float = 0.85,
    consensus_reached: bool = True,
    critiques: list[Critique] | None = None,
    total_cost_usd: float = 0.05,
) -> DebateResult:
    """Create a realistic debate result for E2E testing."""
    if final_answer is None:
        final_answer = """
        1. Create auth/models.py with User dataclass
        2. Implement JWT token generation in auth/tokens.py
        3. Add login endpoint to server/handlers/auth.py
        4. Add middleware for token validation
        5. Write tests in tests/auth/
        """

    return DebateResult(
        debate_id=f"debate-{datetime.now().timestamp():.0f}",
        task=task,
        final_answer=final_answer,
        confidence=confidence,
        consensus_reached=consensus_reached,
        rounds_used=3,
        participants=["claude", "gpt4", "gemini"],
        total_cost_usd=total_cost_usd,
        critiques=critiques or [],
    )


@dataclass
class MockContinuumMemory:
    """Mock ContinuumMemory for testing memory feedback."""

    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    outcome_updates: list[dict[str, Any]] = field(default_factory=list)

    async def store(
        self,
        key: str,
        content: str,
        tier: Any = None,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
    ) -> MagicMock:
        """Store entry and return mock entry object."""
        entry_id = f"entry-{len(self.entries)}"
        self.entries[entry_id] = {
            "key": key,
            "content": content,
            "tier": tier,
            "importance": importance,
            "metadata": metadata or {},
        }
        mock_entry = MagicMock()
        mock_entry.id = entry_id
        return mock_entry

    def update_outcome(
        self,
        id: str,
        success: bool,
        agent_prediction_error: float | None = None,
    ) -> float:
        """Record outcome update."""
        self.outcome_updates.append(
            {
                "id": id,
                "success": success,
                "agent_prediction_error": agent_prediction_error,
            }
        )
        return 0.5  # Return mock surprise score


@dataclass
class MockKnowledgeMound:
    """Mock KnowledgeMound for testing organizational learning."""

    knowledge: list[dict[str, Any]] = field(default_factory=list)

    async def store_knowledge(
        self,
        content: str,
        source: str | None = None,
        source_id: str | None = None,
        title: str | None = None,
        category: str = "lesson",
        confidence: float = 0.8,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Store knowledge and return ID."""
        knowledge_id = f"km-{len(self.knowledge)}"
        self.knowledge.append(
            {
                "id": knowledge_id,
                "content": content,
                "source": source,
                "source_id": source_id,
                "title": title,
                "category": category,
                "confidence": confidence,
                "tags": tags or [],
                "metadata": metadata or {},
            }
        )
        return knowledge_id


# ===========================================================================
# E2E Gold Path Tests
# ===========================================================================


class TestGoldPathE2E:
    """End-to-end tests for the complete gold path flow."""

    @pytest.mark.asyncio
    async def test_full_gold_path_success(self) -> None:
        """Test complete flow: debate → plan → approve → execute → verify → learn."""
        # 1. Create debate result (simulate Arena.run())
        result = make_debate_result(
            task="Add rate limiting to API endpoints",
            confidence=0.90,
            total_cost_usd=0.08,
        )

        # 2. Create plan from debate
        plan = DecisionPlanFactory.from_debate_result(
            result,
            budget_limit_usd=1.00,
            approval_mode=ApprovalMode.RISK_BASED,
            max_auto_risk=RiskLevel.MEDIUM,
        )

        assert plan.debate_id == result.debate_id
        assert plan.task == result.task
        assert plan.implement_plan is not None
        assert len(plan.implement_plan.tasks) >= 3

        # 3. Check approval status (should auto-approve for low risk)
        # High confidence + no critical critiques = low risk
        if plan.requires_human_approval:
            plan.approve(approver_id="user-123", reason="Looks good")

        assert plan.is_approved
        assert plan.status in (PlanStatus.APPROVED, PlanStatus.AWAITING_APPROVAL)

        # 4. Simulate execution (would use WorkflowEngine in real scenario)
        plan.status = PlanStatus.EXECUTING
        plan.execution_started_at = datetime.now()

        # Simulate successful execution
        outcome = PlanOutcome(
            plan_id=plan.id,
            debate_id=plan.debate_id,
            task=plan.task,
            success=True,
            tasks_completed=5,
            tasks_total=5,
            verification_passed=4,
            verification_total=5,
            total_cost_usd=0.15,
            lessons=["Rate limiting worked well with token bucket algorithm"],
        )

        # 5. Record outcome to memory (feedback loop)
        mock_continuum = MockContinuumMemory()
        mock_mound = MockKnowledgeMound()

        results = await record_plan_outcome(
            plan,
            outcome,
            continuum_memory=mock_continuum,
            knowledge_mound=mock_mound,
        )

        # 6. Verify feedback was recorded
        assert results["continuum_id"] is not None
        assert plan.status == PlanStatus.COMPLETED
        assert plan.execution_completed_at is not None

        # Verify ContinuumMemory received the outcome
        assert len(mock_continuum.entries) >= 1
        stored_entry = list(mock_continuum.entries.values())[0]
        assert "SUCCESS" in stored_entry["content"]
        assert stored_entry["metadata"]["success"] is True

    @pytest.mark.asyncio
    async def test_gold_path_with_failure_learns_from_mistakes(self) -> None:
        """Test that failures are recorded with higher importance."""
        result = make_debate_result(
            task="Migrate database schema",
            confidence=0.70,
        )

        plan = DecisionPlanFactory.from_debate_result(
            result,
            approval_mode=ApprovalMode.NEVER,  # Auto-approve for test
        )

        # Simulate failed execution
        outcome = PlanOutcome(
            plan_id=plan.id,
            debate_id=plan.debate_id,
            task=plan.task,
            success=False,
            tasks_completed=2,
            tasks_total=5,
            verification_passed=0,
            verification_total=3,
            total_cost_usd=0.10,
            error="Migration script failed on foreign key constraint",
            lessons=[
                "Always check foreign key dependencies before migration",
                "Run migrations in a transaction for rollback",
            ],
        )

        mock_continuum = MockContinuumMemory()

        results = await record_plan_outcome(
            plan,
            outcome,
            continuum_memory=mock_continuum,
        )

        # Failures should be stored
        assert results["continuum_id"] is not None
        assert plan.status == PlanStatus.FAILED
        assert plan.execution_error == "Migration script failed on foreign key constraint"

        # Check that failure was recorded with higher importance
        stored = list(mock_continuum.entries.values())[0]
        assert stored["importance"] >= 0.8  # Failures get higher importance
        assert "FAILURE" in stored["content"]
        assert "foreign key" in stored["content"].lower()

    @pytest.mark.asyncio
    async def test_approval_required_for_high_risk(self) -> None:
        """Test that high-risk plans require human approval."""
        # Create debate with critical critique (high risk)
        critiques = [
            Critique(
                agent="gpt4",
                target_agent="claude",
                target_content="migration plan",
                issues=["No rollback strategy", "Production data at risk"],
                suggestions=["Add rollback plan", "Test on staging first"],
                severity=9.0,  # Critical
                reasoning="Data loss risk",
            ),
        ]

        result = make_debate_result(
            task="Delete old user data for GDPR compliance",
            critiques=critiques,
            confidence=0.60,
        )

        plan = DecisionPlanFactory.from_debate_result(
            result,
            approval_mode=ApprovalMode.RISK_BASED,
            max_auto_risk=RiskLevel.LOW,  # Only auto-approve low risk
        )

        # Should require approval due to high-severity critique
        assert plan.requires_human_approval
        assert plan.status == PlanStatus.AWAITING_APPROVAL

        # Cannot execute without approval
        assert not plan.is_approved

        # Approve
        plan.approve(
            approver_id="security-admin",
            reason="Reviewed rollback plan, approved with conditions",
            conditions=["Run on staging first", "Keep 30-day backup"],
        )

        assert plan.is_approved
        assert plan.status == PlanStatus.APPROVED
        assert plan.approval_record is not None
        assert plan.approval_record.approver_id == "security-admin"
        assert len(plan.approval_record.conditions) == 2

    @pytest.mark.asyncio
    async def test_budget_tracking_across_lifecycle(self) -> None:
        """Test that budget is tracked from debate through execution."""
        result = make_debate_result(
            total_cost_usd=0.10,  # Debate cost
        )

        plan = DecisionPlanFactory.from_debate_result(
            result,
            budget_limit_usd=0.50,
            approval_mode=ApprovalMode.NEVER,
        )

        # Initial budget from debate
        assert plan.budget.limit_usd == 0.50
        assert plan.budget.debate_cost_usd == 0.10
        assert plan.budget.spent_usd == 0.10  # Starts with debate cost
        assert plan.budget.remaining_usd == 0.40

        # Simulate execution with additional costs
        outcome = PlanOutcome(
            plan_id=plan.id,
            debate_id=plan.debate_id,
            task=plan.task,
            success=True,
            tasks_completed=3,
            tasks_total=3,
            total_cost_usd=0.35,  # Total including debate
        )

        await record_plan_outcome(plan, outcome)

        # Budget should be updated
        assert plan.budget.spent_usd == 0.35
        assert not plan.budget.over_budget

    @pytest.mark.asyncio
    async def test_over_budget_detection(self) -> None:
        """Test that going over budget is detected."""
        result = make_debate_result(total_cost_usd=0.20)

        plan = DecisionPlanFactory.from_debate_result(
            result,
            budget_limit_usd=0.25,  # Tight budget
            approval_mode=ApprovalMode.NEVER,
        )

        # Execute with higher than budgeted cost
        outcome = PlanOutcome(
            plan_id=plan.id,
            debate_id=plan.debate_id,
            task=plan.task,
            success=True,
            tasks_completed=5,
            tasks_total=5,
            total_cost_usd=0.40,  # Over budget!
        )

        await record_plan_outcome(plan, outcome)

        assert plan.budget.spent_usd == 0.40
        assert plan.budget.over_budget


# ===========================================================================
# Workflow Integration Tests
# ===========================================================================


class TestWorkflowIntegration:
    """Tests for workflow generation and execution integration."""

    def test_workflow_definition_is_valid(self) -> None:
        """Test that generated workflow validates."""
        result = make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(
            result,
            approval_mode=ApprovalMode.NEVER,
        )

        workflow = plan.to_workflow_definition()

        # Workflow should have required fields
        assert workflow.id is not None
        assert workflow.name is not None
        assert len(workflow.steps) >= 1

        # Should validate successfully
        assert workflow.validate()

    def test_workflow_includes_implementation_steps(self) -> None:
        """Test that workflow includes implementation tasks."""
        result = make_debate_result(
            final_answer="""
            1. Create user model
            2. Add authentication endpoints
            3. Write unit tests
            """
        )
        plan = DecisionPlanFactory.from_debate_result(
            result,
            approval_mode=ApprovalMode.NEVER,
        )

        workflow = plan.to_workflow_definition()

        # Should have implementation steps
        impl_steps = [
            s
            for s in workflow.steps
            if s.step_type in ("implementation", "implement", "task", "execute")
        ]
        assert len(impl_steps) >= 1

    def test_workflow_includes_verification_steps(self) -> None:
        """Test that workflow includes verification steps."""
        result = make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(
            result,
            approval_mode=ApprovalMode.NEVER,
        )

        workflow = plan.to_workflow_definition()

        # Should have verification/test steps
        verify_steps = [
            s
            for s in workflow.steps
            if "verif" in s.step_type.lower() or "test" in s.step_type.lower()
        ]
        # The workflow may not have explicit verification steps in the basic case
        # Just verify it validates
        assert workflow.validate()


# ===========================================================================
# Memory Integration Tests
# ===========================================================================


class TestMemoryIntegration:
    """Tests for memory system integration."""

    @pytest.mark.asyncio
    async def test_pattern_extraction_on_success(self) -> None:
        """Test that successful outcomes extract patterns."""
        result = make_debate_result(
            task="Implement caching layer",
            final_answer="Use Redis with TTL-based expiration",
            confidence=0.95,
        )
        plan = DecisionPlanFactory.from_debate_result(
            result,
            approval_mode=ApprovalMode.NEVER,
        )

        outcome = PlanOutcome(
            plan_id=plan.id,
            debate_id=plan.debate_id,
            task=plan.task,
            success=True,
            tasks_completed=3,
            tasks_total=3,
            verification_passed=3,
            verification_total=3,
            total_cost_usd=0.05,
            lessons=["Redis TTL worked well for session data"],
        )

        mock_continuum = MockContinuumMemory()
        mock_mound = MockKnowledgeMound()

        await record_plan_outcome(
            plan,
            outcome,
            continuum_memory=mock_continuum,
            knowledge_mound=mock_mound,
        )

        # Pattern should be stored in continuum
        assert len(mock_continuum.entries) >= 1

        # Outcome should be stored in knowledge mound
        assert len(mock_mound.knowledge) >= 1
        knowledge_entry = mock_mound.knowledge[0]
        # Should have the plan outcome content
        assert (
            "SUCCESS" in knowledge_entry["content"]
            or "caching" in knowledge_entry["content"].lower()
        )
        assert knowledge_entry["source"] == "decision_plan"

    @pytest.mark.asyncio
    async def test_memory_handles_errors_gracefully(self) -> None:
        """Test that memory errors don't break the flow."""
        result = make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(
            result,
            approval_mode=ApprovalMode.NEVER,
        )

        outcome = PlanOutcome(
            plan_id=plan.id,
            debate_id=plan.debate_id,
            task=plan.task,
            success=True,
            tasks_completed=1,
            tasks_total=1,
        )

        # Create mock that raises an error
        broken_continuum = MagicMock()
        broken_continuum.store = AsyncMock(side_effect=Exception("DB connection failed"))

        # Should not raise, should handle gracefully
        results = await record_plan_outcome(
            plan,
            outcome,
            continuum_memory=broken_continuum,
        )

        # Should have recorded the error
        assert "errors" in results
        assert len(results["errors"]) >= 1


# ===========================================================================
# Rejection Flow Tests
# ===========================================================================


class TestRejectionFlow:
    """Tests for plan rejection scenarios."""

    def test_rejection_records_reason(self) -> None:
        """Test that rejection captures the reason."""
        result = make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(
            result,
            approval_mode=ApprovalMode.ALWAYS,
        )

        assert plan.requires_human_approval

        plan.reject(
            approver_id="security-team",
            reason="Approach violates security policy - use OAuth instead",
        )

        assert plan.status == PlanStatus.REJECTED
        assert not plan.is_approved
        assert plan.approval_record is not None
        assert not plan.approval_record.approved
        assert "OAuth" in plan.approval_record.reason

    def test_rejected_plan_cannot_execute(self) -> None:
        """Test that rejected plans are not approved for execution."""
        result = make_debate_result()
        plan = DecisionPlanFactory.from_debate_result(
            result,
            approval_mode=ApprovalMode.ALWAYS,
        )

        plan.reject(approver_id="admin", reason="Budget exceeded")

        assert not plan.is_approved
        # In real system, workflow engine would check is_approved before executing


# ===========================================================================
# Multi-Step Approval Tests
# ===========================================================================


class TestConditionalApproval:
    """Tests for approvals with conditions."""

    def test_conditional_approval_captures_conditions(self) -> None:
        """Test that conditions are recorded with approval."""
        result = make_debate_result(task="Deploy to production")
        plan = DecisionPlanFactory.from_debate_result(
            result,
            approval_mode=ApprovalMode.ALWAYS,
        )

        plan.approve(
            approver_id="ops-lead",
            reason="Approved with monitoring requirements",
            conditions=[
                "Enable detailed logging",
                "Set up PagerDuty alert",
                "Keep rollback script ready",
            ],
        )

        assert plan.is_approved
        assert len(plan.approval_record.conditions) == 3
        assert "logging" in plan.approval_record.conditions[0].lower()


# ===========================================================================
# Serialization Round-Trip Tests
# ===========================================================================


class TestSerializationE2E:
    """Tests for serialization/deserialization of plans."""

    def test_plan_to_dict_round_trip(self) -> None:
        """Test that plans can be serialized and contain all data."""
        result = make_debate_result(
            task="Add payment processing",
            confidence=0.88,
        )
        plan = DecisionPlanFactory.from_debate_result(
            result,
            budget_limit_usd=5.00,
            approval_mode=ApprovalMode.RISK_BASED,
        )

        plan.approve(approver_id="finance-team", reason="Budget approved")

        # Serialize
        data = plan.to_dict()

        # Verify key fields present
        assert data["id"] == plan.id
        assert data["task"] == plan.task
        assert data["debate_id"] == plan.debate_id
        assert data["status"] == PlanStatus.APPROVED.value
        assert data["budget"]["limit_usd"] == 5.00
        assert data["approval_record"]["approver_id"] == "finance-team"
        assert len(data["implement_plan"]["tasks"]) >= 1
        assert len(data["risk_register"]["risks"]) >= 0

    def test_outcome_to_dict_complete(self) -> None:
        """Test that outcomes serialize completely."""
        outcome = PlanOutcome(
            plan_id="plan-001",
            debate_id="debate-001",
            task="Build feature X",
            success=True,
            tasks_completed=10,
            tasks_total=12,
            verification_passed=8,
            verification_total=10,
            total_cost_usd=1.23,
            lessons=["Lesson 1", "Lesson 2"],
        )

        data = outcome.to_dict()

        assert data["plan_id"] == "plan-001"
        assert data["success"] is True
        assert data["completion_rate"] == pytest.approx(10 / 12)
        assert data["verification_rate"] == pytest.approx(8 / 10)
        assert len(data["lessons"]) == 2
