import pytest
from types import SimpleNamespace
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from aragora.implement.types import ImplementPlan, ImplementTask
from aragora.pipeline.decision_plan import ApprovalMode, DecisionPlan, PlanStatus
from aragora.pipeline.decision_plan.memory import PlanOutcome
from aragora.server.decision_integrity_utils import (
    _normalize_execution_request_for_safety_mode,
    build_decision_integrity_payload,
    extract_execution_overrides,
)
from aragora.pipeline.execution_mode import ExecutionMode


def test_extract_execution_overrides_computer_use():
    text, overrides = extract_execution_overrides("implement update docs --computer-use")
    assert text == "implement update docs"
    assert overrides["execution_mode"] == "execute"
    assert overrides["execution_engine"] == "computer_use"


def test_extract_execution_overrides_hybrid():
    text, overrides = extract_execution_overrides("implement update docs --hybrid")
    assert text == "implement update docs"
    assert overrides["execution_mode"] == "execute"
    assert overrides["execution_engine"] == "hybrid"


def test_normalize_execution_request_defaults_invalid_mode_to_plan_only():
    assert (
        _normalize_execution_request_for_safety_mode(
            None,
            safety_mode=ExecutionMode.AUTONOMOUS,
        )
        == "plan_only"
    )


def test_normalize_execution_request_downgrades_interactive_execute():
    assert (
        _normalize_execution_request_for_safety_mode(
            "execute",
            safety_mode=ExecutionMode.INTERACTIVE,
        )
        == "request_approval"
    )


@pytest.mark.asyncio
async def test_build_payload_executes_hybrid(monkeypatch):
    monkeypatch.setenv("ARAGORA_ENABLE_IMPLEMENTATION_EXECUTION", "1")

    class DummyResult:
        debate_id = "debate-1"
        task = "Implement a cache"
        final_answer = "Use LRU"
        confidence = 0.9
        consensus_reached = True
        rounds_used = 1
        participants = ["agent-a"]

        def to_dict(self):
            return {
                "debate_id": self.debate_id,
                "task": self.task,
                "final_answer": self.final_answer,
                "confidence": self.confidence,
                "consensus_reached": self.consensus_reached,
                "rounds_used": self.rounds_used,
                "participants": self.participants,
            }

    task = ImplementTask(
        id="task-1",
        description="Add cache layer",
        files=["cache.py"],
        complexity="simple",
    )
    implement_plan = ImplementPlan(design_hash="hash123", tasks=[task])
    package_payload = {"debate_id": "debate-1", "plan": implement_plan.to_dict()}
    package = SimpleNamespace(plan=implement_plan, to_dict=lambda: package_payload)

    monkeypatch.setattr(
        "aragora.pipeline.decision_integrity.build_decision_integrity_package",
        AsyncMock(return_value=package),
    )

    plan = DecisionPlan(
        debate_id="debate-1",
        task="Implement a cache",
        implement_plan=implement_plan,
        approval_mode=ApprovalMode.NEVER,
        status=PlanStatus.APPROVED,
    )
    monkeypatch.setattr(
        "aragora.pipeline.decision_plan.DecisionPlanFactory.from_debate_result",
        lambda *args, **kwargs: plan,
    )

    outcome = PlanOutcome(
        plan_id=plan.id,
        debate_id=plan.debate_id,
        task=plan.task,
        success=True,
    )
    launch = {
        "run_id": "run-di-1",
        "execution_id": "exec-di-1",
        "correlation_id": "corr-di-1",
        "execution_mode": "hybrid",
    }

    with (
        patch(
            "aragora.server.decision_integrity_utils.ensure_decision_plan_backbone_run",
            return_value="run-di-1",
        ),
        patch(
            "aragora.server.decision_integrity_utils.sync_decision_plan_backbone_receipt",
            return_value=True,
        ),
        patch(
            "aragora.server.decision_integrity_utils.execute_decision_plan_with_backbone",
            new=AsyncMock(return_value=(launch, outcome)),
        ) as mock_execute,
    ):
        payload = await build_decision_integrity_payload(
            result=DummyResult(),
            debate_id="debate-1",
            arena=None,
            decision_integrity={
                "include_plan": True,
                "execution_mode": "execute",
                "execution_engine": "hybrid",
                "notify_origin": False,
            },
        )

    assert mock_execute.await_count == 1
    assert payload is not None
    assert payload["execution"]["status"] == "completed"
    assert payload["run_id"] == "run-di-1"
    assert payload["execution"]["run_id"] == "run-di-1"
    assert payload["execution"]["execution_id"] == "exec-di-1"
    assert payload["execution_mode"] == "execute"
    assert payload["execution_engine"] == "hybrid"


@pytest.mark.asyncio
async def test_build_payload_downgrades_interactive_workflow_execute_to_request_approval(
    monkeypatch,
):
    monkeypatch.setenv("ARAGORA_ENABLE_IMPLEMENTATION_EXECUTION", "1")

    class DummyResult:
        debate_id = "debate-2"
        task = "Implement guardrails"
        final_answer = "Add safety checks"
        confidence = 0.9
        consensus_reached = True
        rounds_used = 1
        participants = ["agent-a"]

        def to_dict(self):
            return {
                "debate_id": self.debate_id,
                "task": self.task,
                "final_answer": self.final_answer,
                "confidence": self.confidence,
                "consensus_reached": self.consensus_reached,
                "rounds_used": self.rounds_used,
                "participants": self.participants,
            }

    task = ImplementTask(
        id="task-1",
        description="Add safety checks",
        files=["guardrails.py"],
        complexity="simple",
    )
    implement_plan = ImplementPlan(design_hash="hash456", tasks=[task])
    package_payload = {"debate_id": "debate-2", "plan": implement_plan.to_dict()}
    package = SimpleNamespace(plan=implement_plan, to_dict=lambda: package_payload)

    monkeypatch.setattr(
        "aragora.pipeline.decision_integrity.build_decision_integrity_package",
        AsyncMock(return_value=package),
    )

    plan = DecisionPlan(
        debate_id="debate-2",
        task="Implement guardrails",
        implement_plan=implement_plan,
        approval_mode=ApprovalMode.NEVER,
        status=PlanStatus.APPROVED,
    )
    monkeypatch.setattr(
        "aragora.pipeline.decision_plan.DecisionPlanFactory.from_debate_result",
        lambda *args, **kwargs: plan,
    )

    approval_request = SimpleNamespace(
        id="approval-2",
        title="Approve workflow plan",
        description="Execute workflow plan",
        changes=[],
        risk_level="medium",
        requested_at=datetime.now(timezone.utc),
        requested_by="user-1",
        timeout_seconds=None,
        status=SimpleNamespace(value="pending"),
        approved_by=None,
        approved_at=None,
        rejection_reason=None,
        metadata={},
    )
    arena = SimpleNamespace(
        auth_context=SimpleNamespace(user_id="user-1"),
        continuum_memory=None,
        knowledge_mound=None,
    )

    with (
        patch(
            "aragora.server.decision_integrity_utils.ensure_decision_plan_backbone_run",
            return_value="run-di-2",
        ),
        patch(
            "aragora.server.decision_integrity_utils.sync_decision_plan_backbone_receipt",
            return_value=True,
        ),
        patch(
            "aragora.server.decision_integrity_utils.execute_decision_plan_with_backbone",
            new=AsyncMock(),
        ) as mock_execute,
        patch(
            "aragora.server.handlers.autonomous.approvals.get_approval_flow",
            return_value=SimpleNamespace(request_approval=AsyncMock(return_value=approval_request)),
        ),
    ):
        payload = await build_decision_integrity_payload(
            result=DummyResult(),
            debate_id="debate-2",
            arena=arena,
            decision_integrity={
                "include_plan": True,
                "execution_mode": "execute",
                "execution_engine": "workflow",
                "notify_origin": False,
            },
        )

    assert payload is not None
    assert payload["execution_mode"] == "request_approval"
    assert payload["execution_engine"] == "workflow"
    assert payload["approval"]["id"] == "approval-2"
    assert mock_execute.await_count == 0


@pytest.mark.asyncio
async def test_build_payload_downgrades_interactive_execute_to_request_approval(monkeypatch):
    monkeypatch.setenv("ARAGORA_ENABLE_IMPLEMENTATION_EXECUTION", "1")

    class DummyResult:
        debate_id = "debate-3"
        task = "Implement chat automation"
        final_answer = "Use approval flow"
        confidence = 0.9
        consensus_reached = True
        rounds_used = 1
        participants = ["agent-a"]

        def to_dict(self):
            return {
                "debate_id": self.debate_id,
                "task": self.task,
                "final_answer": self.final_answer,
                "confidence": self.confidence,
                "consensus_reached": self.consensus_reached,
                "rounds_used": self.rounds_used,
                "participants": self.participants,
            }

    task = ImplementTask(
        id="task-1",
        description="Add approval flow",
        files=["chat.py"],
        complexity="simple",
    )
    implement_plan = ImplementPlan(design_hash="hash789", tasks=[task])
    package_payload = {"debate_id": "debate-3", "plan": implement_plan.to_dict()}
    package = SimpleNamespace(plan=implement_plan, to_dict=lambda: package_payload)

    monkeypatch.setattr(
        "aragora.pipeline.decision_integrity.build_decision_integrity_package",
        AsyncMock(return_value=package),
    )

    plan = DecisionPlan(
        debate_id="debate-3",
        task="Implement chat automation",
        implement_plan=implement_plan,
        approval_mode=ApprovalMode.NEVER,
        status=PlanStatus.CREATED,
    )
    monkeypatch.setattr(
        "aragora.pipeline.decision_plan.DecisionPlanFactory.from_debate_result",
        lambda *args, **kwargs: plan,
    )

    approval_request = SimpleNamespace(
        id="approval-1",
        title="Approve plan",
        description="Execute plan",
        changes=[],
        risk_level="medium",
        requested_at=datetime.now(timezone.utc),
        requested_by="user-1",
        timeout_seconds=None,
        status=SimpleNamespace(value="pending"),
        approved_by=None,
        approved_at=None,
        rejection_reason=None,
        metadata={},
    )
    arena = SimpleNamespace(
        auth_context=SimpleNamespace(user_id="user-1"),
        continuum_memory=None,
        knowledge_mound=None,
    )

    with (
        patch(
            "aragora.server.decision_integrity_utils.ensure_decision_plan_backbone_run",
            return_value="run-di-3",
        ),
        patch(
            "aragora.server.decision_integrity_utils.sync_decision_plan_backbone_receipt",
            return_value=True,
        ),
        patch(
            "aragora.server.decision_integrity_utils.execute_decision_plan_with_backbone",
            new=AsyncMock(),
        ) as mock_execute,
        patch(
            "aragora.server.handlers.autonomous.approvals.get_approval_flow",
            return_value=SimpleNamespace(request_approval=AsyncMock(return_value=approval_request)),
        ),
    ):
        payload = await build_decision_integrity_payload(
            result=DummyResult(),
            debate_id="debate-3",
            arena=arena,
            decision_integrity={
                "include_plan": True,
                "execution_mode": "execute",
                "execution_engine": "hybrid",
                "notify_origin": False,
            },
        )

    assert payload is not None
    assert payload["execution_mode"] == "request_approval"
    assert payload["execution_engine"] == "hybrid"
    assert payload["approval"]["id"] == "approval-1"
    assert mock_execute.await_count == 0
