import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aragora.implement.types import ImplementPlan, ImplementTask
from aragora.pipeline.decision_plan import ApprovalMode, DecisionPlan, PlanStatus
from aragora.pipeline.decision_plan.memory import PlanOutcome
from aragora.server.decision_integrity_utils import (
    build_decision_integrity_payload,
    extract_execution_overrides,
)


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
