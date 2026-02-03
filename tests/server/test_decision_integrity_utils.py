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

    captured: dict[str, object] = {}

    async def _execute(*_args, **kwargs):
        captured.update(kwargs)
        return outcome

    with (
        patch(
            "aragora.pipeline.executor.PlanExecutor.execute",
            new=AsyncMock(side_effect=_execute),
        ) as mock_execute,
        patch("aragora.pipeline.execution_notifier.ExecutionNotifier") as mock_notifier,
    ):
        mock_notifier.return_value.on_task_complete = object()
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

    assert mock_execute.called is True
    assert captured.get("on_task_complete") is not None
    assert payload is not None
    assert payload["execution"]["status"] == "completed"
    assert payload["execution_mode"] == "execute"
    assert payload["execution_engine"] == "hybrid"
