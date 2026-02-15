"""Decision-to-action smoke test.

Covers the full path:
    debate -> plan -> approval -> execution -> completion artifact
"""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from aragora.core_types import DebateResult
from aragora.pipeline.decision_plan import (
    ApprovalMode,
    DecisionPlanFactory,
    PlanOutcome,
    PlanStatus,
)
from aragora.pipeline.execution_bridge import ExecutionBridge
from aragora.pipeline.plan_store import PlanStore


def _artifact_path() -> Path:
    artifact_dir = os.environ.get(
        "DECISION_ACTION_SMOKE_ARTIFACT_DIR",
        "test-results/decision-action-smoke",
    )
    return Path(artifact_dir) / "decision_action_smoke.json"


def _write_artifact(payload: dict) -> Path:
    path = _artifact_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _make_debate_result() -> DebateResult:
    return DebateResult(
        debate_id="debate-smoke-001",
        task="Run deterministic decision-to-action smoke scenario",
        final_answer=(
            "1. Prepare workflow inputs\n"
            "2. Execute implementation tasks\n"
            "3. Verify completion artifact"
        ),
        confidence=0.91,
        consensus_reached=True,
        rounds_used=2,
        rounds_completed=2,
        status="consensus_reached",
        participants=["proposer", "critic", "synthesizer"],
        proposals={"proposer": "Execute the deterministic smoke path"},
        dissenting_views=[],
        metadata={"scenario": "decision-action-smoke"},
    )


@pytest.mark.e2e
@pytest.mark.smoke
@pytest.mark.asyncio
async def test_decision_to_action_smoke(tmp_path: Path) -> None:
    """Deterministic end-to-end decision-to-action smoke scenario."""
    artifact: dict = {
        "scenario": "decision_to_action_smoke",
        "started_at": datetime.utcnow().isoformat(),
        "status": "running",
    }

    try:
        store = PlanStore(db_path=str(tmp_path / "decision_action_smoke.db"))
        result = _make_debate_result()

        artifact["debate"] = {
            "debate_id": result.debate_id,
            "confidence": result.confidence,
            "consensus_reached": result.consensus_reached,
            "rounds_completed": result.rounds_completed,
        }

        plan = DecisionPlanFactory.from_debate_result(result, approval_mode=ApprovalMode.ALWAYS)
        artifact["plan_created"] = {
            "plan_id": plan.id,
            "status": plan.status.value,
            "requires_human_approval": plan.requires_human_approval,
        }
        assert plan.status == PlanStatus.AWAITING_APPROVAL
        assert plan.requires_human_approval is True

        store.create(plan)
        stored_plan = store.get(plan.id)
        assert stored_plan is not None
        assert stored_plan.status == PlanStatus.AWAITING_APPROVAL

        plan.approve("smoke-approver", reason="Deterministic smoke approval")
        updated = store.update_status(plan.id, PlanStatus.APPROVED, approved_by="smoke-approver")
        assert updated is True

        approved_plan = store.get(plan.id)
        assert approved_plan is not None
        assert approved_plan.status == PlanStatus.APPROVED
        artifact["plan_approved"] = {
            "status": approved_plan.status.value,
            "approved_by": "smoke-approver",
        }

        mock_executor = AsyncMock()
        mock_executor.execute.return_value = PlanOutcome(
            plan_id=plan.id,
            debate_id=plan.debate_id,
            task=plan.task,
            success=True,
            tasks_completed=2,
            tasks_total=2,
            duration_seconds=1.0,
            lessons=["deterministic smoke execution"],
        )

        bridge = ExecutionBridge(plan_store=store, executor=mock_executor)
        outcome = await bridge.execute_approved_plan(plan.id, execution_mode="workflow")

        final_plan = store.get(plan.id)
        assert final_plan is not None
        assert final_plan.status == PlanStatus.COMPLETED
        assert outcome.success is True
        assert outcome.tasks_completed == 2
        assert outcome.tasks_total == 2

        execution_records = bridge.list_execution_records(plan_id=plan.id)
        assert len(execution_records) == 1
        assert execution_records[0]["status"] == "succeeded"

        artifact["outcome"] = outcome.to_dict()
        artifact["final_plan_status"] = final_plan.status.value
        artifact["execution_record"] = execution_records[0]
        artifact["status"] = "passed"
        artifact["completed_at"] = datetime.utcnow().isoformat()

        artifact_path = _write_artifact(artifact)
        assert artifact_path.exists()

    except Exception as exc:  # pragma: no cover - explicit smoke failure path
        artifact["status"] = "failed"
        artifact["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        artifact["failed_at"] = datetime.utcnow().isoformat()
        artifact_path = _write_artifact(artifact)
        pytest.fail(
            f"Decision-to-action smoke scenario failed. Inspect artifact: {artifact_path}",
            pytrace=False,
        )
