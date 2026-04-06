from __future__ import annotations

from pathlib import Path

from aragora.pipeline.backbone_contracts import BackboneStage, RunLedger
from aragora.pipeline.backbone_runtime import BackboneRuntime
from aragora.pipeline.plan_store import PlanStore


def test_backbone_runtime_create_append_and_get(tmp_path: Path) -> None:
    runtime = BackboneRuntime(PlanStore(db_path=str(tmp_path / "backbone_lifecycle.db")))
    runtime.create_run(RunLedger(run_id="run-123", entrypoint="prompt_engine.run"))

    assert runtime.append_stage_event(
        "run-123", BackboneStage.INTAKE, status="completed", artifact_ref="intake-1"
    )
    assert runtime.append_stage_event(
        "run-123",
        BackboneStage.SPECIFICATION,
        status="completed",
        artifact_ref="spec-1",
        details={"source": "test"},
    )

    run = runtime.get_run("run-123")

    assert run is not None
    assert run.run_id == "run-123"
    assert [event.stage for event in run.stage_events] == [
        BackboneStage.INTAKE.value,
        BackboneStage.SPECIFICATION.value,
    ]
    assert run.stage_events[1].artifact_ref == "spec-1"
    assert run.stage_events[1].details == {"source": "test"}
