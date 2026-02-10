"""Tests for workflow event callbacks."""

import pytest

from aragora.workflow.engine import WorkflowEngine
from aragora.workflow.step import BaseStep, WorkflowContext
from aragora.workflow.types import StepDefinition, WorkflowConfig, WorkflowDefinition


class NoopStep(BaseStep):
    """Simple step for event callback testing."""

    async def execute(self, context: WorkflowContext) -> dict:
        return {"ok": True}


@pytest.mark.asyncio
async def test_workflow_event_callback_emits_events():
    events: list[tuple[str, dict]] = []

    def callback(event_type: str, payload: dict) -> None:
        events.append((event_type, payload))

    engine = WorkflowEngine(
        config=WorkflowConfig(enable_checkpointing=False),
        step_registry={"noop": NoopStep},
    )

    workflow = WorkflowDefinition(
        id="wf_events",
        name="Event Test Workflow",
        steps=[
            StepDefinition(id="step1", name="Step 1", step_type="noop", next_steps=["step2"]),
            StepDefinition(id="step2", name="Step 2", step_type="noop"),
        ],
    )

    result = await engine.execute(
        workflow,
        inputs={"example": "value"},
        workflow_id="exec_test_123",
        event_callback=callback,
    )

    assert result.success is True

    event_types = [event[0] for event in events]
    assert "workflow_start" in event_types
    assert "workflow_step_start" in event_types
    assert "workflow_step_complete" in event_types
    assert "workflow_transition" in event_types
    assert "workflow_complete" in event_types

    # Payloads should include workflow identifiers
    payload = events[0][1]
    assert payload.get("workflow_id") == "exec_test_123"
    assert payload.get("definition_id") == "wf_events"
