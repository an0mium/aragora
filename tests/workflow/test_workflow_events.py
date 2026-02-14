"""Tests for workflow event callbacks and webhook dispatch bridge."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.events.schema import (
    EVENT_SCHEMAS,
    REQUIRED_FIELDS,
    WorkflowCompletedPayload,
    WorkflowFailedPayload,
    WorkflowStartPayload,
    WorkflowStepCompletedPayload,
)
from aragora.events.types import StreamEventType
from aragora.workflow.engine import WorkflowEngine
from aragora.workflow.step import BaseStep, WorkflowContext
from aragora.workflow.types import StepDefinition, WorkflowConfig, WorkflowDefinition


class NoopStep(BaseStep):
    """Simple step for event callback testing."""

    async def execute(self, context: WorkflowContext) -> dict:
        return {"ok": True}


# =========================================================================
# Existing callback tests
# =========================================================================


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


# =========================================================================
# Schema registration tests
# =========================================================================


class TestWorkflowEventSchemas:
    """Test that workflow event payloads are registered."""

    def test_workflow_start_schema_registered(self):
        """WORKFLOW_START has a schema in EVENT_SCHEMAS."""
        assert StreamEventType.WORKFLOW_START in EVENT_SCHEMAS
        assert EVENT_SCHEMAS[StreamEventType.WORKFLOW_START] is WorkflowStartPayload

    def test_workflow_step_complete_schema_registered(self):
        """WORKFLOW_STEP_COMPLETE has a schema in EVENT_SCHEMAS."""
        assert StreamEventType.WORKFLOW_STEP_COMPLETE in EVENT_SCHEMAS
        assert EVENT_SCHEMAS[StreamEventType.WORKFLOW_STEP_COMPLETE] is WorkflowStepCompletedPayload

    def test_workflow_complete_schema_registered(self):
        """WORKFLOW_COMPLETE has a schema in EVENT_SCHEMAS."""
        assert StreamEventType.WORKFLOW_COMPLETE in EVENT_SCHEMAS
        assert EVENT_SCHEMAS[StreamEventType.WORKFLOW_COMPLETE] is WorkflowCompletedPayload

    def test_workflow_failed_schema_registered(self):
        """WORKFLOW_FAILED has a schema in EVENT_SCHEMAS."""
        assert StreamEventType.WORKFLOW_FAILED in EVENT_SCHEMAS
        assert EVENT_SCHEMAS[StreamEventType.WORKFLOW_FAILED] is WorkflowFailedPayload

    def test_required_fields_for_start(self):
        """WorkflowStartPayload has required fields."""
        assert WorkflowStartPayload in REQUIRED_FIELDS
        required = REQUIRED_FIELDS[WorkflowStartPayload]
        assert "workflow_id" in required
        assert "definition_id" in required

    def test_required_fields_for_step_complete(self):
        """WorkflowStepCompletedPayload has required fields."""
        assert WorkflowStepCompletedPayload in REQUIRED_FIELDS
        required = REQUIRED_FIELDS[WorkflowStepCompletedPayload]
        assert "workflow_id" in required
        assert "step_id" in required

    def test_required_fields_for_complete(self):
        """WorkflowCompletedPayload has required fields."""
        assert WorkflowCompletedPayload in REQUIRED_FIELDS
        required = REQUIRED_FIELDS[WorkflowCompletedPayload]
        assert "workflow_id" in required

    def test_required_fields_for_failed(self):
        """WorkflowFailedPayload has required fields."""
        assert WorkflowFailedPayload in REQUIRED_FIELDS
        required = REQUIRED_FIELDS[WorkflowFailedPayload]
        assert "workflow_id" in required


# =========================================================================
# Payload serialization tests
# =========================================================================


class TestWorkflowPayloads:
    """Test workflow event payload creation and serialization."""

    def test_start_payload_to_dict(self):
        """WorkflowStartPayload serializes correctly."""
        payload = WorkflowStartPayload(
            workflow_id="wf_abc123",
            definition_id="def_xyz",
            step_count=5,
            workflow_name="My Workflow",
        )
        d = payload.to_dict()
        assert d["workflow_id"] == "wf_abc123"
        assert d["definition_id"] == "def_xyz"
        assert d["step_count"] == 5
        assert d["workflow_name"] == "My Workflow"

    def test_step_completed_payload_to_dict(self):
        """WorkflowStepCompletedPayload serializes correctly."""
        payload = WorkflowStepCompletedPayload(
            workflow_id="wf_abc123",
            definition_id="def_xyz",
            step_id="step_1",
            step_name="Run analysis",
            step_type="agent",
            duration_ms=150.5,
        )
        d = payload.to_dict()
        assert d["step_id"] == "step_1"
        assert d["step_name"] == "Run analysis"
        assert d["duration_ms"] == 150.5

    def test_completed_payload_to_dict(self):
        """WorkflowCompletedPayload serializes correctly."""
        payload = WorkflowCompletedPayload(
            workflow_id="wf_abc123",
            definition_id="def_xyz",
            success=True,
            duration_ms=500.0,
            steps_executed=3,
        )
        d = payload.to_dict()
        assert d["success"] is True
        assert d["steps_executed"] == 3

    def test_failed_payload_to_dict(self):
        """WorkflowFailedPayload serializes correctly."""
        payload = WorkflowFailedPayload(
            workflow_id="wf_abc123",
            definition_id="def_xyz",
            error="Timed out",
            steps_executed=2,
        )
        d = payload.to_dict()
        assert d["success"] is False
        assert d["error"] == "Timed out"

    def test_start_payload_from_dict(self):
        """WorkflowStartPayload can be created from dict."""
        data = {
            "workflow_id": "wf_test",
            "definition_id": "def_test",
            "step_count": 3,
            "extra_field": "ignored",
        }
        payload = WorkflowStartPayload.from_dict(data)
        assert payload.workflow_id == "wf_test"
        assert payload.step_count == 3


# =========================================================================
# Engine dispatch bridge tests
# =========================================================================


class TestWorkflowEngineDispatch:
    """Test that WorkflowEngine dispatches events to the webhook system."""

    @pytest.fixture
    def engine(self):
        """Create a WorkflowEngine with no checkpoint store."""
        config = WorkflowConfig(enable_checkpointing=False)
        return WorkflowEngine(config=config)

    def test_dispatch_workflow_start(self, engine):
        """workflow_start events are dispatched to event system."""
        ctx = WorkflowContext(workflow_id="wf_test", definition_id="def_test")

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            engine._emit_event(
                ctx,
                StreamEventType.WORKFLOW_START,
                {"workflow_id": "wf_test", "step_count": 3},
            )
            mock_dispatch.assert_called_once_with(
                "workflow_start",
                {"workflow_id": "wf_test", "step_count": 3},
            )

    def test_dispatch_workflow_complete(self, engine):
        """workflow_complete events are dispatched to event system."""
        ctx = WorkflowContext(workflow_id="wf_test", definition_id="def_test")

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            engine._emit_event(
                ctx,
                StreamEventType.WORKFLOW_COMPLETE,
                {"workflow_id": "wf_test", "success": True},
            )
            mock_dispatch.assert_called_once_with(
                "workflow_complete",
                {"workflow_id": "wf_test", "success": True},
            )

    def test_dispatch_workflow_failed(self, engine):
        """workflow_failed events are dispatched to event system."""
        ctx = WorkflowContext(workflow_id="wf_test", definition_id="def_test")

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            engine._emit_event(
                ctx,
                StreamEventType.WORKFLOW_FAILED,
                {"workflow_id": "wf_test", "error": "boom"},
            )
            mock_dispatch.assert_called_once_with(
                "workflow_failed",
                {"workflow_id": "wf_test", "error": "boom"},
            )

    def test_dispatch_step_complete(self, engine):
        """workflow_step_complete events are dispatched to event system."""
        ctx = WorkflowContext(workflow_id="wf_test", definition_id="def_test")

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            engine._emit_event(
                ctx,
                StreamEventType.WORKFLOW_STEP_COMPLETE,
                {"workflow_id": "wf_test", "step_id": "s1"},
            )
            mock_dispatch.assert_called_once_with(
                "workflow_step_complete",
                {"workflow_id": "wf_test", "step_id": "s1"},
            )

    def test_non_lifecycle_events_not_dispatched(self, engine):
        """Non-lifecycle events like step_start are not dispatched to webhooks."""
        ctx = WorkflowContext(workflow_id="wf_test", definition_id="def_test")

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            engine._emit_event(
                ctx,
                StreamEventType.WORKFLOW_STEP_START,
                {"workflow_id": "wf_test", "step_id": "s1"},
            )
            mock_dispatch.assert_not_called()

    def test_transition_events_not_dispatched(self, engine):
        """Transition events are not dispatched to webhooks."""
        ctx = WorkflowContext(workflow_id="wf_test", definition_id="def_test")

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            engine._emit_event(
                ctx,
                StreamEventType.WORKFLOW_TRANSITION,
                {"from_step": "s1", "to_step": "s2"},
            )
            mock_dispatch.assert_not_called()

    def test_checkpoint_events_not_dispatched(self, engine):
        """Checkpoint events are not dispatched to webhooks."""
        ctx = WorkflowContext(workflow_id="wf_test", definition_id="def_test")

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            engine._emit_event(
                ctx,
                StreamEventType.WORKFLOW_CHECKPOINT,
                {"checkpoint_id": "cp_1"},
            )
            mock_dispatch.assert_not_called()

    def test_dispatch_graceful_on_import_error(self, engine):
        """Dispatch gracefully degrades when dispatcher is not available."""
        ctx = WorkflowContext(workflow_id="wf_test", definition_id="def_test")

        with patch.dict("sys.modules", {"aragora.events.dispatcher": None}):
            # Should not raise
            engine._emit_event(
                ctx,
                StreamEventType.WORKFLOW_START,
                {"workflow_id": "wf_test"},
            )

    def test_dispatch_graceful_on_dispatch_error(self, engine):
        """Dispatch gracefully degrades when dispatch_event raises."""
        ctx = WorkflowContext(workflow_id="wf_test", definition_id="def_test")

        with patch(
            "aragora.events.dispatcher.dispatch_event",
            side_effect=RuntimeError("webhook down"),
        ):
            # Should not raise
            engine._emit_event(
                ctx,
                StreamEventType.WORKFLOW_COMPLETE,
                {"workflow_id": "wf_test"},
            )

    def test_event_callback_still_called(self, engine):
        """Event callback is still called alongside webhook dispatch."""
        callback = MagicMock()
        ctx = WorkflowContext(
            workflow_id="wf_test",
            definition_id="def_test",
            event_callback=callback,
        )

        with patch("aragora.events.dispatcher.dispatch_event") as mock_dispatch:
            engine._emit_event(
                ctx,
                StreamEventType.WORKFLOW_START,
                {"workflow_id": "wf_test"},
            )
            callback.assert_called_once_with("workflow_start", {"workflow_id": "wf_test"})
            mock_dispatch.assert_called_once()
