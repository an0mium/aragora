"""Tests for receipt enforcement gate wired into OpenClaw execute action.

Covers:
- Action succeeds with valid receipt when enforcement is enabled
- Action fails without receipt_id when enforcement is enabled
- Action proceeds normally when enforcement flag is off
- Receipt transitions to EXECUTED after successful action
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mixin():
    """Create a SessionOrchestrationMixin with mocked dependencies."""
    from aragora.server.handlers.openclaw.orchestrator import SessionOrchestrationMixin

    mixin = SessionOrchestrationMixin.__new__(SessionOrchestrationMixin)
    return mixin


def _mock_handler(user_id: str = "user-1"):
    """Create a mock handler object."""
    handler = MagicMock()
    handler.request = MagicMock()
    handler.headers = {}
    return handler


def _base_body(receipt_id: str | None = None) -> dict:
    """Minimal valid body for execute action."""
    body = {
        "session_id": "session-1",
        "action_type": "code.execute",
        "input": {"code": "print('ok')"},
        "metadata": {},
    }
    if receipt_id is not None:
        body["receipt_id"] = receipt_id
    return body


def _mock_store():
    """Create a mock store with a valid active session."""
    from aragora.server.handlers.openclaw.models import SessionStatus

    store = MagicMock()
    session = MagicMock()
    session.id = "session-1"
    session.user_id = "user-1"
    session.tenant_id = "tenant-1"
    session.status = SessionStatus.ACTIVE
    session.config = {"timeout": 30}
    store.get_session.return_value = session

    action = MagicMock()
    action.id = "action-1"
    action.action_type = "code.execute"
    action.session_id = "session-1"
    action.input_data = {"code": "print('ok')"}
    action.metadata = {}
    action.to_dict.return_value = {
        "id": "action-1",
        "session_id": "session-1",
        "action_type": "code.execute",
        "status": "completed",
    }
    store.create_action.return_value = action
    store.get_action.return_value = action

    return store


def _runtime_result(*, status, executed: bool, output_data=None, error=None, execution_time_ms=7):
    from aragora.server.handlers.openclaw.runtime import RuntimeDispatchResult

    return RuntimeDispatchResult(
        action_id="action-1",
        status=status,
        executed=executed,
        output_data=output_data,
        error=error,
        execution_time_ms=execution_time_ms,
        audit_result="success" if status.value == "completed" else "failed",
        audit_details={},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOpenClawReceiptGate:
    """Receipt enforcement gate tests for OpenClaw execute action."""

    @patch("aragora.server.handlers.openclaw.orchestrator._get_store")
    @patch("aragora.server.handlers.openclaw.orchestrator.get_openclaw_execution_runtime")
    @patch("aragora.pipeline.receipt_enforcement.is_receipt_enforcement_enabled", return_value=True)
    @patch("aragora.pipeline.receipt_enforcement.require_receipt_gate")
    @patch("aragora.pipeline.receipt_enforcement.transition_receipt_executed")
    @patch("aragora.server.handlers.openclaw.orchestrator.emit_operational_receipt")
    def test_action_succeeds_with_valid_receipt(
        self,
        mock_emit,
        mock_transition,
        mock_gate,
        mock_enabled,
        mock_runtime,
        mock_get_store,
    ):
        """When enforcement is on and a valid receipt_id is provided, action proceeds."""
        from aragora.server.handlers.openclaw.models import ActionStatus

        store = _mock_store()
        mock_get_store.return_value = store
        mock_gate.return_value = MagicMock()  # valid stored receipt
        mock_emit.return_value = "op-123"
        runtime = MagicMock()
        runtime.dispatch_action.return_value = _runtime_result(
            status=ActionStatus.COMPLETED,
            executed=True,
            output_data={"runtime": "openclaw_action_sandbox", "result": "ok"},
        )
        mock_runtime.return_value = runtime

        mixin = _make_mixin()
        mixin._get_user_id = MagicMock(return_value="user-1")
        mixin._get_tenant_id = MagicMock(return_value="tenant-1")
        mixin.get_current_user = MagicMock(return_value=None)

        body = _base_body(receipt_id="receipt-123")
        result = mixin._handle_execute_action(body, _mock_handler())

        assert result.status == 202
        mock_gate.assert_called_once_with(
            action_domain="openclaw",
            action_type="execute_action",
            actor_id="user-1",
            resource_id="session-1",
            receipt_id="receipt-123",
        )
        mock_transition.assert_called_once_with("receipt-123")
        assert mock_emit.call_args.kwargs["metadata"]["decision_receipt_id"] == "receipt-123"
        assert mock_emit.call_args.kwargs["verdict"] == "success"

    @patch("aragora.server.handlers.openclaw.orchestrator._get_store")
    @patch("aragora.pipeline.receipt_enforcement.is_receipt_enforcement_enabled", return_value=True)
    @patch(
        "aragora.pipeline.receipt_enforcement.require_receipt_gate",
        side_effect=__import__(
            "aragora.pipeline.receipt_enforcement", fromlist=["ReceiptEnforcementError"]
        ).ReceiptEnforcementError(
            "Receipt required",
            action_domain="openclaw",
            action_type="execute_action",
        ),
    )
    def test_action_fails_without_receipt(self, mock_gate, mock_enabled, mock_get_store):
        """When enforcement is on and no receipt_id is provided, returns 428."""
        store = _mock_store()
        mock_get_store.return_value = store

        mixin = _make_mixin()
        mixin._get_user_id = MagicMock(return_value="user-1")
        mixin._get_tenant_id = MagicMock(return_value="tenant-1")
        mixin.get_current_user = MagicMock(return_value=None)

        body = _base_body()  # no receipt_id
        result = mixin._handle_execute_action(body, _mock_handler())

        assert result.status == 428

    @patch("aragora.server.handlers.openclaw.orchestrator._get_store")
    @patch("aragora.server.handlers.openclaw.orchestrator.get_openclaw_execution_runtime")
    @patch(
        "aragora.pipeline.receipt_enforcement.is_receipt_enforcement_enabled", return_value=False
    )
    @patch("aragora.server.handlers.openclaw.orchestrator.emit_operational_receipt")
    def test_action_proceeds_without_receipt_flag_off(
        self, mock_emit, mock_enabled, mock_runtime, mock_get_store
    ):
        """When enforcement flag is off, action proceeds without receipt."""
        from aragora.server.handlers.openclaw.models import ActionStatus

        store = _mock_store()
        mock_get_store.return_value = store
        runtime = MagicMock()
        runtime.dispatch_action.return_value = _runtime_result(
            status=ActionStatus.COMPLETED,
            executed=True,
            output_data={"runtime": "openclaw_action_sandbox", "result": "ok"},
        )
        mock_runtime.return_value = runtime

        mixin = _make_mixin()
        mixin._get_user_id = MagicMock(return_value="user-1")
        mixin._get_tenant_id = MagicMock(return_value="tenant-1")
        mixin.get_current_user = MagicMock(return_value=None)

        body = _base_body()  # no receipt_id
        result = mixin._handle_execute_action(body, _mock_handler())

        assert result.status == 202
        store.create_action.assert_called_once()

    @patch("aragora.server.handlers.openclaw.orchestrator._get_store")
    @patch("aragora.server.handlers.openclaw.orchestrator.get_openclaw_execution_runtime")
    @patch("aragora.pipeline.receipt_enforcement.is_receipt_enforcement_enabled", return_value=True)
    @patch("aragora.pipeline.receipt_enforcement.require_receipt_gate")
    @patch("aragora.pipeline.receipt_enforcement.transition_receipt_executed")
    @patch("aragora.server.handlers.openclaw.orchestrator.emit_operational_receipt")
    def test_receipt_transitions_to_executed(
        self,
        mock_emit,
        mock_transition,
        mock_gate,
        mock_enabled,
        mock_runtime,
        mock_get_store,
    ):
        """After successful action, receipt state transitions to EXECUTED."""
        from aragora.server.handlers.openclaw.models import ActionStatus

        store = _mock_store()
        mock_get_store.return_value = store
        mock_gate.return_value = MagicMock()
        mock_emit.return_value = "op-456"
        runtime = MagicMock()
        runtime.dispatch_action.return_value = _runtime_result(
            status=ActionStatus.COMPLETED,
            executed=True,
            output_data={"runtime": "openclaw_action_sandbox", "result": "ok"},
        )
        mock_runtime.return_value = runtime

        mixin = _make_mixin()
        mixin._get_user_id = MagicMock(return_value="user-1")
        mixin._get_tenant_id = MagicMock(return_value="tenant-1")
        mixin.get_current_user = MagicMock(return_value=None)

        body = _base_body(receipt_id="receipt-456")
        result = mixin._handle_execute_action(body, _mock_handler())

        assert result.status == 202
        mock_transition.assert_called_once_with("receipt-456")
        assert mock_emit.call_args.kwargs["metadata"]["decision_receipt_id"] == "receipt-456"

    @patch("aragora.server.handlers.openclaw.orchestrator._get_store")
    @patch("aragora.server.handlers.openclaw.orchestrator.get_openclaw_execution_runtime")
    @patch("aragora.pipeline.receipt_enforcement.is_receipt_enforcement_enabled", return_value=True)
    @patch("aragora.pipeline.receipt_enforcement.require_receipt_gate")
    @patch("aragora.pipeline.receipt_enforcement.transition_receipt_executed")
    @patch("aragora.server.handlers.openclaw.orchestrator.emit_operational_receipt")
    def test_failed_runtime_keeps_decision_receipt_unexecuted(
        self,
        mock_emit,
        mock_transition,
        mock_gate,
        mock_enabled,
        mock_runtime,
        mock_get_store,
    ):
        """Runtime failure should emit a linked receipt without claiming receipt execution."""
        from aragora.server.handlers.openclaw.models import ActionStatus

        store = _mock_store()
        mock_get_store.return_value = store
        mock_gate.return_value = MagicMock()
        mock_emit.return_value = "op-fail-1"
        runtime = MagicMock()
        runtime.dispatch_action.return_value = _runtime_result(
            status=ActionStatus.FAILED,
            executed=True,
            output_data={
                "runtime": "openclaw_action_sandbox",
                "failure_reason": "runtime boom",
            },
            error="runtime boom",
        )
        mock_runtime.return_value = runtime

        mixin = _make_mixin()
        mixin._get_user_id = MagicMock(return_value="user-1")
        mixin._get_tenant_id = MagicMock(return_value="tenant-1")
        mixin.get_current_user = MagicMock(return_value=None)

        body = _base_body(receipt_id="receipt-fail-1")
        result = mixin._handle_execute_action(body, _mock_handler())

        assert result.status == 202
        mock_transition.assert_not_called()
        assert mock_emit.call_args.kwargs["verdict"] == "failure"
        assert mock_emit.call_args.kwargs["metadata"]["decision_receipt_id"] == "receipt-fail-1"
