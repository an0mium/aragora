"""
Tests for Human Checkpoint Step for workflow approval gates.

Tests cover:
1. Basic approval flow (approve/reject)
2. Checklist item completion
3. Timeout handling and escalation
4. State persistence
5. Error handling
6. Auto-approval conditions
7. Helper functions (resolve_approval, get_pending_approvals, etc.)
8. ApprovalRequest and ChecklistItem dataclasses
9. Notification callbacks
10. GovernanceStore integration
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clear_pending_approvals():
    """Clear pending approvals before each test for isolation."""
    from aragora.workflow.nodes.human_checkpoint import _pending_approvals

    _pending_approvals.clear()
    yield
    _pending_approvals.clear()


# ============================================================================
# ApprovalStatus Tests
# ============================================================================


class TestApprovalStatus:
    """Tests for ApprovalStatus enum."""

    def test_status_values(self):
        """Test all expected status values exist."""
        from aragora.workflow.nodes.human_checkpoint import ApprovalStatus

        assert ApprovalStatus.PENDING.value == "pending"
        assert ApprovalStatus.APPROVED.value == "approved"
        assert ApprovalStatus.REJECTED.value == "rejected"
        assert ApprovalStatus.ESCALATED.value == "escalated"
        assert ApprovalStatus.TIMEOUT.value == "timeout"


# ============================================================================
# ChecklistItem Tests
# ============================================================================


class TestChecklistItem:
    """Tests for ChecklistItem dataclass."""

    def test_checklist_item_defaults(self):
        """Test default values for ChecklistItem."""
        from aragora.workflow.nodes.human_checkpoint import ChecklistItem

        item = ChecklistItem(id="item_1", label="Review document")
        assert item.id == "item_1"
        assert item.label == "Review document"
        assert item.required is True
        assert item.checked is False
        assert item.notes == ""

    def test_checklist_item_custom_values(self):
        """Test ChecklistItem with custom values."""
        from aragora.workflow.nodes.human_checkpoint import ChecklistItem

        item = ChecklistItem(
            id="item_2",
            label="Optional review",
            required=False,
            checked=True,
            notes="Looks good",
        )
        assert item.required is False
        assert item.checked is True
        assert item.notes == "Looks good"


# ============================================================================
# ApprovalRequest Tests
# ============================================================================


class TestApprovalRequest:
    """Tests for ApprovalRequest dataclass."""

    def test_approval_request_defaults(self):
        """Test default values for ApprovalRequest."""
        from aragora.workflow.nodes.human_checkpoint import ApprovalRequest, ApprovalStatus

        request = ApprovalRequest(
            id="apr_123",
            workflow_id="wf_456",
            step_id="step_legal",
            title="Legal Review",
            description="Please review contract",
            checklist=[],
        )
        assert request.status == ApprovalStatus.PENDING
        assert request.responded_at is None
        assert request.responder_id is None
        assert request.responder_notes == ""
        assert request.timeout_seconds == 3600.0
        assert request.escalation_emails == []
        assert request.created_at is not None

    def test_approval_request_to_dict(self):
        """Test ApprovalRequest.to_dict serialization."""
        from aragora.workflow.nodes.human_checkpoint import (
            ApprovalRequest,
            ApprovalStatus,
            ChecklistItem,
        )

        request = ApprovalRequest(
            id="apr_abc",
            workflow_id="wf_xyz",
            step_id="review",
            title="Review Required",
            description="Check the PR",
            checklist=[
                ChecklistItem(id="c1", label="Code quality", required=True, checked=True),
                ChecklistItem(id="c2", label="Tests pass", required=True, checked=False),
            ],
            status=ApprovalStatus.APPROVED,
            responder_id="user_123",
            responder_notes="Approved with minor comments",
            timeout_seconds=7200.0,
            escalation_emails=["lead@example.com"],
        )
        request.responded_at = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        d = request.to_dict()
        assert d["id"] == "apr_abc"
        assert d["workflow_id"] == "wf_xyz"
        assert d["step_id"] == "review"
        assert d["title"] == "Review Required"
        assert d["status"] == "approved"
        assert d["responder_id"] == "user_123"
        assert d["responder_notes"] == "Approved with minor comments"
        assert d["timeout_seconds"] == 7200.0
        assert d["escalation_emails"] == ["lead@example.com"]
        assert len(d["checklist"]) == 2
        assert d["checklist"][0]["id"] == "c1"
        assert d["checklist"][0]["checked"] is True
        assert d["responded_at"] is not None


# ============================================================================
# HumanCheckpointStep Initialization Tests
# ============================================================================


class TestHumanCheckpointStepInit:
    """Tests for HumanCheckpointStep initialization."""

    def test_basic_init(self):
        """Test basic HumanCheckpointStep initialization."""
        from aragora.workflow.nodes.human_checkpoint import HumanCheckpointStep

        step = HumanCheckpointStep(
            name="Legal Review",
            config={
                "title": "Legal Review Required",
                "description": "Please review the contract terms",
            },
        )
        assert step.name == "Legal Review"
        assert step.config["title"] == "Legal Review Required"

    def test_default_config(self):
        """Test HumanCheckpointStep with no config."""
        from aragora.workflow.nodes.human_checkpoint import HumanCheckpointStep

        step = HumanCheckpointStep(name="Empty Checkpoint")
        assert step.config == {}

    def test_init_with_checklist(self):
        """Test HumanCheckpointStep with checklist config."""
        from aragora.workflow.nodes.human_checkpoint import HumanCheckpointStep

        step = HumanCheckpointStep(
            name="Review",
            config={
                "checklist": [
                    {"label": "Verified compliance terms", "required": True},
                    {"label": "Checked formatting", "required": False},
                ],
            },
        )
        assert len(step.config["checklist"]) == 2


# ============================================================================
# Basic Approval Flow Tests
# ============================================================================


class TestBasicApprovalFlow:
    """Tests for basic approval and rejection flows."""

    def _make_context(self, inputs=None, state=None, step_outputs=None):
        from aragora.workflow.step import WorkflowContext

        return WorkflowContext(
            workflow_id="wf_test",
            definition_id="def_test",
            inputs=inputs or {},
            state=state or {},
            step_outputs=step_outputs or {},
            current_step_id="test_checkpoint",
        )

    @pytest.fixture(autouse=True)
    def cleanup_approvals(self):
        """Clean up pending approvals before and after each test."""
        from aragora.workflow.nodes.human_checkpoint import (
            clear_pending_approvals,
            reset_approval_recovery,
        )

        clear_pending_approvals()
        reset_approval_recovery()
        yield
        clear_pending_approvals()
        reset_approval_recovery()

    @pytest.mark.asyncio
    async def test_approval_creates_request(self):
        """Test that executing checkpoint creates an approval request."""
        from aragora.workflow.nodes.human_checkpoint import (
            HumanCheckpointStep,
            ApprovalStatus,
            get_pending_approvals,
            resolve_approval,
            _pending_approvals,
        )

        step = HumanCheckpointStep(
            name="Test Checkpoint",
            config={
                "title": "Test Approval",
                "description": "Test description",
                "timeout_seconds": 5.0,  # Longer timeout for test stability
            },
        )
        ctx = self._make_context()

        # Start execution in background
        async def execute_and_approve():
            # Wait a bit for request to be created, poll until we find it
            for _ in range(50):  # Up to 2.5 seconds
                await asyncio.sleep(0.05)
                pending = get_pending_approvals()
                if pending:
                    resolve_approval(pending[0].id, ApprovalStatus.APPROVED, "tester")
                    return
            raise AssertionError("No pending approval found")

        task = asyncio.create_task(step.execute(ctx))
        approve_task = asyncio.create_task(execute_and_approve())

        result = await task
        await approve_task

        assert result["status"] == "approved"

    @pytest.mark.asyncio
    async def test_rejection_flow(self):
        """Test rejection flow returns rejected status."""
        from aragora.workflow.nodes.human_checkpoint import (
            HumanCheckpointStep,
            ApprovalStatus,
            get_pending_approvals,
            resolve_approval,
        )

        step = HumanCheckpointStep(
            name="Reject Test",
            config={
                "title": "Will be rejected",
                "timeout_seconds": 5.0,
            },
        )
        ctx = self._make_context()

        async def execute_and_reject():
            for _ in range(50):
                await asyncio.sleep(0.05)
                pending = get_pending_approvals()
                if pending:
                    resolve_approval(
                        pending[0].id,
                        ApprovalStatus.REJECTED,
                        "reviewer",
                        notes="Not acceptable",
                    )
                    return
            raise AssertionError("No pending approval found")

        task = asyncio.create_task(step.execute(ctx))
        reject_task = asyncio.create_task(execute_and_reject())

        result = await task
        await reject_task

        assert result["status"] == "rejected"
        assert result["responder_id"] == "reviewer"
        assert result["responder_notes"] == "Not acceptable"

    @pytest.mark.asyncio
    async def test_approval_stores_request_in_memory(self):
        """Test that approval request is stored in memory."""
        from aragora.workflow.nodes.human_checkpoint import (
            HumanCheckpointStep,
            ApprovalStatus,
            get_approval_request,
            get_pending_approvals,
            resolve_approval,
            _pending_approvals,
        )

        step = HumanCheckpointStep(
            name="Store Test",
            config={
                "title": "Stored Request",
                "timeout_seconds": 5.0,
            },
        )
        ctx = self._make_context()

        request_id = None

        async def capture_and_approve():
            nonlocal request_id
            for _ in range(50):
                await asyncio.sleep(0.05)
                pending = get_pending_approvals()
                if pending:
                    request_id = pending[0].id
                    # Verify it's in memory
                    assert request_id in _pending_approvals
                    resolve_approval(request_id, ApprovalStatus.APPROVED, "tester")
                    return
            raise AssertionError("No pending approval found")

        task = asyncio.create_task(step.execute(ctx))
        approve_task = asyncio.create_task(capture_and_approve())

        await task
        await approve_task

        assert request_id is not None
        # Request should still be in memory after resolution
        assert get_approval_request(request_id) is not None

    @pytest.mark.asyncio
    async def test_approval_stores_request_id_in_context(self):
        """Test that approval request ID is stored in context state."""
        from aragora.workflow.nodes.human_checkpoint import (
            HumanCheckpointStep,
            ApprovalStatus,
            get_pending_approvals,
            resolve_approval,
        )

        step = HumanCheckpointStep(
            name="Context Test",
            config={
                "title": "Context Request",
                "timeout_seconds": 5.0,
            },
        )
        ctx = self._make_context()

        async def approve_quick():
            for _ in range(50):
                await asyncio.sleep(0.05)
                pending = get_pending_approvals()
                if pending:
                    resolve_approval(pending[0].id, ApprovalStatus.APPROVED, "tester")
                    return
            raise AssertionError("No pending approval found")

        task = asyncio.create_task(step.execute(ctx))
        approve_task = asyncio.create_task(approve_quick())

        await task
        await approve_task

        # Check that request ID was stored in context
        assert f"approval_request_{step.name}" in ctx.state


# ============================================================================
# Checklist Item Completion Tests
# ============================================================================


class TestChecklistCompletion:
    """Tests for checklist item completion validation."""

    def _make_context(self, inputs=None, state=None, step_outputs=None):
        from aragora.workflow.step import WorkflowContext

        return WorkflowContext(
            workflow_id="wf_test",
            definition_id="def_test",
            inputs=inputs or {},
            state=state or {},
            step_outputs=step_outputs or {},
            current_step_id="test_checkpoint",
        )

    @pytest.fixture(autouse=True)
    def cleanup_approvals(self):
        """Clean up pending approvals before and after each test."""
        from aragora.workflow.nodes.human_checkpoint import (
            clear_pending_approvals,
            reset_approval_recovery,
        )

        clear_pending_approvals()
        reset_approval_recovery()
        yield
        clear_pending_approvals()
        reset_approval_recovery()

    @pytest.mark.asyncio
    async def test_approval_with_all_checklist_items_complete(self):
        """Test approval succeeds when all required checklist items are checked."""
        from aragora.workflow.nodes.human_checkpoint import (
            HumanCheckpointStep,
            ApprovalStatus,
            get_pending_approvals,
            resolve_approval,
        )

        step = HumanCheckpointStep(
            name="Checklist Test",
            config={
                "title": "Checklist Approval",
                "checklist": [
                    {"label": "Item 1", "required": True},
                    {"label": "Item 2", "required": True},
                ],
                "require_all_checklist": True,
                "timeout_seconds": 5.0,
            },
        )
        ctx = self._make_context()

        async def approve_with_checklist():
            for _ in range(50):
                await asyncio.sleep(0.05)
                pending = get_pending_approvals()
                if pending:
                    resolve_approval(
                        pending[0].id,
                        ApprovalStatus.APPROVED,
                        "reviewer",
                        checklist_updates={"item_0": True, "item_1": True},
                    )
                    return
            raise AssertionError("No pending approval found")

        task = asyncio.create_task(step.execute(ctx))
        approve_task = asyncio.create_task(approve_with_checklist())

        result = await task
        await approve_task

        assert result["status"] == "approved"

    @pytest.mark.asyncio
    async def test_approval_with_missing_checklist_items_rejected(self):
        """Test approval is rejected when required checklist items are not checked."""
        from aragora.workflow.nodes.human_checkpoint import (
            HumanCheckpointStep,
            ApprovalStatus,
            get_pending_approvals,
            resolve_approval,
        )

        step = HumanCheckpointStep(
            name="Missing Checklist",
            config={
                "title": "Missing Items",
                "checklist": [
                    {"label": "Required Item 1", "required": True},
                    {"label": "Required Item 2", "required": True},
                ],
                "require_all_checklist": True,
                "timeout_seconds": 5.0,
            },
        )
        ctx = self._make_context()

        async def approve_incomplete():
            for _ in range(50):
                await asyncio.sleep(0.05)
                pending = get_pending_approvals()
                if pending:
                    # Only check one item, leaving the other unchecked
                    resolve_approval(
                        pending[0].id,
                        ApprovalStatus.APPROVED,
                        "reviewer",
                        checklist_updates={"item_0": True},  # item_1 not checked
                    )
                    return
            raise AssertionError("No pending approval found")

        task = asyncio.create_task(step.execute(ctx))
        approve_task = asyncio.create_task(approve_incomplete())

        result = await task
        await approve_task

        assert result["status"] == "rejected"
        assert "missing_items" in result
        assert "Required Item 2" in result["missing_items"]

    @pytest.mark.asyncio
    async def test_optional_checklist_items_not_required(self):
        """Test that optional checklist items don't block approval."""
        from aragora.workflow.nodes.human_checkpoint import (
            HumanCheckpointStep,
            ApprovalStatus,
            get_pending_approvals,
            resolve_approval,
        )

        step = HumanCheckpointStep(
            name="Optional Items",
            config={
                "title": "Optional Test",
                "checklist": [
                    {"label": "Required Item", "required": True},
                    {"label": "Optional Item", "required": False},
                ],
                "require_all_checklist": True,
                "timeout_seconds": 5.0,
            },
        )
        ctx = self._make_context()

        async def approve_required_only():
            for _ in range(50):
                await asyncio.sleep(0.05)
                pending = get_pending_approvals()
                if pending:
                    resolve_approval(
                        pending[0].id,
                        ApprovalStatus.APPROVED,
                        "reviewer",
                        checklist_updates={"item_0": True},  # Only required item
                    )
                    return
            raise AssertionError("No pending approval found")

        task = asyncio.create_task(step.execute(ctx))
        approve_task = asyncio.create_task(approve_required_only())

        result = await task
        await approve_task

        assert result["status"] == "approved"


# ============================================================================
# Timeout Handling and Escalation Tests
# ============================================================================


class TestTimeoutAndEscalation:
    """Tests for timeout handling and escalation flows."""

    def _make_context(self, inputs=None, state=None, step_outputs=None):
        from aragora.workflow.step import WorkflowContext

        return WorkflowContext(
            workflow_id="wf_test",
            definition_id="def_test",
            inputs=inputs or {},
            state=state or {},
            step_outputs=step_outputs or {},
            current_step_id="test_checkpoint",
        )

    @pytest.fixture(autouse=True)
    def cleanup_approvals(self):
        """Clean up pending approvals before and after each test."""
        from aragora.workflow.nodes.human_checkpoint import (
            clear_pending_approvals,
            reset_approval_recovery,
        )

        clear_pending_approvals()
        reset_approval_recovery()
        yield
        clear_pending_approvals()
        reset_approval_recovery()

    @pytest.mark.asyncio
    async def test_timeout_returns_timeout_status(self):
        """Test that timeout returns timeout status."""
        from aragora.workflow.nodes.human_checkpoint import HumanCheckpointStep

        step = HumanCheckpointStep(
            name="Timeout Test",
            config={
                "title": "Will Timeout",
                "timeout_seconds": 0.1,  # Very short timeout
            },
        )
        ctx = self._make_context()

        result = await step.execute(ctx)

        assert result["status"] == "timeout"
        assert result["timeout_seconds"] == 0.1

    @pytest.mark.asyncio
    async def test_timeout_triggers_escalation(self):
        """Test that timeout triggers escalation to configured emails."""
        from aragora.workflow.nodes.human_checkpoint import HumanCheckpointStep

        step = HumanCheckpointStep(
            name="Escalation Test",
            config={
                "title": "Will Escalate",
                "timeout_seconds": 0.1,
                "escalation_emails": ["manager@example.com", "lead@example.com"],
            },
        )
        ctx = self._make_context()

        result = await step.execute(ctx)

        assert result["status"] == "timeout"
        assert result["escalated_to"] == ["manager@example.com", "lead@example.com"]

    @pytest.mark.asyncio
    async def test_escalation_updates_request_status(self):
        """Test that escalation updates the request status to ESCALATED."""
        from aragora.workflow.nodes.human_checkpoint import (
            HumanCheckpointStep,
            ApprovalStatus,
            get_pending_approvals,
            get_approval_request,
        )

        step = HumanCheckpointStep(
            name="Status Update Test",
            config={
                "title": "Status Check",
                "timeout_seconds": 0.1,
                "escalation_emails": ["escalate@example.com"],
            },
        )
        ctx = self._make_context()

        request_id = None

        async def capture_request_id():
            nonlocal request_id
            await asyncio.sleep(0.05)
            pending = get_pending_approvals()
            if pending:
                request_id = pending[0].id

        task = asyncio.create_task(step.execute(ctx))
        capture_task = asyncio.create_task(capture_request_id())

        await task
        await capture_task

        if request_id:
            request = get_approval_request(request_id)
            assert request.status == ApprovalStatus.ESCALATED


# ============================================================================
# Auto-Approval Tests
# ============================================================================


class TestAutoApproval:
    """Tests for auto-approval condition evaluation."""

    def _make_context(self, inputs=None, state=None, step_outputs=None):
        from aragora.workflow.step import WorkflowContext

        return WorkflowContext(
            workflow_id="wf_test",
            definition_id="def_test",
            inputs=inputs or {},
            state=state or {},
            step_outputs=step_outputs or {},
            current_step_id="test_checkpoint",
        )

    @pytest.fixture(autouse=True)
    def cleanup_approvals(self):
        """Clean up pending approvals before and after each test."""
        from aragora.workflow.nodes.human_checkpoint import (
            clear_pending_approvals,
            reset_approval_recovery,
        )

        clear_pending_approvals()
        reset_approval_recovery()
        yield
        clear_pending_approvals()
        reset_approval_recovery()

    @pytest.mark.asyncio
    async def test_auto_approve_when_condition_true(self):
        """Test auto-approval when condition evaluates to True."""
        from aragora.workflow.nodes.human_checkpoint import HumanCheckpointStep

        step = HumanCheckpointStep(
            name="Auto Approve",
            config={
                "title": "Auto Approval Test",
                "auto_approve_if": "inputs['amount'] < 100",
                "timeout_seconds": 0.5,
            },
        )
        ctx = self._make_context(inputs={"amount": 50})

        result = await step.execute(ctx)

        assert result["status"] == "approved"
        assert result["auto_approved"] is True

    @pytest.mark.asyncio
    async def test_no_auto_approve_when_condition_false(self):
        """Test no auto-approval when condition evaluates to False."""
        from aragora.workflow.nodes.human_checkpoint import (
            HumanCheckpointStep,
            ApprovalStatus,
            get_pending_approvals,
            resolve_approval,
        )

        step = HumanCheckpointStep(
            name="No Auto Approve",
            config={
                "title": "Manual Approval Needed",
                "auto_approve_if": "inputs['amount'] < 100",
                "timeout_seconds": 5.0,
            },
        )
        ctx = self._make_context(inputs={"amount": 500})  # Above threshold

        async def manual_approve():
            for _ in range(50):
                await asyncio.sleep(0.05)
                pending = get_pending_approvals()
                if pending:
                    resolve_approval(pending[0].id, ApprovalStatus.APPROVED, "reviewer")
                    return
            raise AssertionError("No pending approval found")

        task = asyncio.create_task(step.execute(ctx))
        approve_task = asyncio.create_task(manual_approve())

        result = await task
        await approve_task

        assert result["status"] == "approved"
        assert "auto_approved" not in result

    @pytest.mark.asyncio
    async def test_auto_approve_with_state_condition(self):
        """Test auto-approval with state-based condition."""
        from aragora.workflow.nodes.human_checkpoint import HumanCheckpointStep

        step = HumanCheckpointStep(
            name="State Auto Approve",
            config={
                "title": "State Condition",
                "auto_approve_if": "state['trusted_user'] == True",
                "timeout_seconds": 0.5,
            },
        )
        ctx = self._make_context(state={"trusted_user": True})

        result = await step.execute(ctx)

        assert result["status"] == "approved"
        assert result["auto_approved"] is True

    @pytest.mark.asyncio
    async def test_auto_approve_condition_error_falls_through(self):
        """Test that invalid auto-approve condition doesn't cause failure."""
        from aragora.workflow.nodes.human_checkpoint import (
            HumanCheckpointStep,
            ApprovalStatus,
            get_pending_approvals,
            resolve_approval,
        )

        step = HumanCheckpointStep(
            name="Bad Condition",
            config={
                "title": "Invalid Condition",
                "auto_approve_if": "undefined_variable > 0",  # Invalid
                "timeout_seconds": 5.0,
            },
        )
        ctx = self._make_context()

        async def manual_approve():
            for _ in range(50):
                await asyncio.sleep(0.05)
                pending = get_pending_approvals()
                if pending:
                    resolve_approval(pending[0].id, ApprovalStatus.APPROVED, "reviewer")
                    return
            raise AssertionError("No pending approval found")

        task = asyncio.create_task(step.execute(ctx))
        approve_task = asyncio.create_task(manual_approve())

        result = await task
        await approve_task

        # Should fall through to manual approval, not crash
        assert result["status"] == "approved"


# ============================================================================
# State Persistence Tests
# ============================================================================


class TestStatePersistence:
    """Tests for approval state persistence with GovernanceStore."""

    def _make_context(self, inputs=None, state=None, step_outputs=None):
        from aragora.workflow.step import WorkflowContext

        return WorkflowContext(
            workflow_id="wf_test",
            definition_id="def_test",
            inputs=inputs or {"user_id": "test_user"},
            state=state or {},
            step_outputs=step_outputs or {},
            current_step_id="test_checkpoint",
        )

    @pytest.fixture(autouse=True)
    def cleanup_approvals(self):
        """Clean up pending approvals before and after each test."""
        from aragora.workflow.nodes.human_checkpoint import (
            clear_pending_approvals,
            reset_approval_recovery,
        )

        clear_pending_approvals()
        reset_approval_recovery()
        yield
        clear_pending_approvals()
        reset_approval_recovery()

    @pytest.mark.asyncio
    async def test_approval_persisted_to_governance_store(self):
        """Test that approval request is persisted to GovernanceStore."""
        from aragora.workflow.nodes.human_checkpoint import (
            HumanCheckpointStep,
            ApprovalStatus,
            get_pending_approvals,
            resolve_approval,
        )

        mock_store = MagicMock()
        mock_store.save_approval = MagicMock()

        step = HumanCheckpointStep(
            name="Persist Test",
            config={
                "title": "Persist to Store",
                "timeout_seconds": 5.0,
            },
        )
        ctx = self._make_context()

        async def approve_quick():
            for _ in range(50):
                await asyncio.sleep(0.05)
                pending = get_pending_approvals()
                if pending:
                    resolve_approval(pending[0].id, ApprovalStatus.APPROVED, "reviewer")
                    return
            raise AssertionError("No pending approval found")

        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store",
            return_value=mock_store,
        ):
            task = asyncio.create_task(step.execute(ctx))
            approve_task = asyncio.create_task(approve_quick())

            await task
            await approve_task

        mock_store.save_approval.assert_called_once()
        call_kwargs = mock_store.save_approval.call_args[1]
        assert call_kwargs["title"] == "Persist to Store"
        assert call_kwargs["status"] == "pending"

    @pytest.mark.asyncio
    async def test_timeout_status_persisted(self):
        """Test that timeout status is persisted to GovernanceStore."""
        from aragora.workflow.nodes.human_checkpoint import HumanCheckpointStep

        mock_store = MagicMock()
        mock_store.save_approval = MagicMock()
        mock_store.update_approval_status = MagicMock()

        step = HumanCheckpointStep(
            name="Timeout Persist",
            config={
                "title": "Timeout Status",
                "timeout_seconds": 0.1,
            },
        )
        ctx = self._make_context()

        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store",
            return_value=mock_store,
        ):
            await step.execute(ctx)

        # Should have called update_approval_status with timeout
        mock_store.update_approval_status.assert_called()
        call_kwargs = mock_store.update_approval_status.call_args[1]
        assert call_kwargs["status"] == "timeout"


# ============================================================================
# Recovery Tests
# ============================================================================


class TestRecovery:
    """Tests for approval recovery from GovernanceStore."""

    @pytest.fixture(autouse=True)
    def cleanup_approvals(self):
        """Clean up pending approvals before and after each test."""
        from aragora.workflow.nodes.human_checkpoint import (
            clear_pending_approvals,
            reset_approval_recovery,
        )

        clear_pending_approvals()
        reset_approval_recovery()
        yield
        clear_pending_approvals()
        reset_approval_recovery()

    def test_recover_pending_approvals_from_store(self):
        """Test recovering pending approvals from GovernanceStore."""
        import json
        from aragora.workflow.nodes.human_checkpoint import (
            recover_pending_approvals,
            get_pending_approvals,
            _pending_approvals,
            reset_approval_recovery,
        )

        # Create mock store with pending approvals
        mock_record = MagicMock()
        mock_record.approval_id = "apr_recovered"
        mock_record.title = "Recovered Approval"
        mock_record.description = "From store"
        mock_record.status = "pending"
        mock_record.timeout_seconds = 3600
        mock_record.requested_at = datetime.now(timezone.utc)
        mock_record.approved_by = None
        mock_record.approved_at = None
        mock_record.decided_at = None
        mock_record.metadata_json = json.dumps(
            {
                "workflow_id": "wf_recovered",
                "step_id": "step_1",
                "checklist": [{"id": "c1", "label": "Item 1", "required": True}],
                "escalation_emails": ["test@example.com"],
            }
        )

        mock_store = MagicMock()
        mock_store.list_approvals.return_value = [mock_record]

        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store",
            return_value=mock_store,
        ):
            count = recover_pending_approvals()

        assert count == 1
        assert "apr_recovered" in _pending_approvals
        recovered = _pending_approvals["apr_recovered"]
        assert recovered.title == "Recovered Approval"
        assert recovered.workflow_id == "wf_recovered"

    def test_recover_is_idempotent(self):
        """Test that recover_pending_approvals is idempotent."""
        from aragora.workflow.nodes.human_checkpoint import (
            recover_pending_approvals,
            reset_approval_recovery,
        )

        mock_store = MagicMock()
        mock_store.list_approvals.return_value = []

        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store",
            return_value=mock_store,
        ):
            recover_pending_approvals()
            recover_pending_approvals()  # Second call
            recover_pending_approvals()  # Third call

        # Should only call list_approvals once
        assert mock_store.list_approvals.call_count == 1


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    @pytest.fixture(autouse=True)
    def cleanup_approvals(self):
        """Clean up pending approvals before and after each test."""
        from aragora.workflow.nodes.human_checkpoint import (
            clear_pending_approvals,
            reset_approval_recovery,
        )

        clear_pending_approvals()
        reset_approval_recovery()
        yield
        clear_pending_approvals()
        reset_approval_recovery()

    def test_resolve_approval_success(self):
        """Test resolve_approval successfully updates request."""
        from aragora.workflow.nodes.human_checkpoint import (
            ApprovalRequest,
            ApprovalStatus,
            resolve_approval,
            _pending_approvals,
        )

        # Create a pending request
        request = ApprovalRequest(
            id="apr_resolve",
            workflow_id="wf_1",
            step_id="step_1",
            title="Test",
            description="Test",
            checklist=[],
        )
        _pending_approvals["apr_resolve"] = request

        result = resolve_approval(
            "apr_resolve",
            ApprovalStatus.APPROVED,
            "resolver_user",
            notes="Looks good",
        )

        assert result is True
        assert request.status == ApprovalStatus.APPROVED
        assert request.responder_id == "resolver_user"
        assert request.responder_notes == "Looks good"
        assert request.responded_at is not None

    def test_resolve_approval_not_found(self):
        """Test resolve_approval returns False for unknown request."""
        from aragora.workflow.nodes.human_checkpoint import (
            ApprovalStatus,
            resolve_approval,
        )

        result = resolve_approval(
            "apr_nonexistent",
            ApprovalStatus.APPROVED,
            "resolver_user",
        )

        assert result is False

    def test_resolve_approval_with_checklist_updates(self):
        """Test resolve_approval updates checklist items."""
        from aragora.workflow.nodes.human_checkpoint import (
            ApprovalRequest,
            ApprovalStatus,
            ChecklistItem,
            resolve_approval,
            _pending_approvals,
        )

        request = ApprovalRequest(
            id="apr_checklist",
            workflow_id="wf_1",
            step_id="step_1",
            title="Test",
            description="Test",
            checklist=[
                ChecklistItem(id="c1", label="Item 1", required=True),
                ChecklistItem(id="c2", label="Item 2", required=True),
            ],
        )
        _pending_approvals["apr_checklist"] = request

        resolve_approval(
            "apr_checklist",
            ApprovalStatus.APPROVED,
            "resolver",
            checklist_updates={"c1": True, "c2": False},
        )

        assert request.checklist[0].checked is True
        assert request.checklist[1].checked is False

    def test_get_pending_approvals_filters_by_workflow(self):
        """Test get_pending_approvals filters by workflow_id."""
        from aragora.workflow.nodes.human_checkpoint import (
            ApprovalRequest,
            ApprovalStatus,
            get_pending_approvals,
            _pending_approvals,
        )

        # Add approvals for different workflows
        _pending_approvals["apr_1"] = ApprovalRequest(
            id="apr_1",
            workflow_id="wf_a",
            step_id="step",
            title="Test 1",
            description="",
            checklist=[],
        )
        _pending_approvals["apr_2"] = ApprovalRequest(
            id="apr_2",
            workflow_id="wf_b",
            step_id="step",
            title="Test 2",
            description="",
            checklist=[],
        )
        _pending_approvals["apr_3"] = ApprovalRequest(
            id="apr_3",
            workflow_id="wf_a",
            step_id="step",
            title="Test 3",
            description="",
            checklist=[],
        )

        wf_a_approvals = get_pending_approvals(workflow_id="wf_a")
        assert len(wf_a_approvals) == 2
        assert all(a.workflow_id == "wf_a" for a in wf_a_approvals)

    def test_get_pending_approvals_excludes_resolved(self):
        """Test get_pending_approvals excludes non-pending requests."""
        import aragora.workflow.nodes.human_checkpoint as hc_module
        from aragora.workflow.nodes.human_checkpoint import (
            ApprovalRequest,
            ApprovalStatus,
            get_pending_approvals,
            _pending_approvals,
        )

        _pending_approvals.clear()  # Ensure test isolation
        # Prevent recovery from GovernanceStore during test
        original_recovered = hc_module._approvals_recovered
        hc_module._approvals_recovered = True
        try:
            _pending_approvals["apr_pending"] = ApprovalRequest(
                id="apr_pending",
                workflow_id="wf",
                step_id="step",
                title="Pending",
                description="",
                checklist=[],
                status=ApprovalStatus.PENDING,
            )
            _pending_approvals["apr_approved"] = ApprovalRequest(
                id="apr_approved",
                workflow_id="wf",
                step_id="step",
                title="Approved",
                description="",
                checklist=[],
                status=ApprovalStatus.APPROVED,
            )

            pending = get_pending_approvals()
            assert len(pending) == 1
            assert pending[0].id == "apr_pending"
        finally:
            hc_module._approvals_recovered = original_recovered

    def test_get_approval_request_by_id(self):
        """Test get_approval_request returns request by ID."""
        from aragora.workflow.nodes.human_checkpoint import (
            ApprovalRequest,
            get_approval_request,
            _pending_approvals,
        )

        request = ApprovalRequest(
            id="apr_lookup",
            workflow_id="wf",
            step_id="step",
            title="Lookup Test",
            description="",
            checklist=[],
        )
        _pending_approvals["apr_lookup"] = request

        found = get_approval_request("apr_lookup")
        assert found is not None
        assert found.id == "apr_lookup"
        assert found.title == "Lookup Test"

    def test_get_approval_request_not_found(self):
        """Test get_approval_request returns None for unknown ID."""
        from aragora.workflow.nodes.human_checkpoint import get_approval_request

        result = get_approval_request("apr_does_not_exist")
        assert result is None

    def test_clear_pending_approvals(self):
        """Test clear_pending_approvals removes all approvals."""
        from aragora.workflow.nodes.human_checkpoint import (
            ApprovalRequest,
            clear_pending_approvals,
            _pending_approvals,
        )

        # Add some approvals
        _pending_approvals["a1"] = ApprovalRequest(
            id="a1", workflow_id="wf", step_id="s", title="", description="", checklist=[]
        )
        _pending_approvals["a2"] = ApprovalRequest(
            id="a2", workflow_id="wf", step_id="s", title="", description="", checklist=[]
        )

        count = clear_pending_approvals()

        assert count == 2
        assert len(_pending_approvals) == 0


# ============================================================================
# Notification Callback Tests
# ============================================================================


class TestNotificationCallbacks:
    """Tests for notification callbacks."""

    def _make_context(self, inputs=None, state=None, step_outputs=None):
        from aragora.workflow.step import WorkflowContext

        return WorkflowContext(
            workflow_id="wf_test",
            definition_id="def_test",
            inputs=inputs or {},
            state=state or {},
            step_outputs=step_outputs or {},
            current_step_id="test_checkpoint",
        )

    @pytest.fixture(autouse=True)
    def cleanup_approvals(self):
        """Clean up pending approvals before and after each test."""
        from aragora.workflow.nodes.human_checkpoint import (
            clear_pending_approvals,
            reset_approval_recovery,
        )

        clear_pending_approvals()
        reset_approval_recovery()
        yield
        clear_pending_approvals()
        reset_approval_recovery()

    @pytest.mark.asyncio
    async def test_on_approval_requested_callback(self):
        """Test that on_approval_requested callback is invoked."""
        from aragora.workflow.nodes.human_checkpoint import (
            HumanCheckpointStep,
            ApprovalStatus,
            get_pending_approvals,
            resolve_approval,
        )

        callback_received = None

        def approval_callback(request):
            nonlocal callback_received
            callback_received = request

        step = HumanCheckpointStep(
            name="Callback Test",
            config={
                "title": "Callback Approval",
                "timeout_seconds": 5.0,
            },
        )
        step.on_approval_requested = approval_callback
        ctx = self._make_context()

        async def approve_quick():
            for _ in range(50):
                await asyncio.sleep(0.05)
                pending = get_pending_approvals()
                if pending:
                    resolve_approval(pending[0].id, ApprovalStatus.APPROVED, "reviewer")
                    return
            raise AssertionError("No pending approval found")

        task = asyncio.create_task(step.execute(ctx))
        approve_task = asyncio.create_task(approve_quick())

        await task
        await approve_task

        assert callback_received is not None
        assert callback_received.title == "Callback Approval"

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_block_execution(self):
        """Test that callback exception doesn't block approval flow."""
        from aragora.workflow.nodes.human_checkpoint import (
            HumanCheckpointStep,
            ApprovalStatus,
            get_pending_approvals,
            resolve_approval,
        )

        def failing_callback(request):
            raise ValueError("Callback failed!")

        step = HumanCheckpointStep(
            name="Failing Callback",
            config={
                "title": "Should Continue",
                "timeout_seconds": 5.0,
            },
        )
        step.on_approval_requested = failing_callback
        ctx = self._make_context()

        async def approve_quick():
            for _ in range(50):
                await asyncio.sleep(0.05)
                pending = get_pending_approvals()
                if pending:
                    resolve_approval(pending[0].id, ApprovalStatus.APPROVED, "reviewer")
                    return
            raise AssertionError("No pending approval found")

        task = asyncio.create_task(step.execute(ctx))
        approve_task = asyncio.create_task(approve_quick())

        result = await task
        await approve_task

        # Should still complete despite callback failure
        assert result["status"] == "approved"


# ============================================================================
# Description Building Tests
# ============================================================================


class TestDescriptionBuilding:
    """Tests for description building from context."""

    def _make_context(self, inputs=None, state=None, step_outputs=None):
        from aragora.workflow.step import WorkflowContext

        return WorkflowContext(
            workflow_id="wf_123",
            definition_id="def_test",
            inputs=inputs or {},
            state=state or {},
            step_outputs=step_outputs or {},
            current_step_id="checkpoint_step",
        )

    def test_build_description_includes_base(self):
        """Test that description includes base description."""
        from aragora.workflow.nodes.human_checkpoint import HumanCheckpointStep

        step = HumanCheckpointStep(
            name="Test",
            config={"description": "Base description here"},
        )
        ctx = self._make_context()

        description = step._build_description(step.config, ctx)

        assert "Base description here" in description

    def test_build_description_includes_workflow_info(self):
        """Test that description includes workflow and step info."""
        from aragora.workflow.nodes.human_checkpoint import HumanCheckpointStep

        step = HumanCheckpointStep(
            name="Test",
            config={},
        )
        ctx = self._make_context()

        description = step._build_description(step.config, ctx)

        assert "wf_123" in description
        assert "checkpoint_step" in description

    def test_build_description_includes_inputs(self):
        """Test that description includes input summary."""
        from aragora.workflow.nodes.human_checkpoint import HumanCheckpointStep

        step = HumanCheckpointStep(
            name="Test",
            config={},
        )
        ctx = self._make_context(inputs={"amount": 1000, "user": "alice"})

        description = step._build_description(step.config, ctx)

        assert "amount" in description
        assert "1000" in description
        assert "user" in description
        assert "alice" in description


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling in human checkpoint."""

    def _make_context(self, inputs=None, state=None, step_outputs=None):
        from aragora.workflow.step import WorkflowContext

        return WorkflowContext(
            workflow_id="wf_test",
            definition_id="def_test",
            inputs=inputs or {},
            state=state or {},
            step_outputs=step_outputs or {},
            current_step_id="test_checkpoint",
        )

    @pytest.fixture(autouse=True)
    def cleanup_approvals(self):
        """Clean up pending approvals before and after each test."""
        from aragora.workflow.nodes.human_checkpoint import (
            clear_pending_approvals,
            reset_approval_recovery,
        )

        clear_pending_approvals()
        reset_approval_recovery()
        yield
        clear_pending_approvals()
        reset_approval_recovery()

    @pytest.mark.asyncio
    async def test_governance_store_failure_does_not_block(self):
        """Test that GovernanceStore failure doesn't block approval flow."""
        from aragora.workflow.nodes.human_checkpoint import (
            HumanCheckpointStep,
            ApprovalStatus,
            get_pending_approvals,
            resolve_approval,
        )

        mock_store = MagicMock()
        mock_store.save_approval = MagicMock(side_effect=RuntimeError("Store error"))

        step = HumanCheckpointStep(
            name="Store Failure",
            config={
                "title": "Should Continue",
                "timeout_seconds": 5.0,
            },
        )
        ctx = self._make_context()

        async def approve_quick():
            for _ in range(50):
                await asyncio.sleep(0.05)
                pending = get_pending_approvals()
                if pending:
                    resolve_approval(pending[0].id, ApprovalStatus.APPROVED, "reviewer")
                    return
            raise AssertionError("No pending approval found")

        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store",
            return_value=mock_store,
        ):
            task = asyncio.create_task(step.execute(ctx))
            approve_task = asyncio.create_task(approve_quick())

            result = await task
            await approve_task

        # Should still complete despite store failure
        assert result["status"] == "approved"

    @pytest.mark.asyncio
    async def test_notification_failure_does_not_block(self):
        """Test that notification failure doesn't block approval flow."""
        from aragora.workflow.nodes.human_checkpoint import (
            HumanCheckpointStep,
            ApprovalStatus,
            get_pending_approvals,
            resolve_approval,
        )

        step = HumanCheckpointStep(
            name="Notify Failure",
            config={
                "title": "Should Continue",
                "timeout_seconds": 5.0,
            },
        )
        ctx = self._make_context()

        async def approve_quick():
            for _ in range(50):
                await asyncio.sleep(0.05)
                pending = get_pending_approvals()
                if pending:
                    resolve_approval(pending[0].id, ApprovalStatus.APPROVED, "reviewer")
                    return
            raise AssertionError("No pending approval found")

        # The notification module may not exist, but the code handles ImportError gracefully
        task = asyncio.create_task(step.execute(ctx))
        approve_task = asyncio.create_task(approve_quick())

        result = await task
        await approve_task

        assert result["status"] == "approved"


# ============================================================================
# Action URL Building Tests
# ============================================================================


class TestActionUrlBuilding:
    """Tests for action URL building."""

    def test_build_action_url_with_base_url(self):
        """Test action URL building with ARAGORA_BASE_URL set."""
        from aragora.workflow.nodes.human_checkpoint import (
            HumanCheckpointStep,
            ApprovalRequest,
        )
        import os

        step = HumanCheckpointStep(name="Test", config={})
        request = ApprovalRequest(
            id="apr_123",
            workflow_id="wf_456",
            step_id="step",
            title="",
            description="",
            checklist=[],
        )

        with patch.dict(os.environ, {"ARAGORA_BASE_URL": "https://aragora.example.com"}):
            url = step._build_action_url(request)

        assert url == "https://aragora.example.com/workflows/wf_456/approvals/apr_123"

    def test_build_action_url_without_base_url(self):
        """Test action URL building without ARAGORA_BASE_URL."""
        from aragora.workflow.nodes.human_checkpoint import (
            HumanCheckpointStep,
            ApprovalRequest,
        )
        import os

        step = HumanCheckpointStep(name="Test", config={})
        request = ApprovalRequest(
            id="apr_123",
            workflow_id="wf_456",
            step_id="step",
            title="",
            description="",
            checklist=[],
        )

        with patch.dict(os.environ, {"ARAGORA_BASE_URL": ""}):
            url = step._build_action_url(request)

        assert url is None


# ============================================================================
# Governance Store Polling Tests
# ============================================================================


class TestGovernanceStorePolling:
    """Tests for GovernanceStore polling during wait."""

    def _make_context(self, inputs=None, state=None, step_outputs=None):
        from aragora.workflow.step import WorkflowContext

        return WorkflowContext(
            workflow_id="wf_test",
            definition_id="def_test",
            inputs=inputs or {},
            state=state or {},
            step_outputs=step_outputs or {},
            current_step_id="test_checkpoint",
        )

    @pytest.fixture(autouse=True)
    def cleanup_approvals(self):
        """Clean up pending approvals before and after each test."""
        from aragora.workflow.nodes.human_checkpoint import (
            clear_pending_approvals,
            reset_approval_recovery,
        )

        clear_pending_approvals()
        reset_approval_recovery()
        yield
        clear_pending_approvals()
        reset_approval_recovery()

    @pytest.mark.asyncio
    async def test_detects_approval_change_in_memory(self):
        """Test that approval status change is detected during wait loop."""
        from aragora.workflow.nodes.human_checkpoint import (
            HumanCheckpointStep,
            ApprovalStatus,
            get_pending_approvals,
            _pending_approvals,
        )

        step = HumanCheckpointStep(
            name="Memory Polling Test",
            config={
                "title": "External Approval",
                "timeout_seconds": 10.0,
            },
        )
        ctx = self._make_context()

        async def simulate_external_approval():
            """Simulate an external system modifying the in-memory request."""
            for _ in range(50):
                await asyncio.sleep(0.05)
                pending = get_pending_approvals()
                if pending:
                    # Directly modify the request status (simulating external approval)
                    request = pending[0]
                    request.status = ApprovalStatus.APPROVED
                    request.responder_id = "external_user"
                    request.responded_at = datetime.now(timezone.utc)
                    return
            raise AssertionError("No pending approval found")

        task = asyncio.create_task(step.execute(ctx))
        approve_task = asyncio.create_task(simulate_external_approval())

        result = await task
        await approve_task

        assert result["status"] == "approved"
        assert result["responder_id"] == "external_user"

    def test_governance_store_get_approval_reconstructs_request(self):
        """Test that get_approval_request can reconstruct from GovernanceStore."""
        import json
        from aragora.workflow.nodes.human_checkpoint import (
            get_approval_request,
            _pending_approvals,
            reset_approval_recovery,
        )

        # Make sure we start fresh
        _pending_approvals.clear()
        reset_approval_recovery()

        mock_record = MagicMock()
        mock_record.approval_id = "apr_store_lookup"
        mock_record.title = "Store Lookup Test"
        mock_record.description = "From governance store"
        mock_record.status = "pending"
        mock_record.timeout_seconds = 3600
        mock_record.requested_at = datetime.now(timezone.utc)
        mock_record.approved_by = None
        mock_record.decided_at = None
        mock_record.metadata_json = json.dumps(
            {
                "workflow_id": "wf_store",
                "step_id": "step_store",
                "checklist": [],
                "escalation_emails": [],
            }
        )

        mock_store = MagicMock()
        mock_store.list_approvals.return_value = []  # For recovery
        mock_store.get_approval.return_value = mock_record

        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store",
            return_value=mock_store,
        ):
            # Request not in memory, should fetch from store
            result = get_approval_request("apr_store_lookup")

        assert result is not None
        assert result.id == "apr_store_lookup"
        assert result.title == "Store Lookup Test"
        assert result.workflow_id == "wf_store"
        # Should now be cached in memory
        assert "apr_store_lookup" in _pending_approvals
