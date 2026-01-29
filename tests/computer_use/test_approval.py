"""
Tests for Computer Use Approval Workflow component.

Tests approval requests, decisions, and workflow management.
"""

import asyncio
import pytest
import time

from aragora.computer_use.approval import (
    ApprovalCategory,
    ApprovalConfig,
    ApprovalContext,
    ApprovalPriority,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalWorkflow,
    LoggingNotifier,
    create_approval_workflow,
)


class TestApprovalStatus:
    """Tests for ApprovalStatus enum."""

    def test_approval_statuses(self):
        """Test approval status values."""
        assert ApprovalStatus.PENDING.value == "pending"
        assert ApprovalStatus.APPROVED.value == "approved"
        assert ApprovalStatus.DENIED.value == "denied"
        assert ApprovalStatus.EXPIRED.value == "expired"
        assert ApprovalStatus.CANCELLED.value == "cancelled"


class TestApprovalCategory:
    """Tests for ApprovalCategory enum."""

    def test_approval_categories(self):
        """Test approval category values."""
        assert ApprovalCategory.CREDENTIAL_ACCESS.value == "credential_access"
        assert ApprovalCategory.SENSITIVE_DATA.value == "sensitive_data"
        assert ApprovalCategory.DESTRUCTIVE_ACTION.value == "destructive_action"
        assert ApprovalCategory.FINANCIAL.value == "financial"
        assert ApprovalCategory.PII_ACCESS.value == "pii_access"


class TestApprovalContext:
    """Tests for ApprovalContext."""

    def test_create_context(self):
        """Test creating approval context."""
        context = ApprovalContext(
            task_id="task-123",
            action_type="click",
            action_details={"x": 100, "y": 200},
            category=ApprovalCategory.SENSITIVE_DATA,
            reason="Clicking on password field",
            risk_level="high",
        )

        assert context.task_id == "task-123"
        assert context.action_type == "click"
        assert context.category == ApprovalCategory.SENSITIVE_DATA
        assert context.risk_level == "high"


class TestApprovalRequest:
    """Tests for ApprovalRequest."""

    def test_create_request(self):
        """Test creating approval request."""
        context = ApprovalContext(
            task_id="task-123",
            action_type="type",
            action_details={},
            category=ApprovalCategory.CREDENTIAL_ACCESS,
            reason="Entering password",
        )

        now = time.time()
        request = ApprovalRequest(
            id="req-123",
            context=context,
            status=ApprovalStatus.PENDING,
            priority=ApprovalPriority.HIGH,
            created_at=now,
            expires_at=now + 300,
        )

        assert request.id == "req-123"
        assert request.status == ApprovalStatus.PENDING
        assert request.priority == ApprovalPriority.HIGH
        assert request.is_expired() is False

    def test_request_to_dict(self):
        """Test converting request to dictionary."""
        context = ApprovalContext(
            task_id="task-123",
            action_type="click",
            action_details={},
            category=ApprovalCategory.UNKNOWN,
            reason="Test",
        )

        now = time.time()
        request = ApprovalRequest(
            id="req-123",
            context=context,
            status=ApprovalStatus.PENDING,
            priority=ApprovalPriority.MEDIUM,
            created_at=now,
            expires_at=now + 300,
        )

        result = request.to_dict()

        assert result["id"] == "req-123"
        assert result["task_id"] == "task-123"
        assert result["status"] == "pending"
        assert result["priority"] == "medium"

    def test_request_expired(self):
        """Test expired request detection."""
        context = ApprovalContext(
            task_id="task-123",
            action_type="click",
            action_details={},
            category=ApprovalCategory.UNKNOWN,
            reason="Test",
        )

        now = time.time()
        request = ApprovalRequest(
            id="req-123",
            context=context,
            status=ApprovalStatus.PENDING,
            priority=ApprovalPriority.LOW,
            created_at=now - 400,
            expires_at=now - 100,  # Expired 100 seconds ago
        )

        assert request.is_expired() is True


class TestApprovalWorkflow:
    """Tests for ApprovalWorkflow."""

    @pytest.fixture
    def workflow(self) -> ApprovalWorkflow:
        """Create an approval workflow for testing."""
        config = ApprovalConfig(
            default_timeout_seconds=5.0,
            min_timeout_seconds=0.1,
            notify_on_request=False,  # Disable notifications for tests
            notify_on_decision=False,
            notify_on_expiry=False,
        )
        return ApprovalWorkflow(config=config, notifiers=[])

    @pytest.mark.asyncio
    async def test_request_approval(self, workflow: ApprovalWorkflow):
        """Test creating an approval request."""
        context = ApprovalContext(
            task_id="task-123",
            action_type="click",
            action_details={"x": 100, "y": 200},
            category=ApprovalCategory.SENSITIVE_DATA,
            reason="Test action",
        )

        request = await workflow.request_approval(context)

        assert request is not None
        assert request.id is not None
        assert request.status == ApprovalStatus.PENDING
        assert request.context == context

    @pytest.mark.asyncio
    async def test_approve_request(self, workflow: ApprovalWorkflow):
        """Test approving a request."""
        context = ApprovalContext(
            task_id="task-123",
            action_type="click",
            action_details={},
            category=ApprovalCategory.UNKNOWN,
            reason="Test",
        )

        request = await workflow.request_approval(context)
        result = await workflow.approve(request.id, "admin-user", "Looks good")

        assert result is True

        updated = await workflow.get_request(request.id)
        assert updated is not None
        assert updated.status == ApprovalStatus.APPROVED
        assert updated.approved_by == "admin-user"
        assert updated.decision_reason == "Looks good"

    @pytest.mark.asyncio
    async def test_deny_request(self, workflow: ApprovalWorkflow):
        """Test denying a request."""
        context = ApprovalContext(
            task_id="task-123",
            action_type="type",
            action_details={},
            category=ApprovalCategory.CREDENTIAL_ACCESS,
            reason="Password entry",
        )

        request = await workflow.request_approval(context)
        result = await workflow.deny(request.id, "security-admin", "Not authorized")

        assert result is True

        updated = await workflow.get_request(request.id)
        assert updated is not None
        assert updated.status == ApprovalStatus.DENIED
        assert updated.denied_by == "security-admin"

    @pytest.mark.asyncio
    async def test_cancel_request(self, workflow: ApprovalWorkflow):
        """Test cancelling a request."""
        context = ApprovalContext(
            task_id="task-123",
            action_type="click",
            action_details={},
            category=ApprovalCategory.UNKNOWN,
            reason="Test",
        )

        request = await workflow.request_approval(context)
        result = await workflow.cancel(request.id)

        assert result is True

        updated = await workflow.get_request(request.id)
        assert updated is not None
        assert updated.status == ApprovalStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_wait_for_decision_approved(self, workflow: ApprovalWorkflow):
        """Test waiting for approval decision."""
        context = ApprovalContext(
            task_id="task-123",
            action_type="click",
            action_details={},
            category=ApprovalCategory.UNKNOWN,
            reason="Test",
        )

        request = await workflow.request_approval(context)

        # Approve in background
        async def approve_later():
            await asyncio.sleep(0.1)
            await workflow.approve(request.id, "admin")

        asyncio.create_task(approve_later())

        status = await workflow.wait_for_decision(request.id, timeout=2.0)
        assert status == ApprovalStatus.APPROVED

    @pytest.mark.asyncio
    async def test_wait_for_decision_timeout(self, workflow: ApprovalWorkflow):
        """Test waiting for decision with timeout."""
        context = ApprovalContext(
            task_id="task-123",
            action_type="click",
            action_details={},
            category=ApprovalCategory.UNKNOWN,
            reason="Test",
        )

        request = await workflow.request_approval(context, timeout_seconds=0.5)

        status = await workflow.wait_for_decision(request.id, timeout=1.0)
        assert status == ApprovalStatus.EXPIRED

    @pytest.mark.asyncio
    async def test_list_pending(self, workflow: ApprovalWorkflow):
        """Test listing pending requests."""
        # Create multiple requests
        for i in range(3):
            context = ApprovalContext(
                task_id=f"task-{i}",
                action_type="click",
                action_details={},
                category=ApprovalCategory.UNKNOWN,
                reason=f"Test {i}",
            )
            await workflow.request_approval(context)

        pending = await workflow.list_pending()
        assert len(pending) == 3

    @pytest.mark.asyncio
    async def test_list_all_with_filter(self, workflow: ApprovalWorkflow):
        """Test listing all requests with status filter."""
        # Create and approve one
        context1 = ApprovalContext(
            task_id="task-1",
            action_type="click",
            action_details={},
            category=ApprovalCategory.UNKNOWN,
            reason="Test 1",
        )
        req1 = await workflow.request_approval(context1)
        await workflow.approve(req1.id, "admin")

        # Create and deny one
        context2 = ApprovalContext(
            task_id="task-2",
            action_type="click",
            action_details={},
            category=ApprovalCategory.UNKNOWN,
            reason="Test 2",
        )
        req2 = await workflow.request_approval(context2)
        await workflow.deny(req2.id, "admin")

        # Create pending one
        context3 = ApprovalContext(
            task_id="task-3",
            action_type="click",
            action_details={},
            category=ApprovalCategory.UNKNOWN,
            reason="Test 3",
        )
        await workflow.request_approval(context3)

        approved = await workflow.list_all(status=ApprovalStatus.APPROVED)
        assert len(approved) == 1

        denied = await workflow.list_all(status=ApprovalStatus.DENIED)
        assert len(denied) == 1

        pending = await workflow.list_all(status=ApprovalStatus.PENDING)
        assert len(pending) == 1

    @pytest.mark.asyncio
    async def test_approve_expired_request(self, workflow: ApprovalWorkflow):
        """Test approving an expired request."""
        context = ApprovalContext(
            task_id="task-123",
            action_type="click",
            action_details={},
            category=ApprovalCategory.UNKNOWN,
            reason="Test",
        )

        request = await workflow.request_approval(context, timeout_seconds=0.1)

        # Wait for expiry
        await asyncio.sleep(0.2)

        result = await workflow.approve(request.id, "admin")
        assert result is False

    @pytest.mark.asyncio
    async def test_approve_nonexistent_request(self, workflow: ApprovalWorkflow):
        """Test approving nonexistent request."""
        result = await workflow.approve("nonexistent", "admin")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_stats(self, workflow: ApprovalWorkflow):
        """Test getting workflow statistics."""
        # Create some requests
        context = ApprovalContext(
            task_id="task-1",
            action_type="click",
            action_details={},
            category=ApprovalCategory.CREDENTIAL_ACCESS,
            reason="Test",
        )
        req1 = await workflow.request_approval(context, priority=ApprovalPriority.HIGH)
        await workflow.approve(req1.id, "admin")

        context2 = ApprovalContext(
            task_id="task-2",
            action_type="type",
            action_details={},
            category=ApprovalCategory.SENSITIVE_DATA,
            reason="Test 2",
        )
        await workflow.request_approval(context2, priority=ApprovalPriority.MEDIUM)

        stats = await workflow.get_stats()

        assert stats["total_requests"] == 2
        assert stats["pending_count"] == 1
        assert "approved" in stats["by_status"]
        assert "pending" in stats["by_status"]


class TestCreateApprovalWorkflow:
    """Tests for create_approval_workflow helper."""

    def test_create_default(self):
        """Test creating default workflow."""
        workflow = create_approval_workflow()
        assert workflow is not None
        assert len(workflow._notifiers) == 1
        assert isinstance(workflow._notifiers[0], LoggingNotifier)

    def test_create_with_webhook(self):
        """Test creating workflow with webhook."""
        workflow = create_approval_workflow(
            webhook_url="https://example.com/webhook",
            timeout_seconds=60.0,
        )
        assert workflow is not None
        assert len(workflow._notifiers) == 2

    def test_create_with_custom_config(self):
        """Test creating workflow with custom config."""
        workflow = create_approval_workflow(
            require_reason=True,
            timeout_seconds=120.0,
        )
        assert workflow._config.require_reason is True
        assert workflow._config.default_timeout_seconds == 120.0
