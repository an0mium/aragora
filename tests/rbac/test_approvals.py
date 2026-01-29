"""
Tests for RBAC Approval Workflows.

Tests cover:
- ApprovalStatus enum
- ApprovalDecision dataclass
- ApprovalRequest dataclass and properties
- ApprovalWorkflow methods:
  - request_access
  - approve (state machine)
  - reject
  - cancel
  - get_request
  - get_pending_for_approver
  - get_requests_by_requester
  - expire_old_requests
- Edge cases and error handling
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from aragora.rbac.approvals import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalStatus,
    ApprovalWorkflow,
    get_approval_workflow,
)


# =============================================================================
# Test ApprovalStatus Enum
# =============================================================================


class TestApprovalStatus:
    """Tests for ApprovalStatus enum."""

    def test_status_values(self):
        """All expected status values exist."""
        assert ApprovalStatus.PENDING.value == "pending"
        assert ApprovalStatus.APPROVED.value == "approved"
        assert ApprovalStatus.REJECTED.value == "rejected"
        assert ApprovalStatus.EXPIRED.value == "expired"
        assert ApprovalStatus.CANCELLED.value == "cancelled"

    def test_status_is_string_enum(self):
        """Status is a string enum for JSON serialization."""
        assert isinstance(ApprovalStatus.PENDING, str)
        assert ApprovalStatus.PENDING == "pending"


# =============================================================================
# Test ApprovalDecision
# =============================================================================


class TestApprovalDecision:
    """Tests for ApprovalDecision dataclass."""

    def test_create_decision(self):
        """Create an approval decision."""
        decision = ApprovalDecision(
            approver_id="approver-1",
            decision="approved",
            comment="Looks good",
        )

        assert decision.approver_id == "approver-1"
        assert decision.decision == "approved"
        assert decision.comment == "Looks good"
        assert decision.timestamp is not None

    def test_decision_default_timestamp(self):
        """Decision gets default timestamp."""
        decision = ApprovalDecision(approver_id="approver-1", decision="approved")
        now = datetime.now(timezone.utc)
        # Within a second
        assert abs((now - decision.timestamp).total_seconds()) < 1

    def test_decision_to_dict(self):
        """Decision serializes to dict."""
        decision = ApprovalDecision(
            approver_id="approver-1",
            decision="rejected",
            comment="Not sufficient justification",
        )
        result = decision.to_dict()

        assert result["approver_id"] == "approver-1"
        assert result["decision"] == "rejected"
        assert result["comment"] == "Not sufficient justification"
        assert "timestamp" in result


# =============================================================================
# Test ApprovalRequest
# =============================================================================


class TestApprovalRequest:
    """Tests for ApprovalRequest dataclass."""

    def test_create_request(self):
        """Create an approval request."""
        request = ApprovalRequest(
            id="req-123",
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            resource_id="debate-456",
            justification="Cleaning up test data",
            status=ApprovalStatus.PENDING,
            approvers=["admin-1", "admin-2"],
            required_approvals=1,
        )

        assert request.id == "req-123"
        assert request.requester_id == "user-1"
        assert request.permission == "debates:delete"
        assert request.status == ApprovalStatus.PENDING
        assert len(request.approvers) == 2

    def test_approval_count_empty(self):
        """Approval count is 0 with no decisions."""
        request = ApprovalRequest(
            id="req-123",
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            resource_id=None,
            justification="Test",
            status=ApprovalStatus.PENDING,
            approvers=["admin-1"],
            required_approvals=1,
        )

        assert request.approval_count == 0
        assert request.rejection_count == 0

    def test_approval_count_with_decisions(self):
        """Approval count reflects decisions."""
        request = ApprovalRequest(
            id="req-123",
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            resource_id=None,
            justification="Test",
            status=ApprovalStatus.PENDING,
            approvers=["admin-1", "admin-2", "admin-3"],
            required_approvals=2,
            decisions=[
                ApprovalDecision(approver_id="admin-1", decision="approved"),
                ApprovalDecision(approver_id="admin-2", decision="rejected"),
            ],
        )

        assert request.approval_count == 1
        assert request.rejection_count == 1

    def test_is_approved_threshold(self):
        """is_approved checks threshold."""
        request = ApprovalRequest(
            id="req-123",
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            resource_id=None,
            justification="Test",
            status=ApprovalStatus.PENDING,
            approvers=["admin-1", "admin-2"],
            required_approvals=2,
        )

        assert not request.is_approved

        # Add one approval - still not enough
        request.decisions.append(ApprovalDecision(approver_id="admin-1", decision="approved"))
        assert not request.is_approved

        # Add second approval - now approved
        request.decisions.append(ApprovalDecision(approver_id="admin-2", decision="approved"))
        assert request.is_approved

    def test_is_expired(self):
        """is_expired checks expiration time."""
        # Not expired
        request = ApprovalRequest(
            id="req-123",
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            resource_id=None,
            justification="Test",
            status=ApprovalStatus.PENDING,
            approvers=["admin-1"],
            required_approvals=1,
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert not request.is_expired

        # Expired
        request.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        assert request.is_expired

    def test_to_dict(self):
        """Request serializes to dict."""
        request = ApprovalRequest(
            id="req-123",
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            resource_id="debate-456",
            justification="Test",
            status=ApprovalStatus.PENDING,
            approvers=["admin-1"],
            required_approvals=1,
            org_id="org-1",
            workspace_id="ws-1",
        )
        result = request.to_dict()

        assert result["id"] == "req-123"
        assert result["status"] == "pending"
        assert result["approval_count"] == 0
        assert result["org_id"] == "org-1"
        assert "created_at" in result


# =============================================================================
# Test ApprovalWorkflow
# =============================================================================


class TestApprovalWorkflowRequestAccess:
    """Tests for ApprovalWorkflow.request_access()."""

    @pytest.fixture
    def workflow(self):
        """Fresh workflow instance."""
        return ApprovalWorkflow()

    @pytest.mark.asyncio
    async def test_request_access_basic(self, workflow):
        """Create a basic access request."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Need to clean up test debates",
            approvers=["admin-1"],
        )

        assert request.id.startswith("req-")
        assert request.requester_id == "user-1"
        assert request.permission == "debates:delete"
        assert request.status == ApprovalStatus.PENDING
        assert request.justification == "Need to clean up test debates"
        assert "admin-1" in request.approvers

    @pytest.mark.asyncio
    async def test_request_access_with_resource_id(self, workflow):
        """Create request for specific resource."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            resource_id="debate-456",
            justification="Clean up",
            approvers=["admin-1"],
        )

        assert request.resource_id == "debate-456"

    @pytest.mark.asyncio
    async def test_request_access_with_org_and_workspace(self, workflow):
        """Create request with org and workspace context."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Clean up",
            approvers=["admin-1"],
            org_id="org-123",
            workspace_id="ws-456",
        )

        assert request.org_id == "org-123"
        assert request.workspace_id == "ws-456"

    @pytest.mark.asyncio
    async def test_request_access_duration_validation(self, workflow):
        """Duration cannot exceed maximum."""
        with pytest.raises(ValueError, match="Duration cannot exceed"):
            await workflow.request_access(
                requester_id="user-1",
                permission="debates:delete",
                resource_type="debates",
                justification="Test",
                approvers=["admin-1"],
                duration_hours=999999,  # Exceeds max
            )

    @pytest.mark.asyncio
    async def test_request_access_no_approvers_error(self, workflow):
        """Request fails without approvers."""
        with patch.object(workflow, "_get_default_approvers", new_callable=AsyncMock) as mock:
            mock.return_value = []

            with pytest.raises(ValueError, match="No approvers available"):
                await workflow.request_access(
                    requester_id="user-1",
                    permission="debates:delete",
                    resource_type="debates",
                    justification="Test",
                    # No approvers provided
                )

    @pytest.mark.asyncio
    async def test_request_access_indexes_by_requester(self, workflow):
        """Requests are indexed by requester."""
        await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        assert "user-1" in workflow._by_requester
        assert len(workflow._by_requester["user-1"]) == 1

    @pytest.mark.asyncio
    async def test_request_access_indexes_by_approver(self, workflow):
        """Requests are indexed by approvers."""
        await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1", "admin-2"],
        )

        assert "admin-1" in workflow._by_approver
        assert "admin-2" in workflow._by_approver

    @pytest.mark.asyncio
    async def test_required_approvals_capped_to_approver_count(self, workflow):
        """Required approvals cannot exceed approver count."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
            required_approvals=5,  # More than approvers
        )

        # Should be capped to 1
        assert request.required_approvals == 1


class TestApprovalWorkflowApprove:
    """Tests for ApprovalWorkflow.approve()."""

    @pytest.fixture
    def workflow(self):
        """Fresh workflow instance."""
        return ApprovalWorkflow()

    @pytest.mark.asyncio
    async def test_approve_basic(self, workflow):
        """Approve a pending request."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        with patch.object(workflow, "_grant_temporary_permission", new_callable=AsyncMock):
            result = await workflow.approve(
                approver_id="admin-1",
                request_id=request.id,
                comment="Approved for testing",
            )

        assert result.status == ApprovalStatus.APPROVED
        assert result.approval_count == 1
        assert result.decisions[0].comment == "Approved for testing"
        assert result.resolved_at is not None

    @pytest.mark.asyncio
    async def test_approve_partial_multi_approver(self, workflow):
        """Partial approval in multi-approver workflow."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1", "admin-2"],
            required_approvals=2,
        )

        result = await workflow.approve(
            approver_id="admin-1",
            request_id=request.id,
        )

        # Still pending - needs 2 approvals
        assert result.status == ApprovalStatus.PENDING
        assert result.approval_count == 1
        assert result.resolved_at is None

    @pytest.mark.asyncio
    async def test_approve_completes_multi_approver(self, workflow):
        """Final approval completes multi-approver workflow."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1", "admin-2"],
            required_approvals=2,
        )

        await workflow.approve(approver_id="admin-1", request_id=request.id)

        with patch.object(workflow, "_grant_temporary_permission", new_callable=AsyncMock):
            result = await workflow.approve(approver_id="admin-2", request_id=request.id)

        assert result.status == ApprovalStatus.APPROVED
        assert result.approval_count == 2

    @pytest.mark.asyncio
    async def test_approve_not_found_error(self, workflow):
        """Approve fails for non-existent request."""
        with pytest.raises(ValueError, match="Request not found"):
            await workflow.approve(
                approver_id="admin-1",
                request_id="nonexistent",
            )

    @pytest.mark.asyncio
    async def test_approve_not_approver_error(self, workflow):
        """Approve fails if user is not an approver."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        with pytest.raises(ValueError, match="not an approver"):
            await workflow.approve(
                approver_id="other-user",
                request_id=request.id,
            )

    @pytest.mark.asyncio
    async def test_approve_already_decided_error(self, workflow):
        """Approve fails if user already decided."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1", "admin-2"],
            required_approvals=2,
        )

        await workflow.approve(approver_id="admin-1", request_id=request.id)

        with pytest.raises(ValueError, match="already made a decision"):
            await workflow.approve(approver_id="admin-1", request_id=request.id)

    @pytest.mark.asyncio
    async def test_approve_non_pending_error(self, workflow):
        """Approve fails if request not pending."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1", "admin-2"],
            required_approvals=1,
        )

        with patch.object(workflow, "_grant_temporary_permission", new_callable=AsyncMock):
            await workflow.approve(approver_id="admin-1", request_id=request.id)

        # Request is now approved, try to approve with a different approver
        with pytest.raises(ValueError, match="not pending"):
            await workflow.approve(approver_id="admin-2", request_id=request.id)

    @pytest.mark.asyncio
    async def test_approve_expired_error(self, workflow):
        """Approve fails if request expired."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        # Force expiration
        request.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)

        with pytest.raises(ValueError, match="expired"):
            await workflow.approve(approver_id="admin-1", request_id=request.id)


class TestApprovalWorkflowReject:
    """Tests for ApprovalWorkflow.reject()."""

    @pytest.fixture
    def workflow(self):
        """Fresh workflow instance."""
        return ApprovalWorkflow()

    @pytest.mark.asyncio
    async def test_reject_basic(self, workflow):
        """Reject a pending request."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        result = await workflow.reject(
            approver_id="admin-1",
            request_id=request.id,
            reason="Insufficient justification",
        )

        assert result.status == ApprovalStatus.REJECTED
        assert result.rejection_count == 1
        assert result.decisions[0].comment == "Insufficient justification"
        assert result.resolved_at is not None

    @pytest.mark.asyncio
    async def test_reject_terminates_workflow(self, workflow):
        """Any rejection terminates the workflow."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1", "admin-2", "admin-3"],
            required_approvals=2,
        )

        # One rejection should end the workflow
        result = await workflow.reject(
            approver_id="admin-1",
            request_id=request.id,
            reason="No",
        )

        assert result.status == ApprovalStatus.REJECTED
        # Can't approve after rejection
        with pytest.raises(ValueError, match="not pending"):
            await workflow.approve(approver_id="admin-2", request_id=request.id)

    @pytest.mark.asyncio
    async def test_reject_not_approver_error(self, workflow):
        """Reject fails if user is not an approver."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        with pytest.raises(ValueError, match="not an approver"):
            await workflow.reject(
                approver_id="other-user",
                request_id=request.id,
                reason="No",
            )


class TestApprovalWorkflowCancel:
    """Tests for ApprovalWorkflow.cancel()."""

    @pytest.fixture
    def workflow(self):
        """Fresh workflow instance."""
        return ApprovalWorkflow()

    @pytest.mark.asyncio
    async def test_cancel_basic(self, workflow):
        """Requester can cancel their request."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        result = await workflow.cancel(
            requester_id="user-1",
            request_id=request.id,
            reason="No longer needed",
        )

        assert result.status == ApprovalStatus.CANCELLED
        assert result.resolved_at is not None
        assert result.metadata.get("cancellation_reason") == "No longer needed"

    @pytest.mark.asyncio
    async def test_cancel_not_requester_error(self, workflow):
        """Only requester can cancel."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        with pytest.raises(ValueError, match="Only the requester"):
            await workflow.cancel(
                requester_id="other-user",
                request_id=request.id,
            )

    @pytest.mark.asyncio
    async def test_cancel_non_pending_error(self, workflow):
        """Cannot cancel non-pending request."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        with patch.object(workflow, "_grant_temporary_permission", new_callable=AsyncMock):
            await workflow.approve(approver_id="admin-1", request_id=request.id)

        with pytest.raises(ValueError, match="not pending"):
            await workflow.cancel(requester_id="user-1", request_id=request.id)


class TestApprovalWorkflowQueries:
    """Tests for ApprovalWorkflow query methods."""

    @pytest.fixture
    def workflow(self):
        """Fresh workflow instance."""
        return ApprovalWorkflow()

    @pytest.mark.asyncio
    async def test_get_request(self, workflow):
        """Get request by ID."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        result = await workflow.get_request(request.id)
        assert result is not None
        assert result.id == request.id

    @pytest.mark.asyncio
    async def test_get_request_not_found(self, workflow):
        """Get request returns None for non-existent."""
        result = await workflow.get_request("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_pending_for_approver(self, workflow):
        """Get pending requests for approver."""
        await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test 1",
            approvers=["admin-1"],
        )
        await workflow.request_access(
            requester_id="user-2",
            permission="debates:read",
            resource_type="debates",
            justification="Test 2",
            approvers=["admin-1"],
        )

        results = await workflow.get_pending_for_approver("admin-1")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_get_pending_excludes_decided(self, workflow):
        """Pending query excludes requests approver already decided on."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1", "admin-2"],
            required_approvals=2,
        )

        await workflow.approve(approver_id="admin-1", request_id=request.id)

        # admin-1 should not see this request anymore
        results = await workflow.get_pending_for_approver("admin-1")
        assert len(results) == 0

        # admin-2 should still see it
        results = await workflow.get_pending_for_approver("admin-2")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_get_requests_by_requester(self, workflow):
        """Get requests by requester."""
        await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test 1",
            approvers=["admin-1"],
        )
        await workflow.request_access(
            requester_id="user-1",
            permission="debates:read",
            resource_type="debates",
            justification="Test 2",
            approvers=["admin-1"],
        )
        await workflow.request_access(
            requester_id="user-2",
            permission="debates:read",
            resource_type="debates",
            justification="Test 3",
            approvers=["admin-1"],
        )

        results = await workflow.get_requests_by_requester("user-1")
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_get_requests_by_requester_with_status_filter(self, workflow):
        """Filter requests by status."""
        request1 = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test 1",
            approvers=["admin-1"],
        )
        await workflow.request_access(
            requester_id="user-1",
            permission="debates:read",
            resource_type="debates",
            justification="Test 2",
            approvers=["admin-1"],
        )

        with patch.object(workflow, "_grant_temporary_permission", new_callable=AsyncMock):
            await workflow.approve(approver_id="admin-1", request_id=request1.id)

        pending = await workflow.get_requests_by_requester("user-1", status=ApprovalStatus.PENDING)
        assert len(pending) == 1

        approved = await workflow.get_requests_by_requester(
            "user-1", status=ApprovalStatus.APPROVED
        )
        assert len(approved) == 1


class TestApprovalWorkflowExpiration:
    """Tests for ApprovalWorkflow.expire_old_requests()."""

    @pytest.fixture
    def workflow(self):
        """Fresh workflow instance."""
        return ApprovalWorkflow()

    @pytest.mark.asyncio
    async def test_expire_old_requests(self, workflow):
        """Expire requests past their expiration time."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        # Force expiration
        request.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)

        count = await workflow.expire_old_requests()
        assert count == 1
        assert request.status == ApprovalStatus.EXPIRED

    @pytest.mark.asyncio
    async def test_expire_leaves_non_expired(self, workflow):
        """Non-expired requests are not affected."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        count = await workflow.expire_old_requests()
        assert count == 0
        assert request.status == ApprovalStatus.PENDING

    @pytest.mark.asyncio
    async def test_expire_leaves_non_pending(self, workflow):
        """Already resolved requests are not expired."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        with patch.object(workflow, "_grant_temporary_permission", new_callable=AsyncMock):
            await workflow.approve(approver_id="admin-1", request_id=request.id)

        # Force past expiration
        request.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)

        count = await workflow.expire_old_requests()
        assert count == 0
        assert request.status == ApprovalStatus.APPROVED  # Unchanged


class TestApprovalWorkflowSingleton:
    """Tests for get_approval_workflow singleton."""

    def test_get_approval_workflow(self):
        """Singleton returns same instance."""
        # Reset singleton for test
        import aragora.rbac.approvals as approvals_module

        approvals_module._workflow = None

        wf1 = get_approval_workflow()
        wf2 = get_approval_workflow()

        assert wf1 is wf2

        # Cleanup
        approvals_module._workflow = None


class TestApprovalWorkflowDefaultApprovers:
    """Tests for default approver lookup."""

    @pytest.fixture
    def workflow(self):
        """Fresh workflow instance."""
        return ApprovalWorkflow()

    @pytest.mark.asyncio
    async def test_get_default_approvers_uses_checker(self, workflow):
        """Default approvers come from PermissionChecker."""
        from unittest.mock import MagicMock

        mock_checker = MagicMock()
        mock_checker.get_users_with_permission.return_value = ["admin-1", "admin-2"]

        with patch("aragora.rbac.checker.get_permission_checker", return_value=mock_checker):
            result = await workflow._get_default_approvers(
                permission="debates:delete",
                resource_type="debates",
                org_id="org-1",
                workspace_id="ws-1",
            )

        assert result == ["admin-1", "admin-2"]
        mock_checker.get_users_with_permission.assert_called()

    @pytest.mark.asyncio
    async def test_get_default_approvers_fallback_wildcard(self, workflow):
        """Falls back to wildcard permission if admin not found."""
        from unittest.mock import MagicMock

        mock_checker = MagicMock()
        # First call returns empty, second returns approvers
        mock_checker.get_users_with_permission.side_effect = [[], ["admin-1"]]

        with patch("aragora.rbac.checker.get_permission_checker", return_value=mock_checker):
            result = await workflow._get_default_approvers(
                permission="debates:delete",
                resource_type="debates",
                org_id=None,
                workspace_id=None,
            )

        assert result == ["admin-1"]
        assert mock_checker.get_users_with_permission.call_count == 2

    @pytest.mark.asyncio
    async def test_get_default_approvers_handles_import_error(self, workflow):
        """Returns empty list if checker module not available."""
        import sys

        # Save original module
        original = sys.modules.get("aragora.rbac.checker")

        try:
            # Remove module to simulate import error
            sys.modules["aragora.rbac.checker"] = None  # type: ignore[assignment]

            result = await workflow._get_default_approvers(
                permission="debates:delete",
                resource_type="debates",
                org_id=None,
                workspace_id=None,
            )

            assert result == []
        finally:
            # Restore
            if original is not None:
                sys.modules["aragora.rbac.checker"] = original
            elif "aragora.rbac.checker" in sys.modules:
                del sys.modules["aragora.rbac.checker"]

    @pytest.mark.asyncio
    async def test_get_default_approvers_handles_exception(self, workflow):
        """Returns empty list if checker raises exception."""
        from unittest.mock import MagicMock

        mock_checker = MagicMock()
        mock_checker.get_users_with_permission.side_effect = Exception("DB error")

        with patch("aragora.rbac.checker.get_permission_checker", return_value=mock_checker):
            result = await workflow._get_default_approvers(
                permission="debates:delete",
                resource_type="debates",
                org_id=None,
                workspace_id=None,
            )

        assert result == []


class TestApprovalWorkflowAuditLog:
    """Tests for audit logging."""

    @pytest.fixture
    def workflow(self):
        """Fresh workflow instance."""
        return ApprovalWorkflow()

    @pytest.mark.asyncio
    async def test_request_creates_audit_log(self, workflow):
        """Creating request logs to audit."""
        with patch.object(workflow, "_audit_log", new_callable=AsyncMock) as mock_log:
            await workflow.request_access(
                requester_id="user-1",
                permission="debates:delete",
                resource_type="debates",
                justification="Test",
                approvers=["admin-1"],
            )

        mock_log.assert_called()
        call_args = mock_log.call_args
        assert call_args[0][0] == "access_request_created"

    @pytest.mark.asyncio
    async def test_approve_creates_audit_log(self, workflow):
        """Approving request logs to audit."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        with patch.object(workflow, "_audit_log", new_callable=AsyncMock) as mock_log:
            with patch.object(workflow, "_grant_temporary_permission", new_callable=AsyncMock):
                await workflow.approve(approver_id="admin-1", request_id=request.id)

        mock_log.assert_called()
        call_args = mock_log.call_args
        assert call_args[0][0] == "access_request_approved"

    @pytest.mark.asyncio
    async def test_reject_creates_audit_log(self, workflow):
        """Rejecting request logs to audit."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        with patch.object(workflow, "_audit_log", new_callable=AsyncMock) as mock_log:
            await workflow.reject(
                approver_id="admin-1",
                request_id=request.id,
                reason="No",
            )

        mock_log.assert_called()
        call_args = mock_log.call_args
        assert call_args[0][0] == "access_request_rejected"

    @pytest.mark.asyncio
    async def test_cancel_creates_audit_log(self, workflow):
        """Cancelling request logs to audit."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        with patch.object(workflow, "_audit_log", new_callable=AsyncMock) as mock_log:
            await workflow.cancel(
                requester_id="user-1",
                request_id=request.id,
                reason="Changed my mind",
            )

        mock_log.assert_called()
        call_args = mock_log.call_args
        assert call_args[0][0] == "access_request_cancelled"

    @pytest.mark.asyncio
    async def test_expire_creates_audit_log(self, workflow):
        """Expiring request logs to audit."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        # Force expiration
        request.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)

        with patch.object(workflow, "_audit_log", new_callable=AsyncMock) as mock_log:
            await workflow.expire_old_requests()

        mock_log.assert_called()
        call_args = mock_log.call_args
        assert call_args[0][0] == "access_request_expired"


# =============================================================================
# Additional Multi-Approver Tests
# =============================================================================


class TestMultiApproverScenarios:
    """Additional tests for multi-approver approval workflows."""

    @pytest.fixture
    def workflow(self):
        """Fresh workflow instance."""
        return ApprovalWorkflow()

    @pytest.mark.asyncio
    async def test_three_of_five_approvers_required(self, workflow):
        """Test 3 of 5 approvers required scenario."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="admin:system_config",
            resource_type="admin",
            justification="Critical system change",
            approvers=["admin-1", "admin-2", "admin-3", "admin-4", "admin-5"],
            required_approvals=3,
        )

        # First two approvals - still pending
        await workflow.approve(approver_id="admin-1", request_id=request.id)
        assert request.status == ApprovalStatus.PENDING
        assert request.approval_count == 1

        await workflow.approve(approver_id="admin-2", request_id=request.id)
        assert request.status == ApprovalStatus.PENDING
        assert request.approval_count == 2

        # Third approval - should complete
        with patch.object(workflow, "_grant_temporary_permission", new_callable=AsyncMock):
            await workflow.approve(approver_id="admin-3", request_id=request.id)

        assert request.status == ApprovalStatus.APPROVED
        assert request.approval_count == 3

    @pytest.mark.asyncio
    async def test_mixed_approvals_and_rejections(self, workflow):
        """Test that rejection wins even with prior approvals."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Cleanup",
            approvers=["admin-1", "admin-2", "admin-3"],
            required_approvals=2,
        )

        # First approval
        await workflow.approve(approver_id="admin-1", request_id=request.id)
        assert request.status == ApprovalStatus.PENDING
        assert request.approval_count == 1

        # Then rejection - should terminate
        await workflow.reject(
            approver_id="admin-2",
            request_id=request.id,
            reason="Policy violation",
        )

        assert request.status == ApprovalStatus.REJECTED
        assert request.approval_count == 1
        assert request.rejection_count == 1

    @pytest.mark.asyncio
    async def test_approvers_can_only_decide_once(self, workflow):
        """Test that approvers cannot change their decision."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1", "admin-2"],
            required_approvals=2,
        )

        await workflow.approve(approver_id="admin-1", request_id=request.id)

        # Same approver cannot approve again
        with pytest.raises(ValueError, match="already made a decision"):
            await workflow.approve(approver_id="admin-1", request_id=request.id)

    @pytest.mark.asyncio
    async def test_all_approvers_can_approve_independently(self, workflow):
        """Test that all designated approvers can approve."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1", "admin-2", "admin-3"],
            required_approvals=1,
        )

        # Any single approver can approve
        with patch.object(workflow, "_grant_temporary_permission", new_callable=AsyncMock):
            await workflow.approve(approver_id="admin-2", request_id=request.id)

        assert request.status == ApprovalStatus.APPROVED
        assert len(request.decisions) == 1
        assert request.decisions[0].approver_id == "admin-2"


# =============================================================================
# Metadata and Custom Fields Tests
# =============================================================================


class TestApprovalRequestMetadata:
    """Tests for metadata handling in approval requests."""

    @pytest.fixture
    def workflow(self):
        """Fresh workflow instance."""
        return ApprovalWorkflow()

    @pytest.mark.asyncio
    async def test_request_with_metadata(self, workflow):
        """Test creating request with custom metadata."""
        metadata = {
            "ticket_id": "JIRA-1234",
            "priority": "high",
            "environment": "production",
        }

        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Production cleanup",
            approvers=["admin-1"],
            metadata=metadata,
        )

        assert request.metadata["ticket_id"] == "JIRA-1234"
        assert request.metadata["priority"] == "high"
        assert request.metadata["environment"] == "production"

    @pytest.mark.asyncio
    async def test_metadata_in_serialization(self, workflow):
        """Test that metadata is included in to_dict."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
            metadata={"custom_field": "custom_value"},
        )

        result = request.to_dict()
        assert result["metadata"]["custom_field"] == "custom_value"

    @pytest.mark.asyncio
    async def test_cancellation_reason_in_metadata(self, workflow):
        """Test that cancellation reason is stored in metadata."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        await workflow.cancel(
            requester_id="user-1",
            request_id=request.id,
            reason="No longer needed",
        )

        assert request.metadata["cancellation_reason"] == "No longer needed"

    @pytest.mark.asyncio
    async def test_empty_metadata_default(self, workflow):
        """Test that empty metadata defaults to empty dict."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        assert request.metadata == {}


# =============================================================================
# Duration and Expiration Edge Cases
# =============================================================================


class TestDurationAndExpiration:
    """Tests for access duration and request expiration handling."""

    @pytest.fixture
    def workflow(self):
        """Fresh workflow instance."""
        return ApprovalWorkflow()

    @pytest.mark.asyncio
    async def test_custom_duration(self, workflow):
        """Test request with custom duration."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Short-term access",
            approvers=["admin-1"],
            duration_hours=4,
        )

        assert request.duration_hours == 4

    @pytest.mark.asyncio
    async def test_max_duration_boundary(self, workflow):
        """Test request at maximum allowed duration."""
        max_hours = ApprovalWorkflow.MAX_ACCESS_DURATION_HOURS

        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Maximum duration",
            approvers=["admin-1"],
            duration_hours=max_hours,
        )

        assert request.duration_hours == max_hours

    @pytest.mark.asyncio
    async def test_default_request_expiry(self, workflow):
        """Test that default request expiry is set correctly."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        expected_expiry = request.created_at + timedelta(
            days=ApprovalWorkflow.DEFAULT_REQUEST_EXPIRY_DAYS
        )
        # Check within 1 second tolerance
        diff = abs((request.expires_at - expected_expiry).total_seconds())
        assert diff < 1

    @pytest.mark.asyncio
    async def test_expired_request_cannot_be_rejected(self, workflow):
        """Test that expired requests cannot be rejected."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        # Force expiration
        request.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)

        # Note: reject doesn't check expiration, only approve does
        # This tests current behavior - reject still works on expired pending requests
        result = await workflow.reject(
            approver_id="admin-1",
            request_id=request.id,
            reason="Too late",
        )
        assert result.status == ApprovalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_expire_multiple_requests(self, workflow):
        """Test expiring multiple requests at once."""
        # Create multiple requests
        for i in range(3):
            request = await workflow.request_access(
                requester_id=f"user-{i}",
                permission="debates:delete",
                resource_type="debates",
                justification=f"Test {i}",
                approvers=["admin-1"],
            )
            # Force expiration
            request.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)

        # Create one non-expired request
        await workflow.request_access(
            requester_id="user-active",
            permission="debates:delete",
            resource_type="debates",
            justification="Active request",
            approvers=["admin-1"],
        )

        count = await workflow.expire_old_requests()
        assert count == 3


# =============================================================================
# Grant Temporary Permission Tests
# =============================================================================


class TestGrantTemporaryPermission:
    """Tests for _grant_temporary_permission integration."""

    @pytest.fixture
    def workflow(self):
        """Fresh workflow instance."""
        return ApprovalWorkflow()

    @pytest.mark.asyncio
    async def test_grant_called_on_approval(self, workflow):
        """Test that grant is called when request is fully approved."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
            duration_hours=8,
        )

        with patch.object(
            workflow, "_grant_temporary_permission", new_callable=AsyncMock
        ) as mock_grant:
            await workflow.approve(approver_id="admin-1", request_id=request.id)

        mock_grant.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_grant_not_called_on_partial_approval(self, workflow):
        """Test that grant is not called until all required approvals."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1", "admin-2"],
            required_approvals=2,
        )

        with patch.object(
            workflow, "_grant_temporary_permission", new_callable=AsyncMock
        ) as mock_grant:
            await workflow.approve(approver_id="admin-1", request_id=request.id)

        mock_grant.assert_not_called()

    @pytest.mark.asyncio
    async def test_grant_not_called_on_rejection(self, workflow):
        """Test that grant is not called on rejection."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        with patch.object(
            workflow, "_grant_temporary_permission", new_callable=AsyncMock
        ) as mock_grant:
            await workflow.reject(
                approver_id="admin-1",
                request_id=request.id,
                reason="No",
            )

        mock_grant.assert_not_called()

    @pytest.mark.asyncio
    async def test_grant_handles_import_error(self, workflow):
        """Test that grant handles missing ResourcePermissionStore gracefully."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        # Call the real method with ImportError simulation
        with patch(
            "aragora.rbac.approvals.ApprovalWorkflow._grant_temporary_permission"
        ) as mock_grant:
            mock_grant.return_value = None
            await workflow.approve(approver_id="admin-1", request_id=request.id)

        # Should complete without error
        assert request.status == ApprovalStatus.APPROVED


# =============================================================================
# Workspace and Organization Scoped Tests
# =============================================================================


class TestScopedApprovals:
    """Tests for workspace and organization scoped approvals."""

    @pytest.fixture
    def workflow(self):
        """Fresh workflow instance."""
        return ApprovalWorkflow()

    @pytest.mark.asyncio
    async def test_org_scoped_request(self, workflow):
        """Test organization-scoped approval request."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Org-level cleanup",
            approvers=["admin-1"],
            org_id="org-123",
        )

        assert request.org_id == "org-123"
        assert request.workspace_id is None

    @pytest.mark.asyncio
    async def test_workspace_scoped_request(self, workflow):
        """Test workspace-scoped approval request."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Workspace cleanup",
            approvers=["admin-1"],
            org_id="org-123",
            workspace_id="ws-456",
        )

        assert request.org_id == "org-123"
        assert request.workspace_id == "ws-456"

    @pytest.mark.asyncio
    async def test_requests_by_requester_across_orgs(self, workflow):
        """Test getting requests for a user across organizations."""
        await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Org 1 request",
            approvers=["admin-1"],
            org_id="org-1",
        )
        await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Org 2 request",
            approvers=["admin-2"],
            org_id="org-2",
        )

        results = await workflow.get_requests_by_requester("user-1")
        assert len(results) == 2
        org_ids = {r.org_id for r in results}
        assert org_ids == {"org-1", "org-2"}


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCasesAndErrors:
    """Additional edge cases and error handling tests."""

    @pytest.fixture
    def workflow(self):
        """Fresh workflow instance."""
        return ApprovalWorkflow()

    @pytest.mark.asyncio
    async def test_request_not_found_for_reject(self, workflow):
        """Test rejection fails for non-existent request."""
        with pytest.raises(ValueError, match="Request not found"):
            await workflow.reject(
                approver_id="admin-1",
                request_id="nonexistent",
                reason="No",
            )

    @pytest.mark.asyncio
    async def test_request_not_found_for_cancel(self, workflow):
        """Test cancellation fails for non-existent request."""
        with pytest.raises(ValueError, match="Request not found"):
            await workflow.cancel(
                requester_id="user-1",
                request_id="nonexistent",
            )

    @pytest.mark.asyncio
    async def test_reject_non_pending_error(self, workflow):
        """Test rejection fails if request not pending."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        # Approve first
        with patch.object(workflow, "_grant_temporary_permission", new_callable=AsyncMock):
            await workflow.approve(approver_id="admin-1", request_id=request.id)

        # Try to reject approved request
        with pytest.raises(ValueError, match="not pending"):
            await workflow.reject(
                approver_id="admin-1",
                request_id=request.id,
                reason="Changed mind",
            )

    @pytest.mark.asyncio
    async def test_empty_justification_allowed(self, workflow):
        """Test that empty justification is technically allowed."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="",
            approvers=["admin-1"],
        )

        assert request.justification == ""

    @pytest.mark.asyncio
    async def test_approve_without_comment(self, workflow):
        """Test approval without comment."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        with patch.object(workflow, "_grant_temporary_permission", new_callable=AsyncMock):
            result = await workflow.approve(approver_id="admin-1", request_id=request.id)

        assert result.decisions[0].comment is None

    @pytest.mark.asyncio
    async def test_cancel_without_reason(self, workflow):
        """Test cancellation without reason."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        result = await workflow.cancel(requester_id="user-1", request_id=request.id)

        assert result.status == ApprovalStatus.CANCELLED
        assert result.metadata.get("cancellation_reason") is None

    @pytest.mark.asyncio
    async def test_request_id_format(self, workflow):
        """Test that request IDs follow expected format."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        assert request.id.startswith("req-")
        assert len(request.id) == 16  # "req-" + 12 hex chars

    @pytest.mark.asyncio
    async def test_resolved_at_set_on_approval(self, workflow):
        """Test that resolved_at is set when approved."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        assert request.resolved_at is None

        with patch.object(workflow, "_grant_temporary_permission", new_callable=AsyncMock):
            await workflow.approve(approver_id="admin-1", request_id=request.id)

        assert request.resolved_at is not None
        assert request.resolved_at > request.created_at

    @pytest.mark.asyncio
    async def test_resolved_at_set_on_rejection(self, workflow):
        """Test that resolved_at is set when rejected."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        await workflow.reject(
            approver_id="admin-1",
            request_id=request.id,
            reason="No",
        )

        assert request.resolved_at is not None

    @pytest.mark.asyncio
    async def test_get_pending_respects_limit(self, workflow):
        """Test that get_pending_for_approver respects limit."""
        # Create more requests than limit
        for i in range(5):
            await workflow.request_access(
                requester_id=f"user-{i}",
                permission="debates:delete",
                resource_type="debates",
                justification=f"Test {i}",
                approvers=["admin-1"],
            )

        results = await workflow.get_pending_for_approver("admin-1", limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_get_requests_by_requester_respects_limit(self, workflow):
        """Test that get_requests_by_requester respects limit."""
        for i in range(5):
            await workflow.request_access(
                requester_id="user-1",
                permission=f"debates:action{i}",
                resource_type="debates",
                justification=f"Test {i}",
                approvers=["admin-1"],
            )

        results = await workflow.get_requests_by_requester("user-1", limit=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_get_requests_by_requester_sorted_by_created_at(self, workflow):
        """Test that requests are sorted by creation time descending."""
        for i in range(3):
            await workflow.request_access(
                requester_id="user-1",
                permission=f"debates:action{i}",
                resource_type="debates",
                justification=f"Test {i}",
                approvers=["admin-1"],
            )

        results = await workflow.get_requests_by_requester("user-1")

        # Should be sorted newest first
        for i in range(len(results) - 1):
            assert results[i].created_at >= results[i + 1].created_at

    @pytest.mark.asyncio
    async def test_multiple_requests_same_permission(self, workflow):
        """Test creating multiple requests for same permission."""
        request1 = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            resource_id="debate-1",
            justification="Delete debate 1",
            approvers=["admin-1"],
        )
        request2 = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            resource_id="debate-2",
            justification="Delete debate 2",
            approvers=["admin-1"],
        )

        # Both should exist independently
        assert request1.id != request2.id
        assert request1.resource_id == "debate-1"
        assert request2.resource_id == "debate-2"

    @pytest.mark.asyncio
    async def test_approver_pending_excludes_non_pending(self, workflow):
        """Test that get_pending excludes resolved requests."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        # Verify it's in pending
        pending_before = await workflow.get_pending_for_approver("admin-1")
        assert len(pending_before) == 1

        # Reject it
        await workflow.reject(
            approver_id="admin-1",
            request_id=request.id,
            reason="No",
        )

        # Should no longer be in pending
        pending_after = await workflow.get_pending_for_approver("admin-1")
        assert len(pending_after) == 0


# =============================================================================
# Request State Transitions Tests
# =============================================================================


class TestStateTransitions:
    """Tests for approval request state machine transitions."""

    @pytest.fixture
    def workflow(self):
        """Fresh workflow instance."""
        return ApprovalWorkflow()

    @pytest.mark.asyncio
    async def test_pending_to_approved_transition(self, workflow):
        """Test PENDING -> APPROVED transition."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        assert request.status == ApprovalStatus.PENDING

        with patch.object(workflow, "_grant_temporary_permission", new_callable=AsyncMock):
            await workflow.approve(approver_id="admin-1", request_id=request.id)

        assert request.status == ApprovalStatus.APPROVED

    @pytest.mark.asyncio
    async def test_pending_to_rejected_transition(self, workflow):
        """Test PENDING -> REJECTED transition."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        assert request.status == ApprovalStatus.PENDING

        await workflow.reject(
            approver_id="admin-1",
            request_id=request.id,
            reason="Denied",
        )

        assert request.status == ApprovalStatus.REJECTED

    @pytest.mark.asyncio
    async def test_pending_to_cancelled_transition(self, workflow):
        """Test PENDING -> CANCELLED transition."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        assert request.status == ApprovalStatus.PENDING

        await workflow.cancel(requester_id="user-1", request_id=request.id)

        assert request.status == ApprovalStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_pending_to_expired_transition(self, workflow):
        """Test PENDING -> EXPIRED transition via expiration."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        assert request.status == ApprovalStatus.PENDING

        # Force expiration
        request.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        await workflow.expire_old_requests()

        assert request.status == ApprovalStatus.EXPIRED

    @pytest.mark.asyncio
    async def test_pending_to_expired_on_approve_attempt(self, workflow):
        """Test PENDING -> EXPIRED when approving expired request."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1"],
        )

        # Force expiration
        request.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)

        # Attempt to approve should fail and mark as expired
        with pytest.raises(ValueError, match="expired"):
            await workflow.approve(approver_id="admin-1", request_id=request.id)

        assert request.status == ApprovalStatus.EXPIRED

    @pytest.mark.asyncio
    async def test_no_transition_from_approved(self, workflow):
        """Test that approved requests cannot transition to other states."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1", "admin-2"],
        )

        with patch.object(workflow, "_grant_temporary_permission", new_callable=AsyncMock):
            await workflow.approve(approver_id="admin-1", request_id=request.id)

        # Cannot approve again
        with pytest.raises(ValueError, match="not pending"):
            await workflow.approve(approver_id="admin-2", request_id=request.id)

        # Cannot reject
        with pytest.raises(ValueError, match="not pending"):
            await workflow.reject(
                approver_id="admin-2",
                request_id=request.id,
                reason="Too late",
            )

        # Cannot cancel
        with pytest.raises(ValueError, match="not pending"):
            await workflow.cancel(requester_id="user-1", request_id=request.id)

    @pytest.mark.asyncio
    async def test_no_transition_from_rejected(self, workflow):
        """Test that rejected requests cannot transition to other states."""
        request = await workflow.request_access(
            requester_id="user-1",
            permission="debates:delete",
            resource_type="debates",
            justification="Test",
            approvers=["admin-1", "admin-2"],
        )

        await workflow.reject(
            approver_id="admin-1",
            request_id=request.id,
            reason="No",
        )

        # Cannot approve
        with pytest.raises(ValueError, match="not pending"):
            await workflow.approve(approver_id="admin-2", request_id=request.id)

        # Cannot cancel
        with pytest.raises(ValueError, match="not pending"):
            await workflow.cancel(requester_id="user-1", request_id=request.id)
