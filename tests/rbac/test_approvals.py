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
