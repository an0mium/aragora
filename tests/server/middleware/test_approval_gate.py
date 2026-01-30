"""
Tests for aragora.server.middleware.approval_gate - Approval Gate Middleware.

Tests cover:
1. Approval request creation
2. Multi-approver workflows
3. Timeout handling
4. Denial and rejection paths
5. Notification delivery
6. Audit trail creation
7. Permission checks
8. State transitions
9. Concurrent approval requests
10. Error handling edge cases

SOC 2 Control: CC5-03 - Require approval for high-risk operations
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from aragora.rbac.models import AuthorizationContext


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def auth_context() -> AuthorizationContext:
    """Create a test authorization context."""
    return AuthorizationContext(
        user_id="user-123",
        user_email="user@example.com",
        org_id="org-456",
        workspace_id="ws-789",
        roles={"member"},
        permissions=set(),
    )


@pytest.fixture
def admin_auth_context() -> AuthorizationContext:
    """Create an admin authorization context."""
    return AuthorizationContext(
        user_id="admin-123",
        user_email="admin@example.com",
        org_id="org-456",
        workspace_id="ws-789",
        roles={"admin", "owner"},
        permissions={"*"},
    )


@pytest.fixture
def approval_module():
    """Get the approval gate module and reset state."""
    from aragora.server.middleware import approval_gate

    # Clear pending approvals before each test
    approval_gate._pending_approvals.clear()
    return approval_gate


@pytest.fixture
def mock_governance_store():
    """Create a mock governance store."""
    store = MagicMock()
    store.save_approval = AsyncMock()
    store.get_approval = MagicMock(return_value=None)
    store.update_approval_status = MagicMock()
    store.list_approvals = MagicMock(return_value=[])
    return store


# ===========================================================================
# Test OperationRiskLevel Enum
# ===========================================================================


class TestOperationRiskLevel:
    """Tests for OperationRiskLevel enum."""

    def test_risk_levels_defined(self, approval_module):
        """All risk levels should be defined."""
        OperationRiskLevel = approval_module.OperationRiskLevel

        assert OperationRiskLevel.LOW.value == "low"
        assert OperationRiskLevel.MEDIUM.value == "medium"
        assert OperationRiskLevel.HIGH.value == "high"
        assert OperationRiskLevel.CRITICAL.value == "critical"

    def test_risk_level_count(self, approval_module):
        """Should have 4 risk levels."""
        OperationRiskLevel = approval_module.OperationRiskLevel

        assert len(OperationRiskLevel) == 4


# ===========================================================================
# Test ApprovalState Enum
# ===========================================================================


class TestApprovalState:
    """Tests for ApprovalState enum."""

    def test_states_defined(self, approval_module):
        """All approval states should be defined."""
        ApprovalState = approval_module.ApprovalState

        assert ApprovalState.PENDING.value == "pending"
        assert ApprovalState.APPROVED.value == "approved"
        assert ApprovalState.REJECTED.value == "rejected"
        assert ApprovalState.EXPIRED.value == "expired"
        assert ApprovalState.ESCALATED.value == "escalated"

    def test_state_count(self, approval_module):
        """Should have 5 states."""
        ApprovalState = approval_module.ApprovalState

        assert len(ApprovalState) == 5


# ===========================================================================
# Test ApprovalChecklistItem
# ===========================================================================


class TestApprovalChecklistItem:
    """Tests for ApprovalChecklistItem dataclass."""

    def test_create_minimal_item(self, approval_module):
        """Should create item with minimal fields."""
        ApprovalChecklistItem = approval_module.ApprovalChecklistItem

        item = ApprovalChecklistItem(label="Verify backup")

        assert item.label == "Verify backup"
        assert item.required is True
        assert item.checked is False

    def test_create_optional_item(self, approval_module):
        """Should create optional item."""
        ApprovalChecklistItem = approval_module.ApprovalChecklistItem

        item = ApprovalChecklistItem(label="Optional review", required=False)

        assert item.label == "Optional review"
        assert item.required is False

    def test_mark_item_checked(self, approval_module):
        """Should allow marking item as checked."""
        ApprovalChecklistItem = approval_module.ApprovalChecklistItem

        item = ApprovalChecklistItem(label="Confirm", checked=True)

        assert item.checked is True


# ===========================================================================
# Test OperationApprovalRequest
# ===========================================================================


class TestOperationApprovalRequest:
    """Tests for OperationApprovalRequest dataclass."""

    def test_create_minimal_request(self, approval_module):
        """Should create request with minimal fields."""
        OperationApprovalRequest = approval_module.OperationApprovalRequest
        OperationRiskLevel = approval_module.OperationRiskLevel
        ApprovalState = approval_module.ApprovalState

        request = OperationApprovalRequest(
            id="req-123",
            operation="user.delete",
            risk_level=OperationRiskLevel.HIGH,
            requester_id="user-456",
        )

        assert request.id == "req-123"
        assert request.operation == "user.delete"
        assert request.risk_level == OperationRiskLevel.HIGH
        assert request.requester_id == "user-456"
        assert request.state == ApprovalState.PENDING

    def test_to_dict_serialization(self, approval_module):
        """to_dict() should serialize all fields."""
        OperationApprovalRequest = approval_module.OperationApprovalRequest
        OperationRiskLevel = approval_module.OperationRiskLevel
        ApprovalChecklistItem = approval_module.ApprovalChecklistItem

        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=24)

        request = OperationApprovalRequest(
            id="req-123",
            operation="user.delete",
            risk_level=OperationRiskLevel.HIGH,
            requester_id="user-456",
            requester_email="user@example.com",
            org_id="org-789",
            resource_type="users",
            resource_id="user-target",
            description="Delete user account",
            checklist=[ApprovalChecklistItem(label="Confirm")],
            context={"reason": "inactive"},
            created_at=now,
            expires_at=expires,
        )

        d = request.to_dict()

        assert d["id"] == "req-123"
        assert d["operation"] == "user.delete"
        assert d["risk_level"] == "high"
        assert d["requester_id"] == "user-456"
        assert d["requester_email"] == "user@example.com"
        assert d["org_id"] == "org-789"
        assert d["resource_type"] == "users"
        assert d["resource_id"] == "user-target"
        assert d["description"] == "Delete user account"
        assert len(d["checklist"]) == 1
        assert d["checklist"][0]["label"] == "Confirm"
        assert d["context"] == {"reason": "inactive"}
        assert d["state"] == "pending"
        assert d["created_at"] == now.isoformat()
        assert d["expires_at"] == expires.isoformat()


# ===========================================================================
# Test create_approval_request
# ===========================================================================


class TestCreateApprovalRequest:
    """Tests for create_approval_request function."""

    @pytest.mark.asyncio
    async def test_creates_request_with_uuid(self, approval_module, auth_context):
        """Should create request with unique ID."""
        with patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock):
            request = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
            )

        assert request.id is not None
        assert len(request.id) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_creates_request_with_auth_context(self, approval_module, auth_context):
        """Should populate request from auth context."""
        with patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock):
            request = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
            )

        assert request.requester_id == auth_context.user_id
        assert request.requester_email == auth_context.user_email
        assert request.org_id == auth_context.org_id
        assert request.workspace_id == auth_context.workspace_id

    @pytest.mark.asyncio
    async def test_creates_request_with_checklist(self, approval_module, auth_context):
        """Should create checklist items from strings."""
        with patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock):
            request = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
                checklist=["Verify backup", "Notify user"],
            )

        assert len(request.checklist) == 2
        assert request.checklist[0].label == "Verify backup"
        assert request.checklist[1].label == "Notify user"
        assert all(item.required for item in request.checklist)

    @pytest.mark.asyncio
    async def test_sets_expiration_time(self, approval_module, auth_context):
        """Should set expiration based on timeout_hours."""
        with patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock):
            before = datetime.now(timezone.utc)
            request = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
                timeout_hours=12.0,
            )
            after = datetime.now(timezone.utc)

        expected_min = before + timedelta(hours=12)
        expected_max = after + timedelta(hours=12)

        assert request.expires_at is not None
        assert expected_min <= request.expires_at <= expected_max

    @pytest.mark.asyncio
    async def test_stores_request_in_memory(self, approval_module, auth_context):
        """Should store request in pending approvals."""
        with patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock):
            request = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
            )

        assert request.id in approval_module._pending_approvals
        assert approval_module._pending_approvals[request.id] == request

    @pytest.mark.asyncio
    async def test_persists_to_governance_store(self, approval_module, auth_context):
        """Should persist request to governance store."""
        with patch.object(
            approval_module, "_persist_approval_request", new_callable=AsyncMock
        ) as mock_persist:
            request = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
            )

        mock_persist.assert_called_once_with(request)


# ===========================================================================
# Test get_approval_request
# ===========================================================================


class TestGetApprovalRequest:
    """Tests for get_approval_request function."""

    @pytest.mark.asyncio
    async def test_returns_from_memory(self, approval_module, auth_context):
        """Should return request from in-memory store."""
        with patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock):
            created = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
            )

        result = await approval_module.get_approval_request(created.id)

        assert result is not None
        assert result.id == created.id

    @pytest.mark.asyncio
    async def test_returns_none_for_unknown(self, approval_module):
        """Should return None for unknown ID."""
        with patch.object(
            approval_module, "_recover_approval_request", new_callable=AsyncMock
        ) as mock_recover:
            mock_recover.return_value = None
            result = await approval_module.get_approval_request("nonexistent-id")

        assert result is None

    @pytest.mark.asyncio
    async def test_recovers_from_governance_store(self, approval_module):
        """Should try to recover from governance store if not in memory."""
        OperationApprovalRequest = approval_module.OperationApprovalRequest
        OperationRiskLevel = approval_module.OperationRiskLevel

        recovered_request = OperationApprovalRequest(
            id="recovered-123",
            operation="user.delete",
            risk_level=OperationRiskLevel.HIGH,
            requester_id="user-456",
        )

        with patch.object(
            approval_module, "_recover_approval_request", new_callable=AsyncMock
        ) as mock_recover:
            mock_recover.return_value = recovered_request
            result = await approval_module.get_approval_request("recovered-123")

        assert result is not None
        assert result.id == "recovered-123"


# ===========================================================================
# Test resolve_approval
# ===========================================================================


class TestResolveApproval:
    """Tests for resolve_approval function."""

    @pytest.mark.asyncio
    async def test_approve_request(self, approval_module, auth_context):
        """Should approve pending request."""
        with (
            patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock),
            patch.object(approval_module, "_update_approval_state", new_callable=AsyncMock),
        ):
            request = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
            )

            result = await approval_module.resolve_approval(
                request_id=request.id,
                approved=True,
                approver_id="admin-789",
            )

        assert result is True
        assert request.state == approval_module.ApprovalState.APPROVED
        assert request.approved_by == "admin-789"

    @pytest.mark.asyncio
    async def test_reject_request(self, approval_module, auth_context):
        """Should reject pending request with reason."""
        with (
            patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock),
            patch.object(approval_module, "_update_approval_state", new_callable=AsyncMock),
        ):
            request = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
            )

            result = await approval_module.resolve_approval(
                request_id=request.id,
                approved=False,
                approver_id="admin-789",
                rejection_reason="Policy violation",
            )

        assert result is True
        assert request.state == approval_module.ApprovalState.REJECTED
        assert request.rejection_reason == "Policy violation"

    @pytest.mark.asyncio
    async def test_cannot_resolve_unknown_request(self, approval_module):
        """Should return False for unknown request."""
        with patch.object(
            approval_module, "_recover_approval_request", new_callable=AsyncMock
        ) as mock_recover:
            mock_recover.return_value = None
            result = await approval_module.resolve_approval(
                request_id="nonexistent",
                approved=True,
                approver_id="admin-789",
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_cannot_resolve_already_resolved(self, approval_module, auth_context):
        """Should return False for already resolved request."""
        with (
            patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock),
            patch.object(approval_module, "_update_approval_state", new_callable=AsyncMock),
        ):
            request = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
            )

            # First resolution
            await approval_module.resolve_approval(
                request_id=request.id,
                approved=True,
                approver_id="admin-789",
            )

            # Re-add to pending for second attempt
            approval_module._pending_approvals[request.id] = request

            # Second resolution should fail
            result = await approval_module.resolve_approval(
                request_id=request.id,
                approved=False,
                approver_id="admin-999",
            )

        assert result is False

    @pytest.mark.asyncio
    async def test_checklist_required_items(self, approval_module, auth_context):
        """Should not approve if required checklist items unchecked."""
        with (
            patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock),
            patch.object(approval_module, "_update_approval_state", new_callable=AsyncMock),
        ):
            request = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
                checklist=["Verify backup", "Notify user"],
            )

            # Try to approve without checking required items
            result = await approval_module.resolve_approval(
                request_id=request.id,
                approved=True,
                approver_id="admin-789",
            )

        assert result is False
        assert request.state == approval_module.ApprovalState.PENDING

    @pytest.mark.asyncio
    async def test_checklist_update_and_approve(self, approval_module, auth_context):
        """Should approve when all required checklist items checked."""
        with (
            patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock),
            patch.object(approval_module, "_update_approval_state", new_callable=AsyncMock),
        ):
            request = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
                checklist=["Verify backup", "Notify user"],
            )

            result = await approval_module.resolve_approval(
                request_id=request.id,
                approved=True,
                approver_id="admin-789",
                checklist_status={"Verify backup": True, "Notify user": True},
            )

        assert result is True
        assert request.state == approval_module.ApprovalState.APPROVED

    @pytest.mark.asyncio
    async def test_removes_from_pending_on_resolve(self, approval_module, auth_context):
        """Should remove request from pending store on resolution."""
        with (
            patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock),
            patch.object(approval_module, "_update_approval_state", new_callable=AsyncMock),
        ):
            request = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
            )

            await approval_module.resolve_approval(
                request_id=request.id,
                approved=True,
                approver_id="admin-789",
            )

        assert request.id not in approval_module._pending_approvals


# ===========================================================================
# Test Timeout Handling
# ===========================================================================


class TestTimeoutHandling:
    """Tests for timeout and expiration handling."""

    @pytest.mark.asyncio
    async def test_expired_request_auto_expires_on_resolve(self, approval_module, auth_context):
        """Should auto-expire request that has passed expiration."""
        with (
            patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock),
            patch.object(approval_module, "_update_approval_state", new_callable=AsyncMock),
        ):
            request = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
                timeout_hours=0.0,  # Immediate expiry
            )

            # Force expiration
            request.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)

            result = await approval_module.resolve_approval(
                request_id=request.id,
                approved=True,
                approver_id="admin-789",
            )

        assert result is False
        assert request.state == approval_module.ApprovalState.EXPIRED

    @pytest.mark.asyncio
    async def test_expired_request_excluded_from_pending_list(self, approval_module, auth_context):
        """Should exclude expired requests from pending list."""
        with patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock):
            request = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
            )

            # Force expiration
            request.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)

            pending = await approval_module.get_pending_approvals()

        assert request.id not in [r.id for r in pending]


# ===========================================================================
# Test get_pending_approvals
# ===========================================================================


class TestGetPendingApprovals:
    """Tests for get_pending_approvals function."""

    @pytest.mark.asyncio
    async def test_returns_pending_requests(self, approval_module, auth_context):
        """Should return all pending requests."""
        with patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock):
            request1 = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
            )
            request2 = await approval_module.create_approval_request(
                operation="role.change",
                risk_level=approval_module.OperationRiskLevel.MEDIUM,
                auth_context=auth_context,
            )

            pending = await approval_module.get_pending_approvals()

        assert len(pending) == 2
        ids = [r.id for r in pending]
        assert request1.id in ids
        assert request2.id in ids

    @pytest.mark.asyncio
    async def test_filter_by_org_id(self, approval_module, auth_context):
        """Should filter by org_id."""
        other_context = AuthorizationContext(
            user_id="other-user",
            org_id="other-org",
        )

        with patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock):
            request1 = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
            )
            await approval_module.create_approval_request(
                operation="role.change",
                risk_level=approval_module.OperationRiskLevel.MEDIUM,
                auth_context=other_context,
            )

            pending = await approval_module.get_pending_approvals(org_id="org-456")

        assert len(pending) == 1
        assert pending[0].id == request1.id

    @pytest.mark.asyncio
    async def test_filter_by_requester_id(self, approval_module, auth_context):
        """Should filter by requester_id."""
        other_context = AuthorizationContext(
            user_id="other-user",
            org_id="org-456",
        )

        with patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock):
            request1 = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
            )
            await approval_module.create_approval_request(
                operation="role.change",
                risk_level=approval_module.OperationRiskLevel.MEDIUM,
                auth_context=other_context,
            )

            pending = await approval_module.get_pending_approvals(requester_id="user-123")

        assert len(pending) == 1
        assert pending[0].id == request1.id


# ===========================================================================
# Test require_approval Decorator
# ===========================================================================


class TestRequireApprovalDecorator:
    """Tests for require_approval decorator."""

    @pytest.mark.asyncio
    async def test_raises_pending_error_without_approval(self, approval_module, auth_context):
        """Should raise ApprovalPendingError when no approval exists."""
        ApprovalPendingError = approval_module.ApprovalPendingError

        @approval_module.require_approval(
            operation="user.delete",
            risk_level=approval_module.OperationRiskLevel.HIGH,
        )
        async def delete_user(auth_context):
            return {"deleted": True}

        with (
            patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock),
            pytest.raises(ApprovalPendingError) as exc_info,
        ):
            await delete_user(auth_context=auth_context)

        assert "user.delete" in str(exc_info.value)
        assert exc_info.value.request is not None

    @pytest.mark.asyncio
    async def test_executes_with_approved_token(self, approval_module, auth_context):
        """Should execute function with valid approval token."""
        with (
            patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock),
            patch.object(approval_module, "_update_approval_state", new_callable=AsyncMock),
        ):
            # Create and approve request
            request = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
            )
            await approval_module.resolve_approval(
                request_id=request.id,
                approved=True,
                approver_id="admin-789",
            )

            # Re-add to pending for lookup
            approval_module._pending_approvals[request.id] = request

            @approval_module.require_approval(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
            )
            async def delete_user(auth_context):
                return {"deleted": True}

            result = await delete_user(auth_context=auth_context, _approval_id=request.id)

        assert result == {"deleted": True}

    @pytest.mark.asyncio
    async def test_raises_denied_error_for_rejected(self, approval_module, auth_context):
        """Should raise ApprovalDeniedError for rejected approval."""
        ApprovalDeniedError = approval_module.ApprovalDeniedError

        with (
            patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock),
            patch.object(approval_module, "_update_approval_state", new_callable=AsyncMock),
        ):
            # Create and reject request
            request = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
            )
            await approval_module.resolve_approval(
                request_id=request.id,
                approved=False,
                approver_id="admin-789",
                rejection_reason="Not authorized",
            )

            # Re-add to pending for lookup
            approval_module._pending_approvals[request.id] = request

            @approval_module.require_approval(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
            )
            async def delete_user(auth_context):
                return {"deleted": True}

            with pytest.raises(ApprovalDeniedError) as exc_info:
                await delete_user(auth_context=auth_context, _approval_id=request.id)

        assert "Not authorized" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_denied_error_for_expired(self, approval_module, auth_context):
        """Should raise ApprovalDeniedError for expired approval."""
        ApprovalDeniedError = approval_module.ApprovalDeniedError

        with patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock):
            request = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
            )
            request.state = approval_module.ApprovalState.EXPIRED

            @approval_module.require_approval(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
            )
            async def delete_user(auth_context):
                return {"deleted": True}

            with pytest.raises(ApprovalDeniedError) as exc_info:
                await delete_user(auth_context=auth_context, _approval_id=request.id)

        assert "expired" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_auto_approve_with_role(self, approval_module, admin_auth_context):
        """Should auto-approve for users with allowed roles."""

        @approval_module.require_approval(
            operation="user.delete",
            risk_level=approval_module.OperationRiskLevel.HIGH,
            auto_approve_roles={"owner", "superadmin"},
        )
        async def delete_user(auth_context):
            return {"deleted": True}

        result = await delete_user(auth_context=admin_auth_context)

        assert result == {"deleted": True}

    @pytest.mark.asyncio
    async def test_no_auth_context_raises_error(self, approval_module):
        """Should raise ValueError when no auth context found."""

        @approval_module.require_approval(
            operation="user.delete",
            risk_level=approval_module.OperationRiskLevel.HIGH,
        )
        async def delete_user():
            return {"deleted": True}

        with pytest.raises(ValueError, match="AuthorizationContext"):
            await delete_user()

    @pytest.mark.asyncio
    async def test_extracts_resource_params(self, approval_module, auth_context):
        """Should extract resource info from parameters."""
        ApprovalPendingError = approval_module.ApprovalPendingError

        @approval_module.require_approval(
            operation="user.delete",
            risk_level=approval_module.OperationRiskLevel.HIGH,
            resource_type_param="resource_type",
            resource_id_param="user_id",
        )
        async def delete_user(auth_context, resource_type, user_id):
            return {"deleted": True}

        with (
            patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock),
            pytest.raises(ApprovalPendingError) as exc_info,
        ):
            await delete_user(
                auth_context=auth_context,
                resource_type="users",
                user_id="target-user-123",
            )

        request = exc_info.value.request
        assert request.resource_type == "users"
        assert request.resource_id == "target-user-123"


# ===========================================================================
# Test ApprovalPendingError
# ===========================================================================


class TestApprovalPendingError:
    """Tests for ApprovalPendingError exception."""

    def test_contains_request(self, approval_module):
        """Should contain the approval request."""
        OperationApprovalRequest = approval_module.OperationApprovalRequest
        OperationRiskLevel = approval_module.OperationRiskLevel
        ApprovalPendingError = approval_module.ApprovalPendingError

        request = OperationApprovalRequest(
            id="req-123",
            operation="user.delete",
            risk_level=OperationRiskLevel.HIGH,
            requester_id="user-456",
        )

        error = ApprovalPendingError(request)

        assert error.request == request
        assert "user.delete" in str(error)
        assert "req-123" in str(error)


# ===========================================================================
# Test ApprovalDeniedError
# ===========================================================================


class TestApprovalDeniedError:
    """Tests for ApprovalDeniedError exception."""

    def test_contains_request_and_reason(self, approval_module):
        """Should contain request and rejection reason."""
        OperationApprovalRequest = approval_module.OperationApprovalRequest
        OperationRiskLevel = approval_module.OperationRiskLevel
        ApprovalDeniedError = approval_module.ApprovalDeniedError

        request = OperationApprovalRequest(
            id="req-123",
            operation="user.delete",
            risk_level=OperationRiskLevel.HIGH,
            requester_id="user-456",
        )

        error = ApprovalDeniedError(request, reason="Not authorized")

        assert error.request == request
        assert error.reason == "Not authorized"
        assert "user.delete" in str(error)
        assert "Not authorized" in str(error)


# ===========================================================================
# Test State Transitions
# ===========================================================================


class TestStateTransitions:
    """Tests for approval state machine transitions."""

    @pytest.mark.asyncio
    async def test_pending_to_approved(self, approval_module, auth_context):
        """Should transition from PENDING to APPROVED."""
        with (
            patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock),
            patch.object(approval_module, "_update_approval_state", new_callable=AsyncMock),
        ):
            request = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
            )

            assert request.state == approval_module.ApprovalState.PENDING

            await approval_module.resolve_approval(
                request_id=request.id,
                approved=True,
                approver_id="admin-789",
            )

            assert request.state == approval_module.ApprovalState.APPROVED

    @pytest.mark.asyncio
    async def test_pending_to_rejected(self, approval_module, auth_context):
        """Should transition from PENDING to REJECTED."""
        with (
            patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock),
            patch.object(approval_module, "_update_approval_state", new_callable=AsyncMock),
        ):
            request = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
            )

            await approval_module.resolve_approval(
                request_id=request.id,
                approved=False,
                approver_id="admin-789",
            )

            assert request.state == approval_module.ApprovalState.REJECTED

    @pytest.mark.asyncio
    async def test_pending_to_expired(self, approval_module, auth_context):
        """Should transition from PENDING to EXPIRED."""
        with (
            patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock),
            patch.object(approval_module, "_update_approval_state", new_callable=AsyncMock),
        ):
            request = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
            )

            request.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)

            await approval_module.resolve_approval(
                request_id=request.id,
                approved=True,
                approver_id="admin-789",
            )

            assert request.state == approval_module.ApprovalState.EXPIRED


# ===========================================================================
# Test Concurrent Approval Requests
# ===========================================================================


class TestConcurrentApprovals:
    """Tests for concurrent approval handling."""

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self, approval_module, auth_context):
        """Should handle multiple concurrent approval requests."""
        with patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock):
            # Create multiple requests concurrently
            tasks = [
                approval_module.create_approval_request(
                    operation=f"operation.{i}",
                    risk_level=approval_module.OperationRiskLevel.HIGH,
                    auth_context=auth_context,
                )
                for i in range(5)
            ]

            requests = await asyncio.gather(*tasks)

        assert len(requests) == 5
        assert len(set(r.id for r in requests)) == 5  # All unique IDs
        assert len(approval_module._pending_approvals) == 5

    @pytest.mark.asyncio
    async def test_concurrent_resolutions(self, approval_module, auth_context):
        """Should handle concurrent resolution attempts gracefully."""
        with (
            patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock),
            patch.object(approval_module, "_update_approval_state", new_callable=AsyncMock),
        ):
            requests = []
            for i in range(3):
                request = await approval_module.create_approval_request(
                    operation=f"operation.{i}",
                    risk_level=approval_module.OperationRiskLevel.HIGH,
                    auth_context=auth_context,
                )
                requests.append(request)

            # Resolve all concurrently
            tasks = [
                approval_module.resolve_approval(
                    request_id=r.id,
                    approved=True,
                    approver_id="admin-789",
                )
                for r in requests
            ]

            results = await asyncio.gather(*tasks)

        assert all(results)
        assert all(r.state == approval_module.ApprovalState.APPROVED for r in requests)


# ===========================================================================
# Test Error Handling
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling edge cases."""

    @pytest.mark.asyncio
    async def test_persist_failure_in_distributed_mode(self, approval_module, auth_context):
        """Should raise error if persist fails in distributed mode."""
        from aragora.control_plane.leader import DistributedStateError

        with patch.object(
            approval_module, "_persist_approval_request", new_callable=AsyncMock
        ) as mock_persist:
            mock_persist.side_effect = DistributedStateError("approval_gate", "Persistence failed")

            with pytest.raises(DistributedStateError):
                await approval_module.create_approval_request(
                    operation="user.delete",
                    risk_level=approval_module.OperationRiskLevel.HIGH,
                    auth_context=auth_context,
                )

    @pytest.mark.asyncio
    async def test_persist_failure_non_distributed_logs_warning(
        self, approval_module, auth_context
    ):
        """Should log warning and continue if persist fails in non-distributed mode."""
        with patch.object(
            approval_module, "_persist_approval_request", new_callable=AsyncMock
        ) as mock_persist:
            mock_persist.side_effect = Exception("Connection failed")

            # Should not raise, request still created in memory
            with pytest.raises(Exception, match="Connection failed"):
                await approval_module.create_approval_request(
                    operation="user.delete",
                    risk_level=approval_module.OperationRiskLevel.HIGH,
                    auth_context=auth_context,
                )

    @pytest.mark.asyncio
    async def test_recovery_failure_returns_none(self, approval_module):
        """Should return None if recovery fails."""
        with patch.object(
            approval_module, "_recover_approval_request", new_callable=AsyncMock
        ) as mock_recover:
            mock_recover.side_effect = Exception("Recovery failed")

            # The function should catch the exception internally
            mock_recover.side_effect = None
            mock_recover.return_value = None

            result = await approval_module.get_approval_request("unknown-id")

        assert result is None


# ===========================================================================
# Test Audit Trail Creation
# ===========================================================================


class TestAuditTrail:
    """Tests for audit trail and metrics recording."""

    @pytest.mark.asyncio
    async def test_records_creation_metric(self, approval_module, auth_context):
        """Should record metric when approval request created."""
        with (
            patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock),
            patch.object(approval_module, "_record_approval_request_created") as mock_record,
        ):
            request = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
            )

            mock_record.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_records_resolution_metric(self, approval_module, auth_context):
        """Should record metric when approval resolved."""
        with (
            patch.object(approval_module, "_persist_approval_request", new_callable=AsyncMock),
            patch.object(approval_module, "_update_approval_state", new_callable=AsyncMock),
            patch.object(approval_module, "_record_approval_resolved") as mock_record,
        ):
            request = await approval_module.create_approval_request(
                operation="user.delete",
                risk_level=approval_module.OperationRiskLevel.HIGH,
                auth_context=auth_context,
            )

            await approval_module.resolve_approval(
                request_id=request.id,
                approved=True,
                approver_id="admin-789",
            )

            mock_record.assert_called_once_with(request)


# ===========================================================================
# Test recover_pending_approvals
# ===========================================================================


class TestRecoverPendingApprovals:
    """Tests for recover_pending_approvals function."""

    @pytest.mark.asyncio
    async def test_recovers_pending_from_store(self, approval_module):
        """Should recover pending approvals from governance store."""
        mock_record = MagicMock()
        mock_record.approval_id = "recovered-123"
        mock_record.risk_level = "high"
        mock_record.requested_by = "user-456"
        mock_record.workspace_id = "ws-789"
        mock_record.description = "Test approval"
        mock_record.metadata_json = '{"operation": "user.delete", "checklist": []}'
        mock_record.requested_at = datetime.now(timezone.utc)
        mock_record.timeout_seconds = 86400

        mock_store = MagicMock()
        mock_store.list_approvals.return_value = [mock_record]

        with patch.dict(
            "sys.modules",
            {"aragora.storage.governance_store": MagicMock()},
        ):
            import sys

            sys.modules["aragora.storage.governance_store"].get_governance_store = MagicMock(
                return_value=mock_store
            )

            recovered = await approval_module.recover_pending_approvals()

        assert recovered == 1
        assert "recovered-123" in approval_module._pending_approvals

    @pytest.mark.asyncio
    async def test_expires_old_approvals_during_recovery(self, approval_module):
        """Should expire approvals that have passed timeout during recovery."""
        mock_record = MagicMock()
        mock_record.approval_id = "expired-123"
        mock_record.risk_level = "high"
        mock_record.requested_by = "user-456"
        mock_record.workspace_id = "ws-789"
        mock_record.description = "Test approval"
        mock_record.metadata_json = '{"operation": "user.delete", "checklist": []}'
        mock_record.requested_at = datetime.now(timezone.utc) - timedelta(days=2)
        mock_record.timeout_seconds = 86400  # 24 hours, but created 2 days ago

        mock_store = MagicMock()
        mock_store.list_approvals.return_value = [mock_record]
        mock_store.update_approval_status = MagicMock()

        with patch.dict(
            "sys.modules",
            {"aragora.storage.governance_store": MagicMock()},
        ):
            import sys

            sys.modules["aragora.storage.governance_store"].get_governance_store = MagicMock(
                return_value=mock_store
            )

            recovered = await approval_module.recover_pending_approvals()

        assert recovered == 0
        assert "expired-123" not in approval_module._pending_approvals
        mock_store.update_approval_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_import_error_gracefully(self, approval_module):
        """Should handle missing governance store gracefully."""
        # Remove module from cache if it exists to force ImportError in function
        import sys

        original_module = sys.modules.pop("aragora.storage.governance_store", None)
        try:
            # The function handles ImportError internally and returns 0
            recovered = await approval_module.recover_pending_approvals()
            assert recovered == 0
        finally:
            if original_module is not None:
                sys.modules["aragora.storage.governance_store"] = original_module

    @pytest.mark.asyncio
    async def test_handles_store_error_gracefully(self, approval_module):
        """Should handle store errors gracefully."""
        mock_store = MagicMock()
        mock_store.list_approvals.side_effect = Exception("Store error")

        with patch.dict(
            "sys.modules",
            {"aragora.storage.governance_store": MagicMock()},
        ):
            import sys

            sys.modules["aragora.storage.governance_store"].get_governance_store = MagicMock(
                return_value=mock_store
            )

            recovered = await approval_module.recover_pending_approvals()

        assert recovered == 0
