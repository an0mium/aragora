"""
Tests for approval workflow and governance store integration.

Tests the human checkpoint approval workflow including:
- Approval creation and persistence
- Approval resolution
- Recovery after restart
- Timeout handling
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta

from aragora.workflow.nodes.human_checkpoint import (
    ApprovalRequest,
    ApprovalStatus,
    ChecklistItem,
    get_approval_request,
    resolve_approval,
    get_pending_approvals,
    _pending_approvals,
)


@pytest.fixture(autouse=True)
def clear_approvals():
    """Clear in-memory approvals before each test."""
    _pending_approvals.clear()
    yield
    _pending_approvals.clear()


class TestApprovalRequest:
    """Test ApprovalRequest creation and management."""

    def test_create_approval_request(self):
        """ApprovalRequest should be created with required fields."""
        request = ApprovalRequest(
            id="approval_123",
            workflow_id="wf_456",
            step_id="step_789",
            title="Review Code Changes",
            description="Please review the code changes before deployment",
            checklist=[],
        )

        assert request.id == "approval_123"
        assert request.workflow_id == "wf_456"
        assert request.step_id == "step_789"
        assert request.title == "Review Code Changes"
        assert request.status == ApprovalStatus.PENDING

    def test_approval_request_with_checklist(self):
        """ApprovalRequest should support checklist items."""
        checklist = [
            ChecklistItem(id="1", label="Review security implications", required=True),
            ChecklistItem(id="2", label="Check for breaking changes", required=True),
            ChecklistItem(id="3", label="Verify test coverage", required=False),
        ]
        request = ApprovalRequest(
            id="approval_123",
            workflow_id="wf_456",
            step_id="step_789",
            title="Code Review",
            description="Review the code",
            checklist=checklist,
        )

        assert len(request.checklist) == 3
        assert request.checklist[0].required is True
        assert request.checklist[2].required is False

    def test_approval_request_with_timeout(self):
        """ApprovalRequest should support timeout settings."""
        request = ApprovalRequest(
            id="approval_123",
            workflow_id="wf_456",
            step_id="step_789",
            title="Urgent Review",
            description="Needs review within 1 hour",
            checklist=[],
            timeout_seconds=3600.0,
        )

        assert request.timeout_seconds == 3600.0


class TestApprovalStatus:
    """Test ApprovalStatus enum."""

    def test_status_values(self):
        """ApprovalStatus should have expected values."""
        assert ApprovalStatus.PENDING.value == "pending"
        assert ApprovalStatus.APPROVED.value == "approved"
        assert ApprovalStatus.REJECTED.value == "rejected"
        assert ApprovalStatus.TIMEOUT.value == "timeout"
        assert ApprovalStatus.ESCALATED.value == "escalated"


class TestApprovalStorage:
    """Test approval storage and retrieval."""

    def test_store_approval_in_memory(self):
        """Approvals should be stored in memory."""
        request = ApprovalRequest(
            id="test_approval",
            workflow_id="wf_123",
            step_id="step_456",
            title="Test",
            description="Test approval",
            checklist=[],
        )
        _pending_approvals[request.id] = request

        assert "test_approval" in _pending_approvals
        assert _pending_approvals["test_approval"].title == "Test"

    def test_get_approval_request_from_memory(self):
        """get_approval_request should retrieve from memory."""
        request = ApprovalRequest(
            id="test_get",
            workflow_id="wf_123",
            step_id="step_456",
            title="Test Get",
            description="Test get approval",
            checklist=[],
        )
        _pending_approvals[request.id] = request

        retrieved = get_approval_request("test_get")
        assert retrieved is not None
        assert retrieved.title == "Test Get"

    def test_get_nonexistent_approval(self):
        """get_approval_request should return None for missing approvals."""
        result = get_approval_request("nonexistent_id")
        assert result is None


class TestApprovalResolution:
    """Test approval resolution."""

    def test_approve_request(self):
        """Should be able to approve a pending request."""
        request = ApprovalRequest(
            id="to_approve",
            workflow_id="wf_123",
            step_id="step_456",
            title="To Approve",
            description="Pending approval",
            checklist=[],
        )
        _pending_approvals[request.id] = request

        success = resolve_approval(
            request_id="to_approve",
            status=ApprovalStatus.APPROVED,
            responder_id="user_123",
            notes="Looks good!",
        )

        assert success is True
        assert _pending_approvals["to_approve"].status == ApprovalStatus.APPROVED

    def test_reject_request(self):
        """Should be able to reject a pending request."""
        request = ApprovalRequest(
            id="to_reject",
            workflow_id="wf_123",
            step_id="step_456",
            title="To Reject",
            description="Pending rejection",
            checklist=[],
        )
        _pending_approvals[request.id] = request

        success = resolve_approval(
            request_id="to_reject",
            status=ApprovalStatus.REJECTED,
            responder_id="user_456",
            notes="Needs more work",
        )

        assert success is True
        assert _pending_approvals["to_reject"].status == ApprovalStatus.REJECTED

    def test_resolve_nonexistent_request(self):
        """Should return False for nonexistent requests."""
        success = resolve_approval(
            request_id="nonexistent",
            status=ApprovalStatus.APPROVED,
            responder_id="user_123",
        )

        assert success is False


class TestApprovalRecovery:
    """Test approval recovery from persistent store."""

    def test_get_approval_request_recovers_from_store(self):
        """get_approval_request should recover from GovernanceStore after restart."""
        # Mock the governance store
        mock_store = MagicMock()
        mock_record = MagicMock()
        mock_record.approval_id = "recovered_123"
        mock_record.title = "Recovered Approval"
        mock_record.description = "This was recovered"
        mock_record.status = "pending"
        mock_record.timeout_seconds = 3600
        mock_record.metadata_json = '{"workflow_id": "wf_123", "step_id": "step_456"}'

        mock_store.get_approval.return_value = mock_record

        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store",
            return_value=mock_store,
        ):
            # Clear in-memory cache
            _pending_approvals.clear()

            # Should recover from store
            result = get_approval_request("recovered_123")

            # Should have called the store
            mock_store.get_approval.assert_called_once_with("recovered_123")

            # Should have recovered the request
            assert result is not None
            assert result.id == "recovered_123"
            assert result.title == "Recovered Approval"

    def test_recovered_approval_is_cached(self):
        """Recovered approvals should be cached in memory."""
        mock_store = MagicMock()
        mock_record = MagicMock()
        mock_record.approval_id = "cache_test"
        mock_record.title = "Cache Test"
        mock_record.description = "Testing cache"
        mock_record.status = "pending"
        mock_record.timeout_seconds = 3600
        mock_record.metadata_json = '{"workflow_id": "wf_123", "step_id": "step_456"}'

        mock_store.get_approval.return_value = mock_record

        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store",
            return_value=mock_store,
        ):
            _pending_approvals.clear()

            # First call - should hit store
            get_approval_request("cache_test")
            assert mock_store.get_approval.call_count == 1

            # Second call - should hit cache
            get_approval_request("cache_test")
            assert mock_store.get_approval.call_count == 1  # Not called again


class TestApprovalPersistence:
    """Test that approvals are persisted to GovernanceStore."""

    def test_resolve_updates_store(self):
        """Resolving approval should update GovernanceStore."""
        request = ApprovalRequest(
            id="persist_test",
            workflow_id="wf_123",
            step_id="step_456",
            title="Persistence Test",
            description="Testing persistence",
            checklist=[],
        )
        _pending_approvals[request.id] = request

        mock_store = MagicMock()

        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store",
            return_value=mock_store,
        ):
            resolve_approval(
                request_id="persist_test",
                status=ApprovalStatus.APPROVED,
                responder_id="user_123",
            )

            # Should have updated the store
            mock_store.update_approval_status.assert_called_once()


class TestChecklistItem:
    """Test ChecklistItem handling."""

    def test_create_checklist_item(self):
        """ChecklistItem should be created with required fields."""
        item = ChecklistItem(
            id="item_1",
            label="Review security",
            required=True,
        )

        assert item.id == "item_1"
        assert item.label == "Review security"
        assert item.required is True
        assert item.checked is False

    def test_checklist_item_defaults(self):
        """ChecklistItem should have sensible defaults."""
        item = ChecklistItem(
            id="item_2",
            label="Optional check",
            required=False,
        )

        assert item.required is False
        assert item.checked is False


class TestGetPendingApprovals:
    """Test get_pending_approvals function."""

    def test_returns_in_memory_approvals(self):
        """Should return in-memory pending approvals."""
        request1 = ApprovalRequest(
            id="pending_1",
            workflow_id="wf_1",
            step_id="step_1",
            title="Pending 1",
            description="First pending",
            checklist=[],
        )
        request2 = ApprovalRequest(
            id="pending_2",
            workflow_id="wf_2",
            step_id="step_2",
            title="Pending 2",
            description="Second pending",
            checklist=[],
        )
        _pending_approvals["pending_1"] = request1
        _pending_approvals["pending_2"] = request2

        pending = get_pending_approvals()

        assert len(pending) >= 2
        assert any(r.id == "pending_1" for r in pending)
        assert any(r.id == "pending_2" for r in pending)

    def test_returns_empty_for_no_approvals(self):
        """Should return empty list when no approvals exist."""
        _pending_approvals.clear()
        pending = get_pending_approvals()
        assert isinstance(pending, list)
        assert len(pending) == 0
