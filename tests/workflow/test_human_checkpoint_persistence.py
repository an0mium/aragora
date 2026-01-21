"""
Tests for human checkpoint persistence functionality.

Tests cover:
- Approval persistence to GovernanceStore
- Approval status updates persist
- Pending approvals merge in-memory + persisted
- Graceful degradation when store unavailable
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock, AsyncMock

from aragora.workflow.nodes.human_checkpoint import (
    ApprovalRequest,
    ApprovalStatus,
    ChecklistItem,
    HumanCheckpointStep,
    resolve_approval,
    get_pending_approvals,
    get_approval_request,
    _pending_approvals,
    _get_governance_store,
)


class TestApprovalPersistence:
    """Tests for approval persistence to GovernanceStore."""

    @pytest.fixture(autouse=True)
    def clear_approvals(self):
        """Clear in-memory approvals before each test."""
        _pending_approvals.clear()
        yield
        _pending_approvals.clear()

    def test_resolve_approval_persists_to_store(self):
        """resolve_approval updates GovernanceStore."""
        # Create an in-memory approval
        request = ApprovalRequest(
            id="apr_test123",
            workflow_id="wf_1",
            step_id="step_1",
            title="Test Approval",
            description="Test description",
            checklist=[],
        )
        _pending_approvals[request.id] = request

        mock_store = MagicMock()
        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store",
            return_value=mock_store
        ):
            result = resolve_approval(
                request_id="apr_test123",
                status=ApprovalStatus.APPROVED,
                responder_id="user-1",
                notes="Looks good",
            )

        assert result is True
        mock_store.update_approval_status.assert_called_once_with(
            approval_id="apr_test123",
            status="approved",
            approved_by="user-1",
            rejection_reason=None,
        )

    def test_resolve_rejection_includes_reason(self):
        """Rejection notes are saved as rejection_reason."""
        request = ApprovalRequest(
            id="apr_reject",
            workflow_id="wf_1",
            step_id="step_1",
            title="Test",
            description="Test",
            checklist=[],
        )
        _pending_approvals[request.id] = request

        mock_store = MagicMock()
        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store",
            return_value=mock_store
        ):
            resolve_approval(
                request_id="apr_reject",
                status=ApprovalStatus.REJECTED,
                responder_id="user-1",
                notes="Does not meet requirements",
            )

        mock_store.update_approval_status.assert_called_once_with(
            approval_id="apr_reject",
            status="rejected",
            approved_by=None,
            rejection_reason="Does not meet requirements",
        )

    def test_resolve_handles_store_error(self):
        """Store errors don't break resolve_approval."""
        request = ApprovalRequest(
            id="apr_error",
            workflow_id="wf_1",
            step_id="step_1",
            title="Test",
            description="Test",
            checklist=[],
        )
        _pending_approvals[request.id] = request

        mock_store = MagicMock()
        mock_store.update_approval_status.side_effect = Exception("DB error")

        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store",
            return_value=mock_store
        ):
            # Should not raise
            result = resolve_approval(
                request_id="apr_error",
                status=ApprovalStatus.APPROVED,
                responder_id="user-1",
            )

        assert result is True
        assert _pending_approvals["apr_error"].status == ApprovalStatus.APPROVED


class TestPendingApprovalsMerge:
    """Tests for merging in-memory and persisted approvals."""

    @pytest.fixture(autouse=True)
    def clear_approvals(self):
        """Clear in-memory approvals before each test."""
        _pending_approvals.clear()
        yield
        _pending_approvals.clear()

    def test_get_pending_includes_in_memory(self):
        """get_pending_approvals returns in-memory approvals."""
        request = ApprovalRequest(
            id="apr_mem",
            workflow_id="wf_1",
            step_id="step_1",
            title="In Memory",
            description="Test",
            checklist=[],
            status=ApprovalStatus.PENDING,
        )
        _pending_approvals[request.id] = request

        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store",
            return_value=None
        ):
            result = get_pending_approvals()

        assert len(result) == 1
        assert result[0].id == "apr_mem"

    def test_get_pending_merges_persisted(self):
        """get_pending_approvals merges persisted approvals."""
        # In-memory approval
        mem_request = ApprovalRequest(
            id="apr_mem",
            workflow_id="wf_1",
            step_id="step_1",
            title="In Memory",
            description="Test",
            checklist=[],
            status=ApprovalStatus.PENDING,
        )
        _pending_approvals[mem_request.id] = mem_request

        # Mock persisted approval
        mock_record = MagicMock()
        mock_record.approval_id = "apr_persisted"
        mock_record.title = "Persisted"
        mock_record.description = "From DB"
        mock_record.status = "pending"
        mock_record.timeout_seconds = 3600
        mock_record.metadata_json = '{"workflow_id": "wf_2"}'

        mock_store = MagicMock()
        mock_store.list_approvals.return_value = [mock_record]

        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store",
            return_value=mock_store
        ):
            result = get_pending_approvals()

        assert len(result) == 2
        ids = {r.id for r in result}
        assert "apr_mem" in ids
        assert "apr_persisted" in ids

    def test_get_pending_deduplicates(self):
        """Approvals in both stores are not duplicated."""
        # Same ID in memory and persisted
        mem_request = ApprovalRequest(
            id="apr_both",
            workflow_id="wf_1",
            step_id="step_1",
            title="In Memory",
            description="Test",
            checklist=[],
            status=ApprovalStatus.PENDING,
        )
        _pending_approvals[mem_request.id] = mem_request

        mock_record = MagicMock()
        mock_record.approval_id = "apr_both"  # Same ID
        mock_record.title = "Persisted"
        mock_record.description = "From DB"
        mock_record.status = "pending"
        mock_record.timeout_seconds = 3600
        mock_record.metadata_json = None

        mock_store = MagicMock()
        mock_store.list_approvals.return_value = [mock_record]

        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store",
            return_value=mock_store
        ):
            result = get_pending_approvals()

        # Should only have one entry
        assert len(result) == 1
        assert result[0].id == "apr_both"
        # In-memory version takes precedence
        assert result[0].title == "In Memory"

    def test_get_pending_filters_by_workflow(self):
        """get_pending_approvals filters by workflow_id."""
        for i, wf_id in enumerate(["wf_1", "wf_1", "wf_2"]):
            request = ApprovalRequest(
                id=f"apr_{i}",
                workflow_id=wf_id,
                step_id="step_1",
                title=f"Request {i}",
                description="Test",
                checklist=[],
                status=ApprovalStatus.PENDING,
            )
            _pending_approvals[request.id] = request

        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store",
            return_value=None
        ):
            result = get_pending_approvals(workflow_id="wf_1")

        assert len(result) == 2
        assert all(r.workflow_id == "wf_1" for r in result)

    def test_get_pending_excludes_non_pending(self):
        """get_pending_approvals excludes resolved approvals."""
        pending = ApprovalRequest(
            id="apr_pending",
            workflow_id="wf_1",
            step_id="step_1",
            title="Pending",
            description="Test",
            checklist=[],
            status=ApprovalStatus.PENDING,
        )
        approved = ApprovalRequest(
            id="apr_approved",
            workflow_id="wf_1",
            step_id="step_1",
            title="Approved",
            description="Test",
            checklist=[],
            status=ApprovalStatus.APPROVED,
        )
        _pending_approvals[pending.id] = pending
        _pending_approvals[approved.id] = approved

        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store",
            return_value=None
        ):
            result = get_pending_approvals()

        assert len(result) == 1
        assert result[0].id == "apr_pending"


class TestGracefulDegradation:
    """Tests for graceful degradation when store unavailable."""

    @pytest.fixture(autouse=True)
    def clear_approvals(self):
        """Clear in-memory approvals before each test."""
        _pending_approvals.clear()
        yield
        _pending_approvals.clear()

    def test_resolve_works_without_store(self):
        """resolve_approval works when store unavailable."""
        request = ApprovalRequest(
            id="apr_no_store",
            workflow_id="wf_1",
            step_id="step_1",
            title="Test",
            description="Test",
            checklist=[],
        )
        _pending_approvals[request.id] = request

        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store",
            return_value=None
        ):
            result = resolve_approval(
                request_id="apr_no_store",
                status=ApprovalStatus.APPROVED,
                responder_id="user-1",
            )

        assert result is True
        assert _pending_approvals["apr_no_store"].status == ApprovalStatus.APPROVED

    def test_get_pending_works_without_store(self):
        """get_pending_approvals works when store unavailable."""
        request = ApprovalRequest(
            id="apr_no_store",
            workflow_id="wf_1",
            step_id="step_1",
            title="Test",
            description="Test",
            checklist=[],
            status=ApprovalStatus.PENDING,
        )
        _pending_approvals[request.id] = request

        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store",
            return_value=None
        ):
            result = get_pending_approvals()

        assert len(result) == 1
        assert result[0].id == "apr_no_store"
