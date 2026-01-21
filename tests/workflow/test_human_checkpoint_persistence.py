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
    recover_pending_approvals,
    reset_approval_recovery,
    clear_pending_approvals,
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
            "aragora.workflow.nodes.human_checkpoint._get_governance_store", return_value=mock_store
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
            "aragora.workflow.nodes.human_checkpoint._get_governance_store", return_value=mock_store
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
            "aragora.workflow.nodes.human_checkpoint._get_governance_store", return_value=mock_store
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
        """Clear in-memory approvals and reset recovery state before each test."""
        _pending_approvals.clear()
        reset_approval_recovery()
        yield
        _pending_approvals.clear()
        reset_approval_recovery()

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
            "aragora.workflow.nodes.human_checkpoint._get_governance_store", return_value=None
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
            "aragora.workflow.nodes.human_checkpoint._get_governance_store", return_value=mock_store
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
            "aragora.workflow.nodes.human_checkpoint._get_governance_store", return_value=mock_store
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
            "aragora.workflow.nodes.human_checkpoint._get_governance_store", return_value=None
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
            "aragora.workflow.nodes.human_checkpoint._get_governance_store", return_value=None
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
            "aragora.workflow.nodes.human_checkpoint._get_governance_store", return_value=None
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
            "aragora.workflow.nodes.human_checkpoint._get_governance_store", return_value=None
        ):
            result = get_pending_approvals()

        assert len(result) == 1
        assert result[0].id == "apr_no_store"


class TestApprovalRecovery:
    """Tests for approval recovery from GovernanceStore on startup."""

    @pytest.fixture(autouse=True)
    def clear_state(self):
        """Clear approvals and reset recovery state before each test."""
        clear_pending_approvals()
        reset_approval_recovery()
        yield
        clear_pending_approvals()
        reset_approval_recovery()

    def test_recover_pending_approvals_loads_from_store(self):
        """recover_pending_approvals loads approvals from GovernanceStore."""
        # Mock persisted approval
        mock_record = MagicMock()
        mock_record.approval_id = "apr_recovered"
        mock_record.title = "Recovered Approval"
        mock_record.description = "From DB after restart"
        mock_record.status = "pending"
        mock_record.timeout_seconds = 3600
        mock_record.requested_at = datetime.now(timezone.utc)
        mock_record.approved_by = None
        mock_record.approved_at = None
        mock_record.metadata_json = '{"workflow_id": "wf_1", "step_id": "step_1", "checklist": []}'

        mock_store = MagicMock()
        mock_store.list_approvals.return_value = [mock_record]

        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store", return_value=mock_store
        ):
            count = recover_pending_approvals()

        assert count == 1
        assert "apr_recovered" in _pending_approvals
        assert _pending_approvals["apr_recovered"].title == "Recovered Approval"

    def test_recover_is_idempotent(self):
        """recover_pending_approvals only runs once."""
        mock_record = MagicMock()
        mock_record.approval_id = "apr_once"
        mock_record.title = "Once"
        mock_record.description = "Only once"
        mock_record.status = "pending"
        mock_record.timeout_seconds = 3600
        mock_record.requested_at = datetime.now(timezone.utc)
        mock_record.approved_by = None
        mock_record.approved_at = None
        mock_record.metadata_json = "{}"

        mock_store = MagicMock()
        mock_store.list_approvals.return_value = [mock_record]

        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store", return_value=mock_store
        ):
            count1 = recover_pending_approvals()
            count2 = recover_pending_approvals()

        assert count1 == 1
        assert count2 == 0  # Second call should skip
        # list_approvals should only be called once
        assert mock_store.list_approvals.call_count == 1

    def test_recover_skips_existing_in_memory(self):
        """Recovery doesn't overwrite existing in-memory approvals."""
        # Add in-memory approval first
        existing = ApprovalRequest(
            id="apr_existing",
            workflow_id="wf_1",
            step_id="step_1",
            title="In Memory Version",
            description="Existing",
            checklist=[],
            status=ApprovalStatus.PENDING,
        )
        _pending_approvals[existing.id] = existing

        # Mock same ID from store with different title
        mock_record = MagicMock()
        mock_record.approval_id = "apr_existing"
        mock_record.title = "DB Version"
        mock_record.description = "From DB"
        mock_record.status = "pending"
        mock_record.timeout_seconds = 3600
        mock_record.requested_at = datetime.now(timezone.utc)
        mock_record.metadata_json = "{}"

        mock_store = MagicMock()
        mock_store.list_approvals.return_value = [mock_record]

        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store", return_value=mock_store
        ):
            count = recover_pending_approvals()

        assert count == 0  # Already exists
        # In-memory version preserved
        assert _pending_approvals["apr_existing"].title == "In Memory Version"

    def test_recover_handles_store_error(self):
        """Recovery handles store errors gracefully."""
        mock_store = MagicMock()
        mock_store.list_approvals.side_effect = Exception("DB connection failed")

        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store", return_value=mock_store
        ):
            count = recover_pending_approvals()

        assert count == 0  # No crash, returns 0

    def test_recover_handles_no_store(self):
        """Recovery handles missing store gracefully."""
        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store", return_value=None
        ):
            count = recover_pending_approvals()

        assert count == 0

    def test_get_pending_triggers_recovery(self):
        """get_pending_approvals automatically triggers recovery."""
        mock_record = MagicMock()
        mock_record.approval_id = "apr_auto"
        mock_record.title = "Auto Recovered"
        mock_record.description = "Test"
        mock_record.status = "pending"
        mock_record.timeout_seconds = 3600
        mock_record.requested_at = datetime.now(timezone.utc)
        mock_record.metadata_json = '{"workflow_id": "wf_1"}'

        mock_store = MagicMock()
        mock_store.list_approvals.return_value = [mock_record]

        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store", return_value=mock_store
        ):
            # First call should trigger recovery
            result = get_pending_approvals()

        assert len(result) == 1
        assert result[0].id == "apr_auto"
        # Verify recovery was called (list_approvals was called)
        mock_store.list_approvals.assert_called()

    def test_get_approval_request_triggers_recovery(self):
        """get_approval_request automatically triggers recovery."""
        mock_record = MagicMock()
        mock_record.approval_id = "apr_single"
        mock_record.title = "Single"
        mock_record.description = "Test"
        mock_record.status = "pending"
        mock_record.timeout_seconds = 3600
        mock_record.requested_at = datetime.now(timezone.utc)
        mock_record.metadata_json = '{"workflow_id": "wf_1"}'

        mock_store = MagicMock()
        mock_store.list_approvals.return_value = [mock_record]
        mock_store.get_approval.return_value = None  # Not found after recovery

        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store", return_value=mock_store
        ):
            result = get_approval_request("apr_single")

        # Should have recovered the approval
        assert result is not None
        assert result.id == "apr_single"

    def test_recovery_reconstructs_checklist(self):
        """Recovery properly reconstructs checklist from metadata."""
        mock_record = MagicMock()
        mock_record.approval_id = "apr_checklist"
        mock_record.title = "With Checklist"
        mock_record.description = "Test"
        mock_record.status = "pending"
        mock_record.timeout_seconds = 3600
        mock_record.requested_at = datetime.now(timezone.utc)
        mock_record.approved_by = None
        mock_record.approved_at = None
        mock_record.metadata_json = """
        {
            "workflow_id": "wf_1",
            "step_id": "step_1",
            "checklist": [
                {"id": "item_0", "label": "Check 1", "required": true},
                {"id": "item_1", "label": "Check 2", "required": false}
            ],
            "escalation_emails": ["admin@example.com"]
        }
        """

        mock_store = MagicMock()
        mock_store.list_approvals.return_value = [mock_record]

        with patch(
            "aragora.workflow.nodes.human_checkpoint._get_governance_store", return_value=mock_store
        ):
            recover_pending_approvals()

        request = _pending_approvals["apr_checklist"]
        assert len(request.checklist) == 2
        assert request.checklist[0].label == "Check 1"
        assert request.checklist[0].required is True
        assert request.checklist[1].label == "Check 2"
        assert request.checklist[1].required is False
        assert request.escalation_emails == ["admin@example.com"]
