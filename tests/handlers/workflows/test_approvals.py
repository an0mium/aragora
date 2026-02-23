"""Tests for workflow approval operations (aragora/server/handlers/workflows/approvals.py).

Covers all public functions in the approvals module:
- list_pending_approvals() - list pending human approvals
- resolve_approval() - resolve a human approval request
- get_approval() - get an approval request by ID
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.workflows.approvals import (
    get_approval,
    list_pending_approvals,
    resolve_approval,
)


# ---------------------------------------------------------------------------
# Patch target
# ---------------------------------------------------------------------------

PATCH_MOD = "aragora.server.handlers.workflows"


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockApprovalStatus(Enum):
    """Mock ApprovalStatus enum matching the real one."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    TIMEOUT = "timeout"


def _make_approval(
    id: str = "req-1",
    workflow_id: str = "wf-1",
    step_id: str = "step-1",
    title: str = "Review deployment",
    description: str = "Please review the deployment plan",
    status: str = "pending",
) -> MagicMock:
    """Create a mock approval object with a to_dict method."""
    approval = MagicMock()
    approval.id = id
    approval.workflow_id = workflow_id
    approval.step_id = step_id
    approval.title = title
    approval.description = description
    approval.status = status
    approval.to_dict.return_value = {
        "id": id,
        "workflow_id": workflow_id,
        "step_id": step_id,
        "title": title,
        "description": description,
        "status": status,
    }
    return approval


# ===========================================================================
# list_pending_approvals
# ===========================================================================


class TestListPendingApprovals:
    """Tests for list_pending_approvals()."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_get_pending_approvals_is_none(self):
        """When the upstream function is unavailable, return empty list."""
        with patch(f"{PATCH_MOD}.get_pending_approvals", None):
            result = await list_pending_approvals()
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_approvals_exist(self):
        """When there are no pending approvals, return empty list."""
        mock_fn = MagicMock(return_value=[])
        with patch(f"{PATCH_MOD}.get_pending_approvals", mock_fn):
            result = await list_pending_approvals()
        assert result == []
        mock_fn.assert_called_once_with(None)

    @pytest.mark.asyncio
    async def test_returns_approvals_as_dicts(self):
        """Approvals are converted to dicts via to_dict()."""
        a1 = _make_approval(id="req-1")
        a2 = _make_approval(id="req-2", title="Review schema change")
        mock_fn = MagicMock(return_value=[a1, a2])
        with patch(f"{PATCH_MOD}.get_pending_approvals", mock_fn):
            result = await list_pending_approvals()
        assert len(result) == 2
        assert result[0]["id"] == "req-1"
        assert result[1]["id"] == "req-2"
        a1.to_dict.assert_called_once()
        a2.to_dict.assert_called_once()

    @pytest.mark.asyncio
    async def test_passes_workflow_id_filter(self):
        """When workflow_id is provided, it is forwarded to the upstream function."""
        mock_fn = MagicMock(return_value=[])
        with patch(f"{PATCH_MOD}.get_pending_approvals", mock_fn):
            result = await list_pending_approvals(workflow_id="wf-42")
        mock_fn.assert_called_once_with("wf-42")
        assert result == []

    @pytest.mark.asyncio
    async def test_passes_none_workflow_id_by_default(self):
        """By default, workflow_id is None (no filter)."""
        mock_fn = MagicMock(return_value=[])
        with patch(f"{PATCH_MOD}.get_pending_approvals", mock_fn):
            result = await list_pending_approvals()
        mock_fn.assert_called_once_with(None)

    @pytest.mark.asyncio
    async def test_single_approval_returned(self):
        """Single approval is converted and returned correctly."""
        a = _make_approval(id="only-one", title="Solo review")
        mock_fn = MagicMock(return_value=[a])
        with patch(f"{PATCH_MOD}.get_pending_approvals", mock_fn):
            result = await list_pending_approvals()
        assert len(result) == 1
        assert result[0]["title"] == "Solo review"

    @pytest.mark.asyncio
    async def test_tenant_id_parameter_accepted(self):
        """tenant_id parameter is accepted (currently unused but part of signature)."""
        mock_fn = MagicMock(return_value=[])
        with patch(f"{PATCH_MOD}.get_pending_approvals", mock_fn):
            result = await list_pending_approvals(tenant_id="tenant-xyz")
        assert result == []

    @pytest.mark.asyncio
    async def test_many_approvals(self):
        """Multiple approvals are all converted."""
        approvals = [_make_approval(id=f"req-{i}") for i in range(10)]
        mock_fn = MagicMock(return_value=approvals)
        with patch(f"{PATCH_MOD}.get_pending_approvals", mock_fn):
            result = await list_pending_approvals()
        assert len(result) == 10
        for i, item in enumerate(result):
            assert item["id"] == f"req-{i}"

    @pytest.mark.asyncio
    async def test_to_dict_output_preserved(self):
        """The dict returned by to_dict() is the exact item in the result list."""
        custom_dict = {"id": "x", "custom_field": "special_value", "nested": {"a": 1}}
        a = MagicMock()
        a.to_dict.return_value = custom_dict
        mock_fn = MagicMock(return_value=[a])
        with patch(f"{PATCH_MOD}.get_pending_approvals", mock_fn):
            result = await list_pending_approvals()
        assert result[0] is custom_dict


# ===========================================================================
# resolve_approval
# ===========================================================================


class TestResolveApproval:
    """Tests for resolve_approval()."""

    @pytest.mark.asyncio
    async def test_resolve_approved(self):
        """Resolving with 'approved' status works correctly."""
        mock_resolve = MagicMock(return_value=True)
        with (
            patch(f"{PATCH_MOD}.ApprovalStatus", MockApprovalStatus),
            patch(f"{PATCH_MOD}._resolve", mock_resolve),
        ):
            result = await resolve_approval("req-1", "approved", "user-1")
        assert result is True
        mock_resolve.assert_called_once_with(
            "req-1",
            MockApprovalStatus.APPROVED,
            "user-1",
            "",
            None,
        )

    @pytest.mark.asyncio
    async def test_resolve_rejected(self):
        """Resolving with 'rejected' status works correctly."""
        mock_resolve = MagicMock(return_value=True)
        with (
            patch(f"{PATCH_MOD}.ApprovalStatus", MockApprovalStatus),
            patch(f"{PATCH_MOD}._resolve", mock_resolve),
        ):
            result = await resolve_approval("req-2", "rejected", "user-2")
        assert result is True
        mock_resolve.assert_called_once_with(
            "req-2",
            MockApprovalStatus.REJECTED,
            "user-2",
            "",
            None,
        )

    @pytest.mark.asyncio
    async def test_resolve_escalated(self):
        """Resolving with 'escalated' status works correctly."""
        mock_resolve = MagicMock(return_value=True)
        with (
            patch(f"{PATCH_MOD}.ApprovalStatus", MockApprovalStatus),
            patch(f"{PATCH_MOD}._resolve", mock_resolve),
        ):
            result = await resolve_approval("req-3", "escalated", "user-3")
        assert result is True
        mock_resolve.assert_called_once_with(
            "req-3",
            MockApprovalStatus.ESCALATED,
            "user-3",
            "",
            None,
        )

    @pytest.mark.asyncio
    async def test_resolve_with_notes(self):
        """Notes are forwarded to the resolver."""
        mock_resolve = MagicMock(return_value=True)
        with (
            patch(f"{PATCH_MOD}.ApprovalStatus", MockApprovalStatus),
            patch(f"{PATCH_MOD}._resolve", mock_resolve),
        ):
            result = await resolve_approval("req-1", "approved", "user-1", notes="Looks good")
        assert result is True
        mock_resolve.assert_called_once_with(
            "req-1",
            MockApprovalStatus.APPROVED,
            "user-1",
            "Looks good",
            None,
        )

    @pytest.mark.asyncio
    async def test_resolve_with_checklist_updates(self):
        """Checklist updates are forwarded to the resolver."""
        checklist = {"item-1": True, "item-2": False}
        mock_resolve = MagicMock(return_value=True)
        with (
            patch(f"{PATCH_MOD}.ApprovalStatus", MockApprovalStatus),
            patch(f"{PATCH_MOD}._resolve", mock_resolve),
        ):
            result = await resolve_approval(
                "req-1",
                "approved",
                "user-1",
                notes="All checked",
                checklist_updates=checklist,
            )
        assert result is True
        mock_resolve.assert_called_once_with(
            "req-1",
            MockApprovalStatus.APPROVED,
            "user-1",
            "All checked",
            checklist,
        )

    @pytest.mark.asyncio
    async def test_resolve_returns_false_when_resolver_returns_false(self):
        """When the resolver returns False, we propagate that."""
        mock_resolve = MagicMock(return_value=False)
        with (
            patch(f"{PATCH_MOD}.ApprovalStatus", MockApprovalStatus),
            patch(f"{PATCH_MOD}._resolve", mock_resolve),
        ):
            result = await resolve_approval("req-1", "approved", "user-1")
        assert result is False

    @pytest.mark.asyncio
    async def test_invalid_status_raises_value_error(self):
        """An invalid status string raises ValueError."""
        with (
            patch(f"{PATCH_MOD}.ApprovalStatus", MockApprovalStatus),
            patch(f"{PATCH_MOD}._resolve", MagicMock()),
        ):
            with pytest.raises(ValueError, match="Invalid status: bogus"):
                await resolve_approval("req-1", "bogus", "user-1")

    @pytest.mark.asyncio
    async def test_resolve_unavailable_raises_runtime_error(self):
        """When _resolve is None, RuntimeError is raised."""
        with (
            patch(f"{PATCH_MOD}.ApprovalStatus", MockApprovalStatus),
            patch(f"{PATCH_MOD}._resolve", None),
        ):
            with pytest.raises(RuntimeError, match="Approval resolution is unavailable"):
                await resolve_approval("req-1", "approved", "user-1")

    @pytest.mark.asyncio
    async def test_status_case_insensitive_lowercase(self):
        """Status lookup is case-insensitive (lowercase input)."""
        mock_resolve = MagicMock(return_value=True)
        with (
            patch(f"{PATCH_MOD}.ApprovalStatus", MockApprovalStatus),
            patch(f"{PATCH_MOD}._resolve", mock_resolve),
        ):
            result = await resolve_approval("req-1", "approved", "user-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_status_case_insensitive_mixed_case(self):
        """Status lookup is case-insensitive (mixed case input)."""
        mock_resolve = MagicMock(return_value=True)
        with (
            patch(f"{PATCH_MOD}.ApprovalStatus", MockApprovalStatus),
            patch(f"{PATCH_MOD}._resolve", mock_resolve),
        ):
            result = await resolve_approval("req-1", "Approved", "user-1")
        assert result is True
        mock_resolve.assert_called_once_with(
            "req-1",
            MockApprovalStatus.APPROVED,
            "user-1",
            "",
            None,
        )

    @pytest.mark.asyncio
    async def test_status_case_insensitive_uppercase(self):
        """Status lookup is case-insensitive (uppercase input)."""
        mock_resolve = MagicMock(return_value=True)
        with (
            patch(f"{PATCH_MOD}.ApprovalStatus", MockApprovalStatus),
            patch(f"{PATCH_MOD}._resolve", mock_resolve),
        ):
            result = await resolve_approval("req-1", "REJECTED", "user-1")
        assert result is True
        mock_resolve.assert_called_once_with(
            "req-1",
            MockApprovalStatus.REJECTED,
            "user-1",
            "",
            None,
        )

    @pytest.mark.asyncio
    async def test_resolve_timeout_status(self):
        """Resolving with 'timeout' status works correctly."""
        mock_resolve = MagicMock(return_value=True)
        with (
            patch(f"{PATCH_MOD}.ApprovalStatus", MockApprovalStatus),
            patch(f"{PATCH_MOD}._resolve", mock_resolve),
        ):
            result = await resolve_approval("req-1", "timeout", "system")
        assert result is True
        mock_resolve.assert_called_once_with(
            "req-1",
            MockApprovalStatus.TIMEOUT,
            "system",
            "",
            None,
        )

    @pytest.mark.asyncio
    async def test_resolve_pending_status(self):
        """Resolving with 'pending' status works (resetting to pending)."""
        mock_resolve = MagicMock(return_value=True)
        with (
            patch(f"{PATCH_MOD}.ApprovalStatus", MockApprovalStatus),
            patch(f"{PATCH_MOD}._resolve", mock_resolve),
        ):
            result = await resolve_approval("req-1", "pending", "admin")
        assert result is True
        mock_resolve.assert_called_once_with(
            "req-1",
            MockApprovalStatus.PENDING,
            "admin",
            "",
            None,
        )

    @pytest.mark.asyncio
    async def test_empty_status_raises_value_error(self):
        """An empty status string raises ValueError (after uppercasing becomes '')."""
        with (
            patch(f"{PATCH_MOD}.ApprovalStatus", MockApprovalStatus),
            patch(f"{PATCH_MOD}._resolve", MagicMock()),
        ):
            with pytest.raises(ValueError):
                await resolve_approval("req-1", "", "user-1")

    @pytest.mark.asyncio
    async def test_resolve_with_empty_notes(self):
        """Empty notes string is default and forwarded."""
        mock_resolve = MagicMock(return_value=True)
        with (
            patch(f"{PATCH_MOD}.ApprovalStatus", MockApprovalStatus),
            patch(f"{PATCH_MOD}._resolve", mock_resolve),
        ):
            await resolve_approval("req-1", "approved", "user-1", notes="")
        mock_resolve.assert_called_once_with(
            "req-1",
            MockApprovalStatus.APPROVED,
            "user-1",
            "",
            None,
        )

    @pytest.mark.asyncio
    async def test_resolve_with_empty_checklist(self):
        """Empty checklist dict is forwarded (not converted to None)."""
        mock_resolve = MagicMock(return_value=True)
        with (
            patch(f"{PATCH_MOD}.ApprovalStatus", MockApprovalStatus),
            patch(f"{PATCH_MOD}._resolve", mock_resolve),
        ):
            await resolve_approval("req-1", "approved", "user-1", checklist_updates={})
        mock_resolve.assert_called_once_with(
            "req-1",
            MockApprovalStatus.APPROVED,
            "user-1",
            "",
            {},
        )


# ===========================================================================
# get_approval
# ===========================================================================


class TestGetApproval:
    """Tests for get_approval()."""

    @pytest.mark.asyncio
    async def test_returns_none_when_get_approval_request_is_none(self):
        """When the upstream function is unavailable, return None."""
        with patch(f"{PATCH_MOD}.get_approval_request", None):
            result = await get_approval("req-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_when_approval_not_found(self):
        """When the upstream function returns None, return None."""
        mock_fn = MagicMock(return_value=None)
        with patch(f"{PATCH_MOD}.get_approval_request", mock_fn):
            result = await get_approval("nonexistent")
        assert result is None
        mock_fn.assert_called_once_with("nonexistent")

    @pytest.mark.asyncio
    async def test_returns_approval_dict(self):
        """When an approval is found, return its dict representation."""
        approval = _make_approval(id="req-42", title="Deploy approval")
        mock_fn = MagicMock(return_value=approval)
        with patch(f"{PATCH_MOD}.get_approval_request", mock_fn):
            result = await get_approval("req-42")
        assert result is not None
        assert result["id"] == "req-42"
        assert result["title"] == "Deploy approval"
        mock_fn.assert_called_once_with("req-42")
        approval.to_dict.assert_called_once()

    @pytest.mark.asyncio
    async def test_request_id_is_forwarded(self):
        """The request_id is forwarded to the upstream function."""
        mock_fn = MagicMock(return_value=None)
        with patch(f"{PATCH_MOD}.get_approval_request", mock_fn):
            await get_approval("specific-id-123")
        mock_fn.assert_called_once_with("specific-id-123")

    @pytest.mark.asyncio
    async def test_to_dict_output_returned_directly(self):
        """The exact dict from to_dict() is returned."""
        custom_dict = {"id": "x", "extra": True}
        approval = MagicMock()
        approval.to_dict.return_value = custom_dict
        mock_fn = MagicMock(return_value=approval)
        with patch(f"{PATCH_MOD}.get_approval_request", mock_fn):
            result = await get_approval("x")
        assert result is custom_dict

    @pytest.mark.asyncio
    async def test_empty_request_id(self):
        """Empty string request_id is forwarded (no validation in handler)."""
        mock_fn = MagicMock(return_value=None)
        with patch(f"{PATCH_MOD}.get_approval_request", mock_fn):
            result = await get_approval("")
        assert result is None
        mock_fn.assert_called_once_with("")


# ===========================================================================
# Module-level __all__ export
# ===========================================================================


class TestModuleExports:
    """Tests for module-level __all__ export."""

    def test_all_exports(self):
        """__all__ lists all three public functions."""
        from aragora.server.handlers.workflows import approvals

        assert "list_pending_approvals" in approvals.__all__
        assert "resolve_approval" in approvals.__all__
        assert "get_approval" in approvals.__all__

    def test_all_length(self):
        """__all__ has exactly three entries."""
        from aragora.server.handlers.workflows import approvals

        assert len(approvals.__all__) == 3
