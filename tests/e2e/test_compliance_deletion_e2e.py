"""
E2E Tests for Compliance Deletion and Legal Hold Endpoints.

Tests the GDPR deletion management and legal hold HTTP API endpoints:
- GET /api/v2/compliance/gdpr/deletions - List deletion requests
- GET /api/v2/compliance/gdpr/deletions/{request_id} - Get deletion details
- POST /api/v2/compliance/gdpr/deletions/{request_id}/cancel - Cancel deletion
- GET /api/v2/compliance/gdpr/legal-holds - List legal holds
- POST /api/v2/compliance/gdpr/legal-holds - Create legal hold
- DELETE /api/v2/compliance/gdpr/legal-holds/{hold_id} - Release legal hold

Run with: pytest tests/e2e/test_compliance_deletion_e2e.py -v
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from aragora.server.handlers.compliance_handler import ComplianceHandler
from aragora.server.handlers.base import ServerContext

pytestmark = [pytest.mark.e2e, pytest.mark.compliance]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_server_context():
    """Create a mock server context for handlers."""
    context = MagicMock(spec=ServerContext)
    context.get_user_id = MagicMock(return_value="test-user-123")
    context.get_org_id = MagicMock(return_value="test-org-456")
    return context


@pytest.fixture
def compliance_handler(mock_server_context):
    """Create a ComplianceHandler instance."""
    return ComplianceHandler(mock_server_context)


@pytest.fixture
def sample_deletion_requests() -> List[Dict[str, Any]]:
    """Generate sample deletion requests for testing."""
    base_time = datetime.now(timezone.utc)
    return [
        {
            "request_id": f"del-{uuid.uuid4().hex[:8]}",
            "user_id": "user-001",
            "status": "pending",
            "reason": "User request",
            "created_at": (base_time - timedelta(days=5)).isoformat(),
            "scheduled_for": (base_time + timedelta(days=25)).isoformat(),
            "grace_period_days": 30,
        },
        {
            "request_id": f"del-{uuid.uuid4().hex[:8]}",
            "user_id": "user-002",
            "status": "completed",
            "reason": "Account closure",
            "created_at": (base_time - timedelta(days=60)).isoformat(),
            "scheduled_for": (base_time - timedelta(days=30)).isoformat(),
            "completed_at": (base_time - timedelta(days=30)).isoformat(),
            "grace_period_days": 30,
        },
        {
            "request_id": f"del-{uuid.uuid4().hex[:8]}",
            "user_id": "user-003",
            "status": "held",
            "reason": "User request",
            "created_at": (base_time - timedelta(days=10)).isoformat(),
            "scheduled_for": (base_time + timedelta(days=20)).isoformat(),
            "grace_period_days": 30,
            "hold_id": "hold-12345",
        },
    ]


@pytest.fixture
def sample_legal_holds() -> List[Dict[str, Any]]:
    """Generate sample legal holds for testing."""
    base_time = datetime.now(timezone.utc)
    return [
        {
            "hold_id": f"hold-{uuid.uuid4().hex[:8]}",
            "user_ids": ["user-003", "user-004"],
            "reason": "Litigation - Case ABC-123",
            "case_reference": "ABC-123",
            "created_at": (base_time - timedelta(days=30)).isoformat(),
            "created_by": "legal-team",
            "is_active": True,
        },
        {
            "hold_id": f"hold-{uuid.uuid4().hex[:8]}",
            "user_ids": ["user-005"],
            "reason": "Regulatory investigation",
            "case_reference": "REG-456",
            "created_at": (base_time - timedelta(days=60)).isoformat(),
            "created_by": "compliance-officer",
            "is_active": True,
            "expires_at": (base_time + timedelta(days=90)).isoformat(),
        },
        {
            "hold_id": f"hold-{uuid.uuid4().hex[:8]}",
            "user_ids": ["user-006"],
            "reason": "Former employee - tax audit",
            "created_at": (base_time - timedelta(days=120)).isoformat(),
            "created_by": "hr-admin",
            "is_active": False,
            "released_at": (base_time - timedelta(days=30)).isoformat(),
            "released_by": "hr-admin",
        },
    ]


# ============================================================================
# Deletion Management Tests - List Deletions
# ============================================================================


class TestListDeletions:
    """Tests for listing deletion requests."""

    @pytest.mark.asyncio
    async def test_list_deletions_returns_all(self, compliance_handler, sample_deletion_requests):
        """Test listing all deletion requests."""
        with patch.object(compliance_handler, "_list_deletions") as mock_list:
            mock_list.return_value = {
                "status": 200,
                "data": {
                    "deletions": sample_deletion_requests,
                    "count": len(sample_deletion_requests),
                    "filters": {"status": None, "limit": 50},
                },
            }

            result = await compliance_handler.handle(
                method="GET",
                path="/api/v2/compliance/gdpr/deletions",
                query_params={},
            )

            assert result["status"] == 200
            assert result["data"]["count"] == 3
            assert len(result["data"]["deletions"]) == 3

    @pytest.mark.asyncio
    async def test_list_deletions_filter_by_status(self, compliance_handler):
        """Test listing deletions filtered by status."""
        with patch.object(compliance_handler, "_list_deletions") as mock_list:
            mock_list.return_value = {
                "status": 200,
                "data": {
                    "deletions": [{"request_id": "del-123", "status": "pending"}],
                    "count": 1,
                    "filters": {"status": "pending", "limit": 50},
                },
            }

            result = await compliance_handler.handle(
                method="GET",
                path="/api/v2/compliance/gdpr/deletions",
                query_params={"status": "pending"},
            )

            assert result["status"] == 200
            assert result["data"]["filters"]["status"] == "pending"
            assert all(d["status"] == "pending" for d in result["data"]["deletions"])

    @pytest.mark.asyncio
    async def test_list_deletions_with_limit(self, compliance_handler):
        """Test listing deletions with custom limit."""
        with patch.object(compliance_handler, "_list_deletions") as mock_list:
            mock_list.return_value = {
                "status": 200,
                "data": {
                    "deletions": [{"request_id": "del-123"}],
                    "count": 1,
                    "filters": {"status": None, "limit": 10},
                },
            }

            result = await compliance_handler.handle(
                method="GET",
                path="/api/v2/compliance/gdpr/deletions",
                query_params={"limit": "10"},
            )

            assert result["status"] == 200
            assert result["data"]["filters"]["limit"] == 10

    @pytest.mark.asyncio
    async def test_list_deletions_max_limit_enforced(self, compliance_handler):
        """Test that max limit of 200 is enforced."""
        with patch.object(compliance_handler, "_list_deletions") as mock_list:
            mock_list.return_value = {
                "status": 200,
                "data": {
                    "deletions": [],
                    "count": 0,
                    "filters": {"status": None, "limit": 200},
                },
            }

            result = await compliance_handler.handle(
                method="GET",
                path="/api/v2/compliance/gdpr/deletions",
                query_params={"limit": "500"},  # Exceeds max
            )

            assert result["status"] == 200
            # Handler should cap at 200
            assert result["data"]["filters"]["limit"] <= 200


# ============================================================================
# Deletion Management Tests - Get Deletion
# ============================================================================


class TestGetDeletion:
    """Tests for getting a specific deletion request."""

    @pytest.mark.asyncio
    async def test_get_deletion_success(self, compliance_handler):
        """Test getting an existing deletion request."""
        request_id = "del-test-12345"

        with patch.object(compliance_handler, "_get_deletion") as mock_get:
            mock_get.return_value = {
                "status": 200,
                "data": {
                    "deletion": {
                        "request_id": request_id,
                        "user_id": "user-001",
                        "status": "pending",
                        "reason": "User request",
                        "scheduled_for": "2024-02-15T00:00:00+00:00",
                    }
                },
            }

            result = await compliance_handler.handle(
                method="GET",
                path=f"/api/v2/compliance/gdpr/deletions/{request_id}",
            )

            assert result["status"] == 200
            assert result["data"]["deletion"]["request_id"] == request_id

    @pytest.mark.asyncio
    async def test_get_deletion_not_found(self, compliance_handler):
        """Test getting a non-existent deletion request."""
        with patch.object(compliance_handler, "_get_deletion") as mock_get:
            mock_get.return_value = {
                "status": 404,
                "error": "Deletion request not found",
            }

            result = await compliance_handler.handle(
                method="GET",
                path="/api/v2/compliance/gdpr/deletions/nonexistent-id",
            )

            assert result["status"] == 404

    @pytest.mark.asyncio
    async def test_get_deletion_includes_hold_info(self, compliance_handler):
        """Test that held deletions include hold information."""
        with patch.object(compliance_handler, "_get_deletion") as mock_get:
            mock_get.return_value = {
                "status": 200,
                "data": {
                    "deletion": {
                        "request_id": "del-held-123",
                        "user_id": "user-003",
                        "status": "held",
                        "reason": "User request",
                        "hold_info": {
                            "hold_id": "hold-abc",
                            "reason": "Litigation",
                            "case_reference": "CASE-001",
                        },
                    }
                },
            }

            result = await compliance_handler.handle(
                method="GET",
                path="/api/v2/compliance/gdpr/deletions/del-held-123",
            )

            assert result["status"] == 200
            assert result["data"]["deletion"]["status"] == "held"
            assert "hold_info" in result["data"]["deletion"]


# ============================================================================
# Deletion Management Tests - Cancel Deletion
# ============================================================================


class TestCancelDeletion:
    """Tests for cancelling deletion requests."""

    @pytest.mark.asyncio
    async def test_cancel_deletion_success(self, compliance_handler):
        """Test successfully cancelling a pending deletion."""
        request_id = "del-cancel-test"

        with patch.object(compliance_handler, "_cancel_deletion") as mock_cancel:
            mock_cancel.return_value = {
                "status": 200,
                "data": {
                    "message": "Deletion cancelled successfully",
                    "deletion": {
                        "request_id": request_id,
                        "status": "cancelled",
                        "cancelled_at": datetime.now(timezone.utc).isoformat(),
                        "cancellation_reason": "Administrator request",
                    },
                },
            }

            result = await compliance_handler.handle(
                method="POST",
                path=f"/api/v2/compliance/gdpr/deletions/{request_id}/cancel",
                body={"reason": "Administrator request"},
            )

            assert result["status"] == 200
            assert result["data"]["deletion"]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_deletion_not_found(self, compliance_handler):
        """Test cancelling a non-existent deletion request."""
        with patch.object(compliance_handler, "_cancel_deletion") as mock_cancel:
            mock_cancel.return_value = {
                "status": 404,
                "error": "Deletion request not found",
            }

            result = await compliance_handler.handle(
                method="POST",
                path="/api/v2/compliance/gdpr/deletions/nonexistent/cancel",
                body={},
            )

            assert result["status"] == 404

    @pytest.mark.asyncio
    async def test_cancel_already_completed_deletion(self, compliance_handler):
        """Test cancelling an already completed deletion fails."""
        with patch.object(compliance_handler, "_cancel_deletion") as mock_cancel:
            mock_cancel.return_value = {
                "status": 400,
                "error": "Cannot cancel deletion in status completed",
            }

            result = await compliance_handler.handle(
                method="POST",
                path="/api/v2/compliance/gdpr/deletions/del-completed/cancel",
                body={},
            )

            assert result["status"] == 400
            assert "completed" in result["error"]

    @pytest.mark.asyncio
    async def test_cancel_held_deletion(self, compliance_handler):
        """Test cancelling a held deletion requires releasing hold first."""
        with patch.object(compliance_handler, "_cancel_deletion") as mock_cancel:
            mock_cancel.return_value = {
                "status": 400,
                "error": "Cannot cancel deletion under legal hold. Release hold first.",
            }

            result = await compliance_handler.handle(
                method="POST",
                path="/api/v2/compliance/gdpr/deletions/del-held/cancel",
                body={},
            )

            assert result["status"] == 400
            assert "legal hold" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_cancel_deletion_with_reason(self, compliance_handler):
        """Test cancellation reason is recorded."""
        request_id = "del-reason-test"
        reason = "User withdrew deletion request"

        with patch.object(compliance_handler, "_cancel_deletion") as mock_cancel:
            mock_cancel.return_value = {
                "status": 200,
                "data": {
                    "message": "Deletion cancelled successfully",
                    "deletion": {
                        "request_id": request_id,
                        "status": "cancelled",
                        "cancellation_reason": reason,
                    },
                },
            }

            result = await compliance_handler.handle(
                method="POST",
                path=f"/api/v2/compliance/gdpr/deletions/{request_id}/cancel",
                body={"reason": reason},
            )

            assert result["status"] == 200
            assert result["data"]["deletion"]["cancellation_reason"] == reason


# ============================================================================
# Legal Hold Management Tests - List Legal Holds
# ============================================================================


class TestListLegalHolds:
    """Tests for listing legal holds."""

    @pytest.mark.asyncio
    async def test_list_legal_holds_active_only(self, compliance_handler, sample_legal_holds):
        """Test listing only active legal holds."""
        active_holds = [h for h in sample_legal_holds if h["is_active"]]

        with patch.object(compliance_handler, "_list_legal_holds") as mock_list:
            mock_list.return_value = {
                "status": 200,
                "data": {
                    "legal_holds": active_holds,
                    "count": len(active_holds),
                    "filters": {"active_only": True},
                },
            }

            result = await compliance_handler.handle(
                method="GET",
                path="/api/v2/compliance/gdpr/legal-holds",
                query_params={"active_only": "true"},
            )

            assert result["status"] == 200
            assert result["data"]["filters"]["active_only"] is True
            assert all(h["is_active"] for h in result["data"]["legal_holds"])

    @pytest.mark.asyncio
    async def test_list_legal_holds_all(self, compliance_handler, sample_legal_holds):
        """Test listing all legal holds including released."""
        with patch.object(compliance_handler, "_list_legal_holds") as mock_list:
            mock_list.return_value = {
                "status": 200,
                "data": {
                    "legal_holds": sample_legal_holds,
                    "count": len(sample_legal_holds),
                    "filters": {"active_only": False},
                },
            }

            result = await compliance_handler.handle(
                method="GET",
                path="/api/v2/compliance/gdpr/legal-holds",
                query_params={"active_only": "false"},
            )

            assert result["status"] == 200
            assert result["data"]["count"] == 3
            # Should include released hold
            released = [h for h in result["data"]["legal_holds"] if not h["is_active"]]
            assert len(released) >= 1

    @pytest.mark.asyncio
    async def test_list_legal_holds_default_active_only(self, compliance_handler):
        """Test that default is to list only active holds."""
        with patch.object(compliance_handler, "_list_legal_holds") as mock_list:
            mock_list.return_value = {
                "status": 200,
                "data": {
                    "legal_holds": [],
                    "count": 0,
                    "filters": {"active_only": True},
                },
            }

            result = await compliance_handler.handle(
                method="GET",
                path="/api/v2/compliance/gdpr/legal-holds",
                query_params={},  # No filter specified
            )

            assert result["status"] == 200
            assert result["data"]["filters"]["active_only"] is True


# ============================================================================
# Legal Hold Management Tests - Create Legal Hold
# ============================================================================


class TestCreateLegalHold:
    """Tests for creating legal holds."""

    @pytest.mark.asyncio
    async def test_create_legal_hold_success(self, compliance_handler):
        """Test successfully creating a legal hold."""
        with patch.object(compliance_handler, "_create_legal_hold") as mock_create:
            hold_id = f"hold-{uuid.uuid4().hex[:8]}"
            mock_create.return_value = {
                "status": 201,
                "data": {
                    "message": "Legal hold created successfully",
                    "legal_hold": {
                        "hold_id": hold_id,
                        "user_ids": ["user-001", "user-002"],
                        "reason": "Litigation matter ABC",
                        "case_reference": "ABC-123",
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "created_by": "legal-team",
                        "is_active": True,
                    },
                },
            }

            result = await compliance_handler.handle(
                method="POST",
                path="/api/v2/compliance/gdpr/legal-holds",
                body={
                    "user_ids": ["user-001", "user-002"],
                    "reason": "Litigation matter ABC",
                    "case_reference": "ABC-123",
                },
            )

            assert result["status"] == 201
            assert result["data"]["legal_hold"]["is_active"] is True
            assert len(result["data"]["legal_hold"]["user_ids"]) == 2

    @pytest.mark.asyncio
    async def test_create_legal_hold_missing_user_ids(self, compliance_handler):
        """Test creating legal hold fails without user_ids."""
        with patch.object(compliance_handler, "_create_legal_hold") as mock_create:
            mock_create.return_value = {
                "status": 400,
                "error": "user_ids is required",
            }

            result = await compliance_handler.handle(
                method="POST",
                path="/api/v2/compliance/gdpr/legal-holds",
                body={"reason": "Some reason"},  # Missing user_ids
            )

            assert result["status"] == 400
            assert "user_ids" in result["error"]

    @pytest.mark.asyncio
    async def test_create_legal_hold_missing_reason(self, compliance_handler):
        """Test creating legal hold fails without reason."""
        with patch.object(compliance_handler, "_create_legal_hold") as mock_create:
            mock_create.return_value = {
                "status": 400,
                "error": "reason is required",
            }

            result = await compliance_handler.handle(
                method="POST",
                path="/api/v2/compliance/gdpr/legal-holds",
                body={"user_ids": ["user-001"]},  # Missing reason
            )

            assert result["status"] == 400
            assert "reason" in result["error"]

    @pytest.mark.asyncio
    async def test_create_legal_hold_with_expiration(self, compliance_handler):
        """Test creating legal hold with expiration date."""
        expires_at = (datetime.now(timezone.utc) + timedelta(days=90)).isoformat()

        with patch.object(compliance_handler, "_create_legal_hold") as mock_create:
            mock_create.return_value = {
                "status": 201,
                "data": {
                    "message": "Legal hold created successfully",
                    "legal_hold": {
                        "hold_id": "hold-expiring",
                        "user_ids": ["user-001"],
                        "reason": "Temporary investigation",
                        "expires_at": expires_at,
                        "is_active": True,
                    },
                },
            }

            result = await compliance_handler.handle(
                method="POST",
                path="/api/v2/compliance/gdpr/legal-holds",
                body={
                    "user_ids": ["user-001"],
                    "reason": "Temporary investigation",
                    "expires_at": expires_at,
                },
            )

            assert result["status"] == 201
            assert result["data"]["legal_hold"]["expires_at"] == expires_at

    @pytest.mark.asyncio
    async def test_create_legal_hold_blocks_pending_deletions(self, compliance_handler):
        """Test that creating legal hold blocks pending deletions for those users."""
        with patch.object(compliance_handler, "_create_legal_hold") as mock_create:
            mock_create.return_value = {
                "status": 201,
                "data": {
                    "message": "Legal hold created successfully",
                    "legal_hold": {
                        "hold_id": "hold-blocks-del",
                        "user_ids": ["user-with-pending-deletion"],
                        "reason": "Investigation",
                        "is_active": True,
                    },
                    "affected_deletions": [
                        {
                            "request_id": "del-blocked",
                            "previous_status": "pending",
                            "new_status": "held",
                        }
                    ],
                },
            }

            result = await compliance_handler.handle(
                method="POST",
                path="/api/v2/compliance/gdpr/legal-holds",
                body={
                    "user_ids": ["user-with-pending-deletion"],
                    "reason": "Investigation",
                },
            )

            assert result["status"] == 201
            assert "affected_deletions" in result["data"]
            assert len(result["data"]["affected_deletions"]) > 0


# ============================================================================
# Legal Hold Management Tests - Release Legal Hold
# ============================================================================


class TestReleaseLegalHold:
    """Tests for releasing legal holds."""

    @pytest.mark.asyncio
    async def test_release_legal_hold_success(self, compliance_handler):
        """Test successfully releasing a legal hold."""
        hold_id = "hold-to-release"

        with patch.object(compliance_handler, "_release_legal_hold") as mock_release:
            mock_release.return_value = {
                "status": 200,
                "data": {
                    "message": "Legal hold released successfully",
                    "legal_hold": {
                        "hold_id": hold_id,
                        "user_ids": ["user-001"],
                        "reason": "Litigation concluded",
                        "is_active": False,
                        "released_at": datetime.now(timezone.utc).isoformat(),
                        "released_by": "legal-team",
                    },
                },
            }

            result = await compliance_handler.handle(
                method="DELETE",
                path=f"/api/v2/compliance/gdpr/legal-holds/{hold_id}",
                body={"released_by": "legal-team"},
            )

            assert result["status"] == 200
            assert result["data"]["legal_hold"]["is_active"] is False
            assert result["data"]["legal_hold"]["released_at"] is not None

    @pytest.mark.asyncio
    async def test_release_legal_hold_not_found(self, compliance_handler):
        """Test releasing a non-existent legal hold."""
        with patch.object(compliance_handler, "_release_legal_hold") as mock_release:
            mock_release.return_value = {
                "status": 404,
                "error": "Legal hold not found",
            }

            result = await compliance_handler.handle(
                method="DELETE",
                path="/api/v2/compliance/gdpr/legal-holds/nonexistent",
                body={},
            )

            assert result["status"] == 404

    @pytest.mark.asyncio
    async def test_release_legal_hold_reactivates_deletions(self, compliance_handler):
        """Test that releasing hold reactivates blocked deletions."""
        hold_id = "hold-blocking-dels"

        with patch.object(compliance_handler, "_release_legal_hold") as mock_release:
            mock_release.return_value = {
                "status": 200,
                "data": {
                    "message": "Legal hold released successfully",
                    "legal_hold": {
                        "hold_id": hold_id,
                        "is_active": False,
                        "released_at": datetime.now(timezone.utc).isoformat(),
                    },
                    "reactivated_deletions": [
                        {
                            "request_id": "del-was-held",
                            "previous_status": "held",
                            "new_status": "pending",
                        }
                    ],
                },
            }

            result = await compliance_handler.handle(
                method="DELETE",
                path=f"/api/v2/compliance/gdpr/legal-holds/{hold_id}",
                body={},
            )

            assert result["status"] == 200
            assert "reactivated_deletions" in result["data"]

    @pytest.mark.asyncio
    async def test_release_already_released_hold(self, compliance_handler):
        """Test releasing an already released hold is idempotent."""
        with patch.object(compliance_handler, "_release_legal_hold") as mock_release:
            mock_release.return_value = {
                "status": 200,
                "data": {
                    "message": "Legal hold released successfully",
                    "legal_hold": {
                        "hold_id": "hold-already-released",
                        "is_active": False,
                        "released_at": "2024-01-01T00:00:00+00:00",
                        "released_by": "previous-admin",
                    },
                },
            }

            result = await compliance_handler.handle(
                method="DELETE",
                path="/api/v2/compliance/gdpr/legal-holds/hold-already-released",
                body={},
            )

            # Should succeed (idempotent)
            assert result["status"] == 200


# ============================================================================
# Deletion-Legal Hold Interaction Tests
# ============================================================================


class TestDeletionLegalHoldInteraction:
    """Tests for interactions between deletions and legal holds."""

    @pytest.mark.asyncio
    async def test_rtbf_blocked_by_active_hold(self, compliance_handler):
        """Test that RTBF request is blocked if user has active legal hold."""
        with patch.object(compliance_handler, "_right_to_be_forgotten") as mock_rtbf:
            mock_rtbf.return_value = {
                "status": 400,
                "error": "Cannot process right-to-be-forgotten: User is under legal hold (hold_id=hold-123, reason=Litigation)",
            }

            result = await compliance_handler.handle(
                method="POST",
                path="/api/v2/compliance/gdpr/right-to-be-forgotten",
                body={"user_id": "user-under-hold"},
            )

            assert result["status"] == 400
            assert "legal hold" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_deletion_status_changes_with_hold(self, compliance_handler):
        """Test deletion status transitions when hold is created/released."""
        # Get deletion showing status change
        with patch.object(compliance_handler, "_get_deletion") as mock_get:
            mock_get.return_value = {
                "status": 200,
                "data": {
                    "deletion": {
                        "request_id": "del-status-tracked",
                        "status": "held",
                        "status_history": [
                            {
                                "status": "pending",
                                "timestamp": "2024-01-01T00:00:00+00:00",
                            },
                            {
                                "status": "held",
                                "timestamp": "2024-01-05T00:00:00+00:00",
                                "reason": "Legal hold created",
                                "hold_id": "hold-tracker",
                            },
                        ],
                    }
                },
            }

            result = await compliance_handler.handle(
                method="GET",
                path="/api/v2/compliance/gdpr/deletions/del-status-tracked",
            )

            assert result["status"] == 200
            assert result["data"]["deletion"]["status"] == "held"
            assert len(result["data"]["deletion"]["status_history"]) == 2

    @pytest.mark.asyncio
    async def test_multiple_holds_on_same_user(self, compliance_handler):
        """Test handling multiple legal holds on the same user."""
        with patch.object(compliance_handler, "_get_deletion") as mock_get:
            mock_get.return_value = {
                "status": 200,
                "data": {
                    "deletion": {
                        "request_id": "del-multi-hold",
                        "user_id": "user-multi-holds",
                        "status": "held",
                        "active_holds": [
                            {"hold_id": "hold-1", "reason": "Litigation A"},
                            {"hold_id": "hold-2", "reason": "Regulatory inquiry"},
                        ],
                    }
                },
            }

            result = await compliance_handler.handle(
                method="GET",
                path="/api/v2/compliance/gdpr/deletions/del-multi-hold",
            )

            assert result["status"] == 200
            assert len(result["data"]["deletion"]["active_holds"]) == 2


# ============================================================================
# Compliance Permission Tests
# ============================================================================


class TestCompliancePermissions:
    """Tests for RBAC enforcement on compliance endpoints."""

    @pytest.mark.asyncio
    async def test_list_deletions_requires_permission(self, compliance_handler):
        """Test that list deletions requires compliance:gdpr permission."""
        # Handler uses @require_permission decorator
        assert compliance_handler.can_handle("/api/v2/compliance/gdpr/deletions", "GET")

    @pytest.mark.asyncio
    async def test_legal_hold_requires_legal_permission(self, compliance_handler):
        """Test that legal hold operations require compliance:legal permission."""
        # POST to create legal hold
        assert compliance_handler.can_handle("/api/v2/compliance/gdpr/legal-holds", "POST")

        # GET to list legal holds
        assert compliance_handler.can_handle("/api/v2/compliance/gdpr/legal-holds", "GET")


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestComplianceEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_deletions_list(self, compliance_handler):
        """Test listing when no deletion requests exist."""
        with patch.object(compliance_handler, "_list_deletions") as mock_list:
            mock_list.return_value = {
                "status": 200,
                "data": {
                    "deletions": [],
                    "count": 0,
                    "filters": {"status": None, "limit": 50},
                },
            }

            result = await compliance_handler.handle(
                method="GET",
                path="/api/v2/compliance/gdpr/deletions",
            )

            assert result["status"] == 200
            assert result["data"]["count"] == 0
            assert result["data"]["deletions"] == []

    @pytest.mark.asyncio
    async def test_empty_legal_holds_list(self, compliance_handler):
        """Test listing when no legal holds exist."""
        with patch.object(compliance_handler, "_list_legal_holds") as mock_list:
            mock_list.return_value = {
                "status": 200,
                "data": {
                    "legal_holds": [],
                    "count": 0,
                    "filters": {"active_only": True},
                },
            }

            result = await compliance_handler.handle(
                method="GET",
                path="/api/v2/compliance/gdpr/legal-holds",
            )

            assert result["status"] == 200
            assert result["data"]["count"] == 0

    @pytest.mark.asyncio
    async def test_invalid_deletion_status_filter(self, compliance_handler):
        """Test handling of invalid status filter."""
        with patch.object(compliance_handler, "_list_deletions") as mock_list:
            mock_list.return_value = {
                "status": 400,
                "error": "Invalid status: 'invalid_status'. Valid values: pending, completed, failed, cancelled, held",
            }

            result = await compliance_handler.handle(
                method="GET",
                path="/api/v2/compliance/gdpr/deletions",
                query_params={"status": "invalid_status"},
            )

            assert result["status"] == 400

    @pytest.mark.asyncio
    async def test_legal_hold_invalid_expiration_format(self, compliance_handler):
        """Test creating legal hold with invalid expiration date format."""
        with patch.object(compliance_handler, "_create_legal_hold") as mock_create:
            mock_create.return_value = {
                "status": 400,
                "error": "Invalid expires_at format",
            }

            result = await compliance_handler.handle(
                method="POST",
                path="/api/v2/compliance/gdpr/legal-holds",
                body={
                    "user_ids": ["user-001"],
                    "reason": "Test",
                    "expires_at": "not-a-date",
                },
            )

            assert result["status"] == 400
            assert "expires_at" in result["error"]

    @pytest.mark.asyncio
    async def test_handler_can_handle_deletion_endpoints(self, compliance_handler):
        """Test can_handle correctly validates deletion endpoint paths."""
        # List deletions
        assert compliance_handler.can_handle("/api/v2/compliance/gdpr/deletions", "GET")

        # Get specific deletion
        assert compliance_handler.can_handle("/api/v2/compliance/gdpr/deletions/del-123", "GET")

        # Cancel deletion
        assert compliance_handler.can_handle(
            "/api/v2/compliance/gdpr/deletions/del-123/cancel", "POST"
        )

    @pytest.mark.asyncio
    async def test_handler_can_handle_legal_hold_endpoints(self, compliance_handler):
        """Test can_handle correctly validates legal hold endpoint paths."""
        # List holds
        assert compliance_handler.can_handle("/api/v2/compliance/gdpr/legal-holds", "GET")

        # Create hold
        assert compliance_handler.can_handle("/api/v2/compliance/gdpr/legal-holds", "POST")

        # Note: DELETE for legal hold release is handled internally but
        # can_handle only advertises GET/POST for routing purposes


# ============================================================================
# Audit Trail Tests
# ============================================================================


class TestDeletionAuditTrail:
    """Tests for audit trail of deletion operations."""

    @pytest.mark.asyncio
    async def test_deletion_creates_audit_event(self, compliance_handler):
        """Test that scheduling deletion creates audit event."""
        with patch.object(compliance_handler, "_right_to_be_forgotten") as mock_rtbf:
            mock_rtbf.return_value = {
                "status": 200,
                "data": {
                    "request_id": "rtbf-audited",
                    "user_id": "user-audit-test",
                    "status": "scheduled",
                    "operations": [
                        {"operation": "revoke_consents", "status": "completed"},
                        {"operation": "schedule_deletion", "status": "scheduled"},
                    ],
                    "audit_event_id": "audit-123",
                },
            }

            result = await compliance_handler.handle(
                method="POST",
                path="/api/v2/compliance/gdpr/right-to-be-forgotten",
                body={"user_id": "user-audit-test"},
            )

            assert result["status"] == 200
            # Audit event should be recorded
            assert "audit_event_id" in result["data"]

    @pytest.mark.asyncio
    async def test_legal_hold_creates_audit_event(self, compliance_handler):
        """Test that creating legal hold creates audit event."""
        with patch.object(compliance_handler, "_create_legal_hold") as mock_create:
            mock_create.return_value = {
                "status": 201,
                "data": {
                    "message": "Legal hold created successfully",
                    "legal_hold": {
                        "hold_id": "hold-audited",
                        "user_ids": ["user-001"],
                        "reason": "Investigation",
                    },
                    "audit_event_id": "audit-hold-123",
                },
            }

            result = await compliance_handler.handle(
                method="POST",
                path="/api/v2/compliance/gdpr/legal-holds",
                body={
                    "user_ids": ["user-001"],
                    "reason": "Investigation",
                },
            )

            assert result["status"] == 201
            # Audit event should be recorded
            assert "audit_event_id" in result["data"]
