"""Tests for autonomous approvals handler."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from aragora.server.handlers.autonomous import approvals


# =============================================================================
# Mock Classes
# =============================================================================


class MockApprovalStatus:
    """Mock approval status enum."""

    PENDING = MagicMock(value="pending")
    APPROVED = MagicMock(value="approved")
    REJECTED = MagicMock(value="rejected")


class MockApprovalRequest:
    """Mock approval request for testing."""

    def __init__(
        self,
        id: str = "req-001",
        title: str = "Test Request",
        description: str = "Test description",
        changes: list = None,
        risk_level: str = "low",
        requested_by: str = "test-user",
        timeout_seconds: int = 300,
        status_value: str = "pending",
        approved_by: str = None,
        approved_at=None,
        rejection_reason: str = None,
        metadata: dict = None,
    ):
        self.id = id
        self.title = title
        self.description = description
        self.changes = changes or []
        self.risk_level = risk_level
        self.requested_at = datetime.now()
        self.requested_by = requested_by
        self.timeout_seconds = timeout_seconds
        self.status = MagicMock(value=status_value)
        self.approved_by = approved_by
        self.approved_at = approved_at
        self.rejection_reason = rejection_reason
        self.metadata = metadata or {}


class MockApprovalFlow:
    """Mock ApprovalFlow for testing."""

    def __init__(self):
        self._requests = {}

    def list_pending(self):
        return [r for r in self._requests.values() if r.status.value == "pending"]

    def _load_request(self, request_id):
        return self._requests.get(request_id)

    def approve(self, request_id, approved_by):
        req = self._requests.get(request_id)
        if not req:
            raise ValueError("Request not found")
        req.status = MagicMock(value="approved")
        req.approved_by = approved_by
        req.approved_at = datetime.now()
        return req

    def reject(self, request_id, rejected_by, reason):
        req = self._requests.get(request_id)
        if not req:
            raise ValueError("Request not found")
        req.status = MagicMock(value="rejected")
        req.approved_by = rejected_by
        req.rejection_reason = reason
        return req


class MockAuthContext:
    """Mock authorization context."""

    def __init__(self, user_id="test-user", permissions=None):
        self.user_id = user_id
        self.permissions = permissions or {"approval.grant"}


class MockPermissionDecision:
    """Mock permission decision."""

    def __init__(self, allowed=True, reason=None):
        self.allowed = allowed
        self.reason = reason or ""


class MockPermissionChecker:
    """Mock permission checker."""

    def check_permission(self, ctx, permission):
        return MockPermissionDecision(allowed=True)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_flow():
    """Create mock approval flow."""
    return MockApprovalFlow()


@pytest.fixture
def mock_auth_context():
    """Create mock auth context."""
    return MockAuthContext()


@pytest.fixture
def mock_permission_checker():
    """Create mock permission checker."""
    return MockPermissionChecker()


# =============================================================================
# Test ApprovalHandler.list_pending
# =============================================================================


class TestApprovalHandlerListPending:
    """Tests for GET /api/autonomous/approvals/pending endpoint."""

    @pytest.mark.asyncio
    async def test_list_pending_empty(self, mock_flow, mock_auth_context):
        """Should return empty list when no pending approvals."""
        with (
            patch.object(approvals, "get_approval_flow", return_value=mock_flow),
            patch.object(
                approvals,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
        ):
            request = MagicMock()
            response = await approvals.ApprovalHandler.list_pending(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["pending"] == []
            assert body["count"] == 0

    @pytest.mark.asyncio
    async def test_list_pending_with_requests(self, mock_flow, mock_auth_context):
        """Should return pending approval requests."""
        mock_flow._requests = {
            "req-1": MockApprovalRequest(id="req-1", title="Request 1"),
            "req-2": MockApprovalRequest(id="req-2", title="Request 2"),
        }

        with (
            patch.object(approvals, "get_approval_flow", return_value=mock_flow),
            patch.object(
                approvals,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
        ):
            request = MagicMock()
            response = await approvals.ApprovalHandler.list_pending(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert len(body["pending"]) == 2

    @pytest.mark.asyncio
    async def test_list_pending_unauthorized(self, mock_flow):
        """Should return 401 when unauthorized."""
        with (
            patch.object(approvals, "get_approval_flow", return_value=mock_flow),
            patch.object(
                approvals,
                "get_auth_context",
                AsyncMock(side_effect=approvals.UnauthorizedError("Not authenticated")),
            ),
        ):
            request = MagicMock()
            response = await approvals.ApprovalHandler.list_pending(request)

            assert response.status == 401


# =============================================================================
# Test ApprovalHandler.get_request
# =============================================================================


class TestApprovalHandlerGetRequest:
    """Tests for GET /api/autonomous/approvals/{request_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_request_success(self, mock_flow, mock_auth_context):
        """Should return approval request details."""
        mock_flow._requests = {"req-1": MockApprovalRequest(id="req-1")}

        with (
            patch.object(approvals, "get_approval_flow", return_value=mock_flow),
            patch.object(
                approvals,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "req-1"

            response = await approvals.ApprovalHandler.get_request(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["request"]["id"] == "req-1"

    @pytest.mark.asyncio
    async def test_get_request_not_found(self, mock_flow, mock_auth_context):
        """Should return 404 for non-existent request."""
        with (
            patch.object(approvals, "get_approval_flow", return_value=mock_flow),
            patch.object(
                approvals,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "non-existent"

            response = await approvals.ApprovalHandler.get_request(request)

            assert response.status == 404


# =============================================================================
# Test ApprovalHandler.approve
# =============================================================================


class TestApprovalHandlerApprove:
    """Tests for POST /api/autonomous/approvals/{request_id}/approve endpoint."""

    @pytest.mark.asyncio
    async def test_approve_success(self, mock_flow, mock_auth_context, mock_permission_checker):
        """Should approve request successfully."""
        mock_flow._requests = {"req-1": MockApprovalRequest(id="req-1")}

        with (
            patch.object(approvals, "get_approval_flow", return_value=mock_flow),
            patch.object(
                approvals,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                approvals,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "req-1"
            request.json = AsyncMock(return_value={})

            response = await approvals.ApprovalHandler.approve(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["request"]["status"] == "approved"

    @pytest.mark.asyncio
    async def test_approve_not_found(self, mock_flow, mock_auth_context, mock_permission_checker):
        """Should return 404 for non-existent request."""
        with (
            patch.object(approvals, "get_approval_flow", return_value=mock_flow),
            patch.object(
                approvals,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                approvals,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "non-existent"
            request.json = AsyncMock(return_value={})

            response = await approvals.ApprovalHandler.approve(request)

            assert response.status == 404

    @pytest.mark.asyncio
    async def test_approve_forbidden(self, mock_flow, mock_auth_context):
        """Should return 403 when permission denied."""
        mock_checker = MockPermissionChecker()
        mock_checker.check_permission = MagicMock(
            return_value=MockPermissionDecision(allowed=False, reason="No permission")
        )

        with (
            patch.object(approvals, "get_approval_flow", return_value=mock_flow),
            patch.object(
                approvals,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                approvals,
                "get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "req-1"
            request.json = AsyncMock(return_value={})

            response = await approvals.ApprovalHandler.approve(request)

            assert response.status == 403


# =============================================================================
# Test ApprovalHandler.reject
# =============================================================================


class TestApprovalHandlerReject:
    """Tests for POST /api/autonomous/approvals/{request_id}/reject endpoint."""

    @pytest.mark.asyncio
    async def test_reject_success(self, mock_flow, mock_auth_context, mock_permission_checker):
        """Should reject request successfully."""
        mock_flow._requests = {"req-1": MockApprovalRequest(id="req-1")}

        with (
            patch.object(approvals, "get_approval_flow", return_value=mock_flow),
            patch.object(
                approvals,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                approvals,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "req-1"
            request.json = AsyncMock(return_value={"reason": "Not approved"})

            response = await approvals.ApprovalHandler.reject(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["request"]["status"] == "rejected"


# =============================================================================
# Test Route Registration
# =============================================================================


class TestApprovalHandlerRoutes:
    """Tests for route registration."""

    def test_register_routes(self):
        """Should register all approval routes."""
        app = web.Application()
        approvals.ApprovalHandler.register_routes(app)

        routes = [r.resource.canonical for r in app.router.routes()]
        assert "/api/v1/autonomous/approvals/pending" in routes
        assert "/api/v1/autonomous/approvals/{request_id}" in routes
        assert "/api/v1/autonomous/approvals/{request_id}/approve" in routes
        assert "/api/v1/autonomous/approvals/{request_id}/reject" in routes


# =============================================================================
# Test Global Functions
# =============================================================================


class TestApprovalFlowSingleton:
    """Tests for approval flow singleton functions."""

    def test_get_approval_flow_creates_singleton(self):
        """get_approval_flow should return same instance."""
        approvals._approval_flow = None

        flow1 = approvals.get_approval_flow()
        flow2 = approvals.get_approval_flow()

        assert flow1 is flow2

        # Clean up
        approvals._approval_flow = None

    def test_set_approval_flow(self):
        """set_approval_flow should update the global instance."""
        mock = MockApprovalFlow()
        approvals.set_approval_flow(mock)

        assert approvals.get_approval_flow() is mock

        # Clean up
        approvals._approval_flow = None
