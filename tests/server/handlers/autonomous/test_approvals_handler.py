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


# =============================================================================
# Additional Tests - Circuit Breaker
# =============================================================================


class MockCircuitBreaker:
    """Mock circuit breaker for testing."""

    def __init__(self, can_exec: bool = True):
        self._can_execute = can_exec

    def can_execute(self) -> bool:
        return self._can_execute


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker protection."""

    @pytest.mark.asyncio
    async def test_list_pending_circuit_breaker_open(self, mock_flow):
        """Should return 503 when circuit breaker is open."""
        cb = MockCircuitBreaker(can_exec=False)
        with patch.object(approvals, "_get_circuit_breaker", return_value=cb):
            request = MagicMock()
            response = await approvals.ApprovalHandler.list_pending(request)

            assert response.status == 503
            body = json.loads(response.body)
            assert "unavailable" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_get_request_circuit_breaker_open(self, mock_flow):
        """Should return 503 when circuit breaker is open."""
        cb = MockCircuitBreaker(can_exec=False)
        with patch.object(approvals, "_get_circuit_breaker", return_value=cb):
            request = MagicMock()
            request.match_info.get.return_value = "req-1"
            response = await approvals.ApprovalHandler.get_request(request)

            assert response.status == 503

    @pytest.mark.asyncio
    async def test_approve_circuit_breaker_open(self, mock_flow):
        """Should return 503 when circuit breaker is open."""
        cb = MockCircuitBreaker(can_exec=False)
        with patch.object(approvals, "_get_circuit_breaker", return_value=cb):
            request = MagicMock()
            request.match_info.get.return_value = "req-1"
            response = await approvals.ApprovalHandler.approve(request)

            assert response.status == 503

    @pytest.mark.asyncio
    async def test_reject_circuit_breaker_open(self, mock_flow):
        """Should return 503 when circuit breaker is open."""
        cb = MockCircuitBreaker(can_exec=False)
        with patch.object(approvals, "_get_circuit_breaker", return_value=cb):
            request = MagicMock()
            request.match_info.get.return_value = "req-1"
            response = await approvals.ApprovalHandler.reject(request)

            assert response.status == 503

    def test_circuit_breaker_singleton(self):
        """Should return same circuit breaker instance."""
        approvals._approval_circuit_breaker = None

        with patch.object(approvals, "get_circuit_breaker") as mock_get_cb:
            mock_cb = MockCircuitBreaker()
            mock_get_cb.return_value = mock_cb

            cb1 = approvals._get_circuit_breaker()
            cb2 = approvals._get_circuit_breaker()

            assert cb1 is cb2

        approvals._approval_circuit_breaker = None


# =============================================================================
# Additional Tests - RBAC and Permissions
# =============================================================================


class TestRBACPermissions:
    """Tests for RBAC permission handling."""

    @pytest.mark.asyncio
    async def test_list_pending_with_admin_bypasses_rbac(self, mock_flow):
        """Admin users should bypass RBAC checks."""
        mock_auth = MockAuthContext(user_id="admin-user", permissions=set())
        mock_auth.roles = ["admin"]

        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(approvals, "_get_circuit_breaker", return_value=cb),
            patch.object(approvals, "get_approval_flow", return_value=mock_flow),
            patch.object(approvals, "get_auth_context", AsyncMock(return_value=mock_auth)),
        ):
            request = MagicMock()
            response = await approvals.ApprovalHandler.list_pending(request)

            assert response.status == 200

    @pytest.mark.asyncio
    async def test_list_pending_owner_bypasses_rbac(self, mock_flow):
        """Owner users should bypass RBAC checks."""
        mock_auth = MockAuthContext(user_id="owner-user", permissions=set())
        mock_auth.roles = ["owner"]

        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(approvals, "_get_circuit_breaker", return_value=cb),
            patch.object(approvals, "get_approval_flow", return_value=mock_flow),
            patch.object(approvals, "get_auth_context", AsyncMock(return_value=mock_auth)),
        ):
            request = MagicMock()
            response = await approvals.ApprovalHandler.list_pending(request)

            assert response.status == 200

    @pytest.mark.asyncio
    async def test_get_request_forbidden_without_permission(self, mock_flow):
        """Should return 403 for users without read permission."""
        mock_auth = MockAuthContext(user_id="user", permissions={"some:other:permission"})
        mock_auth.roles = []

        mock_checker = MockPermissionChecker()
        mock_checker.check_permission = MagicMock(
            return_value=MockPermissionDecision(allowed=False, reason="No permission")
        )

        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(approvals, "_get_circuit_breaker", return_value=cb),
            patch.object(approvals, "get_approval_flow", return_value=mock_flow),
            patch.object(approvals, "get_auth_context", AsyncMock(return_value=mock_auth)),
            patch.object(approvals, "get_permission_checker", return_value=mock_checker),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "req-1"
            response = await approvals.ApprovalHandler.get_request(request)

            assert response.status == 403


# =============================================================================
# Additional Tests - Request Details
# =============================================================================


class TestRequestDetails:
    """Tests for request detail handling."""

    @pytest.mark.asyncio
    async def test_get_request_returns_all_fields(self, mock_flow, mock_auth_context):
        """Should return all request fields."""
        test_request = MockApprovalRequest(
            id="req-detail",
            title="Detailed Request",
            description="Full description",
            changes=["change1", "change2"],
            risk_level="high",
            requested_by="requester",
            timeout_seconds=600,
            metadata={"key": "value"},
        )
        mock_flow._requests = {"req-detail": test_request}

        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(approvals, "_get_circuit_breaker", return_value=cb),
            patch.object(approvals, "get_approval_flow", return_value=mock_flow),
            patch.object(approvals, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "req-detail"

            response = await approvals.ApprovalHandler.get_request(request)

            assert response.status == 200
            body = json.loads(response.body)
            req = body["request"]
            assert req["title"] == "Detailed Request"
            assert req["description"] == "Full description"
            assert req["risk_level"] == "high"
            assert req["timeout_seconds"] == 600
            assert "requested_at" in req

    @pytest.mark.asyncio
    async def test_list_pending_filters_non_pending(self, mock_flow, mock_auth_context):
        """Should only return pending requests."""
        mock_flow._requests = {
            "pending": MockApprovalRequest(id="pending", status_value="pending"),
            "approved": MockApprovalRequest(id="approved", status_value="approved"),
            "rejected": MockApprovalRequest(id="rejected", status_value="rejected"),
        }

        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(approvals, "_get_circuit_breaker", return_value=cb),
            patch.object(approvals, "get_approval_flow", return_value=mock_flow),
            patch.object(approvals, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
        ):
            request = MagicMock()
            response = await approvals.ApprovalHandler.list_pending(request)

            body = json.loads(response.body)
            assert body["count"] == 1
            assert body["pending"][0]["id"] == "pending"


# =============================================================================
# Additional Tests - Approve/Reject Flow
# =============================================================================


class TestApproveRejectFlow:
    """Tests for approve and reject operations."""

    @pytest.mark.asyncio
    async def test_approve_with_custom_approved_by(
        self, mock_flow, mock_auth_context, mock_permission_checker
    ):
        """Should use custom approved_by if provided."""
        mock_flow._requests = {"req-1": MockApprovalRequest(id="req-1")}

        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(approvals, "_get_circuit_breaker", return_value=cb),
            patch.object(approvals, "get_approval_flow", return_value=mock_flow),
            patch.object(approvals, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(approvals, "get_permission_checker", return_value=mock_permission_checker),
            patch.object(
                approvals,
                "parse_json_body",
                AsyncMock(return_value=({"approved_by": "custom-approver"}, None)),
            ),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "req-1"

            response = await approvals.ApprovalHandler.approve(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["request"]["approved_by"] == "custom-approver"

    @pytest.mark.asyncio
    async def test_reject_with_reason(self, mock_flow, mock_auth_context, mock_permission_checker):
        """Should store rejection reason."""
        mock_flow._requests = {"req-1": MockApprovalRequest(id="req-1")}

        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(approvals, "_get_circuit_breaker", return_value=cb),
            patch.object(approvals, "get_approval_flow", return_value=mock_flow),
            patch.object(approvals, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(approvals, "get_permission_checker", return_value=mock_permission_checker),
            patch.object(
                approvals,
                "parse_json_body",
                AsyncMock(return_value=({"reason": "Security concern"}, None)),
            ),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "req-1"

            response = await approvals.ApprovalHandler.reject(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["request"]["rejection_reason"] == "Security concern"

    @pytest.mark.asyncio
    async def test_reject_default_reason(
        self, mock_flow, mock_auth_context, mock_permission_checker
    ):
        """Should use default reason if not provided."""
        mock_flow._requests = {"req-1": MockApprovalRequest(id="req-1")}

        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(approvals, "_get_circuit_breaker", return_value=cb),
            patch.object(approvals, "get_approval_flow", return_value=mock_flow),
            patch.object(approvals, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(approvals, "get_permission_checker", return_value=mock_permission_checker),
            patch.object(approvals, "parse_json_body", AsyncMock(return_value=({}, None))),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "req-1"

            response = await approvals.ApprovalHandler.reject(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["request"]["rejection_reason"] == "No reason provided"

    @pytest.mark.asyncio
    async def test_reject_not_found(self, mock_flow, mock_auth_context, mock_permission_checker):
        """Should return 404 for non-existent request when rejecting."""
        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(approvals, "_get_circuit_breaker", return_value=cb),
            patch.object(approvals, "get_approval_flow", return_value=mock_flow),
            patch.object(approvals, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(approvals, "get_permission_checker", return_value=mock_permission_checker),
            patch.object(
                approvals, "parse_json_body", AsyncMock(return_value=({"reason": "test"}, None))
            ),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "non-existent"

            response = await approvals.ApprovalHandler.reject(request)

            assert response.status == 404


# =============================================================================
# Additional Tests - Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_list_pending_internal_error(self, mock_auth_context):
        """Should return 500 on internal error."""
        mock_flow = MockApprovalFlow()
        mock_flow.list_pending = MagicMock(side_effect=Exception("Internal error"))

        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(approvals, "_get_circuit_breaker", return_value=cb),
            patch.object(approvals, "get_approval_flow", return_value=mock_flow),
            patch.object(approvals, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
        ):
            request = MagicMock()
            response = await approvals.ApprovalHandler.list_pending(request)

            assert response.status == 500
            body = json.loads(response.body)
            assert body["success"] is False
            assert "error" in body

    @pytest.mark.asyncio
    async def test_get_request_internal_error(self, mock_auth_context):
        """Should return 500 on internal error."""
        mock_flow = MockApprovalFlow()
        mock_flow._load_request = MagicMock(side_effect=Exception("Database error"))

        cb = MockCircuitBreaker(can_exec=True)

        with (
            patch.object(approvals, "_get_circuit_breaker", return_value=cb),
            patch.object(approvals, "get_approval_flow", return_value=mock_flow),
            patch.object(approvals, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "req-1"
            response = await approvals.ApprovalHandler.get_request(request)

            assert response.status == 500

    @pytest.mark.asyncio
    async def test_approve_json_parse_error(
        self, mock_flow, mock_auth_context, mock_permission_checker
    ):
        """Should handle JSON parse errors."""
        cb = MockCircuitBreaker(can_exec=True)

        error_response = MagicMock()
        error_response.status = 400

        with (
            patch.object(approvals, "_get_circuit_breaker", return_value=cb),
            patch.object(approvals, "get_approval_flow", return_value=mock_flow),
            patch.object(approvals, "get_auth_context", AsyncMock(return_value=mock_auth_context)),
            patch.object(approvals, "get_permission_checker", return_value=mock_permission_checker),
            patch.object(
                approvals, "parse_json_body", AsyncMock(return_value=(None, error_response))
            ),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "req-1"

            response = await approvals.ApprovalHandler.approve(request)

            assert response.status == 400


# =============================================================================
# Additional Tests - Auth Context Enrichment
# =============================================================================


class TestAuthContextEnrichment:
    """Tests for auth context enrichment."""

    def test_ensure_auth_context_adds_missing_attrs(self):
        """Should add missing attributes to auth context."""
        auth_ctx = MagicMock()
        auth_ctx.roles = None
        del auth_ctx.org_id
        del auth_ctx.api_key_scope
        del auth_ctx.workspace_id
        del auth_ctx.permissions

        approvals._ensure_auth_context(auth_ctx)

        assert auth_ctx.roles == []
        assert auth_ctx.org_id is None
        assert auth_ctx.api_key_scope is None
        assert auth_ctx.workspace_id is None
        assert auth_ctx.permissions == set()

    def test_ensure_auth_context_normalizes_permissions(self):
        """Should normalize permissions to set."""
        auth_ctx = MagicMock()
        auth_ctx.roles = ["user"]
        auth_ctx.org_id = "org-1"
        auth_ctx.api_key_scope = None
        auth_ctx.workspace_id = "ws-1"
        auth_ctx.permissions = ["perm1", "perm2"]

        approvals._ensure_auth_context(auth_ctx)

        assert isinstance(auth_ctx.permissions, set)

    def test_ensure_auth_context_upgrades_approval_permissions(self):
        """Should upgrade legacy approval permissions."""
        auth_ctx = MagicMock()
        auth_ctx.roles = ["user"]
        auth_ctx.org_id = "org-1"
        auth_ctx.api_key_scope = None
        auth_ctx.workspace_id = "ws-1"
        auth_ctx.permissions = {"approval.grant"}

        approvals._ensure_auth_context(auth_ctx)

        assert approvals.AUTONOMOUS_READ_PERMISSION in auth_ctx.permissions
        assert approvals.AUTONOMOUS_APPROVE_PERMISSION in auth_ctx.permissions

    def test_is_admin_returns_true_for_admin(self):
        """Should identify admin users."""
        auth_ctx = MagicMock()
        auth_ctx.roles = ["admin"]

        assert approvals._is_admin(auth_ctx) is True

    def test_is_admin_returns_true_for_owner(self):
        """Should identify owner users."""
        auth_ctx = MagicMock()
        auth_ctx.roles = ["owner"]

        assert approvals._is_admin(auth_ctx) is True

    def test_is_admin_returns_false_for_regular_user(self):
        """Should return false for regular users."""
        auth_ctx = MagicMock()
        auth_ctx.roles = ["user", "editor"]

        assert approvals._is_admin(auth_ctx) is False


# =============================================================================
# Additional Tests - Handler Initialization
# =============================================================================


class TestHandlerInitialization:
    """Tests for handler initialization."""

    def test_handler_init_with_context(self):
        """Should initialize with provided context."""
        ctx = {"key": "value"}
        handler = approvals.ApprovalHandler(ctx)
        assert handler.ctx == ctx

    def test_handler_init_without_context(self):
        """Should initialize with empty context if not provided."""
        handler = approvals.ApprovalHandler()
        assert handler.ctx == {}

    def test_handler_init_with_none_context(self):
        """Should initialize with empty context if None provided."""
        handler = approvals.ApprovalHandler(None)
        assert handler.ctx == {}


# =============================================================================
# Additional Tests - Route Registration with Custom Prefix
# =============================================================================


class TestRouteRegistrationCustomPrefix:
    """Tests for route registration with custom prefix."""

    def test_register_routes_custom_prefix(self):
        """Should register routes with custom prefix."""
        app = web.Application()
        approvals.ApprovalHandler.register_routes(app, prefix="/api/v2/custom")

        routes = [r.resource.canonical for r in app.router.routes()]
        assert "/api/v2/custom/approvals/pending" in routes
        assert "/api/v2/custom/approvals/{request_id}" in routes
        assert "/api/v2/custom/approvals/{request_id}/approve" in routes
        assert "/api/v2/custom/approvals/{request_id}/reject" in routes
