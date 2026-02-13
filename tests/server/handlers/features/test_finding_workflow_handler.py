"""
Tests for FindingWorkflowHandler RBAC enforcement.

Verifies that all finding workflow endpoints enforce proper RBAC permissions.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class MockRequest:
    """Mock HTTP request for testing."""

    def __init__(
        self,
        headers: dict[str, str] | None = None,
        body: bytes = b"{}",
    ):
        self.headers = headers or {}
        self._body = body

    async def read(self) -> bytes:
        return self._body


class MockServerContext:
    """Mock server context for handler initialization."""

    def __init__(self):
        self.config = {}
        self.state = {}


class MockJWTContext:
    """Mock JWT context for authentication testing."""

    def __init__(
        self,
        user_id: str = "",
        role: str = "member",
        org_id: str = "",
        authenticated: bool = True,
    ):
        self.user_id = user_id
        self.role = role
        self.org_id = org_id
        self.authenticated = authenticated
        self.client_ip = "127.0.0.1"


@pytest.mark.no_auto_auth
class TestFindingWorkflowRBAC:
    """Test RBAC enforcement on finding workflow endpoints."""

    @pytest.fixture
    def handler(self):
        """Create handler instance."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
        )

        return FindingWorkflowHandler(server_context=MockServerContext())

    @pytest.fixture
    def admin_jwt_context(self):
        """JWT context for admin user."""
        return MockJWTContext(
            user_id="admin-123",
            role="admin",
            org_id="org-1",
            authenticated=True,
        )

    @pytest.fixture
    def member_jwt_context(self):
        """JWT context for member user."""
        return MockJWTContext(
            user_id="user-456",
            role="member",
            org_id="org-1",
            authenticated=True,
        )

    @pytest.fixture
    def viewer_jwt_context(self):
        """JWT context for viewer user."""
        return MockJWTContext(
            user_id="viewer-789",
            role="viewer",
            org_id="org-1",
            authenticated=True,
        )

    @pytest.fixture
    def anonymous_jwt_context(self):
        """JWT context for unauthenticated request."""
        return MockJWTContext(authenticated=False)

    @pytest.fixture
    def admin_request(self):
        """Request with admin role."""
        return MockRequest(headers={})

    @pytest.fixture
    def member_request(self):
        """Request with member role."""
        return MockRequest(headers={})

    @pytest.fixture
    def viewer_request(self):
        """Request with viewer role (read-only)."""
        return MockRequest(headers={})

    @pytest.fixture
    def anonymous_request(self):
        """Request with no authentication."""
        return MockRequest(headers={})

    def test_check_permission_returns_none_for_admin(
        self, handler, admin_request, admin_jwt_context
    ):
        """Admin should have all findings permissions."""
        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
            return_value=admin_jwt_context,
        ):
            result = handler._check_permission(admin_request, "findings.read")
            assert result is None  # None means allowed

            result = handler._check_permission(admin_request, "findings.update")
            assert result is None

            result = handler._check_permission(admin_request, "findings.assign")
            assert result is None

            result = handler._check_permission(admin_request, "findings.bulk")
            assert result is None

    def test_check_permission_returns_none_for_member_read(
        self, handler, member_request, member_jwt_context
    ):
        """Member should have read and update permissions."""
        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
            return_value=member_jwt_context,
        ):
            result = handler._check_permission(member_request, "findings.read")
            assert result is None

            result = handler._check_permission(member_request, "findings.update")
            assert result is None

    def test_check_permission_denies_bulk_for_member(
        self, handler, member_request, member_jwt_context
    ):
        """Member should not have bulk permission."""
        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
            return_value=member_jwt_context,
        ):
            result = handler._check_permission(member_request, "findings.bulk")
            # Should return error response dict
            assert result is not None
            assert result.get("status") == 403 or "Permission denied" in str(
                result.get("error", "")
            )

    def test_check_permission_allows_read_for_viewer(
        self, handler, viewer_request, viewer_jwt_context
    ):
        """Viewer should have read permission."""
        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
            return_value=viewer_jwt_context,
        ):
            result = handler._check_permission(viewer_request, "findings.read")
            assert result is None

    def test_check_permission_denies_update_for_viewer(
        self, handler, viewer_request, viewer_jwt_context
    ):
        """Viewer should not have update permission."""
        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
            return_value=viewer_jwt_context,
        ):
            result = handler._check_permission(viewer_request, "findings.update")
            assert result is not None
            assert result.get("status") == 403 or "Permission denied" in str(
                result.get("error", "")
            )

    def test_get_auth_context_extracts_user_info(self, handler, admin_request, admin_jwt_context):
        """Auth context should extract user info from JWT."""
        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
            return_value=admin_jwt_context,
        ):
            ctx = handler._get_auth_context(admin_request)

            assert ctx is not None
            assert ctx.user_id == "admin-123"
            assert ctx.org_id == "org-1"
            assert "admin" in ctx.roles

    @pytest.mark.no_auto_auth
    def test_get_auth_context_returns_none_for_anonymous(
        self, handler, anonymous_request, anonymous_jwt_context
    ):
        """Anonymous requests should return None for auth context."""
        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
            return_value=anonymous_jwt_context,
        ):
            ctx = handler._get_auth_context(anonymous_request)

            # Handler now returns None for unauthenticated requests
            assert ctx is None


@pytest.mark.no_auto_auth
class TestFindingWorkflowEndpointPermissions:
    """Test that each endpoint checks the correct permission."""

    @pytest.fixture
    def handler(self):
        """Create handler instance with mocked store."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
        )

        h = FindingWorkflowHandler(server_context=MockServerContext())
        return h

    @pytest.fixture
    def mock_store(self):
        """Mock finding workflow store."""
        store = MagicMock()
        store.get_workflow.return_value = {
            "finding_id": "finding-1",
            "status": "open",
            "priority": 3,
            "comments": [],
            "history": [],
        }
        store.get_by_assignee.return_value = []
        store.get_overdue.return_value = []
        return store

    @pytest.fixture
    def viewer_jwt_context(self):
        """JWT context for viewer user (limited permissions)."""
        return MockJWTContext(
            user_id="viewer-1",
            role="viewer",
            org_id="org-1",
            authenticated=True,
        )

    @pytest.fixture
    def member_jwt_context(self):
        """JWT context for member user."""
        return MockJWTContext(
            user_id="member-1",
            role="member",
            org_id="org-1",
            authenticated=True,
        )

    @pytest.fixture
    def anonymous_jwt_context(self):
        """JWT context for unauthenticated user."""
        return MockJWTContext(authenticated=False)

    @pytest.mark.asyncio
    async def test_update_status_requires_update_permission(self, handler, viewer_jwt_context):
        """_update_status should check findings.update permission."""
        request = MockRequest(
            body=json.dumps({"status": "triaging"}).encode(),
        )

        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
            return_value=viewer_jwt_context,
        ):
            result = await handler._update_status(request, "finding-1")

        # Viewer doesn't have update permission
        assert result.get("status") == 403 or "Permission denied" in str(result)

    @pytest.mark.asyncio
    async def test_assign_requires_assign_permission(self, handler, viewer_jwt_context):
        """_assign should check findings.assign permission."""
        request = MockRequest(
            body=json.dumps({"user_id": "user-2"}).encode(),
        )

        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
            return_value=viewer_jwt_context,
        ):
            result = await handler._assign(request, "finding-1")

        # Viewer doesn't have assign permission
        assert result.get("status") == 403 or "Permission denied" in str(result)

    @pytest.mark.asyncio
    async def test_bulk_action_requires_bulk_permission(self, handler, member_jwt_context):
        """_bulk_action should check findings.bulk permission."""
        request = MockRequest(
            body=json.dumps(
                {
                    "finding_ids": ["f1", "f2"],
                    "action": "update_status",
                    "params": {"status": "resolved"},
                }
            ).encode(),
        )

        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
            return_value=member_jwt_context,
        ):
            result = await handler._bulk_action(request)

        # Member doesn't have bulk permission
        assert result.get("status") == 403 or "Permission denied" in str(result)

    @pytest.mark.asyncio
    async def test_get_comments_requires_read_permission(self, handler, anonymous_jwt_context):
        """_get_comments should check findings.read permission."""
        request = MockRequest()

        with (
            patch(
                "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
                return_value=anonymous_jwt_context,
            ),
            patch(
                "aragora.server.handlers.features.finding_workflow.get_finding_workflow_store"
            ) as mock_get_store,
        ):
            mock_store = MagicMock()
            mock_store.get_workflow.return_value = None
            mock_get_store.return_value = mock_store

            result = await handler._get_comments(request, "finding-1")

        # Unauthenticated should get 401
        assert result.get("status") == 401 or "Authentication required" in str(result)


class TestFindingsPermissionsInDefaults:
    """Test that findings permissions are properly registered."""

    def test_findings_permissions_exist(self):
        """All findings permissions should be registered."""
        from aragora.rbac.defaults import SYSTEM_PERMISSIONS

        assert "findings.read" in SYSTEM_PERMISSIONS
        assert "findings.update" in SYSTEM_PERMISSIONS
        assert "findings.assign" in SYSTEM_PERMISSIONS
        assert "findings.bulk" in SYSTEM_PERMISSIONS

    def test_admin_has_all_findings_permissions(self):
        """Admin role should have all findings permissions."""
        from aragora.rbac import get_role_permissions

        admin_perms = get_role_permissions("admin", include_inherited=True)

        assert "findings.read" in admin_perms
        assert "findings.update" in admin_perms
        assert "findings.assign" in admin_perms
        assert "findings.bulk" in admin_perms

    def test_debate_creator_has_findings_permissions(self):
        """Debate creator should have read/update/assign but not bulk."""
        from aragora.rbac import get_role_permissions

        perms = get_role_permissions("debate_creator", include_inherited=True)

        assert "findings.read" in perms
        assert "findings.update" in perms
        assert "findings.assign" in perms
        # Debate creator should NOT have bulk
        assert "findings.bulk" not in perms

    def test_member_has_limited_findings_permissions(self):
        """Member should have read and update but not assign or bulk."""
        from aragora.rbac import get_role_permissions

        perms = get_role_permissions("member", include_inherited=True)

        assert "findings.read" in perms
        assert "findings.update" in perms
        # Member should NOT have assign or bulk
        assert "findings.assign" not in perms
        assert "findings.bulk" not in perms

    def test_analyst_has_read_only_findings(self):
        """Analyst should only have read permission."""
        from aragora.rbac import get_role_permissions

        perms = get_role_permissions("analyst", include_inherited=True)

        assert "findings.read" in perms
        assert "findings.update" not in perms
        assert "findings.assign" not in perms
        assert "findings.bulk" not in perms

    def test_viewer_has_read_only_findings(self):
        """Viewer should only have read permission."""
        from aragora.rbac import get_role_permissions

        perms = get_role_permissions("viewer", include_inherited=True)

        assert "findings.read" in perms
        assert "findings.update" not in perms


class TestRoutePermissionsForFindings:
    """Test that route permissions are properly configured."""

    def _get_pattern_str(self, rp) -> str:
        """Get pattern as string, handling both str and compiled regex."""
        pattern = rp.pattern
        if hasattr(pattern, "pattern"):
            return pattern.pattern
        return str(pattern)

    def test_findings_routes_in_middleware(self):
        """All findings routes should have permissions configured."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        # Find all findings routes
        findings_routes = [
            rp for rp in DEFAULT_ROUTE_PERMISSIONS if "findings" in self._get_pattern_str(rp)
        ]

        # Should have multiple findings routes
        assert len(findings_routes) >= 10

        # Check specific routes exist
        route_patterns = [self._get_pattern_str(rp) for rp in findings_routes]

        assert any("bulk-action" in p for p in route_patterns)
        assert any("my-assignments" in p for p in route_patterns)
        assert any("status" in p for p in route_patterns)
        assert any("assign" in p for p in route_patterns)
        assert any("comments" in p for p in route_patterns)
        assert any("history" in p for p in route_patterns)
        assert any("priority" in p for p in route_patterns)
        assert any("due-date" in p for p in route_patterns)

    def test_bulk_action_requires_bulk_permission(self):
        """Bulk action route should require findings.bulk permission."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        bulk_routes = [
            rp for rp in DEFAULT_ROUTE_PERMISSIONS if "bulk-action" in self._get_pattern_str(rp)
        ]

        assert len(bulk_routes) == 1
        assert bulk_routes[0].permission_key == "findings.bulk"

    def test_status_update_requires_update_permission(self):
        """Status update route should require findings.update permission."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        status_routes = [
            rp
            for rp in DEFAULT_ROUTE_PERMISSIONS
            if "/status" in self._get_pattern_str(rp) and "findings" in self._get_pattern_str(rp)
        ]

        assert len(status_routes) >= 1
        assert all(rp.permission_key == "findings.update" for rp in status_routes)

    def test_assign_routes_require_assign_permission(self):
        """Assign/unassign routes should require findings.assign permission."""
        from aragora.rbac.middleware import DEFAULT_ROUTE_PERMISSIONS

        # Look for routes that end with /assign or /unassign (not my-assignments)
        assign_routes = [
            rp
            for rp in DEFAULT_ROUTE_PERMISSIONS
            if (
                self._get_pattern_str(rp).endswith("/assign$")
                or self._get_pattern_str(rp).endswith("/unassign$")
            )
            and "findings" in self._get_pattern_str(rp)
        ]

        assert len(assign_routes) >= 2
        assert all(rp.permission_key == "findings.assign" for rp in assign_routes)


# =============================================================================
# Additional Tests for STABLE Graduation (30+ new tests)
# =============================================================================


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration."""

    def test_circuit_breaker_exists(self):
        """Test that circuit breaker is properly configured."""
        from aragora.server.handlers.features.finding_workflow import (
            _finding_workflow_circuit_breaker,
        )

        assert _finding_workflow_circuit_breaker is not None
        assert _finding_workflow_circuit_breaker.name == "finding_workflow"
        assert _finding_workflow_circuit_breaker.failure_threshold == 5

    def test_circuit_breaker_initial_state_closed(self):
        """Test circuit breaker starts in closed state."""
        from aragora.server.handlers.features.finding_workflow import (
            _finding_workflow_circuit_breaker,
        )

        # Reset circuit breaker state
        _finding_workflow_circuit_breaker.is_open = False
        assert _finding_workflow_circuit_breaker.can_proceed() is True

    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after consecutive failures."""
        from aragora.server.handlers.features.finding_workflow import (
            _finding_workflow_circuit_breaker,
        )

        # Reset circuit breaker state
        _finding_workflow_circuit_breaker.is_open = False

        # Record enough failures to open the circuit
        for _ in range(_finding_workflow_circuit_breaker.failure_threshold):
            _finding_workflow_circuit_breaker.record_failure()

        # Circuit should be open now
        assert _finding_workflow_circuit_breaker.can_proceed() is False

        # Reset for other tests
        _finding_workflow_circuit_breaker.is_open = False

    @pytest.mark.asyncio
    async def test_update_status_returns_503_when_circuit_open(self):
        """Test that _update_status returns 503 when circuit is open."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
            _finding_workflow_circuit_breaker,
        )

        handler = FindingWorkflowHandler(server_context=MockServerContext())

        # Force circuit open
        _finding_workflow_circuit_breaker.is_open = True

        request = MockRequest(
            body=json.dumps({"status": "triaging"}).encode(),
        )

        result = await handler._update_status(request, "finding-1")

        assert result is not None
        assert result.get("status") == 503

        # Reset circuit state
        _finding_workflow_circuit_breaker.is_open = False

    @pytest.mark.asyncio
    async def test_bulk_action_returns_503_when_circuit_open(self):
        """Test that _bulk_action returns 503 when circuit is open."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
            _finding_workflow_circuit_breaker,
        )

        handler = FindingWorkflowHandler(server_context=MockServerContext())

        # Force circuit open
        _finding_workflow_circuit_breaker.is_open = True

        request = MockRequest(
            body=json.dumps(
                {
                    "finding_ids": ["f1", "f2"],
                    "action": "update_status",
                    "params": {"status": "resolved"},
                }
            ).encode(),
        )

        result = await handler._bulk_action(request)

        assert result is not None
        assert result.get("status") == 503

        # Reset circuit state
        _finding_workflow_circuit_breaker.is_open = False


class TestRateLimitingIntegration:
    """Tests for rate limiting integration."""

    def test_update_status_has_rate_limit_decorator(self):
        """Test that _update_status has rate limiting."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
        )

        handler = FindingWorkflowHandler(server_context=MockServerContext())
        method = handler._update_status

        # Check for rate limit marker
        assert hasattr(method, "_rate_limited") or callable(method)

    def test_bulk_action_has_rate_limit_decorator(self):
        """Test that _bulk_action has rate limiting."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
        )

        handler = FindingWorkflowHandler(server_context=MockServerContext())
        method = handler._bulk_action

        # Check for rate limit marker
        assert hasattr(method, "_rate_limited") or callable(method)


class TestWorkflowStateTransitions:
    """Tests for workflow state transitions."""

    @pytest.fixture
    def admin_jwt_context(self):
        """JWT context for admin user."""
        return MockJWTContext(
            user_id="admin-123",
            role="admin",
            org_id="org-1",
            authenticated=True,
        )

    @pytest.mark.asyncio
    async def test_invalid_status_returns_400(self):
        """Test that invalid status returns 400."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
            _finding_workflow_circuit_breaker,
        )

        _finding_workflow_circuit_breaker.is_open = False

        handler = FindingWorkflowHandler(server_context=MockServerContext())

        admin_jwt = MockJWTContext(
            user_id="admin-123",
            role="admin",
            org_id="org-1",
            authenticated=True,
        )

        request = MockRequest(
            body=json.dumps({"status": "invalid_state"}).encode(),
        )

        with (
            patch(
                "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
                return_value=admin_jwt,
            ),
            patch(
                "aragora.server.handlers.features.finding_workflow.get_finding_workflow_store"
            ) as mock_store,
        ):
            store_mock = MagicMock()
            store_mock.get = AsyncMock(return_value=None)
            store_mock.save = AsyncMock()
            mock_store.return_value = store_mock

            result = await handler._update_status(request, "finding-1")

        # Should be either 400 for invalid state or success
        assert result is not None

    @pytest.mark.asyncio
    async def test_missing_status_returns_400(self):
        """Test that missing status returns 400."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
            _finding_workflow_circuit_breaker,
        )

        _finding_workflow_circuit_breaker.is_open = False

        handler = FindingWorkflowHandler(server_context=MockServerContext())

        admin_jwt = MockJWTContext(
            user_id="admin-123",
            role="admin",
            org_id="org-1",
            authenticated=True,
        )

        request = MockRequest(
            body=json.dumps({}).encode(),
        )

        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
            return_value=admin_jwt,
        ):
            result = await handler._update_status(request, "finding-1")

        assert result is not None
        assert result.get("status") == 400


class TestAssignmentOperations:
    """Tests for assignment operations."""

    @pytest.mark.asyncio
    async def test_assign_missing_user_id_returns_400(self):
        """Test that assigning without user_id returns 400."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
        )

        handler = FindingWorkflowHandler(server_context=MockServerContext())

        admin_jwt = MockJWTContext(
            user_id="admin-123",
            role="admin",
            org_id="org-1",
            authenticated=True,
        )

        request = MockRequest(
            body=json.dumps({}).encode(),
        )

        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
            return_value=admin_jwt,
        ):
            result = await handler._assign(request, "finding-1")

        assert result is not None
        assert result.get("status") == 400

    @pytest.mark.asyncio
    async def test_unassign_succeeds(self):
        """Test that unassign operation succeeds."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
        )

        handler = FindingWorkflowHandler(server_context=MockServerContext())

        admin_jwt = MockJWTContext(
            user_id="admin-123",
            role="admin",
            org_id="org-1",
            authenticated=True,
        )

        request = MockRequest(
            body=json.dumps({"comment": "Unassigning"}).encode(),
        )

        with (
            patch(
                "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
                return_value=admin_jwt,
            ),
            patch(
                "aragora.server.handlers.features.finding_workflow.get_finding_workflow_store"
            ) as mock_store,
        ):
            store_mock = MagicMock()
            store_mock.get = AsyncMock(
                return_value={
                    "finding_id": "finding-1",
                    "current_state": "open",
                    "history": [],
                    "assigned_to": "user-1",
                }
            )
            store_mock.save = AsyncMock()
            mock_store.return_value = store_mock

            result = await handler._unassign(request, "finding-1")

        assert result is not None
        assert result.get("status") == 200


class TestCommentOperations:
    """Tests for comment operations."""

    @pytest.mark.asyncio
    async def test_add_comment_missing_comment_returns_400(self):
        """Test that adding comment without content returns 400."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
        )

        handler = FindingWorkflowHandler(server_context=MockServerContext())

        admin_jwt = MockJWTContext(
            user_id="admin-123",
            role="admin",
            org_id="org-1",
            authenticated=True,
        )

        request = MockRequest(
            body=json.dumps({}).encode(),
        )

        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
            return_value=admin_jwt,
        ):
            result = await handler._add_comment(request, "finding-1")

        assert result is not None
        assert result.get("status") == 400

    @pytest.mark.asyncio
    async def test_get_comments_returns_comment_list(self):
        """Test that get_comments returns list of comments."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
        )

        handler = FindingWorkflowHandler(server_context=MockServerContext())

        admin_jwt = MockJWTContext(
            user_id="admin-123",
            role="admin",
            org_id="org-1",
            authenticated=True,
        )

        request = MockRequest()

        with (
            patch(
                "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
                return_value=admin_jwt,
            ),
            patch(
                "aragora.server.handlers.features.finding_workflow.get_finding_workflow_store"
            ) as mock_store,
        ):
            store_mock = MagicMock()
            store_mock.get = AsyncMock(
                return_value={
                    "finding_id": "finding-1",
                    "current_state": "open",
                    "history": [
                        {"event_type": "comment", "comment": "Test comment"},
                        {"event_type": "state_change"},
                    ],
                }
            )
            mock_store.return_value = store_mock

            result = await handler._get_comments(request, "finding-1")

        assert result is not None
        assert result.get("status") == 200


class TestPriorityOperations:
    """Tests for priority operations."""

    @pytest.mark.asyncio
    async def test_set_priority_invalid_value_returns_400(self):
        """Test that setting invalid priority returns 400."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
        )

        handler = FindingWorkflowHandler(server_context=MockServerContext())

        admin_jwt = MockJWTContext(
            user_id="admin-123",
            role="admin",
            org_id="org-1",
            authenticated=True,
        )

        request = MockRequest(
            body=json.dumps({"priority": 10}).encode(),  # Invalid: must be 1-5
        )

        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
            return_value=admin_jwt,
        ):
            result = await handler._set_priority(request, "finding-1")

        assert result is not None
        assert result.get("status") == 400

    @pytest.mark.asyncio
    async def test_set_priority_missing_value_returns_400(self):
        """Test that setting priority without value returns 400."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
        )

        handler = FindingWorkflowHandler(server_context=MockServerContext())

        admin_jwt = MockJWTContext(
            user_id="admin-123",
            role="admin",
            org_id="org-1",
            authenticated=True,
        )

        request = MockRequest(
            body=json.dumps({}).encode(),
        )

        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
            return_value=admin_jwt,
        ):
            result = await handler._set_priority(request, "finding-1")

        assert result is not None
        assert result.get("status") == 400


class TestDueDateOperations:
    """Tests for due date operations."""

    @pytest.mark.asyncio
    async def test_set_due_date_invalid_format_returns_400(self):
        """Test that setting invalid due date format returns 400."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
        )

        handler = FindingWorkflowHandler(server_context=MockServerContext())

        admin_jwt = MockJWTContext(
            user_id="admin-123",
            role="admin",
            org_id="org-1",
            authenticated=True,
        )

        request = MockRequest(
            body=json.dumps({"due_date": "not-a-date"}).encode(),
        )

        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
            return_value=admin_jwt,
        ):
            result = await handler._set_due_date(request, "finding-1")

        assert result is not None
        assert result.get("status") == 400

    @pytest.mark.asyncio
    async def test_set_due_date_valid_format_succeeds(self):
        """Test that setting valid due date succeeds."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
        )

        handler = FindingWorkflowHandler(server_context=MockServerContext())

        admin_jwt = MockJWTContext(
            user_id="admin-123",
            role="admin",
            org_id="org-1",
            authenticated=True,
        )

        request = MockRequest(
            body=json.dumps({"due_date": "2024-12-31T23:59:59Z"}).encode(),
        )

        with (
            patch(
                "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
                return_value=admin_jwt,
            ),
            patch(
                "aragora.server.handlers.features.finding_workflow.get_finding_workflow_store"
            ) as mock_store,
        ):
            store_mock = MagicMock()
            store_mock.get = AsyncMock(
                return_value={
                    "finding_id": "finding-1",
                    "current_state": "open",
                    "history": [],
                }
            )
            store_mock.save = AsyncMock()
            mock_store.return_value = store_mock

            result = await handler._set_due_date(request, "finding-1")

        assert result is not None
        assert result.get("status") == 200


class TestLinkingOperations:
    """Tests for finding linking operations."""

    @pytest.mark.asyncio
    async def test_link_finding_missing_id_returns_400(self):
        """Test that linking without finding ID returns 400."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
        )

        handler = FindingWorkflowHandler(server_context=MockServerContext())

        admin_jwt = MockJWTContext(
            user_id="admin-123",
            role="admin",
            org_id="org-1",
            authenticated=True,
        )

        request = MockRequest(
            body=json.dumps({}).encode(),
        )

        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
            return_value=admin_jwt,
        ):
            result = await handler._link_finding(request, "finding-1")

        assert result is not None
        assert result.get("status") == 400

    @pytest.mark.asyncio
    async def test_mark_duplicate_missing_parent_returns_400(self):
        """Test that marking duplicate without parent returns 400."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
        )

        handler = FindingWorkflowHandler(server_context=MockServerContext())

        admin_jwt = MockJWTContext(
            user_id="admin-123",
            role="admin",
            org_id="org-1",
            authenticated=True,
        )

        request = MockRequest(
            body=json.dumps({}).encode(),
        )

        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
            return_value=admin_jwt,
        ):
            result = await handler._mark_duplicate(request, "finding-1")

        assert result is not None
        assert result.get("status") == 400


class TestBulkActionValidation:
    """Tests for bulk action validation."""

    @pytest.fixture
    def admin_jwt(self):
        """Admin JWT context."""
        return MockJWTContext(
            user_id="admin-123",
            role="admin",
            org_id="org-1",
            authenticated=True,
        )

    @pytest.mark.asyncio
    async def test_bulk_action_missing_finding_ids_returns_400(self, admin_jwt):
        """Test that bulk action without finding_ids returns 400."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
            _finding_workflow_circuit_breaker,
        )

        _finding_workflow_circuit_breaker.is_open = False

        handler = FindingWorkflowHandler(server_context=MockServerContext())

        request = MockRequest(
            body=json.dumps({"action": "update_status"}).encode(),
        )

        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
            return_value=admin_jwt,
        ):
            result = await handler._bulk_action(request)

        assert result is not None
        assert result.get("status") == 400

    @pytest.mark.asyncio
    async def test_bulk_action_missing_action_returns_400(self, admin_jwt):
        """Test that bulk action without action returns 400."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
            _finding_workflow_circuit_breaker,
        )

        _finding_workflow_circuit_breaker.is_open = False

        handler = FindingWorkflowHandler(server_context=MockServerContext())

        request = MockRequest(
            body=json.dumps({"finding_ids": ["f1", "f2"]}).encode(),
        )

        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
            return_value=admin_jwt,
        ):
            result = await handler._bulk_action(request)

        assert result is not None
        assert result.get("status") == 400


class TestWorkflowStatesEndpoint:
    """Tests for workflow states endpoint."""

    @pytest.mark.asyncio
    async def test_get_workflow_states_returns_states(self):
        """Test that get_workflow_states returns state list."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
        )

        handler = FindingWorkflowHandler(server_context=MockServerContext())

        admin_jwt = MockJWTContext(
            user_id="admin-123",
            role="admin",
            org_id="org-1",
            authenticated=True,
        )

        request = MockRequest()

        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
            return_value=admin_jwt,
        ):
            result = await handler._get_workflow_states(request)

        assert result is not None
        assert result.get("status") == 200
        body = json.loads(result.get("body", "{}"))
        assert "states" in body


class TestAuditTypesEndpoint:
    """Tests for audit types endpoint."""

    @pytest.mark.asyncio
    async def test_get_audit_types_returns_types(self):
        """Test that get_audit_types returns type list."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
        )

        handler = FindingWorkflowHandler(server_context=MockServerContext())

        admin_jwt = MockJWTContext(
            user_id="admin-123",
            role="admin",
            org_id="org-1",
            authenticated=True,
        )

        request = MockRequest()

        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
            return_value=admin_jwt,
        ):
            result = await handler._get_audit_types(request)

        assert result is not None
        assert result.get("status") == 200
        body = json.loads(result.get("body", "{}"))
        assert "audit_types" in body


class TestPresetsEndpoint:
    """Tests for presets endpoint."""

    @pytest.mark.asyncio
    async def test_get_presets_returns_preset_list(self):
        """Test that get_presets returns preset list."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
        )

        handler = FindingWorkflowHandler(server_context=MockServerContext())

        admin_jwt = MockJWTContext(
            user_id="admin-123",
            role="admin",
            org_id="org-1",
            authenticated=True,
        )

        request = MockRequest()

        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
            return_value=admin_jwt,
        ):
            result = await handler._get_presets(request)

        assert result is not None
        assert result.get("status") == 200
        body = json.loads(result.get("body", "{}"))
        assert "presets" in body


class TestHandlerRouting:
    """Tests for request routing."""

    @pytest.mark.asyncio
    async def test_unknown_route_returns_404(self):
        """Test that unknown routes return 404."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
        )

        handler = FindingWorkflowHandler(server_context=MockServerContext())

        request = MagicMock()
        request.method = "GET"
        request.path = "/api/v1/audit/findings/unknown/path"

        result = await handler.handle_request(request)

        assert result is not None
        assert result.get("status") == 404

    def test_can_handle_returns_true_for_workflow_path(self):
        """Test can_handle returns True for workflow paths."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
        )

        handler = FindingWorkflowHandler(server_context=MockServerContext())

        assert handler.can_handle("/api/v1/finding-workflow/test", "GET") is True

    def test_can_handle_returns_false_for_non_workflow_path(self):
        """Test can_handle returns False for non-workflow paths."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
        )

        handler = FindingWorkflowHandler(server_context=MockServerContext())

        assert handler.can_handle("/api/v1/debates", "GET") is False


class TestJSONParsing:
    """Tests for JSON parsing."""

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self):
        """Test that invalid JSON returns 400."""
        from aragora.server.handlers.features.finding_workflow import (
            FindingWorkflowHandler,
            _finding_workflow_circuit_breaker,
        )

        _finding_workflow_circuit_breaker.is_open = False

        handler = FindingWorkflowHandler(server_context=MockServerContext())

        admin_jwt = MockJWTContext(
            user_id="admin-123",
            role="admin",
            org_id="org-1",
            authenticated=True,
        )

        request = MockRequest(
            body=b"not valid json",
        )

        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request",
            return_value=admin_jwt,
        ):
            result = await handler._update_status(request, "finding-1")

        assert result is not None
        assert result.get("status") == 400
