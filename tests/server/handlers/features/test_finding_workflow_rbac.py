"""
Tests for finding workflow RBAC enforcement.

Tests cover:
- Permission checks on all endpoints
- Role-based access control
- JWT vs header-based authentication
- Graceful error handling for permission denials
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime

from aragora.server.handlers.features.finding_workflow import FindingWorkflowHandler
from aragora.rbac import AuthorizationContext, AuthorizationDecision


class TestFindingWorkflowRBAC:
    """Tests for RBAC enforcement in FindingWorkflowHandler."""

    @pytest.fixture
    def handler(self, mock_server_context):
        """Create handler instance."""
        return FindingWorkflowHandler(server_context=mock_server_context)

    @pytest.fixture
    def mock_request(self):
        """Create mock request with configurable headers."""

        def _make_request(
            method="GET",
            path="/api/audit/findings/123/status",
            user_id="user-1",
            roles="member",
            body=None,
        ):
            request = MagicMock()
            request.method = method
            request.path = path
            request.headers = {
                "X-User-ID": user_id,
                "X-User-Roles": roles,
            }
            if body:

                async def read_json():
                    return body

                request.json = read_json
            return request

        return _make_request

    def test_get_auth_context_requires_jwt(self, handler, mock_request):
        """Auth context requires JWT - headers alone are rejected."""
        request = mock_request(user_id="test-user", roles="admin,editor")

        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request"
        ) as mock_extract:
            mock_extract.return_value = MagicMock(authenticated=False, user_id=None)
            context = handler._get_auth_context(request)

        # JWT required - headers alone should return None
        assert context is None

    def test_get_auth_context_anonymous_rejected(self, handler, mock_request):
        """Anonymous requests are rejected without JWT."""
        request = mock_request(user_id="anonymous")

        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request"
        ) as mock_extract:
            mock_extract.return_value = MagicMock(authenticated=False, user_id=None)
            context = handler._get_auth_context(request)

        # No JWT means no auth context
        assert context is None

    @pytest.mark.asyncio
    async def test_update_status_requires_update_permission(self, handler, mock_request):
        """Update status requires findings.update permission."""
        request = mock_request(
            method="PATCH",
            path="/api/audit/findings/123/status",
            body={"status": "in_progress"},
        )

        with patch.object(handler, "_check_permission") as mock_check:
            mock_check.return_value = {"error": "Permission denied", "status": 403}

            result = await handler._update_status(request, "123")

        mock_check.assert_called_once_with(request, "findings.update", "123")
        assert result["status"] == 403

    @pytest.mark.asyncio
    async def test_assign_requires_assign_permission(self, handler, mock_request):
        """Assign requires findings.assign permission."""
        request = mock_request(
            method="PATCH",
            path="/api/audit/findings/123/assign",
            body={"assignee_id": "user-2"},
        )

        with patch.object(handler, "_check_permission") as mock_check:
            mock_check.return_value = {"error": "Permission denied", "status": 403}

            result = await handler._assign(request, "123")

        mock_check.assert_called_once_with(request, "findings.assign", "123")
        assert result["status"] == 403

    @pytest.mark.asyncio
    async def test_bulk_action_requires_bulk_permission(self, handler, mock_request):
        """Bulk action requires findings.bulk permission."""
        request = mock_request(
            method="POST",
            path="/api/audit/findings/bulk-action",
            body={"action": "assign", "finding_ids": ["1", "2"]},
        )

        with patch.object(handler, "_check_permission") as mock_check:
            mock_check.return_value = {"error": "Permission denied", "status": 403}

            result = await handler._bulk_action(request)

        mock_check.assert_called_once_with(request, "findings.bulk")
        assert result["status"] == 403

    @pytest.mark.asyncio
    async def test_get_history_requires_read_permission(self, handler, mock_request):
        """Get history requires findings.read permission."""
        request = mock_request(
            method="GET",
            path="/api/audit/findings/123/history",
        )

        with patch.object(handler, "_check_permission") as mock_check:
            mock_check.return_value = {"error": "Permission denied", "status": 403}

            result = await handler._get_history(request, "123")

        mock_check.assert_called_once_with(request, "findings.read", "123")
        assert result["status"] == 403

    def test_check_permission_allows_authorized(self, handler, mock_request):
        """Check permission allows authorized users with valid JWT."""
        request = mock_request(user_id="admin", roles="admin")

        with patch(
            "aragora.server.handlers.features.finding_workflow.check_permission"
        ) as mock_check:
            mock_check.return_value = AuthorizationDecision(
                allowed=True, reason="Admin role", permission_key="findings.update"
            )
            with patch(
                "aragora.server.handlers.features.finding_workflow.extract_user_from_request"
            ) as mock_extract:
                # Valid JWT authentication
                mock_extract.return_value = MagicMock(
                    authenticated=True,
                    user_id="admin",
                    role="admin",
                    org_id="org-1",
                    client_ip="127.0.0.1",
                )
                result = handler._check_permission(request, "findings.update", "123")

        assert result is None  # No error means allowed

    @pytest.mark.no_auto_auth
    def test_check_permission_denies_unauthenticated(self, handler, mock_request):
        """Check permission returns 401 for unauthenticated users."""
        request = mock_request(user_id="viewer", roles="viewer")

        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request"
        ) as mock_extract:
            # No valid JWT
            mock_extract.return_value = MagicMock(authenticated=False, user_id=None)
            result = handler._check_permission(request, "findings.update", "123")

        assert result is not None
        assert result["status"] == 401  # Unauthenticated

    def test_check_permission_denies_unauthorized(self, handler, mock_request):
        """Check permission returns 403 for authenticated but unauthorized users."""
        request = mock_request(user_id="viewer", roles="viewer")

        with patch(
            "aragora.server.handlers.features.finding_workflow.check_permission"
        ) as mock_check:
            mock_check.return_value = AuthorizationDecision(
                allowed=False, reason="Insufficient permissions", permission_key="findings.update"
            )
            with patch(
                "aragora.server.handlers.features.finding_workflow.extract_user_from_request"
            ) as mock_extract:
                # Valid JWT but insufficient permissions
                mock_extract.return_value = MagicMock(
                    authenticated=True,
                    user_id="viewer",
                    role="viewer",
                    org_id="org-1",
                    client_ip="127.0.0.1",
                )
                result = handler._check_permission(request, "findings.update", "123")

        assert result is not None
        assert result["status"] == 403  # Forbidden (authenticated but not authorized)


class TestFindingWorkflowJWTAuth:
    """Tests for JWT authentication in FindingWorkflowHandler."""

    @pytest.fixture
    def handler(self, mock_server_context):
        """Create handler instance."""
        return FindingWorkflowHandler(server_context=mock_server_context)

    def test_jwt_auth_takes_precedence(self, handler):
        """JWT authentication takes precedence over headers."""
        request = MagicMock()
        request.headers = {
            "X-User-ID": "header-user",
            "X-User-Roles": "viewer",
        }

        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request"
        ) as mock_extract:
            mock_extract.return_value = MagicMock(
                authenticated=True,
                user_id="jwt-user",
                role="admin",
                org_id="org-1",
                email="admin@example.com",
                client_ip="127.0.0.1",
            )
            context = handler._get_auth_context(request)

        assert context.user_id == "jwt-user"
        assert "admin" in context.roles
        assert context.org_id == "org-1"

    def test_no_fallback_to_headers_when_no_jwt(self, handler):
        """Does NOT fall back to headers when JWT not present (security)."""
        request = MagicMock()
        request.headers = {
            "X-User-ID": "header-user",
            "X-User-Roles": "editor",
            "X-Org-ID": "org-2",
        }

        with patch(
            "aragora.server.handlers.features.finding_workflow.extract_user_from_request"
        ) as mock_extract:
            mock_extract.return_value = MagicMock(authenticated=False, user_id=None)
            context = handler._get_auth_context(request)

        # Headers alone are not trusted - JWT required
        assert context is None


class TestFindingWorkflowPermissionKeys:
    """Tests to verify correct permission keys are used."""

    @pytest.fixture
    def handler(self, mock_server_context):
        """Create handler instance."""
        return FindingWorkflowHandler(server_context=mock_server_context)

    @pytest.fixture
    def mock_request(self):
        """Create minimal mock request."""
        request = MagicMock()
        request.method = "GET"
        request.headers = {"X-User-ID": "test", "X-User-Roles": "admin"}
        return request

    @pytest.mark.parametrize(
        "method_name,permission_key,has_resource_id",
        [
            ("_update_status", "findings.update", True),
            ("_assign", "findings.assign", True),
            ("_unassign", "findings.assign", True),
            ("_add_comment", "findings.read", True),
            ("_get_comments", "findings.read", True),
            ("_get_history", "findings.read", True),
            ("_set_priority", "findings.update", True),
            ("_set_due_date", "findings.update", True),
            ("_link_finding", "findings.update", True),
            ("_mark_duplicate", "findings.update", True),
            ("_bulk_action", "findings.bulk", False),
            ("_get_my_assignments", "findings.read", False),
            ("_get_overdue", "findings.read", False),
            ("_get_workflow_states", "findings.read", False),
            ("_get_presets", "findings.read", False),
            ("_get_audit_types", "findings.read", False),
        ],
    )
    @pytest.mark.asyncio
    async def test_permission_key_used(
        self, handler, mock_request, method_name, permission_key, has_resource_id
    ):
        """Verify correct permission key is checked for each endpoint."""
        mock_request.method = "POST" if "bulk" in method_name else "PATCH"

        async def mock_json():
            return {"status": "in_progress"}

        mock_request.json = mock_json

        with patch.object(handler, "_check_permission") as mock_check:
            mock_check.return_value = {"error": "denied", "status": 403}

            method = getattr(handler, method_name)
            if has_resource_id:
                await method(mock_request, "finding-123")
                expected_call = (mock_request, permission_key, "finding-123")
            else:
                await method(mock_request)
                expected_call = (mock_request, permission_key)

            # Verify permission was checked with correct key
            assert mock_check.called
            actual_args = mock_check.call_args[0]
            assert actual_args[1] == permission_key
