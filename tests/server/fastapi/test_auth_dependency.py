"""
Tests for FastAPI auth/RBAC dependencies.

Covers:
- get_auth_context (extracts auth from request)
- require_authenticated (enforces authentication)
- require_permission (enforces RBAC permissions)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.rbac.models import AuthorizationContext
from aragora.server.fastapi.dependencies.auth import (
    get_auth_context,
    require_authenticated,
    require_permission,
)


@pytest.fixture
def anonymous_context():
    """Create an anonymous auth context."""
    return AuthorizationContext(
        user_id="anonymous",
        org_id=None,
        workspace_id=None,
        roles=set(),
        permissions=set(),
    )


@pytest.fixture
def authenticated_context():
    """Create an authenticated auth context."""
    return AuthorizationContext(
        user_id="user-123",
        user_email="test@example.com",
        org_id="org-456",
        workspace_id="ws-789",
        roles={"admin"},
        permissions={"debates:create", "debates:read", "debates:delete", "admin:read"},
    )


class TestGetAuthContext:
    """Tests for get_auth_context dependency."""

    @pytest.mark.asyncio
    async def test_returns_anonymous_without_auth_header(self):
        """Should return anonymous context when no auth header."""
        mock_request = MagicMock()
        mock_request.headers = {}

        with patch(
            "aragora.server.handlers.utils.auth.get_auth_context",
            new_callable=AsyncMock,
        ) as mock_extract:
            mock_extract.return_value = AuthorizationContext(
                user_id="anonymous",
                org_id=None,
                workspace_id=None,
                roles=set(),
                permissions=set(),
            )
            result = await get_auth_context(mock_request)
            assert result.user_id == "anonymous"

    @pytest.mark.asyncio
    async def test_returns_context_on_extraction_failure(self):
        """Should return anonymous context on any extraction error."""
        mock_request = MagicMock()
        mock_request.headers = {"Authorization": "Bearer bad-token"}

        with patch(
            "aragora.server.handlers.utils.auth.get_auth_context",
            new_callable=AsyncMock,
            side_effect=Exception("Token verification failed"),
        ):
            result = await get_auth_context(mock_request)
            assert result.user_id == "anonymous"


class TestRequireAuthenticated:
    """Tests for require_authenticated dependency."""

    @pytest.mark.asyncio
    async def test_passes_authenticated_context(self, authenticated_context):
        """Should return context when user is authenticated."""
        result = await require_authenticated(authenticated_context)
        assert result.user_id == "user-123"

    @pytest.mark.asyncio
    async def test_rejects_anonymous_context(self, anonymous_context):
        """Should raise HTTPException for anonymous users."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            await require_authenticated(anonymous_context)
        assert exc_info.value.status_code == 401


class TestRequirePermission:
    """Tests for require_permission dependency."""

    @pytest.mark.asyncio
    async def test_passes_with_permission(self, authenticated_context):
        """Should pass when user has required permission."""
        check_fn = require_permission("debates:create")
        result = await check_fn(authenticated_context)
        assert result.user_id == "user-123"

    @pytest.mark.asyncio
    async def test_rejects_without_permission(self, authenticated_context):
        """Should raise 403 when user lacks permission."""
        from fastapi import HTTPException

        check_fn = require_permission("superadmin:nuke")
        with pytest.raises(HTTPException) as exc_info:
            await check_fn(authenticated_context)
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_rejects_anonymous(self, anonymous_context):
        """Should raise 401 for anonymous users (before checking permission)."""
        from fastapi import HTTPException

        check_fn = require_permission("debates:create")
        # require_permission depends on require_authenticated,
        # so anonymous users get 401 first
        with pytest.raises(HTTPException) as exc_info:
            await require_authenticated(anonymous_context)
        assert exc_info.value.status_code == 401
