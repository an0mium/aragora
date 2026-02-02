"""
Tests for aragora.server.middleware.tenant_isolation - Tenant Isolation Middleware.

Tests cover:
- TenantIsolationMiddleware class
  - Tenant ID extraction from headers
  - User ID extraction from request
  - Membership verification (fail closed)
  - Access attempt audit logging
- InMemoryMembershipStore
  - Member addition and removal
  - Owner membership (always member)
  - get_user_tenants functionality
- Decorator functionality
  - require_tenant_isolation decorator
  - require_tenant_isolation_with_config factory
- Utility functions
  - verify_tenant_access
  - get_user_accessible_tenants
- Security properties
  - Fail closed on membership check errors
  - Cross-tenant access prevention
  - No graceful fallbacks
  - Audit logging of violations
- Global middleware instance management
- Concurrent access handling
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.middleware.tenant_isolation import (
    InMemoryMembershipStore,
    TenantAccessAttempt,
    TenantAccessDeniedError,
    TenantIdMissingError,
    TenantIsolationError,
    TenantIsolationMiddleware,
    TenantMembershipCheckError,
    get_tenant_isolation_middleware,
    get_user_accessible_tenants,
    require_tenant_isolation,
    require_tenant_isolation_with_config,
    reset_tenant_isolation_middleware,
    set_tenant_isolation_middleware,
    verify_tenant_access,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


class MockRequest:
    """Mock HTTP request for testing."""

    def __init__(
        self,
        headers: dict[str, str] | None = None,
        user_id: str | None = None,
        path: str = "/api/v1/test",
        client_address: tuple[str, int] | None = ("192.168.1.1", 12345),
        auth_context: Any = None,
    ):
        self.headers = headers or {}
        self.user_id = user_id
        self.path = path
        self.client_address = client_address
        self.auth_context = auth_context


@dataclass
class MockAuthContext:
    """Mock auth context."""

    user_id: str = "anonymous"


@pytest.fixture
def mock_request():
    """Create a mock request with no tenant ID."""
    return MockRequest(headers={})


@pytest.fixture
def mock_request_with_tenant():
    """Create a mock request with tenant ID."""
    return MockRequest(
        headers={"X-Tenant-ID": "tenant-123"},
        user_id="user-456",
    )


@pytest.fixture
def mock_request_with_workspace():
    """Create a mock request with workspace ID header."""
    return MockRequest(
        headers={"X-Workspace-ID": "workspace-789"},
        user_id="user-456",
    )


@pytest.fixture
def membership_store():
    """Create a membership store with test data."""
    store = InMemoryMembershipStore()
    store.set_owner("tenant-123", "user-owner")
    store.add_member("tenant-123", "user-456")
    store.add_member("tenant-123", "user-789")
    store.set_owner("tenant-other", "user-other-owner")
    return store


@pytest.fixture
def middleware(membership_store):
    """Create a middleware instance with test store."""
    return TenantIsolationMiddleware(
        membership_store=membership_store,
        audit_enabled=True,
    )


@pytest.fixture(autouse=True)
def reset_global_middleware():
    """Reset global middleware before each test."""
    reset_tenant_isolation_middleware()
    yield
    reset_tenant_isolation_middleware()


def get_status(result) -> int:
    """Extract status code from result."""
    if hasattr(result, "status_code"):
        return result.status_code
    if isinstance(result, tuple):
        return result[1]
    return 0


# ===========================================================================
# Test InMemoryMembershipStore
# ===========================================================================


class TestInMemoryMembershipStore:
    """Tests for InMemoryMembershipStore."""

    @pytest.mark.asyncio
    async def test_add_member(self):
        """Should add member to tenant."""
        store = InMemoryMembershipStore()
        store.add_member("tenant-1", "user-1")

        is_member = await store.is_member("user-1", "tenant-1")
        assert is_member is True

    @pytest.mark.asyncio
    async def test_non_member_returns_false(self):
        """Should return False for non-members."""
        store = InMemoryMembershipStore()
        store.add_member("tenant-1", "user-1")

        is_member = await store.is_member("user-2", "tenant-1")
        assert is_member is False

    @pytest.mark.asyncio
    async def test_set_owner(self):
        """Owner should always be a member."""
        store = InMemoryMembershipStore()
        store.set_owner("tenant-1", "owner-1")

        is_member = await store.is_member("owner-1", "tenant-1")
        assert is_member is True

    @pytest.mark.asyncio
    async def test_owner_takes_precedence(self):
        """Owner membership should work even after removal."""
        store = InMemoryMembershipStore()
        store.set_owner("tenant-1", "owner-1")
        store.remove_member("tenant-1", "owner-1")

        # Owner is still a member
        is_member = await store.is_member("owner-1", "tenant-1")
        assert is_member is True

    @pytest.mark.asyncio
    async def test_remove_member(self):
        """Should remove member from tenant."""
        store = InMemoryMembershipStore()
        store.add_member("tenant-1", "user-1")
        store.remove_member("tenant-1", "user-1")

        is_member = await store.is_member("user-1", "tenant-1")
        assert is_member is False

    @pytest.mark.asyncio
    async def test_remove_nonexistent_member(self):
        """Removing non-existent member should not error."""
        store = InMemoryMembershipStore()
        # Should not raise
        store.remove_member("tenant-1", "user-1")

    @pytest.mark.asyncio
    async def test_get_user_tenants(self):
        """Should return all tenants user belongs to."""
        store = InMemoryMembershipStore()
        store.add_member("tenant-1", "user-1")
        store.add_member("tenant-2", "user-1")
        store.add_member("tenant-3", "user-2")

        tenants = await store.get_user_tenants("user-1")
        assert set(tenants) == {"tenant-1", "tenant-2"}

    @pytest.mark.asyncio
    async def test_get_user_tenants_includes_owned(self):
        """Should include tenants where user is owner."""
        store = InMemoryMembershipStore()
        store.set_owner("tenant-owned", "user-1")
        store.add_member("tenant-member", "user-1")

        tenants = await store.get_user_tenants("user-1")
        assert "tenant-owned" in tenants
        assert "tenant-member" in tenants

    @pytest.mark.asyncio
    async def test_get_user_tenants_empty(self):
        """Should return empty list for user with no tenants."""
        store = InMemoryMembershipStore()

        tenants = await store.get_user_tenants("nonexistent-user")
        assert tenants == []

    @pytest.mark.asyncio
    async def test_membership_to_nonexistent_tenant(self):
        """Should return False for non-existent tenant."""
        store = InMemoryMembershipStore()

        is_member = await store.is_member("user-1", "nonexistent-tenant")
        assert is_member is False


# ===========================================================================
# Test TenantIsolationMiddleware - Extraction
# ===========================================================================


class TestMiddlewareExtraction:
    """Tests for middleware header/user extraction."""

    def test_extract_tenant_id_from_x_tenant_id(self, middleware):
        """Should extract tenant ID from X-Tenant-ID header."""
        request = MockRequest(headers={"X-Tenant-ID": "tenant-123"})

        tenant_id = middleware.extract_tenant_id(request)
        assert tenant_id == "tenant-123"

    def test_extract_tenant_id_from_x_workspace_id(self, middleware):
        """Should extract tenant ID from X-Workspace-ID header."""
        request = MockRequest(headers={"X-Workspace-ID": "workspace-456"})

        tenant_id = middleware.extract_tenant_id(request)
        assert tenant_id == "workspace-456"

    def test_extract_tenant_id_prefers_x_tenant_id(self, middleware):
        """X-Tenant-ID should take precedence over X-Workspace-ID."""
        request = MockRequest(
            headers={
                "X-Tenant-ID": "tenant-123",
                "X-Workspace-ID": "workspace-456",
            }
        )

        tenant_id = middleware.extract_tenant_id(request)
        assert tenant_id == "tenant-123"

    def test_extract_tenant_id_missing(self, middleware):
        """Should return None when no tenant header present."""
        request = MockRequest(headers={})

        tenant_id = middleware.extract_tenant_id(request)
        assert tenant_id is None

    def test_extract_user_id_from_attribute(self, middleware):
        """Should extract user ID from request attribute."""
        request = MockRequest(user_id="user-123")

        user_id = middleware.extract_user_id(request)
        assert user_id == "user-123"

    def test_extract_user_id_from_auth_context(self, middleware):
        """Should extract user ID from auth context."""
        request = MockRequest()
        # Only use auth_context, ensure user_id attribute is not set
        del request.user_id
        request.auth_context = MockAuthContext(user_id="context-user")

        user_id = middleware.extract_user_id(request)
        assert user_id == "context-user"

    def test_extract_user_id_ignores_anonymous(self, middleware):
        """Should return None for anonymous user in auth context."""
        request = MockRequest()
        request.auth_context = MockAuthContext(user_id="anonymous")

        user_id = middleware.extract_user_id(request)
        assert user_id is None

    def test_extract_user_id_none(self, middleware):
        """Should return None when no user ID found."""
        request = MockRequest()

        user_id = middleware.extract_user_id(request)
        assert user_id is None

    def test_extract_source_ip_from_client_address(self, middleware):
        """Should extract IP from client_address tuple."""
        request = MockRequest(client_address=("10.0.0.1", 8080))

        ip = middleware.extract_source_ip(request)
        assert ip == "10.0.0.1"

    def test_extract_source_ip_from_x_forwarded_for(self, middleware):
        """Should extract IP from X-Forwarded-For header."""
        request = MockRequest(headers={"X-Forwarded-For": "203.0.113.1, 192.168.1.1"})
        # Remove client_address attribute entirely so headers are checked
        del request.client_address

        ip = middleware.extract_source_ip(request)
        assert ip == "203.0.113.1"

    def test_extract_request_path(self, middleware):
        """Should extract path from request."""
        request = MockRequest(path="/api/v1/inbox/messages")

        path = middleware.extract_request_path(request)
        assert path == "/api/v1/inbox/messages"


# ===========================================================================
# Test TenantIsolationMiddleware - Access Check
# ===========================================================================


class TestMiddlewareAccessCheck:
    """Tests for middleware access checking."""

    @pytest.mark.asyncio
    async def test_check_access_success(self, middleware):
        """Should return tenant ID when access is granted."""
        request = MockRequest(
            headers={"X-Tenant-ID": "tenant-123"},
            user_id="user-456",
        )

        tenant_id = await middleware.check_access(request)
        assert tenant_id == "tenant-123"

    @pytest.mark.asyncio
    async def test_check_access_owner_success(self, middleware):
        """Owner should have access to their tenant."""
        request = MockRequest(
            headers={"X-Tenant-ID": "tenant-123"},
            user_id="user-owner",
        )

        tenant_id = await middleware.check_access(request)
        assert tenant_id == "tenant-123"

    @pytest.mark.asyncio
    async def test_check_access_missing_tenant_id(self, middleware):
        """Should raise TenantIdMissingError when no tenant header."""
        request = MockRequest(headers={}, user_id="user-456")

        with pytest.raises(TenantIdMissingError) as exc_info:
            await middleware.check_access(request)

        assert "Tenant ID required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_check_access_unauthenticated(self, middleware):
        """Should raise TenantAccessDeniedError when not authenticated."""
        request = MockRequest(headers={"X-Tenant-ID": "tenant-123"})

        with pytest.raises(TenantAccessDeniedError) as exc_info:
            await middleware.check_access(request)

        assert "Authentication required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_check_access_not_member(self, middleware):
        """Should raise TenantAccessDeniedError when user is not a member."""
        request = MockRequest(
            headers={"X-Tenant-ID": "tenant-123"},
            user_id="stranger-user",
        )

        with pytest.raises(TenantAccessDeniedError) as exc_info:
            await middleware.check_access(request)

        assert "does not have access" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_check_access_cross_tenant_denied(self, middleware):
        """Should deny access to tenant user doesn't belong to."""
        request = MockRequest(
            headers={"X-Tenant-ID": "tenant-other"},
            user_id="user-456",  # Member of tenant-123, not tenant-other
        )

        with pytest.raises(TenantAccessDeniedError):
            await middleware.check_access(request)

    @pytest.mark.asyncio
    async def test_check_access_explicit_user_id(self, middleware):
        """Should use explicit user_id parameter."""
        request = MockRequest(
            headers={"X-Tenant-ID": "tenant-123"},
            user_id="stranger-user",  # Not a member
        )

        # But we pass a member as explicit user_id
        tenant_id = await middleware.check_access(request, user_id="user-456")
        assert tenant_id == "tenant-123"


# ===========================================================================
# Test Fail Closed Behavior
# ===========================================================================


class TestFailClosedBehavior:
    """Tests for fail-closed security behavior."""

    @pytest.mark.asyncio
    async def test_membership_check_error_denies_access(self):
        """Membership check errors should result in access denial."""
        failing_store = AsyncMock()
        failing_store.is_member.side_effect = RuntimeError("Database error")

        middleware = TenantIsolationMiddleware(
            membership_store=failing_store,
            audit_enabled=True,
        )

        request = MockRequest(
            headers={"X-Tenant-ID": "tenant-123"},
            user_id="user-456",
        )

        with pytest.raises(TenantMembershipCheckError) as exc_info:
            await middleware.check_access(request)

        assert "Unable to verify tenant membership" in str(exc_info.value)
        assert "Access denied" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_connection_error_denies_access(self):
        """Connection errors should deny access (fail closed)."""
        failing_store = AsyncMock()
        failing_store.is_member.side_effect = ConnectionError("Lost connection")

        middleware = TenantIsolationMiddleware(membership_store=failing_store)

        request = MockRequest(
            headers={"X-Tenant-ID": "tenant-123"},
            user_id="user-456",
        )

        with pytest.raises(TenantMembershipCheckError):
            await middleware.check_access(request)

    @pytest.mark.asyncio
    async def test_timeout_error_denies_access(self):
        """Timeout errors should deny access (fail closed)."""
        failing_store = AsyncMock()
        failing_store.is_member.side_effect = TimeoutError("Check timed out")

        middleware = TenantIsolationMiddleware(membership_store=failing_store)

        request = MockRequest(
            headers={"X-Tenant-ID": "tenant-123"},
            user_id="user-456",
        )

        with pytest.raises(TenantMembershipCheckError):
            await middleware.check_access(request)

    @pytest.mark.asyncio
    async def test_no_graceful_fallback(self):
        """Should NOT allow access when check fails, even if it used to work."""
        store = InMemoryMembershipStore()
        store.add_member("tenant-123", "user-456")

        middleware = TenantIsolationMiddleware(membership_store=store)

        request = MockRequest(
            headers={"X-Tenant-ID": "tenant-123"},
            user_id="user-456",
        )

        # First check works
        tenant_id = await middleware.check_access(request)
        assert tenant_id == "tenant-123"

        # Now make the store fail
        with patch.object(store, "is_member", side_effect=RuntimeError("Failure")):
            # Should NOT fall back to previous success
            with pytest.raises(TenantMembershipCheckError):
                await middleware.check_access(request)


# ===========================================================================
# Test Audit Logging
# ===========================================================================


class TestAuditLogging:
    """Tests for audit logging functionality."""

    @pytest.mark.asyncio
    async def test_successful_access_logged(self, middleware):
        """Successful access should be logged."""
        request = MockRequest(
            headers={"X-Tenant-ID": "tenant-123"},
            user_id="user-456",
        )

        await middleware.check_access(request)

        log = middleware.get_audit_log(limit=1)
        assert len(log) == 1
        assert log[0].allowed is True
        assert log[0].tenant_id == "tenant-123"
        assert log[0].user_id == "user-456"

    @pytest.mark.asyncio
    async def test_denied_access_logged(self, middleware):
        """Denied access should be logged."""
        request = MockRequest(
            headers={"X-Tenant-ID": "tenant-123"},
            user_id="stranger-user",
        )

        with pytest.raises(TenantAccessDeniedError):
            await middleware.check_access(request)

        log = middleware.get_audit_log(allowed=False, limit=1)
        assert len(log) == 1
        assert log[0].allowed is False
        assert log[0].user_id == "stranger-user"

    @pytest.mark.asyncio
    async def test_membership_error_logged(self):
        """Membership check errors should be logged."""
        failing_store = AsyncMock()
        failing_store.is_member.side_effect = RuntimeError("DB error")

        middleware = TenantIsolationMiddleware(
            membership_store=failing_store,
            audit_enabled=True,
        )

        request = MockRequest(
            headers={"X-Tenant-ID": "tenant-123"},
            user_id="user-456",
        )

        with pytest.raises(TenantMembershipCheckError):
            await middleware.check_access(request)

        log = middleware.get_audit_log(allowed=False, limit=1)
        assert len(log) == 1
        assert "check failed" in log[0].reason.lower()

    @pytest.mark.asyncio
    async def test_audit_log_filtering(self, middleware):
        """Audit log should support filtering."""
        # Generate multiple access attempts
        for user_id in ["user-456", "user-789", "stranger"]:
            request = MockRequest(
                headers={"X-Tenant-ID": "tenant-123"},
                user_id=user_id,
            )
            try:
                await middleware.check_access(request)
            except TenantAccessDeniedError:
                pass

        # Filter by user
        log = middleware.get_audit_log(user_id="user-456")
        assert all(e.user_id == "user-456" for e in log)

        # Filter by allowed
        denied = middleware.get_audit_log(allowed=False)
        assert all(not e.allowed for e in denied)

    @pytest.mark.asyncio
    async def test_audit_disabled(self):
        """Should not log when audit is disabled."""
        store = InMemoryMembershipStore()
        store.add_member("tenant-123", "user-456")

        middleware = TenantIsolationMiddleware(
            membership_store=store,
            audit_enabled=False,
        )

        request = MockRequest(
            headers={"X-Tenant-ID": "tenant-123"},
            user_id="user-456",
        )

        await middleware.check_access(request)

        log = middleware.get_audit_log()
        assert len(log) == 0

    def test_clear_audit_log(self, middleware):
        """Should be able to clear audit log."""
        middleware._access_log.append(
            TenantAccessAttempt(
                timestamp=MagicMock(),
                user_id="test",
                tenant_id="test",
                source_ip=None,
                request_path=None,
                allowed=True,
                reason="test",
            )
        )

        assert len(middleware.get_audit_log()) > 0
        middleware.clear_audit_log()
        assert len(middleware.get_audit_log()) == 0


# ===========================================================================
# Test Decorator
# ===========================================================================


class TestRequireTenantIsolationDecorator:
    """Tests for require_tenant_isolation decorator."""

    @pytest.mark.asyncio
    async def test_decorator_injects_tenant_id(self, membership_store):
        """Decorator should inject verified tenant_id."""
        set_tenant_isolation_middleware(
            TenantIsolationMiddleware(membership_store=membership_store)
        )

        @require_tenant_isolation
        async def endpoint(request, tenant_id: str):
            return {"tenant_id": tenant_id}

        request = MockRequest(
            headers={"X-Tenant-ID": "tenant-123"},
            user_id="user-456",
        )

        result = await endpoint(request=request)
        assert result["tenant_id"] == "tenant-123"

    @pytest.mark.asyncio
    async def test_decorator_missing_tenant_returns_400(self, membership_store):
        """Should return 400 when tenant ID missing."""
        set_tenant_isolation_middleware(
            TenantIsolationMiddleware(membership_store=membership_store)
        )

        @require_tenant_isolation
        async def endpoint(request, tenant_id: str):
            return {"tenant_id": tenant_id}

        request = MockRequest(headers={}, user_id="user-456")

        result = await endpoint(request=request)
        assert get_status(result) == 400

    @pytest.mark.asyncio
    async def test_decorator_access_denied_returns_403(self, membership_store):
        """Should return 403 when access denied."""
        set_tenant_isolation_middleware(
            TenantIsolationMiddleware(membership_store=membership_store)
        )

        @require_tenant_isolation
        async def endpoint(request, tenant_id: str):
            return {"tenant_id": tenant_id}

        request = MockRequest(
            headers={"X-Tenant-ID": "tenant-123"},
            user_id="stranger-user",
        )

        result = await endpoint(request=request)
        assert get_status(result) == 403

    @pytest.mark.asyncio
    async def test_decorator_membership_error_returns_403(self):
        """Should return 403 when membership check fails."""
        failing_store = AsyncMock()
        failing_store.is_member.side_effect = RuntimeError("Error")

        set_tenant_isolation_middleware(TenantIsolationMiddleware(membership_store=failing_store))

        @require_tenant_isolation
        async def endpoint(request, tenant_id: str):
            return {"tenant_id": tenant_id}

        request = MockRequest(
            headers={"X-Tenant-ID": "tenant-123"},
            user_id="user-456",
        )

        result = await endpoint(request=request)
        assert get_status(result) == 403

    @pytest.mark.asyncio
    async def test_decorator_no_request_returns_500(self):
        """Should return 500 when no request found."""

        @require_tenant_isolation
        async def endpoint():
            return {"success": True}

        result = await endpoint()
        assert get_status(result) == 500


class TestRequireTenantIsolationWithConfig:
    """Tests for require_tenant_isolation_with_config decorator factory."""

    @pytest.mark.asyncio
    async def test_decorator_with_custom_store(self):
        """Should use custom membership store."""
        custom_store = InMemoryMembershipStore()
        custom_store.add_member("custom-tenant", "custom-user")

        @require_tenant_isolation_with_config(membership_store=custom_store)
        async def endpoint(request, tenant_id: str):
            return {"tenant_id": tenant_id}

        request = MockRequest(
            headers={"X-Tenant-ID": "custom-tenant"},
            user_id="custom-user",
        )

        result = await endpoint(request=request)
        assert result["tenant_id"] == "custom-tenant"

    @pytest.mark.asyncio
    async def test_decorator_with_audit_disabled(self):
        """Should respect audit_enabled setting."""
        store = InMemoryMembershipStore()
        store.add_member("tenant-123", "user-456")

        middleware_ref = []

        @require_tenant_isolation_with_config(
            membership_store=store,
            audit_enabled=False,
        )
        async def endpoint(request, tenant_id: str):
            return {"tenant_id": tenant_id}

        request = MockRequest(
            headers={"X-Tenant-ID": "tenant-123"},
            user_id="user-456",
        )

        result = await endpoint(request=request)
        assert result["tenant_id"] == "tenant-123"


# ===========================================================================
# Test Utility Functions
# ===========================================================================


class TestVerifyTenantAccess:
    """Tests for verify_tenant_access utility."""

    @pytest.mark.asyncio
    async def test_verify_access_success(self, membership_store):
        """Should return True when user has access."""
        set_tenant_isolation_middleware(
            TenantIsolationMiddleware(membership_store=membership_store)
        )

        result = await verify_tenant_access("user-456", "tenant-123")
        assert result is True

    @pytest.mark.asyncio
    async def test_verify_access_denied(self, membership_store):
        """Should return False when user doesn't have access."""
        set_tenant_isolation_middleware(
            TenantIsolationMiddleware(membership_store=membership_store)
        )

        result = await verify_tenant_access("stranger", "tenant-123")
        assert result is False

    @pytest.mark.asyncio
    async def test_verify_access_error_returns_false(self):
        """Should return False on error (fail closed)."""
        failing_store = AsyncMock()
        failing_store.is_member.side_effect = RuntimeError("Error")

        result = await verify_tenant_access(
            "user-456", "tenant-123", membership_store=failing_store
        )
        assert result is False


class TestGetUserAccessibleTenants:
    """Tests for get_user_accessible_tenants utility."""

    @pytest.mark.asyncio
    async def test_get_tenants_success(self, membership_store):
        """Should return list of accessible tenants."""
        set_tenant_isolation_middleware(
            TenantIsolationMiddleware(membership_store=membership_store)
        )

        tenants = await get_user_accessible_tenants("user-456")
        assert "tenant-123" in tenants

    @pytest.mark.asyncio
    async def test_get_tenants_empty(self, membership_store):
        """Should return empty list for user with no tenants."""
        set_tenant_isolation_middleware(
            TenantIsolationMiddleware(membership_store=membership_store)
        )

        tenants = await get_user_accessible_tenants("nonexistent-user")
        assert tenants == []

    @pytest.mark.asyncio
    async def test_get_tenants_error_returns_empty(self):
        """Should return empty list on error."""
        failing_store = AsyncMock()
        failing_store.get_user_tenants.side_effect = RuntimeError("Error")

        tenants = await get_user_accessible_tenants("user-456", membership_store=failing_store)
        assert tenants == []


# ===========================================================================
# Test Global Middleware Management
# ===========================================================================


class TestGlobalMiddleware:
    """Tests for global middleware instance management."""

    def test_get_creates_default(self):
        """Should create default middleware if none set."""
        middleware = get_tenant_isolation_middleware()
        assert middleware is not None
        assert isinstance(middleware, TenantIsolationMiddleware)

    def test_set_and_get(self):
        """Should return set middleware."""
        custom = TenantIsolationMiddleware(audit_enabled=False)
        set_tenant_isolation_middleware(custom)

        retrieved = get_tenant_isolation_middleware()
        assert retrieved is custom

    def test_reset(self):
        """Should reset to None."""
        set_tenant_isolation_middleware(TenantIsolationMiddleware())
        reset_tenant_isolation_middleware()

        # Should create new one
        middleware = get_tenant_isolation_middleware()
        assert middleware is not None

    def test_singleton_behavior(self):
        """get_tenant_isolation_middleware should return same instance."""
        m1 = get_tenant_isolation_middleware()
        m2 = get_tenant_isolation_middleware()
        assert m1 is m2


# ===========================================================================
# Test Cross-Tenant Access Prevention
# ===========================================================================


class TestCrossTenantPrevention:
    """Tests for cross-tenant access prevention."""

    @pytest.mark.asyncio
    async def test_user_cannot_access_other_tenant(self, middleware):
        """User should not be able to access another tenant."""
        request = MockRequest(
            headers={"X-Tenant-ID": "tenant-other"},
            user_id="user-456",  # Member of tenant-123 only
        )

        with pytest.raises(TenantAccessDeniedError):
            await middleware.check_access(request)

    @pytest.mark.asyncio
    async def test_cross_tenant_attempt_logged(self, middleware):
        """Cross-tenant attempts should be logged."""
        request = MockRequest(
            headers={"X-Tenant-ID": "tenant-other"},
            user_id="user-456",
        )

        with pytest.raises(TenantAccessDeniedError):
            await middleware.check_access(request)

        denied = middleware.get_audit_log(allowed=False)
        assert len(denied) >= 1
        assert denied[0].tenant_id == "tenant-other"
        assert denied[0].user_id == "user-456"

    @pytest.mark.asyncio
    async def test_tenant_id_cannot_be_spoofed(self, middleware):
        """Tenant ID from header should be validated against membership."""
        # Attacker tries to access victim's tenant by setting header
        request = MockRequest(
            headers={"X-Tenant-ID": "tenant-123"},  # Victim's tenant
            user_id="attacker-user",  # Attacker not a member
        )

        with pytest.raises(TenantAccessDeniedError):
            await middleware.check_access(request)


# ===========================================================================
# Test Concurrent Access
# ===========================================================================


class TestConcurrentAccess:
    """Tests for concurrent access handling."""

    @pytest.mark.asyncio
    async def test_concurrent_access_checks(self, middleware):
        """Multiple concurrent checks should work correctly."""
        requests = [
            MockRequest(
                headers={"X-Tenant-ID": "tenant-123"},
                user_id="user-456",
            ),
            MockRequest(
                headers={"X-Tenant-ID": "tenant-123"},
                user_id="user-789",
            ),
            MockRequest(
                headers={"X-Tenant-ID": "tenant-123"},
                user_id="stranger",
            ),
        ]

        async def check(req):
            try:
                return await middleware.check_access(req)
            except TenantAccessDeniedError:
                return None

        results = await asyncio.gather(*[check(r) for r in requests])

        # First two should succeed, last should fail
        assert results[0] == "tenant-123"
        assert results[1] == "tenant-123"
        assert results[2] is None

    @pytest.mark.asyncio
    async def test_concurrent_audit_logging(self, middleware):
        """Concurrent access should all be logged correctly."""
        requests = [
            MockRequest(
                headers={"X-Tenant-ID": "tenant-123"},
                user_id=f"user-{i}",
            )
            for i in range(10)
        ]

        # Add all users as members
        for i in range(10):
            middleware._store.add_member("tenant-123", f"user-{i}")

        async def check(req):
            return await middleware.check_access(req)

        await asyncio.gather(*[check(r) for r in requests])

        # All should be logged
        log = middleware.get_audit_log(limit=100)
        assert len(log) >= 10


# ===========================================================================
# Test Exception Classes
# ===========================================================================


class TestExceptions:
    """Tests for exception classes."""

    def test_tenant_isolation_error_attributes(self):
        """Should store tenant_id and user_id."""
        error = TenantIsolationError("Test error", tenant_id="t1", user_id="u1")
        assert error.tenant_id == "t1"
        assert error.user_id == "u1"

    def test_tenant_access_denied_error(self):
        """TenantAccessDeniedError should be a TenantIsolationError."""
        error = TenantAccessDeniedError("Access denied", tenant_id="t1")
        assert isinstance(error, TenantIsolationError)

    def test_tenant_membership_check_error(self):
        """TenantMembershipCheckError should be a TenantIsolationError."""
        error = TenantMembershipCheckError("Check failed")
        assert isinstance(error, TenantIsolationError)

    def test_tenant_id_missing_error(self):
        """TenantIdMissingError should be a TenantIsolationError."""
        error = TenantIdMissingError("Missing")
        assert isinstance(error, TenantIsolationError)


# ===========================================================================
# Test TenantAccessAttempt
# ===========================================================================


class TestTenantAccessAttempt:
    """Tests for TenantAccessAttempt dataclass."""

    def test_to_dict(self):
        """Should serialize to dictionary."""
        from datetime import datetime, timezone

        attempt = TenantAccessAttempt(
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            user_id="user-123",
            tenant_id="tenant-456",
            source_ip="192.168.1.1",
            request_path="/api/v1/test",
            allowed=True,
            reason="Membership verified",
            metadata={"extra": "data"},
        )

        d = attempt.to_dict()
        assert d["user_id"] == "user-123"
        assert d["tenant_id"] == "tenant-456"
        assert d["source_ip"] == "192.168.1.1"
        assert d["request_path"] == "/api/v1/test"
        assert d["allowed"] is True
        assert d["reason"] == "Membership verified"
        assert d["metadata"] == {"extra": "data"}
        assert "2024-01-01" in d["timestamp"]
