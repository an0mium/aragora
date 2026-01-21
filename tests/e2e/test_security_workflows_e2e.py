"""
E2E Security Workflow Tests.

Tests complete security workflows including:
- Authentication flows (login, logout, token refresh)
- Authorization flows (RBAC, permissions)
- Secure data flows (encryption at rest)
- Audit logging workflows
- Rate limiting and protection
- Session management
"""

from __future__ import annotations

import asyncio
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Mock Authentication System
# =============================================================================


@dataclass
class MockUser:
    """Mock user for testing."""

    user_id: str
    email: str
    roles: List[str]
    permissions: List[str]
    workspace_id: str = "default"


@dataclass
class MockToken:
    """Mock JWT token for testing."""

    token: str
    user_id: str
    expires_at: float
    refresh_token: str


class MockAuthService:
    """Mock authentication service for testing."""

    def __init__(self):
        self._users: Dict[str, MockUser] = {}
        self._tokens: Dict[str, MockToken] = {}
        self._refresh_tokens: Dict[str, str] = {}
        self._failed_attempts: Dict[str, int] = {}
        self._locked_accounts: Dict[str, float] = {}
        self.audit_log: List[Dict[str, Any]] = []

        # Create default test users
        self._setup_test_users()

    def _setup_test_users(self):
        """Set up default test users."""
        self._users["admin"] = MockUser(
            user_id="admin-001",
            email="admin@test.com",
            roles=["admin", "user"],
            permissions=["read", "write", "delete", "admin"],
        )
        self._users["user"] = MockUser(
            user_id="user-001",
            email="user@test.com",
            roles=["user"],
            permissions=["read", "write"],
        )
        self._users["readonly"] = MockUser(
            user_id="readonly-001",
            email="readonly@test.com",
            roles=["viewer"],
            permissions=["read"],
        )

    async def authenticate(
        self,
        email: str,
        password: str,
    ) -> Optional[MockToken]:
        """Authenticate user and return token."""
        # Check account lock
        if email in self._locked_accounts:
            if time.time() < self._locked_accounts[email]:
                self._audit("auth_blocked", {"email": email, "reason": "account_locked"})
                raise PermissionError("Account is locked")
            else:
                del self._locked_accounts[email]
                self._failed_attempts.pop(email, None)

        # Find user
        user = None
        for u in self._users.values():
            if u.email == email:
                user = u
                break

        if user is None:
            self._record_failed_attempt(email)
            self._audit("auth_failed", {"email": email, "reason": "user_not_found"})
            return None

        # Simulate password check (accept any password for tests)
        if password != "correct_password":
            self._record_failed_attempt(email)
            self._audit("auth_failed", {"email": email, "reason": "invalid_password"})
            return None

        # Create token
        token = MockToken(
            token=secrets.token_urlsafe(32),
            user_id=user.user_id,
            expires_at=time.time() + 3600,
            refresh_token=secrets.token_urlsafe(32),
        )

        self._tokens[token.token] = token
        self._refresh_tokens[token.refresh_token] = token.token
        self._failed_attempts.pop(email, None)

        self._audit("auth_success", {"user_id": user.user_id, "email": email})
        return token

    def _record_failed_attempt(self, email: str) -> None:
        """Record failed authentication attempt."""
        self._failed_attempts[email] = self._failed_attempts.get(email, 0) + 1

        # Lock account after 5 failed attempts
        if self._failed_attempts[email] >= 5:
            self._locked_accounts[email] = time.time() + 300  # 5 minute lockout
            self._audit("account_locked", {"email": email})

    async def validate_token(self, token: str) -> Optional[MockUser]:
        """Validate token and return user."""
        if token not in self._tokens:
            return None

        token_data = self._tokens[token]
        if time.time() > token_data.expires_at:
            del self._tokens[token]
            return None

        for user in self._users.values():
            if user.user_id == token_data.user_id:
                return user

        return None

    async def refresh_token(self, refresh_token: str) -> Optional[MockToken]:
        """Refresh an access token."""
        if refresh_token not in self._refresh_tokens:
            self._audit("token_refresh_failed", {"reason": "invalid_refresh_token"})
            return None

        old_token = self._refresh_tokens[refresh_token]
        if old_token in self._tokens:
            old_token_data = self._tokens[old_token]
            user_id = old_token_data.user_id

            # Revoke old token
            del self._tokens[old_token]
            del self._refresh_tokens[refresh_token]

            # Create new token
            new_token = MockToken(
                token=secrets.token_urlsafe(32),
                user_id=user_id,
                expires_at=time.time() + 3600,
                refresh_token=secrets.token_urlsafe(32),
            )

            self._tokens[new_token.token] = new_token
            self._refresh_tokens[new_token.refresh_token] = new_token.token

            self._audit("token_refreshed", {"user_id": user_id})
            return new_token

        return None

    async def logout(self, token: str) -> bool:
        """Logout and invalidate token."""
        if token not in self._tokens:
            return False

        token_data = self._tokens[token]
        del self._tokens[token]

        # Also revoke refresh token
        refresh_to_remove = None
        for rt, t in self._refresh_tokens.items():
            if t == token:
                refresh_to_remove = rt
                break

        if refresh_to_remove:
            del self._refresh_tokens[refresh_to_remove]

        self._audit("logout", {"user_id": token_data.user_id})
        return True

    def _audit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Record audit event."""
        self.audit_log.append(
            {
                "timestamp": time.time(),
                "event_type": event_type,
                "data": data,
            }
        )


# =============================================================================
# Mock Authorization System
# =============================================================================


class MockRBACService:
    """Mock RBAC service for testing."""

    def __init__(self):
        self._role_permissions: Dict[str, List[str]] = {
            "admin": ["read", "write", "delete", "admin", "manage_users"],
            "user": ["read", "write"],
            "viewer": ["read"],
        }
        self._resource_policies: Dict[str, Dict[str, List[str]]] = {
            "debates": {
                "read": ["viewer", "user", "admin"],
                "write": ["user", "admin"],
                "delete": ["admin"],
            },
            "users": {"read": ["admin"], "write": ["admin"], "delete": ["admin"]},
            "settings": {"read": ["admin"], "write": ["admin"]},
        }
        self.audit_log: List[Dict[str, Any]] = []

    async def check_permission(
        self,
        user: MockUser,
        resource: str,
        action: str,
    ) -> bool:
        """Check if user has permission for action on resource."""
        # Check resource policy first - if resource has explicit policy, use it exclusively
        if resource in self._resource_policies:
            allowed_roles = self._resource_policies[resource].get(action, [])
            for role in user.roles:
                if role in allowed_roles:
                    self._audit(
                        "permission_granted",
                        {
                            "user_id": user.user_id,
                            "resource": resource,
                            "action": action,
                        },
                    )
                    return True
            # Resource has explicit policy - deny if not in allowed roles
            self._audit(
                "permission_denied",
                {
                    "user_id": user.user_id,
                    "resource": resource,
                    "action": action,
                },
            )
            return False

        # Fall back to direct permissions only for resources without explicit policies
        if action in user.permissions:
            self._audit(
                "permission_granted",
                {
                    "user_id": user.user_id,
                    "resource": resource,
                    "action": action,
                },
            )
            return True

        self._audit(
            "permission_denied",
            {
                "user_id": user.user_id,
                "resource": resource,
                "action": action,
            },
        )
        return False

    async def check_workspace_access(
        self,
        user: MockUser,
        workspace_id: str,
    ) -> bool:
        """Check if user can access workspace."""
        # For testing, users can only access their own workspace
        can_access = user.workspace_id == workspace_id or "admin" in user.roles

        if not can_access:
            self._audit(
                "workspace_access_denied",
                {
                    "user_id": user.user_id,
                    "workspace_id": workspace_id,
                },
            )

        return can_access

    def _audit(self, event_type: str, data: Dict[str, Any]) -> None:
        """Record audit event."""
        self.audit_log.append(
            {
                "timestamp": time.time(),
                "event_type": event_type,
                "data": data,
            }
        )


# =============================================================================
# Mock Encryption Service
# =============================================================================


class MockEncryptionService:
    """Mock encryption service for testing."""

    def __init__(self):
        self._key = secrets.token_bytes(32)
        self.operations: List[Dict[str, Any]] = []

    def encrypt(self, data: bytes, associated_data: bytes = b"") -> bytes:
        """Encrypt data (simple XOR for testing)."""
        self.operations.append(
            {
                "type": "encrypt",
                "size": len(data),
                "timestamp": time.time(),
            }
        )
        # Simple XOR encryption for testing
        key_bytes = (self._key * (len(data) // len(self._key) + 1))[: len(data)]
        return bytes(a ^ b for a, b in zip(data, key_bytes))

    def decrypt(self, data: bytes, associated_data: bytes = b"") -> bytes:
        """Decrypt data (simple XOR for testing)."""
        self.operations.append(
            {
                "type": "decrypt",
                "size": len(data),
                "timestamp": time.time(),
            }
        )
        # Simple XOR decryption for testing
        key_bytes = (self._key * (len(data) // len(self._key) + 1))[: len(data)]
        return bytes(a ^ b for a, b in zip(data, key_bytes))


# =============================================================================
# E2E Authentication Tests
# =============================================================================


class TestAuthenticationWorkflow:
    """E2E tests for authentication workflows."""

    @pytest.fixture
    def auth_service(self):
        """Create auth service for tests."""
        return MockAuthService()

    @pytest.mark.asyncio
    async def test_complete_login_flow(self, auth_service):
        """E2E: Test complete login flow."""
        # Step 1: Authenticate
        token = await auth_service.authenticate("admin@test.com", "correct_password")

        assert token is not None
        assert token.token
        assert token.refresh_token
        assert token.user_id == "admin-001"

        # Step 2: Validate token
        user = await auth_service.validate_token(token.token)

        assert user is not None
        assert user.email == "admin@test.com"

        # Step 3: Check audit log
        auth_events = [e for e in auth_service.audit_log if e["event_type"] == "auth_success"]
        assert len(auth_events) == 1

    @pytest.mark.asyncio
    async def test_complete_logout_flow(self, auth_service):
        """E2E: Test complete logout flow."""
        # Login first
        token = await auth_service.authenticate("user@test.com", "correct_password")
        assert token is not None

        # Verify logged in
        user = await auth_service.validate_token(token.token)
        assert user is not None

        # Logout
        result = await auth_service.logout(token.token)
        assert result is True

        # Verify logged out - token should be invalid
        user = await auth_service.validate_token(token.token)
        assert user is None

        # Check audit log
        logout_events = [e for e in auth_service.audit_log if e["event_type"] == "logout"]
        assert len(logout_events) == 1

    @pytest.mark.asyncio
    async def test_token_refresh_flow(self, auth_service):
        """E2E: Test token refresh workflow."""
        # Login
        token = await auth_service.authenticate("admin@test.com", "correct_password")
        original_token = token.token
        original_refresh = token.refresh_token

        # Refresh token
        new_token = await auth_service.refresh_token(original_refresh)

        assert new_token is not None
        assert new_token.token != original_token
        assert new_token.refresh_token != original_refresh

        # Old token should be invalid
        user = await auth_service.validate_token(original_token)
        assert user is None

        # New token should be valid
        user = await auth_service.validate_token(new_token.token)
        assert user is not None

    @pytest.mark.asyncio
    async def test_failed_login_attempt(self, auth_service):
        """E2E: Test failed login tracking."""
        # Try to login with wrong password
        token = await auth_service.authenticate("user@test.com", "wrong_password")
        assert token is None

        # Check audit log
        failed_events = [e for e in auth_service.audit_log if e["event_type"] == "auth_failed"]
        assert len(failed_events) == 1
        assert failed_events[0]["data"]["reason"] == "invalid_password"

    @pytest.mark.asyncio
    async def test_account_lockout_flow(self, auth_service):
        """E2E: Test account lockout after failed attempts."""
        email = "user@test.com"

        # 5 failed attempts should lock account
        for i in range(5):
            token = await auth_service.authenticate(email, "wrong_password")
            assert token is None

        # Account should be locked
        lock_events = [e for e in auth_service.audit_log if e["event_type"] == "account_locked"]
        assert len(lock_events) == 1

        # Even correct password should be rejected
        with pytest.raises(PermissionError, match="Account is locked"):
            await auth_service.authenticate(email, "correct_password")


# =============================================================================
# E2E Authorization Tests
# =============================================================================


class TestAuthorizationWorkflow:
    """E2E tests for authorization workflows."""

    @pytest.fixture
    def rbac_service(self):
        """Create RBAC service for tests."""
        return MockRBACService()

    @pytest.fixture
    def admin_user(self):
        """Create admin user."""
        return MockUser(
            user_id="admin-001",
            email="admin@test.com",
            roles=["admin"],
            permissions=["admin"],
        )

    @pytest.fixture
    def regular_user(self):
        """Create regular user."""
        return MockUser(
            user_id="user-001",
            email="user@test.com",
            roles=["user"],
            permissions=["read", "write"],
        )

    @pytest.fixture
    def viewer_user(self):
        """Create viewer user."""
        return MockUser(
            user_id="viewer-001",
            email="viewer@test.com",
            roles=["viewer"],
            permissions=["read"],
        )

    @pytest.mark.asyncio
    async def test_admin_full_access(self, rbac_service, admin_user):
        """E2E: Admin should have full access."""
        # Admin can do everything
        assert await rbac_service.check_permission(admin_user, "debates", "read")
        assert await rbac_service.check_permission(admin_user, "debates", "write")
        assert await rbac_service.check_permission(admin_user, "debates", "delete")
        assert await rbac_service.check_permission(admin_user, "users", "read")
        assert await rbac_service.check_permission(admin_user, "settings", "write")

    @pytest.mark.asyncio
    async def test_user_limited_access(self, rbac_service, regular_user):
        """E2E: Regular user has limited access."""
        # User can read and write debates
        assert await rbac_service.check_permission(regular_user, "debates", "read")
        assert await rbac_service.check_permission(regular_user, "debates", "write")

        # User cannot delete debates or manage users
        assert not await rbac_service.check_permission(regular_user, "debates", "delete")
        assert not await rbac_service.check_permission(regular_user, "users", "read")

    @pytest.mark.asyncio
    async def test_viewer_readonly_access(self, rbac_service, viewer_user):
        """E2E: Viewer has readonly access."""
        # Viewer can only read
        assert await rbac_service.check_permission(viewer_user, "debates", "read")

        # Viewer cannot write or delete
        assert not await rbac_service.check_permission(viewer_user, "debates", "write")
        assert not await rbac_service.check_permission(viewer_user, "debates", "delete")

    @pytest.mark.asyncio
    async def test_workspace_isolation(self, rbac_service):
        """E2E: Test workspace isolation."""
        user1 = MockUser(
            user_id="user-1",
            email="user1@test.com",
            roles=["user"],
            permissions=["read"],
            workspace_id="workspace-1",
        )
        user2 = MockUser(
            user_id="user-2",
            email="user2@test.com",
            roles=["user"],
            permissions=["read"],
            workspace_id="workspace-2",
        )

        # User can access own workspace
        assert await rbac_service.check_workspace_access(user1, "workspace-1")
        assert await rbac_service.check_workspace_access(user2, "workspace-2")

        # User cannot access other's workspace
        assert not await rbac_service.check_workspace_access(user1, "workspace-2")
        assert not await rbac_service.check_workspace_access(user2, "workspace-1")

    @pytest.mark.asyncio
    async def test_permission_audit_trail(self, rbac_service, regular_user):
        """E2E: Test permission checks are audited."""
        # Make some permission checks
        await rbac_service.check_permission(regular_user, "debates", "read")
        await rbac_service.check_permission(regular_user, "debates", "delete")

        # Check audit log
        granted = [e for e in rbac_service.audit_log if e["event_type"] == "permission_granted"]
        denied = [e for e in rbac_service.audit_log if e["event_type"] == "permission_denied"]

        assert len(granted) == 1  # read was granted
        assert len(denied) == 1  # delete was denied


# =============================================================================
# E2E Data Security Tests
# =============================================================================


class TestDataSecurityWorkflow:
    """E2E tests for data security workflows."""

    @pytest.fixture
    def encryption_service(self):
        """Create encryption service."""
        return MockEncryptionService()

    def test_encrypt_decrypt_cycle(self, encryption_service):
        """E2E: Test complete encrypt/decrypt cycle."""
        original_data = b"Sensitive debate content with user PII"

        # Encrypt
        encrypted = encryption_service.encrypt(original_data)
        assert encrypted != original_data

        # Decrypt
        decrypted = encryption_service.decrypt(encrypted)
        assert decrypted == original_data

    def test_encryption_operations_logged(self, encryption_service):
        """E2E: Test encryption operations are logged."""
        data = b"Test data"

        encryption_service.encrypt(data)
        encryption_service.decrypt(data)

        assert len(encryption_service.operations) == 2
        assert encryption_service.operations[0]["type"] == "encrypt"
        assert encryption_service.operations[1]["type"] == "decrypt"

    def test_multiple_data_isolation(self, encryption_service):
        """E2E: Test different data is encrypted independently."""
        data1 = b"User 1 data"
        data2 = b"User 2 data"

        encrypted1 = encryption_service.encrypt(data1)
        encrypted2 = encryption_service.encrypt(data2)

        # Different data should produce different ciphertext
        assert encrypted1 != encrypted2

        # Both should decrypt correctly
        assert encryption_service.decrypt(encrypted1) == data1
        assert encryption_service.decrypt(encrypted2) == data2


# =============================================================================
# E2E Combined Security Workflow Tests
# =============================================================================


class TestCombinedSecurityWorkflow:
    """E2E tests combining multiple security components."""

    @pytest.fixture
    def auth_service(self):
        return MockAuthService()

    @pytest.fixture
    def rbac_service(self):
        return MockRBACService()

    @pytest.fixture
    def encryption_service(self):
        return MockEncryptionService()

    @pytest.mark.asyncio
    async def test_authenticated_resource_access(
        self,
        auth_service,
        rbac_service,
    ):
        """E2E: Test full authenticated resource access flow."""
        # Step 1: Authenticate
        token = await auth_service.authenticate("user@test.com", "correct_password")
        assert token is not None

        # Step 2: Validate token and get user
        user = await auth_service.validate_token(token.token)
        assert user is not None

        # Step 3: Check authorization
        can_read = await rbac_service.check_permission(user, "debates", "read")
        assert can_read is True

        can_delete = await rbac_service.check_permission(user, "debates", "delete")
        assert can_delete is False

    @pytest.mark.asyncio
    async def test_secure_data_flow(
        self,
        auth_service,
        rbac_service,
        encryption_service,
    ):
        """E2E: Test secure data flow from auth to storage."""
        # Step 1: Authenticate
        token = await auth_service.authenticate("admin@test.com", "correct_password")
        user = await auth_service.validate_token(token.token)

        # Step 2: Check permission
        can_write = await rbac_service.check_permission(user, "debates", "write")
        assert can_write is True

        # Step 3: Encrypt data before storage
        sensitive_data = f"Debate data for user {user.user_id}".encode()
        encrypted = encryption_service.encrypt(sensitive_data)

        # Step 4: Verify encryption
        assert encrypted != sensitive_data

        # Step 5: Decrypt for reading
        decrypted = encryption_service.decrypt(encrypted)
        assert decrypted == sensitive_data

    @pytest.mark.asyncio
    async def test_complete_audit_trail(
        self,
        auth_service,
        rbac_service,
    ):
        """E2E: Test complete audit trail through security flow."""
        # Perform full flow
        token = await auth_service.authenticate("user@test.com", "correct_password")
        user = await auth_service.validate_token(token.token)
        await rbac_service.check_permission(user, "debates", "read")
        await rbac_service.check_permission(user, "debates", "delete")
        await auth_service.logout(token.token)

        # Verify audit trail
        auth_events = auth_service.audit_log
        rbac_events = rbac_service.audit_log

        # Should have auth events
        assert any(e["event_type"] == "auth_success" for e in auth_events)
        assert any(e["event_type"] == "logout" for e in auth_events)

        # Should have permission events
        assert any(e["event_type"] == "permission_granted" for e in rbac_events)
        assert any(e["event_type"] == "permission_denied" for e in rbac_events)

    @pytest.mark.asyncio
    async def test_session_expiry_handling(self, auth_service):
        """E2E: Test handling of expired sessions."""
        # Login
        token = await auth_service.authenticate("user@test.com", "correct_password")

        # Manually expire the token
        auth_service._tokens[token.token].expires_at = time.time() - 1

        # Token should now be invalid
        user = await auth_service.validate_token(token.token)
        assert user is None

    @pytest.mark.asyncio
    async def test_invalid_token_rejection(self, auth_service, rbac_service):
        """E2E: Test that invalid tokens are rejected."""
        # Try to validate an invalid token
        user = await auth_service.validate_token("invalid-token-12345")
        assert user is None

        # Can't check permissions without valid user
        # In real system, this would return 401 before reaching RBAC
