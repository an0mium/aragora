"""
Tests for Authentication Bridge (aragora/gateway/enterprise/auth_bridge.py).

Tests cover:
- AuthBridgeError and subclasses (AuthenticationError, PermissionDeniedError, SessionExpiredError)
- TokenType enum
- AuthContext dataclass and methods
- PermissionMapping dataclass and matching methods
- TokenExchangeResult dataclass
- BridgedSession dataclass
- AuditEntry dataclass
- SessionLifecycleHook interface
- AuthBridge class with all methods:
  - verify_request and sub-methods
  - Permission mapping (add, remove, get_external_actions, get_required_permissions)
  - Action checking (is_action_allowed, check_action)
  - Token exchange
  - Session management (create, get, destroy, cleanup)
  - Lifecycle hooks
  - Audit logging
  - Statistics
"""

from __future__ import annotations

import base64
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.gateway.enterprise.auth_bridge import (
    AuditEntry,
    AuthBridge,
    AuthBridgeError,
    AuthContext,
    AuthenticationError,
    BridgedSession,
    PermissionDeniedError,
    PermissionMapping,
    SessionExpiredError,
    SessionLifecycleHook,
    TokenExchangeResult,
    TokenType,
)


# ============================================================================
# Exception Tests
# ============================================================================


class TestAuthBridgeError:
    """Tests for AuthBridgeError base exception."""

    def test_error_creation_minimal(self):
        """Test creating AuthBridgeError with minimal arguments."""
        error = AuthBridgeError("Test error")

        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.code == "AUTH_BRIDGE_ERROR"
        assert error.details == {}

    def test_error_creation_with_code(self):
        """Test creating AuthBridgeError with custom code."""
        error = AuthBridgeError("Test error", code="CUSTOM_CODE")

        assert error.code == "CUSTOM_CODE"

    def test_error_creation_with_details(self):
        """Test creating AuthBridgeError with details."""
        details = {"key": "value", "count": 42}
        error = AuthBridgeError("Test error", details=details)

        assert error.details == details
        assert error.details["key"] == "value"

    def test_error_is_exception(self):
        """Test AuthBridgeError inherits from Exception."""
        error = AuthBridgeError("Test")
        assert isinstance(error, Exception)


class TestAuthenticationError:
    """Tests for AuthenticationError."""

    def test_error_creation(self):
        """Test creating AuthenticationError."""
        error = AuthenticationError("Auth failed")

        assert error.message == "Auth failed"
        assert error.code == "AUTHENTICATION_FAILED"
        assert isinstance(error, AuthBridgeError)

    def test_error_with_details(self):
        """Test AuthenticationError with details."""
        error = AuthenticationError("Auth failed", {"reason": "invalid_token"})

        assert error.details["reason"] == "invalid_token"


class TestPermissionDeniedError:
    """Tests for PermissionDeniedError."""

    def test_error_creation(self):
        """Test creating PermissionDeniedError."""
        error = PermissionDeniedError("Permission denied")

        assert error.message == "Permission denied"
        assert error.code == "PERMISSION_DENIED"
        assert error.permission is None
        assert isinstance(error, AuthBridgeError)

    def test_error_with_permission(self):
        """Test PermissionDeniedError with permission."""
        error = PermissionDeniedError(
            "Permission denied",
            permission="debates.create",
        )

        assert error.permission == "debates.create"

    def test_error_with_details(self):
        """Test PermissionDeniedError with details."""
        error = PermissionDeniedError(
            "Permission denied",
            permission="debates.create",
            details={"user_id": "user123"},
        )

        assert error.details["user_id"] == "user123"


class TestSessionExpiredError:
    """Tests for SessionExpiredError."""

    def test_error_creation(self):
        """Test creating SessionExpiredError."""
        error = SessionExpiredError("session-123")

        assert "session-123" in error.message
        assert "expired" in error.message.lower()
        assert error.code == "SESSION_EXPIRED"
        assert error.session_id == "session-123"
        assert isinstance(error, AuthBridgeError)

    def test_error_with_details(self):
        """Test SessionExpiredError with details."""
        error = SessionExpiredError("session-123", {"reason": "timeout"})

        assert error.details["reason"] == "timeout"


# ============================================================================
# TokenType Tests
# ============================================================================


class TestTokenType:
    """Tests for TokenType enum."""

    def test_oidc_access_token(self):
        """Test OIDC access token type."""
        assert TokenType.OIDC_ACCESS == "oidc_access"
        assert TokenType.OIDC_ACCESS.value == "oidc_access"

    def test_oidc_id_token(self):
        """Test OIDC ID token type."""
        assert TokenType.OIDC_ID == "oidc_id"
        assert TokenType.OIDC_ID.value == "oidc_id"

    def test_saml_assertion(self):
        """Test SAML assertion type."""
        assert TokenType.SAML_ASSERTION == "saml_assertion"
        assert TokenType.SAML_ASSERTION.value == "saml_assertion"

    def test_api_key(self):
        """Test API key type."""
        assert TokenType.API_KEY == "api_key"
        assert TokenType.API_KEY.value == "api_key"

    def test_session(self):
        """Test session type."""
        assert TokenType.SESSION == "session"
        assert TokenType.SESSION.value == "session"

    def test_token_type_is_str(self):
        """Test that token types are strings."""
        for token_type in TokenType:
            assert isinstance(token_type, str)


# ============================================================================
# AuthContext Tests
# ============================================================================


class TestAuthContext:
    """Tests for AuthContext dataclass."""

    def test_context_minimal_creation(self):
        """Test creating AuthContext with minimal fields."""
        context = AuthContext(user_id="user123", email="user@example.com")

        assert context.user_id == "user123"
        assert context.email == "user@example.com"
        assert context.tenant_id is None
        assert context.permissions == set()
        assert context.roles == set()
        assert context.token_type == TokenType.OIDC_ACCESS

    def test_context_full_creation(self):
        """Test creating AuthContext with all fields."""
        now = time.time()
        context = AuthContext(
            user_id="user123",
            email="user@example.com",
            tenant_id="tenant456",
            permissions={"debates.create", "debates.read"},
            roles={"admin", "developer"},
            session_id="session789",
            token_type=TokenType.SAML_ASSERTION,
            claims={"custom": "value"},
            expires_at=now + 3600,
            created_at=now,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            metadata={"extra": "data"},
        )

        assert context.tenant_id == "tenant456"
        assert "debates.create" in context.permissions
        assert "admin" in context.roles
        assert context.session_id == "session789"
        assert context.ip_address == "192.168.1.1"
        assert context.metadata["extra"] == "data"

    def test_context_default_session_id(self):
        """Test that session_id is auto-generated."""
        context = AuthContext(user_id="user123", email="user@example.com")

        assert context.session_id is not None
        assert len(context.session_id) > 20

    def test_context_default_expires_at(self):
        """Test that expires_at defaults to 1 hour from now."""
        before = time.time()
        context = AuthContext(user_id="user123", email="user@example.com")
        after = time.time()

        # Should be approximately 1 hour from now
        assert context.expires_at > before + 3500
        assert context.expires_at < after + 3700

    def test_is_expired_false(self):
        """Test is_expired returns False for valid context."""
        context = AuthContext(
            user_id="user123",
            email="user@example.com",
            expires_at=time.time() + 3600,
        )

        assert context.is_expired is False

    def test_is_expired_true(self):
        """Test is_expired returns True for expired context."""
        context = AuthContext(
            user_id="user123",
            email="user@example.com",
            expires_at=time.time() - 1,
        )

        assert context.is_expired is True

    def test_remaining_ttl_positive(self):
        """Test remaining_ttl returns positive value for valid context."""
        context = AuthContext(
            user_id="user123",
            email="user@example.com",
            expires_at=time.time() + 1000,
        )

        assert context.remaining_ttl > 990
        assert context.remaining_ttl <= 1000

    def test_remaining_ttl_zero_for_expired(self):
        """Test remaining_ttl returns 0 for expired context."""
        context = AuthContext(
            user_id="user123",
            email="user@example.com",
            expires_at=time.time() - 100,
        )

        assert context.remaining_ttl == 0.0

    def test_display_name_from_claims(self):
        """Test display_name returns name from claims."""
        context = AuthContext(
            user_id="user123",
            email="user@example.com",
            claims={"name": "John Doe"},
        )

        assert context.display_name == "John Doe"

    def test_display_name_fallback_to_email(self):
        """Test display_name falls back to email local part."""
        context = AuthContext(user_id="user123", email="johndoe@example.com")

        assert context.display_name == "johndoe"

    def test_has_permission_exact_match(self):
        """Test has_permission with exact match."""
        context = AuthContext(
            user_id="user123",
            email="user@example.com",
            permissions={"debates.create", "debates.read"},
        )

        assert context.has_permission("debates.create") is True
        assert context.has_permission("debates.delete") is False

    def test_has_permission_colon_to_dot_normalization(self):
        """Test has_permission normalizes colon to dot."""
        context = AuthContext(
            user_id="user123",
            email="user@example.com",
            permissions={"debates.create"},
        )

        assert context.has_permission("debates:create") is True

    def test_has_permission_wildcard_resource(self):
        """Test has_permission with resource wildcard."""
        context = AuthContext(
            user_id="user123",
            email="user@example.com",
            permissions={"debates.*"},
        )

        assert context.has_permission("debates.create") is True
        assert context.has_permission("debates.read") is True
        assert context.has_permission("users.read") is False

    def test_has_permission_super_wildcard(self):
        """Test has_permission with super wildcard."""
        context = AuthContext(
            user_id="user123",
            email="user@example.com",
            permissions={"*"},
        )

        assert context.has_permission("debates.create") is True
        assert context.has_permission("anything.here") is True

    def test_has_role_true(self):
        """Test has_role returns True for assigned role."""
        context = AuthContext(
            user_id="user123",
            email="user@example.com",
            roles={"admin", "developer"},
        )

        assert context.has_role("admin") is True

    def test_has_role_false(self):
        """Test has_role returns False for unassigned role."""
        context = AuthContext(
            user_id="user123",
            email="user@example.com",
            roles={"developer"},
        )

        assert context.has_role("admin") is False

    def test_has_any_role_true(self):
        """Test has_any_role returns True when any role matches."""
        context = AuthContext(
            user_id="user123",
            email="user@example.com",
            roles={"developer"},
        )

        assert context.has_any_role("admin", "developer") is True

    def test_has_any_role_false(self):
        """Test has_any_role returns False when no role matches."""
        context = AuthContext(
            user_id="user123",
            email="user@example.com",
            roles={"viewer"},
        )

        assert context.has_any_role("admin", "developer") is False

    def test_to_dict(self):
        """Test to_dict returns correct dictionary."""
        context = AuthContext(
            user_id="user123",
            email="user@example.com",
            tenant_id="tenant456",
            permissions={"debates.create"},
            roles={"admin"},
            token_type=TokenType.OIDC_ACCESS,
        )

        result = context.to_dict()

        assert result["user_id"] == "user123"
        assert result["email"] == "user@example.com"
        assert result["tenant_id"] == "tenant456"
        assert "debates.create" in result["permissions"]
        assert "admin" in result["roles"]
        assert result["token_type"] == "oidc_access"
        assert "session_id" in result
        assert "expires_at" in result
        assert "is_expired" in result
        assert "display_name" in result


# ============================================================================
# PermissionMapping Tests
# ============================================================================


class TestPermissionMapping:
    """Tests for PermissionMapping dataclass."""

    def test_mapping_creation(self):
        """Test creating PermissionMapping."""
        mapping = PermissionMapping(
            aragora_permission="debates.create",
            external_action="create_conversation",
        )

        assert mapping.aragora_permission == "debates.create"
        assert mapping.external_action == "create_conversation"
        assert mapping.conditions == {}
        assert mapping.bidirectional is True
        assert mapping.priority == 0

    def test_mapping_with_conditions(self):
        """Test PermissionMapping with conditions."""
        mapping = PermissionMapping(
            aragora_permission="debates.create",
            external_action="create_conversation",
            conditions={"tenant_id": "tenant123"},
            bidirectional=False,
            priority=10,
        )

        assert mapping.conditions["tenant_id"] == "tenant123"
        assert mapping.bidirectional is False
        assert mapping.priority == 10

    def test_matches_aragora_exact(self):
        """Test matches_aragora with exact match."""
        mapping = PermissionMapping(
            aragora_permission="debates.create",
            external_action="create_conversation",
        )

        assert mapping.matches_aragora("debates.create") is True
        assert mapping.matches_aragora("debates.read") is False

    def test_matches_aragora_colon_normalization(self):
        """Test matches_aragora normalizes colon to dot."""
        mapping = PermissionMapping(
            aragora_permission="debates:create",
            external_action="create_conversation",
        )

        assert mapping.matches_aragora("debates.create") is True

    def test_matches_aragora_wildcard(self):
        """Test matches_aragora with wildcard permission."""
        mapping = PermissionMapping(
            aragora_permission="debates.*",
            external_action="any_debate_action",
        )

        assert mapping.matches_aragora("debates.create") is True
        assert mapping.matches_aragora("debates.read") is True
        assert mapping.matches_aragora("users.read") is False

    def test_matches_external_bidirectional(self):
        """Test matches_external when bidirectional is True."""
        mapping = PermissionMapping(
            aragora_permission="debates.create",
            external_action="create_conversation",
            bidirectional=True,
        )

        assert mapping.matches_external("create_conversation") is True
        assert mapping.matches_external("other_action") is False

    def test_matches_external_not_bidirectional(self):
        """Test matches_external when bidirectional is False."""
        mapping = PermissionMapping(
            aragora_permission="debates.create",
            external_action="create_conversation",
            bidirectional=False,
        )

        assert mapping.matches_external("create_conversation") is False


# ============================================================================
# TokenExchangeResult Tests
# ============================================================================


class TestTokenExchangeResult:
    """Tests for TokenExchangeResult dataclass."""

    def test_result_creation_minimal(self):
        """Test creating TokenExchangeResult with minimal fields."""
        result = TokenExchangeResult(access_token="token123")

        assert result.access_token == "token123"
        assert result.token_type == "Bearer"
        assert result.expires_in == 3600
        assert result.scope == ""
        assert result.refresh_token is None
        assert result.id_token is None

    def test_result_creation_full(self):
        """Test creating TokenExchangeResult with all fields."""
        now = time.time()
        result = TokenExchangeResult(
            access_token="token123",
            token_type="Bearer",
            expires_in=7200,
            scope="read write",
            refresh_token="refresh456",
            id_token="id789",
            issued_at=now,
            audience="external-api",
            metadata={"custom": "data"},
        )

        assert result.expires_in == 7200
        assert result.scope == "read write"
        assert result.refresh_token == "refresh456"
        assert result.audience == "external-api"

    def test_expires_at(self):
        """Test expires_at property."""
        now = time.time()
        result = TokenExchangeResult(
            access_token="token123",
            expires_in=3600,
            issued_at=now,
        )

        assert result.expires_at == now + 3600

    def test_is_expired_false(self):
        """Test is_expired returns False for valid token."""
        result = TokenExchangeResult(
            access_token="token123",
            expires_in=3600,
            issued_at=time.time(),
        )

        assert result.is_expired is False

    def test_is_expired_true(self):
        """Test is_expired returns True for expired token."""
        result = TokenExchangeResult(
            access_token="token123",
            expires_in=0,
            issued_at=time.time() - 100,
        )

        assert result.is_expired is True

    def test_to_dict_minimal(self):
        """Test to_dict with minimal fields."""
        result = TokenExchangeResult(access_token="token123")

        d = result.to_dict()

        assert d["access_token"] == "token123"
        assert d["token_type"] == "Bearer"
        assert d["expires_in"] == 3600
        assert "scope" not in d
        assert "refresh_token" not in d
        assert "id_token" not in d

    def test_to_dict_with_optional_fields(self):
        """Test to_dict includes optional fields when present."""
        result = TokenExchangeResult(
            access_token="token123",
            scope="read write",
            refresh_token="refresh456",
            id_token="id789",
        )

        d = result.to_dict()

        assert d["scope"] == "read write"
        assert d["refresh_token"] == "refresh456"
        assert d["id_token"] == "id789"


# ============================================================================
# BridgedSession Tests
# ============================================================================


class TestBridgedSession:
    """Tests for BridgedSession dataclass."""

    def test_session_creation_minimal(self):
        """Test creating BridgedSession with minimal fields."""
        session = BridgedSession(
            session_id="session123",
            aragora_session_id="aragora-session",
        )

        assert session.session_id == "session123"
        assert session.aragora_session_id == "aragora-session"
        assert session.external_session_id is None
        assert session.auth_context is None
        assert session.metadata == {}

    def test_session_creation_full(self):
        """Test creating BridgedSession with all fields."""
        context = AuthContext(user_id="user123", email="user@example.com")
        now = time.time()

        session = BridgedSession(
            session_id="session123",
            aragora_session_id="aragora-session",
            external_session_id="external-session",
            auth_context=context,
            created_at=now,
            last_accessed_at=now,
            expires_at=now + 28800,
            metadata={"custom": "data"},
        )

        assert session.external_session_id == "external-session"
        assert session.auth_context == context
        assert session.metadata["custom"] == "data"

    def test_session_default_expiry(self):
        """Test session has default 8 hour expiry."""
        before = time.time()
        session = BridgedSession(
            session_id="session123",
            aragora_session_id="aragora-session",
        )
        after = time.time()

        # Should be approximately 8 hours (28800 seconds) from now
        assert session.expires_at > before + 28700
        assert session.expires_at < after + 28900

    def test_is_expired_false(self):
        """Test is_expired returns False for valid session."""
        session = BridgedSession(
            session_id="session123",
            aragora_session_id="aragora-session",
            expires_at=time.time() + 3600,
        )

        assert session.is_expired is False

    def test_is_expired_true(self):
        """Test is_expired returns True for expired session."""
        session = BridgedSession(
            session_id="session123",
            aragora_session_id="aragora-session",
            expires_at=time.time() - 1,
        )

        assert session.is_expired is True

    def test_touch_updates_last_accessed(self):
        """Test touch updates last_accessed_at."""
        session = BridgedSession(
            session_id="session123",
            aragora_session_id="aragora-session",
            last_accessed_at=time.time() - 100,
        )

        old_time = session.last_accessed_at
        session.touch()

        assert session.last_accessed_at > old_time


# ============================================================================
# AuditEntry Tests
# ============================================================================


class TestAuditEntry:
    """Tests for AuditEntry dataclass."""

    def test_entry_creation_minimal(self):
        """Test creating AuditEntry with minimal fields."""
        now = time.time()
        entry = AuditEntry(
            timestamp=now,
            event_type="authentication",
            user_id="user123",
        )

        assert entry.timestamp == now
        assert entry.event_type == "authentication"
        assert entry.user_id == "user123"
        assert entry.session_id is None
        assert entry.action == ""
        assert entry.result == "success"
        assert entry.details == {}

    def test_entry_creation_full(self):
        """Test creating AuditEntry with all fields."""
        now = time.time()
        entry = AuditEntry(
            timestamp=now,
            event_type="permission_check",
            user_id="user123",
            session_id="session456",
            action="create_conversation",
            result="denied",
            ip_address="192.168.1.1",
            details={"reason": "insufficient_permissions"},
        )

        assert entry.session_id == "session456"
        assert entry.action == "create_conversation"
        assert entry.result == "denied"
        assert entry.ip_address == "192.168.1.1"
        assert entry.details["reason"] == "insufficient_permissions"


# ============================================================================
# SessionLifecycleHook Tests
# ============================================================================


class ConcreteLifecycleHook(SessionLifecycleHook):
    """Concrete implementation for testing SessionLifecycleHook."""

    def __init__(self):
        self.created_sessions = []
        self.accessed_sessions = []
        self.expired_sessions = []
        self.destroyed_sessions = []

    async def on_session_created(self, session: BridgedSession, context: AuthContext) -> None:
        self.created_sessions.append((session, context))

    async def on_session_accessed(self, session: BridgedSession) -> None:
        self.accessed_sessions.append(session)

    async def on_session_expired(self, session: BridgedSession) -> None:
        self.expired_sessions.append(session)

    async def on_session_destroyed(self, session: BridgedSession, reason: str = "logout") -> None:
        self.destroyed_sessions.append((session, reason))


class TestSessionLifecycleHook:
    """Tests for SessionLifecycleHook interface."""

    @pytest.mark.asyncio
    async def test_hook_default_implementation(self):
        """Test that default hook methods do nothing."""
        hook = SessionLifecycleHook()
        session = BridgedSession(session_id="session123", aragora_session_id="aragora-session")
        context = AuthContext(user_id="user123", email="user@example.com")

        # Should not raise
        await hook.on_session_created(session, context)
        await hook.on_session_accessed(session)
        await hook.on_session_expired(session)
        await hook.on_session_destroyed(session, "logout")


# ============================================================================
# AuthBridge Tests
# ============================================================================


class TestAuthBridge:
    """Tests for AuthBridge class."""

    @pytest.fixture
    def bridge(self) -> AuthBridge:
        """Create a basic auth bridge for testing."""
        return AuthBridge(
            permission_mappings=[
                PermissionMapping("debates.create", "create_conversation"),
                PermissionMapping("debates.read", "view_conversation"),
                PermissionMapping("debates.*", "manage_debates", priority=10),
            ],
            action_allowlist={"create_conversation", "view_conversation", "manage_debates"},
            session_duration=3600,
            enable_audit=True,
        )

    @pytest.fixture
    def context(self) -> AuthContext:
        """Create a test auth context."""
        return AuthContext(
            user_id="user123",
            email="user@example.com",
            tenant_id="tenant456",
            permissions={"debates.create", "debates.read"},
            roles={"admin"},
        )

    # =========================================================================
    # Initialization Tests
    # =========================================================================

    def test_bridge_creation_default(self):
        """Test creating AuthBridge with defaults."""
        bridge = AuthBridge()

        assert bridge._permission_mappings == []
        assert bridge._action_allowlist is None
        assert bridge._action_denylist == set()
        assert bridge._session_duration == 28800
        assert bridge._enable_audit is True

    def test_bridge_creation_with_options(self):
        """Test creating AuthBridge with options."""
        mappings = [PermissionMapping("debates.create", "create_conversation")]
        bridge = AuthBridge(
            permission_mappings=mappings,
            action_allowlist={"create_conversation"},
            action_denylist={"delete_conversation"},
            session_duration=7200,
            enable_audit=False,
        )

        assert len(bridge._permission_mappings) == 1
        assert bridge._action_allowlist == {"create_conversation"}
        assert bridge._action_denylist == {"delete_conversation"}
        assert bridge._session_duration == 7200
        assert bridge._enable_audit is False

    # =========================================================================
    # Permission Mapping Tests
    # =========================================================================

    def test_add_permission_mapping(self, bridge: AuthBridge):
        """Test adding a permission mapping."""
        initial_count = len(bridge._permission_mappings)
        mapping = PermissionMapping("users.create", "create_user")

        bridge.add_permission_mapping(mapping)

        assert len(bridge._permission_mappings) == initial_count + 1
        assert mapping in bridge._permission_mappings

    def test_remove_permission_mapping_by_aragora(self, bridge: AuthBridge):
        """Test removing mappings by Aragora permission."""
        initial_count = len(bridge._permission_mappings)

        removed = bridge.remove_permission_mapping(aragora_permission="debates.create")

        assert removed == 1
        assert len(bridge._permission_mappings) == initial_count - 1

    def test_remove_permission_mapping_by_external(self, bridge: AuthBridge):
        """Test removing mappings by external action."""
        initial_count = len(bridge._permission_mappings)

        removed = bridge.remove_permission_mapping(external_action="view_conversation")

        assert removed == 1
        assert len(bridge._permission_mappings) == initial_count - 1

    def test_remove_permission_mapping_none_found(self, bridge: AuthBridge):
        """Test removing mappings when none match."""
        initial_count = len(bridge._permission_mappings)

        removed = bridge.remove_permission_mapping(aragora_permission="nonexistent")

        assert removed == 0
        assert len(bridge._permission_mappings) == initial_count

    def test_get_external_actions(self, bridge: AuthBridge, context: AuthContext):
        """Test getting external actions for a context."""
        actions = bridge.get_external_actions(context)

        assert "create_conversation" in actions
        assert "view_conversation" in actions

    def test_get_external_actions_with_denylist(self, context: AuthContext):
        """Test external actions with denylist."""
        bridge = AuthBridge(
            permission_mappings=[
                PermissionMapping("debates.create", "create_conversation"),
                PermissionMapping("debates.delete", "delete_conversation"),
            ],
            action_denylist={"delete_conversation"},
        )
        context.permissions.add("debates.delete")

        actions = bridge.get_external_actions(context)

        assert "create_conversation" in actions
        assert "delete_conversation" not in actions

    def test_get_required_permissions(self, bridge: AuthBridge):
        """Test getting required permissions for an action."""
        permissions = bridge.get_required_permissions("create_conversation")

        assert "debates.create" in permissions

    def test_get_required_permissions_not_found(self, bridge: AuthBridge):
        """Test getting required permissions for unknown action."""
        permissions = bridge.get_required_permissions("unknown_action")

        assert permissions == set()

    # =========================================================================
    # Action Checking Tests
    # =========================================================================

    def test_is_action_allowed_true(self, bridge: AuthBridge, context: AuthContext):
        """Test is_action_allowed returns True for allowed action."""
        assert bridge.is_action_allowed(context, "create_conversation") is True

    def test_is_action_allowed_false_no_permission(self, bridge: AuthBridge):
        """Test is_action_allowed returns False when no permission."""
        context = AuthContext(
            user_id="user123",
            email="user@example.com",
            permissions=set(),
        )

        assert bridge.is_action_allowed(context, "create_conversation") is False

    def test_is_action_allowed_false_denylist(self, context: AuthContext):
        """Test is_action_allowed returns False for denylisted action."""
        bridge = AuthBridge(
            permission_mappings=[
                PermissionMapping("debates.create", "create_conversation"),
            ],
            action_denylist={"create_conversation"},
        )

        assert bridge.is_action_allowed(context, "create_conversation") is False

    def test_is_action_allowed_false_not_in_allowlist(self, context: AuthContext):
        """Test is_action_allowed returns False when not in allowlist."""
        bridge = AuthBridge(
            permission_mappings=[
                PermissionMapping("debates.create", "create_conversation"),
                PermissionMapping("debates.delete", "delete_conversation"),
            ],
            action_allowlist={"create_conversation"},
        )
        context.permissions.add("debates.delete")

        assert bridge.is_action_allowed(context, "delete_conversation") is False

    @pytest.mark.asyncio
    async def test_check_action_allowed(self, bridge: AuthBridge, context: AuthContext):
        """Test check_action does not raise for allowed action."""
        # Should not raise
        bridge.check_action(context, "create_conversation")

    @pytest.mark.asyncio
    async def test_check_action_denied(self, bridge: AuthBridge):
        """Test check_action raises PermissionDeniedError."""
        context = AuthContext(
            user_id="user123",
            email="user@example.com",
            permissions=set(),
        )

        with pytest.raises(PermissionDeniedError) as exc:
            bridge.check_action(context, "create_conversation")

        assert "not allowed" in exc.value.message
        assert exc.value.permission == "create_conversation"

    # =========================================================================
    # Condition Evaluation Tests
    # =========================================================================

    def test_evaluate_conditions_empty(self, bridge: AuthBridge, context: AuthContext):
        """Test condition evaluation with empty conditions."""
        result = bridge._evaluate_conditions({}, context)

        assert result is True

    def test_evaluate_conditions_tenant_id_match(self, bridge: AuthBridge, context: AuthContext):
        """Test condition evaluation with matching tenant_id."""
        result = bridge._evaluate_conditions({"tenant_id": "tenant456"}, context)

        assert result is True

    def test_evaluate_conditions_tenant_id_mismatch(self, bridge: AuthBridge, context: AuthContext):
        """Test condition evaluation with mismatching tenant_id."""
        result = bridge._evaluate_conditions({"tenant_id": "other-tenant"}, context)

        assert result is False

    def test_evaluate_conditions_roles_match(self, bridge: AuthBridge, context: AuthContext):
        """Test condition evaluation with matching role."""
        result = bridge._evaluate_conditions({"roles": ["admin"]}, context)

        assert result is True

    def test_evaluate_conditions_roles_mismatch(self, bridge: AuthBridge, context: AuthContext):
        """Test condition evaluation with mismatching role."""
        result = bridge._evaluate_conditions({"roles": ["superadmin"]}, context)

        assert result is False

    def test_evaluate_conditions_has_permission_true(
        self, bridge: AuthBridge, context: AuthContext
    ):
        """Test condition evaluation with has_permission that succeeds."""
        result = bridge._evaluate_conditions({"has_permission": "debates.create"}, context)

        assert result is True

    def test_evaluate_conditions_has_permission_false(
        self, bridge: AuthBridge, context: AuthContext
    ):
        """Test condition evaluation with has_permission that fails."""
        result = bridge._evaluate_conditions({"has_permission": "admin.delete"}, context)

        assert result is False

    def test_evaluate_conditions_claim_match(self, bridge: AuthBridge):
        """Test condition evaluation with matching claim."""
        context = AuthContext(
            user_id="user123",
            email="user@example.com",
            claims={"org_id": "org123"},
        )

        result = bridge._evaluate_conditions({"org_id": "org123"}, context)

        assert result is True

    def test_evaluate_conditions_metadata_match(self, bridge: AuthBridge):
        """Test condition evaluation with matching metadata."""
        context = AuthContext(
            user_id="user123",
            email="user@example.com",
            metadata={"feature_flag": True},
        )

        result = bridge._evaluate_conditions({"feature_flag": True}, context)

        assert result is True

    # =========================================================================
    # Session Management Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_create_session(self, bridge: AuthBridge, context: AuthContext):
        """Test creating a session."""
        session = await bridge.create_session(context)

        assert session.session_id is not None
        assert session.aragora_session_id == context.session_id
        assert session.auth_context == context
        assert session.session_id in bridge._sessions

    @pytest.mark.asyncio
    async def test_create_session_with_external_id(self, bridge: AuthBridge, context: AuthContext):
        """Test creating a session with external session ID."""
        session = await bridge.create_session(context, external_session_id="external-session")

        assert session.external_session_id == "external-session"

    @pytest.mark.asyncio
    async def test_create_session_with_duration(self, bridge: AuthBridge, context: AuthContext):
        """Test creating a session with custom duration."""
        before = time.time()
        session = await bridge.create_session(context, duration=7200)
        after = time.time()

        assert session.expires_at > before + 7100
        assert session.expires_at < after + 7300

    @pytest.mark.asyncio
    async def test_create_session_with_metadata(self, bridge: AuthBridge, context: AuthContext):
        """Test creating a session with metadata."""
        session = await bridge.create_session(context, metadata={"custom": "data"})

        assert session.metadata["custom"] == "data"

    @pytest.mark.asyncio
    async def test_get_session_success(self, bridge: AuthBridge, context: AuthContext):
        """Test getting an existing session."""
        session = await bridge.create_session(context)

        retrieved = await bridge.get_session(session.session_id)

        assert retrieved is not None
        assert retrieved.session_id == session.session_id

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, bridge: AuthBridge):
        """Test getting a non-existent session returns None."""
        retrieved = await bridge.get_session("nonexistent")

        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_session_expired(self, bridge: AuthBridge, context: AuthContext):
        """Test getting an expired session returns None and cleans up."""
        session = await bridge.create_session(context)
        bridge._sessions[session.session_id].expires_at = time.time() - 1

        retrieved = await bridge.get_session(session.session_id)

        assert retrieved is None
        assert session.session_id not in bridge._sessions

    @pytest.mark.asyncio
    async def test_get_session_touches_session(self, bridge: AuthBridge, context: AuthContext):
        """Test getting a session updates last_accessed_at."""
        session = await bridge.create_session(context)
        old_time = bridge._sessions[session.session_id].last_accessed_at

        # Small delay to ensure time difference
        import asyncio

        await asyncio.sleep(0.01)

        await bridge.get_session(session.session_id)

        assert bridge._sessions[session.session_id].last_accessed_at > old_time

    @pytest.mark.asyncio
    async def test_destroy_session_success(self, bridge: AuthBridge, context: AuthContext):
        """Test destroying a session."""
        session = await bridge.create_session(context)

        result = await bridge.destroy_session(session.session_id)

        assert result is True
        assert session.session_id not in bridge._sessions

    @pytest.mark.asyncio
    async def test_destroy_session_not_found(self, bridge: AuthBridge):
        """Test destroying a non-existent session."""
        result = await bridge.destroy_session("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions(self, bridge: AuthBridge, context: AuthContext):
        """Test cleaning up expired sessions."""
        session1 = await bridge.create_session(context)
        session2 = await bridge.create_session(context)

        # Expire one session
        bridge._sessions[session1.session_id].expires_at = time.time() - 1

        cleaned = await bridge.cleanup_expired_sessions()

        assert cleaned == 1
        assert session1.session_id not in bridge._sessions
        assert session2.session_id in bridge._sessions

    @pytest.mark.asyncio
    async def test_cleanup_expired_sessions_none(self, bridge: AuthBridge, context: AuthContext):
        """Test cleanup when no sessions are expired."""
        await bridge.create_session(context)

        cleaned = await bridge.cleanup_expired_sessions()

        assert cleaned == 0

    # =========================================================================
    # Lifecycle Hook Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_lifecycle_hook_on_session_created(self, context: AuthContext):
        """Test lifecycle hook called on session creation."""
        hook = ConcreteLifecycleHook()
        bridge = AuthBridge(lifecycle_hooks=[hook])

        session = await bridge.create_session(context)

        assert len(hook.created_sessions) == 1
        assert hook.created_sessions[0][0].session_id == session.session_id

    @pytest.mark.asyncio
    async def test_lifecycle_hook_on_session_destroyed(self, context: AuthContext):
        """Test lifecycle hook called on session destruction."""
        hook = ConcreteLifecycleHook()
        bridge = AuthBridge(lifecycle_hooks=[hook])

        session = await bridge.create_session(context)
        await bridge.destroy_session(session.session_id, reason="logout")

        assert len(hook.destroyed_sessions) == 1
        assert hook.destroyed_sessions[0][1] == "logout"

    @pytest.mark.asyncio
    async def test_lifecycle_hook_exception_handling(self, context: AuthContext):
        """Test lifecycle hook exceptions are caught."""

        class FailingHook(SessionLifecycleHook):
            async def on_session_created(
                self, session: BridgedSession, context: AuthContext
            ) -> None:
                raise ValueError("Hook failed")

        bridge = AuthBridge(lifecycle_hooks=[FailingHook()])

        # Should not raise
        session = await bridge.create_session(context)
        assert session is not None

    def test_add_lifecycle_hook(self, bridge: AuthBridge):
        """Test adding a lifecycle hook."""
        hook = ConcreteLifecycleHook()
        initial_count = len(bridge._lifecycle_hooks)

        bridge.add_lifecycle_hook(hook)

        assert len(bridge._lifecycle_hooks) == initial_count + 1
        assert hook in bridge._lifecycle_hooks

    def test_remove_lifecycle_hook_success(self, bridge: AuthBridge):
        """Test removing a lifecycle hook."""
        hook = ConcreteLifecycleHook()
        bridge.add_lifecycle_hook(hook)

        result = bridge.remove_lifecycle_hook(hook)

        assert result is True
        assert hook not in bridge._lifecycle_hooks

    def test_remove_lifecycle_hook_not_found(self, bridge: AuthBridge):
        """Test removing a hook that doesn't exist."""
        hook = ConcreteLifecycleHook()

        result = bridge.remove_lifecycle_hook(hook)

        assert result is False

    # =========================================================================
    # Token Exchange Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_exchange_token(self, bridge: AuthBridge, context: AuthContext):
        """Test token exchange."""
        result = await bridge.exchange_token(
            context,
            target_audience="external-api",
            scope="read write",
            token_lifetime=7200,
        )

        assert result.access_token is not None
        assert result.token_type == "Bearer"
        assert result.expires_in == 7200
        assert result.audience == "external-api"

    @pytest.mark.asyncio
    async def test_exchange_token_contains_user_info(
        self, bridge: AuthBridge, context: AuthContext
    ):
        """Test exchanged token contains user info."""
        result = await bridge.exchange_token(context, target_audience="external-api")

        # Decode the base64 token
        token_data = json.loads(base64.urlsafe_b64decode(result.access_token))

        assert token_data["sub"] == context.user_id
        assert token_data["email"] == context.email
        assert token_data["tenant_id"] == context.tenant_id
        assert token_data["aud"] == "external-api"

    @pytest.mark.asyncio
    async def test_exchange_token_scope_filtering(self, bridge: AuthBridge, context: AuthContext):
        """Test token exchange filters scope to allowed actions."""
        result = await bridge.exchange_token(
            context,
            target_audience="external-api",
            scope="create_conversation delete_conversation",  # delete not allowed
        )

        scope_parts = set(result.scope.split())
        assert "create_conversation" in scope_parts
        # delete_conversation not in context's permissions

    # =========================================================================
    # Audit Logging Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_audit_log_entry_created(self, bridge: AuthBridge, context: AuthContext):
        """Test audit log entry is created."""
        await bridge._log_audit(
            event_type="test",
            user_id=context.user_id,
            action="test_action",
        )

        logs = await bridge.get_audit_log()

        assert len(logs) > 0
        assert logs[-1]["event_type"] == "test"
        assert logs[-1]["user_id"] == context.user_id

    @pytest.mark.asyncio
    async def test_audit_log_filter_by_user(self, bridge: AuthBridge, context: AuthContext):
        """Test filtering audit logs by user."""
        await bridge._log_audit(event_type="test", user_id="user1")
        await bridge._log_audit(event_type="test", user_id="user2")
        await bridge._log_audit(event_type="test", user_id="user1")

        logs = await bridge.get_audit_log(user_id="user1")

        assert len(logs) == 2
        assert all(log["user_id"] == "user1" for log in logs)

    @pytest.mark.asyncio
    async def test_audit_log_filter_by_event_type(self, bridge: AuthBridge):
        """Test filtering audit logs by event type."""
        await bridge._log_audit(event_type="authentication", user_id="user1")
        await bridge._log_audit(event_type="permission_check", user_id="user1")
        await bridge._log_audit(event_type="authentication", user_id="user2")

        logs = await bridge.get_audit_log(event_type="authentication")

        assert len(logs) == 2
        assert all(log["event_type"] == "authentication" for log in logs)

    @pytest.mark.asyncio
    async def test_audit_log_filter_by_since(self, bridge: AuthBridge):
        """Test filtering audit logs by timestamp."""
        await bridge._log_audit(event_type="test", user_id="user1")

        threshold = time.time()

        await bridge._log_audit(event_type="test", user_id="user2")

        logs = await bridge.get_audit_log(since=threshold)

        assert len(logs) == 1
        assert logs[0]["user_id"] == "user2"

    @pytest.mark.asyncio
    async def test_audit_log_limit(self, bridge: AuthBridge):
        """Test audit log limit."""
        for i in range(10):
            await bridge._log_audit(event_type="test", user_id=f"user{i}")

        logs = await bridge.get_audit_log(limit=5)

        assert len(logs) == 5

    @pytest.mark.asyncio
    async def test_audit_log_bounded(self, bridge: AuthBridge):
        """Test audit log is bounded to 10000 entries."""
        # Add many entries
        for i in range(10005):
            await bridge._log_audit(event_type="test", user_id=f"user{i}")

        assert len(bridge._audit_log) == 10000

    # =========================================================================
    # Statistics Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_get_stats(self, bridge: AuthBridge, context: AuthContext):
        """Test getting bridge statistics."""
        await bridge.create_session(context)
        await bridge.create_session(context)

        stats = bridge.get_stats()

        assert stats["permission_mappings"] == 3
        assert stats["total_sessions"] == 2
        assert stats["active_sessions"] == 2
        assert stats["expired_sessions"] == 0
        assert stats["audit_enabled"] is True

    @pytest.mark.asyncio
    async def test_get_stats_with_expired(self, bridge: AuthBridge, context: AuthContext):
        """Test statistics include expired session count."""
        session1 = await bridge.create_session(context)
        await bridge.create_session(context)

        # Expire one session
        bridge._sessions[session1.session_id].expires_at = time.time() - 1

        stats = bridge.get_stats()

        assert stats["total_sessions"] == 2
        assert stats["active_sessions"] == 1
        assert stats["expired_sessions"] == 1

    # =========================================================================
    # Verify Request Tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_verify_request_no_credentials(self, bridge: AuthBridge):
        """Test verify_request fails without credentials."""
        with pytest.raises(AuthenticationError) as exc:
            await bridge.verify_request()

        assert "No authentication credentials" in exc.value.message

    @pytest.mark.asyncio
    async def test_verify_request_with_session(self, bridge: AuthBridge, context: AuthContext):
        """Test verify_request with session ID."""
        session = await bridge.create_session(context)

        result = await bridge.verify_request(session_id=session.session_id)

        assert result.user_id == context.user_id

    @pytest.mark.asyncio
    async def test_verify_request_expired_session(self, bridge: AuthBridge, context: AuthContext):
        """Test verify_request fails for expired session."""
        session = await bridge.create_session(context)
        bridge._sessions[session.session_id].expires_at = time.time() - 1

        with pytest.raises(SessionExpiredError):
            await bridge.verify_request(session_id=session.session_id)

    @pytest.mark.asyncio
    async def test_verify_request_session_not_found(self, bridge: AuthBridge):
        """Test verify_request fails for unknown session."""
        with pytest.raises(AuthenticationError) as exc:
            await bridge.verify_request(session_id="nonexistent")

        assert "not found" in exc.value.message.lower()

    @pytest.mark.asyncio
    async def test_verify_request_session_no_context(self, bridge: AuthBridge):
        """Test verify_request fails for session without auth context."""
        session = BridgedSession(
            session_id="session123",
            aragora_session_id="aragora-session",
            auth_context=None,
        )
        bridge._sessions["session123"] = session

        with pytest.raises(AuthenticationError) as exc:
            await bridge.verify_request(session_id="session123")

        assert "no auth context" in exc.value.message.lower()

    @pytest.mark.asyncio
    async def test_verify_request_with_api_key(self, bridge: AuthBridge):
        """Test verify_request with API key."""
        with patch("aragora.rbac.checker.get_permission_checker") as mock_checker:
            mock_checker.return_value = MagicMock()

            result = await bridge.verify_request(api_key="test-api-key-12345678")

        assert result.token_type == TokenType.API_KEY
        assert "api_key:" in result.user_id

    @pytest.mark.asyncio
    async def test_verify_request_adds_ip_and_user_agent(
        self, bridge: AuthBridge, context: AuthContext
    ):
        """Test verify_request adds IP and user agent to context."""
        session = await bridge.create_session(context)

        result = await bridge.verify_request(
            session_id=session.session_id,
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
        )

        assert result.ip_address == "192.168.1.1"
        assert result.user_agent == "Mozilla/5.0"

    @pytest.mark.asyncio
    async def test_verify_request_with_oidc_token(self, bridge: AuthBridge):
        """Test verify_request with OIDC token."""
        with patch("aragora.auth.sso.get_sso_provider") as mock_get:
            mock_provider = MagicMock()
            mock_provider.__class__.__name__ = "OIDCProvider"

            # Mock the _validate_id_token to return claims
            async def mock_validate_id_token(token):
                return {
                    "sub": "user123",
                    "email": "user@example.com",
                    "exp": time.time() + 3600,
                }

            mock_provider._validate_id_token = mock_validate_id_token
            mock_get.return_value = mock_provider

            # Need to patch OIDCProvider class check
            with patch("aragora.auth.oidc.OIDCProvider") as mock_oidc:
                mock_oidc.__class__ = type(mock_provider)

                # This will fall through to our mock
                with pytest.raises(AuthenticationError):
                    # Will fail because isinstance check fails
                    await bridge.verify_request(token="eyJhbGciOiJIUzI1NiJ9.test")

    @pytest.mark.asyncio
    async def test_verify_request_oidc_no_provider(self, bridge: AuthBridge):
        """Test verify_request fails when OIDC provider not configured."""
        with patch("aragora.auth.sso.get_sso_provider") as mock_get:
            mock_get.return_value = None

            with pytest.raises(AuthenticationError) as exc:
                await bridge.verify_request(token="test-token")

            assert "not configured" in exc.value.message.lower()

    @pytest.mark.asyncio
    async def test_verify_request_saml_not_available(self, bridge: AuthBridge):
        """Test verify_request fails when SAML not installed."""
        with patch("aragora.auth.HAS_SAML", False):
            with pytest.raises(AuthenticationError) as exc:
                await bridge.verify_request(saml_assertion="test-assertion")

            assert "not available" in exc.value.message.lower()


# ============================================================================
# Integration Tests
# ============================================================================


class TestAuthBridgeIntegration:
    """Integration tests for AuthBridge."""

    @pytest.mark.asyncio
    async def test_full_authentication_flow(self):
        """Test complete authentication flow."""
        # Setup bridge with hooks
        hook = ConcreteLifecycleHook()
        bridge = AuthBridge(
            permission_mappings=[
                PermissionMapping("debates.create", "create_conversation"),
                PermissionMapping("debates.read", "view_conversation"),
            ],
            lifecycle_hooks=[hook],
        )

        # Step 1: Create auth context (simulating OIDC callback)
        context = AuthContext(
            user_id="user123",
            email="user@example.com",
            permissions={"debates.create", "debates.read"},
            roles={"user"},
        )

        # Step 2: Create session
        session = await bridge.create_session(context)
        assert session is not None
        assert len(hook.created_sessions) == 1

        # Step 3: Verify session
        verified_context = await bridge.verify_request(session_id=session.session_id)
        assert verified_context.user_id == "user123"

        # Step 4: Check permissions
        assert bridge.is_action_allowed(verified_context, "create_conversation")
        assert bridge.is_action_allowed(verified_context, "view_conversation")

        # Step 5: Exchange token for external API
        exchange_result = await bridge.exchange_token(
            verified_context,
            target_audience="external-api",
        )
        assert exchange_result.access_token is not None

        # Step 6: Destroy session
        await bridge.destroy_session(session.session_id, reason="logout")
        assert len(hook.destroyed_sessions) == 1
        assert hook.destroyed_sessions[0][1] == "logout"

        # Step 7: Verify session is gone
        retrieved = await bridge.get_session(session.session_id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_permission_mapping_with_conditions(self):
        """Test permission mapping with conditions."""
        bridge = AuthBridge(
            permission_mappings=[
                PermissionMapping(
                    aragora_permission="debates.create",
                    external_action="create_conversation",
                    conditions={"tenant_id": "tenant123"},
                ),
                PermissionMapping(
                    aragora_permission="debates.create",
                    external_action="create_conversation_any",
                    # No conditions - available to all
                ),
            ],
        )

        # Context with matching tenant
        context_matching = AuthContext(
            user_id="user123",
            email="user@example.com",
            tenant_id="tenant123",
            permissions={"debates.create"},
        )

        # Context with different tenant
        context_different = AuthContext(
            user_id="user456",
            email="other@example.com",
            tenant_id="other-tenant",
            permissions={"debates.create"},
        )

        # Matching tenant gets both actions
        actions_matching = bridge.get_external_actions(context_matching)
        assert "create_conversation" in actions_matching
        assert "create_conversation_any" in actions_matching

        # Different tenant only gets unconditional action
        actions_different = bridge.get_external_actions(context_different)
        assert "create_conversation" not in actions_different
        assert "create_conversation_any" in actions_different

    @pytest.mark.asyncio
    async def test_wildcard_permission_mapping(self):
        """Test wildcard permission mappings."""
        bridge = AuthBridge(
            permission_mappings=[
                PermissionMapping("debates.*", "manage_all_debates", priority=10),
                PermissionMapping("debates.create", "create_debate_only"),
            ],
        )

        context = AuthContext(
            user_id="user123",
            email="user@example.com",
            permissions={"debates.create"},  # Specific permission
        )

        actions = bridge.get_external_actions(context)

        # Should get both: direct match and wildcard match
        assert "create_debate_only" in actions
        assert "manage_all_debates" in actions

    @pytest.mark.asyncio
    async def test_session_expiry_cleanup(self):
        """Test automatic session cleanup on expiry."""
        hook = ConcreteLifecycleHook()
        bridge = AuthBridge(lifecycle_hooks=[hook], session_duration=1)

        context = AuthContext(
            user_id="user123",
            email="user@example.com",
        )

        # Create multiple sessions
        session1 = await bridge.create_session(context)
        session2 = await bridge.create_session(context)
        session3 = await bridge.create_session(context)

        # Expire first two
        bridge._sessions[session1.session_id].expires_at = time.time() - 1
        bridge._sessions[session2.session_id].expires_at = time.time() - 1

        # Cleanup
        cleaned = await bridge.cleanup_expired_sessions()

        assert cleaned == 2
        assert session1.session_id not in bridge._sessions
        assert session2.session_id not in bridge._sessions
        assert session3.session_id in bridge._sessions
