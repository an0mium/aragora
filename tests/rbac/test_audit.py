"""
Comprehensive tests for RBAC audit logging module.

Tests cover:
- AuditEvent creation, serialization, deserialization (from_dict)
- AuditEventType enum values
- HMAC-SHA256 signing and verification
- AuthorizationAuditor event handling and filtering
- Handler management (add/remove)
- Event buffer management
- Global auditor instance management
- Convenience functions (log_permission_check)
- Async log_event method
- PersistentAuditHandler (batching, flushing, querying, stats, break-glass)
- Module-level singletons (get_persistent_handler, enable_persistent_auditing)
- Error handling and edge cases
"""

from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, call

import pytest

from aragora.rbac.audit import (
    AuditEvent,
    AuditEventType,
    AuthorizationAuditor,
    PersistentAuditHandler,
    compute_event_signature,
    enable_persistent_auditing,
    get_audit_signing_key,
    get_auditor,
    get_persistent_handler,
    log_permission_check,
    set_audit_signing_key,
    set_auditor,
    set_persistent_handler,
    verify_event_signature,
)
from aragora.rbac.models import AuthorizationContext, AuthorizationDecision, RoleAssignment

import aragora.rbac.audit as audit_module


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset module-level global singletons before each test."""
    audit_module._auditor = None
    audit_module._persistent_handler = None
    # Reset signing key so each test gets a fresh ephemeral key
    audit_module._AUDIT_SIGNING_KEY = None
    yield
    audit_module._auditor = None
    audit_module._persistent_handler = None
    audit_module._AUDIT_SIGNING_KEY = None


@pytest.fixture
def signing_key():
    """Provide a deterministic 32-byte signing key."""
    key = b"\x01" * 32
    set_audit_signing_key(key)
    return key


@pytest.fixture
def sample_context():
    """Create a sample AuthorizationContext."""
    return AuthorizationContext(
        user_id="user-1",
        org_id="org-1",
        ip_address="192.168.1.1",
        user_agent="TestAgent/1.0",
        request_id="req-001",
    )


@pytest.fixture
def sample_allowed_decision(sample_context):
    """Create an allowed AuthorizationDecision."""
    return AuthorizationDecision(
        permission_key="debates:read",
        allowed=True,
        reason="User has read permission",
        context=sample_context,
    )


@pytest.fixture
def sample_denied_decision(sample_context):
    """Create a denied AuthorizationDecision."""
    return AuthorizationDecision(
        permission_key="debates:delete",
        allowed=False,
        reason="Insufficient permissions",
        context=sample_context,
    )


@pytest.fixture
def mock_store():
    """Create a mock audit store."""
    store = MagicMock()
    store.log_event = MagicMock()
    store.get_log = MagicMock(return_value=[])
    store.get_log_count = MagicMock(return_value=0)
    return store


# ============================================================================
# AuditEventType Tests
# ============================================================================


class TestAuditEventType:
    """Tests for AuditEventType enum."""

    def test_permission_event_types_exist(self):
        """Test permission-related event types exist."""
        assert AuditEventType.PERMISSION_GRANTED
        assert AuditEventType.PERMISSION_DENIED

    def test_role_event_types_exist(self):
        """Test role-related event types exist."""
        assert AuditEventType.ROLE_ASSIGNED
        assert AuditEventType.ROLE_REVOKED
        assert AuditEventType.ROLE_CREATED
        assert AuditEventType.ROLE_DELETED
        assert AuditEventType.ROLE_MODIFIED

    def test_session_event_types_exist(self):
        """Test session-related event types exist."""
        assert AuditEventType.SESSION_CREATED
        assert AuditEventType.SESSION_EXPIRED
        assert AuditEventType.SESSION_REVOKED

    def test_api_key_event_types_exist(self):
        """Test API key event types exist."""
        assert AuditEventType.API_KEY_CREATED
        assert AuditEventType.API_KEY_REVOKED
        assert AuditEventType.API_KEY_USED

    def test_admin_event_types_exist(self):
        """Test admin action event types exist."""
        assert AuditEventType.IMPERSONATION_START
        assert AuditEventType.IMPERSONATION_END
        assert AuditEventType.POLICY_CHANGED

    def test_break_glass_event_types_exist(self):
        """Test break-glass event types exist."""
        assert AuditEventType.BREAK_GLASS_ACTIVATED
        assert AuditEventType.BREAK_GLASS_DEACTIVATED
        assert AuditEventType.BREAK_GLASS_ACTION

    def test_approval_event_types_exist(self):
        """Test approval workflow event types exist."""
        assert AuditEventType.APPROVAL_REQUESTED
        assert AuditEventType.APPROVAL_GRANTED
        assert AuditEventType.APPROVAL_DENIED
        assert AuditEventType.APPROVAL_EXPIRED

    def test_event_type_values_are_strings(self):
        """Test event type values are lowercase strings."""
        assert AuditEventType.PERMISSION_GRANTED.value == "permission_granted"
        assert AuditEventType.ROLE_ASSIGNED.value == "role_assigned"
        assert AuditEventType.BREAK_GLASS_ACTIVATED.value == "break_glass_activated"

    def test_custom_event_type(self):
        """Test CUSTOM event type exists."""
        assert AuditEventType.CUSTOM.value == "custom"


# ============================================================================
# AuditEvent Tests
# ============================================================================


class TestAuditEvent:
    """Tests for AuditEvent dataclass."""

    def test_event_has_auto_generated_id(self):
        """Test event gets automatic UUID."""
        event = AuditEvent()
        assert event.id is not None
        assert len(event.id) > 0

    def test_event_default_event_type(self):
        """Test default event type is PERMISSION_GRANTED."""
        event = AuditEvent()
        assert event.event_type == AuditEventType.PERMISSION_GRANTED

    def test_event_default_timestamp(self):
        """Test event gets automatic timestamp."""
        before = datetime.now(timezone.utc)
        event = AuditEvent()
        after = datetime.now(timezone.utc)
        assert before <= event.timestamp <= after

    def test_event_with_all_fields(self):
        """Test creating event with all fields populated."""
        event = AuditEvent(
            event_type=AuditEventType.PERMISSION_DENIED,
            user_id="user-123",
            org_id="org-456",
            actor_id="admin-789",
            resource_type="debate",
            resource_id="debate-abc",
            permission_key="debates:delete",
            decision=False,
            reason="Insufficient permissions",
            ip_address="192.168.1.1",
            user_agent="Mozilla/5.0",
            request_id="req-xyz",
            metadata={"extra": "data"},
        )

        assert event.event_type == AuditEventType.PERMISSION_DENIED
        assert event.user_id == "user-123"
        assert event.org_id == "org-456"
        assert event.actor_id == "admin-789"
        assert event.resource_type == "debate"
        assert event.resource_id == "debate-abc"
        assert event.permission_key == "debates:delete"
        assert event.decision is False
        assert event.reason == "Insufficient permissions"
        assert event.ip_address == "192.168.1.1"
        assert event.user_agent == "Mozilla/5.0"
        assert event.request_id == "req-xyz"
        assert event.metadata["extra"] == "data"

    def test_to_dict_contains_all_fields(self):
        """Test to_dict includes all fields."""
        event = AuditEvent(
            event_type=AuditEventType.ROLE_ASSIGNED,
            user_id="user-1",
            org_id="org-1",
            permission_key="admin",
            decision=True,
        )

        data = event.to_dict()

        assert "id" in data
        assert data["event_type"] == "role_assigned"
        assert data["user_id"] == "user-1"
        assert data["org_id"] == "org-1"
        assert data["decision"] is True
        assert "timestamp" in data
        # to_dict should NOT include 'signature'
        assert "signature" not in data

    def test_to_dict_timestamp_is_iso_format(self):
        """Test timestamp is serialized as ISO format."""
        event = AuditEvent()
        data = event.to_dict()
        parsed = datetime.fromisoformat(data["timestamp"])
        assert parsed is not None

    def test_to_json_produces_valid_json(self):
        """Test to_json produces valid JSON string."""
        event = AuditEvent(
            event_type=AuditEventType.API_KEY_CREATED,
            user_id="user-1",
            metadata={"scopes": ["read", "write"]},
        )

        json_str = event.to_json()
        parsed = json.loads(json_str)

        assert parsed["event_type"] == "api_key_created"
        assert parsed["user_id"] == "user-1"
        assert parsed["metadata"]["scopes"] == ["read", "write"]

    def test_to_signed_dict_includes_signature(self, signing_key):
        """Test to_signed_dict produces dict with HMAC signature."""
        event = AuditEvent(
            event_type=AuditEventType.PERMISSION_GRANTED,
            user_id="user-1",
        )

        signed = event.to_signed_dict()

        assert "signature" in signed
        assert isinstance(signed["signature"], str)
        assert len(signed["signature"]) == 64  # SHA-256 hex digest
        # Signature should also be stored on the event itself
        assert event.signature == signed["signature"]

    def test_verify_signature_valid(self, signing_key):
        """Test verify_signature returns True for valid signature."""
        event = AuditEvent(
            event_type=AuditEventType.PERMISSION_DENIED,
            user_id="user-1",
            reason="Denied",
        )
        event.to_signed_dict()

        assert event.verify_signature() is True

    def test_verify_signature_no_signature(self):
        """Test verify_signature returns False when no signature exists."""
        event = AuditEvent()
        assert event.signature is None
        assert event.verify_signature() is False

    def test_verify_signature_tampered(self, signing_key):
        """Test verify_signature returns False after data tampering."""
        event = AuditEvent(
            event_type=AuditEventType.PERMISSION_GRANTED,
            user_id="user-1",
            reason="Allowed",
        )
        event.to_signed_dict()

        # Tamper with the event data
        event.reason = "Tampered reason"

        assert event.verify_signature() is False

    def test_from_dict_round_trip(self):
        """Test from_dict correctly reconstructs an AuditEvent."""
        original = AuditEvent(
            event_type=AuditEventType.ROLE_REVOKED,
            user_id="user-42",
            org_id="org-7",
            actor_id="admin-1",
            resource_type="role",
            resource_id="editor",
            permission_key="roles:revoke",
            decision=True,
            reason="Policy violation",
            ip_address="10.0.0.1",
            user_agent="AdminUI/2.0",
            request_id="req-999",
            metadata={"note": "automated"},
        )

        data = original.to_dict()
        restored = AuditEvent.from_dict(data)

        assert restored.event_type == AuditEventType.ROLE_REVOKED
        assert restored.user_id == "user-42"
        assert restored.org_id == "org-7"
        assert restored.actor_id == "admin-1"
        assert restored.resource_type == "role"
        assert restored.resource_id == "editor"
        assert restored.permission_key == "roles:revoke"
        assert restored.decision is True
        assert restored.reason == "Policy violation"
        assert restored.ip_address == "10.0.0.1"
        assert restored.user_agent == "AdminUI/2.0"
        assert restored.request_id == "req-999"
        assert restored.metadata == {"note": "automated"}

    def test_from_dict_with_unknown_event_type(self):
        """Test from_dict falls back to CUSTOM for unknown event types."""
        data = {
            "event_type": "totally_unknown_event",
            "user_id": "user-1",
        }
        event = AuditEvent.from_dict(data)
        assert event.event_type == AuditEventType.CUSTOM

    def test_from_dict_with_z_suffix_timestamp(self):
        """Test from_dict handles ISO timestamps with Z suffix."""
        data = {
            "event_type": "permission_granted",
            "timestamp": "2024-06-15T10:30:00Z",
            "user_id": "user-1",
        }
        event = AuditEvent.from_dict(data)
        assert event.timestamp.year == 2024
        assert event.timestamp.month == 6

    def test_from_dict_with_missing_timestamp(self):
        """Test from_dict generates a timestamp when missing."""
        data = {"event_type": "permission_granted", "user_id": "user-1"}
        before = datetime.now(timezone.utc)
        event = AuditEvent.from_dict(data)
        after = datetime.now(timezone.utc)
        assert before <= event.timestamp <= after

    def test_from_dict_preserves_signature(self, signing_key):
        """Test from_dict preserves signature field for later verification."""
        original = AuditEvent(user_id="user-1")
        signed_data = original.to_signed_dict()

        restored = AuditEvent.from_dict(signed_data)
        assert restored.signature == signed_data["signature"]


# ============================================================================
# HMAC Signing and Verification Tests
# ============================================================================


class TestHMACSigning:
    """Tests for HMAC-SHA256 event signing and verification."""

    def test_compute_event_signature_deterministic(self, signing_key):
        """Test compute_event_signature produces same output for same input."""
        event_data = {"event_type": "permission_granted", "user_id": "u1"}
        sig1 = compute_event_signature(event_data)
        sig2 = compute_event_signature(event_data)
        assert sig1 == sig2

    def test_compute_event_signature_strips_existing_signature(self, signing_key):
        """Test compute_event_signature ignores existing signature field."""
        event_data = {"event_type": "permission_granted", "user_id": "u1"}
        sig_without = compute_event_signature(event_data)

        event_data_with_sig = {**event_data, "signature": "old_sig"}
        sig_with = compute_event_signature(event_data_with_sig)

        assert sig_without == sig_with

    def test_verify_event_signature_valid(self, signing_key):
        """Test verify_event_signature returns True for valid signature."""
        event_data = {"event_type": "role_assigned", "user_id": "u1"}
        sig = compute_event_signature(event_data)
        assert verify_event_signature(event_data, sig) is True

    def test_verify_event_signature_invalid(self, signing_key):
        """Test verify_event_signature returns False for tampered data."""
        event_data = {"event_type": "role_assigned", "user_id": "u1"}
        sig = compute_event_signature(event_data)

        # Tamper
        event_data["user_id"] = "u2"
        assert verify_event_signature(event_data, sig) is False

    def test_set_audit_signing_key_rejects_short_key(self):
        """Test set_audit_signing_key rejects keys shorter than 32 bytes."""
        with pytest.raises(ValueError, match="at least 32 bytes"):
            set_audit_signing_key(b"short")

    def test_set_audit_signing_key_accepts_valid_key(self):
        """Test set_audit_signing_key accepts a 32-byte key."""
        key = secrets.token_bytes(32)
        set_audit_signing_key(key)
        assert get_audit_signing_key() == key

    def test_get_audit_signing_key_generates_dev_key(self):
        """Test get_audit_signing_key generates ephemeral key in dev mode."""
        with patch.dict("os.environ", {}, clear=True):
            audit_module._AUDIT_SIGNING_KEY = None
            key = get_audit_signing_key()
            assert isinstance(key, bytes)
            assert len(key) == 32

    def test_get_audit_signing_key_from_env(self):
        """Test get_audit_signing_key loads key from environment variable."""
        hex_key = secrets.token_hex(32)
        with patch.dict("os.environ", {"ARAGORA_AUDIT_SIGNING_KEY": hex_key}):
            audit_module._AUDIT_SIGNING_KEY = None
            key = get_audit_signing_key()
            assert key == bytes.fromhex(hex_key)

    def test_get_audit_signing_key_rejects_invalid_hex(self):
        """Test get_audit_signing_key raises on invalid hex in env."""
        with patch.dict("os.environ", {"ARAGORA_AUDIT_SIGNING_KEY": "not_valid_hex!"}):
            audit_module._AUDIT_SIGNING_KEY = None
            with pytest.raises(RuntimeError, match="Invalid ARAGORA_AUDIT_SIGNING_KEY"):
                get_audit_signing_key()

    def test_get_audit_signing_key_raises_in_production(self):
        """Test get_audit_signing_key raises when key missing in production."""
        env_vars = {"ARAGORA_ENV": "production"}
        # Ensure ARAGORA_AUDIT_SIGNING_KEY is not present
        with patch.dict("os.environ", env_vars, clear=True):
            audit_module._AUDIT_SIGNING_KEY = None
            with pytest.raises(RuntimeError, match="required in production"):
                get_audit_signing_key()

    def test_get_audit_signing_key_raises_in_staging(self):
        """Test get_audit_signing_key raises when key missing in staging."""
        env_vars = {"ARAGORA_ENV": "staging"}
        with patch.dict("os.environ", env_vars, clear=True):
            audit_module._AUDIT_SIGNING_KEY = None
            with pytest.raises(RuntimeError, match="required in staging"):
                get_audit_signing_key()


# ============================================================================
# AuthorizationAuditor Tests
# ============================================================================


class TestAuthorizationAuditor:
    """Tests for AuthorizationAuditor class."""

    def test_auditor_initialization_default(self):
        """Test auditor initializes with defaults."""
        auditor = AuthorizationAuditor()
        assert len(auditor._handlers) >= 1
        assert not auditor._log_denied_only
        assert not auditor._include_cached

    def test_auditor_with_custom_handlers(self):
        """Test auditor with custom handlers."""
        handler1 = MagicMock()
        handler2 = MagicMock()
        auditor = AuthorizationAuditor(handlers=[handler1, handler2])
        assert handler1 in auditor._handlers
        assert handler2 in auditor._handlers

    def test_auditor_log_denied_only_skips_allowed(
        self, sample_allowed_decision, sample_denied_decision
    ):
        """Test log_denied_only filtering skips allowed decisions."""
        handler = MagicMock()
        auditor = AuthorizationAuditor(handlers=[handler], log_denied_only=True)

        auditor.log_decision(sample_allowed_decision)
        auditor.log_decision(sample_denied_decision)

        events = [c[0][0] for c in handler.call_args_list]
        assert not any(e.event_type == AuditEventType.PERMISSION_GRANTED for e in events)
        denied = [e for e in events if e.event_type == AuditEventType.PERMISSION_DENIED]
        assert len(denied) == 1

    def test_auditor_include_cached_false_skips_cached(self, sample_context):
        """Test include_cached=False skips cached decisions."""
        handler = MagicMock()
        auditor = AuthorizationAuditor(handlers=[handler], include_cached=False)

        cached = AuthorizationDecision(
            permission_key="test:read",
            allowed=True,
            reason="From cache",
            context=sample_context,
            cached=True,
        )
        fresh = AuthorizationDecision(
            permission_key="test:write",
            allowed=True,
            reason="Fresh",
            context=sample_context,
            cached=False,
        )

        auditor.log_decision(cached)
        auditor.log_decision(fresh)

        events = [c[0][0] for c in handler.call_args_list]
        assert not any(e.permission_key == "test:read" for e in events)
        assert any(e.permission_key == "test:write" for e in events)

    def test_log_decision_creates_correct_event_types(self, sample_context):
        """Test log_decision creates correct event types for allowed and denied."""
        handler = MagicMock()
        auditor = AuthorizationAuditor(handlers=[handler])

        auditor.log_decision(
            AuthorizationDecision(
                permission_key="test:read",
                allowed=True,
                reason="OK",
                context=sample_context,
            )
        )
        auditor.log_decision(
            AuthorizationDecision(
                permission_key="test:write",
                allowed=False,
                reason="Denied",
                context=sample_context,
            )
        )

        events = [c[0][0] for c in handler.call_args_list]
        assert any(e.event_type == AuditEventType.PERMISSION_GRANTED for e in events)
        assert any(e.event_type == AuditEventType.PERMISSION_DENIED for e in events)

    def test_log_decision_populates_context_fields(self, sample_context):
        """Test log_decision populates ip_address, user_agent, request_id from context."""
        handler = MagicMock()
        auditor = AuthorizationAuditor(handlers=[handler])

        auditor.log_decision(
            AuthorizationDecision(
                permission_key="test:read",
                allowed=True,
                reason="OK",
                context=sample_context,
            )
        )

        event = handler.call_args_list[0][0][0]
        assert event.user_id == "user-1"
        assert event.org_id == "org-1"
        assert event.ip_address == "192.168.1.1"
        assert event.user_agent == "TestAgent/1.0"
        assert event.request_id == "req-001"

    def test_log_decision_handles_none_context(self):
        """Test log_decision works when decision.context is None."""
        handler = MagicMock()
        auditor = AuthorizationAuditor(handlers=[handler])

        decision = AuthorizationDecision(
            permission_key="test:read",
            allowed=True,
            reason="OK",
            context=None,
        )
        auditor.log_decision(decision)

        event = handler.call_args_list[0][0][0]
        assert event.user_id is None
        assert event.org_id is None

    def test_log_role_assignment(self):
        """Test logging role assignment."""
        handler = MagicMock()
        auditor = AuthorizationAuditor(handlers=[handler])

        assignment = RoleAssignment(id="a-1", user_id="user-1", role_id="admin", org_id="org-1")
        auditor.log_role_assignment(
            assignment=assignment, actor_id="admin-1", ip_address="10.0.0.1"
        )

        events = [c[0][0] for c in handler.call_args_list]
        role_events = [e for e in events if e.event_type == AuditEventType.ROLE_ASSIGNED]
        assert len(role_events) >= 1
        ev = role_events[0]
        assert ev.user_id == "user-1"
        assert ev.resource_id == "admin"
        assert ev.actor_id == "admin-1"
        assert ev.ip_address == "10.0.0.1"
        assert "assignment_id" in ev.metadata

    def test_log_role_revocation(self):
        """Test logging role revocation."""
        handler = MagicMock()
        auditor = AuthorizationAuditor(handlers=[handler])

        auditor.log_role_revocation(
            user_id="user-1",
            role_id="admin",
            org_id="org-1",
            actor_id="superadmin-1",
            reason="Policy violation",
            ip_address="10.0.0.1",
        )

        events = [c[0][0] for c in handler.call_args_list]
        revoke_events = [e for e in events if e.event_type == AuditEventType.ROLE_REVOKED]
        assert len(revoke_events) >= 1
        assert "Policy violation" in revoke_events[0].reason

    def test_log_api_key_created(self):
        """Test logging API key creation."""
        handler = MagicMock()
        auditor = AuthorizationAuditor(handlers=[handler])

        auditor.log_api_key_created(
            user_id="user-1",
            key_id="key-abc",
            scopes={"read", "write"},
            actor_id="user-1",
            ip_address="192.168.1.1",
        )

        events = [c[0][0] for c in handler.call_args_list]
        key_events = [e for e in events if e.event_type == AuditEventType.API_KEY_CREATED]
        assert len(key_events) >= 1
        assert key_events[0].resource_id == "key-abc"
        assert "scopes" in key_events[0].metadata

    def test_log_api_key_created_defaults_actor_to_user(self):
        """Test log_api_key_created defaults actor_id to user_id when not provided."""
        handler = MagicMock()
        auditor = AuthorizationAuditor(handlers=[handler])

        auditor.log_api_key_created(user_id="user-1", key_id="key-1", scopes={"read"})

        events = [c[0][0] for c in handler.call_args_list]
        key_events = [e for e in events if e.event_type == AuditEventType.API_KEY_CREATED]
        assert key_events[0].actor_id == "user-1"

    def test_log_api_key_revoked(self):
        """Test logging API key revocation."""
        handler = MagicMock()
        auditor = AuthorizationAuditor(handlers=[handler])

        auditor.log_api_key_revoked(
            user_id="user-1",
            key_id="key-abc",
            actor_id="admin-1",
            reason="Compromised",
            ip_address="192.168.1.1",
        )

        events = [c[0][0] for c in handler.call_args_list]
        key_events = [e for e in events if e.event_type == AuditEventType.API_KEY_REVOKED]
        assert len(key_events) >= 1
        assert key_events[0].reason == "Compromised"

    def test_log_impersonation_start(self):
        """Test logging impersonation start."""
        handler = MagicMock()
        auditor = AuthorizationAuditor(handlers=[handler])

        auditor.log_impersonation_start(
            actor_id="admin-1",
            target_user_id="user-1",
            org_id="org-1",
            reason="Customer support ticket #123",
            ip_address="10.0.0.1",
        )

        events = [c[0][0] for c in handler.call_args_list]
        imp_events = [e for e in events if e.event_type == AuditEventType.IMPERSONATION_START]
        assert len(imp_events) >= 1
        assert imp_events[0].actor_id == "admin-1"
        assert imp_events[0].user_id == "user-1"
        assert "ticket" in imp_events[0].reason.lower()

    def test_log_impersonation_end(self):
        """Test logging impersonation end."""
        handler = MagicMock()
        auditor = AuthorizationAuditor(handlers=[handler])

        auditor.log_impersonation_end(
            actor_id="admin-1",
            target_user_id="user-1",
            org_id="org-1",
        )

        events = [c[0][0] for c in handler.call_args_list]
        imp_events = [e for e in events if e.event_type == AuditEventType.IMPERSONATION_END]
        assert len(imp_events) >= 1

    def test_log_session_event(self):
        """Test logging session events."""
        handler = MagicMock()
        auditor = AuthorizationAuditor(handlers=[handler])

        auditor.log_session_event(
            event_type=AuditEventType.SESSION_CREATED,
            user_id="user-1",
            session_id="session-xyz",
            ip_address="192.168.1.1",
            user_agent="Chrome/120.0",
            reason="User logged in",
        )

        events = [c[0][0] for c in handler.call_args_list]
        session_events = [e for e in events if e.event_type == AuditEventType.SESSION_CREATED]
        assert len(session_events) >= 1
        assert session_events[0].resource_id == "session-xyz"
        assert session_events[0].user_agent == "Chrome/120.0"

    @pytest.mark.asyncio
    async def test_log_event_known_type(self):
        """Test async log_event with a known AuditEventType string."""
        handler = MagicMock()
        auditor = AuthorizationAuditor(handlers=[handler])

        await auditor.log_event(
            event_type="break_glass_activated",
            details={"ticket": "INC-001"},
            user_id="admin-1",
        )

        events = [c[0][0] for c in handler.call_args_list]
        bg_events = [e for e in events if e.event_type == AuditEventType.BREAK_GLASS_ACTIVATED]
        assert len(bg_events) == 1
        assert bg_events[0].metadata == {"ticket": "INC-001"}

    @pytest.mark.asyncio
    async def test_log_event_unknown_type_falls_back_to_custom(self):
        """Test async log_event falls back to CUSTOM for unrecognized types."""
        handler = MagicMock()
        auditor = AuthorizationAuditor(handlers=[handler])

        await auditor.log_event(
            event_type="my_special_event",
            details={"info": "test"},
            category="special",
        )

        events = [c[0][0] for c in handler.call_args_list]
        custom = [e for e in events if e.event_type == AuditEventType.CUSTOM]
        assert len(custom) == 1
        assert custom[0].reason == "my_special_event"
        assert custom[0].resource_type == "special"

    @pytest.mark.asyncio
    async def test_log_event_defaults(self):
        """Test async log_event uses defaults for missing fields."""
        handler = MagicMock()
        auditor = AuthorizationAuditor(handlers=[handler])

        await auditor.log_event(event_type="permission_granted")

        events = [c[0][0] for c in handler.call_args_list]
        assert events[0].user_id == "system"
        assert events[0].resource_type == "unknown"


# ============================================================================
# Handler Management Tests
# ============================================================================


class TestAuditorHandlerManagement:
    """Tests for auditor handler management."""

    def test_add_handler(self):
        """Test adding a handler."""
        auditor = AuthorizationAuditor()
        new_handler = MagicMock()
        initial_count = len(auditor._handlers)

        auditor.add_handler(new_handler)

        assert len(auditor._handlers) == initial_count + 1
        assert new_handler in auditor._handlers

    def test_remove_handler(self):
        """Test removing a handler."""
        handler = MagicMock()
        auditor = AuthorizationAuditor(handlers=[handler])

        auditor.remove_handler(handler)
        assert handler not in auditor._handlers

    def test_remove_nonexistent_handler_safe(self):
        """Test removing handler that doesn't exist doesn't raise."""
        auditor = AuthorizationAuditor()
        nonexistent = MagicMock()
        auditor.remove_handler(nonexistent)  # Should not raise


# ============================================================================
# Event Buffer Tests
# ============================================================================


class TestAuditorEventBuffer:
    """Tests for auditor event buffer."""

    def test_buffer_stores_events(self, sample_context):
        """Test events are buffered."""
        auditor = AuthorizationAuditor()
        auditor.log_decision(
            AuthorizationDecision(
                permission_key="test:read",
                allowed=True,
                reason="OK",
                context=sample_context,
            )
        )
        assert len(auditor._event_buffer) >= 1

    def test_flush_buffer_returns_events(self, sample_context):
        """Test flush_buffer returns and clears events."""
        auditor = AuthorizationAuditor()
        auditor.log_decision(
            AuthorizationDecision(
                permission_key="test:read",
                allowed=True,
                reason="OK",
                context=sample_context,
            )
        )

        events = auditor.flush_buffer()
        assert len(events) >= 1
        assert len(auditor._event_buffer) == 0

    def test_buffer_truncated_at_max_size(self, sample_context):
        """Test buffer is truncated at max size."""
        auditor = AuthorizationAuditor()
        auditor._buffer_size = 5

        for i in range(10):
            auditor.log_decision(
                AuthorizationDecision(
                    permission_key=f"test:perm_{i}",
                    allowed=True,
                    reason=f"OK {i}",
                    context=sample_context,
                )
            )

        assert len(auditor._event_buffer) <= 5


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestAuditorErrorHandling:
    """Tests for auditor error handling."""

    def test_handler_error_doesnt_stop_other_handlers(self, sample_context):
        """Test handler errors don't prevent other handlers from running."""
        failing_handler = MagicMock(side_effect=Exception("Handler failed"))
        working_handler = MagicMock()

        auditor = AuthorizationAuditor(handlers=[failing_handler, working_handler])

        auditor.log_decision(
            AuthorizationDecision(
                permission_key="test:read",
                allowed=True,
                reason="OK",
                context=sample_context,
            )
        )

        assert working_handler.called

    def test_handler_error_logged(self, sample_context):
        """Test handler errors are logged."""
        failing_handler = MagicMock(side_effect=ValueError("bad value"))
        auditor = AuthorizationAuditor(handlers=[failing_handler])

        with patch("aragora.rbac.audit.logger") as mock_logger:
            auditor.log_decision(
                AuthorizationDecision(
                    permission_key="test:read",
                    allowed=True,
                    reason="OK",
                    context=sample_context,
                )
            )
            assert mock_logger.error.called

    def test_unexpected_exception_handler_also_logged(self, sample_context):
        """Test unexpected exceptions in handlers are also logged."""
        # Use an exception type not in the explicit catch list
        failing_handler = MagicMock(side_effect=KeyError("unexpected"))
        auditor = AuthorizationAuditor(handlers=[failing_handler])

        with patch("aragora.rbac.audit.logger") as mock_logger:
            auditor.log_decision(
                AuthorizationDecision(
                    permission_key="test:read",
                    allowed=True,
                    reason="OK",
                    context=sample_context,
                )
            )
            assert mock_logger.error.called


# ============================================================================
# Global Auditor Tests
# ============================================================================


class TestGlobalAuditor:
    """Tests for global auditor instance."""

    def test_get_auditor_returns_instance(self):
        """Test get_auditor returns an instance."""
        auditor = get_auditor()
        assert auditor is not None
        assert isinstance(auditor, AuthorizationAuditor)

    def test_get_auditor_returns_same_instance(self):
        """Test get_auditor returns singleton."""
        auditor1 = get_auditor()
        auditor2 = get_auditor()
        assert auditor1 is auditor2

    def test_set_auditor_replaces_instance(self):
        """Test set_auditor replaces global instance."""
        custom_auditor = AuthorizationAuditor(log_denied_only=True)
        set_auditor(custom_auditor)
        retrieved = get_auditor()
        assert retrieved is custom_auditor
        assert retrieved._log_denied_only is True


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_log_permission_check_granted(self):
        """Test log_permission_check for granted permission."""
        handler = MagicMock()
        custom_auditor = AuthorizationAuditor(handlers=[handler])
        set_auditor(custom_auditor)

        log_permission_check(
            user_id="user-1",
            permission_key="debates:read",
            allowed=True,
            reason="User has read access",
            resource_id="debate-123",
            org_id="org-1",
            ip_address="10.0.0.1",
        )

        events = [c[0][0] for c in handler.call_args_list]
        assert any(
            e.event_type == AuditEventType.PERMISSION_GRANTED and e.permission_key == "debates:read"
            for e in events
        )

    def test_log_permission_check_denied(self):
        """Test log_permission_check for denied permission."""
        handler = MagicMock()
        custom_auditor = AuthorizationAuditor(handlers=[handler])
        set_auditor(custom_auditor)

        log_permission_check(
            user_id="user-1",
            permission_key="debates:delete",
            allowed=False,
            reason="Insufficient permissions",
        )

        events = [c[0][0] for c in handler.call_args_list]
        assert any(
            e.event_type == AuditEventType.PERMISSION_DENIED
            and e.permission_key == "debates:delete"
            for e in events
        )


# ============================================================================
# Default Log Handler Tests
# ============================================================================


class TestDefaultLogHandler:
    """Tests for default log handler."""

    def test_default_handler_logs_granted_as_info(self, sample_context):
        """Test granted events logged at INFO level."""
        auditor = AuthorizationAuditor()

        with patch("aragora.rbac.audit.logger") as mock_logger:
            auditor.log_decision(
                AuthorizationDecision(
                    permission_key="test:read",
                    allowed=True,
                    reason="OK",
                    context=sample_context,
                )
            )
            mock_logger.log.assert_called()
            assert mock_logger.log.call_args[0][0] == logging.INFO

    def test_default_handler_logs_denied_as_warning(self, sample_context):
        """Test denied events logged at WARNING level."""
        auditor = AuthorizationAuditor()

        with patch("aragora.rbac.audit.logger") as mock_logger:
            auditor.log_decision(
                AuthorizationDecision(
                    permission_key="test:delete",
                    allowed=False,
                    reason="Denied",
                    context=sample_context,
                )
            )
            mock_logger.log.assert_called()
            assert mock_logger.log.call_args[0][0] == logging.WARNING

    def test_default_handler_includes_audit_event_extra(self, sample_context):
        """Test default handler includes audit_event in log extra."""
        auditor = AuthorizationAuditor()

        with patch("aragora.rbac.audit.logger") as mock_logger:
            auditor.log_decision(
                AuthorizationDecision(
                    permission_key="test:read",
                    allowed=True,
                    reason="OK",
                    context=sample_context,
                )
            )
            call_kwargs = mock_logger.log.call_args[1]
            assert "extra" in call_kwargs
            assert "audit_event" in call_kwargs["extra"]


# ============================================================================
# PersistentAuditHandler Tests
# ============================================================================


class TestPersistentAuditHandler:
    """Tests for PersistentAuditHandler."""

    def test_init_defaults(self, mock_store):
        """Test PersistentAuditHandler initializes with correct defaults."""
        handler = PersistentAuditHandler(store=mock_store)
        assert handler._sign_events is True
        assert handler._batch_size == 100
        assert handler._events_written == 0
        assert handler._events_failed == 0

    def test_handle_event_batches(self, mock_store):
        """Test events are batched and not immediately written."""
        handler = PersistentAuditHandler(store=mock_store, batch_size=10)

        event = AuditEvent(user_id="user-1")
        handler.handle_event(event)

        # Should be in batch, not yet written (batch_size=10, only 1 event)
        assert len(handler._batch) == 1
        assert mock_store.log_event.call_count == 0

    def test_handle_event_flushes_at_batch_size(self, mock_store, signing_key):
        """Test batch is flushed when batch_size is reached."""
        handler = PersistentAuditHandler(store=mock_store, batch_size=3)

        for i in range(3):
            handler.handle_event(AuditEvent(user_id=f"user-{i}"))

        # Should have flushed
        assert mock_store.log_event.call_count == 3
        assert handler._events_written == 3
        assert len(handler._batch) == 0

    def test_handle_event_flushes_on_time_interval(self, mock_store, signing_key):
        """Test batch is flushed when flush interval is exceeded."""
        handler = PersistentAuditHandler(
            store=mock_store,
            batch_size=1000,
            flush_interval_seconds=0.0,
        )
        # flush_interval_seconds=0 means any event will trigger a time-based flush
        handler.handle_event(AuditEvent(user_id="user-1"))

        assert mock_store.log_event.call_count == 1

    def test_flush_forces_write(self, mock_store, signing_key):
        """Test flush() forces pending events to be written."""
        handler = PersistentAuditHandler(store=mock_store, batch_size=1000)

        handler.handle_event(AuditEvent(user_id="user-1"))
        handler.handle_event(AuditEvent(user_id="user-2"))
        assert mock_store.log_event.call_count == 0

        handler.flush()
        assert mock_store.log_event.call_count == 2
        assert handler._events_written == 2

    def test_flush_empty_batch_does_nothing(self, mock_store):
        """Test flush() with no pending events is a no-op."""
        handler = PersistentAuditHandler(store=mock_store)
        handler.flush()
        assert mock_store.log_event.call_count == 0

    def test_write_event_signs_when_enabled(self, mock_store, signing_key):
        """Test events are signed before storage when sign_events=True."""
        handler = PersistentAuditHandler(store=mock_store, sign_events=True, batch_size=1)
        handler.handle_event(AuditEvent(user_id="user-1"))

        # Check that the metadata written to store includes a signature
        call_kwargs = mock_store.log_event.call_args[1]
        assert call_kwargs["metadata"]["signature"] is not None

    def test_write_event_unsigned_when_disabled(self, mock_store):
        """Test events are not signed when sign_events=False."""
        handler = PersistentAuditHandler(store=mock_store, sign_events=False, batch_size=1)
        handler.handle_event(AuditEvent(user_id="user-1"))

        call_kwargs = mock_store.log_event.call_args[1]
        assert call_kwargs["metadata"]["signature"] is None

    def test_write_event_failure_increments_failed_count(self, mock_store, signing_key):
        """Test write failure increments events_failed."""
        mock_store.log_event.side_effect = OSError("disk full")
        handler = PersistentAuditHandler(store=mock_store, batch_size=1)

        handler.handle_event(AuditEvent(user_id="user-1"))

        assert handler._events_failed == 1
        assert handler._events_written == 0

    def test_get_events_queries_store(self, mock_store):
        """Test get_events queries the underlying store."""
        mock_store.get_log.return_value = []
        handler = PersistentAuditHandler(store=mock_store)

        events = handler.get_events(user_id="user-1", limit=50)

        mock_store.get_log.assert_called_once()
        call_kwargs = mock_store.get_log.call_args[1]
        assert call_kwargs["user_id"] == "user-1"
        assert call_kwargs["limit"] == 50
        assert events == []

    def test_get_events_reconstructs_audit_events(self, mock_store, signing_key):
        """Test get_events reconstructs AuditEvent from store rows."""
        now = datetime.now(timezone.utc)
        mock_store.get_log.return_value = [
            {
                "id": "row-1",
                "action": "permission_granted",
                "timestamp": now.isoformat(),
                "user_id": "user-1",
                "org_id": "org-1",
                "resource_type": "authorization",
                "resource_id": "res-1",
                "ip_address": "10.0.0.1",
                "user_agent": "Test/1.0",
                "metadata": {
                    "event_id": "evt-1",
                    "actor_id": "admin-1",
                    "permission_key": "debates:read",
                    "decision": True,
                    "reason": "Allowed",
                    "request_id": "req-1",
                    "signature": None,
                },
            }
        ]
        handler = PersistentAuditHandler(store=mock_store)

        events = handler.get_events(verify_signatures=False)

        assert len(events) == 1
        evt = events[0]
        assert evt.id == "evt-1"
        assert evt.event_type == AuditEventType.PERMISSION_GRANTED
        assert evt.user_id == "user-1"
        assert evt.actor_id == "admin-1"
        assert evt.permission_key == "debates:read"

    def test_get_events_with_event_type_enum_filter(self, mock_store):
        """Test get_events passes AuditEventType as action string."""
        mock_store.get_log.return_value = []
        handler = PersistentAuditHandler(store=mock_store)

        handler.get_events(event_type=AuditEventType.ROLE_ASSIGNED)

        call_kwargs = mock_store.get_log.call_args[1]
        assert call_kwargs["action"] == "role_assigned"

    def test_get_events_with_event_type_string_filter(self, mock_store):
        """Test get_events passes string event_type as action."""
        mock_store.get_log.return_value = []
        handler = PersistentAuditHandler(store=mock_store)

        handler.get_events(event_type="custom_event")

        call_kwargs = mock_store.get_log.call_args[1]
        assert call_kwargs["action"] == "custom_event"

    def test_get_events_verifies_valid_signatures(self, mock_store, signing_key):
        """Test get_events verifies signatures and marks valid events."""
        # Create a properly signed event to get the correct signature
        event = AuditEvent(
            id="evt-1",
            event_type=AuditEventType.PERMISSION_GRANTED,
            user_id="user-1",
            decision=True,
            reason="",
        )
        signed_data = event.to_signed_dict()

        mock_store.get_log.return_value = [
            {
                "id": "row-1",
                "action": "permission_granted",
                "timestamp": event.timestamp.isoformat(),
                "user_id": "user-1",
                "org_id": None,
                "resource_type": None,
                "resource_id": None,
                "ip_address": None,
                "user_agent": None,
                "metadata": {
                    "event_id": "evt-1",
                    "actor_id": None,
                    "permission_key": None,
                    "decision": True,
                    "reason": "",
                    "request_id": None,
                    "signature": signed_data["signature"],
                },
            }
        ]
        handler = PersistentAuditHandler(store=mock_store)

        events = handler.get_events(verify_signatures=True)

        assert len(events) == 1
        assert events[0].metadata.get("_signature_valid") is True

    def test_get_events_marks_invalid_signature(self, mock_store, signing_key):
        """Test get_events marks events with invalid signatures."""
        mock_store.get_log.return_value = [
            {
                "id": "row-1",
                "action": "permission_granted",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": "user-1",
                "org_id": None,
                "resource_type": None,
                "resource_id": None,
                "ip_address": None,
                "user_agent": None,
                "metadata": {
                    "event_id": "evt-1",
                    "actor_id": None,
                    "permission_key": None,
                    "decision": True,
                    "reason": "",
                    "request_id": None,
                    "signature": "deadbeef" * 8,  # Invalid 64-char hex sig
                },
            }
        ]
        handler = PersistentAuditHandler(store=mock_store)

        events = handler.get_events(verify_signatures=True)

        assert len(events) == 1
        assert events[0].metadata.get("_signature_valid") is False

    def test_get_break_glass_events(self, mock_store):
        """Test get_break_glass_events queries all three break-glass types."""
        mock_store.get_log.return_value = []
        handler = PersistentAuditHandler(store=mock_store)

        result = handler.get_break_glass_events(limit=50)

        # Should have queried 3 times (activated, deactivated, action)
        assert mock_store.get_log.call_count == 3
        assert result == []

    def test_get_event_count(self, mock_store):
        """Test get_event_count delegates to store."""
        mock_store.get_log_count.return_value = 42
        handler = PersistentAuditHandler(store=mock_store)

        count = handler.get_event_count(user_id="user-1")

        assert count == 42
        mock_store.get_log_count.assert_called_once()

    def test_get_event_count_with_enum_type(self, mock_store):
        """Test get_event_count converts AuditEventType to string action."""
        mock_store.get_log_count.return_value = 5
        handler = PersistentAuditHandler(store=mock_store)

        handler.get_event_count(event_type=AuditEventType.PERMISSION_DENIED)

        call_kwargs = mock_store.get_log_count.call_args[1]
        assert call_kwargs["action"] == "permission_denied"

    def test_get_stats(self, mock_store, signing_key):
        """Test get_stats returns correct statistics."""
        handler = PersistentAuditHandler(store=mock_store, batch_size=100)

        # Write some events
        handler.handle_event(AuditEvent(user_id="user-1"))
        handler.handle_event(AuditEvent(user_id="user-2"))

        stats = handler.get_stats()

        assert stats["events_written"] == 0  # Not flushed yet
        assert stats["pending_count"] == 2
        assert stats["sign_events"] is True
        assert stats["batch_size"] == 100

    def test_close_flushes_pending(self, mock_store, signing_key):
        """Test close() flushes pending events."""
        handler = PersistentAuditHandler(store=mock_store, batch_size=1000)

        handler.handle_event(AuditEvent(user_id="user-1"))
        handler.close()

        assert mock_store.log_event.call_count == 1
        assert handler._events_written == 1

    def test_store_lazy_initialization(self):
        """Test store is lazily initialized via get_audit_store."""
        handler = PersistentAuditHandler(store=None)

        with patch(
            "aragora.rbac.audit.PersistentAuditHandler.store",
            new_callable=lambda: property(lambda self: MagicMock()),
        ):
            # Just verify no error accessing store when _store is None
            pass
        # The _store is None initially
        assert handler._store is None


# ============================================================================
# Module-level Singleton Tests
# ============================================================================


class TestModuleSingletons:
    """Tests for module-level persistent handler singletons."""

    def test_get_persistent_handler_returns_instance(self):
        """Test get_persistent_handler returns PersistentAuditHandler."""
        with patch("aragora.rbac.audit.PersistentAuditHandler.__init__", return_value=None):
            handler = get_persistent_handler()
            assert handler is not None

    def test_set_persistent_handler_replaces(self):
        """Test set_persistent_handler replaces the global instance."""
        mock_handler = MagicMock(spec=PersistentAuditHandler)
        set_persistent_handler(mock_handler)

        assert audit_module._persistent_handler is mock_handler

    def test_enable_persistent_auditing_attaches_handler(self):
        """Test enable_persistent_auditing connects handler to auditor."""
        mock_handler = MagicMock(spec=PersistentAuditHandler)
        set_persistent_handler(mock_handler)

        result = enable_persistent_auditing()

        assert result is mock_handler
        auditor = get_auditor()
        assert mock_handler.handle_event in auditor._handlers
