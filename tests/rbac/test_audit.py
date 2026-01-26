"""
Tests for RBAC audit logging module.

Tests cover:
- AuditEvent creation and serialization
- AuditEventType enum values
- AuthorizationAuditor event handling
- Event filtering (denied only, cached)
- Handler management (add/remove)
- Event buffer management
- Global auditor instance
- Convenience functions
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from aragora.rbac.audit import (
    AuditEvent,
    AuditEventType,
    AuthorizationAuditor,
    get_auditor,
    log_permission_check,
    set_auditor,
)
from aragora.rbac.models import AuthorizationContext, AuthorizationDecision, RoleAssignment


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

    def test_event_type_values_are_strings(self):
        """Test event type values are lowercase strings."""
        assert AuditEventType.PERMISSION_GRANTED.value == "permission_granted"
        assert AuditEventType.ROLE_ASSIGNED.value == "role_assigned"


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
        before = datetime.utcnow()
        event = AuditEvent()
        after = datetime.utcnow()

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

    def test_to_dict_timestamp_is_iso_format(self):
        """Test timestamp is serialized as ISO format."""
        event = AuditEvent()
        data = event.to_dict()

        # Should be parseable as ISO format
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


# ============================================================================
# AuthorizationAuditor Tests
# ============================================================================


class TestAuthorizationAuditor:
    """Tests for AuthorizationAuditor class."""

    def test_auditor_initialization_default(self):
        """Test auditor initializes with defaults."""
        auditor = AuthorizationAuditor()

        # Should have default log handler
        assert len(auditor._handlers) >= 1
        assert not auditor._log_denied_only
        assert not auditor._include_cached

    def test_auditor_with_custom_handlers(self):
        """Test auditor with custom handlers."""
        handler1 = MagicMock()
        handler2 = MagicMock()

        auditor = AuthorizationAuditor(handlers=[handler1, handler2])

        # Should have custom handlers + default
        assert handler1 in auditor._handlers
        assert handler2 in auditor._handlers

    def test_auditor_log_denied_only_option(self):
        """Test log_denied_only filtering."""
        handler = MagicMock()
        auditor = AuthorizationAuditor(handlers=[handler], log_denied_only=True)

        # Create allowed decision
        context = AuthorizationContext(user_id="user-1", org_id="org-1")
        allowed_decision = AuthorizationDecision(
            permission_key="test:read",
            allowed=True,
            reason="Permitted",
            context=context,
        )

        # Create denied decision
        denied_decision = AuthorizationDecision(
            permission_key="test:delete",
            allowed=False,
            reason="Denied",
            context=context,
        )

        auditor.log_decision(allowed_decision)
        auditor.log_decision(denied_decision)

        # Should only log denied (once for denied)
        # Note: default handler is also present, so we check custom handler calls
        call_count = sum(
            1
            for call in handler.call_args_list
            if call[0][0].event_type == AuditEventType.PERMISSION_DENIED
        )
        assert call_count == 1

    def test_auditor_include_cached_option(self):
        """Test include_cached filtering."""
        handler = MagicMock()
        auditor = AuthorizationAuditor(handlers=[handler], include_cached=False)

        context = AuthorizationContext(user_id="user-1", org_id="org-1")

        # Cached decision
        cached_decision = AuthorizationDecision(
            permission_key="test:read",
            allowed=True,
            reason="From cache",
            context=context,
            cached=True,
        )

        # Non-cached decision
        fresh_decision = AuthorizationDecision(
            permission_key="test:write",
            allowed=True,
            reason="Fresh check",
            context=context,
            cached=False,
        )

        auditor.log_decision(cached_decision)
        auditor.log_decision(fresh_decision)

        # Cached should be filtered out
        events = [call[0][0] for call in handler.call_args_list]
        assert not any(e.permission_key == "test:read" for e in events)
        assert any(e.permission_key == "test:write" for e in events)

    def test_log_decision_creates_correct_event_type(self):
        """Test log_decision creates correct event types."""
        handler = MagicMock()
        auditor = AuthorizationAuditor(handlers=[handler])
        context = AuthorizationContext(user_id="user-1")

        # Allowed decision
        auditor.log_decision(
            AuthorizationDecision(
                permission_key="test:read",
                allowed=True,
                reason="User has permission",
                context=context,
            )
        )

        # Denied decision
        auditor.log_decision(
            AuthorizationDecision(
                permission_key="test:write",
                allowed=False,
                reason="Permission denied",
                context=context,
            )
        )

        events = [call[0][0] for call in handler.call_args_list]

        granted = [e for e in events if e.event_type == AuditEventType.PERMISSION_GRANTED]
        denied = [e for e in events if e.event_type == AuditEventType.PERMISSION_DENIED]

        assert len(granted) >= 1
        assert len(denied) >= 1

    def test_log_role_assignment(self):
        """Test logging role assignment."""
        handler = MagicMock()
        auditor = AuthorizationAuditor(handlers=[handler])

        assignment = RoleAssignment(
            id="assignment-1",
            user_id="user-1",
            role_id="admin",
            org_id="org-1",
        )

        auditor.log_role_assignment(
            assignment=assignment,
            actor_id="admin-1",
            ip_address="10.0.0.1",
        )

        # Find the role assigned event
        events = [call[0][0] for call in handler.call_args_list]
        role_events = [e for e in events if e.event_type == AuditEventType.ROLE_ASSIGNED]

        assert len(role_events) >= 1
        event = role_events[0]
        assert event.user_id == "user-1"
        assert event.resource_id == "admin"
        assert event.actor_id == "admin-1"

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

        events = [call[0][0] for call in handler.call_args_list]
        revoke_events = [e for e in events if e.event_type == AuditEventType.ROLE_REVOKED]

        assert len(revoke_events) >= 1
        event = revoke_events[0]
        assert event.user_id == "user-1"
        assert event.resource_id == "admin"
        assert "Policy violation" in event.reason

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

        events = [call[0][0] for call in handler.call_args_list]
        key_events = [e for e in events if e.event_type == AuditEventType.API_KEY_CREATED]

        assert len(key_events) >= 1
        event = key_events[0]
        assert event.user_id == "user-1"
        assert event.resource_id == "key-abc"
        assert "scopes" in event.metadata

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

        events = [call[0][0] for call in handler.call_args_list]
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

        events = [call[0][0] for call in handler.call_args_list]
        imp_events = [e for e in events if e.event_type == AuditEventType.IMPERSONATION_START]

        assert len(imp_events) >= 1
        event = imp_events[0]
        assert event.actor_id == "admin-1"
        assert event.user_id == "user-1"
        assert "ticket" in event.reason.lower()

    def test_log_impersonation_end(self):
        """Test logging impersonation end."""
        handler = MagicMock()
        auditor = AuthorizationAuditor(handlers=[handler])

        auditor.log_impersonation_end(
            actor_id="admin-1",
            target_user_id="user-1",
            org_id="org-1",
        )

        events = [call[0][0] for call in handler.call_args_list]
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

        events = [call[0][0] for call in handler.call_args_list]
        session_events = [e for e in events if e.event_type == AuditEventType.SESSION_CREATED]

        assert len(session_events) >= 1
        event = session_events[0]
        assert event.resource_id == "session-xyz"
        assert event.user_agent == "Chrome/120.0"


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

        # Should not raise
        auditor.remove_handler(nonexistent)


# ============================================================================
# Event Buffer Tests
# ============================================================================


class TestAuditorEventBuffer:
    """Tests for auditor event buffer."""

    def test_buffer_stores_events(self):
        """Test events are buffered."""
        auditor = AuthorizationAuditor()
        context = AuthorizationContext(user_id="user-1")

        auditor.log_decision(
            AuthorizationDecision(
                permission_key="test:read",
                allowed=True,
                reason="Permitted",
                context=context,
            )
        )

        assert len(auditor._event_buffer) >= 1

    def test_flush_buffer_returns_events(self):
        """Test flush_buffer returns and clears events."""
        auditor = AuthorizationAuditor()
        context = AuthorizationContext(user_id="user-1")

        auditor.log_decision(
            AuthorizationDecision(
                permission_key="test:read",
                allowed=True,
                reason="Permitted",
                context=context,
            )
        )

        events = auditor.flush_buffer()

        assert len(events) >= 1
        assert len(auditor._event_buffer) == 0

    def test_buffer_truncated_at_max_size(self):
        """Test buffer is truncated at max size."""
        auditor = AuthorizationAuditor()
        auditor._buffer_size = 5  # Small buffer for testing
        context = AuthorizationContext(user_id="user-1")

        # Add more than buffer size
        for i in range(10):
            auditor.log_decision(
                AuthorizationDecision(
                    permission_key=f"test:perm_{i}",
                    allowed=True,
                    reason=f"Permitted {i}",
                    context=context,
                )
            )

        assert len(auditor._event_buffer) <= 5


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestAuditorErrorHandling:
    """Tests for auditor error handling."""

    def test_handler_error_doesnt_stop_other_handlers(self):
        """Test handler errors don't prevent other handlers from running."""
        failing_handler = MagicMock(side_effect=Exception("Handler failed"))
        working_handler = MagicMock()

        auditor = AuthorizationAuditor(handlers=[failing_handler, working_handler])
        context = AuthorizationContext(user_id="user-1")

        # Should not raise
        auditor.log_decision(
            AuthorizationDecision(
                permission_key="test:read",
                allowed=True,
                reason="Permitted",
                context=context,
            )
        )

        # Working handler should still be called
        assert working_handler.called

    def test_handler_error_logged(self):
        """Test handler errors are logged."""
        failing_handler = MagicMock(side_effect=Exception("Handler failed"))
        auditor = AuthorizationAuditor(handlers=[failing_handler])
        context = AuthorizationContext(user_id="user-1")

        with patch("aragora.rbac.audit.logger") as mock_logger:
            auditor.log_decision(
                AuthorizationDecision(
                    permission_key="test:read",
                    allowed=True,
                    reason="Permitted",
                    context=context,
                )
            )

            # Should log error
            assert mock_logger.error.called


# ============================================================================
# Global Auditor Tests
# ============================================================================


class TestGlobalAuditor:
    """Tests for global auditor instance."""

    def test_get_auditor_returns_instance(self):
        """Test get_auditor returns an instance."""
        # Reset global state
        import aragora.rbac.audit as audit_module

        audit_module._auditor = None

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

        events = [call[0][0] for call in handler.call_args_list]
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

        events = [call[0][0] for call in handler.call_args_list]
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

    def test_default_handler_logs_granted_as_info(self):
        """Test granted events logged at INFO level."""
        auditor = AuthorizationAuditor()
        context = AuthorizationContext(user_id="user-1", org_id="org-1")

        with patch("aragora.rbac.audit.logger") as mock_logger:
            auditor.log_decision(
                AuthorizationDecision(
                    permission_key="test:read",
                    allowed=True,
                    reason="Permitted",
                    context=context,
                )
            )

            # Should log at INFO
            mock_logger.log.assert_called()
            call_args = mock_logger.log.call_args
            assert call_args[0][0] == logging.INFO

    def test_default_handler_logs_denied_as_warning(self):
        """Test denied events logged at WARNING level."""
        auditor = AuthorizationAuditor()
        context = AuthorizationContext(user_id="user-1", org_id="org-1")

        with patch("aragora.rbac.audit.logger") as mock_logger:
            auditor.log_decision(
                AuthorizationDecision(
                    permission_key="test:delete",
                    allowed=False,
                    reason="Denied",
                    context=context,
                )
            )

            # Should log at WARNING
            mock_logger.log.assert_called()
            call_args = mock_logger.log.call_args
            assert call_args[0][0] == logging.WARNING

    def test_default_handler_includes_audit_event_extra(self):
        """Test default handler includes audit_event in log extra."""
        auditor = AuthorizationAuditor()
        context = AuthorizationContext(user_id="user-1", org_id="org-1")

        with patch("aragora.rbac.audit.logger") as mock_logger:
            auditor.log_decision(
                AuthorizationDecision(
                    permission_key="test:read",
                    allowed=True,
                    reason="Permitted",
                    context=context,
                )
            )

            call_kwargs = mock_logger.log.call_args[1]
            assert "extra" in call_kwargs
            assert "audit_event" in call_kwargs["extra"]
