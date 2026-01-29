"""
Tests for RBAC Audit Persistence.

Tests cover:
- Event signing with HMAC-SHA256
- Signature verification
- PersistentAuditHandler functionality
- Event persistence and retrieval
- Break-glass event tracking
- Tamper detection
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    set_audit_signing_key,
    set_auditor,
    set_persistent_handler,
    verify_event_signature,
)


# ============================================================================
# Event Signing Tests
# ============================================================================


class TestEventSigning:
    """Tests for HMAC-SHA256 event signing."""

    def setup_method(self):
        """Set up test signing key."""
        # Use a fixed 32-byte key for reproducible tests
        self.test_key = b"test_key_32_bytes_for_hmac_sign!"  # 32 bytes exactly
        set_audit_signing_key(self.test_key)

    def test_compute_signature_returns_hex_string(self):
        """Test signature is hex-encoded string."""
        event = AuditEvent(
            user_id="user-123",
            event_type=AuditEventType.PERMISSION_GRANTED,
        )
        signature = compute_event_signature(event.to_dict())

        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA256 produces 64 hex chars
        # Verify it's valid hex
        int(signature, 16)

    def test_signature_is_deterministic(self):
        """Test same event produces same signature."""
        event = AuditEvent(
            id="fixed-id",
            user_id="user-123",
            event_type=AuditEventType.PERMISSION_GRANTED,
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )

        sig1 = compute_event_signature(event.to_dict())
        sig2 = compute_event_signature(event.to_dict())

        assert sig1 == sig2

    def test_different_events_have_different_signatures(self):
        """Test different events produce different signatures."""
        event1 = AuditEvent(
            id="event-1",
            user_id="user-123",
        )
        event2 = AuditEvent(
            id="event-2",
            user_id="user-456",
        )

        sig1 = compute_event_signature(event1.to_dict())
        sig2 = compute_event_signature(event2.to_dict())

        assert sig1 != sig2

    def test_verify_valid_signature(self):
        """Test verification of valid signature."""
        event = AuditEvent(
            user_id="user-123",
            event_type=AuditEventType.PERMISSION_DENIED,
        )
        event_data = event.to_dict()
        signature = compute_event_signature(event_data)

        assert verify_event_signature(event_data, signature) is True

    def test_verify_invalid_signature(self):
        """Test verification fails with invalid signature."""
        event = AuditEvent(
            user_id="user-123",
        )
        event_data = event.to_dict()

        # Use wrong signature
        assert verify_event_signature(event_data, "invalid_signature") is False

    def test_verify_detects_tampering(self):
        """Test signature verification detects data tampering."""
        event = AuditEvent(
            user_id="user-123",
            decision=True,
        )
        event_data = event.to_dict()
        signature = compute_event_signature(event_data)

        # Tamper with the data
        event_data["decision"] = False

        assert verify_event_signature(event_data, signature) is False

    def test_to_signed_dict_includes_signature(self):
        """Test to_signed_dict includes signature field."""
        event = AuditEvent(
            user_id="user-123",
        )
        signed_data = event.to_signed_dict()

        assert "signature" in signed_data
        assert event.signature == signed_data["signature"]

    def test_event_verify_signature_method(self):
        """Test AuditEvent.verify_signature() method."""
        event = AuditEvent(
            user_id="user-123",
        )
        event.to_signed_dict()  # This sets the signature

        assert event.verify_signature() is True

    def test_event_without_signature_fails_verification(self):
        """Test event without signature fails verification."""
        event = AuditEvent(
            user_id="user-123",
        )
        # Don't sign it
        assert event.verify_signature() is False


# ============================================================================
# AuditEvent.from_dict Tests
# ============================================================================


class TestAuditEventFromDict:
    """Tests for AuditEvent.from_dict() reconstruction."""

    def test_from_dict_basic(self):
        """Test basic reconstruction from dict."""
        original = AuditEvent(
            id="test-id",
            user_id="user-123",
            org_id="org-456",
            event_type=AuditEventType.PERMISSION_DENIED,
            decision=False,
            reason="Insufficient permissions",
        )
        original.to_signed_dict()  # Sign it

        reconstructed = AuditEvent.from_dict(original.to_signed_dict())

        assert reconstructed.id == original.id
        assert reconstructed.user_id == original.user_id
        assert reconstructed.org_id == original.org_id
        assert reconstructed.event_type == original.event_type
        assert reconstructed.decision == original.decision
        assert reconstructed.reason == original.reason
        assert reconstructed.signature == original.signature

    def test_from_dict_preserves_signature(self):
        """Test signature is preserved through serialization."""
        event = AuditEvent(user_id="user-123")
        signed_data = event.to_signed_dict()

        reconstructed = AuditEvent.from_dict(signed_data)
        assert reconstructed.verify_signature() is True

    def test_from_dict_handles_iso_timestamp(self):
        """Test ISO timestamp parsing."""
        data = {
            "id": "test-id",
            "event_type": "permission_granted",
            "timestamp": "2024-01-15T10:30:00+00:00",
            "user_id": "user-123",
        }
        event = AuditEvent.from_dict(data)

        assert event.timestamp.year == 2024
        assert event.timestamp.month == 1
        assert event.timestamp.day == 15

    def test_from_dict_handles_unknown_event_type(self):
        """Test unknown event type falls back to CUSTOM."""
        data = {
            "id": "test-id",
            "event_type": "unknown_event_type",
            "user_id": "user-123",
        }
        event = AuditEvent.from_dict(data)

        assert event.event_type == AuditEventType.CUSTOM


# ============================================================================
# PersistentAuditHandler Tests
# ============================================================================


class TestPersistentAuditHandler:
    """Tests for PersistentAuditHandler."""

    def setup_method(self):
        """Set up test fixtures."""
        # Use a mock store to avoid database dependencies
        self.mock_store = MagicMock()
        self.mock_store.log_event = MagicMock()
        self.mock_store.get_log = MagicMock(return_value=[])
        self.mock_store.get_log_count = MagicMock(return_value=0)

        self.handler = PersistentAuditHandler(
            store=self.mock_store,
            sign_events=True,
            batch_size=10,
        )

        # Reset global handler
        set_persistent_handler(None)

    def test_handle_event_adds_to_batch(self):
        """Test events are added to batch."""
        event = AuditEvent(user_id="user-123")
        self.handler.handle_event(event)

        # Event should be in batch, not yet flushed
        stats = self.handler.get_stats()
        assert stats["pending_count"] == 1
        assert stats["events_written"] == 0

    def test_batch_flushes_at_size_limit(self):
        """Test batch flushes when size limit is reached."""
        # Add events up to batch size
        for i in range(10):
            event = AuditEvent(id=f"event-{i}", user_id=f"user-{i}")
            self.handler.handle_event(event)

        # Batch should have flushed
        assert self.mock_store.log_event.call_count == 10

        stats = self.handler.get_stats()
        assert stats["events_written"] == 10
        assert stats["pending_count"] == 0

    def test_flush_writes_pending_events(self):
        """Test manual flush writes pending events."""
        event = AuditEvent(user_id="user-123")
        self.handler.handle_event(event)

        assert self.mock_store.log_event.call_count == 0

        self.handler.flush()

        assert self.mock_store.log_event.call_count == 1

    def test_events_are_signed_when_enabled(self):
        """Test events are signed before storage."""
        event = AuditEvent(user_id="user-123")
        self.handler.handle_event(event)
        self.handler.flush()

        # Check that log_event was called with signature in metadata
        call_args = self.mock_store.log_event.call_args
        metadata = call_args.kwargs.get("metadata", {})
        assert "signature" in metadata
        assert metadata["signature"] is not None

    def test_events_not_signed_when_disabled(self):
        """Test events are not signed when sign_events=False."""
        handler = PersistentAuditHandler(
            store=self.mock_store,
            sign_events=False,
        )

        event = AuditEvent(user_id="user-123")
        handler.handle_event(event)
        handler.flush()

        # Check that signature is None
        call_args = self.mock_store.log_event.call_args
        metadata = call_args.kwargs.get("metadata", {})
        assert metadata.get("signature") is None

    def test_get_events_returns_audit_events(self):
        """Test get_events returns AuditEvent instances."""
        # Set up mock to return event data
        self.mock_store.get_log.return_value = [
            {
                "id": 1,
                "timestamp": "2024-01-15T10:00:00+00:00",
                "user_id": "user-123",
                "org_id": None,
                "action": "permission_granted",
                "resource_type": "authorization",
                "resource_id": None,
                "ip_address": "192.168.1.1",
                "user_agent": "test-agent",
                "metadata": {
                    "event_id": "event-123",
                    "decision": True,
                    "reason": "Allowed",
                },
            }
        ]

        events = self.handler.get_events(user_id="user-123")

        assert len(events) == 1
        assert isinstance(events[0], AuditEvent)
        assert events[0].user_id == "user-123"
        assert events[0].event_type == AuditEventType.PERMISSION_GRANTED

    def test_get_break_glass_events(self):
        """Test retrieval of break-glass events."""
        # Set up mock to return break-glass events
        self.mock_store.get_log.return_value = [
            {
                "id": 1,
                "timestamp": "2024-01-15T10:00:00+00:00",
                "user_id": "admin-123",
                "action": "break_glass_activated",
                "resource_type": "authorization",
                "metadata": {
                    "event_id": "bg-event-1",
                    "decision": True,
                    "reason": "Emergency access",
                },
            }
        ]

        events = self.handler.get_break_glass_events(limit=10)

        # Should call get_log for each break-glass event type
        assert self.mock_store.get_log.call_count == 3

    def test_get_stats(self):
        """Test statistics retrieval."""
        stats = self.handler.get_stats()

        assert "events_written" in stats
        assert "events_failed" in stats
        assert "pending_count" in stats
        assert "sign_events" in stats
        assert "batch_size" in stats

    def test_close_flushes_pending(self):
        """Test close flushes pending events."""
        event = AuditEvent(user_id="user-123")
        self.handler.handle_event(event)

        self.handler.close()

        assert self.mock_store.log_event.call_count == 1


# ============================================================================
# Integration Tests
# ============================================================================


class TestAuditPersistenceIntegration:
    """Integration tests for audit persistence."""

    def setup_method(self):
        """Reset global state."""
        set_auditor(None)
        set_persistent_handler(None)

    def test_enable_persistent_auditing(self):
        """Test enable_persistent_auditing attaches handler."""
        mock_store = MagicMock()
        mock_store.log_event = MagicMock()

        with patch(
            "aragora.storage.audit_store.get_audit_store",
            return_value=mock_store,
        ):
            handler = enable_persistent_auditing()

        # Should return the handler
        assert handler is not None
        assert isinstance(handler, PersistentAuditHandler)

        # Auditor should have the handler attached
        auditor = get_auditor()
        assert handler.handle_event in auditor._handlers

    def test_end_to_end_audit_flow(self):
        """Test complete audit flow from event to storage."""
        mock_store = MagicMock()
        mock_store.log_event = MagicMock()
        mock_store.get_log = MagicMock(return_value=[])

        handler = PersistentAuditHandler(store=mock_store, batch_size=1)
        auditor = AuthorizationAuditor()
        auditor.add_handler(handler.handle_event)

        # Log a permission check
        from aragora.rbac.models import AuthorizationContext, AuthorizationDecision

        context = AuthorizationContext(
            user_id="user-123",
            org_id="org-456",
            ip_address="10.0.0.1",
        )
        decision = AuthorizationDecision(
            allowed=True,
            permission_key="debates.create",
            reason="User has permission",
            context=context,
        )

        auditor.log_decision(decision)

        # Event should be persisted
        assert mock_store.log_event.call_count == 1

        call_args = mock_store.log_event.call_args
        assert call_args.kwargs["user_id"] == "user-123"
        assert call_args.kwargs["org_id"] == "org-456"
        assert call_args.kwargs["action"] == "permission_granted"


# ============================================================================
# Signing Key Management Tests
# ============================================================================


class TestSigningKeyManagement:
    """Tests for signing key management."""

    def test_set_signing_key(self):
        """Test setting a custom signing key."""
        key = b"custom_key_with_32_bytes_minimum"
        set_audit_signing_key(key)

        retrieved = get_audit_signing_key()
        assert retrieved == key

    def test_signing_key_minimum_length(self):
        """Test signing key requires minimum length."""
        with pytest.raises(ValueError, match="at least 32 bytes"):
            set_audit_signing_key(b"short_key")

    def test_signing_key_from_environment(self):
        """Test signing key loaded from environment."""
        test_key = b"environment_key_with_32_bytes_ok"
        key_hex = test_key.hex()

        # Reset the key first
        import aragora.rbac.audit as audit_module

        audit_module._AUDIT_SIGNING_KEY = None

        with patch.dict(os.environ, {"ARAGORA_AUDIT_SIGNING_KEY": key_hex}):
            key = get_audit_signing_key()
            assert key == test_key


# ============================================================================
# Break-Glass Event Tests
# ============================================================================


class TestBreakGlassAuditing:
    """Tests for break-glass event auditing."""

    def test_break_glass_event_types_exist(self):
        """Test break-glass event types are defined."""
        assert AuditEventType.BREAK_GLASS_ACTIVATED
        assert AuditEventType.BREAK_GLASS_DEACTIVATED
        assert AuditEventType.BREAK_GLASS_ACTION

    def test_break_glass_event_signature_integrity(self):
        """Test break-glass events maintain signature integrity."""
        event = AuditEvent(
            event_type=AuditEventType.BREAK_GLASS_ACTIVATED,
            user_id="admin-123",
            actor_id="admin-123",
            reason="Emergency production access",
            metadata={
                "justification": "Critical outage",
                "ticket_id": "INC-12345",
            },
        )

        # Sign the event
        signed_data = event.to_signed_dict()

        # Verify signature
        assert verify_event_signature(
            {k: v for k, v in signed_data.items() if k != "signature"},
            signed_data["signature"],
        )

        # Reconstruct and verify
        reconstructed = AuditEvent.from_dict(signed_data)
        assert reconstructed.verify_signature() is True

    def test_break_glass_metadata_preserved(self):
        """Test break-glass metadata is preserved through persistence."""
        mock_store = MagicMock()
        mock_store.log_event = MagicMock()

        handler = PersistentAuditHandler(store=mock_store, batch_size=1)

        event = AuditEvent(
            event_type=AuditEventType.BREAK_GLASS_ACTIVATED,
            user_id="admin-123",
            metadata={
                "justification": "Critical outage",
                "ticket_id": "INC-12345",
                "approver_id": "manager-456",
            },
        )

        handler.handle_event(event)

        # Check metadata was passed to store
        call_args = mock_store.log_event.call_args
        metadata = call_args.kwargs["metadata"]

        assert "justification" in metadata
        assert metadata["justification"] == "Critical outage"
        assert metadata["ticket_id"] == "INC-12345"
