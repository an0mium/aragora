"""
Tests for aragora.server.middleware.audit_logger - Audit logging middleware.

Tests cover:
- AuditSeverity and AuditCategory enums
- AuditEvent dataclass
- MemoryAuditBackend
- FileAuditBackend (basic tests)
- AuditLogger class
- Global logger functions
- Context management
- audit_event() convenience function
- audit_action() decorator
- Pre-defined audit helpers
- Chain integrity verification
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest


# ===========================================================================
# Test AuditSeverity and AuditCategory Enums
# ===========================================================================


class TestAuditSeverity:
    """Tests for AuditSeverity enum."""

    def test_severity_levels_defined(self):
        """All severity levels should be defined."""
        from aragora.server.middleware.audit_logger import AuditSeverity

        assert AuditSeverity.DEBUG.value == "debug"
        assert AuditSeverity.INFO.value == "info"
        assert AuditSeverity.WARNING.value == "warning"
        assert AuditSeverity.ERROR.value == "error"
        assert AuditSeverity.CRITICAL.value == "critical"

    def test_severity_count(self):
        """Should have 5 severity levels."""
        from aragora.server.middleware.audit_logger import AuditSeverity

        assert len(AuditSeverity) == 5


class TestAuditCategory:
    """Tests for AuditCategory enum."""

    def test_categories_defined(self):
        """All audit categories should be defined."""
        from aragora.server.middleware.audit_logger import AuditCategory

        assert AuditCategory.AUTHENTICATION.value == "authentication"
        assert AuditCategory.AUTHORIZATION.value == "authorization"
        assert AuditCategory.DATA_ACCESS.value == "data_access"
        assert AuditCategory.DATA_MODIFICATION.value == "data_modification"
        assert AuditCategory.CONFIGURATION.value == "configuration"
        assert AuditCategory.ADMINISTRATIVE.value == "administrative"
        assert AuditCategory.SECURITY.value == "security"
        assert AuditCategory.SYSTEM.value == "system"

    def test_category_count(self):
        """Should have 8 categories."""
        from aragora.server.middleware.audit_logger import AuditCategory

        assert len(AuditCategory) == 8


# ===========================================================================
# Test AuditEvent Dataclass
# ===========================================================================


class TestAuditEvent:
    """Tests for AuditEvent dataclass."""

    def test_create_minimal_event(self):
        """Should create event with minimal required fields."""
        from aragora.server.middleware.audit_logger import AuditEvent

        event = AuditEvent(
            event_id="test-123",
            timestamp=datetime.now(timezone.utc),
            actor="user@example.com",
            action="test.action",
        )

        assert event.event_id == "test-123"
        assert event.actor == "user@example.com"
        assert event.action == "test.action"

    def test_default_values(self):
        """Should have sensible default values."""
        from aragora.server.middleware.audit_logger import (
            AuditCategory,
            AuditEvent,
            AuditSeverity,
        )

        event = AuditEvent(
            event_id="test",
            timestamp=datetime.now(timezone.utc),
            actor="test",
            action="test",
        )

        assert event.actor_type == "user"
        assert event.category == AuditCategory.SYSTEM
        assert event.severity == AuditSeverity.INFO
        assert event.outcome == "success"
        assert event.details == {}

    def test_to_dict(self):
        """to_dict() should serialize all fields."""
        from aragora.server.middleware.audit_logger import (
            AuditCategory,
            AuditEvent,
            AuditSeverity,
        )

        timestamp = datetime.now(timezone.utc)
        event = AuditEvent(
            event_id="event-123",
            timestamp=timestamp,
            actor="admin@example.com",
            action="user.create",
            resource="users",
            resource_id="user-456",
            outcome="success",
            category=AuditCategory.DATA_MODIFICATION,
            severity=AuditSeverity.INFO,
        )

        d = event.to_dict()

        assert d["event_id"] == "event-123"
        assert d["timestamp"] == timestamp.isoformat()
        assert d["actor"] == "admin@example.com"
        assert d["action"] == "user.create"
        assert d["resource"] == "users"
        assert d["resource_id"] == "user-456"
        assert d["category"] == "data_modification"
        assert d["severity"] == "info"

    def test_compute_hash_deterministic(self):
        """compute_hash() should be deterministic."""
        from aragora.server.middleware.audit_logger import AuditEvent

        timestamp = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        event = AuditEvent(
            event_id="test-hash",
            timestamp=timestamp,
            actor="tester",
            action="test.hash",
            resource="hash-test",
            outcome="success",
        )

        hash1 = event.compute_hash()
        hash2 = event.compute_hash()

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex

    def test_compute_hash_includes_previous_hash(self):
        """compute_hash() should include previous_hash for chaining."""
        from aragora.server.middleware.audit_logger import AuditEvent

        timestamp = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        event1 = AuditEvent(
            event_id="test",
            timestamp=timestamp,
            actor="test",
            action="test",
        )
        event1.previous_hash = None

        event2 = AuditEvent(
            event_id="test",
            timestamp=timestamp,
            actor="test",
            action="test",
        )
        event2.previous_hash = "abc123"

        # Different previous_hash should produce different hashes
        assert event1.compute_hash() != event2.compute_hash()


# ===========================================================================
# Test MemoryAuditBackend
# ===========================================================================


class TestMemoryAuditBackend:
    """Tests for MemoryAuditBackend."""

    def test_write_event(self):
        """Should write events to memory."""
        from aragora.server.middleware.audit_logger import AuditEvent, MemoryAuditBackend

        backend = MemoryAuditBackend()
        event = AuditEvent(
            event_id="mem-test",
            timestamp=datetime.now(timezone.utc),
            actor="tester",
            action="test.write",
        )

        backend.write(event)
        events = backend.query(limit=10)

        assert len(events) == 1
        assert events[0].event_id == "mem-test"

    def test_sets_hash_chain(self):
        """Should set previous_hash and event_hash."""
        from aragora.server.middleware.audit_logger import AuditEvent, MemoryAuditBackend

        backend = MemoryAuditBackend()

        event1 = AuditEvent(
            event_id="event-1",
            timestamp=datetime.now(timezone.utc),
            actor="tester",
            action="test.1",
        )
        backend.write(event1)

        event2 = AuditEvent(
            event_id="event-2",
            timestamp=datetime.now(timezone.utc),
            actor="tester",
            action="test.2",
        )
        backend.write(event2)

        # First event has no previous hash
        assert event1.previous_hash is None
        assert event1.event_hash is not None

        # Second event chains to first
        assert event2.previous_hash == event1.event_hash
        assert event2.event_hash is not None

    def test_max_events_limit(self):
        """Should trim events when exceeding max."""
        from aragora.server.middleware.audit_logger import AuditEvent, MemoryAuditBackend

        backend = MemoryAuditBackend(max_events=5)

        for i in range(10):
            event = AuditEvent(
                event_id=f"event-{i}",
                timestamp=datetime.now(timezone.utc),
                actor="tester",
                action=f"test.{i}",
            )
            backend.write(event)

        events = backend.query(limit=100)
        assert len(events) == 5
        # Should keep most recent
        assert events[-1].event_id == "event-9"

    def test_query_by_actor(self):
        """Should filter by actor."""
        from aragora.server.middleware.audit_logger import AuditEvent, MemoryAuditBackend

        backend = MemoryAuditBackend()

        for actor in ["alice", "bob", "alice"]:
            event = AuditEvent(
                event_id=f"event-{actor}",
                timestamp=datetime.now(timezone.utc),
                actor=actor,
                action="test",
            )
            backend.write(event)

        alice_events = backend.query(actor="alice", limit=100)
        assert len(alice_events) == 2

    def test_query_by_action(self):
        """Should filter by action."""
        from aragora.server.middleware.audit_logger import AuditEvent, MemoryAuditBackend

        backend = MemoryAuditBackend()

        for action in ["user.create", "user.update", "user.create"]:
            event = AuditEvent(
                event_id=f"event-{action}",
                timestamp=datetime.now(timezone.utc),
                actor="tester",
                action=action,
            )
            backend.write(event)

        create_events = backend.query(action="user.create", limit=100)
        assert len(create_events) == 2

    def test_query_with_limit(self):
        """Should respect query limit."""
        from aragora.server.middleware.audit_logger import AuditEvent, MemoryAuditBackend

        backend = MemoryAuditBackend()

        for i in range(10):
            event = AuditEvent(
                event_id=f"event-{i}",
                timestamp=datetime.now(timezone.utc),
                actor="tester",
                action="test",
            )
            backend.write(event)

        limited = backend.query(limit=3)
        assert len(limited) == 3

    def test_clear(self):
        """clear() should remove all events."""
        from aragora.server.middleware.audit_logger import AuditEvent, MemoryAuditBackend

        backend = MemoryAuditBackend()

        event = AuditEvent(
            event_id="test",
            timestamp=datetime.now(timezone.utc),
            actor="tester",
            action="test",
        )
        backend.write(event)
        assert len(backend.query()) == 1

        backend.clear()
        assert len(backend.query()) == 0
        assert backend.get_last_hash() is None


# ===========================================================================
# Test FileAuditBackend
# ===========================================================================


class TestFileAuditBackend:
    """Tests for FileAuditBackend."""

    def test_writes_to_file(self):
        """Should write events to JSONL file."""
        from aragora.server.middleware.audit_logger import AuditEvent, FileAuditBackend

        with tempfile.TemporaryDirectory() as tmp_dir:
            backend = FileAuditBackend(log_dir=tmp_dir)

            event = AuditEvent(
                event_id="file-test",
                timestamp=datetime.now(timezone.utc),
                actor="tester",
                action="test.file",
            )
            backend.write(event)

            # Check file exists
            log_files = list(Path(tmp_dir).glob("audit-*.jsonl"))
            assert len(log_files) == 1

    def test_queries_from_file(self):
        """Should query events from file."""
        from aragora.server.middleware.audit_logger import AuditEvent, FileAuditBackend

        with tempfile.TemporaryDirectory() as tmp_dir:
            backend = FileAuditBackend(log_dir=tmp_dir)

            event = AuditEvent(
                event_id="query-test",
                timestamp=datetime.now(timezone.utc),
                actor="querier",
                action="test.query",
            )
            backend.write(event)

            results = backend.query(limit=10)
            assert len(results) == 1
            assert results[0].event_id == "query-test"


# ===========================================================================
# Test AuditLogger Class
# ===========================================================================


class TestAuditLogger:
    """Tests for AuditLogger class."""

    def test_log_creates_event(self):
        """log() should create and write audit event."""
        from aragora.server.middleware.audit_logger import (
            AuditLogger,
            MemoryAuditBackend,
        )

        backend = MemoryAuditBackend()
        audit_logger = AuditLogger(backend=backend)

        event = audit_logger.log(
            action="user.login",
            actor="alice@example.com",
            resource="auth/session",
            outcome="success",
        )

        assert event.event_id.startswith("audit-")
        assert event.actor == "alice@example.com"
        assert event.action == "user.login"

        # Verify written to backend
        events = backend.query(limit=10)
        assert len(events) == 1

    def test_log_respects_min_severity(self):
        """log() should skip events below min_severity."""
        from aragora.server.middleware.audit_logger import (
            AuditLogger,
            AuditSeverity,
            MemoryAuditBackend,
        )

        backend = MemoryAuditBackend()
        audit_logger = AuditLogger(backend=backend, min_severity=AuditSeverity.WARNING)

        # DEBUG event should be skipped
        audit_logger.log(
            action="debug.event",
            actor="tester",
            severity=AuditSeverity.DEBUG,
        )

        # WARNING event should be logged
        audit_logger.log(
            action="warning.event",
            actor="tester",
            severity=AuditSeverity.WARNING,
        )

        events = backend.query(limit=10)
        assert len(events) == 1
        assert events[0].action == "warning.event"

    def test_verify_chain_integrity_valid(self):
        """verify_chain_integrity() should return True for valid chain."""
        from aragora.server.middleware.audit_logger import (
            AuditLogger,
            MemoryAuditBackend,
        )

        backend = MemoryAuditBackend()
        audit_logger = AuditLogger(backend=backend)

        # Log several events
        for i in range(5):
            audit_logger.log(
                action=f"test.{i}",
                actor="tester",
            )

        events = backend.query(limit=100)
        assert audit_logger.verify_chain_integrity(events) is True

    def test_verify_chain_integrity_tampered(self):
        """verify_chain_integrity() should return False for tampered events."""
        from aragora.server.middleware.audit_logger import (
            AuditLogger,
            MemoryAuditBackend,
        )

        backend = MemoryAuditBackend()
        audit_logger = AuditLogger(backend=backend)

        audit_logger.log(action="test.1", actor="tester")
        audit_logger.log(action="test.2", actor="tester")

        events = backend.query(limit=100)

        # Tamper with an event
        events[0].actor = "attacker"  # Change actor after hash computed

        assert audit_logger.verify_chain_integrity(events) is False


# ===========================================================================
# Test Global Logger Functions
# ===========================================================================


class TestGlobalLoggerFunctions:
    """Tests for global logger functions."""

    def setup_method(self):
        """Reset global logger before each test."""
        import aragora.server.middleware.audit_logger as audit_module

        audit_module._audit_logger = None

    def test_get_audit_logger_creates_instance(self):
        """get_audit_logger() should create logger on first call."""
        from aragora.server.middleware.audit_logger import AuditLogger, get_audit_logger

        logger = get_audit_logger()
        assert isinstance(logger, AuditLogger)

    def test_get_audit_logger_returns_same_instance(self):
        """get_audit_logger() should return same instance."""
        from aragora.server.middleware.audit_logger import get_audit_logger

        logger1 = get_audit_logger()
        logger2 = get_audit_logger()
        assert logger1 is logger2

    def test_set_audit_logger(self):
        """set_audit_logger() should replace global instance."""
        from aragora.server.middleware.audit_logger import (
            AuditLogger,
            MemoryAuditBackend,
            get_audit_logger,
            set_audit_logger,
        )

        custom = AuditLogger(backend=MemoryAuditBackend())
        set_audit_logger(custom)

        assert get_audit_logger() is custom


# ===========================================================================
# Test Context Management
# ===========================================================================


class TestContextManagement:
    """Tests for audit context management."""

    def setup_method(self):
        """Clear context before each test."""
        from aragora.server.middleware.audit_logger import clear_audit_context

        clear_audit_context()

    def test_set_audit_context(self):
        """set_audit_context() should set context variables."""
        import aragora.server.middleware.audit_logger as audit_module
        from aragora.server.middleware.audit_logger import set_audit_context

        set_audit_context(
            request_id="req-123",
            session_id="sess-456",
            actor="context-user",
            actor_ip="192.168.1.1",
        )

        assert audit_module._current_request_id.get() == "req-123"
        assert audit_module._current_session_id.get() == "sess-456"
        assert audit_module._current_actor.get() == "context-user"
        assert audit_module._current_actor_ip.get() == "192.168.1.1"

    def test_clear_audit_context(self):
        """clear_audit_context() should reset context variables."""
        import aragora.server.middleware.audit_logger as audit_module
        from aragora.server.middleware.audit_logger import (
            clear_audit_context,
            set_audit_context,
        )

        set_audit_context(
            request_id="req-123",
            session_id="sess-456",
        )
        clear_audit_context()

        assert audit_module._current_request_id.get() is None
        assert audit_module._current_session_id.get() is None


# ===========================================================================
# Test audit_event Convenience Function
# ===========================================================================


class TestAuditEventFunction:
    """Tests for audit_event() convenience function."""

    def setup_method(self):
        """Reset global logger and context before each test."""
        import aragora.server.middleware.audit_logger as audit_module

        audit_module._audit_logger = None
        audit_module.clear_audit_context()

    def test_audit_event_basic(self):
        """audit_event() should log event with provided values."""
        from aragora.server.middleware.audit_logger import (
            AuditLogger,
            MemoryAuditBackend,
            audit_event,
            set_audit_logger,
        )

        backend = MemoryAuditBackend()
        set_audit_logger(AuditLogger(backend=backend))

        event = audit_event(
            action="user.action",
            actor="test-user",
            resource="test-resource",
        )

        assert event.action == "user.action"
        assert event.actor == "test-user"
        assert event.resource == "test-resource"

    def test_audit_event_uses_context_actor(self):
        """audit_event() should use actor from context if not provided."""
        from aragora.server.middleware.audit_logger import (
            AuditLogger,
            MemoryAuditBackend,
            audit_event,
            set_audit_context,
            set_audit_logger,
        )

        backend = MemoryAuditBackend()
        set_audit_logger(AuditLogger(backend=backend))
        set_audit_context(actor="context-actor")

        event = audit_event(action="test.action")

        assert event.actor == "context-actor"


# ===========================================================================
# Test audit_action Decorator
# ===========================================================================


class TestAuditActionDecorator:
    """Tests for audit_action() decorator."""

    def setup_method(self):
        """Reset global logger before each test."""
        import aragora.server.middleware.audit_logger as audit_module

        audit_module._audit_logger = None
        backend = audit_module.MemoryAuditBackend()
        audit_module.set_audit_logger(audit_module.AuditLogger(backend=backend))

    def test_decorator_logs_sync_success(self):
        """Should log success for sync function."""
        from aragora.server.middleware.audit_logger import (
            audit_action,
            get_audit_logger,
        )

        @audit_action("test.sync_action")
        def sync_function():
            return "result"

        result = sync_function()

        assert result == "result"
        events = get_audit_logger().query(limit=10)
        assert len(events) == 1
        assert events[0].action == "test.sync_action"
        assert events[0].outcome == "success"

    def test_decorator_logs_sync_error(self):
        """Should log error for sync function exception."""
        from aragora.server.middleware.audit_logger import (
            audit_action,
            get_audit_logger,
        )

        @audit_action("test.sync_error")
        def failing_function():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            failing_function()

        events = get_audit_logger().query(limit=10)
        assert len(events) == 1
        assert events[0].outcome == "error"

    @pytest.mark.asyncio
    async def test_decorator_logs_async_success(self):
        """Should log success for async function."""
        from aragora.server.middleware.audit_logger import (
            audit_action,
            get_audit_logger,
        )

        @audit_action("test.async_action")
        async def async_function():
            return "async result"

        result = await async_function()

        assert result == "async result"
        events = get_audit_logger().query(limit=10)
        assert len(events) == 1
        assert events[0].action == "test.async_action"
        assert events[0].outcome == "success"

    @pytest.mark.asyncio
    async def test_decorator_logs_async_error(self):
        """Should log error for async function exception."""
        from aragora.server.middleware.audit_logger import (
            audit_action,
            get_audit_logger,
        )

        @audit_action("test.async_error")
        async def failing_async():
            raise KeyError("not found")

        with pytest.raises(KeyError):
            await failing_async()

        events = get_audit_logger().query(limit=10)
        assert len(events) == 1
        assert events[0].outcome == "error"


# ===========================================================================
# Test Pre-defined Audit Helpers
# ===========================================================================


class TestPredefinedAuditHelpers:
    """Tests for pre-defined audit helper functions."""

    def setup_method(self):
        """Reset global logger before each test."""
        import aragora.server.middleware.audit_logger as audit_module

        audit_module._audit_logger = None
        backend = audit_module.MemoryAuditBackend()
        audit_module.set_audit_logger(audit_module.AuditLogger(backend=backend))

    def test_audit_auth_login_success(self):
        """audit_auth_login() should log successful login."""
        from aragora.server.middleware.audit_logger import (
            AuditCategory,
            AuditSeverity,
            audit_auth_login,
        )

        event = audit_auth_login(
            user_id="alice",
            success=True,
            method="oauth",
        )

        assert event.action == "auth.login"
        assert event.actor == "alice"
        assert event.outcome == "success"
        assert event.category == AuditCategory.AUTHENTICATION
        assert event.severity == AuditSeverity.INFO

    def test_audit_auth_login_failure(self):
        """audit_auth_login() should log failed login."""
        from aragora.server.middleware.audit_logger import (
            AuditSeverity,
            audit_auth_login,
        )

        event = audit_auth_login(
            user_id="attacker",
            success=False,
            reason="invalid_password",
        )

        assert event.outcome == "failure"
        assert event.severity == AuditSeverity.WARNING
        assert event.details["failure_reason"] == "invalid_password"

    def test_audit_auth_logout(self):
        """audit_auth_logout() should log logout."""
        from aragora.server.middleware.audit_logger import audit_auth_logout

        event = audit_auth_logout(user_id="alice")

        assert event.action == "auth.logout"
        assert event.actor == "alice"
        assert event.outcome == "success"

    def test_audit_token_revoked(self):
        """audit_token_revoked() should log token revocation."""
        from aragora.server.middleware.audit_logger import audit_token_revoked

        event = audit_token_revoked(
            token_hash="abc123def456",
            revoked_by="admin",
            reason="security_breach",
        )

        assert event.action == "auth.token_revoked"
        assert event.resource_id == "abc123de"  # Truncated
        assert event.details["reason"] == "security_breach"

    def test_audit_access_denied(self):
        """audit_access_denied() should log access denied."""
        from aragora.server.middleware.audit_logger import (
            AuditCategory,
            AuditSeverity,
            audit_access_denied,
        )

        event = audit_access_denied(
            user_id="bob",
            resource="/admin/settings",
            required_permission="admin:write",
        )

        assert event.action == "authz.access_denied"
        assert event.outcome == "denied"
        assert event.category == AuditCategory.AUTHORIZATION
        assert event.severity == AuditSeverity.WARNING

    def test_audit_data_modified(self):
        """audit_data_modified() should log data modification."""
        from aragora.server.middleware.audit_logger import (
            AuditCategory,
            audit_data_modified,
        )

        event = audit_data_modified(
            user_id="editor",
            resource_type="debates",
            resource_id="debate-123",
            operation="update",
            changes={"title": "New Title"},
        )

        assert event.action == "data.update"
        assert event.resource == "debates"
        assert event.resource_id == "debate-123"
        assert event.category == AuditCategory.DATA_MODIFICATION

    def test_audit_config_changed(self):
        """audit_config_changed() should log config change."""
        from aragora.server.middleware.audit_logger import (
            AuditCategory,
            AuditSeverity,
            audit_config_changed,
        )

        event = audit_config_changed(
            user_id="admin",
            config_key="rate_limit.max",
            old_value="100",
            new_value="200",
        )

        assert event.action == "config.changed"
        assert event.category == AuditCategory.CONFIGURATION
        assert event.severity == AuditSeverity.WARNING
        assert event.details["old_value"] == "100"
        assert event.details["new_value"] == "200"

    def test_audit_security_event(self):
        """audit_security_event() should log security event."""
        from aragora.server.middleware.audit_logger import (
            AuditCategory,
            audit_security_event,
        )

        event = audit_security_event(
            event_type="rate_limit",
            actor="spammer@bad.com",
            details={"requests_blocked": 100},
        )

        assert event.action == "security.rate_limit"
        assert event.category == AuditCategory.SECURITY
        assert event.details["requests_blocked"] == 100
