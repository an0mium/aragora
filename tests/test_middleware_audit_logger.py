"""
Tests for the Audit Logging Middleware.

Covers:
- AuditEvent dataclass and hash computation
- MemoryAuditBackend (write, query, clear)
- FileAuditBackend (write, query, rotation)
- AuditLogger (log, query, verify_chain_integrity)
- Global functions (audit_event, get_audit_logger, set_audit_context)
- audit_action decorator (sync and async)
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.middleware.audit_logger import (
    AuditSeverity,
    AuditCategory,
    AuditEvent,
    MemoryAuditBackend,
    FileAuditBackend,
    AuditLogger,
    get_audit_logger,
    set_audit_logger,
    audit_event,
    audit_action,
    set_audit_context,
    clear_audit_context,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def memory_backend():
    """Create a fresh memory backend for testing."""
    return MemoryAuditBackend(max_events=100)


@pytest.fixture
def file_backend(tmp_path):
    """Create a file backend with temp directory."""
    return FileAuditBackend(
        log_dir=str(tmp_path / "audit"),
        file_prefix="test-audit",
        max_file_size_mb=1,
    )


@pytest.fixture
def audit_logger(memory_backend):
    """Create an audit logger with memory backend."""
    return AuditLogger(backend=memory_backend)


@pytest.fixture
def sample_event():
    """Create a sample audit event."""
    return AuditEvent(
        event_id="test-event-001",
        timestamp=datetime.now(timezone.utc),
        actor="user@example.com",
        action="user.login",
        category=AuditCategory.AUTHENTICATION,
        severity=AuditSeverity.INFO,
        resource="auth/session",
        outcome="success",
    )


@pytest.fixture(autouse=True)
def reset_global_logger():
    """Reset global audit logger before each test."""
    import aragora.server.middleware.audit_logger as audit_module

    original = audit_module._audit_logger
    audit_module._audit_logger = None
    clear_audit_context()
    yield
    audit_module._audit_logger = original
    clear_audit_context()


# =============================================================================
# AuditSeverity and AuditCategory Tests
# =============================================================================


class TestAuditEnums:
    """Tests for audit enums."""

    def test_severity_levels(self):
        """Test all severity levels exist."""
        assert AuditSeverity.DEBUG.value == "debug"
        assert AuditSeverity.INFO.value == "info"
        assert AuditSeverity.WARNING.value == "warning"
        assert AuditSeverity.ERROR.value == "error"
        assert AuditSeverity.CRITICAL.value == "critical"

    def test_category_values(self):
        """Test all category values exist."""
        assert AuditCategory.AUTHENTICATION.value == "authentication"
        assert AuditCategory.AUTHORIZATION.value == "authorization"
        assert AuditCategory.DATA_ACCESS.value == "data_access"
        assert AuditCategory.DATA_MODIFICATION.value == "data_modification"
        assert AuditCategory.CONFIGURATION.value == "configuration"
        assert AuditCategory.ADMINISTRATIVE.value == "administrative"
        assert AuditCategory.SECURITY.value == "security"
        assert AuditCategory.SYSTEM.value == "system"


# =============================================================================
# AuditEvent Tests
# =============================================================================


class TestAuditEvent:
    """Tests for AuditEvent dataclass."""

    def test_create_minimal_event(self):
        """Test creating event with minimal fields."""
        event = AuditEvent(
            event_id="evt-001",
            timestamp=datetime.now(timezone.utc),
            actor="system",
            action="startup",
        )
        assert event.event_id == "evt-001"
        assert event.actor == "system"
        assert event.action == "startup"
        assert event.outcome == "success"  # default
        assert event.severity == AuditSeverity.INFO  # default

    def test_create_full_event(self, sample_event):
        """Test creating event with all fields."""
        assert sample_event.actor == "user@example.com"
        assert sample_event.category == AuditCategory.AUTHENTICATION
        assert sample_event.resource == "auth/session"

    def test_to_dict(self, sample_event):
        """Test converting event to dictionary."""
        data = sample_event.to_dict()

        assert data["event_id"] == "test-event-001"
        assert data["actor"] == "user@example.com"
        assert data["action"] == "user.login"
        assert data["category"] == "authentication"
        assert data["severity"] == "info"
        assert data["outcome"] == "success"
        assert "timestamp" in data

    def test_to_dict_serializable(self, sample_event):
        """Test that to_dict output is JSON serializable."""
        data = sample_event.to_dict()
        json_str = json.dumps(data)
        assert json_str is not None

        # Round-trip
        parsed = json.loads(json_str)
        assert parsed["actor"] == sample_event.actor

    def test_compute_hash(self, sample_event):
        """Test hash computation is deterministic."""
        hash1 = sample_event.compute_hash()
        hash2 = sample_event.compute_hash()

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex

    def test_compute_hash_different_events(self):
        """Test different events have different hashes."""
        event1 = AuditEvent(
            event_id="evt-001",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            actor="user1",
            action="login",
        )
        event2 = AuditEvent(
            event_id="evt-002",
            timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            actor="user2",
            action="login",
        )

        assert event1.compute_hash() != event2.compute_hash()

    def test_event_with_details(self):
        """Test event with additional details."""
        event = AuditEvent(
            event_id="evt-001",
            timestamp=datetime.now(timezone.utc),
            actor="admin",
            action="config.change",
            details={"key": "max_users", "old_value": 100, "new_value": 200},
        )

        data = event.to_dict()
        assert data["details"]["key"] == "max_users"
        assert data["details"]["old_value"] == 100


# =============================================================================
# MemoryAuditBackend Tests
# =============================================================================


class TestMemoryAuditBackend:
    """Tests for in-memory audit backend."""

    def test_write_event(self, memory_backend, sample_event):
        """Test writing an event."""
        memory_backend.write(sample_event)

        events = memory_backend.query()
        assert len(events) == 1
        assert events[0].actor == sample_event.actor

    def test_write_sets_hash_chain(self, memory_backend):
        """Test that writing events creates hash chain."""
        event1 = AuditEvent(
            event_id="evt-001",
            timestamp=datetime.now(timezone.utc),
            actor="user1",
            action="action1",
        )
        event2 = AuditEvent(
            event_id="evt-002",
            timestamp=datetime.now(timezone.utc),
            actor="user2",
            action="action2",
        )

        memory_backend.write(event1)
        memory_backend.write(event2)

        events = memory_backend.query()
        assert events[0].event_hash is not None
        assert events[1].previous_hash == events[0].event_hash

    def test_query_empty(self, memory_backend):
        """Test querying empty backend."""
        events = memory_backend.query()
        assert events == []

    def test_query_with_actor_filter(self, memory_backend):
        """Test querying with actor filter."""
        for i in range(5):
            event = AuditEvent(
                event_id=f"evt-{i}",
                timestamp=datetime.now(timezone.utc),
                actor="user1" if i % 2 == 0 else "user2",
                action="test",
            )
            memory_backend.write(event)

        events = memory_backend.query(actor="user1")
        assert len(events) == 3
        assert all(e.actor == "user1" for e in events)

    def test_query_with_action_filter(self, memory_backend):
        """Test querying with action filter."""
        for action in ["login", "logout", "login", "update"]:
            event = AuditEvent(
                event_id=f"evt-{action}",
                timestamp=datetime.now(timezone.utc),
                actor="user",
                action=action,
            )
            memory_backend.write(event)

        events = memory_backend.query(action="login")
        assert len(events) == 2

    def test_query_with_time_filter(self, memory_backend):
        """Test querying with time filters."""
        now = datetime.now(timezone.utc)

        # Write events at different times
        for i in range(5):
            event = AuditEvent(
                event_id=f"evt-{i}",
                timestamp=now - timedelta(hours=i),
                actor="user",
                action="test",
            )
            memory_backend.write(event)

        # Query last 2 hours
        events = memory_backend.query(
            start_time=now - timedelta(hours=2),
            end_time=now + timedelta(minutes=1),
        )
        assert len(events) == 3  # 0, 1, 2 hours ago

    def test_query_with_limit(self, memory_backend):
        """Test query respects limit."""
        for i in range(10):
            event = AuditEvent(
                event_id=f"evt-{i}",
                timestamp=datetime.now(timezone.utc),
                actor="user",
                action="test",
            )
            memory_backend.write(event)

        events = memory_backend.query(limit=5)
        assert len(events) == 5

    def test_max_events_limit(self):
        """Test that backend respects max events limit."""
        backend = MemoryAuditBackend(max_events=5)

        for i in range(10):
            event = AuditEvent(
                event_id=f"evt-{i}",
                timestamp=datetime.now(timezone.utc),
                actor="user",
                action="test",
            )
            backend.write(event)

        events = backend.query(limit=100)
        assert len(events) == 5  # Only 5 retained
        # Should have newest events (6-9)
        assert events[0].event_id == "evt-5"

    def test_get_last_hash(self, memory_backend, sample_event):
        """Test getting last hash."""
        assert memory_backend.get_last_hash() is None

        memory_backend.write(sample_event)
        assert memory_backend.get_last_hash() is not None

    def test_clear(self, memory_backend, sample_event):
        """Test clearing all events."""
        memory_backend.write(sample_event)
        assert len(memory_backend.query()) == 1

        memory_backend.clear()
        assert len(memory_backend.query()) == 0
        assert memory_backend.get_last_hash() is None


# =============================================================================
# FileAuditBackend Tests
# =============================================================================


class TestFileAuditBackend:
    """Tests for file-based audit backend."""

    def test_creates_log_directory(self, tmp_path):
        """Test that backend creates log directory."""
        log_dir = tmp_path / "audit" / "logs"
        backend = FileAuditBackend(log_dir=str(log_dir))

        assert log_dir.exists()

    def test_write_event(self, file_backend, sample_event):
        """Test writing event to file."""
        file_backend.write(sample_event)

        events = file_backend.query()
        assert len(events) == 1
        assert events[0].actor == sample_event.actor

    def test_write_creates_jsonl_file(self, file_backend, sample_event, tmp_path):
        """Test that write creates JSONL file."""
        file_backend.write(sample_event)

        log_dir = tmp_path / "audit"
        log_files = list(log_dir.glob("*.jsonl"))
        assert len(log_files) == 1

        # Verify file contents
        with open(log_files[0]) as f:
            line = f.readline()
            data = json.loads(line)
            assert data["actor"] == sample_event.actor

    def test_hash_chain_persisted(self, file_backend):
        """Test that hash chain is persisted to file."""
        event1 = AuditEvent(
            event_id="evt-001",
            timestamp=datetime.now(timezone.utc),
            actor="user1",
            action="action1",
        )
        event2 = AuditEvent(
            event_id="evt-002",
            timestamp=datetime.now(timezone.utc),
            actor="user2",
            action="action2",
        )

        file_backend.write(event1)
        file_backend.write(event2)

        events = file_backend.query()
        assert events[1].previous_hash == events[0].event_hash

    def test_query_with_filters(self, file_backend):
        """Test querying with filters."""
        for i in range(5):
            event = AuditEvent(
                event_id=f"evt-{i}",
                timestamp=datetime.now(timezone.utc),
                actor="admin" if i % 2 == 0 else "user",
                action="test",
            )
            file_backend.write(event)

        events = file_backend.query(actor="admin")
        assert len(events) == 3

    def test_get_last_hash(self, file_backend, sample_event):
        """Test getting last hash from file backend."""
        assert file_backend.get_last_hash() is None

        file_backend.write(sample_event)
        last_hash = file_backend.get_last_hash()
        assert last_hash is not None

    def test_load_last_hash_on_init(self, tmp_path, sample_event):
        """Test that backend loads last hash from existing file."""
        # Create first backend and write event
        backend1 = FileAuditBackend(log_dir=str(tmp_path / "audit"))
        backend1.write(sample_event)
        last_hash = backend1.get_last_hash()

        # Create new backend - should load existing hash
        backend2 = FileAuditBackend(log_dir=str(tmp_path / "audit"))
        assert backend2.get_last_hash() == last_hash


# =============================================================================
# AuditLogger Tests
# =============================================================================


class TestAuditLogger:
    """Tests for the main AuditLogger class."""

    def test_log_event(self, audit_logger):
        """Test logging an event."""
        event = audit_logger.log(
            action="user.login",
            actor="user@example.com",
            resource="auth/session",
            outcome="success",
        )

        assert event.event_id.startswith("audit-")
        assert event.action == "user.login"
        assert event.actor == "user@example.com"

    def test_log_with_severity(self, audit_logger):
        """Test logging with different severities."""
        event = audit_logger.log(
            action="security.blocked",
            actor="attacker",
            severity=AuditSeverity.CRITICAL,
        )

        assert event.severity == AuditSeverity.CRITICAL

    def test_log_below_min_severity(self, memory_backend):
        """Test that events below min severity are not logged."""
        logger = AuditLogger(
            backend=memory_backend,
            min_severity=AuditSeverity.WARNING,
        )

        # DEBUG and INFO should not be logged
        logger.log(action="debug.event", severity=AuditSeverity.DEBUG)
        logger.log(action="info.event", severity=AuditSeverity.INFO)

        # WARNING should be logged
        logger.log(action="warning.event", severity=AuditSeverity.WARNING)

        events = memory_backend.query()
        assert len(events) == 1
        assert events[0].action == "warning.event"

    def test_log_with_context(self, audit_logger):
        """Test that context variables are captured."""
        set_audit_context(
            request_id="req-123",
            session_id="sess-456",
        )

        event = audit_logger.log(action="test.action", actor="user")

        assert event.request_id == "req-123"
        assert event.session_id == "sess-456"

    def test_query_events(self, audit_logger):
        """Test querying events through logger."""
        for i in range(5):
            audit_logger.log(action=f"action-{i}", actor="user")

        events = audit_logger.query(limit=3)
        assert len(events) == 3

    def test_verify_chain_integrity_valid(self, audit_logger):
        """Test chain verification with valid chain."""
        for i in range(5):
            audit_logger.log(action=f"action-{i}", actor="user")

        events = audit_logger.query()
        assert audit_logger.verify_chain_integrity(events) is True

    def test_verify_chain_integrity_empty(self, audit_logger):
        """Test chain verification with empty list."""
        assert audit_logger.verify_chain_integrity([]) is True

    def test_verify_chain_integrity_tampered_hash(self, audit_logger):
        """Test chain verification detects tampered hash."""
        audit_logger.log(action="action1", actor="user")
        audit_logger.log(action="action2", actor="user")

        events = audit_logger.query()

        # Tamper with hash
        events[0].event_hash = "tampered"

        assert audit_logger.verify_chain_integrity(events) is False

    def test_verify_chain_integrity_broken_chain(self, audit_logger):
        """Test chain verification detects broken chain."""
        audit_logger.log(action="action1", actor="user")
        audit_logger.log(action="action2", actor="user")

        events = audit_logger.query()

        # Break chain linkage
        events[1].previous_hash = "wrong-hash"

        assert audit_logger.verify_chain_integrity(events) is False


# =============================================================================
# Global Functions Tests
# =============================================================================


class TestGlobalFunctions:
    """Tests for global audit functions."""

    def test_get_audit_logger_creates_singleton(self):
        """Test that get_audit_logger returns singleton."""
        logger1 = get_audit_logger()
        logger2 = get_audit_logger()

        assert logger1 is logger2

    def test_set_audit_logger(self, memory_backend):
        """Test setting custom audit logger."""
        custom_logger = AuditLogger(backend=memory_backend)
        set_audit_logger(custom_logger)

        assert get_audit_logger() is custom_logger

    def test_audit_event_convenience_function(self, memory_backend):
        """Test audit_event convenience function."""
        custom_logger = AuditLogger(backend=memory_backend)
        set_audit_logger(custom_logger)

        event = audit_event(
            action="test.action",
            actor="test-user",
            resource="test-resource",
        )

        assert event.action == "test.action"
        events = memory_backend.query()
        assert len(events) == 1

    def test_audit_event_uses_context(self, memory_backend):
        """Test audit_event uses context variables."""
        custom_logger = AuditLogger(backend=memory_backend)
        set_audit_logger(custom_logger)

        set_audit_context(actor="context-user", actor_ip="192.168.1.1")

        event = audit_event(action="test.action")

        assert event.actor == "context-user"
        assert event.actor_ip == "192.168.1.1"

    def test_audit_event_actor_override(self, memory_backend):
        """Test audit_event allows actor override."""
        custom_logger = AuditLogger(backend=memory_backend)
        set_audit_logger(custom_logger)

        set_audit_context(actor="context-user")

        event = audit_event(action="test.action", actor="override-user")

        assert event.actor == "override-user"

    def test_clear_audit_context(self, memory_backend):
        """Test clearing audit context."""
        custom_logger = AuditLogger(backend=memory_backend)
        set_audit_logger(custom_logger)

        set_audit_context(actor="test-user", request_id="req-123")
        clear_audit_context()

        event = audit_event(action="test.action")

        assert event.actor == "unknown"  # Falls back to default
        assert event.request_id is None


# =============================================================================
# Decorator Tests
# =============================================================================


class TestAuditActionDecorator:
    """Tests for the audit_action decorator."""

    def test_async_decorator_success(self, memory_backend):
        """Test decorator logs successful async function."""
        custom_logger = AuditLogger(backend=memory_backend)
        set_audit_logger(custom_logger)
        set_audit_context(actor="test-user")

        @audit_action("test.async_action", category=AuditCategory.DATA_MODIFICATION)
        async def async_func():
            return "result"

        result = asyncio.run(async_func())

        assert result == "result"
        events = memory_backend.query()
        assert len(events) == 1
        assert events[0].action == "test.async_action"
        assert events[0].outcome == "success"
        assert events[0].category == AuditCategory.DATA_MODIFICATION

    def test_async_decorator_with_kwargs(self, memory_backend):
        """Test decorator extracts resource from kwargs."""
        custom_logger = AuditLogger(backend=memory_backend)
        set_audit_logger(custom_logger)
        set_audit_context(actor="test-user")

        @audit_action(
            "debate.create",
            resource_from="debate_type",
            resource_id_from="debate_id",
        )
        async def create_debate(debate_type: str, debate_id: str):
            return f"Created {debate_id}"

        result = asyncio.run(
            create_debate(debate_type="standard", debate_id="dbt-123")
        )

        events = memory_backend.query()
        assert events[0].resource == "standard"
        assert events[0].resource_id == "dbt-123"

    def test_async_decorator_error(self, memory_backend):
        """Test decorator logs errors with elevated severity."""
        custom_logger = AuditLogger(backend=memory_backend)
        set_audit_logger(custom_logger)
        set_audit_context(actor="test-user")

        @audit_action("test.failing_action")
        async def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            asyncio.run(failing_func())

        events = memory_backend.query()
        assert len(events) == 1
        assert events[0].outcome == "error"
        assert events[0].severity == AuditSeverity.ERROR
        assert "Test error" in events[0].details.get("outcome_reason", "")

    def test_async_decorator_permission_denied(self, memory_backend):
        """Test decorator logs permission errors."""
        custom_logger = AuditLogger(backend=memory_backend)
        set_audit_logger(custom_logger)
        set_audit_context(actor="test-user")

        @audit_action("test.protected_action")
        async def protected_func():
            raise PermissionError("Access denied")

        with pytest.raises(PermissionError):
            asyncio.run(protected_func())

        events = memory_backend.query()
        assert events[0].outcome == "denied"
        assert events[0].severity == AuditSeverity.WARNING

    def test_sync_decorator_success(self, memory_backend):
        """Test decorator logs successful sync function."""
        custom_logger = AuditLogger(backend=memory_backend)
        set_audit_logger(custom_logger)
        set_audit_context(actor="test-user")

        @audit_action("test.sync_action")
        def sync_func(value: int):
            return value * 2

        result = sync_func(21)

        assert result == 42
        events = memory_backend.query()
        assert len(events) == 1
        assert events[0].action == "test.sync_action"
        assert events[0].outcome == "success"

    def test_decorator_records_elapsed_time(self, memory_backend):
        """Test decorator records elapsed time."""
        custom_logger = AuditLogger(backend=memory_backend)
        set_audit_logger(custom_logger)
        set_audit_context(actor="test-user")

        @audit_action("test.timed_action")
        async def slow_func():
            await asyncio.sleep(0.05)
            return "done"

        asyncio.get_event_loop().run_until_complete(slow_func())

        events = memory_backend.query()
        elapsed = events[0].details.get("elapsed_ms", 0)
        assert elapsed >= 50  # At least 50ms


# =============================================================================
# Integration Tests
# =============================================================================


class TestAuditLoggerIntegration:
    """Integration tests for audit logging."""

    def test_full_audit_flow(self, tmp_path):
        """Test complete audit flow with file backend."""
        # Setup
        backend = FileAuditBackend(log_dir=str(tmp_path / "audit"))
        logger = AuditLogger(backend=backend)
        set_audit_logger(logger)

        # Log events
        set_audit_context(
            request_id="req-integration-test",
            actor="integration-user",
            actor_ip="10.0.0.1",
        )

        audit_event(
            action="user.login",
            category=AuditCategory.AUTHENTICATION,
            details={"method": "password"},
        )

        audit_event(
            action="debate.create",
            category=AuditCategory.DATA_MODIFICATION,
            resource="debates",
            resource_id="dbt-test-123",
        )

        audit_event(
            action="user.logout",
            category=AuditCategory.AUTHENTICATION,
        )

        clear_audit_context()

        # Query and verify
        events = logger.query(actor="integration-user")
        assert len(events) == 3

        # Verify chain integrity
        assert logger.verify_chain_integrity(events) is True

        # Verify file persistence
        log_files = list((tmp_path / "audit").glob("*.jsonl"))
        assert len(log_files) == 1

        with open(log_files[0]) as f:
            lines = f.readlines()
            assert len(lines) == 3

    def test_concurrent_writes(self, memory_backend):
        """Test concurrent audit writes are thread-safe."""
        import concurrent.futures

        logger = AuditLogger(backend=memory_backend)

        def write_event(i: int):
            logger.log(action=f"concurrent.action-{i}", actor=f"user-{i}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(write_event, i) for i in range(100)]
            concurrent.futures.wait(futures)

        events = memory_backend.query(limit=200)
        assert len(events) == 100

        # Verify chain integrity even with concurrent writes
        assert logger.verify_chain_integrity(events) is True
