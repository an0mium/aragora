"""
Tests for Control Plane Audit Log.

Tests cover:
- Enums: AuditAction, ActorType
- Dataclasses: AuditActor, AuditEntry, AuditQuery
- AuditLog: log, query, integrity, export, retention, stats
- Helper functions: create_system_actor, create_agent_actor, create_user_actor
- Convenience functions: log_policy_decision, log_deliberation_event, etc.
- Global audit log: get_audit_log, set_audit_log
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from aragora.control_plane.audit import (
    AUDIT_RETENTION_DAYS,
    AUDIT_STREAM_KEY,
    ActorType,
    AuditAction,
    AuditActor,
    AuditEntry,
    AuditLog,
    AuditQuery,
    create_agent_actor,
    create_system_actor,
    create_user_actor,
    get_audit_log,
    log_deliberation_completed,
    log_deliberation_event,
    log_deliberation_sla_event,
    log_deliberation_started,
    log_policy_decision,
    set_audit_log,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def system_actor():
    return AuditActor(
        actor_type=ActorType.SYSTEM,
        actor_id="test-system",
        actor_name="Test System",
    )


@pytest.fixture
def audit_log():
    """Create an in-memory audit log (no Redis)."""
    log = AuditLog()
    log._redis = None  # Force local mode
    return log


@pytest.fixture(autouse=True)
def reset_global_audit_log():
    """Reset global audit log before/after each test."""
    import aragora.control_plane.audit as audit_mod

    old = audit_mod._audit_log
    audit_mod._audit_log = None
    yield
    audit_mod._audit_log = old


# ============================================================================
# Enum Tests
# ============================================================================


class TestAuditAction:
    """Tests for AuditAction enum."""

    def test_agent_actions(self):
        assert AuditAction.AGENT_REGISTERED.value == "agent.registered"
        assert AuditAction.AGENT_UNREGISTERED.value == "agent.unregistered"
        assert AuditAction.AGENT_STATUS_CHANGED.value == "agent.status_changed"

    def test_task_actions(self):
        assert AuditAction.TASK_SUBMITTED.value == "task.submitted"
        assert AuditAction.TASK_COMPLETED.value == "task.completed"
        assert AuditAction.TASK_FAILED.value == "task.failed"

    def test_deliberation_actions(self):
        assert AuditAction.DELIBERATION_STARTED.value == "deliberation.started"
        assert AuditAction.DELIBERATION_CONSENSUS.value == "deliberation.consensus"

    def test_policy_actions(self):
        assert AuditAction.POLICY_DECISION_ALLOW.value == "policy.decision_allow"
        assert AuditAction.POLICY_DECISION_DENY.value == "policy.decision_deny"
        assert AuditAction.POLICY_DECISION_WARN.value == "policy.decision_warn"

    def test_auth_actions(self):
        assert AuditAction.AUTH_LOGIN.value == "auth.login"
        assert AuditAction.AUTH_TOKEN_REVOKED.value == "auth.token_revoked"

    def test_system_actions(self):
        assert AuditAction.SYSTEM_STARTUP.value == "system.startup"
        assert AuditAction.SYSTEM_ERROR.value == "system.error"

    def test_total_count(self):
        assert len(AuditAction) == 36


class TestActorType:
    """Tests for ActorType enum."""

    def test_all_values(self):
        assert ActorType.AGENT.value == "agent"
        assert ActorType.USER.value == "user"
        assert ActorType.SYSTEM.value == "system"
        assert ActorType.API.value == "api"
        assert ActorType.SCHEDULER.value == "scheduler"

    def test_count(self):
        assert len(ActorType) == 5


# ============================================================================
# AuditActor Tests
# ============================================================================


class TestAuditActor:
    """Tests for AuditActor dataclass."""

    def test_basic_creation(self):
        actor = AuditActor(actor_type=ActorType.AGENT, actor_id="agent-1")
        assert actor.actor_type == ActorType.AGENT
        assert actor.actor_id == "agent-1"
        assert actor.actor_name is None
        assert actor.ip_address is None

    def test_full_creation(self):
        actor = AuditActor(
            actor_type=ActorType.USER,
            actor_id="user-1",
            actor_name="John",
            ip_address="192.168.1.1",
            user_agent="Test/1.0",
        )
        assert actor.actor_name == "John"
        assert actor.ip_address == "192.168.1.1"
        assert actor.user_agent == "Test/1.0"

    def test_to_dict(self):
        actor = AuditActor(
            actor_type=ActorType.SYSTEM,
            actor_id="sys-1",
            actor_name="System",
        )
        d = actor.to_dict()
        assert d["type"] == "system"
        assert d["id"] == "sys-1"
        assert d["name"] == "System"

    def test_from_dict(self):
        data = {
            "type": "agent",
            "id": "agent-2",
            "name": "Test Agent",
            "ip_address": "10.0.0.1",
        }
        actor = AuditActor.from_dict(data)
        assert actor.actor_type == ActorType.AGENT
        assert actor.actor_id == "agent-2"
        assert actor.actor_name == "Test Agent"

    def test_from_dict_defaults(self):
        actor = AuditActor.from_dict({})
        assert actor.actor_type == ActorType.SYSTEM
        assert actor.actor_id == "unknown"

    def test_round_trip(self):
        original = AuditActor(
            actor_type=ActorType.USER,
            actor_id="u-123",
            actor_name="Alice",
            ip_address="10.0.0.1",
            user_agent="Browser/2.0",
        )
        restored = AuditActor.from_dict(original.to_dict())
        assert restored.actor_type == original.actor_type
        assert restored.actor_id == original.actor_id
        assert restored.actor_name == original.actor_name


# ============================================================================
# AuditEntry Tests
# ============================================================================


class TestAuditEntry:
    """Tests for AuditEntry dataclass."""

    def test_basic_creation(self, system_actor):
        entry = AuditEntry(
            action=AuditAction.TASK_SUBMITTED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-123",
        )
        assert entry.action == AuditAction.TASK_SUBMITTED
        assert entry.resource_type == "task"
        assert entry.outcome == "success"
        assert entry.sequence_number == 0
        assert entry.entry_id is not None

    def test_compute_hash(self, system_actor):
        entry = AuditEntry(
            action=AuditAction.TASK_COMPLETED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-456",
            sequence_number=1,
        )
        h = entry.compute_hash()
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_hash_deterministic(self, system_actor):
        entry = AuditEntry(
            entry_id="fixed-id",
            action=AuditAction.TASK_COMPLETED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-456",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
            sequence_number=1,
        )
        assert entry.compute_hash() == entry.compute_hash()

    def test_hash_changes_with_data(self, system_actor):
        entry1 = AuditEntry(
            entry_id="fixed",
            action=AuditAction.TASK_COMPLETED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-1",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        entry2 = AuditEntry(
            entry_id="fixed",
            action=AuditAction.TASK_FAILED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-1",
            timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        assert entry1.compute_hash() != entry2.compute_hash()

    def test_to_dict(self, system_actor):
        entry = AuditEntry(
            action=AuditAction.AUTH_LOGIN,
            actor=system_actor,
            resource_type="session",
            resource_id="s-1",
            workspace_id="ws-1",
            details={"method": "password"},
            outcome="success",
        )
        d = entry.to_dict()
        assert d["action"] == "auth.login"
        assert d["resource_type"] == "session"
        assert d["workspace_id"] == "ws-1"
        assert d["details"]["method"] == "password"

    def test_from_dict(self):
        data = {
            "entry_id": "e-1",
            "action": "task.completed",
            "actor": {"type": "agent", "id": "a-1"},
            "resource_type": "task",
            "resource_id": "t-1",
            "timestamp": "2026-01-01T12:00:00+00:00",
            "outcome": "success",
            "sequence_number": 5,
        }
        entry = AuditEntry.from_dict(data)
        assert entry.action == AuditAction.TASK_COMPLETED
        assert entry.sequence_number == 5
        assert entry.actor.actor_type == ActorType.AGENT

    def test_round_trip(self, system_actor):
        original = AuditEntry(
            action=AuditAction.POLICY_EVALUATED,
            actor=system_actor,
            resource_type="policy",
            resource_id="p-1",
            workspace_id="ws-1",
            details={"key": "value"},
            outcome="success",
            sequence_number=10,
            previous_hash="abc123",
            entry_hash="def456",
        )
        d = original.to_dict()
        restored = AuditEntry.from_dict(d)
        assert restored.action == original.action
        assert restored.resource_id == original.resource_id
        assert restored.workspace_id == original.workspace_id
        assert restored.sequence_number == original.sequence_number
        assert restored.previous_hash == original.previous_hash


# ============================================================================
# AuditQuery Tests
# ============================================================================


class TestAuditQuery:
    """Tests for AuditQuery dataclass."""

    def test_default_values(self):
        q = AuditQuery()
        assert q.limit == 100
        assert q.offset == 0
        assert q.actions is None

    def test_matches_no_filters(self, system_actor):
        q = AuditQuery()
        entry = AuditEntry(
            action=AuditAction.TASK_COMPLETED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-1",
        )
        assert q.matches(entry) is True

    def test_matches_action_filter(self, system_actor):
        q = AuditQuery(actions=[AuditAction.TASK_COMPLETED])
        entry_match = AuditEntry(
            action=AuditAction.TASK_COMPLETED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-1",
        )
        entry_no_match = AuditEntry(
            action=AuditAction.TASK_FAILED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-2",
        )
        assert q.matches(entry_match) is True
        assert q.matches(entry_no_match) is False

    def test_matches_actor_type_filter(self, system_actor):
        q = AuditQuery(actor_types=[ActorType.AGENT])
        entry = AuditEntry(
            action=AuditAction.TASK_COMPLETED,
            actor=system_actor,  # SYSTEM actor
            resource_type="task",
            resource_id="t-1",
        )
        assert q.matches(entry) is False

    def test_matches_time_range(self, system_actor):
        now = datetime.now(timezone.utc)
        q = AuditQuery(
            start_time=now - timedelta(hours=1),
            end_time=now + timedelta(hours=1),
        )
        entry_in_range = AuditEntry(
            action=AuditAction.TASK_COMPLETED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-1",
            timestamp=now,
        )
        entry_out_range = AuditEntry(
            action=AuditAction.TASK_COMPLETED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-2",
            timestamp=now - timedelta(hours=2),
        )
        assert q.matches(entry_in_range) is True
        assert q.matches(entry_out_range) is False

    def test_matches_resource_filter(self, system_actor):
        q = AuditQuery(resource_types=["task"], resource_ids=["t-1"])
        entry = AuditEntry(
            action=AuditAction.TASK_COMPLETED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-1",
        )
        assert q.matches(entry) is True

    def test_matches_workspace_filter(self, system_actor):
        q = AuditQuery(workspace_ids=["ws-1"])
        entry_match = AuditEntry(
            action=AuditAction.TASK_COMPLETED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-1",
            workspace_id="ws-1",
        )
        entry_no_match = AuditEntry(
            action=AuditAction.TASK_COMPLETED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-2",
            workspace_id="ws-2",
        )
        assert q.matches(entry_match) is True
        assert q.matches(entry_no_match) is False

    def test_matches_outcome_filter(self, system_actor):
        q = AuditQuery(outcomes=["failure"])
        entry = AuditEntry(
            action=AuditAction.TASK_FAILED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-1",
            outcome="failure",
        )
        assert q.matches(entry) is True


# ============================================================================
# AuditLog Core Tests
# ============================================================================


class TestAuditLogCore:
    """Tests for AuditLog class - core operations."""

    def test_init_defaults(self):
        log = AuditLog()
        assert log._redis_url == "redis://localhost:6379"
        assert log._stream_key == AUDIT_STREAM_KEY
        assert log._retention_days == AUDIT_RETENTION_DAYS
        assert log._sequence_number == 0

    def test_init_custom(self):
        log = AuditLog(
            redis_url="redis://custom:6380",
            stream_key="custom:audit",
            retention_days=30,
        )
        assert log._redis_url == "redis://custom:6380"
        assert log._stream_key == "custom:audit"
        assert log._retention_days == 30

    @pytest.mark.asyncio
    async def test_log_entry(self, audit_log, system_actor):
        entry = await audit_log.log(
            action=AuditAction.TASK_SUBMITTED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-1",
        )
        assert entry.action == AuditAction.TASK_SUBMITTED
        assert entry.sequence_number == 1
        assert entry.entry_hash is not None
        assert entry.previous_hash is None  # First entry

    @pytest.mark.asyncio
    async def test_log_chain(self, audit_log, system_actor):
        e1 = await audit_log.log(
            action=AuditAction.TASK_SUBMITTED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-1",
        )
        e2 = await audit_log.log(
            action=AuditAction.TASK_COMPLETED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-1",
        )
        assert e2.previous_hash == e1.entry_hash
        assert e2.sequence_number == 2

    @pytest.mark.asyncio
    async def test_log_with_details(self, audit_log, system_actor):
        entry = await audit_log.log(
            action=AuditAction.TASK_COMPLETED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-1",
            workspace_id="ws-1",
            details={"result": "42"},
            outcome="success",
        )
        assert entry.details["result"] == "42"
        assert entry.workspace_id == "ws-1"

    @pytest.mark.asyncio
    async def test_log_failure(self, audit_log, system_actor):
        entry = await audit_log.log(
            action=AuditAction.TASK_FAILED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-1",
            outcome="failure",
            error_message="Timeout",
        )
        assert entry.outcome == "failure"
        assert entry.error_message == "Timeout"


# ============================================================================
# AuditLog Query Tests
# ============================================================================


class TestAuditLogQuery:
    """Tests for AuditLog query operations."""

    @pytest.mark.asyncio
    async def test_query_all(self, audit_log, system_actor):
        for i in range(5):
            await audit_log.log(
                action=AuditAction.TASK_SUBMITTED,
                actor=system_actor,
                resource_type="task",
                resource_id=f"t-{i}",
            )
        entries = await audit_log.query(AuditQuery())
        assert len(entries) == 5

    @pytest.mark.asyncio
    async def test_query_with_action_filter(self, audit_log, system_actor):
        await audit_log.log(
            action=AuditAction.TASK_SUBMITTED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-1",
        )
        await audit_log.log(
            action=AuditAction.TASK_COMPLETED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-2",
        )
        entries = await audit_log.query(AuditQuery(actions=[AuditAction.TASK_COMPLETED]))
        assert len(entries) == 1
        assert entries[0].resource_id == "t-2"

    @pytest.mark.asyncio
    async def test_query_pagination(self, audit_log, system_actor):
        for i in range(10):
            await audit_log.log(
                action=AuditAction.TASK_SUBMITTED,
                actor=system_actor,
                resource_type="task",
                resource_id=f"t-{i}",
            )
        entries = await audit_log.query(AuditQuery(limit=3, offset=2))
        assert len(entries) == 3

    @pytest.mark.asyncio
    async def test_query_empty(self, audit_log):
        entries = await audit_log.query(AuditQuery())
        assert entries == []


# ============================================================================
# Integrity Verification Tests
# ============================================================================


class TestAuditLogIntegrity:
    """Tests for audit log integrity verification."""

    @pytest.mark.asyncio
    async def test_integrity_empty(self, audit_log):
        assert await audit_log.verify_integrity() is True

    @pytest.mark.asyncio
    async def test_integrity_valid_chain(self, audit_log, system_actor):
        for i in range(5):
            await audit_log.log(
                action=AuditAction.TASK_SUBMITTED,
                actor=system_actor,
                resource_type="task",
                resource_id=f"t-{i}",
            )
        assert await audit_log.verify_integrity() is True

    @pytest.mark.asyncio
    async def test_integrity_tampered_hash(self, audit_log, system_actor):
        for i in range(3):
            await audit_log.log(
                action=AuditAction.TASK_SUBMITTED,
                actor=system_actor,
                resource_type="task",
                resource_id=f"t-{i}",
            )
        # Tamper with an entry
        audit_log._local_entries[1].entry_hash = "tampered"
        assert await audit_log.verify_integrity() is False

    @pytest.mark.asyncio
    async def test_integrity_tampered_chain(self, audit_log, system_actor):
        for i in range(3):
            await audit_log.log(
                action=AuditAction.TASK_SUBMITTED,
                actor=system_actor,
                resource_type="task",
                resource_id=f"t-{i}",
            )
        # Tamper with the chain link
        audit_log._local_entries[2].previous_hash = "broken-chain"
        assert await audit_log.verify_integrity() is False


# ============================================================================
# Export Tests
# ============================================================================


class TestAuditLogExport:
    """Tests for audit log export operations."""

    @pytest.mark.asyncio
    async def test_export_json(self, audit_log, system_actor):
        await audit_log.log(
            action=AuditAction.TASK_COMPLETED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-1",
        )
        result = await audit_log.export(AuditQuery(), format="json")
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["action"] == "task.completed"

    @pytest.mark.asyncio
    async def test_export_csv(self, audit_log, system_actor):
        await audit_log.log(
            action=AuditAction.TASK_COMPLETED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-1",
        )
        result = await audit_log.export(AuditQuery(), format="csv")
        lines = result.strip().split("\n")
        assert len(lines) == 2  # Header + 1 entry
        assert "entry_id" in lines[0]  # Header

    @pytest.mark.asyncio
    async def test_export_syslog(self, audit_log, system_actor):
        await audit_log.log(
            action=AuditAction.AUTH_LOGIN,
            actor=system_actor,
            resource_type="session",
            resource_id="s-1",
        )
        result = await audit_log.export(AuditQuery(), format="syslog")
        assert "aragora" in result
        assert "auth.login" in result

    @pytest.mark.asyncio
    async def test_export_soc2(self, audit_log, system_actor):
        await audit_log.log(
            action=AuditAction.AUTH_LOGIN,
            actor=system_actor,
            resource_type="session",
            resource_id="s-1",
        )
        await audit_log.log(
            action=AuditAction.TASK_COMPLETED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-1",
        )
        result = await audit_log.export(AuditQuery(), format="soc2")
        data = json.loads(result)
        assert data["report_type"] == "SOC 2 Type II Audit Evidence"
        assert "controls" in data
        assert data["summary"]["total_events"] == 2

    @pytest.mark.asyncio
    async def test_export_iso27001(self, audit_log, system_actor):
        await audit_log.log(
            action=AuditAction.AUTH_LOGIN,
            actor=system_actor,
            resource_type="session",
            resource_id="s-1",
        )
        result = await audit_log.export(AuditQuery(), format="iso27001")
        data = json.loads(result)
        assert data["standard"] == "ISO/IEC 27001:2022"
        assert "control_domains" in data

    @pytest.mark.asyncio
    async def test_export_unsupported_format(self, audit_log):
        with pytest.raises(ValueError, match="Unsupported export format"):
            await audit_log.export(AuditQuery(), format="xml")


# ============================================================================
# Retention Tests
# ============================================================================


class TestAuditLogRetention:
    """Tests for audit log retention enforcement."""

    @pytest.mark.asyncio
    async def test_enforce_retention_removes_old(self, audit_log, system_actor):
        # Add old entry
        old_entry = AuditEntry(
            action=AuditAction.TASK_COMPLETED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-old",
            timestamp=datetime.now(timezone.utc) - timedelta(days=180),
        )
        audit_log._local_entries.append(old_entry)

        # Add recent entry
        await audit_log.log(
            action=AuditAction.TASK_SUBMITTED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-new",
        )

        removed = await audit_log.enforce_retention()
        assert removed == 1
        assert len(audit_log._local_entries) == 1

    @pytest.mark.asyncio
    async def test_enforce_retention_keeps_recent(self, audit_log, system_actor):
        await audit_log.log(
            action=AuditAction.TASK_SUBMITTED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-1",
        )
        removed = await audit_log.enforce_retention()
        assert removed == 0
        assert len(audit_log._local_entries) == 1

    @pytest.mark.asyncio
    async def test_get_retention_status(self, audit_log, system_actor):
        await audit_log.log(
            action=AuditAction.TASK_SUBMITTED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-1",
        )
        status = await audit_log.get_retention_status()
        assert status["retention_days"] == AUDIT_RETENTION_DAYS
        assert status["total_entries"] == 1
        assert status["oldest_entry"] is not None


# ============================================================================
# Stats Tests
# ============================================================================


class TestAuditLogStats:
    """Tests for audit log statistics."""

    def test_stats_empty(self, audit_log):
        stats = audit_log.get_stats()
        assert stats["total_entries"] == 0
        assert stats["last_hash"] is None
        assert stats["storage_backend"] == "memory"

    @pytest.mark.asyncio
    async def test_stats_after_logging(self, audit_log, system_actor):
        await audit_log.log(
            action=AuditAction.TASK_SUBMITTED,
            actor=system_actor,
            resource_type="task",
            resource_id="t-1",
        )
        stats = audit_log.get_stats()
        assert stats["total_entries"] == 1
        assert stats["last_hash"] is not None


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_system_actor(self):
        actor = create_system_actor()
        assert actor.actor_type == ActorType.SYSTEM
        assert actor.actor_id == "aragora-control-plane"
        assert actor.actor_name == "Aragora Control Plane"

    def test_create_agent_actor(self):
        actor = create_agent_actor("claude-3", "Claude 3")
        assert actor.actor_type == ActorType.AGENT
        assert actor.actor_id == "claude-3"
        assert actor.actor_name == "Claude 3"

    def test_create_agent_actor_default_name(self):
        actor = create_agent_actor("gpt-4")
        assert actor.actor_name == "gpt-4"

    def test_create_user_actor(self):
        actor = create_user_actor("user-1", "Alice", "10.0.0.1")
        assert actor.actor_type == ActorType.USER
        assert actor.actor_id == "user-1"
        assert actor.ip_address == "10.0.0.1"

    def test_create_user_actor_minimal(self):
        actor = create_user_actor("user-2")
        assert actor.actor_name is None
        assert actor.ip_address is None


# ============================================================================
# Global Audit Log Tests
# ============================================================================


class TestGlobalAuditLog:
    """Tests for global audit log management."""

    def test_get_audit_log_default_none(self):
        assert get_audit_log() is None

    def test_set_and_get_audit_log(self):
        log = AuditLog()
        set_audit_log(log)
        assert get_audit_log() is log


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestConvenienceFunctions:
    """Tests for convenience logging functions."""

    @pytest.mark.asyncio
    async def test_log_policy_decision_no_audit_log(self):
        result = await log_policy_decision(
            policy_id="p-1",
            decision="allow",
            task_type="debate",
            reason="Allowed by default",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_log_policy_decision_allow(self):
        log = AuditLog()
        log._redis = None
        set_audit_log(log)

        entry = await log_policy_decision(
            policy_id="p-1",
            decision="allow",
            task_type="debate",
            reason="Allowed",
        )
        assert entry is not None
        assert entry.action == AuditAction.POLICY_DECISION_ALLOW
        assert entry.outcome == "success"

    @pytest.mark.asyncio
    async def test_log_policy_decision_deny(self):
        log = AuditLog()
        log._redis = None
        set_audit_log(log)

        entry = await log_policy_decision(
            policy_id="p-1",
            decision="deny",
            task_type="debate",
            reason="Denied",
        )
        assert entry is not None
        assert entry.action == AuditAction.POLICY_DECISION_DENY
        assert entry.outcome == "failure"

    @pytest.mark.asyncio
    async def test_log_policy_decision_warn(self):
        log = AuditLog()
        log._redis = None
        set_audit_log(log)

        entry = await log_policy_decision(
            policy_id="p-1",
            decision="warn",
            task_type="debate",
            reason="Warning",
        )
        assert entry.action == AuditAction.POLICY_DECISION_WARN
        assert entry.outcome == "success"

    @pytest.mark.asyncio
    async def test_log_deliberation_event_no_log(self):
        result = await log_deliberation_event(
            task_id="t-1",
            event_type="started",
            details={},
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_log_deliberation_event_with_agent(self):
        log = AuditLog()
        log._redis = None
        set_audit_log(log)

        entry = await log_deliberation_event(
            task_id="t-1",
            event_type="consensus",
            details={"confidence": 0.95},
            agent_id="claude-3",
        )
        assert entry is not None
        assert entry.action == AuditAction.DELIBERATION_CONSENSUS
        assert entry.actor.actor_type == ActorType.AGENT

    @pytest.mark.asyncio
    async def test_log_deliberation_started(self):
        log = AuditLog()
        log._redis = None
        set_audit_log(log)

        entry = await log_deliberation_started(
            task_id="t-1",
            question="What is the meaning of life?",
            agents=["claude", "gpt"],
            sla_timeout_seconds=300.0,
        )
        assert entry is not None
        assert entry.action == AuditAction.DELIBERATION_STARTED

    @pytest.mark.asyncio
    async def test_log_deliberation_completed_success(self):
        log = AuditLog()
        log._redis = None
        set_audit_log(log)

        entry = await log_deliberation_completed(
            task_id="t-1",
            success=True,
            consensus_reached=True,
            confidence=0.92,
            duration_seconds=120.0,
            sla_compliant=True,
            winner="claude",
        )
        assert entry is not None
        assert entry.action == AuditAction.DELIBERATION_CONSENSUS
        assert entry.outcome == "success"

    @pytest.mark.asyncio
    async def test_log_deliberation_completed_failure(self):
        log = AuditLog()
        log._redis = None
        set_audit_log(log)

        entry = await log_deliberation_completed(
            task_id="t-1",
            success=False,
            consensus_reached=False,
            confidence=0.3,
            duration_seconds=300.0,
            sla_compliant=False,
        )
        assert entry.action == AuditAction.DELIBERATION_FAILED
        assert entry.outcome == "failure"

    @pytest.mark.asyncio
    async def test_log_deliberation_sla_event(self):
        log = AuditLog()
        log._redis = None
        set_audit_log(log)

        entry = await log_deliberation_sla_event(
            task_id="t-1",
            level="warning",
            elapsed_seconds=200.0,
            timeout_seconds=300.0,
        )
        assert entry is not None
        assert entry.action == AuditAction.DELIBERATION_SLA_WARNING
        assert entry.outcome == "partial"

    @pytest.mark.asyncio
    async def test_log_deliberation_sla_violated(self):
        log = AuditLog()
        log._redis = None
        set_audit_log(log)

        entry = await log_deliberation_sla_event(
            task_id="t-1",
            level="violated",
            elapsed_seconds=310.0,
            timeout_seconds=300.0,
        )
        assert entry.action == AuditAction.DELIBERATION_SLA_VIOLATED
        assert entry.outcome == "failure"
