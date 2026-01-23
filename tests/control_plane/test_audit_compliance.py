"""
Tests for Audit Compliance Export Features.

Tests cover:
- SOC 2 compliance export format
- ISO 27001 audit evidence format
- Syslog (RFC 5424) format
- Retention policy enforcement
"""

import json
import pytest
from datetime import datetime, timezone, timedelta

from aragora.control_plane.audit import (
    ActorType,
    AuditAction,
    AuditActor,
    AuditEntry,
    AuditLog,
    AuditQuery,
    AUDIT_RETENTION_DAYS,
)


class TestAuditEntry:
    """Tests for AuditEntry."""

    def test_create_entry(self):
        """Test creating an audit entry."""
        actor = AuditActor(
            actor_type=ActorType.USER,
            actor_id="user-123",
            actor_name="Test User",
        )
        entry = AuditEntry(
            action=AuditAction.TASK_COMPLETED,
            actor=actor,
            resource_type="task",
            resource_id="task-456",
            outcome="success",
        )

        assert entry.action == AuditAction.TASK_COMPLETED
        assert entry.actor.actor_id == "user-123"
        assert entry.resource_type == "task"
        assert entry.outcome == "success"

    def test_entry_to_dict(self):
        """Test entry serialization."""
        actor = AuditActor(
            actor_type=ActorType.AGENT,
            actor_id="claude-3",
        )
        entry = AuditEntry(
            action=AuditAction.DELIBERATION_CONSENSUS,
            actor=actor,
            resource_type="deliberation",
            resource_id="delib-789",
            details={"confidence": 0.95},
        )

        data = entry.to_dict()

        assert data["action"] == "deliberation.consensus"
        assert data["actor"]["id"] == "claude-3"
        assert data["details"]["confidence"] == 0.95


class TestAuditLog:
    """Tests for AuditLog."""

    @pytest.fixture
    def audit_log(self):
        """Create an in-memory audit log (not connected to Redis)."""
        log = AuditLog(retention_days=30)
        # Don't connect - use local storage
        return log

    @pytest.fixture
    def sample_entries(self, audit_log):
        """Create sample audit entries."""
        entries = []

        # Auth events (CC6.1 / A.9)
        entries.append(
            AuditEntry(
                action=AuditAction.AUTH_LOGIN,
                actor=AuditActor(ActorType.USER, "user-1"),
                resource_type="session",
                resource_id="session-1",
            )
        )

        # Task events (CC6.2 / A.12)
        entries.append(
            AuditEntry(
                action=AuditAction.TASK_COMPLETED,
                actor=AuditActor(ActorType.AGENT, "claude"),
                resource_type="task",
                resource_id="task-1",
            )
        )

        # Config events (CC6.3 / A.14)
        entries.append(
            AuditEntry(
                action=AuditAction.CONFIG_UPDATED,
                actor=AuditActor(ActorType.USER, "admin"),
                resource_type="config",
                resource_id="config-1",
            )
        )

        # Deliberation events (CC7.1)
        entries.append(
            AuditEntry(
                action=AuditAction.DELIBERATION_CONSENSUS,
                actor=AuditActor(ActorType.SYSTEM, "coordinator"),
                resource_type="deliberation",
                resource_id="delib-1",
            )
        )

        # Policy events (A.18)
        entries.append(
            AuditEntry(
                action=AuditAction.POLICY_VIOLATION,
                actor=AuditActor(ActorType.AGENT, "agent-x"),
                resource_type="policy",
                resource_id="policy-1",
                outcome="failure",
            )
        )

        for e in entries:
            audit_log._local_entries.append(e)

        return entries

    @pytest.mark.asyncio
    async def test_export_json(self, audit_log, sample_entries):
        """Test JSON export."""
        query = AuditQuery()
        result = await audit_log.export(query, format="json")

        data = json.loads(result)
        assert len(data) == 5
        assert data[0]["action"] == "auth.login"

    @pytest.mark.asyncio
    async def test_export_csv(self, audit_log, sample_entries):
        """Test CSV export."""
        query = AuditQuery()
        result = await audit_log.export(query, format="csv")

        lines = result.split("\n")
        assert lines[0].startswith("entry_id,timestamp")
        assert len(lines) == 6  # Header + 5 entries

    @pytest.mark.asyncio
    async def test_export_syslog(self, audit_log, sample_entries):
        """Test Syslog RFC 5424 export."""
        query = AuditQuery()
        result = await audit_log.export(query, format="syslog")

        lines = result.split("\n")
        assert len(lines) == 5

        # Check syslog format
        assert lines[0].startswith("<")  # Priority
        assert "aragora" in lines[0]
        assert "[aragora@1" in lines[0]  # Structured data

    @pytest.mark.asyncio
    async def test_export_soc2(self, audit_log, sample_entries):
        """Test SOC 2 compliance export."""
        query = AuditQuery()
        result = await audit_log.export(query, format="soc2")

        data = json.loads(result)

        assert data["report_type"] == "SOC 2 Type II Audit Evidence"
        assert "summary" in data
        assert data["summary"]["total_events"] == 5
        assert "controls" in data

        # Check control categorization
        assert "CC6.1" in data["controls"]  # Auth events
        assert "CC6.2" in data["controls"]  # Task events
        assert "CC7.1" in data["controls"]  # Deliberation events

    @pytest.mark.asyncio
    async def test_export_iso27001(self, audit_log, sample_entries):
        """Test ISO 27001 audit evidence export."""
        query = AuditQuery()
        result = await audit_log.export(query, format="iso27001")

        data = json.loads(result)

        assert data["standard"] == "ISO/IEC 27001:2022"
        assert "executive_summary" in data
        assert data["executive_summary"]["total_log_entries"] == 5
        assert "control_domains" in data

        # Check domain categorization
        assert "A.9" in data["control_domains"]  # Access control
        assert "A.12" in data["control_domains"]  # Operations security
        # A.16 gets policy violations (outcome=failure) per implementation
        assert "A.16" in data["control_domains"]  # Incident management

    @pytest.mark.asyncio
    async def test_export_unsupported_format(self, audit_log, sample_entries):
        """Test error for unsupported format."""
        query = AuditQuery()

        with pytest.raises(ValueError, match="Unsupported export format"):
            await audit_log.export(query, format="xml")


class TestRetentionPolicy:
    """Tests for retention policy enforcement."""

    @pytest.fixture
    def audit_log_with_old_entries(self):
        """Create audit log with entries of varying ages."""
        log = AuditLog(retention_days=30)

        # Add old entries (beyond retention)
        old_time = datetime.now(timezone.utc) - timedelta(days=45)
        for i in range(3):
            entry = AuditEntry(
                action=AuditAction.TASK_COMPLETED,
                actor=AuditActor(ActorType.SYSTEM, "system"),
                resource_type="task",
                resource_id=f"old-task-{i}",
                timestamp=old_time,
            )
            log._local_entries.append(entry)

        # Add recent entries (within retention)
        recent_time = datetime.now(timezone.utc) - timedelta(days=10)
        for i in range(5):
            entry = AuditEntry(
                action=AuditAction.TASK_COMPLETED,
                actor=AuditActor(ActorType.SYSTEM, "system"),
                resource_type="task",
                resource_id=f"recent-task-{i}",
                timestamp=recent_time,
            )
            log._local_entries.append(entry)

        return log

    @pytest.mark.asyncio
    async def test_enforce_retention(self, audit_log_with_old_entries):
        """Test retention enforcement removes old entries."""
        log = audit_log_with_old_entries

        assert len(log._local_entries) == 8

        removed = await log.enforce_retention()

        assert removed == 3
        assert len(log._local_entries) == 5

        # All remaining should be recent
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        for entry in log._local_entries:
            assert entry.timestamp >= cutoff

    @pytest.mark.asyncio
    async def test_get_retention_status(self, audit_log_with_old_entries):
        """Test retention status reporting."""
        log = audit_log_with_old_entries

        status = await log.get_retention_status()

        assert status["retention_days"] == 30
        assert status["total_entries"] == 8
        assert status["oldest_entry"] is not None
        assert status["entries_eligible_for_removal"] == 3

    @pytest.mark.asyncio
    async def test_retention_with_no_entries(self):
        """Test retention on empty log."""
        log = AuditLog()

        removed = await log.enforce_retention()

        assert removed == 0

        status = await log.get_retention_status()
        assert status["total_entries"] == 0
        assert status["oldest_entry"] is None


class TestAuditQuery:
    """Tests for AuditQuery filtering."""

    def test_query_by_action(self):
        """Test filtering by action."""
        query = AuditQuery(actions=[AuditAction.TASK_COMPLETED])

        entry_match = AuditEntry(
            action=AuditAction.TASK_COMPLETED,
            actor=AuditActor(ActorType.AGENT, "agent"),
            resource_type="task",
            resource_id="task-1",
        )
        entry_no_match = AuditEntry(
            action=AuditAction.AUTH_LOGIN,
            actor=AuditActor(ActorType.USER, "user"),
            resource_type="session",
            resource_id="session-1",
        )

        assert query.matches(entry_match) is True
        assert query.matches(entry_no_match) is False

    def test_query_by_actor(self):
        """Test filtering by actor."""
        query = AuditQuery(actor_ids=["user-123"])

        entry_match = AuditEntry(
            action=AuditAction.AUTH_LOGIN,
            actor=AuditActor(ActorType.USER, "user-123"),
            resource_type="session",
            resource_id="session-1",
        )
        entry_no_match = AuditEntry(
            action=AuditAction.AUTH_LOGIN,
            actor=AuditActor(ActorType.USER, "user-456"),
            resource_type="session",
            resource_id="session-2",
        )

        assert query.matches(entry_match) is True
        assert query.matches(entry_no_match) is False

    def test_query_by_time_range(self):
        """Test filtering by time range."""
        now = datetime.now(timezone.utc)
        query = AuditQuery(
            start_time=now - timedelta(hours=1),
            end_time=now,
        )

        entry_in_range = AuditEntry(
            action=AuditAction.TASK_COMPLETED,
            actor=AuditActor(ActorType.AGENT, "agent"),
            resource_type="task",
            resource_id="task-1",
            timestamp=now - timedelta(minutes=30),
        )
        entry_out_of_range = AuditEntry(
            action=AuditAction.TASK_COMPLETED,
            actor=AuditActor(ActorType.AGENT, "agent"),
            resource_type="task",
            resource_id="task-2",
            timestamp=now - timedelta(hours=2),
        )

        assert query.matches(entry_in_range) is True
        assert query.matches(entry_out_of_range) is False


class TestAuditAction:
    """Tests for AuditAction enum."""

    def test_all_action_categories(self):
        """Test all action categories exist."""
        # Agent actions
        assert AuditAction.AGENT_REGISTERED.value.startswith("agent.")

        # Task actions
        assert AuditAction.TASK_SUBMITTED.value.startswith("task.")

        # Deliberation actions
        assert AuditAction.DELIBERATION_STARTED.value.startswith("deliberation.")

        # Policy actions
        assert AuditAction.POLICY_EVALUATED.value.startswith("policy.")

        # Auth actions
        assert AuditAction.AUTH_LOGIN.value.startswith("auth.")

        # Config actions
        assert AuditAction.CONFIG_UPDATED.value.startswith("config.")

        # System actions
        assert AuditAction.SYSTEM_STARTUP.value.startswith("system.")


class TestActorType:
    """Tests for ActorType enum."""

    def test_all_actor_types(self):
        """Test all actor types exist."""
        assert ActorType.AGENT.value == "agent"
        assert ActorType.USER.value == "user"
        assert ActorType.SYSTEM.value == "system"
        assert ActorType.API.value == "api"
        assert ActorType.SCHEDULER.value == "scheduler"
