"""
End-to-end tests for Audit Retention Lifecycle.

Tests the complete audit retention workflow including:
- Audit entry creation and immutability
- Retention policy configuration (90-day, 7-year SOC 2)
- Retention execution and enforcement
- Compliance reporting (SOC 2 Type II, ISO 27001)
- Integrity verification after retention operations

These tests verify SOC 2 CC6.2, CC6.3, CC7.1 compliance for audit trail retention.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Any

import pytest

# Mark all tests in this module as audit and e2e
pytestmark = [pytest.mark.e2e, pytest.mark.audit]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def audit_log():
    """Create an in-memory audit log for testing."""
    from aragora.control_plane.audit import AuditLog

    log = AuditLog(retention_days=30)
    return log


@pytest.fixture
def long_term_audit_log():
    """Create audit log with 7-year retention for SOC 2 compliance."""
    from aragora.control_plane.audit import AuditLog

    log = AuditLog(retention_days=365 * 7)  # 7 years = 2555 days
    return log


@pytest.fixture
def retention_manager():
    """Create a retention policy manager for testing."""
    from aragora.privacy.retention import RetentionPolicyManager

    return RetentionPolicyManager()


@pytest.fixture
def sample_audit_entries(audit_log):
    """Create sample audit entries of varying ages."""
    from aragora.control_plane.audit import (
        AuditEntry,
        AuditActor,
        AuditAction,
        ActorType,
    )

    entries = []

    # Create entries at different timestamps
    timestamps = [
        datetime.now(timezone.utc) - timedelta(days=5),  # Recent
        datetime.now(timezone.utc) - timedelta(days=15),  # Within retention
        datetime.now(timezone.utc) - timedelta(days=45),  # Beyond 30-day retention
        datetime.now(timezone.utc) - timedelta(days=100),  # Well beyond retention
    ]

    actions = [
        AuditAction.AUTH_LOGIN,
        AuditAction.TASK_COMPLETED,
        AuditAction.DELIBERATION_CONSENSUS,
        AuditAction.CONFIG_UPDATED,
    ]

    for i, (ts, action) in enumerate(zip(timestamps, actions)):
        entry = AuditEntry(
            action=action,
            actor=AuditActor(
                actor_type=ActorType.USER if i % 2 == 0 else ActorType.AGENT,
                actor_id=f"actor-{i}",
                actor_name=f"Test Actor {i}",
            ),
            resource_type="test_resource",
            resource_id=f"resource-{i}",
            timestamp=ts,
            workspace_id="ws_test_retention",
            details={"test_index": i},
        )
        audit_log._local_entries.append(entry)
        entries.append(entry)

    return entries


@pytest.fixture
def soc2_compliant_entries(long_term_audit_log):
    """Create audit entries mapped to SOC 2 controls."""
    from aragora.control_plane.audit import (
        AuditEntry,
        AuditActor,
        AuditAction,
        ActorType,
    )

    entries = []
    now = datetime.now(timezone.utc)

    # CC6.1 - Logical and physical access controls
    entries.append(
        AuditEntry(
            action=AuditAction.AUTH_LOGIN,
            actor=AuditActor(ActorType.USER, "user-admin", "Admin User"),
            resource_type="session",
            resource_id="session-001",
            timestamp=now - timedelta(days=30),
            details={"ip": "192.168.1.1", "method": "password"},
        )
    )

    # CC6.2 - System operations
    entries.append(
        AuditEntry(
            action=AuditAction.TASK_COMPLETED,
            actor=AuditActor(ActorType.AGENT, "claude-3", "Claude"),
            resource_type="task",
            resource_id="task-001",
            timestamp=now - timedelta(days=60),
            details={"duration_ms": 1500, "success": True},
        )
    )

    # CC6.3 - Change management
    entries.append(
        AuditEntry(
            action=AuditAction.CONFIG_UPDATED,
            actor=AuditActor(ActorType.USER, "user-dev", "Developer"),
            resource_type="config",
            resource_id="config-rate-limit",
            timestamp=now - timedelta(days=90),
            details={"old_value": 100, "new_value": 200},
        )
    )

    # CC7.1 - System monitoring
    entries.append(
        AuditEntry(
            action=AuditAction.DELIBERATION_CONSENSUS,
            actor=AuditActor(ActorType.SYSTEM, "coordinator", "Debate Coordinator"),
            resource_type="deliberation",
            resource_id="delib-001",
            timestamp=now - timedelta(days=120),
            details={"confidence": 0.92, "rounds": 3},
        )
    )

    # Policy violation (for incident tracking)
    entries.append(
        AuditEntry(
            action=AuditAction.POLICY_VIOLATION,
            actor=AuditActor(ActorType.AGENT, "rogue-agent", "Rogue Agent"),
            resource_type="policy",
            resource_id="policy-001",
            outcome="failure",
            timestamp=now - timedelta(days=150),
            details={"violation": "rate_limit_exceeded"},
        )
    )

    for entry in entries:
        long_term_audit_log._local_entries.append(entry)

    return entries


# ============================================================================
# Audit Entry Lifecycle Tests
# ============================================================================


class TestAuditEntryLifecycle:
    """Tests for audit entry creation and immutability."""

    def test_create_audit_entry(self, audit_log):
        """Test creating an audit entry with all fields."""
        from aragora.control_plane.audit import (
            AuditEntry,
            AuditActor,
            AuditAction,
            ActorType,
        )

        actor = AuditActor(
            actor_type=ActorType.USER,
            actor_id="user-123",
            actor_name="Test User",
            ip_address="192.168.1.100",
        )

        entry = AuditEntry(
            action=AuditAction.AUTH_LOGIN,
            actor=actor,
            resource_type="session",
            resource_id="session-abc",
            workspace_id="ws_test",
            details={"method": "oauth", "provider": "google"},
            outcome="success",
        )

        assert entry.action == AuditAction.AUTH_LOGIN
        assert entry.actor.actor_id == "user-123"
        assert entry.actor.ip_address == "192.168.1.100"
        assert entry.resource_type == "session"
        assert entry.workspace_id == "ws_test"
        assert entry.outcome == "success"
        assert entry.entry_id is not None

    def test_entry_hash_computation(self):
        """Test cryptographic hash computation for tamper detection."""
        from aragora.control_plane.audit import (
            AuditEntry,
            AuditActor,
            AuditAction,
            ActorType,
        )

        entry = AuditEntry(
            action=AuditAction.TASK_COMPLETED,
            actor=AuditActor(ActorType.AGENT, "agent-1"),
            resource_type="task",
            resource_id="task-1",
        )

        hash1 = entry.compute_hash()
        assert hash1 is not None
        assert len(hash1) == 64  # SHA-256 produces 64 hex chars

        # Same entry should produce same hash
        hash2 = entry.compute_hash()
        assert hash1 == hash2

    def test_entry_hash_changes_on_modification(self):
        """Test that hash changes when entry is modified (tamper detection)."""
        from aragora.control_plane.audit import (
            AuditEntry,
            AuditActor,
            AuditAction,
            ActorType,
        )

        entry = AuditEntry(
            action=AuditAction.TASK_COMPLETED,
            actor=AuditActor(ActorType.AGENT, "agent-1"),
            resource_type="task",
            resource_id="task-1",
            details={"original": True},
        )

        original_hash = entry.compute_hash()

        # Modify the entry (simulating tampering)
        entry.details = {"modified": True}
        modified_hash = entry.compute_hash()

        assert original_hash != modified_hash

    def test_entry_serialization_roundtrip(self):
        """Test entry to_dict and from_dict maintain data integrity."""
        from aragora.control_plane.audit import (
            AuditEntry,
            AuditActor,
            AuditAction,
            ActorType,
        )

        original = AuditEntry(
            action=AuditAction.DELIBERATION_CONSENSUS,
            actor=AuditActor(
                actor_type=ActorType.SYSTEM,
                actor_id="coordinator",
                actor_name="Debate Coordinator",
            ),
            resource_type="deliberation",
            resource_id="delib-123",
            workspace_id="ws_test",
            details={"confidence": 0.95, "rounds": 4},
            outcome="success",
        )

        # Roundtrip
        data = original.to_dict()
        restored = AuditEntry.from_dict(data)

        assert restored.action == original.action
        assert restored.actor.actor_id == original.actor.actor_id
        assert restored.resource_type == original.resource_type
        assert restored.details == original.details
        assert restored.workspace_id == original.workspace_id

    def test_entry_with_error_message(self):
        """Test entry creation with failure outcome and error message."""
        from aragora.control_plane.audit import (
            AuditEntry,
            AuditActor,
            AuditAction,
            ActorType,
        )

        entry = AuditEntry(
            action=AuditAction.TASK_FAILED,
            actor=AuditActor(ActorType.AGENT, "agent-fail"),
            resource_type="task",
            resource_id="task-failed",
            outcome="failure",
            error_message="Timeout after 60 seconds",
        )

        assert entry.outcome == "failure"
        assert entry.error_message == "Timeout after 60 seconds"


# ============================================================================
# Retention Policy Tests
# ============================================================================


class TestRetentionPolicies:
    """Tests for retention policy configuration and management."""

    def test_default_retention_policies(self, retention_manager):
        """Test that default retention policies are registered."""
        policies = retention_manager.list_policies()

        policy_ids = [p.id for p in policies]
        assert "default_90_days" in policy_ids
        assert "audit_7_years" in policy_ids

    def test_90_day_retention_policy(self, retention_manager):
        """Test 90-day default retention policy."""
        policy = retention_manager.get_policy("default_90_days")

        assert policy is not None
        assert policy.retention_days == 90
        assert policy.name == "Standard 90-Day Retention"

    def test_7_year_audit_retention_policy(self, retention_manager):
        """Test 7-year audit retention for SOC 2 compliance."""
        from aragora.privacy.retention import RetentionAction

        policy = retention_manager.get_policy("audit_7_years")

        assert policy is not None
        assert policy.retention_days == 365 * 7  # 2555 days
        assert policy.action == RetentionAction.ARCHIVE
        assert "audit_logs" in policy.applies_to

    def test_create_custom_retention_policy(self, retention_manager):
        """Test creating a custom retention policy."""
        from aragora.privacy.retention import RetentionAction

        policy = retention_manager.create_policy(
            name="HIPAA 6-Year Retention",
            retention_days=365 * 6,  # 6 years
            action=RetentionAction.ARCHIVE,
            applies_to=["audit_logs", "phi_records"],
            grace_period_days=30,
            notify_before_days=90,
        )

        assert policy.name == "HIPAA 6-Year Retention"
        assert policy.retention_days == 365 * 6
        assert policy.grace_period_days == 30
        assert policy.notify_before_days == 90
        assert policy.enabled is True

    def test_update_retention_policy(self, retention_manager):
        """Test updating a retention policy."""
        from aragora.privacy.retention import RetentionAction

        # Create policy
        policy = retention_manager.create_policy(
            name="Updateable Policy",
            retention_days=60,
        )

        # Update it
        updated = retention_manager.update_policy(
            policy.id,
            retention_days=120,
            action=RetentionAction.ANONYMIZE,
            enabled=False,
        )

        assert updated.retention_days == 120
        assert updated.action == RetentionAction.ANONYMIZE
        assert updated.enabled is False

    def test_delete_retention_policy(self, retention_manager):
        """Test deleting a retention policy."""
        # Create policy
        policy = retention_manager.create_policy(
            name="Deletable Policy",
            retention_days=30,
        )

        policy_id = policy.id
        assert retention_manager.get_policy(policy_id) is not None

        # Delete
        retention_manager.delete_policy(policy_id)

        assert retention_manager.get_policy(policy_id) is None

    def test_policy_workspace_filtering(self, retention_manager):
        """Test listing policies filtered by workspace."""
        # Create workspace-specific policy
        policy = retention_manager.create_policy(
            name="Workspace Policy",
            retention_days=45,
            workspace_ids=["ws_specific"],
        )

        # List for specific workspace
        ws_policies = retention_manager.list_policies(workspace_id="ws_specific")
        policy_ids = [p.id for p in ws_policies]

        assert policy.id in policy_ids

        # Global policies should also be included (workspace_ids=None)
        assert "default_90_days" in policy_ids


# ============================================================================
# Retention Enforcement Tests
# ============================================================================


class TestRetentionEnforcement:
    """Tests for retention policy enforcement."""

    @pytest.mark.asyncio
    async def test_enforce_retention_removes_old_entries(self, audit_log, sample_audit_entries):
        """Test retention enforcement removes entries beyond retention period."""
        # Audit log has 30-day retention
        assert len(audit_log._local_entries) == 4

        removed = await audit_log.enforce_retention()

        # Should remove entries older than 30 days (2 entries: 45 days and 100 days)
        assert removed == 2
        assert len(audit_log._local_entries) == 2

        # Verify remaining entries are within retention
        cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        for entry in audit_log._local_entries:
            assert entry.timestamp >= cutoff

    @pytest.mark.asyncio
    async def test_enforce_retention_preserves_recent_entries(
        self, audit_log, sample_audit_entries
    ):
        """Test retention enforcement preserves recent entries."""
        await audit_log.enforce_retention()

        # The two recent entries (5 days and 15 days) should remain
        remaining_ages = []
        now = datetime.now(timezone.utc)
        for entry in audit_log._local_entries:
            age_days = (now - entry.timestamp).days
            remaining_ages.append(age_days)

        assert all(age <= 30 for age in remaining_ages)

    @pytest.mark.asyncio
    async def test_retention_status_report(self, audit_log, sample_audit_entries):
        """Test retention status reporting."""
        status = await audit_log.get_retention_status()

        assert status["retention_days"] == 30
        assert status["total_entries"] == 4
        assert status["oldest_entry"] is not None
        assert status["entries_eligible_for_removal"] == 2

    @pytest.mark.asyncio
    async def test_7_year_retention_preserves_entries(
        self, long_term_audit_log, soc2_compliant_entries
    ):
        """Test that 7-year retention preserves all recent entries."""
        # All entries are within 7 years (max is 150 days old)
        assert len(long_term_audit_log._local_entries) == 5

        removed = await long_term_audit_log.enforce_retention()

        # No entries should be removed (all within 7 years)
        assert removed == 0
        assert len(long_term_audit_log._local_entries) == 5

    @pytest.mark.asyncio
    async def test_retention_execution_dry_run(self, retention_manager):
        """Test retention policy execution in dry run mode."""
        report = await retention_manager.execute_policy(
            "default_90_days",
            dry_run=True,
        )

        assert report.policy_id == "default_90_days"
        assert report.items_deleted == 0  # Dry run doesn't actually delete
        assert len(report.errors) == 0

    @pytest.mark.asyncio
    async def test_retention_empty_log(self):
        """Test retention on empty audit log."""
        from aragora.control_plane.audit import AuditLog

        empty_log = AuditLog()
        removed = await empty_log.enforce_retention()

        assert removed == 0

        status = await empty_log.get_retention_status()
        assert status["total_entries"] == 0
        assert status["oldest_entry"] is None


# ============================================================================
# Compliance Export Tests
# ============================================================================


class TestComplianceExport:
    """Tests for compliance report export formats."""

    @pytest.mark.asyncio
    async def test_export_soc2_format(self, long_term_audit_log, soc2_compliant_entries):
        """Test SOC 2 Type II compliance export format."""
        from aragora.control_plane.audit import AuditQuery

        query = AuditQuery()
        result = await long_term_audit_log.export(query, format="soc2")

        data = json.loads(result)

        assert data["report_type"] == "SOC 2 Type II Audit Evidence"
        assert "summary" in data
        assert data["summary"]["total_events"] == 5
        assert "controls" in data

        # Verify control categorization
        controls = data["controls"]
        assert "CC6.1" in controls  # Auth events
        assert "CC6.2" in controls  # Task events
        assert "CC7.1" in controls  # Deliberation events

    @pytest.mark.asyncio
    async def test_export_iso27001_format(self, long_term_audit_log, soc2_compliant_entries):
        """Test ISO 27001 audit evidence export format."""
        from aragora.control_plane.audit import AuditQuery

        query = AuditQuery()
        result = await long_term_audit_log.export(query, format="iso27001")

        data = json.loads(result)

        assert data["standard"] == "ISO/IEC 27001:2022"
        assert "executive_summary" in data
        assert data["executive_summary"]["total_log_entries"] == 5
        assert "control_domains" in data

        # Verify domain categorization
        domains = data["control_domains"]
        assert "A.9" in domains  # Access control
        assert "A.12" in domains  # Operations security
        assert "A.16" in domains  # Incident management (policy violations)

    @pytest.mark.asyncio
    async def test_export_json_format(self, long_term_audit_log, soc2_compliant_entries):
        """Test JSON export format."""
        from aragora.control_plane.audit import AuditQuery

        query = AuditQuery()
        result = await long_term_audit_log.export(query, format="json")

        data = json.loads(result)
        assert len(data) == 5
        assert data[0]["action"] == "auth.login"

    @pytest.mark.asyncio
    async def test_export_csv_format(self, long_term_audit_log, soc2_compliant_entries):
        """Test CSV export format."""
        from aragora.control_plane.audit import AuditQuery

        query = AuditQuery()
        result = await long_term_audit_log.export(query, format="csv")

        lines = result.split("\n")
        assert lines[0].startswith("entry_id,timestamp")
        assert len(lines) == 6  # Header + 5 entries

    @pytest.mark.asyncio
    async def test_export_syslog_format(self, long_term_audit_log, soc2_compliant_entries):
        """Test Syslog RFC 5424 export format."""
        from aragora.control_plane.audit import AuditQuery

        query = AuditQuery()
        result = await long_term_audit_log.export(query, format="syslog")

        lines = result.split("\n")
        assert len(lines) == 5

        # Verify syslog format
        assert lines[0].startswith("<")  # Priority
        assert "aragora" in lines[0]

    @pytest.mark.asyncio
    async def test_export_unsupported_format(self, long_term_audit_log, soc2_compliant_entries):
        """Test error for unsupported export format."""
        from aragora.control_plane.audit import AuditQuery

        query = AuditQuery()

        with pytest.raises(ValueError, match="Unsupported export format"):
            await long_term_audit_log.export(query, format="xml")


# ============================================================================
# Query and Filter Tests
# ============================================================================


class TestAuditQuery:
    """Tests for audit log querying and filtering."""

    def test_query_by_action(self, long_term_audit_log, soc2_compliant_entries):
        """Test filtering entries by action."""
        from aragora.control_plane.audit import AuditQuery, AuditAction

        query = AuditQuery(actions=[AuditAction.AUTH_LOGIN])

        matching = [e for e in long_term_audit_log._local_entries if query.matches(e)]
        assert len(matching) == 1
        assert matching[0].action == AuditAction.AUTH_LOGIN

    def test_query_by_actor_type(self, long_term_audit_log, soc2_compliant_entries):
        """Test filtering entries by actor type."""
        from aragora.control_plane.audit import AuditQuery, ActorType

        query = AuditQuery(actor_types=[ActorType.USER])

        matching = [e for e in long_term_audit_log._local_entries if query.matches(e)]
        assert len(matching) == 2  # AUTH_LOGIN and CONFIG_UPDATED

    def test_query_by_actor_id(self, long_term_audit_log, soc2_compliant_entries):
        """Test filtering entries by actor ID."""
        from aragora.control_plane.audit import AuditQuery

        query = AuditQuery(actor_ids=["claude-3"])

        matching = [e for e in long_term_audit_log._local_entries if query.matches(e)]
        assert len(matching) == 1
        assert matching[0].actor.actor_id == "claude-3"

    def test_query_by_time_range(self, long_term_audit_log, soc2_compliant_entries):
        """Test filtering entries by time range."""
        from aragora.control_plane.audit import AuditQuery

        now = datetime.now(timezone.utc)
        query = AuditQuery(
            start_time=now - timedelta(days=100),
            end_time=now - timedelta(days=50),
        )

        matching = [e for e in long_term_audit_log._local_entries if query.matches(e)]
        # Entries at 60 and 90 days old should match
        assert len(matching) == 2

    def test_query_by_outcome(self, long_term_audit_log, soc2_compliant_entries):
        """Test filtering entries by outcome."""
        from aragora.control_plane.audit import AuditQuery

        query = AuditQuery(outcomes=["failure"])

        matching = [e for e in long_term_audit_log._local_entries if query.matches(e)]
        assert len(matching) == 1
        assert matching[0].outcome == "failure"

    def test_combined_query(self, long_term_audit_log, soc2_compliant_entries):
        """Test combined query with multiple filters."""
        from aragora.control_plane.audit import AuditQuery, ActorType

        now = datetime.now(timezone.utc)
        query = AuditQuery(
            actor_types=[ActorType.USER],
            start_time=now - timedelta(days=100),
        )

        matching = [e for e in long_term_audit_log._local_entries if query.matches(e)]
        # USER entries within 100 days: AUTH_LOGIN (30 days) and CONFIG_UPDATED (90 days)
        assert len(matching) == 2


# ============================================================================
# Integrity Verification Tests
# ============================================================================


class TestIntegrityVerification:
    """Tests for audit log integrity verification."""

    def test_hash_chain_integrity(self, audit_log):
        """Test hash chain integrity for tamper detection."""
        from aragora.control_plane.audit import (
            AuditEntry,
            AuditActor,
            AuditAction,
            ActorType,
        )

        entries = []
        previous_hash = None

        for i in range(5):
            entry = AuditEntry(
                action=AuditAction.TASK_COMPLETED,
                actor=AuditActor(ActorType.AGENT, f"agent-{i}"),
                resource_type="task",
                resource_id=f"task-{i}",
                sequence_number=i,
                previous_hash=previous_hash,
            )
            entry.entry_hash = entry.compute_hash()
            previous_hash = entry.entry_hash
            entries.append(entry)

        # Verify chain integrity
        for i, entry in enumerate(entries):
            # Recompute hash should match stored hash
            assert entry.compute_hash() == entry.entry_hash

            # Previous hash should match
            if i > 0:
                assert entry.previous_hash == entries[i - 1].entry_hash

    def test_tamper_detection(self, audit_log):
        """Test that tampering is detected via hash mismatch."""
        from aragora.control_plane.audit import (
            AuditEntry,
            AuditActor,
            AuditAction,
            ActorType,
        )

        entry = AuditEntry(
            action=AuditAction.AUTH_LOGIN,
            actor=AuditActor(ActorType.USER, "user-1"),
            resource_type="session",
            resource_id="session-1",
        )
        entry.entry_hash = entry.compute_hash()

        original_hash = entry.entry_hash

        # Simulate tampering
        entry.details = {"tampered": True}

        # Hash should no longer match
        new_hash = entry.compute_hash()
        assert new_hash != original_hash

    def test_sequence_number_continuity(self):
        """Test sequence number continuity for completeness check."""
        from aragora.control_plane.audit import (
            AuditEntry,
            AuditActor,
            AuditAction,
            ActorType,
        )

        entries = []
        for i in range(10):
            entry = AuditEntry(
                action=AuditAction.TASK_COMPLETED,
                actor=AuditActor(ActorType.AGENT, f"agent-{i}"),
                resource_type="task",
                resource_id=f"task-{i}",
                sequence_number=i,
            )
            entries.append(entry)

        # Verify sequence continuity
        for i, entry in enumerate(entries):
            assert entry.sequence_number == i

        # Simulate missing entry (gap detection)
        del entries[5]

        # Check for gaps
        has_gap = False
        for i in range(len(entries) - 1):
            if entries[i + 1].sequence_number - entries[i].sequence_number != 1:
                has_gap = True
                break

        assert has_gap is True


# ============================================================================
# Compliance Report Tests
# ============================================================================


class TestComplianceReports:
    """Tests for retention compliance reporting."""

    @pytest.mark.asyncio
    async def test_retention_compliance_report(self, retention_manager):
        """Test generating retention compliance report."""
        report = await retention_manager.get_compliance_report()

        assert "report_period" in report
        assert "total_deletions" in report
        assert "active_policies" in report
        assert report["active_policies"] >= 2  # At least default policies

    @pytest.mark.asyncio
    async def test_compliance_report_date_range(self, retention_manager):
        """Test compliance report with custom date range."""
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=7)
        end = now

        report = await retention_manager.get_compliance_report(
            start_date=start,
            end_date=end,
        )

        assert report["report_period"]["start"] == start.isoformat()
        assert report["report_period"]["end"] == end.isoformat()

    @pytest.mark.asyncio
    async def test_compliance_report_workspace_filter(self, retention_manager):
        """Test compliance report filtered by workspace."""
        report = await retention_manager.get_compliance_report(
            workspace_id="ws_test",
        )

        # Report should be generated even with no matching deletions
        assert "total_deletions" in report

    def test_days_until_expiry_calculation(self, retention_manager):
        """Test calculating days until expiry for resources."""
        policy = retention_manager.get_policy("default_90_days")
        assert policy is not None

        # Resource created 60 days ago should have ~30 days left
        created = datetime.now(timezone.utc) - timedelta(days=60)
        days_left = policy.days_until_expiry(created)

        assert 29 <= days_left <= 31

    def test_is_expired_check(self, retention_manager):
        """Test checking if resource has expired under policy."""
        policy = retention_manager.get_policy("default_90_days")
        assert policy is not None

        # Old resource (100 days)
        old_date = datetime.now(timezone.utc) - timedelta(days=100)
        assert policy.is_expired(old_date) is True

        # Recent resource (10 days)
        recent_date = datetime.now(timezone.utc) - timedelta(days=10)
        assert policy.is_expired(recent_date) is False


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_execute_nonexistent_policy(self, retention_manager):
        """Test executing a nonexistent policy raises error."""
        with pytest.raises(ValueError, match="Policy not found"):
            await retention_manager.execute_policy("nonexistent_policy")

    @pytest.mark.asyncio
    async def test_execute_disabled_policy(self, retention_manager):
        """Test executing a disabled policy returns appropriate response."""
        # Create and disable policy
        policy = retention_manager.create_policy(
            name="Disabled Policy",
            retention_days=30,
        )
        retention_manager.update_policy(policy.id, enabled=False)

        report = await retention_manager.execute_policy(policy.id)

        assert "Policy is disabled" in report.errors

    def test_update_nonexistent_policy(self, retention_manager):
        """Test updating a nonexistent policy raises error."""
        with pytest.raises(ValueError, match="Policy not found"):
            retention_manager.update_policy("nonexistent", retention_days=60)

    @pytest.mark.asyncio
    async def test_retention_with_exclusions(self, retention_manager):
        """Test retention policy with sensitivity exclusions."""
        from aragora.privacy.retention import RetentionAction

        policy = retention_manager.create_policy(
            name="Policy With Exclusions",
            retention_days=30,
            action=RetentionAction.DELETE,
            exclude_sensitivity_levels=["high", "critical"],
            exclude_tags=["legal_hold", "audit_required"],
        )

        assert "high" in policy.exclude_sensitivity_levels
        assert "legal_hold" in policy.exclude_tags


# ============================================================================
# Integration Tests
# ============================================================================


class TestAuditRetentionIntegration:
    """Integration tests combining audit and retention systems."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, audit_log, retention_manager):
        """Test complete audit → retention → compliance lifecycle."""
        from aragora.control_plane.audit import (
            AuditEntry,
            AuditActor,
            AuditAction,
            ActorType,
            AuditQuery,
        )

        # Step 1: Create audit entries
        for i in range(10):
            age_days = i * 10  # 0, 10, 20, 30, 40, 50, 60, 70, 80, 90 days old
            entry = AuditEntry(
                action=AuditAction.TASK_COMPLETED,
                actor=AuditActor(ActorType.AGENT, f"agent-{i}"),
                resource_type="task",
                resource_id=f"task-{i}",
                timestamp=datetime.now(timezone.utc) - timedelta(days=age_days),
            )
            audit_log._local_entries.append(entry)

        assert len(audit_log._local_entries) == 10

        # Step 2: Check retention status
        status = await audit_log.get_retention_status()
        # Entries > 30 days old (40, 50, 60, 70, 80, 90) = 6 eligible
        assert status["entries_eligible_for_removal"] >= 6

        # Step 3: Enforce retention
        removed = await audit_log.enforce_retention()
        assert removed == 6
        # 0, 10, 20, 30 days old remain (30 is at boundary, preserved)
        assert len(audit_log._local_entries) == 4

        # Step 4: Generate compliance export
        query = AuditQuery()
        report = await audit_log.export(query, format="soc2")
        data = json.loads(report)

        assert data["summary"]["total_events"] == 4

    @pytest.mark.asyncio
    async def test_soc2_7_year_retention_workflow(self, long_term_audit_log, retention_manager):
        """Test SOC 2 compliant 7-year retention workflow."""
        from aragora.control_plane.audit import (
            AuditEntry,
            AuditActor,
            AuditAction,
            ActorType,
            AuditQuery,
        )

        # Create entries spanning multiple years
        ages_days = [
            30,  # 1 month
            365,  # 1 year
            730,  # 2 years
            1095,  # 3 years
            1825,  # 5 years
            2190,  # 6 years
        ]

        for i, age in enumerate(ages_days):
            entry = AuditEntry(
                action=AuditAction.AUTH_LOGIN,
                actor=AuditActor(ActorType.USER, f"user-{i}"),
                resource_type="session",
                resource_id=f"session-{i}",
                timestamp=datetime.now(timezone.utc) - timedelta(days=age),
            )
            long_term_audit_log._local_entries.append(entry)

        # Verify all entries within 7-year retention
        status = await long_term_audit_log.get_retention_status()
        assert status["entries_eligible_for_removal"] == 0

        # Enforce retention (should remove nothing)
        removed = await long_term_audit_log.enforce_retention()
        assert removed == 0

        # Generate SOC 2 report
        query = AuditQuery()
        report = await long_term_audit_log.export(query, format="soc2")
        data = json.loads(report)

        assert data["summary"]["total_events"] == 6


__all__ = [
    "TestAuditEntryLifecycle",
    "TestRetentionPolicies",
    "TestRetentionEnforcement",
    "TestComplianceExport",
    "TestAuditQuery",
    "TestIntegrityVerification",
    "TestComplianceReports",
    "TestEdgeCases",
    "TestAuditRetentionIntegration",
]
