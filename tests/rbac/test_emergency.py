"""
Tests for Break-Glass Emergency Access.

Tests cover:
- Access activation with validation
- Access deactivation and expiration
- Security team revocation
- Action recording during emergency
- Post-incident review workflow
- Access history and querying
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from aragora.rbac.emergency import (
    BreakGlassAccess,
    EmergencyAccessRecord,
    EmergencyAccessStatus,
    get_break_glass_access,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def emergency():
    """Fresh BreakGlassAccess instance for each test."""
    return BreakGlassAccess()


@pytest.fixture
def sample_record():
    """Sample EmergencyAccessRecord for testing."""
    now = datetime.now(timezone.utc)
    return EmergencyAccessRecord(
        id="emerg-test123",
        user_id="user-456",
        reason="Testing emergency access functionality",
        status=EmergencyAccessStatus.ACTIVE,
        activated_at=now,
        expires_at=now + timedelta(hours=1),
        ip_address="192.168.1.100",
        user_agent="TestClient/1.0",
        metadata={"ticket": "INC-001"},
    )


# =============================================================================
# EmergencyAccessStatus Tests
# =============================================================================


class TestEmergencyAccessStatus:
    """Tests for EmergencyAccessStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert EmergencyAccessStatus.ACTIVE.value == "active"
        assert EmergencyAccessStatus.EXPIRED.value == "expired"
        assert EmergencyAccessStatus.DEACTIVATED.value == "deactivated"
        assert EmergencyAccessStatus.REVOKED.value == "revoked"

    def test_status_is_string_enum(self):
        """Test status is a string enum."""
        assert isinstance(EmergencyAccessStatus.ACTIVE.value, str)
        assert str(EmergencyAccessStatus.ACTIVE) == "EmergencyAccessStatus.ACTIVE"


# =============================================================================
# EmergencyAccessRecord Tests
# =============================================================================


class TestEmergencyAccessRecord:
    """Tests for EmergencyAccessRecord dataclass."""

    def test_create_record(self, sample_record):
        """Test creating an access record."""
        assert sample_record.id == "emerg-test123"
        assert sample_record.user_id == "user-456"
        assert sample_record.status == EmergencyAccessStatus.ACTIVE
        assert sample_record.review_required is True
        assert sample_record.review_completed is False

    def test_is_active_true(self, sample_record):
        """Test is_active returns True for active record."""
        assert sample_record.is_active is True

    def test_is_active_false_expired(self, sample_record):
        """Test is_active returns False when expired."""
        sample_record.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        assert sample_record.is_active is False

    def test_is_active_false_wrong_status(self, sample_record):
        """Test is_active returns False for non-active status."""
        sample_record.status = EmergencyAccessStatus.DEACTIVATED
        assert sample_record.is_active is False

    def test_duration_minutes(self, sample_record):
        """Test duration_minutes calculation."""
        assert sample_record.duration_minutes == 60  # 1 hour

    def test_to_dict(self, sample_record):
        """Test to_dict conversion."""
        data = sample_record.to_dict()

        assert data["id"] == "emerg-test123"
        assert data["user_id"] == "user-456"
        assert data["status"] == "active"
        assert data["duration_minutes"] == 60
        assert data["is_active"] is True
        assert data["review_required"] is True
        assert data["ip_address"] == "192.168.1.100"
        assert data["metadata"]["ticket"] == "INC-001"

    def test_to_dict_with_deactivation(self, sample_record):
        """Test to_dict includes deactivation info."""
        sample_record.status = EmergencyAccessStatus.DEACTIVATED
        sample_record.deactivated_at = datetime.now(timezone.utc)
        sample_record.deactivated_by = "admin-001"

        data = sample_record.to_dict()
        assert data["status"] == "deactivated"
        assert data["deactivated_by"] == "admin-001"
        assert data["deactivated_at"] is not None

    def test_default_fields(self):
        """Test default field values."""
        now = datetime.now(timezone.utc)
        record = EmergencyAccessRecord(
            id="test",
            user_id="user",
            reason="test reason",
            status=EmergencyAccessStatus.ACTIVE,
            activated_at=now,
            expires_at=now + timedelta(hours=1),
        )

        assert record.deactivated_at is None
        assert record.deactivated_by is None
        assert record.ip_address is None
        assert record.user_agent is None
        assert record.actions_taken == []
        assert record.review_notes is None
        assert record.metadata == {}


# =============================================================================
# BreakGlassAccess Activation Tests
# =============================================================================


@pytest.mark.asyncio
class TestActivation:
    """Tests for activating break-glass access."""

    async def test_activate_success(self, emergency):
        """Test successful activation."""
        access_id = await emergency.activate(
            user_id="user-123",
            reason="Production database corruption incident",
            duration_minutes=60,
        )

        assert access_id.startswith("emerg-")
        assert len(access_id) == 18  # "emerg-" + 12 hex chars

    async def test_activate_with_all_params(self, emergency):
        """Test activation with all parameters."""
        access_id = await emergency.activate(
            user_id="user-123",
            reason="Major outage requiring immediate access",
            duration_minutes=120,
            ip_address="10.0.0.50",
            user_agent="AdminConsole/2.0",
            metadata={"ticket": "INC-999", "severity": "critical"},
        )

        record = await emergency.get_active_access("user-123")
        assert record is not None
        assert record.ip_address == "10.0.0.50"
        assert record.user_agent == "AdminConsole/2.0"
        assert record.metadata["ticket"] == "INC-999"
        assert record.duration_minutes == 120

    async def test_activate_validates_reason_length(self, emergency):
        """Test activation requires reason of at least 10 chars."""
        with pytest.raises(ValueError, match="at least 10 characters"):
            await emergency.activate(
                user_id="user-123",
                reason="short",
                duration_minutes=60,
            )

    async def test_activate_validates_min_duration(self, emergency):
        """Test activation enforces minimum duration."""
        with pytest.raises(ValueError, match="at least 15 minutes"):
            await emergency.activate(
                user_id="user-123",
                reason="Valid reason for emergency access",
                duration_minutes=10,
            )

    async def test_activate_validates_max_duration(self, emergency):
        """Test activation enforces maximum duration (24 hours)."""
        with pytest.raises(ValueError, match="cannot exceed"):
            await emergency.activate(
                user_id="user-123",
                reason="Valid reason for emergency access",
                duration_minutes=25 * 60,  # 25 hours
            )

    async def test_activate_extends_existing_access(self, emergency):
        """Test activating again extends existing access."""
        # First activation
        access_id1 = await emergency.activate(
            user_id="user-123",
            reason="First emergency access request",
            duration_minutes=30,
        )

        record_before = await emergency.get_active_access("user-123")
        expires_before = record_before.expires_at

        # Second activation should extend
        access_id2 = await emergency.activate(
            user_id="user-123",
            reason="Extending emergency access",
            duration_minutes=60,
        )

        # Same access ID
        assert access_id1 == access_id2

        # Expiration extended
        record_after = await emergency.get_active_access("user-123")
        assert record_after.expires_at > expires_before

    async def test_activate_uses_default_duration(self, emergency):
        """Test activation uses default duration when not specified."""
        await emergency.activate(
            user_id="user-123",
            reason="Using default duration for access",
        )

        record = await emergency.get_active_access("user-123")
        assert record.duration_minutes == 60  # Default


# =============================================================================
# BreakGlassAccess Deactivation Tests
# =============================================================================


@pytest.mark.asyncio
class TestDeactivation:
    """Tests for deactivating break-glass access."""

    async def test_deactivate_success(self, emergency):
        """Test successful deactivation."""
        access_id = await emergency.activate(
            user_id="user-123",
            reason="Emergency access for testing",
        )

        record = await emergency.deactivate(access_id)

        assert record.status == EmergencyAccessStatus.DEACTIVATED
        assert record.deactivated_at is not None
        assert record.deactivated_by == "user-123"

    async def test_deactivate_by_other_user(self, emergency):
        """Test deactivation by another user (e.g., admin)."""
        access_id = await emergency.activate(
            user_id="user-123",
            reason="Emergency access to be deactivated",
        )

        record = await emergency.deactivate(access_id, deactivated_by="admin-001")

        assert record.deactivated_by == "admin-001"

    async def test_deactivate_removes_from_active(self, emergency):
        """Test deactivation removes from active records."""
        access_id = await emergency.activate(
            user_id="user-123",
            reason="Emergency access test",
        )

        assert await emergency.is_active("user-123") is True

        await emergency.deactivate(access_id)

        assert await emergency.is_active("user-123") is False

    async def test_deactivate_not_found(self, emergency):
        """Test deactivating non-existent access."""
        with pytest.raises(ValueError, match="not found"):
            await emergency.deactivate("emerg-nonexistent")

    async def test_deactivate_already_inactive(self, emergency):
        """Test deactivating already inactive access."""
        access_id = await emergency.activate(
            user_id="user-123",
            reason="Emergency access test",
        )

        await emergency.deactivate(access_id)

        with pytest.raises(ValueError, match="not active"):
            await emergency.deactivate(access_id)


# =============================================================================
# BreakGlassAccess Revocation Tests
# =============================================================================


@pytest.mark.asyncio
class TestRevocation:
    """Tests for security team revoking access."""

    async def test_revoke_success(self, emergency):
        """Test successful revocation."""
        access_id = await emergency.activate(
            user_id="user-123",
            reason="Emergency access to be revoked",
        )

        record = await emergency.revoke(
            access_id,
            revoked_by="security-admin",
            reason="Policy violation detected",
        )

        assert record.status == EmergencyAccessStatus.REVOKED
        assert record.deactivated_by == "security-admin"
        assert record.metadata["revocation_reason"] == "Policy violation detected"

    async def test_revoke_removes_from_active(self, emergency):
        """Test revocation removes from active records."""
        access_id = await emergency.activate(
            user_id="user-123",
            reason="Emergency access test",
        )

        await emergency.revoke(
            access_id,
            revoked_by="security-admin",
            reason="Security concern",
        )

        assert await emergency.is_active("user-123") is False

    async def test_revoke_not_found(self, emergency):
        """Test revoking non-existent access."""
        with pytest.raises(ValueError, match="not found"):
            await emergency.revoke(
                "emerg-nonexistent",
                revoked_by="security-admin",
                reason="Testing",
            )


# =============================================================================
# BreakGlassAccess Active Check Tests
# =============================================================================


@pytest.mark.asyncio
class TestActiveCheck:
    """Tests for checking active access."""

    async def test_is_active_true(self, emergency):
        """Test is_active returns True for active user."""
        await emergency.activate(
            user_id="user-123",
            reason="Emergency access test",
        )

        assert await emergency.is_active("user-123") is True

    async def test_is_active_false_no_access(self, emergency):
        """Test is_active returns False for user without access."""
        assert await emergency.is_active("user-999") is False

    async def test_is_active_false_after_deactivation(self, emergency):
        """Test is_active returns False after deactivation."""
        access_id = await emergency.activate(
            user_id="user-123",
            reason="Emergency access test",
        )
        await emergency.deactivate(access_id)

        assert await emergency.is_active("user-123") is False

    async def test_get_active_access_returns_record(self, emergency):
        """Test get_active_access returns the record."""
        await emergency.activate(
            user_id="user-123",
            reason="Emergency access test",
        )

        record = await emergency.get_active_access("user-123")
        assert record is not None
        assert record.user_id == "user-123"

    async def test_get_active_access_returns_none(self, emergency):
        """Test get_active_access returns None for inactive user."""
        record = await emergency.get_active_access("user-999")
        assert record is None


# =============================================================================
# BreakGlassAccess Action Recording Tests
# =============================================================================


@pytest.mark.asyncio
class TestActionRecording:
    """Tests for recording actions during emergency access."""

    async def test_record_action(self, emergency):
        """Test recording an action."""
        await emergency.activate(
            user_id="user-123",
            reason="Emergency database repair",
        )

        await emergency.record_action(
            user_id="user-123",
            action="delete",
            resource_type="backup",
            resource_id="backup-001",
            details={"size_mb": 500},
        )

        record = await emergency.get_active_access("user-123")
        assert len(record.actions_taken) == 1

        action = record.actions_taken[0]
        assert action["action"] == "delete"
        assert action["resource_type"] == "backup"
        assert action["resource_id"] == "backup-001"
        assert action["details"]["size_mb"] == 500
        assert "timestamp" in action

    async def test_record_multiple_actions(self, emergency):
        """Test recording multiple actions."""
        await emergency.activate(
            user_id="user-123",
            reason="Emergency database repair",
        )

        await emergency.record_action(
            user_id="user-123",
            action="read",
            resource_type="config",
        )
        await emergency.record_action(
            user_id="user-123",
            action="update",
            resource_type="config",
        )
        await emergency.record_action(
            user_id="user-123",
            action="restart",
            resource_type="service",
        )

        record = await emergency.get_active_access("user-123")
        assert len(record.actions_taken) == 3

    async def test_record_action_no_active_access(self, emergency):
        """Test recording action without active access does nothing."""
        await emergency.record_action(
            user_id="user-999",
            action="delete",
            resource_type="backup",
        )

        # Should not raise, just do nothing
        record = await emergency.get_active_access("user-999")
        assert record is None


# =============================================================================
# BreakGlassAccess Expiration Tests
# =============================================================================


@pytest.mark.asyncio
class TestExpiration:
    """Tests for access expiration."""

    async def test_expire_old_access(self, emergency):
        """Test expiring old access records."""
        # Activate with past expiration
        access_id = await emergency.activate(
            user_id="user-123",
            reason="Emergency access that will expire",
            duration_minutes=15,
        )

        # Manually set expiration to past
        record = await emergency.get_active_access("user-123")
        record.expires_at = datetime.now(timezone.utc) - timedelta(minutes=5)

        # Run expiration
        count = await emergency.expire_old_access()

        assert count == 1
        assert await emergency.is_active("user-123") is False

        # Check status
        expired_record = emergency._all_records.get(access_id)
        assert expired_record.status == EmergencyAccessStatus.EXPIRED

    async def test_expire_only_old_records(self, emergency):
        """Test only old records are expired."""
        # Create one expired and one active
        await emergency.activate(
            user_id="user-old",
            reason="Old access that will expire",
            duration_minutes=15,
        )
        await emergency.activate(
            user_id="user-new",
            reason="New access that won't expire",
            duration_minutes=60,
        )

        # Make one expired
        record = await emergency.get_active_access("user-old")
        record.expires_at = datetime.now(timezone.utc) - timedelta(minutes=5)

        count = await emergency.expire_old_access()

        assert count == 1
        assert await emergency.is_active("user-old") is False
        assert await emergency.is_active("user-new") is True


# =============================================================================
# BreakGlassAccess History Tests
# =============================================================================


@pytest.mark.asyncio
class TestHistory:
    """Tests for access history."""

    async def test_get_history_all(self, emergency):
        """Test getting all history."""
        # Create multiple records
        await emergency.activate(
            user_id="user-1",
            reason="Emergency access 1",
        )
        await emergency.activate(
            user_id="user-2",
            reason="Emergency access 2",
        )
        await emergency.activate(
            user_id="user-3",
            reason="Emergency access 3",
        )

        history = await emergency.get_history()

        assert len(history) == 3
        # Most recent first
        assert history[0].user_id == "user-3"
        assert history[-1].user_id == "user-1"

    async def test_get_history_by_user(self, emergency):
        """Test getting history for specific user."""
        await emergency.activate(
            user_id="user-1",
            reason="Emergency access 1",
        )
        access_id = await emergency.activate(
            user_id="user-2",
            reason="Emergency access 2",
        )
        await emergency.deactivate(access_id)
        await emergency.activate(
            user_id="user-2",
            reason="Emergency access 2 again",
        )

        history = await emergency.get_history(user_id="user-2")

        assert len(history) == 2
        assert all(r.user_id == "user-2" for r in history)

    async def test_get_history_with_limit(self, emergency):
        """Test history limit."""
        for i in range(10):
            await emergency.activate(
                user_id=f"user-{i}",
                reason=f"Emergency access {i}",
            )

        history = await emergency.get_history(limit=5)

        assert len(history) == 5


# =============================================================================
# BreakGlassAccess Review Tests
# =============================================================================


@pytest.mark.asyncio
class TestReview:
    """Tests for post-incident review."""

    async def test_complete_review(self, emergency):
        """Test completing a review."""
        access_id = await emergency.activate(
            user_id="user-123",
            reason="Emergency that needs review",
        )
        await emergency.deactivate(access_id)

        record = await emergency.complete_review(
            access_id=access_id,
            reviewer_id="manager-001",
            notes="Actions were appropriate. No policy violations.",
        )

        assert record.review_completed is True
        assert record.review_notes == "Actions were appropriate. No policy violations."
        assert record.metadata["reviewed_by"] == "manager-001"
        assert "reviewed_at" in record.metadata

    async def test_review_not_found(self, emergency):
        """Test reviewing non-existent access."""
        with pytest.raises(ValueError, match="not found"):
            await emergency.complete_review(
                access_id="emerg-nonexistent",
                reviewer_id="manager-001",
                notes="Test",
            )

    async def test_cannot_review_active_access(self, emergency):
        """Test cannot review while access is still active."""
        access_id = await emergency.activate(
            user_id="user-123",
            reason="Emergency access still active",
        )

        with pytest.raises(ValueError, match="Cannot review active"):
            await emergency.complete_review(
                access_id=access_id,
                reviewer_id="manager-001",
                notes="Test",
            )


# =============================================================================
# BreakGlassAccess Singleton Tests
# =============================================================================


class TestSingleton:
    """Tests for singleton instance."""

    def test_get_break_glass_access(self):
        """Test getting singleton instance."""
        instance1 = get_break_glass_access()
        instance2 = get_break_glass_access()

        assert instance1 is instance2
        assert isinstance(instance1, BreakGlassAccess)


# =============================================================================
# BreakGlassAccess Constants Tests
# =============================================================================


class TestConstants:
    """Tests for class constants."""

    def test_duration_limits(self):
        """Test duration limit constants."""
        assert BreakGlassAccess.DEFAULT_DURATION_MINUTES == 60
        assert BreakGlassAccess.MAX_DURATION_MINUTES == 24 * 60
        assert BreakGlassAccess.MIN_DURATION_MINUTES == 15

    def test_emergency_permissions(self):
        """Test emergency permissions list."""
        perms = BreakGlassAccess.EMERGENCY_PERMISSIONS

        assert "admin" in perms
        assert "debates:*" in perms
        assert "backups:*" in perms
        assert "audit_log:read" in perms


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for full workflow."""

    async def test_full_emergency_workflow(self, emergency):
        """Test complete emergency access workflow."""
        # 1. Activate
        access_id = await emergency.activate(
            user_id="user-123",
            reason="Production database corruption requiring immediate access",
            duration_minutes=60,
            ip_address="10.0.0.100",
            metadata={"incident": "INC-001"},
        )

        assert await emergency.is_active("user-123") is True

        # 2. Record actions
        await emergency.record_action(
            user_id="user-123",
            action="read",
            resource_type="database",
            resource_id="prod-db-01",
        )
        await emergency.record_action(
            user_id="user-123",
            action="repair",
            resource_type="database",
            resource_id="prod-db-01",
            details={"tables_repaired": ["users", "orders"]},
        )

        # 3. Check actions recorded
        record = await emergency.get_active_access("user-123")
        assert len(record.actions_taken) == 2

        # 4. Deactivate
        record = await emergency.deactivate(access_id)
        assert record.status == EmergencyAccessStatus.DEACTIVATED
        assert await emergency.is_active("user-123") is False

        # 5. Complete review
        record = await emergency.complete_review(
            access_id=access_id,
            reviewer_id="security-lead",
            notes="Actions appropriate for the incident. No policy violations.",
        )
        assert record.review_completed is True

        # 6. Verify in history
        history = await emergency.get_history(user_id="user-123")
        assert len(history) == 1
        assert history[0].review_completed is True

    async def test_multiple_users_workflow(self, emergency):
        """Test multiple users with concurrent emergency access."""
        # Activate for multiple users
        id1 = await emergency.activate(
            user_id="user-1",
            reason="Emergency access for user 1",
        )
        id2 = await emergency.activate(
            user_id="user-2",
            reason="Emergency access for user 2",
        )

        assert await emergency.is_active("user-1") is True
        assert await emergency.is_active("user-2") is True

        # Deactivate one
        await emergency.deactivate(id1)

        assert await emergency.is_active("user-1") is False
        assert await emergency.is_active("user-2") is True

        # Revoke the other
        await emergency.revoke(id2, revoked_by="security", reason="Test")

        assert await emergency.is_active("user-2") is False
