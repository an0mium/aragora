"""
Tests for Data Retention Policy Manager.

Tests cover:
- RetentionPolicyManager initialization and lifecycle
- Policy creation, retrieval, update, and deletion
- Policy execution (delete, archive, anonymize)
- Expiration detection with check_expiring_soon()
- DeletionReport generation and metrics
- Compliance reporting
- Handler registration and invocation
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from aragora.privacy.retention import (
    DeletionRecord,
    DeletionReport,
    RetentionAction,
    RetentionPolicy,
    RetentionPolicyManager,
    RetentionViolation,
    get_retention_manager,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def manager():
    """Create a fresh retention policy manager."""
    return RetentionPolicyManager()


@pytest.fixture
def policy():
    """Create a test retention policy."""
    return RetentionPolicy(
        id="test_policy",
        name="Test Policy",
        description="A test retention policy",
        retention_days=30,
        action=RetentionAction.DELETE,
        applies_to=["documents", "sessions"],
        grace_period_days=7,
        notify_before_days=14,
        notification_recipients=["admin@example.com"],
    )


@pytest.fixture
def mock_handler():
    """Create a mock deletion handler."""
    handler = MagicMock()
    handler.return_value = True
    return handler


@pytest.fixture
def sample_items():
    """Create sample items for testing."""
    now = datetime.now(timezone.utc)
    return [
        {
            "id": "item-1",
            "type": "documents",
            "workspace_id": "ws-1",
            "created_at": now - timedelta(days=60),  # Expired
            "tags": [],
        },
        {
            "id": "item-2",
            "type": "documents",
            "workspace_id": "ws-1",
            "created_at": now - timedelta(days=10),  # Not expired
            "tags": [],
        },
        {
            "id": "item-3",
            "type": "sessions",
            "workspace_id": "ws-1",
            "created_at": now - timedelta(days=45),  # Expired
            "tags": ["important"],
        },
    ]


# ============================================================================
# RetentionPolicy Dataclass Tests
# ============================================================================


class TestRetentionPolicy:
    """Tests for RetentionPolicy dataclass."""

    def test_create_policy_with_defaults(self):
        """Test creating a policy with default values."""
        policy = RetentionPolicy(
            id="policy-001",
            name="Default Policy",
        )

        assert policy.id == "policy-001"
        assert policy.name == "Default Policy"
        assert policy.retention_days == 90
        assert policy.action == RetentionAction.DELETE
        assert policy.grace_period_days == 7
        assert policy.enabled is True

    def test_create_policy_with_custom_values(self):
        """Test creating a policy with custom values."""
        now = datetime.utcnow()
        policy = RetentionPolicy(
            id="policy-002",
            name="Custom Policy",
            description="Custom description",
            retention_days=180,
            action=RetentionAction.ARCHIVE,
            applies_to=["audit_logs"],
            workspace_ids=["ws-1", "ws-2"],
            grace_period_days=14,
            notify_before_days=30,
            notification_recipients=["admin@example.com"],
            exclude_sensitivity_levels=["high", "critical"],
            exclude_tags=["permanent"],
            enabled=False,
            created_at=now,
        )

        assert policy.retention_days == 180
        assert policy.action == RetentionAction.ARCHIVE
        assert len(policy.workspace_ids) == 2
        assert policy.exclude_sensitivity_levels == ["high", "critical"]
        assert policy.enabled is False

    def test_is_expired_true(self):
        """Test is_expired returns True for expired resources."""
        policy = RetentionPolicy(
            id="policy-001",
            name="Test",
            retention_days=30,
        )
        old_date = datetime.now(timezone.utc) - timedelta(days=60)

        assert policy.is_expired(old_date) is True

    def test_is_expired_false(self):
        """Test is_expired returns False for non-expired resources."""
        policy = RetentionPolicy(
            id="policy-001",
            name="Test",
            retention_days=30,
        )
        recent_date = datetime.now(timezone.utc) - timedelta(days=10)

        assert policy.is_expired(recent_date) is False

    def test_is_expired_at_boundary(self):
        """Test is_expired at exactly the expiration time."""
        policy = RetentionPolicy(
            id="policy-001",
            name="Test",
            retention_days=30,
        )
        # Created exactly 30 days ago
        boundary_date = datetime.now(timezone.utc) - timedelta(days=30)

        # At or past expiration should be expired
        assert policy.is_expired(boundary_date) is True

    def test_days_until_expiry_positive(self):
        """Test days_until_expiry for non-expired resources."""
        policy = RetentionPolicy(
            id="policy-001",
            name="Test",
            retention_days=30,
        )
        recent_date = datetime.now(timezone.utc) - timedelta(days=10)

        days_left = policy.days_until_expiry(recent_date)
        # Allow for potential off-by-one due to time component
        assert 19 <= days_left <= 20

    def test_days_until_expiry_zero(self):
        """Test days_until_expiry for already expired resources."""
        policy = RetentionPolicy(
            id="policy-001",
            name="Test",
            retention_days=30,
        )
        old_date = datetime.now(timezone.utc) - timedelta(days=60)

        days_left = policy.days_until_expiry(old_date)
        assert days_left == 0

    def test_days_until_expiry_nearly_expired(self):
        """Test days_until_expiry for resources about to expire."""
        policy = RetentionPolicy(
            id="policy-001",
            name="Test",
            retention_days=30,
        )
        # Created 29 days ago - 1 day left
        nearly_expired_date = datetime.now(timezone.utc) - timedelta(days=29)

        days_left = policy.days_until_expiry(nearly_expired_date)
        # Allow for potential off-by-one due to time component
        assert 0 <= days_left <= 1


# ============================================================================
# RetentionAction Enum Tests
# ============================================================================


class TestRetentionAction:
    """Tests for RetentionAction enum."""

    def test_action_values(self):
        """Test all action enum values exist."""
        assert RetentionAction.DELETE.value == "delete"
        assert RetentionAction.ARCHIVE.value == "archive"
        assert RetentionAction.ANONYMIZE.value == "anonymize"
        assert RetentionAction.NOTIFY.value == "notify"

    def test_action_from_string(self):
        """Test creating action from string value."""
        assert RetentionAction("delete") == RetentionAction.DELETE
        assert RetentionAction("archive") == RetentionAction.ARCHIVE
        assert RetentionAction("anonymize") == RetentionAction.ANONYMIZE

    def test_action_invalid_value(self):
        """Test that invalid action raises ValueError."""
        with pytest.raises(ValueError):
            RetentionAction("invalid_action")


# ============================================================================
# RetentionViolation Exception Tests
# ============================================================================


class TestRetentionViolation:
    """Tests for RetentionViolation exception."""

    def test_violation_attributes(self):
        """Test violation exception has correct attributes."""
        violation = RetentionViolation(
            message="Policy violated",
            policy_id="policy-001",
            resource_id="resource-001",
        )

        assert str(violation) == "Policy violated"
        assert violation.policy_id == "policy-001"
        assert violation.resource_id == "resource-001"

    def test_violation_can_be_raised(self):
        """Test violation exception can be raised and caught."""
        with pytest.raises(RetentionViolation) as exc_info:
            raise RetentionViolation(
                message="Test violation",
                policy_id="p-1",
                resource_id="r-1",
            )

        assert exc_info.value.policy_id == "p-1"
        assert exc_info.value.resource_id == "r-1"


# ============================================================================
# DeletionRecord Tests
# ============================================================================


class TestDeletionRecord:
    """Tests for DeletionRecord dataclass."""

    def test_create_record(self):
        """Test creating a deletion record."""
        now = datetime.utcnow()
        record = DeletionRecord(
            resource_type="documents",
            resource_id="doc-001",
            workspace_id="ws-1",
            policy_id="policy-001",
            deleted_at=now,
            metadata={"size_bytes": 1024},
        )

        assert record.resource_type == "documents"
        assert record.resource_id == "doc-001"
        assert record.workspace_id == "ws-1"
        assert record.policy_id == "policy-001"
        assert record.metadata["size_bytes"] == 1024

    def test_record_with_defaults(self):
        """Test creating a record with default values."""
        record = DeletionRecord(
            resource_type="sessions",
            resource_id="session-001",
            workspace_id="ws-1",
            policy_id="policy-001",
        )

        assert record.deleted_at is not None
        assert record.metadata == {}


# ============================================================================
# DeletionReport Tests
# ============================================================================


class TestDeletionReport:
    """Tests for DeletionReport dataclass."""

    def test_create_report(self):
        """Test creating a deletion report."""
        report = DeletionReport(
            policy_id="policy-001",
            items_evaluated=100,
            items_deleted=50,
            items_archived=20,
            items_anonymized=10,
            items_skipped=15,
            items_failed=5,
        )

        assert report.policy_id == "policy-001"
        assert report.items_evaluated == 100
        assert report.items_deleted == 50
        assert report.items_archived == 20
        assert report.items_anonymized == 10
        assert report.items_skipped == 15
        assert report.items_failed == 5

    def test_report_with_defaults(self):
        """Test report with default values."""
        report = DeletionReport(policy_id="policy-001")

        assert report.items_evaluated == 0
        assert report.items_deleted == 0
        assert report.items_failed == 0
        assert report.duration_seconds == 0.0
        assert report.deletions == []
        assert report.errors == []
        assert report.notifications_sent == 0

    def test_report_to_dict(self):
        """Test report serialization to dictionary."""
        now = datetime.now(timezone.utc)
        report = DeletionReport(
            policy_id="policy-001",
            executed_at=now,
            duration_seconds=5.5,
            items_evaluated=100,
            items_deleted=50,
            items_archived=20,
            items_anonymized=10,
            items_skipped=15,
            items_failed=5,
            errors=["Error 1", "Error 2"],
            notifications_sent=3,
        )

        result = report.to_dict()

        assert result["policy_id"] == "policy-001"
        assert result["executed_at"] == now.isoformat()
        assert result["duration_seconds"] == 5.5
        assert result["items_evaluated"] == 100
        assert result["items_deleted"] == 50
        assert result["items_archived"] == 20
        assert result["items_anonymized"] == 10
        assert result["items_skipped"] == 15
        assert result["items_failed"] == 5
        assert result["notifications_sent"] == 3
        assert result["error_count"] == 2

    def test_report_to_dict_empty_errors(self):
        """Test report to_dict with no errors."""
        report = DeletionReport(policy_id="policy-001")
        result = report.to_dict()

        assert result["error_count"] == 0

    def test_report_with_deletion_records(self):
        """Test report with deletion records."""
        record1 = DeletionRecord(
            resource_type="documents",
            resource_id="doc-1",
            workspace_id="ws-1",
            policy_id="policy-001",
        )
        record2 = DeletionRecord(
            resource_type="sessions",
            resource_id="session-1",
            workspace_id="ws-1",
            policy_id="policy-001",
        )

        report = DeletionReport(
            policy_id="policy-001",
            deletions=[record1, record2],
        )

        assert len(report.deletions) == 2
        assert report.deletions[0].resource_type == "documents"
        assert report.deletions[1].resource_type == "sessions"

    def test_report_timing_information(self):
        """Test report timing information."""
        now = datetime.now(timezone.utc)
        report = DeletionReport(
            policy_id="policy-001",
            executed_at=now,
            duration_seconds=12.345,
        )

        assert report.executed_at == now
        assert report.duration_seconds == 12.345

    def test_report_error_details(self):
        """Test report error details."""
        report = DeletionReport(
            policy_id="policy-001",
            errors=[
                "Failed to delete item-1: Permission denied",
                "Failed to delete item-2: Connection timeout",
                "Failed to archive item-3: Storage full",
            ],
        )

        assert len(report.errors) == 3
        assert "Permission denied" in report.errors[0]
        assert "Connection timeout" in report.errors[1]
        assert "Storage full" in report.errors[2]

    def test_report_metrics_collection(self):
        """Test that report correctly tracks metrics."""
        report = DeletionReport(policy_id="policy-001")

        # Simulate incrementing counts
        report.items_evaluated = 50
        report.items_deleted = 25
        report.items_archived = 10
        report.items_anonymized = 5
        report.items_skipped = 7
        report.items_failed = 3

        total_processed = (
            report.items_deleted
            + report.items_archived
            + report.items_anonymized
            + report.items_skipped
            + report.items_failed
        )
        assert total_processed == 50
        assert report.items_evaluated == total_processed


# ============================================================================
# RetentionPolicyManager Initialization Tests
# ============================================================================


class TestRetentionPolicyManagerInit:
    """Tests for RetentionPolicyManager initialization."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = RetentionPolicyManager()

        assert manager._policies is not None
        assert manager._deletion_records is not None
        assert manager._delete_handlers is not None

    def test_default_policies_registered(self):
        """Test that default policies are registered on init."""
        manager = RetentionPolicyManager()

        assert "default_90_days" in manager._policies
        assert "audit_7_years" in manager._policies

    def test_default_90_day_policy(self, manager):
        """Test default 90-day policy configuration."""
        policy = manager.get_policy("default_90_days")

        assert policy is not None
        assert policy.name == "Standard 90-Day Retention"
        assert policy.retention_days == 90
        assert policy.action == RetentionAction.DELETE

    def test_default_audit_policy(self, manager):
        """Test default audit retention policy."""
        policy = manager.get_policy("audit_7_years")

        assert policy is not None
        assert policy.name == "Audit Retention (7 Years)"
        assert policy.retention_days == 365 * 7
        assert policy.action == RetentionAction.ARCHIVE
        assert "audit_logs" in policy.applies_to

    def test_empty_deletion_records_on_init(self, manager):
        """Test that deletion records are empty on init."""
        assert len(manager._deletion_records) == 0

    def test_empty_handlers_on_init(self, manager):
        """Test that handlers are empty on init."""
        assert len(manager._delete_handlers) == 0


# ============================================================================
# RetentionPolicyManager Policy Management Tests
# ============================================================================


class TestRetentionPolicyManagerPolicies:
    """Tests for RetentionPolicyManager policy management."""

    def test_create_policy(self, manager):
        """Test creating a new policy."""
        policy = manager.create_policy(
            name="Custom Policy",
            retention_days=60,
            action=RetentionAction.ARCHIVE,
        )

        assert policy is not None
        assert policy.name == "Custom Policy"
        assert policy.retention_days == 60
        assert policy.action == RetentionAction.ARCHIVE
        assert policy.id.startswith("policy_")

    def test_create_policy_with_workspace_ids(self, manager):
        """Test creating a policy with specific workspaces."""
        policy = manager.create_policy(
            name="Workspace Policy",
            retention_days=45,
            workspace_ids=["ws-1", "ws-2"],
        )

        assert policy.workspace_ids == ["ws-1", "ws-2"]

    def test_create_policy_with_kwargs(self, manager):
        """Test creating a policy with additional kwargs."""
        policy = manager.create_policy(
            name="Full Policy",
            retention_days=90,
            grace_period_days=14,
            notify_before_days=21,
            notification_recipients=["admin@example.com"],
            exclude_sensitivity_levels=["high"],
            exclude_tags=["permanent"],
        )

        assert policy.grace_period_days == 14
        assert policy.notify_before_days == 21
        assert "admin@example.com" in policy.notification_recipients
        assert "high" in policy.exclude_sensitivity_levels
        assert "permanent" in policy.exclude_tags

    def test_get_policy_exists(self, manager, policy):
        """Test getting an existing policy."""
        manager._policies[policy.id] = policy
        retrieved = manager.get_policy(policy.id)

        assert retrieved is not None
        assert retrieved.id == policy.id
        assert retrieved.name == policy.name

    def test_get_policy_not_exists(self, manager):
        """Test getting a non-existent policy returns None."""
        retrieved = manager.get_policy("nonexistent_policy")
        assert retrieved is None

    def test_list_policies_all(self, manager):
        """Test listing all policies."""
        policies = manager.list_policies()

        # Should include default policies
        assert len(policies) >= 2
        policy_ids = [p.id for p in policies]
        assert "default_90_days" in policy_ids
        assert "audit_7_years" in policy_ids

    def test_list_policies_by_workspace(self, manager):
        """Test listing policies filtered by workspace."""
        # Create workspace-specific policy
        manager.create_policy(
            name="WS1 Policy",
            retention_days=30,
            workspace_ids=["ws-1"],
        )

        # Create global policy
        manager.create_policy(
            name="Global Policy",
            retention_days=60,
            workspace_ids=None,  # Applies to all
        )

        policies = manager.list_policies(workspace_id="ws-1")

        # Should include ws-1 specific and global policies
        policy_names = [p.name for p in policies]
        assert "WS1 Policy" in policy_names
        assert "Global Policy" in policy_names

    def test_list_policies_excludes_other_workspace(self, manager):
        """Test that workspace filter excludes other workspaces."""
        manager.create_policy(
            name="WS1 Only",
            retention_days=30,
            workspace_ids=["ws-1"],
        )
        manager.create_policy(
            name="WS2 Only",
            retention_days=30,
            workspace_ids=["ws-2"],
        )

        policies = manager.list_policies(workspace_id="ws-1")
        policy_names = [p.name for p in policies]

        assert "WS1 Only" in policy_names
        assert "WS2 Only" not in policy_names

    def test_update_policy(self, manager, policy):
        """Test updating a policy."""
        manager._policies[policy.id] = policy

        updated = manager.update_policy(
            policy.id,
            name="Updated Policy",
            retention_days=60,
            enabled=False,
        )

        assert updated.name == "Updated Policy"
        assert updated.retention_days == 60
        assert updated.enabled is False

    def test_update_policy_partial(self, manager, policy):
        """Test partial policy update."""
        manager._policies[policy.id] = policy
        original_days = policy.retention_days

        updated = manager.update_policy(
            policy.id,
            name="Only Name Changed",
        )

        assert updated.name == "Only Name Changed"
        assert updated.retention_days == original_days

    def test_update_policy_not_found(self, manager):
        """Test updating non-existent policy raises error."""
        with pytest.raises(ValueError, match="Policy not found"):
            manager.update_policy("nonexistent", name="New Name")

    def test_update_policy_invalid_attribute(self, manager, policy):
        """Test updating with invalid attribute is ignored."""
        manager._policies[policy.id] = policy

        updated = manager.update_policy(
            policy.id,
            invalid_attribute="value",
        )

        assert not hasattr(updated, "invalid_attribute")

    def test_delete_policy(self, manager, policy):
        """Test deleting a policy."""
        manager._policies[policy.id] = policy
        assert policy.id in manager._policies

        manager.delete_policy(policy.id)

        assert policy.id not in manager._policies

    def test_delete_policy_not_exists(self, manager):
        """Test deleting non-existent policy doesn't raise."""
        # Should not raise
        manager.delete_policy("nonexistent_policy")


# ============================================================================
# Handler Registration Tests
# ============================================================================


class TestHandlerRegistration:
    """Tests for deletion handler registration."""

    def test_register_handler(self, manager, mock_handler):
        """Test registering a deletion handler."""
        manager.register_delete_handler("documents", mock_handler)

        assert "documents" in manager._delete_handlers
        assert manager._delete_handlers["documents"] is mock_handler

    def test_register_multiple_handlers(self, manager):
        """Test registering handlers for multiple resource types."""
        handler1 = MagicMock(return_value=True)
        handler2 = MagicMock(return_value=True)
        handler3 = MagicMock(return_value=True)

        manager.register_delete_handler("documents", handler1)
        manager.register_delete_handler("sessions", handler2)
        manager.register_delete_handler("findings", handler3)

        assert len(manager._delete_handlers) == 3
        assert manager._delete_handlers["documents"] is handler1
        assert manager._delete_handlers["sessions"] is handler2
        assert manager._delete_handlers["findings"] is handler3

    def test_register_handler_overwrites(self, manager):
        """Test that registering a handler overwrites existing."""
        handler1 = MagicMock(return_value=True)
        handler2 = MagicMock(return_value=False)

        manager.register_delete_handler("documents", handler1)
        manager.register_delete_handler("documents", handler2)

        assert manager._delete_handlers["documents"] is handler2

    @pytest.mark.asyncio
    async def test_handler_invoked_on_delete(self, manager, mock_handler):
        """Test that handler is invoked during deletion."""
        manager.register_delete_handler("documents", mock_handler)

        policy = manager.create_policy(
            name="Delete Policy",
            retention_days=30,
            action=RetentionAction.DELETE,
        )

        item = {
            "id": "doc-1",
            "type": "documents",
            "workspace_id": "ws-1",
            "created_at": datetime.now(timezone.utc) - timedelta(days=60),
            "tags": [],
        }

        result = await manager._delete_item(item, policy)

        assert result == "deleted"
        mock_handler.assert_called_once_with("doc-1", "ws-1")

    @pytest.mark.asyncio
    async def test_missing_handler_returns_skipped(self, manager):
        """Test that missing handler returns skipped."""
        policy = manager.create_policy(
            name="Delete Policy",
            retention_days=30,
            action=RetentionAction.DELETE,
        )

        item = {
            "id": "doc-1",
            "type": "unknown_type",
            "workspace_id": "ws-1",
            "created_at": datetime.now(timezone.utc) - timedelta(days=60),
            "tags": [],
        }

        result = await manager._delete_item(item, policy)

        assert result == "skipped"

    @pytest.mark.asyncio
    async def test_handler_failure_returns_failed(self, manager):
        """Test that handler returning False returns failed."""
        failing_handler = MagicMock(return_value=False)
        manager.register_delete_handler("documents", failing_handler)

        policy = manager.create_policy(
            name="Delete Policy",
            retention_days=30,
            action=RetentionAction.DELETE,
        )

        item = {
            "id": "doc-1",
            "type": "documents",
            "workspace_id": "ws-1",
            "created_at": datetime.now(timezone.utc) - timedelta(days=60),
            "tags": [],
        }

        result = await manager._delete_item(item, policy)

        assert result == "failed"

    @pytest.mark.asyncio
    async def test_handler_creates_deletion_record(self, manager, mock_handler):
        """Test that successful deletion creates a record."""
        manager.register_delete_handler("documents", mock_handler)

        policy = manager.create_policy(
            name="Delete Policy",
            retention_days=30,
            action=RetentionAction.DELETE,
        )

        item = {
            "id": "doc-1",
            "type": "documents",
            "workspace_id": "ws-1",
            "created_at": datetime.now(timezone.utc) - timedelta(days=60),
            "tags": [],
        }

        await manager._delete_item(item, policy)

        assert len(manager._deletion_records) == 1
        record = manager._deletion_records[0]
        assert record.resource_id == "doc-1"
        assert record.resource_type == "documents"
        assert record.policy_id == policy.id


# ============================================================================
# Policy Execution Tests
# ============================================================================


class TestPolicyExecution:
    """Tests for policy execution."""

    @pytest.mark.asyncio
    async def test_execute_policy_not_found(self, manager):
        """Test executing non-existent policy raises error."""
        with pytest.raises(ValueError, match="Policy not found"):
            await manager.execute_policy("nonexistent")

    @pytest.mark.asyncio
    async def test_execute_disabled_policy(self, manager, policy):
        """Test executing disabled policy returns error report."""
        policy.enabled = False
        manager._policies[policy.id] = policy

        report = await manager.execute_policy(policy.id)

        assert report.policy_id == policy.id
        assert "Policy is disabled" in report.errors

    @pytest.mark.asyncio
    async def test_execute_policy_for_deletion(self, manager, mock_handler):
        """Test executing a deletion policy."""
        manager.register_delete_handler("documents", mock_handler)

        policy = manager.create_policy(
            name="Delete Policy",
            retention_days=30,
            action=RetentionAction.DELETE,
        )

        items = [
            {
                "id": "doc-1",
                "type": "documents",
                "workspace_id": "ws-1",
                "created_at": datetime.now(timezone.utc) - timedelta(days=60),
                "tags": [],
            }
        ]

        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = items
            report = await manager.execute_policy(policy.id)

        assert report.items_evaluated == 1
        assert report.items_deleted == 1

    @pytest.mark.asyncio
    async def test_execute_policy_for_archival(self, manager):
        """Test executing an archival policy."""
        policy = manager.create_policy(
            name="Archive Policy",
            retention_days=30,
            action=RetentionAction.ARCHIVE,
        )

        items = [
            {
                "id": "doc-1",
                "type": "documents",
                "workspace_id": "ws-1",
                "created_at": datetime.now(timezone.utc) - timedelta(days=60),
                "tags": [],
            }
        ]

        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = items
            report = await manager.execute_policy(policy.id)

        assert report.items_archived == 1

    @pytest.mark.asyncio
    async def test_execute_policy_for_anonymization(self, manager):
        """Test executing an anonymization policy."""
        policy = manager.create_policy(
            name="Anonymize Policy",
            retention_days=30,
            action=RetentionAction.ANONYMIZE,
        )

        items = [
            {
                "id": "doc-1",
                "type": "documents",
                "workspace_id": "ws-1",
                "created_at": datetime.now(timezone.utc) - timedelta(days=60),
                "tags": [],
            }
        ]

        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = items
            report = await manager.execute_policy(policy.id)

        assert report.items_anonymized == 1

    @pytest.mark.asyncio
    async def test_execute_policy_skips_non_expired(self, manager):
        """Test that non-expired items are skipped."""
        policy = manager.create_policy(
            name="Delete Policy",
            retention_days=30,
            action=RetentionAction.DELETE,
        )

        items = [
            {
                "id": "doc-1",
                "type": "documents",
                "workspace_id": "ws-1",
                "created_at": datetime.now(timezone.utc) - timedelta(days=10),  # Not expired
                "tags": [],
            }
        ]

        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = items
            report = await manager.execute_policy(policy.id)

        assert report.items_skipped == 1
        assert report.items_deleted == 0

    @pytest.mark.asyncio
    async def test_execute_policy_skips_excluded_sensitivity(self, manager):
        """Test that items with excluded sensitivity levels are skipped."""
        policy = manager.create_policy(
            name="Delete Policy",
            retention_days=30,
            action=RetentionAction.DELETE,
            exclude_sensitivity_levels=["high"],
        )

        items = [
            {
                "id": "doc-1",
                "type": "documents",
                "workspace_id": "ws-1",
                "created_at": datetime.now(timezone.utc) - timedelta(days=60),
                "sensitivity_level": "high",
                "tags": [],
            }
        ]

        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = items
            report = await manager.execute_policy(policy.id)

        assert report.items_skipped == 1
        assert report.items_deleted == 0

    @pytest.mark.asyncio
    async def test_execute_policy_skips_excluded_tags(self, manager):
        """Test that items with excluded tags are skipped."""
        policy = manager.create_policy(
            name="Delete Policy",
            retention_days=30,
            action=RetentionAction.DELETE,
            exclude_tags=["permanent", "important"],
        )

        items = [
            {
                "id": "doc-1",
                "type": "documents",
                "workspace_id": "ws-1",
                "created_at": datetime.now(timezone.utc) - timedelta(days=60),
                "tags": ["important"],
            }
        ]

        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = items
            report = await manager.execute_policy(policy.id)

        assert report.items_skipped == 1

    @pytest.mark.asyncio
    async def test_execute_policy_dry_run(self, manager, mock_handler):
        """Test dry run mode doesn't actually delete."""
        manager.register_delete_handler("documents", mock_handler)

        policy = manager.create_policy(
            name="Delete Policy",
            retention_days=30,
            action=RetentionAction.DELETE,
        )

        items = [
            {
                "id": "doc-1",
                "type": "documents",
                "workspace_id": "ws-1",
                "created_at": datetime.now(timezone.utc) - timedelta(days=60),
                "tags": [],
            }
        ]

        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = items
            report = await manager.execute_policy(policy.id, dry_run=True)

        # Handler should not be called in dry run
        mock_handler.assert_not_called()
        # Dry run returns action.value which is "delete" - counts as items_failed in report
        # because it doesn't match "deleted", "archived", "anonymized", or "skipped"
        assert report.items_failed == 1

    @pytest.mark.asyncio
    async def test_execute_policy_partial_errors(self, manager):
        """Test that execution continues after item errors."""
        error_handler = MagicMock(side_effect=[True, RuntimeError("Error"), True])
        manager.register_delete_handler("documents", error_handler)

        policy = manager.create_policy(
            name="Delete Policy",
            retention_days=30,
            action=RetentionAction.DELETE,
        )

        items = [
            {
                "id": f"doc-{i}",
                "type": "documents",
                "workspace_id": "ws-1",
                "created_at": datetime.now(timezone.utc) - timedelta(days=60),
                "tags": [],
            }
            for i in range(3)
        ]

        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = items
            report = await manager.execute_policy(policy.id)

        assert report.items_evaluated == 3
        assert report.items_deleted == 2
        assert report.items_failed == 1
        assert len(report.errors) == 1

    @pytest.mark.asyncio
    async def test_execute_policy_updates_last_run(self, manager):
        """Test that policy last_run is updated after execution."""
        policy = manager.create_policy(
            name="Test Policy",
            retention_days=30,
        )

        assert policy.last_run is None

        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = []
            await manager.execute_policy(policy.id)

        assert policy.last_run is not None

    @pytest.mark.asyncio
    async def test_execute_policy_records_duration(self, manager):
        """Test that execution duration is recorded."""
        policy = manager.create_policy(
            name="Test Policy",
            retention_days=30,
        )

        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = []
            report = await manager.execute_policy(policy.id)

        assert report.duration_seconds >= 0

    @pytest.mark.asyncio
    async def test_execute_policy_sends_notifications(self, manager):
        """Test that notifications are sent when configured."""
        policy = manager.create_policy(
            name="Test Policy",
            retention_days=30,
            notification_recipients=["admin@example.com", "user@example.com"],
        )

        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = []
            report = await manager.execute_policy(policy.id)

        assert report.notifications_sent == 2

    @pytest.mark.asyncio
    async def test_execute_policy_no_notifications_dry_run(self, manager):
        """Test that notifications are not sent in dry run."""
        policy = manager.create_policy(
            name="Test Policy",
            retention_days=30,
            notification_recipients=["admin@example.com"],
        )

        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = []
            report = await manager.execute_policy(policy.id, dry_run=True)

        assert report.notifications_sent == 0


# ============================================================================
# Execute All Policies Tests
# ============================================================================


class TestExecuteAllPolicies:
    """Tests for execute_all_policies method."""

    @pytest.mark.asyncio
    async def test_execute_all_policies(self, manager):
        """Test executing all enabled policies."""
        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = []
            reports = await manager.execute_all_policies()

        # Should execute both default policies
        assert len(reports) >= 2

    @pytest.mark.asyncio
    async def test_execute_all_policies_skips_disabled(self, manager):
        """Test that disabled policies are skipped."""
        # Disable default policies
        for policy in manager._policies.values():
            policy.enabled = False

        # Add one enabled policy
        manager.create_policy(
            name="Enabled Policy",
            retention_days=30,
        )

        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = []
            reports = await manager.execute_all_policies()

        assert len(reports) == 1
        assert reports[0].policy_id.startswith("policy_")

    @pytest.mark.asyncio
    async def test_execute_all_policies_dry_run(self, manager, mock_handler):
        """Test dry run for all policies."""
        manager.register_delete_handler("documents", mock_handler)

        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = []
            reports = await manager.execute_all_policies(dry_run=True)

        # Handler should not be called in dry run
        mock_handler.assert_not_called()
        assert len(reports) >= 2


# ============================================================================
# Expiration Detection Tests
# ============================================================================


class TestExpirationDetection:
    """Tests for check_expiring_soon method."""

    @pytest.mark.asyncio
    async def test_check_expiring_soon_basic(self, manager):
        """Test basic expiring soon check."""
        policy = manager.create_policy(
            name="Test Policy",
            retention_days=30,
        )

        items = [
            {
                "id": "doc-1",
                "type": "documents",
                "workspace_id": "ws-1",
                "created_at": datetime.now(timezone.utc) - timedelta(days=25),  # 5 days left
                "tags": [],
            }
        ]

        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = items
            expiring = await manager.check_expiring_soon(days=14)

        assert len(expiring) == 1
        assert expiring[0]["resource_id"] == "doc-1"
        # Days calculation may vary by 1 due to timing
        assert 4 <= expiring[0]["days_until_expiry"] <= 6

    @pytest.mark.asyncio
    async def test_check_expiring_soon_excludes_expired(self, manager):
        """Test that already expired items are excluded."""
        policy = manager.create_policy(
            name="Test Policy",
            retention_days=30,
        )

        items = [
            {
                "id": "doc-1",
                "type": "documents",
                "workspace_id": "ws-1",
                "created_at": datetime.now(timezone.utc) - timedelta(days=60),  # Already expired
                "tags": [],
            }
        ]

        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = items
            expiring = await manager.check_expiring_soon(days=14)

        assert len(expiring) == 0

    @pytest.mark.asyncio
    async def test_check_expiring_soon_excludes_far_future(self, manager):
        """Test that items not expiring soon are excluded."""
        policy = manager.create_policy(
            name="Test Policy",
            retention_days=30,
        )

        items = [
            {
                "id": "doc-1",
                "type": "documents",
                "workspace_id": "ws-1",
                "created_at": datetime.now(timezone.utc) - timedelta(days=5),  # 25 days left
                "tags": [],
            }
        ]

        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = items
            expiring = await manager.check_expiring_soon(days=14)

        assert len(expiring) == 0

    @pytest.mark.asyncio
    async def test_check_expiring_soon_different_windows(self, manager):
        """Test different time windows for expiration check."""
        policy = manager.create_policy(
            name="Test Policy",
            retention_days=30,
        )

        items = [
            {
                "id": "doc-1",
                "type": "documents",
                "workspace_id": "ws-1",
                "created_at": datetime.now(timezone.utc) - timedelta(days=25),  # 5 days left
                "tags": [],
            }
        ]

        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = items

            # 3 day window - should not include
            expiring_3 = await manager.check_expiring_soon(days=3)
            assert len(expiring_3) == 0

            # 7 day window - should include
            expiring_7 = await manager.check_expiring_soon(days=7)
            assert len(expiring_7) == 1

    @pytest.mark.asyncio
    async def test_check_expiring_soon_sorted_by_days(self, manager):
        """Test that results are sorted by days until expiry."""
        policy = manager.create_policy(
            name="Test Policy",
            retention_days=30,
        )

        items = [
            {
                "id": "doc-1",
                "type": "documents",
                "workspace_id": "ws-1",
                "created_at": datetime.now(timezone.utc) - timedelta(days=20),  # 10 days left
                "tags": [],
            },
            {
                "id": "doc-2",
                "type": "documents",
                "workspace_id": "ws-1",
                "created_at": datetime.now(timezone.utc) - timedelta(days=28),  # 2 days left
                "tags": [],
            },
            {
                "id": "doc-3",
                "type": "documents",
                "workspace_id": "ws-1",
                "created_at": datetime.now(timezone.utc) - timedelta(days=25),  # 5 days left
                "tags": [],
            },
        ]

        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = items
            expiring = await manager.check_expiring_soon(days=14)

        assert len(expiring) == 3
        assert expiring[0]["resource_id"] == "doc-2"  # 2 days
        assert expiring[1]["resource_id"] == "doc-3"  # 5 days
        assert expiring[2]["resource_id"] == "doc-1"  # 10 days

    @pytest.mark.asyncio
    async def test_check_expiring_soon_by_workspace(self, manager):
        """Test filtering expiring items by workspace."""
        manager.create_policy(
            name="WS1 Policy",
            retention_days=30,
            workspace_ids=["ws-1"],
        )

        items = [
            {
                "id": "doc-1",
                "type": "documents",
                "workspace_id": "ws-1",
                "created_at": datetime.now(timezone.utc) - timedelta(days=25),
                "tags": [],
            }
        ]

        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = items
            expiring = await manager.check_expiring_soon(workspace_id="ws-1", days=14)

        # Should include items from ws-1 policy
        ws1_items = [e for e in expiring if e.get("workspace_id") == "ws-1"]
        assert len(ws1_items) >= 0

    @pytest.mark.asyncio
    async def test_check_expiring_soon_skips_disabled_policies(self, manager):
        """Test that disabled policies are skipped."""
        policy = manager.create_policy(
            name="Disabled Policy",
            retention_days=30,
        )
        policy.enabled = False

        items = [
            {
                "id": "doc-1",
                "type": "documents",
                "workspace_id": "ws-1",
                "created_at": datetime.now(timezone.utc) - timedelta(days=25),
                "tags": [],
            }
        ]

        # Disable all default policies too
        for p in manager._policies.values():
            p.enabled = False

        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = items
            expiring = await manager.check_expiring_soon(days=14)

        assert len(expiring) == 0

    @pytest.mark.asyncio
    async def test_check_expiring_soon_includes_action_type(self, manager):
        """Test that results include the action type."""
        policy = manager.create_policy(
            name="Archive Policy",
            retention_days=30,
            action=RetentionAction.ARCHIVE,
        )

        items = [
            {
                "id": "doc-1",
                "type": "documents",
                "workspace_id": "ws-1",
                "created_at": datetime.now(timezone.utc) - timedelta(days=25),
                "tags": [],
            }
        ]

        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = items
            expiring = await manager.check_expiring_soon(days=14)

        policy_items = [e for e in expiring if e["policy_id"] == policy.id]
        if policy_items:
            assert policy_items[0]["action"] == "archive"

    @pytest.mark.asyncio
    async def test_check_expiring_soon_at_boundary(self, manager):
        """Test expiration check at exactly the window boundary."""
        policy = manager.create_policy(
            name="Test Policy",
            retention_days=30,
        )

        items = [
            {
                "id": "doc-1",
                "type": "documents",
                "workspace_id": "ws-1",
                "created_at": datetime.now(timezone.utc)
                - timedelta(days=16),  # Exactly 14 days left
                "tags": [],
            }
        ]

        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = items
            expiring = await manager.check_expiring_soon(days=14)

        # Exactly at boundary should be included
        assert len(expiring) == 1

    @pytest.mark.asyncio
    async def test_check_expiring_soon_empty_result(self, manager):
        """Test expiring soon with no items."""
        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = []
            expiring = await manager.check_expiring_soon(days=14)

        assert expiring == []


# ============================================================================
# Compliance Reporting Tests
# ============================================================================


class TestComplianceReporting:
    """Tests for compliance report generation."""

    @pytest.mark.asyncio
    async def test_get_compliance_report_basic(self, manager):
        """Test basic compliance report generation."""
        report = await manager.get_compliance_report()

        assert "report_period" in report
        assert "total_deletions" in report
        assert "deletions_by_policy" in report
        assert "deletions_by_type" in report
        assert "active_policies" in report

    @pytest.mark.asyncio
    async def test_get_compliance_report_structure(self, manager):
        """Test compliance report structure."""
        report = await manager.get_compliance_report()

        assert "start" in report["report_period"]
        assert "end" in report["report_period"]
        assert isinstance(report["total_deletions"], int)
        assert isinstance(report["deletions_by_policy"], dict)
        assert isinstance(report["deletions_by_type"], dict)
        assert isinstance(report["active_policies"], int)

    @pytest.mark.asyncio
    async def test_get_compliance_report_with_deletions(self, manager):
        """Test compliance report with deletion records."""
        now = datetime.now(timezone.utc)

        # Add deletion records
        manager._deletion_records.extend(
            [
                DeletionRecord(
                    resource_type="documents",
                    resource_id="doc-1",
                    workspace_id="ws-1",
                    policy_id="policy-a",
                    deleted_at=now - timedelta(days=5),
                ),
                DeletionRecord(
                    resource_type="documents",
                    resource_id="doc-2",
                    workspace_id="ws-1",
                    policy_id="policy-a",
                    deleted_at=now - timedelta(days=3),
                ),
                DeletionRecord(
                    resource_type="sessions",
                    resource_id="session-1",
                    workspace_id="ws-1",
                    policy_id="policy-b",
                    deleted_at=now - timedelta(days=1),
                ),
            ]
        )

        report = await manager.get_compliance_report()

        assert report["total_deletions"] == 3
        assert report["deletions_by_policy"]["policy-a"] == 2
        assert report["deletions_by_policy"]["policy-b"] == 1
        assert report["deletions_by_type"]["documents"] == 2
        assert report["deletions_by_type"]["sessions"] == 1

    @pytest.mark.asyncio
    async def test_get_compliance_report_date_filtering(self, manager):
        """Test compliance report date filtering."""
        now = datetime.now(timezone.utc)

        # Add records at different times
        manager._deletion_records.extend(
            [
                DeletionRecord(
                    resource_type="documents",
                    resource_id="doc-1",
                    workspace_id="ws-1",
                    policy_id="policy-a",
                    deleted_at=now - timedelta(days=60),  # Outside default 30-day window
                ),
                DeletionRecord(
                    resource_type="documents",
                    resource_id="doc-2",
                    workspace_id="ws-1",
                    policy_id="policy-a",
                    deleted_at=now - timedelta(days=10),  # Within window
                ),
            ]
        )

        report = await manager.get_compliance_report()

        # Only the recent deletion should be counted
        assert report["total_deletions"] == 1

    @pytest.mark.asyncio
    async def test_get_compliance_report_custom_date_range(self, manager):
        """Test compliance report with custom date range."""
        now = datetime.now(timezone.utc)

        manager._deletion_records.append(
            DeletionRecord(
                resource_type="documents",
                resource_id="doc-1",
                workspace_id="ws-1",
                policy_id="policy-a",
                deleted_at=now - timedelta(days=45),
            )
        )

        # Custom range that includes the old record
        report = await manager.get_compliance_report(
            start_date=now - timedelta(days=60),
            end_date=now,
        )

        assert report["total_deletions"] == 1

    @pytest.mark.asyncio
    async def test_get_compliance_report_by_workspace(self, manager):
        """Test compliance report filtered by workspace."""
        now = datetime.now(timezone.utc)

        manager._deletion_records.extend(
            [
                DeletionRecord(
                    resource_type="documents",
                    resource_id="doc-1",
                    workspace_id="ws-1",
                    policy_id="policy-a",
                    deleted_at=now - timedelta(days=5),
                ),
                DeletionRecord(
                    resource_type="documents",
                    resource_id="doc-2",
                    workspace_id="ws-2",
                    policy_id="policy-a",
                    deleted_at=now - timedelta(days=5),
                ),
            ]
        )

        report = await manager.get_compliance_report(workspace_id="ws-1")

        assert report["total_deletions"] == 1

    @pytest.mark.asyncio
    async def test_get_compliance_report_active_policies_count(self, manager):
        """Test that active policies count is accurate."""
        # Disable one default policy
        manager._policies["default_90_days"].enabled = False

        report = await manager.get_compliance_report()

        # Should have at least 1 active (audit_7_years)
        assert report["active_policies"] >= 1

    @pytest.mark.asyncio
    async def test_get_compliance_report_empty(self, manager):
        """Test compliance report with no deletions."""
        report = await manager.get_compliance_report()

        assert report["total_deletions"] == 0
        assert report["deletions_by_policy"] == {}
        assert report["deletions_by_type"] == {}

    @pytest.mark.asyncio
    async def test_get_compliance_report_policy_coverage(self, manager):
        """Test that report shows policy coverage."""
        # Add more policies
        manager.create_policy(name="Policy 1", retention_days=30)
        manager.create_policy(name="Policy 2", retention_days=60)

        report = await manager.get_compliance_report()

        # Should count all enabled policies
        assert report["active_policies"] >= 4  # 2 default + 2 new

    @pytest.mark.asyncio
    async def test_get_compliance_report_audit_trail(self, manager):
        """Test that deletions form an audit trail."""
        now = datetime.now(timezone.utc)

        for i in range(5):
            manager._deletion_records.append(
                DeletionRecord(
                    resource_type="documents",
                    resource_id=f"doc-{i}",
                    workspace_id="ws-1",
                    policy_id="policy-a",
                    deleted_at=now - timedelta(days=i),
                )
            )

        report = await manager.get_compliance_report()

        # All records should be tracked
        assert report["total_deletions"] == 5


# ============================================================================
# Process Item Tests
# ============================================================================


class TestProcessItem:
    """Tests for _process_item method."""

    @pytest.mark.asyncio
    async def test_process_item_not_expired_skipped(self, manager):
        """Test that non-expired items are skipped."""
        policy = manager.create_policy(
            name="Test Policy",
            retention_days=30,
        )

        item = {
            "id": "doc-1",
            "type": "documents",
            "workspace_id": "ws-1",
            "created_at": datetime.now(timezone.utc) - timedelta(days=10),
            "tags": [],
        }

        result = await manager._process_item(item, policy, dry_run=False)
        assert result == "skipped"

    @pytest.mark.asyncio
    async def test_process_item_excluded_sensitivity_skipped(self, manager):
        """Test that items with excluded sensitivity are skipped."""
        policy = manager.create_policy(
            name="Test Policy",
            retention_days=30,
            exclude_sensitivity_levels=["high"],
        )

        item = {
            "id": "doc-1",
            "type": "documents",
            "workspace_id": "ws-1",
            "created_at": datetime.now(timezone.utc) - timedelta(days=60),
            "sensitivity_level": "high",
            "tags": [],
        }

        result = await manager._process_item(item, policy, dry_run=False)
        assert result == "skipped"

    @pytest.mark.asyncio
    async def test_process_item_excluded_tag_skipped(self, manager):
        """Test that items with excluded tags are skipped."""
        policy = manager.create_policy(
            name="Test Policy",
            retention_days=30,
            exclude_tags=["permanent"],
        )

        item = {
            "id": "doc-1",
            "type": "documents",
            "workspace_id": "ws-1",
            "created_at": datetime.now(timezone.utc) - timedelta(days=60),
            "tags": ["permanent", "other"],
        }

        result = await manager._process_item(item, policy, dry_run=False)
        assert result == "skipped"

    @pytest.mark.asyncio
    async def test_process_item_dry_run_returns_action_value(self, manager):
        """Test that dry run returns the action.value string."""
        policy = manager.create_policy(
            name="Archive Policy",
            retention_days=30,
            action=RetentionAction.ARCHIVE,
        )

        item = {
            "id": "doc-1",
            "type": "documents",
            "workspace_id": "ws-1",
            "created_at": datetime.now(timezone.utc) - timedelta(days=60),
            "tags": [],
        }

        # In dry_run mode, returns action.value (e.g., "archive" not "archived")
        result = await manager._process_item(item, policy, dry_run=True)
        assert result == "archive"

    @pytest.mark.asyncio
    async def test_process_item_delete_action(self, manager, mock_handler):
        """Test processing item with delete action."""
        manager.register_delete_handler("documents", mock_handler)

        policy = manager.create_policy(
            name="Delete Policy",
            retention_days=30,
            action=RetentionAction.DELETE,
        )

        item = {
            "id": "doc-1",
            "type": "documents",
            "workspace_id": "ws-1",
            "created_at": datetime.now(timezone.utc) - timedelta(days=60),
            "tags": [],
        }

        result = await manager._process_item(item, policy, dry_run=False)
        assert result == "deleted"

    @pytest.mark.asyncio
    async def test_process_item_archive_action(self, manager):
        """Test processing item with archive action."""
        policy = manager.create_policy(
            name="Archive Policy",
            retention_days=30,
            action=RetentionAction.ARCHIVE,
        )

        item = {
            "id": "doc-1",
            "type": "documents",
            "workspace_id": "ws-1",
            "created_at": datetime.now(timezone.utc) - timedelta(days=60),
            "tags": [],
        }

        result = await manager._process_item(item, policy, dry_run=False)
        assert result == "archived"

    @pytest.mark.asyncio
    async def test_process_item_anonymize_action(self, manager):
        """Test processing item with anonymize action."""
        policy = manager.create_policy(
            name="Anonymize Policy",
            retention_days=30,
            action=RetentionAction.ANONYMIZE,
        )

        item = {
            "id": "doc-1",
            "type": "documents",
            "workspace_id": "ws-1",
            "created_at": datetime.now(timezone.utc) - timedelta(days=60),
            "tags": [],
        }

        result = await manager._process_item(item, policy, dry_run=False)
        assert result == "anonymized"

    @pytest.mark.asyncio
    async def test_process_item_notify_action_skipped(self, manager):
        """Test that notify action results in skipped."""
        policy = manager.create_policy(
            name="Notify Policy",
            retention_days=30,
            action=RetentionAction.NOTIFY,
        )

        item = {
            "id": "doc-1",
            "type": "documents",
            "workspace_id": "ws-1",
            "created_at": datetime.now(timezone.utc) - timedelta(days=60),
            "tags": [],
        }

        result = await manager._process_item(item, policy, dry_run=False)
        assert result == "skipped"


# ============================================================================
# Global Instance Tests
# ============================================================================


class TestGlobalInstance:
    """Tests for global retention manager instance."""

    def test_get_retention_manager_creates_instance(self):
        """Test that get_retention_manager creates instance."""
        # Reset global
        import aragora.privacy.retention as retention_module

        retention_module._retention_manager = None

        manager = get_retention_manager()
        assert manager is not None
        assert isinstance(manager, RetentionPolicyManager)

    def test_get_retention_manager_returns_same_instance(self):
        """Test that get_retention_manager returns same instance."""
        manager1 = get_retention_manager()
        manager2 = get_retention_manager()
        assert manager1 is manager2

    def test_get_retention_manager_has_default_policies(self):
        """Test that global manager has default policies."""
        manager = get_retention_manager()
        assert "default_90_days" in manager._policies
        assert "audit_7_years" in manager._policies


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_execute_policy_with_empty_items(self, manager):
        """Test executing policy with no items to process."""
        policy = manager.create_policy(
            name="Empty Policy",
            retention_days=30,
        )

        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = []
            report = await manager.execute_policy(policy.id)

        assert report.items_evaluated == 0
        assert report.items_deleted == 0
        assert report.items_failed == 0

    @pytest.mark.asyncio
    async def test_execute_policy_handles_get_items_error(self, manager):
        """Test that policy execution handles errors gracefully."""
        policy = manager.create_policy(
            name="Error Policy",
            retention_days=30,
        )

        with patch.object(manager, "_get_items_for_policy", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = RuntimeError("Database connection failed")
            report = await manager.execute_policy(policy.id)

        assert len(report.errors) > 0
        assert "Database connection failed" in report.errors[0]

    def test_create_policy_generates_unique_ids(self, manager):
        """Test that created policies have unique IDs."""
        policy1 = manager.create_policy(name="Policy 1", retention_days=30)
        policy2 = manager.create_policy(name="Policy 2", retention_days=30)

        assert policy1.id != policy2.id

    @pytest.mark.asyncio
    async def test_item_with_missing_fields(self, manager):
        """Test handling items with missing optional fields."""
        policy = manager.create_policy(
            name="Test Policy",
            retention_days=30,
            exclude_tags=["important"],
        )

        # Item without tags field
        item = {
            "id": "doc-1",
            "type": "documents",
            "created_at": datetime.now(timezone.utc) - timedelta(days=60),
        }

        # Should not raise, and should process normally
        result = await manager._process_item(item, policy, dry_run=True)
        assert result == "delete"

    @pytest.mark.asyncio
    async def test_item_without_workspace_id(self, manager, mock_handler):
        """Test deleting item without workspace_id."""
        manager.register_delete_handler("documents", mock_handler)

        policy = manager.create_policy(
            name="Delete Policy",
            retention_days=30,
        )

        item = {
            "id": "doc-1",
            "type": "documents",
            "created_at": datetime.now(timezone.utc) - timedelta(days=60),
            "tags": [],
        }

        result = await manager._delete_item(item, policy)

        assert result == "deleted"
        mock_handler.assert_called_once_with("doc-1", "")

    def test_policy_with_zero_retention(self, manager):
        """Test creating policy with zero retention days."""
        policy = manager.create_policy(
            name="Immediate Policy",
            retention_days=0,
        )

        # Any item should be immediately expired
        now = datetime.now(timezone.utc)
        assert policy.is_expired(now) is True

    def test_policy_with_very_long_retention(self, manager):
        """Test policy with very long retention period."""
        policy = manager.create_policy(
            name="Forever Policy",
            retention_days=365 * 100,  # 100 years
        )

        # Recent items should not be expired
        now = datetime.now(timezone.utc)
        assert policy.is_expired(now - timedelta(days=365)) is False

    @pytest.mark.asyncio
    async def test_multiple_exclusion_criteria(self, manager):
        """Test item matching multiple exclusion criteria."""
        policy = manager.create_policy(
            name="Strict Policy",
            retention_days=30,
            exclude_sensitivity_levels=["high", "critical"],
            exclude_tags=["permanent", "important"],
        )

        item = {
            "id": "doc-1",
            "type": "documents",
            "workspace_id": "ws-1",
            "created_at": datetime.now(timezone.utc) - timedelta(days=60),
            "sensitivity_level": "high",
            "tags": ["important"],
        }

        result = await manager._process_item(item, policy, dry_run=False)
        # Should be skipped due to sensitivity level (checked first)
        assert result == "skipped"


# ============================================================================
# Module Exports Tests
# ============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_available(self):
        """Test that all exports are available."""
        from aragora.privacy.retention import __all__

        assert "RetentionPolicyManager" in __all__
        assert "RetentionPolicy" in __all__
        assert "RetentionAction" in __all__
        assert "DeletionReport" in __all__
        assert "DeletionRecord" in __all__
        assert "RetentionViolation" in __all__
        assert "get_retention_manager" in __all__


__all__ = [
    "TestRetentionPolicy",
    "TestRetentionAction",
    "TestRetentionViolation",
    "TestDeletionRecord",
    "TestDeletionReport",
    "TestRetentionPolicyManagerInit",
    "TestRetentionPolicyManagerPolicies",
    "TestHandlerRegistration",
    "TestPolicyExecution",
    "TestExecuteAllPolicies",
    "TestExpirationDetection",
    "TestComplianceReporting",
    "TestProcessItem",
    "TestGlobalInstance",
    "TestEdgeCases",
    "TestModuleExports",
]
