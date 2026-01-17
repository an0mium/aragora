"""
Tests for the privacy module.

Tests data isolation, retention policies, sensitivity classification,
and audit logging functionality.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.privacy import (
    AccessDeniedException,
    AuditAction,
    AuditEntry,
    AuditOutcome,
    ClassificationConfig,
    ClassificationResult,
    DataIsolationManager,
    DeletionReport,
    IsolationConfig,
    PrivacyAuditLog,
    RetentionPolicy,
    RetentionPolicyManager,
    SensitivityClassifier,
    SensitivityLevel,
    Workspace,
    WorkspacePermission,
)
from aragora.privacy.audit_log import Actor, AuditLogConfig, Resource
from aragora.privacy.classifier import IndicatorMatch, SensitivityIndicator
from aragora.privacy.isolation import WorkspaceMember
from aragora.privacy.retention import DeletionRecord, RetentionAction


# =============================================================================
# DataIsolationManager Tests
# =============================================================================


class TestDataIsolationManager:
    """Tests for DataIsolationManager."""

    @pytest.fixture
    def isolation_manager(self, tmp_path):
        """Create isolation manager with temp directory."""
        config = IsolationConfig(
            workspace_data_root=str(tmp_path / "workspaces"),
            enable_encryption_at_rest=False,  # Disable for testing
        )
        return DataIsolationManager(config)

    @pytest.mark.asyncio
    async def test_create_workspace(self, isolation_manager):
        """Test workspace creation."""
        workspace = await isolation_manager.create_workspace(
            organization_id="org_123",
            name="Test Workspace",
            created_by="user_1",
        )

        assert workspace.id is not None
        assert workspace.organization_id == "org_123"
        assert workspace.name == "Test Workspace"
        assert workspace.created_by == "user_1"
        assert "user_1" in workspace.members

    @pytest.mark.asyncio
    async def test_creator_has_admin_permissions(self, isolation_manager):
        """Test that workspace creator gets admin permissions."""
        workspace = await isolation_manager.create_workspace(
            organization_id="org_123",
            name="Test Workspace",
            created_by="user_1",
        )

        member = workspace.members["user_1"]
        assert WorkspacePermission.ADMIN in member.permissions
        assert WorkspacePermission.READ in member.permissions
        assert WorkspacePermission.WRITE in member.permissions

    @pytest.mark.asyncio
    async def test_create_workspace_with_initial_members(self, isolation_manager):
        """Test workspace creation with initial members."""
        workspace = await isolation_manager.create_workspace(
            organization_id="org_123",
            name="Team Workspace",
            created_by="user_1",
            initial_members=["user_2", "user_3"],
        )

        assert "user_1" in workspace.members
        assert "user_2" in workspace.members
        assert "user_3" in workspace.members
        # Initial members get default permissions (READ only)
        assert WorkspacePermission.READ in workspace.members["user_2"].permissions

    @pytest.mark.asyncio
    async def test_get_workspace_success(self, isolation_manager):
        """Test getting a workspace with valid access."""
        workspace = await isolation_manager.create_workspace(
            organization_id="org_123",
            name="Test Workspace",
            created_by="user_1",
        )

        retrieved = await isolation_manager.get_workspace(workspace.id, "user_1")
        assert retrieved.id == workspace.id
        assert retrieved.name == "Test Workspace"

    @pytest.mark.asyncio
    async def test_get_workspace_access_denied(self, isolation_manager):
        """Test access denied for non-member."""
        workspace = await isolation_manager.create_workspace(
            organization_id="org_123",
            name="Private Workspace",
            created_by="user_1",
        )

        with pytest.raises(AccessDeniedException) as exc:
            await isolation_manager.get_workspace(workspace.id, "user_999")

        assert exc.value.workspace_id == workspace.id
        assert exc.value.actor == "user_999"

    @pytest.mark.asyncio
    async def test_check_access(self, isolation_manager):
        """Test permission checking."""
        workspace = await isolation_manager.create_workspace(
            organization_id="org_123",
            name="Test Workspace",
            created_by="user_1",
            initial_members=["user_2"],
        )

        # Admin has all permissions
        assert await isolation_manager.check_access(
            workspace.id, "user_1", WorkspacePermission.ADMIN
        )
        assert await isolation_manager.check_access(
            workspace.id, "user_1", WorkspacePermission.DELETE
        )

        # Regular member has only READ
        assert await isolation_manager.check_access(
            workspace.id, "user_2", WorkspacePermission.READ
        )
        assert not await isolation_manager.check_access(
            workspace.id, "user_2", WorkspacePermission.ADMIN
        )

    @pytest.mark.asyncio
    async def test_add_member(self, isolation_manager):
        """Test adding a member to workspace."""
        workspace = await isolation_manager.create_workspace(
            organization_id="org_123",
            name="Test Workspace",
            created_by="user_1",
        )

        await isolation_manager.add_member(
            workspace_id=workspace.id,
            user_id="user_2",
            permissions=[WorkspacePermission.READ, WorkspacePermission.WRITE],
            added_by="user_1",
        )

        workspace = await isolation_manager.get_workspace(workspace.id, "user_1")
        assert "user_2" in workspace.members
        assert WorkspacePermission.WRITE in workspace.members["user_2"].permissions

    @pytest.mark.asyncio
    async def test_add_member_requires_admin(self, isolation_manager):
        """Test that adding member requires admin permission."""
        workspace = await isolation_manager.create_workspace(
            organization_id="org_123",
            name="Test Workspace",
            created_by="user_1",
            initial_members=["user_2"],
        )

        with pytest.raises(AccessDeniedException):
            await isolation_manager.add_member(
                workspace_id=workspace.id,
                user_id="user_3",
                permissions=[WorkspacePermission.READ],
                added_by="user_2",  # Not admin
            )

    @pytest.mark.asyncio
    async def test_remove_member(self, isolation_manager):
        """Test removing a member from workspace."""
        workspace = await isolation_manager.create_workspace(
            organization_id="org_123",
            name="Test Workspace",
            created_by="user_1",
            initial_members=["user_2"],
        )

        await isolation_manager.remove_member(
            workspace_id=workspace.id,
            user_id="user_2",
            removed_by="user_1",
        )

        workspace = await isolation_manager.get_workspace(workspace.id, "user_1")
        assert "user_2" not in workspace.members

    @pytest.mark.asyncio
    async def test_delete_workspace(self, isolation_manager):
        """Test workspace deletion."""
        workspace = await isolation_manager.create_workspace(
            organization_id="org_123",
            name="Test Workspace",
            created_by="user_1",
        )
        workspace_id = workspace.id

        await isolation_manager.delete_workspace(workspace_id, "user_1")

        with pytest.raises(AccessDeniedException):
            await isolation_manager.get_workspace(workspace_id, "user_1")

    @pytest.mark.asyncio
    async def test_list_workspaces(self, isolation_manager):
        """Test listing accessible workspaces."""
        await isolation_manager.create_workspace(
            organization_id="org_1",
            name="Workspace 1",
            created_by="user_1",
        )
        await isolation_manager.create_workspace(
            organization_id="org_1",
            name="Workspace 2",
            created_by="user_1",
        )
        await isolation_manager.create_workspace(
            organization_id="org_2",
            name="Workspace 3",
            created_by="user_2",
        )

        # user_1 should see their workspaces
        workspaces = await isolation_manager.list_workspaces("user_1")
        assert len(workspaces) == 2

        # Filter by organization
        workspaces = await isolation_manager.list_workspaces(
            "user_1", organization_id="org_1"
        )
        assert len(workspaces) == 2

    @pytest.mark.asyncio
    async def test_workspace_to_dict(self, isolation_manager):
        """Test workspace serialization."""
        workspace = await isolation_manager.create_workspace(
            organization_id="org_123",
            name="Test Workspace",
            created_by="user_1",
        )

        data = workspace.to_dict()
        assert data["id"] == workspace.id
        assert data["organization_id"] == "org_123"
        assert data["name"] == "Test Workspace"
        assert data["member_count"] == 1


# =============================================================================
# RetentionPolicyManager Tests
# =============================================================================


class TestRetentionPolicyManager:
    """Tests for RetentionPolicyManager."""

    @pytest.fixture
    def retention_manager(self):
        """Create retention manager."""
        return RetentionPolicyManager()

    def test_default_policies(self, retention_manager):
        """Test that default policies are registered."""
        policies = retention_manager.list_policies()
        policy_ids = [p.id for p in policies]

        assert "default_90_days" in policy_ids
        assert "audit_7_years" in policy_ids

    def test_create_policy(self, retention_manager):
        """Test policy creation."""
        policy = retention_manager.create_policy(
            name="Short Retention",
            retention_days=30,
            action=RetentionAction.DELETE,
        )

        assert policy.name == "Short Retention"
        assert policy.retention_days == 30
        assert policy.action == RetentionAction.DELETE

    def test_get_policy(self, retention_manager):
        """Test getting a policy by ID."""
        policy = retention_manager.create_policy(
            name="Test Policy",
            retention_days=60,
        )

        retrieved = retention_manager.get_policy(policy.id)
        assert retrieved.name == "Test Policy"
        assert retrieved.retention_days == 60

    def test_update_policy(self, retention_manager):
        """Test updating a policy."""
        policy = retention_manager.create_policy(
            name="Original Name",
            retention_days=30,
        )

        updated = retention_manager.update_policy(
            policy.id,
            name="Updated Name",
            retention_days=60,
        )

        assert updated.name == "Updated Name"
        assert updated.retention_days == 60

    def test_delete_policy(self, retention_manager):
        """Test deleting a policy."""
        policy = retention_manager.create_policy(
            name="Temporary Policy",
            retention_days=7,
        )
        policy_id = policy.id

        retention_manager.delete_policy(policy_id)
        assert retention_manager.get_policy(policy_id) is None

    def test_policy_is_expired(self, retention_manager):
        """Test expiration checking."""
        policy = RetentionPolicy(
            id="test",
            name="Test",
            retention_days=30,
        )

        # Item created 31 days ago should be expired
        old_date = datetime.utcnow() - timedelta(days=31)
        assert policy.is_expired(old_date) is True

        # Item created 29 days ago should not be expired
        recent_date = datetime.utcnow() - timedelta(days=29)
        assert policy.is_expired(recent_date) is False

    def test_days_until_expiry(self, retention_manager):
        """Test days until expiry calculation."""
        policy = RetentionPolicy(
            id="test",
            name="Test",
            retention_days=30,
        )

        # Item created 20 days ago has ~10 days left
        created = datetime.utcnow() - timedelta(days=20)
        days_left = policy.days_until_expiry(created)
        assert 9 <= days_left <= 11  # Allow for timing variance

    def test_list_policies_by_workspace(self, retention_manager):
        """Test filtering policies by workspace."""
        retention_manager.create_policy(
            name="Global Policy",
            retention_days=90,
            workspace_ids=None,  # Applies to all
        )
        retention_manager.create_policy(
            name="Workspace Specific",
            retention_days=30,
            workspace_ids=["ws_123"],
        )

        # Get policies for ws_123
        policies = retention_manager.list_policies(workspace_id="ws_123")
        assert len(policies) >= 2  # Global + specific + defaults

        # Get policies for ws_456
        policies = retention_manager.list_policies(workspace_id="ws_456")
        # Should not include workspace-specific policy
        names = [p.name for p in policies]
        assert "Workspace Specific" not in names

    @pytest.mark.asyncio
    async def test_execute_policy_disabled(self, retention_manager):
        """Test that disabled policies don't execute."""
        policy = retention_manager.create_policy(
            name="Disabled Policy",
            retention_days=1,
            enabled=False,
        )

        report = await retention_manager.execute_policy(policy.id)
        assert "Policy is disabled" in report.errors

    @pytest.mark.asyncio
    async def test_execute_policy_dry_run(self, retention_manager):
        """Test dry run mode."""
        policy = retention_manager.create_policy(
            name="Test Policy",
            retention_days=30,
        )

        report = await retention_manager.execute_policy(policy.id, dry_run=True)
        assert report.items_deleted == 0  # No actual deletions

    @pytest.mark.asyncio
    async def test_check_expiring_soon(self, retention_manager):
        """Test checking for soon-to-expire items."""
        expiring = await retention_manager.check_expiring_soon(days=14)
        assert isinstance(expiring, list)

    @pytest.mark.asyncio
    async def test_compliance_report(self, retention_manager):
        """Test compliance report generation."""
        report = await retention_manager.get_compliance_report()

        assert "report_period" in report
        assert "total_deletions" in report
        assert "active_policies" in report

    def test_deletion_report_to_dict(self, retention_manager):
        """Test DeletionReport serialization."""
        report = DeletionReport(
            policy_id="test_policy",
            items_evaluated=100,
            items_deleted=10,
            items_skipped=90,
        )

        data = report.to_dict()
        assert data["policy_id"] == "test_policy"
        assert data["items_evaluated"] == 100
        assert data["items_deleted"] == 10


# =============================================================================
# SensitivityClassifier Tests
# =============================================================================


class TestSensitivityClassifier:
    """Tests for SensitivityClassifier."""

    @pytest.fixture
    def classifier(self):
        """Create sensitivity classifier."""
        return SensitivityClassifier()

    @pytest.mark.asyncio
    async def test_classify_public_content(self, classifier):
        """Test classification of public content."""
        result = await classifier.classify(
            "This is a public announcement about our new product launch."
        )

        # No sensitive indicators, should default to INTERNAL
        assert result.level == SensitivityLevel.INTERNAL
        assert result.pii_detected is False
        assert result.secrets_detected is False

    @pytest.mark.asyncio
    async def test_detect_ssn(self, classifier):
        """Test SSN detection."""
        result = await classifier.classify(
            "Customer SSN: 123-45-6789"
        )

        assert result.level == SensitivityLevel.CONFIDENTIAL
        assert result.pii_detected is True
        assert len(result.indicators_found) > 0

    @pytest.mark.asyncio
    async def test_detect_credit_card(self, classifier):
        """Test credit card detection."""
        result = await classifier.classify(
            "Payment card: 4111-1111-1111-1111"
        )

        assert result.level == SensitivityLevel.CONFIDENTIAL
        assert result.pii_detected is True

    @pytest.mark.asyncio
    async def test_detect_api_key(self, classifier):
        """Test API key detection."""
        result = await classifier.classify(
            'api_key = "sk_example_abcdefghijklmnopqrstuvwxyz"'
        )

        assert result.level == SensitivityLevel.RESTRICTED
        assert result.secrets_detected is True

    @pytest.mark.asyncio
    async def test_detect_private_key(self, classifier):
        """Test private key detection."""
        result = await classifier.classify(
            "-----BEGIN RSA PRIVATE KEY-----\nMIIE..."
        )

        assert result.level == SensitivityLevel.RESTRICTED
        assert result.secrets_detected is True

    @pytest.mark.asyncio
    async def test_detect_password(self, classifier):
        """Test password detection."""
        result = await classifier.classify(
            'password = "super_secret_password_123"'
        )

        assert result.level == SensitivityLevel.RESTRICTED
        assert result.secrets_detected is True

    @pytest.mark.asyncio
    async def test_detect_email(self, classifier):
        """Test email detection."""
        result = await classifier.classify(
            "Contact: john.doe@example.com"
        )

        assert result.level == SensitivityLevel.INTERNAL
        assert len(result.indicators_found) > 0

    @pytest.mark.asyncio
    async def test_detect_phone_number(self, classifier):
        """Test phone number detection."""
        result = await classifier.classify(
            "Call us at (555) 123-4567"
        )

        assert result.pii_detected is True

    @pytest.mark.asyncio
    async def test_detect_national_security(self, classifier):
        """Test national security marker detection."""
        result = await classifier.classify(
            "TOP SECRET // NOFORN"
        )

        assert result.level == SensitivityLevel.TOP_SECRET

    @pytest.mark.asyncio
    async def test_detect_medical_info(self, classifier):
        """Test medical information detection."""
        result = await classifier.classify(
            "Patient diagnosis: Type 2 Diabetes. HIPAA protected."
        )

        assert result.level == SensitivityLevel.CONFIDENTIAL

    @pytest.mark.asyncio
    async def test_detect_financial_data(self, classifier):
        """Test financial data detection."""
        result = await classifier.classify(
            "Bank account routing number: 123456789"
        )

        assert result.level == SensitivityLevel.CONFIDENTIAL

    @pytest.mark.asyncio
    async def test_multiple_indicators(self, classifier):
        """Test content with multiple indicators."""
        result = await classifier.classify(
            """
            Customer: John Doe
            SSN: 123-45-6789
            Credit Card: 4111-1111-1111-1111
            API Key: api_key="sk_example_abcdefghijklmnop123"
            """
        )

        # Should be highest level found
        assert result.level == SensitivityLevel.RESTRICTED
        assert result.pii_detected is True
        assert result.secrets_detected is True
        assert len(result.indicators_found) >= 3

    @pytest.mark.asyncio
    async def test_classify_document(self, classifier):
        """Test document classification interface."""
        document = {
            "id": "doc_123",
            "content": "Internal memo: Do not distribute",
            "metadata": {"author": "admin"},
        }

        result = await classifier.classify_document(document)
        assert result.document_id == "doc_123"
        assert result.level == SensitivityLevel.INTERNAL

    @pytest.mark.asyncio
    async def test_batch_classify(self, classifier):
        """Test batch classification."""
        documents = [
            {"id": "doc_1", "content": "Public info"},
            {"id": "doc_2", "content": "SSN: 123-45-6789"},
            {"id": "doc_3", "content": "api_key='secret12345678901234'"},
        ]

        results = await classifier.batch_classify(documents)
        assert len(results) == 3

    def test_add_custom_indicator(self, classifier):
        """Test adding custom indicators."""
        classifier.add_indicator(
            SensitivityIndicator(
                name="custom_pattern",
                pattern=r"CUSTOM-\d{6}",
                level=SensitivityLevel.CONFIDENTIAL,
                confidence=0.9,
            )
        )

        assert any(i.name == "custom_pattern" for i in classifier._indicators)

    def test_remove_indicator(self, classifier):
        """Test removing indicators."""
        initial_count = len(classifier._indicators)
        classifier.remove_indicator("email_addresses")

        assert len(classifier._indicators) == initial_count - 1
        assert not any(i.name == "email_addresses" for i in classifier._indicators)

    def test_get_level_policy(self, classifier):
        """Test getting policy recommendations for levels."""
        policy = classifier.get_level_policy(SensitivityLevel.PUBLIC)
        assert policy["encryption_required"] is False
        assert policy["sharing_allowed"] is True

        policy = classifier.get_level_policy(SensitivityLevel.TOP_SECRET)
        assert policy["encryption_required"] is True
        assert policy["mfa_required"] is True
        assert policy["sharing_allowed"] is False

    def test_classification_result_to_dict(self, classifier):
        """Test ClassificationResult serialization."""
        result = ClassificationResult(
            level=SensitivityLevel.CONFIDENTIAL,
            confidence=0.85,
            pii_detected=True,
        )

        data = result.to_dict()
        assert data["level"] == "confidential"
        assert data["confidence"] == 0.85
        assert data["pii_detected"] is True

    def test_custom_config(self):
        """Test classifier with custom config."""
        config = ClassificationConfig(
            default_level=SensitivityLevel.PUBLIC,
            min_confidence=0.8,
        )
        classifier = SensitivityClassifier(config)

        assert classifier.config.default_level == SensitivityLevel.PUBLIC
        assert classifier.config.min_confidence == 0.8


# =============================================================================
# PrivacyAuditLog Tests
# =============================================================================


class TestPrivacyAuditLog:
    """Tests for PrivacyAuditLog."""

    @pytest.fixture
    def audit_log(self, tmp_path):
        """Create audit log with temp directory."""
        config = AuditLogConfig(
            log_directory=str(tmp_path / "audit_logs"),
            enable_checksums=True,
            enable_chain_verification=True,
        )
        return PrivacyAuditLog(config)

    @pytest.mark.asyncio
    async def test_log_action(self, audit_log):
        """Test logging an action."""
        entry = await audit_log.log(
            action=AuditAction.READ,
            actor=Actor(id="user_1", type="user", name="John"),
            resource=Resource(id="doc_123", type="document", workspace_id="ws_1"),
            outcome=AuditOutcome.SUCCESS,
        )

        assert entry.id.startswith("audit_")
        assert entry.action == AuditAction.READ
        assert entry.outcome == AuditOutcome.SUCCESS
        assert entry.actor.id == "user_1"
        assert entry.resource.id == "doc_123"

    @pytest.mark.asyncio
    async def test_log_with_details(self, audit_log):
        """Test logging with additional details."""
        entry = await audit_log.log(
            action=AuditAction.QUERY,
            actor=Actor(id="user_1"),
            resource=Resource(id="knowledge_base", type="query"),
            outcome=AuditOutcome.SUCCESS,
            details={"query": "What are the key terms?", "results": 5},
            duration_ms=150,
        )

        assert entry.details["query"] == "What are the key terms?"
        assert entry.duration_ms == 150

    @pytest.mark.asyncio
    async def test_log_denied_access(self, audit_log):
        """Test logging denied access."""
        entry = await audit_log.log(
            action=AuditAction.WRITE,
            actor=Actor(id="user_unauthorized"),
            resource=Resource(id="doc_secret", type="document"),
            outcome=AuditOutcome.DENIED,
            error_message="Insufficient permissions",
        )

        assert entry.outcome == AuditOutcome.DENIED
        assert entry.error_message == "Insufficient permissions"

    @pytest.mark.asyncio
    async def test_checksum_generation(self, audit_log):
        """Test that checksums are generated."""
        entry = await audit_log.log(
            action=AuditAction.READ,
            actor=Actor(id="user_1"),
            resource=Resource(id="doc_1", type="document"),
            outcome=AuditOutcome.SUCCESS,
        )

        assert entry.checksum != ""
        assert len(entry.checksum) == 64  # SHA256 hex

    @pytest.mark.asyncio
    async def test_checksum_chain(self, audit_log):
        """Test that checksums form a chain."""
        entry1 = await audit_log.log(
            action=AuditAction.READ,
            actor=Actor(id="user_1"),
            resource=Resource(id="doc_1", type="document"),
            outcome=AuditOutcome.SUCCESS,
        )

        entry2 = await audit_log.log(
            action=AuditAction.WRITE,
            actor=Actor(id="user_1"),
            resource=Resource(id="doc_1", type="document"),
            outcome=AuditOutcome.SUCCESS,
        )

        # Second entry should reference first entry's checksum
        assert entry2.previous_checksum == entry1.checksum

    @pytest.mark.asyncio
    async def test_query_by_actor(self, audit_log):
        """Test querying by actor."""
        await audit_log.log(
            action=AuditAction.READ,
            actor=Actor(id="user_1"),
            resource=Resource(id="doc_1", type="document"),
            outcome=AuditOutcome.SUCCESS,
        )
        await audit_log.log(
            action=AuditAction.WRITE,
            actor=Actor(id="user_2"),
            resource=Resource(id="doc_2", type="document"),
            outcome=AuditOutcome.SUCCESS,
        )

        entries = await audit_log.query(actor_id="user_1")
        assert len(entries) == 1
        assert entries[0].actor.id == "user_1"

    @pytest.mark.asyncio
    async def test_query_by_action(self, audit_log):
        """Test querying by action type."""
        await audit_log.log(
            action=AuditAction.READ,
            actor=Actor(id="user_1"),
            resource=Resource(id="doc_1", type="document"),
            outcome=AuditOutcome.SUCCESS,
        )
        await audit_log.log(
            action=AuditAction.DELETE,
            actor=Actor(id="user_1"),
            resource=Resource(id="doc_1", type="document"),
            outcome=AuditOutcome.SUCCESS,
        )

        entries = await audit_log.query(action=AuditAction.DELETE)
        assert len(entries) == 1
        assert entries[0].action == AuditAction.DELETE

    @pytest.mark.asyncio
    async def test_query_by_date_range(self, audit_log):
        """Test querying by date range."""
        await audit_log.log(
            action=AuditAction.READ,
            actor=Actor(id="user_1"),
            resource=Resource(id="doc_1", type="document"),
            outcome=AuditOutcome.SUCCESS,
        )

        # Query with date range including now
        entries = await audit_log.query(
            start_date=datetime.utcnow() - timedelta(hours=1),
            end_date=datetime.utcnow() + timedelta(hours=1),
        )
        assert len(entries) >= 1

    @pytest.mark.asyncio
    async def test_verify_integrity_valid(self, audit_log):
        """Test integrity verification with valid chain."""
        for i in range(5):
            await audit_log.log(
                action=AuditAction.READ,
                actor=Actor(id=f"user_{i}"),
                resource=Resource(id=f"doc_{i}", type="document"),
                outcome=AuditOutcome.SUCCESS,
            )

        is_valid, errors = await audit_log.verify_integrity()
        assert is_valid is True
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_generate_compliance_report(self, audit_log):
        """Test compliance report generation."""
        # Log various actions
        await audit_log.log(
            action=AuditAction.READ,
            actor=Actor(id="user_1"),
            resource=Resource(id="doc_1", type="document"),
            outcome=AuditOutcome.SUCCESS,
        )
        await audit_log.log(
            action=AuditAction.WRITE,
            actor=Actor(id="user_1"),
            resource=Resource(id="doc_1", type="document"),
            outcome=AuditOutcome.DENIED,
        )

        report = await audit_log.generate_compliance_report()

        assert "report_id" in report
        assert report["summary"]["total_entries"] == 2
        assert report["summary"]["denied_count"] == 1
        assert report["integrity"]["verified"] is True

    @pytest.mark.asyncio
    async def test_get_denied_access_attempts(self, audit_log):
        """Test getting denied access attempts."""
        await audit_log.log(
            action=AuditAction.READ,
            actor=Actor(id="user_1"),
            resource=Resource(id="doc_1", type="document"),
            outcome=AuditOutcome.SUCCESS,
        )
        await audit_log.log(
            action=AuditAction.DELETE,
            actor=Actor(id="hacker"),
            resource=Resource(id="doc_secret", type="document"),
            outcome=AuditOutcome.DENIED,
        )

        denied = await audit_log.get_denied_access_attempts()
        assert len(denied) == 1
        assert denied[0].actor.id == "hacker"

    @pytest.mark.asyncio
    async def test_get_actor_history(self, audit_log):
        """Test getting actor's action history."""
        for i in range(3):
            await audit_log.log(
                action=AuditAction.READ,
                actor=Actor(id="active_user"),
                resource=Resource(id=f"doc_{i}", type="document"),
                outcome=AuditOutcome.SUCCESS,
            )

        history = await audit_log.get_actor_history("active_user")
        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_get_resource_history(self, audit_log):
        """Test getting resource's access history."""
        for action in [AuditAction.CREATE_WORKSPACE, AuditAction.READ, AuditAction.WRITE]:
            await audit_log.log(
                action=action,
                actor=Actor(id="user_1"),
                resource=Resource(id="important_doc", type="document"),
                outcome=AuditOutcome.SUCCESS,
            )

        history = await audit_log.get_resource_history("important_doc")
        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_export_logs(self, audit_log, tmp_path):
        """Test exporting audit logs."""
        for i in range(10):
            await audit_log.log(
                action=AuditAction.READ,
                actor=Actor(id=f"user_{i}"),
                resource=Resource(id=f"doc_{i}", type="document"),
                outcome=AuditOutcome.SUCCESS,
            )

        export_path = tmp_path / "export.jsonl"
        count = await audit_log.export_logs(
            start_date=datetime.utcnow() - timedelta(hours=1),
            end_date=datetime.utcnow() + timedelta(hours=1),
            output_path=export_path,
        )

        assert count == 10
        assert export_path.exists()

        # Verify export format
        with open(export_path) as f:
            lines = f.readlines()
        assert len(lines) == 10
        entry = json.loads(lines[0])
        assert "id" in entry
        assert "action" in entry

    @pytest.mark.asyncio
    async def test_exclude_actors(self, tmp_path):
        """Test actor exclusion from logging."""
        config = AuditLogConfig(
            log_directory=str(tmp_path / "audit_logs"),
            exclude_actors=["system_bot"],
        )
        audit_log = PrivacyAuditLog(config)

        result = await audit_log.log(
            action=AuditAction.READ,
            actor=Actor(id="system_bot"),
            resource=Resource(id="doc_1", type="document"),
            outcome=AuditOutcome.SUCCESS,
        )

        assert result is None
        entries = await audit_log.query()
        assert len(entries) == 0

    @pytest.mark.asyncio
    async def test_skip_read_operations(self, tmp_path):
        """Test skipping read operations when configured."""
        config = AuditLogConfig(
            log_directory=str(tmp_path / "audit_logs"),
            log_read_operations=False,
        )
        audit_log = PrivacyAuditLog(config)

        await audit_log.log(
            action=AuditAction.READ,
            actor=Actor(id="user_1"),
            resource=Resource(id="doc_1", type="document"),
            outcome=AuditOutcome.SUCCESS,
        )
        await audit_log.log(
            action=AuditAction.WRITE,
            actor=Actor(id="user_1"),
            resource=Resource(id="doc_1", type="document"),
            outcome=AuditOutcome.SUCCESS,
        )

        entries = await audit_log.query()
        assert len(entries) == 1
        assert entries[0].action == AuditAction.WRITE

    def test_audit_entry_serialization(self):
        """Test AuditEntry serialization round-trip."""
        entry = AuditEntry(
            id="audit_123",
            timestamp=datetime.utcnow(),
            action=AuditAction.DELETE,
            outcome=AuditOutcome.SUCCESS,
            actor=Actor(id="user_1", type="user", name="John"),
            resource=Resource(id="doc_1", type="document", workspace_id="ws_1"),
            details={"reason": "expired"},
            checksum="abc123",
        )

        data = entry.to_dict()
        restored = AuditEntry.from_dict(data)

        assert restored.id == entry.id
        assert restored.action == entry.action
        assert restored.actor.id == entry.actor.id
        assert restored.resource.id == entry.resource.id

    def test_all_audit_actions_defined(self):
        """Test that all audit action types are available."""
        actions = list(AuditAction)
        assert len(actions) >= 15

        # Check key actions exist
        assert AuditAction.READ in actions
        assert AuditAction.WRITE in actions
        assert AuditAction.DELETE in actions
        assert AuditAction.CREATE_WORKSPACE in actions
        assert AuditAction.LOGIN in actions


# =============================================================================
# Integration Tests
# =============================================================================


class TestPrivacyIntegration:
    """Integration tests for privacy module components."""

    @pytest.fixture
    def privacy_stack(self, tmp_path):
        """Create full privacy stack."""
        isolation_config = IsolationConfig(
            workspace_data_root=str(tmp_path / "workspaces"),
            enable_encryption_at_rest=False,
        )
        audit_config = AuditLogConfig(
            log_directory=str(tmp_path / "audit_logs"),
        )

        return {
            "isolation": DataIsolationManager(isolation_config),
            "retention": RetentionPolicyManager(),
            "classifier": SensitivityClassifier(),
            "audit": PrivacyAuditLog(audit_config),
        }

    @pytest.mark.asyncio
    async def test_workspace_with_audit_logging(self, privacy_stack):
        """Test workspace operations are audit logged."""
        isolation = privacy_stack["isolation"]
        audit = privacy_stack["audit"]

        # Create workspace
        workspace = await isolation.create_workspace(
            organization_id="org_1",
            name="Audited Workspace",
            created_by="admin",
        )

        # Log the creation
        await audit.log(
            action=AuditAction.CREATE_WORKSPACE,
            actor=Actor(id="admin", type="user"),
            resource=Resource(
                id=workspace.id,
                type="workspace",
                workspace_id=workspace.id,
            ),
            outcome=AuditOutcome.SUCCESS,
        )

        # Verify audit trail
        entries = await audit.query(action=AuditAction.CREATE_WORKSPACE)
        assert len(entries) == 1
        assert entries[0].resource.id == workspace.id

    @pytest.mark.asyncio
    async def test_document_classification_with_policy(self, privacy_stack):
        """Test document classification informs retention policy."""
        classifier = privacy_stack["classifier"]
        retention = privacy_stack["retention"]

        # Classify a document
        result = await classifier.classify(
            "Patient diagnosis: Diabetes Type 2. HIPAA protected information."
        )

        assert result.level == SensitivityLevel.CONFIDENTIAL

        # Get policy for this level
        policy_settings = classifier.get_level_policy(result.level)

        # Create retention policy based on classification
        if result.level in (SensitivityLevel.CONFIDENTIAL, SensitivityLevel.RESTRICTED):
            policy = retention.create_policy(
                name="Sensitive Document Retention",
                retention_days=policy_settings.get("retention_days", 90),
                action=RetentionAction.ARCHIVE,
            )

            assert policy.retention_days == 90
            assert policy.action == RetentionAction.ARCHIVE

    @pytest.mark.asyncio
    async def test_access_denied_audit_trail(self, privacy_stack):
        """Test that access denials are properly logged."""
        isolation = privacy_stack["isolation"]
        audit = privacy_stack["audit"]

        # Create private workspace
        workspace = await isolation.create_workspace(
            organization_id="org_1",
            name="Private Workspace",
            created_by="owner",
        )

        # Attempt unauthorized access
        try:
            await isolation.get_workspace(workspace.id, "unauthorized_user")
        except AccessDeniedException as e:
            # Log the denied access
            await audit.log(
                action=AuditAction.READ,
                actor=Actor(id=e.actor, type="user"),
                resource=Resource(
                    id=e.workspace_id,
                    type="workspace",
                ),
                outcome=AuditOutcome.DENIED,
                error_message=str(e),
            )

        # Verify denial was logged
        denied = await audit.get_denied_access_attempts()
        assert len(denied) == 1
        assert denied[0].actor.id == "unauthorized_user"

    @pytest.mark.asyncio
    async def test_full_compliance_workflow(self, privacy_stack):
        """Test complete compliance workflow."""
        isolation = privacy_stack["isolation"]
        retention = privacy_stack["retention"]
        classifier = privacy_stack["classifier"]
        audit = privacy_stack["audit"]

        # 1. Create workspace
        workspace = await isolation.create_workspace(
            organization_id="acme_corp",
            name="Legal Documents",
            created_by="legal_admin",
        )

        await audit.log(
            action=AuditAction.CREATE_WORKSPACE,
            actor=Actor(id="legal_admin"),
            resource=Resource(
                id=workspace.id,
                type="workspace",
                workspace_id=workspace.id,  # Include workspace_id for filtering
            ),
            outcome=AuditOutcome.SUCCESS,
        )

        # 2. Classify document
        doc_content = "Contract with SSN: 123-45-6789"
        classification = await classifier.classify(doc_content)

        await audit.log(
            action=AuditAction.CLASSIFY_DOCUMENT,
            actor=Actor(id="system", type="system"),
            resource=Resource(
                id="doc_contract",
                type="document",
                workspace_id=workspace.id,
                sensitivity_level=classification.level.value,
            ),
            outcome=AuditOutcome.SUCCESS,
            details={"level": classification.level.value, "confidence": classification.confidence},
        )

        # 3. Set retention policy
        retention.create_policy(
            name="Legal Document Retention",
            retention_days=365 * 7,  # 7 years
            workspace_ids=[workspace.id],
        )

        # 4. Generate compliance report
        report = await audit.generate_compliance_report(workspace_id=workspace.id)

        assert report["summary"]["total_entries"] >= 2
        assert report["integrity"]["verified"] is True
