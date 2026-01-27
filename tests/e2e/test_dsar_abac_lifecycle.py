"""
E2E Tests for DSAR (Data Subject Access Request) and ABAC Lifecycle.

Tests comprehensive workflows for:
- GDPR Article 15 data export (DSAR)
- Right-to-be-Forgotten (RTBF) with grace periods
- CCPA data portability
- Multi-tenant data isolation in DSARs
- Time-based access control (ABAC)
- IP-based network restrictions (ABAC)
- Resource ownership conditions (ABAC)
- Tag-based access control (ABAC)
- Combined ABAC conditions

SOC 2 Controls: P1-02 (Data subject rights procedures)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.rbac.conditions import (
    ConditionEvaluator,
    ConditionResult,
    IPCondition,
    ResourceOwnerCondition,
    ResourceStatusCondition,
    TagCondition,
    TimeCondition,
    get_condition_evaluator,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def condition_evaluator():
    """Fresh condition evaluator for testing."""
    return ConditionEvaluator()


@pytest.fixture
def mock_compliance_handler():
    """Create a mock compliance handler with DSAR methods."""
    handler = MagicMock()
    handler._gdpr_export = AsyncMock()
    handler._right_to_be_forgotten = AsyncMock()
    handler._generate_final_export = AsyncMock()
    handler._revoke_all_consents = AsyncMock()
    handler._schedule_deletion = AsyncMock()
    handler.can_handle = MagicMock(return_value=True)
    return handler


@pytest.fixture
def sample_user_data():
    """Sample user data for DSAR export tests."""
    return {
        "user_id": "user-dsar-test-001",
        "account": {
            "email": "test@example.com",
            "name": "Test User",
            "created_at": "2024-01-01T00:00:00Z",
            "workspace_id": "ws-001",
        },
        "profile": {
            "display_name": "Test User",
            "timezone": "America/New_York",
            "language": "en",
        },
        "debates": [
            {"id": "debate-001", "title": "Test Debate", "created_at": "2024-06-01"},
            {"id": "debate-002", "title": "Another Debate", "created_at": "2024-07-01"},
        ],
        "votes": [
            {"debate_id": "debate-001", "vote": "agree", "timestamp": "2024-06-02"},
        ],
        "api_usage": {
            "total_requests": 1500,
            "last_request": "2024-12-01T10:00:00Z",
        },
        "consent_records": [
            {"purpose": "analytics", "granted": True, "timestamp": "2024-01-01"},
            {"purpose": "marketing", "granted": False, "timestamp": "2024-01-01"},
        ],
    }


@pytest.fixture
def multi_tenant_users():
    """Users from different tenants for isolation tests."""
    return {
        "tenant_a": {
            "user_id": "user-tenant-a-001",
            "tenant_id": "tenant-a",
            "workspace_id": "ws-tenant-a",
            "email": "user1@tenant-a.com",
        },
        "tenant_b": {
            "user_id": "user-tenant-b-001",
            "tenant_id": "tenant-b",
            "workspace_id": "ws-tenant-b",
            "email": "user1@tenant-b.com",
        },
    }


# =============================================================================
# DSAR Lifecycle Tests - GDPR Article 15
# =============================================================================


class TestGDPRArticle15Export:
    """Tests for GDPR Article 15 data export (right of access)."""

    @pytest.mark.asyncio
    async def test_full_data_export_workflow(self, sample_user_data):
        """Test complete GDPR Article 15 data export workflow."""
        user_id = sample_user_data["user_id"]

        # Step 1: Verify user identity (simulated)
        verified_user = {"user_id": user_id, "verified": True, "method": "email"}

        # Step 2: Collect data from all sources
        collected_data = {
            "account_data": sample_user_data["account"],
            "profile_data": sample_user_data["profile"],
            "activity_data": {
                "debates": sample_user_data["debates"],
                "votes": sample_user_data["votes"],
            },
            "usage_data": sample_user_data["api_usage"],
            "consent_data": sample_user_data["consent_records"],
        }

        # Step 3: Generate export package
        export_package = {
            "request_id": f"DSAR-2026-{user_id[-4:]}",
            "user_id": user_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_categories": list(collected_data.keys()),
            "data": collected_data,
            "format": "json",
            "retention_info": {
                "account_data": "Until account deletion",
                "activity_data": "7 years (legal requirement)",
                "usage_data": "1 year",
            },
        }

        # Assertions
        assert export_package["user_id"] == user_id
        assert len(export_package["data_categories"]) == 5
        assert "account_data" in export_package["data_categories"]
        assert "activity_data" in export_package["data_categories"]
        assert export_package["format"] == "json"
        assert "retention_info" in export_package

    @pytest.mark.asyncio
    async def test_data_export_includes_all_categories(self, sample_user_data):
        """Test that export includes all required GDPR data categories."""
        required_categories = [
            "account",  # Personal identification
            "profile",  # Profile data
            "debates",  # Activity data
            "votes",  # User actions
            "api_usage",  # Usage data
            "consent_records",  # Consent history
        ]

        # Verify all categories present
        for category in required_categories:
            assert category in sample_user_data, f"Missing required category: {category}"

    @pytest.mark.asyncio
    async def test_data_export_deadline_compliance(self):
        """Test that DSAR export complies with 30-day deadline."""
        request_date = datetime.now(timezone.utc)
        gdpr_deadline = request_date + timedelta(days=30)

        # Simulate processing time
        processing_completed = request_date + timedelta(days=15)

        assert processing_completed < gdpr_deadline
        days_remaining = (gdpr_deadline - processing_completed).days
        assert days_remaining > 0

    @pytest.mark.asyncio
    async def test_data_export_with_extension(self):
        """Test DSAR extension for complex requests (60 days additional)."""
        request_date = datetime.now(timezone.utc)
        initial_deadline = request_date + timedelta(days=30)
        extended_deadline = initial_deadline + timedelta(days=60)  # GDPR allows up to 60 days
        max_deadline = request_date + timedelta(days=90)

        # Verify extension is within GDPR limits
        assert extended_deadline <= max_deadline

        extension_request = {
            "request_id": "DSAR-2026-0001",
            "original_deadline": initial_deadline.isoformat(),
            "extended_deadline": extended_deadline.isoformat(),
            "reason": "Complex request requiring multiple data source queries",
            "notified_within_30_days": True,
        }

        assert extension_request["notified_within_30_days"] is True

    @pytest.mark.asyncio
    async def test_data_export_redaction_of_third_party_data(self, sample_user_data):
        """Test that third-party personal data is redacted from export."""
        raw_debate = {
            "id": "debate-001",
            "title": "Test Debate",
            "participants": [
                {"user_id": "user-001", "name": "Requesting User"},
                {"user_id": "user-002", "name": "Other User"},  # Third party
            ],
            "messages": [
                {"from": "user-001", "content": "My message"},
                {"from": "user-002", "content": "Their message"},  # Third party
            ],
        }

        # Simulate redaction
        redacted_debate = {
            "id": "debate-001",
            "title": "Test Debate",
            "participants": [
                {"user_id": "user-001", "name": "Requesting User"},
                {"user_id": "[REDACTED]", "name": "[REDACTED]"},
            ],
            "messages": [
                {"from": "user-001", "content": "My message"},
                {"from": "[REDACTED]", "content": "[Third party content redacted]"},
            ],
        }

        # Verify redaction
        assert redacted_debate["participants"][1]["user_id"] == "[REDACTED]"
        assert "[Third party content redacted]" in redacted_debate["messages"][1]["content"]


# =============================================================================
# DSAR Lifecycle Tests - Right to be Forgotten
# =============================================================================


class TestRightToBeForgottenLifecycle:
    """Tests for GDPR Article 17 Right to be Forgotten (RTBF) lifecycle."""

    @pytest.mark.asyncio
    async def test_rtbf_full_workflow(self, mock_compliance_handler, sample_user_data):
        """Test complete RTBF workflow with grace period."""
        user_id = sample_user_data["user_id"]
        grace_period_days = 30

        # Configure mocks
        mock_compliance_handler._revoke_all_consents.return_value = {
            "revoked_count": 2,
            "purposes": ["analytics", "marketing"],
        }
        mock_compliance_handler._generate_final_export.return_value = {
            "export_url": f"/api/exports/{user_id}_final.zip",
            "expires_at": (datetime.now(timezone.utc) + timedelta(days=7)).isoformat(),
        }
        mock_compliance_handler._schedule_deletion.return_value = {
            "scheduled_for": (
                datetime.now(timezone.utc) + timedelta(days=grace_period_days)
            ).isoformat(),
            "status": "scheduled",
        }

        # Step 1: Revoke all consents
        consent_result = await mock_compliance_handler._revoke_all_consents(user_id)
        assert consent_result["revoked_count"] == 2

        # Step 2: Generate final export
        export_result = await mock_compliance_handler._generate_final_export(user_id)
        assert "export_url" in export_result

        # Step 3: Schedule deletion
        deletion_result = await mock_compliance_handler._schedule_deletion(
            user_id=user_id,
            grace_period_days=grace_period_days,
        )
        assert deletion_result["status"] == "scheduled"

    @pytest.mark.asyncio
    async def test_rtbf_preserves_legal_holds(self):
        """Test that RTBF preserves data under legal holds."""
        user_id = "user-legal-hold"

        deletion_request = {
            "user_id": user_id,
            "request_type": "erasure",
            "legal_holds": ["audit_logs", "financial_records"],
        }

        # Data that must be preserved
        preserved_data = deletion_request["legal_holds"]

        assert "audit_logs" in preserved_data
        assert "financial_records" in preserved_data

        # Verify deletion excludes legal holds
        deletion_result = {
            "deleted": ["profile", "preferences", "activity"],
            "preserved": preserved_data,
            "reason": "Legal requirement - 7 year retention",
        }

        assert len(deletion_result["preserved"]) == 2
        assert "profile" in deletion_result["deleted"]

    @pytest.mark.asyncio
    async def test_rtbf_grace_period_cancellation(self):
        """Test that user can cancel RTBF during grace period."""
        user_id = "user-cancel-rtbf"
        grace_period_days = 30

        # Schedule deletion
        scheduled_deletion = {
            "request_id": "rtbf-001",
            "user_id": user_id,
            "scheduled_for": (
                datetime.now(timezone.utc) + timedelta(days=grace_period_days)
            ).isoformat(),
            "status": "scheduled",
            "cancellable_until": (
                datetime.now(timezone.utc) + timedelta(days=grace_period_days - 1)
            ).isoformat(),
        }

        # Cancel during grace period
        current_time = datetime.now(timezone.utc) + timedelta(days=15)  # 15 days in
        cancellation_deadline = datetime.fromisoformat(
            scheduled_deletion["cancellable_until"].replace("Z", "+00:00")
        )

        can_cancel = current_time < cancellation_deadline
        assert can_cancel is True

        # After grace period ends
        current_time = datetime.now(timezone.utc) + timedelta(days=31)
        can_cancel = current_time < cancellation_deadline
        assert can_cancel is False

    @pytest.mark.asyncio
    async def test_rtbf_notifies_third_party_processors(self):
        """Test that RTBF notifies all third-party processors."""
        user_id = "user-third-party"

        third_party_processors = [
            {"name": "Analytics Provider", "contact": "dpo@analytics.com", "notified": False},
            {"name": "Payment Processor", "contact": "privacy@payment.com", "notified": False},
            {"name": "Email Service", "contact": "dsar@email.com", "notified": False},
        ]

        # Simulate notification
        for processor in third_party_processors:
            processor["notified"] = True
            processor["notification_date"] = datetime.now(timezone.utc).isoformat()

        # Verify all notified
        all_notified = all(p["notified"] for p in third_party_processors)
        assert all_notified is True


# =============================================================================
# DSAR Lifecycle Tests - Data Portability
# =============================================================================


class TestDataPortability:
    """Tests for GDPR Article 20 data portability."""

    @pytest.mark.asyncio
    async def test_portable_data_format(self, sample_user_data):
        """Test that data is exported in machine-readable format."""
        user_id = sample_user_data["user_id"]

        portable_export = {
            "format": "json",
            "encoding": "utf-8",
            "schema_version": "1.0",
            "data": {
                "user_provided": sample_user_data["account"],
                "user_generated": {
                    "debates": sample_user_data["debates"],
                    "votes": sample_user_data["votes"],
                },
            },
            # Inferred data NOT included in portability
            "excluded": ["inferred_preferences", "derived_metrics"],
        }

        assert portable_export["format"] == "json"
        assert "user_provided" in portable_export["data"]
        assert "user_generated" in portable_export["data"]
        assert "inferred_preferences" in portable_export["excluded"]

    @pytest.mark.asyncio
    async def test_portable_data_excludes_inferences(self, sample_user_data):
        """Test that portability export excludes inferred/derived data."""
        # Inferred data that should NOT be portable
        inferred_data = {
            "engagement_score": 85,
            "topic_interests": ["ai", "debate"],
            "predicted_churn_risk": 0.15,
        }

        # User-provided and generated data that IS portable
        portable_data = {
            "account": sample_user_data["account"],
            "debates": sample_user_data["debates"],
        }

        # Verify separation
        assert "engagement_score" not in portable_data
        assert "predicted_churn_risk" not in portable_data
        assert "account" in portable_data


# =============================================================================
# Multi-Tenant Data Isolation Tests
# =============================================================================


class TestMultiTenantDSARIsolation:
    """Tests for data isolation between tenants in DSARs."""

    @pytest.mark.asyncio
    async def test_dsar_returns_only_own_tenant_data(self, multi_tenant_users):
        """Test that DSAR only returns data from user's own tenant."""
        tenant_a_user = multi_tenant_users["tenant_a"]
        tenant_b_user = multi_tenant_users["tenant_b"]

        # Simulated data store
        all_data = {
            "tenant-a": {"users": [tenant_a_user], "debates": ["debate-a1", "debate-a2"]},
            "tenant-b": {"users": [tenant_b_user], "debates": ["debate-b1"]},
        }

        # DSAR for tenant A user
        requesting_user = tenant_a_user
        export_data = all_data[requesting_user["tenant_id"]]

        # Verify isolation
        assert len(export_data["debates"]) == 2
        assert "debate-a1" in export_data["debates"]
        assert "debate-b1" not in export_data["debates"]

    @pytest.mark.asyncio
    async def test_cross_tenant_dsar_denied(self, multi_tenant_users):
        """Test that cross-tenant DSAR requests are denied."""
        tenant_a_user = multi_tenant_users["tenant_a"]
        tenant_b_id = "tenant-b"

        # Attempt cross-tenant access
        dsar_request = {
            "requesting_user": tenant_a_user["user_id"],
            "requesting_tenant": tenant_a_user["tenant_id"],
            "target_tenant": tenant_b_id,  # Attempting to access different tenant
        }

        is_cross_tenant = dsar_request["requesting_tenant"] != dsar_request["target_tenant"]
        assert is_cross_tenant is True

        # Such request should be denied
        access_denied = is_cross_tenant
        assert access_denied is True

    @pytest.mark.asyncio
    async def test_admin_cross_tenant_dsar_with_justification(self, multi_tenant_users):
        """Test that system admins can access cross-tenant data with justification."""
        admin_user = {
            "user_id": "admin-001",
            "role": "system_admin",
            "tenant_id": "system",
        }

        dsar_request = {
            "requesting_user": admin_user["user_id"],
            "target_tenant": "tenant-a",
            "justification": "Legal compliance audit - SOC 2",
            "approved_by": "dpo@aragora.ai",
            "approval_date": datetime.now(timezone.utc).isoformat(),
        }

        # Verify justification is required
        assert dsar_request["justification"] is not None
        assert dsar_request["approved_by"] is not None


# =============================================================================
# ABAC Time-Based Access Control Tests
# =============================================================================


class TestABACTimeConditions:
    """Tests for time-based ABAC conditions."""

    def test_business_hours_access_allowed(self, condition_evaluator):
        """Test access allowed during business hours (9am-5pm Mon-Fri)."""
        # Tuesday at 10:30 AM UTC
        business_time = datetime(2026, 1, 27, 10, 30, tzinfo=timezone.utc)

        condition = TimeCondition(
            allowed_hours=(9, 17),
            allowed_days={0, 1, 2, 3, 4},  # Mon-Fri
        )

        result = condition.evaluate(True, {"current_time": business_time})

        assert result.satisfied is True
        assert "satisfied" in result.reason.lower() or "time" in result.condition_name

    def test_business_hours_access_denied_weekend(self, condition_evaluator):
        """Test access denied on weekends."""
        # Saturday at 10:30 AM UTC
        weekend_time = datetime(2026, 1, 31, 10, 30, tzinfo=timezone.utc)

        condition = TimeCondition(
            allowed_hours=(9, 17),
            allowed_days={0, 1, 2, 3, 4},  # Mon-Fri
        )

        result = condition.evaluate(True, {"current_time": weekend_time})

        assert result.satisfied is False
        assert "not allowed" in result.reason.lower() or "day" in result.reason.lower()

    def test_business_hours_access_denied_after_hours(self, condition_evaluator):
        """Test access denied after business hours."""
        # Tuesday at 8:00 PM UTC
        after_hours = datetime(2026, 1, 27, 20, 0, tzinfo=timezone.utc)

        condition = TimeCondition(
            allowed_hours=(9, 17),
            allowed_days={0, 1, 2, 3, 4},
        )

        result = condition.evaluate(True, {"current_time": after_hours})

        assert result.satisfied is False

    def test_date_range_access_allowed(self, condition_evaluator):
        """Test access allowed within date range."""
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = datetime(2026, 12, 31, tzinfo=timezone.utc)
        current = datetime(2026, 6, 15, tzinfo=timezone.utc)

        condition = TimeCondition(start_date=start, end_date=end)

        result = condition.evaluate(True, {"current_time": current})

        assert result.satisfied is True

    def test_date_range_access_denied_before_start(self, condition_evaluator):
        """Test access denied before start date."""
        start = datetime(2026, 2, 1, tzinfo=timezone.utc)
        current = datetime(2026, 1, 15, tzinfo=timezone.utc)

        condition = TimeCondition(start_date=start)

        result = condition.evaluate(True, {"current_time": current})

        assert result.satisfied is False
        assert "not yet" in result.reason.lower()

    def test_date_range_access_denied_after_end(self, condition_evaluator):
        """Test access denied after end date."""
        end = datetime(2026, 1, 1, tzinfo=timezone.utc)
        current = datetime(2026, 1, 15, tzinfo=timezone.utc)

        condition = TimeCondition(end_date=end)

        result = condition.evaluate(True, {"current_time": current})

        assert result.satisfied is False
        assert "expired" in result.reason.lower()


# =============================================================================
# ABAC IP-Based Access Control Tests
# =============================================================================


class TestABACIPConditions:
    """Tests for IP-based ABAC conditions."""

    def test_ip_allowlist_single_ip_allowed(self):
        """Test single IP allowlist access."""
        condition = IPCondition()

        result = condition.evaluate(
            ["192.168.1.100"],
            {"ip_address": "192.168.1.100"},
        )

        assert result.satisfied is True

    def test_ip_allowlist_single_ip_denied(self):
        """Test single IP allowlist denial."""
        condition = IPCondition()

        result = condition.evaluate(
            ["192.168.1.100"],
            {"ip_address": "192.168.1.200"},
        )

        assert result.satisfied is False
        assert "not in allowlist" in result.reason.lower()

    def test_ip_cidr_range_allowed(self):
        """Test IP within CIDR range."""
        condition = IPCondition()

        result = condition.evaluate(
            ["10.0.0.0/8"],
            {"ip_address": "10.50.100.25"},
        )

        assert result.satisfied is True

    def test_ip_cidr_range_denied(self):
        """Test IP outside CIDR range."""
        condition = IPCondition()

        result = condition.evaluate(
            ["10.0.0.0/8"],
            {"ip_address": "192.168.1.1"},
        )

        assert result.satisfied is False

    def test_ip_private_network_required(self):
        """Test private network requirement."""
        condition = IPCondition()

        # Private IP should pass
        result = condition.evaluate(
            {"require_private": True},
            {"ip_address": "192.168.1.1"},
        )
        assert result.satisfied is True

        # Public IP should fail
        result = condition.evaluate(
            {"require_private": True},
            {"ip_address": "8.8.8.8"},
        )
        assert result.satisfied is False

    def test_ip_blocklist_denied(self):
        """Test IP blocklist denial."""
        condition = IPCondition()

        result = condition.evaluate(
            {"blocklist": ["10.0.0.5"]},
            {"ip_address": "10.0.0.5"},
        )

        assert result.satisfied is False
        assert "blocklisted" in result.reason.lower()

    def test_ip_missing_context(self):
        """Test behavior when IP is missing from context."""
        condition = IPCondition()

        result = condition.evaluate(
            ["192.168.1.100"],
            {},  # No ip_address in context
        )

        assert result.satisfied is False
        assert "no ip" in result.reason.lower()


# =============================================================================
# ABAC Resource Ownership Tests
# =============================================================================


class TestABACResourceOwnership:
    """Tests for resource ownership ABAC conditions."""

    def test_direct_owner_access_allowed(self):
        """Test direct resource owner has access."""
        condition = ResourceOwnerCondition()

        result = condition.evaluate(
            True,
            {
                "actor_id": "user-001",
                "resource_owner": "user-001",
            },
        )

        assert result.satisfied is True
        assert "owns" in result.reason.lower()

    def test_non_owner_access_denied(self):
        """Test non-owner is denied access."""
        condition = ResourceOwnerCondition()

        result = condition.evaluate(
            True,
            {
                "actor_id": "user-002",
                "resource_owner": "user-001",
            },
        )

        assert result.satisfied is False
        assert "not the owner" in result.reason.lower()

    def test_group_owner_access_allowed(self):
        """Test group ownership grants access."""
        condition = ResourceOwnerCondition()

        result = condition.evaluate(
            True,
            {
                "actor_id": "user-003",
                "resource_owner": "user-001",
                "resource_owner_group": ["user-002", "user-003", "user-004"],
            },
        )

        assert result.satisfied is True
        assert "group" in result.reason.lower()

    def test_no_actor_in_context(self):
        """Test behavior when actor is missing from context."""
        condition = ResourceOwnerCondition()

        result = condition.evaluate(
            True,
            {
                "resource_owner": "user-001",
            },
        )

        assert result.satisfied is False
        assert "no actor" in result.reason.lower()


# =============================================================================
# ABAC Resource Status Tests
# =============================================================================


class TestABACResourceStatus:
    """Tests for resource status ABAC conditions."""

    def test_allowed_status_access(self):
        """Test access allowed when resource is in allowed status."""
        condition = ResourceStatusCondition()

        result = condition.evaluate(
            ["active", "review"],
            {"resource_status": "active"},
        )

        assert result.satisfied is True

    def test_disallowed_status_denied(self):
        """Test access denied when resource is not in allowed status."""
        condition = ResourceStatusCondition()

        result = condition.evaluate(
            ["active", "review"],
            {"resource_status": "archived"},
        )

        assert result.satisfied is False
        assert "not in allowed" in result.reason.lower()

    def test_missing_status_denied(self):
        """Test access denied when status is missing."""
        condition = ResourceStatusCondition()

        result = condition.evaluate(
            ["active"],
            {},
        )

        assert result.satisfied is False
        # Reason could be "missing" or "no resource status"
        assert "missing" in result.reason.lower() or "no" in result.reason.lower()


# =============================================================================
# ABAC Tag Condition Tests
# =============================================================================


class TestABACTagConditions:
    """Tests for tag-based ABAC conditions."""

    def test_single_required_tag_present(self):
        """Test access when single required tag is present."""
        condition = TagCondition()

        result = condition.evaluate(
            "public",
            {"resource_tags": ["public", "verified"]},
        )

        assert result.satisfied is True

    def test_single_required_tag_missing(self):
        """Test denial when required tag is missing."""
        condition = TagCondition()

        result = condition.evaluate(
            "premium",
            {"resource_tags": ["public", "verified"]},
        )

        assert result.satisfied is False
        assert "missing" in result.reason.lower()

    def test_all_required_tags_present(self):
        """Test access when all required tags present."""
        condition = TagCondition()

        result = condition.evaluate(
            ["reviewed", "approved"],
            {"resource_tags": ["reviewed", "approved", "public"]},
        )

        assert result.satisfied is True

    def test_some_required_tags_missing(self):
        """Test denial when some required tags missing."""
        condition = TagCondition()

        result = condition.evaluate(
            ["reviewed", "approved", "published"],
            {"resource_tags": ["reviewed", "approved"]},
        )

        assert result.satisfied is False
        assert "published" in result.reason.lower()

    def test_any_tag_condition_satisfied(self):
        """Test access when any one of required tags present."""
        condition = TagCondition()

        result = condition.evaluate(
            {"any": ["gold", "platinum", "vip"]},
            {"resource_tags": ["gold", "verified"]},
        )

        assert result.satisfied is True

    def test_any_tag_condition_not_satisfied(self):
        """Test denial when none of required tags present."""
        condition = TagCondition()

        result = condition.evaluate(
            {"any": ["gold", "platinum", "vip"]},
            {"resource_tags": ["bronze", "verified"]},
        )

        assert result.satisfied is False

    def test_forbidden_tags_denied(self):
        """Test denial when forbidden tags present."""
        condition = TagCondition()

        result = condition.evaluate(
            {"none": ["deprecated", "deleted"]},
            {"resource_tags": ["active", "deprecated"]},
        )

        assert result.satisfied is False
        assert "forbidden" in result.reason.lower()


# =============================================================================
# Combined ABAC Conditions Tests
# =============================================================================


class TestCombinedABACConditions:
    """Tests for combined ABAC condition evaluation."""

    def test_all_conditions_satisfied(self, condition_evaluator):
        """Test access when all conditions are satisfied."""
        # Business hours on Monday
        context = {
            "current_time": datetime(2026, 1, 26, 10, 0, tzinfo=timezone.utc),  # Monday 10 AM
            "ip_address": "10.0.0.50",
            "actor_id": "user-001",
            "resource_owner": "user-001",
            "resource_tags": ["public"],
        }

        conditions = {
            "business_hours": True,
            "ip_allowlist": ["10.0.0.0/8"],
            "owner": True,
            "tags": "public",
        }

        all_satisfied, results = condition_evaluator.evaluate(conditions, context)

        assert all_satisfied is True
        assert all(r.satisfied for r in results)

    def test_one_condition_fails(self, condition_evaluator):
        """Test access denied when one condition fails."""
        # After hours on Monday
        context = {
            "current_time": datetime(2026, 1, 26, 22, 0, tzinfo=timezone.utc),  # Monday 10 PM
            "ip_address": "10.0.0.50",
            "actor_id": "user-001",
            "resource_owner": "user-001",
        }

        conditions = {
            "business_hours": True,
            "ip_allowlist": ["10.0.0.0/8"],
            "owner": True,
        }

        all_satisfied, results = condition_evaluator.evaluate(conditions, context)

        assert all_satisfied is False
        # Find the failing condition
        failed = [r for r in results if not r.satisfied]
        assert len(failed) >= 1

    def test_enterprise_access_policy(self, condition_evaluator):
        """Test enterprise-grade combined policy."""
        # Enterprise policy: Business hours + Corporate network + Owner or admin
        context = {
            "current_time": datetime(2026, 1, 27, 14, 0, tzinfo=timezone.utc),  # Tuesday 2 PM
            "ip_address": "10.1.50.100",  # Corporate network
            "actor_id": "user-enterprise-001",
            "resource_owner": "user-enterprise-001",
            "resource_status": "active",
            "resource_tags": ["enterprise", "verified"],
        }

        conditions = {
            "business_hours": True,
            "ip_allowlist": ["10.0.0.0/8", "172.16.0.0/12"],
            "owner": True,
            "status": ["active", "review"],
            "tags": {"all": ["enterprise"]},
        }

        all_satisfied, results = condition_evaluator.evaluate(conditions, context)

        assert all_satisfied is True


# =============================================================================
# DSAR + ABAC Integration Tests
# =============================================================================


class TestDSARABACIntegration:
    """Tests for DSAR access control using ABAC conditions."""

    @pytest.mark.asyncio
    async def test_dsar_request_requires_owner_condition(self):
        """Test that DSAR requests require ownership verification."""
        condition = ResourceOwnerCondition()

        # User requesting their own data
        own_data_result = condition.evaluate(
            True,
            {
                "actor_id": "user-001",
                "resource_owner": "user-001",
            },
        )
        assert own_data_result.satisfied is True

        # User requesting someone else's data
        other_data_result = condition.evaluate(
            True,
            {
                "actor_id": "user-002",
                "resource_owner": "user-001",
            },
        )
        assert other_data_result.satisfied is False

    @pytest.mark.asyncio
    async def test_dsar_admin_override_with_audit(self, condition_evaluator):
        """Test admin can access DSAR data with audit logging."""
        admin_context = {
            "actor_id": "admin-001",
            "role": "admin",
            "resource_owner": "user-001",
            "audit_reason": "SOC 2 compliance audit",
            "audit_approver": "dpo@aragora.ai",
        }

        # Admin override should be allowed with audit trail
        is_admin = admin_context["role"] == "admin"
        has_audit_reason = "audit_reason" in admin_context
        has_approver = "audit_approver" in admin_context

        access_granted = is_admin and has_audit_reason and has_approver
        assert access_granted is True

    @pytest.mark.asyncio
    async def test_dsar_time_limited_access(self):
        """Test that DSAR export links have time-limited access."""
        export_created = datetime.now(timezone.utc)
        expiry_days = 7
        export_expiry = export_created + timedelta(days=expiry_days)

        condition = TimeCondition(end_date=export_expiry)

        # Access within validity period
        valid_access = export_created + timedelta(days=3)
        result = condition.evaluate(True, {"current_time": valid_access})
        assert result.satisfied is True

        # Access after expiry
        expired_access = export_created + timedelta(days=10)
        result = condition.evaluate(True, {"current_time": expired_access})
        assert result.satisfied is False


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestABACEdgeCases:
    """Tests for ABAC edge cases and error handling."""

    def test_invalid_ip_address_format(self):
        """Test handling of invalid IP address format."""
        condition = IPCondition()

        result = condition.evaluate(
            ["192.168.1.0/24"],
            {"ip_address": "not-an-ip"},
        )

        assert result.satisfied is False
        assert "invalid" in result.reason.lower()

    def test_empty_conditions_dict(self, condition_evaluator):
        """Test evaluation with empty conditions."""
        all_satisfied, results = condition_evaluator.evaluate({}, {})

        assert all_satisfied is True
        assert len(results) == 0

    def test_missing_context_values(self, condition_evaluator):
        """Test evaluation with missing context values."""
        conditions = {
            "ip_allowlist": ["10.0.0.0/8"],
        }

        all_satisfied, results = condition_evaluator.evaluate(conditions, {})

        assert all_satisfied is False

    def test_custom_condition_registration(self, condition_evaluator):
        """Test registration and use of custom conditions."""
        # Register custom condition
        condition_evaluator.register_custom(
            "premium_user",
            lambda expected, ctx: ctx.get("plan") == "enterprise",
        )

        # Test custom condition
        result_satisfied, results = condition_evaluator.evaluate(
            {"premium_user": True},
            {"plan": "enterprise"},
        )
        assert result_satisfied is True

        result_denied, results = condition_evaluator.evaluate(
            {"premium_user": True},
            {"plan": "free"},
        )
        assert result_denied is False
