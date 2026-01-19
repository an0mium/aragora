"""
Tests for aragora.billing.usage_sync module.

Tests cover:
- UsageSyncRecord dataclass
- OrgBillingConfig dataclass
- UsageSyncService operations
- Token and debate usage syncing
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, AsyncMock

from aragora.billing.models import SubscriptionTier
from aragora.billing.usage_sync import (
    UsageSyncRecord,
    OrgBillingConfig,
    UsageSyncService,
)


# ============================================================================
# UsageSyncRecord Tests
# ============================================================================


class TestUsageSyncRecord:
    """Tests for UsageSyncRecord dataclass."""

    def test_create_default(self):
        """Test creating a sync record with defaults."""
        record = UsageSyncRecord()

        assert record.id != ""  # Auto-generated UUID
        assert record.org_id == ""
        assert record.subscription_id == ""
        assert record.sync_type == ""
        assert record.quantity == 0
        assert record.synced_at is not None
        assert record.stripe_record_id == ""
        assert record.success is True
        assert record.error == ""

    def test_create_with_values(self):
        """Test creating a sync record with custom values."""
        record = UsageSyncRecord(
            id="sync-123",
            org_id="org-456",
            subscription_id="sub_abc",
            sync_type="tokens_input",
            quantity=10000,
            stripe_record_id="usage_xyz",
            success=True,
        )

        assert record.id == "sync-123"
        assert record.org_id == "org-456"
        assert record.subscription_id == "sub_abc"
        assert record.sync_type == "tokens_input"
        assert record.quantity == 10000
        assert record.stripe_record_id == "usage_xyz"
        assert record.success is True

    def test_create_with_error(self):
        """Test creating a failed sync record."""
        record = UsageSyncRecord(
            org_id="org-456",
            sync_type="debates",
            quantity=5,
            success=False,
            error="Stripe API error: rate limited",
        )

        assert record.success is False
        assert record.error == "Stripe API error: rate limited"

    def test_unique_ids(self):
        """Test that default IDs are unique."""
        record1 = UsageSyncRecord()
        record2 = UsageSyncRecord()

        assert record1.id != record2.id


# ============================================================================
# OrgBillingConfig Tests
# ============================================================================


class TestOrgBillingConfig:
    """Tests for OrgBillingConfig dataclass."""

    def test_create_basic(self):
        """Test creating a basic billing config."""
        config = OrgBillingConfig(
            org_id="org-123",
            stripe_customer_id="cus_abc",
            stripe_subscription_id="sub_xyz",
            tier=SubscriptionTier.PROFESSIONAL,
        )

        assert config.org_id == "org-123"
        assert config.stripe_customer_id == "cus_abc"
        assert config.stripe_subscription_id == "sub_xyz"
        assert config.tier == SubscriptionTier.PROFESSIONAL
        assert config.metered_enabled is False

    def test_create_with_metered(self):
        """Test creating config with metered billing enabled."""
        config = OrgBillingConfig(
            org_id="org-123",
            stripe_customer_id="cus_abc",
            stripe_subscription_id="sub_xyz",
            tier=SubscriptionTier.ENTERPRISE,
            metered_enabled=True,
            tokens_input_item_id="si_input",
            tokens_output_item_id="si_output",
            debates_item_id="si_debates",
        )

        assert config.metered_enabled is True
        assert config.tokens_input_item_id == "si_input"
        assert config.tokens_output_item_id == "si_output"
        assert config.debates_item_id == "si_debates"

    def test_tier_types(self):
        """Test different subscription tier types."""
        for tier in SubscriptionTier:
            config = OrgBillingConfig(
                org_id="org-test",
                stripe_customer_id="cus_test",
                stripe_subscription_id="sub_test",
                tier=tier,
            )
            assert config.tier == tier


# ============================================================================
# UsageSyncService Tests
# ============================================================================


class TestUsageSyncService:
    """Tests for UsageSyncService."""

    @pytest.fixture
    def mock_usage_tracker(self):
        """Create a mock UsageTracker."""
        tracker = MagicMock()
        tracker.get_usage = MagicMock(return_value={
            "tokens_in": 5000,
            "tokens_out": 2000,
            "debates": 10,
        })
        return tracker

    @pytest.fixture
    def mock_stripe_client(self):
        """Create a mock StripeClient."""
        client = MagicMock()
        client.report_usage = MagicMock(return_value={"id": "usage_123"})
        return client

    @pytest.fixture
    def service(self, mock_usage_tracker, mock_stripe_client):
        """Create UsageSyncService with mocks."""
        return UsageSyncService(
            usage_tracker=mock_usage_tracker,
            stripe_client=mock_stripe_client,
            sync_interval=60,  # 1 minute for testing
        )

    def test_service_creation_defaults(self):
        """Test service creation with default values."""
        with patch("aragora.billing.usage_sync.UsageTracker"), \
             patch("aragora.billing.usage_sync.get_stripe_client"):
            service = UsageSyncService()
            assert service.sync_interval == UsageSyncService.DEFAULT_SYNC_INTERVAL

    def test_service_creation_custom(self, mock_usage_tracker, mock_stripe_client):
        """Test service creation with custom values."""
        service = UsageSyncService(
            usage_tracker=mock_usage_tracker,
            stripe_client=mock_stripe_client,
            sync_interval=120,
        )

        assert service.usage_tracker == mock_usage_tracker
        assert service.stripe_client == mock_stripe_client
        assert service.sync_interval == 120

    def test_min_tokens_threshold(self, service):
        """Test MIN_TOKENS_THRESHOLD constant."""
        assert UsageSyncService.MIN_TOKENS_THRESHOLD == 1000

    def test_default_sync_interval(self, service):
        """Test DEFAULT_SYNC_INTERVAL constant."""
        assert UsageSyncService.DEFAULT_SYNC_INTERVAL == 300  # 5 minutes

    def test_tracking_initialization(self, service):
        """Test that tracking dictionaries are initialized."""
        assert service._last_sync == {}
        assert service._synced_tokens_in == {}
        assert service._synced_tokens_out == {}
        assert service._synced_debates == {}


# ============================================================================
# Usage Tracking Tests
# ============================================================================


class TestUsageTracking:
    """Tests for usage tracking state management."""

    @pytest.fixture
    def service(self):
        """Create service with mocks."""
        with patch("aragora.billing.usage_sync.UsageTracker"), \
             patch("aragora.billing.usage_sync.get_stripe_client"):
            return UsageSyncService(sync_interval=60)

    def test_track_synced_tokens_input(self, service):
        """Test tracking synced input tokens."""
        service._synced_tokens_in["org-123"] = 5000

        assert service._synced_tokens_in["org-123"] == 5000

    def test_track_synced_tokens_output(self, service):
        """Test tracking synced output tokens."""
        service._synced_tokens_out["org-123"] = 2000

        assert service._synced_tokens_out["org-123"] == 2000

    def test_track_synced_debates(self, service):
        """Test tracking synced debate counts."""
        service._synced_debates["org-123"] = 10

        assert service._synced_debates["org-123"] == 10

    def test_track_last_sync_time(self, service):
        """Test tracking last sync time per org."""
        now = datetime.utcnow()
        service._last_sync["org-123"] = now

        assert service._last_sync["org-123"] == now

    def test_multiple_orgs_isolated(self, service):
        """Test that tracking is isolated per org."""
        service._synced_tokens_in["org-1"] = 1000
        service._synced_tokens_in["org-2"] = 2000

        assert service._synced_tokens_in["org-1"] == 1000
        assert service._synced_tokens_in["org-2"] == 2000
        assert service._synced_tokens_in.get("org-3") is None


# ============================================================================
# Sync Record State Tests
# ============================================================================


class TestSyncRecordState:
    """Tests for sync record state tracking."""

    def test_success_record(self):
        """Test creating a successful sync record."""
        record = UsageSyncRecord(
            org_id="org-123",
            subscription_id="sub_abc",
            sync_type="tokens_input",
            quantity=10000,
            stripe_record_id="usage_xyz",
            success=True,
        )

        assert record.success is True
        assert record.error == ""
        assert record.stripe_record_id == "usage_xyz"

    def test_failure_record(self):
        """Test creating a failed sync record."""
        record = UsageSyncRecord(
            org_id="org-123",
            subscription_id="sub_abc",
            sync_type="tokens_input",
            quantity=10000,
            success=False,
            error="API rate limit exceeded",
        )

        assert record.success is False
        assert record.error == "API rate limit exceeded"
        assert record.stripe_record_id == ""

    def test_sync_types(self):
        """Test different sync type values."""
        for sync_type in ["tokens_input", "tokens_output", "debates"]:
            record = UsageSyncRecord(
                org_id="org-test",
                sync_type=sync_type,
            )
            assert record.sync_type == sync_type


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_quantity_sync(self):
        """Test sync record with zero quantity."""
        record = UsageSyncRecord(
            org_id="org-123",
            sync_type="debates",
            quantity=0,
        )

        assert record.quantity == 0

    def test_large_quantity_sync(self):
        """Test sync record with large quantity."""
        record = UsageSyncRecord(
            org_id="org-123",
            sync_type="tokens_input",
            quantity=1_000_000_000,  # 1 billion tokens
        )

        assert record.quantity == 1_000_000_000

    def test_empty_org_id(self):
        """Test with empty org_id."""
        config = OrgBillingConfig(
            org_id="",
            stripe_customer_id="cus_abc",
            stripe_subscription_id="sub_xyz",
            tier=SubscriptionTier.FREE,
        )

        assert config.org_id == ""

    def test_none_subscription_item_ids(self):
        """Test with None subscription item IDs (non-metered)."""
        config = OrgBillingConfig(
            org_id="org-123",
            stripe_customer_id="cus_abc",
            stripe_subscription_id="sub_xyz",
            tier=SubscriptionTier.STARTER,
            metered_enabled=False,
            tokens_input_item_id=None,
            tokens_output_item_id=None,
            debates_item_id=None,
        )

        assert config.tokens_input_item_id is None
        assert config.tokens_output_item_id is None
        assert config.debates_item_id is None
