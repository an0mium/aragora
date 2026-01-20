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


# ============================================================================
# Persistence Tests - Watermark Survival Across Restarts
# ============================================================================


class TestUsageSyncPersistence:
    """Tests for sync watermark persistence across service restarts."""

    @pytest.fixture
    def temp_db_dir(self, tmp_path):
        """Create a temporary directory for database files."""
        return tmp_path / ".nomic"

    @pytest.fixture
    def mock_usage_tracker(self):
        """Create a mock UsageTracker."""
        tracker = MagicMock()
        return tracker

    @pytest.fixture
    def mock_stripe_client(self):
        """Create a mock StripeClient that simulates successful reports."""
        client = MagicMock()
        # Return a mock usage record with an id
        mock_record = MagicMock()
        mock_record.id = "usage_record_123"
        client.report_usage = MagicMock(return_value=mock_record)
        return client

    def test_watermarks_persist_to_database(self, temp_db_dir, mock_usage_tracker, mock_stripe_client):
        """Test that sync watermarks are persisted to the database."""
        # Create first service instance
        service1 = UsageSyncService(
            usage_tracker=mock_usage_tracker,
            stripe_client=mock_stripe_client,
            sync_interval=60,
            nomic_dir=temp_db_dir,
        )

        # Simulate syncing by setting watermarks directly
        service1._synced_tokens_in["org-123"] = 10000
        service1._synced_tokens_out["org-123"] = 5000
        service1._synced_debates["org-123"] = 10

        # Save the state
        service1._save_sync_state("org-123")

        # Create a new service instance (simulating restart)
        service2 = UsageSyncService(
            usage_tracker=mock_usage_tracker,
            stripe_client=mock_stripe_client,
            sync_interval=60,
            nomic_dir=temp_db_dir,
        )

        # Verify watermarks were loaded from database
        assert service2._synced_tokens_in.get("org-123") == 10000
        assert service2._synced_tokens_out.get("org-123") == 5000
        assert service2._synced_debates.get("org-123") == 10

    def test_watermarks_survive_restart_no_double_reporting(
        self, temp_db_dir, mock_usage_tracker, mock_stripe_client
    ):
        """Test that watermarks prevent double-reporting after restart.

        This is the main bug fix verification test.
        """
        # Setup mock usage tracker to return consistent usage
        mock_summary = MagicMock()
        mock_summary.total_tokens_in = 15000
        mock_summary.total_tokens_out = 7000
        mock_summary.total_debates = 5
        mock_usage_tracker.get_summary = MagicMock(return_value=mock_summary)

        # Create first service and sync
        service1 = UsageSyncService(
            usage_tracker=mock_usage_tracker,
            stripe_client=mock_stripe_client,
            sync_interval=60,
            nomic_dir=temp_db_dir,
        )

        # Register org with metered billing
        config = OrgBillingConfig(
            org_id="org-456",
            stripe_customer_id="cus_test",
            stripe_subscription_id="sub_test",
            tier=SubscriptionTier.ENTERPRISE,
            metered_enabled=True,
            tokens_input_item_id="si_input",
            tokens_output_item_id="si_output",
        )
        service1.register_org(config)

        # Perform first sync - should report the full usage
        records1 = service1.sync_org(config)

        # Verify first sync reported usage (15000 tokens = 15 x 1K)
        assert len(records1) >= 1  # At least tokens_input reported
        report_calls_1 = mock_stripe_client.report_usage.call_count

        # Reset mock for second service
        mock_stripe_client.reset_mock()

        # Simulate restart - create new service instance
        service2 = UsageSyncService(
            usage_tracker=mock_usage_tracker,
            stripe_client=mock_stripe_client,
            sync_interval=60,
            nomic_dir=temp_db_dir,
        )
        service2.register_org(config)

        # Verify watermarks were loaded
        assert service2._synced_tokens_in.get("org-456") == 15000
        assert service2._synced_tokens_out.get("org-456") == 7000

        # Perform sync after "restart" - should NOT report again (delta = 0)
        records2 = service2.sync_org(config)

        # No new usage to report (delta is 0)
        assert len(records2) == 0

        # Stripe should NOT have been called (no double reporting)
        assert mock_stripe_client.report_usage.call_count == 0

    def test_database_table_created(self, temp_db_dir, mock_usage_tracker, mock_stripe_client):
        """Test that the sync watermark table is created on init."""
        import sqlite3

        service = UsageSyncService(
            usage_tracker=mock_usage_tracker,
            stripe_client=mock_stripe_client,
            sync_interval=60,
            nomic_dir=temp_db_dir,
        )

        # Verify table exists
        db_path = service._db_path
        assert db_path.exists()

        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='usage_sync_watermarks'"
            )
            result = cursor.fetchone()
            assert result is not None
            assert result[0] == "usage_sync_watermarks"

    def test_multiple_orgs_persist_independently(
        self, temp_db_dir, mock_usage_tracker, mock_stripe_client
    ):
        """Test that multiple org watermarks persist independently."""
        service1 = UsageSyncService(
            usage_tracker=mock_usage_tracker,
            stripe_client=mock_stripe_client,
            sync_interval=60,
            nomic_dir=temp_db_dir,
        )

        # Save watermarks for multiple orgs
        service1._synced_tokens_in["org-a"] = 1000
        service1._synced_tokens_in["org-b"] = 2000
        service1._synced_tokens_in["org-c"] = 3000

        service1._save_sync_state("org-a")
        service1._save_sync_state("org-b")
        service1._save_sync_state("org-c")

        # Restart with new service
        service2 = UsageSyncService(
            usage_tracker=mock_usage_tracker,
            stripe_client=mock_stripe_client,
            sync_interval=60,
            nomic_dir=temp_db_dir,
        )

        # Verify all orgs loaded correctly
        assert service2._synced_tokens_in.get("org-a") == 1000
        assert service2._synced_tokens_in.get("org-b") == 2000
        assert service2._synced_tokens_in.get("org-c") == 3000

    def test_new_billing_period_resets_watermarks(
        self, temp_db_dir, mock_usage_tracker, mock_stripe_client
    ):
        """Test that watermarks from a previous billing period are not loaded."""
        import sqlite3
        from datetime import datetime

        # Manually insert a watermark with a different period
        temp_db_dir.mkdir(parents=True, exist_ok=True)
        db_path = temp_db_dir / "billing.db"

        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_sync_watermarks (
                    org_id TEXT NOT NULL,
                    tokens_in INTEGER DEFAULT 0,
                    tokens_out INTEGER DEFAULT 0,
                    debates INTEGER DEFAULT 0,
                    period_start TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (org_id, period_start)
                )
            """)
            # Insert a watermark from last month (different period)
            conn.execute(
                """
                INSERT INTO usage_sync_watermarks
                (org_id, tokens_in, tokens_out, debates, period_start, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                ("org-old", 99999, 88888, 77, "2020-01-01T00:00:00", datetime.utcnow().isoformat()),
            )
            conn.commit()

        # Create service - should NOT load the old period's watermarks
        service = UsageSyncService(
            usage_tracker=mock_usage_tracker,
            stripe_client=mock_stripe_client,
            sync_interval=60,
            nomic_dir=temp_db_dir,
        )

        # Old period watermarks should not be loaded
        assert service._synced_tokens_in.get("org-old") is None


class TestRemainderCarry:
    """Tests for remainder carry and period-end flush."""

    @pytest.fixture
    def temp_db_dir(self, tmp_path):
        """Create a temporary directory for database files."""
        return tmp_path / ".nomic"

    @pytest.fixture
    def mock_usage_tracker(self):
        """Create a mock UsageTracker."""
        tracker = MagicMock()
        return tracker

    @pytest.fixture
    def mock_stripe_client(self):
        """Create a mock StripeClient that simulates successful reports."""
        client = MagicMock()
        mock_record = MagicMock()
        mock_record.id = "usage_record_123"
        client.report_usage = MagicMock(return_value=mock_record)
        return client

    def test_remainder_preserved_during_sync(
        self, mock_usage_tracker, mock_stripe_client, temp_db_dir
    ):
        """Test that sub-1000 token remainders are preserved across syncs.

        When 1500 tokens are used:
        - 1000 should be billed (1 unit)
        - 500 should remain for next sync cycle
        """
        from aragora.billing.usage import UsageSummary

        service = UsageSyncService(
            usage_tracker=mock_usage_tracker,
            stripe_client=mock_stripe_client,
            nomic_dir=temp_db_dir,
        )

        # Register org with metered billing
        from aragora.billing.models import SubscriptionTier
        config = OrgBillingConfig(
            org_id="org-remainder",
            stripe_customer_id="cus_test",
            stripe_subscription_id="sub_test",
            tier=SubscriptionTier.STARTER,
            metered_enabled=True,
            tokens_input_item_id="si_input",
            tokens_output_item_id="si_output",
        )
        service.register_org(config)

        # Mock usage tracker to return 1500 tokens
        mock_usage_tracker.get_summary.return_value = UsageSummary(
            org_id="org-remainder",
            period_start=datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0),
            period_end=datetime.utcnow(),
            total_tokens_in=1500,  # 1000 billable + 500 remainder
            total_tokens_out=0,
            total_debates=0,
        )

        # First sync
        records = service.sync_org(config)

        # Should have billed 1 unit (1000 tokens)
        assert len(records) == 1
        assert records[0].quantity == 1  # 1500 // 1000 = 1 unit
        assert records[0].sync_type == "tokens_input"

        # Watermark should only advance to 1000 (preserving 500 remainder)
        assert service._synced_tokens_in["org-remainder"] == 1000

    def test_remainder_accumulates_across_syncs(
        self, mock_usage_tracker, mock_stripe_client, temp_db_dir
    ):
        """Test that remainders accumulate until threshold is met.

        Sync 1: 800 tokens (0 billed, 800 remainder)
        Sync 2: 1600 total (1000 billed, 600 remainder)
        Sync 3: 2500 total (2000 billed, 500 remainder)
        """
        from aragora.billing.usage import UsageSummary
        from aragora.billing.models import SubscriptionTier

        service = UsageSyncService(
            usage_tracker=mock_usage_tracker,
            stripe_client=mock_stripe_client,
            nomic_dir=temp_db_dir,
        )

        config = OrgBillingConfig(
            org_id="org-accum",
            stripe_customer_id="cus_test",
            stripe_subscription_id="sub_test",
            tier=SubscriptionTier.STARTER,
            metered_enabled=True,
            tokens_input_item_id="si_input",
        )
        service.register_org(config)

        period_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        # Sync 1: 800 tokens (below threshold)
        mock_usage_tracker.get_summary.return_value = UsageSummary(
            org_id="org-accum",
            period_start=period_start,
            period_end=datetime.utcnow(),
            total_tokens_in=800, total_tokens_out=0, total_debates=0
        )
        records = service.sync_org(config)
        assert len(records) == 0  # Nothing billed yet
        assert service._synced_tokens_in.get("org-accum", 0) == 0

        # Sync 2: 1600 total tokens (800 + 800 more)
        mock_usage_tracker.get_summary.return_value = UsageSummary(
            org_id="org-accum",
            period_start=period_start,
            period_end=datetime.utcnow(),
            total_tokens_in=1600, total_tokens_out=0, total_debates=0
        )
        records = service.sync_org(config)
        assert len(records) == 1
        assert records[0].quantity == 1  # 1600 // 1000 = 1 unit
        assert service._synced_tokens_in["org-accum"] == 1000

        # Sync 3: 2500 total tokens
        mock_usage_tracker.get_summary.return_value = UsageSummary(
            org_id="org-accum",
            period_start=period_start,
            period_end=datetime.utcnow(),
            total_tokens_in=2500, total_tokens_out=0, total_debates=0
        )
        records = service.sync_org(config)
        assert len(records) == 1
        assert records[0].quantity == 1  # (2500 - 1000) // 1000 = 1 unit
        assert service._synced_tokens_in["org-accum"] == 2000

    def test_flush_period_bills_remainder(
        self, mock_usage_tracker, mock_stripe_client, temp_db_dir
    ):
        """Test that flush_period bills any remaining sub-threshold tokens."""
        from aragora.billing.usage import UsageSummary
        from aragora.billing.models import SubscriptionTier

        service = UsageSyncService(
            usage_tracker=mock_usage_tracker,
            stripe_client=mock_stripe_client,
            nomic_dir=temp_db_dir,
        )

        config = OrgBillingConfig(
            org_id="org-flush",
            stripe_customer_id="cus_test",
            stripe_subscription_id="sub_test",
            tier=SubscriptionTier.STARTER,
            metered_enabled=True,
            tokens_input_item_id="si_input",
            tokens_output_item_id="si_output",
        )
        service.register_org(config)

        # Set watermark (previously synced 2000)
        service._synced_tokens_in["org-flush"] = 2000
        service._synced_tokens_out["org-flush"] = 1000

        # Total usage is 2500 input, 1300 output (500 and 300 remainder)
        period_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        mock_usage_tracker.get_summary.return_value = UsageSummary(
            org_id="org-flush",
            period_start=period_start,
            period_end=datetime.utcnow(),
            total_tokens_in=2500,  # 500 remainder
            total_tokens_out=1300,  # 300 remainder
            total_debates=0,
        )

        # Flush should bill the remainder
        records = service.flush_period("org-flush")

        # Should bill remainders (rounded up to 1 unit each)
        assert len(records) == 2
        # Input: 500 remainder -> 1 unit
        input_record = next(r for r in records if r.sync_type == "tokens_input")
        assert input_record.quantity == 1
        # Output: 300 remainder -> 1 unit
        output_record = next(r for r in records if r.sync_type == "tokens_output")
        assert output_record.quantity == 1

    def test_auto_flush_on_period_transition(
        self, mock_usage_tracker, mock_stripe_client, temp_db_dir
    ):
        """Test automatic flush when billing period changes."""
        from aragora.billing.usage import UsageSummary
        from aragora.billing.models import SubscriptionTier
        from unittest.mock import patch

        service = UsageSyncService(
            usage_tracker=mock_usage_tracker,
            stripe_client=mock_stripe_client,
            nomic_dir=temp_db_dir,
        )

        config = OrgBillingConfig(
            org_id="org-transition",
            stripe_customer_id="cus_test",
            stripe_subscription_id="sub_test",
            tier=SubscriptionTier.STARTER,
            metered_enabled=True,
            tokens_input_item_id="si_input",
        )
        service.register_org(config)

        # Set last billing period to January
        jan_start = datetime(2024, 1, 1, 0, 0, 0)
        service._last_billing_period = jan_start

        # Mock get_summary for the previous period (January)
        mock_usage_tracker.get_summary.return_value = UsageSummary(
            org_id="org-transition",
            period_start=jan_start,
            period_end=datetime(2024, 1, 31, 23, 59, 59),
            total_tokens_in=500,  # Unbilled remainder from January
            total_tokens_out=0,
            total_debates=0,
        )

        # Mock _get_billing_period_start to return February (new period)
        feb_start = datetime(2024, 2, 1, 0, 0, 0)
        with patch.object(service, "_get_billing_period_start", return_value=feb_start):
            # sync_all should detect transition and flush January remainder
            records = service.sync_all()

        # Should have flushed January's 500 tokens (1 unit)
        flush_records = [r for r in records if r.sync_type == "tokens_input"]
        assert len(flush_records) >= 1
        assert any(r.quantity == 1 for r in flush_records)
