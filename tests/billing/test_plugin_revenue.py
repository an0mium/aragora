"""
Tests for Plugin Revenue Sharing System.

Covers:
- Revenue split calculations (70/30 default)
- Plugin installation tracking
- Developer balance calculations
- Payout creation and completion
- Revenue history queries
- Edge cases (refunds, minimum threshold)
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aragora.billing.plugin_revenue import (
    DeveloperPayout,
    PluginInstall,
    PluginRevenueEvent,
    PluginRevenueTracker,
    RevenueEventType,
    get_plugin_revenue_tracker,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_revenue.db"
        yield db_path


@pytest.fixture
def tracker(temp_db):
    """Create a tracker instance with a temp database."""
    return PluginRevenueTracker(db_path=temp_db)


class TestPluginRevenueEvent:
    """Tests for PluginRevenueEvent dataclass."""

    def test_calculate_split_default(self):
        """Default split should be 70% developer, 30% platform."""
        event = PluginRevenueEvent(gross_amount_cents=10000)  # $100
        event.calculate_split()

        assert event.developer_amount_cents == 7000  # $70
        assert event.platform_fee_cents == 3000  # $30
        assert event.developer_amount_cents + event.platform_fee_cents == event.gross_amount_cents

    def test_calculate_split_custom(self):
        """Custom split percentages should work correctly."""
        event = PluginRevenueEvent(gross_amount_cents=10000)

        # 80% developer share
        event.calculate_split(developer_share_percent=80)
        assert event.developer_amount_cents == 8000
        assert event.platform_fee_cents == 2000

        # 50% developer share
        event.calculate_split(developer_share_percent=50)
        assert event.developer_amount_cents == 5000
        assert event.platform_fee_cents == 5000

    def test_calculate_split_rounding(self):
        """Split should handle rounding correctly (favor platform on fractional cents)."""
        event = PluginRevenueEvent(gross_amount_cents=333)  # $3.33
        event.calculate_split(developer_share_percent=70)

        # 333 * 0.70 = 233.1, truncated to 233
        assert event.developer_amount_cents == 233
        assert event.platform_fee_cents == 100  # 333 - 233
        assert event.developer_amount_cents + event.platform_fee_cents == event.gross_amount_cents

    def test_to_dict(self):
        """to_dict should return correct dictionary representation."""
        event = PluginRevenueEvent(
            plugin_name="test-plugin",
            plugin_version="1.0.0",
            developer_id="dev-123",
            gross_amount_cents=5000,
            event_type=RevenueEventType.SUBSCRIPTION,
        )
        event.calculate_split()

        d = event.to_dict()
        assert d["plugin_name"] == "test-plugin"
        assert d["plugin_version"] == "1.0.0"
        assert d["developer_id"] == "dev-123"
        assert d["event_type"] == "subscription"
        assert d["gross_amount_cents"] == 5000
        assert d["developer_amount_cents"] == 3500
        assert d["platform_fee_cents"] == 1500


class TestPluginInstall:
    """Tests for PluginInstall dataclass."""

    def test_default_values(self):
        """Default values should be set correctly."""
        install = PluginInstall(plugin_name="test-plugin", org_id="org-123")

        assert install.plugin_name == "test-plugin"
        assert install.org_id == "org-123"
        assert install.subscription_active is False
        assert install.uninstalled_at is None
        assert install.trial_ends_at is None
        assert install.id  # Should have generated UUID

    def test_to_dict(self):
        """to_dict should return correct dictionary representation."""
        install = PluginInstall(
            plugin_name="test-plugin",
            plugin_version="2.0.0",
            org_id="org-456",
            user_id="user-789",
            subscription_active=True,
        )

        d = install.to_dict()
        assert d["plugin_name"] == "test-plugin"
        assert d["plugin_version"] == "2.0.0"
        assert d["org_id"] == "org-456"
        assert d["user_id"] == "user-789"
        assert d["subscription_active"] is True


class TestDeveloperPayout:
    """Tests for DeveloperPayout dataclass."""

    def test_default_values(self):
        """Default values should be set correctly."""
        payout = DeveloperPayout(developer_id="dev-123", amount_cents=5000)

        assert payout.developer_id == "dev-123"
        assert payout.amount_cents == 5000
        assert payout.status == "pending"
        assert payout.currency == "USD"
        assert payout.completed_at is None

    def test_to_dict_with_completed(self):
        """to_dict should handle completed payouts."""
        now = datetime.now(timezone.utc)
        payout = DeveloperPayout(
            developer_id="dev-456",
            amount_cents=10000,
            status="completed",
            stripe_transfer_id="tr_12345",
            completed_at=now,
        )

        d = payout.to_dict()
        assert d["status"] == "completed"
        assert d["stripe_transfer_id"] == "tr_12345"
        assert d["completed_at"] is not None


class TestPluginRevenueTracker:
    """Tests for PluginRevenueTracker."""

    def test_init_creates_schema(self, temp_db):
        """Initialization should create database schema."""
        tracker = PluginRevenueTracker(db_path=temp_db)

        assert temp_db.exists()
        # Verify tables exist by querying them
        with tracker._connection() as conn:
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            table_names = {t["name"] for t in tables}
            assert "plugin_installs" in table_names
            assert "revenue_events" in table_names
            assert "developer_payouts" in table_names

    def test_record_install(self, tracker):
        """record_install should create installation record."""
        install = tracker.record_install(
            plugin_name="my-plugin",
            plugin_version="1.0.0",
            org_id="org-123",
            user_id="user-456",
        )

        assert install.plugin_name == "my-plugin"
        assert install.plugin_version == "1.0.0"
        assert install.org_id == "org-123"
        assert install.user_id == "user-456"
        assert install.trial_ends_at is None

    def test_record_install_with_trial(self, tracker):
        """record_install should set trial end date when trial_days > 0."""
        install = tracker.record_install(
            plugin_name="trial-plugin",
            plugin_version="1.0.0",
            org_id="org-123",
            user_id="user-456",
            trial_days=14,
        )

        assert install.trial_ends_at is not None
        # Trial should end approximately 14 days from now
        expected_end = datetime.now(timezone.utc) + timedelta(days=14)
        assert abs((install.trial_ends_at - expected_end).total_seconds()) < 10

    def test_record_revenue(self, tracker):
        """record_revenue should create revenue event with correct split."""
        event = tracker.record_revenue(
            plugin_name="paid-plugin",
            plugin_version="2.0.0",
            developer_id="dev-123",
            org_id="org-456",
            user_id="user-789",
            event_type=RevenueEventType.INSTALL,
            gross_amount_cents=9900,  # $99
        )

        assert event.plugin_name == "paid-plugin"
        assert event.gross_amount_cents == 9900
        assert event.developer_amount_cents == 6930  # 70% of 9900
        assert event.platform_fee_cents == 2970  # 30% of 9900

    def test_record_revenue_custom_split(self, tracker):
        """record_revenue should respect custom developer share."""
        event = tracker.record_revenue(
            plugin_name="premium-plugin",
            plugin_version="1.0.0",
            developer_id="dev-vip",
            org_id="org-123",
            user_id="user-456",
            event_type=RevenueEventType.SUBSCRIPTION,
            gross_amount_cents=10000,
            developer_share_percent=85,  # Premium developer gets 85%
        )

        assert event.developer_amount_cents == 8500
        assert event.platform_fee_cents == 1500

    def test_get_developer_balance_empty(self, tracker):
        """get_developer_balance should return zeros for new developer."""
        balance = tracker.get_developer_balance("new-dev")

        assert balance["developer_id"] == "new-dev"
        assert balance["total_earnings_cents"] == 0
        assert balance["total_refunds_cents"] == 0
        assert balance["total_payouts_cents"] == 0
        assert balance["available_balance_cents"] == 0
        assert balance["payout_eligible"] is False

    def test_get_developer_balance_with_earnings(self, tracker):
        """get_developer_balance should sum up earnings correctly."""
        dev_id = "earning-dev"

        # Record multiple revenue events
        tracker.record_revenue(
            plugin_name="plugin-a",
            plugin_version="1.0",
            developer_id=dev_id,
            org_id="org-1",
            user_id="user-1",
            event_type=RevenueEventType.INSTALL,
            gross_amount_cents=10000,  # $100, developer gets $70
        )
        tracker.record_revenue(
            plugin_name="plugin-a",
            plugin_version="1.0",
            developer_id=dev_id,
            org_id="org-2",
            user_id="user-2",
            event_type=RevenueEventType.SUBSCRIPTION,
            gross_amount_cents=5000,  # $50, developer gets $35
        )

        balance = tracker.get_developer_balance(dev_id)

        assert balance["total_earnings_cents"] == 7000 + 3500  # 10500
        assert balance["available_balance_cents"] == 10500
        assert balance["payout_eligible"] is True  # Above $10 threshold

    def test_get_developer_balance_below_threshold(self, tracker):
        """Developer with balance below threshold should not be payout eligible."""
        dev_id = "small-dev"

        tracker.record_revenue(
            plugin_name="small-plugin",
            plugin_version="1.0",
            developer_id=dev_id,
            org_id="org-1",
            user_id="user-1",
            event_type=RevenueEventType.USAGE,
            gross_amount_cents=500,  # $5, developer gets $3.50
        )

        balance = tracker.get_developer_balance(dev_id)

        assert balance["available_balance_cents"] == 350
        assert balance["payout_eligible"] is False
        assert balance["min_payout_cents"] == 1000  # $10 minimum

    def test_get_plugin_stats(self, tracker):
        """get_plugin_stats should return correct statistics."""
        plugin = "popular-plugin"

        # Record installs
        tracker.record_install(plugin, "1.0", "org-1", "user-1")
        tracker.record_install(plugin, "1.0", "org-2", "user-2")
        tracker.record_install(plugin, "1.1", "org-3", "user-3")

        # Record revenue
        tracker.record_revenue(
            plugin,
            "1.0",
            "dev-1",
            "org-1",
            "user-1",
            RevenueEventType.INSTALL,
            5000,
        )
        tracker.record_revenue(
            plugin,
            "1.0",
            "dev-1",
            "org-2",
            "user-2",
            RevenueEventType.SUBSCRIPTION,
            2000,
        )

        stats = tracker.get_plugin_stats(plugin)

        assert stats["plugin_name"] == plugin
        assert stats["total_installs"] == 3
        assert stats["current_installs"] == 3  # None uninstalled
        assert stats["total_revenue_cents"] == 7000
        assert stats["transaction_count"] == 2

    def test_create_payout_eligible(self, tracker):
        """create_payout should create payout for eligible developer."""
        dev_id = "payout-dev"

        # Build up balance
        tracker.record_revenue(
            "plugin-1",
            "1.0",
            dev_id,
            "org-1",
            "user-1",
            RevenueEventType.INSTALL,
            20000,  # Developer gets $140
        )

        now = datetime.now(timezone.utc)
        payout = tracker.create_payout(
            developer_id=dev_id,
            period_start=now - timedelta(days=30),
            period_end=now,
        )

        assert payout is not None
        assert payout.developer_id == dev_id
        assert payout.amount_cents == 14000  # 70% of $200
        assert payout.status == "pending"

    def test_create_payout_not_eligible(self, tracker):
        """create_payout should return None for ineligible developer."""
        dev_id = "poor-dev"

        # Small amount below threshold
        tracker.record_revenue(
            "plugin-1",
            "1.0",
            dev_id,
            "org-1",
            "user-1",
            RevenueEventType.USAGE,
            100,  # Developer gets $0.70
        )

        now = datetime.now(timezone.utc)
        payout = tracker.create_payout(
            developer_id=dev_id,
            period_start=now - timedelta(days=30),
            period_end=now,
        )

        assert payout is None

    def test_complete_payout(self, tracker):
        """complete_payout should mark payout as completed."""
        dev_id = "complete-dev"

        # Build balance and create payout
        tracker.record_revenue(
            "plugin-1",
            "1.0",
            dev_id,
            "org-1",
            "user-1",
            RevenueEventType.INSTALL,
            50000,
        )

        now = datetime.now(timezone.utc)
        payout = tracker.create_payout(dev_id, now - timedelta(days=30), now)
        assert payout is not None

        # Complete the payout
        result = tracker.complete_payout(payout.id, "tr_stripe_12345")
        assert result is True

        # Verify balance is reduced
        balance = tracker.get_developer_balance(dev_id)
        assert balance["total_payouts_cents"] == payout.amount_cents
        assert balance["available_balance_cents"] == 0

    def test_complete_payout_idempotent(self, tracker):
        """complete_payout should not complete already completed payout."""
        dev_id = "idempotent-dev"

        tracker.record_revenue(
            "plugin-1",
            "1.0",
            dev_id,
            "org-1",
            "user-1",
            RevenueEventType.INSTALL,
            50000,
        )

        now = datetime.now(timezone.utc)
        payout = tracker.create_payout(dev_id, now - timedelta(days=30), now)

        # Complete once
        result1 = tracker.complete_payout(payout.id, "tr_first")
        assert result1 is True

        # Try to complete again
        result2 = tracker.complete_payout(payout.id, "tr_second")
        assert result2 is False

    def test_get_developer_revenue_history(self, tracker):
        """get_developer_revenue_history should return paginated history."""
        dev_id = "history-dev"

        # Create multiple events
        for i in range(10):
            tracker.record_revenue(
                f"plugin-{i}",
                "1.0",
                dev_id,
                f"org-{i}",
                f"user-{i}",
                RevenueEventType.SUBSCRIPTION,
                1000 * (i + 1),
            )

        # Get first page
        history = tracker.get_developer_revenue_history(dev_id, limit=5, offset=0)
        assert len(history) == 5

        # Get second page
        history2 = tracker.get_developer_revenue_history(dev_id, limit=5, offset=5)
        assert len(history2) == 5

        # Verify different records
        assert history[0]["id"] != history2[0]["id"]


class TestRevenueEventTypes:
    """Tests for different revenue event types."""

    def test_refund_handling(self, tracker):
        """Refunds should reduce available balance correctly."""
        dev_id = "refund-dev"

        # Initial purchase
        tracker.record_revenue(
            "plugin-1",
            "1.0",
            dev_id,
            "org-1",
            "user-1",
            RevenueEventType.INSTALL,
            10000,  # Developer gets $70
        )

        # Refund
        tracker.record_revenue(
            "plugin-1",
            "1.0",
            dev_id,
            "org-1",
            "user-1",
            RevenueEventType.REFUND,
            10000,  # Developer loses $70
        )

        balance = tracker.get_developer_balance(dev_id)
        # Earnings exclude refunds, refunds tracked separately
        assert balance["total_earnings_cents"] == 7000  # Only the purchase
        assert balance["total_refunds_cents"] == 7000  # The refund amount
        # Balance = earnings - refunds = 7000 - 7000 = 0
        assert balance["available_balance_cents"] == 0
        assert balance["payout_eligible"] is False  # Nothing to pay out

    def test_usage_events(self, tracker):
        """Usage-based billing events should be tracked."""
        dev_id = "usage-dev"

        # Multiple usage events
        for _ in range(5):
            tracker.record_revenue(
                "api-plugin",
                "1.0",
                dev_id,
                "org-1",
                "user-1",
                RevenueEventType.USAGE,
                100,  # $1 per use
            )

        balance = tracker.get_developer_balance(dev_id)
        assert balance["total_earnings_cents"] == 350  # 5 * $0.70


class TestDefaultTracker:
    """Tests for default tracker singleton."""

    def test_get_plugin_revenue_tracker(self, monkeypatch, temp_db):
        """get_plugin_revenue_tracker should return singleton instance."""
        import aragora.billing.plugin_revenue as module

        # Reset the singleton
        monkeypatch.setattr(module, "_default_tracker", None)

        # Monkeypatch the default path to use temp
        original_init = PluginRevenueTracker.__init__

        def patched_init(self, db_path=None):
            original_init(self, db_path=temp_db)

        monkeypatch.setattr(PluginRevenueTracker, "__init__", patched_init)

        tracker1 = get_plugin_revenue_tracker()
        tracker2 = get_plugin_revenue_tracker()

        assert tracker1 is tracker2
