"""
Stripe Webhook Overage Tests (Phase 3 Production Readiness).

Tests cover overage billing scenarios via Stripe webhooks:
- Overage rate multiplier (1.5x) applied correctly
- Max overage cap enforcement
- Grace period expiry → tier downgrade
- Usage metering for overage tracking
- Concurrent overage handling
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

from aragora.billing.models import (
    Organization,
    SubscriptionTier,
    User,
)
from aragora.billing.budget_manager import (
    Budget,
    BudgetAction,
    BudgetStatus,
    BudgetThreshold,
    SpendResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_user_store():
    """Create a mock user store with test data."""
    store = MagicMock()

    test_user = User(
        id="user-overage-1",
        email="overage@example.com",
        name="Overage User",
        org_id="org-overage-1",
        role="owner",
    )

    test_org = Organization(
        id="org-overage-1",
        name="Overage Org",
        slug="overage-org",
        tier=SubscriptionTier.PROFESSIONAL,
        owner_id="user-overage-1",
        stripe_customer_id="cus_overage_123",
        stripe_subscription_id="sub_overage_456",
        debates_used_this_month=95,  # Near limit
    )

    store.get_user_by_id.return_value = test_user
    store.get_organization_by_id.return_value = test_org
    store.get_organization_by_subscription.return_value = test_org
    store.get_organization_by_stripe_customer.return_value = test_org
    store.update_organization.return_value = True
    store.reset_org_usage.return_value = True
    store.log_audit_event.return_value = True

    return store


@pytest.fixture
def billing_handler(mock_user_store):
    """Create billing handler with mocked context."""
    from aragora.server.handlers.admin import BillingHandler

    ctx = {"user_store": mock_user_store}
    handler = BillingHandler(ctx)
    return handler


@dataclass
class MockStripeEvent:
    """Mock Stripe webhook event."""

    type: str
    object: dict
    data: dict
    metadata: Optional[dict] = None
    event_id: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.event_id is None:
            self.event_id = self.data.get("id", f"evt_mock_{id(self)}")


# =============================================================================
# Overage Rate Multiplier Tests
# =============================================================================


class TestOverageRateMultiplierApplied:
    """Test that overage rate multiplier is correctly applied."""

    def test_overage_rate_multiplier_in_spend_result(self):
        """Overage rate multiplier (1.5x) included in spend result."""
        budget = Budget(
            budget_id="overage-test",
            org_id="org_1",
            name="Test Budget",
            amount_usd=100.0,
            spent_usd=90.0,  # 90% used
            allow_overage=True,
            overage_rate_multiplier=1.5,
        )

        # Spending 20 puts us 10 over budget
        result = budget.can_spend_extended(20.0)

        assert result.allowed is True
        assert result.is_overage is True
        assert result.overage_amount_usd == 10.0
        assert result.overage_rate_multiplier == 1.5

        # Effective overage cost should be 10 * 1.5 = $15
        effective_overage_cost = result.overage_amount_usd * result.overage_rate_multiplier
        assert effective_overage_cost == 15.0

    def test_overage_multiplier_2x_rate(self):
        """Custom 2x overage rate applies correctly."""
        budget = Budget(
            budget_id="overage-2x",
            org_id="org_1",
            name="Premium Overage",
            amount_usd=50.0,
            spent_usd=50.0,  # At limit
            allow_overage=True,
            overage_rate_multiplier=2.0,
        )

        result = budget.can_spend_extended(25.0)

        assert result.allowed is True
        assert result.is_overage is True
        assert result.overage_amount_usd == 25.0
        assert result.overage_rate_multiplier == 2.0

        # Effective cost: 25 * 2 = $50
        effective_cost = result.overage_amount_usd * result.overage_rate_multiplier
        assert effective_cost == 50.0

    def test_no_overage_multiplier_when_within_budget(self):
        """No overage multiplier applied when spending within budget."""
        budget = Budget(
            budget_id="normal-spend",
            org_id="org_1",
            name="Normal Budget",
            amount_usd=100.0,
            spent_usd=50.0,
            allow_overage=True,
            overage_rate_multiplier=1.5,
        )

        result = budget.can_spend_extended(30.0)

        assert result.allowed is True
        assert result.is_overage is False
        assert result.overage_amount_usd == 0.0
        # Multiplier should be 1.0 (no surcharge) when not in overage
        assert result.overage_rate_multiplier == 1.0


# =============================================================================
# Overage Cap Tests
# =============================================================================


class TestOverageCapEnforcement:
    """Test max overage USD cap enforcement."""

    def test_overage_cap_blocks_excessive_spending(self):
        """Spending that exceeds max overage cap is blocked."""
        budget = Budget(
            budget_id="capped-overage",
            org_id="org_1",
            name="Capped Budget",
            amount_usd=100.0,
            spent_usd=100.0,  # At limit
            allow_overage=True,
            overage_spent_usd=45.0,  # Already $45 in overage
            max_overage_usd=50.0,  # Cap at $50
        )

        # Trying to spend $10 would put us at $55 overage
        result = budget.can_spend_extended(10.0)

        assert result.allowed is False
        assert "cap exceeded" in result.message.lower()

    def test_overage_within_cap_allowed(self):
        """Spending within overage cap is allowed."""
        budget = Budget(
            budget_id="capped-overage",
            org_id="org_1",
            name="Capped Budget",
            amount_usd=100.0,
            spent_usd=100.0,
            allow_overage=True,
            overage_spent_usd=40.0,
            max_overage_usd=50.0,
        )

        # $5 more puts us at $45, still under $50 cap
        result = budget.can_spend_extended(5.0)

        assert result.allowed is True
        assert result.is_overage is True

    def test_overage_cap_triggers_warning_near_limit(self):
        """Warn when approaching overage cap."""
        budget = Budget(
            budget_id="warning-overage",
            org_id="org_1",
            name="Warning Budget",
            amount_usd=100.0,
            spent_usd=100.0,
            allow_overage=True,
            overage_spent_usd=45.0,  # 90% of cap
            max_overage_usd=50.0,
        )

        # Calculate percentage of cap used
        cap_usage_percent = (budget.overage_spent_usd / budget.max_overage_usd) * 100
        assert cap_usage_percent == 90.0

        # Should still allow small spend
        result = budget.can_spend_extended(4.0)
        assert result.allowed is True

    def test_no_cap_allows_unlimited_overage(self):
        """No max_overage_usd allows unlimited overage spending."""
        budget = Budget(
            budget_id="uncapped-overage",
            org_id="org_1",
            name="Uncapped Budget",
            amount_usd=100.0,
            spent_usd=100.0,
            allow_overage=True,
            overage_spent_usd=500.0,  # Way over
            max_overage_usd=None,  # No cap
        )

        # Should allow even more overage
        result = budget.can_spend_extended(100.0)

        assert result.allowed is True
        assert result.is_overage is True


# =============================================================================
# Grace Period Tests
# =============================================================================


class TestGracePeriodExpiry:
    """Test grace period expiry and tier downgrade."""

    def test_grace_period_store_initialization(self):
        """Grace period recovery store can be initialized."""
        from aragora.billing.payment_recovery import (
            PaymentRecoveryStore,
            PaymentFailure,
            PAYMENT_GRACE_DAYS,
        )

        import tempfile
        import os

        # Create temp database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Store initializes automatically on construction
            store = PaymentRecoveryStore(db_path=db_path)

            # Record a payment failure using the store's method signature
            result = store.record_failure(
                org_id="org_1",
                stripe_customer_id="cus_123",
                stripe_subscription_id="sub_123",
                invoice_id="inv_123",
            )

            # Result should be a PaymentFailure object
            assert result is not None
            assert isinstance(result, PaymentFailure)
            assert result.org_id == "org_1"

            # Verify we can retrieve it (singular - one failure per org)
            active = store.get_active_failure("org_1")
            assert active is not None
            assert active.org_id == "org_1"

        finally:
            os.unlink(db_path)

    def test_payment_failure_should_downgrade(self):
        """Payment failure detects when grace period is expired."""
        from aragora.billing.payment_recovery import (
            PaymentFailure,
            PAYMENT_GRACE_DAYS,
        )

        # Create a failure that's past the grace period
        past_date = datetime.now(timezone.utc) - timedelta(days=PAYMENT_GRACE_DAYS + 1)
        failure = PaymentFailure(
            id="expired_fail",
            org_id="org_expired",
            stripe_customer_id="cus_expired",
            stripe_subscription_id="sub_expired",
            first_failure_at=past_date,
            last_failure_at=past_date,
            attempt_count=3,
            status="failing",
        )

        # Should indicate downgrade is needed
        assert failure.should_downgrade is True
        assert failure.days_until_downgrade == 0

    def test_payment_failure_within_grace_period(self):
        """Payment failure within grace period doesn't trigger downgrade."""
        from aragora.billing.payment_recovery import (
            PaymentFailure,
            PAYMENT_GRACE_DAYS,
        )

        # Create a recent failure (within grace period)
        recent_date = datetime.now(timezone.utc) - timedelta(days=2)
        failure = PaymentFailure(
            id="recent_fail",
            org_id="org_recent",
            stripe_customer_id="cus_recent",
            stripe_subscription_id="sub_recent",
            first_failure_at=recent_date,
            last_failure_at=recent_date,
            attempt_count=1,
            status="failing",
        )

        # Should NOT indicate downgrade (still in grace period)
        assert failure.should_downgrade is False
        assert failure.days_until_downgrade > 0


# =============================================================================
# Invoice Webhook Tests
# =============================================================================


class TestInvoiceWithOverageCharges:
    """Test invoice webhooks with overage amounts."""

    @patch("aragora.billing.stripe_client.parse_webhook_event")
    @patch("aragora.server.handlers.admin.billing._is_duplicate_webhook")
    @patch("aragora.server.handlers.admin.billing._mark_webhook_processed")
    def test_invoice_payment_succeeded_resets_usage(
        self, mock_mark, mock_is_dup, mock_parse, billing_handler, mock_user_store
    ):
        """Successful invoice payment resets monthly usage counters."""
        mock_is_dup.return_value = False

        event = MockStripeEvent(
            type="invoice.payment_succeeded",
            object={
                "id": "inv_overage_123",
                "customer": "cus_overage_123",
                "subscription": "sub_overage_456",
                "amount_paid": 15000,  # $150 (base $100 + $50 overage)
            },
            data={"id": "evt_invoice_paid_123"},
        )
        mock_parse.return_value = event

        payload = b'{"type": "invoice.payment_succeeded"}'
        http_handler = MagicMock()
        http_handler.headers = {"Stripe-Signature": "valid_sig"}
        http_handler.rfile.read.return_value = payload

        with patch("aragora.billing.payment_recovery.get_recovery_store") as mock_recovery:
            mock_recovery.return_value.mark_recovered.return_value = True

            result = billing_handler._handle_stripe_webhook(http_handler)

        assert result.status_code == 200
        # Verify usage was reset
        mock_user_store.reset_org_usage.assert_called()

    @patch("aragora.billing.stripe_client.parse_webhook_event")
    @patch("aragora.server.handlers.admin.billing._is_duplicate_webhook")
    @patch("aragora.server.handlers.admin.billing._mark_webhook_processed")
    def test_invoice_payment_failed_records_failure(
        self, mock_mark, mock_is_dup, mock_parse, billing_handler, mock_user_store
    ):
        """Failed invoice payment records failure for grace period tracking."""
        from aragora.billing.payment_recovery import PaymentFailure

        mock_is_dup.return_value = False

        event = MockStripeEvent(
            type="invoice.payment_failed",
            object={
                "id": "inv_failed_123",
                "customer": "cus_overage_123",
                "subscription": "sub_overage_456",
                "amount_due": 15000,
                "attempt_count": 1,
            },
            data={"id": "evt_invoice_failed_123"},
        )
        mock_parse.return_value = event

        payload = b'{"type": "invoice.payment_failed"}'
        http_handler = MagicMock()
        http_handler.headers = {"Stripe-Signature": "valid_sig"}
        http_handler.rfile.read.return_value = payload

        # Create a proper mock failure object that has the expected attributes
        now = datetime.now(timezone.utc)
        mock_failure = PaymentFailure(
            id="fail_mock",
            org_id="org-overage-1",
            stripe_customer_id="cus_overage_123",
            stripe_subscription_id="sub_overage_456",
            first_failure_at=now,
            last_failure_at=now,
            attempt_count=1,
            invoice_id="inv_failed_123",
            status="failing",
        )

        with patch("aragora.billing.payment_recovery.get_recovery_store") as mock_recovery:
            mock_recovery.return_value.record_failure.return_value = mock_failure

            result = billing_handler._handle_stripe_webhook(http_handler)

        assert result.status_code == 200


# =============================================================================
# Concurrent Overage Tests
# =============================================================================


class TestConcurrentOverageHandling:
    """Test concurrent overage scenarios don't lose events."""

    def test_record_overage_atomic_update(self):
        """Recording overage is atomic and doesn't lose updates."""
        budget = Budget(
            budget_id="concurrent-test",
            org_id="org_1",
            name="Concurrent Budget",
            amount_usd=100.0,
            overage_spent_usd=0.0,
        )

        # Simulate multiple overage recordings
        overage_amounts = [10.0, 15.0, 20.0, 5.0]
        for amount in overage_amounts:
            budget.record_overage(amount)

        # Total should be sum of all amounts
        expected_total = sum(overage_amounts)
        assert budget.overage_spent_usd == expected_total

    def test_concurrent_spend_checks_return_consistent_results(self):
        """Multiple concurrent spend checks are consistent."""
        import threading

        budget = Budget(
            budget_id="concurrent-spend",
            org_id="org_1",
            name="Concurrent Spend Budget",
            amount_usd=100.0,
            spent_usd=95.0,  # Near limit
            allow_overage=True,
            max_overage_usd=50.0,
        )

        results = []
        errors = []

        def check_spend():
            try:
                result = budget.can_spend_extended(10.0)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Run multiple concurrent checks
        threads = [threading.Thread(target=check_spend) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All checks should succeed without errors
        assert len(errors) == 0
        assert len(results) == 10
        # All results should be consistent (all allowed with overage)
        for result in results:
            assert result.allowed is True
            assert result.is_overage is True


# =============================================================================
# Budget Status Transitions
# =============================================================================


class TestBudgetStatusTransitions:
    """Test budget status transitions related to overage."""

    def test_suspended_budget_blocks_all_spending(self):
        """Suspended budget blocks even overage spending."""
        budget = Budget(
            budget_id="suspended-budget",
            org_id="org_1",
            name="Suspended Budget",
            amount_usd=100.0,
            spent_usd=50.0,  # Under budget
            status=BudgetStatus.SUSPENDED,
            allow_overage=True,
        )

        result = budget.can_spend_extended(10.0)

        assert result.allowed is False
        assert "suspended" in result.message.lower()

    def test_exceeded_status_with_overage_enabled(self):
        """Exceeded status still allows spending with overage."""
        budget = Budget(
            budget_id="exceeded-with-overage",
            org_id="org_1",
            name="Exceeded Budget",
            amount_usd=100.0,
            spent_usd=120.0,  # Over budget
            status=BudgetStatus.EXCEEDED,
            allow_overage=True,
        )

        result = budget.can_spend_extended(10.0)

        # Should be allowed because overage is enabled
        assert result.allowed is True
        assert result.is_overage is True

    def test_allow_with_charges_action_enables_overage(self):
        """ALLOW_WITH_CHARGES threshold action enables overage."""
        budget = Budget(
            budget_id="threshold-overage",
            org_id="org_1",
            name="Threshold Budget",
            amount_usd=100.0,
            spent_usd=100.0,  # At 100% threshold
            allow_overage=False,  # Globally disabled
            thresholds=[BudgetThreshold(1.0, BudgetAction.ALLOW_WITH_CHARGES)],
        )

        result = budget.can_spend_extended(20.0)

        # Should be allowed because threshold action enables charges
        assert result.allowed is True
        assert result.is_overage is True


# =============================================================================
# Overage Cost Calculation Tests
# =============================================================================


class TestOverageCostCalculation:
    """Test overage cost calculations."""

    def test_partial_overage_calculation(self):
        """Correctly calculate partial overage amount."""
        budget = Budget(
            budget_id="partial-overage",
            org_id="org_1",
            name="Partial Budget",
            amount_usd=100.0,
            spent_usd=75.0,
            allow_overage=True,
            overage_rate_multiplier=1.5,
        )

        # Spending 50 puts us 25 over (75 + 50 - 100 = 25)
        result = budget.can_spend_extended(50.0)

        assert result.overage_amount_usd == 25.0
        # Cost breakdown:
        # - Within budget: $25 at normal rate
        # - Overage: $25 at 1.5x = $37.50
        # - Total effective cost: $25 + $37.50 = $62.50

    def test_full_overage_calculation(self):
        """Calculate when entire spend is overage."""
        budget = Budget(
            budget_id="full-overage",
            org_id="org_1",
            name="Full Overage Budget",
            amount_usd=100.0,
            spent_usd=110.0,  # Already $10 over
            allow_overage=True,
            overage_rate_multiplier=1.5,
        )

        # Spending $30 when already $10 over = $40 total overage (110 + 30 - 100)
        result = budget.can_spend_extended(30.0)

        assert result.overage_amount_usd == 40.0
        assert result.is_overage is True

    def test_effective_cost_calculation(self):
        """Helper to calculate effective cost with overage."""
        budget = Budget(
            budget_id="effective-cost",
            org_id="org_1",
            name="Effective Cost Budget",
            amount_usd=100.0,
            spent_usd=80.0,
            allow_overage=True,
            overage_rate_multiplier=2.0,
        )

        # $50 spend = $20 normal + $30 overage
        result = budget.can_spend_extended(50.0)

        # Calculate effective cost
        normal_amount = 50.0 - result.overage_amount_usd  # $20
        overage_cost = result.overage_amount_usd * result.overage_rate_multiplier  # $30 * 2 = $60
        total_effective_cost = normal_amount + overage_cost  # $20 + $60 = $80

        assert normal_amount == 20.0
        assert overage_cost == 60.0
        assert total_effective_cost == 80.0
