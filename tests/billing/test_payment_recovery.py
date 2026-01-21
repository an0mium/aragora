"""
Tests for aragora.billing.payment_recovery module.

Tests cover:
- PaymentFailure dataclass and properties
- PaymentRecoveryStore operations
- Grace period calculations
- Status transitions
"""

import pytest
import tempfile
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from aragora.billing.payment_recovery import (
    PaymentFailure,
    PaymentRecoveryStore,
    PAYMENT_GRACE_DAYS,
)


# ============================================================================
# PaymentFailure Dataclass Tests
# ============================================================================


class TestPaymentFailure:
    """Tests for PaymentFailure dataclass."""

    def test_create_payment_failure(self):
        """Test creating a payment failure record."""
        now = datetime.now(timezone.utc)
        failure = PaymentFailure(
            id="fail-123",
            org_id="org-456",
            stripe_customer_id="cus_abc",
            stripe_subscription_id="sub_xyz",
            first_failure_at=now,
            last_failure_at=now,
            attempt_count=1,
        )

        assert failure.id == "fail-123"
        assert failure.org_id == "org-456"
        assert failure.stripe_customer_id == "cus_abc"
        assert failure.stripe_subscription_id == "sub_xyz"
        assert failure.attempt_count == 1
        assert failure.status == "failing"

    def test_grace_period_auto_calculated(self):
        """Test that grace_ends_at is auto-calculated from first_failure_at."""
        now = datetime.now(timezone.utc)
        failure = PaymentFailure(
            id="fail-1",
            org_id="org-1",
            stripe_customer_id="cus_1",
            stripe_subscription_id="sub_1",
            first_failure_at=now,
            last_failure_at=now,
            attempt_count=1,
        )

        expected_grace_end = now + timedelta(days=PAYMENT_GRACE_DAYS)
        # Allow for small time differences
        assert abs((failure.grace_ends_at - expected_grace_end).total_seconds()) < 1

    def test_days_failing_property(self):
        """Test days_failing property calculation."""
        # Failure started 5 days ago
        five_days_ago = datetime.now(timezone.utc) - timedelta(days=5)
        failure = PaymentFailure(
            id="fail-1",
            org_id="org-1",
            stripe_customer_id="cus_1",
            stripe_subscription_id=None,
            first_failure_at=five_days_ago,
            last_failure_at=datetime.now(timezone.utc),
            attempt_count=3,
        )

        assert failure.days_failing == 5

    def test_days_until_downgrade_property(self):
        """Test days_until_downgrade property calculation."""
        # Failure started 3 days ago (grace is 10 days by default)
        three_days_ago = datetime.now(timezone.utc) - timedelta(days=3)
        failure = PaymentFailure(
            id="fail-1",
            org_id="org-1",
            stripe_customer_id="cus_1",
            stripe_subscription_id=None,
            first_failure_at=three_days_ago,
            last_failure_at=datetime.now(timezone.utc),
            attempt_count=2,
        )

        # Allow for partial day calculations (6-7 days remaining)
        assert PAYMENT_GRACE_DAYS - 4 <= failure.days_until_downgrade <= PAYMENT_GRACE_DAYS - 2

    def test_days_until_downgrade_never_negative(self):
        """Test that days_until_downgrade is never negative."""
        # Failure from 20 days ago - well past grace period
        twenty_days_ago = datetime.now(timezone.utc) - timedelta(days=20)
        failure = PaymentFailure(
            id="fail-1",
            org_id="org-1",
            stripe_customer_id="cus_1",
            stripe_subscription_id=None,
            first_failure_at=twenty_days_ago,
            last_failure_at=datetime.now(timezone.utc),
            attempt_count=4,
        )

        assert failure.days_until_downgrade == 0

    def test_should_downgrade_false_within_grace(self):
        """Test should_downgrade is False within grace period."""
        now = datetime.now(timezone.utc)
        failure = PaymentFailure(
            id="fail-1",
            org_id="org-1",
            stripe_customer_id="cus_1",
            stripe_subscription_id=None,
            first_failure_at=now,
            last_failure_at=now,
            attempt_count=1,
        )

        assert failure.should_downgrade is False

    def test_should_downgrade_true_after_grace(self):
        """Test should_downgrade is True after grace period ends."""
        # Failure from 15 days ago - past 10-day grace period
        fifteen_days_ago = datetime.now(timezone.utc) - timedelta(days=15)
        failure = PaymentFailure(
            id="fail-1",
            org_id="org-1",
            stripe_customer_id="cus_1",
            stripe_subscription_id=None,
            first_failure_at=fifteen_days_ago,
            last_failure_at=datetime.now(timezone.utc),
            attempt_count=4,
        )

        assert failure.should_downgrade is True

    def test_to_dict(self):
        """Test to_dict conversion for API response."""
        now = datetime.now(timezone.utc)
        failure = PaymentFailure(
            id="fail-123",
            org_id="org-456",
            stripe_customer_id="cus_abc",
            stripe_subscription_id="sub_xyz",
            first_failure_at=now,
            last_failure_at=now,
            attempt_count=2,
            invoice_id="inv_123",
            invoice_url="https://stripe.com/invoice/123",
            status="failing",
        )

        result = failure.to_dict()

        assert result["id"] == "fail-123"
        assert result["org_id"] == "org-456"
        assert result["stripe_customer_id"] == "cus_abc"
        assert result["stripe_subscription_id"] == "sub_xyz"
        assert result["attempt_count"] == 2
        assert result["invoice_id"] == "inv_123"
        assert result["invoice_url"] == "https://stripe.com/invoice/123"
        assert result["status"] == "failing"
        assert "days_failing" in result
        assert "days_until_downgrade" in result
        assert "should_downgrade" in result

    def test_optional_fields_none(self):
        """Test that optional fields can be None."""
        now = datetime.now(timezone.utc)
        failure = PaymentFailure(
            id="fail-1",
            org_id="org-1",
            stripe_customer_id="cus_1",
            stripe_subscription_id=None,
            first_failure_at=now,
            last_failure_at=now,
            attempt_count=1,
            invoice_id=None,
            invoice_url=None,
        )

        assert failure.stripe_subscription_id is None
        assert failure.invoice_id is None
        assert failure.invoice_url is None


# ============================================================================
# PaymentRecoveryStore Tests
# ============================================================================


class TestPaymentRecoveryStore:
    """Tests for PaymentRecoveryStore SQLite operations."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_recovery.db"
            yield str(db_path)

    @pytest.fixture
    def store(self, temp_db):
        """Create a PaymentRecoveryStore with temp database."""
        store = PaymentRecoveryStore(db_path=temp_db)
        return store

    def test_store_creation(self, temp_db):
        """Test store can be created."""
        store = PaymentRecoveryStore(db_path=temp_db)
        assert store is not None

    def test_record_failure_new(self, store):
        """Test recording a new payment failure."""
        failure = store.record_failure(
            org_id="org-123",
            stripe_customer_id="cus_abc",
            stripe_subscription_id="sub_xyz",
            invoice_id="inv_456",
            invoice_url="https://stripe.com/inv/456",
        )

        assert failure is not None
        assert failure.org_id == "org-123"
        assert failure.stripe_customer_id == "cus_abc"
        assert failure.attempt_count == 1
        assert failure.status == "failing"

    def test_record_failure_increments_count(self, store):
        """Test that recording same org failure increments count."""
        # First failure
        failure1 = store.record_failure(
            org_id="org-123",
            stripe_customer_id="cus_abc",
            stripe_subscription_id="sub_xyz",
        )
        assert failure1.attempt_count == 1

        # Second failure
        failure2 = store.record_failure(
            org_id="org-123",
            stripe_customer_id="cus_abc",
            stripe_subscription_id="sub_xyz",
        )
        assert failure2.attempt_count == 2
        assert failure2.id == failure1.id  # Same record

    def test_get_active_failure_exists(self, store):
        """Test retrieving an existing active failure."""
        store.record_failure(
            org_id="org-123",
            stripe_customer_id="cus_abc",
            stripe_subscription_id="sub_xyz",
        )

        failure = store.get_active_failure("org-123")
        assert failure is not None
        assert failure.org_id == "org-123"

    def test_get_active_failure_not_exists(self, store):
        """Test retrieving non-existent failure returns None."""
        failure = store.get_active_failure("org-nonexistent")
        assert failure is None

    def test_mark_recovered(self, store):
        """Test marking a failure as recovered."""
        store.record_failure(
            org_id="org-123",
            stripe_customer_id="cus_abc",
            stripe_subscription_id="sub_xyz",
        )

        result = store.mark_recovered("org-123")
        assert result is True

        # After recovery, get_active_failure returns None (no longer "failing")
        assert store.get_active_failure("org-123") is None

    def test_mark_downgraded(self, store):
        """Test marking a failure as downgraded."""
        store.record_failure(
            org_id="org-123",
            stripe_customer_id="cus_abc",
            stripe_subscription_id="sub_xyz",
        )

        result = store.mark_downgraded("org-123")
        assert result is True

        # After downgrade, get_active_failure returns None (no longer "failing")
        assert store.get_active_failure("org-123") is None

    def test_get_expired_failures(self, store):
        """Test getting failures past their grace period."""
        now = datetime.now(timezone.utc)
        past_grace = now - timedelta(days=15)

        store.record_failure(
            org_id="org-past",
            stripe_customer_id="cus_past",
            stripe_subscription_id=None,
        )

        # Update the grace_ends_at to simulate expired grace period
        with store.connection() as conn:
            conn.execute(
                """
                UPDATE payment_failures
                SET first_failure_at = ?, grace_ends_at = ?
                WHERE org_id = ?
                """,
                (past_grace.isoformat(), (now - timedelta(days=1)).isoformat(), "org-past"),
            )

        expired = store.get_expired_failures()
        org_ids = [f.org_id for f in expired]

        assert "org-past" in org_ids

    def test_get_failures_needing_notification(self, store):
        """Test getting failures that need notification."""
        now = datetime.now(timezone.utc)
        five_days_ago = now - timedelta(days=5)

        store.record_failure(
            org_id="org-old",
            stripe_customer_id="cus_old",
            stripe_subscription_id=None,
        )

        # Update first_failure_at to simulate old failure
        with store.connection() as conn:
            conn.execute(
                """
                UPDATE payment_failures
                SET first_failure_at = ?
                WHERE org_id = ?
                """,
                (five_days_ago.isoformat(), "org-old"),
            )

        # Get failures older than 3 days
        needing_notification = store.get_failures_needing_notification(days_threshold=3)
        org_ids = [f.org_id for f in needing_notification]

        assert "org-old" in org_ids


# ============================================================================
# Grace Period Edge Cases
# ============================================================================


class TestGracePeriodEdgeCases:
    """Tests for edge cases in grace period handling."""

    def test_grace_period_boundary_exact(self):
        """Test behavior at exact grace period boundary."""
        # Create failure exactly at grace period end
        grace_end = datetime.now(timezone.utc)
        failure = PaymentFailure(
            id="fail-1",
            org_id="org-1",
            stripe_customer_id="cus_1",
            stripe_subscription_id=None,
            first_failure_at=grace_end - timedelta(days=PAYMENT_GRACE_DAYS),
            last_failure_at=datetime.now(timezone.utc),
            attempt_count=4,
            grace_ends_at=grace_end,
        )

        # At exact boundary, should_downgrade should be True
        assert failure.should_downgrade is True

    def test_custom_grace_period(self):
        """Test with custom grace_ends_at value."""
        now = datetime.now(timezone.utc)
        custom_grace_end = now + timedelta(days=30)  # Custom 30-day grace

        failure = PaymentFailure(
            id="fail-1",
            org_id="org-1",
            stripe_customer_id="cus_1",
            stripe_subscription_id=None,
            first_failure_at=now,
            last_failure_at=now,
            attempt_count=1,
            grace_ends_at=custom_grace_end,
        )

        assert failure.grace_ends_at == custom_grace_end
        # Allow for partial day truncation (29-30 days)
        assert 29 <= failure.days_until_downgrade <= 30

    def test_multiple_attempts_same_day(self):
        """Test multiple failure attempts on the same day."""
        now = datetime.now(timezone.utc)
        failure = PaymentFailure(
            id="fail-1",
            org_id="org-1",
            stripe_customer_id="cus_1",
            stripe_subscription_id=None,
            first_failure_at=now,
            last_failure_at=now,
            attempt_count=3,  # Multiple attempts
        )

        # Days failing should still be 0
        assert failure.days_failing == 0


# ============================================================================
# Status Transition Tests
# ============================================================================


class TestStatusTransitions:
    """Tests for payment failure status transitions."""

    def test_status_values(self):
        """Test valid status values."""
        now = datetime.now(timezone.utc)

        for status in ["failing", "recovered", "downgraded"]:
            failure = PaymentFailure(
                id="fail-1",
                org_id="org-1",
                stripe_customer_id="cus_1",
                stripe_subscription_id=None,
                first_failure_at=now,
                last_failure_at=now,
                attempt_count=1,
                status=status,
            )
            assert failure.status == status

    def test_to_dict_includes_status(self):
        """Test that status is included in to_dict output."""
        now = datetime.now(timezone.utc)
        failure = PaymentFailure(
            id="fail-1",
            org_id="org-1",
            stripe_customer_id="cus_1",
            stripe_subscription_id=None,
            first_failure_at=now,
            last_failure_at=now,
            attempt_count=1,
            status="recovered",
        )

        result = failure.to_dict()
        assert result["status"] == "recovered"
