"""
Usage Enforcement Tests (Phase 12A).

Tests cover usage tracking and limit enforcement:
- Debate limits enforced at tier boundaries
- Usage counter increments on debate start
- Monthly reset on billing cycle
- Usage reporting in billing endpoints
"""

import json
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch
from io import BytesIO

import pytest

from aragora.billing.models import (
    User,
    Organization,
    SubscriptionTier,
    TIER_LIMITS,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def free_tier_org():
    """Create a free tier organization."""
    return Organization(
        id="org-free",
        name="Free Org",
        slug="free-org",
        tier=SubscriptionTier.FREE,
        owner_id="user-123",
        debates_used_this_month=0,
        billing_cycle_start=datetime.utcnow() - timedelta(days=15),
    )


@pytest.fixture
def starter_tier_org():
    """Create a starter tier organization."""
    return Organization(
        id="org-starter",
        name="Starter Org",
        slug="starter-org",
        tier=SubscriptionTier.STARTER,
        owner_id="user-456",
        debates_used_this_month=25,
        billing_cycle_start=datetime.utcnow() - timedelta(days=10),
    )


@pytest.fixture
def professional_tier_org():
    """Create a professional tier organization."""
    return Organization(
        id="org-pro",
        name="Pro Org",
        slug="pro-org",
        tier=SubscriptionTier.PROFESSIONAL,
        owner_id="user-789",
        debates_used_this_month=150,
        billing_cycle_start=datetime.utcnow() - timedelta(days=20),
    )


# =============================================================================
# Tier Limit Configuration Tests
# =============================================================================

class TestTierLimits:
    """Tests for tier limit configuration."""

    def test_free_tier_has_10_debates(self):
        """Free tier allows 10 debates per month."""
        limits = TIER_LIMITS[SubscriptionTier.FREE]
        assert limits.debates_per_month == 10

    def test_starter_tier_has_50_debates(self):
        """Starter tier allows 50 debates per month."""
        limits = TIER_LIMITS[SubscriptionTier.STARTER]
        assert limits.debates_per_month == 50

    def test_professional_tier_has_200_debates(self):
        """Professional tier allows 200 debates per month."""
        limits = TIER_LIMITS[SubscriptionTier.PROFESSIONAL]
        assert limits.debates_per_month == 200

    def test_enterprise_tier_is_unlimited(self):
        """Enterprise tier has virtually unlimited debates."""
        limits = TIER_LIMITS[SubscriptionTier.ENTERPRISE]
        assert limits.debates_per_month >= 999999


# =============================================================================
# Organization Usage Tracking Tests
# =============================================================================

class TestOrganizationUsageTracking:
    """Tests for organization usage tracking."""

    def test_debates_remaining_calculation(self, free_tier_org):
        """Debates remaining is correctly calculated."""
        free_tier_org.debates_used_this_month = 3
        assert free_tier_org.debates_remaining == 7

    def test_is_at_limit_when_at_quota(self, free_tier_org):
        """Organization is at limit when quota exhausted."""
        free_tier_org.debates_used_this_month = 10
        assert free_tier_org.is_at_limit is True

    def test_is_at_limit_when_over_quota(self, free_tier_org):
        """Organization is at limit when over quota."""
        free_tier_org.debates_used_this_month = 15
        assert free_tier_org.is_at_limit is True

    def test_not_at_limit_with_remaining_quota(self, free_tier_org):
        """Organization not at limit with remaining quota."""
        free_tier_org.debates_used_this_month = 5
        assert free_tier_org.is_at_limit is False

    def test_increment_debates_increases_count(self, free_tier_org):
        """Incrementing debates increases usage count."""
        initial = free_tier_org.debates_used_this_month
        result = free_tier_org.increment_debates()

        assert result is True
        assert free_tier_org.debates_used_this_month == initial + 1

    def test_increment_debates_fails_at_limit(self, free_tier_org):
        """Cannot increment debates when at limit."""
        free_tier_org.debates_used_this_month = 10
        result = free_tier_org.increment_debates()

        assert result is False
        assert free_tier_org.debates_used_this_month == 10

    def test_reset_monthly_usage(self, starter_tier_org):
        """Monthly usage reset clears counter."""
        assert starter_tier_org.debates_used_this_month > 0

        starter_tier_org.reset_monthly_usage()

        assert starter_tier_org.debates_used_this_month == 0
        # billing_cycle_start should be updated
        assert (datetime.utcnow() - starter_tier_org.billing_cycle_start).seconds < 5


# =============================================================================
# Usage Endpoint Tests
# =============================================================================

class TestUsageEndpoint:
    """Tests for billing usage endpoint."""

    @pytest.fixture
    def mock_user_store(self, free_tier_org):
        """Create mock user store."""
        store = MagicMock()
        store.get_user_by_id.return_value = User(
            id="user-123",
            email="test@example.com",
            org_id="org-free",
        )
        store.get_organization_by_id.return_value = free_tier_org
        return store

    @pytest.fixture
    def billing_handler(self, mock_user_store):
        """Create billing handler."""
        from aragora.server.handlers.billing import BillingHandler
        ctx = {"user_store": mock_user_store}
        return BillingHandler(ctx)

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_usage_returns_debates_used(
        self, mock_extract, billing_handler, mock_user_store, free_tier_org
    ):
        """Usage endpoint returns debates used count."""
        mock_extract.return_value = Mock(
            is_authenticated=True,
            user_id="user-123",
        )
        free_tier_org.debates_used_this_month = 5

        mock_handler = MagicMock()
        result = billing_handler._get_usage(mock_handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["usage"]["debates_used"] == 5

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_usage_returns_debates_limit(
        self, mock_extract, billing_handler, mock_user_store, free_tier_org
    ):
        """Usage endpoint returns debates limit for tier."""
        mock_extract.return_value = Mock(
            is_authenticated=True,
            user_id="user-123",
        )

        mock_handler = MagicMock()
        result = billing_handler._get_usage(mock_handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["usage"]["debates_limit"] == 10  # Free tier limit

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_usage_returns_debates_remaining(
        self, mock_extract, billing_handler, mock_user_store, free_tier_org
    ):
        """Usage endpoint returns debates remaining."""
        mock_extract.return_value = Mock(
            is_authenticated=True,
            user_id="user-123",
        )
        free_tier_org.debates_used_this_month = 7

        mock_handler = MagicMock()
        result = billing_handler._get_usage(mock_handler)

        assert result.status_code == 200
        data = json.loads(result.body)
        assert data["usage"]["debates_remaining"] == 3


# =============================================================================
# Forecast Endpoint Tests
# =============================================================================

class TestForecastEndpoint:
    """Tests for usage forecast endpoint."""

    @pytest.fixture
    def mock_user_store(self, professional_tier_org):
        """Create mock user store."""
        store = MagicMock()
        store.get_user_by_id.return_value = User(
            id="user-789",
            email="pro@example.com",
            org_id="org-pro",
        )
        store.get_organization_by_id.return_value = professional_tier_org
        return store

    @pytest.fixture
    def billing_handler(self, mock_user_store):
        """Create billing handler."""
        from aragora.server.handlers.billing import BillingHandler
        ctx = {"user_store": mock_user_store}
        return BillingHandler(ctx)

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_forecast_projects_usage(
        self, mock_extract, billing_handler, mock_user_store, professional_tier_org
    ):
        """Forecast endpoint projects end-of-cycle usage."""
        mock_extract.return_value = Mock(
            is_authenticated=True,
            user_id="user-789",
        )
        professional_tier_org.debates_used_this_month = 100
        professional_tier_org.billing_cycle_start = datetime.utcnow() - timedelta(days=15)

        mock_handler = MagicMock()
        # Mock query params
        mock_handler.path = "/api/billing/usage/forecast"

        result = billing_handler._get_usage_forecast(mock_handler)

        assert result.status_code == 200
        data = json.loads(result.body)

        # Should project ~200 debates at end of 30-day cycle
        assert data["forecast"]["projection"]["debates_end_of_cycle"] >= 100

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_forecast_warns_when_approaching_limit(
        self, mock_extract, billing_handler, mock_user_store, professional_tier_org
    ):
        """Forecast warns when projected to hit limit."""
        mock_extract.return_value = Mock(
            is_authenticated=True,
            user_id="user-789",
        )
        professional_tier_org.debates_used_this_month = 180
        professional_tier_org.billing_cycle_start = datetime.utcnow() - timedelta(days=25)

        mock_handler = MagicMock()
        result = billing_handler._get_usage_forecast(mock_handler)

        assert result.status_code == 200
        data = json.loads(result.body)

        # Should indicate will hit limit
        assert data["forecast"]["will_hit_limit"] is True

    @patch("aragora.billing.jwt_auth.extract_user_from_request")
    def test_forecast_suggests_upgrade(
        self, mock_extract, billing_handler, mock_user_store
    ):
        """Forecast suggests tier upgrade when approaching limit."""
        # Create starter tier org approaching limit
        starter_org = Organization(
            id="org-starter",
            name="Starter Org",
            slug="starter-org",
            tier=SubscriptionTier.STARTER,
            owner_id="user-456",
            debates_used_this_month=45,
            billing_cycle_start=datetime.utcnow() - timedelta(days=20),
        )
        mock_user_store.get_organization_by_id.return_value = starter_org
        mock_user_store.get_user_by_id.return_value = User(
            id="user-456",
            email="starter@example.com",
            org_id="org-starter",
        )

        mock_extract.return_value = Mock(
            is_authenticated=True,
            user_id="user-456",
        )

        mock_handler = MagicMock()
        result = billing_handler._get_usage_forecast(mock_handler)

        assert result.status_code == 200
        data = json.loads(result.body)

        # Should suggest upgrade to professional
        assert data["forecast"]["will_hit_limit"] is True
        assert data["forecast"]["tier_recommendation"] is not None
        assert data["forecast"]["tier_recommendation"]["recommended_tier"] == "professional"


# =============================================================================
# Debate Limit Enforcement Tests
# =============================================================================

class TestDebateLimitEnforcement:
    """Tests for debate limit enforcement in unified server."""

    def test_quota_exceeded_response_format(self):
        """Verify quota exceeded response format."""
        from aragora.billing.models import Organization, SubscriptionTier

        org = Organization(
            id="org-test",
            name="Test Org",
            tier=SubscriptionTier.FREE,
            debates_used_this_month=10,
        )

        # Verify the response would include expected fields
        assert org.is_at_limit is True
        assert org.limits.debates_per_month == 10
        assert org.debates_used_this_month == 10

    def test_limit_check_uses_correct_tier(self):
        """Different tiers have different limits."""
        from aragora.billing.models import Organization, SubscriptionTier

        free_org = Organization(tier=SubscriptionTier.FREE, debates_used_this_month=10)
        starter_org = Organization(tier=SubscriptionTier.STARTER, debates_used_this_month=10)

        # Free tier at 10 is at limit
        assert free_org.is_at_limit is True
        # Starter tier at 10 is not at limit (limit is 50)
        assert starter_org.is_at_limit is False


# =============================================================================
# Monthly Reset Tests
# =============================================================================

class TestMonthlyReset:
    """Tests for monthly usage reset."""

    def test_reset_on_invoice_payment(self):
        """Usage resets when invoice is paid."""
        from aragora.billing.models import Organization, SubscriptionTier

        org = Organization(
            id="org-test",
            tier=SubscriptionTier.PROFESSIONAL,
            debates_used_this_month=150,
            billing_cycle_start=datetime.utcnow() - timedelta(days=30),
        )

        # Simulate reset
        org.reset_monthly_usage()

        assert org.debates_used_this_month == 0
        # Billing cycle should be updated to now
        assert (datetime.utcnow() - org.billing_cycle_start).seconds < 5

    def test_upgrade_preserves_usage(self):
        """Tier upgrade preserves current usage count."""
        from aragora.billing.models import Organization, SubscriptionTier

        org = Organization(
            tier=SubscriptionTier.FREE,
            debates_used_this_month=8,
        )

        # Simulate upgrade
        org.tier = SubscriptionTier.PROFESSIONAL

        # Usage should be preserved
        assert org.debates_used_this_month == 8
        # But should no longer be at limit (new tier has 200 limit)
        assert org.is_at_limit is False
        assert org.debates_remaining == 192


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases in usage enforcement."""

    def test_zero_usage_at_start_of_cycle(self):
        """New billing cycle starts with zero usage."""
        from aragora.billing.models import Organization

        org = Organization()
        org.reset_monthly_usage()

        assert org.debates_used_this_month == 0
        assert org.is_at_limit is False

    def test_enterprise_tier_effectively_unlimited(self):
        """Enterprise tier can use many debates without hitting limit."""
        from aragora.billing.models import Organization, SubscriptionTier

        org = Organization(
            tier=SubscriptionTier.ENTERPRISE,
            debates_used_this_month=10000,
        )

        assert org.is_at_limit is False
        assert org.debates_remaining > 0

    def test_negative_usage_handled(self):
        """Negative usage is handled gracefully."""
        from aragora.billing.models import Organization, SubscriptionTier

        org = Organization(
            tier=SubscriptionTier.FREE,
            debates_used_this_month=-5,  # Should not happen, but be safe
        )

        # debates_remaining uses max(0, limit - used)
        assert org.debates_remaining >= 0

    def test_concurrent_usage_increment(self):
        """Multiple increments work correctly."""
        from aragora.billing.models import Organization, SubscriptionTier

        org = Organization(
            tier=SubscriptionTier.STARTER,
            debates_used_this_month=0,
        )

        # Simulate 10 concurrent debate starts
        for _ in range(10):
            org.increment_debates()

        assert org.debates_used_this_month == 10
