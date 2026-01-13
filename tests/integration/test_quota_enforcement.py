"""
Integration tests for quota enforcement.

Tests cover:
- Free tier debate limit enforcement (10 debates/month)
- Batch request quota checking
- Gauntlet run quota checking
- Usage increment on successful operations
- Upgrade path responses
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from aragora.billing.models import Organization, SubscriptionTier, User, TIER_LIMITS
from aragora.storage.user_store import UserStore


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_users.db"
        yield db_path


@pytest.fixture
def user_store(temp_db):
    """Create a UserStore with temporary database."""
    return UserStore(temp_db)


@pytest.fixture
def free_org(user_store):
    """Create a free tier organization."""
    org = user_store.create_organization(
        name="Free Test Org",
        owner_id="user-123",
    )
    return org


@pytest.fixture
def free_user(user_store, free_org):
    """Create a free tier user."""
    from aragora.billing.models import hash_password

    password_hash, password_salt = hash_password("testpass123")
    user = user_store.create_user(
        email="free@example.com",
        password_hash=password_hash,
        password_salt=password_salt,
        name="Free User",
        org_id=free_org.id,
        role="owner",
    )
    return user


@pytest.fixture
def pro_org(user_store):
    """Create a professional tier organization."""
    org = user_store.create_organization(
        name="Pro Test Org",
        owner_id="user-pro",
    )
    # Upgrade to professional
    user_store.update_organization(org.id, tier="professional")
    return user_store.get_organization_by_id(org.id)


class TestOrganizationQuota:
    """Tests for Organization quota tracking."""

    def test_free_tier_limits(self):
        """Free tier should have 10 debates/month limit."""
        org = Organization(tier=SubscriptionTier.FREE)
        assert org.limits.debates_per_month == 10
        assert org.debates_remaining == 10
        assert org.is_at_limit is False

    def test_at_limit_detection(self):
        """Organization should report at_limit correctly."""
        org = Organization(tier=SubscriptionTier.FREE)
        org.debates_used_this_month = 10

        assert org.is_at_limit is True
        assert org.debates_remaining == 0

    def test_increment_debates(self):
        """Increment should track usage correctly."""
        org = Organization(tier=SubscriptionTier.FREE)
        assert org.debates_used_this_month == 0

        # Increment by 1
        result = org.increment_debates(1)
        assert result is True
        assert org.debates_used_this_month == 1

        # Increment to limit
        org.debates_used_this_month = 9
        result = org.increment_debates(1)
        assert result is True
        assert org.debates_used_this_month == 10

        # At limit - should fail
        assert org.is_at_limit is True
        result = org.increment_debates(1)
        assert result is False

    def test_tier_limits_defined(self):
        """All tiers should have defined limits."""
        for tier in SubscriptionTier:
            limits = TIER_LIMITS[tier]
            assert limits.debates_per_month > 0
            assert limits.users_per_org > 0


class TestUserStoreUsageTracking:
    """Tests for UserStore usage tracking."""

    def test_increment_usage(self, user_store, free_org):
        """User store should increment org usage."""
        org_id = free_org.id

        new_count = user_store.increment_usage(org_id, 1)
        assert new_count == 1

        new_count = user_store.increment_usage(org_id, 5)
        assert new_count == 6

        # Verify persisted
        org = user_store.get_organization_by_id(org_id)
        assert org.debates_used_this_month == 6

    def test_increment_usage_multiple(self, user_store, free_org):
        """User store should handle bulk increments."""
        org_id = free_org.id

        new_count = user_store.increment_usage(org_id, 10)
        assert new_count == 10

        org = user_store.get_organization_by_id(org_id)
        assert org.is_at_limit is True


class TestRequireQuotaDecorator:
    """Tests for the @require_quota decorator."""

    def test_decorator_import(self):
        """Decorator should be importable from base."""
        from aragora.server.handlers.base import require_quota

        assert callable(require_quota)

    def test_quota_check_blocks_at_limit(self, user_store, free_org, free_user):
        """Should block requests when org is at limit."""
        from aragora.server.handlers.base import require_quota, json_response

        # Set org to limit
        user_store.increment_usage(free_org.id, 10)

        # Create mock handler
        mock_handler = MagicMock()
        mock_handler.headers = {"Authorization": "Bearer test-token"}
        mock_handler.__class__.user_store = user_store

        # Create mock user context
        mock_user_ctx = MagicMock()
        mock_user_ctx.is_authenticated = True
        mock_user_ctx.org_id = free_org.id
        mock_user_ctx.user_id = free_user.id
        mock_user_ctx.role = "owner"

        @require_quota()
        def test_handler(handler, user=None):
            return json_response({"success": True})

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_user_ctx
        ):
            result = test_handler(mock_handler)

        # Should return 429 quota exceeded
        assert result.status_code == 429
        body = json.loads(result.body)
        assert body["code"] == "quota_exceeded"
        assert body["upgrade_url"] == "/pricing"

    def test_quota_check_allows_under_limit(self, user_store, free_org, free_user):
        """Should allow requests when under limit."""
        from aragora.server.handlers.base import require_quota, json_response

        # Set org to under limit
        user_store.increment_usage(free_org.id, 5)

        mock_handler = MagicMock()
        mock_handler.headers = {"Authorization": "Bearer test-token"}
        mock_handler.__class__.user_store = user_store

        mock_user_ctx = MagicMock()
        mock_user_ctx.is_authenticated = True
        mock_user_ctx.org_id = free_org.id
        mock_user_ctx.user_id = free_user.id
        mock_user_ctx.role = "owner"

        @require_quota()
        def test_handler(handler, user=None):
            return json_response({"success": True})

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_user_ctx
        ):
            result = test_handler(mock_handler)

        # Should succeed
        assert result.status_code == 200

    def test_quota_check_batch_insufficient(self, user_store, free_org, free_user):
        """Should block batch when insufficient quota remaining."""
        from aragora.server.handlers.base import require_quota, json_response

        # Set org to 8/10 used
        user_store.increment_usage(free_org.id, 8)

        mock_handler = MagicMock()
        mock_handler.headers = {"Authorization": "Bearer test-token"}
        mock_handler.__class__.user_store = user_store

        mock_user_ctx = MagicMock()
        mock_user_ctx.is_authenticated = True
        mock_user_ctx.org_id = free_org.id
        mock_user_ctx.user_id = free_user.id

        # Request 5 debates, but only 2 remaining
        @require_quota(debate_count=5)
        def test_batch_handler(handler, user=None):
            return json_response({"success": True})

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_user_ctx
        ):
            result = test_batch_handler(mock_handler)

        # Should return 429 with insufficient quota
        assert result.status_code == 429
        body = json.loads(result.body)
        assert body["code"] == "quota_insufficient"
        assert body["requested"] == 5
        assert body["remaining"] == 2


class TestQuotaEnforcementEndToEnd:
    """End-to-end tests for quota enforcement."""

    def test_free_tier_limit_enforced(self, user_store, free_org, free_user):
        """Free tier should be limited to 10 debates."""
        org_id = free_org.id

        # Simulate 10 successful debates
        for i in range(10):
            user_store.increment_usage(org_id, 1)

        # Verify at limit
        org = user_store.get_organization_by_id(org_id)
        assert org.debates_used_this_month == 10
        assert org.is_at_limit is True
        assert org.debates_remaining == 0

    def test_pro_tier_higher_limit(self, user_store, pro_org):
        """Professional tier should have higher limit."""
        org_id = pro_org.id

        # Simulate 100 debates (free limit is 10)
        user_store.increment_usage(org_id, 100)

        org = user_store.get_organization_by_id(org_id)
        assert org.debates_used_this_month == 100
        # Pro tier has 200 limit
        assert org.is_at_limit is False
        assert org.debates_remaining == 100  # 200 - 100

    def test_quota_response_format(self, user_store, free_org):
        """Quota exceeded response should include upgrade info."""
        from aragora.server.handlers.base import require_quota, json_response

        # Set to limit
        user_store.increment_usage(free_org.id, 10)

        mock_handler = MagicMock()
        mock_handler.__class__.user_store = user_store

        mock_user_ctx = MagicMock()
        mock_user_ctx.is_authenticated = True
        mock_user_ctx.org_id = free_org.id

        @require_quota()
        def test_handler(handler, user=None):
            return json_response({"success": True})

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_user_ctx
        ):
            result = test_handler(mock_handler)

        body = json.loads(result.body)

        # Verify response format
        assert "error" in body
        assert body["code"] == "quota_exceeded"
        assert body["limit"] == 10
        assert body["used"] == 10
        assert body["remaining"] == 0
        assert body["tier"] == "free"
        assert body["upgrade_url"] == "/pricing"
        assert "message" in body


class TestUsageIncrement:
    """Tests for usage increment on success."""

    def test_increment_on_success(self, user_store, free_org, free_user):
        """Usage should be incremented after successful operation."""
        from aragora.server.handlers.base import require_quota, json_response, HandlerResult

        mock_handler = MagicMock()
        mock_handler.__class__.user_store = user_store

        mock_user_ctx = MagicMock()
        mock_user_ctx.is_authenticated = True
        mock_user_ctx.org_id = free_org.id

        @require_quota()
        def test_handler(handler, user=None):
            return json_response({"success": True})

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_user_ctx
        ):
            result = test_handler(mock_handler)

        assert result.status_code == 200

        # Verify usage was incremented
        org = user_store.get_organization_by_id(free_org.id)
        assert org.debates_used_this_month == 1

    def test_no_increment_on_failure(self, user_store, free_org, free_user):
        """Usage should NOT be incremented on error response."""
        from aragora.server.handlers.base import require_quota, error_response

        initial_usage = user_store.get_organization_by_id(free_org.id).debates_used_this_month

        mock_handler = MagicMock()
        mock_handler.__class__.user_store = user_store

        mock_user_ctx = MagicMock()
        mock_user_ctx.is_authenticated = True
        mock_user_ctx.org_id = free_org.id

        @require_quota()
        def test_handler(handler, user=None):
            # Simulate error response
            return error_response("Something went wrong", 500)

        with patch(
            "aragora.billing.jwt_auth.extract_user_from_request", return_value=mock_user_ctx
        ):
            result = test_handler(mock_handler)

        assert result.status_code == 500

        # Verify usage was NOT incremented
        org = user_store.get_organization_by_id(free_org.id)
        assert org.debates_used_this_month == initial_usage


class TestMonthlyReset:
    """Tests for monthly usage reset."""

    def test_reset_org_usage(self, user_store, free_org):
        """Monthly reset should clear usage."""
        org_id = free_org.id

        # Use some quota
        user_store.increment_usage(org_id, 8)

        org = user_store.get_organization_by_id(org_id)
        assert org.debates_used_this_month == 8

        # Reset
        user_store.reset_org_usage(org_id)

        org = user_store.get_organization_by_id(org_id)
        assert org.debates_used_this_month == 0
        assert org.is_at_limit is False
