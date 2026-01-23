"""
Tests for Aragora Billing Module.

Tests cover:
- User model and password hashing
- Organization model and tier limits
- Subscription model
- JWT authentication
- Usage tracking
- Auth handlers
- Billing handlers
"""

import json
import os
import sqlite3
import tempfile
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

# =============================================================================
# Model Tests
# =============================================================================


class TestUserModel:
    """Tests for User model."""

    def test_create_user(self):
        from aragora.billing.models import User

        user = User(email="test@example.com", name="Test User")
        assert user.email == "test@example.com"
        assert user.name == "Test User"
        assert user.id  # UUID generated
        assert user.is_active is True
        assert user.role == "member"

    def test_set_password(self):
        from aragora.billing.models import User

        user = User(email="test@example.com")
        user.set_password("securepassword123")

        assert user.password_hash
        # bcrypt embeds salt in hash, so password_salt is empty
        # Hash format: "bcrypt:$2b$12$..."
        assert user.password_hash.startswith("bcrypt:")
        assert len(user.password_hash) > 20  # bcrypt hashes are long

    def test_verify_password(self):
        from aragora.billing.models import User

        user = User(email="test@example.com")
        user.set_password("mypassword")

        assert user.verify_password("mypassword") is True
        assert user.verify_password("wrongpassword") is False
        assert user.verify_password("") is False

    def test_generate_api_key(self):
        from aragora.billing.models import User

        user = User(email="test@example.com")
        api_key = user.generate_api_key()

        assert api_key.startswith("ara_")
        assert len(api_key) > 10
        # Only hash and prefix are stored for verification (plaintext is never stored)
        assert user.api_key_hash is not None  # Hash stored
        assert user.api_key_prefix == api_key[:12]  # Prefix stored
        assert user.api_key_created_at is not None

    def test_revoke_api_key(self):
        from aragora.billing.models import User

        user = User(email="test@example.com")
        user.generate_api_key()
        user.revoke_api_key()

        assert user.api_key_hash is None
        assert user.api_key_prefix is None
        assert user.api_key_created_at is None

    def test_user_to_dict(self):
        from aragora.billing.models import User

        user = User(email="test@example.com", name="Test")
        data = user.to_dict()

        assert data["email"] == "test@example.com"
        assert data["name"] == "Test"
        assert "password_hash" not in data
        assert "password_salt" not in data
        assert data["has_api_key"] is False

    def test_user_to_dict_sensitive(self):
        from aragora.billing.models import User

        user = User(email="test@example.com")
        api_key = user.generate_api_key()
        data = user.to_dict(include_sensitive=True)

        # Sensitive mode includes api_key_prefix for identification
        # (plaintext is never stored, only the hash)
        assert "api_key_prefix" in data
        assert data["api_key_prefix"].startswith("ara_")
        # The hash is NOT exposed in to_dict for security
        assert "api_key_hash" not in data

    def test_user_from_dict(self):
        from aragora.billing.models import User

        data = {
            "id": "user-123",
            "email": "test@example.com",
            "name": "Test",
            "role": "admin",
            "is_active": True,
        }
        user = User.from_dict(data)

        assert user.id == "user-123"
        assert user.email == "test@example.com"
        assert user.role == "admin"


class TestOrganizationModel:
    """Tests for Organization model."""

    def test_create_organization(self):
        from aragora.billing.models import Organization, SubscriptionTier

        org = Organization(name="Test Org", slug="test-org")
        assert org.name == "Test Org"
        assert org.slug == "test-org"
        assert org.tier == SubscriptionTier.FREE
        assert org.debates_used_this_month == 0

    def test_tier_limits(self):
        from aragora.billing.models import Organization, SubscriptionTier

        org = Organization(tier=SubscriptionTier.PROFESSIONAL)
        limits = org.limits

        assert limits.debates_per_month == 200
        assert limits.users_per_org == 10
        assert limits.api_access is True
        assert limits.all_agents is True

    def test_debates_remaining(self):
        from aragora.billing.models import Organization, SubscriptionTier

        org = Organization(tier=SubscriptionTier.FREE)
        assert org.debates_remaining == 10

        org.debates_used_this_month = 7
        assert org.debates_remaining == 3

        org.debates_used_this_month = 15
        assert org.debates_remaining == 0

    def test_is_at_limit(self):
        from aragora.billing.models import Organization, SubscriptionTier

        org = Organization(tier=SubscriptionTier.FREE)
        assert org.is_at_limit is False

        org.debates_used_this_month = 10
        assert org.is_at_limit is True

    def test_increment_debates(self):
        from aragora.billing.models import Organization, SubscriptionTier

        org = Organization(tier=SubscriptionTier.FREE)
        assert org.increment_debates(1) is True
        assert org.debates_used_this_month == 1

        org.debates_used_this_month = 10
        assert org.increment_debates(1) is False  # At limit

    def test_reset_monthly_usage(self):
        from aragora.billing.models import Organization

        org = Organization()
        org.debates_used_this_month = 50
        org.reset_monthly_usage()

        assert org.debates_used_this_month == 0

    def test_generate_slug(self):
        from aragora.billing.models import generate_slug

        assert generate_slug("My Company") == "my-company"
        assert generate_slug("Test & Demo!") == "test-demo"
        assert generate_slug("UPPERCASE") == "uppercase"
        assert generate_slug("  spaces  ") == "spaces"


class TestSubscriptionModel:
    """Tests for Subscription model."""

    def test_create_subscription(self):
        from aragora.billing.models import Subscription, SubscriptionTier

        sub = Subscription(org_id="org-123", tier=SubscriptionTier.STARTER)
        assert sub.org_id == "org-123"
        assert sub.tier == SubscriptionTier.STARTER
        assert sub.status == "active"
        assert sub.is_active is True

    def test_subscription_statuses(self):
        from aragora.billing.models import Subscription

        sub = Subscription()
        assert sub.is_active is True

        sub.status = "canceled"
        assert sub.is_active is False

        sub.status = "trialing"
        assert sub.is_active is True

    def test_days_until_renewal(self):
        from aragora.billing.models import Subscription

        sub = Subscription()
        sub.current_period_end = datetime.now(timezone.utc) + timedelta(days=15, hours=12)
        # Allow for timing variance (14 or 15 days)
        assert 14 <= sub.days_until_renewal <= 15


# =============================================================================
# JWT Authentication Tests
# =============================================================================


class TestJWTAuth:
    """Tests for JWT authentication."""

    def test_create_access_token(self):
        from aragora.billing.jwt_auth import create_access_token, decode_jwt

        token = create_access_token(
            user_id="user-123",
            email="test@example.com",
            org_id="org-456",
            role="admin",
        )

        assert token
        assert "." in token  # JWT format
        parts = token.split(".")
        assert len(parts) == 3

    def test_decode_valid_token(self):
        from aragora.billing.jwt_auth import (
            create_access_token,
            decode_jwt,
            validate_access_token,
        )

        token = create_access_token(
            user_id="user-123",
            email="test@example.com",
        )

        payload = decode_jwt(token)
        assert payload is not None
        assert payload.user_id == "user-123"
        assert payload.email == "test@example.com"
        assert payload.type == "access"

    def test_validate_access_token(self):
        from aragora.billing.jwt_auth import (
            create_access_token,
            validate_access_token,
        )

        token = create_access_token(user_id="user-123", email="test@example.com")
        payload = validate_access_token(token)

        assert payload is not None
        assert payload.type == "access"

    def test_validate_refresh_token(self):
        from aragora.billing.jwt_auth import (
            create_refresh_token,
            validate_refresh_token,
            validate_access_token,
        )

        token = create_refresh_token(user_id="user-123")

        # Should validate as refresh token
        payload = validate_refresh_token(token)
        assert payload is not None
        assert payload.type == "refresh"

        # Should NOT validate as access token
        payload = validate_access_token(token)
        assert payload is None

    def test_expired_token(self):
        from aragora.billing.jwt_auth import (
            create_access_token,
            decode_jwt,
        )

        # Create a token with minimum expiry (1 hour)
        # Note: expiry_hours=0 gets capped to 1 hour for security
        token = create_access_token(
            user_id="user-123",
            email="test@example.com",
            expiry_hours=1,  # Minimum allowed
        )

        # Mock time.time() to simulate 2 hours in the future
        original_time = time.time
        try:
            time.time = lambda: original_time() + 7200  # 2 hours later
            payload = decode_jwt(token)
            assert payload is None  # Expired
        finally:
            time.time = original_time

    def test_invalid_token_format(self):
        from aragora.billing.jwt_auth import decode_jwt

        assert decode_jwt("invalid") is None
        assert decode_jwt("not.a.jwt") is None
        assert decode_jwt("") is None

    def test_token_pair(self):
        from aragora.billing.jwt_auth import create_token_pair

        pair = create_token_pair(
            user_id="user-123",
            email="test@example.com",
            org_id="org-456",
        )

        assert pair.access_token
        assert pair.refresh_token
        assert pair.token_type == "Bearer"
        assert pair.expires_in > 0

        data = pair.to_dict()
        assert "access_token" in data
        assert "refresh_token" in data


class TestUserAuthContext:
    """Tests for UserAuthContext."""

    def test_auth_context_defaults(self):
        from aragora.billing.jwt_auth import UserAuthContext

        ctx = UserAuthContext()
        assert ctx.authenticated is False
        assert ctx.is_authenticated is False
        assert ctx.is_owner is False
        assert ctx.is_admin is False

    def test_auth_context_roles(self):
        from aragora.billing.jwt_auth import UserAuthContext

        ctx = UserAuthContext(authenticated=True, role="owner")
        assert ctx.is_owner is True
        assert ctx.is_admin is True

        ctx = UserAuthContext(authenticated=True, role="admin")
        assert ctx.is_owner is False
        assert ctx.is_admin is True

        ctx = UserAuthContext(authenticated=True, role="member")
        assert ctx.is_owner is False
        assert ctx.is_admin is False


# =============================================================================
# Usage Tracking Tests
# =============================================================================


class TestUsageTracking:
    """Tests for usage tracking system."""

    def test_create_usage_event(self):
        from aragora.billing.usage import UsageEvent, UsageEventType

        event = UsageEvent(
            user_id="user-123",
            org_id="org-456",
            event_type=UsageEventType.DEBATE,
            tokens_in=1000,
            tokens_out=500,
            provider="anthropic",
            model="claude-sonnet-4",
        )

        assert event.user_id == "user-123"
        assert event.tokens_in == 1000
        assert event.tokens_out == 500
        # Calculate cost
        event.calculate_cost()
        assert event.cost_usd > 0

    def test_calculate_token_cost(self):
        from aragora.billing.usage import calculate_token_cost

        # Test known pricing - function signature is (provider, model, tokens_in, tokens_out)
        cost = calculate_token_cost(
            provider="anthropic",
            model="claude-sonnet-4",
            tokens_in=1_000_000,
            tokens_out=0,
        )
        assert cost == Decimal("3.00")  # $3 per 1M input tokens

    def test_usage_tracker_record(self):
        from pathlib import Path
        from aragora.billing.usage import UsageTracker, UsageEvent, UsageEventType

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "usage.db"

            tracker = UsageTracker(db_path)

            # Record some usage using record_debate helper
            event = tracker.record_debate(
                user_id="user-123",
                org_id="org-456",
                debate_id="debate-789",
                tokens_in=1000,
                tokens_out=500,
                provider="anthropic",
                model="claude-sonnet-4",
            )

            assert event.id is not None
            assert event.tokens_in == 1000
            assert event.cost_usd > 0

    def test_usage_summary(self):
        from aragora.billing.usage import UsageSummary

        now = datetime.now(timezone.utc)
        summary = UsageSummary(
            org_id="org-123",
            period_start=now - timedelta(days=30),
            period_end=now,
            total_debates=10,
            total_tokens_in=30000,
            total_tokens_out=20000,
            total_cost_usd=Decimal("1.50"),
            cost_by_provider={"anthropic": Decimal("1.25"), "openai": Decimal("0.25")},
        )

        data = summary.to_dict()
        assert data["total_debates"] == 10
        assert data["total_tokens_in"] == 30000


# =============================================================================
# Auth Handler Tests
# =============================================================================


class TestAuthHandler:
    """Tests for auth handlers."""

    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.auth import AuthHandler, InMemoryUserStore

        store = InMemoryUserStore()
        ctx = {"user_store": store}
        return AuthHandler(ctx), store

    def test_can_handle(self, handler):
        auth_handler, _ = handler
        assert auth_handler.can_handle("/api/v1/auth/register") is True
        assert auth_handler.can_handle("/api/v1/auth/login") is True
        assert auth_handler.can_handle("/api/v1/auth/me") is True
        assert auth_handler.can_handle("/api/v1/other") is False

    def test_validate_email(self):
        from aragora.server.handlers.auth import validate_email

        assert validate_email("test@example.com")[0] is True
        assert validate_email("user.name+tag@domain.co")[0] is True
        assert validate_email("invalid")[0] is False
        assert validate_email("@domain.com")[0] is False
        assert validate_email("")[0] is False

    def test_validate_password(self):
        from aragora.server.handlers.auth import validate_password

        assert validate_password("securepass123")[0] is True
        assert validate_password("short")[0] is False
        assert validate_password("")[0] is False
        assert validate_password("a" * 200)[0] is False  # Too long


class TestInMemoryUserStore:
    """Tests for in-memory user store."""

    def test_save_and_get_user(self):
        from aragora.server.handlers.auth import InMemoryUserStore
        from aragora.billing.models import User

        store = InMemoryUserStore()
        user = User(id="user-123", email="test@example.com")

        store.save_user(user)

        assert store.get_user_by_id("user-123") == user
        assert store.get_user_by_email("test@example.com") == user
        assert store.get_user_by_email("TEST@EXAMPLE.COM") == user  # Case insensitive

    def test_save_and_get_organization(self):
        from aragora.server.handlers.auth import InMemoryUserStore
        from aragora.billing.models import Organization

        store = InMemoryUserStore()
        org = Organization(id="org-123", name="Test Org")

        store.save_organization(org)

        assert store.get_organization_by_id("org-123") == org

    def test_api_key_lookup(self):
        from aragora.server.handlers.auth import InMemoryUserStore
        from aragora.billing.models import User

        store = InMemoryUserStore()
        user = User(id="user-123", email="test@example.com")
        api_key = user.generate_api_key()

        store.save_user(user)

        assert store.get_user_by_api_key(api_key) == user


# =============================================================================
# Billing Handler Tests
# =============================================================================


class TestBillingHandler:
    """Tests for billing handlers."""

    @pytest.fixture
    def handler(self):
        from aragora.server.handlers.admin import BillingHandler

        ctx = {}
        return BillingHandler(ctx)

    def test_can_handle(self, handler):
        assert handler.can_handle("/api/v1/billing/plans") is True
        assert handler.can_handle("/api/v1/billing/usage") is True
        assert handler.can_handle("/api/v1/billing/checkout") is True
        assert handler.can_handle("/api/v1/webhooks/stripe") is True
        assert handler.can_handle("/api/v1/other") is False

    def test_get_plans(self, handler):
        result = handler._get_plans()
        assert result.status_code == 200

        data = json.loads(result.body)
        assert "plans" in data
        assert len(data["plans"]) == 5  # FREE, STARTER, PROFESSIONAL, ENTERPRISE, ENTERPRISE_PLUS

        # Check plan structure
        starter = next(p for p in data["plans"] if p["id"] == "starter")
        assert starter["name"] == "Starter"
        assert starter["price_monthly_cents"] == 9900
        assert starter["features"]["debates_per_month"] == 50


# =============================================================================
# Stripe Client Tests
# =============================================================================


class TestStripeClient:
    """Tests for Stripe client."""

    def test_stripe_client_init(self):
        from aragora.billing.stripe_client import StripeClient

        client = StripeClient(api_key="sk_test_123")
        assert client.api_key == "sk_test_123"
        assert client._is_configured() is True

    def test_stripe_client_not_configured(self):
        from aragora.billing.stripe_client import StripeClient, StripeConfigError

        client = StripeClient(api_key="")
        assert client._is_configured() is False

        with pytest.raises(StripeConfigError):
            client._request("GET", "/customers")

    def test_encode_form_data(self):
        from aragora.billing.stripe_client import StripeClient

        client = StripeClient(api_key="test")

        # Simple data - accept both encoded and unencoded @ symbol
        result = client._encode_form_data({"email": "test@example.com"})
        assert "email=" in result
        assert "test" in result and "example.com" in result

        # Nested data
        result = client._encode_form_data({"metadata": {"user_id": "123", "org_id": "456"}})
        assert "metadata[user_id]=123" in result
        assert "metadata[org_id]=456" in result

    def test_tier_mapping(self):
        from aragora.billing.stripe_client import (
            get_tier_from_price_id,
            get_price_id_for_tier,
            STRIPE_PRICES,
        )
        from aragora.billing.models import SubscriptionTier

        # Set a test price ID
        with patch.dict(
            STRIPE_PRICES,
            {SubscriptionTier.STARTER: "price_starter_123"},
        ):
            assert get_tier_from_price_id("price_starter_123") == SubscriptionTier.STARTER
            assert get_tier_from_price_id("unknown") is None
            assert get_price_id_for_tier(SubscriptionTier.STARTER) == "price_starter_123"


class TestWebhookVerification:
    """Tests for Stripe webhook verification."""

    def test_verify_webhook_signature_invalid(self):
        from aragora.billing.stripe_client import verify_webhook_signature

        # No secret configured
        assert verify_webhook_signature(b"payload", "sig") is False

    def test_verify_webhook_signature_format(self):
        from aragora.billing.stripe_client import verify_webhook_signature
        import hmac
        import hashlib

        secret = "whsec_test_secret"
        payload = b'{"type":"test"}'
        timestamp = str(int(time.time()))

        # Create valid signature
        signed_payload = f"{timestamp}.".encode() + payload
        expected_sig = hmac.new(
            secret.encode("utf-8"),
            signed_payload,
            hashlib.sha256,
        ).hexdigest()

        signature = f"t={timestamp},v1={expected_sig}"

        with patch("aragora.billing.stripe_client.STRIPE_WEBHOOK_SECRET", secret):
            assert verify_webhook_signature(payload, signature, secret) is True

    def test_parse_webhook_event(self):
        from aragora.billing.stripe_client import parse_webhook_event, WebhookEvent

        # Without valid signature, should return None
        result = parse_webhook_event(b'{"type":"test"}', "invalid")
        assert result is None


class TestWebhookEvent:
    """Tests for WebhookEvent parsing."""

    def test_webhook_event_properties(self):
        from aragora.billing.stripe_client import WebhookEvent

        event = WebhookEvent(
            event_type="customer.subscription.created",
            data={
                "object": {
                    "id": "sub_123",
                    "customer": "cus_456",
                    "metadata": {"user_id": "user-789"},
                }
            },
        )

        assert event.type == "customer.subscription.created"
        assert event.subscription_id == "sub_123"
        assert event.customer_id == "cus_456"
        assert event.metadata["user_id"] == "user-789"


# =============================================================================
# Integration Tests
# =============================================================================


class TestBillingIntegration:
    """Integration tests for billing flow."""

    def test_full_registration_flow(self):
        """Test complete user registration flow."""
        from aragora.billing.models import User, Organization
        from aragora.billing.jwt_auth import create_token_pair, validate_access_token
        from aragora.server.handlers.auth import InMemoryUserStore

        store = InMemoryUserStore()

        # Create organization
        org = Organization(name="Test Company", slug="test-company")
        store.save_organization(org)

        # Create user
        user = User(
            email="user@test.com",
            name="Test User",
            org_id=org.id,
            role="owner",
        )
        user.set_password("securepassword")
        org.owner_id = user.id
        store.save_user(user)
        store.save_organization(org)

        # Generate tokens
        tokens = create_token_pair(
            user_id=user.id,
            email=user.email,
            org_id=org.id,
            role=user.role,
        )

        # Validate access token
        payload = validate_access_token(tokens.access_token)
        assert payload is not None
        assert payload.user_id == user.id
        assert payload.org_id == org.id
        assert payload.role == "owner"

        # Verify user can be looked up
        stored_user = store.get_user_by_email("user@test.com")
        assert stored_user.verify_password("securepassword")

    def test_usage_tracking_flow(self):
        """Test complete usage tracking flow."""
        from pathlib import Path
        from aragora.billing.models import Organization, SubscriptionTier
        from aragora.billing.usage import UsageTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "usage.db"

            org = Organization(
                id="org-123",
                name="Test Org",
                tier=SubscriptionTier.STARTER,
            )
            tracker = UsageTracker(db_path)

            # Record multiple usage events using record_debate helper
            for i in range(5):
                tracker.record_debate(
                    user_id="user-123",
                    org_id=org.id,
                    debate_id=f"debate-{i}",
                    tokens_in=1000,
                    tokens_out=500,
                    provider="anthropic",
                    model="claude-sonnet-4",
                )

            # Check org limits
            org.debates_used_this_month = 5
            assert org.debates_remaining == 45  # STARTER has 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
