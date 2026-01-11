"""
Tests for billing data models.

Covers:
- User creation and password handling
- Organization management
- Subscription tiers and limits
- API key generation
"""

import pytest
from datetime import datetime, timedelta

from aragora.billing.models import (
    User,
    Organization,
    SubscriptionTier,
    TierLimits,
    TIER_LIMITS,
    hash_password,
    verify_password,
)


class TestPasswordHashing:
    """Tests for password hashing functions."""

    def test_hash_password_generates_unique_hashes(self):
        """Each hash should be unique (bcrypt uses random salt internally)."""
        hash1, salt1 = hash_password("password123")
        hash2, salt2 = hash_password("password123")

        # With bcrypt, salt is embedded in hash (salt field is empty)
        # Different hashes are generated each time due to random salt
        assert hash1 != hash2

    def test_hash_password_returns_versioned_hash(self):
        """Hash should have version prefix for migration support."""
        password_hash, salt = hash_password("password123")

        # Should have either bcrypt: or sha256: prefix
        assert password_hash.startswith("bcrypt:") or password_hash.startswith("sha256:")

    def test_verify_password_correct(self):
        """Correct password should verify."""
        password = "secure_password_123"
        password_hash, salt = hash_password(password)

        assert verify_password(password, password_hash, salt) is True

    def test_verify_password_incorrect(self):
        """Incorrect password should not verify."""
        password_hash, salt = hash_password("correct_password")

        assert verify_password("wrong_password", password_hash, salt) is False

    def test_hash_password_different_passwords(self):
        """Different passwords should produce different hashes."""
        hash1, _ = hash_password("password1")
        hash2, _ = hash_password("password2")

        assert hash1 != hash2


class TestUser:
    """Tests for User model."""

    def test_user_creation_generates_id(self):
        """User should get auto-generated ID."""
        user = User(email="test@example.com")

        assert user.id is not None
        assert len(user.id) > 0

    def test_user_set_password(self):
        """Setting password should hash it with version prefix."""
        user = User(email="test@example.com")
        user.set_password("my_password")

        assert user.password_hash != ""
        assert user.password_hash != "my_password"
        # Should have version prefix (bcrypt: or sha256:)
        assert user.password_hash.startswith("bcrypt:") or user.password_hash.startswith("sha256:")

    def test_user_verify_password_correct(self):
        """User should verify correct password."""
        user = User(email="test@example.com")
        user.set_password("my_password")

        assert user.verify_password("my_password") is True

    def test_user_verify_password_incorrect(self):
        """User should reject incorrect password."""
        user = User(email="test@example.com")
        user.set_password("my_password")

        assert user.verify_password("wrong_password") is False

    def test_user_generate_api_key(self):
        """API key generation should follow format and store hash."""
        user = User(email="test@example.com")
        api_key = user.generate_api_key()

        assert api_key.startswith("ara_")
        assert len(api_key) > 15
        # API key is NOT stored in plaintext (security)
        assert user.api_key is None
        # Hash is stored instead
        assert user.api_key_hash is not None
        assert user.api_key_prefix == api_key[:12]
        assert user.api_key_created_at is not None
        # Key can be verified
        assert user.verify_api_key(api_key) is True

    def test_user_revoke_api_key(self):
        """Revoking API key should clear it."""
        user = User(email="test@example.com")
        user.generate_api_key()
        user.revoke_api_key()

        assert user.api_key is None
        assert user.api_key_created_at is None

    def test_user_to_dict_excludes_sensitive(self):
        """Default to_dict should exclude sensitive data."""
        user = User(email="test@example.com")
        user.generate_api_key()

        data = user.to_dict()

        assert "api_key" not in data
        assert data["has_api_key"] is True

    def test_user_to_dict_includes_sensitive(self):
        """With flag, to_dict should include API key."""
        user = User(email="test@example.com")
        user.generate_api_key()

        data = user.to_dict(include_sensitive=True)

        assert "api_key" in data
        assert data["api_key"].startswith("ara_")

    def test_user_from_dict(self):
        """Should reconstruct user from dict."""
        original = User(
            email="test@example.com",
            name="Test User",
            role="admin",
        )
        original.set_password("password")

        data = {
            "id": original.id,
            "email": original.email,
            "name": original.name,
            "role": original.role,
            "password_hash": original.password_hash,
            "password_salt": original.password_salt,
        }

        restored = User.from_dict(data)

        assert restored.id == original.id
        assert restored.email == original.email
        assert restored.verify_password("password")

    def test_user_default_role_is_member(self):
        """Default role should be member."""
        user = User(email="test@example.com")
        assert user.role == "member"


class TestOrganization:
    """Tests for Organization model."""

    def test_organization_creation(self):
        """Organization should be created with defaults."""
        org = Organization(name="Test Org", slug="test-org")

        assert org.id is not None
        assert org.tier == SubscriptionTier.FREE

    def test_organization_default_tier(self):
        """New orgs should default to FREE tier."""
        org = Organization(name="New Org")
        assert org.tier == SubscriptionTier.FREE


class TestSubscriptionTiers:
    """Tests for subscription tier configuration."""

    def test_all_tiers_have_limits(self):
        """Every tier should have defined limits."""
        for tier in SubscriptionTier:
            assert tier in TIER_LIMITS
            limits = TIER_LIMITS[tier]
            assert isinstance(limits, TierLimits)

    def test_free_tier_has_restrictions(self):
        """FREE tier should have appropriate restrictions."""
        limits = TIER_LIMITS[SubscriptionTier.FREE]

        assert limits.api_access is False
        assert limits.all_agents is False
        assert limits.sso_enabled is False
        assert limits.price_monthly_cents == 0

    def test_enterprise_tier_has_all_features(self):
        """ENTERPRISE tier should have all features."""
        limits = TIER_LIMITS[SubscriptionTier.ENTERPRISE]

        assert limits.api_access is True
        assert limits.all_agents is True
        assert limits.custom_agents is True
        assert limits.sso_enabled is True
        assert limits.audit_logs is True
        assert limits.priority_support is True

    def test_tier_limits_to_dict(self):
        """TierLimits should serialize to dict."""
        limits = TIER_LIMITS[SubscriptionTier.PROFESSIONAL]
        data = limits.to_dict()

        assert "debates_per_month" in data
        assert "api_access" in data
        assert isinstance(data["debates_per_month"], int)

    def test_tier_prices_increase(self):
        """Higher tiers should cost more."""
        free_price = TIER_LIMITS[SubscriptionTier.FREE].price_monthly_cents
        starter_price = TIER_LIMITS[SubscriptionTier.STARTER].price_monthly_cents
        pro_price = TIER_LIMITS[SubscriptionTier.PROFESSIONAL].price_monthly_cents
        enterprise_price = TIER_LIMITS[SubscriptionTier.ENTERPRISE].price_monthly_cents

        assert free_price < starter_price < pro_price < enterprise_price

    def test_tier_debate_limits_increase(self):
        """Higher tiers should have more debates."""
        free_debates = TIER_LIMITS[SubscriptionTier.FREE].debates_per_month
        starter_debates = TIER_LIMITS[SubscriptionTier.STARTER].debates_per_month
        pro_debates = TIER_LIMITS[SubscriptionTier.PROFESSIONAL].debates_per_month
        enterprise_debates = TIER_LIMITS[SubscriptionTier.ENTERPRISE].debates_per_month

        assert free_debates < starter_debates < pro_debates < enterprise_debates
