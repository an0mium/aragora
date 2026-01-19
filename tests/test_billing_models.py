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
        # Only hash and prefix are stored (plaintext never stored)
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

        assert user.api_key_hash is None
        assert user.api_key_prefix is None
        assert user.api_key_created_at is None

    def test_user_to_dict_excludes_sensitive(self):
        """Default to_dict should exclude sensitive data."""
        user = User(email="test@example.com")
        user.generate_api_key()

        data = user.to_dict()

        assert "api_key" not in data
        assert data["has_api_key"] is True

    def test_user_to_dict_includes_sensitive(self):
        """With flag, to_dict should include API key prefix."""
        user = User(email="test@example.com")
        api_key = user.generate_api_key()

        data = user.to_dict(include_sensitive=True)

        # Prefix is available for identification (plaintext never stored)
        assert data["api_key_prefix"] == api_key[:12]
        # Hash is NOT exposed in to_dict for security
        assert "api_key_hash" not in data

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

    def test_organization_limits_property(self):
        """Organization should expose tier limits via property."""
        org = Organization(tier=SubscriptionTier.PROFESSIONAL)
        limits = org.limits

        assert limits == TIER_LIMITS[SubscriptionTier.PROFESSIONAL]
        assert limits.debates_per_month == 200

    def test_organization_debates_remaining(self):
        """Should calculate remaining debates correctly."""
        org = Organization(tier=SubscriptionTier.PROFESSIONAL)
        org.debates_used_this_month = 50

        assert org.debates_remaining == 150  # 200 - 50

    def test_organization_debates_remaining_at_limit(self):
        """Debates remaining should be 0 when at limit."""
        org = Organization(tier=SubscriptionTier.FREE)  # 10 debates
        org.debates_used_this_month = 10

        assert org.debates_remaining == 0

    def test_organization_is_at_limit(self):
        """Should detect when at limit."""
        org = Organization(tier=SubscriptionTier.FREE)  # 10 debates
        org.debates_used_this_month = 5

        assert org.is_at_limit is False

        org.debates_used_this_month = 10
        assert org.is_at_limit is True

    def test_organization_increment_debates_success(self):
        """Should increment debates when under limit."""
        org = Organization(tier=SubscriptionTier.PROFESSIONAL)

        result = org.increment_debates(5)

        assert result is True
        assert org.debates_used_this_month == 5

    def test_organization_increment_debates_at_limit(self):
        """Should fail to increment when at limit."""
        org = Organization(tier=SubscriptionTier.FREE)
        org.debates_used_this_month = 10

        result = org.increment_debates(1)

        assert result is False
        assert org.debates_used_this_month == 10  # Unchanged

    def test_organization_reset_monthly_usage(self):
        """Should reset monthly usage counters."""
        org = Organization(tier=SubscriptionTier.PROFESSIONAL)
        org.debates_used_this_month = 150
        old_cycle = org.billing_cycle_start

        org.reset_monthly_usage()

        assert org.debates_used_this_month == 0
        assert org.billing_cycle_start >= old_cycle

    def test_organization_to_dict(self):
        """Should serialize organization to dict."""
        org = Organization(
            name="Test Org",
            slug="test-org",
            tier=SubscriptionTier.PROFESSIONAL,
            owner_id="owner-123",
        )
        org.debates_used_this_month = 50

        data = org.to_dict()

        assert data["name"] == "Test Org"
        assert data["tier"] == "professional"
        assert data["debates_used_this_month"] == 50
        assert data["debates_remaining"] == 150
        assert "limits" in data

    def test_organization_from_dict(self):
        """Should deserialize organization from dict."""
        data = {
            "id": "org-123",
            "name": "Loaded Org",
            "slug": "loaded-org",
            "tier": "enterprise",
            "debates_used_this_month": 100,
            "created_at": "2024-01-01T00:00:00",
        }

        org = Organization.from_dict(data)

        assert org.id == "org-123"
        assert org.tier == SubscriptionTier.ENTERPRISE
        assert org.debates_used_this_month == 100


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

    def test_enterprise_plus_has_exclusive_features(self):
        """ENTERPRISE_PLUS should have dedicated infrastructure features."""
        limits = TIER_LIMITS[SubscriptionTier.ENTERPRISE_PLUS]

        assert limits.dedicated_infrastructure is True
        assert limits.sla_guarantee is True
        assert limits.custom_model_training is True
        assert limits.private_model_deployment is True
        assert limits.compliance_certifications is True
        assert limits.unlimited_api_calls is True
        assert limits.token_based_billing is True


class TestNeedsRehash:
    """Tests for password rehash detection."""

    def test_needs_rehash_empty_hash(self):
        """Empty hash should need rehash."""
        from aragora.billing.models import needs_rehash

        assert needs_rehash("") is True

    def test_needs_rehash_bcrypt_hash(self):
        """Bcrypt hash should not need rehash (when bcrypt available)."""
        from aragora.billing.models import HAS_BCRYPT, needs_rehash

        password_hash, _ = hash_password("test")
        if HAS_BCRYPT:
            assert needs_rehash(password_hash) is False

    def test_needs_rehash_sha256_hash(self):
        """SHA256 hash should need rehash when bcrypt available."""
        from aragora.billing.models import HAS_BCRYPT, needs_rehash

        sha256_hash = "sha256:abc123def456"
        if HAS_BCRYPT:
            assert needs_rehash(sha256_hash) is True


class TestUserAdvanced:
    """Advanced tests for User model."""

    def test_user_needs_password_rehash(self):
        """User should detect when password needs rehash."""
        user = User(email="test@example.com")
        user.set_password("password")
        # Fresh bcrypt hash should not need rehash
        from aragora.billing.models import HAS_BCRYPT

        if HAS_BCRYPT:
            assert user.needs_password_rehash() is False

    def test_user_upgrade_password_hash_when_not_needed(self):
        """Upgrade should return False when hash is already current."""
        user = User(email="test@example.com")
        user.set_password("password")
        from aragora.billing.models import HAS_BCRYPT

        if HAS_BCRYPT:
            assert user.upgrade_password_hash("password") is False

    def test_user_api_key_expiration(self):
        """API key should expire after set duration."""
        user = User(email="test@example.com")
        api_key = user.generate_api_key(expires_days=0)  # Expire immediately

        # Wait a tiny moment to ensure expiration
        import time

        time.sleep(0.01)
        assert user.is_api_key_expired() is True
        assert user.verify_api_key(api_key) is False

    def test_user_api_key_not_expired(self):
        """API key should be valid before expiration."""
        user = User(email="test@example.com")
        api_key = user.generate_api_key(expires_days=365)

        assert user.is_api_key_expired() is False
        assert user.verify_api_key(api_key) is True

    def test_user_verify_api_key_wrong_key(self):
        """Wrong API key should fail verification."""
        user = User(email="test@example.com")
        user.generate_api_key()

        assert user.verify_api_key("ara_wrong_key_here") is False

    def test_user_verify_api_key_no_key_set(self):
        """Verification should fail when no key is set."""
        user = User(email="test@example.com")
        assert user.verify_api_key("ara_some_key") is False


class TestSubscription:
    """Tests for Subscription model."""

    def test_subscription_creation(self):
        """Subscription should be created with defaults."""
        from aragora.billing.models import Subscription

        sub = Subscription(org_id="org-123")

        assert sub.id is not None
        assert sub.org_id == "org-123"
        assert sub.tier == SubscriptionTier.FREE
        assert sub.status == "active"

    def test_subscription_is_active(self):
        """Active subscription should return True for is_active."""
        from aragora.billing.models import Subscription

        sub = Subscription(status="active")
        assert sub.is_active is True

        sub.status = "trialing"
        assert sub.is_active is True

        sub.status = "canceled"
        assert sub.is_active is False

        sub.status = "past_due"
        assert sub.is_active is False

    def test_subscription_is_trialing(self):
        """Trialing subscription should be detected correctly."""
        from aragora.billing.models import Subscription

        sub = Subscription(
            status="trialing",
            trial_start=datetime.utcnow(),
            trial_end=datetime.utcnow() + timedelta(days=14),
        )
        assert sub.is_trialing is True

    def test_subscription_is_trialing_expired(self):
        """Expired trial should not be considered trialing."""
        from aragora.billing.models import Subscription

        sub = Subscription(
            status="trialing",
            trial_start=datetime.utcnow() - timedelta(days=15),
            trial_end=datetime.utcnow() - timedelta(days=1),
        )
        assert sub.is_trialing is False

    def test_subscription_is_trialing_wrong_status(self):
        """Non-trialing status should not be considered trialing."""
        from aragora.billing.models import Subscription

        sub = Subscription(
            status="active",
            trial_end=datetime.utcnow() + timedelta(days=14),
        )
        assert sub.is_trialing is False

    def test_subscription_days_until_renewal(self):
        """Days until renewal should be calculated correctly."""
        from aragora.billing.models import Subscription

        sub = Subscription(
            current_period_end=datetime.utcnow() + timedelta(days=15),
        )
        # Allow for small timing differences (14 or 15 days)
        assert sub.days_until_renewal in (14, 15)

    def test_subscription_days_until_renewal_past(self):
        """Past renewal date should return 0."""
        from aragora.billing.models import Subscription

        sub = Subscription(
            current_period_end=datetime.utcnow() - timedelta(days=5),
        )
        assert sub.days_until_renewal == 0

    def test_subscription_to_dict(self):
        """Subscription should serialize to dict."""
        from aragora.billing.models import Subscription

        sub = Subscription(
            org_id="org-123",
            tier=SubscriptionTier.PROFESSIONAL,
            status="active",
            stripe_subscription_id="sub_stripe123",
        )
        data = sub.to_dict()

        assert data["org_id"] == "org-123"
        assert data["tier"] == "professional"
        assert data["status"] == "active"
        assert data["stripe_subscription_id"] == "sub_stripe123"
        assert "is_active" in data
        assert "is_trialing" in data
        assert "days_until_renewal" in data

    def test_subscription_from_dict(self):
        """Subscription should deserialize from dict."""
        from aragora.billing.models import Subscription

        data = {
            "id": "sub-123",
            "org_id": "org-456",
            "tier": "enterprise",
            "status": "trialing",
            "cancel_at_period_end": True,
            "created_at": "2024-01-01T00:00:00",
        }
        sub = Subscription.from_dict(data)

        assert sub.id == "sub-123"
        assert sub.org_id == "org-456"
        assert sub.tier == SubscriptionTier.ENTERPRISE
        assert sub.status == "trialing"
        assert sub.cancel_at_period_end is True


class TestOrganizationInvitation:
    """Tests for OrganizationInvitation model."""

    def test_invitation_creation(self):
        """Invitation should be created with defaults."""
        from aragora.billing.models import OrganizationInvitation

        inv = OrganizationInvitation(org_id="org-123", email="user@example.com")

        assert inv.id is not None
        assert inv.token is not None
        assert len(inv.token) > 20
        assert inv.status == "pending"
        assert inv.role == "member"

    def test_invitation_is_pending(self):
        """Fresh invitation should be pending."""
        from aragora.billing.models import OrganizationInvitation

        inv = OrganizationInvitation(org_id="org-123", email="user@example.com")
        assert inv.is_pending is True
        assert inv.is_expired is False

    def test_invitation_is_expired(self):
        """Expired invitation should be detected."""
        from aragora.billing.models import OrganizationInvitation

        inv = OrganizationInvitation(
            org_id="org-123",
            email="user@example.com",
            expires_at=datetime.utcnow() - timedelta(days=1),
        )
        assert inv.is_expired is True
        assert inv.is_pending is False

    def test_invitation_accept(self):
        """Accepting invitation should update status."""
        from aragora.billing.models import OrganizationInvitation

        inv = OrganizationInvitation(org_id="org-123", email="user@example.com")

        result = inv.accept()

        assert result is True
        assert inv.status == "accepted"
        assert inv.accepted_at is not None

    def test_invitation_accept_already_accepted(self):
        """Cannot accept already accepted invitation."""
        from aragora.billing.models import OrganizationInvitation

        inv = OrganizationInvitation(org_id="org-123", email="user@example.com", status="accepted")

        result = inv.accept()

        assert result is False

    def test_invitation_accept_expired(self):
        """Cannot accept expired invitation."""
        from aragora.billing.models import OrganizationInvitation

        inv = OrganizationInvitation(
            org_id="org-123",
            email="user@example.com",
            expires_at=datetime.utcnow() - timedelta(days=1),
        )

        result = inv.accept()

        assert result is False
        assert inv.status == "pending"  # Status unchanged

    def test_invitation_revoke(self):
        """Revoking invitation should update status."""
        from aragora.billing.models import OrganizationInvitation

        inv = OrganizationInvitation(org_id="org-123", email="user@example.com")

        result = inv.revoke()

        assert result is True
        assert inv.status == "revoked"

    def test_invitation_revoke_already_accepted(self):
        """Cannot revoke already accepted invitation."""
        from aragora.billing.models import OrganizationInvitation

        inv = OrganizationInvitation(org_id="org-123", email="user@example.com", status="accepted")

        result = inv.revoke()

        assert result is False
        assert inv.status == "accepted"

    def test_invitation_to_dict_without_token(self):
        """to_dict should exclude token by default."""
        from aragora.billing.models import OrganizationInvitation

        inv = OrganizationInvitation(org_id="org-123", email="user@example.com", role="admin")
        data = inv.to_dict()

        assert "token" not in data
        assert data["org_id"] == "org-123"
        assert data["email"] == "user@example.com"
        assert data["role"] == "admin"
        assert "is_pending" in data
        assert "is_expired" in data

    def test_invitation_to_dict_with_token(self):
        """to_dict should include token when requested."""
        from aragora.billing.models import OrganizationInvitation

        inv = OrganizationInvitation(org_id="org-123", email="user@example.com")
        data = inv.to_dict(include_token=True)

        assert "token" in data
        assert data["token"] == inv.token

    def test_invitation_from_dict(self):
        """Invitation should deserialize from dict."""
        from aragora.billing.models import OrganizationInvitation

        data = {
            "id": "inv-123",
            "org_id": "org-456",
            "email": "USER@Example.com",  # Test lowercase normalization
            "role": "admin",
            "token": "custom_token_here",
            "status": "pending",
            "invited_by": "user-789",
            "created_at": "2024-01-01T00:00:00",
        }
        inv = OrganizationInvitation.from_dict(data)

        assert inv.id == "inv-123"
        assert inv.org_id == "org-456"
        assert inv.email == "user@example.com"  # Lowercased
        assert inv.role == "admin"
        assert inv.token == "custom_token_here"
        assert inv.invited_by == "user-789"


class TestGenerateSlug:
    """Tests for slug generation."""

    def test_generate_slug_basic(self):
        """Basic name should be slugified."""
        from aragora.billing.models import generate_slug

        assert generate_slug("My Company") == "my-company"

    def test_generate_slug_special_chars(self):
        """Special characters should be removed."""
        from aragora.billing.models import generate_slug

        assert generate_slug("Acme Corp!@#$%") == "acme-corp"

    def test_generate_slug_multiple_spaces(self):
        """Multiple spaces should become single hyphen."""
        from aragora.billing.models import generate_slug

        assert generate_slug("My   Big    Company") == "my-big-company"

    def test_generate_slug_underscores(self):
        """Underscores are removed (not converted to hyphens)."""
        from aragora.billing.models import generate_slug

        # Note: underscores are stripped by the first regex before conversion
        assert generate_slug("my_company_name") == "mycompanyname"

    def test_generate_slug_trailing_special(self):
        """Leading/trailing special chars should be stripped."""
        from aragora.billing.models import generate_slug

        assert generate_slug("---Company---") == "company"

    def test_generate_slug_empty(self):
        """Empty or all-special string should return 'org'."""
        from aragora.billing.models import generate_slug

        assert generate_slug("") == "org"
        assert generate_slug("!@#$%^") == "org"

    def test_generate_slug_max_length(self):
        """Slug should be truncated to 50 chars."""
        from aragora.billing.models import generate_slug

        long_name = "A" * 100
        slug = generate_slug(long_name)
        assert len(slug) <= 50
