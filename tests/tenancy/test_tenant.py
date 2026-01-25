"""Tests for Tenant model and configuration."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest

from aragora.tenancy.tenant import (
    Tenant,
    TenantConfig,
    TenantStatus,
    TenantSuspendedError,
    TenantTier,
)


class TestTenantTier:
    """Tests for TenantTier enum."""

    def test_all_tiers_exist(self):
        """Test all expected tiers exist."""
        assert TenantTier.FREE.value == "free"
        assert TenantTier.STARTER.value == "starter"
        assert TenantTier.PROFESSIONAL.value == "professional"
        assert TenantTier.ENTERPRISE.value == "enterprise"
        assert TenantTier.CUSTOM.value == "custom"

    def test_tier_count(self):
        """Test expected number of tiers."""
        assert len(TenantTier) == 5


class TestTenantStatus:
    """Tests for TenantStatus enum."""

    def test_all_statuses_exist(self):
        """Test all expected statuses exist."""
        assert TenantStatus.ACTIVE.value == "active"
        assert TenantStatus.SUSPENDED.value == "suspended"
        assert TenantStatus.PENDING.value == "pending"
        assert TenantStatus.TRIAL.value == "trial"
        assert TenantStatus.CANCELLED.value == "cancelled"

    def test_status_count(self):
        """Test expected number of statuses."""
        assert len(TenantStatus) == 5


class TestTenantConfig:
    """Tests for TenantConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TenantConfig()

        # Limits
        assert config.max_debates_per_day == 100
        assert config.max_agents_per_debate == 10
        assert config.max_rounds_per_debate == 20
        assert config.max_concurrent_debates == 5
        assert config.max_users == 10
        assert config.max_connectors == 5

        # Features
        assert config.enable_rlm is True
        assert config.enable_api_access is True
        assert config.enable_audit_log is True
        assert config.enable_extended_debates is False
        assert config.enable_custom_agents is False
        assert config.enable_webhooks is False
        assert config.enable_sso is False

        # Storage
        assert config.storage_quota == 10 * 1024 * 1024 * 1024
        assert config.knowledge_quota == 1 * 1024 * 1024 * 1024

        # Rate limits
        assert config.api_requests_per_minute == 60
        assert config.api_requests_per_day == 10000

        # Token limits
        assert config.tokens_per_month == 1_000_000
        assert config.tokens_per_debate == 50_000

    def test_for_tier_free(self):
        """Test FREE tier configuration."""
        config = TenantConfig.for_tier(TenantTier.FREE)

        assert config.max_debates_per_day == 10
        assert config.max_agents_per_debate == 5
        assert config.max_concurrent_debates == 1
        assert config.max_users == 3
        assert config.enable_extended_debates is False
        assert config.enable_custom_agents is False
        assert config.enable_webhooks is False
        assert config.enable_sso is False
        assert config.tokens_per_month == 100_000

    def test_for_tier_starter(self):
        """Test STARTER tier configuration."""
        config = TenantConfig.for_tier(TenantTier.STARTER)

        assert config.max_debates_per_day == 50
        assert config.max_agents_per_debate == 8
        assert config.max_concurrent_debates == 3
        assert config.max_users == 10
        assert config.enable_extended_debates is True
        assert config.enable_webhooks is True
        assert config.tokens_per_month == 500_000

    def test_for_tier_professional(self):
        """Test PROFESSIONAL tier configuration."""
        config = TenantConfig.for_tier(TenantTier.PROFESSIONAL)

        assert config.max_debates_per_day == 200
        assert config.max_agents_per_debate == 15
        assert config.max_concurrent_debates == 10
        assert config.max_users == 50
        assert config.enable_extended_debates is True
        assert config.enable_custom_agents is True
        assert config.enable_webhooks is True
        assert config.enable_sso is True
        assert config.tokens_per_month == 5_000_000

    def test_for_tier_enterprise(self):
        """Test ENTERPRISE tier configuration."""
        config = TenantConfig.for_tier(TenantTier.ENTERPRISE)

        assert config.max_debates_per_day == 10000
        assert config.max_agents_per_debate == 50
        assert config.max_concurrent_debates == 100
        assert config.max_users == 1000
        assert config.enable_extended_debates is True
        assert config.enable_custom_agents is True
        assert config.enable_webhooks is True
        assert config.enable_sso is True
        assert config.tokens_per_month == 50_000_000
        assert config.storage_quota == 1024 * 1024 * 1024 * 1024  # 1TB

    def test_for_tier_custom_returns_default(self):
        """Test CUSTOM tier returns default configuration."""
        config = TenantConfig.for_tier(TenantTier.CUSTOM)
        default = TenantConfig()

        assert config.max_debates_per_day == default.max_debates_per_day
        assert config.tokens_per_month == default.tokens_per_month


class TestTenant:
    """Tests for Tenant dataclass."""

    def test_create_minimal_tenant(self):
        """Test creating tenant with minimal fields."""
        tenant = Tenant(
            id="test-123",
            name="Test Org",
            slug="test-org",
        )

        assert tenant.id == "test-123"
        assert tenant.name == "Test Org"
        assert tenant.slug == "test-org"
        assert tenant.tier == TenantTier.FREE
        assert tenant.status == TenantStatus.ACTIVE
        assert tenant.owner_email == ""

    def test_create_full_tenant(self):
        """Test creating tenant with all fields."""
        config = TenantConfig.for_tier(TenantTier.PROFESSIONAL)
        created = datetime(2024, 1, 1)

        tenant = Tenant(
            id="full-tenant",
            name="Full Organization",
            slug="full-org",
            tier=TenantTier.PROFESSIONAL,
            status=TenantStatus.TRIAL,
            config=config,
            owner_email="owner@example.com",
            billing_email="billing@example.com",
            created_at=created,
            logo_url="https://example.com/logo.png",
            theme={"primary": "#0066cc"},
        )

        assert tenant.tier == TenantTier.PROFESSIONAL
        assert tenant.status == TenantStatus.TRIAL
        assert tenant.owner_email == "owner@example.com"
        assert tenant.billing_email == "billing@example.com"
        assert tenant.logo_url == "https://example.com/logo.png"
        assert tenant.theme["primary"] == "#0066cc"


class TestTenantFactory:
    """Tests for Tenant.create factory method."""

    def test_create_generates_slug(self):
        """Test that create generates a URL-safe slug."""
        tenant = Tenant.create(
            name="My Test Organization",
            owner_email="test@example.com",
        )

        assert "my-test-organization" in tenant.slug
        assert " " not in tenant.slug

    def test_create_generates_unique_id(self):
        """Test that create generates unique IDs."""
        tenant1 = Tenant.create(name="Same Name", owner_email="a@example.com")
        tenant2 = Tenant.create(name="Same Name", owner_email="b@example.com")

        assert tenant1.id != tenant2.id
        assert "same-name" in tenant1.slug
        assert "same-name" in tenant2.slug

    def test_create_applies_tier_config(self):
        """Test that create applies correct tier configuration."""
        tenant = Tenant.create(
            name="Pro Org",
            owner_email="pro@example.com",
            tier=TenantTier.PROFESSIONAL,
        )

        assert tenant.tier == TenantTier.PROFESSIONAL
        assert tenant.config.max_debates_per_day == 200
        assert tenant.config.enable_sso is True

    def test_create_sets_owner_email(self):
        """Test that create sets owner email."""
        tenant = Tenant.create(
            name="Email Test",
            owner_email="owner@company.com",
        )

        assert tenant.owner_email == "owner@company.com"


class TestSlugGeneration:
    """Tests for slug generation."""

    def test_slug_lowercase(self):
        """Test slug is lowercased."""
        slug = Tenant._generate_slug("UPPERCASE NAME")
        assert slug == "uppercase-name"

    def test_slug_special_chars_replaced(self):
        """Test special characters are replaced with hyphens."""
        slug = Tenant._generate_slug("Name with @special! chars")
        assert "@" not in slug
        assert "!" not in slug
        assert slug == "name-with-special-chars"

    def test_slug_truncated_to_50_chars(self):
        """Test slug is truncated to 50 characters."""
        long_name = "A" * 100
        slug = Tenant._generate_slug(long_name)
        assert len(slug) <= 50

    def test_slug_no_double_hyphens(self):
        """Test no double hyphens in slug."""
        slug = Tenant._generate_slug("Name  with   spaces")
        assert "--" not in slug


class TestApiKey:
    """Tests for API key generation and verification."""

    def test_generate_api_key(self):
        """Test API key generation."""
        tenant = Tenant(id="test", name="Test", slug="test")
        api_key = tenant.generate_api_key()

        assert api_key.startswith("ara_test_")
        assert len(api_key) > 50
        assert tenant.api_key_hash is not None

    def test_verify_api_key_success(self):
        """Test successful API key verification."""
        tenant = Tenant(id="test", name="Test", slug="test")
        api_key = tenant.generate_api_key()

        assert tenant.verify_api_key(api_key) is True

    def test_verify_api_key_failure(self):
        """Test failed API key verification."""
        tenant = Tenant(id="test", name="Test", slug="test")
        tenant.generate_api_key()

        assert tenant.verify_api_key("wrong_key") is False

    def test_verify_api_key_no_hash(self):
        """Test verification when no hash is set."""
        tenant = Tenant(id="test", name="Test", slug="test")

        assert tenant.verify_api_key("any_key") is False

    def test_api_key_regeneration(self):
        """Test that regenerating API key invalidates old key."""
        tenant = Tenant(id="test", name="Test", slug="test")
        old_key = tenant.generate_api_key()
        new_key = tenant.generate_api_key()

        assert tenant.verify_api_key(old_key) is False
        assert tenant.verify_api_key(new_key) is True


class TestTenantStatusMethods:
    """Tests for tenant status methods."""

    def test_is_active_with_active_status(self):
        """Test is_active returns True for ACTIVE status."""
        tenant = Tenant(id="test", name="Test", slug="test", status=TenantStatus.ACTIVE)
        assert tenant.is_active() is True

    def test_is_active_with_trial_status(self):
        """Test is_active returns True for TRIAL status."""
        tenant = Tenant(id="test", name="Test", slug="test", status=TenantStatus.TRIAL)
        assert tenant.is_active() is True

    def test_is_active_with_suspended_status(self):
        """Test is_active returns False for SUSPENDED status."""
        tenant = Tenant(id="test", name="Test", slug="test", status=TenantStatus.SUSPENDED)
        assert tenant.is_active() is False

    def test_is_active_with_pending_status(self):
        """Test is_active returns False for PENDING status."""
        tenant = Tenant(id="test", name="Test", slug="test", status=TenantStatus.PENDING)
        assert tenant.is_active() is False

    def test_is_active_with_cancelled_status(self):
        """Test is_active returns False for CANCELLED status."""
        tenant = Tenant(id="test", name="Test", slug="test", status=TenantStatus.CANCELLED)
        assert tenant.is_active() is False


class TestUsageLimits:
    """Tests for usage limit checking."""

    def test_can_create_debate_active_under_limit(self):
        """Test can_create_debate with active tenant under limit."""
        tenant = Tenant(id="test", name="Test", slug="test")
        tenant.current_month_debates = 5

        assert tenant.can_create_debate() is True

    def test_can_create_debate_at_limit(self):
        """Test can_create_debate at limit."""
        config = TenantConfig(max_debates_per_day=10)
        tenant = Tenant(id="test", name="Test", slug="test", config=config)
        tenant.current_month_debates = 10

        assert tenant.can_create_debate() is False

    def test_can_create_debate_suspended(self):
        """Test can_create_debate with suspended tenant."""
        tenant = Tenant(
            id="test",
            name="Test",
            slug="test",
            status=TenantStatus.SUSPENDED,
        )
        tenant.current_month_debates = 0

        assert tenant.can_create_debate() is False

    def test_can_use_tokens_under_limit(self):
        """Test can_use_tokens under monthly limit."""
        config = TenantConfig(tokens_per_month=1000)
        tenant = Tenant(id="test", name="Test", slug="test", config=config)
        tenant.current_month_tokens = 500

        assert tenant.can_use_tokens(400) is True

    def test_can_use_tokens_at_limit(self):
        """Test can_use_tokens at monthly limit."""
        config = TenantConfig(tokens_per_month=1000)
        tenant = Tenant(id="test", name="Test", slug="test", config=config)
        tenant.current_month_tokens = 500

        assert tenant.can_use_tokens(500) is True

    def test_can_use_tokens_over_limit(self):
        """Test can_use_tokens over monthly limit."""
        config = TenantConfig(tokens_per_month=1000)
        tenant = Tenant(id="test", name="Test", slug="test", config=config)
        tenant.current_month_tokens = 500

        assert tenant.can_use_tokens(501) is False


class TestTenantSerialization:
    """Tests for tenant serialization."""

    def test_to_dict(self):
        """Test to_dict includes expected fields."""
        tenant = Tenant(
            id="test-123",
            name="Test Org",
            slug="test-org",
            tier=TenantTier.PROFESSIONAL,
            status=TenantStatus.ACTIVE,
            owner_email="test@example.com",
        )
        tenant.current_month_tokens = 5000
        tenant.current_month_debates = 10
        tenant.storage_used = 1024

        result = tenant.to_dict()

        assert result["id"] == "test-123"
        assert result["name"] == "Test Org"
        assert result["slug"] == "test-org"
        assert result["tier"] == "professional"
        assert result["status"] == "active"
        assert result["owner_email"] == "test@example.com"
        assert result["current_month_tokens"] == 5000
        assert result["current_month_debates"] == 10
        assert result["storage_used"] == 1024
        assert "created_at" in result

    def test_to_dict_iso_format_dates(self):
        """Test that dates are in ISO format."""
        tenant = Tenant(id="test", name="Test", slug="test")
        result = tenant.to_dict()

        # Should be parseable as ISO format
        datetime.fromisoformat(result["created_at"])


class TestTenantSuspendedError:
    """Tests for TenantSuspendedError exception."""

    def test_error_message(self):
        """Test error has correct message."""
        error = TenantSuspendedError("tenant-123")
        assert "tenant-123" in str(error)
        assert "suspended" in str(error)

    def test_error_message_with_reason(self):
        """Test error message includes reason."""
        error = TenantSuspendedError("tenant-123", reason="billing issue")
        assert "tenant-123" in str(error)
        assert "billing issue" in str(error)

    def test_error_properties(self):
        """Test error has tenant_id and reason properties."""
        error = TenantSuspendedError("test-tenant", reason="test reason")
        assert error.tenant_id == "test-tenant"
        assert error.reason == "test reason"

    def test_error_inheritance(self):
        """Test error inherits from Exception."""
        assert issubclass(TenantSuspendedError, Exception)
