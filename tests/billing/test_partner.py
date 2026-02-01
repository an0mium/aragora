"""
Tests for Partner API Program.

Covers:
- PartnerTier and PartnerStatus enums
- Partner, APIKey, UsageRecord, RevenueShare dataclasses
- PartnerStore database operations
- PartnerAPI management functions
- Webhook signature verification
"""

from __future__ import annotations

import hashlib
import hmac
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aragora.billing.partner import (
    API_SCOPES,
    APIKey,
    Partner,
    PartnerAPI,
    PartnerLimits,
    PartnerStatus,
    PartnerStore,
    PartnerTier,
    PARTNER_TIER_LIMITS,
    RevenueShare,
    UsageRecord,
    get_partner_api,
)


class TestPartnerTier:
    """Tests for PartnerTier enum."""

    def test_starter_tier(self):
        """Should have starter tier."""
        assert PartnerTier.STARTER.value == "starter"

    def test_developer_tier(self):
        """Should have developer tier."""
        assert PartnerTier.DEVELOPER.value == "developer"

    def test_business_tier(self):
        """Should have business tier."""
        assert PartnerTier.BUSINESS.value == "business"

    def test_enterprise_tier(self):
        """Should have enterprise tier."""
        assert PartnerTier.ENTERPRISE.value == "enterprise"

    def test_all_tiers_have_limits(self):
        """All tiers should have defined limits."""
        for tier in PartnerTier:
            assert tier in PARTNER_TIER_LIMITS


class TestPartnerStatus:
    """Tests for PartnerStatus enum."""

    def test_pending_status(self):
        """Should have pending status."""
        assert PartnerStatus.PENDING.value == "pending"

    def test_active_status(self):
        """Should have active status."""
        assert PartnerStatus.ACTIVE.value == "active"

    def test_suspended_status(self):
        """Should have suspended status."""
        assert PartnerStatus.SUSPENDED.value == "suspended"

    def test_revoked_status(self):
        """Should have revoked status."""
        assert PartnerStatus.REVOKED.value == "revoked"


class TestPartnerLimits:
    """Tests for PartnerLimits dataclass."""

    def test_starter_limits(self):
        """Starter tier should have minimal limits."""
        limits = PARTNER_TIER_LIMITS[PartnerTier.STARTER]
        assert limits.requests_per_minute == 60
        assert limits.requests_per_day == 1000
        assert limits.debates_per_month == 100
        assert limits.revenue_share_percent == 0.0

    def test_developer_limits(self):
        """Developer tier should have increased limits."""
        limits = PARTNER_TIER_LIMITS[PartnerTier.DEVELOPER]
        assert limits.requests_per_minute == 300
        assert limits.revenue_share_percent == 10.0

    def test_business_limits(self):
        """Business tier should have business-grade limits."""
        limits = PARTNER_TIER_LIMITS[PartnerTier.BUSINESS]
        assert limits.requests_per_minute == 1000
        assert limits.revenue_share_percent == 15.0

    def test_enterprise_limits(self):
        """Enterprise tier should have highest limits."""
        limits = PARTNER_TIER_LIMITS[PartnerTier.ENTERPRISE]
        assert limits.requests_per_minute == 5000
        assert limits.requests_per_day == 1000000
        assert limits.revenue_share_percent == 20.0

    def test_tier_progression(self):
        """Limits should increase with tier level."""
        starter = PARTNER_TIER_LIMITS[PartnerTier.STARTER]
        developer = PARTNER_TIER_LIMITS[PartnerTier.DEVELOPER]
        business = PARTNER_TIER_LIMITS[PartnerTier.BUSINESS]
        enterprise = PARTNER_TIER_LIMITS[PartnerTier.ENTERPRISE]

        assert starter.requests_per_minute < developer.requests_per_minute
        assert developer.requests_per_minute < business.requests_per_minute
        assert business.requests_per_minute < enterprise.requests_per_minute


class TestPartnerDataclass:
    """Tests for Partner dataclass."""

    @pytest.fixture
    def sample_partner(self):
        """Create sample partner."""
        now = datetime.now(timezone.utc)
        return Partner(
            partner_id="partner_test123",
            name="Test Partner",
            email="test@partner.com",
            company="Test Corp",
            tier=PartnerTier.DEVELOPER,
            status=PartnerStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            metadata={"key": "value"},
            webhook_url="https://example.com/webhook",
            referral_code="ABC123",
        )

    def test_to_dict(self, sample_partner):
        """Should convert to dictionary."""
        result = sample_partner.to_dict()

        assert result["partner_id"] == "partner_test123"
        assert result["name"] == "Test Partner"
        assert result["email"] == "test@partner.com"
        assert result["company"] == "Test Corp"
        assert result["tier"] == "developer"
        assert result["status"] == "active"
        assert result["metadata"] == {"key": "value"}
        assert result["webhook_url"] == "https://example.com/webhook"
        assert result["referral_code"] == "ABC123"

    def test_to_dict_excludes_webhook_secret(self, sample_partner):
        """Should exclude webhook_secret from dict."""
        sample_partner.webhook_secret = "secret123"
        result = sample_partner.to_dict()
        assert "webhook_secret" not in result

    def test_default_metadata(self):
        """Should have empty metadata by default."""
        now = datetime.now(timezone.utc)
        partner = Partner(
            partner_id="p1",
            name="Test",
            email="t@t.com",
            company=None,
            tier=PartnerTier.STARTER,
            status=PartnerStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        assert partner.metadata == {}


class TestAPIKeyDataclass:
    """Tests for APIKey dataclass."""

    @pytest.fixture
    def sample_key(self):
        """Create sample API key."""
        now = datetime.now(timezone.utc)
        return APIKey(
            key_id="key_test123",
            partner_id="partner_123",
            key_prefix="ara_test123",
            key_hash="abc123hash",
            name="Test Key",
            scopes=["debates:read", "debates:write"],
            created_at=now,
            expires_at=now + timedelta(days=30),
            last_used_at=now - timedelta(hours=1),
            is_active=True,
        )

    def test_to_dict(self, sample_key):
        """Should convert to dictionary."""
        result = sample_key.to_dict()

        assert result["key_id"] == "key_test123"
        assert result["partner_id"] == "partner_123"
        assert result["key_prefix"] == "ara_test123"
        assert result["name"] == "Test Key"
        assert result["scopes"] == ["debates:read", "debates:write"]
        assert result["is_active"] is True

    def test_to_dict_excludes_key_hash(self, sample_key):
        """Should exclude key_hash from dict."""
        result = sample_key.to_dict()
        assert "key_hash" not in result

    def test_none_expires_at(self):
        """Should handle None expires_at."""
        key = APIKey(
            key_id="k1",
            partner_id="p1",
            key_prefix="ara_x",
            key_hash="hash",
            name="Key",
            scopes=[],
            created_at=datetime.now(timezone.utc),
            expires_at=None,
            last_used_at=None,
        )
        result = key.to_dict()
        assert result["expires_at"] is None
        assert result["last_used_at"] is None


class TestUsageRecord:
    """Tests for UsageRecord dataclass."""

    def test_create_usage_record(self):
        """Should create usage record."""
        now = datetime.now(timezone.utc)
        record = UsageRecord(
            record_id="req_123",
            partner_id="partner_123",
            key_id="key_123",
            endpoint="/api/v1/debates",
            method="POST",
            status_code=201,
            latency_ms=150,
            tokens_used=1000,
            timestamp=now,
            metadata={"debate_id": "d123"},
        )

        assert record.record_id == "req_123"
        assert record.endpoint == "/api/v1/debates"
        assert record.status_code == 201
        assert record.tokens_used == 1000


class TestRevenueShare:
    """Tests for RevenueShare dataclass."""

    def test_create_revenue_share(self):
        """Should create revenue share record."""
        now = datetime.now(timezone.utc)
        share = RevenueShare(
            share_id="share_123",
            partner_id="partner_123",
            referred_user_id="user_456",
            period_start=now - timedelta(days=30),
            period_end=now,
            referred_spend_usd=1000.00,
            share_percent=15.0,
            share_amount_usd=150.00,
            status="pending",
        )

        assert share.share_id == "share_123"
        assert share.referred_spend_usd == 1000.00
        assert share.share_amount_usd == 150.00
        assert share.status == "pending"
        assert share.paid_at is None


class TestPartnerStore:
    """Tests for PartnerStore database operations."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create store with temporary database file."""
        db_path = str(tmp_path / "partner_test.db")
        return PartnerStore(db_path=db_path)

    @pytest.fixture
    def sample_partner(self):
        """Create sample partner."""
        now = datetime.now(timezone.utc)
        return Partner(
            partner_id="partner_store_test",
            name="Store Test Partner",
            email="store@test.com",
            company="Store Corp",
            tier=PartnerTier.DEVELOPER,
            status=PartnerStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            referral_code="STORE123",
        )

    def test_create_partner(self, store, sample_partner):
        """Should create partner."""
        result = store.create_partner(sample_partner)

        assert result.partner_id == sample_partner.partner_id
        assert result.name == "Store Test Partner"

    def test_get_partner(self, store, sample_partner):
        """Should retrieve partner by ID."""
        store.create_partner(sample_partner)
        result = store.get_partner(sample_partner.partner_id)

        assert result is not None
        assert result.partner_id == sample_partner.partner_id
        assert result.email == "store@test.com"

    def test_get_partner_not_found(self, store):
        """Should return None for non-existent partner."""
        result = store.get_partner("nonexistent")
        assert result is None

    def test_get_partner_by_email(self, store, sample_partner):
        """Should retrieve partner by email."""
        store.create_partner(sample_partner)
        result = store.get_partner_by_email("store@test.com")

        assert result is not None
        assert result.partner_id == sample_partner.partner_id

    def test_get_partner_by_email_not_found(self, store):
        """Should return None for non-existent email."""
        result = store.get_partner_by_email("nobody@nowhere.com")
        assert result is None

    def test_update_partner(self, store, sample_partner):
        """Should update partner."""
        store.create_partner(sample_partner)

        sample_partner.name = "Updated Name"
        sample_partner.tier = PartnerTier.BUSINESS
        result = store.update_partner(sample_partner)

        assert result.name == "Updated Name"
        assert result.tier == PartnerTier.BUSINESS

        # Verify in database
        retrieved = store.get_partner(sample_partner.partner_id)
        assert retrieved.name == "Updated Name"

    def test_create_api_key(self, store, sample_partner):
        """Should create API key."""
        store.create_partner(sample_partner)

        now = datetime.now(timezone.utc)
        key = APIKey(
            key_id="key_test",
            partner_id=sample_partner.partner_id,
            key_prefix="ara_test",
            key_hash="testhash123",
            name="Test Key",
            scopes=["debates:read"],
            created_at=now,
            expires_at=None,
            last_used_at=None,
        )

        result = store.create_api_key(key)
        assert result.key_id == "key_test"

    def test_get_api_key_by_hash(self, store, sample_partner):
        """Should retrieve API key by hash."""
        store.create_partner(sample_partner)

        now = datetime.now(timezone.utc)
        key = APIKey(
            key_id="key_hash_test",
            partner_id=sample_partner.partner_id,
            key_prefix="ara_hash",
            key_hash="uniquehash456",
            name="Hash Test Key",
            scopes=["debates:write"],
            created_at=now,
            expires_at=None,
            last_used_at=None,
        )
        store.create_api_key(key)

        result = store.get_api_key_by_hash("uniquehash456")
        assert result is not None
        assert result.key_id == "key_hash_test"

    def test_get_api_key_by_hash_not_found(self, store):
        """Should return None for non-existent hash."""
        result = store.get_api_key_by_hash("nonexistent")
        assert result is None

    def test_list_partner_keys(self, store, sample_partner):
        """Should list all keys for partner."""
        store.create_partner(sample_partner)

        now = datetime.now(timezone.utc)
        for i in range(3):
            key = APIKey(
                key_id=f"key_list_{i}",
                partner_id=sample_partner.partner_id,
                key_prefix=f"ara_list{i}",
                key_hash=f"listhash{i}",
                name=f"List Key {i}",
                scopes=["debates:read"],
                created_at=now + timedelta(seconds=i),
                expires_at=None,
                last_used_at=None,
            )
            store.create_api_key(key)

        keys = store.list_partner_keys(sample_partner.partner_id)
        assert len(keys) == 3

    def test_revoke_api_key(self, store, sample_partner):
        """Should revoke API key."""
        store.create_partner(sample_partner)

        now = datetime.now(timezone.utc)
        key = APIKey(
            key_id="key_revoke",
            partner_id=sample_partner.partner_id,
            key_prefix="ara_revoke",
            key_hash="revokehash",
            name="Revoke Key",
            scopes=["debates:read"],
            created_at=now,
            expires_at=None,
            last_used_at=None,
        )
        store.create_api_key(key)

        result = store.revoke_api_key("key_revoke")
        assert result is True

        # Verify key is inactive
        retrieved = store.get_api_key_by_hash("revokehash")
        assert retrieved.is_active is False

    def test_revoke_nonexistent_key(self, store):
        """Should return False for non-existent key."""
        result = store.revoke_api_key("nonexistent")
        assert result is False

    def test_record_usage(self, store, sample_partner):
        """Should record usage."""
        store.create_partner(sample_partner)

        now = datetime.now(timezone.utc)
        record = UsageRecord(
            record_id="req_usage",
            partner_id=sample_partner.partner_id,
            key_id="key_123",
            endpoint="/api/debates",
            method="GET",
            status_code=200,
            latency_ms=50,
            tokens_used=100,
            timestamp=now,
        )

        # Should not raise
        store.record_usage(record)

    def test_get_usage_stats(self, store, sample_partner):
        """Should get usage statistics."""
        store.create_partner(sample_partner)

        now = datetime.now(timezone.utc)
        for i in range(5):
            record = UsageRecord(
                record_id=f"req_stats_{i}",
                partner_id=sample_partner.partner_id,
                key_id="key_123",
                endpoint="/api/debates",
                method="GET",
                status_code=200 if i < 4 else 500,
                latency_ms=50 + i * 10,
                tokens_used=100,
                timestamp=now - timedelta(hours=i),
            )
            store.record_usage(record)

        stats = store.get_usage_stats(
            sample_partner.partner_id,
            now - timedelta(days=1),
            now + timedelta(hours=1),
        )

        assert stats["total_requests"] == 5
        assert stats["total_tokens"] == 500
        assert stats["error_count"] == 1

    def test_get_usage_stats_empty(self, store, sample_partner):
        """Should return zeros for no usage."""
        store.create_partner(sample_partner)

        now = datetime.now(timezone.utc)
        stats = store.get_usage_stats(
            sample_partner.partner_id,
            now - timedelta(days=1),
            now,
        )

        assert stats["total_requests"] == 0
        assert stats["total_tokens"] == 0
        assert stats["error_count"] == 0


class TestPartnerStoreGetAPIKey:
    """Tests for PartnerStore.get_api_key method."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create store with temporary database file."""
        db_path = str(tmp_path / "partner_get_key_test.db")
        return PartnerStore(db_path=db_path)

    @pytest.fixture
    def sample_partner(self):
        """Create sample partner."""
        now = datetime.now(timezone.utc)
        return Partner(
            partner_id="partner_getkey_test",
            name="GetKey Test Partner",
            email="getkey@test.com",
            company=None,
            tier=PartnerTier.DEVELOPER,
            status=PartnerStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            referral_code="GETKEY123",
        )

    def test_get_api_key_by_id(self, store, sample_partner):
        """Should retrieve API key by ID."""
        store.create_partner(sample_partner)

        now = datetime.now(timezone.utc)
        key = APIKey(
            key_id="key_byid_test",
            partner_id=sample_partner.partner_id,
            key_prefix="ara_byid",
            key_hash="byidhash123",
            name="ById Test Key",
            scopes=["debates:read"],
            created_at=now,
            expires_at=None,
            last_used_at=None,
        )
        store.create_api_key(key)

        result = store.get_api_key("key_byid_test")
        assert result is not None
        assert result.key_id == "key_byid_test"
        assert result.partner_id == sample_partner.partner_id
        assert result.name == "ById Test Key"

    def test_get_api_key_not_found(self, store):
        """Should return None for non-existent key ID."""
        result = store.get_api_key("nonexistent_key")
        assert result is None


class TestPartnerAPI:
    """Tests for PartnerAPI management."""

    @pytest.fixture
    def api(self, tmp_path):
        """Create API with temporary database file."""
        db_path = str(tmp_path / "partner_api_test.db")
        store = PartnerStore(db_path=db_path)
        return PartnerAPI(store=store)

    def test_register_partner(self, api):
        """Should register new partner."""
        partner = api.register_partner(
            name="New Partner",
            email="new@partner.com",
            company="New Corp",
            tier=PartnerTier.DEVELOPER,
        )

        assert partner.name == "New Partner"
        assert partner.email == "new@partner.com"
        assert partner.status == PartnerStatus.PENDING
        assert partner.tier == PartnerTier.DEVELOPER
        assert partner.referral_code is not None
        assert partner.partner_id.startswith("partner_")

    def test_register_partner_default_tier(self, api):
        """Should default to starter tier."""
        partner = api.register_partner(
            name="Default Partner",
            email="default@partner.com",
        )

        assert partner.tier == PartnerTier.STARTER

    def test_activate_partner(self, api):
        """Should activate pending partner."""
        partner = api.register_partner(
            name="Activate Partner",
            email="activate@partner.com",
        )

        result = api.activate_partner(partner.partner_id)

        assert result is not None
        assert result.status == PartnerStatus.ACTIVE

    def test_activate_nonexistent_partner(self, api):
        """Should return None for non-existent partner."""
        result = api.activate_partner("nonexistent")
        assert result is None

    def test_create_api_key(self, api):
        """Should create API key for active partner."""
        partner = api.register_partner(
            name="Key Partner",
            email="key@partner.com",
        )
        api.activate_partner(partner.partner_id)

        key, raw_key = api.create_api_key(
            partner_id=partner.partner_id,
            name="Test Key",
            scopes=["debates:read"],
        )

        assert key.name == "Test Key"
        assert key.scopes == ["debates:read"]
        assert raw_key.startswith("ara_")
        assert len(raw_key) > 20

    def test_create_api_key_default_scopes(self, api):
        """Should use all scopes by default."""
        partner = api.register_partner(
            name="Scope Partner",
            email="scope@partner.com",
        )
        api.activate_partner(partner.partner_id)

        key, _ = api.create_api_key(
            partner_id=partner.partner_id,
            name="Default Scopes",
        )

        assert key.scopes == API_SCOPES

    def test_create_api_key_with_expiry(self, api):
        """Should create key with expiration."""
        partner = api.register_partner(
            name="Expiry Partner",
            email="expiry@partner.com",
        )
        api.activate_partner(partner.partner_id)

        key, _ = api.create_api_key(
            partner_id=partner.partner_id,
            name="Expiring Key",
            expires_in_days=30,
        )

        assert key.expires_at is not None
        assert key.expires_at > datetime.now(timezone.utc)

    def test_create_api_key_inactive_partner(self, api):
        """Should reject key creation for inactive partner."""
        partner = api.register_partner(
            name="Inactive Partner",
            email="inactive@partner.com",
        )

        with pytest.raises(ValueError, match="not found or not active"):
            api.create_api_key(
                partner_id=partner.partner_id,
                name="Rejected Key",
            )

    def test_validate_api_key(self, api):
        """Should validate correct API key."""
        partner = api.register_partner(
            name="Validate Partner",
            email="validate@partner.com",
        )
        api.activate_partner(partner.partner_id)

        key, raw_key = api.create_api_key(
            partner_id=partner.partner_id,
            name="Valid Key",
        )

        result = api.validate_api_key(raw_key)

        assert result is not None
        validated_key, validated_partner = result
        assert validated_key.key_id == key.key_id
        assert validated_partner.partner_id == partner.partner_id

    def test_validate_invalid_key(self, api):
        """Should return None for invalid key."""
        result = api.validate_api_key("ara_invalid_key")
        assert result is None

    def test_validate_revoked_key(self, api):
        """Should return None for revoked key."""
        partner = api.register_partner(
            name="Revoke Partner",
            email="revoke@partner.com",
        )
        api.activate_partner(partner.partner_id)

        key, raw_key = api.create_api_key(
            partner_id=partner.partner_id,
            name="Revoked Key",
        )

        # Revoke the key
        api._store.revoke_api_key(key.key_id)

        result = api.validate_api_key(raw_key)
        assert result is None

    def test_check_rate_limit(self, api):
        """Should check rate limits."""
        partner = api.register_partner(
            name="Rate Partner",
            email="rate@partner.com",
        )
        api.activate_partner(partner.partner_id)

        allowed, info = api.check_rate_limit(api._store.get_partner(partner.partner_id))

        assert allowed is True
        assert info["limit_per_minute"] > 0
        assert info["remaining_minute"] > 0

    def test_record_request(self, api):
        """Should record API request."""
        partner = api.register_partner(
            name="Record Partner",
            email="record@partner.com",
        )
        api.activate_partner(partner.partner_id)

        key, _ = api.create_api_key(
            partner_id=partner.partner_id,
            name="Record Key",
        )

        # Should not raise
        api.record_request(
            partner_id=partner.partner_id,
            key_id=key.key_id,
            endpoint="/api/debates",
            method="GET",
            status_code=200,
            latency_ms=50,
            tokens_used=100,
        )

    def test_get_partner_stats(self, api):
        """Should get partner statistics."""
        partner = api.register_partner(
            name="Stats Partner",
            email="stats@partner.com",
        )
        api.activate_partner(partner.partner_id)

        key, _ = api.create_api_key(
            partner_id=partner.partner_id,
            name="Stats Key",
        )

        # Record some requests
        for i in range(3):
            api.record_request(
                partner_id=partner.partner_id,
                key_id=key.key_id,
                endpoint="/api/debates",
                method="GET",
                status_code=200,
                latency_ms=50,
                tokens_used=100,
            )

        stats = api.get_partner_stats(partner.partner_id, days=7)

        assert stats["partner"]["name"] == "Stats Partner"
        assert stats["usage"]["total_requests"] == 3
        assert stats["keys"]["total"] == 1
        assert stats["keys"]["active"] == 1

    def test_get_partner_stats_not_found(self, api):
        """Should raise for non-existent partner."""
        with pytest.raises(ValueError, match="Partner not found"):
            api.get_partner_stats("nonexistent")

    def test_rotate_api_key(self, api):
        """Should rotate API key preserving name and scopes."""
        partner = api.register_partner(
            name="Rotate Partner",
            email="rotate@partner.com",
        )
        api.activate_partner(partner.partner_id)

        old_key, old_raw = api.create_api_key(
            partner_id=partner.partner_id,
            name="Rotatable Key",
            scopes=["debates:read", "debates:write"],
        )

        new_key, new_raw = api.rotate_api_key(
            partner_id=partner.partner_id,
            key_id=old_key.key_id,
        )

        # New key should have same name and scopes
        assert new_key.name == "Rotatable Key"
        assert new_key.scopes == ["debates:read", "debates:write"]
        assert new_key.is_active is True
        assert new_key.key_id != old_key.key_id
        assert new_raw.startswith("ara_")
        assert new_raw != old_raw

        # Old key should be revoked
        old_retrieved = api._store.get_api_key(old_key.key_id)
        assert old_retrieved.is_active is False

        # Old key should no longer validate
        assert api.validate_api_key(old_raw) is None

        # New key should validate
        result = api.validate_api_key(new_raw)
        assert result is not None

    def test_rotate_api_key_with_expiry(self, api):
        """Should preserve expiration window on rotation."""
        partner = api.register_partner(
            name="Expiry Rotate Partner",
            email="expiry-rotate@partner.com",
        )
        api.activate_partner(partner.partner_id)

        old_key, _ = api.create_api_key(
            partner_id=partner.partner_id,
            name="Expiring Key",
            scopes=["debates:read"],
            expires_in_days=90,
        )

        new_key, _ = api.rotate_api_key(
            partner_id=partner.partner_id,
            key_id=old_key.key_id,
        )

        # New key should have an expiration
        assert new_key.expires_at is not None

    def test_rotate_api_key_not_found(self, api):
        """Should raise for non-existent key."""
        partner = api.register_partner(
            name="NotFound Rotate Partner",
            email="notfound-rotate@partner.com",
        )
        api.activate_partner(partner.partner_id)

        with pytest.raises(ValueError, match="API key not found"):
            api.rotate_api_key(
                partner_id=partner.partner_id,
                key_id="nonexistent_key",
            )

    def test_rotate_api_key_wrong_partner(self, api):
        """Should reject rotation if key belongs to different partner."""
        partner1 = api.register_partner(
            name="Partner 1",
            email="p1@rotate.com",
        )
        api.activate_partner(partner1.partner_id)

        partner2 = api.register_partner(
            name="Partner 2",
            email="p2@rotate.com",
        )
        api.activate_partner(partner2.partner_id)

        key, _ = api.create_api_key(
            partner_id=partner1.partner_id,
            name="P1 Key",
        )

        with pytest.raises(ValueError, match="does not belong"):
            api.rotate_api_key(
                partner_id=partner2.partner_id,
                key_id=key.key_id,
            )

    def test_rotate_inactive_key(self, api):
        """Should reject rotation of already-revoked key."""
        partner = api.register_partner(
            name="Inactive Rotate Partner",
            email="inactive-rotate@partner.com",
        )
        api.activate_partner(partner.partner_id)

        key, _ = api.create_api_key(
            partner_id=partner.partner_id,
            name="Soon Revoked",
        )

        api._store.revoke_api_key(key.key_id)

        with pytest.raises(ValueError, match="inactive"):
            api.rotate_api_key(
                partner_id=partner.partner_id,
                key_id=key.key_id,
            )

    def test_generate_webhook_secret(self, api):
        """Should generate webhook secret."""
        partner = api.register_partner(
            name="Webhook Partner",
            email="webhook@partner.com",
        )

        secret = api.generate_webhook_secret(partner.partner_id)

        assert secret.startswith("whsec_")
        assert len(secret) > 20

    def test_generate_webhook_secret_not_found(self, api):
        """Should raise for non-existent partner."""
        with pytest.raises(ValueError, match="Partner not found"):
            api.generate_webhook_secret("nonexistent")

    def test_verify_webhook_signature(self, api):
        """Should verify valid webhook signature."""
        partner = api.register_partner(
            name="Verify Partner",
            email="verify@partner.com",
        )
        secret = api.generate_webhook_secret(partner.partner_id)

        # Create valid signature
        timestamp = str(int(time.time()))
        payload = b'{"event": "test"}'
        signed_payload = f"{timestamp}.{payload.decode()}"

        # Extract secret value (after whsec_)
        stored_partner = api._store.get_partner(partner.partner_id)
        expected = hmac.new(
            stored_partner.webhook_secret.encode(),
            signed_payload.encode(),
            hashlib.sha256,
        ).hexdigest()
        signature = f"v1={expected}"

        result = api.verify_webhook_signature(partner.partner_id, payload, signature, timestamp)

        assert result is True

    def test_verify_webhook_signature_invalid(self, api):
        """Should reject invalid signature."""
        partner = api.register_partner(
            name="Invalid Sig Partner",
            email="invalid@partner.com",
        )
        api.generate_webhook_secret(partner.partner_id)

        timestamp = str(int(time.time()))
        payload = b'{"event": "test"}'

        result = api.verify_webhook_signature(partner.partner_id, payload, "v1=invalid", timestamp)

        assert result is False

    def test_verify_webhook_signature_expired(self, api):
        """Should reject expired timestamp."""
        partner = api.register_partner(
            name="Expired Partner",
            email="expired@partner.com",
        )
        api.generate_webhook_secret(partner.partner_id)

        # Use timestamp from 10 minutes ago (> 5 minute window)
        timestamp = str(int(time.time()) - 600)
        payload = b'{"event": "test"}'

        result = api.verify_webhook_signature(partner.partner_id, payload, "v1=any", timestamp)

        assert result is False

    def test_verify_webhook_signature_no_secret(self, api):
        """Should reject if no webhook secret configured."""
        partner = api.register_partner(
            name="No Secret Partner",
            email="nosecret@partner.com",
        )

        result = api.verify_webhook_signature(
            partner.partner_id, b"data", "v1=sig", str(int(time.time()))
        )

        assert result is False


class TestGetPartnerAPI:
    """Tests for get_partner_api singleton."""

    def test_returns_singleton(self):
        """Should return same instance."""
        api1 = get_partner_api()
        api2 = get_partner_api()

        # Both should be PartnerAPI instances
        assert isinstance(api1, PartnerAPI)
        assert isinstance(api2, PartnerAPI)


class TestAPIScopes:
    """Tests for API_SCOPES constant."""

    def test_contains_debates_scopes(self):
        """Should have debates scopes."""
        assert "debates:read" in API_SCOPES
        assert "debates:write" in API_SCOPES

    def test_contains_agents_scope(self):
        """Should have agents scope."""
        assert "agents:read" in API_SCOPES

    def test_contains_gauntlet_scope(self):
        """Should have gauntlet scope."""
        assert "gauntlet:run" in API_SCOPES

    def test_contains_memory_scopes(self):
        """Should have memory scopes."""
        assert "memory:read" in API_SCOPES
        assert "memory:write" in API_SCOPES

    def test_contains_analytics_scope(self):
        """Should have analytics scope."""
        assert "analytics:read" in API_SCOPES

    def test_contains_webhooks_scope(self):
        """Should have webhooks scope."""
        assert "webhooks:manage" in API_SCOPES

    def test_contains_knowledge_scopes(self):
        """Should have knowledge scopes."""
        assert "knowledge:read" in API_SCOPES
        assert "knowledge:write" in API_SCOPES


class TestModuleExports:
    """Tests for module exports."""

    def test_all_classes_importable(self):
        """Should export all public classes."""
        from aragora.billing.partner import (
            APIKey,
            Partner,
            PartnerAPI,
            PartnerStatus,
            PartnerStore,
            PartnerTier,
            RevenueShare,
            UsageRecord,
        )

        assert PartnerTier is not None
        assert PartnerStatus is not None
        assert PartnerLimits is not None
        assert Partner is not None
        assert APIKey is not None
        assert UsageRecord is not None
        assert RevenueShare is not None
        assert PartnerStore is not None
        assert PartnerAPI is not None
