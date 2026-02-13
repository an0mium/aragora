"""Tests for TenantsAPI client resource."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.client.client import AragoraClient
from aragora.client.resources.tenants import (
    Tenant,
    TenantQuota,
    TenantSettings,
    TenantUsage,
    TenantsAPI,
)


@pytest.fixture
def mock_client() -> AragoraClient:
    client = MagicMock(spec=AragoraClient)
    return client


@pytest.fixture
def api(mock_client: AragoraClient) -> TenantsAPI:
    return TenantsAPI(mock_client)


SAMPLE_TENANT = {
    "id": "t-001",
    "name": "Acme Corp",
    "slug": "acme-corp",
    "status": "active",
    "tier": "pro",
    "owner_id": "u-owner-1",
    "quotas": {
        "debates_per_month": 5000,
        "users_per_org": 100,
        "storage_gb": 50,
        "api_calls_per_minute": 500,
        "concurrent_debates": 20,
        "knowledge_nodes": 50000,
    },
    "settings": {
        "allow_external_agents": False,
        "enable_knowledge_sharing": True,
        "data_retention_days": 730,
        "require_mfa": True,
        "allowed_domains": ["acme.com", "acme.io"],
        "custom_branding": {"logo_url": "https://acme.com/logo.png"},
    },
    "created_at": "2026-01-10T08:00:00Z",
    "updated_at": "2026-02-01T12:30:00Z",
    "metadata": {"industry": "tech"},
}

SAMPLE_TENANT_MINIMAL = {
    "id": "t-002",
    "name": "Startup Inc",
    "slug": "startup-inc",
    "status": "pending",
    "tier": "free",
    "owner_id": "u-owner-2",
}

SAMPLE_USAGE = {
    "tenant_id": "t-001",
    "debates_this_month": 42,
    "active_users": 15,
    "storage_used_gb": 3.7,
    "api_calls_today": 230,
    "knowledge_nodes_count": 1200,
    "period_start": "2026-02-01T00:00:00Z",
    "period_end": "2026-02-28T23:59:59Z",
}


# =========================================================================
# Tenant List
# =========================================================================


class TestTenantsList:
    def test_list_default(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"tenants": [SAMPLE_TENANT], "total": 1}
        tenants, total = api.list()
        assert len(tenants) == 1
        assert total == 1
        assert tenants[0].id == "t-001"
        assert tenants[0].name == "Acme Corp"
        mock_client._get.assert_called_once()

    def test_list_with_status_filter(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"tenants": [], "total": 0}
        api.list(status="suspended")
        params = mock_client._get.call_args[1]["params"]
        assert params["status"] == "suspended"

    def test_list_with_tier_filter(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"tenants": [], "total": 0}
        api.list(tier="enterprise")
        params = mock_client._get.call_args[1]["params"]
        assert params["tier"] == "enterprise"

    def test_list_with_all_filters(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"tenants": [], "total": 0}
        api.list(status="active", tier="pro", limit=10, offset=5)
        params = mock_client._get.call_args[1]["params"]
        assert params["status"] == "active"
        assert params["tier"] == "pro"
        assert params["limit"] == 10
        assert params["offset"] == 5

    def test_list_no_filters_excludes_none_params(
        self, api: TenantsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = {"tenants": [], "total": 0}
        api.list()
        params = mock_client._get.call_args[1]["params"]
        assert "status" not in params
        assert "tier" not in params

    def test_list_total_fallback(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"tenants": [SAMPLE_TENANT, SAMPLE_TENANT_MINIMAL]}
        tenants, total = api.list()
        assert total == 2

    @pytest.mark.asyncio
    async def test_list_async(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(
            return_value={"tenants": [SAMPLE_TENANT], "total": 1}
        )
        tenants, total = await api.list_async()
        assert len(tenants) == 1
        assert total == 1
        assert tenants[0].slug == "acme-corp"

    @pytest.mark.asyncio
    async def test_list_async_with_filters(
        self, api: TenantsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get_async = AsyncMock(
            return_value={"tenants": [], "total": 0}
        )
        await api.list_async(status="active", tier="free", limit=25, offset=10)
        params = mock_client._get_async.call_args[1]["params"]
        assert params["status"] == "active"
        assert params["tier"] == "free"
        assert params["limit"] == 25
        assert params["offset"] == 10


# =========================================================================
# Tenant Get
# =========================================================================


class TestTenantsGet:
    def test_get(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = {"tenant": SAMPLE_TENANT}
        tenant = api.get("t-001")
        assert isinstance(tenant, Tenant)
        assert tenant.id == "t-001"
        assert tenant.tier == "pro"
        mock_client._get.assert_called_once_with("/api/v1/tenants/t-001")

    def test_get_unwrapped_response(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_TENANT
        tenant = api.get("t-001")
        assert tenant.name == "Acme Corp"

    @pytest.mark.asyncio
    async def test_get_async(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value={"tenant": SAMPLE_TENANT})
        tenant = await api.get_async("t-001")
        assert tenant.status == "active"
        assert tenant.owner_id == "u-owner-1"


# =========================================================================
# Tenant Create
# =========================================================================


class TestTenantsCreate:
    def test_create_minimal(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"tenant": SAMPLE_TENANT_MINIMAL}
        tenant = api.create("Startup Inc", "startup-inc")
        assert isinstance(tenant, Tenant)
        assert tenant.name == "Startup Inc"
        assert tenant.slug == "startup-inc"
        mock_client._post.assert_called_once()
        body = mock_client._post.call_args[0][1]
        assert body["name"] == "Startup Inc"
        assert body["slug"] == "startup-inc"
        assert body["tier"] == "free"

    def test_create_with_tier(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"tenant": SAMPLE_TENANT}
        api.create("Acme Corp", "acme-corp", tier="enterprise")
        body = mock_client._post.call_args[0][1]
        assert body["tier"] == "enterprise"

    def test_create_with_owner(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"tenant": SAMPLE_TENANT}
        api.create("Acme Corp", "acme-corp", owner_id="u-owner-1")
        body = mock_client._post.call_args[0][1]
        assert body["owner_id"] == "u-owner-1"

    def test_create_with_settings(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"tenant": SAMPLE_TENANT}
        settings = {"require_mfa": True, "allowed_domains": ["acme.com"]}
        api.create("Acme Corp", "acme-corp", settings=settings)
        body = mock_client._post.call_args[0][1]
        assert body["settings"] == settings

    def test_create_with_quotas(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"tenant": SAMPLE_TENANT}
        quotas = {"debates_per_month": 10000, "storage_gb": 100}
        api.create("Acme Corp", "acme-corp", quotas=quotas)
        body = mock_client._post.call_args[0][1]
        assert body["quotas"] == quotas

    def test_create_excludes_none_optional_fields(
        self, api: TenantsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = {"tenant": SAMPLE_TENANT_MINIMAL}
        api.create("Test", "test")
        body = mock_client._post.call_args[0][1]
        assert "owner_id" not in body
        assert "settings" not in body
        assert "quotas" not in body

    def test_create_posts_to_correct_endpoint(
        self, api: TenantsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = {"tenant": SAMPLE_TENANT_MINIMAL}
        api.create("Test", "test")
        assert mock_client._post.call_args[0][0] == "/api/v1/tenants"

    @pytest.mark.asyncio
    async def test_create_async(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._post_async = AsyncMock(return_value={"tenant": SAMPLE_TENANT})
        tenant = await api.create_async("Acme Corp", "acme-corp", tier="pro", owner_id="u-owner-1")
        assert tenant.id == "t-001"
        body = mock_client._post_async.call_args[0][1]
        assert body["owner_id"] == "u-owner-1"
        assert body["tier"] == "pro"

    @pytest.mark.asyncio
    async def test_create_async_excludes_none_fields(
        self, api: TenantsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post_async = AsyncMock(return_value={"tenant": SAMPLE_TENANT_MINIMAL})
        await api.create_async("Test", "test")
        body = mock_client._post_async.call_args[0][1]
        assert "owner_id" not in body
        assert "settings" not in body
        assert "quotas" not in body


# =========================================================================
# Tenant Update
# =========================================================================


class TestTenantsUpdate:
    def test_update_name(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._patch.return_value = {"tenant": SAMPLE_TENANT}
        tenant = api.update("t-001", name="New Name")
        assert isinstance(tenant, Tenant)
        mock_client._patch.assert_called_once()
        body = mock_client._patch.call_args[0][1]
        assert body["name"] == "New Name"

    def test_update_tier(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._patch.return_value = {"tenant": SAMPLE_TENANT}
        api.update("t-001", tier="enterprise")
        body = mock_client._patch.call_args[0][1]
        assert body["tier"] == "enterprise"

    def test_update_settings(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._patch.return_value = {"tenant": SAMPLE_TENANT}
        settings = {"require_mfa": True}
        api.update("t-001", settings=settings)
        body = mock_client._patch.call_args[0][1]
        assert body["settings"] == settings

    def test_update_quotas(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._patch.return_value = {"tenant": SAMPLE_TENANT}
        quotas = {"storage_gb": 200}
        api.update("t-001", quotas=quotas)
        body = mock_client._patch.call_args[0][1]
        assert body["quotas"] == quotas

    def test_update_excludes_none_fields(
        self, api: TenantsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._patch.return_value = {"tenant": SAMPLE_TENANT}
        api.update("t-001", name="Only Name")
        body = mock_client._patch.call_args[0][1]
        assert "name" in body
        assert "tier" not in body
        assert "settings" not in body
        assert "quotas" not in body

    def test_update_endpoint(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._patch.return_value = {"tenant": SAMPLE_TENANT}
        api.update("t-001", name="X")
        assert mock_client._patch.call_args[0][0] == "/api/v1/tenants/t-001"

    @pytest.mark.asyncio
    async def test_update_async(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._patch_async = AsyncMock(return_value={"tenant": SAMPLE_TENANT})
        tenant = await api.update_async("t-001", name="Updated", tier="enterprise")
        assert tenant.id == "t-001"
        body = mock_client._patch_async.call_args[0][1]
        assert body["name"] == "Updated"
        assert body["tier"] == "enterprise"

    @pytest.mark.asyncio
    async def test_update_async_excludes_none(
        self, api: TenantsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._patch_async = AsyncMock(return_value={"tenant": SAMPLE_TENANT})
        await api.update_async("t-001", name="Only")
        body = mock_client._patch_async.call_args[0][1]
        assert "tier" not in body


# =========================================================================
# Tenant Delete
# =========================================================================


class TestTenantsDelete:
    def test_delete(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._delete.return_value = None
        result = api.delete("t-001")
        assert result is True
        mock_client._delete.assert_called_once_with("/api/v1/tenants/t-001")

    @pytest.mark.asyncio
    async def test_delete_async(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._delete_async = AsyncMock(return_value=None)
        result = await api.delete_async("t-001")
        assert result is True
        mock_client._delete_async.assert_called_once_with("/api/v1/tenants/t-001")


# =========================================================================
# Tenant Suspend / Activate
# =========================================================================


class TestTenantsSuspend:
    def test_suspend(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        suspended = {**SAMPLE_TENANT, "status": "suspended"}
        mock_client._post.return_value = {"tenant": suspended}
        tenant = api.suspend("t-001")
        assert isinstance(tenant, Tenant)
        assert tenant.status == "suspended"
        mock_client._post.assert_called_once()
        endpoint = mock_client._post.call_args[0][0]
        body = mock_client._post.call_args[0][1]
        assert endpoint == "/api/v1/tenants/t-001/status"
        assert body["action"] == "suspend"

    def test_suspend_with_reason(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"tenant": SAMPLE_TENANT}
        api.suspend("t-001", reason="Payment overdue")
        body = mock_client._post.call_args[0][1]
        assert body["reason"] == "Payment overdue"
        assert body["action"] == "suspend"

    def test_suspend_without_reason_excludes_field(
        self, api: TenantsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._post.return_value = {"tenant": SAMPLE_TENANT}
        api.suspend("t-001")
        body = mock_client._post.call_args[0][1]
        assert "reason" not in body

    @pytest.mark.asyncio
    async def test_suspend_async(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._post_async = AsyncMock(return_value={"tenant": SAMPLE_TENANT})
        tenant = await api.suspend_async("t-001", reason="TOS violation")
        assert isinstance(tenant, Tenant)
        body = mock_client._post_async.call_args[0][1]
        assert body["reason"] == "TOS violation"


class TestTenantsActivate:
    def test_activate(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        activated = {**SAMPLE_TENANT, "status": "active"}
        mock_client._post.return_value = {"tenant": activated}
        tenant = api.activate("t-001")
        assert isinstance(tenant, Tenant)
        assert tenant.status == "active"
        body = mock_client._post.call_args[0][1]
        assert body["action"] == "activate"

    def test_activate_endpoint(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._post.return_value = {"tenant": SAMPLE_TENANT}
        api.activate("t-001")
        assert mock_client._post.call_args[0][0] == "/api/v1/tenants/t-001/status"

    @pytest.mark.asyncio
    async def test_activate_async(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._post_async = AsyncMock(return_value={"tenant": SAMPLE_TENANT})
        tenant = await api.activate_async("t-001")
        assert isinstance(tenant, Tenant)


# =========================================================================
# Usage and Quotas
# =========================================================================


class TestTenantsGetUsage:
    def test_get_usage(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._get.return_value = SAMPLE_USAGE
        usage = api.get_usage("t-001")
        assert isinstance(usage, TenantUsage)
        assert usage.tenant_id == "t-001"
        assert usage.debates_this_month == 42
        assert usage.active_users == 15
        assert usage.storage_used_gb == 3.7
        assert usage.api_calls_today == 230
        assert usage.knowledge_nodes_count == 1200
        mock_client._get.assert_called_once_with("/api/v1/tenants/t-001/usage")

    def test_get_usage_parses_period_dates(
        self, api: TenantsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._get.return_value = SAMPLE_USAGE
        usage = api.get_usage("t-001")
        assert usage.period_start is not None
        assert usage.period_start.year == 2026
        assert usage.period_start.month == 2
        assert usage.period_end is not None

    @pytest.mark.asyncio
    async def test_get_usage_async(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._get_async = AsyncMock(return_value=SAMPLE_USAGE)
        usage = await api.get_usage_async("t-001")
        assert usage.debates_this_month == 42


class TestTenantsUpdateQuotas:
    def test_update_quotas(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._patch.return_value = {
            "quotas": {
                "debates_per_month": 9999,
                "users_per_org": 200,
                "storage_gb": 100,
                "api_calls_per_minute": 1000,
                "concurrent_debates": 50,
                "knowledge_nodes": 100000,
            }
        }
        quotas = api.update_quotas("t-001", {"debates_per_month": 9999})
        assert isinstance(quotas, TenantQuota)
        assert quotas.debates_per_month == 9999
        mock_client._patch.assert_called_once()
        assert mock_client._patch.call_args[0][0] == "/api/v1/tenants/t-001/quotas"

    def test_update_quotas_unwrapped(self, api: TenantsAPI, mock_client: AragoraClient) -> None:
        mock_client._patch.return_value = {"storage_gb": 500}
        quotas = api.update_quotas("t-001", {"storage_gb": 500})
        assert quotas.storage_gb == 500

    @pytest.mark.asyncio
    async def test_update_quotas_async(
        self, api: TenantsAPI, mock_client: AragoraClient
    ) -> None:
        mock_client._patch_async = AsyncMock(
            return_value={"quotas": {"concurrent_debates": 100}}
        )
        quotas = await api.update_quotas_async("t-001", {"concurrent_debates": 100})
        assert isinstance(quotas, TenantQuota)
        assert quotas.concurrent_debates == 100


# =========================================================================
# Parse Helpers
# =========================================================================


class TestParseTenant:
    def test_parse_full_tenant(self, api: TenantsAPI) -> None:
        tenant = api._parse_tenant(SAMPLE_TENANT)
        assert tenant.id == "t-001"
        assert tenant.name == "Acme Corp"
        assert tenant.slug == "acme-corp"
        assert tenant.status == "active"
        assert tenant.tier == "pro"
        assert tenant.owner_id == "u-owner-1"
        assert tenant.metadata == {"industry": "tech"}

    def test_parse_tenant_datetimes(self, api: TenantsAPI) -> None:
        tenant = api._parse_tenant(SAMPLE_TENANT)
        assert tenant.created_at is not None
        assert tenant.created_at.year == 2026
        assert tenant.created_at.month == 1
        assert tenant.updated_at is not None
        assert tenant.updated_at.year == 2026

    def test_parse_tenant_missing_datetimes(self, api: TenantsAPI) -> None:
        tenant = api._parse_tenant(SAMPLE_TENANT_MINIMAL)
        assert tenant.created_at is None
        assert tenant.updated_at is None

    def test_parse_tenant_invalid_datetime(self, api: TenantsAPI) -> None:
        data = {**SAMPLE_TENANT, "created_at": "not-a-date", "updated_at": "also-bad"}
        tenant = api._parse_tenant(data)
        assert tenant.created_at is None
        assert tenant.updated_at is None

    def test_parse_tenant_defaults(self, api: TenantsAPI) -> None:
        tenant = api._parse_tenant({})
        assert tenant.id == ""
        assert tenant.name == ""
        assert tenant.slug == ""
        assert tenant.status == "active"
        assert tenant.tier == "free"
        assert tenant.owner_id == ""
        assert tenant.metadata == {}

    def test_parse_tenant_quotas(self, api: TenantsAPI) -> None:
        tenant = api._parse_tenant(SAMPLE_TENANT)
        assert tenant.quotas.debates_per_month == 5000
        assert tenant.quotas.users_per_org == 100
        assert tenant.quotas.storage_gb == 50
        assert tenant.quotas.concurrent_debates == 20

    def test_parse_tenant_settings(self, api: TenantsAPI) -> None:
        tenant = api._parse_tenant(SAMPLE_TENANT)
        assert tenant.settings.allow_external_agents is False
        assert tenant.settings.enable_knowledge_sharing is True
        assert tenant.settings.data_retention_days == 730
        assert tenant.settings.require_mfa is True
        assert tenant.settings.allowed_domains == ["acme.com", "acme.io"]
        assert tenant.settings.custom_branding == {"logo_url": "https://acme.com/logo.png"}


class TestParseQuota:
    def test_parse_full_quota(self, api: TenantsAPI) -> None:
        data = {
            "debates_per_month": 3000,
            "users_per_org": 75,
            "storage_gb": 25,
            "api_calls_per_minute": 250,
            "concurrent_debates": 10,
            "knowledge_nodes": 20000,
        }
        quota = api._parse_quota(data)
        assert quota.debates_per_month == 3000
        assert quota.users_per_org == 75
        assert quota.storage_gb == 25
        assert quota.api_calls_per_minute == 250
        assert quota.concurrent_debates == 10
        assert quota.knowledge_nodes == 20000

    def test_parse_quota_defaults(self, api: TenantsAPI) -> None:
        quota = api._parse_quota({})
        assert quota.debates_per_month == 1000
        assert quota.users_per_org == 50
        assert quota.storage_gb == 10
        assert quota.api_calls_per_minute == 100
        assert quota.concurrent_debates == 5
        assert quota.knowledge_nodes == 10000


class TestParseSettings:
    def test_parse_full_settings(self, api: TenantsAPI) -> None:
        data = {
            "allow_external_agents": False,
            "enable_knowledge_sharing": True,
            "data_retention_days": 90,
            "require_mfa": True,
            "allowed_domains": ["example.com"],
            "custom_branding": {"color": "#ff0000"},
        }
        settings = api._parse_settings(data)
        assert settings.allow_external_agents is False
        assert settings.enable_knowledge_sharing is True
        assert settings.data_retention_days == 90
        assert settings.require_mfa is True
        assert settings.allowed_domains == ["example.com"]
        assert settings.custom_branding == {"color": "#ff0000"}

    def test_parse_settings_defaults(self, api: TenantsAPI) -> None:
        settings = api._parse_settings({})
        assert settings.allow_external_agents is True
        assert settings.enable_knowledge_sharing is False
        assert settings.data_retention_days == 365
        assert settings.require_mfa is False
        assert settings.allowed_domains == []
        assert settings.custom_branding == {}


class TestParseUsage:
    def test_parse_full_usage(self, api: TenantsAPI) -> None:
        usage = api._parse_usage(SAMPLE_USAGE)
        assert usage.tenant_id == "t-001"
        assert usage.debates_this_month == 42
        assert usage.active_users == 15
        assert usage.storage_used_gb == 3.7
        assert usage.api_calls_today == 230
        assert usage.knowledge_nodes_count == 1200
        assert usage.period_start is not None
        assert usage.period_end is not None

    def test_parse_usage_defaults(self, api: TenantsAPI) -> None:
        usage = api._parse_usage({})
        assert usage.tenant_id == ""
        assert usage.debates_this_month == 0
        assert usage.active_users == 0
        assert usage.storage_used_gb == 0.0
        assert usage.api_calls_today == 0
        assert usage.knowledge_nodes_count == 0
        assert usage.period_start is None
        assert usage.period_end is None

    def test_parse_usage_invalid_dates(self, api: TenantsAPI) -> None:
        data = {**SAMPLE_USAGE, "period_start": "bad-date", "period_end": "bad-date"}
        usage = api._parse_usage(data)
        assert usage.period_start is None
        assert usage.period_end is None


# =========================================================================
# Dataclass Construction
# =========================================================================


class TestDataclasses:
    def test_tenant_quota_defaults(self) -> None:
        quota = TenantQuota()
        assert quota.debates_per_month == 1000
        assert quota.users_per_org == 50
        assert quota.storage_gb == 10
        assert quota.api_calls_per_minute == 100
        assert quota.concurrent_debates == 5
        assert quota.knowledge_nodes == 10000

    def test_tenant_quota_custom(self) -> None:
        quota = TenantQuota(debates_per_month=9999, storage_gb=500)
        assert quota.debates_per_month == 9999
        assert quota.storage_gb == 500

    def test_tenant_settings_defaults(self) -> None:
        settings = TenantSettings()
        assert settings.allow_external_agents is True
        assert settings.enable_knowledge_sharing is False
        assert settings.data_retention_days == 365
        assert settings.require_mfa is False
        assert settings.allowed_domains == []
        assert settings.custom_branding == {}

    def test_tenant_settings_custom(self) -> None:
        settings = TenantSettings(
            require_mfa=True,
            allowed_domains=["corp.com"],
            custom_branding={"theme": "dark"},
        )
        assert settings.require_mfa is True
        assert settings.allowed_domains == ["corp.com"]
        assert settings.custom_branding == {"theme": "dark"}

    def test_tenant_construction(self) -> None:
        tenant = Tenant(
            id="t-x",
            name="Test Tenant",
            slug="test-tenant",
            status="active",
            tier="free",
            owner_id="u-1",
            quotas=TenantQuota(),
            settings=TenantSettings(),
        )
        assert tenant.id == "t-x"
        assert tenant.name == "Test Tenant"
        assert tenant.created_at is None
        assert tenant.updated_at is None
        assert tenant.metadata == {}

    def test_tenant_usage_defaults(self) -> None:
        usage = TenantUsage(tenant_id="t-1")
        assert usage.tenant_id == "t-1"
        assert usage.debates_this_month == 0
        assert usage.active_users == 0
        assert usage.storage_used_gb == 0.0
        assert usage.api_calls_today == 0
        assert usage.knowledge_nodes_count == 0
        assert usage.period_start is None
        assert usage.period_end is None

    def test_tenant_usage_custom(self) -> None:
        usage = TenantUsage(tenant_id="t-2", debates_this_month=100, storage_used_gb=5.5)
        assert usage.debates_this_month == 100
        assert usage.storage_used_gb == 5.5
