"""Tests for Partner namespace API.

Tests both PartnerAPI (sync) and AsyncPartnerAPI (async) classes for:
- Partner registration and profile
- API key management
- Usage statistics
- Webhook configuration
- Rate limit information
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient

# =========================================================================
# Registration
# =========================================================================


class TestPartnerRegister:
    """Tests for partner registration."""

    def test_register_minimal(self) -> None:
        """Register with required fields only."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "partner_id": "ptr_123",
                "status": "pending",
                "referral_code": "REF123",
            }

            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.partner.register(
                name="Test Partner",
                email="partner@example.com",
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/partners/register",
                json={
                    "name": "Test Partner",
                    "email": "partner@example.com",
                },
            )
            assert result["partner_id"] == "ptr_123"
            assert result["status"] == "pending"
            assert result["referral_code"] == "REF123"
            client.close()

    def test_register_with_company(self) -> None:
        """Register with optional company name."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"partner_id": "ptr_456"}

            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.partner.register(
                name="John Doe",
                email="john@company.com",
                company="Acme Inc",
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/partners/register",
                json={
                    "name": "John Doe",
                    "email": "john@company.com",
                    "company": "Acme Inc",
                },
            )
            client.close()


class TestPartnerProfile:
    """Tests for partner profile retrieval."""

    def test_get_profile(self) -> None:
        """Get current partner profile."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "partner_id": "ptr_123",
                "name": "Test Partner",
                "email": "partner@example.com",
                "company": "Test Company",
                "status": "active",
                "tier": "gold",
                "total_referrals": 42,
                "total_revenue": 1500.00,
            }

            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.partner.get_profile()

            mock_request.assert_called_once_with("GET", "/api/v1/partners/me")
            assert result["partner_id"] == "ptr_123"
            assert result["status"] == "active"
            assert result["tier"] == "gold"
            assert result["total_referrals"] == 42
            client.close()


# =========================================================================
# API Keys
# =========================================================================


class TestPartnerApiKeys:
    """Tests for API key management."""

    def test_create_api_key_minimal(self) -> None:
        """Create an API key with default name."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "key_id": "key_abc123",
                "key": "ak_live_supersecretkey",
                "scopes": [],
            }

            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.partner.create_api_key()

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/partners/keys",
                json={"name": "API Key"},
            )
            assert result["key_id"] == "key_abc123"
            assert result["key"] == "ak_live_supersecretkey"
            client.close()

    def test_create_api_key_with_name(self) -> None:
        """Create an API key with custom name."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"key_id": "key_xyz"}

            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.partner.create_api_key(name="Production Key")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/partners/keys",
                json={"name": "Production Key"},
            )
            client.close()

    def test_create_api_key_with_scopes(self) -> None:
        """Create an API key with specific scopes."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "key_id": "key_scoped",
                "scopes": ["debates:read", "debates:write"],
            }

            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.partner.create_api_key(
                name="Scoped Key",
                scopes=["debates:read", "debates:write"],
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/partners/keys",
                json={
                    "name": "Scoped Key",
                    "scopes": ["debates:read", "debates:write"],
                },
            )
            assert result["scopes"] == ["debates:read", "debates:write"]
            client.close()

    def test_create_api_key_with_expiration(self) -> None:
        """Create an API key with expiration."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"key_id": "key_exp"}

            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.partner.create_api_key(
                name="Expiring Key",
                expires_in_days=30,
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/partners/keys",
                json={
                    "name": "Expiring Key",
                    "expires_in_days": 30,
                },
            )
            client.close()

    def test_create_api_key_full_options(self) -> None:
        """Create an API key with all options."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"key_id": "key_full"}

            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.partner.create_api_key(
                name="Full Key",
                scopes=["debates:read", "agents:read"],
                expires_in_days=90,
            )

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/partners/keys",
                json={
                    "name": "Full Key",
                    "scopes": ["debates:read", "agents:read"],
                    "expires_in_days": 90,
                },
            )
            client.close()

    def test_list_api_keys(self) -> None:
        """List all API keys."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "keys": [
                    {"key_id": "key_1", "name": "Production", "active": True},
                    {"key_id": "key_2", "name": "Development", "active": True},
                    {"key_id": "key_3", "name": "Old Key", "active": False},
                ],
                "total": 3,
                "active_count": 2,
            }

            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.partner.list_api_keys()

            mock_request.assert_called_once_with("GET", "/api/v1/partners/keys")
            assert len(result["keys"]) == 3
            assert result["total"] == 3
            assert result["active_count"] == 2
            client.close()

    def test_revoke_api_key(self) -> None:
        """Revoke an API key."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"revoked": True, "message": "Key revoked successfully"}

            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.partner.revoke_api_key("key_123")

            mock_request.assert_called_once_with("DELETE", "/api/v1/partners/keys/key_123")
            assert result["revoked"] is True
            client.close()


# =========================================================================
# Usage
# =========================================================================


class TestPartnerUsage:
    """Tests for usage statistics."""

    def test_get_usage_default_days(self) -> None:
        """Get usage statistics with default 30 days."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "period_days": 30,
                "total_api_calls": 15000,
                "total_debates": 500,
                "total_referrals": 10,
                "revenue_generated": 250.00,
                "daily_breakdown": [],
            }

            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.partner.get_usage()

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/partners/usage",
                params={"days": 30},
            )
            assert result["period_days"] == 30
            assert result["total_api_calls"] == 15000
            client.close()

    def test_get_usage_custom_days(self) -> None:
        """Get usage statistics with custom days."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "period_days": 7,
                "total_api_calls": 3500,
            }

            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.partner.get_usage(days=7)

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/partners/usage",
                params={"days": 7},
            )
            assert result["period_days"] == 7
            client.close()

    def test_get_usage_max_days(self) -> None:
        """Get usage statistics with maximum days (365)."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"period_days": 365}

            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.partner.get_usage(days=365)

            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/partners/usage",
                params={"days": 365},
            )
            client.close()


# =========================================================================
# Webhooks
# =========================================================================


class TestPartnerWebhook:
    """Tests for webhook configuration."""

    def test_configure_webhook(self) -> None:
        """Configure webhook endpoint."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "webhook_url": "https://example.com/webhook",
                "webhook_secret": "whsec_supersecret123",
            }

            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.partner.configure_webhook("https://example.com/webhook")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/partners/webhooks",
                json={"url": "https://example.com/webhook"},
            )
            assert result["webhook_url"] == "https://example.com/webhook"
            assert result["webhook_secret"] == "whsec_supersecret123"
            client.close()

    def test_configure_webhook_https_required(self) -> None:
        """Configure webhook with HTTPS URL."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"webhook_url": "https://secure.example.com/hook"}

            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.partner.configure_webhook("https://secure.example.com/hook")

            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/partners/webhooks",
                json={"url": "https://secure.example.com/hook"},
            )
            assert result["webhook_url"].startswith("https://")
            client.close()


# =========================================================================
# Limits
# =========================================================================


class TestPartnerLimits:
    """Tests for rate limit information."""

    def test_get_limits(self) -> None:
        """Get rate limits for partner tier."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "tier": "gold",
                "limits": {
                    "requests_per_minute": 1000,
                    "requests_per_day": 100000,
                    "debates_per_day": 1000,
                    "concurrent_debates": 50,
                },
                "current_usage": {
                    "requests_this_minute": 42,
                    "requests_today": 5000,
                    "debates_today": 100,
                },
                "allowed": True,
            }

            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.partner.get_limits()

            mock_request.assert_called_once_with("GET", "/api/v1/partners/limits")
            assert result["tier"] == "gold"
            assert result["limits"]["requests_per_minute"] == 1000
            assert result["current_usage"]["requests_this_minute"] == 42
            assert result["allowed"] is True
            client.close()

    def test_get_limits_approaching_threshold(self) -> None:
        """Get limits when approaching threshold."""
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "tier": "silver",
                "limits": {"requests_per_minute": 500},
                "current_usage": {"requests_this_minute": 480},
                "allowed": True,
                "warning": "Approaching rate limit",
            }

            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.partner.get_limits()

            assert result["warning"] == "Approaching rate limit"
            assert result["allowed"] is True
            client.close()


# =========================================================================
# Async Tests
# =========================================================================


class TestAsyncPartnerRegister:
    """Tests for async partner registration."""

    @pytest.mark.asyncio
    async def test_async_register_minimal(self) -> None:
        """Register asynchronously with minimal fields."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"partner_id": "ptr_async"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.partner.register(
                    name="Async Partner",
                    email="async@example.com",
                )

                mock_request.assert_called_once_with(
                    "POST",
                    "/api/v1/partners/register",
                    json={
                        "name": "Async Partner",
                        "email": "async@example.com",
                    },
                )
                assert result["partner_id"] == "ptr_async"

    @pytest.mark.asyncio
    async def test_async_register_with_company(self) -> None:
        """Register asynchronously with company."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"partner_id": "ptr_company"}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.partner.register(
                    name="Company Partner",
                    email="partner@company.com",
                    company="Big Corp",
                )

                call_kwargs = mock_request.call_args
                assert call_kwargs[1]["json"]["company"] == "Big Corp"


class TestAsyncPartnerProfile:
    """Tests for async partner profile."""

    @pytest.mark.asyncio
    async def test_async_get_profile(self) -> None:
        """Get profile asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "partner_id": "ptr_123",
                "status": "active",
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.partner.get_profile()

                mock_request.assert_called_once_with("GET", "/api/v1/partners/me")
                assert result["status"] == "active"


class TestAsyncPartnerApiKeys:
    """Tests for async API key management."""

    @pytest.mark.asyncio
    async def test_async_create_api_key(self) -> None:
        """Create API key asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "key_id": "key_async",
                "key": "ak_live_asynckey",
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.partner.create_api_key(
                    name="Async Key",
                    scopes=["debates:read"],
                    expires_in_days=60,
                )

                mock_request.assert_called_once_with(
                    "POST",
                    "/api/v1/partners/keys",
                    json={
                        "name": "Async Key",
                        "scopes": ["debates:read"],
                        "expires_in_days": 60,
                    },
                )
                assert result["key_id"] == "key_async"

    @pytest.mark.asyncio
    async def test_async_list_api_keys(self) -> None:
        """List API keys asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "keys": [{"key_id": "key_1"}],
                "total": 1,
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.partner.list_api_keys()

                mock_request.assert_called_once_with("GET", "/api/v1/partners/keys")
                assert len(result["keys"]) == 1

    @pytest.mark.asyncio
    async def test_async_revoke_api_key(self) -> None:
        """Revoke API key asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"revoked": True}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.partner.revoke_api_key("key_to_revoke")

                mock_request.assert_called_once_with(
                    "DELETE", "/api/v1/partners/keys/key_to_revoke"
                )
                assert result["revoked"] is True


class TestAsyncPartnerUsage:
    """Tests for async usage statistics."""

    @pytest.mark.asyncio
    async def test_async_get_usage_default(self) -> None:
        """Get usage asynchronously with defaults."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "period_days": 30,
                "total_api_calls": 10000,
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.partner.get_usage()

                mock_request.assert_called_once_with(
                    "GET",
                    "/api/v1/partners/usage",
                    params={"days": 30},
                )
                assert result["period_days"] == 30

    @pytest.mark.asyncio
    async def test_async_get_usage_custom_days(self) -> None:
        """Get usage asynchronously with custom days."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"period_days": 14}

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                await client.partner.get_usage(days=14)

                call_kwargs = mock_request.call_args
                assert call_kwargs[1]["params"]["days"] == 14


class TestAsyncPartnerWebhook:
    """Tests for async webhook configuration."""

    @pytest.mark.asyncio
    async def test_async_configure_webhook(self) -> None:
        """Configure webhook asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "webhook_url": "https://async.example.com/hook",
                "webhook_secret": "whsec_async123",
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.partner.configure_webhook("https://async.example.com/hook")

                mock_request.assert_called_once_with(
                    "POST",
                    "/api/v1/partners/webhooks",
                    json={"url": "https://async.example.com/hook"},
                )
                assert result["webhook_secret"] == "whsec_async123"


class TestAsyncPartnerLimits:
    """Tests for async rate limit information."""

    @pytest.mark.asyncio
    async def test_async_get_limits(self) -> None:
        """Get limits asynchronously."""
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "tier": "platinum",
                "limits": {"requests_per_minute": 5000},
                "allowed": True,
            }

            async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
                result = await client.partner.get_limits()

                mock_request.assert_called_once_with("GET", "/api/v1/partners/limits")
                assert result["tier"] == "platinum"
                assert result["limits"]["requests_per_minute"] == 5000
                assert result["allowed"] is True
