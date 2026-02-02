"""Tests for Billing namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora_sdk.client import AragoraAsyncClient, AragoraClient


class TestBillingUsage:
    """Tests for billing usage and plans methods."""

    def test_get_plans(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "plans": [
                    {"tier": "starter", "price": 29},
                    {"tier": "professional", "price": 99},
                ],
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.billing.get_plans()
            mock_request.assert_called_once_with("GET", "/api/v1/billing/plans")
            assert len(result["plans"]) == 2
            assert result["plans"][0]["tier"] == "starter"
            client.close()

    def test_get_usage(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "debates": 42,
                "tokens": 150000,
                "cost": 12.50,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.billing.get_usage()
            mock_request.assert_called_once_with("GET", "/api/v1/billing/usage")
            assert result["debates"] == 42
            assert result["cost"] == 12.50
            client.close()

    def test_get_usage_forecast(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "projected_cost": 45.00,
                "recommendations": ["Consider upgrading to professional tier"],
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.billing.get_usage_forecast()
            mock_request.assert_called_once_with("GET", "/api/v1/billing/usage/forecast")
            assert result["projected_cost"] == 45.00
            client.close()

    def test_export_usage_csv_with_date_range(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = b"date,cost\n2025-01-01,5.00"
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.billing.export_usage_csv(
                start="2025-01-01",
                end="2025-01-31",
            )
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/billing/usage/export",
                params={"start": "2025-01-01", "end": "2025-01-31"},
            )
            assert result == b"date,cost\n2025-01-01,5.00"
            client.close()


class TestBillingInvoices:
    """Tests for invoice operations."""

    def test_get_invoices_default_limit(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "invoices": [{"id": "inv_001", "amount": 99.00, "status": "paid"}],
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.billing.get_invoices()
            mock_request.assert_called_once_with(
                "GET", "/api/v1/billing/invoices", params={"limit": 10}
            )
            assert result["invoices"][0]["id"] == "inv_001"
            client.close()

    def test_get_invoices_custom_limit(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"invoices": []}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            client.billing.get_invoices(limit=50)
            mock_request.assert_called_once_with(
                "GET", "/api/v1/billing/invoices", params={"limit": 50}
            )
            client.close()


class TestBillingSubscription:
    """Tests for subscription operations."""

    def test_get_subscription(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "tier": "professional",
                "status": "active",
                "limits": {"debates_per_month": 100},
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.billing.get_subscription()
            mock_request.assert_called_once_with("GET", "/api/v1/billing/subscription")
            assert result["tier"] == "professional"
            assert result["status"] == "active"
            client.close()

    def test_create_checkout(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "session_id": "cs_abc123",
                "url": "https://checkout.stripe.com/session/cs_abc123",
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.billing.create_checkout(
                tier="professional",
                success_url="https://app.aragora.ai/billing?success=true",
                cancel_url="https://app.aragora.ai/billing?canceled=true",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/billing/checkout",
                json={
                    "tier": "professional",
                    "success_url": "https://app.aragora.ai/billing?success=true",
                    "cancel_url": "https://app.aragora.ai/billing?canceled=true",
                },
            )
            assert result["session_id"] == "cs_abc123"
            client.close()

    def test_cancel_subscription(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "status": "canceling",
                "effective_date": "2025-02-28",
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.billing.cancel_subscription()
            mock_request.assert_called_once_with("POST", "/api/v1/billing/cancel")
            assert result["status"] == "canceling"
            client.close()

    def test_resume_subscription(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"status": "active", "tier": "professional"}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.billing.resume_subscription()
            mock_request.assert_called_once_with("POST", "/api/v1/billing/resume")
            assert result["status"] == "active"
            client.close()

    def test_create_portal(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "url": "https://billing.stripe.com/session/bps_abc123",
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.billing.create_portal(
                return_url="https://app.aragora.ai/billing",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/billing/portal",
                json={"return_url": "https://app.aragora.ai/billing"},
            )
            assert "url" in result
            client.close()


class TestBillingAuditLog:
    """Tests for billing audit log."""

    def test_get_audit_log_defaults(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"entries": [], "total": 0}
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.billing.get_audit_log()
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/billing/audit-log",
                params={"limit": 50, "offset": 0},
            )
            assert result["total"] == 0
            client.close()

    def test_get_audit_log_with_action_filter(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "entries": [{"action": "subscription.canceled"}],
                "total": 1,
            }
            client = AragoraClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = client.billing.get_audit_log(
                limit=10, offset=0, action="subscription.canceled"
            )
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/billing/audit-log",
                params={"limit": 10, "offset": 0, "action": "subscription.canceled"},
            )
            assert result["entries"][0]["action"] == "subscription.canceled"
            client.close()


class TestAsyncBilling:
    """Tests for async billing methods."""

    @pytest.mark.asyncio
    async def test_get_usage(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"debates": 10, "tokens": 50000, "cost": 5.00}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.billing.get_usage()
            mock_request.assert_called_once_with("GET", "/api/v1/billing/usage")
            assert result["debates"] == 10
            await client.close()

    @pytest.mark.asyncio
    async def test_get_subscription(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"tier": "starter", "status": "active"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.billing.get_subscription()
            mock_request.assert_called_once_with("GET", "/api/v1/billing/subscription")
            assert result["tier"] == "starter"
            await client.close()

    @pytest.mark.asyncio
    async def test_create_checkout(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "session_id": "cs_async_123",
                "url": "https://checkout.stripe.com/cs_async_123",
            }
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.billing.create_checkout(
                tier="enterprise",
                success_url="https://app.aragora.ai/success",
                cancel_url="https://app.aragora.ai/cancel",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/billing/checkout",
                json={
                    "tier": "enterprise",
                    "success_url": "https://app.aragora.ai/success",
                    "cancel_url": "https://app.aragora.ai/cancel",
                },
            )
            assert result["session_id"] == "cs_async_123"
            await client.close()

    @pytest.mark.asyncio
    async def test_get_invoices(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"invoices": [{"id": "inv_async_001", "amount": 29.00}]}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.billing.get_invoices(limit=5)
            mock_request.assert_called_once_with(
                "GET", "/api/v1/billing/invoices", params={"limit": 5}
            )
            assert result["invoices"][0]["id"] == "inv_async_001"
            await client.close()

    @pytest.mark.asyncio
    async def test_cancel_subscription(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"status": "canceling"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.billing.cancel_subscription()
            mock_request.assert_called_once_with("POST", "/api/v1/billing/cancel")
            assert result["status"] == "canceling"
            await client.close()

    @pytest.mark.asyncio
    async def test_get_audit_log(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"entries": [], "total": 0}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai", api_key="test-key")
            result = await client.billing.get_audit_log(limit=25, offset=10)
            mock_request.assert_called_once_with(
                "GET",
                "/api/v1/billing/audit-log",
                params={"limit": 25, "offset": 10},
            )
            assert result["total"] == 0
            await client.close()
