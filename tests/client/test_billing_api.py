"""
Tests for BillingAPI resource.

Tests cover:
- Plans (list, get) sync and async
- Subscriptions (get, update, cancel) sync and async
- Usage (get, forecast) sync and async
- Invoices (list, get) sync and async
- Payment Methods (list, add, remove) sync and async
- Dataclass parsing (BillingPlan, Subscription, Invoice, UsageMetrics, UsageForecast)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.client.client import AragoraClient
from aragora.client.resources.billing import (
    BillingAPI,
    BillingPlan,
    Invoice,
    Subscription,
    UsageForecast,
    UsageMetrics,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_client() -> AragoraClient:
    """Create a mock AragoraClient."""
    return MagicMock(spec=AragoraClient)


@pytest.fixture
def billing_api(mock_client: AragoraClient) -> BillingAPI:
    """Create a BillingAPI with mock client."""
    return BillingAPI(mock_client)


# ============================================================================
# Dataclass Tests
# ============================================================================


class TestBillingPlanDataclass:
    """Tests for BillingPlan dataclass."""

    def test_billing_plan_required_fields(self):
        """Test BillingPlan with all required fields."""
        plan = BillingPlan(
            id="plan-starter",
            name="Starter",
            description="For small teams",
            price_monthly=29.0,
            price_yearly=290.0,
            features=["5 debates/month"],
            limits={"debates_per_month": 5},
        )
        assert plan.id == "plan-starter"
        assert plan.name == "Starter"
        assert plan.description == "For small teams"
        assert plan.price_monthly == 29.0
        assert plan.price_yearly == 290.0
        assert plan.features == ["5 debates/month"]
        assert plan.limits == {"debates_per_month": 5}
        assert plan.is_current is False

    def test_billing_plan_is_current(self):
        """Test BillingPlan with is_current flag."""
        plan = BillingPlan(
            id="plan-pro",
            name="Pro",
            description="For professionals",
            price_monthly=99.0,
            price_yearly=990.0,
            features=["Unlimited debates"],
            limits={},
            is_current=True,
        )
        assert plan.is_current is True


class TestSubscriptionDataclass:
    """Tests for Subscription dataclass."""

    def test_subscription_required_fields(self):
        """Test Subscription with required fields."""
        sub = Subscription(
            id="sub-123",
            plan_id="plan-pro",
            plan_name="Pro",
            status="active",
            current_period_start="2026-01-01T00:00:00Z",
            current_period_end="2026-02-01T00:00:00Z",
        )
        assert sub.id == "sub-123"
        assert sub.plan_id == "plan-pro"
        assert sub.plan_name == "Pro"
        assert sub.status == "active"
        assert sub.cancel_at_period_end is False

    def test_subscription_with_cancel(self):
        """Test Subscription with cancel_at_period_end."""
        sub = Subscription(
            id="sub-456",
            plan_id="plan-starter",
            plan_name="Starter",
            status="active",
            current_period_start="2026-01-01T00:00:00Z",
            current_period_end="2026-02-01T00:00:00Z",
            cancel_at_period_end=True,
        )
        assert sub.cancel_at_period_end is True


class TestInvoiceDataclass:
    """Tests for Invoice dataclass."""

    def test_invoice_required_fields(self):
        """Test Invoice with required fields."""
        inv = Invoice(
            id="inv-001",
            amount=29.99,
            currency="USD",
            status="paid",
            created_at="2026-01-15T10:00:00Z",
        )
        assert inv.id == "inv-001"
        assert inv.amount == 29.99
        assert inv.currency == "USD"
        assert inv.status == "paid"
        assert inv.paid_at is None
        assert inv.pdf_url is None

    def test_invoice_all_fields(self):
        """Test Invoice with all fields."""
        inv = Invoice(
            id="inv-002",
            amount=99.00,
            currency="EUR",
            status="paid",
            created_at="2026-01-15T10:00:00Z",
            paid_at="2026-01-15T10:01:00Z",
            pdf_url="https://billing.example.com/inv-002.pdf",
        )
        assert inv.paid_at == "2026-01-15T10:01:00Z"
        assert inv.pdf_url == "https://billing.example.com/inv-002.pdf"


class TestUsageMetricsDataclass:
    """Tests for UsageMetrics dataclass."""

    def test_usage_metrics_fields(self):
        """Test UsageMetrics with all fields."""
        usage = UsageMetrics(
            debates_count=42,
            api_calls=15000,
            storage_bytes=1048576,
            bandwidth_bytes=5242880,
            period_start="2026-01-01T00:00:00Z",
            period_end="2026-02-01T00:00:00Z",
        )
        assert usage.debates_count == 42
        assert usage.api_calls == 15000
        assert usage.storage_bytes == 1048576
        assert usage.bandwidth_bytes == 5242880


class TestUsageForecastDataclass:
    """Tests for UsageForecast dataclass."""

    def test_usage_forecast_fields(self):
        """Test UsageForecast with all fields."""
        forecast = UsageForecast(
            projected_cost=149.50,
            projected_debates=85,
            projected_api_calls=30000,
            confidence=0.87,
            based_on_days=15,
        )
        assert forecast.projected_cost == 149.50
        assert forecast.projected_debates == 85
        assert forecast.projected_api_calls == 30000
        assert forecast.confidence == 0.87
        assert forecast.based_on_days == 15


# ============================================================================
# Plans Tests
# ============================================================================


class TestListPlans:
    """Tests for list_plans() method."""

    def test_list_plans(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test listing billing plans."""
        mock_client._get.return_value = {
            "plans": [
                {
                    "id": "plan-starter",
                    "name": "Starter",
                    "description": "For small teams",
                    "price_monthly": 29.0,
                    "price_yearly": 290.0,
                    "features": ["5 debates/month"],
                    "limits": {"debates_per_month": 5},
                },
                {
                    "id": "plan-pro",
                    "name": "Pro",
                    "description": "For professionals",
                    "price_monthly": 99.0,
                    "price_yearly": 990.0,
                    "features": ["Unlimited debates", "Priority support"],
                    "limits": {},
                    "is_current": True,
                },
            ]
        }

        result = billing_api.list_plans()

        mock_client._get.assert_called_once_with("/api/v1/billing/plans")
        assert len(result) == 2
        assert all(isinstance(p, BillingPlan) for p in result)
        assert result[0].id == "plan-starter"
        assert result[0].name == "Starter"
        assert result[1].is_current is True

    def test_list_plans_empty(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test listing plans when empty."""
        mock_client._get.return_value = {"plans": []}

        result = billing_api.list_plans()

        assert result == []

    def test_list_plans_missing_key(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test listing plans when 'plans' key is missing from response."""
        mock_client._get.return_value = {}

        result = billing_api.list_plans()

        assert result == []


class TestListPlansAsync:
    """Tests for list_plans_async() method."""

    @pytest.mark.asyncio
    async def test_list_plans_async(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test async listing billing plans."""
        mock_client._get_async = AsyncMock(
            return_value={
                "plans": [
                    {
                        "id": "plan-async",
                        "name": "Async Plan",
                        "description": "Test",
                        "price_monthly": 10.0,
                        "price_yearly": 100.0,
                        "features": [],
                        "limits": {},
                    }
                ]
            }
        )

        result = await billing_api.list_plans_async()

        mock_client._get_async.assert_called_once_with("/api/v1/billing/plans")
        assert len(result) == 1
        assert isinstance(result[0], BillingPlan)
        assert result[0].id == "plan-async"


class TestGetPlan:
    """Tests for get_plan() method."""

    def test_get_plan(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test getting a specific plan."""
        mock_client._get.return_value = {
            "id": "plan-enterprise",
            "name": "Enterprise",
            "description": "Custom pricing",
            "price_monthly": 499.0,
            "price_yearly": 4990.0,
            "features": ["Unlimited everything", "Dedicated support", "SLA"],
            "limits": {},
        }

        result = billing_api.get_plan("plan-enterprise")

        mock_client._get.assert_called_once_with("/api/v1/billing/plans/plan-enterprise")
        assert isinstance(result, BillingPlan)
        assert result.id == "plan-enterprise"
        assert result.name == "Enterprise"
        assert "SLA" in result.features


class TestGetPlanAsync:
    """Tests for get_plan_async() method."""

    @pytest.mark.asyncio
    async def test_get_plan_async(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test async getting a specific plan."""
        mock_client._get_async = AsyncMock(
            return_value={
                "id": "plan-pro",
                "name": "Pro",
                "description": "Professional plan",
                "price_monthly": 99.0,
                "price_yearly": 990.0,
                "features": ["Unlimited debates"],
                "limits": {},
            }
        )

        result = await billing_api.get_plan_async("plan-pro")

        mock_client._get_async.assert_called_once_with("/api/v1/billing/plans/plan-pro")
        assert isinstance(result, BillingPlan)
        assert result.id == "plan-pro"


# ============================================================================
# Subscription Tests
# ============================================================================


class TestGetSubscription:
    """Tests for get_subscription() method."""

    def test_get_subscription(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test getting current subscription."""
        mock_client._get.return_value = {
            "id": "sub-789",
            "plan_id": "plan-pro",
            "plan_name": "Pro",
            "status": "active",
            "current_period_start": "2026-01-01T00:00:00Z",
            "current_period_end": "2026-02-01T00:00:00Z",
            "cancel_at_period_end": False,
        }

        result = billing_api.get_subscription()

        mock_client._get.assert_called_once_with("/api/v1/billing/subscription")
        assert isinstance(result, Subscription)
        assert result.id == "sub-789"
        assert result.plan_id == "plan-pro"
        assert result.status == "active"
        assert result.cancel_at_period_end is False


class TestGetSubscriptionAsync:
    """Tests for get_subscription_async() method."""

    @pytest.mark.asyncio
    async def test_get_subscription_async(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test async getting current subscription."""
        mock_client._get_async = AsyncMock(
            return_value={
                "id": "sub-async",
                "plan_id": "plan-starter",
                "plan_name": "Starter",
                "status": "active",
                "current_period_start": "2026-01-01T00:00:00Z",
                "current_period_end": "2026-02-01T00:00:00Z",
            }
        )

        result = await billing_api.get_subscription_async()

        mock_client._get_async.assert_called_once_with("/api/v1/billing/subscription")
        assert isinstance(result, Subscription)
        assert result.id == "sub-async"


class TestUpdateSubscription:
    """Tests for update_subscription() method."""

    def test_update_subscription(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test updating subscription to a new plan."""
        mock_client._post.return_value = {
            "id": "sub-789",
            "plan_id": "plan-enterprise",
            "plan_name": "Enterprise",
            "status": "active",
            "current_period_start": "2026-02-01T00:00:00Z",
            "current_period_end": "2026-03-01T00:00:00Z",
        }

        result = billing_api.update_subscription("plan-enterprise")

        mock_client._post.assert_called_once_with(
            "/api/v1/billing/subscription", data={"plan_id": "plan-enterprise"}
        )
        assert isinstance(result, Subscription)
        assert result.plan_id == "plan-enterprise"
        assert result.plan_name == "Enterprise"


class TestUpdateSubscriptionAsync:
    """Tests for update_subscription_async() method."""

    @pytest.mark.asyncio
    async def test_update_subscription_async(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test async updating subscription."""
        mock_client._post_async = AsyncMock(
            return_value={
                "id": "sub-async-up",
                "plan_id": "plan-pro",
                "plan_name": "Pro",
                "status": "active",
                "current_period_start": "2026-02-01T00:00:00Z",
                "current_period_end": "2026-03-01T00:00:00Z",
            }
        )

        result = await billing_api.update_subscription_async("plan-pro")

        mock_client._post_async.assert_called_once_with(
            "/api/v1/billing/subscription", data={"plan_id": "plan-pro"}
        )
        assert isinstance(result, Subscription)
        assert result.plan_id == "plan-pro"


class TestCancelSubscription:
    """Tests for cancel_subscription() method."""

    def test_cancel_subscription_at_period_end(
        self, billing_api: BillingAPI, mock_client: MagicMock
    ):
        """Test cancelling subscription at period end (default)."""
        mock_client._delete.return_value = {
            "id": "sub-789",
            "plan_id": "plan-pro",
            "plan_name": "Pro",
            "status": "active",
            "current_period_start": "2026-01-01T00:00:00Z",
            "current_period_end": "2026-02-01T00:00:00Z",
            "cancel_at_period_end": True,
        }

        result = billing_api.cancel_subscription()

        mock_client._delete.assert_called_once_with(
            "/api/v1/billing/subscription",
            params={"at_period_end": True},
        )
        assert isinstance(result, Subscription)
        assert result.cancel_at_period_end is True

    def test_cancel_subscription_immediately(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test cancelling subscription immediately."""
        mock_client._delete.return_value = {
            "id": "sub-789",
            "plan_id": "plan-pro",
            "plan_name": "Pro",
            "status": "cancelled",
            "current_period_start": "2026-01-01T00:00:00Z",
            "current_period_end": "2026-02-01T00:00:00Z",
            "cancel_at_period_end": False,
        }

        result = billing_api.cancel_subscription(at_period_end=False)

        mock_client._delete.assert_called_once_with(
            "/api/v1/billing/subscription",
            params={"at_period_end": False},
        )
        assert result.status == "cancelled"


class TestCancelSubscriptionAsync:
    """Tests for cancel_subscription_async() method."""

    @pytest.mark.asyncio
    async def test_cancel_subscription_async(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test async cancelling subscription."""
        mock_client._delete_async = AsyncMock(
            return_value={
                "id": "sub-async-cancel",
                "plan_id": "plan-pro",
                "plan_name": "Pro",
                "status": "active",
                "current_period_start": "2026-01-01T00:00:00Z",
                "current_period_end": "2026-02-01T00:00:00Z",
                "cancel_at_period_end": True,
            }
        )

        result = await billing_api.cancel_subscription_async()

        mock_client._delete_async.assert_called_once_with(
            "/api/v1/billing/subscription",
            params={"at_period_end": True},
        )
        assert isinstance(result, Subscription)

    @pytest.mark.asyncio
    async def test_cancel_subscription_async_immediately(
        self, billing_api: BillingAPI, mock_client: MagicMock
    ):
        """Test async cancelling subscription immediately."""
        mock_client._delete_async = AsyncMock(
            return_value={
                "id": "sub-async-cancel-now",
                "plan_id": "plan-pro",
                "plan_name": "Pro",
                "status": "cancelled",
                "current_period_start": "2026-01-01T00:00:00Z",
                "current_period_end": "2026-02-01T00:00:00Z",
            }
        )

        result = await billing_api.cancel_subscription_async(at_period_end=False)

        mock_client._delete_async.assert_called_once_with(
            "/api/v1/billing/subscription",
            params={"at_period_end": False},
        )
        assert result.status == "cancelled"


# ============================================================================
# Usage Tests
# ============================================================================


class TestGetUsage:
    """Tests for get_usage() method."""

    def test_get_usage_current_period(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test getting usage for current period."""
        mock_client._get.return_value = {
            "debates_count": 42,
            "api_calls": 15000,
            "storage_bytes": 1048576,
            "bandwidth_bytes": 5242880,
            "period_start": "2026-02-01T00:00:00Z",
            "period_end": "2026-03-01T00:00:00Z",
        }

        result = billing_api.get_usage()

        mock_client._get.assert_called_once_with("/api/v1/billing/usage", params={})
        assert isinstance(result, UsageMetrics)
        assert result.debates_count == 42
        assert result.api_calls == 15000

    def test_get_usage_specific_period(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test getting usage for a specific period."""
        mock_client._get.return_value = {
            "debates_count": 30,
            "api_calls": 10000,
            "storage_bytes": 524288,
            "bandwidth_bytes": 2621440,
            "period_start": "2026-01-01T00:00:00Z",
            "period_end": "2026-02-01T00:00:00Z",
        }

        result = billing_api.get_usage(period="2026-01")

        mock_client._get.assert_called_once_with(
            "/api/v1/billing/usage", params={"period": "2026-01"}
        )
        assert isinstance(result, UsageMetrics)
        assert result.debates_count == 30


class TestGetUsageAsync:
    """Tests for get_usage_async() method."""

    @pytest.mark.asyncio
    async def test_get_usage_async(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test async getting usage."""
        mock_client._get_async = AsyncMock(
            return_value={
                "debates_count": 10,
                "api_calls": 500,
                "storage_bytes": 256000,
                "bandwidth_bytes": 128000,
                "period_start": "2026-02-01T00:00:00Z",
                "period_end": "2026-03-01T00:00:00Z",
            }
        )

        result = await billing_api.get_usage_async()

        mock_client._get_async.assert_called_once_with("/api/v1/billing/usage", params={})
        assert isinstance(result, UsageMetrics)
        assert result.debates_count == 10

    @pytest.mark.asyncio
    async def test_get_usage_async_with_period(
        self, billing_api: BillingAPI, mock_client: MagicMock
    ):
        """Test async getting usage for a specific period."""
        mock_client._get_async = AsyncMock(
            return_value={
                "debates_count": 20,
                "api_calls": 8000,
                "storage_bytes": 512000,
                "bandwidth_bytes": 1024000,
                "period_start": "2026-01-01T00:00:00Z",
                "period_end": "2026-02-01T00:00:00Z",
            }
        )

        result = await billing_api.get_usage_async(period="2026-01")

        mock_client._get_async.assert_called_once_with(
            "/api/v1/billing/usage", params={"period": "2026-01"}
        )
        assert result.debates_count == 20


class TestGetUsageForecast:
    """Tests for get_usage_forecast() method."""

    def test_get_usage_forecast(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test getting usage forecast."""
        mock_client._get.return_value = {
            "projected_cost": 149.50,
            "projected_debates": 85,
            "projected_api_calls": 30000,
            "confidence": 0.87,
            "based_on_days": 15,
        }

        result = billing_api.get_usage_forecast()

        mock_client._get.assert_called_once_with("/api/v1/billing/usage/forecast")
        assert isinstance(result, UsageForecast)
        assert result.projected_cost == 149.50
        assert result.projected_debates == 85
        assert result.confidence == 0.87
        assert result.based_on_days == 15


class TestGetUsageForecastAsync:
    """Tests for get_usage_forecast_async() method."""

    @pytest.mark.asyncio
    async def test_get_usage_forecast_async(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test async getting usage forecast."""
        mock_client._get_async = AsyncMock(
            return_value={
                "projected_cost": 200.00,
                "projected_debates": 100,
                "projected_api_calls": 40000,
                "confidence": 0.92,
                "based_on_days": 20,
            }
        )

        result = await billing_api.get_usage_forecast_async()

        mock_client._get_async.assert_called_once_with("/api/v1/billing/usage/forecast")
        assert isinstance(result, UsageForecast)
        assert result.projected_cost == 200.00
        assert result.confidence == 0.92


# ============================================================================
# Invoice Tests
# ============================================================================


class TestListInvoices:
    """Tests for list_invoices() method."""

    def test_list_invoices_defaults(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test listing invoices with default pagination."""
        mock_client._get.return_value = {
            "invoices": [
                {
                    "id": "inv-001",
                    "amount": 99.00,
                    "currency": "USD",
                    "status": "paid",
                    "created_at": "2026-01-15T10:00:00Z",
                    "paid_at": "2026-01-15T10:01:00Z",
                },
                {
                    "id": "inv-002",
                    "amount": 99.00,
                    "currency": "USD",
                    "status": "paid",
                    "created_at": "2025-12-15T10:00:00Z",
                },
            ],
            "total": 5,
        }

        invoices, total = billing_api.list_invoices()

        mock_client._get.assert_called_once_with(
            "/api/v1/billing/invoices", params={"limit": 10, "offset": 0}
        )
        assert len(invoices) == 2
        assert total == 5
        assert all(isinstance(inv, Invoice) for inv in invoices)
        assert invoices[0].id == "inv-001"
        assert invoices[0].paid_at == "2026-01-15T10:01:00Z"
        assert invoices[1].paid_at is None

    def test_list_invoices_custom_pagination(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test listing invoices with custom pagination."""
        mock_client._get.return_value = {
            "invoices": [
                {
                    "id": "inv-003",
                    "amount": 29.00,
                    "currency": "USD",
                    "status": "pending",
                    "created_at": "2026-02-01T10:00:00Z",
                }
            ],
            "total": 10,
        }

        invoices, total = billing_api.list_invoices(limit=5, offset=5)

        mock_client._get.assert_called_once_with(
            "/api/v1/billing/invoices", params={"limit": 5, "offset": 5}
        )
        assert len(invoices) == 1
        assert total == 10

    def test_list_invoices_empty(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test listing invoices when none exist."""
        mock_client._get.return_value = {"invoices": [], "total": 0}

        invoices, total = billing_api.list_invoices()

        assert invoices == []
        assert total == 0

    def test_list_invoices_missing_total(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test listing invoices when total is not in response."""
        mock_client._get.return_value = {
            "invoices": [
                {
                    "id": "inv-x",
                    "amount": 50.0,
                    "currency": "USD",
                    "status": "paid",
                    "created_at": "2026-01-01T00:00:00Z",
                }
            ]
        }

        invoices, total = billing_api.list_invoices()

        assert len(invoices) == 1
        assert total == 1  # Falls back to len(invoices)


class TestListInvoicesAsync:
    """Tests for list_invoices_async() method."""

    @pytest.mark.asyncio
    async def test_list_invoices_async(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test async listing invoices."""
        mock_client._get_async = AsyncMock(
            return_value={
                "invoices": [
                    {
                        "id": "inv-async-001",
                        "amount": 99.00,
                        "currency": "USD",
                        "status": "paid",
                        "created_at": "2026-01-15T10:00:00Z",
                    }
                ],
                "total": 1,
            }
        )

        invoices, total = await billing_api.list_invoices_async()

        mock_client._get_async.assert_called_once_with(
            "/api/v1/billing/invoices", params={"limit": 10, "offset": 0}
        )
        assert len(invoices) == 1
        assert total == 1
        assert isinstance(invoices[0], Invoice)

    @pytest.mark.asyncio
    async def test_list_invoices_async_with_pagination(
        self, billing_api: BillingAPI, mock_client: MagicMock
    ):
        """Test async listing invoices with custom pagination."""
        mock_client._get_async = AsyncMock(return_value={"invoices": [], "total": 0})

        invoices, total = await billing_api.list_invoices_async(limit=20, offset=40)

        mock_client._get_async.assert_called_once_with(
            "/api/v1/billing/invoices", params={"limit": 20, "offset": 40}
        )
        assert invoices == []
        assert total == 0


class TestGetInvoice:
    """Tests for get_invoice() method."""

    def test_get_invoice(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test getting a specific invoice."""
        mock_client._get.return_value = {
            "id": "inv-details",
            "amount": 199.00,
            "currency": "USD",
            "status": "paid",
            "created_at": "2026-01-15T10:00:00Z",
            "paid_at": "2026-01-15T10:05:00Z",
            "pdf_url": "https://billing.example.com/inv-details.pdf",
        }

        result = billing_api.get_invoice("inv-details")

        mock_client._get.assert_called_once_with("/api/v1/billing/invoices/inv-details")
        assert isinstance(result, Invoice)
        assert result.id == "inv-details"
        assert result.amount == 199.00
        assert result.pdf_url == "https://billing.example.com/inv-details.pdf"


class TestGetInvoiceAsync:
    """Tests for get_invoice_async() method."""

    @pytest.mark.asyncio
    async def test_get_invoice_async(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test async getting a specific invoice."""
        mock_client._get_async = AsyncMock(
            return_value={
                "id": "inv-async-detail",
                "amount": 49.00,
                "currency": "EUR",
                "status": "pending",
                "created_at": "2026-02-10T12:00:00Z",
            }
        )

        result = await billing_api.get_invoice_async("inv-async-detail")

        mock_client._get_async.assert_called_once_with("/api/v1/billing/invoices/inv-async-detail")
        assert isinstance(result, Invoice)
        assert result.currency == "EUR"
        assert result.status == "pending"


# ============================================================================
# Payment Methods Tests
# ============================================================================


class TestListPaymentMethods:
    """Tests for list_payment_methods() method."""

    def test_list_payment_methods(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test listing payment methods."""
        mock_client._get.return_value = {
            "payment_methods": [
                {
                    "id": "pm-visa",
                    "type": "card",
                    "brand": "visa",
                    "last4": "4242",
                    "exp_month": 12,
                    "exp_year": 2028,
                    "is_default": True,
                },
                {
                    "id": "pm-mc",
                    "type": "card",
                    "brand": "mastercard",
                    "last4": "5555",
                    "exp_month": 6,
                    "exp_year": 2027,
                    "is_default": False,
                },
            ]
        }

        result = billing_api.list_payment_methods()

        mock_client._get.assert_called_once_with("/api/v1/billing/payment-methods")
        assert len(result) == 2
        assert result[0]["brand"] == "visa"
        assert result[0]["is_default"] is True
        assert result[1]["last4"] == "5555"

    def test_list_payment_methods_empty(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test listing payment methods when none exist."""
        mock_client._get.return_value = {"payment_methods": []}

        result = billing_api.list_payment_methods()

        assert result == []

    def test_list_payment_methods_missing_key(
        self, billing_api: BillingAPI, mock_client: MagicMock
    ):
        """Test listing payment methods when key is missing."""
        mock_client._get.return_value = {}

        result = billing_api.list_payment_methods()

        assert result == []


class TestListPaymentMethodsAsync:
    """Tests for list_payment_methods_async() method."""

    @pytest.mark.asyncio
    async def test_list_payment_methods_async(
        self, billing_api: BillingAPI, mock_client: MagicMock
    ):
        """Test async listing payment methods."""
        mock_client._get_async = AsyncMock(
            return_value={
                "payment_methods": [
                    {"id": "pm-async", "type": "card", "last4": "1234"},
                ]
            }
        )

        result = await billing_api.list_payment_methods_async()

        mock_client._get_async.assert_called_once_with("/api/v1/billing/payment-methods")
        assert len(result) == 1
        assert result[0]["id"] == "pm-async"


class TestAddPaymentMethod:
    """Tests for add_payment_method() method."""

    def test_add_payment_method_default(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test adding a payment method as default."""
        mock_client._post.return_value = {
            "id": "pm-new",
            "type": "card",
            "brand": "visa",
            "last4": "9999",
            "is_default": True,
        }

        result = billing_api.add_payment_method("tok_visa_test")

        mock_client._post.assert_called_once_with(
            "/api/v1/billing/payment-methods",
            data={"token": "tok_visa_test", "set_default": True},
        )
        assert result["id"] == "pm-new"
        assert result["is_default"] is True

    def test_add_payment_method_not_default(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test adding a payment method without setting as default."""
        mock_client._post.return_value = {
            "id": "pm-secondary",
            "type": "card",
            "brand": "mastercard",
            "last4": "8888",
            "is_default": False,
        }

        result = billing_api.add_payment_method("tok_mc_test", set_default=False)

        mock_client._post.assert_called_once_with(
            "/api/v1/billing/payment-methods",
            data={"token": "tok_mc_test", "set_default": False},
        )
        assert result["is_default"] is False


class TestAddPaymentMethodAsync:
    """Tests for add_payment_method_async() method."""

    @pytest.mark.asyncio
    async def test_add_payment_method_async(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test async adding a payment method."""
        mock_client._post_async = AsyncMock(
            return_value={"id": "pm-async-new", "type": "card", "last4": "7777"}
        )

        result = await billing_api.add_payment_method_async("tok_async_test")

        mock_client._post_async.assert_called_once_with(
            "/api/v1/billing/payment-methods",
            data={"token": "tok_async_test", "set_default": True},
        )
        assert result["id"] == "pm-async-new"

    @pytest.mark.asyncio
    async def test_add_payment_method_async_not_default(
        self, billing_api: BillingAPI, mock_client: MagicMock
    ):
        """Test async adding a non-default payment method."""
        mock_client._post_async = AsyncMock(
            return_value={"id": "pm-async-sec", "is_default": False}
        )

        result = await billing_api.add_payment_method_async("tok_test", set_default=False)

        mock_client._post_async.assert_called_once_with(
            "/api/v1/billing/payment-methods",
            data={"token": "tok_test", "set_default": False},
        )
        assert result["is_default"] is False


class TestRemovePaymentMethod:
    """Tests for remove_payment_method() method."""

    def test_remove_payment_method(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test removing a payment method."""
        mock_client._delete.return_value = None

        billing_api.remove_payment_method("pm-old")

        mock_client._delete.assert_called_once_with("/api/v1/billing/payment-methods/pm-old")


class TestRemovePaymentMethodAsync:
    """Tests for remove_payment_method_async() method."""

    @pytest.mark.asyncio
    async def test_remove_payment_method_async(
        self, billing_api: BillingAPI, mock_client: MagicMock
    ):
        """Test async removing a payment method."""
        mock_client._delete_async = AsyncMock(return_value=None)

        await billing_api.remove_payment_method_async("pm-async-old")

        mock_client._delete_async.assert_called_once_with(
            "/api/v1/billing/payment-methods/pm-async-old"
        )


# ============================================================================
# Integration-like Tests
# ============================================================================


class TestBillingAPIIntegration:
    """Integration-like tests for BillingAPI."""

    def test_full_subscription_workflow(self, billing_api: BillingAPI, mock_client: MagicMock):
        """Test full workflow: list plans -> subscribe -> check usage -> cancel."""
        # List plans
        mock_client._get.return_value = {
            "plans": [
                {
                    "id": "plan-pro",
                    "name": "Pro",
                    "description": "Professional",
                    "price_monthly": 99.0,
                    "price_yearly": 990.0,
                    "features": ["Unlimited debates"],
                    "limits": {},
                },
            ]
        }
        plans = billing_api.list_plans()
        assert len(plans) == 1
        chosen_plan = plans[0]

        # Subscribe
        mock_client._post.return_value = {
            "id": "sub-new",
            "plan_id": chosen_plan.id,
            "plan_name": chosen_plan.name,
            "status": "active",
            "current_period_start": "2026-02-01T00:00:00Z",
            "current_period_end": "2026-03-01T00:00:00Z",
        }
        subscription = billing_api.update_subscription(chosen_plan.id)
        assert subscription.plan_id == "plan-pro"
        assert subscription.status == "active"

        # Check usage
        mock_client._get.return_value = {
            "debates_count": 10,
            "api_calls": 500,
            "storage_bytes": 100000,
            "bandwidth_bytes": 50000,
            "period_start": "2026-02-01T00:00:00Z",
            "period_end": "2026-03-01T00:00:00Z",
        }
        usage = billing_api.get_usage()
        assert usage.debates_count == 10

        # Cancel
        mock_client._delete.return_value = {
            "id": "sub-new",
            "plan_id": "plan-pro",
            "plan_name": "Pro",
            "status": "active",
            "current_period_start": "2026-02-01T00:00:00Z",
            "current_period_end": "2026-03-01T00:00:00Z",
            "cancel_at_period_end": True,
        }
        cancelled = billing_api.cancel_subscription()
        assert cancelled.cancel_at_period_end is True

    def test_payment_method_and_invoice_workflow(
        self, billing_api: BillingAPI, mock_client: MagicMock
    ):
        """Test adding payment method and viewing invoices."""
        # Add payment method
        mock_client._post.return_value = {
            "id": "pm-new",
            "type": "card",
            "brand": "visa",
            "last4": "4242",
            "is_default": True,
        }
        pm = billing_api.add_payment_method("tok_visa")
        assert pm["brand"] == "visa"

        # List invoices
        mock_client._get.return_value = {
            "invoices": [
                {
                    "id": "inv-recent",
                    "amount": 99.00,
                    "currency": "USD",
                    "status": "paid",
                    "created_at": "2026-02-01T00:00:00Z",
                    "pdf_url": "https://billing.example.com/inv-recent.pdf",
                }
            ],
            "total": 1,
        }
        invoices, total = billing_api.list_invoices()
        assert total == 1
        assert invoices[0].pdf_url is not None

        # Get specific invoice
        mock_client._get.return_value = {
            "id": "inv-recent",
            "amount": 99.00,
            "currency": "USD",
            "status": "paid",
            "created_at": "2026-02-01T00:00:00Z",
            "paid_at": "2026-02-01T00:05:00Z",
            "pdf_url": "https://billing.example.com/inv-recent.pdf",
        }
        invoice = billing_api.get_invoice("inv-recent")
        assert invoice.paid_at is not None
