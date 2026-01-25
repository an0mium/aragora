"""BillingAPI resource for the Aragora client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..client import AragoraClient


@dataclass
class BillingPlan:
    """Billing plan details."""

    id: str
    name: str
    description: str
    price_monthly: float
    price_yearly: float
    features: list[str]
    limits: dict[str, Any]
    is_current: bool = False


@dataclass
class Subscription:
    """User subscription details."""

    id: str
    plan_id: str
    plan_name: str
    status: str
    current_period_start: str
    current_period_end: str
    cancel_at_period_end: bool = False


@dataclass
class Invoice:
    """Invoice details."""

    id: str
    amount: float
    currency: str
    status: str
    created_at: str
    paid_at: Optional[str] = None
    pdf_url: Optional[str] = None


@dataclass
class UsageMetrics:
    """Usage metrics for billing."""

    debates_count: int
    api_calls: int
    storage_bytes: int
    bandwidth_bytes: int
    period_start: str
    period_end: str


@dataclass
class UsageForecast:
    """Usage forecast for the billing period."""

    projected_cost: float
    projected_debates: int
    projected_api_calls: int
    confidence: float
    based_on_days: int


class BillingAPI:
    """API interface for billing and subscription management."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    # -------------------------------------------------------------------------
    # Plans
    # -------------------------------------------------------------------------

    def list_plans(self) -> list[BillingPlan]:
        """
        List all available billing plans.

        Returns:
            List of BillingPlan objects.
        """
        response = self._client._get("/api/v1/billing/plans")
        plans = response.get("plans", [])
        return [BillingPlan(**p) for p in plans]

    async def list_plans_async(self) -> list[BillingPlan]:
        """Async version of list_plans()."""
        response = await self._client._get_async("/api/v1/billing/plans")
        plans = response.get("plans", [])
        return [BillingPlan(**p) for p in plans]

    def get_plan(self, plan_id: str) -> BillingPlan:
        """
        Get a specific billing plan.

        Args:
            plan_id: The plan ID.

        Returns:
            BillingPlan object.
        """
        response = self._client._get(f"/api/v1/billing/plans/{plan_id}")
        return BillingPlan(**response)

    async def get_plan_async(self, plan_id: str) -> BillingPlan:
        """Async version of get_plan()."""
        response = await self._client._get_async(f"/api/v1/billing/plans/{plan_id}")
        return BillingPlan(**response)

    # -------------------------------------------------------------------------
    # Subscriptions
    # -------------------------------------------------------------------------

    def get_subscription(self) -> Subscription:
        """
        Get the current user's subscription.

        Returns:
            Subscription object.
        """
        response = self._client._get("/api/v1/billing/subscription")
        return Subscription(**response)

    async def get_subscription_async(self) -> Subscription:
        """Async version of get_subscription()."""
        response = await self._client._get_async("/api/v1/billing/subscription")
        return Subscription(**response)

    def update_subscription(self, plan_id: str) -> Subscription:
        """
        Change the subscription plan.

        Args:
            plan_id: The new plan ID to subscribe to.

        Returns:
            Updated Subscription object.
        """
        response = self._client._post("/api/v1/billing/subscription", json={"plan_id": plan_id})
        return Subscription(**response)

    async def update_subscription_async(self, plan_id: str) -> Subscription:
        """Async version of update_subscription()."""
        response = await self._client._post_async(
            "/api/v1/billing/subscription", json={"plan_id": plan_id}
        )
        return Subscription(**response)

    def cancel_subscription(self, at_period_end: bool = True) -> Subscription:
        """
        Cancel the subscription.

        Args:
            at_period_end: If True, cancel at end of billing period.

        Returns:
            Updated Subscription object.
        """
        response = self._client._delete(
            "/api/v1/billing/subscription",
            params={"at_period_end": at_period_end},
        )
        return Subscription(**response)

    async def cancel_subscription_async(self, at_period_end: bool = True) -> Subscription:
        """Async version of cancel_subscription()."""
        response = await self._client._delete_async(
            "/api/v1/billing/subscription",
            params={"at_period_end": at_period_end},
        )
        return Subscription(**response)

    # -------------------------------------------------------------------------
    # Usage
    # -------------------------------------------------------------------------

    def get_usage(self, period: Optional[str] = None) -> UsageMetrics:
        """
        Get usage metrics for the current billing period.

        Args:
            period: Optional period identifier (e.g., "2024-01").

        Returns:
            UsageMetrics object.
        """
        params = {"period": period} if period else {}
        response = self._client._get("/api/v1/billing/usage", params=params)
        return UsageMetrics(**response)

    async def get_usage_async(self, period: Optional[str] = None) -> UsageMetrics:
        """Async version of get_usage()."""
        params = {"period": period} if period else {}
        response = await self._client._get_async("/api/v1/billing/usage", params=params)
        return UsageMetrics(**response)

    def get_usage_forecast(self) -> UsageForecast:
        """
        Get projected usage and cost forecast.

        Returns:
            UsageForecast object.
        """
        response = self._client._get("/api/v1/billing/usage/forecast")
        return UsageForecast(**response)

    async def get_usage_forecast_async(self) -> UsageForecast:
        """Async version of get_usage_forecast()."""
        response = await self._client._get_async("/api/v1/billing/usage/forecast")
        return UsageForecast(**response)

    # -------------------------------------------------------------------------
    # Invoices
    # -------------------------------------------------------------------------

    def list_invoices(self, limit: int = 10, offset: int = 0) -> tuple[list[Invoice], int]:
        """
        List invoices.

        Args:
            limit: Maximum number of invoices to return.
            offset: Offset for pagination.

        Returns:
            Tuple of (list of Invoice objects, total count).
        """
        response = self._client._get(
            "/api/v1/billing/invoices", params={"limit": limit, "offset": offset}
        )
        invoices = [Invoice(**i) for i in response.get("invoices", [])]
        return invoices, response.get("total", len(invoices))

    async def list_invoices_async(
        self, limit: int = 10, offset: int = 0
    ) -> tuple[list[Invoice], int]:
        """Async version of list_invoices()."""
        response = await self._client._get_async(
            "/api/v1/billing/invoices", params={"limit": limit, "offset": offset}
        )
        invoices = [Invoice(**i) for i in response.get("invoices", [])]
        return invoices, response.get("total", len(invoices))

    def get_invoice(self, invoice_id: str) -> Invoice:
        """
        Get a specific invoice.

        Args:
            invoice_id: The invoice ID.

        Returns:
            Invoice object.
        """
        response = self._client._get(f"/api/v1/billing/invoices/{invoice_id}")
        return Invoice(**response)

    async def get_invoice_async(self, invoice_id: str) -> Invoice:
        """Async version of get_invoice()."""
        response = await self._client._get_async(f"/api/v1/billing/invoices/{invoice_id}")
        return Invoice(**response)

    # -------------------------------------------------------------------------
    # Payment Methods
    # -------------------------------------------------------------------------

    def list_payment_methods(self) -> list[dict[str, Any]]:
        """
        List saved payment methods.

        Returns:
            List of payment method dictionaries.
        """
        response = self._client._get("/api/v1/billing/payment-methods")
        return response.get("payment_methods", [])

    async def list_payment_methods_async(self) -> list[dict[str, Any]]:
        """Async version of list_payment_methods()."""
        response = await self._client._get_async("/api/v1/billing/payment-methods")
        return response.get("payment_methods", [])

    def add_payment_method(self, token: str, set_default: bool = True) -> dict[str, Any]:
        """
        Add a new payment method.

        Args:
            token: Payment provider token (e.g., Stripe token).
            set_default: Whether to set this as the default payment method.

        Returns:
            Payment method details.
        """
        return self._client._post(
            "/api/v1/billing/payment-methods",
            json={"token": token, "set_default": set_default},
        )

    async def add_payment_method_async(
        self, token: str, set_default: bool = True
    ) -> dict[str, Any]:
        """Async version of add_payment_method()."""
        return await self._client._post_async(
            "/api/v1/billing/payment-methods",
            json={"token": token, "set_default": set_default},
        )

    def remove_payment_method(self, payment_method_id: str) -> None:
        """
        Remove a payment method.

        Args:
            payment_method_id: The payment method ID to remove.
        """
        self._client._delete(f"/api/v1/billing/payment-methods/{payment_method_id}")

    async def remove_payment_method_async(self, payment_method_id: str) -> None:
        """Async version of remove_payment_method()."""
        await self._client._delete_async(f"/api/v1/billing/payment-methods/{payment_method_id}")
