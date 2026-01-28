"""
Billing Namespace API

Provides methods for billing and subscription management:
- Subscription plans and pricing
- Usage tracking and forecasting
- Checkout and billing portal
- Invoice history
- Audit logs
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class BillingAPI:
    """
    Synchronous Billing API.

    Provides methods for subscription and billing management:
    - Get available plans
    - Track usage and costs
    - Manage subscriptions
    - Access invoices and audit logs

    Example:
        >>> client = AragoraClient(base_url="https://api.aragora.ai")
        >>> plans = client.billing.get_plans()
        >>> usage = client.billing.get_usage()
    """

    def __init__(self, client: AragoraClient):
        self._client = client

    # ===========================================================================
    # Plans and Pricing
    # ===========================================================================

    def get_plans(self) -> dict[str, Any]:
        """
        Get available subscription plans.

        Returns:
            Dict with plans array containing tier info and pricing
        """
        return self._client.request("GET", "/api/v1/billing/plans")

    # ===========================================================================
    # Usage
    # ===========================================================================

    def get_usage(self) -> dict[str, Any]:
        """
        Get current usage for authenticated user.

        Returns:
            Dict with usage info (debates, tokens, costs)
        """
        return self._client.request("GET", "/api/v1/billing/usage")

    def export_usage_csv(
        self,
        start: str | None = None,
        end: str | None = None,
    ) -> bytes:
        """
        Export usage data as CSV.

        Args:
            start: Start date (ISO 8601)
            end: End date (ISO 8601)

        Returns:
            CSV file content as bytes
        """
        params: dict[str, Any] = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        return self._client.request("GET", "/api/v1/billing/usage/export", params=params)

    def get_usage_forecast(self) -> dict[str, Any]:
        """
        Get usage forecast and cost projection.

        Returns:
            Dict with forecast info (projections, recommendations)
        """
        return self._client.request("GET", "/api/v1/billing/usage/forecast")

    # ===========================================================================
    # Subscription
    # ===========================================================================

    def get_subscription(self) -> dict[str, Any]:
        """
        Get current subscription status.

        Returns:
            Dict with subscription info (tier, status, limits)
        """
        return self._client.request("GET", "/api/v1/billing/subscription")

    def create_checkout(
        self,
        tier: str,
        success_url: str,
        cancel_url: str,
    ) -> dict[str, Any]:
        """
        Create a Stripe checkout session.

        Args:
            tier: Subscription tier (starter, professional, enterprise)
            success_url: URL to redirect on success
            cancel_url: URL to redirect on cancel

        Returns:
            Dict with checkout session info
        """
        return self._client.request(
            "POST",
            "/api/v1/billing/checkout",
            json={
                "tier": tier,
                "success_url": success_url,
                "cancel_url": cancel_url,
            },
        )

    def create_portal(self, return_url: str) -> dict[str, Any]:
        """
        Create a Stripe billing portal session.

        Args:
            return_url: URL to return to after portal session

        Returns:
            Dict with portal session info
        """
        return self._client.request(
            "POST",
            "/api/v1/billing/portal",
            json={"return_url": return_url},
        )

    def cancel_subscription(self) -> dict[str, Any]:
        """
        Cancel subscription at end of billing period.

        Returns:
            Dict with cancellation confirmation
        """
        return self._client.request("POST", "/api/v1/billing/cancel")

    def resume_subscription(self) -> dict[str, Any]:
        """
        Resume a canceled subscription.

        Returns:
            Dict with subscription info
        """
        return self._client.request("POST", "/api/v1/billing/resume")

    # ===========================================================================
    # Invoices
    # ===========================================================================

    def get_invoices(self, limit: int = 10) -> dict[str, Any]:
        """
        Get invoice history.

        Args:
            limit: Maximum invoices to return (default: 10, max: 100)

        Returns:
            Dict with invoices array
        """
        return self._client.request("GET", "/api/v1/billing/invoices", params={"limit": limit})

    # ===========================================================================
    # Audit Log
    # ===========================================================================

    def get_audit_log(
        self,
        limit: int = 50,
        offset: int = 0,
        action: str | None = None,
    ) -> dict[str, Any]:
        """
        Get billing audit log (Enterprise feature).

        Args:
            limit: Maximum entries (default: 50, max: 100)
            offset: Pagination offset
            action: Filter by action type

        Returns:
            Dict with audit entries and total count
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if action:
            params["action"] = action
        return self._client.request("GET", "/api/v1/billing/audit-log", params=params)


class AsyncBillingAPI:
    """
    Asynchronous Billing API.

    Example:
        >>> async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
        ...     plans = await client.billing.get_plans()
        ...     usage = await client.billing.get_usage()
    """

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    # ===========================================================================
    # Plans and Pricing
    # ===========================================================================

    async def get_plans(self) -> dict[str, Any]:
        """Get available subscription plans."""
        return await self._client.request("GET", "/api/v1/billing/plans")

    # ===========================================================================
    # Usage
    # ===========================================================================

    async def get_usage(self) -> dict[str, Any]:
        """Get current usage."""
        return await self._client.request("GET", "/api/v1/billing/usage")

    async def export_usage_csv(
        self,
        start: str | None = None,
        end: str | None = None,
    ) -> bytes:
        """Export usage data as CSV."""
        params: dict[str, Any] = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        return await self._client.request("GET", "/api/v1/billing/usage/export", params=params)

    async def get_usage_forecast(self) -> dict[str, Any]:
        """Get usage forecast."""
        return await self._client.request("GET", "/api/v1/billing/usage/forecast")

    # ===========================================================================
    # Subscription
    # ===========================================================================

    async def get_subscription(self) -> dict[str, Any]:
        """Get current subscription."""
        return await self._client.request("GET", "/api/v1/billing/subscription")

    async def create_checkout(
        self,
        tier: str,
        success_url: str,
        cancel_url: str,
    ) -> dict[str, Any]:
        """Create checkout session."""
        return await self._client.request(
            "POST",
            "/api/v1/billing/checkout",
            json={
                "tier": tier,
                "success_url": success_url,
                "cancel_url": cancel_url,
            },
        )

    async def create_portal(self, return_url: str) -> dict[str, Any]:
        """Create billing portal session."""
        return await self._client.request(
            "POST",
            "/api/v1/billing/portal",
            json={"return_url": return_url},
        )

    async def cancel_subscription(self) -> dict[str, Any]:
        """Cancel subscription."""
        return await self._client.request("POST", "/api/v1/billing/cancel")

    async def resume_subscription(self) -> dict[str, Any]:
        """Resume subscription."""
        return await self._client.request("POST", "/api/v1/billing/resume")

    # ===========================================================================
    # Invoices
    # ===========================================================================

    async def get_invoices(self, limit: int = 10) -> dict[str, Any]:
        """Get invoice history."""
        return await self._client.request(
            "GET", "/api/v1/billing/invoices", params={"limit": limit}
        )

    # ===========================================================================
    # Audit Log
    # ===========================================================================

    async def get_audit_log(
        self,
        limit: int = 50,
        offset: int = 0,
        action: str | None = None,
    ) -> dict[str, Any]:
        """Get billing audit log."""
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if action:
            params["action"] = action
        return await self._client.request("GET", "/api/v1/billing/audit-log", params=params)
