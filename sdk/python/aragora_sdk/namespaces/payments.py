"""
Payments Namespace API.

Provides payment processing capabilities for Stripe and Authorize.net:
- Payment processing (charge, authorize, capture, refund)
- Customer profile management
- Subscription management
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient

PaymentProvider = Literal["stripe", "authorize_net"]
PaymentStatus = Literal["pending", "approved", "declined", "error", "void", "refunded"]
SubscriptionInterval = Literal["day", "week", "month", "year"]
SubscriptionStatus = Literal["active", "paused", "cancelled", "past_due"]


class PaymentsAPI:
    """
    Synchronous Payments API.

    Provides methods for:
    - Payment processing (charge, authorize, capture, refund)
    - Customer profile management
    - Subscription management
    """

    def __init__(self, client: AragoraClient) -> None:
        self._client = client

    # =========================================================================
    # Payment Operations
    # =========================================================================

    def charge(self, **kwargs: Any) -> dict[str, Any]:
        """Process a payment charge."""
        return self._client.request("POST", "/api/v1/payments/charge", json=kwargs)

    def authorize(self, **kwargs: Any) -> dict[str, Any]:
        """Authorize a payment (hold funds)."""
        return self._client.request("POST", "/api/v1/payments/authorize", json=kwargs)

    def capture(self, **kwargs: Any) -> dict[str, Any]:
        """Capture an authorized payment."""
        return self._client.request("POST", "/api/v1/payments/capture", json=kwargs)

    def refund(self, **kwargs: Any) -> dict[str, Any]:
        """Refund a payment."""
        return self._client.request("POST", "/api/v1/payments/refund", json=kwargs)

    def void(self, **kwargs: Any) -> dict[str, Any]:
        """Void a payment."""
        return self._client.request("POST", "/api/v1/payments/void", json=kwargs)

    def get_transaction(self, transaction_id: str) -> dict[str, Any]:
        """Get a transaction by ID."""
        return self._client.request("GET", f"/api/v1/payments/transaction/{transaction_id}")

    # =========================================================================
    # Customer Management
    # =========================================================================

    def create_customer(self, **kwargs: Any) -> dict[str, Any]:
        """Create a customer profile."""
        return self._client.request("POST", "/api/v1/payments/customer", json=kwargs)

    def get_customer(self, customer_id: str) -> dict[str, Any]:
        """Get a customer profile."""
        return self._client.request("GET", f"/api/v1/payments/customer/{customer_id}")

    def update_customer(self, customer_id: str, **kwargs: Any) -> dict[str, Any]:
        """Update a customer profile."""
        return self._client.request("PUT", f"/api/v1/payments/customer/{customer_id}", json=kwargs)

    def delete_customer(self, customer_id: str) -> dict[str, Any]:
        """Delete a customer profile."""
        return self._client.request("DELETE", f"/api/v1/payments/customer/{customer_id}")

    # =========================================================================
    # Subscription Management
    # =========================================================================

    def create_subscription(self, **kwargs: Any) -> dict[str, Any]:
        """Create a subscription."""
        return self._client.request("POST", "/api/v1/payments/subscription", json=kwargs)

    def get_subscription(self, subscription_id: str) -> dict[str, Any]:
        """Get a subscription."""
        return self._client.request("GET", f"/api/v1/payments/subscription/{subscription_id}")

    def update_subscription(self, subscription_id: str, **kwargs: Any) -> dict[str, Any]:
        """Update a subscription."""
        return self._client.request(
            "PUT", f"/api/v1/payments/subscription/{subscription_id}", json=kwargs
        )

    def cancel_subscription(self, subscription_id: str) -> dict[str, Any]:
        """Cancel a subscription."""
        return self._client.request("DELETE", f"/api/v1/payments/subscription/{subscription_id}")


class AsyncPaymentsAPI:
    """Asynchronous Payments API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    async def charge(self, **kwargs: Any) -> dict[str, Any]:
        """Process a payment charge."""
        return await self._client.request("POST", "/api/v1/payments/charge", json=kwargs)

    async def authorize(self, **kwargs: Any) -> dict[str, Any]:
        """Authorize a payment."""
        return await self._client.request("POST", "/api/v1/payments/authorize", json=kwargs)

    async def capture(self, **kwargs: Any) -> dict[str, Any]:
        """Capture an authorized payment."""
        return await self._client.request("POST", "/api/v1/payments/capture", json=kwargs)

    async def refund(self, **kwargs: Any) -> dict[str, Any]:
        """Refund a payment."""
        return await self._client.request("POST", "/api/v1/payments/refund", json=kwargs)

    async def void(self, **kwargs: Any) -> dict[str, Any]:
        """Void a payment."""
        return await self._client.request("POST", "/api/v1/payments/void", json=kwargs)

    async def get_transaction(self, transaction_id: str) -> dict[str, Any]:
        """Get a transaction by ID."""
        return await self._client.request("GET", f"/api/v1/payments/transaction/{transaction_id}")

    async def create_customer(self, **kwargs: Any) -> dict[str, Any]:
        """Create a customer profile."""
        return await self._client.request("POST", "/api/v1/payments/customer", json=kwargs)

    async def get_customer(self, customer_id: str) -> dict[str, Any]:
        """Get a customer profile."""
        return await self._client.request("GET", f"/api/v1/payments/customer/{customer_id}")

    async def update_customer(self, customer_id: str, **kwargs: Any) -> dict[str, Any]:
        """Update a customer profile."""
        return await self._client.request(
            "PUT", f"/api/v1/payments/customer/{customer_id}", json=kwargs
        )

    async def delete_customer(self, customer_id: str) -> dict[str, Any]:
        """Delete a customer profile."""
        return await self._client.request("DELETE", f"/api/v1/payments/customer/{customer_id}")

    async def create_subscription(self, **kwargs: Any) -> dict[str, Any]:
        """Create a subscription."""
        return await self._client.request("POST", "/api/v1/payments/subscription", json=kwargs)

    async def get_subscription(self, subscription_id: str) -> dict[str, Any]:
        """Get a subscription."""
        return await self._client.request("GET", f"/api/v1/payments/subscription/{subscription_id}")

    async def update_subscription(self, subscription_id: str, **kwargs: Any) -> dict[str, Any]:
        """Update a subscription."""
        return await self._client.request(
            "PUT", f"/api/v1/payments/subscription/{subscription_id}", json=kwargs
        )

    async def cancel_subscription(self, subscription_id: str) -> dict[str, Any]:
        """Cancel a subscription."""
        return await self._client.request(
            "DELETE", f"/api/v1/payments/subscription/{subscription_id}"
        )
