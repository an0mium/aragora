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

    def charge(
        self,
        amount: float,
        currency: str = "USD",
        description: str | None = None,
        customer_id: str | None = None,
        payment_method: str | dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        provider: PaymentProvider = "stripe",
    ) -> dict[str, Any]:
        """
        Process a payment charge.

        Args:
            amount: Payment amount.
            currency: Currency code (default: USD).
            description: Payment description.
            customer_id: Customer ID for saved payment methods.
            payment_method: Payment method token or details.
            metadata: Additional metadata.
            provider: Payment provider (stripe or authorize_net).

        Returns:
            Transaction result.
        """
        data: dict[str, Any] = {
            "amount": amount,
            "currency": currency,
            "provider": provider,
        }
        if description:
            data["description"] = description
        if customer_id:
            data["customer_id"] = customer_id
        if payment_method:
            data["payment_method"] = payment_method
        if metadata:
            data["metadata"] = metadata

        return self._client.request("POST", "/api/payments/charge", json=data)

    def authorize(
        self,
        amount: float,
        currency: str = "USD",
        description: str | None = None,
        customer_id: str | None = None,
        payment_method: str | dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        provider: PaymentProvider = "stripe",
        capture: bool = False,
    ) -> dict[str, Any]:
        """
        Authorize a payment for later capture.

        Args:
            amount: Payment amount.
            currency: Currency code.
            description: Payment description.
            customer_id: Customer ID.
            payment_method: Payment method token or details.
            metadata: Additional metadata.
            provider: Payment provider.
            capture: Whether to capture immediately (default: false).

        Returns:
            Authorization result with transaction ID.
        """
        data: dict[str, Any] = {
            "amount": amount,
            "currency": currency,
            "provider": provider,
            "capture": capture,
        }
        if description:
            data["description"] = description
        if customer_id:
            data["customer_id"] = customer_id
        if payment_method:
            data["payment_method"] = payment_method
        if metadata:
            data["metadata"] = metadata

        return self._client.request("POST", "/api/payments/authorize", json=data)

    def capture(
        self,
        transaction_id: str,
        amount: float | None = None,
        provider: PaymentProvider | None = None,
    ) -> dict[str, Any]:
        """
        Capture a previously authorized payment.

        Args:
            transaction_id: Transaction to capture.
            amount: Capture amount (defaults to full authorization).
            provider: Payment provider.

        Returns:
            Capture result.
        """
        data: dict[str, Any] = {"transaction_id": transaction_id}
        if amount is not None:
            data["amount"] = amount
        if provider:
            data["provider"] = provider

        return self._client.request("POST", "/api/payments/capture", json=data)

    def refund(
        self,
        transaction_id: str,
        amount: float | None = None,
        reason: str | None = None,
        provider: PaymentProvider | None = None,
        card_last_four: str | None = None,
    ) -> dict[str, Any]:
        """
        Refund a payment.

        Args:
            transaction_id: Transaction to refund.
            amount: Refund amount (defaults to full amount).
            reason: Reason for refund.
            provider: Payment provider.
            card_last_four: Last 4 digits of card for verification.

        Returns:
            Refund result.
        """
        data: dict[str, Any] = {"transaction_id": transaction_id}
        if amount is not None:
            data["amount"] = amount
        if reason:
            data["reason"] = reason
        if provider:
            data["provider"] = provider
        if card_last_four:
            data["card_last_four"] = card_last_four

        return self._client.request("POST", "/api/payments/refund", json=data)

    def void(
        self,
        transaction_id: str,
        provider: PaymentProvider | None = None,
    ) -> dict[str, Any]:
        """
        Void a transaction.

        Args:
            transaction_id: Transaction to void.
            provider: Payment provider.

        Returns:
            Void result.
        """
        data: dict[str, Any] = {"transaction_id": transaction_id}
        if provider:
            data["provider"] = provider

        return self._client.request("POST", "/api/payments/void", json=data)

    def get_transaction(self, transaction_id: str) -> dict[str, Any]:
        """
        Get transaction details.

        Args:
            transaction_id: Transaction identifier.

        Returns:
            Transaction details.
        """
        return self._client.request("GET", f"/api/payments/transaction/{transaction_id}")

    # =========================================================================
    # Customer Management
    # =========================================================================

    def create_customer(
        self,
        email: str,
        name: str | None = None,
        phone: str | None = None,
        payment_method: str | dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        provider: PaymentProvider = "stripe",
    ) -> dict[str, Any]:
        """
        Create a customer profile.

        Args:
            email: Customer email.
            name: Customer name.
            phone: Customer phone.
            payment_method: Payment method token or details.
            metadata: Additional metadata.
            provider: Payment provider.

        Returns:
            Created customer profile.
        """
        data: dict[str, Any] = {"email": email, "provider": provider}
        if name:
            data["name"] = name
        if phone:
            data["phone"] = phone
        if payment_method:
            data["payment_method"] = payment_method
        if metadata:
            data["metadata"] = metadata

        return self._client.request("POST", "/api/payments/customer", json=data)

    def get_customer(self, customer_id: str) -> dict[str, Any]:
        """
        Get a customer profile.

        Args:
            customer_id: Customer identifier.

        Returns:
            Customer profile.
        """
        return self._client.request("GET", f"/api/payments/customer/{customer_id}")

    def update_customer(
        self,
        customer_id: str,
        email: str | None = None,
        name: str | None = None,
        phone: str | None = None,
        default_payment_method: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Update a customer profile.

        Args:
            customer_id: Customer identifier.
            email: New email.
            name: New name.
            phone: New phone.
            default_payment_method: Default payment method ID.
            metadata: Updated metadata.

        Returns:
            Updated customer profile.
        """
        data: dict[str, Any] = {}
        if email:
            data["email"] = email
        if name:
            data["name"] = name
        if phone:
            data["phone"] = phone
        if default_payment_method:
            data["default_payment_method"] = default_payment_method
        if metadata:
            data["metadata"] = metadata

        return self._client.request("PUT", f"/api/payments/customer/{customer_id}", json=data)

    def delete_customer(self, customer_id: str) -> dict[str, Any]:
        """
        Delete a customer profile.

        Args:
            customer_id: Customer identifier.

        Returns:
            Success status.
        """
        return self._client.request("DELETE", f"/api/payments/customer/{customer_id}")

    # =========================================================================
    # Subscription Management
    # =========================================================================

    def create_subscription(
        self,
        customer_id: str,
        name: str,
        amount: float,
        interval: SubscriptionInterval,
        currency: str = "USD",
        interval_count: int = 1,
        price_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        provider: PaymentProvider = "stripe",
    ) -> dict[str, Any]:
        """
        Create a subscription.

        Args:
            customer_id: Customer identifier.
            name: Subscription name.
            amount: Subscription amount.
            interval: Billing interval.
            currency: Currency code.
            interval_count: Number of intervals between billings.
            price_id: Price ID (for Stripe).
            metadata: Additional metadata.
            provider: Payment provider.

        Returns:
            Created subscription.
        """
        data: dict[str, Any] = {
            "customer_id": customer_id,
            "name": name,
            "amount": amount,
            "interval": interval,
            "currency": currency,
            "interval_count": interval_count,
            "provider": provider,
        }
        if price_id:
            data["price_id"] = price_id
        if metadata:
            data["metadata"] = metadata

        return self._client.request("POST", "/api/payments/subscription", json=data)

    def get_subscription(self, subscription_id: str) -> dict[str, Any]:
        """
        Get a subscription.

        Args:
            subscription_id: Subscription identifier.

        Returns:
            Subscription details.
        """
        return self._client.request("GET", f"/api/payments/subscription/{subscription_id}")

    def update_subscription(
        self,
        subscription_id: str,
        name: str | None = None,
        amount: float | None = None,
        cancel_at_period_end: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Update a subscription.

        Args:
            subscription_id: Subscription identifier.
            name: New subscription name.
            amount: New amount.
            cancel_at_period_end: Cancel at period end.
            metadata: Updated metadata.

        Returns:
            Updated subscription.
        """
        data: dict[str, Any] = {}
        if name:
            data["name"] = name
        if amount is not None:
            data["amount"] = amount
        if cancel_at_period_end is not None:
            data["cancel_at_period_end"] = cancel_at_period_end
        if metadata:
            data["metadata"] = metadata

        return self._client.request(
            "PUT", f"/api/payments/subscription/{subscription_id}", json=data
        )

    def cancel_subscription(
        self,
        subscription_id: str,
        cancel_at_period_end: bool = True,
    ) -> dict[str, Any]:
        """
        Cancel a subscription.

        Args:
            subscription_id: Subscription identifier.
            cancel_at_period_end: If true, cancels at period end.

        Returns:
            Cancelled subscription.
        """
        return self._client.request(
            "DELETE",
            f"/api/payments/subscription/{subscription_id}",
            json={"cancel_at_period_end": cancel_at_period_end},
        )


class AsyncPaymentsAPI:
    """Asynchronous Payments API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # =========================================================================
    # Payment Operations
    # =========================================================================

    async def charge(
        self,
        amount: float,
        currency: str = "USD",
        description: str | None = None,
        customer_id: str | None = None,
        payment_method: str | dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        provider: PaymentProvider = "stripe",
    ) -> dict[str, Any]:
        """Process a payment charge."""
        data: dict[str, Any] = {
            "amount": amount,
            "currency": currency,
            "provider": provider,
        }
        if description:
            data["description"] = description
        if customer_id:
            data["customer_id"] = customer_id
        if payment_method:
            data["payment_method"] = payment_method
        if metadata:
            data["metadata"] = metadata

        return await self._client.request("POST", "/api/payments/charge", json=data)

    async def authorize(
        self,
        amount: float,
        currency: str = "USD",
        description: str | None = None,
        customer_id: str | None = None,
        payment_method: str | dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        provider: PaymentProvider = "stripe",
        capture: bool = False,
    ) -> dict[str, Any]:
        """Authorize a payment for later capture."""
        data: dict[str, Any] = {
            "amount": amount,
            "currency": currency,
            "provider": provider,
            "capture": capture,
        }
        if description:
            data["description"] = description
        if customer_id:
            data["customer_id"] = customer_id
        if payment_method:
            data["payment_method"] = payment_method
        if metadata:
            data["metadata"] = metadata

        return await self._client.request("POST", "/api/payments/authorize", json=data)

    async def capture(
        self,
        transaction_id: str,
        amount: float | None = None,
        provider: PaymentProvider | None = None,
    ) -> dict[str, Any]:
        """Capture a previously authorized payment."""
        data: dict[str, Any] = {"transaction_id": transaction_id}
        if amount is not None:
            data["amount"] = amount
        if provider:
            data["provider"] = provider

        return await self._client.request("POST", "/api/payments/capture", json=data)

    async def refund(
        self,
        transaction_id: str,
        amount: float | None = None,
        reason: str | None = None,
        provider: PaymentProvider | None = None,
        card_last_four: str | None = None,
    ) -> dict[str, Any]:
        """Refund a payment."""
        data: dict[str, Any] = {"transaction_id": transaction_id}
        if amount is not None:
            data["amount"] = amount
        if reason:
            data["reason"] = reason
        if provider:
            data["provider"] = provider
        if card_last_four:
            data["card_last_four"] = card_last_four

        return await self._client.request("POST", "/api/payments/refund", json=data)

    async def void(
        self,
        transaction_id: str,
        provider: PaymentProvider | None = None,
    ) -> dict[str, Any]:
        """Void a transaction."""
        data: dict[str, Any] = {"transaction_id": transaction_id}
        if provider:
            data["provider"] = provider

        return await self._client.request("POST", "/api/payments/void", json=data)

    async def get_transaction(self, transaction_id: str) -> dict[str, Any]:
        """Get transaction details."""
        return await self._client.request("GET", f"/api/payments/transaction/{transaction_id}")

    # =========================================================================
    # Customer Management
    # =========================================================================

    async def create_customer(
        self,
        email: str,
        name: str | None = None,
        phone: str | None = None,
        payment_method: str | dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        provider: PaymentProvider = "stripe",
    ) -> dict[str, Any]:
        """Create a customer profile."""
        data: dict[str, Any] = {"email": email, "provider": provider}
        if name:
            data["name"] = name
        if phone:
            data["phone"] = phone
        if payment_method:
            data["payment_method"] = payment_method
        if metadata:
            data["metadata"] = metadata

        return await self._client.request("POST", "/api/payments/customer", json=data)

    async def get_customer(self, customer_id: str) -> dict[str, Any]:
        """Get a customer profile."""
        return await self._client.request("GET", f"/api/payments/customer/{customer_id}")

    async def update_customer(
        self,
        customer_id: str,
        email: str | None = None,
        name: str | None = None,
        phone: str | None = None,
        default_payment_method: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update a customer profile."""
        data: dict[str, Any] = {}
        if email:
            data["email"] = email
        if name:
            data["name"] = name
        if phone:
            data["phone"] = phone
        if default_payment_method:
            data["default_payment_method"] = default_payment_method
        if metadata:
            data["metadata"] = metadata

        return await self._client.request("PUT", f"/api/payments/customer/{customer_id}", json=data)

    async def delete_customer(self, customer_id: str) -> dict[str, Any]:
        """Delete a customer profile."""
        return await self._client.request("DELETE", f"/api/payments/customer/{customer_id}")

    # =========================================================================
    # Subscription Management
    # =========================================================================

    async def create_subscription(
        self,
        customer_id: str,
        name: str,
        amount: float,
        interval: SubscriptionInterval,
        currency: str = "USD",
        interval_count: int = 1,
        price_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        provider: PaymentProvider = "stripe",
    ) -> dict[str, Any]:
        """Create a subscription."""
        data: dict[str, Any] = {
            "customer_id": customer_id,
            "name": name,
            "amount": amount,
            "interval": interval,
            "currency": currency,
            "interval_count": interval_count,
            "provider": provider,
        }
        if price_id:
            data["price_id"] = price_id
        if metadata:
            data["metadata"] = metadata

        return await self._client.request("POST", "/api/payments/subscription", json=data)

    async def get_subscription(self, subscription_id: str) -> dict[str, Any]:
        """Get a subscription."""
        return await self._client.request("GET", f"/api/payments/subscription/{subscription_id}")

    async def update_subscription(
        self,
        subscription_id: str,
        name: str | None = None,
        amount: float | None = None,
        cancel_at_period_end: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update a subscription."""
        data: dict[str, Any] = {}
        if name:
            data["name"] = name
        if amount is not None:
            data["amount"] = amount
        if cancel_at_period_end is not None:
            data["cancel_at_period_end"] = cancel_at_period_end
        if metadata:
            data["metadata"] = metadata

        return await self._client.request(
            "PUT", f"/api/payments/subscription/{subscription_id}", json=data
        )

    async def cancel_subscription(
        self,
        subscription_id: str,
        cancel_at_period_end: bool = True,
    ) -> dict[str, Any]:
        """Cancel a subscription."""
        return await self._client.request(
            "DELETE",
            f"/api/payments/subscription/{subscription_id}",
            json={"cancel_at_period_end": cancel_at_period_end},
        )
