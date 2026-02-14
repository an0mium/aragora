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

class AsyncPaymentsAPI:
    """Asynchronous Payments API."""

    def __init__(self, client: AragoraAsyncClient) -> None:
        self._client = client

    # =========================================================================
    # Payment Operations

