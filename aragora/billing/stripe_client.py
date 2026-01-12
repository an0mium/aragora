"""
Stripe Integration for Aragora Billing.

Provides Stripe API client, checkout sessions, webhooks, and subscription management.

Environment Variables:
    STRIPE_SECRET_KEY: Stripe secret API key (sk_test_xxx or sk_live_xxx)
    STRIPE_WEBHOOK_SECRET: Webhook endpoint signing secret (whsec_xxx)
    STRIPE_PRICE_STARTER: Stripe price ID for Starter tier
    STRIPE_PRICE_PROFESSIONAL: Stripe price ID for Professional tier
    STRIPE_PRICE_ENTERPRISE: Stripe price ID for Enterprise tier
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from aragora.billing.models import SubscriptionTier

logger = logging.getLogger(__name__)

# Stripe API configuration
STRIPE_API_VERSION = "2023-10-16"
STRIPE_API_BASE = "https://api.stripe.com/v1"

# Environment configuration
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

# Price IDs for each tier (configure in Stripe Dashboard)
STRIPE_PRICES = {
    SubscriptionTier.STARTER: os.environ.get("STRIPE_PRICE_STARTER", ""),
    SubscriptionTier.PROFESSIONAL: os.environ.get("STRIPE_PRICE_PROFESSIONAL", ""),
    SubscriptionTier.ENTERPRISE: os.environ.get("STRIPE_PRICE_ENTERPRISE", ""),
}


class StripeError(Exception):
    """Base exception for Stripe API errors."""

    def __init__(self, message: str, code: str = "", status: int = 0):
        super().__init__(message)
        self.code = code
        self.status = status


class StripeConfigError(StripeError):
    """Stripe configuration is missing or invalid."""

    pass


class StripeAPIError(StripeError):
    """Stripe API returned an error."""

    pass


@dataclass
class StripeCustomer:
    """Stripe customer data."""

    id: str
    email: str
    name: Optional[str] = None
    metadata: dict = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "metadata": self.metadata or {},
        }


@dataclass
class StripeSubscription:
    """Stripe subscription data."""

    id: str
    customer_id: str
    status: str
    price_id: str
    current_period_start: datetime
    current_period_end: datetime
    cancel_at_period_end: bool = False
    trial_start: Optional[datetime] = None
    trial_end: Optional[datetime] = None

    @property
    def is_trialing(self) -> bool:
        """Check if subscription is in trial period."""
        if self.status != "trialing":
            return False
        if self.trial_end is None:
            return False
        return datetime.utcnow() < self.trial_end

    def to_dict(self) -> dict[str, Any]:
        result = {
            "id": self.id,
            "customer_id": self.customer_id,
            "status": self.status,
            "price_id": self.price_id,
            "current_period_start": self.current_period_start.isoformat(),
            "current_period_end": self.current_period_end.isoformat(),
            "cancel_at_period_end": self.cancel_at_period_end,
        }
        if self.trial_start:
            result["trial_start"] = self.trial_start.isoformat()
        if self.trial_end:
            result["trial_end"] = self.trial_end.isoformat()
        result["is_trialing"] = self.is_trialing
        return result


@dataclass
class CheckoutSession:
    """Stripe checkout session data."""

    id: str
    url: str
    customer_id: Optional[str] = None
    subscription_id: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "url": self.url,
            "customer_id": self.customer_id,
            "subscription_id": self.subscription_id,
        }


@dataclass
class BillingPortalSession:
    """Stripe billing portal session data."""

    id: str
    url: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "url": self.url}


class StripeClient:
    """
    Stripe API client.

    Handles all Stripe API interactions for subscriptions and billing.
    Uses urllib to avoid external dependencies.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Stripe client.

        Args:
            api_key: Stripe secret key (defaults to STRIPE_SECRET_KEY env var)
        """
        self.api_key = api_key or STRIPE_SECRET_KEY
        if not self.api_key:
            logger.warning(
                "Stripe API key not configured. "
                "Set STRIPE_SECRET_KEY environment variable."
            )

    def _is_configured(self) -> bool:
        """Check if Stripe is properly configured."""
        return bool(self.api_key)

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[dict] = None,
        idempotency_key: Optional[str] = None,
    ) -> dict:
        """
        Make a request to the Stripe API.

        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint (e.g., /customers)
            data: Request body data
            idempotency_key: Optional idempotency key for POST requests

        Returns:
            Response JSON as dict

        Raises:
            StripeConfigError: If not configured
            StripeAPIError: If API returns an error
        """
        if not self._is_configured():
            raise StripeConfigError("Stripe API key not configured")

        url = f"{STRIPE_API_BASE}{endpoint}"

        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Stripe-Version": STRIPE_API_VERSION,
        }

        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key

        # Encode data as form data (Stripe API format)
        body = None
        if data:
            body = self._encode_form_data(data).encode("utf-8")

        req = Request(url, data=body, headers=headers, method=method)

        try:
            with urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            error_body = e.read().decode("utf-8")
            try:
                error_data = json.loads(error_body)
                error_msg = error_data.get("error", {}).get("message", str(e))
                error_code = error_data.get("error", {}).get("code", "")
            except json.JSONDecodeError:
                error_msg = str(e)
                error_code = ""
            raise StripeAPIError(error_msg, error_code, e.code)
        except URLError as e:
            raise StripeAPIError(f"Connection error: {e}")

    def _encode_form_data(self, data: dict, prefix: str = "") -> str:
        """
        Encode nested dict as form data for Stripe API.

        Stripe uses bracket notation for nested objects:
        {"metadata": {"user_id": "123"}} -> "metadata[user_id]=123"
        """
        pairs = []
        for key, value in data.items():
            full_key = f"{prefix}[{key}]" if prefix else key

            if isinstance(value, dict):
                pairs.append(self._encode_form_data(value, full_key))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        pairs.append(self._encode_form_data(item, f"{full_key}[{i}]"))
                    else:
                        pairs.append(f"{full_key}[{i}]={item}")
            elif value is not None:
                pairs.append(f"{full_key}={value}")

        return "&".join(p for p in pairs if p)

    # =========================================================================
    # Customer Management
    # =========================================================================

    def create_customer(
        self,
        email: str,
        name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> StripeCustomer:
        """
        Create a Stripe customer.

        Args:
            email: Customer email
            name: Customer name
            metadata: Additional metadata (e.g., user_id, org_id)

        Returns:
            StripeCustomer with the created customer data
        """
        data: dict[str, Any] = {"email": email}
        if name:
            data["name"] = name
        if metadata:
            data["metadata"] = metadata

        response = self._request("POST", "/customers", data)

        return StripeCustomer(
            id=response["id"],
            email=response.get("email", email),
            name=response.get("name"),
            metadata=response.get("metadata", {}),
        )

    def get_customer(self, customer_id: str) -> Optional[StripeCustomer]:
        """
        Get a Stripe customer by ID.

        Args:
            customer_id: Stripe customer ID

        Returns:
            StripeCustomer or None if not found
        """
        try:
            response = self._request("GET", f"/customers/{customer_id}")
            return StripeCustomer(
                id=response["id"],
                email=response.get("email", ""),
                name=response.get("name"),
                metadata=response.get("metadata", {}),
            )
        except StripeAPIError as e:
            if e.status == 404:
                return None
            raise

    def update_customer(
        self,
        customer_id: str,
        email: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> StripeCustomer:
        """Update a Stripe customer."""
        data: dict[str, Any] = {}
        if email:
            data["email"] = email
        if name:
            data["name"] = name
        if metadata:
            data["metadata"] = metadata

        response = self._request("POST", f"/customers/{customer_id}", data)

        return StripeCustomer(
            id=response["id"],
            email=response.get("email", ""),
            name=response.get("name"),
            metadata=response.get("metadata", {}),
        )

    # =========================================================================
    # Checkout Sessions
    # =========================================================================

    def create_checkout_session(
        self,
        tier: SubscriptionTier,
        customer_email: str,
        success_url: str,
        cancel_url: str,
        customer_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        trial_days: Optional[int] = None,
    ) -> CheckoutSession:
        """
        Create a Stripe Checkout session for subscription.

        Args:
            tier: Subscription tier to purchase
            customer_email: Customer's email
            success_url: URL to redirect after successful payment
            cancel_url: URL to redirect if customer cancels
            customer_id: Existing Stripe customer ID
            metadata: Additional metadata
            trial_days: Trial period in days

        Returns:
            CheckoutSession with session ID and URL
        """
        price_id = STRIPE_PRICES.get(tier)
        if not price_id:
            raise StripeConfigError(
                f"No price configured for tier {tier.value}. "
                f"Set STRIPE_PRICE_{tier.name} environment variable."
            )

        data: dict[str, Any] = {
            "mode": "subscription",
            "success_url": success_url,
            "cancel_url": cancel_url,
            "line_items": [{"price": price_id, "quantity": 1}],
        }

        if customer_id:
            data["customer"] = customer_id
        else:
            data["customer_email"] = customer_email

        subscription_data: dict[str, Any] = {}
        if metadata:
            data["metadata"] = metadata
            subscription_data["metadata"] = metadata

        if trial_days and trial_days > 0:
            subscription_data["trial_period_days"] = trial_days

        if subscription_data:
            data["subscription_data"] = subscription_data

        response = self._request("POST", "/checkout/sessions", data)

        return CheckoutSession(
            id=response["id"],
            url=response["url"],
            customer_id=response.get("customer"),
            subscription_id=response.get("subscription"),
        )

    # =========================================================================
    # Billing Portal
    # =========================================================================

    def create_portal_session(
        self,
        customer_id: str,
        return_url: str,
    ) -> BillingPortalSession:
        """
        Create a Stripe Billing Portal session.

        Allows customers to manage their subscription, payment methods,
        and billing history.

        Args:
            customer_id: Stripe customer ID
            return_url: URL to return to after portal session

        Returns:
            BillingPortalSession with session ID and URL
        """
        data = {
            "customer": customer_id,
            "return_url": return_url,
        }

        response = self._request("POST", "/billing_portal/sessions", data)

        return BillingPortalSession(
            id=response["id"],
            url=response["url"],
        )

    # =========================================================================
    # Subscription Management
    # =========================================================================

    def get_subscription(self, subscription_id: str) -> Optional[StripeSubscription]:
        """Get a subscription by ID."""
        try:
            response = self._request("GET", f"/subscriptions/{subscription_id}")
            return self._parse_subscription(response)
        except StripeAPIError as e:
            if e.status == 404:
                return None
            raise

    def cancel_subscription(
        self,
        subscription_id: str,
        at_period_end: bool = True,
    ) -> StripeSubscription:
        """
        Cancel a subscription.

        Args:
            subscription_id: Stripe subscription ID
            at_period_end: If True, cancel at end of billing period

        Returns:
            Updated StripeSubscription
        """
        if at_period_end:
            # Schedule cancellation at period end
            data = {"cancel_at_period_end": "true"}
            response = self._request(
                "POST", f"/subscriptions/{subscription_id}", data
            )
        else:
            # Cancel immediately
            response = self._request("DELETE", f"/subscriptions/{subscription_id}")

        return self._parse_subscription(response)

    def resume_subscription(self, subscription_id: str) -> StripeSubscription:
        """
        Resume a subscription scheduled for cancellation.

        Args:
            subscription_id: Stripe subscription ID

        Returns:
            Updated StripeSubscription
        """
        data = {"cancel_at_period_end": "false"}
        response = self._request("POST", f"/subscriptions/{subscription_id}", data)
        return self._parse_subscription(response)

    # =========================================================================
    # Invoices
    # =========================================================================

    def list_invoices(
        self,
        customer_id: str,
        limit: int = 10,
        starting_after: Optional[str] = None,
    ) -> list[dict]:
        """
        List invoices for a customer.

        Args:
            customer_id: Stripe customer ID
            limit: Maximum number of invoices to return
            starting_after: Pagination cursor

        Returns:
            List of invoice data dicts
        """
        params = {
            "customer": customer_id,
            "limit": limit,
        }
        if starting_after:
            params["starting_after"] = starting_after

        # Build query string
        query = "&".join(f"{k}={v}" for k, v in params.items())
        response = self._request("GET", f"/invoices?{query}")

        return response.get("data", [])

    def _parse_subscription(self, data: dict) -> StripeSubscription:
        """Parse Stripe subscription response into StripeSubscription."""
        # Get price ID from items
        price_id = ""
        items = data.get("items", {}).get("data", [])
        if items:
            price_id = items[0].get("price", {}).get("id", "")

        # Parse trial dates if present
        trial_start = None
        trial_end = None
        if data.get("trial_start"):
            trial_start = datetime.fromtimestamp(data["trial_start"])
        if data.get("trial_end"):
            trial_end = datetime.fromtimestamp(data["trial_end"])

        return StripeSubscription(
            id=data["id"],
            customer_id=data["customer"],
            status=data["status"],
            price_id=price_id,
            current_period_start=datetime.fromtimestamp(
                data.get("current_period_start", 0)
            ),
            current_period_end=datetime.fromtimestamp(
                data.get("current_period_end", 0)
            ),
            cancel_at_period_end=data.get("cancel_at_period_end", False),
            trial_start=trial_start,
            trial_end=trial_end,
        )


# =============================================================================
# Webhook Handling
# =============================================================================


class WebhookEvent:
    """Parsed Stripe webhook event."""

    def __init__(self, event_type: str, data: dict, event_id: str = ""):
        self.type = event_type
        self.data = data
        self.object = data.get("object", {})
        self.event_id = event_id  # Top-level Stripe event ID for idempotency

    @property
    def customer_id(self) -> Optional[str]:
        """Get customer ID from event."""
        return self.object.get("customer") or self.object.get("id")

    @property
    def subscription_id(self) -> Optional[str]:
        """Get subscription ID from event."""
        if self.type.startswith("customer.subscription"):
            return self.object.get("id")
        return self.object.get("subscription")

    @property
    def metadata(self) -> dict:
        """Get metadata from event object."""
        return self.object.get("metadata", {})


def verify_webhook_signature(
    payload: bytes,
    signature: str,
    secret: Optional[str] = None,
) -> bool:
    """
    Verify Stripe webhook signature.

    Args:
        payload: Raw request body
        signature: Stripe-Signature header value
        secret: Webhook signing secret (defaults to env var)

    Returns:
        True if signature is valid
    """
    secret = secret or STRIPE_WEBHOOK_SECRET
    if not secret:
        logger.warning("Webhook secret not configured")
        return False

    # Parse signature header
    # Format: t=timestamp,v1=signature,v1=signature2...
    try:
        sig_parts = dict(
            part.split("=", 1) for part in signature.split(",") if "=" in part
        )
    except ValueError:
        return False

    timestamp = sig_parts.get("t")
    signatures = [v for k, v in sig_parts.items() if k.startswith("v1")]

    if not timestamp or not signatures:
        return False

    # Check timestamp (reject if too old - 5 minute tolerance)
    try:
        ts = int(timestamp)
        if abs(time.time() - ts) > 300:
            logger.warning("Webhook timestamp too old")
            return False
    except ValueError:
        return False

    # Compute expected signature
    signed_payload = f"{timestamp}.".encode() + payload
    expected = hmac.new(
        secret.encode("utf-8"),
        signed_payload,
        hashlib.sha256,
    ).hexdigest()

    # Check if any signature matches
    return any(hmac.compare_digest(expected, sig) for sig in signatures)


def parse_webhook_event(payload: bytes, signature: str) -> Optional[WebhookEvent]:
    """
    Parse and verify a Stripe webhook event.

    Args:
        payload: Raw request body
        signature: Stripe-Signature header value

    Returns:
        WebhookEvent if valid, None otherwise
    """
    if not verify_webhook_signature(payload, signature):
        logger.warning("Invalid webhook signature")
        return None

    try:
        data = json.loads(payload)
        return WebhookEvent(
            event_type=data.get("type", ""),
            data=data.get("data", {}),
            event_id=data.get("id", ""),  # Preserve top-level Stripe event ID
        )
    except json.JSONDecodeError:
        logger.warning("Invalid webhook payload")
        return None


# =============================================================================
# Tier Mapping
# =============================================================================


def get_tier_from_price_id(price_id: str) -> Optional[SubscriptionTier]:
    """
    Get subscription tier from Stripe price ID.

    Args:
        price_id: Stripe price ID

    Returns:
        SubscriptionTier or None if not found
    """
    for tier, tier_price_id in STRIPE_PRICES.items():
        if tier_price_id == price_id:
            return tier
    return None


def get_price_id_for_tier(tier: SubscriptionTier) -> Optional[str]:
    """
    Get Stripe price ID for a subscription tier.

    Args:
        tier: Subscription tier

    Returns:
        Price ID string or None if not configured
    """
    return STRIPE_PRICES.get(tier)


# =============================================================================
# Default client instance
# =============================================================================

_default_client: Optional[StripeClient] = None


def get_stripe_client() -> StripeClient:
    """Get the default Stripe client instance."""
    global _default_client
    if _default_client is None:
        _default_client = StripeClient()
    return _default_client


__all__ = [
    # Client
    "StripeClient",
    "get_stripe_client",
    # Data classes
    "StripeCustomer",
    "StripeSubscription",
    "CheckoutSession",
    "BillingPortalSession",
    # Exceptions
    "StripeError",
    "StripeConfigError",
    "StripeAPIError",
    # Webhooks
    "WebhookEvent",
    "verify_webhook_signature",
    "parse_webhook_event",
    # Utilities
    "get_tier_from_price_id",
    "get_price_id_for_tier",
    "STRIPE_PRICES",
]
