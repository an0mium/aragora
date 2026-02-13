"""
Shared payment infrastructure: data models, connectors, resilience, and helpers.

This module contains the foundational components used by all payment submodules:
- Data models (PaymentProvider, PaymentStatus, PaymentRequest, PaymentResult)
- Connector management (Stripe, Authorize.net)
- Circuit breakers and retry configuration
- Rate limiting utilities
- Webhook idempotency helpers
- RBAC permission constants
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any

from aiohttp import web

from aragora.server.handlers.utils.rate_limit import RateLimiter
from aragora.resilience import (
    JitterMode,
    RetryConfig,
    RetryStrategy,
    get_v2_circuit_breaker as get_circuit_breaker,
    with_retry,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Route Registry (for SDK audit visibility)
# =============================================================================

ROUTES = [
    "/api/payments/charge",
    "/api/payments/authorize",
    "/api/payments/capture",
    "/api/payments/refund",
    "/api/payments/void",
    "/api/payments/transaction/*",
    "/api/payments/customer",
    "/api/payments/customer/*",
    "/api/payments/subscription",
    "/api/payments/subscription/*",
    "/api/payments/webhook/stripe",
    "/api/payments/webhook/authnet",
    "/api/v1/payments/charge",
    "/api/v1/payments/authorize",
    "/api/v1/payments/capture",
    "/api/v1/payments/refund",
    "/api/v1/payments/void",
    "/api/v1/payments/transaction/*",
    "/api/v1/payments/customer",
    "/api/v1/payments/customer/*",
    "/api/v1/payments/subscription",
    "/api/v1/payments/subscription/*",
]


# =============================================================================
# RBAC Permission Constants for Payments
# =============================================================================

# Payment transaction permissions
PERM_PAYMENTS_READ = "payments:read"
PERM_PAYMENTS_CHARGE = "payments:charge"
PERM_PAYMENTS_AUTHORIZE = "payments:authorize"
PERM_PAYMENTS_CAPTURE = "payments:capture"
PERM_PAYMENTS_REFUND = "payments:refund"
PERM_PAYMENTS_VOID = "payments:void"
PERM_PAYMENTS_ADMIN = "payments:admin"

# Customer profile permissions
PERM_CUSTOMER_READ = "payments:customer:read"
PERM_CUSTOMER_CREATE = "payments:customer:create"
PERM_CUSTOMER_UPDATE = "payments:customer:update"
PERM_CUSTOMER_DELETE = "payments:customer:delete"

# Subscription permissions
PERM_SUBSCRIPTION_READ = "payments:subscription:read"
PERM_SUBSCRIPTION_CREATE = "payments:subscription:create"
PERM_SUBSCRIPTION_UPDATE = "payments:subscription:update"
PERM_SUBSCRIPTION_CANCEL = "payments:subscription:cancel"

# Webhook permissions (for audit logging - primary auth is signature verification)
PERM_WEBHOOK_STRIPE = "payments:webhook:stripe"
PERM_WEBHOOK_AUTHNET = "payments:webhook:authnet"

# Billing permissions (legacy compatibility)
PERM_BILLING_DELETE = "billing:delete"
PERM_BILLING_CANCEL = "billing:cancel"

# =============================================================================
# Resilience Configuration
# =============================================================================

# Circuit breakers for payment providers
_stripe_cb = get_circuit_breaker("stripe_payments", failure_threshold=5, cooldown_seconds=60)
_authnet_cb = get_circuit_breaker("authnet_payments", failure_threshold=5, cooldown_seconds=60)

# Retry configuration for transient failures
_payment_retry_config = RetryConfig(
    max_retries=2,
    base_delay=0.5,
    max_delay=5.0,
    strategy=RetryStrategy.EXPONENTIAL,
    jitter_mode=JitterMode.MULTIPLICATIVE,
    retryable_exceptions=(ConnectionError, TimeoutError, OSError),
)

# Rate limiters for payment endpoints (strict limits for financial operations)
# Charge/refund operations: 10 per minute (prevents abuse)
_payment_write_limiter = RateLimiter(requests_per_minute=10)
# Read operations: 30 per minute (less sensitive)
_payment_read_limiter = RateLimiter(requests_per_minute=30)
# Webhooks: higher limit since they're server-to-server (idempotency handles dupes)
_webhook_limiter = RateLimiter(requests_per_minute=100)


def _get_client_identifier(request: web.Request) -> str:
    """Extract client identifier for rate limiting.

    Uses X-Forwarded-For if behind proxy, otherwise uses peer IP.
    Falls back to user_id if available for authenticated requests.
    """
    # Try user_id first for authenticated requests
    user_id = request.get("user_id")
    if user_id:
        return f"user:{user_id}"

    # Try X-Forwarded-For for proxied requests
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take first IP in chain (original client)
        return forwarded.split(",")[0].strip()

    # Fall back to peer IP
    peername = request.transport.get_extra_info("peername") if request.transport else None
    if peername:
        return peername[0]

    return "unknown"


def _check_rate_limit(request: web.Request, limiter: RateLimiter) -> web.Response | None:
    """Check rate limit and return error response if exceeded, None if allowed."""
    client_id = _get_client_identifier(request)
    if not limiter.is_allowed(client_id):
        logger.warning(f"Rate limit exceeded for payment endpoint: {client_id}")
        return web.json_response(
            {"error": "Rate limit exceeded. Please try again later."},
            status=429,
            headers={"Retry-After": "60"},
        )
    return None


# =============================================================================
# Webhook Idempotency
# =============================================================================


def _is_duplicate_webhook(event_id: str) -> bool:
    """Check if webhook event was already processed.

    Uses persistent storage (SQLite/PostgreSQL) to track processed events,
    ensuring idempotency survives server restarts.
    """
    from aragora.storage.webhook_store import get_webhook_store

    store = get_webhook_store()
    return store.is_processed(event_id)


def _mark_webhook_processed(event_id: str, result: str = "success") -> None:
    """Mark webhook event as processed.

    Stores the event ID with a 24-hour TTL to prevent duplicate processing.
    """
    from aragora.storage.webhook_store import get_webhook_store

    store = get_webhook_store()
    store.mark_processed(event_id, result)


# =============================================================================
# Data Models
# =============================================================================


class PaymentProvider(Enum):
    """Supported payment providers."""

    STRIPE = "stripe"
    AUTHORIZE_NET = "authorize_net"


class PaymentStatus(Enum):
    """Payment transaction status."""

    PENDING = "pending"
    APPROVED = "approved"
    DECLINED = "declined"
    ERROR = "error"
    VOID = "void"
    REFUNDED = "refunded"


@dataclass
class PaymentRequest:
    """Unified payment request."""

    amount: Decimal
    currency: str = "USD"
    description: str | None = None
    customer_id: str | None = None
    payment_method: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provider: PaymentProvider = PaymentProvider.STRIPE


@dataclass
class PaymentResult:
    """Unified payment result."""

    transaction_id: str
    provider: PaymentProvider
    status: PaymentStatus
    amount: Decimal
    currency: str
    message: str | None = None
    auth_code: str | None = None
    avs_result: str | None = None
    cvv_result: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "transaction_id": self.transaction_id,
            "provider": self.provider.value,
            "status": self.status.value,
            "amount": str(self.amount),
            "currency": self.currency,
            "message": self.message,
            "auth_code": self.auth_code,
            "avs_result": self.avs_result,
            "cvv_result": self.cvv_result,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


# =============================================================================
# Connector Management
# =============================================================================

_stripe_connector: Any | None = None
_authnet_connector: Any | None = None


async def get_stripe_connector(request: web.Request) -> Any | None:
    """Get or create Stripe connector."""
    global _stripe_connector
    if _stripe_connector is None:
        try:
            from aragora.connectors.payments.stripe import StripeConnector, StripeCredentials

            _stripe_connector = StripeConnector(
                StripeCredentials(
                    secret_key=os.environ.get("STRIPE_SECRET_KEY", ""),
                    webhook_secret=os.environ.get("STRIPE_WEBHOOK_SECRET"),
                )
            )
            logger.info("Stripe connector initialized")
        except ImportError:
            logger.warning("Stripe connector not available")
            return None
        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"Failed to initialize Stripe connector: {e}")
            return None
    return _stripe_connector


async def get_authnet_connector(request: web.Request) -> Any | None:
    """Get or create Authorize.net connector."""
    global _authnet_connector
    if _authnet_connector is None:
        try:
            from aragora.connectors.payments.authorize_net import (
                create_authorize_net_connector,
            )

            _authnet_connector = create_authorize_net_connector()
            if _authnet_connector:
                logger.info("Authorize.net connector initialized")
            else:
                logger.warning("Authorize.net credentials not configured")
        except ImportError:
            logger.warning("Authorize.net connector not available")
            return None
        except (OSError, ValueError, RuntimeError) as e:
            logger.error(f"Failed to initialize Authorize.net connector: {e}")
            return None
    return _authnet_connector


def _get_provider_from_request(request: web.Request, body: dict[str, Any]) -> PaymentProvider:
    """Determine payment provider from request."""
    provider_str = body.get("provider", "stripe").lower()
    if provider_str == "authorize_net" or provider_str == "authnet":
        return PaymentProvider.AUTHORIZE_NET
    return PaymentProvider.STRIPE


async def _resilient_stripe_call(operation: str, func, *args, **kwargs):
    """Execute a Stripe API call with circuit breaker and retry.

    Args:
        operation: Name of the operation (for logging)
        func: Async function to execute
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Result from the function

    Raises:
        ConnectionError: If circuit is open
        Exception: If all retries exhausted
    """
    if not _stripe_cb.can_execute():
        logger.warning(f"Stripe circuit breaker open for {operation}")
        raise ConnectionError("Stripe service temporarily unavailable")

    @with_retry(_payment_retry_config)
    async def _execute():
        return await func(*args, **kwargs)

    try:
        result = await _execute()
        _stripe_cb.record_success()
        return result
    except Exception as e:
        # Intentionally broad: circuit breaker needs to catch all failures
        _stripe_cb.record_failure(e)
        logger.error(f"Stripe {operation} failed: {e}")
        raise


async def _resilient_authnet_call(operation: str, func, *args, **kwargs):
    """Execute an Authorize.net API call with circuit breaker and retry.

    Args:
        operation: Name of the operation (for logging)
        func: Async function to execute
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func

    Returns:
        Result from the function

    Raises:
        ConnectionError: If circuit is open
        Exception: If all retries exhausted
    """
    if not _authnet_cb.can_execute():
        logger.warning(f"Authorize.net circuit breaker open for {operation}")
        raise ConnectionError("Authorize.net service temporarily unavailable")

    @with_retry(_payment_retry_config)
    async def _execute():
        return await func(*args, **kwargs)

    try:
        result = await _execute()
        _authnet_cb.record_success()
        return result
    except Exception as e:
        # Intentionally broad: circuit breaker needs to catch all failures
        _authnet_cb.record_failure(e)
        logger.error(f"Authorize.net {operation} failed: {e}")
        raise
