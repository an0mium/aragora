"""
Payment processing handlers for Stripe and Authorize.net.

This package provides HTTP endpoints for:
- Payment processing (charge, authorize, capture, refund)
- Customer profile management
- Subscription management
- Webhook handling

All public symbols are re-exported here for backward compatibility.
"""

# Audit functions (used by submodules via _pkg() runtime lookup for patchability)
from aragora.audit.unified import audit_data, audit_security  # noqa: F401

# Data models and shared infrastructure
from .handler import (
    PaymentProvider,
    PaymentStatus,
    PaymentRequest,
    PaymentResult,
    # RBAC permission constants
    PERM_PAYMENTS_READ,
    PERM_PAYMENTS_CHARGE,
    PERM_PAYMENTS_AUTHORIZE,
    PERM_PAYMENTS_CAPTURE,
    PERM_PAYMENTS_REFUND,
    PERM_PAYMENTS_VOID,
    PERM_PAYMENTS_ADMIN,
    PERM_CUSTOMER_READ,
    PERM_CUSTOMER_CREATE,
    PERM_CUSTOMER_UPDATE,
    PERM_CUSTOMER_DELETE,
    PERM_SUBSCRIPTION_READ,
    PERM_SUBSCRIPTION_CREATE,
    PERM_SUBSCRIPTION_UPDATE,
    PERM_SUBSCRIPTION_CANCEL,
    PERM_WEBHOOK_STRIPE,
    PERM_WEBHOOK_AUTHNET,
    PERM_BILLING_DELETE,  # noqa: F401
    PERM_BILLING_CANCEL,  # noqa: F401
    # Connectors and helpers (used by tests)
    get_stripe_connector,  # noqa: F401
    get_authnet_connector,  # noqa: F401
    _get_provider_from_request,  # noqa: F401
    _resilient_stripe_call,  # noqa: F401
    _resilient_authnet_call,  # noqa: F401
    _stripe_cb,  # noqa: F401
    _authnet_cb,  # noqa: F401
    _check_rate_limit,  # noqa: F401
    _is_duplicate_webhook,  # noqa: F401
    _mark_webhook_processed,  # noqa: F401
    _payment_write_limiter,  # noqa: F401
    _payment_read_limiter,  # noqa: F401
    _webhook_limiter,  # noqa: F401
    _get_client_identifier,  # noqa: F401
    _payment_retry_config,  # noqa: F401
    _stripe_connector,  # noqa: F401
    _authnet_connector,  # noqa: F401
)

# Payment transaction handlers
from .stripe import (
    handle_charge,
    handle_authorize,
    handle_capture,
    handle_refund,
    handle_void,
    handle_get_transaction,
    # Webhook handlers
    handle_stripe_webhook,
    handle_authnet_webhook,
)

# Customer and subscription handlers
from .billing import (
    handle_create_customer,
    handle_get_customer,
    handle_update_customer,
    handle_delete_customer,
    handle_create_subscription,
    handle_get_subscription,
    handle_update_subscription,
    handle_cancel_subscription,
)

# Route registration
from .plans import register_payment_routes

__all__ = [
    # Route registration
    "register_payment_routes",
    # Payment transaction handlers
    "handle_charge",
    "handle_authorize",
    "handle_capture",
    "handle_refund",
    "handle_void",
    "handle_get_transaction",
    # Customer handlers
    "handle_create_customer",
    "handle_get_customer",
    "handle_update_customer",
    "handle_delete_customer",
    # Subscription handlers
    "handle_create_subscription",
    "handle_get_subscription",
    "handle_update_subscription",
    "handle_cancel_subscription",
    # Webhook handlers
    "handle_stripe_webhook",
    "handle_authnet_webhook",
    # Data models
    "PaymentProvider",
    "PaymentStatus",
    "PaymentRequest",
    "PaymentResult",
    # RBAC permission constants
    "PERM_PAYMENTS_READ",
    "PERM_PAYMENTS_CHARGE",
    "PERM_PAYMENTS_AUTHORIZE",
    "PERM_PAYMENTS_CAPTURE",
    "PERM_PAYMENTS_REFUND",
    "PERM_PAYMENTS_VOID",
    "PERM_PAYMENTS_ADMIN",
    "PERM_CUSTOMER_READ",
    "PERM_CUSTOMER_CREATE",
    "PERM_CUSTOMER_UPDATE",
    "PERM_CUSTOMER_DELETE",
    "PERM_SUBSCRIPTION_READ",
    "PERM_SUBSCRIPTION_CREATE",
    "PERM_SUBSCRIPTION_UPDATE",
    "PERM_SUBSCRIPTION_CANCEL",
    "PERM_WEBHOOK_STRIPE",
    "PERM_WEBHOOK_AUTHNET",
]
