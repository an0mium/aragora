"""
Plan management and route registration for payment endpoints.

This module provides the route registration function that wires up all
payment-related HTTP endpoints to the aiohttp application.
"""

from __future__ import annotations

from aiohttp import web

from .stripe import (
    handle_charge,
    handle_authorize,
    handle_capture,
    handle_refund,
    handle_void,
    handle_get_transaction,
    handle_stripe_webhook,
    handle_authnet_webhook,
)
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


def register_payment_routes(app: web.Application) -> None:
    """Register payment routes with the application."""
    # v1 canonical routes
    # Core payment operations
    app.router.add_post("/api/v1/payments/charge", handle_charge)
    app.router.add_post("/api/v1/payments/authorize", handle_authorize)
    app.router.add_post("/api/v1/payments/capture", handle_capture)
    app.router.add_post("/api/v1/payments/refund", handle_refund)
    app.router.add_post("/api/v1/payments/void", handle_void)
    app.router.add_get("/api/v1/payments/transaction/{transaction_id}", handle_get_transaction)

    # Customer management
    app.router.add_post("/api/v1/payments/customer", handle_create_customer)
    app.router.add_get("/api/v1/payments/customer/{customer_id}", handle_get_customer)
    app.router.add_put("/api/v1/payments/customer/{customer_id}", handle_update_customer)
    app.router.add_delete("/api/v1/payments/customer/{customer_id}", handle_delete_customer)

    # Subscription management
    app.router.add_post("/api/v1/payments/subscription", handle_create_subscription)
    app.router.add_get("/api/v1/payments/subscription/{subscription_id}", handle_get_subscription)
    app.router.add_put(
        "/api/v1/payments/subscription/{subscription_id}", handle_update_subscription
    )
    app.router.add_delete(
        "/api/v1/payments/subscription/{subscription_id}", handle_cancel_subscription
    )

    # Webhooks
    app.router.add_post("/api/v1/payments/webhook/stripe", handle_stripe_webhook)
    app.router.add_post("/api/v1/payments/webhook/authnet", handle_authnet_webhook)

    # legacy routes
    # Core payment operations
    app.router.add_post("/api/payments/charge", handle_charge)
    app.router.add_post("/api/payments/authorize", handle_authorize)
    app.router.add_post("/api/payments/capture", handle_capture)
    app.router.add_post("/api/payments/refund", handle_refund)
    app.router.add_post("/api/payments/void", handle_void)
    app.router.add_get("/api/payments/transaction/{transaction_id}", handle_get_transaction)

    # Customer management
    app.router.add_post("/api/payments/customer", handle_create_customer)
    app.router.add_get("/api/payments/customer/{customer_id}", handle_get_customer)
    app.router.add_put("/api/payments/customer/{customer_id}", handle_update_customer)
    app.router.add_delete("/api/payments/customer/{customer_id}", handle_delete_customer)

    # Subscription management
    app.router.add_post("/api/payments/subscription", handle_create_subscription)
    app.router.add_get("/api/payments/subscription/{subscription_id}", handle_get_subscription)
    app.router.add_put("/api/payments/subscription/{subscription_id}", handle_update_subscription)
    app.router.add_delete(
        "/api/payments/subscription/{subscription_id}", handle_cancel_subscription
    )

    # Webhooks
    app.router.add_post("/api/payments/webhook/stripe", handle_stripe_webhook)
    app.router.add_post("/api/payments/webhook/authnet", handle_authnet_webhook)
