"""
Stripe-specific payment processing and webhook handling.

Provides HTTP endpoints for:
- POST /api/payments/charge (Stripe path)
- POST /api/payments/authorize (Stripe path)
- POST /api/payments/capture (Stripe path)
- POST /api/payments/refund (Stripe path)
- POST /api/payments/void (Stripe path)
- GET  /api/payments/transaction/{id} (Stripe path)
- POST /api/payments/webhook/stripe
- POST /api/payments/webhook/authnet
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from typing import Any

from aiohttp import web

from aragora.server.handlers.utils import parse_json_body
from aragora.server.handlers.utils.aiohttp_responses import web_error_response
from aragora.observability.metrics import track_handler
from aragora.rbac.decorators import PermissionDeniedError, require_permission
from aragora.rbac.models import AuthorizationContext

from .handler import (
    PaymentProvider,
    PaymentStatus,
    PaymentResult,
    _payment_write_limiter,
    _payment_read_limiter,
    _webhook_limiter,
    PERM_PAYMENTS_CHARGE,
    PERM_PAYMENTS_AUTHORIZE,
    PERM_PAYMENTS_CAPTURE,
    PERM_PAYMENTS_REFUND,
    PERM_PAYMENTS_VOID,
    PERM_PAYMENTS_READ,
    logger,
)


def _pkg():
    """Get the parent package module for runtime attribute lookup.

    This enables mock.patch at 'aragora.server.handlers.payments.X' to work
    correctly by resolving patchable names from the package namespace at call time,
    rather than using module-level imports that create local bindings.
    """
    return sys.modules[__package__]


def _coerce_request(
    request_or_context: Any, maybe_request: web.Request | None = None
) -> web.Request:
    """Support legacy (auth_context, request) handler signatures."""
    return maybe_request if maybe_request is not None else request_or_context


def _enforce_permission_if_context(
    context: Any, permission_key: str, resource_id: str | None = None
) -> None:
    """Fallback permission check when a raw AuthorizationContext is provided."""
    if isinstance(context, AuthorizationContext):
        # Resolve checker at runtime so tests can patch aragora.rbac.decorators.get_permission_checker.
        from aragora.rbac import decorators as rbac_decorators

        checker = rbac_decorators.get_permission_checker()
        decision = checker.check_permission(context, permission_key, resource_id)
        if not decision.allowed:
            reason = decision.reason or f"Permission denied: {permission_key}"
            if permission_key not in reason:
                dotted = permission_key.replace(":", ".")
                if dotted in reason:
                    reason = reason.replace(dotted, permission_key)
            raise PermissionDeniedError(reason, decision)


# =============================================================================
# Payment Handlers
# =============================================================================


@require_permission(PERM_PAYMENTS_CHARGE)
@track_handler("payments/charge")
async def handle_charge(
    request: web.Request | Any, maybe_request: web.Request | None = None
) -> web.Response:
    """
    POST /api/payments/charge

    Process a payment charge.

    Request body:
    {
        "provider": "stripe" | "authorize_net",
        "amount": 100.00,
        "currency": "USD",
        "description": "Order #123",
        "customer_id": "cus_...",  // Optional for Stripe
        "payment_method": {
            "type": "card",
            "card_number": "4111111111111111",  // For Authorize.net
            "exp_month": "12",
            "exp_year": "2025",
            "cvv": "123"
        } | "pm_...",  // Payment method ID for Stripe
        "metadata": {}
    }
    """
    _enforce_permission_if_context(request, PERM_PAYMENTS_CHARGE)
    request = _coerce_request(request, maybe_request)

    # Rate limit check for financial operations
    rate_limit_response = _pkg()._check_rate_limit(request, _payment_write_limiter)
    if rate_limit_response:
        return rate_limit_response

    try:
        body, err = await parse_json_body(request, context="handle_charge")
        if err:
            return err
        provider = _pkg()._get_provider_from_request(request, body)

        amount = Decimal(str(body.get("amount", 0)))
        if amount <= 0:
            return web.json_response(
                {"error": "Amount must be greater than 0"},
                status=400,
            )

        currency = body.get("currency", "USD")
        description = body.get("description")
        customer_id = body.get("customer_id")
        payment_method = body.get("payment_method")
        metadata = body.get("metadata", {})

        if provider == PaymentProvider.STRIPE:
            result = await _charge_stripe(
                request, amount, currency, description, customer_id, payment_method, metadata
            )
        else:
            result = await _charge_authnet(
                request, amount, currency, description, payment_method, metadata
            )

        return web.json_response(
            {
                "success": result.status == PaymentStatus.APPROVED,
                "transaction": result.to_dict(),
            }
        )

    except Exception as e:
        logger.exception(f"Error processing charge: {e}")
        return web_error_response(str(e), 500)


async def _charge_stripe(
    request: web.Request,
    amount: Decimal,
    currency: str,
    description: str | None,
    customer_id: str | None,
    payment_method: Any,
    metadata: dict[str, Any],
) -> PaymentResult:
    """Process charge via Stripe."""
    connector = await _pkg().get_stripe_connector(request)
    if not connector:
        return PaymentResult(
            transaction_id="",
            provider=PaymentProvider.STRIPE,
            status=PaymentStatus.ERROR,
            amount=amount,
            currency=currency,
            message="Stripe connector not available",
        )

    try:
        # Convert amount to cents for Stripe
        amount_cents = int(amount * 100)

        intent = await _pkg()._resilient_stripe_call(
            "create_payment_intent",
            connector.create_payment_intent,
            amount=amount_cents,
            currency=currency.lower(),
            description=description,
            customer=customer_id,
            payment_method=payment_method if isinstance(payment_method, str) else None,
            metadata=metadata,
        )

        return PaymentResult(
            transaction_id=intent.id,
            provider=PaymentProvider.STRIPE,
            status=(
                PaymentStatus.APPROVED if intent.status == "succeeded" else PaymentStatus.PENDING
            ),
            amount=amount,
            currency=currency,
            message=f"Payment intent {intent.status}",
            metadata=(
                {"client_secret": intent.client_secret} if hasattr(intent, "client_secret") else {}
            ),
        )

    except ConnectionError as e:
        return PaymentResult(
            transaction_id="",
            provider=PaymentProvider.STRIPE,
            status=PaymentStatus.ERROR,
            amount=amount,
            currency=currency,
            message=f"Service unavailable: {e}",
        )
    except Exception as e:
        return PaymentResult(
            transaction_id="",
            provider=PaymentProvider.STRIPE,
            status=PaymentStatus.ERROR,
            amount=amount,
            currency=currency,
            message=str(e),
        )


async def _charge_authnet(
    request: web.Request,
    amount: Decimal,
    currency: str,
    description: str | None,
    payment_method: Any,
    metadata: dict[str, Any],
) -> PaymentResult:
    """Process charge via Authorize.net."""
    connector = await _pkg().get_authnet_connector(request)
    if not connector:
        return PaymentResult(
            transaction_id="",
            provider=PaymentProvider.AUTHORIZE_NET,
            status=PaymentStatus.ERROR,
            amount=amount,
            currency=currency,
            message="Authorize.net connector not available",
        )

    try:
        from aragora.connectors.payments.authorize_net import CreditCard, BillingAddress

        # Parse payment method
        if isinstance(payment_method, dict):
            card = CreditCard(
                card_number=payment_method.get("card_number", ""),
                expiration_date=f"{payment_method.get('exp_month', '01')}{payment_method.get('exp_year', '2025')[-2:]}",
                card_code=payment_method.get("cvv"),
            )
            billing = None
            if payment_method.get("billing"):
                billing_data = payment_method["billing"]
                billing = BillingAddress(
                    first_name=billing_data.get("first_name", ""),
                    last_name=billing_data.get("last_name", ""),
                    address=billing_data.get("address"),
                    city=billing_data.get("city"),
                    state=billing_data.get("state"),
                    zip_code=billing_data.get("zip"),
                    country=billing_data.get("country"),
                )
        else:
            return PaymentResult(
                transaction_id="",
                provider=PaymentProvider.AUTHORIZE_NET,
                status=PaymentStatus.ERROR,
                amount=amount,
                currency=currency,
                message="Invalid payment method for Authorize.net",
            )

        async def _do_charge():
            async with connector:
                return await connector.charge(
                    amount=amount,
                    payment_method=card,
                    billing_address=billing,
                    description=description,
                )

        result = await _pkg()._resilient_authnet_call("charge", _do_charge)

        return PaymentResult(
            transaction_id=result.transaction_id,
            provider=PaymentProvider.AUTHORIZE_NET,
            status=PaymentStatus.APPROVED if result.approved else PaymentStatus.DECLINED,
            amount=amount,
            currency=currency,
            message=result.message,
            auth_code=result.auth_code,
            avs_result=result.avs_result,
            cvv_result=result.cvv_result,
        )

    except ConnectionError as e:
        return PaymentResult(
            transaction_id="",
            provider=PaymentProvider.AUTHORIZE_NET,
            status=PaymentStatus.ERROR,
            amount=amount,
            currency=currency,
            message=f"Service unavailable: {e}",
        )
    except Exception as e:
        return PaymentResult(
            transaction_id="",
            provider=PaymentProvider.AUTHORIZE_NET,
            status=PaymentStatus.ERROR,
            amount=amount,
            currency=currency,
            message=str(e),
        )


@require_permission(PERM_PAYMENTS_AUTHORIZE)
@track_handler("payments/authorize")
async def handle_authorize(request: web.Request) -> web.Response:
    """
    POST /api/payments/authorize

    Authorize a payment (capture later).
    """
    # Rate limit check for financial operations
    rate_limit_response = _pkg()._check_rate_limit(request, _payment_write_limiter)
    if rate_limit_response:
        return rate_limit_response

    try:
        body, err = await parse_json_body(request, context="handle_authorize")
        if err:
            return err
        provider = _pkg()._get_provider_from_request(request, body)

        amount = Decimal(str(body.get("amount", 0)))
        if amount <= 0:
            return web_error_response("Amount must be greater than 0", 400)

        currency = body.get("currency", "USD")
        payment_method = body.get("payment_method")

        if provider == PaymentProvider.AUTHORIZE_NET:
            connector = await _pkg().get_authnet_connector(request)
            if not connector:
                return web.json_response(
                    {"error": "Authorize.net connector not available"}, status=503
                )

            from aragora.connectors.payments.authorize_net import CreditCard

            if isinstance(payment_method, dict):
                card = CreditCard(
                    card_number=payment_method.get("card_number", ""),
                    expiration_date=f"{payment_method.get('exp_month', '01')}{payment_method.get('exp_year', '2025')[-2:]}",
                    card_code=payment_method.get("cvv"),
                )

                async with connector:
                    result = await connector.authorize(amount=amount, payment_method=card)

                return web.json_response(
                    {
                        "success": result.approved,
                        "transaction_id": result.transaction_id,
                        "auth_code": result.auth_code,
                        "message": result.message,
                    }
                )
        else:
            # Stripe authorization
            connector = await _pkg().get_stripe_connector(request)
            if not connector:
                return web_error_response("Stripe connector not available", 503)

            amount_cents = int(amount * 100)
            intent = await connector.create_payment_intent(
                amount=amount_cents,
                currency=currency.lower(),
                capture_method="manual",  # Authorize only
                payment_method=payment_method if isinstance(payment_method, str) else None,
            )

            return web.json_response(
                {
                    "success": True,
                    "transaction_id": intent.id,
                    "client_secret": intent.client_secret,
                    "status": intent.status,
                }
            )

        return web_error_response("Invalid request", 400)

    except Exception as e:
        logger.exception(f"Error authorizing payment: {e}")
        return web_error_response(str(e), 500)


@require_permission(PERM_PAYMENTS_CAPTURE)
@track_handler("payments/capture")
async def handle_capture(request: web.Request) -> web.Response:
    """
    POST /api/payments/capture

    Capture an authorized payment.

    Request body:
    {
        "provider": "stripe" | "authorize_net",
        "transaction_id": "...",
        "amount": 100.00  // Optional, for partial capture
    }
    """
    # Rate limit check for financial operations
    rate_limit_response = _pkg()._check_rate_limit(request, _payment_write_limiter)
    if rate_limit_response:
        return rate_limit_response

    try:
        body, err = await parse_json_body(request, context="handle_capture")
        if err:
            return err
        provider = _pkg()._get_provider_from_request(request, body)
        transaction_id = body.get("transaction_id")
        amount = body.get("amount")

        if not transaction_id:
            return web_error_response("Missing transaction_id", 400)

        if provider == PaymentProvider.AUTHORIZE_NET:
            connector = await _pkg().get_authnet_connector(request)
            if not connector:
                return web.json_response(
                    {"error": "Authorize.net connector not available"}, status=503
                )

            async with connector:
                result = await connector.capture(
                    transaction_id=transaction_id,
                    amount=Decimal(str(amount)) if amount else None,
                )

            return web.json_response(
                {
                    "success": result.approved,
                    "transaction_id": result.transaction_id,
                    "message": result.message,
                }
            )
        else:
            connector = await _pkg().get_stripe_connector(request)
            if not connector:
                return web_error_response("Stripe connector not available", 503)

            intent = await connector.capture_payment_intent(
                payment_intent_id=transaction_id,
                amount_to_capture=int(Decimal(str(amount)) * 100) if amount else None,
            )

            return web.json_response(
                {
                    "success": intent.status == "succeeded",
                    "transaction_id": intent.id,
                    "status": intent.status,
                }
            )

    except Exception as e:
        logger.exception(f"Error capturing payment: {e}")
        return web_error_response(str(e), 500)


@require_permission(PERM_PAYMENTS_REFUND)
@track_handler("payments/refund")
async def handle_refund(
    request: web.Request | Any, maybe_request: web.Request | None = None
) -> web.Response:
    """
    POST /api/payments/refund

    Refund a payment.

    Request body:
    {
        "provider": "stripe" | "authorize_net",
        "transaction_id": "...",
        "amount": 100.00,
        "card_last_four": "1111"  // Required for Authorize.net
    }
    """
    _enforce_permission_if_context(request, PERM_PAYMENTS_REFUND)
    request = _coerce_request(request, maybe_request)

    # Rate limit check for financial operations
    rate_limit_response = _pkg()._check_rate_limit(request, _payment_write_limiter)
    if rate_limit_response:
        return rate_limit_response

    body = None
    try:
        body, err = await parse_json_body(request, context="handle_refund")
        if err:
            return err
        provider = _pkg()._get_provider_from_request(request, body)
        transaction_id = body.get("transaction_id")
        amount = Decimal(str(body.get("amount", 0)))

        if not transaction_id:
            return web_error_response("Missing transaction_id", 400)
        if amount <= 0:
            return web_error_response("Amount must be greater than 0", 400)

        if provider == PaymentProvider.AUTHORIZE_NET:
            connector = await _pkg().get_authnet_connector(request)
            if not connector:
                return web.json_response(
                    {"error": "Authorize.net connector not available"}, status=503
                )

            card_last_four = body.get("card_last_four")
            if not card_last_four:
                return web.json_response(
                    {"error": "card_last_four required for Authorize.net refunds"}, status=400
                )

            async with connector:
                result = await connector.refund(
                    transaction_id=transaction_id,
                    amount=amount,
                    card_last_four=card_last_four,
                )

            # Audit the refund
            _pkg().audit_data(
                user_id=request.get("user_id", "unknown"),
                action="payment_refund",
                resource_type="payment",
                resource_id=result.transaction_id,
                provider="authorize_net",
                amount=str(amount),
                success=result.approved,
            )

            return web.json_response(
                {
                    "success": result.approved,
                    "transaction_id": result.transaction_id,
                    "message": result.message,
                }
            )
        else:
            connector = await _pkg().get_stripe_connector(request)
            if not connector:
                return web_error_response("Stripe connector not available", 503)

            refund = await connector.create_refund(
                payment_intent=transaction_id,
                amount=int(amount * 100),
            )

            # Audit the refund
            _pkg().audit_data(
                user_id=request.get("user_id", "unknown"),
                action="payment_refund",
                resource_type="payment",
                resource_id=refund.id,
                original_transaction_id=transaction_id,
                provider="stripe",
                amount=str(amount),
                status=refund.status,
                success=refund.status == "succeeded",
            )

            return web.json_response(
                {
                    "success": refund.status == "succeeded",
                    "refund_id": refund.id,
                    "status": refund.status,
                }
            )

    except Exception as e:
        logger.exception(f"Error processing refund: {e}")
        _pkg().audit_security(
            event_type="refund_error",
            actor_id=request.get("user_id", "unknown"),
            resource_type="payment",
            resource_id=body.get("transaction_id", "unknown") if body else "unknown",
            reason=str(e),
        )
        return web_error_response(str(e), 500)


@require_permission(PERM_PAYMENTS_VOID)
@track_handler("payments/void")
async def handle_void(
    request: web.Request | Any, maybe_request: web.Request | None = None
) -> web.Response:
    """
    POST /api/payments/void

    Void a transaction.

    Request body:
    {
        "provider": "stripe" | "authorize_net",
        "transaction_id": "..."
    }
    """
    _enforce_permission_if_context(request, PERM_PAYMENTS_VOID)
    request = _coerce_request(request, maybe_request)

    # Rate limit check for financial operations
    rate_limit_response = _pkg()._check_rate_limit(request, _payment_write_limiter)
    if rate_limit_response:
        return rate_limit_response

    try:
        body, err = await parse_json_body(request, context="handle_void")
        if err:
            return err
        provider = _pkg()._get_provider_from_request(request, body)
        transaction_id = body.get("transaction_id")

        if not transaction_id:
            return web_error_response("Missing transaction_id", 400)

        if provider == PaymentProvider.AUTHORIZE_NET:
            connector = await _pkg().get_authnet_connector(request)
            if not connector:
                return web.json_response(
                    {"error": "Authorize.net connector not available"}, status=503
                )

            async with connector:
                result = await connector.void(transaction_id=transaction_id)

            return web.json_response(
                {
                    "success": result.approved,
                    "transaction_id": result.transaction_id,
                    "message": result.message,
                }
            )
        else:
            connector = await _pkg().get_stripe_connector(request)
            if not connector:
                return web_error_response("Stripe connector not available", 503)

            intent = await connector.cancel_payment_intent(transaction_id)

            return web.json_response(
                {
                    "success": intent.status == "canceled",
                    "transaction_id": intent.id,
                    "status": intent.status,
                }
            )

    except Exception as e:
        logger.exception(f"Error voiding transaction: {e}")
        return web_error_response(str(e), 500)


@require_permission(PERM_PAYMENTS_READ)
async def handle_get_transaction(
    request: web.Request | Any, maybe_request: web.Request | None = None
) -> web.Response:
    """
    GET /api/payments/transaction/{transaction_id}

    Get transaction details.
    """
    _enforce_permission_if_context(request, PERM_PAYMENTS_READ)
    request = _coerce_request(request, maybe_request)

    # Rate limit check for read operations
    rate_limit_response = _pkg()._check_rate_limit(request, _payment_read_limiter)
    if rate_limit_response:
        return rate_limit_response

    try:
        transaction_id = request.match_info.get("transaction_id")
        provider_str = request.query.get("provider", "stripe")

        if not transaction_id:
            return web_error_response("Missing transaction_id", 400)

        provider = (
            PaymentProvider.AUTHORIZE_NET
            if provider_str in ("authorize_net", "authnet")
            else PaymentProvider.STRIPE
        )

        if provider == PaymentProvider.AUTHORIZE_NET:
            connector = await _pkg().get_authnet_connector(request)
            if not connector:
                return web.json_response(
                    {"error": "Authorize.net connector not available"}, status=503
                )

            async with connector:
                details = await connector.get_transaction_details(transaction_id)

            if not details:
                return web_error_response("Transaction not found", 404)

            return web.json_response({"transaction": details})
        else:
            connector = await _pkg().get_stripe_connector(request)
            if not connector:
                return web_error_response("Stripe connector not available", 503)

            intent = await connector.retrieve_payment_intent(transaction_id)

            return web.json_response(
                {
                    "transaction": {
                        "id": intent.id,
                        "amount": intent.amount,
                        "currency": intent.currency,
                        "status": intent.status,
                        "created": intent.created,
                        "metadata": intent.metadata,
                    }
                }
            )

    except Exception as e:
        logger.exception(f"Error getting transaction: {e}")
        return web_error_response(str(e), 500)


# =============================================================================
# Webhook Handlers
# =============================================================================


@track_handler("payments/webhook/stripe")
async def handle_stripe_webhook(request: web.Request) -> web.Response:
    """
    POST /api/payments/webhook/stripe

    Handle Stripe webhook events.
    """
    # Rate limit check for webhooks (higher limit, server-to-server)
    rate_limit_response = _pkg()._check_rate_limit(request, _webhook_limiter)
    if rate_limit_response:
        return rate_limit_response

    try:
        payload = await request.read()
        sig_header = request.headers.get("Stripe-Signature")

        connector = await _pkg().get_stripe_connector(request)
        if not connector:
            return web_error_response("Stripe connector not available", 503)

        # Verify webhook signature
        try:
            event = await connector.construct_webhook_event(payload, sig_header)
        except ValueError:
            return web_error_response("Invalid payload", 400)
        except Exception as e:
            return web_error_response(f"Signature verification failed: {e}", 400)

        # Get event ID for idempotency check
        event_id = event.id
        if not event_id:
            logger.warning("Webhook event missing ID, cannot check idempotency")
        elif _pkg()._is_duplicate_webhook(event_id):
            logger.info(f"Skipping duplicate Stripe webhook: {event_id}")
            return web.json_response({"received": True, "duplicate": True})

        # Handle the event
        event_type = event.type
        logger.info(f"Received Stripe webhook: {event_type} (id={event_id})")

        # Process different event types
        if event_type == "payment_intent.succeeded":
            logger.info(f"Payment succeeded: {event.data.object.id}")
        elif event_type == "payment_intent.payment_failed":
            logger.warning(f"Payment failed: {event.data.object.id}")
        elif event_type == "customer.subscription.created":
            logger.info(f"Subscription created: {event.data.object.id}")
        elif event_type == "customer.subscription.deleted":
            logger.info(f"Subscription canceled: {event.data.object.id}")
        elif event_type == "invoice.payment_failed":
            logger.warning(f"Invoice payment failed: {event.data.object.id}")

        # Mark as processed after successful handling
        if event_id:
            _pkg()._mark_webhook_processed(event_id)

        return web.json_response({"received": True})

    except Exception as e:
        logger.exception(f"Error handling Stripe webhook: {e}")
        return web_error_response(str(e), 500)


@track_handler("payments/webhook/authnet")
async def handle_authnet_webhook(request: web.Request) -> web.Response:
    """
    POST /api/payments/webhook/authnet

    Handle Authorize.net webhook events.
    """
    # Rate limit check for webhooks (higher limit, server-to-server)
    rate_limit_response = _pkg()._check_rate_limit(request, _webhook_limiter)
    if rate_limit_response:
        return rate_limit_response

    try:
        payload, err = await parse_json_body(request, context="handle_authnet_webhook")
        if err:
            return err
        signature = request.headers.get("X-ANET-Signature")

        connector = await _pkg().get_authnet_connector(request)
        if not connector:
            return web_error_response("Authorize.net connector not available", 503)

        # Verify webhook signature
        async with connector:
            if not await connector.verify_webhook_signature(payload, signature or ""):
                return web_error_response("Invalid signature", 400)

        # Get event ID for idempotency check
        event_id = payload.get("notificationId") or payload.get("payload", {}).get("id")
        if not event_id:
            # Generate deterministic ID from payload if not provided
            import hashlib

            payload_str = json.dumps(payload, sort_keys=True)
            event_id = f"authnet_{hashlib.sha256(payload_str.encode()).hexdigest()[:16]}"

        if _pkg()._is_duplicate_webhook(event_id):
            logger.info(f"Skipping duplicate Authorize.net webhook: {event_id}")
            return web.json_response({"received": True, "duplicate": True})

        # Handle the event
        event_type = payload.get("eventType", "")
        logger.info(f"Received Authorize.net webhook: {event_type} (id={event_id})")

        # Process different event types
        if event_type == "net.authorize.payment.authcapture.created":
            logger.info(f"Payment captured: {payload.get('payload', {}).get('id')}")
        elif event_type == "net.authorize.payment.refund.created":
            logger.info(f"Refund processed: {payload.get('payload', {}).get('id')}")
        elif event_type == "net.authorize.customer.subscription.created":
            logger.info(f"Subscription created: {payload.get('payload', {}).get('id')}")
        elif event_type == "net.authorize.customer.subscription.cancelled":
            logger.info(f"Subscription canceled: {payload.get('payload', {}).get('id')}")

        # Mark as processed after successful handling
        _pkg()._mark_webhook_processed(event_id)

        return web.json_response({"received": True})

    except Exception as e:
        logger.exception(f"Error handling Authorize.net webhook: {e}")
        return web_error_response(str(e), 500)
