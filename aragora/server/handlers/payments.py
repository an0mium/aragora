"""
Payment processing handlers for Stripe and Authorize.net.

Provides HTTP endpoints for:
- Payment processing (charge, authorize, capture, refund)
- Customer profile management
- Subscription management
- Webhook handling

Endpoints:
- POST /api/payments/charge              - Process a payment
- POST /api/payments/authorize           - Authorize a payment (capture later)
- POST /api/payments/capture             - Capture an authorized payment
- POST /api/payments/refund              - Refund a payment
- POST /api/payments/void                - Void a transaction
- GET  /api/payments/transaction/{id}    - Get transaction details
- POST /api/payments/customer            - Create customer profile
- GET  /api/payments/customer/{id}       - Get customer profile
- PUT  /api/payments/customer/{id}       - Update customer profile
- DELETE /api/payments/customer/{id}     - Delete customer profile
- POST /api/payments/subscription        - Create subscription
- GET  /api/payments/subscription/{id}   - Get subscription
- PUT  /api/payments/subscription/{id}   - Update subscription
- DELETE /api/payments/subscription/{id} - Cancel subscription
- POST /api/payments/webhook/stripe      - Stripe webhook endpoint
- POST /api/payments/webhook/authnet     - Authorize.net webhook endpoint
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

from aiohttp import web

from aragora.audit.unified import audit_data, audit_security
from aragora.server.handlers.utils.decorators import require_permission

logger = logging.getLogger(__name__)


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
    description: Optional[str] = None
    customer_id: Optional[str] = None
    payment_method: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    provider: PaymentProvider = PaymentProvider.STRIPE


@dataclass
class PaymentResult:
    """Unified payment result."""

    transaction_id: str
    provider: PaymentProvider
    status: PaymentStatus
    amount: Decimal
    currency: str
    message: Optional[str] = None
    auth_code: Optional[str] = None
    avs_result: Optional[str] = None
    cvv_result: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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


_stripe_connector: Optional[Any] = None
_authnet_connector: Optional[Any] = None


async def get_stripe_connector(request: web.Request) -> Optional[Any]:
    """Get or create Stripe connector."""
    global _stripe_connector
    if _stripe_connector is None:
        try:
            from aragora.connectors.payments.stripe import StripeConnector

            _stripe_connector = StripeConnector()  # type: ignore[call-arg]
            logger.info("Stripe connector initialized")
        except ImportError:
            logger.warning("Stripe connector not available")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize Stripe connector: {e}")
            return None
    return _stripe_connector


async def get_authnet_connector(request: web.Request) -> Optional[Any]:
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
        except Exception as e:
            logger.error(f"Failed to initialize Authorize.net connector: {e}")
            return None
    return _authnet_connector


def _get_provider_from_request(request: web.Request, body: Dict[str, Any]) -> PaymentProvider:
    """Determine payment provider from request."""
    provider_str = body.get("provider", "stripe").lower()
    if provider_str == "authorize_net" or provider_str == "authnet":
        return PaymentProvider.AUTHORIZE_NET
    return PaymentProvider.STRIPE


# =============================================================================
# Payment Handlers
# =============================================================================


@require_permission("payments:charge")
async def handle_charge(request: web.Request) -> web.Response:
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
    try:
        body = await request.json()
        provider = _get_provider_from_request(request, body)

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

    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    except Exception as e:
        logger.exception(f"Error processing charge: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def _charge_stripe(
    request: web.Request,
    amount: Decimal,
    currency: str,
    description: Optional[str],
    customer_id: Optional[str],
    payment_method: Any,
    metadata: Dict[str, Any],
) -> PaymentResult:
    """Process charge via Stripe."""
    connector = await get_stripe_connector(request)
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

        intent = await connector.create_payment_intent(
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
    description: Optional[str],
    payment_method: Any,
    metadata: Dict[str, Any],
) -> PaymentResult:
    """Process charge via Authorize.net."""
    connector = await get_authnet_connector(request)
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

        async with connector:
            result = await connector.charge(
                amount=amount,
                payment_method=card,
                billing_address=billing,
                description=description,
            )

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

    except Exception as e:
        return PaymentResult(
            transaction_id="",
            provider=PaymentProvider.AUTHORIZE_NET,
            status=PaymentStatus.ERROR,
            amount=amount,
            currency=currency,
            message=str(e),
        )


@require_permission("payments:authorize")
async def handle_authorize(request: web.Request) -> web.Response:
    """
    POST /api/payments/authorize

    Authorize a payment (capture later).
    """
    try:
        body = await request.json()
        provider = _get_provider_from_request(request, body)

        amount = Decimal(str(body.get("amount", 0)))
        if amount <= 0:
            return web.json_response({"error": "Amount must be greater than 0"}, status=400)

        currency = body.get("currency", "USD")
        payment_method = body.get("payment_method")

        if provider == PaymentProvider.AUTHORIZE_NET:
            connector = await get_authnet_connector(request)
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
            connector = await get_stripe_connector(request)
            if not connector:
                return web.json_response({"error": "Stripe connector not available"}, status=503)

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

        return web.json_response({"error": "Invalid request"}, status=400)

    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    except Exception as e:
        logger.exception(f"Error authorizing payment: {e}")
        return web.json_response({"error": str(e)}, status=500)


@require_permission("payments:capture")
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
    try:
        body = await request.json()
        provider = _get_provider_from_request(request, body)
        transaction_id = body.get("transaction_id")
        amount = body.get("amount")

        if not transaction_id:
            return web.json_response({"error": "Missing transaction_id"}, status=400)

        if provider == PaymentProvider.AUTHORIZE_NET:
            connector = await get_authnet_connector(request)
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
            connector = await get_stripe_connector(request)
            if not connector:
                return web.json_response({"error": "Stripe connector not available"}, status=503)

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

    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    except Exception as e:
        logger.exception(f"Error capturing payment: {e}")
        return web.json_response({"error": str(e)}, status=500)


@require_permission("payments:refund")
async def handle_refund(request: web.Request) -> web.Response:
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
    try:
        body = await request.json()
        provider = _get_provider_from_request(request, body)
        transaction_id = body.get("transaction_id")
        amount = Decimal(str(body.get("amount", 0)))

        if not transaction_id:
            return web.json_response({"error": "Missing transaction_id"}, status=400)
        if amount <= 0:
            return web.json_response({"error": "Amount must be greater than 0"}, status=400)

        if provider == PaymentProvider.AUTHORIZE_NET:
            connector = await get_authnet_connector(request)
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
            audit_data(
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
            connector = await get_stripe_connector(request)
            if not connector:
                return web.json_response({"error": "Stripe connector not available"}, status=503)

            refund = await connector.create_refund(
                payment_intent=transaction_id,
                amount=int(amount * 100),
            )

            # Audit the refund
            audit_data(
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

    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    except Exception as e:
        logger.exception(f"Error processing refund: {e}")
        audit_security(
            event_type="refund_error",
            actor_id=request.get("user_id", "unknown"),
            resource_type="payment",
            resource_id=body.get("transaction_id", "unknown"),
            reason=str(e),
        )
        return web.json_response({"error": str(e)}, status=500)


@require_permission("payments:void")
async def handle_void(request: web.Request) -> web.Response:
    """
    POST /api/payments/void

    Void a transaction.

    Request body:
    {
        "provider": "stripe" | "authorize_net",
        "transaction_id": "..."
    }
    """
    try:
        body = await request.json()
        provider = _get_provider_from_request(request, body)
        transaction_id = body.get("transaction_id")

        if not transaction_id:
            return web.json_response({"error": "Missing transaction_id"}, status=400)

        if provider == PaymentProvider.AUTHORIZE_NET:
            connector = await get_authnet_connector(request)
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
            connector = await get_stripe_connector(request)
            if not connector:
                return web.json_response({"error": "Stripe connector not available"}, status=503)

            intent = await connector.cancel_payment_intent(transaction_id)

            return web.json_response(
                {
                    "success": intent.status == "canceled",
                    "transaction_id": intent.id,
                    "status": intent.status,
                }
            )

    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    except Exception as e:
        logger.exception(f"Error voiding transaction: {e}")
        return web.json_response({"error": str(e)}, status=500)


@require_permission("payments:read")
async def handle_get_transaction(request: web.Request) -> web.Response:
    """
    GET /api/payments/transaction/{transaction_id}

    Get transaction details.
    """
    try:
        transaction_id = request.match_info.get("transaction_id")
        provider_str = request.query.get("provider", "stripe")

        if not transaction_id:
            return web.json_response({"error": "Missing transaction_id"}, status=400)

        provider = (
            PaymentProvider.AUTHORIZE_NET
            if provider_str in ("authorize_net", "authnet")
            else PaymentProvider.STRIPE
        )

        if provider == PaymentProvider.AUTHORIZE_NET:
            connector = await get_authnet_connector(request)
            if not connector:
                return web.json_response(
                    {"error": "Authorize.net connector not available"}, status=503
                )

            async with connector:
                details = await connector.get_transaction_details(transaction_id)

            if not details:
                return web.json_response({"error": "Transaction not found"}, status=404)

            return web.json_response({"transaction": details})
        else:
            connector = await get_stripe_connector(request)
            if not connector:
                return web.json_response({"error": "Stripe connector not available"}, status=503)

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
        return web.json_response({"error": str(e)}, status=500)


# =============================================================================
# Customer Profile Handlers
# =============================================================================


@require_permission("payments:customer:create")
async def handle_create_customer(request: web.Request) -> web.Response:
    """
    POST /api/payments/customer

    Create a customer profile.

    Request body:
    {
        "provider": "stripe" | "authorize_net",
        "email": "customer@example.com",
        "name": "John Doe",
        "merchant_customer_id": "cust_123",  // For Authorize.net
        "payment_method": {...}  // Optional
    }
    """
    try:
        body = await request.json()
        provider = _get_provider_from_request(request, body)

        email = body.get("email")
        name = body.get("name")

        if provider == PaymentProvider.AUTHORIZE_NET:
            connector = await get_authnet_connector(request)
            if not connector:
                return web.json_response(
                    {"error": "Authorize.net connector not available"}, status=503
                )

            merchant_customer_id = body.get("merchant_customer_id", str(uuid4())[:12])

            async with connector:
                profile = await connector.create_customer_profile(
                    merchant_customer_id=merchant_customer_id,
                    email=email,
                    description=name,
                )

            return web.json_response(
                {
                    "success": True,
                    "customer_id": profile.profile_id,
                    "merchant_customer_id": profile.merchant_customer_id,
                }
            )
        else:
            connector = await get_stripe_connector(request)
            if not connector:
                return web.json_response({"error": "Stripe connector not available"}, status=503)

            customer = await connector.create_customer(
                email=email,
                name=name,
                metadata=body.get("metadata", {}),
            )

            return web.json_response(
                {
                    "success": True,
                    "customer_id": customer.id,
                    "email": customer.email,
                }
            )

    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    except Exception as e:
        logger.exception(f"Error creating customer: {e}")
        return web.json_response({"error": str(e)}, status=500)


@require_permission("payments:customer:read")
async def handle_get_customer(request: web.Request) -> web.Response:
    """
    GET /api/payments/customer/{customer_id}

    Get customer profile.
    """
    try:
        customer_id = request.match_info.get("customer_id")
        provider_str = request.query.get("provider", "stripe")

        if not customer_id:
            return web.json_response({"error": "Missing customer_id"}, status=400)

        provider = (
            PaymentProvider.AUTHORIZE_NET
            if provider_str in ("authorize_net", "authnet")
            else PaymentProvider.STRIPE
        )

        if provider == PaymentProvider.AUTHORIZE_NET:
            connector = await get_authnet_connector(request)
            if not connector:
                return web.json_response(
                    {"error": "Authorize.net connector not available"}, status=503
                )

            async with connector:
                profile = await connector.get_customer_profile(customer_id)

            if not profile:
                return web.json_response({"error": "Customer not found"}, status=404)

            return web.json_response(
                {
                    "customer": {
                        "id": profile.profile_id,
                        "merchant_customer_id": profile.merchant_customer_id,
                        "email": profile.email,
                        "description": profile.description,
                        "payment_profiles": len(profile.payment_profiles or []),
                    }
                }
            )
        else:
            connector = await get_stripe_connector(request)
            if not connector:
                return web.json_response({"error": "Stripe connector not available"}, status=503)

            customer = await connector.retrieve_customer(customer_id)

            return web.json_response(
                {
                    "customer": {
                        "id": customer.id,
                        "email": customer.email,
                        "name": customer.name,
                        "created": customer.created,
                        "metadata": customer.metadata,
                    }
                }
            )

    except Exception as e:
        logger.exception(f"Error getting customer: {e}")
        return web.json_response({"error": str(e)}, status=500)


@require_permission("billing:delete")
async def handle_delete_customer(request: web.Request) -> web.Response:
    """
    DELETE /api/payments/customer/{customer_id}

    Delete customer profile.
    """
    try:
        customer_id = request.match_info.get("customer_id")
        provider_str = request.query.get("provider", "stripe")

        if not customer_id:
            return web.json_response({"error": "Missing customer_id"}, status=400)

        provider = (
            PaymentProvider.AUTHORIZE_NET
            if provider_str in ("authorize_net", "authnet")
            else PaymentProvider.STRIPE
        )

        if provider == PaymentProvider.AUTHORIZE_NET:
            connector = await get_authnet_connector(request)
            if not connector:
                return web.json_response(
                    {"error": "Authorize.net connector not available"}, status=503
                )

            async with connector:
                success = await connector.delete_customer_profile(customer_id)

            return web.json_response({"success": success})
        else:
            connector = await get_stripe_connector(request)
            if not connector:
                return web.json_response({"error": "Stripe connector not available"}, status=503)

            result = await connector.delete_customer(customer_id)

            return web.json_response({"success": result.deleted})

    except Exception as e:
        logger.exception(f"Error deleting customer: {e}")
        return web.json_response({"error": str(e)}, status=500)


# =============================================================================
# Subscription Handlers
# =============================================================================


@require_permission("payments:subscription:create")
async def handle_create_subscription(request: web.Request) -> web.Response:
    """
    POST /api/payments/subscription

    Create a subscription.

    Request body:
    {
        "provider": "stripe" | "authorize_net",
        "customer_id": "cus_...",
        "name": "Premium Plan",
        "amount": 99.99,
        "interval": "month",
        "interval_count": 1,
        "start_date": "2025-02-01"  // Optional
    }
    """
    try:
        body = await request.json()
        provider = _get_provider_from_request(request, body)

        customer_id = body.get("customer_id")
        name = body.get("name", "Subscription")
        amount = Decimal(str(body.get("amount", 0)))
        interval = body.get("interval", "month")
        interval_count = int(body.get("interval_count", 1))

        if not customer_id:
            return web.json_response({"error": "Missing customer_id"}, status=400)
        if amount <= 0:
            return web.json_response({"error": "Amount must be greater than 0"}, status=400)

        if provider == PaymentProvider.AUTHORIZE_NET:
            connector = await get_authnet_connector(request)
            if not connector:
                return web.json_response(
                    {"error": "Authorize.net connector not available"}, status=503
                )

            # Map interval to Authorize.net interval unit (string: "days" or "months")
            interval_unit = "months" if interval == "month" else "days"

            async with connector:
                subscription = await connector.create_subscription(
                    name=name,
                    amount=amount,
                    interval_length=interval_count,
                    interval_unit=interval_unit,
                    customer_profile_id=customer_id,
                )

            return web.json_response(
                {
                    "success": True,
                    "subscription_id": subscription.subscription_id,
                    "name": subscription.name,
                    "status": subscription.status.value if subscription.status else "active",
                }
            )
        else:
            connector = await get_stripe_connector(request)
            if not connector:
                return web.json_response({"error": "Stripe connector not available"}, status=503)

            # Create or get price
            price_id = body.get("price_id")
            if not price_id:
                # Create a price on the fly (simplified)
                return web.json_response(
                    {"error": "price_id required for Stripe subscriptions"}, status=400
                )

            subscription = await connector.create_subscription(
                customer=customer_id,
                items=[{"price": price_id}],
            )

            return web.json_response(
                {
                    "success": True,
                    "subscription_id": subscription.id,
                    "status": subscription.status,
                    "current_period_end": subscription.current_period_end,
                }
            )

    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    except Exception as e:
        logger.exception(f"Error creating subscription: {e}")
        return web.json_response({"error": str(e)}, status=500)


@require_permission("billing:cancel")
async def handle_cancel_subscription(request: web.Request) -> web.Response:
    """
    DELETE /api/payments/subscription/{subscription_id}

    Cancel a subscription.
    """
    try:
        subscription_id = request.match_info.get("subscription_id")
        provider_str = request.query.get("provider", "stripe")

        if not subscription_id:
            return web.json_response({"error": "Missing subscription_id"}, status=400)

        provider = (
            PaymentProvider.AUTHORIZE_NET
            if provider_str in ("authorize_net", "authnet")
            else PaymentProvider.STRIPE
        )

        if provider == PaymentProvider.AUTHORIZE_NET:
            connector = await get_authnet_connector(request)
            if not connector:
                return web.json_response(
                    {"error": "Authorize.net connector not available"}, status=503
                )

            async with connector:
                success = await connector.cancel_subscription(subscription_id)

            return web.json_response({"success": success})
        else:
            connector = await get_stripe_connector(request)
            if not connector:
                return web.json_response({"error": "Stripe connector not available"}, status=503)

            subscription = await connector.cancel_subscription(subscription_id)

            return web.json_response(
                {
                    "success": subscription.status == "canceled",
                    "subscription_id": subscription.id,
                    "status": subscription.status,
                }
            )

    except Exception as e:
        logger.exception(f"Error canceling subscription: {e}")
        return web.json_response({"error": str(e)}, status=500)


# =============================================================================
# Webhook Handlers
# =============================================================================


async def handle_stripe_webhook(request: web.Request) -> web.Response:
    """
    POST /api/payments/webhook/stripe

    Handle Stripe webhook events.
    """
    try:
        payload = await request.read()
        sig_header = request.headers.get("Stripe-Signature")

        connector = await get_stripe_connector(request)
        if not connector:
            return web.json_response({"error": "Stripe connector not available"}, status=503)

        # Verify webhook signature
        try:
            event = await connector.construct_webhook_event(payload, sig_header)
        except ValueError:
            return web.json_response({"error": "Invalid payload"}, status=400)
        except Exception as e:
            return web.json_response({"error": f"Signature verification failed: {e}"}, status=400)

        # Get event ID for idempotency check
        event_id = event.id
        if not event_id:
            logger.warning("Webhook event missing ID, cannot check idempotency")
        elif _is_duplicate_webhook(event_id):
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
            _mark_webhook_processed(event_id)

        return web.json_response({"received": True})

    except Exception as e:
        logger.exception(f"Error handling Stripe webhook: {e}")
        return web.json_response({"error": str(e)}, status=500)


async def handle_authnet_webhook(request: web.Request) -> web.Response:
    """
    POST /api/payments/webhook/authnet

    Handle Authorize.net webhook events.
    """
    try:
        payload = await request.json()
        signature = request.headers.get("X-ANET-Signature")

        connector = await get_authnet_connector(request)
        if not connector:
            return web.json_response({"error": "Authorize.net connector not available"}, status=503)

        # Verify webhook signature
        async with connector:
            if not await connector.verify_webhook_signature(payload, signature or ""):
                return web.json_response({"error": "Invalid signature"}, status=400)

        # Get event ID for idempotency check
        event_id = payload.get("notificationId") or payload.get("payload", {}).get("id")
        if not event_id:
            # Generate deterministic ID from payload if not provided
            import hashlib

            payload_str = json.dumps(payload, sort_keys=True)
            event_id = f"authnet_{hashlib.sha256(payload_str.encode()).hexdigest()[:16]}"

        if _is_duplicate_webhook(event_id):
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
        _mark_webhook_processed(event_id)

        return web.json_response({"received": True})

    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON body"}, status=400)
    except Exception as e:
        logger.exception(f"Error handling Authorize.net webhook: {e}")
        return web.json_response({"error": str(e)}, status=500)


# =============================================================================
# Route Registration
# =============================================================================


def register_payment_routes(app: web.Application) -> None:
    """Register payment routes with the application."""
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
    app.router.add_delete("/api/payments/customer/{customer_id}", handle_delete_customer)

    # Subscription management
    app.router.add_post("/api/payments/subscription", handle_create_subscription)
    app.router.add_delete(
        "/api/payments/subscription/{subscription_id}", handle_cancel_subscription
    )

    # Webhooks
    app.router.add_post("/api/payments/webhook/stripe", handle_stripe_webhook)
    app.router.add_post("/api/payments/webhook/authnet", handle_authnet_webhook)


__all__ = [
    "register_payment_routes",
    "handle_charge",
    "handle_authorize",
    "handle_capture",
    "handle_refund",
    "handle_void",
    "handle_get_transaction",
    "handle_create_customer",
    "handle_get_customer",
    "handle_delete_customer",
    "handle_create_subscription",
    "handle_cancel_subscription",
    "handle_stripe_webhook",
    "handle_authnet_webhook",
    "PaymentProvider",
    "PaymentStatus",
    "PaymentRequest",
    "PaymentResult",
]
