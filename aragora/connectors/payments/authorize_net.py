"""
Authorize.net Payment Connector.

Full integration with Authorize.net Payment Gateway:
- Customer profiles (CIM)
- Payment transactions
- Recurring billing (ARB)
- Fraud detection (AFDS)
- Webhooks for real-time events
- Transaction reporting

Environment Variables:
    AUTHORIZE_NET_API_LOGIN_ID - API Login ID
    AUTHORIZE_NET_TRANSACTION_KEY - Transaction Key
    AUTHORIZE_NET_ENVIRONMENT - 'sandbox' or 'production'
    AUTHORIZE_NET_SIGNATURE_KEY - Webhook signature key (optional)

Dependencies:
    pip install httpx
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)


class AuthorizeNetEnvironment(str, Enum):
    """Authorize.net environment."""

    SANDBOX = "sandbox"
    PRODUCTION = "production"


class TransactionType(str, Enum):
    """Transaction types."""

    AUTH_CAPTURE = "authCaptureTransaction"
    AUTH_ONLY = "authOnlyTransaction"
    CAPTURE_ONLY = "captureOnlyTransaction"
    REFUND = "refundTransaction"
    VOID = "voidTransaction"
    PRIOR_AUTH_CAPTURE = "priorAuthCaptureTransaction"


class TransactionStatus(str, Enum):
    """Transaction status."""

    APPROVED = "approved"
    DECLINED = "declined"
    ERROR = "error"
    HELD_FOR_REVIEW = "held_for_review"
    PENDING = "pending"
    VOIDED = "voided"
    REFUNDED = "refunded"


class PaymentMethodType(str, Enum):
    """Payment method types."""

    CREDIT_CARD = "credit_card"
    BANK_ACCOUNT = "bank_account"
    APPLE_PAY = "apple_pay"
    GOOGLE_PAY = "google_pay"


class CardType(str, Enum):
    """Credit card types."""

    VISA = "Visa"
    MASTERCARD = "MasterCard"
    AMEX = "AmericanExpress"
    DISCOVER = "Discover"
    JCB = "JCB"
    DINERS = "DinersClub"
    UNKNOWN = "Unknown"


@dataclass
class AuthorizeNetCredentials:
    """Authorize.net API credentials."""

    api_login_id: str
    transaction_key: str
    environment: AuthorizeNetEnvironment = AuthorizeNetEnvironment.SANDBOX
    signature_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> AuthorizeNetCredentials:
        """Create credentials from environment variables."""
        api_login_id = os.getenv("AUTHORIZE_NET_API_LOGIN_ID", "")
        transaction_key = os.getenv("AUTHORIZE_NET_TRANSACTION_KEY", "")
        env = os.getenv("AUTHORIZE_NET_ENVIRONMENT", "sandbox")
        signature_key = os.getenv("AUTHORIZE_NET_SIGNATURE_KEY")

        if not api_login_id or not transaction_key:
            raise ValueError(
                "AUTHORIZE_NET_API_LOGIN_ID and AUTHORIZE_NET_TRANSACTION_KEY required"
            )

        return cls(
            api_login_id=api_login_id,
            transaction_key=transaction_key,
            environment=AuthorizeNetEnvironment(env),
            signature_key=signature_key,
        )


@dataclass
class CreditCard:
    """Credit card payment method."""

    card_number: str
    expiration_date: str  # MMYY format
    card_code: Optional[str] = None  # CVV

    def to_api(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "cardNumber": self.card_number,
            "expirationDate": self.expiration_date,
        }
        if self.card_code:
            result["cardCode"] = self.card_code
        return result


@dataclass
class BankAccount:
    """Bank account payment method (ACH)."""

    account_type: str  # checking, savings, businessChecking
    routing_number: str
    account_number: str
    name_on_account: str
    echeck_type: str = "WEB"  # WEB, CCD, PPD, TEL, ARC, BOC

    def to_api(self) -> Dict[str, Any]:
        return {
            "accountType": self.account_type,
            "routingNumber": self.routing_number,
            "accountNumber": self.account_number,
            "nameOnAccount": self.name_on_account,
            "echeckType": self.echeck_type,
        }


@dataclass
class BillingAddress:
    """Billing address for transactions."""

    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company: Optional[str] = None
    address: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    country: Optional[str] = None
    phone: Optional[str] = None
    email: Optional[str] = None

    def to_api(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        if self.first_name:
            result["firstName"] = self.first_name
        if self.last_name:
            result["lastName"] = self.last_name
        if self.company:
            result["company"] = self.company
        if self.address:
            result["address"] = self.address
        if self.city:
            result["city"] = self.city
        if self.state:
            result["state"] = self.state
        if self.zip_code:
            result["zip"] = self.zip_code
        if self.country:
            result["country"] = self.country
        if self.phone:
            result["phoneNumber"] = self.phone
        return result


@dataclass
class TransactionResult:
    """Result of a transaction."""

    transaction_id: str
    response_code: str
    message_code: str
    message: str
    auth_code: Optional[str] = None
    avs_result: Optional[str] = None
    cvv_result: Optional[str] = None
    account_number: Optional[str] = None  # Masked
    account_type: Optional[str] = None
    status: TransactionStatus = TransactionStatus.PENDING
    errors: List[str] = field(default_factory=list)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> TransactionResult:
        """Parse API response into TransactionResult."""
        trans_response = data.get("transactionResponse", {})
        messages = data.get("messages", {})

        # Determine status from response code
        response_code = trans_response.get("responseCode", "0")
        if response_code == "1":
            status = TransactionStatus.APPROVED
        elif response_code == "2":
            status = TransactionStatus.DECLINED
        elif response_code == "4":
            status = TransactionStatus.HELD_FOR_REVIEW
        else:
            status = TransactionStatus.ERROR

        # Extract errors
        errors = []
        if "errors" in trans_response:
            for err in trans_response["errors"]:
                errors.append(f"{err.get('errorCode', '')}: {err.get('errorText', '')}")

        return cls(
            transaction_id=trans_response.get("transId", ""),
            response_code=response_code,
            message_code=messages.get("message", [{}])[0].get("code", ""),
            message=messages.get("message", [{}])[0].get("text", ""),
            auth_code=trans_response.get("authCode"),
            avs_result=trans_response.get("avsResultCode"),
            cvv_result=trans_response.get("cvvResultCode"),
            account_number=trans_response.get("accountNumber"),
            account_type=trans_response.get("accountType"),
            status=status,
            errors=errors,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transaction_id": self.transaction_id,
            "response_code": self.response_code,
            "message_code": self.message_code,
            "message": self.message,
            "auth_code": self.auth_code,
            "avs_result": self.avs_result,
            "cvv_result": self.cvv_result,
            "account_number": self.account_number,
            "account_type": self.account_type,
            "status": self.status.value,
            "errors": self.errors,
        }


@dataclass
class CustomerProfile:
    """Customer profile for recurring billing."""

    profile_id: str
    merchant_customer_id: str
    email: Optional[str] = None
    description: Optional[str] = None
    payment_profiles: List[Dict[str, Any]] = field(default_factory=list)
    shipping_addresses: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_api(cls, data: Dict[str, Any]) -> CustomerProfile:
        profile = data.get("profile", data)
        return cls(
            profile_id=profile.get("customerProfileId", ""),
            merchant_customer_id=profile.get("merchantCustomerId", ""),
            email=profile.get("email"),
            description=profile.get("description"),
            payment_profiles=profile.get("paymentProfiles", []),
            shipping_addresses=profile.get("shipToList", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "merchant_customer_id": self.merchant_customer_id,
            "email": self.email,
            "description": self.description,
            "payment_profile_count": len(self.payment_profiles),
            "shipping_address_count": len(self.shipping_addresses),
        }


@dataclass
class Subscription:
    """ARB subscription for recurring billing."""

    subscription_id: str
    name: str
    status: str
    amount: Decimal
    interval_length: int
    interval_unit: str  # days, months
    start_date: Optional[datetime] = None
    total_occurrences: Optional[int] = None
    trial_occurrences: Optional[int] = None
    trial_amount: Optional[Decimal] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subscription_id": self.subscription_id,
            "name": self.name,
            "status": self.status,
            "amount": float(self.amount),
            "interval_length": self.interval_length,
            "interval_unit": self.interval_unit,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "total_occurrences": self.total_occurrences,
        }


class AuthorizeNetConnector:
    """
    Authorize.net payment gateway connector.

    Supports:
    - One-time transactions (auth, capture, refund, void)
    - Customer Information Manager (CIM) for stored payment methods
    - Automated Recurring Billing (ARB) for subscriptions
    - Transaction reporting
    - Webhook signature verification
    """

    def __init__(self, credentials: Optional[AuthorizeNetCredentials] = None):
        """Initialize connector with credentials."""
        self.credentials = credentials or AuthorizeNetCredentials.from_env()
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def api_url(self) -> str:
        """Get API endpoint based on environment."""
        if self.credentials.environment == AuthorizeNetEnvironment.PRODUCTION:
            return "https://api.authorize.net/xml/v1/request.api"
        return "https://apitest.authorize.net/xml/v1/request.api"

    async def __aenter__(self) -> AuthorizeNetConnector:
        self._client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _get_auth(self) -> Dict[str, str]:
        """Get authentication object for API requests."""
        return {
            "name": self.credentials.api_login_id,
            "transactionKey": self.credentials.transaction_key,
        }

    async def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request."""
        if not self._client:
            raise RuntimeError("Connector not initialized. Use async with.")

        # Add authentication
        for key in payload:
            if isinstance(payload[key], dict):
                payload[key]["merchantAuthentication"] = self._get_auth()
                break

        response = await self._client.post(
            self.api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()

        # Parse response (Authorize.net returns JSON with BOM sometimes)
        text = response.text.lstrip("\ufeff")
        return json.loads(text)

    # =========================================================================
    # Transaction Methods
    # =========================================================================

    async def charge(
        self,
        amount: Decimal,
        payment_method: CreditCard | BankAccount,
        order_id: Optional[str] = None,
        description: Optional[str] = None,
        billing: Optional[BillingAddress] = None,
        customer_ip: Optional[str] = None,
    ) -> TransactionResult:
        """
        Charge a payment method (auth + capture).

        Args:
            amount: Amount to charge
            payment_method: Credit card or bank account
            order_id: Optional order/invoice ID
            description: Optional description
            billing: Optional billing address
            customer_ip: Customer IP for fraud detection
        """
        transaction_request: Dict[str, Any] = {
            "transactionType": TransactionType.AUTH_CAPTURE.value,
            "amount": str(amount),
        }

        # Add payment method
        if isinstance(payment_method, CreditCard):
            transaction_request["payment"] = {"creditCard": payment_method.to_api()}
        else:
            transaction_request["payment"] = {"bankAccount": payment_method.to_api()}

        # Add optional fields
        if order_id or description:
            transaction_request["order"] = {}
            if order_id:
                transaction_request["order"]["invoiceNumber"] = order_id
            if description:
                transaction_request["order"]["description"] = description

        if billing:
            transaction_request["billTo"] = billing.to_api()
            if billing.email:
                transaction_request["customer"] = {"email": billing.email}

        if customer_ip:
            transaction_request["customerIP"] = customer_ip

        payload = {"createTransactionRequest": {"transactionRequest": transaction_request}}

        result = await self._request(payload)
        return TransactionResult.from_api(result)

    async def authorize(
        self,
        amount: Decimal,
        payment_method: CreditCard | BankAccount,
        order_id: Optional[str] = None,
        billing: Optional[BillingAddress] = None,
    ) -> TransactionResult:
        """Authorize a payment (hold funds without capture)."""
        transaction_request: Dict[str, Any] = {
            "transactionType": TransactionType.AUTH_ONLY.value,
            "amount": str(amount),
        }

        if isinstance(payment_method, CreditCard):
            transaction_request["payment"] = {"creditCard": payment_method.to_api()}
        else:
            transaction_request["payment"] = {"bankAccount": payment_method.to_api()}

        if order_id:
            transaction_request["order"] = {"invoiceNumber": order_id}
        if billing:
            transaction_request["billTo"] = billing.to_api()

        payload = {"createTransactionRequest": {"transactionRequest": transaction_request}}
        result = await self._request(payload)
        return TransactionResult.from_api(result)

    async def capture(
        self, transaction_id: str, amount: Optional[Decimal] = None
    ) -> TransactionResult:
        """Capture a previously authorized transaction."""
        transaction_request: Dict[str, Any] = {
            "transactionType": TransactionType.PRIOR_AUTH_CAPTURE.value,
            "refTransId": transaction_id,
        }
        if amount:
            transaction_request["amount"] = str(amount)

        payload = {"createTransactionRequest": {"transactionRequest": transaction_request}}
        result = await self._request(payload)
        return TransactionResult.from_api(result)

    async def refund(
        self,
        transaction_id: str,
        amount: Decimal,
        card_last_four: str,
        expiration_date: Optional[str] = None,
    ) -> TransactionResult:
        """Refund a settled transaction."""
        transaction_request: Dict[str, Any] = {
            "transactionType": TransactionType.REFUND.value,
            "amount": str(amount),
            "refTransId": transaction_id,
            "payment": {
                "creditCard": {
                    "cardNumber": card_last_four,
                    "expirationDate": expiration_date or "XXXX",
                }
            },
        }

        payload = {"createTransactionRequest": {"transactionRequest": transaction_request}}
        result = await self._request(payload)
        return TransactionResult.from_api(result)

    async def void(self, transaction_id: str) -> TransactionResult:
        """Void an unsettled transaction."""
        transaction_request = {
            "transactionType": TransactionType.VOID.value,
            "refTransId": transaction_id,
        }

        payload = {"createTransactionRequest": {"transactionRequest": transaction_request}}
        result = await self._request(payload)
        return TransactionResult.from_api(result)

    async def get_transaction(self, transaction_id: str) -> Dict[str, Any]:
        """Get details of a specific transaction."""
        payload = {"getTransactionDetailsRequest": {"transId": transaction_id}}
        return await self._request(payload)

    # =========================================================================
    # Customer Profile Methods (CIM)
    # =========================================================================

    async def create_customer_profile(
        self,
        merchant_customer_id: str,
        email: Optional[str] = None,
        description: Optional[str] = None,
        payment_method: Optional[CreditCard | BankAccount] = None,
        billing: Optional[BillingAddress] = None,
    ) -> CustomerProfile:
        """Create a customer profile for storing payment methods."""
        profile: Dict[str, Any] = {"merchantCustomerId": merchant_customer_id}

        if email:
            profile["email"] = email
        if description:
            profile["description"] = description

        if payment_method:
            payment_profile: Dict[str, Any] = {}
            if isinstance(payment_method, CreditCard):
                payment_profile["payment"] = {"creditCard": payment_method.to_api()}
            else:
                payment_profile["payment"] = {"bankAccount": payment_method.to_api()}

            if billing:
                payment_profile["billTo"] = billing.to_api()

            profile["paymentProfiles"] = [payment_profile]

        payload = {
            "createCustomerProfileRequest": {
                "profile": profile,
                "validationMode": "testMode"
                if self.credentials.environment == AuthorizeNetEnvironment.SANDBOX
                else "liveMode",
            }
        }

        result = await self._request(payload)

        return CustomerProfile(
            profile_id=result.get("customerProfileId", ""),
            merchant_customer_id=merchant_customer_id,
            email=email,
            description=description,
        )

    async def get_customer_profile(self, profile_id: str) -> CustomerProfile:
        """Get a customer profile by ID."""
        payload = {
            "getCustomerProfileRequest": {
                "customerProfileId": profile_id,
                "includeIssuerInfo": "true",
            }
        }

        result = await self._request(payload)
        return CustomerProfile.from_api(result)

    async def delete_customer_profile(self, profile_id: str) -> bool:
        """Delete a customer profile."""
        payload = {"deleteCustomerProfileRequest": {"customerProfileId": profile_id}}
        result = await self._request(payload)
        return result.get("messages", {}).get("resultCode") == "Ok"

    async def charge_customer_profile(
        self,
        profile_id: str,
        payment_profile_id: str,
        amount: Decimal,
        order_id: Optional[str] = None,
    ) -> TransactionResult:
        """Charge a stored payment method."""
        transaction_request: Dict[str, Any] = {
            "transactionType": TransactionType.AUTH_CAPTURE.value,
            "amount": str(amount),
            "profile": {
                "customerProfileId": profile_id,
                "paymentProfile": {"paymentProfileId": payment_profile_id},
            },
        }

        if order_id:
            transaction_request["order"] = {"invoiceNumber": order_id}

        payload = {"createTransactionRequest": {"transactionRequest": transaction_request}}
        result = await self._request(payload)
        return TransactionResult.from_api(result)

    # =========================================================================
    # Subscription Methods (ARB)
    # =========================================================================

    async def create_subscription(
        self,
        name: str,
        amount: Decimal,
        interval_length: int,
        interval_unit: str,  # "days" or "months"
        start_date: datetime,
        payment_method: CreditCard | BankAccount,
        billing: BillingAddress,
        total_occurrences: int = 9999,
        trial_occurrences: int = 0,
        trial_amount: Optional[Decimal] = None,
    ) -> Subscription:
        """Create a recurring subscription."""
        subscription: Dict[str, Any] = {
            "name": name,
            "paymentSchedule": {
                "interval": {
                    "length": str(interval_length),
                    "unit": interval_unit,
                },
                "startDate": start_date.strftime("%Y-%m-%d"),
                "totalOccurrences": str(total_occurrences),
            },
            "amount": str(amount),
            "billTo": billing.to_api(),
        }

        if trial_occurrences > 0:
            subscription["paymentSchedule"]["trialOccurrences"] = str(trial_occurrences)
            if trial_amount:
                subscription["trialAmount"] = str(trial_amount)

        if isinstance(payment_method, CreditCard):
            subscription["payment"] = {"creditCard": payment_method.to_api()}
        else:
            subscription["payment"] = {"bankAccount": payment_method.to_api()}

        payload = {"ARBCreateSubscriptionRequest": {"subscription": subscription}}
        result = await self._request(payload)

        return Subscription(
            subscription_id=result.get("subscriptionId", ""),
            name=name,
            status="active",
            amount=amount,
            interval_length=interval_length,
            interval_unit=interval_unit,
            start_date=start_date,
            total_occurrences=total_occurrences,
            trial_occurrences=trial_occurrences if trial_occurrences > 0 else None,
            trial_amount=trial_amount,
        )

    async def cancel_subscription(self, subscription_id: str) -> bool:
        """Cancel a subscription."""
        payload = {"ARBCancelSubscriptionRequest": {"subscriptionId": subscription_id}}
        result = await self._request(payload)
        return result.get("messages", {}).get("resultCode") == "Ok"

    async def get_subscription(self, subscription_id: str) -> Dict[str, Any]:
        """Get subscription details."""
        payload = {
            "ARBGetSubscriptionRequest": {
                "subscriptionId": subscription_id,
                "includeTransactions": "true",
            }
        }
        return await self._request(payload)

    # =========================================================================
    # Webhook Methods
    # =========================================================================

    def verify_webhook_signature(self, payload: bytes, signature: str) -> bool:
        """Verify webhook signature."""
        if not self.credentials.signature_key:
            logger.warning("No signature key configured, skipping verification")
            return True

        expected = hmac.new(
            self.credentials.signature_key.encode(),
            payload,
            hashlib.sha512,
        ).hexdigest()

        return hmac.compare_digest(expected.upper(), signature.upper())

    def parse_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parse webhook payload into standardized format."""
        event_type = payload.get("eventType", "")
        webhook_id = payload.get("webhookId", str(uuid4()))

        return {
            "webhook_id": webhook_id,
            "event_type": event_type,
            "payload": payload.get("payload", {}),
            "received_at": datetime.now(timezone.utc).isoformat(),
        }

    # =========================================================================
    # Reporting Methods
    # =========================================================================

    async def get_settled_batch_list(
        self,
        first_settlement_date: datetime,
        last_settlement_date: datetime,
    ) -> List[Dict[str, Any]]:
        """Get list of settled batches."""
        payload = {
            "getSettledBatchListRequest": {
                "includeStatistics": "true",
                "firstSettlementDate": first_settlement_date.strftime("%Y-%m-%dT%H:%M:%S"),
                "lastSettlementDate": last_settlement_date.strftime("%Y-%m-%dT%H:%M:%S"),
            }
        }
        result = await self._request(payload)
        return result.get("batchList", [])

    async def get_transaction_list(self, batch_id: str) -> List[Dict[str, Any]]:
        """Get transactions in a settled batch."""
        payload = {"getTransactionListRequest": {"batchId": batch_id}}
        result = await self._request(payload)
        return result.get("transactions", [])


# Convenience function for quick setup
async def create_authorize_net_connector() -> AuthorizeNetConnector:
    """Create and initialize an Authorize.net connector from environment."""
    connector = AuthorizeNetConnector()
    await connector.__aenter__()
    return connector


__all__ = [
    "AuthorizeNetConnector",
    "AuthorizeNetCredentials",
    "AuthorizeNetEnvironment",
    "TransactionType",
    "TransactionStatus",
    "TransactionResult",
    "CreditCard",
    "BankAccount",
    "BillingAddress",
    "CustomerProfile",
    "Subscription",
    "create_authorize_net_connector",
]
