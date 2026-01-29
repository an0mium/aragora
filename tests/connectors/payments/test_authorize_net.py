"""
Tests for Authorize.net Payment Connector.

Tests cover:
- Enum values
- Credentials initialization
- Data model parsing
- Connector configuration
- Webhook verification
- Transaction operations (mocked)
- Customer profiles (mocked)
- Subscriptions (mocked)
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import hashlib
import hmac
import json

import pytest


# =============================================================================
# Enum Tests
# =============================================================================


class TestAuthorizeNetEnums:
    """Tests for Authorize.net enums."""

    def test_environment_values(self):
        """AuthorizeNetEnvironment enum has expected values."""
        from aragora.connectors.payments.authorize_net import AuthorizeNetEnvironment

        assert AuthorizeNetEnvironment.SANDBOX.value == "sandbox"
        assert AuthorizeNetEnvironment.PRODUCTION.value == "production"

    def test_transaction_type_values(self):
        """TransactionType enum has expected values."""
        from aragora.connectors.payments.authorize_net import TransactionType

        assert TransactionType.AUTH_CAPTURE.value == "authCaptureTransaction"
        assert TransactionType.AUTH_ONLY.value == "authOnlyTransaction"
        assert TransactionType.CAPTURE_ONLY.value == "captureOnlyTransaction"
        assert TransactionType.REFUND.value == "refundTransaction"
        assert TransactionType.VOID.value == "voidTransaction"
        assert TransactionType.PRIOR_AUTH_CAPTURE.value == "priorAuthCaptureTransaction"

    def test_transaction_status_values(self):
        """TransactionStatus enum has expected values."""
        from aragora.connectors.payments.authorize_net import TransactionStatus

        assert TransactionStatus.APPROVED.value == "approved"
        assert TransactionStatus.DECLINED.value == "declined"
        assert TransactionStatus.ERROR.value == "error"
        assert TransactionStatus.HELD_FOR_REVIEW.value == "held_for_review"
        assert TransactionStatus.PENDING.value == "pending"
        assert TransactionStatus.VOIDED.value == "voided"
        assert TransactionStatus.REFUNDED.value == "refunded"

    def test_payment_method_type_values(self):
        """PaymentMethodType enum has expected values."""
        from aragora.connectors.payments.authorize_net import PaymentMethodType

        assert PaymentMethodType.CREDIT_CARD.value == "credit_card"
        assert PaymentMethodType.BANK_ACCOUNT.value == "bank_account"
        assert PaymentMethodType.APPLE_PAY.value == "apple_pay"
        assert PaymentMethodType.GOOGLE_PAY.value == "google_pay"

    def test_card_type_values(self):
        """CardType enum has expected values."""
        from aragora.connectors.payments.authorize_net import CardType

        assert CardType.VISA.value == "Visa"
        assert CardType.MASTERCARD.value == "MasterCard"
        assert CardType.AMEX.value == "AmericanExpress"
        assert CardType.DISCOVER.value == "Discover"
        assert CardType.JCB.value == "JCB"
        assert CardType.DINERS.value == "DinersClub"
        assert CardType.UNKNOWN.value == "Unknown"


# =============================================================================
# Credentials Tests
# =============================================================================


class TestAuthorizeNetCredentials:
    """Tests for AuthorizeNetCredentials."""

    def test_credentials_init(self):
        """Create credentials with required fields."""
        from aragora.connectors.payments.authorize_net import (
            AuthorizeNetCredentials,
            AuthorizeNetEnvironment,
        )

        creds = AuthorizeNetCredentials(
            api_login_id="test_login",
            transaction_key="test_key",
        )

        assert creds.api_login_id == "test_login"
        assert creds.transaction_key == "test_key"
        assert creds.environment == AuthorizeNetEnvironment.SANDBOX
        assert creds.signature_key is None

    def test_credentials_with_environment(self):
        """Create credentials with custom environment."""
        from aragora.connectors.payments.authorize_net import (
            AuthorizeNetCredentials,
            AuthorizeNetEnvironment,
        )

        creds = AuthorizeNetCredentials(
            api_login_id="prod_login",
            transaction_key="prod_key",
            environment=AuthorizeNetEnvironment.PRODUCTION,
            signature_key="webhook_secret",
        )

        assert creds.api_login_id == "prod_login"
        assert creds.environment == AuthorizeNetEnvironment.PRODUCTION
        assert creds.signature_key == "webhook_secret"

    def test_credentials_from_env(self, monkeypatch):
        """Create credentials from environment variables."""
        from aragora.connectors.payments.authorize_net import (
            AuthorizeNetCredentials,
            AuthorizeNetEnvironment,
        )

        monkeypatch.setenv("AUTHORIZE_NET_API_LOGIN_ID", "env_login")
        monkeypatch.setenv("AUTHORIZE_NET_TRANSACTION_KEY", "env_key")
        monkeypatch.setenv("AUTHORIZE_NET_ENVIRONMENT", "production")
        monkeypatch.setenv("AUTHORIZE_NET_SIGNATURE_KEY", "sig_key")

        creds = AuthorizeNetCredentials.from_env()

        assert creds.api_login_id == "env_login"
        assert creds.transaction_key == "env_key"
        assert creds.environment == AuthorizeNetEnvironment.PRODUCTION
        assert creds.signature_key == "sig_key"

    def test_credentials_from_env_missing(self, monkeypatch):
        """Raise error when required env vars missing."""
        from aragora.connectors.payments.authorize_net import AuthorizeNetCredentials

        monkeypatch.delenv("AUTHORIZE_NET_API_LOGIN_ID", raising=False)
        monkeypatch.delenv("AUTHORIZE_NET_TRANSACTION_KEY", raising=False)

        with pytest.raises(ValueError, match="AUTHORIZE_NET_API_LOGIN_ID"):
            AuthorizeNetCredentials.from_env()

    def test_credentials_from_env_defaults(self, monkeypatch):
        """Use defaults when optional env vars missing."""
        from aragora.connectors.payments.authorize_net import (
            AuthorizeNetCredentials,
            AuthorizeNetEnvironment,
        )

        monkeypatch.setenv("AUTHORIZE_NET_API_LOGIN_ID", "login")
        monkeypatch.setenv("AUTHORIZE_NET_TRANSACTION_KEY", "key")
        monkeypatch.delenv("AUTHORIZE_NET_ENVIRONMENT", raising=False)
        monkeypatch.delenv("AUTHORIZE_NET_SIGNATURE_KEY", raising=False)

        creds = AuthorizeNetCredentials.from_env()

        assert creds.environment == AuthorizeNetEnvironment.SANDBOX
        assert creds.signature_key is None


# =============================================================================
# Data Model Tests - CreditCard
# =============================================================================


class TestCreditCard:
    """Tests for CreditCard dataclass."""

    def test_credit_card_basic(self):
        """Create credit card with required fields."""
        from aragora.connectors.payments.authorize_net import CreditCard

        card = CreditCard(
            card_number="4111111111111111",
            expiration_date="1225",
        )

        assert card.card_number == "4111111111111111"
        assert card.expiration_date == "1225"
        assert card.card_code is None

    def test_credit_card_with_cvv(self):
        """Create credit card with CVV."""
        from aragora.connectors.payments.authorize_net import CreditCard

        card = CreditCard(
            card_number="4111111111111111",
            expiration_date="1225",
            card_code="123",
        )

        assert card.card_code == "123"

    def test_credit_card_to_api(self):
        """Convert credit card to API format."""
        from aragora.connectors.payments.authorize_net import CreditCard

        card = CreditCard(
            card_number="4111111111111111",
            expiration_date="1225",
            card_code="123",
        )

        api_data = card.to_api()

        assert api_data["cardNumber"] == "4111111111111111"
        assert api_data["expirationDate"] == "1225"
        assert api_data["cardCode"] == "123"

    def test_credit_card_to_api_no_cvv(self):
        """Convert credit card without CVV to API format."""
        from aragora.connectors.payments.authorize_net import CreditCard

        card = CreditCard(
            card_number="4111111111111111",
            expiration_date="1225",
        )

        api_data = card.to_api()

        assert "cardCode" not in api_data


# =============================================================================
# Data Model Tests - BankAccount
# =============================================================================


class TestBankAccount:
    """Tests for BankAccount dataclass."""

    def test_bank_account_basic(self):
        """Create bank account with required fields."""
        from aragora.connectors.payments.authorize_net import BankAccount

        account = BankAccount(
            account_type="checking",
            routing_number="121000248",
            account_number="12345678",
            name_on_account="John Doe",
        )

        assert account.account_type == "checking"
        assert account.routing_number == "121000248"
        assert account.echeck_type == "WEB"

    def test_bank_account_to_api(self):
        """Convert bank account to API format."""
        from aragora.connectors.payments.authorize_net import BankAccount

        account = BankAccount(
            account_type="savings",
            routing_number="121000248",
            account_number="87654321",
            name_on_account="Jane Doe",
            echeck_type="CCD",
        )

        api_data = account.to_api()

        assert api_data["accountType"] == "savings"
        assert api_data["routingNumber"] == "121000248"
        assert api_data["accountNumber"] == "87654321"
        assert api_data["nameOnAccount"] == "Jane Doe"
        assert api_data["echeckType"] == "CCD"


# =============================================================================
# Data Model Tests - BillingAddress
# =============================================================================


class TestBillingAddress:
    """Tests for BillingAddress dataclass."""

    def test_billing_address_full(self):
        """Create billing address with all fields."""
        from aragora.connectors.payments.authorize_net import BillingAddress

        address = BillingAddress(
            first_name="John",
            last_name="Doe",
            company="Acme Inc",
            address="123 Main St",
            city="San Francisco",
            state="CA",
            zip_code="94105",
            country="USA",
            phone="415-555-1234",
            email="john@example.com",
        )

        assert address.first_name == "John"
        assert address.last_name == "Doe"
        assert address.company == "Acme Inc"

    def test_billing_address_to_api(self):
        """Convert billing address to API format."""
        from aragora.connectors.payments.authorize_net import BillingAddress

        address = BillingAddress(
            first_name="John",
            last_name="Doe",
            address="123 Main St",
            city="San Francisco",
            state="CA",
            zip_code="94105",
        )

        api_data = address.to_api()

        assert api_data["firstName"] == "John"
        assert api_data["lastName"] == "Doe"
        assert api_data["address"] == "123 Main St"
        assert api_data["city"] == "San Francisco"
        assert api_data["state"] == "CA"
        assert api_data["zip"] == "94105"

    def test_billing_address_to_api_minimal(self):
        """Convert minimal billing address to API format."""
        from aragora.connectors.payments.authorize_net import BillingAddress

        address = BillingAddress(first_name="John")

        api_data = address.to_api()

        assert api_data == {"firstName": "John"}

    def test_billing_address_empty(self):
        """Empty billing address produces empty dict."""
        from aragora.connectors.payments.authorize_net import BillingAddress

        address = BillingAddress()

        api_data = address.to_api()

        assert api_data == {}


# =============================================================================
# Data Model Tests - TransactionResult
# =============================================================================


class TestTransactionResult:
    """Tests for TransactionResult dataclass."""

    def test_transaction_result_from_api_approved(self):
        """Parse approved transaction from API response."""
        from aragora.connectors.payments.authorize_net import (
            TransactionResult,
            TransactionStatus,
        )

        data = {
            "transactionResponse": {
                "responseCode": "1",
                "transId": "123456789",
                "authCode": "ABC123",
                "avsResultCode": "Y",
                "cvvResultCode": "M",
                "accountNumber": "XXXX1111",
                "accountType": "Visa",
            },
            "messages": {
                "resultCode": "Ok",
                "message": [{"code": "I00001", "text": "Successful."}],
            },
        }

        result = TransactionResult.from_api(data)

        assert result.transaction_id == "123456789"
        assert result.response_code == "1"
        assert result.auth_code == "ABC123"
        assert result.avs_result == "Y"
        assert result.cvv_result == "M"
        assert result.account_number == "XXXX1111"
        assert result.status == TransactionStatus.APPROVED
        assert result.errors == []

    def test_transaction_result_from_api_declined(self):
        """Parse declined transaction from API response."""
        from aragora.connectors.payments.authorize_net import (
            TransactionResult,
            TransactionStatus,
        )

        data = {
            "transactionResponse": {
                "responseCode": "2",
                "transId": "987654321",
                "errors": [{"errorCode": "2", "errorText": "This transaction has been declined."}],
            },
            "messages": {
                "resultCode": "Error",
                "message": [{"code": "E00027", "text": "Transaction declined."}],
            },
        }

        result = TransactionResult.from_api(data)

        assert result.transaction_id == "987654321"
        assert result.response_code == "2"
        assert result.status == TransactionStatus.DECLINED
        assert len(result.errors) == 1
        assert "declined" in result.errors[0].lower()

    def test_transaction_result_from_api_held(self):
        """Parse held for review transaction from API response."""
        from aragora.connectors.payments.authorize_net import (
            TransactionResult,
            TransactionStatus,
        )

        data = {
            "transactionResponse": {
                "responseCode": "4",
                "transId": "111222333",
            },
            "messages": {
                "resultCode": "Ok",
                "message": [{"code": "I00004", "text": "Held for review."}],
            },
        }

        result = TransactionResult.from_api(data)

        assert result.status == TransactionStatus.HELD_FOR_REVIEW

    def test_transaction_result_from_api_error(self):
        """Parse error transaction from API response."""
        from aragora.connectors.payments.authorize_net import (
            TransactionResult,
            TransactionStatus,
        )

        data = {
            "transactionResponse": {
                "responseCode": "3",
                "transId": "",
                "errors": [{"errorCode": "6", "errorText": "Invalid card number."}],
            },
            "messages": {
                "resultCode": "Error",
                "message": [{"code": "E00027", "text": "An error occurred."}],
            },
        }

        result = TransactionResult.from_api(data)

        assert result.status == TransactionStatus.ERROR
        assert len(result.errors) == 1

    def test_transaction_result_to_dict(self):
        """Convert transaction result to dict."""
        from aragora.connectors.payments.authorize_net import (
            TransactionResult,
            TransactionStatus,
        )

        result = TransactionResult(
            transaction_id="123",
            response_code="1",
            message_code="I00001",
            message="Successful.",
            auth_code="ABC",
            status=TransactionStatus.APPROVED,
        )

        data = result.to_dict()

        assert data["transaction_id"] == "123"
        assert data["response_code"] == "1"
        assert data["auth_code"] == "ABC"
        assert data["status"] == "approved"


# =============================================================================
# Data Model Tests - CustomerProfile
# =============================================================================


class TestCustomerProfile:
    """Tests for CustomerProfile dataclass."""

    def test_customer_profile_from_api(self):
        """Parse customer profile from API response."""
        from aragora.connectors.payments.authorize_net import CustomerProfile

        data = {
            "profile": {
                "customerProfileId": "12345",
                "merchantCustomerId": "CUST001",
                "email": "customer@example.com",
                "description": "Premium customer",
                "paymentProfiles": [{"paymentProfileId": "pp_123"}],
                "shipToList": [{"customerAddressId": "addr_123"}],
            }
        }

        profile = CustomerProfile.from_api(data)

        assert profile.profile_id == "12345"
        assert profile.merchant_customer_id == "CUST001"
        assert profile.email == "customer@example.com"
        assert profile.description == "Premium customer"
        assert len(profile.payment_profiles) == 1
        assert len(profile.shipping_addresses) == 1

    def test_customer_profile_from_api_minimal(self):
        """Parse customer profile with minimal data."""
        from aragora.connectors.payments.authorize_net import CustomerProfile

        data = {
            "customerProfileId": "999",
            "merchantCustomerId": "MIN",
        }

        profile = CustomerProfile.from_api(data)

        assert profile.profile_id == "999"
        assert profile.merchant_customer_id == "MIN"
        assert profile.email is None
        assert profile.payment_profiles == []

    def test_customer_profile_to_dict(self):
        """Convert customer profile to dict."""
        from aragora.connectors.payments.authorize_net import CustomerProfile

        profile = CustomerProfile(
            profile_id="123",
            merchant_customer_id="CUST",
            email="test@example.com",
            payment_profiles=[{"id": "pp1"}, {"id": "pp2"}],
        )

        data = profile.to_dict()

        assert data["profile_id"] == "123"
        assert data["merchant_customer_id"] == "CUST"
        assert data["email"] == "test@example.com"
        assert data["payment_profile_count"] == 2


# =============================================================================
# Data Model Tests - Subscription
# =============================================================================


class TestSubscription:
    """Tests for Subscription dataclass."""

    def test_subscription_creation(self):
        """Create subscription with all fields."""
        from aragora.connectors.payments.authorize_net import Subscription

        start = datetime(2024, 1, 15, tzinfo=timezone.utc)
        sub = Subscription(
            subscription_id="SUB123",
            name="Monthly Plan",
            status="active",
            amount=Decimal("29.99"),
            interval_length=1,
            interval_unit="months",
            start_date=start,
            total_occurrences=12,
        )

        assert sub.subscription_id == "SUB123"
        assert sub.name == "Monthly Plan"
        assert sub.amount == Decimal("29.99")
        assert sub.interval_unit == "months"

    def test_subscription_to_dict(self):
        """Convert subscription to dict."""
        from aragora.connectors.payments.authorize_net import Subscription

        start = datetime(2024, 1, 15, tzinfo=timezone.utc)
        sub = Subscription(
            subscription_id="SUB456",
            name="Annual Plan",
            status="active",
            amount=Decimal("299.99"),
            interval_length=1,
            interval_unit="months",
            start_date=start,
            total_occurrences=9999,
        )

        data = sub.to_dict()

        assert data["subscription_id"] == "SUB456"
        assert data["name"] == "Annual Plan"
        assert data["amount"] == 299.99
        assert data["start_date"] == start.isoformat()

    def test_subscription_to_dict_no_start_date(self):
        """Convert subscription without start date to dict."""
        from aragora.connectors.payments.authorize_net import Subscription

        sub = Subscription(
            subscription_id="SUB789",
            name="Basic Plan",
            status="pending",
            amount=Decimal("9.99"),
            interval_length=30,
            interval_unit="days",
        )

        data = sub.to_dict()

        assert data["start_date"] is None


# =============================================================================
# Connector Tests
# =============================================================================


class TestAuthorizeNetConnectorInit:
    """Tests for AuthorizeNetConnector initialization."""

    def test_connector_creation(self):
        """Create connector with credentials."""
        from aragora.connectors.payments.authorize_net import (
            AuthorizeNetConnector,
            AuthorizeNetCredentials,
        )

        creds = AuthorizeNetCredentials(
            api_login_id="test_login",
            transaction_key="test_key",
        )
        connector = AuthorizeNetConnector(creds)

        assert connector.credentials == creds
        assert connector._client is None

    def test_connector_api_url_sandbox(self):
        """Sandbox environment uses test API URL."""
        from aragora.connectors.payments.authorize_net import (
            AuthorizeNetConnector,
            AuthorizeNetCredentials,
            AuthorizeNetEnvironment,
        )

        creds = AuthorizeNetCredentials(
            api_login_id="test",
            transaction_key="key",
            environment=AuthorizeNetEnvironment.SANDBOX,
        )
        connector = AuthorizeNetConnector(creds)

        assert connector.api_url == "https://apitest.authorize.net/xml/v1/request.api"

    def test_connector_api_url_production(self):
        """Production environment uses live API URL."""
        from aragora.connectors.payments.authorize_net import (
            AuthorizeNetConnector,
            AuthorizeNetCredentials,
            AuthorizeNetEnvironment,
        )

        creds = AuthorizeNetCredentials(
            api_login_id="prod",
            transaction_key="key",
            environment=AuthorizeNetEnvironment.PRODUCTION,
        )
        connector = AuthorizeNetConnector(creds)

        assert connector.api_url == "https://api.authorize.net/xml/v1/request.api"

    def test_connector_get_auth(self):
        """Get authentication object for API requests."""
        from aragora.connectors.payments.authorize_net import (
            AuthorizeNetConnector,
            AuthorizeNetCredentials,
        )

        creds = AuthorizeNetCredentials(
            api_login_id="my_login",
            transaction_key="my_key",
        )
        connector = AuthorizeNetConnector(creds)

        auth = connector._get_auth()

        assert auth["name"] == "my_login"
        assert auth["transactionKey"] == "my_key"


# =============================================================================
# Webhook Tests
# =============================================================================


class TestWebhookVerification:
    """Tests for webhook signature verification."""

    def test_verify_webhook_signature_valid(self):
        """Verify valid webhook signature."""
        from aragora.connectors.payments.authorize_net import (
            AuthorizeNetConnector,
            AuthorizeNetCredentials,
        )

        signature_key = "test_signature_key"
        creds = AuthorizeNetCredentials(
            api_login_id="login",
            transaction_key="key",
            signature_key=signature_key,
        )
        connector = AuthorizeNetConnector(creds)

        payload = b'{"eventType": "net.authorize.payment.authcapture.created"}'
        expected_sig = (
            hmac.new(
                signature_key.encode(),
                payload,
                hashlib.sha512,
            )
            .hexdigest()
            .upper()
        )

        result = connector.verify_webhook_signature(payload, expected_sig)

        assert result is True

    def test_verify_webhook_signature_invalid(self):
        """Reject invalid webhook signature."""
        from aragora.connectors.payments.authorize_net import (
            AuthorizeNetConnector,
            AuthorizeNetCredentials,
        )

        creds = AuthorizeNetCredentials(
            api_login_id="login",
            transaction_key="key",
            signature_key="secret",
        )
        connector = AuthorizeNetConnector(creds)

        payload = b'{"eventType": "test"}'
        result = connector.verify_webhook_signature(payload, "invalid_signature")

        assert result is False

    def test_verify_webhook_signature_no_key(self):
        """Skip verification when no signature key configured."""
        from aragora.connectors.payments.authorize_net import (
            AuthorizeNetConnector,
            AuthorizeNetCredentials,
        )

        creds = AuthorizeNetCredentials(
            api_login_id="login",
            transaction_key="key",
        )
        connector = AuthorizeNetConnector(creds)

        result = connector.verify_webhook_signature(b"payload", "any_signature")

        assert result is True

    def test_parse_webhook(self):
        """Parse webhook payload."""
        from aragora.connectors.payments.authorize_net import (
            AuthorizeNetConnector,
            AuthorizeNetCredentials,
        )

        creds = AuthorizeNetCredentials(
            api_login_id="login",
            transaction_key="key",
        )
        connector = AuthorizeNetConnector(creds)

        payload = {
            "webhookId": "wh_123",
            "eventType": "net.authorize.payment.authcapture.created",
            "payload": {"id": "trans_123", "amount": "100.00"},
        }

        result = connector.parse_webhook(payload)

        assert result["webhook_id"] == "wh_123"
        assert result["event_type"] == "net.authorize.payment.authcapture.created"
        assert result["payload"]["id"] == "trans_123"
        assert "received_at" in result

    def test_parse_webhook_missing_fields(self):
        """Parse webhook with missing optional fields."""
        from aragora.connectors.payments.authorize_net import (
            AuthorizeNetConnector,
            AuthorizeNetCredentials,
        )

        creds = AuthorizeNetCredentials(
            api_login_id="login",
            transaction_key="key",
        )
        connector = AuthorizeNetConnector(creds)

        payload = {}

        result = connector.parse_webhook(payload)

        assert result["event_type"] == ""
        assert result["payload"] == {}
        assert result["webhook_id"]  # Generated UUID


# =============================================================================
# Transaction Operations Tests (Mocked)
# =============================================================================


@pytest.fixture
def mock_connector():
    """Create connector with mocked client."""
    from aragora.connectors.payments.authorize_net import (
        AuthorizeNetConnector,
        AuthorizeNetCredentials,
    )

    creds = AuthorizeNetCredentials(
        api_login_id="test_login",
        transaction_key="test_key",
    )
    connector = AuthorizeNetConnector(creds)
    connector._client = MagicMock()
    return connector


class TestChargeTransaction:
    """Tests for charge transaction."""

    @pytest.mark.asyncio
    async def test_charge_credit_card(self, mock_connector):
        """Charge a credit card."""
        from aragora.connectors.payments.authorize_net import (
            CreditCard,
            TransactionStatus,
        )

        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "transactionResponse": {
                    "responseCode": "1",
                    "transId": "12345",
                    "authCode": "AUTH",
                },
                "messages": {
                    "resultCode": "Ok",
                    "message": [{"code": "I00001", "text": "Successful."}],
                },
            }
        )
        mock_response.raise_for_status = MagicMock()
        mock_connector._client.post = AsyncMock(return_value=mock_response)

        card = CreditCard(card_number="4111111111111111", expiration_date="1225")
        result = await mock_connector.charge(
            amount=Decimal("99.99"),
            payment_method=card,
            order_id="ORD001",
        )

        assert result.transaction_id == "12345"
        assert result.status == TransactionStatus.APPROVED
        mock_connector._client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_charge_with_billing(self, mock_connector):
        """Charge with billing address."""
        from aragora.connectors.payments.authorize_net import (
            CreditCard,
            BillingAddress,
        )

        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "transactionResponse": {"responseCode": "1", "transId": "67890"},
                "messages": {"resultCode": "Ok", "message": [{}]},
            }
        )
        mock_response.raise_for_status = MagicMock()
        mock_connector._client.post = AsyncMock(return_value=mock_response)

        card = CreditCard(card_number="4111111111111111", expiration_date="1225")
        billing = BillingAddress(
            first_name="John",
            last_name="Doe",
            email="john@example.com",
        )

        result = await mock_connector.charge(
            amount=Decimal("50.00"),
            payment_method=card,
            billing=billing,
        )

        assert result.transaction_id == "67890"
        call_args = mock_connector._client.post.call_args
        payload = call_args[1]["json"]
        trans_req = payload["createTransactionRequest"]["transactionRequest"]
        assert "billTo" in trans_req
        assert "customer" in trans_req


class TestAuthorizeTransaction:
    """Tests for authorize transaction."""

    @pytest.mark.asyncio
    async def test_authorize(self, mock_connector):
        """Authorize a payment."""
        from aragora.connectors.payments.authorize_net import (
            CreditCard,
            TransactionStatus,
        )

        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "transactionResponse": {"responseCode": "1", "transId": "AUTH123"},
                "messages": {"resultCode": "Ok", "message": [{}]},
            }
        )
        mock_response.raise_for_status = MagicMock()
        mock_connector._client.post = AsyncMock(return_value=mock_response)

        card = CreditCard(card_number="4111111111111111", expiration_date="1225")
        result = await mock_connector.authorize(
            amount=Decimal("100.00"),
            payment_method=card,
        )

        assert result.transaction_id == "AUTH123"
        assert result.status == TransactionStatus.APPROVED


class TestCaptureTransaction:
    """Tests for capture transaction."""

    @pytest.mark.asyncio
    async def test_capture(self, mock_connector):
        """Capture a previously authorized transaction."""
        from aragora.connectors.payments.authorize_net import TransactionStatus

        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "transactionResponse": {"responseCode": "1", "transId": "CAP123"},
                "messages": {"resultCode": "Ok", "message": [{}]},
            }
        )
        mock_response.raise_for_status = MagicMock()
        mock_connector._client.post = AsyncMock(return_value=mock_response)

        result = await mock_connector.capture(
            transaction_id="AUTH123",
            amount=Decimal("100.00"),
        )

        assert result.transaction_id == "CAP123"
        assert result.status == TransactionStatus.APPROVED


class TestRefundTransaction:
    """Tests for refund transaction."""

    @pytest.mark.asyncio
    async def test_refund(self, mock_connector):
        """Refund a settled transaction."""
        from aragora.connectors.payments.authorize_net import TransactionStatus

        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "transactionResponse": {"responseCode": "1", "transId": "REF123"},
                "messages": {"resultCode": "Ok", "message": [{}]},
            }
        )
        mock_response.raise_for_status = MagicMock()
        mock_connector._client.post = AsyncMock(return_value=mock_response)

        result = await mock_connector.refund(
            transaction_id="ORIG123",
            amount=Decimal("50.00"),
            card_last_four="1111",
        )

        assert result.transaction_id == "REF123"
        assert result.status == TransactionStatus.APPROVED


class TestVoidTransaction:
    """Tests for void transaction."""

    @pytest.mark.asyncio
    async def test_void(self, mock_connector):
        """Void an unsettled transaction."""
        from aragora.connectors.payments.authorize_net import TransactionStatus

        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "transactionResponse": {"responseCode": "1", "transId": "VOID123"},
                "messages": {"resultCode": "Ok", "message": [{}]},
            }
        )
        mock_response.raise_for_status = MagicMock()
        mock_connector._client.post = AsyncMock(return_value=mock_response)

        result = await mock_connector.void(transaction_id="ORIG123")

        assert result.transaction_id == "VOID123"
        assert result.status == TransactionStatus.APPROVED


# =============================================================================
# Customer Profile Tests (Mocked)
# =============================================================================


class TestCreateCustomerProfile:
    """Tests for creating customer profiles."""

    @pytest.mark.asyncio
    async def test_create_customer_profile_basic(self, mock_connector):
        """Create basic customer profile."""
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "customerProfileId": "PROF123",
                "messages": {"resultCode": "Ok", "message": [{}]},
            }
        )
        mock_response.raise_for_status = MagicMock()
        mock_connector._client.post = AsyncMock(return_value=mock_response)

        profile = await mock_connector.create_customer_profile(
            merchant_customer_id="CUST001",
            email="customer@example.com",
        )

        assert profile.profile_id == "PROF123"
        assert profile.merchant_customer_id == "CUST001"
        assert profile.email == "customer@example.com"

    @pytest.mark.asyncio
    async def test_create_customer_profile_with_payment(self, mock_connector):
        """Create customer profile with payment method."""
        from aragora.connectors.payments.authorize_net import CreditCard, BillingAddress

        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "customerProfileId": "PROF456",
                "messages": {"resultCode": "Ok", "message": [{}]},
            }
        )
        mock_response.raise_for_status = MagicMock()
        mock_connector._client.post = AsyncMock(return_value=mock_response)

        card = CreditCard(card_number="4111111111111111", expiration_date="1225")
        billing = BillingAddress(first_name="John", last_name="Doe")

        profile = await mock_connector.create_customer_profile(
            merchant_customer_id="CUST002",
            payment_method=card,
            billing=billing,
        )

        assert profile.profile_id == "PROF456"


class TestGetCustomerProfile:
    """Tests for getting customer profiles."""

    @pytest.mark.asyncio
    async def test_get_customer_profile(self, mock_connector):
        """Get customer profile by ID."""
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "profile": {
                    "customerProfileId": "PROF789",
                    "merchantCustomerId": "CUST003",
                    "email": "test@example.com",
                    "paymentProfiles": [{"paymentProfileId": "PP1"}],
                    "shipToList": [],
                },
                "messages": {"resultCode": "Ok", "message": [{}]},
            }
        )
        mock_response.raise_for_status = MagicMock()
        mock_connector._client.post = AsyncMock(return_value=mock_response)

        profile = await mock_connector.get_customer_profile("PROF789")

        assert profile.profile_id == "PROF789"
        assert profile.email == "test@example.com"
        assert len(profile.payment_profiles) == 1


class TestDeleteCustomerProfile:
    """Tests for deleting customer profiles."""

    @pytest.mark.asyncio
    async def test_delete_customer_profile_success(self, mock_connector):
        """Delete customer profile successfully."""
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "messages": {"resultCode": "Ok", "message": [{}]},
            }
        )
        mock_response.raise_for_status = MagicMock()
        mock_connector._client.post = AsyncMock(return_value=mock_response)

        result = await mock_connector.delete_customer_profile("PROF123")

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_customer_profile_failure(self, mock_connector):
        """Delete customer profile fails."""
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "messages": {"resultCode": "Error", "message": [{}]},
            }
        )
        mock_response.raise_for_status = MagicMock()
        mock_connector._client.post = AsyncMock(return_value=mock_response)

        result = await mock_connector.delete_customer_profile("INVALID")

        assert result is False


# =============================================================================
# Subscription Tests (Mocked)
# =============================================================================


class TestCreateSubscription:
    """Tests for creating subscriptions."""

    @pytest.mark.asyncio
    async def test_create_subscription(self, mock_connector):
        """Create a recurring subscription."""
        from aragora.connectors.payments.authorize_net import (
            CreditCard,
            BillingAddress,
        )

        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "subscriptionId": "SUB123",
                "messages": {"resultCode": "Ok", "message": [{}]},
            }
        )
        mock_response.raise_for_status = MagicMock()
        mock_connector._client.post = AsyncMock(return_value=mock_response)

        card = CreditCard(card_number="4111111111111111", expiration_date="1225")
        billing = BillingAddress(first_name="John", last_name="Doe")
        start = datetime(2024, 2, 1, tzinfo=timezone.utc)

        sub = await mock_connector.create_subscription(
            name="Monthly Plan",
            amount=Decimal("29.99"),
            interval_length=1,
            interval_unit="months",
            start_date=start,
            payment_method=card,
            billing=billing,
        )

        assert sub.subscription_id == "SUB123"
        assert sub.name == "Monthly Plan"
        assert sub.amount == Decimal("29.99")
        assert sub.status == "active"

    @pytest.mark.asyncio
    async def test_create_subscription_with_trial(self, mock_connector):
        """Create subscription with trial period."""
        from aragora.connectors.payments.authorize_net import (
            CreditCard,
            BillingAddress,
        )

        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "subscriptionId": "SUB_TRIAL",
                "messages": {"resultCode": "Ok", "message": [{}]},
            }
        )
        mock_response.raise_for_status = MagicMock()
        mock_connector._client.post = AsyncMock(return_value=mock_response)

        card = CreditCard(card_number="4111111111111111", expiration_date="1225")
        billing = BillingAddress(first_name="Jane")
        start = datetime(2024, 3, 1, tzinfo=timezone.utc)

        sub = await mock_connector.create_subscription(
            name="Premium Plan",
            amount=Decimal("99.99"),
            interval_length=1,
            interval_unit="months",
            start_date=start,
            payment_method=card,
            billing=billing,
            trial_occurrences=1,
            trial_amount=Decimal("0.00"),
        )

        assert sub.subscription_id == "SUB_TRIAL"
        assert sub.trial_occurrences == 1
        assert sub.trial_amount == Decimal("0.00")


class TestCancelSubscription:
    """Tests for canceling subscriptions."""

    @pytest.mark.asyncio
    async def test_cancel_subscription_success(self, mock_connector):
        """Cancel subscription successfully."""
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "messages": {"resultCode": "Ok", "message": [{}]},
            }
        )
        mock_response.raise_for_status = MagicMock()
        mock_connector._client.post = AsyncMock(return_value=mock_response)

        result = await mock_connector.cancel_subscription("SUB123")

        assert result is True

    @pytest.mark.asyncio
    async def test_cancel_subscription_failure(self, mock_connector):
        """Cancel subscription fails."""
        mock_response = MagicMock()
        mock_response.text = json.dumps(
            {
                "messages": {"resultCode": "Error", "message": [{}]},
            }
        )
        mock_response.raise_for_status = MagicMock()
        mock_connector._client.post = AsyncMock(return_value=mock_response)

        result = await mock_connector.cancel_subscription("INVALID")

        assert result is False


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestConnectorContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_enter(self):
        """Enter context manager initializes client."""
        from aragora.connectors.payments.authorize_net import (
            AuthorizeNetConnector,
            AuthorizeNetCredentials,
        )

        creds = AuthorizeNetCredentials(
            api_login_id="test",
            transaction_key="key",
        )
        connector = AuthorizeNetConnector(creds)

        async with connector as ctx:
            assert ctx._client is not None

    @pytest.mark.asyncio
    async def test_context_manager_exit(self):
        """Exit context manager closes client."""
        from aragora.connectors.payments.authorize_net import (
            AuthorizeNetConnector,
            AuthorizeNetCredentials,
        )

        creds = AuthorizeNetCredentials(
            api_login_id="test",
            transaction_key="key",
        )
        connector = AuthorizeNetConnector(creds)

        async with connector:
            pass

        assert connector._client is None

    @pytest.mark.asyncio
    async def test_request_without_context(self):
        """Request without context manager raises error."""
        from aragora.connectors.payments.authorize_net import (
            AuthorizeNetConnector,
            AuthorizeNetCredentials,
        )

        creds = AuthorizeNetCredentials(
            api_login_id="test",
            transaction_key="key",
        )
        connector = AuthorizeNetConnector(creds)

        with pytest.raises(RuntimeError, match="not initialized"):
            await connector._request({"test": {}})


# =============================================================================
# Module Import Tests
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_importable(self):
        """All __all__ exports are importable."""
        from aragora.connectors.payments.authorize_net import (
            AuthorizeNetConnector,
            AuthorizeNetCredentials,
            AuthorizeNetEnvironment,
            TransactionType,
            TransactionStatus,
            TransactionResult,
            CreditCard,
            BankAccount,
            BillingAddress,
            CustomerProfile,
            Subscription,
            create_authorize_net_connector,
        )

        assert AuthorizeNetConnector is not None
        assert AuthorizeNetCredentials is not None
        assert AuthorizeNetEnvironment is not None
        assert TransactionType is not None
        assert TransactionStatus is not None
        assert TransactionResult is not None
        assert CreditCard is not None
        assert BankAccount is not None
        assert BillingAddress is not None
        assert CustomerProfile is not None
        assert Subscription is not None
        assert create_authorize_net_connector is not None
