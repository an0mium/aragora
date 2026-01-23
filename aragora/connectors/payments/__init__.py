"""
Payment Platform Connectors.

Integrations for payment processing:
- Stripe (payments, subscriptions, invoices)
- Authorize.net (transactions, CIM, ARB)
- PayPal - planned
- Square - planned
"""

from aragora.connectors.payments.stripe import (
    StripeConnector,
    StripeCredentials,
    StripeCustomer,
    StripeProduct,
    StripePrice,
    StripeSubscription,
    StripeInvoice,
    PaymentIntent,
    BalanceTransaction,
    StripeError,
    PaymentStatus,
    SubscriptionStatus,
    InvoiceStatus,
    PriceType,
    get_mock_customer,
    get_mock_subscription,
    get_mock_invoice,
)
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
    Subscription as AuthorizeNetSubscription,
    create_authorize_net_connector,
)

__all__ = [
    # Stripe
    "StripeConnector",
    "StripeCredentials",
    "StripeCustomer",
    "StripeProduct",
    "StripePrice",
    "StripeSubscription",
    "StripeInvoice",
    "PaymentIntent",
    "BalanceTransaction",
    "StripeError",
    "PaymentStatus",
    "SubscriptionStatus",
    "InvoiceStatus",
    "PriceType",
    "get_mock_customer",
    "get_mock_subscription",
    "get_mock_invoice",
    # Authorize.net
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
    "AuthorizeNetSubscription",
    "create_authorize_net_connector",
]
