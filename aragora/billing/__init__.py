"""
Aragora Billing Module.

Provides user management, organization handling, subscription tiers,
usage tracking, and Stripe integration for monetization.
"""

from aragora.billing.jwt_auth import (
    JWTPayload,
    TokenPair,
    UserAuthContext,
    create_access_token,
    create_refresh_token,
    create_token_pair,
    decode_jwt,
    extract_user_from_request,
    validate_access_token,
    validate_refresh_token,
)
from aragora.billing.models import (
    TIER_LIMITS,
    Organization,
    Subscription,
    SubscriptionTier,
    TierLimits,
    User,
)
from aragora.billing.notifications import (
    BillingNotifier,
    NotificationResult,
    get_billing_notifier,
)
from aragora.billing.stripe_client import (
    BillingPortalSession,
    CheckoutSession,
    StripeClient,
    StripeCustomer,
    StripeSubscription,
    get_stripe_client,
    parse_webhook_event,
    verify_webhook_signature,
)
from aragora.billing.usage import (
    UsageEvent,
    UsageSummary,
    UsageTracker,
)

__all__ = [
    # Models
    "User",
    "Organization",
    "Subscription",
    "SubscriptionTier",
    "TierLimits",
    "TIER_LIMITS",
    # Usage
    "UsageEvent",
    "UsageTracker",
    "UsageSummary",
    # JWT Auth
    "JWTPayload",
    "UserAuthContext",
    "TokenPair",
    "create_access_token",
    "create_refresh_token",
    "create_token_pair",
    "decode_jwt",
    "validate_access_token",
    "validate_refresh_token",
    "extract_user_from_request",
    # Stripe
    "StripeClient",
    "StripeCustomer",
    "StripeSubscription",
    "CheckoutSession",
    "BillingPortalSession",
    "get_stripe_client",
    "parse_webhook_event",
    "verify_webhook_signature",
    # Notifications
    "BillingNotifier",
    "NotificationResult",
    "get_billing_notifier",
]
