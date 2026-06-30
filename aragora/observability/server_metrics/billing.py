"""
Billing Metrics for Aragora.

Tracks subscription events, revenue, and token usage.
"""

from __future__ import annotations

from .types import Counter, Gauge

# =============================================================================
# Billing Metrics
# =============================================================================

SUBSCRIPTION_EVENTS = Counter(
    name="aragora_subscription_events_total",
    help="Total subscription events by type and tier",
    label_names=["event", "tier"],
)

SUBSCRIPTION_ACTIVE = Gauge(
    name="aragora_subscriptions_active",
    help="Currently active subscriptions by tier",
    label_names=["tier"],
)

USAGE_DEBATES = Counter(
    name="aragora_debates_total",
    help="Total debates run by tier",
    label_names=["tier", "org_id"],
)

USAGE_TOKENS = Counter(
    name="aragora_tokens_total",
    help="Total tokens used by provider",
    label_names=["provider", "tier"],
)

BILLING_REVENUE = Counter(
    name="aragora_revenue_cents_total",
    help="Total revenue in cents by tier",
    label_names=["tier"],
)

PAYMENT_FAILURES = Counter(
    name="aragora_payment_failures_total",
    help="Payment failure count by tier",
    label_names=["tier"],
)


# =============================================================================
# Helpers
# =============================================================================


def track_subscription_event(event: str, tier: str) -> None:
    """Track a subscription event."""
    SUBSCRIPTION_EVENTS.inc(event=event, tier=tier)


def track_debate(tier: str, org_id: str) -> None:
    """Track a debate execution."""
    USAGE_DEBATES.inc(tier=tier, org_id=org_id)


def track_tokens(provider: str, tier: str, count: int) -> None:
    """Track token usage."""
    USAGE_TOKENS.inc(count, provider=provider, tier=tier)


__all__ = [
    "SUBSCRIPTION_EVENTS",
    "SUBSCRIPTION_ACTIVE",
    "USAGE_DEBATES",
    "USAGE_TOKENS",
    "BILLING_REVENUE",
    "PAYMENT_FAILURES",
    "track_subscription_event",
    "track_debate",
    "track_tokens",
]
