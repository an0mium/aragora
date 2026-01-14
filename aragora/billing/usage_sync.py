"""
Usage Sync Service for Stripe Metered Billing.

Syncs usage data from local UsageTracker to Stripe for metered billing.
Runs periodically to report token usage and debate counts.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from aragora.billing.models import TIER_LIMITS, SubscriptionTier
from aragora.billing.stripe_client import (
    StripeAPIError,
    StripeClient,
    get_stripe_client,
)
from aragora.billing.usage import UsageTracker

logger = logging.getLogger(__name__)


@dataclass
class UsageSyncRecord:
    """Record of a usage sync operation."""

    id: str = field(default_factory=lambda: str(uuid4()))
    org_id: str = ""
    subscription_id: str = ""
    sync_type: str = ""  # "tokens_input", "tokens_output", "debates"
    quantity: int = 0
    synced_at: datetime = field(default_factory=datetime.utcnow)
    stripe_record_id: str = ""
    success: bool = True
    error: str = ""


@dataclass
class OrgBillingConfig:
    """Billing configuration for an organization."""

    org_id: str
    stripe_customer_id: str
    stripe_subscription_id: str
    tier: SubscriptionTier
    metered_enabled: bool = False

    # Subscription item IDs for metered components
    tokens_input_item_id: Optional[str] = None
    tokens_output_item_id: Optional[str] = None
    debates_item_id: Optional[str] = None


class UsageSyncService:
    """
    Service that syncs usage data to Stripe for metered billing.

    Periodically checks for new usage and reports it to Stripe.
    Supports:
    - Token usage (input/output separately)
    - Debate count overages
    """

    # Sync interval in seconds (default: 5 minutes)
    DEFAULT_SYNC_INTERVAL = 300

    # Minimum tokens to trigger sync (to avoid tiny API calls)
    MIN_TOKENS_THRESHOLD = 1000

    def __init__(
        self,
        usage_tracker: Optional[UsageTracker] = None,
        stripe_client: Optional[StripeClient] = None,
        sync_interval: int = DEFAULT_SYNC_INTERVAL,
    ):
        """
        Initialize usage sync service.

        Args:
            usage_tracker: Usage tracker instance
            stripe_client: Stripe client instance
            sync_interval: Seconds between sync operations
        """
        self.usage_tracker = usage_tracker or UsageTracker()
        self.stripe_client = stripe_client or get_stripe_client()
        self.sync_interval = sync_interval

        # Track last sync time per org
        self._last_sync: dict[str, datetime] = {}

        # Track synced usage to avoid double-reporting
        self._synced_tokens_in: dict[str, int] = {}
        self._synced_tokens_out: dict[str, int] = {}
        self._synced_debates: dict[str, int] = {}

        # Background thread for periodic sync
        self._sync_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Org billing configs (cached)
        self._org_configs: dict[str, OrgBillingConfig] = {}

        # Sync history for debugging
        self._sync_history: list[UsageSyncRecord] = []
        self._max_history = 1000

    def register_org(self, config: OrgBillingConfig) -> None:
        """
        Register an organization for metered billing sync.

        Args:
            config: Organization billing configuration
        """
        self._org_configs[config.org_id] = config
        logger.info(f"Registered org {config.org_id} for usage sync")

    def unregister_org(self, org_id: str) -> None:
        """Remove an organization from sync."""
        self._org_configs.pop(org_id, None)
        self._last_sync.pop(org_id, None)
        self._synced_tokens_in.pop(org_id, None)
        self._synced_tokens_out.pop(org_id, None)
        self._synced_debates.pop(org_id, None)

    def sync_org_by_id(self, org_id: str) -> list[UsageSyncRecord]:
        """
        Sync usage for an organization by its ID.

        If the org is registered, syncs immediately. Otherwise returns empty.

        Args:
            org_id: Organization ID to sync

        Returns:
            List of sync records (empty if org not registered)
        """
        config = self._org_configs.get(org_id)
        if not config:
            logger.debug(f"Org {org_id} not registered for usage sync")
            return []

        if not config.metered_enabled:
            logger.debug(f"Org {org_id} does not have metered billing enabled")
            return []

        return self.sync_org(config)

    def start(self) -> None:
        """Start the background sync thread."""
        if self._sync_thread is not None and self._sync_thread.is_alive():
            return

        self._stop_event.clear()
        self._sync_thread = threading.Thread(
            target=self._sync_loop,
            name="usage-sync",
            daemon=True,
        )
        self._sync_thread.start()
        logger.info(f"Usage sync service started (interval={self.sync_interval}s)")

    def stop(self) -> None:
        """Stop the background sync thread."""
        self._stop_event.set()
        if self._sync_thread is not None:
            self._sync_thread.join(timeout=5.0)
            self._sync_thread = None
        logger.info("Usage sync service stopped")

    def _sync_loop(self) -> None:
        """Background loop that periodically syncs usage."""
        while not self._stop_event.is_set():
            if self._stop_event.wait(timeout=self.sync_interval):
                break
            try:
                self.sync_all()
            except Exception as e:
                logger.error(f"Error in usage sync loop: {e}")

    def sync_all(self) -> list[UsageSyncRecord]:
        """
        Sync usage for all registered organizations.

        Returns:
            List of sync records for this run
        """
        records = []
        for org_id, config in list(self._org_configs.items()):
            if not config.metered_enabled:
                continue
            try:
                org_records = self.sync_org(config)
                records.extend(org_records)
            except Exception as e:
                logger.error(f"Error syncing org {org_id}: {e}")
                records.append(
                    UsageSyncRecord(
                        org_id=org_id,
                        sync_type="error",
                        success=False,
                        error=str(e),
                    )
                )
        return records

    def sync_org(self, config: OrgBillingConfig) -> list[UsageSyncRecord]:
        """
        Sync usage for a single organization.

        Args:
            config: Organization billing configuration

        Returns:
            List of sync records
        """
        records = []
        org_id = config.org_id

        # Get current usage from tracker
        summary = self.usage_tracker.get_summary(
            org_id=org_id,
            period_start=self._get_billing_period_start(),
        )

        # Calculate delta since last sync
        prev_tokens_in = self._synced_tokens_in.get(org_id, 0)
        prev_tokens_out = self._synced_tokens_out.get(org_id, 0)
        prev_debates = self._synced_debates.get(org_id, 0)

        delta_tokens_in = summary.total_tokens_in - prev_tokens_in
        delta_tokens_out = summary.total_tokens_out - prev_tokens_out
        delta_debates = summary.total_debates - prev_debates

        # Report input tokens
        if delta_tokens_in >= self.MIN_TOKENS_THRESHOLD and config.tokens_input_item_id:
            record = self._report_usage(
                config=config,
                subscription_item_id=config.tokens_input_item_id,
                quantity=delta_tokens_in // 1000,  # Report per 1K tokens
                sync_type="tokens_input",
            )
            records.append(record)
            if record.success:
                self._synced_tokens_in[org_id] = summary.total_tokens_in

        # Report output tokens
        if delta_tokens_out >= self.MIN_TOKENS_THRESHOLD and config.tokens_output_item_id:
            record = self._report_usage(
                config=config,
                subscription_item_id=config.tokens_output_item_id,
                quantity=delta_tokens_out // 1000,  # Report per 1K tokens
                sync_type="tokens_output",
            )
            records.append(record)
            if record.success:
                self._synced_tokens_out[org_id] = summary.total_tokens_out

        # Report debate overages (only for tiers with limits)
        tier_limits = TIER_LIMITS.get(config.tier)
        if tier_limits and delta_debates > 0 and config.debates_item_id:
            debates_limit = tier_limits.debates_per_month
            if debates_limit > 0 and summary.total_debates > debates_limit:
                # Only report overage debates
                overage = max(0, summary.total_debates - debates_limit)
                prev_overage = max(0, prev_debates - debates_limit)
                delta_overage = overage - prev_overage

                if delta_overage > 0:
                    record = self._report_usage(
                        config=config,
                        subscription_item_id=config.debates_item_id,
                        quantity=delta_overage,
                        sync_type="debates",
                    )
                    records.append(record)
                    if record.success:
                        self._synced_debates[org_id] = summary.total_debates

        # Update last sync time
        self._last_sync[org_id] = datetime.utcnow()

        # Store records in history
        for record in records:
            self._add_to_history(record)

        return records

    def _report_usage(
        self,
        config: OrgBillingConfig,
        subscription_item_id: str,
        quantity: int,
        sync_type: str,
    ) -> UsageSyncRecord:
        """
        Report usage to Stripe.

        Args:
            config: Organization billing config
            subscription_item_id: Stripe subscription item ID
            quantity: Usage quantity to report
            sync_type: Type of usage being synced

        Returns:
            Sync record with result
        """
        record = UsageSyncRecord(
            org_id=config.org_id,
            subscription_id=config.stripe_subscription_id,
            sync_type=sync_type,
            quantity=quantity,
        )

        try:
            # Generate idempotency key to prevent duplicate reports
            idempotency_key = f"{config.org_id}-{sync_type}-{int(time.time())}"

            usage_record = self.stripe_client.report_usage(
                subscription_item_id=subscription_item_id,
                quantity=quantity,
                idempotency_key=idempotency_key,
            )

            record.stripe_record_id = usage_record.id
            record.success = True

            logger.info(f"Reported {sync_type} usage for org {config.org_id}: quantity={quantity}")

        except StripeAPIError as e:
            record.success = False
            record.error = str(e)
            logger.error(f"Failed to report {sync_type} usage for org {config.org_id}: {e}")

        return record

    def _get_billing_period_start(self) -> datetime:
        """Get the start of the current billing period (first of month)."""
        now = datetime.utcnow()
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    def _add_to_history(self, record: UsageSyncRecord) -> None:
        """Add a sync record to history (with size limit)."""
        self._sync_history.append(record)
        if len(self._sync_history) > self._max_history:
            self._sync_history = self._sync_history[-self._max_history :]

    def get_sync_history(
        self,
        org_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Get sync history records.

        Args:
            org_id: Filter by organization ID
            limit: Maximum records to return

        Returns:
            List of sync record dictionaries
        """
        records = self._sync_history
        if org_id:
            records = [r for r in records if r.org_id == org_id]
        records = records[-limit:]
        return [
            {
                "id": r.id,
                "org_id": r.org_id,
                "subscription_id": r.subscription_id,
                "sync_type": r.sync_type,
                "quantity": r.quantity,
                "synced_at": r.synced_at.isoformat(),
                "stripe_record_id": r.stripe_record_id,
                "success": r.success,
                "error": r.error,
            }
            for r in records
        ]

    def get_sync_status(self, org_id: str) -> dict[str, Any]:
        """
        Get current sync status for an organization.

        Args:
            org_id: Organization ID

        Returns:
            Status dict with last sync time and current totals
        """
        return {
            "org_id": org_id,
            "last_sync": (
                self._last_sync.get(org_id, datetime.min).isoformat()
                if org_id in self._last_sync
                else None
            ),
            "synced_tokens_in": self._synced_tokens_in.get(org_id, 0),
            "synced_tokens_out": self._synced_tokens_out.get(org_id, 0),
            "synced_debates": self._synced_debates.get(org_id, 0),
            "metered_enabled": org_id in self._org_configs
            and self._org_configs[org_id].metered_enabled,
        }


# Default service instance
_default_service: Optional[UsageSyncService] = None


def get_usage_sync_service() -> UsageSyncService:
    """Get the default usage sync service instance."""
    global _default_service
    if _default_service is None:
        _default_service = UsageSyncService()
    return _default_service


def start_usage_sync() -> UsageSyncService:
    """Start the usage sync service."""
    service = get_usage_sync_service()
    service.start()
    return service


def stop_usage_sync() -> None:
    """Stop the usage sync service."""
    global _default_service
    if _default_service is not None:
        _default_service.stop()


__all__ = [
    "UsageSyncService",
    "UsageSyncRecord",
    "OrgBillingConfig",
    "get_usage_sync_service",
    "start_usage_sync",
    "stop_usage_sync",
]
