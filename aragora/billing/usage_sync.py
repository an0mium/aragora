"""
Usage Sync Service for Stripe Metered Billing.

Syncs usage data from local UsageTracker to Stripe for metered billing.
Runs periodically to report token usage and debate counts.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from aragora.billing.models import TIER_LIMITS, SubscriptionTier
from aragora.billing.stripe_client import (
    StripeAPIError,
    StripeClient,
    get_stripe_client,
)
from aragora.billing.usage import UsageTracker
from aragora.persistence.db_config import DatabaseType, get_db_path, get_nomic_dir

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
        nomic_dir: Optional[Path] = None,
    ):
        """
        Initialize usage sync service.

        Args:
            usage_tracker: Usage tracker instance
            stripe_client: Stripe client instance
            sync_interval: Seconds between sync operations
            nomic_dir: Base directory for databases (defaults to ARAGORA_DATA_DIR)
        """
        self.usage_tracker = usage_tracker or UsageTracker()
        self.stripe_client = stripe_client or get_stripe_client()
        self.sync_interval = sync_interval

        # Database path for persisting sync watermarks
        self._nomic_dir = nomic_dir or get_nomic_dir()
        self._db_path = get_db_path(DatabaseType.BILLING, self._nomic_dir)

        # Track last sync time per org
        self._last_sync: dict[str, datetime] = {}

        # Track synced usage to avoid double-reporting (persisted to DB)
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

        # Track last known billing period for auto-flush on transition
        self._last_billing_period: Optional[datetime] = None

        # Initialize database and load persisted state
        self._init_sync_db()
        self._load_sync_state()

        # Load last billing period from DB or use current
        # This ensures period transition detection works across restarts
        self._last_billing_period = self._load_last_billing_period()
        current_period = self._get_billing_period_start()

        # Check for missed period transition on startup
        if self._last_billing_period and current_period > self._last_billing_period:
            logger.info(
                f"Detected missed period transition during startup: "
                f"{self._last_billing_period} -> {current_period}"
            )
            # Flush remainders from the previous period
            flush_records = self._flush_previous_period(self._last_billing_period)
            if flush_records:
                logger.info(
                    f"Startup flush: billed {len(flush_records)} remainder records "
                    f"from previous period"
                )
            # Reload watermarks for new period
            self._load_sync_state()

        # Update persisted billing period
        self._last_billing_period = current_period
        self._save_last_billing_period(current_period)

    def _init_sync_db(self) -> None:
        """Initialize the sync watermark and sync records tables."""
        # Ensure parent directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self._db_path) as conn:
            # Watermarks table (existing)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_sync_watermarks (
                    org_id TEXT NOT NULL,
                    tokens_in INTEGER DEFAULT 0,
                    tokens_out INTEGER DEFAULT 0,
                    debates INTEGER DEFAULT 0,
                    period_start TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (org_id, period_start)
                )
            """)

            # Sync records table for two-phase commit (prevents double-billing on restart)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_sync_records (
                    id TEXT PRIMARY KEY,
                    org_id TEXT NOT NULL,
                    sync_type TEXT NOT NULL,
                    period_start TEXT NOT NULL,
                    quantity_reported INTEGER NOT NULL,
                    cumulative_total INTEGER NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL DEFAULT 'pending',
                    stripe_record_id TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    error TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sync_records_status
                ON usage_sync_records (status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_sync_records_org_period
                ON usage_sync_records (org_id, period_start)
            """)

            # Billing period state table (for detecting transitions across restarts)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS usage_sync_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def _load_last_billing_period(self) -> Optional[datetime]:
        """Load persisted last billing period from database."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute(
                    "SELECT value FROM usage_sync_state WHERE key = 'last_billing_period'"
                )
                row = cursor.fetchone()
                if row:
                    dt = datetime.fromisoformat(row[0])
                    # Ensure timezone awareness
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
        except sqlite3.Error as e:
            logger.warning(f"Error loading last billing period: {e}")
        return None

    def _save_last_billing_period(self, period: datetime) -> None:
        """Persist last billing period to database."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO usage_sync_state (key, value, updated_at)
                    VALUES ('last_billing_period', ?, ?)
                    """,
                    (period.isoformat(), datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error saving last billing period: {e}")

    def _load_sync_state(self) -> None:
        """Load sync watermarks from database for the current billing period."""
        period_start = self._get_billing_period_start().isoformat()

        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute(
                    """
                    SELECT org_id, tokens_in, tokens_out, debates
                    FROM usage_sync_watermarks
                    WHERE period_start = ?
                    """,
                    (period_start,),
                ).fetchall()

                for org_id, tokens_in, tokens_out, debates in rows:
                    self._synced_tokens_in[org_id] = tokens_in
                    self._synced_tokens_out[org_id] = tokens_out
                    self._synced_debates[org_id] = debates

                if rows:
                    logger.info(
                        f"Loaded sync watermarks for {len(rows)} orgs (period: {period_start})"
                    )

            # Reconcile any pending syncs from previous process (crash recovery)
            self._reconcile_pending_syncs()

        except sqlite3.Error as e:
            logger.warning(f"Failed to load sync state from database: {e}")

    def _get_idempotency_key(self, org_id: str, sync_type: str, cumulative_total: int) -> str:
        """Generate content-based idempotency key for Stripe.

        Unlike timestamp-based keys, this key is stable for the same usage amount
        within a billing period, preventing duplicate reports after restart.
        """
        period_start = self._get_billing_period_start().isoformat()
        return f"{org_id}-{sync_type}-{period_start}-{cumulative_total}"

    def _record_pending_sync(
        self,
        org_id: str,
        sync_type: str,
        quantity: int,
        cumulative_total: int,
        idempotency_key: str,
    ) -> str:
        """Record a pending sync operation BEFORE calling Stripe (phase 1).

        Returns the record ID for later completion or failure marking.
        """
        record_id = str(uuid4())
        period_start = self._get_billing_period_start().isoformat()
        created_at = datetime.now(timezone.utc).isoformat()

        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO usage_sync_records
                    (id, org_id, sync_type, period_start, quantity_reported,
                     cumulative_total, idempotency_key, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                    """,
                    (
                        record_id,
                        org_id,
                        sync_type,
                        period_start,
                        quantity,
                        cumulative_total,
                        idempotency_key,
                        created_at,
                    ),
                )
                conn.commit()
        except sqlite3.IntegrityError:
            # Idempotency key already exists - this is a duplicate attempt
            # Find and return existing record ID
            with sqlite3.connect(self._db_path) as conn:
                row = conn.execute(
                    "SELECT id FROM usage_sync_records WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if row:
                    return row[0]
            raise

        return record_id

    def _complete_sync(
        self,
        record_id: str,
        stripe_record_id: Optional[str],
        org_id: str,
        sync_type: str,
        cumulative_total: int,
    ) -> None:
        """Mark a sync operation as completed and update watermark (phase 2)."""
        completed_at = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(self._db_path) as conn:
            # Mark record as completed
            conn.execute(
                """
                UPDATE usage_sync_records
                SET status = 'completed', stripe_record_id = ?, completed_at = ?
                WHERE id = ?
                """,
                (stripe_record_id, completed_at, record_id),
            )

            # Update watermark atomically
            if sync_type == "tokens_input":
                self._synced_tokens_in[org_id] = cumulative_total
            elif sync_type == "tokens_output":
                self._synced_tokens_out[org_id] = cumulative_total
            elif sync_type == "debates":
                self._synced_debates[org_id] = cumulative_total

            conn.commit()

        # Persist watermarks
        self._save_sync_state(org_id)

    def _fail_sync(self, record_id: str, error: str) -> None:
        """Mark a sync operation as failed."""
        completed_at = datetime.now(timezone.utc).isoformat()

        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    UPDATE usage_sync_records
                    SET status = 'failed', error = ?, completed_at = ?
                    WHERE id = ?
                    """,
                    (error, completed_at, record_id),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to mark sync record {record_id} as failed: {e}")

    def _reconcile_pending_syncs(self) -> None:
        """Reconcile pending syncs from a crashed process on startup.

        Uses Stripe's idempotency behavior to safely verify pending syncs:
        - Re-attempts the usage report with the same idempotency key
        - If original call succeeded, Stripe returns the same response (no duplicate)
        - If original call never reached Stripe, creates the record now
        - Either way, we can safely mark as completed

        For each pending sync:
        - Get org's billing config to find subscription_item_id
        - Re-attempt with same idempotency key (safe due to Stripe idempotency)
        - Mark as completed or failed based on result
        """
        try:
            with sqlite3.connect(self._db_path) as conn:
                pending = conn.execute("""
                    SELECT id, org_id, sync_type, quantity_reported,
                           cumulative_total, idempotency_key
                    FROM usage_sync_records
                    WHERE status = 'pending'
                    """).fetchall()

            if not pending:
                return

            logger.info(f"Reconciling {len(pending)} pending sync records from previous run")

            for (
                record_id,
                org_id,
                sync_type,
                quantity,
                cumulative_total,
                idempotency_key,
            ) in pending:
                try:
                    # Get org's billing config
                    config = self._get_billing_config(org_id)
                    if not config:
                        logger.warning(
                            f"No billing config for org {org_id}, "
                            f"marking sync {record_id} as failed"
                        )
                        self._fail_sync(record_id, "No billing config found for organization")
                        continue

                    # Get subscription item ID for this sync type
                    item_id = self._get_subscription_item_id(config, sync_type)
                    if not item_id:
                        logger.warning(
                            f"No subscription item for {sync_type} in org {org_id}, "
                            f"marking sync {record_id} as failed"
                        )
                        self._fail_sync(record_id, f"No subscription item for {sync_type}")
                        continue

                    # Re-attempt with same idempotency key (safe - Stripe returns same response)
                    logger.info(
                        f"Verifying pending sync {record_id} for org {org_id} "
                        f"({sync_type}) via Stripe idempotency re-attempt"
                    )

                    usage_record = self.stripe_client.report_usage(
                        subscription_item_id=item_id,
                        quantity=quantity,
                        idempotency_key=idempotency_key,
                    )

                    # If we got here, either original succeeded or we just created it
                    # Either way, mark as completed
                    self._complete_sync(
                        record_id=record_id,
                        stripe_record_id=usage_record.id,
                        org_id=org_id,
                        sync_type=sync_type,
                        cumulative_total=cumulative_total,
                    )
                    logger.info(
                        f"Reconciled pending sync {record_id}: "
                        f"stripe_id={usage_record.id}, org={org_id}, type={sync_type}"
                    )

                except StripeAPIError as e:
                    logger.error(f"Failed to reconcile sync {record_id} for org {org_id}: {e}")
                    self._fail_sync(record_id, f"Stripe verification failed: {e}")

                except Exception as e:
                    logger.error(f"Unexpected error reconciling sync {record_id}: {e}")
                    self._fail_sync(record_id, f"Reconciliation error: {e}")

        except sqlite3.Error as e:
            logger.error(f"Failed to reconcile pending syncs: {e}")

    def _get_billing_config(self, org_id: str) -> Optional[OrgBillingConfig]:
        """Get billing configuration for an organization.

        Args:
            org_id: Organization ID

        Returns:
            OrgBillingConfig if registered and metered billing enabled, None otherwise
        """
        config = self._org_configs.get(org_id)
        if config and config.metered_enabled:
            return config
        return None

    def _get_subscription_item_id(self, config: OrgBillingConfig, sync_type: str) -> Optional[str]:
        """Get the subscription item ID for a sync type."""
        if sync_type == "tokens_input":
            return config.tokens_input_item_id
        elif sync_type == "tokens_output":
            return config.tokens_output_item_id
        elif sync_type == "debates":
            return config.debates_item_id
        return None

    def _save_sync_state(self, org_id: str) -> None:
        """Persist sync watermarks for an organization after successful sync."""
        period_start = self._get_billing_period_start().isoformat()
        updated_at = datetime.now(timezone.utc).isoformat()

        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO usage_sync_watermarks
                    (org_id, tokens_in, tokens_out, debates, period_start, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        org_id,
                        self._synced_tokens_in.get(org_id, 0),
                        self._synced_tokens_out.get(org_id, 0),
                        self._synced_debates.get(org_id, 0),
                        period_start,
                        updated_at,
                    ),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to save sync state for org {org_id}: {e}")

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

        Automatically detects billing period transitions and flushes
        remainder tokens from the previous period before starting new period.

        Returns:
            List of sync records for this run
        """
        records = []

        # Check for billing period transition and auto-flush previous period
        current_period = self._get_billing_period_start()
        if self._last_billing_period and current_period > self._last_billing_period:
            logger.info(
                f"Billing period transition detected: {self._last_billing_period} -> {current_period}"
            )
            # Flush remainder tokens from previous period before transitioning
            flush_records = self._flush_previous_period(self._last_billing_period)
            records.extend(flush_records)
            # Reload watermarks for new period (they'll be fresh/zero)
            self._load_sync_state()
            # Persist the new billing period
            self._save_last_billing_period(current_period)

        self._last_billing_period = current_period

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
        # Only advance watermark by what we actually bill (quantity * 1000)
        # to preserve remainder tokens for next sync or period-end flush
        if delta_tokens_in >= self.MIN_TOKENS_THRESHOLD and config.tokens_input_item_id:
            billed_units = delta_tokens_in // 1000
            billed_tokens = billed_units * 1000  # Actual tokens being billed
            record = self._report_usage(
                config=config,
                subscription_item_id=config.tokens_input_item_id,
                quantity=billed_units,  # Report per 1K tokens
                sync_type="tokens_input",
                cumulative_total=prev_tokens_in + billed_tokens,  # Only advance by billed amount
            )
            records.append(record)
            # Note: watermark update now handled by two-phase commit in _report_usage

        # Report output tokens
        # Only advance watermark by what we actually bill (quantity * 1000)
        if delta_tokens_out >= self.MIN_TOKENS_THRESHOLD and config.tokens_output_item_id:
            billed_units = delta_tokens_out // 1000
            billed_tokens = billed_units * 1000  # Actual tokens being billed
            record = self._report_usage(
                config=config,
                subscription_item_id=config.tokens_output_item_id,
                quantity=billed_units,  # Report per 1K tokens
                sync_type="tokens_output",
                cumulative_total=prev_tokens_out + billed_tokens,  # Only advance by billed amount
            )
            records.append(record)
            # Note: watermark update now handled by two-phase commit in _report_usage

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
                        cumulative_total=summary.total_debates,
                    )
                    records.append(record)
                    # Note: watermark update now handled by two-phase commit in _report_usage

        # Update last sync time
        self._last_sync[org_id] = datetime.now(timezone.utc)

        # Note: sync watermarks are now persisted atomically in _complete_sync
        # (two-phase commit ensures crash safety)

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
        cumulative_total: int,
    ) -> UsageSyncRecord:
        """
        Report usage to Stripe using two-phase commit for crash safety.

        Args:
            config: Organization billing config
            subscription_item_id: Stripe subscription item ID
            quantity: Usage quantity to report (delta)
            sync_type: Type of usage being synced
            cumulative_total: Total usage after this sync (for idempotency)

        Returns:
            Sync record with result
        """
        record = UsageSyncRecord(
            org_id=config.org_id,
            subscription_id=config.stripe_subscription_id,
            sync_type=sync_type,
            quantity=quantity,
        )

        # Phase 1: Generate content-based idempotency key and record pending sync
        idempotency_key = self._get_idempotency_key(config.org_id, sync_type, cumulative_total)

        try:
            # Record pending sync BEFORE calling Stripe
            pending_record_id = self._record_pending_sync(
                org_id=config.org_id,
                sync_type=sync_type,
                quantity=quantity,
                cumulative_total=cumulative_total,
                idempotency_key=idempotency_key,
            )
        except sqlite3.IntegrityError:
            # Idempotency key already exists - this was already synced
            logger.info(
                f"Skipping duplicate sync for {sync_type} (org={config.org_id}, "
                f"total={cumulative_total})"
            )
            record.success = True
            record.error = "Already synced (idempotency key exists)"
            return record

        try:
            # Phase 2: Call Stripe
            usage_record = self.stripe_client.report_usage(
                subscription_item_id=subscription_item_id,
                quantity=quantity,
                idempotency_key=idempotency_key,
            )

            # Phase 3: Mark completed and update watermark atomically
            self._complete_sync(
                record_id=pending_record_id,
                stripe_record_id=usage_record.id,
                org_id=config.org_id,
                sync_type=sync_type,
                cumulative_total=cumulative_total,
            )

            record.stripe_record_id = usage_record.id
            record.success = True

            logger.info(
                f"Reported {sync_type} usage for org {config.org_id}: "
                f"quantity={quantity}, total={cumulative_total}"
            )

        except StripeAPIError as e:
            # Mark sync as failed
            self._fail_sync(pending_record_id, str(e))
            record.success = False
            record.error = str(e)
            logger.error(f"Failed to report {sync_type} usage for org {config.org_id}: {e}")

        return record

    def _get_billing_period_start(self) -> datetime:
        """Get the start of the current billing period (first of month)."""
        now = datetime.now(timezone.utc)
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

    def _flush_previous_period(
        self,
        previous_period: datetime,
    ) -> list[UsageSyncRecord]:
        """
        Flush remainder tokens from a previous billing period.

        Called automatically during period transitions to ensure all usage
        from the previous period is billed before starting the new period.

        Args:
            previous_period: Start of the previous billing period

        Returns:
            List of sync records for flushed usage
        """
        records: list[UsageSyncRecord] = []

        for config in list(self._org_configs.values()):
            if not config.metered_enabled:
                continue

            org = config.org_id

            # Get usage summary for the previous period
            summary = self.usage_tracker.get_summary(
                org_id=org,
                period_start=previous_period,
            )

            # Get watermarks for the previous period from database
            prev_tokens_in = 0
            prev_tokens_out = 0

            try:
                with sqlite3.connect(self._db_path) as conn:
                    cursor = conn.execute(
                        """
                        SELECT tokens_in, tokens_out FROM usage_sync_watermarks
                        WHERE org_id = ? AND period_start = ?
                        """,
                        (org, previous_period.isoformat()),
                    )
                    row = cursor.fetchone()
                    if row:
                        prev_tokens_in, prev_tokens_out = row
            except sqlite3.Error as e:
                logger.warning(f"Error loading previous period watermarks for {org}: {e}")

            # Calculate remainder tokens
            remainder_in = summary.total_tokens_in - prev_tokens_in
            remainder_out = summary.total_tokens_out - prev_tokens_out

            # Bill remainder input tokens (round up to 1 unit minimum)
            if remainder_in > 0 and config.tokens_input_item_id:
                billed_units = (remainder_in + 999) // 1000
                record = self._report_usage(
                    config=config,
                    subscription_item_id=config.tokens_input_item_id,
                    quantity=billed_units,
                    sync_type="tokens_input",
                    cumulative_total=summary.total_tokens_in,
                )
                records.append(record)
                logger.info(
                    f"Period transition flush: billed {billed_units} units for "
                    f"{remainder_in} remainder input tokens for org {org} "
                    f"(period: {previous_period.isoformat()})"
                )

            # Bill remainder output tokens
            if remainder_out > 0 and config.tokens_output_item_id:
                billed_units = (remainder_out + 999) // 1000
                record = self._report_usage(
                    config=config,
                    subscription_item_id=config.tokens_output_item_id,
                    quantity=billed_units,
                    sync_type="tokens_output",
                    cumulative_total=summary.total_tokens_out,
                )
                records.append(record)
                logger.info(
                    f"Period transition flush: billed {billed_units} units for "
                    f"{remainder_out} remainder output tokens for org {org} "
                    f"(period: {previous_period.isoformat()})"
                )

        return records

    def flush_period(
        self,
        org_id: Optional[str] = None,
    ) -> list[UsageSyncRecord]:
        """
        Flush all remaining usage at end of billing period.

        This bills any remainder tokens that didn't meet the MIN_TOKENS_THRESHOLD
        during regular sync cycles. Should be called at billing period end.

        Args:
            org_id: Specific org to flush, or None for all orgs

        Returns:
            List of sync records for flushed usage
        """
        records: list[UsageSyncRecord] = []

        configs_to_flush = (
            [self._org_configs[org_id]]
            if org_id and org_id in self._org_configs
            else list(self._org_configs.values())
        )

        for config in configs_to_flush:
            if not config.metered_enabled:
                continue

            org = config.org_id
            summary = self.usage_tracker.get_summary(
                org_id=org,
                period_start=self._get_billing_period_start(),
            )

            # Calculate remainder tokens (what's left after regular syncs)
            prev_tokens_in = self._synced_tokens_in.get(org, 0)
            prev_tokens_out = self._synced_tokens_out.get(org, 0)

            remainder_in = summary.total_tokens_in - prev_tokens_in
            remainder_out = summary.total_tokens_out - prev_tokens_out

            # Bill remainder input tokens (even if < 1000, round up to 1 unit)
            if remainder_in > 0 and config.tokens_input_item_id:
                # Round up to nearest 1K unit for final flush
                billed_units = (remainder_in + 999) // 1000
                record = self._report_usage(
                    config=config,
                    subscription_item_id=config.tokens_input_item_id,
                    quantity=billed_units,
                    sync_type="tokens_input",
                    cumulative_total=summary.total_tokens_in,
                )
                records.append(record)
                logger.info(
                    f"Period flush: billed {billed_units} units for {remainder_in} "
                    f"remainder input tokens for org {org}"
                )

            # Bill remainder output tokens
            if remainder_out > 0 and config.tokens_output_item_id:
                billed_units = (remainder_out + 999) // 1000
                record = self._report_usage(
                    config=config,
                    subscription_item_id=config.tokens_output_item_id,
                    quantity=billed_units,
                    sync_type="tokens_output",
                    cumulative_total=summary.total_tokens_out,
                )
                records.append(record)
                logger.info(
                    f"Period flush: billed {billed_units} units for {remainder_out} "
                    f"remainder output tokens for org {org}"
                )

        return records


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
