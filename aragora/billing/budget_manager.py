"""
Budget Manager - Comprehensive budget operations and enforcement.

Provides:
- Budget CRUD operations with persistence
- Real-time budget enforcement with hard stops
- Budget policy management
- Alert threshold configuration
- Budget override mechanism for authorized users
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class BudgetPeriod(Enum):
    """Budget period types."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    UNLIMITED = "unlimited"


class BudgetStatus(Enum):
    """Budget status states."""

    ACTIVE = "active"
    WARNING = "warning"  # 75%+ usage
    CRITICAL = "critical"  # 90%+ usage
    EXCEEDED = "exceeded"  # 100%+ usage
    SUSPENDED = "suspended"  # Auto-suspended due to exceed
    PAUSED = "paused"  # Manually paused
    CLOSED = "closed"  # Budget period ended


class BudgetAction(Enum):
    """Actions when budget thresholds are reached."""

    NOTIFY = "notify"  # Send alert only
    WARN = "warn"  # Warn users + slow down operations
    SOFT_LIMIT = "soft_limit"  # Warn + require confirmation
    HARD_LIMIT = "hard_limit"  # Block operations
    SUSPEND = "suspend"  # Suspend all operations


@dataclass
class BudgetThreshold:
    """Budget alert threshold configuration."""

    percentage: float  # 0.0 - 1.0
    action: BudgetAction
    notify_channels: List[str] = field(default_factory=lambda: ["email"])
    cooldown_minutes: int = 60  # Minimum time between alerts


# Default thresholds for new budgets
DEFAULT_THRESHOLDS = [
    BudgetThreshold(0.50, BudgetAction.NOTIFY),
    BudgetThreshold(0.75, BudgetAction.WARN),
    BudgetThreshold(0.90, BudgetAction.SOFT_LIMIT),
    BudgetThreshold(1.00, BudgetAction.HARD_LIMIT),
]


@dataclass
class Budget:
    """Budget configuration and state."""

    budget_id: str
    org_id: str
    name: str
    description: str = ""

    # Limits
    amount_usd: float = 0.0  # Total budget amount
    period: BudgetPeriod = BudgetPeriod.MONTHLY

    # Current usage
    spent_usd: float = 0.0
    period_start: float = 0.0  # Unix timestamp
    period_end: float = 0.0

    # Status
    status: BudgetStatus = BudgetStatus.ACTIVE
    auto_suspend: bool = True  # Auto-suspend on exceed

    # Thresholds (default: warn at 75%, critical at 90%, hard limit at 100%)
    thresholds: List[BudgetThreshold] = field(default_factory=lambda: list(DEFAULT_THRESHOLDS))

    # Overrides
    override_user_ids: List[str] = field(default_factory=list)  # Users who can bypass
    override_until: Optional[float] = None  # Temporary override expiry

    # Metadata
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    created_by: Optional[str] = None

    @property
    def usage_percentage(self) -> float:
        """Get current usage as percentage (0.0 - 1.0+)."""
        if self.amount_usd <= 0:
            return 0.0
        return self.spent_usd / self.amount_usd

    @property
    def remaining_usd(self) -> float:
        """Get remaining budget in USD."""
        return max(0.0, self.amount_usd - self.spent_usd)

    @property
    def is_exceeded(self) -> bool:
        """Check if budget is exceeded."""
        return self.spent_usd >= self.amount_usd and self.amount_usd > 0

    @property
    def current_action(self) -> BudgetAction:
        """Get the action for current usage level."""
        pct = self.usage_percentage
        action = BudgetAction.NOTIFY

        for threshold in sorted(self.thresholds, key=lambda t: t.percentage):
            if pct >= threshold.percentage:
                action = threshold.action

        return action

    def can_spend(self, amount_usd: float, user_id: Optional[str] = None) -> tuple[bool, str]:
        """Check if spending is allowed.

        Returns:
            Tuple of (allowed, reason)
        """
        # Check override
        if user_id and user_id in self.override_user_ids:
            if self.override_until is None or time.time() < self.override_until:
                return True, "Override active"

        # Check status
        if self.status == BudgetStatus.SUSPENDED:
            return False, "Budget suspended"
        if self.status == BudgetStatus.PAUSED:
            return False, "Budget paused"
        if self.status == BudgetStatus.CLOSED:
            return False, "Budget period closed"

        # Check if period expired
        if self.period_end > 0 and time.time() > self.period_end:
            return False, "Budget period expired"

        # Check hard limit
        new_total = self.spent_usd + amount_usd
        if new_total > self.amount_usd and self.amount_usd > 0:
            action = self.current_action
            if action == BudgetAction.HARD_LIMIT:
                return False, f"Budget exceeded (${self.spent_usd:.2f}/${self.amount_usd:.2f})"
            if action == BudgetAction.SUSPEND and self.auto_suspend:
                return False, "Budget auto-suspended"

        return True, "OK"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "budget_id": self.budget_id,
            "org_id": self.org_id,
            "name": self.name,
            "description": self.description,
            "amount_usd": self.amount_usd,
            "period": self.period.value,
            "spent_usd": self.spent_usd,
            "remaining_usd": self.remaining_usd,
            "usage_percentage": self.usage_percentage,
            "period_start": self.period_start,
            "period_start_iso": datetime.fromtimestamp(
                self.period_start, tz=timezone.utc
            ).isoformat()
            if self.period_start
            else None,
            "period_end": self.period_end,
            "period_end_iso": datetime.fromtimestamp(self.period_end, tz=timezone.utc).isoformat()
            if self.period_end
            else None,
            "status": self.status.value,
            "current_action": self.current_action.value,
            "auto_suspend": self.auto_suspend,
            "is_exceeded": self.is_exceeded,
            "thresholds": [
                {"percentage": t.percentage, "action": t.action.value} for t in self.thresholds
            ],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class BudgetAlert:
    """Budget alert event."""

    alert_id: str
    budget_id: str
    org_id: str
    threshold_percentage: float
    action: BudgetAction
    spent_usd: float
    amount_usd: float
    message: str
    created_at: float = field(default_factory=lambda: time.time())
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "budget_id": self.budget_id,
            "org_id": self.org_id,
            "threshold_percentage": self.threshold_percentage,
            "action": self.action.value,
            "spent_usd": self.spent_usd,
            "amount_usd": self.amount_usd,
            "usage_percentage": self.spent_usd / self.amount_usd if self.amount_usd > 0 else 0,
            "message": self.message,
            "created_at": self.created_at,
            "created_at_iso": datetime.fromtimestamp(self.created_at, tz=timezone.utc).isoformat(),
            "acknowledged": self.acknowledged,
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at,
        }


class BudgetManager:
    """
    Manages budget lifecycle and enforcement.

    Thread-safe with SQLite persistence.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS budgets (
        budget_id TEXT PRIMARY KEY,
        org_id TEXT NOT NULL,
        name TEXT NOT NULL,
        description TEXT DEFAULT '',
        amount_usd REAL NOT NULL,
        period TEXT NOT NULL,
        spent_usd REAL DEFAULT 0,
        period_start REAL,
        period_end REAL,
        status TEXT DEFAULT 'active',
        auto_suspend INTEGER DEFAULT 1,
        thresholds_json TEXT,
        override_user_ids_json TEXT,
        override_until REAL,
        created_at REAL NOT NULL,
        updated_at REAL NOT NULL,
        created_by TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_budgets_org ON budgets(org_id);
    CREATE INDEX IF NOT EXISTS idx_budgets_status ON budgets(status);

    CREATE TABLE IF NOT EXISTS budget_alerts (
        alert_id TEXT PRIMARY KEY,
        budget_id TEXT NOT NULL,
        org_id TEXT NOT NULL,
        threshold_percentage REAL NOT NULL,
        action TEXT NOT NULL,
        spent_usd REAL NOT NULL,
        amount_usd REAL NOT NULL,
        message TEXT,
        created_at REAL NOT NULL,
        acknowledged INTEGER DEFAULT 0,
        acknowledged_by TEXT,
        acknowledged_at REAL,
        FOREIGN KEY (budget_id) REFERENCES budgets(budget_id)
    );

    CREATE INDEX IF NOT EXISTS idx_alerts_budget ON budget_alerts(budget_id);
    CREATE INDEX IF NOT EXISTS idx_alerts_org ON budget_alerts(org_id);

    CREATE TABLE IF NOT EXISTS budget_transactions (
        transaction_id TEXT PRIMARY KEY,
        budget_id TEXT NOT NULL,
        amount_usd REAL NOT NULL,
        description TEXT,
        debate_id TEXT,
        user_id TEXT,
        created_at REAL NOT NULL,
        FOREIGN KEY (budget_id) REFERENCES budgets(budget_id)
    );

    CREATE INDEX IF NOT EXISTS idx_transactions_budget ON budget_transactions(budget_id);
    """

    def __init__(self, db_path: str = "data/budgets.db"):
        """Initialize budget manager.

        Args:
            db_path: Path to SQLite database
        """
        self._db_path = db_path
        self._local = threading.local()
        self._init_lock = threading.Lock()
        self._initialized = False
        self._alert_callbacks: List[Callable[[BudgetAlert], None]] = []
        self._alert_cooldowns: Dict[str, float] = {}  # budget_id -> last alert time

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "connection") or self._local.connection is None:
            import os

            db_dir = os.path.dirname(self._db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)

            self._local.connection = sqlite3.connect(self._db_path, check_same_thread=False)
            self._local.connection.row_factory = sqlite3.Row
            self._ensure_schema(self._local.connection)

        return self._local.connection

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        """Ensure database schema exists."""
        with self._init_lock:
            if not self._initialized:
                conn.executescript(self.SCHEMA)
                conn.commit()
                self._initialized = True

    def register_alert_callback(self, callback: Callable[[BudgetAlert], None]) -> None:
        """Register a callback for budget alerts."""
        self._alert_callbacks.append(callback)

    # =========================================================================
    # Budget CRUD Operations
    # =========================================================================

    def create_budget(
        self,
        org_id: str,
        name: str,
        amount_usd: float,
        period: BudgetPeriod = BudgetPeriod.MONTHLY,
        description: str = "",
        auto_suspend: bool = True,
        thresholds: Optional[List[BudgetThreshold]] = None,
        created_by: Optional[str] = None,
    ) -> Budget:
        """Create a new budget.

        Args:
            org_id: Organization ID
            name: Budget name
            amount_usd: Budget limit in USD
            period: Budget period type
            description: Optional description
            auto_suspend: Auto-suspend on exceed
            thresholds: Custom alert thresholds
            created_by: User who created the budget

        Returns:
            Created Budget object
        """
        import json

        budget_id = f"budget-{uuid.uuid4().hex[:12]}"
        now = time.time()

        # Calculate period bounds
        period_start, period_end = self._calculate_period_bounds(period)

        budget = Budget(
            budget_id=budget_id,
            org_id=org_id,
            name=name,
            description=description,
            amount_usd=amount_usd,
            period=period,
            period_start=period_start,
            period_end=period_end,
            auto_suspend=auto_suspend,
            thresholds=thresholds or DEFAULT_THRESHOLDS,
            created_at=now,
            updated_at=now,
            created_by=created_by,
        )

        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO budgets
            (budget_id, org_id, name, description, amount_usd, period,
             spent_usd, period_start, period_end, status, auto_suspend,
             thresholds_json, override_user_ids_json, created_at, updated_at, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                budget.budget_id,
                budget.org_id,
                budget.name,
                budget.description,
                budget.amount_usd,
                budget.period.value,
                budget.spent_usd,
                budget.period_start,
                budget.period_end,
                budget.status.value,
                1 if budget.auto_suspend else 0,
                json.dumps(
                    [
                        {"percentage": t.percentage, "action": t.action.value}
                        for t in budget.thresholds
                    ]
                ),
                json.dumps(budget.override_user_ids),
                budget.created_at,
                budget.updated_at,
                budget.created_by,
            ),
        )
        conn.commit()

        logger.info(
            f"Created budget {budget_id} for org {org_id}: ${amount_usd:.2f}/{period.value}"
        )
        return budget

    def get_budget(self, budget_id: str) -> Optional[Budget]:
        """Get a budget by ID."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM budgets WHERE budget_id = ?", (budget_id,))
        row = cursor.fetchone()

        if row:
            return self._budget_from_row(row)
        return None

    def get_budgets_for_org(self, org_id: str, active_only: bool = True) -> List[Budget]:
        """Get all budgets for an organization."""
        conn = self._get_connection()

        if active_only:
            cursor = conn.execute(
                "SELECT * FROM budgets WHERE org_id = ? AND status NOT IN ('closed', 'suspended') ORDER BY created_at DESC",
                (org_id,),
            )
        else:
            cursor = conn.execute(
                "SELECT * FROM budgets WHERE org_id = ? ORDER BY created_at DESC",
                (org_id,),
            )

        return [self._budget_from_row(row) for row in cursor.fetchall()]

    def update_budget(
        self,
        budget_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        amount_usd: Optional[float] = None,
        auto_suspend: Optional[bool] = None,
        status: Optional[BudgetStatus] = None,
    ) -> Optional[Budget]:
        """Update a budget."""
        budget = self.get_budget(budget_id)
        if not budget:
            return None

        if name is not None:
            budget.name = name
        if description is not None:
            budget.description = description
        if amount_usd is not None:
            budget.amount_usd = amount_usd
        if auto_suspend is not None:
            budget.auto_suspend = auto_suspend
        if status is not None:
            budget.status = status

        budget.updated_at = time.time()

        conn = self._get_connection()
        conn.execute(
            """
            UPDATE budgets SET
                name = ?, description = ?, amount_usd = ?,
                auto_suspend = ?, status = ?, updated_at = ?
            WHERE budget_id = ?
            """,
            (
                budget.name,
                budget.description,
                budget.amount_usd,
                1 if budget.auto_suspend else 0,
                budget.status.value,
                budget.updated_at,
                budget_id,
            ),
        )
        conn.commit()

        logger.info(f"Updated budget {budget_id}")
        return budget

    def delete_budget(self, budget_id: str) -> bool:
        """Delete a budget (soft delete by setting status to closed)."""
        conn = self._get_connection()
        conn.execute(
            "UPDATE budgets SET status = 'closed', updated_at = ? WHERE budget_id = ?",
            (time.time(), budget_id),
        )
        conn.commit()
        return True

    # =========================================================================
    # Budget Enforcement
    # =========================================================================

    def check_budget(
        self,
        org_id: str,
        estimated_cost_usd: float,
        user_id: Optional[str] = None,
    ) -> tuple[bool, str, Optional[BudgetAction]]:
        """Check if an operation is allowed within budget.

        Args:
            org_id: Organization ID
            estimated_cost_usd: Estimated cost of operation
            user_id: User requesting the operation (for override check)

        Returns:
            Tuple of (allowed, reason, action_required)
        """
        budgets = self.get_budgets_for_org(org_id, active_only=True)

        if not budgets:
            # No budget configured - allow
            return True, "No budget configured", None

        for budget in budgets:
            allowed, reason = budget.can_spend(estimated_cost_usd, user_id)
            if not allowed:
                return False, reason, budget.current_action

            # Check if this would trigger a warning
            new_total = budget.spent_usd + estimated_cost_usd
            new_pct = new_total / budget.amount_usd if budget.amount_usd > 0 else 0

            for threshold in budget.thresholds:
                if (
                    new_pct >= threshold.percentage
                    and budget.usage_percentage < threshold.percentage
                ):
                    # This operation would cross a threshold
                    if threshold.action == BudgetAction.SOFT_LIMIT:
                        return (
                            True,
                            f"Warning: This will use {new_pct:.0%} of budget",
                            threshold.action,
                        )

        return True, "OK", None

    def record_spend(
        self,
        org_id: str,
        amount_usd: float,
        description: str = "",
        debate_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        """Record spending against the organization's budget.

        Args:
            org_id: Organization ID
            amount_usd: Amount spent in USD
            description: Description of the spend
            debate_id: Associated debate ID
            user_id: User who incurred the spend

        Returns:
            True if recorded successfully
        """
        budgets = self.get_budgets_for_org(org_id, active_only=True)

        if not budgets:
            return True  # No budget to track against

        conn = self._get_connection()

        for budget in budgets:
            old_pct = budget.usage_percentage
            budget.spent_usd += amount_usd
            new_pct = budget.usage_percentage

            # Update database
            conn.execute(
                "UPDATE budgets SET spent_usd = ?, updated_at = ? WHERE budget_id = ?",
                (budget.spent_usd, time.time(), budget.budget_id),
            )

            # Record transaction
            conn.execute(
                """
                INSERT INTO budget_transactions
                (transaction_id, budget_id, amount_usd, description, debate_id, user_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"txn-{uuid.uuid4().hex[:12]}",
                    budget.budget_id,
                    amount_usd,
                    description,
                    debate_id,
                    user_id,
                    time.time(),
                ),
            )

            # Check thresholds and trigger alerts
            self._check_thresholds(budget, old_pct, new_pct)

            # Auto-suspend if exceeded
            if budget.is_exceeded and budget.auto_suspend:
                budget.status = BudgetStatus.SUSPENDED
                conn.execute(
                    "UPDATE budgets SET status = 'suspended', updated_at = ? WHERE budget_id = ?",
                    (time.time(), budget.budget_id),
                )
                logger.warning(f"Budget {budget.budget_id} auto-suspended (exceeded)")

        conn.commit()
        return True

    def _check_thresholds(self, budget: Budget, old_pct: float, new_pct: float) -> None:
        """Check if any thresholds were crossed and trigger alerts."""
        for threshold in budget.thresholds:
            if old_pct < threshold.percentage <= new_pct:
                # Threshold crossed - check cooldown
                cooldown_key = f"{budget.budget_id}:{threshold.percentage}"
                last_alert = self._alert_cooldowns.get(cooldown_key, 0)

                if time.time() - last_alert > threshold.cooldown_minutes * 60:
                    alert = self._create_alert(budget, threshold)
                    self._alert_cooldowns[cooldown_key] = time.time()

                    # Trigger callbacks
                    for callback in self._alert_callbacks:
                        try:
                            callback(alert)
                        except Exception as e:
                            logger.error(f"Alert callback failed: {e}")

    def _create_alert(self, budget: Budget, threshold: BudgetThreshold) -> BudgetAlert:
        """Create and persist a budget alert."""
        alert = BudgetAlert(
            alert_id=f"alert-{uuid.uuid4().hex[:12]}",
            budget_id=budget.budget_id,
            org_id=budget.org_id,
            threshold_percentage=threshold.percentage,
            action=threshold.action,
            spent_usd=budget.spent_usd,
            amount_usd=budget.amount_usd,
            message=f"Budget '{budget.name}' reached {threshold.percentage:.0%} "
            f"(${budget.spent_usd:.2f}/${budget.amount_usd:.2f})",
        )

        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO budget_alerts
            (alert_id, budget_id, org_id, threshold_percentage, action,
             spent_usd, amount_usd, message, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                alert.alert_id,
                alert.budget_id,
                alert.org_id,
                alert.threshold_percentage,
                alert.action.value,
                alert.spent_usd,
                alert.amount_usd,
                alert.message,
                alert.created_at,
            ),
        )
        conn.commit()

        logger.info(f"Budget alert: {alert.message}")
        return alert

    # =========================================================================
    # Alerts Management
    # =========================================================================

    def get_alerts(
        self,
        org_id: Optional[str] = None,
        budget_id: Optional[str] = None,
        unacknowledged_only: bool = False,
        limit: int = 50,
    ) -> List[BudgetAlert]:
        """Get budget alerts."""
        conn = self._get_connection()

        query = "SELECT * FROM budget_alerts WHERE 1=1"
        params: List[Any] = []

        if org_id:
            query += " AND org_id = ?"
            params.append(org_id)
        if budget_id:
            query += " AND budget_id = ?"
            params.append(budget_id)
        if unacknowledged_only:
            query += " AND acknowledged = 0"

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor = conn.execute(query, params)

        alerts = []
        for row in cursor.fetchall():
            alerts.append(
                BudgetAlert(
                    alert_id=row["alert_id"],
                    budget_id=row["budget_id"],
                    org_id=row["org_id"],
                    threshold_percentage=row["threshold_percentage"],
                    action=BudgetAction(row["action"]),
                    spent_usd=row["spent_usd"],
                    amount_usd=row["amount_usd"],
                    message=row["message"],
                    created_at=row["created_at"],
                    acknowledged=bool(row["acknowledged"]),
                    acknowledged_by=row["acknowledged_by"],
                    acknowledged_at=row["acknowledged_at"],
                )
            )

        return alerts

    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge a budget alert."""
        conn = self._get_connection()
        conn.execute(
            "UPDATE budget_alerts SET acknowledged = 1, acknowledged_by = ?, acknowledged_at = ? WHERE alert_id = ?",
            (user_id, time.time(), alert_id),
        )
        conn.commit()
        return True

    # =========================================================================
    # Overrides
    # =========================================================================

    def add_override(
        self,
        budget_id: str,
        user_id: str,
        duration_hours: Optional[float] = None,
    ) -> bool:
        """Add a budget override for a user.

        Args:
            budget_id: Budget ID
            user_id: User ID to grant override
            duration_hours: Override duration (None = permanent)
        """
        import json

        budget = self.get_budget(budget_id)
        if not budget:
            return False

        if user_id not in budget.override_user_ids:
            budget.override_user_ids.append(user_id)

        if duration_hours:
            budget.override_until = time.time() + (duration_hours * 3600)

        conn = self._get_connection()
        conn.execute(
            "UPDATE budgets SET override_user_ids_json = ?, override_until = ?, updated_at = ? WHERE budget_id = ?",
            (json.dumps(budget.override_user_ids), budget.override_until, time.time(), budget_id),
        )
        conn.commit()

        logger.info(f"Added override for user {user_id} on budget {budget_id}")
        return True

    def remove_override(self, budget_id: str, user_id: str) -> bool:
        """Remove a budget override for a user."""
        import json

        budget = self.get_budget(budget_id)
        if not budget:
            return False

        if user_id in budget.override_user_ids:
            budget.override_user_ids.remove(user_id)

        conn = self._get_connection()
        conn.execute(
            "UPDATE budgets SET override_user_ids_json = ?, updated_at = ? WHERE budget_id = ?",
            (json.dumps(budget.override_user_ids), time.time(), budget_id),
        )
        conn.commit()

        return True

    # =========================================================================
    # Utilities
    # =========================================================================

    def _budget_from_row(self, row: sqlite3.Row) -> Budget:
        """Create Budget from database row."""
        import json

        thresholds_data = json.loads(row["thresholds_json"] or "[]")
        thresholds = [
            BudgetThreshold(t["percentage"], BudgetAction(t["action"])) for t in thresholds_data
        ]

        override_ids = json.loads(row["override_user_ids_json"] or "[]")

        return Budget(
            budget_id=row["budget_id"],
            org_id=row["org_id"],
            name=row["name"],
            description=row["description"] or "",
            amount_usd=row["amount_usd"],
            period=BudgetPeriod(row["period"]),
            spent_usd=row["spent_usd"],
            period_start=row["period_start"] or 0,
            period_end=row["period_end"] or 0,
            status=BudgetStatus(row["status"]),
            auto_suspend=bool(row["auto_suspend"]),
            thresholds=thresholds or DEFAULT_THRESHOLDS,
            override_user_ids=override_ids,
            override_until=row["override_until"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            created_by=row["created_by"],
        )

    def _calculate_period_bounds(self, period: BudgetPeriod) -> tuple[float, float]:
        """Calculate period start and end timestamps."""
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if period == BudgetPeriod.DAILY:
            start = start_of_day
            end = start + timedelta(days=1)
        elif period == BudgetPeriod.WEEKLY:
            start = start_of_day - timedelta(days=now.weekday())
            end = start + timedelta(weeks=1)
        elif period == BudgetPeriod.MONTHLY:
            start = start_of_day.replace(day=1)
            if start.month == 12:
                end = start.replace(year=start.year + 1, month=1)
            else:
                end = start.replace(month=start.month + 1)
        elif period == BudgetPeriod.QUARTERLY:
            quarter_start_month = ((now.month - 1) // 3) * 3 + 1
            start = start_of_day.replace(month=quarter_start_month, day=1)
            end_month = quarter_start_month + 3
            if end_month > 12:
                end = start.replace(year=start.year + 1, month=end_month - 12)
            else:
                end = start.replace(month=end_month)
        elif period == BudgetPeriod.ANNUAL:
            start = start_of_day.replace(month=1, day=1)
            end = start.replace(year=start.year + 1)
        else:  # UNLIMITED
            start = start_of_day
            end = start_of_day.replace(year=start.year + 100)

        return start.timestamp(), end.timestamp()

    def reset_period(self, budget_id: str) -> Optional[Budget]:
        """Reset a budget's spending for a new period."""
        budget = self.get_budget(budget_id)
        if not budget:
            return None

        budget.spent_usd = 0.0
        budget.status = BudgetStatus.ACTIVE
        period_start, period_end = self._calculate_period_bounds(budget.period)
        budget.period_start = period_start
        budget.period_end = period_end
        budget.updated_at = time.time()

        conn = self._get_connection()
        conn.execute(
            """
            UPDATE budgets SET
                spent_usd = 0, status = 'active',
                period_start = ?, period_end = ?, updated_at = ?
            WHERE budget_id = ?
            """,
            (period_start, period_end, budget.updated_at, budget_id),
        )
        conn.commit()

        logger.info(f"Reset budget {budget_id} for new period")
        return budget

    def get_summary(self, org_id: str) -> Dict[str, Any]:
        """Get budget summary for an organization."""
        budgets = self.get_budgets_for_org(org_id, active_only=False)

        total_budget = sum(b.amount_usd for b in budgets if b.status == BudgetStatus.ACTIVE)
        total_spent = sum(b.spent_usd for b in budgets if b.status == BudgetStatus.ACTIVE)
        active_count = len([b for b in budgets if b.status == BudgetStatus.ACTIVE])
        exceeded_count = len([b for b in budgets if b.is_exceeded])

        return {
            "org_id": org_id,
            "total_budget_usd": total_budget,
            "total_spent_usd": total_spent,
            "total_remaining_usd": total_budget - total_spent,
            "overall_usage_percentage": total_spent / total_budget if total_budget > 0 else 0,
            "active_budgets": active_count,
            "exceeded_budgets": exceeded_count,
            "budgets": [b.to_dict() for b in budgets],
        }


# Module-level singleton
_budget_manager: Optional[BudgetManager] = None


def get_budget_manager(db_path: str = "data/budgets.db") -> BudgetManager:
    """Get or create the budget manager singleton."""
    global _budget_manager
    if _budget_manager is None:
        _budget_manager = BudgetManager(db_path)
    return _budget_manager
