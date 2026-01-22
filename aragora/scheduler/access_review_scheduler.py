"""
Automated Access Review Scheduler.

Provides periodic access review automation for SOC 2 CC6.1 compliance:
- Monthly user access reviews
- Role certification workflows
- Stale credential detection (90+ days unused)
- Orphaned account cleanup
- Manager sign-off requirements
- Non-compliance alerting

SOC 2 Compliance: CC6.1, CC6.2 (Access Control)

Usage:
    from aragora.scheduler.access_review_scheduler import (
        AccessReviewScheduler,
        get_access_review_scheduler,
        schedule_access_review,
    )

    # Initialize scheduler
    scheduler = AccessReviewScheduler(storage_path="access_reviews.db")

    # Start automated reviews
    await scheduler.start()

    # Manually trigger a review
    review = await scheduler.create_review(
        review_type="monthly",
        scope=["workspace_123"],
    )
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Types and Enums
# =============================================================================


class ReviewType(Enum):
    """Types of access reviews."""

    MONTHLY = "monthly"  # Regular monthly review
    QUARTERLY = "quarterly"  # Quarterly certification
    AD_HOC = "ad_hoc"  # Manual trigger
    STALE_CREDENTIALS = "stale_credentials"  # 90+ day unused
    ORPHANED_ACCOUNTS = "orphaned_accounts"  # No manager/inactive
    PRIVILEGE_ESCALATION = "privilege_escalation"  # High privilege review


class ReviewStatus(Enum):
    """Status of an access review."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class ReviewItemStatus(Enum):
    """Status of a review item."""

    PENDING = "pending"
    APPROVED = "approved"
    REVOKED = "revoked"
    MODIFIED = "modified"
    SKIPPED = "skipped"


@dataclass
class AccessReviewItem:
    """An item in an access review."""

    item_id: str
    user_id: str
    user_email: str
    resource_type: str  # role, permission, api_key, workspace_access
    resource_id: str
    resource_name: str
    granted_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    granted_by: Optional[str] = None
    risk_level: str = "low"  # low, medium, high, critical
    status: ReviewItemStatus = ReviewItemStatus.PENDING
    decision_by: Optional[str] = None
    decision_at: Optional[datetime] = None
    decision_notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessReview:
    """An access review instance."""

    review_id: str
    review_type: ReviewType
    status: ReviewStatus = ReviewStatus.PENDING
    scope_workspaces: List[str] = field(default_factory=list)
    scope_users: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    due_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_by: str = "system"
    assigned_reviewer: Optional[str] = None
    items: List[AccessReviewItem] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "review_id": self.review_id,
            "review_type": self.review_type.value,
            "status": self.status.value,
            "scope_workspaces": self.scope_workspaces,
            "scope_users": self.scope_users,
            "created_at": self.created_at.isoformat(),
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_by": self.created_by,
            "assigned_reviewer": self.assigned_reviewer,
            "items_count": len(self.items),
            "summary": self.summary,
        }


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class AccessReviewConfig:
    """Configuration for access review scheduler."""

    # Review frequency
    monthly_review_day: int = 1  # Day of month for monthly reviews
    quarterly_review_months: List[int] = field(default_factory=lambda: [1, 4, 7, 10])

    # Thresholds
    stale_credential_days: int = 90
    orphan_detection_days: int = 30
    review_due_days: int = 14  # Days to complete a review

    # Notifications
    notification_email: Optional[str] = None
    slack_webhook: Optional[str] = None

    # Storage
    storage_path: Optional[str] = None


# =============================================================================
# Storage Layer
# =============================================================================


class AccessReviewStorage:
    """SQLite-backed storage for access reviews."""

    def __init__(self, db_path: Optional[str] = None):
        """Initialize storage."""
        self._db_path = db_path or ":memory:"
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_schema(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS access_reviews (
                review_id TEXT PRIMARY KEY,
                review_type TEXT NOT NULL,
                status TEXT NOT NULL,
                scope_workspaces_json TEXT,
                scope_users_json TEXT,
                created_at TEXT NOT NULL,
                due_date TEXT,
                completed_at TEXT,
                created_by TEXT,
                assigned_reviewer TEXT,
                summary_json TEXT
            );

            CREATE TABLE IF NOT EXISTS review_items (
                item_id TEXT PRIMARY KEY,
                review_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_email TEXT,
                resource_type TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                resource_name TEXT,
                granted_at TEXT,
                last_used TEXT,
                granted_by TEXT,
                risk_level TEXT DEFAULT 'low',
                status TEXT NOT NULL,
                decision_by TEXT,
                decision_at TEXT,
                decision_notes TEXT,
                metadata_json TEXT,
                FOREIGN KEY (review_id) REFERENCES access_reviews(review_id)
            );

            CREATE TABLE IF NOT EXISTS user_access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_id TEXT NOT NULL,
                last_accessed TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS review_schedule (
                schedule_id TEXT PRIMARY KEY,
                review_type TEXT NOT NULL,
                cron_expression TEXT,
                next_run TEXT,
                last_run TEXT,
                enabled INTEGER DEFAULT 1
            );

            CREATE INDEX IF NOT EXISTS idx_reviews_status ON access_reviews(status);
            CREATE INDEX IF NOT EXISTS idx_reviews_due ON access_reviews(due_date);
            CREATE INDEX IF NOT EXISTS idx_items_review ON review_items(review_id);
            CREATE INDEX IF NOT EXISTS idx_items_user ON review_items(user_id);
            CREATE INDEX IF NOT EXISTS idx_access_log_user ON user_access_log(user_id, resource_type);
            """
        )
        conn.commit()

    def save_review(self, review: AccessReview) -> None:
        """Save or update a review."""
        import json

        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO access_reviews (
                review_id, review_type, status, scope_workspaces_json,
                scope_users_json, created_at, due_date, completed_at,
                created_by, assigned_reviewer, summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                review.review_id,
                review.review_type.value,
                review.status.value,
                json.dumps(review.scope_workspaces),
                json.dumps(review.scope_users),
                review.created_at.isoformat(),
                review.due_date.isoformat() if review.due_date else None,
                review.completed_at.isoformat() if review.completed_at else None,
                review.created_by,
                review.assigned_reviewer,
                json.dumps(review.summary),
            ),
        )

        # Save items
        for item in review.items:
            conn.execute(
                """
                INSERT OR REPLACE INTO review_items (
                    item_id, review_id, user_id, user_email, resource_type,
                    resource_id, resource_name, granted_at, last_used,
                    granted_by, risk_level, status, decision_by,
                    decision_at, decision_notes, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.item_id,
                    review.review_id,
                    item.user_id,
                    item.user_email,
                    item.resource_type,
                    item.resource_id,
                    item.resource_name,
                    item.granted_at.isoformat() if item.granted_at else None,
                    item.last_used.isoformat() if item.last_used else None,
                    item.granted_by,
                    item.risk_level,
                    item.status.value,
                    item.decision_by,
                    item.decision_at.isoformat() if item.decision_at else None,
                    item.decision_notes,
                    json.dumps(item.metadata),
                ),
            )

        conn.commit()

    def get_review(self, review_id: str) -> Optional[AccessReview]:
        """Get a review by ID."""
        import json

        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM access_reviews WHERE review_id = ?",
            (review_id,),
        ).fetchone()

        if not row:
            return None

        # Load items
        items = []
        item_rows = conn.execute(
            "SELECT * FROM review_items WHERE review_id = ?",
            (review_id,),
        ).fetchall()

        for ir in item_rows:
            items.append(
                AccessReviewItem(
                    item_id=ir["item_id"],
                    user_id=ir["user_id"],
                    user_email=ir["user_email"] or "",
                    resource_type=ir["resource_type"],
                    resource_id=ir["resource_id"],
                    resource_name=ir["resource_name"] or "",
                    granted_at=datetime.fromisoformat(ir["granted_at"])
                    if ir["granted_at"]
                    else None,
                    last_used=datetime.fromisoformat(ir["last_used"]) if ir["last_used"] else None,
                    granted_by=ir["granted_by"],
                    risk_level=ir["risk_level"],
                    status=ReviewItemStatus(ir["status"]),
                    decision_by=ir["decision_by"],
                    decision_at=datetime.fromisoformat(ir["decision_at"])
                    if ir["decision_at"]
                    else None,
                    decision_notes=ir["decision_notes"],
                    metadata=json.loads(ir["metadata_json"] or "{}"),
                )
            )

        return AccessReview(
            review_id=row["review_id"],
            review_type=ReviewType(row["review_type"]),
            status=ReviewStatus(row["status"]),
            scope_workspaces=json.loads(row["scope_workspaces_json"] or "[]"),
            scope_users=json.loads(row["scope_users_json"] or "[]"),
            created_at=datetime.fromisoformat(row["created_at"]),
            due_date=datetime.fromisoformat(row["due_date"]) if row["due_date"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"])
            if row["completed_at"]
            else None,
            created_by=row["created_by"],
            assigned_reviewer=row["assigned_reviewer"],
            items=items,
            summary=json.loads(row["summary_json"] or "{}"),
        )

    def get_pending_reviews(self) -> List[AccessReview]:
        """Get all pending reviews."""
        conn = self._get_conn()
        rows = conn.execute(
            """
            SELECT review_id FROM access_reviews
            WHERE status IN ('pending', 'in_progress', 'awaiting_approval')
            ORDER BY due_date ASC
            """,
        ).fetchall()

        reviews = []
        for row in rows:
            review = self.get_review(row["review_id"])
            if review:
                reviews.append(review)

        return reviews

    def get_overdue_reviews(self) -> List[AccessReview]:
        """Get overdue reviews."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()

        rows = conn.execute(
            """
            SELECT review_id FROM access_reviews
            WHERE status IN ('pending', 'in_progress', 'awaiting_approval')
            AND due_date < ?
            """,
            (now,),
        ).fetchall()

        reviews = []
        for row in rows:
            review = self.get_review(row["review_id"])
            if review:
                reviews.append(review)

        return reviews

    def record_user_access(self, user_id: str, resource_type: str, resource_id: str) -> None:
        """Record user access to a resource."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()

        # Update or insert
        conn.execute(
            """
            INSERT INTO user_access_log (user_id, resource_type, resource_id, last_accessed)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, resource_type, resource_id) DO UPDATE SET last_accessed = ?
            """,
            (user_id, resource_type, resource_id, now, now),
        )
        conn.commit()

    def get_last_access(
        self, user_id: str, resource_type: str, resource_id: str
    ) -> Optional[datetime]:
        """Get last access time for a user/resource."""
        conn = self._get_conn()
        row = conn.execute(
            """
            SELECT last_accessed FROM user_access_log
            WHERE user_id = ? AND resource_type = ? AND resource_id = ?
            """,
            (user_id, resource_type, resource_id),
        ).fetchone()

        if row and row["last_accessed"]:
            return datetime.fromisoformat(row["last_accessed"])
        return None

    def get_stale_credentials(self, days: int = 90) -> List[Dict[str, Any]]:
        """Get credentials not used in N days."""
        conn = self._get_conn()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        rows = conn.execute(
            """
            SELECT user_id, resource_type, resource_id, last_accessed
            FROM user_access_log
            WHERE last_accessed < ?
            ORDER BY last_accessed ASC
            """,
            (cutoff,),
        ).fetchall()

        return [
            {
                "user_id": row["user_id"],
                "resource_type": row["resource_type"],
                "resource_id": row["resource_id"],
                "last_accessed": row["last_accessed"],
                "days_stale": (
                    datetime.now(timezone.utc) - datetime.fromisoformat(row["last_accessed"])
                ).days,
            }
            for row in rows
        ]

    def save_schedule(
        self,
        schedule_id: str,
        review_type: str,
        next_run: datetime,
        enabled: bool = True,
    ) -> None:
        """Save a review schedule."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO review_schedule (
                schedule_id, review_type, next_run, enabled
            ) VALUES (?, ?, ?, ?)
            """,
            (schedule_id, review_type, next_run.isoformat(), 1 if enabled else 0),
        )
        conn.commit()

    def get_due_schedules(self) -> List[Dict[str, Any]]:
        """Get schedules that are due to run."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()

        rows = conn.execute(
            """
            SELECT * FROM review_schedule
            WHERE enabled = 1 AND next_run <= ?
            """,
            (now,),
        ).fetchall()

        return [
            {
                "schedule_id": row["schedule_id"],
                "review_type": row["review_type"],
                "next_run": row["next_run"],
                "last_run": row["last_run"],
            }
            for row in rows
        ]

    def update_schedule_run(self, schedule_id: str, next_run: datetime) -> None:
        """Update schedule after a run."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()

        conn.execute(
            """
            UPDATE review_schedule
            SET last_run = ?, next_run = ?
            WHERE schedule_id = ?
            """,
            (now, next_run.isoformat(), schedule_id),
        )
        conn.commit()


# =============================================================================
# Access Review Scheduler
# =============================================================================


class AccessReviewScheduler:
    """Main access review scheduler."""

    def __init__(self, config: Optional[AccessReviewConfig] = None):
        """Initialize scheduler.

        Args:
            config: Scheduler configuration
        """
        self.config = config or AccessReviewConfig()
        self._storage = AccessReviewStorage(self.config.storage_path)
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Callbacks for integrating with access control systems
        self._access_providers: List[Callable[[], List[Dict[str, Any]]]] = []
        self._revocation_handlers: List[Callable[[str, str, str], None]] = []
        self._notification_handlers: List[Callable[[Dict[str, Any]], None]] = []

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return

        self._running = True
        self._init_schedules()
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Access review scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Access review scheduler stopped")

    def _init_schedules(self) -> None:
        """Initialize default schedules."""
        now = datetime.now(timezone.utc)

        # Monthly review schedule
        next_monthly = now.replace(day=self.config.monthly_review_day, hour=9, minute=0, second=0)
        if next_monthly <= now:
            if now.month == 12:
                next_monthly = next_monthly.replace(year=now.year + 1, month=1)
            else:
                next_monthly = next_monthly.replace(month=now.month + 1)

        self._storage.save_schedule("monthly_review", "monthly", next_monthly)

        # Stale credential check (weekly)
        next_stale = now + timedelta(days=(7 - now.weekday()) % 7 or 7)
        next_stale = next_stale.replace(hour=8, minute=0, second=0)
        self._storage.save_schedule("stale_check", "stale_credentials", next_stale)

        logger.info(f"Initialized schedules: monthly={next_monthly}, stale_check={next_stale}")

    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                # Check for due schedules
                due_schedules = self._storage.get_due_schedules()
                for schedule in due_schedules:
                    await self._execute_scheduled_review(schedule)

                # Check for overdue reviews
                overdue = self._storage.get_overdue_reviews()
                for review in overdue:
                    await self._handle_overdue_review(review)

                # Sleep before next check (every 5 minutes)
                await asyncio.sleep(300)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in access review scheduler: {e}")
                await asyncio.sleep(60)

    async def _execute_scheduled_review(self, schedule: Dict[str, Any]) -> None:
        """Execute a scheduled review."""
        review_type = ReviewType(schedule["review_type"])
        logger.info(f"Executing scheduled review: {review_type.value}")

        # Create the review
        review = await self.create_review(review_type=review_type)

        # Calculate next run
        if review_type == ReviewType.MONTHLY:
            next_run = datetime.fromisoformat(schedule["next_run"])
            if next_run.month == 12:
                next_run = next_run.replace(year=next_run.year + 1, month=1)
            else:
                next_run = next_run.replace(month=next_run.month + 1)
        elif review_type == ReviewType.STALE_CREDENTIALS:
            next_run = datetime.now(timezone.utc) + timedelta(days=7)
        else:
            next_run = datetime.now(timezone.utc) + timedelta(days=30)

        self._storage.update_schedule_run(schedule["schedule_id"], next_run)

        # Send notifications
        await self._notify_review_created(review)

    async def _handle_overdue_review(self, review: AccessReview) -> None:
        """Handle an overdue review."""
        logger.warning(f"Review {review.review_id} is overdue")

        # Mark as expired if too old
        if review.due_date:
            days_overdue = (datetime.now(timezone.utc) - review.due_date).days
            if days_overdue > 7:
                review.status = ReviewStatus.EXPIRED
                self._storage.save_review(review)

                # Notify about expiration
                await self._notify_review_expired(review)

    # =========================================================================
    # Review Creation
    # =========================================================================

    async def create_review(
        self,
        review_type: ReviewType,
        scope_workspaces: Optional[List[str]] = None,
        scope_users: Optional[List[str]] = None,
        assigned_reviewer: Optional[str] = None,
        created_by: str = "system",
    ) -> AccessReview:
        """Create a new access review.

        Args:
            review_type: Type of review
            scope_workspaces: Limit to specific workspaces
            scope_users: Limit to specific users
            assigned_reviewer: User ID of reviewer
            created_by: User ID who created the review

        Returns:
            Created review
        """
        review = AccessReview(
            review_id=str(uuid.uuid4()),
            review_type=review_type,
            status=ReviewStatus.PENDING,
            scope_workspaces=scope_workspaces or [],
            scope_users=scope_users or [],
            due_date=datetime.now(timezone.utc) + timedelta(days=self.config.review_due_days),
            created_by=created_by,
            assigned_reviewer=assigned_reviewer,
        )

        # Gather access items to review
        items = await self._gather_review_items(review)
        review.items = items

        # Calculate risk summary
        review.summary = self._calculate_summary(items)

        # Persist
        self._storage.save_review(review)

        logger.info(
            f"Created access review {review.review_id}: "
            f"type={review_type.value}, items={len(items)}"
        )

        return review

    async def _gather_review_items(self, review: AccessReview) -> List[AccessReviewItem]:
        """Gather items to review based on type and scope."""
        items: List[AccessReviewItem] = []

        # Get access data from providers
        all_access: List[Dict[str, Any]] = []
        for provider in self._access_providers:
            try:
                access_data = provider()
                all_access.extend(access_data)
            except Exception as e:
                logger.error(f"Error getting access data from provider: {e}")

        # If no providers, use mock data for demo
        if not all_access:
            all_access = self._get_mock_access_data()

        # Filter by scope
        for access in all_access:
            # Filter by workspace scope
            if review.scope_workspaces:
                if access.get("workspace_id") not in review.scope_workspaces:
                    continue

            # Filter by user scope
            if review.scope_users:
                if access.get("user_id") not in review.scope_users:
                    continue

            # Filter by review type
            if review.review_type == ReviewType.STALE_CREDENTIALS:
                last_used = self._storage.get_last_access(
                    access["user_id"],
                    access["resource_type"],
                    access["resource_id"],
                )
                if last_used:
                    days_since = (datetime.now(timezone.utc) - last_used).days
                    if days_since < self.config.stale_credential_days:
                        continue

            # Determine risk level
            risk_level = self._assess_risk_level(access)

            # Create review item
            item = AccessReviewItem(
                item_id=str(uuid.uuid4()),
                user_id=access["user_id"],
                user_email=access.get("user_email", ""),
                resource_type=access["resource_type"],
                resource_id=access["resource_id"],
                resource_name=access.get("resource_name", access["resource_id"]),
                granted_at=access.get("granted_at"),
                last_used=self._storage.get_last_access(
                    access["user_id"],
                    access["resource_type"],
                    access["resource_id"],
                ),
                granted_by=access.get("granted_by"),
                risk_level=risk_level,
                metadata=access.get("metadata", {}),
            )
            items.append(item)

        return items

    def _assess_risk_level(self, access: Dict[str, Any]) -> str:
        """Assess risk level for an access item."""
        resource_type = access.get("resource_type", "")
        resource_id = access.get("resource_id", "")

        # High risk: admin roles, API keys, sensitive data access
        high_risk_patterns = ["admin", "superuser", "owner", "billing", "api_key"]
        for pattern in high_risk_patterns:
            if pattern in resource_type.lower() or pattern in resource_id.lower():
                return "high"

        # Critical risk: root access, all-access
        critical_patterns = ["root", "all_access", "super_admin"]
        for pattern in critical_patterns:
            if pattern in resource_type.lower() or pattern in resource_id.lower():
                return "critical"

        # Medium risk: write access, edit permissions
        medium_risk_patterns = ["write", "edit", "delete", "manage"]
        for pattern in medium_risk_patterns:
            if pattern in resource_type.lower() or pattern in resource_id.lower():
                return "medium"

        return "low"

    def _calculate_summary(self, items: List[AccessReviewItem]) -> Dict[str, Any]:
        """Calculate summary statistics for a review."""
        summary = {
            "total_items": len(items),
            "by_risk": {"low": 0, "medium": 0, "high": 0, "critical": 0},
            "by_type": {},
            "by_status": {},
            "unique_users": set(),
            "stale_count": 0,
        }

        now = datetime.now(timezone.utc)
        stale_threshold = now - timedelta(days=self.config.stale_credential_days)

        for item in items:
            summary["by_risk"][item.risk_level] += 1  # type: ignore[index]

            if item.resource_type not in summary["by_type"]:  # type: ignore[operator]
                summary["by_type"][item.resource_type] = 0  # type: ignore[index]
            summary["by_type"][item.resource_type] += 1  # type: ignore[index]

            status = item.status.value
            if status not in summary["by_status"]:  # type: ignore[operator]
                summary["by_status"][status] = 0  # type: ignore[index]
            summary["by_status"][status] += 1  # type: ignore[index]

            summary["unique_users"].add(item.user_id)  # type: ignore[attr-defined]

            if item.last_used and item.last_used < stale_threshold:
                summary["stale_count"] += 1  # type: ignore[operator]

        summary["unique_users"] = len(summary["unique_users"])  # type: ignore[arg-type]

        return summary

    def _get_mock_access_data(self) -> List[Dict[str, Any]]:
        """Get mock access data for testing."""
        # This would be replaced with actual RBAC/IAM data in production
        return [
            {
                "user_id": "user_1",
                "user_email": "user1@example.com",
                "resource_type": "role",
                "resource_id": "admin",
                "resource_name": "Administrator",
                "workspace_id": "workspace_1",
            },
            {
                "user_id": "user_2",
                "user_email": "user2@example.com",
                "resource_type": "api_key",
                "resource_id": "key_abc123",
                "resource_name": "Production API Key",
                "workspace_id": "workspace_1",
            },
        ]

    # =========================================================================
    # Review Processing
    # =========================================================================

    async def approve_item(
        self,
        review_id: str,
        item_id: str,
        decision_by: str,
        notes: Optional[str] = None,
    ) -> bool:
        """Approve a review item (keep access)."""
        return await self._process_item(
            review_id, item_id, ReviewItemStatus.APPROVED, decision_by, notes
        )

    async def revoke_item(
        self,
        review_id: str,
        item_id: str,
        decision_by: str,
        notes: Optional[str] = None,
    ) -> bool:
        """Revoke a review item (remove access)."""
        success = await self._process_item(
            review_id, item_id, ReviewItemStatus.REVOKED, decision_by, notes
        )

        if success:
            # Execute revocation
            review = self._storage.get_review(review_id)
            if review:
                item = next((i for i in review.items if i.item_id == item_id), None)
                if item:
                    await self._execute_revocation(item)

        return success

    async def _process_item(
        self,
        review_id: str,
        item_id: str,
        status: ReviewItemStatus,
        decision_by: str,
        notes: Optional[str],
    ) -> bool:
        """Process a review item decision."""
        review = self._storage.get_review(review_id)
        if not review:
            return False

        item = next((i for i in review.items if i.item_id == item_id), None)
        if not item:
            return False

        item.status = status
        item.decision_by = decision_by
        item.decision_at = datetime.now(timezone.utc)
        item.decision_notes = notes

        # Check if all items processed
        pending = [i for i in review.items if i.status == ReviewItemStatus.PENDING]
        if not pending:
            review.status = ReviewStatus.AWAITING_APPROVAL
            review.summary = self._calculate_summary(review.items)

        self._storage.save_review(review)
        return True

    async def complete_review(self, review_id: str, approved_by: str) -> Optional[AccessReview]:
        """Complete a review after all items processed."""
        review = self._storage.get_review(review_id)
        if not review:
            return None

        review.status = ReviewStatus.APPROVED
        review.completed_at = datetime.now(timezone.utc)
        review.summary["approved_by"] = approved_by

        self._storage.save_review(review)

        # Execute any pending revocations
        revoked_items = [i for i in review.items if i.status == ReviewItemStatus.REVOKED]
        for item in revoked_items:
            await self._execute_revocation(item)

        logger.info(f"Completed access review {review_id}")
        return review

    async def _execute_revocation(self, item: AccessReviewItem) -> None:
        """Execute access revocation."""
        logger.info(
            f"Revoking access: user={item.user_id}, "
            f"resource={item.resource_type}/{item.resource_id}"
        )

        for handler in self._revocation_handlers:
            try:
                handler(item.user_id, item.resource_type, item.resource_id)
            except Exception as e:
                logger.error(f"Error executing revocation: {e}")

    # =========================================================================
    # Notifications
    # =========================================================================

    async def _notify_review_created(self, review: AccessReview) -> None:
        """Send notification when review is created."""
        notification = {
            "type": "review_created",
            "review_id": review.review_id,
            "review_type": review.review_type.value,
            "items_count": len(review.items),
            "due_date": review.due_date.isoformat() if review.due_date else None,
            "high_risk_count": review.summary.get("by_risk", {}).get("high", 0)
            + review.summary.get("by_risk", {}).get("critical", 0),
        }

        for handler in self._notification_handlers:
            try:
                handler(notification)
            except Exception as e:
                logger.error(f"Error sending notification: {e}")

    async def _notify_review_expired(self, review: AccessReview) -> None:
        """Send notification when review expires."""
        notification = {
            "type": "review_expired",
            "review_id": review.review_id,
            "review_type": review.review_type.value,
            "items_count": len(review.items),
            "severity": "high",
        }

        for handler in self._notification_handlers:
            try:
                handler(notification)
            except Exception as e:
                logger.error(f"Error sending notification: {e}")

    # =========================================================================
    # Registration
    # =========================================================================

    def register_access_provider(self, provider: Callable[[], List[Dict[str, Any]]]) -> None:
        """Register an access data provider."""
        self._access_providers.append(provider)

    def register_revocation_handler(self, handler: Callable[[str, str, str], None]) -> None:
        """Register a revocation handler."""
        self._revocation_handlers.append(handler)

    def register_notification_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Register a notification handler."""
        self._notification_handlers.append(handler)

    # =========================================================================
    # Queries
    # =========================================================================

    def get_review(self, review_id: str) -> Optional[AccessReview]:
        """Get a review by ID."""
        return self._storage.get_review(review_id)

    def get_pending_reviews(self) -> List[AccessReview]:
        """Get all pending reviews."""
        return self._storage.get_pending_reviews()

    def get_stale_credentials(self) -> List[Dict[str, Any]]:
        """Get credentials not used in 90+ days."""
        return self._storage.get_stale_credentials(self.config.stale_credential_days)

    def record_access(self, user_id: str, resource_type: str, resource_id: str) -> None:
        """Record user access for tracking."""
        self._storage.record_user_access(user_id, resource_type, resource_id)


# =============================================================================
# Global Instance
# =============================================================================

_scheduler: Optional[AccessReviewScheduler] = None
_scheduler_lock = threading.Lock()


def get_access_review_scheduler(
    config: Optional[AccessReviewConfig] = None,
) -> AccessReviewScheduler:
    """Get or create the global access review scheduler."""
    global _scheduler
    with _scheduler_lock:
        if _scheduler is None:
            _scheduler = AccessReviewScheduler(config)
        return _scheduler


async def schedule_access_review(
    review_type: ReviewType = ReviewType.MONTHLY,
    scope_workspaces: Optional[List[str]] = None,
) -> AccessReview:
    """Convenience function to schedule an access review."""
    return await get_access_review_scheduler().create_review(
        review_type=review_type,
        scope_workspaces=scope_workspaces,
    )


__all__ = [
    # Types
    "ReviewType",
    "ReviewStatus",
    "ReviewItemStatus",
    "AccessReviewItem",
    "AccessReview",
    # Configuration
    "AccessReviewConfig",
    # Core
    "AccessReviewScheduler",
    "get_access_review_scheduler",
    # Convenience
    "schedule_access_review",
]
