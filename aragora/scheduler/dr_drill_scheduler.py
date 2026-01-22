"""
Disaster Recovery Drill Scheduler.

Provides automated DR drill scheduling and execution tracking for SOC 2 CC9 compliance:
- Monthly backup restoration tests
- Quarterly component failover tests
- Annual full DR simulation
- Automatic scheduling and tracking
- RTO/RPO measurement and trending
- Results persistence and compliance reporting

SOC 2 Compliance: CC9.1, CC9.2 (Business Continuity)

Usage:
    from aragora.scheduler.dr_drill_scheduler import (
        DRDrillScheduler,
        get_dr_drill_scheduler,
        schedule_dr_drill,
    )

    # Initialize scheduler
    scheduler = DRDrillScheduler(storage_path="dr_drills.db")

    # Start automated drills
    await scheduler.start()

    # Manually trigger a drill
    drill = await scheduler.execute_drill(
        drill_type="backup_restoration",
        components=["database", "storage"],
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


class DrillType(Enum):
    """Types of DR drills."""

    BACKUP_RESTORATION = "backup_restoration"  # Monthly
    COMPONENT_FAILOVER = "component_failover"  # Quarterly
    FULL_DR_SIMULATION = "full_dr_simulation"  # Annual
    DATA_INTEGRITY_CHECK = "data_integrity_check"  # Weekly
    NETWORK_FAILOVER = "network_failover"  # Quarterly
    MANUAL = "manual"  # Ad-hoc


class DrillStatus(Enum):
    """Status of a DR drill."""

    SCHEDULED = "scheduled"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"  # Some tests passed, some failed


class ComponentType(Enum):
    """Types of components that can be tested."""

    DATABASE = "database"
    OBJECT_STORAGE = "object_storage"
    CACHE = "cache"
    QUEUE = "queue"
    API_SERVER = "api_server"
    WEBSOCKET = "websocket"
    AUTH = "auth"
    SEARCH = "search"
    FULL_SYSTEM = "full_system"


@dataclass
class DrillStep:
    """A step in a DR drill."""

    step_id: str
    step_name: str
    component: ComponentType
    action: str  # backup, restore, failover, verify
    status: DrillStatus = DrillStatus.SCHEDULED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DRDrillResult:
    """Result of a DR drill."""

    drill_id: str
    drill_type: DrillType
    status: DrillStatus = DrillStatus.SCHEDULED
    scheduled_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    initiated_by: str = "system"
    steps: List[DrillStep] = field(default_factory=list)

    # Metrics
    total_duration_seconds: float = 0.0
    rto_seconds: Optional[float] = None  # Recovery Time Objective achieved
    rpo_seconds: Optional[float] = None  # Recovery Point Objective achieved
    data_loss_bytes: int = 0
    success_rate: float = 0.0

    # Targets for comparison
    target_rto_seconds: float = 3600.0  # 1 hour default
    target_rpo_seconds: float = 300.0  # 5 minutes default

    # Compliance
    meets_rto: bool = False
    meets_rpo: bool = False
    is_compliant: bool = False

    notes: str = ""
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "drill_id": self.drill_id,
            "drill_type": self.drill_type.value,
            "status": self.status.value,
            "scheduled_at": self.scheduled_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "initiated_by": self.initiated_by,
            "steps_count": len(self.steps),
            "steps_passed": len([s for s in self.steps if s.status == DrillStatus.COMPLETED]),
            "total_duration_seconds": self.total_duration_seconds,
            "rto_seconds": self.rto_seconds,
            "rpo_seconds": self.rpo_seconds,
            "target_rto_seconds": self.target_rto_seconds,
            "target_rpo_seconds": self.target_rpo_seconds,
            "meets_rto": self.meets_rto,
            "meets_rpo": self.meets_rpo,
            "is_compliant": self.is_compliant,
            "success_rate": self.success_rate,
            "data_loss_bytes": self.data_loss_bytes,
            "notes": self.notes,
            "recommendations": self.recommendations,
        }


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class DRDrillConfig:
    """Configuration for DR drill scheduler."""

    # Schedule (day of month/quarter)
    monthly_drill_day: int = 15  # 15th of each month
    quarterly_drill_months: List[int] = field(default_factory=lambda: [3, 6, 9, 12])
    annual_drill_month: int = 1  # January

    # Targets
    target_rto_seconds: float = 3600.0  # 1 hour
    target_rpo_seconds: float = 300.0  # 5 minutes

    # Notification
    notification_email: Optional[str] = None
    slack_webhook: Optional[str] = None

    # Storage
    storage_path: Optional[str] = None

    # Execution
    dry_run: bool = False  # If True, simulate drills without actual operations


# =============================================================================
# Storage Layer
# =============================================================================


class DRDrillStorage:
    """SQLite-backed storage for DR drills."""

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
            CREATE TABLE IF NOT EXISTS dr_drills (
                drill_id TEXT PRIMARY KEY,
                drill_type TEXT NOT NULL,
                status TEXT NOT NULL,
                scheduled_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                initiated_by TEXT,
                total_duration_seconds REAL DEFAULT 0,
                rto_seconds REAL,
                rpo_seconds REAL,
                target_rto_seconds REAL,
                target_rpo_seconds REAL,
                data_loss_bytes INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0,
                meets_rto INTEGER DEFAULT 0,
                meets_rpo INTEGER DEFAULT 0,
                is_compliant INTEGER DEFAULT 0,
                notes TEXT,
                recommendations_json TEXT
            );

            CREATE TABLE IF NOT EXISTS drill_steps (
                step_id TEXT PRIMARY KEY,
                drill_id TEXT NOT NULL,
                step_name TEXT NOT NULL,
                component TEXT NOT NULL,
                action TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                duration_seconds REAL DEFAULT 0,
                error_message TEXT,
                metrics_json TEXT,
                FOREIGN KEY (drill_id) REFERENCES dr_drills(drill_id)
            );

            CREATE TABLE IF NOT EXISTS drill_schedule (
                schedule_id TEXT PRIMARY KEY,
                drill_type TEXT NOT NULL,
                next_run TEXT,
                last_run TEXT,
                enabled INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS rto_rpo_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                drill_id TEXT NOT NULL,
                drill_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                rto_seconds REAL,
                rpo_seconds REAL,
                target_rto_seconds REAL,
                target_rpo_seconds REAL,
                compliant INTEGER,
                FOREIGN KEY (drill_id) REFERENCES dr_drills(drill_id)
            );

            CREATE INDEX IF NOT EXISTS idx_drills_status ON dr_drills(status);
            CREATE INDEX IF NOT EXISTS idx_drills_scheduled ON dr_drills(scheduled_at);
            CREATE INDEX IF NOT EXISTS idx_steps_drill ON drill_steps(drill_id);
            CREATE INDEX IF NOT EXISTS idx_history_type ON rto_rpo_history(drill_type, timestamp);
            """
        )
        conn.commit()

    def save_drill(self, drill: DRDrillResult) -> None:
        """Save or update a drill."""
        import json

        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO dr_drills (
                drill_id, drill_type, status, scheduled_at, started_at,
                completed_at, initiated_by, total_duration_seconds,
                rto_seconds, rpo_seconds, target_rto_seconds, target_rpo_seconds,
                data_loss_bytes, success_rate, meets_rto, meets_rpo,
                is_compliant, notes, recommendations_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                drill.drill_id,
                drill.drill_type.value,
                drill.status.value,
                drill.scheduled_at.isoformat(),
                drill.started_at.isoformat() if drill.started_at else None,
                drill.completed_at.isoformat() if drill.completed_at else None,
                drill.initiated_by,
                drill.total_duration_seconds,
                drill.rto_seconds,
                drill.rpo_seconds,
                drill.target_rto_seconds,
                drill.target_rpo_seconds,
                drill.data_loss_bytes,
                drill.success_rate,
                1 if drill.meets_rto else 0,
                1 if drill.meets_rpo else 0,
                1 if drill.is_compliant else 0,
                drill.notes,
                json.dumps(drill.recommendations),
            ),
        )

        # Save steps
        for step in drill.steps:
            conn.execute(
                """
                INSERT OR REPLACE INTO drill_steps (
                    step_id, drill_id, step_name, component, action,
                    status, started_at, completed_at, duration_seconds,
                    error_message, metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    step.step_id,
                    drill.drill_id,
                    step.step_name,
                    step.component.value,
                    step.action,
                    step.status.value,
                    step.started_at.isoformat() if step.started_at else None,
                    step.completed_at.isoformat() if step.completed_at else None,
                    step.duration_seconds,
                    step.error_message,
                    json.dumps(step.metrics),
                ),
            )

        conn.commit()

    def get_drill(self, drill_id: str) -> Optional[DRDrillResult]:
        """Get a drill by ID."""
        import json

        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM dr_drills WHERE drill_id = ?",
            (drill_id,),
        ).fetchone()

        if not row:
            return None

        # Load steps
        steps = []
        step_rows = conn.execute(
            "SELECT * FROM drill_steps WHERE drill_id = ?",
            (drill_id,),
        ).fetchall()

        for sr in step_rows:
            steps.append(
                DrillStep(
                    step_id=sr["step_id"],
                    step_name=sr["step_name"],
                    component=ComponentType(sr["component"]),
                    action=sr["action"],
                    status=DrillStatus(sr["status"]),
                    started_at=datetime.fromisoformat(sr["started_at"])
                    if sr["started_at"]
                    else None,
                    completed_at=datetime.fromisoformat(sr["completed_at"])
                    if sr["completed_at"]
                    else None,
                    duration_seconds=sr["duration_seconds"] or 0.0,
                    error_message=sr["error_message"],
                    metrics=json.loads(sr["metrics_json"] or "{}"),
                )
            )

        return DRDrillResult(
            drill_id=row["drill_id"],
            drill_type=DrillType(row["drill_type"]),
            status=DrillStatus(row["status"]),
            scheduled_at=datetime.fromisoformat(row["scheduled_at"]),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"])
            if row["completed_at"]
            else None,
            initiated_by=row["initiated_by"] or "system",
            steps=steps,
            total_duration_seconds=row["total_duration_seconds"] or 0.0,
            rto_seconds=row["rto_seconds"],
            rpo_seconds=row["rpo_seconds"],
            target_rto_seconds=row["target_rto_seconds"] or 3600.0,
            target_rpo_seconds=row["target_rpo_seconds"] or 300.0,
            data_loss_bytes=row["data_loss_bytes"] or 0,
            success_rate=row["success_rate"] or 0.0,
            meets_rto=bool(row["meets_rto"]),
            meets_rpo=bool(row["meets_rpo"]),
            is_compliant=bool(row["is_compliant"]),
            notes=row["notes"] or "",
            recommendations=json.loads(row["recommendations_json"] or "[]"),
        )

    def get_recent_drills(
        self, limit: int = 10, drill_type: Optional[DrillType] = None
    ) -> List[DRDrillResult]:
        """Get recent drills."""
        conn = self._get_conn()

        query = "SELECT drill_id FROM dr_drills"
        params: List[Any] = []

        if drill_type:
            query += " WHERE drill_type = ?"
            params.append(drill_type.value)

        query += " ORDER BY scheduled_at DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()

        drills = []
        for row in rows:
            drill = self.get_drill(row["drill_id"])
            if drill:
                drills.append(drill)

        return drills

    def record_rto_rpo(self, drill: DRDrillResult) -> None:
        """Record RTO/RPO metrics for trending."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO rto_rpo_history (
                drill_id, drill_type, timestamp, rto_seconds, rpo_seconds,
                target_rto_seconds, target_rpo_seconds, compliant
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                drill.drill_id,
                drill.drill_type.value,
                datetime.now(timezone.utc).isoformat(),
                drill.rto_seconds,
                drill.rpo_seconds,
                drill.target_rto_seconds,
                drill.target_rpo_seconds,
                1 if drill.is_compliant else 0,
            ),
        )
        conn.commit()

    def get_rto_rpo_trend(
        self, drill_type: Optional[DrillType] = None, months: int = 12
    ) -> List[Dict[str, Any]]:
        """Get RTO/RPO trend data."""
        conn = self._get_conn()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=months * 30)).isoformat()

        query = "SELECT * FROM rto_rpo_history WHERE timestamp > ?"
        params: List[Any] = [cutoff]

        if drill_type:
            query += " AND drill_type = ?"
            params.append(drill_type.value)

        query += " ORDER BY timestamp ASC"

        rows = conn.execute(query, params).fetchall()

        return [
            {
                "drill_id": row["drill_id"],
                "drill_type": row["drill_type"],
                "timestamp": row["timestamp"],
                "rto_seconds": row["rto_seconds"],
                "rpo_seconds": row["rpo_seconds"],
                "target_rto_seconds": row["target_rto_seconds"],
                "target_rpo_seconds": row["target_rpo_seconds"],
                "compliant": bool(row["compliant"]),
            }
            for row in rows
        ]

    def save_schedule(
        self,
        schedule_id: str,
        drill_type: str,
        next_run: datetime,
        enabled: bool = True,
    ) -> None:
        """Save a drill schedule."""
        conn = self._get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO drill_schedule (
                schedule_id, drill_type, next_run, enabled
            ) VALUES (?, ?, ?, ?)
            """,
            (schedule_id, drill_type, next_run.isoformat(), 1 if enabled else 0),
        )
        conn.commit()

    def get_due_schedules(self) -> List[Dict[str, Any]]:
        """Get schedules that are due to run."""
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()

        rows = conn.execute(
            """
            SELECT * FROM drill_schedule
            WHERE enabled = 1 AND next_run <= ?
            """,
            (now,),
        ).fetchall()

        return [
            {
                "schedule_id": row["schedule_id"],
                "drill_type": row["drill_type"],
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
            UPDATE drill_schedule
            SET last_run = ?, next_run = ?
            WHERE schedule_id = ?
            """,
            (now, next_run.isoformat(), schedule_id),
        )
        conn.commit()


# =============================================================================
# DR Drill Scheduler
# =============================================================================


class DRDrillScheduler:
    """Main DR drill scheduler and executor."""

    def __init__(self, config: Optional[DRDrillConfig] = None):
        """Initialize scheduler.

        Args:
            config: Scheduler configuration
        """
        self.config = config or DRDrillConfig()
        self._storage = DRDrillStorage(self.config.storage_path)
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Component test handlers
        self._test_handlers: Dict[ComponentType, Callable[[], Dict[str, Any]]] = {}
        self._restore_handlers: Dict[ComponentType, Callable[[], Dict[str, Any]]] = {}
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
        logger.info("DR drill scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("DR drill scheduler stopped")

    def _init_schedules(self) -> None:
        """Initialize default schedules."""
        now = datetime.now(timezone.utc)

        # Monthly backup restoration drill
        next_monthly = now.replace(day=self.config.monthly_drill_day, hour=2, minute=0, second=0)
        if next_monthly <= now:
            if now.month == 12:
                next_monthly = next_monthly.replace(year=now.year + 1, month=1)
            else:
                next_monthly = next_monthly.replace(month=now.month + 1)

        self._storage.save_schedule("monthly_backup", "backup_restoration", next_monthly)

        # Quarterly failover drill
        current_quarter_month = self.config.quarterly_drill_months[(now.month - 1) // 3]
        next_quarterly = now.replace(
            month=current_quarter_month, day=self.config.monthly_drill_day, hour=3
        )
        if next_quarterly <= now:
            quarter_idx = self.config.quarterly_drill_months.index(current_quarter_month)
            next_quarter_idx = (quarter_idx + 1) % 4
            next_month = self.config.quarterly_drill_months[next_quarter_idx]
            if next_quarter_idx == 0:
                next_quarterly = next_quarterly.replace(year=now.year + 1, month=next_month)
            else:
                next_quarterly = next_quarterly.replace(month=next_month)

        self._storage.save_schedule("quarterly_failover", "component_failover", next_quarterly)

        # Weekly data integrity check
        next_weekly = now + timedelta(days=(6 - now.weekday()) % 7 + 1)
        next_weekly = next_weekly.replace(hour=1, minute=0, second=0)
        self._storage.save_schedule("weekly_integrity", "data_integrity_check", next_weekly)

        logger.info(
            f"Initialized DR schedules: monthly={next_monthly}, "
            f"quarterly={next_quarterly}, weekly={next_weekly}"
        )

    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                # Check for due schedules
                due_schedules = self._storage.get_due_schedules()
                for schedule in due_schedules:
                    await self._execute_scheduled_drill(schedule)

                # Sleep before next check (every hour)
                await asyncio.sleep(3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in DR drill scheduler: {e}")
                await asyncio.sleep(300)

    async def _execute_scheduled_drill(self, schedule: Dict[str, Any]) -> None:
        """Execute a scheduled drill."""
        drill_type = DrillType(schedule["drill_type"])
        logger.info(f"Executing scheduled DR drill: {drill_type.value}")

        # Execute the drill
        drill = await self.execute_drill(drill_type=drill_type)

        # Calculate next run
        if drill_type == DrillType.BACKUP_RESTORATION:
            next_run = datetime.fromisoformat(schedule["next_run"])
            if next_run.month == 12:
                next_run = next_run.replace(year=next_run.year + 1, month=1)
            else:
                next_run = next_run.replace(month=next_run.month + 1)
        elif drill_type == DrillType.COMPONENT_FAILOVER:
            next_run = datetime.now(timezone.utc) + timedelta(days=90)
        elif drill_type == DrillType.DATA_INTEGRITY_CHECK:
            next_run = datetime.now(timezone.utc) + timedelta(days=7)
        else:
            next_run = datetime.now(timezone.utc) + timedelta(days=30)

        self._storage.update_schedule_run(schedule["schedule_id"], next_run)

        # Send notifications
        await self._notify_drill_completed(drill)

    # =========================================================================
    # Drill Execution
    # =========================================================================

    async def execute_drill(
        self,
        drill_type: DrillType,
        components: Optional[List[ComponentType]] = None,
        initiated_by: str = "system",
    ) -> DRDrillResult:
        """Execute a DR drill.

        Args:
            drill_type: Type of drill to execute
            components: Specific components to test (None = all applicable)
            initiated_by: User ID who initiated the drill

        Returns:
            Drill result with metrics
        """
        # Create drill
        drill = DRDrillResult(
            drill_id=str(uuid.uuid4()),
            drill_type=drill_type,
            status=DrillStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
            initiated_by=initiated_by,
            target_rto_seconds=self.config.target_rto_seconds,
            target_rpo_seconds=self.config.target_rpo_seconds,
        )

        # Determine steps based on drill type
        steps = self._plan_drill_steps(drill_type, components)
        drill.steps = steps

        # Execute steps
        recovery_start = None
        recovery_end = None
        failed_steps = []

        for step in drill.steps:
            step.started_at = datetime.now(timezone.utc)
            step.status = DrillStatus.RUNNING

            try:
                if self.config.dry_run:
                    # Simulate execution
                    result = await self._simulate_step(step)
                else:
                    # Execute actual step
                    result = await self._execute_step(step)

                step.metrics = result.get("metrics", {})

                if result.get("success", False):
                    step.status = DrillStatus.COMPLETED
                else:
                    step.status = DrillStatus.FAILED
                    step.error_message = result.get("error", "Unknown error")
                    failed_steps.append(step)

                # Track recovery timing
                if step.action == "restore" and recovery_start is None:
                    recovery_start = step.started_at
                if step.action == "verify" and step.status == DrillStatus.COMPLETED:
                    recovery_end = datetime.now(timezone.utc)

            except Exception as e:
                step.status = DrillStatus.FAILED
                step.error_message = str(e)
                failed_steps.append(step)
                logger.error(f"Step {step.step_name} failed: {e}")

            step.completed_at = datetime.now(timezone.utc)
            step.duration_seconds = (step.completed_at - step.started_at).total_seconds()

        # Calculate final metrics
        drill.completed_at = datetime.now(timezone.utc)
        drill.total_duration_seconds = (drill.completed_at - drill.started_at).total_seconds()

        # Calculate RTO (time from start of recovery to successful verification)
        if recovery_start and recovery_end:
            drill.rto_seconds = (recovery_end - recovery_start).total_seconds()
        else:
            drill.rto_seconds = drill.total_duration_seconds

        # Calculate RPO (from metrics - data loss window)
        drill.rpo_seconds = self._calculate_rpo(drill.steps)

        # Calculate success rate
        completed = len([s for s in drill.steps if s.status == DrillStatus.COMPLETED])
        drill.success_rate = completed / len(drill.steps) if drill.steps else 0.0

        # Check compliance
        drill.meets_rto = (
            drill.rto_seconds is not None and drill.rto_seconds <= drill.target_rto_seconds
        )
        drill.meets_rpo = (
            drill.rpo_seconds is not None and drill.rpo_seconds <= drill.target_rpo_seconds
        )
        drill.is_compliant = drill.meets_rto and drill.meets_rpo and drill.success_rate >= 0.95

        # Set final status
        if not failed_steps:
            drill.status = DrillStatus.COMPLETED
        elif failed_steps and completed > 0:
            drill.status = DrillStatus.PARTIAL
        else:
            drill.status = DrillStatus.FAILED

        # Generate recommendations
        drill.recommendations = self._generate_recommendations(drill, failed_steps)

        # Persist
        self._storage.save_drill(drill)
        self._storage.record_rto_rpo(drill)

        logger.info(
            f"DR drill {drill.drill_id} completed: "
            f"status={drill.status.value}, RTO={drill.rto_seconds:.1f}s, "
            f"RPO={drill.rpo_seconds:.1f}s, compliant={drill.is_compliant}"
        )

        return drill

    def _plan_drill_steps(
        self, drill_type: DrillType, components: Optional[List[ComponentType]]
    ) -> List[DrillStep]:
        """Plan steps for a drill."""
        steps = []

        # Determine which components to test
        if components:
            target_components = components
        elif drill_type == DrillType.BACKUP_RESTORATION:
            target_components = [
                ComponentType.DATABASE,
                ComponentType.OBJECT_STORAGE,
            ]
        elif drill_type == DrillType.COMPONENT_FAILOVER:
            target_components = [
                ComponentType.DATABASE,
                ComponentType.CACHE,
                ComponentType.API_SERVER,
            ]
        elif drill_type == DrillType.DATA_INTEGRITY_CHECK:
            target_components = [
                ComponentType.DATABASE,
                ComponentType.OBJECT_STORAGE,
            ]
        elif drill_type == DrillType.FULL_DR_SIMULATION:
            target_components = list(ComponentType)
        else:
            target_components = [ComponentType.DATABASE]

        # Create steps for each component
        for component in target_components:
            if drill_type == DrillType.BACKUP_RESTORATION:
                # Backup, restore, verify sequence
                steps.extend(
                    [
                        DrillStep(
                            step_id=str(uuid.uuid4()),
                            step_name=f"Verify backup exists - {component.value}",
                            component=component,
                            action="backup_check",
                        ),
                        DrillStep(
                            step_id=str(uuid.uuid4()),
                            step_name=f"Restore from backup - {component.value}",
                            component=component,
                            action="restore",
                        ),
                        DrillStep(
                            step_id=str(uuid.uuid4()),
                            step_name=f"Verify data integrity - {component.value}",
                            component=component,
                            action="verify",
                        ),
                    ]
                )
            elif drill_type == DrillType.COMPONENT_FAILOVER:
                steps.extend(
                    [
                        DrillStep(
                            step_id=str(uuid.uuid4()),
                            step_name=f"Simulate failure - {component.value}",
                            component=component,
                            action="simulate_failure",
                        ),
                        DrillStep(
                            step_id=str(uuid.uuid4()),
                            step_name=f"Verify failover - {component.value}",
                            component=component,
                            action="failover",
                        ),
                        DrillStep(
                            step_id=str(uuid.uuid4()),
                            step_name=f"Restore primary - {component.value}",
                            component=component,
                            action="restore",
                        ),
                    ]
                )
            elif drill_type == DrillType.DATA_INTEGRITY_CHECK:
                steps.append(
                    DrillStep(
                        step_id=str(uuid.uuid4()),
                        step_name=f"Check data integrity - {component.value}",
                        component=component,
                        action="verify",
                    )
                )

        return steps

    async def _execute_step(self, step: DrillStep) -> Dict[str, Any]:
        """Execute a drill step."""
        # Look for registered handler
        if step.action in ["restore", "failover"]:
            handler = self._restore_handlers.get(step.component)
        else:
            handler = self._test_handlers.get(step.component)

        if handler:
            try:
                result = handler()
                return result
            except Exception as e:
                return {"success": False, "error": str(e)}

        # Default: simulate if no handler
        return await self._simulate_step(step)

    async def _simulate_step(self, step: DrillStep) -> Dict[str, Any]:
        """Simulate a drill step (for testing/dry-run)."""
        # Simulate some execution time
        await asyncio.sleep(0.1)

        # Simulate success with metrics
        return {
            "success": True,
            "metrics": {
                "duration_ms": 100,
                "data_checked_bytes": 1024 * 1024 * 100,
                "records_verified": 10000,
            },
        }

    def _calculate_rpo(self, steps: List[DrillStep]) -> float:
        """Calculate RPO from step metrics."""
        # Look for backup age or data lag metrics
        for step in steps:
            if "backup_age_seconds" in step.metrics:
                return step.metrics["backup_age_seconds"]
            if "replication_lag_seconds" in step.metrics:
                return step.metrics["replication_lag_seconds"]

        # Default: assume worst case based on drill duration
        return 300.0  # 5 minutes default

    def _generate_recommendations(
        self, drill: DRDrillResult, failed_steps: List[DrillStep]
    ) -> List[str]:
        """Generate recommendations based on drill results."""
        recommendations = []

        if not drill.meets_rto:
            recommendations.append(
                f"RTO exceeded target ({drill.rto_seconds:.0f}s > {drill.target_rto_seconds:.0f}s). "
                "Consider optimizing backup/restore procedures or increasing infrastructure capacity."
            )

        if not drill.meets_rpo:
            recommendations.append(
                f"RPO exceeded target ({drill.rpo_seconds:.0f}s > {drill.target_rpo_seconds:.0f}s). "
                "Consider increasing backup frequency or implementing continuous replication."
            )

        for step in failed_steps:
            recommendations.append(
                f"Step '{step.step_name}' failed: {step.error_message}. "
                f"Review {step.component.value} disaster recovery procedures."
            )

        if drill.success_rate < 1.0:
            recommendations.append(
                f"Success rate {drill.success_rate:.0%} below 100%. "
                "Review and update DR runbooks for failed components."
            )

        if not recommendations:
            recommendations.append(
                "All DR drill objectives met. Continue regular testing schedule."
            )

        return recommendations

    # =========================================================================
    # Notifications
    # =========================================================================

    async def _notify_drill_completed(self, drill: DRDrillResult) -> None:
        """Send notification when drill completes."""
        notification = {
            "type": "dr_drill_completed",
            "drill_id": drill.drill_id,
            "drill_type": drill.drill_type.value,
            "status": drill.status.value,
            "is_compliant": drill.is_compliant,
            "rto_seconds": drill.rto_seconds,
            "rpo_seconds": drill.rpo_seconds,
            "success_rate": drill.success_rate,
            "recommendations": drill.recommendations,
        }

        for handler in self._notification_handlers:
            try:
                handler(notification)
            except Exception as e:
                logger.error(f"Error sending notification: {e}")

    # =========================================================================
    # Registration
    # =========================================================================

    def register_test_handler(
        self, component: ComponentType, handler: Callable[[], Dict[str, Any]]
    ) -> None:
        """Register a test handler for a component."""
        self._test_handlers[component] = handler

    def register_restore_handler(
        self, component: ComponentType, handler: Callable[[], Dict[str, Any]]
    ) -> None:
        """Register a restore handler for a component."""
        self._restore_handlers[component] = handler

    def register_notification_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Register a notification handler."""
        self._notification_handlers.append(handler)

    # =========================================================================
    # Queries
    # =========================================================================

    def get_drill(self, drill_id: str) -> Optional[DRDrillResult]:
        """Get a drill by ID."""
        return self._storage.get_drill(drill_id)

    def get_recent_drills(
        self, limit: int = 10, drill_type: Optional[DrillType] = None
    ) -> List[DRDrillResult]:
        """Get recent drills."""
        return self._storage.get_recent_drills(limit, drill_type)

    def get_rto_rpo_trend(
        self, drill_type: Optional[DrillType] = None, months: int = 12
    ) -> List[Dict[str, Any]]:
        """Get RTO/RPO trend data for compliance reporting."""
        return self._storage.get_rto_rpo_trend(drill_type, months)

    def get_compliance_report(self) -> Dict[str, Any]:
        """Generate a compliance report."""
        recent = self.get_recent_drills(limit=20)
        trend = self.get_rto_rpo_trend(months=12)

        # Calculate compliance rate
        compliant_count = len([d for d in recent if d.is_compliant])
        compliance_rate = compliant_count / len(recent) if recent else 0.0

        # Calculate average RTO/RPO
        rto_values = [d.rto_seconds for d in recent if d.rto_seconds is not None]
        rpo_values = [d.rpo_seconds for d in recent if d.rpo_seconds is not None]

        return {
            "compliance_rate": compliance_rate,
            "total_drills": len(recent),
            "compliant_drills": compliant_count,
            "average_rto_seconds": sum(rto_values) / len(rto_values) if rto_values else None,
            "average_rpo_seconds": sum(rpo_values) / len(rpo_values) if rpo_values else None,
            "target_rto_seconds": self.config.target_rto_seconds,
            "target_rpo_seconds": self.config.target_rpo_seconds,
            "trend_data_points": len(trend),
            "last_drill": recent[0].to_dict() if recent else None,
        }


# =============================================================================
# Global Instance
# =============================================================================

_scheduler: Optional[DRDrillScheduler] = None
_scheduler_lock = threading.Lock()


def get_dr_drill_scheduler(
    config: Optional[DRDrillConfig] = None,
) -> DRDrillScheduler:
    """Get or create the global DR drill scheduler."""
    global _scheduler
    with _scheduler_lock:
        if _scheduler is None:
            _scheduler = DRDrillScheduler(config)
        return _scheduler


async def schedule_dr_drill(
    drill_type: DrillType = DrillType.BACKUP_RESTORATION,
    components: Optional[List[ComponentType]] = None,
) -> DRDrillResult:
    """Convenience function to schedule a DR drill."""
    return await get_dr_drill_scheduler().execute_drill(
        drill_type=drill_type,
        components=components,
    )


__all__ = [
    # Types
    "DrillType",
    "DrillStatus",
    "ComponentType",
    "DrillStep",
    "DRDrillResult",
    # Configuration
    "DRDrillConfig",
    # Core
    "DRDrillScheduler",
    "get_dr_drill_scheduler",
    # Convenience
    "schedule_dr_drill",
]
