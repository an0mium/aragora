"""SQLite-backed persistent store for DecisionPlans.

Provides CRUD operations for DecisionPlan persistence with filtering
by debate_id and approval status. Replaces the in-memory store in
executor.py for production use.

Usage:
    store = PlanStore()
    store.create(plan)
    plan = store.get(plan_id)
    plans = store.list(status=PlanStatus.AWAITING_APPROVAL, limit=20)
    store.update_status(plan_id, PlanStatus.APPROVED, approved_by="user-123")
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from aragora.pipeline.decision_plan.core import (
    ApprovalMode,
    ApprovalRecord,
    BudgetAllocation,
    DecisionPlan,
    PlanStatus,
)

logger = logging.getLogger(__name__)

# Default database location
_DEFAULT_DB_DIR = os.environ.get("ARAGORA_DATA_DIR", str(Path.home() / ".aragora"))
_DEFAULT_DB_PATH = os.path.join(_DEFAULT_DB_DIR, "plans.db")


def _get_db_path() -> str:
    """Resolve the plan store database path."""
    try:
        from aragora.persistence.db_config import resolve_db_path

        return resolve_db_path("plans.db")
    except ImportError:
        return _DEFAULT_DB_PATH


class PlanStore:
    """SQLite-backed store for DecisionPlan objects.

    Thread-safe via SQLite WAL mode. Each method creates its own
    connection to support concurrent access from handler threads.
    """

    def __init__(self, db_path: str | None = None) -> None:
        self._db_path = db_path or _get_db_path()
        self._ensure_dir()
        self._ensure_table()

    def _ensure_dir(self) -> None:
        """Create parent directory if needed."""
        parent = Path(self._db_path).parent
        parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        """Create a new connection with WAL mode."""
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_table(self) -> None:
        """Create the plans table if it does not exist."""
        conn = self._connect()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS plans (
                    id TEXT PRIMARY KEY,
                    debate_id TEXT NOT NULL,
                    task TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'created',
                    approval_mode TEXT NOT NULL DEFAULT 'risk_based',
                    approved_by TEXT,
                    rejection_reason TEXT,
                    budget_json TEXT,
                    approval_record_json TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    approved_at TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_plans_debate_id
                ON plans(debate_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_plans_status
                ON plans(status)
            """)
            conn.commit()
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # CRUD
    # -------------------------------------------------------------------------

    def create(self, plan: DecisionPlan) -> None:
        """Insert a new plan into the store."""
        now = datetime.utcnow().isoformat()
        budget_json = json.dumps(plan.budget.to_dict()) if plan.budget else "{}"
        approval_json = (
            json.dumps(plan.approval_record.to_dict()) if plan.approval_record else None
        )
        metadata_json = json.dumps(plan.metadata) if plan.metadata else "{}"

        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO plans (
                    id, debate_id, task, status, approval_mode,
                    approved_by, rejection_reason, budget_json,
                    approval_record_json, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan.id,
                    plan.debate_id,
                    plan.task,
                    plan.status.value,
                    plan.approval_mode.value,
                    plan.approval_record.approver_id if plan.approval_record else None,
                    plan.approval_record.reason
                    if plan.approval_record and not plan.approval_record.approved
                    else None,
                    budget_json,
                    approval_json,
                    metadata_json,
                    plan.created_at.isoformat(),
                    now,
                ),
            )
            conn.commit()
            logger.info("Stored plan %s for debate %s", plan.id, plan.debate_id)
        finally:
            conn.close()

    def get(self, plan_id: str) -> DecisionPlan | None:
        """Retrieve a plan by ID."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM plans WHERE id = ?", (plan_id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_plan(row)
        finally:
            conn.close()

    def list(
        self,
        *,
        debate_id: str | None = None,
        status: PlanStatus | str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[DecisionPlan]:
        """List plans with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []

        if debate_id is not None:
            clauses.append("debate_id = ?")
            params.append(debate_id)
        if status is not None:
            status_val = status.value if isinstance(status, PlanStatus) else status
            clauses.append("status = ?")
            params.append(status_val)

        where = ""
        if clauses:
            where = "WHERE " + " AND ".join(clauses)

        query = f"SELECT * FROM plans {where} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        conn = self._connect()
        try:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_plan(row) for row in rows]
        finally:
            conn.close()

    def count(
        self,
        *,
        debate_id: str | None = None,
        status: PlanStatus | str | None = None,
    ) -> int:
        """Count plans matching the given filters."""
        clauses: list[str] = []
        params: list[Any] = []

        if debate_id is not None:
            clauses.append("debate_id = ?")
            params.append(debate_id)
        if status is not None:
            status_val = status.value if isinstance(status, PlanStatus) else status
            clauses.append("status = ?")
            params.append(status_val)

        where = ""
        if clauses:
            where = "WHERE " + " AND ".join(clauses)

        conn = self._connect()
        try:
            row = conn.execute(f"SELECT COUNT(*) FROM plans {where}", params).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def update_status(
        self,
        plan_id: str,
        status: PlanStatus,
        *,
        approved_by: str | None = None,
        rejection_reason: str | None = None,
    ) -> bool:
        """Update a plan's status. Returns True if the plan was found and updated."""
        now = datetime.utcnow().isoformat()
        fields = ["status = ?", "updated_at = ?"]
        params: list[Any] = [status.value, now]

        if approved_by is not None:
            fields.append("approved_by = ?")
            params.append(approved_by)

        if rejection_reason is not None:
            fields.append("rejection_reason = ?")
            params.append(rejection_reason)

        if status == PlanStatus.APPROVED:
            fields.append("approved_at = ?")
            params.append(now)
            # Store approval record
            approval_record = ApprovalRecord(
                approved=True,
                approver_id=approved_by or "unknown",
                reason="",
            )
            fields.append("approval_record_json = ?")
            params.append(json.dumps(approval_record.to_dict()))

        if status == PlanStatus.REJECTED:
            approval_record = ApprovalRecord(
                approved=False,
                approver_id=approved_by or "unknown",
                reason=rejection_reason or "",
            )
            fields.append("approval_record_json = ?")
            params.append(json.dumps(approval_record.to_dict()))

        params.append(plan_id)

        conn = self._connect()
        try:
            cursor = conn.execute(
                f"UPDATE plans SET {', '.join(fields)} WHERE id = ?",
                params,
            )
            conn.commit()
            updated = cursor.rowcount > 0
            if updated:
                logger.info("Updated plan %s to status %s", plan_id, status.value)
            return updated
        finally:
            conn.close()

    def delete(self, plan_id: str) -> bool:
        """Delete a plan by ID. Returns True if deleted."""
        conn = self._connect()
        try:
            cursor = conn.execute("DELETE FROM plans WHERE id = ?", (plan_id,))
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _row_to_plan(row: sqlite3.Row) -> DecisionPlan:
        """Convert a database row to a DecisionPlan."""
        budget_data = json.loads(row["budget_json"] or "{}")
        budget = BudgetAllocation(
            limit_usd=budget_data.get("limit_usd"),
            estimated_usd=budget_data.get("estimated_usd", 0.0),
            spent_usd=budget_data.get("spent_usd", 0.0),
            debate_cost_usd=budget_data.get("debate_cost_usd", 0.0),
            implementation_cost_usd=budget_data.get("implementation_cost_usd", 0.0),
            verification_cost_usd=budget_data.get("verification_cost_usd", 0.0),
        )

        approval_record = None
        if row["approval_record_json"]:
            ar_data = json.loads(row["approval_record_json"])
            approval_record = ApprovalRecord(
                approved=ar_data.get("approved", False),
                approver_id=ar_data.get("approver_id", ""),
                reason=ar_data.get("reason", ""),
                conditions=ar_data.get("conditions", []),
            )

        metadata = json.loads(row["metadata_json"] or "{}")

        created_at = datetime.fromisoformat(row["created_at"])

        plan = DecisionPlan(
            id=row["id"],
            debate_id=row["debate_id"],
            task=row["task"],
            status=PlanStatus(row["status"]),
            approval_mode=ApprovalMode(row["approval_mode"]),
            budget=budget,
            approval_record=approval_record,
            metadata=metadata,
            created_at=created_at,
        )

        return plan


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_store: PlanStore | None = None


def get_plan_store() -> PlanStore:
    """Get or create the module-level PlanStore singleton."""
    global _store
    if _store is None:
        _store = PlanStore()
    return _store


__all__ = ["PlanStore", "get_plan_store"]
