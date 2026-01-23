"""
Expense Storage Backends.

Persistent storage for expense records with OCR processing results.

Backends:
- InMemoryExpenseStore: For testing
- SQLiteExpenseStore: For single-instance deployments
- PostgresExpenseStore: For multi-instance production

Usage:
    from aragora.storage.expense_store import get_expense_store

    store = get_expense_store()
    await store.save(expense_data)
    expense = await store.get("exp_123")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from asyncpg import Pool

logger = logging.getLogger(__name__)

# Global singleton
_expense_store: Optional["ExpenseStoreBackend"] = None
_store_lock = threading.RLock()


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return str(obj)
        return super().default(obj)


def decimal_decoder(dct: dict[str, Any]) -> dict[str, Any]:
    """JSON decoder hook that converts decimal strings back to Decimal."""
    for key in ["amount", "tax_amount", "tip_amount", "total_amount"]:
        if key in dct and isinstance(dct[key], str):
            try:
                dct[key] = Decimal(dct[key])
            except Exception:
                pass
    return dct


class ExpenseStoreBackend(ABC):
    """Abstract base class for expense storage backends."""

    @abstractmethod
    async def get(self, expense_id: str) -> Optional[dict[str, Any]]:
        """Get expense by ID."""
        pass

    @abstractmethod
    async def save(self, data: dict[str, Any]) -> None:
        """Save expense data."""
        pass

    @abstractmethod
    async def delete(self, expense_id: str) -> bool:
        """Delete expense."""
        pass

    @abstractmethod
    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List all expenses with pagination."""
        pass

    @abstractmethod
    async def list_by_status(
        self,
        status: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List expenses by status."""
        pass

    @abstractmethod
    async def list_by_employee(
        self,
        employee_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List expenses by employee."""
        pass

    @abstractmethod
    async def list_by_category(
        self,
        category: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List expenses by category."""
        pass

    @abstractmethod
    async def list_pending_sync(self) -> list[dict[str, Any]]:
        """List expenses pending QBO sync."""
        pass

    @abstractmethod
    async def find_duplicates(
        self,
        vendor_name: str,
        amount: Decimal,
        date_tolerance_days: int = 3,
    ) -> list[dict[str, Any]]:
        """Find potential duplicate expenses."""
        pass

    @abstractmethod
    async def get_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, Any]:
        """Get expense statistics."""
        pass

    @abstractmethod
    async def update_status(
        self,
        expense_id: str,
        status: str,
        approved_by: Optional[str] = None,
    ) -> bool:
        """Update expense status."""
        pass

    @abstractmethod
    async def mark_synced(
        self,
        expense_id: str,
        qbo_id: str,
    ) -> bool:
        """Mark expense as synced to QBO."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any resources."""
        pass


class InMemoryExpenseStore(ExpenseStoreBackend):
    """In-memory expense store for testing."""

    def __init__(self) -> None:
        self._data: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    async def get(self, expense_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            return self._data.get(expense_id)

    async def save(self, data: dict[str, Any]) -> None:
        expense_id = data.get("id")
        if not expense_id:
            raise ValueError("id is required")
        with self._lock:
            self._data[expense_id] = data

    async def delete(self, expense_id: str) -> bool:
        with self._lock:
            if expense_id in self._data:
                del self._data[expense_id]
                return True
            return False

    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        with self._lock:
            items = sorted(
                self._data.values(),
                key=lambda x: x.get("created_at", ""),
                reverse=True,
            )
            return items[offset : offset + limit]

    async def list_by_status(
        self,
        status: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            items = [e for e in self._data.values() if e.get("status") == status]
            return sorted(items, key=lambda x: x.get("created_at", ""), reverse=True)[:limit]

    async def list_by_employee(
        self,
        employee_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            items = [e for e in self._data.values() if e.get("employee_id") == employee_id]
            return sorted(items, key=lambda x: x.get("created_at", ""), reverse=True)[:limit]

    async def list_by_category(
        self,
        category: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            items = [e for e in self._data.values() if e.get("category") == category]
            return sorted(items, key=lambda x: x.get("created_at", ""), reverse=True)[:limit]

    async def list_pending_sync(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                e
                for e in self._data.values()
                if not e.get("synced_to_qbo") and e.get("status") == "approved"
            ]

    async def find_duplicates(
        self,
        vendor_name: str,
        amount: Decimal,
        date_tolerance_days: int = 3,
    ) -> list[dict[str, Any]]:
        with self._lock:
            results = []
            now = datetime.now(timezone.utc)

            for expense in self._data.values():
                if expense.get("vendor_name", "").lower() == vendor_name.lower():
                    exp_amount = expense.get("amount")
                    if isinstance(exp_amount, str):
                        exp_amount = Decimal(exp_amount)
                    if exp_amount == amount:
                        exp_date = expense.get("expense_date")
                        if exp_date:
                            if isinstance(exp_date, str):
                                exp_date = datetime.fromisoformat(exp_date.replace("Z", "+00:00"))
                            if abs((now - exp_date).days) <= date_tolerance_days:
                                results.append(expense)
            return results

    async def get_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, Any]:
        with self._lock:
            expenses = list(self._data.values())

            # Filter by date if specified
            if start_date or end_date:
                filtered = []
                for e in expenses:
                    exp_date = e.get("expense_date")
                    if exp_date:
                        if isinstance(exp_date, str):
                            exp_date = datetime.fromisoformat(exp_date.replace("Z", "+00:00"))
                        if start_date and exp_date < start_date:
                            continue
                        if end_date and exp_date > end_date:
                            continue
                        filtered.append(e)
                expenses = filtered

            total = Decimal("0")
            by_category: dict[str, Decimal] = {}
            by_status: dict[str, int] = {}

            for e in expenses:
                amount = e.get("amount", Decimal("0"))
                if isinstance(amount, str):
                    amount = Decimal(amount)
                total += amount

                category = e.get("category", "uncategorized")
                by_category[category] = by_category.get(category, Decimal("0")) + amount

                status = e.get("status", "pending")
                by_status[status] = by_status.get(status, 0) + 1

            return {
                "total_count": len(expenses),
                "total_amount": str(total),
                "by_category": {k: str(v) for k, v in by_category.items()},
                "by_status": by_status,
            }

    async def update_status(
        self,
        expense_id: str,
        status: str,
        approved_by: Optional[str] = None,
    ) -> bool:
        with self._lock:
            if expense_id not in self._data:
                return False
            self._data[expense_id]["status"] = status
            self._data[expense_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
            if approved_by:
                self._data[expense_id]["approved_by"] = approved_by
            return True

    async def mark_synced(
        self,
        expense_id: str,
        qbo_id: str,
    ) -> bool:
        with self._lock:
            if expense_id not in self._data:
                return False
            self._data[expense_id]["synced_to_qbo"] = True
            self._data[expense_id]["qbo_expense_id"] = qbo_id
            self._data[expense_id]["synced_at"] = datetime.now(timezone.utc).isoformat()
            return True

    async def close(self) -> None:
        pass


class SQLiteExpenseStore(ExpenseStoreBackend):
    """SQLite-backed expense store."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        if db_path is None:
            data_dir = os.getenv("ARAGORA_DATA_DIR", "/tmp/aragora")
            db_path = Path(data_dir) / "expenses.db"

        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS expenses (
                        id TEXT PRIMARY KEY,
                        vendor_name TEXT,
                        amount TEXT NOT NULL,
                        category TEXT,
                        status TEXT NOT NULL DEFAULT 'pending',
                        employee_id TEXT,
                        approved_by TEXT,
                        expense_date TEXT,
                        synced_to_qbo INTEGER DEFAULT 0,
                        qbo_expense_id TEXT,
                        synced_at TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        data_json TEXT NOT NULL
                    )
                    """
                )
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_expense_status ON expenses(status)")
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_expense_employee ON expenses(employee_id)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_expense_category ON expenses(category)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_expense_vendor ON expenses(vendor_name)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_expense_date ON expenses(expense_date)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_expense_sync ON expenses(synced_to_qbo)"
                )
                conn.commit()
            finally:
                conn.close()

    async def get(self, expense_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT data_json FROM expenses WHERE id = ?",
                    (expense_id,),
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0], object_hook=decimal_decoder)
                return None
            finally:
                conn.close()

    async def save(self, data: dict[str, Any]) -> None:
        expense_id = data.get("id")
        if not expense_id:
            raise ValueError("id is required")

        now = time.time()
        data_json = json.dumps(data, cls=DecimalEncoder)

        # Convert amount to string for storage
        amount = data.get("amount", "0")
        if isinstance(amount, Decimal):
            amount = str(amount)

        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO expenses
                    (id, vendor_name, amount, category, status, employee_id,
                     approved_by, expense_date, synced_to_qbo, qbo_expense_id,
                     synced_at, created_at, updated_at, data_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        expense_id,
                        data.get("vendor_name"),
                        amount,
                        data.get("category"),
                        data.get("status", "pending"),
                        data.get("employee_id"),
                        data.get("approved_by"),
                        data.get("expense_date"),
                        1 if data.get("synced_to_qbo") else 0,
                        data.get("qbo_expense_id"),
                        data.get("synced_at"),
                        now,
                        now,
                        data_json,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    async def delete(self, expense_id: str) -> bool:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM expenses WHERE id = ?", (expense_id,))
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT data_json FROM expenses
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                )
                return [
                    json.loads(row[0], object_hook=decimal_decoder) for row in cursor.fetchall()
                ]
            finally:
                conn.close()

    async def list_by_status(
        self,
        status: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT data_json FROM expenses
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (status, limit),
                )
                return [
                    json.loads(row[0], object_hook=decimal_decoder) for row in cursor.fetchall()
                ]
            finally:
                conn.close()

    async def list_by_employee(
        self,
        employee_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT data_json FROM expenses
                    WHERE employee_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (employee_id, limit),
                )
                return [
                    json.loads(row[0], object_hook=decimal_decoder) for row in cursor.fetchall()
                ]
            finally:
                conn.close()

    async def list_by_category(
        self,
        category: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT data_json FROM expenses
                    WHERE category = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (category, limit),
                )
                return [
                    json.loads(row[0], object_hook=decimal_decoder) for row in cursor.fetchall()
                ]
            finally:
                conn.close()

    async def list_pending_sync(self) -> list[dict[str, Any]]:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT data_json FROM expenses
                    WHERE synced_to_qbo = 0 AND status = 'approved'
                    ORDER BY created_at ASC
                    """
                )
                return [
                    json.loads(row[0], object_hook=decimal_decoder) for row in cursor.fetchall()
                ]
            finally:
                conn.close()

    async def find_duplicates(
        self,
        vendor_name: str,
        amount: Decimal,
        date_tolerance_days: int = 3,
    ) -> list[dict[str, Any]]:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT data_json FROM expenses
                    WHERE LOWER(vendor_name) = LOWER(?)
                      AND amount = ?
                      AND expense_date >= date('now', ?)
                    """,
                    (vendor_name, str(amount), f"-{date_tolerance_days} days"),
                )
                return [
                    json.loads(row[0], object_hook=decimal_decoder) for row in cursor.fetchall()
                ]
            finally:
                conn.close()

    async def get_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, Any]:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()

                where_clause = "1=1"
                params: list[Any] = []

                if start_date:
                    where_clause += " AND expense_date >= ?"
                    params.append(start_date.isoformat())
                if end_date:
                    where_clause += " AND expense_date <= ?"
                    params.append(end_date.isoformat())

                # Total count and amount
                cursor.execute(
                    f"""
                    SELECT COUNT(*), COALESCE(SUM(CAST(amount AS REAL)), 0)
                    FROM expenses
                    WHERE {where_clause}
                    """,
                    params,
                )
                row = cursor.fetchone()
                total_count = row[0] if row else 0
                total_amount = Decimal(str(row[1])) if row else Decimal("0")

                # By category
                cursor.execute(
                    f"""
                    SELECT category, SUM(CAST(amount AS REAL))
                    FROM expenses
                    WHERE {where_clause}
                    GROUP BY category
                    """,
                    params,
                )
                by_category = {
                    row[0] or "uncategorized": str(Decimal(str(row[1])))
                    for row in cursor.fetchall()
                }

                # By status
                cursor.execute(
                    f"""
                    SELECT status, COUNT(*)
                    FROM expenses
                    WHERE {where_clause}
                    GROUP BY status
                    """,
                    params,
                )
                by_status = {row[0]: row[1] for row in cursor.fetchall()}

                return {
                    "total_count": total_count,
                    "total_amount": str(total_amount),
                    "by_category": by_category,
                    "by_status": by_status,
                }
            finally:
                conn.close()

    async def update_status(
        self,
        expense_id: str,
        status: str,
        approved_by: Optional[str] = None,
    ) -> bool:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()

                # Get current data
                cursor.execute(
                    "SELECT data_json FROM expenses WHERE id = ?",
                    (expense_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return False

                data = json.loads(row[0], object_hook=decimal_decoder)
                data["status"] = status
                data["updated_at"] = datetime.now(timezone.utc).isoformat()
                if approved_by:
                    data["approved_by"] = approved_by

                cursor.execute(
                    """
                    UPDATE expenses
                    SET status = ?, approved_by = ?, updated_at = ?, data_json = ?
                    WHERE id = ?
                    """,
                    (
                        status,
                        approved_by,
                        time.time(),
                        json.dumps(data, cls=DecimalEncoder),
                        expense_id,
                    ),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    async def mark_synced(
        self,
        expense_id: str,
        qbo_id: str,
    ) -> bool:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()

                # Get current data
                cursor.execute(
                    "SELECT data_json FROM expenses WHERE id = ?",
                    (expense_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return False

                synced_at = datetime.now(timezone.utc).isoformat()
                data = json.loads(row[0], object_hook=decimal_decoder)
                data["synced_to_qbo"] = True
                data["qbo_expense_id"] = qbo_id
                data["synced_at"] = synced_at

                cursor.execute(
                    """
                    UPDATE expenses
                    SET synced_to_qbo = 1, qbo_expense_id = ?, synced_at = ?,
                        updated_at = ?, data_json = ?
                    WHERE id = ?
                    """,
                    (
                        qbo_id,
                        synced_at,
                        time.time(),
                        json.dumps(data, cls=DecimalEncoder),
                        expense_id,
                    ),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    async def close(self) -> None:
        pass


class PostgresExpenseStore(ExpenseStoreBackend):
    """PostgreSQL-backed expense store for production."""

    SCHEMA_NAME = "expenses"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS expenses (
            id TEXT PRIMARY KEY,
            vendor_name TEXT,
            amount NUMERIC NOT NULL,
            category TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            employee_id TEXT,
            approved_by TEXT,
            expense_date TIMESTAMPTZ,
            synced_to_qbo BOOLEAN DEFAULT FALSE,
            qbo_expense_id TEXT,
            synced_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            data_json JSONB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_expense_status ON expenses(status);
        CREATE INDEX IF NOT EXISTS idx_expense_employee ON expenses(employee_id);
        CREATE INDEX IF NOT EXISTS idx_expense_category ON expenses(category);
        CREATE INDEX IF NOT EXISTS idx_expense_vendor ON expenses(vendor_name);
        CREATE INDEX IF NOT EXISTS idx_expense_date ON expenses(expense_date);
        CREATE INDEX IF NOT EXISTS idx_expense_sync ON expenses(synced_to_qbo);
    """

    def __init__(self, pool: "Pool"):
        self._pool = pool
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        async with self._pool.acquire() as conn:
            await conn.execute(self.INITIAL_SCHEMA)
        self._initialized = True

    async def get(self, expense_id: str) -> Optional[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data_json FROM expenses WHERE id = $1",
                expense_id,
            )
            if row:
                data = row["data_json"]
                return json.loads(data) if isinstance(data, str) else data
            return None

    async def save(self, data: dict[str, Any]) -> None:
        expense_id = data.get("id")
        if not expense_id:
            raise ValueError("id is required")

        amount = data.get("amount", "0")
        if isinstance(amount, Decimal):
            amount = float(amount)
        elif isinstance(amount, str):
            amount = float(Decimal(amount))

        expense_date = data.get("expense_date")
        if isinstance(expense_date, str):
            expense_date = datetime.fromisoformat(expense_date.replace("Z", "+00:00"))

        synced_at = data.get("synced_at")
        if isinstance(synced_at, str):
            synced_at = datetime.fromisoformat(synced_at.replace("Z", "+00:00"))

        data_json = json.dumps(data, cls=DecimalEncoder)

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO expenses
                (id, vendor_name, amount, category, status, employee_id,
                 approved_by, expense_date, synced_to_qbo, qbo_expense_id,
                 synced_at, data_json)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (id) DO UPDATE SET
                    vendor_name = EXCLUDED.vendor_name,
                    amount = EXCLUDED.amount,
                    category = EXCLUDED.category,
                    status = EXCLUDED.status,
                    employee_id = EXCLUDED.employee_id,
                    approved_by = EXCLUDED.approved_by,
                    expense_date = EXCLUDED.expense_date,
                    synced_to_qbo = EXCLUDED.synced_to_qbo,
                    qbo_expense_id = EXCLUDED.qbo_expense_id,
                    synced_at = EXCLUDED.synced_at,
                    updated_at = NOW(),
                    data_json = EXCLUDED.data_json
                """,
                expense_id,
                data.get("vendor_name"),
                amount,
                data.get("category"),
                data.get("status", "pending"),
                data.get("employee_id"),
                data.get("approved_by"),
                expense_date,
                data.get("synced_to_qbo", False),
                data.get("qbo_expense_id"),
                synced_at,
                data_json,
            )

    async def delete(self, expense_id: str) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM expenses WHERE id = $1",
                expense_id,
            )
            return result != "DELETE 0"

    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM expenses
                ORDER BY created_at DESC
                LIMIT $1 OFFSET $2
                """,
                limit,
                offset,
            )
            results = []
            for row in rows:
                data = row["data_json"]
                results.append(json.loads(data) if isinstance(data, str) else data)
            return results

    async def list_by_status(
        self,
        status: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM expenses
                WHERE status = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                status,
                limit,
            )
            results = []
            for row in rows:
                data = row["data_json"]
                results.append(json.loads(data) if isinstance(data, str) else data)
            return results

    async def list_by_employee(
        self,
        employee_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM expenses
                WHERE employee_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                employee_id,
                limit,
            )
            results = []
            for row in rows:
                data = row["data_json"]
                results.append(json.loads(data) if isinstance(data, str) else data)
            return results

    async def list_by_category(
        self,
        category: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM expenses
                WHERE category = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                category,
                limit,
            )
            results = []
            for row in rows:
                data = row["data_json"]
                results.append(json.loads(data) if isinstance(data, str) else data)
            return results

    async def list_pending_sync(self) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM expenses
                WHERE synced_to_qbo = FALSE AND status = 'approved'
                ORDER BY created_at ASC
                """
            )
            results = []
            for row in rows:
                data = row["data_json"]
                results.append(json.loads(data) if isinstance(data, str) else data)
            return results

    async def find_duplicates(
        self,
        vendor_name: str,
        amount: Decimal,
        date_tolerance_days: int = 3,
    ) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM expenses
                WHERE LOWER(vendor_name) = LOWER($1)
                  AND amount = $2
                  AND expense_date >= NOW() - INTERVAL '$3 days'
                """,
                vendor_name,
                float(amount),
                date_tolerance_days,
            )
            results = []
            for row in rows:
                data = row["data_json"]
                results.append(json.loads(data) if isinstance(data, str) else data)
            return results

    async def get_statistics(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> dict[str, Any]:
        async with self._pool.acquire() as conn:
            where_clause = "1=1"
            params: list[Any] = []
            param_idx = 1

            if start_date:
                where_clause += f" AND expense_date >= ${param_idx}"
                params.append(start_date)
                param_idx += 1
            if end_date:
                where_clause += f" AND expense_date <= ${param_idx}"
                params.append(end_date)
                param_idx += 1

            # Total count and amount
            row = await conn.fetchrow(
                f"""
                SELECT COUNT(*), COALESCE(SUM(amount), 0)
                FROM expenses
                WHERE {where_clause}
                """,
                *params,
            )
            total_count = row[0] if row else 0
            total_amount = Decimal(str(row[1])) if row else Decimal("0")

            # By category
            cat_rows = await conn.fetch(
                f"""
                SELECT category, SUM(amount)
                FROM expenses
                WHERE {where_clause}
                GROUP BY category
                """,
                *params,
            )
            by_category = {row[0] or "uncategorized": str(Decimal(str(row[1]))) for row in cat_rows}

            # By status
            status_rows = await conn.fetch(
                f"""
                SELECT status, COUNT(*)
                FROM expenses
                WHERE {where_clause}
                GROUP BY status
                """,
                *params,
            )
            by_status = {row[0]: row[1] for row in status_rows}

            return {
                "total_count": total_count,
                "total_amount": str(total_amount),
                "by_category": by_category,
                "by_status": by_status,
            }

    async def update_status(
        self,
        expense_id: str,
        status: str,
        approved_by: Optional[str] = None,
    ) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data_json FROM expenses WHERE id = $1",
                expense_id,
            )
            if not row:
                return False

            raw_data = row["data_json"]
            data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
            data["status"] = status
            data["updated_at"] = datetime.now(timezone.utc).isoformat()
            if approved_by:
                data["approved_by"] = approved_by

            result = await conn.execute(
                """
                UPDATE expenses
                SET status = $1, approved_by = $2, updated_at = NOW(), data_json = $3
                WHERE id = $4
                """,
                status,
                approved_by,
                json.dumps(data, cls=DecimalEncoder),
                expense_id,
            )
            return result != "UPDATE 0"

    async def mark_synced(
        self,
        expense_id: str,
        qbo_id: str,
    ) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data_json FROM expenses WHERE id = $1",
                expense_id,
            )
            if not row:
                return False

            synced_at = datetime.now(timezone.utc)
            raw_data = row["data_json"]
            data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
            data["synced_to_qbo"] = True
            data["qbo_expense_id"] = qbo_id
            data["synced_at"] = synced_at.isoformat()

            result = await conn.execute(
                """
                UPDATE expenses
                SET synced_to_qbo = TRUE, qbo_expense_id = $1, synced_at = $2,
                    updated_at = NOW(), data_json = $3
                WHERE id = $4
                """,
                qbo_id,
                synced_at,
                json.dumps(data, cls=DecimalEncoder),
                expense_id,
            )
            return result != "UPDATE 0"

    async def close(self) -> None:
        pass


def get_expense_store() -> ExpenseStoreBackend:
    """
    Get the global expense store instance.

    Backend is selected based on ARAGORA_EXPENSE_STORE_BACKEND env var:
    - "memory": InMemoryExpenseStore (for testing)
    - "sqlite": SQLiteExpenseStore (default, single-instance)
    - "postgres": PostgresExpenseStore (multi-instance)
    """
    global _expense_store

    with _store_lock:
        if _expense_store is not None:
            return _expense_store

        backend = os.getenv("ARAGORA_EXPENSE_STORE_BACKEND")
        if not backend:
            backend = os.getenv("ARAGORA_DB_BACKEND", "sqlite")
        backend = backend.lower()

        if backend == "memory":
            _expense_store = InMemoryExpenseStore()
            logger.info("Using in-memory expense store")
        elif backend in ("postgres", "postgresql"):
            logger.info("Using PostgreSQL expense store")
            try:
                from aragora.storage.postgres_store import get_postgres_pool

                pool = asyncio.get_event_loop().run_until_complete(get_postgres_pool())
                store = PostgresExpenseStore(pool)
                asyncio.get_event_loop().run_until_complete(store.initialize())
                _expense_store = store
            except Exception as e:
                logger.warning(f"PostgreSQL not available, falling back to SQLite: {e}")
                _expense_store = SQLiteExpenseStore()
        else:
            _expense_store = SQLiteExpenseStore()
            logger.info("Using SQLite expense store")

        return _expense_store


def set_expense_store(store: ExpenseStoreBackend) -> None:
    """Set a custom expense store instance."""
    global _expense_store
    with _store_lock:
        _expense_store = store


def reset_expense_store() -> None:
    """Reset the global expense store (for testing)."""
    global _expense_store
    with _store_lock:
        _expense_store = None


__all__ = [
    "ExpenseStoreBackend",
    "InMemoryExpenseStore",
    "SQLiteExpenseStore",
    "PostgresExpenseStore",
    "get_expense_store",
    "set_expense_store",
    "reset_expense_store",
]
