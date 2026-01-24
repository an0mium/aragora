"""
AR Invoice Storage Backends (Accounts Receivable).

Persistent storage for AR invoices, customer data, and collection tracking.

Backends:
- InMemoryARInvoiceStore: For testing
- SQLiteARInvoiceStore: For single-instance deployments
- PostgresARInvoiceStore: For multi-instance production

Usage:
    from aragora.storage.ar_invoice_store import get_ar_invoice_store

    store = get_ar_invoice_store()
    await store.save(invoice_data)
    invoice = await store.get("ar_inv_123")
"""

from __future__ import annotations

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
_ar_invoice_store: Optional["ARInvoiceStoreBackend"] = None
_store_lock = threading.RLock()


class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Decimal):
            return str(obj)
        return super().default(obj)


def decimal_decoder(dct: dict[str, Any]) -> dict[str, Any]:
    """JSON decoder hook that converts decimal strings back to Decimal."""
    for key in ["subtotal", "tax_amount", "total_amount", "amount_paid", "balance_due"]:
        if key in dct and isinstance(dct[key], str):
            try:
                dct[key] = Decimal(dct[key])
            except Exception:
                pass
    return dct


class ARInvoiceStoreBackend(ABC):
    """Abstract base class for AR invoice storage backends."""

    @abstractmethod
    async def get(self, invoice_id: str) -> Optional[dict[str, Any]]:
        """Get invoice by ID."""
        pass

    @abstractmethod
    async def save(self, data: dict[str, Any]) -> None:
        """Save invoice data."""
        pass

    @abstractmethod
    async def delete(self, invoice_id: str) -> bool:
        """Delete invoice."""
        pass

    @abstractmethod
    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List all invoices with pagination."""
        pass

    @abstractmethod
    async def list_by_status(
        self,
        status: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List invoices by status."""
        pass

    @abstractmethod
    async def list_by_customer(
        self,
        customer_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List invoices by customer."""
        pass

    @abstractmethod
    async def list_overdue(self) -> list[dict[str, Any]]:
        """List overdue invoices."""
        pass

    @abstractmethod
    async def get_aging_buckets(self) -> dict[str, list[dict[str, Any]]]:
        """Get invoices grouped by aging buckets (current, 1-30, 31-60, 61-90, 90+)."""
        pass

    @abstractmethod
    async def get_customer_balance(self, customer_id: str) -> Decimal:
        """Get total outstanding balance for a customer."""
        pass

    @abstractmethod
    async def update_status(
        self,
        invoice_id: str,
        status: str,
    ) -> bool:
        """Update invoice status."""
        pass

    @abstractmethod
    async def record_payment(
        self,
        invoice_id: str,
        amount: Decimal,
        payment_date: datetime,
        payment_method: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> bool:
        """Record a payment against an invoice."""
        pass

    @abstractmethod
    async def record_reminder_sent(
        self,
        invoice_id: str,
        reminder_level: int,
    ) -> bool:
        """Record that a reminder was sent."""
        pass

    # Customer management
    @abstractmethod
    async def get_customer(self, customer_id: str) -> Optional[dict[str, Any]]:
        """Get customer by ID."""
        pass

    @abstractmethod
    async def save_customer(self, data: dict[str, Any]) -> None:
        """Save customer data."""
        pass

    @abstractmethod
    async def list_customers(self, limit: int = 100) -> list[dict[str, Any]]:
        """List all customers."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any resources."""
        pass


class InMemoryARInvoiceStore(ARInvoiceStoreBackend):
    """In-memory AR invoice store for testing."""

    def __init__(self) -> None:
        self._invoices: dict[str, dict[str, Any]] = {}
        self._customers: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    async def get(self, invoice_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            return self._invoices.get(invoice_id)

    async def save(self, data: dict[str, Any]) -> None:
        invoice_id = data.get("id")
        if not invoice_id:
            raise ValueError("id is required")
        with self._lock:
            self._invoices[invoice_id] = data

    async def delete(self, invoice_id: str) -> bool:
        with self._lock:
            if invoice_id in self._invoices:
                del self._invoices[invoice_id]
                return True
            return False

    async def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        with self._lock:
            items = sorted(
                self._invoices.values(),
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
            items = [i for i in self._invoices.values() if i.get("status") == status]
            return sorted(items, key=lambda x: x.get("created_at", ""), reverse=True)[:limit]

    async def list_by_customer(
        self,
        customer_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            items = [i for i in self._invoices.values() if i.get("customer_id") == customer_id]
            return sorted(items, key=lambda x: x.get("created_at", ""), reverse=True)[:limit]

    async def list_overdue(self) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc)
        with self._lock:
            results = []
            for inv in self._invoices.values():
                if inv.get("status") in ("paid", "void"):
                    continue
                due_date = inv.get("due_date")
                if due_date:
                    if isinstance(due_date, str):
                        due_date = datetime.fromisoformat(due_date.replace("Z", "+00:00"))
                    if due_date < now:
                        results.append(inv)
            return sorted(results, key=lambda x: x.get("due_date", ""))

    async def get_aging_buckets(self) -> dict[str, list[dict[str, Any]]]:
        now = datetime.now(timezone.utc)
        buckets: dict[str, list[dict[str, Any]]] = {
            "current": [],
            "1_30": [],
            "31_60": [],
            "61_90": [],
            "90_plus": [],
        }

        with self._lock:
            for inv in self._invoices.values():
                if inv.get("status") in ("paid", "void"):
                    continue

                due_date = inv.get("due_date")
                if not due_date:
                    buckets["current"].append(inv)
                    continue

                if isinstance(due_date, str):
                    due_date = datetime.fromisoformat(due_date.replace("Z", "+00:00"))

                days_overdue = (now - due_date).days

                if days_overdue <= 0:
                    buckets["current"].append(inv)
                elif days_overdue <= 30:
                    buckets["1_30"].append(inv)
                elif days_overdue <= 60:
                    buckets["31_60"].append(inv)
                elif days_overdue <= 90:
                    buckets["61_90"].append(inv)
                else:
                    buckets["90_plus"].append(inv)

        return buckets

    async def get_customer_balance(self, customer_id: str) -> Decimal:
        with self._lock:
            total = Decimal("0")
            for inv in self._invoices.values():
                if inv.get("customer_id") != customer_id:
                    continue
                if inv.get("status") in ("paid", "void"):
                    continue

                balance = inv.get("balance_due", inv.get("total_amount", Decimal("0")))
                if isinstance(balance, str):
                    balance = Decimal(balance)
                total += balance
            return total

    async def update_status(
        self,
        invoice_id: str,
        status: str,
    ) -> bool:
        with self._lock:
            if invoice_id not in self._invoices:
                return False
            self._invoices[invoice_id]["status"] = status
            self._invoices[invoice_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
            return True

    async def record_payment(
        self,
        invoice_id: str,
        amount: Decimal,
        payment_date: datetime,
        payment_method: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> bool:
        with self._lock:
            if invoice_id not in self._invoices:
                return False

            inv = self._invoices[invoice_id]
            payments = inv.get("payments", [])
            payments.append(
                {
                    "amount": str(amount),
                    "payment_date": payment_date.isoformat(),
                    "payment_method": payment_method,
                    "reference": reference,
                }
            )
            inv["payments"] = payments

            total_paid = sum(Decimal(p["amount"]) for p in payments)
            inv["amount_paid"] = str(total_paid)

            total = inv.get("total_amount", Decimal("0"))
            if isinstance(total, str):
                total = Decimal(total)
            inv["balance_due"] = str(total - total_paid)

            if total_paid >= total:
                inv["status"] = "paid"

            inv["updated_at"] = datetime.now(timezone.utc).isoformat()
            return True

    async def record_reminder_sent(
        self,
        invoice_id: str,
        reminder_level: int,
    ) -> bool:
        with self._lock:
            if invoice_id not in self._invoices:
                return False

            inv = self._invoices[invoice_id]
            reminders = inv.get("reminders_sent", [])
            reminders.append(
                {
                    "level": reminder_level,
                    "sent_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            inv["reminders_sent"] = reminders
            inv["last_reminder_level"] = reminder_level
            inv["last_reminder_at"] = datetime.now(timezone.utc).isoformat()
            return True

    async def get_customer(self, customer_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            return self._customers.get(customer_id)

    async def save_customer(self, data: dict[str, Any]) -> None:
        customer_id = data.get("customer_id")
        if not customer_id:
            raise ValueError("customer_id is required")
        with self._lock:
            self._customers[customer_id] = data

    async def list_customers(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._customers.values())[:limit]

    async def close(self) -> None:
        pass


class SQLiteARInvoiceStore(ARInvoiceStoreBackend):
    """SQLite-backed AR invoice store."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        if db_path is None:
            data_dir = os.getenv("ARAGORA_DATA_DIR", "/tmp/aragora")
            db_path = Path(data_dir) / "ar_invoices.db"

        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()

                # AR Invoices table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ar_invoices (
                        id TEXT PRIMARY KEY,
                        customer_id TEXT NOT NULL,
                        customer_name TEXT,
                        invoice_number TEXT,
                        total_amount TEXT NOT NULL,
                        balance_due TEXT,
                        status TEXT NOT NULL DEFAULT 'draft',
                        due_date TEXT,
                        sent_at TEXT,
                        last_reminder_level INTEGER DEFAULT 0,
                        last_reminder_at TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL,
                        data_json TEXT NOT NULL
                    )
                    """
                )
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_ar_status ON ar_invoices(status)")
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_ar_customer ON ar_invoices(customer_id)"
                )
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_ar_due ON ar_invoices(due_date)")

                # Customers table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ar_customers (
                        customer_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        email TEXT,
                        payment_terms TEXT DEFAULT 'Net 30',
                        created_at REAL NOT NULL,
                        data_json TEXT NOT NULL
                    )
                    """
                )

                conn.commit()
            finally:
                conn.close()

    async def get(self, invoice_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT data_json FROM ar_invoices WHERE id = ?",
                    (invoice_id,),
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0], object_hook=decimal_decoder)
                return None
            finally:
                conn.close()

    async def save(self, data: dict[str, Any]) -> None:
        invoice_id = data.get("id")
        if not invoice_id:
            raise ValueError("id is required")

        now = time.time()
        data_json = json.dumps(data, cls=DecimalEncoder)

        total_amount = data.get("total_amount", "0")
        if isinstance(total_amount, Decimal):
            total_amount = str(total_amount)

        balance_due = data.get("balance_due", total_amount)
        if isinstance(balance_due, Decimal):
            balance_due = str(balance_due)

        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO ar_invoices
                    (id, customer_id, customer_name, invoice_number, total_amount,
                     balance_due, status, due_date, sent_at, last_reminder_level,
                     last_reminder_at, created_at, updated_at, data_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        invoice_id,
                        data.get("customer_id"),
                        data.get("customer_name"),
                        data.get("invoice_number"),
                        total_amount,
                        balance_due,
                        data.get("status", "draft"),
                        data.get("due_date"),
                        data.get("sent_at"),
                        data.get("last_reminder_level", 0),
                        data.get("last_reminder_at"),
                        now,
                        now,
                        data_json,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    async def delete(self, invoice_id: str) -> bool:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM ar_invoices WHERE id = ?", (invoice_id,))
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
                    SELECT data_json FROM ar_invoices
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
                    SELECT data_json FROM ar_invoices
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

    async def list_by_customer(
        self,
        customer_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT data_json FROM ar_invoices
                    WHERE customer_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (customer_id, limit),
                )
                return [
                    json.loads(row[0], object_hook=decimal_decoder) for row in cursor.fetchall()
                ]
            finally:
                conn.close()

    async def list_overdue(self) -> list[dict[str, Any]]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT data_json FROM ar_invoices
                    WHERE status NOT IN ('paid', 'void')
                      AND due_date IS NOT NULL
                      AND due_date < ?
                    ORDER BY due_date ASC
                    """,
                    (now,),
                )
                return [
                    json.loads(row[0], object_hook=decimal_decoder) for row in cursor.fetchall()
                ]
            finally:
                conn.close()

    async def get_aging_buckets(self) -> dict[str, list[dict[str, Any]]]:
        now = datetime.now(timezone.utc)
        buckets: dict[str, list[dict[str, Any]]] = {
            "current": [],
            "1_30": [],
            "31_60": [],
            "61_90": [],
            "90_plus": [],
        }

        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT data_json, due_date FROM ar_invoices
                    WHERE status NOT IN ('paid', 'void')
                    """
                )

                for row in cursor.fetchall():
                    inv = json.loads(row[0], object_hook=decimal_decoder)
                    due_date_str = row[1]

                    if not due_date_str:
                        buckets["current"].append(inv)
                        continue

                    due_date = datetime.fromisoformat(due_date_str.replace("Z", "+00:00"))
                    days_overdue = (now - due_date).days

                    if days_overdue <= 0:
                        buckets["current"].append(inv)
                    elif days_overdue <= 30:
                        buckets["1_30"].append(inv)
                    elif days_overdue <= 60:
                        buckets["31_60"].append(inv)
                    elif days_overdue <= 90:
                        buckets["61_90"].append(inv)
                    else:
                        buckets["90_plus"].append(inv)

            finally:
                conn.close()

        return buckets

    async def get_customer_balance(self, customer_id: str) -> Decimal:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT COALESCE(SUM(CAST(balance_due AS REAL)), 0)
                    FROM ar_invoices
                    WHERE customer_id = ?
                      AND status NOT IN ('paid', 'void')
                    """,
                    (customer_id,),
                )
                row = cursor.fetchone()
                return Decimal(str(row[0])) if row else Decimal("0")
            finally:
                conn.close()

    async def update_status(
        self,
        invoice_id: str,
        status: str,
    ) -> bool:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT data_json FROM ar_invoices WHERE id = ?",
                    (invoice_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return False

                data = json.loads(row[0], object_hook=decimal_decoder)
                data["status"] = status
                data["updated_at"] = datetime.now(timezone.utc).isoformat()

                cursor.execute(
                    """
                    UPDATE ar_invoices
                    SET status = ?, updated_at = ?, data_json = ?
                    WHERE id = ?
                    """,
                    (
                        status,
                        time.time(),
                        json.dumps(data, cls=DecimalEncoder),
                        invoice_id,
                    ),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    async def record_payment(
        self,
        invoice_id: str,
        amount: Decimal,
        payment_date: datetime,
        payment_method: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> bool:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT data_json FROM ar_invoices WHERE id = ?",
                    (invoice_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return False

                data = json.loads(row[0], object_hook=decimal_decoder)

                payments = data.get("payments", [])
                payments.append(
                    {
                        "amount": str(amount),
                        "payment_date": payment_date.isoformat(),
                        "payment_method": payment_method,
                        "reference": reference,
                    }
                )
                data["payments"] = payments

                total_paid = sum(Decimal(p["amount"]) for p in payments)
                data["amount_paid"] = str(total_paid)

                total = data.get("total_amount", Decimal("0"))
                if isinstance(total, str):
                    total = Decimal(total)
                balance = total - total_paid
                data["balance_due"] = str(balance)

                if total_paid >= total:
                    data["status"] = "paid"

                data["updated_at"] = datetime.now(timezone.utc).isoformat()

                cursor.execute(
                    """
                    UPDATE ar_invoices
                    SET status = ?, balance_due = ?, updated_at = ?, data_json = ?
                    WHERE id = ?
                    """,
                    (
                        data["status"],
                        str(balance),
                        time.time(),
                        json.dumps(data, cls=DecimalEncoder),
                        invoice_id,
                    ),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    async def record_reminder_sent(
        self,
        invoice_id: str,
        reminder_level: int,
    ) -> bool:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()

                cursor.execute(
                    "SELECT data_json FROM ar_invoices WHERE id = ?",
                    (invoice_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return False

                now_iso = datetime.now(timezone.utc).isoformat()
                data = json.loads(row[0], object_hook=decimal_decoder)

                reminders = data.get("reminders_sent", [])
                reminders.append({"level": reminder_level, "sent_at": now_iso})
                data["reminders_sent"] = reminders
                data["last_reminder_level"] = reminder_level
                data["last_reminder_at"] = now_iso

                cursor.execute(
                    """
                    UPDATE ar_invoices
                    SET last_reminder_level = ?, last_reminder_at = ?,
                        updated_at = ?, data_json = ?
                    WHERE id = ?
                    """,
                    (
                        reminder_level,
                        now_iso,
                        time.time(),
                        json.dumps(data, cls=DecimalEncoder),
                        invoice_id,
                    ),
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    async def get_customer(self, customer_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT data_json FROM ar_customers WHERE customer_id = ?",
                    (customer_id,),
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
                return None
            finally:
                conn.close()

    async def save_customer(self, data: dict[str, Any]) -> None:
        customer_id = data.get("customer_id")
        if not customer_id:
            raise ValueError("customer_id is required")

        now = time.time()
        data_json = json.dumps(data)

        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO ar_customers
                    (customer_id, name, email, payment_terms, created_at, data_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        customer_id,
                        data.get("name"),
                        data.get("email"),
                        data.get("payment_terms", "Net 30"),
                        now,
                        data_json,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    async def list_customers(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            conn = sqlite3.connect(str(self._db_path))
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT data_json FROM ar_customers
                    ORDER BY name ASC
                    LIMIT ?
                    """,
                    (limit,),
                )
                return [json.loads(row[0]) for row in cursor.fetchall()]
            finally:
                conn.close()

    async def close(self) -> None:
        pass


class PostgresARInvoiceStore(ARInvoiceStoreBackend):
    """PostgreSQL-backed AR invoice store for production."""

    SCHEMA_NAME = "ar_invoices"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS ar_invoices (
            id TEXT PRIMARY KEY,
            customer_id TEXT NOT NULL,
            customer_name TEXT,
            invoice_number TEXT,
            total_amount NUMERIC NOT NULL,
            balance_due NUMERIC,
            status TEXT NOT NULL DEFAULT 'draft',
            due_date TIMESTAMPTZ,
            sent_at TIMESTAMPTZ,
            last_reminder_level INTEGER DEFAULT 0,
            last_reminder_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            data_json JSONB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ar_status ON ar_invoices(status);
        CREATE INDEX IF NOT EXISTS idx_ar_customer ON ar_invoices(customer_id);
        CREATE INDEX IF NOT EXISTS idx_ar_due ON ar_invoices(due_date);

        CREATE TABLE IF NOT EXISTS ar_customers (
            customer_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            payment_terms TEXT DEFAULT 'Net 30',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            data_json JSONB NOT NULL
        );
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

    async def get(self, invoice_id: str) -> Optional[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data_json FROM ar_invoices WHERE id = $1",
                invoice_id,
            )
            if row:
                data = row["data_json"]
                return json.loads(data) if isinstance(data, str) else data
            return None

    async def save(self, data: dict[str, Any]) -> None:
        invoice_id = data.get("id")
        if not invoice_id:
            raise ValueError("id is required")

        total_amount = data.get("total_amount", "0")
        if isinstance(total_amount, Decimal):
            total_amount = float(total_amount)
        elif isinstance(total_amount, str):
            total_amount = float(Decimal(total_amount))

        balance_due = data.get("balance_due", total_amount)
        if isinstance(balance_due, Decimal):
            balance_due = float(balance_due)
        elif isinstance(balance_due, str):
            balance_due = float(Decimal(balance_due))

        def parse_date(val: Any) -> Optional[datetime]:
            if val is None:
                return None
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            return None

        data_json = json.dumps(data, cls=DecimalEncoder)

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ar_invoices
                (id, customer_id, customer_name, invoice_number, total_amount,
                 balance_due, status, due_date, sent_at, last_reminder_level,
                 last_reminder_at, data_json)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (id) DO UPDATE SET
                    customer_id = EXCLUDED.customer_id,
                    customer_name = EXCLUDED.customer_name,
                    invoice_number = EXCLUDED.invoice_number,
                    total_amount = EXCLUDED.total_amount,
                    balance_due = EXCLUDED.balance_due,
                    status = EXCLUDED.status,
                    due_date = EXCLUDED.due_date,
                    sent_at = EXCLUDED.sent_at,
                    last_reminder_level = EXCLUDED.last_reminder_level,
                    last_reminder_at = EXCLUDED.last_reminder_at,
                    updated_at = NOW(),
                    data_json = EXCLUDED.data_json
                """,
                invoice_id,
                data.get("customer_id"),
                data.get("customer_name"),
                data.get("invoice_number"),
                total_amount,
                balance_due,
                data.get("status", "draft"),
                parse_date(data.get("due_date")),
                parse_date(data.get("sent_at")),
                data.get("last_reminder_level", 0),
                parse_date(data.get("last_reminder_at")),
                data_json,
            )

    async def delete(self, invoice_id: str) -> bool:
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM ar_invoices WHERE id = $1",
                invoice_id,
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
                SELECT data_json FROM ar_invoices
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
                SELECT data_json FROM ar_invoices
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

    async def list_by_customer(
        self,
        customer_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM ar_invoices
                WHERE customer_id = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                customer_id,
                limit,
            )
            results = []
            for row in rows:
                data = row["data_json"]
                results.append(json.loads(data) if isinstance(data, str) else data)
            return results

    async def list_overdue(self) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM ar_invoices
                WHERE status NOT IN ('paid', 'void')
                  AND due_date IS NOT NULL
                  AND due_date < NOW()
                ORDER BY due_date ASC
                """
            )
            results = []
            for row in rows:
                data = row["data_json"]
                results.append(json.loads(data) if isinstance(data, str) else data)
            return results

    async def get_aging_buckets(self) -> dict[str, list[dict[str, Any]]]:
        buckets: dict[str, list[dict[str, Any]]] = {
            "current": [],
            "1_30": [],
            "31_60": [],
            "61_90": [],
            "90_plus": [],
        }

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json,
                       EXTRACT(DAY FROM NOW() - due_date) as days_overdue
                FROM ar_invoices
                WHERE status NOT IN ('paid', 'void')
                """
            )

            for row in rows:
                data = row["data_json"]
                inv = json.loads(data) if isinstance(data, str) else data
                days_overdue = row["days_overdue"] or 0

                if days_overdue <= 0:
                    buckets["current"].append(inv)
                elif days_overdue <= 30:
                    buckets["1_30"].append(inv)
                elif days_overdue <= 60:
                    buckets["31_60"].append(inv)
                elif days_overdue <= 90:
                    buckets["61_90"].append(inv)
                else:
                    buckets["90_plus"].append(inv)

        return buckets

    async def get_customer_balance(self, customer_id: str) -> Decimal:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT COALESCE(SUM(balance_due), 0)
                FROM ar_invoices
                WHERE customer_id = $1
                  AND status NOT IN ('paid', 'void')
                """,
                customer_id,
            )
            return Decimal(str(row[0])) if row else Decimal("0")

    async def update_status(
        self,
        invoice_id: str,
        status: str,
    ) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data_json FROM ar_invoices WHERE id = $1",
                invoice_id,
            )
            if not row:
                return False

            raw_data = row["data_json"]
            data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
            data["status"] = status
            data["updated_at"] = datetime.now(timezone.utc).isoformat()

            result = await conn.execute(
                """
                UPDATE ar_invoices
                SET status = $1, updated_at = NOW(), data_json = $2
                WHERE id = $3
                """,
                status,
                json.dumps(data, cls=DecimalEncoder),
                invoice_id,
            )
            return result != "UPDATE 0"

    async def record_payment(
        self,
        invoice_id: str,
        amount: Decimal,
        payment_date: datetime,
        payment_method: Optional[str] = None,
        reference: Optional[str] = None,
    ) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data_json FROM ar_invoices WHERE id = $1",
                invoice_id,
            )
            if not row:
                return False

            raw_data = row["data_json"]
            data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data

            payments = data.get("payments", [])
            payments.append(
                {
                    "amount": str(amount),
                    "payment_date": payment_date.isoformat(),
                    "payment_method": payment_method,
                    "reference": reference,
                }
            )
            data["payments"] = payments

            total_paid = sum(Decimal(p["amount"]) for p in payments)
            data["amount_paid"] = str(total_paid)

            total = data.get("total_amount", Decimal("0"))
            if isinstance(total, str):
                total = Decimal(total)
            balance = total - total_paid
            data["balance_due"] = str(balance)

            if total_paid >= total:
                data["status"] = "paid"

            data["updated_at"] = datetime.now(timezone.utc).isoformat()

            result = await conn.execute(
                """
                UPDATE ar_invoices
                SET status = $1, balance_due = $2, updated_at = NOW(), data_json = $3
                WHERE id = $4
                """,
                data["status"],
                float(balance),
                json.dumps(data, cls=DecimalEncoder),
                invoice_id,
            )
            return result != "UPDATE 0"

    async def record_reminder_sent(
        self,
        invoice_id: str,
        reminder_level: int,
    ) -> bool:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data_json FROM ar_invoices WHERE id = $1",
                invoice_id,
            )
            if not row:
                return False

            now = datetime.now(timezone.utc)
            raw_data = row["data_json"]
            data = json.loads(raw_data) if isinstance(raw_data, str) else raw_data

            reminders = data.get("reminders_sent", [])
            reminders.append({"level": reminder_level, "sent_at": now.isoformat()})
            data["reminders_sent"] = reminders
            data["last_reminder_level"] = reminder_level
            data["last_reminder_at"] = now.isoformat()

            result = await conn.execute(
                """
                UPDATE ar_invoices
                SET last_reminder_level = $1, last_reminder_at = $2,
                    updated_at = NOW(), data_json = $3
                WHERE id = $4
                """,
                reminder_level,
                now,
                json.dumps(data, cls=DecimalEncoder),
                invoice_id,
            )
            return result != "UPDATE 0"

    async def get_customer(self, customer_id: str) -> Optional[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT data_json FROM ar_customers WHERE customer_id = $1",
                customer_id,
            )
            if row:
                data = row["data_json"]
                return json.loads(data) if isinstance(data, str) else data
            return None

    async def save_customer(self, data: dict[str, Any]) -> None:
        customer_id = data.get("customer_id")
        if not customer_id:
            raise ValueError("customer_id is required")

        data_json = json.dumps(data)

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO ar_customers
                (customer_id, name, email, payment_terms, data_json)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (customer_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    email = EXCLUDED.email,
                    payment_terms = EXCLUDED.payment_terms,
                    data_json = EXCLUDED.data_json
                """,
                customer_id,
                data.get("name"),
                data.get("email"),
                data.get("payment_terms", "Net 30"),
                data_json,
            )

    async def list_customers(self, limit: int = 100) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT data_json FROM ar_customers
                ORDER BY name ASC
                LIMIT $1
                """,
                limit,
            )
            results = []
            for row in rows:
                data = row["data_json"]
                results.append(json.loads(data) if isinstance(data, str) else data)
            return results

    async def close(self) -> None:
        pass


def get_ar_invoice_store() -> ARInvoiceStoreBackend:
    """
    Get the global AR invoice store instance.

    Backend is selected based on ARAGORA_AR_STORE_BACKEND env var:
    - "memory": InMemoryARInvoiceStore (for testing)
    - "sqlite": SQLiteARInvoiceStore (default, single-instance)
    - "postgres": PostgresARInvoiceStore (multi-instance)
    """
    global _ar_invoice_store

    with _store_lock:
        if _ar_invoice_store is not None:
            return _ar_invoice_store

        backend = os.getenv("ARAGORA_AR_STORE_BACKEND")
        if not backend:
            backend = os.getenv("ARAGORA_DB_BACKEND", "sqlite")
        backend = backend.lower()

        if backend == "memory":
            _ar_invoice_store = InMemoryARInvoiceStore()
            logger.info("Using in-memory AR invoice store")
        elif backend in ("postgres", "postgresql"):
            logger.info("Using PostgreSQL AR invoice store")
            try:
                from aragora.storage.postgres_store import get_postgres_pool
                from aragora.utils.async_utils import run_async

                # Initialize PostgreSQL store with connection pool using run_async
                # to safely handle both sync and async contexts
                async def init_postgres_store():
                    pool = await get_postgres_pool()
                    store = PostgresARInvoiceStore(pool)
                    await store.initialize()
                    return store

                _ar_invoice_store = run_async(init_postgres_store())
            except Exception as e:
                logger.warning(f"PostgreSQL not available, falling back to SQLite: {e}")
                _ar_invoice_store = SQLiteARInvoiceStore()
        else:
            _ar_invoice_store = SQLiteARInvoiceStore()
            logger.info("Using SQLite AR invoice store")

        return _ar_invoice_store


def set_ar_invoice_store(store: ARInvoiceStoreBackend) -> None:
    """Set a custom AR invoice store instance."""
    global _ar_invoice_store
    with _store_lock:
        _ar_invoice_store = store


def reset_ar_invoice_store() -> None:
    """Reset the global AR invoice store (for testing)."""
    global _ar_invoice_store
    with _store_lock:
        _ar_invoice_store = None


__all__ = [
    "ARInvoiceStoreBackend",
    "InMemoryARInvoiceStore",
    "SQLiteARInvoiceStore",
    "PostgresARInvoiceStore",
    "get_ar_invoice_store",
    "set_ar_invoice_store",
    "reset_ar_invoice_store",
]
