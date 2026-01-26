"""
Credits System for Aragora.

Provides prepaid credit management for organizations:
- Issue promotional, referral, or refund credits
- Deduct credits before charging cards
- Track credit transactions and expiration
- Support multiple credit types with different policies

Usage:
    from aragora.billing.credits import get_credit_manager

    manager = get_credit_manager()
    await manager.issue_credit(
        org_id="org_123",
        amount_cents=2000,  # $20
        credit_type=CreditTransactionType.PROMOTIONAL,
        description="Welcome credit"
    )
    balance = await manager.get_balance("org_123")
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CreditTransactionType(str, Enum):
    """Types of credit transactions."""

    PROMOTIONAL = "promotional"  # Marketing/welcome credits
    REFERRAL = "referral"  # Earned from referrals
    REFUND = "refund"  # Converted from refund
    PURCHASE = "purchase"  # Prepaid credit purchase
    USAGE = "usage"  # Deducted for API usage
    ADJUSTMENT = "adjustment"  # Manual adjustment by admin
    EXPIRED = "expired"  # Credits that expired


@dataclass
class CreditTransaction:
    """A credit transaction record."""

    id: str
    org_id: str
    amount_cents: int  # Positive = credit, negative = debit
    transaction_type: CreditTransactionType
    description: str
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: Optional[str] = None  # User who created the transaction
    reference_id: Optional[str] = None  # Reference to related entity (e.g., refund ID)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "org_id": self.org_id,
            "amount_cents": self.amount_cents,
            "transaction_type": self.transaction_type.value,
            "description": self.description,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "reference_id": self.reference_id,
        }

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> CreditTransaction:
        """Create from database row."""
        return cls(
            id=row["id"],
            org_id=row["org_id"],
            amount_cents=row["amount_cents"],
            transaction_type=CreditTransactionType(row["transaction_type"]),
            description=row["description"],
            expires_at=(datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None),
            created_at=datetime.fromisoformat(row["created_at"]),
            created_by=row["created_by"],
            reference_id=row["reference_id"],
        )


@dataclass
class CreditAccount:
    """Credit account summary for an organization."""

    org_id: str
    balance_cents: int = 0
    lifetime_issued_cents: int = 0
    lifetime_redeemed_cents: int = 0
    lifetime_expired_cents: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "org_id": self.org_id,
            "balance_cents": self.balance_cents,
            "balance_usd": self.balance_cents / 100,
            "lifetime_issued_cents": self.lifetime_issued_cents,
            "lifetime_issued_usd": self.lifetime_issued_cents / 100,
            "lifetime_redeemed_cents": self.lifetime_redeemed_cents,
            "lifetime_redeemed_usd": self.lifetime_redeemed_cents / 100,
            "lifetime_expired_cents": self.lifetime_expired_cents,
            "lifetime_expired_usd": self.lifetime_expired_cents / 100,
        }


@dataclass
class DeductionResult:
    """Result of attempting to deduct credits."""

    success: bool
    amount_deducted_cents: int = 0
    remaining_amount_cents: int = 0  # Amount still owed after credits
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "amount_deducted_cents": self.amount_deducted_cents,
            "remaining_amount_cents": self.remaining_amount_cents,
            "message": self.message,
        }


class CreditManager:
    """
    Manages credit accounts and transactions.

    Thread-safe SQLite-backed credit storage with support for:
    - Credit issuance with expiration
    - Credit deduction with FIFO expiration
    - Balance tracking per organization
    - Transaction history
    """

    def __init__(self, db_path: Optional[str] = None):
        """Initialize credit manager.

        Args:
            db_path: Path to SQLite database. Defaults to ~/.aragora/credits.db
        """
        if db_path is None:
            db_dir = Path.home() / ".aragora"
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "credits.db")

        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS credit_transactions (
                id TEXT PRIMARY KEY,
                org_id TEXT NOT NULL,
                amount_cents INTEGER NOT NULL,
                transaction_type TEXT NOT NULL,
                description TEXT NOT NULL,
                expires_at TEXT,
                created_at TEXT NOT NULL,
                created_by TEXT,
                reference_id TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_credits_org_id
            ON credit_transactions(org_id);

            CREATE INDEX IF NOT EXISTS idx_credits_expires_at
            ON credit_transactions(expires_at)
            WHERE expires_at IS NOT NULL;

            CREATE INDEX IF NOT EXISTS idx_credits_created_at
            ON credit_transactions(created_at);
            """)
        conn.commit()

    async def issue_credit(
        self,
        org_id: str,
        amount_cents: int,
        credit_type: CreditTransactionType,
        description: str,
        expires_at: Optional[datetime] = None,
        created_by: Optional[str] = None,
        reference_id: Optional[str] = None,
    ) -> CreditTransaction:
        """Issue credits to an organization.

        Args:
            org_id: Organization ID
            amount_cents: Amount in cents (must be positive)
            credit_type: Type of credit
            description: Human-readable description
            expires_at: Optional expiration date
            created_by: User who issued the credit
            reference_id: Optional reference to related entity

        Returns:
            The created transaction

        Raises:
            ValueError: If amount is not positive
        """
        if amount_cents <= 0:
            raise ValueError("Credit amount must be positive")

        transaction = CreditTransaction(
            id=f"cred_{uuid.uuid4().hex[:16]}",
            org_id=org_id,
            amount_cents=amount_cents,
            transaction_type=credit_type,
            description=description,
            expires_at=expires_at,
            created_by=created_by,
            reference_id=reference_id,
        )

        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO credit_transactions
            (id, org_id, amount_cents, transaction_type, description,
             expires_at, created_at, created_by, reference_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transaction.id,
                transaction.org_id,
                transaction.amount_cents,
                transaction.transaction_type.value,
                transaction.description,
                transaction.expires_at.isoformat() if transaction.expires_at else None,
                transaction.created_at.isoformat(),
                transaction.created_by,
                transaction.reference_id,
            ),
        )
        conn.commit()

        logger.info(f"Issued {amount_cents} cents credit to org {org_id}: {description}")
        return transaction

    async def deduct_credit(
        self,
        org_id: str,
        amount_cents: int,
        description: str,
        created_by: Optional[str] = None,
        reference_id: Optional[str] = None,
    ) -> DeductionResult:
        """Deduct credits from an organization.

        Uses available credits up to the requested amount.
        Does not fail if insufficient credits - returns partial deduction.

        Args:
            org_id: Organization ID
            amount_cents: Amount to deduct in cents
            description: Reason for deduction
            created_by: User who initiated the deduction
            reference_id: Optional reference to related entity

        Returns:
            DeductionResult with amount actually deducted
        """
        if amount_cents <= 0:
            return DeductionResult(
                success=True,
                amount_deducted_cents=0,
                remaining_amount_cents=0,
                message="No deduction needed",
            )

        # Get available balance (expires non-expired credits first)
        balance = await self.get_balance(org_id)

        if balance <= 0:
            return DeductionResult(
                success=True,
                amount_deducted_cents=0,
                remaining_amount_cents=amount_cents,
                message="No credits available",
            )

        # Deduct up to available balance
        amount_to_deduct = min(amount_cents, balance)

        transaction = CreditTransaction(
            id=f"cred_{uuid.uuid4().hex[:16]}",
            org_id=org_id,
            amount_cents=-amount_to_deduct,  # Negative for deduction
            transaction_type=CreditTransactionType.USAGE,
            description=description,
            created_by=created_by,
            reference_id=reference_id,
        )

        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO credit_transactions
            (id, org_id, amount_cents, transaction_type, description,
             expires_at, created_at, created_by, reference_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transaction.id,
                transaction.org_id,
                transaction.amount_cents,
                transaction.transaction_type.value,
                transaction.description,
                None,  # Deductions don't expire
                transaction.created_at.isoformat(),
                transaction.created_by,
                transaction.reference_id,
            ),
        )
        conn.commit()

        remaining = amount_cents - amount_to_deduct
        logger.info(
            f"Deducted {amount_to_deduct} cents from org {org_id}, "
            f"remaining to bill: {remaining} cents"
        )

        return DeductionResult(
            success=True,
            amount_deducted_cents=amount_to_deduct,
            remaining_amount_cents=remaining,
            message=f"Deducted {amount_to_deduct} cents from credits",
        )

    async def get_balance(self, org_id: str) -> int:
        """Get current credit balance for an organization.

        Excludes expired credits from balance calculation.

        Args:
            org_id: Organization ID

        Returns:
            Current balance in cents
        """
        now = datetime.now(timezone.utc).isoformat()

        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT COALESCE(SUM(amount_cents), 0) as balance
            FROM credit_transactions
            WHERE org_id = ?
            AND (expires_at IS NULL OR expires_at > ?)
            """,
            (org_id, now),
        )
        row = cursor.fetchone()
        return row["balance"] if row else 0

    async def get_account(self, org_id: str) -> CreditAccount:
        """Get full credit account details for an organization.

        Args:
            org_id: Organization ID

        Returns:
            CreditAccount with balance and lifetime totals
        """
        balance = await self.get_balance(org_id)

        conn = self._get_conn()

        # Get lifetime issued (positive transactions)
        cursor = conn.execute(
            """
            SELECT COALESCE(SUM(amount_cents), 0) as total
            FROM credit_transactions
            WHERE org_id = ? AND amount_cents > 0
            """,
            (org_id,),
        )
        lifetime_issued = cursor.fetchone()["total"]

        # Get lifetime redeemed (negative transactions, excluding expired)
        cursor = conn.execute(
            """
            SELECT COALESCE(SUM(ABS(amount_cents)), 0) as total
            FROM credit_transactions
            WHERE org_id = ? AND amount_cents < 0
            AND transaction_type != 'expired'
            """,
            (org_id,),
        )
        lifetime_redeemed = cursor.fetchone()["total"]

        # Get lifetime expired
        cursor = conn.execute(
            """
            SELECT COALESCE(SUM(ABS(amount_cents)), 0) as total
            FROM credit_transactions
            WHERE org_id = ? AND transaction_type = 'expired'
            """,
            (org_id,),
        )
        lifetime_expired = cursor.fetchone()["total"]

        return CreditAccount(
            org_id=org_id,
            balance_cents=balance,
            lifetime_issued_cents=lifetime_issued,
            lifetime_redeemed_cents=lifetime_redeemed,
            lifetime_expired_cents=lifetime_expired,
        )

    async def get_transactions(
        self,
        org_id: str,
        limit: int = 100,
        offset: int = 0,
        transaction_type: Optional[CreditTransactionType] = None,
    ) -> List[CreditTransaction]:
        """Get credit transactions for an organization.

        Args:
            org_id: Organization ID
            limit: Maximum transactions to return
            offset: Number of transactions to skip
            transaction_type: Filter by transaction type

        Returns:
            List of transactions ordered by creation date (newest first)
        """
        conn = self._get_conn()

        if transaction_type:
            cursor = conn.execute(
                """
                SELECT * FROM credit_transactions
                WHERE org_id = ? AND transaction_type = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (org_id, transaction_type.value, limit, offset),
            )
        else:
            cursor = conn.execute(
                """
                SELECT * FROM credit_transactions
                WHERE org_id = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (org_id, limit, offset),
            )

        return [CreditTransaction.from_row(row) for row in cursor.fetchall()]

    async def get_expiring_credits(
        self, org_id: str, within_days: int = 30
    ) -> List[CreditTransaction]:
        """Get credits that will expire within the specified period.

        Args:
            org_id: Organization ID
            within_days: Number of days to look ahead

        Returns:
            List of transactions with upcoming expirations
        """
        now = datetime.now(timezone.utc)
        deadline = (now + timedelta(days=within_days)).isoformat()

        conn = self._get_conn()
        cursor = conn.execute(
            """
            SELECT * FROM credit_transactions
            WHERE org_id = ?
            AND amount_cents > 0
            AND expires_at IS NOT NULL
            AND expires_at > ?
            AND expires_at <= ?
            ORDER BY expires_at ASC
            """,
            (org_id, now.isoformat(), deadline),
        )

        return [CreditTransaction.from_row(row) for row in cursor.fetchall()]

    async def expire_credits(self) -> int:
        """Expire credits that have passed their expiration date.

        Creates negative transactions to zero out expired credits.

        Returns:
            Number of credits expired
        """
        now = datetime.now(timezone.utc)

        conn = self._get_conn()

        # Find expired credits that haven't been recorded as expired
        cursor = conn.execute(
            """
            SELECT org_id, SUM(amount_cents) as expired_amount
            FROM credit_transactions
            WHERE expires_at IS NOT NULL
            AND expires_at <= ?
            AND amount_cents > 0
            GROUP BY org_id
            HAVING expired_amount > 0
            """,
            (now.isoformat(),),
        )

        expired_count = 0
        for row in cursor.fetchall():
            org_id = row["org_id"]
            expired_amount = row["expired_amount"]

            # Create expiration transaction
            transaction_id = f"cred_{uuid.uuid4().hex[:16]}"
            conn.execute(
                """
                INSERT INTO credit_transactions
                (id, org_id, amount_cents, transaction_type, description,
                 expires_at, created_at, created_by, reference_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transaction_id,
                    org_id,
                    -expired_amount,
                    CreditTransactionType.EXPIRED.value,
                    f"Credits expired on {now.date()}",
                    None,
                    now.isoformat(),
                    "system",
                    None,
                ),
            )
            expired_count += 1
            logger.info(f"Expired {expired_amount} cents for org {org_id}")

        conn.commit()
        return expired_count

    async def adjust_balance(
        self,
        org_id: str,
        amount_cents: int,
        description: str,
        created_by: str,
    ) -> CreditTransaction:
        """Make a manual balance adjustment (positive or negative).

        Args:
            org_id: Organization ID
            amount_cents: Adjustment amount (positive or negative)
            description: Reason for adjustment
            created_by: User making the adjustment

        Returns:
            The adjustment transaction
        """
        transaction = CreditTransaction(
            id=f"cred_{uuid.uuid4().hex[:16]}",
            org_id=org_id,
            amount_cents=amount_cents,
            transaction_type=CreditTransactionType.ADJUSTMENT,
            description=description,
            created_by=created_by,
        )

        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO credit_transactions
            (id, org_id, amount_cents, transaction_type, description,
             expires_at, created_at, created_by, reference_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transaction.id,
                transaction.org_id,
                transaction.amount_cents,
                transaction.transaction_type.value,
                transaction.description,
                None,
                transaction.created_at.isoformat(),
                transaction.created_by,
                None,
            ),
        )
        conn.commit()

        logger.info(f"Adjusted balance for org {org_id} by {amount_cents} cents: {description}")
        return transaction


# Global credit manager instance
_credit_manager: Optional[CreditManager] = None
_credit_manager_lock = threading.Lock()


def get_credit_manager(db_path: Optional[str] = None) -> CreditManager:
    """Get or create the global credit manager.

    Args:
        db_path: Optional database path (only used for first initialization)

    Returns:
        CreditManager instance
    """
    global _credit_manager
    with _credit_manager_lock:
        if _credit_manager is None:
            _credit_manager = CreditManager(db_path)
        return _credit_manager


def set_credit_manager(manager: CreditManager) -> None:
    """Set the global credit manager (for testing).

    Args:
        manager: CreditManager instance to use
    """
    global _credit_manager
    with _credit_manager_lock:
        _credit_manager = manager


def reset_credit_manager() -> None:
    """Reset the global credit manager (for testing)."""
    global _credit_manager
    with _credit_manager_lock:
        _credit_manager = None


__all__ = [
    "CreditTransactionType",
    "CreditTransaction",
    "CreditAccount",
    "DeductionResult",
    "CreditManager",
    "get_credit_manager",
    "set_credit_manager",
    "reset_credit_manager",
]
