"""
Decision Receipt Storage with Signature Support.

Provides persistent storage for decision receipts with:
- Cryptographic signature storage and verification
- Date range queries
- Retention policy enforcement
- Full-text search on receipt data

Extends the basic receipt storage in audit_trail_store.py with
advanced features for compliance and auditing.

Usage:
    from aragora.storage.receipt_store import get_receipt_store

    store = get_receipt_store()
    await store.save(receipt, signed_receipt=signed_data)
    receipt = await store.get(receipt_id)
    is_valid = await store.verify_signature(receipt_id)
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aragora.storage.backends import (
    POSTGRESQL_AVAILABLE,
    DatabaseBackend,
    PostgreSQLBackend,
    SQLiteBackend,
)

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_RETENTION_DAYS = int(os.environ.get("ARAGORA_RECEIPT_RETENTION_DAYS", "2555"))  # ~7 years
DEFAULT_DB_PATH = (
    Path(os.environ.get("ARAGORA_DATA_DIR", str(Path.home() / ".aragora"))) / "receipts.db"
)

# Global singleton
_receipt_store: Optional["ReceiptStore"] = None
_store_lock = threading.RLock()


@dataclass
class StoredReceipt:
    """A stored decision receipt with signature metadata."""

    receipt_id: str
    gauntlet_id: str
    debate_id: Optional[str]
    created_at: float
    expires_at: Optional[float]
    verdict: str
    confidence: float
    risk_level: str
    risk_score: float
    checksum: str
    # Signature fields
    signature: Optional[str] = None
    signature_algorithm: Optional[str] = None
    signature_key_id: Optional[str] = None
    signed_at: Optional[float] = None
    # Links
    audit_trail_id: Optional[str] = None
    # Full data
    data: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        result = {
            "receipt_id": self.receipt_id,
            "gauntlet_id": self.gauntlet_id,
            "debate_id": self.debate_id,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "verdict": self.verdict,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "risk_score": self.risk_score,
            "checksum": self.checksum,
            "audit_trail_id": self.audit_trail_id,
            "is_signed": self.signature is not None,
        }
        if self.signature:
            result["signature_metadata"] = {
                "algorithm": self.signature_algorithm,
                "key_id": self.signature_key_id,
                "signed_at": self.signed_at,
            }
        return result

    def to_full_dict(self) -> Dict[str, Any]:
        """Convert to full dictionary including data payload."""
        result = self.to_dict()
        result.update(self.data)
        return result


@dataclass
class SignatureVerificationResult:
    """Result of signature verification."""

    receipt_id: str
    is_valid: bool
    algorithm: Optional[str] = None
    key_id: Optional[str] = None
    signed_at: Optional[float] = None
    verified_at: float = field(default_factory=lambda: time.time())
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API response."""
        return {
            "receipt_id": self.receipt_id,
            "signature_valid": self.is_valid,
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "signed_at": self.signed_at,
            "verification_timestamp": datetime.fromtimestamp(
                self.verified_at, tz=timezone.utc
            ).isoformat(),
            "error": self.error,
        }


class ReceiptStore:
    """
    Database-backed storage for decision receipts with signature support.

    Supports SQLite (default) and PostgreSQL backends.
    Provides full CRUD operations, signature verification, and retention management.
    """

    # SQLite schema (uses REAL for floating point)
    SCHEMA_STATEMENTS_SQLITE = [
        """
        CREATE TABLE IF NOT EXISTS receipts (
            receipt_id TEXT PRIMARY KEY,
            gauntlet_id TEXT NOT NULL UNIQUE,
            debate_id TEXT,
            created_at REAL NOT NULL,
            expires_at REAL,
            verdict TEXT NOT NULL,
            confidence REAL NOT NULL,
            risk_level TEXT NOT NULL,
            risk_score REAL NOT NULL DEFAULT 0.0,
            checksum TEXT NOT NULL,
            signature TEXT,
            signature_algorithm TEXT,
            signature_key_id TEXT,
            signed_at REAL,
            audit_trail_id TEXT,
            data_json TEXT NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_receipts_created ON receipts(created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_receipts_expires ON receipts(expires_at)",
        "CREATE INDEX IF NOT EXISTS idx_receipts_gauntlet ON receipts(gauntlet_id)",
        "CREATE INDEX IF NOT EXISTS idx_receipts_debate ON receipts(debate_id)",
        "CREATE INDEX IF NOT EXISTS idx_receipts_verdict ON receipts(verdict)",
        "CREATE INDEX IF NOT EXISTS idx_receipts_risk ON receipts(risk_level)",
        "CREATE INDEX IF NOT EXISTS idx_receipts_signed ON receipts(signed_at)",
    ]

    # PostgreSQL schema (uses DOUBLE PRECISION for floating point, JSONB for data)
    SCHEMA_STATEMENTS_POSTGRESQL = [
        """
        CREATE TABLE IF NOT EXISTS receipts (
            receipt_id TEXT PRIMARY KEY,
            gauntlet_id TEXT NOT NULL UNIQUE,
            debate_id TEXT,
            created_at DOUBLE PRECISION NOT NULL,
            expires_at DOUBLE PRECISION,
            verdict TEXT NOT NULL,
            confidence DOUBLE PRECISION NOT NULL,
            risk_level TEXT NOT NULL,
            risk_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
            checksum TEXT NOT NULL,
            signature TEXT,
            signature_algorithm TEXT,
            signature_key_id TEXT,
            signed_at DOUBLE PRECISION,
            audit_trail_id TEXT,
            data_json JSONB NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_receipts_created ON receipts(created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_receipts_expires ON receipts(expires_at)",
        "CREATE INDEX IF NOT EXISTS idx_receipts_gauntlet ON receipts(gauntlet_id)",
        "CREATE INDEX IF NOT EXISTS idx_receipts_debate ON receipts(debate_id)",
        "CREATE INDEX IF NOT EXISTS idx_receipts_verdict ON receipts(verdict)",
        "CREATE INDEX IF NOT EXISTS idx_receipts_risk ON receipts(risk_level)",
        "CREATE INDEX IF NOT EXISTS idx_receipts_signed ON receipts(signed_at)",
        # PostgreSQL-specific: GIN index for JSONB queries
        "CREATE INDEX IF NOT EXISTS idx_receipts_data_gin ON receipts USING GIN (data_json)",
    ]

    # Legacy property for backwards compatibility
    SCHEMA_STATEMENTS = SCHEMA_STATEMENTS_SQLITE

    def __init__(
        self,
        db_path: Optional[Path] = None,
        backend: Optional[str] = None,
        database_url: Optional[str] = None,
        retention_days: int = DEFAULT_RETENTION_DAYS,
    ):
        """
        Initialize receipt store.

        Args:
            db_path: Path to SQLite database (used when backend="sqlite")
            backend: Database backend ("sqlite" or "postgresql")
            database_url: PostgreSQL connection URL
            retention_days: Days to retain receipts (default: 2555 = ~7 years)
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self.retention_days = retention_days
        self._local = threading.local()

        # Determine backend type
        env_url = os.environ.get("DATABASE_URL") or os.environ.get("ARAGORA_DATABASE_URL")
        actual_url = database_url or env_url

        if backend is None:
            env_backend = os.environ.get("ARAGORA_DB_BACKEND", "sqlite").lower()
            backend = "postgresql" if (actual_url and env_backend == "postgresql") else "sqlite"

        self.backend_type = backend
        self._backend: Optional[DatabaseBackend] = None

        if backend == "postgresql":
            if not actual_url:
                raise ValueError("PostgreSQL backend requires DATABASE_URL")
            if not POSTGRESQL_AVAILABLE:
                raise ImportError("psycopg2 required for PostgreSQL")
            self._backend = PostgreSQLBackend(actual_url)
            logger.info("ReceiptStore using PostgreSQL backend")
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._backend = SQLiteBackend(str(self.db_path))
            logger.info(f"ReceiptStore using SQLite backend: {self.db_path}")

        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema based on backend type."""
        if self._backend is None:
            return

        # Select appropriate schema for backend
        if self.backend_type == "postgresql":
            schema_statements = self.SCHEMA_STATEMENTS_POSTGRESQL
        else:
            schema_statements = self.SCHEMA_STATEMENTS_SQLITE

        for statement in schema_statements:
            try:
                self._backend.execute_write(statement)
            except Exception as e:
                logger.debug(f"Schema statement skipped: {e}")

    # =========================================================================
    # Core CRUD Operations
    # =========================================================================

    def save(
        self,
        receipt_dict: Dict[str, Any],
        signed_receipt: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a decision receipt.

        Args:
            receipt_dict: Receipt data from DecisionReceipt.to_dict()
            signed_receipt: Optional SignedReceipt data with signature

        Returns:
            receipt_id of saved receipt
        """
        if self._backend is None:
            raise RuntimeError("ReceiptStore not initialized")

        receipt_id = receipt_dict.get("receipt_id", "")
        gauntlet_id = receipt_dict.get("gauntlet_id", "")
        debate_id = receipt_dict.get("debate_id")

        # Parse timestamp
        created_at = receipt_dict.get("timestamp", time.time())
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
            except (ValueError, AttributeError):
                created_at = time.time()

        # Calculate expiration
        expires_at = created_at + (self.retention_days * 86400)

        # Extract signature data if provided
        signature = None
        signature_algorithm = None
        signature_key_id = None
        signed_at = None

        if signed_receipt:
            signature = signed_receipt.get("signature")
            sig_meta = signed_receipt.get("signature_metadata", {})
            signature_algorithm = sig_meta.get("algorithm")
            signature_key_id = sig_meta.get("key_id")
            signed_at_str = sig_meta.get("timestamp")
            if signed_at_str:
                try:
                    signed_at = datetime.fromisoformat(
                        signed_at_str.replace("Z", "+00:00")
                    ).timestamp()
                except (ValueError, AttributeError):
                    signed_at = time.time()

        params = (
            receipt_id,
            gauntlet_id,
            debate_id,
            created_at,
            expires_at,
            receipt_dict.get("verdict", ""),
            receipt_dict.get("confidence", 0.0),
            receipt_dict.get("risk_level", "MEDIUM"),
            receipt_dict.get("risk_score", 0.0),
            receipt_dict.get("checksum", ""),
            signature,
            signature_algorithm,
            signature_key_id,
            signed_at,
            receipt_dict.get("audit_trail_id"),
            json.dumps(receipt_dict),
        )

        # Use backend-specific upsert syntax
        if self.backend_type == "postgresql":
            self._backend.execute_write(
                """
                INSERT INTO receipts
                (receipt_id, gauntlet_id, debate_id, created_at, expires_at,
                 verdict, confidence, risk_level, risk_score, checksum,
                 signature, signature_algorithm, signature_key_id, signed_at,
                 audit_trail_id, data_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (receipt_id) DO UPDATE SET
                    gauntlet_id = EXCLUDED.gauntlet_id,
                    debate_id = EXCLUDED.debate_id,
                    created_at = EXCLUDED.created_at,
                    expires_at = EXCLUDED.expires_at,
                    verdict = EXCLUDED.verdict,
                    confidence = EXCLUDED.confidence,
                    risk_level = EXCLUDED.risk_level,
                    risk_score = EXCLUDED.risk_score,
                    checksum = EXCLUDED.checksum,
                    signature = EXCLUDED.signature,
                    signature_algorithm = EXCLUDED.signature_algorithm,
                    signature_key_id = EXCLUDED.signature_key_id,
                    signed_at = EXCLUDED.signed_at,
                    audit_trail_id = EXCLUDED.audit_trail_id,
                    data_json = EXCLUDED.data_json
                """,
                params,
            )
        else:
            # SQLite uses INSERT OR REPLACE
            self._backend.execute_write(
                """
                INSERT OR REPLACE INTO receipts
                (receipt_id, gauntlet_id, debate_id, created_at, expires_at,
                 verdict, confidence, risk_level, risk_score, checksum,
                 signature, signature_algorithm, signature_key_id, signed_at,
                 audit_trail_id, data_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                params,
            )
        logger.debug(f"Saved receipt: {receipt_id}")
        return receipt_id

    def get(self, receipt_id: str) -> Optional[StoredReceipt]:
        """
        Get a receipt by ID.

        Args:
            receipt_id: Receipt ID to retrieve

        Returns:
            StoredReceipt or None if not found
        """
        if self._backend is None:
            return None

        row = self._backend.fetch_one(
            """
            SELECT receipt_id, gauntlet_id, debate_id, created_at, expires_at,
                   verdict, confidence, risk_level, risk_score, checksum,
                   signature, signature_algorithm, signature_key_id, signed_at,
                   audit_trail_id, data_json
            FROM receipts WHERE receipt_id = ?
            """,
            (receipt_id,),
        )
        if row:
            return self._row_to_stored_receipt(row)
        return None

    def get_by_gauntlet(self, gauntlet_id: str) -> Optional[StoredReceipt]:
        """Get receipt by gauntlet ID."""
        if self._backend is None:
            return None

        row = self._backend.fetch_one(
            """
            SELECT receipt_id, gauntlet_id, debate_id, created_at, expires_at,
                   verdict, confidence, risk_level, risk_score, checksum,
                   signature, signature_algorithm, signature_key_id, signed_at,
                   audit_trail_id, data_json
            FROM receipts WHERE gauntlet_id = ?
            """,
            (gauntlet_id,),
        )
        if row:
            return self._row_to_stored_receipt(row)
        return None

    def _row_to_stored_receipt(self, row: Tuple) -> StoredReceipt:
        """Convert database row to StoredReceipt."""
        return StoredReceipt(
            receipt_id=row[0],
            gauntlet_id=row[1],
            debate_id=row[2],
            created_at=row[3],
            expires_at=row[4],
            verdict=row[5],
            confidence=row[6],
            risk_level=row[7],
            risk_score=row[8],
            checksum=row[9],
            signature=row[10],
            signature_algorithm=row[11],
            signature_key_id=row[12],
            signed_at=row[13],
            audit_trail_id=row[14],
            data=json.loads(row[15]) if row[15] else {},
        )

    def list(
        self,
        limit: int = 20,
        offset: int = 0,
        verdict: Optional[str] = None,
        risk_level: Optional[str] = None,
        date_from: Optional[float] = None,
        date_to: Optional[float] = None,
        signed_only: bool = False,
        sort_by: str = "created_at",
        order: str = "desc",
    ) -> List[StoredReceipt]:
        """
        List receipts with filtering and pagination.

        Args:
            limit: Maximum receipts to return
            offset: Pagination offset
            verdict: Filter by verdict (APPROVED, REJECTED, etc.)
            risk_level: Filter by risk level (LOW, MEDIUM, HIGH, CRITICAL)
            date_from: Filter by created_at >= date_from (timestamp)
            date_to: Filter by created_at <= date_to (timestamp)
            signed_only: Only return signed receipts
            sort_by: Field to sort by (created_at, confidence, risk_score)
            order: Sort order (asc, desc)

        Returns:
            List of StoredReceipt objects
        """
        if self._backend is None:
            return []

        conditions = []
        params: List[Any] = []

        if verdict:
            conditions.append("verdict = ?")
            params.append(verdict)
        if risk_level:
            conditions.append("risk_level = ?")
            params.append(risk_level)
        if date_from:
            conditions.append("created_at >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("created_at <= ?")
            params.append(date_to)
        if signed_only:
            conditions.append("signature IS NOT NULL")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        # Validate sort field
        valid_sort_fields = {"created_at", "confidence", "risk_score", "signed_at"}
        if sort_by not in valid_sort_fields:
            sort_by = "created_at"
        order_clause = "DESC" if order.lower() == "desc" else "ASC"

        params.extend([limit, offset])

        rows = self._backend.fetch_all(
            f"""
            SELECT receipt_id, gauntlet_id, debate_id, created_at, expires_at,
                   verdict, confidence, risk_level, risk_score, checksum,
                   signature, signature_algorithm, signature_key_id, signed_at,
                   audit_trail_id, data_json
            FROM receipts
            WHERE {where_clause}
            ORDER BY {sort_by} {order_clause}
            LIMIT ? OFFSET ?
            """,  # nosec B608 - where_clause built from hardcoded conditions
            tuple(params),
        )

        return [self._row_to_stored_receipt(row) for row in rows]

    def count(
        self,
        verdict: Optional[str] = None,
        risk_level: Optional[str] = None,
        date_from: Optional[float] = None,
        date_to: Optional[float] = None,
        signed_only: bool = False,
    ) -> int:
        """Get total count of receipts matching filters."""
        if self._backend is None:
            return 0

        conditions = []
        params: List[Any] = []

        if verdict:
            conditions.append("verdict = ?")
            params.append(verdict)
        if risk_level:
            conditions.append("risk_level = ?")
            params.append(risk_level)
        if date_from:
            conditions.append("created_at >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("created_at <= ?")
            params.append(date_to)
        if signed_only:
            conditions.append("signature IS NOT NULL")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        row = self._backend.fetch_one(
            f"SELECT COUNT(*) FROM receipts WHERE {where_clause}",  # nosec B608
            tuple(params),
        )
        return row[0] if row else 0

    # =========================================================================
    # Signature Operations
    # =========================================================================

    def update_signature(
        self,
        receipt_id: str,
        signature: str,
        algorithm: str,
        key_id: str,
    ) -> bool:
        """
        Update receipt with signature.

        Args:
            receipt_id: Receipt to sign
            signature: Base64-encoded signature
            algorithm: Signing algorithm (HMAC-SHA256, RSA-SHA256, Ed25519)
            key_id: Identifier of signing key

        Returns:
            True if updated, False if receipt not found
        """
        if self._backend is None:
            return False

        # Check receipt exists
        existing = self.get(receipt_id)
        if not existing:
            return False

        signed_at = time.time()

        self._backend.execute_write(
            """
            UPDATE receipts
            SET signature = ?, signature_algorithm = ?,
                signature_key_id = ?, signed_at = ?
            WHERE receipt_id = ?
            """,
            (signature, algorithm, key_id, signed_at, receipt_id),
        )
        logger.info(f"Updated signature for receipt: {receipt_id}")
        return True

    def verify_signature(self, receipt_id: str) -> SignatureVerificationResult:
        """
        Verify the cryptographic signature of a receipt.

        Args:
            receipt_id: Receipt ID to verify

        Returns:
            SignatureVerificationResult with validation status
        """
        receipt = self.get(receipt_id)

        if not receipt:
            return SignatureVerificationResult(
                receipt_id=receipt_id,
                is_valid=False,
                error="Receipt not found",
            )

        if not receipt.signature:
            return SignatureVerificationResult(
                receipt_id=receipt_id,
                is_valid=False,
                error="Receipt is not signed",
            )

        try:
            from aragora.gauntlet.signing import (
                ReceiptSigner,
                SignatureMetadata,
                SignedReceipt,
            )

            # Reconstruct signature metadata
            sig_meta = SignatureMetadata(
                algorithm=receipt.signature_algorithm or "HMAC-SHA256",
                key_id=receipt.signature_key_id or "unknown",
                timestamp=(
                    datetime.fromtimestamp(receipt.signed_at or 0, tz=timezone.utc).isoformat()
                    if receipt.signed_at
                    else datetime.now(timezone.utc).isoformat()
                ),
            )

            # Reconstruct signed receipt
            signed = SignedReceipt(
                receipt_data=receipt.data,
                signature=receipt.signature,
                signature_metadata=sig_meta,
            )

            # Verify using signer
            signer = ReceiptSigner()
            is_valid = signer.verify(signed)

            return SignatureVerificationResult(
                receipt_id=receipt_id,
                is_valid=is_valid,
                algorithm=receipt.signature_algorithm,
                key_id=receipt.signature_key_id,
                signed_at=receipt.signed_at,
            )

        except ImportError as e:
            return SignatureVerificationResult(
                receipt_id=receipt_id,
                is_valid=False,
                error=f"Signing module not available: {e}",
            )
        except Exception as e:
            logger.warning(f"Signature verification failed: {e}")
            return SignatureVerificationResult(
                receipt_id=receipt_id,
                is_valid=False,
                algorithm=receipt.signature_algorithm,
                key_id=receipt.signature_key_id,
                signed_at=receipt.signed_at,
                error=str(e),
            )

    def verify_batch(
        self, receipt_ids: List[str]
    ) -> Tuple[List[SignatureVerificationResult], Dict[str, int]]:
        """
        Verify signatures for multiple receipts.

        Args:
            receipt_ids: List of receipt IDs to verify

        Returns:
            Tuple of (results list, summary dict)
        """
        results = []
        summary = {"total": len(receipt_ids), "valid": 0, "invalid": 0, "not_signed": 0}

        for receipt_id in receipt_ids:
            result = self.verify_signature(receipt_id)
            results.append(result)

            if result.is_valid:
                summary["valid"] += 1
            elif result.error == "Receipt is not signed":
                summary["not_signed"] += 1
            else:
                summary["invalid"] += 1

        return results, summary

    # =========================================================================
    # Integrity Verification
    # =========================================================================

    def verify_integrity(self, receipt_id: str) -> Dict[str, Any]:
        """
        Verify the integrity checksum of a receipt.

        Args:
            receipt_id: Receipt ID to verify

        Returns:
            Dict with checksum verification result
        """
        receipt = self.get(receipt_id)

        if not receipt:
            return {
                "receipt_id": receipt_id,
                "integrity_valid": False,
                "error": "Receipt not found",
            }

        try:
            from aragora.export.decision_receipt import DecisionReceipt

            # Recompute checksum from data
            loaded_receipt = DecisionReceipt.from_dict(receipt.data)
            computed_checksum = loaded_receipt._compute_checksum()

            is_valid = computed_checksum == receipt.checksum

            return {
                "receipt_id": receipt_id,
                "integrity_valid": is_valid,
                "stored_checksum": receipt.checksum,
                "computed_checksum": computed_checksum,
                "verified_at": datetime.now(timezone.utc).isoformat(),
            }

        except Exception as e:
            return {
                "receipt_id": receipt_id,
                "integrity_valid": False,
                "error": str(e),
            }

    # =========================================================================
    # Retention & Cleanup
    # =========================================================================

    def cleanup_expired(
        self,
        retention_days: Optional[int] = None,
        operator: str = "system:retention_cleanup",
        log_deletions: bool = True,
    ) -> int:
        """
        Remove receipts older than retention period with audit trail.

        Logs all deletions to the receipt deletion log before removing
        for GDPR/SOC2 compliance.

        Args:
            retention_days: Override default retention (default: use store's setting)
            operator: Identifier for who/what initiated the cleanup
            log_deletions: Whether to log deletions to audit trail (default True)

        Returns:
            Number of receipts removed
        """
        if self._backend is None:
            return 0

        days = retention_days if retention_days is not None else self.retention_days
        cutoff = time.time() - (days * 86400)

        # Get receipts to be deleted (for audit logging)
        rows = self._backend.fetch_all(
            """
            SELECT receipt_id, checksum, gauntlet_id, verdict
            FROM receipts WHERE created_at < ?
            """,
            (cutoff,),
        )

        if not rows:
            return 0

        count = len(rows)

        # Log deletions before removing
        if log_deletions:
            try:
                from aragora.storage.receipt_deletion_log import get_receipt_deletion_log

                deletion_log = get_receipt_deletion_log()
                receipts_to_log = [
                    {
                        "receipt_id": row[0],
                        "checksum": row[1],
                        "gauntlet_id": row[2],
                        "verdict": row[3],
                        "metadata": {"retention_days": days},
                    }
                    for row in rows
                ]
                deletion_log.log_batch_deletion(
                    receipts=receipts_to_log,
                    reason="retention_expired",
                    operator=operator,
                )
                logger.info(f"Logged {count} receipt deletions to audit trail")
            except Exception as e:
                logger.warning(f"Failed to log deletions to audit trail: {e}")
                # Continue with deletion even if logging fails
                # (configurable behavior could be added)

        # Now delete the receipts
        self._backend.execute_write(
            "DELETE FROM receipts WHERE created_at < ?",
            (cutoff,),
        )
        logger.info(f"Removed {count} expired receipts (older than {days} days)")

        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get receipt statistics."""
        if self._backend is None:
            return {}

        total = self.count()
        signed = self.count(signed_only=True)

        # Verdict breakdown
        verdict_counts = {}
        for verdict in ["APPROVED", "REJECTED", "NEEDS_REVIEW", "INCONCLUSIVE"]:
            verdict_counts[verdict.lower()] = self.count(verdict=verdict)

        # Risk level breakdown
        risk_counts = {}
        for risk in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            risk_counts[risk.lower()] = self.count(risk_level=risk)

        return {
            "total": total,
            "signed": signed,
            "unsigned": total - signed,
            "by_verdict": verdict_counts,
            "by_risk_level": risk_counts,
            "retention_days": self.retention_days,
        }


# =========================================================================
# Module-level Functions
# =========================================================================


def get_receipt_store() -> ReceiptStore:
    """
    Get or create the global receipt store.

    Returns:
        ReceiptStore singleton instance
    """
    global _receipt_store

    with _store_lock:
        if _receipt_store is None:
            _receipt_store = ReceiptStore()
        return _receipt_store


def set_receipt_store(store: Optional[ReceiptStore]) -> None:
    """
    Set the global receipt store (for testing).

    Args:
        store: ReceiptStore instance or None to reset
    """
    global _receipt_store

    with _store_lock:
        _receipt_store = store
