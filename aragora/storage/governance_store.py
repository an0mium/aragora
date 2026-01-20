"""
GovernanceStore - Database persistence for decision governance artifacts.

Provides durable storage for:
- Approval requests (human-in-the-loop decisions)
- Verification history (formal verification results)
- Decision records (debate outcomes with provenance)
- Rollback points (safety checkpoints)

Supports SQLite (default) and PostgreSQL backends.

Usage:
    from aragora.storage.governance_store import GovernanceStore

    store = GovernanceStore()

    # Save approval request
    store.save_approval(approval_request)

    # Query pending approvals
    pending = store.list_approvals(status="pending")

    # Save verification result
    store.save_verification(verification_entry)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from aragora.storage.backends import (
    POSTGRESQL_AVAILABLE,
    DatabaseBackend,
    PostgreSQLBackend,
    SQLiteBackend,
)

logger = logging.getLogger(__name__)


@dataclass
class ApprovalRecord:
    """Persistent approval request record."""

    approval_id: str
    title: str
    description: str
    risk_level: str  # low, medium, high, critical
    status: str  # pending, approved, rejected, expired, cancelled
    requested_by: str
    requested_at: datetime
    changes_json: str  # JSON serialized changes
    timeout_seconds: int = 3600
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    org_id: Optional[str] = None
    workspace_id: Optional[str] = None
    metadata_json: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "approval_id": self.approval_id,
            "title": self.title,
            "description": self.description,
            "risk_level": self.risk_level,
            "status": self.status,
            "requested_by": self.requested_by,
            "requested_at": self.requested_at.isoformat() if isinstance(self.requested_at, datetime) else self.requested_at,
            "changes": json.loads(self.changes_json) if self.changes_json else [],
            "timeout_seconds": self.timeout_seconds,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat() if isinstance(self.approved_at, datetime) else self.approved_at,
            "rejection_reason": self.rejection_reason,
            "org_id": self.org_id,
            "workspace_id": self.workspace_id,
            "metadata": json.loads(self.metadata_json) if self.metadata_json else {},
        }


@dataclass
class VerificationRecord:
    """Persistent verification history entry."""

    verification_id: str
    claim: str
    claim_type: Optional[str]
    context: str
    result_json: str  # JSON serialized result
    timestamp: datetime
    verified_by: str  # system, agent name, etc.
    confidence: float = 0.0
    proof_tree_json: Optional[str] = None
    org_id: Optional[str] = None
    workspace_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "verification_id": self.verification_id,
            "claim": self.claim,
            "claim_type": self.claim_type,
            "context": self.context,
            "result": json.loads(self.result_json) if self.result_json else {},
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            "verified_by": self.verified_by,
            "confidence": self.confidence,
            "proof_tree": json.loads(self.proof_tree_json) if self.proof_tree_json else None,
            "org_id": self.org_id,
            "workspace_id": self.workspace_id,
        }


@dataclass
class DecisionRecord:
    """Persistent decision outcome record."""

    decision_id: str
    debate_id: str
    conclusion: str
    consensus_reached: bool
    confidence: float
    timestamp: datetime
    evidence_chain_json: str  # JSON serialized
    vote_pivots_json: str  # JSON serialized
    belief_changes_json: str  # JSON serialized
    agents_involved_json: str  # JSON serialized list
    org_id: Optional[str] = None
    workspace_id: Optional[str] = None
    metadata_json: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "decision_id": self.decision_id,
            "debate_id": self.debate_id,
            "conclusion": self.conclusion,
            "consensus_reached": self.consensus_reached,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat() if isinstance(self.timestamp, datetime) else self.timestamp,
            "evidence_chain": json.loads(self.evidence_chain_json) if self.evidence_chain_json else [],
            "vote_pivots": json.loads(self.vote_pivots_json) if self.vote_pivots_json else [],
            "belief_changes": json.loads(self.belief_changes_json) if self.belief_changes_json else [],
            "agents_involved": json.loads(self.agents_involved_json) if self.agents_involved_json else [],
            "org_id": self.org_id,
            "workspace_id": self.workspace_id,
            "metadata": json.loads(self.metadata_json) if self.metadata_json else {},
        }


class GovernanceStore:
    """
    Persistent storage for decision governance artifacts.

    Stores:
    - Approval requests with lifecycle tracking
    - Verification history with results
    - Decision records with provenance

    Supports SQLite (default) and PostgreSQL backends.
    """

    def __init__(
        self,
        db_path: str = "aragora_governance.db",
        backend: Optional[str] = None,
        database_url: Optional[str] = None,
    ):
        """
        Initialize governance store.

        Args:
            db_path: Path to SQLite database (used when backend="sqlite")
            backend: Database backend ("sqlite" or "postgresql")
            database_url: PostgreSQL connection URL
        """
        # Determine backend
        env_url = os.environ.get("DATABASE_URL") or os.environ.get("ARAGORA_DATABASE_URL")
        actual_url = database_url or env_url

        if backend is None:
            backend = "postgresql" if actual_url else "sqlite"

        self.backend_type = backend

        # Create backend
        if backend == "postgresql":
            if not actual_url:
                raise ValueError("PostgreSQL backend requires DATABASE_URL")
            if not POSTGRESQL_AVAILABLE:
                raise ImportError("psycopg2 required for PostgreSQL")
            self._backend: DatabaseBackend = PostgreSQLBackend(actual_url)
            logger.info("GovernanceStore using PostgreSQL backend")
        else:
            self.db_path = Path(db_path)
            self._backend = SQLiteBackend(db_path)
            logger.info(f"GovernanceStore using SQLite backend: {db_path}")

        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        # Approvals table
        self._backend.execute_write("""
            CREATE TABLE IF NOT EXISTS governance_approvals (
                approval_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                risk_level TEXT NOT NULL,
                status TEXT NOT NULL,
                requested_by TEXT,
                requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                changes_json TEXT,
                timeout_seconds INTEGER DEFAULT 3600,
                approved_by TEXT,
                approved_at TIMESTAMP,
                rejection_reason TEXT,
                org_id TEXT,
                workspace_id TEXT,
                metadata_json TEXT
            )
        """)

        # Verification history table
        self._backend.execute_write("""
            CREATE TABLE IF NOT EXISTS governance_verifications (
                verification_id TEXT PRIMARY KEY,
                claim TEXT NOT NULL,
                claim_type TEXT,
                context TEXT,
                result_json TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                verified_by TEXT,
                confidence REAL DEFAULT 0,
                proof_tree_json TEXT,
                org_id TEXT,
                workspace_id TEXT
            )
        """)

        # Decisions table
        self._backend.execute_write("""
            CREATE TABLE IF NOT EXISTS governance_decisions (
                decision_id TEXT PRIMARY KEY,
                debate_id TEXT NOT NULL,
                conclusion TEXT,
                consensus_reached INTEGER,
                confidence REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                evidence_chain_json TEXT,
                vote_pivots_json TEXT,
                belief_changes_json TEXT,
                agents_involved_json TEXT,
                org_id TEXT,
                workspace_id TEXT,
                metadata_json TEXT
            )
        """)

        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_approvals_status ON governance_approvals(status)",
            "CREATE INDEX IF NOT EXISTS idx_approvals_org ON governance_approvals(org_id)",
            "CREATE INDEX IF NOT EXISTS idx_approvals_requested ON governance_approvals(requested_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_verifications_timestamp ON governance_verifications(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_verifications_org ON governance_verifications(org_id)",
            "CREATE INDEX IF NOT EXISTS idx_decisions_debate ON governance_decisions(debate_id)",
            "CREATE INDEX IF NOT EXISTS idx_decisions_org ON governance_decisions(org_id)",
            "CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON governance_decisions(timestamp DESC)",
        ]
        for idx in indexes:
            try:
                self._backend.execute_write(idx)
            except Exception as e:
                logger.debug(f"Index creation skipped: {e}")

    # =========================================================================
    # Approval Management
    # =========================================================================

    def save_approval(
        self,
        approval_id: str,
        title: str,
        description: str,
        risk_level: str,
        status: str,
        requested_by: str,
        changes: list,
        timeout_seconds: int = 3600,
        org_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Save a new approval request.

        Args:
            approval_id: Unique approval ID
            title: Approval request title
            description: Description of what needs approval
            risk_level: Risk level (low, medium, high, critical)
            status: Current status
            requested_by: Who requested the approval
            changes: List of changes being approved
            timeout_seconds: Timeout for approval
            org_id: Organization ID
            workspace_id: Workspace ID
            metadata: Additional metadata

        Returns:
            The approval_id
        """
        now = datetime.now().isoformat()

        if self.backend_type == "postgresql":
            sql = """
                INSERT INTO governance_approvals (
                    approval_id, title, description, risk_level, status,
                    requested_by, requested_at, changes_json, timeout_seconds,
                    org_id, workspace_id, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (approval_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    metadata_json = EXCLUDED.metadata_json
            """
        else:
            sql = """
                INSERT OR REPLACE INTO governance_approvals (
                    approval_id, title, description, risk_level, status,
                    requested_by, requested_at, changes_json, timeout_seconds,
                    org_id, workspace_id, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

        self._backend.execute_write(
            sql,
            (
                approval_id,
                title,
                description,
                risk_level,
                status,
                requested_by,
                now,
                json.dumps(changes),
                timeout_seconds,
                org_id,
                workspace_id,
                json.dumps(metadata or {}),
            ),
        )

        logger.debug(f"Saved approval: {approval_id}")
        return approval_id

    def update_approval_status(
        self,
        approval_id: str,
        status: str,
        approved_by: Optional[str] = None,
        rejection_reason: Optional[str] = None,
    ) -> bool:
        """
        Update approval status.

        Args:
            approval_id: Approval to update
            status: New status
            approved_by: Who approved (if approved)
            rejection_reason: Reason (if rejected)

        Returns:
            True if updated
        """
        now = datetime.now().isoformat()

        updates = ["status = ?"]
        params: list = [status]

        if approved_by:
            updates.append("approved_by = ?")
            updates.append("approved_at = ?")
            params.extend([approved_by, now])

        if rejection_reason:
            updates.append("rejection_reason = ?")
            params.append(rejection_reason)

        params.append(approval_id)

        sql = f"UPDATE governance_approvals SET {', '.join(updates)} WHERE approval_id = ?"
        self._backend.execute_write(sql, tuple(params))

        logger.debug(f"Updated approval {approval_id} -> {status}")
        return True

    def get_approval(self, approval_id: str) -> Optional[ApprovalRecord]:
        """Get an approval by ID."""
        row = self._backend.fetch_one(
            """
            SELECT approval_id, title, description, risk_level, status,
                   requested_by, requested_at, changes_json, timeout_seconds,
                   approved_by, approved_at, rejection_reason, org_id,
                   workspace_id, metadata_json
            FROM governance_approvals
            WHERE approval_id = ?
            """,
            (approval_id,),
        )

        if not row:
            return None

        return self._row_to_approval(row)

    def list_approvals(
        self,
        status: Optional[str] = None,
        org_id: Optional[str] = None,
        risk_level: Optional[str] = None,
        limit: int = 100,
    ) -> list[ApprovalRecord]:
        """
        List approvals with filters.

        Args:
            status: Filter by status
            org_id: Filter by organization
            risk_level: Filter by risk level
            limit: Maximum results

        Returns:
            List of ApprovalRecord
        """
        query = """
            SELECT approval_id, title, description, risk_level, status,
                   requested_by, requested_at, changes_json, timeout_seconds,
                   approved_by, approved_at, rejection_reason, org_id,
                   workspace_id, metadata_json
            FROM governance_approvals
            WHERE 1=1
        """
        params: list = []

        if status:
            query += " AND status = ?"
            params.append(status)

        if org_id:
            query += " AND org_id = ?"
            params.append(org_id)

        if risk_level:
            query += " AND risk_level = ?"
            params.append(risk_level)

        query += " ORDER BY requested_at DESC LIMIT ?"
        params.append(limit)

        rows = self._backend.fetch_all(query, tuple(params))
        return [self._row_to_approval(row) for row in rows]

    def delete_approval(self, approval_id: str) -> bool:
        """Delete an approval record."""
        self._backend.execute_write(
            "DELETE FROM governance_approvals WHERE approval_id = ?",
            (approval_id,),
        )
        return True

    def _row_to_approval(self, row: tuple) -> ApprovalRecord:
        """Convert database row to ApprovalRecord."""
        def parse_dt(val: Any) -> datetime:
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                try:
                    return datetime.fromisoformat(val)
                except ValueError:
                    pass
            return datetime.now()

        return ApprovalRecord(
            approval_id=row[0],
            title=row[1],
            description=row[2] or "",
            risk_level=row[3],
            status=row[4],
            requested_by=row[5] or "",
            requested_at=parse_dt(row[6]),
            changes_json=row[7] or "[]",
            timeout_seconds=row[8] or 3600,
            approved_by=row[9],
            approved_at=parse_dt(row[10]) if row[10] else None,
            rejection_reason=row[11],
            org_id=row[12],
            workspace_id=row[13],
            metadata_json=row[14],
        )

    # =========================================================================
    # Verification History
    # =========================================================================

    def save_verification(
        self,
        verification_id: str,
        claim: str,
        context: str,
        result: dict,
        verified_by: str = "system",
        claim_type: Optional[str] = None,
        confidence: float = 0.0,
        proof_tree: Optional[list] = None,
        org_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> str:
        """
        Save a verification result.

        Args:
            verification_id: Unique verification ID
            claim: The claim that was verified
            context: Context for verification
            result: Verification result dict
            verified_by: Who/what verified
            claim_type: Type of claim
            confidence: Confidence score
            proof_tree: Optional proof tree
            org_id: Organization ID
            workspace_id: Workspace ID

        Returns:
            The verification_id
        """
        now = datetime.now().isoformat()

        if self.backend_type == "postgresql":
            sql = """
                INSERT INTO governance_verifications (
                    verification_id, claim, claim_type, context, result_json,
                    timestamp, verified_by, confidence, proof_tree_json,
                    org_id, workspace_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (verification_id) DO UPDATE SET
                    result_json = EXCLUDED.result_json,
                    confidence = EXCLUDED.confidence
            """
        else:
            sql = """
                INSERT OR REPLACE INTO governance_verifications (
                    verification_id, claim, claim_type, context, result_json,
                    timestamp, verified_by, confidence, proof_tree_json,
                    org_id, workspace_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

        self._backend.execute_write(
            sql,
            (
                verification_id,
                claim,
                claim_type,
                context,
                json.dumps(result),
                now,
                verified_by,
                confidence,
                json.dumps(proof_tree) if proof_tree else None,
                org_id,
                workspace_id,
            ),
        )

        logger.debug(f"Saved verification: {verification_id}")
        return verification_id

    def get_verification(self, verification_id: str) -> Optional[VerificationRecord]:
        """Get a verification by ID."""
        row = self._backend.fetch_one(
            """
            SELECT verification_id, claim, claim_type, context, result_json,
                   timestamp, verified_by, confidence, proof_tree_json,
                   org_id, workspace_id
            FROM governance_verifications
            WHERE verification_id = ?
            """,
            (verification_id,),
        )

        if not row:
            return None

        return self._row_to_verification(row)

    def list_verifications(
        self,
        org_id: Optional[str] = None,
        claim_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[VerificationRecord]:
        """List verifications with filters."""
        query = """
            SELECT verification_id, claim, claim_type, context, result_json,
                   timestamp, verified_by, confidence, proof_tree_json,
                   org_id, workspace_id
            FROM governance_verifications
            WHERE 1=1
        """
        params: list = []

        if org_id:
            query += " AND org_id = ?"
            params.append(org_id)

        if claim_type:
            query += " AND claim_type = ?"
            params.append(claim_type)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._backend.fetch_all(query, tuple(params))
        return [self._row_to_verification(row) for row in rows]

    def _row_to_verification(self, row: tuple) -> VerificationRecord:
        """Convert database row to VerificationRecord."""
        def parse_dt(val: Any) -> datetime:
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                try:
                    return datetime.fromisoformat(val)
                except ValueError:
                    pass
            return datetime.now()

        return VerificationRecord(
            verification_id=row[0],
            claim=row[1],
            claim_type=row[2],
            context=row[3] or "",
            result_json=row[4] or "{}",
            timestamp=parse_dt(row[5]),
            verified_by=row[6] or "system",
            confidence=row[7] or 0.0,
            proof_tree_json=row[8],
            org_id=row[9],
            workspace_id=row[10],
        )

    # =========================================================================
    # Decision Records
    # =========================================================================

    def save_decision(
        self,
        decision_id: str,
        debate_id: str,
        conclusion: str,
        consensus_reached: bool,
        confidence: float,
        evidence_chain: Optional[list] = None,
        vote_pivots: Optional[list] = None,
        belief_changes: Optional[list] = None,
        agents_involved: Optional[list] = None,
        org_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Save a decision record.

        Args:
            decision_id: Unique decision ID
            debate_id: Associated debate ID
            conclusion: Decision conclusion
            consensus_reached: Whether consensus was reached
            confidence: Confidence in decision
            evidence_chain: Chain of evidence
            vote_pivots: Voting pivots
            belief_changes: Belief changes
            agents_involved: Agents that participated
            org_id: Organization ID
            workspace_id: Workspace ID
            metadata: Additional metadata

        Returns:
            The decision_id
        """
        now = datetime.now().isoformat()

        if self.backend_type == "postgresql":
            sql = """
                INSERT INTO governance_decisions (
                    decision_id, debate_id, conclusion, consensus_reached,
                    confidence, timestamp, evidence_chain_json, vote_pivots_json,
                    belief_changes_json, agents_involved_json, org_id,
                    workspace_id, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (decision_id) DO UPDATE SET
                    conclusion = EXCLUDED.conclusion,
                    confidence = EXCLUDED.confidence
            """
        else:
            sql = """
                INSERT OR REPLACE INTO governance_decisions (
                    decision_id, debate_id, conclusion, consensus_reached,
                    confidence, timestamp, evidence_chain_json, vote_pivots_json,
                    belief_changes_json, agents_involved_json, org_id,
                    workspace_id, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

        self._backend.execute_write(
            sql,
            (
                decision_id,
                debate_id,
                conclusion,
                1 if consensus_reached else 0,
                confidence,
                now,
                json.dumps(evidence_chain or []),
                json.dumps(vote_pivots or []),
                json.dumps(belief_changes or []),
                json.dumps(agents_involved or []),
                org_id,
                workspace_id,
                json.dumps(metadata or {}),
            ),
        )

        logger.debug(f"Saved decision: {decision_id}")
        return decision_id

    def get_decision(self, decision_id: str) -> Optional[DecisionRecord]:
        """Get a decision by ID."""
        row = self._backend.fetch_one(
            """
            SELECT decision_id, debate_id, conclusion, consensus_reached,
                   confidence, timestamp, evidence_chain_json, vote_pivots_json,
                   belief_changes_json, agents_involved_json, org_id,
                   workspace_id, metadata_json
            FROM governance_decisions
            WHERE decision_id = ?
            """,
            (decision_id,),
        )

        if not row:
            return None

        return self._row_to_decision(row)

    def get_decisions_for_debate(self, debate_id: str) -> list[DecisionRecord]:
        """Get all decisions for a debate."""
        rows = self._backend.fetch_all(
            """
            SELECT decision_id, debate_id, conclusion, consensus_reached,
                   confidence, timestamp, evidence_chain_json, vote_pivots_json,
                   belief_changes_json, agents_involved_json, org_id,
                   workspace_id, metadata_json
            FROM governance_decisions
            WHERE debate_id = ?
            ORDER BY timestamp DESC
            """,
            (debate_id,),
        )
        return [self._row_to_decision(row) for row in rows]

    def list_decisions(
        self,
        org_id: Optional[str] = None,
        consensus_only: bool = False,
        limit: int = 100,
    ) -> list[DecisionRecord]:
        """List decisions with filters."""
        query = """
            SELECT decision_id, debate_id, conclusion, consensus_reached,
                   confidence, timestamp, evidence_chain_json, vote_pivots_json,
                   belief_changes_json, agents_involved_json, org_id,
                   workspace_id, metadata_json
            FROM governance_decisions
            WHERE 1=1
        """
        params: list = []

        if org_id:
            query += " AND org_id = ?"
            params.append(org_id)

        if consensus_only:
            query += " AND consensus_reached = 1"

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = self._backend.fetch_all(query, tuple(params))
        return [self._row_to_decision(row) for row in rows]

    def _row_to_decision(self, row: tuple) -> DecisionRecord:
        """Convert database row to DecisionRecord."""
        def parse_dt(val: Any) -> datetime:
            if isinstance(val, datetime):
                return val
            if isinstance(val, str):
                try:
                    return datetime.fromisoformat(val)
                except ValueError:
                    pass
            return datetime.now()

        return DecisionRecord(
            decision_id=row[0],
            debate_id=row[1],
            conclusion=row[2] or "",
            consensus_reached=bool(row[3]),
            confidence=row[4] or 0.0,
            timestamp=parse_dt(row[5]),
            evidence_chain_json=row[6] or "[]",
            vote_pivots_json=row[7] or "[]",
            belief_changes_json=row[8] or "[]",
            agents_involved_json=row[9] or "[]",
            org_id=row[10],
            workspace_id=row[11],
            metadata_json=row[12],
        )

    # =========================================================================
    # Cleanup
    # =========================================================================

    def cleanup_old_records(
        self,
        approvals_days: int = 30,
        verifications_days: int = 7,
        decisions_days: int = 90,
    ) -> dict:
        """
        Clean up old records.

        Args:
            approvals_days: Days to keep completed approvals
            verifications_days: Days to keep verifications
            decisions_days: Days to keep decisions

        Returns:
            Counts of deleted records
        """
        counts = {}

        # Clean approvals (only completed/rejected/expired)
        result = self._backend.fetch_one(
            """
            SELECT COUNT(*) FROM governance_approvals
            WHERE status IN ('approved', 'rejected', 'expired', 'cancelled')
            AND datetime(requested_at) < datetime('now', ? || ' days')
            """,
            (f"-{approvals_days}",),
        )
        counts["approvals"] = result[0] if result else 0

        if counts["approvals"] > 0:
            self._backend.execute_write(
                """
                DELETE FROM governance_approvals
                WHERE status IN ('approved', 'rejected', 'expired', 'cancelled')
                AND datetime(requested_at) < datetime('now', ? || ' days')
                """,
                (f"-{approvals_days}",),
            )

        # Clean verifications
        result = self._backend.fetch_one(
            """
            SELECT COUNT(*) FROM governance_verifications
            WHERE datetime(timestamp) < datetime('now', ? || ' days')
            """,
            (f"-{verifications_days}",),
        )
        counts["verifications"] = result[0] if result else 0

        if counts["verifications"] > 0:
            self._backend.execute_write(
                """
                DELETE FROM governance_verifications
                WHERE datetime(timestamp) < datetime('now', ? || ' days')
                """,
                (f"-{verifications_days}",),
            )

        logger.info(f"Cleaned up governance records: {counts}")
        return counts

    def close(self) -> None:
        """Close database connection."""
        self._backend.close()


# Module-level singleton
_default_store: Optional[GovernanceStore] = None


def get_governance_store(
    db_path: str = "aragora_governance.db",
    backend: Optional[str] = None,
    database_url: Optional[str] = None,
) -> GovernanceStore:
    """Get or create the default GovernanceStore instance."""
    global _default_store
    if _default_store is None:
        _default_store = GovernanceStore(
            db_path=db_path,
            backend=backend,
            database_url=database_url,
        )
    return _default_store


def reset_governance_store() -> None:
    """Reset the default store instance (for testing)."""
    global _default_store
    if _default_store is not None:
        _default_store.close()
        _default_store = None


__all__ = [
    "GovernanceStore",
    "ApprovalRecord",
    "VerificationRecord",
    "DecisionRecord",
    "get_governance_store",
    "reset_governance_store",
]
