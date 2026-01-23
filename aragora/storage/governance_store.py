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

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from asyncpg import Pool

from aragora.storage.backends import (
    POSTGRESQL_AVAILABLE,
    DatabaseBackend,
    PostgreSQLBackend,
    SQLiteBackend,
)

logger = logging.getLogger(__name__)


def _record_governance_verification(verification_type: str, result: str) -> None:
    """Record governance verification metric if available."""
    try:
        from aragora.observability.metrics import record_governance_verification  # type: ignore[attr-defined]

        record_governance_verification(verification_type, result)
    except ImportError:
        pass


def _record_governance_decision(decision_type: str, outcome: str) -> None:
    """Record governance decision metric if available."""
    try:
        from aragora.observability.metrics import record_governance_decision  # type: ignore[attr-defined]

        record_governance_decision(decision_type, outcome)
    except ImportError:
        pass


def _record_governance_approval(approval_type: str, status: str) -> None:
    """Record governance approval metric if available."""
    try:
        from aragora.observability.metrics import record_governance_approval  # type: ignore[attr-defined]

        record_governance_approval(approval_type, status)
    except ImportError:
        pass


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
            "requested_at": self.requested_at.isoformat()
            if isinstance(self.requested_at, datetime)
            else self.requested_at,
            "changes": json.loads(self.changes_json) if self.changes_json else [],
            "timeout_seconds": self.timeout_seconds,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at.isoformat()
            if isinstance(self.approved_at, datetime)
            else self.approved_at,
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
            "timestamp": self.timestamp.isoformat()
            if isinstance(self.timestamp, datetime)
            else self.timestamp,
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
            "timestamp": self.timestamp.isoformat()
            if isinstance(self.timestamp, datetime)
            else self.timestamp,
            "evidence_chain": json.loads(self.evidence_chain_json)
            if self.evidence_chain_json
            else [],
            "vote_pivots": json.loads(self.vote_pivots_json) if self.vote_pivots_json else [],
            "belief_changes": json.loads(self.belief_changes_json)
            if self.belief_changes_json
            else [],
            "agents_involved": json.loads(self.agents_involved_json)
            if self.agents_involved_json
            else [],
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

        # Record metrics
        approval_type = "nomic" if "nomic" in title.lower() else "change"
        _record_governance_approval(approval_type, status)

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

        # Record metrics
        verification_type = claim_type or "formal"
        verification_result = "valid" if result.get("valid", False) else "invalid"
        _record_governance_verification(verification_type, verification_result)

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

        # Record metrics
        decision_type = "auto"  # Could be expanded based on metadata
        outcome = "approved" if consensus_reached else "rejected"
        _record_governance_decision(decision_type, outcome)

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


class PostgresGovernanceStore:
    """
    PostgreSQL-backed governance store.

    Async implementation for production multi-instance deployments
    with horizontal scaling and concurrent access to governance records.
    """

    SCHEMA_NAME = "governance"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        CREATE TABLE IF NOT EXISTS governance_approvals (
            approval_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT,
            risk_level TEXT NOT NULL,
            status TEXT NOT NULL,
            requested_by TEXT,
            requested_at TIMESTAMPTZ DEFAULT NOW(),
            changes_json JSONB,
            timeout_seconds INTEGER DEFAULT 3600,
            approved_by TEXT,
            approved_at TIMESTAMPTZ,
            rejection_reason TEXT,
            org_id TEXT,
            workspace_id TEXT,
            metadata_json JSONB
        );
        CREATE INDEX IF NOT EXISTS idx_approvals_status ON governance_approvals(status);
        CREATE INDEX IF NOT EXISTS idx_approvals_org ON governance_approvals(org_id);
        CREATE INDEX IF NOT EXISTS idx_approvals_requested ON governance_approvals(requested_at DESC);

        CREATE TABLE IF NOT EXISTS governance_verifications (
            verification_id TEXT PRIMARY KEY,
            claim TEXT NOT NULL,
            claim_type TEXT,
            context TEXT,
            result_json JSONB NOT NULL,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            verified_by TEXT,
            confidence REAL DEFAULT 0,
            proof_tree_json JSONB,
            org_id TEXT,
            workspace_id TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_verifications_timestamp ON governance_verifications(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_verifications_org ON governance_verifications(org_id);

        CREATE TABLE IF NOT EXISTS governance_decisions (
            decision_id TEXT PRIMARY KEY,
            debate_id TEXT NOT NULL,
            conclusion TEXT,
            consensus_reached BOOLEAN,
            confidence REAL,
            timestamp TIMESTAMPTZ DEFAULT NOW(),
            evidence_chain_json JSONB,
            vote_pivots_json JSONB,
            belief_changes_json JSONB,
            agents_involved_json JSONB,
            org_id TEXT,
            workspace_id TEXT,
            metadata_json JSONB
        );
        CREATE INDEX IF NOT EXISTS idx_decisions_debate ON governance_decisions(debate_id);
        CREATE INDEX IF NOT EXISTS idx_decisions_org ON governance_decisions(org_id);
        CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON governance_decisions(timestamp DESC);
    """

    def __init__(self, pool: "Pool"):
        self._pool = pool
        self._initialized = False
        logger.info("PostgresGovernanceStore initialized")

    async def initialize(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        async with self._pool.acquire() as conn:
            await conn.execute(self.INITIAL_SCHEMA)

        self._initialized = True
        logger.debug(f"[{self.SCHEMA_NAME}] Schema initialized")

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
        """Save a new approval request (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(
            self.save_approval_async(
                approval_id,
                title,
                description,
                risk_level,
                status,
                requested_by,
                changes,
                timeout_seconds,
                org_id,
                workspace_id,
                metadata,
            )
        )

    async def save_approval_async(
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
        """Save a new approval request asynchronously."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO governance_approvals (
                    approval_id, title, description, risk_level, status,
                    requested_by, requested_at, changes_json, timeout_seconds,
                    org_id, workspace_id, metadata_json
                )
                VALUES ($1, $2, $3, $4, $5, $6, NOW(), $7, $8, $9, $10, $11)
                ON CONFLICT (approval_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    metadata_json = EXCLUDED.metadata_json""",
                approval_id,
                title,
                description,
                risk_level,
                status,
                requested_by,
                json.dumps(changes),
                timeout_seconds,
                org_id,
                workspace_id,
                json.dumps(metadata or {}),
            )

        # Record metrics
        approval_type = "nomic" if "nomic" in title.lower() else "change"
        _record_governance_approval(approval_type, status)

        logger.debug(f"Saved approval: {approval_id}")
        return approval_id

    def update_approval_status(
        self,
        approval_id: str,
        status: str,
        approved_by: Optional[str] = None,
        rejection_reason: Optional[str] = None,
    ) -> bool:
        """Update approval status (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(
            self.update_approval_status_async(approval_id, status, approved_by, rejection_reason)
        )

    async def update_approval_status_async(
        self,
        approval_id: str,
        status: str,
        approved_by: Optional[str] = None,
        rejection_reason: Optional[str] = None,
    ) -> bool:
        """Update approval status asynchronously."""
        updates = ["status = $1"]
        params: list = [status]
        param_idx = 2

        if approved_by:
            updates.append(f"approved_by = ${param_idx}")
            params.append(approved_by)
            param_idx += 1
            updates.append("approved_at = NOW()")

        if rejection_reason:
            updates.append(f"rejection_reason = ${param_idx}")
            params.append(rejection_reason)
            param_idx += 1

        params.append(approval_id)

        async with self._pool.acquire() as conn:
            await conn.execute(
                f"UPDATE governance_approvals SET {', '.join(updates)} WHERE approval_id = ${param_idx}",
                *params,
            )

        logger.debug(f"Updated approval {approval_id} -> {status}")
        return True

    def get_approval(self, approval_id: str) -> Optional[ApprovalRecord]:
        """Get an approval by ID (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(self.get_approval_async(approval_id))

    async def get_approval_async(self, approval_id: str) -> Optional[ApprovalRecord]:
        """Get an approval by ID asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT approval_id, title, description, risk_level, status,
                          requested_by,
                          EXTRACT(EPOCH FROM requested_at) as requested_at,
                          changes_json, timeout_seconds, approved_by,
                          EXTRACT(EPOCH FROM approved_at) as approved_at,
                          rejection_reason, org_id, workspace_id, metadata_json
                   FROM governance_approvals
                   WHERE approval_id = $1""",
                approval_id,
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
        """List approvals with filters (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(
            self.list_approvals_async(status, org_id, risk_level, limit)
        )

    async def list_approvals_async(
        self,
        status: Optional[str] = None,
        org_id: Optional[str] = None,
        risk_level: Optional[str] = None,
        limit: int = 100,
    ) -> list[ApprovalRecord]:
        """List approvals asynchronously."""
        query = """SELECT approval_id, title, description, risk_level, status,
                          requested_by,
                          EXTRACT(EPOCH FROM requested_at) as requested_at,
                          changes_json, timeout_seconds, approved_by,
                          EXTRACT(EPOCH FROM approved_at) as approved_at,
                          rejection_reason, org_id, workspace_id, metadata_json
                   FROM governance_approvals
                   WHERE 1=1"""
        params: list = []
        param_idx = 1

        if status:
            query += f" AND status = ${param_idx}"
            params.append(status)
            param_idx += 1

        if org_id:
            query += f" AND org_id = ${param_idx}"
            params.append(org_id)
            param_idx += 1

        if risk_level:
            query += f" AND risk_level = ${param_idx}"
            params.append(risk_level)
            param_idx += 1

        query += f" ORDER BY requested_at DESC LIMIT ${param_idx}"
        params.append(limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._row_to_approval(row) for row in rows]

    def delete_approval(self, approval_id: str) -> bool:
        """Delete an approval record (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(self.delete_approval_async(approval_id))

    async def delete_approval_async(self, approval_id: str) -> bool:
        """Delete an approval record asynchronously."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM governance_approvals WHERE approval_id = $1",
                approval_id,
            )
        return True

    def _row_to_approval(self, row: Any) -> ApprovalRecord:
        """Convert database row to ApprovalRecord."""

        def parse_dt(val: Any) -> datetime:
            if val is None:
                return datetime.now()
            if isinstance(val, (int, float)):
                return datetime.fromtimestamp(val)
            if isinstance(val, datetime):
                return val
            return datetime.now()

        return ApprovalRecord(
            approval_id=row["approval_id"],
            title=row["title"],
            description=row["description"] or "",
            risk_level=row["risk_level"],
            status=row["status"],
            requested_by=row["requested_by"] or "",
            requested_at=parse_dt(row["requested_at"]),
            changes_json=row["changes_json"]
            if isinstance(row["changes_json"], str)
            else json.dumps(row["changes_json"] or []),
            timeout_seconds=row["timeout_seconds"] or 3600,
            approved_by=row["approved_by"],
            approved_at=parse_dt(row["approved_at"]) if row["approved_at"] else None,
            rejection_reason=row["rejection_reason"],
            org_id=row["org_id"],
            workspace_id=row["workspace_id"],
            metadata_json=row["metadata_json"]
            if isinstance(row["metadata_json"], str)
            else json.dumps(row["metadata_json"] or {}),
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
        """Save a verification result (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(
            self.save_verification_async(
                verification_id,
                claim,
                context,
                result,
                verified_by,
                claim_type,
                confidence,
                proof_tree,
                org_id,
                workspace_id,
            )
        )

    async def save_verification_async(
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
        """Save a verification result asynchronously."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO governance_verifications (
                    verification_id, claim, claim_type, context, result_json,
                    timestamp, verified_by, confidence, proof_tree_json,
                    org_id, workspace_id
                )
                VALUES ($1, $2, $3, $4, $5, NOW(), $6, $7, $8, $9, $10)
                ON CONFLICT (verification_id) DO UPDATE SET
                    result_json = EXCLUDED.result_json,
                    confidence = EXCLUDED.confidence""",
                verification_id,
                claim,
                claim_type,
                context,
                json.dumps(result),
                verified_by,
                confidence,
                json.dumps(proof_tree) if proof_tree else None,
                org_id,
                workspace_id,
            )

        # Record metrics
        verification_type = claim_type or "formal"
        verification_result = "valid" if result.get("valid", False) else "invalid"
        _record_governance_verification(verification_type, verification_result)

        logger.debug(f"Saved verification: {verification_id}")
        return verification_id

    def get_verification(self, verification_id: str) -> Optional[VerificationRecord]:
        """Get a verification by ID (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_verification_async(verification_id)
        )

    async def get_verification_async(self, verification_id: str) -> Optional[VerificationRecord]:
        """Get a verification by ID asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT verification_id, claim, claim_type, context, result_json,
                          EXTRACT(EPOCH FROM timestamp) as timestamp,
                          verified_by, confidence, proof_tree_json,
                          org_id, workspace_id
                   FROM governance_verifications
                   WHERE verification_id = $1""",
                verification_id,
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
        """List verifications with filters (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(
            self.list_verifications_async(org_id, claim_type, limit)
        )

    async def list_verifications_async(
        self,
        org_id: Optional[str] = None,
        claim_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[VerificationRecord]:
        """List verifications asynchronously."""
        query = """SELECT verification_id, claim, claim_type, context, result_json,
                          EXTRACT(EPOCH FROM timestamp) as timestamp,
                          verified_by, confidence, proof_tree_json,
                          org_id, workspace_id
                   FROM governance_verifications
                   WHERE 1=1"""
        params: list = []
        param_idx = 1

        if org_id:
            query += f" AND org_id = ${param_idx}"
            params.append(org_id)
            param_idx += 1

        if claim_type:
            query += f" AND claim_type = ${param_idx}"
            params.append(claim_type)
            param_idx += 1

        query += f" ORDER BY timestamp DESC LIMIT ${param_idx}"
        params.append(limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._row_to_verification(row) for row in rows]

    def _row_to_verification(self, row: Any) -> VerificationRecord:
        """Convert database row to VerificationRecord."""

        def parse_dt(val: Any) -> datetime:
            if val is None:
                return datetime.now()
            if isinstance(val, (int, float)):
                return datetime.fromtimestamp(val)
            if isinstance(val, datetime):
                return val
            return datetime.now()

        return VerificationRecord(
            verification_id=row["verification_id"],
            claim=row["claim"],
            claim_type=row["claim_type"],
            context=row["context"] or "",
            result_json=row["result_json"]
            if isinstance(row["result_json"], str)
            else json.dumps(row["result_json"] or {}),
            timestamp=parse_dt(row["timestamp"]),
            verified_by=row["verified_by"] or "system",
            confidence=row["confidence"] or 0.0,
            proof_tree_json=row["proof_tree_json"]
            if isinstance(row["proof_tree_json"], str)
            else json.dumps(row["proof_tree_json"])
            if row["proof_tree_json"]
            else None,
            org_id=row["org_id"],
            workspace_id=row["workspace_id"],
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
        """Save a decision record (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(
            self.save_decision_async(
                decision_id,
                debate_id,
                conclusion,
                consensus_reached,
                confidence,
                evidence_chain,
                vote_pivots,
                belief_changes,
                agents_involved,
                org_id,
                workspace_id,
                metadata,
            )
        )

    async def save_decision_async(
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
        """Save a decision record asynchronously."""
        async with self._pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO governance_decisions (
                    decision_id, debate_id, conclusion, consensus_reached,
                    confidence, timestamp, evidence_chain_json, vote_pivots_json,
                    belief_changes_json, agents_involved_json, org_id,
                    workspace_id, metadata_json
                )
                VALUES ($1, $2, $3, $4, $5, NOW(), $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (decision_id) DO UPDATE SET
                    conclusion = EXCLUDED.conclusion,
                    confidence = EXCLUDED.confidence""",
                decision_id,
                debate_id,
                conclusion,
                consensus_reached,
                confidence,
                json.dumps(evidence_chain or []),
                json.dumps(vote_pivots or []),
                json.dumps(belief_changes or []),
                json.dumps(agents_involved or []),
                org_id,
                workspace_id,
                json.dumps(metadata or {}),
            )

        # Record metrics
        decision_type = "auto"
        outcome = "approved" if consensus_reached else "rejected"
        _record_governance_decision(decision_type, outcome)

        logger.debug(f"Saved decision: {decision_id}")
        return decision_id

    def get_decision(self, decision_id: str) -> Optional[DecisionRecord]:
        """Get a decision by ID (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(self.get_decision_async(decision_id))

    async def get_decision_async(self, decision_id: str) -> Optional[DecisionRecord]:
        """Get a decision by ID asynchronously."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT decision_id, debate_id, conclusion, consensus_reached,
                          confidence, EXTRACT(EPOCH FROM timestamp) as timestamp,
                          evidence_chain_json, vote_pivots_json,
                          belief_changes_json, agents_involved_json, org_id,
                          workspace_id, metadata_json
                   FROM governance_decisions
                   WHERE decision_id = $1""",
                decision_id,
            )

        if not row:
            return None

        return self._row_to_decision(row)

    def get_decisions_for_debate(self, debate_id: str) -> list[DecisionRecord]:
        """Get all decisions for a debate (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(
            self.get_decisions_for_debate_async(debate_id)
        )

    async def get_decisions_for_debate_async(self, debate_id: str) -> list[DecisionRecord]:
        """Get all decisions for a debate asynchronously."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """SELECT decision_id, debate_id, conclusion, consensus_reached,
                          confidence, EXTRACT(EPOCH FROM timestamp) as timestamp,
                          evidence_chain_json, vote_pivots_json,
                          belief_changes_json, agents_involved_json, org_id,
                          workspace_id, metadata_json
                   FROM governance_decisions
                   WHERE debate_id = $1
                   ORDER BY timestamp DESC""",
                debate_id,
            )
            return [self._row_to_decision(row) for row in rows]

    def list_decisions(
        self,
        org_id: Optional[str] = None,
        consensus_only: bool = False,
        limit: int = 100,
    ) -> list[DecisionRecord]:
        """List decisions with filters (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(
            self.list_decisions_async(org_id, consensus_only, limit)
        )

    async def list_decisions_async(
        self,
        org_id: Optional[str] = None,
        consensus_only: bool = False,
        limit: int = 100,
    ) -> list[DecisionRecord]:
        """List decisions asynchronously."""
        query = """SELECT decision_id, debate_id, conclusion, consensus_reached,
                          confidence, EXTRACT(EPOCH FROM timestamp) as timestamp,
                          evidence_chain_json, vote_pivots_json,
                          belief_changes_json, agents_involved_json, org_id,
                          workspace_id, metadata_json
                   FROM governance_decisions
                   WHERE 1=1"""
        params: list = []
        param_idx = 1

        if org_id:
            query += f" AND org_id = ${param_idx}"
            params.append(org_id)
            param_idx += 1

        if consensus_only:
            query += " AND consensus_reached = TRUE"

        query += f" ORDER BY timestamp DESC LIMIT ${param_idx}"
        params.append(limit)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            return [self._row_to_decision(row) for row in rows]

    def _row_to_decision(self, row: Any) -> DecisionRecord:
        """Convert database row to DecisionRecord."""

        def parse_dt(val: Any) -> datetime:
            if val is None:
                return datetime.now()
            if isinstance(val, (int, float)):
                return datetime.fromtimestamp(val)
            if isinstance(val, datetime):
                return val
            return datetime.now()

        def to_json_str(val: Any) -> str:
            if val is None:
                return "[]"
            if isinstance(val, str):
                return val
            return json.dumps(val)

        return DecisionRecord(
            decision_id=row["decision_id"],
            debate_id=row["debate_id"],
            conclusion=row["conclusion"] or "",
            consensus_reached=bool(row["consensus_reached"]),
            confidence=row["confidence"] or 0.0,
            timestamp=parse_dt(row["timestamp"]),
            evidence_chain_json=to_json_str(row["evidence_chain_json"]),
            vote_pivots_json=to_json_str(row["vote_pivots_json"]),
            belief_changes_json=to_json_str(row["belief_changes_json"]),
            agents_involved_json=to_json_str(row["agents_involved_json"]),
            org_id=row["org_id"],
            workspace_id=row["workspace_id"],
            metadata_json=row["metadata_json"]
            if isinstance(row["metadata_json"], str)
            else json.dumps(row["metadata_json"] or {}),
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
        """Clean up old records (sync wrapper for async)."""
        return asyncio.get_event_loop().run_until_complete(
            self.cleanup_old_records_async(approvals_days, verifications_days, decisions_days)
        )

    async def cleanup_old_records_async(
        self,
        approvals_days: int = 30,
        verifications_days: int = 7,
        decisions_days: int = 90,
    ) -> dict:
        """Clean up old records asynchronously.

        Uses explicit transaction for atomic cleanup across multiple tables.
        """
        counts = {}

        async with self._pool.acquire() as conn:
            # Use explicit transaction for atomic multi-table cleanup
            async with conn.transaction():
                # Count and delete approvals
                result = await conn.fetchrow(
                    """SELECT COUNT(*) as count FROM governance_approvals
                       WHERE status IN ('approved', 'rejected', 'expired', 'cancelled')
                       AND requested_at < NOW() - INTERVAL '1 day' * $1""",
                    approvals_days,
                )
                counts["approvals"] = result["count"] if result else 0

                if counts["approvals"] > 0:
                    await conn.execute(
                        """DELETE FROM governance_approvals
                           WHERE status IN ('approved', 'rejected', 'expired', 'cancelled')
                           AND requested_at < NOW() - INTERVAL '1 day' * $1""",
                        approvals_days,
                    )

                # Count and delete verifications
                result = await conn.fetchrow(
                    """SELECT COUNT(*) as count FROM governance_verifications
                       WHERE timestamp < NOW() - INTERVAL '1 day' * $1""",
                    verifications_days,
                )
                counts["verifications"] = result["count"] if result else 0

                if counts["verifications"] > 0:
                    await conn.execute(
                        """DELETE FROM governance_verifications
                           WHERE timestamp < NOW() - INTERVAL '1 day' * $1""",
                        verifications_days,
                    )

        logger.info(f"Cleaned up governance records: {counts}")
        return counts

    def close(self) -> None:
        """Close is a no-op for pool-based stores (pool managed externally)."""
        pass


# Module-level singleton
_default_store: Optional[GovernanceStore] = None
_postgres_store: Optional[PostgresGovernanceStore] = None


def get_governance_store(
    db_path: str = "aragora_governance.db",
    backend: Optional[str] = None,
    database_url: Optional[str] = None,
) -> GovernanceStore | PostgresGovernanceStore:
    """
    Get or create the default GovernanceStore instance.

    Uses environment variables to configure:
    - ARAGORA_DB_BACKEND: Global database backend ("sqlite", "postgres", or "postgresql")
    - ARAGORA_GOVERNANCE_STORE_BACKEND: Store-specific backend override
    - DATABASE_URL or ARAGORA_DATABASE_URL: PostgreSQL connection string

    Returns:
        Configured GovernanceStore or PostgresGovernanceStore instance
    """
    global _default_store, _postgres_store

    # Check store-specific backend first, then global database backend
    backend_type = os.environ.get("ARAGORA_GOVERNANCE_STORE_BACKEND")
    if not backend_type and backend is None:
        # Fall back to global database backend setting
        backend_type = os.environ.get("ARAGORA_DB_BACKEND", "sqlite").lower()
    elif backend:
        backend_type = backend.lower()

    if backend_type in ("postgres", "postgresql"):
        if _postgres_store is not None:
            return _postgres_store

        logger.info("Using PostgreSQL governance store")
        try:
            from aragora.storage.postgres_store import get_postgres_pool

            # Initialize PostgreSQL store with connection pool
            pool = asyncio.get_event_loop().run_until_complete(get_postgres_pool())
            store = PostgresGovernanceStore(pool)
            asyncio.get_event_loop().run_until_complete(store.initialize())
            _postgres_store = store
            return _postgres_store
        except Exception as e:
            logger.warning(f"PostgreSQL not available, falling back to SQLite: {e}")

            # Enforce distributed storage in production for RBAC policies
            from aragora.storage.production_guards import (
                require_distributed_store,
                StorageMode,
            )

            require_distributed_store(
                "governance_store",
                StorageMode.SQLITE,
                f"RBAC/governance policies must use distributed storage in production. "
                f"PostgreSQL unavailable: {e}",
            )
            # Fall through to SQLite

    if _default_store is None:
        # Enforce distributed storage in production for RBAC policies
        from aragora.storage.production_guards import (
            require_distributed_store,
            StorageMode,
        )

        require_distributed_store(
            "governance_store",
            StorageMode.SQLITE,
            "RBAC/governance policies must use distributed storage in production. "
            "Configure ARAGORA_DB_BACKEND=postgres.",
        )
        _default_store = GovernanceStore(
            db_path=db_path,
            backend=backend,
            database_url=database_url,
        )
    return _default_store


def reset_governance_store() -> None:
    """Reset the default store instance (for testing)."""
    global _default_store, _postgres_store
    if _default_store is not None:
        _default_store.close()
        _default_store = None
    if _postgres_store is not None:
        _postgres_store.close()
        _postgres_store = None


__all__ = [
    "GovernanceStore",
    "PostgresGovernanceStore",
    "ApprovalRecord",
    "VerificationRecord",
    "DecisionRecord",
    "get_governance_store",
    "reset_governance_store",
]
