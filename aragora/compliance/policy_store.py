"""
Policy Store for managing compliance policies and violations.

Provides persistent storage for:
- Policy configurations (enabled frameworks, rules, thresholds)
- Compliance violations (detected issues with status tracking)
- Audit trail of policy changes
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from aragora.config.legacy import get_db_path
from aragora.storage.base_store import SQLiteStore


def _get_default_db_path() -> Path:
    """Get the default database path for policy store."""
    return get_db_path("compliance/policy_store.db")


@dataclass
class PolicyRule:
    """A rule within a policy."""

    rule_id: str
    name: str
    description: str
    severity: str  # critical, high, medium, low
    enabled: bool = True
    custom_threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "severity": self.severity,
            "enabled": self.enabled,
            "custom_threshold": self.custom_threshold,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PolicyRule":
        return cls(
            rule_id=data["rule_id"],
            name=data["name"],
            description=data.get("description", ""),
            severity=data.get("severity", "medium"),
            enabled=data.get("enabled", True),
            custom_threshold=data.get("custom_threshold"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Policy:
    """A compliance policy configuration."""

    id: str
    name: str
    description: str
    framework_id: str  # Maps to ComplianceFramework from framework.py
    workspace_id: str
    vertical_id: str
    level: str = "recommended"  # mandatory, recommended, optional
    enabled: bool = True
    rules: List[PolicyRule] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "framework_id": self.framework_id,
            "workspace_id": self.workspace_id,
            "vertical_id": self.vertical_id,
            "level": self.level,
            "enabled": self.enabled,
            "rules": [r.to_dict() for r in self.rules],
            "rules_count": len(self.rules),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "created_by": self.created_by,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Policy":
        rules = [PolicyRule.from_dict(r) for r in data.get("rules", [])]
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.utcnow()
        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif updated_at is None:
            updated_at = datetime.utcnow()

        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            framework_id=data["framework_id"],
            workspace_id=data.get("workspace_id", "default"),
            vertical_id=data.get("vertical_id", ""),
            level=data.get("level", "recommended"),
            enabled=data.get("enabled", True),
            rules=rules,
            created_at=created_at,
            updated_at=updated_at,
            created_by=data.get("created_by"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Violation:
    """A compliance violation."""

    id: str
    policy_id: str
    rule_id: str
    rule_name: str
    framework_id: str
    vertical_id: str
    workspace_id: str
    severity: str  # critical, high, medium, low
    status: str  # open, investigating, resolved, false_positive
    description: str
    source: str  # File/location where violation was detected
    detected_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolution_notes: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "policy_id": self.policy_id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "framework_id": self.framework_id,
            "vertical_id": self.vertical_id,
            "workspace_id": self.workspace_id,
            "severity": self.severity,
            "status": self.status,
            "description": self.description,
            "source": self.source,
            "detected_at": self.detected_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "resolution_notes": self.resolution_notes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Violation":
        detected_at = data.get("detected_at")
        if isinstance(detected_at, str):
            detected_at = datetime.fromisoformat(detected_at)
        elif detected_at is None:
            detected_at = datetime.utcnow()

        resolved_at = data.get("resolved_at")
        if isinstance(resolved_at, str):
            resolved_at = datetime.fromisoformat(resolved_at)

        return cls(
            id=data["id"],
            policy_id=data.get("policy_id", ""),
            rule_id=data["rule_id"],
            rule_name=data.get("rule_name", ""),
            framework_id=data["framework_id"],
            vertical_id=data.get("vertical_id", ""),
            workspace_id=data.get("workspace_id", "default"),
            severity=data.get("severity", "medium"),
            status=data.get("status", "open"),
            description=data.get("description", ""),
            source=data.get("source", ""),
            detected_at=detected_at,
            resolved_at=resolved_at,
            resolved_by=data.get("resolved_by"),
            resolution_notes=data.get("resolution_notes"),
            metadata=data.get("metadata", {}),
        )


class PolicyStore(SQLiteStore):
    """
    SQLite-backed store for compliance policies and violations.

    Provides:
    - CRUD operations for policies
    - Violation tracking with status management
    - Filtering by workspace, vertical, framework, and status
    - Audit trail for policy changes
    """

    SCHEMA_NAME = "policy_store"
    SCHEMA_VERSION = 1

    INITIAL_SCHEMA = """
        -- Policies table
        CREATE TABLE IF NOT EXISTS policies (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            framework_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL DEFAULT 'default',
            vertical_id TEXT NOT NULL,
            level TEXT DEFAULT 'recommended',
            enabled INTEGER DEFAULT 1,
            rules_json TEXT DEFAULT '[]',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
            created_by TEXT,
            metadata_json TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_policies_workspace ON policies(workspace_id);
        CREATE INDEX IF NOT EXISTS idx_policies_framework ON policies(framework_id);
        CREATE INDEX IF NOT EXISTS idx_policies_vertical ON policies(vertical_id);

        -- Violations table
        CREATE TABLE IF NOT EXISTS violations (
            id TEXT PRIMARY KEY,
            policy_id TEXT,
            rule_id TEXT NOT NULL,
            rule_name TEXT,
            framework_id TEXT NOT NULL,
            vertical_id TEXT NOT NULL,
            workspace_id TEXT NOT NULL DEFAULT 'default',
            severity TEXT DEFAULT 'medium',
            status TEXT DEFAULT 'open',
            description TEXT,
            source TEXT,
            detected_at TEXT DEFAULT CURRENT_TIMESTAMP,
            resolved_at TEXT,
            resolved_by TEXT,
            resolution_notes TEXT,
            metadata_json TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_violations_workspace ON violations(workspace_id);
        CREATE INDEX IF NOT EXISTS idx_violations_framework ON violations(framework_id);
        CREATE INDEX IF NOT EXISTS idx_violations_status ON violations(status);
        CREATE INDEX IF NOT EXISTS idx_violations_severity ON violations(severity);
        CREATE INDEX IF NOT EXISTS idx_violations_policy ON violations(policy_id);

        -- Policy audit log
        CREATE TABLE IF NOT EXISTS policy_audit (
            id TEXT PRIMARY KEY,
            policy_id TEXT NOT NULL,
            action TEXT NOT NULL,
            changed_by TEXT,
            changed_at TEXT DEFAULT CURRENT_TIMESTAMP,
            old_value TEXT,
            new_value TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_audit_policy ON policy_audit(policy_id);
    """

    def __init__(self, db_path: Optional[Path] = None):
        super().__init__(db_path or _get_default_db_path())

    # =========================================================================
    # Policy CRUD
    # =========================================================================

    def create_policy(self, policy: Policy) -> Policy:
        """Create a new policy."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO policies (
                    id, name, description, framework_id, workspace_id, vertical_id,
                    level, enabled, rules_json, created_at, updated_at, created_by, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    policy.id,
                    policy.name,
                    policy.description,
                    policy.framework_id,
                    policy.workspace_id,
                    policy.vertical_id,
                    policy.level,
                    1 if policy.enabled else 0,
                    json.dumps([r.to_dict() for r in policy.rules]),
                    policy.created_at.isoformat(),
                    policy.updated_at.isoformat(),
                    policy.created_by,
                    json.dumps(policy.metadata),
                ),
            )
            self._log_audit(conn, policy.id, "create", None, policy.to_dict(), policy.created_by)
        return policy

    def get_policy(self, policy_id: str) -> Optional[Policy]:
        """Get a policy by ID."""
        row = self.fetch_one("SELECT * FROM policies WHERE id = ?", (policy_id,))
        if not row:
            return None
        return self._row_to_policy(row)

    def list_policies(
        self,
        workspace_id: Optional[str] = None,
        vertical_id: Optional[str] = None,
        framework_id: Optional[str] = None,
        enabled_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Policy]:
        """List policies with optional filters."""
        conditions = []
        params: List[Any] = []

        if workspace_id:
            conditions.append("workspace_id = ?")
            params.append(workspace_id)
        if vertical_id:
            conditions.append("vertical_id = ?")
            params.append(vertical_id)
        if framework_id:
            conditions.append("framework_id = ?")
            params.append(framework_id)
        if enabled_only:
            conditions.append("enabled = 1")

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM policies {where} ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self.fetch_all(sql, tuple(params))
        return [self._row_to_policy(row) for row in rows]

    def update_policy(
        self,
        policy_id: str,
        updates: Dict[str, Any],
        changed_by: Optional[str] = None,
    ) -> Optional[Policy]:
        """Update a policy."""
        current = self.get_policy(policy_id)
        if not current:
            return None

        old_value = current.to_dict()

        # Apply updates
        if "name" in updates:
            current.name = updates["name"]
        if "description" in updates:
            current.description = updates["description"]
        if "level" in updates:
            current.level = updates["level"]
        if "enabled" in updates:
            current.enabled = updates["enabled"]
        if "rules" in updates:
            current.rules = [PolicyRule.from_dict(r) for r in updates["rules"]]
        if "metadata" in updates:
            current.metadata.update(updates["metadata"])

        current.updated_at = datetime.utcnow()

        with self.connection() as conn:
            conn.execute(
                """
                UPDATE policies SET
                    name = ?, description = ?, level = ?, enabled = ?,
                    rules_json = ?, updated_at = ?, metadata_json = ?
                WHERE id = ?
                """,
                (
                    current.name,
                    current.description,
                    current.level,
                    1 if current.enabled else 0,
                    json.dumps([r.to_dict() for r in current.rules]),
                    current.updated_at.isoformat(),
                    json.dumps(current.metadata),
                    policy_id,
                ),
            )
            self._log_audit(conn, policy_id, "update", old_value, current.to_dict(), changed_by)

        return current

    def delete_policy(self, policy_id: str, deleted_by: Optional[str] = None) -> bool:
        """Delete a policy."""
        policy = self.get_policy(policy_id)
        if not policy:
            return False

        with self.connection() as conn:
            self._log_audit(conn, policy_id, "delete", policy.to_dict(), None, deleted_by)
            conn.execute("DELETE FROM policies WHERE id = ?", (policy_id,))
        return True

    def toggle_policy(self, policy_id: str, enabled: bool, changed_by: Optional[str] = None) -> bool:
        """Toggle policy enabled status."""
        result = self.update_policy(policy_id, {"enabled": enabled}, changed_by)
        return result is not None

    # =========================================================================
    # Violation CRUD
    # =========================================================================

    def create_violation(self, violation: Violation) -> Violation:
        """Create a new violation."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO violations (
                    id, policy_id, rule_id, rule_name, framework_id, vertical_id,
                    workspace_id, severity, status, description, source,
                    detected_at, resolved_at, resolved_by, resolution_notes, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    violation.id,
                    violation.policy_id,
                    violation.rule_id,
                    violation.rule_name,
                    violation.framework_id,
                    violation.vertical_id,
                    violation.workspace_id,
                    violation.severity,
                    violation.status,
                    violation.description,
                    violation.source,
                    violation.detected_at.isoformat(),
                    violation.resolved_at.isoformat() if violation.resolved_at else None,
                    violation.resolved_by,
                    violation.resolution_notes,
                    json.dumps(violation.metadata),
                ),
            )
        return violation

    def get_violation(self, violation_id: str) -> Optional[Violation]:
        """Get a violation by ID."""
        row = self.fetch_one("SELECT * FROM violations WHERE id = ?", (violation_id,))
        if not row:
            return None
        return self._row_to_violation(row)

    def list_violations(
        self,
        workspace_id: Optional[str] = None,
        vertical_id: Optional[str] = None,
        framework_id: Optional[str] = None,
        policy_id: Optional[str] = None,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Violation]:
        """List violations with optional filters."""
        conditions = []
        params: List[Any] = []

        if workspace_id:
            conditions.append("workspace_id = ?")
            params.append(workspace_id)
        if vertical_id:
            conditions.append("vertical_id = ?")
            params.append(vertical_id)
        if framework_id:
            conditions.append("framework_id = ?")
            params.append(framework_id)
        if policy_id:
            conditions.append("policy_id = ?")
            params.append(policy_id)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if severity:
            conditions.append("severity = ?")
            params.append(severity)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"SELECT * FROM violations {where} ORDER BY detected_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self.fetch_all(sql, tuple(params))
        return [self._row_to_violation(row) for row in rows]

    def update_violation_status(
        self,
        violation_id: str,
        status: str,
        resolved_by: Optional[str] = None,
        resolution_notes: Optional[str] = None,
    ) -> Optional[Violation]:
        """Update a violation's status."""
        violation = self.get_violation(violation_id)
        if not violation:
            return None

        resolved_at = None
        if status in ("resolved", "false_positive"):
            resolved_at = datetime.utcnow()

        with self.connection() as conn:
            conn.execute(
                """
                UPDATE violations SET status = ?, resolved_at = ?, resolved_by = ?, resolution_notes = ?
                WHERE id = ?
                """,
                (
                    status,
                    resolved_at.isoformat() if resolved_at else None,
                    resolved_by,
                    resolution_notes,
                    violation_id,
                ),
            )

        violation.status = status
        violation.resolved_at = resolved_at
        violation.resolved_by = resolved_by
        violation.resolution_notes = resolution_notes
        return violation

    def delete_violation(self, violation_id: str) -> bool:
        """Delete a violation."""
        return self.delete_by_id("violations", "id", violation_id)

    def count_violations(
        self,
        workspace_id: Optional[str] = None,
        status: Optional[str] = None,
        severity: Optional[str] = None,
    ) -> Dict[str, int]:
        """Get violation counts by severity."""
        conditions = []
        params: List[Any] = []

        if workspace_id:
            conditions.append("workspace_id = ?")
            params.append(workspace_id)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if severity:
            conditions.append("severity = ?")
            params.append(severity)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT severity, COUNT(*) as count FROM violations {where}
            GROUP BY severity
        """

        rows = self.fetch_all(sql, tuple(params))
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "total": 0}
        for row in rows:
            severity_val = row[0]
            count_val = row[1]
            if severity_val in counts:
                counts[severity_val] = count_val
            counts["total"] += count_val
        return counts

    # =========================================================================
    # Helpers
    # =========================================================================

    def _row_to_policy(self, row: tuple) -> Policy:
        """Convert database row to Policy object."""
        # Column order: id, name, description, framework_id, workspace_id, vertical_id,
        #              level, enabled, rules_json, created_at, updated_at, created_by, metadata_json
        return Policy(
            id=row[0],
            name=row[1],
            description=row[2] or "",
            framework_id=row[3],
            workspace_id=row[4],
            vertical_id=row[5],
            level=row[6] or "recommended",
            enabled=bool(row[7]),
            rules=[PolicyRule.from_dict(r) for r in json.loads(row[8] or "[]")],
            created_at=datetime.fromisoformat(row[9]) if row[9] else datetime.utcnow(),
            updated_at=datetime.fromisoformat(row[10]) if row[10] else datetime.utcnow(),
            created_by=row[11],
            metadata=json.loads(row[12] or "{}"),
        )

    def _row_to_violation(self, row: tuple) -> Violation:
        """Convert database row to Violation object."""
        # Column order: id, policy_id, rule_id, rule_name, framework_id, vertical_id,
        #              workspace_id, severity, status, description, source,
        #              detected_at, resolved_at, resolved_by, resolution_notes, metadata_json
        return Violation(
            id=row[0],
            policy_id=row[1] or "",
            rule_id=row[2],
            rule_name=row[3] or "",
            framework_id=row[4],
            vertical_id=row[5],
            workspace_id=row[6],
            severity=row[7] or "medium",
            status=row[8] or "open",
            description=row[9] or "",
            source=row[10] or "",
            detected_at=datetime.fromisoformat(row[11]) if row[11] else datetime.utcnow(),
            resolved_at=datetime.fromisoformat(row[12]) if row[12] else None,
            resolved_by=row[13],
            resolution_notes=row[14],
            metadata=json.loads(row[15] or "{}"),
        )

    def _log_audit(
        self,
        conn,
        policy_id: str,
        action: str,
        old_value: Optional[Dict],
        new_value: Optional[Dict],
        changed_by: Optional[str],
    ) -> None:
        """Log a policy change to the audit table."""
        audit_id = f"audit_{uuid.uuid4().hex[:12]}"
        conn.execute(
            """
            INSERT INTO policy_audit (id, policy_id, action, changed_by, old_value, new_value)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                audit_id,
                policy_id,
                action,
                changed_by,
                json.dumps(old_value) if old_value else None,
                json.dumps(new_value) if new_value else None,
            ),
        )


# Singleton instance
_policy_store: Optional[PolicyStore] = None


def get_policy_store(db_path: Optional[Path] = None) -> PolicyStore:
    """Get or create the policy store singleton."""
    global _policy_store
    if _policy_store is None:
        _policy_store = PolicyStore(db_path)
    return _policy_store


__all__ = [
    "Policy",
    "PolicyRule",
    "Violation",
    "PolicyStore",
    "get_policy_store",
]
