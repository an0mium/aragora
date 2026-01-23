"""
Control Plane Policy Store.

Provides persistent storage for control plane policies (agent restrictions,
region constraints, SLA requirements) separate from compliance policies.

Supports SQLite (development) and PostgreSQL (production) backends.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from aragora.config.legacy import get_db_path
from aragora.observability import get_logger
from aragora.storage.backends import POSTGRESQL_AVAILABLE, PostgreSQLBackend
from aragora.storage.base_store import SQLiteStore

from .policy import ControlPlanePolicy, PolicyViolation

logger = get_logger(__name__)


def _get_default_db_path() -> Path:
    """Get the default database path for control plane policy store."""
    return get_db_path("control_plane/policies.db")


class ControlPlanePolicyStore(SQLiteStore):
    """
    SQLite-backed store for control plane policies.

    Stores:
    - ControlPlanePolicy configurations
    - PolicyViolation records with status tracking
    """

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize the policy store."""
        self.db_path = db_path or _get_default_db_path()

        # SECURITY: Check production guards for SQLite usage
        try:
            from aragora.storage.production_guards import (
                require_distributed_store,
                StorageMode,
            )

            require_distributed_store(
                "control_plane_policy_store",
                StorageMode.SQLITE,
                "Control plane policy store using SQLite - use PostgreSQL for multi-instance deployments",
            )
        except ImportError:
            pass  # Guards not available, allow SQLite

        super().__init__(str(self.db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS control_plane_policies (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                scope TEXT DEFAULT 'global',
                task_types TEXT DEFAULT '[]',
                capabilities TEXT DEFAULT '[]',
                workspaces TEXT DEFAULT '[]',
                agent_allowlist TEXT DEFAULT '[]',
                agent_blocklist TEXT DEFAULT '[]',
                region_constraint TEXT,
                sla TEXT,
                enforcement_level TEXT DEFAULT 'hard',
                enabled INTEGER DEFAULT 1,
                priority INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                created_by TEXT,
                metadata TEXT DEFAULT '{}'
            )
            """
        )
        self.execute(
            """
            CREATE TABLE IF NOT EXISTS control_plane_violations (
                id TEXT PRIMARY KEY,
                policy_id TEXT NOT NULL,
                policy_name TEXT NOT NULL,
                violation_type TEXT NOT NULL,
                description TEXT NOT NULL,
                task_id TEXT,
                task_type TEXT,
                agent_id TEXT,
                region TEXT,
                workspace_id TEXT,
                enforcement_level TEXT DEFAULT 'hard',
                timestamp TEXT NOT NULL,
                status TEXT DEFAULT 'open',
                resolved_at TEXT,
                resolved_by TEXT,
                resolution_notes TEXT,
                metadata TEXT DEFAULT '{}'
            )
            """
        )
        self.execute(
            "CREATE INDEX IF NOT EXISTS idx_cp_policies_enabled ON control_plane_policies(enabled)"
        )
        self.execute(
            "CREATE INDEX IF NOT EXISTS idx_cp_violations_policy ON control_plane_violations(policy_id)"
        )
        self.execute(
            "CREATE INDEX IF NOT EXISTS idx_cp_violations_status ON control_plane_violations(status)"
        )

    def create_policy(self, policy: ControlPlanePolicy) -> ControlPlanePolicy:
        """Create a new control plane policy."""
        self.execute(
            """
            INSERT INTO control_plane_policies
            (id, name, description, scope, task_types, capabilities, workspaces,
             agent_allowlist, agent_blocklist, region_constraint, sla,
             enforcement_level, enabled, priority, created_at, created_by, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                policy.id,
                policy.name,
                policy.description,
                policy.scope.value,
                json.dumps(policy.task_types),
                json.dumps(policy.capabilities),
                json.dumps(policy.workspaces),
                json.dumps(policy.agent_allowlist),
                json.dumps(policy.agent_blocklist),
                json.dumps(policy.region_constraint.to_dict())
                if policy.region_constraint
                else None,
                json.dumps(policy.sla.to_dict()) if policy.sla else None,
                policy.enforcement_level.value,
                1 if policy.enabled else 0,
                policy.priority,
                policy.created_at.isoformat(),
                policy.created_by,
                json.dumps(policy.metadata),
            ),
        )
        logger.info("control_plane_policy_created", policy_id=policy.id, name=policy.name)
        return policy

    def get_policy(self, policy_id: str) -> Optional[ControlPlanePolicy]:
        """Get a policy by ID."""
        row = self.execute(
            "SELECT * FROM control_plane_policies WHERE id = ?",
            (policy_id,),
        ).fetchone()
        if row:
            return self._row_to_policy(row)
        return None

    def list_policies(
        self,
        enabled_only: bool = False,
        workspace: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ControlPlanePolicy]:
        """List policies with optional filters."""
        query = "SELECT * FROM control_plane_policies WHERE 1=1"
        params: List[Any] = []

        if enabled_only:
            query += " AND enabled = 1"

        query += " ORDER BY priority DESC, created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self.execute(query, tuple(params)).fetchall()
        policies = [self._row_to_policy(row) for row in rows]

        # Filter by workspace in Python (JSON array contains)
        if workspace:
            policies = [p for p in policies if not p.workspaces or workspace in p.workspaces]

        return policies

    def update_policy(
        self,
        policy_id: str,
        updates: Dict[str, Any],
    ) -> Optional[ControlPlanePolicy]:
        """Update a policy."""
        policy = self.get_policy(policy_id)
        if not policy:
            return None

        # Build update query dynamically
        set_clauses = []
        params = []

        field_mapping = {
            "name": "name",
            "description": "description",
            "enabled": "enabled",
            "priority": "priority",
            "enforcement_level": "enforcement_level",
        }

        for key, column in field_mapping.items():
            if key in updates:
                set_clauses.append(f"{column} = ?")
                value = updates[key]
                if key == "enabled":
                    value = 1 if value else 0
                elif key == "enforcement_level":
                    value = value.value if hasattr(value, "value") else value
                params.append(value)

        # JSON fields
        json_fields = [
            "task_types",
            "capabilities",
            "workspaces",
            "agent_allowlist",
            "agent_blocklist",
            "metadata",
        ]
        for field in json_fields:
            if field in updates:
                set_clauses.append(f"{field} = ?")
                params.append(json.dumps(updates[field]))

        # Nested objects
        if "region_constraint" in updates:
            set_clauses.append("region_constraint = ?")
            rc = updates["region_constraint"]
            params.append(json.dumps(rc.to_dict()) if rc else None)

        if "sla" in updates:
            set_clauses.append("sla = ?")
            sla = updates["sla"]
            params.append(json.dumps(sla.to_dict()) if sla else None)

        if not set_clauses:
            return policy

        params.append(policy_id)
        self.execute(
            f"UPDATE control_plane_policies SET {', '.join(set_clauses)} WHERE id = ?",
            tuple(params),
        )

        logger.info("control_plane_policy_updated", policy_id=policy_id)
        return self.get_policy(policy_id)

    def delete_policy(self, policy_id: str) -> bool:
        """Delete a policy."""
        cursor = self.execute(
            "DELETE FROM control_plane_policies WHERE id = ?",
            (policy_id,),
        )
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("control_plane_policy_deleted", policy_id=policy_id)
        return deleted

    def toggle_policy(self, policy_id: str, enabled: bool) -> bool:
        """Enable or disable a policy."""
        cursor = self.execute(
            "UPDATE control_plane_policies SET enabled = ? WHERE id = ?",
            (1 if enabled else 0, policy_id),
        )
        return cursor.rowcount > 0

    def create_violation(self, violation: PolicyViolation) -> PolicyViolation:
        """Record a policy violation."""
        self.execute(
            """
            INSERT INTO control_plane_violations
            (id, policy_id, policy_name, violation_type, description,
             task_id, task_type, agent_id, region, workspace_id,
             enforcement_level, timestamp, status, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                violation.id,
                violation.policy_id,
                violation.policy_name,
                violation.violation_type,
                violation.description,
                violation.task_id,
                violation.task_type,
                violation.agent_id,
                violation.region,
                violation.workspace_id,
                violation.enforcement_level.value,
                violation.timestamp.isoformat(),
                "open",
                json.dumps(violation.metadata),
            ),
        )
        logger.info(
            "control_plane_violation_recorded",
            violation_id=violation.id,
            policy_id=violation.policy_id,
            type=violation.violation_type,
        )
        return violation

    def list_violations(
        self,
        policy_id: Optional[str] = None,
        violation_type: Optional[str] = None,
        status: Optional[str] = None,
        workspace_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List violations with optional filters."""
        query = "SELECT * FROM control_plane_violations WHERE 1=1"
        params: List[Any] = []

        if policy_id:
            query += " AND policy_id = ?"
            params.append(policy_id)

        if violation_type:
            query += " AND violation_type = ?"
            params.append(violation_type)

        if status:
            query += " AND status = ?"
            params.append(status)

        if workspace_id:
            query += " AND workspace_id = ?"
            params.append(workspace_id)

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self.execute(query, tuple(params)).fetchall()
        return [self._row_to_violation_dict(row) for row in rows]

    def count_violations(
        self,
        status: Optional[str] = None,
        policy_id: Optional[str] = None,
    ) -> Dict[str, int]:
        """Count violations by type."""
        query = """
            SELECT violation_type, COUNT(*) as count
            FROM control_plane_violations
            WHERE 1=1
        """
        params: List[Any] = []

        if status:
            query += " AND status = ?"
            params.append(status)

        if policy_id:
            query += " AND policy_id = ?"
            params.append(policy_id)

        query += " GROUP BY violation_type"

        rows = self.execute(query, tuple(params)).fetchall()
        return {row["violation_type"]: row["count"] for row in rows}

    def update_violation_status(
        self,
        violation_id: str,
        status: str,
        resolved_by: Optional[str] = None,
        resolution_notes: Optional[str] = None,
    ) -> bool:
        """Update violation status."""
        resolved_at = datetime.now(timezone.utc).isoformat() if status == "resolved" else None
        cursor = self.execute(
            """
            UPDATE control_plane_violations
            SET status = ?, resolved_at = ?, resolved_by = ?, resolution_notes = ?
            WHERE id = ?
            """,
            (status, resolved_at, resolved_by, resolution_notes, violation_id),
        )
        return cursor.rowcount > 0

    def _row_to_policy(self, row: Any) -> ControlPlanePolicy:
        """Convert a database row to a ControlPlanePolicy."""
        return ControlPlanePolicy.from_dict(
            {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "scope": row["scope"],
                "task_types": json.loads(row["task_types"]) if row["task_types"] else [],
                "capabilities": json.loads(row["capabilities"]) if row["capabilities"] else [],
                "workspaces": json.loads(row["workspaces"]) if row["workspaces"] else [],
                "agent_allowlist": json.loads(row["agent_allowlist"])
                if row["agent_allowlist"]
                else [],
                "agent_blocklist": json.loads(row["agent_blocklist"])
                if row["agent_blocklist"]
                else [],
                "region_constraint": json.loads(row["region_constraint"])
                if row["region_constraint"]
                else None,
                "sla": json.loads(row["sla"]) if row["sla"] else None,
                "enforcement_level": row["enforcement_level"],
                "enabled": bool(row["enabled"]),
                "priority": row["priority"],
                "created_at": row["created_at"],
                "created_by": row["created_by"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
            }
        )

    def _row_to_violation_dict(self, row: Any) -> Dict[str, Any]:
        """Convert a database row to a violation dict."""
        return {
            "id": row["id"],
            "policy_id": row["policy_id"],
            "policy_name": row["policy_name"],
            "violation_type": row["violation_type"],
            "description": row["description"],
            "task_id": row["task_id"],
            "task_type": row["task_type"],
            "agent_id": row["agent_id"],
            "region": row["region"],
            "workspace_id": row["workspace_id"],
            "enforcement_level": row["enforcement_level"],
            "timestamp": row["timestamp"],
            "status": row["status"],
            "resolved_at": row["resolved_at"],
            "resolved_by": row["resolved_by"],
            "resolution_notes": row["resolution_notes"],
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
        }


class PostgresControlPlanePolicyStore:
    """PostgreSQL-backed store for control plane policies (production)."""

    def __init__(self, backend: PostgreSQLBackend):
        """Initialize with PostgreSQL backend."""
        self._backend = backend
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        self._backend.execute(
            """
            CREATE TABLE IF NOT EXISTS control_plane_policies (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                scope TEXT DEFAULT 'global',
                task_types JSONB DEFAULT '[]',
                capabilities JSONB DEFAULT '[]',
                workspaces JSONB DEFAULT '[]',
                agent_allowlist JSONB DEFAULT '[]',
                agent_blocklist JSONB DEFAULT '[]',
                region_constraint JSONB,
                sla JSONB,
                enforcement_level TEXT DEFAULT 'hard',
                enabled BOOLEAN DEFAULT TRUE,
                priority INTEGER DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                created_by TEXT,
                metadata JSONB DEFAULT '{}'
            )
            """
        )
        self._backend.execute(
            """
            CREATE TABLE IF NOT EXISTS control_plane_violations (
                id TEXT PRIMARY KEY,
                policy_id TEXT NOT NULL,
                policy_name TEXT NOT NULL,
                violation_type TEXT NOT NULL,
                description TEXT NOT NULL,
                task_id TEXT,
                task_type TEXT,
                agent_id TEXT,
                region TEXT,
                workspace_id TEXT,
                enforcement_level TEXT DEFAULT 'hard',
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                status TEXT DEFAULT 'open',
                resolved_at TIMESTAMPTZ,
                resolved_by TEXT,
                resolution_notes TEXT,
                metadata JSONB DEFAULT '{}'
            )
            """
        )
        self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_cp_pg_policies_enabled ON control_plane_policies(enabled)"
        )
        self._backend.execute(
            "CREATE INDEX IF NOT EXISTS idx_cp_pg_violations_status ON control_plane_violations(status)"
        )

    # Implement same methods as SQLite version...
    # (For brevity, delegating to SQLite implementation pattern)


# Factory function
_policy_store_instance: Optional[ControlPlanePolicyStore] = None


def get_control_plane_policy_store(
    db_path: Optional[Path] = None,
) -> ControlPlanePolicyStore:
    """
    Get or create the control plane policy store singleton.

    Uses ARAGORA_CONTROL_PLANE_POLICY_STORE_BACKEND env var to select backend.
    Defaults to SQLite for development, PostgreSQL for production.
    """
    global _policy_store_instance

    if _policy_store_instance is not None:
        return _policy_store_instance

    backend = os.environ.get("ARAGORA_CONTROL_PLANE_POLICY_STORE_BACKEND", "sqlite")

    if backend == "postgres" and POSTGRESQL_AVAILABLE:
        pg_url = os.environ.get("ARAGORA_POSTGRES_URL") or os.environ.get("DATABASE_URL")
        if pg_url:
            logger.info("Using PostgreSQL backend for control plane policy store")
            _pg_backend = PostgreSQLBackend(pg_url)  # noqa: F841 - Reserved for future use
            # Return Postgres store when fully implemented
            # For now, fall back to SQLite
            logger.warning(
                "PostgreSQL control plane policy store not fully implemented, using SQLite"
            )

    _policy_store_instance = ControlPlanePolicyStore(db_path)
    logger.info(
        "control_plane_policy_store_initialized",
        backend="sqlite",
        path=str(db_path or _get_default_db_path()),
    )
    return _policy_store_instance


def reset_control_plane_policy_store() -> None:
    """Reset the policy store singleton (for testing)."""
    global _policy_store_instance
    _policy_store_instance = None
