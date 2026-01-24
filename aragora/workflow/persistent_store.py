"""
Persistent Workflow Storage (SQLite by default, PostgreSQL optional).

Replaces in-memory storage with database-backed persistence for:
- Workflow definitions
- Workflow versions
- Workflow templates
- Execution history

Usage:
    from aragora.workflow.persistent_store import get_workflow_store

    store = get_workflow_store()
    workflow = store.get_workflow("wf_123")
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from aragora.workflow.types import WorkflowDefinition

if TYPE_CHECKING:
    from aragora.workflow.postgres_workflow_store import PostgresWorkflowStore

logger = logging.getLogger(__name__)

# Type alias for workflow store backends
WorkflowStoreType = Union["PersistentWorkflowStore", "PostgresWorkflowStore"]


# Default database path
DEFAULT_DB_PATH = Path(os.getenv("ARAGORA_WORKFLOW_DB", Path.home() / ".aragora" / "workflows.db"))


# Schema definition
WORKFLOW_SCHEMA = """
-- Workflow definitions table
CREATE TABLE IF NOT EXISTS workflows (
    id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    category TEXT DEFAULT 'general',
    version TEXT DEFAULT '1.0.0',
    definition JSON NOT NULL,
    tags TEXT DEFAULT '[]',
    is_template INTEGER DEFAULT 0,
    is_enabled INTEGER DEFAULT 1,
    created_by TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Workflow versions table
CREATE TABLE IF NOT EXISTS workflow_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    workflow_id TEXT NOT NULL,
    version TEXT NOT NULL,
    definition JSON NOT NULL,
    created_by TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE CASCADE
);

-- Workflow templates table (for gallery)
CREATE TABLE IF NOT EXISTS workflow_templates (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT DEFAULT '',
    category TEXT DEFAULT 'general',
    definition JSON NOT NULL,
    tags TEXT DEFAULT '[]',
    usage_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Workflow executions table
CREATE TABLE IF NOT EXISTS workflow_executions (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    tenant_id TEXT NOT NULL DEFAULT 'default',
    status TEXT NOT NULL DEFAULT 'pending',
    inputs JSON DEFAULT '{}',
    outputs JSON DEFAULT '{}',
    steps JSON DEFAULT '[]',
    error TEXT,
    started_at TEXT,
    completed_at TEXT,
    duration_ms REAL,
    FOREIGN KEY (workflow_id) REFERENCES workflows(id) ON DELETE SET NULL
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_workflows_tenant ON workflows(tenant_id);
CREATE INDEX IF NOT EXISTS idx_workflows_category ON workflows(category);
CREATE INDEX IF NOT EXISTS idx_workflow_versions_workflow ON workflow_versions(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_workflow ON workflow_executions(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_status ON workflow_executions(status);
CREATE INDEX IF NOT EXISTS idx_workflow_executions_tenant ON workflow_executions(tenant_id);
"""


class PersistentWorkflowStore:
    """
    SQLite-backed persistent storage for workflows.

    Provides the same interface as the in-memory WorkflowStore
    but persists data to disk.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the persistent store.

        Args:
            db_path: Path to SQLite database file. If None, uses default.

        Raises:
            DistributedStateError: In production if PostgreSQL is not available
        """
        # SECURITY: Check production guards for SQLite usage
        try:
            from aragora.storage.production_guards import (
                require_distributed_store,
                StorageMode,
            )

            require_distributed_store(
                "workflow_store",
                StorageMode.SQLITE,
                "Workflow store using SQLite - use PostgreSQL for multi-instance deployments",
            )
        except ImportError:
            pass  # Guards not available, allow SQLite

        self._db_path = db_path or DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

        logger.info(f"Initialized workflow store at {self._db_path}")

    def _init_db(self) -> None:
        """Initialize database schema."""
        import sqlite3

        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.executescript(WORKFLOW_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def _get_conn(self):
        """Get a database connection."""
        import sqlite3

        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # =========================================================================
    # Workflow CRUD
    # =========================================================================

    def save_workflow(self, workflow: WorkflowDefinition) -> None:
        """
        Save or update a workflow.

        Args:
            workflow: WorkflowDefinition to save
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            now = datetime.now(timezone.utc).isoformat()

            # Check if exists
            cursor.execute("SELECT id FROM workflows WHERE id = ?", (workflow.id,))
            exists = cursor.fetchone() is not None

            # Convert to storage format
            data = workflow.to_dict()
            definition_json = json.dumps(data)
            tags_json = json.dumps(workflow.tags or [])

            if exists:
                cursor.execute(
                    """
                    UPDATE workflows SET
                        name = ?,
                        description = ?,
                        category = ?,
                        version = ?,
                        definition = ?,
                        tags = ?,
                        is_template = ?,
                        is_enabled = ?,
                        updated_at = ?
                    WHERE id = ?
                """,
                    (
                        workflow.name,
                        workflow.description,
                        (
                            workflow.category.value
                            if hasattr(workflow.category, "value")
                            else str(workflow.category)
                        ),
                        workflow.version,
                        definition_json,
                        tags_json,
                        1 if workflow.is_template else 0,
                        1 if getattr(workflow, "is_enabled", True) else 0,
                        now,
                        workflow.id,
                    ),
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO workflows (
                        id, tenant_id, name, description, category, version,
                        definition, tags, is_template, is_enabled, created_by,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        workflow.id,
                        getattr(workflow, "tenant_id", "default"),
                        workflow.name,
                        workflow.description,
                        (
                            workflow.category.value
                            if hasattr(workflow.category, "value")
                            else str(workflow.category)
                        ),
                        workflow.version,
                        definition_json,
                        tags_json,
                        1 if workflow.is_template else 0,
                        1 if getattr(workflow, "is_enabled", True) else 0,
                        getattr(workflow, "created_by", ""),
                        workflow.created_at.isoformat() if workflow.created_at else now,
                        now,
                    ),
                )

            conn.commit()
            logger.debug(f"Saved workflow {workflow.id}")

        finally:
            conn.close()

    def get_workflow(
        self, workflow_id: str, tenant_id: str = "default"
    ) -> Optional[WorkflowDefinition]:
        """
        Get a workflow by ID.

        Args:
            workflow_id: Workflow ID
            tenant_id: Tenant ID for isolation

        Returns:
            WorkflowDefinition or None if not found
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT definition FROM workflows
                WHERE id = ? AND tenant_id = ?
            """,
                (workflow_id, tenant_id),
            )

            row = cursor.fetchone()
            if row:
                data = json.loads(row["definition"])
                return WorkflowDefinition.from_dict(data)
            return None

        finally:
            conn.close()

    def list_workflows(
        self,
        tenant_id: str = "default",
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[List[WorkflowDefinition], int]:
        """
        List workflows with filtering.

        Returns:
            Tuple of (workflows, total_count)
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            # Build query
            query = "SELECT definition FROM workflows WHERE tenant_id = ?"
            count_query = "SELECT COUNT(*) FROM workflows WHERE tenant_id = ?"
            params: List[Any] = [tenant_id]

            if category:
                query += " AND category = ?"
                count_query += " AND category = ?"
                params.append(category)

            if search:
                query += " AND (name LIKE ? OR description LIKE ?)"
                count_query += " AND (name LIKE ? OR description LIKE ?)"
                search_param = f"%{search}%"
                params.extend([search_param, search_param])

            # Get total count
            cursor.execute(count_query, params)
            total = cursor.fetchone()[0]

            # Get workflows with pagination
            query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)

            workflows = []
            for row in cursor.fetchall():
                data = json.loads(row["definition"])
                workflow = WorkflowDefinition.from_dict(data)

                # Filter by tags if specified
                if tags:
                    if not any(t in (workflow.tags or []) for t in tags):
                        continue

                workflows.append(workflow)

            return workflows, total

        finally:
            conn.close()

    def delete_workflow(self, workflow_id: str, tenant_id: str = "default") -> bool:
        """
        Delete a workflow.

        Args:
            workflow_id: Workflow ID
            tenant_id: Tenant ID for isolation

        Returns:
            True if deleted, False if not found
        """
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM workflows
                WHERE id = ? AND tenant_id = ?
            """,
                (workflow_id, tenant_id),
            )

            deleted = cursor.rowcount > 0
            conn.commit()

            if deleted:
                logger.info(f"Deleted workflow {workflow_id}")

            return deleted

        finally:
            conn.close()

    # =========================================================================
    # Version Management
    # =========================================================================

    def save_version(self, workflow: WorkflowDefinition) -> None:
        """Save a workflow version to history."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            data = workflow.to_dict()
            definition_json = json.dumps(data)
            now = datetime.now(timezone.utc).isoformat()

            cursor.execute(
                """
                INSERT INTO workflow_versions (
                    workflow_id, version, definition, created_by, created_at
                ) VALUES (?, ?, ?, ?, ?)
            """,
                (
                    workflow.id,
                    workflow.version,
                    definition_json,
                    getattr(workflow, "created_by", ""),
                    now,
                ),
            )

            conn.commit()

        finally:
            conn.close()

    def get_versions(
        self,
        workflow_id: str,
        tenant_id: str = "default",
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Get version history for a workflow."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT version, created_by, created_at, definition
                FROM workflow_versions
                WHERE workflow_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (workflow_id, limit),
            )

            versions = []
            for row in cursor.fetchall():
                data = json.loads(row["definition"])
                versions.append(
                    {
                        "version": row["version"],
                        "created_by": row["created_by"],
                        "created_at": row["created_at"],
                        "step_count": len(data.get("steps", [])),
                    }
                )

            return versions

        finally:
            conn.close()

    def get_version(
        self,
        workflow_id: str,
        version: str,
    ) -> Optional[WorkflowDefinition]:
        """Get a specific version of a workflow."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT definition FROM workflow_versions
                WHERE workflow_id = ? AND version = ?
            """,
                (workflow_id, version),
            )

            row = cursor.fetchone()
            if row:
                data = json.loads(row["definition"])
                return WorkflowDefinition.from_dict(data)
            return None

        finally:
            conn.close()

    # =========================================================================
    # Templates
    # =========================================================================

    def save_template(self, template: WorkflowDefinition) -> None:
        """Save a workflow template."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            now = datetime.now(timezone.utc).isoformat()
            data = template.to_dict()
            definition_json = json.dumps(data)
            tags_json = json.dumps(template.tags or [])

            cursor.execute(
                """
                INSERT OR REPLACE INTO workflow_templates (
                    id, name, description, category, definition, tags,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    template.id,
                    template.name,
                    template.description,
                    (
                        template.category.value
                        if hasattr(template.category, "value")
                        else str(template.category)
                    ),
                    definition_json,
                    tags_json,
                    now,
                    now,
                ),
            )

            conn.commit()

        finally:
            conn.close()

    def get_template(self, template_id: str) -> Optional[WorkflowDefinition]:
        """Get a template by ID."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT definition FROM workflow_templates
                WHERE id = ?
            """,
                (template_id,),
            )

            row = cursor.fetchone()
            if row:
                data = json.loads(row["definition"])
                return WorkflowDefinition.from_dict(data)
            return None

        finally:
            conn.close()

    def list_templates(
        self,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 50,
    ) -> List[WorkflowDefinition]:
        """List workflow templates."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            query = "SELECT definition FROM workflow_templates"
            params: List[Any] = []

            if category:
                query += " WHERE category = ?"
                params.append(category)

            query += " ORDER BY usage_count DESC, updated_at DESC LIMIT ?"
            params.append(limit)

            cursor.execute(query, params)

            templates = []
            for row in cursor.fetchall():
                data = json.loads(row["definition"])
                template = WorkflowDefinition.from_dict(data)

                if tags and not any(t in (template.tags or []) for t in tags):
                    continue

                templates.append(template)

            return templates

        finally:
            conn.close()

    def increment_template_usage(self, template_id: str) -> None:
        """Increment template usage count."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE workflow_templates
                SET usage_count = usage_count + 1
                WHERE id = ?
            """,
                (template_id,),
            )
            conn.commit()
        finally:
            conn.close()

    # =========================================================================
    # Executions
    # =========================================================================

    def save_execution(self, execution: Dict[str, Any]) -> None:
        """Save or update an execution record."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR REPLACE INTO workflow_executions (
                    id, workflow_id, tenant_id, status, inputs, outputs,
                    steps, error, started_at, completed_at, duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    execution["id"],
                    execution.get("workflow_id"),
                    execution.get("tenant_id", "default"),
                    execution.get("status", "pending"),
                    json.dumps(execution.get("inputs", {})),
                    json.dumps(execution.get("outputs", {})),
                    json.dumps(execution.get("steps", [])),
                    execution.get("error"),
                    execution.get("started_at"),
                    execution.get("completed_at"),
                    execution.get("duration_ms"),
                ),
            )

            conn.commit()

        finally:
            conn.close()

    def get_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get an execution by ID."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM workflow_executions
                WHERE id = ?
            """,
                (execution_id,),
            )

            row = cursor.fetchone()
            if row:
                return {
                    "id": row["id"],
                    "workflow_id": row["workflow_id"],
                    "tenant_id": row["tenant_id"],
                    "status": row["status"],
                    "inputs": json.loads(row["inputs"] or "{}"),
                    "outputs": json.loads(row["outputs"] or "{}"),
                    "steps": json.loads(row["steps"] or "[]"),
                    "error": row["error"],
                    "started_at": row["started_at"],
                    "completed_at": row["completed_at"],
                    "duration_ms": row["duration_ms"],
                }
            return None

        finally:
            conn.close()

    def list_executions(
        self,
        workflow_id: Optional[str] = None,
        tenant_id: str = "default",
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[List[Dict[str, Any]], int]:
        """List executions with filtering."""
        conn = self._get_conn()
        try:
            cursor = conn.cursor()

            query = "SELECT * FROM workflow_executions WHERE tenant_id = ?"
            count_query = "SELECT COUNT(*) FROM workflow_executions WHERE tenant_id = ?"
            params: List[Any] = [tenant_id]

            if workflow_id:
                query += " AND workflow_id = ?"
                count_query += " AND workflow_id = ?"
                params.append(workflow_id)

            if status:
                query += " AND status = ?"
                count_query += " AND status = ?"
                params.append(status)

            # Get total count
            cursor.execute(count_query, params)
            total = cursor.fetchone()[0]

            # Get executions
            query += " ORDER BY started_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)

            executions = []
            for row in cursor.fetchall():
                executions.append(
                    {
                        "id": row["id"],
                        "workflow_id": row["workflow_id"],
                        "tenant_id": row["tenant_id"],
                        "status": row["status"],
                        "inputs": json.loads(row["inputs"] or "{}"),
                        "outputs": json.loads(row["outputs"] or "{}"),
                        "steps": json.loads(row["steps"] or "[]"),
                        "error": row["error"],
                        "started_at": row["started_at"],
                        "completed_at": row["completed_at"],
                        "duration_ms": row["duration_ms"],
                    }
                )

            return executions, total

        finally:
            conn.close()


# Global store instance
_store: Optional[PersistentWorkflowStore] = None

# Global workflow store instance for factory function
_workflow_store_instance: Optional[WorkflowStoreType] = None


def get_workflow_store(
    db_path: Optional[Path] = None,
    force_new: bool = False,
) -> WorkflowStoreType:
    """
    Get or create workflow store with automatic backend selection.

    Uses Supabase/PostgreSQL if configured, otherwise falls back to SQLite.

    Args:
        db_path: Optional database path (only used for SQLite backend)
        force_new: If True, creates a new store instance ignoring the singleton

    Returns:
        PersistentWorkflowStore (SQLite) or PostgresWorkflowStore (PostgreSQL)

    Note:
        For PostgreSQL backend, the store must be initialized with
        `await store.initialize()` before use.
    """
    global _workflow_store_instance

    if _workflow_store_instance is not None and not force_new:
        return _workflow_store_instance

    # Check for PostgreSQL backend
    from aragora.storage.factory import get_storage_backend, StorageBackend

    backend = get_storage_backend()

    if backend in (StorageBackend.POSTGRES, StorageBackend.SUPABASE):
        from aragora.storage.postgres_store import ASYNCPG_AVAILABLE

        if not ASYNCPG_AVAILABLE:
            logger.warning(
                "PostgreSQL backend selected but asyncpg not available. "
                "Falling back to SQLite. Install with: pip install asyncpg"
            )
            _workflow_store_instance = PersistentWorkflowStore(db_path)
        else:
            try:
                from aragora.utils.async_utils import run_async

                logger.info("Using PostgreSQL workflow store backend (sync init via run_async)")
                _workflow_store_instance = run_async(create_postgres_workflow_store())
            except Exception as e:
                logger.warning(
                    f"PostgreSQL workflow store initialization failed, falling back to SQLite: {e}",
                )
                _workflow_store_instance = PersistentWorkflowStore(db_path)
    else:
        _workflow_store_instance = PersistentWorkflowStore(db_path)

    return _workflow_store_instance


async def get_async_workflow_store(
    force_new: bool = False,
) -> WorkflowStoreType:
    """
    Get or create workflow store asynchronously with automatic backend selection.

    Uses Supabase/PostgreSQL if configured, otherwise falls back to SQLite.

    This is the preferred method for getting the workflow store in async contexts.

    Args:
        force_new: If True, creates a new store instance ignoring the singleton

    Returns:
        PersistentWorkflowStore (SQLite) or PostgresWorkflowStore (PostgreSQL)
    """
    global _workflow_store_instance

    if _workflow_store_instance is not None and not force_new:
        return _workflow_store_instance

    # Check for PostgreSQL backend
    from aragora.storage.factory import get_storage_backend, StorageBackend

    backend = get_storage_backend()

    if backend in (StorageBackend.POSTGRES, StorageBackend.SUPABASE):
        from aragora.storage.postgres_store import ASYNCPG_AVAILABLE

        if ASYNCPG_AVAILABLE:
            store = await create_postgres_workflow_store()
            _workflow_store_instance = store
            return store
        else:
            logger.warning(
                "PostgreSQL backend selected but asyncpg not available. "
                "Falling back to SQLite. Install with: pip install asyncpg"
            )

    _workflow_store_instance = PersistentWorkflowStore()
    return _workflow_store_instance


async def create_postgres_workflow_store() -> "PostgresWorkflowStore":
    """
    Create and initialize a PostgreSQL workflow store.

    This is a convenience function that handles pool creation and
    store initialization.

    Returns:
        Initialized PostgresWorkflowStore

    Raises:
        RuntimeError: If PostgreSQL is not configured or asyncpg not available
    """
    from aragora.workflow.postgres_workflow_store import PostgresWorkflowStore
    from aragora.storage.postgres_store import get_postgres_pool

    pool = await get_postgres_pool()
    store = PostgresWorkflowStore(pool)
    await store.initialize()
    return store


def reset_workflow_store() -> None:
    """Reset the global store (for testing)."""
    global _store, _workflow_store_instance
    _store = None
    _workflow_store_instance = None


__all__ = [
    "PersistentWorkflowStore",
    "get_workflow_store",
    "get_async_workflow_store",
    "create_postgres_workflow_store",
    "reset_workflow_store",
    "DEFAULT_DB_PATH",
]
