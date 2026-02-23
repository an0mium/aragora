"""
CRUD operations for workflows.

Provides create, read, update, and delete operations for workflow definitions.
"""

from __future__ import annotations

import sys
import uuid
from datetime import datetime, timezone
from typing import Any

from .core import (
    logger,
    _get_store,
    WorkflowDefinition,
    audit_data,
)


def _get_workflow_definition_cls() -> Any:
    """Return WorkflowDefinition, honoring test overrides."""
    try:
        pkg = sys.modules.get("aragora.server.handlers.workflows")
        override = getattr(pkg, "WorkflowDefinition", None) if pkg is not None else None
        if override is not None and override is not WorkflowDefinition:
            return override
    except (AttributeError, TypeError) as e:
        logger.debug("Failed to resolve WorkflowDefinition override: %s", e)
    return WorkflowDefinition


def _get_audit_fn() -> Any:
    """Return audit_data, honoring test overrides."""
    try:
        pkg = sys.modules.get("aragora.server.handlers.workflows")
        override = getattr(pkg, "audit_data", None) if pkg is not None else None
        if override is not None and override is not audit_data:
            return override
    except (AttributeError, TypeError) as e:
        logger.debug("Failed to resolve audit_data override: %s", e)
    return audit_data


async def list_workflows(
    tenant_id: str = "default",
    category: str | None = None,
    tags: list[str] | None = None,
    search: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """
    List workflows with optional filtering.

    Returns:
        {
            "workflows": [...],
            "total_count": int,
            "limit": int,
            "offset": int,
        }
    """
    store = _get_store()
    workflows, total = store.list_workflows(
        tenant_id=tenant_id,
        category=category,
        tags=tags,
        search=search,
        limit=limit,
        offset=offset,
    )

    return {
        "workflows": [w.to_dict() for w in workflows],
        "total_count": total,
        "limit": limit,
        "offset": offset,
    }


async def get_workflow(workflow_id: str, tenant_id: str = "default") -> dict[str, Any] | None:
    """Get a workflow by ID."""
    store = _get_store()
    workflow = store.get_workflow(workflow_id, tenant_id)
    return workflow.to_dict() if workflow else None


async def create_workflow(
    data: dict[str, Any],
    tenant_id: str = "default",
    created_by: str = "",
) -> dict[str, Any]:
    """
    Create a new workflow.

    Args:
        data: Workflow definition data
        tenant_id: Tenant ID for isolation
        created_by: User ID of creator

    Returns:
        Created workflow definition
    """
    store = _get_store()

    # Generate ID if not provided
    workflow_id = data.get("id") or f"wf_{uuid.uuid4().hex[:12]}"
    data["id"] = workflow_id
    data["tenant_id"] = tenant_id
    data["created_by"] = created_by
    data["created_at"] = datetime.now(timezone.utc).isoformat()
    data["updated_at"] = data["created_at"]

    workflow_cls = _get_workflow_definition_cls()
    workflow = workflow_cls.from_dict(data)

    # Validate
    is_valid, errors = workflow.validate()
    if not is_valid:
        raise ValueError(f"Invalid workflow: {', '.join(errors)}")

    # Save to persistent store
    store.save_workflow(workflow)

    # Save initial version
    store.save_version(workflow)

    logger.info("Created workflow %s: %s", workflow_id, workflow.name)
    _get_audit_fn()(
        user_id=created_by or "system",
        resource_type="workflow",
        resource_id=workflow_id,
        action="create",
        workflow_name=workflow.name,
        tenant_id=tenant_id,
    )
    return workflow.to_dict()


async def update_workflow(
    workflow_id: str,
    data: dict[str, Any],
    tenant_id: str = "default",
) -> dict[str, Any] | None:
    """
    Update an existing workflow.

    Args:
        workflow_id: ID of workflow to update
        data: Updated workflow data
        tenant_id: Tenant ID for isolation

    Returns:
        Updated workflow definition or None if not found
    """
    store = _get_store()
    existing = store.get_workflow(workflow_id, tenant_id)
    if not existing:
        return None

    # Preserve metadata
    data["id"] = workflow_id
    data["tenant_id"] = tenant_id
    data["created_by"] = existing.created_by
    data["created_at"] = existing.created_at.isoformat() if existing.created_at else None
    data["updated_at"] = datetime.now(timezone.utc).isoformat()

    # Increment version
    old_version = existing.version
    parts = old_version.split(".")
    parts[-1] = str(int(parts[-1]) + 1)
    data["version"] = ".".join(parts)

    workflow_cls = _get_workflow_definition_cls()
    workflow = workflow_cls.from_dict(data)

    # Validate
    is_valid, errors = workflow.validate()
    if not is_valid:
        raise ValueError(f"Invalid workflow: {', '.join(errors)}")

    # Save to persistent store
    store.save_workflow(workflow)

    # Save version history
    store.save_version(workflow)

    logger.info("Updated workflow %s to version %s", workflow_id, workflow.version)
    _get_audit_fn()(
        user_id="system",
        resource_type="workflow",
        resource_id=workflow_id,
        action="update",
        new_version=workflow.version,
        tenant_id=tenant_id,
    )
    return workflow.to_dict()


async def delete_workflow(workflow_id: str, tenant_id: str = "default") -> bool:
    """
    Delete a workflow.

    Args:
        workflow_id: ID of workflow to delete
        tenant_id: Tenant ID for isolation

    Returns:
        True if deleted, False if not found
    """
    store = _get_store()
    deleted = store.delete_workflow(workflow_id, tenant_id)

    if deleted:
        logger.info("Deleted workflow %s", workflow_id)
        _get_audit_fn()(
            user_id="system",
            resource_type="workflow",
            resource_id=workflow_id,
            action="delete",
            tenant_id=tenant_id,
        )

    return deleted


__all__ = [
    "list_workflows",
    "get_workflow",
    "create_workflow",
    "update_workflow",
    "delete_workflow",
]
