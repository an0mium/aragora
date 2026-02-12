"""
Version management for workflows.

Provides operations for retrieving version history and restoring previous versions.
"""

from __future__ import annotations

from typing import Any, Optional

from .core import _get_store
from .crud import update_workflow


async def get_workflow_versions(
    workflow_id: str,
    tenant_id: str = "default",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Get version history for a workflow."""
    store = _get_store()
    return store.get_versions(workflow_id, tenant_id, limit)


async def restore_workflow_version(
    workflow_id: str,
    version: str,
    tenant_id: str = "default",
) -> dict[str, Any] | None:
    """Restore a workflow to a specific version."""
    store = _get_store()
    old_workflow = store.get_version(workflow_id, version)

    if old_workflow:
        # Create new version from old
        restored = old_workflow.clone(new_id=workflow_id, new_name=old_workflow.name)
        return await update_workflow(workflow_id, restored.to_dict(), tenant_id)

    return None


__all__ = [
    "get_workflow_versions",
    "restore_workflow_version",
]
