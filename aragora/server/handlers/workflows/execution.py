"""
Workflow execution operations.

Provides operations for executing workflows and managing execution state.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from .core import (
    logger,
    _get_store,
    _get_engine,
    _step_result_to_dict,
)


async def execute_workflow(
    workflow_id: str,
    inputs: Optional[dict[str, Any]] = None,
    tenant_id: str = "default",
) -> dict[str, Any]:
    """
    Execute a workflow.

    Args:
        workflow_id: ID of workflow to execute
        inputs: Input parameters for the workflow
        tenant_id: Tenant ID for isolation

    Returns:
        Execution result
    """
    store = _get_store()
    workflow = store.get_workflow(workflow_id, tenant_id)
    if not workflow:
        raise ValueError(f"Workflow not found: {workflow_id}")

    execution_id = f"exec_{uuid.uuid4().hex[:12]}"

    # Store execution state - typed as dict[str, Any] to accommodate mixed value types on update
    execution: dict[str, Any] = {
        "id": execution_id,
        "workflow_id": workflow_id,
        "tenant_id": tenant_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "inputs": inputs or {},
    }
    store.save_execution(execution)

    try:
        result = await _get_engine().execute(workflow, inputs, execution_id)

        execution.update(
            {
                "status": "completed" if result.success else "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "outputs": result.final_output,
                "steps": [_step_result_to_dict(s) for s in result.steps],
                "error": result.error,
                "duration_ms": result.total_duration_ms,
            }
        )
        store.save_execution(execution)

        from aragora.server.handlers import workflows as workflows_module

        if workflows_module.audit_data is not None:
            workflows_module.audit_data(
                user_id="system",
                resource_type="workflow_execution",
                resource_id=execution_id,
                action="execute",
                workflow_id=workflow_id,
                status=execution["status"],
                tenant_id=tenant_id,
            )

        return execution

    except (ValueError, KeyError, TypeError) as e:
        logger.warning(f"Invalid workflow configuration or inputs: {e}")
        execution.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }
        )
        store.save_execution(execution)
        raise
    except (OSError, IOError) as e:
        logger.error(f"Storage error during workflow execution: {e}")
        execution.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }
        )
        store.save_execution(execution)
        raise
    except (ConnectionError, TimeoutError) as e:
        logger.error(f"Connection error during workflow execution: {e}")
        execution.update(
            {
                "status": "failed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
            }
        )
        store.save_execution(execution)
        raise


async def get_execution(execution_id: str) -> Optional[dict[str, Any]]:
    """Get execution status and result."""
    store = _get_store()
    return store.get_execution(execution_id)


async def list_executions(
    workflow_id: str | None = None,
    tenant_id: str = "default",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """List workflow executions."""
    store = _get_store()
    executions, _ = store.list_executions(
        workflow_id=workflow_id,
        tenant_id=tenant_id,
        limit=limit,
    )
    return executions


async def terminate_execution(execution_id: str) -> bool:
    """Request termination of a running execution."""
    store = _get_store()
    execution = store.get_execution(execution_id)
    if not execution:
        return False

    if execution.get("status") != "running":
        return False

    _get_engine().request_termination("User requested")
    execution["status"] = "terminated"
    execution["completed_at"] = datetime.now(timezone.utc).isoformat()
    store.save_execution(execution)

    return True


__all__ = [
    "execute_workflow",
    "get_execution",
    "list_executions",
    "terminate_execution",
]
