"""
MCP Tools for Workflow operations.

Provides tools for executing and managing Aragora workflows:
- run_workflow: Execute a workflow
- get_workflow_status: Get workflow execution status
- list_workflow_templates: List available workflow templates
- cancel_workflow: Cancel a running workflow
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class WorkflowExecutionStatus:
    """Status of a workflow execution."""

    status: str
    progress: float
    current_node: str | None
    started_at: str | None
    completed_at: str | None
    error: str | None = None


class AsyncWorkflowEngine(Protocol):
    """Protocol for async workflow engine methods used by MCP tools."""

    async def start_async(
        self,
        template: Any,
        inputs: dict[str, Any],
    ) -> str:
        """Start async execution and return execution ID."""
        ...

    async def run(
        self,
        template: Any,
        inputs: dict[str, Any],
        timeout: int,
    ) -> Any:
        """Run workflow synchronously."""
        ...

    async def get_status(self, execution_id: str) -> WorkflowExecutionStatus | None:
        """Get execution status."""
        ...

    async def cancel(self, execution_id: str, reason: str = "") -> bool:
        """Cancel workflow execution."""
        ...


async def run_workflow_tool(
    template: str,
    inputs: str = "",
    async_execution: bool = False,
    timeout_seconds: int = 300,
) -> dict[str, Any]:
    """
    Execute a workflow from a template.

    Args:
        template: Workflow template name or ID
        inputs: JSON string of workflow inputs
        async_execution: Run asynchronously (return immediately with workflow ID)
        timeout_seconds: Timeout for synchronous execution

    Returns:
        Dict with workflow result or execution ID
    """
    import json as json_module

    try:
        from typing import cast

        from aragora.workflow.engine import WorkflowEngine
        from aragora.workflow.templates import get_template

        # Parse inputs with type validation
        workflow_inputs = {}
        if inputs:
            try:
                parsed = json_module.loads(inputs)
                if not isinstance(parsed, dict):
                    return {"error": "Inputs must be a JSON object, not " + type(parsed).__name__}
                workflow_inputs = parsed
            except json_module.JSONDecodeError:
                return {"error": "Invalid JSON in inputs parameter"}

        # Get template
        template_def = get_template(template)
        if not template_def:
            return {"error": f"Workflow template '{template}' not found"}

        # Create engine and run - cast to protocol for extended methods
        engine = cast(AsyncWorkflowEngine, WorkflowEngine())

        if async_execution:
            # Start async execution
            execution_id = await engine.start_async(
                template=template_def,
                inputs=workflow_inputs,
            )
            return {
                "execution_id": execution_id,
                "status": "started",
                "template": template,
                "async": True,
            }
        else:
            # Synchronous execution
            result = await engine.run(
                template=template_def,
                inputs=workflow_inputs,
                timeout=timeout_seconds,
            )
            return {
                "status": "completed",
                "template": template,
                "outputs": result.outputs,
                "duration_seconds": result.duration_seconds,
                "nodes_executed": result.nodes_executed,
            }

    except ImportError:
        logger.warning("Workflow engine not available")
        return {"error": "Workflow engine module not available"}
    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"Workflow execution failed: {e}")
        return {"error": f"Execution failed: {str(e)}"}


async def get_workflow_status_tool(
    execution_id: str,
) -> dict[str, Any]:
    """
    Get the status of a workflow execution.

    Args:
        execution_id: The workflow execution ID

    Returns:
        Dict with workflow status and progress
    """
    try:
        from typing import cast

        from aragora.workflow.engine import WorkflowEngine

        # Cast to protocol for extended methods
        engine = cast(AsyncWorkflowEngine, WorkflowEngine())
        status = await engine.get_status(execution_id)

        if not status:
            return {"error": f"Execution {execution_id} not found"}

        return {
            "execution_id": execution_id,
            "status": status.status,
            "progress": status.progress,
            "current_node": status.current_node,
            "started_at": status.started_at,
            "completed_at": status.completed_at,
            "error": status.error if status.error else None,
        }

    except ImportError:
        logger.warning("Workflow engine not available")
        return {"error": "Workflow engine module not available"}
    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"Failed to get workflow status: {e}")
        return {"error": f"Status check failed: {str(e)}"}


async def list_workflow_templates_tool(
    category: str = "all",
) -> dict[str, Any]:
    """
    List available workflow templates.

    Args:
        category: Filter by category (all, debate, audit, analysis, automation)

    Returns:
        Dict with list of templates
    """
    try:
        from aragora.workflow.templates import list_templates

        templates = list_templates(category if category != "all" else None)

        return {
            "templates": [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "category": t["category"],
                    "inputs": list(t.get("inputs", {}).keys())
                    if isinstance(t.get("inputs"), dict)
                    else t.get("inputs", []),
                    "outputs": list(t.get("outputs", {}).keys())
                    if isinstance(t.get("outputs"), dict)
                    else t.get("outputs", []),
                }
                for t in templates
            ],
            "count": len(templates),
            "category": category,
        }

    except ImportError:
        logger.warning("Workflow patterns not available")
        # Return default templates
        return {
            "templates": [
                {
                    "name": "debate_and_audit",
                    "description": "Run a debate followed by audit",
                    "category": "debate",
                    "inputs": ["question", "documents"],
                    "outputs": ["decision", "findings"],
                },
                {
                    "name": "multi_agent_review",
                    "description": "Multi-agent document review",
                    "category": "analysis",
                    "inputs": ["documents", "review_type"],
                    "outputs": ["review_summary", "recommendations"],
                },
                {
                    "name": "evidence_gathering",
                    "description": "Gather evidence for a claim",
                    "category": "analysis",
                    "inputs": ["claim", "sources"],
                    "outputs": ["evidence", "confidence"],
                },
            ],
            "count": 3,
            "category": category,
            "note": "Default templates - workflow engine not available",
        }
    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"Failed to list templates: {e}")
        return {"error": f"List failed: {str(e)}"}


async def cancel_workflow_tool(
    execution_id: str,
    reason: str = "",
) -> dict[str, Any]:
    """
    Cancel a running workflow execution.

    Args:
        execution_id: The workflow execution ID to cancel
        reason: Optional reason for cancellation

    Returns:
        Dict with cancellation result
    """
    try:
        from typing import cast

        from aragora.workflow.engine import WorkflowEngine

        # Cast to protocol for extended methods
        engine = cast(AsyncWorkflowEngine, WorkflowEngine())
        success = await engine.cancel(execution_id, reason=reason)

        return {
            "execution_id": execution_id,
            "cancelled": success,
            "reason": reason or "User requested",
        }

    except ImportError:
        logger.warning("Workflow engine not available")
        return {"error": "Workflow engine module not available"}
    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"Failed to cancel workflow: {e}")
        return {"error": f"Cancel failed: {str(e)}"}


# Export all tools
__all__ = [
    "run_workflow_tool",
    "get_workflow_status_tool",
    "list_workflow_templates_tool",
    "cancel_workflow_tool",
]
