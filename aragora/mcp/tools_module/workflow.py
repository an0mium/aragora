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
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


async def run_workflow_tool(
    template: str,
    inputs: str = "",
    async_execution: bool = False,
    timeout_seconds: int = 300,
) -> Dict[str, Any]:
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
        from aragora.workflow.engine import WorkflowEngine
        from aragora.workflow.patterns import get_workflow_template

        # Parse inputs
        workflow_inputs = {}
        if inputs:
            try:
                workflow_inputs = json_module.loads(inputs)
            except json_module.JSONDecodeError:
                return {"error": "Invalid JSON in inputs parameter"}

        # Get template
        template_def = get_workflow_template(template)
        if not template_def:
            return {"error": f"Workflow template '{template}' not found"}

        # Create engine and run
        engine = WorkflowEngine()

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
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        return {"error": f"Execution failed: {str(e)}"}


async def get_workflow_status_tool(
    execution_id: str,
) -> Dict[str, Any]:
    """
    Get the status of a workflow execution.

    Args:
        execution_id: The workflow execution ID

    Returns:
        Dict with workflow status and progress
    """
    try:
        from aragora.workflow.engine import WorkflowEngine

        engine = WorkflowEngine()
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
    except Exception as e:
        logger.error(f"Failed to get workflow status: {e}")
        return {"error": f"Status check failed: {str(e)}"}


async def list_workflow_templates_tool(
    category: str = "all",
) -> Dict[str, Any]:
    """
    List available workflow templates.

    Args:
        category: Filter by category (all, debate, audit, analysis, automation)

    Returns:
        Dict with list of templates
    """
    try:
        from aragora.workflow.patterns import list_workflow_templates

        templates = list_workflow_templates(category if category != "all" else None)

        return {
            "templates": [
                {
                    "name": t.name,
                    "description": t.description,
                    "category": t.category,
                    "inputs": list(t.inputs.keys()) if t.inputs else [],
                    "outputs": list(t.outputs.keys()) if t.outputs else [],
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
    except Exception as e:
        logger.error(f"Failed to list templates: {e}")
        return {"error": f"List failed: {str(e)}"}


async def cancel_workflow_tool(
    execution_id: str,
    reason: str = "",
) -> Dict[str, Any]:
    """
    Cancel a running workflow execution.

    Args:
        execution_id: The workflow execution ID to cancel
        reason: Optional reason for cancellation

    Returns:
        Dict with cancellation result
    """
    try:
        from aragora.workflow.engine import WorkflowEngine

        engine = WorkflowEngine()
        success = await engine.cancel(execution_id, reason=reason)

        return {
            "execution_id": execution_id,
            "cancelled": success,
            "reason": reason or "User requested",
        }

    except ImportError:
        logger.warning("Workflow engine not available")
        return {"error": "Workflow engine module not available"}
    except Exception as e:
        logger.error(f"Failed to cancel workflow: {e}")
        return {"error": f"Cancel failed: {str(e)}"}


# Export all tools
__all__ = [
    "run_workflow_tool",
    "get_workflow_status_tool",
    "list_workflow_templates_tool",
    "cancel_workflow_tool",
]
