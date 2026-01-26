"""
MCP Canvas Tools.

Provides tools for canvas manipulation:
- canvas_create: Create a new canvas
- canvas_get: Get canvas state
- canvas_add_node: Add a node to canvas
- canvas_add_edge: Add an edge between nodes
- canvas_execute_action: Execute canvas action (start_debate, run_workflow)
- canvas_list: List available canvases
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Global manager instance for session persistence
_manager = None


async def _get_manager():
    """Get or create the canvas manager instance."""
    global _manager
    if _manager is None:
        from aragora.canvas.manager import CanvasStateManager

        _manager = CanvasStateManager()
    return _manager


async def canvas_create_tool(
    name: str = "Untitled Canvas",
    description: str = "",
    owner_id: str = "",
    workspace_id: str = "",
) -> Dict[str, Any]:
    """
    Create a new canvas.

    Args:
        name: Canvas name
        description: Canvas description
        owner_id: Owner user ID
        workspace_id: Workspace ID

    Returns:
        Dict with canvas_id and name
    """
    try:
        manager = await _get_manager()
        canvas = await manager.create_canvas(
            name=name,
            owner_id=owner_id if owner_id else None,
            workspace_id=workspace_id if workspace_id else None,
        )
        return {
            "success": True,
            "canvas_id": canvas.id,
            "name": canvas.name,
        }
    except ImportError:
        return {"success": False, "error": "Canvas module not available"}
    except Exception as e:
        logger.error(f"Canvas creation failed: {e}")
        return {"success": False, "error": str(e)}


async def canvas_get_tool(
    canvas_id: str,
) -> Dict[str, Any]:
    """
    Get canvas state.

    Args:
        canvas_id: Canvas ID to retrieve

    Returns:
        Dict with canvas data including nodes and edges
    """
    if not canvas_id:
        return {"error": "canvas_id is required"}

    try:
        manager = await _get_manager()
        canvas = await manager.get_canvas(canvas_id)

        if not canvas:
            return {"error": f"Canvas '{canvas_id}' not found"}

        return {
            "success": True,
            "canvas": {
                "id": canvas.id,
                "name": canvas.name,
                "nodes": [n.to_dict() for n in canvas.nodes.values()],
                "edges": [e.to_dict() for e in canvas.edges.values()],
                "node_count": len(canvas.nodes),
                "edge_count": len(canvas.edges),
            },
        }
    except ImportError:
        return {"error": "Canvas module not available"}
    except Exception as e:
        logger.error(f"Canvas get failed: {e}")
        return {"error": str(e)}


async def canvas_add_node_tool(
    canvas_id: str,
    node_type: str = "text",
    label: str = "",
    x: int = 100,
    y: int = 100,
    data: str = "{}",
) -> Dict[str, Any]:
    """
    Add a node to a canvas.

    Args:
        canvas_id: Canvas ID
        node_type: Node type (text, agent, debate, knowledge, workflow, browser, input, output)
        label: Node label
        x: X position
        y: Y position
        data: JSON string of node data

    Returns:
        Dict with node_id
    """
    if not canvas_id:
        return {"error": "canvas_id is required"}

    try:
        from aragora.canvas.models import CanvasNodeType, Position

        manager = await _get_manager()

        # Parse node type
        try:
            node_type_enum = CanvasNodeType[node_type.upper()]
        except KeyError:
            node_type_enum = CanvasNodeType.TEXT

        # Parse data
        try:
            node_data = json.loads(data) if data else {}
        except json.JSONDecodeError:
            node_data = {}

        node = await manager.add_node(
            canvas_id=canvas_id,
            node_type=node_type_enum,
            position=Position(x, y),
            label=label,
            data=node_data,
        )

        if node:
            return {
                "success": True,
                "node_id": node.id,
                "node_type": node.node_type.value,
            }
        return {"success": False, "error": "Failed to add node"}

    except ImportError:
        return {"error": "Canvas module not available"}
    except Exception as e:
        logger.error(f"Canvas add_node failed: {e}")
        return {"error": str(e)}


async def canvas_add_edge_tool(
    canvas_id: str,
    source_id: str,
    target_id: str,
    edge_type: str = "default",
    label: str = "",
) -> Dict[str, Any]:
    """
    Add an edge between nodes.

    Args:
        canvas_id: Canvas ID
        source_id: Source node ID
        target_id: Target node ID
        edge_type: Edge type (default, data_flow, control_flow, reference, dependency)
        label: Edge label

    Returns:
        Dict with edge_id
    """
    if not canvas_id or not source_id or not target_id:
        return {"error": "canvas_id, source_id, and target_id are required"}

    try:
        from aragora.canvas.models import EdgeType

        manager = await _get_manager()

        # Parse edge type
        try:
            edge_type_enum = EdgeType[edge_type.upper()]
        except KeyError:
            edge_type_enum = EdgeType.DEFAULT

        edge = await manager.add_edge(
            canvas_id=canvas_id,
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type_enum,
            label=label,
        )

        if edge:
            return {
                "success": True,
                "edge_id": edge.id,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
            }
        return {"success": False, "error": "Failed to add edge"}

    except ImportError:
        return {"error": "Canvas module not available"}
    except Exception as e:
        logger.error(f"Canvas add_edge failed: {e}")
        return {"error": str(e)}


async def canvas_execute_action_tool(
    canvas_id: str,
    action: str,
    params: str = "{}",
) -> Dict[str, Any]:
    """
    Execute a canvas action.

    Args:
        canvas_id: Canvas ID
        action: Action to execute (start_debate, run_workflow, query_knowledge, clear_canvas)
        params: JSON string of action parameters

    Returns:
        Dict with action result
    """
    if not canvas_id or not action:
        return {"error": "canvas_id and action are required"}

    try:
        manager = await _get_manager()

        # Parse params
        try:
            params_dict = json.loads(params) if params else {}
        except json.JSONDecodeError:
            params_dict = {}

        result = await manager.execute_action(
            canvas_id=canvas_id,
            action=action,
            params=params_dict,
        )

        return result

    except ImportError:
        return {"error": "Canvas module not available"}
    except Exception as e:
        logger.error(f"Canvas execute_action failed: {e}")
        return {"error": str(e)}


async def canvas_list_tool(
    owner_id: str = "",
    workspace_id: str = "",
    limit: int = 20,
) -> Dict[str, Any]:
    """
    List available canvases.

    Args:
        owner_id: Filter by owner
        workspace_id: Filter by workspace
        limit: Max canvases to return

    Returns:
        Dict with canvas list
    """
    limit = min(max(limit, 1), 100)

    try:
        manager = await _get_manager()
        canvases = await manager.list_canvases(
            owner_id=owner_id if owner_id else None,
            workspace_id=workspace_id if workspace_id else None,
        )

        # Apply limit
        canvases = canvases[:limit]

        return {
            "canvases": [
                {
                    "id": c.id,
                    "name": c.name,
                    "node_count": len(c.nodes),
                    "edge_count": len(c.edges),
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                }
                for c in canvases
            ],
            "count": len(canvases),
        }

    except ImportError:
        return {"canvases": [], "count": 0, "error": "Canvas module not available"}
    except Exception as e:
        logger.error(f"Canvas list failed: {e}")
        return {"canvases": [], "count": 0, "error": str(e)}


async def canvas_delete_node_tool(
    canvas_id: str,
    node_id: str,
) -> Dict[str, Any]:
    """
    Delete a node from canvas.

    Args:
        canvas_id: Canvas ID
        node_id: Node ID to delete

    Returns:
        Dict with success status
    """
    if not canvas_id or not node_id:
        return {"error": "canvas_id and node_id are required"}

    try:
        manager = await _get_manager()
        success = await manager.delete_node(canvas_id=canvas_id, node_id=node_id)

        return {"success": success}

    except ImportError:
        return {"error": "Canvas module not available"}
    except Exception as e:
        logger.error(f"Canvas delete_node failed: {e}")
        return {"error": str(e)}


__all__ = [
    "canvas_create_tool",
    "canvas_get_tool",
    "canvas_add_node_tool",
    "canvas_add_edge_tool",
    "canvas_execute_action_tool",
    "canvas_list_tool",
    "canvas_delete_node_tool",
]
