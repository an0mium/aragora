"""
Canvas State Manager.

Manages canvas state, handles operations, and broadcasts updates
to connected clients via WebSocket for real-time collaboration.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from .models import (
    Canvas,
    CanvasEdge,
    CanvasEvent,
    CanvasEventType,
    CanvasNode,
    CanvasNodeType,
    EdgeType,
    Position,
)

logger = logging.getLogger(__name__)


class CanvasStateManager:
    """
    Manages canvas state and operations for real-time collaboration.

    Handles:
    - Canvas CRUD operations
    - Node and edge management
    - Event broadcasting to connected clients
    - Undo/redo history
    - Collaborative editing with user selections
    """

    def __init__(self, max_history: int = 100):
        """
        Initialize the canvas state manager.

        Args:
            max_history: Maximum undo history size per canvas
        """
        self._canvases: Dict[str, Canvas] = {}
        self._subscribers: Dict[str, Set[Callable[[CanvasEvent], Coroutine[Any, Any, None]]]] = {}
        self._user_selections: Dict[
            str, Dict[str, Set[str]]
        ] = {}  # canvas_id -> user_id -> node_ids
        self._history: Dict[str, List[CanvasEvent]] = {}  # canvas_id -> events
        self._max_history = max_history
        self._lock = asyncio.Lock()

    # =========================================================================
    # Canvas Management
    # =========================================================================

    async def create_canvas(
        self,
        canvas_id: Optional[str] = None,
        name: str = "Untitled Canvas",
        owner_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        **metadata: Any,
    ) -> Canvas:
        """Create a new canvas."""
        async with self._lock:
            if canvas_id is None:
                canvas_id = str(uuid.uuid4())

            canvas = Canvas(
                id=canvas_id,
                name=name,
                owner_id=owner_id,
                workspace_id=workspace_id,
                metadata=metadata,
            )
            self._canvases[canvas_id] = canvas
            self._subscribers[canvas_id] = set()
            self._history[canvas_id] = []
            logger.info(f"Created canvas: {canvas_id} ({name})")
            return canvas

    async def get_canvas(self, canvas_id: str) -> Optional[Canvas]:
        """Get a canvas by ID."""
        return self._canvases.get(canvas_id)

    async def get_or_create_canvas(
        self,
        canvas_id: str,
        name: str = "Untitled Canvas",
        **kwargs: Any,
    ) -> Canvas:
        """Get an existing canvas or create a new one."""
        canvas = self._canvases.get(canvas_id)
        if canvas is None:
            canvas = await self.create_canvas(canvas_id=canvas_id, name=name, **kwargs)
        return canvas

    async def delete_canvas(self, canvas_id: str) -> bool:
        """Delete a canvas."""
        async with self._lock:
            if canvas_id in self._canvases:
                del self._canvases[canvas_id]
                self._subscribers.pop(canvas_id, None)
                self._history.pop(canvas_id, None)
                self._user_selections.pop(canvas_id, None)
                logger.info(f"Deleted canvas: {canvas_id}")
                return True
            return False

    async def list_canvases(
        self,
        owner_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> List[Canvas]:
        """List canvases, optionally filtered by owner or workspace."""
        canvases = list(self._canvases.values())

        if owner_id:
            canvases = [c for c in canvases if c.owner_id == owner_id]
        if workspace_id:
            canvases = [c for c in canvases if c.workspace_id == workspace_id]

        return canvases

    # =========================================================================
    # Node Operations
    # =========================================================================

    async def add_node(
        self,
        canvas_id: str,
        node_type: CanvasNodeType,
        position: Position,
        label: str = "",
        data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[CanvasNode]:
        """Add a node to the canvas."""
        canvas = self._canvases.get(canvas_id)
        if not canvas:
            return None

        node = canvas.add_node(
            node_type=node_type,
            position=position,
            label=label,
            data=data,
            **kwargs,
        )

        # Broadcast event
        event = CanvasEvent(
            event_type=CanvasEventType.NODE_CREATE,
            canvas_id=canvas_id,
            node_id=node.id,
            user_id=user_id,
            data=node.to_dict(),
        )
        await self._broadcast(canvas_id, event)
        self._add_to_history(canvas_id, event)

        return node

    async def update_node(
        self,
        canvas_id: str,
        node_id: str,
        user_id: Optional[str] = None,
        **updates: Any,
    ) -> Optional[CanvasNode]:
        """Update a node's properties."""
        canvas = self._canvases.get(canvas_id)
        if not canvas:
            return None

        node = canvas.get_node(node_id)
        if not node:
            return None

        # Apply updates
        if "label" in updates:
            node.label = updates["label"]
        if "data" in updates:
            node.data.update(updates["data"])
        if "style" in updates:
            node.style.update(updates["style"])
        if "locked" in updates:
            node.locked = updates["locked"]

        node.updated_at = datetime.now(timezone.utc)

        # Broadcast event
        event = CanvasEvent(
            event_type=CanvasEventType.NODE_UPDATE,
            canvas_id=canvas_id,
            node_id=node_id,
            user_id=user_id,
            data={"updates": updates, "node": node.to_dict()},
        )
        await self._broadcast(canvas_id, event)
        self._add_to_history(canvas_id, event)

        return node

    async def move_node(
        self,
        canvas_id: str,
        node_id: str,
        x: float,
        y: float,
        user_id: Optional[str] = None,
    ) -> Optional[CanvasNode]:
        """Move a node to a new position."""
        canvas = self._canvases.get(canvas_id)
        if not canvas:
            return None

        node = canvas.get_node(node_id)
        if not node:
            return None

        if node.locked:
            return None

        old_position = node.position.to_dict()
        node.move(x, y)

        # Broadcast event
        event = CanvasEvent(
            event_type=CanvasEventType.NODE_MOVE,
            canvas_id=canvas_id,
            node_id=node_id,
            user_id=user_id,
            data={
                "old_position": old_position,
                "new_position": node.position.to_dict(),
            },
        )
        await self._broadcast(canvas_id, event)
        self._add_to_history(canvas_id, event)

        return node

    async def resize_node(
        self,
        canvas_id: str,
        node_id: str,
        width: float,
        height: float,
        user_id: Optional[str] = None,
    ) -> Optional[CanvasNode]:
        """Resize a node."""
        canvas = self._canvases.get(canvas_id)
        if not canvas:
            return None

        node = canvas.get_node(node_id)
        if not node:
            return None

        if node.locked:
            return None

        old_size = node.size.to_dict()
        node.resize(width, height)

        # Broadcast event
        event = CanvasEvent(
            event_type=CanvasEventType.NODE_RESIZE,
            canvas_id=canvas_id,
            node_id=node_id,
            user_id=user_id,
            data={
                "old_size": old_size,
                "new_size": node.size.to_dict(),
            },
        )
        await self._broadcast(canvas_id, event)
        self._add_to_history(canvas_id, event)

        return node

    async def delete_node(
        self,
        canvas_id: str,
        node_id: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """Delete a node from the canvas."""
        canvas = self._canvases.get(canvas_id)
        if not canvas:
            return False

        node = canvas.remove_node(node_id)
        if not node:
            return False

        # Broadcast event
        event = CanvasEvent(
            event_type=CanvasEventType.NODE_DELETE,
            canvas_id=canvas_id,
            node_id=node_id,
            user_id=user_id,
            data={"node": node.to_dict()},
        )
        await self._broadcast(canvas_id, event)
        self._add_to_history(canvas_id, event)

        return True

    async def select_node(
        self,
        canvas_id: str,
        node_id: str,
        user_id: str,
        multi_select: bool = False,
    ) -> None:
        """Select a node (for collaborative cursors)."""
        if canvas_id not in self._user_selections:
            self._user_selections[canvas_id] = {}

        if user_id not in self._user_selections[canvas_id]:
            self._user_selections[canvas_id][user_id] = set()

        if not multi_select:
            self._user_selections[canvas_id][user_id].clear()

        self._user_selections[canvas_id][user_id].add(node_id)

        # Broadcast event
        event = CanvasEvent(
            event_type=CanvasEventType.NODE_SELECT,
            canvas_id=canvas_id,
            node_id=node_id,
            user_id=user_id,
            data={
                "selected_nodes": list(self._user_selections[canvas_id][user_id]),
                "multi_select": multi_select,
            },
        )
        await self._broadcast(canvas_id, event)

    # =========================================================================
    # Edge Operations
    # =========================================================================

    async def add_edge(
        self,
        canvas_id: str,
        source_id: str,
        target_id: str,
        edge_type: EdgeType = EdgeType.DEFAULT,
        label: str = "",
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[CanvasEdge]:
        """Add an edge between two nodes."""
        canvas = self._canvases.get(canvas_id)
        if not canvas:
            return None

        edge = canvas.add_edge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            label=label,
            **kwargs,
        )
        if not edge:
            return None

        # Broadcast event
        event = CanvasEvent(
            event_type=CanvasEventType.EDGE_CREATE,
            canvas_id=canvas_id,
            edge_id=edge.id,
            user_id=user_id,
            data=edge.to_dict(),
        )
        await self._broadcast(canvas_id, event)
        self._add_to_history(canvas_id, event)

        return edge

    async def update_edge(
        self,
        canvas_id: str,
        edge_id: str,
        user_id: Optional[str] = None,
        **updates: Any,
    ) -> Optional[CanvasEdge]:
        """Update an edge's properties."""
        canvas = self._canvases.get(canvas_id)
        if not canvas:
            return None

        edge = canvas.get_edge(edge_id)
        if not edge:
            return None

        # Apply updates
        if "label" in updates:
            edge.label = updates["label"]
        if "data" in updates:
            edge.data.update(updates["data"])
        if "style" in updates:
            edge.style.update(updates["style"])
        if "animated" in updates:
            edge.animated = updates["animated"]

        # Broadcast event
        event = CanvasEvent(
            event_type=CanvasEventType.EDGE_UPDATE,
            canvas_id=canvas_id,
            edge_id=edge_id,
            user_id=user_id,
            data={"updates": updates, "edge": edge.to_dict()},
        )
        await self._broadcast(canvas_id, event)
        self._add_to_history(canvas_id, event)

        return edge

    async def delete_edge(
        self,
        canvas_id: str,
        edge_id: str,
        user_id: Optional[str] = None,
    ) -> bool:
        """Delete an edge from the canvas."""
        canvas = self._canvases.get(canvas_id)
        if not canvas:
            return False

        edge = canvas.remove_edge(edge_id)
        if not edge:
            return False

        # Broadcast event
        event = CanvasEvent(
            event_type=CanvasEventType.EDGE_DELETE,
            canvas_id=canvas_id,
            edge_id=edge_id,
            user_id=user_id,
            data={"edge": edge.to_dict()},
        )
        await self._broadcast(canvas_id, event)
        self._add_to_history(canvas_id, event)

        return True

    # =========================================================================
    # Actions
    # =========================================================================

    async def execute_action(
        self,
        canvas_id: str,
        action: str,
        params: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a canvas action.

        Actions can trigger debates, workflows, or other operations.
        """
        canvas = self._canvases.get(canvas_id)
        if not canvas:
            return {"success": False, "error": "Canvas not found"}

        # Broadcast action event
        action_event = CanvasEvent(
            event_type=CanvasEventType.ACTION,
            canvas_id=canvas_id,
            user_id=user_id,
            data={"action": action, "params": params},
        )
        await self._broadcast(canvas_id, action_event)

        result: Dict[str, Any] = {"success": True, "action": action}

        # Handle built-in actions
        if action == "start_debate":
            result = await self._handle_start_debate(canvas, params, user_id)
        elif action == "run_workflow":
            result = await self._handle_run_workflow(canvas, params, user_id)
        elif action == "query_knowledge":
            result = await self._handle_query_knowledge(canvas, params, user_id)
        elif action == "clear_canvas":
            canvas.clear()
            result = {"success": True, "action": action}
        else:
            result = {"success": False, "error": f"Unknown action: {action}"}

        # Broadcast result event
        result_event = CanvasEvent(
            event_type=CanvasEventType.ACTION_RESULT,
            canvas_id=canvas_id,
            user_id=user_id,
            data={"action": action, "result": result},
        )
        await self._broadcast(canvas_id, result_event)

        return result

    async def _handle_start_debate(
        self,
        canvas: Canvas,
        params: Dict[str, Any],
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Handle starting a debate from the canvas."""
        question = params.get("question", "")
        if not question:
            return {"success": False, "error": "Question is required"}

        # Create a debate node
        debate_node = canvas.add_node(
            node_type=CanvasNodeType.DEBATE,
            position=Position(params.get("x", 100), params.get("y", 100)),
            label=question[:50] + "..." if len(question) > 50 else question,
            data={"question": question, "status": "pending"},
        )

        # Broadcast debate start event
        event = CanvasEvent(
            event_type=CanvasEventType.DEBATE_START,
            canvas_id=canvas.id,
            node_id=debate_node.id,
            user_id=user_id,
            data={"question": question, "node_id": debate_node.id},
        )
        await self._broadcast(canvas.id, event)

        # Actually run the debate
        try:
            from aragora.core import Environment, DebateProtocol
            from aragora.debate.orchestrator import Arena

            # Update node status to running
            debate_node.data["status"] = "running"
            await self._broadcast(
                canvas.id,
                CanvasEvent(
                    event_type=CanvasEventType.NODE_UPDATED,
                    canvas_id=canvas.id,
                    node_id=debate_node.id,
                    data={"status": "running"},
                ),
            )

            # Create environment and run debate
            env = Environment(task=question)
            protocol = DebateProtocol(
                rounds=params.get("rounds", 3),
                consensus=params.get("consensus", "majority"),
            )

            # Get agents - use configured defaults or from params
            agents = await self._get_debate_agents(params.get("agents"))

            arena = Arena(env, agents, protocol)
            result = await arena.run()

            # Update node with results
            debate_node.data["status"] = "completed"
            debate_node.data["result"] = {
                "decision": result.decision if hasattr(result, "decision") else str(result),
                "consensus_reached": getattr(result, "consensus_reached", False),
                "rounds_used": getattr(result, "rounds_used", protocol.rounds),
            }

            # Broadcast completion
            await self._broadcast(
                canvas.id,
                CanvasEvent(
                    event_type=CanvasEventType.DEBATE_END,
                    canvas_id=canvas.id,
                    node_id=debate_node.id,
                    data=debate_node.data,
                ),
            )

            return {
                "success": True,
                "action": "start_debate",
                "debate_node_id": debate_node.id,
                "result": debate_node.data["result"],
            }

        except ImportError as e:
            logger.warning(f"Debate modules not available: {e}")
            debate_node.data["status"] = "error"
            debate_node.data["error"] = "Debate modules not available"
            return {
                "success": False,
                "action": "start_debate",
                "debate_node_id": debate_node.id,
                "error": "Debate modules not available",
            }
        except Exception as e:
            logger.error(f"Debate execution failed: {e}")
            debate_node.data["status"] = "error"
            debate_node.data["error"] = str(e)
            await self._broadcast(
                canvas.id,
                CanvasEvent(
                    event_type=CanvasEventType.ERROR,
                    canvas_id=canvas.id,
                    node_id=debate_node.id,
                    data={"error": str(e)},
                ),
            )
            return {
                "success": False,
                "action": "start_debate",
                "debate_node_id": debate_node.id,
                "error": str(e),
            }

    async def _get_debate_agents(self, agent_config: Optional[List[str]] = None):
        """Get agents for debate, using defaults if not specified."""
        try:
            from aragora.agents.cli_agents import get_agent

            if agent_config:
                return [get_agent(name) for name in agent_config if get_agent(name)]

            # Default agents
            default_agents = ["claude", "gpt4"]
            agents = []
            for name in default_agents:
                agent = get_agent(name)
                if agent:
                    agents.append(agent)

            if not agents:
                # Fallback to any available agent
                from aragora.agents.registry import get_available_agents

                available = get_available_agents()
                if available:
                    agents = [available[0]]

            return agents
        except ImportError:
            return []

    async def _handle_run_workflow(
        self,
        canvas: Canvas,
        params: Dict[str, Any],
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Handle running a workflow from the canvas."""
        workflow_id = params.get("workflow_id")
        workflow_definition = params.get("definition")

        if not workflow_id and not workflow_definition:
            return {"success": False, "error": "workflow_id or definition is required"}

        # Create workflow node
        workflow_node = canvas.add_node(
            node_type=CanvasNodeType.WORKFLOW,
            position=Position(params.get("x", 100), params.get("y", 100)),
            label=f"Workflow: {workflow_id or 'custom'}",
            data={"workflow_id": workflow_id, "status": "pending"},
        )

        # Actually run the workflow
        try:
            from aragora.workflow.engine import WorkflowEngine
            from aragora.workflow.models import WorkflowDefinition

            # Update status
            workflow_node.data["status"] = "running"
            await self._broadcast(
                canvas.id,
                CanvasEvent(
                    event_type=CanvasEventType.NODE_UPDATED,
                    canvas_id=canvas.id,
                    node_id=workflow_node.id,
                    data={"status": "running"},
                ),
            )

            # Load or create workflow definition
            if workflow_definition:
                definition = WorkflowDefinition(**workflow_definition)
            elif workflow_id:
                # Try to load from templates
                definition = await self._load_workflow_definition(workflow_id)
                if not definition:
                    workflow_node.data["status"] = "error"
                    return {
                        "success": False,
                        "error": f"Workflow '{workflow_id}' not found",
                    }
            else:
                return {"success": False, "error": "No workflow definition provided"}

            # Execute workflow
            engine = WorkflowEngine()
            inputs = params.get("inputs", {})
            result = await engine.execute(definition, inputs, workflow_id or "canvas-workflow")

            # Update node with results
            workflow_node.data["status"] = "completed"
            workflow_node.data["result"] = {
                "success": getattr(result, "success", True),
                "outputs": getattr(result, "outputs", {}),
            }

            await self._broadcast(
                canvas.id,
                CanvasEvent(
                    event_type=CanvasEventType.NODE_UPDATED,
                    canvas_id=canvas.id,
                    node_id=workflow_node.id,
                    data=workflow_node.data,
                ),
            )

            return {
                "success": True,
                "action": "run_workflow",
                "workflow_node_id": workflow_node.id,
                "result": workflow_node.data["result"],
            }

        except ImportError as e:
            logger.warning(f"Workflow modules not available: {e}")
            workflow_node.data["status"] = "error"
            return {
                "success": False,
                "action": "run_workflow",
                "workflow_node_id": workflow_node.id,
                "error": "Workflow modules not available",
            }
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            workflow_node.data["status"] = "error"
            workflow_node.data["error"] = str(e)
            return {
                "success": False,
                "action": "run_workflow",
                "workflow_node_id": workflow_node.id,
                "error": str(e),
            }

    async def _load_workflow_definition(self, workflow_id: str):
        """Load a workflow definition by ID."""
        try:
            from aragora.workflow.templates import get_template

            template = get_template(workflow_id)
            if template:
                return template.to_definition()
            return None
        except ImportError:
            return None

    async def _handle_query_knowledge(
        self,
        canvas: Canvas,
        params: Dict[str, Any],
        user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Handle querying knowledge from the canvas."""
        query = params.get("query", "")
        if not query:
            return {"success": False, "error": "Query is required"}

        # Create knowledge node
        knowledge_node = canvas.add_node(
            node_type=CanvasNodeType.KNOWLEDGE,
            position=Position(params.get("x", 100), params.get("y", 100)),
            label=f"Query: {query[:30]}..." if len(query) > 30 else f"Query: {query}",
            data={"query": query, "status": "pending"},
        )

        # Actually query knowledge
        try:
            # Update status
            knowledge_node.data["status"] = "searching"
            await self._broadcast(
                canvas.id,
                CanvasEvent(
                    event_type=CanvasEventType.NODE_UPDATED,
                    canvas_id=canvas.id,
                    node_id=knowledge_node.id,
                    data={"status": "searching"},
                ),
            )

            # Try to query knowledge mound
            results = await self._query_knowledge_mound(
                query,
                limit=params.get("limit", 10),
                min_confidence=params.get("min_confidence", 0.0),
            )

            # Update node with results
            knowledge_node.data["status"] = "completed"
            knowledge_node.data["results"] = results
            knowledge_node.data["result_count"] = len(results)

            await self._broadcast(
                canvas.id,
                CanvasEvent(
                    event_type=CanvasEventType.NODE_UPDATED,
                    canvas_id=canvas.id,
                    node_id=knowledge_node.id,
                    data=knowledge_node.data,
                ),
            )

            return {
                "success": True,
                "action": "query_knowledge",
                "knowledge_node_id": knowledge_node.id,
                "results": results,
                "count": len(results),
            }

        except ImportError as e:
            logger.warning(f"Knowledge modules not available: {e}")
            knowledge_node.data["status"] = "error"
            return {
                "success": False,
                "action": "query_knowledge",
                "knowledge_node_id": knowledge_node.id,
                "error": "Knowledge modules not available",
            }
        except Exception as e:
            logger.error(f"Knowledge query failed: {e}")
            knowledge_node.data["status"] = "error"
            knowledge_node.data["error"] = str(e)
            return {
                "success": False,
                "action": "query_knowledge",
                "knowledge_node_id": knowledge_node.id,
                "error": str(e),
            }

    async def _query_knowledge_mound(
        self,
        query: str,
        limit: int = 10,
        min_confidence: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Query the knowledge mound for relevant information."""
        try:
            from aragora.knowledge.mound.core import KnowledgeMound

            mound = KnowledgeMound()
            results = await mound.search(query, limit=limit)

            # Format results
            formatted = []
            for result in results:
                confidence = getattr(result, "confidence", 0.0)
                if confidence >= min_confidence:
                    formatted.append(
                        {
                            "id": getattr(result, "id", str(uuid.uuid4())),
                            "content": getattr(result, "content", str(result)),
                            "confidence": confidence,
                            "source": getattr(result, "source", "knowledge_mound"),
                        }
                    )

            return formatted
        except ImportError:
            # Fallback: try knowledge bridges
            try:
                from aragora.knowledge.bridges import KnowledgeBridgeHub

                hub = KnowledgeBridgeHub()
                results = await hub.search(query, limit=limit)
                return [
                    {
                        "id": str(i),
                        "content": str(r),
                        "confidence": 0.5,
                        "source": "knowledge_bridge",
                    }
                    for i, r in enumerate(results)
                ]
            except ImportError:
                return []

    # =========================================================================
    # Subscription Management
    # =========================================================================

    async def subscribe(
        self,
        canvas_id: str,
        callback: Callable[[CanvasEvent], Coroutine[Any, Any, None]],
    ) -> bool:
        """Subscribe to canvas events."""
        if canvas_id not in self._subscribers:
            self._subscribers[canvas_id] = set()

        self._subscribers[canvas_id].add(callback)
        logger.debug(f"Subscriber added to canvas {canvas_id}")
        return True

    async def unsubscribe(
        self,
        canvas_id: str,
        callback: Callable[[CanvasEvent], Coroutine[Any, Any, None]],
    ) -> bool:
        """Unsubscribe from canvas events."""
        if canvas_id in self._subscribers:
            self._subscribers[canvas_id].discard(callback)
            logger.debug(f"Subscriber removed from canvas {canvas_id}")
            return True
        return False

    async def _broadcast(self, canvas_id: str, event: CanvasEvent) -> None:
        """Broadcast an event to all subscribers."""
        subscribers = self._subscribers.get(canvas_id, set())
        if not subscribers:
            return

        # Send to all subscribers concurrently
        tasks = [callback(event) for callback in subscribers]
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"Error broadcasting event: {result}")

    # =========================================================================
    # History Management
    # =========================================================================

    def _add_to_history(self, canvas_id: str, event: CanvasEvent) -> None:
        """Add an event to the history for undo support."""
        if canvas_id not in self._history:
            self._history[canvas_id] = []

        self._history[canvas_id].append(event)

        # Trim history if too long
        if len(self._history[canvas_id]) > self._max_history:
            self._history[canvas_id] = self._history[canvas_id][-self._max_history :]

    async def get_history(
        self,
        canvas_id: str,
        limit: int = 50,
    ) -> List[CanvasEvent]:
        """Get recent history for a canvas."""
        history = self._history.get(canvas_id, [])
        return history[-limit:] if limit else history

    # =========================================================================
    # State Sync
    # =========================================================================

    async def get_state(self, canvas_id: str) -> Optional[Dict[str, Any]]:
        """Get the full state of a canvas for sync."""
        canvas = self._canvases.get(canvas_id)
        if not canvas:
            return None

        return {
            "canvas": canvas.to_dict(),
            "selections": {
                user_id: list(nodes)
                for user_id, nodes in self._user_selections.get(canvas_id, {}).items()
            },
        }

    async def sync_state(
        self,
        canvas_id: str,
        user_id: Optional[str] = None,
    ) -> None:
        """Broadcast current state to all subscribers."""
        state = await self.get_state(canvas_id)
        if not state:
            return

        event = CanvasEvent(
            event_type=CanvasEventType.STATE,
            canvas_id=canvas_id,
            user_id=user_id,
            data=state,
        )
        await self._broadcast(canvas_id, event)


# Global manager instance
_manager: Optional[CanvasStateManager] = None


def get_canvas_manager() -> CanvasStateManager:
    """Get or create the global canvas state manager."""
    global _manager
    if _manager is None:
        _manager = CanvasStateManager()
    return _manager


__all__ = [
    "CanvasStateManager",
    "get_canvas_manager",
]
