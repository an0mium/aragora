"""
Graph debates endpoint handlers.

Endpoints:
- POST /api/debates/graph - Run a graph-structured debate with branching
- GET /api/debates/graph/{id} - Get graph debate by ID
- GET /api/debates/graph/{id}/branches - Get all branches for a debate
- GET /api/debates/graph/{id}/nodes - Get all nodes in debate graph
"""

import logging
from typing import Any, Optional

from .base import (
    BaseHandler,
    HandlerResult,
    json_response,
    error_response,
    handle_errors,
)

logger = logging.getLogger(__name__)


class GraphDebatesHandler(BaseHandler):
    """Handler for graph debate endpoints."""

    ROUTES = [
        "/api/debates/graph",
        "/api/debates/graph/",
    ]

    AUTH_REQUIRED_ENDPOINTS = [
        "/api/debates/graph",
    ]

    @handle_errors
    async def handle_get(self, handler, path: str, query_params: dict) -> HandlerResult:
        """Handle GET requests for graph debates."""
        # Extract debate ID from path if present
        parts = path.rstrip("/").split("/")

        # GET /api/debates/graph/{id} - Get specific graph debate
        if len(parts) >= 5 and parts[3] == "graph":
            debate_id = parts[4]

            # GET /api/debates/graph/{id}/branches
            if len(parts) >= 6 and parts[5] == "branches":
                return await self._get_branches(handler, debate_id)

            # GET /api/debates/graph/{id}/nodes
            if len(parts) >= 6 and parts[5] == "nodes":
                return await self._get_nodes(handler, debate_id)

            return await self._get_graph_debate(handler, debate_id)

        return error_response("Not found", 404)

    @handle_errors
    async def handle_post(self, handler, path: str, data: dict) -> HandlerResult:
        """Handle POST requests for graph debates.

        POST /api/debates/graph - Run a new graph debate
        """
        if not path.rstrip("/").endswith("/debates/graph"):
            return error_response("Not found", 404)

        return await self._run_graph_debate(handler, data)

    async def _run_graph_debate(self, handler, data: dict) -> HandlerResult:
        """Run a graph-structured debate with automatic branching.

        Request body:
            task: str - The debate topic/question
            agents: list[str] - Agent names to participate
            max_rounds: int - Maximum rounds per branch (default: 5)
            branch_policy: dict - Custom branch policy settings
        """
        task = data.get("task")
        if not task:
            return error_response("task is required", 400)

        agent_names = data.get("agents", [])
        max_rounds = data.get("max_rounds", 5)
        branch_policy_data = data.get("branch_policy", {})

        try:
            from aragora.debate.graph import (
                GraphDebateOrchestrator,
                BranchPolicy,
                BranchReason,
                MergeStrategy,
            )
            from aragora.agents import load_agents
            import uuid

            # Load agents
            agents = await self._load_agents(agent_names)
            if not agents:
                return error_response("No valid agents found", 400)

            # Create branch policy
            policy = BranchPolicy(
                min_disagreement=branch_policy_data.get("min_disagreement", 0.7),
                max_branches=branch_policy_data.get("max_branches", 3),
                auto_merge=branch_policy_data.get("auto_merge", True),
                merge_strategy=MergeStrategy(
                    branch_policy_data.get("merge_strategy", "synthesis")
                ),
            )

            # Create orchestrator
            orchestrator = GraphDebateOrchestrator(agents=agents, policy=policy)

            # Generate debate ID
            debate_id = str(uuid.uuid4())

            # Get event emitter if available
            event_emitter = getattr(handler, "event_emitter", None)

            # Define run_agent function
            async def run_agent(agent, prompt: str, context: list) -> str:
                return await agent.generate(prompt, context)

            # Run the debate
            graph = await orchestrator.run_debate(
                task=task,
                max_rounds=max_rounds,
                run_agent_fn=run_agent,
                event_emitter=event_emitter,
                debate_id=debate_id,
            )

            # Convert to response format
            return json_response({
                "debate_id": debate_id,
                "task": task,
                "graph": graph.to_dict(),
                "branches": [b.to_dict() for b in graph.branches.values()],
                "merge_results": [m.to_dict() for m in graph.merge_results],
                "node_count": len(graph.nodes),
                "branch_count": len(graph.branches),
            })

        except ImportError as e:
            logger.error(f"Import error for graph debates: {e}")
            return error_response("Graph debate module not available", 500)
        except Exception as e:
            logger.exception(f"Graph debate failed: {e}")
            return error_response(f"Graph debate failed: {str(e)}", 500)

    async def _load_agents(self, agent_names: list[str]) -> list:
        """Load agents by name."""
        try:
            from aragora.agents import load_agents
            return load_agents(agent_names or ["claude", "gpt4"])
        except Exception as e:
            logger.warning(f"Failed to load agents: {e}")
            return []

    async def _get_graph_debate(self, handler, debate_id: str) -> HandlerResult:
        """Get a graph debate by ID."""
        storage = getattr(handler, "storage", None)
        if not storage:
            return error_response("Storage not configured", 503)

        try:
            debate = await storage.get_graph_debate(debate_id)
            if not debate:
                return error_response("Graph debate not found", 404)

            return json_response(debate)
        except Exception as e:
            logger.error(f"Failed to get graph debate {debate_id}: {e}")
            return error_response("Failed to retrieve graph debate", 500)

    async def _get_branches(self, handler, debate_id: str) -> HandlerResult:
        """Get all branches for a graph debate."""
        storage = getattr(handler, "storage", None)
        if not storage:
            return error_response("Storage not configured", 503)

        try:
            branches = await storage.get_debate_branches(debate_id)
            return json_response({"debate_id": debate_id, "branches": branches})
        except Exception as e:
            logger.error(f"Failed to get branches for {debate_id}: {e}")
            return error_response("Failed to retrieve branches", 500)

    async def _get_nodes(self, handler, debate_id: str) -> HandlerResult:
        """Get all nodes in a graph debate."""
        storage = getattr(handler, "storage", None)
        if not storage:
            return error_response("Storage not configured", 503)

        try:
            nodes = await storage.get_debate_nodes(debate_id)
            return json_response({"debate_id": debate_id, "nodes": nodes})
        except Exception as e:
            logger.error(f"Failed to get nodes for {debate_id}: {e}")
            return error_response("Failed to retrieve nodes", 500)
