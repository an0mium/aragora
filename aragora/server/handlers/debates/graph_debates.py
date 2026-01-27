"""
Graph debates endpoint handlers.

Endpoints:
- POST /api/debates/graph - Run a graph-structured debate with branching
- GET /api/debates/graph/{id} - Get graph debate by ID
- GET /api/debates/graph/{id}/branches - Get all branches for a debate
- GET /api/debates/graph/{id}/nodes - Get all nodes in debate graph
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from ..base import (
    SAFE_AGENT_PATTERN,
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    safe_error_message,
)
from ..secure import SecureHandler, ForbiddenError, UnauthorizedError
from ..versioning.compat import strip_version_prefix

# Suspicious patterns for task sanitization
_SUSPICIOUS_PATTERNS = [
    re.compile(r"<script", re.IGNORECASE),
    re.compile(r"javascript:", re.IGNORECASE),
    re.compile(r"\x00"),  # Null byte injection
    re.compile(r"\{\{.*\}\}"),  # Template injection
]
from ..utils.rate_limit import RateLimiter, get_client_ip

logger = logging.getLogger(__name__)

# RBAC permissions for graph debates
DEBATES_READ_PERMISSION = "debates:read"
DEBATES_CREATE_PERMISSION = "debates:create"

# Rate limiter for graph debates (5 requests per minute - branching debates are expensive)
_graph_limiter = RateLimiter(requests_per_minute=5)


class GraphDebatesHandler(SecureHandler):
    """Handler for graph debate endpoints.

    RBAC Protected:
    - debates:read - required for GET endpoints
    - debates:create - required for POST endpoints
    """

    ROUTES = [
        "/api/v1/debates/graph",
        "/api/v1/debates/graph/",
    ]

    AUTH_REQUIRED_ENDPOINTS = [
        "/api/v1/debates/graph",
    ]

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        normalized = strip_version_prefix(path)
        return normalized.startswith("/api/debates/graph")

    def handle(  # type: ignore[override]
        self, path: str, query_params: dict, handler: Any
    ) -> Optional[HandlerResult]:
        """Route GET requests through the async handler."""
        return self.handle_get(handler, path, query_params)

    @handle_errors("graph debates GET")
    async def handle_get(self, handler, path: str, query_params: dict) -> HandlerResult:
        """Handle GET requests for graph debates with RBAC."""
        # RBAC: Require authentication and debates:read permission
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
            self.check_permission(auth_context, DEBATES_READ_PERMISSION)
        except UnauthorizedError:
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            logger.warning(f"Graph debates GET access denied: {e}")
            return error_response(str(e), 403)

        # Extract debate ID from path if present
        normalized = strip_version_prefix(path)
        parts = normalized.rstrip("/").split("/")

        # GET /api/debates/graph/{id} - Get specific graph debate
        # Path structure: ['', 'api', 'debates', 'graph', '{id}', ...]
        if len(parts) >= 5 and parts[3] == "graph":
            debate_id = parts[4]

            # GET /api/v1/debates/graph/{id}/branches
            if len(parts) >= 6 and parts[5] == "branches":
                return await self._get_branches(handler, debate_id)

            # GET /api/v1/debates/graph/{id}/nodes
            if len(parts) >= 6 and parts[5] == "nodes":
                return await self._get_nodes(handler, debate_id)

            return await self._get_graph_debate(handler, debate_id)

        return error_response("Not found", 404)

    @handle_errors("graph debates POST")
    async def handle_post(self, *args, **kwargs) -> HandlerResult:
        """Handle POST requests for graph debates with RBAC.

        POST /api/debates/graph - Run a new graph debate
        """
        handler = None
        path = ""
        data: dict = {}

        if len(args) >= 3:
            if isinstance(args[0], str):
                path = args[0]
                handler = args[2]
                data, error = self.read_json_body_validated(handler)
                if error:
                    return error
            else:
                handler = args[0]
                path = args[1]
                data = args[2] or {}
        else:
            handler = kwargs.get("handler")
            path = kwargs.get("path", "")
            data = kwargs.get("data") or kwargs.get("body") or {}
            if handler is None:
                return error_response("Invalid request", 400)
            if not data:
                data, error = self.read_json_body_validated(handler)
                if error:
                    return error

        normalized = strip_version_prefix(path)
        if not normalized.rstrip("/").endswith("/debates/graph"):
            return error_response("Not found", 404)

        # RBAC: Require authentication and debates:create permission
        try:
            auth_context = await self.get_auth_context(handler, require_auth=True)
            self.check_permission(auth_context, DEBATES_CREATE_PERMISSION)
        except UnauthorizedError:
            return error_response("Authentication required", 401)
        except ForbiddenError as e:
            logger.warning(f"Graph debates POST access denied: {e}")
            return error_response(str(e), 403)

        # Rate limit check (5/min - expensive branching operations)
        client_ip = get_client_ip(handler)
        if not _graph_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for graph debates: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        logger.debug("POST /api/debates/graph - running graph debate")
        return await self._run_graph_debate(handler, data)

    async def _run_graph_debate(self, handler, data: dict) -> HandlerResult:
        """Run a graph-structured debate with automatic branching.

        Request body:
            task: str - The debate topic/question (10-5000 chars)
            agents: list[str] - Agent names to participate (2-10 agents)
            max_rounds: int - Maximum rounds per branch (1-20, default: 5)
            branch_policy: dict - Custom branch policy settings
        """
        # Validate task
        task = data.get("task")
        if not task:
            return error_response("task is required", 400)
        if not isinstance(task, str):
            return error_response("task must be a string", 400)
        task = task.strip()
        if len(task) < 10:
            return error_response("task must be at least 10 characters", 400)
        if len(task) > 5000:
            return error_response("task must be at most 5000 characters", 400)

        # Check for suspicious patterns in task (injection prevention)
        for pattern in _SUSPICIOUS_PATTERNS:
            if pattern.search(task):
                return error_response("task contains invalid characters", 400)

        # Validate agents
        agent_names = data.get("agents", [])
        if not isinstance(agent_names, list):
            return error_response("agents must be an array", 400)
        if len(agent_names) < 2:
            return error_response("At least 2 agents required for a debate", 400)
        if len(agent_names) > 10:
            return error_response("Maximum 10 agents allowed", 400)
        # Validate each agent name using security pattern
        for i, name in enumerate(agent_names):
            if not isinstance(name, str):
                return error_response(f"agents[{i}] must be a string", 400)
            if len(name) > 50:
                return error_response(f"agents[{i}] name too long (max 50 chars)", 400)
            if not SAFE_AGENT_PATTERN.match(name):
                return error_response(
                    f"agents[{i}]: invalid agent name (alphanumeric, hyphens, underscores only)",
                    400,
                )

        # Validate max_rounds
        max_rounds = data.get("max_rounds", 5)
        if not isinstance(max_rounds, int):
            try:
                max_rounds = int(max_rounds)
            except (ValueError, TypeError):
                return error_response("max_rounds must be an integer", 400)
        if max_rounds < 1:
            return error_response("max_rounds must be at least 1", 400)
        if max_rounds > 20:
            return error_response("max_rounds must be at most 20", 400)

        # Validate branch_policy
        branch_policy_data = data.get("branch_policy", {})
        if not isinstance(branch_policy_data, dict):
            return error_response("branch_policy must be an object", 400)

        # Validate branch_policy fields
        if "min_disagreement" in branch_policy_data:
            min_dis = branch_policy_data["min_disagreement"]
            if not isinstance(min_dis, (int, float)) or min_dis < 0 or min_dis > 1:
                return error_response("branch_policy.min_disagreement must be 0-1", 400)
        if "max_branches" in branch_policy_data:
            max_br = branch_policy_data["max_branches"]
            if not isinstance(max_br, int) or max_br < 1 or max_br > 10:
                return error_response("branch_policy.max_branches must be 1-10", 400)
        if "merge_strategy" in branch_policy_data:
            strategy = branch_policy_data["merge_strategy"]
            if strategy not in ["synthesis", "vote", "best"]:
                return error_response(
                    "branch_policy.merge_strategy must be 'synthesis', 'vote', or 'best'", 400
                )

        try:
            import uuid

            from aragora.debate.graph import (
                BranchPolicy,
                GraphDebateOrchestrator,
            )

            # Load agents
            agents = await self._load_agents(agent_names)
            if not agents:
                return error_response("No valid agents found", 400)

            # Create branch policy
            policy = BranchPolicy(
                disagreement_threshold=branch_policy_data.get("min_disagreement", 0.7),
                max_branches=branch_policy_data.get("max_branches", 3),
                auto_merge_on_convergence=branch_policy_data.get("auto_merge", True),
            )

            # Create orchestrator
            orchestrator = GraphDebateOrchestrator(agents=agents, policy=policy)

            # Generate debate ID
            debate_id = str(uuid.uuid4())

            # Get event emitter if available
            event_emitter = getattr(handler, "event_emitter", None)

            # Define run_agent function
            async def run_agent(agent, prompt: str, context: list) -> str:
                from aragora.server.stream.arena_hooks import streaming_task_context

                agent_name = getattr(agent, "name", "graph-agent")
                task_id = f"{agent_name}:graph_debate"
                with streaming_task_context(task_id):
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
            return json_response(
                {
                    "debate_id": debate_id,
                    "task": task,
                    "graph": graph.to_dict(),
                    "branches": [b.to_dict() for b in graph.branches.values()],
                    "merge_results": [
                        {
                            "merged_node_id": m.merged_node_id,
                            "source_branch_ids": m.source_branch_ids,
                            "strategy": m.strategy.value,
                            "conflicts_resolved": m.conflicts_resolved,
                            "insights_preserved": m.insights_preserved,
                        }
                        for m in graph.merge_history
                    ],
                    "node_count": len(graph.nodes),
                    "branch_count": len(graph.branches),
                }
            )

        except ImportError as e:
            logger.error(f"Import error for graph debates: {e}")
            return error_response("Graph debate module not available", 500)
        except Exception as e:
            logger.exception(f"Graph debate failed: {e}")
            return error_response(safe_error_message(e, "graph debate"), 500)

    async def _load_agents(self, agent_names: list[str]) -> list:
        """Load agents by name."""
        try:
            from aragora.agents import create_agent

            agents = []
            for name in agent_names or ["claude", "gpt4"]:
                try:
                    # Cast to AgentType - validation already done in handle_post
                    agent = create_agent(model_type=name, name=name)  # type: ignore[arg-type]
                    agents.append(agent)
                except Exception as e:
                    logger.warning(f"Failed to create agent {name}: {e}")
            return agents
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
