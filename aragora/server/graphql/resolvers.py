"""
GraphQL Resolvers for Aragora API.

This module contains resolver functions that bridge GraphQL operations
to the existing REST handlers and services.

Resolvers follow a consistent pattern:
1. Extract arguments from the GraphQL field
2. Call the appropriate service/handler method
3. Transform the result to match the GraphQL schema

Usage:
    from aragora.server.graphql.resolvers import QueryResolvers, MutationResolvers

    # Execute a resolver
    result = await QueryResolvers.resolve_debate(context, id="debate-123")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.server.handlers.base import ServerContext

logger = logging.getLogger(__name__)


# =============================================================================
# Context and Types
# =============================================================================


@dataclass
class ResolverContext:
    """Context passed to resolvers during execution.

    Attributes:
        server_context: Server context with storage, ELO system, etc.
        user_id: Authenticated user ID (if authenticated)
        org_id: Organization/workspace ID
        trace_id: Request trace ID for logging
        variables: Query variables
    """

    server_context: "ServerContext"
    user_id: Optional[str] = None
    org_id: Optional[str] = None
    trace_id: Optional[str] = None
    variables: Dict[str, Any] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.variables is None:
            self.variables = {}


@dataclass
class ResolverResult:
    """Result from a resolver execution.

    Attributes:
        data: Resolved data
        errors: List of error messages
    """

    data: Any = None
    errors: List[str] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


# =============================================================================
# Status Mapping Utilities
# =============================================================================


def _normalize_debate_status(status: Optional[str]) -> str:
    """Normalize internal debate status to GraphQL enum value."""
    status_map = {
        "starting": "PENDING",
        "running": "RUNNING",
        "active": "RUNNING",
        "completed": "COMPLETED",
        "concluded": "COMPLETED",
        "failed": "FAILED",
        "cancelled": "CANCELLED",
        "canceled": "CANCELLED",
        "paused": "PENDING",
    }
    if not status:
        return "PENDING"
    return status_map.get(status.lower(), status.upper())


def _normalize_agent_status(status: Optional[str]) -> str:
    """Normalize agent status to GraphQL enum value."""
    status_map = {
        "available": "AVAILABLE",
        "idle": "AVAILABLE",
        "busy": "BUSY",
        "running": "BUSY",
        "offline": "OFFLINE",
        "degraded": "DEGRADED",
        "error": "DEGRADED",
    }
    if not status:
        return "OFFLINE"
    return status_map.get(status.lower(), status.upper())


def _normalize_task_status(status: Optional[str]) -> str:
    """Normalize task status to GraphQL enum value."""
    status_map = {
        "pending": "PENDING",
        "queued": "PENDING",
        "running": "RUNNING",
        "in_progress": "RUNNING",
        "completed": "COMPLETED",
        "done": "COMPLETED",
        "failed": "FAILED",
        "error": "FAILED",
        "cancelled": "CANCELLED",
        "canceled": "CANCELLED",
    }
    if not status:
        return "PENDING"
    return status_map.get(status.lower(), status.upper())


def _normalize_priority(priority: Optional[str]) -> str:
    """Normalize priority to GraphQL enum value."""
    priority_map = {
        "low": "LOW",
        "normal": "NORMAL",
        "medium": "NORMAL",
        "high": "HIGH",
        "urgent": "URGENT",
        "critical": "URGENT",
    }
    if not priority:
        return "NORMAL"
    return priority_map.get(priority.lower(), priority.upper())


def _normalize_health_status(status: Optional[str]) -> str:
    """Normalize health status to GraphQL enum value."""
    status_map = {
        "healthy": "HEALTHY",
        "ok": "HEALTHY",
        "degraded": "DEGRADED",
        "warning": "DEGRADED",
        "unhealthy": "UNHEALTHY",
        "error": "UNHEALTHY",
        "down": "UNHEALTHY",
    }
    if not status:
        return "HEALTHY"
    return status_map.get(status.lower(), status.upper())


def _to_iso_datetime(value: Any) -> Optional[str]:
    """Convert various datetime formats to ISO string."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value).isoformat()
    return str(value)


# =============================================================================
# Query Resolvers
# =============================================================================


class QueryResolvers:
    """Resolvers for GraphQL Query operations."""

    @staticmethod
    async def resolve_debate(
        ctx: ResolverContext,
        id: str,
    ) -> ResolverResult:
        """Resolve a single debate by ID.

        Args:
            ctx: Resolver context
            id: Debate ID

        Returns:
            ResolverResult with debate data or errors
        """
        try:
            storage = ctx.server_context.get("storage")
            if not storage:
                return ResolverResult(errors=["Storage not available"])

            debate = storage.get_debate(id)
            if not debate:
                return ResolverResult(errors=[f"Debate not found: {id}"])

            # Transform to GraphQL format
            data = _transform_debate(debate)
            return ResolverResult(data=data)

        except Exception as e:
            logger.exception(f"Error resolving debate {id}: {e}")
            return ResolverResult(errors=[f"Failed to resolve debate: {e}"])

    @staticmethod
    async def resolve_debates(
        ctx: ResolverContext,
        status: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> ResolverResult:
        """Resolve a list of debates with optional filtering.

        Args:
            ctx: Resolver context
            status: Optional status filter
            limit: Maximum results (default 20)
            offset: Pagination offset

        Returns:
            ResolverResult with DebateConnection data
        """
        try:
            storage = ctx.server_context.get("storage")
            if not storage:
                return ResolverResult(errors=["Storage not available"])

            # Get debates from storage
            debates = storage.list_recent(limit=limit + 1, org_id=ctx.org_id)

            # Filter by status if provided
            if status:
                target_status = status.lower()
                debates = [
                    d
                    for d in debates
                    if _normalize_debate_status(
                        d.get("status") if isinstance(d, dict) else getattr(d, "status", None)
                    ).lower()
                    == target_status
                ]

            # Apply pagination
            has_more = len(debates) > limit
            debates = debates[offset : offset + limit]

            # Transform debates
            transformed = [
                _transform_debate(d.__dict__ if hasattr(d, "__dict__") else d) for d in debates
            ]

            return ResolverResult(
                data={
                    "debates": transformed,
                    "total": len(transformed),
                    "hasMore": has_more,
                    "cursor": None,
                }
            )

        except Exception as e:
            logger.exception(f"Error resolving debates: {e}")
            return ResolverResult(errors=[f"Failed to resolve debates: {e}"])

    @staticmethod
    async def resolve_search_debates(
        ctx: ResolverContext,
        query: str,
        limit: int = 20,
    ) -> ResolverResult:
        """Search debates by query string.

        Args:
            ctx: Resolver context
            query: Search query
            limit: Maximum results

        Returns:
            ResolverResult with DebateConnection data
        """
        try:
            storage = ctx.server_context.get("storage")
            if not storage:
                return ResolverResult(errors=["Storage not available"])

            # Use storage search if available
            if hasattr(storage, "search"):
                debates = storage.search(query=query, limit=limit, org_id=ctx.org_id)
            else:
                # Fallback: get recent and filter
                debates = storage.list_recent(limit=limit * 2, org_id=ctx.org_id)
                query_lower = query.lower()
                debates = [
                    d
                    for d in debates
                    if query_lower
                    in str(
                        d.get("task", "") if isinstance(d, dict) else getattr(d, "task", "")
                    ).lower()
                ][:limit]

            transformed = [
                _transform_debate(d.__dict__ if hasattr(d, "__dict__") else d) for d in debates
            ]

            return ResolverResult(
                data={
                    "debates": transformed,
                    "total": len(transformed),
                    "hasMore": False,
                    "cursor": None,
                }
            )

        except Exception as e:
            logger.exception(f"Error searching debates: {e}")
            return ResolverResult(errors=[f"Failed to search debates: {e}"])

    @staticmethod
    async def resolve_agent(
        ctx: ResolverContext,
        id: str,
    ) -> ResolverResult:
        """Resolve a single agent by ID.

        Args:
            ctx: Resolver context
            id: Agent ID/name

        Returns:
            ResolverResult with agent data
        """
        try:
            elo_system = ctx.server_context.get("elo_system")
            if not elo_system:
                return ResolverResult(errors=["ELO system not available"])

            rating = elo_system.get_rating(id)
            if not rating:
                return ResolverResult(errors=[f"Agent not found: {id}"])

            data = _transform_agent(rating, id)
            return ResolverResult(data=data)

        except Exception as e:
            logger.exception(f"Error resolving agent {id}: {e}")
            return ResolverResult(errors=[f"Failed to resolve agent: {e}"])

    @staticmethod
    async def resolve_agents(
        ctx: ResolverContext,
        status: Optional[str] = None,
        capability: Optional[str] = None,
        region: Optional[str] = None,
    ) -> ResolverResult:
        """Resolve a list of agents with optional filtering.

        Args:
            ctx: Resolver context
            status: Optional status filter
            capability: Optional capability filter
            region: Optional region filter

        Returns:
            ResolverResult with list of agents
        """
        try:
            elo_system = ctx.server_context.get("elo_system")
            if not elo_system:
                return ResolverResult(errors=["ELO system not available"])

            # Get all agents from leaderboard
            rankings = elo_system.get_leaderboard(limit=500)

            agents = []
            for agent in rankings:
                agent_data = _transform_agent(agent)

                # Apply filters
                if status and agent_data.get("status") != status:
                    continue
                if capability:
                    caps = agent_data.get("capabilities", [])
                    if capability not in caps:
                        continue
                if region and agent_data.get("region") != region:
                    continue

                agents.append(agent_data)

            return ResolverResult(data=agents)

        except Exception as e:
            logger.exception(f"Error resolving agents: {e}")
            return ResolverResult(errors=[f"Failed to resolve agents: {e}"])

    @staticmethod
    async def resolve_leaderboard(
        ctx: ResolverContext,
        limit: int = 20,
        domain: Optional[str] = None,
    ) -> ResolverResult:
        """Get agent leaderboard.

        Args:
            ctx: Resolver context
            limit: Maximum results
            domain: Optional domain filter

        Returns:
            ResolverResult with list of agents sorted by ELO
        """
        try:
            elo_system = ctx.server_context.get("elo_system")
            if not elo_system:
                return ResolverResult(errors=["ELO system not available"])

            # Get leaderboard
            if hasattr(elo_system, "get_cached_leaderboard") and domain is None:
                rankings = elo_system.get_cached_leaderboard(limit=min(limit, 50))
            else:
                rankings = elo_system.get_leaderboard(limit=min(limit, 50), domain=domain)

            agents = [_transform_agent(agent) for agent in rankings]
            return ResolverResult(data=agents)

        except Exception as e:
            logger.exception(f"Error resolving leaderboard: {e}")
            return ResolverResult(errors=[f"Failed to resolve leaderboard: {e}"])

    @staticmethod
    async def resolve_task(
        ctx: ResolverContext,
        id: str,
    ) -> ResolverResult:
        """Resolve a single task by ID.

        Args:
            ctx: Resolver context
            id: Task ID

        Returns:
            ResolverResult with task data
        """
        try:
            coordinator = ctx.server_context.get("control_plane_coordinator")
            if not coordinator:
                return ResolverResult(errors=["Control plane not available"])

            task = await coordinator.get_task(id)
            if not task:
                return ResolverResult(errors=[f"Task not found: {id}"])

            data = _transform_task(task)
            return ResolverResult(data=data)

        except Exception as e:
            logger.exception(f"Error resolving task {id}: {e}")
            return ResolverResult(errors=[f"Failed to resolve task: {e}"])

    @staticmethod
    async def resolve_tasks(
        ctx: ResolverContext,
        status: Optional[str] = None,
        type: Optional[str] = None,
        limit: int = 20,
    ) -> ResolverResult:
        """Resolve a list of tasks with optional filtering.

        Args:
            ctx: Resolver context
            status: Optional status filter
            type: Optional task type filter
            limit: Maximum results

        Returns:
            ResolverResult with TaskConnection data
        """
        try:
            coordinator = ctx.server_context.get("control_plane_coordinator")
            if not coordinator:
                return ResolverResult(errors=["Control plane not available"])

            # Get tasks based on status filter
            from aragora.control_plane.scheduler import TaskStatus as CPTaskStatus

            tasks = []
            if status:
                try:
                    cp_status = CPTaskStatus(status.lower())
                    tasks = await coordinator._scheduler.list_by_status(cp_status, limit=limit)
                except ValueError:
                    pass
            else:
                # Get pending and running tasks
                pending = await coordinator._scheduler.list_by_status(
                    CPTaskStatus.PENDING, limit=limit
                )
                running = await coordinator._scheduler.list_by_status(
                    CPTaskStatus.RUNNING, limit=limit
                )
                tasks = list(running) + list(pending)

            # Filter by type if provided
            if type:
                tasks = [t for t in tasks if t.task_type == type]

            transformed = [_transform_task(t) for t in tasks[:limit]]

            return ResolverResult(
                data={
                    "tasks": transformed,
                    "total": len(transformed),
                    "hasMore": len(tasks) > limit,
                }
            )

        except Exception as e:
            logger.exception(f"Error resolving tasks: {e}")
            return ResolverResult(errors=[f"Failed to resolve tasks: {e}"])

    @staticmethod
    async def resolve_system_health(ctx: ResolverContext) -> ResolverResult:
        """Resolve system health status.

        Args:
            ctx: Resolver context

        Returns:
            ResolverResult with SystemHealth data
        """
        try:
            import time

            coordinator = ctx.server_context.get("control_plane_coordinator")

            components = []
            overall_status = "HEALTHY"

            # Check coordinator
            if coordinator:
                components.append(
                    {
                        "name": "Coordinator",
                        "status": "HEALTHY",
                        "latencyMs": 0,
                        "error": None,
                    }
                )
            else:
                components.append(
                    {
                        "name": "Coordinator",
                        "status": "UNHEALTHY",
                        "latencyMs": None,
                        "error": "Not initialized",
                    }
                )
                overall_status = "DEGRADED"

            # Check storage
            storage = ctx.server_context.get("storage")
            if storage:
                components.append(
                    {
                        "name": "Storage",
                        "status": "HEALTHY",
                        "latencyMs": 5,
                        "error": None,
                    }
                )
            else:
                components.append(
                    {
                        "name": "Storage",
                        "status": "UNHEALTHY",
                        "latencyMs": None,
                        "error": "Not available",
                    }
                )
                overall_status = "DEGRADED"

            # Check ELO system
            elo_system = ctx.server_context.get("elo_system")
            if elo_system:
                components.append(
                    {
                        "name": "ELO System",
                        "status": "HEALTHY",
                        "latencyMs": 2,
                        "error": None,
                    }
                )

            start_time = ctx.server_context.get("_start_time", time.time())
            uptime = int(time.time() - start_time)

            return ResolverResult(
                data={
                    "status": overall_status,
                    "uptimeSeconds": uptime,
                    "version": "2.1.0",
                    "components": components,
                }
            )

        except Exception as e:
            logger.exception(f"Error resolving system health: {e}")
            return ResolverResult(errors=[f"Failed to resolve system health: {e}"])

    @staticmethod
    async def resolve_stats(ctx: ResolverContext) -> ResolverResult:
        """Resolve system statistics.

        Args:
            ctx: Resolver context

        Returns:
            ResolverResult with SystemStats data
        """
        try:
            coordinator = ctx.server_context.get("control_plane_coordinator")

            stats = {
                "activeJobs": 0,
                "queuedJobs": 0,
                "completedJobsToday": 0,
                "availableAgents": 0,
                "busyAgents": 0,
                "totalAgents": 0,
                "documentsProcessedToday": 0,
            }

            if coordinator:
                cp_stats = await coordinator.get_stats()
                scheduler_stats = cp_stats.get("scheduler", {})
                registry_stats = cp_stats.get("registry", {})
                by_status = scheduler_stats.get("by_status", {})

                stats["activeJobs"] = by_status.get("running", 0)
                stats["queuedJobs"] = by_status.get("pending", 0)
                stats["completedJobsToday"] = by_status.get("completed", 0)
                stats["availableAgents"] = registry_stats.get("available_agents", 0)
                stats["busyAgents"] = registry_stats.get("by_status", {}).get("busy", 0)
                stats["totalAgents"] = registry_stats.get("total_agents", 0)
                stats["documentsProcessedToday"] = scheduler_stats.get("by_type", {}).get(
                    "document_processing", 0
                )

            return ResolverResult(data=stats)

        except Exception as e:
            logger.exception(f"Error resolving stats: {e}")
            return ResolverResult(errors=[f"Failed to resolve stats: {e}"])


# =============================================================================
# Mutation Resolvers
# =============================================================================


class MutationResolvers:
    """Resolvers for GraphQL Mutation operations."""

    @staticmethod
    async def resolve_start_debate(
        ctx: ResolverContext,
        input: Dict[str, Any],
    ) -> ResolverResult:
        """Start a new debate.

        Args:
            ctx: Resolver context
            input: StartDebateInput fields

        Returns:
            ResolverResult with created debate data
        """
        try:
            question = input.get("question")
            if not question:
                return ResolverResult(errors=["Question is required"])

            # Build debate request
            from aragora.core.decision import DecisionRequest, DecisionType, InputSource

            request = DecisionRequest(
                content=question,
                decision_type=DecisionType.DEBATE,
                source=InputSource.GRAPHQL,
            )

            # Set optional fields
            if input.get("agents"):
                request.context.agents = input["agents"]
            if input.get("rounds"):
                request.context.rounds = input["rounds"]
            if input.get("consensus"):
                request.context.consensus_method = input["consensus"]
            if input.get("autoSelect"):
                request.context.auto_select = input["autoSelect"]
            if input.get("tags"):
                request.context.tags = input["tags"]

            # Set user context
            if ctx.user_id:
                request.context.user_id = ctx.user_id
            if ctx.org_id:
                request.context.workspace_id = ctx.org_id

            # Route through decision router
            from aragora.core.decision import get_decision_router

            router = get_decision_router()
            result = await router.route(request)

            if not result.success:
                return ResolverResult(errors=[result.error or "Failed to start debate"])

            # Get the created debate
            storage = ctx.server_context.get("storage")
            debate_id = getattr(result, "debate_id", request.request_id)

            if storage:
                debate = storage.get_debate(debate_id)
                if debate:
                    return ResolverResult(data=_transform_debate(debate))

            # Return minimal response
            return ResolverResult(
                data={
                    "id": debate_id,
                    "topic": question,
                    "status": "PENDING",
                    "rounds": [],
                    "participants": [],
                    "consensus": None,
                    "createdAt": datetime.now().isoformat(),
                    "completedAt": None,
                    "roundCount": input.get("rounds", 3),
                    "tags": input.get("tags", []),
                    "consensusReached": False,
                    "confidence": None,
                    "winner": None,
                }
            )

        except Exception as e:
            logger.exception(f"Error starting debate: {e}")
            return ResolverResult(errors=[f"Failed to start debate: {e}"])

    @staticmethod
    async def resolve_submit_vote(
        ctx: ResolverContext,
        debate_id: str,
        vote: Dict[str, Any],
    ) -> ResolverResult:
        """Submit a vote for an agent in a debate.

        Args:
            ctx: Resolver context
            debate_id: Debate ID
            vote: VoteInput fields

        Returns:
            ResolverResult with vote data
        """
        try:
            agent_id = vote.get("agentId")
            if not agent_id:
                return ResolverResult(errors=["Agent ID is required"])

            storage = ctx.server_context.get("storage")
            if not storage:
                return ResolverResult(errors=["Storage not available"])

            # Verify debate exists
            debate = storage.get_debate(debate_id)
            if not debate:
                return ResolverResult(errors=[f"Debate not found: {debate_id}"])

            # Record vote
            import uuid

            vote_id = str(uuid.uuid4())

            vote_data = {
                "id": vote_id,
                "debateId": debate_id,
                "agentId": agent_id,
                "reason": vote.get("reason"),
                "confidence": vote.get("confidence", 1.0),
                "createdAt": datetime.now().isoformat(),
                "userId": ctx.user_id,
            }

            # Store vote if storage supports it
            if hasattr(storage, "add_vote"):
                storage.add_vote(debate_id, vote_data)

            return ResolverResult(data=vote_data)

        except Exception as e:
            logger.exception(f"Error submitting vote: {e}")
            return ResolverResult(errors=[f"Failed to submit vote: {e}"])

    @staticmethod
    async def resolve_cancel_debate(
        ctx: ResolverContext,
        id: str,
    ) -> ResolverResult:
        """Cancel a running debate.

        Args:
            ctx: Resolver context
            id: Debate ID

        Returns:
            ResolverResult with cancelled debate data
        """
        try:
            from aragora.server.debate_utils import update_debate_status
            from aragora.server.state import get_state_manager

            manager = get_state_manager()
            state = manager.get_debate(id)

            if not state:
                storage = ctx.server_context.get("storage")
                if storage:
                    debate = storage.get_debate(id)
                    if debate:
                        return ResolverResult(
                            errors=[
                                f"Debate {id} already completed (status: {debate.get('status', 'unknown')})"
                            ]
                        )
                return ResolverResult(errors=[f"Debate not found: {id}"])

            if state.status not in ("running", "starting"):
                return ResolverResult(
                    errors=[f"Debate {id} cannot be cancelled (status: {state.status})"]
                )

            # Cancel the debate
            update_debate_status(id, "cancelled", error="Cancelled via GraphQL")
            manager.update_debate_status(id, status="cancelled")

            # Return updated debate
            storage = ctx.server_context.get("storage")
            if storage:
                debate = storage.get_debate(id)
                if debate:
                    return ResolverResult(data=_transform_debate(debate))

            return ResolverResult(
                data={
                    "id": id,
                    "status": "CANCELLED",
                }
            )

        except Exception as e:
            logger.exception(f"Error cancelling debate: {e}")
            return ResolverResult(errors=[f"Failed to cancel debate: {e}"])

    @staticmethod
    async def resolve_submit_task(
        ctx: ResolverContext,
        input: Dict[str, Any],
    ) -> ResolverResult:
        """Submit a new task to the control plane.

        Args:
            ctx: Resolver context
            input: SubmitTaskInput fields

        Returns:
            ResolverResult with created task data
        """
        try:
            task_type = input.get("taskType")
            if not task_type:
                return ResolverResult(errors=["Task type is required"])

            coordinator = ctx.server_context.get("control_plane_coordinator")
            if not coordinator:
                return ResolverResult(errors=["Control plane not available"])

            from aragora.control_plane.scheduler import TaskPriority

            priority_str = input.get("priority", "NORMAL")
            try:
                priority = TaskPriority[priority_str.upper()]
            except KeyError:
                priority = TaskPriority.NORMAL

            task_id = await coordinator.submit_task(
                task_type=task_type,
                payload=input.get("payload", {}),
                required_capabilities=input.get("requiredCapabilities", []),
                priority=priority,
                timeout_seconds=input.get("timeoutSeconds"),
                metadata=input.get("metadata", {}),
            )

            # Get created task
            task = await coordinator.get_task(task_id)
            if task:
                return ResolverResult(data=_transform_task(task))

            return ResolverResult(
                data={
                    "id": task_id,
                    "type": task_type,
                    "status": "PENDING",
                    "priority": priority_str,
                    "assignedAgent": None,
                    "result": None,
                    "createdAt": datetime.now().isoformat(),
                    "completedAt": None,
                    "payload": input.get("payload"),
                    "metadata": input.get("metadata"),
                }
            )

        except Exception as e:
            logger.exception(f"Error submitting task: {e}")
            return ResolverResult(errors=[f"Failed to submit task: {e}"])

    @staticmethod
    async def resolve_cancel_task(
        ctx: ResolverContext,
        id: str,
    ) -> ResolverResult:
        """Cancel a pending or running task.

        Args:
            ctx: Resolver context
            id: Task ID

        Returns:
            ResolverResult with cancelled task data
        """
        try:
            coordinator = ctx.server_context.get("control_plane_coordinator")
            if not coordinator:
                return ResolverResult(errors=["Control plane not available"])

            success = await coordinator.cancel_task(id)
            if not success:
                return ResolverResult(errors=[f"Task not found or already completed: {id}"])

            # Get updated task
            task = await coordinator.get_task(id)
            if task:
                return ResolverResult(data=_transform_task(task))

            return ResolverResult(
                data={
                    "id": id,
                    "status": "CANCELLED",
                }
            )

        except Exception as e:
            logger.exception(f"Error cancelling task: {e}")
            return ResolverResult(errors=[f"Failed to cancel task: {e}"])

    @staticmethod
    async def resolve_register_agent(
        ctx: ResolverContext,
        input: Dict[str, Any],
    ) -> ResolverResult:
        """Register a new agent with the control plane.

        Args:
            ctx: Resolver context
            input: RegisterAgentInput fields

        Returns:
            ResolverResult with registered agent data
        """
        try:
            agent_id = input.get("agentId")
            if not agent_id:
                return ResolverResult(errors=["Agent ID is required"])

            coordinator = ctx.server_context.get("control_plane_coordinator")
            if not coordinator:
                return ResolverResult(errors=["Control plane not available"])

            _agent = await coordinator.register_agent(
                agent_id=agent_id,
                capabilities=input.get("capabilities", []),
                model=input.get("model", "unknown"),
                provider=input.get("provider", "unknown"),
                metadata=input.get("metadata", {}),
            )

            return ResolverResult(
                data={
                    "id": agent_id,
                    "name": agent_id,
                    "status": "AVAILABLE",
                    "capabilities": input.get("capabilities", []),
                    "region": input.get("metadata", {}).get("region"),
                    "currentTask": None,
                    "stats": {
                        "totalGames": 0,
                        "wins": 0,
                        "losses": 0,
                        "draws": 0,
                        "winRate": 0.0,
                        "elo": 1500,
                        "calibrationAccuracy": None,
                        "consistencyScore": None,
                    },
                    "elo": 1500,
                    "model": input.get("model"),
                    "provider": input.get("provider"),
                }
            )

        except Exception as e:
            logger.exception(f"Error registering agent: {e}")
            return ResolverResult(errors=[f"Failed to register agent: {e}"])

    @staticmethod
    async def resolve_unregister_agent(
        ctx: ResolverContext,
        id: str,
    ) -> ResolverResult:
        """Unregister an agent from the control plane.

        Args:
            ctx: Resolver context
            id: Agent ID

        Returns:
            ResolverResult with boolean success
        """
        try:
            coordinator = ctx.server_context.get("control_plane_coordinator")
            if not coordinator:
                return ResolverResult(errors=["Control plane not available"])

            success = await coordinator.unregister_agent(id)
            if not success:
                return ResolverResult(errors=[f"Agent not found: {id}"])

            return ResolverResult(data=True)

        except Exception as e:
            logger.exception(f"Error unregistering agent: {e}")
            return ResolverResult(errors=[f"Failed to unregister agent: {e}"])


# =============================================================================
# Subscription Resolvers
# =============================================================================


class SubscriptionResolvers:
    """Resolvers for GraphQL Subscription operations.

    Note: Full subscription support requires WebSocket integration.
    These resolvers provide the async generator pattern for subscriptions.
    """

    @staticmethod
    async def subscribe_debate_updates(
        ctx: ResolverContext,
        debate_id: str,
    ):
        """Subscribe to debate updates.

        Args:
            ctx: Resolver context
            debate_id: Debate ID to subscribe to

        Yields:
            DebateEvent data
        """
        ws_manager = ctx.server_context.get("ws_manager")
        if not ws_manager:
            raise RuntimeError("WebSocket manager not available")

        # Create a queue for this subscription
        import asyncio

        queue: asyncio.Queue = asyncio.Queue()

        # Register with WebSocket manager
        # This would integrate with the existing stream infrastructure
        _subscription_id = f"debate_{debate_id}_{id(queue)}"  # For future WebSocket integration

        try:
            while True:
                event = await queue.get()
                yield {
                    "type": event.get("type", "update"),
                    "debateId": debate_id,
                    "data": event.get("data", {}),
                    "timestamp": datetime.now().isoformat(),
                }
        except asyncio.CancelledError:
            # Cleanup on unsubscribe
            pass

    @staticmethod
    async def subscribe_task_updates(
        ctx: ResolverContext,
        task_id: Optional[str] = None,
    ):
        """Subscribe to task updates.

        Args:
            ctx: Resolver context
            task_id: Optional specific task ID to subscribe to

        Yields:
            TaskEvent data
        """
        ws_manager = ctx.server_context.get("ws_manager")
        if not ws_manager:
            raise RuntimeError("WebSocket manager not available")

        import asyncio

        queue: asyncio.Queue = asyncio.Queue()

        try:
            while True:
                event = await queue.get()
                yield {
                    "type": event.get("type", "update"),
                    "taskId": event.get("task_id", task_id),
                    "data": event.get("data", {}),
                    "timestamp": datetime.now().isoformat(),
                }
        except asyncio.CancelledError:
            pass


# =============================================================================
# Transform Functions
# =============================================================================


def _transform_debate(debate: Dict[str, Any]) -> Dict[str, Any]:
    """Transform internal debate format to GraphQL format."""
    messages = debate.get("messages", [])
    critiques = debate.get("critiques", [])

    # Group messages by round
    rounds_map: Dict[int, Dict[str, Any]] = {}
    for msg in messages:
        round_num = msg.get("round", 1)
        if round_num not in rounds_map:
            rounds_map[round_num] = {
                "number": round_num,
                "messages": [],
                "critiques": [],
                "completed": False,
            }
        rounds_map[round_num]["messages"].append(
            {
                "index": len(rounds_map[round_num]["messages"]),
                "role": msg.get("role", "agent"),
                "content": msg.get("content", ""),
                "agent": msg.get("agent") or msg.get("name"),
                "round": round_num,
                "timestamp": _to_iso_datetime(msg.get("timestamp")),
            }
        )

    # Add critiques to rounds
    for critique in critiques:
        round_num = critique.get("round", 1)
        if round_num in rounds_map:
            rounds_map[round_num]["critiques"].append(
                {
                    "id": critique.get("id", ""),
                    "critic": critique.get("critic", ""),
                    "target": critique.get("target", ""),
                    "content": critique.get("content", ""),
                    "severity": critique.get("severity", 0.5),
                    "accepted": critique.get("accepted"),
                }
            )

    # Mark rounds as completed
    total_rounds = debate.get("rounds", 3)
    for round_num in rounds_map:
        if round_num < total_rounds or debate.get("status") in ("completed", "concluded"):
            rounds_map[round_num]["completed"] = True

    rounds = [rounds_map[k] for k in sorted(rounds_map.keys())]

    # Build participants list
    participants = []
    agents = debate.get("agents", [])
    if isinstance(agents, str):
        agents = agents.split(",")
    for agent_name in agents:
        agent_name = agent_name.strip()
        if agent_name:
            participants.append(
                {
                    "id": agent_name,
                    "name": agent_name,
                    "status": "AVAILABLE",
                    "capabilities": [],
                    "region": None,
                    "currentTask": None,
                    "stats": {
                        "totalGames": 0,
                        "wins": 0,
                        "losses": 0,
                        "draws": 0,
                        "winRate": 0.0,
                        "elo": 1500,
                        "calibrationAccuracy": None,
                        "consistencyScore": None,
                    },
                    "elo": 1500,
                    "model": None,
                    "provider": None,
                }
            )

    # Build consensus
    consensus = None
    if debate.get("consensus_reached"):
        consensus = {
            "reached": True,
            "answer": debate.get("final_answer", ""),
            "agreeingAgents": debate.get("agreeing_agents", []),
            "dissentingAgents": debate.get("dissenting_agents", []),
            "confidence": debate.get("confidence"),
            "method": debate.get("consensus_method", "majority"),
        }

    return {
        "id": debate.get("id") or debate.get("debate_id", ""),
        "topic": debate.get("task") or debate.get("question", ""),
        "task": debate.get("task") or debate.get("question", ""),
        "status": _normalize_debate_status(debate.get("status")),
        "rounds": rounds,
        "participants": participants,
        "consensus": consensus,
        "createdAt": _to_iso_datetime(debate.get("created_at") or debate.get("timestamp")),
        "completedAt": _to_iso_datetime(debate.get("completed_at")),
        "roundCount": debate.get("rounds", 3),
        "tags": debate.get("tags", []),
        "consensusReached": debate.get("consensus_reached", False),
        "confidence": debate.get("confidence"),
        "winner": debate.get("winner"),
    }


def _transform_agent(agent: Any, agent_id: Optional[str] = None) -> Dict[str, Any]:
    """Transform internal agent format to GraphQL format."""
    if isinstance(agent, dict):
        name = agent.get("name") or agent.get("agent_name") or agent_id or "unknown"
        return {
            "id": name,
            "name": name,
            "status": _normalize_agent_status(agent.get("status")),
            "capabilities": agent.get("capabilities", []),
            "region": agent.get("region"),
            "currentTask": None,
            "stats": {
                "totalGames": agent.get("games", 0) + agent.get("matches", 0),
                "wins": agent.get("wins", 0),
                "losses": agent.get("losses", 0),
                "draws": agent.get("draws", 0),
                "winRate": agent.get("win_rate", 0.0),
                "elo": agent.get("elo", 1500),
                "calibrationAccuracy": agent.get("calibration_accuracy"),
                "consistencyScore": agent.get("consistency"),
            },
            "elo": agent.get("elo", 1500),
            "model": agent.get("model"),
            "provider": agent.get("provider"),
        }

    # Handle object-based agent (e.g., AgentRating)
    name = (
        getattr(agent, "name", None) or getattr(agent, "agent_name", None) or agent_id or "unknown"
    )
    wins = getattr(agent, "wins", 0)
    losses = getattr(agent, "losses", 0)
    draws = getattr(agent, "draws", 0)
    total_games = wins + losses + draws

    return {
        "id": name,
        "name": name,
        "status": "AVAILABLE",
        "capabilities": getattr(agent, "capabilities", []),
        "region": getattr(agent, "region", None),
        "currentTask": None,
        "stats": {
            "totalGames": total_games,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "winRate": getattr(agent, "win_rate", wins / total_games if total_games > 0 else 0.0),
            "elo": getattr(agent, "elo", 1500),
            "calibrationAccuracy": getattr(agent, "calibration_accuracy", None),
            "consistencyScore": getattr(agent, "consistency", None),
        },
        "elo": getattr(agent, "elo", 1500),
        "model": getattr(agent, "model", None),
        "provider": getattr(agent, "provider", None),
    }


def _transform_task(task: Any) -> Dict[str, Any]:
    """Transform internal task format to GraphQL format."""
    if isinstance(task, dict):
        return {
            "id": task.get("id", ""),
            "type": task.get("task_type") or task.get("type", ""),
            "status": _normalize_task_status(task.get("status")),
            "priority": _normalize_priority(task.get("priority")),
            "assignedAgent": task.get("assigned_agent"),
            "result": task.get("result"),
            "createdAt": _to_iso_datetime(task.get("created_at")),
            "completedAt": _to_iso_datetime(task.get("completed_at")),
            "payload": task.get("payload"),
            "metadata": task.get("metadata"),
        }

    # Handle object-based task
    status = getattr(task, "status", None)
    if hasattr(status, "value"):
        status = status.value

    priority = getattr(task, "priority", None)
    if hasattr(priority, "name"):
        priority = priority.name

    return {
        "id": getattr(task, "id", ""),
        "type": getattr(task, "task_type", ""),
        "status": _normalize_task_status(status),
        "priority": _normalize_priority(priority),
        "assignedAgent": getattr(task, "assigned_agent", None),
        "result": getattr(task, "result", None),
        "createdAt": _to_iso_datetime(getattr(task, "created_at", None)),
        "completedAt": _to_iso_datetime(getattr(task, "completed_at", None)),
        "payload": getattr(task, "payload", None),
        "metadata": getattr(task, "metadata", None),
    }


# =============================================================================
# Resolver Registry
# =============================================================================

# Map of field names to resolver functions
QUERY_RESOLVERS = {
    "debate": QueryResolvers.resolve_debate,
    "debates": QueryResolvers.resolve_debates,
    "searchDebates": QueryResolvers.resolve_search_debates,
    "agent": QueryResolvers.resolve_agent,
    "agents": QueryResolvers.resolve_agents,
    "leaderboard": QueryResolvers.resolve_leaderboard,
    "task": QueryResolvers.resolve_task,
    "tasks": QueryResolvers.resolve_tasks,
    "systemHealth": QueryResolvers.resolve_system_health,
    "stats": QueryResolvers.resolve_stats,
}

MUTATION_RESOLVERS = {
    "startDebate": MutationResolvers.resolve_start_debate,
    "submitVote": MutationResolvers.resolve_submit_vote,
    "cancelDebate": MutationResolvers.resolve_cancel_debate,
    "submitTask": MutationResolvers.resolve_submit_task,
    "cancelTask": MutationResolvers.resolve_cancel_task,
    "registerAgent": MutationResolvers.resolve_register_agent,
    "unregisterAgent": MutationResolvers.resolve_unregister_agent,
}

SUBSCRIPTION_RESOLVERS = {
    "debateUpdates": SubscriptionResolvers.subscribe_debate_updates,
    "taskUpdates": SubscriptionResolvers.subscribe_task_updates,
}
