"""
GraphQL Resolvers for Debate operations.

Contains query, mutation, and subscription resolvers for debates,
plus transform functions for debate data.

Separated from resolvers.py for maintainability.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, TYPE_CHECKING

from aragora.config import DEFAULT_ROUNDS

from .resolvers import (
    ResolverContext,
    ResolverResult,
    _normalize_debate_status,
    _to_iso_datetime,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Debate Transform Functions
# =============================================================================


def _transform_debate(debate: dict[str, Any]) -> dict[str, Any]:
    """Transform internal debate format to GraphQL format."""
    messages = debate.get("messages", [])
    critiques = debate.get("critiques", [])

    # Group messages by round
    rounds_map: dict[int, dict[str, Any]] = {}
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
    total_rounds = debate.get("rounds", DEFAULT_ROUNDS)
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
        "roundCount": debate.get("rounds", DEFAULT_ROUNDS),
        "tags": debate.get("tags", []),
        "consensusReached": debate.get("consensus_reached", False),
        "confidence": debate.get("confidence"),
        "winner": debate.get("winner"),
    }


async def _transform_debate_async(
    debate: dict[str, Any],
    ctx: ResolverContext,
) -> dict[str, Any]:
    """Transform internal debate format to GraphQL format with DataLoader batching.

    Uses DataLoaders to batch-fetch agent stats, solving the N+1 query problem.

    Args:
        debate: Internal debate dict
        ctx: ResolverContext with DataLoaders

    Returns:
        GraphQL-formatted debate dict with real agent stats
    """
    # Import here to avoid circular import
    from aragora.server.graphql.dataloaders import load_agents_batch

    messages = debate.get("messages", [])
    critiques = debate.get("critiques", [])

    # Group messages by round
    rounds_map: dict[int, dict[str, Any]] = {}
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
    total_rounds = debate.get("rounds", DEFAULT_ROUNDS)
    for round_num in rounds_map:
        if round_num < total_rounds or debate.get("status") in ("completed", "concluded"):
            rounds_map[round_num]["completed"] = True

    rounds = [rounds_map[k] for k in sorted(rounds_map.keys())]

    # Build participants list using DataLoader for batch fetching
    agents = debate.get("agents", [])
    if isinstance(agents, str):
        agents = agents.split(",")
    agent_names = [name.strip() for name in agents if name.strip()]

    # Batch load all participant data in a single query
    participants = await load_agents_batch(ctx.loaders, agent_names)

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
        "roundCount": debate.get("rounds", DEFAULT_ROUNDS),
        "tags": debate.get("tags", []),
        "consensusReached": debate.get("consensus_reached", False),
        "confidence": debate.get("confidence"),
        "winner": debate.get("winner"),
    }


# =============================================================================
# Debate Query Resolvers
# =============================================================================


class DebateQueryResolvers:
    """Query resolvers for debate operations."""

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

            # Transform to GraphQL format using DataLoader for agent stats
            if ctx.loaders:
                data = await _transform_debate_async(debate, ctx)
            else:
                data = _transform_debate(debate)
            return ResolverResult(data=data)

        except (KeyError, AttributeError, TypeError, ValueError) as e:
            # Data access or transformation errors
            logger.exception(f"Error resolving debate {id}: {e}")
            return ResolverResult(errors=[f"Failed to resolve debate: {e}"])

    @staticmethod
    async def resolve_debates(
        ctx: ResolverContext,
        status: str | None = None,
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

            # Transform debates using DataLoader for batched agent stats
            transformed: list[dict[str, Any]] = []
            for d in debates:
                debate_dict = (
                    d
                    if isinstance(d, dict)
                    else (d.__dict__ if hasattr(d, "__dict__") else {"id": str(d)})
                )
                if ctx.loaders:
                    transformed.append(await _transform_debate_async(debate_dict, ctx))
                else:
                    transformed.append(_transform_debate(debate_dict))

            return ResolverResult(
                data={
                    "debates": transformed,
                    "total": len(transformed),
                    "hasMore": has_more,
                    "cursor": None,
                }
            )

        except (KeyError, AttributeError, TypeError, ValueError) as e:
            # Data access or transformation errors
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
            debates: list[Any]
            if hasattr(storage, "search"):
                debates = list(storage.search(query=query, limit=limit, org_id=ctx.org_id))
            else:
                # Fallback: get recent and filter
                all_debates = storage.list_recent(limit=limit * 2, org_id=ctx.org_id)
                query_lower = query.lower()
                debates = [
                    d
                    for d in all_debates
                    if query_lower
                    in str(
                        d.get("task", "") if isinstance(d, dict) else getattr(d, "task", "")
                    ).lower()
                ][:limit]

            # Transform using DataLoader for batched agent stats
            transformed: list[dict[str, Any]] = []
            for d in debates:
                debate_dict = (
                    d
                    if isinstance(d, dict)
                    else (d.__dict__ if hasattr(d, "__dict__") else {"id": str(d)})
                )
                if ctx.loaders:
                    transformed.append(await _transform_debate_async(debate_dict, ctx))
                else:
                    transformed.append(_transform_debate(debate_dict))

            return ResolverResult(
                data={
                    "debates": transformed,
                    "total": len(transformed),
                    "hasMore": False,
                    "cursor": None,
                }
            )

        except (KeyError, AttributeError, TypeError, ValueError) as e:
            # Data access or transformation errors
            logger.exception(f"Error searching debates: {e}")
            return ResolverResult(errors=[f"Failed to search debates: {e}"])


# =============================================================================
# Debate Mutation Resolvers
# =============================================================================


class DebateMutationResolvers:
    """Mutation resolvers for debate operations."""

    @staticmethod
    async def resolve_start_debate(
        ctx: ResolverContext,
        input: dict[str, Any],
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
                source=InputSource.HTTP_API,  # GraphQL is served over HTTP
            )

            # Set optional fields on config (DecisionConfig has these)
            if input.get("agents"):
                request.config.agents = input["agents"]
            if input.get("rounds"):
                request.config.rounds = input["rounds"]
            if input.get("consensus"):
                request.config.consensus = input["consensus"]
            # autoSelect is not directly supported, skip

            # Set context fields (RequestContext has tags)
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
                    if ctx.loaders:
                        return ResolverResult(data=await _transform_debate_async(debate, ctx))
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
                    "roundCount": input.get("rounds", DEFAULT_ROUNDS),
                    "tags": input.get("tags", []),
                    "consensusReached": False,
                    "confidence": None,
                    "winner": None,
                }
            )

        except (KeyError, AttributeError, TypeError, ValueError, RuntimeError) as e:
            # Request building, routing, or response transformation errors
            logger.exception(f"Error starting debate: {e}")
            return ResolverResult(errors=[f"Failed to start debate: {e}"])

    @staticmethod
    async def resolve_submit_vote(
        ctx: ResolverContext,
        debate_id: str,
        vote: dict[str, Any],
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

        except (KeyError, AttributeError, TypeError, ValueError) as e:
            # Vote data access or storage errors
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
                    if ctx.loaders:
                        return ResolverResult(data=await _transform_debate_async(debate, ctx))
                    return ResolverResult(data=_transform_debate(debate))

            return ResolverResult(
                data={
                    "id": id,
                    "status": "CANCELLED",
                }
            )

        except (KeyError, AttributeError, TypeError, ValueError, RuntimeError) as e:
            # State access, update, or transformation errors
            logger.exception(f"Error cancelling debate: {e}")
            return ResolverResult(errors=[f"Failed to cancel debate: {e}"])


# =============================================================================
# Debate Subscription Resolvers
# =============================================================================


class DebateSubscriptionResolvers:
    """Subscription resolvers for debate operations."""

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
