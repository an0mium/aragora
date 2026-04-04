"""
Vetted Decisionmaking (Deliberations) API Handler.

Provides endpoints for the vetted decisionmaking dashboard:
- List active vetted decisionmaking sessions
- Get vetted decisionmaking statistics
- WebSocket stream for real-time updates

Usage:
    GET    /api/v1/deliberations/active    - List active vetted decisionmaking sessions
    GET    /api/v1/deliberations/stats     - Get aggregate statistics
    GET    /api/v1/deliberations/{id}      - Get vetted decisionmaking details
    WS     /api/v1/deliberations/stream    - Real-time event stream
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from aragora.config import DEFAULT_ROUNDS
from aragora.memory.debate_store import get_debate_store
from aragora.server.handlers.base import BaseHandler
from aragora.server.handlers.utils.decorators import require_permission
from aragora.server.handlers.utils.responses import error_dict

# RBAC imports - graceful fallback if not available
AuthorizationContext: Any
check_permission: Any
try:
    from aragora.rbac import AuthorizationContext, check_permission

    RBAC_AVAILABLE = True
except ImportError:
    RBAC_AVAILABLE = False
    AuthorizationContext = None
    check_permission = None

from aragora.server.handlers.utils.rbac_guard import rbac_fail_closed

# JWT auth import for extracting user context
extract_user_from_request: Any
try:
    from aragora.billing.jwt_auth import extract_user_from_request

    JWT_AUTH_AVAILABLE = True
except ImportError:
    JWT_AUTH_AVAILABLE = False
    extract_user_from_request = None

logger = logging.getLogger(__name__)

# In-memory deliberation tracking
_active_deliberations: dict[str, dict[str, Any]] = {}
_stream_clients: list[asyncio.Queue[dict[str, Any]]] = []
_stats: dict[str, Any] = {
    "active_count": 0,
    "completed_today": 0,
    "average_consensus_time": 0,
    "average_rounds": 0,
    "top_agents": [],
}


class DeliberationsHandler(BaseHandler):
    """
    Handler for vetted decisionmaking dashboard endpoints.

    Provides visibility into multi-agent vetted decisionmaking sessions across the system.

    RBAC Permissions:
    - analytics.read - View active deliberations, stats, and details
    - analytics.read - Subscribe to deliberation stream
    """

    ROUTES = [
        "/api/v1/deliberations/active",
        "/api/v1/deliberations/stats",
        "/api/v1/deliberations/stream",
        "/api/v1/deliberations/{deliberation_id}",
    ]
    _LIVE_DELIVERATION_STATUSES = {"initializing", "active", "consensus_forming"}

    def _get_auth_context(self, request: Any) -> AuthorizationContext | None:
        """Build RBAC authorization context from request.

        Returns None if RBAC/JWT auth is not available (allows request in dev mode).
        """
        if not RBAC_AVAILABLE or not JWT_AUTH_AVAILABLE or AuthorizationContext is None:
            return None

        try:
            # Extract user from JWT token
            auth_ctx = extract_user_from_request(request, user_store=None)
            if not auth_ctx or not auth_ctx.is_authenticated:
                return None

            # Build RBAC context
            roles = set([auth_ctx.role]) if hasattr(auth_ctx, "role") and auth_ctx.role else set()

            return AuthorizationContext(
                user_id=auth_ctx.user_id,
                roles=roles,
                org_id=getattr(auth_ctx, "org_id", None),
            )
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.debug("Failed to extract auth context: %s", e)
            return None

    def _check_rbac_permission(
        self, request: Any, permission_key: str
    ) -> tuple[dict[str, Any], int] | None:
        """Check RBAC permission.

        Returns None if allowed, or an error response tuple if denied.
        If RBAC is not available, allows the request (development mode).
        In production, denies access when RBAC is unavailable (fail-closed).
        """
        if not RBAC_AVAILABLE or check_permission is None:
            if rbac_fail_closed():
                return ({"error": "Service unavailable: access control module not loaded"}, 503)
            return None

        rbac_ctx = self._get_auth_context(request)
        if not rbac_ctx:
            # No auth context - in dev mode, allow request
            return None

        decision = check_permission(rbac_ctx, permission_key)
        if not decision.allowed:
            logger.warning(
                "RBAC denied: user=%s permission=%s reason=%s",
                rbac_ctx.user_id,
                permission_key,
                decision.reason,
            )
            return (error_dict("Permission denied", code="FORBIDDEN"), 403)

        return None

    @require_permission("debates:read")
    async def handle_request(self, request: Any) -> Any:
        """Route request to appropriate handler."""
        path = request.path
        method = request.method

        # Active deliberations - requires analytics.read
        if path == "/api/v1/deliberations/active" and method == "GET":
            if rbac_error := self._check_rbac_permission(request, "analytics.read"):
                return rbac_error
            return await self._get_active_deliberations(request)

        # Stats - requires analytics.read
        if path == "/api/v1/deliberations/stats" and method == "GET":
            if rbac_error := self._check_rbac_permission(request, "analytics.read"):
                return rbac_error
            return await self._get_stats(request)

        # WebSocket stream - requires analytics.read for subscribing to updates
        if path == "/api/v1/deliberations/stream":
            if rbac_error := self._check_rbac_permission(request, "analytics.read"):
                return rbac_error
            return await self._handle_stream(request)

        # Single deliberation - requires analytics.read
        if path.startswith("/api/v1/deliberations/") and method == "GET":
            deliberation_id = path.split("/")[-1]
            if deliberation_id not in ("active", "stats", "stream"):
                if rbac_error := self._check_rbac_permission(request, "analytics.read"):
                    return rbac_error
                return await self._get_deliberation(request, deliberation_id)

        return (error_dict("Not found", code="NOT_FOUND"), 404)

    async def _get_active_deliberations(self, request: Any) -> tuple[dict[str, Any], int]:
        """Get list of active vetted decisionmaking sessions."""
        try:
            # Try to get real deliberations from debate store
            deliberations = await self._fetch_active_from_store()

            return {
                "deliberations": deliberations,
                "count": len(deliberations),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, 200
        except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.error("Error fetching deliberations: %s", e)
            return (error_dict("Internal server error", code="INTERNAL_ERROR"), 500)

    async def _fetch_active_from_store(self) -> list[dict[str, Any]]:
        """Fetch active vetted decisionmaking sessions from the debate store."""
        deliberations = []
        for debate in self._get_recent_deliberations():
            status = self._map_debate_status(str(debate.get("status", "unknown")))
            if status in self._LIVE_DELIVERATION_STATUSES:
                deliberations.append(self._format_deliberation(debate))
        return deliberations

    def _map_debate_status(self, status: str) -> str:
        """Map debate status to deliberation status."""
        status_map = {
            "pending": "initializing",
            "running": "active",
            "streaming": "active",
            "voting": "consensus_forming",
            "complete": "complete",
            "completed": "complete",
            "failed": "failed",
            "error": "failed",
        }
        return status_map.get(status, status)

    def _format_deliberation(self, debate: dict[str, Any]) -> dict[str, Any]:
        """Format a debate as a deliberation."""
        agents = self._normalize_agents(debate.get("agents", []))

        messages = debate.get("messages", [])
        if not isinstance(messages, list):
            messages = []
        current_round = debate.get("current_round", 0)
        if not current_round and messages:
            current_round = max((m.get("round", 0) for m in messages), default=0)

        explicit_message_count = debate.get("message_count")
        if isinstance(explicit_message_count, int) and explicit_message_count >= 0:
            message_count = explicit_message_count
        else:
            message_count = len(messages)

        return {
            "id": debate.get("id", debate.get("debate_id", "")),
            "task": debate.get("task", debate.get("question", "")),
            "status": self._map_debate_status(debate.get("status", "unknown")),
            "agents": agents,
            "current_round": current_round,
            "total_rounds": debate.get("total_rounds", debate.get("rounds", DEFAULT_ROUNDS)),
            "consensus_score": debate.get("consensus_score", 0),
            "started_at": debate.get("started_at", debate.get("created_at", "")),
            "updated_at": debate.get("updated_at", debate.get("started_at", "")),
            "message_count": message_count,
            "votes": debate.get("votes", {}),
        }

    def _get_recent_deliberations(self, limit: int = 50) -> list[dict[str, Any]]:
        """Collect recent debates from durable and in-memory sources."""
        recent_by_id: dict[str, dict[str, Any]] = {}

        try:
            store = get_debate_store()
            if store:
                recent = store.get_recent(limit=limit)
                if isinstance(recent, list):
                    for debate in recent:
                        if not isinstance(debate, dict):
                            continue
                        debate_id = str(debate.get("id", debate.get("debate_id", ""))).strip()
                        key = debate_id or f"store-{len(recent_by_id)}"
                        recent_by_id[key] = debate
        except ImportError:
            pass

        for delib_id, delib in _active_deliberations.items():
            if isinstance(delib, dict):
                recent_by_id[str(delib_id)] = delib

        return list(recent_by_id.values())

    def _normalize_agents(self, agents: Any) -> list[str]:
        """Normalize agent lists that may mix strings and lightweight objects."""
        if isinstance(agents, str):
            raw_agents: list[Any] = [agents]
        elif isinstance(agents, list):
            raw_agents = agents
        else:
            raw_agents = []

        normalized: list[str] = []
        seen: set[str] = set()
        for agent in raw_agents:
            agent_id = ""
            if isinstance(agent, str):
                agent_id = agent.strip()
            elif isinstance(agent, dict):
                for key in ("agent_id", "agent", "name", "id"):
                    value = agent.get(key)
                    if isinstance(value, str) and value.strip():
                        agent_id = value.strip()
                        break
            if agent_id and agent_id not in seen:
                normalized.append(agent_id)
                seen.add(agent_id)
        return normalized

    @staticmethod
    def _normalize_ratio(value: Any, default: float = 0.0) -> float:
        """Clamp ratio-like values into the UI's expected 0..1 range."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = default
        return max(0.0, min(1.0, number))

    def _extract_message_agent(self, message: Any) -> str | None:
        """Extract the originating agent name from a stored debate message."""
        if not isinstance(message, dict):
            return None

        for key in ("agent", "author", "name"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, dict):
                for nested_key in ("agent", "agent_id", "name", "id"):
                    nested_value = value.get(nested_key)
                    if isinstance(nested_value, str) and nested_value.strip():
                        return nested_value.strip()
        return None

    def _derive_top_agents(
        self, debates: list[dict[str, Any]], limit: int = 5
    ) -> list[dict[str, Any]]:
        """Build stable agent influence metrics from active debate payloads."""
        aggregates: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "debates": 0.0,
                "message_count": 0.0,
                "influence_total": 0.0,
                "consensus_total": 0.0,
                "confidence_total": 0.0,
            }
        )

        for debate in debates:
            if not isinstance(debate, dict):
                continue

            status = self._map_debate_status(str(debate.get("status", "unknown")))
            if status not in self._LIVE_DELIVERATION_STATUSES:
                continue

            agents = self._normalize_agents(debate.get("agents", []))
            if not agents:
                continue

            messages = debate.get("messages", [])
            if not isinstance(messages, list):
                messages = []
            votes = debate.get("votes", {})
            if not isinstance(votes, dict):
                votes = {}

            message_counts = {agent: 0 for agent in agents}
            confidence_totals: dict[str, float] = defaultdict(float)
            confidence_counts: dict[str, int] = defaultdict(int)
            total_known_messages = 0

            for message in messages:
                agent_id = self._extract_message_agent(message)
                if not agent_id or agent_id not in message_counts:
                    continue

                message_counts[agent_id] += 1
                total_known_messages += 1

                if isinstance(message, dict) and "confidence" in message:
                    confidence = self._normalize_ratio(message.get("confidence"), default=-1.0)
                    if confidence >= 0:
                        confidence_totals[agent_id] += confidence
                        confidence_counts[agent_id] += 1

            total_votes = 0.0
            normalized_votes: dict[str, float] = {}
            for agent in agents:
                try:
                    vote_value = float(votes.get(agent, 0) or 0)
                except (TypeError, ValueError):
                    vote_value = 0.0
                if vote_value > 0:
                    normalized_votes[agent] = vote_value
                    total_votes += vote_value

            total_rounds = debate.get("total_rounds", debate.get("rounds", DEFAULT_ROUNDS))
            current_round = debate.get("current_round", 0)
            try:
                round_progress = float(current_round) / max(float(total_rounds), 1.0)
            except (TypeError, ValueError):
                round_progress = 0.0
            round_progress = self._normalize_ratio(round_progress)
            consensus_score = self._normalize_ratio(
                debate.get("consensus_score"), default=round_progress
            )

            for agent in agents:
                actual_messages = message_counts.get(agent, 0)
                message_share = (
                    actual_messages / total_known_messages
                    if total_known_messages > 0
                    else 1 / len(agents)
                )
                vote_share = (
                    normalized_votes.get(agent, 0.0) / total_votes
                    if total_votes > 0
                    else message_share
                )
                average_confidence = (
                    confidence_totals[agent] / confidence_counts[agent]
                    if confidence_counts[agent] > 0
                    else consensus_score
                )
                consensus_contribution = (
                    vote_share if total_votes > 0 else consensus_score * message_share
                )
                influence_score = (
                    (message_share * 0.5)
                    + (consensus_contribution * 0.3)
                    + (average_confidence * 0.2)
                )

                aggregate = aggregates[agent]
                aggregate["debates"] += 1
                aggregate["message_count"] += actual_messages
                aggregate["influence_total"] += self._normalize_ratio(influence_score)
                aggregate["consensus_total"] += self._normalize_ratio(consensus_contribution)
                aggregate["confidence_total"] += self._normalize_ratio(average_confidence)

        top_agents = []
        for agent_id, metrics in aggregates.items():
            debates_participated = max(int(metrics["debates"]), 1)
            top_agents.append(
                {
                    "agent_id": agent_id,
                    "influence_score": round(
                        self._normalize_ratio(metrics["influence_total"] / debates_participated),
                        3,
                    ),
                    "message_count": int(metrics["message_count"]),
                    "consensus_contributions": round(
                        self._normalize_ratio(metrics["consensus_total"] / debates_participated),
                        3,
                    ),
                    "average_confidence": round(
                        self._normalize_ratio(metrics["confidence_total"] / debates_participated),
                        3,
                    ),
                }
            )

        top_agents.sort(
            key=lambda agent: (
                -agent["influence_score"],
                -agent["message_count"],
                agent["agent_id"],
            )
        )
        return top_agents[:limit]

    async def _get_stats(self, request: Any) -> tuple[dict[str, Any], int]:
        """Get deliberation statistics."""
        try:
            # Calculate live stats
            active = await self._fetch_active_from_store()
            recent_debates = self._get_recent_deliberations()
            active_count = len(
                [d for d in active if d["status"] in ("active", "consensus_forming")]
            )

            # Get completed today
            completed_today = 0
            try:
                store = get_debate_store()
                if store:
                    today = datetime.now(timezone.utc).date()
                    recent = store.get_recent(limit=100)
                    for debate in recent:
                        if debate.get("status") in ("complete", "completed"):
                            completed_at = debate.get("completed_at", debate.get("updated_at", ""))
                            if completed_at:
                                try:
                                    completed_date = datetime.fromisoformat(
                                        completed_at.replace("Z", "+00:00")
                                    ).date()
                                    if completed_date == today:
                                        completed_today += 1
                                except (ValueError, AttributeError):
                                    pass
            except ImportError:
                pass

            return {
                "active_count": active_count,
                "completed_today": completed_today,
                "average_consensus_time": _stats.get("average_consensus_time", 420),
                "average_rounds": _stats.get("average_rounds", 4.2),
                "top_agents": self._derive_top_agents(recent_debates),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }, 200
        except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.error("Error fetching stats: %s", e)
            return (error_dict("Internal server error", code="INTERNAL_ERROR"), 500)

    async def _get_deliberation(
        self, request: Any, deliberation_id: str
    ) -> tuple[dict[str, Any], int]:
        """Get a single deliberation by ID."""
        try:
            # Check in-memory first
            if deliberation_id in _active_deliberations:
                return _active_deliberations[deliberation_id], 200

            # Try debate store
            try:
                store = get_debate_store()
                if store:
                    debate = store.get(deliberation_id)
                    if debate:
                        return self._format_deliberation(debate), 200
            except ImportError:
                pass

            return (error_dict("Deliberation not found", code="NOT_FOUND"), 404)
        except (KeyError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.error("Error fetching deliberation %s: %s", deliberation_id, e)
            return (error_dict("Internal server error", code="INTERNAL_ERROR"), 500)

    async def _handle_stream(self, request: Any) -> Any:
        """Handle WebSocket stream for real-time updates."""
        # WebSocket handling would be done at the server level
        # This returns the stream configuration
        return {
            "type": "websocket",
            "path": "/api/v1/deliberations/stream",
            "events": [
                "agent_message",
                "vote",
                "consensus_progress",
                "round_complete",
                "deliberation_complete",
            ],
        }, 200


# Module-level functions for event broadcasting
async def broadcast_deliberation_event(event: dict[str, Any]) -> None:
    """Broadcast an event to all connected stream clients."""
    for queue in _stream_clients:
        try:
            await queue.put(event)
        except (ValueError, TypeError, OSError) as e:
            logger.debug("Failed to broadcast event to stream client: %s", e)


def register_deliberation(deliberation_id: str, data: dict[str, Any]) -> None:
    """Register an active deliberation."""
    _active_deliberations[deliberation_id] = {
        "id": deliberation_id,
        **data,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def update_deliberation(deliberation_id: str, updates: dict[str, Any]) -> None:
    """Update a deliberation's data."""
    if deliberation_id in _active_deliberations:
        _active_deliberations[deliberation_id].update(updates)
        _active_deliberations[deliberation_id]["updated_at"] = datetime.now(
            timezone.utc
        ).isoformat()


def complete_deliberation(deliberation_id: str) -> None:
    """Mark a deliberation as complete and remove from active."""
    if deliberation_id in _active_deliberations:
        _active_deliberations[deliberation_id]["status"] = "complete"
        # Update stats
        _stats["completed_today"] = _stats.get("completed_today", 0) + 1


# Handler instance (lazy initialization - instantiated by unified_server.py)
