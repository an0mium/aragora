"""
Playground Handler - Public demo endpoint for the aragora-debate engine.

Stability: STABLE

Allows anyone to run a mock debate without authentication or API keys.
Uses StyledMockAgent from the aragora-debate standalone package for
deterministic, zero-dependency debates.

Routes:
    POST /api/v1/playground/debate  - Run a mock debate
    GET  /api/v1/playground/status  - Health check for the playground
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Literal

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiting (in-memory, per-IP, 5 req/min)
# ---------------------------------------------------------------------------

_PLAYGROUND_RATE_LIMIT = 5  # requests per window
_PLAYGROUND_RATE_WINDOW = 60.0  # seconds

# IP -> list of timestamps
_request_timestamps: dict[str, list[float]] = {}


def _check_rate_limit(client_ip: str) -> tuple[bool, int]:
    """Check whether the client IP is within the rate limit.

    Returns:
        (allowed, retry_after_seconds)
    """
    now = time.monotonic()
    cutoff = now - _PLAYGROUND_RATE_WINDOW

    timestamps = _request_timestamps.get(client_ip, [])
    # Prune old entries
    timestamps = [t for t in timestamps if t > cutoff]

    if len(timestamps) >= _PLAYGROUND_RATE_LIMIT:
        oldest_in_window = timestamps[0]
        retry_after = int(oldest_in_window + _PLAYGROUND_RATE_WINDOW - now) + 1
        _request_timestamps[client_ip] = timestamps
        return False, max(retry_after, 1)

    timestamps.append(now)
    _request_timestamps[client_ip] = timestamps
    return True, 0


def _reset_rate_limits() -> None:
    """Reset all rate limit state. Used by tests."""
    _request_timestamps.clear()


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

_MAX_TOPIC_LENGTH = 500
_MAX_ROUNDS = 2
_MAX_AGENTS = 5
_MIN_AGENTS = 2

_DEFAULT_TOPIC = "Should we use microservices or a monolith?"
_DEFAULT_ROUNDS = 2
_DEFAULT_AGENTS = 3

_AGENT_STYLES: list[Literal["supportive", "critical", "balanced", "contrarian"]] = [
    "supportive", "critical", "balanced", "contrarian",
]


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class PlaygroundHandler(BaseHandler):
    """HTTP handler for the public playground demo.

    Runs zero-cost mock debates using StyledMockAgent from aragora-debate.
    No authentication required. Rate limited to 5 requests per minute per IP.
    """

    ROUTES = [
        "/api/v1/playground/debate",
        "/api/v1/playground/status",
    ]

    def __init__(self, ctx: dict | None = None):
        self.ctx = ctx or {}

    def can_handle(self, path: str) -> bool:
        return path in (
            "/api/v1/playground/debate",
            "/api/v1/playground/status",
        )

    # ------------------------------------------------------------------
    # GET /api/v1/playground/status
    # ------------------------------------------------------------------

    def handle(
        self, path: str, query_params: dict[str, Any], handler: Any,
    ) -> HandlerResult | None:
        if path == "/api/v1/playground/status":
            return self._handle_status()
        return None

    def _handle_status(self) -> HandlerResult:
        return json_response({
            "status": "ok",
            "engine": "aragora-debate",
            "mock_agents": True,
            "max_rounds": _MAX_ROUNDS,
            "max_agents": _MAX_AGENTS,
            "rate_limit": f"{_PLAYGROUND_RATE_LIMIT} requests per {int(_PLAYGROUND_RATE_WINDOW)}s",
        })

    # ------------------------------------------------------------------
    # POST /api/v1/playground/debate
    # ------------------------------------------------------------------

    def handle_post(
        self, path: str, query_params: dict[str, Any], handler: Any,
    ) -> HandlerResult | None:
        if path != "/api/v1/playground/debate":
            return None

        # Rate limiting
        client_ip = "unknown"
        if handler and hasattr(handler, "client_address"):
            addr = handler.client_address
            if isinstance(addr, (list, tuple)) and len(addr) >= 1:
                client_ip = str(addr[0])

        allowed, retry_after = _check_rate_limit(client_ip)
        if not allowed:
            return json_response(
                {
                    "error": "Rate limit exceeded. Please try again later.",
                    "code": "rate_limit_exceeded",
                    "retry_after": retry_after,
                },
                status=429,
            )

        # Parse body
        body = self.read_json_body(handler) if handler else {}
        if body is None:
            body = {}

        topic = str(body.get("topic", _DEFAULT_TOPIC) or _DEFAULT_TOPIC).strip()
        if not topic:
            topic = _DEFAULT_TOPIC
        if len(topic) > _MAX_TOPIC_LENGTH:
            return error_response(
                f"Topic must be {_MAX_TOPIC_LENGTH} characters or less", 400,
            )

        try:
            rounds = int(body.get("rounds", _DEFAULT_ROUNDS))
        except (TypeError, ValueError):
            rounds = _DEFAULT_ROUNDS
        rounds = max(1, min(rounds, _MAX_ROUNDS))

        try:
            agent_count = int(body.get("agents", _DEFAULT_AGENTS))
        except (TypeError, ValueError):
            agent_count = _DEFAULT_AGENTS
        agent_count = max(_MIN_AGENTS, min(agent_count, _MAX_AGENTS))

        return self._run_debate(topic, rounds, agent_count)

    def _run_debate(
        self, topic: str, rounds: int, agent_count: int,
    ) -> HandlerResult:
        try:
            from aragora_debate.styled_mock import StyledMockAgent
            from aragora_debate.arena import Arena
            from aragora_debate.types import DebateConfig
        except ImportError:
            logger.warning("aragora-debate package not available for playground")
            return error_response(
                "Playground unavailable: aragora-debate package not installed",
                503,
            )

        # Build agents with rotating styles
        agent_names = ["analyst", "critic", "moderator", "contrarian", "synthesizer"]
        agents = []
        for i in range(agent_count):
            name = agent_names[i] if i < len(agent_names) else f"agent_{i}"
            style = _AGENT_STYLES[i % len(_AGENT_STYLES)]
            agents.append(StyledMockAgent(name, style=style))

        config = DebateConfig(
            rounds=rounds,
            early_stopping=True,
        )

        arena = Arena(
            question=topic,
            agents=agents,  # type: ignore[arg-type]  # StyledMockAgent extends Agent
            config=config,
        )

        try:
            result = asyncio.run(arena.run())
        except RuntimeError:
            # Already in an event loop -- use a helper
            try:
                import nest_asyncio  # type: ignore[import-untyped]
                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(arena.run())
            except ImportError:
                # Fallback: create a new loop in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(asyncio.run, arena.run()).result(timeout=30)

        # Build response
        critiques_out = []
        for c in result.critiques:
            critiques_out.append({
                "agent": c.agent,
                "target_agent": c.target_agent,
                "issues": c.issues,
                "suggestions": c.suggestions,
                "severity": c.severity,
            })

        votes_out = []
        for v in result.votes:
            votes_out.append({
                "agent": v.agent,
                "choice": v.choice,
                "confidence": v.confidence,
                "reasoning": v.reasoning,
            })

        receipt_data = None
        receipt_hash = None
        if result.receipt:
            receipt_data = result.receipt.to_dict()
            receipt_hash = result.receipt.signature

        response = {
            "id": result.id,
            "topic": result.task,
            "status": result.status,
            "rounds_used": result.rounds_used,
            "consensus_reached": result.consensus_reached,
            "confidence": result.confidence,
            "verdict": result.verdict.value if result.verdict else None,
            "duration_seconds": round(result.duration_seconds, 3),
            "participants": result.participants,
            "proposals": result.proposals,
            "critiques": critiques_out,
            "votes": votes_out,
            "dissenting_views": result.dissenting_views,
            "final_answer": result.final_answer,
            "receipt": receipt_data,
            "receipt_hash": receipt_hash,
        }

        return json_response(response)
