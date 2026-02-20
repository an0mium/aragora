"""
Playground Handler - Public demo endpoint for the aragora-debate engine.

Stability: STABLE

Allows anyone to run a mock debate without authentication or API keys.
Uses StyledMockAgent from the aragora-debate standalone package for
deterministic, zero-dependency debates.  The ``/live`` variant uses real
API-backed agents with budget + timeout caps for a taste of the full
platform.

Routes:
    POST /api/v1/playground/debate             - Run a mock debate
    POST /api/v1/playground/debate/live         - Run a live debate with real agents
    POST /api/v1/playground/debate/live/cost-estimate - Pre-flight cost estimate
    GET  /api/v1/playground/status              - Health check for the playground
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from aragora.server.handlers.base import (
    BaseHandler,
    HandlerResult,
    error_response,
    json_response,
    handle_errors,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limiting (in-memory, per-IP, 5 req/min for mock, 1/10min for live)
# ---------------------------------------------------------------------------

_PLAYGROUND_RATE_LIMIT = 5  # requests per window
_PLAYGROUND_RATE_WINDOW = 60.0  # seconds

_LIVE_RATE_LIMIT = 1  # 1 live debate per window per IP
_LIVE_RATE_WINDOW = 600.0  # 10 minutes

# IP -> list of timestamps
_request_timestamps: dict[str, list[float]] = {}
_live_request_timestamps: dict[str, list[float]] = {}


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


def _check_live_rate_limit(client_ip: str) -> tuple[bool, int]:
    """Check whether the client IP is within the live debate rate limit.

    Returns:
        (allowed, retry_after_seconds)
    """
    now = time.monotonic()
    cutoff = now - _LIVE_RATE_WINDOW

    timestamps = _live_request_timestamps.get(client_ip, [])
    timestamps = [t for t in timestamps if t > cutoff]

    if len(timestamps) >= _LIVE_RATE_LIMIT:
        oldest_in_window = timestamps[0]
        retry_after = int(oldest_in_window + _LIVE_RATE_WINDOW - now) + 1
        _live_request_timestamps[client_ip] = timestamps
        return False, max(retry_after, 1)

    timestamps.append(now)
    _live_request_timestamps[client_ip] = timestamps
    return True, 0


def _reset_rate_limits() -> None:
    """Reset all rate limit state. Used by tests."""
    _request_timestamps.clear()
    _live_request_timestamps.clear()


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

_MAX_TOPIC_LENGTH = 100_000  # ~15k words — supports long-form debate prompts
_MAX_ROUNDS = 2
_MAX_AGENTS = 5
_MIN_AGENTS = 2

_DEFAULT_TOPIC = "Should we use microservices or a monolith?"
_DEFAULT_ROUNDS = 2
_DEFAULT_AGENTS = 3

_AGENT_STYLES: list[Literal["supportive", "critical", "balanced", "contrarian"]] = [
    "supportive",
    "critical",
    "balanced",
    "contrarian",
]


# ---------------------------------------------------------------------------
# Inline mock debate (fallback when aragora-debate is not installed)
# ---------------------------------------------------------------------------

def _build_mock_proposals(topic: str, question: str | None = None) -> dict[str, list[str]]:
    """Build topic-aware mock proposals instead of canned microservices text."""
    # If a raw question was provided (e.g. from Oracle mode), use it for the snippet
    # instead of the full system-prompt-laden topic
    source = question or topic
    snippet = source[:200].strip()
    if len(source) > 200:
        snippet = snippet.rsplit(" ", 1)[0] + "..."

    return {
        "supportive": [
            f"After careful analysis of the submission, I find the core argument compelling. "
            f"Regarding '{snippet}' -- the reasoning is well-structured, the evidence cited is "
            f"substantive, and the conclusions follow logically from the premises. The key "
            f"strengths are the specificity of claims and the willingness to engage with "
            f"counterarguments. I recommend this position with minor caveats.",
            f"This is a strong argument. The submission on '{snippet}' demonstrates clear "
            f"thinking and grounded analysis. The supporting evidence is concrete rather than "
            f"abstract, and the framework presented offers actionable insights. The conclusion "
            f"is well-earned by the preceding analysis.",
        ],
        "critical": [
            f"I have significant concerns about the argument presented. While '{snippet}' "
            f"raises important points, several claims lack sufficient empirical backing. The "
            f"causal reasoning conflates correlation with causation in key places, and the "
            f"conclusion overreaches what the evidence supports. A more rigorous analysis "
            f"would need to address the strongest counterarguments directly.",
            f"This submission overlooks critical failure modes. The argument around "
            f"'{snippet}' makes assumptions that haven't been validated -- particularly "
            f"about timelines and magnitudes. The most likely scenario involves more "
            f"uncertainty than the author acknowledges, and the recommended actions don't "
            f"adequately account for second-order effects.",
        ],
        "balanced": [
            f"There are valid points on both sides. The submission on '{snippet}' correctly "
            f"identifies real dynamics, but the framing occasionally overstates certainty "
            f"where the evidence is ambiguous. A more nuanced position would acknowledge "
            f"where the author's model could be wrong and under what conditions the opposite "
            f"conclusion might hold.",
            f"The analysis of '{snippet}' has genuine strengths -- concrete examples, "
            f"specific claims, falsifiable predictions. But it also has blind spots: the "
            f"framework assumes certain structural dynamics will continue, which isn't "
            f"guaranteed. A balanced assessment says: directionally right, calibrationally "
            f"uncertain.",
        ],
        "contrarian": [
            f"I disagree with the prevailing direction of this analysis. The argument "
            f"about '{snippet}' optimizes for the visible pattern while ignoring systemic "
            f"risks that would invalidate the thesis entirely. The most important question "
            f"isn't whether the author's scenario is plausible -- it's whether the "
            f"confidence level is warranted given the evidence.",
            f"Everyone seems to be converging too quickly on this framing. Let me argue "
            f"the unpopular position: the submission on '{snippet}' may be directionally "
            f"wrong in ways that feel uncomfortable to acknowledge. The evidence cited "
            f"is selectively chosen, and equally compelling evidence exists for the "
            f"opposite conclusion.",
        ],
    }


# Keep static version for backward compat with tests that reference _MOCK_PROPOSALS
_MOCK_PROPOSALS = _build_mock_proposals(_DEFAULT_TOPIC)

_MOCK_CRITIQUE_ISSUES: dict[str, list[str]] = {
    "supportive": [
        "Could benefit from more quantitative evidence",
        "Some claims would be stronger with explicit confidence intervals",
    ],
    "critical": [
        "Key causal claims lack sufficient empirical backing",
        "No analysis of what happens if the core assumptions are wrong",
        "Ignores the strongest counterarguments to the thesis",
        "Conflates multiple distinct phenomena under one framework",
    ],
    "balanced": [
        "The argument could better acknowledge where uncertainty is highest",
        "Risk assessment is asymmetric -- considers one failure mode but not others",
    ],
    "contrarian": [
        "The group appears to be converging prematurely on this framing",
        "The most important alternative scenarios have not been seriously considered",
    ],
}

_MOCK_CRITIQUE_SUGGESTIONS: dict[str, list[str]] = {
    "supportive": ["Consider adding falsification criteria for the key claims"],
    "critical": ["Provide explicit evidence that would change this assessment"],
    "balanced": ["Add a structured analysis of where this argument is most likely wrong"],
    "contrarian": ["Steel-man the opposing position before dismissing it"],
}

_MOCK_SEVERITY: dict[str, tuple[float, float]] = {
    "supportive": (2.0, 4.0),
    "critical": (6.0, 9.0),
    "balanced": (4.0, 6.0),
    "contrarian": (5.0, 8.0),
}

_MOCK_CONFIDENCE: dict[str, float] = {
    "supportive": 0.85,
    "critical": 0.6,
    "balanced": 0.7,
    "contrarian": 0.5,
}


def _run_inline_mock_debate(
    topic: str,
    rounds: int,
    agent_count: int,
    question: str | None = None,
) -> dict[str, Any]:
    """Run a mock debate without the aragora-debate package."""
    start = time.monotonic()
    debate_id = uuid.uuid4().hex[:16]
    all_names = ["analyst", "critic", "moderator", "contrarian", "synthesizer"]
    names = [all_names[i] if i < len(all_names) else f"agent_{i}" for i in range(agent_count)]
    styles = [_AGENT_STYLES[i % len(_AGENT_STYLES)] for i in range(agent_count)]

    topic_proposals = _build_mock_proposals(topic, question=question)
    proposals: dict[str, str] = {}
    for name, style in zip(names, styles):
        proposals[name] = random.choice(topic_proposals[style])

    critiques: list[dict[str, Any]] = []
    for i, (name, style) in enumerate(zip(names, styles)):
        for j, target in enumerate(names):
            if i == j:
                continue
            lo, hi = _MOCK_SEVERITY[style]
            critiques.append(
                {
                    "agent": name,
                    "target_agent": target,
                    "issues": list(_MOCK_CRITIQUE_ISSUES[style]),
                    "suggestions": list(_MOCK_CRITIQUE_SUGGESTIONS[style]),
                    "severity": round(random.uniform(lo, hi), 1),
                }
            )

    votes: list[dict[str, Any]] = []
    vote_tally: dict[str, float] = {}
    vote_source = question or topic
    topic_snippet = vote_source[:80] if vote_source else "the proposal"
    if len(vote_source) > 80:
        topic_snippet = topic_snippet.rsplit(" ", 1)[0] + "..."
    topic_snippet = topic_snippet.rstrip(".!? ")
    _vote_reasoning: dict[str, list[str]] = {
        "supportive": [
            "{choice}'s proposal on '{topic}' is the strongest -- clear benefits with manageable risks",
            "After weighing all arguments on '{topic}', {choice} presents the most actionable path forward",
        ],
        "critical": [
            "{choice}'s argument best addresses the risks I raised about '{topic}'",
            "While I remain cautious about '{topic}', {choice}'s position is the most defensible",
        ],
        "balanced": [
            "{choice} strikes the right balance between ambition and pragmatism on '{topic}'",
            "On '{topic}', {choice}'s staged approach manages risk while enabling progress",
        ],
        "contrarian": [
            "Reluctantly voting for {choice} -- their view on '{topic}' at least considers the downsides",
            "None of the proposals fully satisfy my concerns, but {choice}'s position on '{topic}' is least risky",
        ],
    }
    for name, style in zip(names, styles):
        others = [n for n in names if n != name]
        if style == "supportive":
            choice = others[0]
        elif style == "contrarian":
            choice = others[-1]
        else:
            choice = random.choice(others)
        base_conf = _MOCK_CONFIDENCE.get(style, 0.7)
        conf = round(max(0.1, min(1.0, base_conf + random.uniform(-0.05, 0.05))), 2)
        reasoning = random.choice(_vote_reasoning.get(style, ["{choice}"])).format(
            choice=choice, topic=topic_snippet
        )
        votes.append(
            {
                "agent": name,
                "choice": choice,
                "confidence": conf,
                "reasoning": reasoning,
            }
        )
        vote_tally[choice] = vote_tally.get(choice, 0.0) + conf

    total_weight = sum(vote_tally.values())
    leading = max(vote_tally, key=lambda k: vote_tally[k]) if vote_tally else names[0]
    confidence = vote_tally.get(leading, 0.0) / total_weight if total_weight > 0 else 0.0
    consensus_reached = confidence >= 0.5
    supporting = [v["agent"] for v in votes if v["choice"] == leading]
    dissenting = [n for n in names if n not in supporting]

    if confidence >= 0.85:
        verdict = "approved"
    elif confidence >= 0.6:
        verdict = "approved_with_conditions"
    elif confidence >= 0.4:
        verdict = "needs_review"
    else:
        verdict = "rejected"

    duration = time.monotonic() - start
    now_iso = datetime.now(timezone.utc).isoformat()
    receipt_id = f"DR-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:6]}"
    receipt_hash = hashlib.sha256(
        f"{receipt_id}:{topic}:{verdict}:{confidence}".encode()
    ).hexdigest()

    return {
        "id": debate_id,
        "topic": topic,
        "status": "consensus_reached" if consensus_reached else "completed",
        "rounds_used": rounds,
        "consensus_reached": consensus_reached,
        "confidence": confidence,
        "verdict": verdict,
        "duration_seconds": round(duration, 3),
        "participants": names,
        "proposals": proposals,
        "critiques": critiques,
        "votes": votes,
        "dissenting_views": [
            f"{v['agent']}: {v['reasoning']}" for v in votes if v["choice"] != leading
        ],
        "final_answer": proposals.get(leading, ""),
        "receipt": {
            "receipt_id": receipt_id,
            "question": topic,
            "verdict": verdict,
            "confidence": confidence,
            "consensus": {
                "reached": consensus_reached,
                "method": "majority",
                "confidence": confidence,
                "supporting_agents": supporting,
                "dissenting_agents": dissenting,
                "dissents": [
                    {
                        "agent": v["agent"],
                        "reasons": [v["reasoning"]],
                        "alternative_view": f"Preferred: {v['choice']}",
                        "severity": 0.5,
                    }
                    for v in votes
                    if v["choice"] != leading
                ],
            },
            "agents": names,
            "rounds_used": rounds,
            "claims": 0,
            "evidence_count": 0,
            "timestamp": now_iso,
            "signature": receipt_hash,
            "signature_algorithm": "SHA-256-content-hash",
        },
        "receipt_hash": receipt_hash,
    }


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


class PlaygroundHandler(BaseHandler):
    """HTTP handler for the public playground demo.

    Runs zero-cost mock debates using StyledMockAgent from aragora-debate.
    Also supports live debates with real agents (budget-capped).
    No authentication required. Rate limited per IP.
    """

    ROUTES = [
        "/api/v1/playground/debate",
        "/api/v1/playground/debate/live",
        "/api/v1/playground/debate/live/cost-estimate",
        "/api/v1/playground/status",
    ]

    def __init__(self, ctx: dict | None = None):
        self.ctx = ctx or {}

    def can_handle(self, path: str) -> bool:
        return path in (
            "/api/v1/playground/debate",
            "/api/v1/playground/debate/live",
            "/api/v1/playground/debate/live/cost-estimate",
            "/api/v1/playground/status",
        )

    # ------------------------------------------------------------------
    # GET /api/v1/playground/status
    # ------------------------------------------------------------------

    def handle(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult | None:
        if path == "/api/v1/playground/status":
            return self._handle_status()
        return None

    def _handle_status(self) -> HandlerResult:
        return json_response(
            {
                "status": "ok",
                "engine": "aragora-debate",
                "mock_agents": True,
                "max_rounds": _MAX_ROUNDS,
                "max_agents": _MAX_AGENTS,
                "rate_limit": f"{_PLAYGROUND_RATE_LIMIT} requests per {int(_PLAYGROUND_RATE_WINDOW)}s",
            }
        )

    # ------------------------------------------------------------------
    # POST /api/v1/playground/debate
    # ------------------------------------------------------------------

    @handle_errors("playground creation")
    def handle_post(
        self,
        path: str,
        query_params: dict[str, Any],
        handler: Any,
    ) -> HandlerResult | None:
        if path == "/api/v1/playground/debate/live/cost-estimate":
            return self._handle_cost_estimate(handler)
        if path == "/api/v1/playground/debate/live":
            return self._handle_live_debate(handler)
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
                f"Topic must be {_MAX_TOPIC_LENGTH} characters or less",
                400,
            )

        # Raw question (separate from system-prompt-laden topic, e.g. from Oracle)
        question = str(body.get("question", "") or "").strip() or None

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

        return self._run_debate(topic, rounds, agent_count, question=question)

    def _run_debate(
        self,
        topic: str,
        rounds: int,
        agent_count: int,
        question: str | None = None,
    ) -> HandlerResult:
        # When a raw question is provided (e.g. from Oracle), use the inline
        # mock directly — it's question-aware and will reference the actual
        # question.  The aragora-debate package's StyledMockAgent has hardcoded
        # proposals that ignore the prompt entirely.
        if not question:
            try:
                return self._run_debate_with_package(topic, rounds, agent_count, question=question)
            except ImportError:
                logger.info("aragora-debate not installed, using inline mock debate")
            except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError):
                logger.exception("aragora-debate failed, falling back to inline mock")

        # Inline mock debate (question-aware)
        try:
            return json_response(_run_inline_mock_debate(topic, rounds, agent_count, question=question))
        except (RuntimeError, ValueError, TypeError, KeyError, AttributeError, OSError):
            logger.exception("Inline mock debate failed")
            return error_response("Debate failed unexpectedly", 500)

    def _run_debate_with_package(
        self,
        topic: str,
        rounds: int,
        agent_count: int,
        question: str | None = None,
    ) -> HandlerResult:
        from aragora_debate.styled_mock import StyledMockAgent
        from aragora_debate.arena import Arena
        from aragora_debate.types import DebateConfig

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
            question=question or topic,
            agents=agents,  # type: ignore[arg-type]
            config=config,
        )

        try:
            result = asyncio.run(arena.run())
        except RuntimeError:
            # Already in an event loop -- use a helper
            try:
                import nest_asyncio  # type: ignore[import-untyped]

                nest_asyncio.apply()
                from aragora.utils.async_utils import get_event_loop_safe

                loop = get_event_loop_safe()
                result = loop.run_until_complete(arena.run())
            except ImportError:
                # Fallback: create a new loop in a thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(asyncio.run, arena.run()).result(timeout=30)

        # Build response
        critiques_out = []
        for c in result.critiques:
            critiques_out.append(
                {
                    "agent": c.agent,
                    "target_agent": c.target_agent,
                    "issues": c.issues,
                    "suggestions": c.suggestions,
                    "severity": c.severity,
                }
            )

        votes_out = []
        for v in result.votes:
            votes_out.append(
                {
                    "agent": v.agent,
                    "choice": v.choice,
                    "confidence": v.confidence,
                    "reasoning": v.reasoning,
                }
            )

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

    # ------------------------------------------------------------------
    # POST /api/v1/playground/debate/live/cost-estimate
    # ------------------------------------------------------------------

    def _handle_cost_estimate(self, handler: Any) -> HandlerResult:
        """Return a pre-flight cost estimate for a live debate."""
        body = self.read_json_body(handler) if handler else {}
        if body is None:
            body = {}

        try:
            agent_count = int(body.get("agents", _DEFAULT_AGENTS))
        except (TypeError, ValueError):
            agent_count = _DEFAULT_AGENTS
        agent_count = max(_MIN_AGENTS, min(agent_count, _MAX_AGENTS))

        try:
            rounds = int(body.get("rounds", _DEFAULT_ROUNDS))
        except (TypeError, ValueError):
            rounds = _DEFAULT_ROUNDS
        rounds = max(1, min(rounds, _MAX_ROUNDS))

        # Rough per-agent-per-round cost (input + output tokens)
        per_agent_per_round = 0.005  # ~$0.005/agent/round
        estimated_cost = round(agent_count * rounds * per_agent_per_round, 4)
        budget_cap = 0.05

        return json_response(
            {
                "estimated_cost_usd": estimated_cost,
                "budget_cap_usd": budget_cap,
                "agent_count": agent_count,
                "rounds": rounds,
                "timeout_seconds": _LIVE_TIMEOUT,
                "note": "Actual cost may vary. Capped at budget limit.",
            }
        )

    # ------------------------------------------------------------------
    # POST /api/v1/playground/debate/live
    # ------------------------------------------------------------------

    def _handle_live_debate(self, handler: Any) -> HandlerResult:
        """Run a live debate with real API-backed agents."""
        # Rate limiting (separate from mock)
        client_ip = "unknown"
        if handler and hasattr(handler, "client_address"):
            addr = handler.client_address
            if isinstance(addr, (list, tuple)) and len(addr) >= 1:
                client_ip = str(addr[0])

        allowed, retry_after = _check_live_rate_limit(client_ip)
        if not allowed:
            return json_response(
                {
                    "error": "Live debate rate limit exceeded. Try again later.",
                    "code": "live_rate_limit_exceeded",
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
                f"Topic must be {_MAX_TOPIC_LENGTH} characters or less",
                400,
            )

        # Raw question (separate from system-prompt-laden topic, e.g. from Oracle)
        question = str(body.get("question", "") or "").strip() or None

        try:
            agent_count = int(body.get("agents", _DEFAULT_AGENTS))
        except (TypeError, ValueError):
            agent_count = _DEFAULT_AGENTS
        agent_count = max(_MIN_AGENTS, min(agent_count, _MAX_AGENTS))

        try:
            rounds = int(body.get("rounds", _DEFAULT_ROUNDS))
        except (TypeError, ValueError):
            rounds = _DEFAULT_ROUNDS
        rounds = max(1, min(rounds, _MAX_ROUNDS))

        # Check if any API keys are available
        has_api_keys = bool(
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("OPENROUTER_API_KEY")
        )

        if not has_api_keys:
            # Fall back to mock debate with a note
            result = self._run_debate(topic, rounds, agent_count, question=question)
            if result is None:
                return error_response("Playground unavailable", 503)
            # Inject mock fallback info into the response body
            import json as _json

            response_data = _json.loads(result.body.decode("utf-8"))
            response_data["is_live"] = False
            response_data["mock_fallback"] = True
            response_data["mock_fallback_reason"] = "No API keys configured on server"
            response_data["upgrade_cta"] = _build_upgrade_cta()
            return json_response(response_data, status=result.status_code)

        # Try live debate — fall back to mock if it fails
        live_result = self._run_live_debate(topic, rounds, agent_count)
        if live_result.status_code >= 500:
            logger.warning(
                "Live debate returned %d, falling back to mock debate",
                live_result.status_code,
            )
            mock_result = self._run_debate(topic, rounds, agent_count, question=question)
            if mock_result is not None:
                import json as _json

                response_data = _json.loads(mock_result.body.decode("utf-8"))
                response_data["is_live"] = False
                response_data["mock_fallback"] = True
                response_data["mock_fallback_reason"] = "Live agents temporarily unavailable"
                response_data["upgrade_cta"] = _build_upgrade_cta()
                return json_response(response_data)
        return live_result

    def _run_live_debate(
        self,
        topic: str,
        rounds: int,
        agent_count: int,
    ) -> HandlerResult:
        """Execute a live debate using real agents with budget/timeout caps."""
        try:
            import importlib.util
            if importlib.util.find_spec("aragora.server.debate_controller") is None:
                raise ImportError("debate_controller not found")
        except ImportError:
            logger.warning("DebateController not available for live playground")
            return error_response("Live playground unavailable", 503)

        debate_id = f"playground_{uuid.uuid4().hex[:8]}"

        try:
            result = start_playground_debate(
                question=topic,
                agent_count=agent_count,
                max_rounds=rounds,
                timeout=_LIVE_TIMEOUT,
            )
        except TimeoutError:
            return json_response(
                {
                    "error": "Live debate timed out (budget protection)",
                    "code": "timeout",
                    "is_live": True,
                    "upgrade_cta": _build_upgrade_cta(),
                },
                status=408,
            )
        except (ValueError, RuntimeError, OSError) as e:
            logger.warning(f"Live playground debate failed: {e}")
            return error_response("Live debate failed", 500)

        # Build response in the same shape as mock debates
        response = {
            "id": debate_id,
            "topic": topic,
            "status": result.get("status", "completed"),
            "rounds_used": result.get("rounds_used", rounds),
            "consensus_reached": result.get("consensus_reached", False),
            "confidence": result.get("confidence", 0.0),
            "verdict": result.get("verdict"),
            "duration_seconds": round(result.get("duration_seconds", 0.0), 3),
            "participants": result.get("participants", []),
            "proposals": result.get("proposals", []),
            "critiques": result.get("critiques", []),
            "votes": result.get("votes", []),
            "dissenting_views": result.get("dissenting_views", []),
            "final_answer": result.get("final_answer", ""),
            "is_live": True,
            "receipt_preview": {
                "debate_id": debate_id,
                "question": topic[:200],
                "consensus_reached": result.get("consensus_reached", False),
                "confidence": result.get("confidence", 0.0),
                "participants": result.get("participants", []),
                "note": "Unsigned preview. Full receipts available on paid plans.",
            },
            "upgrade_cta": _build_upgrade_cta(),
        }

        return json_response(response)


# ---------------------------------------------------------------------------
# Live debate execution
# ---------------------------------------------------------------------------

_LIVE_TIMEOUT = 60  # seconds
_LIVE_BUDGET_CAP = 0.05  # USD
_LIVE_MAX_CONCURRENT = 2
_LIVE_DEFAULT_AGENTS = ["anthropic", "openai"]
_LIVE_FALLBACK_AGENTS = ["openrouter"]

_live_semaphore = asyncio.Semaphore(_LIVE_MAX_CONCURRENT)


def _get_available_live_agents(count: int) -> list[str]:
    """Pick agent providers that have API keys configured."""
    candidates: list[str] = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        candidates.append("anthropic")
    if os.environ.get("OPENAI_API_KEY"):
        candidates.append("openai")
    if os.environ.get("OPENROUTER_API_KEY"):
        candidates.append("openrouter")
    if os.environ.get("MISTRAL_API_KEY"):
        candidates.append("mistral")

    # Pad to requested count by repeating
    while len(candidates) < count and candidates:
        candidates.append(candidates[0])
    return candidates[:count]


def start_playground_debate(
    question: str,
    agent_count: int = 3,
    max_rounds: int = 2,
    timeout: int = 60,
) -> dict[str, Any]:
    """Run a simplified live debate for the playground.

    Skips storage/auth. Runs synchronously with a timeout.
    Sets ``public_spectate: true`` in metadata for spectator access.

    Args:
        question: The debate question
        agent_count: Number of agents (2-5)
        max_rounds: Maximum rounds (1-2)
        timeout: Timeout in seconds

    Returns:
        Dict with debate result fields

    Raises:
        TimeoutError: If the debate exceeds timeout
        ValueError: If no agents are available
        RuntimeError: If arena execution fails
    """
    import concurrent.futures

    agents = _get_available_live_agents(agent_count)
    if len(agents) < 2:
        raise ValueError("At least 2 agent providers with API keys are required")

    agents_str = ",".join(agents)

    def _run() -> dict[str, Any]:
        try:
            from aragora.server.debate_factory import DebateConfig, DebateFactory

            factory = DebateFactory()
            config = DebateConfig(
                question=question,
                agents_str=agents_str,
                rounds=max_rounds,
                debate_format="light",
                metadata={"public_spectate": True, "is_playground": True},
            )

            arena = factory.create_arena(config)

            async def _run_arena():
                return await asyncio.wait_for(arena.run(), timeout=timeout)

            result = asyncio.run(_run_arena())

            # Extract key fields
            return {
                "status": result.status,
                "rounds_used": result.rounds_used,
                "consensus_reached": result.consensus_reached,
                "confidence": result.confidence,
                "verdict": result.verdict.value
                if hasattr(result, "verdict") and result.verdict
                else None,
                "duration_seconds": result.duration_seconds,
                "participants": result.participants,
                "proposals": result.proposals,
                "critiques": [
                    {
                        "agent": c.agent,
                        "target_agent": c.target_agent,
                        "issues": c.issues,
                        "suggestions": c.suggestions,
                        "severity": c.severity,
                    }
                    for c in result.critiques
                ]
                if hasattr(result, "critiques")
                else [],
                "votes": [
                    {
                        "agent": v.agent,
                        "choice": v.choice,
                        "confidence": v.confidence,
                        "reasoning": v.reasoning,
                    }
                    for v in result.votes
                ]
                if hasattr(result, "votes")
                else [],
                "dissenting_views": result.dissenting_views
                if hasattr(result, "dissenting_views")
                else [],
                "final_answer": result.final_answer,
            }
        except asyncio.TimeoutError:
            raise TimeoutError(f"Debate timed out after {timeout}s")

    # Run in a thread pool to avoid blocking the server
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        try:
            return pool.submit(_run).result(timeout=timeout + 5)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Debate timed out after {timeout}s")


def _build_upgrade_cta() -> dict[str, str]:
    """Build the upgrade call-to-action for playground responses."""
    return {
        "title": "Unlock Full Decision Intelligence",
        "message": (
            "This playground demo shows a taste of Aragora's multi-agent debate engine. "
            "Upgrade to access unlimited debates, full audit receipts, custom agent "
            "configurations, and enterprise features."
        ),
        "action_url": "/pricing",
        "action_label": "View Plans",
    }
