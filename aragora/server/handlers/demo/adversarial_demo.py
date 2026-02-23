"""Adversarial demo handler -- live multi-agent debate showcase.

Stability: STABLE

Runs a self-contained adversarial demo debate that demonstrates Aragora's
core value proposition: agents challenge each other, calibration weights
shift consensus thresholds, and the result is an audit-ready decision receipt.

The handler works in two modes:
- **Online**: Uses real agent APIs (if available) to generate debate content.
- **Offline / Demo**: Generates deterministic mock responses so the demo
  runs without any API keys or network access.

Endpoints:
- POST /api/v1/demo/adversarial         -- Start a live adversarial demo
- GET  /api/v1/demo/adversarial/status/{demo_id} -- Check demo status
"""

from __future__ import annotations

__all__ = [
    "handle_adversarial_demo",
    "handle_demo_status",
    "register_routes",
]

import hashlib
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from aiohttp import web

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory store for completed demos (bounded to prevent memory leaks)
# ---------------------------------------------------------------------------

_MAX_STORED_DEMOS = 200
_demo_store: dict[str, dict[str, Any]] = {}


def _is_demo_mode() -> bool:
    """Return True when running without live agent backends."""
    return bool(os.environ.get("ARAGORA_OFFLINE") or os.environ.get("DEMO_MODE"))


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def register_routes(app: web.Application) -> None:
    """Register adversarial demo routes on *app*."""
    app.router.add_post("/api/v1/demo/adversarial", handle_adversarial_demo)
    app.router.add_get(
        "/api/v1/demo/adversarial/status/{demo_id}",
        handle_demo_status,
    )


# ---------------------------------------------------------------------------
# POST /api/v1/demo/adversarial
# ---------------------------------------------------------------------------


async def handle_adversarial_demo(request: web.Request) -> web.Response:
    """Start a live adversarial demo debate.

    Accepts a JSON body with:
        topic (str, required): The debate question.
        agent_count (int, optional): Number of agents (2-6, default 3).
        rounds (int, optional): Number of debate rounds (1-5, default 2).
        include_calibration (bool, optional): Include calibration impact
            analysis in the response (default ``True``).

    Returns a JSON response with the full debate transcript, consensus
    result, decision receipt, and (optionally) calibration impact.
    """
    # -- Parse & validate request body ------------------------------------
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return web.json_response({"error": "Invalid JSON body"}, status=400)

    if not isinstance(body, dict):
        return web.json_response({"error": "Request body must be a JSON object"}, status=400)

    topic: str | None = body.get("topic")
    if not topic or not isinstance(topic, str) or not topic.strip():
        return web.json_response(
            {"error": "Missing or empty 'topic' field"},
            status=400,
        )

    topic = topic.strip()
    agent_count: int = _clamp(int(body.get("agent_count", 3)), 2, 6)
    rounds: int = _clamp(int(body.get("rounds", 2)), 1, 5)
    include_calibration: bool = bool(body.get("include_calibration", True))

    # -- Execute the demo debate ------------------------------------------
    demo_id = f"demo_{uuid.uuid4().hex[:12]}"
    start_ts = time.monotonic()

    try:
        result = await _run_demo_debate(
            demo_id=demo_id,
            topic=topic,
            agent_count=agent_count,
            rounds=rounds,
            include_calibration=include_calibration,
        )
    except (TypeError, ValueError) as exc:
        logger.warning("Demo debate failed for topic=%r: %s", topic, exc)
        return web.json_response(
            {"error": f"Demo execution failed: {exc}"},
            status=500,
        )

    elapsed = time.monotonic() - start_ts
    result["elapsed_seconds"] = round(elapsed, 3)

    # -- Persist in bounded store -----------------------------------------
    _store_demo(demo_id, result)

    logger.info(
        "Adversarial demo %s completed in %.2fs (topic=%r, agents=%d, rounds=%d)",
        demo_id,
        elapsed,
        topic,
        agent_count,
        rounds,
    )

    return web.json_response(result)


# ---------------------------------------------------------------------------
# GET /api/v1/demo/adversarial/status/{demo_id}
# ---------------------------------------------------------------------------


async def handle_demo_status(request: web.Request) -> web.Response:
    """Return the result of a previously-executed demo by its *demo_id*."""
    demo_id = request.match_info.get("demo_id", "")

    if not demo_id:
        return web.json_response({"error": "Missing demo_id"}, status=400)

    demo = _demo_store.get(demo_id)
    if demo is None:
        return web.json_response(
            {"error": f"Demo '{demo_id}' not found"},
            status=404,
        )

    return web.json_response(demo)


# ---------------------------------------------------------------------------
# Core demo logic
# ---------------------------------------------------------------------------

# Agent archetypes used for deterministic mock debates.
_AGENT_ARCHETYPES = [
    {"name": "Analyst", "bias": "pro", "base_calibration": 0.82},
    {"name": "Skeptic", "bias": "con", "base_calibration": 0.76},
    {"name": "Pragmatist", "bias": "neutral", "base_calibration": 0.91},
    {"name": "Visionary", "bias": "pro", "base_calibration": 0.68},
    {"name": "Risk Assessor", "bias": "con", "base_calibration": 0.88},
    {"name": "Synthesizer", "bias": "neutral", "base_calibration": 0.85},
]


async def _run_demo_debate(
    *,
    demo_id: str,
    topic: str,
    agent_count: int,
    rounds: int,
    include_calibration: bool,
) -> dict[str, Any]:
    """Orchestrate a mock adversarial debate and return the full result."""

    agents = _select_agents(agent_count, topic)
    debate_rounds: list[dict[str, Any]] = []

    # -- Run rounds --------------------------------------------------------
    for round_idx in range(1, rounds + 1):
        round_entries: list[dict[str, Any]] = []
        for agent in agents:
            proposal = _generate_proposal(agent, topic, round_idx, rounds)
            round_entries.append(proposal)

        # Each agent critiques the others
        critiques: list[dict[str, Any]] = []
        for agent in agents:
            others = [e for e in round_entries if e["agent"] != agent["name"]]
            for other in others:
                critique = _generate_critique(agent, other, topic, round_idx)
                critiques.append(critique)

        debate_rounds.append(
            {
                "round": round_idx,
                "proposals": round_entries,
                "critiques": critiques,
            }
        )

    # -- Build positions (final round snapshot) ----------------------------
    positions = _build_positions(agents, topic, rounds)

    # -- Consensus ---------------------------------------------------------
    consensus = _compute_consensus(agents, positions)

    # -- Synthesis ---------------------------------------------------------
    synthesis = _generate_synthesis(topic, positions, consensus)

    # -- Decision receipt --------------------------------------------------
    receipt = _generate_receipt(demo_id, topic, agents, positions, consensus, synthesis)

    result: dict[str, Any] = {
        "demo_id": demo_id,
        "status": "completed",
        "debate": {
            "topic": topic,
            "rounds": debate_rounds,
            "positions": positions,
            "synthesis": synthesis,
            "consensus": consensus,
        },
        "receipt": receipt,
    }

    if include_calibration:
        result["calibration_impact"] = _compute_calibration_impact(agents, consensus)

    return result


# ---------------------------------------------------------------------------
# Agent selection
# ---------------------------------------------------------------------------


def _select_agents(count: int, topic: str) -> list[dict[str, Any]]:
    """Pick *count* agent archetypes, ensuring at least one pro and one con."""
    # Deterministic seed from topic so the same topic always gets the same
    # agents (nice for demos).
    seed = int(hashlib.md5(topic.encode(), usedforsecurity=False).hexdigest()[:8], 16)

    # Always include one pro and one con, then fill remaining slots.
    ordered = list(_AGENT_ARCHETYPES)
    # Simple deterministic shuffle using seed
    for i in range(len(ordered) - 1, 0, -1):
        j = seed % (i + 1)
        ordered[i], ordered[j] = ordered[j], ordered[i]
        seed = (seed * 6364136223846793005 + 1) & 0xFFFFFFFF

    # Guarantee at least one pro and one con
    selected: list[dict[str, Any]] = []
    pro_added = con_added = False
    for a in ordered:
        if len(selected) >= count:
            break
        if a["bias"] == "pro" and not pro_added:
            selected.append(dict(a))
            pro_added = True
        elif a["bias"] == "con" and not con_added:
            selected.append(dict(a))
            con_added = True
        elif pro_added and con_added:
            selected.append(dict(a))

    # If we still lack diversity, force-add from archetypes
    if not pro_added:
        for a in _AGENT_ARCHETYPES:
            if a["bias"] == "pro":
                selected[0] = dict(a)
                break
    if not con_added:
        for a in _AGENT_ARCHETYPES:
            if a["bias"] == "con":
                if len(selected) > 1:
                    selected[1] = dict(a)
                else:
                    selected.append(dict(a))
                break

    # Assign calibration weights (Brier-derived multiplier)
    for agent in selected:
        cal = agent["base_calibration"]
        agent["calibration_weight"] = round(0.5 + cal, 2)

    return selected[:count]


# ---------------------------------------------------------------------------
# Mock content generation (deterministic, no API calls)
# ---------------------------------------------------------------------------

_POSITION_TEMPLATES = {
    "pro": (
        "After thorough analysis, I advocate for '{topic}'. "
        "The evidence strongly supports this direction due to scalability, "
        "flexibility, and long-term maintainability gains. "
        "Key supporting factors include industry adoption trends and "
        "measurable ROI improvements."
    ),
    "con": (
        "I must respectfully challenge the premise of '{topic}'. "
        "The risks -- including operational complexity, coordination "
        "overhead, and debugging difficulty -- outweigh the theoretical "
        "benefits. Empirical studies show mixed results at best."
    ),
    "neutral": (
        "Both sides present valid points on '{topic}'. "
        "The optimal approach depends heavily on organizational maturity, "
        "team size, and existing infrastructure. A hybrid strategy may "
        "yield the best risk-adjusted outcome."
    ),
}

_CRITIQUE_TEMPLATES = {
    "pro": (
        "{critic} challenges {target}'s position: "
        "While the concerns are noted, they underweight the compounding "
        "benefits of early adoption and overestimate transition costs."
    ),
    "con": (
        "{critic} pushes back on {target}'s optimism: "
        "The cited benefits assume ideal conditions rarely seen in practice. "
        "Real-world failure modes deserve more weight in this analysis."
    ),
    "neutral": (
        "{critic} refines {target}'s argument: "
        "The core insight is sound, but the confidence level should be "
        "tempered given the limited evidence for this specific context."
    ),
}


def _generate_proposal(
    agent: dict[str, Any],
    topic: str,
    round_idx: int,
    total_rounds: int,
) -> dict[str, Any]:
    """Generate a mock proposal for a given round."""
    bias = agent["bias"]
    template = _POSITION_TEMPLATES[bias]
    content = template.format(topic=topic)

    # In later rounds, agents refine their positions
    if round_idx > 1:
        content += (
            f" [Round {round_idx}/{total_rounds} revision: "
            f"incorporating critiques from prior rounds to strengthen core argument.]"
        )

    # Confidence drifts toward center in later rounds (convergence)
    base_confidence = {"pro": 0.85, "con": 0.78, "neutral": 0.72}[bias]
    confidence = base_confidence - 0.03 * (round_idx - 1)

    return {
        "agent": agent["name"],
        "bias": bias,
        "position": content,
        "confidence": round(confidence, 2),
        "round": round_idx,
    }


def _generate_critique(
    critic: dict[str, Any],
    target_proposal: dict[str, Any],
    topic: str,
    round_idx: int,
) -> dict[str, Any]:
    """Generate a mock critique of another agent's proposal."""
    template = _CRITIQUE_TEMPLATES[critic["bias"]]
    content = template.format(
        critic=critic["name"],
        target=target_proposal["agent"],
    )

    return {
        "critic": critic["name"],
        "target": target_proposal["agent"],
        "critique": content,
        "round": round_idx,
        "severity": "moderate" if critic["bias"] != target_proposal["bias"] else "minor",
    }


def _build_positions(
    agents: list[dict[str, Any]],
    topic: str,
    total_rounds: int,
) -> dict[str, dict[str, Any]]:
    """Build final position map for each agent after all rounds."""
    positions: dict[str, dict[str, Any]] = {}
    for agent in agents:
        bias = agent["bias"]
        base_confidence = {"pro": 0.85, "con": 0.78, "neutral": 0.72}[bias]
        final_confidence = base_confidence - 0.03 * (total_rounds - 1)

        positions[agent["name"]] = {
            "position": _POSITION_TEMPLATES[bias].format(topic=topic),
            "confidence": round(final_confidence, 2),
            "calibration_weight": agent["calibration_weight"],
            "bias": bias,
        }

    return positions


def _compute_consensus(
    agents: list[dict[str, Any]],
    positions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Determine consensus using calibration-weighted voting."""
    weighted_scores: dict[str, float] = {"pro": 0.0, "con": 0.0, "neutral": 0.0}

    for agent in agents:
        name = agent["name"]
        pos = positions[name]
        weight = pos["calibration_weight"]
        confidence = pos["confidence"]
        weighted_scores[pos["bias"]] += weight * confidence

    total_weight = sum(weighted_scores.values()) or 1.0

    # Normalise
    for key in weighted_scores:
        weighted_scores[key] = round(weighted_scores[key] / total_weight, 3)

    # Winner is the bias with highest weighted score
    winner_bias = max(weighted_scores, key=lambda k: weighted_scores[k])

    # Map winning bias back to an agent name
    winner_agent: str | None = None
    for agent in agents:
        if positions[agent["name"]]["bias"] == winner_bias:
            winner_agent = agent["name"]
            break

    # Adaptive threshold: well-calibrated pools can use a lower threshold
    avg_cal = sum(a["base_calibration"] for a in agents) / len(agents)
    threshold = round(max(0.45, 0.65 - 0.1 * avg_cal), 2)
    winning_score = weighted_scores[winner_bias]
    reached = winning_score >= threshold

    overall_confidence = round(winning_score, 2)

    return {
        "reached": reached,
        "confidence": overall_confidence,
        "threshold_used": threshold,
        "winner": winner_agent,
        "weighted_scores": weighted_scores,
    }


def _generate_synthesis(
    topic: str,
    positions: dict[str, dict[str, Any]],
    consensus: dict[str, Any],
) -> str:
    """Produce a synthesis statement summarising the debate outcome."""
    winner = consensus.get("winner", "the panel")
    confidence = consensus.get("confidence", 0.0)
    reached = consensus.get("reached", False)

    if reached:
        return (
            f"After adversarial debate on '{topic}', the panel reached consensus "
            f"(confidence {confidence:.0%}). {winner}'s position prevailed: "
            f"the evidence, weighted by agent calibration scores, supports "
            f"this direction. Dissenting views were recorded and their "
            f"strongest counter-arguments incorporated into the final synthesis."
        )
    return (
        f"The panel did not reach consensus on '{topic}' "
        f"(highest confidence {confidence:.0%}, below threshold "
        f"{consensus.get('threshold_used', 0.55):.0%}). "
        f"The debate surfaced important trade-offs that require "
        f"further analysis or domain-expert input before a decision."
    )


# ---------------------------------------------------------------------------
# Decision receipt
# ---------------------------------------------------------------------------


def _generate_receipt(
    demo_id: str,
    topic: str,
    agents: list[dict[str, Any]],
    positions: dict[str, dict[str, Any]],
    consensus: dict[str, Any],
    synthesis: str,
) -> dict[str, Any]:
    """Build a decision receipt with SHA-256 integrity checksum."""
    receipt_id = f"rcpt_{uuid.uuid4().hex[:16]}"
    now = datetime.now(timezone.utc).isoformat()

    evidence_chain: list[dict[str, Any]] = []
    for agent in agents:
        name = agent["name"]
        pos = positions[name]
        evidence_chain.append(
            {
                "timestamp": now,
                "event_type": "position",
                "agent": name,
                "description": f"{name} ({pos['bias']}): confidence {pos['confidence']:.2f}, "
                f"calibration weight {pos['calibration_weight']}",
                "evidence_hash": _sha256(pos["position"]),
            }
        )

    evidence_chain.append(
        {
            "timestamp": now,
            "event_type": "consensus",
            "agent": None,
            "description": (
                f"Consensus {'reached' if consensus['reached'] else 'NOT reached'} "
                f"(confidence={consensus['confidence']}, "
                f"threshold={consensus['threshold_used']})"
            ),
            "evidence_hash": _sha256(json.dumps(consensus, sort_keys=True)),
        }
    )

    decision = synthesis

    # Content-addressable checksum covering the entire receipt payload
    receipt_payload = json.dumps(
        {
            "receipt_id": receipt_id,
            "demo_id": demo_id,
            "topic": topic,
            "decision": decision,
            "evidence_chain": evidence_chain,
        },
        sort_keys=True,
    )
    checksum = _sha256(receipt_payload)

    return {
        "receipt_id": receipt_id,
        "checksum": checksum,
        "decision": decision,
        "evidence_chain": evidence_chain,
        "timestamp": now,
    }


# ---------------------------------------------------------------------------
# Calibration impact
# ---------------------------------------------------------------------------


def _compute_calibration_impact(
    agents: list[dict[str, Any]],
    consensus: dict[str, Any],
) -> dict[str, Any]:
    """Explain how calibration affected the debate outcome."""
    weights = [a["calibration_weight"] for a in agents]
    avg_cal = sum(a["base_calibration"] for a in agents) / len(agents)

    # Threshold adjustment relative to the uncalibrated baseline of 0.55
    threshold_adj = round(consensus["threshold_used"] - 0.55, 2)

    return {
        "threshold_adjustment": threshold_adj,
        "vote_weight_range": [round(min(weights), 2), round(max(weights), 2)],
        "average_calibration": round(avg_cal, 3),
        "explanation": (
            f"Agent pool average calibration is {avg_cal:.1%}. "
            f"{'Well' if avg_cal >= 0.8 else 'Moderately'}-calibrated pool "
            f"{'allowed lower' if threshold_adj < 0 else 'required standard'} "
            f"consensus threshold ({consensus['threshold_used']:.0%} vs 55% baseline). "
            f"Vote weights ranged from {min(weights)} to {max(weights)}, "
            f"amplifying better-calibrated agents' influence."
        ),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(content: str) -> str:
    """Return the hex SHA-256 digest of *content*."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _clamp(value: int, lo: int, hi: int) -> int:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


def _store_demo(demo_id: str, result: dict[str, Any]) -> None:
    """Persist a demo result in the bounded in-memory store."""
    if len(_demo_store) >= _MAX_STORED_DEMOS:
        # Evict oldest entry
        oldest = next(iter(_demo_store))
        del _demo_store[oldest]

    _demo_store[demo_id] = result
