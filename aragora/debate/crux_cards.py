"""Crux cards for standard debates (#8227 / #9046 phase 1).

Attaches the load-bearing disagreements of a *standard* debate to its
result — without changing the debate goal. (Making crux-finding the goal
itself is ``consensus="crux_finder"``; see ``crux_mode.py``.)

Gated by ``DebateProtocol.enable_crux_cards`` (default OFF). When on, the
consensus phase stores a crux-cards block on
``DebateResult.metadata["crux_cards"]``; ``DecisionReceipt.from_debate_result``
carries it additively into ``DecisionReceipt.cruxes``, from which the ODR
export's ``cruxes`` block is populated (previously always the absent marker
for standard debates).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

CRUX_CARDS_METADATA_KEY = "crux_cards"

# An agent that states a claim in a debate is asserting it. ``add_claim`` defaults
# to 0.5, i.e. maximum entropy — "no opinion" — which left every author recorded as
# neutral about their own claim, so the disagreement variance was zero and
# ``contesting_agents`` came back empty even for a genuinely contested debate.
# Deliberately short of certainty: an asserted debate claim is not a proof.
_ASSERTED_CONFIDENCE = 0.8


def build_crux_cards(
    *,
    belief_network: Any = None,
    messages: list[Any] | None = None,
    critiques: list[Any] | None = None,
    top_k: int = 5,
    min_score: float = 0.3,
) -> dict[str, Any] | None:
    """Detect cruxes and package them as a crux-cards block.

    Prefers a populated belief network (from the belief-analysis phase);
    falls back to building one from proposer/critic messages. Returns ``None``
    when no crux clears ``min_score`` or there is no material to analyze —
    callers must then leave the result untouched so flag-off behavior stays
    byte-identical.

    ``critiques`` are what make detection possible at all.
    :class:`~aragora.reasoning.crux_detector.CruxDetector` scores a claim by how
    much *authors disagree about it*, which it reads from the network's factor
    edges. A network of claims with no edges therefore always reports zero
    disagreements and zero cruxes, however contested the debate actually was
    (#9581). Each :class:`~aragora.core_types.Critique` already records which
    agent contested whose proposal and how hard (``severity``), so those become
    the ``CONTRADICTS`` edges.
    """
    from aragora.reasoning.crux_detector import CruxDetector

    network = belief_network
    if network is None:
        network = _network_from_messages(messages or [], critiques or [])
    elif critiques:
        # A KM-seeded network still carries no debate-relative edges.
        _link_critiques(network, messages or [], critiques)
    if network is None or not getattr(network, "nodes", None):
        return None

    detector = CruxDetector(network=network)
    analysis = detector.detect_cruxes(top_k=top_k, min_score=min_score)
    if not analysis.cruxes:
        return None

    return {
        # CruxClaim.to_dict() carries per-crux dissent attribution:
        # author, contesting_agents, affected_claims, component scores.
        "items": [crux.to_dict() for crux in analysis.cruxes],
        "total_claims": analysis.total_claims,
        "total_disagreements": analysis.total_disagreements,
        "convergence_barrier": round(float(analysis.convergence_barrier), 4),
        "detector": "belief_network",
    }


def _network_from_messages(messages: list[Any], critiques: list[Any] | None = None) -> Any | None:
    try:
        from aragora.reasoning.belief import BeliefNetwork
    except ImportError:
        return None

    network = BeliefNetwork(max_iterations=3)
    added = 0
    for i, msg in enumerate(messages):
        if getattr(msg, "role", "") in ("proposer", "critic"):
            network.add_claim(
                claim_id=f"msg_{i}_{getattr(msg, 'agent', 'unknown')}",
                statement=str(getattr(msg, "content", ""))[:500],
                author=str(getattr(msg, "agent", "unknown")),
                initial_confidence=_ASSERTED_CONFIDENCE,
            )
            added += 1
    if not added:
        return None
    _link_critiques(network, messages, critiques or [])
    return network


def _first_claim_id_for_agent(messages: list[Any], agent: str, role: str) -> str | None:
    """The claim id of ``agent``'s first ``role`` message, matching the ids above."""
    for i, msg in enumerate(messages):
        if getattr(msg, "role", "") == role and str(getattr(msg, "agent", "")) == agent:
            return f"msg_{i}_{agent}"
    return None


def _link_critiques(network: Any, messages: list[Any], critiques: list[Any]) -> int:
    """Add a ``CONTRADICTS`` edge per critique; return how many were added.

    A critique is a *recorded, agent-attributed disagreement*: agent X contested
    agent Y's proposal with a model-assessed ``severity``. That is exactly the
    relation :class:`CruxDetector` needs, so edges are derived from the debate's
    own structured records rather than inferred from prose.

    Edge strength is ``severity / 10`` (the documented 0-10 scale) clamped to a
    small floor, so a critical objection weighs more than a cosmetic one and a
    ``severity=0`` nit still registers as contested rather than vanishing.

    Best-effort: an unmappable critique is skipped rather than raising, since
    the caller treats crux-building as optional enrichment.
    """
    try:
        from aragora.reasoning.claims import RelationType
    except ImportError:
        return 0

    linked = 0
    for n, critique in enumerate(critiques):
        critic = str(getattr(critique, "agent", "") or "")
        target = str(getattr(critique, "target_agent", "") or getattr(critique, "target", "") or "")
        if not critic or not target or critic == target:
            continue

        target_claim = _first_claim_id_for_agent(messages, target, "proposer")
        if target_claim is None:
            continue

        source_claim = _first_claim_id_for_agent(messages, critic, "critic")
        if source_claim is None:
            # The critique exists but its text never became a message (e.g. a
            # critique-only round); represent it as its own claim so the
            # disagreement is still attributable to its author.
            source_claim = f"critique_{n}_{critic}"
            network.add_claim(
                claim_id=source_claim,
                statement=str(getattr(critique, "reasoning", "") or "")[:500],
                author=critic,
                initial_confidence=_ASSERTED_CONFIDENCE,
            )

        try:
            severity = float(getattr(critique, "severity", 0.0) or 0.0)
        except (TypeError, ValueError):
            severity = 0.0
        strength = min(1.0, max(0.1, severity / 10.0))

        if network.add_factor(source_claim, target_claim, RelationType.CONTRADICTS, strength):
            linked += 1
    return linked


__all__ = [
    "CRUX_CARDS_METADATA_KEY",
    "build_crux_cards",
]
