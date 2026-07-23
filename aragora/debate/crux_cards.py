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


def build_crux_cards(
    *,
    belief_network: Any = None,
    messages: list[Any] | None = None,
    top_k: int = 5,
    min_score: float = 0.3,
) -> dict[str, Any] | None:
    """Detect cruxes and package them as a crux-cards block.

    Prefers a populated belief network (from the belief-analysis phase);
    falls back to building one from proposer/critic messages, mirroring
    ``WinnerSelector.analyze_belief_network``. Returns ``None`` when no crux
    clears ``min_score`` or there is no material to analyze — callers must
    then leave the result untouched so flag-off behavior stays byte-identical.
    """
    from aragora.reasoning.crux_detector import CruxDetector

    network = belief_network
    if network is None:
        network = _network_from_messages(messages or [])
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


def _network_from_messages(messages: list[Any]) -> Any | None:
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
            )
            added += 1
    return network if added else None


__all__ = [
    "CRUX_CARDS_METADATA_KEY",
    "build_crux_cards",
]
