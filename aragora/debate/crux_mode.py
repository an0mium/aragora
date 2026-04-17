"""Crux-finder debate mode (Crux A1 / #6038).

This module elevates existing crux detection into a first-class debate
goal. A crux-finder run extracts the load-bearing disagreements in a
completed debate and packages them into a `CruxFinderResult`, which a
downstream builder in `aragora.debate.consensus` converts to a signed
`ConsensusProof` (sentinel final claim = "no verdict by design").

Only thin-wiring is implemented here (Approach A of the design doc —
`docs/plans/2026-04-16-crux-mode-design.md`). Debate prompts are not
shaped; cruxes are extracted from the `BeliefNetwork` the standard
belief-analysis phase already populates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from aragora.reasoning.crux_detector import (
    CruxAnalysisResult,
    CruxClaim,
    CruxDetector,
)

if TYPE_CHECKING:
    from aragora.debate.protocol import DebateProtocol
    from aragora.reasoning.belief import BeliefNetwork


# Sentinel value attached to `ConsensusProof.final_claim` for crux-finder
# runs. Downstream consumers assuming a verdict can detect this prefix and
# route to the CruxReceipt surface instead.
CRUX_MAP_SENTINEL = "__CRUX_MAP__: no verdict by design; see CruxReceipt.cruxes"


@dataclass
class CruxFinderResult:
    """Output of a crux-finder debate.

    Distinct from a `ConsensusProof` because the deliverable is *not* a
    verdict. Carries everything needed to build both a ConsensusProof (for
    protocol compatibility) and a CruxReceipt (for signed export, landing
    in a follow-up under DIC-16 / #6026).
    """

    debate_id: str
    question: str
    analysis: CruxAnalysisResult
    counterfactuals: list[dict[str, Any]] = field(default_factory=list)
    agents: list[str] = field(default_factory=list)
    rounds: int = 0
    raw_claims: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def top_cruxes(self) -> list[CruxClaim]:
        return self.analysis.cruxes

    def convergence_barrier(self) -> float:
        return self.analysis.convergence_barrier

    def to_dict(self) -> dict[str, Any]:
        return {
            "debate_id": self.debate_id,
            "question": self.question,
            "analysis": self.analysis.to_dict(),
            "counterfactuals": list(self.counterfactuals),
            "agents": list(self.agents),
            "rounds": self.rounds,
            "raw_claims": list(self.raw_claims),
            "metadata": dict(self.metadata),
        }


def build_crux_finder_result(
    *,
    belief_network: BeliefNetwork | None,
    protocol: DebateProtocol,
    debate_id: str,
    question: str,
    agents: list[str],
    rounds: int = 0,
    raw_claims: list[dict[str, Any]] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> CruxFinderResult:
    """Detect cruxes in a populated belief network and package the result.

    Raises:
        RuntimeError: if `belief_network` is None. The A1 design makes this
            an explicit failure rather than a silent fallback — a missing
            network means the belief-analysis phase was not run, and a
            crux-finder answer without it would not be trustworthy.
    """
    if belief_network is None:
        raise RuntimeError(
            "crux_finder mode requires a populated belief network. "
            "Check that the belief_analysis phase ran before consensus."
        )

    detector = CruxDetector(network=belief_network)
    analysis = detector.detect_cruxes(
        top_k=int(protocol.crux_finder_top_k),
        min_score=float(protocol.crux_finder_min_score),
    )

    counterfactuals: list[dict[str, Any]] = []
    if protocol.crux_finder_counterfactual_validation:
        for crux in analysis.cruxes:
            counterfactuals.append(
                {
                    "claim_id": crux.claim_id,
                    "condition": f"Resolve '{crux.statement}' to high confidence",
                    "outcome_change": (
                        f"Reduces total network uncertainty by {crux.resolution_impact:.3f}"
                    ),
                    "likelihood": round(float(crux.uncertainty_score), 4),
                    "affected_claims": list(crux.affected_claims),
                }
            )

    metadata: dict[str, Any] = {"mode": "crux_finder", "approach": "A"}
    if extra_metadata:
        metadata.update(extra_metadata)

    return CruxFinderResult(
        debate_id=debate_id,
        question=question,
        analysis=analysis,
        counterfactuals=counterfactuals,
        agents=list(agents),
        rounds=rounds,
        raw_claims=list(raw_claims or []),
        metadata=metadata,
    )


__all__ = [
    "CRUX_MAP_SENTINEL",
    "CruxFinderResult",
    "build_crux_finder_result",
]
