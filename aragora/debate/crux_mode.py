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


def extract_crux_payload(source: Any) -> dict[str, Any] | None:
    """Extract the recorded crux map from a debate result or stored debate dict.

    A ``crux_finder`` run records its full crux map on
    ``ConsensusProof.metadata`` (see :func:`aragora.debate.consensus.build_proof_from_crux_finder`).
    This helper reads that record back from either:

    - a live ``DebateResult`` (``consensus_proof`` is an object), or
    - a stored debate artifact dict (``consensus_proof`` is a serialized dict,
      possibly nested under a ``result`` key).

    Returns:
        A dict with ``cruxes``, ``crux_count``, ``convergence_barrier``,
        ``counterfactuals`` and ``recommended_focus`` when crux data was
        recorded, or ``None`` when the debate has no crux record (crux mode
        not enabled, or it fell back to another consensus). Never fabricates.
    """
    if source is None:
        return None

    candidates: list[Any] = [source]
    if isinstance(source, dict):
        nested = source.get("result")
        if isinstance(nested, dict):
            candidates.append(nested)

    for candidate in candidates:
        if isinstance(candidate, dict):
            proof = candidate.get("consensus_proof")
        else:
            proof = getattr(candidate, "consensus_proof", None)
        if proof is None:
            continue

        if isinstance(proof, dict):
            metadata = proof.get("metadata")
        else:
            metadata = getattr(proof, "metadata", None)
        if not isinstance(metadata, dict):
            continue

        # The crux map is only trustworthy when the crux_finder mode actually
        # recorded it; "cruxes" present in metadata is that record.
        if "cruxes" not in metadata:
            continue

        cruxes_raw = metadata.get("cruxes") or []
        cruxes = [dict(c) for c in cruxes_raw if isinstance(c, dict)]
        barrier = metadata.get("convergence_barrier")
        return {
            "consensus_mode": str(metadata.get("consensus_mode") or "crux_finder"),
            "cruxes": cruxes,
            "crux_count": int(metadata.get("crux_count", len(cruxes)) or 0),
            "convergence_barrier": float(barrier) if barrier is not None else None,
            "counterfactuals": [
                dict(c) for c in (metadata.get("counterfactuals") or []) if isinstance(c, dict)
            ],
            "recommended_focus": [str(f) for f in (metadata.get("recommended_focus") or [])],
        }

    return None


def extract_crux_skip_reason(source: Any) -> str | None:
    """Return the recorded reason a crux_finder run fell back, if any.

    The consensus phase records ``crux_finder_skipped_reason`` on the debate
    result metadata when crux mode was requested but could not run (e.g. no
    belief network). Returns ``None`` when no skip was recorded.
    """
    if source is None:
        return None
    if isinstance(source, dict):
        metadata = source.get("metadata")
        if not isinstance(metadata, dict):
            nested = source.get("result")
            metadata = nested.get("metadata") if isinstance(nested, dict) else None
    else:
        metadata = getattr(source, "metadata", None)
    if not isinstance(metadata, dict):
        return None
    reason = metadata.get("crux_finder_skipped_reason")
    return str(reason) if reason else None


__all__ = [
    "CRUX_MAP_SENTINEL",
    "CruxFinderResult",
    "build_crux_finder_result",
    "extract_crux_payload",
    "extract_crux_skip_reason",
]
