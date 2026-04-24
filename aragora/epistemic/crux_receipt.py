"""Signed CruxReceipt for crux-finder debate runs (DIC-16 / #6026).

Converts a :class:`~aragora.debate.crux_mode.CruxFinderResult` into a
SHA-256-attested receipt where load-bearing cruxes are the headline.
Flag: ``ARAGORA_CRUX_RECEIPT_ENABLED`` (default False).
Construction is always safe; the flag gates downstream actions only.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.debate.crux_mode import CruxFinderResult


def crux_receipt_enabled() -> bool:
    """Return True if callers may act on CruxReceipt outputs."""
    raw = str(os.environ.get("ARAGORA_CRUX_RECEIPT_ENABLED") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def enable_crux_receipt() -> None:
    """Enable DIC-16 receipt actions for the current process."""
    os.environ["ARAGORA_CRUX_RECEIPT_ENABLED"] = "1"


@dataclass(frozen=True)
class CruxEntry:
    """One load-bearing disagreement from a crux-finder run.

    Field names align with DIC-13 claim manifests and
    :class:`~aragora.epistemic.proof_unit.ProofCarryingCodeUnit`.
    """

    crux_id: str
    statement: str
    load_bearing_score: float
    uncertainty_score: float
    contesting_agents: list[str]
    affected_claims: list[str]
    resolution_impact: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "crux_id": self.crux_id,
            "statement": self.statement,
            "load_bearing_score": round(self.load_bearing_score, 4),
            "uncertainty_score": round(self.uncertainty_score, 4),
            "contesting_agents": list(self.contesting_agents),
            "affected_claims": list(self.affected_claims),
            "resolution_impact": round(self.resolution_impact, 4),
        }


@dataclass(frozen=True)
class CruxReceipt:
    """Signed receipt for a crux-finder debate run.

    ``checksum`` is a 64-char SHA-256 hex digest over ``receipt_id``,
    ``debate_id``, ``question``, ``cruxes``, ``convergence_barrier`` —
    same convention as the quarantine_policy and repair DIC modules.
    """

    receipt_id: str
    debate_id: str
    question: str
    cruxes: list[CruxEntry]
    convergence_barrier: float
    counterfactuals: list[dict[str, Any]]
    agents: list[str]
    rounds: int
    metadata: dict[str, Any]
    checksum: str  # 64-char lowercase SHA-256 hex

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_id,
            "debate_id": self.debate_id,
            "question": self.question,
            "cruxes": [c.to_dict() for c in self.cruxes],
            "convergence_barrier": round(self.convergence_barrier, 4),
            "counterfactuals": list(self.counterfactuals),
            "agents": list(self.agents),
            "rounds": self.rounds,
            "metadata": dict(self.metadata),
            "checksum": self.checksum,
        }


def build_crux_receipt(result: "CruxFinderResult") -> CruxReceipt:
    """Build a signed CruxReceipt from a CruxFinderResult.

    Construction is always safe; callers acting on the receipt must check
    :func:`crux_receipt_enabled` themselves.
    """
    receipt_id = "crux_rcpt_" + uuid.uuid4().hex[:16]
    cruxes = [
        CruxEntry(
            crux_id=c.claim_id,
            statement=c.statement,
            load_bearing_score=c.crux_score,
            uncertainty_score=c.uncertainty_score,
            contesting_agents=list(c.contesting_agents),
            affected_claims=list(c.affected_claims),
            resolution_impact=c.resolution_impact,
        )
        for c in result.top_cruxes()
    ]
    content: dict[str, Any] = {
        "receipt_id": receipt_id,
        "debate_id": result.debate_id,
        "question": result.question,
        "cruxes": [c.to_dict() for c in cruxes],
        "convergence_barrier": round(result.convergence_barrier(), 4),
    }
    checksum = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CruxReceipt(
        receipt_id=receipt_id,
        debate_id=result.debate_id,
        question=result.question,
        cruxes=cruxes,
        convergence_barrier=result.convergence_barrier(),
        counterfactuals=list(result.counterfactuals),
        agents=list(result.agents),
        rounds=result.rounds,
        metadata=dict(result.metadata),
        checksum=checksum,
    )


__all__ = [
    "CruxEntry",
    "CruxReceipt",
    "build_crux_receipt",
    "crux_receipt_enabled",
    "enable_crux_receipt",
]
