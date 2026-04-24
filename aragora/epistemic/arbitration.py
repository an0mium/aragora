"""Operator Crux Arbitration — human resolution of persistent cruxes (DIC-27 / #6221).

A *persistent crux* is a load-bearing disagreement (``load_bearing_score >=
PERSISTENT_CRUX_MIN_SCORE``) that surfaces in ``PERSISTENT_CRUX_MIN_CONSECUTIVE``
or more consecutive debates sharing the same ``question_family_id``.

Operator arbitrations capture the chosen side, rationale, expiry, and optional
evidence citations so that downstream debates receive the decision as pinned
context rather than tribal memory.  Reversal is a first-class operation: it
produces a :class:`CruxArbitrationReversal` receipt and does **not** delete the
original arbitration.

Flag gate: ``ARAGORA_CRUX_ARBITRATION_ENABLED`` (default off).
Construction of :class:`CruxArbitration` records is always safe; priors-update,
belief-network injection, and any GitHub-visible or queue-mutating action must
check :func:`crux_arbitration_enabled` first.

Out of scope for this slice (future work):
- Belief-network priors-update path (depends on DIC-15/16 being production-green).
- CLI ``aragora crux arbitrate`` interactive prompt.
- DIC-18 truth-map integration for arbitration age display.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

# ──────────────────────────── constants ────────────────────────────────────────

PERSISTENT_CRUX_MIN_SCORE: float = 0.6
PERSISTENT_CRUX_MIN_CONSECUTIVE: int = 3
DEFAULT_EXPIRY_DAYS: int = 90

# ──────────────────────────── flag gate ────────────────────────────────────────


def crux_arbitration_enabled() -> bool:
    """Return True when callers may act on arbitration outputs.

    Reads ``ARAGORA_CRUX_ARBITRATION_ENABLED`` from the environment.
    Default is False; construction of :class:`CruxArbitration` objects is
    always safe, but priors updates and any queue-visible actions must be gated.
    """
    raw = str(os.environ.get("ARAGORA_CRUX_ARBITRATION_ENABLED") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


# ──────────────────────────── core types ───────────────────────────────────────

#: Operator's chosen side when resolving a persistent crux.
ArbitrationSide = Literal["accept", "reject", "defer", "split"]


@dataclass(frozen=True)
class PersistentCrux:
    """A crux that has remained load-bearing across N consecutive debates.

    Only identifies the crux for arbitration; does not carry a resolution.
    Check :attr:`qualifies` before creating a :class:`CruxArbitration`.
    """

    crux_id: str
    statement: str
    question_family_id: str
    consecutive_debate_count: int
    load_bearing_score: float
    cruxset_receipt_ids: tuple[str, ...]

    @property
    def qualifies(self) -> bool:
        """True when this crux meets both persistence and load-bearing thresholds."""
        return (
            self.consecutive_debate_count >= PERSISTENT_CRUX_MIN_CONSECUTIVE
            and self.load_bearing_score >= PERSISTENT_CRUX_MIN_SCORE
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "crux_id": self.crux_id,
            "statement": self.statement,
            "question_family_id": self.question_family_id,
            "consecutive_debate_count": self.consecutive_debate_count,
            "load_bearing_score": round(self.load_bearing_score, 4),
            "cruxset_receipt_ids": list(self.cruxset_receipt_ids),
            "qualifies": self.qualifies,
        }


@dataclass(frozen=True)
class CruxArbitration:
    """Signed operator arbitration of a persistent crux.

    An arbitration **pins** an operator's chosen side as context for
    future debates on the same ``question_family_id``.  It is never a
    delete; use :func:`build_reversal` to reverse one.

    ``checksum`` covers the arbitration ID, full crux state, operator, side,
    rationale, evidence citations, and timestamps — all fields an attester
    would expect the signature to attest.
    """

    arbitration_id: str
    crux: PersistentCrux
    operator: str
    side: ArbitrationSide
    rationale: str
    evidence_citations: tuple[str, ...]
    created_at: str
    expires_at: str
    is_reversed: bool
    reversal_receipt_id: str
    checksum: str

    @property
    def is_expired(self) -> bool:
        """Return True if this arbitration has passed its expiry timestamp.

        Fails **closed**: an unparseable ``expires_at`` is treated as already
        expired rather than silently granting indefinite validity.
        """
        try:
            return datetime.fromisoformat(self.expires_at) < datetime.now(timezone.utc)
        except ValueError:
            return True  # fail closed — unknown expiry is treated as expired

    def to_dict(self) -> dict[str, Any]:
        return {
            "arbitration_id": self.arbitration_id,
            "crux": self.crux.to_dict(),
            "operator": self.operator,
            "side": self.side,
            "rationale": self.rationale,
            "evidence_citations": list(self.evidence_citations),
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "is_reversed": self.is_reversed,
            "reversal_receipt_id": self.reversal_receipt_id,
            "checksum": self.checksum,
        }


@dataclass(frozen=True)
class CruxArbitrationReversal:
    """Signed reversal receipt for a :class:`CruxArbitration`.

    Reversal is first-class: this receipt documents the reversal, and the
    original arbitration is updated to ``is_reversed=True`` with
    ``reversal_receipt_id`` set.  No record is ever deleted.
    """

    reversal_id: str
    arbitration_id: str
    reversed_by: str
    reason: str
    created_at: str
    checksum: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "reversal_id": self.reversal_id,
            "arbitration_id": self.arbitration_id,
            "reversed_by": self.reversed_by,
            "reason": self.reason,
            "created_at": self.created_at,
            "checksum": self.checksum,
        }


# ──────────────────────────── builders ─────────────────────────────────────────


def _sha256_of(content: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_arbitration(
    crux: PersistentCrux,
    *,
    operator: str,
    side: ArbitrationSide,
    rationale: str,
    evidence_citations: list[str] | None = None,
    expiry_days: int = DEFAULT_EXPIRY_DAYS,
) -> CruxArbitration:
    """Build a new :class:`CruxArbitration` from a :class:`PersistentCrux`.

    Always safe to call regardless of :func:`crux_arbitration_enabled`.
    Callers that act on the result (priors updates, queue mutations) must
    check the flag themselves.
    """
    arbitration_id = "arb_" + uuid.uuid4().hex[:16]
    now = datetime.now(timezone.utc)
    created_at = now.isoformat()
    expires_at = (now + timedelta(days=expiry_days)).isoformat()
    citations: tuple[str, ...] = tuple(evidence_citations or [])
    # Canonical covers all fields an attester would expect to be signed,
    # including full crux state and evidence citations.
    canonical: dict[str, Any] = {
        "arbitration_id": arbitration_id,
        "crux": crux.to_dict(),
        "operator": operator,
        "side": side,
        "rationale": rationale,
        "evidence_citations": sorted(citations),  # sorted for determinism
        "created_at": created_at,
        "expires_at": expires_at,
    }
    return CruxArbitration(
        arbitration_id=arbitration_id,
        crux=crux,
        operator=operator,
        side=side,
        rationale=rationale,
        evidence_citations=citations,
        created_at=created_at,
        expires_at=expires_at,
        is_reversed=False,
        reversal_receipt_id="",
        checksum=_sha256_of(canonical),
    )


def build_reversal(
    arbitration: CruxArbitration,
    *,
    reversed_by: str,
    reason: str,
) -> tuple[CruxArbitration, CruxArbitrationReversal]:
    """Reverse an arbitration.

    Returns ``(updated_arbitration, reversal_receipt)``.  The updated
    arbitration has ``is_reversed=True`` and ``reversal_receipt_id`` set; its
    original ``checksum`` is preserved.  No record is deleted.
    """
    reversal_id = "rev_" + uuid.uuid4().hex[:16]
    now_str = datetime.now(timezone.utc).isoformat()
    canonical: dict[str, Any] = {
        "reversal_id": reversal_id,
        "arbitration_id": arbitration.arbitration_id,
        "reversed_by": reversed_by,
        "reason": reason,
        "created_at": now_str,
    }
    reversal = CruxArbitrationReversal(
        reversal_id=reversal_id,
        arbitration_id=arbitration.arbitration_id,
        reversed_by=reversed_by,
        reason=reason,
        created_at=now_str,
        checksum=_sha256_of(canonical),
    )
    updated = CruxArbitration(
        arbitration_id=arbitration.arbitration_id,
        crux=arbitration.crux,
        operator=arbitration.operator,
        side=arbitration.side,
        rationale=arbitration.rationale,
        evidence_citations=arbitration.evidence_citations,  # already a tuple, safe to share
        created_at=arbitration.created_at,
        expires_at=arbitration.expires_at,
        is_reversed=True,
        reversal_receipt_id=reversal_id,
        checksum=arbitration.checksum,
    )
    return updated, reversal


# ──────────────────────────── public API ───────────────────────────────────────

__all__ = [
    "PERSISTENT_CRUX_MIN_CONSECUTIVE",
    "PERSISTENT_CRUX_MIN_SCORE",
    "DEFAULT_EXPIRY_DAYS",
    "ArbitrationSide",
    "CruxArbitration",
    "CruxArbitrationReversal",
    "PersistentCrux",
    "build_arbitration",
    "build_reversal",
    "crux_arbitration_enabled",
]
