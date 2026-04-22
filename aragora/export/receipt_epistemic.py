"""DIC-16: bridge helpers for epistemic claim/crux provenance in receipts.

Wires DIC-14 claim-verifier output or DIC-15 CruxSet entries into the
receipt + KM path. Default OFF — caller opt-in only; nothing in the live
debate or dispatch path imports this module.
"""

from __future__ import annotations

import hashlib
from typing import Literal

from aragora.export.decision_receipt import ReceiptVerification

# The five statuses defined by the DIC-14 claim verification runner,
# plus "open" for CruxSet entries not yet resolved.
VerificationStatus = Literal["pass", "fail", "stale", "unsupported", "error", "open"]

_STATUS_TO_VERIFIED: dict[str, bool] = {
    "pass": True,
    "fail": False,
    "stale": False,
    "unsupported": False,
    "error": False,
    "open": False,
}


def _provenance_hash(material: str) -> str:
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def receipt_verification_from_claim_result(
    *,
    claim_id: str,
    statement: str,
    status: VerificationStatus,
    method: str = "claim_verifier",
    evidence_ids: list[str] | None = None,
    source_receipt_id: str | None = None,
    proof_hash: str | None = None,
) -> ReceiptVerification:
    """Build a ``ReceiptVerification`` from a DIC-14 claim-verifier result.

    ``status`` follows the five DIC-14 runner statuses; only ``"pass"``
    sets ``verified=True``. The ``claim_id`` is preserved so the KM
    adapter can link this knowledge item back to the originating
    ``ExecutableClaim`` manifest entry.
    """
    if not claim_id.strip():
        raise ValueError("claim_id must be non-empty")
    if status not in _STATUS_TO_VERIFIED:
        raise ValueError(
            f"unknown status: {status!r}; expected one of {sorted(_STATUS_TO_VERIFIED)}"
        )
    computed_hash = proof_hash or (
        _provenance_hash(f"{claim_id}:{status}") if status == "pass" else None
    )
    return ReceiptVerification(
        claim=statement,
        verified=_STATUS_TO_VERIFIED[status],
        method=method,
        proof_hash=computed_hash,
        claim_id=claim_id,
        crux_id=None,
        evidence_ids=list(evidence_ids or []),
        verification_status=status,
        source_receipt_id=source_receipt_id,
    )


def receipt_verification_from_crux(
    *,
    crux_id: str,
    question: str,
    load_bearing_score: float,
    source_receipt_id: str | None = None,
    evidence_gap_ids: list[str] | None = None,
) -> ReceiptVerification:
    """Build a ``ReceiptVerification`` from a DIC-15 CruxSet crux entry.

    Cruxes are inherently unresolved at debate time. ``verified`` is
    always ``False``; ``verification_status`` is ``"open"``. The
    ``crux_id`` and ``load_bearing_score`` are preserved in the provenance
    chain so the AGT-05 settlement flow can later resolve them.
    """
    if not crux_id.strip():
        raise ValueError("crux_id must be non-empty")
    if not (0.0 <= load_bearing_score <= 1.0):
        raise ValueError(f"load_bearing_score must be in [0, 1]; got {load_bearing_score}")
    return ReceiptVerification(
        claim=question,
        verified=False,
        method="crux_set",
        proof_hash=None,
        claim_id=None,
        crux_id=crux_id,
        evidence_ids=list(evidence_gap_ids or []),
        verification_status="open",
        source_receipt_id=source_receipt_id,
    )


__all__ = [
    "VerificationStatus",
    "receipt_verification_from_claim_result",
    "receipt_verification_from_crux",
]
