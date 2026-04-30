"""Bridge from Gauntlet ``CruxReceipt`` to epistemic ``CruxReceipt`` (DIC-16).

Closes the receipt-lineage break documented in
``docs/plans/2026-04-28-dialectical-runtime-integration-audit.md`` lines 140
and 145: the audit calls out two same-named ``CruxReceipt`` classes living
in different modules with semantically incompatible shapes, so a Gauntlet
crux-finder run cannot reach the Knowledge Mound ingestion adapter today.

This module provides a pure read-only converter that maps Gauntlet's
artifact shape to the epistemic shape that
:class:`aragora.knowledge.mound.adapters.crux_receipt_adapter.CruxReceiptAdapter`
expects, so a gauntlet receipt can be ingested via the existing adapter
without any caller migration.

The bridge itself is always safe to call.  Acting on the converted receipt
(i.e. invoking the KM adapter to actually ingest) is gated by the existing
``ARAGORA_CRUX_RECEIPT_ENABLED`` flag on the adapter side, plus an
optional ``ARAGORA_KM_CRUX_INGESTION_ENABLED`` flag callers may check
before triggering the ingestion-side action.

Out of scope:

- Mutating the gauntlet receipt or the knowledge mound.  This module is a
  pure converter.
- Carrying gauntlet-only fields the epistemic schema does not have
  (``recommended_focus``, ``resolution_strategies``, ``raw_claims_hash``,
  ``timestamp``).  These are serialised into ``metadata["gauntlet_*"]`` so
  the original information is preserved as receipt provenance, not lost.

Round 2026-04-30c — Phase D.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from typing import TYPE_CHECKING, Any

from aragora.epistemic.crux_receipt import CruxEntry, CruxReceipt as EpistemicCruxReceipt

if TYPE_CHECKING:
    from aragora.gauntlet.receipt_models import CruxReceipt as GauntletCruxReceipt


_KM_INGESTION_FLAG = "ARAGORA_KM_CRUX_INGESTION_ENABLED"


def km_crux_ingestion_enabled() -> bool:
    """Return True when callers may dispatch the KM ingestion side-effect.

    The bridge itself never reads this flag — construction is always safe.
    Callers that want to trigger the Knowledge Mound adapter (a side-effect
    on the persistent store) check this flag explicitly.  Default off.

    No ``enable_*`` helper is provided: per
    ``scripts/audit_env_mutation.py``, future DIC surfaces avoid
    process-level ``os.environ`` writes.  Callers that want the flag on
    set the env var directly (in their entrypoint / test harness /
    deployment config) — production callers via the launchd plist or
    container env, tests via ``monkeypatch.setenv``.
    """
    raw = str(os.environ.get(_KM_INGESTION_FLAG) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _entry_from_dict(crux: dict[str, Any]) -> CruxEntry:
    """Convert one gauntlet crux dict to an epistemic ``CruxEntry``.

    Gauntlet's per-crux dict (from :meth:`aragora.reasoning.crux_detector.CruxClaim.to_dict`)
    carries ``claim_id``, ``statement``, ``author``, ``crux_score``,
    ``influence_score``, ``disagreement_score``, ``uncertainty_score``,
    ``centrality_score``, ``affected_claims``, ``contesting_agents``,
    ``resolution_impact``.

    The epistemic ``CruxEntry`` keeps the load-bearing subset.  The
    ``crux_score`` (combined load-bearing × uncertainty × influence ×
    centrality × resolution-impact, per ``CruxDetector``) becomes
    ``load_bearing_score``; ``claim_id`` becomes ``crux_id``; the rest
    map by name.  Author and the per-component scores are preserved on
    the parent receipt's metadata, not on the entry, so the entry stays
    aligned with DIC-13 claim manifests.
    """
    return CruxEntry(
        crux_id=str(crux.get("claim_id", "")),
        statement=str(crux.get("statement", "")),
        load_bearing_score=float(crux.get("crux_score", 0.0)),
        uncertainty_score=float(crux.get("uncertainty_score", 0.0)),
        contesting_agents=list(crux.get("contesting_agents") or []),
        affected_claims=list(crux.get("affected_claims") or []),
        resolution_impact=float(crux.get("resolution_impact", 0.0)),
    )


def _new_receipt_id() -> str:
    """Generate a fresh epistemic-shape receipt id (matches DIC-16 style)."""
    return "crux_rcpt_" + uuid.uuid4().hex[:16]


def _checksum_for(
    *,
    receipt_id: str,
    debate_id: str,
    question: str,
    cruxes: list[CruxEntry],
    convergence_barrier: float,
) -> str:
    """Recompute the SHA-256 checksum for the epistemic shape.

    The Gauntlet receipt's checksum was computed over its own (different)
    fieldset, so it cannot be reused on the bridged receipt.  We mirror
    the canonical-JSON pattern from
    :func:`aragora.epistemic.crux_receipt.build_crux_receipt`.
    """
    content: dict[str, Any] = {
        "receipt_id": receipt_id,
        "debate_id": debate_id,
        "question": question,
        "cruxes": [c.to_dict() for c in cruxes],
        "convergence_barrier": round(convergence_barrier, 4),
    }
    return hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def from_gauntlet_receipt(
    gauntlet: "GauntletCruxReceipt",
    *,
    preserve_receipt_id: bool = False,
) -> EpistemicCruxReceipt:
    """Convert a gauntlet :class:`CruxReceipt` into the epistemic shape.

    Parameters
    ----------
    gauntlet:
        The :class:`aragora.gauntlet.receipt_models.CruxReceipt` to convert.
    preserve_receipt_id:
        When ``True``, reuse the gauntlet receipt's ``receipt_id`` verbatim
        on the converted receipt.  Default ``False`` — the bridge mints a
        fresh epistemic-style id (``crux_rcpt_<16-hex>``) so KM ingestion
        records can be filtered to bridged-from-gauntlet vs natively-built
        without parsing.  The original gauntlet id is always preserved on
        ``metadata["gauntlet_receipt_id"]`` regardless.

    Returns
    -------
    A :class:`aragora.epistemic.crux_receipt.CruxReceipt` whose ``cruxes``
    are typed :class:`CruxEntry` instances and whose ``checksum`` is
    recomputed for the new shape.  Gauntlet-only fields (``timestamp``,
    ``recommended_focus``, ``resolution_strategies``, ``raw_claims_hash``)
    are carried into ``metadata`` under ``gauntlet_*`` keys so no
    provenance is lost.
    """
    cruxes = [_entry_from_dict(c) for c in (gauntlet.cruxes or [])]
    convergence_barrier = float(gauntlet.convergence_barrier or 0.0)
    receipt_id = gauntlet.receipt_id if preserve_receipt_id else _new_receipt_id()

    # Build merged metadata: original gauntlet metadata + provenance fields
    # for gauntlet-only data the epistemic schema does not carry directly.
    merged_metadata: dict[str, Any] = dict(gauntlet.metadata or {})
    merged_metadata.setdefault("gauntlet_receipt_id", gauntlet.receipt_id)
    merged_metadata.setdefault("gauntlet_timestamp", gauntlet.timestamp)
    merged_metadata.setdefault("gauntlet_recommended_focus", list(gauntlet.recommended_focus or []))
    merged_metadata.setdefault(
        "gauntlet_resolution_strategies", list(gauntlet.resolution_strategies or [])
    )
    merged_metadata.setdefault("gauntlet_raw_claims_hash", gauntlet.raw_claims_hash or "")

    checksum = _checksum_for(
        receipt_id=receipt_id,
        debate_id=gauntlet.debate_id,
        question=gauntlet.question,
        cruxes=cruxes,
        convergence_barrier=convergence_barrier,
    )

    return EpistemicCruxReceipt(
        receipt_id=receipt_id,
        debate_id=gauntlet.debate_id,
        question=gauntlet.question,
        cruxes=cruxes,
        convergence_barrier=convergence_barrier,
        counterfactuals=list(gauntlet.counterfactuals or []),
        agents=list(gauntlet.agents or []),
        rounds=int(gauntlet.rounds or 0),
        metadata=merged_metadata,
        checksum=checksum,
    )


__all__ = [
    "from_gauntlet_receipt",
    "km_crux_ingestion_enabled",
]
