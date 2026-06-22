"""Public crux-finder exposure handler mixin (#8227).

Surfaces the load-bearing disagreements produced by a
``consensus="crux_finder"`` debate run over the public API. The
deliverable of a crux-finder run is *not* a verdict: it is a ranked map
of the cruxes (load-bearing disagreements) extracted from the debate's
belief network.

Storage contract (current main, #8366-aware):
    The crux-finder consensus mode persists its output on the debate's
    ``ConsensusProof``. ``build_proof_from_crux_finder`` (in
    ``aragora.debate.consensus``) writes the serialized cruxes to
    ``consensus_proof.metadata["cruxes"]`` and stamps
    ``consensus_proof.metadata["consensus_mode"] == "crux_finder"``. A
    summary also lands in ``formal_verification["crux_finder"]``.

    Note: the flag-gated DIC-15 ``CruxSet`` bridge
    (``ARAGORA_CRUXSET_EMISSION_ENABLED``, default OFF) is a *separate*
    receipt-ingestion path; it does not replace the proof-metadata path
    that the live arena always writes. This handler therefore reads from
    the proof metadata, which is the authoritative arena-path location.

Honest-absence contract:
    - 404 only when the debate itself does not exist.
    - When the debate exists but crux mode was not run (or was skipped /
      fell back to another consensus mode), return HTTP 200 with
      ``{"status": "absent", "reason": ...}`` and ``cruxes: []``. We never
      fabricate cruxes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

from aragora.rbac.decorators import require_permission

from ..base import (
    HandlerResult,
    error_response,
    handle_errors,
    json_response,
    require_storage,
)
from ..openapi_decorator import api_endpoint

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class _DebatesHandlerProtocol(Protocol):
    """Protocol defining the interface expected by CruxOperationsMixin."""

    ctx: dict[str, Any]

    def get_storage(self) -> Any | None:
        """Get debate storage instance."""
        ...


def _as_dict(value: Any) -> dict[str, Any]:
    """Coerce a possibly-None or object value into a plain dict.

    Stored debates round-trip through JSON, so nested structures are
    normally already dicts. We coerce defensively (via ``to_dict`` when
    present) so the handler is robust against in-memory objects too.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            result = to_dict()
        except (TypeError, ValueError):
            return {}
        return result if isinstance(result, dict) else {}
    return {}


def extract_cruxes(debate: dict[str, Any]) -> dict[str, Any]:
    """Extract crux-finder output from a stored debate dict.

    Returns a normalized payload. When crux mode was not run, the payload
    has ``status="absent"`` and an empty ``cruxes`` list — never fabricated
    data.

    Read order (authoritative arena-path first):
      1. ``consensus_proof.metadata["cruxes"]`` — the canonical location
         written by ``build_proof_from_crux_finder``.
      2. ``formal_verification["crux_finder"]`` — summary block (count,
         convergence_barrier, recommended_focus). Used to enrich the
         response and as a secondary presence signal.
    """
    consensus_proof = _as_dict(debate.get("consensus_proof"))
    proof_metadata = _as_dict(consensus_proof.get("metadata"))
    formal_verification = _as_dict(debate.get("formal_verification"))
    crux_summary = _as_dict(formal_verification.get("crux_finder"))

    consensus_mode = proof_metadata.get("consensus_mode")
    raw_cruxes = proof_metadata.get("cruxes")
    cruxes = list(raw_cruxes) if isinstance(raw_cruxes, list) else []

    # Detect whether crux mode was actually run.
    crux_mode_ran = consensus_mode == "crux_finder" or bool(crux_summary)

    if not crux_mode_ran:
        # Honest absence: crux mode was not run for this debate.
        reason = (
            "Crux-finder mode was not run for this debate. "
            "Run the debate with consensus='crux_finder' "
            "(CLI: `aragora ask <task> --crux`) to populate cruxes."
        )
        return {
            "status": "absent",
            "reason": reason,
            "cruxes": [],
            "crux_count": 0,
        }

    # Crux mode ran. It may legitimately have found zero cruxes (e.g. the
    # belief network had no load-bearing disagreements) — that is still a
    # present, honest result distinct from "mode not run".
    convergence_barrier = crux_summary.get(
        "convergence_barrier", proof_metadata.get("convergence_barrier")
    )
    recommended_focus = crux_summary.get(
        "recommended_focus", proof_metadata.get("recommended_focus", [])
    )
    counterfactuals = proof_metadata.get("counterfactuals", [])

    payload: dict[str, Any] = {
        "status": "present",
        "consensus_mode": "crux_finder",
        "cruxes": cruxes,
        "crux_count": len(cruxes),
        "convergence_barrier": convergence_barrier,
        "recommended_focus": list(recommended_focus) if isinstance(recommended_focus, list) else [],
        "counterfactuals": list(counterfactuals) if isinstance(counterfactuals, list) else [],
    }

    # Surface a skip reason if the run fell back (e.g. no belief network).
    debate_metadata = _as_dict(debate.get("metadata"))
    skip_reason = debate_metadata.get("crux_finder_skipped_reason")
    if skip_reason:
        payload["status"] = "absent"
        payload["reason"] = f"Crux-finder mode was requested but did not complete: {skip_reason}."
        payload["fallback_consensus"] = debate_metadata.get("crux_finder_fallback_consensus")

    return payload


class CruxOperationsMixin:
    """Mixin providing public crux-finder exposure for DebatesHandler."""

    @api_endpoint(
        method="GET",
        path="/api/v1/debates/{id}/cruxes",
        summary="Get crux-finder map",
        description=(
            "Get the ranked map of load-bearing disagreements (cruxes) "
            "produced by a crux-finder debate run. Returns an explicit "
            "honest-absence response when crux mode was not run; never "
            "fabricates cruxes. 404 only for a missing debate."
        ),
        tags=["Debates", "Analysis"],
        parameters=[
            {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}},
        ],
        responses={
            "200": {
                "description": "Crux map (present) or honest-absence response",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "debate_id": {"type": "string"},
                                "status": {
                                    "type": "string",
                                    "enum": ["present", "absent"],
                                },
                                "reason": {"type": "string"},
                                "consensus_mode": {"type": "string"},
                                "crux_count": {"type": "integer"},
                                "convergence_barrier": {"type": "number"},
                                "recommended_focus": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "cruxes": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "claim_id": {"type": "string"},
                                            "statement": {"type": "string"},
                                            "crux_score": {"type": "number"},
                                            "influence_score": {"type": "number"},
                                            "disagreement_score": {"type": "number"},
                                            "uncertainty_score": {"type": "number"},
                                            "resolution_impact": {"type": "number"},
                                            "contesting_agents": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                            "affected_claims": {
                                                "type": "array",
                                                "items": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                                "counterfactuals": {
                                    "type": "array",
                                    "items": {"type": "object"},
                                },
                            },
                        },
                    },
                },
            },
            "401": {"description": "Unauthorized"},
            "404": {"description": "Debate not found"},
        },
    )
    @require_permission("debates:read")
    @require_storage
    @handle_errors("get cruxes")
    def _get_cruxes(self: _DebatesHandlerProtocol, handler: Any, debate_id: str) -> HandlerResult:
        """Get the crux-finder map for a debate.

        See :func:`extract_cruxes` for the storage contract and the
        honest-absence semantics.
        """
        storage = self.get_storage()
        if storage is None:
            return error_response("Storage not available", 503)
        debate = storage.get_debate(debate_id)
        if not debate:
            return error_response(f"Debate not found: {debate_id}", 404)

        payload = extract_cruxes(debate)
        payload["debate_id"] = debate_id
        return json_response(payload)


__all__ = ["CruxOperationsMixin", "extract_cruxes"]
