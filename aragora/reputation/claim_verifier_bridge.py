"""Bridge from DIC-14 ClaimResult to AGT-05 reputation types (AGT-05 #6066).

Converts :class:`aragora.epistemic.claim_verifier.ClaimResult` into a
(:class:`~aragora.reputation.types.StakeableClaim`,
:class:`~aragora.reputation.types.ResolvedClaim`) pair for settlement.

Outcome mapping: PASS→"yes", FAIL→"no", STALE→"no", UNSUPPORTED/ERROR→"inconclusive".
Domain: ``DOMAIN_EPISTEMIC_CLAIM``.  Feature-gated via
``ARAGORA_REPUTATION_FLOW_ENABLED`` (off by default).

Advances: AGT-05 (#6066) sub-deliverable 3 — DIC-14 claim verifier resolution
ingestion adapter.  Manifold adapter: :mod:`aragora.reputation.bridge`;
CruxSet adapter: :mod:`aragora.reputation.crux_bridge`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from aragora.reputation.settlement import settle_claim
from aragora.reputation.types import (
    DOMAIN_EPISTEMIC_CLAIM,
    ClaimOutcome,
    ReputationDelta,
    ResolvedClaim,
    StakeableClaim,
)

if TYPE_CHECKING:
    from aragora.epistemic.claim_verifier import ClaimResult

# Maps DIC-14 ClaimStatus values to AGT-05 ClaimOutcome literals.
_STATUS_TO_OUTCOME: dict[str, ClaimOutcome] = {
    "pass": "yes",
    "fail": "no",
    "stale": "no",  # stale evidence → treat as verification failure
    "unsupported": "inconclusive",
    "error": "inconclusive",
}

_RESOLUTION_SOURCE = "dic14_claim_verifier"


def bridge_from_claim_result(
    result: "ClaimResult",
    agent_id: str,
    *,
    stake_units: int = 1,
    resolution_source: str = _RESOLUTION_SOURCE,
) -> tuple[StakeableClaim, ResolvedClaim]:
    """Convert a DIC-14 :class:`~aragora.epistemic.claim_verifier.ClaimResult`
    into an AGT-05 ``(StakeableClaim, ResolvedClaim)`` pair.

    The returned pair feeds directly into
    :func:`aragora.reputation.settlement.settle_claim`.

    Parameters
    ----------
    result:
        The :class:`~aragora.epistemic.claim_verifier.ClaimResult` returned by
        :meth:`~aragora.epistemic.claim_verifier.ClaimVerifier.verify_claim`.
    agent_id:
        The agent that originally asserted or owns this claim.
    stake_units:
        Compute-credit stake committed to this claim assertion.  Minimum 1.
    resolution_source:
        Provenance tag.  Defaults to ``"dic14_claim_verifier"``.

    Notes
    -----
    The implicit position is always ``"yes"`` — the agent claims the
    organizational claim *will* pass verification.  The settlement layer
    then compares this against ``outcome`` to determine the delta.
    """
    now = datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")
    outcome: ClaimOutcome = _STATUS_TO_OUTCOME.get(result.status.value, "inconclusive")

    stakeable = StakeableClaim.create(
        agent_id=agent_id,
        domain=DOMAIN_EPISTEMIC_CLAIM,
        statement=f"Claim {result.claim_id!r} will pass executable verification",
        position="yes",
        stake_units=stake_units,
        resolution_source=resolution_source,
        resolution_id=result.claim_id,
        predicted_probability=None,  # binary claim — no probability needed
        stake_policy="forfeit_on_loss",
        provenance=_provenance(result, agent_id, resolution_source),
        created_at=now,
    )

    resolved = ResolvedClaim(
        claim_id=stakeable.claim_id,
        outcome=outcome,
        resolved_at=now,
        resolution_source=resolution_source,
        evidence=_evidence(result),
    )
    return stakeable, resolved


def settle_from_claim_result(
    result: "ClaimResult",
    agent_id: str,
    *,
    stake_units: int = 1,
    resolution_source: str = _RESOLUTION_SOURCE,
    decay_half_life_days: float | None = 30.0,
) -> ReputationDelta:
    """Bridge + settle a DIC-14 result in one call, always using ``binary`` scoring.

    Equivalent to::

        stakeable, resolved = bridge_from_claim_result(result, agent_id, ...)
        delta = settle_claim(stakeable, resolved, scoring_rule="binary", ...)

    Prevents callers from accidentally using the ``brier_proper`` default,
    which would raise :class:`~aragora.reputation.settlement.SettlementError`
    because ``bridge_from_claim_result`` always sets ``predicted_probability=None``.
    """
    stakeable, resolved = bridge_from_claim_result(
        result,
        agent_id,
        stake_units=stake_units,
        resolution_source=resolution_source,
    )
    return settle_claim(
        stakeable,
        resolved,
        scoring_rule="binary",
        decay_half_life_days=decay_half_life_days,
    )


def _provenance(
    result: "ClaimResult",
    agent_id: str,
    resolution_source: str,
) -> dict[str, Any]:
    return {
        "claim_id": result.claim_id,
        "agent_id": agent_id,
        "source": resolution_source,
        "severity": result.severity,
        "allowed_action": result.allowed_action,
    }


def _evidence(result: "ClaimResult") -> dict[str, Any]:
    return {
        "claim_id": result.claim_id,
        "status": result.status.value,
        "message": result.message,
        "elapsed_ms": result.elapsed_ms,
        "detail": dict(result.detail),
    }
