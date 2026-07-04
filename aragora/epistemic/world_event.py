"""WorldStateEvent translation layer for DIC-20 / #6031.

Converts external world-state events (CVE drops, API changes,
dependency version bumps, corpus revisions) into ClaimResult objects
that :func:`~aragora.epistemic.decay_monitor.evaluate_unit` can consume.

Flag: ``ARAGORA_WORLD_EVENTS_ENABLED`` (default off).  All computation
is side-effect-free.  The flag gates :func:`world_event_to_claim_results`
default behaviour; pass ``require_enabled=False`` for test-only use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from .claim_verifier import ClaimResult, ClaimStatus

if TYPE_CHECKING:
    from .proof_unit import ProofCarryingCodeUnit


class WorldEventKind(str, Enum):
    CVE = "cve"
    API_CHANGE = "api_change"
    DEPENDENCY_BUMP = "dependency_bump"
    CORPUS_REVISION = "corpus_revision"


@dataclass(frozen=True)
class WorldStateEvent:
    """An external event that may invalidate proof-unit assumptions.

    ``affected_scope`` is matched against claim IDs: a claim is affected
    when its ID starts with or contains any non-empty pattern in the list.
    An empty list matches nothing (safe default).
    """

    event_id: str
    kind: WorldEventKind
    description: str
    affected_scope: list[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "event_id": self.event_id,
            "kind": self.kind.value,
            "description": self.description,
            "affected_scope": list(self.affected_scope),
        }
        if self.timestamp:
            d["timestamp"] = self.timestamp
        return d


def world_events_enabled() -> bool:
    """Return True if callers may act on world-event claim results."""
    raw = str(os.environ.get("ARAGORA_WORLD_EVENTS_ENABLED") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def enable_world_events() -> None:
    """Enable world-event claim translation for the current process."""
    os.environ["ARAGORA_WORLD_EVENTS_ENABLED"] = "1"


def claims_affected_by_event(
    unit: "ProofCarryingCodeUnit",
    event: WorldStateEvent,
) -> frozenset[str]:
    """Return claim IDs from *unit* matching any pattern in *event.affected_scope*.

    Flag-free; pure computation, no side effects.
    """
    affected: set[str] = set()
    for claim_id in unit.claims:
        for pattern in event.affected_scope:
            if pattern and (claim_id.startswith(pattern) or pattern in claim_id):
                affected.add(claim_id)
                break
    return frozenset(affected)


def world_event_to_claim_results(
    unit: "ProofCarryingCodeUnit",
    event: WorldStateEvent,
    *,
    require_enabled: bool = True,
) -> dict[str, ClaimResult]:
    """Translate *event* into a ``claim_id → ClaimResult(STALE)`` mapping.

    The dict is ready to pass to
    :func:`~aragora.epistemic.decay_monitor.evaluate_unit` as
    ``claim_results``.

    Raises :exc:`RuntimeError` when ``require_enabled=True`` and
    ``ARAGORA_WORLD_EVENTS_ENABLED`` is not set.  Pass
    ``require_enabled=False`` for test code that needs the translation
    without environment setup.
    """
    if require_enabled and not world_events_enabled():
        raise RuntimeError(
            "ARAGORA_WORLD_EVENTS_ENABLED is not set; "
            "world-event claim results must not propagate to live state. "
            "Pass require_enabled=False for test-only use."
        )
    affected = claims_affected_by_event(unit, event)
    msg = (
        f"Invalidated by {event.kind.value} event {event.event_id!r}: "
        f"{event.description}"
    )
    return {
        claim_id: ClaimResult(claim_id=claim_id, status=ClaimStatus.STALE, message=msg)
        for claim_id in sorted(affected)
    }
