"""WorldStateEvent translation layer for DIC-20 / #6031.

Converts external world-state events (CVE drops, API changes,
dependency version bumps, corpus revisions) into ClaimResult objects
that :func:`~aragora.epistemic.decay_monitor.evaluate_unit` can consume.

Flag: ``ARAGORA_WORLD_EVENTS_ENABLED`` (default off).  All computation
is side-effect-free.  Production ingest must use
:func:`world_event_to_claim_results`; tests that need raw translation use
the private unchecked helper.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, cast

from .claim_verifier import ClaimResult, ClaimStatus

if TYPE_CHECKING:
    from .proof_unit import ProofCarryingCodeUnit


class WorldEventKind(str, Enum):
    CVE = "cve"
    API_CHANGE = "api_change"
    DEPENDENCY_BUMP = "dependency_bump"
    CORPUS_REVISION = "corpus_revision"


_WORLD_EVENTS_FLAG = "ARAGORA_WORLD_EVENTS_ENABLED"
_world_events_enabled_override: bool | None = None
_OVERBROAD_SCOPE_TOKENS = {
    "api",
    "auth",
    "com",
    "dev",
    "go",
    "http",
    "https",
    "io",
    "jwt",
    "net",
    "org",
    "ssl",
    "tls",
    "web",
    "www",
}
_VERSION_LIKE_SCOPE_RE = re.compile(r"v\d+[a-z0-9._-]*$")


@dataclass(frozen=True)
class WorldStateEvent:
    """An external event that may invalidate proof-unit assumptions.

    ``affected_scope`` is matched against claim IDs on claim-id boundaries:
    exact IDs, dotted prefixes, or dotted path segments.  Empty or overly broad
    patterns match nothing (safe default).
    """

    event_id: str
    kind: WorldEventKind | str
    description: str
    affected_scope: tuple[str, ...] = field(default_factory=tuple)
    timestamp: str = ""

    def __post_init__(self) -> None:
        try:
            kind = self.kind if isinstance(self.kind, WorldEventKind) else WorldEventKind(self.kind)
        except ValueError as exc:
            raise ValueError(f"Unsupported world event kind: {self.kind!r}") from exc
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "affected_scope", tuple(self.affected_scope))

    def to_dict(self) -> dict[str, Any]:
        kind = cast(WorldEventKind, self.kind)
        d: dict[str, Any] = {
            "event_id": self.event_id,
            "kind": kind.value,
            "description": self.description,
            "affected_scope": list(self.affected_scope),
        }
        if self.timestamp:
            d["timestamp"] = self.timestamp
        return d


def world_events_enabled() -> bool:
    """Return True if callers may act on world-event claim results."""
    if _world_events_enabled_override is not None:
        return _world_events_enabled_override
    raw = str(os.environ.get(_WORLD_EVENTS_FLAG) or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def enable_world_events() -> None:
    """Enable world-event claim translation for the current process.

    Sets a module-level override rather than mutating ``os.environ``.
    Call :func:`reset_world_events` to restore env-var-driven behavior.
    """
    global _world_events_enabled_override
    _world_events_enabled_override = True


def reset_world_events() -> None:
    """Clear the module-level override, reverting to env-var-driven behavior."""
    global _world_events_enabled_override
    _world_events_enabled_override = None


def _safe_scope_pattern(raw: str) -> str | None:
    """Return a scope pattern safe enough to compare against claim IDs."""

    pattern = str(raw).strip().strip(".")
    normalized = pattern.lower()
    if (
        not pattern
        or not any(char.isalpha() for char in pattern)
        or normalized in _OVERBROAD_SCOPE_TOKENS
        or bool(_VERSION_LIKE_SCOPE_RE.fullmatch(normalized))
    ):
        return None
    return pattern


def _claim_matches_scope_pattern(claim_id: str, pattern: str) -> bool:
    """Match *pattern* against *claim_id* without unconstrained substring hits."""

    claim_segments = tuple(segment for segment in claim_id.strip(".").lower().split(".") if segment)
    pattern_segments = tuple(
        segment for segment in pattern.strip(".").lower().split(".") if segment
    )
    if not claim_segments or not pattern_segments or len(pattern_segments) > len(claim_segments):
        return False
    if claim_segments == pattern_segments:
        return True
    if claim_segments[: len(pattern_segments)] == pattern_segments:
        return True
    return any(
        claim_segments[index : index + len(pattern_segments)] == pattern_segments
        for index in range(1, len(claim_segments) - len(pattern_segments) + 1)
    )


def claims_affected_by_event(
    unit: "ProofCarryingCodeUnit",
    event: WorldStateEvent,
) -> frozenset[str]:
    """Return claim IDs from *unit* matching any pattern in *event.affected_scope*.

    Flag-free; pure computation, no side effects.
    """
    safe_patterns = [
        p for raw in event.affected_scope if (p := _safe_scope_pattern(raw)) is not None
    ]
    affected: set[str] = set()
    for claim_id in unit.claims:
        for pattern in safe_patterns:
            if _claim_matches_scope_pattern(claim_id, pattern):
                affected.add(claim_id)
                break
    return frozenset(affected)


def _world_event_to_claim_results_unchecked(
    unit: "ProofCarryingCodeUnit",
    event: WorldStateEvent,
) -> dict[str, ClaimResult]:
    """Translate *event* without checking the live-action feature flag.

    This exists for focused translator tests. Production callers must use
    :func:`world_event_to_claim_results` so stale world-state signals cannot
    propagate while the feature flag is disabled.
    """

    affected = claims_affected_by_event(unit, event)
    kind = cast(WorldEventKind, event.kind)
    msg = f"Invalidated by {kind.value} event {event.event_id!r}: {event.description}"
    return {
        claim_id: ClaimResult(
            claim_id=claim_id,
            status=ClaimStatus.STALE,
            message=msg,
            detail={
                "source": "world_event",
                "event_id": event.event_id,
                "kind": kind.value,
            },
        )
        for claim_id in sorted(affected)
    }


def world_event_to_claim_results(
    unit: "ProofCarryingCodeUnit",
    event: WorldStateEvent,
) -> dict[str, ClaimResult]:
    """Translate *event* into a ``claim_id → ClaimResult(STALE)`` mapping.

    The dict is ready to pass to
    :func:`~aragora.epistemic.decay_monitor.evaluate_unit` as
    ``claim_results``.

    Raises :exc:`RuntimeError` when ``ARAGORA_WORLD_EVENTS_ENABLED`` is not set.
    This keeps world-event propagation fail-closed for production callers.
    """
    if not world_events_enabled():
        raise RuntimeError(
            "ARAGORA_WORLD_EVENTS_ENABLED is not set; "
            "world-event claim results must not propagate to live state."
        )
    return _world_event_to_claim_results_unchecked(unit, event)
