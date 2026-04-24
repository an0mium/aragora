"""Queue autofill types.

This module previously contained ``maybe_autofill_queue`` — an autonomous
dispatcher that scanned the repository for boss-ready candidates and created
GitHub issues when the boss-loop queue stayed empty.  It was removed because
it conflicted with the human-in-the-loop architecture described in
``docs/plans/2026-04-19-batched-pr-review-triage.md``: the batched triage loop
requires that proposed work be visible to the operator *before* it enters the
review queue, not produced by a self-restocking closed loop.

The dispatcher, its feature flag (``ARAGORA_QUEUE_AUTOFILL``), the sentinel
persistence helpers, the default scan / classify / validate fallbacks, and
the metrics emit path have all been removed.  This module now only exposes
the passive data types so a future advisory-packet producer can describe a
queue-autofill proposal without reaching for a parallel set of names.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_EMPTY_TICK_THRESHOLD = 3
DEFAULT_MAX_ISSUES = 3
DEFAULT_MIN_INTERVAL_SECONDS = 3600.0

# Only well-proven scanner categories are eligible for advisory autofill
# proposals.  Anything else is dropped by the classifier before it reaches
# the advisory packet.
ALLOWED_CATEGORIES: frozenset[str] = frozenset({"test_coverage", "broad_exception"})


@dataclass(frozen=True)
class AutofillCandidate:
    """Lightweight description of one proposed autofill candidate."""

    title: str
    category: str
    fingerprint: str
    file_scope: tuple[str, ...]
    lane: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "category": self.category,
            "fingerprint": self.fingerprint,
            "file_scope": list(self.file_scope),
            "lane": self.lane,
        }


@dataclass(frozen=True)
class AutofillResult:
    """Summary of an autofill evaluation.

    In the advisory-only model ``attempted`` records whether the scan ran;
    ``created`` is always empty because nothing is written to GitHub from
    this module.  The future advisory-packet producer can populate these
    fields without also performing any side effects.
    """

    attempted: bool
    reason: str
    consecutive_empty_ticks: int
    threshold: int
    rate_limited: bool = False
    seconds_since_last: float | None = None
    scanned_count: int = 0
    eligible_count: int = 0
    filtered_out: int = 0
    duplicate_count: int = 0
    created: tuple[AutofillCandidate, ...] = ()
    errors: tuple[str, ...] = ()

    @property
    def created_count(self) -> int:
        return len(self.created)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": "queue_autofill",
            "attempted": self.attempted,
            "reason": self.reason,
            "consecutive_empty_ticks": self.consecutive_empty_ticks,
            "threshold": self.threshold,
            "rate_limited": self.rate_limited,
            "seconds_since_last": self.seconds_since_last,
            "scanned_count": self.scanned_count,
            "eligible_count": self.eligible_count,
            "filtered_out": self.filtered_out,
            "duplicate_count": self.duplicate_count,
            "created_count": self.created_count,
            "created": [candidate.to_dict() for candidate in self.created],
            "errors": list(self.errors),
        }


__all__: Sequence[str] = (
    "ALLOWED_CATEGORIES",
    "AutofillCandidate",
    "AutofillResult",
    "DEFAULT_EMPTY_TICK_THRESHOLD",
    "DEFAULT_MAX_ISSUES",
    "DEFAULT_MIN_INTERVAL_SECONDS",
)
