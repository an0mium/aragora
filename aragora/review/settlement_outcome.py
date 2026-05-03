"""Pure-function observation of post-settlement invalidation signals (#6375 phase 4).

This module bridges the gap between :class:`aragora.cli.commands.review_queue.SettlementReceipt`
(which captures *settlement* events) and the canonical invalidation signals in
:mod:`aragora.review.invalidation` (which capture *post-settlement* invalidation
events). The :func:`observe_outcome` function takes a settled receipt plus a
windowed slice of GitHub timeline events and returns a *new* receipt with the
five ``outcome_*`` signals populated.

Scope
-----

Phase 1 (#6602): pure-function classification + baseline + threshold.
Phase 2 (#6898): on-disk adapters reading existing stores.
Phase 3 (#6898): recalibration scheduler + ``ThresholdUpdateReceipt``.
Phase 4 (this module): observation function that closes the
``schema_gap_human_numerator`` note in the existing invalidation event source.
Phase 5 (separate PR, requires real data): replace the literal ``5%`` in
``docs/THESIS.md`` Commitment 3.

Design constraints
------------------

This is intentionally a **pure function**: no I/O, no SQL, no GitHub API
fetch. Callers fan out the GitHub fetch separately (and rate-limit it) and
then feed the timeline list here. This keeps the observation logic testable
with synthetic fixtures and avoids coupling the schema-augmentation work to
any specific live-fetch path.

The five canonical signals are pinned by their string labels in
:data:`aragora.review.invalidation.INVALIDATION_SIGNALS`. New signals require
an additive change there (and an explicit acknowledgement); this module is
forbidden from adding new signals on its own.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping

from aragora.review.invalidation import (
    DEFAULT_REVERT_WINDOW_DAYS,
    INVALIDATION_HUMAN_OVERRIDE_REDO,
    INVALIDATION_POST_MERGE_INCIDENT,
    INVALIDATION_REOPENED_PR,
    INVALIDATION_REVERT_WITHIN_WINDOW,
    INVALIDATION_ROLLBACK,
    INVALIDATION_SIGNALS,
)

UTC = timezone.utc

__all__ = [
    "INCIDENT_LABELS",
    "ROLLBACK_LABELS",
    "ObservationWindow",
    "observe_outcome",
]


#: Issue labels that count as ``post_merge_incident`` when the issue body
#: mentions the merge SHA, PR number, or named files. Pinned conservatively;
#: new labels must be added here explicitly so the predicate vocabulary
#: remains reviewable.
INCIDENT_LABELS: frozenset[str] = frozenset(
    {"incident", "regression", "revert-target", "boss-stuck"}
)

#: PR or commit-message markers that count as an explicit ``rollback``
#: distinct from a clean revert (e.g., feature-flag rollback, infra rollback).
ROLLBACK_LABELS: frozenset[str] = frozenset({"rollback", "feature-flag-rollback"})


class ObservationWindow:
    """Resolved time window for outcome observation.

    Constructed from a settlement receipt's ``reviewed_at`` plus a
    ``window_days`` parameter. Defaults to ``DEFAULT_REVERT_WINDOW_DAYS``
    (= 14 days) so a slow rollback two weeks later still counts.
    """

    __slots__ = ("start", "end")

    def __init__(self, settled_at: datetime, window_days: int) -> None:
        if settled_at.tzinfo is None:
            raise ValueError("settled_at must be timezone-aware")
        if window_days <= 0:
            raise ValueError(f"window_days must be positive; got {window_days}")
        self.start: datetime = settled_at
        self.end: datetime = settled_at + timedelta(days=window_days)


def _parse_iso(ts: str) -> datetime:
    """Parse ISO 8601 with ``Z`` suffix or offset; always returns UTC."""
    cleaned = ts.replace("Z", "+00:00") if ts.endswith("Z") else ts
    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _event_in_window(event_at: datetime, window: ObservationWindow) -> bool:
    return window.start <= event_at <= window.end


def _is_revert_commit(event: Mapping[str, Any], head_sha: str) -> bool:
    """A commit-event is a revert when its message starts with ``Revert "``
    AND mentions the merge SHA explicitly."""
    if event.get("type") != "commit":
        return False
    message = str(event.get("message", ""))
    if not message.startswith('Revert "'):
        return False
    return head_sha in message or head_sha[:7] in message


def _is_rollback_pr(event: Mapping[str, Any]) -> bool:
    """A PR event with title prefix ``Revert `` or label in ``ROLLBACK_LABELS``."""
    if event.get("type") != "pr_opened":
        return False
    title = str(event.get("title", "")).lower()
    if title.startswith("revert "):
        return True
    labels = {str(label).lower() for label in event.get("labels", [])}
    return bool(labels & ROLLBACK_LABELS)


def _is_post_merge_incident(event: Mapping[str, Any], head_sha: str, pr_number: int) -> bool:
    """An issue event with an incident-class label that mentions the merge."""
    if event.get("type") != "issue_opened":
        return False
    labels = {str(label).lower() for label in event.get("labels", [])}
    if not (labels & INCIDENT_LABELS):
        return False
    body = str(event.get("body", ""))
    title = str(event.get("title", ""))
    haystack = f"{title}\n{body}"
    return head_sha in haystack or head_sha[:7] in haystack or f"#{pr_number}" in haystack


def _is_human_override_redo(event: Mapping[str, Any], pr_number: int) -> bool:
    """A follow-up PR that closes/fixes/supersedes the original PR within the window."""
    if event.get("type") != "pr_opened":
        return False
    body = str(event.get("body", "")).lower()
    title = str(event.get("title", "")).lower()
    haystack = f"{title}\n{body}"
    refs = (
        f"closes #{pr_number}",
        f"fixes #{pr_number}",
        f"resolves #{pr_number}",
        f"supersedes #{pr_number}",
        f"replaces #{pr_number}",
    )
    return any(ref in haystack for ref in refs)


def _is_reopened_pr(event: Mapping[str, Any], pr_number: int) -> bool:
    """A ``pr_reopened`` event for the same PR number."""
    return event.get("type") == "pr_reopened" and event.get("pr_number") == pr_number


def observe_outcome(
    receipt: Any,
    *,
    github_timeline: Iterable[Mapping[str, Any]],
    window_days: int = DEFAULT_REVERT_WINDOW_DAYS,
    observed_at: datetime | None = None,
) -> Any:
    """Return a new receipt with the five ``outcome_*`` signals populated.

    The original receipt is not mutated; ``dataclasses.replace`` is used to
    construct a new frozen-equivalent instance.

    :param receipt: a :class:`SettlementReceipt`. Imported from
        :mod:`aragora.cli.commands.review_queue`; typed as ``Any`` here to
        avoid an import cycle (the CLI module imports many things this
        module's framework path does not need).
    :param github_timeline: an iterable of timeline-event mappings. Each
        event is ``{"type": str, "at": iso8601, ...}`` plus type-specific
        fields. See module docstring for the canonical types.
    :param window_days: observation window width in days. Defaults to
        :data:`DEFAULT_REVERT_WINDOW_DAYS` (= 14). Must be positive.
    :param observed_at: the time at which the observation is being recorded.
        Defaults to ``datetime.now(UTC)``. Set explicitly in tests for
        deterministic receipts.
    :returns: a new ``SettlementReceipt`` with the five ``outcome_*`` fields
        and ``outcome_observed_at`` populated. Fields that are ``False``
        explicitly mean "checked, no signal"; ``None`` means "not yet
        observed" and should never be returned by this function — every
        field is set explicitly to ``True`` or ``False``.

    All five signals are evaluated independently: a single PR can fire
    multiple signals if e.g. it was reverted *and* triggered a post-merge
    incident.
    """
    settled_at = _parse_iso(receipt.reviewed_at)
    window = ObservationWindow(settled_at=settled_at, window_days=window_days)
    head_sha = receipt.head_sha
    pr_number = receipt.pr_number

    revert_within_window = False
    post_merge_incident = False
    human_override_redo = False
    rollback = False
    reopened_pr = False

    for event in github_timeline:
        ts = event.get("at")
        if not isinstance(ts, str):
            continue
        try:
            event_at = _parse_iso(ts)
        except (TypeError, ValueError):
            continue
        if not _event_in_window(event_at, window):
            continue
        if _is_revert_commit(event, head_sha):
            revert_within_window = True
        if _is_rollback_pr(event):
            rollback = True
        if _is_post_merge_incident(event, head_sha, pr_number):
            post_merge_incident = True
        if _is_human_override_redo(event, pr_number):
            human_override_redo = True
        if _is_reopened_pr(event, pr_number):
            reopened_pr = True

    if observed_at is None:
        observed_at = datetime.now(UTC)
    elif observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=UTC)
    else:
        observed_at = observed_at.astimezone(UTC)

    return replace(
        receipt,
        outcome_revert_within_window=revert_within_window,
        outcome_post_merge_incident=post_merge_incident,
        outcome_human_override_redo=human_override_redo,
        outcome_rollback=rollback,
        outcome_reopened_pr=reopened_pr,
        outcome_observed_at=observed_at.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    )


# Sanity check at import time: the five outcome signals here must correspond
# exactly to the canonical invalidation set so the schema cannot silently
# drift. Raises ``RuntimeError`` (rather than ``assert``) so the check is not
# elided under ``python -O`` and survives ruff S101.
_EXPECTED_SIGNALS = {
    INVALIDATION_REVERT_WITHIN_WINDOW,
    INVALIDATION_POST_MERGE_INCIDENT,
    INVALIDATION_HUMAN_OVERRIDE_REDO,
    INVALIDATION_ROLLBACK,
    INVALIDATION_REOPENED_PR,
}
if _EXPECTED_SIGNALS != INVALIDATION_SIGNALS:
    raise RuntimeError(
        "settlement_outcome fields drifted from canonical INVALIDATION_SIGNALS; "
        f"expected={sorted(INVALIDATION_SIGNALS)}, "
        f"got={sorted(_EXPECTED_SIGNALS)}"
    )
