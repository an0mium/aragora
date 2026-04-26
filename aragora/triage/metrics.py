"""Pure-function rolling-window triage metrics (gap #6373).

This module implements the aggregation layer for Commitment 5 of the
Aragora thesis (docs/THESIS.md). The thesis MUSTs four per-window
metrics:

  1. escalation rate            = escalations_to_human / total_decisions
  2. auto-handle override rate  = overridden_auto_handles / auto_handled
  3. human-override-outcome correlation
     = agreement_fraction - disagreement_fraction  (range [-1, 1])
  4. time-per-settlement        = median and p95 settlement duration

Computation is deliberately I/O free. Callers feed an iterable of
:class:`TriageDecisionEvent` values (typically produced by
:mod:`aragora.triage.event_source`) plus a window end time and a
window width in days, and receive a :class:`TriageWindowMetrics`
snapshot.

Sparse-data policy
------------------

When the filtered window contains fewer than :data:`MIN_EVENTS_FOR_METRICS`
events, every rate-style metric is returned as ``None`` rather than a
false-precision fraction computed from too-few samples. The threshold
is exposed as a module constant so callers can override it. The
``total_decisions`` field is always populated so the caller can tell
the difference between "no data yet" and "metrics suppressed".

Schema stability
----------------

All public dataclasses are ``frozen=True``. The module is load-bearing
for two external contracts:

  - ``GET /api/v1/review-queue/triage-metrics`` (handler and OpenAPI
    response schema consume :meth:`TriageWindowMetrics.to_dict`).
  - ``compute_window`` is imported by tests as a pure-function contract.

Changing the key names in :meth:`TriageWindowMetrics.to_dict` breaks
the API surface; add new keys, do not rename existing ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from statistics import median
from typing import Any, Iterable

__all__ = [
    "MIN_EVENTS_FOR_METRICS",
    "TriageDecisionEvent",
    "TriageWindowMetrics",
    "compute_window",
    "detect_drift",
]

UTC = timezone.utc

# Sparse-data threshold — windows with fewer events than this return
# ``None`` for rate/duration metrics. Chosen conservatively (10) per
# the PR acceptance criteria: anything less produces false-precision
# ratios that can mislead drift detection.
MIN_EVENTS_FOR_METRICS = 10


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TriageDecisionEvent:
    """One decision the triage layer made.

    The shape of this event is the contract between
    :mod:`aragora.triage.event_source` (producer) and
    :func:`compute_window` (consumer). Fields that cannot be reconstructed
    from the existing receipt schemas today are typed as ``| None`` so
    the event source can emit them honestly rather than fabricate values.

    Attributes:
        decision_id:
            Stable identifier for the decision. For PR settlements this
            is typically ``pr-{n}-{session_id}-{action}``. Used only for
            debugging / idempotency; not aggregated.
        ts:
            Timezone-aware timestamp at which the decision was
            **settled** (not queued or generated). Rolling windows are
            computed relative to this moment.
        was_escalated:
            True iff the triage layer routed this decision to a human
            instead of auto-handling it. For current PDB receipts, this
            maps to the ``queue_bucket`` ∈ ``{"needs_attention",
            "repairable"}`` or any decision whose ``machine_recommendation``
            was ``needs_human_attention`` / ``needs_human_review``.
        was_auto_handled:
            True iff the decision was handled without a human in the
            loop. Currently always False in Aragora (every PR settlement
            is human-gated by design); exposed on the event so the
            metric stays forward-compatible with the #6372 auto-handle
            lane when it lands.
        was_human_override:
            True iff the human's settlement action contradicts the
            ensemble's ``ensemble_recommendation``. Used for both the
            auto-handle override rate and the outcome-correlation
            metric.
        ensemble_recommendation:
            The ensemble / machine recommendation attached to the
            decision (e.g., ``"approve_candidate"``,
            ``"needs_human_attention"``, ``"approve"``, ``"defer"``).
            Empty string when no machine recommendation was recorded —
            event source must not silently coerce to a plausible value.
        final_outcome:
            The downstream outcome (e.g., ``"merged"``, ``"closed"``,
            ``"pending"``). ``None`` when the outcome has not been
            recorded yet — most current receipts do NOT carry this
            field, which is documented as a follow-up gap.
        settlement_duration_seconds:
            Wall-clock seconds spent between
            ``settlement_started_at`` and ``settlement_completed_at``.
            For legacy receipts without explicit start/stop, callers
            may supply the ``elapsed_seconds`` field from the review-
            queue settlement receipt. ``None`` when unknown.
    """

    decision_id: str
    ts: datetime
    was_escalated: bool
    was_auto_handled: bool
    was_human_override: bool
    ensemble_recommendation: str
    final_outcome: str | None
    settlement_duration_seconds: float | None

    def __post_init__(self) -> None:
        # Normalise naive timestamps to UTC so window arithmetic is
        # deterministic. frozen=True + slots=True means we go through
        # object.__setattr__ for mutation.
        if self.ts.tzinfo is None:
            object.__setattr__(self, "ts", self.ts.replace(tzinfo=UTC))


@dataclass(frozen=True, slots=True)
class TriageWindowMetrics:
    """Aggregated triage metrics for one rolling window.

    Every rate-style field is ``float | None``. ``None`` means "cannot
    be computed from current window data" — either because the window
    has fewer than :data:`MIN_EVENTS_FOR_METRICS` events, because the
    denominator for that metric is zero (e.g., no auto-handled events
    to compute the override rate over), or because required upstream
    data (``final_outcome``, ``settlement_duration_seconds``) is not
    populated yet. The handler serializes ``None`` as JSON ``null``.

    The ``notes`` field carries human-readable annotations for each
    ``None`` value so API consumers can tell "zero data" apart from
    "upstream schema gap".
    """

    window_start: datetime
    window_end: datetime
    window_label: str  # canonical human label, e.g. "7d" / "30d"
    window_days: int
    total_decisions: int

    escalation_rate: float | None
    auto_handle_override_rate: float | None
    human_override_outcome_correlation: float | None
    settlement_duration_median_s: float | None
    settlement_duration_p95_s: float | None

    # Counts used in the ratios — handy for clients wanting to render
    # confidence bars or expose the raw numbers in dashboards.
    escalations: int = 0
    auto_handled: int = 0
    auto_handle_overrides: int = 0
    human_overrides: int = 0
    human_overrides_with_outcome: int = 0
    settlement_samples: int = 0

    # Human-readable explanation keyed by metric name for every None
    # value above. Empty when nothing was suppressed.
    notes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for JSON response. Timestamps are ISO-8601."""
        return {
            "window_label": self.window_label,
            "window_days": int(self.window_days),
            "window_start": self.window_start.astimezone(UTC).isoformat(),
            "window_end": self.window_end.astimezone(UTC).isoformat(),
            "total_decisions": int(self.total_decisions),
            "escalation_rate": _float_or_none(self.escalation_rate),
            "auto_handle_override_rate": _float_or_none(self.auto_handle_override_rate),
            "human_override_outcome_correlation": _float_or_none(
                self.human_override_outcome_correlation
            ),
            "settlement_duration_median_s": _float_or_none(self.settlement_duration_median_s),
            "settlement_duration_p95_s": _float_or_none(self.settlement_duration_p95_s),
            "counts": {
                "escalations": int(self.escalations),
                "auto_handled": int(self.auto_handled),
                "auto_handle_overrides": int(self.auto_handle_overrides),
                "human_overrides": int(self.human_overrides),
                "human_overrides_with_outcome": int(self.human_overrides_with_outcome),
                "settlement_samples": int(self.settlement_samples),
            },
            "notes": dict(self.notes),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_window(
    events: Iterable[TriageDecisionEvent],
    *,
    window_end: datetime,
    window_days: int,
    min_events: int = MIN_EVENTS_FOR_METRICS,
) -> TriageWindowMetrics:
    """Compute the four Commitment-5 metrics over a rolling window.

    Pure function — no I/O, no global state, no mutation of inputs.

    Args:
        events: Iterable of :class:`TriageDecisionEvent`. May be
            unsorted and may contain events outside the window; both
            are filtered inside this function.
        window_end: Right edge of the rolling window (inclusive). Naive
            datetimes are normalized to UTC.
        window_days: Width of the window in days. Must be positive.
        min_events: Minimum events required before rate-style metrics
            are returned. Defaults to :data:`MIN_EVENTS_FOR_METRICS`.

    Returns:
        A :class:`TriageWindowMetrics` snapshot. All None-valued
        metrics carry an entry in ``notes`` explaining the suppression.
    """
    if window_days <= 0:
        raise ValueError("window_days must be positive")

    window_end = _ensure_utc(window_end)
    window_start = window_end - timedelta(days=window_days)

    filtered = [e for e in events if window_start <= _ensure_utc(e.ts) <= window_end]

    total_decisions = len(filtered)
    notes: dict[str, str] = {}

    escalations = sum(1 for e in filtered if e.was_escalated)
    auto_handled = sum(1 for e in filtered if e.was_auto_handled)
    auto_handle_overrides = sum(1 for e in filtered if e.was_auto_handled and e.was_human_override)
    human_overrides = sum(1 for e in filtered if e.was_human_override)

    # Outcome correlation only counts human-override events that have
    # a resolved final_outcome. An override "confirms" the ensemble
    # when the outcome matches what the ensemble recommended; it
    # "disagrees" otherwise.
    agreement = disagreement = 0
    overrides_with_outcome = 0
    for event in filtered:
        if not event.was_human_override:
            continue
        if event.final_outcome is None:
            continue
        overrides_with_outcome += 1
        if _recommendation_matches_outcome(event.ensemble_recommendation, event.final_outcome):
            agreement += 1
        else:
            disagreement += 1

    # Settlement durations for escalated decisions only — the thesis
    # asks for time-per-settlement specifically on the escalated path.
    durations = [
        float(e.settlement_duration_seconds)
        for e in filtered
        if e.was_escalated and e.settlement_duration_seconds is not None
    ]

    sparse = total_decisions < min_events

    # --- Escalation rate ---------------------------------------------------
    if sparse:
        escalation_rate: float | None = None
        notes["escalation_rate"] = (
            f"insufficient data: {total_decisions} decisions < min_events={min_events}"
        )
    else:
        escalation_rate = escalations / total_decisions if total_decisions else None
        if escalation_rate is None:
            notes["escalation_rate"] = "no decisions in window"

    # --- Auto-handle override rate -----------------------------------------
    if sparse:
        auto_handle_override_rate: float | None = None
        notes["auto_handle_override_rate"] = (
            f"insufficient data: {total_decisions} decisions < min_events={min_events}"
        )
    elif auto_handled == 0:
        auto_handle_override_rate = None
        notes["auto_handle_override_rate"] = (
            "no auto-handled decisions in window — auto-handle lane not yet active"
        )
    else:
        auto_handle_override_rate = auto_handle_overrides / auto_handled

    # --- Human-override-outcome correlation -------------------------------
    if sparse:
        human_override_outcome_correlation: float | None = None
        notes["human_override_outcome_correlation"] = (
            f"insufficient data: {total_decisions} decisions < min_events={min_events}"
        )
    elif overrides_with_outcome == 0:
        human_override_outcome_correlation = None
        notes["human_override_outcome_correlation"] = (
            "no human-override decisions with recorded final_outcome; "
            "outcome field not yet populated in settlement receipts "
            "(tracked as follow-up: receipt-schema extension)"
        )
    else:
        human_override_outcome_correlation = (agreement - disagreement) / overrides_with_outcome

    # --- Settlement duration ----------------------------------------------
    if sparse:
        settlement_duration_median: float | None = None
        settlement_duration_p95: float | None = None
        notes["settlement_duration_median_s"] = (
            f"insufficient data: {total_decisions} decisions < min_events={min_events}"
        )
        notes["settlement_duration_p95_s"] = notes["settlement_duration_median_s"]
    elif not durations:
        settlement_duration_median = None
        settlement_duration_p95 = None
        notes["settlement_duration_median_s"] = (
            "no escalated decisions with settlement_duration_seconds in window"
        )
        notes["settlement_duration_p95_s"] = notes["settlement_duration_median_s"]
    else:
        settlement_duration_median = float(median(durations))
        settlement_duration_p95 = _percentile(durations, 95)

    return TriageWindowMetrics(
        window_start=window_start,
        window_end=window_end,
        window_label=f"{int(window_days)}d",
        window_days=int(window_days),
        total_decisions=total_decisions,
        escalation_rate=escalation_rate,
        auto_handle_override_rate=auto_handle_override_rate,
        human_override_outcome_correlation=human_override_outcome_correlation,
        settlement_duration_median_s=settlement_duration_median,
        settlement_duration_p95_s=settlement_duration_p95,
        escalations=escalations,
        auto_handled=auto_handled,
        auto_handle_overrides=auto_handle_overrides,
        human_overrides=human_overrides,
        human_overrides_with_outcome=overrides_with_outcome,
        settlement_samples=len(durations),
        notes=notes,
    )


def detect_drift(
    current: TriageWindowMetrics,
    previous: TriageWindowMetrics,
    *,
    threshold: float = 0.10,
) -> dict[str, dict[str, Any]]:
    """Return advisory drift info per metric.

    The result is a flat mapping ``metric_name → {current, previous,
    delta, exceeded_threshold}``. ``delta`` is ``current - previous``
    (signed). ``exceeded_threshold`` is ``abs(delta) > threshold`` when
    both values are present; ``False`` when either value is ``None``
    (we cannot drift-check a metric we don't have).

    This is advisory only — the thesis reserves the *decision* to
    revise the triage policy to humans. This function does not raise
    or block anything.

    Args:
        current: Latest window snapshot.
        previous: Prior window snapshot (same or different width).
        threshold: Absolute-delta threshold. Default 10 % matches the
            acceptance criteria in gap #6373; callers may tighten or
            loosen it.

    Returns:
        Mapping with entries for each of the four Commitment-5 metrics.
    """
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    result: dict[str, dict[str, Any]] = {}
    for name in (
        "escalation_rate",
        "auto_handle_override_rate",
        "human_override_outcome_correlation",
        "settlement_duration_median_s",
        "settlement_duration_p95_s",
    ):
        cur = getattr(current, name)
        prev = getattr(previous, name)
        if cur is None or prev is None:
            result[name] = {
                "current": _float_or_none(cur),
                "previous": _float_or_none(prev),
                "delta": None,
                "exceeded_threshold": False,
            }
            continue
        delta = float(cur) - float(prev)
        result[name] = {
            "current": float(cur),
            "previous": float(prev),
            "delta": delta,
            "exceeded_threshold": abs(delta) > threshold,
        }
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Mapping of ensemble recommendations to the "success" outcome labels
# that would *confirm* them. Kept conservative: anything not listed
# here treats the recommendation as ambiguous (so a human override is
# neither confirmed nor disagreed).
_APPROVE_RECOMMENDATIONS = frozenset(
    {
        "approve",
        "approve_candidate",
        "approve_with_followups",
        "approve_now",
        "ready_now",
        "safe_to_merge",
    }
)
_REJECT_RECOMMENDATIONS = frozenset(
    {
        "reject",
        "request_changes",
        "needs_human_attention",
        "needs_human_review",
        "block",
    }
)
_POSITIVE_OUTCOMES = frozenset({"merged", "shipped", "approved", "accepted"})
_NEGATIVE_OUTCOMES = frozenset({"closed", "reverted", "rejected", "blocked"})


def _recommendation_matches_outcome(recommendation: str, outcome: str) -> bool:
    """Return True when the final outcome agrees with the recommendation.

    - approve-family rec + positive outcome → agreement
    - reject-family rec + negative outcome → agreement
    - otherwise → disagreement (including unknown labels, which are
      counted as disagreement so unrecognized vocab is flagged rather
      than silently treated as confirmation).
    """
    rec = (recommendation or "").strip().lower()
    out = (outcome or "").strip().lower()
    if rec in _APPROVE_RECOMMENDATIONS and out in _POSITIVE_OUTCOMES:
        return True
    if rec in _REJECT_RECOMMENDATIONS and out in _NEGATIVE_OUTCOMES:
        return True
    return False


def _ensure_utc(ts: datetime) -> datetime:
    """Return ``ts`` as a UTC-aware datetime (no conversion of aware ts)."""
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts


def _percentile(values: list[float], pct: float) -> float:
    """Return the ``pct`` percentile of ``values`` using linear interpolation.

    Mirrors numpy's default (``linear`` interpolation). Callers are
    responsible for passing a non-empty list; this function will raise
    ValueError otherwise.
    """
    if not values:
        raise ValueError("percentile of empty sequence")
    if not 0 <= pct <= 100:
        raise ValueError("pct must be in [0, 100]")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (len(ordered) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(ordered[int(rank)])
    weight = rank - lower
    return float(ordered[lower]) * (1.0 - weight) + float(ordered[upper]) * weight


def _float_or_none(value: float | None) -> float | None:
    """Safe float coercion that preserves ``None``."""
    if value is None:
        return None
    return float(value)
