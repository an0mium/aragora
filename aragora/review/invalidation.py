"""Empirical outcome-invalidation classification + threshold derivation (#6375).

This module implements the framework for grounding Commitment 3 of the
Aragora thesis (``docs/THESIS.md``) in empirical baseline measurement.
The thesis specifies a 5% auto-handle outcome-invalidation threshold as
a placeholder; this module supplies the pure-function machinery to
replace it with a measured baseline + safety margin.

Scope
-----

Phase 1 (this module): pure-function classification, baseline computation,
and threshold derivation. No I/O, no SQL, no scheduler, no CLI.
Composes against the existing
:class:`aragora.triage.metrics.TriageDecisionEvent` shape rather than
defining a parallel event type.

Deferred to follow-ups:

  - Recalibration scheduler that emits ``ThresholdUpdateReceipt`` (issue
    #6375 work breakdown step 5).
  - CLI surfacing through ``aragora review-queue baseline`` (separate PR).
  - Replacing the literal ``5%`` in ``docs/THESIS.md`` Commitment 3 with
    the empirically measured value (separate PR; requires real settled-
    decision data, not synthetic).

Why a separate module
---------------------

Outcome-invalidation classification is conceptually narrower than
:mod:`aragora.triage.auto_handle_calibration`: that module tracks
*per-decision-class* success rate to gate auto-handling on a single
class. This module tracks the *aggregate* invalidation rate over
human-settled decisions, which is the reference baseline the auto-
handle threshold derives from. Keeping the two layers separate makes
the audit story crisp: "aggregate human-settled baseline → safety
margin → per-class auto-handle gate".

Outcome-invalidation predicate vocabulary
-----------------------------------------

Per the issue body, an outcome counts as **invalidated** when any of
these signals fire on the decision:

  - ``revert_within_window``  — the merged change was reverted within a
    bounded window (default 14 days). Maps to
    :data:`aragora.triage.auto_handle_calibration.OUTCOME_REVERT`.
  - ``post_merge_incident``   — a post-merge incident was attributed to
    the change. Maps to
    :data:`aragora.triage.auto_handle_calibration.OUTCOME_INCIDENT`.
  - ``human_override_redo``   — a human override forced a redo (e.g.,
    re-opened PR with substantive new commits, or a follow-up PR that
    explicitly fixes the prior settlement).
  - ``rollback``              — an explicit rollback was issued (separate
    from a clean revert; e.g., feature-flag rollback, infra rollback).
  - ``reopened_pr``           — the same PR was reopened after settle.

Predicates are kept as constants (``frozenset`` of canonical strings)
so the vocabulary is reviewable and tests can pin it explicitly. New
signals require an additive change here plus an update to the
classification module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

__all__ = [
    "DEFAULT_BASELINE_WINDOW_DAYS",
    "DEFAULT_MIN_BASELINE_SAMPLES",
    "DEFAULT_REVERT_WINDOW_DAYS",
    "DEFAULT_SAFETY_MARGIN",
    "DEFAULT_MINIMUM_MEANINGFUL_RATE",
    "INVALIDATION_HUMAN_OVERRIDE_REDO",
    "INVALIDATION_POST_MERGE_INCIDENT",
    "INVALIDATION_REOPENED_PR",
    "INVALIDATION_REVERT_WITHIN_WINDOW",
    "INVALIDATION_ROLLBACK",
    "INVALIDATION_SIGNALS",
    "BaselineMeasurement",
    "InvalidatedDecision",
    "ThresholdProposal",
    "classify_invalidation",
    "compute_baseline",
    "derive_threshold",
    "is_invalidated",
]

UTC = timezone.utc

# ---------------------------------------------------------------------------
# Module constants (kept explicit so tests + reviews can reason about them)
# ---------------------------------------------------------------------------

#: Default baseline measurement window (days). The issue specifies a
#: 30-day window for the Commitment 3 revision trigger; we mirror that.
DEFAULT_BASELINE_WINDOW_DAYS: int = 30

#: Default minimum sample size before a baseline is considered usable.
#: The issue specifies "minimum 50 settled decisions; target 200". We
#: take the floor so callers can still surface partial data with the
#: right notes attached, but the threshold-derivation function refuses
#: to emit a non-placeholder threshold below this.
DEFAULT_MIN_BASELINE_SAMPLES: int = 50

#: Default window (days) within which a revert is treated as
#: outcome-invalidating. Issue body suggests 14 days; chosen
#: conservatively so a slow revert (e.g., a flag rollback two weeks
#: later) still counts.
DEFAULT_REVERT_WINDOW_DAYS: int = 14

#: Default safety margin applied to the measured baseline when deriving
#: an auto-handle threshold. Issue body suggests
#: ``max(baseline × 0.5, minimum_meaningful_rate)``; we take 0.5 here
#: and floor it with :data:`DEFAULT_MINIMUM_MEANINGFUL_RATE`.
DEFAULT_SAFETY_MARGIN: float = 0.5

#: Minimum meaningful invalidation rate. Below this, threshold drift
#: noise (single-event flips on small samples) dominates the signal.
#: 0.01 = "1 in 100", which is the smallest fraction we expect to be
#: able to measure reliably with realistic sample sizes.
DEFAULT_MINIMUM_MEANINGFUL_RATE: float = 0.01

# Canonical invalidation-signal labels. Values are the strings that
# appear in serialized payloads / drift receipts; downstream consumers
# branch on them, so keeping them stable is part of the contract.
INVALIDATION_REVERT_WITHIN_WINDOW: str = "revert_within_window"
INVALIDATION_POST_MERGE_INCIDENT: str = "post_merge_incident"
INVALIDATION_HUMAN_OVERRIDE_REDO: str = "human_override_redo"
INVALIDATION_ROLLBACK: str = "rollback"
INVALIDATION_REOPENED_PR: str = "reopened_pr"

#: All canonical invalidation signals. Tests pin this set so adding a
#: new signal forces an explicit acknowledgement.
INVALIDATION_SIGNALS: frozenset[str] = frozenset(
    {
        INVALIDATION_REVERT_WITHIN_WINDOW,
        INVALIDATION_POST_MERGE_INCIDENT,
        INVALIDATION_HUMAN_OVERRIDE_REDO,
        INVALIDATION_ROLLBACK,
        INVALIDATION_REOPENED_PR,
    }
)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InvalidatedDecision:
    """One settled decision that the classification function flagged.

    Captures the decision identifier, the firing signals, and a short
    human-readable rationale per signal so a baseline run is auditable
    after the fact.

    Used by :func:`classify_invalidation` and :func:`compute_baseline`;
    not persisted by this module.
    """

    decision_id: str
    settled_at: datetime
    signals: tuple[str, ...]  # subset of INVALIDATION_SIGNALS
    rationales: tuple[str, ...]  # parallel to signals
    was_human_settled: bool
    was_auto_handled: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "signals", tuple(self.signals))
        object.__setattr__(self, "rationales", tuple(self.rationales))
        if bool(self.was_human_settled) == bool(self.was_auto_handled):
            raise ValueError(
                "exactly one of was_human_settled or was_auto_handled must be true; "
                f"saw was_human_settled={self.was_human_settled!r} "
                f"and was_auto_handled={self.was_auto_handled!r} "
                f"for {self.decision_id!r}"
            )
        if len(self.signals) != len(self.rationales):
            raise ValueError(
                "signals and rationales must be the same length; "
                f"got {len(self.signals)} signals vs {len(self.rationales)} rationales"
            )
        for signal in self.signals:
            if signal not in INVALIDATION_SIGNALS:
                raise ValueError(
                    f"unknown invalidation signal: {signal!r} "
                    f"(must be one of {sorted(INVALIDATION_SIGNALS)})"
                )
        if self.settled_at.tzinfo is None:
            object.__setattr__(self, "settled_at", self.settled_at.replace(tzinfo=UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "settled_at": self.settled_at.astimezone(UTC).isoformat(),
            "signals": list(self.signals),
            "rationales": list(self.rationales),
            "was_human_settled": bool(self.was_human_settled),
            "was_auto_handled": bool(self.was_auto_handled),
        }


@dataclass(frozen=True, slots=True)
class BaselineMeasurement:
    """The output of one baseline-measurement run.

    All rate fields are ``float | None`` — ``None`` means "cannot be
    computed from current data" (sample below
    :data:`DEFAULT_MIN_BASELINE_SAMPLES` or no human-settled decisions in
    the window). The ``notes`` field carries a per-field explanation
    so consumers can tell "no data yet" apart from "zero invalidations".
    """

    window_start: datetime
    window_end: datetime
    window_days: int

    # Sample shapes
    total_human_settled: int
    invalidated_human_settled: int
    total_auto_handled: int
    invalidated_auto_handled: int

    # Rates and confidence interval (Wilson 95%)
    baseline_human_rate: float | None
    baseline_human_rate_ci_low: float | None
    baseline_human_rate_ci_high: float | None
    auto_handle_rate: float | None
    auto_handle_rate_ci_low: float | None
    auto_handle_rate_ci_high: float | None

    # Per-class breakdown (decision_class → invalidated_count / total)
    per_class_human: dict[str, tuple[int, int]] = field(default_factory=dict)
    per_class_auto: dict[str, tuple[int, int]] = field(default_factory=dict)

    # Sample-size acceptability
    min_samples_required: int = DEFAULT_MIN_BASELINE_SAMPLES
    sample_size_acceptable: bool = False

    # Per-field human-readable notes for None values + caveats.
    notes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_start": self.window_start.astimezone(UTC).isoformat(),
            "window_end": self.window_end.astimezone(UTC).isoformat(),
            "window_days": int(self.window_days),
            "total_human_settled": int(self.total_human_settled),
            "invalidated_human_settled": int(self.invalidated_human_settled),
            "total_auto_handled": int(self.total_auto_handled),
            "invalidated_auto_handled": int(self.invalidated_auto_handled),
            "baseline_human_rate": _float_or_none(self.baseline_human_rate),
            "baseline_human_rate_ci_low": _float_or_none(self.baseline_human_rate_ci_low),
            "baseline_human_rate_ci_high": _float_or_none(self.baseline_human_rate_ci_high),
            "auto_handle_rate": _float_or_none(self.auto_handle_rate),
            "auto_handle_rate_ci_low": _float_or_none(self.auto_handle_rate_ci_low),
            "auto_handle_rate_ci_high": _float_or_none(self.auto_handle_rate_ci_high),
            "per_class_human": {k: list(v) for k, v in self.per_class_human.items()},
            "per_class_auto": {k: list(v) for k, v in self.per_class_auto.items()},
            "min_samples_required": int(self.min_samples_required),
            "sample_size_acceptable": bool(self.sample_size_acceptable),
            "notes": dict(self.notes),
        }


@dataclass(frozen=True, slots=True)
class ThresholdProposal:
    """Output of :func:`derive_threshold`.

    Carries the proposed auto-handle invalidation threshold, the
    baseline rate it derives from, the safety margin used, and a
    human-readable rationale.

    The proposal is *advisory*: replacing the literal ``5%`` in
    ``docs/THESIS.md`` Commitment 3 is a separate, human-gated step.
    This dataclass carries all the inputs that decision needs.
    """

    threshold: float | None  # None when baseline is not yet usable
    baseline: float | None
    sample_size: int
    safety_margin: float
    minimum_meaningful_rate: float
    is_placeholder: bool  # True when baseline below sample-size floor
    rationale: str
    measured_at: datetime
    measurement_window_days: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": _float_or_none(self.threshold),
            "baseline": _float_or_none(self.baseline),
            "sample_size": int(self.sample_size),
            "safety_margin": float(self.safety_margin),
            "minimum_meaningful_rate": float(self.minimum_meaningful_rate),
            "is_placeholder": bool(self.is_placeholder),
            "rationale": self.rationale,
            "measured_at": self.measured_at.astimezone(UTC).isoformat(),
            "measurement_window_days": int(self.measurement_window_days),
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_invalidated(signals: Sequence[str]) -> bool:
    """Return True if any of the provided signals is a recognized
    invalidation signal.

    Unknown signals raise ``ValueError`` so vocabulary drift fails loud.
    """
    if not signals:
        return False
    for signal in signals:
        if signal not in INVALIDATION_SIGNALS:
            raise ValueError(
                f"unknown invalidation signal: {signal!r} "
                f"(must be one of {sorted(INVALIDATION_SIGNALS)})"
            )
    return True


def classify_invalidation(
    *,
    decision_id: str,
    settled_at: datetime,
    was_human_settled: bool,
    was_auto_handled: bool,
    reverted_at: datetime | None = None,
    incident_attributed: bool = False,
    rolled_back: bool = False,
    pr_reopened: bool = False,
    human_override_redo_pr: str | None = None,
    revert_window_days: int = DEFAULT_REVERT_WINDOW_DAYS,
) -> InvalidatedDecision | None:
    """Classify a settled decision against the invalidation vocabulary.

    Pure function. Returns ``None`` if no invalidation signal fires;
    otherwise returns an :class:`InvalidatedDecision` carrying the
    matching signals + rationales.

    Args:
        decision_id: Stable identifier for the decision.
        settled_at: When the decision was settled (timezone-aware
            preferred; naive timestamps are coerced to UTC).
        was_human_settled: True when a human took the settlement
            action. Exactly one of this or ``was_auto_handled`` must be
            true.
        was_auto_handled: True when the auto-handle lane handled it.
            Exactly one of this or ``was_human_settled`` must be true.
        reverted_at: When the change was reverted, if it was. ``None``
            means no revert.
        incident_attributed: True when a post-merge incident was
            attributed to this decision.
        rolled_back: True when an explicit rollback was issued
            (separate from a revert).
        pr_reopened: True when the PR was reopened after settlement.
        human_override_redo_pr: PR identifier for a follow-up that
            explicitly fixes the original settlement, or ``None``.
        revert_window_days: Days within which a revert counts as
            invalidation. Defaults to
            :data:`DEFAULT_REVERT_WINDOW_DAYS`.

    Raises:
        ValueError: If the settlement source is ambiguous (both flags
            true or both false).
    """
    if bool(was_human_settled) == bool(was_auto_handled):
        raise ValueError(
            "decision must have exactly one settlement source: "
            f"saw was_human_settled={was_human_settled!r} "
            f"and was_auto_handled={was_auto_handled!r} for {decision_id!r}"
        )
    if revert_window_days <= 0:
        raise ValueError("revert_window_days must be positive")

    settled_at = _ensure_utc(settled_at)
    signals: list[str] = []
    rationales: list[str] = []

    if reverted_at is not None:
        reverted_at = _ensure_utc(reverted_at)
        delta = reverted_at - settled_at
        if timedelta(0) <= delta <= timedelta(days=revert_window_days):
            signals.append(INVALIDATION_REVERT_WITHIN_WINDOW)
            rationales.append(
                f"reverted {_format_timedelta(delta)} after settlement "
                f"(window={revert_window_days}d)"
            )

    if incident_attributed:
        signals.append(INVALIDATION_POST_MERGE_INCIDENT)
        rationales.append("post-merge incident attributed to the decision")

    if human_override_redo_pr:
        signals.append(INVALIDATION_HUMAN_OVERRIDE_REDO)
        rationales.append(f"follow-up PR {human_override_redo_pr} fixes the prior settlement")

    if rolled_back:
        signals.append(INVALIDATION_ROLLBACK)
        rationales.append("explicit rollback issued (distinct from revert)")

    if pr_reopened:
        signals.append(INVALIDATION_REOPENED_PR)
        rationales.append("PR reopened after settlement")

    if not signals:
        return None

    return InvalidatedDecision(
        decision_id=decision_id,
        settled_at=settled_at,
        signals=tuple(signals),
        rationales=tuple(rationales),
        was_human_settled=bool(was_human_settled),
        was_auto_handled=bool(was_auto_handled),
    )


def compute_baseline(
    decisions: Iterable[InvalidatedDecision | dict[str, Any]],
    *,
    total_human_settled: int,
    total_auto_handled: int,
    window_end: datetime,
    window_days: int = DEFAULT_BASELINE_WINDOW_DAYS,
    per_class_human: dict[str, int] | None = None,
    per_class_auto: dict[str, int] | None = None,
    min_samples: int = DEFAULT_MIN_BASELINE_SAMPLES,
) -> BaselineMeasurement:
    """Aggregate classified invalidations into a baseline measurement.

    Pure function. Callers feed:

      - The classified invalidations within the measurement window
        (output of :func:`classify_invalidation`, filtered).
      - The full decision counts (human-settled and auto-handled) over
        the window — these are the *denominators*; classified
        invalidations are the *numerators*.
      - Optional per-class denominator counts to compute per-class
        invalidation rates when sample size supports it.

    The split denominator/numerator design lets callers reuse existing
    decision-event indexes (settlement-receipt counts, brief-receipt
    counts) without re-classifying every settled decision; only
    invalidations need the classification call.

    Args:
        decisions: Iterable of :class:`InvalidatedDecision` (or dict
            payloads of the same shape from a JSON store). Only
            decisions that fall within ``[window_end - window_days,
            window_end]`` are counted; the function filters for you.
        total_human_settled: Total human-settled decision count over
            the window. Must be ``>= number of classified human
            invalidations`` or the function raises.
        total_auto_handled: Total auto-handled decision count over the
            window. Must be ``>= number of classified auto
            invalidations`` or the function raises.
        window_end: Right edge of the measurement window (inclusive).
        window_days: Width of the window in days; defaults to
            :data:`DEFAULT_BASELINE_WINDOW_DAYS`.
        per_class_human: Optional per-class human-settled denominators.
            Keys are decision class strings; values are total counts.
        per_class_auto: Optional per-class auto-handled denominators.
        min_samples: Minimum total sample size before the baseline is
            considered usable; defaults to
            :data:`DEFAULT_MIN_BASELINE_SAMPLES`.

    Returns:
        A :class:`BaselineMeasurement` with rates + Wilson 95% CIs +
        per-field notes.
    """
    if window_days <= 0:
        raise ValueError("window_days must be positive")
    if total_human_settled < 0 or total_auto_handled < 0:
        raise ValueError("decision counts must be non-negative")
    if min_samples <= 0:
        raise ValueError("min_samples must be positive")

    window_end = _ensure_utc(window_end)
    window_start = window_end - timedelta(days=window_days)

    invalidations: list[InvalidatedDecision] = []
    for d in decisions:
        if isinstance(d, dict):
            d = _decision_from_dict(d)
        if not isinstance(d, InvalidatedDecision):
            raise TypeError(
                f"decisions must contain InvalidatedDecision or compatible dicts, got {type(d).__name__}"
            )
        if window_start <= d.settled_at <= window_end:
            invalidations.append(d)

    invalidated_human = sum(1 for d in invalidations if d.was_human_settled)
    invalidated_auto = sum(1 for d in invalidations if d.was_auto_handled)

    if invalidated_human > total_human_settled:
        raise ValueError(
            f"classified human invalidations ({invalidated_human}) exceed "
            f"total human-settled count ({total_human_settled})"
        )
    if invalidated_auto > total_auto_handled:
        raise ValueError(
            f"classified auto invalidations ({invalidated_auto}) exceed "
            f"total auto-handled count ({total_auto_handled})"
        )

    notes: dict[str, str] = {}

    sample_total = total_human_settled + total_auto_handled
    sample_size_acceptable = total_human_settled >= min_samples

    # --- Aggregate rates --------------------------------------------------
    if total_human_settled == 0:
        baseline_rate: float | None = None
        baseline_low = baseline_high = None
        notes["baseline_human_rate"] = "no human-settled decisions in window"
    elif total_human_settled < min_samples:
        baseline_rate = invalidated_human / total_human_settled
        baseline_low, baseline_high = _wilson_interval(invalidated_human, total_human_settled)
        notes["baseline_human_rate"] = (
            f"sample size {total_human_settled} < min_samples={min_samples}; "
            "rate is informational only and must not be used to derive a non-placeholder threshold"
        )
    else:
        baseline_rate = invalidated_human / total_human_settled
        baseline_low, baseline_high = _wilson_interval(invalidated_human, total_human_settled)

    if total_auto_handled == 0:
        auto_rate: float | None = None
        auto_low = auto_high = None
        notes["auto_handle_rate"] = (
            "no auto-handled decisions in window — auto-handle lane not yet active "
            "or all decisions were escalated to human"
        )
    else:
        auto_rate = invalidated_auto / total_auto_handled
        auto_low, auto_high = _wilson_interval(invalidated_auto, total_auto_handled)

    # --- Per-class breakdown ----------------------------------------------
    per_class_human_breakdown: dict[str, tuple[int, int]] = {}
    per_class_auto_breakdown: dict[str, tuple[int, int]] = {}

    # Group classified invalidations by an implicit decision_class field.
    # Since InvalidatedDecision does not carry class today, callers are
    # expected to embed it via the decision_id prefix when needed; for
    # this phase 1 module, per-class breakdown is populated only when
    # the caller passes per-class denominators. The numerator side
    # remains a follow-up (issue #6375 work breakdown step 2 sub-bullet
    # "split by decision class where the sample supports it").
    if per_class_human:
        for cls, total in per_class_human.items():
            per_class_human_breakdown[cls] = (0, int(total))
    if per_class_auto:
        for cls, total in per_class_auto.items():
            per_class_auto_breakdown[cls] = (0, int(total))

    if (per_class_human or per_class_auto) and "per_class_breakdown_numerator" not in notes:
        notes["per_class_breakdown_numerator"] = (
            "per-class invalidation numerators are 0 in phase 1; "
            "follow-up PR will plumb decision_class through InvalidatedDecision"
        )

    if not sample_size_acceptable:
        notes.setdefault(
            "sample_size_acceptable",
            f"need >= {min_samples} human-settled decisions; have {total_human_settled} "
            "(target 200; minimum 50 per issue #6375 work breakdown)",
        )

    if sample_total == 0:
        notes.setdefault(
            "total_decisions",
            "no decisions in window — baseline cannot be measured",
        )

    return BaselineMeasurement(
        window_start=window_start,
        window_end=window_end,
        window_days=int(window_days),
        total_human_settled=int(total_human_settled),
        invalidated_human_settled=int(invalidated_human),
        total_auto_handled=int(total_auto_handled),
        invalidated_auto_handled=int(invalidated_auto),
        baseline_human_rate=baseline_rate,
        baseline_human_rate_ci_low=baseline_low,
        baseline_human_rate_ci_high=baseline_high,
        auto_handle_rate=auto_rate,
        auto_handle_rate_ci_low=auto_low,
        auto_handle_rate_ci_high=auto_high,
        per_class_human=per_class_human_breakdown,
        per_class_auto=per_class_auto_breakdown,
        min_samples_required=int(min_samples),
        sample_size_acceptable=bool(sample_size_acceptable),
        notes=notes,
    )


def derive_threshold(
    measurement: BaselineMeasurement,
    *,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    minimum_meaningful_rate: float = DEFAULT_MINIMUM_MEANINGFUL_RATE,
    measured_at: datetime | None = None,
    placeholder_value: float = 0.05,
) -> ThresholdProposal:
    """Derive an auto-handle invalidation threshold from a baseline.

    The formula is:

        threshold = max(baseline * safety_margin, minimum_meaningful_rate)

    When the baseline is below the sample-size floor (``baseline is None``
    or ``sample_size_acceptable=False``), the function returns a
    *placeholder* proposal that pegs the threshold at ``placeholder_value``
    (default 5%) and marks ``is_placeholder=True``. This preserves the
    current Commitment 3 number while making the placeholder status
    explicit and machine-checkable.

    Args:
        measurement: A :class:`BaselineMeasurement` from
            :func:`compute_baseline`.
        safety_margin: Multiplier applied to the baseline. Defaults to
            :data:`DEFAULT_SAFETY_MARGIN` (0.5). Must be in (0, 1].
        minimum_meaningful_rate: Floor below which threshold drift is
            indistinguishable from sample noise. Defaults to
            :data:`DEFAULT_MINIMUM_MEANINGFUL_RATE` (0.01).
        measured_at: Timestamp recorded on the proposal. Defaults to
            ``datetime.now(UTC)``.
        placeholder_value: Value to use when the baseline is not yet
            usable. Defaults to 0.05 to match the existing Commitment 3
            placeholder.

    Returns:
        A :class:`ThresholdProposal`.
    """
    if not 0 < safety_margin <= 1:
        raise ValueError("safety_margin must be in (0, 1]")
    if minimum_meaningful_rate <= 0:
        raise ValueError("minimum_meaningful_rate must be positive")
    if not 0 < placeholder_value < 1:
        raise ValueError("placeholder_value must be in (0, 1)")

    measured_at = _ensure_utc(measured_at) if measured_at is not None else datetime.now(UTC)

    baseline = measurement.baseline_human_rate
    sample_size = measurement.total_human_settled

    if baseline is None or not measurement.sample_size_acceptable:
        rationale = (
            f"baseline not yet usable: total_human_settled={sample_size} "
            f"vs min_samples_required={measurement.min_samples_required}; "
            f"falling back to placeholder threshold {placeholder_value:.2%} until "
            "enough settled decisions accumulate"
        )
        return ThresholdProposal(
            threshold=float(placeholder_value),
            baseline=_float_or_none(baseline),
            sample_size=int(sample_size),
            safety_margin=float(safety_margin),
            minimum_meaningful_rate=float(minimum_meaningful_rate),
            is_placeholder=True,
            rationale=rationale,
            measured_at=measured_at,
            measurement_window_days=int(measurement.window_days),
        )

    derived = max(baseline * safety_margin, minimum_meaningful_rate)
    rationale = (
        f"threshold = max(baseline {baseline:.4f} * safety_margin {safety_margin:.2f}, "
        f"minimum_meaningful_rate {minimum_meaningful_rate:.4f}) = {derived:.4f} "
        f"({derived:.2%}); measured over {measurement.window_days}d window with "
        f"sample_size={sample_size}"
    )
    return ThresholdProposal(
        threshold=float(derived),
        baseline=float(baseline),
        sample_size=int(sample_size),
        safety_margin=float(safety_margin),
        minimum_meaningful_rate=float(minimum_meaningful_rate),
        is_placeholder=False,
        rationale=rationale,
        measured_at=measured_at,
        measurement_window_days=int(measurement.window_days),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=UTC) if ts.tzinfo is None else ts


def _float_or_none(value: float | None) -> float | None:
    return None if value is None else float(value)


def _format_timedelta(delta: timedelta) -> str:
    """Compact human-readable timedelta (days + hours)."""
    total_seconds = int(delta.total_seconds())
    days, rem = divmod(total_seconds, 86400)
    hours, _ = divmod(rem, 3600)
    if days:
        return f"{days}d{hours}h" if hours else f"{days}d"
    return f"{hours}h" if hours else f"{total_seconds}s"


def _wilson_interval(successes: int, total: int) -> tuple[float | None, float | None]:
    """Return Wilson 95% CI for a binomial proportion.

    More accurate than the normal-approximation interval at small n;
    matches the standard "score" interval used in
    https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval
    with z = 1.96.
    """
    if total <= 0:
        return None, None
    z = 1.96  # 95%
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half_width = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    low = max(0.0, center - half_width)
    high = min(1.0, center + half_width)
    return float(low), float(high)


def _decision_from_dict(payload: dict[str, Any]) -> InvalidatedDecision:
    """Hydrate an :class:`InvalidatedDecision` from a JSON-style dict.

    Tolerant of missing optional fields; raises ``ValueError`` on
    required-field absence so loading a malformed store fails loud.
    """
    try:
        return InvalidatedDecision(
            decision_id=str(payload["decision_id"]),
            settled_at=_parse_dt(payload["settled_at"]),
            signals=tuple(payload.get("signals", ())),
            rationales=tuple(payload.get("rationales", ())),
            was_human_settled=bool(payload.get("was_human_settled", False)),
            was_auto_handled=bool(payload.get("was_auto_handled", False)),
        )
    except KeyError as exc:
        raise ValueError(
            f"missing required field in invalidation payload: {exc.args[0]!r}"
        ) from exc


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if isinstance(value, str):
        # Tolerate trailing Z by replacing with +00:00.
        return _ensure_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))
    raise TypeError(f"cannot parse datetime from {type(value).__name__}: {value!r}")
