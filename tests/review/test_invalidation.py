"""Unit tests for aragora.review.invalidation (#6375).

Validates the empirical outcome-invalidation classification + baseline
+ threshold-derivation pipeline that grounds Commitment 3 of the
Aragora thesis. All tests are pure (no I/O, no SQL) and feed
synthetic decisions through the public API.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aragora.review import (
    BaselineMeasurement,
    DEFAULT_BASELINE_WINDOW_DAYS,
    DEFAULT_MINIMUM_MEANINGFUL_RATE,
    DEFAULT_MIN_BASELINE_SAMPLES,
    DEFAULT_REVERT_WINDOW_DAYS,
    DEFAULT_SAFETY_MARGIN,
    INVALIDATION_HUMAN_OVERRIDE_REDO,
    INVALIDATION_POST_MERGE_INCIDENT,
    INVALIDATION_REOPENED_PR,
    INVALIDATION_REVERT_WITHIN_WINDOW,
    INVALIDATION_ROLLBACK,
    INVALIDATION_SIGNALS,
    InvalidatedDecision,
    ThresholdProposal,
    classify_invalidation,
    compute_baseline,
    derive_threshold,
    is_invalidated,
)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Vocabulary tests
# ---------------------------------------------------------------------------


def test_invalidation_signals_pinned() -> None:
    """The vocabulary is exact — adding a signal must update this set."""
    assert INVALIDATION_SIGNALS == frozenset(
        {
            INVALIDATION_REVERT_WITHIN_WINDOW,
            INVALIDATION_POST_MERGE_INCIDENT,
            INVALIDATION_HUMAN_OVERRIDE_REDO,
            INVALIDATION_ROLLBACK,
            INVALIDATION_REOPENED_PR,
        }
    )


def test_canonical_signal_strings_pinned() -> None:
    """Serialized labels are part of the contract — pin them."""
    assert INVALIDATION_REVERT_WITHIN_WINDOW == "revert_within_window"
    assert INVALIDATION_POST_MERGE_INCIDENT == "post_merge_incident"
    assert INVALIDATION_HUMAN_OVERRIDE_REDO == "human_override_redo"
    assert INVALIDATION_ROLLBACK == "rollback"
    assert INVALIDATION_REOPENED_PR == "reopened_pr"


def test_default_constants_documented() -> None:
    """Defaults match docstring + issue #6375 work breakdown."""
    assert DEFAULT_BASELINE_WINDOW_DAYS == 30
    assert DEFAULT_MIN_BASELINE_SAMPLES == 50
    assert DEFAULT_REVERT_WINDOW_DAYS == 14
    assert DEFAULT_SAFETY_MARGIN == 0.5
    assert DEFAULT_MINIMUM_MEANINGFUL_RATE == 0.01


def test_is_invalidated_truthy_for_known_signals() -> None:
    assert is_invalidated([INVALIDATION_REVERT_WITHIN_WINDOW]) is True
    assert (
        is_invalidated([INVALIDATION_REVERT_WITHIN_WINDOW, INVALIDATION_POST_MERGE_INCIDENT])
        is True
    )


def test_is_invalidated_false_for_empty_signals() -> None:
    assert is_invalidated([]) is False
    assert is_invalidated(()) is False


def test_is_invalidated_rejects_unknown_signal() -> None:
    with pytest.raises(ValueError, match="unknown invalidation signal"):
        is_invalidated(["future_invented_signal"])


# ---------------------------------------------------------------------------
# classify_invalidation tests
# ---------------------------------------------------------------------------


def _settled(at: datetime = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)) -> datetime:
    return at


def test_classify_returns_none_when_no_signals() -> None:
    result = classify_invalidation(
        decision_id="pr-1",
        settled_at=_settled(),
        was_human_settled=True,
        was_auto_handled=False,
    )
    assert result is None


def test_classify_revert_within_window() -> None:
    settled = _settled()
    result = classify_invalidation(
        decision_id="pr-2",
        settled_at=settled,
        was_human_settled=True,
        was_auto_handled=False,
        reverted_at=settled + timedelta(days=3),
    )
    assert result is not None
    assert result.signals == (INVALIDATION_REVERT_WITHIN_WINDOW,)
    assert "3d" in result.rationales[0]
    assert result.was_human_settled is True


def test_classify_revert_outside_window_does_not_fire() -> None:
    settled = _settled()
    # Outside the default 14-day window
    result = classify_invalidation(
        decision_id="pr-3",
        settled_at=settled,
        was_human_settled=True,
        was_auto_handled=False,
        reverted_at=settled + timedelta(days=20),
    )
    assert result is None


def test_classify_revert_window_can_be_widened() -> None:
    settled = _settled()
    result = classify_invalidation(
        decision_id="pr-4",
        settled_at=settled,
        was_human_settled=True,
        was_auto_handled=False,
        reverted_at=settled + timedelta(days=20),
        revert_window_days=30,
    )
    assert result is not None
    assert result.signals == (INVALIDATION_REVERT_WITHIN_WINDOW,)


def test_classify_multiple_signals_preserved_in_order() -> None:
    settled = _settled()
    result = classify_invalidation(
        decision_id="pr-5",
        settled_at=settled,
        was_human_settled=True,
        was_auto_handled=False,
        reverted_at=settled + timedelta(days=2),
        incident_attributed=True,
        rolled_back=True,
        pr_reopened=True,
        human_override_redo_pr="pr-5-fix",
    )
    assert result is not None
    # Preservation order matches the function's defined order:
    # revert -> incident -> human_override_redo -> rollback -> reopened
    assert result.signals == (
        INVALIDATION_REVERT_WITHIN_WINDOW,
        INVALIDATION_POST_MERGE_INCIDENT,
        INVALIDATION_HUMAN_OVERRIDE_REDO,
        INVALIDATION_ROLLBACK,
        INVALIDATION_REOPENED_PR,
    )
    assert len(result.rationales) == 5
    assert "pr-5-fix" in result.rationales[2]


def test_classify_rejects_both_human_and_auto() -> None:
    with pytest.raises(ValueError, match="exactly one settlement source"):
        classify_invalidation(
            decision_id="pr-6",
            settled_at=_settled(),
            was_human_settled=True,
            was_auto_handled=True,
            reverted_at=_settled() + timedelta(days=1),
        )


def test_classify_rejects_neither_human_nor_auto() -> None:
    with pytest.raises(ValueError, match="exactly one settlement source"):
        classify_invalidation(
            decision_id="pr-6b",
            settled_at=_settled(),
            was_human_settled=False,
            was_auto_handled=False,
            reverted_at=_settled() + timedelta(days=1),
        )


def test_classify_negative_revert_window_rejected() -> None:
    with pytest.raises(ValueError, match="revert_window_days must be positive"):
        classify_invalidation(
            decision_id="pr-7",
            settled_at=_settled(),
            was_human_settled=True,
            was_auto_handled=False,
            revert_window_days=0,
        )


def test_classify_naive_timestamps_coerced_to_utc() -> None:
    # Naive datetime — function should normalise.
    settled_naive = datetime(2026, 4, 1, 12, 0)  # no tzinfo
    result = classify_invalidation(
        decision_id="pr-8",
        settled_at=settled_naive,
        was_human_settled=True,
        was_auto_handled=False,
        reverted_at=settled_naive + timedelta(days=1),
    )
    assert result is not None
    assert result.settled_at.tzinfo is UTC


def test_classify_revert_at_exactly_window_boundary() -> None:
    """Equality at the boundary counts as within-window (inclusive)."""
    settled = _settled()
    result = classify_invalidation(
        decision_id="pr-9",
        settled_at=settled,
        was_human_settled=True,
        was_auto_handled=False,
        reverted_at=settled + timedelta(days=DEFAULT_REVERT_WINDOW_DAYS),
    )
    assert result is not None


def test_invalidated_decision_signals_rationales_must_be_aligned() -> None:
    with pytest.raises(ValueError, match="same length"):
        InvalidatedDecision(
            decision_id="pr-x",
            settled_at=datetime.now(UTC),
            signals=(INVALIDATION_REVERT_WITHIN_WINDOW,),
            rationales=("a", "b"),
            was_human_settled=True,
            was_auto_handled=False,
        )


def test_invalidated_decision_rejects_unknown_signal_at_construction() -> None:
    with pytest.raises(ValueError, match="unknown invalidation signal"):
        InvalidatedDecision(
            decision_id="pr-x",
            settled_at=datetime.now(UTC),
            signals=("not_a_real_signal",),
            rationales=("anything",),
            was_human_settled=True,
            was_auto_handled=False,
        )


def test_invalidated_decision_requires_exactly_one_settlement_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        InvalidatedDecision(
            decision_id="pr-x",
            settled_at=datetime.now(UTC),
            signals=(INVALIDATION_REVERT_WITHIN_WINDOW,),
            rationales=("reverted",),
            was_human_settled=False,
            was_auto_handled=False,
        )


def test_invalidated_decision_to_dict_round_trip() -> None:
    settled = _settled()
    inv = classify_invalidation(
        decision_id="pr-roundtrip",
        settled_at=settled,
        was_human_settled=True,
        was_auto_handled=False,
        reverted_at=settled + timedelta(days=2),
    )
    assert inv is not None
    payload = inv.to_dict()
    assert payload["decision_id"] == "pr-roundtrip"
    assert payload["signals"] == [INVALIDATION_REVERT_WITHIN_WINDOW]
    assert isinstance(payload["rationales"], list)
    assert payload["was_human_settled"] is True
    assert payload["was_auto_handled"] is False


# ---------------------------------------------------------------------------
# compute_baseline tests
# ---------------------------------------------------------------------------


def _seed_invalidations(
    *,
    n_human_invalidated: int,
    n_auto_invalidated: int = 0,
    settled_at: datetime | None = None,
) -> list[InvalidatedDecision]:
    if settled_at is None:
        settled_at = datetime(2026, 4, 10, 12, 0, tzinfo=UTC)
    decisions: list[InvalidatedDecision] = []
    for i in range(n_human_invalidated):
        d = classify_invalidation(
            decision_id=f"human-{i}",
            settled_at=settled_at - timedelta(hours=i),
            was_human_settled=True,
            was_auto_handled=False,
            reverted_at=settled_at + timedelta(days=2),
        )
        assert d is not None
        decisions.append(d)
    for i in range(n_auto_invalidated):
        d = classify_invalidation(
            decision_id=f"auto-{i}",
            settled_at=settled_at - timedelta(hours=i),
            was_human_settled=False,
            was_auto_handled=True,
            reverted_at=settled_at + timedelta(days=1),
        )
        assert d is not None
        decisions.append(d)
    return decisions


def test_compute_baseline_below_min_samples_marks_unacceptable() -> None:
    invalidations = _seed_invalidations(n_human_invalidated=1)
    measurement = compute_baseline(
        invalidations,
        total_human_settled=10,
        total_auto_handled=0,
        window_end=datetime(2026, 4, 25, 12, 0, tzinfo=UTC),
    )
    assert measurement.sample_size_acceptable is False
    # Rate is still computed and surfaced for inspection
    assert measurement.baseline_human_rate == pytest.approx(0.1)
    # But the note is present so callers know not to use it
    assert "min_samples" in measurement.notes["baseline_human_rate"]


def test_compute_baseline_with_acceptable_sample() -> None:
    # 50 human-settled total, 5 invalidated → 10% baseline
    invalidations = _seed_invalidations(n_human_invalidated=5)
    measurement = compute_baseline(
        invalidations,
        total_human_settled=50,
        total_auto_handled=0,
        window_end=datetime(2026, 4, 25, 12, 0, tzinfo=UTC),
    )
    assert measurement.sample_size_acceptable is True
    assert measurement.baseline_human_rate == pytest.approx(0.1)
    # Wilson 95% CI bounds the rate
    assert (
        measurement.baseline_human_rate_ci_low is not None
        and measurement.baseline_human_rate_ci_high is not None
    )
    assert (
        measurement.baseline_human_rate_ci_low
        <= measurement.baseline_human_rate
        <= measurement.baseline_human_rate_ci_high
    )
    # CI is wider for small samples — sanity check
    assert measurement.baseline_human_rate_ci_high - measurement.baseline_human_rate_ci_low > 0.05


def test_compute_baseline_zero_human_settled_returns_none_rate() -> None:
    measurement = compute_baseline(
        [],
        total_human_settled=0,
        total_auto_handled=0,
        window_end=datetime(2026, 4, 25, 12, 0, tzinfo=UTC),
    )
    assert measurement.baseline_human_rate is None
    assert "no human-settled" in measurement.notes["baseline_human_rate"]


def test_compute_baseline_filters_out_of_window_decisions() -> None:
    way_old = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    in_window = datetime(2026, 4, 10, 12, 0, tzinfo=UTC)
    decisions = _seed_invalidations(
        n_human_invalidated=2, settled_at=in_window
    ) + _seed_invalidations(n_human_invalidated=10, settled_at=way_old)
    measurement = compute_baseline(
        decisions,
        total_human_settled=50,
        total_auto_handled=0,
        window_end=datetime(2026, 4, 25, 12, 0, tzinfo=UTC),
        window_days=30,
    )
    # Only 2 in-window invalidations counted
    assert measurement.invalidated_human_settled == 2
    assert measurement.baseline_human_rate == pytest.approx(0.04)


def test_compute_baseline_invalidations_exceeding_total_raises() -> None:
    invalidations = _seed_invalidations(n_human_invalidated=10)
    with pytest.raises(ValueError, match="exceed total human-settled"):
        compute_baseline(
            invalidations,
            total_human_settled=5,  # less than classified count
            total_auto_handled=0,
            window_end=datetime(2026, 4, 25, 12, 0, tzinfo=UTC),
        )


def test_compute_baseline_negative_total_rejected() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        compute_baseline(
            [],
            total_human_settled=-1,
            total_auto_handled=0,
            window_end=datetime(2026, 4, 25, 12, 0, tzinfo=UTC),
        )


def test_compute_baseline_rejects_zero_window_days() -> None:
    with pytest.raises(ValueError, match="window_days must be positive"):
        compute_baseline(
            [],
            total_human_settled=0,
            total_auto_handled=0,
            window_end=datetime(2026, 4, 25, 12, 0, tzinfo=UTC),
            window_days=0,
        )


def test_compute_baseline_to_dict_serialises_window_and_notes() -> None:
    invalidations = _seed_invalidations(n_human_invalidated=5)
    measurement = compute_baseline(
        invalidations,
        total_human_settled=100,
        total_auto_handled=10,
        window_end=datetime(2026, 4, 25, 12, 0, tzinfo=UTC),
    )
    payload = measurement.to_dict()
    assert payload["window_days"] == DEFAULT_BASELINE_WINDOW_DAYS
    assert payload["total_human_settled"] == 100
    assert payload["total_auto_handled"] == 10
    assert payload["sample_size_acceptable"] is True
    assert isinstance(payload["notes"], dict)
    assert payload["window_start"].endswith("+00:00")


def test_compute_baseline_dict_input_payloads_are_hydrated() -> None:
    invalidations = _seed_invalidations(n_human_invalidated=3)
    payloads = [d.to_dict() for d in invalidations]
    measurement = compute_baseline(
        payloads,
        total_human_settled=60,
        total_auto_handled=0,
        window_end=datetime(2026, 4, 25, 12, 0, tzinfo=UTC),
    )
    assert measurement.invalidated_human_settled == 3


# ---------------------------------------------------------------------------
# derive_threshold tests
# ---------------------------------------------------------------------------


def test_derive_threshold_below_min_samples_returns_placeholder() -> None:
    invalidations = _seed_invalidations(n_human_invalidated=2)
    measurement = compute_baseline(
        invalidations,
        total_human_settled=20,  # below min 50
        total_auto_handled=0,
        window_end=datetime(2026, 4, 25, 12, 0, tzinfo=UTC),
    )
    proposal = derive_threshold(measurement)
    assert proposal.is_placeholder is True
    assert proposal.threshold == pytest.approx(0.05)
    assert "placeholder" in proposal.rationale


def test_derive_threshold_above_min_samples_uses_safety_margin() -> None:
    # Construct a baseline of 20% (10 invalidated of 50)
    invalidations = _seed_invalidations(n_human_invalidated=10)
    measurement = compute_baseline(
        invalidations,
        total_human_settled=50,
        total_auto_handled=0,
        window_end=datetime(2026, 4, 25, 12, 0, tzinfo=UTC),
    )
    proposal = derive_threshold(measurement)
    assert proposal.is_placeholder is False
    # 0.20 * 0.5 = 0.10 (above floor of 0.01)
    assert proposal.threshold == pytest.approx(0.10)
    assert proposal.baseline == pytest.approx(0.20)
    assert proposal.safety_margin == pytest.approx(0.5)


def test_derive_threshold_floors_at_minimum_meaningful_rate() -> None:
    # baseline 0.01 * margin 0.5 = 0.005 → floored at 0.01
    invalidations = _seed_invalidations(n_human_invalidated=1)
    measurement = compute_baseline(
        invalidations,
        total_human_settled=100,  # 1% baseline, above min_samples
        total_auto_handled=0,
        window_end=datetime(2026, 4, 25, 12, 0, tzinfo=UTC),
    )
    proposal = derive_threshold(measurement)
    assert proposal.is_placeholder is False
    assert proposal.threshold == pytest.approx(DEFAULT_MINIMUM_MEANINGFUL_RATE)


def test_derive_threshold_custom_safety_margin() -> None:
    invalidations = _seed_invalidations(n_human_invalidated=10)
    measurement = compute_baseline(
        invalidations,
        total_human_settled=50,
        total_auto_handled=0,
        window_end=datetime(2026, 4, 25, 12, 0, tzinfo=UTC),
    )
    proposal = derive_threshold(measurement, safety_margin=0.25)
    # 0.20 * 0.25 = 0.05
    assert proposal.threshold == pytest.approx(0.05)
    assert proposal.safety_margin == pytest.approx(0.25)


def test_derive_threshold_custom_minimum_meaningful_rate() -> None:
    invalidations = _seed_invalidations(n_human_invalidated=1)
    measurement = compute_baseline(
        invalidations,
        total_human_settled=100,  # 1% baseline
        total_auto_handled=0,
        window_end=datetime(2026, 4, 25, 12, 0, tzinfo=UTC),
    )
    proposal = derive_threshold(measurement, minimum_meaningful_rate=0.005)
    # 0.01 * 0.5 = 0.005 == floor
    assert proposal.threshold == pytest.approx(0.005)


def test_derive_threshold_rejects_invalid_safety_margin() -> None:
    measurement = compute_baseline(
        [],
        total_human_settled=0,
        total_auto_handled=0,
        window_end=datetime(2026, 4, 25, 12, 0, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="safety_margin"):
        derive_threshold(measurement, safety_margin=0)
    with pytest.raises(ValueError, match="safety_margin"):
        derive_threshold(measurement, safety_margin=1.5)


def test_derive_threshold_rejects_invalid_placeholder_value() -> None:
    measurement = compute_baseline(
        [],
        total_human_settled=0,
        total_auto_handled=0,
        window_end=datetime(2026, 4, 25, 12, 0, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="placeholder_value"):
        derive_threshold(measurement, placeholder_value=0)
    with pytest.raises(ValueError, match="placeholder_value"):
        derive_threshold(measurement, placeholder_value=1.0)


def test_derive_threshold_to_dict_round_trip() -> None:
    invalidations = _seed_invalidations(n_human_invalidated=10)
    measurement = compute_baseline(
        invalidations,
        total_human_settled=50,
        total_auto_handled=0,
        window_end=datetime(2026, 4, 25, 12, 0, tzinfo=UTC),
    )
    proposal = derive_threshold(measurement)
    payload = proposal.to_dict()
    assert payload["is_placeholder"] is False
    assert payload["threshold"] == pytest.approx(0.10)
    assert payload["measurement_window_days"] == DEFAULT_BASELINE_WINDOW_DAYS
    assert payload["measured_at"].endswith("+00:00")


def test_derive_threshold_records_measurement_timestamp() -> None:
    measurement = compute_baseline(
        [],
        total_human_settled=0,
        total_auto_handled=0,
        window_end=datetime(2026, 4, 25, 12, 0, tzinfo=UTC),
    )
    pinned = datetime(2026, 4, 25, 8, 0, tzinfo=UTC)
    proposal = derive_threshold(measurement, measured_at=pinned)
    assert proposal.measured_at == pinned


# ---------------------------------------------------------------------------
# End-to-end seed → baseline → threshold (integration-style, still pure)
# ---------------------------------------------------------------------------


def test_full_pipeline_synthetic_human_settled_sample() -> None:
    """Seed 200 human-settled decisions, 14 invalidated → 7% baseline,
    threshold = max(0.07 * 0.5, 0.01) = 0.035 (3.5%)."""
    settled = datetime(2026, 4, 10, 12, 0, tzinfo=UTC)
    invalidations: list[InvalidatedDecision] = []
    for i in range(14):
        d = classify_invalidation(
            decision_id=f"pr-{i}",
            settled_at=settled - timedelta(hours=i),
            was_human_settled=True,
            was_auto_handled=False,
            reverted_at=settled + timedelta(days=3),
        )
        assert d is not None
        invalidations.append(d)

    measurement = compute_baseline(
        invalidations,
        total_human_settled=200,
        total_auto_handled=0,
        window_end=datetime(2026, 4, 25, 12, 0, tzinfo=UTC),
    )
    assert measurement.sample_size_acceptable is True
    assert measurement.baseline_human_rate == pytest.approx(0.07)

    proposal = derive_threshold(measurement)
    assert proposal.is_placeholder is False
    assert proposal.threshold == pytest.approx(0.035)
    assert "0.07" in proposal.rationale or "0.0700" in proposal.rationale


def test_full_pipeline_auto_handled_path_below_baseline() -> None:
    """Auto-handled path (5%) running below baseline (10%) is the
    expected steady state — auto-handle should be more conservative."""
    settled = datetime(2026, 4, 10, 12, 0, tzinfo=UTC)
    invalidations: list[InvalidatedDecision] = []
    for i in range(20):  # 20 / 200 = 10% baseline
        d = classify_invalidation(
            decision_id=f"hum-{i}",
            settled_at=settled - timedelta(hours=i),
            was_human_settled=True,
            was_auto_handled=False,
            reverted_at=settled + timedelta(days=2),
        )
        assert d is not None
        invalidations.append(d)
    for i in range(5):  # 5 / 100 = 5% auto-handled rate
        d = classify_invalidation(
            decision_id=f"auto-{i}",
            settled_at=settled - timedelta(hours=i),
            was_human_settled=False,
            was_auto_handled=True,
            reverted_at=settled + timedelta(days=2),
        )
        assert d is not None
        invalidations.append(d)

    measurement = compute_baseline(
        invalidations,
        total_human_settled=200,
        total_auto_handled=100,
        window_end=datetime(2026, 4, 25, 12, 0, tzinfo=UTC),
    )
    assert measurement.baseline_human_rate == pytest.approx(0.10)
    assert measurement.auto_handle_rate == pytest.approx(0.05)

    # Threshold is 5% (10% baseline * 0.5 safety margin)
    proposal = derive_threshold(measurement)
    assert proposal.threshold == pytest.approx(0.05)
    # Auto-handle rate currently at threshold — borderline, would alert
    # in a realistic monitor.
    assert measurement.auto_handle_rate == pytest.approx(proposal.threshold)


# --------------------------------------------------------------------------
# Step D: per-class invalidation numerator (gap #6608 step D)
# --------------------------------------------------------------------------


def test_invalidated_decision_decision_class_defaults_to_none():
    now = datetime.now(timezone.utc)
    decision = InvalidatedDecision(
        decision_id="d1",
        settled_at=now,
        signals=(INVALIDATION_REVERT_WITHIN_WINDOW,),
        rationales=("test",),
        was_human_settled=True,
        was_auto_handled=False,
    )
    assert decision.decision_class is None


def test_invalidated_decision_to_dict_includes_decision_class():
    now = datetime.now(timezone.utc)
    decision = InvalidatedDecision(
        decision_id="d1",
        settled_at=now,
        signals=(INVALIDATION_REVERT_WITHIN_WINDOW,),
        rationales=("test",),
        was_human_settled=True,
        was_auto_handled=False,
        decision_class="small-bugfix",
    )
    payload = decision.to_dict()
    assert payload["decision_class"] == "small-bugfix"


def test_invalidated_decision_to_dict_serialises_none_decision_class():
    now = datetime.now(timezone.utc)
    decision = InvalidatedDecision(
        decision_id="d1",
        settled_at=now,
        signals=(INVALIDATION_REVERT_WITHIN_WINDOW,),
        rationales=("test",),
        was_human_settled=True,
        was_auto_handled=False,
    )
    assert "decision_class" in decision.to_dict()
    assert decision.to_dict()["decision_class"] is None


def test_classify_invalidation_propagates_decision_class():
    now = datetime.now(timezone.utc)
    decision = classify_invalidation(
        decision_id="d1",
        settled_at=now,
        was_human_settled=True,
        was_auto_handled=False,
        reverted_at=now,
        decision_class="big-refactor",
    )
    assert decision is not None
    assert decision.decision_class == "big-refactor"


def test_classify_invalidation_decision_class_remains_none_by_default():
    now = datetime.now(timezone.utc)
    decision = classify_invalidation(
        decision_id="d1",
        settled_at=now,
        was_human_settled=True,
        was_auto_handled=False,
        reverted_at=now,
    )
    assert decision is not None
    assert decision.decision_class is None


def test_compute_baseline_populates_per_class_human_numerator_when_decisions_carry_class():
    now = datetime.now(timezone.utc)
    decisions = [
        InvalidatedDecision(
            "d1",
            now,
            (INVALIDATION_REVERT_WITHIN_WINDOW,),
            ("r",),
            True,
            False,
            decision_class="bugfix",
        ),
        InvalidatedDecision(
            "d2",
            now,
            (INVALIDATION_REVERT_WITHIN_WINDOW,),
            ("r",),
            True,
            False,
            decision_class="bugfix",
        ),
        InvalidatedDecision(
            "d3",
            now,
            (INVALIDATION_REVERT_WITHIN_WINDOW,),
            ("r",),
            True,
            False,
            decision_class="refactor",
        ),
    ]
    measurement = compute_baseline(
        decisions,
        total_human_settled=200,
        total_auto_handled=0,
        window_end=now,
        per_class_human={"bugfix": 100, "refactor": 60, "docs": 40},
    )
    assert measurement.per_class_human == {
        "bugfix": (2, 100),
        "refactor": (1, 60),
        "docs": (0, 40),
    }


def test_compute_baseline_populates_per_class_auto_numerator_when_decisions_carry_class():
    now = datetime.now(timezone.utc)
    decisions = [
        InvalidatedDecision(
            "a1",
            now,
            (INVALIDATION_ROLLBACK,),
            ("r",),
            False,
            True,
            decision_class="auto-merge",
        ),
        InvalidatedDecision(
            "a2",
            now,
            (INVALIDATION_ROLLBACK,),
            ("r",),
            False,
            True,
            decision_class="auto-merge",
        ),
    ]
    measurement = compute_baseline(
        decisions,
        total_human_settled=DEFAULT_MIN_BASELINE_SAMPLES,
        total_auto_handled=50,
        window_end=now,
        per_class_auto={"auto-merge": 30, "auto-decline": 20},
    )
    assert measurement.per_class_auto == {
        "auto-merge": (2, 30),
        "auto-decline": (0, 20),
    }


def test_compute_baseline_emits_coverage_note_when_invalidations_lack_decision_class():
    now = datetime.now(timezone.utc)
    decisions = [
        InvalidatedDecision(
            "typed",
            now,
            (INVALIDATION_REVERT_WITHIN_WINDOW,),
            ("r",),
            True,
            False,
            decision_class="bugfix",
        ),
        InvalidatedDecision(
            "untyped",
            now,
            (INVALIDATION_REVERT_WITHIN_WINDOW,),
            ("r",),
            True,
            False,
        ),
    ]
    measurement = compute_baseline(
        decisions,
        total_human_settled=200,
        total_auto_handled=0,
        window_end=now,
        per_class_human={"bugfix": 100},
    )
    assert "per_class_breakdown_coverage" in measurement.notes
    assert (
        "1 human + 0 auto invalidations carry no decision_class"
        in (measurement.notes["per_class_breakdown_coverage"])
    )


def test_compute_baseline_emits_coverage_note_for_unmatched_classes():
    now = datetime.now(timezone.utc)
    decisions = [
        InvalidatedDecision(
            "d1",
            now,
            (INVALIDATION_REVERT_WITHIN_WINDOW,),
            ("r",),
            True,
            False,
            decision_class="ghost-class",
        ),
    ]
    measurement = compute_baseline(
        decisions,
        total_human_settled=200,
        total_auto_handled=0,
        window_end=now,
        per_class_human={"bugfix": 100},  # ghost-class is not here
    )
    assert "per_class_breakdown_coverage" in measurement.notes
    assert (
        "absent from the per-class denominator inputs"
        in (measurement.notes["per_class_breakdown_coverage"])
    )


def test_compute_baseline_keeps_phase1_note_when_no_decisions_carry_class():
    now = datetime.now(timezone.utc)
    decisions = [
        InvalidatedDecision(
            "d1",
            now,
            (INVALIDATION_REVERT_WITHIN_WINDOW,),
            ("r",),
            True,
            False,
        ),
    ]
    measurement = compute_baseline(
        decisions,
        total_human_settled=200,
        total_auto_handled=0,
        window_end=now,
        per_class_human={"bugfix": 100},
    )
    # When no invalidation carries decision_class, the original phase-1
    # note still fires so callers know plumbing is incomplete on their side.
    assert measurement.notes.get("per_class_breakdown_numerator", "").startswith(
        "per-class denominators provided but no invalidation"
    )


def test_compute_baseline_no_per_class_inputs_skips_breakdown_entirely():
    now = datetime.now(timezone.utc)
    decisions = [
        InvalidatedDecision(
            "d1",
            now,
            (INVALIDATION_REVERT_WITHIN_WINDOW,),
            ("r",),
            True,
            False,
            decision_class="bugfix",
        ),
    ]
    measurement = compute_baseline(
        decisions,
        total_human_settled=200,
        total_auto_handled=0,
        window_end=now,
    )
    assert measurement.per_class_human == {}
    assert measurement.per_class_auto == {}
    assert "per_class_breakdown_numerator" not in measurement.notes
    assert "per_class_breakdown_coverage" not in measurement.notes


def test_decision_from_dict_round_trips_decision_class():
    now = datetime.now(timezone.utc)
    original = InvalidatedDecision(
        decision_id="d1",
        settled_at=now,
        signals=(INVALIDATION_REVERT_WITHIN_WINDOW,),
        rationales=("test",),
        was_human_settled=True,
        was_auto_handled=False,
        decision_class="big-refactor",
    )
    payload = original.to_dict()
    measurement = compute_baseline(
        [payload],
        total_human_settled=200,
        total_auto_handled=0,
        window_end=now,
        per_class_human={"big-refactor": 50},
    )
    assert measurement.per_class_human == {"big-refactor": (1, 50)}


def test_decision_from_dict_round_trips_missing_decision_class_field():
    now = datetime.now(timezone.utc)
    payload = {
        "decision_id": "d1",
        "settled_at": now.isoformat(),
        "signals": [INVALIDATION_REVERT_WITHIN_WINDOW],
        "rationales": ["test"],
        "was_human_settled": True,
        "was_auto_handled": False,
        # decision_class intentionally omitted to simulate legacy payload
    }
    measurement = compute_baseline(
        [payload],
        total_human_settled=200,
        total_auto_handled=0,
        window_end=now,
    )
    assert measurement.invalidated_human_settled == 1
