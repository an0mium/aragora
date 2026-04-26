"""Pure-function tests for :mod:`aragora.triage.metrics`.

These tests exercise the metric aggregator with synthetic event
sequences so there is no dependency on filesystem layout or settlement
receipt format. They cover the Commitment-5 matrix named in
docs/THESIS.md and gap #6373:

  - empty events
  - all escalated
  - all auto-handled (forward-compatibility with #6372)
  - mixed population
  - sparse data suppression
  - drift detection
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from aragora.triage.metrics import (
    MIN_EVENTS_FOR_METRICS,
    TriageDecisionEvent,
    compute_window,
    detect_drift,
)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event(
    *,
    ts: datetime,
    decision_id: str = "",
    escalated: bool = False,
    auto_handled: bool = False,
    override: bool = False,
    recommendation: str = "approve_candidate",
    outcome: str | None = None,
    duration: float | None = None,
) -> TriageDecisionEvent:
    return TriageDecisionEvent(
        decision_id=decision_id or f"d-{ts.isoformat()}",
        ts=ts,
        was_escalated=escalated,
        was_auto_handled=auto_handled,
        was_human_override=override,
        ensemble_recommendation=recommendation,
        final_outcome=outcome,
        settlement_duration_seconds=duration,
    )


def _seed(count: int, *, window_end: datetime, spread_days: int = 5, **kwargs):
    """Return ``count`` events evenly spread over the final ``spread_days``."""
    step = timedelta(days=spread_days) / max(count, 1)
    return [
        _event(ts=window_end - step * (i + 1), decision_id=f"d-{i}", **kwargs) for i in range(count)
    ]


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------


class TestEmpty:
    def test_no_events_returns_none_rates(self):
        now = datetime.now(UTC)
        result = compute_window([], window_end=now, window_days=7)
        assert result.total_decisions == 0
        assert result.escalation_rate is None
        assert result.auto_handle_override_rate is None
        assert result.human_override_outcome_correlation is None
        assert result.settlement_duration_median_s is None
        assert result.settlement_duration_p95_s is None
        # Notes must explain every None so consumers aren't left guessing.
        for name in (
            "escalation_rate",
            "auto_handle_override_rate",
            "human_override_outcome_correlation",
            "settlement_duration_median_s",
            "settlement_duration_p95_s",
        ):
            assert name in result.notes

    def test_zero_window_days_raises(self):
        with pytest.raises(ValueError):
            compute_window([], window_end=datetime.now(UTC), window_days=0)

    def test_filters_events_outside_window(self):
        now = datetime.now(UTC)
        events = [
            _event(ts=now - timedelta(days=1)),
            _event(ts=now - timedelta(days=100)),  # excluded
        ]
        # Use a tiny min_events so we don't trigger sparse suppression.
        result = compute_window(events, window_end=now, window_days=7, min_events=1)
        assert result.total_decisions == 1


# ---------------------------------------------------------------------------
# Escalation rate
# ---------------------------------------------------------------------------


class TestEscalationRate:
    def test_all_escalated(self):
        now = datetime.now(UTC)
        events = _seed(
            MIN_EVENTS_FOR_METRICS,
            window_end=now,
            escalated=True,
            duration=60.0,
        )
        result = compute_window(events, window_end=now, window_days=7)
        assert result.total_decisions == MIN_EVENTS_FOR_METRICS
        assert result.escalation_rate == 1.0
        assert result.escalations == MIN_EVENTS_FOR_METRICS

    def test_none_escalated(self):
        now = datetime.now(UTC)
        events = _seed(MIN_EVENTS_FOR_METRICS, window_end=now, escalated=False)
        result = compute_window(events, window_end=now, window_days=7)
        assert result.escalation_rate == 0.0

    def test_half_escalated(self):
        now = datetime.now(UTC)
        escalated = [
            _event(
                ts=now - timedelta(hours=i + 1),
                decision_id=f"esc-{i}",
                escalated=True,
                duration=10.0,
            )
            for i in range(5)
        ]
        auto = [
            _event(
                ts=now - timedelta(hours=i + 20),
                decision_id=f"auto-{i}",
                escalated=False,
            )
            for i in range(5)
        ]
        result = compute_window(escalated + auto, window_end=now, window_days=7)
        assert result.total_decisions == 10
        assert result.escalation_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Auto-handle override rate
# ---------------------------------------------------------------------------


class TestAutoHandleOverrideRate:
    def test_all_auto_handled_none_overridden(self):
        now = datetime.now(UTC)
        events = _seed(
            MIN_EVENTS_FOR_METRICS,
            window_end=now,
            auto_handled=True,
            override=False,
        )
        result = compute_window(events, window_end=now, window_days=7)
        assert result.auto_handle_override_rate == 0.0

    def test_all_auto_handled_all_overridden(self):
        now = datetime.now(UTC)
        events = _seed(
            MIN_EVENTS_FOR_METRICS,
            window_end=now,
            auto_handled=True,
            override=True,
        )
        result = compute_window(events, window_end=now, window_days=7)
        assert result.auto_handle_override_rate == 1.0

    def test_no_auto_handled_returns_none_with_note(self):
        now = datetime.now(UTC)
        events = _seed(MIN_EVENTS_FOR_METRICS, window_end=now, auto_handled=False)
        result = compute_window(events, window_end=now, window_days=7)
        assert result.auto_handle_override_rate is None
        note = result.notes["auto_handle_override_rate"]
        assert "auto-handle" in note.lower()


# ---------------------------------------------------------------------------
# Human-override-outcome correlation
# ---------------------------------------------------------------------------


class TestOutcomeCorrelation:
    def test_no_outcome_data_returns_none(self):
        now = datetime.now(UTC)
        events = _seed(
            MIN_EVENTS_FOR_METRICS,
            window_end=now,
            override=True,
            recommendation="approve_candidate",
            outcome=None,
        )
        result = compute_window(events, window_end=now, window_days=7)
        assert result.human_override_outcome_correlation is None
        assert "outcome field" in result.notes["human_override_outcome_correlation"].lower()

    def test_all_overrides_confirm_ensemble(self):
        # Human said reject, ensemble said approve, outcome was NEGATIVE
        # → override AGAINST ensemble but outcome didn't confirm ensemble.
        # That means human was right; agreement with ensemble = 0.
        now = datetime.now(UTC)
        events = _seed(
            MIN_EVENTS_FOR_METRICS,
            window_end=now,
            override=True,
            recommendation="approve_candidate",
            outcome="closed",
        )
        result = compute_window(events, window_end=now, window_days=7)
        # recommendation=approve + outcome=closed → no agreement → -1
        assert result.human_override_outcome_correlation == -1.0

    def test_mixed_override_outcomes(self):
        now = datetime.now(UTC)
        agree = [
            _event(
                ts=now - timedelta(hours=i + 1),
                decision_id=f"a-{i}",
                override=True,
                recommendation="approve_candidate",
                outcome="merged",  # ensemble recommended approve; merged → confirms ensemble
            )
            for i in range(6)
        ]
        disagree = [
            _event(
                ts=now - timedelta(hours=i + 20),
                decision_id=f"d-{i}",
                override=True,
                recommendation="approve_candidate",
                outcome="closed",
            )
            for i in range(4)
        ]
        result = compute_window(agree + disagree, window_end=now, window_days=7)
        # (6 - 4) / 10 = 0.2
        assert result.human_override_outcome_correlation == pytest.approx(0.2)
        assert result.human_overrides_with_outcome == 10


# ---------------------------------------------------------------------------
# Settlement duration
# ---------------------------------------------------------------------------


class TestSettlementDuration:
    def test_median_and_p95(self):
        now = datetime.now(UTC)
        # 10 escalated events with durations 10..100
        durations = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        events = [
            _event(
                ts=now - timedelta(hours=i + 1),
                decision_id=f"e-{i}",
                escalated=True,
                duration=d,
            )
            for i, d in enumerate(durations)
        ]
        result = compute_window(events, window_end=now, window_days=7)
        # statistics.median of even-count uses the mean of two middle values
        assert result.settlement_duration_median_s == pytest.approx(55.0)
        # Linear-interpolated p95 of [10,20,...,100] at rank 0.95*9 = 8.55
        # -> ordered[8]*0.45 + ordered[9]*0.55 = 90*0.45 + 100*0.55 = 95.5
        assert result.settlement_duration_p95_s == pytest.approx(95.5)
        assert result.settlement_samples == 10

    def test_non_escalated_durations_ignored(self):
        now = datetime.now(UTC)
        # 10 events, only 2 escalated have durations.
        events = [
            _event(
                ts=now - timedelta(hours=1),
                decision_id="e-1",
                escalated=True,
                duration=50.0,
            ),
            _event(
                ts=now - timedelta(hours=2),
                decision_id="e-2",
                escalated=True,
                duration=150.0,
            ),
        ] + _seed(8, window_end=now, escalated=False, duration=999.0)
        result = compute_window(events, window_end=now, window_days=7)
        assert result.settlement_samples == 2
        assert result.settlement_duration_median_s == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Sparse-data policy
# ---------------------------------------------------------------------------


class TestSparseData:
    def test_below_min_events_all_metrics_suppressed(self):
        now = datetime.now(UTC)
        events = _seed(
            MIN_EVENTS_FOR_METRICS - 1,
            window_end=now,
            escalated=True,
            duration=10.0,
        )
        result = compute_window(events, window_end=now, window_days=7)
        assert result.total_decisions == MIN_EVENTS_FOR_METRICS - 1
        assert result.escalation_rate is None
        assert result.auto_handle_override_rate is None
        assert result.human_override_outcome_correlation is None
        assert result.settlement_duration_median_s is None
        assert result.settlement_duration_p95_s is None
        for name in (
            "escalation_rate",
            "auto_handle_override_rate",
            "human_override_outcome_correlation",
            "settlement_duration_median_s",
            "settlement_duration_p95_s",
        ):
            assert "insufficient data" in result.notes[name]

    def test_custom_min_events_override(self):
        now = datetime.now(UTC)
        events = _seed(3, window_end=now, escalated=True, duration=10.0)
        result = compute_window(events, window_end=now, window_days=7, min_events=2)
        # With min_events=2, 3 events is enough; rate should compute.
        assert result.escalation_rate == 1.0


# ---------------------------------------------------------------------------
# to_dict shape
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_to_dict_has_stable_keys(self):
        now = datetime.now(UTC)
        events = _seed(MIN_EVENTS_FOR_METRICS, window_end=now, escalated=True, duration=1.0)
        snapshot = compute_window(events, window_end=now, window_days=7)
        payload = snapshot.to_dict()
        assert set(payload.keys()) == {
            "window_label",
            "window_days",
            "window_start",
            "window_end",
            "total_decisions",
            "escalation_rate",
            "auto_handle_override_rate",
            "human_override_outcome_correlation",
            "settlement_duration_median_s",
            "settlement_duration_p95_s",
            "counts",
            "notes",
        }
        assert payload["window_label"] == "7d"
        assert payload["window_days"] == 7
        assert set(payload["counts"].keys()) == {
            "escalations",
            "auto_handled",
            "auto_handle_overrides",
            "human_overrides",
            "human_overrides_with_outcome",
            "settlement_samples",
        }


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------


class TestDriftDetection:
    def test_drift_flags_exceeding_threshold(self):
        now = datetime.now(UTC)
        old_events = _seed(MIN_EVENTS_FOR_METRICS, window_end=now, escalated=False)
        new_events = _seed(MIN_EVENTS_FOR_METRICS, window_end=now, escalated=True, duration=10.0)
        old = compute_window(old_events, window_end=now, window_days=7)
        new = compute_window(new_events, window_end=now, window_days=7)
        drift = detect_drift(new, old, threshold=0.10)
        esc = drift["escalation_rate"]
        assert esc["current"] == 1.0
        assert esc["previous"] == 0.0
        assert esc["delta"] == pytest.approx(1.0)
        assert esc["exceeded_threshold"] is True

    def test_drift_within_threshold(self):
        now = datetime.now(UTC)
        a = _seed(10, window_end=now, escalated=False)
        # Build a second window with 1/10 escalated == rate 0.1
        b = a[:-1] + [_event(ts=now - timedelta(hours=1), decision_id="x", escalated=True)]
        old = compute_window(a, window_end=now, window_days=7)
        new = compute_window(b, window_end=now, window_days=7)
        drift = detect_drift(new, old, threshold=0.2)
        assert drift["escalation_rate"]["exceeded_threshold"] is False

    def test_drift_with_none_inputs_not_flagged(self):
        now = datetime.now(UTC)
        empty = compute_window([], window_end=now, window_days=7)
        full = compute_window(
            _seed(MIN_EVENTS_FOR_METRICS, window_end=now, escalated=True, duration=10.0),
            window_end=now,
            window_days=7,
        )
        drift = detect_drift(full, empty)
        # Previous was None (empty window) → not flagged even if current is 1.0
        assert drift["escalation_rate"]["exceeded_threshold"] is False
        assert drift["escalation_rate"]["delta"] is None

    def test_drift_threshold_must_be_nonneg(self):
        now = datetime.now(UTC)
        snap = compute_window([], window_end=now, window_days=7)
        with pytest.raises(ValueError):
            detect_drift(snap, snap, threshold=-0.1)
