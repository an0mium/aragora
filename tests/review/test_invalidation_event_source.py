"""Unit tests for aragora.review.invalidation_event_source (#6375 phase 2).

Covers the on-disk adapter that translates auto-handle calibration
rows + settlement receipts into :class:`InvalidatedDecision`. Uses a
``:memory:`` calibration store + a tmp-path receipts directory; no
real network, no real disk outside ``tmp_path``.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from aragora.review import (
    INVALIDATION_HUMAN_OVERRIDE_REDO,
    INVALIDATION_POST_MERGE_INCIDENT,
    INVALIDATION_REVERT_WITHIN_WINDOW,
    InvalidatedDecision,
)
from aragora.review.invalidation_event_source import (
    count_decisions_from_settlement_receipts,
    iter_invalidations_from_calibration_store,
    iter_invalidations_from_settlement_receipts,
    measure_baseline_from_stores,
    resolve_review_queue_root,
)
from aragora.triage.auto_handle_calibration import (
    AUTO_HANDLE_PATH_FIRE_AND_FORGET,
    OUTCOME_HUMAN_OVERRIDE,
    OUTCOME_INCIDENT,
    OUTCOME_REVERT,
    OUTCOME_SUCCESS,
    AutoHandleCalibrationStore,
)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Calibration-store side
# ---------------------------------------------------------------------------


def _seed_calibration_store(
    store: AutoHandleCalibrationStore,
    *,
    success: int = 0,
    revert: int = 0,
    incident: int = 0,
    human_override: int = 0,
    decision_class: str = "low_risk:scope=tests:size=S",
    pr_offset: int = 0,
) -> None:
    """Seed N decisions of each outcome class into the calibration store."""
    counter = pr_offset
    for outcome, n in (
        (OUTCOME_SUCCESS, success),
        (OUTCOME_REVERT, revert),
        (OUTCOME_INCIDENT, incident),
        (OUTCOME_HUMAN_OVERRIDE, human_override),
    ):
        for _ in range(n):
            counter += 1
            store.record_outcome(
                decision_id=f"d-{counter}-{outcome}",
                auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
                decision_class=decision_class,
                outcome=outcome,
                pr_number=counter,
            )


def test_iter_calibration_store_yields_only_failures() -> None:
    store = AutoHandleCalibrationStore(db_path=":memory:")
    _seed_calibration_store(store, success=5, revert=2, incident=1, human_override=1)
    invalidations = list(iter_invalidations_from_calibration_store(store))
    assert len(invalidations) == 4
    signals = sorted(d.signals[0] for d in invalidations)
    assert signals == sorted(
        [
            INVALIDATION_HUMAN_OVERRIDE_REDO,
            INVALIDATION_POST_MERGE_INCIDENT,
            INVALIDATION_REVERT_WITHIN_WINDOW,
            INVALIDATION_REVERT_WITHIN_WINDOW,
        ]
    )
    for d in invalidations:
        assert d.was_auto_handled is True
        assert d.was_human_settled is False
        assert len(d.rationales) == 1
        assert "calibration-store outcome" in d.rationales[0]


def test_iter_calibration_store_outcome_to_signal_mapping() -> None:
    """Each failure outcome maps to a distinct invalidation signal."""
    store = AutoHandleCalibrationStore(db_path=":memory:")
    _seed_calibration_store(store, revert=1, incident=1, human_override=1)
    invalidations = list(iter_invalidations_from_calibration_store(store))
    by_signal = {d.signals[0] for d in invalidations}
    assert by_signal == {
        INVALIDATION_REVERT_WITHIN_WINDOW,
        INVALIDATION_POST_MERGE_INCIDENT,
        INVALIDATION_HUMAN_OVERRIDE_REDO,
    }


def test_iter_calibration_store_empty_yields_nothing() -> None:
    store = AutoHandleCalibrationStore(db_path=":memory:")
    invalidations = list(iter_invalidations_from_calibration_store(store))
    assert invalidations == []


def test_iter_calibration_store_window_filter() -> None:
    """When a window is provided, rows outside it are excluded."""
    store = AutoHandleCalibrationStore(db_path=":memory:")
    _seed_calibration_store(store, revert=3)
    # Row decided_at is "now" (record_outcome calls time.time()), so a
    # window ending well in the past must yield 0 rows.
    long_ago = datetime(2000, 1, 1, tzinfo=UTC)
    invalidations = list(
        iter_invalidations_from_calibration_store(store, window_end=long_ago, window_days=30)
    )
    assert invalidations == []


def test_iter_calibration_store_window_includes_current_rows() -> None:
    store = AutoHandleCalibrationStore(db_path=":memory:")
    _seed_calibration_store(store, revert=2)
    # A window ending now should include the just-inserted rows
    now = datetime.now(UTC)
    invalidations = list(
        iter_invalidations_from_calibration_store(store, window_end=now, window_days=30)
    )
    assert len(invalidations) == 2


def test_iter_calibration_store_rejects_zero_window_days() -> None:
    store = AutoHandleCalibrationStore(db_path=":memory:")
    with pytest.raises(ValueError, match="window_days must be positive"):
        list(iter_invalidations_from_calibration_store(store, window_days=0))


# ---------------------------------------------------------------------------
# Settlement-receipt side
# ---------------------------------------------------------------------------


def _write_receipt(
    receipts_dir: Path,
    *,
    pr_number: int,
    action: str = "approve",
    reviewed_at: datetime | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a synthetic settlement-receipt JSON file."""
    if reviewed_at is None:
        reviewed_at = datetime.now(UTC)
    payload: dict[str, Any] = {
        "pr_number": pr_number,
        "action": action,
        "reviewed_at": reviewed_at.isoformat(),
        "session_id": f"sess-{pr_number}",
    }
    if extra:
        payload.update(extra)
    path = receipts_dir / f"pr-{pr_number}-sess-{pr_number}-{action}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_count_decisions_from_receipts_basic(tmp_path: Path) -> None:
    receipts = tmp_path / ".aragora" / "review-queue" / "receipts"
    receipts.mkdir(parents=True)
    now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    _write_receipt(receipts, pr_number=1, reviewed_at=now - timedelta(days=1))
    _write_receipt(receipts, pr_number=2, reviewed_at=now - timedelta(days=10))
    _write_receipt(receipts, pr_number=3, reviewed_at=now - timedelta(days=29))

    total = count_decisions_from_settlement_receipts(
        store_root=tmp_path / ".aragora" / "review-queue",
        window_end=now,
        window_days=30,
    )
    assert total == 3


def test_count_decisions_from_receipts_window_excludes_old(tmp_path: Path) -> None:
    receipts = tmp_path / ".aragora" / "review-queue" / "receipts"
    receipts.mkdir(parents=True)
    now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    _write_receipt(receipts, pr_number=1, reviewed_at=now - timedelta(days=1))
    _write_receipt(receipts, pr_number=2, reviewed_at=now - timedelta(days=100))

    total = count_decisions_from_settlement_receipts(
        store_root=tmp_path / ".aragora" / "review-queue",
        window_end=now,
        window_days=30,
    )
    assert total == 1


def test_count_decisions_from_receipts_handles_missing_dir(tmp_path: Path) -> None:
    # The receipts dir does not exist yet
    total = count_decisions_from_settlement_receipts(
        store_root=tmp_path / "does-not-exist",
        window_end=datetime.now(UTC),
        window_days=30,
    )
    assert total == 0


def test_count_decisions_skips_invalid_reviewed_at(tmp_path: Path) -> None:
    receipts = tmp_path / ".aragora" / "review-queue" / "receipts"
    receipts.mkdir(parents=True)
    now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    _write_receipt(receipts, pr_number=1, reviewed_at=now)
    bad_path = receipts / "pr-bad-sess-bad-approve.json"
    bad_path.write_text(
        json.dumps(
            {
                "pr_number": 99,
                "action": "approve",
                "reviewed_at": "not-a-real-iso-string",
                "session_id": "x",
            }
        ),
        encoding="utf-8",
    )

    total = count_decisions_from_settlement_receipts(
        store_root=tmp_path / ".aragora" / "review-queue",
        window_end=now,
        window_days=30,
    )
    assert total == 1


def test_count_decisions_skips_malformed_json(tmp_path: Path) -> None:
    receipts = tmp_path / ".aragora" / "review-queue" / "receipts"
    receipts.mkdir(parents=True)
    now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    _write_receipt(receipts, pr_number=1, reviewed_at=now)
    (receipts / "garbage.json").write_text("not json {", encoding="utf-8")

    total = count_decisions_from_settlement_receipts(
        store_root=tmp_path / ".aragora" / "review-queue",
        window_end=now,
        window_days=30,
    )
    assert total == 1


def test_count_decisions_rejects_zero_window_days() -> None:
    with pytest.raises(ValueError, match="window_days must be positive"):
        count_decisions_from_settlement_receipts(
            store_root=Path("/tmp"),
            window_end=datetime.now(UTC),
            window_days=0,
        )


def test_iter_settlement_receipts_yields_when_revert_recorded(tmp_path: Path) -> None:
    receipts = tmp_path / ".aragora" / "review-queue" / "receipts"
    receipts.mkdir(parents=True)
    settled = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
    _write_receipt(
        receipts,
        pr_number=42,
        reviewed_at=settled,
        extra={"reverted_at": (settled + timedelta(days=3)).isoformat()},
    )

    invalidations = list(
        iter_invalidations_from_settlement_receipts(
            store_root=tmp_path / ".aragora" / "review-queue"
        )
    )
    assert len(invalidations) == 1
    inv = invalidations[0]
    assert inv.was_human_settled is True
    assert inv.was_auto_handled is False
    assert INVALIDATION_REVERT_WITHIN_WINDOW in inv.signals


def test_iter_settlement_receipts_yields_when_incident_recorded(tmp_path: Path) -> None:
    receipts = tmp_path / ".aragora" / "review-queue" / "receipts"
    receipts.mkdir(parents=True)
    _write_receipt(
        receipts,
        pr_number=43,
        extra={"post_merge_incident": True},
    )
    invalidations = list(
        iter_invalidations_from_settlement_receipts(
            store_root=tmp_path / ".aragora" / "review-queue"
        )
    )
    assert len(invalidations) == 1
    assert invalidations[0].signals == (INVALIDATION_POST_MERGE_INCIDENT,)


def test_iter_settlement_receipts_yields_when_redo_pr_recorded(tmp_path: Path) -> None:
    receipts = tmp_path / ".aragora" / "review-queue" / "receipts"
    receipts.mkdir(parents=True)
    _write_receipt(
        receipts,
        pr_number=44,
        extra={"redo_pr": "synaptent/aragora#1234"},
    )
    invalidations = list(
        iter_invalidations_from_settlement_receipts(
            store_root=tmp_path / ".aragora" / "review-queue"
        )
    )
    assert len(invalidations) == 1
    inv = invalidations[0]
    assert inv.signals == (INVALIDATION_HUMAN_OVERRIDE_REDO,)
    assert "1234" in inv.rationales[0]


def test_iter_settlement_receipts_no_signals_yields_nothing(tmp_path: Path) -> None:
    """Plain receipts (no future-schema fields) yield no invalidations."""
    receipts = tmp_path / ".aragora" / "review-queue" / "receipts"
    receipts.mkdir(parents=True)
    for i in range(5):
        _write_receipt(receipts, pr_number=100 + i)
    invalidations = list(
        iter_invalidations_from_settlement_receipts(
            store_root=tmp_path / ".aragora" / "review-queue"
        )
    )
    assert invalidations == []


def test_iter_settlement_receipts_revert_outside_window_excluded(tmp_path: Path) -> None:
    receipts = tmp_path / ".aragora" / "review-queue" / "receipts"
    receipts.mkdir(parents=True)
    settled = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
    _write_receipt(
        receipts,
        pr_number=200,
        reviewed_at=settled,
        # 30 days post-settlement is outside the 14-day default window
        extra={"reverted_at": (settled + timedelta(days=30)).isoformat()},
    )
    invalidations = list(
        iter_invalidations_from_settlement_receipts(
            store_root=tmp_path / ".aragora" / "review-queue"
        )
    )
    assert invalidations == []


def test_iter_settlement_receipts_handles_missing_dir(tmp_path: Path) -> None:
    invalidations = list(
        iter_invalidations_from_settlement_receipts(store_root=tmp_path / "does-not-exist")
    )
    assert invalidations == []


# ---------------------------------------------------------------------------
# resolve_review_queue_root
# ---------------------------------------------------------------------------


def test_resolve_review_queue_root_explicit_override() -> None:
    out = resolve_review_queue_root("/explicit/path")
    assert out == Path("/explicit/path")


def test_resolve_review_queue_root_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARAGORA_REVIEW_QUEUE_ROOT", "/from/env")
    monkeypatch.delenv("ARAGORA_REVIEW_QUEUE_ROOT", raising=False)
    monkeypatch.setenv("ARAGORA_REVIEW_QUEUE_ROOT", "/from/env")
    out = resolve_review_queue_root()
    assert out == Path("/from/env")


def test_resolve_review_queue_root_walk_finds_aragora(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ARAGORA_REVIEW_QUEUE_ROOT", raising=False)
    (tmp_path / ".aragora").mkdir()
    nested = tmp_path / "nested" / "deeper"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    out = resolve_review_queue_root()
    assert out == tmp_path / ".aragora" / "review-queue"


# ---------------------------------------------------------------------------
# measure_baseline_from_stores — integration of both sides
# ---------------------------------------------------------------------------


def test_measure_baseline_combines_calibration_and_receipts(tmp_path: Path) -> None:
    # Auto-handle side: 50 success, 5 revert → 5/55 = 9.09%
    store = AutoHandleCalibrationStore(db_path=":memory:")
    _seed_calibration_store(store, success=50, revert=5)

    # Human-settled side: 60 receipts (denominator); 0 numerator
    # because plain receipts don't have invalidation fields
    receipts = tmp_path / ".aragora" / "review-queue" / "receipts"
    receipts.mkdir(parents=True)
    now = datetime.now(UTC)
    for i in range(60):
        _write_receipt(receipts, pr_number=i, reviewed_at=now - timedelta(days=1))

    measurement = measure_baseline_from_stores(
        calibration_store=store,
        review_queue_root=tmp_path / ".aragora" / "review-queue",
        window_end=now,
        window_days=30,
    )
    assert measurement.total_human_settled == 60
    # Auto-handle counts include success + failures
    assert measurement.total_auto_handled == 55
    assert measurement.invalidated_human_settled == 0
    assert measurement.invalidated_auto_handled == 5

    # Auto-handle rate is computed
    assert measurement.auto_handle_rate == pytest.approx(5 / 55)
    # Human baseline rate is 0 (no human invalidations yet) but sample
    # size is acceptable
    assert measurement.baseline_human_rate == pytest.approx(0.0)
    assert measurement.sample_size_acceptable is True

    # Note explicitly flags the schema gap so downstream consumers
    # don't conflate "0 invalidations" with "0 measured rate"
    assert "human_invalidations_source" in measurement.notes


def test_measure_baseline_below_min_samples_marks_unacceptable(tmp_path: Path) -> None:
    store = AutoHandleCalibrationStore(db_path=":memory:")
    _seed_calibration_store(store, success=10, revert=1)

    receipts = tmp_path / ".aragora" / "review-queue" / "receipts"
    receipts.mkdir(parents=True)
    now = datetime.now(UTC)
    for i in range(10):  # below default min 50
        _write_receipt(receipts, pr_number=i, reviewed_at=now)

    measurement = measure_baseline_from_stores(
        calibration_store=store,
        review_queue_root=tmp_path / ".aragora" / "review-queue",
        window_end=now,
        window_days=30,
    )
    assert measurement.sample_size_acceptable is False
    assert "sample_size_acceptable" in measurement.notes


def test_measure_baseline_no_data_returns_none_rates(tmp_path: Path) -> None:
    store = AutoHandleCalibrationStore(db_path=":memory:")
    measurement = measure_baseline_from_stores(
        calibration_store=store,
        review_queue_root=tmp_path / ".aragora" / "review-queue",
        window_end=datetime.now(UTC),
        window_days=30,
    )
    assert measurement.total_human_settled == 0
    assert measurement.total_auto_handled == 0
    assert measurement.baseline_human_rate is None
    assert measurement.auto_handle_rate is None


def test_measure_baseline_human_invalidation_picks_up_when_recorded(
    tmp_path: Path,
) -> None:
    """When receipts grow invalidation fields, the numerator activates
    without changing the public API."""
    store = AutoHandleCalibrationStore(db_path=":memory:")

    receipts = tmp_path / ".aragora" / "review-queue" / "receipts"
    receipts.mkdir(parents=True)
    now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    # 49 plain receipts + 3 with invalidation signals = 52 total
    for i in range(49):
        _write_receipt(receipts, pr_number=i, reviewed_at=now - timedelta(days=1))
    settled = now - timedelta(days=2)
    _write_receipt(
        receipts,
        pr_number=1001,
        reviewed_at=settled,
        extra={"reverted_at": (settled + timedelta(days=2)).isoformat()},
    )
    _write_receipt(receipts, pr_number=1002, reviewed_at=now, extra={"post_merge_incident": True})
    _write_receipt(receipts, pr_number=1003, reviewed_at=now, extra={"redo_pr": "x/y#9"})

    measurement = measure_baseline_from_stores(
        calibration_store=store,
        review_queue_root=tmp_path / ".aragora" / "review-queue",
        window_end=now,
        window_days=30,
    )
    assert measurement.total_human_settled == 52
    assert measurement.invalidated_human_settled == 3
    assert measurement.sample_size_acceptable is True
    # 3/52 ≈ 5.77%
    assert measurement.baseline_human_rate == pytest.approx(3 / 52)
    # When the numerator is non-zero, the schema-gap note must NOT
    # claim the source is missing
    assert "human_invalidations_source" not in measurement.notes


def test_measure_baseline_rejects_zero_window_days(tmp_path: Path) -> None:
    store = AutoHandleCalibrationStore(db_path=":memory:")
    with pytest.raises(ValueError, match="window_days must be positive"):
        measure_baseline_from_stores(
            calibration_store=store,
            review_queue_root=tmp_path,
            window_end=datetime.now(UTC),
            window_days=0,
        )


# ---------------------------------------------------------------------------
# Schema-stability sanity
# ---------------------------------------------------------------------------


def test_invalidated_decisions_from_calibration_round_trip_to_dict() -> None:
    store = AutoHandleCalibrationStore(db_path=":memory:")
    _seed_calibration_store(store, revert=1)
    invalidations = list(iter_invalidations_from_calibration_store(store))
    assert len(invalidations) == 1
    payload = invalidations[0].to_dict()
    assert payload["was_auto_handled"] is True
    assert payload["was_human_settled"] is False
    assert payload["signals"] == [INVALIDATION_REVERT_WITHIN_WINDOW]
    assert isinstance(payload["rationales"], list)


def test_invalidated_decisions_from_receipt_round_trip_to_dict(tmp_path: Path) -> None:
    receipts = tmp_path / ".aragora" / "review-queue" / "receipts"
    receipts.mkdir(parents=True)
    settled = datetime(2026, 4, 1, 12, 0, tzinfo=UTC)
    _write_receipt(
        receipts,
        pr_number=99,
        reviewed_at=settled,
        extra={"reverted_at": (settled + timedelta(days=2)).isoformat()},
    )
    invalidations = list(
        iter_invalidations_from_settlement_receipts(
            store_root=tmp_path / ".aragora" / "review-queue"
        )
    )
    payload = invalidations[0].to_dict()
    assert payload["was_human_settled"] is True
    assert payload["was_auto_handled"] is False
    assert payload["signals"] == [INVALIDATION_REVERT_WITHIN_WINDOW]
    assert "decision_id" in payload
    assert "settled_at" in payload


def test_validates_inputs_consistently_with_phase_1() -> None:
    """Spot-check that constructed InvalidatedDecision instances
    satisfy the phase-1 validation rules."""
    store = AutoHandleCalibrationStore(db_path=":memory:")
    _seed_calibration_store(store, revert=1, incident=1, human_override=1)
    for inv in iter_invalidations_from_calibration_store(store):
        # phase-1 InvalidatedDecision constructor enforces both:
        # (a) signals subset of INVALIDATION_SIGNALS
        # (b) signals/rationales same length
        assert isinstance(inv, InvalidatedDecision)
