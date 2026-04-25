"""Tests for #6375 Step B threshold recalibration receipts."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aragora.review import (
    DEFAULT_THRESHOLD_RECEIPT_DIR,
    THRESHOLD_UPDATE_RECEIPT_SCHEMA_VERSION,
    INVALIDATION_REVERT_WITHIN_WINDOW,
    InvalidationRecalibrationSample,
    InvalidatedDecision,
    ThresholdRecalibrationScheduler,
    ThresholdUpdateReceipt,
    compute_threshold_update_receipt_id,
    write_threshold_update_receipt,
)

UTC = timezone.utc


def _invalidated_decision(
    decision_id: str,
    *,
    settled_at: datetime,
    human: bool = True,
) -> InvalidatedDecision:
    return InvalidatedDecision(
        decision_id=decision_id,
        settled_at=settled_at,
        signals=(INVALIDATION_REVERT_WITHIN_WINDOW,),
        rationales=("reverted during window",),
        was_human_settled=human,
        was_auto_handled=not human,
    )


def test_scheduler_emits_placeholder_receipt_below_sample_floor() -> None:
    now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    scheduler = ThresholdRecalibrationScheduler(min_samples=50)
    sample = InvalidationRecalibrationSample(
        invalidations=(_invalidated_decision("human-1", settled_at=now - timedelta(days=1)),),
        total_human_settled=10,
        total_auto_handled=2,
        source_name="fake-event-source",
        source_version="test",
    )

    receipt = scheduler.run_from_sample(sample, now=now)

    assert receipt.schema_version == THRESHOLD_UPDATE_RECEIPT_SCHEMA_VERSION
    assert receipt.sample_count == 10
    assert receipt.measurement.invalidated_human_settled == 1
    assert receipt.proposal.threshold == pytest.approx(0.05)
    assert receipt.proposal.is_placeholder is True
    assert receipt.threshold_delta is None
    assert receipt.baseline_human_rate_delta is None
    payload = receipt.to_dict()
    assert payload["sample_count"] == 10
    assert payload["baseline"]["human_rate"] == pytest.approx(0.1)
    assert payload["baseline"]["human_rate_ci"]["low"] is not None
    assert payload["threshold"]["derived"] == pytest.approx(0.05)
    assert payload["prior"] == {"threshold": None, "baseline_human_rate": None}


def test_scheduler_derives_threshold_and_deltas_from_previous_receipt() -> None:
    now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    scheduler = ThresholdRecalibrationScheduler(min_samples=50)
    previous = scheduler.run_from_sample(
        InvalidationRecalibrationSample(
            invalidations=(
                _invalidated_decision("prev-1", settled_at=now - timedelta(days=1)),
                _invalidated_decision("prev-2", settled_at=now - timedelta(days=2)),
            ),
            total_human_settled=100,
            total_auto_handled=0,
            source_name="fake-event-source",
        ),
        now=now - timedelta(days=1),
    )
    current = scheduler.run_from_sample(
        InvalidationRecalibrationSample(
            invalidations=(
                _invalidated_decision("curr-1", settled_at=now - timedelta(days=1)),
                _invalidated_decision("curr-2", settled_at=now - timedelta(days=2)),
                _invalidated_decision("curr-3", settled_at=now - timedelta(days=3)),
                _invalidated_decision("curr-4", settled_at=now - timedelta(days=4)),
            ),
            total_human_settled=100,
            total_auto_handled=0,
            source_name="fake-event-source",
        ),
        previous_receipt=previous,
        now=now,
    )

    assert previous.proposal.threshold == pytest.approx(0.01)
    assert current.measurement.baseline_human_rate == pytest.approx(0.04)
    assert current.proposal.threshold == pytest.approx(0.02)
    assert current.threshold_delta == pytest.approx(0.01)
    assert current.baseline_human_rate_delta == pytest.approx(0.02)
    payload = current.to_dict()
    assert payload["prior"]["threshold"] == pytest.approx(previous.proposal.threshold)
    assert payload["delta"]["threshold"] == pytest.approx(0.01)
    assert payload["delta"]["baseline_human_rate"] == pytest.approx(0.02)


def test_receipt_id_is_stable_and_excludes_receipt_id() -> None:
    now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    scheduler = ThresholdRecalibrationScheduler(min_samples=1)
    sample = InvalidationRecalibrationSample(
        invalidations=(_invalidated_decision("human-1", settled_at=now - timedelta(days=1)),),
        total_human_settled=10,
        total_auto_handled=0,
        source_name="fake-event-source",
    )

    receipt_a = scheduler.run_from_sample(sample, now=now)
    receipt_b = scheduler.run_from_sample(sample, now=now)

    assert receipt_a.receipt_id == receipt_b.receipt_id
    tampered = receipt_a.to_dict()
    tampered["receipt_id"] = "not-the-real-id"
    assert compute_threshold_update_receipt_id(tampered) == receipt_a.receipt_id


class _FakeEventSource:
    def __init__(self, sample: InvalidationRecalibrationSample) -> None:
        self.sample = sample
        self.calls: list[tuple[datetime, datetime]] = []

    def collect_recalibration_sample(
        self,
        *,
        window_start: datetime,
        window_end: datetime,
    ) -> InvalidationRecalibrationSample:
        self.calls.append((window_start, window_end))
        return self.sample


def test_run_from_source_passes_window_to_event_source() -> None:
    now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    scheduler = ThresholdRecalibrationScheduler(window_days=14)
    source = _FakeEventSource(
        InvalidationRecalibrationSample(
            total_human_settled=0,
            total_auto_handled=0,
            source_name="factory-step-a-stub",
        )
    )

    receipt = scheduler.run_from_source(source, now=now)

    assert source.calls == [(now - timedelta(days=14), now)]
    assert receipt.source_name == "factory-step-a-stub"
    assert receipt.proposal.is_placeholder is True


def test_write_threshold_update_receipt_uses_threshold_receipt_dir(tmp_path: Path) -> None:
    now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    scheduler = ThresholdRecalibrationScheduler(min_samples=1)
    receipt = scheduler.run_from_sample(
        InvalidationRecalibrationSample(total_human_settled=1, total_auto_handled=0),
        now=now,
    )

    path = write_threshold_update_receipt(receipt, repo_root=tmp_path)

    assert path.parent == tmp_path / DEFAULT_THRESHOLD_RECEIPT_DIR
    assert path.name == f"{receipt.receipt_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == receipt.to_dict()


def test_rejects_ambiguous_previous_threshold_inputs() -> None:
    now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    scheduler = ThresholdRecalibrationScheduler(min_samples=1)
    receipt = scheduler.run_from_sample(
        InvalidationRecalibrationSample(total_human_settled=1, total_auto_handled=0),
        now=now,
    )

    with pytest.raises(ValueError, match="pass either previous_receipt"):
        scheduler.run_from_sample(
            InvalidationRecalibrationSample(total_human_settled=1, total_auto_handled=0),
            previous_receipt=receipt,
            previous_threshold=0.01,
            now=now,
        )


def test_sample_freezes_mapping_fields() -> None:
    notes = {"source": "fake"}
    sample = InvalidationRecalibrationSample(notes=notes, per_class_human={"merge": 3})
    notes["source"] = "mutated"

    assert sample.notes["source"] == "fake"
    assert sample.per_class_human["merge"] == 3
    with pytest.raises(TypeError):
        sample.notes["source"] = "blocked"  # type: ignore[index]


def test_receipt_freezes_notes_mapping() -> None:
    now = datetime(2026, 4, 25, 12, 0, tzinfo=UTC)
    scheduler = ThresholdRecalibrationScheduler()
    receipt = scheduler.run_from_sample(
        InvalidationRecalibrationSample(notes={"mode": "dry-run"}),
        now=now,
    )

    assert receipt.notes["mode"] == "dry-run"
    with pytest.raises(TypeError):
        receipt.notes["mode"] = "mutated"  # type: ignore[index]
