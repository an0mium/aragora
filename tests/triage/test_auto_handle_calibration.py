"""Tests for auto-handle calibration and drift gating (#6372)."""

from __future__ import annotations

import json
from pathlib import Path

from aragora.triage.auto_handle_calibration import (
    AUTO_HANDLE_PATH_FIRE_AND_FORGET,
    AutoHandleCalibrationStore,
    OUTCOME_HUMAN_OVERRIDE,
    OUTCOME_SUCCESS,
    auto_handle_decision_id,
)


def _seed_successes(
    store: AutoHandleCalibrationStore,
    *,
    count: int,
    decision_class: str,
    repo_root: Path | None = None,
) -> None:
    for idx in range(count):
        store.record_outcome(
            decision_id=f"fire_and_forget:https://example.com/pr/{idx}",
            auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
            decision_class=decision_class,
            outcome=OUTCOME_SUCCESS,
            pr_url=f"https://example.com/pr/{idx}",
            repo_root=repo_root,
        )


def test_gate_rejects_below_threshold_classes() -> None:
    store = AutoHandleCalibrationStore(
        db_path=":memory:",
        min_samples=2,
        min_success_rate=0.80,
        drift_threshold=0.05,
    )
    decision_class = "tier=1|lanes=1|files=1|scope=aragora"
    _seed_successes(store, count=1, decision_class=decision_class)
    store.record_outcome(
        decision_id="fire_and_forget:https://example.com/pr/bad",
        auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
        decision_class=decision_class,
        outcome=OUTCOME_HUMAN_OVERRIDE,
        pr_url="https://example.com/pr/bad",
    )

    gate = store.evaluate_gate(
        auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
        decision_class=decision_class,
    )

    assert gate.allowed is False
    assert "drift gating" in gate.reason
    assert gate.summary.total_samples == 2
    assert gate.summary.failures == 1


def test_decision_ids_are_scoped_by_decision_class() -> None:
    decision_a = auto_handle_decision_id(
        auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
        pr_url="https://example.com/pr/1",
        decision_class="tier=1|lanes=1|files=1|scope=aragora",
    )
    decision_b = auto_handle_decision_id(
        auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
        pr_url="https://example.com/pr/1",
        decision_class="tier=1|lanes=1|files=2-3|scope=aragora+tests",
    )

    assert decision_a != decision_b


def test_drift_detector_emits_receipt_and_blocks_until_recovery(tmp_path: Path) -> None:
    store = AutoHandleCalibrationStore(
        db_path=":memory:",
        min_samples=2,
        min_success_rate=0.75,
        drift_threshold=0.10,
    )
    decision_class = "tier=1|lanes=1|files=2-3|scope=aragora+tests"
    _seed_successes(store, count=2, decision_class=decision_class, repo_root=tmp_path)

    result = store.record_outcome(
        decision_id="fire_and_forget:https://example.com/pr/regressed",
        auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
        decision_class=decision_class,
        outcome=OUTCOME_HUMAN_OVERRIDE,
        pr_url="https://example.com/pr/regressed",
        repo_root=tmp_path,
    )

    alert = result["alert"]
    assert isinstance(alert, dict)
    receipt_path = Path(str(alert["receipt_path"]))
    assert receipt_path.exists()
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["auto_handle_path"] == AUTO_HANDLE_PATH_FIRE_AND_FORGET
    assert payload["decision_class"] == decision_class

    blocked = store.evaluate_gate(
        auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
        decision_class=decision_class,
    )
    assert blocked.allowed is False
    assert blocked.active_drift_alert is True

    store.record_outcome(
        decision_id="fire_and_forget:https://example.com/pr/recovery",
        auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
        decision_class=decision_class,
        outcome=OUTCOME_SUCCESS,
        pr_url="https://example.com/pr/recovery",
        repo_root=tmp_path,
    )

    recovered = store.evaluate_gate(
        auto_handle_path=AUTO_HANDLE_PATH_FIRE_AND_FORGET,
        decision_class=decision_class,
    )
    assert recovered.allowed is True
    assert recovered.active_drift_alert is False
