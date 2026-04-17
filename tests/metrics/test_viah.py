"""Tests for aragora.metrics.viah — VIAH score from ShiftLedger entries."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aragora.metrics.viah import (
    ViahCoefficients,
    ViahReport,
    compute_viah,
    viah_score,
)
from aragora.swarm.shift_ledger import ShiftLedger


def _ts(at: datetime) -> str:
    return at.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _seed_ledger(tmp_path: Path, entries: list[dict]) -> ShiftLedger:
    """Write a synthetic ledger to ``tmp_path`` and return a ShiftLedger over it."""
    path = tmp_path / "shift_ledger.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
    return ShiftLedger(path=path)


class TestViahScore:
    def test_returns_none_when_agent_hours_is_zero(self) -> None:
        assert (
            viah_score(
                merged_autonomous_prs=5,
                cruxes_correctly_detected=0,
                predictions_above_brier_threshold=0,
                rescues_required=0,
                failed_claims_promoted_without_repair=0,
                agent_hours=0,
            )
            is None
        )

    def test_returns_none_when_agent_hours_is_nan(self) -> None:
        assert (
            viah_score(
                merged_autonomous_prs=5,
                cruxes_correctly_detected=0,
                predictions_above_brier_threshold=0,
                rescues_required=0,
                failed_claims_promoted_without_repair=0,
                agent_hours=float("nan"),
            )
            is None
        )

    def test_default_coefficients_match_plan(self) -> None:
        # 4 PRs, 2 cruxes, 2 predictions, 1 rescue, 0 failed claims, 8 agent-hours
        # numerator = 4*1.0 + 2*0.5 + 2*0.5 - 1*0.5 - 0*1.0 = 5.5
        # viah = 5.5 / 8 = 0.6875
        score = viah_score(
            merged_autonomous_prs=4,
            cruxes_correctly_detected=2,
            predictions_above_brier_threshold=2,
            rescues_required=1,
            failed_claims_promoted_without_repair=0,
            agent_hours=8.0,
        )
        assert score == pytest.approx(0.6875)

    def test_score_can_go_negative_when_failures_dominate(self) -> None:
        # 1 PR, 0 cruxes/preds, 0 rescues, 5 failed claims, 4 agent-hours
        # numerator = 1 + 0 + 0 - 0 - 5 = -4
        # viah = -4 / 4 = -1.0
        score = viah_score(
            merged_autonomous_prs=1,
            cruxes_correctly_detected=0,
            predictions_above_brier_threshold=0,
            rescues_required=0,
            failed_claims_promoted_without_repair=5,
            agent_hours=4.0,
        )
        assert score == pytest.approx(-1.0)

    def test_custom_coefficients_apply(self) -> None:
        score = viah_score(
            merged_autonomous_prs=2,
            cruxes_correctly_detected=0,
            predictions_above_brier_threshold=0,
            rescues_required=0,
            failed_claims_promoted_without_repair=0,
            agent_hours=2.0,
            coefficients=ViahCoefficients(merged_pr_weight=2.0),
        )
        assert score == pytest.approx(2.0)


class TestComputeViahFromLedger:
    def test_empty_ledger_returns_zero_signals_and_none_viah(self, tmp_path: Path) -> None:
        ledger = _seed_ledger(tmp_path, [])
        report = compute_viah(
            ledger=ledger,
            window_hours=168.0,
            now=datetime(2026, 4, 17, tzinfo=UTC),
        )
        assert report.merged_autonomous_prs == 0
        assert report.rescues_required == 0
        assert report.agent_hours == 0.0
        assert report.viah is None

    def test_counts_pr_merged_within_window(self, tmp_path: Path) -> None:
        now = datetime(2026, 4, 17, 12, 0, tzinfo=UTC)
        in_window = now - timedelta(hours=24)
        out_of_window = now - timedelta(hours=200)
        ledger = _seed_ledger(
            tmp_path,
            [
                {
                    "entry_type": "pr_merged",
                    "timestamp": _ts(in_window),
                    "payload": {"pr_number": 1, "title": "fix: x"},
                },
                {
                    "entry_type": "pr_merged",
                    "timestamp": _ts(in_window + timedelta(minutes=30)),
                    "payload": {"pr_number": 2, "title": "fix: y"},
                },
                {
                    "entry_type": "pr_merged",
                    "timestamp": _ts(out_of_window),
                    "payload": {"pr_number": 3, "title": "older"},
                },
                {
                    "entry_type": "shift_start",
                    "timestamp": _ts(in_window - timedelta(hours=2)),
                    "payload": {"shift_id": "s1"},
                },
                {
                    "entry_type": "shift_stop",
                    "timestamp": _ts(in_window + timedelta(hours=6)),
                    "payload": {"shift_id": "s1"},
                },
            ],
        )
        report = compute_viah(ledger=ledger, window_hours=168.0, now=now)
        assert report.merged_autonomous_prs == 2  # 3rd PR is out of window

    def test_sums_rescue_counts_from_cycle_ticks(self, tmp_path: Path) -> None:
        now = datetime(2026, 4, 17, 12, 0, tzinfo=UTC)
        ledger = _seed_ledger(
            tmp_path,
            [
                {
                    "entry_type": "cycle_tick",
                    "timestamp": _ts(now - timedelta(hours=1)),
                    "payload": {"rescue_count": 2},
                },
                {
                    "entry_type": "cycle_tick",
                    "timestamp": _ts(now - timedelta(hours=2)),
                    "payload": {"rescue_count": 3},
                },
                {
                    "entry_type": "cycle_tick",
                    "timestamp": _ts(now - timedelta(hours=3)),
                    "payload": {},  # missing rescue_count → treated as 0
                },
                {
                    "entry_type": "shift_start",
                    "timestamp": _ts(now - timedelta(hours=4)),
                    "payload": {"shift_id": "s1"},
                },
                {
                    "entry_type": "shift_stop",
                    "timestamp": _ts(now),
                    "payload": {"shift_id": "s1"},
                },
            ],
        )
        report = compute_viah(ledger=ledger, window_hours=168.0, now=now)
        assert report.rescues_required == 5

    def test_agent_hours_sum_paired_shifts(self, tmp_path: Path) -> None:
        now = datetime(2026, 4, 17, 12, 0, tzinfo=UTC)
        ledger = _seed_ledger(
            tmp_path,
            [
                {
                    "entry_type": "shift_start",
                    "timestamp": _ts(now - timedelta(hours=10)),
                    "payload": {"shift_id": "s1"},
                },
                {
                    "entry_type": "shift_stop",
                    "timestamp": _ts(now - timedelta(hours=6)),
                    "payload": {"shift_id": "s1"},
                },
                {
                    "entry_type": "shift_start",
                    "timestamp": _ts(now - timedelta(hours=4)),
                    "payload": {"shift_id": "s2"},
                },
                {
                    "entry_type": "shift_stop",
                    "timestamp": _ts(now - timedelta(hours=1)),
                    "payload": {"shift_id": "s2"},
                },
            ],
        )
        report = compute_viah(ledger=ledger, window_hours=168.0, now=now)
        assert report.agent_hours == pytest.approx(7.0)  # 4h + 3h

    def test_in_progress_shift_contributes_partial_hours(self, tmp_path: Path) -> None:
        now = datetime(2026, 4, 17, 12, 0, tzinfo=UTC)
        ledger = _seed_ledger(
            tmp_path,
            [
                {
                    "entry_type": "shift_start",
                    "timestamp": _ts(now - timedelta(hours=3)),
                    "payload": {"shift_id": "s1"},
                },
                # No matching shift_stop — still in progress
            ],
        )
        report = compute_viah(ledger=ledger, window_hours=168.0, now=now)
        assert report.agent_hours == pytest.approx(3.0)

    def test_shift_intersected_with_window_only(self, tmp_path: Path) -> None:
        now = datetime(2026, 4, 17, 12, 0, tzinfo=UTC)
        # Window is last 4 hours; shift ran for 10 hours, ending mid-window
        ledger = _seed_ledger(
            tmp_path,
            [
                {
                    "entry_type": "shift_start",
                    "timestamp": _ts(now - timedelta(hours=10)),
                    "payload": {"shift_id": "s1"},
                },
                {
                    "entry_type": "shift_stop",
                    "timestamp": _ts(now - timedelta(hours=2)),
                    "payload": {"shift_id": "s1"},
                },
            ],
        )
        report = compute_viah(ledger=ledger, window_hours=4.0, now=now)
        # Window is [now-4h, now]; shift is [now-10h, now-2h];
        # intersection is [now-4h, now-2h] = 2h
        assert report.agent_hours == pytest.approx(2.0)

    def test_sidecar_signals_pass_through(self, tmp_path: Path) -> None:
        now = datetime(2026, 4, 17, 12, 0, tzinfo=UTC)
        ledger = _seed_ledger(
            tmp_path,
            [
                {
                    "entry_type": "shift_start",
                    "timestamp": _ts(now - timedelta(hours=2)),
                    "payload": {"shift_id": "s1"},
                },
                {
                    "entry_type": "shift_stop",
                    "timestamp": _ts(now),
                    "payload": {"shift_id": "s1"},
                },
                {
                    "entry_type": "pr_merged",
                    "timestamp": _ts(now - timedelta(minutes=30)),
                    "payload": {"pr_number": 1},
                },
            ],
        )
        report = compute_viah(
            ledger=ledger,
            window_hours=24.0,
            now=now,
            cruxes_correctly_detected=2,
            predictions_above_brier_threshold=3,
            failed_claims_promoted_without_repair=1,
        )
        # 1 PR + 2 cruxes*0.5 + 3 predictions*0.5 - 0 rescues - 1 failed*1.0
        # = 1 + 1.0 + 1.5 - 0 - 1.0 = 2.5
        # 2 agent-hours → VIAH = 1.25
        assert report.viah == pytest.approx(1.25)
        assert report.cruxes_correctly_detected == 2
        assert report.predictions_above_brier_threshold == 3
        assert report.failed_claims_promoted_without_repair == 1


class TestViahReportSerialization:
    def test_to_dict_includes_all_fields(self, tmp_path: Path) -> None:
        now = datetime(2026, 4, 17, 12, 0, tzinfo=UTC)
        ledger = _seed_ledger(
            tmp_path,
            [
                {
                    "entry_type": "shift_start",
                    "timestamp": _ts(now - timedelta(hours=1)),
                    "payload": {"shift_id": "s1"},
                },
                {
                    "entry_type": "shift_stop",
                    "timestamp": _ts(now),
                    "payload": {"shift_id": "s1"},
                },
            ],
        )
        report = compute_viah(ledger=ledger, window_hours=24.0, now=now)
        d = report.to_dict()
        for key in (
            "window_start",
            "window_end",
            "agent_hours",
            "merged_autonomous_prs",
            "rescues_required",
            "viah",
            "coefficients",
            "inputs",
        ):
            assert key in d
        assert d["coefficients"]["merged_pr_weight"] == 1.0
        assert isinstance(d["inputs"], dict)

    def test_window_start_end_are_iso8601_z(self, tmp_path: Path) -> None:
        ledger = _seed_ledger(tmp_path, [])
        report = compute_viah(
            ledger=ledger, window_hours=1.0, now=datetime(2026, 4, 17, tzinfo=UTC)
        )
        assert report.window_end.endswith("Z")
        assert report.window_start.endswith("Z")
