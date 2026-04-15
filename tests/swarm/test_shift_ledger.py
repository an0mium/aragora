"""Tests for the shift ledger (RS-10 runtime truth)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from aragora.swarm.shift_ledger import LedgerEntry, ShiftLedger


@pytest.fixture()
def ledger(tmp_path: Path) -> ShiftLedger:
    return ShiftLedger(path=tmp_path / "test_ledger.jsonl")


class TestLedgerEntry:
    def test_roundtrip(self) -> None:
        entry = LedgerEntry(
            entry_type="shift_start",
            timestamp="2026-04-15T12:00:00Z",
            payload={"shift_id": "s1", "max_hours": 12},
        )
        d = entry.to_dict()
        restored = LedgerEntry.from_dict(d)
        assert restored.entry_type == "shift_start"
        assert restored.payload["shift_id"] == "s1"
        assert restored.timestamp == "2026-04-15T12:00:00Z"

    def test_from_dict_defaults(self) -> None:
        entry = LedgerEntry.from_dict({})
        assert entry.entry_type == "unknown"
        assert entry.payload == {}


class TestShiftLedger:
    def test_append_and_read_all(self, ledger: ShiftLedger) -> None:
        ledger.append("shift_start", shift_id="s1")
        ledger.append("cycle_tick", queue_size=5)
        entries = ledger.read_all()
        assert len(entries) == 2
        assert entries[0].entry_type == "shift_start"
        assert entries[1].entry_type == "cycle_tick"
        assert entries[1].payload["queue_size"] == 5

    def test_read_empty_ledger(self, ledger: ShiftLedger) -> None:
        assert ledger.read_all() == []

    def test_read_by_type(self, ledger: ShiftLedger) -> None:
        ledger.append("cycle_tick", queue_size=3)
        ledger.append("pr_merged", pr_number=123)
        ledger.append("cycle_tick", queue_size=2)
        ticks = ledger.read_by_type("cycle_tick")
        assert len(ticks) == 2
        merges = ledger.read_by_type("pr_merged")
        assert len(merges) == 1

    def test_jsonl_format(self, ledger: ShiftLedger) -> None:
        ledger.append("shift_start", shift_id="s1")
        lines = ledger.path.read_text().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["entry_type"] == "shift_start"
        assert "timestamp" in parsed

    def test_malformed_lines_skipped(self, ledger: ShiftLedger) -> None:
        ledger.path.write_text("not json\n{}\n")
        entries = ledger.read_all()
        assert len(entries) == 1
        assert entries[0].entry_type == "unknown"

    def test_record_shift_lifecycle(self, ledger: ShiftLedger) -> None:
        ledger.record_shift_start(
            shift_id="s1", max_hours=12, benchmark_mode="hybrid", queue_size=5
        )
        ledger.record_cycle_tick(
            queue_size=5,
            open_prs=2,
            boss_running=True,
            merge_running=True,
            benchmark_fresh=True,
        )
        ledger.record_pr_merged(pr_number=100, title="fix something")
        ledger.record_service_restart(service="boss_loop", success=True)
        ledger.record_benchmark_run(run_id=999, conclusion="success")
        ledger.record_shift_stop(shift_id="s1", reason="completed", cycles=5, duration_seconds=3600)

        entries = ledger.read_all()
        assert len(entries) == 6
        types = [e.entry_type for e in entries]
        assert types == [
            "shift_start",
            "cycle_tick",
            "pr_merged",
            "service_restart",
            "benchmark_run",
            "shift_stop",
        ]

    def test_record_failure(self, ledger: ShiftLedger) -> None:
        ledger.record_failure(failure_type="auth_failure", detail="401 Unauthorized")
        entries = ledger.read_all()
        assert len(entries) == 1
        assert entries[0].entry_type == "auth_failure"
        assert entries[0].payload["detail"] == "401 Unauthorized"


class TestStatusSummary:
    def test_empty_summary(self, ledger: ShiftLedger) -> None:
        summary = ledger.get_status_summary()
        assert summary["total_entries"] == 0
        assert summary["prs_merged"] == 0
        assert summary["current_queue_size"] is None

    def test_populated_summary(self, ledger: ShiftLedger) -> None:
        ledger.record_shift_start(
            shift_id="s1", max_hours=12, benchmark_mode="hybrid", queue_size=5
        )
        ledger.record_cycle_tick(
            queue_size=3,
            open_prs=1,
            boss_running=True,
            merge_running=True,
            benchmark_fresh=True,
        )
        ledger.record_pr_merged(pr_number=100)
        ledger.record_pr_merged(pr_number=101)
        ledger.record_service_restart(service="boss_loop", success=True)
        ledger.record_service_restart(service="boss_loop", success=False, detail="timeout")
        ledger.record_benchmark_run(run_id=999, conclusion="success")
        ledger.record_failure(failure_type="auth_failure", detail="401")
        ledger.record_shift_stop(shift_id="s1", reason="completed", cycles=1, duration_seconds=3600)

        summary = ledger.get_status_summary()
        assert summary["shifts_started"] == 1
        assert summary["shifts_stopped"] == 1
        assert summary["last_stop_reason"] == "completed"
        assert summary["prs_merged"] == 2
        assert summary["pr_numbers_merged"] == [100, 101]
        assert summary["service_restarts"] == 2
        assert summary["restart_successes"] == 1
        assert summary["restart_failures"] == 1
        assert summary["auth_failures"] == 1
        assert summary["benchmark_runs"] == 1
        assert summary["last_benchmark_conclusion"] == "success"
        assert summary["current_queue_size"] == 3
        assert summary["current_boss_running"] is True
        assert summary["current_benchmark_fresh"] is True

    def test_multiple_ticks_uses_latest(self, ledger: ShiftLedger) -> None:
        ledger.record_cycle_tick(
            queue_size=10,
            open_prs=5,
            boss_running=False,
            merge_running=True,
            benchmark_fresh=False,
        )
        ledger.record_cycle_tick(
            queue_size=3,
            open_prs=1,
            boss_running=True,
            merge_running=True,
            benchmark_fresh=True,
        )
        summary = ledger.get_status_summary()
        assert summary["current_queue_size"] == 3
        assert summary["current_boss_running"] is True
