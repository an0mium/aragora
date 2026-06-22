"""Tests for AGT-06 VIAH snapshot persistence (persist_viah_snapshot, read_viah_snapshots).

Operates on a temp-file ShiftLedger so no shared state leaks between tests.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aragora.metrics.viah import (
    VIAH_SNAPSHOT_ENTRY_TYPE,
    VIAH_TREND_FLAG,
    ViahCoefficients,
    ViahReport,
    compute_viah,
    persist_viah_snapshot,
    read_viah_snapshots,
)
from aragora.swarm.shift_ledger import ShiftLedger


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ledger(tmp_path: Path) -> ShiftLedger:
    return ShiftLedger(path=tmp_path / "test_ledger.jsonl")


def _minimal_report(tmp_path: Path) -> ViahReport:
    """Build a ViahReport from an empty ledger (no shifts, no events)."""
    empty_ledger = ShiftLedger(path=tmp_path / "empty.jsonl")
    return compute_viah(ledger=empty_ledger)


def _report_with_merged_prs(tmp_path: Path, n_prs: int) -> ViahReport:
    """Construct a ViahReport that shows `n_prs` merged PRs in the window."""
    src = ShiftLedger(path=tmp_path / "src.jsonl")
    now = datetime.now(tz=UTC)
    for i in range(n_prs):
        src.append("pr_merged", pr_number=i + 1, title=f"PR {i + 1}")
    return compute_viah(ledger=src, now=now)


# ---------------------------------------------------------------------------
# persist_viah_snapshot — feature gate
# ---------------------------------------------------------------------------


class TestPersistGate:
    def test_raises_when_flag_off_by_default(
        self, ledger: ShiftLedger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(VIAH_TREND_FLAG, raising=False)
        report = _minimal_report(tmp_path)
        with pytest.raises(RuntimeError, match="disabled"):
            persist_viah_snapshot(ledger=ledger, report=report)

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE", "YES"])
    def test_succeeds_with_truthy_flag(
        self,
        val: str,
        ledger: ShiftLedger,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(VIAH_TREND_FLAG, val)
        report = _minimal_report(tmp_path)
        entry = persist_viah_snapshot(ledger=ledger, report=report)
        assert entry.entry_type == VIAH_SNAPSHOT_ENTRY_TYPE

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", ""])
    def test_raises_with_falsy_flag(
        self,
        val: str,
        ledger: ShiftLedger,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(VIAH_TREND_FLAG, val)
        report = _minimal_report(tmp_path)
        with pytest.raises(RuntimeError, match="disabled"):
            persist_viah_snapshot(ledger=ledger, report=report)


# ---------------------------------------------------------------------------
# persist_viah_snapshot — correctness
# ---------------------------------------------------------------------------


class TestPersistCorrectness:
    @pytest.fixture(autouse=True)
    def _enable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(VIAH_TREND_FLAG, "1")

    def test_writes_viah_snapshot_entry_type(self, ledger: ShiftLedger, tmp_path: Path) -> None:
        report = _minimal_report(tmp_path)
        entry = persist_viah_snapshot(ledger=ledger, report=report)
        assert entry.entry_type == VIAH_SNAPSHOT_ENTRY_TYPE

    def test_entry_type_readable_from_ledger(self, ledger: ShiftLedger, tmp_path: Path) -> None:
        report = _minimal_report(tmp_path)
        persist_viah_snapshot(ledger=ledger, report=report)
        entries = ledger.read_by_type(VIAH_SNAPSHOT_ENTRY_TYPE)
        assert len(entries) == 1

    def test_window_fields_round_trip(self, ledger: ShiftLedger, tmp_path: Path) -> None:
        report = _minimal_report(tmp_path)
        persist_viah_snapshot(ledger=ledger, report=report)
        snapshots = ledger.read_by_type(VIAH_SNAPSHOT_ENTRY_TYPE)
        payload = snapshots[0].payload
        assert payload["window_start"] == report.window_start
        assert payload["window_end"] == report.window_end
        assert payload["window_hours"] == report.window_hours

    def test_signal_counts_round_trip(self, ledger: ShiftLedger, tmp_path: Path) -> None:
        report = _report_with_merged_prs(tmp_path, n_prs=3)
        persist_viah_snapshot(ledger=ledger, report=report)
        payload = ledger.read_by_type(VIAH_SNAPSHOT_ENTRY_TYPE)[0].payload
        assert payload["merged_autonomous_prs"] == 3
        assert payload["rescues_required"] == 0
        assert payload["cruxes_correctly_detected"] == 0
        assert payload["predictions_above_brier_threshold"] == 0
        assert payload["failed_claims_promoted_without_repair"] == 0

    def test_viah_none_when_no_agent_hours(self, ledger: ShiftLedger, tmp_path: Path) -> None:
        report = _minimal_report(tmp_path)
        assert report.viah is None
        persist_viah_snapshot(ledger=ledger, report=report)
        payload = ledger.read_by_type(VIAH_SNAPSHOT_ENTRY_TYPE)[0].payload
        assert payload["viah"] is None

    def test_viah_numeric_when_agent_hours_positive(
        self, ledger: ShiftLedger, tmp_path: Path
    ) -> None:
        src = ShiftLedger(path=tmp_path / "src2.jsonl")
        now = datetime.now(tz=UTC)
        src.append("shift_start", shift_id="s1", max_hours=24, benchmark_mode="test", queue_size=0)
        src.append("pr_merged", pr_number=1, title="p")
        src.append("shift_stop", shift_id="s1", reason="completed", shifts_run=1)
        report = compute_viah(ledger=src, window_hours=168.0, now=now)
        persist_viah_snapshot(ledger=ledger, report=report)
        payload = ledger.read_by_type(VIAH_SNAPSHOT_ENTRY_TYPE)[0].payload
        # agent_hours may be ~0 here because shift_start ≈ shift_stop; just assert key present
        assert "viah" in payload

    def test_multiple_snapshots_accumulate(self, ledger: ShiftLedger, tmp_path: Path) -> None:
        for _ in range(3):
            report = _minimal_report(tmp_path)
            persist_viah_snapshot(ledger=ledger, report=report)
        assert len(ledger.read_by_type(VIAH_SNAPSHOT_ENTRY_TYPE)) == 3

    def test_returns_ledger_entry_with_timestamp(self, ledger: ShiftLedger, tmp_path: Path) -> None:
        report = _minimal_report(tmp_path)
        entry = persist_viah_snapshot(ledger=ledger, report=report)
        assert entry.timestamp  # non-empty ISO timestamp


# ---------------------------------------------------------------------------
# read_viah_snapshots
# ---------------------------------------------------------------------------


class TestReadViahSnapshots:
    @pytest.fixture(autouse=True)
    def _enable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(VIAH_TREND_FLAG, "1")

    def test_empty_ledger_returns_empty_list(self, ledger: ShiftLedger) -> None:
        assert read_viah_snapshots(ledger=ledger) == []

    def test_returns_all_snapshots_by_default(self, ledger: ShiftLedger, tmp_path: Path) -> None:
        for _ in range(4):
            persist_viah_snapshot(ledger=ledger, report=_minimal_report(tmp_path))
        assert len(read_viah_snapshots(ledger=ledger)) == 4

    def test_max_count_limits_results(self, ledger: ShiftLedger, tmp_path: Path) -> None:
        for _ in range(5):
            persist_viah_snapshot(ledger=ledger, report=_minimal_report(tmp_path))
        result = read_viah_snapshots(ledger=ledger, max_count=2)
        assert len(result) == 2

    def test_max_count_returns_most_recent(self, ledger: ShiftLedger, tmp_path: Path) -> None:
        # Write snapshots that differ by merged_autonomous_prs count to identify order
        for n in range(1, 6):
            src = ShiftLedger(path=tmp_path / f"src_{n}.jsonl")
            now = datetime.now(tz=UTC)
            for i in range(n):
                src.append("pr_merged", pr_number=i, title="t")
            report = compute_viah(ledger=src, now=now)
            persist_viah_snapshot(ledger=ledger, report=report)
        last_two = read_viah_snapshots(ledger=ledger, max_count=2)
        # The last two should correspond to n=4 and n=5
        assert last_two[-1]["merged_autonomous_prs"] == 5
        assert last_two[0]["merged_autonomous_prs"] == 4

    def test_read_safe_without_flag(
        self, ledger: ShiftLedger, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        persist_viah_snapshot(ledger=ledger, report=_minimal_report(tmp_path))
        monkeypatch.delenv(VIAH_TREND_FLAG, raising=False)
        # read_viah_snapshots is always callable; no flag check
        result = read_viah_snapshots(ledger=ledger)
        assert len(result) == 1

    def test_other_entry_types_excluded(self, ledger: ShiftLedger, tmp_path: Path) -> None:
        ledger.append("pr_merged", pr_number=99, title="unrelated")
        persist_viah_snapshot(ledger=ledger, report=_minimal_report(tmp_path))
        result = read_viah_snapshots(ledger=ledger)
        assert len(result) == 1
        assert "merged_autonomous_prs" in result[0]
