"""Tests for aragora.metrics.viah_status — VIAH operator-truth report generator.

Pre-registers an aragora.swarm package stub so Python never runs the heavy
``aragora/swarm/__init__.py`` chain.  The real shift_ledger submodule
(stdlib-only imports) loads normally and tests use the genuine ShiftLedger.
"""

from __future__ import annotations

# ── Swarm package stub ───────────────────────────────────────────────────────
# MUST precede any aragora import; prevents aragora/swarm/__init__.py from running.
import pathlib as _pathlib
import sys
import types as _types

if "aragora.swarm" not in sys.modules:
    _swarm_mod = _types.ModuleType("aragora.swarm")
    _swarm_mod.__path__ = [str(_pathlib.Path(__file__).parents[2] / "aragora" / "swarm")]
    _swarm_mod.__package__ = "aragora.swarm"
    sys.modules["aragora.swarm"] = _swarm_mod
# ─────────────────────────────────────────────────────────────────────────────

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aragora.metrics.viah import VIAH_TREND_FLAG
from aragora.metrics.viah_status import generate_viah_status_report
from aragora.swarm.shift_ledger import ShiftLedger


def _ts(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ledger(tmp_path: Path, entries: list[dict]) -> ShiftLedger:
    path = tmp_path / "sl.jsonl"
    path.write_text("\n".join(json.dumps(e) for e in entries) + ("\n" if entries else ""))
    return ShiftLedger(path=path)


def _active_ledger(
    tmp_path: Path, *, now: datetime, shift_h: float = 4.0, prs: int = 2, rescues: int = 0
) -> ShiftLedger:
    sid = "s1"
    entries: list[dict] = [
        {
            "entry_type": "shift_start",
            "timestamp": _ts(now - timedelta(hours=shift_h)),
            "payload": {"shift_id": sid},
        },
        {
            "entry_type": "shift_stop",
            "timestamp": _ts(now - timedelta(minutes=5)),
            "payload": {"shift_id": sid},
        },
        *[
            {
                "entry_type": "pr_merged",
                "timestamp": _ts(now - timedelta(hours=shift_h - 1 - i)),
                "payload": {"pr": i},
            }
            for i in range(prs)
        ],
    ]
    if rescues:
        entries.append(
            {
                "entry_type": "cycle_tick",
                "timestamp": _ts(now - timedelta(hours=1)),
                "payload": {"rescue_count": rescues},
            }
        )
    return _ledger(tmp_path, entries)


# ---------------------------------------------------------------------------
# Feature gate
# ---------------------------------------------------------------------------


class TestFeatureGate:
    def test_raises_when_unset(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(VIAH_TREND_FLAG, raising=False)
        with pytest.raises(RuntimeError, match=VIAH_TREND_FLAG):
            generate_viah_status_report(_ledger(tmp_path, []))

    @pytest.mark.parametrize("val", ["0", "false", "no", "off", ""])
    def test_raises_for_falsy(
        self, val: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(VIAH_TREND_FLAG, val)
        with pytest.raises(RuntimeError):
            generate_viah_status_report(_ledger(tmp_path, []))

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE"])
    def test_passes_for_truthy(
        self, val: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(VIAH_TREND_FLAG, val)
        assert generate_viah_status_report(_ledger(tmp_path, []))


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


class TestReportStructure:
    @pytest.fixture(autouse=True)
    def _on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(VIAH_TREND_FLAG, "1")

    def _report(self, tmp_path: Path) -> str:
        return generate_viah_status_report(_ledger(tmp_path, []))

    def test_heading(self, tmp_path: Path) -> None:
        assert self._report(tmp_path).startswith("# VIAH Status")

    def test_required_sections(self, tmp_path: Path) -> None:
        r = self._report(tmp_path)
        for section in ("## Summary", "## Rolling Trend", "## Signal Breakdown", "## Caveats"):
            assert section in r

    def test_issue_and_flag_referenced(self, tmp_path: Path) -> None:
        r = self._report(tmp_path)
        assert "#6067" in r and VIAH_TREND_FLAG in r

    def test_agt05_and_pr7133_caveats(self, tmp_path: Path) -> None:
        r = self._report(tmp_path)
        assert "AGT-05" in r and "#7133" in r

    def test_trend_table_row_count(self, tmp_path: Path) -> None:
        now = datetime(2026, 5, 14, 12, 0, tzinfo=UTC)
        for weeks in (2, 4):
            report = generate_viah_status_report(_ledger(tmp_path, []), weeks=weeks, now=now)
            data_rows = [
                ln for ln in report.splitlines() if ln.startswith("| W") and ln[3:4].isdigit()
            ]
            assert len(data_rows) == weeks


# ---------------------------------------------------------------------------
# Empty ledger
# ---------------------------------------------------------------------------


class TestEmptyLedger:
    @pytest.fixture(autouse=True)
    def _on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(VIAH_TREND_FLAG, "1")

    def test_na_when_no_agent_hours(self, tmp_path: Path) -> None:
        assert "N/A" in generate_viah_status_report(_ledger(tmp_path, []))

    def test_zero_prs(self, tmp_path: Path) -> None:
        assert "Merged autonomous PRs (7 d):** 0" in generate_viah_status_report(
            _ledger(tmp_path, [])
        )

    def test_insufficient_data_trend(self, tmp_path: Path) -> None:
        now = datetime(2026, 5, 14, 12, 0, tzinfo=UTC)
        assert "insufficient_data" in generate_viah_status_report(_ledger(tmp_path, []), now=now)


# ---------------------------------------------------------------------------
# Ledger with activity
# ---------------------------------------------------------------------------


class TestLedgerWithActivity:
    @pytest.fixture(autouse=True)
    def _on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(VIAH_TREND_FLAG, "1")

    def test_pr_count_in_summary(self, tmp_path: Path) -> None:
        now = datetime(2026, 5, 14, 12, 0, tzinfo=UTC)
        assert "Merged autonomous PRs (7 d):** 3" in generate_viah_status_report(
            _active_ledger(tmp_path, now=now, prs=3), now=now
        )

    def test_rescues_in_summary(self, tmp_path: Path) -> None:
        now = datetime(2026, 5, 14, 12, 0, tzinfo=UTC)
        assert "Rescues required (7 d):** 2" in generate_viah_status_report(
            _active_ledger(tmp_path, now=now, prs=1, rescues=2), now=now
        )

    def test_nonzero_viah_when_prs_and_hours(self, tmp_path: Path) -> None:
        now = datetime(2026, 5, 14, 12, 0, tzinfo=UTC)
        summary = (
            generate_viah_status_report(
                _active_ledger(tmp_path, now=now, prs=2, shift_h=4.0), now=now
            )
            .split("## Summary")[1]
            .split("## ")[0]
        )
        assert "N/A" not in summary

    def test_now_controls_timestamp(self, tmp_path: Path) -> None:
        now_a = datetime(2026, 5, 14, 12, 0, tzinfo=UTC)
        now_b = datetime(2026, 5, 28, 12, 0, tzinfo=UTC)
        ledger = _active_ledger(tmp_path, now=now_a, prs=1, shift_h=2.0)
        assert "2026-05-14" in generate_viah_status_report(ledger, now=now_a)
        assert "2026-05-28" in generate_viah_status_report(ledger, now=now_b)

    def test_out_of_window_shows_zero_prs(self, tmp_path: Path) -> None:
        now = datetime(2026, 5, 14, 12, 0, tzinfo=UTC)
        ledger = _active_ledger(tmp_path, now=now, prs=3, shift_h=2.0)
        future = now + timedelta(weeks=4)
        assert "Merged autonomous PRs (7 d):** 0" in generate_viah_status_report(ledger, now=future)
