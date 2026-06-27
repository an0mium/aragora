"""Tests for `aragora metrics status` CLI verb (AGT-06 / #6067).

Verifies flag-gating, output content, file-write mode, and the --weeks param.
No live API calls; all I/O through tmp_path ShiftLedger fixtures.
"""

from __future__ import annotations

# ── Swarm package stub ───────────────────────────────────────────────────────
# MUST precede any aragora import; prevents aragora/swarm/__init__.py from
# running the heavy SwarmCommander→pydantic dependency chain.
import pathlib as _pathlib
import sys
import types as _types

if "aragora.swarm" not in sys.modules:
    _swarm_stub = _types.ModuleType("aragora.swarm")
    _swarm_stub.__path__ = [str(_pathlib.Path(__file__).parents[2] / "aragora" / "swarm")]
    _swarm_stub.__package__ = "aragora.swarm"
    sys.modules["aragora.swarm"] = _swarm_stub
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aragora.cli.commands.agt_metrics import cmd_metrics_status
from aragora.metrics.viah import VIAH_TREND_FLAG
from aragora.swarm.shift_ledger import ShiftLedger


def _ts(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _seed_ledger(path: Path, entries: list[dict]) -> ShiftLedger:
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, sort_keys=True) + "\n")
    return ShiftLedger(path=path)


def _args(tmp_path: Path, **kwargs) -> argparse.Namespace:
    ledger_path = tmp_path / "ledger.jsonl"
    if not ledger_path.exists():
        ledger_path.write_text("", encoding="utf-8")
    defaults: dict = {"ledger_path": str(ledger_path), "weeks": 4, "output": None}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Feature-gate tests
# ---------------------------------------------------------------------------


class TestFlagGating:
    def test_returns_nonzero_when_flag_off(
        self, tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(VIAH_TREND_FLAG, raising=False)
        rc = cmd_metrics_status(_args(tmp_path))
        assert rc != 0

    def test_prints_flag_name_to_stderr_when_off(
        self, tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(VIAH_TREND_FLAG, raising=False)
        cmd_metrics_status(_args(tmp_path))
        assert VIAH_TREND_FLAG in capsys.readouterr().err

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on"])
    def test_returns_zero_when_flag_set(
        self,
        val: str,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv(VIAH_TREND_FLAG, val)
        rc = cmd_metrics_status(_args(tmp_path))
        assert rc == 0


# ---------------------------------------------------------------------------
# Output content
# ---------------------------------------------------------------------------


class TestOutputContent:
    @pytest.fixture(autouse=True)
    def _enable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(VIAH_TREND_FLAG, "1")

    def test_report_contains_viah_heading(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        assert cmd_metrics_status(_args(tmp_path)) == 0
        assert "# VIAH Status" in capsys.readouterr().out

    def test_report_contains_summary_section(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        assert cmd_metrics_status(_args(tmp_path)) == 0
        assert "## Summary" in capsys.readouterr().out

    def test_report_shows_rolling_trend_section(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        assert cmd_metrics_status(_args(tmp_path, weeks=2)) == 0
        assert "Rolling Trend" in capsys.readouterr().out

    def test_empty_ledger_shows_na_for_viah(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        assert cmd_metrics_status(_args(tmp_path)) == 0
        assert "N/A (no agent-hours)" in capsys.readouterr().out

    def test_active_ledger_shows_numeric_viah(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        now = datetime.now(tz=UTC)
        ledger_path = tmp_path / "active.jsonl"
        _seed_ledger(
            ledger_path,
            [
                {
                    "entry_type": "shift_start",
                    "timestamp": _ts(now - timedelta(hours=4)),
                    "payload": {"shift_id": "s1"},
                },
                {
                    "entry_type": "shift_stop",
                    "timestamp": _ts(now - timedelta(minutes=10)),
                    "payload": {"shift_id": "s1"},
                },
                {
                    "entry_type": "pr_merged",
                    "timestamp": _ts(now - timedelta(hours=2)),
                    "payload": {"pr_number": 1},
                },
            ],
        )
        assert cmd_metrics_status(_args(tmp_path, ledger_path=str(ledger_path))) == 0
        out = capsys.readouterr().out
        assert "N/A (no agent-hours)" not in out


# ---------------------------------------------------------------------------
# File output mode
# ---------------------------------------------------------------------------


class TestFileOutput:
    @pytest.fixture(autouse=True)
    def _enable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(VIAH_TREND_FLAG, "1")

    def test_writes_markdown_to_path(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        out_path = tmp_path / "viah_status.md"
        assert cmd_metrics_status(_args(tmp_path, output=str(out_path))) == 0
        assert out_path.exists()
        assert "# VIAH Status" in out_path.read_text(encoding="utf-8")

    def test_stdout_shows_written_path(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        out_path = tmp_path / "viah_status.md"
        cmd_metrics_status(_args(tmp_path, output=str(out_path)))
        assert str(out_path) in capsys.readouterr().out

    def test_report_not_duplicated_to_stdout_when_writing_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        out_path = tmp_path / "viah_status.md"
        cmd_metrics_status(_args(tmp_path, output=str(out_path)))
        assert "# VIAH Status" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# --weeks parameter
# ---------------------------------------------------------------------------


class TestWeeksParam:
    @pytest.fixture(autouse=True)
    def _enable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(VIAH_TREND_FLAG, "1")

    def test_two_week_trend_shows_w0_and_w1(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        assert cmd_metrics_status(_args(tmp_path, weeks=2)) == 0
        out = capsys.readouterr().out
        assert "W0" in out
        assert "W1" in out

    def test_four_week_trend_shows_w3(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        assert cmd_metrics_status(_args(tmp_path, weeks=4)) == 0
        out = capsys.readouterr().out
        assert "W3" in out

    def test_default_weeks_is_four(self, tmp_path: Path) -> None:
        args = _args(tmp_path)
        assert args.weeks == 4
