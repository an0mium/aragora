"""Tests for the AGT-06 ``aragora metrics viah --persist`` CLI flag (SD-4 / #6067).

Verifies that:
- ``--persist`` without ``ARAGORA_VIAH_TREND_ENABLED`` returns 1 with an error message
- default (no ``--persist``) leaves the ledger unchanged
- ``--persist`` with the flag set writes one ``viah_snapshot`` entry
- the snapshot payload round-trips the key signal counts from the report

All tests are self-contained: they use a ``tmp_path`` ledger, inject the
environment flag via ``monkeypatch``, and invoke ``cmd_metrics_viah`` directly
(no subprocess, no network, no queue mutation).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from aragora.cli.commands.agt_metrics import cmd_metrics_viah
from aragora.metrics.viah import VIAH_TREND_FLAG, read_viah_snapshots
from aragora.swarm.shift_ledger import ShiftLedger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(
    ledger_path: str,
    *,
    persist: bool = False,
    window_hours: float = 24.0,
    as_json: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        ledger_path=ledger_path,
        window_hours=window_hours,
        cruxes_correctly_detected=0,
        predictions_above_brier_threshold=0,
        failed_claims_promoted_without_repair=0,
        json=as_json,
        persist=persist,
    )


def _seed_shift(path: Path) -> None:
    """Write minimal shift_start / pr_merged / shift_stop entries within the last 2 hours."""
    now = datetime.now(UTC)
    entries = [
        {
            "entry_type": "shift_start",
            "timestamp": (now - timedelta(hours=2)).isoformat(),
            "payload": {"shift_id": "s1"},
        },
        {
            "entry_type": "pr_merged",
            "timestamp": (now - timedelta(hours=1)).isoformat(),
            "payload": {},
        },
        {
            "entry_type": "shift_stop",
            "timestamp": now.isoformat(),
            "payload": {"shift_id": "s1"},
        },
    ]
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestViahPersistFlag:
    def test_persist_without_trend_flag_returns_error(self, tmp_path, monkeypatch, capsys) -> None:
        """--persist without ARAGORA_VIAH_TREND_ENABLED should return 1 and print to stderr."""
        monkeypatch.delenv(VIAH_TREND_FLAG, raising=False)
        ledger_path = tmp_path / "ledger.jsonl"
        ledger_path.write_text("", encoding="utf-8")

        rc = cmd_metrics_viah(_make_args(str(ledger_path), persist=True))

        assert rc == 1
        err = capsys.readouterr().err
        assert VIAH_TREND_FLAG in err
        assert "--persist" in err

    def test_no_persist_flag_leaves_ledger_unchanged(self, tmp_path, monkeypatch, capsys) -> None:
        """Omitting --persist must not write any snapshot, even with trend flag on."""
        monkeypatch.setenv(VIAH_TREND_FLAG, "1")
        ledger_path = tmp_path / "ledger.jsonl"
        ledger_path.write_text("", encoding="utf-8")

        rc = cmd_metrics_viah(_make_args(str(ledger_path), persist=False))

        assert rc == 0
        ledger = ShiftLedger(path=ledger_path)
        assert read_viah_snapshots(ledger=ledger) == []

    def test_persist_writes_one_snapshot_entry(self, tmp_path, monkeypatch, capsys) -> None:
        """--persist with flag set should write exactly one viah_snapshot entry."""
        monkeypatch.setenv(VIAH_TREND_FLAG, "1")
        ledger_path = tmp_path / "ledger.jsonl"
        _seed_shift(ledger_path)

        rc = cmd_metrics_viah(_make_args(str(ledger_path), persist=True))

        assert rc == 0
        ledger = ShiftLedger(path=ledger_path)
        snaps = read_viah_snapshots(ledger=ledger)
        assert len(snaps) == 1
        snap = snaps[0]
        assert snap["merged_autonomous_prs"] == 1
        assert "agent_hours" in snap
        assert "window_start" in snap
        assert "window_end" in snap

    def test_persist_snapshot_signal_counts_match_input(self, tmp_path, monkeypatch) -> None:
        """Persisted snapshot payload carries the same signal counts as the computed report."""
        monkeypatch.setenv(VIAH_TREND_FLAG, "1")
        ledger_path = tmp_path / "ledger.jsonl"
        _seed_shift(ledger_path)

        cmd_metrics_viah(_make_args(str(ledger_path), persist=True))

        ledger = ShiftLedger(path=ledger_path)
        snap = read_viah_snapshots(ledger=ledger)[-1]
        assert snap["rescues_required"] == 0
        assert snap["cruxes_correctly_detected"] == 0
        assert snap["predictions_above_brier_threshold"] == 0
        assert snap["failed_claims_promoted_without_repair"] == 0
        assert snap["merged_autonomous_prs"] >= 1
