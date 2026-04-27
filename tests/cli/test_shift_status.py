from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from aragora.cli.commands.shift_status import (
    cmd_shift_status,
    load_shift_status,
    render_shift_status,
)
from aragora.swarm.live_shift_status import _count_live_queue_depth
from aragora.swarm.shift_ledger import ShiftLedger


def _seed_shift_ledger(repo_root: Path) -> ShiftLedger:
    ledger = ShiftLedger(path=repo_root / ".aragora" / "proof_first_shift" / "shift_ledger.jsonl")
    ledger.record_shift_start(
        shift_id="shift-1",
        max_hours=12.0,
        benchmark_mode="hybrid",
        queue_size=3,
    )
    ledger.record_cycle_tick(
        queue_size=2,
        open_prs=4,
        boss_running=False,
        merge_running=True,
        benchmark_fresh=True,
        actions=["steady_state"],
        stop_reason="completed",
    )
    ledger.record_pr_merged(pr_number=6105)
    ledger.record_shift_stop(
        shift_id="shift-1",
        reason="completed",
        cycles=1,
        duration_seconds=60.0,
    )
    return ledger


def _shift_status_args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "json": False,
        "shift_ledger": None,
        "max_age_hours": 48.0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_load_shift_status_reads_ledger_summary(tmp_path: Path) -> None:
    ledger = _seed_shift_ledger(tmp_path)

    payload = load_shift_status(tmp_path, max_age_hours=48.0)

    assert payload["available"] is True
    assert payload["ledger_path"] == str(ledger.path)
    assert payload["current_queue_size"] == 2
    assert payload["current_open_prs"] == 4
    assert payload["current_boss_running"] is False
    assert payload["current_merge_running"] is True
    assert payload["current_benchmark_fresh"] is True
    assert payload["prs_merged"] == 1
    assert payload["last_stop_reason"] == "completed"


def test_load_shift_status_reconciles_live_truth_when_repo_available(tmp_path: Path) -> None:
    _seed_shift_ledger(tmp_path)
    (tmp_path / ".git").mkdir()

    with (
        patch(
            "aragora.swarm.live_shift_status._infer_repo_name",
            return_value="synaptent/aragora",
        ),
        patch(
            "aragora.swarm.live_shift_status._detect_swarm_process",
            side_effect=[True, False],
        ),
        patch(
            "aragora.swarm.live_shift_status._count_live_queue_depth",
            return_value=7,
        ),
        patch(
            "aragora.swarm.live_shift_status._count_live_open_prs",
            return_value=9,
        ),
        patch(
            "aragora.swarm.live_shift_status._detect_observer_state",
            return_value={
                "observer_branch": "main",
                "observer_has_uncommitted_changes": True,
                "observer_behind_origin_main": 54,
                "observer_ahead_of_origin_main": 1,
                "observer_warning": "observer checkout is dirty checkout, 54 behind origin/main, 1 ahead of origin/main",
            },
        ),
    ):
        payload = load_shift_status(tmp_path, max_age_hours=48.0)

    assert payload["current_queue_size"] == 7
    assert payload["current_open_prs"] == 9
    assert payload["current_boss_running"] is True
    assert payload["current_merge_running"] is False
    assert payload["observer_branch"] == "main"
    assert payload["observer_has_uncommitted_changes"] is True
    assert payload["observer_behind_origin_main"] == 54
    assert payload["observer_ahead_of_origin_main"] == 1
    assert "observer checkout is dirty checkout" in payload["observer_warning"]
    assert payload["prs_merged"] == 1


def test_count_live_queue_depth_uses_canonical_boss_ready_queue_only(tmp_path: Path) -> None:
    commands: list[list[str]] = []

    def _fake_run(cmd: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout='[{"number": 101}]', stderr="")

    with (
        patch("aragora.swarm.live_shift_status.shutil.which", return_value="/usr/bin/gh"),
        patch("aragora.swarm.live_shift_status.subprocess.run", side_effect=_fake_run),
    ):
        assert _count_live_queue_depth(tmp_path, repo_name="synaptent/aragora") == 1

    assert commands == [
        [
            "/usr/bin/gh",
            "issue",
            "list",
            "--repo",
            "synaptent/aragora",
            "--label",
            "boss-ready",
            "--state",
            "open",
            "--limit",
            "500",
            "--json",
            "number",
        ]
    ]


def test_load_shift_status_keeps_ledger_truth_when_live_probe_unavailable(tmp_path: Path) -> None:
    _seed_shift_ledger(tmp_path)
    (tmp_path / ".git").mkdir()

    with (
        patch(
            "aragora.swarm.live_shift_status._infer_repo_name",
            return_value="synaptent/aragora",
        ),
        patch(
            "aragora.swarm.live_shift_status._detect_swarm_process",
            return_value=None,
        ),
        patch(
            "aragora.swarm.live_shift_status._count_live_queue_depth",
            return_value=None,
        ),
        patch(
            "aragora.swarm.live_shift_status._count_live_open_prs",
            return_value=None,
        ),
    ):
        payload = load_shift_status(tmp_path, max_age_hours=48.0)

    assert payload["current_queue_size"] == 2
    assert payload["current_open_prs"] == 4
    assert payload["current_boss_running"] is False
    assert payload["current_merge_running"] is True


def test_load_shift_status_reports_missing_ledger_without_creating_it(tmp_path: Path) -> None:
    ledger_path = tmp_path / ".aragora" / "proof_first_shift" / "shift_ledger.jsonl"

    payload = load_shift_status(tmp_path, max_age_hours=12.0)

    assert payload["available"] is False
    assert payload["ledger_path"] == str(ledger_path)
    assert payload["total_entries"] == 0
    assert payload["current_queue_size"] is None
    assert not ledger_path.exists()


def test_render_shift_status_includes_operator_summary(tmp_path: Path) -> None:
    _seed_shift_ledger(tmp_path)
    payload = load_shift_status(tmp_path, max_age_hours=48.0)
    payload.update(
        {
            "observer_branch": "codex/observer-truth-governance",
            "observer_has_uncommitted_changes": True,
            "observer_behind_origin_main": 54,
            "observer_ahead_of_origin_main": 1,
            "observer_warning": "observer checkout is dirty checkout, 54 behind origin/main, 1 ahead of origin/main",
        }
    )

    text = render_shift_status(payload)

    assert "proof-first shift ledger:" in text
    assert "available=True" in text
    assert "queue=2" in text
    assert "boss=False" in text
    assert "merge=True" in text
    assert "benchmark_fresh=True" in text
    assert "merged_prs=1" in text
    assert "last_stop=completed" in text
    assert (
        "observer=branch=codex/observer-truth-governance dirty=True behind_origin_main=54 ahead_of_origin_main=1"
        in text
    )
    assert (
        "observer_warning=observer checkout is dirty checkout, 54 behind origin/main, 1 ahead of origin/main"
        in text
    )


def test_cmd_shift_status_json_prints_ledger_summary(tmp_path: Path, capsys) -> None:
    _seed_shift_ledger(tmp_path)

    with patch("aragora.worktree.fleet.resolve_repo_root", return_value=tmp_path):
        cmd_shift_status(_shift_status_args(json=True))

    payload = json.loads(capsys.readouterr().out)
    assert payload["available"] is True
    assert payload["current_queue_size"] == 2
    assert payload["prs_merged"] == 1


def test_cmd_shift_status_text_prints_ledger_summary(tmp_path: Path, capsys) -> None:
    _seed_shift_ledger(tmp_path)

    with patch("aragora.worktree.fleet.resolve_repo_root", return_value=tmp_path):
        cmd_shift_status(_shift_status_args())

    out = capsys.readouterr().out
    assert "proof-first shift ledger:" in out
    assert "queue=2" in out
    assert "merged_prs=1" in out
