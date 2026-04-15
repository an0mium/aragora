from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from aragora.cli.commands.swarm import cmd_swarm
from aragora.cli.commands.swarm_status import (
    _optional_float,
    _optional_int,
    load_operator_status,
    render_operator_status,
)
from aragora.swarm.shift_ledger import ShiftLedger


def _write_metrics(metrics_path: Path, rows: list[dict[str, object]]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _swarm_status_args(**overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "swarm_action_or_goal": "status",
        "swarm_goal": None,
        "swarm_campaign_target": None,
        "json": False,
        "run_id": None,
        "status_limit": 20,
        "findings_limit": 10,
        "refresh_scaling": False,
        "target_branch": "main",
        "boss_repo": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_load_operator_status_summarizes_metrics(tmp_path: Path) -> None:
    metrics_path = tmp_path / ".aragora" / "overnight" / "boss_metrics.jsonl"
    _write_metrics(
        metrics_path,
        [
            {
                "iteration": 1,
                "issue_number": 101,
                "worker_status": "completed",
                "terminal_class": "deliverable_pr_created",
                "elapsed_seconds": 12.0,
                "publish_action": "opened_pr",
                "sanitizer_outcome": None,
            },
            {
                "iteration": 2,
                "issue_number": 101,
                "worker_status": "needs_human",
                "terminal_class": "rescue_publish_deferred",
                "elapsed_seconds": 18.0,
                "publish_action": "deferred_due_to_open_boss_prs",
                "sanitizer_outcome": None,
            },
            {
                "iteration": 3,
                "issue_number": 202,
                "worker_status": "needs_human",
                "terminal_class": "blocked_scope_conflict",
                "elapsed_seconds": 9.0,
                "publish_action": None,
                "sanitizer_outcome": "quarantined",
            },
        ],
    )

    with patch("aragora.cli.commands.swarm_status._boss_ready_queue_depth", return_value=7):
        payload = load_operator_status(
            tmp_path, boss_repo="synaptent/aragora", metrics_path=metrics_path
        )

    assert payload["summary"]["unique_issues_attempted"] == 2
    assert payload["summary"]["unique_issues_completed"] == 1
    assert payload["summary"]["deferred_publish_count"] == 1
    assert payload["summary"]["sanitizer_rejection_count"] == 1
    assert payload["summary"]["queue_depth"] == 7
    assert payload["per_issue_success"][0]["issue_number"] == 101
    assert payload["per_issue_success"][0]["success_rate"] == 0.5


def test_numeric_coercion_helpers_accept_common_object_inputs() -> None:
    assert _optional_int(7) == 7
    assert _optional_int("9") == 9
    assert _optional_int(object()) is None

    assert _optional_float(7) == 7.0
    assert _optional_float("9.5") == 9.5
    assert _optional_float(object()) is None


def test_load_operator_status_reports_unknown_queue_depth_when_probe_unavailable(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / ".aragora" / "overnight" / "boss_metrics.jsonl"
    _write_metrics(metrics_path, [])

    with patch("aragora.cli.commands.swarm_status._boss_ready_queue_depth", return_value=None):
        payload = load_operator_status(tmp_path, metrics_path=metrics_path)

    assert payload["summary"]["queue_depth"] == "unknown"


def test_load_operator_status_prefers_ledger_truth_for_queue_depth(tmp_path: Path) -> None:
    metrics_path = tmp_path / ".aragora" / "overnight" / "boss_metrics.jsonl"
    _write_metrics(metrics_path, [])
    ledger = ShiftLedger(path=tmp_path / ".aragora" / "proof_first_shift" / "shift_ledger.jsonl")
    ledger.record_shift_start(
        shift_id="shift-1",
        max_hours=12.0,
        benchmark_mode="hybrid",
        queue_size=0,
    )
    ledger.record_cycle_tick(
        queue_size=2,
        open_prs=1,
        boss_running=False,
        merge_running=True,
        benchmark_fresh=True,
        actions=["steady_state"],
        stop_reason="completed",
    )
    ledger.record_pr_merged(pr_number=5857)
    ledger.record_shift_stop(
        shift_id="shift-1",
        reason="completed",
        cycles=1,
        duration_seconds=30.0,
    )

    with patch("aragora.cli.commands.swarm_status._boss_ready_queue_depth", return_value=9):
        payload = load_operator_status(tmp_path, metrics_path=metrics_path)

    assert payload["summary"]["queue_depth"] == 2
    assert payload["ledger_status"]["current_queue_size"] == 2
    assert payload["ledger_status"]["prs_merged"] == 1


def test_render_operator_status_includes_recent_iterations() -> None:
    text = render_operator_status(
        {
            "summary": {
                "unique_issues_attempted": 2,
                "unique_issues_completed": 1,
                "sanitizer_rejection_rate": 0.25,
                "deferred_publish_count": 1,
                "queue_depth": 4,
            },
            "ledger_status": {
                "current_queue_size": 0,
                "current_boss_running": False,
                "current_merge_running": True,
                "current_benchmark_fresh": True,
                "prs_merged": 3,
                "last_stop_reason": "completed",
                "green_shift": {"is_green": False},
            },
            "last_iterations": [
                {
                    "issue_number": 101,
                    "terminal_class": "deliverable_pr_created",
                    "elapsed_seconds": 12.0,
                    "worker_status": "completed",
                }
            ],
            "per_issue_success": [
                {
                    "issue_number": 101,
                    "completed_iterations": 1,
                    "attempts": 2,
                    "success_rate": 0.5,
                }
            ],
        }
    )

    assert "operator attempted=2 completed=1" in text
    assert "proof-first queue=0 boss=False merge=True benchmark_fresh=True" in text
    assert "recent iterations:" in text
    assert "#101 deliverable_pr_created" in text
    assert "per-issue success:" in text


def test_cmd_swarm_status_json_includes_operator_status(tmp_path: Path, capsys) -> None:
    metrics_path = tmp_path / ".aragora" / "overnight" / "boss_metrics.jsonl"
    _write_metrics(
        metrics_path,
        [
            {
                "iteration": 1,
                "issue_number": 101,
                "worker_status": "completed",
                "terminal_class": "deliverable_pr_created",
                "elapsed_seconds": 12.0,
                "publish_action": "opened_pr",
                "sanitizer_outcome": None,
            }
        ],
    )

    with (
        patch("aragora.worktree.fleet.resolve_repo_root", return_value=tmp_path),
        patch("aragora.worktree.fleet.build_fleet_rows", return_value=[]),
        patch("aragora.worktree.fleet.FleetCoordinationStore") as store_cls,
        patch(
            "aragora.swarm.reporter.build_integrator_view",
            return_value={"summary": {}, "next_actions": []},
        ),
        patch("aragora.swarm.SwarmSupervisor") as supervisor_cls,
        patch("aragora.cli.commands.swarm_status._boss_ready_queue_depth", return_value=5),
    ):
        store_cls.return_value.list_claims.return_value = []
        store_cls.return_value.list_merge_queue.return_value = []
        supervisor_cls.return_value.status_summary.return_value = {
            "runs": [],
            "counts": {
                "runs": 0,
                "queued_work_orders": 0,
                "leased_work_orders": 0,
                "completed_work_orders": 0,
            },
            "coordination": {"counts": {"active_leases": 0}},
        }
        cmd_swarm(_swarm_status_args(json=True, boss_repo="synaptent/aragora"))

    payload = json.loads(capsys.readouterr().out)
    assert payload["operator_status"]["summary"]["queue_depth"] == 5
    assert payload["operator_status"]["summary"]["unique_issues_attempted"] == 1


def test_cmd_swarm_status_text_includes_operator_metrics(tmp_path: Path, capsys) -> None:
    metrics_path = tmp_path / ".aragora" / "overnight" / "boss_metrics.jsonl"
    _write_metrics(
        metrics_path,
        [
            {
                "iteration": 1,
                "issue_number": 101,
                "worker_status": "completed",
                "terminal_class": "deliverable_pr_created",
                "elapsed_seconds": 12.0,
                "publish_action": "opened_pr",
                "sanitizer_outcome": None,
            }
        ],
    )

    with (
        patch("aragora.worktree.fleet.resolve_repo_root", return_value=tmp_path),
        patch("aragora.worktree.fleet.build_fleet_rows", return_value=[]),
        patch("aragora.worktree.fleet.FleetCoordinationStore") as store_cls,
        patch(
            "aragora.swarm.reporter.build_integrator_view",
            return_value={
                "summary": {
                    "ready_lanes": 0,
                    "review_lanes": 0,
                    "blocked_lanes": 0,
                    "stale_heartbeat_lanes": 0,
                    "collision_lanes": 0,
                    "missing_receipt_lanes": 0,
                    "superseded_lanes": 0,
                },
                "next_actions": [],
            },
        ),
        patch("aragora.swarm.SwarmSupervisor") as supervisor_cls,
        patch(
            "aragora.swarm.session_coordinator.read_directives",
            return_value={
                "summary": {
                    "directive_count": 0,
                    "session_count": 0,
                    "claim_count": 0,
                    "finding_count": 0,
                },
                "directives": [],
                "claims": [],
                "findings": [],
            },
        ),
        patch("aragora.cli.commands.swarm_status._boss_ready_queue_depth", return_value=3),
    ):
        store_cls.return_value.list_claims.return_value = []
        store_cls.return_value.list_merge_queue.return_value = []
        supervisor_cls.return_value.status_summary.return_value = {
            "runs": [],
            "counts": {
                "runs": 1,
                "queued_work_orders": 0,
                "leased_work_orders": 0,
                "completed_work_orders": 1,
            },
            "coordination": {"counts": {"active_leases": 0}},
        }
        cmd_swarm(_swarm_status_args())

    out = capsys.readouterr().out
    assert "runs=1 queued=0 leased=0 completed=1" in out
    assert "operator attempted=1 completed=1" in out
    assert "queue_depth=3" in out
