from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_module(script_name: str) -> Any:
    here = Path(__file__).resolve()
    script_path = here.parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_under_test", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


goal_state_ledger = _load_module("goal_state_ledger.py")


def _settle_report(
    *,
    status: str = "packet_authorized_dry_run",
    blockers: list[str] | None = None,
    suggested_commands: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "version": "settle-one-pr/v1",
        "generated_at": "2026-05-28T05:00:00Z",
        "dry_run": True,
        "selected_pr": 7496,
        "head_sha": "24dfdbaae7f64b3e225f2a74bbb3fe6a97127d3d",
        "status": status,
        "blockers": blockers or [],
        "suggested_commands": suggested_commands
        or [
            "gh pr merge 7496 --squash --match-head-commit 24dfdbaae7f64b3e225f2a74bbb3fe6a97127d3d"
        ],
        "policy_context": {
            "operator_snapshot_command": {
                "stdout": "large raw transcript-like command output should not persist"
            }
        },
        "recursive_best_next_prompt": "Start from live truth and merge only if gates pass.",
    }


def _runner_for(report: dict[str, Any]) -> goal_state_ledger.CommandRunner:
    def runner(
        command: list[str],
        *,
        cwd: Path,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        assert cwd
        assert timeout > 0
        assert command[:2] == ["python3", "scripts/settle_one_pr.py"]
        assert "--json" in command
        return subprocess.CompletedProcess(command, 0, json.dumps(report), "")

    return runner


def test_authorized_merge_goal_reports_one_dry_run_action_without_writing(
    tmp_path: Path,
) -> None:
    ledger_root = tmp_path / "goal-state"

    state = goal_state_ledger.build_goal_state(
        goal_id="merge_authorized_prs",
        cwd=tmp_path,
        ledger_root=ledger_root,
        runner=_runner_for(_settle_report()),
        write_ledger=False,
    )

    assert state["schema_version"] == "aragora-goal-state/0.1"
    assert state["goal_id"] == "merge_authorized_prs"
    assert state["dry_run"] is True
    assert state["write_ledger"] is False
    assert state["selected_pr"] == 7496
    assert state["head_sha"] == "24dfdbaae7f64b3e225f2a74bbb3fe6a97127d3d"
    assert state["next_action"]["kind"] == "merge_authorized_pr"
    assert state["next_action"]["safe_to_execute"] is False
    assert state["next_action"]["commands"] == [
        "gh pr merge 7496 --squash --match-head-commit 24dfdbaae7f64b3e225f2a74bbb3fe6a97127d3d"
    ]
    assert "policy_context" not in state["source_report"]
    assert "Start from live truth" in state["resume_prompt"]
    assert not ledger_root.exists()


def test_blocked_merge_goal_preserves_exact_blocker(tmp_path: Path) -> None:
    state = goal_state_ledger.build_goal_state(
        goal_id="merge_authorized_prs",
        cwd=tmp_path,
        ledger_root=tmp_path / "goal-state",
        runner=_runner_for(
            _settle_report(status="blocked", blockers=["required checks not green"])
        ),
        write_ledger=False,
    )

    assert state["next_action"]["kind"] == "blocked"
    assert state["next_action"]["blockers"] == ["required checks not green"]
    assert state["next_action"]["commands"] == []


def test_write_ledger_persists_latest_and_jsonl_receipt(tmp_path: Path) -> None:
    ledger_root = tmp_path / "goal-state"

    state = goal_state_ledger.build_goal_state(
        goal_id="merge_authorized_prs",
        cwd=tmp_path,
        ledger_root=ledger_root,
        runner=_runner_for(_settle_report(status="no_candidate", blockers=["no candidate"])),
        write_ledger=True,
    )

    latest_path = Path(state["state_path"])
    ledger_path = Path(state["receipt_ledger_path"])
    assert latest_path == ledger_root / "merge_authorized_prs" / "latest.json"
    assert ledger_path == ledger_root / "merge_authorized_prs" / "ledger.jsonl"
    assert latest_path.exists()
    assert ledger_path.exists()

    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    receipts = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines()]
    assert latest["goal_id"] == "merge_authorized_prs"
    assert receipts == [latest]


def test_unsupported_goal_fails_closed(tmp_path: Path) -> None:
    try:
        goal_state_ledger.build_goal_state(
            goal_id="new_autonomous_empire",
            cwd=tmp_path,
            ledger_root=tmp_path / "goal-state",
            runner=_runner_for(_settle_report()),
            write_ledger=False,
        )
    except ValueError as exc:
        assert "unsupported goal" in str(exc)
    else:
        raise AssertionError("unsupported goal should fail closed")
