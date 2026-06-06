"""Tests for ``scripts/overnight_conductor.py``."""

from __future__ import annotations

import importlib.util
import json
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


conductor = _load_module("overnight_conductor.py")


def _runner(name: str, status: str) -> dict[str, str]:
    return {"name": name, "status": status}


def _pr(
    number: int,
    *,
    draft: bool,
    head: str = "abc123",
    branch: str | None = None,
    merge_state: str = "CLEAN",
) -> dict[str, Any]:
    return {
        "number": number,
        "title": f"PR {number}",
        "isDraft": draft,
        "headRefName": branch or f"codex/pr-{number}",
        "headRefOid": head,
        "mergeable": "MERGEABLE",
        "mergeStateStatus": merge_state,
        "url": f"https://github.com/synaptent/aragora/pull/{number}",
    }


def _green_summary() -> dict[str, Any]:
    return {"summary": conductor.check_summary([{"name": "Tests", "state": "SUCCESS"}])}


def _state(*, prs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    prs = prs or []
    return {
        "generated_at": "2026-06-05T00:00:00+00:00",
        "repo_root": "/repo",
        "runners": {
            "data": {
                "runners": [
                    _runner("mac-studio-m3ultra", "online"),
                    _runner("aragora-hetzner-cpu1", "online"),
                    _runner("aragora-hetzner-cpu2", "offline"),
                    _runner("aragora-hetzner-cpu3", "offline"),
                ]
            }
        },
        "operator_snapshot": {"data": {"lane_conflicts": [], "health": {"ok": True}}},
        "open_prs": {"data": prs},
        "checks_by_pr": {str(pr["number"]): _green_summary() for pr in prs},
        "publisher": {"data": {"verdict": "ready"}},
    }


def test_runner_unavailable_blocks_before_pr_selection() -> None:
    state = _state(prs=[_pr(7771, draft=True)])
    state["runners"]["data"] = {
        "runners": [
            _runner("mac-studio-m3ultra", "offline"),
            _runner("aragora-hetzner-cpu1", "offline"),
            _runner("aragora-hetzner-cpu2", "offline"),
        ]
    }

    action = conductor.select_action(state)

    assert action["kind"] == "blocker_report"
    assert action["reason"] == "runner fleet unavailable"
    assert "mac-studio-m3ultra is offline" in action["blockers"]
    assert "all aragora-hetzner-cpu* runners are offline" in action["blockers"]
    assert "Do not mutate source files" in action["prompt"]


def test_lane_conflict_blocks_before_pr_selection() -> None:
    state = _state(prs=[_pr(7771, draft=True)])
    state["operator_snapshot"]["data"]["lane_conflicts"] = [{"branch": "codex/pr-7771"}]

    action = conductor.select_action(state)

    assert action["kind"] == "blocker_report"
    assert action["reason"] == "active lane conflicts"


def test_selects_merge_ready_prompt_before_draft_gate() -> None:
    state = _state(prs=[_pr(7780, draft=False), _pr(7771, draft=True)])

    action = conductor.select_action(state)

    assert action["kind"] == "merge_ready_prompt"
    assert action["reason"] == "first mergeable non-draft PR has green required checks"
    assert action["target"]["pr"] == 7780
    assert "output the exact merge authorization prompt" in action["prompt"]
    assert "Do not merge without separate explicit operator authorization" in action["prompt"]


def test_selects_unstable_green_pr_before_stale_owner_coordination() -> None:
    state = _state(prs=[_pr(7780, draft=False, merge_state="UNSTABLE")])
    state["operator_snapshot"]["data"]["health"] = {
        "ok": False,
        "issues": [
            {
                "type": "lane_missing_steering_outcome",
                "session": "stale-owner",
                "detail": "missing steering outcome",
            }
        ],
    }

    action = conductor.select_action(state)

    assert action["kind"] == "merge_ready_prompt"
    assert action["target"]["pr"] == 7780


def test_dirty_pr_is_not_selected_for_merge_prompt() -> None:
    state = _state(prs=[_pr(7780, draft=False, merge_state="DIRTY")])

    action = conductor.select_action(state)

    assert action["kind"] == "blocker_report"
    assert action["reason"] == "no safe candidate"


def test_pr_check_probe_uses_required_checks_only() -> None:
    args = conductor.required_checks_args(7780)

    assert args[:4] == ["gh", "pr", "checks", "7780"]
    assert "--required" in args
    assert "--json" in args


def test_selects_draft_gate_preparation_for_green_draft_pr() -> None:
    state = _state(prs=[_pr(7771, draft=True, head="head7771")])

    action = conductor.select_action(state)

    assert action["kind"] == "draft_gate_preparation"
    assert action["target"]["pr"] == 7771
    assert action["target"]["head"] == "head7771"
    assert "output the exact ready authorization prompt" in action["prompt"]
    assert "Do not mark ready or merge" in action["prompt"]


def test_selects_real_failing_check_repair_prompt() -> None:
    pr = _pr(7772, draft=False)
    state = _state(prs=[pr])
    state["checks_by_pr"]["7772"] = {
        "summary": conductor.check_summary(
            [{"name": "Version Alignment", "state": "FAILURE", "bucket": "fail"}]
        )
    }

    action = conductor.select_action(state)

    assert action["kind"] == "failing_check_repair_prompt"
    assert action["failure"]["name"] == "Version Alignment"
    assert "repair only the exact-head real check failure" in action["prompt"]
    assert "Do not rerun CI, mark ready, or merge" in action["prompt"]


def test_stale_owner_health_issue_routes_coordination_after_other_candidates() -> None:
    state = _state(prs=[])
    state["operator_snapshot"]["data"]["health"] = {
        "ok": False,
        "issues": [
            {
                "type": "lane_missing_heartbeat",
                "session": "claude-7749-evidence",
                "detail": "missing heartbeat",
            }
        ],
    }

    action = conductor.select_action(state)

    assert action["kind"] == "stale_owner_coordination_prompt"
    assert action["target"]["session"] == "claude-7749-evidence"
    assert "Do not supersede" in action["prompt"]


def test_build_packet_preserves_forbidden_actions() -> None:
    packet = conductor.build_packet(_state(prs=[_pr(7771, draft=True)]))

    assert packet["action"]["kind"] == "draft_gate_preparation"
    assert "merge" in packet["forbidden_actions"]
    assert "mark_ready" in packet["forbidden_actions"]
    assert "record_tier4_settlement" in packet["forbidden_actions"]


def test_build_packet_tolerates_failed_operator_snapshot_probe() -> None:
    state = _state(prs=[_pr(7771, draft=True)])
    state["operator_snapshot"]["data"] = None

    packet = conductor.build_packet(state)

    assert packet["summary"]["operator_health_ok"] is None


def test_append_ledger_writes_jsonl(tmp_path: Path) -> None:
    ledger = tmp_path / "overnight" / "conductor.jsonl"
    packet = {"generated_at": "now", "action": {"kind": "blocker_report"}}

    conductor.append_ledger(ledger, packet)

    rows = ledger.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0]) == packet
