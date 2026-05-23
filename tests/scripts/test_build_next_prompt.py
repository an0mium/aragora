"""Tests for ``scripts/build_next_prompt.py``."""

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


prompt_builder = _load_module("build_next_prompt.py")


def test_prompt_starts_with_mailbox_and_owner_verification(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text(
        json.dumps(
            [
                {
                    "lane_id": "P106-merge-gate-settlement",
                    "owner_session": "droid-P106-merge-gate-settlement-20260521T2118Z",
                    "status": "working",
                    "pr_number": 7423,
                    "branch": "claude/recover-merge-gate-reconciliation",
                    "next_action": "settle exact-head governance gate",
                }
            ]
        ),
        encoding="utf-8",
    )

    prompt = prompt_builder.build_prompt(
        registry_path=registry,
        lane_id="P106-merge-gate-settlement",
        pr=7423,
    )

    assert prompt.startswith("Start from live repo truth")
    assert "Before lane work, check your Aragora operator-steering mailbox" in prompt
    assert (
        "python3 scripts/read_operator_steering.py --lane-id P106-merge-gate-settlement" in prompt
    )
    assert (
        "Continue only if you are owner_session droid-P106-merge-gate-settlement-20260521T2118Z"
        in prompt
    )
    assert (
        "If the prompt above accomplishes no incremental progress make the next prompt one that does"
        in prompt
    )


def test_prompt_for_non_owner_read_only_when_no_lane_match(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text("[]\n", encoding="utf-8")

    prompt = prompt_builder.build_prompt(registry_path=registry, pr=7407)

    assert "If you cannot map yourself to a lane, run read-only only" in prompt
    assert "Do not paste raw transcripts" in prompt


def test_prompt_shell_quotes_live_lane_values(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text(
        json.dumps(
            [
                {
                    "lane_id": "lane; echo pwned",
                    "owner_session": "codex-owner",
                    "status": "working",
                    "branch": "branch; echo pwned",
                    "pr_number": 7425,
                }
            ]
        ),
        encoding="utf-8",
    )

    prompt = prompt_builder.build_prompt(registry_path=registry, pr=7425)

    assert "--lane-id 'lane; echo pwned'" in prompt
    assert "--lane-id lane; echo pwned" not in prompt


def test_decision_packet_redacts_transcript_fields_and_captures_pr_truth(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text("[]\n", encoding="utf-8")

    def fake_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        joined = " ".join(command)
        if command[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(
                command, 0, "## main...origin/main\n M dirty.py\n", ""
            )
        if "operator-snapshot" in joined:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "health": {"ok": True},
                        "process_census": {"records": []},
                        "diagnostic": "transcript file not found",
                        "body": "ordinary PR body text",
                    }
                ),
                "",
            )
        if "list_active_agent_sessions.py" in joined:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "sessions": [
                            {
                                "id": "codex-secret",
                                "transcript_path": "/secret/transcript.jsonl",
                                "prompt": "raw prompt text",
                            }
                        ]
                    }
                ),
                "",
            )
        if command[:3] == ["gh", "pr", "view"]:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "number": 7425,
                        "headRefOid": "91172e10a3",
                        "state": "OPEN",
                        "isDraft": True,
                        "mergeStateStatus": "CLEAN",
                    }
                ),
                "",
            )
        if command[:3] == ["gh", "pr", "checks"]:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps([{"name": "lint", "state": "SUCCESS"}]),
                "",
            )
        if "merge-packet" in joined:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps({"admin_squash_allowed": False, "not_ready": [7425]}),
                "",
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    packet = prompt_builder.build_decision_packet(
        registry_path=registry,
        pr=7425,
        command_runner=fake_runner,
    )

    assert packet["root"]["dirty"] is True
    assert packet["pr"]["headRefOid"] == "91172e10a3"
    assert packet["checks"]["required"][0]["name"] == "lint"
    assert packet["merge_packet"]["not_ready"] == [7425]
    serialized = json.dumps(packet)
    assert "transcript_path" not in serialized
    assert "raw prompt text" not in serialized
    assert "/secret/transcript.jsonl" not in serialized
    assert "transcript file not found" in serialized
    assert "ordinary PR body text" in serialized


def test_decision_packet_reports_active_owner_blocker(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text(
        json.dumps(
            [
                {
                    "lane_id": "Q50-harden-7425-control-plane",
                    "owner_session": "codex-owner",
                    "status": "working",
                    "pr_number": 7425,
                }
            ]
        ),
        encoding="utf-8",
    )

    packet = prompt_builder.build_decision_packet(
        registry_path=registry,
        pr=7425,
        command_runner=lambda command: subprocess.CompletedProcess(command, 0, "{}", ""),
    )

    assert packet["owner"]["owner_session"] == "codex-owner"
    assert "active owner exists for target" in packet["blockers"]


def _settlement_runner(
    *,
    live_head: str = "live-head",
    packet_head: str = "live-head",
    packet_ready: bool = True,
    include_packet: bool = True,
    include_packet_entry: bool = True,
    pending_checks: bool = False,
) -> Any:
    def fake_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        joined = " ".join(command)
        if command[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(command, 0, "## main...origin/main\n", "")
        if command[:3] == ["gh", "pr", "view"]:
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "number": 7435,
                        "headRefOid": live_head,
                        "state": "OPEN",
                        "isDraft": False,
                        "mergeStateStatus": "CLEAN",
                    }
                ),
                "",
            )
        if command[:3] == ["gh", "pr", "checks"]:
            checks = [
                {
                    "name": "aragora-merge-quorum",
                    "workflow": "Aragora Merge Quorum",
                    "state": "IN_PROGRESS" if pending_checks else "SUCCESS",
                    "bucket": "pending" if pending_checks else "pass",
                }
            ]
            return subprocess.CompletedProcess(command, 0, json.dumps(checks), "")
        if "merge-packet" in joined:
            if not include_packet:
                return subprocess.CompletedProcess(command, 0, "{}", "")
            entries = [
                {
                    "pr_number": 7435,
                    "head_sha": packet_head,
                    "admin_squash_allowed": packet_ready,
                }
            ]
            if not include_packet_entry:
                entries = []
            return subprocess.CompletedProcess(
                command,
                0,
                json.dumps(
                    {
                        "entries": entries,
                        "not_ready": [] if packet_ready else [7435],
                    }
                ),
                "",
            )
        return subprocess.CompletedProcess(command, 0, "{}", "")

    return fake_runner


def test_settlement_guard_fails_closed_on_expected_head_drift(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text("[]\n", encoding="utf-8")

    packet = prompt_builder.build_decision_packet(
        registry_path=registry,
        pr=7435,
        expected_head="old-head",
        command_runner=_settlement_runner(live_head="live-head"),
    )

    guard = packet["settlement_guard"]
    assert guard["verdict"] == "fail_closed"
    assert "expected head old-head does not match live head live-head" in guard["reasons"]


def test_settlement_guard_fails_closed_on_duplicate_active_lanes(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text(
        json.dumps(
            [
                {
                    "lane_id": "one",
                    "owner_session": "owner-one",
                    "status": "active",
                    "pr_number": 7435,
                    "branch": "same",
                },
                {
                    "lane_id": "two",
                    "owner_session": "owner-two",
                    "status": "blocked",
                    "pr_number": 7435,
                    "branch": "same",
                },
            ]
        ),
        encoding="utf-8",
    )

    packet = prompt_builder.build_decision_packet(
        registry_path=registry,
        pr=7435,
        expected_head="live-head",
        command_runner=_settlement_runner(),
    )

    guard = packet["settlement_guard"]
    assert guard["verdict"] == "fail_closed"
    assert "multiple active owners exist for target" in packet["blockers"]
    assert any("multiple active target owners" in reason for reason in guard["reasons"])


def test_settlement_guard_fails_closed_on_stale_merge_packet_head(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text("[]\n", encoding="utf-8")

    packet = prompt_builder.build_decision_packet(
        registry_path=registry,
        pr=7435,
        expected_head="live-head",
        command_runner=_settlement_runner(live_head="live-head", packet_head="old-head"),
    )

    guard = packet["settlement_guard"]
    assert guard["verdict"] == "fail_closed"
    assert "merge-packet head old-head does not match live head live-head" in guard["reasons"]


def test_settlement_guard_fails_closed_on_missing_merge_packet(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text("[]\n", encoding="utf-8")

    packet = prompt_builder.build_decision_packet(
        registry_path=registry,
        pr=7435,
        expected_head="live-head",
        command_runner=_settlement_runner(include_packet=False),
    )

    guard = packet["settlement_guard"]
    assert guard["verdict"] == "fail_closed"
    assert "merge-packet authorization is missing or malformed" in guard["reasons"]


def test_settlement_guard_fails_closed_on_missing_pr_packet_entry(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text("[]\n", encoding="utf-8")

    packet = prompt_builder.build_decision_packet(
        registry_path=registry,
        pr=7435,
        expected_head="live-head",
        command_runner=_settlement_runner(include_packet_entry=False),
    )

    guard = packet["settlement_guard"]
    assert guard["verdict"] == "fail_closed"
    assert "merge-packet has no entry for PR #7435" in guard["reasons"]


def test_settlement_guard_fails_closed_on_unauthorized_merge_packet(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text("[]\n", encoding="utf-8")

    packet = prompt_builder.build_decision_packet(
        registry_path=registry,
        pr=7435,
        expected_head="live-head",
        command_runner=_settlement_runner(packet_ready=False),
    )

    guard = packet["settlement_guard"]
    assert guard["verdict"] == "fail_closed"
    assert "merge-packet does not authorize admin squash" in guard["reasons"]


def test_settlement_guard_fails_closed_on_ready_packet_with_pending_checks(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text("[]\n", encoding="utf-8")

    packet = prompt_builder.build_decision_packet(
        registry_path=registry,
        pr=7435,
        expected_head="live-head",
        command_runner=_settlement_runner(pending_checks=True),
    )

    guard = packet["settlement_guard"]
    assert guard["verdict"] == "fail_closed"
    assert guard["pending_checks"][0]["name"] == "aragora-merge-quorum"
    assert any("checks are pending" in reason for reason in guard["reasons"])


def test_settlement_guard_passes_for_clean_authorized_packet(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text("[]\n", encoding="utf-8")

    packet = prompt_builder.build_decision_packet(
        registry_path=registry,
        pr=7435,
        expected_head="live-head",
        command_runner=_settlement_runner(),
    )

    guard = packet["settlement_guard"]
    assert guard["verdict"] == "pass"
    assert guard["merge_packet_authorizes"] is True
    assert guard["reasons"] == []


def test_settlement_guard_prompt_includes_live_state_and_mailbox(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text("[]\n", encoding="utf-8")
    packet = prompt_builder.build_decision_packet(
        registry_path=registry,
        pr=7435,
        expected_head="live-head",
        command_runner=_settlement_runner(pending_checks=True),
    )

    prompt = prompt_builder.build_settlement_guard_prompt(packet, pr=7435)

    assert "python3 scripts/read_operator_steering.py --pr 7435" in prompt
    assert "Live head: live-head" in prompt
    assert "Pending required checks: Aragora Merge Quorum / aragora-merge-quorum" in prompt
    assert (
        "If the prompt above accomplishes no incremental progress make the next prompt one that does"
        in prompt
    )


def test_settlement_guard_prompt_uses_pr_mailbox_when_only_completed_lane_matches(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text(
        json.dumps(
            [
                {
                    "lane_id": "completed-lane",
                    "owner_session": "completed-owner",
                    "status": "completed",
                    "pr_number": 7435,
                }
            ]
        ),
        encoding="utf-8",
    )
    packet = prompt_builder.build_decision_packet(
        registry_path=registry,
        pr=7435,
        expected_head="live-head",
        command_runner=_settlement_runner(),
    )

    prompt = prompt_builder.build_settlement_guard_prompt(packet, pr=7435)

    assert "python3 scripts/read_operator_steering.py --pr 7435" in prompt
    assert "--lane-id completed-lane" not in prompt
