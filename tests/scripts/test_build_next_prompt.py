"""Tests for ``scripts/build_next_prompt.py``."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


class _BrokenStdout:
    def write(self, _text: str) -> int:
        raise BrokenPipeError

    def flush(self) -> None:
        raise BrokenPipeError


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


def _clean_checkout_runner(
    *,
    repo_root: Path,
    worktree: Path | None = None,
    root_dirty: bool = True,
    worktree_dirty: bool = False,
    worktree_head: str = "clean-head",
    origin_main: str = "clean-head",
    worktree_list_error: bool = False,
) -> Any:
    def fake_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command == ["git", "status", "--short", "--branch", "--untracked-files=all"]:
            status = "## feature...origin/main\n"
            if root_dirty:
                status += " M dirty.py\n"
            return subprocess.CompletedProcess(command, 0, status, "")
        if command == ["git", "worktree", "list", "--porcelain"]:
            if worktree_list_error:
                return subprocess.CompletedProcess(command, 1, "", "worktree metadata locked")
            text = f"worktree {repo_root}\nHEAD root-head\nbranch refs/heads/main\n"
            if worktree is not None:
                text += f"\nworktree {worktree}\nHEAD {worktree_head}\ndetached\n"
            return subprocess.CompletedProcess(command, 0, text, "")
        if command[:3] == ["git", "-C", str(repo_root)]:
            if command[3:6] == ["status", "--short", "--branch"]:
                status = "## feature...origin/main\n"
                if root_dirty:
                    status += " M dirty.py\n"
                return subprocess.CompletedProcess(command, 0, status, "")
            if command[3:] == ["rev-parse", "HEAD", "origin/main"]:
                return subprocess.CompletedProcess(command, 0, f"root-head\n{origin_main}\n", "")
        if worktree is not None and command[:3] == ["git", "-C", str(worktree)]:
            if command[3:6] == ["status", "--short", "--branch"]:
                status = "## HEAD (no branch)\n"
                if worktree_dirty:
                    status += " M generated.py\n"
                return subprocess.CompletedProcess(command, 0, status, "")
            if command[3:] == ["rev-parse", "HEAD", "origin/main"]:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    f"{worktree_head}\n{origin_main}\n",
                    "",
                )
        if command[:2] == ["df", "-h"]:
            return subprocess.CompletedProcess(command, 0, "Filesystem Size Used Avail\n", "")
        return subprocess.CompletedProcess(command, 0, "{}", "")

    return fake_runner


def test_prompt_starts_with_mailbox_and_owner_verification(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
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
        repo_root=repo_root,
        lane_id="P106-merge-gate-settlement",
        pr=7423,
        command_runner=_clean_checkout_runner(repo_root=repo_root, root_dirty=False),
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
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    registry.write_text("[]\n", encoding="utf-8")

    prompt = prompt_builder.build_prompt(
        registry_path=registry,
        repo_root=repo_root,
        pr=7407,
        command_runner=_clean_checkout_runner(repo_root=repo_root, root_dirty=False),
    )

    assert "If you cannot map yourself to a lane, run read-only only" in prompt
    assert "Do not paste raw transcripts" in prompt


def test_prompt_shell_quotes_live_lane_values(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
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

    prompt = prompt_builder.build_prompt(
        registry_path=registry,
        repo_root=repo_root,
        pr=7425,
        command_runner=_clean_checkout_runner(repo_root=repo_root, root_dirty=False),
    )

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


def test_decision_packet_counts_shared_outbox_when_local_outbox_absent(
    tmp_path: Path, monkeypatch: Any
) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text("[]\n", encoding="utf-8")
    repo_root = tmp_path / "disposable-worktree"
    repo_root.mkdir()
    state_root = tmp_path / "shared-checkout"
    outbox = state_root / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    (outbox / "one.json").write_text("{}", encoding="utf-8")
    (outbox / "two.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("ARAGORA_AUTOMATION_STATE_ROOT", str(state_root))

    def fake_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(command, 0, "## main...origin/main\n", "")
        if command[:2] == ["df", "-h"]:
            assert command[2] == str(outbox)
            return subprocess.CompletedProcess(
                command,
                0,
                "Filesystem      Size   Used  Avail Capacity iused ifree %iused Mounted on\n",
                "",
            )
        return subprocess.CompletedProcess(command, 0, "{}", "")

    packet = prompt_builder.build_decision_packet(
        registry_path=registry,
        repo_root=repo_root,
        command_runner=fake_runner,
    )

    assert packet["disk_outbox"]["outbox_file_count"] == 2
    assert packet["disk_outbox"]["outbox_dir"] == str(outbox)
    assert packet["disk_outbox"]["outbox_returncode"] == 0


def test_clean_checkout_packet_selects_clean_detached_origin_main_worktree(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    worktree = tmp_path / "clean-wt"
    repo_root.mkdir()
    worktree.mkdir()

    packet = prompt_builder._clean_checkout_packet(
        repo_root,
        _clean_checkout_runner(repo_root=repo_root, worktree=worktree),
        pr=7561,
        expected_head="expected-head",
    )

    assert packet["status"] == "selected"
    assert packet["selected_path"] == str(worktree)
    assert packet["candidates"][1]["status"] == "usable_clean_origin_main"
    assert packet["candidates"][1]["detached"] is True
    assert f"git -C {worktree} fetch origin main" in packet["recommended_prompt"]
    assert "HEAD equals the refreshed origin/main after fetch" in packet["recommended_prompt"]


def test_clean_checkout_packet_rejects_stale_clean_worktree(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    worktree = tmp_path / "stale-wt"
    repo_root.mkdir()
    worktree.mkdir()

    packet = prompt_builder._clean_checkout_packet(
        repo_root,
        _clean_checkout_runner(
            repo_root=repo_root,
            worktree=worktree,
            worktree_head="old-head",
            origin_main="new-head",
        ),
        pr=7561,
        expected_head="expected-head",
    )

    assert packet["status"] == "needs_disposable_worktree"
    assert packet["selected_path"] is None
    assert packet["candidates"][1]["status"] == "stale_vs_origin_main"
    assert (
        "git worktree add --detach /private/tmp/aragora-pr7561-triage origin/main"
        in packet["recommended_prompt"]
    )


def test_clean_checkout_packet_rejects_dirty_worktree_even_when_head_matches(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    worktree = tmp_path / "dirty-wt"
    repo_root.mkdir()
    worktree.mkdir()

    packet = prompt_builder._clean_checkout_packet(
        repo_root,
        _clean_checkout_runner(repo_root=repo_root, worktree=worktree, worktree_dirty=True),
        pr=7561,
        expected_head="expected-head",
    )

    assert packet["status"] == "needs_disposable_worktree"
    assert packet["selected_path"] is None
    assert packet["candidates"][1]["status"] == "dirty"
    assert "generated.py" in packet["candidates"][1]["dirty_paths"]


def test_prompt_emits_disposable_worktree_prompt_when_no_clean_checkout(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text("[]\n", encoding="utf-8")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    prompt = prompt_builder.build_prompt(
        registry_path=registry,
        repo_root=repo_root,
        pr=7561,
        expected_head="expected-head",
        command_runner=_clean_checkout_runner(repo_root=repo_root, worktree=None),
    )

    assert "Clean-checkout routing: no registered clean origin/main checkout is available" in prompt
    assert "git fetch origin main" in prompt
    assert "git worktree add --detach /private/tmp/aragora-pr7561-triage origin/main" in prompt
    assert "python3 scripts/read_operator_steering.py --pr 7561 --no-receipt --json" in prompt
    assert "Stop if PR #7561 head drifted from expected-head" in prompt


def test_prompt_emits_disposable_worktree_prompt_when_worktree_scan_fails(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text("[]\n", encoding="utf-8")
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    runner = _clean_checkout_runner(repo_root=repo_root, worktree_list_error=True)
    packet = prompt_builder._clean_checkout_packet(
        repo_root,
        runner,
        pr=7561,
        expected_head="expected-head",
    )
    prompt = prompt_builder.build_prompt(
        registry_path=registry,
        repo_root=repo_root,
        pr=7561,
        expected_head="expected-head",
        command_runner=runner,
    )
    decision = prompt_builder.build_decision_packet(
        registry_path=registry,
        repo_root=repo_root,
        pr=7561,
        expected_head="expected-head",
        command_runner=runner,
    )

    assert packet["status"] == "error"
    assert packet["error"] == "worktree metadata locked"
    assert "Clean-checkout routing: the registered clean-checkout scan failed" in prompt
    assert "git worktree add --detach /private/tmp/aragora-pr7561-triage origin/main" in prompt
    assert "Stop if PR #7561 head drifted from expected-head" in prompt
    assert decision["selected_action"] == "create_clean_checkout_prompt"


def test_prompt_includes_selected_clean_checkout_path(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text("[]\n", encoding="utf-8")
    repo_root = tmp_path / "repo"
    worktree = tmp_path / "clean-wt"
    repo_root.mkdir()
    worktree.mkdir()

    prompt = prompt_builder.build_prompt(
        registry_path=registry,
        repo_root=repo_root,
        pr=7561,
        command_runner=_clean_checkout_runner(repo_root=repo_root, worktree=worktree),
    )

    assert "Clean-checkout routing: root is not suitable" in prompt
    assert f"Run repo-native helpers only from this checkout: {worktree}" in prompt
    assert f"git -C {worktree} fetch origin main" in prompt
    assert f"git -C {worktree} rev-parse HEAD origin/main" in prompt
    assert "If it is dirty or stale after fetch, do not use it" in prompt


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


def test_decision_packet_detects_merged_pr_with_active_tmux_evidence_lane(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text("[]\n", encoding="utf-8")

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
                        "number": 7735,
                        "state": "MERGED",
                        "headRefOid": "merged-head",
                        "headRefName": "droid/merge-quorum-reconcile",
                        "mergedAt": "2026-06-04T13:43:56Z",
                        "mergeCommit": {"oid": "merge-commit"},
                    }
                ),
                "",
            )
        if command[:3] == ["gh", "pr", "checks"]:
            return subprocess.CompletedProcess(command, 0, "[]", "")
        if "merge-packet" in joined:
            return subprocess.CompletedProcess(command, 0, "{}", "")
        if command[:2] == ["df", "-h"]:
            return subprocess.CompletedProcess(command, 0, "Filesystem Size Used Avail\n", "")
        if command[:2] == ["tmux", "list-panes"]:
            return subprocess.CompletedProcess(
                command,
                0,
                "aragora\tclaude-7735-evidence-20260604T1258Z\t0\t30664\t/tmp/wt\tclaude\n",
                "",
            )
        return subprocess.CompletedProcess(command, 0, "{}", "")

    packet = prompt_builder.build_decision_packet(
        registry_path=registry,
        pr=7735,
        command_runner=fake_runner,
    )

    coordination = packet["post_merge_lane_coordination"]
    assert coordination["detected"] is True
    assert coordination["active_lanes"][0]["source"] == "tmux_pane"
    assert coordination["active_lanes"][0]["tmux_target"] == (
        "aragora:claude-7735-evidence-20260604T1258Z"
    )
    assert packet["selected_action"] == "post_merge_lane_retirement_coordination"
    assert "merged PR still has active target lane" in packet["blockers"]

    prompt = prompt_builder.build_post_merge_lane_coordination_prompt(packet, pr=7735)

    assert prompt is not None
    assert "Goal: coordinate stale active lane(s) after PR #7735 already merged." in prompt
    assert "merge_commit=merge-commit" in prompt
    assert "aragora:claude-7735-evidence-20260604T1258Z" in prompt
    assert "do not collect evidence, rerun checks, mark statuses, or merge" in prompt


def test_main_replaces_standard_prompt_with_post_merge_coordination(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text("[]\n", encoding="utf-8")
    packet = {
        "pr": {
            "state": "MERGED",
            "headRefOid": "merged-head",
            "headRefName": "droid/merge-quorum-reconcile",
            "mergedAt": "2026-06-04T13:43:56Z",
            "mergeCommit": {"oid": "merge-commit"},
        },
        "target_active_lanes": [],
        "tmux_panes": {
            "panes": [
                {
                    "tmux_target": "aragora:claude-7735-evidence-20260604T1258Z",
                    "window_name": "claude-7735-evidence-20260604T1258Z",
                }
            ]
        },
        "active_sessions": {},
    }

    monkeypatch.setattr(prompt_builder, "build_prompt", lambda **_kwargs: "standard prompt\n")
    monkeypatch.setattr(prompt_builder, "build_post_merge_fast_packet", lambda **_kwargs: packet)
    monkeypatch.setattr(
        prompt_builder,
        "build_decision_packet",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("full packet not needed")),
    )

    assert (
        prompt_builder.main(
            ["--pr", "7735", "--registry-path", str(registry), "--repo-root", str(tmp_path)]
        )
        == 0
    )

    out = capsys.readouterr().out
    assert "standard prompt" not in out
    assert "coordinate stale active lane(s) after PR #7735 already merged" in out


def test_main_routes_stale_mailbox_only_owner_to_steering_prompt(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text("[]\n", encoding="utf-8")
    packet = {
        "owner_state": {
            "lane_id": "Q324-repair-build-next-prompt-merged-active-lane-handoff",
            "owner_session": "engineering-autopilot-2-Q324-build-next-prompt",
            "status": "blocked",
            "branch": "codex/build-next-prompt-merged-active-lane-20260604",
            "last_heartbeat_at": "2026-06-04T14:45:34Z",
            "pending_message_count": 1,
            "unread_message_count": 1,
            "read_receipt_count": 0,
            "harness_confidence": "mailbox_only_fuzzy_thread",
            "live_prompt_dispatchable": False,
            "live_process": {"found": False},
        }
    }

    monkeypatch.setattr(prompt_builder, "build_decision_packet", lambda **_kwargs: packet)
    monkeypatch.setattr(
        prompt_builder,
        "build_prompt",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("standard prompt not needed")),
    )

    assert (
        prompt_builder.main(
            [
                "--branch",
                "codex/build-next-prompt-merged-active-lane-20260604",
                "--registry-path",
                str(registry),
                "--repo-root",
                str(tmp_path),
            ]
        )
        == 0
    )

    out = capsys.readouterr().out
    assert "Goal: steer stale mailbox-only owner lane" in out
    assert "scripts/send_operator_steering.py" in out
    assert "--to engineering-autopilot-2-Q324-build-next-prompt" in out
    assert "--lane-id Q324-repair-build-next-prompt-merged-active-lane-handoff" in out
    assert "last heartbeat is 2026-06-04T14:45:34Z" in out


def test_main_guards_unresolved_operator_choice_placeholders(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(
        prompt_builder,
        "build_prompt",
        lambda **_kwargs: (
            "Operator action:\n"
            "I explicitly choose option 1|2|3>: <let lane finish | retire | supersede>.\n"
        ),
    )

    assert (
        prompt_builder.main(["--registry-path", str(registry), "--repo-root", str(tmp_path)]) == 0
    )

    out = capsys.readouterr().out
    assert "unresolved operator-choice placeholder" in out
    assert "Do not continue lane work" in out
    assert "I explicitly choose option 1|2|3" not in out


def test_main_json_suppresses_broken_pipe(tmp_path: Path, monkeypatch: Any) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text("[]\n", encoding="utf-8")

    monkeypatch.setattr(prompt_builder.sys, "stdout", _BrokenStdout())
    monkeypatch.setattr(prompt_builder.os, "dup2", lambda *_args: None)
    monkeypatch.setattr(prompt_builder, "build_prompt", lambda **_kwargs: "prompt\n")
    monkeypatch.setattr(
        prompt_builder,
        "build_decision_packet",
        lambda **_kwargs: {"selected_action": "read_only_owner_routing"},
    )
    monkeypatch.setattr(
        prompt_builder,
        "build_settlement_guard_prompt",
        lambda *_args, **_kwargs: "guard\n",
    )

    assert (
        prompt_builder.main(
            ["--json", "--registry-path", str(registry), "--repo-root", str(tmp_path)]
        )
        == 0
    )
