"""Tests for ``scripts/settlement_followup.py``."""

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


followup = _load_module("settlement_followup.py")


class FakeRunner:
    def __init__(
        self,
        *,
        worktree_found: bool = True,
        settlement_ok: bool = False,
        pr_mergeable: str = "CONFLICTING",
        merge_state: str = "DIRTY",
        pr_head: str = "5a692b5dd54f05f2befe0df7b497c56e3c6ead6f",
        root_dirty: bool = True,
    ) -> None:
        self.commands: list[tuple[Path, list[str]]] = []
        self.worktree_found = worktree_found
        self.settlement_ok = settlement_ok
        self.pr_mergeable = pr_mergeable
        self.merge_state = merge_state
        self.pr_head = pr_head
        self.root_dirty = root_dirty

    def __call__(self, command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        self.commands.append((cwd, command))
        joined = " ".join(command)
        if "identify_lane_owner.py" in joined:
            return self._json(
                command,
                {
                    "owner_session": "droid-7443-tier4-settlement-20260526",
                    "status": "completed",
                    "steering_inbox_path": "/repo/.aragora/operator-steering/droid",
                    "pending_message_count": 1,
                    "unread_message_count": 0,
                    "read_receipt_count": 3,
                    "last_steering_outcome": "completed",
                    "latest_read_receipt": {"outcome": "completed"},
                },
            )
        if command[:3] == ["git", "status", "--short"]:
            if str(cwd).endswith("repair"):
                return subprocess.CompletedProcess(
                    command,
                    0,
                    "## codex/droid-7443-tier4-settlement-20260526...origin/codex/droid-7443-tier4-settlement-20260526\n",
                    "",
                )
            if not self.root_dirty:
                return subprocess.CompletedProcess(command, 0, "## main...origin/main\n", "")
            return subprocess.CompletedProcess(
                command,
                0,
                "## main...origin/main\n M scripts/settle_tier4_pr.py\n",
                "",
            )
        if "agent_bridge.py --json health" in joined:
            return self._json(command, {"ok": True, "issues": []})
        if command[:3] == ["gh", "pr", "view"]:
            return self._json(
                command,
                {
                    "number": 7443,
                    "state": "OPEN",
                    "isDraft": False,
                    "headRefName": "codex/harvest-provider-readiness-secrets-cli-20260523",
                    "headRefOid": self.pr_head,
                    "mergeable": self.pr_mergeable,
                    "mergeStateStatus": self.merge_state,
                    "url": "https://github.com/synaptent/aragora/pull/7443",
                },
            )
        if command[:3] == ["git", "ls-remote", "origin"]:
            return subprocess.CompletedProcess(
                command,
                0,
                "63ff1513b4eb1e9a73b5ce3dbbc92a0179c00f67\trefs/heads/codex/droid-7443-tier4-settlement-20260526\n",
                "",
            )
        if command[:3] == ["git", "worktree", "list"]:
            stdout = (
                "worktree /tmp/repair\n"
                "HEAD 63ff1513b4eb1e9a73b5ce3dbbc92a0179c00f67\n"
                "branch refs/heads/codex/droid-7443-tier4-settlement-20260526\n\n"
                if self.worktree_found
                else ""
            )
            return subprocess.CompletedProcess(command, 0, stdout, "")
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(
                command,
                0,
                "63ff1513b4eb1e9a73b5ce3dbbc92a0179c00f67\n",
                "",
            )
        if "settle_tier4_pr.py --check" in joined:
            gate = {
                "ok": self.settlement_ok,
                "pr": 7443,
                "expected_head": "5a692b5dd54f05f2befe0df7b497c56e3c6ead6f",
                "actual_head": "5a692b5dd54f05f2befe0df7b497c56e3c6ead6f",
                "merge_state": self.merge_state,
                "blockers": []
                if self.settlement_ok
                else ["PR #7443 is DIRTY", "merge-packet has unexpected blockers: 7443"],
                "authorized_actions": ["merge"],
            }
            return subprocess.CompletedProcess(
                command,
                0 if self.settlement_ok else 1,
                json.dumps({"gate": gate, "applied_commands": []}),
                "",
            )
        if "-m pytest" in joined:
            return subprocess.CompletedProcess(
                command,
                0,
                "tests/scripts/test_settle_tier4_pr.py ........................ [100%]\n24 passed\n",
                "",
            )
        return subprocess.CompletedProcess(command, 0, "{}", "")

    @staticmethod
    def _json(command: list[str], payload: dict[str, Any]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, json.dumps(payload), "")


def test_followup_packet_reports_owner_branch_validation_and_prompt(tmp_path: Path) -> None:
    runner = FakeRunner()

    packet = followup.build_followup_packet(
        pr=7443,
        head="5a692b5dd54f05f2befe0df7b497c56e3c6ead6f",
        repair_branch="origin/codex/droid-7443-tier4-settlement-20260526",
        repair_head="63ff1513b4eb1e9a73b5ce3dbbc92a0179c00f67",
        repo_root=tmp_path,
        include_prompt=True,
        runner=runner,
    )

    assert packet["mailbox"]["owner_session"] == "droid-7443-tier4-settlement-20260526"
    assert packet["root"]["dirty"] is True
    assert packet["repair_branch"]["published"] is True
    assert packet["repair_branch"]["matches_expected"] is True
    assert packet["repair_worktree"]["found"] is True
    assert packet["validation"]["focused_tests"]["ok"] is True
    assert packet["apply_blockers"] == [
        "PR #7443 is DIRTY",
        "merge-packet has unexpected blockers: 7443",
    ]
    assert "settlement_followup.py --pr 7443" in packet["next_prompt"]
    assert followup.CONVERGENCE_SENTENCE in packet["next_prompt"]
    assert followup.AUTOMATION_SENTENCE in packet["next_prompt"]


def test_followup_does_not_run_apply_push_or_cleanup(tmp_path: Path) -> None:
    runner = FakeRunner(settlement_ok=True, pr_mergeable="MERGEABLE", merge_state="UNSTABLE")

    packet = followup.build_followup_packet(
        pr=7443,
        head="5a692b5dd54f05f2befe0df7b497c56e3c6ead6f",
        repair_branch="codex/droid-7443-tier4-settlement-20260526",
        repair_head="63ff1513b4eb1e9a73b5ce3dbbc92a0179c00f67",
        repo_root=tmp_path,
        include_prompt=True,
        runner=runner,
    )

    flat_commands = [" ".join(command) for _, command in runner.commands]
    assert all("--apply" not in command for command in flat_commands)
    assert all("git push" not in command for command in flat_commands)
    assert all("pr merge" not in command for command in flat_commands)
    assert all("worktree add" not in command for command in flat_commands)
    assert packet["validation"]["settlement_check"]["ok"] is True


def test_missing_repair_worktree_reports_validation_blocker(tmp_path: Path) -> None:
    runner = FakeRunner(worktree_found=False)

    packet = followup.build_followup_packet(
        pr=7443,
        head="5a692b5dd54f05f2befe0df7b497c56e3c6ead6f",
        repair_branch="codex/droid-7443-tier4-settlement-20260526",
        repair_head="63ff1513b4eb1e9a73b5ce3dbbc92a0179c00f67",
        repo_root=tmp_path,
        runner=runner,
    )

    assert packet["repair_worktree"]["found"] is False
    assert packet["validation"]["settlement_check"]["ran"] is False
    assert packet["validation"]["settlement_check"]["blockers"] == ["repair worktree not found"]
    assert packet["validation"]["focused_tests"]["ran"] is False


def test_live_head_drift_blocks_settlement_apply_prompt(tmp_path: Path) -> None:
    runner = FakeRunner(
        settlement_ok=True,
        pr_mergeable="MERGEABLE",
        merge_state="UNSTABLE",
        pr_head="new-head",
        root_dirty=False,
    )

    packet = followup.build_followup_packet(
        pr=7443,
        head="old-head",
        repair_branch="codex/droid-7443-tier4-settlement-20260526",
        repair_head="63ff1513b4eb1e9a73b5ce3dbbc92a0179c00f67",
        repo_root=tmp_path,
        include_prompt=True,
        runner=runner,
    )

    assert packet["requested_head_matches_live"] is False
    assert packet["live_head"] == "new-head"
    assert packet["apply_blockers"][0] == (
        "PR #7443 head drifted from requested old-head to live new-head; "
        "do not prepare settlement apply for a stale head"
    )
    assert packet["validation"]["settlement_check"]["ok"] is True
    assert "Stop before settlement apply" in packet["next_prompt"]
    assert "Prepare exact-head settlement apply" not in packet["next_prompt"]
