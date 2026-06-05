from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from scripts.github_cli_health import GitHubCLIHealth

import scripts.drain_codex_automation_value as mod


def _pr_view(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "number": 7768,
        "state": "OPEN",
        "isDraft": False,
        "title": "Safe queue repair",
        "headRefName": "codex/safe-queue-repair",
        "headRefOid": "abc123def456",
        "mergeable": "MERGEABLE",
        "mergeStateStatus": "CLEAN",
        "url": "https://github.example/pr/7768",
        "files": [{"path": "scripts/example.py"}],
    }
    payload.update(overrides)
    return payload


def _checks(*, state: str = "SUCCESS", bucket: str = "pass") -> list[dict[str, str]]:
    return [
        {
            "name": "Tests",
            "state": state,
            "bucket": bucket,
            "workflow": "Tests",
            "link": "https://github.example/run/1",
        }
    ]


def _packet(**entry_overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "pr_number": 7768,
        "head_sha": "abc123def456",
        "tier": 2,
        "requires_human_risk_settlement": False,
        "unresolved_dissent": False,
        "admin_squash_allowed": True,
        "status": "satisfied",
        "verdict": "admin_squash_allowed",
    }
    entry.update(entry_overrides)
    return {
        "version": "merge_authorization_packet.v1",
        "entries": [entry],
        "admin_squash_order": [7768],
        "not_ready": [],
    }


def _settle(blockers: list[str] | None = None) -> dict[str, Any]:
    return {"blockers": [] if blockers is None else blockers}


def test_protected_squash_command_never_uses_admin() -> None:
    command = mod.protected_squash_merge_command(7768, "abc123def456")

    assert command == [
        "gh",
        "pr",
        "merge",
        "7768",
        "--squash",
        "--match-head-commit",
        "abc123def456",
    ]
    assert "--admin" not in command


def test_evaluate_merge_candidate_accepts_tier_0_to_2_exact_head_green_pr() -> None:
    evaluation = mod.evaluate_merge_candidate(
        pr_view=_pr_view(),
        required_checks=_checks(),
        merge_packet=_packet(),
        settle_one=_settle(),
        owner={},
    )

    assert evaluation.eligible is True
    assert evaluation.reason == "eligible"
    assert evaluation.command == mod.protected_squash_merge_command(7768, "abc123def456")


def test_evaluate_merge_candidate_rejects_required_check_json_skew() -> None:
    evaluation = mod.evaluate_merge_candidate(
        pr_view=_pr_view(),
        required_checks=[],
        merge_packet=_packet(),
        settle_one=_settle(),
        owner={},
    )

    assert evaluation.eligible is False
    assert evaluation.reason == "required checks unavailable: empty required-check JSON"


def test_evaluate_merge_candidate_rejects_tier_4_human_risk() -> None:
    evaluation = mod.evaluate_merge_candidate(
        pr_view=_pr_view(),
        required_checks=_checks(),
        merge_packet=_packet(tier=4, requires_human_risk_settlement=True),
        settle_one=_settle(),
        owner={},
    )

    assert evaluation.eligible is False
    assert "Tier 4 requires report-only handling" in evaluation.blockers
    assert "requires_human_risk_settlement=true" in evaluation.blockers


def test_evaluate_merge_candidate_rejects_active_owner() -> None:
    evaluation = mod.evaluate_merge_candidate(
        pr_view=_pr_view(),
        required_checks=_checks(),
        merge_packet=_packet(),
        settle_one=_settle(),
        owner={"status": "active", "owner_session": "codex-live-owner"},
    )

    assert evaluation.eligible is False
    assert evaluation.reason == "active owner: codex-live-owner (active)"


def test_evaluate_merge_candidate_rejects_unavailable_owner_status() -> None:
    evaluation = mod.evaluate_merge_candidate(
        pr_view=_pr_view(),
        required_checks=_checks(),
        merge_packet=_packet(),
        settle_one=_settle(),
        owner=None,
    )

    assert evaluation.eligible is False
    assert evaluation.reason == "owner status unavailable"


def test_owner_blockers_allows_unowned_no_lane_match_payload() -> None:
    assert mod.owner_blockers({"ok": False, "error": "no lane matched criteria {'pr': 7768}"}) == []


def test_run_gh_invalid_timeout_env_falls_back(monkeypatch: Any, tmp_path: Path) -> None:
    observed: dict[str, Any] = {}

    def fake_gh_run(
        args: Any,
        *,
        timeout: float,
        prefer_app: bool,
        write_op: bool,
        env: Any,
        max_retries: int,
    ) -> subprocess.CompletedProcess[str]:
        observed.update(
            {
                "args": list(args),
                "timeout": timeout,
                "prefer_app": prefer_app,
                "write_op": write_op,
                "env": env,
                "max_retries": max_retries,
            }
        )
        return subprocess.CompletedProcess(
            args=["gh", *list(args)], returncode=0, stdout="[]", stderr=""
        )

    monkeypatch.setenv("ARAGORA_AUTOMATION_GH_TIMEOUT_SECONDS", "not-an-int")
    monkeypatch.setattr(mod, "github_cli_env", lambda env: {"GH_TOKEN": "redacted"})
    monkeypatch.setattr(mod, "gh_subprocess_run", fake_gh_run)

    proc = mod._run(["gh", "pr", "list"], tmp_path)

    assert proc.returncode == 0
    assert observed["args"] == ["pr", "list"]
    assert observed["timeout"] == mod.DEFAULT_GH_TIMEOUT_SECONDS
    assert observed["write_op"] is False


def test_issue_publish_blocker_respects_open_issue_cap() -> None:
    blocker = mod.issue_publish_blocker(
        {
            "github_queue": {
                "available": True,
                "open_issue_count": 16,
                "pressure": {"open_issue_cap_reached": True},
            }
        },
        max_open_issues=16,
    )

    assert blocker == "open issue cap reached"


def test_branch_publish_command_opens_draft_prs() -> None:
    config = mod.DrainConfig(
        repo_root=Path("/repo"),
        github_repo="owner/repo",
        state_root=Path("/state"),
        outbox_dir=Path("/state/.aragora/automation-outbox"),
        receipt_dir=Path("/state/.aragora/automation-receipts"),
        cache_output=None,
        base="origin/main",
        branch_limit=2,
        issue_limit=4,
        merge_limit=1,
        max_open_prs=12,
        max_open_issues=16,
        branch_scan_limit=40,
        apply=True,
    )

    command = mod._branch_publish_command(config)

    assert "--draft" in command
    assert "--apply" in command


def test_run_drain_stops_when_github_unavailable(monkeypatch: Any, tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_health(_repo_root: Path) -> GitHubCLIHealth:
        return GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode="connectivity_failed",
            error="lookup api.github.com: no such host",
            repo=str(tmp_path),
        )

    def fake_runner(args: Any, cwd: Path) -> subprocess.CompletedProcess[str]:
        calls.append(list(args))
        return subprocess.CompletedProcess(args=list(args), returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(mod, "check_github_cli_health", fake_health)
    config = mod.DrainConfig(
        repo_root=tmp_path,
        github_repo="owner/repo",
        state_root=tmp_path,
        outbox_dir=tmp_path / ".aragora" / "automation-outbox",
        receipt_dir=tmp_path / ".aragora" / "automation-receipts",
        cache_output=None,
        base="origin/main",
        branch_limit=2,
        issue_limit=4,
        merge_limit=1,
        max_open_prs=12,
        max_open_issues=16,
        branch_scan_limit=40,
        apply=True,
    )

    report = mod.run_drain(config, runner=fake_runner)

    assert report["status"] == "github_unavailable"
    assert report["blockers"] == ["lookup api.github.com: no such host"]
    assert calls == []


def test_run_drain_skips_issue_publish_when_cap_reached(monkeypatch: Any, tmp_path: Path) -> None:
    def fake_health(_repo_root: Path) -> GitHubCLIHealth:
        return GitHubCLIHealth(
            ready=True,
            auth_ok=True,
            api_ok=True,
            mode="ready",
            error="",
            repo=str(tmp_path),
        )

    def fake_runner(args: Any, cwd: Path) -> subprocess.CompletedProcess[str]:
        command = list(args)
        stdout = "{}"
        if any(str(part).endswith("cache_codex_automation_github_status.py") for part in command):
            stdout = json.dumps(
                {
                    "github_queue": {
                        "available": True,
                        "open_issue_count": 16,
                        "pressure": {"open_issue_cap_reached": True},
                    },
                    "local_queue": {"outbox_count": 3},
                }
            )
        elif any(str(part).endswith("reconcile_automation_outbox.py") for part in command):
            stdout = json.dumps({"archived": 0, "kept": 3})
        elif command[:3] == ["gh", "pr", "list"]:
            stdout = "[]"
        elif any(str(part).endswith("publish_codex_automation_branches.py") for part in command):
            stdout = json.dumps({"published": []})
        return subprocess.CompletedProcess(args=command, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(mod, "check_github_cli_health", fake_health)
    config = mod.DrainConfig(
        repo_root=tmp_path,
        github_repo="owner/repo",
        state_root=tmp_path,
        outbox_dir=tmp_path / ".aragora" / "automation-outbox",
        receipt_dir=tmp_path / ".aragora" / "automation-receipts",
        cache_output=None,
        base="origin/main",
        branch_limit=2,
        issue_limit=4,
        merge_limit=1,
        max_open_prs=12,
        max_open_issues=16,
        branch_scan_limit=40,
        apply=False,
    )

    report = mod.run_drain(config, runner=fake_runner)

    issue_phase = [
        phase for phase in report["phases"] if phase["name"] == "publish_handoff_issues"
    ][0]
    assert issue_phase["skipped"] == [{"reason": "open issue cap reached"}]


def test_publisher_wrapper_propagates_value_drain_failure() -> None:
    wrapper = Path("scripts/run_codex_automation_publisher.sh").read_text(encoding="utf-8")

    assert "authenticated value drain failed (exit ${rc})" in wrapper
    assert 'exit "${rc}"' in wrapper
