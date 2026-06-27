"""Tests for ``scripts/resolve_lane_conflicts.py``."""

from __future__ import annotations

import importlib.util
import json
import os
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


resolver = _load_module("resolve_lane_conflicts.py")
SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "resolve_lane_conflicts.py"


def _init_repo_with_origin(
    path: Path, origin: str = "https://github.com/synaptent/aragora.git"
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init"], cwd=path, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    subprocess.run(
        ["git", "config", "remote.origin.url", origin],
        cwd=path,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fake_gh(tmp_path: Path, payload: dict[str, Any], *, exit_code: int = 0) -> Path:
    gh_dir = tmp_path / f"fake-gh-{len(list(tmp_path.glob('fake-gh-*')))}"
    gh_dir.mkdir()
    gh = gh_dir / "gh"
    gh.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        f"payload = {json.dumps(payload)!r}\n"
        f"exit_code = {exit_code!r}\n"
        "if exit_code:\n"
        "    print('gh unavailable', file=sys.stderr)\n"
        "    raise SystemExit(exit_code)\n"
        "print(payload)\n",
        encoding="utf-8",
    )
    gh.chmod(0o755)
    return gh


def _merged_pr_gh(tmp_path: Path, *, merge_commit: str = "merge") -> Path:
    return _fake_gh(
        tmp_path,
        {
            "number": 7435,
            "state": "MERGED",
            "headRefOid": "head",
            "mergedAt": "2026-05-23T19:16:23Z",
            "mergeCommit": {"oid": merge_commit},
            "url": "https://github.com/synaptent/aragora/pull/7435",
        },
    )


def _closed_pr_gh(
    tmp_path: Path,
    *,
    closed_at: str = "2026-05-23T19:16:23Z",
    head_sha: str = "head",
) -> Path:
    return _fake_gh(
        tmp_path,
        {
            "number": 7435,
            "state": "CLOSED",
            "headRefOid": head_sha,
            "closedAt": closed_at,
            "mergedAt": None,
            "mergeCommit": None,
            "url": "https://github.com/synaptent/aragora/pull/7435",
        },
    )


def _write_read_receipt(inbox: Path, *, message_filename: str, message_sha256: str = "") -> None:
    receipt_dir = inbox / "_read_receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        receipt_dir / "receipt.json",
        {
            "schema_version": "aragora-operator-steering-read-receipt/1.0",
            "message_filename": message_filename,
            "message_sha256": message_sha256,
            "outcome": "read",
            "read_at_utc": "2026-05-23T19:20:00Z",
        },
    )


def test_detects_completed_owner_conflict_without_mutating(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    registry.write_text(
        json.dumps(
            [
                {
                    "lane_id": "P104-ssd-cleanup-continuation",
                    "owner_session": "codex-P104",
                    "status": "conflict",
                    "conflict_session": "codex-R03",
                    "conflict_reason": "stale cleanup overlap",
                },
                {
                    "lane_id": "R03-post-p102-harvest-followthrough",
                    "owner_session": "codex-R03",
                    "status": "completed",
                },
            ]
        ),
        encoding="utf-8",
    )

    candidates = resolver.find_resolvable_conflicts(registry)

    assert [candidate["lane_id"] for candidate in candidates] == ["P104-ssd-cleanup-continuation"]
    assert json.loads(registry.read_text(encoding="utf-8"))[0]["status"] == "conflict"


def test_cli_defaults_to_automation_state_root_for_registry(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    state_root = tmp_path / "state-root"
    registry = state_root / ".aragora" / "agent-bridge" / "lanes.json"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        json.dumps(
            [
                {
                    "lane_id": "P104-ssd-cleanup-continuation",
                    "owner_session": "codex-P104",
                    "status": "conflict",
                    "conflict_session": "codex-R03",
                    "conflict_reason": "stale cleanup overlap",
                },
                {
                    "lane_id": "R03-post-p102-harvest-followthrough",
                    "owner_session": "codex-R03",
                    "status": "completed",
                },
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARAGORA_AUTOMATION_STATE_ROOT", str(state_root))
    monkeypatch.setattr(
        resolver,
        "_trusted_automation_state_roots",
        lambda repo_root=resolver.DEFAULT_REPO_ROOT: {(state_root / ".aragora").resolve()},
    )

    rc = resolver.main(["--json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["registry_path"] == str(registry)
    assert payload["candidate_count"] == 1
    assert payload["candidates"][0]["lane_id"] == "P104-ssd-cleanup-continuation"


def test_cli_rejects_untrusted_automation_state_root(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    trusted_root = (tmp_path / "repo" / ".aragora").resolve()
    monkeypatch.setenv("ARAGORA_AUTOMATION_STATE_ROOT", str(tmp_path / "attacker-state"))
    monkeypatch.setattr(
        resolver,
        "_trusted_automation_state_roots",
        lambda repo_root=resolver.DEFAULT_REPO_ROOT: {trusted_root},
    )

    rc = resolver.main(["--json"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert payload["blocked_reason"] == "invalid_automation_state_root"
    assert "untrusted ARAGORA_AUTOMATION_STATE_ROOT" in payload["error"]
    assert str(trusted_root) in payload["error"]


def test_cli_explicit_paths_survive_untrusted_automation_state_root(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    monkeypatch.setenv("ARAGORA_AUTOMATION_STATE_ROOT", str(tmp_path / "attacker-state"))
    monkeypatch.setattr(
        resolver,
        "_trusted_automation_state_roots",
        lambda repo_root=resolver.DEFAULT_REPO_ROOT: {(tmp_path / "repo" / ".aragora").resolve()},
    )
    registry = tmp_path / "lanes.json"
    receipts = tmp_path / "receipts"
    registry.write_text("[]", encoding="utf-8")
    receipts.mkdir()

    rc = resolver.main(
        [
            "--json",
            "--registry-path",
            str(registry),
            "--receipt-dir",
            str(receipts),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["registry_path"] == str(registry)
    assert payload["candidate_count"] == 0


def test_merged_pr_audit_blocks_untrusted_default_safety_paths(
    tmp_path: Path,
    monkeypatch: Any,
    capsys: Any,
) -> None:
    monkeypatch.setenv("ARAGORA_AUTOMATION_STATE_ROOT", str(tmp_path / "attacker-state"))
    monkeypatch.setattr(
        resolver,
        "_trusted_automation_state_roots",
        lambda repo_root=resolver.DEFAULT_REPO_ROOT: {(tmp_path / "repo" / ".aragora").resolve()},
    )
    registry = tmp_path / "lanes.json"
    receipts = tmp_path / "receipts"
    registry.write_text("[]", encoding="utf-8")
    receipts.mkdir()

    rc = resolver.main(
        [
            "--merged-pr-lane-audit",
            "--pr",
            "7435",
            "--registry-path",
            str(registry),
            "--receipt-dir",
            str(receipts),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 2
    assert payload["blocked_reason"] == "invalid_automation_state_root"
    assert "untrusted ARAGORA_AUTOMATION_STATE_ROOT" in payload["error"]


def test_automation_state_root_accepts_registered_worktree_checkout(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    repo = tmp_path / "repo"
    shared = tmp_path / "shared-checkout"
    _init_repo_with_origin(repo)
    _init_repo_with_origin(shared, "git@github.com:synaptent/aragora.git")
    (shared / ".aragora").mkdir()
    monkeypatch.setenv("ARAGORA_AUTOMATION_STATE_ROOT", str(shared))
    monkeypatch.setattr(
        resolver,
        "_registered_worktree_roots",
        lambda repo_root=resolver.DEFAULT_REPO_ROOT: {repo.resolve(), shared.resolve()},
    )

    assert resolver._automation_state_root(repo) == (shared / ".aragora").resolve()


def test_automation_state_root_rejects_unregistered_same_origin_checkout(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    repo = tmp_path / "repo"
    shared = tmp_path / "shared-checkout"
    _init_repo_with_origin(repo)
    _init_repo_with_origin(shared)
    (shared / ".aragora").mkdir()
    monkeypatch.setenv("ARAGORA_AUTOMATION_STATE_ROOT", str(shared))
    monkeypatch.setattr(
        resolver,
        "_registered_worktree_roots",
        lambda repo_root=resolver.DEFAULT_REPO_ROOT: {repo.resolve()},
    )

    try:
        resolver._automation_state_root(repo)
    except ValueError as exc:
        assert "untrusted ARAGORA_AUTOMATION_STATE_ROOT" in str(exc)
        assert "registered worktree" in str(exc)
    else:
        raise AssertionError("unregistered same-origin automation state root was accepted")


def test_automation_state_root_rejects_different_origin_checkout(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    repo = tmp_path / "repo"
    shared = tmp_path / "shared-checkout"
    _init_repo_with_origin(repo)
    _init_repo_with_origin(shared, "https://github.com/elsewhere/other.git")
    (shared / ".aragora").mkdir()
    monkeypatch.setenv("ARAGORA_AUTOMATION_STATE_ROOT", str(shared))

    try:
        resolver._automation_state_root(repo)
    except ValueError as exc:
        assert "untrusted ARAGORA_AUTOMATION_STATE_ROOT" in str(exc)
    else:
        raise AssertionError("different-origin automation state root was accepted")


def test_automation_state_root_rejects_repo_subdirectory_bypass(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    repo = tmp_path / "repo"
    _init_repo_with_origin(repo)
    subdir = repo / "nested"
    (subdir / ".aragora").mkdir(parents=True)
    monkeypatch.setenv("ARAGORA_AUTOMATION_STATE_ROOT", str(subdir))
    monkeypatch.setattr(
        resolver,
        "_registered_worktree_roots",
        lambda repo_root=resolver.DEFAULT_REPO_ROOT: {repo.resolve()},
    )

    try:
        resolver._automation_state_root(repo)
    except ValueError as exc:
        assert "untrusted ARAGORA_AUTOMATION_STATE_ROOT" in str(exc)
    else:
        raise AssertionError("repo subdirectory automation state root was accepted")


def test_fetch_pr_state_rejects_unsafe_gh_bin(monkeypatch: Any) -> None:
    def run_should_not_execute(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("unsafe gh_bin must not reach subprocess")

    monkeypatch.setattr(resolver.subprocess, "run", run_should_not_execute)

    result = resolver._fetch_pr_state(pr=7435, gh_bin="python3 -c gh")

    assert result["available"] is False
    assert "gh_bin" in result["error"]
    assert result["command"] == []


def test_validate_gh_bin_accepts_absolute_gh_executable(tmp_path: Path) -> None:
    gh = tmp_path / "gh"
    gh.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    gh.chmod(0o755)

    assert resolver._validate_gh_bin(str(gh)) == str(gh.resolve())


def test_validate_gh_bin_accepts_absolute_executable_wrapper(tmp_path: Path) -> None:
    wrapper = tmp_path / "gh-wrapper"
    wrapper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    wrapper.chmod(0o755)

    assert resolver._validate_gh_bin(str(wrapper)) == str(wrapper.resolve())


def test_validate_gh_bin_accepts_relative_executable_wrapper(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    tools = tmp_path / "tools"
    tools.mkdir()
    wrapper = tools / "gh"
    wrapper.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    wrapper.chmod(0o755)
    monkeypatch.chdir(tmp_path)

    assert resolver._validate_gh_bin("./tools/gh") == str(wrapper.resolve())


def test_apply_marks_conflict_superseded_and_writes_receipt(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    receipt_dir = tmp_path / "receipts"
    registry.write_text(
        json.dumps(
            [
                {
                    "lane_id": "P104-ssd-cleanup-continuation",
                    "owner_session": "codex-P104",
                    "status": "conflict",
                    "conflict_session": "codex-R03",
                    "conflict_reason": "stale cleanup overlap",
                },
                {
                    "lane_id": "R03-post-p102-harvest-followthrough",
                    "owner_session": "codex-R03",
                    "status": "released",
                },
            ]
        ),
        encoding="utf-8",
    )

    result = resolver.resolve_conflicts(
        registry_path=registry,
        receipt_dir=receipt_dir,
        apply=True,
        resolved_at="2026-05-21T23:30:00Z",
    )

    rows = {row["lane_id"]: row for row in json.loads(registry.read_text(encoding="utf-8"))}
    assert result["resolved_count"] == 1
    assert rows["P104-ssd-cleanup-continuation"]["status"] == "superseded"
    receipts = sorted(receipt_dir.glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["schema_version"] == "aragora-lane-conflict-resolution/1.0"
    assert receipt["lane_id"] == "P104-ssd-cleanup-continuation"
    assert receipt["new_status"] == "superseded"


def test_apply_supersedes_only_exact_conflict_row(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    receipt_dir = tmp_path / "receipts"
    registry.write_text(
        json.dumps(
            [
                {
                    "lane_id": "shared-lane",
                    "owner_session": "codex-conflict-a",
                    "status": "conflict",
                    "conflict_session": "codex-done",
                },
                {
                    "lane_id": "shared-lane",
                    "owner_session": "codex-conflict-b",
                    "status": "conflict",
                    "conflict_session": "codex-unknown",
                },
                {
                    "lane_id": "done-lane",
                    "owner_session": "codex-done",
                    "status": "completed",
                },
            ]
        ),
        encoding="utf-8",
    )

    result = resolver.resolve_conflicts(
        registry_path=registry,
        receipt_dir=receipt_dir,
        apply=True,
        resolved_at="2026-05-21T23:45:00Z",
    )

    rows = json.loads(registry.read_text(encoding="utf-8"))
    by_owner = {row["owner_session"]: row for row in rows}
    assert result["resolved_count"] == 1
    assert result["unknown_session_count"] == 1
    assert by_owner["codex-conflict-a"]["status"] == "superseded"
    assert by_owner["codex-conflict-b"]["status"] == "conflict"


def test_concurrent_apply_preserves_registry_json(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    receipt_dir = tmp_path / "receipts"
    registry.write_text(
        json.dumps(
            [
                {
                    "lane_id": f"conflict-{idx:02d}",
                    "owner_session": f"codex-conflict-{idx:02d}",
                    "status": "conflict",
                    "conflict_session": f"codex-done-{idx:02d}",
                }
                for idx in range(8)
            ]
            + [
                {
                    "lane_id": f"done-{idx:02d}",
                    "owner_session": f"codex-done-{idx:02d}",
                    "status": "completed",
                }
                for idx in range(8)
            ]
        ),
        encoding="utf-8",
    )

    procs = [
        subprocess.Popen(
            [
                sys.executable,
                str(SCRIPT_PATH),
                "--apply",
                "--registry-path",
                str(registry),
                "--receipt-dir",
                str(receipt_dir),
                "--json",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _idx in range(4)
    ]
    results = [proc.communicate(timeout=30) + (proc.returncode,) for proc in procs]

    assert all(returncode == 0 for _stdout, _stderr, returncode in results), results
    payload = json.loads(registry.read_text(encoding="utf-8"))
    by_lane = {row["lane_id"]: row for row in payload}
    assert all(by_lane[f"conflict-{idx:02d}"]["status"] == "superseded" for idx in range(8))


def test_merged_pr_audit_reports_active_rows_without_mutating(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    receipt_dir = tmp_path / "receipts"
    _write_json(
        registry,
        [
            {
                "lane_id": "codex-7435-tier0-settlement",
                "owner_session": "codex-a",
                "status": "blocked",
                "branch": "worktree-queue-drain-final",
                "worktree": "/repo",
                "pr_number": 7435,
            },
            {
                "lane_id": "codex-7435-repair-settle",
                "owner_session": "codex-b",
                "status": "active",
                "branch": "worktree-queue-drain-final",
                "worktree": "/repo/.claude/worktrees/queue-drain-final",
                "pr_number": 7435,
            },
            {
                "lane_id": "done",
                "owner_session": "codex-done",
                "status": "completed",
                "pr_number": 7435,
            },
            {
                "lane_id": "missing-pr",
                "owner_session": "codex-missing-pr",
                "status": "active",
            },
        ],
    )
    gh = _fake_gh(
        tmp_path,
        {
            "number": 7435,
            "state": "MERGED",
            "headRefOid": "96ea60500851ac459aa542a0d31afc06d92c288a",
            "mergedAt": "2026-05-23T19:16:23Z",
            "mergeCommit": {"oid": "4e8b21e98a0ddbcb383d9c92e6c20b343e49d151"},
            "url": "https://github.com/synaptent/aragora/pull/7435",
        },
    )

    result = resolver.audit_merged_pr_lanes(
        registry_path=registry,
        receipt_dir=receipt_dir,
        pr=7435,
        gh_bin=str(gh),
        apply=False,
    )

    assert result["github_state"]["state"] == "MERGED"
    assert result["finding_count"] == 2
    assert result["requires_operator_authorization"] is True
    assert result["apply_eligible"] is False
    assert result["receipt_paths"] == []
    assert "send_operator_steering.py --to codex-a" in result["owner_steering_text"]
    assert (
        "claim_active_agent_lane.py --lane-id codex-7435-tier0-settlement"
        in result["owner_release_commands"][0]
    )
    rows = json.loads(registry.read_text(encoding="utf-8"))
    assert [row["status"] for row in rows] == ["blocked", "active", "completed", "active"]


def test_merged_pr_audit_ignores_open_and_unmerged_prs(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    _write_json(
        registry,
        [
            {
                "lane_id": "codex-open",
                "owner_session": "codex-open",
                "status": "active",
                "pr_number": 7441,
            },
            {
                "lane_id": "no-pr",
                "owner_session": "codex-no-pr",
                "status": "active",
            },
        ],
    )
    gh = _fake_gh(
        tmp_path,
        {
            "number": 7441,
            "state": "OPEN",
            "headRefOid": "abc",
            "mergedAt": None,
            "mergeCommit": None,
            "url": "https://github.com/synaptent/aragora/pull/7441",
        },
    )

    result = resolver.audit_merged_pr_lanes(
        registry_path=registry,
        receipt_dir=tmp_path / "receipts",
        pr=7441,
        gh_bin=str(gh),
        apply=False,
    )

    assert result["finding_count"] == 0
    assert result["github_state"]["state"] == "OPEN"
    assert result["apply_eligible"] is False
    assert result["blocked_reason"] == "pr_not_merged"


def test_merged_pr_audit_reports_github_state_unavailable(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    _write_json(
        registry,
        [
            {
                "lane_id": "codex-merged",
                "owner_session": "codex-owner",
                "status": "active",
                "pr_number": 7435,
            }
        ],
    )
    gh = _fake_gh(tmp_path, {}, exit_code=1)

    result = resolver.audit_merged_pr_lanes(
        registry_path=registry,
        receipt_dir=tmp_path / "receipts",
        pr=7435,
        gh_bin=str(gh),
        apply=False,
    )

    assert result["finding_count"] == 0
    assert result["github_state"]["available"] is False
    assert result["blocked_reason"] == "github_state_unavailable"


def test_closed_pr_audit_reports_active_rows_without_mutating(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    _write_json(
        registry,
        [
            {
                "lane_id": "codex-closed",
                "owner_session": "codex-owner",
                "status": "active",
                "pr_number": 7435,
            }
        ],
    )

    result = resolver.audit_merged_pr_lanes(
        registry_path=registry,
        receipt_dir=tmp_path / "receipts",
        pr=7435,
        gh_bin=str(_closed_pr_gh(tmp_path)),
        apply=False,
    )

    assert result["github_state"]["state"] == "CLOSED"
    assert result["github_state"]["closedAt"] == "2026-05-23T19:16:23Z"
    assert result["finding_count"] == 1
    assert result["apply_eligible"] is False
    assert result["blocked_reason"] is None
    assert "--expected-closed-at 2026-05-23T19:16:23Z" in result["operator_apply_command"]
    assert "--expected-head-sha head" in result["operator_apply_command"]
    assert json.loads(registry.read_text(encoding="utf-8"))[0]["status"] == "active"


def test_closed_pr_audit_apply_requires_expected_closed_at(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    _write_json(
        registry,
        [
            {
                "lane_id": "codex-closed",
                "owner_session": "codex-owner",
                "status": "active",
                "pr_number": 7435,
            }
        ],
    )

    result = resolver.audit_merged_pr_lanes(
        registry_path=registry,
        receipt_dir=tmp_path / "receipts",
        pr=7435,
        gh_bin=str(_closed_pr_gh(tmp_path)),
        apply=True,
        operator_authorized=True,
        heartbeat_path=tmp_path / "heartbeats.json",
        steering_inbox_root=tmp_path / "operator-steering",
    )

    assert result["resolved_count"] == 0
    assert result["blocked_reason"] == "expected_closed_at_required"
    assert json.loads(registry.read_text(encoding="utf-8"))[0]["status"] == "active"


def test_closed_pr_audit_apply_requires_expected_head_sha(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    _write_json(
        registry,
        [
            {
                "lane_id": "codex-closed",
                "owner_session": "codex-owner",
                "status": "active",
                "pr_number": 7435,
            }
        ],
    )

    result = resolver.audit_merged_pr_lanes(
        registry_path=registry,
        receipt_dir=tmp_path / "receipts",
        pr=7435,
        gh_bin=str(_closed_pr_gh(tmp_path)),
        apply=True,
        operator_authorized=True,
        expected_closed_at="2026-05-23T19:16:23Z",
        heartbeat_path=tmp_path / "heartbeats.json",
        steering_inbox_root=tmp_path / "operator-steering",
    )

    assert result["resolved_count"] == 0
    assert result["blocked_reason"] == "expected_head_sha_required"
    assert json.loads(registry.read_text(encoding="utf-8"))[0]["status"] == "active"


def test_closed_pr_audit_apply_rejects_closed_at_mismatch(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    _write_json(
        registry,
        [
            {
                "lane_id": "codex-closed",
                "owner_session": "codex-owner",
                "status": "active",
                "pr_number": 7435,
            }
        ],
    )

    result = resolver.audit_merged_pr_lanes(
        registry_path=registry,
        receipt_dir=tmp_path / "receipts",
        pr=7435,
        gh_bin=str(_closed_pr_gh(tmp_path, closed_at="2026-05-23T19:16:23Z")),
        apply=True,
        operator_authorized=True,
        expected_closed_at="2026-05-23T19:17:00Z",
        expected_head_sha="head",
        heartbeat_path=tmp_path / "heartbeats.json",
        steering_inbox_root=tmp_path / "operator-steering",
    )

    assert result["resolved_count"] == 0
    assert result["blocked_reason"] == "closed_at_mismatch"
    assert json.loads(registry.read_text(encoding="utf-8"))[0]["status"] == "active"


def test_closed_pr_audit_apply_rejects_head_sha_mismatch(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    _write_json(
        registry,
        [
            {
                "lane_id": "codex-closed",
                "owner_session": "codex-owner",
                "status": "active",
                "pr_number": 7435,
            }
        ],
    )

    result = resolver.audit_merged_pr_lanes(
        registry_path=registry,
        receipt_dir=tmp_path / "receipts",
        pr=7435,
        gh_bin=str(_closed_pr_gh(tmp_path, head_sha="actual-head")),
        apply=True,
        operator_authorized=True,
        expected_closed_at="2026-05-23T19:16:23Z",
        expected_head_sha="expected-head",
        heartbeat_path=tmp_path / "heartbeats.json",
        steering_inbox_root=tmp_path / "operator-steering",
    )

    assert result["resolved_count"] == 0
    assert result["blocked_reason"] == "head_sha_mismatch"
    assert json.loads(registry.read_text(encoding="utf-8"))[0]["status"] == "active"


def test_closed_pr_audit_apply_supersedes_safe_rows_with_closed_at_guard(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "lanes.json"
    receipt_dir = tmp_path / "receipts"
    _write_json(
        registry,
        [
            {
                "lane_id": "codex-closed",
                "owner_session": "codex-owner",
                "status": "active",
                "pr_number": 7435,
            }
        ],
    )

    result = resolver.audit_merged_pr_lanes(
        registry_path=registry,
        receipt_dir=receipt_dir,
        pr=7435,
        gh_bin=str(_closed_pr_gh(tmp_path, closed_at="2026-05-23T19:16:23Z")),
        apply=True,
        operator_authorized=True,
        expected_closed_at="2026-05-23T19:16:23Z",
        expected_head_sha="head",
        heartbeat_path=tmp_path / "heartbeats.json",
        steering_inbox_root=tmp_path / "operator-steering",
        resolved_at="2026-05-23T19:20:00Z",
    )

    rows = json.loads(registry.read_text(encoding="utf-8"))
    receipts = sorted(receipt_dir.glob("*.json"))
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert result["resolved_count"] == 1
    assert result["blocked_reason"] is None
    assert rows[0]["status"] == "superseded"
    assert receipt["schema_version"] == "aragora-merged-pr-lane-audit/1.0"
    assert receipt["closed_at"] == "2026-05-23T19:16:23Z"
    assert receipt["terminal_state"] == "CLOSED"


def test_merged_pr_audit_apply_requires_operator_authorization(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    _write_json(
        registry,
        [
            {
                "lane_id": "codex-merged",
                "owner_session": "codex-owner",
                "status": "active",
                "pr_number": 7435,
            }
        ],
    )
    gh = _fake_gh(
        tmp_path,
        {
            "number": 7435,
            "state": "MERGED",
            "headRefOid": "head",
            "mergedAt": "2026-05-23T19:16:23Z",
            "mergeCommit": {"oid": "merge"},
            "url": "https://github.com/synaptent/aragora/pull/7435",
        },
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--merged-pr-lane-audit",
            "--pr",
            "7435",
            "--apply",
            "--gh-bin",
            str(gh),
            "--registry-path",
            str(registry),
            "--receipt-dir",
            str(tmp_path / "receipts"),
            "--heartbeat-path",
            str(tmp_path / "heartbeats.json"),
            "--steering-inbox-root",
            str(tmp_path / "operator-steering"),
            "--json",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert proc.returncode == 2
    result = json.loads(proc.stdout)
    assert result["blocked_reason"] == "operator_authorization_required"
    assert json.loads(registry.read_text(encoding="utf-8"))[0]["status"] == "active"


def test_merged_pr_audit_apply_supersedes_only_target_pr_rows(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    receipt_dir = tmp_path / "receipts"
    _write_json(
        registry,
        [
            {
                "lane_id": "codex-7435-a",
                "owner_session": "codex-a",
                "status": "blocked",
                "next_action": "old",
                "pr_number": 7435,
            },
            {
                "lane_id": "codex-7435-b",
                "owner_session": "codex-b",
                "status": "active",
                "pr_number": 7435,
            },
            {
                "lane_id": "codex-7441",
                "owner_session": "codex-c",
                "status": "active",
                "pr_number": 7441,
            },
        ],
    )
    gh = _fake_gh(
        tmp_path,
        {
            "number": 7435,
            "state": "MERGED",
            "headRefOid": "96ea60500851ac459aa542a0d31afc06d92c288a",
            "mergedAt": "2026-05-23T19:16:23Z",
            "mergeCommit": {"oid": "4e8b21e98a0ddbcb383d9c92e6c20b343e49d151"},
            "url": "https://github.com/synaptent/aragora/pull/7435",
        },
    )

    result = resolver.audit_merged_pr_lanes(
        registry_path=registry,
        receipt_dir=receipt_dir,
        pr=7435,
        gh_bin=str(gh),
        apply=True,
        operator_authorized=True,
        expected_merge_commit="4e8b21e98a0ddbcb383d9c92e6c20b343e49d151",
        resolved_at="2026-05-23T19:20:00Z",
        heartbeat_path=tmp_path / "heartbeats.json",
        steering_inbox_root=tmp_path / "operator-steering",
    )

    rows = {row["lane_id"]: row for row in json.loads(registry.read_text(encoding="utf-8"))}
    assert result["resolved_count"] == 2
    assert rows["codex-7435-a"]["status"] == "superseded"
    assert rows["codex-7435-a"]["next_action"] == "old"
    assert rows["codex-7435-a"]["last_steering_outcome"] == "superseded"
    assert rows["codex-7441"]["status"] == "active"
    receipts = sorted(receipt_dir.glob("*.json"))
    assert len(receipts) == 2
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["schema_version"] == "aragora-merged-pr-lane-audit/1.0"
    assert receipt["pr_number"] == 7435
    assert receipt["merge_commit"] == "4e8b21e98a0ddbcb383d9c92e6c20b343e49d151"
    assert receipt["old_status"] in {"active", "blocked"}


def test_merged_pr_audit_apply_rejects_merge_commit_mismatch(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    _write_json(
        registry,
        [
            {
                "lane_id": "codex-merged",
                "owner_session": "codex-owner",
                "status": "active",
                "pr_number": 7435,
            }
        ],
    )
    gh = _fake_gh(
        tmp_path,
        {
            "number": 7435,
            "state": "MERGED",
            "headRefOid": "head",
            "mergedAt": "2026-05-23T19:16:23Z",
            "mergeCommit": {"oid": "actual"},
            "url": "https://github.com/synaptent/aragora/pull/7435",
        },
    )

    result = resolver.audit_merged_pr_lanes(
        registry_path=registry,
        receipt_dir=tmp_path / "receipts",
        pr=7435,
        gh_bin=str(gh),
        apply=True,
        operator_authorized=True,
        expected_merge_commit="expected",
    )

    assert result["resolved_count"] == 0
    assert result["blocked_reason"] == "merge_commit_mismatch"
    assert json.loads(registry.read_text(encoding="utf-8"))[0]["status"] == "active"


def test_merged_pr_audit_apply_rejects_fresh_heartbeat(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    heartbeats = tmp_path / "heartbeats.json"
    _write_json(
        registry,
        [
            {
                "lane_id": "codex-merged",
                "owner_session": "codex-owner",
                "status": "active",
                "pr_number": 7435,
            }
        ],
    )
    _write_json(
        heartbeats,
        [
            {
                "lane_id": "codex-merged",
                "owner_session": "codex-owner",
                "last_seen_at": "2026-05-23T19:19:30Z",
            }
        ],
    )

    result = resolver.audit_merged_pr_lanes(
        registry_path=registry,
        receipt_dir=tmp_path / "receipts",
        pr=7435,
        gh_bin=str(_merged_pr_gh(tmp_path)),
        apply=True,
        operator_authorized=True,
        expected_merge_commit="merge",
        resolved_at="2026-05-23T19:20:00Z",
        heartbeat_path=heartbeats,
        steering_inbox_root=tmp_path / "operator-steering",
    )

    assert result["resolved_count"] == 0
    assert result["blocked_reason"] == "unsafe_terminal_owner_gates"
    assert result["findings"][0]["terminal_safety_blockers"] == ["fresh_heartbeat"]
    assert json.loads(registry.read_text(encoding="utf-8"))[0]["status"] == "active"


def test_merged_pr_audit_apply_rejects_untrusted_heartbeat_state(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    heartbeats = tmp_path / "heartbeats.json"
    _write_json(
        registry,
        [
            {
                "lane_id": "codex-merged",
                "owner_session": "codex-owner",
                "status": "active",
                "pr_number": 7435,
            }
        ],
    )
    heartbeats.write_text("{not-json", encoding="utf-8")

    result = resolver.audit_merged_pr_lanes(
        registry_path=registry,
        receipt_dir=tmp_path / "receipts",
        pr=7435,
        gh_bin=str(_merged_pr_gh(tmp_path)),
        apply=True,
        operator_authorized=True,
        expected_merge_commit="merge",
        heartbeat_path=heartbeats,
        steering_inbox_root=tmp_path / "operator-steering",
    )

    assert result["resolved_count"] == 0
    assert result["blocked_reason"] == "unsafe_terminal_owner_gates"
    assert result["findings"][0]["terminal_safety_blockers"] == ["heartbeat_state_untrusted"]
    assert result["findings"][0]["terminal_safety_details"]["heartbeat_read_error"].startswith(
        "invalid_json:"
    )
    assert json.loads(registry.read_text(encoding="utf-8"))[0]["status"] == "active"


def test_merged_pr_audit_apply_command_preserves_safety_overrides(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    heartbeats = tmp_path / "heartbeats.json"
    steering_root = tmp_path / "operator-steering"
    _write_json(
        registry,
        [
            {
                "lane_id": "codex-merged",
                "owner_session": "codex-owner",
                "status": "active",
                "pr_number": 7435,
            }
        ],
    )
    _write_json(heartbeats, [])

    result = resolver.audit_merged_pr_lanes(
        registry_path=registry,
        receipt_dir=tmp_path / "receipts",
        pr=7435,
        gh_bin=str(_merged_pr_gh(tmp_path)),
        heartbeat_path=heartbeats,
        steering_inbox_root=steering_root,
        heartbeat_fresh_seconds=123,
    )

    command = result["operator_apply_command"]
    assert "--heartbeat-path" in command
    assert str(heartbeats) in command
    assert "--steering-inbox-root" in command
    assert str(steering_root) in command
    assert "--heartbeat-fresh-seconds 123" in command


def test_merged_pr_audit_apply_rejects_unread_mailbox(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    steering_root = tmp_path / "operator-steering"
    inbox = steering_root / "codex-owner"
    inbox.mkdir(parents=True)
    (inbox / "message.json").write_text("{}", encoding="utf-8")
    _write_json(
        registry,
        [
            {
                "lane_id": "codex-merged",
                "owner_session": "codex-owner",
                "status": "active",
                "pr_number": 7435,
            }
        ],
    )

    result = resolver.audit_merged_pr_lanes(
        registry_path=registry,
        receipt_dir=tmp_path / "receipts",
        pr=7435,
        gh_bin=str(_merged_pr_gh(tmp_path)),
        apply=True,
        operator_authorized=True,
        expected_merge_commit="merge",
        heartbeat_path=tmp_path / "heartbeats.json",
        steering_inbox_root=steering_root,
    )

    assert result["resolved_count"] == 0
    assert result["blocked_reason"] == "unsafe_terminal_owner_gates"
    assert result["findings"][0]["terminal_safety_blockers"] == ["unread_mailbox"]
    assert result["findings"][0]["terminal_safety_details"]["pending_mailbox_messages"] == [
        "message.json"
    ]
    assert json.loads(registry.read_text(encoding="utf-8"))[0]["status"] == "active"


def test_merged_pr_audit_apply_rejects_read_but_unacked_mailbox_message(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "lanes.json"
    steering_root = tmp_path / "operator-steering"
    inbox = steering_root / "codex-owner"
    inbox.mkdir(parents=True)
    (inbox / "message.json").write_text("{}", encoding="utf-8")
    _write_read_receipt(inbox, message_filename="message.json")
    _write_json(
        registry,
        [
            {
                "lane_id": "codex-merged",
                "owner_session": "codex-owner",
                "status": "active",
                "pr_number": 7435,
            }
        ],
    )

    result = resolver.audit_merged_pr_lanes(
        registry_path=registry,
        receipt_dir=tmp_path / "receipts",
        pr=7435,
        gh_bin=str(_merged_pr_gh(tmp_path)),
        apply=True,
        operator_authorized=True,
        expected_merge_commit="merge",
        heartbeat_path=tmp_path / "heartbeats.json",
        steering_inbox_root=steering_root,
    )

    assert result["resolved_count"] == 0
    assert result["blocked_reason"] == "unsafe_terminal_owner_gates"
    assert result["findings"][0]["terminal_safety_blockers"] == ["unread_mailbox"]
    assert result["findings"][0]["terminal_safety_details"]["pending_mailbox_messages"] == [
        "message.json"
    ]
    assert json.loads(registry.read_text(encoding="utf-8"))[0]["status"] == "active"


def test_merged_pr_audit_apply_labels_invalid_owner_session_separately(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    _write_json(
        registry,
        [
            {
                "lane_id": "codex-merged",
                "owner_session": "../codex-owner",
                "status": "active",
                "pr_number": 7435,
            }
        ],
    )

    result = resolver.audit_merged_pr_lanes(
        registry_path=registry,
        receipt_dir=tmp_path / "receipts",
        pr=7435,
        gh_bin=str(_merged_pr_gh(tmp_path)),
        apply=True,
        operator_authorized=True,
        expected_merge_commit="merge",
        heartbeat_path=tmp_path / "heartbeats.json",
        steering_inbox_root=tmp_path / "operator-steering",
    )

    assert result["resolved_count"] == 0
    assert result["blocked_reason"] == "unsafe_terminal_owner_gates"
    assert result["findings"][0]["terminal_safety_blockers"] == ["invalid_owner_session"]
    assert "pending_mailbox_messages" not in result["findings"][0]["terminal_safety_details"]
    assert json.loads(registry.read_text(encoding="utf-8"))[0]["status"] == "active"


def test_merged_pr_audit_apply_rejects_live_owner_process(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    _write_json(
        registry,
        [
            {
                "lane_id": "codex-merged",
                "owner_session": "codex-owner",
                "status": "active",
                "pid": os.getpid(),
                "pr_number": 7435,
            }
        ],
    )

    result = resolver.audit_merged_pr_lanes(
        registry_path=registry,
        receipt_dir=tmp_path / "receipts",
        pr=7435,
        gh_bin=str(_merged_pr_gh(tmp_path)),
        apply=True,
        operator_authorized=True,
        expected_merge_commit="merge",
        heartbeat_path=tmp_path / "heartbeats.json",
        steering_inbox_root=tmp_path / "operator-steering",
    )

    assert result["resolved_count"] == 0
    assert result["blocked_reason"] == "unsafe_terminal_owner_gates"
    assert result["findings"][0]["terminal_safety_blockers"] == ["live_process"]
    assert result["findings"][0]["terminal_safety_details"]["live_pids"] == [os.getpid()]
    assert json.loads(registry.read_text(encoding="utf-8"))[0]["status"] == "active"


def test_merged_pr_audit_apply_rejects_local_work_claim(tmp_path: Path) -> None:
    registry = tmp_path / "lanes.json"
    _write_json(
        registry,
        [
            {
                "lane_id": "codex-merged",
                "owner_session": "codex-owner",
                "status": "active",
                "worktree": "/tmp/active-worktree",
                "possible_unpushed_work": True,
                "pr_number": 7435,
            }
        ],
    )

    result = resolver.audit_merged_pr_lanes(
        registry_path=registry,
        receipt_dir=tmp_path / "receipts",
        pr=7435,
        gh_bin=str(_merged_pr_gh(tmp_path)),
        apply=True,
        operator_authorized=True,
        expected_merge_commit="merge",
        heartbeat_path=tmp_path / "heartbeats.json",
        steering_inbox_root=tmp_path / "operator-steering",
    )

    assert result["resolved_count"] == 0
    assert result["blocked_reason"] == "unsafe_terminal_owner_gates"
    assert result["findings"][0]["terminal_safety_blockers"] == ["local_work_claim"]
    assert result["findings"][0]["terminal_safety_details"]["local_work_claims"] == [
        "/tmp/active-worktree"
    ]
    assert json.loads(registry.read_text(encoding="utf-8"))[0]["status"] == "active"


def test_merged_pr_audit_apply_allows_recorded_worktree_without_local_work_risk(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "lanes.json"
    _write_json(
        registry,
        [
            {
                "lane_id": "codex-merged",
                "owner_session": "codex-owner",
                "status": "active",
                "worktree": "/tmp/active-worktree",
                "possible_unpushed_work": "false",
                "has_unpushed_work": False,
                "pr_number": 7435,
            }
        ],
    )

    result = resolver.audit_merged_pr_lanes(
        registry_path=registry,
        receipt_dir=tmp_path / "receipts",
        pr=7435,
        gh_bin=str(_merged_pr_gh(tmp_path)),
        apply=True,
        operator_authorized=True,
        expected_merge_commit="merge",
        heartbeat_path=tmp_path / "heartbeats.json",
        steering_inbox_root=tmp_path / "operator-steering",
    )

    assert result["resolved_count"] == 1
    assert result["blocked_reason"] is None
    rows = json.loads(registry.read_text(encoding="utf-8"))
    assert rows[0]["status"] == "superseded"


def test_merged_pr_audit_apply_supersedes_safe_rows_when_other_rows_are_unsafe(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "lanes.json"
    steering_root = tmp_path / "operator-steering"
    inbox = steering_root / "codex-blocked"
    inbox.mkdir(parents=True)
    (inbox / "message.json").write_text("{}", encoding="utf-8")
    _write_json(
        registry,
        [
            {
                "lane_id": "codex-safe",
                "owner_session": "codex-safe",
                "status": "active",
                "pr_number": 7435,
            },
            {
                "lane_id": "codex-blocked",
                "owner_session": "codex-blocked",
                "status": "active",
                "pr_number": 7435,
            },
        ],
    )

    result = resolver.audit_merged_pr_lanes(
        registry_path=registry,
        receipt_dir=tmp_path / "receipts",
        pr=7435,
        gh_bin=str(_merged_pr_gh(tmp_path)),
        apply=True,
        operator_authorized=True,
        expected_merge_commit="merge",
        heartbeat_path=tmp_path / "heartbeats.json",
        steering_inbox_root=steering_root,
    )

    rows = {row["lane_id"]: row for row in json.loads(registry.read_text(encoding="utf-8"))}
    assert result["resolved_count"] == 1
    assert result["safe_finding_count"] == 1
    assert result["unsafe_finding_count"] == 1
    assert rows["codex-safe"]["status"] == "superseded"
    assert rows["codex-blocked"]["status"] == "active"


def test_merged_pr_audit_apply_rejects_negative_heartbeat_fresh_seconds(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "lanes.json"
    _write_json(
        registry,
        [
            {
                "lane_id": "codex-merged",
                "owner_session": "codex-owner",
                "status": "active",
                "pr_number": 7435,
            }
        ],
    )

    result = resolver.audit_merged_pr_lanes(
        registry_path=registry,
        receipt_dir=tmp_path / "receipts",
        pr=7435,
        gh_bin=str(_merged_pr_gh(tmp_path)),
        apply=True,
        operator_authorized=True,
        expected_merge_commit="merge",
        heartbeat_path=tmp_path / "heartbeats.json",
        steering_inbox_root=tmp_path / "operator-steering",
        heartbeat_fresh_seconds=-1,
    )

    assert result["resolved_count"] == 0
    assert result["apply_eligible"] is False
    assert result["blocked_reason"] == "invalid_heartbeat_fresh_seconds"
    assert result["operator_apply_command"] == ""
    assert result["findings"][0]["terminal_safety_blockers"] == ["invalid_heartbeat_fresh_seconds"]
    assert json.loads(registry.read_text(encoding="utf-8"))[0]["status"] == "active"
