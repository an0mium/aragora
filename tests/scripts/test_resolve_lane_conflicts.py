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


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fake_gh(tmp_path: Path, payload: dict[str, Any], *, exit_code: int = 0) -> Path:
    gh = tmp_path / "fake-gh"
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

    gh_closed = _fake_gh(
        tmp_path,
        {
            "number": 7441,
            "state": "CLOSED",
            "headRefOid": "abc",
            "mergedAt": None,
            "mergeCommit": None,
            "url": "https://github.com/synaptent/aragora/pull/7441",
        },
    )

    closed_result = resolver.audit_merged_pr_lanes(
        registry_path=registry,
        receipt_dir=tmp_path / "receipts",
        pr=7441,
        gh_bin=str(gh_closed),
        apply=False,
    )

    assert closed_result["finding_count"] == 0
    assert closed_result["github_state"]["state"] == "CLOSED"
    assert closed_result["blocked_reason"] == "pr_not_merged"


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
