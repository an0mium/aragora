from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scripts.github_cli_health import GitHubCLIHealth

import scripts.audit_codex_branch_backlog as mod


def _branch_row(
    name: str = "codex/example",
    *,
    committed_at: datetime | None = None,
    ahead_count: str = "1",
    behind_count: str = "0",
    head_sha: str = "abc1234",
) -> dict[str, str]:
    return {
        "name": name,
        "upstream": "",
        "head_sha": head_sha,
        "committed_at": (committed_at or datetime.now(timezone.utc)).isoformat(),
        "ahead_count": ahead_count,
        "behind_count": behind_count,
        "subject": "test branch",
    }


def _stub_git_inventory(monkeypatch: Any, row: dict[str, str]) -> None:
    monkeypatch.setattr(mod, "local_branches", lambda _root, _prefix, _base: [row])
    monkeypatch.setattr(mod, "remote_branch_names", lambda _root, _prefix: set())
    monkeypatch.setattr(mod, "merged_branch_names", lambda _root, _base, _prefix: set())
    monkeypatch.setattr(mod, "worktree_map", lambda _root: {})


def test_local_branches_defers_expensive_divergence_lookup(
    tmp_path: Path, monkeypatch: Any
) -> None:
    calls: list[list[str]] = []

    def fake_run_git(
        args: list[str], cwd: Path, *, timeout: int = 60
    ) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="codex/example||abc1234|2026-04-30 00:00:00 +0000|test branch\n",
            stderr="",
        )

    monkeypatch.setattr(mod, "run_git", fake_run_git)

    rows = mod.local_branches(tmp_path, "codex/", "origin/main")

    assert "%(ahead-behind:" not in calls[0][1]
    assert rows[0]["ahead_count"] == ""
    assert rows[0]["behind_count"] == ""


def test_branch_divergence_parses_rev_list_left_right_counts(
    tmp_path: Path, monkeypatch: Any
) -> None:
    def fake_run_git(
        args: list[str], cwd: Path, *, timeout: int = 60
    ) -> subprocess.CompletedProcess[str]:
        assert args == ["rev-list", "--left-right", "--count", "origin/main...codex/example"]
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="3\t7\n", stderr="")

    monkeypatch.setattr(mod, "run_git", fake_run_git)

    assert mod.branch_divergence(tmp_path, "origin/main", "codex/example") == (7, 3)


def test_summary_only_payload_omits_records_without_mutating_source() -> None:
    payload = {
        "branch_count": 2,
        "summary": {"publishable_branch_backlog": 0},
        "records": [{"name": "codex/one"}, {"name": "codex/two"}],
    }

    compact = mod.summary_only_payload(payload)

    assert compact["branch_count"] == 2
    assert compact["summary"] == {"publishable_branch_backlog": 0}
    assert compact["records"] == []
    assert compact["records_omitted"] is True
    assert payload["records"] == [{"name": "codex/one"}, {"name": "codex/two"}]


def test_audit_skips_open_pr_lookup_when_github_health_degraded(
    tmp_path: Path, monkeypatch: Any
) -> None:
    _stub_git_inventory(monkeypatch, _branch_row())
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda _root: GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode="connectivity_failed",
            error="offline",
            repo=str(tmp_path),
        ),
    )

    def fail_open_pr_lookup(*_args: Any, **_kwargs: Any) -> dict[str, int]:
        raise AssertionError("open PR lookup should be skipped when GitHub is unhealthy")

    monkeypatch.setattr(mod, "open_pr_heads", fail_open_pr_lookup)

    payload = mod.audit(
        root=tmp_path,
        base="origin/main",
        repo="synaptent/aragora",
        prefix="codex/",
        recent_hours=72,
        max_branches=None,
        include_patch_equivalence=False,
        publisher_backlog_limit=12,
    )

    assert payload["github_health"]["mode"] == "connectivity_failed"
    assert payload["open_pr_lookup_skipped"] is True
    assert payload["records"][0]["open_pr"] is None
    assert payload["records"][0]["category"] == "salvage_recent_unique"


def test_audit_uses_open_pr_lookup_when_github_health_is_ready(
    tmp_path: Path, monkeypatch: Any
) -> None:
    row = _branch_row("codex/has-pr")
    _stub_git_inventory(monkeypatch, row)
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda _root: GitHubCLIHealth(
            ready=True,
            auth_ok=True,
            api_ok=True,
            mode="ready",
            error="",
            repo=str(tmp_path),
        ),
    )
    monkeypatch.setattr(mod, "open_pr_heads", lambda _root, _repo, _prefix: {"codex/has-pr": 6500})

    payload = mod.audit(
        root=tmp_path,
        base="origin/main",
        repo="synaptent/aragora",
        prefix="codex/",
        recent_hours=72,
        max_branches=None,
        include_patch_equivalence=False,
        publisher_backlog_limit=12,
    )

    assert payload["github_health"]["ready"] is True
    assert payload["open_pr_lookup_skipped"] is False
    assert payload["records"][0]["open_pr"] == 6500
    assert payload["records"][0]["category"] == "protected_open_pr"


def test_audit_ignores_missing_worktree_paths(tmp_path: Path, monkeypatch: Any) -> None:
    row = _branch_row("codex/stale-worktree")
    missing_worktree = tmp_path / "missing-worktree"
    monkeypatch.setattr(mod, "local_branches", lambda _root, _prefix, _base: [row])
    monkeypatch.setattr(mod, "remote_branch_names", lambda _root, _prefix: set())
    monkeypatch.setattr(mod, "merged_branch_names", lambda _root, _base, _prefix: set())
    monkeypatch.setattr(
        mod, "worktree_map", lambda _root: {"codex/stale-worktree": [missing_worktree]}
    )
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda _root: GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode="connectivity_failed",
            error="offline",
            repo=str(tmp_path),
        ),
    )

    payload = mod.audit(
        root=tmp_path,
        base="origin/main",
        repo="synaptent/aragora",
        prefix="codex/",
        recent_hours=72,
        max_branches=None,
        include_patch_equivalence=False,
        publisher_backlog_limit=1,
    )

    assert payload["records"][0]["worktree_paths"] == [str(missing_worktree)]
    assert payload["records"][0]["dirty_worktree_paths"] == []
    assert payload["records"][0]["category"] == "salvage_recent_unique"


def test_audit_publishable_backlog_excludes_stale_local_only_branches(
    tmp_path: Path, monkeypatch: Any
) -> None:
    now = datetime.now(timezone.utc)
    rows = [
        _branch_row("codex/recent-local", committed_at=now),
        _branch_row("codex/stale-local", committed_at=now.replace(year=now.year - 1)),
        _branch_row("codex/stale-remote", committed_at=now.replace(year=now.year - 1)),
    ]
    monkeypatch.setattr(mod, "local_branches", lambda _root, _prefix, _base: rows)
    monkeypatch.setattr(mod, "remote_branch_names", lambda _root, _prefix: {"codex/stale-remote"})
    monkeypatch.setattr(mod, "merged_branch_names", lambda _root, _base, _prefix: set())
    monkeypatch.setattr(mod, "worktree_map", lambda _root: {})
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda _root: GitHubCLIHealth(
            ready=True,
            auth_ok=True,
            api_ok=True,
            mode="ready",
            error="",
            repo=str(tmp_path),
        ),
    )
    monkeypatch.setattr(mod, "open_pr_heads", lambda _root, _repo, _prefix: {})

    payload = mod.audit(
        root=tmp_path,
        base="origin/main",
        repo="synaptent/aragora",
        prefix="codex/",
        recent_hours=72,
        max_branches=None,
        include_patch_equivalence=False,
        publisher_backlog_limit=3,
    )

    by_category = payload["summary"]["by_category"]
    assert by_category["salvage_recent_unique"] == 1
    assert by_category["salvage_stale_remote_unique"] == 1
    assert by_category["salvage_stale_local_unique"] == 1
    assert payload["summary"]["salvage_candidates"] == 3
    assert payload["summary"]["publishable_branch_backlog"] == 2
    assert payload["summary"]["stale_local_only_salvage_candidates"] == 1
    assert payload["summary"]["writer_should_pause_for_branch_backlog"] is False


def test_audit_excludes_diverged_branches_from_publishable_backlog(
    tmp_path: Path, monkeypatch: Any
) -> None:
    now = datetime.now(timezone.utc)
    rows = [
        _branch_row("codex/recent-ready", committed_at=now),
        _branch_row("codex/recent-diverged", committed_at=now, behind_count="12"),
        _branch_row(
            "codex/stale-remote-diverged",
            committed_at=now.replace(year=now.year - 1),
            behind_count="4",
        ),
    ]
    monkeypatch.setattr(mod, "local_branches", lambda _root, _prefix, _base: rows)
    monkeypatch.setattr(
        mod, "remote_branch_names", lambda _root, _prefix: {"codex/stale-remote-diverged"}
    )
    monkeypatch.setattr(mod, "merged_branch_names", lambda _root, _base, _prefix: set())
    monkeypatch.setattr(mod, "worktree_map", lambda _root: {})
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda _root: GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode="connectivity_failed",
            error="offline",
            repo=str(tmp_path),
        ),
    )

    payload = mod.audit(
        root=tmp_path,
        base="origin/main",
        repo="synaptent/aragora",
        prefix="codex/",
        recent_hours=72,
        max_branches=None,
        include_patch_equivalence=False,
        publisher_backlog_limit=2,
    )

    by_category = payload["summary"]["by_category"]
    assert by_category["salvage_recent_unique"] == 1
    assert by_category["salvage_diverged_recent"] == 1
    assert by_category["salvage_diverged_remote"] == 1
    assert payload["summary"]["salvage_candidates"] == 3
    assert payload["summary"]["publishable_branch_backlog"] == 1
    assert payload["summary"]["diverged_salvage_candidates"] == 2
    assert payload["summary"]["writer_should_pause_for_branch_backlog"] is False
    diverged = next(
        record for record in payload["records"] if record["name"] == "codex/recent-diverged"
    )
    assert diverged["behind_count"] == 12
    assert diverged["category"] == "salvage_diverged_recent"


def test_audit_excludes_terminal_outbox_receipts_from_publishable_backlog(
    tmp_path: Path, monkeypatch: Any
) -> None:
    now = datetime.now(timezone.utc)
    rows = [
        _branch_row("codex/receipted", committed_at=now),
        _branch_row("codex/stale-remote", committed_at=now.replace(year=now.year - 1)),
    ]
    outbox = tmp_path / ".aragora" / "automation-outbox"
    receipts = tmp_path / ".aragora" / "automation-receipts"
    outbox.mkdir(parents=True)
    receipts.mkdir(parents=True)
    key = "open-pr-codex-receipted-abc123"
    (outbox / "receipted.json").write_text(
        json.dumps(
            {
                "task": "Publish receipted branch",
                "requires_github": True,
                "requested_action": "open_pr",
                "repo": "synaptent/aragora",
                "local_evidence": {"branch": "codex/receipted", "head": "abc123"},
                "validation": ["pytest tests/example.py -q"],
                "idempotency_key": key,
                "created_at": "2026-04-24T16:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (receipts / f"{key}.json").write_text(
        json.dumps({"idempotency_key": key, "status": "published"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "local_branches", lambda _root, _prefix, _base: rows)
    monkeypatch.setattr(mod, "remote_branch_names", lambda _root, _prefix: {"codex/stale-remote"})
    monkeypatch.setattr(mod, "merged_branch_names", lambda _root, _base, _prefix: set())
    monkeypatch.setattr(mod, "worktree_map", lambda _root: {})
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda _root: GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode="connectivity_failed",
            error="offline",
            repo=str(tmp_path),
        ),
    )

    payload = mod.audit(
        root=tmp_path,
        base="origin/main",
        repo="synaptent/aragora",
        prefix="codex/",
        recent_hours=72,
        max_branches=None,
        include_patch_equivalence=False,
        publisher_backlog_limit=2,
    )

    by_category = payload["summary"]["by_category"]
    assert by_category["protected_handoff_receipt"] == 1
    assert by_category["salvage_stale_remote_unique"] == 1
    assert payload["summary"]["protected"] == 1
    assert payload["summary"]["publishable_branch_backlog"] == 1
    assert payload["summary"]["handoff_receipted_branches"] == 1
    assert payload["summary"]["writer_should_pause_for_branch_backlog"] is False
    receipted = next(record for record in payload["records"] if record["name"] == "codex/receipted")
    assert receipted["handoff_receipt_exists"] is True
    assert receipted["category"] == "protected_handoff_receipt"


def test_audit_excludes_terminal_receipt_branch_without_outbox_payload(
    tmp_path: Path, monkeypatch: Any
) -> None:
    now = datetime.now(timezone.utc)
    rows = [
        _branch_row("codex/receipted", committed_at=now),
        _branch_row("codex/new-work", committed_at=now),
    ]
    receipts = tmp_path / ".aragora" / "automation-receipts"
    receipts.mkdir(parents=True)
    key = "open-pr-codex-receipted-abc123"
    (receipts / f"{key}.json").write_text(
        json.dumps(
            {
                "branch": "codex/receipted",
                "idempotency_key": key,
                "status": "already_satisfied",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "local_branches", lambda _root, _prefix, _base: rows)
    monkeypatch.setattr(mod, "remote_branch_names", lambda _root, _prefix: set())
    monkeypatch.setattr(mod, "merged_branch_names", lambda _root, _base, _prefix: set())
    monkeypatch.setattr(mod, "worktree_map", lambda _root: {})
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda _root: GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode="connectivity_failed",
            error="offline",
            repo=str(tmp_path),
        ),
    )

    payload = mod.audit(
        root=tmp_path,
        base="origin/main",
        repo="synaptent/aragora",
        prefix="codex/",
        recent_hours=72,
        max_branches=None,
        include_patch_equivalence=False,
        publisher_backlog_limit=2,
    )

    by_category = payload["summary"]["by_category"]
    assert by_category["protected_handoff_receipt"] == 1
    assert by_category["salvage_recent_unique"] == 1
    assert payload["summary"]["handoff_receipted_branches"] == 1
    assert payload["summary"]["publishable_branch_backlog"] == 1
    receipted = next(record for record in payload["records"] if record["name"] == "codex/receipted")
    assert receipted["handoff_receipt_exists"] is True
    assert receipted["category"] == "protected_handoff_receipt"


def test_audit_matches_terminal_receipt_by_idempotency_key_when_outbox_missing(
    tmp_path: Path, monkeypatch: Any
) -> None:
    now = datetime.now(timezone.utc)
    rows = [
        _branch_row(
            "codex/gpt-55-model-migration",
            committed_at=now,
            behind_count="143",
            head_sha="ab3db55f3",
        ),
        _branch_row("codex/new-work", committed_at=now),
    ]
    receipts = tmp_path / ".aragora" / "automation-receipts"
    receipts.mkdir(parents=True)
    key = "open-pr-codex-gpt-55-model-migration-ab3db55f341e"
    (receipts / f"{key}.json").write_text(
        json.dumps(
            {
                "idempotency_key": key,
                "status": "published",
                "task": "Open PR for repaired GPT-5.5 model default migration branch",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "local_branches", lambda _root, _prefix, _base: rows)
    monkeypatch.setattr(mod, "remote_branch_names", lambda _root, _prefix: set())
    monkeypatch.setattr(mod, "merged_branch_names", lambda _root, _base, _prefix: set())
    monkeypatch.setattr(mod, "worktree_map", lambda _root: {})
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda _root: GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode="connectivity_failed",
            error="offline",
            repo=str(tmp_path),
        ),
    )

    payload = mod.audit(
        root=tmp_path,
        base="origin/main",
        repo="synaptent/aragora",
        prefix="codex/",
        recent_hours=72,
        max_branches=None,
        include_patch_equivalence=False,
        publisher_backlog_limit=2,
    )

    receipted = next(
        record for record in payload["records"] if record["name"] == "codex/gpt-55-model-migration"
    )
    assert receipted["handoff_receipt_exists"] is True
    assert receipted["category"] == "protected_handoff_receipt"
    assert payload["summary"]["handoff_receipted_branches"] == 1
    assert payload["summary"]["publishable_branch_backlog"] == 1


def test_audit_excludes_unresolved_outbox_handoffs_from_publishable_backlog(
    tmp_path: Path, monkeypatch: Any
) -> None:
    now = datetime.now(timezone.utc)
    rows = [
        _branch_row("codex/handed-off", committed_at=now),
        _branch_row("codex/new-work", committed_at=now),
    ]
    outbox = tmp_path / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    (outbox / "handed-off.json").write_text(
        json.dumps(
            {
                "task": "Publish handed-off branch",
                "requires_github": True,
                "requested_action": "open_pr",
                "repo": "synaptent/aragora",
                "local_evidence": {"branch": "codex/handed-off", "head": "abc123"},
                "validation": ["pytest tests/example.py -q"],
                "idempotency_key": "open-pr-codex-handed-off-abc123",
                "created_at": "2026-04-24T16:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "local_branches", lambda _root, _prefix, _base: rows)
    monkeypatch.setattr(mod, "remote_branch_names", lambda _root, _prefix: set())
    monkeypatch.setattr(mod, "merged_branch_names", lambda _root, _base, _prefix: set())
    monkeypatch.setattr(mod, "worktree_map", lambda _root: {})
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda _root: GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode="connectivity_failed",
            error="offline",
            repo=str(tmp_path),
        ),
    )

    payload = mod.audit(
        root=tmp_path,
        base="origin/main",
        repo="synaptent/aragora",
        prefix="codex/",
        recent_hours=72,
        max_branches=None,
        include_patch_equivalence=False,
        publisher_backlog_limit=2,
    )

    by_category = payload["summary"]["by_category"]
    assert by_category["protected_handoff_outbox"] == 1
    assert by_category["salvage_recent_unique"] == 1
    assert payload["summary"]["protected"] == 1
    assert payload["summary"]["publishable_branch_backlog"] == 1
    assert payload["summary"]["handoff_outbox_branches"] == 1
    assert payload["summary"]["writer_should_pause_for_branch_backlog"] is False
    handed_off = next(
        record for record in payload["records"] if record["name"] == "codex/handed-off"
    )
    assert handed_off["handoff_outbox_exists"] is True
    assert handed_off["category"] == "protected_handoff_outbox"


def test_audit_protects_list_local_evidence_branch_handoffs(
    tmp_path: Path, monkeypatch: Any
) -> None:
    now = datetime.now(timezone.utc)
    rows = [
        _branch_row("codex/list-evidence", committed_at=now),
        _branch_row("codex/new-work", committed_at=now),
    ]
    outbox = tmp_path / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    (outbox / "list-evidence.json").write_text(
        json.dumps(
            {
                "task": "Publish list-evidence branch",
                "requires_github": True,
                "requested_action": "open_pr",
                "repo": "synaptent/aragora",
                "local_evidence": [
                    "legacy note",
                    {"branch": "codex/list-evidence", "head": "abc123"},
                ],
                "validation": ["pytest tests/example.py -q"],
                "idempotency_key": "open-pr-codex-list-evidence-abc123",
                "created_at": "2026-05-01T10:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "local_branches", lambda _root, _prefix, _base: rows)
    monkeypatch.setattr(mod, "remote_branch_names", lambda _root, _prefix: set())
    monkeypatch.setattr(mod, "merged_branch_names", lambda _root, _base, _prefix: set())
    monkeypatch.setattr(mod, "worktree_map", lambda _root: {})
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda _root: GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode="connectivity_failed",
            error="offline",
            repo=str(tmp_path),
        ),
    )

    payload = mod.audit(
        root=tmp_path,
        base="origin/main",
        repo="synaptent/aragora",
        prefix="codex/",
        recent_hours=72,
        max_branches=None,
        include_patch_equivalence=False,
        publisher_backlog_limit=2,
    )

    list_evidence = next(
        record for record in payload["records"] if record["name"] == "codex/list-evidence"
    )
    assert list_evidence["handoff_outbox_exists"] is True
    assert list_evidence["category"] == "protected_handoff_outbox"
    assert payload["summary"]["handoff_outbox_branches"] == 1
    assert payload["summary"]["publishable_branch_backlog"] == 1


def test_audit_protects_structured_action_branch_handoffs(tmp_path: Path, monkeypatch: Any) -> None:
    now = datetime.now(timezone.utc)
    rows = [
        _branch_row("codex/structured-action", committed_at=now),
        _branch_row("codex/new-work", committed_at=now),
    ]
    outbox = tmp_path / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    (outbox / "structured-action.json").write_text(
        json.dumps(
            {
                "task": "Publish structured action branch",
                "requires_github": True,
                "requested_action": {
                    "type": "open_pull_request",
                    "branch": "codex/structured-action",
                    "base": "main",
                },
                "repo": "synaptent/aragora",
                "local_evidence": {},
                "validation": ["pytest tests/example.py -q"],
                "idempotency_key": "open-pr-codex-structured-action-abc123",
                "created_at": "2026-04-27T10:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "local_branches", lambda _root, _prefix, _base: rows)
    monkeypatch.setattr(mod, "remote_branch_names", lambda _root, _prefix: set())
    monkeypatch.setattr(mod, "merged_branch_names", lambda _root, _base, _prefix: set())
    monkeypatch.setattr(mod, "worktree_map", lambda _root: {})
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda _root: GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode="connectivity_failed",
            error="offline",
            repo=str(tmp_path),
        ),
    )

    payload = mod.audit(
        root=tmp_path,
        base="origin/main",
        repo="synaptent/aragora",
        prefix="codex/",
        recent_hours=72,
        max_branches=None,
        include_patch_equivalence=False,
        publisher_backlog_limit=2,
    )

    structured = next(
        record for record in payload["records"] if record["name"] == "codex/structured-action"
    )
    assert structured["handoff_outbox_exists"] is True
    assert structured["category"] == "protected_handoff_outbox"
    assert payload["summary"]["handoff_outbox_branches"] == 1
    assert payload["summary"]["publishable_branch_backlog"] == 1


def test_audit_protects_json_string_action_branch_handoffs(
    tmp_path: Path, monkeypatch: Any
) -> None:
    now = datetime.now(timezone.utc)
    rows = [
        _branch_row("codex/json-action", committed_at=now),
        _branch_row("codex/new-work", committed_at=now),
    ]
    outbox = tmp_path / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    (outbox / "json-action.json").write_text(
        json.dumps(
            {
                "task": "Publish JSON-string action branch",
                "requires_github": True,
                "requested_action": json.dumps(
                    {
                        "type": "push_branch_and_open_pr",
                        "branch": "codex/json-action",
                        "requires_github": True,
                    }
                ),
                "repo": "synaptent/aragora",
                "local_evidence": {},
                "validation": ["pytest tests/example.py -q"],
                "idempotency_key": "open-pr-codex-json-action-abc123",
                "created_at": "2026-04-29T12:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "local_branches", lambda _root, _prefix, _base: rows)
    monkeypatch.setattr(mod, "remote_branch_names", lambda _root, _prefix: set())
    monkeypatch.setattr(mod, "merged_branch_names", lambda _root, _base, _prefix: set())
    monkeypatch.setattr(mod, "worktree_map", lambda _root: {})
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda _root: GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode="connectivity_failed",
            error="offline",
            repo=str(tmp_path),
        ),
    )

    payload = mod.audit(
        root=tmp_path,
        base="origin/main",
        repo="synaptent/aragora",
        prefix="codex/",
        recent_hours=72,
        max_branches=None,
        include_patch_equivalence=False,
        publisher_backlog_limit=2,
    )

    by_category = payload["summary"]["by_category"]
    assert by_category["protected_handoff_outbox"] == 1
    assert by_category["salvage_recent_unique"] == 1
    protected = next(
        record for record in payload["records"] if record["name"] == "codex/json-action"
    )
    assert protected["handoff_outbox_exists"] is True
    assert protected["category"] == "protected_handoff_outbox"


def test_audit_protects_patch_equivalent_unresolved_handoff_branches(
    tmp_path: Path, monkeypatch: Any
) -> None:
    now = datetime.now(timezone.utc)
    rows = [
        _branch_row("codex/refreshed", committed_at=now),
        _branch_row("codex/stale-copy", committed_at=now),
        _branch_row("codex/new-work", committed_at=now),
    ]
    outbox = tmp_path / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    (outbox / "refreshed.json").write_text(
        json.dumps(
            {
                "task": "Publish refreshed branch",
                "requires_github": True,
                "requested_action": "open_pr",
                "repo": "synaptent/aragora",
                "local_evidence": {"branch": "codex/refreshed", "head": "abc123"},
                "validation": ["pytest tests/example.py -q"],
                "idempotency_key": "open-pr-codex-refreshed-abc123",
                "created_at": "2026-04-24T16:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "local_branches", lambda _root, _prefix, _base: rows)
    monkeypatch.setattr(mod, "remote_branch_names", lambda _root, _prefix: set())
    monkeypatch.setattr(mod, "merged_branch_names", lambda _root, _base, _prefix: set())
    monkeypatch.setattr(mod, "worktree_map", lambda _root: {})
    monkeypatch.setattr(
        mod,
        "is_patch_equivalent",
        lambda _root, _base, _branch, **_kwargs: False,
    )
    monkeypatch.setattr(
        mod,
        "branch_patch_id",
        lambda _root, _base, branch, **_kwargs: {
            "codex/refreshed": "same-patch",
            "codex/stale-copy": "same-patch",
            "codex/new-work": "new-patch",
        }[branch],
    )
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda _root: GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode="connectivity_failed",
            error="offline",
            repo=str(tmp_path),
        ),
    )

    payload = mod.audit(
        root=tmp_path,
        base="origin/main",
        repo="synaptent/aragora",
        prefix="codex/",
        recent_hours=72,
        max_branches=None,
        include_patch_equivalence=True,
        publisher_backlog_limit=2,
    )

    by_category = payload["summary"]["by_category"]
    assert by_category["protected_handoff_outbox"] == 2
    assert by_category["salvage_recent_unique"] == 1
    assert payload["summary"]["publishable_branch_backlog"] == 1
    assert payload["summary"]["writer_should_pause_for_branch_backlog"] is False
    stale_copy = next(
        record for record in payload["records"] if record["name"] == "codex/stale-copy"
    )
    assert stale_copy["handoff_outbox_exists"] is True
    assert stale_copy["category"] == "protected_handoff_outbox"


def test_audit_treats_superseded_branch_as_unresolved_handoff(
    tmp_path: Path, monkeypatch: Any
) -> None:
    now = datetime.now(timezone.utc)
    rows = [
        _branch_row("codex/stale-original", committed_at=now),
        _branch_row("codex/new-work", committed_at=now),
    ]
    outbox = tmp_path / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    (outbox / "refresh.json").write_text(
        json.dumps(
            {
                "task": "Publish refreshed branch",
                "requires_github": True,
                "requested_action": "open_pr",
                "repo": "synaptent/aragora",
                "local_evidence": {
                    "branch": "codex/stale-original-refresh",
                    "head": "def456",
                    "supersedes_branch": "codex/stale-original",
                },
                "validation": ["pytest tests/example.py -q"],
                "idempotency_key": "open-pr-codex-stale-original-refresh-def456",
                "created_at": "2026-04-24T16:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "local_branches", lambda _root, _prefix, _base: rows)
    monkeypatch.setattr(mod, "remote_branch_names", lambda _root, _prefix: set())
    monkeypatch.setattr(mod, "merged_branch_names", lambda _root, _base, _prefix: set())
    monkeypatch.setattr(mod, "worktree_map", lambda _root: {})
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda _root: GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode="connectivity_failed",
            error="offline",
            repo=str(tmp_path),
        ),
    )

    payload = mod.audit(
        root=tmp_path,
        base="origin/main",
        repo="synaptent/aragora",
        prefix="codex/",
        recent_hours=72,
        max_branches=None,
        include_patch_equivalence=False,
        publisher_backlog_limit=2,
    )

    assert payload["summary"]["handoff_outbox_branches"] == 1
    assert payload["summary"]["publishable_branch_backlog"] == 1
    original = next(
        record for record in payload["records"] if record["name"] == "codex/stale-original"
    )
    assert original["handoff_outbox_exists"] is True
    assert original["category"] == "protected_handoff_outbox"


def test_audit_treats_superseded_branch_as_terminal_receipt(
    tmp_path: Path, monkeypatch: Any
) -> None:
    outbox = tmp_path / ".aragora" / "automation-outbox"
    receipts = tmp_path / ".aragora" / "automation-receipts"
    outbox.mkdir(parents=True)
    receipts.mkdir(parents=True)
    key = "open-pr-codex-stale-original-refresh-def456"
    (outbox / "refresh.json").write_text(
        json.dumps(
            {
                "task": "Publish refreshed branch",
                "requires_github": True,
                "requested_action": "open_pr",
                "repo": "synaptent/aragora",
                "local_evidence": {
                    "branch": "codex/stale-original-refresh",
                    "head": "def456",
                    "supersedes": {"branch": "codex/stale-original", "head_sha": "abc123"},
                },
                "validation": ["pytest tests/example.py -q"],
                "idempotency_key": key,
                "created_at": "2026-04-24T16:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (receipts / f"{key}.json").write_text(
        json.dumps({"idempotency_key": key, "status": "published"}),
        encoding="utf-8",
    )
    _stub_git_inventory(monkeypatch, _branch_row("codex/stale-original"))
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda _root: GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode="connectivity_failed",
            error="offline",
            repo=str(tmp_path),
        ),
    )

    payload = mod.audit(
        root=tmp_path,
        base="origin/main",
        repo="synaptent/aragora",
        prefix="codex/",
        recent_hours=72,
        max_branches=None,
        include_patch_equivalence=False,
        publisher_backlog_limit=1,
    )

    assert payload["summary"]["handoff_receipted_branches"] == 1
    assert payload["summary"]["publishable_branch_backlog"] == 0
    assert payload["records"][0]["handoff_receipt_exists"] is True
    assert payload["records"][0]["category"] == "protected_handoff_receipt"


def test_audit_reads_top_level_branch_for_terminal_outbox_receipts(
    tmp_path: Path, monkeypatch: Any
) -> None:
    outbox = tmp_path / ".aragora" / "automation-outbox"
    receipts = tmp_path / ".aragora" / "automation-receipts"
    outbox.mkdir(parents=True)
    receipts.mkdir(parents=True)
    key = "open-pr-codex-top-level-abc123"
    (outbox / "top-level.json").write_text(
        json.dumps(
            {
                "task": "Publish top-level branch",
                "requires_github": True,
                "requested_action": "open_pr",
                "repo": "synaptent/aragora",
                "branch": "codex/top-level",
                "head_sha": "abc123",
                "local_evidence": {},
                "validation": ["pytest tests/example.py -q"],
                "idempotency_key": key,
                "created_at": "2026-04-24T16:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (receipts / f"{key}.json").write_text(
        json.dumps({"idempotency_key": key, "status": "published"}),
        encoding="utf-8",
    )
    _stub_git_inventory(monkeypatch, _branch_row("codex/top-level"))
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda _root: GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode="connectivity_failed",
            error="offline",
            repo=str(tmp_path),
        ),
    )

    payload = mod.audit(
        root=tmp_path,
        base="origin/main",
        repo="synaptent/aragora",
        prefix="codex/",
        recent_hours=72,
        max_branches=None,
        include_patch_equivalence=False,
        publisher_backlog_limit=1,
    )

    assert payload["summary"]["publishable_branch_backlog"] == 0
    assert payload["summary"]["handoff_receipted_branches"] == 1
    assert payload["records"][0]["handoff_receipt_exists"] is True
    assert payload["records"][0]["category"] == "protected_handoff_receipt"


def test_audit_excludes_unresolved_top_level_outbox_handoffs_from_publishable_backlog(
    tmp_path: Path, monkeypatch: Any
) -> None:
    now = datetime.now(timezone.utc)
    rows = [
        _branch_row("codex/top-level", committed_at=now),
        _branch_row("codex/new-work", committed_at=now),
    ]
    outbox = tmp_path / ".aragora" / "automation-outbox"
    outbox.mkdir(parents=True)
    (outbox / "top-level.json").write_text(
        json.dumps(
            {
                "task": "Publish top-level branch",
                "requires_github": True,
                "requested_action": "open_pr",
                "repo": "synaptent/aragora",
                "branch": "codex/top-level",
                "head_sha": "abc123",
                "local_evidence": {},
                "validation": ["pytest tests/example.py -q"],
                "idempotency_key": "open-pr-codex-top-level-abc123",
                "created_at": "2026-04-24T16:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "local_branches", lambda _root, _prefix, _base: rows)
    monkeypatch.setattr(mod, "remote_branch_names", lambda _root, _prefix: set())
    monkeypatch.setattr(mod, "merged_branch_names", lambda _root, _base, _prefix: set())
    monkeypatch.setattr(mod, "worktree_map", lambda _root: {})
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda _root: GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode="connectivity_failed",
            error="offline",
            repo=str(tmp_path),
        ),
    )

    payload = mod.audit(
        root=tmp_path,
        base="origin/main",
        repo="synaptent/aragora",
        prefix="codex/",
        recent_hours=72,
        max_branches=None,
        include_patch_equivalence=False,
        publisher_backlog_limit=2,
    )

    by_category = payload["summary"]["by_category"]
    assert by_category["protected_handoff_outbox"] == 1
    assert by_category["salvage_recent_unique"] == 1
    assert payload["summary"]["protected"] == 1
    assert payload["summary"]["publishable_branch_backlog"] == 1
    assert payload["summary"]["handoff_outbox_branches"] == 1
    assert payload["summary"]["writer_should_pause_for_branch_backlog"] is False
    protected = next(record for record in payload["records"] if record["name"] == "codex/top-level")
    assert protected["handoff_outbox_exists"] is True
    assert protected["category"] == "protected_handoff_outbox"


def test_audit_uses_automation_state_root_for_default_handoff_dirs(
    tmp_path: Path, monkeypatch: Any
) -> None:
    repo_root = tmp_path / "worktree"
    state_root = tmp_path / "state-root"
    repo_root.mkdir()
    outbox = state_root / ".aragora" / "automation-outbox"
    receipts = state_root / ".aragora" / "automation-receipts"
    outbox.mkdir(parents=True)
    receipts.mkdir(parents=True)

    key = "open-pr-codex-receipted-abc123"
    (outbox / "receipted.json").write_text(
        json.dumps(
            {
                "task": "Publish receipted branch",
                "requires_github": True,
                "requested_action": "open_pr",
                "repo": "synaptent/aragora",
                "local_evidence": {"branch": "codex/receipted", "head": "abc123"},
                "validation": ["pytest tests/example.py -q"],
                "idempotency_key": key,
                "created_at": "2026-04-24T16:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    (receipts / f"{key}.json").write_text(
        json.dumps({"idempotency_key": key, "status": "published"}),
        encoding="utf-8",
    )

    monkeypatch.setenv("ARAGORA_AUTOMATION_STATE_ROOT", str(state_root))
    monkeypatch.setattr(
        mod,
        "local_branches",
        lambda _root, _prefix, _base: [_branch_row("codex/receipted")],
    )
    monkeypatch.setattr(mod, "remote_branch_names", lambda _root, _prefix: set())
    monkeypatch.setattr(mod, "merged_branch_names", lambda _root, _base, _prefix: set())
    monkeypatch.setattr(mod, "worktree_map", lambda _root: {})
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda _root: GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode="connectivity_failed",
            error="offline",
            repo=str(repo_root),
        ),
    )

    payload = mod.audit(
        root=repo_root,
        base="origin/main",
        repo="synaptent/aragora",
        prefix="codex/",
        recent_hours=72,
        max_branches=None,
        include_patch_equivalence=False,
        publisher_backlog_limit=1,
    )

    assert payload["outbox_dir"] == str(outbox)
    assert payload["receipt_dir"] == str(receipts)
    assert payload["summary"]["handoff_receipted_branches"] == 1
    assert payload["summary"]["publishable_branch_backlog"] == 0
    assert payload["summary"]["writer_should_pause_for_branch_backlog"] is False
    assert payload["records"][0]["category"] == "protected_handoff_receipt"


def test_parser_checks_patch_equivalence_by_default() -> None:
    args = mod.build_parser().parse_args([])

    assert args.include_patch_equivalence is True
    assert args.patch_equivalence_time_budget_seconds == 90.0


def test_parser_can_skip_patch_equivalence() -> None:
    args = mod.build_parser().parse_args(["--skip-patch-equivalence"])

    assert args.include_patch_equivalence is False


def test_parser_can_disable_patch_equivalence_time_budget() -> None:
    args = mod.build_parser().parse_args(["--patch-equivalence-time-budget-seconds", "-1"])

    assert args.patch_equivalence_time_budget_seconds == -1


def test_run_git_returns_completed_process_on_timeout(tmp_path: Path, monkeypatch: Any) -> None:
    def raise_timeout(*_args: Any, **kwargs: Any) -> SimpleNamespace:
        raise subprocess.TimeoutExpired(cmd=["git", "status"], timeout=kwargs["timeout"])

    monkeypatch.setattr(mod.subprocess, "run", raise_timeout)

    proc = mod.run_git(["status"], tmp_path, timeout=3)

    assert proc.returncode == 124
    assert "command timed out after 3s: git status" in proc.stderr


def test_audit_skips_patch_equivalence_after_time_budget(tmp_path: Path, monkeypatch: Any) -> None:
    rows = [_branch_row("codex/one"), _branch_row("codex/two")]
    monkeypatch.setattr(mod, "local_branches", lambda _root, _prefix, _base: rows)
    monkeypatch.setattr(mod, "remote_branch_names", lambda _root, _prefix: set())
    monkeypatch.setattr(mod, "merged_branch_names", lambda _root, _base, _prefix: set())
    monkeypatch.setattr(mod, "worktree_map", lambda _root: {})
    monkeypatch.setattr(mod, "terminal_receipted_handoff_branches", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(mod, "unresolved_outbox_handoff_branches", lambda *_args, **_kwargs: set())
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda _root: GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode="connectivity_failed",
            error="offline",
            repo=str(tmp_path),
        ),
    )

    def fail_patch_equivalence(*_args: Any, **_kwargs: Any) -> bool:
        raise AssertionError("patch equivalence should not run after budget exhaustion")

    monkeypatch.setattr(mod, "is_patch_equivalent", fail_patch_equivalence)

    payload = mod.audit(
        root=tmp_path,
        base="origin/main",
        repo="synaptent/aragora",
        prefix="codex/",
        recent_hours=72,
        max_branches=None,
        include_patch_equivalence=True,
        patch_equivalence_time_budget_seconds=0,
        publisher_backlog_limit=12,
    )

    assert payload["patch_equivalence_budget_exhausted"] is True
    assert payload["patch_equivalence_skipped_branches"] == 2
    assert [record["category"] for record in payload["records"]] == [
        "salvage_recent_unique",
        "salvage_recent_unique",
    ]


def test_audit_skip_patch_equivalence_still_cleans_empty_branch_diff(
    tmp_path: Path, monkeypatch: Any
) -> None:
    now = datetime.now(timezone.utc)
    rows = [
        _branch_row("codex/cancels-out", committed_at=now),
        _branch_row("codex/new-work", committed_at=now),
    ]
    monkeypatch.setattr(mod, "local_branches", lambda _root, _prefix, _base: rows)
    monkeypatch.setattr(mod, "remote_branch_names", lambda _root, _prefix: set())
    monkeypatch.setattr(mod, "merged_branch_names", lambda _root, _base, _prefix: set())
    monkeypatch.setattr(mod, "worktree_map", lambda _root: {})
    monkeypatch.setattr(
        mod,
        "has_empty_branch_diff",
        lambda _root, _base, branch, **_kwargs: branch == "codex/cancels-out",
    )
    patch_checked_branches: list[str] = []

    def is_patch_equivalent(_root: Path, _base: str, branch: str, **_kwargs: Any) -> bool:
        patch_checked_branches.append(branch)
        return False

    monkeypatch.setattr(mod, "is_patch_equivalent", is_patch_equivalent)
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda _root: GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode="connectivity_failed",
            error="offline",
            repo=str(tmp_path),
        ),
    )

    payload = mod.audit(
        root=tmp_path,
        base="origin/main",
        repo="synaptent/aragora",
        prefix="codex/",
        recent_hours=72,
        max_branches=None,
        include_patch_equivalence=False,
        publisher_backlog_limit=2,
    )

    by_category = payload["summary"]["by_category"]
    assert by_category["cleanup_patch_equivalent"] == 1
    assert by_category["salvage_recent_unique"] == 1
    assert payload["summary"]["publishable_branch_backlog"] == 1
    empty_diff = next(
        record for record in payload["records"] if record["name"] == "codex/cancels-out"
    )
    assert empty_diff["patch_equivalent_to_base"] is True
    assert empty_diff["category"] == "cleanup_patch_equivalent"
    assert patch_checked_branches == ["codex/new-work"]


def test_audit_skip_patch_equivalence_verifies_salvage_candidates(
    tmp_path: Path, monkeypatch: Any
) -> None:
    now = datetime.now(timezone.utc)
    rows = [
        _branch_row("codex/replayed-diverged", committed_at=now, behind_count="4"),
        _branch_row("codex/real-diverged", committed_at=now, behind_count="4"),
        _branch_row("codex/protected-outbox", committed_at=now, behind_count="4"),
    ]
    outbox = tmp_path / ".aragora" / "automation-outbox"
    receipts = tmp_path / ".aragora" / "automation-receipts"
    outbox.mkdir(parents=True)
    receipts.mkdir(parents=True)
    (outbox / "protected.json").write_text(
        json.dumps(
            {
                "task": "Publish protected branch",
                "requires_github": True,
                "requested_action": "open_pr",
                "repo": "synaptent/aragora",
                "local_evidence": {"branch": "codex/protected-outbox", "head": "abc1234"},
                "validation": ["pytest tests/example.py -q"],
                "idempotency_key": "open-pr-codex-protected-outbox-abc1234",
                "created_at": "2026-04-27T16:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "local_branches", lambda _root, _prefix, _base: rows)
    monkeypatch.setattr(mod, "remote_branch_names", lambda _root, _prefix: set())
    monkeypatch.setattr(mod, "merged_branch_names", lambda _root, _base, _prefix: set())
    monkeypatch.setattr(mod, "worktree_map", lambda _root: {})
    monkeypatch.setattr(mod, "has_empty_branch_diff", lambda *_args, **_kwargs: False)
    patch_checked_branches: list[str] = []

    def is_patch_equivalent(_root: Path, _base: str, branch: str, **_kwargs: Any) -> bool:
        patch_checked_branches.append(branch)
        return branch == "codex/replayed-diverged"

    monkeypatch.setattr(mod, "is_patch_equivalent", is_patch_equivalent)
    monkeypatch.setattr(
        mod,
        "check_github_cli_health",
        lambda _root: GitHubCLIHealth(
            ready=False,
            auth_ok=False,
            api_ok=False,
            mode="connectivity_failed",
            error="offline",
            repo=str(tmp_path),
        ),
    )

    payload = mod.audit(
        root=tmp_path,
        base="origin/main",
        repo="synaptent/aragora",
        prefix="codex/",
        recent_hours=72,
        max_branches=None,
        include_patch_equivalence=False,
        publisher_backlog_limit=2,
        outbox_dir=outbox,
        receipt_dir=receipts,
    )

    by_name = {record["name"]: record for record in payload["records"]}
    by_category = payload["summary"]["by_category"]
    assert by_name["codex/replayed-diverged"]["category"] == "cleanup_patch_equivalent"
    assert by_name["codex/replayed-diverged"]["patch_equivalent_to_base"] is True
    assert by_name["codex/real-diverged"]["category"] == "salvage_diverged_recent"
    assert by_name["codex/protected-outbox"]["category"] == "protected_handoff_outbox"
    assert by_category["cleanup_patch_equivalent"] == 1
    assert by_category["salvage_diverged_recent"] == 1
    assert payload["summary"]["salvage_candidates"] == 1
    assert patch_checked_branches == ["codex/replayed-diverged", "codex/real-diverged"]


def test_main_derives_handoff_dirs_from_explicit_aragora_state_root(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    state_root = tmp_path / ".aragora"
    captured: dict[str, Any] = {}
    monkeypatch.setattr(mod, "repo_root", lambda _path: tmp_path / "worktree")

    def fake_audit(**kwargs: Any) -> dict[str, object]:
        captured.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(mod, "audit", fake_audit)

    exit_code = mod.main(
        [
            "--repo",
            str(tmp_path),
            "--state-root",
            str(state_root),
            "--json",
        ]
    )

    assert exit_code == 0
    assert captured["outbox_dir"] == state_root / "automation-outbox"
    assert captured["receipt_dir"] == state_root / "automation-receipts"
    assert '"ok": true' in capsys.readouterr().out


def test_patch_equivalence_treats_empty_branch_diff_as_cleanup(
    tmp_path: Path, monkeypatch: Any
) -> None:
    calls: list[list[str]] = []

    def fake_run_git(args: list[str], _cwd: Path, **_kwargs: Any) -> SimpleNamespace:
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(mod, "run_git", fake_run_git)

    assert mod.is_patch_equivalent(tmp_path, "origin/main", "codex/cancels-out") is True
    assert calls == [["diff", "--quiet", "origin/main...codex/cancels-out"]]


def test_patch_equivalence_treats_identical_touched_files_as_cleanup(
    tmp_path: Path, monkeypatch: Any
) -> None:
    calls: list[list[str]] = []

    def fake_run_git(args: list[str], _cwd: Path, **_kwargs: Any) -> SimpleNamespace:
        calls.append(args)
        if args == ["diff", "--quiet", "origin/main...codex/squashed"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        if args == ["diff", "--name-only", "-z", "origin/main...codex/squashed"]:
            return SimpleNamespace(
                returncode=0,
                stdout="scripts/agent_bridge.py\0scripts/tmux_send_prompt.sh\0",
                stderr="",
            )
        if args == [
            "diff",
            "--quiet",
            "origin/main..codex/squashed",
            "--",
            "scripts/agent_bridge.py",
            "scripts/tmux_send_prompt.sh",
        ]:
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(mod, "run_git", fake_run_git)

    assert mod.is_patch_equivalent(tmp_path, "origin/main", "codex/squashed") is True
    assert calls == [
        ["diff", "--quiet", "origin/main...codex/squashed"],
        ["diff", "--name-only", "-z", "origin/main...codex/squashed"],
        [
            "diff",
            "--quiet",
            "origin/main..codex/squashed",
            "--",
            "scripts/agent_bridge.py",
            "scripts/tmux_send_prompt.sh",
        ],
    ]


def test_patch_equivalence_falls_back_to_cherry_when_branch_has_diff(
    tmp_path: Path, monkeypatch: Any
) -> None:
    calls: list[list[str]] = []

    def fake_run_git(args: list[str], _cwd: Path, **_kwargs: Any) -> SimpleNamespace:
        calls.append(args)
        if args == ["diff", "--quiet", "origin/main...codex/replayed"]:
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        if args == ["diff", "--name-only", "-z", "origin/main...codex/replayed"]:
            return SimpleNamespace(returncode=0, stdout="scripts/example.py\0", stderr="")
        if args == [
            "diff",
            "--quiet",
            "origin/main..codex/replayed",
            "--",
            "scripts/example.py",
        ]:
            return SimpleNamespace(returncode=1, stdout="", stderr="")
        return SimpleNamespace(returncode=0, stdout="- abc123 already applied\n", stderr="")

    monkeypatch.setattr(mod, "run_git", fake_run_git)

    assert mod.is_patch_equivalent(tmp_path, "origin/main", "codex/replayed") is True
    assert calls == [
        ["diff", "--quiet", "origin/main...codex/replayed"],
        ["diff", "--name-only", "-z", "origin/main...codex/replayed"],
        [
            "diff",
            "--quiet",
            "origin/main..codex/replayed",
            "--",
            "scripts/example.py",
        ],
        ["cherry", "origin/main", "codex/replayed"],
    ]
