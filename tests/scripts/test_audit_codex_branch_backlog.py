from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.github_cli_health import GitHubCLIHealth

import scripts.audit_codex_branch_backlog as mod


def _branch_row(
    name: str = "codex/example",
    *,
    committed_at: datetime | None = None,
) -> dict[str, str]:
    return {
        "name": name,
        "upstream": "",
        "head_sha": "abc1234",
        "committed_at": (committed_at or datetime.now(timezone.utc)).isoformat(),
        "ahead_count": "1",
        "subject": "test branch",
    }


def _stub_git_inventory(monkeypatch: Any, row: dict[str, str]) -> None:
    monkeypatch.setattr(mod, "local_branches", lambda _root, _prefix, _base: [row])
    monkeypatch.setattr(mod, "remote_branch_names", lambda _root, _prefix: set())
    monkeypatch.setattr(mod, "merged_branch_names", lambda _root, _base, _prefix: set())
    monkeypatch.setattr(mod, "worktree_map", lambda _root: {})


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


def test_parser_can_skip_patch_equivalence() -> None:
    args = mod.build_parser().parse_args(["--skip-patch-equivalence"])

    assert args.include_patch_equivalence is False
