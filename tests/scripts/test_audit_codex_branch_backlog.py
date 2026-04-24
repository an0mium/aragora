from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.github_cli_health import GitHubCLIHealth

import scripts.audit_codex_branch_backlog as mod


def _branch_row(name: str = "codex/example") -> dict[str, str]:
    return {
        "name": name,
        "upstream": "",
        "head_sha": "abc1234",
        "committed_at": datetime.now(timezone.utc).isoformat(),
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
    )

    assert payload["github_health"]["ready"] is True
    assert payload["open_pr_lookup_skipped"] is False
    assert payload["records"][0]["open_pr"] == 6500
    assert payload["records"][0]["category"] == "protected_open_pr"
