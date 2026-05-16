"""Tests for scripts/codex_worktree_value_inventory.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections.abc import Generator
from typing import Any

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"


@pytest.fixture(autouse=True)
def _setup_path() -> Generator[None, None, None]:
    sys.path.insert(0, str(SCRIPTS_DIR))
    yield
    sys.path.remove(str(SCRIPTS_DIR))


def _context(tmp_path: Path, **overrides: Any) -> Any:
    import codex_worktree_value_inventory as mod

    values: dict[str, Any] = {
        "repo": tmp_path,
        "base": "origin/main",
        "base_sha": "base-sha",
        "outbox_dir": tmp_path / ".aragora" / "automation-outbox",
        "receipt_dir": tmp_path / ".aragora" / "automation-receipts",
        "worktrees_by_path": {},
        "unresolved_outbox_branches": set(),
        "terminal_receipt_branch_heads": {},
        "skip_gh": False,
        "git_timeout": 1,
        "gh_timeout": 1,
        "patch_timeout": 1,
    }
    values.update(overrides)
    outbox_dir = values["outbox_dir"]
    receipt_dir = values["receipt_dir"]
    assert isinstance(outbox_dir, Path)
    assert isinstance(receipt_dir, Path)
    outbox_dir.mkdir(parents=True, exist_ok=True)
    receipt_dir.mkdir(parents=True, exist_ok=True)
    return mod.InventoryContext(**values)


def _candidate(tmp_path: Path, name: str = "abcd", *, repo: bool = True) -> Path:
    root = tmp_path / name
    root.mkdir(parents=True)
    if repo:
        repo_path = root / "aragora"
        repo_path.mkdir()
        (repo_path / ".git").mkdir()
    return root


def _stub_clean_git(
    monkeypatch: pytest.MonkeyPatch,
    *,
    branch: str | None = "codex/test",
    head: str | None = "abcdef123456",
    ahead: int | None = 0,
    behind: int | None = 0,
    dirty: bool = False,
    open_prs: list[dict[str, Any]] | None = None,
    open_pr_failed: bool = False,
    patch_equivalent: bool = False,
) -> None:
    import codex_worktree_value_inventory as mod

    monkeypatch.setattr(mod, "git_branch", lambda *_args, **_kwargs: (branch, False, None))
    monkeypatch.setattr(mod, "git_head", lambda *_args, **_kwargs: (head, False, None))
    monkeypatch.setattr(mod, "git_status_dirty", lambda *_args, **_kwargs: (dirty, False, None))
    monkeypatch.setattr(
        mod,
        "git_ahead_behind",
        lambda *_args, **_kwargs: (ahead, behind, False, None),
    )
    monkeypatch.setattr(
        mod,
        "lookup_open_prs",
        lambda *_args, **_kwargs: (
            open_prs or [],
            open_pr_failed,
            "open PR lookup failed" if open_pr_failed else None,
        ),
    )
    monkeypatch.setattr(mod, "is_patch_equivalent", lambda *_args, **_kwargs: patch_equivalent)


def test_no_git_cache_residue_is_cleanup_candidate(tmp_path: Path) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path, repo=False)

    candidate = mod.classify_candidate(
        root,
        context=_context(tmp_path),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "no_git_cache_residue"
    assert candidate.cleanup_candidate is True
    assert candidate.decision == "cleanup_candidate"


def test_no_git_active_marker_is_preserved(tmp_path: Path) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path, repo=False)
    (root / ".codex_session_active").write_text("active\n")

    candidate = mod.classify_candidate(
        root,
        context=_context(tmp_path),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "active_or_dirty"
    assert candidate.cleanup_candidate is False
    assert candidate.decision == "preserve"


def test_dirty_repo_takes_priority_over_unique_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    _stub_clean_git(monkeypatch, ahead=3, dirty=True)

    candidate = mod.classify_candidate(
        root,
        context=_context(tmp_path),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "active_or_dirty"
    assert "git status is dirty or unavailable" in candidate.proof


def test_open_pr_classification_blocks_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    _stub_clean_git(
        monkeypatch,
        ahead=2,
        open_prs=[{"number": 1, "title": "PR", "url": "https://example.test/pr/1"}],
    )

    candidate = mod.classify_candidate(
        root,
        context=_context(tmp_path),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "open_pr_or_outbox"
    assert candidate.cleanup_candidate is False


def test_unresolved_outbox_classification_blocks_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    _stub_clean_git(monkeypatch, ahead=2)

    candidate = mod.classify_candidate(
        root,
        context=_context(tmp_path, unresolved_outbox_branches={"codex/test"}),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "open_pr_or_outbox"
    assert "unresolved automation outbox references branch" in candidate.proof


def test_terminal_receipt_classification_blocks_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    _stub_clean_git(monkeypatch, ahead=2, head="abcdef123456")

    candidate = mod.classify_candidate(
        root,
        context=_context(tmp_path, terminal_receipt_branch_heads={"codex/test": {"abcdef1"}}),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "receipt_protected"
    assert candidate.cleanup_candidate is False


def test_unique_unharvested_when_ahead_and_not_patch_equivalent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    _stub_clean_git(monkeypatch, ahead=2, patch_equivalent=False)

    candidate = mod.classify_candidate(
        root,
        context=_context(tmp_path),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "unique_unharvested"
    assert candidate.decision == "harvest_candidate"
    assert candidate.cleanup_candidate is False


def test_patch_equivalent_ahead_work_is_cleanup_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    _stub_clean_git(monkeypatch, ahead=2, patch_equivalent=True)

    candidate = mod.classify_candidate(
        root,
        context=_context(tmp_path),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "patch_equivalent_or_merged"
    assert candidate.cleanup_candidate is True


def test_lookup_failure_preserves_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    _stub_clean_git(monkeypatch, ahead=0, open_pr_failed=True)

    candidate = mod.classify_candidate(
        root,
        context=_context(tmp_path),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "lookup_failed"
    assert candidate.cleanup_candidate is False
    assert "open PR lookup failed" in candidate.git.lookup_errors


def test_summary_reports_top_cleanup_and_harvest_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    cleanup_root = _candidate(tmp_path, "cleanup")
    unique_root = _candidate(tmp_path, "unique")
    _stub_clean_git(monkeypatch, ahead=0)
    cleanup = mod.classify_candidate(
        cleanup_root,
        context=_context(tmp_path),
        size_bytes=2048,
        size_lookup_failed=False,
    )
    _stub_clean_git(monkeypatch, ahead=1, patch_equivalent=False)
    unique = mod.classify_candidate(
        unique_root,
        context=_context(tmp_path),
        size_bytes=4096,
        size_lookup_failed=False,
    )

    summary = mod.build_summary([cleanup, unique])

    assert summary["cleanup_candidate_count"] == 1
    assert summary["harvest_candidate_count"] == 1
    assert summary["top_cleanup_candidates"][0]["path"] == str(cleanup_root)
    assert summary["top_unique_unharvested"][0]["path"] == str(unique_root)


def test_write_ledger_creates_snapshot_latest_and_jsonl(tmp_path: Path) -> None:
    import codex_worktree_value_inventory as mod

    payload = {
        "schema": mod.SCHEMA,
        "generated_at": "2026-05-16T17:00:00+00:00",
        "root": "/tmp/root",
        "summary": {"total_candidates": 0},
    }

    written = mod.write_ledger(tmp_path / "ledger", payload)

    assert Path(written["snapshot"]).is_file()
    assert Path(written["latest"]).is_file()
    ledger_lines = Path(written["ledger"]).read_text(encoding="utf-8").splitlines()
    assert len(ledger_lines) == 1
    assert json.loads(ledger_lines[0])["event_type"] == "inventory"
