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
        "repo_remote_urls": {"https://example.test/target"},
        "strict_repo_identity": False,
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
    monkeypatch.setattr(
        mod,
        "branch_unique_merge_commits",
        lambda *_args, **_kwargs: ([], None),
    )
    monkeypatch.setattr(
        mod,
        "branch_merge_tree_matches_base",
        lambda *_args, **_kwargs: (False, None),
    )


def test_default_scan_preserves_foreign_repo_as_lookup_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    _stub_clean_git(monkeypatch, ahead=0)
    monkeypatch.setattr(
        mod, "repo_remote_urls", lambda *_args, **_kwargs: {"https://example.test/other"}
    )

    candidate = mod.classify_candidate(
        root,
        context=_context(
            tmp_path,
            strict_repo_identity=True,
            repo_remote_urls={"https://example.test/target"},
        ),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "lookup_failed"
    assert candidate.cleanup_candidate is False
    assert candidate.decision == "preserve"
    assert "repo identity does not match target repo" in candidate.proof
    assert "repo identity does not match target repo" in candidate.git.lookup_errors


def test_explicit_root_scan_allows_foreign_repo_for_backwards_compat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    _stub_clean_git(monkeypatch, ahead=0)
    monkeypatch.setattr(
        mod, "repo_remote_urls", lambda *_args, **_kwargs: {"https://example.test/other"}
    )

    candidate = mod.classify_candidate(
        root,
        context=_context(
            tmp_path,
            strict_repo_identity=False,
            repo_remote_urls={"https://example.test/target"},
        ),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "unregistered_git_residue"
    assert candidate.cleanup_candidate is True


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
    assert candidate.cleanup_safety.safety_class == "stale_residue"
    assert candidate.cleanup_safety.requires_live_cleanup_inspect is True
    assert candidate.cleanup_safety.safe_to_delete is False


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
    assert candidate.cleanup_safety.safety_class == "owned"
    assert candidate.cleanup_safety.preserve is True
    assert candidate.cleanup_safety.signals == ["owned"]


def test_anchor_only_wrapper_is_not_active(tmp_path: Path) -> None:
    """An anchor-only wrapper (passive sentinel without active lock) is NOT
    classified as active_or_dirty. Anchor files are passive markers left by
    wrapper scripts and don't indicate an active session."""
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path, repo=False)
    (root / ".claude-session-anchor").write_text("anchor\n")

    candidate = mod.classify_candidate(
        root,
        context=_context(tmp_path),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "no_git_cache_residue"
    assert candidate.cleanup_candidate is True
    assert candidate.decision == "cleanup_candidate"


def test_nested_foreign_git_repo_is_lookup_failed_not_cleanup_candidate(tmp_path: Path) -> None:
    import codex_worktree_value_inventory as mod

    root = tmp_path / "desktop-session"
    foreign = root / "RingRift"
    foreign.mkdir(parents=True)
    (foreign / ".git").mkdir()
    (foreign / "pyproject.toml").write_text('[project]\nname = "ringrift"\n')

    candidate = mod.classify_candidate(
        root,
        context=_context(tmp_path, strict_repo_identity=True),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "lookup_failed"
    assert candidate.cleanup_candidate is False
    assert candidate.repo_path == str(foreign)
    assert "repo identity does not match target repo" in candidate.proof


def test_ambiguous_no_git_project_directory_is_lookup_failed(tmp_path: Path) -> None:
    import codex_worktree_value_inventory as mod

    root = tmp_path / "desktop-session"
    project = root / "project"
    project.mkdir(parents=True)
    (project / "package.json").write_text('{"name": "not-aragora"}\n')

    candidate = mod.classify_candidate(
        root,
        context=_context(tmp_path, strict_repo_identity=True),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "lookup_failed"
    assert candidate.cleanup_candidate is False
    assert candidate.links["project_markers"] == [str(project / "package.json")]


def test_explicit_root_ambiguous_project_marker_keeps_legacy_cleanup_behavior(
    tmp_path: Path,
) -> None:
    import codex_worktree_value_inventory as mod

    root = tmp_path / "desktop-session"
    project = root / "project"
    project.mkdir(parents=True)
    (project / "package.json").write_text('{"name": "not-aragora"}\n')

    candidate = mod.classify_candidate(
        root,
        context=_context(tmp_path),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "no_git_cache_residue"
    assert candidate.cleanup_candidate is True


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
    assert candidate.cleanup_safety.safety_class == "unsafe_to_delete"
    assert "unsafe_to_delete" in candidate.cleanup_safety.signals


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
    assert candidate.cleanup_safety.safety_class == "referenced_preserve"
    assert "duplicate" in candidate.cleanup_safety.signals


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
    assert candidate.cleanup_safety.safety_class == "referenced_preserve"
    assert "harvested" in candidate.cleanup_safety.signals


def test_terminal_path_receipt_classification_blocks_branchless_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    repo_path = root / "aragora"
    _stub_clean_git(monkeypatch, branch=None, ahead=2, head="abcdef123456")

    candidate = mod.classify_candidate(
        root,
        context=_context(
            tmp_path,
            terminal_receipt_path_heads={str(repo_path.resolve()): {"abcdef1"}},
        ),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "receipt_protected"
    assert candidate.cleanup_candidate is False
    assert "terminal receipt references path/head" in candidate.proof


def test_terminal_path_receipt_head_mismatch_stays_unique(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    repo_path = root / "aragora"
    _stub_clean_git(
        monkeypatch,
        branch=None,
        ahead=2,
        head="abcdef123456",
        patch_equivalent=False,
    )

    candidate = mod.classify_candidate(
        root,
        context=_context(
            tmp_path,
            terminal_receipt_path_heads={str(repo_path.resolve()): {"9999999"}},
        ),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "unique_unharvested"
    assert candidate.decision == "harvest_candidate"


def test_terminal_receipt_path_heads_reads_harvest_receipt_source_candidate(
    tmp_path: Path,
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    repo_path = root / "aragora"
    receipt_dir = tmp_path / ".aragora" / "worktree-harvest" / "harvest-receipts"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "preserve.json").write_text(
        json.dumps(
            {
                "decision": "preserve_existing_pr",
                "source_candidate": {
                    "path": str(root),
                    "repo_path": str(repo_path),
                    "head": "abcdef123456",
                },
            }
        ),
        encoding="utf-8",
    )

    refs = mod.terminal_receipt_path_heads([receipt_dir])

    assert refs[str(root.resolve())] == {"abcdef123456"}
    assert refs[str(repo_path.resolve())] == {"abcdef123456"}


def test_terminal_receipt_path_heads_reads_object_decision_outcome(
    tmp_path: Path,
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    receipt_dir = tmp_path / ".aragora" / "worktree-harvest" / "harvest-receipts"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "preserve.json").write_text(
        json.dumps(
            {
                "decision": {"outcome": "preserve_existing_merged_pr"},
                "selected_candidate": {
                    "path": str(root),
                    "head": "abcdef123456",
                },
            }
        ),
        encoding="utf-8",
    )

    refs = mod.terminal_receipt_path_heads([receipt_dir])

    assert refs[str(root.resolve())] == {"abcdef123456"}


def test_terminal_receipt_path_heads_reads_object_decision_status(
    tmp_path: Path,
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    receipt_dir = tmp_path / ".aragora" / "worktree-harvest" / "harvest-receipts"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "completed.json").write_text(
        json.dumps(
            {
                "decision": {"status": "completed"},
                "selected_candidate": {
                    "path": str(root),
                    "head": "abcdef123456",
                },
            }
        ),
        encoding="utf-8",
    )

    refs = mod.terminal_receipt_path_heads([receipt_dir])

    assert refs[str(root.resolve())] == {"abcdef123456"}


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
    assert candidate.cleanup_safety.safety_class == "unsafe_to_delete"
    assert candidate.cleanup_safety.preserve is True


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
    assert candidate.cleanup_safety.safety_class == "harvested_or_duplicate"
    assert {"harvested", "duplicate"} <= set(candidate.cleanup_safety.signals)


def test_registered_no_unique_commits_is_stale_or_merged_safety(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    repo_path = root / "aragora"
    _stub_clean_git(monkeypatch, ahead=0)

    candidate = mod.classify_candidate(
        root,
        context=_context(
            tmp_path,
            worktrees_by_path={
                str(repo_path.resolve()): mod.WorktreeEntry(path=repo_path, branch="codex/test")
            },
        ),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "patch_equivalent_or_merged"
    assert candidate.cleanup_safety.safety_class == "stale_or_merged"
    assert candidate.cleanup_safety.requires_live_cleanup_inspect is True


def test_smart_merge_detection_default_off_preserves_unique_harvest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    _stub_clean_git(monkeypatch, ahead=2, patch_equivalent=False)
    monkeypatch.setattr(
        mod,
        "branch_subjects_match_recent_main",
        lambda *_args, **_kwargs: pytest.fail("smart detector should be opt-in"),
    )

    candidate = mod.classify_candidate(
        root,
        context=_context(
            tmp_path,
            smart_merge_detection=False,
            smart_merge_main_subjects=["feat(scripts): already merged"],
        ),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "unique_unharvested"
    assert candidate.decision == "harvest_candidate"


def test_smart_merge_detection_keeps_matching_subjects_harvestable_without_patch_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    _stub_clean_git(monkeypatch, ahead=2, patch_equivalent=False)
    monkeypatch.setattr(
        mod,
        "branch_unique_commit_subjects",
        lambda *_args, **_kwargs: [
            "feat(scripts): list active sessions [lane: P42]",
            "docs(status): inventory receipt [lane: P42]",
        ],
    )
    monkeypatch.setattr(
        mod,
        "branch_patches_present_on_base",
        lambda *_args, **_kwargs: (False, []),
    )

    candidate = mod.classify_candidate(
        root,
        context=_context(
            tmp_path,
            smart_merge_detection=True,
            smart_merge_main_subjects=[
                "feat(scripts): list active sessions (#7260)",
                "docs(status): inventory receipt (#7258)",
            ],
        ),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "unique_unharvested"
    assert candidate.cleanup_candidate is False
    assert candidate.git.smart_merge_equivalent_to_base is False
    assert (
        "all unique commit subjects match recent main squash-merge subjects "
        "(advisory; patch proof still required)"
    ) in candidate.proof
    assert candidate.links["smart_merge_matched_subjects"] == [
        "feat(scripts): list active sessions [lane: P42]",
        "docs(status): inventory receipt [lane: P42]",
    ]


def test_smart_merge_detection_reclassifies_already_present_patches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    _stub_clean_git(monkeypatch, ahead=2, patch_equivalent=False)
    monkeypatch.setattr(
        mod,
        "branch_subjects_match_recent_main",
        lambda *_args, **_kwargs: (False, []),
    )
    monkeypatch.setattr(
        mod,
        "branch_patches_present_on_base",
        lambda *_args, **_kwargs: (True, ["2f2a1f6", "b46d5c6"]),
    )

    candidate = mod.classify_candidate(
        root,
        context=_context(
            tmp_path,
            smart_merge_detection=True,
            smart_merge_main_subjects=[],
        ),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "patch_equivalent_or_merged"
    assert candidate.cleanup_candidate is True
    assert candidate.git.smart_merge_equivalent_to_base is True
    assert "all unique commit patches are already present on base" in candidate.proof
    assert candidate.links["smart_merge_matched_commits"] == ["2f2a1f6", "b46d5c6"]


def test_smart_merge_detection_reclassifies_noop_merge_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    _stub_clean_git(monkeypatch, ahead=3, patch_equivalent=False)
    monkeypatch.setattr(
        mod,
        "branch_merge_tree_matches_base",
        lambda *_args, **_kwargs: (True, None),
    )

    candidate = mod.classify_candidate(
        root,
        context=_context(
            tmp_path,
            smart_merge_detection=True,
            smart_merge_main_subjects=[],
        ),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "patch_equivalent_or_merged"
    assert candidate.cleanup_candidate is True
    assert candidate.git.smart_merge_equivalent_to_base is True
    assert "merging branch into base leaves base tree unchanged" in candidate.proof
    assert candidate.links["smart_merge_merge_tree"] == "origin/main"


def test_smart_merge_detection_merge_tree_timeout_is_protected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    _stub_clean_git(monkeypatch, ahead=3, patch_equivalent=False)
    monkeypatch.setattr(
        mod,
        "branch_merge_tree_matches_base",
        lambda *_args, **_kwargs: (None, "command timed out after 1s"),
    )

    candidate = mod.classify_candidate(
        root,
        context=_context(
            tmp_path,
            smart_merge_detection=True,
            smart_merge_main_subjects=[],
        ),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "lookup_failed"
    assert candidate.cleanup_candidate is False
    assert candidate.git.lookup_failed is True
    assert any(
        "smart merge merge-tree lookup failed" in item for item in candidate.git.lookup_errors
    )


def test_smart_merge_detection_merge_tree_conflict_falls_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    _stub_clean_git(monkeypatch, ahead=3, patch_equivalent=False)
    monkeypatch.setattr(
        mod,
        "branch_merge_tree_matches_base",
        lambda *_args, **_kwargs: (False, "CONFLICT (content): tracked.txt"),
    )
    monkeypatch.setattr(
        mod,
        "branch_subjects_match_recent_main",
        lambda *_args, **_kwargs: (True, ["feat: looks merged"]),
    )
    monkeypatch.setattr(
        mod,
        "branch_patches_present_on_base",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("conflicts must not fall through to patch heuristics")
        ),
    )

    candidate = mod.classify_candidate(
        root,
        context=_context(
            tmp_path,
            smart_merge_detection=True,
            smart_merge_main_subjects=[],
        ),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "unique_unharvested"
    assert candidate.cleanup_candidate is False
    assert candidate.git.lookup_failed is False
    assert candidate.links["smart_merge_merge_tree_error"] == "CONFLICT (content): tracked.txt"
    assert "merge-tree did not prove branch is already represented on base" in candidate.proof


def test_branch_patches_present_on_base_uses_temp_index_only(tmp_path: Path) -> None:
    import subprocess

    import codex_worktree_value_inventory as mod

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return proc.stdout.strip()

    git("init", "-b", "master")
    git("config", "user.email", "test@example.test")
    git("config", "user.name", "Test User")
    (repo / "tracked.txt").write_text("old\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-m", "base")
    base_commit = git("rev-parse", "HEAD")

    git("checkout", "-b", "stale")
    (repo / "tracked.txt").write_text("new\n", encoding="utf-8")
    git("commit", "-am", "stale patch")
    stale_commit = git("rev-parse", "HEAD")

    git("checkout", "master")
    (repo / "tracked.txt").write_text("new\n", encoding="utf-8")
    git("commit", "-am", "main contains patch")
    git("checkout", "-b", "unrelated-worktree", base_commit)

    present, matched = mod.branch_patches_present_on_base(
        repo,
        "master",
        "stale",
        timeout=5,
    )

    assert present is True
    assert matched == [stale_commit]


def test_branch_merge_tree_matches_base_for_noop_merge_commit(tmp_path: Path) -> None:
    import subprocess

    import codex_worktree_value_inventory as mod

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return proc.stdout.strip()

    git("init", "-b", "master")
    git("config", "user.email", "test@example.test")
    git("config", "user.name", "Test User")
    (repo / "tracked.txt").write_text("base\n", encoding="utf-8")
    git("add", "tracked.txt")
    git("commit", "-m", "base")

    git("checkout", "-b", "stale")
    (repo / "tracked.txt").write_text("base\nstale value\n", encoding="utf-8")
    git("commit", "-am", "stale value")

    git("checkout", "master")
    (repo / "tracked.txt").write_text("base\nstale value\n", encoding="utf-8")
    git("commit", "-am", "main already has stale value")
    git("checkout", "stale")
    git("merge", "--no-ff", "master", "-m", "merge main into stale")

    matches, error = mod.branch_merge_tree_matches_base(
        repo,
        "master",
        "stale",
        timeout=5,
    )

    assert error is None
    assert matches is True


def test_smart_merge_detection_preserves_branches_with_merge_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    _stub_clean_git(monkeypatch, ahead=3, patch_equivalent=False)
    monkeypatch.setattr(
        mod,
        "branch_unique_merge_commits",
        lambda *_args, **_kwargs: (["merge-sha"], None),
    )
    monkeypatch.setattr(
        mod,
        "branch_subjects_match_recent_main",
        lambda *_args, **_kwargs: (True, ["feat: already merged"]),
    )
    monkeypatch.setattr(
        mod,
        "branch_patches_present_on_base",
        lambda *_args, **_kwargs: (True, ["commit-sha"]),
    )

    candidate = mod.classify_candidate(
        root,
        context=_context(
            tmp_path,
            smart_merge_detection=True,
            smart_merge_main_subjects=["feat: already merged (#123)"],
        ),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "unique_unharvested"
    assert candidate.cleanup_candidate is False
    assert candidate.git.smart_merge_equivalent_to_base is False
    assert "smart merge detection skipped because branch contains merge commits" in candidate.proof
    assert candidate.links["smart_merge_merge_commits"] == ["merge-sha"]


def test_smart_merge_patch_timeout_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    _stub_clean_git(monkeypatch, ahead=2, patch_equivalent=False)
    monkeypatch.setattr(
        mod,
        "branch_subjects_match_recent_main",
        lambda *_args, **_kwargs: (False, []),
    )

    def fake_patches_present(
        *_args: Any, lookup_errors: list[str] | None = None, **_kwargs: Any
    ) -> tuple[bool, list[str]]:
        if lookup_errors is not None:
            lookup_errors.append("patch-present rev-list failed: command timed out after 1s")
        return False, []

    monkeypatch.setattr(mod, "branch_patches_present_on_base", fake_patches_present)

    candidate = mod.classify_candidate(
        root,
        context=_context(
            tmp_path,
            smart_merge_detection=True,
            smart_merge_main_subjects=[],
        ),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "lookup_failed"
    assert candidate.cleanup_candidate is False
    assert candidate.git.lookup_failed is True
    assert candidate.git.inspect_timeout is True
    assert any("patch-present rev-list failed" in item for item in candidate.git.lookup_errors)
    assert any("inspect_timeout" in item for item in candidate.proof)


def test_smart_merge_merge_commit_lookup_failure_is_protected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    _stub_clean_git(monkeypatch, ahead=2, patch_equivalent=False)
    monkeypatch.setattr(
        mod,
        "branch_unique_merge_commits",
        lambda *_args, **_kwargs: (None, "command timed out after 1s"),
    )

    candidate = mod.classify_candidate(
        root,
        context=_context(
            tmp_path,
            smart_merge_detection=True,
            smart_merge_main_subjects=[],
        ),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "lookup_failed"
    assert candidate.cleanup_candidate is False
    assert candidate.git.lookup_failed is True
    assert candidate.git.inspect_timeout is True
    assert any(
        "smart merge merge-commit lookup failed" in item for item in candidate.git.lookup_errors
    )


def test_smart_merge_patch_commit_budget_is_protected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    _stub_clean_git(monkeypatch, ahead=100, patch_equivalent=False)
    monkeypatch.setattr(
        mod,
        "branch_subjects_match_recent_main",
        lambda *_args, **_kwargs: (False, []),
    )

    many_commits = "\n".join(f"{idx:040x}" for idx in range(mod.MAX_SMART_MERGE_PATCH_COMMITS + 1))

    def fake_run_git(args: list[str], *_args: Any, **_kwargs: Any) -> Any:
        if args[:3] == ["rev-list", "--reverse", "--no-merges"]:
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout=many_commits, stderr=""
            )
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(mod, "run_git", fake_run_git)

    candidate = mod.classify_candidate(
        root,
        context=_context(
            tmp_path,
            smart_merge_detection=True,
            smart_merge_main_subjects=[],
        ),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "lookup_failed"
    assert candidate.cleanup_candidate is False
    assert candidate.git.lookup_failed is True
    assert any("commits exceeds budget" in item for item in candidate.git.lookup_errors)


def test_smart_merge_detection_keeps_unmatched_subjects_harvestable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    _stub_clean_git(monkeypatch, ahead=2, patch_equivalent=False)
    monkeypatch.setattr(
        mod,
        "branch_unique_commit_subjects",
        lambda *_args, **_kwargs: [
            "feat(scripts): list active sessions",
            "fix(swarm): new unmerged behavior",
        ],
    )
    monkeypatch.setattr(
        mod,
        "branch_patches_present_on_base",
        lambda *_args, **_kwargs: (False, []),
    )

    candidate = mod.classify_candidate(
        root,
        context=_context(
            tmp_path,
            smart_merge_detection=True,
            smart_merge_main_subjects=["feat(scripts): list active sessions (#7260)"],
        ),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "unique_unharvested"
    assert candidate.cleanup_candidate is False
    assert candidate.git.smart_merge_equivalent_to_base is False


def test_smart_merge_detection_keeps_unapplied_patches_harvestable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    _stub_clean_git(monkeypatch, ahead=2, patch_equivalent=False)
    monkeypatch.setattr(
        mod,
        "branch_subjects_match_recent_main",
        lambda *_args, **_kwargs: (False, []),
    )
    monkeypatch.setattr(
        mod,
        "branch_patches_present_on_base",
        lambda *_args, **_kwargs: (False, ["2f2a1f6"]),
    )

    candidate = mod.classify_candidate(
        root,
        context=_context(
            tmp_path,
            smart_merge_detection=True,
            smart_merge_main_subjects=[],
        ),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "unique_unharvested"
    assert candidate.cleanup_candidate is False
    assert candidate.git.smart_merge_equivalent_to_base is False


def test_smart_merge_detection_log_failure_keeps_candidate_harvestable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    _stub_clean_git(monkeypatch, ahead=2, patch_equivalent=False)
    monkeypatch.setattr(mod, "branch_unique_commit_subjects", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        mod,
        "branch_patches_present_on_base",
        lambda *_args, **_kwargs: (False, []),
    )

    candidate = mod.classify_candidate(
        root,
        context=_context(
            tmp_path,
            smart_merge_detection=True,
            smart_merge_main_subjects=["feat(scripts): list active sessions (#7260)"],
        ),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "unique_unharvested"
    assert candidate.decision == "harvest_candidate"


def test_commit_subject_matching_is_loose_but_not_unbounded() -> None:
    import codex_worktree_value_inventory as mod

    main_subjects = [
        "feat(scripts): list active sessions (#7260)",
        "docs(status): inventory receipt (#7258)",
    ]

    assert mod.commit_subject_matches_recent_main(
        "feat(scripts): list active sessions [lane: P42]",
        main_subjects,
    )
    assert mod.commit_subject_matches_recent_main(
        "docs(status): inventory receipt",
        main_subjects,
    )
    assert not mod.commit_subject_matches_recent_main(
        "fix(swarm): unrelated dispatch behavior",
        main_subjects,
    )


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
    assert candidate.cleanup_safety.safety_class == "unknown_preserve"
    assert "unsafe_to_delete" in candidate.cleanup_safety.signals


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
    assert summary["count_by_safety_class"]["stale_residue"] == 1
    assert summary["count_by_safety_class"]["unsafe_to_delete"] == 1
    assert summary["top_cleanup_candidates"][0]["path"] == str(cleanup_root)
    assert summary["top_cleanup_candidates"][0]["safety_class"] == "stale_residue"
    assert summary["top_cleanup_candidates"][0]["cleanup_safety"]["safe_to_delete"] is False
    assert summary["top_unique_unharvested"][0]["path"] == str(unique_root)
    assert summary["top_unique_unharvested"][0]["safety_class"] == "unsafe_to_delete"


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


def test_resolve_default_roots_picks_canonical_then_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    repo = tmp_path / "repo"
    canonical = repo / mod.DEFAULT_CANONICAL_REL_ROOT
    legacy = tmp_path / "home" / ".codex" / "worktrees"
    canonical.mkdir(parents=True)
    legacy.mkdir(parents=True)
    monkeypatch.setattr(mod, "DEFAULT_LEGACY_ROOT", legacy)

    roots = mod.resolve_default_roots(repo)

    assert roots == [canonical.resolve(), legacy.resolve()]


def test_resolve_default_roots_skips_missing_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    repo = tmp_path / "repo"
    repo.mkdir()
    legacy = tmp_path / "home" / ".codex" / "worktrees"
    legacy.mkdir(parents=True)
    monkeypatch.setattr(mod, "DEFAULT_LEGACY_ROOT", legacy)

    roots = mod.resolve_default_roots(repo)

    assert roots == [legacy.resolve()]


def test_resolve_default_roots_empty_when_neither_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    repo = tmp_path / "repo"
    repo.mkdir()
    legacy = tmp_path / "nonexistent" / ".codex" / "worktrees"
    monkeypatch.setattr(mod, "DEFAULT_LEGACY_ROOT", legacy)

    roots = mod.resolve_default_roots(repo)

    assert roots == []


def test_resolve_default_roots_dedups_when_paths_resolve_equal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    repo = tmp_path / "repo"
    canonical = repo / mod.DEFAULT_CANONICAL_REL_ROOT
    canonical.mkdir(parents=True)
    legacy_alias = tmp_path / "legacy-link"
    legacy_alias.symlink_to(canonical, target_is_directory=True)
    monkeypatch.setattr(mod, "DEFAULT_LEGACY_ROOT", legacy_alias)

    roots = mod.resolve_default_roots(repo)

    assert roots == [canonical.resolve()]


def test_resolve_default_roots_uses_git_common_dir_for_managed_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    repo = tmp_path / "repo"
    main_canonical = repo / mod.DEFAULT_CANONICAL_REL_ROOT
    managed_worktree = main_canonical / "codex-session"
    git_common_dir = repo / ".git"
    main_canonical.mkdir(parents=True)
    managed_worktree.mkdir()
    git_common_dir.mkdir()
    legacy = tmp_path / "home" / ".codex" / "worktrees"
    legacy.mkdir(parents=True)
    monkeypatch.setattr(mod, "DEFAULT_LEGACY_ROOT", legacy)
    monkeypatch.setattr(mod, "_git_common_dir", lambda _repo: git_common_dir)

    roots = mod.resolve_default_roots(managed_worktree)

    assert roots == [main_canonical.resolve(), legacy.resolve()]


def test_candidate_roots_from_unions_entries_across_roots(tmp_path: Path) -> None:
    import codex_worktree_value_inventory as mod

    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    (root_a / "alpha").mkdir(parents=True)
    (root_a / "beta").mkdir()
    (root_b / "gamma").mkdir(parents=True)
    (root_a / "ignored.txt").write_text("not a dir")

    result = mod.candidate_roots_from([root_a, root_b])

    assert result == [root_a / "alpha", root_a / "beta", root_b / "gamma"]


def test_candidate_roots_from_dedups_same_resolved_path(tmp_path: Path) -> None:
    import codex_worktree_value_inventory as mod

    root_a = tmp_path / "a"
    root_b_alias = tmp_path / "b-link"
    (root_a / "alpha").mkdir(parents=True)
    root_b_alias.symlink_to(root_a, target_is_directory=True)

    result = mod.candidate_roots_from([root_a, root_b_alias])

    assert result == [root_a / "alpha"]


def test_candidate_roots_from_applies_overall_limit(tmp_path: Path) -> None:
    import codex_worktree_value_inventory as mod

    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    for name in ("alpha", "beta", "gamma"):
        (root_a / name).mkdir(parents=True)
    for name in ("delta",):
        (root_b / name).mkdir(parents=True)

    result = mod.candidate_roots_from([root_a, root_b], limit=2)

    assert len(result) == 2
    assert result[0].name == "alpha"
    assert result[1].name == "beta"


def test_candidate_roots_from_skips_missing_roots(tmp_path: Path) -> None:
    import codex_worktree_value_inventory as mod

    root_a = tmp_path / "a"
    missing = tmp_path / "missing"
    (root_a / "alpha").mkdir(parents=True)

    result = mod.candidate_roots_from([missing, root_a])

    assert result == [root_a / "alpha"]


def test_build_parser_root_action_append(tmp_path: Path) -> None:
    import codex_worktree_value_inventory as mod

    parser = mod.build_parser()

    args = parser.parse_args(["--root", "/tmp/a", "--root", "/tmp/b"])

    assert args.root == [Path("/tmp/a"), Path("/tmp/b")]


def test_build_parser_root_omitted_yields_none(tmp_path: Path) -> None:
    import codex_worktree_value_inventory as mod

    parser = mod.build_parser()

    args = parser.parse_args([])

    assert args.root is None


def test_build_parser_smart_merge_detection_default_off() -> None:
    import codex_worktree_value_inventory as mod

    parser = mod.build_parser()

    assert parser.parse_args([]).smart_merge_detection is False
    assert parser.parse_args(["--smart-merge-detection"]).smart_merge_detection is True


def test_build_parser_include_pr_state_default_off() -> None:
    import codex_worktree_value_inventory as mod

    parser = mod.build_parser()

    assert parser.parse_args([]).include_pr_state is False
    assert parser.parse_args(["--include-pr-state"]).include_pr_state is True


def test_lookup_open_prs_uses_cached_open_pr_heads_when_provided(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    called: dict[str, bool] = {"subprocess": False}

    def fake_run_cmd(*_args: Any, **_kwargs: Any) -> Any:
        called["subprocess"] = True
        raise RuntimeError("should not be called when cache is supplied")

    monkeypatch.setattr(mod, "run_cmd", fake_run_cmd)

    cache: dict[str, list[dict[str, Any]]] = {
        "feat/x": [{"number": 123, "title": "Open feature", "url": "https://example.test/pr/123"}],
        "feat/y": [{"number": 456, "title": "Other PR", "url": "https://example.test/pr/456"}],
    }

    prs, failed, err = mod.lookup_open_prs(
        tmp_path,
        "feat/x",
        timeout=1,
        skip_gh=True,
        cached_open_pr_heads=cache,
    )

    assert prs == [{"number": 123, "title": "Open feature", "url": "https://example.test/pr/123"}]
    assert failed is False
    assert err is None
    assert called["subprocess"] is False


def test_lookup_open_prs_returns_empty_when_branch_not_in_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    def fake_run_cmd(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("should not be called when cache is supplied")

    monkeypatch.setattr(mod, "run_cmd", fake_run_cmd)

    prs, failed, err = mod.lookup_open_prs(
        tmp_path,
        "feat/unknown",
        timeout=1,
        skip_gh=True,
        cached_open_pr_heads={"feat/x": [{"number": 123}]},
    )

    assert prs == []
    assert failed is False
    assert err is None


def test_prefetch_open_pr_heads_uses_open_only_cache_for_open_heads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    calls: list[list[str]] = []

    def fake_run_cmd(args: list[str], *_args: Any, **_kwargs: Any) -> Any:
        calls.append(args)
        if "--state" in args and args[args.index("--state") + 1] == "open":
            payload = [
                {
                    "number": 901,
                    "title": "Older still-open PR",
                    "url": "https://example.test/pr/901",
                    "headRefName": "feat/older-open",
                    "body": "",
                    "state": "OPEN",
                    "headRefOid": "older-head",
                }
            ]
        else:
            payload = [
                {
                    "number": 902,
                    "title": "Recent closed PR",
                    "url": "https://example.test/pr/902",
                    "headRefName": "feat/recent-closed",
                    "body": "",
                    "state": "CLOSED",
                    "headRefOid": "closed-head",
                }
            ]
        return mod.subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=json.dumps(payload),
            stderr="",
        )

    monkeypatch.setattr(mod, "run_cmd", fake_run_cmd)

    cache, records, branch_records, failed, err = mod.prefetch_open_pr_heads(
        tmp_path,
        timeout=1,
    )

    assert failed is False
    assert err is None
    assert [call[call.index("--state") + 1] for call in calls] == ["open", "all"]
    assert cache == {
        "feat/older-open": [
            {
                "number": 901,
                "title": "Older still-open PR",
                "url": "https://example.test/pr/901",
                "headRefName": "feat/older-open",
                "body": "",
                "state": "OPEN",
                "headRefOid": "older-head",
            }
        ]
    }
    assert records == cache["feat/older-open"]
    assert branch_records == {
        "feat/recent-closed": [
            {
                "number": 902,
                "title": "Recent closed PR",
                "url": "https://example.test/pr/902",
                "headRefName": "feat/recent-closed",
                "body": "",
                "state": "CLOSED",
                "headRefOid": "closed-head",
            }
        ]
    }


def test_classify_candidate_marks_open_pr_when_cache_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    monkeypatch.setattr(mod, "git_branch", lambda *_a, **_k: ("feat/x", False, None))
    monkeypatch.setattr(mod, "git_head", lambda *_a, **_k: ("abc1234", False, None))
    monkeypatch.setattr(mod, "git_status_dirty", lambda *_a, **_k: (False, False, None))
    monkeypatch.setattr(mod, "git_ahead_behind", lambda *_a, **_k: (3, 0, False, None))
    monkeypatch.setattr(mod, "is_patch_equivalent", lambda *_a, **_k: False)

    cache: dict[str, list[dict[str, Any]]] = {
        "feat/x": [
            {"number": 999, "title": "Open PR for feat/x", "url": "https://example.test/pr/999"}
        ],
    }
    ctx = _context(
        tmp_path,
        strict_repo_identity=False,
        open_pr_heads_cache=cache,
        skip_gh=True,
    )

    candidate = mod.classify_candidate(
        root,
        context=ctx,
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "open_pr_or_outbox"
    assert candidate.decision == "preserve"
    assert candidate.links["open_prs"] == [
        {"number": 999, "title": "Open PR for feat/x", "url": "https://example.test/pr/999"}
    ]


def test_classify_candidate_preserves_closed_pr_superseded_by_open_pr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    _stub_clean_git(monkeypatch, branch="feat/stale", ahead=3, patch_equivalent=False)

    def fail_if_called(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("cached branch PR records should avoid per-candidate gh calls")

    monkeypatch.setattr(mod, "run_cmd", fail_if_called)

    candidate = mod.classify_candidate(
        root,
        context=_context(
            tmp_path,
            branch_pr_records_cache={
                "feat/stale": [
                    {
                        "number": 8255,
                        "state": "CLOSED",
                        "title": "stale source",
                        "url": "https://example.test/pr/8255",
                    }
                ],
            },
            open_pr_records_cache=[
                {
                    "number": 8543,
                    "title": "feat: replacement",
                    "body": "Re-cut fresh against current main; supersedes stale PR #8255.",
                    "url": "https://example.test/pr/8543",
                    "headRefName": "feat/recut",
                }
            ],
        ),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "open_pr_or_outbox"
    assert candidate.decision == "preserve"
    assert "open PR explicitly supersedes closed source PR for branch" in candidate.proof
    assert candidate.links["superseding_open_prs"] == [
        {
            "number": 8543,
            "title": "feat: replacement",
            "url": "https://example.test/pr/8543",
            "headRefName": "feat/recut",
            "supersedes_pr": 8255,
        }
    ]


def test_classify_candidate_keeps_generic_pr_reference_harvestable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)
    _stub_clean_git(monkeypatch, branch="feat/stale", ahead=3, patch_equivalent=False)
    monkeypatch.setattr(
        mod,
        "lookup_branch_prs",
        lambda *_args, **_kwargs: (
            [
                {
                    "number": 8255,
                    "state": "CLOSED",
                    "title": "stale source",
                    "url": "https://example.test/pr/8255",
                }
            ],
            False,
            None,
        ),
    )

    candidate = mod.classify_candidate(
        root,
        context=_context(
            tmp_path,
            branch_pr_records_cache={
                "feat/stale": [
                    {
                        "number": 8255,
                        "state": "CLOSED",
                        "title": "stale source",
                        "url": "https://example.test/pr/8255",
                    }
                ],
            },
            open_pr_records_cache=[
                {
                    "number": 8543,
                    "title": "feat: related work",
                    "body": "Refs #8255 for historical background.",
                    "url": "https://example.test/pr/8543",
                    "headRefName": "feat/related",
                }
            ],
        ),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.classification == "unique_unharvested"
    assert candidate.links["superseding_open_prs"] == []


# ---------------------------------------------------------------------------
# Git-timeout hardening: a hung `git status` (or any git lookup) must never
# hang the inventory, and a timed-out candidate must stay protected.
# ---------------------------------------------------------------------------


def test_run_cmd_timeout_returns_124_and_never_hangs(tmp_path: Path) -> None:
    """A child that would outlive the timeout is killed (whole session) and
    run_cmd returns promptly with the timeout annotation in stderr."""
    import subprocess as subprocess_mod
    import time

    import codex_worktree_value_inventory as mod

    started = time.monotonic()
    # The backgrounded grandchild inherits the pipes; with plain
    # subprocess.run(timeout=...) the post-kill drain would block on it.
    proc = mod.run_cmd(
        ["/bin/sh", "-c", "sleep 30 & exec sleep 30"],
        tmp_path,
        timeout=1,
    )
    elapsed = time.monotonic() - started

    assert isinstance(proc, subprocess_mod.CompletedProcess)
    assert proc.returncode == 124
    assert "timed out after 1s" in proc.stderr
    assert elapsed < 15, f"run_cmd must return promptly after a timeout, took {elapsed:.1f}s"


def test_run_cmd_missing_binary_still_returns_completed_process(tmp_path: Path) -> None:
    import codex_worktree_value_inventory as mod

    proc = mod.run_cmd(["/nonexistent-binary-for-test"], tmp_path, timeout=1)

    assert proc.returncode == 124
    assert proc.stdout == ""


def test_run_cmd_replaces_invalid_utf8_output(tmp_path: Path) -> None:
    import codex_worktree_value_inventory as mod

    proc = mod.run_cmd(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(b'patch\\xffpayload')",
        ],
        tmp_path,
        timeout=5,
    )

    assert proc.returncode == 0
    assert "patch\ufffdpayload" in proc.stdout


def test_timeout_raising_runner_marks_candidate_inspect_timeout_protected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fake Popen whose communicate raises subprocess.TimeoutExpired: the
    candidate is annotated inspect_timeout and treated as protected (never
    safe-to-clean), and classification completes instead of crashing."""
    import subprocess as subprocess_mod

    import codex_worktree_value_inventory as mod

    root = _candidate(tmp_path)

    class FakeHungPopen:
        def __init__(self, args: list[str], **_kwargs: Any) -> None:
            self.args = args
            self.pid = 999_999_999  # killpg on this raises and falls back to kill()
            self.returncode: int | None = None
            self.stdout = None
            self.stderr = None
            self._calls = 0

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            self._calls += 1
            if self._calls == 1:
                raise subprocess_mod.TimeoutExpired(self.args, timeout or 0)
            return "", ""

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr(mod.subprocess, "Popen", FakeHungPopen)

    candidate = mod.classify_candidate(
        root,
        context=_context(tmp_path),
        size_bytes=1024,
        size_lookup_failed=False,
    )

    assert candidate.git.inspect_timeout is True
    assert any("inspect_timeout" in item for item in candidate.proof)
    assert candidate.classification in mod.PROTECTED_CLASSES
    assert candidate.cleanup_candidate is False
    assert candidate.decision == "preserve"
    assert candidate.cleanup_safety.safe_to_delete is False
    assert candidate.cleanup_safety.preserve is True


def test_status_timeout_protects_candidate_and_run_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A candidate whose `git status` timed out is protected; remaining
    candidates are still classified normally and the summary counts the
    timeout."""
    import codex_worktree_value_inventory as mod

    hung_root = _candidate(tmp_path, "hung")
    clean_root = _candidate(tmp_path, "clean")
    _stub_clean_git(monkeypatch, ahead=0)

    def fake_status(repo_path: Path, *, timeout: int) -> tuple[bool, bool, str | None]:
        if "hung" in str(repo_path):
            return True, True, f"command timed out after {timeout}s: git status --porcelain"
        return False, False, None

    monkeypatch.setattr(mod, "git_status_dirty", fake_status)

    hung = mod.classify_candidate(
        hung_root, context=_context(tmp_path), size_bytes=1024, size_lookup_failed=False
    )
    clean = mod.classify_candidate(
        clean_root, context=_context(tmp_path), size_bytes=1024, size_lookup_failed=False
    )

    assert hung.git.inspect_timeout is True
    assert hung.classification in mod.PROTECTED_CLASSES
    assert hung.cleanup_candidate is False
    assert hung.cleanup_safety.safe_to_delete is False
    # The run continued: the clean candidate is unaffected.
    assert clean.git.inspect_timeout is False
    assert clean.classification == "unregistered_git_residue"

    summary = mod.build_summary([hung, clean])
    assert summary["inspect_timeouts"] == 1


def test_build_parser_git_timeout_seconds_alias() -> None:
    import codex_worktree_value_inventory as mod

    parser = mod.build_parser()

    assert mod.GIT_TIMEOUT_SECONDS == 30
    assert parser.parse_args([]).git_timeout == mod.GIT_TIMEOUT_SECONDS
    assert parser.parse_args(["--git-timeout-seconds", "7"]).git_timeout == 7
    assert parser.parse_args(["--git-timeout", "9"]).git_timeout == 9
