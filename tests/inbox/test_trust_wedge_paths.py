from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import pytest


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return _run(cwd, "git", *args)


def _make_git_repo_with_linked_worktree(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    (repo / ".nomic").mkdir()
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "initial")

    linked = tmp_path / "linked-worktree"
    _git(repo, "worktree", "add", "-b", "feature/test", str(linked), "main")
    return repo, linked


def test_default_inbox_wedge_paths_use_repo_shared_nomic_for_linked_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, linked = _make_git_repo_with_linked_worktree(tmp_path)
    monkeypatch.delenv("ARAGORA_INBOX_TRUST_WEDGE_DB", raising=False)
    monkeypatch.delenv("ARAGORA_INBOX_TRUST_WEDGE_KEY_FILE", raising=False)
    monkeypatch.delenv("ARAGORA_DATA_DIR", raising=False)
    monkeypatch.delenv("ARAGORA_NOMIC_DIR", raising=False)
    monkeypatch.chdir(linked)

    import aragora.inbox.trust_wedge as trust_wedge_module

    trust_wedge_module = importlib.reload(trust_wedge_module)
    assert trust_wedge_module.DEFAULT_DB_PATH == repo / ".nomic" / "inbox_trust_wedge.db"
    assert trust_wedge_module.DEFAULT_SIGNING_KEY_PATH == (
        repo / ".nomic" / "inbox_trust_wedge_signing.key"
    )


def test_default_inbox_wedge_paths_respect_explicit_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    custom_dir = tmp_path / "custom-data"
    monkeypatch.setenv("ARAGORA_DATA_DIR", str(custom_dir))
    monkeypatch.delenv("ARAGORA_INBOX_TRUST_WEDGE_DB", raising=False)
    monkeypatch.delenv("ARAGORA_INBOX_TRUST_WEDGE_KEY_FILE", raising=False)

    import aragora.inbox.trust_wedge as trust_wedge_module

    trust_wedge_module = importlib.reload(trust_wedge_module)
    assert trust_wedge_module.DEFAULT_DB_PATH == custom_dir / "inbox_trust_wedge.db"
    assert trust_wedge_module.DEFAULT_SIGNING_KEY_PATH == (
        custom_dir / "inbox_trust_wedge_signing.key"
    )


def test_default_inbox_wedge_paths_respect_explicit_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "explicit-wedge.db"
    key_path = tmp_path / "explicit-wedge.key"
    monkeypatch.setenv("ARAGORA_INBOX_TRUST_WEDGE_DB", str(db_path))
    monkeypatch.setenv("ARAGORA_INBOX_TRUST_WEDGE_KEY_FILE", str(key_path))
    monkeypatch.delenv("ARAGORA_DATA_DIR", raising=False)
    monkeypatch.delenv("ARAGORA_NOMIC_DIR", raising=False)

    import aragora.inbox.trust_wedge as trust_wedge_module

    trust_wedge_module = importlib.reload(trust_wedge_module)
    assert trust_wedge_module.DEFAULT_DB_PATH == db_path
    assert trust_wedge_module.DEFAULT_SIGNING_KEY_PATH == key_path
