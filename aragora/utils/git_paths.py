from __future__ import annotations

import subprocess
from pathlib import Path


def git_common_repo_root(repo_root: Path) -> Path | None:
    """Return the canonical repo root behind git's common dir, if any."""

    proc = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo_root,
    )
    if proc.returncode != 0:
        return None
    common_dir = Path(proc.stdout.strip())
    if common_dir.name != ".git":
        return None
    return common_dir.parent.resolve()


def resolve_repo_fallback_path(
    candidate: Path,
    *,
    repo_root: Path,
    common_root: Path | None = None,
) -> Path:
    """Resolve a repo-local path, falling back through git-common-dir.

    Managed worktrees often keep shared runtime artifacts under the canonical
    repo root instead of each linked checkout. If ``candidate`` is missing from
    the current worktree but exists at the same relative path under the shared
    repo root, return that shared path instead.
    """

    repo_root = repo_root.resolve()
    if candidate.exists():
        return candidate.resolve()

    if candidate.is_absolute():
        try:
            relative = candidate.relative_to(repo_root)
        except ValueError:
            return candidate
    else:
        repo_relative = repo_root / candidate
        if repo_relative.exists():
            return repo_relative.resolve()
        relative = candidate

    shared_root = common_root if common_root is not None else git_common_repo_root(repo_root)
    if shared_root is not None:
        shared_candidate = (shared_root / relative).resolve()
        if shared_candidate.exists():
            return shared_candidate

    return candidate if candidate.is_absolute() else (repo_root / candidate).resolve()
