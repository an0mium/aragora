"""External Repository Targeting.

Enables Aragora's pipeline to target any git repository,
not just itself. Handles cloning, worktree isolation,
and PR creation for external codebases.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RepoTarget:
    """Target repository specification."""

    url: str  # git clone URL
    branch: str = "main"
    subdirectory: str = ""  # focus on specific subdirectory
    clone_depth: int = 1  # shallow clone depth, 0 for full
    auth_token: str = ""  # optional auth for private repos

    @property
    def repo_name(self) -> str:
        """Extract repo name from URL."""
        name = self.url.rstrip("/").split("/")[-1]
        if name.endswith(".git"):
            name = name[:-4]
        return name

    @property
    def clone_url(self) -> str:
        """URL with auth token if provided."""
        if self.auth_token and "github.com" in self.url:
            return self.url.replace(
                "https://github.com",
                f"https://x-access-token:{self.auth_token}@github.com",
            )
        return self.url


@dataclass
class RepoContext:
    """Context gathered from an external repository."""

    target: RepoTarget
    local_path: Path
    file_count: int = 0
    language_breakdown: dict[str, int] = field(default_factory=dict)
    has_tests: bool = False
    has_ci: bool = False
    readme_summary: str = ""
    entry_points: list[str] = field(default_factory=list)


@dataclass
class ExternalRunResult:
    """Result of running a pipeline on an external repo."""

    target: RepoTarget
    worktree_path: Path | None = None
    branch_name: str = ""
    files_changed: list[str] = field(default_factory=list)
    pr_url: str = ""
    success: bool = False
    error: str = ""


class ExternalRepoManager:
    """Manages external repository targeting for pipeline execution.

    Lifecycle:
    1. clone() — shallow clone the target repo
    2. analyze() — gather codebase context
    3. prepare_worktree() — create isolated branch for changes
    4. After pipeline execution, create_pr() — push and open PR
    5. cleanup() — remove local clone
    """

    def __init__(self, work_dir: str | None = None) -> None:
        self._work_dir = (
            Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="aragora-ext-"))
        )
        self._clones: dict[str, Path] = {}

    def clone(self, target: RepoTarget) -> Path:
        """Clone the target repository.

        Returns:
            Path to the local clone.
        """
        clone_path = self._work_dir / target.repo_name
        if clone_path.exists():
            return clone_path

        cmd = ["git", "clone"]
        if target.clone_depth > 0:
            cmd.extend(["--depth", str(target.clone_depth)])
        if target.branch:
            cmd.extend(["--branch", target.branch])
        cmd.extend([target.clone_url, str(clone_path)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"Clone failed: {result.stderr}")

        self._clones[target.repo_name] = clone_path
        return clone_path

    def analyze(self, target: RepoTarget, clone_path: Path) -> RepoContext:
        """Analyze the cloned repository structure."""
        ctx = RepoContext(target=target, local_path=clone_path)

        scan_path = clone_path / target.subdirectory if target.subdirectory else clone_path

        # Count files and detect languages
        extensions: dict[str, int] = {}
        file_count = 0
        for root, _dirs, files in os.walk(scan_path):
            if ".git" in root:
                continue
            for f in files:
                file_count += 1
                ext = Path(f).suffix.lower()
                if ext:
                    extensions[ext] = extensions.get(ext, 0) + 1

        ctx.file_count = file_count
        ctx.language_breakdown = dict(
            sorted(extensions.items(), key=lambda x: x[1], reverse=True)[:10]
        )

        # Detect tests and CI
        ctx.has_tests = any(
            (clone_path / d).exists() for d in ["tests", "test", "spec", "__tests__"]
        )
        ctx.has_ci = any(
            (clone_path / f).exists()
            for f in [".github/workflows", ".gitlab-ci.yml", "Jenkinsfile", ".circleci"]
        )

        # Read README
        for readme_name in ["README.md", "README.rst", "README.txt", "README"]:
            readme_path = clone_path / readme_name
            if readme_path.exists():
                content = readme_path.read_text(errors="ignore")
                ctx.readme_summary = content[:2000]
                break

        # Detect entry points
        for entry in ["main.py", "app.py", "index.js", "index.ts", "src/main.rs", "cmd/main.go"]:
            if (clone_path / entry).exists():
                ctx.entry_points.append(entry)

        return ctx

    def prepare_worktree(self, clone_path: Path, branch_name: str) -> Path:
        """Create a new branch in the clone for pipeline changes."""
        result = subprocess.run(
            ["git", "checkout", "-b", branch_name],
            cwd=clone_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Branch creation failed: {result.stderr}")
        return clone_path

    def get_changed_files(self, clone_path: Path) -> list[str]:
        """List files changed in the working tree."""
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=clone_path,
            capture_output=True,
            text=True,
        )
        return [f for f in result.stdout.strip().split("\n") if f]

    def cleanup(self, target: RepoTarget) -> None:
        """Remove the local clone."""
        clone_path = self._clones.pop(target.repo_name, None)
        if clone_path and clone_path.exists():
            import shutil

            shutil.rmtree(clone_path, ignore_errors=True)
