"""Tests for session artifact prevention in autonomous worker lanes.

Regression coverage for issue #879: autonomous workers that commit
.codex_session_meta.json must be detected and the artifact stripped from
deliverable output, not treated as valid user work.

Forensic evidence from the #873 Boss-loop failure showed a worker
committed .codex_session_meta.json alongside wrong-scope files.  The
supervisor accepted it as a valid changed path.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from aragora.nomic.dev_coordination import DevCoordinationStore
from aragora.swarm.supervisor import SwarmSupervisor
from aragora.swarm.worker_launcher import (
    SESSION_ARTIFACTS,
    LaunchConfig,
    WorkerLauncher,
    WorkerProcess,
)


UTC = __import__("datetime").timezone.utc


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "git", "init", "-b", "main")
    _run(repo, "git", "config", "user.email", "test@example.com")
    _run(repo, "git", "config", "user.name", "Test User")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _run(repo, "git", "add", "README.md")
    _run(repo, "git", "commit", "-m", "initial")
    _run(repo, "git", "remote", "add", "origin", str(repo))
    _run(repo, "git", "update-ref", "refs/remotes/origin/main", "HEAD")
    return repo


@pytest.fixture()
def store(repo: Path) -> DevCoordinationStore:
    return DevCoordinationStore(repo_root=repo)


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), cwd=cwd, text=True, capture_output=True, check=True)


def _head(repo: Path) -> str:
    return _run(repo, "git", "rev-parse", "HEAD").stdout.strip()


# ---------------------------------------------------------------------------
# SESSION_ARTIFACTS constant
# ---------------------------------------------------------------------------


class TestSessionArtifactsConstant:
    """Verify the canonical list of session artifacts."""

    def test_contains_codex_session_meta(self) -> None:
        assert ".codex_session_meta.json" in SESSION_ARTIFACTS

    def test_contains_codex_session_log(self) -> None:
        assert ".codex_session.log" in SESSION_ARTIFACTS

    def test_is_frozen(self) -> None:
        assert isinstance(SESSION_ARTIFACTS, frozenset)


# ---------------------------------------------------------------------------
# _strip_session_artifacts
# ---------------------------------------------------------------------------


class TestStripSessionArtifacts:
    """Test SwarmSupervisor._strip_session_artifacts."""

    def test_strips_session_meta(self) -> None:
        paths = [".codex_session_meta.json", "aragora/live/package.json"]
        result = SwarmSupervisor._strip_session_artifacts(paths)
        assert result == ["aragora/live/package.json"]

    def test_strips_session_log(self) -> None:
        paths = [".codex_session.log", "src/app.ts"]
        result = SwarmSupervisor._strip_session_artifacts(paths)
        assert result == ["src/app.ts"]

    def test_keeps_non_artifacts(self) -> None:
        paths = ["aragora/live/package.json", "README.md"]
        result = SwarmSupervisor._strip_session_artifacts(paths)
        assert result == paths

    def test_empty_list(self) -> None:
        assert SwarmSupervisor._strip_session_artifacts([]) == []

    def test_only_artifacts_returns_empty(self) -> None:
        """If session artifacts are the ONLY changes, result is empty."""
        paths = [".codex_session_meta.json", ".codex_session.log"]
        result = SwarmSupervisor._strip_session_artifacts(paths)
        assert result == []

    def test_nested_path_with_same_name_stripped(self) -> None:
        """Session artifacts under subdirs should also be stripped."""
        paths = ["subdir/.codex_session_meta.json", "aragora/live/app.ts"]
        result = SwarmSupervisor._strip_session_artifacts(paths)
        assert result == ["aragora/live/app.ts"]


# ---------------------------------------------------------------------------
# Worker prompt regression
# ---------------------------------------------------------------------------


class TestWorkerPromptNoGitAddAll:
    """Verify the worker prompt no longer instructs git add -A."""

    def test_prompt_does_not_instruct_git_add_all(self) -> None:
        """Prompt must not tell the worker to USE git add -A (warning against it is fine)."""
        work_order = {
            "title": "Test task",
            "description": "Do something",
            "file_scope": [],
            "metadata": {},
        }
        prompt = WorkerLauncher._build_prompt(work_order)
        # Must not contain an affirmative instruction to use git add -A
        assert "Stage all changed files with `git add -A`" not in prompt
        # Must contain a warning against using it
        assert "Do NOT use `git add -A`" in prompt

    def test_prompt_instructs_explicit_staging(self) -> None:
        work_order = {
            "title": "Test task",
            "description": "Do something",
            "file_scope": [],
            "metadata": {},
        }
        prompt = WorkerLauncher._build_prompt(work_order)
        assert "git add <file>" in prompt
        assert "session metadata files must not be committed" in prompt


# ---------------------------------------------------------------------------
# _apply_worker_result with session artifacts
# ---------------------------------------------------------------------------


class TestApplyWorkerResultSessionArtifacts:
    """Test that _apply_worker_result strips session artifacts."""

    def _make_supervisor(self, repo: Path, store: DevCoordinationStore) -> SwarmSupervisor:
        launcher = WorkerLauncher(config=LaunchConfig())
        return SwarmSupervisor(repo_root=repo, store=store, launcher=launcher)

    def test_session_meta_stripped_from_changed_paths(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        """Regression: .codex_session_meta.json must not appear as deliverable."""
        supervisor = self._make_supervisor(repo, store)
        item = {
            "work_order_id": "issue-879",
            "file_scope": [],
            "status": "dispatched",
            "lease_id": "",
            "target_agent": "codex",
        }
        result = WorkerProcess(
            work_order_id="issue-879",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=0,
            changed_paths=[
                ".codex_session_meta.json",
                "aragora/live/package.json",
            ],
            commit_shas=["abc123"],
            head_sha="abc123",
        )
        supervisor._apply_worker_result(item, result)

        assert ".codex_session_meta.json" not in item["changed_paths"]
        assert "aragora/live/package.json" in item["changed_paths"]

    def test_only_session_artifacts_rejected_not_completed(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        """If session artifacts are the ONLY changes, the lane must NOT be completed.

        The supervisor must fail closed: a worker that produced only harness
        metadata has not delivered real work, even if exit_code == 0.
        """
        supervisor = self._make_supervisor(repo, store)
        item = {
            "work_order_id": "empty-work",
            "file_scope": [],
            "status": "dispatched",
            "lease_id": "",
            "target_agent": "codex",
        }
        result = WorkerProcess(
            work_order_id="empty-work",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=0,
            changed_paths=[".codex_session_meta.json"],
            commit_shas=["def456"],
            head_sha="def456",
        )
        supervisor._apply_worker_result(item, result)

        assert item["changed_paths"] == []
        assert item["status"] != "completed"
        assert item["status"] == "needs_human"

    def test_only_session_log_rejected_not_completed(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        """Same as above but for .codex_session.log."""
        supervisor = self._make_supervisor(repo, store)
        item = {
            "work_order_id": "log-only",
            "file_scope": [],
            "status": "dispatched",
            "lease_id": "",
            "target_agent": "codex",
        }
        result = WorkerProcess(
            work_order_id="log-only",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=0,
            changed_paths=[".codex_session.log"],
            commit_shas=["abc"],
            head_sha="abc",
        )
        supervisor._apply_worker_result(item, result)

        assert item["status"] == "needs_human"
        assert "only session artifacts" in item.get("dispatch_error", "")

    def test_both_session_artifacts_rejected(self, repo: Path, store: DevCoordinationStore) -> None:
        """Both session artifacts together still yields no deliverables."""
        supervisor = self._make_supervisor(repo, store)
        item = {
            "work_order_id": "both-artifacts",
            "file_scope": [],
            "status": "dispatched",
            "lease_id": "",
            "target_agent": "codex",
        }
        result = WorkerProcess(
            work_order_id="both-artifacts",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=0,
            changed_paths=[".codex_session_meta.json", ".codex_session.log"],
            commit_shas=["xyz"],
            head_sha="xyz",
        )
        supervisor._apply_worker_result(item, result)

        assert item["status"] == "needs_human"

    def test_regression_issue_873_forensic_paths(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        """Regression: exact paths from the #873 failure.

        The worker committed:
        - .codex_session_meta.json  (artifact — must be stripped)
        - README.md                 (wrong scope — must trigger scope violation)
        - docs/LANDING_PAGE.md      (wrong scope — must trigger scope violation)
        """
        supervisor = self._make_supervisor(repo, store)
        item = {
            "work_order_id": "issue-873-forensic",
            "file_scope": ["aragora/live"],
            "status": "dispatched",
            "lease_id": "",
            "target_agent": "codex",
        }
        result = WorkerProcess(
            work_order_id="issue-873-forensic",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=0,
            changed_paths=[
                ".codex_session_meta.json",
                "README.md",
                "docs/LANDING_PAGE.md",
            ],
            commit_shas=["ghi789"],
            head_sha="ghi789",
        )
        supervisor._apply_worker_result(item, result)

        # Session artifact stripped
        assert ".codex_session_meta.json" not in item["changed_paths"]
        # Scope violation triggered for remaining wrong-scope files
        assert item["status"] == "scope_violation"


# ---------------------------------------------------------------------------
# Detached auto-commit prevention
# ---------------------------------------------------------------------------


class TestAutoCommitSessionArtifactPrevention:
    """Test that _auto_commit unstages session artifacts before committing."""

    def test_auto_commit_excludes_session_meta(self, repo: Path) -> None:
        """_auto_commit must not include .codex_session_meta.json in the commit."""
        # Create a session artifact and a real file in the worktree
        (repo / ".codex_session_meta.json").write_text('{"pid": 1}', encoding="utf-8")
        (repo / "real_work.py").write_text("print('hello')\n", encoding="utf-8")

        worker = WorkerProcess(
            work_order_id="detach-test",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
        )

        asyncio.run(WorkerLauncher._auto_commit(worker))

        # Check what was committed
        result = _run(repo, "git", "diff", "--name-only", "HEAD~1", "HEAD")
        committed_files = set(result.stdout.strip().splitlines())
        assert "real_work.py" in committed_files
        assert ".codex_session_meta.json" not in committed_files

    def test_auto_commit_excludes_session_log(self, repo: Path) -> None:
        """_auto_commit must not include .codex_session.log in the commit."""
        (repo / ".codex_session.log").write_text("log data", encoding="utf-8")
        (repo / "work.txt").write_text("content\n", encoding="utf-8")

        worker = WorkerProcess(
            work_order_id="detach-log-test",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
        )

        asyncio.run(WorkerLauncher._auto_commit(worker))

        result = _run(repo, "git", "diff", "--name-only", "HEAD~1", "HEAD")
        committed_files = set(result.stdout.strip().splitlines())
        assert "work.txt" in committed_files
        assert ".codex_session.log" not in committed_files

    def test_auto_commit_only_artifacts_skips_commit(self, repo: Path) -> None:
        """If only session artifacts exist, auto-commit skips (no commit created)."""
        initial_head = _head(repo)
        (repo / ".codex_session_meta.json").write_text('{"pid": 2}', encoding="utf-8")

        worker = WorkerProcess(
            work_order_id="artifacts-only",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
        )

        asyncio.run(WorkerLauncher._auto_commit(worker))

        # HEAD should not advance — no real changes to commit
        assert _head(repo) == initial_head


# ---------------------------------------------------------------------------
# _collect_changed_paths filtering
# ---------------------------------------------------------------------------


class TestCollectChangedPathsFiltering:
    """Test that _collect_changed_paths strips session artifacts."""

    def test_collect_excludes_session_meta(self, repo: Path) -> None:
        """Session artifacts in git diff/status must be stripped from results."""
        # Get initial HEAD
        initial = _run(repo, "git", "rev-parse", "HEAD").stdout.strip()

        # Create artifact + real file, commit both via raw git
        (repo / ".codex_session_meta.json").write_text("{}", encoding="utf-8")
        (repo / "real.py").write_text("x = 1\n", encoding="utf-8")
        _run(repo, "git", "add", "-A")
        _run(repo, "git", "commit", "-m", "test")
        head = _run(repo, "git", "rev-parse", "HEAD").stdout.strip()

        paths = asyncio.run(
            WorkerLauncher._collect_changed_paths(str(repo), initial_head=initial, head_sha=head)
        )
        assert "real.py" in paths
        assert ".codex_session_meta.json" not in paths


# ---------------------------------------------------------------------------
# End-to-end detached worker fail-closed path
# ---------------------------------------------------------------------------


class TestDetachedWorkerEndToEndFailClosed:
    """Prove the full detached path rejects artifact-only results.

    The critical path is:
        collect_detached_result() -> stripped changed_paths -> _apply_worker_result()

    When _collect_changed_paths strips session artifacts, result.changed_paths
    arrives empty at the supervisor.  But commit_shas may contain an
    --allow-empty commit.  The supervisor must reject this as needs_human.
    """

    def _make_supervisor(self, repo: Path, store: DevCoordinationStore) -> SwarmSupervisor:
        launcher = WorkerLauncher(config=LaunchConfig())
        return SwarmSupervisor(repo_root=repo, store=store, launcher=launcher)

    def test_detached_artifact_only_rejected(self, repo: Path, store: DevCoordinationStore) -> None:
        """Detached worker with only session artifacts must be rejected.

        Simulates the detached path: changed_paths is already empty (stripped
        by _collect_changed_paths), but commit_shas is non-empty from the
        --allow-empty auto-commit.  The supervisor must NOT mark this completed.
        """
        supervisor = self._make_supervisor(repo, store)
        item = {
            "work_order_id": "detached-artifact-only",
            "file_scope": [],
            "status": "dispatched",
            "lease_id": "",
            "target_agent": "codex",
        }
        # Simulates what collect_detached_result returns after
        # _collect_changed_paths strips the session artifact
        result = WorkerProcess(
            work_order_id="detached-artifact-only",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=0,
            changed_paths=[],  # already stripped by _collect_changed_paths
            commit_shas=["abc123"],  # --allow-empty commit
            head_sha="abc123",
        )
        supervisor._apply_worker_result(item, result)

        assert item["status"] == "needs_human"
        assert "no real deliverables" in item.get("dispatch_error", "")
        assert item["status"] != "completed"

    def test_detached_real_deliverables_plus_artifacts_accepted(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        """Detached worker with real deliverables (artifacts already stripped) succeeds."""
        supervisor = self._make_supervisor(repo, store)
        item = {
            "work_order_id": "detached-real-work",
            "file_scope": [],
            "status": "dispatched",
            "lease_id": "",
            "target_agent": "codex",
        }
        # Simulates detached result where real files remain after strip
        result = WorkerProcess(
            work_order_id="detached-real-work",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=0,
            changed_paths=["aragora/live/package.json"],  # real deliverable
            commit_shas=["def456"],
            head_sha="def456",
        )
        supervisor._apply_worker_result(item, result)

        # Should proceed to completed, not rejected
        assert item["status"] != "needs_human"

    def test_genuine_no_op_worker_fails_closed(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        """A worker with no changes AND no commits must fail closed to needs_human."""
        supervisor = self._make_supervisor(repo, store)
        item = {
            "work_order_id": "no-op",
            "file_scope": [],
            "status": "dispatched",
            "lease_id": "",
            "target_agent": "codex",
        }
        result = WorkerProcess(
            work_order_id="no-op",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=0,
            changed_paths=[],
            commit_shas=[],
            head_sha="abc",
        )
        supervisor._apply_worker_result(item, result)

        # Genuine no-op must fail closed — a worker that produces nothing is not "completed"
        assert item["status"] == "needs_human"
