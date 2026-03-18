"""Tests for file-scope enforcement in supervisor-backed swarm execution.

Regression coverage for issue #878: workers that edit files outside their
permitted scope must be detected and rejected, not treated as successful.

The forensic evidence from the #873 Boss-loop failure showed a worker was
scoped to aragora/live/** but committed changes to README.md, docs/*, and
.codex_session_meta.json instead. The supervisor did not detect the scope
violation and waited the full timeout.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aragora.nomic.dev_coordination import DevCoordinationStore
from aragora.nomic.task_decomposer import SubTask, TaskDecomposition
from aragora.swarm.spec import SwarmSpec
from aragora.swarm.supervisor import SwarmSupervisor, _path_in_scope
from aragora.swarm.worker_launcher import LaunchConfig, WorkerLauncher, WorkerProcess
from aragora.worktree.lifecycle import ManagedWorktreeSession

UTC = timezone.utc


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
    return subprocess.run(
        list(args),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )


# ---------------------------------------------------------------------------
# Tests for _path_in_scope helper
# ---------------------------------------------------------------------------


class TestPathInScope:
    """Test the _path_in_scope helper function."""

    def test_exact_match(self) -> None:
        assert _path_in_scope("aragora/live/package.json", "aragora/live/package.json")

    def test_directory_prefix_match(self) -> None:
        assert _path_in_scope("aragora/live/package.json", "aragora/live")

    def test_deep_nested_match(self) -> None:
        assert _path_in_scope("aragora/live/src/app.tsx", "aragora/live")

    def test_glob_suffix_match(self) -> None:
        assert _path_in_scope("aragora/live/package.json", "aragora/live/**")

    def test_no_match_different_directory(self) -> None:
        assert not _path_in_scope("README.md", "aragora/live")

    def test_no_match_sibling_directory(self) -> None:
        assert not _path_in_scope("aragora/debate/orchestrator.py", "aragora/live")

    def test_no_match_partial_name(self) -> None:
        """aragora/live-extra should NOT match scope aragora/live."""
        assert not _path_in_scope("aragora/live-extra/foo.py", "aragora/live")

    def test_dotfile_outside_scope(self) -> None:
        assert not _path_in_scope(".codex_session_meta.json", "aragora/live")

    def test_root_docs_outside_scope(self) -> None:
        assert not _path_in_scope("docs/LANDING_PAGE.md", "aragora/live")

    def test_leading_dot_slash_normalized(self) -> None:
        assert _path_in_scope("./aragora/live/package.json", "aragora/live")

    def test_empty_scope_returns_false(self) -> None:
        assert not _path_in_scope("anything.py", "")

    def test_empty_path_returns_false(self) -> None:
        assert not _path_in_scope("", "aragora/live")

    def test_star_json_glob(self) -> None:
        """Standard glob *.json should match files in the same directory."""
        assert _path_in_scope("aragora/live/package.json", "aragora/live/*.json")

    def test_star_json_glob_no_nested(self) -> None:
        """*.json should NOT match nested subdirectories."""
        assert not _path_in_scope("aragora/live/src/config.json", "aragora/live/*.json")

    def test_double_star_ts_glob(self) -> None:
        """**/*.ts should match nested TypeScript files."""
        assert _path_in_scope("aragora/live/src/app.ts", "aragora/live/**/*.ts")

    def test_double_star_py_glob(self) -> None:
        """tests/**/*.py should match test files in subdirectories."""
        assert _path_in_scope("tests/swarm/test_supervisor.py", "tests/**/*.py")

    def test_question_mark_glob(self) -> None:
        """? wildcard should match a single character."""
        assert _path_in_scope("aragora/live/v1/app.ts", "aragora/live/v?/app.ts")

    def test_star_glob_matches_from_right(self) -> None:
        """PurePosixPath.match() matches relative patterns from the right."""
        # *.json matches any .json file at any depth — this is the canonical
        # PurePosixPath.match() behavior used by _path_matches_glob.
        assert _path_in_scope("aragora/live/package.json", "*.json")


# ---------------------------------------------------------------------------
# Tests for _check_file_scope_violations
# ---------------------------------------------------------------------------


class TestCheckFileScopeViolations:
    """Test SwarmSupervisor._check_file_scope_violations."""

    def test_no_scope_no_violations(self) -> None:
        """Work orders without file_scope should not trigger violations."""
        work_order = {"work_order_id": "test", "file_scope": []}
        violations = SwarmSupervisor._check_file_scope_violations(work_order, ["README.md"])
        assert violations == []

    def test_in_scope_no_violations(self) -> None:
        """Changes within scope should produce zero violations."""
        work_order = {
            "work_order_id": "test",
            "file_scope": ["aragora/live/package.json", "aragora/live/package-lock.json"],
        }
        violations = SwarmSupervisor._check_file_scope_violations(
            work_order,
            ["aragora/live/package.json", "aragora/live/package-lock.json"],
        )
        assert violations == []

    def test_directory_scope_allows_nested_files(self) -> None:
        """Directory scope should allow any file under that directory."""
        work_order = {
            "work_order_id": "test",
            "file_scope": ["aragora/live"],
        }
        violations = SwarmSupervisor._check_file_scope_violations(
            work_order,
            ["aragora/live/package.json", "aragora/live/src/app.tsx"],
        )
        assert violations == []

    def test_out_of_scope_detected(self) -> None:
        """Changes outside scope must be detected as violations."""
        work_order = {
            "work_order_id": "test",
            "file_scope": ["aragora/live/package.json", "aragora/live/package-lock.json"],
        }
        violations = SwarmSupervisor._check_file_scope_violations(
            work_order,
            ["README.md", "docs/LANDING_PAGE.md"],
        )
        assert len(violations) == 2
        assert all(v["type"] == "out_of_scope" for v in violations)

    def test_regression_issue_873_wrong_files(self) -> None:
        """Regression: exact #873 failure — worker edited wrong files.

        The worker was scoped to aragora/live/** but committed changes to:
        - .codex_session_meta.json
        - README.md
        - aragora-debate/README.md
        - docs/LANDING_PAGE.md
        - docs/STRATEGIC_ANALYSIS.md
        - docs/WHY_ADVERSARIAL_DEBATE.md
        """
        work_order = {
            "work_order_id": "issue-873",
            "file_scope": [
                "aragora/live/package.json",
                "aragora/live/package-lock.json",
                "aragora/live/eslint.config.mjs",
            ],
        }
        wrong_paths = [
            ".codex_session_meta.json",
            "README.md",
            "aragora-debate/README.md",
            "docs/LANDING_PAGE.md",
            "docs/STRATEGIC_ANALYSIS.md",
            "docs/WHY_ADVERSARIAL_DEBATE.md",
        ]
        violations = SwarmSupervisor._check_file_scope_violations(work_order, wrong_paths)
        assert len(violations) == 6
        violation_paths = {v["path"] for v in violations}
        assert violation_paths == set(wrong_paths)

    def test_mixed_in_and_out_of_scope(self) -> None:
        """If some paths are in scope and some are not, only out-of-scope are violations."""
        work_order = {
            "work_order_id": "test",
            "file_scope": ["aragora/live"],
        }
        violations = SwarmSupervisor._check_file_scope_violations(
            work_order,
            ["aragora/live/package.json", "README.md"],
        )
        assert len(violations) == 1
        assert violations[0]["path"] == "README.md"

    def test_no_changed_paths_no_violations(self) -> None:
        """Empty changed_paths should produce no violations."""
        work_order = {
            "work_order_id": "test",
            "file_scope": ["aragora/live"],
        }
        violations = SwarmSupervisor._check_file_scope_violations(work_order, [])
        assert violations == []


# ---------------------------------------------------------------------------
# Tests for _validate_file_scope (dispatch-time hallucination stripping)
# ---------------------------------------------------------------------------


class TestValidateFileScope:
    """Planner-hallucinated scope entries must be stripped before dispatch."""

    def test_hallucinated_root_stripped(self, tmp_path: Path) -> None:
        """Scope entries with non-existent top-level dirs are dropped."""
        wt = tmp_path / "worktree"
        wt.mkdir()
        (wt / ".git").write_text("gitdir: /tmp/fake\n")  # mark as git checkout
        (wt / "aragora" / "nomic").mkdir(parents=True)
        (wt / "tests").mkdir()

        raw_scope = [
            "src/orchestrator/hardened_orchestrator.py",  # hallucinated
            "aragora/nomic/hardened_orchestrator.py",  # real
            "tests/truth_suite/test_budget_caps.py",  # real root
        ]
        valid = SwarmSupervisor._validate_file_scope(raw_scope, str(wt))
        assert "src/orchestrator/hardened_orchestrator.py" not in valid
        assert "aragora/nomic/hardened_orchestrator.py" in valid
        assert "tests/truth_suite/test_budget_caps.py" in valid

    def test_all_hallucinated_returns_empty(self, tmp_path: Path) -> None:
        wt = tmp_path / "worktree"
        wt.mkdir()
        (wt / ".git").write_text("gitdir: /tmp/fake\n")
        scope = ["src/foo.py", "lib/bar.py"]
        assert SwarmSupervisor._validate_file_scope(scope, str(wt)) == []

    def test_empty_worktree_path_passes_through(self) -> None:
        scope = ["aragora/foo.py"]
        assert SwarmSupervisor._validate_file_scope(scope, "") == scope

    def test_all_valid_passes_through(self, tmp_path: Path) -> None:
        wt = tmp_path / "worktree"
        wt.mkdir()
        (wt / ".git").write_text("gitdir: /tmp/fake\n")
        (wt / "aragora").mkdir()
        (wt / "tests").mkdir()

        scope = ["aragora/swarm/supervisor.py", "tests/swarm/test_supervisor.py"]
        assert SwarmSupervisor._validate_file_scope(scope, str(wt)) == scope

    def test_no_git_dir_skips_validation(self, tmp_path: Path) -> None:
        """Directories without .git (e.g. test fixtures) skip validation."""
        wt = tmp_path / "worktree"
        wt.mkdir()
        scope = ["src/foo.py"]
        assert SwarmSupervisor._validate_file_scope(scope, str(wt)) == scope

    def test_enforcement_after_stripping(self, tmp_path: Path) -> None:
        """End-to-end: hallucinated scope stripped → worker edits real files →
        enforcement passes (because file_scope is now empty/valid).

        Regression test for B-3 benchmark: planner generated src/orchestrator/
        paths but real code lives at aragora/nomic/, causing 0% deliverable rate.
        """
        wt = tmp_path / "worktree"
        wt.mkdir()
        (wt / ".git").write_text("gitdir: /tmp/fake\n")
        (wt / "aragora" / "nomic").mkdir(parents=True)

        # Planner hallucinated all scope entries
        raw_scope = ["src/orchestrator/hardened_orchestrator.py"]
        valid_scope = SwarmSupervisor._validate_file_scope(raw_scope, str(wt))
        assert valid_scope == []

        # Worker edits real file — with empty scope, enforcement is open
        work_order = {"work_order_id": "test", "file_scope": valid_scope}
        violations = SwarmSupervisor._check_file_scope_violations(
            work_order,
            ["aragora/nomic/hardened_orchestrator.py"],
        )
        assert violations == [], (
            "Real edits must not be rejected after hallucinated scope is stripped"
        )


# ---------------------------------------------------------------------------
# Tests for _mark_scope_violation
# ---------------------------------------------------------------------------


class TestMarkScopeViolation:
    """Test SwarmSupervisor._mark_scope_violation."""

    def _make_supervisor(self, repo: Path, store: DevCoordinationStore) -> SwarmSupervisor:
        launcher = WorkerLauncher(config=LaunchConfig())
        return SwarmSupervisor(repo_root=repo, store=store, launcher=launcher)

    def test_marks_status_as_scope_violation(self, repo: Path, store: DevCoordinationStore) -> None:
        supervisor = self._make_supervisor(repo, store)
        item: dict = {"status": "dispatched", "changed_paths": ["README.md"]}
        violations = [
            {"type": "out_of_scope", "path": "README.md", "allowed_scope": ["aragora/live"]}
        ]
        supervisor._mark_scope_violation(item, violations)
        assert item["status"] == "scope_violation"
        assert item["review_status"] == "changes_requested"
        assert "scope_violation" in item
        assert "README.md" in item["dispatch_error"]

    def test_extra_reason_prepended(self, repo: Path, store: DevCoordinationStore) -> None:
        supervisor = self._make_supervisor(repo, store)
        item: dict = {"status": "dispatched", "changed_paths": []}
        violations = [{"type": "out_of_scope", "path": "foo.py", "allowed_scope": []}]
        supervisor._mark_scope_violation(item, violations, extra_reason="timed out")
        assert item["dispatch_error"].startswith("timed out;")

    def test_pid_removed(self, repo: Path, store: DevCoordinationStore) -> None:
        supervisor = self._make_supervisor(repo, store)
        item: dict = {"status": "dispatched", "pid": 12345}
        violations = [{"type": "out_of_scope", "path": "x.py", "allowed_scope": []}]
        supervisor._mark_scope_violation(item, violations)
        assert "pid" not in item

    def test_violation_persisted_to_lease_metadata(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        """Regression: scope violations must appear in store-backed status surfaces.

        After _mark_scope_violation, the lease must remain active with
        last_scope_violation metadata so that status_summary() can surface it
        to fleet/integrator views.
        """
        # Create a real lease in the coordination store
        lease = store.claim_lease(
            task_id="scope-test",
            title="Scope violation test",
            owner_agent="codex",
            owner_session_id="sess-1",
            branch="test-branch",
            worktree_path=str(repo),
            allowed_globs=["aragora/live/**"],
        )

        supervisor = self._make_supervisor(repo, store)
        item: dict = {
            "status": "dispatched",
            "lease_id": lease.lease_id,
            "changed_paths": ["README.md"],
        }
        violations = [
            {"type": "out_of_scope", "path": "README.md", "allowed_scope": ["aragora/live"]}
        ]
        supervisor._mark_scope_violation(item, violations)

        # Lease should still be active (not released)
        active = store.list_active_leases()
        active_ids = {l.lease_id for l in active}
        assert lease.lease_id in active_ids

        # Violation metadata should be in the lease
        matching = [l for l in active if l.lease_id == lease.lease_id]
        assert len(matching) == 1
        assert "last_scope_violation" in matching[0].metadata
        assert matching[0].metadata["last_scope_violation"]["violations"] == violations

        # status_summary should surface the violation
        summary = store.status_summary()
        violation_lease_ids = {v["lease_id"] for v in summary.get("scope_violations", [])}
        assert lease.lease_id in violation_lease_ids


# ---------------------------------------------------------------------------
# Tests for _apply_worker_result with scope enforcement
# ---------------------------------------------------------------------------


class TestApplyWorkerResultScopeEnforcement:
    """Test that _apply_worker_result rejects out-of-scope results."""

    def _make_supervisor(self, repo: Path, store: DevCoordinationStore) -> SwarmSupervisor:
        launcher = WorkerLauncher(config=LaunchConfig())
        return SwarmSupervisor(repo_root=repo, store=store, launcher=launcher)

    def test_successful_result_with_scope_violation_rejected(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        """A worker that exits 0 but edited wrong files must NOT be treated as completed."""
        supervisor = self._make_supervisor(repo, store)

        item = {
            "work_order_id": "issue-873",
            "file_scope": ["aragora/live"],
            "status": "dispatched",
            "lease_id": "",
            "target_agent": "codex",
        }
        result = WorkerProcess(
            work_order_id="issue-873",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=0,
            changed_paths=["README.md", "docs/LANDING_PAGE.md"],
            commit_shas=["abc123"],
            head_sha="abc123",
        )
        supervisor._apply_worker_result(item, result)

        assert item["status"] == "scope_violation"
        assert "scope_violation" in item
        assert item["status"] != "completed"

    def test_successful_result_in_scope_accepted(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        """A worker that exits 0 and edited correct files should proceed normally."""
        supervisor = self._make_supervisor(repo, store)

        item = {
            "work_order_id": "test-ok",
            "file_scope": ["aragora/live"],
            "status": "dispatched",
            "lease_id": "",
            "target_agent": "codex",
        }
        result = WorkerProcess(
            work_order_id="test-ok",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=0,
            changed_paths=["aragora/live/package.json"],
            commit_shas=["def456"],
            head_sha="def456",
        )
        supervisor._apply_worker_result(item, result)

        # Should not be scope_violation — should proceed to completion path
        assert item["status"] != "scope_violation"

    def test_no_file_scope_allows_any_changes(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        """Work orders without file_scope should accept any changed paths."""
        supervisor = self._make_supervisor(repo, store)

        item = {
            "work_order_id": "open-scope",
            "file_scope": [],
            "status": "dispatched",
            "lease_id": "",
            "target_agent": "codex",
        }
        result = WorkerProcess(
            work_order_id="open-scope",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=0,
            changed_paths=["README.md", "anything/goes.py"],
            commit_shas=["ghi789"],
            head_sha="ghi789",
        )
        supervisor._apply_worker_result(item, result)

        assert item["status"] != "scope_violation"


# ---------------------------------------------------------------------------
# Tests for worker prompt file-scope language
# ---------------------------------------------------------------------------


class TestWorkerPromptScopeLanguage:
    """Test that the worker prompt includes mandatory scope constraints."""

    def test_prompt_includes_mandatory_scope_when_file_scope_set(self) -> None:
        work_order = {
            "title": "Bump eslintrc",
            "description": "Update @eslint/eslintrc to 3.3.5",
            "file_scope": ["aragora/live/package.json", "aragora/live/package-lock.json"],
            "metadata": {},
        }
        prompt = WorkerLauncher._build_prompt(work_order)
        assert "FILE SCOPE GUIDANCE" in prompt
        assert "verify these paths exist" in prompt
        assert "aragora/live/package.json" in prompt
        assert "aragora/live/package-lock.json" in prompt

    def test_prompt_no_scope_section_when_no_file_scope(self) -> None:
        work_order = {
            "title": "Open task",
            "description": "Do something",
            "file_scope": [],
            "metadata": {},
        }
        prompt = WorkerLauncher._build_prompt(work_order)
        assert "FILE SCOPE GUIDANCE" not in prompt


# ---------------------------------------------------------------------------
# Tests for _derive_status with scope_violation
# ---------------------------------------------------------------------------


class TestDeriveStatusScopeViolation:
    """Test that scope_violation is treated as a terminal status."""

    def test_scope_violation_alone_is_terminal(self) -> None:
        work_orders = [{"status": "scope_violation"}]
        status = SwarmSupervisor._derive_status(work_orders)
        assert status == "completed"

    def test_scope_violation_mixed_with_completed(self) -> None:
        work_orders = [{"status": "scope_violation"}, {"status": "completed"}]
        status = SwarmSupervisor._derive_status(work_orders)
        assert status == "completed"

    def test_scope_violation_with_active_keeps_active(self) -> None:
        work_orders = [{"status": "scope_violation"}, {"status": "dispatched"}]
        status = SwarmSupervisor._derive_status(work_orders)
        assert status == "active"
