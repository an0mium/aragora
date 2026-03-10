"""Regression tests for worker no-deliverable hardening.

Reproduces the exact failure shape from dogfood runs #1 and #2: workers
dispatch and exit cleanly but produce no concrete deliverable (no pushed
branch, no PR, no committed artifact).

Root cause: worker prompt did not instruct push; auto_commit committed
locally but never pushed; _extract_deliverable gate requires branch +
commit_shas (which local commits satisfy) but higher-level orchestrators
need the pushed artifact for review/PR creation.

Fix: (1) worker prompt now instructs git push, (2) _auto_commit now
pushes after committing, (3) WorkerOutcome enum provides structured
classification.
"""

from __future__ import annotations

import asyncio
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.swarm.boss_loop import (
    _classify_terminal_run_outcome,
    _extract_deliverable,
    _extract_worker_outcome,
)
from aragora.swarm.worker_launcher import (
    LaunchConfig,
    WorkerLauncher,
    WorkerProcess,
)

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Prompt instructs push
# ---------------------------------------------------------------------------


class TestPromptIncludesPushInstruction:
    """Verify the worker prompt now tells the worker to push."""

    def test_prompt_contains_git_push(self) -> None:
        wo: dict[str, Any] = {
            "title": "Update docs",
            "description": "Fix typo in README",
        }
        prompt = WorkerLauncher._build_prompt(wo)
        assert "git push" in prompt.lower()

    def test_prompt_push_is_after_commit(self) -> None:
        wo: dict[str, Any] = {"title": "Task", "description": "Do work"}
        prompt = WorkerLauncher._build_prompt(wo)
        commit_pos = prompt.lower().find("git commit")
        push_pos = prompt.lower().find("git push")
        assert commit_pos >= 0, "prompt must mention git commit"
        assert push_pos >= 0, "prompt must mention git push"
        assert push_pos > commit_pos, "push instruction must come after commit"

    def test_prompt_push_failure_is_acceptable(self) -> None:
        """Push failure should be explicitly acceptable (the harness will retry)."""
        wo: dict[str, Any] = {"title": "Task"}
        prompt = WorkerLauncher._build_prompt(wo)
        assert "acceptable" in prompt.lower() or "harness" in prompt.lower()


# ---------------------------------------------------------------------------
# Auto-push after auto-commit
# ---------------------------------------------------------------------------


class TestAutoCommitPush:
    """Verify _auto_commit now pushes after committing."""

    @pytest.mark.asyncio
    async def test_auto_push_called_after_successful_commit(self, tmp_path: Path) -> None:
        repo = tmp_path / "repo"
        repo.mkdir()
        _run(repo, "git", "init", "-b", "main")
        _run(repo, "git", "config", "user.email", "test@example.com")
        _run(repo, "git", "config", "user.name", "Test User")
        (repo / "README.md").write_text("hello\n", encoding="utf-8")
        _run(repo, "git", "add", ".")
        _run(repo, "git", "commit", "-m", "initial")

        # Make a change for auto_commit to pick up
        (repo / "file.py").write_text("# new file\n", encoding="utf-8")

        worker = WorkerProcess(
            work_order_id="wo-push-test",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=0,
        )

        with patch.object(WorkerLauncher, "_auto_push", new_callable=AsyncMock) as mock_push:
            await WorkerLauncher._auto_commit(worker)
            mock_push.assert_called_once_with(worker)

    @pytest.mark.asyncio
    async def test_auto_push_not_called_on_commit_failure(self, tmp_path: Path) -> None:
        """If git commit fails, push should not be attempted."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _run(repo, "git", "init", "-b", "main")
        _run(repo, "git", "config", "user.email", "test@example.com")
        _run(repo, "git", "config", "user.name", "Test User")
        (repo / "README.md").write_text("hello\n", encoding="utf-8")
        _run(repo, "git", "add", ".")
        _run(repo, "git", "commit", "-m", "initial")

        # No changes — commit will find nothing staged
        worker = WorkerProcess(
            work_order_id="wo-no-changes",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=0,
        )

        with patch.object(WorkerLauncher, "_auto_push", new_callable=AsyncMock) as mock_push:
            await WorkerLauncher._auto_commit(worker)
            mock_push.assert_not_called()

    @pytest.mark.asyncio
    async def test_auto_push_best_effort(self, tmp_path: Path) -> None:
        """Push failure should not crash the auto-commit flow."""
        repo = tmp_path / "repo"
        repo.mkdir()
        _run(repo, "git", "init", "-b", "main")
        _run(repo, "git", "config", "user.email", "test@example.com")
        _run(repo, "git", "config", "user.name", "Test User")
        (repo / "README.md").write_text("hello\n", encoding="utf-8")
        _run(repo, "git", "add", ".")
        _run(repo, "git", "commit", "-m", "initial")

        (repo / "file.py").write_text("# new\n", encoding="utf-8")

        worker = WorkerProcess(
            work_order_id="wo-push-fail",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=0,
        )

        # Push will fail (no remote) but should not raise
        await WorkerLauncher._auto_commit(worker)
        # Verify the commit was still made
        result = _run(repo, "git", "log", "--oneline", "-1")
        assert "wo-push-fail" in result.stdout


# ---------------------------------------------------------------------------
# Exact dogfood failure shape reproduction
# ---------------------------------------------------------------------------


class TestDogfoodNoDeliverableShape:
    """Reproduce the exact failure shape from dogfood runs #1 and #2.

    Shape: worker dispatches, exits 0, produces no commits, no changed
    paths, branch is set — classified as clean_exit_no_deliverable.
    """

    def test_clean_exit_zero_commits_zero_paths_is_no_deliverable(self) -> None:
        """Exact shape: completed work order with branch but no commits."""
        run_dict: dict[str, Any] = {
            "status": "completed",
            "work_orders": [
                {
                    "work_order_id": "wo-dogfood",
                    "status": "completed",
                    "branch": "codex/swarm-abcd1234-work-1234",
                    "commit_shas": [],
                    "changed_paths": [],
                    "pr_url": "",
                    "worktree_path": "/tmp/worktree",
                },
            ],
        }
        assert _extract_deliverable(run_dict) is None
        assert _classify_terminal_run_outcome(run_dict) == "clean_exit_no_deliverable"

    def test_clean_exit_with_commits_is_deliverable(self) -> None:
        """After fix: worker commits + pushes → deliverable detected."""
        run_dict: dict[str, Any] = {
            "status": "completed",
            "work_orders": [
                {
                    "work_order_id": "wo-fixed",
                    "status": "completed",
                    "branch": "codex/swarm-abcd1234-work-1234",
                    "commit_shas": ["abc123"],
                    "changed_paths": ["docs/README.md"],
                    "pr_url": "",
                },
            ],
        }
        deliverable = _extract_deliverable(run_dict)
        assert deliverable is not None
        assert deliverable["type"] == "branch"
        assert deliverable["commit_shas"] == ["abc123"]
        assert _classify_terminal_run_outcome(run_dict) == "deliverable_created"


# ---------------------------------------------------------------------------
# WorkerOutcome classification
# ---------------------------------------------------------------------------


class TestWorkerOutcomeExtraction:
    def test_extract_worker_outcome_from_run(self) -> None:
        run_dict: dict[str, Any] = {
            "work_orders": [
                {"worker_outcome": "clean_exit_no_effect", "status": "completed"},
            ],
        }
        assert _extract_worker_outcome(run_dict) == "clean_exit_no_effect"

    def test_extract_worker_outcome_none_when_missing(self) -> None:
        run_dict: dict[str, Any] = {"work_orders": [{"status": "completed"}]}
        assert _extract_worker_outcome(run_dict) is None

    def test_extract_worker_outcome_skips_non_dict(self) -> None:
        run_dict: dict[str, Any] = {"work_orders": ["not_a_dict", None]}
        assert _extract_worker_outcome(run_dict) is None

    def test_extract_worker_outcome_first_wins(self) -> None:
        run_dict: dict[str, Any] = {
            "work_orders": [
                {"worker_outcome": "completed"},
                {"worker_outcome": "crash"},
            ],
        }
        assert _extract_worker_outcome(run_dict) == "completed"


# ---------------------------------------------------------------------------
# WorkerOutcome in supervisor _apply_worker_result
# ---------------------------------------------------------------------------


class TestSupervisorWorkerOutcome:
    """Verify _apply_worker_result sets worker_outcome."""

    @pytest.mark.asyncio
    async def test_completed_with_commits_sets_completed_outcome(self, tmp_path: Path) -> None:
        from aragora.nomic.dev_coordination import DevCoordinationStore
        from aragora.swarm.supervisor import SwarmSupervisor, WorkerOutcome

        repo = _init_repo(tmp_path)
        store = DevCoordinationStore(repo_root=repo)
        supervisor = _make_supervisor(repo, store)
        head = _head(repo)

        (repo / "README.md").write_text("updated\n", encoding="utf-8")
        _run(repo, "git", "add", "README.md")
        _run(repo, "git", "commit", "-m", "worker change")
        new_head = _head(repo)

        item = _make_dispatched_item(repo, initial_head=head)
        run_id = _create_run(store, item)

        result = WorkerProcess(
            work_order_id="wo-outcome",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=0,
            changed_paths=["README.md"],
            commit_shas=[new_head],
            head_sha=new_head,
        )

        with patch.object(
            supervisor.launcher,
            "collect_finished",
            new_callable=AsyncMock,
            return_value=[result],
        ):
            await supervisor.collect_finished_results(run_id)

        record = store.get_supervisor_run(run_id)
        wo = record["work_orders"][0]
        assert wo["worker_outcome"] == WorkerOutcome.COMPLETED.value

    @pytest.mark.asyncio
    async def test_clean_exit_no_changes_sets_clean_exit_no_effect(self, tmp_path: Path) -> None:
        from aragora.nomic.dev_coordination import DevCoordinationStore
        from aragora.swarm.supervisor import SwarmSupervisor, WorkerOutcome

        repo = _init_repo(tmp_path)
        store = DevCoordinationStore(repo_root=repo)
        supervisor = _make_supervisor(repo, store)
        head = _head(repo)

        item = _make_dispatched_item(repo, initial_head=head)
        run_id = _create_run(store, item)

        result = WorkerProcess(
            work_order_id="wo-outcome",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=0,
            changed_paths=[],
            commit_shas=[],
            head_sha=head,
        )

        with patch.object(
            supervisor.launcher,
            "collect_finished",
            new_callable=AsyncMock,
            return_value=[result],
        ):
            await supervisor.collect_finished_results(run_id)

        record = store.get_supervisor_run(run_id)
        wo = record["work_orders"][0]
        assert wo["worker_outcome"] == WorkerOutcome.CLEAN_EXIT_NO_EFFECT.value
        assert wo["status"] == "needs_human", (
            "clean_exit_no_effect must fail closed to needs_human, not completed"
        )

    @pytest.mark.asyncio
    async def test_nonzero_exit_no_commits_sets_crash(self, tmp_path: Path) -> None:
        from aragora.nomic.dev_coordination import DevCoordinationStore
        from aragora.swarm.supervisor import SwarmSupervisor, WorkerOutcome

        repo = _init_repo(tmp_path)
        store = DevCoordinationStore(repo_root=repo)
        supervisor = _make_supervisor(repo, store)
        head = _head(repo)

        item = _make_dispatched_item(repo, initial_head=head)
        run_id = _create_run(store, item)

        result = WorkerProcess(
            work_order_id="wo-outcome",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=1,
            changed_paths=[],
            commit_shas=[],
            head_sha=head,
        )

        with patch.object(
            supervisor.launcher,
            "collect_finished",
            new_callable=AsyncMock,
            return_value=[result],
        ):
            await supervisor.collect_finished_results(run_id)

        record = store.get_supervisor_run(run_id)
        wo = record["work_orders"][0]
        assert wo["worker_outcome"] == WorkerOutcome.CRASH.value
        assert wo["status"] == "failed"


# ---------------------------------------------------------------------------
# BossIterationStatus worker_outcome field
# ---------------------------------------------------------------------------


class TestBossIterationStatusWorkerOutcome:
    def test_includes_worker_outcome_when_set(self) -> None:
        from aragora.swarm.boss_loop import BossIterationStatus

        status = BossIterationStatus(
            iteration=1,
            run_id="test",
            timestamp="2026-01-01T00:00:00Z",
            runner_freshness={},
            selected_issue=None,
            worker_status="needs_human",
            stop_reason="needs_human",
            needs_human_reasons=["test"],
            next_actions=[],
            worker_outcome="clean_exit_no_effect",
        )
        d = status.to_dict()
        assert d["worker_outcome"] == "clean_exit_no_effect"

    def test_omits_worker_outcome_when_none(self) -> None:
        from aragora.swarm.boss_loop import BossIterationStatus

        status = BossIterationStatus(
            iteration=1,
            run_id="test",
            timestamp="2026-01-01T00:00:00Z",
            runner_freshness={},
            selected_issue=None,
            worker_status="idle",
            stop_reason=None,
            needs_human_reasons=[],
            next_actions=[],
        )
        d = status.to_dict()
        assert "worker_outcome" not in d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), cwd=cwd, text=True, capture_output=True, check=True)


def _head(repo: Path) -> str:
    return _run(repo, "git", "rev-parse", "HEAD").stdout.strip()


def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "git", "init", "-b", "main")
    _run(repo, "git", "config", "user.email", "test@example.com")
    _run(repo, "git", "config", "user.name", "Test User")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _run(repo, "git", "add", ".")
    _run(repo, "git", "commit", "-m", "initial")
    _run(repo, "git", "remote", "add", "origin", str(repo))
    _run(repo, "git", "update-ref", "refs/remotes/origin/main", "HEAD")
    return repo


def _make_supervisor(repo: Path, store: Any) -> Any:
    from aragora.swarm.supervisor import SwarmSupervisor

    config = LaunchConfig(no_progress_timeout_seconds=1.0)
    launcher = WorkerLauncher(config=config)
    return SwarmSupervisor(repo_root=repo, store=store, launcher=launcher)


def _make_dispatched_item(
    repo: Path,
    *,
    work_order_id: str = "wo-outcome",
    initial_head: str | None = None,
) -> dict[str, Any]:
    head = initial_head or _head(repo)
    dispatched_at = datetime(2020, 1, 1, tzinfo=UTC).isoformat()
    return {
        "work_order_id": work_order_id,
        "status": "dispatched",
        "target_agent": "codex",
        "worktree_path": str(repo),
        "branch": "main",
        "initial_head": head,
        "lease_id": "",
        "file_scope": [],
        "pid": 99999,
        "dispatched_at": dispatched_at,
        "last_progress_at": dispatched_at,
        "changed_paths": [],
    }


def _create_run(store: Any, item: dict[str, Any]) -> str:
    record = store.create_supervisor_run(
        goal="test outcome",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "test outcome"},
        work_orders=[item],
    )
    return record["run_id"]


# ---------------------------------------------------------------------------
# Campaign dispatch disables managed session script
# ---------------------------------------------------------------------------


class TestCampaignDisablesManagedSessionScript:
    """Campaign executor must pass use_managed_session_script=False to avoid
    the nested-worktree problem where codex_session.sh creates a second
    worktree that the harness never inspects."""

    @pytest.mark.asyncio
    async def test_campaign_dispatch_passes_use_managed_session_script_false(
        self, tmp_path: Path
    ) -> None:
        """Verify dispatch_bounded_spec receives use_managed_session_script=False
        when called from the campaign executor."""
        from unittest.mock import ANY

        from aragora.swarm.campaign import (
            CampaignExecutor,
            CampaignManifest,
            CampaignProject,
            save_campaign_manifest,
        )
        from aragora.swarm.spec import SwarmSpec

        manifest_dir = tmp_path / ".aragora"
        manifest_dir.mkdir()
        manifest_path = manifest_dir / "campaign_manifest.yaml"

        spec = SwarmSpec(
            raw_goal="test",
            refined_goal="test",
            acceptance_criteria=["test passes"],
            file_scope_hints=["test.md"],
        )
        project = CampaignProject(
            project_id="proj-001",
            title="test project",
            spec=spec,
            acceptance_criteria=["test passes"],
            file_scope_hints=["test.md"],
            estimated_cost_usd=1.0,
        )
        manifest = CampaignManifest(
            campaign_id="test-campaign",
            created_at="2026-01-01T00:00:00Z",
            source_kind="prebuilt",
            source_ref="test",
            worker_model="codex",
            review_model="claude",
            max_retries_per_project=0,
            projects=[project],
        )
        save_campaign_manifest(manifest_path, manifest)

        captured_kwargs: dict[str, Any] = {}

        async def mock_dispatch(spec: Any, **kwargs: Any) -> dict[str, Any]:
            captured_kwargs.update(kwargs)
            return {
                "status": "needs_human",
                "outcome": "clean_exit_no_deliverable",
                "run": {"status": "completed", "work_orders": []},
                "run_id": "test-run-id",
            }

        executor = CampaignExecutor(
            manifest_path=manifest_path,
            repo_root=tmp_path,
        )
        with patch(
            "aragora.swarm.campaign.dispatch_bounded_spec",
            side_effect=mock_dispatch,
        ):
            await executor.execute_once()

        assert captured_kwargs.get("use_managed_session_script") is False, (
            "Campaign executor must pass use_managed_session_script=False "
            "to prevent codex_session.sh from creating a nested worktree"
        )
