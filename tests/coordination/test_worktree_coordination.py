"""Tests for multi-agent worktree coordination modules.

Covers:
- WorktreeManager: creation, cleanup, stall/abandon detection
- TaskDispatcher: submit, assign, complete, dependencies, retries, stall detection
- HealthWatchdog: stall detection, auto-recovery, stats tracking
- GitReconciler: conflict detection, classification, merge history
- Full cycle: create worktree -> assign task -> complete -> merge
"""

from __future__ import annotations

import asyncio
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from aragora.coordination.worktree_manager import (
    WorktreeManager,
    WorktreeManagerConfig,
    WorktreeState,
)
from aragora.coordination.task_dispatcher import (
    TaskDispatcher,
    DispatcherConfig,
    Task,
)
from aragora.coordination.health_watchdog import (
    HealthWatchdog,
    WatchdogConfig,
    HealthEvent,
    RecoveryStats,
)
from aragora.coordination.reconciler import (
    GitReconciler,
    ReconcilerConfig,
    MergeAttempt,
    ConflictInfo,
    ConflictCategory,
)


# ---------------------------------------------------------------------------
# WorktreeManager tests
# ---------------------------------------------------------------------------

class TestWorktreeManager:
    """Tests for WorktreeManager lifecycle and health tracking."""

    def _make_manager(self, **kwargs):
        defaults = {
            "max_worktrees": 5,
            "stall_timeout_seconds": 60,
            "abandon_timeout_seconds": 300,
        }
        defaults.update(kwargs)
        config = WorktreeManagerConfig(**defaults)
        return WorktreeManager(repo_path=Path("/tmp/fake-repo"), config=config)

    @pytest.mark.asyncio
    async def test_create_worktree_success(self):
        manager = self._make_manager()

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "abc123\n"

        with patch.object(manager, "_run_git", return_value=mock_result):
            wt = await manager.create("auth-feature", track="security")

        assert wt.worktree_id
        assert "auth-feature" in wt.branch_name
        assert wt.track == "security"
        assert wt.status == "active"
        assert wt.worktree_id in manager.worktrees

    @pytest.mark.asyncio
    async def test_create_worktree_max_exceeded(self):
        manager = self._make_manager(max_worktrees=1)

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "abc123\n"

        with patch.object(manager, "_run_git", return_value=mock_result):
            await manager.create("first")

        with pytest.raises(RuntimeError, match="Max worktrees"):
            with patch.object(manager, "_run_git", return_value=mock_result):
                await manager.create("second")

    @pytest.mark.asyncio
    async def test_create_worktree_git_failure(self):
        manager = self._make_manager()

        fail_result = MagicMock()
        fail_result.returncode = 1
        fail_result.stderr = "fatal: cannot create worktree"

        with patch.object(manager, "_run_git", return_value=fail_result):
            with pytest.raises(RuntimeError, match="Failed to create worktree"):
                await manager.create("broken")

    @pytest.mark.asyncio
    async def test_destroy_worktree(self):
        manager = self._make_manager()

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "abc123\n"

        with patch.object(manager, "_run_git", return_value=mock_result):
            wt = await manager.create("to-destroy")
            result = await manager.destroy(wt.worktree_id)

        assert result is True
        assert manager.worktrees[wt.worktree_id].status == "destroyed"

    @pytest.mark.asyncio
    async def test_destroy_nonexistent(self):
        manager = self._make_manager()
        result = await manager.destroy("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_active_worktrees_property(self):
        manager = self._make_manager()

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "abc123\n"

        with patch.object(manager, "_run_git", return_value=mock_result):
            wt1 = await manager.create("one")
            wt2 = await manager.create("two")

        assert len(manager.active_worktrees) == 2

        with patch.object(manager, "_run_git", return_value=mock_result):
            await manager.destroy(wt1.worktree_id)

        assert len(manager.active_worktrees) == 1

    def test_record_activity(self):
        manager = self._make_manager()
        state = WorktreeState(
            worktree_id="test-id",
            branch_name="dev/test",
            path=Path("/tmp/wt"),
            last_activity=datetime(2020, 1, 1, tzinfo=timezone.utc),
        )
        manager._worktrees["test-id"] = state

        manager.record_activity("test-id")
        assert state.last_activity > datetime(2020, 1, 1, tzinfo=timezone.utc)

    def test_stall_detection(self):
        manager = self._make_manager(stall_timeout_seconds=60)

        old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        state = WorktreeState(
            worktree_id="stalled",
            branch_name="dev/stalled",
            path=Path("/tmp/wt-stalled"),
            last_activity=old_time,
        )
        manager._worktrees["stalled"] = state

        stalled = manager.get_stalled_worktrees()
        assert len(stalled) == 1
        assert stalled[0].worktree_id == "stalled"

    def test_stall_detection_active_not_stalled(self):
        manager = self._make_manager(stall_timeout_seconds=60)

        recent_time = datetime.now(timezone.utc) - timedelta(seconds=10)
        state = WorktreeState(
            worktree_id="active",
            branch_name="dev/active",
            path=Path("/tmp/wt-active"),
            last_activity=recent_time,
        )
        manager._worktrees["active"] = state

        stalled = manager.get_stalled_worktrees()
        assert len(stalled) == 0

    def test_abandon_detection(self):
        manager = self._make_manager(abandon_timeout_seconds=300)

        old_time = datetime.now(timezone.utc) - timedelta(seconds=600)
        state = WorktreeState(
            worktree_id="abandoned",
            branch_name="dev/abandoned",
            path=Path("/tmp/wt-abandoned"),
            last_activity=old_time,
        )
        manager._worktrees["abandoned"] = state

        abandoned = manager.get_abandoned_worktrees()
        assert len(abandoned) == 1

    def test_check_for_new_commits(self):
        manager = self._make_manager()

        state = WorktreeState(
            worktree_id="with-commits",
            branch_name="dev/commits",
            path=Path("/tmp/wt-commits"),
            last_commit_sha="old-sha",
        )
        manager._worktrees["with-commits"] = state

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "new-sha\n"

        with patch.object(manager, "_run_git", return_value=mock_result):
            with patch.object(Path, "exists", return_value=True):
                has_new = manager.check_for_new_commits("with-commits")

        assert has_new is True
        assert state.last_commit_sha == "new-sha"
        assert state.commit_count == 1

    def test_check_for_new_commits_no_change(self):
        manager = self._make_manager()

        state = WorktreeState(
            worktree_id="no-change",
            branch_name="dev/no-change",
            path=Path("/tmp/wt-no-change"),
            last_commit_sha="same-sha",
        )
        manager._worktrees["no-change"] = state

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "same-sha\n"

        with patch.object(manager, "_run_git", return_value=mock_result):
            with patch.object(Path, "exists", return_value=True):
                has_new = manager.check_for_new_commits("no-change")

        assert has_new is False

    def test_summary(self):
        manager = self._make_manager()
        manager._worktrees["a"] = WorktreeState(
            worktree_id="a", branch_name="dev/a", path=Path("/tmp/a"), status="active",
        )
        manager._worktrees["b"] = WorktreeState(
            worktree_id="b", branch_name="dev/b", path=Path("/tmp/b"), status="stalled",
        )
        manager._worktrees["c"] = WorktreeState(
            worktree_id="c", branch_name="dev/c", path=Path("/tmp/c"), status="active",
        )

        s = manager.summary()
        assert s == {"active": 2, "stalled": 1}

    @pytest.mark.asyncio
    async def test_mark_stalled(self):
        manager = self._make_manager()
        state = WorktreeState(
            worktree_id="x", branch_name="dev/x", path=Path("/tmp/x"),
        )
        manager._worktrees["x"] = state
        await manager.mark_stalled("x")
        assert state.status == "stalled"

    @pytest.mark.asyncio
    async def test_mark_abandoned(self):
        manager = self._make_manager()
        state = WorktreeState(
            worktree_id="x", branch_name="dev/x", path=Path("/tmp/x"),
        )
        manager._worktrees["x"] = state
        await manager.mark_abandoned("x")
        assert state.status == "abandoned"


# ---------------------------------------------------------------------------
# TaskDispatcher tests
# ---------------------------------------------------------------------------

class TestTaskDispatcher:
    """Tests for TaskDispatcher priority queue and dependency tracking."""

    def test_submit_task(self):
        dispatcher = TaskDispatcher()
        task = dispatcher.submit("Fix auth bug", priority=1, track="security")

        assert task.task_id
        assert task.title == "Fix auth bug"
        assert task.priority == 1
        assert task.status == "pending"

    def test_priority_ordering(self):
        dispatcher = TaskDispatcher()
        dispatcher.submit("Low priority", priority=8)
        dispatcher.submit("High priority", priority=1)
        dispatcher.submit("Medium priority", priority=5)

        next_task = dispatcher.get_next()
        assert next_task is not None
        assert next_task.title == "High priority"

    def test_track_affinity(self):
        dispatcher = TaskDispatcher()
        dispatcher.submit("Security task", priority=5, track="security")
        dispatcher.submit("SME task", priority=1, track="sme")

        # With track affinity, should get security task even though sme has higher priority
        next_task = dispatcher.get_next(track="security")
        assert next_task is not None
        assert next_task.title == "Security task"

    def test_assign_and_start(self):
        dispatcher = TaskDispatcher()
        task = dispatcher.submit("Do work")
        assert dispatcher.assign(task.task_id, "wt-1")
        assert task.status == "assigned"
        assert task.worktree_id == "wt-1"

        assert dispatcher.start(task.task_id)
        assert task.status == "running"

    def test_complete_task(self):
        dispatcher = TaskDispatcher()
        task = dispatcher.submit("Do work")
        dispatcher.assign(task.task_id, "wt-1")
        dispatcher.start(task.task_id)

        assert dispatcher.complete(task.task_id, result={"files": 3})
        assert task.status == "completed"
        assert task.result == {"files": 3}

    def test_dependencies_blocking(self):
        dispatcher = TaskDispatcher()
        task_a = dispatcher.submit("Step A", priority=1)
        task_b = dispatcher.submit("Step B", priority=1, depends_on=[task_a.task_id])

        assert task_b.blocked_by == [task_a.task_id]

        # B should not be available
        next_task = dispatcher.get_next()
        assert next_task is not None
        assert next_task.task_id == task_a.task_id

    def test_dependencies_unblocking(self):
        dispatcher = TaskDispatcher()
        task_a = dispatcher.submit("Step A", priority=1)
        task_b = dispatcher.submit("Step B", priority=1, depends_on=[task_a.task_id])

        # Complete A
        dispatcher.assign(task_a.task_id, "wt-1")
        dispatcher.complete(task_a.task_id)

        # B should now be available
        assert task_b.blocked_by == []
        next_task = dispatcher.get_next()
        assert next_task is not None
        assert next_task.task_id == task_b.task_id

    def test_fail_and_retry(self):
        dispatcher = TaskDispatcher(config=DispatcherConfig(max_retries=2))
        task = dispatcher.submit("Flaky work")
        dispatcher.assign(task.task_id, "wt-1")
        dispatcher.start(task.task_id)

        # First failure -> retry
        result = dispatcher.fail(task.task_id, error="timeout")
        assert result is False  # Not permanently failed
        assert task.status == "pending"
        assert task.retry_count == 1

        # Can reassign
        dispatcher.assign(task.task_id, "wt-2")
        dispatcher.start(task.task_id)

        # Second failure -> retry
        result = dispatcher.fail(task.task_id, error="timeout again")
        assert result is False
        assert task.retry_count == 2

        # Third failure -> permanent failure
        dispatcher.assign(task.task_id, "wt-3")
        result = dispatcher.fail(task.task_id, error="still broken")
        assert result is True
        assert task.status == "failed"

    def test_cancel_task(self):
        dispatcher = TaskDispatcher()
        task = dispatcher.submit("Cancel me")
        assert dispatcher.cancel(task.task_id)
        assert task.status == "cancelled"

    def test_reassign_task(self):
        dispatcher = TaskDispatcher()
        task = dispatcher.submit("Move me")
        dispatcher.assign(task.task_id, "wt-1")

        assert dispatcher.reassign(task.task_id, "wt-2", agent_id="agent-b")
        assert task.worktree_id == "wt-2"
        assert task.agent_id == "agent-b"

    def test_stall_detection(self):
        dispatcher = TaskDispatcher()
        task = dispatcher.submit("Stall me", stall_timeout=60)
        dispatcher.assign(task.task_id, "wt-1")
        dispatcher.start(task.task_id)

        # Backdate started_at
        task.started_at = datetime.now(timezone.utc) - timedelta(seconds=120)

        stalled = dispatcher.get_stalled_tasks()
        assert len(stalled) == 1
        assert stalled[0].task_id == task.task_id

    def test_stall_detection_not_stalled(self):
        dispatcher = TaskDispatcher()
        task = dispatcher.submit("Recent work", stall_timeout=600)
        dispatcher.assign(task.task_id, "wt-1")
        dispatcher.start(task.task_id)

        stalled = dispatcher.get_stalled_tasks()
        assert len(stalled) == 0

    def test_summary(self):
        dispatcher = TaskDispatcher()
        t1 = dispatcher.submit("A")
        t2 = dispatcher.submit("B")
        dispatcher.assign(t1.task_id, "wt-1")
        dispatcher.complete(t1.task_id)

        s = dispatcher.summary()
        assert s["completed"] == 1
        assert s["pending"] == 1

    def test_decompose(self):
        dispatcher = TaskDispatcher()
        tasks = dispatcher.decompose(
            "Improve auth",
            [
                {"id": "0", "title": "Research", "priority": 1},
                {"id": "1", "title": "Implement", "priority": 3, "depends_on": ["0"]},
                {"id": "2", "title": "Test", "priority": 5, "depends_on": ["1"]},
            ],
        )
        assert len(tasks) == 3
        assert tasks[0].title == "Research"
        assert len(tasks[1].depends_on) == 1
        assert len(tasks[2].depends_on) == 1

    def test_pending_and_running_properties(self):
        dispatcher = TaskDispatcher()
        t1 = dispatcher.submit("A")
        t2 = dispatcher.submit("B")
        dispatcher.assign(t1.task_id, "wt-1")
        dispatcher.start(t1.task_id)

        assert len(dispatcher.pending_tasks) == 1
        assert len(dispatcher.running_tasks) == 1

    def test_assign_blocked_task_fails(self):
        dispatcher = TaskDispatcher()
        task_a = dispatcher.submit("Prereq")
        task_b = dispatcher.submit("Blocked", depends_on=[task_a.task_id])

        result = dispatcher.assign(task_b.task_id, "wt-1")
        assert result is False


# ---------------------------------------------------------------------------
# HealthWatchdog tests
# ---------------------------------------------------------------------------

class TestHealthWatchdog:
    """Tests for HealthWatchdog stall detection and recovery."""

    def _make_watchdog(self, stall_timeout=60, abandon_timeout=300):
        manager = WorktreeManager(
            repo_path=Path("/tmp/fake-repo"),
            config=WorktreeManagerConfig(
                stall_timeout_seconds=stall_timeout,
                abandon_timeout_seconds=abandon_timeout,
            ),
        )
        dispatcher = TaskDispatcher()
        watchdog = HealthWatchdog(
            worktree_manager=manager,
            task_dispatcher=dispatcher,
            config=WatchdogConfig(auto_reassign_stalled=True),
        )
        return watchdog, manager, dispatcher

    @pytest.mark.asyncio
    async def test_check_all_no_worktrees(self):
        watchdog, manager, _ = self._make_watchdog()
        events = await watchdog.check_all()
        assert events == []

    @pytest.mark.asyncio
    async def test_stall_detected(self):
        watchdog, manager, dispatcher = self._make_watchdog(stall_timeout=60)

        # Add a stalled worktree
        old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        state = WorktreeState(
            worktree_id="stalled-wt",
            branch_name="dev/stalled",
            path=Path("/tmp/wt-stalled"),
            last_activity=old_time,
            assigned_task="task-1",
        )
        manager._worktrees["stalled-wt"] = state

        # Submit and assign the corresponding task
        task = dispatcher.submit("Stalled task")
        task.task_id = "task-1"
        dispatcher._tasks["task-1"] = task
        dispatcher.assign("task-1", "stalled-wt")
        dispatcher.start("task-1")

        with patch.object(manager, "check_for_new_commits", return_value=False):
            events = await watchdog.check_all()

        stall_events = [e for e in events if e.event_type == "stall_detected"]
        assert len(stall_events) == 1
        assert stall_events[0].worktree_id == "stalled-wt"
        assert watchdog.stats.stalls_detected == 1

    @pytest.mark.asyncio
    async def test_recovery_attempted(self):
        watchdog, manager, dispatcher = self._make_watchdog(stall_timeout=60)

        old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        state = WorktreeState(
            worktree_id="stalled-wt",
            branch_name="dev/stalled",
            path=Path("/tmp/wt-stalled"),
            last_activity=old_time,
            assigned_task="task-1",
        )
        manager._worktrees["stalled-wt"] = state

        task = Task(task_id="task-1", title="Stalled task", status="running")
        dispatcher._tasks["task-1"] = task

        with patch.object(manager, "check_for_new_commits", return_value=False):
            events = await watchdog.check_all()

        recovery_events = [e for e in events if "recovery" in e.event_type]
        assert len(recovery_events) == 1
        assert watchdog.stats.recoveries_attempted == 1
        assert watchdog.stats.recoveries_succeeded == 1
        assert watchdog.stats.recovery_rate == 1.0

    @pytest.mark.asyncio
    async def test_recovery_max_attempts(self):
        watchdog, manager, dispatcher = self._make_watchdog(stall_timeout=60)
        watchdog.config.max_recovery_attempts = 1

        old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        state = WorktreeState(
            worktree_id="stalled-wt",
            branch_name="dev/stalled",
            path=Path("/tmp/wt-stalled"),
            last_activity=old_time,
        )
        manager._worktrees["stalled-wt"] = state

        # First check - recovers
        with patch.object(manager, "check_for_new_commits", return_value=False):
            await watchdog.check_all()

        # Reset worktree to stalled state for second check
        state.status = "active"
        state.last_activity = old_time

        # Second check - max attempts reached
        with patch.object(manager, "check_for_new_commits", return_value=False):
            events = await watchdog.check_all()

        failed_events = [e for e in events if e.event_type == "recovery_failed"]
        assert len(failed_events) == 1
        assert watchdog.stats.recoveries_failed == 1

    @pytest.mark.asyncio
    async def test_abandoned_cleanup(self):
        watchdog, manager, _ = self._make_watchdog(abandon_timeout=300)

        old_time = datetime.now(timezone.utc) - timedelta(seconds=600)
        state = WorktreeState(
            worktree_id="old-wt",
            branch_name="dev/old",
            path=Path("/tmp/wt-old"),
            last_activity=old_time,
        )
        manager._worktrees["old-wt"] = state

        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch.object(manager, "check_for_new_commits", return_value=False):
            with patch.object(manager, "_run_git", return_value=mock_result):
                events = await watchdog.check_all()

        abandoned_events = [e for e in events if e.event_type == "abandoned"]
        assert len(abandoned_events) >= 1
        assert watchdog.stats.abandoned_cleaned >= 1

    @pytest.mark.asyncio
    async def test_on_stall_callback(self):
        watchdog, manager, _ = self._make_watchdog(stall_timeout=60)
        callback_events = []
        watchdog.on_stall = lambda e: callback_events.append(e)

        old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        state = WorktreeState(
            worktree_id="cb-wt",
            branch_name="dev/cb",
            path=Path("/tmp/wt-cb"),
            last_activity=old_time,
        )
        manager._worktrees["cb-wt"] = state

        with patch.object(manager, "check_for_new_commits", return_value=False):
            await watchdog.check_all()

        assert len(callback_events) == 1
        assert callback_events[0].event_type == "stall_detected"

    def test_recovery_stats_default(self):
        stats = RecoveryStats()
        assert stats.recovery_rate == 0.0

    def test_recovery_stats_rate(self):
        stats = RecoveryStats(recoveries_attempted=10, recoveries_succeeded=7)
        assert stats.recovery_rate == 0.7


# ---------------------------------------------------------------------------
# GitReconciler tests
# ---------------------------------------------------------------------------

class TestGitReconciler:
    """Tests for GitReconciler conflict detection and merge logic."""

    def _make_reconciler(self, **kwargs):
        defaults = {
            "pre_merge_tests": False,
            "post_merge_tests": False,
        }
        defaults.update(kwargs)
        config = ReconcilerConfig(**defaults)
        return GitReconciler(repo_path=Path("/tmp/fake-repo"), config=config)

    def test_conflict_classification_test_file(self):
        reconciler = self._make_reconciler()
        assert reconciler._classify_conflict("tests/test_auth.py") == ConflictCategory.TEST

    def test_conflict_classification_config(self):
        reconciler = self._make_reconciler()
        assert reconciler._classify_conflict("pyproject.toml") == ConflictCategory.CONFIG

    def test_conflict_classification_init(self):
        reconciler = self._make_reconciler()
        assert reconciler._classify_conflict("aragora/__init__.py") == ConflictCategory.IMPORT_ORDER

    def test_conflict_classification_unknown(self):
        reconciler = self._make_reconciler()
        assert reconciler._classify_conflict("aragora/server/handler.py") == ConflictCategory.UNKNOWN

    def test_detect_conflicts_no_conflicts(self):
        reconciler = self._make_reconciler()

        base_result = MagicMock()
        base_result.returncode = 0
        base_result.stdout = "abc123\n"

        tree_result = MagicMock()
        tree_result.returncode = 0
        tree_result.stdout = "clean merge"

        with patch.object(reconciler, "_run_git", side_effect=[base_result, tree_result]):
            conflicts = reconciler.detect_conflicts("dev/feature", "main")

        assert len(conflicts) == 0

    def test_detect_conflicts_with_conflicts(self):
        reconciler = self._make_reconciler()

        base_result = MagicMock()
        base_result.returncode = 0
        base_result.stdout = "abc123\n"

        tree_result = MagicMock()
        tree_result.returncode = 1
        tree_result.stdout = "CONFLICT (content): Merge conflict in aragora/auth/handler.py\n"

        with patch.object(reconciler, "_run_git", side_effect=[base_result, tree_result]):
            conflicts = reconciler.detect_conflicts("dev/feature", "main")

        assert len(conflicts) >= 1

    @pytest.mark.asyncio
    async def test_safe_merge_success(self):
        reconciler = self._make_reconciler()

        mock_results = {
            "calls": []
        }

        def mock_git(*args, cwd=None, check=True):
            mock_results["calls"].append(args)
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""

            if "merge-base" in args:
                result.stdout = "abc123\n"
            elif "merge-tree" in args:
                result.stdout = "clean\n"
            elif "rev-parse" in args:
                result.stdout = "newsha123\n"
            elif "merge" in args and "--no-ff" in args:
                result.returncode = 0
            return result

        with patch.object(reconciler, "_run_git", side_effect=mock_git):
            attempt = await reconciler.safe_merge("dev/feature", "main")

        assert attempt.success is True
        assert attempt.commit_sha == "newsha123"

    @pytest.mark.asyncio
    async def test_safe_merge_with_conflicts(self):
        reconciler = self._make_reconciler()

        def mock_git(*args, cwd=None, check=True):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""

            if "merge-base" in args:
                result.stdout = "abc123\n"
            elif "merge-tree" in args:
                result.stdout = "CONFLICT (content): Merge conflict in aragora/server/handler.py\n"
                result.returncode = 1
            return result

        with patch.object(reconciler, "_run_git", side_effect=mock_git):
            attempt = await reconciler.safe_merge("dev/feature", "main")

        assert attempt.success is False
        assert "conflict" in attempt.error.lower()

    @pytest.mark.asyncio
    async def test_safe_merge_rollback_on_test_failure(self):
        reconciler = self._make_reconciler(
            pre_merge_tests=False,
            post_merge_tests=True,
            rollback_on_test_failure=True,
        )

        call_count = {"n": 0}

        def mock_git(*args, cwd=None, check=True):
            call_count["n"] += 1
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""

            if "merge-base" in args:
                result.stdout = "abc123\n"
            elif "merge-tree" in args:
                result.stdout = "clean\n"
            elif "rev-parse" in args:
                result.stdout = "mergesha\n"
            elif "revert" in args:
                result.returncode = 0
            return result

        with patch.object(reconciler, "_run_git", side_effect=mock_git):
            with patch.object(reconciler, "_run_tests", return_value=False):
                attempt = await reconciler.safe_merge("dev/broken", "main")

        assert attempt.success is False
        assert attempt.rolled_back is True
        assert attempt.tests_passed is False

    @pytest.mark.asyncio
    async def test_safe_merge_pre_merge_tests_fail(self):
        reconciler = self._make_reconciler(pre_merge_tests=True)

        def mock_git(*args, cwd=None, check=True):
            result = MagicMock()
            result.returncode = 0
            result.stdout = ""
            result.stderr = ""

            if "merge-base" in args:
                result.stdout = "abc123\n"
            elif "merge-tree" in args:
                result.stdout = "clean\n"
            return result

        with patch.object(reconciler, "_run_git", side_effect=mock_git):
            with patch.object(reconciler, "_run_tests", return_value=False):
                attempt = await reconciler.safe_merge("dev/failing", "main")

        assert attempt.success is False
        assert attempt.error == "Pre-merge tests failed"

    def test_merge_history(self):
        reconciler = self._make_reconciler()
        attempt = MergeAttempt(
            source_branch="dev/a",
            target_branch="main",
            success=True,
            commit_sha="abc",
        )
        reconciler._merge_history.append(attempt)

        assert len(reconciler.merge_history) == 1
        assert reconciler.summary() == {
            "total": 1,
            "succeeded": 1,
            "failed": 0,
            "rolled_back": 0,
        }

    def test_summary_mixed(self):
        reconciler = self._make_reconciler()
        reconciler._merge_history = [
            MergeAttempt(source_branch="a", target_branch="main", success=True),
            MergeAttempt(source_branch="b", target_branch="main", success=False),
            MergeAttempt(source_branch="c", target_branch="main", success=False, rolled_back=True),
        ]
        s = reconciler.summary()
        assert s == {"total": 3, "succeeded": 1, "failed": 2, "rolled_back": 1}

    def test_get_commits_ahead(self):
        reconciler = self._make_reconciler()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "5\n"

        with patch.object(reconciler, "_run_git", return_value=mock_result):
            count = reconciler.get_commits_ahead("dev/feature", "main")

        assert count == 5

    def test_get_changed_files(self):
        reconciler = self._make_reconciler()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "aragora/auth.py\naragora/server.py\n"

        with patch.object(reconciler, "_run_git", return_value=mock_result):
            files = reconciler.get_changed_files("dev/feature", "main")

        assert files == ["aragora/auth.py", "aragora/server.py"]

    def test_conflict_info_auto_resolvable(self):
        info = ConflictInfo(
            file_path="aragora/__init__.py",
            category=ConflictCategory.IMPORT_ORDER,
            auto_resolvable=True,
        )
        assert info.auto_resolvable is True

    def test_conflict_category_values(self):
        assert ConflictCategory.IMPORT_ORDER.value == "import_order"
        assert ConflictCategory.SEMANTIC.value == "semantic"


# ---------------------------------------------------------------------------
# Integration: full cycle test
# ---------------------------------------------------------------------------

class TestFullCycle:
    """End-to-end test: create worktree -> assign task -> complete -> merge."""

    @pytest.mark.asyncio
    async def test_full_cycle(self):
        """Simulate the full worktree coordination lifecycle."""
        # 1. Setup
        manager = WorktreeManager(
            repo_path=Path("/tmp/fake-repo"),
            config=WorktreeManagerConfig(max_worktrees=5),
        )
        dispatcher = TaskDispatcher()
        watchdog = HealthWatchdog(manager, dispatcher)
        reconciler = GitReconciler(
            repo_path=Path("/tmp/fake-repo"),
            config=ReconcilerConfig(pre_merge_tests=False, post_merge_tests=False),
        )

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "sha123\n"

        # 2. Create worktree
        with patch.object(manager, "_run_git", return_value=mock_result):
            wt = await manager.create("auth-feature", track="security")

        assert wt.status == "active"

        # 3. Submit and assign task
        task = dispatcher.submit("Fix auth bug", track="security")
        wt_state = manager.worktrees[wt.worktree_id]
        wt_state.assigned_task = task.task_id

        assert dispatcher.assign(task.task_id, wt.worktree_id)
        assert dispatcher.start(task.task_id)
        assert task.status == "running"

        # 4. Simulate work completion
        assert dispatcher.complete(task.task_id, result={"files_changed": 3})
        assert task.status == "completed"

        # 5. Run health check (should find nothing stalled)
        with patch.object(manager, "check_for_new_commits", return_value=False):
            events = await watchdog.check_all()
        # No stall events since we just completed
        stall_events = [e for e in events if e.event_type == "stall_detected"]
        # The worktree still exists but task is done - no stall expected

        # 6. Merge via reconciler
        def mock_git(*args, cwd=None, check=True):
            r = MagicMock()
            r.returncode = 0
            r.stdout = ""
            r.stderr = ""
            if "merge-base" in args:
                r.stdout = "base123\n"
            elif "merge-tree" in args:
                r.stdout = "clean\n"
            elif "rev-parse" in args:
                r.stdout = "mergesha\n"
            return r

        with patch.object(reconciler, "_run_git", side_effect=mock_git):
            merge = await reconciler.safe_merge(wt.branch_name, "main")

        assert merge.success is True
        assert merge.commit_sha == "mergesha"

        # 7. Cleanup worktree
        with patch.object(manager, "_run_git", return_value=mock_result):
            destroyed = await manager.destroy(wt.worktree_id)

        assert destroyed is True

        # Verify final state
        assert dispatcher.summary()["completed"] == 1
        assert reconciler.summary()["succeeded"] == 1
