"""
Tests for GUPP Hook Persistence (aragora/fabric/hooks.py).

Tests the git worktree-backed per-agent work queue including:
- Hook creation and lifecycle
- GUPP patrol (pending/stale hook detection)
- Resume, re-assign, complete, fail, abandon flows
- JSONL persistence and recovery
- Cleanup of completed hooks
- Stats reporting
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from aragora.fabric.hooks import (
    Hook,
    HookManager,
    HookManagerConfig,
    HookStatus,
    HOOK_STATE_FILE,
    HOOK_RESULT_FILE,
)


# =============================================================================
# Hook Dataclass Tests
# =============================================================================


class TestHookDataclass:
    """Test Hook dataclass serialization."""

    def test_hook_creation(self):
        hook = Hook(
            hook_id="hk-abc12",
            agent_id="agent-1",
            workspace_id="ws-1",
        )
        assert hook.hook_id == "hk-abc12"
        assert hook.agent_id == "agent-1"
        assert hook.status == HookStatus.PENDING
        assert hook.retry_count == 0

    def test_hook_to_dict(self):
        hook = Hook(
            hook_id="hk-abc12",
            agent_id="agent-1",
            workspace_id="ws-1",
            status=HookStatus.RUNNING,
        )
        d = hook.to_dict()
        assert d["hook_id"] == "hk-abc12"
        assert d["status"] == "running"
        assert isinstance(d["created_at"], float)

    def test_hook_from_dict(self):
        data = {
            "hook_id": "hk-abc12",
            "agent_id": "agent-1",
            "workspace_id": "ws-1",
            "status": "assigned",
            "work_item": {"task": "test"},
            "created_at": 1000.0,
            "updated_at": 1000.0,
            "assigned_at": 1000.0,
            "completed_at": None,
            "result": None,
            "error": None,
            "git_worktree_path": None,
            "retry_count": 0,
            "max_retries": 3,
        }
        hook = Hook.from_dict(data)
        assert hook.hook_id == "hk-abc12"
        assert hook.status == HookStatus.ASSIGNED
        assert hook.work_item == {"task": "test"}

    def test_hook_roundtrip(self):
        hook = Hook(
            hook_id="hk-test",
            agent_id="agent-1",
            workspace_id="ws-1",
            status=HookStatus.RUNNING,
            work_item={"action": "build"},
        )
        d = hook.to_dict()
        restored = Hook.from_dict(d)
        assert restored.hook_id == hook.hook_id
        assert restored.status == hook.status
        assert restored.work_item == hook.work_item


class TestHookStatus:
    """Test HookStatus enum."""

    def test_all_statuses_exist(self):
        assert HookStatus.PENDING.value == "pending"
        assert HookStatus.ASSIGNED.value == "assigned"
        assert HookStatus.RUNNING.value == "running"
        assert HookStatus.COMPLETED.value == "completed"
        assert HookStatus.FAILED.value == "failed"
        assert HookStatus.ABANDONED.value == "abandoned"


# =============================================================================
# HookManager Tests
# =============================================================================


class TestHookManagerInit:
    """Test HookManager initialization."""

    def test_default_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = HookManagerConfig(workspace_root=tmpdir, use_git_worktrees=False)
            mgr = HookManager(config=config)
            assert mgr._hooks == {}

    def test_creates_hooks_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = HookManagerConfig(workspace_root=tmpdir, use_git_worktrees=False)
            mgr = HookManager(config=config)
            assert (Path(tmpdir) / ".aragora_hooks").exists()


class TestHookCreation:
    """Test hook creation."""

    @pytest.fixture
    def hook_mgr(self, tmp_path):
        config = HookManagerConfig(workspace_root=str(tmp_path), use_git_worktrees=False)
        return HookManager(config=config)

    @pytest.mark.asyncio
    async def test_create_hook(self, hook_mgr):
        hook = await hook_mgr.create_hook(
            agent_id="agent-1",
            work_item={"task": "build"},
        )
        assert hook.agent_id == "agent-1"
        assert hook.status == HookStatus.ASSIGNED
        assert hook.work_item == {"task": "build"}
        assert hook.hook_id.startswith("hk-")

    @pytest.mark.asyncio
    async def test_create_hook_custom_id(self, hook_mgr):
        hook = await hook_mgr.create_hook(
            agent_id="agent-1",
            work_item={},
            hook_id="hk-custom",
        )
        assert hook.hook_id == "hk-custom"

    @pytest.mark.asyncio
    async def test_create_hook_persists_to_disk(self, hook_mgr):
        hook = await hook_mgr.create_hook(
            agent_id="agent-1",
            work_item={"key": "value"},
            hook_id="hk-persist",
        )
        state_file = hook_mgr._hook_dir("hk-persist") / HOOK_STATE_FILE
        assert state_file.exists()
        content = state_file.read_text().strip()
        data = json.loads(content)
        assert data["hook_id"] == "hk-persist"
        assert data["status"] == "assigned"


class TestHookRetrieval:
    """Test hook retrieval."""

    @pytest.fixture
    def hook_mgr(self, tmp_path):
        config = HookManagerConfig(workspace_root=str(tmp_path), use_git_worktrees=False)
        return HookManager(config=config)

    @pytest.mark.asyncio
    async def test_get_hook(self, hook_mgr):
        hook = await hook_mgr.create_hook("agent-1", {"task": "a"}, hook_id="hk-get")
        retrieved = await hook_mgr.get_hook("hk-get")
        assert retrieved is not None
        assert retrieved.hook_id == "hk-get"

    @pytest.mark.asyncio
    async def test_get_nonexistent_hook(self, hook_mgr):
        result = await hook_mgr.get_hook("hk-missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_hooks_by_agent(self, hook_mgr):
        await hook_mgr.create_hook("agent-1", {}, hook_id="hk-a1")
        await hook_mgr.create_hook("agent-2", {}, hook_id="hk-a2")
        await hook_mgr.create_hook("agent-1", {}, hook_id="hk-a3")

        hooks = await hook_mgr.list_hooks(agent_id="agent-1")
        assert len(hooks) == 2

    @pytest.mark.asyncio
    async def test_list_hooks_by_status(self, hook_mgr):
        h1 = await hook_mgr.create_hook("agent-1", {}, hook_id="hk-s1")
        await hook_mgr.create_hook("agent-2", {}, hook_id="hk-s2")
        await hook_mgr.complete_hook("hk-s1")

        hooks = await hook_mgr.list_hooks(status=HookStatus.COMPLETED)
        assert len(hooks) == 1
        assert hooks[0].hook_id == "hk-s1"


# =============================================================================
# GUPP Patrol Tests
# =============================================================================


class TestGUPPPatrol:
    """Test GUPP patrol functionality."""

    @pytest.fixture
    def hook_mgr(self, tmp_path):
        config = HookManagerConfig(workspace_root=str(tmp_path), use_git_worktrees=False)
        return HookManager(config=config)

    @pytest.mark.asyncio
    async def test_check_pending_hooks_finds_assigned(self, hook_mgr):
        await hook_mgr.create_hook("agent-1", {"task": "build"}, hook_id="hk-p1")
        pending = await hook_mgr.check_pending_hooks()
        assert len(pending) == 1
        assert pending[0].hook_id == "hk-p1"

    @pytest.mark.asyncio
    async def test_check_pending_hooks_finds_running(self, hook_mgr):
        await hook_mgr.create_hook("agent-1", {}, hook_id="hk-p2")
        await hook_mgr.resume_hook("hk-p2")
        pending = await hook_mgr.check_pending_hooks()
        assert len(pending) == 1

    @pytest.mark.asyncio
    async def test_check_pending_hooks_ignores_completed(self, hook_mgr):
        await hook_mgr.create_hook("agent-1", {}, hook_id="hk-p3")
        await hook_mgr.complete_hook("hk-p3")
        pending = await hook_mgr.check_pending_hooks()
        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_check_stale_hooks(self, hook_mgr):
        hook_mgr._config.stale_threshold_seconds = 0.0  # Everything is stale
        await hook_mgr.create_hook("agent-1", {}, hook_id="hk-stale")
        stale = await hook_mgr.check_stale_hooks()
        assert len(stale) == 1


# =============================================================================
# Hook Lifecycle Tests
# =============================================================================


class TestHookLifecycle:
    """Test hook lifecycle transitions."""

    @pytest.fixture
    def hook_mgr(self, tmp_path):
        config = HookManagerConfig(workspace_root=str(tmp_path), use_git_worktrees=False)
        return HookManager(config=config)

    @pytest.mark.asyncio
    async def test_resume_hook(self, hook_mgr):
        await hook_mgr.create_hook("agent-1", {}, hook_id="hk-resume")
        hook = await hook_mgr.resume_hook("hk-resume")
        assert hook is not None
        assert hook.status == HookStatus.RUNNING

    @pytest.mark.asyncio
    async def test_resume_nonexistent(self, hook_mgr):
        result = await hook_mgr.resume_hook("hk-missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_reassign_hook(self, hook_mgr):
        await hook_mgr.create_hook("agent-1", {}, hook_id="hk-reassign")
        hook = await hook_mgr.reassign_hook("hk-reassign", "agent-2")
        assert hook is not None
        assert hook.agent_id == "agent-2"
        assert hook.status == HookStatus.ASSIGNED
        assert hook.retry_count == 1

    @pytest.mark.asyncio
    async def test_complete_hook(self, hook_mgr):
        await hook_mgr.create_hook("agent-1", {}, hook_id="hk-complete")
        hook = await hook_mgr.complete_hook("hk-complete", result={"output": "done"})
        assert hook is not None
        assert hook.status == HookStatus.COMPLETED
        assert hook.result == {"output": "done"}
        assert hook.completed_at is not None

    @pytest.mark.asyncio
    async def test_complete_hook_writes_result_file(self, hook_mgr):
        await hook_mgr.create_hook("agent-1", {}, hook_id="hk-result")
        await hook_mgr.complete_hook("hk-result", result={"key": "val"})
        result_path = hook_mgr._hook_dir("hk-result") / HOOK_RESULT_FILE
        assert result_path.exists()
        data = json.loads(result_path.read_text())
        assert data["key"] == "val"

    @pytest.mark.asyncio
    async def test_fail_hook_with_retries(self, hook_mgr):
        await hook_mgr.create_hook("agent-1", {}, hook_id="hk-fail")
        hook = await hook_mgr.fail_hook("hk-fail", "timeout")
        assert hook.status == HookStatus.ASSIGNED  # Retried
        assert hook.retry_count == 1
        assert hook.error == "timeout"

    @pytest.mark.asyncio
    async def test_fail_hook_exhausts_retries(self, hook_mgr):
        await hook_mgr.create_hook("agent-1", {}, hook_id="hk-exhaust")
        hook = hook_mgr._hooks["hk-exhaust"]
        hook.retry_count = hook.max_retries  # Already at max

        result = await hook_mgr.fail_hook("hk-exhaust", "final failure")
        assert result.status == HookStatus.FAILED
        assert result.completed_at is not None

    @pytest.mark.asyncio
    async def test_abandon_hook(self, hook_mgr):
        await hook_mgr.create_hook("agent-1", {}, hook_id="hk-abandon")
        hook = await hook_mgr.abandon_hook("hk-abandon", "no longer needed")
        assert hook.status == HookStatus.ABANDONED
        assert hook.error == "no longer needed"

    @pytest.mark.asyncio
    async def test_abandon_nonexistent(self, hook_mgr):
        result = await hook_mgr.abandon_hook("hk-missing")
        assert result is None


# =============================================================================
# Persistence Tests
# =============================================================================


class TestPersistence:
    """Test JSONL persistence and recovery."""

    @pytest.mark.asyncio
    async def test_load_hook_from_disk(self, tmp_path):
        config = HookManagerConfig(workspace_root=str(tmp_path), use_git_worktrees=False)
        mgr = HookManager(config=config)

        await mgr.create_hook("agent-1", {"task": "build"}, hook_id="hk-disk")

        # Create a fresh manager to simulate restart
        mgr2 = HookManager(config=config)
        hook = await mgr2.get_hook("hk-disk")
        assert hook is not None
        assert hook.agent_id == "agent-1"
        assert hook.work_item == {"task": "build"}

    @pytest.mark.asyncio
    async def test_load_all_hooks_from_disk(self, tmp_path):
        config = HookManagerConfig(workspace_root=str(tmp_path), use_git_worktrees=False)
        mgr = HookManager(config=config)

        await mgr.create_hook("agent-1", {}, hook_id="hk-d1")
        await mgr.create_hook("agent-2", {}, hook_id="hk-d2")

        # Fresh manager
        mgr2 = HookManager(config=config)
        hooks = await mgr2.list_hooks()
        assert len(hooks) == 2

    @pytest.mark.asyncio
    async def test_jsonl_appends_history(self, tmp_path):
        config = HookManagerConfig(workspace_root=str(tmp_path), use_git_worktrees=False)
        mgr = HookManager(config=config)

        await mgr.create_hook("agent-1", {}, hook_id="hk-hist")
        await mgr.resume_hook("hk-hist")
        await mgr.complete_hook("hk-hist")

        state_file = mgr._hook_dir("hk-hist") / HOOK_STATE_FILE
        lines = state_file.read_text().strip().split("\n")
        assert len(lines) == 3  # create + resume + complete


# =============================================================================
# Cleanup Tests
# =============================================================================


class TestCleanup:
    """Test hook cleanup."""

    @pytest.fixture
    def hook_mgr(self, tmp_path):
        config = HookManagerConfig(workspace_root=str(tmp_path), use_git_worktrees=False)
        return HookManager(config=config)

    @pytest.mark.asyncio
    async def test_cleanup_completed(self, hook_mgr):
        await hook_mgr.create_hook("agent-1", {}, hook_id="hk-clean")
        await hook_mgr.complete_hook("hk-clean")
        # Set completed_at to a very old time to trigger cleanup
        hook_mgr._hooks["hk-clean"].completed_at = 1.0

        removed = await hook_mgr.cleanup_completed(max_age_seconds=1.0)
        assert removed == 1
        assert "hk-clean" not in hook_mgr._hooks

    @pytest.mark.asyncio
    async def test_cleanup_skips_active(self, hook_mgr):
        await hook_mgr.create_hook("agent-1", {}, hook_id="hk-active")
        removed = await hook_mgr.cleanup_completed(max_age_seconds=0)
        assert removed == 0


# =============================================================================
# Stats Tests
# =============================================================================


class TestHookStats:
    """Test hook statistics."""

    @pytest.fixture
    def hook_mgr(self, tmp_path):
        config = HookManagerConfig(workspace_root=str(tmp_path), use_git_worktrees=False)
        return HookManager(config=config)

    @pytest.mark.asyncio
    async def test_get_stats(self, hook_mgr):
        await hook_mgr.create_hook("agent-1", {}, hook_id="hk-st1")
        await hook_mgr.create_hook("agent-2", {}, hook_id="hk-st2")
        await hook_mgr.complete_hook("hk-st2")

        stats = await hook_mgr.get_stats()
        assert stats["total_hooks"] == 2
        assert stats["by_status"]["assigned"] == 1
        assert stats["by_status"]["completed"] == 1


# =============================================================================
# Path Safety Tests
# =============================================================================


class TestPathSafety:
    """Test path traversal prevention."""

    def test_hook_dir_sanitizes_path_traversal(self, tmp_path):
        config = HookManagerConfig(workspace_root=str(tmp_path), use_git_worktrees=False)
        mgr = HookManager(config=config)

        hook_dir = mgr._hook_dir("../../../etc/passwd")
        # Path traversal dots should be replaced
        assert ".." not in str(hook_dir.name)
        # The resulting path must be under the hooks directory
        assert str(hook_dir).startswith(str(mgr._hooks_dir))
