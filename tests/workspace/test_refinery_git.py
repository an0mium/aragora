"""Tests for Refinery git operations (subprocess-based)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from aragora.workspace.refinery import (
    MergeRequest,
    MergeStatus,
    Refinery,
    RefineryConfig,
)


@pytest.fixture
def merge_request():
    """Create a sample merge request."""
    return MergeRequest(
        convoy_id="convoy-1",
        rig_id="rig-1",
        source_branch="feature/test",
        target_branch="main",
    )


class TestRunGit:
    """Test the _run_git helper."""

    @pytest.mark.asyncio
    async def test_run_git_success(self):
        refinery = Refinery()
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"output\n", b""))
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc

            rc, stdout, stderr = await refinery._run_git("status")

            assert rc == 0
            assert stdout == "output"
            assert stderr == ""
            mock_exec.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_git_failure(self):
        refinery = Refinery()
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b"error msg\n"))
            mock_proc.returncode = 1
            mock_exec.return_value = mock_proc

            rc, stdout, stderr = await refinery._run_git("checkout", "nonexistent")

            assert rc == 1
            assert stderr == "error msg"

    @pytest.mark.asyncio
    async def test_run_git_uses_work_dir(self):
        config = RefineryConfig(work_dir="/tmp/test-repo")
        refinery = Refinery(config=config)
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc

            await refinery._run_git("status")

            _, kwargs = mock_exec.call_args
            assert kwargs["cwd"] == "/tmp/test-repo"


class TestRunTests:
    """Test the _run_tests method."""

    @pytest.mark.asyncio
    async def test_tests_pass(self):
        refinery = Refinery()
        req = MergeRequest(convoy_id="c1", rig_id="r1", source_branch="b1")
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))
            mock_proc.returncode = 0
            mock_exec.return_value = mock_proc

            result = await refinery._run_tests(req)
            assert result is True

    @pytest.mark.asyncio
    async def test_tests_fail(self):
        refinery = Refinery()
        req = MergeRequest(convoy_id="c1", rig_id="r1", source_branch="b1")
        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_proc = AsyncMock()
            mock_proc.communicate = AsyncMock(return_value=(b"", b"FAILED"))
            mock_proc.returncode = 1
            mock_exec.return_value = mock_proc

            result = await refinery._run_tests(req)
            assert result is False

    @pytest.mark.asyncio
    async def test_empty_test_command_skips(self):
        config = RefineryConfig(test_command=[])
        refinery = Refinery(config=config)
        req = MergeRequest(convoy_id="c1", rig_id="r1", source_branch="b1")

        result = await refinery._run_tests(req)
        assert result is True

    @pytest.mark.asyncio
    async def test_missing_test_runner_passes(self):
        refinery = Refinery()
        req = MergeRequest(convoy_id="c1", rig_id="r1", source_branch="b1")
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError):
            result = await refinery._run_tests(req)
            assert result is True  # Doesn't block merge


class TestCheckApproval:
    """Test the _check_approval method."""

    @pytest.mark.asyncio
    async def test_auto_approve_config(self):
        config = RefineryConfig(auto_approve=True)
        refinery = Refinery(config=config)
        req = MergeRequest(convoy_id="c1", rig_id="r1", source_branch="b1")

        result = await refinery._check_approval(req)
        assert result is True

    @pytest.mark.asyncio
    async def test_env_var_approve(self):
        refinery = Refinery()
        req = MergeRequest(convoy_id="c1", rig_id="r1", source_branch="b1")
        with patch.dict("os.environ", {"ARAGORA_AUTO_APPROVE": "true"}):
            result = await refinery._check_approval(req)
            assert result is True

    @pytest.mark.asyncio
    async def test_metadata_approved(self):
        refinery = Refinery()
        req = MergeRequest(
            convoy_id="c1",
            rig_id="r1",
            source_branch="b1",
            metadata={"approved": True},
        )
        result = await refinery._check_approval(req)
        assert result is True

    @pytest.mark.asyncio
    async def test_not_approved(self):
        refinery = Refinery()
        req = MergeRequest(convoy_id="c1", rig_id="r1", source_branch="b1")
        with patch.dict("os.environ", {}, clear=True):
            result = await refinery._check_approval(req)
            assert result is False


class TestRebase:
    """Test the _rebase method."""

    @pytest.mark.asyncio
    async def test_rebase_success(self, merge_request):
        refinery = Refinery()
        refinery._run_git = AsyncMock(return_value=(0, "", ""))

        result = await refinery._rebase(merge_request)
        assert result is True
        assert refinery._run_git.call_count == 2  # checkout + rebase

    @pytest.mark.asyncio
    async def test_rebase_checkout_failure(self, merge_request):
        refinery = Refinery()
        refinery._run_git = AsyncMock(return_value=(1, "", "branch not found"))

        result = await refinery._rebase(merge_request)
        assert result is False

    @pytest.mark.asyncio
    async def test_rebase_conflict(self, merge_request):
        refinery = Refinery()
        call_count = 0

        async def mock_git(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # checkout
                return (0, "", "")
            elif call_count == 2:  # rebase
                return (1, "", "CONFLICT in file.py")
            elif call_count == 3:  # diff --name-only
                return (0, "file.py\nother.py", "")
            elif call_count == 4:  # rebase --abort
                return (0, "", "")
            return (0, "", "")

        refinery._run_git = mock_git

        result = await refinery._rebase(merge_request)
        assert result is False
        assert merge_request.conflict_files == ["file.py", "other.py"]


class TestDoMerge:
    """Test the _do_merge method."""

    @pytest.mark.asyncio
    async def test_merge_success(self, merge_request):
        refinery = Refinery()
        call_count = 0

        async def mock_git(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # checkout target
                return (0, "", "")
            elif call_count == 2:  # merge
                return (0, "", "")
            elif call_count == 3:  # rev-parse HEAD
                return (0, "abc123def456", "")
            return (0, "", "")

        refinery._run_git = mock_git

        sha = await refinery._do_merge(merge_request)
        assert sha == "abc123def456"

    @pytest.mark.asyncio
    async def test_merge_conflict(self, merge_request):
        refinery = Refinery()
        call_count = 0

        async def mock_git(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # checkout
                return (0, "", "")
            elif call_count == 2:  # merge fails
                return (1, "", "CONFLICT")
            elif call_count == 3:  # diff --name-only
                return (0, "conflict.py", "")
            elif call_count == 4:  # merge --abort
                return (0, "", "")
            return (0, "", "")

        refinery._run_git = mock_git

        sha = await refinery._do_merge(merge_request)
        assert sha is None
        assert "conflict.py" in merge_request.conflict_files

    @pytest.mark.asyncio
    async def test_merge_checkout_failure(self, merge_request):
        refinery = Refinery()
        refinery._run_git = AsyncMock(return_value=(1, "", "error"))

        sha = await refinery._do_merge(merge_request)
        assert sha is None


class TestDoRollback:
    """Test the _do_rollback method."""

    @pytest.mark.asyncio
    async def test_rollback_success(self, merge_request):
        merge_request.merge_commit = "abc123"
        refinery = Refinery()
        refinery._run_git = AsyncMock(return_value=(0, "", ""))

        result = await refinery._do_rollback(merge_request)
        assert result is True

    @pytest.mark.asyncio
    async def test_rollback_no_commit(self, merge_request):
        merge_request.merge_commit = None
        refinery = Refinery()

        result = await refinery._do_rollback(merge_request)
        assert result is False

    @pytest.mark.asyncio
    async def test_rollback_failure(self, merge_request):
        merge_request.merge_commit = "abc123"
        refinery = Refinery()
        call_count = 0

        async def mock_git(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:  # checkout
                return (0, "", "")
            elif call_count == 2:  # revert fails
                return (1, "", "revert failed")
            return (0, "", "")

        refinery._run_git = mock_git

        result = await refinery._do_rollback(merge_request)
        assert result is False
