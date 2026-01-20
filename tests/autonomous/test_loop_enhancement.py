"""Tests for Nomic Loop Enhancement (Phase 5.1)."""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.autonomous.loop_enhancement import (
    ApprovalFlow,
    ApprovalRequest,
    ApprovalStatus,
    CodeVerifier,
    RollbackManager,
    RollbackPoint,
    SelfImprovementManager,
    VerificationResult,
)


class TestApprovalStatus:
    """Tests for ApprovalStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert ApprovalStatus.PENDING.value == "pending"
        assert ApprovalStatus.APPROVED.value == "approved"
        assert ApprovalStatus.REJECTED.value == "rejected"
        assert ApprovalStatus.TIMEOUT.value == "timeout"
        assert ApprovalStatus.AUTO_APPROVED.value == "auto_approved"


class TestApprovalFlow:
    """Tests for ApprovalFlow class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def approval_flow(self, temp_dir):
        """Create an ApprovalFlow instance."""
        return ApprovalFlow(
            storage_dir=temp_dir / "approvals",
            auto_approve_low_risk=True,
            default_timeout_seconds=60,
        )

    @pytest.mark.asyncio
    async def test_request_approval_low_risk_auto_approved(self, approval_flow):
        """Test that low risk requests are auto-approved."""
        request = await approval_flow.request_approval(
            title="Low risk change",
            description="A simple change",
            changes=[{"file": "test.py", "action": "modify"}],
            risk_level="low",
        )

        assert request.status == ApprovalStatus.AUTO_APPROVED
        assert request.approved_by == "auto"
        assert request.approved_at is not None

    @pytest.mark.asyncio
    async def test_request_approval_medium_risk_pending(self, approval_flow):
        """Test that medium risk requests require approval."""
        request = await approval_flow.request_approval(
            title="Medium risk change",
            description="A moderate change",
            changes=[{"file": "test.py", "action": "modify"}],
            risk_level="medium",
        )

        assert request.status == ApprovalStatus.PENDING
        assert request.approved_by is None

    @pytest.mark.asyncio
    async def test_request_approval_with_callback(self, temp_dir):
        """Test notification callback is called."""
        callback = MagicMock()
        approval_flow = ApprovalFlow(
            storage_dir=temp_dir / "approvals",
            notification_callback=callback,
        )

        await approval_flow.request_approval(
            title="Test",
            description="Test",
            changes=[],
            risk_level="high",
        )

        callback.assert_called_once()

    def test_approve_request(self, approval_flow):
        """Test approving a pending request."""
        # First create a request
        asyncio.run(
            approval_flow.request_approval(
                title="Test",
                description="Test",
                changes=[],
                risk_level="high",
            )
        )

        # Get the pending request
        pending = approval_flow.list_pending()
        assert len(pending) == 1

        # Approve it
        request = approval_flow.approve(pending[0].id, "test_user")

        assert request.status == ApprovalStatus.APPROVED
        assert request.approved_by == "test_user"
        assert request.approved_at is not None

    def test_reject_request(self, approval_flow):
        """Test rejecting a pending request."""
        asyncio.run(
            approval_flow.request_approval(
                title="Test",
                description="Test",
                changes=[],
                risk_level="high",
            )
        )

        pending = approval_flow.list_pending()
        request = approval_flow.reject(pending[0].id, "test_user", "Not needed")

        assert request.status == ApprovalStatus.REJECTED
        assert request.rejection_reason == "Not needed"

    def test_approve_nonexistent_request(self, approval_flow):
        """Test approving a nonexistent request raises error."""
        with pytest.raises(ValueError, match="Unknown approval request"):
            approval_flow.approve("nonexistent", "user")

    def test_list_pending(self, approval_flow):
        """Test listing pending requests."""
        # Create multiple requests
        asyncio.run(
            approval_flow.request_approval(
                title="Test 1",
                description="Test",
                changes=[],
                risk_level="high",
            )
        )
        asyncio.run(
            approval_flow.request_approval(
                title="Test 2",
                description="Test",
                changes=[],
                risk_level="critical",
            )
        )

        pending = approval_flow.list_pending()
        assert len(pending) == 2

    def test_persistence(self, temp_dir):
        """Test that requests are persisted to disk."""
        storage_dir = temp_dir / "approvals"

        # Create request
        flow1 = ApprovalFlow(storage_dir=storage_dir)
        asyncio.run(
            flow1.request_approval(
                title="Persist Test",
                description="Test",
                changes=[],
                risk_level="high",
            )
        )

        # Verify file exists
        files = list(storage_dir.glob("*.json"))
        assert len(files) == 1

        # Load in new instance
        flow2 = ApprovalFlow(storage_dir=storage_dir)
        pending = flow2.list_pending()
        assert len(pending) == 1
        assert pending[0].title == "Persist Test"


class TestRollbackManager:
    """Tests for RollbackManager class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def rollback_manager(self, temp_dir):
        """Create a RollbackManager instance."""
        return RollbackManager(
            backup_dir=temp_dir / "rollback",
            max_rollback_points=5,
            repo_path=temp_dir,
        )

    def test_create_rollback_point(self, rollback_manager):
        """Test creating a rollback point."""
        point = rollback_manager.create_rollback_point(
            description="Test rollback point",
            metadata={"test": True},
        )

        assert point.id is not None
        assert point.description == "Test rollback point"
        assert point.created_at is not None
        assert point.metadata == {"test": True}

    def test_create_rollback_point_with_file_backup(self, rollback_manager, temp_dir):
        """Test creating a rollback point with file backups."""
        # Create a test file
        test_file = temp_dir / "test.txt"
        test_file.write_text("original content")

        point = rollback_manager.create_rollback_point(
            description="File backup test",
            files_to_backup=[test_file],
        )

        assert str(test_file) in point.file_backups
        assert Path(point.file_backups[str(test_file)]).exists()

    def test_rollback_restores_files(self, rollback_manager, temp_dir):
        """Test that rollback restores backed up files."""
        # Create and backup a file
        test_file = temp_dir / "test.txt"
        test_file.write_text("original content")

        point = rollback_manager.create_rollback_point(
            description="Restore test",
            files_to_backup=[test_file],
        )

        # Modify the file
        test_file.write_text("modified content")
        assert test_file.read_text() == "modified content"

        # Rollback
        success = rollback_manager.rollback(point.id, restore_files=True)

        assert success
        assert test_file.read_text() == "original content"

    def test_list_rollback_points(self, rollback_manager):
        """Test listing rollback points."""
        rollback_manager.create_rollback_point(description="Point 1")
        rollback_manager.create_rollback_point(description="Point 2")

        points = rollback_manager.list_rollback_points()
        assert len(points) == 2
        # Should be sorted newest first
        assert points[0].description == "Point 2"

    def test_delete_rollback_point(self, rollback_manager):
        """Test deleting a rollback point."""
        point = rollback_manager.create_rollback_point(description="Delete me")

        success = rollback_manager.delete_rollback_point(point.id)
        assert success

        assert rollback_manager.get_rollback_point(point.id) is None

    def test_max_rollback_points_cleanup(self, rollback_manager):
        """Test that old rollback points are cleaned up."""
        # Create more than max_rollback_points
        for i in range(7):
            rollback_manager.create_rollback_point(description=f"Point {i}")

        points = rollback_manager.list_rollback_points()
        assert len(points) == 5  # max_rollback_points

    def test_rollback_nonexistent_point(self, rollback_manager):
        """Test rollback of nonexistent point fails."""
        success = rollback_manager.rollback("nonexistent")
        assert not success


class TestCodeVerifier:
    """Tests for CodeVerifier class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def code_verifier(self, temp_dir):
        """Create a CodeVerifier instance."""
        return CodeVerifier(
            repo_path=temp_dir,
            run_tests=False,  # Skip for unit tests
            run_lint=False,
            run_security=False,
        )

    @pytest.mark.asyncio
    async def test_syntax_check_valid_python(self, code_verifier, temp_dir):
        """Test syntax check passes for valid Python."""
        test_file = temp_dir / "valid.py"
        test_file.write_text("def foo():\n    return 42\n")

        result = await code_verifier.verify(files=[test_file])

        assert result.passed
        assert len(result.syntax_errors) == 0

    @pytest.mark.asyncio
    async def test_syntax_check_invalid_python(self, code_verifier, temp_dir):
        """Test syntax check fails for invalid Python."""
        test_file = temp_dir / "invalid.py"
        test_file.write_text("def foo(\n    return 42\n")

        result = await code_verifier.verify(files=[test_file])

        assert not result.passed
        assert len(result.syntax_errors) > 0

    @pytest.mark.asyncio
    async def test_quick_mode_skips_tests(self, temp_dir):
        """Test quick mode skips slow checks."""
        verifier = CodeVerifier(
            repo_path=temp_dir,
            run_tests=True,
            run_lint=True,
            run_security=True,
        )

        test_file = temp_dir / "test.py"
        test_file.write_text("x = 1\n")

        # In quick mode, tests shouldn't run
        result = await verifier.verify(files=[test_file], quick_mode=True)

        assert result.tests_run == 0


class TestSelfImprovementManager:
    """Tests for SelfImprovementManager class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def improvement_manager(self, temp_dir):
        """Create a SelfImprovementManager instance."""
        return SelfImprovementManager(
            rollback_manager=RollbackManager(backup_dir=temp_dir / "rollback"),
            code_verifier=CodeVerifier(
                repo_path=temp_dir,
                run_tests=False,
                run_lint=False,
                run_security=False,
            ),
            approval_flow=ApprovalFlow(
                storage_dir=temp_dir / "approvals",
                auto_approve_low_risk=True,
            ),
            auto_rollback_on_failure=True,
            require_approval_for_risk_levels={"high", "critical"},
        )

    @pytest.mark.asyncio
    async def test_start_improvement_cycle_low_risk(self, improvement_manager, temp_dir):
        """Test starting improvement cycle for low risk change."""
        test_file = temp_dir / "test.py"
        test_file.write_text("x = 1\n")

        can_proceed, rollback_id = await improvement_manager.start_improvement_cycle(
            description="Low risk change",
            files_to_modify=[test_file],
            risk_level="low",
        )

        assert can_proceed
        assert rollback_id is not None

    @pytest.mark.asyncio
    async def test_start_improvement_cycle_high_risk_pending(self, improvement_manager, temp_dir):
        """Test high risk change waits for approval (times out)."""
        test_file = temp_dir / "test.py"
        test_file.write_text("x = 1\n")

        # This will timeout waiting for approval
        improvement_manager.approval_flow.default_timeout_seconds = 0

        can_proceed, rollback_id = await improvement_manager.start_improvement_cycle(
            description="High risk change",
            files_to_modify=[test_file],
            risk_level="high",
        )

        # Should timeout and not proceed
        assert not can_proceed

    @pytest.mark.asyncio
    async def test_verify_and_complete_success(self, improvement_manager, temp_dir):
        """Test successful verification completes cycle."""
        test_file = temp_dir / "test.py"
        test_file.write_text("x = 1\n")

        # Start cycle
        await improvement_manager.start_improvement_cycle(
            description="Test",
            files_to_modify=[test_file],
            risk_level="low",
        )

        # Verify
        success, result = await improvement_manager.verify_and_complete(
            modified_files=[test_file],
            quick_verify=True,
        )

        assert success
        assert result.passed

    @pytest.mark.asyncio
    async def test_verify_and_complete_failure_rollback(self, improvement_manager, temp_dir):
        """Test failed verification triggers rollback."""
        test_file = temp_dir / "test.py"
        test_file.write_text("original content")

        # Start cycle (backs up file)
        await improvement_manager.start_improvement_cycle(
            description="Test",
            files_to_modify=[test_file],
            risk_level="low",
        )

        # Modify file with invalid syntax
        test_file.write_text("def broken(\n")

        # Verify (should fail and rollback)
        success, result = await improvement_manager.verify_and_complete(
            modified_files=[test_file],
            quick_verify=True,
        )

        assert not success
        assert not result.passed
        # File should be rolled back
        assert test_file.read_text() == "original content"

    def test_rollback_current(self, improvement_manager, temp_dir):
        """Test manual rollback of current cycle."""
        test_file = temp_dir / "test.py"
        test_file.write_text("original")

        asyncio.run(
            improvement_manager.start_improvement_cycle(
                description="Test",
                files_to_modify=[test_file],
                risk_level="low",
            )
        )

        test_file.write_text("modified")

        success = improvement_manager.rollback_current()
        assert success
        assert test_file.read_text() == "original"


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_default_values(self):
        """Test default values for VerificationResult."""
        result = VerificationResult(passed=True)

        assert result.passed
        assert result.tests_run == 0
        assert result.tests_passed == 0
        assert result.tests_failed == 0
        assert result.syntax_errors == []
        assert result.security_issues == []
        assert result.lint_warnings == []
        assert result.coverage_percent is None

    def test_with_failures(self):
        """Test VerificationResult with failures."""
        result = VerificationResult(
            passed=False,
            tests_run=10,
            tests_passed=8,
            tests_failed=2,
            syntax_errors=["test.py:1: syntax error"],
        )

        assert not result.passed
        assert result.tests_failed == 2
        assert len(result.syntax_errors) == 1
