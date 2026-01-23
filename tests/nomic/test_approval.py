"""
Tests for approval workflow module.

Tests:
- Approval levels and policies
- Approval requests and voting
- Timeout behavior
- Multi-level approval gates
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from aragora.nomic.approval import (
    ApprovalLevel,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalResult,
    ApprovalStatus,
    ApprovalWorkflow,
    ApproverVote,
    FileChange,
    is_protected_file,
    quick_approval_check,
)


class TestApprovalLevel:
    """Tests for ApprovalLevel enum."""

    def test_level_values(self):
        """Test level value strings."""
        assert ApprovalLevel.INFO.value == "info"
        assert ApprovalLevel.REVIEW.value == "review"
        assert ApprovalLevel.CRITICAL.value == "critical"


class TestFileChange:
    """Tests for FileChange dataclass."""

    def test_create_file_change(self):
        """Test creating a file change."""
        change = FileChange(
            path="src/module.py",
            change_type="modify",
            lines_added=10,
            lines_removed=5,
        )
        assert change.path == "src/module.py"
        assert change.change_type == "modify"
        assert change.lines_added == 10
        assert change.lines_removed == 5

    def test_file_change_to_dict(self):
        """Test serialization."""
        change = FileChange(path="test.py", change_type="add")
        data = change.to_dict()
        assert data["path"] == "test.py"
        assert data["change_type"] == "add"


class TestApprovalPolicy:
    """Tests for ApprovalPolicy."""

    @pytest.fixture
    def policy(self):
        """Create a policy for testing."""
        return ApprovalPolicy()

    def test_critical_files(self, policy):
        """Test critical file detection."""
        assert policy.get_approval_level("CLAUDE.md") == ApprovalLevel.CRITICAL
        assert policy.get_approval_level(".env") == ApprovalLevel.CRITICAL
        assert policy.get_approval_level("aragora/core.py") == ApprovalLevel.CRITICAL
        assert policy.get_approval_level(".github/workflows/ci.yml") == ApprovalLevel.CRITICAL

    def test_review_files(self, policy):
        """Test review-level files."""
        assert policy.get_approval_level("src/handlers/api.py") == ApprovalLevel.REVIEW
        assert policy.get_approval_level("models/user.py") == ApprovalLevel.REVIEW
        assert policy.get_approval_level("requirements.txt") == ApprovalLevel.REVIEW

    def test_info_files(self, policy):
        """Test info-level files."""
        assert policy.get_approval_level("tests/test_module.py") == ApprovalLevel.INFO
        assert policy.get_approval_level("docs/README.md") == ApprovalLevel.INFO

    def test_default_to_review(self, policy):
        """Test that unknown files default to review."""
        assert policy.get_approval_level("random_script.py") == ApprovalLevel.REVIEW

    def test_add_custom_pattern(self, policy):
        """Test adding custom patterns."""
        policy.add_critical_pattern("custom/**")
        assert policy.get_approval_level("custom/secret.py") == ApprovalLevel.CRITICAL

    def test_get_max_approval_level(self, policy):
        """Test getting max level for multiple changes."""
        changes = [
            FileChange(path="tests/test.py", change_type="modify"),
            FileChange(path="src/module.py", change_type="modify"),
            FileChange(path="CLAUDE.md", change_type="modify"),
        ]
        assert policy.get_max_approval_level(changes) == ApprovalLevel.CRITICAL

    def test_get_max_approval_level_empty(self, policy):
        """Test max level with no changes."""
        assert policy.get_max_approval_level([]) == ApprovalLevel.INFO


class TestApprovalWorkflow:
    """Tests for ApprovalWorkflow."""

    @pytest.fixture
    def workflow(self):
        """Create a workflow for testing."""
        return ApprovalWorkflow(default_approvers=["alice", "bob"])

    def test_create_request(self, workflow):
        """Test creating an approval request."""
        changes = [FileChange(path="test.py", change_type="add")]
        request = workflow.create_request(
            changes=changes,
            description="Add test file",
        )
        assert request.request_id
        assert request.status == ApprovalStatus.PENDING
        assert len(request.approvers) == 2

    def test_create_request_with_level_override(self, workflow):
        """Test creating request with level override."""
        changes = [FileChange(path="test.py", change_type="add")]
        request = workflow.create_request(
            changes=changes,
            level=ApprovalLevel.CRITICAL,
        )
        assert request.level == ApprovalLevel.CRITICAL

    def test_submit_vote_approve(self, workflow):
        """Test submitting an approval vote."""
        changes = [FileChange(path="test.py", change_type="add")]
        request = workflow.create_request(changes=changes)

        result = workflow.submit_vote(
            request_id=request.request_id,
            approver_id="alice",
            approved=True,
            comment="Looks good!",
        )
        assert result is True
        assert len(request.votes) == 1
        assert request.votes[0].approved is True

    def test_submit_vote_reject(self, workflow):
        """Test submitting a rejection vote."""
        changes = [FileChange(path="test.py", change_type="add")]
        request = workflow.create_request(changes=changes)

        workflow.submit_vote(
            request_id=request.request_id,
            approver_id="alice",
            approved=False,
            comment="Needs changes",
        )
        assert request.status == ApprovalStatus.REJECTED

    def test_submit_vote_duplicate(self, workflow):
        """Test that duplicate votes are rejected."""
        changes = [FileChange(path="test.py", change_type="add")]
        request = workflow.create_request(changes=changes)

        workflow.submit_vote(request.request_id, "alice", True)
        result = workflow.submit_vote(request.request_id, "alice", True)
        assert result is False

    def test_submit_vote_unauthorized(self, workflow):
        """Test that unauthorized voters are rejected."""
        changes = [FileChange(path="test.py", change_type="add")]
        request = workflow.create_request(changes=changes, approvers=["alice"])

        result = workflow.submit_vote(request.request_id, "charlie", True)
        assert result is False

    def test_cancel_request(self, workflow):
        """Test cancelling a request."""
        changes = [FileChange(path="test.py", change_type="add")]
        request = workflow.create_request(changes=changes)

        result = workflow.cancel_request(request.request_id, "Changed plans")
        assert result is True
        assert request.status == ApprovalStatus.CANCELLED

    def test_get_pending_requests(self, workflow):
        """Test getting pending requests."""
        changes = [FileChange(path="test.py", change_type="add")]
        workflow.create_request(changes=changes)
        workflow.create_request(changes=changes)

        pending = workflow.get_pending_requests()
        assert len(pending) == 2

    @pytest.mark.asyncio
    async def test_request_approval_info_level(self, workflow):
        """Test INFO level auto-approves."""
        changes = [FileChange(path="docs/readme.md", change_type="modify")]

        result = await workflow.request_approval(
            changes=changes,
            level=ApprovalLevel.INFO,
            timeout_seconds=1,
        )

        assert result.approved is True
        assert result.status == ApprovalStatus.APPROVED

    @pytest.mark.asyncio
    async def test_request_approval_review_level(self, workflow):
        """Test REVIEW level requires vote."""
        changes = [FileChange(path="src/module.py", change_type="modify")]

        # Create request and immediately vote
        async def vote_task():
            await asyncio.sleep(0.1)
            pending = workflow.get_pending_requests()
            if pending:
                workflow.submit_vote(pending[0].request_id, "alice", True)

        asyncio.create_task(vote_task())

        result = await workflow.request_approval(
            changes=changes,
            level=ApprovalLevel.REVIEW,
            timeout_seconds=1,
        )

        assert result.approved is True

    @pytest.mark.asyncio
    async def test_request_approval_timeout(self, workflow):
        """Test timeout behavior."""
        changes = [FileChange(path="src/module.py", change_type="modify")]

        result = await workflow.request_approval(
            changes=changes,
            level=ApprovalLevel.REVIEW,
            timeout_seconds=0.1,  # Very short timeout
        )

        assert result.status == ApprovalStatus.TIMED_OUT
        assert result.approved is False


class TestApprovalRequest:
    """Tests for ApprovalRequest dataclass."""

    def test_create_request(self):
        """Test creating a request."""
        request = ApprovalRequest(
            request_id="test-123",
            changes=[FileChange(path="test.py", change_type="add")],
            level=ApprovalLevel.REVIEW,
            approvers=["alice"],
        )
        assert request.request_id == "test-123"
        assert request.status == ApprovalStatus.PENDING

    def test_request_to_dict(self):
        """Test serialization."""
        request = ApprovalRequest(
            request_id="test-123",
            changes=[FileChange(path="test.py", change_type="add")],
            level=ApprovalLevel.REVIEW,
            approvers=["alice"],
        )
        data = request.to_dict()
        assert data["request_id"] == "test-123"
        assert data["level"] == "review"


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_quick_approval_check(self):
        """Test quick approval check."""
        changes = [FileChange(path="CLAUDE.md", change_type="modify")]
        level = quick_approval_check(changes)
        assert level == ApprovalLevel.CRITICAL

    def test_is_protected_file(self):
        """Test protected file check."""
        assert is_protected_file("CLAUDE.md") is True
        assert is_protected_file(".env") is True
        assert is_protected_file("tests/test.py") is False
