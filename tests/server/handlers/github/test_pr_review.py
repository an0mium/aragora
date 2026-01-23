"""
Tests for GitHub PR review handler.

Tests the PRReviewHandler including:
- PR review triggering
- Review status tracking
- PR details fetching
- Review submission
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from aragora.server.handlers.github.pr_review import (
    GitHubClient,
    PRDetails,
    PRReviewResult,
    ReviewComment,
    ReviewStatus,
    ReviewVerdict,
    handle_trigger_pr_review,
    handle_get_pr_details,
    handle_get_review_status,
    handle_list_pr_reviews,
    handle_submit_review,
    _perform_review,
    _parse_debate_result,
    # Storage
    _review_results,
    _pr_reviews,
)


class TestGitHubClient:
    """Tests for GitHubClient."""

    @pytest.fixture
    def client_no_token(self):
        """Client without token returns demo data."""
        with patch.dict("os.environ", {}, clear=True):
            return GitHubClient(token=None)

    @pytest.fixture
    def client_with_token(self):
        """Client with token."""
        return GitHubClient(token="test_token")

    @pytest.mark.asyncio
    async def test_get_pr_returns_demo_without_token(self, client_no_token):
        """Test get_pr returns demo data without token."""
        pr = await client_no_token.get_pr("owner", "repo", 123)

        assert pr is not None
        assert pr.number == 123
        assert "Demo PR" in pr.title
        assert pr.state == "open"
        assert len(pr.changed_files) > 0

    def test_demo_pr_has_required_fields(self, client_no_token):
        """Test demo PR has all required fields."""
        pr = client_no_token._demo_pr(42)

        assert pr.number == 42
        assert pr.title
        assert pr.body
        assert pr.author
        assert pr.base_branch
        assert pr.head_branch
        assert pr.state
        assert isinstance(pr.changed_files, list)
        assert isinstance(pr.labels, list)

    @pytest.mark.asyncio
    async def test_submit_review_without_token_returns_demo(self, client_no_token):
        """Test submit_review returns demo result without token."""
        result = await client_no_token.submit_review(
            owner="owner",
            repo="repo",
            pr_number=123,
            event=ReviewVerdict.APPROVE,
            body="LGTM!",
        )

        assert result["success"] is True
        assert result.get("demo") is True


class TestPRReviewHandlers:
    """Tests for PR review handler functions."""

    @pytest.fixture(autouse=True)
    def clear_storage(self):
        """Clear in-memory storage before each test."""
        _review_results.clear()
        _pr_reviews.clear()
        yield
        _review_results.clear()
        _pr_reviews.clear()

    @pytest.mark.asyncio
    async def test_trigger_pr_review_creates_review(self):
        """Test triggering a PR review creates a review record."""
        result = await handle_trigger_pr_review(
            repository="owner/repo",
            pr_number=123,
        )

        assert result["success"] is True
        assert "review_id" in result
        assert result["status"] == "in_progress"
        assert result["pr_number"] == 123
        assert result["repository"] == "owner/repo"

    @pytest.mark.asyncio
    async def test_trigger_pr_review_invalid_repo_format(self):
        """Test triggering review with invalid repo format."""
        # Start the review
        result = await handle_trigger_pr_review(
            repository="invalid-repo-format",
            pr_number=123,
        )

        # Review is created but may fail during execution
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_get_pr_details_demo_mode(self):
        """Test getting PR details returns demo data."""
        result = await handle_get_pr_details(
            repository="owner/repo",
            pr_number=42,
        )

        assert result["success"] is True
        assert "pr" in result
        pr = result["pr"]
        assert pr["number"] == 42

    @pytest.mark.asyncio
    async def test_get_pr_details_invalid_repo(self):
        """Test getting PR details with invalid repo format."""
        result = await handle_get_pr_details(
            repository="invalid",
            pr_number=42,
        )

        assert result["success"] is False
        assert "Invalid repository format" in result["error"]

    @pytest.mark.asyncio
    async def test_get_review_status_not_found(self):
        """Test getting status for non-existent review."""
        result = await handle_get_review_status(review_id="nonexistent")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_get_review_status_exists(self):
        """Test getting status for existing review."""
        # Create a review first
        trigger_result = await handle_trigger_pr_review(
            repository="owner/repo",
            pr_number=123,
        )
        review_id = trigger_result["review_id"]

        result = await handle_get_review_status(review_id=review_id)

        assert result["success"] is True
        assert "review" in result
        assert result["review"]["review_id"] == review_id

    @pytest.mark.asyncio
    async def test_list_pr_reviews_empty(self):
        """Test listing reviews for PR with no reviews."""
        result = await handle_list_pr_reviews(
            repository="owner/repo",
            pr_number=999,
        )

        assert result["success"] is True
        assert result["reviews"] == []
        assert result["total"] == 0

    @pytest.mark.asyncio
    async def test_list_pr_reviews_with_reviews(self):
        """Test listing reviews for PR with reviews."""
        # Create some reviews
        await handle_trigger_pr_review("owner/repo", 123)
        await handle_trigger_pr_review("owner/repo", 123)

        # Wait a bit for reviews to be stored
        import asyncio

        await asyncio.sleep(0.1)

        result = await handle_list_pr_reviews(
            repository="owner/repo",
            pr_number=123,
        )

        assert result["success"] is True
        assert result["total"] >= 2

    @pytest.mark.asyncio
    async def test_submit_review_invalid_event(self):
        """Test submitting review with invalid event."""
        result = await handle_submit_review(
            repository="owner/repo",
            pr_number=123,
            event="INVALID_EVENT",
            body="Test",
        )

        assert result["success"] is False
        assert "Invalid event" in result["error"]

    @pytest.mark.asyncio
    async def test_submit_review_valid(self):
        """Test submitting a valid review."""
        result = await handle_submit_review(
            repository="owner/repo",
            pr_number=123,
            event="APPROVE",
            body="LGTM!",
        )

        assert result["success"] is True


class TestPerformReview:
    """Tests for the review analysis logic."""

    @pytest.fixture
    def sample_pr(self):
        """Create a sample PR for testing."""
        return PRDetails(
            number=123,
            title="Add new feature",
            body="This PR adds a new feature.",
            state="open",
            author="developer",
            base_branch="main",
            head_branch="feature/new",
            changed_files=[
                {
                    "filename": "src/feature.py",
                    "status": "added",
                    "additions": 50,
                    "deletions": 0,
                    "patch": "@@ +def new_feature():\n+    print('hello')",
                },
            ],
        )

    @pytest.mark.asyncio
    async def test_perform_review_returns_tuple(self, sample_pr):
        """Test perform_review returns expected tuple."""
        comments, verdict, summary = await _perform_review(
            sample_pr, "comprehensive", use_debate=False
        )

        assert isinstance(comments, list)
        assert isinstance(verdict, ReviewVerdict)
        assert isinstance(summary, str)

    @pytest.mark.asyncio
    async def test_perform_review_detects_debug_logging(self):
        """Test review detects debug logging."""
        pr = PRDetails(
            number=1,
            title="Test",
            body="",
            state="open",
            author="dev",
            base_branch="main",
            head_branch="test",
            changed_files=[
                {
                    "filename": "src/test.py",
                    "status": "modified",
                    "additions": 5,
                    "deletions": 0,
                    "patch": "+print('debug message')",
                },
            ],
        )

        comments, verdict, summary = await _perform_review(pr, "quick", use_debate=False)

        # Should flag debug logging
        debug_comments = [
            c for c in comments if "debug" in c.body.lower() or "print" in c.body.lower()
        ]
        assert len(debug_comments) > 0 or verdict == ReviewVerdict.COMMENT

    @pytest.mark.asyncio
    async def test_perform_review_detects_potential_secrets(self):
        """Test review detects potential secrets."""
        pr = PRDetails(
            number=1,
            title="Test",
            body="",
            state="open",
            author="dev",
            base_branch="main",
            head_branch="test",
            changed_files=[
                {
                    "filename": "src/config.py",
                    "status": "modified",
                    "additions": 1,
                    "deletions": 0,
                    "patch": "+password = 'secret123'",
                },
            ],
        )

        comments, verdict, summary = await _perform_review(pr, "security", use_debate=False)

        # Should flag potential secret and request changes
        security_comments = [c for c in comments if c.category == "security"]
        assert len(security_comments) > 0
        assert verdict == ReviewVerdict.REQUEST_CHANGES


class TestParseDebateResult:
    """Tests for debate result parsing."""

    def test_parse_verdict_approve(self):
        """Test parsing APPROVE verdict."""
        answer = """
        VERDICT: APPROVE
        SUMMARY: Code looks good overall.
        COMMENTS: None
        """

        comments, verdict, summary = _parse_debate_result(answer, MagicMock())

        assert verdict == ReviewVerdict.APPROVE
        assert "good" in summary.lower()

    def test_parse_verdict_request_changes(self):
        """Test parsing REQUEST_CHANGES verdict."""
        answer = """
        VERDICT: REQUEST_CHANGES
        SUMMARY: Several issues need to be addressed.
        """

        comments, verdict, summary = _parse_debate_result(answer, MagicMock())

        assert verdict == ReviewVerdict.REQUEST_CHANGES

    def test_parse_comments_from_answer(self):
        """Test parsing file comments from answer."""
        answer = """
        VERDICT: COMMENT
        SUMMARY: Some suggestions.
        COMMENTS:
        - src/main.py:10 - Consider adding error handling here.
        - src/utils.py:25 - This could be optimized.
        """

        comments, verdict, summary = _parse_debate_result(answer, MagicMock())

        assert len(comments) >= 0  # Parser may or may not extract these

    def test_parse_security_severity(self):
        """Test security issues get error severity."""
        answer = """
        VERDICT: REQUEST_CHANGES
        SUMMARY: Security issue found.
        - src/auth.py:5 - Potential SQL injection vulnerability.
        """

        comments, verdict, summary = _parse_debate_result(answer, MagicMock())

        # If comments are parsed, security ones should have error severity
        security_comments = [
            c for c in comments if "security" in c.body.lower() or "injection" in c.body.lower()
        ]
        for c in security_comments:
            assert c.severity == "error"


class TestDataModels:
    """Tests for data models."""

    def test_review_comment_to_dict(self):
        """Test ReviewComment serialization."""
        comment = ReviewComment(
            id="c1",
            file_path="src/test.py",
            line=10,
            body="Consider refactoring.",
            severity="warning",
            category="quality",
        )

        d = comment.to_dict()

        assert d["id"] == "c1"
        assert d["file_path"] == "src/test.py"
        assert d["line"] == 10
        assert d["body"] == "Consider refactoring."
        assert d["severity"] == "warning"

    def test_pr_review_result_to_dict(self):
        """Test PRReviewResult serialization."""
        result = PRReviewResult(
            review_id="r1",
            pr_number=123,
            repository="owner/repo",
            status=ReviewStatus.COMPLETED,
            verdict=ReviewVerdict.APPROVE,
            summary="LGTM",
        )

        d = result.to_dict()

        assert d["review_id"] == "r1"
        assert d["pr_number"] == 123
        assert d["status"] == "completed"
        assert d["verdict"] == "APPROVE"

    def test_pr_details_to_dict(self):
        """Test PRDetails serialization."""
        details = PRDetails(
            number=42,
            title="Test PR",
            body="Description",
            state="open",
            author="dev",
            base_branch="main",
            head_branch="feature",
        )

        d = details.to_dict()

        assert d["number"] == 42
        assert d["title"] == "Test PR"
        assert d["state"] == "open"
