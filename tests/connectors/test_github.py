"""Tests for GitHub connector."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.github import (
    ALLOWED_STATES,
    GitHubConnector,
    MAX_QUERY_LENGTH,
    VALID_NUMBER_PATTERN,
    VALID_REPO_PATTERN,
)


class TestValidationPatterns:
    """Test input validation patterns."""

    def test_valid_repo_pattern_accepts_valid(self):
        """Test valid repo patterns are accepted."""
        valid_repos = [
            "owner/repo",
            "my-org/my-repo",
            "user123/project_name",
            "Company.Inc/repo.js",
            "a/b",
        ]
        for repo in valid_repos:
            assert VALID_REPO_PATTERN.match(repo), f"Should accept: {repo}"

    def test_valid_repo_pattern_rejects_invalid(self):
        """Test invalid repo patterns are rejected."""
        invalid_repos = [
            "owner",  # No slash
            "/repo",  # Missing owner
            "owner/",  # Missing repo
            "owner/repo/extra",  # Too many parts
            "owner repo",  # Space instead of slash
            "owner;rm -rf /",  # Injection attempt
            "../etc/passwd",  # Path traversal
        ]
        for repo in invalid_repos:
            assert not VALID_REPO_PATTERN.match(repo), f"Should reject: {repo}"

    def test_valid_number_pattern_accepts_valid(self):
        """Test valid issue/PR numbers are accepted."""
        valid_numbers = ["1", "123", "9999", "1234567890"]
        for num in valid_numbers:
            assert VALID_NUMBER_PATTERN.match(num), f"Should accept: {num}"

    def test_valid_number_pattern_rejects_invalid(self):
        """Test invalid numbers are rejected."""
        invalid_numbers = [
            "",  # Empty
            "abc",  # Letters
            "123abc",  # Mixed
            "-1",  # Negative
            "1.5",  # Decimal
            "12345678901",  # Too long (> 10 digits)
        ]
        for num in invalid_numbers:
            assert not VALID_NUMBER_PATTERN.match(num), f"Should reject: {num}"


class TestGitHubConnectorInit:
    """Test GitHubConnector initialization."""

    def test_init_with_valid_repo(self):
        """Test initialization with valid repo."""
        connector = GitHubConnector(repo="owner/repo")
        assert connector.repo == "owner/repo"
        assert connector.use_gh_cli is True
        assert connector.token is None

    def test_init_with_invalid_repo_raises(self):
        """Test initialization with invalid repo raises ValueError."""
        with pytest.raises(ValueError, match="Invalid repo format"):
            GitHubConnector(repo="invalid;injection")

    def test_init_with_token(self):
        """Test initialization with API token."""
        connector = GitHubConnector(repo="owner/repo", token="ghp_xxx")
        assert connector.token == "ghp_xxx"

    def test_init_without_repo(self):
        """Test initialization without repo (for code search)."""
        connector = GitHubConnector()
        assert connector.repo is None

    def test_source_type(self):
        """Test source_type property."""
        connector = GitHubConnector(repo="owner/repo")
        from aragora.reasoning.provenance import SourceType

        assert connector.source_type == SourceType.EXTERNAL_API

    def test_name_property(self):
        """Test name property."""
        connector = GitHubConnector(repo="owner/repo")
        assert connector.name == "GitHub"


class TestValidationMethods:
    """Test validation helper methods."""

    def test_validate_repo_valid(self):
        """Test _validate_repo with valid input."""
        assert GitHubConnector._validate_repo("owner/repo")
        assert GitHubConnector._validate_repo("my-org/project.js")

    def test_validate_repo_invalid(self):
        """Test _validate_repo with invalid input."""
        assert not GitHubConnector._validate_repo("")
        assert not GitHubConnector._validate_repo(None)
        assert not GitHubConnector._validate_repo("invalid")

    def test_validate_number_valid(self):
        """Test _validate_number with valid input."""
        assert GitHubConnector._validate_number("123")
        assert GitHubConnector._validate_number("1")

    def test_validate_number_invalid(self):
        """Test _validate_number with invalid input."""
        assert not GitHubConnector._validate_number("")
        assert not GitHubConnector._validate_number(None)
        assert not GitHubConnector._validate_number("abc")

    def test_validate_state_valid(self):
        """Test _validate_state with valid states."""
        assert GitHubConnector._validate_state("open") == "open"
        assert GitHubConnector._validate_state("closed") == "closed"
        assert GitHubConnector._validate_state("all") == "all"
        assert GitHubConnector._validate_state("merged") == "merged"

    def test_validate_state_normalizes(self):
        """Test _validate_state normalizes input."""
        assert GitHubConnector._validate_state("OPEN") == "open"
        assert GitHubConnector._validate_state("  Closed  ") == "closed"

    def test_validate_state_invalid_returns_all(self):
        """Test _validate_state returns 'all' for invalid states."""
        assert GitHubConnector._validate_state("invalid") == "all"
        assert GitHubConnector._validate_state("") == "all"
        assert GitHubConnector._validate_state(None) == "all"


class TestGhCliCheck:
    """Test gh CLI availability checking."""

    def test_check_gh_cli_available(self):
        """Test gh CLI check when available."""
        connector = GitHubConnector(repo="owner/repo")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = connector._check_gh_cli()

        assert result is True
        assert connector._gh_available is True

    def test_check_gh_cli_not_available(self):
        """Test gh CLI check when not available."""
        connector = GitHubConnector(repo="owner/repo")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            result = connector._check_gh_cli()

        assert result is False

    def test_check_gh_cli_not_installed(self):
        """Test gh CLI check when gh is not installed."""
        connector = GitHubConnector(repo="owner/repo")

        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = connector._check_gh_cli()

        assert result is False

    def test_check_gh_cli_cached(self):
        """Test gh CLI check result is cached."""
        connector = GitHubConnector(repo="owner/repo")
        connector._gh_available = True

        # Should not call subprocess.run
        with patch("subprocess.run") as mock_run:
            result = connector._check_gh_cli()

        assert result is True
        mock_run.assert_not_called()


class TestSearch:
    """Test search functionality."""

    @pytest.mark.asyncio
    async def test_search_returns_empty_for_short_query(self):
        """Test search returns empty for too short queries."""
        connector = GitHubConnector(repo="owner/repo")
        results = await connector.search("a")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_returns_empty_for_long_query(self):
        """Test search returns empty for too long queries."""
        connector = GitHubConnector(repo="owner/repo")
        results = await connector.search("x" * (MAX_QUERY_LENGTH + 1))
        assert results == []

    @pytest.mark.asyncio
    async def test_search_issues_requires_repo(self):
        """Test issue search requires repo."""
        connector = GitHubConnector()  # No repo
        results = await connector.search("bug", search_type="issues")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_code_without_repo(self):
        """Test code search works without repo."""
        connector = GitHubConnector()
        connector._gh_available = True

        mock_output = json.dumps(
            [
                {
                    "path": "src/main.py",
                    "repository": {"fullName": "owner/repo"},
                    "textMatches": [{"fragment": "def main():"}],
                }
            ]
        )

        with patch.object(connector, "_run_gh", return_value=mock_output):
            results = await connector.search("def main", search_type="code")

        assert len(results) == 1
        assert results[0].metadata["type"] == "code"

    @pytest.mark.asyncio
    async def test_search_issues_success(self):
        """Test successful issue search."""
        connector = GitHubConnector(repo="owner/repo")
        connector._gh_available = True

        mock_output = json.dumps(
            [
                {
                    "number": 123,
                    "title": "Bug: Something broken",
                    "body": "Description of the bug",
                    "author": {"login": "user1"},
                    "createdAt": "2024-01-15T10:00:00Z",
                    "url": "https://github.com/owner/repo/issues/123",
                    "state": "open",
                    "labels": [{"name": "bug"}],
                }
            ]
        )

        with patch.object(connector, "_run_gh", return_value=mock_output):
            results = await connector.search("bug", search_type="issues")

        assert len(results) == 1
        assert results[0].title == "Issue #123: Bug: Something broken"
        assert results[0].metadata["type"] == "issue"
        assert results[0].metadata["number"] == 123
        assert results[0].authority == 0.8  # bug label increases authority

    @pytest.mark.asyncio
    async def test_search_prs_success(self):
        """Test successful PR search."""
        connector = GitHubConnector(repo="owner/repo")
        connector._gh_available = True

        mock_output = json.dumps(
            [
                {
                    "number": 456,
                    "title": "Add feature X",
                    "body": "Implementation details",
                    "author": {"login": "contributor"},
                    "createdAt": "2024-01-20T15:00:00Z",
                    "url": "https://github.com/owner/repo/pull/456",
                    "state": "merged",
                    "mergedAt": "2024-01-21T10:00:00Z",
                }
            ]
        )

        with patch.object(connector, "_run_gh", return_value=mock_output):
            results = await connector.search("feature", search_type="prs")

        assert len(results) == 1
        assert results[0].title == "PR #456: Add feature X"
        assert results[0].metadata["type"] == "pr"
        assert results[0].metadata["merged"] is True
        assert results[0].authority == 0.8  # merged PR has higher authority


class TestFetch:
    """Test fetch functionality."""

    @pytest.mark.asyncio
    async def test_fetch_invalid_evidence_id(self):
        """Test fetch returns None for invalid evidence ID."""
        connector = GitHubConnector(repo="owner/repo")
        result = await connector.fetch("invalid-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_issue_validates_repo(self):
        """Test fetch_issue validates repo format."""
        connector = GitHubConnector(repo="owner/repo")
        connector._gh_available = True

        # Invalid repo in evidence_id
        result = await connector.fetch("gh-issue:invalid;repo:123")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_issue_validates_number(self):
        """Test fetch_issue validates issue number."""
        connector = GitHubConnector(repo="owner/repo")
        connector._gh_available = True

        # Invalid number in evidence_id
        result = await connector.fetch("gh-issue:owner/repo:abc")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_issue_success(self):
        """Test successful issue fetch."""
        connector = GitHubConnector(repo="owner/repo")
        connector._gh_available = True

        mock_output = json.dumps(
            {
                "number": 123,
                "title": "Issue Title",
                "body": "Issue body content",
                "author": {"login": "author"},
                "createdAt": "2024-01-15T10:00:00Z",
                "url": "https://github.com/owner/repo/issues/123",
                "state": "open",
                "labels": [],
                "comments": [{"author": {"login": "commenter"}, "body": "Comment text"}],
            }
        )

        with patch.object(connector, "_run_gh", return_value=mock_output):
            result = await connector.fetch("gh-issue:owner/repo:123")

        assert result is not None
        assert result.title == "Issue #123: Issue Title"
        assert result.metadata["comment_count"] == 1

    @pytest.mark.asyncio
    async def test_fetch_pr_success(self):
        """Test successful PR fetch."""
        connector = GitHubConnector(repo="owner/repo")
        connector._gh_available = True

        mock_output = json.dumps(
            {
                "number": 456,
                "title": "PR Title",
                "body": "PR description",
                "author": {"login": "author"},
                "createdAt": "2024-01-20T15:00:00Z",
                "url": "https://github.com/owner/repo/pull/456",
                "state": "merged",
                "mergedAt": "2024-01-21T10:00:00Z",
                "reviews": [{"author": {"login": "reviewer"}, "state": "APPROVED", "body": "LGTM"}],
            }
        )

        with patch.object(connector, "_run_gh", return_value=mock_output):
            result = await connector.fetch("gh-pr:owner/repo:456")

        assert result is not None
        assert result.title == "PR #456: PR Title"
        assert result.metadata["merged"] is True
        assert result.authority == 0.85  # merged PR


class TestPROperations:
    """Test PR-specific operations."""

    @pytest.mark.asyncio
    async def test_fetch_pr_diff_invalid_url(self):
        """Test fetch_pr_diff with invalid URL."""
        connector = GitHubConnector(repo="owner/repo")
        result = await connector.fetch_pr_diff("not-a-url")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_pr_diff_success(self):
        """Test successful PR diff fetch."""
        connector = GitHubConnector(repo="owner/repo")
        connector._gh_available = True

        mock_diff = "diff --git a/file.py b/file.py\n..."

        with patch.object(connector, "_run_gh", return_value=mock_diff):
            result = await connector.fetch_pr_diff("https://github.com/owner/repo/pull/123")

        assert result == mock_diff

    @pytest.mark.asyncio
    async def test_fetch_pr_files_success(self):
        """Test successful PR files fetch."""
        connector = GitHubConnector(repo="owner/repo")
        connector._gh_available = True

        mock_output = json.dumps(
            {
                "files": [
                    {"path": "file1.py", "additions": 10, "deletions": 5},
                    {"path": "file2.py", "additions": 20, "deletions": 0},
                ]
            }
        )

        with patch.object(connector, "_run_gh", return_value=mock_output):
            result = await connector.fetch_pr_files("https://github.com/owner/repo/pull/123")

        assert len(result) == 2
        assert result[0]["path"] == "file1.py"

    @pytest.mark.asyncio
    async def test_post_pr_review_invalid_url(self):
        """Test post_pr_review with invalid URL."""
        connector = GitHubConnector(repo="owner/repo")
        result = await connector.post_pr_review("invalid", "Review body")
        assert result is False

    @pytest.mark.asyncio
    async def test_post_pr_review_success(self):
        """Test successful PR review posting."""
        connector = GitHubConnector(repo="owner/repo")
        connector._gh_available = True

        with patch.object(connector, "_run_gh", return_value="{}"):
            result = await connector.post_pr_review(
                "https://github.com/owner/repo/pull/123",
                "Review body",
                event="APPROVE",
            )

        assert result is True


class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_search_handles_json_decode_error(self):
        """Test search handles invalid JSON response."""
        connector = GitHubConnector(repo="owner/repo")
        connector._gh_available = True

        with patch.object(connector, "_run_gh", return_value="invalid json"):
            results = await connector.search("query", search_type="issues")

        assert results == []

    @pytest.mark.asyncio
    async def test_search_handles_gh_failure(self):
        """Test search handles gh CLI failure."""
        connector = GitHubConnector(repo="owner/repo")
        connector._gh_available = True

        with patch.object(connector, "_run_gh", return_value=None):
            results = await connector.search("query", search_type="issues")

        assert results == []

    @pytest.mark.asyncio
    async def test_fetch_handles_json_decode_error(self):
        """Test fetch handles invalid JSON response."""
        connector = GitHubConnector(repo="owner/repo")
        connector._gh_available = True

        with patch.object(connector, "_run_gh", return_value="invalid json"):
            result = await connector.fetch("gh-issue:owner/repo:123")

        assert result is None


class TestAllowedStates:
    """Test ALLOWED_STATES constant."""

    def test_contains_expected_states(self):
        """Test ALLOWED_STATES contains expected values."""
        assert "all" in ALLOWED_STATES
        assert "open" in ALLOWED_STATES
        assert "closed" in ALLOWED_STATES
        assert "merged" in ALLOWED_STATES

    def test_is_frozen(self):
        """Test ALLOWED_STATES is immutable."""
        assert isinstance(ALLOWED_STATES, frozenset)
