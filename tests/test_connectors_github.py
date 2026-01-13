"""
Tests for GitHub Connector.

Tests cover:
- Input validation (repo pattern, query length)
- gh CLI integration (mocked)
- Issue/PR/Code search
- Fetch operations
- Authority calculation
"""

import json
from unittest.mock import Mock, AsyncMock, patch

import pytest

from aragora.connectors.github import (
    GitHubConnector,
    VALID_REPO_PATTERN,
    VALID_NUMBER_PATTERN,
    ALLOWED_STATES,
    MAX_QUERY_LENGTH,
)
from aragora.connectors.base import Evidence
from aragora.reasoning.provenance import SourceType


class TestValidRepoPattern:
    """Tests for VALID_REPO_PATTERN regex."""

    def test_valid_repo_patterns_match(self):
        """Valid repo formats should match pattern."""
        valid_repos = [
            "owner/repo",
            "my-org/my-repo",
            "user123/project.js",
            "Org_Name/Repo-Name",
            "a/b",
            "microsoft/vscode",
            "anthropics/claude-code",
        ]
        for repo in valid_repos:
            assert VALID_REPO_PATTERN.match(repo), f"{repo} should be valid"

    def test_invalid_repo_patterns_rejected(self):
        """Invalid repo formats should not match."""
        invalid_repos = [
            "../../../etc/passwd",
            "path/to/file/extra",
            "just-repo",
            "",
            "/repo",
            "owner/",
            "owner//repo",
            "owner/repo;rm -rf /",
        ]
        for repo in invalid_repos:
            assert not VALID_REPO_PATTERN.match(repo), f"{repo} should be invalid"


class TestGitHubConnectorInit:
    """Tests for GitHubConnector initialization."""

    def test_valid_repo_accepted(self):
        """Valid repo format should be accepted."""
        connector = GitHubConnector(repo="owner/repo")
        assert connector.repo == "owner/repo"

    def test_invalid_repo_raises(self):
        """Invalid repo format should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid repo format"):
            GitHubConnector(repo="../../../etc/passwd")

    def test_no_repo_allowed(self):
        """Connector can be created without repo (for code search)."""
        connector = GitHubConnector()
        assert connector.repo is None

    def test_source_type_is_external_api(self):
        """source_type should be EXTERNAL_API."""
        connector = GitHubConnector(repo="owner/repo")
        assert connector.source_type == SourceType.EXTERNAL_API

    def test_name_is_github(self):
        """name should be 'GitHub'."""
        connector = GitHubConnector(repo="owner/repo")
        assert connector.name == "GitHub"

    def test_gh_cli_flag(self):
        """use_gh_cli flag should be configurable."""
        connector = GitHubConnector(repo="owner/repo", use_gh_cli=False)
        assert connector.use_gh_cli is False


class TestQueryValidation:
    """Tests for query input validation."""

    @pytest.fixture
    def connector(self):
        return GitHubConnector(repo="owner/repo")

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self, connector):
        """Empty query should return empty list."""
        with patch.object(connector, "_run_gh", return_value=None):
            results = await connector.search("")
            assert results == []

    @pytest.mark.asyncio
    async def test_short_query_returns_empty(self, connector):
        """Query with < 2 chars should return empty list."""
        with patch.object(connector, "_run_gh", return_value=None):
            results = await connector.search("a")
            assert results == []

    @pytest.mark.asyncio
    async def test_long_query_returns_empty(self, connector):
        """Query exceeding MAX_QUERY_LENGTH should return empty list."""
        long_query = "x" * (MAX_QUERY_LENGTH + 1)
        with patch.object(connector, "_run_gh", return_value=None):
            results = await connector.search(long_query)
            assert results == []

    @pytest.mark.asyncio
    async def test_valid_query_length_accepted(self, connector):
        """Query within limits should be processed."""
        with patch.object(connector, "_run_gh", return_value="[]"):
            results = await connector.search("valid query")
            assert results == []  # Empty but processed


class TestGhCliIntegration:
    """Tests for gh CLI integration."""

    @pytest.fixture
    def connector(self):
        return GitHubConnector(repo="owner/repo")

    def test_check_gh_cli_caches_result(self, connector):
        """_check_gh_cli should cache result after first call."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0)

            # First call
            result1 = connector._check_gh_cli()
            # Second call (should use cache)
            result2 = connector._check_gh_cli()

            assert result1 is True
            assert result2 is True
            # Only called once due to caching
            assert mock_run.call_count == 1

    def test_check_gh_cli_handles_failure(self, connector):
        """_check_gh_cli should return False on failure."""
        connector._gh_available = None  # Reset cache
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=1)
            result = connector._check_gh_cli()
            assert result is False

    def test_check_gh_cli_handles_exception(self, connector):
        """_check_gh_cli should return False on exception."""
        connector._gh_available = None  # Reset cache
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = connector._check_gh_cli()
            assert result is False

    @pytest.mark.asyncio
    async def test_run_gh_returns_none_when_unavailable(self, connector):
        """_run_gh should return None when gh CLI unavailable."""
        connector._gh_available = False
        result = await connector._run_gh(["issue", "list"])
        assert result is None

    @pytest.mark.asyncio
    async def test_search_without_repo_for_issues_returns_empty(self):
        """search for issues without repo should return empty."""
        connector = GitHubConnector()  # No repo
        results = await connector.search("test", search_type="issues")
        assert results == []


class TestIssueSearch:
    """Tests for issue search functionality."""

    @pytest.fixture
    def connector(self):
        return GitHubConnector(repo="owner/repo")

    @pytest.fixture
    def mock_issues_json(self):
        """Sample gh CLI output for issues."""
        return json.dumps(
            [
                {
                    "number": 1,
                    "title": "Bug Report",
                    "body": "This is a bug",
                    "author": {"login": "user1"},
                    "createdAt": "2026-01-01T00:00:00Z",
                    "url": "https://github.com/owner/repo/issues/1",
                    "state": "open",
                    "labels": [{"name": "bug"}],
                },
                {
                    "number": 2,
                    "title": "Feature Request",
                    "body": "Please add feature",
                    "author": {"login": "user2"},
                    "createdAt": "2026-01-02T00:00:00Z",
                    "url": "https://github.com/owner/repo/issues/2",
                    "state": "open",
                    "labels": [],
                },
            ]
        )

    @pytest.mark.asyncio
    async def test_search_issues_parses_json(self, connector, mock_issues_json):
        """search issues should parse gh CLI JSON output."""
        with patch.object(connector, "_run_gh", return_value=mock_issues_json):
            results = await connector.search("test", search_type="issues")

            assert len(results) == 2
            assert results[0].title == "Issue #1: Bug Report"
            assert results[0].metadata["type"] == "issue"
            assert results[0].metadata["number"] == 1

    @pytest.mark.asyncio
    async def test_search_issues_bug_label_increases_authority(self, connector, mock_issues_json):
        """Issues with 'bug' label should have higher authority."""
        with patch.object(connector, "_run_gh", return_value=mock_issues_json):
            results = await connector.search("test", search_type="issues")

            # First issue has "bug" label
            assert results[0].authority == 0.8
            # Second issue has no special labels
            assert results[1].authority == 0.6

    @pytest.mark.asyncio
    async def test_search_issues_handles_invalid_json(self, connector):
        """search should handle invalid JSON gracefully."""
        with patch.object(connector, "_run_gh", return_value="not json"):
            results = await connector.search("test", search_type="issues")
            assert results == []

    @pytest.mark.asyncio
    async def test_search_issues_handles_none_output(self, connector):
        """search should handle None output (gh failure)."""
        with patch.object(connector, "_run_gh", return_value=None):
            results = await connector.search("test", search_type="issues")
            assert results == []


class TestPRSearch:
    """Tests for PR search functionality."""

    @pytest.fixture
    def connector(self):
        return GitHubConnector(repo="owner/repo")

    @pytest.fixture
    def mock_prs_json(self):
        """Sample gh CLI output for PRs."""
        return json.dumps(
            [
                {
                    "number": 10,
                    "title": "Add feature",
                    "body": "This PR adds a feature",
                    "author": {"login": "dev1"},
                    "createdAt": "2026-01-01T00:00:00Z",
                    "url": "https://github.com/owner/repo/pull/10",
                    "state": "open",
                    "mergedAt": None,
                },
                {
                    "number": 11,
                    "title": "Fix bug",
                    "body": "This PR fixes a bug",
                    "author": {"login": "dev2"},
                    "createdAt": "2026-01-02T00:00:00Z",
                    "url": "https://github.com/owner/repo/pull/11",
                    "state": "merged",
                    "mergedAt": "2026-01-03T00:00:00Z",
                },
            ]
        )

    @pytest.mark.asyncio
    async def test_search_prs_parses_json(self, connector, mock_prs_json):
        """search prs should parse gh CLI JSON output."""
        with patch.object(connector, "_run_gh", return_value=mock_prs_json):
            results = await connector.search("test", search_type="prs")

            assert len(results) == 2
            assert results[0].title == "PR #10: Add feature"
            assert results[0].metadata["type"] == "pr"

    @pytest.mark.asyncio
    async def test_search_prs_merged_has_higher_authority(self, connector, mock_prs_json):
        """Merged PRs should have higher authority."""
        with patch.object(connector, "_run_gh", return_value=mock_prs_json):
            results = await connector.search("test", search_type="prs")

            # First PR is not merged
            assert results[0].authority == 0.6
            assert results[0].metadata["merged"] is False

            # Second PR is merged
            assert results[1].authority == 0.8
            assert results[1].metadata["merged"] is True


class TestCodeSearch:
    """Tests for code search functionality."""

    @pytest.fixture
    def connector(self):
        return GitHubConnector(repo="owner/repo")

    @pytest.fixture
    def mock_code_json(self):
        """Sample gh CLI output for code search."""
        return json.dumps(
            [
                {
                    "path": "src/main.py",
                    "repository": {"fullName": "owner/repo"},
                    "textMatches": [
                        {"fragment": "def hello():\n    print('world')"},
                    ],
                },
            ]
        )

    @pytest.mark.asyncio
    async def test_search_code_parses_json(self, connector, mock_code_json):
        """search code should parse gh CLI JSON output."""
        with patch.object(connector, "_run_gh", return_value=mock_code_json):
            results = await connector.search("hello", search_type="code")

            assert len(results) == 1
            assert "src/main.py" in results[0].title
            assert results[0].metadata["type"] == "code"

    @pytest.mark.asyncio
    async def test_search_code_without_repo(self, mock_code_json):
        """Code search should work without repo (global search)."""
        connector = GitHubConnector()  # No repo
        with patch.object(connector, "_run_gh", return_value=mock_code_json):
            results = await connector.search("hello", search_type="code")
            assert len(results) == 1


class TestFetchOperations:
    """Tests for fetch operations."""

    @pytest.fixture
    def connector(self):
        return GitHubConnector(repo="owner/repo")

    @pytest.fixture
    def mock_issue_detail(self):
        """Sample gh CLI output for single issue."""
        return json.dumps(
            {
                "number": 1,
                "title": "Bug Report",
                "body": "Detailed bug description",
                "author": {"login": "user1"},
                "createdAt": "2026-01-01T00:00:00Z",
                "url": "https://github.com/owner/repo/issues/1",
                "state": "open",
                "labels": [],
                "comments": [
                    {"author": {"login": "dev1"}, "body": "Looking into this"},
                ],
            }
        )

    @pytest.fixture
    def mock_pr_detail(self):
        """Sample gh CLI output for single PR."""
        return json.dumps(
            {
                "number": 10,
                "title": "Fix Bug",
                "body": "This fixes the bug",
                "author": {"login": "dev1"},
                "createdAt": "2026-01-01T00:00:00Z",
                "url": "https://github.com/owner/repo/pull/10",
                "state": "open",
                "mergedAt": None,
                "reviews": [
                    {"author": {"login": "reviewer"}, "state": "APPROVED", "body": "LGTM"},
                ],
            }
        )

    @pytest.mark.asyncio
    async def test_fetch_returns_cached(self, connector):
        """fetch should return cached evidence if available."""
        import time

        cached_evidence = Evidence(
            id="gh-issue:owner/repo:1",
            source_type=SourceType.EXTERNAL_API,
            source_id="github/owner/repo/issues/1",
            content="Cached content",
        )
        # Cache stores (timestamp, evidence) tuples
        connector._cache["gh-issue:owner/repo:1"] = (time.time(), cached_evidence)

        result = await connector.fetch("gh-issue:owner/repo:1")

        assert result == cached_evidence

    @pytest.mark.asyncio
    async def test_fetch_issue_includes_comments(self, connector, mock_issue_detail):
        """fetch issue should include comments in content."""
        with patch.object(connector, "_run_gh", return_value=mock_issue_detail):
            result = await connector._fetch_issue("owner/repo", "1")

            assert result is not None
            assert "Comments" in result.content
            assert "Looking into this" in result.content
            assert result.metadata["comment_count"] == 1

    @pytest.mark.asyncio
    async def test_fetch_pr_includes_reviews(self, connector, mock_pr_detail):
        """fetch PR should include reviews in content."""
        with patch.object(connector, "_run_gh", return_value=mock_pr_detail):
            result = await connector._fetch_pr("owner/repo", "10")

            assert result is not None
            assert "Reviews" in result.content
            assert "LGTM" in result.content
            assert result.metadata["review_count"] == 1

    @pytest.mark.asyncio
    async def test_fetch_parses_issue_id(self, connector, mock_issue_detail):
        """fetch should parse gh-issue: evidence ID."""
        with patch.object(connector, "_fetch_issue", return_value=Mock()) as mock_fetch:
            await connector.fetch("gh-issue:owner/repo:123")
            mock_fetch.assert_called_once_with("owner/repo", "123")

    @pytest.mark.asyncio
    async def test_fetch_parses_pr_id(self, connector, mock_pr_detail):
        """fetch should parse gh-pr: evidence ID."""
        with patch.object(connector, "_fetch_pr", return_value=Mock()) as mock_fetch:
            await connector.fetch("gh-pr:owner/repo:456")
            mock_fetch.assert_called_once_with("owner/repo", "456")

    @pytest.mark.asyncio
    async def test_fetch_unknown_id_returns_none(self, connector):
        """fetch with unknown ID format should return None."""
        result = await connector.fetch("unknown:format:123")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_malformed_id_returns_none(self, connector):
        """fetch with malformed ID should return None."""
        result = await connector.fetch("gh-issue:incomplete")
        assert result is None


class TestSearchTypeRouting:
    """Tests for search type routing."""

    @pytest.fixture
    def connector(self):
        return GitHubConnector(repo="owner/repo")

    @pytest.mark.asyncio
    async def test_search_routes_to_issues(self, connector):
        """search_type='issues' should call _search_issues."""
        with patch.object(connector, "_search_issues", return_value=[]) as mock:
            await connector.search("test", search_type="issues")
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_routes_to_prs(self, connector):
        """search_type='prs' should call _search_prs."""
        with patch.object(connector, "_search_prs", return_value=[]) as mock:
            await connector.search("test", search_type="prs")
            mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_routes_to_code(self, connector):
        """search_type='code' should call _search_code."""
        with patch.object(connector, "_search_code", return_value=[]) as mock:
            await connector.search("test", search_type="code")
            mock.assert_called_once()


class TestInputSanitization:
    """Tests for input sanitization - SECURITY CRITICAL."""

    def test_validate_number_valid(self):
        """Valid issue/PR numbers should pass."""
        connector = GitHubConnector(repo="owner/repo")
        assert connector._validate_number("1") is True
        assert connector._validate_number("123") is True
        assert connector._validate_number("999999999") is True

    def test_validate_number_rejects_non_numeric(self):
        """Non-numeric values should be rejected."""
        connector = GitHubConnector(repo="owner/repo")
        assert connector._validate_number("abc") is False
        assert connector._validate_number("123abc") is False
        assert connector._validate_number("12.34") is False
        assert connector._validate_number("-1") is False

    def test_validate_number_rejects_injection(self):
        """Injection attempts should be rejected."""
        connector = GitHubConnector(repo="owner/repo")
        assert connector._validate_number("123;rm -rf /") is False
        assert connector._validate_number("123 && whoami") is False
        assert connector._validate_number("$(cat /etc/passwd)") is False
        assert connector._validate_number("1`id`") is False

    def test_validate_number_rejects_empty(self):
        """Empty/null values should be rejected."""
        connector = GitHubConnector(repo="owner/repo")
        assert connector._validate_number("") is False
        assert connector._validate_number(None) is False

    def test_validate_number_rejects_too_long(self):
        """Excessively long numbers should be rejected."""
        connector = GitHubConnector(repo="owner/repo")
        # Pattern allows max 10 digits
        assert connector._validate_number("12345678901") is False

    def test_validate_repo_valid(self):
        """Valid repo formats should pass."""
        assert GitHubConnector._validate_repo("owner/repo") is True
        assert GitHubConnector._validate_repo("my-org/my-project") is True
        assert GitHubConnector._validate_repo("user.name/repo_name") is True

    def test_validate_repo_rejects_injection(self):
        """Path traversal and injection should be rejected."""
        assert GitHubConnector._validate_repo("../../../etc/passwd") is False
        assert GitHubConnector._validate_repo("owner/repo;rm -rf /") is False
        assert GitHubConnector._validate_repo("owner/repo && whoami") is False

    def test_validate_state_normalizes(self):
        """State values should be normalized."""
        connector = GitHubConnector(repo="owner/repo")
        assert connector._validate_state("OPEN") == "open"
        assert connector._validate_state("  closed  ") == "closed"
        assert connector._validate_state("All") == "all"

    def test_validate_state_allows_valid(self):
        """Valid states should be allowed."""
        connector = GitHubConnector(repo="owner/repo")
        assert connector._validate_state("open") == "open"
        assert connector._validate_state("closed") == "closed"
        assert connector._validate_state("merged") == "merged"
        assert connector._validate_state("all") == "all"

    def test_validate_state_rejects_invalid(self):
        """Invalid states should default to 'all'."""
        connector = GitHubConnector(repo="owner/repo")
        assert connector._validate_state("invalid") == "all"
        assert connector._validate_state("; rm -rf /") == "all"
        assert connector._validate_state("open && whoami") == "all"

    def test_validate_state_handles_empty(self):
        """Empty/null states should default to 'all'."""
        connector = GitHubConnector(repo="owner/repo")
        assert connector._validate_state("") == "all"
        assert connector._validate_state(None) == "all"


class TestFetchInputValidation:
    """Tests for fetch() input validation - SECURITY CRITICAL."""

    @pytest.fixture
    def connector(self):
        return GitHubConnector(repo="owner/repo")

    @pytest.mark.asyncio
    async def test_fetch_rejects_invalid_repo_in_issue_id(self, connector):
        """fetch should reject issue IDs with invalid repo format."""
        result = await connector.fetch("gh-issue:../../../etc/passwd:123")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_rejects_invalid_number_in_issue_id(self, connector):
        """fetch should reject issue IDs with non-numeric issue number."""
        result = await connector.fetch("gh-issue:owner/repo:abc")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_rejects_injection_in_issue_number(self, connector):
        """fetch should reject injection attempts in issue number."""
        result = await connector.fetch("gh-issue:owner/repo:123;rm -rf /")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_rejects_invalid_repo_in_pr_id(self, connector):
        """fetch should reject PR IDs with invalid repo format."""
        result = await connector.fetch("gh-pr:owner/repo;whoami:123")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_rejects_invalid_number_in_pr_id(self, connector):
        """fetch should reject PR IDs with non-numeric PR number."""
        result = await connector.fetch("gh-pr:owner/repo:$(id)")
        assert result is None

    @pytest.mark.asyncio
    async def test_fetch_accepts_valid_issue_id(self, connector):
        """fetch should accept valid issue IDs and call _fetch_issue."""
        with patch.object(connector, "_fetch_issue", return_value=Mock()) as mock:
            await connector.fetch("gh-issue:valid/repo:12345")
            mock.assert_called_once_with("valid/repo", "12345")

    @pytest.mark.asyncio
    async def test_fetch_accepts_valid_pr_id(self, connector):
        """fetch should accept valid PR IDs and call _fetch_pr."""
        with patch.object(connector, "_fetch_pr", return_value=Mock()) as mock:
            await connector.fetch("gh-pr:another/repo:67890")
            mock.assert_called_once_with("another/repo", "67890")


class TestLimitCapping:
    """Tests for result limit capping."""

    @pytest.fixture
    def connector(self):
        return GitHubConnector(repo="owner/repo")

    @pytest.mark.asyncio
    async def test_search_issues_caps_limit(self, connector):
        """_search_issues should cap limit at 100."""
        with patch.object(connector, "_run_gh", return_value="[]") as mock:
            await connector._search_issues("test", limit=500, state="all")
            call_args = mock.call_args[0][0]
            # Find the --limit argument
            limit_idx = call_args.index("--limit")
            assert call_args[limit_idx + 1] == "100"

    @pytest.mark.asyncio
    async def test_search_prs_caps_limit(self, connector):
        """_search_prs should cap limit at 100."""
        with patch.object(connector, "_run_gh", return_value="[]") as mock:
            await connector._search_prs("test", limit=1000, state="all")
            call_args = mock.call_args[0][0]
            limit_idx = call_args.index("--limit")
            assert call_args[limit_idx + 1] == "100"

    @pytest.mark.asyncio
    async def test_search_code_caps_limit(self, connector):
        """_search_code should cap limit at 100."""
        with patch.object(connector, "_run_gh", return_value="[]") as mock:
            await connector._search_code("test", limit=200)
            call_args = mock.call_args[0][0]
            limit_idx = call_args.index("--limit")
            assert call_args[limit_idx + 1] == "100"

    @pytest.mark.asyncio
    async def test_small_limit_not_changed(self, connector):
        """Small limits should not be changed."""
        with patch.object(connector, "_run_gh", return_value="[]") as mock:
            await connector._search_issues("test", limit=10, state="all")
            call_args = mock.call_args[0][0]
            limit_idx = call_args.index("--limit")
            assert call_args[limit_idx + 1] == "10"
