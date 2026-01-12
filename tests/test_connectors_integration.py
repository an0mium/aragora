"""
Integration tests for external connectors.

Tests GitHub, Twitter, and YouTube connectors with mocked external APIs.
Covers: validation, OAuth flows, rate limiting, circuit breakers, error handling.
"""

import asyncio
import base64
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.base import BaseConnector, Evidence
from aragora.reasoning.provenance import SourceType


# =============================================================================
# GitHub Connector Tests
# =============================================================================

class TestGitHubConnectorValidation:
    """Tests for GitHubConnector input validation."""

    def test_valid_repo_pattern(self):
        """Test valid repository patterns."""
        from aragora.connectors.github import VALID_REPO_PATTERN

        # Valid patterns
        assert VALID_REPO_PATTERN.match("owner/repo")
        assert VALID_REPO_PATTERN.match("org-name/my-project")
        assert VALID_REPO_PATTERN.match("user_123/repo_v2")
        assert VALID_REPO_PATTERN.match("A.B/C.D")

    def test_invalid_repo_pattern(self):
        """Test invalid repository patterns."""
        from aragora.connectors.github import VALID_REPO_PATTERN

        # Invalid patterns
        assert not VALID_REPO_PATTERN.match("owner")
        assert not VALID_REPO_PATTERN.match("owner/")
        assert not VALID_REPO_PATTERN.match("/repo")
        assert not VALID_REPO_PATTERN.match("owner/repo/extra")
        assert not VALID_REPO_PATTERN.match("")
        assert not VALID_REPO_PATTERN.match("owner;rm -rf /repo")

    def test_connector_rejects_invalid_repo(self):
        """Test connector raises error for invalid repo format."""
        from aragora.connectors.github import GitHubConnector

        with pytest.raises(ValueError, match="Invalid repo format"):
            GitHubConnector(repo="invalid")

        with pytest.raises(ValueError, match="Invalid repo format"):
            GitHubConnector(repo="owner; rm -rf /")

    def test_connector_accepts_valid_repo(self):
        """Test connector accepts valid repo format."""
        from aragora.connectors.github import GitHubConnector

        connector = GitHubConnector(repo="owner/repo")
        assert connector.repo == "owner/repo"

    def test_connector_allows_no_repo(self):
        """Test connector can be created without repo for code search."""
        from aragora.connectors.github import GitHubConnector

        connector = GitHubConnector()
        assert connector.repo is None

    def test_valid_number_pattern(self):
        """Test issue/PR number validation."""
        from aragora.connectors.github import VALID_NUMBER_PATTERN

        assert VALID_NUMBER_PATTERN.match("1")
        assert VALID_NUMBER_PATTERN.match("123")
        assert VALID_NUMBER_PATTERN.match("1234567890")
        assert not VALID_NUMBER_PATTERN.match("")
        assert not VALID_NUMBER_PATTERN.match("abc")
        assert not VALID_NUMBER_PATTERN.match("12345678901")  # Too long

    def test_state_validation(self):
        """Test state parameter validation."""
        from aragora.connectors.github import GitHubConnector

        connector = GitHubConnector(repo="owner/repo")

        assert connector._validate_state("open") == "open"
        assert connector._validate_state("CLOSED") == "closed"
        assert connector._validate_state("all") == "all"
        assert connector._validate_state("invalid") == "all"
        assert connector._validate_state("") == "all"
        assert connector._validate_state(None) == "all"


class TestGitHubConnectorSearch:
    """Tests for GitHubConnector search functionality."""

    @pytest.fixture
    def connector(self):
        """Create a GitHub connector for testing."""
        from aragora.connectors.github import GitHubConnector
        return GitHubConnector(repo="owner/repo", use_gh_cli=True)

    @pytest.mark.asyncio
    async def test_search_requires_query(self, connector):
        """Test search returns empty for empty query."""
        results = await connector.search("", limit=10)
        assert results == []

        results = await connector.search("x", limit=10)  # Too short
        assert results == []

    @pytest.mark.asyncio
    async def test_search_rejects_long_query(self, connector):
        """Test search rejects excessively long queries."""
        long_query = "x" * 600
        results = await connector.search(long_query, limit=10)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_issues_without_repo(self):
        """Test issue search without repo returns empty."""
        from aragora.connectors.github import GitHubConnector

        connector = GitHubConnector()  # No repo
        results = await connector.search("test query", search_type="issues")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_issues_with_gh_cli(self, connector):
        """Test issue search via mocked gh CLI."""
        mock_output = json.dumps([
            {
                "number": 123,
                "title": "Test Issue",
                "body": "Issue body content",
                "author": {"login": "testuser"},
                "createdAt": "2024-01-01T12:00:00Z",
                "url": "https://github.com/owner/repo/issues/123",
                "state": "open",
                "labels": [{"name": "bug"}],
            }
        ])

        with patch.object(connector, "_check_gh_cli", return_value=True):
            with patch.object(connector, "_run_gh", new_callable=AsyncMock, return_value=mock_output):
                results = await connector.search("test query", search_type="issues", limit=5)

                assert len(results) == 1
                assert results[0].id == "gh-issue:owner/repo:123"
                assert results[0].title == "Issue #123: Test Issue"
                assert results[0].metadata["type"] == "issue"
                assert results[0].metadata["state"] == "open"
                assert "bug" in results[0].metadata["labels"]

    @pytest.mark.asyncio
    async def test_search_prs_with_merged(self, connector):
        """Test PR search includes merge status."""
        mock_output = json.dumps([
            {
                "number": 456,
                "title": "Test PR",
                "body": "PR description",
                "author": {"login": "contributor"},
                "createdAt": "2024-01-15T10:00:00Z",
                "url": "https://github.com/owner/repo/pull/456",
                "state": "merged",
                "mergedAt": "2024-01-16T12:00:00Z",
            }
        ])

        with patch.object(connector, "_check_gh_cli", return_value=True):
            with patch.object(connector, "_run_gh", new_callable=AsyncMock, return_value=mock_output):
                results = await connector.search("test", search_type="prs")

                assert len(results) == 1
                assert results[0].id == "gh-pr:owner/repo:456"
                assert results[0].metadata["merged"] is True
                assert results[0].authority == 0.8  # Merged PRs have higher authority

    @pytest.mark.asyncio
    async def test_search_code_global(self):
        """Test code search works without repo."""
        from aragora.connectors.github import GitHubConnector

        connector = GitHubConnector()  # No repo for global search

        mock_output = json.dumps([
            {
                "path": "src/main.py",
                "repository": {"fullName": "other/repo"},
                "textMatches": [{"fragment": "def test_function():"}],
            }
        ])

        with patch.object(connector, "_check_gh_cli", return_value=True):
            with patch.object(connector, "_run_gh", new_callable=AsyncMock, return_value=mock_output):
                results = await connector.search("test_function", search_type="code")

                assert len(results) == 1
                assert results[0].source_type == SourceType.CODE_ANALYSIS
                assert results[0].metadata["repository"] == "other/repo"
                assert results[0].metadata["path"] == "src/main.py"

    @pytest.mark.asyncio
    async def test_search_gh_cli_not_available(self, connector):
        """Test search returns empty when gh CLI not available."""
        with patch.object(connector, "_check_gh_cli", return_value=False):
            results = await connector.search("test query", search_type="issues")
            assert results == []

    @pytest.mark.asyncio
    async def test_search_handles_json_error(self, connector):
        """Test search handles malformed JSON gracefully."""
        with patch.object(connector, "_check_gh_cli", return_value=True):
            with patch.object(connector, "_run_gh", new_callable=AsyncMock, return_value="not json"):
                results = await connector.search("test", search_type="issues")
                assert results == []


class TestGitHubConnectorFetch:
    """Tests for GitHubConnector fetch functionality."""

    @pytest.fixture
    def connector(self):
        from aragora.connectors.github import GitHubConnector
        return GitHubConnector(repo="owner/repo")

    @pytest.mark.asyncio
    async def test_fetch_issue_by_id(self, connector):
        """Test fetching specific issue by evidence ID."""
        mock_output = json.dumps({
            "number": 123,
            "title": "Fetched Issue",
            "body": "Full issue body",
            "author": {"login": "author"},
            "createdAt": "2024-01-01T12:00:00Z",
            "url": "https://github.com/owner/repo/issues/123",
            "state": "open",
            "labels": [],
            "comments": [
                {"author": {"login": "commenter"}, "body": "Comment text"}
            ],
        })

        with patch.object(connector, "_check_gh_cli", return_value=True):
            with patch.object(connector, "_run_gh", new_callable=AsyncMock, return_value=mock_output):
                evidence = await connector.fetch("gh-issue:owner/repo:123")

                assert evidence is not None
                assert evidence.id == "gh-issue:owner/repo:123"
                assert "Comments" in evidence.content
                assert evidence.metadata["comment_count"] == 1

    @pytest.mark.asyncio
    async def test_fetch_pr_by_id(self, connector):
        """Test fetching specific PR by evidence ID."""
        mock_output = json.dumps({
            "number": 456,
            "title": "Fetched PR",
            "body": "PR body",
            "author": {"login": "author"},
            "createdAt": "2024-01-01T12:00:00Z",
            "url": "https://github.com/owner/repo/pull/456",
            "state": "merged",
            "mergedAt": "2024-01-02T10:00:00Z",
            "reviews": [
                {"author": {"login": "reviewer"}, "state": "APPROVED", "body": "LGTM"}
            ],
        })

        with patch.object(connector, "_check_gh_cli", return_value=True):
            with patch.object(connector, "_run_gh", new_callable=AsyncMock, return_value=mock_output):
                evidence = await connector.fetch("gh-pr:owner/repo:456")

                assert evidence is not None
                assert evidence.metadata["merged"] is True
                assert evidence.metadata["review_count"] == 1
                assert evidence.authority == 0.85  # Merged PR

    @pytest.mark.asyncio
    async def test_fetch_invalid_id_format(self, connector):
        """Test fetch with invalid evidence ID."""
        evidence = await connector.fetch("invalid-id")
        assert evidence is None

    @pytest.mark.asyncio
    async def test_fetch_rejects_injection(self, connector):
        """Test fetch rejects malicious evidence IDs."""
        # Attempt injection via repo
        with patch.object(connector, "_check_gh_cli", return_value=True):
            evidence = await connector.fetch("gh-issue:owner;rm -rf /:123")
            assert evidence is None

        # Attempt injection via number
        with patch.object(connector, "_check_gh_cli", return_value=True):
            evidence = await connector.fetch("gh-issue:owner/repo:123;whoami")
            assert evidence is None

    @pytest.mark.asyncio
    async def test_fetch_uses_cache(self, connector):
        """Test fetch uses cache for repeated calls."""
        mock_output = json.dumps({
            "number": 123,
            "title": "Cached Issue",
            "body": "Body",
            "author": {"login": "author"},
            "createdAt": "2024-01-01T12:00:00Z",
            "url": "https://github.com/owner/repo/issues/123",
            "state": "open",
            "labels": [],
            "comments": [],
        })

        with patch.object(connector, "_check_gh_cli", return_value=True):
            with patch.object(connector, "_run_gh", new_callable=AsyncMock, return_value=mock_output) as mock_run:
                # First call
                evidence1 = await connector.fetch("gh-issue:owner/repo:123")
                # Second call should use cache
                evidence2 = await connector.fetch("gh-issue:owner/repo:123")

                assert evidence1 is not None
                assert evidence2 is not None
                assert mock_run.call_count == 1  # Only called once due to cache


class TestGitHubConnectorCLI:
    """Tests for GitHubConnector CLI integration."""

    def test_check_gh_cli_available(self):
        """Test gh CLI availability check."""
        from aragora.connectors.github import GitHubConnector

        connector = GitHubConnector(repo="owner/repo")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert connector._check_gh_cli() is True

    def test_check_gh_cli_not_available(self):
        """Test gh CLI not available."""
        from aragora.connectors.github import GitHubConnector

        connector = GitHubConnector(repo="owner/repo")

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()
            assert connector._check_gh_cli() is False

    def test_check_gh_cli_cached(self):
        """Test gh CLI check is cached."""
        from aragora.connectors.github import GitHubConnector

        connector = GitHubConnector(repo="owner/repo")
        connector._gh_available = True

        # Should not call subprocess since cached
        with patch("subprocess.run") as mock_run:
            assert connector._check_gh_cli() is True
            mock_run.assert_not_called()


# =============================================================================
# Twitter Connector Tests
# =============================================================================

class TestTwitterConnectorConfiguration:
    """Tests for TwitterPosterConnector configuration."""

    def test_is_configured_with_all_credentials(self):
        """Test is_configured returns True with all credentials."""
        from aragora.connectors.twitter_poster import TwitterPosterConnector

        connector = TwitterPosterConnector(
            api_key="key",
            api_secret="secret",
            access_token="token",
            access_secret="access_secret",
        )
        assert connector.is_configured is True

    def test_is_configured_missing_credentials(self):
        """Test is_configured returns False with missing credentials."""
        from aragora.connectors.twitter_poster import TwitterPosterConnector

        connector = TwitterPosterConnector(api_key="key")  # Missing others
        assert connector.is_configured is False

    def test_loads_from_environment(self):
        """Test connector loads credentials from environment."""
        from aragora.connectors.twitter_poster import TwitterPosterConnector

        with patch.dict(os.environ, {
            "TWITTER_API_KEY": "env_key",
            "TWITTER_API_SECRET": "env_secret",
            "TWITTER_ACCESS_TOKEN": "env_token",
            "TWITTER_ACCESS_SECRET": "env_access",
        }):
            connector = TwitterPosterConnector()
            assert connector.api_key == "env_key"
            assert connector.api_secret == "env_secret"


class TestTwitterConnectorOAuth:
    """Tests for TwitterPosterConnector OAuth signature generation."""

    @pytest.fixture
    def connector(self):
        from aragora.connectors.twitter_poster import TwitterPosterConnector
        return TwitterPosterConnector(
            api_key="test_key",
            api_secret="test_secret",
            access_token="test_token",
            access_secret="test_access_secret",
        )

    def test_generate_oauth_signature(self, connector):
        """Test OAuth signature generation produces consistent output."""
        # Signature depends on timestamp/nonce so we test it's non-empty
        signature = connector._generate_oauth_signature(
            method="POST",
            url="https://api.twitter.com/2/tweets",
            params={},
            oauth_params={
                "oauth_consumer_key": "test_key",
                "oauth_token": "test_token",
                "oauth_signature_method": "HMAC-SHA1",
                "oauth_timestamp": "1234567890",
                "oauth_nonce": "abc123",
                "oauth_version": "1.0",
            },
        )
        assert signature  # Non-empty
        assert isinstance(signature, str)

    def test_generate_oauth_header(self, connector):
        """Test OAuth header generation."""
        header = connector._generate_oauth_header(
            method="POST",
            url="https://api.twitter.com/2/tweets",
        )
        assert header.startswith("OAuth ")
        assert "oauth_consumer_key" in header
        assert "oauth_signature" in header


class TestTwitterConnectorPosting:
    """Tests for TwitterPosterConnector tweet posting."""

    @pytest.fixture
    def connector(self):
        from aragora.connectors.twitter_poster import TwitterPosterConnector
        return TwitterPosterConnector(
            api_key="test_key",
            api_secret="test_secret",
            access_token="test_token",
            access_secret="test_access_secret",
        )

    @pytest.mark.asyncio
    async def test_post_tweet_not_configured(self):
        """Test posting fails gracefully when not configured."""
        from aragora.connectors.twitter_poster import TwitterPosterConnector

        connector = TwitterPosterConnector()  # No credentials
        result = await connector.post_tweet("Test tweet")

        assert result.success is False
        assert "not configured" in result.error

    @pytest.mark.asyncio
    async def test_post_tweet_success(self, connector):
        """Test successful tweet posting."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"data": {"id": "1234567890"}}

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            result = await connector.post_tweet("Test tweet")

            assert result.success is True
            assert result.tweet_id == "1234567890"
            assert result.url == "https://twitter.com/i/status/1234567890"

    @pytest.mark.asyncio
    async def test_post_tweet_truncates_long_text(self, connector):
        """Test tweet text is truncated if too long."""
        long_text = "x" * 300  # Over 280 limit

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"data": {"id": "123"}}

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            result = await connector.post_tweet(long_text)

            # Text should be truncated
            assert len(result.text) <= 280
            assert result.text.endswith("...")

    @pytest.mark.asyncio
    async def test_post_tweet_api_error(self, connector):
        """Test handling of API error response."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            result = await connector.post_tweet("Test tweet")

            assert result.success is False
            assert "403" in result.error

    @pytest.mark.asyncio
    async def test_post_tweet_network_error(self, connector):
        """Test handling of network errors."""
        import httpx

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.side_effect = httpx.RequestError("Network error")
            MockClient.return_value.__aenter__.return_value = mock_client

            result = await connector.post_tweet("Test tweet")

            assert result.success is False
            assert "Network error" in result.error

    @pytest.mark.asyncio
    async def test_post_tweet_circuit_breaker(self, connector):
        """Test circuit breaker opens after failures."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Server error"

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            # Exhaust circuit breaker
            for _ in range(6):  # threshold is 5
                await connector.post_tweet("Test tweet")

            # Circuit should be open
            result = await connector.post_tweet("Another tweet")
            assert result.success is False
            assert "Circuit breaker" in result.error


class TestTwitterConnectorThread:
    """Tests for TwitterPosterConnector thread posting."""

    @pytest.fixture
    def connector(self):
        from aragora.connectors.twitter_poster import TwitterPosterConnector
        return TwitterPosterConnector(
            api_key="test_key",
            api_secret="test_secret",
            access_token="test_token",
            access_secret="test_access_secret",
        )

    @pytest.mark.asyncio
    async def test_post_thread_empty(self, connector):
        """Test posting empty thread fails."""
        result = await connector.post_thread([])
        assert result.success is False
        assert "No tweets" in result.error

    @pytest.mark.asyncio
    async def test_post_thread_success(self, connector):
        """Test successful thread posting."""
        tweet_ids = ["111", "222", "333"]
        call_count = [0]

        async def mock_post_tweet(text, reply_to=None, media_ids=None):
            from aragora.connectors.twitter_poster import TweetResult
            tweet_id = tweet_ids[call_count[0]]
            call_count[0] += 1
            return TweetResult(
                tweet_id=tweet_id,
                text=text,
                created_at=datetime.now().isoformat(),
                url=f"https://twitter.com/i/status/{tweet_id}",
                success=True,
            )

        with patch.object(connector, "post_tweet", side_effect=mock_post_tweet):
            result = await connector.post_thread(["Tweet 1", "Tweet 2", "Tweet 3"])

            assert result.success is True
            assert result.thread_id == "111"
            assert len(result.tweets) == 3
            assert result.url == "https://twitter.com/i/status/111"

    @pytest.mark.asyncio
    async def test_post_thread_partial_failure(self, connector):
        """Test thread fails gracefully on partial failure."""
        call_count = [0]

        async def mock_post_tweet(text, reply_to=None, media_ids=None):
            from aragora.connectors.twitter_poster import TweetResult
            call_count[0] += 1
            if call_count[0] == 2:
                return TweetResult(
                    tweet_id="",
                    text=text,
                    created_at=datetime.now().isoformat(),
                    url="",
                    success=False,
                    error="Rate limited",
                )
            return TweetResult(
                tweet_id=str(call_count[0]),
                text=text,
                created_at=datetime.now().isoformat(),
                url=f"https://twitter.com/i/status/{call_count[0]}",
                success=True,
            )

        with patch.object(connector, "post_tweet", side_effect=mock_post_tweet):
            result = await connector.post_thread(["Tweet 1", "Tweet 2", "Tweet 3"])

            assert result.success is False
            assert "Failed at tweet 2" in result.error
            assert len(result.tweets) == 2  # First success + the failure

    @pytest.mark.asyncio
    async def test_post_thread_truncates_long_threads(self, connector):
        """Test threads are truncated to max length."""
        from aragora.connectors.twitter_poster import MAX_THREAD_LENGTH

        async def mock_post_tweet(text, reply_to=None, media_ids=None):
            from aragora.connectors.twitter_poster import TweetResult
            return TweetResult(
                tweet_id="123",
                text=text,
                created_at=datetime.now().isoformat(),
                url="https://twitter.com/i/status/123",
                success=True,
            )

        with patch.object(connector, "post_tweet", side_effect=mock_post_tweet):
            long_thread = [f"Tweet {i}" for i in range(50)]
            result = await connector.post_thread(long_thread)

            assert result.success is True
            assert len(result.tweets) == MAX_THREAD_LENGTH


class TestTwitterRateLimiter:
    """Tests for TwitterRateLimiter."""

    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        from aragora.connectors.twitter_poster import TwitterRateLimiter

        limiter = TwitterRateLimiter(calls_per_window=10, window_seconds=60)
        assert limiter.calls_per_window == 10
        assert limiter.window_seconds == 60

    @pytest.mark.asyncio
    async def test_rate_limiter_allows_calls(self):
        """Test rate limiter allows calls under limit."""
        from aragora.connectors.twitter_poster import TwitterRateLimiter

        limiter = TwitterRateLimiter(calls_per_window=5, window_seconds=60)

        for _ in range(5):
            await limiter.acquire()  # Should not block

        assert len(limiter.call_times) == 5


class TestTwitterContentFormatter:
    """Tests for DebateContentFormatter."""

    def test_format_announcement(self):
        """Test announcement formatting."""
        from aragora.connectors.twitter_poster import DebateContentFormatter

        formatter = DebateContentFormatter()
        tweet = formatter.format_announcement(
            task="Should AI have rights?",
            agents=["Claude", "GPT-4", "Gemini"],
            debate_url="https://example.com/debate/123",
        )

        assert "Should AI have rights?" in tweet
        assert "Claude" in tweet
        assert len(tweet) <= 280

    def test_format_result(self):
        """Test result formatting."""
        from aragora.connectors.twitter_poster import DebateContentFormatter

        formatter = DebateContentFormatter()
        tweet = formatter.format_result(
            task="Test debate topic",
            agents=["Agent1", "Agent2"],
            consensus_reached=True,
            debate_url="https://example.com/result",
        )

        assert "Consensus reached" in tweet
        assert len(tweet) <= 280

    def test_format_thread(self):
        """Test thread formatting."""
        from aragora.connectors.twitter_poster import DebateContentFormatter

        formatter = DebateContentFormatter()
        tweets = formatter.format_thread(
            task="Test topic",
            agents=["A", "B"],
            highlights=["Point 1", "Point 2", "Point 3"],
            consensus_reached=False,
            debate_url="https://example.com",
        )

        assert len(tweets) >= 3  # Intro + highlights + result
        assert all(len(t) <= 280 for t in tweets)


# =============================================================================
# YouTube Connector Tests
# =============================================================================

class TestYouTubeConnectorConfiguration:
    """Tests for YouTubeUploaderConnector configuration."""

    def test_is_configured_with_all_credentials(self):
        """Test is_configured returns True with all credentials."""
        from aragora.connectors.youtube_uploader import YouTubeUploaderConnector

        connector = YouTubeUploaderConnector(
            client_id="client_id",
            client_secret="client_secret",
            refresh_token="refresh_token",
        )
        assert connector.is_configured is True

    def test_is_configured_missing_credentials(self):
        """Test is_configured returns False with missing credentials."""
        from aragora.connectors.youtube_uploader import YouTubeUploaderConnector

        connector = YouTubeUploaderConnector(client_id="id_only")
        assert connector.is_configured is False

    def test_loads_from_environment(self):
        """Test connector loads credentials from environment."""
        from aragora.connectors.youtube_uploader import YouTubeUploaderConnector

        with patch.dict(os.environ, {
            "YOUTUBE_CLIENT_ID": "env_client_id",
            "YOUTUBE_CLIENT_SECRET": "env_client_secret",
            "YOUTUBE_REFRESH_TOKEN": "env_refresh_token",
        }):
            connector = YouTubeUploaderConnector()
            assert connector.client_id == "env_client_id"
            assert connector.client_secret == "env_client_secret"


class TestYouTubeVideoMetadata:
    """Tests for YouTubeVideoMetadata."""

    def test_metadata_truncates_title(self):
        """Test title is truncated if too long."""
        from aragora.connectors.youtube_uploader import YouTubeVideoMetadata, MAX_TITLE_LENGTH

        long_title = "x" * 200
        metadata = YouTubeVideoMetadata(title=long_title, description="desc")

        assert len(metadata.title) <= MAX_TITLE_LENGTH
        assert metadata.title.endswith("...")

    def test_metadata_truncates_description(self):
        """Test description is truncated if too long."""
        from aragora.connectors.youtube_uploader import YouTubeVideoMetadata, MAX_DESCRIPTION_LENGTH

        long_desc = "x" * 6000
        metadata = YouTubeVideoMetadata(title="title", description=long_desc)

        assert len(metadata.description) <= MAX_DESCRIPTION_LENGTH

    def test_metadata_truncates_tags(self):
        """Test tags are truncated if total length exceeds limit."""
        from aragora.connectors.youtube_uploader import YouTubeVideoMetadata, MAX_TAGS_LENGTH

        long_tags = [f"tag{i}" * 20 for i in range(50)]  # Many long tags
        metadata = YouTubeVideoMetadata(title="title", description="desc", tags=long_tags)

        total_tag_length = sum(len(tag) for tag in metadata.tags)
        assert total_tag_length <= MAX_TAGS_LENGTH

    def test_metadata_to_api_body(self):
        """Test conversion to API request body."""
        from aragora.connectors.youtube_uploader import YouTubeVideoMetadata

        metadata = YouTubeVideoMetadata(
            title="Test Video",
            description="Test description",
            tags=["tag1", "tag2"],
            privacy_status="unlisted",
        )
        body = metadata.to_api_body()

        assert body["snippet"]["title"] == "Test Video"
        assert body["snippet"]["description"] == "Test description"
        assert body["snippet"]["tags"] == ["tag1", "tag2"]
        assert body["status"]["privacyStatus"] == "unlisted"


class TestYouTubeRateLimiter:
    """Tests for YouTubeRateLimiter."""

    def test_can_upload_within_quota(self):
        """Test upload allowed within daily quota."""
        from aragora.connectors.youtube_uploader import YouTubeRateLimiter

        limiter = YouTubeRateLimiter(daily_quota=10000)
        assert limiter.can_upload() is True
        assert limiter.remaining_quota == 10000

    def test_cannot_upload_over_quota(self):
        """Test upload rejected when over quota."""
        from aragora.connectors.youtube_uploader import YouTubeRateLimiter

        limiter = YouTubeRateLimiter(daily_quota=1600)
        limiter.record_upload()  # Uses 1600 units

        assert limiter.can_upload() is False
        assert limiter.remaining_quota == 0

    def test_quota_resets(self):
        """Test quota resets after reset time."""
        from aragora.connectors.youtube_uploader import YouTubeRateLimiter

        limiter = YouTubeRateLimiter(daily_quota=10000)
        limiter.record_upload()

        # Force reset by setting reset_time in past
        limiter.reset_time = time.time() - 1

        assert limiter.remaining_quota == 10000


class TestYouTubeConnectorOAuth:
    """Tests for YouTubeUploaderConnector OAuth."""

    @pytest.fixture
    def connector(self):
        from aragora.connectors.youtube_uploader import YouTubeUploaderConnector
        return YouTubeUploaderConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            refresh_token="test_refresh_token",
        )

    def test_get_auth_url(self, connector):
        """Test OAuth authorization URL generation."""
        url = connector.get_auth_url(
            redirect_uri="http://localhost:8080/callback",
            state="test_state",
        )

        assert "accounts.google.com" in url
        assert "client_id=test_client_id" in url
        assert "state=test_state" in url
        assert "youtube.upload" in url

    @pytest.mark.asyncio
    async def test_exchange_code_success(self, connector):
        """Test successful authorization code exchange."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new_access_token",
            "refresh_token": "new_refresh_token",
            "expires_in": 3600,
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            result = await connector.exchange_code(
                code="auth_code",
                redirect_uri="http://localhost:8080/callback",
            )

            assert result["access_token"] == "new_access_token"
            assert connector.refresh_token == "new_refresh_token"

    @pytest.mark.asyncio
    async def test_exchange_code_failure(self, connector):
        """Test failed authorization code exchange."""
        from aragora.connectors.youtube_uploader import YouTubeAuthError

        mock_response = MagicMock()
        mock_response.status_code = 400

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            with pytest.raises(YouTubeAuthError):
                await connector.exchange_code(
                    code="invalid_code",
                    redirect_uri="http://localhost:8080/callback",
                )

    @pytest.mark.asyncio
    async def test_refresh_access_token_success(self, connector):
        """Test successful token refresh."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "refreshed_token",
            "expires_in": 3600,
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            result = await connector._refresh_access_token()

            assert result is True
            assert connector._access_token == "refreshed_token"


class TestYouTubeConnectorUpload:
    """Tests for YouTubeUploaderConnector upload."""

    @pytest.fixture
    def connector(self):
        from aragora.connectors.youtube_uploader import YouTubeUploaderConnector
        connector = YouTubeUploaderConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            refresh_token="test_refresh_token",
        )
        # Pre-set access token
        connector._access_token = "test_access_token"
        connector._token_expiry = time.time() + 3600
        return connector

    @pytest.fixture
    def temp_video(self):
        """Create temporary video file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"fake video content")
            yield Path(f.name)
        os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_upload_not_configured(self):
        """Test upload fails when not configured."""
        from aragora.connectors.youtube_uploader import YouTubeUploaderConnector, YouTubeVideoMetadata

        connector = YouTubeUploaderConnector()  # No credentials
        metadata = YouTubeVideoMetadata(title="Test", description="Test")

        result = await connector.upload(Path("/fake/video.mp4"), metadata)

        assert result.success is False
        assert "not configured" in result.error

    @pytest.mark.asyncio
    async def test_upload_file_not_found(self, connector):
        """Test upload fails for non-existent file."""
        from aragora.connectors.youtube_uploader import YouTubeVideoMetadata

        metadata = YouTubeVideoMetadata(title="Test", description="Test")
        result = await connector.upload(Path("/nonexistent/video.mp4"), metadata)

        assert result.success is False
        assert "not found" in result.error

    @pytest.mark.asyncio
    async def test_upload_success(self, connector, temp_video):
        """Test successful video upload."""
        from aragora.connectors.youtube_uploader import YouTubeVideoMetadata

        # Mock init response
        init_response = MagicMock()
        init_response.status_code = 200
        init_response.headers = {"Location": "https://upload.googleapis.com/upload/123"}

        # Mock upload response
        upload_response = MagicMock()
        upload_response.status_code = 200
        upload_response.json.return_value = {
            "id": "video123",
            "status": {"uploadStatus": "complete"},
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post.return_value = init_response
            mock_client.put.return_value = upload_response
            MockClient.return_value.__aenter__.return_value = mock_client

            metadata = YouTubeVideoMetadata(title="Test Video", description="Test")
            result = await connector.upload(temp_video, metadata)

            assert result.success is True
            assert result.video_id == "video123"
            assert "youtube.com" in result.url

    @pytest.mark.asyncio
    async def test_upload_quota_exceeded(self, connector, temp_video):
        """Test upload fails when quota exceeded."""
        from aragora.connectors.youtube_uploader import YouTubeVideoMetadata

        # Exhaust quota
        for _ in range(7):  # 7 * 1600 > 10000
            connector.rate_limiter.record_upload()

        metadata = YouTubeVideoMetadata(title="Test", description="Test")
        result = await connector.upload(temp_video, metadata)

        assert result.success is False
        assert "quota exceeded" in result.error.lower()

    @pytest.mark.asyncio
    async def test_upload_circuit_breaker(self, connector, temp_video):
        """Test circuit breaker prevents uploads after failures."""
        from aragora.connectors.youtube_uploader import YouTubeVideoMetadata

        # Exhaust circuit breaker
        for _ in range(4):  # threshold is 3
            connector.circuit_breaker.record_failure()

        metadata = YouTubeVideoMetadata(title="Test", description="Test")
        result = await connector.upload(temp_video, metadata)

        assert result.success is False
        assert "Circuit breaker" in result.error


class TestYouTubeConnectorStatus:
    """Tests for YouTubeUploaderConnector video status."""

    @pytest.fixture
    def connector(self):
        from aragora.connectors.youtube_uploader import YouTubeUploaderConnector
        connector = YouTubeUploaderConnector(
            client_id="test_client_id",
            client_secret="test_client_secret",
            refresh_token="test_refresh_token",
        )
        connector._access_token = "test_access_token"
        connector._token_expiry = time.time() + 3600
        return connector

    @pytest.mark.asyncio
    async def test_get_video_status_success(self, connector):
        """Test getting video status."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "items": [{
                "id": "video123",
                "status": {"uploadStatus": "processed"},
                "processingDetails": {"processingStatus": "succeeded"},
            }]
        }

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            status = await connector.get_video_status("video123")

            assert status["id"] == "video123"
            assert status["status"]["uploadStatus"] == "processed"

    @pytest.mark.asyncio
    async def test_get_video_status_not_found(self, connector):
        """Test getting status for non-existent video."""
        from aragora.connectors.youtube_uploader import YouTubeAPIError

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"items": []}

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            with pytest.raises(YouTubeAPIError, match="not found"):
                await connector.get_video_status("nonexistent")


class TestYouTubeMetadataFromDebate:
    """Tests for create_video_metadata_from_debate helper."""

    def test_create_metadata_basic(self):
        """Test basic metadata creation."""
        from aragora.connectors.youtube_uploader import create_video_metadata_from_debate

        metadata = create_video_metadata_from_debate(
            task="Should AI have rights?",
            agents=["Claude", "GPT-4"],
            consensus_reached=True,
            debate_id="debate-123",
        )

        assert "AI Debate" in metadata.title
        assert "Claude" in metadata.description
        assert "GPT-4" in metadata.description
        assert "Consensus reached" in metadata.description
        assert "debate-123" in metadata.description
        assert "AI" in metadata.tags

    def test_create_metadata_truncates_long_task(self):
        """Test metadata truncates long task titles."""
        from aragora.connectors.youtube_uploader import create_video_metadata_from_debate, MAX_TITLE_LENGTH

        long_task = "x" * 200
        metadata = create_video_metadata_from_debate(
            task=long_task,
            agents=["Agent"],
            consensus_reached=False,
            debate_id="123",
        )

        assert len(metadata.title) <= MAX_TITLE_LENGTH


# =============================================================================
# Base Connector Tests
# =============================================================================

class TestBaseConnectorCache:
    """Tests for BaseConnector caching."""

    @pytest.fixture
    def connector(self):
        """Create a concrete connector for testing cache."""
        from aragora.connectors.github import GitHubConnector
        return GitHubConnector(repo="owner/repo")

    def test_cache_put_and_get(self, connector):
        """Test basic cache put and get."""
        evidence = Evidence(
            id="test-1",
            source_type=SourceType.EXTERNAL_API,
            source_id="test",
            content="Test content",
        )

        connector._cache_put("test-1", evidence)
        cached = connector._cache_get("test-1")

        assert cached is not None
        assert cached.id == "test-1"

    def test_cache_miss(self, connector):
        """Test cache miss returns None."""
        cached = connector._cache_get("nonexistent")
        assert cached is None

    def test_cache_ttl_expiry(self, connector):
        """Test cache entry expires after TTL."""
        evidence = Evidence(
            id="test-1",
            source_type=SourceType.EXTERNAL_API,
            source_id="test",
            content="Test content",
        )

        connector._cache_ttl = 0.1  # 100ms TTL
        connector._cache_put("test-1", evidence)

        # Wait for expiry
        time.sleep(0.2)

        cached = connector._cache_get("test-1")
        assert cached is None

    def test_cache_lru_eviction(self, connector):
        """Test LRU eviction when cache is full."""
        connector._max_cache_entries = 3

        for i in range(5):
            evidence = Evidence(
                id=f"test-{i}",
                source_type=SourceType.EXTERNAL_API,
                source_id=f"test-{i}",
                content=f"Content {i}",
            )
            connector._cache_put(f"test-{i}", evidence)

        # Oldest entries should be evicted
        assert connector._cache_get("test-0") is None
        assert connector._cache_get("test-1") is None
        # Newest should remain
        assert connector._cache_get("test-4") is not None

    def test_cache_stats(self, connector):
        """Test cache statistics."""
        evidence = Evidence(
            id="test-1",
            source_type=SourceType.EXTERNAL_API,
            source_id="test",
            content="Test content",
        )
        connector._cache_put("test-1", evidence)

        stats = connector._cache_stats()
        assert stats["total_entries"] == 1
        assert stats["max_entries"] == 500  # Default


class TestEvidenceDataclass:
    """Tests for Evidence dataclass."""

    def test_evidence_content_hash(self):
        """Test content hash generation."""
        evidence = Evidence(
            id="test-1",
            source_type=SourceType.EXTERNAL_API,
            source_id="test",
            content="Test content",
        )

        hash1 = evidence.content_hash
        assert len(hash1) == 16

        # Same content = same hash
        evidence2 = Evidence(
            id="test-2",
            source_type=SourceType.WEB_SEARCH,
            source_id="test2",
            content="Test content",
        )
        assert evidence2.content_hash == hash1

    def test_evidence_reliability_score(self):
        """Test reliability score calculation."""
        evidence = Evidence(
            id="test-1",
            source_type=SourceType.EXTERNAL_API,
            source_id="test",
            content="Test content",
            confidence=0.8,
            freshness=0.9,
            authority=0.7,
        )

        score = evidence.reliability_score
        expected = 0.4 * 0.8 + 0.3 * 0.9 + 0.3 * 0.7
        assert abs(score - expected) < 0.01

    def test_evidence_to_dict(self):
        """Test evidence serialization."""
        evidence = Evidence(
            id="test-1",
            source_type=SourceType.EXTERNAL_API,
            source_id="test",
            content="Test content",
            title="Test Title",
            metadata={"key": "value"},
        )

        d = evidence.to_dict()
        assert d["id"] == "test-1"
        assert d["source_type"] == "external_api"
        assert d["metadata"]["key"] == "value"

    def test_evidence_from_dict(self):
        """Test evidence deserialization."""
        d = {
            "id": "test-1",
            "source_type": "external_api",
            "source_id": "test",
            "content": "Test content",
            "title": "Test Title",
            "confidence": 0.8,
        }

        evidence = Evidence.from_dict(d)
        assert evidence.id == "test-1"
        assert evidence.source_type == SourceType.EXTERNAL_API
        assert evidence.confidence == 0.8


class TestConnectorFreshness:
    """Tests for BaseConnector freshness calculation."""

    @pytest.fixture
    def connector(self):
        from aragora.connectors.github import GitHubConnector
        return GitHubConnector(repo="owner/repo")

    def test_freshness_recent(self, connector):
        """Test freshness for recent content."""
        recent = datetime.now().isoformat()
        freshness = connector.calculate_freshness(recent)
        assert freshness == 1.0

    def test_freshness_old(self, connector):
        """Test freshness for old content."""
        old = "2020-01-01T00:00:00Z"
        freshness = connector.calculate_freshness(old)
        assert freshness == 0.3  # > 1 year old

    def test_freshness_invalid(self, connector):
        """Test freshness for invalid date."""
        freshness = connector.calculate_freshness("not a date")
        assert freshness == 0.5  # Default for unknown


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestConnectorErrors:
    """Tests for connector error classes."""

    def test_twitter_error_hierarchy(self):
        """Test Twitter error class hierarchy."""
        from aragora.connectors.twitter_poster import (
            TwitterError,
            TwitterAuthError,
            TwitterRateLimitError,
            TwitterAPIError,
        )
        from aragora.connectors.exceptions import ConnectorError

        assert issubclass(TwitterError, ConnectorError)
        assert issubclass(TwitterAuthError, TwitterError)
        assert issubclass(TwitterRateLimitError, TwitterError)
        assert issubclass(TwitterAPIError, TwitterError)

    def test_youtube_error_hierarchy(self):
        """Test YouTube error class hierarchy."""
        from aragora.connectors.youtube_uploader import (
            YouTubeError,
            YouTubeAuthError,
            YouTubeQuotaError,
            YouTubeUploadError,
            YouTubeAPIError,
        )
        from aragora.connectors.exceptions import ConnectorError

        assert issubclass(YouTubeError, ConnectorError)
        assert issubclass(YouTubeAuthError, YouTubeError)
        assert issubclass(YouTubeQuotaError, YouTubeError)
        assert issubclass(YouTubeUploadError, YouTubeError)
        assert issubclass(YouTubeAPIError, YouTubeError)

    def test_twitter_rate_limit_error_message(self):
        """Test TwitterRateLimitError includes retry info."""
        from aragora.connectors.twitter_poster import TwitterRateLimitError

        error = TwitterRateLimitError(retry_after=120)
        assert "120s" in str(error)

    def test_youtube_quota_error_message(self):
        """Test YouTubeQuotaError includes quota info."""
        from aragora.connectors.youtube_uploader import YouTubeQuotaError

        error = YouTubeQuotaError(remaining=500, reset_hours=12)
        assert "500" in str(error)
        assert "12h" in str(error)
