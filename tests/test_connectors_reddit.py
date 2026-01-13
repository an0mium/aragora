"""
Tests for Reddit Connector.

Tests the RedditConnector class for searching and fetching Reddit content.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import time

from aragora.connectors.reddit import RedditConnector, HTTPX_AVAILABLE
from aragora.connectors.base import Evidence
from aragora.reasoning.provenance import SourceType


class TestRedditConnectorBasics:
    """Test basic connector properties."""

    def test_connector_name(self):
        """Test connector name property."""
        connector = RedditConnector()
        assert connector.name == "Reddit"

    def test_source_type(self):
        """Test source type is EXTERNAL_API."""
        connector = RedditConnector()
        assert connector.source_type == SourceType.EXTERNAL_API

    def test_default_confidence(self):
        """Test default confidence is lower than HN due to less moderation."""
        connector = RedditConnector()
        assert connector.default_confidence == 0.6

    def test_is_available(self):
        """Test is_available matches httpx availability."""
        connector = RedditConnector()
        assert connector.is_available == HTTPX_AVAILABLE


class TestRedditConnectorSearch:
    """Test search functionality."""

    @pytest.fixture
    def mock_search_response(self):
        """Mock Reddit search API response."""
        return {
            "data": {
                "children": [
                    {
                        "kind": "t3",
                        "data": {
                            "id": "abc123",
                            "title": "Test Post Title",
                            "selftext": "This is the post content.",
                            "author": "testuser",
                            "subreddit": "programming",
                            "score": 500,
                            "num_comments": 50,
                            "upvote_ratio": 0.95,
                            "created_utc": 1704067200,  # 2024-01-01 00:00:00 UTC
                            "permalink": "/r/programming/comments/abc123/test_post_title/",
                            "url": "https://www.reddit.com/r/programming/comments/abc123/test_post_title/",
                            "is_self": True,
                            "over_18": False,
                            "spoiler": False,
                            "stickied": False,
                            "subreddit_subscribers": 5000000,
                        },
                    },
                    {
                        "kind": "t3",
                        "data": {
                            "id": "def456",
                            "title": "External Link Post",
                            "selftext": "",
                            "author": "linkposter",
                            "subreddit": "technology",
                            "score": 1000,
                            "num_comments": 200,
                            "upvote_ratio": 0.85,
                            "created_utc": 1704153600,  # 2024-01-02 00:00:00 UTC
                            "permalink": "/r/technology/comments/def456/external_link_post/",
                            "url": "https://example.com/article",
                            "is_self": False,
                            "over_18": False,
                            "subreddit_subscribers": 15000000,
                        },
                    },
                ]
            }
        }

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
    async def test_search_parses_results(self, mock_search_response):
        """Test that search correctly parses Reddit response."""
        connector = RedditConnector()

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_search_response
            mock_response.raise_for_status = MagicMock()

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            results = await connector.search("test query")

            assert len(results) == 2
            assert all(isinstance(r, Evidence) for r in results)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
    async def test_search_evidence_properties(self, mock_search_response):
        """Test evidence object has correct properties."""
        connector = RedditConnector()

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_search_response
            mock_response.raise_for_status = MagicMock()

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            results = await connector.search("test query")
            evidence = results[0]

            assert evidence.id == "reddit:abc123"
            assert evidence.title == "Test Post Title"
            assert "This is the post content" in evidence.content
            assert evidence.author == "testuser"
            assert evidence.source_type == SourceType.EXTERNAL_API
            assert evidence.url == "https://www.reddit.com/r/programming/comments/abc123/test_post_title/"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
    async def test_search_metadata(self, mock_search_response):
        """Test evidence metadata contains Reddit-specific fields."""
        connector = RedditConnector()

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_search_response
            mock_response.raise_for_status = MagicMock()

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            results = await connector.search("test query")
            metadata = results[0].metadata

            assert metadata["score"] == 500
            assert metadata["num_comments"] == 50
            assert metadata["upvote_ratio"] == 0.95
            assert metadata["subreddit"] == "programming"
            assert metadata["is_self"] is True

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
    async def test_search_external_link(self, mock_search_response):
        """Test external link posts include link URL."""
        connector = RedditConnector()

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_search_response
            mock_response.raise_for_status = MagicMock()

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            results = await connector.search("test query")
            link_post = results[1]

            assert "Link: https://example.com/article" in link_post.content
            assert link_post.metadata["is_self"] is False

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
    async def test_search_handles_timeout(self):
        """Test search gracefully handles timeout."""
        import httpx

        connector = RedditConnector()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.TimeoutException("Timeout")
            )

            results = await connector.search("test query")
            assert results == []

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
    async def test_search_handles_http_error(self):
        """Test search gracefully handles HTTP errors."""
        import httpx

        connector = RedditConnector()

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=httpx.HTTPStatusError("Rate limited", request=MagicMock(), response=mock_response)
            )

            results = await connector.search("test query")
            assert results == []


class TestRedditConnectorFetch:
    """Test fetch functionality."""

    @pytest.fixture
    def mock_post_response(self):
        """Mock Reddit post detail API response."""
        return [
            {
                "data": {
                    "children": [
                        {
                            "kind": "t3",
                            "data": {
                                "id": "abc123",
                                "title": "Detailed Post",
                                "selftext": "Detailed content here.",
                                "author": "detailuser",
                                "subreddit": "programming",
                                "score": 1000,
                                "num_comments": 100,
                                "upvote_ratio": 0.90,
                                "created_utc": 1704067200,
                                "permalink": "/r/programming/comments/abc123/detailed_post/",
                                "url": "https://www.reddit.com/r/programming/comments/abc123/detailed_post/",
                                "is_self": True,
                                "subreddit_subscribers": 5000000,
                            },
                        }
                    ]
                }
            },
            {
                "data": {
                    "children": [
                        {
                            "kind": "t1",
                            "data": {
                                "body": "Great post!",
                                "author": "commenter1",
                                "score": 50,
                            },
                        },
                        {
                            "kind": "t1",
                            "data": {
                                "body": "Thanks for sharing.",
                                "author": "commenter2",
                                "score": 25,
                            },
                        },
                    ]
                }
            },
        ]

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
    async def test_fetch_post(self, mock_post_response):
        """Test fetching a specific post."""
        connector = RedditConnector()

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_post_response
            mock_response.raise_for_status = MagicMock()

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            evidence = await connector.fetch("abc123")

            assert evidence is not None
            assert evidence.id == "reddit:abc123"
            assert evidence.title == "Detailed Post"
            assert "Detailed content here" in evidence.content

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
    async def test_fetch_includes_comments(self, mock_post_response):
        """Test fetch includes top comments in metadata."""
        connector = RedditConnector()

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_post_response
            mock_response.raise_for_status = MagicMock()

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            evidence = await connector.fetch("abc123")

            assert "top_comments" in evidence.metadata
            assert len(evidence.metadata["top_comments"]) == 2
            assert evidence.metadata["top_comments"][0]["body"] == "Great post!"

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
    async def test_fetch_caches_result(self, mock_post_response):
        """Test fetch caches result to avoid duplicate API calls."""
        connector = RedditConnector()

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_post_response
            mock_response.raise_for_status = MagicMock()

            mock_get = AsyncMock(return_value=mock_response)
            mock_client.return_value.__aenter__.return_value.get = mock_get

            # First fetch
            await connector.fetch("abc123")

            # Second fetch should use cache
            await connector.fetch("abc123")

            # API should only be called once
            assert mock_get.call_count == 1

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
    async def test_fetch_handles_prefix(self, mock_post_response):
        """Test fetch handles 'reddit:' prefix in ID."""
        connector = RedditConnector()

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.json.return_value = mock_post_response
            mock_response.raise_for_status = MagicMock()

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            evidence = await connector.fetch("reddit:abc123")

            assert evidence is not None
            assert evidence.id == "reddit:abc123"


class TestRedditConnectorConfidence:
    """Test confidence and authority calculations."""

    def test_confidence_increases_with_engagement(self):
        """Test that higher engagement leads to higher confidence."""
        connector = RedditConnector()

        # High engagement post
        high_engagement = {
            "id": "high",
            "title": "Popular",
            "selftext": "Content",
            "author": "user",
            "subreddit": "test",
            "score": 5000,
            "num_comments": 500,
            "upvote_ratio": 0.98,
            "created_utc": time.time(),
            "permalink": "/test",
            "url": "https://reddit.com/test",
            "is_self": True,
            "subreddit_subscribers": 100000,
        }

        # Low engagement post
        low_engagement = {
            "id": "low",
            "title": "Unpopular",
            "selftext": "Content",
            "author": "user",
            "subreddit": "test",
            "score": 5,
            "num_comments": 2,
            "upvote_ratio": 0.60,
            "created_utc": time.time(),
            "permalink": "/test2",
            "url": "https://reddit.com/test2",
            "is_self": True,
            "subreddit_subscribers": 100000,
        }

        high_evidence = connector._parse_post(high_engagement)
        low_evidence = connector._parse_post(low_engagement)

        assert high_evidence.confidence > low_evidence.confidence

    def test_authority_based_on_subreddit_size(self):
        """Test authority varies by subreddit size."""
        connector = RedditConnector()

        base_post = {
            "id": "test",
            "title": "Test",
            "selftext": "Content",
            "author": "user",
            "subreddit": "test",
            "score": 100,
            "num_comments": 10,
            "upvote_ratio": 0.90,
            "created_utc": time.time(),
            "permalink": "/test",
            "url": "https://reddit.com/test",
            "is_self": True,
        }

        # Large subreddit (>1M)
        large_sub = {**base_post, "subreddit_subscribers": 5_000_000}
        large_evidence = connector._parse_post(large_sub)

        # Medium subreddit (100K-1M)
        medium_sub = {**base_post, "subreddit_subscribers": 500_000}
        medium_evidence = connector._parse_post(medium_sub)

        # Small subreddit (<10K)
        small_sub = {**base_post, "subreddit_subscribers": 5_000}
        small_evidence = connector._parse_post(small_sub)

        # Medium subreddits should have highest authority (expert communities)
        assert medium_evidence.authority > large_evidence.authority
        assert medium_evidence.authority > small_evidence.authority

    def test_confidence_capped_below_hn(self):
        """Test confidence is capped below HackerNews (less expert moderation)."""
        connector = RedditConnector()

        # Maximum engagement post
        max_post = {
            "id": "max",
            "title": "Viral",
            "selftext": "Content",
            "author": "user",
            "subreddit": "all",
            "score": 100000,
            "num_comments": 10000,
            "upvote_ratio": 1.0,
            "created_utc": time.time(),
            "permalink": "/test",
            "url": "https://reddit.com/test",
            "is_self": True,
            "subreddit_subscribers": 50_000_000,
        }

        evidence = connector._parse_post(max_post)

        # Should be capped at 0.85 (below HN's 0.95)
        assert evidence.confidence <= 0.85


class TestRedditConnectorHelperMethods:
    """Test helper methods for getting subreddit content."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
    async def test_get_hot(self):
        """Test get_hot calls correct endpoint."""
        connector = RedditConnector()

        with patch.object(connector, "get_subreddit", new_callable=AsyncMock) as mock:
            mock.return_value = []

            await connector.get_hot("programming", limit=10)

            mock.assert_called_once_with("programming", sort="hot", limit=10)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
    async def test_get_top(self):
        """Test get_top calls correct endpoint with time filter."""
        connector = RedditConnector()

        with patch.object(connector, "get_subreddit", new_callable=AsyncMock) as mock:
            mock.return_value = []

            await connector.get_top("programming", limit=10, time_filter="week")

            mock.assert_called_once_with(
                "programming", sort="top", limit=10, time_filter="week"
            )

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
    async def test_get_new(self):
        """Test get_new calls correct endpoint."""
        connector = RedditConnector()

        with patch.object(connector, "get_subreddit", new_callable=AsyncMock) as mock:
            mock.return_value = []

            await connector.get_new("programming", limit=10)

            mock.assert_called_once_with("programming", sort="new", limit=10)


class TestRedditConnectorRateLimiting:
    """Test rate limiting behavior."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
    async def test_rate_limit_delay(self):
        """Test rate limiting enforces delay between requests."""
        connector = RedditConnector(rate_limit_delay=0.1)

        start_time = time.time()

        # Make first request to set last request time
        await connector._rate_limit()

        # Make second request immediately
        await connector._rate_limit()

        elapsed = time.time() - start_time

        # Should have waited at least the delay
        assert elapsed >= 0.1


class TestRedditConnectorEdgeCases:
    """Test edge cases and error handling."""

    def test_parse_post_missing_id(self):
        """Test parsing post without ID returns None."""
        connector = RedditConnector()

        post = {"title": "No ID", "selftext": "Content"}
        evidence = connector._parse_post(post)

        assert evidence is None

    def test_parse_post_deleted_author(self):
        """Test parsing post with deleted author."""
        connector = RedditConnector()

        post = {
            "id": "deleted",
            "title": "Deleted Author Post",
            "selftext": "Content",
            "author": "[deleted]",
            "subreddit": "test",
            "score": 10,
            "num_comments": 5,
            "upvote_ratio": 0.80,
            "created_utc": time.time(),
            "permalink": "/test",
            "url": "https://reddit.com/test",
            "is_self": True,
            "subreddit_subscribers": 10000,
        }

        evidence = connector._parse_post(post)

        assert evidence is not None
        assert evidence.author == "[deleted]"

    def test_parse_empty_listing(self):
        """Test parsing empty listing returns empty list."""
        connector = RedditConnector()

        data = {"data": {"children": []}}
        results = connector._parse_listing(data)

        assert results == []

    def test_parse_listing_skips_non_posts(self):
        """Test parsing listing skips non-post items."""
        connector = RedditConnector()

        data = {
            "data": {
                "children": [
                    {"kind": "t1", "data": {}},  # Comment, not post
                    {
                        "kind": "t3",
                        "data": {
                            "id": "post",
                            "title": "Post",
                            "selftext": "",
                            "author": "user",
                            "subreddit": "test",
                            "score": 1,
                            "num_comments": 0,
                            "upvote_ratio": 0.5,
                            "created_utc": time.time(),
                            "permalink": "/test",
                            "url": "https://reddit.com/test",
                            "is_self": True,
                            "subreddit_subscribers": 1000,
                        },
                    },
                ]
            }
        }

        results = connector._parse_listing(data)

        assert len(results) == 1


# Integration-style test (requires httpx)
@pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed")
class TestRedditConnectorIntegration:
    """Integration tests that verify the full flow."""

    @pytest.mark.asyncio
    async def test_search_to_evidence_flow(self):
        """Test complete search to evidence flow with mocks."""
        connector = RedditConnector()

        mock_response = {
            "data": {
                "children": [
                    {
                        "kind": "t3",
                        "data": {
                            "id": "integration",
                            "title": "Integration Test Post",
                            "selftext": "Testing the full flow.",
                            "author": "tester",
                            "subreddit": "test",
                            "score": 100,
                            "num_comments": 10,
                            "upvote_ratio": 0.90,
                            "created_utc": time.time(),
                            "permalink": "/r/test/comments/integration/",
                            "url": "https://www.reddit.com/r/test/comments/integration/",
                            "is_self": True,
                            "subreddit_subscribers": 50000,
                        },
                    }
                ]
            }
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_http_response = MagicMock()
            mock_http_response.json.return_value = mock_response
            mock_http_response.raise_for_status = MagicMock()

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_http_response
            )

            results = await connector.search("integration test")

            assert len(results) == 1
            evidence = results[0]

            # Verify evidence structure
            assert evidence.id.startswith("reddit:")
            assert evidence.source_type == SourceType.EXTERNAL_API
            assert evidence.title == "Integration Test Post"
            assert evidence.content is not None
            assert evidence.url is not None
            assert evidence.confidence > 0
            assert evidence.freshness > 0
            assert evidence.authority > 0
            assert "score" in evidence.metadata
            assert "subreddit" in evidence.metadata
